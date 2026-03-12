import torch
import torch.nn as nn
import torch.nn.functional as F


class VariableGrouping(nn.Module):
    def __init__(
        self,
        num_vars: int,
        d_var: int,
        num_groups: int,
        seq_len: int,
        warmup_steps: int = 0,
        vq_beta: float = 0.25,
        softmax_temperature: float = 1.0,
        use_straight_through: bool = True,
        fixed_assignment: torch.Tensor | None = None,
    ):
        super().__init__()

        self.num_vars = int(num_vars)
        self.d_var = int(d_var)
        self.num_groups = int(num_groups)
        self.seq_len = int(seq_len)
        self.warmup_steps = int(warmup_steps)

        self.attn_proj = nn.Linear(self.seq_len, 1)  # [T] -> [1], compresses time dim to a scalar score per variable

        self.vq_beta = float(vq_beta)
        self.softmax_temperature = float(softmax_temperature)
        self.use_straight_through = bool(use_straight_through)

        self.var_embed = nn.Parameter(torch.randn(self.num_vars, self.d_var) * 0.02)  # [V, D_var]
        self.codebook = nn.Parameter(torch.randn(self.num_groups, self.d_var) * 0.02)  # [G, D_var]

        if fixed_assignment is None:
            fixed = torch.arange(self.num_vars) * self.num_groups // self.num_vars  # [V]
            fixed = fixed.clamp(min=0, max=self.num_groups - 1)  # [V] -> [V]
        else:
            fixed = fixed_assignment.to(torch.long)
            if fixed.numel() != self.num_vars:
                raise ValueError(f"fixed_assignment must have shape [V], got numel={fixed.numel()} vs V={self.num_vars}")
            fixed = fixed.clamp(min=0, max=self.num_groups - 1)  # [V] -> [V]

        self.register_buffer("fixed_group_id", fixed, persistent=False)  # [V]

    @torch.no_grad()
    def set_fixed_assignment(self, fixed_assignment: torch.Tensor):
        fixed = fixed_assignment.to(torch.long)
        if fixed.numel() != self.num_vars:
            raise ValueError(f"fixed_assignment must have shape [V], got numel={fixed.numel()} vs V={self.num_vars}")
        fixed = fixed.clamp(min=0, max=self.num_groups - 1)
        self.fixed_group_id = fixed  # [V]

    def _assignment_to_matrix(self, group_id: torch.Tensor) -> torch.Tensor:
        a = F.one_hot(group_id, num_classes=self.num_groups).to(self.var_embed.dtype)  # [V] -> [V, G]
        return a

    def get_group_id(self, step: int | None = None) -> torch.Tensor:
        if step is not None and step < self.warmup_steps:
            return self.fixed_group_id  # [V]

        var_embed = self.var_embed  # [V, D_var]
        codebook = self.codebook  # [G, D_var]

        var_norm2 = (var_embed ** 2).sum(dim=-1, keepdim=True)  # [V, D_var] -> [V, 1]
        code_norm2 = (codebook ** 2).sum(dim=-1).unsqueeze(0)  # [G, D_var] -> [G] -> [1, G]
        dots = torch.matmul(var_embed, codebook.transpose(0, 1))  # [V, D_var] x [D_var, G] -> [V, G]
        dist2 = var_norm2 + code_norm2 - 2.0 * dots  # [V, 1] + [1, G] - [V, G] -> [V, G]

        group_id = dist2.argmin(dim=-1)  # [V, G] -> [V]
        return group_id

    def vq_loss(self, group_id: torch.Tensor) -> torch.Tensor:
        var_embed = self.var_embed  # [V, D_var]
        code = self.codebook[group_id]  # [V] -> [V, D_var]

        codebook_loss = F.mse_loss(code, var_embed.detach())
        commit_loss = F.mse_loss(var_embed, code.detach())
        loss = codebook_loss + self.vq_beta * commit_loss
        return loss

    def forward(self, x: torch.Tensor, step: int | None = None):
        group_id = self.get_group_id(step=step)  # [V]
        a_hard = self._assignment_to_matrix(group_id)  # [V] -> [V, G]

        if step is not None and step < self.warmup_steps:
            a = a_hard  # [V, G]
            aux_loss = x.new_zeros(())
        else:
            logits = torch.matmul(self.var_embed, self.codebook.transpose(0, 1))  # [V, D_var] x [D_var, G] -> [V, G]
            a_soft = F.softmax(logits / self.softmax_temperature, dim=-1)  # [V, G]

            if self.use_straight_through:
                a = a_hard + (a_soft - a_soft.detach())  # [V, G]
            else:
                a = a_soft  # [V, G]

            aux_loss = self.vq_loss(group_id=group_id)

        # ---- Soft Attention Aggregation ----
        attn_scores = self.attn_proj(x.transpose(1, 2)).squeeze(-1)  # [B, T, V] -> [B, V, T] -> [B, V, 1] -> [B, V]
        a_mask = a_hard.unsqueeze(0)  # [V, G] -> [1, V, G]
        masked_scores = attn_scores.unsqueeze(-1) + (1.0 - a_mask) * -1e9  # [B, V, 1] + [1, V, G] -> [B, V, G]
        attn_weights = F.softmax(masked_scores, dim=1)  # [B, V, G] softmax over variable dim (V)
        attn_weights_ste = attn_weights * a.unsqueeze(0)  # [B, V, G] * [1, V, G] -> [B, V, G]; pipes STE grad to codebook
        x_group = torch.einsum('btv,bvg->btg', x, attn_weights_ste)  # [B, T, V] x [B, V, G] -> [B, T, G]
        # ------------------------------------

        return x_group, group_id, aux_loss, attn_weights_ste, a  # attn_weights_ste: [B, V, G]; a: [V, G] STE routing matrix
