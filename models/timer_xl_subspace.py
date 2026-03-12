import torch
import torch.nn.functional as F
from torch import nn
from layers.Transformer_EncDec import TimerBlock, TimerLayer
from layers.SelfAttention_Family import AttentionLayer, TimeAttention
from layers.SubspaceModule import VariableGrouping


class Model(nn.Module):
    """
    Timer-XL: Long-Context Transformers for Unified Time Series Forecasting 

    Paper: https://arxiv.org/abs/2410.04803
    
    GitHub: https://github.com/thuml/Timer-XL
    
    Citation: @article{liu2024timer,
        title={Timer-XL: Long-Context Transformers for Unified Time Series Forecasting},
        author={Liu, Yong and Qin, Guo and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
        journal={arXiv preprint arXiv:2410.04803},
        year={2024}
    }
    """
    def __init__(self, configs):
        super().__init__()
        self.input_token_len = configs.input_token_len
        self.embedding = nn.Linear(self.input_token_len, configs.d_model)
        self.output_attention = configs.output_attention

        # ---- Subspace Variable Grouping ----
        num_vars = configs.enc_in
        num_groups = getattr(configs, 'num_groups', 16)
        d_var = getattr(configs, 'd_var', 64)
        warmup_steps = getattr(configs, 'warmup_steps', 0)
        self.num_groups = num_groups
        self.variable_grouping = VariableGrouping(
            num_vars=num_vars,              # V = configs.enc_in
            d_var=d_var,                    # D_var
            num_groups=num_groups,          # G
            seq_len=configs.input_token_len,  # T, for attn_proj
            warmup_steps=warmup_steps,
        )
        self.register_buffer('_current_step', torch.zeros(1, dtype=torch.long), persistent=True)
        self._aux_loss = None
        # ------------------------------------

        self.blocks = TimerBlock(
            [
                TimerLayer(
                    AttentionLayer(
                        TimeAttention(True, attention_dropout=configs.dropout,
                                    output_attention=self.output_attention, 
                                    d_model=configs.d_model, num_heads=configs.n_heads,
                                    covariate=configs.covariate, flash_attention=configs.flash_attention),
                                    configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.head = nn.Linear(configs.d_model, configs.output_token_len)
        self.use_norm = configs.use_norm

        # ---- Global Shared Detail Reconstruction (Non-linear MLP) ----
        pred_len = (configs.seq_len // configs.input_token_len) * configs.output_token_len
        hidden_dim = getattr(configs, 'd_model', 512)

        self.detail_proj = nn.Sequential(
            nn.Linear(configs.seq_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(getattr(configs, 'dropout', 0.1)),
            nn.Linear(hidden_dim, pred_len)
        )
        nn.init.zeros_(self.detail_proj[-1].weight)  # zero-init: residual starts at 0
        nn.init.zeros_(self.detail_proj[-1].bias)

    def forecast(self, x, x_mark, y_mark):
        # x: [B, T, V]
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()  # [B, T, V] -> [B, 1, V]
            x = x - means  # [B, T, V]
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [B, 1, V]
            x /= stdev  # [B, T, V]

        B, T, V = x.shape
        G = self.num_groups
        x_raw = x  # [B, T, V] normalized input, saved for detail path

        # ---- Step 1: Subspace Routing ----
        x_group, group_id, aux_loss, attn_weights_ste, a_route = self.variable_grouping(x, step=self._current_step.item())
        # x_group: [B, T, G], group_id: [V], aux_loss: scalar, attn_weights_ste: [B, V, G], a_route: [V, G] STE
        self._aux_loss = aux_loss

        # ---- Step 2: CI Logic on G groups (replacing V with G) ----
        # [B, G, T]
        x_group = x_group.permute(0, 2, 1)  # [B, T, G] -> [B, G, T]
        # [B, G, N, P]
        x_group = x_group.unfold(
            dimension=-1, size=self.input_token_len, step=self.input_token_len)  # [B, G, T] -> [B, G, N, P]
        N = x_group.shape[2]
        # [B, G, N, D]
        embed_out = self.embedding(x_group)  # [B, G, N, P] -> [B, G, N, D]
        # [B, G * N, D]
        embed_out = embed_out.reshape(B, G * N, -1)  # [B, G, N, D] -> [B, G*N, D]
        embed_out, attns = self.blocks(embed_out, n_vars=G, n_tokens=N)  # [B, G*N, D]

        # ---- Step 3: Head in Group Space ----
        # [B, G, N, D]
        enc_out = embed_out.reshape(B, G, N, -1)  # [B, G*N, D] -> [B, G, N, D]
        # [B, G, N, P_out]
        y_group_patches = self.head(enc_out)  # [B, G, N, D] -> [B, G, N, P_out]
        # [B, pred_len, G]  where pred_len = N * P_out
        y_group = y_group_patches.reshape(B, G, -1).permute(0, 2, 1)  # [B, G, N*P_out] -> [B, N*P_out, G]

        # ---- Step 4: Dual-Path Reconstruction ----
        # Path 1: Group Trend Broadcast — magnitude-correct 1:1 mapping via a_route [V, G]
        y_base = torch.einsum('bpg,vg->bpv', y_group, a_route)  # [B, pred_len, G] x [V, G] -> [B, pred_len, V]

        # Path 2: Global Shared Detail Reconstruction (MLP)
        x_raw_v = x_raw.permute(0, 2, 1)  # [B, T, V] -> [B, V, T]
        y_detail_v = self.detail_proj(x_raw_v)  # [B, V, T] -> [B, V, pred_len]
        y_detail = y_detail_v.permute(0, 2, 1)  # [B, V, pred_len] -> [B, pred_len, V]

        dec_out = y_base + y_detail  # [B, pred_len, V]

        if self.use_norm:
            dec_out = dec_out * stdev + means  # [B, pred_len, V]
        if self.output_attention:
            return dec_out, attns
        return dec_out

    def forward(self, x, x_mark, y_mark):
        out = self.forecast(x, x_mark, y_mark)
        if self.training:
            self._current_step += 1  # persistent buffer, survives checkpoint save/load
        return out
