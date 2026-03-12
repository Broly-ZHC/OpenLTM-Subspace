import torch
from layers.SubspaceModule import VariableGrouping

B, T, V = 2, 96, 862
G, D_var = 32, 64

x = torch.randn(B, T, V)
vg = VariableGrouping(num_vars=V, d_var=D_var, num_groups=G, warmup_steps=100)

xg, gid, aux = vg(x, step=0)
print(xg.shape, gid.shape, aux.item())   # torch.Size([B, T, G]) torch.Size([V]) 0.0