# --*-- conding:utf-8 --*--
# @time:7/28/25 12:14
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:nn.py


import torch, torch.nn as nn
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.nn import Gate
from e3nn.math import soft_one_hot_linspace

class GNN_H(nn.Module):
    def __init__(self, n_radial=8, lmax=1, hidden=64):
        super().__init__()
        # 1) 生成原子特征
        irreps_node = o3.Irreps("0e")  # 标量
        irreps_out   = o3.Irreps.spherical_harmonics(lmax)
        self.embed_Z  = nn.Embedding(120, hidden)

        # 2) 消息传递 4 层
        self.layers = nn.ModuleList()
        for _ in range(4):
            self.layers.append(FullyConnectedTensorProduct(
                irreps_node, irreps_out, irreps_node))

        # 3) 读出
        self.pool = o3.Irreps("0e")  # 只要不变标量
        self.fc_h = nn.Linear(hidden, dim_h)  # dim_h = n(n+1)/2
        self.fc_g = nn.Linear(hidden, dim_g)  # dim_g = 独立二电子数

    def forward(self, Z, pos, ghost):
        x = self.embed_Z(Z)               # [N, hidden]
        for layer in self.layers:
            x = layer(x, pos)             # 等变更新
            x = Gate(x)                   # SiLU + 姿态
        g_feat = x.sum(0)                 # 全分子池化

        h = self.fc_h(g_feat)
        g = self.fc_g(g_feat)
        h = symmetrize_h(h)               # 强制 h_{pq}=h_{qp}
        g = symmetrize_g(g)               # 同理
        return torch.cat([h, g], dim=-1)
