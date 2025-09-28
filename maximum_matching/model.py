import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv, GINConv, JumpingKnowledge

class HybridBlock(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, p_drop: float = 0.1, use_residual: bool = True):
        super().__init__()
        self.gat = GATv2Conv(in_dim, hid_dim, heads=1, add_self_loops=False)
        self.sage = SAGEConv(in_dim, hid_dim)
        self.gin = GINConv(nn.Sequential(
            nn.Linear(in_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim)
        ))
        self.proj = nn.Linear(3 * hid_dim, hid_dim)
        self.norm = nn.LayerNorm(hid_dim)
        self.drop = nn.Dropout(p_drop)
        self.use_residual = use_residual and (in_dim == hid_dim)

    def forward(self, x, edge_index):
        h = torch.cat([self.gat(x, edge_index), self.sage(x, edge_index), self.gin(x, edge_index)], dim=-1)
        h = self.proj(h)
        h = self.norm(F.relu(h))
        h = self.drop(h)
        if self.use_residual:
            h = h + x
        return h

class HybridGNN(nn.Module):
    def __init__(self, in_dim: int = 4, hid_dim: int = 64, n_layers: int = 3, jk_mode: str = 'lstm'):
        super().__init__()
        self.layers = nn.ModuleList([HybridBlock(in_dim if i == 0 else hid_dim, hid_dim) for i in range(n_layers)])
        if jk_mode == 'cat':
            self.jk = JumpingKnowledge('cat')
            out_dim = hid_dim * n_layers
        else:
            self.jk = JumpingKnowledge(jk_mode, channels=hid_dim, num_layers=n_layers)
            out_dim = hid_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(4 * out_dim, 2 * out_dim),
            nn.ReLU(),
            nn.Linear(2 * out_dim, 1)
        )

    def forward(self, x, edge_index):
        xs = []
        for gnn in self.layers:
            x = gnn(x, edge_index); xs.append(x)
        h = self.jk(xs)
        src, dst = edge_index
        h_src, h_dst = h[src], h[dst]
        e_feat = torch.cat([h_src, h_dst, torch.abs(h_src - h_dst), h_src * h_dst], dim=-1)
        logits = self.edge_mlp(e_feat).view(-1)
        return logits
    


class GNN(nn.Module):
    """A minimal fixed-architecture GNN for scoring edges."""
    def __init__(self):
        super().__init__()
        self.enc = nn.LazyLinear(64)
        self.conv1 = SAGEConv(64, 64)
        self.conv2 = SAGEConv(64, 64)
        self.drop = nn.Dropout(0.1)
        self.edge_mlp = nn.Sequential(
            nn.Linear(4 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Project features to 64-dim and run two message-passing layers
        x = F.relu(self.enc(x))
        x = F.relu(self.conv1(x, edge_index))
        x = self.drop(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.drop(x)

        # Edge features and score
        src, dst = edge_index
        h_src, h_dst = x[src], x[dst]
        e = torch.cat([h_src, h_dst, torch.abs(h_src - h_dst), h_src * h_dst], dim=-1)
        return self.edge_mlp(e).view(-1)