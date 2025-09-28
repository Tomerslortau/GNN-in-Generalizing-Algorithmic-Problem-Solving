import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class PolicyNet(nn.Module):
    """
    GCN over the board line graph.
    Scores candidate actions (i->j) using [h_i || h_j || |i-j|].
    """
    def __init__(self, hidden=128, n_layers=2):
        super().__init__()
        # Infer input feature size automatically
        self.gcn1 = GCNConv(-1, hidden)
        self.gcn2 = GCNConv(hidden, hidden) if n_layers >= 2 else None
        self.head = nn.Linear(2*hidden + 1, 1) 
        # nn.Sequential(
        #     nn.Linear(2 * hidden + 1, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1)
        # )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = F.relu(self.gcn1(x, edge_index))
        if self.gcn2 is not None:
            h = F.relu(self.gcn2(h, edge_index))

        src = h[data.actions_src]      # [A, H]
        dst = h[data.actions_dst]      # [A, H]
        dist = (data.actions_dst - data.actions_src).abs().unsqueeze(1).to(h.dtype)  # [A,1]
        feats = torch.cat([src, dst, dist], dim=1)  # [A, 2H+1]
        logits = self.head(feats).squeeze(-1)       # [A]
        return logits

    @staticmethod
    def masked_ce(logits: torch.Tensor, mask: torch.Tensor, y_global: int):
        if mask is None or not mask.any():
            return None
        masked_logits = logits[mask]
        legal_ids = torch.nonzero(mask, as_tuple=False).flatten()
        y_pos = (legal_ids == y_global).nonzero(as_tuple=False)
        if y_pos.numel() == 0:
            return None
        y = y_pos.item()
        return F.cross_entropy(masked_logits.unsqueeze(0),
                               torch.tensor([y], device=masked_logits.device))
