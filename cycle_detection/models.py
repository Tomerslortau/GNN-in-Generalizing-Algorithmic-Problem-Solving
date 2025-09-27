# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GINConv, global_mean_pool, BatchNorm

class DirectedGINLayer(nn.Module):
    """Aggregates incoming and outgoing neighbors separately, then mixes with the node state."""
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.4):
        super().__init__()
        self.conv_in = GINConv(nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        ), train_eps=True)
        self.conv_out = GINConv(nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        ), train_eps=True)
        self.mix = nn.Linear(in_dim + 2 * hidden_dim, hidden_dim)
        self.bn = BatchNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h_in = self.conv_in(x, edge_index)                    # neighbors j -> i
        rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        h_out = self.conv_out(x, rev_edge_index)              # neighbors i -> j
        h = torch.cat([x, h_in, h_out], dim=-1)
        h = self.mix(h)
        h = self.bn(h)
        h = F.relu(h)
        return self.drop(h)

class DirectedGINClassifier(nn.Module):
    """Graph level classifier. Forward returns class logits."""
    def __init__(self, in_dim: int, hidden: int, depth: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        d_in = in_dim
        for _ in range(depth):
            layers.append(DirectedGINLayer(d_in, hidden, dropout))
            d_in = hidden
        self.gnn = nn.ModuleList(layers)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, data) -> Tensor:
        x, edge_index = data.x, data.edge_index
        # create a batch vector if not provided
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        for layer in self.gnn:
            x = layer(x, edge_index)
        g = global_mean_pool(x, batch)
        return self.head(g)  # logits

def build_model(in_dim: int, num_classes: int, hidden: int = 128, depth: int = 4, dropout: float = 0.1) -> nn.Module:
    return DirectedGINClassifier(in_dim=in_dim, hidden=hidden, depth=depth, num_classes=num_classes, dropout=dropout)

@torch.no_grad()
def predict_label(model: nn.Module, data) -> int:
    """Returns the predicted class index for a single graph Data object."""
    model.eval()
    device = next(model.parameters()).device
    logits = model(data.to(device))
    # probs = torch.softmax(logits, dim=1)[:, 1]  # P(cycle)
    # t = 0.30
    # return int((probs >= t).long().item())
    return int(logits.argmax(dim=1).item())
