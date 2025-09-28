import torch
from decoding import greedy_decode_with_scores

def auroc_from_scores(scores: torch.Tensor, labels: torch.Tensor) -> float:
    s = scores.detach().cpu()
    y = labels.detach().cpu().to(torch.int64)
    pos = (y == 1); neg = (y == 0)
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0: return float('nan')
    ranks = s.argsort().argsort().to(torch.float32) + 1.0
    rank_sum_pos = float(ranks[pos].sum().item())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

def auprc_from_scores(scores: torch.Tensor, labels: torch.Tensor) -> float:
    s = scores.detach().cpu()
    y = labels.detach().cpu().to(torch.int64)
    n_pos = int((y == 1).sum())
    if n_pos == 0: return float('nan')
    order = torch.argsort(s, descending=True)
    y_sorted = y[order]
    cpos = torch.cumsum((y_sorted == 1), dim=0)
    ranks = torch.arange(1, y_sorted.numel() + 1, dtype=torch.float32)
    prec_at_pos = (cpos[y_sorted == 1].to(torch.float32) / ranks[y_sorted == 1])
    return float(prec_at_pos.mean().item())

