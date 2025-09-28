import torch

@torch.no_grad()
def greedy_decode_with_scores(edge_index: torch.Tensor, scores: torch.Tensor, num_nodes: int):
    """Return count of edges chosen by greedy under 'scores' ensuring a legal matching."""
    order = torch.argsort(scores, descending=True)
    used = torch.zeros(num_nodes, dtype=torch.bool, device=scores.device)
    chosen = 0
    for idx in order.tolist():
        u, v = edge_index[:, idx].tolist()
        if not used[u] and not used[v]:
            used[u] = used[v] = True
            chosen += 1
    return chosen

def check_matching_legal(edge_index: torch.Tensor, chosen_mask: torch.Tensor) -> bool:
    """Optional: verify that no node appears twice among chosen edges (undirected)."""
    used = set()
    for u, v in edge_index.t()[chosen_mask].tolist():
        a, b = (u, v) if u < v else (v, u)
        if a in used or b in used:
            return False
        used.add(a); used.add(b)
    return True