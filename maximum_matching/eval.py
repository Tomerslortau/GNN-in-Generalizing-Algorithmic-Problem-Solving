import torch
from generators import generate_dataset
from metrics import auprc_from_scores, auroc_from_scores
from decoding import greedy_decode_with_scores
from torch_geometric.loader import DataLoader
from typing import Dict

@torch.no_grad()
def evaluate_with_baselines(model, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_edges = total_correct = 0
    num_pred = num_rand = num_deg = num_opt = 0
    all_scores, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        probs = torch.sigmoid(logits)
        all_scores.append(probs.detach().cpu())
        all_labels.append(batch.edge_y.detach().cpu())

        if batch.edge_y.numel() > 0:
            preds = (probs >= 0.5).float()
            total_correct += (preds == batch.edge_y).sum().item()
            total_edges += batch.edge_y.numel()

        src = batch.edge_index[0]
        edge_gid = batch.batch[src]  # map each edge to its graph by source node's batch id

        for gid in edge_gid.unique().tolist():
            mask = (edge_gid == gid)
            if not mask.any(): continue
            sub_edges = batch.edge_index[:, mask]
            sub_scores = probs[mask]

            # model ordering
            num_pred += greedy_decode_with_scores(sub_edges, sub_scores, num_nodes=batch.x.size(0))

            # random ordering
            rand_scores = torch.rand_like(sub_scores)
            num_rand += greedy_decode_with_scores(sub_edges, rand_scores, num_nodes=batch.x.size(0))

            # degree heuristic (within subgraph)
            u, v = sub_edges
            deg_sub = torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
            deg_sub.scatter_add_(0, u, torch.ones_like(u, dtype=torch.long))
            deg_sub.scatter_add_(0, v, torch.ones_like(v, dtype=torch.long))
            deg_scores = -(deg_sub[u] + deg_sub[v]).to(torch.float32)
            num_deg += greedy_decode_with_scores(sub_edges, deg_scores, num_nodes=batch.x.size(0))

        num_opt += int(batch.opt_size.sum().item())

    edge_acc = (total_correct / total_edges) if total_edges > 0 else float('nan')
    approx_ratio = (num_pred / max(num_opt, 1))
    approx_rand  = (num_rand / max(num_opt, 1))
    approx_deg   = (num_deg  / max(num_opt, 1))

    scores_cat = torch.cat(all_scores) if len(all_scores) else torch.tensor([])
    labels_cat = torch.cat(all_labels) if len(all_labels) else torch.tensor([])
    auroc = auroc_from_scores(scores_cat, labels_cat) if scores_cat.numel() else float('nan')
    auprc = auprc_from_scores(scores_cat, labels_cat) if scores_cat.numel() else float('nan')

    return dict(edge_acc=edge_acc, approx_ratio=approx_ratio,
                approx_rand=approx_rand, approx_deg=approx_deg,
                auroc=auroc, auprc=auprc)

def print_eval_results(results: Dict[str, float]):
    print(f"Edge Acc: {results['edge_acc']:.3f} | "
          f"Approx: {results['approx_ratio']:.3f} (Rand: {results['approx_rand']:.3f}, Deg: {results['approx_deg']:.3f}) | "
          f"AUROC: {results['auroc']:.3f} | AUPRC: {results['auprc']:.3f}")