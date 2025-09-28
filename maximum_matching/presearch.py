import torch
from typing import List
from torch_geometric.data import DataLoader
import numpy as np
from eval import evaluate_with_baselines
from generators import generate_dataset
from config import GraphConfig


# --------------------------------------------------------------------------------------
# Sweep over family-specific density knob (coeff)
# --------------------------------------------------------------------------------------

class DummyModel(torch.nn.Module):
    def forward(self, x, edge_index):
        return torch.zeros(edge_index.size(1), dtype=torch.float32, device=x.device)

@torch.no_grad()
def sweep_coeff_for_n(family: str, n: int, sweep_param: str, sweep_array: List[float], device: torch.device):
    best_param, best_score = None, float('inf')
    rows = []  # (param, approx_rand, approx_deg, avg_opt_cov)

    for c in sweep_array:
        config = GraphConfig()
        config[sweep_param] = c
        ds = generate_dataset(config, 100)
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        metrics = evaluate_with_baselines(DummyModel().to(device), loader, device)

        # Coverage metric: bipartite uses min(side sizes); general uses floor(n/2)
        total_opt = int(sum(int(d.opt_size.item()) for d in ds))
        if 'bipartite' in family:
            mins = [min(int((d.x[:,0] > 0.5).sum()), int((d.x[:,1] > 0.5).sum())) for d in ds]
            denom = float(np.mean(mins)) if mins else 1.0
        else:
            ns = [int(d.num_nodes) for d in ds]
            denom = float(np.mean([max(1, t // 2) for t in ns])) if ns else 1.0
        avg_opt_cov = (total_opt / max(len(ds), 1)) / max(denom, 1.0)

        rows.append((c, metrics['approx_rand'], metrics['approx_deg'], avg_opt_cov))

        if avg_opt_cov >= 0.2 and metrics['approx_rand'] < best_score:
            best_score, best_param = metrics['approx_rand'], c

        disp_p = c / n if family.startswith('er_') else c
        print(f"[sweep-n={n}|{family}] param={disp_p:.6f} | rand={metrics['approx_rand']:.3f} | "
              f"deg={metrics['approx_deg']:.3f} | cov={avg_opt_cov:.3f}")

    return best_param, rows


def select_worst_param(rows):
    """
    Given rows: [(coeff, approx_rand, approx_deg, avg_opt_cov)], select the
    coefficient minimizing (rand + deg). Returns (coeff_or_None, chosen_score)
    """
    worst_coeff, worst_score = None, float('inf')
    for coeff, approx_rand, approx_deg, avg_opt_cov in rows:
        score = (approx_rand + approx_deg)
        if score < worst_score:
            worst_score, worst_coeff = score, coeff
    return worst_coeff, worst_score


def presearch_param(sweep_param, sweep_range, graph_type, device, n=100):
    _best_unused, rows = sweep_coeff_for_n(
        graph_type, n, sweep_param, sweep_range, device
    )
    worst_coeff, worst_score = select_worst_param(rows)
    if worst_coeff is None:
        worst_coeff = (sweep_range[0] + sweep_range[1]) / 2.0
        print(f"[presearch] No coeff met coverage floor; defaulting to {worst_coeff:.6f}")
    disp_p = worst_coeff / n if graph_type.startswith('er_') else worst_coeff
    print(f"[presearch] worst coeff={disp_p:.6f} at n={n}")
    return worst_coeff