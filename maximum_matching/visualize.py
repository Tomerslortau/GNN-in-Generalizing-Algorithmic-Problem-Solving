from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import torch

from .generators import maximum_matching_edges_bipartite, maximum_matching_edges_general

def build_positions_bipartite(U: List[int], V: List[int]):
    pos = {}
    pos.update((u, (0, i)) for i, u in enumerate(U))
    pos.update((v, (1, i)) for i, v in enumerate(V))
    return pos

def build_positions_general(G: nx.Graph):
    return nx.spring_layout(G, seed=np.random.randint(1e9))

def visualize_graph_process(G: nx.Graph, pos: Dict[int, Tuple[float, float]],
                            optimal_edges: set, greedy_order: List[Tuple[int,int]], out_path: str):
    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(G, pos, node_size=80)
    nx.draw_networkx_labels(G, pos, font_size=6)
    nx.draw_networkx_edges(G, pos, width=0.5, style='solid')
    if optimal_edges:
        nx.draw_networkx_edges(G, pos, edgelist=list(optimal_edges), width=2.5, style='solid')
    if greedy_order:
        nx.draw_networkx_edges(G, pos, edgelist=greedy_order, width=2.5, style='dashed')
        edge_labels = {e: i+1 for i, e in enumerate(greedy_order)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.axis('off'); plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

@torch.no_grad()
def visualize_from_data_bipartite(model, data, out_path: str):
    n = data.x.size(0)
    U = data.U_idx.tolist() if hasattr(data, "U_idx") else [i for i in range(n) if data.x[i,0] > 0.5]
    V = data.V_idx.tolist() if hasattr(data, "V_idx") else [i for i in range(n) if data.x[i,1] > 0.5]
    G = nx.Graph(); G.add_nodes_from(U, bipartite=0); G.add_nodes_from(V, bipartite=1)
    seen = set()
    for u, v in data.edge_index.t().tolist():
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen: continue
        seen.add((a, b)); G.add_edge(a, b)
    optimal = maximum_matching_edges_bipartite(G, U)
    logits = model(data.x, data.edge_index); probs = torch.sigmoid(logits).cpu().numpy()
    scores = {}
    for idx, (u, v) in enumerate(data.edge_index.t().tolist()):
        a, b = (u, v) if u < v else (v, u)
        scores[(a, b)] = max(scores.get((a, b), 0.0), float(probs[idx]))
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    used = set(); greedy = []
    for (a, b), _ in ordered:
        if a in used or b in used: continue
        greedy.append((a, b)); used.add(a); used.add(b)
    pos = build_positions_bipartite(U, V)
    visualize_graph_process(G, pos, optimal, greedy, out_path)