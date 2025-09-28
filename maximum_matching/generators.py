from typing import Dict, List, Tuple
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from networkx.algorithms import bipartite, matching

from config import GraphConfig

# ---------- Graph generators (bipartite) ----------

def gen_er_bipartite(cfg: GraphConfig):
    n_u = np.random.randint(cfg.nU_min, cfg.nU_max + 1)
    n_v = np.random.randint(cfg.nV_min, cfg.nV_max + 1)
    U = np.arange(n_u, dtype=np.int64)
    V = np.arange(n_v, dtype=np.int64) + n_u
    p = np.random.uniform(cfg.p_min, cfg.p_max) if cfg.p_min < cfg.p_max else cfg.p
    mask = (np.random.rand(n_u, n_v) < p)
    u_idx, v_idx = np.where(mask)
    G = nx.Graph()
    G.add_nodes_from(U.tolist(), bipartite=0)
    G.add_nodes_from(V.tolist(), bipartite=1)
    edges = [(int(u), int(n_u + v)) for u, v in zip(u_idx, v_idx)]
    G.add_edges_from(edges)
    return G, U.tolist(), V.tolist()

def gen_sbm_bipartite(cfg: GraphConfig):
    n_u = np.random.randint(cfg.nU_min, cfg.nU_max + 1)
    n_v = np.random.randint(cfg.nV_min, cfg.nV_max + 1)
    Bu, Bv = cfg.sbm_blocks_u, cfg.sbm_blocks_v
    sizes_u = [n_u // Bu + (1 if i < n_u % Bu else 0) for i in range(Bu)]
    sizes_v = [n_v // Bv + (1 if i < n_v % Bv else 0) for i in range(Bv)]
    U_blocks, start = [], 0
    for sz in sizes_u:
        U_blocks.append(list(range(start, start + sz))); start += sz
    V_blocks = []
    start = n_u
    for sz in sizes_v:
        V_blocks.append(list(range(start, start + sz))); start += sz
    G = nx.Graph()
    for u in range(n_u): G.add_node(u, bipartite=0)
    for v in range(n_u, n_u + n_v): G.add_node(v, bipartite=1)
    for i, Ub in enumerate(U_blocks):
        for j, Vb in enumerate(V_blocks):
            p = cfg.sbm_p_in if i == j else cfg.sbm_p_out
            mask = (np.random.rand(len(Ub), len(Vb)) < p)
            u_idx, v_idx = np.where(mask)
            G.add_edges_from((Ub[int(u)], Vb[int(v)]) for u, v in zip(u_idx, v_idx))
    U = [u for b in U_blocks for u in b]
    V = [v for b in V_blocks for v in b]
    return G, U, V

def _sample_powerlaw(n, exp, min_deg):
    return np.random.zipf(exp, size=n) + (min_deg - 1)

def gen_powerlaw_bipartite(cfg: GraphConfig):
    n_u = np.random.randint(cfg.nU_min, cfg.nU_max + 1)
    n_v = np.random.randint(cfg.nV_min, cfg.nV_max + 1)
    du = _sample_powerlaw(n_u, cfg.plaw_exp_u, cfg.plaw_min_deg)
    dv = _sample_powerlaw(n_v, cfg.plaw_exp_v, cfg.plaw_min_deg)
    Su, Sv = int(du.sum()), int(dv.sum())
    if Su != Sv:
        if Su > Sv: du[-1] = max(1, du[-1] - (Su - Sv))
        else:       dv[-1] = max(1, dv[-1] - (Sv - Su))
    U = list(range(n_u))
    V = [n_u + i for i in range(n_v)]
    left_stubs  = np.repeat(U, du)
    right_stubs = np.repeat(V, dv)
    np.random.shuffle(left_stubs); np.random.shuffle(right_stubs)
    m = min(len(left_stubs), len(right_stubs))
    edges = [(int(left_stubs[i]), int(right_stubs[i])) for i in range(m)]
    G = nx.Graph(); G.add_nodes_from(U, bipartite=0); G.add_nodes_from(V, bipartite=1); G.add_edges_from(edges)
    return G, U, V

# ---------- Graph generators (general) ----------

def gen_er_general(cfg: GraphConfig):
    n = np.random.randint(cfg.n_min, cfg.n_max + 1)
    return nx.erdos_renyi_graph(n, cfg.p)

def gen_sbm_general(cfg: GraphConfig):
    B = cfg.sbm_blocks
    n = np.random.randint(cfg.n_min, cfg.n_max + 1)
    sizes = [n // B + (1 if i < n % B else 0) for i in range(B)]
    P = np.full((B, B), cfg.sbm_p_out, dtype=float); np.fill_diagonal(P, cfg.sbm_p_in)
    G = nx.stochastic_block_model(sizes, P, seed=np.random.randint(1e9))
    return nx.Graph(G)

def gen_powerlaw_general(cfg: GraphConfig):
    n = np.random.randint(cfg.n_min, cfg.n_max + 1)
    degs = _sample_powerlaw(n, cfg.plaw_exp, cfg.plaw_min_deg)
    if degs.sum() % 2 == 1: degs[-1] += 1
    G = nx.configuration_model(degs); G = nx.Graph(G); G.remove_edges_from(nx.selfloop_edges(G))
    return G

# ---------- Ground truth ----------

def maximum_matching_edges_bipartite(G: nx.Graph, U: List[int]):
    M = bipartite.maximum_matching(G, top_nodes=set(U))
    matched = set()
    for u in U:
        v = M.get(u, None)
        if v is None: continue
        a, b = (u, v) if u < v else (v, u)
        matched.add((a, b))
    return matched

def maximum_matching_edges_general(G: nx.Graph):
    M = matching.max_weight_matching(G, maxcardinality=True)
    return {(min(u, v), max(u, v)) for (u, v) in M}

# ---------- Build PyG Data ----------

def build_pyg_data_bipartite(G: nx.Graph, U: List[int], V: List[int]) -> Data:
    n = G.number_of_nodes()
    part0 = torch.zeros(n, 1); part1 = torch.zeros(n, 1)
    part0[U] = 1.0; part1[V] = 1.0
    deg = torch.tensor([G.degree[i] for i in range(n)], dtype=torch.float32).view(-1, 1)
    x = torch.cat([part0, part1, torch.log1p(deg), torch.randn_like(deg)], dim=1)

    matched = maximum_matching_edges_bipartite(G, U)
    opt_size = len(matched)

    edges, labels = [], []
    for u, v in G.edges():
        a, b = (u, v) if u < v else (v, u)
        edges.append([u, v]); labels.append(1 if (a, b) in matched else 0)

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_y = torch.empty((0,), dtype=torch.float32)
    else:
        e = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = torch.cat([e, e.flip(0)], dim=1)
        edge_y = torch.tensor(labels, dtype=torch.float32).repeat(2)

    data = Data(x=x, edge_index=edge_index, edge_y=edge_y)
    data.opt_size = torch.tensor([opt_size], dtype=torch.long)
    data.num_nodes = n
    data.U_idx = torch.tensor(U, dtype=torch.long)
    data.V_idx = torch.tensor(V, dtype=torch.long)
    return data

def build_pyg_data_general(G: nx.Graph) -> Data:
    n = G.number_of_nodes()
    deg = torch.tensor([G.degree[i] for i in range(n)], dtype=torch.float32).view(-1, 1)
    dummy_part0 = torch.ones(n, 1)  # All 1s
    dummy_part1 = torch.zeros(n, 1) # All 0s
    x = torch.cat([dummy_part0, dummy_part1, torch.log1p(deg), torch.randn_like(deg)], dim=1)

    matched = maximum_matching_edges_general(G)
    opt_size = len(matched)

    edges, labels = [], []
    for u, v in G.edges():
        a, b = (u, v) if u < v else (v, u)
        edges.append([u, v]); labels.append(1 if (a, b) in matched else 0)

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_y = torch.empty((0,), dtype=torch.float32)
    else:
        e = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = torch.cat([e, e.flip(0)], dim=1)
        edge_y = torch.tensor(labels, dtype=torch.float32).repeat(2)

    data = Data(x=x, edge_index=edge_index, edge_y=edge_y)
    data.opt_size = torch.tensor([opt_size], dtype=torch.long)
    data.num_nodes = n
    return data

# ---------- Sparse bipartite ER (for large n) ----------

def gen_sparse_er_bipartite(n: int, p: float, seed: int = None):
    rng = np.random.default_rng(seed)
    U = list(range(n)); V = list(range(n, 2*n))
    G = nx.Graph(); G.add_nodes_from(U, bipartite=0); G.add_nodes_from(V, bipartite=1)
    for u in U:
        d = rng.binomial(n, p)
        if d == 0: continue
        if d >= n: nbr = list(range(n))
        else:      nbr = rng.choice(n, size=d, replace=False).tolist()
        G.add_edges_from((u, n + v) for v in nbr)
    return G, U, V

def make_sparse_dataset(n: int, p: float, num_graphs: int, seed: int = 123):
    from tqdm import tqdm
    ds = []
    for i in tqdm(range(num_graphs), desc=f"sparse ER_bip n={n}, p={p:.6f}", leave=False):
        G, U, V = gen_sparse_er_bipartite(n, p, seed=seed + i)
        ds.append(build_pyg_data_bipartite(G, U, V))
    return ds

def generate_dataset(cfg: GraphConfig, num_graphs: int = 1):
    """
    Generates a dataset of graphs based on the configuration.
    
    Args:
        cfg: GraphConfig object containing graph type and parameters
        num_graphs: number of graphs to generate
    
    Returns:
        List of PyG Data objects
    """
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    ds = []
    
    for i in range(num_graphs):
        if cfg.graph_type == 'er_bipartite':
            G, U, V = gen_er_bipartite(cfg)
            data = build_pyg_data_bipartite(G, U, V)
        elif cfg.graph_type == 'sbm_bipartite':
            G, U, V = gen_sbm_bipartite(cfg)
            data = build_pyg_data_bipartite(G, U, V)
        elif cfg.graph_type == 'powerlaw_bipartite':
            G, U, V = gen_powerlaw_bipartite(cfg)
            data = build_pyg_data_bipartite(G, U, V)
        elif cfg.graph_type == 'er_general':
            G = gen_er_general(cfg)
            data = build_pyg_data_general(G)
        elif cfg.graph_type == 'sbm_general':
            G = gen_sbm_general(cfg)
            data = build_pyg_data_general(G)
        elif cfg.graph_type == 'powerlaw_general':
            G = gen_powerlaw_general(cfg)
            data = build_pyg_data_general(G)
        elif cfg.graph_type == 'sparse_er_bipartite':
            G, U, V = gen_sparse_er_bipartite(cfg.nU_max, cfg.p, seed=cfg.seed+i)
            data = build_pyg_data_bipartite(G, U, V)
        else:
            raise ValueError(f"Unknown graph_type: {cfg.graph_type}")
        ds.append(data)
        
    return ds


