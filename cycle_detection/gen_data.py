# gen_data.py - Compact version with same interface
import random
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from typing import List, Tuple

# -------- Minimal cycle detection functions --------

def shortest_directed_cycle_length(G: nx.DiGraph) -> int:
    """Find shortest cycle length. None if acyclic."""
    try:
        cycle = nx.find_cycle(G, orientation='original')
        return len(cycle)
    except nx.NetworkXNoCycle:
        return None

def get_all_cycle_lengths(G: nx.DiGraph) -> List[int]:
    """Get all cycle lengths in graph."""
    try:
        cycles = list(nx.simple_cycles(G))
        return sorted(list(set(len(c) for c in cycles)))
    except:
        return []

def has_only_cycle_length(G: nx.DiGraph, L_target: int) -> bool:
    """Check if graph has only cycles of target length."""
    lengths = set(len(c) for c in nx.simple_cycles(G))
    if L_target == 0:
        return len(lengths) == 0
    return lengths == {L_target}

# -------- Simple graph generation --------

def make_graph_unique_cycle(L_target: int,
                            n_layers_rng: Tuple[int, int] = (10, 10),
                            layer_size_rng: Tuple[int, int] = (3, 7),
                            p_extra_forward: float = 0.05,
                            seed: int = None) -> nx.DiGraph:
    """
    Generate graph with exactly one cycle of target length.
    - Fixed depth 10, up to 1000 edges
    - L_target=0: acyclic tree
    - L_target>0: tree + one back edge
    """
    if seed: random.seed(seed)
    
    # Create 10 layers with enough nodes to reach ~1000 edges in tree
    # For a tree with N nodes, we have N-1 edges
    # So for ~1000 edges, we need ~1000 nodes
    target_nodes = 2000
    layers = []
    G = nx.DiGraph()

    # create 10 layers with ~1000 total nodes
    n_layers = 10  # Fixed depth
    nodes_per_layer = target_nodes // n_layers  # ~100 nodes per layer
    nid = 0
    for i in range(n_layers):
        # Vary layer size around the average
        min_size = max(1, nodes_per_layer - 20)
        max_size = nodes_per_layer + 20
        sz = random.randint(min_size, max_size)
        
        nodes = list(range(nid, nid + sz))
        nid += sz
        layers.append(nodes)
        G.add_nodes_from(nodes)

    # make it a tree: every node in layer i>0 picks exactly one parent in layer i-1
    parent_of = {}
    for i in range(1, n_layers):
        prev = layers[i - 1]
        cur = layers[i]
        for v in cur:
            u = random.choice(prev)
            G.add_edge(u, v)
            parent_of[v] = u

    # acyclic case
    if L_target == 0:
        return G

    # add exactly one back edge that creates a cycle of the requested length
    if not (2 <= L_target <= n_layers):
        raise ValueError("L_target must be 0 or between 2 and 10 for a depth-10 tree")

    # choose a node v deep enough so that we can go up L_target-1 ancestors
    candidate_layers = list(range(L_target - 1, n_layers))
    v_layer = random.choice(candidate_layers)
    v = random.choice(layers[v_layer])

    # walk up exactly L_target-1 parents to get u
    steps_up = L_target - 1
    u = v
    for _ in range(steps_up):
        u = parent_of[u]

    # add the single back edge v -> u
    G.add_edge(v, u)

    return G

def graph_to_data(G: nx.DiGraph, label: int) -> Data:
    """Convert NetworkX graph to PyTorch Geometric format."""
    nodes = list(G.nodes())
    node_map = {u: i for i, u in enumerate(nodes)}
    #     # Constant node features (1.0) + small random noise
    eps = np.random.normal(0, 1e-3, size=(len(nodes), 1)).astype(np.float32)
    x = torch.ones((len(nodes), 1), dtype=torch.float32) + torch.from_numpy(eps)
    #x = torch.ones((len(nodes), 1), dtype=torch.float32)
    # # Node features based on degree
    # degrees = [G.degree(node) for node in nodes]
    # x = torch.tensor([[float(deg)] for deg in degrees], dtype=torch.float32)
    # Node features based on topological rank (layer depth)
    # try:
    #     # Try topological sort to get rank/layer information
    #     topo_order = list(nx.topological_sort(G))
    #     node_ranks = {node: rank for rank, node in enumerate(topo_order)}
    # except (nx.NetworkXError, nx.NetworkXUnfeasible):
    #     # If graph has cycles, use node ID as rank (fallback)
    #     node_ranks = {node: node for node in nodes}
    
    # ranks = [node_ranks[node] for node in nodes]
    # # Normalize ranks to [0, 1] range
    # max_rank = max(ranks) if ranks else 1
    # normalized_ranks = [rank / max_rank for rank in ranks]
    # x = torch.tensor([[float(rank)] for rank in normalized_ranks], dtype=torch.float32)
    
    # Edge index
    edges = [(node_map[u], node_map[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty(2, 0, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))

# -------- Dataset building with same interface as original --------

def build_splits(k: int,
                 n_train=5000, n_val_per_length=200,
                 seed: int = 42,
                 n_layers_rng=(10, 10),  # Fixed depth 10
                 layer_size_rng=(3, 8),
                 p_forward=0.3,
                 p_er=0.04):
    """
    Build training and validation sets.
    Returns (train_data, val_sets) where val_sets is dict {cycle_length: data_list}
    """
    random.seed(seed)
    
    print("Generating training set...")
    train_data = []
    
    # Training: 30% acyclic, 70% cyclic (lengths 2 to k)
    target_acyclic = int(n_train * 0.5)  # 30% acyclic
    target_cyclic = n_train - target_acyclic  # 70% cyclic
    
    # Acyclic graphs
    for i in range(target_acyclic):
        G = make_graph_unique_cycle(0, n_layers_rng, layer_size_rng, p_forward)
        train_data.append(graph_to_data(G, 0))
        if (i + 1) % 100 == 0:
            print(f"    Generated {i + 1}/{target_acyclic} acyclic training graphs")
    
    # Cyclic graphs  
    cycle_lengths = list(range(2, k + 1))
    for i in range(target_cyclic):
        cycle_len = random.choice(cycle_lengths)
        G = make_graph_unique_cycle(cycle_len, n_layers_rng, layer_size_rng, p_forward)
        train_data.append(graph_to_data(G, 1))
        if (i + 1) % 100 == 0:
            print(f"    Generated {i + 1}/{target_cyclic} cyclic training graphs")
    
    print(f"  Final training distribution: {target_acyclic} acyclic, {target_cyclic} cyclic")
    
    # Validation sets
    print(f"\nGenerating validation sets for cycle lengths: {list(range(2, k + 7))}")
    val_sets = {}
    
    # Pure acyclic validation set
    print(f"Generating pure acyclic validation set...")
    acyclic_val = []
    for i in range(n_val_per_length):
        G = make_graph_unique_cycle(0, n_layers_rng, layer_size_rng, p_forward)
        acyclic_val.append(graph_to_data(G, 0))
        if (i + 1) % 50 == 0:
            print(f"    Generated {i + 1}/{n_val_per_length} acyclic validation graphs")
    val_sets[0] = acyclic_val
    
    # Mixed validation sets for each cycle length
    for cycle_len in range(2, k + 7):
        print(f"  Generating validation set for cycle length {cycle_len}...")
        val_data = []
        
        if cycle_len > k:
            # Pure cyclic for out-of-distribution
            print(f"    Using pure cyclic composition (cycle length {cycle_len} > k={k})")
            for i in range(n_val_per_length):
                G = make_graph_unique_cycle(cycle_len, n_layers_rng, layer_size_rng, p_forward)
                val_data.append(graph_to_data(G, 1))
                if (i + 1) % 25 == 0:
                    print(f"      Generated {i + 1}/{n_val_per_length} graphs with cycle length {cycle_len}")
            print(f"  Completed validation set for cycle length {cycle_len}: {len(val_data)} graphs (100% cyclic)")
        else:
            # Mixed: 30% acyclic + 70% target cycle
            print(f"    Using mixed composition (50% acyclic + 70% cyclic)")
            acyclic_target = int(n_val_per_length * 0.5)
            cyclic_target = n_val_per_length - acyclic_target
            
            # Acyclic part
            for i in range(acyclic_target):
                G = make_graph_unique_cycle(0, n_layers_rng, layer_size_rng, p_forward)
                val_data.append(graph_to_data(G, 0))
            
            # Cyclic part
            for i in range(cyclic_target):
                G = make_graph_unique_cycle(cycle_len, n_layers_rng, layer_size_rng, p_forward)
                val_data.append(graph_to_data(G, 1))
                if (i + 1) % 25 == 0:
                    print(f"      Generated {i + 1}/{cyclic_target} graphs with cycle length {cycle_len}")
            
            print(f"  Completed validation set for cycle length {cycle_len}: {len(val_data)} graphs ({acyclic_target} acyclic, {cyclic_target} cyclic)")
        
        val_sets[cycle_len] = val_data
    
    return train_data, val_sets

# -------- Optional CLI --------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--val", type=int, default=200)
    args = parser.parse_args()
    
    train, val = build_splits(args.k, args.train, args.val)
    torch.save(train, f"train_k{args.k}.pt")
    torch.save(val, f"val_k{args.k}.pt")
    print(f"Saved train_k{args.k}.pt and val_k{args.k}.pt")