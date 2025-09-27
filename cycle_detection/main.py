import argparse
import random
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.loader import DataLoader

from gen_data import build_splits, make_graph_unique_cycle, shortest_directed_cycle_length, has_only_cycle_length, graph_to_data
from models import DirectedGINClassifier
from training import train
from evaluation import evaluate

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def find_cycle_edges(G):
    """Find edges that form a cycle in graph G."""
    try:
        cycle = nx.find_cycle(G, orientation='original')
        return [(u, v) for u, v, _ in cycle]
    except nx.NetworkXNoCycle:
        return []

def visualize_graph(G, cycle_length, save_path):
    """Visualize graph with cycle highlighted."""
    if G.number_of_nodes() > 150:
        print(f"    Skipping visualization (too large: {G.number_of_nodes()} nodes)")
        return
        
    plt.figure(figsize=(10, 8))
    
    try:
        pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
    except (ImportError, ModuleNotFoundError):
        pos = nx.circular_layout(G)
    
    cycle_edges = find_cycle_edges(G)
    
    # Draw all edges
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1, alpha=0.6)
    
    # Highlight cycle edges
    if cycle_edges:
        nx.draw_networkx_edges(G, pos, edgelist=cycle_edges, edge_color='red', width=3)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=100, alpha=0.8)
    
    plt.title(f"Graph with {'no cycle' if cycle_length == 0 else f'cycle length {cycle_length}'}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")

def generate_example_graphs(cycle_lengths, args):
    """Generate and save example graphs."""
    print("=" * 40)
    print("GENERATING EXAMPLE GRAPHS")
    print("=" * 40)
    
    os.makedirs("example_graphs", exist_ok=True)
    
    for cycle_len in [0] + cycle_lengths:
        print(f"Generating example for cycle length {cycle_len}...")
        
        G = make_graph_unique_cycle(
            L_target=cycle_len,
            n_layers_rng=(10, 10),
            layer_size_rng=(args.min_layer_size, args.max_layer_size),
            p_extra_forward=args.p_forward,
            seed=None
        )
        
        if has_only_cycle_length(G, cycle_len):
            filename = f"acyclic_example.png" if cycle_len == 0 else f"cycle_length_{cycle_len}_example.png"
            visualize_graph(G, cycle_len, f"example_graphs/{filename}")
        else:
            print(f"    Warning: Invalid example for cycle length {cycle_len}")

def plot_results(cycle_lengths, accuracies, precisions, recalls, f1s, args):
    """Plot evaluation results."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Filter valid (non-NaN) values for precision/recall/F1
    valid_mask = [not np.isnan(p) for p in precisions]
    valid_cycles = [c for c, valid in zip(cycle_lengths, valid_mask) if valid]
    valid_prec = [p for p, valid in zip(precisions, valid_mask) if valid]
    valid_rec = [r for r, valid in zip(recalls, valid_mask) if valid]
    valid_f1 = [f for f, valid in zip(f1s, valid_mask) if valid]
    
    # Accuracy plot
    ax1.plot(cycle_lengths, accuracies, 'bo-', linewidth=2, markersize=6)
    ax1.axvline(x=args.depth, color='red', linestyle='--', alpha=0.7, label=f'GNN Depth = {args.depth}')
    ax1.axvline(x=args.k, color='green', linestyle='--', alpha=0.7, label=f'Training Max = {args.k}')
    ax1.set_xlabel('Cycle Length')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Cycle Length')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    
    # Precision plot
    if valid_prec:
        ax2.plot(valid_cycles, valid_prec, 'ro-', linewidth=2, markersize=6)
    ax2.axvline(x=args.depth, color='red', linestyle='--', alpha=0.7)
    ax2.axvline(x=args.k, color='green', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Cycle Length')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision vs Cycle Length')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    # Recall plot
    if valid_rec:
        ax3.plot(valid_cycles, valid_rec, 'go-', linewidth=2, markersize=6)
    ax3.axvline(x=args.depth, color='red', linestyle='--', alpha=0.7)
    ax3.axvline(x=args.k, color='green', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Cycle Length')
    ax3.set_ylabel('Recall')
    ax3.set_title('Recall vs Cycle Length')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)
    
    # F1 plot
    if valid_f1:
        ax4.plot(valid_cycles, valid_f1, 'mo-', linewidth=2, markersize=6)
    ax4.axvline(x=args.depth, color='red', linestyle='--', alpha=0.7)
    ax4.axvline(x=args.k, color='green', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Cycle Length')
    ax4.set_ylabel('F1 Score')
    ax4.set_title('F1 Score vs Cycle Length')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.1)
    
    plt.suptitle(f'Model Performance: Depth={args.depth}, Training k={args.k}', fontsize=14)
    plt.tight_layout()
    plt.savefig('experiment_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved plot: experiment_results.png")

def main():
    parser = argparse.ArgumentParser(description="Train DirectedGIN for binary cycle detection.")
    parser.add_argument("--k", type=int, required=True, help="Train on lengths {0,2..k} and test on {k+1}")
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size-per-length", "--val-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--depth", type=int, default=3, help="GNN depth")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-layer-size", type=int, default=3)
    parser.add_argument("--max-layer-size", type=int, default=8)
    parser.add_argument("--p-forward", type=float, default=0.3)
    parser.add_argument("--p-er", type=float, default=0.04)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    print("=" * 50)
    print("DIRECTED GIN CYCLE DETECTION")
    print("=" * 50)
    print(f"Training on cycle lengths: {{0, 2, 3, ..., {args.k}}}")
    print(f"Testing on cycle lengths: {{2, 3, ..., {args.k+6}}}")
    print(f"GNN depth: {args.depth}")
    
    if args.k > args.depth:
        print(f"WARNING: Training max cycle length ({args.k}) > GNN depth ({args.depth})")
        print("GNN may not detect cycles longer than its depth!")
    
    # Generate example graphs
    val_cycle_lengths = list(range(2, args.k + 7))
    generate_example_graphs(val_cycle_lengths, args)
    
    # Generate data
    print("\n" + "=" * 50)
    print("GENERATING DATA")
    print("=" * 50)
    
    train_ds, val_sets = build_splits(
        k=args.k,
        n_train=args.train_size,
        n_val_per_length=args.val_size_per_length,
        seed=args.seed,
        n_layers_rng=(10, 10),
        layer_size_rng=(args.min_layer_size, args.max_layer_size),
        p_forward=args.p_forward,
        p_er=args.p_er,
    )
    
    print(f"Training set: {len(train_ds)} graphs")
    print(f"Validation sets: {len(val_sets)} sets")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    in_dim = train_ds[0].x.size(1)
    model = DirectedGINClassifier(in_dim=in_dim, hidden=args.hidden_dim, 
                                  num_classes=2, depth=args.depth).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n" + "=" * 50)
    print("TRAINING")
    print("=" * 50)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_sets[2], batch_size=args.batch_size)  # Use cycle length 2 for validation
    
    train(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)
    
    # Evaluate on all validation sets
    print("\n" + "=" * 50)
    print("EVALUATION")
    print("=" * 50)
    
    cycle_lengths, accuracies, precisions, recalls, f1s = [], [], [], [], []
    
    for cycle_len, val_data in val_sets.items():
        val_loader = DataLoader(val_data, batch_size=args.batch_size)
        metrics = evaluate(model, val_loader, device)
        
        acc = metrics['accuracy']
        prec = metrics['precision']
        rec = metrics['recall']
        f1 = metrics['f1']
        
        cycle_lengths.append(cycle_len)
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        
        prec_str = f"{prec:.4f}" if not np.isnan(prec) else "N/A"
        rec_str = f"{rec:.4f}" if not np.isnan(rec) else "N/A"
        f1_str = f"{f1:.4f}" if not np.isnan(f1) else "N/A"
        
        if cycle_len == 0:
            print(f"Acyclic (0)    : Acc={acc:.4f}, Prec={prec_str}, Rec={rec_str}, F1={f1_str}")
        else:
            print(f"Cycle length {cycle_len:2d}: Acc={acc:.4f}, Prec={prec_str}, Rec={rec_str}, F1={f1_str}")
    
    # Plot results
    plot_results(cycle_lengths, accuracies, precisions, recalls, f1s, args)
    
    print("\nExperiment completed!")

if __name__ == "__main__":
    main()