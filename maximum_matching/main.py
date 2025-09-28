from benchmark import sweep_param_benchmark
from train import train_loop
from eval import evaluate_with_baselines, print_eval_results
from model import HybridGNN, GNN
from presearch import presearch_param
from generators import generate_dataset
from torch_geometric.loader import DataLoader
from config import GraphConfig
import torch
import torch.nn as nn
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_name", type=str, default="hidden_dim")
    parser.add_argument("--param_range", type=float, nargs="+", default=[0.004, 0.05])
    parser.add_argument("--graph_type", type=str, default="er_bipartite")
    parser.add_argument("--num_graphs", type=int, default=400)
    parser.add_argument("--action", type=str, default="train")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GraphConfig(args.graph_type)

    if args.action == "train":
        model = HybridGNN()
        train_data = generate_dataset(config, args.num_graphs)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_data = generate_dataset(config, args.num_graphs // 5)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        train_loop(model, train_loader, val_loader, device)
        torch.save(model.state_dict(), "model.pt")
    elif args.action == "evaluate":
        model = HybridGNN()
        model.load_state_dict(torch.load("model.pt"))
        data = generate_dataset(config, args.num_graphs)
        loader = DataLoader(data, batch_size=32, shuffle=False)
        results = evaluate_with_baselines(model, loader, device)
        print_eval_results(results)
    elif args.action == "sweep":
        model = HybridGNN()
        model.load_state_dict(torch.load("../model.pt"))
        sweep_range = np.geomspace(args.param_range[0], args.param_range[1], num=10)
        sweep_param_benchmark(model, args.graph_type, args.param_name, sweep_range, device, num_graphs=args.num_graphs)
    elif args.action == "presearch":
        model = HybridGNN()
        sweep_range = np.geomspace(args.param_range[0], args.param_range[1], num=10)
        presearch_param(args.param_name, sweep_range, args.graph_type, device)


if __name__ == "__main__":
    main()
