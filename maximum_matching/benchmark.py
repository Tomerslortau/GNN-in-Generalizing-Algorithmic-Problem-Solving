from eval import evaluate_with_baselines
from generators import generate_dataset
from config import GraphConfig
import torch.nn as nn
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import pickle


def sweep_param_benchmark(model: nn.Module, graph_type: str, param_name: str, param_values: list, device: torch.device, num_graphs: int = 100):
    config = GraphConfig(graph_type=graph_type)
    training_n = (config.n_min + config.n_max) / 2
    training_p = config.p
    results = []
    for value in param_values:
        if param_name == 'n':
            config.set_fixed_n(int(value))
            config.p = training_p * training_n / value # adjust p to keep density roughly constant.
        else:
            config[param_name] = value
        dataset = generate_dataset(config, num_graphs=num_graphs)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        metrics = evaluate_with_baselines(model.to(device), loader, device)
        results.append((value, metrics))
        print(f"Param: {param_name}={value}, Metrics: {metrics}")
    plot_param_sweep(results, param_name)
    return results


def plot_param_sweep(results: list, param_name: str = "p"):
    plt.figure()
    plt.title(f"Parameter Sweep: {param_name}")
    plt.xlabel('Approximation Success Rate')
    plt.ylabel("Metric Value")
    approx_ratios = [metrics['approx_ratio'] for _, metrics in results]
    approx_rands = [metrics['approx_rand'] for _, metrics in results]
    approx_degs = [metrics['approx_deg'] for _, metrics in results]
    plt.plot([value for value, _ in results], approx_ratios, label="Approx Ratio")
    plt.plot([value for value, _ in results], approx_rands, label="Approx Rand")
    plt.plot([value for value, _ in results], approx_degs, label="Approx Deg")
    plt.legend()
    pickle.dump(results, open(f"param_sweep_{param_name}.pkl", "wb"))
    plt.savefig(f"param_sweep_{param_name}.png")
    plt.show()

import os

def _load_results(pkl_path: str):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def plot_param_sweep_compare(pkl_path_a: str,
                              pkl_path_b: str,
                              label_a: str = 'Run A',
                              label_b: str = 'Run B',
                              param_name: str = 'p',
                              save_path: str | None = None):
    """
    Load two pickle dumps produced by plot_param_sweep/sweep_param_benchmark
    (each is a list of (value, metrics) dicts) and draw them on one graph.

    The plot overlays six curves: Approx Ratio/Rand/Deg for each run.
    """
    results_a = _load_results(pkl_path_a)
    results_b = _load_results(pkl_path_b)

    # Extract x-values and metrics
    x_a = [v for v, _ in results_a]
    x_b = [v for v, _ in results_b]

    approx_ratio_a = [m['approx_ratio']*100 for _, m in results_a]
    approx_ratio_b = [m['approx_ratio']*100 for _, m in results_b]

    approx_rand_a = [m['approx_rand']*100 for _, m in results_a]
    approx_rand_b = [m['approx_rand']*100 for _, m in results_b]

    approx_deg_a = [m['approx_deg']*100 for _, m in results_a]
    approx_deg_b = [m['approx_deg']*100 for _, m in results_b]

    plt.figure()
    plt.title(f"Success Rate Over Number of Nodes")
    plt.xlabel('Number of Nodes')
    plt.ylabel("Success Rate [%]")

    # Define consistent color mapping for metrics
    colors = {
        'ratio': 'tab:blue',
        'rand': 'tab:orange',
        'deg': 'tab:green',
    }

    # Overlay lines: use same color per metric, solid for Run A, dashed for Run B
    plt.plot(x_a, approx_ratio_a, color=colors['ratio'], label=f"{label_a} – Model")
    plt.plot(x_b, approx_ratio_b, color=colors['ratio'], linestyle='--', label=f"{label_b} – Model")

    plt.plot(x_a, approx_rand_a, color=colors['rand'], label=f"{label_a} – Random")
    plt.plot(x_b, approx_rand_b, color=colors['rand'], linestyle='--', label=f"{label_b} – Random")

    plt.plot(x_a, approx_deg_a, color=colors['deg'], label=f"{label_a} – Degree")
    plt.plot(x_b, approx_deg_b, color=colors['deg'], linestyle='--', label=f"{label_b} – Degree")

    plt.legend(loc='center left')

    # Determine save path if not supplied
    if save_path is None:
        base = f"param_sweep_compare_{param_name}.png"
        save_path = os.path.join(os.getcwd(), base)

    plt.savefig(save_path)
    plt.show()
    return save_path


#plot_param_sweep_compare("./param_sweep_n_erbp.pkl", "./param_sweep_n.pkl", label_a="BP-ER", label_b="G-ER", param_name="n", save_path="param_sweep_compare_n.png")