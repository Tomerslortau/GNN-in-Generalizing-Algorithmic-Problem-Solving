import argparse
import csv
import math
import os
import time
import sys
sys.path.insert(0, '.')
from typing import Dict, List, Tuple, Callable

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from puzzles.checkers import CheckerJumpingEnv
from data.dataset import build_samples_for_N, CheckersPyGDataset, split_list, Sample, load_eval_starts_for_N
from models.policy_gnn import PolicyNet


# -------------------------
# Utils
# -------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_csv(rows: List[dict], out_csv: str):
    if not rows:
        return
    ensure_dir(os.path.dirname(out_csv) or ".")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_fig(fig, out_png: str):
    ensure_dir(os.path.dirname(out_png) or ".")
    fig.savefig(out_png, bbox_inches='tight', dpi=180)
    plt.close(fig)


# -------------------------
# Per-(N,device) cache + tiny on-device LRU
# -------------------------

class _LRU:
    """Simple on-device LRU for (x, mask) keyed by state tuple."""
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self._d: OrderedDict[Tuple[str, ...], Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    def get(self, key: Tuple[str, ...]):
        v = self._d.get(key)
        if v is not None:
            self._d.move_to_end(key)
        return v

    def put(self, key: Tuple[str, ...], value: Tuple[torch.Tensor, torch.Tensor]):
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self.capacity:
            self._d.popitem(last=False)


def _ensure_env_cache(env: CheckerJumpingEnv, device: torch.device):
    """Cache static tensors (edge_index, action lists, reusable x buffer) and an LRU."""
    if not hasattr(env, "_cached"):
        env._cached = {}
    key = str(device)
    if key in env._cached:
        return
    g0 = env.graph_from_state(env.init_state)
    edge_index = g0["edge_index"].to(device).contiguous()
    actions_src = torch.tensor([i for (i, j) in env.act_space.actions], dtype=torch.long, device=device)
    actions_dst = torch.tensor([j for (i, j) in env.act_space.actions], dtype=torch.long, device=device)
    x_buf = g0["x"].to(device).clone()
    env._cached[key] = {
        "edge_index": edge_index,
        "actions_src": actions_src,
        "actions_dst": actions_dst,
        "x_buf": x_buf,
        "lru": _LRU(capacity=100_000),
    }


def _cached_tensors(env: CheckerJumpingEnv, device: torch.device) -> Dict[str, torch.Tensor]:
    return env._cached[str(device)]


# -------------------------
# Action pickers (policies)
# -------------------------

@torch.no_grad()
def _model_step_cached(env: CheckerJumpingEnv, model: PolicyNet, s: Tuple[str, ...], device: torch.device):
    """
    Greedy step using cached topology and an on-device LRU for (x, mask).
    """
    C = _cached_tensors(env, device)
    lru: _LRU = C["lru"]

    cached = lru.get(s)
    if cached is not None:
        x_s, mask = cached
    else:
        x_s = env.graph_from_state(s)["x"].to(device, non_blocking=True)
        mask = env.action_mask(s).to(device, non_blocking=True)
        lru.put(s, (x_s, mask))

    if not mask.any():
        return None

    # Refresh x buffer in-place to avoid allocs
    if C["x_buf"].shape != x_s.shape or C["x_buf"].dtype != x_s.dtype:
        C["x_buf"] = x_s.clone()
    else:
        C["x_buf"].copy_(x_s)

    data = Data(
        x=C["x_buf"],
        edge_index=C["edge_index"],
        actions_src=C["actions_src"],
        actions_dst=C["actions_dst"],
        action_mask=mask
    )
    logits = model(data)
    a_id = logits.masked_fill(~mask, -1e9).argmax().item()
    return env.act_space.actions[a_id]


def make_model_picker(model: PolicyNet) -> Callable[[CheckerJumpingEnv, Tuple[str, ...], torch.device], Tuple[int, int] | None]:
    def _fn(env: CheckerJumpingEnv, s: Tuple[str, ...], device):
        return _model_step_cached(env, model, s, device)
    return _fn


def make_random_picker(seed: int) -> Callable[[CheckerJumpingEnv, Tuple[str, ...], torch.device], Tuple[int, int] | None]:
    rng = torch.Generator().manual_seed(seed)
    def _fn(env: CheckerJumpingEnv, s: Tuple[str, ...], device):
        mask = env.action_mask(s)  # CPU is fine
        if not mask.any():
            return None
        idxs = mask.nonzero(as_tuple=True)[0]
        ridx = int(torch.randint(low=0, high=len(idxs), size=(1,), generator=rng).item())
        a_id = int(idxs[ridx].item())
        return env.act_space.actions[a_id]
    return _fn


@torch.no_grad()
def heuristic_picker(env: CheckerJumpingEnv, s: Tuple[str, ...], device):
    """
    Simple greedy heuristic, no BFS.
    """
    mask = env.action_mask(s)
    if not mask.any():
        return None
    center = env.N
    best = None
    best_score = -1e18
    legal_ids = mask.nonzero(as_tuple=True)[0].tolist()
    for a_id in legal_ids:
        i, j = env.act_space.actions[a_id]
        jump = 1 if abs(j - i) > 1 else 0
        toward_center = 1 if (i < center and j > i) or (i > center and j < i) else 0
        blank_gain = -abs(i - center)  # blank ends at i
        score = 100 * jump + 10 * toward_center + blank_gain
        if score > best_score:
            best_score = score
            best = (i, j)
    return best


# -------------------------
# Core eval helpers (no BFS)
# -------------------------

@torch.no_grad()
def rollout_from_state(env: CheckerJumpingEnv,
                       step_fn: Callable[[CheckerJumpingEnv, Tuple[str, ...], torch.device], Tuple[int, int] | None],
                       start: Tuple[str, ...],
                       device: torch.device,
                       *,
                       d_opt: int,
                       slack: float = 1.0,
                       resid_on_fail: bool = False):
    """
    Greedy rollout with budget = ceil(slack * d_opt). No BFS inside.
    Returns (ok, steps_used, delta_opt_if_ok_else_nan, resid_to_goal_or_-1, d_opt).
    """
    max_steps = int(math.ceil(slack * d_opt))
    s = start
    steps = 0

    is_goal = env.is_goal
    apply_move = env.apply

    for _ in range(max_steps):
        if is_goal(s):
            return True, steps, float(steps - d_opt), 0, d_opt
        mv = step_fn(env, s, device)
        if mv is None:
            break
        s = apply_move(s, mv)
        steps += 1

    if is_goal(s):
        return True, steps, float(steps - d_opt), 0, d_opt

    # No BFS for residuals (fast path)
    resid = -1
    return False, -1, float('nan'), resid, d_opt


# -------------------------
# Training (curriculum + optional DAgger)
# -------------------------

@torch.no_grad()
def dagger_collect_states(env: CheckerJumpingEnv, model: PolicyNet, device,
                          *, rollouts: int = 3, slack: float = 1.5) -> List[Sample]:
    samples: List[Sample] = []
    optimal = (env.N + 1) ** 2 - 1
    max_steps = int(slack * optimal)
    _ensure_env_cache(env, device)  # ensure cache for fast model step
    for _ in range(rollouts):
        s = env.init_state
        seen_states = set()
        for _ in range(max_steps):
            if env.is_goal(s): break
            if s in seen_states: break
            seen_states.add(s)
            path = env.shortest_path(s)
            if not path: break
            first = path[0]
            y = env.act_space.index_of[first]
            samples.append(Sample(s, y, env.action_mask(s)))
            mv = _model_step_cached(env, model, s, device)
            if mv is None: break
            s = env.apply(s, mv)
    return samples


def dedup_samples(samples: List[Sample]) -> List[Sample]:
    seen = set()
    out = []
    for smp in samples:
        if smp.state in seen: continue
        seen.add(smp.state)
        out.append(smp)
    return out


def build_loaders_for_range(Ns: List[int], per_N: int, seed: int, cache_dir: str):
    out = {}
    for N in Ns:
        env, samples = build_samples_for_N(N, per_N=per_N, seed=seed, cache_dir=cache_dir)
        train_s, val_s, test_s = split_list(samples, frac_val=0.1, frac_test=0.1, seed=seed)
        ds_train = CheckersPyGDataset(env, train_s)
        ds_val = CheckersPyGDataset(env, val_s)
        ds_test = CheckersPyGDataset(env, test_s)
        out[N] = dict(
            env=env,
            train_loader=DataLoader(ds_train, batch_size=1, shuffle=True),
            val_loader=DataLoader(ds_val, batch_size=1, shuffle=False),
            test_loader=DataLoader(ds_test, batch_size=1, shuffle=False),
        )
        print(f"[N={N}] dataset sizes — train: {len(ds_train)}, val: {len(ds_val)}, test: {len(ds_test)}")
    return out


def train_curriculum(model: PolicyNet, train_bundle: Dict[int, dict], val_bundle: Dict[int, dict],
                     *, epochs: int, lr: float, device, early_stop_patience: int = 10,
                     dagger_rollouts: int = 0, dagger_slack: float = 1.5):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state, best_val = None, -1.0
    patience = early_stop_patience

    for ep in range(1, epochs + 1):
        model.train()
        run_loss, seen = 0.0, 0
        for N in sorted(train_bundle.keys()):
            for data in train_bundle[N]["train_loader"]:
                data = data.to(device)
                logits = model(data)
                loss = PolicyNet.masked_ce(logits, data.action_mask, int(data.y_global.item()))
                if loss is None: continue
                opt.zero_grad(); loss.backward(); opt.step()
                run_loss += float(loss.item()); seen += 1

        # validate
        model.eval()
        val_accs = []
        with torch.inference_mode():
            for N in sorted(val_bundle.keys()):
                total, correct = 0, 0
                for data in val_bundle[N]["val_loader"]:
                    data = data.to(device)
                    logits = model(data)
                    mask = data.action_mask
                    if not mask.any(): continue
                    pred = logits.masked_fill(~mask, -1e9).argmax().item()
                    correct += int(pred == int(data.y_global.item()))
                    total += 1
                val_accs.append(correct / max(total, 1))
        val_mean = float(sum(val_accs) / max(len(val_accs), 1))

        if ep % 5 == 0 or ep == 1:
            acc_str = " ".join([f"N={N}:{a*100:.1f}%" for N, a in zip(sorted(val_bundle.keys()), val_accs)])
            print(f"epoch {ep:>3} | train_loss {run_loss/max(seen,1):.4f} | val@1 mean {val_mean*100:5.1f}% | {acc_str}")

        if val_mean > best_val:
            best_val = val_mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = early_stop_patience
        else:
            patience -= 1
            if patience <= 0:
                print(f"Early stopping at epoch {ep} (best val@1 mean {best_val*100:.2f}%).")
                break

        # DAgger (optional)
        if dagger_rollouts > 0:
            with torch.inference_mode():
                for N in sorted(train_bundle.keys()):
                    env = train_bundle[N]["env"]
                    _ensure_env_cache(env, device)
                    new_samps = dagger_collect_states(env, model, device,
                                                      rollouts=dagger_rollouts, slack=dagger_slack)
                    if new_samps:
                        ds_train: CheckersPyGDataset = train_bundle[N]["train_loader"].dataset
                        ds_train.samples = dedup_samples(ds_train.samples + new_samps)
                        train_bundle[N]["train_loader"] = DataLoader(ds_train, batch_size=1, shuffle=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N_train_low", type=int, default=3)
    ap.add_argument("--cache_dir", type=str, default="data/cache")
    ap.add_argument("--N_train_high", type=int, default=5)
    ap.add_argument("--N_eval_low", type=int, default=6)
    ap.add_argument("--N_eval_high", type=int, default=9)
    ap.add_argument("--per_N", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out_dir", type=str, default="results/statewise_delta")
    ap.add_argument("--starts_per_N", type=int, default=800)
    ap.add_argument("--rollout_slack", type=float, default=1.0)   # per-start budget multiplier
    ap.add_argument("--dagger_rollouts", type=int, default=0)
    ap.add_argument("--dagger_slack", type=float, default=1.5)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--hiddens", type=int, nargs="+", default=[16, 32, 64])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ensure_dir(args.out_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv_states   = os.path.join(args.out_dir, f"statewise_{stamp}.csv")
    out_png_median   = os.path.join(args.out_dir, f"median_delta_vs_N_{stamp}.png")
    out_png_box      = os.path.join(args.out_dir, f"box_delta_vs_N_{stamp}.png")
    out_png_success  = os.path.join(args.out_dir, f"success_vs_N_{stamp}.png")

    Ns_train = list(range(args.N_train_low, args.N_train_high + 1))
    Ns_eval  = list(range(args.N_eval_low,  args.N_eval_high  + 1))

    # Build datasets/loaders once (training path)
    print("Building datasets...")
    bundle = build_loaders_for_range(Ns_train, per_N=args.per_N, seed=args.seed, cache_dir=args.cache_dir)
    device = torch.device(args.device)

    rows: List[dict] = []

    # -------------------------
    # Evaluate baselines ONCE (fast)
    # -------------------------
    def eval_baselines(Ns: List[int], split_name: str):
        nonlocal rows
        for N in Ns:
            env = CheckerJumpingEnv(N)
            _ensure_env_cache(env, device)

            cached = load_eval_starts_for_N(N, args.cache_dir, num_starts=args.starts_per_N, seed=args.seed + 77)
            starts = [(tuple(item["state"]), int(item["d_opt"])) for item in cached]

            # Optional: warm LRU for starts (helps since we evaluate 2 agents here)
            C = _cached_tensors(env, device); lru: _LRU = C["lru"]
            with torch.inference_mode():
                for s, _ in starts:
                    if lru.get(s) is None:
                        x = env.graph_from_state(s)["x"].to(device, non_blocking=True)
                        m = env.action_mask(s).to(device, non_blocking=True)
                        lru.put(s, (x, m))

            pick_rand  = make_random_picker(seed=args.seed * 1009 + N * 17 + 12345)
            def pick_heur(env_, s_, device_):
                return heuristic_picker(env_, s_, device_)

            policies = {"random": pick_rand, "heuristic": pick_heur}
            succs = {k: 0 for k in policies.keys()}

            rollout = rollout_from_state
            with torch.inference_mode():
                for s, d_opt in starts:
                    for agent_label, step_fn in policies.items():
                        ok, steps, delta, resid, _ = rollout(
                            env, step_fn, s, device, d_opt=d_opt,
                            slack=args.rollout_slack, resid_on_fail=False
                        )
                        succs[agent_label] += int(ok)
                        rows.append(dict(
                            agent="baseline",
                            agent_label=agent_label,
                            split=split_name,
                            N=N,
                            d_opt=d_opt,
                            success=int(ok),
                            steps_used=steps,
                            delta_opt=delta,
                            resid_to_goal=resid,
                            model_depth=None,
                            model_hidden=None,
                        ))

            total = max(len(starts), 1)
            msg = " | ".join([f"{a}: {succs[a]/total*100:.1f}%" for a in ("heuristic", "random")])
            print(f"[{split_name}] N={N} | baseline success — {msg} "
                  f"(slack={args.rollout_slack}× d_opt, {len(starts)} starts)")

    print("Evaluating TRAIN baselines...")
    eval_baselines(Ns_train, "train-Ns")
    print("Evaluating EVAL  baselines...")
    eval_baselines(Ns_eval,  "eval-Ns")

    # -------------------------
    # Train + evaluate models for all (depth, hidden)
    # -------------------------
    for depth in args.depths:
        for hidden in args.hiddens:
            print(f"\n=== Training model(d={depth}, h={hidden}) ===")
            model = PolicyNet(hidden=hidden, n_layers=depth).to(device)
            if device.type != "cpu":
                try:
                    model = torch.compile(model)
                except Exception:
                    pass

            model, best_val = train_curriculum(
                model, bundle, bundle,
                epochs=args.epochs, lr=args.lr, device=device,
                early_stop_patience=10,
                dagger_rollouts=args.dagger_rollouts, dagger_slack=args.dagger_slack
            )
            print(f"Finished training model(d={depth}, h={hidden}) | best val@1 mean ≈ {best_val*100:.2f}%")

            # Evaluate on both splits
            def eval_model_only(Ns: List[int], split_name: str):
                nonlocal rows
                for N in Ns:
                    env = CheckerJumpingEnv(N)
                    _ensure_env_cache(env, device)

                    cached = load_eval_starts_for_N(N, args.cache_dir, num_starts=args.starts_per_N, seed=args.seed + 77)
                    starts = [(tuple(item["state"]), int(item["d_opt"])) for item in cached]

                    # Warm LRU for starts to amortize step cost
                    C = _cached_tensors(env, device); lru: _LRU = C["lru"]
                    with torch.inference_mode():
                        for s, _ in starts:
                            if lru.get(s) is None:
                                x = env.graph_from_state(s)["x"].to(device, non_blocking=True)
                                m = env.action_mask(s).to(device, non_blocking=True)
                                lru.put(s, (x, m))

                    pick_model = make_model_picker(model)
                    succ = 0
                    rollout = rollout_from_state
                    with torch.inference_mode():
                        for s, d_opt in starts:
                            ok, steps, delta, resid, _ = rollout(
                                env, pick_model, s, device,
                                d_opt=d_opt, slack=args.rollout_slack, resid_on_fail=False
                            )
                            succ += int(ok)
                            rows.append(dict(
                                agent="model",
                                agent_label=f"model(d{depth},h{hidden})",
                                split=split_name,
                                N=N,
                                d_opt=d_opt,
                                success=int(ok),
                                steps_used=steps,
                                delta_opt=delta,
                                resid_to_goal=resid,
                                model_depth=depth,
                                model_hidden=hidden,
                            ))
                    total = max(len(starts), 1)
                    print(f"[{split_name}] N={N} | model(d={depth},h={hidden}) success: {succ/total*100:.1f}% "
                          f"(slack={args.rollout_slack}× d_opt, {len(starts)} starts)")

            print("Evaluating TRAIN Ns (context) for this model...")
            eval_model_only(Ns_train, "train-Ns")
            print("Evaluating EVAL  Ns (OOD) for this model...")
            eval_model_only(Ns_eval,  "eval-Ns")

    # Save per-state CSV
    save_csv(rows, out_csv_states)
    print(f"\nSaved statewise CSV to {out_csv_states}")

    # -------------------------
    # Plots
    # -------------------------

    agent_labels = sorted({r["agent_label"] for r in rows})

    def success_deltas_by_N(rows, split, agent_label):
        tmp: Dict[int, List[float]] = {}
        for r in rows:
            if r["split"] != split: continue
            if r.get("agent_label") != agent_label: continue
            delta = r["delta_opt"]
            if isinstance(delta, float) and not math.isnan(delta):
                tmp.setdefault(r["N"], []).append(delta)
        Ns = sorted(tmp.keys())
        return Ns, [tmp[n] for n in Ns]

    # (1) Median distance-from-optimal vs N (success-only, OOD)
    fig1 = plt.figure()
    markers = {}
    base_seq = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', 'p']
    for idx, agent_label in enumerate(agent_labels):
        Ns_eval_sorted, deltas_eval_lists = success_deltas_by_N(rows, "eval-Ns", agent_label)
        medians_eval = [float(np.median(lst)) if len(lst) > 0 else float('nan') for lst in deltas_eval_lists]
        if Ns_eval_sorted:
            m = base_seq[idx % len(base_seq)]
            markers[agent_label] = m
            plt.plot(Ns_eval_sorted, medians_eval, marker=m, label=agent_label)
    plt.xlabel("N (OOD)")
    plt.ylabel("Median distance from optimal (success-only)")
    plt.title("Suboptimality vs N (OOD)")
    plt.grid(True)
    plt.legend()
    save_fig(fig1, out_png_median)
    print(f"Saved median-delta plot to {out_png_median}")

    # (2) Boxplots
    Ns_eval_all = sorted({r["N"] for r in rows if r["split"] == "eval-Ns"})
    fig2 = plt.figure()
    offsets = {label: (-0.3 + i*(0.6/max(1, len(agent_labels)-1))) for i, label in enumerate(agent_labels)}
    width = 0.6 / max(1, len(agent_labels))
    for label in agent_labels:
        perN = {n: lst for n, lst in zip(*success_deltas_by_N(rows, "eval-Ns", label))}
        data = [perN.get(n, []) or [np.nan] for n in Ns_eval_all]
        pos = [n + offsets[label] for n in Ns_eval_all]
        plt.boxplot(data, positions=pos, widths=width, manage_ticks=False)
    plt.xlabel("N (OOD)")
    plt.ylabel("Distance from optimal (success-only)")
    plt.title("Suboptimality distribution per N (OOD)")
    plt.grid(True, axis='y')
    plt.xticks(Ns_eval_all)
    from matplotlib.lines import Line2D
    proxies = [Line2D([0], [0], color='k', lw=0, marker=markers.get(lbl, 'o'), label=lbl) for lbl in agent_labels]
    plt.legend(handles=proxies, labels=agent_labels)
    save_fig(fig2, out_png_box)
    print(f"Saved boxplot to {out_png_box}")

    # (3) Success rate vs N (both splits)
    def success_rate_by_N(rows, split, agent_label):
        tmp: Dict[int, List[int]] = {}
        for r in rows:
            if r["split"] != split: continue
            if r.get("agent_label") != agent_label: continue
            tmp.setdefault(r["N"], []).append(r["success"])
        Ns = sorted(tmp.keys())
        rates = [sum(v) / max(len(v), 1) * 100.0 for v in (tmp[n] for n in Ns)]
        return Ns, rates

    fig3 = plt.figure()
    for split in ["train-Ns", "eval-Ns"]:
        for label in agent_labels:
            Ns_s, rates_s = success_rate_by_N(rows, split, label)
            if Ns_s:
                linestyle = "-" if split == "train-Ns" else "--"
                marker = markers.get(label, 'o')
                nice_split = "train" if split == "train-Ns" else "eval"
                plt.plot(Ns_s, rates_s, marker=marker, linestyle=linestyle, label=f"{label} ({nice_split})")
    plt.xlabel("N")
    plt.ylabel("Success from random starts (%)")
    plt.title("Success vs N")
    plt.grid(True)
    plt.legend(ncol=2)
    save_fig(fig3, out_png_success)
    print(f"Saved success-rate plot to {out_png_success}")

    print("\nDone. Artifacts:")
    print(f"- Per-state CSV: {out_csv_states}")
    print(f"- Median Δ vs N: {out_png_median}")
    print(f"- Boxplots Δ vs N: {out_png_box}")
    print(f"- Success vs N: {out_png_success}")


if __name__ == "__main__":
    main()
