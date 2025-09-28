import random
from dataclasses import dataclass
import os
from typing import Dict, List, Tuple, Optional
import sys
sys.path.insert(0, ".")
import torch
from torch_geometric.data import Data, Dataset

from puzzles.checkers import CheckerJumpingEnv

@dataclass
class Sample:
    state: Tuple[str, ...]
    y_global: int
    action_mask: torch.Tensor  # [A] bool

# ---------------- NEW: cache helpers ----------------

def _cache_paths(cache_dir: Optional[str], N: int):
    if not cache_dir:
        return None, None
    sup = f"{cache_dir.rstrip('/')}/supervised_N{N}.pt"
    eva = f"{cache_dir.rstrip('/')}/eval_starts_N{N}.pt"
    return sup, eva

def build_samples_for_N(
    N: int,
    per_N: int = 1500,
    walk_min: int = 0,
    walk_max: int | None = None,
    seed: int = 0,
    cache_dir: Optional[str] = None
) -> Tuple[CheckerJumpingEnv, List[Sample]]:
    """
    If cache_dir contains supervised_N{N}.pt, load from cache.
    Else fall back to on-the-fly sampling + BFS labeling.
    """
    env = CheckerJumpingEnv(N)

    sup_path, _ = _cache_paths(cache_dir, N)
    if sup_path and os.path.exists(sup_path):
        raw = torch.load(sup_path)
        samples = [
            Sample(tuple(item["state"]), int(item["y_global"]), item["action_mask"])
            for item in raw
        ]
        return env, samples

    # fallback: on-the-fly (original behavior)
    rng = random.Random(seed)
    torch.manual_seed(seed)

    out: List[Sample] = []
    # seed with initial state
    init_path = env.shortest_path(env.init_state)
    if init_path:
        first = init_path[0]
        y = env.act_space.index_of[first]
        out.append(Sample(env.init_state, y, env.action_mask(env.init_state)))

    attempts = 0
    if walk_max is None:
        walk_max = min((N + 1) ** 2 - 2, 20)
    while len(out) < per_N and attempts < per_N * 20:
        attempts += 1
        k = rng.randint(walk_min, walk_max)
        s = env.random_walk(k)
        path = env.shortest_path(s)
        if not path:
            continue
        first = path[0]
        y = env.act_space.index_of[first]
        out.append(Sample(s, y, env.action_mask(s)))

    rng.shuffle(out)
    return env, out

def load_eval_starts_for_N(N: int, cache_dir: Optional[str], num_starts: Optional[int] = None,
                           seed: int = 0) -> List[Dict]:
    """
    Returns list of dicts with keys: 'state', 'd_opt'.
    If cache exists, loads from cache (optionally truncate to num_starts).
    Else, generates on-the-fly.
    """
    env = CheckerJumpingEnv(N)
    _, eval_path = _cache_paths(cache_dir, N)

    if eval_path and os.path.exists(eval_path):
        lst = torch.load(eval_path)
        if num_starts is not None:
            lst = lst[:num_starts]
        return lst

    # fallback: generate quickly
    from .precompute_datasets import random_walk_samples  # local import
    triples = random_walk_samples(env, count=num_starts or 1000, seed=seed, walk_min=0, walk_max=None)
    return [{"state": s, "d_opt": int(d)} for (s, _, d) in triples]
class CheckersPyGDataset(Dataset):
    """
    PyG dataset with one Data per state:
      Data.x            [L, 8]
      Data.edge_index   [2, E]
      Data.actions_src  [A]  (long)
      Data.actions_dst  [A]  (long)
      Data.action_mask  [A]  (bool)
      Data.y_global     []   (long)
      Data.N, Data.L    []   (long)
    """
    def __init__(self, env: CheckerJumpingEnv, samples: List[Sample]):
        super().__init__()
        self.env = env
        self.samples = samples
        self.actions = env.act_space.actions

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int) -> Data:
        s = self.samples[idx]
        g = self.env.graph_from_state(s.state)
        x = g["x"]
        edge_index = g["edge_index"]
        actions_src = torch.tensor([i for (i, j) in self.actions], dtype=torch.long)
        actions_dst = torch.tensor([j for (i, j) in self.actions], dtype=torch.long)
        y = torch.tensor(s.y_global, dtype=torch.long)
        mask = s.action_mask.clone()  # [A] bool
        data = Data(
            x=x,
            edge_index=edge_index,
            actions_src=actions_src,
            actions_dst=actions_dst,
            action_mask=mask,
            y_global=y,
        )
        data.N = torch.tensor(self.env.N, dtype=torch.long)
        data.L = torch.tensor(self.env.L, dtype=torch.long)
        return data


def split_list(xs, frac_val=0.1, frac_test=0.1, seed=0):
    rng = random.Random(seed)
    xs = xs[:]
    rng.shuffle(xs)
    n = len(xs)
    n_val = int(n * frac_val)
    n_test = int(n * frac_test)
    val = xs[:n_val]
    test = xs[n_val:n_val+n_test]
    train = xs[n_val+n_test:]
    return train, val, test
