import argparse
import os
from collections import deque
from typing import Dict, List, Tuple, Optional

import torch
import sys
sys.path.insert(0, ".")
from puzzles.checkers import CheckerJumpingEnv
from data.dataset import Sample


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def enumerate_reachable(env: CheckerJumpingEnv, limit: Optional[int] = None) -> List[Tuple[str, ...]]:
    """
    Forward exploration from the initial state under legal moves (no backward-in-time edges),
    collecting EVERY reachable state once. This is typically small for small N.
    """
    init = env.init_state
    Q = deque([init])
    seen = {init}
    out = [init]
    while Q:
        s = Q.popleft()
        for mv in env.legal_moves(s):
            nxt = env.apply(s, mv)
            if nxt in seen:
                continue
            seen.add(nxt)
            out.append(nxt)
            Q.append(nxt)
            if limit is not None and len(out) >= limit:
                return out
    return out


@torch.no_grad()
def label_state_first_action_and_dist(env: CheckerJumpingEnv, s: Tuple[str, ...]):
    """
    Return (y_global, d_opt). If goal already reached, d_opt=0 and y_global=None.
    In this dataset we skip states with d_opt=0 for supervised (no move to take).
    """
    path = env.shortest_path(s)
    if path is None:
        return None, None
    d_opt = len(path)
    if d_opt == 0:
        return None, 0
    first = path[0]
    y = env.act_space.index_of[first]
    return int(y), int(d_opt)


def build_supervised_from_set(env: CheckerJumpingEnv, states: List[Tuple[str, ...]]) -> List[dict]:
    items = []
    for s in states:
        y, d = label_state_first_action_and_dist(env, s)
        if y is None:  # either goal state (d=0) or unreachable (shouldn't happen)
            continue
        items.append({
            "state": s,
            "y_global": y,
            "action_mask": env.action_mask(s)  # bool tensor [A]
        })
    return items


def build_eval_from_set(env: CheckerJumpingEnv, states: List[Tuple[str, ...]]) -> List[dict]:
    items = []
    for s in states:
        _, d = label_state_first_action_and_dist(env, s)
        if d is None:
            continue
        items.append({"state": s, "d_opt": int(d)})
    return items


def with_replacement(topup_needed: int, pool: List[dict]) -> List[dict]:
    """Sample with replacement uniformly from 'pool' to produce 'topup_needed' extra items."""
    if topup_needed <= 0 or not pool:
        return []
    idx = torch.randint(low=0, high=len(pool), size=(topup_needed,))
    return [pool[i] for i in idx.tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N_min", type=int, default=1)
    ap.add_argument("--N_max", type=int, default=30)
    ap.add_argument("--per_N", type=int, default=2000, help="desired supervised items per N (will top up with replacement if needed)")
    ap.add_argument("--eval_starts_per_N", type=int, default=2000, help="desired eval starts per N (unique first, then with replacement)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", type=str, default="data/cache")
    ap.add_argument("--enumeration_limit", type=int, default=0, help="optional cap on reachable enumeration (0 = no cap)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ensure_dir(args.cache_dir)

    for N in range(args.N_min, args.N_max + 1):
        env = CheckerJumpingEnv(N)
        sup_path = os.path.join(args.cache_dir, f"supervised_N{N}.pt")
        eval_path = os.path.join(args.cache_dir, f"eval_starts_N{N}.pt")

        print(f"[N={N}] generating…")
        # 1) Enumerate all reachable states from init
        limit = args.enumeration_limit if args.enumeration_limit > 0 else None
        reach = enumerate_reachable(env, limit=limit)

        # 2) Build supervised and eval lists from that set
        supervised_all = build_supervised_from_set(env, reach)  # skips d_opt=0 states
        eval_all = build_eval_from_set(env, reach)              # includes goal with d_opt=0

        # 3) If user asked for more than exist, top up with replacement
        sup = supervised_all[:]
        if len(sup) < args.per_N:
            extra = with_replacement(args.per_N - len(sup), supervised_all)
            sup.extend(extra)

        ev = eval_all[:]
        if len(ev) < args.eval_starts_per_N:
            extra = with_replacement(args.eval_starts_per_N - len(ev), eval_all)
            ev.extend(extra)

        # 4) Save
        torch.save(sup, sup_path)
        torch.save(ev, eval_path)

        print(f"[N={N}] reachable={len(reach)} | supervised_unique={len(supervised_all)} | "
              f"eval_unique={len(eval_all)}")
        print(f"[N={N}] saved: {sup_path} ({len(sup)} items), {eval_path} ({len(ev)} starts)")

    print("Done.")


if __name__ == "__main__":
    main()
