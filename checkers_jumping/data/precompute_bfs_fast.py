import argparse
import os
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Set, Iterable

import torch
import sys
sys.path.insert(0, ".")
from puzzles.checkers import CheckerJumpingEnv


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


class EnvCache:
    def __init__(self, env: CheckerJumpingEnv):
        self.env = env
        self._legal_moves: Dict[Tuple[str, ...], Tuple] = {}
        self._apply: Dict[Tuple[Tuple[str, ...], object], Tuple[str, ...]] = {}
        self._action_mask: Dict[Tuple[str, ...], torch.Tensor] = {}

    def legal_moves(self, s: Tuple[str, ...]) -> Tuple:
        m = self._legal_moves.get(s)
        if m is None:
            m = tuple(self.env.legal_moves(s))
            self._legal_moves[s] = m
        return m

    def apply(self, s: Tuple[str, ...], a) -> Tuple[str, ...]:
        key = (s, a)
        nxt = self._apply.get(key)
        if nxt is None:
            nxt = self.env.apply(s, a)
            self._apply[key] = nxt
        return nxt

    def action_mask(self, s: Tuple[str, ...]) -> torch.Tensor:
        m = self._action_mask.get(s)
        if m is None:
            m = self.env.action_mask(s)
            self._action_mask[s] = m
        return m


def enumerate_graph_and_predecessors(
    env: CheckerJumpingEnv,
    limit: Optional[int] = None,
) -> Tuple[List[Tuple[str, ...]], Dict[Tuple[str, ...], List[Tuple[Tuple[str, ...], object]]], List[Tuple[str, ...]]]:
    """
    Returns:
      - states: list of all reachable states from init (BFS order)
      - preds:  reverse adjacency:
                preds[v] contains tuples (u, a) for every edge u --a--> v
      - goals:  list of states that are goals
    """
    cache = EnvCache(env)
    init = env.init_state
    Q = deque([init])
    seen: Set[Tuple[str, ...]] = {init}
    states: List[Tuple[str, ...]] = [init]
    preds: Dict[Tuple[str, ...], List[Tuple[Tuple[str, ...], object]]] = defaultdict(list)
    goals: List[Tuple[str, ...]] = []

    # Robust goal check
    has_is_goal = hasattr(env, "is_goal")
    def is_goal(s: Tuple[str, ...]) -> bool:
        if has_is_goal:
            return env.is_goal(s)  # type: ignore[attr-defined]
        # Fallback: treat no explicit goal checker as “no goal” (reverse BFS still works if goals appear later)
        return False

    if is_goal(init):
        goals.append(init)

    while Q:
        s = Q.popleft()
        for a in cache.legal_moves(s):
            v = cache.apply(s, a)
            preds[v].append((s, a))
            if v not in seen:
                seen.add(v)
                states.append(v)
                if limit is not None and len(states) >= limit:
                    # Ensure preds for already discovered edges are kept; terminate cleanly
                    return states, preds, goals
                if is_goal(v):
                    goals.append(v)
                Q.append(v)

    return states, preds, goals



def reverse_multisource_bfs_labels(
    env: CheckerJumpingEnv,
    preds: Dict[Tuple[str, ...], List[Tuple[Tuple[str, ...], object]]],
    goals: Iterable[Tuple[str, ...]],
) -> Tuple[Dict[Tuple[str, ...], int], Dict[Tuple[str, ...], object]]:
    """
    For each state s reachable from a goal, computes:
      - dist[s]  = length of shortest path from s to any goal
      - first[s] = optimal first action to take at s toward a goal (None at goals)
    Uses only reverse edges (predecessors).
    """
    dist: Dict[Tuple[str, ...], int] = {}
    first_action: Dict[Tuple[str, ...], object] = {}

    dq = deque()
    # Seed with all known goals
    for g in goals:
        if g not in dist:
            dist[g] = 0
            first_action[g] = None  # goal has no action to take
            dq.append(g)

    # If no explicit goals were found, try to infer goals as states with no outgoing moves
    # (useful when env doesn't expose is_goal)
    if not dist:
        # For nodes with no outgoing moves, they’re terminal; treat as goals
        terminals = [v for v in preds.keys() if len(preds[v]) == 0]
        for t in terminals:
            dist[t] = 0
            first_action[t] = None
            dq.append(t)

    # BFS over reverse edges: for each edge (u --a--> v), if v is labeled and u is not, label u
    while dq:
        v = dq.popleft()
        dv = dist[v]
        for (u, a_uv) in preds.get(v, ()):
            if u in dist:
                continue
            dist[u] = dv + 1
            # The optimal first action at u is precisely the action that moves u -> v
            first_action[u] = a_uv
            dq.append(u)

    return dist, first_action


# ───────────────────────────────────────────────────────────────
# 3) Dataset builders via O(1) lookup from the labeling tables
# ───────────────────────────────────────────────────────────────
def build_supervised_from_states(
    env: CheckerJumpingEnv,
    states: List[Tuple[str, ...]],
    dist: Dict[Tuple[str, ...], int],
    first_action: Dict[Tuple[str, ...], object],
) -> List[dict]:
    items = []
    cache = EnvCache(env)  # tiny cache helps if masks repeat
    for s in states:
        d = dist.get(s)
        if d is None:
            continue  # unreachable from any goal
        if d == 0:
            continue  # goal states: no action to take
        a = first_action[s]
        y = env.act_space.index_of[a]
        items.append({
            "state": s,
            "y_global": int(y),
            "action_mask": cache.action_mask(s),  # bool tensor [A]
        })
    return items


def build_eval_from_states(
    states: List[Tuple[str, ...]],
    dist: Dict[Tuple[str, ...], int],
) -> List[dict]:
    items = []
    for s in states:
        d = dist.get(s)
        if d is None:
            continue
        items.append({"state": s, "d_opt": int(d)})
    return items


def with_replacement(topup_needed: int, pool: List[dict]) -> List[dict]:
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
    ap.add_argument("--enumeration_limit", type=int, default=0, help="cap on reachable enumeration (0 = no cap)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ensure_dir(args.cache_dir)

    enum_limit = args.enumeration_limit if args.enumeration_limit > 0 else None

    for N in range(args.N_min, args.N_max + 1):
        env = CheckerJumpingEnv(N)
        sup_path = os.path.join(args.cache_dir, f"supervised_N{N}.pt")
        eval_path = os.path.join(args.cache_dir, f"eval_starts_N{N}.pt")

        print(f"[N={N}] enumerating graph…")
        states, preds, goals = enumerate_graph_and_predecessors(env, limit=enum_limit)
        print(f"[N={N}] reachable={len(states)} | goals_found={len(goals)}")

        print(f"[N={N}] reverse BFS labeling…")
        dist, first_action = reverse_multisource_bfs_labels(env, preds, goals)
        print(f"[N={N}] labeled={len(dist)} states (incl. goals)")

        # Build datasets via O(1) lookups
        supervised_all = build_supervised_from_states(env, states, dist, first_action)  # skips d=0
        eval_all = build_eval_from_states(states, dist)                                 # includes d=0

        # Top up with replacement if requested
        sup = supervised_all[:]
        if len(sup) < args.per_N:
            sup.extend(with_replacement(args.per_N - len(sup), supervised_all))

        ev = eval_all[:]
        if len(ev) < args.eval_starts_per_N:
            ev.extend(with_replacement(args.eval_starts_per_N - len(ev), eval_all))

        torch.save(sup, sup_path)
        torch.save(ev, eval_path)

        print(f"[N={N}] supervised_unique={len(supervised_all)} | eval_unique={len(eval_all)}")
        print(f"[N={N}] saved: {sup_path} ({len(sup)} items), {eval_path} ({len(ev)} starts)")

    print("Done.")


if __name__ == "__main__":
    main()
