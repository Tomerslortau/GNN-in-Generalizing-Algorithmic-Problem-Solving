from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class ActionSpace:
    actions: List[Tuple[int, int]]
    index_of: Dict[Tuple[int, int], int]

    @staticmethod
    def build(L: int) -> "ActionSpace":
        actions = []
        for i in range(L):
            for d in (-2, -1, +1, +2):
                j = i + d
                if 0 <= j < L:
                    actions.append((i, j))
        index_of = {(i, j): a for a, (i, j) in enumerate(actions)}
        return ActionSpace(actions, index_of)


class CheckerJumpingEnv:
    """
    Board length L = 2*N + 1
    Initial: R...R _ B...B
    Goal   : B...B _ R...R
    R moves right; B moves left; slides (±1) into empty; jumps (±2) over opponent into empty.
    """

    def __init__(self, N: int):
        self.N = N
        self.L = 2 * N + 1
        self.init_state = tuple(['R'] * N + ['_'] + ['B'] * N)
        self.goal_state = tuple(['B'] * N + ['_'] + ['R'] * N)
        self.act_space = ActionSpace.build(self.L)

    def is_goal(self, s: Tuple[str, ...]) -> bool:
        return s == self.goal_state

    # ---------- rules ----------
    def legal_moves(self, s: Tuple[str, ...]) -> List[Tuple[int, int]]:
        L = self.L
        moves = []
        for i, p in enumerate(s):
            if p == '_':
                continue
            if p == 'R':
                j = i + 1
                if j < L and s[j] == '_':
                    moves.append((i, j))
                j = i + 2
                if j < L and s[j] == '_' and s[i + 1] == 'B':
                    moves.append((i, j))
            else:  # 'B'
                j = i - 1
                if j >= 0 and s[j] == '_':
                    moves.append((i, j))
                j = i - 2
                if j >= 0 and s[j] == '_' and s[i - 1] == 'R':
                    moves.append((i, j))
        return moves

    def apply(self, s: Tuple[str, ...], move: Tuple[int, int]) -> Tuple[str, ...]:
        i, j = move
        s_list = list(s)
        s_list[i], s_list[j] = s_list[j], s_list[i]
        return tuple(s_list)

    # ---------- BFS shortest path ----------
    def shortest_path(self, start: Tuple[str, ...], goal: Optional[Tuple[str, ...]] = None):
        if goal is None:
            goal = self.goal_state
        if start == goal:
            return []
        Q = deque([start])
        parent: Dict[Tuple[str, ...], Tuple[Tuple[str, ...], Tuple[int, int]]] = {}
        seen = {start}
        while Q:
            cur = Q.popleft()
            for mv in self.legal_moves(cur):
                nxt = self.apply(cur, mv)
                if nxt in seen:
                    continue
                parent[nxt] = (cur, mv)
                if nxt == goal:
                    # reconstruct
                    path = []
                    s = nxt
                    while s != start:
                        prev, m = parent[s]
                        path.append(m)
                        s = prev
                    return list(reversed(path))
                seen.add(nxt)
                Q.append(nxt)
        return None

    # ---------- random walk sampling ----------
    def random_walk(self, steps: int) -> Tuple[str, ...]:
        s = self.init_state
        for _ in range(steps):
            moves = self.legal_moves(s)
            if not moves:
                break
            s = self.apply(s, moves[torch.randint(len(moves), (1,)).item()])
        return s

    # ---------- graph features ----------
    def graph_from_state(self, s: Tuple[str, ...]) -> Dict[str, torch.Tensor]:
        """
        x: [L, 8]  one-hot(R,B,_) + pos + (N/10) + blank_idx + nR + nB
        edge_index: [2, E], undirected edges (±1 and ±2 hops)
        """
        L = self.L
        feats = []
        blank_idx = s.index('_') / (L - 1)
        nR = s.count('R') / (L - 1)
        nB = s.count('B') / (L - 1)
        for i, tok in enumerate(s):
            occ = {'R': [1., 0., 0.], 'B': [0., 1., 0.], '_': [0., 0., 1.]}[tok]
            pos = i / (L - 1)
            feats.append(occ + [pos, self.N / 10.0, blank_idx])
        x = torch.tensor(feats, dtype=torch.float32)

        edges = []
        for i in range(L):
            for d in (1, 2):
                j = i + d
                if j < L:
                    edges.append((i, j))
                    edges.append((j, i))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return {"x": x, "edge_index": edge_index}

    def action_mask(self, s: Tuple[str, ...]) -> torch.Tensor:
        mask = torch.zeros(len(self.act_space.actions), dtype=torch.bool)
        for mv in self.legal_moves(s):
            mask[self.act_space.index_of[mv]] = True
        return mask
