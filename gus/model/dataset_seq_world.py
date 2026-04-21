"""Dataset yielding (decision, sampled_world) pairs for the full v1 student.

Each __getitem__ returns:
- tokens, attention_mask        (decision-level state, tokenized play sequence)
- world_mask                    [28] — for each domino, is it "unseen"?
- world_assignment              [28, 3] float — one-hot of seat per unseen domino in this world
  (all zeros for dominoes in my hand or already played)
- q_per_world                   [7] — oracle Q per action for this specific world
- legal_mask                    [7] — legal actions at this decision
- e_q                           [7] — marginal E[Q] (V_head target; uses action_taken)
- action_taken                  int — π_me_head target
- belief_target, belief_mask    [28] / [28] — belief truth target

One random world per __getitem__ call — the Q_head sees different worlds
across epochs. This is the 3400×-denser supervision signal compared to
belief-only.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .features import extract_belief_target, reconstruct_prior_plays
from .tokenize import SEQ_LEN, tokenize_decision
from .voids import voids_feature_vector

N_DOMINOES = 28
N_SEATS = 3  # left_opp, partner, right_opp


class JointWorldFullDataset(Dataset):
    """One sample per (decision, random-world) pair.

    Memory: holds all games' joint-world tensors on CPU. For 1000 games × 28
    decisions × avg M=3400 × (world_hands [3,7] + q_per_world [7]) floats,
    total ~11 GB. Fits in typical laptop RAM; if not, use lazy loading.
    """

    def __init__(self, corpus_path: str | Path | list[str | Path], seed: int | None = None):
        # Accept a single .pt path, a glob, or a list of paths/globs.
        from glob import glob
        raw: list[str]
        if isinstance(corpus_path, (list, tuple)):
            raw = [str(p) for p in corpus_path]
        else:
            raw = [str(corpus_path)]

        paths: list[Path] = []
        for s in raw:
            if any(ch in s for ch in "*?["):
                matches = sorted(glob(s))
                if not matches:
                    raise FileNotFoundError(f"glob matched no files: {s}")
                paths.extend(Path(m) for m in matches)
            else:
                paths.append(Path(s))

        self.games: list = []
        self.seeds: list = []
        for path in paths:
            blob = torch.load(str(path), weights_only=False)
            self.games.extend(blob["results"])
            self.seeds.extend(blob.get("seeds", []))

        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

        # Flatten to (game_idx, decision_idx) index. Only include decisions
        # that actually carry a joint-world tensor.
        self.index: list[tuple[int, int]] = []
        for g, game in enumerate(self.games):
            for d_idx, dec in enumerate(game.decisions):
                if dec.world_hands is not None and dec.q_per_world is not None:
                    self.index.append((g, d_idx))

    def __len__(self) -> int:
        return len(self.index)

    def _world_to_assignment(
        self,
        world_hands_m: torch.Tensor,  # [3, 7] dominoes in relative seats {left, partner, right}
    ) -> torch.Tensor:
        """Convert one sampled world's hand layout to a [28, 3] seat-one-hot
        tensor. Dominoes not in world_hands (i.e., in my hand or played) get
        the zero vector.
        """
        assign = torch.zeros(N_DOMINOES, N_SEATS, dtype=torch.float32)
        for seat in range(N_SEATS):
            for d in world_hands_m[seat].tolist():
                d = int(d)
                if 0 <= d < N_DOMINOES:
                    assign[d, seat] = 1.0
        return assign

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        g_idx, d_idx = self.index[idx]
        game = self.games[g_idx]
        decision = game.decisions[d_idx]
        current_player = int(decision.player)

        # Tokenized state
        tokens, attn_mask = tokenize_decision(
            game.hands,
            int(game.decl_id),
            game.decisions,
            d_idx,
        )

        # Belief target (for belief_head)
        prior_plays = reconstruct_prior_plays(game.hands, game.decisions, d_idx)
        belief_target, belief_mask = extract_belief_target(
            game.hands,
            prior_plays,
            current_player,
        )

        # Random world from this decision's joint-world tensor
        world_hands = decision.world_hands  # [M, 3, 7]
        q_per_world = decision.q_per_world  # [M, 7]
        M = world_hands.shape[0]
        m = int(torch.randint(0, M, (1,), generator=self._rng).item())

        world_assignment = self._world_to_assignment(world_hands[m])  # [28, 3]
        q_for_world = q_per_world[m].float()  # [7]

        # Marginal E[Q] target (V_head); use action_taken's slot
        e_q = decision.e_q.float()  # [7]
        action_taken = int(decision.action_taken)
        legal_mask = decision.legal_mask.bool()  # [7]

        # Engine-computed void features (explicit signal for belief head)
        voids = voids_feature_vector(prior_plays, int(game.decl_id), current_player)  # [24]

        return {
            "tokens": tokens,
            "attention_mask": attn_mask,
            "belief_target": belief_target,
            "belief_mask": belief_mask,
            "world_assignment": world_assignment,  # [28, 3]
            "q_per_world": q_for_world,            # [7]
            "e_q": e_q,                            # [7]
            "action_taken": torch.tensor(action_taken, dtype=torch.long),
            "legal_mask": legal_mask,
            "decision_idx": torch.tensor(d_idx, dtype=torch.long),
            "player": torch.tensor(current_player, dtype=torch.long),
            "voids": voids,                        # [24]
        }
