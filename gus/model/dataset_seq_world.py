"""Dataset yielding (decision, sampled_world) pairs for the full v1 student.

Each item returns (Schema v1):
- tokens, attention_mask        (decision-level state, tokenized play sequence)
- world_mask                    [28] — for each domino, is it "unseen"?
- world_assignment              [28, 3] float — one-hot of seat per unseen domino in this world
  (all zeros for dominoes in my hand or already played)
- q_per_world                   [7] — oracle Q per action for this specific world
- legal_mask                    [7] — legal actions at this decision
- e_q                           [7] — marginal E[Q] (V_head target; uses action_taken)
- action_taken                  int — π_me_head target
- belief_target, belief_mask    [28] / [28] — belief truth target

Schema v2 adds (if present in the corpus — graceful fallback to None if absent):
- oracle_softmax_per_seat       [4, 7] — p_make-based softmax per seat (π_opp head target)
- legal_mask_per_seat           [4, 7] — boolean legal mask per seat
- voids_per_seat                [4, 3, 8] — void inferences per seat

One random world per item — the Q_head sees different worlds across epochs.
This is the 3400×-denser supervision signal compared to belief-only.

Two classes live in this module:

- JointWorldFullDataset (map-style): holds all games in memory. Suitable for
  eval corpora (a few hundred MB). OOMs at 10k-game train corpora (~110 GB).
- JointWorldFullIterable (iterable-style): streams chunks one at a time with
  a shuffle buffer. Suitable for train corpora at arbitrary scale — memory
  footprint is bounded by one chunk + buffer.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, IterableDataset

from .auction import auction_feature_vector
from .features import extract_belief_target, reconstruct_prior_plays
from .strategy_features import extract_strategy_action_features, extract_strategy_features
from .tokenize import SEQ_LEN, tokenize_decision
from .voids import voids_feature_vector

N_DOMINOES = 28
N_SEATS = 3  # left_opp, partner, right_opp


def valid_world_indices_for_decision(game, d_idx: int) -> torch.Tensor:
    """Indices m into ``game.decisions[d_idx].world_hands`` ([M, 3, 7]) whose
    stored sampled world is VALID for the decision.

    A world is valid iff, reconstructing the decision's ground truth from the
    game record alone (initial deal + prior plays):

    - its in-range tiles (0 <= d < 28) form EXACTLY the decision's unseen set as
      a multiset — occupancy 1 on every unseen domino, 0 everywhere else (so no
      duplicate, no tile from the actor's own hand, no already-played tile, and
      nothing missing); and
    - each relative-seat row r holds exactly ``7 - (plays already made by
      absolute seat (P + r + 1) % 4)`` in-range tiles, where P is the actor.

    The pre-repair sampler injected domino 0-0 rather than rejecting infeasible
    draws, so 27-67% of stored worlds per decision are malformed (#52). This
    predicate is the read-time filter that excludes them from training.

    Vectorized over M; entries outside [0, 28) are treated as padding. Returns a
    1-D long tensor (possibly empty) on CPU.
    """
    decision = game.decisions[d_idx]
    world_hands = decision.world_hands
    if world_hands is None:
        return torch.empty(0, dtype=torch.long)
    wh = world_hands.detach().to("cpu").long()  # [M, 3, 7]
    M = wh.shape[0]
    if M == 0:
        return torch.empty(0, dtype=torch.long)

    actor = int(decision.player)

    # Replay prior plays from the record: decision j played hands[player][slot].
    played_by_seat = [0, 0, 0, 0]
    played_tiles: set[int] = set()
    for j in range(d_idx):
        dj = game.decisions[j]
        pj = int(dj.player)
        slot = int(dj.action_taken)
        hand = [int(x) for x in game.hands[pj]]
        if 0 <= slot < len(hand) and hand[slot] >= 0:
            played_tiles.add(hand[slot])
            played_by_seat[pj] += 1

    actor_initial = {int(x) for x in game.hands[actor] if int(x) >= 0}

    # unseen = all 28 dominoes minus the actor's initial hand minus opponents'
    # prior plays (equivalently: minus the actor's remaining hand and ALL plays).
    unseen = torch.zeros(N_DOMINOES, dtype=torch.bool)
    for d in range(N_DOMINOES):
        if d not in actor_initial and d not in played_tiles:
            unseen[d] = True

    # Occupancy [M, 28] over in-range tiles via scatter_add (no Python loop over M).
    flat = wh.reshape(M, N_SEATS * 7)              # [M, 21]
    in_range = (flat >= 0) & (flat < N_DOMINOES)   # [M, 21]
    idx = flat.clamp(0, N_DOMINOES - 1)            # out-of-range folded to 0, masked below
    occ = torch.zeros(M, N_DOMINOES, dtype=torch.long)
    occ.scatter_add_(1, idx, in_range.long())      # [M, 28]

    # Exact-cover: occ == 1 exactly on the unseen set, 0 elsewhere.
    unseen_row = unseen.unsqueeze(0)               # [1, 28]
    cover_ok = torch.where(unseen_row, occ == 1, occ == 0).all(dim=1)  # [M]

    # Row-cardinality: per relative seat r, in-range count == 7 - plays by seat.
    row_counts = in_range.reshape(M, N_SEATS, 7).sum(dim=2)  # [M, 3]
    expected = torch.tensor(
        [7 - played_by_seat[(actor + r + 1) % 4] for r in range(N_SEATS)],
        dtype=torch.long,
    )
    row_ok = (row_counts == expected.unsqueeze(0)).all(dim=1)  # [M]

    return torch.nonzero(cover_ok & row_ok, as_tuple=False).squeeze(1).long()


class JointWorldFullDataset(Dataset):
    """One sample per (decision, random-world) pair.

    Memory: holds all games' joint-world tensors on CPU. For 1000 games × 28
    decisions × avg M=3400 × (world_hands [3,7] + q_per_world [7]) floats,
    total ~11 GB. Fits in typical laptop RAM; if not, use lazy loading.
    """

    def __init__(
        self,
        corpus_path: str | Path | list[str | Path],
        seed: int | None = None,
        include_strategy_features: bool = False,
        shuffle_bids: bool = False,
        filter_invalid_worlds: bool = True,
    ):
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
        self.include_strategy_features = include_strategy_features

        # Capacity control (#24): when shuffle_bids, source each game's auction
        # feature from a DIFFERENT game (a fixed derangement), keeping everything
        # else (tokens, belief target, voids, worlds) real. This gives the auction
        # student the SAME BidsEncoder capacity + real auction-feature distribution
        # but breaks the auction↔deal correlation — so any belief-acc edge over the
        # voids control must come from added capacity, not auction information.
        self.shuffle_bids = shuffle_bids
        self.bid_perm: list[int] | None = None
        if shuffle_bids:
            n = len(self.games)
            # Cyclic shift by 1: game g uses game (g+1)'s auction. A guaranteed
            # derangement for n>1 (no game keeps its own auction), deterministic,
            # and fully decorrelated since deals are independent across games.
            self.bid_perm = [(g + 1) % n for g in range(n)]

        # Flatten to (game_idx, decision_idx) index. Only include decisions
        # that actually carry a joint-world tensor.
        #
        # With filter_invalid_worlds (default), each such decision is screened
        # by valid_world_indices_for_decision: decisions with ZERO valid stored
        # worlds are dropped, and per-item sampling in __getitem__ draws only
        # from the valid indices. self._valid_idx is None when filtering is off,
        # in which case behavior is bit-identical to the pre-filter dataset.
        self.filter_invalid_worlds = filter_invalid_worlds
        self._valid_idx: dict[tuple[int, int], torch.Tensor] | None = None
        self.index: list[tuple[int, int]] = []
        if filter_invalid_worlds:
            self._valid_idx = {}
            n_dropped = 0
            frac_sum = 0.0
            frac_n = 0
            for g, game in enumerate(self.games):
                for d_idx, dec in enumerate(game.decisions):
                    if dec.world_hands is None or dec.q_per_world is None:
                        continue
                    valid = valid_world_indices_for_decision(game, d_idx)
                    M = int(dec.world_hands.shape[0])
                    if M > 0:
                        frac_sum += valid.numel() / M
                        frac_n += 1
                    if valid.numel() == 0:
                        n_dropped += 1
                        continue
                    self.index.append((g, d_idx))
                    self._valid_idx[(g, d_idx)] = valid
            mean_frac = frac_sum / frac_n if frac_n else 0.0
            print(
                f"[JointWorldFullDataset] filter_invalid_worlds: "
                f"{n_dropped} decisions dropped (0 valid worlds), "
                f"{len(self.index)} items kept, mean valid fraction {mean_frac:.3f}"
            )
        else:
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

        # Random world from this decision's joint-world tensor. When filtering,
        # draw only from the decision's valid world indices.
        world_hands = decision.world_hands  # [M, 3, 7]
        q_per_world = decision.q_per_world  # [M, 7]
        M = world_hands.shape[0]
        if self._valid_idx is not None:
            valid = self._valid_idx[(g_idx, d_idx)]
            pos = int(torch.randint(0, int(valid.numel()), (1,), generator=self._rng).item())
            m = int(valid[pos].item())
        else:
            m = int(torch.randint(0, M, (1,), generator=self._rng).item())

        world_assignment = self._world_to_assignment(world_hands[m])  # [28, 3]
        q_for_world = q_per_world[m].float()  # [7]

        # Marginal E[Q] target (V_head); use action_taken's slot
        e_q = decision.e_q.float()  # [7]
        action_taken = int(decision.action_taken)
        legal_mask = decision.legal_mask.bool()  # [7]

        # Engine-computed void features (explicit signal for belief head)
        voids = voids_feature_vector(prior_plays, int(game.decl_id), current_player)  # [24]

        # Auction features (#24): zero vector for the seed-imposed corpus that
        # carries no real bids — degrades to the play+voids belief. Under
        # shuffle_bids (capacity control) the auction comes from a different game.
        bid_game = self.games[self.bid_perm[g_idx]] if self.shuffle_bids else game
        bids = auction_feature_vector(
            getattr(bid_game, "bids", None),
            getattr(bid_game, "bidder", None),
            getattr(bid_game, "bid_value", None),
            int(bid_game.decl_id),
            current_player,
        )  # [28]

        item = {
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
            "bids": bids,                          # [28]
        }
        if self.include_strategy_features:
            item["strategy_features"] = extract_strategy_features(
                game.hands, int(game.decl_id), game.decisions, d_idx
            )
            item["strategy_action_features"] = extract_strategy_action_features(
                game.hands, int(game.decl_id), game.decisions, d_idx
            )
        # Schema v2 fields — present only if corpus was generated with --schema v2
        if decision.oracle_softmax_per_seat is not None:
            item["oracle_softmax_per_seat"] = decision.oracle_softmax_per_seat.float()   # [4, 7]
            item["legal_mask_per_seat"] = decision.legal_mask_per_seat.bool()            # [4, 7]
            item["voids_per_seat"] = decision.voids_per_seat.float().reshape(4, 24)      # [4, 24]
        return item


def _expand_paths(corpus_path: str | Path | list[str | Path]) -> list[Path]:
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
    return paths


def _world_to_assignment(world_hands_m: torch.Tensor) -> torch.Tensor:
    assign = torch.zeros(N_DOMINOES, N_SEATS, dtype=torch.float32)
    for seat in range(N_SEATS):
        for d in world_hands_m[seat].tolist():
            d = int(d)
            if 0 <= d < N_DOMINOES:
                assign[d, seat] = 1.0
    return assign


def _build_item(
    game,
    d_idx: int,
    rng: torch.Generator,
    include_strategy_features: bool = False,
    valid_idx: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    decision = game.decisions[d_idx]
    current_player = int(decision.player)
    tokens, attn_mask = tokenize_decision(
        game.hands, int(game.decl_id), game.decisions, d_idx
    )
    prior_plays = reconstruct_prior_plays(game.hands, game.decisions, d_idx)
    belief_target, belief_mask = extract_belief_target(
        game.hands, prior_plays, current_player
    )
    world_hands = decision.world_hands  # [M, 3, 7]
    q_per_world = decision.q_per_world  # [M, 7]
    M = world_hands.shape[0]
    # When valid_idx is given, draw the world uniformly from the valid indices;
    # otherwise draw uniformly over all M (bit-identical to the pre-filter path).
    if valid_idx is not None:
        pos = int(torch.randint(0, int(valid_idx.numel()), (1,), generator=rng).item())
        m = int(valid_idx[pos].item())
    else:
        m = int(torch.randint(0, M, (1,), generator=rng).item())
    world_assignment = _world_to_assignment(world_hands[m])
    q_for_world = q_per_world[m].float()
    e_q = decision.e_q.float()
    action_taken = int(decision.action_taken)
    legal_mask = decision.legal_mask.bool()
    voids = voids_feature_vector(prior_plays, int(game.decl_id), current_player)
    bids = auction_feature_vector(
        getattr(game, "bids", None),
        getattr(game, "bidder", None),
        getattr(game, "bid_value", None),
        int(game.decl_id),
        current_player,
    )
    item = {
        "tokens": tokens,
        "attention_mask": attn_mask,
        "belief_target": belief_target,
        "belief_mask": belief_mask,
        "world_assignment": world_assignment,
        "q_per_world": q_for_world,
        "e_q": e_q,
        "action_taken": torch.tensor(action_taken, dtype=torch.long),
        "legal_mask": legal_mask,
        "decision_idx": torch.tensor(d_idx, dtype=torch.long),
        "player": torch.tensor(current_player, dtype=torch.long),
        "voids": voids,
        "bids": bids,
    }
    if include_strategy_features:
        item["strategy_features"] = extract_strategy_features(
            game.hands, int(game.decl_id), game.decisions, d_idx
        )
        item["strategy_action_features"] = extract_strategy_action_features(
            game.hands, int(game.decl_id), game.decisions, d_idx
        )
    # Schema v2 fields — present only if corpus was generated with --schema v2
    if decision.oracle_softmax_per_seat is not None:
        item["oracle_softmax_per_seat"] = decision.oracle_softmax_per_seat.float()   # [4, 7]
        item["legal_mask_per_seat"] = decision.legal_mask_per_seat.bool()            # [4, 7]
        item["voids_per_seat"] = decision.voids_per_seat.float().reshape(4, 24)      # [4, 24]
    return item


class JointWorldFullIterable(IterableDataset):
    """Streams (decision, sampled-world) items from chunked .pt files one
    chunk at a time, with a shuffle buffer.

    Memory footprint is bounded: one chunk in RAM + `buffer_size` built items.
    For 100 chunks × ~1.1 GB each, peak is ~1.5 GB (vs 110 GB eager).

    __len__ is computed by a one-shot metadata scan on init (loads each chunk,
    counts qualifying decisions, discards). On a 100-chunk corpus this scan
    takes a few minutes; set `length_cache_path` to cache the result to disk.

    Shuffling:
    - shuffle=True: random chunk order per __iter__, plus a shuffle buffer
      that mixes items across the trailing window of chunks. Good enough
      diversity for SGD; not globally uniform.
    - shuffle=False: deterministic chunk order, items in chunk-native order.
      Use this for eval.

    num_workers: tested with 0. Multi-worker requires splitting paths across
    workers (not implemented yet).
    """

    def __init__(
        self,
        corpus_path: str | Path | list[str | Path],
        shuffle: bool = True,
        seed: int | None = None,
        buffer_size: int = 8192,
        length_cache_path: str | Path | None = None,
        include_strategy_features: bool = False,
        filter_invalid_worlds: bool = True,
    ):
        self.paths = _expand_paths(corpus_path)
        self.shuffle = shuffle
        self.seed = seed
        self.buffer_size = max(1, buffer_size)
        self.include_strategy_features = include_strategy_features
        self.filter_invalid_worlds = filter_invalid_worlds
        self._len = self._compute_length(length_cache_path)

    def _compute_length(self, cache_path: str | Path | None) -> int:
        # The filter state is part of the cache key: a cache written with a
        # different filter setting counts a different set of decisions and must
        # not be reused (a stale count would make __len__ lie about iteration).
        filt = "filter=1|" if self.filter_invalid_worlds else "filter=0|"
        key = filt + "|".join(str(p.resolve()) for p in self.paths)
        cache: dict = {}
        if cache_path and Path(cache_path).exists():
            cache = torch.load(str(cache_path), weights_only=False)
            if cache.get("key") == key:
                return int(cache["length"])
        total = 0
        for p in self.paths:
            blob = torch.load(str(p), weights_only=False)
            for game in blob["results"]:
                for d_idx, dec in enumerate(game.decisions):
                    if dec.world_hands is None or dec.q_per_world is None:
                        continue
                    if self.filter_invalid_worlds:
                        # Count only decisions that keep at least one valid world;
                        # zero-valid decisions are skipped during iteration.
                        if valid_world_indices_for_decision(game, d_idx).numel() == 0:
                            continue
                    total += 1
            del blob
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"key": key, "length": total}, str(cache_path))
        return total

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker is not None:
            raise RuntimeError(
                "JointWorldFullIterable does not yet support num_workers > 0"
            )

        py_rng = random.Random(self.seed)
        torch_rng = torch.Generator()
        if self.seed is not None:
            torch_rng.manual_seed(self.seed)

        paths = list(self.paths)
        if self.shuffle:
            py_rng.shuffle(paths)

        buffer: list = []

        for path in paths:
            blob = torch.load(str(path), weights_only=False)
            games = blob["results"]
            items: list[tuple[int, int]] = []
            for g, game in enumerate(games):
                for d_idx, dec in enumerate(game.decisions):
                    if dec.world_hands is not None and dec.q_per_world is not None:
                        items.append((g, d_idx))
            if self.shuffle:
                py_rng.shuffle(items)

            for g_idx, d_idx in items:
                valid_idx = None
                if self.filter_invalid_worlds:
                    valid_idx = valid_world_indices_for_decision(games[g_idx], d_idx)
                    if valid_idx.numel() == 0:
                        continue  # every stored world malformed — skip (keeps __len__ honest)
                built = _build_item(
                    games[g_idx],
                    d_idx,
                    torch_rng,
                    include_strategy_features=self.include_strategy_features,
                    valid_idx=valid_idx,
                )
                if not self.shuffle:
                    yield built
                    continue
                buffer.append(built)
                if len(buffer) >= self.buffer_size:
                    k = py_rng.randrange(len(buffer))
                    buffer[k], buffer[-1] = buffer[-1], buffer[k]
                    yield buffer.pop()

            del blob, games

        if self.shuffle:
            py_rng.shuffle(buffer)
        for item in buffer:
            yield item
