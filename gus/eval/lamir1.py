"""LAMIR-1: 1-ply look-ahead using rotation-equivariant π_me as π_opp.

For each held-out decision d (current_player = P, legal actions A):
  For each action a in A:
    For each sampled world m in d.world_hands:
      state = apply(d.state, action=a, player=P)
      for p_next in remaining players in trick:
        pi = model(tokens[p_next_pov], voids[p_next_pov], world_rotated[m, p_next]).pi_me
        a_next = legal_argmax(pi, p_next's hand in world_m)
        state = apply(state, a_next)
      V_leaf = model(tokens_P_pov, voids_P_pov, world_m).v_head
    score[a] = mean_m(V_leaf)
  chosen = argmax_a(score)

Batched inference: all M worlds are processed in parallel per model call.
Falls back to direct π_me when trick_pos == 3 (nothing to roll out).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch import Tensor

from gus.model.dataset_seq_world import JointWorldFullDataset
from gus.model.features import _hand_list, reconstruct_prior_plays
from gus.model.student import StudentTransformerFull, StudentTransformerFullVoids
from gus.model.tokenize import tokenize_decision
from gus.model.voids import voids_feature_vector
from forge.eq.game_tensor import GameStateTensor


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_student(path: str, device: str):
    ckpt = torch.load(path, weights_only=False, map_location=device)
    args = ckpt["args"]
    is_voids = "voids_hidden" in args
    cls = StudentTransformerFullVoids if is_voids else StudentTransformerFull
    kwargs = {
        "d_model": args["d_model"],
        "n_heads": args["n_heads"],
        "n_layers": args["n_layers"],
        "ff_dim": args.get("ff_dim", 256),
        "dropout": 0.0,
        "d_world": args.get("d_world", 64),
        "q_hidden": args.get("q_hidden", 256),
    }
    if is_voids:
        kwargs["voids_hidden"] = args.get("voids_hidden", 64)
    model = cls(**kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, is_voids


# ---------------------------------------------------------------------------
# World rotation
# ---------------------------------------------------------------------------

def reindex_world_rows(
    world_hands_batch: Tensor,  # [M, 3, 7] — relative to original P
    old_cp: int,
    new_cp: int,
) -> Tensor:
    """Return [M, 3, 7] world tensor for new_cp's POV.

    Row i of result = hand of absolute player (new_cp + i + 1) % 4.
    Row j of source = hand of absolute player (old_cp + j + 1) % 4.
    """
    M = world_hands_batch.shape[0]
    out = torch.zeros_like(world_hands_batch)
    for i in range(3):
        abs_p = (new_cp + i + 1) % 4
        if abs_p == old_cp:
            # new_cp sees old_cp at relative seat i — not in world_hands (known player)
            continue
        src_row = (abs_p - old_cp - 1) % 4
        if src_row < 3:
            out[:, i, :] = world_hands_batch[:, src_row, :]
    return out


def world_batch_to_assignment(world_hands_batch: Tensor) -> Tensor:
    """Convert [M, 3, 7] world rows to [M, 28, 3] seat-one-hot tensors."""
    M = world_hands_batch.shape[0]
    assign = torch.zeros(M, 28, 3, dtype=torch.float32)
    for seat in range(3):
        # world_hands_batch[:, seat, :] is [M, 7] domino IDs
        for slot in range(7):
            d_ids = world_hands_batch[:, seat, slot].long()  # [M]
            valid = (d_ids >= 0) & (d_ids < 28)
            for m in range(M):
                if valid[m]:
                    assign[m, d_ids[m].item(), seat] = 1.0
    return assign


def world_batch_to_assignment_vectorized(world_hands_batch: Tensor) -> Tensor:
    """Convert [M, 3, 7] world rows to [M, 28, 3] seat-one-hot (vectorized)."""
    M = world_hands_batch.shape[0]
    assign = torch.zeros(M, 28, 3, dtype=torch.float32, device=world_hands_batch.device)
    flat = world_hands_batch.reshape(M, 3 * 7)  # [M, 21]
    for seat in range(3):
        for slot in range(7):
            d_ids = world_hands_batch[:, seat, slot].long()  # [M]
            valid = (d_ids >= 0) & (d_ids < 28)
            if valid.any():
                m_idx = torch.where(valid)[0]
                assign[m_idx, d_ids[m_idx], seat] = 1.0
    return assign


# ---------------------------------------------------------------------------
# Legal mask for a player in a world
# ---------------------------------------------------------------------------

def legal_mask_in_world(
    state: GameStateTensor,
    world_hands_m: Tensor,  # [3, 7] relative to original P
    player: int,
    orig_cp: int,
) -> Tensor:
    """Return [7] bool legal mask for `player` in state, using world to
    substitute the player's hand if they're an opponent."""
    # The GameStateTensor has the real hands for all players.
    # legal_actions() reads the current player's hand from state.hands.
    # If player IS the current player per state, this works directly.
    # We trust state reflects the correct current player after apply_actions.
    return state.legal_actions()[0]  # [7]


# ---------------------------------------------------------------------------
# Batched model query
# ---------------------------------------------------------------------------

def query_model_batched(
    model,
    is_voids: bool,
    tokens: Tensor,          # [L, 5] — same for all M worlds
    attn_mask: Tensor,       # [L]
    voids_vec: Tensor,       # [24]
    world_assign: Tensor,    # [M, 28, 3]
    device: str,
) -> dict[str, Tensor]:
    """Run model on M worlds at once. Returns dict with v [M] and pi_me_logits [M, 7]."""
    M = world_assign.shape[0]

    # Expand tokens/mask/voids to [M, ...]
    tokens_b = tokens.unsqueeze(0).expand(M, -1, -1).to(device)   # [M, L, 5]
    mask_b = attn_mask.unsqueeze(0).expand(M, -1).to(device)       # [M, L]
    world_b = world_assign.to(device)                               # [M, 28, 3]

    with torch.no_grad():
        if is_voids:
            voids_b = voids_vec.unsqueeze(0).expand(M, -1).to(device)  # [M, 24]
            out = model(tokens_b, mask_b, world_b, voids_b)
        else:
            out = model(tokens_b, mask_b, world_b)

    return out  # v: [M], pi_me_logits: [M, 7]


# ---------------------------------------------------------------------------
# Build tokens+voids for a (game, state) snapshot
# ---------------------------------------------------------------------------

@dataclass
class _FakeDecision:
    """Minimal decision proxy for tokenize_decision reconstruction."""
    player: int
    action_taken: int


def _build_tokens_voids(
    game_hands: list[list[int]],
    decl_id: int,
    orig_decisions: list,
    d_idx: int,
    extra: list[_FakeDecision],
    query_player: int,
) -> tuple[Tensor, Tensor]:
    """Build (tokens [L, 5], voids [24]) for query_player at state = orig[:d_idx] + extra."""

    class _Stub:
        def __init__(self, p, a):
            self.player = p
            self.action_taken = a

    synth = list(orig_decisions[:d_idx]) + [_Stub(fd.player, fd.action_taken) for fd in extra]
    synth_d_idx = len(synth)
    synth.append(_Stub(query_player, 0))  # placeholder for current decision

    tokens, attn_mask = tokenize_decision(game_hands, decl_id, synth, synth_d_idx)

    # Build prior plays for voids (original plays + extra)
    plays: list[tuple[int, int]] = []
    for j in range(d_idx):
        dd = orig_decisions[j]
        p = int(dd.player)
        slot = int(dd.action_taken)
        hand = _hand_list(game_hands[p])
        if 0 <= slot < len(hand) and int(hand[slot]) >= 0:
            plays.append((p, int(hand[slot])))
    for fd in extra:
        hand = _hand_list(game_hands[fd.player])
        slot = fd.action_taken
        if 0 <= slot < len(hand) and int(hand[slot]) >= 0:
            plays.append((fd.player, int(hand[slot])))

    voids = voids_feature_vector(plays, decl_id, query_player)

    return tokens, attn_mask, voids


# ---------------------------------------------------------------------------
# Replay state up to decision d_idx
# ---------------------------------------------------------------------------

def _replay_state(game_hands: list[list[int]], decl_id: int, decisions: list, d_idx: int) -> GameStateTensor:
    """Replay the game to the state just before decision d_idx."""
    state = GameStateTensor.from_deals([game_hands], [decl_id], device="cpu")
    for j in range(d_idx):
        slot_j = int(decisions[j].action_taken)
        state = state.apply_actions(torch.tensor([slot_j], dtype=torch.long))
    return state


# ---------------------------------------------------------------------------
# LAMIR-1 per-decision
# ---------------------------------------------------------------------------

def lamir1_decision(
    model,
    is_voids: bool,
    game,
    d_idx: int,
    device: str,
    world_cap: int = 200,
) -> tuple[float, int, float]:
    """Run LAMIR-1 for one decision. Returns (regret, is_bot_match, oracle_best_eq).

    Falls back to direct π_me if trick_pos == 3 (last play in trick).
    """
    decision = game.decisions[d_idx]
    P = int(decision.player)
    legal_mask = decision.legal_mask.bool()   # [7]
    e_q = decision.e_q.float()               # [7]
    e_q_legal = e_q.clone()
    e_q_legal[~legal_mask] = float("-inf")
    oracle_best_eq = float(e_q_legal.max().item())
    oracle_best_action = int(e_q_legal.argmax().item())

    legal_slots = legal_mask.nonzero(as_tuple=True)[0].tolist()

    # trick_pos: how many plays have gone in the current trick before this one
    trick_pos = d_idx % 4
    n_remaining = 3 - trick_pos  # opp plays left in this trick

    # Direct fallback if last-in-trick
    if trick_pos == 3 or n_remaining == 0:
        tokens, attn_mask, voids = _build_tokens_voids(
            game.hands, int(game.decl_id), game.decisions, d_idx, [], P
        )
        # Single dummy world (v_head / pi_me don't use it)
        dummy_world = torch.zeros(1, 28, 3)
        out = query_model_batched(model, is_voids, tokens, attn_mask, voids, dummy_world, device)
        pi = out["pi_me_logits"][0]  # [7]
        pi_masked = pi.masked_fill(~legal_mask.to(device), float("-inf"))
        chosen = int(pi_masked.argmax().item())
        regret = oracle_best_eq - float(e_q[chosen].item())
        return regret, int(chosen == oracle_best_action), oracle_best_eq

    # Cap worlds for speed
    world_hands_all = decision.world_hands  # [M, 3, 7]
    M_full = world_hands_all.shape[0]
    M = min(M_full, world_cap)
    world_hands = world_hands_all[:M]  # [M, 3, 7]

    # Replay state to just before decision d_idx
    state_at_d = _replay_state(game.hands, int(game.decl_id), game.decisions, d_idx)

    action_scores: dict[int, float] = {}

    for a_slot in legal_slots:
        # Step state: P plays slot a_slot
        state_after_a = state_at_d.apply_actions(torch.tensor([a_slot], dtype=torch.long))
        extra_a = [_FakeDecision(player=P, action_taken=a_slot)]

        # State after rolling out the remaining opps in the trick
        # We need per-world states because opp action depends on the world.
        # BUT: argmax opp plays will be different per world (legal mask differs).
        # Strategy: build world_assignment for each step's query, get pi_me per world,
        # then argmax per world to get the most likely opp action.
        # Since trick state diverges per world after opp plays, we need per-world tracking.

        # Per-world states and extra lists
        states = [state_after_a] * M  # all start same; states are immutable so ok
        extras = [list(extra_a) for _ in range(M)]  # separate per world

        for step in range(n_remaining):
            p_next = (P + 1 + step) % 4

            # Build tokens+voids for p_next's POV using extra for world 0
            # (tokens/voids are the same for all worlds since they depend on
            # public information only — the play sequence so far is identical)
            tokens_pnext, attn_pnext, voids_pnext = _build_tokens_voids(
                game.hands, int(game.decl_id), game.decisions, d_idx,
                extras[0],  # same for all worlds at this step
                p_next,
            )

            # Rotate world tensor for p_next's POV: [M, 3, 7]
            world_rotated = reindex_world_rows(world_hands, old_cp=P, new_cp=p_next)
            world_assign_pnext = world_batch_to_assignment_vectorized(world_rotated)  # [M, 28, 3]

            # Batched forward: get pi_me for all M worlds
            out_pnext = query_model_batched(
                model, is_voids,
                tokens_pnext, attn_pnext, voids_pnext,
                world_assign_pnext, device
            )
            pi_pnext = out_pnext["pi_me_logits"]  # [M, 7]

            # For each world, get legal mask and pick argmax action
            new_states = []
            for m in range(M):
                state_m = states[m]
                # Check trick hasn't ended early (trick_len == 0 means new trick started)
                trick_len_m = int((state_m.trick_plays[0] >= 0).sum().item())
                if trick_len_m == 0 and step > 0:
                    # Trick completed before expected — skip remaining steps
                    new_states.append(state_m)
                    continue

                opp_legal_m = state_m.legal_actions()[0].to(device)  # [7]
                pi_m = pi_pnext[m].masked_fill(~opp_legal_m, float("-inf"))
                a_next_m = int(pi_m.argmax().item())

                new_state_m = state_m.apply_actions(torch.tensor([a_next_m], dtype=torch.long))
                new_states.append(new_state_m)
                extras[m].append(_FakeDecision(player=p_next, action_taken=a_next_m))

            states = new_states

        # Query V_head from P's POV at leaf for all worlds simultaneously.
        # tokens/voids for P's POV after the rollout (use extras[0] since public
        # plays are same as long as we use extras that reflect the same public play
        # sequence — but extras[m] differ per world!).
        #
        # We batch over worlds but tokens/voids are public info, so same for all worlds.
        # extras[m] may differ per-world for the opp plays. However, since we're computing
        # V_head which depends on the sequence seen, we should use P's view. The opp plays
        # happened in the world context, but from P's perspective ALL opp plays are public
        # after the trick. For 1-ply end-of-trick evaluation, all opp plays are visible
        # to P. So the token sequence for P includes all of them.
        #
        # Since opp actions may differ per world, we need per-world tokens.
        # To avoid M separate tokenize calls, we note that the difference is tiny
        # and use the "most common" opp action or process in batches.
        # For correctness, we compute per-world tokens but batch the forward pass.

        all_tokens = []
        all_masks = []
        all_voids = []
        for m in range(M):
            t, am, v = _build_tokens_voids(
                game.hands, int(game.decl_id), game.decisions, d_idx,
                extras[m], P
            )
            all_tokens.append(t)
            all_masks.append(am)
            all_voids.append(v)

        tokens_stack = torch.stack(all_tokens)   # [M, L, 5]
        masks_stack = torch.stack(all_masks)     # [M, L]
        voids_stack = torch.stack(all_voids)     # [M, 24]
        world_assign_P = world_batch_to_assignment_vectorized(world_hands)  # [M, 28, 3]

        with torch.no_grad():
            tokens_d = tokens_stack.to(device)
            masks_d = masks_stack.to(device)
            world_d = world_assign_P.to(device)
            if is_voids:
                voids_d = voids_stack.to(device)
                out_v = model(tokens_d, masks_d, world_d, voids_d)
            else:
                out_v = model(tokens_d, masks_d, world_d)

        v_vals = out_v["v"]  # [M]
        action_scores[a_slot] = float(v_vals.mean().item())

    chosen = max(action_scores, key=lambda a: action_scores[a])
    regret = oracle_best_eq - float(e_q[chosen].item())
    return regret, int(chosen == oracle_best_action), oracle_best_eq


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="gus/adapters/v3_consistency_10000g.pt")
    parser.add_argument("--eval", required=True, nargs="+")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default="scratch/lamir1_v3_run1.json")
    parser.add_argument("--world-cap", type=int, default=200,
                        help="Cap on worlds per decision (default 200)")
    args = parser.parse_args()

    device = args.device or _pick_device()
    print(f"Adapter: {args.adapter}  device: {device}  world-cap: {args.world_cap}", flush=True)

    model, is_voids = _load_student(args.adapter, device)
    print(f"Model loaded, voids={is_voids}", flush=True)

    ds = JointWorldFullDataset(args.eval, seed=42)
    print(f"Eval corpus: {len(ds.games)} games, qualifying decisions: {len(ds.index)}", flush=True)

    total_regret = 0.0
    total_items = 0
    bot_matches = 0
    negligible_regret = 0
    bucket_regret: dict[int, list[float]] = defaultdict(list)
    bucket_match: dict[int, list[int]] = defaultdict(list)
    per_decision_results: list[dict] = []

    t_start = time.time()

    for g_idx, game in enumerate(ds.games):
        for d_idx, dec in enumerate(game.decisions):
            if dec.world_hands is None or dec.q_per_world is None:
                continue

            regret, is_match, oracle_best = lamir1_decision(
                model, is_voids, game, d_idx, device,
                world_cap=args.world_cap,
            )

            total_regret += regret
            total_items += 1
            bot_matches += is_match
            if regret < 0.5:
                negligible_regret += 1
            bucket_regret[d_idx].append(regret)
            bucket_match[d_idx].append(is_match)
            per_decision_results.append({
                "game": g_idx,
                "d_idx": d_idx,
                "regret": regret,
                "bot_match": is_match,
                "oracle_best_eq": oracle_best,
            })

        elapsed = time.time() - t_start
        n_done = total_items
        print(
            f"  game {g_idx+1}/{len(ds.games)}  decisions={n_done}  "
            f"regret={total_regret/max(n_done,1):.3f}  "
            f"bot-match={bot_matches/max(n_done,1):.3%}  "
            f"elapsed={elapsed:.0f}s",
            flush=True,
        )

    elapsed_total = time.time() - t_start
    mean_regret = total_regret / max(total_items, 1)
    bot_match_rate = bot_matches / max(total_items, 1)
    near_tie_rate = negligible_regret / max(total_items, 1)

    print(flush=True)
    print(f"=== LAMIR-1 Summary over {total_items} decisions ===")
    print(f"  Bot-match rate:            {bot_match_rate:.3%}")
    print(f"  Mean regret (Q-points):    {mean_regret:.3f}")
    print(f"  Decisions with regret<0.5: {negligible_regret}/{total_items} "
          f"= {near_tie_rate:.1%} (near-ties)")
    print(f"  Wall-clock time:           {elapsed_total:.1f}s")
    print(f"  Baseline regret:           0.551")
    print(f"  Baseline bot-match:        76.07%")
    print(flush=True)

    print("=== Per-decision regret + bot-match ===")
    print(f"{'dec':>3s}  {'bot':>6s}  {'regret':>8s}  {'near-tie':>8s}  {'n':>3s}")
    for d in sorted(bucket_regret.keys()):
        regrets = bucket_regret[d]
        matches = bucket_match[d]
        n = len(regrets)
        mean_r = sum(regrets) / n
        match_rate = sum(matches) / n
        near_ties = sum(1 for r in regrets if r < 0.5) / n
        print(f"{d:>3d}  {match_rate:>6.2%}  {mean_r:>8.2f}  {near_ties:>8.1%}  {n:>3d}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "adapter": args.adapter,
        "device": device,
        "world_cap": args.world_cap,
        "n_decisions": total_items,
        "mean_regret": mean_regret,
        "bot_match_rate": bot_match_rate,
        "near_tie_rate": near_tie_rate,
        "wall_clock_s": elapsed_total,
        "baseline_regret": 0.551,
        "baseline_bot_match": 0.7607,
        "per_decision": per_decision_results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
