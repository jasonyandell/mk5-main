"""Arena evaluation: Gus student vs E[Q] bot at the game level.

For each (seed, decl_id) pair, play a full 28-decision game with the student
at one seat and the E[Q] bot at the other three. Record final team scores
and who made/set the contract. Aggregate across all games.

Also run an all-E[Q]-bot baseline on the same games for direct comparison.

This reuses the GPU E[Q] pipeline as the game engine (GameStateTensor,
legal-action logic, trick resolution) and replaces `select_actions` for
the student's seat with a student-inference call.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch import Tensor

from forge.eq.game_tensor import GameStateTensor
from forge.eq.generate.actions import select_actions
from forge.eq.generate.deals import build_hypothetical_deals
from forge.eq.generate.eq_compute import compute_eq_pdf
from forge.eq.generate.model import query_model
from forge.eq.generate.sampling import sample_worlds_batched
from forge.eq.generate.tokenization import tokenize_batched
from forge.eq.generate.types import AdaptiveConfig
from forge.eq.generate.adaptive import sample_until_convergence
from forge.eq.oracle import Stage1Oracle
from forge.eq.sampling_mrv_gpu import WorldSamplerMRV
from forge.eq.tokenize_gpu import GPUTokenizer
from forge.oracle.rng import deal_from_seed
from forge.oracle.tables import score_trick

from gus.model.features import reconstruct_prior_plays
from gus.model.student import StudentTransformerFull, StudentTransformerFullVoids
from gus.model.tokenize import tokenize_decision
from gus.model.voids import voids_feature_vector


# -----------------------------------------------------------------------------
# Student loading — matches eval_regret pattern.
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Fake-decisions helper — we build one per played action to reuse the
# gus/model/features.reconstruct_prior_plays helper, which expects
# decisions[j].player / decisions[j].action_taken (slot index into the
# initial hand).
# -----------------------------------------------------------------------------


@dataclass
class _FakeDecision:
    player: int
    action_taken: int


# -----------------------------------------------------------------------------
# Score computation: walk states.history after the game ends and produce
# (team_0_points, team_1_points).
# -----------------------------------------------------------------------------


def compute_final_scores(states: GameStateTensor) -> Tensor:
    """Return [n_games, 2] int tensor of (team_0, team_1) points.

    Team 0 = P0 & P2, Team 1 = P1 & P3. Total always 42.
    """
    n_games = states.n_games
    scores = torch.zeros(n_games, 2, dtype=torch.int32)

    # states.history: [n_games, 28, 3]  (player, domino, lead_domino)
    # Trick i spans history[:, 4i : 4i+4].
    history = states.history.cpu()

    for g in range(n_games):
        h = history[g]  # [28, 3]
        for t in range(7):
            base = 4 * t
            dominoes = tuple(int(h[base + i, 1].item()) for i in range(4))
            if any(d < 0 for d in dominoes):
                continue
            lead = int(h[base, 1].item())
            from forge.oracle.tables import resolve_trick
            outcome = resolve_trick(lead, dominoes, int(states.decl_ids[g].item()))
            leader = int(h[base, 0].item())
            winner = (leader + outcome.winner_offset) % 4
            scores[g, winner % 2] += outcome.points

    return scores


# -----------------------------------------------------------------------------
# Student action selection for one decision.
# -----------------------------------------------------------------------------


def student_pick_actions(
    student_model,
    is_voids: bool,
    states: GameStateTensor,
    hands: list[list[list[int]]],
    decl_ids: list[int],
    game_decisions: list[list[_FakeDecision]],
    student_mask: Tensor,  # [n_games] bool — True where student should pick
    device: str,
) -> Tensor:
    """Return a [n_games] long tensor of actions, where student_mask is True
    slots contain student picks and other slots are arbitrary (-1)."""
    n_games = states.n_games
    actions = torch.full((n_games,), -1, dtype=torch.long)

    if not student_mask.any():
        return actions

    # Build tokenized inputs for each game where student plays.
    tokens_list: list[Tensor] = []
    mask_list: list[Tensor] = []
    voids_list: list[Tensor] = []
    game_indices: list[int] = []

    legal = states.legal_actions().cpu()  # [n_games, 7]

    for g in range(n_games):
        if not student_mask[g].item():
            continue
        decisions = game_decisions[g]
        d_idx = len(decisions)
        # tokenize_decision reads decisions[d_idx].player to learn the current
        # player. We haven't recorded that decision yet, so append a stub
        # (only `.player` is read at index d_idx; `.action_taken` is only used
        # for indices < d_idx).
        current_player = int(states.current_player[g].item())
        decisions_with_stub = decisions + [_FakeDecision(player=current_player, action_taken=0)]
        tokens, attn_mask = tokenize_decision(
            hands[g], int(decl_ids[g]), decisions_with_stub, d_idx
        )
        prior_plays = reconstruct_prior_plays(hands[g], decisions_with_stub, d_idx)
        voids = voids_feature_vector(prior_plays, int(decl_ids[g]), current_player)

        tokens_list.append(tokens)
        mask_list.append(attn_mask)
        voids_list.append(voids)
        game_indices.append(g)

    batch_tokens = torch.stack(tokens_list).to(device)
    batch_masks = torch.stack(mask_list).to(device)
    batch_voids = torch.stack(voids_list).to(device)
    B = batch_tokens.shape[0]

    # Dummy world_assignment (student doesn't need it for policy head).
    world_assignment = torch.zeros(B, 28, 3, dtype=torch.float32, device=device)

    with torch.no_grad():
        if is_voids:
            out = student_model(batch_tokens, batch_masks, world_assignment, batch_voids)
        else:
            out = student_model(batch_tokens, batch_masks, world_assignment)

    pi_logits = out["pi_me_logits"]  # [B, 7]

    # Apply legal mask per game.
    for b, g in enumerate(game_indices):
        legal_g = legal[g].to(device)
        masked = pi_logits[b].masked_fill(~legal_g, float("-inf"))
        actions[g] = int(masked.argmax().item())

    return actions


# -----------------------------------------------------------------------------
# Main arena game-loop — mirrors generate_eq_games_gpu but overrides the
# action for the student's seat.
# -----------------------------------------------------------------------------


def run_arena_games(
    oracle_model,
    student_model,
    is_voids: bool,
    hands: list[list[list[int]]],
    decl_ids: list[int],
    student_seat: int,
    device: str,
    adaptive_config: AdaptiveConfig,
) -> tuple[Tensor, list[list[_FakeDecision]]]:
    """Play n_games in parallel with the student at seat `student_seat`.

    Returns (final_scores [n_games, 2], per-game decisions-list).
    """
    n_games = len(hands)
    states = GameStateTensor.from_deals(hands, decl_ids, device)

    sampler = WorldSamplerMRV(
        max_games=n_games, max_samples=adaptive_config.batch_size, device=device
    )
    tokenizer = GPUTokenizer(
        max_batch=n_games * adaptive_config.batch_size, device=device
    )

    game_decisions: list[list[_FakeDecision]] = [[] for _ in range(n_games)]

    decision_idx = 0
    while states.active_games().any():
        current_players = states.current_player  # [n_games]
        student_mask = (current_players == student_seat) & states.active_games()
        bot_mask = (~student_mask) & states.active_games()

        n_student = int(student_mask.sum().item())
        n_bot = int(bot_mask.sum().item())

        # --- Student picks (if any) ---
        student_actions = torch.full((n_games,), -1, dtype=torch.long)
        if n_student > 0:
            student_actions = student_pick_actions(
                student_model, is_voids, states, hands, decl_ids,
                game_decisions, student_mask.cpu(), device,
            )

        # --- Bot picks: run oracle E[Q] *only for the bot-seat games* ---
        # For efficiency, we run the oracle on ALL games in this batch step
        # but just discard the results for student-seat games. (Running on
        # a subset would require slicing the GameStateTensor, which it
        # doesn't natively support.)
        bot_actions = torch.full((n_games,), -1, dtype=torch.long)
        if n_bot > 0 or n_student > 0:  # always need oracle pdf to drive bot games
            e_q, e_q_var, e_q_pdf, _diag, _n, _conv, _wh, _qpw = sample_until_convergence(
                states=states,
                sampler=sampler,
                tokenizer=tokenizer,
                model=oracle_model,
                adaptive_config=adaptive_config,
                device=device,
                decision_idx=decision_idx,
                seeds=None,
                use_cuda_graph=False,
                save_joint_worlds=False,
            )
            if e_q.device != states.hands.device:
                e_q = e_q.to(states.hands.device)
                e_q_pdf = e_q_pdf.to(states.hands.device)

            acts, _ = select_actions(states, e_q, e_q_pdf, greedy=True)
            bot_actions = acts.cpu()

        # Merge: student's choice where student_mask, else bot.
        actions = torch.where(student_mask.cpu(), student_actions, bot_actions)
        # Clamp: inactive games get 0 (doesn't matter, they've no hand left).
        actions = torch.where(actions >= 0, actions, torch.zeros_like(actions))

        # Record decisions (player + slot index) for each active game.
        for g in range(n_games):
            if not states.active_games()[g].item():
                continue
            game_decisions[g].append(_FakeDecision(
                player=int(current_players[g].item()),
                action_taken=int(actions[g].item()),
            ))

        # Apply actions.
        states = states.apply_actions(actions.to(device))
        decision_idx += 1

    scores = compute_final_scores(states)
    return scores, game_decisions


# -----------------------------------------------------------------------------
# All-bot baseline (reuses the standard pipeline).
# -----------------------------------------------------------------------------


def run_bot_baseline(
    oracle_model,
    hands: list[list[list[int]]],
    decl_ids: list[int],
    device: str,
    adaptive_config: AdaptiveConfig,
) -> Tensor:
    """Run the standard all-bot pipeline on the same hands, return scores."""
    from forge.eq.generate.pipeline import generate_eq_games_gpu

    n_games = len(hands)
    results = generate_eq_games_gpu(
        model=oracle_model,
        hands=hands,
        decl_ids=decl_ids,
        device=device,
        adaptive_config=adaptive_config,
        save_joint_worlds=False,
    )
    # Reconstruct final states and score them. Easiest: replay actions through
    # a fresh GameStateTensor.
    states = GameStateTensor.from_deals(hands, decl_ids, device)
    max_decisions = max(len(r.decisions) for r in results)
    for d_idx in range(max_decisions):
        actions = torch.zeros(n_games, dtype=torch.long)
        for g, r in enumerate(results):
            if d_idx < len(r.decisions):
                actions[g] = int(r.decisions[d_idx].action_taken)
        states = states.apply_actions(actions.to(device))
    return compute_final_scores(states)


# -----------------------------------------------------------------------------
# Aggregation.
# -----------------------------------------------------------------------------


def aggregate_stats(
    student_scores: Tensor,      # [G, 2] — team 0, team 1 points
    bot_scores: Tensor,          # [G, 2]
    student_seat: int,
    bidders: list[int],          # per-game bidder seat (all 0 by default)
) -> dict:
    """Slice into (as bidder) vs (as defender) based on bidder seat."""
    G = student_scores.shape[0]
    student_team = student_seat % 2

    stats = {
        "games": G,
        "student_seat": student_seat,
        "student_team": student_team,
    }

    # Bidder team is always bidder % 2. Here bidders are all 0 in this arena,
    # but we compute it in general.
    bidder_teams = [b % 2 for b in bidders]

    # Student on bidder team?
    student_on_bidder = [bt == student_team for bt in bidder_teams]
    n_as_bidder = sum(student_on_bidder)
    n_as_defender = G - n_as_bidder

    # Points per hand by role.
    bidder_points_student = []
    defender_points_student = []
    bidder_points_bot = []
    defender_points_bot = []
    student_made = 0
    student_set = 0
    bot_made = 0
    bot_set = 0

    for g in range(G):
        bt = bidder_teams[g]
        s_student_bidder = student_scores[g, bt].item()
        s_student_def = student_scores[g, 1 - bt].item()
        s_bot_bidder = bot_scores[g, bt].item()
        s_bot_def = bot_scores[g, 1 - bt].item()

        if student_on_bidder[g]:
            # Student's team was bidder.
            bidder_points_student.append(s_student_bidder)
            bidder_points_bot.append(s_bot_bidder)
            # "Made contract" = bidder team got >= 30 points.
            if s_student_bidder >= 30:
                student_made += 1
            if s_bot_bidder >= 30:
                bot_made += 1
        else:
            # Student's team was defender.
            defender_points_student.append(s_student_def)
            defender_points_bot.append(s_bot_def)
            # "Set bidder" = defender held bidder below 30 (bidder got < 30).
            if s_bot_bidder < 30:  # bidder's score in the student game
                # Actually we want: in the student game, the bidder team's
                # score was kept < 30, i.e., s_student_bidder < 30 (since
                # student_scores[g, bt] is bidder team's points in the
                # student arena).
                pass
            if s_student_bidder < 30:
                student_set += 1
            if s_bot_bidder < 30:
                bot_set += 1

    def _mean(xs):
        return (sum(xs) / len(xs)) if xs else 0.0

    stats.update({
        "n_as_bidder": n_as_bidder,
        "n_as_defender": n_as_defender,
        "student_made": student_made,
        "student_set": student_set,
        "bot_made": bot_made,
        "bot_set": bot_set,
        "avg_bidder_points_student": _mean(bidder_points_student),
        "avg_bidder_points_bot": _mean(bidder_points_bot),
        "avg_defender_points_student": _mean(defender_points_student),
        "avg_defender_points_bot": _mean(defender_points_bot),
    })

    # Overall wins: whichever team has more points per hand wins.
    s_wins = 0
    b_wins = 0
    for g in range(G):
        st = student_team
        if student_scores[g, st].item() > student_scores[g, 1 - st].item():
            s_wins += 1
        if bot_scores[g, st].item() > bot_scores[g, 1 - st].item():
            b_wins += 1
    stats["student_team_wins_in_student_arena"] = s_wins
    stats["student_team_wins_in_bot_arena"] = b_wins

    return stats


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------


def print_summary(stats: dict) -> None:
    G = stats["games"]
    print()
    print(f"=== Arena summary ===")
    print(f"Games: {G}  (student at seat {stats['student_seat']}, team {stats['student_team']})")
    print()
    print(f"                      {'student':>12s}   {'all-bot baseline':>16s}")
    if stats["n_as_bidder"] > 0:
        print(f"  as bidder ({stats['n_as_bidder']:>3d}):   "
              f"{stats['student_made']:>2d}/{stats['n_as_bidder']:<2d} made  "
              f"{stats['bot_made']:>2d}/{stats['n_as_bidder']:<2d} made")
        print(f"  avg bidder points:   {stats['avg_bidder_points_student']:>12.2f}   "
              f"{stats['avg_bidder_points_bot']:>16.2f}")
    if stats["n_as_defender"] > 0:
        print(f"  as defender ({stats['n_as_defender']:>2d}):  "
              f"{stats['student_set']:>2d}/{stats['n_as_defender']:<2d} set  "
              f"{stats['bot_set']:>2d}/{stats['n_as_defender']:<2d} set")
        print(f"  avg defender points: {stats['avg_defender_points_student']:>12.2f}   "
              f"{stats['avg_defender_points_bot']:>16.2f}")
    print()
    print(f"  team hands won:     {stats['student_team_wins_in_student_arena']:>3d} / {G}   "
          f"{stats['student_team_wins_in_bot_arena']:>3d} / {G}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to student checkpoint")
    parser.add_argument("--student-seat", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs=2, default=[900020, 900050],
                        help="[start, end) seed range (end exclusive)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--oracle-ckpt", type=str,
                        default="forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt")
    parser.add_argument("--out", type=str, default=None,
                        help="Output .pt path for aggregate stats")
    parser.add_argument("--jsonl", type=str, default=None,
                        help="Output JSONL path for per-game records")
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sem-threshold", type=float, default=1.0)
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip the all-bot baseline run (student-only arena).")
    args = parser.parse_args()

    device = args.device

    # Output defaults derived from adapter stem.
    adapter_stem = Path(args.adapter).stem
    if args.out is None:
        args.out = f"gus/adapters/arena_{adapter_stem}.pt"
    if args.jsonl is None:
        args.jsonl = f"scratch/arena_{adapter_stem}.jsonl"

    print(f"Arena: adapter={args.adapter}", flush=True)
    print(f"  seeds=[{args.seeds[0]}, {args.seeds[1]})   seat={args.student_seat}   device={device}", flush=True)

    # Load models.
    print("Loading student...", flush=True)
    student_model, is_voids = _load_student(args.adapter, device)
    print(f"  student is_voids={is_voids}", flush=True)

    print(f"Loading oracle from {args.oracle_ckpt}...", flush=True)
    oracle = Stage1Oracle(args.oracle_ckpt, device=device, compile=False)

    # Build the game list: cross-product of seeds x 10 declarations.
    seeds = list(range(args.seeds[0], args.seeds[1]))
    hands: list[list[list[int]]] = []
    decl_ids: list[int] = []
    game_meta: list[dict] = []
    for seed in seeds:
        deal = deal_from_seed(seed)
        for decl in range(10):
            hands.append(deal)
            decl_ids.append(decl)
            game_meta.append({"seed": seed, "decl_id": decl})

    n_games = len(hands)
    print(f"Will play {n_games} games ({len(seeds)} seeds x 10 declarations)", flush=True)

    adaptive_config = AdaptiveConfig(
        enabled=True,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        sem_threshold=args.sem_threshold,
    )

    # --- Student arena ---
    t0 = time.perf_counter()
    print("Running student arena...", flush=True)
    student_scores, game_decisions = run_arena_games(
        oracle_model=oracle.model,
        student_model=student_model,
        is_voids=is_voids,
        hands=hands,
        decl_ids=decl_ids,
        student_seat=args.student_seat,
        device=device,
        adaptive_config=adaptive_config,
    )
    t_student = time.perf_counter() - t0
    print(f"  student arena: {t_student:.1f}s ({n_games / max(t_student, 1e-6):.2f} games/s)", flush=True)

    # --- All-bot baseline ---
    if args.skip_baseline:
        bot_scores = torch.zeros(n_games, 2, dtype=torch.int32)
    else:
        t0 = time.perf_counter()
        print("Running all-bot baseline...", flush=True)
        bot_scores = run_bot_baseline(
            oracle_model=oracle.model,
            hands=hands,
            decl_ids=decl_ids,
            device=device,
            adaptive_config=adaptive_config,
        )
        t_bot = time.perf_counter() - t0
        print(f"  all-bot baseline: {t_bot:.1f}s", flush=True)

    # --- Aggregate ---
    bidders = [0] * n_games  # all games use bidder = P0 in this arena
    stats = aggregate_stats(student_scores, bot_scores, args.student_seat, bidders)
    stats["adapter"] = args.adapter
    stats["seeds"] = seeds
    stats["n_decls"] = 10

    print_summary(stats)

    # Save stats + scores.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "stats": stats,
        "student_scores": student_scores,
        "bot_scores": bot_scores,
        "game_meta": game_meta,
        "adapter": args.adapter,
        "student_seat": args.student_seat,
        "seeds": seeds,
        "adaptive_config": adaptive_config.__dict__,
    }, args.out)
    print(f"\nSaved stats to {args.out}", flush=True)

    # Save per-game JSONL log.
    Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.jsonl, "w") as f:
        for g in range(n_games):
            rec = {
                "seed": game_meta[g]["seed"],
                "decl_id": game_meta[g]["decl_id"],
                "student_team_points_student_arena": int(student_scores[g, args.student_seat % 2].item()),
                "opp_team_points_student_arena": int(student_scores[g, 1 - args.student_seat % 2].item()),
                "student_team_points_bot_arena": int(bot_scores[g, args.student_seat % 2].item()),
                "opp_team_points_bot_arena": int(bot_scores[g, 1 - args.student_seat % 2].item()),
                "actions": [d.action_taken for d in game_decisions[g]],
                "players": [d.player for d in game_decisions[g]],
            }
            f.write(json.dumps(rec) + "\n")
    print(f"Saved per-game JSONL to {args.jsonl}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
