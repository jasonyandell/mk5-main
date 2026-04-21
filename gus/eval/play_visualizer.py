"""Trick-by-trick play visualizer for a Gus student adapter.

Loads an adapter and an eval corpus, picks one game, and walks the 28
decisions one by one. For each decision where the current player is the
seat that the student would control (every seat, since same net runs in
every seat via weight-sharing), emits the student's chosen action next
to the oracle argmax, with regret color-coding.

Output: markdown, readable as a story with coffee.

Usage:
    python -u -m gus.eval.play_visualizer \\
        --adapter gus/adapters/v2_voids_big_3000g.pt \\
        --eval gus/data/corpus_eval_20.pt \\
        --game-idx 0 \\
        --out scratch/game_viz_game0.md \\
        --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch import Tensor

from gus.model.dataset_seq_world import JointWorldFullDataset
from gus.model.features import extract_belief_target, reconstruct_prior_plays
from gus.model.student import StudentTransformerFull, StudentTransformerFullVoids
from gus.model.tokenize import tokenize_decision
from gus.model.voids import is_trump, led_suit, voids_feature_vector

# Canonical domino names — copied locally (don't import from scratch/).
# 0=(0,0), 1=(1,0), 2=(1,1), ..., 27=(6,6)
DOMINO_NAMES: list[str] = [f"{a}-{b}" for a in range(7) for b in range(a + 1)]

# 5-point dominoes: pip sum == 5  →  (4,1), (3,2)  →  ids 11, 13
# 10-point dominoes: pip sum == 10 →  (6,4), (5,5) →  ids 22, 20
COUNT_VALUES: dict[int, int] = {}
for _i, (_a, _b) in enumerate([(a, b) for a in range(7) for b in range(a + 1)]):
    _s = _a + _b
    if _s == 5:
        COUNT_VALUES[_i] = 5
    elif _s == 10:
        COUNT_VALUES[_i] = 10

SEAT_ROLES = {0: "student team", 1: "opponent", 2: "student team (partner)", 3: "opponent"}

DECL_NAMES = {
    0: "blanks (0s)",
    1: "ones",
    2: "twos",
    3: "threes",
    4: "fours",
    5: "fives",
    6: "sixes",
    7: "doubles",
    8: "follow-me variant 8",
    9: "follow-me variant 9",
}


def domino_name(d: int) -> str:
    if 0 <= d < len(DOMINO_NAMES):
        return DOMINO_NAMES[d]
    return f"d{d}"


def pip_sum(d: int) -> int:
    a, b = d_pips(d)
    return a + b


def d_pips(d: int) -> tuple[int, int]:
    """Canonical (high, low) for domino id d."""
    idx = 0
    for a in range(7):
        for b in range(a + 1):
            if idx == d:
                return a, b
            idx += 1
    return 0, 0


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


def slot_to_domino(slot: int, player: int, hands: list[list[int]]) -> int:
    """Map action slot (0..6) to the domino id in that player's initial hand."""
    hand = [int(x) for x in hands[player]]
    if 0 <= slot < len(hand):
        return int(hand[slot])
    return -1


def _regret_badge(regret: float) -> str:
    """Markdown-safe symbol for a regret magnitude."""
    if regret < 0.5:
        return "✓"
    if regret < 2.0:
        return "▽"
    if regret < 8.0:
        return "⚠"
    return "🔥"


def _role_for_play(
    play_idx: int,
    leader_dom: int,
    this_dom: int,
    decl_id: int,
) -> str:
    """Short tag for a play's role within its trick: leader / follow-trump /
    follow / off-suit (void signal)."""
    if play_idx == 0:
        return "lead"
    led = led_suit(leader_dom, decl_id)
    # Did we follow the led suit?
    a, b = d_pips(this_dom)
    is_t_this = is_trump(this_dom, decl_id)
    if led == decl_id:
        # Led trump — following is also trump
        return "follow (trump)" if is_t_this else "OFF (void in trump)"
    # Non-trump led
    if not is_t_this and (a == led or b == led):
        return "follow"
    if is_t_this:
        return "trump-in"
    return "OFF (void)"


def _trick_rank(d: int, led: int, decl_id: int) -> tuple[int, int, int]:
    """Quick ranking key. Trump always beats non-trump; within trump, higher
    pip sum then higher-pip wins; within non-trump, must match led suit
    and then highest pip (of the led suit) wins. Matches standard 42 well
    enough for trick-winner display; forge/eq/viewer has the canonical
    version but we only need a simple comparator here.
    """
    a, b = d_pips(d)
    is_t = is_trump(d, decl_id)
    follows = is_t if led == decl_id else ((a == led or b == led) and not is_t)
    if not follows:
        return (-1, 0, 0)
    if is_t:
        # Trump: rank by (is_double, high_pip, low_pip) — doubles top within trump
        return (2, 1 if a == b else 0, a + b)
    # Non-trump: rank by (led == high?, pip_sum)
    high_is_led = max(a, b) == led
    # The higher end is the "rank" in suit, with double of led being the top
    if a == b == led:
        return (1, 3, 2 * a)
    return (1, 1 if high_is_led else 0, a + b)


def _determine_trick_winner(
    trick: list[tuple[int, int]], decl_id: int
) -> int:
    """trick: list of (abs_player, domino) in play order. Returns winning
    abs_player."""
    if not trick:
        raise ValueError("empty trick")
    leader_dom = trick[0][1]
    led = led_suit(leader_dom, decl_id)
    best_i = 0
    best_rank = _trick_rank(leader_dom, led, decl_id)
    for i in range(1, len(trick)):
        r = _trick_rank(trick[i][1], led, decl_id)
        if r > best_rank:
            best_i = i
            best_rank = r
    return trick[best_i][0]


def _trick_points(trick: list[tuple[int, int]]) -> int:
    """1 pt per trick + count dominoes in the trick."""
    return 1 + sum(COUNT_VALUES.get(d, 0) for _, d in trick)


def _top_belief(
    belief_logits: Tensor,   # [28, 3]
    belief_mask: Tensor,     # [28]
    current_player: int,
    top_k: int = 3,
) -> list[str]:
    """Return top-k (domino, seat_label, prob) as human strings."""
    probs = torch.softmax(belief_logits, dim=-1)  # [28, 3]
    seat_labels = {
        0: "left-opp",
        1: "partner",
        2: "right-opp",
    }
    # Score each (domino, seat) confidence among unseen dominoes.
    rows: list[tuple[float, int, int]] = []
    for d in range(28):
        if not bool(belief_mask[d].item()):
            continue
        p, seat = probs[d].max(dim=-1)
        rows.append((float(p.item()), d, int(seat.item())))
    rows.sort(reverse=True)
    out: list[str] = []
    for pv, d, s in rows[:top_k]:
        out.append(f"{domino_name(d)} → {seat_labels[s]} (p={pv:.2f})")
    return out


def _fmt_hand(hand: list[int]) -> str:
    """'6-6, 5-4, ...  (pips: 44)'."""
    ids = [int(x) for x in hand if int(x) >= 0]
    ids_sorted = sorted(ids, key=lambda d: -pip_sum(d))
    pretty = ", ".join(domino_name(d) for d in ids_sorted)
    total_pips = sum(pip_sum(d) for d in ids_sorted)
    return f"{pretty}  (pips: {total_pips})"


def visualize_game(
    game,
    seed: int | None,
    game_idx: int,
    model,
    is_voids: bool,
    device: str,
) -> str:
    decl_id = int(game.decl_id)
    hands: list[list[int]] = [list(h) for h in game.hands]
    decisions = game.decisions

    lines: list[str] = []
    seed_str = f"seed {seed}" if seed is not None else f"game-idx {game_idx}"
    lines.append(f"# Game: {seed_str}, declaration {DECL_NAMES.get(decl_id, decl_id)}")
    lines.append("")
    lines.append("**Dealer's deal:**")
    for p in range(4):
        team = "student team" if p % 2 == 0 else "opponents"
        if p == 0:
            tag = "seat 0 (student viewpoint)"
        elif p == 2:
            tag = "seat 2 (student partner)"
        elif p == 1:
            tag = "seat 1 (opponent, left of seat 0)"
        else:
            tag = "seat 3 (opponent, right of seat 0)"
        lines.append(f"- **Player {p}** — {tag} — {_fmt_hand(hands[p])}")
    lines.append("")
    lines.append(f"**Declaration**: `{DECL_NAMES.get(decl_id, decl_id)}`.")
    lines.append("")
    lines.append("Rules of thumb: partnership 0+2 vs 1+3. Seat 0's perspective "
                 "everywhere below — 'partner' = seat 2.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Build per-decision oracle data + student inference, grouped into tricks.
    regrets: list[float] = []
    own_seat_matches = 0
    own_seat_total = 0
    per_decision_rows: list[dict] = []

    # Walk decisions in order, building trick structures on-the-fly.
    for d_idx, dec in enumerate(decisions):
        current_player = int(dec.player)
        prior_plays = reconstruct_prior_plays(hands, decisions, d_idx)

        # Tokenize current state
        tokens, attn_mask = tokenize_decision(hands, decl_id, decisions, d_idx)
        tokens = tokens.unsqueeze(0).to(device)
        attn_mask = attn_mask.unsqueeze(0).to(device)

        # Need a world_assignment for the model call. The dataset picks a
        # random world per epoch; at visualization time we pick world 0 (or
        # skip if the decision has no world tensor — fall back to zeros).
        if dec.world_hands is not None and dec.q_per_world is not None:
            wh = dec.world_hands[0]  # [3, 7]
            world_assignment = torch.zeros(28, 3, dtype=torch.float32)
            for seat in range(3):
                for d in wh[seat].tolist():
                    d = int(d)
                    if 0 <= d < 28:
                        world_assignment[d, seat] = 1.0
        else:
            world_assignment = torch.zeros(28, 3, dtype=torch.float32)
        world_assignment = world_assignment.unsqueeze(0).to(device)

        voids = voids_feature_vector(prior_plays, decl_id, current_player).unsqueeze(0).to(device)

        with torch.no_grad():
            if is_voids:
                out = model(tokens, attn_mask, world_assignment, voids)
            else:
                out = model(tokens, attn_mask, world_assignment)

        # Legal-masked argmax over π_me
        legal_mask = dec.legal_mask.bool().to(device)  # [7]
        pi = out["pi_me_logits"][0].masked_fill(~legal_mask, -1e9)
        pi_probs = torch.softmax(pi, dim=-1)
        student_slot = int(pi.argmax().item())
        student_v = float(out["v"][0].item())
        belief_logits = out["belief_logits"][0].cpu()  # [28, 3]

        # Oracle data
        e_q = dec.e_q.float().cpu()       # [7]
        e_q_legal = e_q.masked_fill(~dec.legal_mask.bool().cpu(), float("-inf"))
        oracle_slot = int(e_q_legal.argmax().item())
        oracle_best = float(e_q_legal.max().item())
        action_slot = int(dec.action_taken)  # what the oracle bot actually played
        student_eq = float(e_q[student_slot].item())
        regret = oracle_best - student_eq

        # Map to dominoes
        student_dom = slot_to_domino(student_slot, current_player, hands)
        oracle_dom = slot_to_domino(oracle_slot, current_player, hands)
        actual_dom = slot_to_domino(action_slot, current_player, hands)

        # Belief mask (for top-3 belief summary)
        _, belief_mask = extract_belief_target(hands, prior_plays, current_player)

        per_decision_rows.append({
            "d_idx": d_idx,
            "player": current_player,
            "student_slot": student_slot,
            "oracle_slot": oracle_slot,
            "action_slot": action_slot,
            "student_dom": student_dom,
            "oracle_dom": oracle_dom,
            "actual_dom": actual_dom,
            "student_v": student_v,
            "student_eq": student_eq,
            "oracle_best": oracle_best,
            "regret": regret,
            "e_q": e_q,
            "pi_probs": pi_probs.cpu(),
            "legal_mask": dec.legal_mask.bool().cpu(),
            "belief_logits": belief_logits,
            "belief_mask": belief_mask,
        })

        regrets.append(regret)
        own_seat_total += 1
        if student_slot == oracle_slot:
            own_seat_matches += 1

    # Now render trick-by-trick. Each trick = 4 consecutive decisions.
    n_tricks = len(decisions) // 4
    us_points = 0
    them_points = 0
    for t in range(n_tricks):
        rows = per_decision_rows[t * 4:(t + 1) * 4]
        # Reconstruct the (player, actual_dom) pairs in order actually played.
        trick_plays = [(r["player"], r["actual_dom"]) for r in rows]
        leader_dom = trick_plays[0][1]
        led = led_suit(leader_dom, decl_id)
        winner = _determine_trick_winner(trick_plays, decl_id)
        points = _trick_points(trick_plays)
        counts_only = points - 1  # 1 pt per trick is implicit
        leader_seat = trick_plays[0][0]

        if winner % 2 == 0:
            us_points += points
        else:
            them_points += points

        lines.append(f"## Trick {t + 1} — led by Player {leader_seat} "
                     f"({domino_name(leader_dom)}, suit: {led})")
        lines.append("")
        lines.append(f"| seat | actual play | student pick | oracle pick | regret | role |")
        lines.append(f"|---|---|---|---|---|---|")
        for r in rows:
            p = r["player"]
            actual = domino_name(r["actual_dom"])
            sp = domino_name(r["student_dom"])
            op = domino_name(r["oracle_dom"])
            match = "✓" if r["student_slot"] == r["oracle_slot"] else "✗"
            badge = _regret_badge(r["regret"])
            role = _role_for_play(
                play_idx=rows.index(r),
                leader_dom=leader_dom,
                this_dom=r["actual_dom"],
                decl_id=decl_id,
            )
            tag = f"P{p}" + (" (partner of seat 0)" if p == 2 else
                             " (seat 0)" if p == 0 else
                             " (opp)")
            lines.append(
                f"| {tag} | **{actual}** | {sp} | {op} {match} | "
                f"{r['regret']:+.2f} {badge} | {role} |"
            )
        lines.append("")
        win_team = "student team" if winner % 2 == 0 else "opponents"
        lines.append(f"**Winner**: Player {winner} ({win_team}). "
                     f"Points this trick: {points} (trick=1, count={counts_only}).")
        lines.append("")
        # Highlight the most interesting student thought — pick one decision,
        # prefer seat 0 (or seat 2) with the most regret, fall back to trick
        # leader.
        own_rows = [r for r in rows if r["player"] in (0, 2)]
        feature = max(own_rows, key=lambda r: r["regret"]) if own_rows else rows[0]

        lines.append(f"### Student brainstate — P{feature['player']} at decision {feature['d_idx']}")
        # Show legal-masked oracle e_q vs student π
        legal_slots = feature["legal_mask"].nonzero(as_tuple=True)[0].tolist()
        # Map legal slots to domino names
        legals_dom = [
            domino_name(slot_to_domino(s, feature["player"], hands))
            for s in legal_slots
        ]
        eq_vec = feature["e_q"]
        pi_vec = feature["pi_probs"]
        lines.append("")
        lines.append(f"- Legal dominoes: {', '.join(legals_dom)}")
        eq_pairs = [f"{domino_name(slot_to_domino(s, feature['player'], hands))}:{eq_vec[s].item():+.2f}"
                    for s in legal_slots]
        pi_pairs = [f"{domino_name(slot_to_domino(s, feature['player'], hands))}:{pi_vec[s].item():.2f}"
                    for s in legal_slots]
        lines.append(f"- Oracle E[Q]: `{'  '.join(eq_pairs)}`")
        lines.append(f"- Student π_me: `{'  '.join(pi_pairs)}`")
        lines.append(
            f"- V_head estimate: {feature['student_v']:+.2f}  "
            f"(oracle E[Q] at student's pick: {feature['student_eq']:+.2f})"
        )
        # Belief top-3 (from *this* player's perspective)
        top_beliefs = _top_belief(
            feature["belief_logits"],
            feature["belief_mask"],
            feature["player"],
            top_k=3,
        )
        if top_beliefs:
            lines.append("- Top-3 belief (unseen→seat):")
            for b in top_beliefs:
                lines.append(f"    - {b}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # --- Summary ---
    total = us_points + them_points
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Final tally: **student team {us_points}** vs opponents {them_points} "
                 f"(total {total} / 42 expected).")
    mean_regret = sum(regrets) / max(1, len(regrets))
    lines.append(f"- Mean regret across all 28 decisions: **{mean_regret:.2f} Q-pts**.")
    # Largest miss
    if regrets:
        worst_idx = max(range(len(regrets)), key=lambda i: regrets[i])
        worst_row = per_decision_rows[worst_idx]
        lines.append(
            f"- Biggest miss: decision {worst_idx} "
            f"(trick {worst_idx // 4 + 1}, play {worst_idx % 4 + 1}), "
            f"regret {regrets[worst_idx]:.2f}. "
            f"Student picked **{domino_name(worst_row['student_dom'])}**, "
            f"oracle preferred **{domino_name(worst_row['oracle_dom'])}** "
            f"(E[Q] {worst_row['oracle_best']:+.2f} vs {worst_row['student_eq']:+.2f})."
        )
    lines.append(f"- Student argmax matched oracle argmax on "
                 f"{own_seat_matches}/{own_seat_total} decisions "
                 f"({own_seat_matches / max(own_seat_total, 1):.0%}).")
    # Regret histogram
    buckets = {"✓ <0.5": 0, "▽ 0.5–2": 0, "⚠ 2–8": 0, "🔥 8+": 0}
    for r in regrets:
        if r < 0.5:
            buckets["✓ <0.5"] += 1
        elif r < 2.0:
            buckets["▽ 0.5–2"] += 1
        elif r < 8.0:
            buckets["⚠ 2–8"] += 1
        else:
            buckets["🔥 8+"] += 1
    hist = ", ".join(f"{k}: {v}" for k, v in buckets.items())
    lines.append(f"- Regret histogram: {hist}")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--game-idx", type=int, default=0)
    parser.add_argument("--out", type=str, default=None, help="default: stdout")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or _pick_device()
    print(f"Adapter: {args.adapter}  device: {device}", file=sys.stderr, flush=True)

    model, is_voids = _load_student(args.adapter, device)
    ds = JointWorldFullDataset(args.eval, seed=42)

    g = args.game_idx
    if g < 0 or g >= len(ds.games):
        print(f"ERROR: game-idx {g} out of range (0..{len(ds.games) - 1}).",
              file=sys.stderr)
        return 1
    game = ds.games[g]
    seed = ds.seeds[g] if g < len(ds.seeds) else None

    md = visualize_game(game, seed, g, model, is_voids, device)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md)
        print(f"Wrote {args.out}  ({len(md)} chars)", file=sys.stderr)
    else:
        sys.stdout.write(md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
