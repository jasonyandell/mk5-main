#!/usr/bin/env python3
"""Generate comprehension Q&A from game records at trick 6.

Five question types, all engine-verified, compact prompt format (~170 tokens).
Each example: game state + one question + ground-truth answer.

Usage:
    # Training data (200 seeds)
    python -u -m lem.rules.generate_comprehension \
        --start-seed 0 --count 200 --output lem/data/comprehension_train.jsonl

    # Eval data (held-out seeds)
    python -u -m lem.rules.generate_comprehension \
        --start-seed 900000 --count 50 --allow-eval-seeds --exhaustive \
        --output lem/data/comprehension_eval.jsonl

    # Quick test
    python -u -m lem.rules.generate_comprehension \
        --start-seed 0 --count 2 --output scratch/comp_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

EVAL_SEED_START = 900_000
EVAL_SEED_END = 909_999
DEFAULT_CHECKPOINT = "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Domino helpers (loaded lazily)
# ---------------------------------------------------------------------------

class _Tables:
    """Lazy-loaded forge engine references."""
    _loaded = False

    DOMINO_HIGH = ()
    DOMINO_LOW = ()
    DOMINO_COUNT_POINTS = ()
    DECL_ID_TO_NAME: dict = {}
    N_DECLS = 0
    NOTRUMP = 9
    DOUBLES_TRUMP = 7
    DOUBLES_SUIT = 8

    @classmethod
    def load(cls):
        if cls._loaded:
            return
        from forge.oracle.tables import (
            DOMINO_COUNT_POINTS, DOMINO_HIGH, DOMINO_LOW,
            can_follow, is_in_called_suit, led_suit_for_lead_domino,
            resolve_trick, trick_rank,
        )
        from forge.oracle.declarations import (
            DECL_ID_TO_NAME, DOUBLES_SUIT, DOUBLES_TRUMP, N_DECLS, NOTRUMP,
            has_trump_power,
        )
        cls.DOMINO_HIGH = DOMINO_HIGH
        cls.DOMINO_LOW = DOMINO_LOW
        cls.DOMINO_COUNT_POINTS = DOMINO_COUNT_POINTS
        cls.DECL_ID_TO_NAME = DECL_ID_TO_NAME
        cls.N_DECLS = N_DECLS
        cls.NOTRUMP = NOTRUMP
        cls.DOUBLES_TRUMP = DOUBLES_TRUMP
        cls.DOUBLES_SUIT = DOUBLES_SUIT
        cls.can_follow = staticmethod(can_follow)
        cls.is_in_called_suit = staticmethod(is_in_called_suit)
        cls.led_suit_for_lead_domino = staticmethod(led_suit_for_lead_domino)
        cls.resolve_trick = staticmethod(resolve_trick)
        cls.trick_rank = staticmethod(trick_rank)
        cls.has_trump_power = staticmethod(has_trump_power)
        cls._loaded = True


def _t() -> type[_Tables]:
    """Lazy-load heavy forge imports."""
    _Tables.load()
    return _Tables


def _dom(dom_id: int) -> str:
    t = _t()
    return f"{t.DOMINO_HIGH[dom_id]}-{t.DOMINO_LOW[dom_id]}"


def _dom_id(h: int, lo: int) -> int:
    """Get domino ID from high-low pips."""
    t = _t()
    for i in range(28):
        if t.DOMINO_HIGH[i] == h and t.DOMINO_LOW[i] == lo:
            return i
    raise ValueError(f"No domino {h}-{lo}")


def _is_trump(dom_id: int, decl_id: int) -> bool:
    t = _t()
    return t.has_trump_power(decl_id) and t.is_in_called_suit(dom_id, decl_id)


def _count_pts(dom_id: int) -> int:
    return _t().DOMINO_COUNT_POINTS[dom_id]


def _decl_name(decl_id: int) -> str:
    return _t().DECL_ID_TO_NAME[decl_id]


def _trump_reason(dom_id: int, decl_id: int) -> str:
    """Human-readable reason why a domino is or isn't trump."""
    t = _t()
    name = _decl_name(decl_id)
    h, lo = t.DOMINO_HIGH[dom_id], t.DOMINO_LOW[dom_id]

    if decl_id == t.NOTRUMP:
        return f"No trump is declared."
    if decl_id == t.DOUBLES_SUIT:
        return f"Doubles are their own suit (no trump)."
    if decl_id == t.DOUBLES_TRUMP:
        if h == lo:
            return f"Doubles are trump. The {h}-{lo} is a double, so it is trump."
        return f"Doubles are trump. The {h}-{lo} is not a double, so it is not trump."
    # Pip suit
    pip = decl_id  # 0-6
    if h == pip or lo == pip:
        return f"{name.capitalize()} are trump. The {h}-{lo} contains a {pip}, so it is trump."
    return f"{name.capitalize()} are trump. The {h}-{lo} does not contain a {pip}, so it is not trump."


def _suit_name(suit_id: int, decl_id: int) -> str:
    """Human name for a suit ID (0-6 = pip suits, 7 = trump)."""
    if suit_id == 7:
        return "trump"
    names = {0: "blanks", 1: "ones", 2: "twos", 3: "threes",
             4: "fours", 5: "fives", 6: "sixes"}
    return names[suit_id]


# ---------------------------------------------------------------------------
# Game state extraction
# ---------------------------------------------------------------------------

@dataclass
class TrickPlay:
    """One play within a trick."""
    player: int
    dom_id: int
    is_forced: bool


@dataclass
class CompletedTrick:
    """A fully played trick."""
    trick_num: int  # 1-based
    plays: list[TrickPlay]
    lead_dom: int
    winner_player: int
    winner_team: int
    points: int


@dataclass
class GamePosition:
    """Everything known at a decision point during trick 6."""
    seed: int
    decl_id: int
    narrator: int
    partner: int
    narrator_team: int
    initial_hand: list[int]  # narrator's starting 7 dominoes
    remaining_hand: list[int]  # narrator's remaining dominoes
    completed_tricks: list[CompletedTrick]  # tricks 1-5
    # Current trick (partial, if narrator is following)
    current_trick_plays: list[TrickPlay]
    current_lead_dom: int | None  # None if narrator leads
    is_leading: bool
    # Scoring
    team_points: list[int]  # [team0, team1]
    # All 4 initial hands (for answer verification — hidden from prompt)
    all_hands: list[list[int]]
    # Decision metadata
    legal_actions: list[int]  # domino IDs the narrator can legally play
    # Count tracking
    count_taken: list[tuple[int, int]]  # (dom_id, winning_team)


def extract_position(record, narrator: int, seed: int) -> GamePosition | None:
    """Extract game state at narrator's trick-6 decision from a GameRecordGPU."""
    t = _t()
    decl_id = record.decl_id
    initial_hands = record.hands
    partner = (narrator + 2) % 4
    narrator_team = narrator % 2

    # Find narrator's 6th decision (trick 6)
    narrator_turns = [
        (i, d) for i, d in enumerate(record.decisions) if d.player == narrator
    ]
    if len(narrator_turns) < 6:
        return None

    dec_idx, decision = narrator_turns[5]

    # Need at least 2 legal moves
    n_legal = int(decision.legal_mask.sum().item())
    if n_legal < 2:
        return None

    # Process completed tricks (1-5) and partial trick 6
    team_points = [0, 0]
    completed_tricks: list[CompletedTrick] = []
    count_taken: list[tuple[int, int]] = []
    narrator_remaining = list(initial_hands[narrator])

    for trick_num in range(7):
        trick_start = trick_num * 4
        trick_decs = record.decisions[trick_start:trick_start + 4]

        if trick_start + 4 <= dec_idx:
            # Fully completed trick before our decision
            plays = []
            dom_ids = []
            for d in trick_decs:
                did = initial_hands[d.player][d.action_taken]
                dom_ids.append(did)
                is_forced = int(d.legal_mask.sum().item()) == 1
                plays.append(TrickPlay(d.player, did, is_forced))
                if did in narrator_remaining:
                    narrator_remaining.remove(did)

            lead_dom = dom_ids[0]
            outcome = t.resolve_trick(lead_dom, tuple(dom_ids), decl_id)
            winner_player = plays[outcome.winner_offset].player
            winner_team = winner_player % 2
            team_points[winner_team] += outcome.points

            for did in dom_ids:
                if t.DOMINO_COUNT_POINTS[did] > 0:
                    count_taken.append((did, winner_team))

            completed_tricks.append(CompletedTrick(
                trick_num=trick_num + 1,
                plays=plays,
                lead_dom=lead_dom,
                winner_player=winner_player,
                winner_team=winner_team,
                points=outcome.points,
            ))

        elif trick_start <= dec_idx < trick_start + 4:
            # Partial trick — plays before narrator's decision
            stop_offset = dec_idx - trick_start
            current_plays = []
            current_lead_dom = None
            for i in range(stop_offset):
                d = trick_decs[i]
                did = initial_hands[d.player][d.action_taken]
                is_forced = int(d.legal_mask.sum().item()) == 1
                current_plays.append(TrickPlay(d.player, did, is_forced))
                if i == 0:
                    current_lead_dom = did
                if did in narrator_remaining:
                    narrator_remaining.remove(did)

            is_leading = stop_offset == 0
            legal_actions = [
                initial_hands[narrator][s]
                for s in range(7) if decision.legal_mask[s]
            ]

            return GamePosition(
                seed=seed,
                decl_id=decl_id,
                narrator=narrator,
                partner=partner,
                narrator_team=narrator_team,
                initial_hand=list(initial_hands[narrator]),
                remaining_hand=narrator_remaining,
                completed_tricks=completed_tricks,
                current_trick_plays=current_plays,
                current_lead_dom=current_lead_dom,
                is_leading=is_leading,
                team_points=list(team_points),
                count_taken=count_taken,
                all_hands=initial_hands,
                legal_actions=legal_actions,
            )

    return None


# ---------------------------------------------------------------------------
# Compact prompt renderer
# ---------------------------------------------------------------------------

def _player_ref(player: int, narrator: int, partner: int, *, lower: bool = False) -> str:
    if player == narrator:
        return "you" if lower else "You"
    if player == partner:
        return "partner" if lower else "Partner"
    return f"P{player}"


def _classify_follow(dom_id: int, lead_dom: int, decl_id: int) -> str:
    t = _t()
    led_suit = t.led_suit_for_lead_domino(lead_dom, decl_id)
    if t.has_trump_power(decl_id) and t.is_in_called_suit(dom_id, decl_id):
        if led_suit == 7:
            return "follows"
        return "trumps"
    if t.can_follow(dom_id, led_suit, decl_id):
        return "follows"
    return "sluffs"


def render_compact_prompt(pos: GamePosition) -> str:
    """Render game state in compact format (~170 tokens)."""
    t = _t()
    lines = []

    # Header
    lines.append(f"Trump: {_decl_name(pos.decl_id)}.")
    lines.append(f"You are P{pos.narrator}. Partner: P{pos.partner}.")
    lines.append(f"Your starting hand: {', '.join(_dom(d) for d in pos.initial_hand)}.")
    lines.append("")

    # Completed tricks
    for trick in pos.completed_tricks:
        parts = []
        for i, play in enumerate(trick.plays):
            pref = _player_ref(play.player, pos.narrator, pos.partner)
            is_narrator = play.player == pos.narrator
            if i == 0:
                verb = "lead" if is_narrator else "leads"
            else:
                kind = _classify_follow(play.dom_id, trick.lead_dom, pos.decl_id)
                # "follows" → "follow", "trumps" → "trump", "sluffs" → "sluff" for narrator
                verb = kind.rstrip("s") if is_narrator else kind
            forced = " (forced)" if play.is_forced else ""
            parts.append(f"{pref} {verb} {_dom(play.dom_id)}{forced}")
        winner = _player_ref(trick.winner_player, pos.narrator, pos.partner)
        win_verb = "win" if trick.winner_player == pos.narrator else "wins"
        my_pts = pos.team_points[pos.narrator_team] if trick.trick_num <= len(pos.completed_tricks) else "?"
        # Compute running score at this trick
        running = [0, 0]
        for ct in pos.completed_tricks[:trick.trick_num]:
            running[ct.winner_team] += ct.points
        you_pts = running[pos.narrator_team]
        them_pts = running[1 - pos.narrator_team]
        lines.append(
            f"Trick {trick.trick_num}: {'. '.join(parts)}."
        )
        lines.append(f"  → {winner} {win_verb}. [{you_pts}-{them_pts}]")

    lines.append("")

    # Current trick (partial or leading)
    you_pts = pos.team_points[pos.narrator_team]
    them_pts = pos.team_points[1 - pos.narrator_team]

    if pos.is_leading:
        lines.append(f"Trick 6: Your lead.")
    else:
        parts = []
        for i, play in enumerate(pos.current_trick_plays):
            pref = _player_ref(play.player, pos.narrator, pos.partner)
            is_narrator = play.player == pos.narrator
            if i == 0:
                verb = "lead" if is_narrator else "leads"
            else:
                kind = _classify_follow(play.dom_id, pos.current_lead_dom, pos.decl_id)
                verb = kind.rstrip("s") if is_narrator else kind
            parts.append(f"{pref} {verb} {_dom(play.dom_id)}")
        lines.append(f"Trick 6: {'. '.join(parts)}. Your turn.")

    lines.append(f"Your hand: {', '.join(_dom(d) for d in pos.remaining_hand)}.")
    lines.append(f"Score: you {you_pts}, them {them_pts}.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Question generators
# ---------------------------------------------------------------------------

def _all_played_doms(pos: GamePosition) -> list[tuple[int, int, int]]:
    """Return (dom_id, player, trick_num) for all played dominoes."""
    played = []
    for trick in pos.completed_tricks:
        for play in trick.plays:
            played.append((play.dom_id, play.player, trick.trick_num))
    for play in pos.current_trick_plays:
        played.append((play.dom_id, play.player, 6))
    return played


def gen_where_is(pos: GamePosition, rng: random.Random) -> dict | None:
    """Q: Where is the X-Y?"""
    t = _t()
    played = _all_played_doms(pos)
    played_ids = {d[0] for d in played}

    # Pick a domino — weight toward interesting ones (counts, trumps, in-hand)
    candidates = list(range(28))
    rng.shuffle(candidates)
    dom_id = candidates[0]

    d = _dom(dom_id)
    if dom_id in pos.remaining_hand:
        answer = f"The {d} is in your hand."
    elif dom_id in played_ids:
        info = next(p for p in played if p[0] == dom_id)
        _, player, trick_num = info
        pref = _player_ref(player, pos.narrator, pos.partner, lower=True)
        answer = f"The {d} was played by {pref} on trick {trick_num}."
    else:
        # Not played, not in narrator's hand → in someone else's hand
        answer = f"The {d} has not been played yet and is not in your hand. It is held by an opponent or partner (unknown which)."

    return {
        "category": "where_is",
        "question": f"Where is the {d}?",
        "answer": answer,
    }


def gen_count_status(pos: GamePosition, rng: random.Random) -> dict | None:
    """Q: Who captured the [count domino]?"""
    t = _t()
    count_dom_ids = [i for i in range(28) if t.DOMINO_COUNT_POINTS[i] > 0]
    rng.shuffle(count_dom_ids)
    dom_id = count_dom_ids[0]

    d = _dom(dom_id)
    pts = t.DOMINO_COUNT_POINTS[dom_id]

    # Check if captured
    captured = [(did, team) for did, team in pos.count_taken if did == dom_id]
    if captured:
        team = captured[0][1]
        team_name = "your team" if team == pos.narrator_team else "the opponents"
        # Find who played it and on which trick
        played = _all_played_doms(pos)
        info = next((p for p in played if p[0] == dom_id), None)
        if info:
            _, player, trick_num = info
            pref = _player_ref(player, pos.narrator, pos.partner, lower=True)
            answer = f"The {d} ({pts} count) was captured by {team_name}. It was played by {pref} on trick {trick_num}."
        else:
            answer = f"The {d} ({pts} count) was captured by {team_name}."
    elif dom_id in pos.remaining_hand:
        answer = f"The {d} ({pts} count) has not been captured. It is in your hand."
    else:
        # Check if it's in the current partial trick (not yet resolved)
        current_doms = {p.dom_id for p in pos.current_trick_plays}
        if dom_id in current_doms:
            answer = f"The {d} ({pts} count) has been played on the current trick but the trick is not yet resolved."
        else:
            answer = f"The {d} ({pts} count) has not been played yet. It is still out."

    return {
        "category": "count_status",
        "question": f"What is the status of the {d}?",
        "answer": answer,
    }


def gen_is_trump(pos: GamePosition, rng: random.Random) -> dict | None:
    """Q: Is the X-Y a trump?"""
    t = _t()

    # Pick from hand dominoes (most relevant) plus some random ones
    candidates = list(pos.remaining_hand)
    # Add a few random non-hand dominoes for variety
    others = [i for i in range(28) if i not in pos.remaining_hand]
    rng.shuffle(others)
    candidates.extend(others[:3])
    rng.shuffle(candidates)
    dom_id = candidates[0]

    d = _dom(dom_id)
    is_t = _is_trump(dom_id, pos.decl_id)
    reason = _trump_reason(dom_id, pos.decl_id)

    if is_t:
        answer = f"Yes. {reason}"
    else:
        answer = f"No. {reason}"

    return {
        "category": "is_trump",
        "question": f"Is the {d} a trump?",
        "answer": answer,
    }


def gen_what_beats(pos: GamePosition, rng: random.Random) -> dict | None:
    """Q: What dominoes can beat X-Y when [suit] is led?"""
    t = _t()

    if pos.is_leading:
        # For leading, pick one of narrator's dominoes and ask what beats it
        # if narrator leads it (i.e., what's the led suit and what ranks above)
        dom_id = rng.choice(pos.remaining_hand)
        led_suit = t.led_suit_for_lead_domino(dom_id, pos.decl_id)
        suit_name = _suit_name(led_suit, pos.decl_id)
    else:
        # For following, ask about the lead domino
        dom_id = pos.current_lead_dom
        if dom_id is None:
            return None
        led_suit = t.led_suit_for_lead_domino(dom_id, pos.decl_id)
        suit_name = _suit_name(led_suit, pos.decl_id)

    d = _dom(dom_id)
    my_rank = t.trick_rank(dom_id, led_suit, pos.decl_id)

    # Find all dominoes that beat it
    beaters = []
    for i in range(28):
        r = t.trick_rank(i, led_suit, pos.decl_id)
        if r > my_rank:
            beaters.append((i, r))
    beaters.sort(key=lambda x: -x[1])

    if not beaters:
        answer = f"Nothing can beat the {d}. It is the highest-ranked domino when {suit_name} is led."
    else:
        beater_strs = [_dom(b[0]) for b in beaters]
        # Separate into in-suit and trump beaters
        in_suit = [b for b in beaters if not _is_trump(b[0], pos.decl_id) or led_suit == 7]
        trumps = [b for b in beaters if _is_trump(b[0], pos.decl_id) and led_suit != 7]

        parts = []
        if in_suit:
            in_suit_strs = [_dom(b[0]) for b in in_suit]
            parts.append(f"In {suit_name}: {', '.join(in_suit_strs)}")
        if trumps:
            trump_strs = [_dom(b[0]) for b in trumps]
            parts.append(f"By trumping: {', '.join(trump_strs)}")

        answer = f"Dominoes that beat the {d} when {suit_name} is led: {'. '.join(parts)}."

    return {
        "category": "what_beats",
        "question": f"What dominoes can beat the {d} when {suit_name} is led?",
        "answer": answer,
    }


def gen_what_do_you_play(pos: GamePosition, rng: random.Random) -> dict | None:
    """Q: What are your legal moves? (derivation showing work)"""
    t = _t()

    legal_strs = [_dom(d) for d in pos.legal_actions]

    if pos.is_leading:
        answer = (
            f"You have the lead, so you may play any domino in your hand. "
            f"Legal moves: {', '.join(legal_strs)}."
        )
    else:
        lead_dom = pos.current_lead_dom
        led_suit = t.led_suit_for_lead_domino(lead_dom, pos.decl_id)
        suit_name = _suit_name(led_suit, pos.decl_id)

        # Check each domino in hand
        checks = []
        followers = []
        for dom_id in pos.remaining_hand:
            d = _dom(dom_id)
            h, lo = t.DOMINO_HIGH[dom_id], t.DOMINO_LOW[dom_id]

            trump_name = _decl_name(pos.decl_id)

            if led_suit == 7:
                # Trump was led
                if _is_trump(dom_id, pos.decl_id):
                    checks.append(f"{d}: trump. Can follow.")
                    followers.append(d)
                else:
                    checks.append(f"{d}: not trump. Cannot follow.")
            else:
                # Pip suit was led
                if _is_trump(dom_id, pos.decl_id):
                    # Trump — can't follow a non-trump suit
                    if pos.decl_id <= 6:
                        checks.append(
                            f"{d}: contains a {pos.decl_id} ({trump_name} are trump), "
                            f"so it is trump, not in {suit_name}. Cannot follow."
                        )
                    elif pos.decl_id == t.DOUBLES_TRUMP:
                        checks.append(
                            f"{d}: double (doubles are trump), not in {suit_name}. Cannot follow."
                        )
                    else:
                        checks.append(f"{d}: trump, not in {suit_name}. Cannot follow.")
                elif t.can_follow(dom_id, led_suit, pos.decl_id):
                    checks.append(f"{d}: contains a {led_suit}, in {suit_name}. Can follow.")
                    followers.append(d)
                else:
                    checks.append(f"{d}: pips {h} and {lo}, not in {suit_name}. Cannot follow.")

        check_text = "\n".join(f"- {c}" for c in checks)

        if followers:
            answer = (
                f"{_dom(lead_dom)} was led. Led suit: {suit_name}.\n"
                f"{check_text}\n"
                f"You must follow suit. Legal moves: {', '.join(followers)}."
            )
        else:
            answer = (
                f"{_dom(lead_dom)} was led. Led suit: {suit_name}.\n"
                f"{check_text}\n"
                f"You hold no {suit_name}. You may play any domino. "
                f"Legal moves: {', '.join(legal_strs)}."
            )

    return {
        "category": "legal_moves",
        "question": "What are your legal moves?",
        "answer": answer,
    }


# All generators
GENERATORS = [
    ("where_is", gen_where_is, 1.0),
    ("count_status", gen_count_status, 1.0),
    ("is_trump", gen_is_trump, 1.5),  # weight up — key comprehension skill
    ("legal_moves", gen_what_do_you_play, 2.0),  # weight up — the 0% illegal target
    ("what_beats", gen_what_beats, 1.5),  # weight up — ranking comprehension
]


def generate_questions(
    pos: GamePosition,
    n_questions: int,
    rng: random.Random,
    exhaustive: bool = False,
) -> list[dict]:
    """Generate questions for a game position."""
    prompt = render_compact_prompt(pos)

    if exhaustive:
        # Generate one of each type
        results = []
        for name, gen_fn, _ in GENERATORS:
            q = gen_fn(pos, rng)
            if q:
                q["prompt"] = prompt
                q["seed"] = pos.seed
                q["decl_id"] = pos.decl_id
                q["decl_name"] = _decl_name(pos.decl_id)
                q["narrator"] = pos.narrator
                results.append(q)
        return results

    # Weighted sampling
    names, gen_fns, weights = zip(*GENERATORS)
    results = []
    for _ in range(n_questions):
        chosen = rng.choices(range(len(GENERATORS)), weights=weights, k=1)[0]
        gen_fn = gen_fns[chosen]
        q = gen_fn(pos, rng)
        if q:
            q["prompt"] = prompt
            q["seed"] = pos.seed
            q["decl_id"] = pos.decl_id
            q["decl_name"] = _decl_name(pos.decl_id)
            q["narrator"] = pos.narrator
            results.append(q)

    return results


# ---------------------------------------------------------------------------
# Main: batch generation
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comprehension Q&A from game records")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--questions-per-position", type=int, default=5)
    parser.add_argument("--allow-eval-seeds", action="store_true")
    parser.add_argument("--exhaustive", action="store_true",
                        help="Generate one of each question type per position (for eval)")
    parser.add_argument("--bid", type=int, default=30)
    args = parser.parse_args()

    # Heavy imports
    import torch
    from forge.oracle.declarations import DECL_ID_TO_NAME, N_DECLS
    from forge.oracle.rng import deal_from_seed
    from forge.eq.generate.pipeline import generate_eq_games_gpu
    from forge.eq.oracle import Stage1Oracle

    log(f"[model] Loading oracle from {args.checkpoint}")
    oracle = Stage1Oracle(args.checkpoint, device="cuda", compile=False)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_decl_ids = list(range(N_DECLS))
    total_questions = 0
    total_positions = 0
    total_skipped = 0
    t_start = time.time()
    rng = random.Random(42)

    with open(output_path, "w") as f:
        for seed_offset in range(args.count):
            seed = args.start_seed + seed_offset

            if not args.allow_eval_seeds and EVAL_SEED_START <= seed <= EVAL_SEED_END:
                continue

            hands = deal_from_seed(seed)
            batch_hands = [hands] * N_DECLS
            batch_decls = all_decl_ids

            try:
                records = generate_eq_games_gpu(
                    model=oracle.model,
                    hands=batch_hands,
                    decl_ids=batch_decls,
                    n_samples=args.n_samples,
                    device="cuda",
                    seeds=[seed * 10 + d for d in range(N_DECLS)],
                )
            except Exception as e:
                log(f"[error] seed {seed}: {e}")
                continue

            for record in records:
                for narrator in range(4):
                    pos = extract_position(record, narrator, seed)
                    if pos is None:
                        total_skipped += 1
                        continue

                    total_positions += 1
                    questions = generate_questions(
                        pos, args.questions_per_position, rng,
                        exhaustive=args.exhaustive,
                    )

                    for q in questions:
                        f.write(json.dumps(q) + "\n")
                        total_questions += 1

            # Progress
            seeds_done = seed_offset + 1
            elapsed = time.time() - t_start
            rate = seeds_done / elapsed if elapsed > 0 else 0
            if seeds_done % 10 == 0 or seeds_done == args.count:
                log(
                    f"[progress] {seeds_done}/{args.count} seeds | "
                    f"{total_positions} positions | {total_questions} questions | "
                    f"{rate:.1f} seeds/s"
                )

    elapsed = time.time() - t_start
    log(f"\n[done] {total_questions} questions from {total_positions} positions "
        f"({args.count} seeds) in {elapsed:.0f}s")
    log(f"  Skipped (forced/missing): {total_skipped}")
    log(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
