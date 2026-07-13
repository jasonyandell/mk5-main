"""Lightweight template-based rationalizer for the student's per-decision output.

No ML training, no new heads. Reads the student's existing outputs
(belief_logits, V_head, π_me_logits, Q_head across sampled worlds) and
turns them into a 2-3 sentence natural language explanation — "a cricket
commentator making sense of a shot".

Public entry point:
    explain_decision(ctx) -> str

`ctx` is a dict built by play_visualizer.py per decision, containing
everything we need without re-running the model.

Design notes:
- Pure template-filling. No generation. Deterministic.
- Only surfaces slots that are "signal-full" — if π is split and belief is
  flat, we say both. If π is peaked and risk is low, we emit one crisp line.
- Blunder hotspots (d_idx in {4, 8, 11, 12, 16}) and end-game (d_idx >= 24)
  get explicit pattern labels.

Self-contained — no new dependencies beyond what play_visualizer already
imports.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

# Blunder hotspots observed in prior evaluation — see forge/ receipt #4 and
# the play_visualizer regret histograms across the 20-game eval corpus.
BLUNDER_HOTSPOTS = {4, 8, 11, 12, 16}


def _entropy(probs: Tensor, eps: float = 1e-12) -> float:
    """Shannon entropy in nats, ignoring zero-mass entries."""
    p = probs.clamp(min=eps)
    return float(-(p * p.log()).sum().item())


def _top2(probs: Tensor, labels: list[str]) -> tuple[tuple[str, float], tuple[str, float] | None]:
    """Return ((best_label, p_best), (second_label, p_second)) or None if no second."""
    pairs = sorted(zip(labels, probs.tolist()), key=lambda kv: -kv[1])
    first = (pairs[0][0], float(pairs[0][1]))
    second = (pairs[1][0], float(pairs[1][1])) if len(pairs) > 1 else None
    return first, second


def _fmt_confidence(
    pi_probs_legal: Tensor,
    legal_labels: list[str],
    student_pick_label: str,
) -> str:
    """'confidently picks 5-5' / 'picks 5-5 (close: 3-2)' / 'splits: 5-5
    (p=0.40) vs 3-2 (p=0.30)'."""
    (best_label, p_best), second = _top2(pi_probs_legal, legal_labels)
    # student_pick_label is the legal-argmax — should equal best_label. Use
    # whichever matches to be robust.
    label = student_pick_label if student_pick_label else best_label
    if p_best > 0.9:
        return f"confidently picks **{label}** (peak {p_best:.2f})"
    if p_best > 0.5 and second is not None:
        return f"picks **{label}** (p={p_best:.2f}; second-best {second[0]} p={second[1]:.2f})"
    # Split decision
    if second is not None:
        return (
            f"splits: **{label}** (p={p_best:.2f}) vs "
            f"{second[0]} (p={second[1]:.2f})"
        )
    return f"picks **{label}** (p={p_best:.2f})"


def _fmt_belief(
    belief_logits: Tensor,   # [28, 3]
    belief_mask: Tensor,     # [28]
    domino_name_fn,
    top_k: int = 2,
    min_prob: float = 0.5,
) -> str:
    """'believes partner holds 6-6 (p=0.71), left-opp is short on fives (p=0.58)'.

    Returns empty string if no belief is confident enough to mention.
    """
    seat_labels = {0: "left-opp", 1: "partner", 2: "right-opp"}
    probs = torch.softmax(belief_logits, dim=-1)  # [28, 3]
    rows: list[tuple[float, int, int]] = []
    for d in range(28):
        if not bool(belief_mask[d].item()):
            continue
        p, seat = probs[d].max(dim=-1)
        pv = float(p.item())
        if pv >= min_prob:
            rows.append((pv, d, int(seat.item())))
    rows.sort(reverse=True)
    if not rows:
        return ""
    phrases: list[str] = []
    for pv, d, s in rows[:top_k]:
        phrases.append(f"{seat_labels[s]} holds {domino_name_fn(d)} (p={pv:.2f})")
    return "believes " + ", ".join(phrases)


def _fmt_value(
    v_student: float,
    oracle_best: float,
    student_eq: float,
) -> str:
    """'expects +12 Q; oracle's best +14.' Flags V/π disagreement if V is
    noticeably above the chosen action's oracle E[Q]."""
    # V_head says the state is worth v_student. The student's chosen action
    # has oracle E[Q] of student_eq. If V >> student_eq, V/π disagree.
    disagreement = v_student - student_eq
    base = f"expects {v_student:+.1f} Q (oracle top: {oracle_best:+.1f})"
    if disagreement > 3.0:
        base += f" — V/π disagreement: V says {v_student:+.1f} but π's pick only scores {student_eq:+.1f}"
    return base


def _fmt_risk(q_std_at_pick: float | None) -> str:
    """'low risk' / 'moderate risk (Q ±4.2)' / 'high variance — could swing ±12 Q'.

    Thresholds are calibrated to what's actually signal-full in the v2 student:
    below ~5 Q-pts of cross-world std is "this action's outcome doesn't depend
    much on belief uncertainty". Above ~10 is genuinely high variance.
    """
    if q_std_at_pick is None:
        return ""
    s = q_std_at_pick
    if s < 5.0:
        return f"low risk (Q spread ±{s:.1f} across worlds)"
    if s < 10.0:
        return f"moderate risk (Q spread ±{s:.1f} across worlds)"
    return f"high variance — Q could swing ±{s:.1f} across worlds"


def _pattern_label(
    d_idx: int,
    n_legal: int,
    oracle_spread: float,
) -> str | None:
    """Label special decision types. Returns None if decision is 'ordinary'."""
    if n_legal == 1:
        return "forced move"
    if d_idx in BLUNDER_HOTSPOTS and oracle_spread > 5.0:
        return "strategic spike — high-stakes pick"
    if d_idx >= 24:
        return "end-game, mostly deterministic"
    return None


def explain_decision(ctx: dict[str, Any]) -> str:
    """Build a short NL explanation from the student's outputs at one decision.

    Expected `ctx` keys (all already computed in play_visualizer.py per
    decision, plus a few extras this module needs):

        d_idx           : int
        player          : int
        pi_probs        : Tensor [7]            — softmax over legal/illegal
        legal_mask      : Tensor [7] bool
        legal_slots     : list[int]             — the legal slot indices
        legal_labels    : list[str]             — domino names for each legal slot
        student_slot    : int                   — legal-argmax of π
        student_pick_label : str                — domino name of chosen action
        student_v       : float                 — V_head output
        student_eq      : float                 — oracle E[Q] at student's pick
        oracle_best     : float                 — oracle's legal-max E[Q]
        e_q             : Tensor [7]            — oracle E[Q] per action
        belief_logits   : Tensor [28, 3]
        belief_mask     : Tensor [28] bool
        q_std_at_pick   : float | None          — std of Q_head across worlds at chosen action
        domino_name_fn  : callable(int) -> str

    Returns a 2-3 sentence markdown string (no leading bullet).
    """
    d_idx = int(ctx["d_idx"])
    legal_mask: Tensor = ctx["legal_mask"]
    n_legal = int(legal_mask.sum().item())

    e_q = ctx["e_q"].clone()
    e_q_legal = e_q.masked_fill(~legal_mask, float("-inf"))
    finite = e_q_legal[torch.isfinite(e_q_legal)]
    if finite.numel() >= 2:
        oracle_spread = float(finite.max().item() - finite.min().item())
    else:
        oracle_spread = 0.0

    # Renormalise π over legal actions (pi_probs may already be over 7 with
    # illegals near zero thanks to -1e9 masking upstream, but we want it strictly
    # over legal slots for cleaner labelling).
    pi = ctx["pi_probs"]
    legal_slots: list[int] = ctx["legal_slots"]
    legal_labels: list[str] = ctx["legal_labels"]
    pi_legal = pi[torch.tensor(legal_slots)]
    pi_legal = pi_legal / pi_legal.sum().clamp(min=1e-9)

    pi_entropy = _entropy(pi_legal)
    pi_peak = float(pi_legal.max().item())

    # --- Build sentence 1: what + how confident --------------------------
    pattern = _pattern_label(d_idx, n_legal, oracle_spread)
    confidence = _fmt_confidence(pi_legal, legal_labels, ctx["student_pick_label"])
    lead_bits: list[str] = []
    if pattern:
        lead_bits.append(f"_{pattern}_")
    lead_bits.append(confidence)
    # Entropy is meaningless for forced moves; skip it.
    if n_legal > 1:
        lead_bits.append(f"entropy {pi_entropy:.2f}")
    sent_1 = " — ".join(lead_bits) + "."

    # --- Build sentence 2: value and risk --------------------------------
    sent_2_parts: list[str] = [
        _fmt_value(
            v_student=float(ctx["student_v"]),
            oracle_best=float(ctx["oracle_best"]),
            student_eq=float(ctx["student_eq"]),
        ),
    ]
    # Risk talk is meaningless for forced moves — we can't avoid the variance.
    if n_legal > 1:
        risk = _fmt_risk(ctx.get("q_std_at_pick"))
        if risk:
            sent_2_parts.append(risk)
    sent_2 = "; ".join(sent_2_parts)
    # Capitalise the first word (value phrase starts with "expects").
    sent_2 = sent_2[0].upper() + sent_2[1:] + "."

    # --- Build sentence 3: belief (only if confident enough) -------------
    belief_str = _fmt_belief(
        ctx["belief_logits"],
        ctx["belief_mask"],
        ctx["domino_name_fn"],
        top_k=2,
        min_prob=0.5,
    )
    if belief_str:
        sent_3 = belief_str.capitalize() + "."
    else:
        sent_3 = ""

    out = sent_1 + " " + sent_2
    if sent_3:
        out += " " + sent_3
    return out
