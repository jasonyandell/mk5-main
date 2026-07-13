"""Belief trajectory tool — exposes Gus's forward-pass belief + V + pi_me
+ final-layer CLS attention for the current decision state.

This is the Gus<->Burl bridge: instead of Burl sampling worlds to guess
opponents' hands, it reads Gus's calibrated belief head directly. Gus's
belief is at the Bayes ceiling on the corpus (see
wiki/topics/belief-bayes-ceiling.md) — so this trajectory is as good as any
oracle-free belief can be.

Return shape is LLM-legible:
  - posterior_by_domino:      per-unseen 3-way posterior + entropy + argmax
  - top_shifts_since_last:    top-K movers vs the previous call (empty if first)
  - gus_value_estimate:       V_head scalar
  - gus_policy_top:           top-K pi_me entries over legal actions
  - attention_top_tokens:     top-3 state tokens the final-layer CLS attends to

Load strategy mirrors ``burl/tools/eq_distribution.py``: a module-level
``lru_cache`` holds the StudentTransformerFullVoids adapter, keyed on
(adapter_path, device). First call pays the load; subsequent calls are cheap.

Not in v0 (punt to v1): per-layer attention trajectory, multi-head breakdown,
prose rendering of attention beyond the token label, counterfactual probes.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from burl.tools._gus_adapter import build_gus_inputs
from burl.tools.engine import _abs_current_player, is_legal
from gus.model.student import StudentTransformerFull, StudentTransformerFullVoids
from gus.model.tokenize import (
    CLS_TOKEN,
    DECL_OFFSET,
    PAD_TOKEN,
    SEQ_LEN,
    TYPE_CLS,
    TYPE_DECL,
    TYPE_MINE,
    TYPE_PLAY,
)
from gus.model.voids import is_trump as _gus_is_trump

DEFAULT_ADAPTER = (
    Path(__file__).resolve().parents[2] / "gus/adapters/v3_consistency_10000g.pt"
)
_SEATS = ("left_opp", "partner", "right_opp")
_SEAT_ROLES = {0: "me", 1: "left_opp", 2: "partner", 3: "right_opp"}


# --------------------------------------------------------------------------- #
# Adapter loader (module-level cache)                                          #
# --------------------------------------------------------------------------- #


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=2)
def _load_gus_cached(adapter_path: str, device: str):
    """Key-on-(path, device) cache. Loads the student + records adapter label."""
    ckpt = torch.load(adapter_path, weights_only=False, map_location=device)
    cfg = ckpt["args"]
    is_voids = "voids_hidden" in cfg
    cls = StudentTransformerFullVoids if is_voids else StudentTransformerFull
    kwargs = {
        "d_model": cfg["d_model"],
        "n_heads": cfg["n_heads"],
        "n_layers": cfg["n_layers"],
        "ff_dim": cfg.get("ff_dim", 256),
        "dropout": 0.0,
        "d_world": cfg.get("d_world", 64),
        "q_hidden": cfg.get("q_hidden", 256),
    }
    if is_voids:
        kwargs["voids_hidden"] = cfg.get("voids_hidden", 64)
    model = cls(**kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    label = Path(adapter_path).stem
    return model, is_voids, label


def load_gus(
    adapter_path: str | Path | None = None,
    device: str | None = None,
):
    if device is None:
        device = _pick_device()
    path = str(adapter_path) if adapter_path is not None else str(DEFAULT_ADAPTER)
    return _load_gus_cached(path, device)


# --------------------------------------------------------------------------- #
# Trajectory memory (cross-call shift detection)                               #
# --------------------------------------------------------------------------- #


def _state_hash(state: Any) -> str:
    """Deterministic hash over the fields that uniquely identify a decision.

    Used to key the previous-call posterior for shift detection. Includes the
    full play history + decl + current player, so two decisions that differ
    only by current_player hash distinctly (important — shift is seat-specific).
    """
    history = tuple(
        (int(x[0]), int(x[1])) for x in state.play_history
    )
    me = _abs_current_player(state)
    key = (int(state.decl_id), me, history)
    return hashlib.sha1(repr(key).encode()).hexdigest()


# Keyed by a "game identity" — caller-provided or derived from hands. Holds the
# previous decision's (posterior[28,3], state_hash) pair. Best-effort: cache
# miss -> empty top_shifts_since_last, which is the documented behavior.
_PREV_POSTERIOR: dict[str, tuple[torch.Tensor, str]] = {}


def _game_key(state: Any) -> str:
    """Stable identity for one game — the initial deal. Independent of play
    order so we track a trajectory across decisions in the same game."""
    hands_repr = tuple(
        tuple(int(d) for d in state.hands[p]) for p in range(4)
    )
    return hashlib.sha1(
        repr((int(state.decl_id), hands_repr)).encode()
    ).hexdigest()


# --------------------------------------------------------------------------- #
# Token label reconstruction (tokenize.py layout is fixed)                     #
# --------------------------------------------------------------------------- #


def _domino_label(d: int) -> str:
    # Matches the (high, low) canonical order used in gus/model/voids.py
    # (see ``domino_pips``). Triangular: [(a,b) for a in 0..6 for b in 0..a].
    idx = 0
    for a in range(7):
        for b in range(a + 1):
            if idx == d:
                return f"{a}-{b}"
            idx += 1
    return f"?-{d}"


_DECL_NAMES = (
    "blanks", "ones", "twos", "threes", "fours", "fives", "sixes",
    "doubles-trump", "doubles-suit", "notrump",
)


def _reconstruct_token_labels(
    tokens: torch.Tensor,        # [SEQ_LEN, 5] long, already detached on CPU
    attention_mask: torch.Tensor, # [SEQ_LEN] bool
    prior_plays: list[tuple[int, int]],
) -> list[str | None]:
    """Map each of the 33 token positions to a human-readable label.

    Layout (from tokenize.py docstring):
      pos 0     -> CLS
      pos 1     -> DECL=<name>
      pos 2..8  -> MINE[i]=H-L       (current player's hand slot)
      pos 9..32 -> PLAY[t=T,pos=P,SEAT]=H-L
    PAD positions return None so the hook can skip them.
    """
    labels: list[str | None] = [None] * SEQ_LEN
    tok = tokens[:, 0].tolist()
    typ = tokens[:, 1].tolist()
    trick = tokens[:, 2].tolist()
    pos_in_trick = tokens[:, 3].tolist()
    player_rel = tokens[:, 4].tolist()

    for i in range(SEQ_LEN):
        if not bool(attention_mask[i].item()):
            labels[i] = None
            continue
        t_id = int(tok[i])
        t_type = int(typ[i])
        if t_type == TYPE_CLS or t_id == CLS_TOKEN:
            labels[i] = "CLS"
        elif t_type == TYPE_DECL:
            decl_k = t_id - DECL_OFFSET
            name = _DECL_NAMES[decl_k] if 0 <= decl_k < len(_DECL_NAMES) else f"?{decl_k}"
            labels[i] = f"DECL={name}"
        elif t_type == TYPE_MINE:
            if t_id == PAD_TOKEN:
                labels[i] = None
                continue
            slot = i - 2  # MINE starts at position 2
            labels[i] = f"MINE[{slot}]={_domino_label(t_id)}"
        elif t_type == TYPE_PLAY:
            if t_id == PAD_TOKEN:
                labels[i] = None
                continue
            t = int(trick[i])
            p = int(pos_in_trick[i])
            rel = int(player_rel[i])
            seat = _SEAT_ROLES.get(rel, f"rel{rel}")
            labels[i] = f"PLAY[t={t},pos={p},{seat}]={_domino_label(t_id)}"
        else:
            labels[i] = None
    return labels


# --------------------------------------------------------------------------- #
# Attention capture                                                            #
# --------------------------------------------------------------------------- #


class _CLSAttentionHook:
    """Hooks the FINAL encoder layer's self-attention to capture CLS attn.

    nn.TransformerEncoderLayer.forward internally calls
    ``self.self_attn(..., need_weights=False)``, so weights never surface. We
    wrap that call: a forward_pre_hook rewrites the kwargs to need_weights=True,
    and a forward_hook captures ``output[1]`` (the [B, L, L] avg-head matrix).

    Head reduction: mean over heads (the PyTorch default for
    ``average_attn_weights=True``). Note: this averaging happens inside
    nn.MultiheadAttention when need_weights=True.
    """

    def __init__(self):
        self.weights: torch.Tensor | None = None  # [B, L, L]
        self._pre_handle = None
        self._post_handle = None

    def _pre(self, module, args, kwargs):
        # Force weights ON. TransformerEncoderLayer passes need_weights=False
        # as a positional-via-kwarg; we override.
        kwargs = dict(kwargs)
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = True
        return args, kwargs

    def _post(self, module, inputs, output):
        # MultiheadAttention returns (attn_output, attn_output_weights).
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            self.weights = output[1].detach()

    def attach(self, model: torch.nn.Module) -> None:
        # Find the final encoder layer's self-attention module.
        blocks = model.encoder.blocks.layers  # nn.TransformerEncoder.layers
        final_layer = blocks[-1]
        mha = final_layer.self_attn
        self._pre_handle = mha.register_forward_pre_hook(self._pre, with_kwargs=True)
        self._post_handle = mha.register_forward_hook(self._post)

    def detach(self) -> None:
        if self._pre_handle is not None:
            self._pre_handle.remove()
        if self._post_handle is not None:
            self._post_handle.remove()
        self._pre_handle = None
        self._post_handle = None


# --------------------------------------------------------------------------- #
# Public entry point                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class _BeliefForwardResult:
    """Raw tensors from one forward pass, before LLM-legible shaping."""
    belief_probs: torch.Tensor        # [28, 3] on CPU
    belief_mask: torch.Tensor         # [28] bool — True where unseen
    v: float
    pi_me_probs_legal: list[tuple[int, float]]  # (slot_idx_or_domino_id, p)
    cls_attention: torch.Tensor       # [SEQ_LEN] CPU float — row 0 of attn
    token_labels: list[str | None]


def _run_forward(
    state: Any,
    model: torch.nn.Module,
    is_voids: bool,
    device: str,
) -> _BeliefForwardResult:
    """Single forward pass with the attention hook engaged."""
    inputs = build_gus_inputs(state)
    tokens = inputs["tokens"].unsqueeze(0).to(device)           # [1, 33, 5]
    attn_mask = inputs["attention_mask"].unsqueeze(0).to(device)  # [1, 33]
    voids = inputs["voids"].unsqueeze(0).to(device)              # [1, 24]
    # World assignment is a zero dummy: Q-head output is not consumed in v0,
    # and the belief/V/pi_me heads don't depend on world_emb (they branch off
    # state_emb, which is encoder output + voids projection — see
    # StudentTransformerFullVoids.forward in gus/model/student.py).
    world_assignment = torch.zeros(1, 28, 3, device=device)

    hook = _CLSAttentionHook()
    hook.attach(model)
    try:
        with torch.no_grad():
            if is_voids:
                out = model(tokens, attn_mask, world_assignment, voids)
            else:
                out = model(tokens, attn_mask, world_assignment)
    finally:
        hook.detach()

    belief_logits = out["belief_logits"][0]         # [28, 3]
    belief_probs = F.softmax(belief_logits, dim=-1).cpu()  # [28, 3]
    v = float(out["v"][0].item())
    pi_me_logits = out["pi_me_logits"][0].cpu()     # [7]

    # Build the belief_mask from the state: a domino is "unseen" to the
    # current player iff it's not in their initial hand AND not yet played.
    me = inputs["current_player"]
    initial_hand_set = {int(d) for d in inputs["game_hands"][me]}
    played_set = {int(d) for (_p, d) in inputs["prior_plays"]}
    mask = torch.zeros(28, dtype=torch.bool)
    for d in range(28):
        if d in initial_hand_set or d in played_set:
            continue
        mask[d] = True

    # pi_me_probs over LEGAL actions. pi_me_logits are per-slot (0..6 = hand
    # slot). Map slot -> domino_id via initial hand; filter to legal plays.
    # Uses burl/tools/engine.is_legal to mirror the conventions already used
    # elsewhere in burl.
    initial_hand = inputs["game_hands"][me]
    remaining = [
        (slot, int(d)) for slot, d in enumerate(initial_hand)
        if int(d) not in played_set
    ]
    # Legal mask per slot.
    legal_slot_dom: list[tuple[int, int]] = []
    for slot, dom in remaining:
        ok, _reason = is_legal(state, dom)
        if ok:
            legal_slot_dom.append((slot, dom))
    if legal_slot_dom:
        slot_idxs = torch.tensor([s for s, _ in legal_slot_dom], dtype=torch.long)
        sub = pi_me_logits[slot_idxs]
        probs = F.softmax(sub, dim=-1).tolist()
        pi_me_pairs = [(dom, float(p)) for (_s, dom), p in zip(legal_slot_dom, probs)]
    else:
        pi_me_pairs = []

    # CLS attention row — row 0 of the [L, L] matrix.
    if hook.weights is not None:
        attn = hook.weights[0, 0, :].cpu()           # [SEQ_LEN]
    else:
        attn = torch.zeros(SEQ_LEN)

    # Labels for every token position — masked positions get None.
    labels = _reconstruct_token_labels(
        inputs["tokens"],                # [33, 5] on CPU (build_gus_inputs returns CPU)
        inputs["attention_mask"],
        inputs["prior_plays"],
    )

    return _BeliefForwardResult(
        belief_probs=belief_probs,
        belief_mask=mask,
        v=v,
        pi_me_probs_legal=pi_me_pairs,
        cls_attention=attn,
        token_labels=labels,
    )


def _entropy_bits(probs: torch.Tensor) -> float:
    """Shannon entropy in bits for a probability vector."""
    p = probs.clamp(min=1e-12)
    h = -(p * p.log2()).sum()
    return float(h.item())


def _kl_bits(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL(p || q) in bits for 1-D distributions."""
    p = p.clamp(min=1e-12)
    q = q.clamp(min=1e-12)
    return float((p * (p.log2() - q.log2())).sum().item())


def _summarize_posteriors(
    belief_probs: torch.Tensor,  # [28, 3]
    mask: torch.Tensor,          # [28] bool
) -> list[dict]:
    rows: list[dict] = []
    for d in range(28):
        if not bool(mask[d].item()):
            continue
        row = belief_probs[d]
        argmax = int(row.argmax().item())
        rows.append({
            "domino_id": d,
            "pip_label": _domino_label(d),
            "p_left_opp": float(row[0].item()),
            "p_partner": float(row[1].item()),
            "p_right_opp": float(row[2].item()),
            "entropy_bits": round(_entropy_bits(row), 3),
            "argmax_seat": _SEATS[argmax],
            "confidence": round(float(row[argmax].item()), 3),
        })
    rows.sort(key=lambda r: -r["confidence"])
    return rows


def _top_shifts(
    current: torch.Tensor,    # [28, 3]
    previous: torch.Tensor,   # [28, 3]
    mask_now: torch.Tensor,   # [28] bool
    k: int,
) -> list[dict]:
    """Top-K dominoes by KL(current || previous), restricted to dominoes that
    are unseen NOW. Dominoes that became seen since the last call are dropped
    (their posterior is no longer meaningful)."""
    rows: list[dict] = []
    for d in range(28):
        if not bool(mask_now[d].item()):
            continue
        kl = _kl_bits(current[d], previous[d])
        if kl < 1e-4:
            continue
        rows.append({
            "domino_id": d,
            "pip_label": _domino_label(d),
            "kl_shift_bits": round(kl, 3),
            "from": {
                "left_opp": round(float(previous[d, 0].item()), 3),
                "partner": round(float(previous[d, 1].item()), 3),
                "right_opp": round(float(previous[d, 2].item()), 3),
            },
            "to": {
                "left_opp": round(float(current[d, 0].item()), 3),
                "partner": round(float(current[d, 1].item()), 3),
                "right_opp": round(float(current[d, 2].item()), 3),
            },
        })
    rows.sort(key=lambda r: -r["kl_shift_bits"])
    return rows[:k]


def _top_attention(
    cls_attn: torch.Tensor,           # [SEQ_LEN]
    labels: list[str | None],
    k: int = 3,
) -> list[dict]:
    # Mask out positions with no label (PAD or CLS itself — CLS-to-CLS
    # attention isn't informative for "what does CLS look at?").
    values = cls_attn.tolist()
    scored: list[tuple[float, str]] = []
    for i, w in enumerate(values):
        lbl = labels[i]
        if lbl is None or lbl == "CLS":
            continue
        scored.append((float(w), lbl))
    scored.sort(key=lambda r: -r[0])
    return [{"label": lbl, "weight": round(w, 4)} for (w, lbl) in scored[:k]]


# --------------------------------------------------------------------------- #
# Prose rendering (LLM-legible format, optional)                               #
# --------------------------------------------------------------------------- #


_SEAT_UPPER = {"left_opp": "LEFT", "partner": "PARTNER", "right_opp": "RIGHT"}
_MINE_LABEL_RE = re.compile(r"^MINE\[\d+\]=(\d)-(\d)$")


def _pct(p: float) -> int:
    """Round a probability in [0, 1] to a whole-integer percentage."""
    return int(round(float(p) * 100))


def _format_posterior_tier(
    posteriors: list[dict],
) -> tuple[list[str], list[str], list[str]]:
    """Split posterior rows into (strong, medium, weak) rendered-line lists.

    STRONG: confidence > 0.60
    MEDIUM: 0.40 <= confidence <= 0.60
    WEAK:   confidence < 0.40 (collapsed into a comma-separated summary)
    """
    strong_rows: list[str] = []
    medium_rows: list[str] = []
    weak_pips: list[str] = []
    weak_devs: list[int] = []

    for r in posteriors:
        conf = float(r["confidence"])
        conf_pct = _pct(conf)
        l_pct = _pct(r["p_left_opp"])
        p_pct = _pct(r["p_partner"])
        r_pct = _pct(r["p_right_opp"])
        seat_up = _SEAT_UPPER.get(r["argmax_seat"], r["argmax_seat"].upper())

        if conf > 0.60:
            line = (
                f"  {r['pip_label']:4s} -> {seat_up:8s} {conf_pct:>2d}   "
                f"L:{l_pct:>2d} P:{p_pct:>2d} R:{r_pct:>2d}"
            )
            # confidence > 0.60 is never soft — no tag needed here.
            strong_rows.append(line)
        elif conf >= 0.40:
            soft_tag = "   (soft)" if conf < 0.50 else ""
            line = (
                f"  {r['pip_label']:4s} -> {seat_up:8s} {conf_pct:>2d}   "
                f"L:{l_pct:>2d} P:{p_pct:>2d} R:{r_pct:>2d}{soft_tag}"
            )
            medium_rows.append(line)
        else:
            weak_pips.append(r["pip_label"])
            weak_devs.append(abs(conf_pct - 33))

    # Collapse weak into wrapped ~80-char lines.
    if weak_pips:
        max_dev = max(weak_devs)
        lines: list[str] = []
        cur = "  "
        for i, pip in enumerate(weak_pips):
            token = pip + ("" if i == len(weak_pips) - 1 else ",")
            # +1 for the space separator when not at start.
            sep = "" if cur == "  " else " "
            if len(cur) + len(sep) + len(token) > 80:
                lines.append(cur)
                cur = "  " + token
            else:
                cur = cur + sep + token
        if cur.strip():
            lines.append(cur)
        lines.append(f"  (all within {max_dev}pp of 33/33/33)")
        weak_lines = lines
    else:
        weak_lines = []

    return strong_rows, medium_rows, weak_lines


def _format_shifts(shifts: list[dict], had_prior: bool) -> list[str]:
    """Render the SHIFTS section as a list of lines. ``had_prior`` is True iff
    there was a previous-posterior snapshot in cache (shift list may still be
    empty if nothing moved, but that's semantically different from "first
    call"). The spec: if the shift list is empty AND there was no prior
    snapshot, emit "SHIFTS: (no prior snapshot)"."""
    if not had_prior:
        return ["SHIFTS: (no prior snapshot)"]

    rendered: list[str] = ["SHIFTS since last decision:"]
    omitted = 0
    for s in shifts:
        kl = float(s["kl_shift_bits"])
        if kl < 0.01:
            omitted += 1
            continue
        to_lp = s["to"]
        argmax_key = max(to_lp, key=to_lp.get)  # "left_opp" | "partner" | "right_opp"
        to_seat = _SEAT_UPPER[argmax_key]
        rendered.append(
            f"  {s['pip_label']:4s} moved toward {to_seat:8s}  "
            f"L:{_pct(s['from']['left_opp'])}->{_pct(to_lp['left_opp'])}  "
            f"P:{_pct(s['from']['partner'])}->{_pct(to_lp['partner'])}  "
            f"R:{_pct(s['from']['right_opp'])}->{_pct(to_lp['right_opp'])}   "
            f"(KL={kl:.2f})"
        )
    if omitted:
        rendered.append(f"  ({omitted} smaller shifts under 0.01 bits, omitted)")
    if len(rendered) == 1:
        # All shifts filtered out but the cache had a prior — still say "no".
        rendered.append("  (all shifts under 0.01 bits)")
    return rendered


def _format_attention(
    attention: list[dict],
    decl_id: int,
    game_hands: list[list[int]],
    current_player: int,
) -> list[str]:
    """Render the ATTENTION section. MINE[] entries get a non-trump annotation
    when their domino is not trump under ``decl_id`` (strategically ambiguous
    holdings that CLS attention keeps returning to)."""
    initial_hand = game_hands[current_player]
    lines: list[str] = ["ATTENTION (final layer, CLS):"]
    for row in attention:
        label: str = row["label"]
        weight: float = float(row["weight"])
        annotation = ""
        if label.startswith("DECL="):
            annotation = "  <- trump declaration"
        else:
            m = _MINE_LABEL_RE.match(label)
            if m is not None:
                # Reconstruct the domino_id from the pip pair; the label is
                # authoritative (tokenizer uses canonical triangular order, so
                # the pip pair "a-b" with a >= b matches _domino_label).
                a = int(m.group(1))
                b = int(m.group(2))
                # Walk the same triangular order used in _domino_label.
                dom = -1
                idx = 0
                for aa in range(7):
                    for bb in range(aa + 1):
                        if aa == a and bb == b:
                            dom = idx
                        idx += 1
                if dom >= 0 and dom in initial_hand:
                    if not _gus_is_trump(dom, decl_id):
                        annotation = "  <- non-trump holding, strategically ambiguous"
        lines.append(f"  {label:30s}  {weight:.2f}{annotation}")
    return lines


def _format_prose(
    dict_out: dict,
    decl_id: int,
    game_hands: list[list[int]],
    current_player: int,
    had_prior: bool,
) -> str:
    """Assemble the four-section prose block. Sections separated by blank
    lines; each tier/section header renders even when empty ("(none)") so the
    LLM never has to guess whether a tier was omitted."""
    # Section 1 — header.
    posteriors = dict_out["posterior_by_domino"]
    if posteriors:
        mean_h = sum(r["entropy_bits"] for r in posteriors) / len(posteriors)
    else:
        mean_h = 0.0
    header = (
        f"{dict_out['n_unseen']} unseen. "
        f"Mean H={mean_h:.2f} bits. "
        f"V={dict_out['gus_value_estimate']:+.1f}."
    )

    # Section 2 — posterior tiers.
    strong, medium, weak = _format_posterior_tier(posteriors)
    tiers: list[str] = []
    tiers.append("STRONG (>60% on one seat):")
    tiers.extend(strong if strong else ["  (none)"])
    tiers.append("")
    tiers.append("MEDIUM (40-60%):")
    tiers.extend(medium if medium else ["  (none)"])
    tiers.append("")
    tiers.append("WEAK (<40%, near-uniform):")
    tiers.extend(weak if weak else ["  (none)"])

    # Section 3 — shifts.
    shifts_lines = _format_shifts(dict_out["top_shifts_since_last"], had_prior)

    # Section 4 — attention.
    attn_lines = _format_attention(
        dict_out["attention_top_tokens"], decl_id, game_hands, current_player,
    )

    blocks = [
        header,
        "\n".join(tiers),
        "\n".join(shifts_lines),
        "\n".join(attn_lines),
    ]
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# Public entry point                                                           #
# --------------------------------------------------------------------------- #


def belief_trajectory(
    game_state: Any,
    top_k_shifts: int = 5,
    include_policy: bool = True,
    format: Literal["dict", "prose", "both"] = "both",
    adapter_path: str | Path | None = None,
    device: str | None = None,
) -> dict | str:
    """Call Gus's belief head on the current decision and return a
    narration-friendly summary.

    ``format`` controls the return shape:
      - "dict" — only the structured dict (current v0 behavior).
      - "prose" — only the rendered prose string.
      - "both" (default) — dict with an added "prose" key carrying the string.

    Structured dict keys:
      posterior_by_domino, top_shifts_since_last, gus_value_estimate,
      gus_policy_top (optional), attention_top_tokens, gus_adapter, n_unseen,
      decision_idx.
    """
    if format not in ("dict", "prose", "both"):
        raise ValueError(f"format must be 'dict', 'prose', or 'both'; got {format!r}")
    if device is None:
        device = _pick_device()
    model, is_voids, adapter_label = load_gus(adapter_path=adapter_path, device=device)

    result = _run_forward(game_state, model, is_voids, device)

    # Shift vs previous call — keyed on (game identity, state hash).
    game_key = _game_key(game_state)
    now_hash = _state_hash(game_state)
    prev = _PREV_POSTERIOR.get(game_key)
    had_prior = prev is not None and prev[1] != now_hash
    if had_prior:
        prev_probs = prev[0]
        shifts = _top_shifts(
            result.belief_probs, prev_probs, result.belief_mask, top_k_shifts,
        )
    else:
        shifts = []
    _PREV_POSTERIOR[game_key] = (result.belief_probs.clone(), now_hash)

    out: dict = {
        "posterior_by_domino": _summarize_posteriors(
            result.belief_probs, result.belief_mask,
        ),
        "top_shifts_since_last": shifts,
        "gus_value_estimate": round(result.v, 3),
        "attention_top_tokens": _top_attention(
            result.cls_attention, result.token_labels, k=3,
        ),
        "gus_adapter": adapter_label,
        "n_unseen": int(result.belief_mask.sum().item()),
        "decision_idx": int(len(game_state.play_history)),
    }
    if include_policy:
        top_policy = sorted(
            result.pi_me_probs_legal, key=lambda r: -r[1],
        )[:3]
        out["gus_policy_top"] = [
            {"domino_id": int(d), "pip_label": _domino_label(int(d)), "p": round(p, 3)}
            for d, p in top_policy
        ]

    if format == "dict":
        return out

    # format in {"prose", "both"} — render prose. Need decl_id + the current
    # player's initial hand for the ATTENTION annotations. build_gus_inputs
    # already computed these; re-derive cheaply rather than plumb through the
    # forward result (keeps _BeliefForwardResult lean).
    current_player = _abs_current_player(game_state)
    game_hands = [[int(d) for d in game_state.hands[p]] for p in range(4)]
    prose = _format_prose(
        out,
        decl_id=int(game_state.decl_id),
        game_hands=game_hands,
        current_player=current_player,
        had_prior=had_prior,
    )
    if format == "prose":
        return prose
    out["prose"] = prose
    return out
