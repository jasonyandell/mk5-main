DESCRIPTION = "Same outcome distribution as explore_game, rendered with Burl's preferred shape: HEADLINE (variance + bid-success rate), MODES (each with center, mass %, catalyst, label), RISK PROFILE, then the original numbers as ground truth. Reuses the explore_game cache so it costs nothing extra if you've already explored this play."

from burl.harness.agent_runner import _DOMINO_LABELS


def _label(d):
    return f"{int(d)}({_DOMINO_LABELS[int(d)]})"


def _variance_label(stdev):
    if stdev < 8:
        return "LOW VARIANCE", "tight outcome distribution; the play is what it is"
    if stdev < 18:
        return "MEDIUM VARIANCE", "outcome shifts noticeably with which dominoes the opponents hold"
    return "HIGH VARIANCE", "outcome is dominated by hidden information; partner's hand swings the result by 30+ Q"


def _mode_label(center):
    if center >= 15:
        return "BIG WIN"
    if center >= 5:
        return "WIN"
    if center >= -5:
        return "NEAR-BREAKEVEN"
    if center >= -15:
        return "LOSS"
    return "DISASTER"


def _seat_word(seat_label, me_abs):
    return {
        "partner": f"PARTNER (seat {(me_abs + 2) % 4})",
        "left_opp": f"LEFT OPP (seat {(me_abs + 1) % 4})",
        "right_opp": f"RIGHT OPP (seat {(me_abs + 3) % 4})",
        "self": "MYSELF",
    }.get(seat_label, seat_label.upper())


def _catalyst_for_mode(mode_center, spike_drivers, me_abs):
    if not spike_drivers:
        return None
    best = None
    best_delta = float("inf")
    for sd in spike_drivers:
        delta = abs(float(sd.get("mode_center", 0.0)) - float(mode_center))
        if delta < best_delta:
            best_delta = delta
            best = sd
    if not best or not best.get("catalysts"):
        return None
    cat = best["catalysts"][0]
    return {
        "seat": _seat_word(cat["seat"], me_abs),
        "domino": int(cat["domino"]),
        "freq": float(cat.get("freq_in_spike", 0.0)),
        "lift": float(cat.get("lift", 1.0)),
    }


def tool(ctx, play, **kwargs):
    play = int(play)
    cache = ctx.get_or_build(play)
    dist = cache.dist
    is_offense = bool(dist.is_offense)
    p_make = float(dist.p_make)
    mean = float(dist.mean)
    stdev = float(dist.stdev)
    shape = dist.distribution_shape
    modes = list(dist.modes or [])
    n = int(dist.n_samples)
    spikes = list(dist.spike_drivers or [])

    var_label, var_explainer = _variance_label(stdev)
    bid_role_phrase = (
        f"makes the bid {p_make * 100:.0f}% of worlds"
        if is_offense
        else f"sets the bidder {p_make * 100:.0f}% of worlds"
    )

    lines = []
    lines.append(f"PLAY {_label(play)}  —  {var_label}, {shape}")
    lines.append(f"  Headline: {bid_role_phrase}; mean Q = {mean:+.1f}, stdev = {stdev:.1f}")
    lines.append(f"  Risk note: {var_explainer}")
    lines.append("")

    sorted_modes = sorted(modes, key=lambda m: float(m.get("mass", 0.0)), reverse=True)
    if sorted_modes:
        lines.append("MODES (sorted by mass; this is where outcomes actually land):")
        for i, m in enumerate(sorted_modes, 1):
            center = float(m.get("center", 0.0))
            mass = float(m.get("mass", 0.0))
            label = _mode_label(center)
            cat = _catalyst_for_mode(center, spikes, ctx.me_abs)
            cat_phrase = (
                f"catalyst: {cat['seat']} holds {_label(cat['domino'])}"
                f" (freq_in_spike={cat['freq']:.2f}, lift={cat['lift']:.2f})"
                if cat else "no clean catalyst (spread across worlds)"
            )
            lines.append(
                f"  Mode {i} [{label}]:  center Q = {center:+.1f}    mass = {mass * 100:.0f}%"
            )
            lines.append(f"          {cat_phrase}")
    else:
        lines.append("MODES: (unimodal; see best/worst probes for the conditional shape)")
    lines.append("")

    if shape == "unimodal":
        risk = "FOCUSED — one cluster, partner's hand barely changes the answer"
    elif shape == "bimodal":
        risk = "TWO-PATH — outcome forks into two distinct futures based on hidden info"
    else:
        risk = "FANNED — multiple outcome clusters; high information dependence"
    lines.append(f"Risk profile: {risk}")

    if cache.best_case_assumption:
        ba = cache.best_case_assumption
        lines.append(
            f"Upside catalyst available: probe_best_case(play={play}) — "
            f"assumes seat {ba['player']} holds {_label(ba['holds'])}"
        )
    if cache.worst_case_assumption:
        wa = cache.worst_case_assumption
        lines.append(
            f"Downside catalyst available: probe_worst_case(play={play}) — "
            f"assumes seat {wa['player']} holds {_label(wa['holds'])}"
        )
    lines.append("")
    lines.append(
        f"(N={n} sampled worlds. Same data explore_game uses; this is a render, "
        f"not a new computation. To compare with another candidate: play_brief(play=Y).)"
    )

    structured = {
        "play": play,
        "is_offense": is_offense,
        "p_make": round(p_make, 3),
        "mean": round(mean, 2),
        "stdev": round(stdev, 2),
        "n_samples": n,
        "shape": shape,
        "variance_label": var_label,
        "modes": [
            {
                "center": round(float(m.get("center", 0.0)), 2),
                "mass": round(float(m.get("mass", 0.0)), 3),
                "label": _mode_label(float(m.get("center", 0.0))),
                "catalyst": _catalyst_for_mode(
                    float(m.get("center", 0.0)), spikes, ctx.me_abs
                ),
            }
            for m in sorted_modes
        ],
        "best_case_assumption": cache.best_case_assumption,
        "worst_case_assumption": cache.worst_case_assumption,
    }
    return {"prose": "\n".join(lines), "structured": structured}
