"""Validate a model's scratchpad claims against engine ground truth.

Parses structured scratchpad output (HAND/VOIDS/COUNTS/PLAY) and checks
each claim against the narration's known game state. Returns a validation
result with per-field pass/fail and an overall verdict.

Only traces where ALL checks pass should be used for training.
"""

from __future__ import annotations

import re


# The five count dominoes and their point values.
COUNT_DOMINOES = {
    "5-5": 10, "6-4": 10, "5-0": 5, "4-1": 5, "3-2": 5,
}


def _parse_domino_list(text: str) -> set[str]:
    """Extract all H-L domino references from a string."""
    doms = re.findall(r"\b(\d-\d)\b", text)
    # Normalize: always high-low
    normalized = set()
    for d in doms:
        a, b = d.split("-")
        hi, lo = max(int(a), int(b)), min(int(a), int(b))
        normalized.add(f"{hi}-{lo}")
    return normalized


def _parse_voids(text: str) -> dict[str, set[str]]:
    """Parse VOIDS line into {player_ref: {suit, ...}}.

    Handles formats like:
      Player 2: fives, blanks
      Player 0: sixes; Player 2: fives
      none observed
    """
    voids: dict[str, set[str]] = {}
    if "none" in text.lower():
        return voids

    suit_names = {"blanks", "ones", "twos", "threes", "fours", "fives", "sixes", "trump"}

    # Split by Player references
    parts = re.split(r"(Player \d|Your partner|You)", text)
    current_player = None
    for part in parts:
        part = part.strip().strip(":").strip()
        if re.match(r"Player \d|Your partner|You", part):
            current_player = part
            continue
        if current_player:
            suits = set()
            for word in re.split(r"[,;.\s]+", part.lower()):
                if word in suit_names:
                    suits.add(word)
            if suits:
                voids[current_player] = suits
            current_player = None

    return voids


def _parse_counts(text: str) -> dict[str, str]:
    """Parse COUNTS line into {domino: 'played'|'out'}.

    Handles formats like:
      5-5: played, 6-4: still out, 5-0: played, 4-1: still out, 3-2: still out
      5-5 played in trick 1, 6-4 still out
    """
    result = {}
    for dom in COUNT_DOMINOES:
        # Look for the domino followed by played/out/still/taken/captured
        pattern = rf"{re.escape(dom)}[:\s]*(\w[\w\s]*?)(?=[,;.]|\d-\d|$)"
        match = re.search(pattern, text)
        if match:
            status_text = match.group(1).strip().lower()
            if any(w in status_text for w in ("played", "taken", "captured", "won")):
                result[dom] = "played"
            elif any(w in status_text for w in ("out", "remain", "still", "unplayed", "live")):
                result[dom] = "out"
    return result


def _parse_play(text: str) -> str | None:
    """Extract the PLAY domino."""
    doms = _parse_domino_list(text)
    if len(doms) == 1:
        return doms.pop()
    # If multiple, take the last one mentioned
    all_doms = re.findall(r"\b(\d-\d)\b", text)
    if all_doms:
        a, b = all_doms[-1].split("-")
        return f"{max(int(a),int(b))}-{min(int(a),int(b))}"
    return None


def parse_scratchpad(response: str) -> dict:
    """Parse structured scratchpad from model response.

    Returns dict with raw parsed fields. Returns None for unparseable fields.
    """
    result = {"hand": None, "voids": None, "counts": None, "play": None}

    # Extract sections by label
    for label, key in [("HAND", "hand"), ("VOIDS", "voids"),
                        ("COUNTS", "counts"), ("PLAY", "play")]:
        pattern = rf"{label}\s*:\s*(.+?)(?=\n(?:HAND|VOIDS|COUNTS|PLAY)\s*:|<|$)"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()

    # If labels not found, try to find play from the whole response
    if result["play"] is None:
        # Fallback: look for "Play: X-Y" or "I play X-Y" anywhere
        play_patterns = [
            r"[Pp]lay[:\s]+(?:the\s+)?(\d-\d)",
            r"I (?:would |will |should )?play (?:the )?(\d-\d)",
            r"\*\*(\d-\d)\*\*",
        ]
        for pat in play_patterns:
            m = re.search(pat, response)
            if m:
                result["play"] = m.group(1)
                break

    return result


def validate_scratchpad(
    parsed: dict,
    true_hand: list[str],
    true_voids: dict[int, set[int]],
    played_dominoes: set[str],
    legal_actions: list[str],
    narrator: int,
    partner: int,
) -> dict:
    """Validate parsed scratchpad against engine ground truth.

    Args:
        parsed: Output of parse_scratchpad().
        true_hand: Narrator's actual remaining dominoes as ["H-L", ...].
        true_voids: {player_id: {suit_id, ...}} from engine void inference.
        played_dominoes: Set of "H-L" strings already played.
        legal_actions: List of "H-L" strings that are legal plays.
        narrator: Narrator player index.
        partner: Partner player index.

    Returns dict with:
        - per-field results (hand_ok, voids_ok, counts_ok, play_ok, play_legal)
        - overall 'valid' bool (all fields correct)
        - 'play' extracted domino string or None
        - 'errors' list of human-readable error descriptions
    """
    errors = []
    true_hand_set = set(true_hand)

    # --- Hand check ---
    hand_ok = False
    if parsed["hand"]:
        claimed_hand = _parse_domino_list(parsed["hand"])
        if claimed_hand == true_hand_set:
            hand_ok = True
        else:
            missing = true_hand_set - claimed_hand
            extra = claimed_hand - true_hand_set
            if missing:
                errors.append(f"hand missing: {missing}")
            if extra:
                errors.append(f"hand extra: {extra}")
    else:
        errors.append("hand not found in scratchpad")

    # --- Voids check ---
    voids_ok = False
    if parsed["voids"]:
        claimed_voids = _parse_voids(parsed["voids"])

        suit_id_to_name = {0: "blanks", 1: "ones", 2: "twos", 3: "threes",
                           4: "fours", 5: "fives", 6: "sixes", 7: "trump"}

        def _player_ref_to_id(ref: str) -> int | None:
            m = re.match(r"Player (\d)", ref)
            if m:
                return int(m.group(1))
            if ref == "Your partner":
                return partner
            if ref == "You":
                return narrator
            return None

        # Build claimed voids as {player_id: {suit_name, ...}}
        claimed_by_id: dict[int, set[str]] = {}
        for ref, suits in claimed_voids.items():
            pid = _player_ref_to_id(ref)
            if pid is not None:
                claimed_by_id[pid] = suits

        # Build true voids as {player_id: {suit_name, ...}}
        true_by_id: dict[int, set[str]] = {}
        for pid, suit_ids in true_voids.items():
            true_by_id[pid] = {suit_id_to_name.get(s, f"suit{s}") for s in suit_ids}

        # Check: every true void must be claimed (no missed voids)
        # We allow extra claimed voids (model may infer more than our tracker)
        all_true_found = True
        for pid, true_suits in true_by_id.items():
            claimed = claimed_by_id.get(pid, set())
            missed = true_suits - claimed
            if missed:
                errors.append(f"Player {pid} void in {missed} but model didn't claim it")
                all_true_found = False

        # Check: no false voids for the narrator (we know narrator's hand)
        narrator_claimed = claimed_by_id.get(narrator, set())
        if narrator_claimed:
            errors.append(f"model claims narrator is void in {narrator_claimed} (suspicious)")
            all_true_found = False

        voids_ok = all_true_found
    else:
        # "none observed" is valid if there are truly no voids
        if not true_voids or all(len(s) == 0 for s in true_voids.values()):
            voids_ok = True
        else:
            errors.append("voids not found but true voids exist")

    # --- Counts check ---
    counts_ok = False
    if parsed["counts"]:
        claimed_counts = _parse_counts(parsed["counts"])
        all_correct = True
        for dom, pts in COUNT_DOMINOES.items():
            actually_played = dom in played_dominoes
            claimed = claimed_counts.get(dom)
            if claimed is None:
                errors.append(f"count domino {dom} not mentioned")
                all_correct = False
            elif claimed == "played" and not actually_played:
                errors.append(f"{dom} claimed played but still out")
                all_correct = False
            elif claimed == "out" and actually_played:
                errors.append(f"{dom} claimed out but was played")
                all_correct = False
        counts_ok = all_correct
    else:
        errors.append("counts not found in scratchpad")

    # --- Play check ---
    play = None
    play_legal = False
    if parsed["play"]:
        play = _parse_play(parsed["play"])
    if play is None:
        # Fallback: try the whole response
        all_doms = re.findall(r"\b(\d-\d)\b", parsed.get("play", ""))
        if all_doms:
            a, b = all_doms[-1].split("-")
            play = f"{max(int(a),int(b))}-{min(int(a),int(b))}"

    if play:
        play_legal = play in legal_actions
        if not play_legal:
            errors.append(f"play {play} not in legal actions {legal_actions}")
    else:
        errors.append("could not parse play from scratchpad")

    valid = hand_ok and voids_ok and counts_ok and play_legal

    return {
        "valid": valid,
        "hand_ok": hand_ok,
        "voids_ok": voids_ok,
        "counts_ok": counts_ok,
        "play_ok": play is not None,
        "play_legal": play_legal,
        "play": play,
        "errors": errors,
    }
