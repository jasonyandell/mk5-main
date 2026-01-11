"""Infer void suits from play history."""

from forge.oracle.tables import can_follow, led_suit_for_lead_domino


def infer_voids(plays: list[tuple[int, int, int]], decl_id: int) -> dict[int, set[int]]:
    """Returns {player: {void_suits}} based on failed follows.

    A player is void in a suit if they failed to follow when that suit was led.

    Args:
        plays: List of (player, domino_id, lead_domino_id) tuples
        decl_id: Declaration ID (0-9)

    Returns:
        Dict mapping player (0-3) to set of suits they're void in
    """
    voids: dict[int, set[int]] = {0: set(), 1: set(), 2: set(), 3: set()}

    for player, domino_id, lead_domino_id in plays:
        led_suit = led_suit_for_lead_domino(lead_domino_id, decl_id)

        # If player can't follow the led suit, they're void in it
        if not can_follow(domino_id, led_suit, decl_id):
            voids[player].add(led_suit)

    return voids
