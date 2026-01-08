from __future__ import annotations

from dataclasses import dataclass


PIP_TRUMP_IDS = tuple(range(7))
DOUBLES_TRUMP = 7
DOUBLES_SUIT = 8
NOTRUMP = 9

N_DECLS = 10

DECL_ID_TO_NAME: dict[int, str] = {
    0: "blanks",
    1: "ones",
    2: "twos",
    3: "threes",
    4: "fours",
    5: "fives",
    6: "sixes",
    DOUBLES_TRUMP: "doubles-trump",
    DOUBLES_SUIT: "doubles-suit",
    NOTRUMP: "notrump",
}

DECL_NAME_TO_ID: dict[str, int] = {v: k for k, v in DECL_ID_TO_NAME.items()}
DECL_NAME_TO_ID |= {str(k): k for k in DECL_ID_TO_NAME.keys()}
DECL_NAME_TO_ID |= {
    "no-trump": NOTRUMP,
    "nt": NOTRUMP,
    "doubles": DOUBLES_TRUMP,
    "dt": DOUBLES_TRUMP,
    "ds": DOUBLES_SUIT,
}


def has_trump_power(decl_id: int) -> bool:
    return decl_id in PIP_TRUMP_IDS or decl_id == DOUBLES_TRUMP


@dataclass(frozen=True)
class ParsedDecls:
    decl_ids: tuple[int, ...]


def parse_decl_arg(value: str) -> ParsedDecls:
    v = value.strip().lower()
    if v in ("all", "*"):
        return ParsedDecls(tuple(range(N_DECLS)))

    if v not in DECL_NAME_TO_ID:
        valid = ", ".join(sorted(set(DECL_NAME_TO_ID.keys()) | {"all"}))
        raise ValueError(f"Unknown decl '{value}'. Valid: {valid}")

    decl_id = DECL_NAME_TO_ID[v]
    if not (0 <= decl_id < N_DECLS):
        raise ValueError(f"decl_id out of range: {decl_id}")
    return ParsedDecls((decl_id,))
