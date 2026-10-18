# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Seat a frozen observer: unchanged visible prefix, legal-output argmax."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from forge.zeb.game import current_player, legal_actions
from scenario.rules import mask_of

from .data import SEAT, VERSION, encode
from .model import Config, Observer


def prefix(state, bid):
    """Only the acting seat's original hand and the public prefix are read."""
    if bid != 30 or state.bid_state.high_bid != 30:
        raise ValueError("observer supports only bid 30; do not relabel other contracts")
    viewer = current_player(state)
    return np.append(encode(mask_of(state.hands[viewer]), viewer, state.bidder,
                            state.decl_id, state.play_history), SEAT)


class ObserverPlay:
    def __init__(self, checkpoint, device="mps"):
        path = Path(checkpoint)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["tokenizer"] != VERSION:
            raise ValueError("checkpoint has a different input encoding")
        self.model = Observer(Config(**saved["config"])).to(device).eval()
        self.model.load_state_dict(saved["model"])
        self.device = device
        self.identity = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "step": saved["step"], "tokenizer": saved["tokenizer"],
                         "data_sha256": saved["data_sha256"], "config": saved["config"],
                         "selection": "highest legal logit; lowest tile id breaks exact ties"}

    def plays(self, decl):
        return decl in (*range(8), 9)

    @torch.inference_mode()
    def choose(self, states, bid_values, marks=None, marks_to_win=7):
        if len(states) != len(bid_values):
            raise ValueError("one bid per state is required")
        if not states:
            return []
        tokens = [prefix(s, bid) for s, bid in zip(states, bid_values)]
        # Right padding is after the queried position, behind the causal mask.
        x = np.zeros((len(tokens), max(map(len, tokens))), dtype=np.int64)
        for i, row in enumerate(tokens):
            x[i, :len(row)] = row
        logits = self.model(torch.as_tensor(x, device=self.device))
        scores = logits[torch.arange(len(tokens), device=self.device),
                        torch.tensor([len(t) - 1 for t in tokens], device=self.device)].cpu().numpy()
        result = []
        for state, row in zip(states, scores):
            hand = state.hands[current_player(state)]
            if not np.isfinite(row).all():
                raise ValueError("nonfinite tile scores")
            legal = sorted(legal_actions(state), key=lambda slot: hand[slot])
            result.append(max(legal, key=lambda slot: row[hand[slot]]))
        return result
