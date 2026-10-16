# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The probe kit: tools that read a 42 net's neurons in the game's own words.

Stage 1 of the Lem program (wiki/topics/lem-the-teacher.md) audits the field
net Zeb ships, `zeb/net/margin_d250k_h128x64.fnet`: 192 neurons, 384 poles.
Every pole is labelled two ways, exactly by Silas predicates over the
viewer's information (`concepts`), and cheaply by Haiku readers working from
short position cards (`cards`, `readers`). The statistics (`audit`) report
every share with its interval against a shuffled-label baseline. The
pre-registration is wiki/experiments/lem-field-net-audit-2026-10-15.md.

The kit reads a seat's information and nothing else: a concept query is
evaluated on a belief that resolves no hidden tile, so a query that reads a
hidden world is refused by Silas before it can label anything
(wiki/topics/strategy-fusion-fence.md).

    python -u -m lem.probe sample   POUR_DIR OUT.npz        # the audit rows of one pour
    python -u -m lem.probe merge    OUT.npz PART.npz ...    # one sample from the parts
    python -u -m lem.probe concepts SAMPLE.npz OUT.npz      # Silas labels, every row
    python -u -m lem.probe audit    SAMPLE.npz LABELS.npz OUT_DIR   # the pole table
    python -u -m lem.probe cards    SAMPLE.npz AUDIT_DIR OUT_DIR    # cards per pole
    python -u -m lem.probe readers  CARDS_DIR OUT_DIR        # Haiku explain + score
    python -u -m lem.probe ablate   SAMPLE.npz OUT.json      # the neuron-count curve
"""
