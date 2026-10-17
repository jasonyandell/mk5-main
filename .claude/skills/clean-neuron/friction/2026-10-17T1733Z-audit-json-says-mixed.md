<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

**Doing:** reading `audit.json` for #196.

**Where the skill failed:** the skill says never to call a unit "mixed", but the audit's own
output uses `"kind": "mixed"` and a `shares.mixed` count (`lem/probe`). Anyone who reads the JSON
picks up the word from the tool. The paper's §6 column had done exactly that.

**What I did instead:** wrote "not named in list vN" in the result.

**Change:** rename the kind in `lem.probe audit` (for example `unnamed`), with a note that old
evidence on HF says `mixed`. Until then, the skill's table should say what the audit's three
kinds mean: clean ≥ 0.9; partial means the best label is in [0.7, 0.9);
`mixed` means no label reaches 0.7.
