---
title: The License Line — the Science CC0, Zeb MIT
kind: decision
first_seen: 2026-10-14
last_updated: 2026-10-14
status: active
---

## Decision

Jason (2026-10-14): *"everything I made up to and including Walt is the science. cc0."* For
Zeb: *"keep doing what we are doing"*, so Zeb stays MIT. He wants "Powered by Walt" in lights and
his name credited, by desirability rather than force: *"I don't want lawyers."*

As built:

| paths | license |
|---|---|
| everything by default: `src/`, `scenario/` (Walt: rules, record, fiber, the Lean proofs, the research harnesses), `silas/`, `arena/`, `tournament/`, `walt/`, `scripts/`, `wiki/`, `docs/`, the oracle era (`forge/`, `gus/`, `champion/`, `w42/`, `burl/`, `lem/`, `otis/`, `parlor/`), data | CC0-1.0 |
| `zeb/` (the package); `scenario/field.py`, `scenario/field_train.py`, `scenario/core/src/field.rs`; the tests of the field, the posterior, the reading, the auction read and the tie rule | MIT |
| `zeb/net/` (the field net's weights, already CC0 on Hugging Face) | CC0-1.0 |
| `scenario/planner.py`, `scenario/compiled.py`, `scenario/core/src/planner.rs`, `scenario/core/src/python.rs`, `arena/walt_play.py`: Walt with Zeb's layer woven through | CC0-1.0 AND MIT, until #158 splits them |
| the K-Dense skills and the Anthropic pdf skill under `.claude/skills/`, `.agents/skills/` | LicenseRef-third-party (not ours) |

Every code file in the live trees carries an SPDX header. `REUSE.toml` covers the rest: data,
docs, the wiki, generated files, and the finished oracle-era trees, which stay untouched as
archaeology. `reuse lint` is the `reuse` suite of `scripts/check`. `LICENSE` draws the line in
words. One sentence reserves the names: the code is free (CC0 / MIT); the names Walt, Zeb, and
Plunge are not licensed as the name of your product. The credit line is a request, never a
condition: "Powered by Walt — Jason Yandell's scenario player for Texas 42", with the badge
`docs/powered-by-walt.svg`.

## Why

- **The science is a mountain anyone may climb.** The engine, Walt's mathematics and code, and
  the measurements were always meant to be reused, and the data was CC0 from its first
  publication ([[publish-data-to-huggingface]]).
- **Zeb is the product layer.** MIT keeps Jason's name on every copy, the only condition he
  wants, and nothing more.
- **Rejected: a code split now.** Zeb's layer runs through the planner's hot path in both the
  referee and the core, which must stay bit-identical. The split is #158. Until it lands, the
  `AND` marking is honest, and a reader who wants Zeb-free Walt reads the five files as MIT.
- **Rejected: a trademark or a runtime banner.** Names are reserved in one plain sentence. The
  npm package prints nothing.

## Links

[[scenario-player]] · [[silas]] · [[publish-data-to-huggingface]] · [[huggingface-assets]]
