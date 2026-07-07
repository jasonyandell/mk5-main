Reviewed against code on 2026-07-07 — no issues found.

- Verified: +532 tokens/row, ~130 rows x 3 epochs ~ 200K tokens, rank 16 / lr 1e-4 / B200 (burl/train/star_iter4_thoughts.py, commit 20f4fa2), and the truncation reframe numbers (median 2054, max 4210, TRL default 1024) match commit edf86e9.
- Follow-up: once the iter-5 max_seq_length=4096 A/B lands, update the status line here (page still says "retired" pending that result).
- "Byte-identical output on the held-out eval set" was not re-verified against a stored eval artifact in-repo; if the eval JSON exists, linking it from the page would make the null claim auditable.
