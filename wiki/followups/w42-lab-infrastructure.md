Reviewed against code on 2026-07-07 — no issues found.

Verified: `w42/raw_public_state_baseline.py` + `w42/wandb_utils.py` implement the described flags (`--wandb/--no-wandb`, `--wandb-mode auto|online|offline|disabled` with auto→online iff logged in), failure capture (`status=failed`, `failure/type|message|traceback_tail`, `status/failed=1`), offline sync-command recording, and default group `t42-csw6`. Commit `489c1fd` and all [[wiki links]] exist.

Not verifiable in-repo (expected, not errors): W&B run ids/URLs (`rwexij8m`, `as3xy7oz`, `wv7pkuco`, `5ychhiid`), `w42/wandb_smoke/` outputs and `gus/data/*.pt` corpora (gitignored local artifacts).

- Minor tension worth a future tidy: the page's run-group schema is `w42-{bead_slug}-{experiment_slug}` but the implemented default group is `t42-csw6`; one convention should win.
