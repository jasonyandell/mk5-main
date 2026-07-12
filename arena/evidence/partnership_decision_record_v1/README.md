# Partnership decision record v1 evidence

This is the retained integration smoke for the architecture-neutral decision
exporter. It is an instrument receipt, not a policy comparison: both sides use
the same canonical C0 stack.

```bash
python -m arena.cli \
  --team-a 'margin:wp,model=champion/margin_net_r8.pt+lens:ev' \
  --team-b 'margin:wp,model=champion/margin_net_r8.pt+lens:ev' \
  --n-games 2 --n-samples 10 --marks-to-win 7 \
  --device cpu --no-fast-batching --base-seed 20260711 \
  --out-dir arena/evidence/partnership_decision_record_v1 \
  --emit-decisions \
    arena/evidence/partnership_decision_record_v1/decisions.jsonl.gz
```

The run starts from clean commit `bc4eb386`. It completed 2 games / 23 hands,
then replayed all 644 play decisions into deterministic gzip. The manifest
records:

- code status `clean_commit` at full SHA
  `bc4eb3864a188631cbfaf9f890e1f74b62f643aa`;
- bidder head `champion/margin_net_r8.pt` at SHA-256
  `29ec29e4234c691adb1a7187d61d3f6a1b4a22d6bd41dd1cd0d0e5718d008e35`;
- player oracle `domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt` at SHA-256
  `5f3f00de6215a28bcff76fc80844b53b595f4e98e6bc6ef4c53928bd73b325c1`;
- sampler `uniform-completion-dp-v1`, CPU, 10 worlds; and
- decision artifact SHA-256
  `7c07e026094660e984934f77062277de9dedcf60500b52c7105c0e5c1b440356`.

`base_seed` and each row's raw deal provenance are offline-only. Record and
trajectory IDs are grouping-only. The three online-safe state identities are
the rules, actor-information, and decision-context IDs named in the manifest.

Files:

- `summary.json`, `per_game.csv`, `per_hand.csv` — ordinary Arena result;
- `decisions.jsonl.gz` — 644 canonical decision records; and
- `decisions.jsonl.gz.manifest.json` — counts, hashes, policy provenance, and
  leakage boundary.
