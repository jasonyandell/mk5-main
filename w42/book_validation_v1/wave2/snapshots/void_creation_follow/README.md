# void_creation_follow — Snapshot Corpus

**Wave:** 2  
**Bead:** t42-z31l  
**Claim:** ch05-void-creation (follow-position sub-claim)

## Filter definition

Canonical follow-position void scenario (ch05 setter defense):

1. Player is setter (player 1 or 3).
2. The led suit is non-trump.
3. Setter cannot follow the led suit (no legal tile in led suit).
4. Setter holds exactly 1 tile in some non-led, non-trump suit ("singleton").
5. Setter has at least one alternative legal play from a different suit.
6. Trick number ≥ 1 (no benefit to voiding in the final trick).

## Mining stats

| Stat | Value |
|------|-------|
| Candidates scanned | 7,827 |
| Filter hit rate | 6.39% |
| Snapshots collected | 500 (target: 500) |
| Chunks used | 3 of 100 (corpus_train_chunk_0-99.pt, 100-199.pt, 1000-1099.pt) |
| Invalid snapshots | 0 |

## Round-trip validation

5-snapshot sample verified through `GameStateTensor.from_snapshot`:

```
[0] decl=0 led_suit=3 n_trick_plays=1 GameStateTensor type=GameStateTensor OK
[1] decl=0 led_suit=3 n_trick_plays=3 GameStateTensor type=GameStateTensor OK
[2] decl=1 led_suit=4 n_trick_plays=1 GameStateTensor type=GameStateTensor OK
[3] decl=1 led_suit=5 n_trick_plays=2 GameStateTensor type=GameStateTensor OK
[4] decl=2 led_suit=5 n_trick_plays=3 GameStateTensor type=GameStateTensor OK
```

All 500 snapshots valid (0 invalid).

## Schema

Standard `forge.eq.snapshot.v1` fields plus private keys (`_*`):

- `_source_chunk` — source corpus chunk filename
- `_game_idx` — game index in chunk
- `_decision_idx` — decision index in game
- `_action_taken_slot` — greedy oracle slot taken
- `_e_q` — oracle E[Q] values at this decision
- `_legal_mask` — legal action mask
- `_led_suit` — suit led in the current trick
- `_trick_lead_domino` — domino that led the trick
- `_n_trick_plays` — number of plays before setter's turn

## Files

| File | Description |
|------|-------------|
| `snapshots.jsonl` | 500 snapshots, one JSON object per line |
| `manifest.json` | Provenance, filter definition, SHA256 |
| `build_void_creation_follow_corpus.py` | Mining script |
