# Follow-ups: iter5-e2-candlewax-null

Reviewed against code on 2026-07-07.

## Corrections

- page said the source was `burl/experiments/iter5_e2_candlewax_writeup.md`; the file is actually `burl/experiments/iter5_e2_candlewax_eval_writeup.md` (evidence: ls of burl/experiments/).

## Verified

- T11 smoke (8/10 non-unimodal with populated rationales), T12 (base Gemma N=10 with `--enable-rules-tools`, 0 `eq_outcome_distribution` calls, rules-tools crowd-out), and candlewax field names all match `burl/experiments/iter5_e2_candlewax_eval_writeup.md`.
- E3 rollout numbers (N=500, 98 eq calls, 74 non-unimodal, 53 mixed-mode, 0 `conditional_outcome`) match commit message 1efb9c5 (wiki/sources/1efb9c5.md).

## Not verifiable in repo

- E3 raw traces (`burl/eval/results/e2|e3`) are not in the repo; the "three traces show Gemma considering `conditional_outcome` then declining" claim rests on commit-era artifacts only.

## Follow-ups

- Writeup follow-up #2 still open: stack candlewax + the EQ-gate's `tool-nudge` variant on the stubborn non-match T12 decisions to force `eq_outcome_distribution` exposure even under rules-as-tools (writeup lines 395-400; no later commit runs this).
- Writeup follow-up #4 still open: tune the 0.04 prominence threshold on an N=50 stability sweep (current knee came from a 10-play smoke).

## Review (second pass, 2026-07-07)

- Filename correction stands: `burl/experiments/iter5_e2_candlewax_eval_writeup.md` exists; no `iter5_e2_candlewax_writeup.md` (ls of burl/experiments/).
- Page numbers independently re-verified: T11 8/10 non-unimodal with populated `suggested_counterfactuals`, T12 (base Gemma N=10, `--enable-rules-tools`, 0 eq calls, crowd-out to `trick_winner_if`/`is_legal`), and the four candlewax field names all match the writeup; E3 N=500 / 98 eq / 74 non-unimodal / 53 mixed-mode / 0 `conditional_outcome` match commit messages 1efb9c5 and ceca203.
- The "three traces" claim is better-sourced than the followup implied: both ceca203 and 1efb9c5 commit bodies state it explicitly ("Three thought-prose mentions where the model considers the tool then declines"). Raw E3 traces are still not in the repo, so it remains commit-attested rather than artifact-attested.
- Amended the Follow-ups section: (a) the original first suggestion conflated writeup lever #1 (trimmed-primer rerun) with lever #2 (tool-nudge EQ-gate) — lever #1 was already executed as the E3 N=500 rollout plus the trimmed-primer variant recorded in ceca203's commit body, so only the EQ-gate stack remains open; (b) dropped "prototype Candidate C" — `what_would_change_my_mind` was implemented in commit 7321952 (`burl/tools/meta_tools.py`, see wiki/sources/7321952.md).
