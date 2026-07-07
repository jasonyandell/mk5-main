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

- A cheap next probe (also flagged in the writeup): rerun under the trimmed primer + tool-nudge EQ-gate to force `eq_outcome_distribution` exposure and directly test the downstream hint→probe chain.
- Prototype Candidate C (`what_would_change_my_mind`) since it sits on the tool menu before `eq_outcome_distribution` and dodges the upstream-crowd-out issue entirely.
