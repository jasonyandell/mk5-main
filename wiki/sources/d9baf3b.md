---
title: "Source digest: d9baf3b — foundational tool harness + first-contact findings"
kind: source
first_seen: 2026-04-18
last_updated: 2026-04-18
status: active
---

## Commit

- **SHA:** d9baf3b6bda0a953f55be51bf81ade3ca2dedde3
- **Date:** 2026-04-18
- **Author:** Jason Yandell

> feat(burl): foundational tool harness + first-contact findings
>
> Four load-bearing pieces landed via parallel agents:
>
> tools/
>   - eq_distribution.py — E[Q] N=10 outcome PDF as Burl's belief primitive.
>     Counterfactual shift validated on seed 900013 (Δmean +15, p_make 0.6→1.0).
>     290ms/play, 49.8 MB peak VRAM. Replaces Zeb after calibration eval below.
>   - engine.py — is_legal/is_trump/unseen/void_audit/trump_declared as thin
>     wrappers over forge/oracle/tables + forge/eq/voids. Duck-typed state.
>   - zeb.py — parked but present. Default checkpoint fixed from untrained
>     large-belief-bootstrap.pt (std 0.036) to lb-v-eq-3740-bootstrap.pt.
>
> harness/
>   - tool_loop.py + retry.py + trace.py — ReAct think/act/observe loop with
>     XML-tag parser (<think>, <tool>, <commit>), illegal-retry with traces,
>     stable JSON schema for STaR training corpora. RetryExhausted raises on
>     cap — silent fallback would mask Move 3's actual failure mode.
>
> eval/
>   - belief_calibration.py — surfaced that Zeb's advertised 72% top-1 is
>     inflated by already-played dominoes. Hidden-only top-1 ≈ 39%, Brier
>     0.224, ECE 0.067. This pivoted Burl away from Zeb as belief tool.
>
> Gemma 4 base (no fine-tune) emits <tool>{"name":"..."}</tool> XML zero-shot —
> parses cleanly via the harness regex. Burl's premise survives first contact.
>
> Beads: t42-56gu, t42-oyq9, t42-jljb, t42-2ap0, t42-ozju (all closed).
> Founding epic t42-14h4 remains open.

## Files introduced

| Path | LOC | Purpose |
|---|---|---|
| `burl/tools/eq_distribution.py` | 593 | E[Q] N=10 outcome PDF — belief primitive; counterfactual validated |
| `burl/tools/engine.py` | 307 | 5 epistemic tools: `is_legal`, `is_trump`, `unseen`, `void_audit`, `trump_declared` — thin wrappers over `forge/oracle/tables` + `forge/eq/voids` |
| `burl/tools/zeb.py` | 401 | [[zeb]] wrapper — parked; checkpoint fixed to trained `lb-v-eq-3740-bootstrap.pt` |
| `burl/harness/tool_loop.py` | 249 | ReAct loop with XML-tag parser |
| `burl/harness/retry.py` | 58 | Retry logic; `RetryExhausted` carries `.trace` for diagnosis |
| `burl/harness/trace.py` | 78 | Stable JSON trace schema for STaR corpora |
| `burl/eval/belief_calibration.py` | 992 | Zeb calibration eval — hidden-only vs all-28 accuracy separation |

## Key outcomes

- **Zeb parked:** hidden-only top-1 ~39% (vs advertised 72%). See [[decisions/zeb-parked-eq-primitive]] and [[experiments/zeb-calibration-eval]].
- **E[Q] distribution validated:** seed 900013 counterfactual shows Δmean +15 and p_make 0.6→1.0 on a deliberate play shift.
- **Gemma 4 XML zero-shot confirmed:** base model emits `<tool>{"name":"..."}</tool>` without fine-tuning — harness regex parses cleanly. Premise survives first contact.

## Beads closed

t42-56gu (zeb wrapper), t42-oyq9 (engine+harness), t42-jljb (belief calibration), t42-2ap0 (eq_distribution), t42-ozju (Gemma warmup). Founding epic t42-14h4 remains open.

## Related pages

[[burl]] · [[zeb]] · [[experiments/zeb-calibration-eval]] · [[decisions/zeb-parked-eq-primitive]] · [[tool-orchestration]] · [[forge]] · [[sources/4b3ba3d]]
