# Follow-ups: chat-template-fix-validation

Reviewed against code on 2026-07-07.

## Corrections

- Page said the tool-response fix landed "two commits later" than the iter3-rules-adapter's 90% measurement; git says dbadb5f -> 54f7776 is 17 commits (one day) apart (evidence: `git rev-list --count dbadb5f..54f7776`). Note: `wiki/entities/iter3-rules-adapter.md` says "three commits after" — also wrong, not edited per audit scope.

## Verified

- 5/5 bot-match (base Gemma 4 E2B, N=5 held-out trick-6) and Qwen3.6-35B-A3B-4bit multi-thousand-token `<think>` behavior match the 54f7776 commit message and `burl/wax_museum/README.md`.
- Mechanism (tool responses on `assistant.tool_responses=[{name, response}]` because Gemma's template drops `role="tool"`) matches `burl/wax_museum/harness.py`.
- Linked pages (`wiki/decisions/gemma-tool-response-shape.md`, `wiki/entities/iter3-rules-adapter.md`, `wiki/sources/54f7776.md`) exist; iter3 page does carry the era-6 confound caveat as claimed.

## Follow-ups

- Per-decision N=5 result logs (e.g. `burl/wax_museum/logs/n5_qwen/live.log` referenced by the README) are not in git; 5/5 rests on the commit message and README, not raw artifacts.
- Fix the parallel "three commits after" phrasing in `wiki/entities/iter3-rules-adapter.md` in a future pass.

## Review (second pass, 2026-07-07)

- Verified — corrections stand. Independently re-derived: `git rev-list --count dbadb5f..54f7776` = 17; commit dates 2026-04-19 14:06 → 2026-04-20 22:56 ("one day later" is accurate). The 5/5 Gemma result, Qwen 6K-token `<think>` behavior, and the `assistant.tool_responses=[{name, response}]` mechanism all match the 54f7776 commit message and `burl/wax_museum/harness.py` (lines 108, 247-308). All three linked pages exist. Both follow-up suggestions remain valid: `git ls-files burl/wax_museum/logs` is empty, and `wiki/entities/iter3-rules-adapter.md:56` still says "three commits after".
