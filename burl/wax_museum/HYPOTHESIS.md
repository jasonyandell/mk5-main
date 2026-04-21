# wax_museum — pre-observation hypothesis (2026-04-20)

State the prior before the traces land. Re-read this file after the pilot runs and note what was right, what was wrong, and where reality surprised us. That delta is the signal.

## The premise

Hard-gate Gemma 4 E2B's tool surface so it **cannot** skip the distribution-probing step. `conditional_outcome` was invisible across 145+ decisions (Practicality 4) because the menu offered escape routes — raw PDF, commit without probing. Here, the only tool on turn 1 is `explore_game`. After it fires, `probe_best_case` / `probe_worst_case` appear. Only after ≥1 probe does `commit_play` become callable.

32k context per call instead of the current 2k. So when we say "the thinking channel ate the budget" (Practicality 1), the response can be "we gave it 16× more budget; what did it do with it?"

## What I expect to see

### Turn 1 (only `explore_game` available)
- Gemma calls `explore_game(play=X)` for some candidate. Native tool-call envelope lands cleanly; the harness will treat a missing call as a bail trigger.
- The thought preamble will be 500–2000 tokens of "let me look at the state, my partner holds X, the trumps are Y, I'll try probing play Z first." This is the "plan once, execute" pattern the user expects from E2B.
- **Prior probability it reaches for `explore_game` on turn 1: 85%.** It's the only tool. The risk is Gemma's reflex to emit free-text commentary without a tool call — Practicality 1 showed its native channel can ramble.

### Turn 2 (post-explore; probe_best_case / probe_worst_case / ask_rule visible)
- **This is the load-bearing turn.** If Gemma actually engages with the spike_drivers payload, it picks `probe_worst_case` (humans and Opus 4.7 both lean defensive when shown bimodal distributions — the downside mode feels more actionable).
- **Prior probability of ≥1 probe call across N=5 decisions: 70%.** Hard gate plus in-context spike_drivers should move the number off 0/145. If it doesn't, the hypothesis is dead and we stop.
- Failure mode I expect: Gemma "probes" by exploring a *different play* (calling `explore_game` again) instead of conditioning on a belief. That's still progress — it's using the distribution — but it's not the reasoning shape we want. Count it separately.

### Turn 3+ (commit available)
- Gemma commits. If the probing moved the E[Q], the commit will be different from what a plain unconditional `argmax(E[Q])` would have chosen.
- **Prior on bot-match: 60%.** Lower than iter-3-rules' 90% because the gated surface is alien to the base model. Bot-match is not the thing to grade on — the thing to grade on is "did it call a probe."

### Reasoning legibility at 32k
- I expect thoughts to be verbose and well-structured for the first 2–3 turns, then degrade on turn 4+ as the chat history fills context. The 32k ceiling means we can read the full trace — whether it shows counterfactual reasoning or just pip-value heuristics wearing a probe-call costume is the thing to eyeball.

## What would falsify the premise

- **Silent turn 1 on 2+ decisions**: Gemma emits no tool call with only `explore_game` visible. Means the hard gate itself is alien and the model refuses. Run-level bail triggers here.
- **Zero probe calls across all 5 decisions**: the "gate it so it has to reach" bet is wrong; the model reaches `explore_game` then ritualizes straight to `commit_play` on turn 2. If the 0/145 number stays 0/5 here, the environment-shape approach is exhausted and the problem is a training-data one.
- **All probes are `probe_best_case`**: Gemma is cargo-culting the tool menu, not picking strategically.

## What would confirm it

- ≥1 probe call on ≥3/5 decisions.
- Thoughts on turn 2 reference spike_drivers vocabulary — "the disaster branch," "if partner holds the 5-5," etc.
- A decision where the committed play is *different* from the unconditional argmax because of what the probe revealed.

## Log this back against reality

After running `run_pilot.py`, read `logs/<run_id>/tail.log`, eyeball 2–3 full thoughts in `logs/<run_id>/thoughts/`, and come back here to write:

> **2026-04-??**: Ran N=5. [What happened]. Delta from prior: [what was wrong]. Next try: [what to change].

Keep the prior and the delta both. That's the loop.

## 2026-04-20 — N=3 local (`logs/20260420_201531_local_n3/`)

**What happened**: 65 s wall on M5 Max (MLX-LM, max_tokens=4096). 3/3 committed, **3/3 probed**, 1/3 bot-match, mean Δ = −2.42 Q, 0 bails, 0 gated_commit_leaks.

**Delta from prior:**
- Probe rate: predicted 70% → observed **100%** (3/3). The hard gate landed harder than I expected.
- Candidate-switching: I didn't predict it explicitly, but d0 did it — `explore(13) → probe_worst(13) → explore(21) → commit(21)`, exactly the reasoning shape the candlewax work was targeting. Worth calling out: the environment-shape redesigns (P4) + the hard gate (P9) compose cleanly on this trace.
- Plan-then-execute: **correct** on d0 (lucid turn-1 analysis), **emergent meta-protocol reasoning** on d1 t2 ("I must wait for the output of explore_game…"). At 32 k context the model has room to reason about the protocol, not just the play.
- Turn-1 bail: **correct** — no bails, every turn 1 emitted a real tool call with substantive thought.

**Surprises (negative, worth next iteration):**
- **Hallucinated tool-response values.** d0 t3's summary quotes `mean ≈ 24.5, p_make ≈ 0.65` that aren't in the actual response. The model reads the shape field and paraphrases plausible numbers. If we STaR on these traces, we teach the hallucination.
- **Duplicate adjacent tool calls.** d0 t1 → t2 re-fires `explore_game(13)` before reading the response. Harness artifact or model pathology? Unknown — audit on next run.
- **Thinking-channel leak at `enable_thinking=False`.** d1 t2 raw text begins with `<|channel>thought`. P1 lived again under the new prompt.

**Next try:**
1. Bump to N=10 local to confirm probe rate holds (3/3 is low evidence; could still be a lucky sample).
2. Audit the duplicate-call pattern — is the harness missing the tool response in `messages` between turns, or is Gemma double-emitting in one completion?
3. Add a "quote the tool response verbatim" line to the system prompt and see if the hallucination shrinks.
4. If the probe rate holds on N=10, ship to Modal L4 for N=50 at 32 k context — the full pilot the plan originally targeted.

**Keeping the prior:** I was directionally right. Under-estimated both the lift (70 → 100%) and the richness of reasoning at 32 k. The negative surprises (hallucinations, duplicates) were invisible to me in the prior — needed the traces to see them.
