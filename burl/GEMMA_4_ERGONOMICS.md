# Gemma 4 E2B — Ergonomics probe for Burl

Goal: find out what Gemma 4 E2B actually wants to emit, and why Move 3's 3/3
rollouts exhausted retries producing `turns=[]`. Answer: our XML protocol is
not reaching the model because Gemma 4's native **thinking channel** eats the
full 512-token budget before it ever starts emitting the answer section.

Probes: 6 calls, $0.08 total, 2026-04-19. Raw outputs saved at
`scratch/burl_gemma_ergo_results.json`.

---

## 1. Canonical Gemma 4 format (from sources)

Authoritative sources (all Google-owned or vLLM recipes):

- [Gemma 4 Prompt Formatting](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)
- [Function calling with Gemma 4](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4)
- [vLLM Gemma 4 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html)
- [HF blog: Welcome Gemma 4](https://huggingface.co/blog/gemma4)

### Special tokens

| Class | Tokens |
|---|---|
| Turn delimiters | `<\|turn>` … `<turn\|>` |
| Roles | `system`, `user`, `assistant` (HF API; `model` in raw wire format) |
| Thinking | `<\|think\|>` (opt-in trigger), `<\|channel>thought` … `<channel\|>` (model-emitted reasoning block) |
| Tools — declare | `<\|tool>` … `<tool\|>` |
| Tools — call | `<\|tool_call>call:NAME{param:<\|"\|>VALUE<\|"\|>}<tool_call\|>` |
| Tools — response | `<\|tool_response>response:NAME{…}<tool_response\|>` |
| String delimiter (inside structured blocks) | `<\|"\|>` |

### Canonical multi-turn tool-use cycle

1. System turn carries `<|think|>` (opt in to thinking) **and** a `<|tool>…<tool|>` block declaring available functions as Python signatures.
2. User turn asks the question.
3. Model generates: `<|channel>thought\n…reasoning…<channel|>` then (optionally) `<|tool_call>call:NAME{args}<tool_call|>`.
4. Host executes the tool, appends an assistant message with `tool_calls=[…]` + `tool_responses=[…]` fields; the chat-template renders it as `<|tool_response>response:NAME{…}<tool_response|>`.
5. Model continues; emits final answer or another tool call. Thoughts from prior turns are **stripped** except within the same multi-tool-call cycle.

Gemma 4 is trained to emit this specific shape. `processor.parse_response(text)` on `transformers`-side returns `{thinking, content, tool_calls}`.

---

## 2. What our endpoint actually emits (raw from Modal, 2026-04-19)

Config: vLLM 0.19.0, `enforce_eager=True`, `enable_thinking=True`, `max_tokens=512`, temp 0.6 unless noted.

### Probe 1 — baseline "Say hello"

```
Hello there!
```

12 chars, no thinking preamble. Simple prompts bypass the thinking channel.

### Probe 2 — "What is 17 + 25? Think step by step" (max_tok=256)

```
thought
Thinking Process:
1.  **Analyze the request:** …
2.  **Perform the calculation (Step-by-step):**
    …
    *   $17 + 20 = 37$.
    *   $37 + 5 = 42$.
    Method 2: Standard addition (Adding ones and tens).
        *   Ones column: $7 + 5 = 12$.
        *   Tens column: $10 + 20 = 30$.
```

Cut off at 256 tokens **still inside the thinking channel**. No answer emitted. The literal token `thought` at the top is the residue of `<|channel>thought` with the special-token bracket stripped by vLLM's detokenizer — the `<channel|>` closing token never appears because the model hadn't finished thinking.

### Probe 3 — Burl XML protocol, 765-byte prompt, temp 0.6

Full 512-token output, every token inside the thinking channel:

```
thought
Burl, a Texas 42 dominoes agent, needs to decide the next play.
…
Goal: Play the best domino to maximize the chance of winning the trick …
…
1.  **0(0-0)**: A low domino. …
2.  **13(5-1)**: A medium domino. …
3.  **20(6-1)**: A high domino. …

Without knowing the other players' hands or the current state of the trick …
```

**Zero `<tool>` tags. Zero `<commit>` tag. Zero content on the answer side of the channel.** Exactly matches Move 3's empty-turn failure mode.

### Probe 4 — Gemma-4-native tool format, 510-byte prompt

Prompt wraps tool signatures in `<|tool>…<tool|>` and includes `<|think|>`:

```
thought
Thinking Process:
…
8.  **Formulate the Tool Call:** Call `trump_declared()`.call: trump_declared()
```

The last 20 characters are telling: the model started to emit Gemma 4's native call syntax (`call: trump_declared()`) on its very last token before the 512 cap. It was about to produce `<|tool_call>call:trump_declared(){}<tool_call|>` — but ran out of budget.

### Probe 5 — Burl XML, full 1811-byte prompt, temp 0.6

Same pattern. 489 output tokens, all thinking channel, never closes. No tools/commit.

### Probe 6 — Burl XML, 765-byte prompt, temp **0.0** (greedy)

Same pattern. Thinking channel rambles 512 tokens without emitting tools or commit. Greedy decoding does **not** help.

---

## 3. Delta between "what Gemma wants" and "what Burl sends"

| Aspect | Gemma 4 native | Burl current | Verdict |
|---|---|---|---|
| Thinking trigger | `<\|think\|>` in system prompt | No trigger, but `apply_chat_template(enable_thinking=True)` on every call | **Thinking is unconditionally ON** |
| Thinking channel | Model emits `<\|channel>thought…<channel\|>` | Parser ignores — expects `<think>…</think>` (never arrives) | **Protocol mismatch** |
| Tool-call syntax | `<\|tool_call>call:NAME{k:<\|"\|>v<\|"\|>}<tool_call\|>` | `<tool>{"name":"NAME","args":{…}}</tool>` | **Protocol mismatch** |
| Tool declarations | `<\|tool>` block with Python signatures + `tools=[…]` kwarg to `apply_chat_template` | Plain English tool menu inside user turn | **Not using native binding** |
| Tool response | Rendered from `tool_responses=[…]` → `<\|tool_response>…<tool_response\|>` | `<observation tool="NAME">{…}</observation>` inline string | **Protocol mismatch** |
| Commit | `<commit>INT</commit>` (Burl-specific) | `<commit>INT</commit>` | OK — custom but legal |
| Max tokens per turn | Thinking alone consistently hits 200-500+ tokens | `max_tokens=512` default in `gemma_serve.py` | **Undersized; thinking alone exceeds it** |

The critical mismatch is that Burl's output protocol is XML-tag that Gemma 4 **isn't** trained to emit. Probe 3 shows the model defaults to its trained behavior — the native thinking channel — and never reaches the emission layer where Burl's parser is listening.

---

## 4. Recommendations, ranked by impact on Move 3's 3/3 exhaustion rate

### R1 (must-do, single-line change) — turn thinking **off** for tool-use calls

**File:** `burl/modal/gemma_serve.py` line 93.
**Change:** `enable_thinking=True` → `enable_thinking=False`.
**Why:** Without the thinking channel eating the token budget, the model will emit answer-channel text immediately — which is where our `<tool>` and `<commit>` XML tags can surface. The HF docs confirm E2B still emits empty `<|channel>thought\n<channel|>` tags when thinking is off; those are special tokens stripped by default. The model's remaining behavior (which it IS trained on) is to honor formatting instructions in the user turn.

**Probability this alone fixes Move 3's exhaustion**: high. The failure mode is "no tool/commit in output" and thinking is the reason no tool/commit appears.

**Risk**: Thinking is genuinely helpful for the reasoning Burl wants (counterfactual distributions, assumption chaining). If disabling it lowers decision quality, re-enable it *only after R3* (below) which makes tool-call reachability independent of thinking-channel length.

### R2 (parser safety net) — accept Gemma 4's native tool-call syntax too

**File:** `burl/harness/tool_loop.py` lines 41-43.
**Change:** Add a second regex for `<\|tool_call>call:(\w+)\(([^)]*)\)<tool_call\|>` and translate to the `{name, args}` tuple.
**Why:** Probe 4's final tokens were `call: trump_declared()` — the native syntax leaking through. With R1 disabling thinking the model will often emit our XML tags, but under harder questions it's likely to revert to trained behavior. Accepting both formats costs one regex and half an hour.
**Risk**: None — if the regex doesn't match, existing `<tool>…</tool>` regex fires.

### R3 (principled fix, larger) — use the native tool-use chat template

**Files:** `burl/modal/gemma_serve.py` (add `tools=[…]` support), `burl/harness/agent_runner.py::render_system_prompt` (drop the hand-rolled tool menu), `burl/harness/tool_loop.py` (parse `<|tool_call>…<tool_call|>` as primary path, translate observations to `tool_responses`).
**Why:** Gemma 4 was post-trained on the native shape. Every deviation costs reliability — more so under distribution shift as prompts get longer (probe 5 was worse than probe 3). Using the native shape gets us a free parser via `processor.parse_response()` and aligns STaR fine-tuning data with how the base model wants to be talked to.
**Scope**: half-day of work including a parser rewrite. Save for Move 4 if R1 alone unblocks Move 3.

### R4 (tactical) — raise `max_tokens` on *thinking-enabled* calls

If we want to keep thinking mode, bump `gemma_serve.py` `max_tokens` default from 512 → 2048. Probe 2 (256-tok cap) cut off mid-reasoning; probe 3 (512-tok cap) did too. Thinking mode consistently wants 300-500 tokens just for the reasoning block; 512 leaves nothing for the answer. Cost scales linearly — 4× tokens is 4× time. R1 makes this unnecessary for most calls.

### R5 (prompt hygiene) — move protocol instructions into system role

Move the `Output protocol` block from the user turn into a system turn, and inject `<|think|>` explicitly if/when you want thinking. This matches how Gemma 4 expects to see global instructions + thinking opt-in. Low-impact but aligns us with the documented shape.

---

## 5. Verdict on Move 3's 3/3 exhaustion

**Type**: Prompt + server-config bug, not parser bug, not model-capability ceiling.

**Evidence**:
- Probes 3/5/6 show Gemma 4 never emits `<tool>` or `<commit>` when thinking is enabled with a 512-token cap. This matches `turns=[]` exactly: each harness turn gets a thinking-only output, `parse_completion` finds no tool specs and no commit, turn counter advances, `max_turns=8` hits → `RetryExhausted`.
- Probe 1 (simple prompt, no thinking triggered) returned a clean short answer. Model is functional.
- Probe 4 showed the model beginning to emit Gemma's native `call: NAME()` syntax mid-reasoning, confirming the model is both willing and trained to call tools — just not *these* tools, not in *this* syntax, not inside *this* token budget.

**Minimum fix to re-run Move 3**: R1 alone (one-character edit to `enable_thinking`).
**Recommended fix before re-run**: R1 + R2 (one flag flip + one regex).
**Right fix eventually**: R3 for Move 4.

---

## 6. Budget

Modal spend: **$0.08 / $0.30 cap**. Two cold starts (~56 s each) drove most of the cost. Warm per-call ≈ 17 s at 28.9 tok/s on L4. Raw outputs at `scratch/burl_gemma_ergo_results.json`.
