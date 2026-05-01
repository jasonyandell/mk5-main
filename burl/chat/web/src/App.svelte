<script lang="ts">
  import { onMount } from "svelte";
  import {
    streamChat,
    health,
    listHarvests,
    listDecisions,
    loadDecision,
    type Message,
    type Harvest,
    type DecisionMeta,
    type LoadedDecision,
  } from "./lib/api";
  import { CompletionParser, type Segment } from "./lib/parse";

  // Display segments (typed). The model still gets a flat messages[] reconstructed
  // from these on each send.
  let segments = $state<Segment[]>([]);
  let input = $state("");
  let streaming = $state(false);
  let modelInfo = $state<{ model_repo?: string; adapter_path?: string | null } | null>(null);
  let scrollEl: HTMLDivElement;

  let harvests = $state<Harvest[]>([]);
  let selectedHarvest = $state<string | null>(null);
  let buckets = $derived<Array<[string, number]>>(
    selectedHarvest
      ? Object.entries(
          harvests.find((h) => h.name === selectedHarvest)?.buckets ?? {},
        ).sort((a, b) => b[1] - a[1])
      : [],
  );
  let selectedBucket = $state<string | null>(null);
  let decisionList = $state<DecisionMeta[]>([]);
  let loadedDecision = $state<LoadedDecision | null>(null);

  // Collapsed state per-segment, keyed by index in the segments[] array
  let collapsed = $state<Record<number, boolean>>({});

  // Toggle: ask the model to emit a <|channel>thought block before answering.
  // Enables visible reasoning for adapters trained on preserve_thoughts.
  let enableThinking = $state(true);
  let maxTokens = $state(1024);

  // Streaming-progress indicator: shows elapsed time while we wait for the
  // first token (prefill on a ~14K-token prefix takes a few seconds).
  let streamStart = $state<number | null>(null);
  let firstTokenAt = $state<number | null>(null);
  let now = $state<number>(performance.now());
  let timerHandle: ReturnType<typeof setInterval> | null = null;

  onMount(async () => {
    try {
      const h = await health();
      if (h.ok) modelInfo = h;
    } catch {}
    try {
      harvests = await listHarvests();
      if (harvests.length > 0) {
        selectedHarvest = harvests[0].name;
        await refreshDecisions();
      }
    } catch (e) {
      console.error("listHarvests failed", e);
    }
  });

  async function refreshDecisions() {
    if (!selectedHarvest) return;
    decisionList = await listDecisions(selectedHarvest, {
      bucket: selectedBucket ?? undefined,
      limit: 60,
    });
  }

  async function pickBucket(b: string | null) {
    selectedBucket = b;
    await refreshDecisions();
  }

  async function pickDecision(d: DecisionMeta) {
    if (!selectedHarvest) return;
    const got = await loadDecision(selectedHarvest, d.global_idx);
    if ((got as any).error) {
      alert("load failed: " + (got as any).error);
      return;
    }
    loadedDecision = got;
    // Chat-mode primer: a synthetic assistant turn that breaks the
    // play-decision rhythm. Without it, adapters trained on STaR play
    // traces snap back to commit_play after every user turn (the training
    // distribution they were distilled on). With it, the most-recent
    // turn establishes a Q&A cadence the model continues.
    const play = got.meta.burl_play;
    const primer = `Yeah, I committed ${play}. The decision is done — ask me anything about it and I'll talk it through with you. No more tool calls.`;
    segments = [
      ...got.segments,
      { kind: "assistant_text", content: primer },
    ];
    collapsed = {};
    segments.forEach((s, i) => {
      if (s.kind === "system") collapsed[i] = true;
    });
  }

  function newSession() {
    loadedDecision = null;
    segments = [];
    collapsed = {};
  }

  /** Build the OpenAI-shaped messages[] the model gets, from current segments. */
  function buildMessagesForModel(): Message[] {
    const out: Message[] = [];
    let curAssistantParts: string[] = [];
    const flushAssistant = () => {
      if (curAssistantParts.length > 0) {
        out.push({ role: "assistant", content: curAssistantParts.join("\n\n") });
        curAssistantParts = [];
      }
    };
    for (const s of segments) {
      if (s.kind === "system") {
        flushAssistant();
        out.push({ role: "system", content: s.content });
      } else if (s.kind === "user") {
        flushAssistant();
        out.push({ role: "user", content: s.content });
      } else if (s.kind === "thinking") {
        curAssistantParts.push(`[thought] ${s.content}`);
      } else if (s.kind === "tool_call") {
        curAssistantParts.push(`[tool_call ${s.tool}] ${JSON.stringify(s.args)}`);
      } else if (s.kind === "tool_result") {
        curAssistantParts.push(`[tool_result ${s.tool}]\n${s.content}`);
      } else if (s.kind === "assistant_text") {
        curAssistantParts.push(s.content);
      } else if (s.kind === "commit") {
        curAssistantParts.push(`[committed play ${s.final_play}]`);
      }
    }
    flushAssistant();
    return out;
  }

  async function send() {
    const content = input.trim();
    if (!content || streaming) return;
    input = "";
    segments = [...segments, { kind: "user", content }];
    streaming = true;
    streamStart = performance.now();
    firstTokenAt = null;
    now = streamStart;
    if (timerHandle) clearInterval(timerHandle);
    timerHandle = setInterval(() => {
      now = performance.now();
    }, 100);

    const messages = buildMessagesForModel();
    const parser = new CompletionParser();
    const anchor = segments.length;
    queueMicrotask(() => scrollEl?.scrollTo(0, scrollEl.scrollHeight));

    try {
      for await (const ev of streamChat(messages, {
        enable_thinking: enableThinking,
        max_tokens: maxTokens,
      })) {
        if (ev.type === "token") {
          if (firstTokenAt === null) firstTokenAt = performance.now();
          parser.feed(ev.text);
          segments = [...segments.slice(0, anchor), ...parser.segments];
          queueMicrotask(() => scrollEl?.scrollTo(0, scrollEl.scrollHeight));
        } else if (ev.type === "done") {
          parser.end();
          segments = [...segments.slice(0, anchor), ...parser.segments];
        } else if (ev.type === "error") {
          segments = [
            ...segments,
            { kind: "assistant_text", content: `[error] ${ev.message}` },
          ];
        }
      }
    } catch (e) {
      segments = [
        ...segments,
        { kind: "assistant_text", content: `[error] ${e}` },
      ];
    } finally {
      streaming = false;
      if (timerHandle) {
        clearInterval(timerHandle);
        timerHandle = null;
      }
    }
  }

  let prefillSecs = $derived(
    streamStart !== null && firstTokenAt === null
      ? ((now - streamStart) / 1000).toFixed(1)
      : null,
  );
  let totalSecs = $derived(
    streamStart !== null
      ? ((now - streamStart) / 1000).toFixed(1)
      : null,
  );

  function onKey(e: KeyboardEvent) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      send();
    }
  }

  function bucketColor(b: string): string {
    if (b.includes("RIGHT") || b === "ALL_AGREE_CORRECT") return "#3a6d4a";
    if (b.includes("BREAKS") || b === "BURL_INDEPENDENT_WRONG") return "#9c4a2a";
    if (b.includes("WRONG") || b.includes("PARROT") || b === "BURL_DRIFTS_FROM_PI") return "#7c3a4a";
    if (b === "FORCED_COMMIT" || b === "ILLEGAL") return "#555";
    if (b === "BOTH_FIX" || b.includes("FIXES")) return "#4a5d8a";
    return "#444";
  }

  function toggleCollapse(i: number) {
    collapsed[i] = !collapsed[i];
  }

  function turnLabel(s: Segment): string {
    if ("turn" in s && s.turn != null) return `t${s.turn}`;
    return "";
  }
</script>

<div class="app">
  <header>
    <strong>burl chat</strong>
    {#if modelInfo}
      <span class="info">
        {modelInfo.model_repo}
        {#if modelInfo.adapter_path}· {modelInfo.adapter_path}{/if}
      </span>
    {:else}
      <span class="info dim">connecting…</span>
    {/if}
    <span class="spacer"></span>
    {#if loadedDecision}
      <span class="info">
        decision #{loadedDecision.meta.global_idx} · seed {loadedDecision.meta.seed}
        · burl {loadedDecision.meta.burl_play} vs oracle {loadedDecision.meta.oracle_play}
        · regret {loadedDecision.meta.burl_regret.toFixed(2)}
      </span>
      <button class="ghost" onclick={newSession}>new session</button>
    {/if}
  </header>

  <main>
    <aside class="sidebar">
      <section>
        <label>harvest</label>
        <select bind:value={selectedHarvest} onchange={refreshDecisions}>
          {#each harvests as h}
            <option value={h.name}>{h.name.replace("harvest_batched_", "")} ({h.n_decisions})</option>
          {/each}
        </select>
      </section>

      <section>
        <label>bucket</label>
        <div class="buckets">
          <button
            class="chip"
            class:active={selectedBucket === null}
            onclick={() => pickBucket(null)}
          >all</button>
          {#each buckets as [b, n]}
            <button
              class="chip"
              class:active={selectedBucket === b}
              style="border-left: 4px solid {bucketColor(b)}"
              onclick={() => pickBucket(b)}
            >{b} <span class="dim">{n}</span></button>
          {/each}
        </div>
      </section>

      <section class="decisions">
        <label>decisions ({decisionList.length})</label>
        <div class="dlist">
          {#each decisionList as d}
            <button
              class="decision"
              class:active={loadedDecision?.meta.global_idx === d.global_idx}
              onclick={() => pickDecision(d)}
            >
              <span class="bar" style="background: {bucketColor(d.bucket)}"></span>
              <span class="dlabel">
                <span>#{d.global_idx} · seed {d.seed}</span>
                <span class="dim">{d.bucket}</span>
                <span class="dim">b={d.burl_play} o={d.oracle_play} r={d.burl_regret.toFixed(2)}</span>
              </span>
            </button>
          {/each}
          {#if decisionList.length === 0}
            <span class="dim small">— pick a harvest + bucket —</span>
          {/if}
        </div>
      </section>
    </aside>

    <section class="chat">
      <div class="conv" bind:this={scrollEl}>
        {#each segments as s, i (i)}
          {#if s.kind === "system"}
            <div class="seg system">
              <div class="seg-head" onclick={() => toggleCollapse(i)} role="button" tabindex="0">
                <span class="badge sys">system</span>
                <span class="title">prompt · {s.content.length} chars</span>
                <span class="caret">{collapsed[i] ? "▸" : "▾"}</span>
              </div>
              {#if !collapsed[i]}
                <div class="seg-body mono">{s.content}</div>
              {/if}
            </div>
          {:else if s.kind === "user"}
            <div class="seg user">
              <span class="badge user">user{turnLabel(s) ? ` · ${turnLabel(s)}` : ""}</span>
              <div class="seg-body mono">{s.content}</div>
            </div>
          {:else if s.kind === "thinking"}
            <div class="seg thinking">
              <div class="seg-head" onclick={() => toggleCollapse(i)} role="button" tabindex="0">
                <span class="badge think">thought {turnLabel(s)}</span>
                <span class="caret">{collapsed[i] ? "▸" : "▾"}</span>
              </div>
              {#if !collapsed[i]}
                <div class="seg-body mono">{s.content}</div>
              {/if}
            </div>
          {:else if s.kind === "tool_call"}
            <div class="seg tool-call">
              <span class="badge call">→ tool · {turnLabel(s)}</span>
              <code class="call-line">
                <span class="tool-name">{s.tool}</span>(<span class="args">{JSON.stringify(s.args)}</span>)
              </code>
            </div>
          {:else if s.kind === "tool_result"}
            <div class="seg tool-result">
              <div class="seg-head" onclick={() => toggleCollapse(i)} role="button" tabindex="0">
                <span class="badge result">← {s.tool} {turnLabel(s)}</span>
                <span class="caret">{collapsed[i] ? "▸" : "▾"}</span>
              </div>
              {#if !collapsed[i]}
                <div class="seg-body mono">{s.content}</div>
              {/if}
            </div>
          {:else if s.kind === "commit"}
            <div class="seg commit">
              <span class="badge commit">commit_play({s.final_play}){s.legal === false ? " · ILLEGAL" : ""}</span>
            </div>
          {:else if s.kind === "assistant_text"}
            <div class="seg assistant">
              <span class="badge burl">burl{turnLabel(s) ? ` · ${turnLabel(s)}` : ""}</span>
              <div class="seg-body mono">{s.content || (streaming && i === segments.length - 1 ? "…" : "")}</div>
            </div>
          {/if}
        {/each}
        {#if streaming && firstTokenAt === null}
          <div class="seg pending">
            <span class="badge pending">burl · thinking</span>
            <span class="pulse">●</span>
            <span class="info dim">prefilling prompt — first token in {prefillSecs}s…</span>
          </div>
        {/if}
        {#if segments.length === 0 && !streaming}
          <div class="hint dim">
            pick a decision on the left to seed a conversation, or just start typing
            below to chat with the base model.
          </div>
        {/if}
      </div>

      <div class="composer-wrap">
        <div class="status-bar">
          <label class="toggle">
            <input type="checkbox" bind:checked={enableThinking} disabled={streaming}>
            think
          </label>
          <label class="toggle dim small">
            max
            <input
              type="number"
              bind:value={maxTokens}
              min="64"
              max="4096"
              step="128"
              disabled={streaming}
              class="maxt"
            >
          </label>
          {#if streaming}
            {#if firstTokenAt === null}
              <span class="pulse">●</span>
              <span>prefilling prompt · {prefillSecs}s</span>
            {:else}
              <span class="pulse green">●</span>
              <span>streaming · {totalSecs}s · ttft {((firstTokenAt - (streamStart ?? 0)) / 1000).toFixed(1)}s</span>
            {/if}
          {/if}
        </div>
        <div class="composer">
          <textarea
            bind:value={input}
            placeholder={loadedDecision
              ? "ask burl about this decision — ⌘/ctrl+enter to send"
              : "message burl — ⌘/ctrl+enter to send"}
            rows="3"
            onkeydown={onKey}
            disabled={streaming}
          ></textarea>
          <button onclick={send} disabled={streaming || !input.trim()}>
            {streaming ? "…" : "send"}
          </button>
        </div>
      </div>
    </section>
  </main>
</div>

<style>
  .app {
    display: grid;
    grid-template-rows: auto 1fr;
    height: 100%;
  }
  header {
    padding: 0.5rem 1rem;
    border-bottom: 1px solid #333;
    display: flex;
    gap: 1rem;
    align-items: center;
  }
  .spacer { flex: 1; }
  .info { font-size: 0.8rem; color: #aaa; font-family: ui-monospace, monospace; }
  .info.dim { color: #666; }

  main {
    display: grid;
    grid-template-columns: 320px 1fr;
    overflow: hidden;
  }

  .sidebar {
    border-right: 1px solid #333;
    overflow-y: auto;
    padding: 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    min-height: 0;
  }
  .sidebar section { display: flex; flex-direction: column; gap: 0.4rem; }
  .sidebar label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #888;
  }

  select {
    background: #111;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 0.3rem;
    font-family: ui-monospace, monospace;
    font-size: 0.8rem;
  }

  .buckets {
    display: flex;
    flex-direction: column;
    gap: 2px;
    max-height: 220px;
    overflow-y: auto;
  }
  .chip {
    background: #1c1c1c;
    border: 1px solid #2c2c2c;
    border-radius: 3px;
    color: #ddd;
    font-family: ui-monospace, monospace;
    font-size: 0.7rem;
    padding: 0.25rem 0.4rem;
    text-align: left;
    cursor: pointer;
  }
  .chip.active { background: #2a3a4a; border-color: #4a6a8a; }
  .dim { color: #777; }
  .small { font-size: 0.75rem; }

  .decisions {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
  .dlist {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .decision {
    background: #1c1c1c;
    border: 1px solid #2c2c2c;
    border-radius: 3px;
    color: #ddd;
    font-family: ui-monospace, monospace;
    font-size: 0.7rem;
    padding: 0.25rem 0.4rem;
    text-align: left;
    cursor: pointer;
    display: flex;
    gap: 0.4rem;
    align-items: stretch;
  }
  .decision.active { background: #2a3a4a; border-color: #4a6a8a; }
  .decision .bar { width: 3px; flex-shrink: 0; border-radius: 1px; }
  .decision .dlabel { display: flex; flex-direction: column; gap: 1px; min-width: 0; }
  .decision .dlabel > span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  .chat {
    display: grid;
    grid-template-rows: 1fr auto;
    min-height: 0;
    padding: 0 1rem;
  }
  .conv {
    overflow-y: auto;
    padding: 0.75rem 0;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .hint { padding: 1rem; }

  /* Segment styling — distinct per kind */
  .seg {
    border-radius: 6px;
    padding: 0.45rem 0.65rem;
    border: 1px solid transparent;
  }
  .seg-head {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
    user-select: none;
  }
  .seg-body {
    margin-top: 0.35rem;
    white-space: pre-wrap;
    line-height: 1.45;
  }
  .seg-body.mono {
    font-family: ui-monospace, "SF Mono", monospace;
    font-size: 0.82rem;
  }
  .badge {
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    font-family: ui-monospace, monospace;
    font-weight: 600;
  }
  .caret {
    color: #888;
    margin-left: auto;
    font-size: 0.7rem;
  }
  .title { color: #aaa; font-family: ui-monospace, monospace; font-size: 0.78rem; }

  .seg.system { background: #1d1d1d; border-color: #333; }
  .badge.sys { background: #333; color: #aaa; }

  .seg.user { background: #1f2c3a; border-color: #2c4263; }
  .badge.user { background: #2c4263; color: #cfe2ff; }
  .seg.user .seg-body { margin-top: 0.3rem; }

  .seg.thinking { background: #2a2618; border-color: #4a4028; border-left: 3px solid #b8923a; }
  .badge.think { background: #5a4a18; color: #f5d77a; }

  .seg.tool-call { background: #2a1f1f; border-color: #4a2c2c; border-left: 3px solid #c46a6a; padding: 0.3rem 0.65rem; display: flex; gap: 0.5rem; align-items: center; }
  .badge.call { background: #4a2c2c; color: #f3a4a4; }
  .call-line {
    font-family: ui-monospace, monospace;
    font-size: 0.82rem;
    color: #f3c0c0;
  }
  .call-line .tool-name { color: #ff9f9f; font-weight: 600; }
  .call-line .args { color: #d4a4a4; }

  .seg.tool-result { background: #1a2a2a; border-color: #2a4a4a; border-left: 3px solid #4ab8a8; }
  .badge.result { background: #1a4a44; color: #a4f3e8; }

  .seg.commit { background: #1c2e1c; border-color: #3a6a3a; border-left: 3px solid #6acf6a; padding: 0.3rem 0.65rem; }
  .badge.commit { background: #2a5a2a; color: #b4f3b4; }

  .seg.assistant { background: #1f2a1f; border-color: #2c4a2c; border-left: 3px solid #5a8a5a; }
  .badge.burl { background: #3a6d4a; color: #cfead0; }

  .composer-wrap {
    border-top: 1px solid #333;
    padding-top: 0.4rem;
  }
  .status-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 0.75rem;
    color: #c8a55a;
    font-family: ui-monospace, monospace;
    padding: 0.2rem 0.4rem;
  }
  .toggle {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    color: #aaa;
    cursor: pointer;
    user-select: none;
  }
  .toggle input { margin: 0; cursor: pointer; }
  .maxt {
    background: #111;
    border: 1px solid #333;
    border-radius: 3px;
    color: #ddd;
    padding: 0.05rem 0.25rem;
    width: 4.2rem;
    font-family: ui-monospace, monospace;
    font-size: 0.75rem;
  }
  .pulse {
    color: #c8a55a;
    animation: pulse 0.9s ease-in-out infinite;
  }
  .pulse.green { color: #6acf6a; }
  @keyframes pulse {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 1.0; }
  }
  .seg.pending {
    background: #2a2618;
    border: 1px dashed #4a4028;
    border-left: 3px solid #b8923a;
    padding: 0.4rem 0.65rem;
    display: flex;
    gap: 0.6rem;
    align-items: center;
  }
  .badge.pending { background: #5a4a18; color: #f5d77a; }
  .composer {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 0.5rem;
    padding: 0.4rem 0 0.75rem;
  }
  textarea {
    background: #111;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 0.5rem;
    resize: vertical;
    font-family: ui-monospace, monospace;
  }
  button {
    background: #2a4d3a;
    border: 1px solid #3a6d4a;
    border-radius: 6px;
    color: white;
    padding: 0 1.2rem;
    cursor: pointer;
    font: inherit;
  }
  button.ghost {
    background: transparent;
    border-color: #555;
    padding: 0.2rem 0.6rem;
    font-size: 0.75rem;
  }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
</style>
