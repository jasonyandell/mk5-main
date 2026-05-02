<script lang="ts">
  import { onMount } from "svelte";
  import {
    streamChat,
    health,
    listHarvests,
    listDecisions,
    loadDecision,
    runTool,
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
  let stopAtToolCall = $state(true);

  // Tool-call interception state: when streaming stops at a tool_call|>, the
  // last segment is an open tool_call. We surface inline composer + auto-fill
  // below it so the user can supply / replay / customize the response, then
  // continue generation with the response in context.
  let pendingToolCallIdx = $state<number | null>(null);
  let toolResponseDraft = $state("");

  // Auto-serve: tools that are cheap, deterministic, and trusted run live
  // without pausing for user review. The user can still inspect the result
  // afterwards; they just don't have to click. The improvised registry is
  // refreshed from the server and added to this set on every poll.
  const AUTO_SERVE_BASE = new Set([
    "full_board_snapshot",
    "game_state_snapshot",
    "belief_trajectory",
    "ask_rule",
  ]);
  let improvisedTools = $state<{ name: string; description: string; declaration: string }[]>([]);

  function isAutoServeTool(name: string): boolean {
    return AUTO_SERVE_BASE.has(name) || improvisedTools.some((t) => t.name === name);
  }

  async function refreshImprovisedTools() {
    try {
      const r = await fetch("/api/improvised_tools");
      const j = await r.json();
      improvisedTools = j.tools ?? [];
    } catch (e) {
      console.error("refreshImprovisedTools failed", e);
    }
  }

  let shareStatus = $state<string>("");

  async function shareWithClaude() {
    const body = {
      decision: loadedDecision
        ? { harvest: selectedHarvest, meta: loadedDecision.meta }
        : null,
      segments,
      note: `shared at ${new Date().toISOString()}`,
    };
    try {
      const r = await fetch("/api/chat/share", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const j = await r.json();
      shareStatus = j.ok ? `shared ${j.n_segments} segments` : "share failed";
      setTimeout(() => (shareStatus = ""), 3000);
    } catch (e) {
      shareStatus = `share error: ${e}`;
    }
  }

  // Tool-library popover state. The picker stays open across registry
  // refreshes so the user can curate which tools to advertise on this turn.
  let toolMenuOpen = $state(false);
  let selectedToolNames = $state<Set<string>>(new Set());
  // Tracks names already shown in the picker — used to detect *first*
  // appearances so newly registered tools auto-select once, but the user's
  // subsequent unchecks stick.
  let seenToolNames = $state<Set<string>>(new Set());

  $effect(() => {
    const known = new Set(improvisedTools.map((t) => t.name));
    const nextSelected = new Set<string>();
    let selChanged = false;
    for (const n of selectedToolNames) {
      if (known.has(n)) nextSelected.add(n);
      else selChanged = true;
    }
    for (const n of known) {
      if (!seenToolNames.has(n)) {
        nextSelected.add(n);
        selChanged = true;
      }
    }
    if (selChanged) selectedToolNames = nextSelected;
    // Snapshot the new "known" set so future unchecks of these names persist.
    let seenChanged = known.size !== seenToolNames.size;
    if (!seenChanged) {
      for (const n of known) if (!seenToolNames.has(n)) { seenChanged = true; break; }
    }
    if (seenChanged) seenToolNames = known;
  });

  function toggleToolSelection(name: string) {
    const next = new Set(selectedToolNames);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    selectedToolNames = next;
  }

  async function advertiseImprovisedTools(names?: string[]) {
    await refreshImprovisedTools();
    if (improvisedTools.length === 0) {
      alert("no improvised tools registered yet.");
      return;
    }
    const want = names ?? [...selectedToolNames];
    const picked = improvisedTools.filter((t) => want.includes(t.name));
    if (picked.length === 0) {
      alert("no tools selected — check at least one in the menu.");
      return;
    }
    const lines = [
      "New tools just came online. Add them to your toolbox — they will be served live when you call them.",
      "",
      ...picked.map((t) => t.declaration),
    ];
    segments = [...segments, { kind: "user", content: lines.join("\n\n") }];
    toolMenuOpen = false;
    await send("");
  }

  async function rerunFresh() {
    if (streaming || !loadedDecision) return;
    await refreshImprovisedTools();
    // Find the harvested system message and the first user-state message.
    const original = loadedDecision.segments;
    const sys = original.find((s) => s.kind === "system");
    const firstUser = original.find((s) => s.kind === "user");
    if (!sys || !firstUser) {
      alert("decision is missing system or user segment; cannot rerun.");
      return;
    }
    // Append declarations for the user-checked improvised tools to the
    // harvested system prompt. Burl sees them as additional available tools
    // alongside the base ones; the protocol section is left intact.
    const picked = improvisedTools.filter((t) => selectedToolNames.has(t.name));
    const extraDecls = picked.map((t) => t.declaration).join("");
    const newSystem: Segment = {
      kind: "system",
      content: extraDecls
        ? `${sys.content}\n\n${extraDecls}`
        : sys.content,
    };
    segments = [newSystem, firstUser];
    collapsed = { 0: true };
    toolMenuOpen = false;
    await send("");
  }

  async function unregisterTool(name: string) {
    if (!confirm(`Remove tool "${name}" from the library?`)) return;
    try {
      await fetch(`/api/improvised_tools/${encodeURIComponent(name)}`, {
        method: "DELETE",
      });
      await refreshImprovisedTools();
    } catch (e) {
      alert(`unregister failed: ${e}`);
    }
  }

  // Primer: arbitrary system-prompt prefix the user can paste in (e.g.,
  // Roberson chapter excerpts). Prepended to the harvested system message
  // so the model sees: [primer, original system, user state, ...]. Persists
  // in localStorage for cross-session continuity.
  let primerText = $state("");
  let primerOpen = $state(false);

  $effect(() => {
    // load on mount
    const saved = localStorage.getItem("burl-chat-primer");
    if (saved !== null) primerText = saved;
  });

  $effect(() => {
    // save on change (untracks-ok: we want to persist)
    localStorage.setItem("burl-chat-primer", primerText);
  });

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
    await refreshImprovisedTools();
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
    const primer = primerText.trim();
    let primerInjected = false;
    for (const s of segments) {
      if (s.kind === "system") {
        flushAssistant();
        const sys = primer && !primerInjected
          ? `# Roberson primer (selected passages)\n\n${primer}\n\n---\n\n${s.content}`
          : s.content;
        primerInjected = true;
        out.push({ role: "system", content: sys });
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

  async function send(prependedUserContent?: string) {
    if (streaming) return;
    const content = (prependedUserContent ?? input).trim();
    // Allow continuation with no new user content (e.g. after feeding a tool response).
    const isContinuation = prependedUserContent === "";
    if (!content && !isContinuation) return;
    if (!isContinuation) {
      input = "";
      segments = [...segments, { kind: "user", content }];
    }
    streaming = true;
    pendingToolCallIdx = null;
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

    let stoppedAtToolCall = false;
    try {
      for await (const ev of streamChat(messages, {
        enable_thinking: enableThinking,
        max_tokens: maxTokens,
        stop_at_tool_call: stopAtToolCall,
      })) {
        if (ev.type === "token") {
          if (firstTokenAt === null) firstTokenAt = performance.now();
          parser.feed(ev.text);
          segments = [...segments.slice(0, anchor), ...parser.segments];
          queueMicrotask(() => scrollEl?.scrollTo(0, scrollEl.scrollHeight));
        } else if (ev.type === "done" || ev.type === "stopped_at_tool_call") {
          parser.end();
          segments = [...segments.slice(0, anchor), ...parser.segments];
          if (ev.type === "stopped_at_tool_call") stoppedAtToolCall = true;
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

    // Detect an open tool_call: stream ended at one, OR last meaningful
    // assistant segment is a tool_call with no following tool_result.
    if (stoppedAtToolCall) {
      // Find the last tool_call segment.
      for (let i = segments.length - 1; i >= 0; i--) {
        if (segments[i].kind === "tool_call") {
          pendingToolCallIdx = i;
          const tc = segments[i] as any;
          if (loadedDecision && isAutoServeTool(tc.tool)) {
            await autoServeTool();
            return;
          }
          toolResponseDraft = autoFillFromHarvest(tc) ?? "";
          break;
        }
      }
    }
  }

  async function autoServeTool() {
    if (pendingToolCallIdx === null || !loadedDecision) return;
    const tc = segments[pendingToolCallIdx] as any;
    runningLiveTool = true;
    try {
      const res = await runTool(selectedHarvest!, loadedDecision.meta.global_idx, tc.tool, tc.args);
      const content = res.ok ? res.prose : `[tool error] ${res.error}`;
      segments = [
        ...segments.slice(0, pendingToolCallIdx + 1),
        { kind: "tool_result", tool: tc.tool, content, turn: tc.turn },
      ];
      pendingToolCallIdx = null;
      toolResponseDraft = "";
      await send("");
    } catch (e) {
      segments = [
        ...segments,
        { kind: "assistant_text", content: `[autoserve error] ${e}` },
      ];
    } finally {
      runningLiveTool = false;
    }
  }

  /** Find a matching tool_result in the harvest events for an open tool_call.
   *  Match on tool name and (when present) arg equality; falls back to first
   *  matching tool name if args are different. Returns null if not found. */
  function autoFillFromHarvest(tc: any): string | null {
    const events = loadedDecision?.events;
    if (!events) return null;
    const matches = events.filter(
      (e: any) => e.kind === "tool_result" && e.tool === tc.tool,
    );
    if (matches.length === 0) return null;
    const argsKey = JSON.stringify(tc.args ?? {});
    // Prefer the result whose preceding tool_call had the same args.
    for (let i = 0; i < events.length; i++) {
      const e = events[i];
      if (e.kind === "tool_call" && e.tool === tc.tool &&
          JSON.stringify(e.args ?? {}) === argsKey) {
        // walk forward to next tool_result for this turn
        for (let j = i + 1; j < events.length; j++) {
          if (events[j].kind === "tool_result" && events[j].tool === tc.tool) {
            return events[j].content;
          }
        }
      }
    }
    return matches[0].content;
  }

  async function feedToolResponse() {
    if (pendingToolCallIdx === null) return;
    const tc = segments[pendingToolCallIdx] as any;
    const content = toolResponseDraft.trim();
    if (!content) return;
    // Cull anything after the tool_call (model may have hallucinated past it).
    segments = [
      ...segments.slice(0, pendingToolCallIdx + 1),
      { kind: "tool_result", tool: tc.tool, content, turn: tc.turn },
    ];
    pendingToolCallIdx = null;
    toolResponseDraft = "";
    // Continue generation with the supplied response now in context.
    await send("");
  }

  function skipToolResponse() {
    pendingToolCallIdx = null;
    toolResponseDraft = "";
  }

  let runningLiveTool = $state(false);

  async function runLiveTool() {
    if (pendingToolCallIdx === null || !loadedDecision) return;
    const tc = segments[pendingToolCallIdx] as any;
    runningLiveTool = true;
    try {
      const res = await runTool(selectedHarvest, loadedDecision.meta.global_idx, tc.tool, tc.args);
      if (res.ok) {
        toolResponseDraft = res.prose;
      } else {
        toolResponseDraft = `[tool error] ${res.error}`;
      }
    } catch (e) {
      toolResponseDraft = `[tool error] ${e}`;
    } finally {
      runningLiveTool = false;
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
      <button
        class="ghost"
        onclick={shareWithClaude}
        title="push current chat state to /api/chat/shared so the MCP server can surface it to Claude">
        share with claude{shareStatus ? ` — ${shareStatus}` : ""}
      </button>
      <button
        class="ghost"
        onclick={rerunFresh}
        disabled={streaming}
        title="replay this decision from turn 1 with the selected improvised tools available from the system prompt onward">
        rerun fresh{selectedToolNames.size > 0 ? ` (+${selectedToolNames.size})` : ""}
      </button>
      <div class="tool-menu-wrap">
        <button
          class="ghost"
          onclick={async () => {
            toolMenuOpen = !toolMenuOpen;
            if (toolMenuOpen) await refreshImprovisedTools();
          }}
          title="open the improvised-tools library; pick which to advertise into chat">
          tools{improvisedTools.length > 0 ? ` (${selectedToolNames.size}/${improvisedTools.length})` : ""}
          <span class="caret">{toolMenuOpen ? "▾" : "▸"}</span>
        </button>
        {#if toolMenuOpen}
          <div class="tool-menu" role="menu">
            <div class="tool-menu-head">
              <span class="dim small">improvised library — toggle, then advertise</span>
            </div>
            {#if improvisedTools.length === 0}
              <div class="tool-menu-empty dim small">
                no tools yet. register one via the MCP server.
              </div>
            {:else}
              {#each improvisedTools as t (t.name)}
                <label class="tool-row">
                  <input
                    type="checkbox"
                    checked={selectedToolNames.has(t.name)}
                    onchange={() => toggleToolSelection(t.name)}
                  />
                  <span class="tool-row-body">
                    <span class="tool-row-name">{t.name}</span>
                    <span class="tool-row-desc dim small">{t.description}</span>
                  </span>
                  <button
                    class="ghost tool-row-x"
                    onclick={(e) => { e.preventDefault(); unregisterTool(t.name); }}
                    title="remove this tool from the library">×</button>
                </label>
              {/each}
            {/if}
            <div class="tool-menu-foot">
              <button
                class="ghost"
                onclick={() => (selectedToolNames = new Set(improvisedTools.map((t) => t.name)))}
                disabled={improvisedTools.length === 0}>all</button>
              <button
                class="ghost"
                onclick={() => (selectedToolNames = new Set())}
                disabled={selectedToolNames.size === 0}>none</button>
              <span class="spacer"></span>
              <button
                onclick={() => advertiseImprovisedTools()}
                disabled={selectedToolNames.size === 0 || streaming}>
                advertise {selectedToolNames.size}
              </button>
            </div>
          </div>
        {/if}
      </div>
      <button class="ghost" onclick={newSession}>new session</button>
    {/if}
  </header>

  <main>
    <aside class="sidebar">
      <section class="primer">
        <button
          class="primer-head"
          onclick={() => (primerOpen = !primerOpen)}
        >
          <span>roberson primer</span>
          <span class="dim">{primerText ? `${primerText.length}c` : "(empty)"}</span>
          <span class="caret">{primerOpen ? "▾" : "▸"}</span>
        </button>
        {#if primerOpen}
          <textarea
            bind:value={primerText}
            rows="8"
            placeholder="paste foreword + chapter 2 excerpts here. prepended to the system prompt, persisted to localStorage."
          ></textarea>
          <div class="primer-actions">
            <button class="ghost" onclick={() => (primerText = "")} disabled={!primerText}>
              clear
            </button>
            <span class="dim small">
              {primerText.split(/\s+/).filter(Boolean).length} words
            </span>
          </div>
        {/if}
      </section>

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
            {#if pendingToolCallIdx === i}
              <div class="seg tool-pending">
                <span class="badge pending">awaiting tool response</span>
                <textarea
                  bind:value={toolResponseDraft}
                  rows="6"
                  placeholder={`type or paste a response for ${s.tool}(${JSON.stringify(s.args)})`}
                  disabled={streaming}
                ></textarea>
                <div class="tool-actions">
                  <button onclick={feedToolResponse} disabled={streaming || !toolResponseDraft.trim()}>
                    feed → continue
                  </button>
                  <button class="ghost" onclick={runLiveTool} disabled={streaming || runningLiveTool || !loadedDecision}>
                    {runningLiveTool ? "running…" : "run live"}
                  </button>
                  <button class="ghost" onclick={() => {
                    const auto = autoFillFromHarvest(s);
                    if (auto) toolResponseDraft = auto;
                  }} disabled={streaming || !autoFillFromHarvest(s)}>
                    auto-fill from harvest
                  </button>
                  <button class="ghost" onclick={skipToolResponse} disabled={streaming}>
                    skip
                  </button>
                </div>
              </div>
            {/if}
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
          <label class="toggle">
            <input type="checkbox" bind:checked={stopAtToolCall} disabled={streaming}>
            stop@tool
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
          <button onclick={() => send()} disabled={streaming || !input.trim()}>
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
  .primer { display: flex; flex-direction: column; gap: 0.4rem; }
  .primer-head {
    background: #1c1c1c;
    border: 1px solid #2c2c2c;
    border-radius: 4px;
    color: #ddd;
    font-family: ui-monospace, monospace;
    font-size: 0.75rem;
    padding: 0.4rem 0.5rem;
    text-align: left;
    cursor: pointer;
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }
  .primer textarea {
    background: #111;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 0.5rem;
    font-family: ui-monospace, monospace;
    font-size: 0.78rem;
    color: #e8e8e8;
    resize: vertical;
    width: 100%;
  }
  .primer-actions {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    justify-content: space-between;
  }
  .primer-actions button { font-size: 0.72rem; padding: 0.15rem 0.5rem; }

  .tool-menu-wrap { position: relative; display: inline-block; }
  .tool-menu {
    position: absolute;
    top: calc(100% + 4px);
    right: 0;
    min-width: 360px;
    max-width: 520px;
    max-height: 60vh;
    overflow-y: auto;
    background: #161616;
    border: 1px solid #333;
    border-radius: 6px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
    padding: 0.4rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    z-index: 50;
  }
  .tool-menu-head {
    padding: 0.2rem 0.35rem 0.35rem;
    border-bottom: 1px solid #2a2a2a;
  }
  .tool-menu-empty { padding: 0.6rem 0.35rem; }
  .tool-row {
    display: flex;
    gap: 0.5rem;
    align-items: flex-start;
    padding: 0.35rem 0.35rem;
    border-radius: 4px;
    cursor: pointer;
  }
  .tool-row:hover { background: #1f1f1f; }
  .tool-row input[type="checkbox"] { margin-top: 0.2rem; }
  .tool-row-body { display: flex; flex-direction: column; gap: 0.15rem; flex: 1; min-width: 0; }
  .tool-row-name {
    font-family: ui-monospace, monospace;
    font-size: 0.78rem;
    color: #e8e8e8;
  }
  .tool-row-desc {
    font-size: 0.7rem;
    line-height: 1.35;
    color: #9a9a9a;
    word-break: break-word;
  }
  .tool-row-x {
    font-size: 0.85rem;
    line-height: 1;
    padding: 0.1rem 0.4rem;
    color: #c46a6a;
  }
  .tool-menu-foot {
    display: flex;
    gap: 0.4rem;
    align-items: center;
    border-top: 1px solid #2a2a2a;
    padding: 0.4rem 0.25rem 0.1rem;
  }
  .tool-menu-foot button { font-size: 0.74rem; padding: 0.2rem 0.55rem; }

  .seg.tool-pending {
    background: #2a2618;
    border: 1px dashed #4a4028;
    border-left: 3px solid #b8923a;
    padding: 0.5rem 0.65rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .seg.tool-pending textarea {
    background: #111;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 0.5rem;
    font-family: ui-monospace, monospace;
    font-size: 0.82rem;
    color: #e8e8e8;
    resize: vertical;
  }
  .tool-actions {
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }
  .tool-actions button {
    padding: 0.3rem 0.7rem;
    font-size: 0.78rem;
  }
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
