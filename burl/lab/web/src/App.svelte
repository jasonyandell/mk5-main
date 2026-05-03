<script lang="ts">
  import { onMount } from "svelte";
  import { health, createSession, getFrame, postMove } from "./lib/api";
  import type { Frame, Health, Move, Option, Segment } from "./lib/phase";
  import { summarizeStamp, summarizeTiming, fmtMs, fmtTok } from "./lib/stamp";
  import { toolRows } from "./lib/tool";

  /** ## Streaming reactivity (Svelte 5)
   *
   *  burl-chat hit a real bug here: `EngineToken` arrives N times during a
   *  turn, and the natural code is to mutate the trailing `assistant_text`
   *  segment in place — `last.content += text`. That works in Svelte 4
   *  but in Svelte 5 the `$state` proxy is reference-tracked: once we
   *  *reference* `segments[i]` and mutate through that reference, Svelte's
   *  read graph never sees the write and the DOM doesn't re-render.
   *
   *  Fix: replace the segment at its index with a fresh object on every
   *  append. `segments = [...segments.slice(0,i), {...last, text:...}, ...segments.slice(i+1)]`
   *  — assignment to `segments` re-triggers reactivity.
   *
   *  Same shape applies to tool_call → ToolResult: replace the tool_call
   *  segment with a fresh object that has `evidence` filled in, don't
   *  mutate the existing one. */
  let healthInfo = $state<Health | null>(null);
  let healthError = $state<string | null>(null);
  let sessionId = $state<string | null>(null);
  let frame = $state<Frame | null>(null);
  let options = $state<Option[]>([]);
  // segments live separately from `frame` so streaming-token Moves can
  // append to them without re-fetching /frame for every token.
  let segments = $state<Segment[]>([]);
  let timing = $state<{ wall_ms: number; tok_cum_in: number; tok_cum_out: number; tok_per_s: number } | null>(null);
  let phaseName = $state<string>("");
  let streaming = $state(false);
  let streamError = $state<string | null>(null);

  let scrollEl: HTMLDivElement;

  // Composer state — populated by the option clicked. The args dict is
  // built from a textarea / number inputs / JSON textarea depending on
  // the option's args_schema.
  let activeComposerOption = $state<Option | null>(null);
  let composerText = $state("");
  let composerNum = $state<number | null>(null);
  let composerJson = $state("{}");
  let composerError = $state<string | null>(null);
  let wizardHarvest = $state("harvest_batched_20260425_072910");
  let wizardMessage = $state("");
  let wizardError = $state<string | null>(null);
  let wizardSeed = $state<number | null>(1);

  // HATEOAS aside: tools that are active but not advertised.
  let asideOpen = $state(false);
  let toolDraft = $state<Set<string>>(new Set());

  let surfacedTools = $derived(
    frame
      ? frame.active_tools.filter((t) => !frame!.advertised.includes(t))
      : [],
  );

  onMount(async () => {
    try {
      healthInfo = await health();
    } catch (e) {
      healthError = String(e);
    }
  });

  async function onCreateSession() {
    try {
      streamError = null;
      const r = await createSession();
      sessionId = r.session_id;
      await refreshFrame();
    } catch (e) {
      streamError = String(e);
    }
  }

  async function refreshFrame() {
    if (!sessionId) return;
    const r = await getFrame(sessionId);
    frame = r.frame;
    options = r.options;
    phaseName = r.phase;
    segments = [...r.frame.segments];
    timing = r.frame.timing;
    toolDraft = new Set(r.frame.advertised);
    closeComposer();
  }

  function applyMove(mv: Move) {
    // Update the timing ribbon on every Stamp.
    const s = mv.stamp;
    if (s) {
      timing = {
        wall_ms: s.t_wall_ms,
        tok_cum_in: s.tok_cum_in,
        tok_cum_out: s.tok_cum_out,
        tok_per_s: s.tok_per_s ?? 0,
      };
    }

    // Mirror the server-side fold for the few kinds that change segments
    // mid-stream. After the stream ends we'll call refreshFrame() to
    // re-fetch the canonical fold; this is just for live UX.
    switch (mv.kind) {
      case "PhaseEnter":
        phaseName = String((mv as unknown as { phase: string }).phase);
        segments = [...segments, { kind: "phase_enter", phase: phaseName }];
        break;
      case "PhaseExit":
        segments = [
          ...segments,
          { kind: "phase_exit", phase: String((mv as unknown as { phase: string }).phase) },
        ];
        break;
      case "EngineStart":
        segments = [...segments, { kind: "assistant_text", text: "" }];
        break;
      case "EngineToken": {
        // Svelte 5 reactivity: replace the trailing segment with a fresh
        // object — never mutate `segments[i].text` in place.
        const text = String((mv as unknown as { text: string }).text);
        const last = segments[segments.length - 1];
        if (last && last.kind === "assistant_text") {
          const updated: Segment = { ...last, text: String(last.text ?? "") + text };
          segments = [...segments.slice(0, -1), updated];
        } else {
          segments = [...segments, { kind: "assistant_text", text }];
        }
        break;
      }
      case "EngineToolCall": {
        const m = mv as unknown as { name: string; args: Record<string, unknown>; call_id: string };
        segments = [
          ...segments,
          {
            kind: "tool_call",
            name: m.name,
            args: m.args,
            call_id: m.call_id,
            evidence: null,
          },
        ];
        break;
      }
      case "ToolResult": {
        const m = mv as unknown as {
          name: string;
          call_id: string;
          evidence: Record<string, unknown>;
          next_tools: string[];
        };
        // Find the open tool_call with matching call_id and replace it
        // with a fresh object carrying evidence + next_tools.
        let foundIdx = -1;
        for (let i = segments.length - 1; i >= 0; i--) {
          if (
            segments[i].kind === "tool_call" &&
            (segments[i] as unknown as { call_id?: string }).call_id === m.call_id
          ) {
            foundIdx = i;
            break;
          }
        }
        if (foundIdx >= 0) {
          const updated: Segment = {
            ...segments[foundIdx],
            evidence: m.evidence,
            next_tools: m.next_tools,
          };
          segments = [
            ...segments.slice(0, foundIdx),
            updated,
            ...segments.slice(foundIdx + 1),
          ];
        } else {
          segments = [
            ...segments,
            {
              kind: "tool_result",
              name: m.name,
              call_id: m.call_id,
              evidence: m.evidence,
              next_tools: m.next_tools,
            },
          ];
        }
        break;
      }
      case "EngineCommit":
        segments = [
          ...segments,
          { kind: "commit", final: (mv as unknown as { final: unknown }).final },
        ];
        break;
      case "SessionOutcome":
        segments = [
          ...segments,
          {
            kind: "session_outcome",
            summary: (mv as unknown as { summary: Record<string, unknown> }).summary,
          },
        ];
        break;
      case "EngineDone":
        segments = [
          ...segments,
          { kind: "engine_done", reason: String((mv as unknown as { reason: string }).reason) },
        ];
        break;
      case "EngineError":
        segments = [
          ...segments,
          {
            kind: "engine_error",
            message: String((mv as unknown as { message: string }).message),
          },
        ];
        break;
      case "SystemSet":
        segments = [
          ...segments,
          { kind: "system_set", text: String((mv as unknown as { text: string }).text) },
        ];
        break;
      case "AdvertisedSet":
        segments = [
          ...segments,
          {
            kind: "advertised_set",
            names: (mv as unknown as { names: string[] }).names,
          },
        ];
        break;
      case "ToolAdded":
        segments = [
          ...segments,
          { kind: "tool_added", name: String((mv as unknown as { name: string }).name) },
        ];
        break;
      case "ToolRemoved":
        segments = [
          ...segments,
          { kind: "tool_removed", name: String((mv as unknown as { name: string }).name) },
        ];
        break;
      case "UserText":
        segments = [
          ...segments,
          { kind: "user_text", text: String((mv as unknown as { text: string }).text) },
        ];
        break;
      case "UserChoice": {
        const m = mv as unknown as { option_name: string; args: Record<string, unknown> };
        segments = [
          ...segments,
          { kind: "user_choice", option_name: m.option_name, args: m.args },
        ];
        break;
      }
    }

    queueMicrotask(() => scrollEl?.scrollTo(0, scrollEl.scrollHeight));
  }

  async function sendMove(payload: { kind?: string; option_name?: string; args?: Record<string, unknown>; text?: string }) {
    if (!sessionId || streaming) return;
    streaming = true;
    streamError = null;
    wizardError = null;
    try {
      for await (const ev of postMove(sessionId, payload)) {
        if ("error" in ev) {
          streamError = String((ev as { error: string }).error);
          continue;
        }
        applyMove(ev as Move);
      }
    } catch (e) {
      streamError = String(e);
    } finally {
      streaming = false;
      // Re-fetch canonical state from the server's fold.
      try {
        await refreshFrame();
      } catch (e) {
        streamError = String(e);
      }
    }
  }

  // ---- Composer wiring -------------------------------------------------- #

  function composerKindFor(opt: Option): "none" | "text" | "number" | "json" {
    const props = opt.args_schema?.properties ?? {};
    const keys = Object.keys(props);
    if (keys.length === 0) return "none";
    // Single-string schema: render a textarea.
    if (keys.length === 1 && props[keys[0]]?.type === "string") return "text";
    // Single-int schema: render a number input.
    if (keys.length === 1 && (props[keys[0]]?.type === "integer" || props[keys[0]]?.type === "number"))
      return "number";
    // Anything else: JSON textarea.
    return "json";
  }

  function clickOption(opt: Option) {
    composerError = null;
    const kind = composerKindFor(opt);
    if (kind === "none") {
      // Fire immediately.
      void sendMove({ option_name: opt.name, args: {} });
      return;
    }
    activeComposerOption = opt;
    composerText = "";
    composerNum = null;
    // Pre-fill JSON textarea with an empty object shaped against the schema.
    const props = opt.args_schema?.properties ?? {};
    const skeleton: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(props)) {
      const t = (v as { type?: string }).type;
      if (t === "string") skeleton[k] = "";
      else if (t === "integer" || t === "number") skeleton[k] = 0;
      else if (t === "array") skeleton[k] = [];
      else skeleton[k] = null;
    }
    composerJson = JSON.stringify(skeleton, null, 2);
  }

  function closeComposer() {
    activeComposerOption = null;
    composerText = "";
    composerNum = null;
    composerJson = "{}";
    composerError = null;
  }

  async function submitComposer() {
    const opt = activeComposerOption;
    if (!opt) return;
    const kind = composerKindFor(opt);
    let args: Record<string, unknown> = {};
    if (kind === "text") {
      const props = opt.args_schema?.properties ?? {};
      const key = Object.keys(props)[0] ?? "text";
      args = { [key]: composerText };
    } else if (kind === "number") {
      const props = opt.args_schema?.properties ?? {};
      const key = Object.keys(props)[0] ?? "n";
      if (composerNum == null) {
        composerError = "enter a number";
        return;
      }
      args = { [key]: composerNum };
    } else if (kind === "json") {
      try {
        args = JSON.parse(composerJson);
      } catch (e) {
        composerError = `invalid JSON: ${e}`;
        return;
      }
    }
    await sendMove({ option_name: opt.name, args });
  }

  // ---- HATEOAS aside ---------------------------------------------------- #

  function toggleAsideTool(name: string) {
    const next = new Set(toolDraft);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    toolDraft = next;
  }

  async function submitAside() {
    if (!frame) return;
    const names = Array.from(toolDraft);
    asideOpen = false;
    await sendMove({ option_name: "set_advertised", args: { names } });
  }

  function sortedNames(names: Iterable<string>): string {
    return Array.from(names).sort().join("\n");
  }

  // ---- Guided wizard ---------------------------------------------------- #

  function hasOption(name: string): boolean {
    return options.some((o) => o.name === name);
  }

  function chooseToolPreset(names: string[]) {
    const available = new Set(preGameToolRows.map((row) => row.name));
    toolDraft = new Set(names.filter((name) => available.has(name)));
  }

  function chooseAllTools() {
    toolDraft = new Set(preGameToolRows.map((row) => row.name));
  }

  async function wizardApplyTools() {
    if (!hasOption("set_advertised")) return;
    await submitAside();
  }

  async function wizardGenerateSystem() {
    if (!hasOption("generate_system")) return;
    await sendMove({ option_name: "generate_system", args: {} });
  }

  async function wizardLoadDecision() {
    if (!hasOption("load_decision")) return;
    const harvest = wizardHarvest.trim();
    if (!harvest) {
      wizardError = "enter a harvest name";
      return;
    }
    await sendMove({
      option_name: "load_decision",
      args: { harvest, idx: wizardSeed ?? 0 },
    });
  }

  async function wizardShipToGemma() {
    if (!hasOption("start_run")) return;
    await sendMove({ option_name: "start_run", args: {} });
  }

  async function wizardSendSeededDecision() {
    if (!hasOption("send_seeded_decision")) return;
    const harvest = wizardHarvest.trim();
    if (!harvest) {
      wizardError = "enter a harvest name";
      return;
    }
    if (wizardSeed == null) {
      wizardError = "enter a seed";
      return;
    }
    await sendMove({
      option_name: "send_seeded_decision",
      args: { harvest, seed: wizardSeed },
    });
  }

  async function wizardSendMessage() {
    const text = wizardMessage.trim();
    if (!text) {
      wizardError = "write a message first";
      return;
    }
    wizardMessage = "";
    if (phaseName === "pre_game" && hasOption("ask_gemma")) {
      await sendMove({ option_name: "ask_gemma", args: { text } });
    } else if (phaseName === "in_run" && hasOption("interject")) {
      await sendMove({ option_name: "interject", args: { text } });
    }
  }

  async function wizardStartAnotherSession() {
    if (phaseName === "post_turn" && hasOption("start_new_session")) {
      await sendMove({ option_name: "start_new_session", args: {} });
    }
  }

  // ---- Helpers ---------------------------------------------------------- #

  let prettyHealth = $derived(() => {
    if (healthError) return `error: ${healthError}`;
    if (!healthInfo) return "connecting…";
    return [
      healthInfo.engine_loaded ? "engine ✓" : "engine ✗",
      `${healthInfo.n_tools} tools`,
      healthInfo.model ?? "",
      healthInfo.adapter ?? "",
    ]
      .filter(Boolean)
      .join(" · ");
  });

  let healthOk = $derived(!!(healthInfo && healthInfo.ok));

  let preGameToolRows = $derived(frame ? toolRows(frame.segments) : []);
  let advertisedCount = $derived(frame ? frame.advertised.length : 0);
  let toolSurfaceCount = $derived(
    preGameToolRows.length > 0 ? preGameToolRows.length : (frame?.active_tools.length ?? 0),
  );
  let toolDraftChanged = $derived(
    frame ? sortedNames(frame.advertised) !== sortedNames(toolDraft) : false,
  );
  let promptBuilderSegment = $derived(
    [...segments].reverse().find((s) => s.kind === "prompt_builder") ?? null,
  );
  let renderedSystemSegment = $derived(
    [...segments].reverse().find((s) => s.kind === "rendered_system") ?? null,
  );
  let systemBaseChars = $derived(Number(promptBuilderSegment?.base_chars ?? 0));
  let renderedSystemChars = $derived(Number(promptBuilderSegment?.rendered_chars ?? 0));
  let hasUserPrompt = $derived(Boolean(promptBuilderSegment?.has_user_message));
  let hasSystemPrompt = $derived(systemBaseChars > 0);
  let hasRenderedSystem = $derived(
    renderedSystemChars > 0 || String(renderedSystemSegment?.text ?? "").length > 0,
  );
  let hasCommitTool = $derived(Boolean(frame?.advertised.includes("commit_play")));
  let canShipToGemma = $derived(
    hasRenderedSystem && hasUserPrompt && !toolDraftChanged && hasOption("start_run"),
  );
  let canSendSeededDecision = $derived(
    hasRenderedSystem &&
      hasCommitTool &&
      !toolDraftChanged &&
      hasOption("send_seeded_decision") &&
      wizardHarvest.trim().length > 0 &&
      wizardSeed != null,
  );
</script>

<div class="app">
  <header class="phase-header">
    <div class="left">
      <span class="dot" class:ok={healthOk} class:err={!!healthError}></span>
      <strong>burl lab</strong>
      <span class="dim mono small">{prettyHealth()}</span>
    </div>
    <div class="center">
      {#if phaseName}
        <span class="phase-pill">{phaseName}</span>
      {/if}
      {#if timing}
        <span class="ribbon mono small dim">
          {summarizeTiming(timing)}
        </span>
      {/if}
      {#if streaming}
        <span class="pulse">●</span>
        <span class="dim small">streaming</span>
      {/if}
    </div>
    <div class="right">
      {#if sessionId}
        <span class="dim mono small">session {sessionId.slice(0, 8)}…</span>
        <button class="ghost" onclick={refreshFrame} disabled={streaming}>refresh</button>
        <button class="ghost" onclick={() => { sessionId = null; frame = null; options = []; segments = []; phaseName = ""; }}>close</button>
      {:else}
        <button onclick={onCreateSession} disabled={!healthOk}>create session</button>
      {/if}
    </div>
  </header>

  <main>
    {#if !sessionId}
      <section class="wizard">
        <div class="wizard-title">
          <strong>Start here</strong>
          <span class="dim small">new session → prompt → tools → Gemma</span>
        </div>
        <div class="wizard-steps compact">
          <div class="wizard-step active">
            <span class="step-num">1</span>
            <div class="step-main">
              <strong>New session</strong>
              <span class="dim small">fresh event log, fresh prompt builder</span>
            </div>
            <button onclick={onCreateSession} disabled={!healthOk || streaming}>
              start
            </button>
          </div>
          <div class="wizard-step muted">
            <span class="step-num">2</span>
            <div class="step-main">
              <strong>Prompt or chat</strong>
              <span class="dim small">optional chat, then generate system rules</span>
            </div>
          </div>
          <div class="wizard-step muted">
            <span class="step-num">3</span>
            <div class="step-main">
              <strong>Select tools</strong>
              <span class="dim small">choose the protocol Gemma will see</span>
            </div>
          </div>
          <div class="wizard-step muted">
            <span class="step-num">4</span>
            <div class="step-main">
              <strong>Send play</strong>
              <span class="dim small">seeded decision prompt into Gemma</span>
            </div>
          </div>
        </div>
      </section>
    {:else if phaseName === "pre_game"}
      <section class="wizard">
        <div class="wizard-title">
          <strong>Setup wizard</strong>
          <span class="dim small">session {sessionId.slice(0, 8)}…</span>
        </div>
        {#if wizardError}
          <div class="wizard-error mono small">{wizardError}</div>
        {/if}
        <div class="wizard-steps">
          <div class="wizard-step done">
            <span class="step-num">1</span>
            <div class="step-main">
              <strong>New session</strong>
              <span class="dim small">ready</span>
            </div>
          </div>

          <div class="wizard-step" class:done={hasSystemPrompt} class:active={!hasSystemPrompt}>
            <span class="step-num">2</span>
            <div class="step-main">
              <strong>Prompt or chat</strong>
              <div class="wizard-actions">
                <button onclick={wizardGenerateSystem} disabled={streaming || !hasOption("generate_system")}>
                  use default rules + system
                </button>
              </div>
              <div class="wizard-chat">
                <textarea
                  bind:value={wizardMessage}
                  rows="2"
                  placeholder="or ask Gemma directly"
                  disabled={streaming}
                ></textarea>
                <button onclick={wizardSendMessage} disabled={streaming || !hasOption("ask_gemma")}>
                  chat
                </button>
              </div>
            </div>
          </div>

          <div class="wizard-step" class:done={!toolDraftChanged && advertisedCount > 0} class:active={hasSystemPrompt && advertisedCount === 0}>
            <span class="step-num">3</span>
            <div class="step-main">
              <strong>Select tools</strong>
              <div class="wizard-actions">
                <button class="ghost" onclick={() => chooseToolPreset(["state_brief", "legal_plays", "commit_play"])} disabled={streaming}>
                  basic
                </button>
                <button class="ghost" onclick={() => chooseToolPreset(["belief_trajectory", "explore_game", "commit_play"])} disabled={streaming}>
                  classic
                </button>
                <button class="ghost" onclick={chooseAllTools} disabled={streaming}>
                  all
                </button>
                <button class="ghost" onclick={() => (toolDraft = new Set())} disabled={streaming}>
                  clear
                </button>
              </div>
              <div class="wizard-tool-grid">
                {#each preGameToolRows as row (row.name)}
                  <label class="wizard-tool">
                    <input
                      type="checkbox"
                      checked={toolDraft.has(row.name)}
                      onchange={() => toggleAsideTool(row.name)}
                      disabled={streaming}
                    />
                    <span>
                      <span class="mono">{row.name}</span>
                      <span class="dim small">{row.role}</span>
                    </span>
                  </label>
                {/each}
              </div>
              <div class="wizard-actions">
                <button onclick={wizardApplyTools} disabled={streaming || !toolDraftChanged || !hasOption("set_advertised")}>
                  apply tools ({toolDraft.size})
                </button>
                <span class="dim small">{advertisedCount}/{toolSurfaceCount} advertised</span>
              </div>
            </div>
          </div>

          <div class="wizard-step" class:active={hasRenderedSystem && !toolDraftChanged} class:done={false}>
            <span class="step-num">4</span>
            <div class="step-main">
              <strong>Send play</strong>
              <div class="wizard-status mono small">
                <span class:ok-text={hasSystemPrompt}>system {systemBaseChars} chars</span>
                <span class:ok-text={advertisedCount > 0}>tools {advertisedCount}</span>
                <span class:ok-text={hasCommitTool}>{hasCommitTool ? "commit tool ready" : "need commit_play"}</span>
                <span class:ok-text={hasRenderedSystem}>rendered {renderedSystemChars} chars</span>
              </div>
              {#if toolDraftChanged}
                <div class="dim small">
                  apply your tool selection before sending; the selected tools are rendered into the system prompt.
                </div>
              {:else if !hasCommitTool}
                <div class="dim small">
                  include `commit_play` if this run should end by committing a domino.
                </div>
              {/if}
              <div class="wizard-load">
                <input
                  bind:value={wizardHarvest}
                  placeholder="harvest name"
                  disabled={streaming}
                />
                <input
                  class="idx-input"
                  type="number"
                  aria-label="seed"
                  bind:value={wizardSeed}
                  disabled={streaming}
                />
                <button onclick={wizardLoadDecision} disabled={streaming || !hasOption("load_decision")}>
                  load only
                </button>
              </div>
              {#if hasUserPrompt}
                <div class="dim small">
                  user prompt loaded; `ship to Gemma` will run that loaded prompt.
                </div>
              {/if}
              <div class="wizard-actions">
                <button onclick={wizardGenerateSystem} disabled={streaming || !hasOption("generate_system")}>
                  refresh prompt
                </button>
                <button onclick={wizardSendSeededDecision} disabled={streaming || !canSendSeededDecision}>
                  send seed {wizardSeed ?? ""}
                </button>
                <button onclick={wizardShipToGemma} disabled={streaming || !canShipToGemma}>
                  ship loaded
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>
    {:else if phaseName === "in_run"}
      <section class="wizard run-wizard">
        <div class="wizard-title">
          <strong>Gemma is running</strong>
          <span class="dim small">chat before or after the next tool/answer</span>
        </div>
        <div class="wizard-chat run-chat">
          <textarea
            bind:value={wizardMessage}
            rows="2"
            placeholder="send a note into this run"
            disabled={streaming}
          ></textarea>
          <button onclick={wizardSendMessage} disabled={streaming || !hasOption("interject")}>
            send
          </button>
        </div>
      </section>
    {:else if phaseName === "post_turn"}
      <section class="wizard run-wizard">
        <div class="wizard-title">
          <strong>Turn finished</strong>
          <span class="dim small">review the trace, then start another setup</span>
        </div>
        <button onclick={wizardStartAnotherSession} disabled={streaming || !hasOption("start_new_session")}>
          new setup
        </button>
      </section>
    {/if}

    <section class="feed">
      <div class="conv" bind:this={scrollEl}>
        {#if !sessionId}
          <div class="hint dim">
            no session yet. click <strong>create session</strong> to start a new
            run on the lab server (port 8002).
          </div>
        {/if}
        {#if streamError}
          <div class="seg engine-error">
            <span class="badge err">stream error</span>
            <div class="seg-body mono">{streamError}</div>
          </div>
        {/if}
        {#each segments as s, i (i)}
          {@const k = s.kind}
          {#if k === "phase_enter"}
            <div class="seg phase">
              <span class="badge phase">→ {String(s.phase)}</span>
            </div>
          {:else if k === "phase_exit"}
            <div class="seg phase">
              <span class="badge phase dim">← {String(s.phase)}</span>
            </div>
          {:else if k === "user_text"}
            <div class="seg user">
              <span class="badge user">user</span>
              <div class="seg-body mono">{String(s.text ?? "")}</div>
            </div>
          {:else if k === "user_choice"}
            <div class="seg user-choice">
              <span class="badge choice">choice · {String(s.option_name ?? "")}</span>
              {#if s.args && Object.keys(s.args as Record<string, unknown>).length > 0}
                <code class="args-line">{JSON.stringify(s.args)}</code>
              {/if}
            </div>
          {:else if k === "assistant_text"}
            <div class="seg assistant">
              <span class="badge burl">burl</span>
              <div class="seg-body mono">{String(s.text ?? "") || (streaming && i === segments.length - 1 ? "…" : "")}</div>
            </div>
          {:else if k === "tool_call"}
            <div class="seg tool-call">
              <span class="badge call">→ {String(s.name)}</span>
              <code class="args-line">{JSON.stringify(s.args ?? {})}</code>
              {#if s.evidence}
                <div class="seg-body mono dim small">
                  {String((s.evidence as Record<string, unknown>).prose ?? "")}
                </div>
                {#if Array.isArray(s.next_tools) && (s.next_tools as string[]).length > 0}
                  <div class="next-tools small dim">
                    surfaced: {(s.next_tools as string[]).join(", ")}
                  </div>
                {/if}
              {/if}
            </div>
          {:else if k === "tool_result"}
            <div class="seg tool-result">
              <span class="badge result">← {String(s.name)}</span>
              <div class="seg-body mono">
                {String((s.evidence as Record<string, unknown>)?.prose ?? "")}
              </div>
            </div>
          {:else if k === "commit"}
            <div class="seg commit">
              <span class="badge commit">commit</span>
              <code class="args-line">{JSON.stringify(s.final)}</code>
            </div>
          {:else if k === "session_outcome"}
            {@const outcome = s.summary as Record<string, unknown>}
            {@const matches = (outcome.matches ?? {}) as Record<string, unknown>}
            {@const refs = (outcome.references ?? {}) as Record<string, unknown>}
            <div class="seg cfg">
              <span class="badge result">outcome</span>
              <div class="seg-body mono small">
                final {String(outcome.final_domino_id)} ·
                {outcome.legal ? "legal" : "illegal"} ·
                tools {Array.isArray(outcome.selected_tools) ? (outcome.selected_tools as string[]).join(", ") : ""}
                <br />
                pi {String(refs.pi_play)} {matches.pi ? "✓" : "·"}
                qmean {String(refs.qmean_play)} {matches.qmean ? "✓" : "·"}
                oracle {String(refs.oracle_play)} {matches.oracle ? "✓" : "·"}
                consensus {String(outcome.consensus_play ?? "none")}
                {matches.consensus === true ? " ✓" : ""}
              </div>
            </div>
          {:else if k === "engine_done"}
            <div class="seg engine-done dim small">
              engine done · {String(s.reason)}
            </div>
          {:else if k === "engine_error"}
            <div class="seg engine-error">
              <span class="badge err">error</span>
              <div class="seg-body mono">{String(s.message)}</div>
            </div>
          {:else if k === "system_set"}
            <div class="seg system">
              <span class="badge sys">system set · {String(s.text ?? "").length} chars</span>
              <div class="seg-body mono">{String(s.text ?? "")}</div>
            </div>
          {:else if k === "rendered_system"}
            <div class="seg system">
              <span class="badge sys">rendered system · {String(s.text ?? "").length} chars</span>
              {#if String(s.text ?? "").length === 0}
                <div class="seg-body dim small">
                  no system prompt yet. pick tools, then use
                  <strong>Generate system prompt</strong> for the default Burl prompt,
                  <strong>Set system prompt</strong> for manual text, or
                  <strong>Load harvested decision into builder</strong> to import
                  harvested `prompt_system` and `prompt_user`.
                </div>
              {:else}
                <details>
                  <summary class="dim small">show / hide</summary>
                  <div class="seg-body mono">{String(s.text ?? "")}</div>
                </details>
              {/if}
            </div>
          {:else if k === "prompt_builder"}
            <div class="seg builder">
              <span class="badge builder">prompt builder</span>
              <div class="builder-grid mono small">
                <span>base {Number(s.base_chars ?? 0)} chars</span>
                <span>rendered {Number(s.rendered_chars ?? 0)} chars</span>
                <span>tools {Number(s.advertised_count ?? 0)}/{Number(s.tool_count ?? 0)}</span>
                <span>{s.has_user_message ? "user prompt loaded" : "no user prompt"}</span>
              </div>
            </div>
          {:else if k === "advertised_set"}
            <div class="seg cfg dim small">
              advertised set: {Array.isArray(s.names) ? (s.names as string[]).join(", ") : ""}
            </div>
          {:else if k === "tool_added"}
            <div class="seg cfg dim small">+ tool {String(s.name)}</div>
          {:else if k === "tool_removed"}
            <div class="seg cfg dim small">− tool {String(s.name)}</div>
          {:else if k === "tool_row"}
            <!-- pre_game enriches its frame with tool_row segments; we
                 render them in the aside, not in the main feed. -->
          {/if}
        {/each}
      </div>
    </section>

    {#if frame && (preGameToolRows.length > 0 || surfacedTools.length > 0)}
      <aside class="hateoas" class:open={asideOpen}>
        <button class="aside-toggle ghost" onclick={() => (asideOpen = !asideOpen)}>
          tools
          <span class="dim small">
            {advertisedCount}/{toolSurfaceCount} advertised
            {#if surfacedTools.length > 0}
              · {surfacedTools.length} surfaced
            {/if}
          </span>
          <span class="caret">{asideOpen ? "▾" : "▸"}</span>
        </button>
        {#if asideOpen}
          <div class="aside-body">
            {#if preGameToolRows.length > 0}
              <div class="aside-section">
                <div class="aside-head dim small">registry</div>
                {#each preGameToolRows as row (row.name)}
                  <label class="tool-row">
                    <input
                      type="checkbox"
                      checked={toolDraft.has(row.name)}
                      onchange={() => toggleAsideTool(row.name)}
                    />
                    <span class="tool-row-body">
                      <span class="tool-row-name mono">{row.name}</span>
                      <span class="tool-row-meta dim small">
                        {row.role}
                        {#if row.advertised}· advertised{/if}
                      </span>
                      <span class="tool-row-phrase dim small">{row.phrase}</span>
                    </span>
                  </label>
                {/each}
              </div>
            {/if}
            {#if surfacedTools.length > 0}
              <div class="aside-section">
                <div class="aside-head dim small">HATEOAS surfaced (active but not advertised)</div>
                {#each surfacedTools as name (name)}
                  <label class="tool-row">
                    <input
                      type="checkbox"
                      checked={toolDraft.has(name)}
                      onchange={() => toggleAsideTool(name)}
                    />
                    <span class="tool-row-body">
                      <span class="tool-row-name mono">{name}</span>
                    </span>
                  </label>
                {/each}
              </div>
            {/if}
            <div class="aside-foot">
              {#if options.find((o) => o.name === "set_advertised")}
                <button onclick={submitAside} disabled={!toolDraftChanged || streaming}>
                  apply ({toolDraft.size})
                </button>
              {:else}
                <span class="dim small">phase has no set_advertised option</span>
              {/if}
            </div>
          </div>
        {/if}
      </aside>
    {/if}

    <section class="composer-region">
      <details class="advanced-console">
        <summary>advanced console</summary>
        {#if options.length === 0}
          <div class="dim small">no options in this phase</div>
        {:else}
          <div class="options">
            {#each options as opt (opt.name + ":" + JSON.stringify(opt.args_schema))}
              <button class="ghost" onclick={() => clickOption(opt)} disabled={streaming}>
                {opt.label}
                <span class="dim small">· {opt.name}</span>
              </button>
            {/each}
          </div>
          {#if activeComposerOption}
            {@const kind = composerKindFor(activeComposerOption)}
            <div class="composer">
              <div class="composer-head">
                <strong>{activeComposerOption.label}</strong>
                <span class="dim small mono">{activeComposerOption.name}</span>
                <span class="spacer"></span>
                <button class="ghost" onclick={closeComposer} disabled={streaming}>cancel</button>
              </div>
              {#if kind === "text"}
                <textarea
                  bind:value={composerText}
                  rows="4"
                  placeholder="text"
                  disabled={streaming}
                ></textarea>
              {:else if kind === "number"}
                <input type="number" bind:value={composerNum} disabled={streaming} />
              {:else if kind === "json"}
                <textarea
                  bind:value={composerJson}
                  rows="6"
                  placeholder={'{"key": "value"}'}
                  disabled={streaming}
                ></textarea>
              {/if}
              {#if composerError}
                <div class="composer-error">{composerError}</div>
              {/if}
              <div class="composer-actions">
                <button onclick={submitComposer} disabled={streaming}>send</button>
              </div>
            </div>
          {/if}
        {/if}
      </details>
    </section>
  </main>
</div>

<style>
  .app {
    display: grid;
    grid-template-rows: auto 1fr;
    height: 100%;
  }

  .phase-header {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 1rem;
    align-items: center;
    padding: 0.55rem 1rem;
    border-bottom: 1px solid #333;
    background: #161616;
  }
  .phase-header .left { display: flex; gap: 0.6rem; align-items: center; }
  .phase-header .center { display: flex; gap: 0.6rem; align-items: center; justify-content: center; }
  .phase-header .right { display: flex; gap: 0.5rem; align-items: center; justify-content: flex-end; }

  .phase-pill {
    background: #2a3a4a;
    border: 1px solid #4a6a8a;
    border-radius: 999px;
    padding: 0.15rem 0.6rem;
    font-family: ui-monospace, monospace;
    font-size: 0.78rem;
    color: #cfe2ff;
    letter-spacing: 0.04em;
  }
  .ribbon {
    color: #8aa;
    font-family: ui-monospace, monospace;
  }

  .dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #555;
  }
  .dot.ok { background: #6acf6a; }
  .dot.err { background: #c46a6a; }

  .dim { color: #888; }
  .small { font-size: 0.78rem; }
  .mono { font-family: ui-monospace, "SF Mono", monospace; }

  main {
    display: grid;
    grid-template-rows: auto 1fr auto;
    grid-template-columns: 1fr;
    overflow: hidden;
    position: relative;
  }

  .wizard {
    border-bottom: 1px solid #303030;
    background: #171a1a;
    padding: 0.65rem 1rem;
  }
  .wizard-title {
    display: flex;
    gap: 0.6rem;
    align-items: baseline;
    margin-bottom: 0.55rem;
  }
  .wizard-steps {
    display: grid;
    grid-template-columns: minmax(8rem, 0.85fr) minmax(11rem, 1fr) minmax(12rem, 1.05fr) minmax(9rem, 0.95fr);
    gap: 0.55rem;
  }
  .wizard-steps.compact .wizard-step {
    min-height: 4.2rem;
  }
  .wizard-step {
    display: flex;
    gap: 0.55rem;
    align-items: flex-start;
    min-width: 0;
    border: 1px solid #303535;
    border-radius: 6px;
    padding: 0.55rem;
    background: #1d2020;
  }
  .wizard-step.active {
    border-color: #54708a;
    background: #1e2730;
  }
  .wizard-step.done {
    border-color: #355d49;
    background: #1b241f;
  }
  .wizard-step.muted {
    opacity: 0.58;
  }
  .step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.45rem;
    height: 1.45rem;
    flex: 0 0 auto;
    border-radius: 999px;
    background: #2c3333;
    color: #d6e2de;
    font-family: ui-monospace, monospace;
    font-size: 0.75rem;
  }
  .wizard-step.done .step-num {
    background: #28533d;
  }
  .wizard-step.active .step-num {
    background: #315574;
  }
  .step-main {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
    min-width: 0;
    flex: 1;
  }
  .wizard-actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.35rem;
  }
  .wizard-load {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 4.5rem;
    gap: 0.35rem;
  }
  .wizard-load button {
    grid-column: 1 / -1;
  }
  .wizard-chat {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 0.35rem;
    align-items: stretch;
  }
  .wizard-load input,
  .wizard-chat textarea {
    min-width: 0;
    width: 100%;
  }
  .idx-input {
    text-align: right;
  }
  .wizard-tool-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 0.25rem;
    max-height: 18rem;
    overflow-y: auto;
    padding-right: 0.15rem;
  }
  .wizard-tool {
    display: flex;
    align-items: flex-start;
    gap: 0.4rem;
    min-width: 0;
    padding: 0.25rem;
    border-radius: 4px;
    cursor: pointer;
  }
  .wizard-tool:hover {
    background: #242828;
  }
  .wizard-tool span {
    display: flex;
    min-width: 0;
    flex-direction: column;
    gap: 0.05rem;
  }
  .wizard-tool .mono {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .wizard-status {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
  }
  .wizard-status span {
    border: 1px solid #333;
    border-radius: 999px;
    padding: 0.12rem 0.45rem;
    color: #888;
  }
  .wizard-status span.ok-text {
    border-color: #31583f;
    color: #bfe6c6;
  }
  .wizard-error {
    color: #f3a4a4;
    margin-bottom: 0.4rem;
  }
  .run-wizard {
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  .run-wizard .wizard-title {
    margin: 0;
    flex: 0 0 auto;
  }
  .run-chat {
    flex: 1;
  }
  @media (max-width: 900px) {
    .wizard-steps {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
  @media (max-width: 650px) {
    .wizard {
      max-height: 64vh;
      overflow-y: auto;
    }
    .wizard-steps {
      grid-template-columns: 1fr;
    }
  }

  .feed { overflow: hidden; }
  .conv {
    height: 100%;
    overflow-y: auto;
    padding: 0.75rem 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
  }
  .hint { padding: 1.2rem; }

  .seg {
    border-radius: 6px;
    padding: 0.45rem 0.65rem;
    border: 1px solid transparent;
  }
  .seg-body {
    margin-top: 0.35rem;
    white-space: pre-wrap;
    line-height: 1.45;
    font-family: ui-monospace, monospace;
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
  .args-line {
    display: inline-block;
    margin-left: 0.4rem;
    font-family: ui-monospace, monospace;
    font-size: 0.78rem;
    color: #d4a4a4;
  }

  .seg.phase { padding: 0.25rem 0.65rem; }
  .badge.phase { background: #2a3a4a; color: #cfe2ff; }

  .seg.user { background: #1f2c3a; border-color: #2c4263; }
  .badge.user { background: #2c4263; color: #cfe2ff; }

  .seg.user-choice { background: #1c1c2c; border-color: #2c2c4a; padding: 0.3rem 0.65rem; }
  .badge.choice { background: #3c3c6a; color: #c4c4f3; }

  .seg.assistant { background: #1f2a1f; border-color: #2c4a2c; border-left: 3px solid #5a8a5a; }
  .badge.burl { background: #3a6d4a; color: #cfead0; }

  .seg.tool-call { background: #2a1f1f; border-color: #4a2c2c; border-left: 3px solid #c46a6a; }
  .badge.call { background: #4a2c2c; color: #f3a4a4; }

  .seg.tool-result { background: #1a2a2a; border-color: #2a4a4a; border-left: 3px solid #4ab8a8; }
  .badge.result { background: #1a4a44; color: #a4f3e8; }

  .seg.commit { background: #1c2e1c; border-color: #3a6a3a; border-left: 3px solid #6acf6a; }
  .badge.commit { background: #2a5a2a; color: #b4f3b4; }

  .seg.engine-error { background: #2e1c1c; border-color: #6a3a3a; border-left: 3px solid #cf6a6a; }
  .badge.err { background: #5a2a2a; color: #f3b4b4; }

  .seg.system { background: #1d1d1d; border-color: #333; }
  .badge.sys { background: #333; color: #aaa; }

  .seg.builder { background: #191f1d; border-color: #2c3e36; }
  .badge.builder { background: #254238; color: #bfe6d2; }
  .builder-grid {
    margin-top: 0.35rem;
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.35rem 0.8rem;
    color: #a8b8b0;
  }

  .seg.engine-done, .seg.cfg { padding: 0.2rem 0.65rem; }

  .next-tools { margin-top: 0.3rem; font-family: ui-monospace, monospace; }

  /* HATEOAS aside */
  .hateoas {
    border-top: 1px solid #333;
    background: #181818;
  }
  .aside-toggle {
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    border-bottom: 1px solid #2a2a2a;
    padding: 0.5rem 1rem;
    color: #ccc;
    cursor: pointer;
    display: flex;
    gap: 0.6rem;
    align-items: center;
  }
  .caret { margin-left: auto; color: #666; }
  .aside-body {
    padding: 0.5rem 1rem 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    max-height: 32vh;
    overflow-y: auto;
  }
  .aside-head {
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.2rem 0;
    border-bottom: 1px solid #2a2a2a;
  }
  .tool-row {
    display: flex;
    gap: 0.6rem;
    align-items: flex-start;
    padding: 0.3rem 0.2rem;
    border-radius: 4px;
    cursor: pointer;
  }
  .tool-row:hover { background: #1f1f1f; }
  .tool-row input[type="checkbox"] { margin-top: 0.2rem; }
  .tool-row-body { display: flex; flex-direction: column; gap: 0.1rem; min-width: 0; flex: 1; }
  .tool-row-name { color: #e8e8e8; font-size: 0.82rem; }
  .tool-row-meta { font-family: ui-monospace, monospace; }
  .tool-row-phrase { line-height: 1.35; word-break: break-word; }
  .aside-foot {
    display: flex;
    justify-content: flex-end;
    padding-top: 0.3rem;
    border-top: 1px solid #2a2a2a;
  }

  /* Composer */
  .composer-region {
    border-top: 1px solid #333;
    padding: 0.6rem 1rem 0.9rem;
    background: #141414;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }
  .advanced-console summary {
    cursor: pointer;
    color: #aaa;
    font-size: 0.82rem;
  }
  .advanced-console[open] summary {
    margin-bottom: 0.55rem;
  }
  .options {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
  }
  .composer {
    background: #181818;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 0.6rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .composer-head {
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }
  .composer-head .spacer { flex: 1; }
  .composer textarea, .composer input {
    width: 100%;
  }
  .composer-actions {
    display: flex;
    justify-content: flex-end;
  }
  .composer-error {
    color: #f3a4a4;
    font-family: ui-monospace, monospace;
    font-size: 0.8rem;
  }

  .pulse {
    color: #c8a55a;
    animation: pulse 0.9s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 1.0; }
  }
</style>
