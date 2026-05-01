export type Message = { role: "user" | "assistant" | "system"; content: string };

export type ChatEvent =
  | { type: "token"; text: string }
  | { type: "done"; n_tokens: number }
  | { type: "error"; message: string };

export async function* streamChat(
  messages: Message[],
  opts: { enable_thinking?: boolean; max_tokens?: number; temperature?: number } = {},
): AsyncGenerator<ChatEvent> {
  const resp = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({
      messages,
      enable_thinking: opts.enable_thinking ?? false,
      max_tokens: opts.max_tokens ?? 1024,
      temperature: opts.temperature ?? 0.6,
    }),
  });
  if (!resp.ok || !resp.body) {
    throw new Error(`chat failed: ${resp.status}`);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  // SSE per spec separates frames by either \n\n or \r\n\r\n. sse-starlette
  // uses \r\n\r\n. Strip CR up front so the splitter is line-ending-agnostic.
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true }).replace(/\r/g, "");
    const parts = buf.split("\n\n");
    buf = parts.pop() ?? "";
    for (const part of parts) {
      const dataLines = part
        .split("\n")
        .filter((l) => l.startsWith("data: "))
        .map((l) => l.slice(6));
      if (dataLines.length === 0) continue;
      try {
        yield JSON.parse(dataLines.join("\n")) as ChatEvent;
      } catch {
        // ignore keepalive/comment frames
      }
    }
  }
}

export async function health(): Promise<{ ok: boolean; model_repo?: string; adapter_path?: string | null }> {
  const r = await fetch("/api/health");
  return r.json();
}

export type Harvest = {
  name: string;
  n_decisions: number;
  buckets: Record<string, number>;
};

export type DecisionMeta = {
  global_idx: number;
  seed: number;
  player: number;
  bucket: string;
  burl_play: number;
  oracle_play: number;
  burl_regret: number;
  pi_peak: number;
  pi_entropy: number;
  forced_commit: boolean;
  legal_final: boolean;
};

export type LoadedDecision = {
  meta: DecisionMeta;
  messages: Message[];
  segments: import("./parse").Segment[];
  events: any[];
};

export async function listHarvests(): Promise<Harvest[]> {
  const r = await fetch("/api/harvests");
  return r.json();
}

export async function listDecisions(
  harvest: string,
  opts: { bucket?: string; limit?: number; offset?: number } = {},
): Promise<DecisionMeta[]> {
  const params = new URLSearchParams();
  if (opts.bucket) params.set("bucket", opts.bucket);
  if (opts.limit) params.set("limit", String(opts.limit));
  if (opts.offset) params.set("offset", String(opts.offset));
  const r = await fetch(`/api/harvests/${harvest}/decisions?${params}`);
  return r.json();
}

export async function loadDecision(
  harvest: string,
  globalIdx: number,
): Promise<LoadedDecision> {
  const r = await fetch(`/api/harvests/${harvest}/decisions/${globalIdx}`);
  return r.json();
}
