/**
 * Streaming parser for Gemma 4 / Burl native completion stream.
 *
 * Tokens may split markers, so we accumulate into a buffer and emit segment
 * deltas only when we are sure we're not mid-marker. Markers we recognize:
 *
 *   <|channel>thought   ...   <channel|>      → thinking segment
 *   <|tool_call>call:NAME{ARGS}<tool_call|>   → tool_call segment
 *
 * Anything outside these markers is "assistant_text".
 */
export type Segment =
  | { kind: "thinking"; content: string; turn?: number }
  | { kind: "tool_call"; tool: string; args: any; raw?: string; turn?: number }
  | { kind: "tool_result"; tool: string; content: string; turn?: number }
  | { kind: "assistant_text"; content: string; turn?: number }
  | { kind: "user"; content: string }
  | { kind: "system"; content: string }
  | { kind: "commit"; final_play: number; legal?: boolean };

const OPEN_THOUGHT = "<|channel>thought";
const CLOSE_THOUGHT = "<channel|>";
const OPEN_TOOL = "<|tool_call>";
const CLOSE_TOOL = "<tool_call|>";

/**
 * Largest suffix of `s` that is a non-empty prefix of any marker. Used to
 * decide how much trailing text to hold back for the next chunk.
 */
function holdback(s: string): number {
  const markers = [OPEN_THOUGHT, CLOSE_THOUGHT, OPEN_TOOL, CLOSE_TOOL];
  for (let n = Math.min(s.length, OPEN_THOUGHT.length); n > 0; n--) {
    const tail = s.slice(s.length - n);
    if (markers.some((m) => m.startsWith(tail))) return n;
  }
  return 0;
}

export class CompletionParser {
  /** Mode-aware character buffer: characters we've not yet committed. */
  private buf = "";
  /** Current emit segments, last one mutable as text streams in. */
  segments: Segment[] = [];
  /** Mode: where the next text is going. */
  private mode: "text" | "thinking" | "tool" = "text";

  reset() {
    this.buf = "";
    this.segments = [];
    this.mode = "text";
  }

  /** Feed a chunk of token text. Returns the current segments[] (same ref). */
  feed(chunk: string): Segment[] {
    this.buf += chunk;
    this.process();
    return this.segments;
  }

  /** End-of-stream: flush any remaining held-back chars. */
  end(): Segment[] {
    if (this.buf.length > 0) {
      this.flushText(this.buf);
      this.buf = "";
    }
    return this.segments;
  }

  private process() {
    while (true) {
      if (this.mode === "text") {
        const iThought = this.buf.indexOf(OPEN_THOUGHT);
        const iTool = this.buf.indexOf(OPEN_TOOL);
        const candidates = [iThought, iTool].filter((i) => i >= 0);
        if (candidates.length === 0) {
          const safe = this.buf.length - holdback(this.buf);
          if (safe > 0) {
            this.flushText(this.buf.slice(0, safe));
            this.buf = this.buf.slice(safe);
          }
          return;
        }
        const idx = Math.min(...candidates);
        if (idx > 0) this.flushText(this.buf.slice(0, idx));
        if (idx === iThought) {
          this.buf = this.buf.slice(idx + OPEN_THOUGHT.length);
          this.mode = "thinking";
          this.segments.push({ kind: "thinking", content: "" });
        } else {
          this.buf = this.buf.slice(idx + OPEN_TOOL.length);
          this.mode = "tool";
          this.segments.push({ kind: "tool_call", tool: "", args: {}, raw: "" });
        }
      } else if (this.mode === "thinking") {
        const i = this.buf.indexOf(CLOSE_THOUGHT);
        if (i < 0) {
          const safe = this.buf.length - holdback(this.buf);
          if (safe > 0) {
            this.appendToLast("content", this.buf.slice(0, safe));
            this.buf = this.buf.slice(safe);
          }
          return;
        }
        this.appendToLast("content", this.buf.slice(0, i));
        // Trim leading/trailing whitespace on close
        const last = this.segments[this.segments.length - 1] as any;
        this.segments[this.segments.length - 1] = {
          ...last,
          content: (last.content as string).replace(/^\s+|\s+$/g, ""),
        };
        this.buf = this.buf.slice(i + CLOSE_THOUGHT.length);
        this.mode = "text";
      } else if (this.mode === "tool") {
        const i = this.buf.indexOf(CLOSE_TOOL);
        if (i < 0) {
          const safe = this.buf.length - holdback(this.buf);
          if (safe > 0) {
            this.appendToLast("raw", this.buf.slice(0, safe));
            this.buf = this.buf.slice(safe);
          }
          return;
        }
        this.appendToLast("raw", this.buf.slice(0, i));
        const last = this.segments[this.segments.length - 1] as any;
        const m = (last.raw as string).match(/^call:([A-Za-z_][\w]*)\s*({.*})\s*$/s);
        const updated: any = { ...last };
        if (m) {
          updated.tool = m[1];
          try {
            updated.args = relaxedJson(m[2]);
          } catch {
            updated.args = { _raw: m[2] };
          }
        }
        this.segments[this.segments.length - 1] = updated;
        this.buf = this.buf.slice(i + CLOSE_TOOL.length);
        this.mode = "text";
      }
    }
  }

  /** Replace the last segment with a fresh object whose given field has
   *  `text` appended. Preserves reactivity for $state-wrapped consumers
   *  by never mutating an already-spliced object reference. */
  private appendToLast(field: "content" | "raw", text: string) {
    if (!text) return;
    const idx = this.segments.length - 1;
    const last = this.segments[idx] as any;
    this.segments[idx] = { ...last, [field]: (last[field] ?? "") + text };
  }

  private flushText(text: string) {
    if (!text) return;
    const last = this.segments[this.segments.length - 1];
    if (last && last.kind === "assistant_text") {
      this.appendToLast("content", text);
    } else {
      this.segments.push({ kind: "assistant_text", content: text });
    }
  }
}

/**
 * The harness's tool_call args use a relaxed JSON-ish syntax like
 *   {domino_id:7}   or  {play:7,n_samples:10}
 * (unquoted keys, sometimes unquoted strings). Try strict JSON first,
 * fall back to a loose evaluator that quotes bareword keys.
 */
function relaxedJson(s: string): any {
  try {
    return JSON.parse(s);
  } catch {
    const fixed = s.replace(/([{,]\s*)([A-Za-z_][\w]*)\s*:/g, '$1"$2":');
    try {
      return JSON.parse(fixed);
    } catch {
      return { _raw: s };
    }
  }
}
