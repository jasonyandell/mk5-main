/** SSE consumer for /api/move.
 *
 *  ## CRLF awareness
 *
 *  sse-starlette emits frames terminated by `\r\n\r\n` (per the SSE spec
 *  the terminator is two consecutive end-of-lines, where each end-of-line
 *  is one of CR, LF, or CRLF). The browser's `TextDecoder` returns the
 *  raw bytes — including any `\r` — so a naive `buf.split("\n\n")` will
 *  *never* find the boundary and you'll buffer forever.
 *
 *  burl-chat hit this exact bug and the fix is documented in
 *  `burl/chat/web/src/lib/api.ts` as well as `burl/lab/server/stream.py`.
 *  Two equivalent invariants: split on `/\r?\n\r?\n/`, OR strip `\r`
 *  before splitting on `\n\n`. We do the latter — it's terser and means
 *  any stray `\r` inside a `data:` payload also gets normalised away.
 *
 *  Methodological note: when this class of bug recurs, dump
 *  `JSON.stringify(buf)` (or `buf.replace(...)` of `\r` to a literal) —
 *  terminals silently hide `\r`. */

import type { Move } from "./phase";

export type SseEvent = Move | { kind: "Error"; error: string };

/** Stream Move events from a POST /api/move response.
 *
 *  Yields each parsed Move as soon as the SSE frame arrives. Returns when
 *  the server closes the stream. */
export async function* streamMove(
  resp: Response,
): AsyncGenerator<SseEvent> {
  if (!resp.ok || !resp.body) {
    throw new Error(`/api/move failed: ${resp.status}`);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    // Normalise CR — see comment block above. Without this we never
    // detect frame boundaries because sse-starlette uses \r\n\r\n.
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
        yield JSON.parse(dataLines.join("\n")) as SseEvent;
      } catch {
        // ignore comments / keepalive frames
      }
    }
  }
  // Flush any trailing data line that didn't end with a blank-line frame.
  if (buf.trim()) {
    const dataLines = buf
      .split("\n")
      .filter((l) => l.startsWith("data: "))
      .map((l) => l.slice(6));
    if (dataLines.length > 0) {
      try {
        yield JSON.parse(dataLines.join("\n")) as SseEvent;
      } catch {
        /* ignore */
      }
    }
  }
}
