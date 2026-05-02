/** REST + SSE bindings for burl/lab/server. All endpoints are proxied
 *  through Vite to http://localhost:8002 (see vite.config.ts). */

import type {
  CreateSessionResponse,
  FrameResponse,
  Health,
} from "./phase";
import { streamMove } from "./stream";
import type { SseEvent } from "./stream";

export async function health(): Promise<Health> {
  const r = await fetch("/api/health");
  if (!r.ok) throw new Error(`/api/health: ${r.status}`);
  return r.json();
}

export async function createSession(): Promise<CreateSessionResponse> {
  const r = await fetch("/api/sessions", { method: "POST" });
  if (!r.ok) throw new Error(`POST /api/sessions: ${r.status}`);
  return r.json();
}

export async function listSessions(): Promise<{ sessions: { id: string; has_events: boolean }[] }> {
  const r = await fetch("/api/sessions");
  if (!r.ok) throw new Error(`GET /api/sessions: ${r.status}`);
  return r.json();
}

export async function getFrame(sessionId: string): Promise<FrameResponse> {
  const r = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/frame`);
  if (!r.ok) throw new Error(`GET /api/sessions/{id}/frame: ${r.status}`);
  return r.json();
}

/** POST /api/move and yield each SSE Move as it arrives. */
export async function* postMove(
  sessionId: string,
  move: { kind?: string; option_name?: string; args?: Record<string, unknown>; text?: string },
): AsyncGenerator<SseEvent> {
  const resp = await fetch("/api/move", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({ session_id: sessionId, move }),
  });
  yield* streamMove(resp);
}
