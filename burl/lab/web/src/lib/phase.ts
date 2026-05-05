/** Client mirror of the burl/lab core types. Kept narrow — the wire format
 *  is the contract; these types describe what we receive over /api. */

export type Stamp = {
  t_wall_ms: number;
  t_mono_ns: number;
  tok_in: number;
  tok_out: number;
  tok_cum_in: number;
  tok_cum_out: number;
  ms_ttft: number | null;
  ms_decode: number | null;
  tok_per_s: number | null;
};

/** All Move kinds the server may emit on /api/move SSE. The client only
 *  needs to discriminate by `kind`; the server has already shaped the
 *  payload into a flat dict. */
export type MoveKind =
  | "PhaseEnter"
  | "PhaseExit"
  | "UserText"
  | "UserChoice"
  | "EngineStart"
  | "EngineToken"
  | "EngineToolCall"
  | "ToolResult"
  | "EngineCommit"
  | "SessionOutcome"
  | "LmStudioChatRequest"
  | "LmStudioChatResponse"
  | "LmStudioChatError"
  | "EngineError"
  | "EngineDone"
  | "SystemSet"
  | "AdvertisedSet"
  | "ToolAdded"
  | "ToolRemoved";

export type Move = {
  kind: MoveKind;
  stamp: Stamp;
  // payload fields vary by kind; widen to any-extra for ergonomic access.
  [k: string]: unknown;
};

/** Shape of `Frame.timing` that phases publish. */
export type Timing = {
  wall_ms: number;
  tok_cum_in: number;
  tok_cum_out: number;
  tok_per_s: number;
};

/** A render-ready segment from `state.segments`. The kind discriminates
 *  the rest of the payload. We treat the shape loosely on the client —
 *  rendering switches on `kind` and pulls the few fields it needs. */
export type Segment = {
  kind: string;
  [k: string]: unknown;
};

export type Frame = {
  phase: string;
  segments: Segment[];
  active_tools: string[];
  advertised: string[];
  timing: Timing;
};

export type Option = {
  name: string;
  label: string;
  args_schema: {
    type?: string;
    properties?: Record<string, { type?: string; const?: unknown; items?: unknown }>;
    required?: string[];
    [k: string]: unknown;
  };
};

export type FrameResponse = {
  session_id: string;
  phase: string;
  frame: Frame;
  options: Option[];
};

export type CreateSessionResponse = {
  session_id: string;
};

export type Health = {
  ok: boolean;
  engine_loaded: boolean;
  model: string | null;
  adapter: string | null;
  n_tools: number;
};

export type LmStudioLaunchRequest = {
  model?: string;
  input?: string;
  system_prompt?: string;
  previous_response_id?: string;
  harvest?: string;
  seed?: number;
  store?: boolean;
  temperature?: number;
  max_output_tokens?: number;
};

export type LmStudioChatResponse = {
  session_id: string;
  request: Record<string, unknown>;
  response: Record<string, unknown>;
};
