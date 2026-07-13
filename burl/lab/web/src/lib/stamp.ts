/** Stamp / Timing formatting helpers. */

import type { Stamp, Timing } from "./phase";

export function fmtMs(ms: number | null | undefined): string {
  if (ms == null) return "—";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export function fmtTok(n: number | null | undefined): string {
  if (n == null) return "—";
  if (n < 1000) return String(n);
  return `${(n / 1000).toFixed(1)}k`;
}

export function fmtTokPerS(v: number | null | undefined): string {
  if (v == null || v === 0) return "—";
  return `${v.toFixed(1)} tok/s`;
}

/** A compact one-line summary of a Timing block (Frame.timing). */
export function summarizeTiming(t: Timing): string {
  return [
    fmtMs(t.wall_ms),
    `in ${fmtTok(t.tok_cum_in)}`,
    `out ${fmtTok(t.tok_cum_out)}`,
    fmtTokPerS(t.tok_per_s),
  ].join(" · ");
}

/** Compact label for a single Stamp on a segment (delta + token deltas). */
export function summarizeStamp(s: Stamp): string {
  const parts: string[] = [`+${fmtMs(s.t_wall_ms)}`];
  if (s.tok_in) parts.push(`Δin ${s.tok_in}`);
  if (s.tok_out) parts.push(`Δout ${s.tok_out}`);
  if (s.ms_ttft != null) parts.push(`ttft ${fmtMs(s.ms_ttft)}`);
  if (s.tok_per_s != null) parts.push(fmtTokPerS(s.tok_per_s));
  return parts.join(" · ");
}
