/** ToolSpec rendering helpers. The server publishes ToolSpecs only as
 *  names on the wire (full specs live in the Python registry). The
 *  pre_game phase enriches its frame with `tool_row` segments that carry
 *  name / role / phrase / advertised — those drive the UI. */

import type { Segment } from "./phase";

export type ToolRow = {
  kind: "tool_row";
  name: string;
  role: string;
  phrase: string;
  advertised: boolean;
};

export function isToolRow(s: Segment): s is Segment & ToolRow {
  return s.kind === "tool_row" && typeof (s as { name?: unknown }).name === "string";
}

/** Pull tool_row segments out of a frame. pre_game renders one row per
 *  tool in the registry; other phases don't, so this is generally empty. */
export function toolRows(segments: Segment[]): ToolRow[] {
  return segments.filter(isToolRow) as unknown as ToolRow[];
}

/** Group tool rows by protocol_role for display. */
export function groupByRole(rows: ToolRow[]): Map<string, ToolRow[]> {
  const out = new Map<string, ToolRow[]>();
  for (const r of rows) {
    const arr = out.get(r.role) ?? [];
    arr.push(r);
    out.set(r.role, arr);
  }
  return out;
}
