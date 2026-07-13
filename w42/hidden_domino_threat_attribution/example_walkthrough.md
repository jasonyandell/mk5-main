## Example Walkthrough

Status: illustrative schema walkthrough, not an empirical result.

The rows in `example_rows.jsonl` show the intended attribution grain using a
toy ten-world decision:

- actor seat: `0`
- action slot: `0`, domino `6-6`
- hidden domino under attribution: `5-5`
- baseline Q distribution for the action across ten worlds:
  `[-22, -18, -12, -8, 0, 6, 12, 18, 24, 42]`
- baseline mean Q: `4.2`
- baseline disaster-tail mass, `Q <= -18`: `0.2`
- baseline high-shelf mass, `Q >= 18`: `0.3`

When relative holder `0` (absolute seat `1`) owns `5-5`, the conditioned worlds
are `[-22, -18, 10]`. The mean drops from `4.2` to `-10.0`, the disaster-tail
mass rises from `0.2` to `0.667`, and the high-shelf mass falls from `0.3` to
`0.0`. That holder/domino pair would therefore be a high-impact hidden threat
label for this decision/action.

When relative holder `1` (absolute seat `2`) owns `5-5`, the conditioned worlds
are `[12, 18, 24, 10]`. The same hidden domino has positive impact for the
acting side. That contrast is the core reason the metric is holder-conditioned,
not only domino-conditioned.

The intended downstream label is not "the live player knows seat 1 has `5-5`."
The label is "if a belief model assigns probability to seat 1 holding `5-5`,
that probability should matter more than a harmless hidden holding with small
mean/tail deltas."
