"""The scenario player — Walt, written from the mathematics.

    Walt = R_live[ R_model[ U ] ]

A player plans from one point of view: it holds a weighted set of scenarios
(possible hidden situations), chooses ONE action across every scenario it
cannot tell apart, and lets its plan branch only where an observation
separates them. The other seats are modeled as the same construction one
rung down; random legal play is the bottom rung.

Modules:
  rules    — per-declaration rule tables, tabulated from forge.oracle.tables
  record   — the public record a cell of scenarios shares, and its advance
  fiber    — exact-uniform sampling of hidden-hand completions
  planner  — the finite planning rule and the ladder of players
  atlas    — one known deal's perfect-information tree, to be labeled by Walt
"""
