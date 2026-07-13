---
title: "vs-Random Win Rate Is a Suspect Metric for Marks-to-7 Play"
kind: decision
first_seen: 2026-07-06
last_updated: 2026-07-06
status: active
---

## The gap

Every headline number in [[zeb]]'s era-4 build — 70.9% vs random at 557K params
(`a7d6b5b`), 76.3% peak at 3.3M params (`large-belief-recap.md`), the ~74% "capacity
ceiling" from [[full-teacher-eq-experiment]] (`6081420`) — is graded by the era's own eval
harness as "winner = more points." Jason flagged the mismatch himself, mid-era, and it was
never fixed in-window:

> "we're just doing 'winner = more points'. but more points does not necessarily win the
> game! ... you have to get 30 AT LEAST. there is no bid anywhere where getting <30 is a
> win. but our measurements are all score a>score b." (2026-02-09T22:04)

Texas 42's actual win condition is marks-to-7 across hands, gated by whether the bidding
team makes their contract (30+ of a possible points, or the specific count needed for a
special contract) — not raw point totals. A team can outscore its opponent on total points
across a sample of hands while still not being the team that would win more marks-to-7
games, because points earned inside a failed (set) contract don't convert to marks the way
points earned inside a made one do.

## What this means for the era-4 numbers

The capacity-ceiling conclusion in [[full-teacher-eq-experiment]] may be **partly an
artifact of grading the wrong thing**. A policy that is unusually good at padding point
totals inside contracts it's already going to lose, or unusually bad at converting close
contracts into makes, could show a stable vs-random percentage that says little about how
many marks-to-7 games it would actually win. The ~74% ceiling was measured this way at
every model size tested; no re-grading against marks was performed in this window.

## A second confound, surfaced the next day

Random-vs-random play does not even converge to 50/50: at 400K games it sat at 47.8/52.2%,
attributed to a leader/bidder seating disadvantage under random play (2026-02-10T00:48).
[[eval-matrix-bradley-terry]]'s eval harness already double-seats matchups to cancel this
out, but the underlying vs-random baselines cited elsewhere in the era do not all specify
whether double-seating was applied.

## Decision

Treat every era-4 vs-random percentage — and any figure derived from it, including the
"~74% capacity ceiling" — as a **point-total metric, not a marks-to-7 win-rate metric**,
until re-graded. Do not cite these numbers as settled evidence of play strength under 42's
actual win condition without this caveat attached.

The one number in the era that *is* graded correctly is the
[[eval-matrix-bradley-terry|Bradley-Terry Elo]] snapshot (`zeb-large-belief` = 1579 vs
`eq:n=100` anchor 1600) — Elo from win/loss records sidesteps the points-vs-marks
confound, though the underlying win/loss records themselves may still inherit whatever
seating asymmetry applied to that eval run.

## Why this wasn't caught sooner

The bug was named, in question form, at 22:04 on 2026-02-09 — the same week the
[[eval-matrix-bradley-terry|Elo infrastructure]] existed and could in principle have
re-graded head-to-head results on marks. The correct target (head-to-head vs
[[expected-q-value|E[Q]]], on marks) had the infrastructure to measure it by Feb 9; the
era's load-bearing verdict still leaned on the metric Jason had already disowned by
closeout: *"it's not really the right target anyway"* (2026-02-16T15:38, see [[zeb]]).

## Generalizable principle

When a headline percentage is "win rate," check what "win" means against the game's actual
terminal condition, not an easily-computed proxy (total points, single-hand score
comparison). A proxy that correlates with the real objective most of the time can still
silently misgrade the specific comparisons — capacity ceilings, teacher-signal nulls —
that a project treats as load-bearing.

See [[zeb]] · [[full-teacher-eq-experiment]] · [[eval-matrix-bradley-terry]] ·
[[era4-zeb-era|conversation digest]].
