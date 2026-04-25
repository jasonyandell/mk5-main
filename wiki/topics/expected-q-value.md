---
title: E[Q] — Expected Q-Value
kind: topic
first_seen: a8bccfa
last_updated: a8bccfa
status: active
---

## Overview

E[Q] is a scalar measuring how good a position is for the team taking a given action, averaged over game outcomes. It is produced by the Q-value checkpoint from [[forge]] (lem/OVERVIEW.md @ a8bccfa).

## Roles in LEM

E[Q] serves two distinct functions in [[lem]]:

1. **Game generation policy** — games are played E[Q]-greedy with N=10 rollouts. This setting was validated during an earlier experiment (zeb) as not materially worse than N=100 and significantly cheaper (lem/OVERVIEW.md @ a8bccfa).

2. **Ground-truth grader for [[k1-grading]]** — the bot's E[Q] sets the acceptance threshold for [[star]] traces. The model's chosen action must match or exceed the bot's E[Q] to keep the trace (lem/OVERVIEW.md @ a8bccfa).

## Relationship to narration

[[narration]] carries running score information, which reflects trick points accumulated. E[Q] over game outcomes is the deeper signal that determines which actions are worth learning from (lem/OVERVIEW.md @ a8bccfa).

## Links

[[forge]] [[k1-grading]] [[narration]] [[lem]]
