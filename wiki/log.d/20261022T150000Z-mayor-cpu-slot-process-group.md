## [2026-10-22 | mayor | cpu-slot caps the whole job, not just its wrapper (#249)]

`scripts/cpu-slot` now starts the job in its own process group. The timeout ends the whole
group, and an interrupt or hangup to cpu-slot is passed on to it. Before this, a wrapper such as
`/usr/bin/time -l` exited on the cap and left the runner going outside any slot. One trick-1
rung ran 18 minutes at 21 GB after "timed out". Two new tests fail on the old script and pass on
the new one. The cost line still counts only the direct child, so a wrapper goes outside
cpu-slot, not inside it (documented in the script). The night's pause file for the ladder
(`scratch/ladder/PAUSED`, 988ded87) is in the same playbook spirit: a stop must outlive a respawn.
