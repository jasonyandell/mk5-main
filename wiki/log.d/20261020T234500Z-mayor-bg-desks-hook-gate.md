## [2026-10-20 | claude | mayor: desks are background sessions after the reboot; the Lem frame is gated by session id]

- The machine rebooted at 18:17 CDT and took the tmux server, every desk, the ladder and the
  watcher with it. The desks came back as `claude --bg --resume <id>` sessions; the ladder and
  watcher were restarted by the Mayor.
- `.claude/hooks/after-compact.sh` gated the Lem essay reload on a tmux session name, which a
  background session doesn't have, so every desk would have paid for the ~4,300-word frame at
  each compaction. It now reloads the frame only for a session whose id is listed in
  `scratch/lem/sessions` (local, gitignored), or a tmux `desk-lem`; everyone else gets one line.
- [[standing-desks]] records the new launch form and the session list.
