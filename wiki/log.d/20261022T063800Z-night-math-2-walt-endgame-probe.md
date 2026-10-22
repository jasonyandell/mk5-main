## [2026-10-22 | claude | night-math-2: the Walt endgame probe (#244)]

- Filed the comparison algebra's three feasibility items as #242, #243 and #244 ([[comparison-algebra-2026-10-21]]).
- #244 is done. On 27,118 real endgame decisions with two or three tricks left (Zeb's arena form and the champion),
  Walt's tile was checked against the exact information-state optimum of its own model. The belief is uniform on
  the whole fiber, and the others are played by Walt's own model of them.
  - Silas (`silas family --courses`, new) and a Python Fraction reference agree at every decision.
  - Wherever Walt enumerates, its tile is in the exact set: 24,179 of 24,179.
  - Where it samples (three left, fiber > 512), it misses 5.7% [4.7, 6.7] (Zeb) and 5.6% [4.2, 7.1] (champion),
    against chance at about 33%. The regret is 0.00077 [0.00058, 0.00098] and 0.00052 [0.00036, 0.00070] success
    probability, under the pre-registered 0.001 bar.
  - For Zeb, the floor is about half the endgame regret.
- Enumerating up to 1,680 worlds closes the gap exactly. Filed as #246 (gated, low priority).
- Tool: `lem/endgame.py`. Data: HF `texas-42-zeb-research-2026-10/zeb/walt-endgame-probe-2026-10-22`.
- Page: [[walt-endgame-probe-2026-10-22]].
