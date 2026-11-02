## [2026-11-02 | claude | Walt bids: the honest plan's root splat as a live bidder, piloted]

**Touched pages:** [[walt-bidder-2026-11-02]], [[scenario-player]], [[arena]]
**Added:** `arena/walt_bid.py`, spec `waltbid[:<walt play options>][,c<cutoff>]`: Zeb's bidder (`ZebBidder`) holding Walt's own planner, so Walt's honest forecast (floor 16) is read as a bid curve the way Zeb's is. A clock-sized pilot against the heuristic bidder with Walt play on both sides wins 34/48 games, +1.25 marks/game [+0.35, +2.12]; Walt bids 25% of hands at a mean of 32.7 and makes 88.8% against a mean forecast of 0.840 at the call, the heuristic 55.4% at 30.4; 48 games, 3.04e13 instructions. Follow-ups filed as #254.
**Questions opened:** whether the forecast at the chosen call is honest at power (the max over seven declarations and thirteen targets); a Walt mode for the package so Plunge can bid live with Walt's play.
