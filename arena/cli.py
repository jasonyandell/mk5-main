"""Arena CLI: full-game head-to-head between two (bidder + play) players.

    python -u -m arena.cli --team-a heuristic+lens:ev --team-b bid30+lens:ev \
        --n-games 64 --n-samples 10 --device mps

Player spec: <bidder>+<play>
    bidder: heuristic | heuristic:<min_trumps>,<caution> | bid30 | random[:<p_bid>]
            | gus[:<samples>[,wp][,pass[<q>]][,max]]   (rung #21; wp = marks-to-7
              utility, pass<q> = rung #27 v2 equilibrium-aware pass baseline,
              max = rung #31 utility-MAXIMIZING bid so strong hands reach 84)
            | net[:[wp][,pass[<q>]][,max]]   (rung #22 distilled bid-strength net,
              <1ms/hand; same utility/max options as gus)
            | margin[:[wp][,pass[<q>]][,model=<path>]]   (jud v0 #32 value-native
              bidder: the realized-value head prices contracts; pass baseline uses
              net:wp's measured make-rate; model= selects a versioned head; no
              pmake_scale, no max)
            | jud[:[wp][,pass[<q>]][,model=<path>]]   (jud v1: the unified organ
              prices contracts at its empty-history root; same ValueBidder walk
              and options as margin)
    play:   lens:<utility> | scorelens[:<band>] | belieflens[:<utility>]
            | judplay[:model=<path>] | judsearch[:n<worlds>[,model=<path>]]
            | random
            (scorelens = rung #27 v2 score-conditioned play risk;
             belieflens = rung #25 belief-weighted world sampling, --gus-adapter;
             judplay = jud v1 value-native play — argmax E[pts] over the jud
             head, defenders minimize; no oracle, no world sampling;
             judsearch = jud v1 JS1 — roll the current trick to resolution in
             N sampled worlds with the jud head at every seat, evaluate the
             post-trick leaf from the mover's POV; no oracle)

Writes summary.json, per_hand.csv, per_game.csv under --out-dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from arena.auction import BidPolicy
from arena.bidders import Bid30Bidder, HeuristicBidder, RandomBidder
from arena.engine import ArenaConfig
from arena.match import game_rows, hand_rows, run_match, snapshot_rows, summarize
from arena.play import PlayPolicy, RandomPlay


def parse_bidder(
    spec: str, *, device: str, gus_adapter: str | None,
    model=None, belief_bidder_adapter: str | None = None,
) -> BidPolicy:
    name, _, arg = spec.partition(":")
    if name == "heuristic":
        if arg:
            min_trumps, _, caution = arg.partition(",")
            return HeuristicBidder(int(min_trumps), int(caution or 0))
        return HeuristicBidder()
    if name == "bid30":
        return Bid30Bidder()
    if name == "random":
        return RandomBidder(float(arg)) if arg else RandomBidder()
    if name == "gus":
        from champion.bidder import GusBidder, GusPointsEvaluator
        from champion.utility import MarksToSeven

        parts = [p for p in arg.split(",") if p]
        samples = parts[0] if parts and parts[0].isdigit() else ""
        tags = parts[1:] if samples else parts
        evaluator = GusPointsEvaluator(
            adapter=gus_adapter, device=device,
            n_samples=int(samples) if samples else 32,
        )
        print(f"Gus bidder: {evaluator}", flush=True)
        # tags: "wp" selects the marks-to-7 utility; "pass[<q>]" adds the
        # equilibrium-aware pass baseline (rung #27 v2) with P(opp takes)=q.
        pass_q = 0.0
        for t in tags:
            if t.startswith("pass"):
                pass_q = float(t[4:]) if t[4:] else 0.4
        if pass_q > 0.0:
            utility = MarksToSeven(pass_q_opp=pass_q, pass_make_rate=0.55)
        elif "wp" in tags:
            utility = MarksToSeven()
        else:
            utility = None
        return GusBidder(evaluator, utility, maximize="max" in tags)
    if name == "net":
        # Distilled bid-strength net (rung #22) as the auction policy — same
        # GusBidder walk, P(make) from a <1ms forward pass instead of Gus sim.
        from champion.bidder import GusBidder, NetPointsEvaluator
        from champion.utility import MarksToSeven

        tags = [p for p in arg.split(",") if p]
        evaluator = NetPointsEvaluator()
        print(f"Net bidder: {evaluator}", flush=True)
        pass_q = 0.0
        for t in tags:
            if t.startswith("pass"):
                pass_q = float(t[4:]) if t[4:] else 0.4
        if pass_q > 0.0:
            utility = MarksToSeven(pass_q_opp=pass_q, pass_make_rate=0.55)
        elif "wp" in tags:
            utility = MarksToSeven()
        else:
            utility = None
        return GusBidder(pmake_fn=evaluator, utility=utility, maximize="max" in tags)
    if name == "margin":
        # Value-native bidder (jud v0, #32): the realized-value head prices each
        # contract instead of the double-dummy oracle.
        # Spec: margin[:wp][,pass[<q>]][,model=<path>].
        #   wp        marks-to-7 utility (default; value-native pricing wants the
        #             score-conditioned utility), mirroring `net:wp`.
        #   pass[<q>] rung #27-v2 equilibrium-aware pass baseline, P(opp takes)=q
        #             (default 0.4). pass_make_rate is net:wp's MEASURED offense
        #             make-rate (0.754, Step-3 A/B) — passing to net:wp correctly
        #             credits that they make ~75%, not the old 0.55 placeholder.
        #   model=<p> load a specific margin_net checkpoint (loop-versioned heads).
        # NO pmake_scale, NO max — the head is realized-calibrated and #31 measured
        # bid-magnitude maximization dead.
        from champion.utility import MarksToSeven
        from champion.value_bidder import ValueBidder, load_margin_net

        NET_WP_MAKE_RATE = 0.754  # measured net:wp offense make-rate (Step-3 A/B)
        tags = [p for p in arg.split(",") if p]
        model_path = "champion/margin_net.pt"
        pass_q = 0.0
        for t in tags:
            if t.startswith("model="):
                model_path = t[len("model="):]
            elif t.startswith("pass"):
                pass_q = float(t[4:]) if t[4:] else 0.4
        margin_model = load_margin_net(model_path, device="cpu")  # tiny MLP; CPU fastest
        if pass_q > 0.0:
            utility = MarksToSeven(pass_q_opp=pass_q, pass_make_rate=NET_WP_MAKE_RATE)
        else:
            utility = MarksToSeven()
        bidder = ValueBidder(margin_model, utility)
        print(f"Value bidder: {bidder} (model={model_path})", flush=True)
        return bidder
    if name == "jud":
        # jud v1 unified organ as the auction policy: the SAME ValueBidder walk
        # as `margin`, pointed at JudNet's empty-history root (pmake_table has
        # MarginNet's exact signature). Spec: jud[:wp][,pass[<q>]][,model=<path>].
        from champion.jud_net import load_jud_net
        from champion.utility import MarksToSeven
        from champion.value_bidder import ValueBidder

        NET_WP_MAKE_RATE = 0.754  # measured net:wp offense make-rate (Step-3 A/B)
        tags = [p for p in arg.split(",") if p]
        model_path = "champion/jud_net.pt"
        pass_q = 0.0
        for t in tags:
            if t.startswith("model="):
                model_path = t[len("model="):]
            elif t.startswith("pass"):
                pass_q = float(t[4:]) if t[4:] else 0.4
        jud_model = load_jud_net(model_path, device="cpu")  # small MLP; CPU fastest
        if pass_q > 0.0:
            utility = MarksToSeven(pass_q_opp=pass_q, pass_make_rate=NET_WP_MAKE_RATE)
        else:
            utility = MarksToSeven()
        bidder = ValueBidder(jud_model, utility)
        print(f"Jud bidder: {bidder} (model={model_path})", flush=True)
        return bidder
    if name == "belief":
        # Belief-conditioned bidder (rung #26 keystone): the hypothetical
        # completed auction + #24 belief-weighted oracle E[Q] -> P(make) per
        # contract. Spec: belief[:<adapter>]; adapter falls back to
        # --belief-bidder-adapter, then the #24 default. Needs the oracle model.
        from champion.belief import load_belief
        from champion.belief_bidder import BeliefBidder
        from champion.utility import MarksToSeven

        if model is None:
            raise ValueError("belief bidder needs the oracle model (load it first)")
        # Spec: belief[:<adapter>][,s<pmake_scale>][,m<margin>]. The adapter is a
        # path (may itself start with 's'), so disambiguate tags by float-parseability:
        # s0.7 = rung-#26 optimism correction (scale the double-dummy P(make)), m0.05
        # = utility margin. Adapter falls back to --belief-bidder-adapter.
        def _floatable(x: str) -> bool:
            try:
                float(x)
                return True
            except ValueError:
                return False

        adapter = None
        pmake_scale, margin = 1.0, 0.0
        for part in (p for p in arg.split(",") if p):
            if part[0] == "s" and _floatable(part[1:]):
                pmake_scale = float(part[1:])
            elif part[0] == "m" and _floatable(part[1:]):
                margin = float(part[1:])
            else:
                adapter = part
        adapter = adapter or belief_bidder_adapter
        belief_model, is_voids = load_belief(adapter, device)
        bidder = BeliefBidder(
            belief_model, model, is_voids=is_voids, device=device,
            utility=MarksToSeven(), maximize=True,
            pmake_scale=pmake_scale, margin=margin,
        )
        print(f"Belief bidder: {bidder} (adapter={adapter!r}, is_voids={is_voids}, "
              f"pmake_scale={pmake_scale}, margin={margin})", flush=True)
        return bidder
    raise ValueError(
        f"Unknown bidder: {spec!r} "
        f"(heuristic | bid30 | random | gus | net | margin | jud | belief)"
    )


def parse_play(
    spec: str, *, model, n_samples: int, device: str, seed: int, gus_adapter: str | None = None,
) -> PlayPolicy:
    name, _, arg = spec.partition(":")
    if name == "random":
        return RandomPlay(seed=seed)
    if name == "lens":
        from arena.lens_play import LensPlay
        return LensPlay(model, utility=arg or "ev", n_samples=n_samples, device=device)
    if name == "scorelens":
        # Score-conditioned play risk (rung #27 v2); optional arg is the WP band.
        from champion.play_risk import ScoreConditionedLensPlay
        band = float(arg) if arg else 0.15
        return ScoreConditionedLensPlay(
            model, n_samples=n_samples, device=device, band=band,
        )
    if name == "belieflens":
        # Belief-weighted world sampling (rung #25); belief model = --gus-adapter.
        from champion.play import BeliefLensPlay
        return BeliefLensPlay(
            model, utility=arg or "ev", n_samples=n_samples, device=device,
            belief_adapter=gus_adapter,
        )
    if name == "judplay":
        # jud v1 value-native play: argmax E[pts] over the jud head's post-move
        # info-states, defenders minimize. Spec: judplay[:model=<path>].
        from arena.jud_play import JudPlay
        from champion.jud_net import load_jud_net

        model_path = "champion/jud_net.pt"
        for t in (p for p in arg.split(",") if p):
            if t.startswith("model="):
                model_path = t[len("model="):]
        play = JudPlay(load_jud_net(model_path, device="cpu"))  # small MLP; CPU fastest
        print(f"Jud play: {play} (model={model_path})", flush=True)
        return play
    if name == "judauxplay":
        # Lane B diagnostic consumer: argmax the per-legal-action aux head on
        # the CURRENT state (acting-seat orientation, no sign flip).
        # Spec: judauxplay:model=<path> (checkpoint must carry the aux head).
        from arena.jud_play import JudAuxPlay
        from champion.jud_net import load_jud_net

        model_path = "champion/jud_net.pt"
        for t in (p for p in arg.split(",") if p):
            if t.startswith("model="):
                model_path = t[len("model="):]
        play = JudAuxPlay(load_jud_net(model_path, device="cpu"))
        print(f"Jud aux play: {play} (model={model_path})", flush=True)
        return play
    if name == "judsearch":
        # jud v1 search rung (JS1): sample N consistent worlds, roll the current
        # trick to resolution with the jud head playing every seat, average the
        # post-trick leaf E[pts] over worlds, argmax (defenders minimize).
        # Spec: judsearch[:n<worlds>][,model=<path>], default n10.
        from arena.jud_search import JudSearch
        from champion.jud_net import load_jud_net

        model_path = "champion/jud_net.pt"
        n_worlds = 10
        for t in (p for p in arg.split(",") if p):
            if t.startswith("model="):
                model_path = t[len("model="):]
            elif t.startswith("n") and t[1:].isdigit():
                n_worlds = int(t[1:])
        # CPU throughout: the 470k MLP dispatches faster than MPS at this size,
        # and the MRV sampler is tiny at arena batch widths.
        play = JudSearch(load_jud_net(model_path, device="cpu"), n_worlds=n_worlds)
        print(f"Jud search: {play} (model={model_path})", flush=True)
        return play
    raise ValueError(
        f"Unknown play policy: {spec!r} "
        f"(lens:<utility> | scorelens[:<band>] | belieflens[:<utility>] "
        f"| judplay[:model=<path>] | judsearch[:n<worlds>[,model=<path>]] | random)"
    )


def needs_model(*specs: str) -> bool:
    # The play side needs the oracle (lens family); so does the `belief` bidder,
    # which queries the oracle E[Q] for the hypothetical completed auction.
    plays = any(
        s.split("+", 1)[1].split(":")[0] in ("lens", "scorelens", "belieflens")
        for s in specs
    )
    bidders = any(s.split("+", 1)[0].split(":")[0] == "belief" for s in specs)
    return plays or bidders


def needs_gus(*specs: str) -> bool:
    return any(s.split("+", 1)[0].startswith("gus") for s in specs)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--team-a", required=True, help="e.g. heuristic+lens:ev")
    parser.add_argument("--team-b", required=True, help="e.g. bid30+lens:ev")
    parser.add_argument("--n-games", type=int, default=64,
                        help="Total games (split evenly across the two halves)")
    parser.add_argument("--marks-to-win", type=int, default=7)
    parser.add_argument("--max-redeals", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Worlds per Lens decision")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--gus-adapter", type=str, default=None,
                        help="Gus adapter for the gus bidder (default: gus/bidding's)")
    parser.add_argument("--belief-bidder-adapter", type=str, default=None,
                        help="Auction-belief adapter for the `belief` bidder "
                             "(rung #26; separate from --gus-adapter). Falls back "
                             "to the spec's belief[:<adapter>] then the #24 default.")
    parser.add_argument("--fast-batching", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Pool both halves into one lockstep batch (twice the "
                             "batch width, one straggler tail; games are "
                             "statistically equivalent, not byte-identical). "
                             "--no-fast-batching runs the sequential halves, whose "
                             "actions are exactly reproducible run-to-run.")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str,
                        default=str(Path(__file__).parent / "results"))
    parser.add_argument("--emit-snapshots", type=str, default=None,
                        help="Write per-hand deal+auction snapshots (JSON) to this "
                             "path for the #26 belief-corpus bridge "
                             "(forge.cli.generate_eq_from_snapshots).")
    parser.add_argument("--emit-decisions", type=str, default=None,
                        help="Write canonical per-play decision JSONL plus a "
                             "checksum/provenance manifest. This post-match "
                             "export does not alter policy play.")
    args = parser.parse_args()

    for spec in (args.team_a, args.team_b):
        if "+" not in spec:
            parser.error(f"Player spec needs <bidder>+<play>, got {spec!r}")

    # Code provenance must describe the policy implementation before the run
    # creates or modifies summary/CSV/decision artifacts in the worktree.
    decision_code_provenance = None
    if args.emit_decisions:
        from arena.decision_records import git_provenance
        decision_code_provenance = git_provenance(PROJECT_ROOT)

    model = None
    oracle_checkpoint = None
    device = args.device
    if needs_model(args.team_a, args.team_b) or needs_gus(args.team_a, args.team_b):
        import torch

        if device == "mps" and not torch.backends.mps.is_available():
            print("MPS unavailable; falling back to CPU.", flush=True)
            device = "cpu"
        torch.manual_seed(args.base_seed)
    if needs_model(args.team_a, args.team_b):
        from forge.zeb.eval.loading import DEFAULT_ORACLE, load_oracle

        oracle_checkpoint = Path(args.checkpoint or PROJECT_ROOT / DEFAULT_ORACLE)
        print(f"Loading oracle: {oracle_checkpoint} on {device}", flush=True)
        model = load_oracle(str(oracle_checkpoint), device)

    bid_a = parse_bidder(
        args.team_a.split("+", 1)[0], device=device, gus_adapter=args.gus_adapter,
        model=model, belief_bidder_adapter=args.belief_bidder_adapter,
    )
    bid_b = parse_bidder(
        args.team_b.split("+", 1)[0], device=device, gus_adapter=args.gus_adapter,
        model=model, belief_bidder_adapter=args.belief_bidder_adapter,
    )
    play_a = parse_play(args.team_a.split("+", 1)[1], model=model,
                        n_samples=args.n_samples, device=device, seed=args.base_seed,
                        gus_adapter=args.gus_adapter)
    play_b = parse_play(args.team_b.split("+", 1)[1], model=model,
                        n_samples=args.n_samples, device=device, seed=args.base_seed + 1,
                        gus_adapter=args.gus_adapter)

    cfg = ArenaConfig(
        marks_to_win=args.marks_to_win,
        max_redeals=args.max_redeals,
        base_seed=args.base_seed,
    )
    print(
        f"Arena: {args.team_a} vs {args.team_b}  "
        f"games={args.n_games} to {cfg.marks_to_win} marks  seed={cfg.base_seed}",
        flush=True,
    )
    result = run_match(
        bid_a=bid_a, bid_b=bid_b, play_a=play_a, play_b=play_b,
        n_games=args.n_games, cfg=cfg,
        label_a=args.team_a, label_b=args.team_b, verbose=True,
        fast_batching=args.fast_batching,
    )

    s = summarize(result)
    print(
        f"\nA ({s['label_a']}) wins {s['a_wins']}/{s['n_games']} "
        f"({s['a_game_win_rate']:.1%}; halves {s['a_win_rate_half1']:.1%} / {s['a_win_rate_half2']:.1%})\n"
        f"mark margin {s['mean_mark_margin']:+.2f}/game "
        f"(95% CI [{s['mark_margin_ci_lo_95']:+.2f}, {s['mark_margin_ci_hi_95']:+.2f}])  "
        f"point margin {s['mean_hand_point_margin']:+.2f}/hand\n"
        f"hands/game {s['mean_hands_per_game']:.1f}  "
        f"A offense share {s['auction']['a_offense_share']:.1%}  "
        f"made: A {s['contracts']['a_offense']['made_rate']:.1%} "
        f"(bid {s['contracts']['a_offense']['mean_bid']:.1f}) / "
        f"B {s['contracts']['b_offense']['made_rate']:.1%} "
        f"(bid {s['contracts']['b_offense']['mean_bid']:.1f})\n"
        f"bids {s['auction']['bid_hist']}\n"
        f"decls {s['auction']['decl_hist']}\n"
        f"({s['elapsed_s']:.1f}s)",
        flush=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        **s,
        "cfg": {
            "marks_to_win": cfg.marks_to_win,
            "max_redeals": cfg.max_redeals,
            "base_seed": cfg.base_seed,
            "n_samples": args.n_samples,
            "device": device,
            "fast_batching": args.fast_batching,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(out_dir / "per_hand.csv", hand_rows(result))
    write_csv(out_dir / "per_game.csv", game_rows(result))
    print(f"\nWrote {out_dir}/summary.json, per_hand.csv, per_game.csv", flush=True)

    if args.emit_snapshots:
        snaps = snapshot_rows(result)
        snap_path = Path(args.emit_snapshots)
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "snapshots": snaps,
            "metadata": {
                "team_a": args.team_a,
                "team_b": args.team_b,
                "n_games": result.n_games,
                "n_snapshots": len(snaps),
                "base_seed": cfg.base_seed,
                "marks_to_win": cfg.marks_to_win,
                "max_redeals": cfg.max_redeals,
            },
        }
        # Compact dump: each row now carries a 28-step play history, so the
        # pretty-printed form would be ~30x the lines for zero information.
        snap_path.write_text(json.dumps(payload) + "\n")
        print(f"Wrote {len(snaps)} snapshots -> {snap_path}", flush=True)

    if args.emit_decisions:
        from arena.decision_records import (
            build_decision_records,
            policy_fingerprint_from_spec,
            write_decision_records,
        )

        common = {
            "n_samples": args.n_samples,
            "device": device,
            "oracle_checkpoint": oracle_checkpoint,
            "gus_adapter": args.gus_adapter,
            "belief_bidder_adapter": args.belief_bidder_adapter,
            "repo_root": PROJECT_ROOT,
            "code_provenance": decision_code_provenance,
        }
        policy_a = policy_fingerprint_from_spec(args.team_a, **common)
        policy_b = policy_fingerprint_from_spec(args.team_b, **common)
        decisions = build_decision_records(
            result, policy_a=policy_a, policy_b=policy_b,
        )
        decision_path = Path(args.emit_decisions)
        manifest_path = write_decision_records(
            decision_path,
            decisions,
            result=result,
            policy_a=policy_a,
            policy_b=policy_b,
            run_metadata={
                "fast_batching": args.fast_batching,
                "n_samples": args.n_samples,
                "device": device,
                "max_redeals": args.max_redeals,
            },
        )
        print(
            f"Wrote {len(decisions)} decisions -> {decision_path}; "
            f"manifest -> {manifest_path}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
