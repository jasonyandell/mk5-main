"""jud v1 loop — the one-organ self-play loop (#33).

Round r (1..R):
  1. GENERATE jud(r-1)+judplay(r-1) SELF-PLAY (both teams the current head; no
     oracle at runtime) with snapshot emission → on-policy bid AND play data.
  2. RETRAIN jud_net on the CUMULATIVE corpus: base chunks (mixed regimes:
     margin:wp+lens:ev, net:wp+lens:ev, random+random) + every round's
     self-play + every prior round's A/B snaps (v0's cumulative recipe — the
     recipe-fork winner). Save versioned champion/jud_net_r{r}.pt.
  3. A/B jud_r+judplay_r vs net:wp+lens:ev (E[Q] n=10), 256 games — measures
     the round AND feeds round r+1 (mixed-opponent corpus).
  4. EVAL: held-out reliability + per-trick calibration slices.

Round 0 (r=0) trains the base head and runs the GRADED round-0 A/Bs
(full-stack + bid-only + play-only) per scratch/jud-v0/jud_v1_predictions.md.
Graded/definitive measurements use reserved seeds 7000000/9000000 and are
NEVER fed to training; loop rounds use the 13M seed region.

Run:  PYTHONPATH=<repo> forge/venv/bin/python -u scratch/jud-v1/run_jud_loop.py [ROUNDS] [--train-device cpu|mps]
Resumable: a round with BOTH metrics_r{r}.json and jud_net_r{r}.pt is skipped.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from arena.engine import ArenaConfig
from arena.match import run_match, snapshot_rows, summarize
from arena.lens_play import LensPlay
from arena.jud_play import JudPlay
from champion import jud_net as JN
from champion.value_bidder import ValueBidder
from champion.bidder import GusBidder, NetPointsEvaluator
from champion.utility import MarksToSeven
from forge.zeb.eval.loading import load_oracle

# --------------------------------------------------------------------- #
LOOP = Path("scratch/jud-v1/loop")
BASE = sorted(Path("scratch/jud-v1/corpus").glob("snaps_*.json"))
CKPT = "forge/models/domino-large-817k-valuehead-acc97.8-qgap0.07.ckpt"
DEVICE = "mps"          # oracle (lens side of A/Bs) only
JUD_DEVICE = "cpu"      # 470k-param MLP; CPU dispatch beats MPS at this size
N_SAMPLES = 10
SP_GAMES = 1000
AB_GAMES = 256
SEED_BASE = 13_000_000
RESERVED = (7_000_000, 9_000_000)  # graded/definitive only; never trained on

LOOP.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write_snaps(path: Path, res, meta: dict) -> None:
    path.write_text(json.dumps({"snapshots": snapshot_rows(res), "metadata": meta}) + "\n")


def _jud_team(head_path):
    head = JN.load_jud_net(head_path, JUD_DEVICE)
    return ValueBidder(head, MarksToSeven()), JudPlay(head, JUD_DEVICE)


def _net_team(model):
    return (GusBidder(pmake_fn=NetPointsEvaluator(), utility=MarksToSeven()),
            LensPlay(model, "ev", N_SAMPLES, DEVICE))


def _summ(res) -> dict:
    s = summarize(res)
    c = s["contracts"]["a_offense"]
    return {
        "a_wins": s["a_wins"], "n_games": s["n_games"],
        "mean_mark_margin": round(s["mean_mark_margin"], 3),
        "ci_lo": round(s["mark_margin_ci_lo_95"], 3),
        "ci_hi": round(s["mark_margin_ci_hi_95"], 3),
        "a_offense_share": round(s["auction"]["a_offense_share"], 4),
        "a_made_rate": round(c["made_rate"], 4),
        "a_mean_bid": round(c["mean_bid"], 2),
        "point_margin": round(s["mean_hand_point_margin"], 3),
        "decl_hist": s["auction"]["decl_hist"],
    }


def self_play(head_path, r: int) -> Path:
    out = LOOP / f"sp_r{r}.json"
    if out.exists():
        log(f"  round {r} self-play: cached")
        return out
    bid, play = _jud_team(head_path)
    res = run_match(
        bid_a=bid, bid_b=bid, play_a=play, play_b=play,
        n_games=SP_GAMES, cfg=ArenaConfig(base_seed=SEED_BASE + r * 100_000),
        label_a="jud", label_b="jud", verbose=True, fast_batching=True,
    )
    s = summarize(res)
    _write_snaps(out, res, {"regime": "jud self-play", "round": r, "head": str(head_path)})
    log(f"  round {r} self-play: made {s['contracts']['a_offense']['made_rate']:.3f}, "
        f"mean_bid {s['contracts']['a_offense']['mean_bid']:.2f}, {s['elapsed_s']:.0f}s")
    return out


def ab_vs_champion(head_path, r: int, model, n_games: int, seed: int,
                   tag: str, emit: bool) -> dict:
    """jud+judplay vs net:wp+lens:ev. emit=False for graded seeds (never trained on)."""
    out = LOOP / f"ab_{tag}.json"
    summ_path = LOOP / f"ab_{tag}_summary.json"
    if summ_path.exists():
        return json.loads(summ_path.read_text())
    bid, play = _jud_team(head_path)
    res = run_match(
        bid_a=bid, bid_b=_net_team(model)[0], play_a=play, play_b=_net_team(model)[1],
        n_games=n_games, cfg=ArenaConfig(base_seed=seed),
        label_a="jud+judplay", label_b="net:wp+lens:ev", verbose=True, fast_batching=True,
    )
    if emit:
        _write_snaps(out, res, {"regime": "jud vs net:wp A/B", "round": r})
    s = _summ(res)
    summ_path.write_text(json.dumps(s, indent=2) + "\n")
    return s


def train_round(r: int, device: str) -> Path:
    head = Path(f"champion/jud_net_r{r}.pt")
    corpus = ([str(p) for p in BASE]
              + [str(LOOP / f"sp_r{k}.json") for k in range(1, r + 1)]
              + [str(LOOP / f"ab_r{k}.json") for k in range(1, r)])
    log(f"round {r}: train on {len(corpus)} corpus files (cumulative)")
    if not head.exists():
        JN.train(corpus=corpus, out_model=head, epochs=40, patience=5,
                 lr=3e-4, device=device)
    return head


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    train_device = "mps"
    if "--train-device" in sys.argv:
        train_device = sys.argv[sys.argv.index("--train-device") + 1]
    log(f"jud v1 loop: rounds={rounds}, base chunks={len(BASE)}, train_device={train_device}")
    assert BASE, "no base corpus chunks — run gen_corpus.sh first"
    model = load_oracle(CKPT, DEVICE)

    metrics: list[dict] = []
    # ---- round 0: base head + graded measurements (JP1/JP2) ----
    head0 = Path("champion/jud_net_r0.pt")
    m0path = LOOP / "metrics_r0.json"
    if m0path.exists() and head0.exists():
        metrics.append(json.loads(m0path.read_text()))
        log("round 0: cached — skipping")
    else:
        t0 = time.time()
        if not head0.exists():
            log(f"round 0: train base head on {len(BASE)} chunks")
            JN.train(corpus=[str(p) for p in BASE], out_model=head0,
                     epochs=40, patience=5, lr=3e-4, device=train_device)
        log("round 0 GRADED: full-stack 512 @ reserved seed 7M (JP1)")
        full = ab_vs_champion(head0, 0, model, 512, RESERVED[0], "r0_full512", emit=False)
        log(f"  JP1 full-stack: {full['mean_mark_margin']:+.2f} "
            f"[{full['ci_lo']:+.2f},{full['ci_hi']:+.2f}] pts {full['point_margin']:+.2f}")
        log("round 0 GRADED: bid-only 256 (JP2: jud+lens:ev vs net:wp+lens:ev)")
        bid0, _ = _jud_team(head0)
        res = run_match(bid_a=bid0, bid_b=_net_team(model)[0],
                        play_a=_net_team(model)[1], play_b=_net_team(model)[1],
                        n_games=256, cfg=ArenaConfig(base_seed=7_100_000),
                        label_a="jud+lens:ev", label_b="net:wp+lens:ev",
                        verbose=True, fast_batching=True)
        bidonly = _summ(res)
        (LOOP / "ab_r0_bidonly_summary.json").write_text(json.dumps(bidonly, indent=2) + "\n")
        log(f"  JP2 bid-only: {bidonly['mean_mark_margin']:+.2f} "
            f"[{bidonly['ci_lo']:+.2f},{bidonly['ci_hi']:+.2f}]")
        ev = JN.evaluate(corpus=[str(p) for p in BASE], model_path=head0,
                         out_json=LOOP / "eval_r0.json", device="cpu")
        m = {"round": 0, "head": str(head0),
             "ab_full512": full, "ab_bidonly256": bidonly,
             "eval": {k: ev[k] for k in ("test_ce", "mae_mean_pts", "ece_p30")
                      if k in ev}}
        m0path.write_text(json.dumps(m, indent=2) + "\n")
        metrics.append(m)
        log(f"round 0 DONE {time.time()-t0:.0f}s")

    # ---- loop rounds ----
    prev_head = str(head0)
    for r in range(1, rounds + 1):
        mpath = LOOP / f"metrics_r{r}.json"
        head_r = Path(f"champion/jud_net_r{r}.pt")
        if mpath.exists() and head_r.exists():
            metrics.append(json.loads(mpath.read_text()))
            prev_head = str(head_r)
            log(f"round {r}: cached — skipping")
            continue
        t0 = time.time()
        log(f"round {r}: self-play (head={prev_head})")
        self_play(prev_head, r)
        head = train_round(r, train_device)
        log(f"round {r}: A/B vs net:wp+lens:ev ({AB_GAMES} games)")
        ab = ab_vs_champion(head, r, model, AB_GAMES,
                            SEED_BASE + r * 100_000 + 50_000, f"r{r}", emit=True)
        ev = JN.evaluate(corpus=[str(p) for p in BASE], model_path=head,
                         out_json=LOOP / f"eval_r{r}.json", device="cpu")
        m = {"round": r, "head": str(head), "ab": ab,
             "eval": {k: ev[k] for k in ("test_ce", "mae_mean_pts", "ece_p30")
                      if k in ev}}
        mpath.write_text(json.dumps(m, indent=2) + "\n")
        metrics.append(m)
        prev_head = str(head)
        log(f"round {r} DONE {time.time()-t0:.0f}s: {ab['mean_mark_margin']:+.2f} "
            f"[{ab['ci_lo']:+.2f},{ab['ci_hi']:+.2f}] offense={ab['a_offense_share']:.1%} "
            f"made={ab['a_made_rate']:.1%}")

    (LOOP / "loop_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    log("=== jud v1 cross-round summary ===")
    for m in metrics:
        ab = m.get("ab") or m.get("ab_full512")
        log(f"r{m['round']:>2} {ab['mean_mark_margin']:>+7.2f} "
            f"[{ab['ci_lo']:>+5.2f},{ab['ci_hi']:>+5.2f}] "
            f"offense {ab['a_offense_share']:.1%} made {ab['a_made_rate']:.1%}")
    log(f"wrote {LOOP / 'loop_metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
