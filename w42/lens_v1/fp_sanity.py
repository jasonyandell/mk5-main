"""fp32 vs fp16 sanity check on the Q-net forward pass at fixed game states.

Method:
  1. Roll out N hands using fp32 + ev-greedy. At each decision, save the
     game state (ZebGameState).
  2. Sample 100 saved states from across the rollout.
  3. Evaluate them as ONE batch with fp32 (with `torch.manual_seed(S)`).
  4. Evaluate them as ONE batch with fp16 (with the same seed S).
  5. Compare argmaxes per utility. Same-state, same-world-sample, only
     model precision differs.

This isolates the precision effect cleanly.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from forge.eq.generate.deals import build_hypothetical_deals
from forge.eq.generate.eq_compute import compute_eq_pdf
from forge.eq.generate.sampling import sample_worlds_batched
from forge.eq.generate.tokenization import tokenize_batched
from forge.eq.sampling_mrv_gpu import WorldSamplerMRV
from forge.eq.tokenize_gpu import GPUTokenizer
from forge.zeb.eq_player import zeb_states_to_game_state_tensor
from forge.zeb.eval.loading import DEFAULT_ORACLE, load_oracle
from forge.zeb.game import apply_action, game_seed, is_terminal, new_game

from w42.lens_v1.lens import argmax_under_utility, UTILITIES
from w42.lens_v1.parallel_match import _force_bid_30


def _set_seed(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if device == "mps":
        torch.mps.manual_seed(seed)


def _forward(model, gst, n_samples, sampler, tokenizer, device, fp_dtype):
    """Run a forward pass; returns (e_q, e_q_pdf). Caller seeds RNG before
    calling so world sampling is reproducible across precisions.
    """
    if fp_dtype == "fp16":
        model = model.half()
    else:
        model = model.float()
    model.eval()
    with torch.no_grad():
        worlds = sample_worlds_batched(gst, sampler, n_samples)
        deals = build_hypothetical_deals(gst, worlds)
        tokens, masks = tokenize_batched(gst, deals, tokenizer)
        current_players = (gst.current_player.unsqueeze(1).expand(-1, n_samples)
                            .reshape(-1).long())
        tokens_in = tokens.to(torch.int32)
        with torch.inference_mode():
            q_values, _ = model(tokens_in, masks, current_players)
        q_values = q_values.float()
        n_g = gst.n_games
        q_reshaped = q_values.view(n_g, n_samples, 7)
        e_q = q_reshaped.mean(dim=1)
        e_q_pdf = compute_eq_pdf(q_reshaped)
        return e_q, e_q_pdf


def collect_states(model, *, n_hands, n_samples, device, base_seed,
                    drive_utility="ev"):
    """Roll out fp32 ev-greedy; collect all decision-point states."""
    states = [_force_bid_30(new_game(seed=game_seed(base_seed, i))) for i in range(n_hands)]
    active = [True] * n_hands

    sampler = WorldSamplerMRV(max_games=n_hands, max_samples=n_samples, device=device)
    tokenizer = GPUTokenizer(max_batch=n_hands * n_samples, device=device)

    saved: list = []
    while any(active):
        active_indices = [i for i, a in enumerate(active) if a]
        active_states = [states[i] for i in active_indices]
        gst = zeb_states_to_game_state_tensor(active_states, device)

        e_q, e_q_pdf = _forward(model, gst, n_samples, sampler, tokenizer, device, "fp32")
        a_chosen = argmax_under_utility(
            utility=drive_utility, e_q=e_q, e_q_pdf=e_q_pdf,
            bidder=gst.bidder.long(),
            current_players=gst.current_player.long(),
            legal_mask=gst.legal_actions(),
            bid_values=[30] * len(active_indices),
        )

        for k, idx in enumerate(active_indices):
            saved.append(active_states[k])
            states[idx] = apply_action(states[idx], int(a_chosen[k].item()))
            if is_terminal(states[idx]):
                active[idx] = False
    return saved


def evaluate_batch(model, states, *, n_samples, device, fp_dtype, seed):
    """Single-batch evaluation of all states. Returns argmaxes per utility."""
    n = len(states)
    sampler = WorldSamplerMRV(max_games=n, max_samples=n_samples, device=device)
    tokenizer = GPUTokenizer(max_batch=n * n_samples, device=device)
    gst = zeb_states_to_game_state_tensor(states, device)

    _set_seed(seed, device)
    e_q, e_q_pdf = _forward(model, gst, n_samples, sampler, tokenizer, device, fp_dtype)

    argmaxes_per_util = {}
    for u in UTILITIES:
        a = argmax_under_utility(
            utility=u, e_q=e_q, e_q_pdf=e_q_pdf,
            bidder=gst.bidder.long(),
            current_players=gst.current_player.long(),
            legal_mask=gst.legal_actions(),
            bid_values=[30] * n,
        )
        argmaxes_per_util[u] = [int(x.item()) for x in a]
    return argmaxes_per_util


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-hands", type=int, default=20,
                        help="Hands to roll out for state collection (small)")
    parser.add_argument("--n-states-sampled", type=int, default=100,
                        help="States to sample from the rollout for fp comparison")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--checkpoint", type=str, default=str(PROJECT_ROOT / DEFAULT_ORACLE))
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).parent / "results"))
    parser.add_argument("--base-seed", type=int, default=99000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    print(f"Loading {args.checkpoint} on {device}...", flush=True)
    model32 = load_oracle(args.checkpoint, device)
    print(f"Phase 1: roll out {args.n_hands} hands fp32 ev-greedy to collect states...", flush=True)
    t = time.time()
    saved = collect_states(
        model32, n_hands=args.n_hands, n_samples=args.n_samples,
        device=device, base_seed=args.base_seed,
    )
    print(f"  {len(saved)} states collected in {time.time()-t:.1f}s", flush=True)

    rng = random.Random(args.base_seed)
    if len(saved) > args.n_states_sampled:
        saved = rng.sample(saved, args.n_states_sampled)
    print(f"Phase 2: evaluate {len(saved)} states with fp32 and fp16 (same seed)...", flush=True)

    eval_seed = 12345
    t = time.time()
    fp32_args = evaluate_batch(model32, saved, n_samples=args.n_samples,
                                device=device, fp_dtype="fp32", seed=eval_seed)
    print(f"  fp32 batch eval: {time.time()-t:.1f}s", flush=True)

    model16 = load_oracle(args.checkpoint, device)
    t = time.time()
    try:
        fp16_args = evaluate_batch(model16, saved, n_samples=args.n_samples,
                                    device=device, fp_dtype="fp16", seed=eval_seed)
        print(f"  fp16 batch eval: {time.time()-t:.1f}s", flush=True)
    except Exception as e:
        print(f"  fp16 EVAL FAILED: {type(e).__name__}: {e}", flush=True)
        out_csv = out_dir / "fp_sanity.csv"
        with out_csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["status", "fp16_error", "n_states", "n_samples"])
            w.writerow(["fp16_failed", str(e), len(saved), args.n_samples])
        print(f"Wrote {out_csv} (fp16 failure recorded)", flush=True)
        return 0

    rows = []
    for u in UTILITIES:
        n_match = sum(1 for a32, a16 in zip(fp32_args[u], fp16_args[u]) if a32 == a16)
        n_total = len(fp32_args[u])
        rows.append({
            "utility": u,
            "n_states_compared": n_total,
            "n_argmax_match": n_match,
            "match_rate": n_match / max(n_total, 1),
            "n_samples": args.n_samples,
        })

    out_csv = out_dir / "fp_sanity.csv"
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_csv}", flush=True)
    for r in rows:
        print(f"  {r['utility']:>12}: {r['n_argmax_match']}/{r['n_states_compared']} = {r['match_rate']*100:.2f}%", flush=True)

    min_match = min(r["match_rate"] for r in rows)
    if min_match < 0.99:
        print(f"\nWARNING: min argmax-match rate {min_match*100:.2f}% < 99%; "
              f"keep round-robin in fp32.", flush=True)
        verdict = "fp32_required"
    else:
        print(f"\nOK: all utilities >= 99% argmax-match. fp16 safe to use.", flush=True)
        verdict = "fp16_safe"
    # Append a verdict row
    with out_csv.open("a", newline="") as fh:
        fh.write(f"\n# verdict: {verdict}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
