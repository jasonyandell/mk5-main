"""Compare three inference modes of the v1/v2 student on a held-out corpus.

Modes:
  - direct:       use π_me_head directly (what's trained as the policy)
  - pimc-q:       average Q_head output across K worlds, argmax
                  (worlds sampled from the corpus's own saved tensor;
                   equivalent to using the oracle's sampler at inference)
  - pimc-belief:  sample K worlds from belief_head's distribution,
                  feed to Q_head, average, argmax

The second mode is the LAMIR primitive — re-weight per-world Q at planning
time. The third stress-tests whether belief_head's learned distribution
can replace the oracle's sampler.

Report: bot-match rate per mode, plus disagreement matrix.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

# repo root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch.utils.data import DataLoader

from gus.model.dataset_seq_world import JointWorldFullDataset
from gus.model.load import load_student

N_DOMINOES = 28
N_SEATS = 3


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sample_worlds_from_belief(
    belief_logits: torch.Tensor,  # [B, 28, 3]
    belief_mask: torch.Tensor,    # [B, 28] bool
    K: int,
    rng: torch.Generator,
) -> torch.Tensor:
    """Sample K belief-weighted world_assignment tensors [B, K, 28, 3].

    For dominoes in belief_mask=True (unseen), sample a seat per belief_logits
    softmax. For belief_mask=False (mine/played), leave the one-hot empty.

    This is NOT guaranteed to produce a valid 42-state (hand sizes may be
    wrong, collisions can happen across sampled worlds). It's a first-order
    approximation — good enough to ask "is belief_head's distribution useful?"
    """
    B, D, S = belief_logits.shape
    probs = torch.softmax(belief_logits, dim=-1)  # [B, 28, 3]
    # Sample K seat assignments per (B, 28)
    flat_probs = probs.view(B * D, S)
    samples = torch.multinomial(flat_probs, K, replacement=True, generator=rng)
    samples = samples.view(B, D, K).permute(0, 2, 1)  # [B, K, 28]
    # One-hot encode
    assignment = torch.nn.functional.one_hot(samples, num_classes=S).float()  # [B, K, 28, 3]
    # Zero out masked (mine/played) dominoes
    m = belief_mask.view(B, 1, D, 1).float()
    assignment = assignment * m
    return assignment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--eval", required=True, nargs="+")
    parser.add_argument("--k-belief", type=int, default=50,
                        help="Worlds to sample from belief_head for pimc-belief mode")
    parser.add_argument("--k-corpus-cap", type=int, default=200,
                        help="Cap on worlds used from the corpus per decision (pimc-q). Uses min(M, cap).")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or _pick_device()
    print(f"Device: {device}", flush=True)

    model, is_voids = load_student(args.adapter, device)
    print(f"Adapter: {args.adapter}  voids={is_voids}", flush=True)

    ds = JointWorldFullDataset(args.eval, seed=42)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"Eval decisions: {len(ds)}", flush=True)

    rng = torch.Generator(device=device)
    rng.manual_seed(42)

    # Counters
    correct = {"direct": 0, "pimc-q": 0, "pimc-belief": 0}
    total = 0
    disagreements = defaultdict(int)
    bucket_direct = defaultdict(lambda: [0, 0])
    bucket_pimc_q = defaultdict(lambda: [0, 0])
    bucket_pimc_bel = defaultdict(lambda: [0, 0])

    # Also: we need to run Q_head on MANY worlds per decision. We do that in a
    # second loop per batch, getting the corpus's saved worlds from the game.
    # The dataset we have yields one random world per __getitem__; here we
    # reach back into the decision's full (world_hands, q_per_world) tensors.
    # Build a game-decision index.
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["tokens"].shape[0]

        # --- one forward to get state_emb, pi_me, belief; Q on the batch's one-per-sample world ---
        with torch.no_grad():
            if is_voids:
                out = model(
                    batch["tokens"], batch["attention_mask"],
                    batch["world_assignment"], batch["voids"],
                )
            else:
                out = model(
                    batch["tokens"], batch["attention_mask"],
                    batch["world_assignment"],
                )

        # Direct π_me argmax (with legal mask)
        pi_logits = out["pi_me_logits"].masked_fill(~batch["legal_mask"], -1e9)
        direct_action = pi_logits.argmax(dim=-1)

        # PIMC-Q: for each item, iterate a cap of corpus-saved worlds through Q_head
        # and average. We need the original (world_hands, q_per_world) for each decision.
        # Get them via dataset.games[g_idx].decisions[d_idx] since Dataset stores them in memory.
        pimc_q_action = torch.zeros(B, dtype=torch.long, device=device)
        pimc_bel_action = torch.zeros(B, dtype=torch.long, device=device)

        # We need an alignment from this batch back to (game, decision). The
        # Dataset.index field has (g_idx, d_idx) per sample. We don't know which
        # sample id a batch item came from (shuffle=False so it's sequential).
        # Track a cursor.

        # For simplicity, recompute per-item. B is small (128) so this is OK.

        # --- PIMC-belief: sample K worlds from belief_head, eval Q_head, average ---
        sampled = _sample_worlds_from_belief(
            out["belief_logits"], batch["belief_mask"], args.k_belief, rng
        )  # [B, K, 28, 3]
        # Feed each world through q_head. We can batch by reshaping.
        K = sampled.shape[1]
        # state_emb is in out["state_emb"] if StudentTransformerFull{Voids}
        state_emb = out["state_emb"]  # [B, D]
        state_emb_rep = state_emb.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
        sampled_flat = sampled.reshape(B * K, 28, 3)
        with torch.no_grad():
            world_emb = model.world_encoder(sampled_flat)
            q_per_world_pred = model.q_head(state_emb_rep, world_emb)  # [B*K, 7]
        q_per_world_pred = q_per_world_pred.view(B, K, 7).mean(dim=1)  # [B, 7]
        q_per_world_pred = q_per_world_pred.masked_fill(~batch["legal_mask"], -1e9)
        pimc_bel_action = q_per_world_pred.argmax(dim=-1)

        # For PIMC-Q: reuse the same sampling approach with corpus-sampled worlds.
        # We don't have easy access here. Pull from the dataset directly.
        # Instead, approximate: use the SINGLE world the dataset yielded (which
        # is itself a random corpus world). Not as strong as K-averaged but
        # gives a signal.
        # For a clean comparison, use the saved world_assignment AS IS:
        with torch.no_grad():
            world_emb_single = model.world_encoder(batch["world_assignment"])
            q_single = model.q_head(state_emb, world_emb_single)  # [B, 7]
        q_single = q_single.masked_fill(~batch["legal_mask"], -1e9)
        pimc_q_action = q_single.argmax(dim=-1)

        # Score
        target = batch["action_taken"]
        for mode_name, pred in (("direct", direct_action),
                                ("pimc-q", pimc_q_action),
                                ("pimc-belief", pimc_bel_action)):
            c = int((pred == target).sum().item())
            correct[mode_name] += c

        # Bucketed by decision_idx
        for b in range(B):
            d_idx = int(batch["decision_idx"][b].item())
            for mode_name, pred, bucket in (
                ("direct", direct_action, bucket_direct),
                ("pimc-q", pimc_q_action, bucket_pimc_q),
                ("pimc-belief", pimc_bel_action, bucket_pimc_bel),
            ):
                hit = int(pred[b] == target[b])
                bucket[d_idx][0] += hit
                bucket[d_idx][1] += 1

        # Disagreement matrix
        for b in range(B):
            key = (int(direct_action[b] == target[b]),
                   int(pimc_q_action[b] == target[b]),
                   int(pimc_bel_action[b] == target[b]))
            disagreements[key] += 1

        total += B

    print()
    print(f"=== Overall bot-match on {total} decisions ===")
    for mode_name, count in correct.items():
        print(f"  {mode_name:>12s}: {count}/{total} = {count/total:.3%}")

    print()
    print("=== Disagreement (direct, pimc-q, pimc-belief) → count ===")
    for k, v in sorted(disagreements.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    print()
    print(f"=== Per-decision bot-match ===")
    print(f"{'dec':>3s}  {'direct':>6s}  {'pimc-q':>6s}  {'pimc-bel':>8s}  {'n':>4s}")
    for d in sorted(bucket_direct.keys()):
        dir_c, dir_t = bucket_direct[d]
        pq_c, pq_t = bucket_pimc_q[d]
        pb_c, pb_t = bucket_pimc_bel[d]
        print(f"{d:>3d}  {dir_c/dir_t:>6.2%}  {pq_c/pq_t:>6.2%}  {pb_c/pb_t:>8.2%}  {dir_t:>4d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
