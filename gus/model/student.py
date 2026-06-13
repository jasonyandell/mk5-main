"""Student network: shared state encoder + heads.

v0: MLP encoder + belief_head only — FAILED to generalize (overfits to
~34% eval acc ≈ chance). Kept for reference.

v1: transformer encoder over tokenized play sequence + belief_head. The
transformer sees each play as a positional token, enabling it to learn
void inference attentionally — the signal v0's bag-of-features couldn't
express.

Future: V_head, π_me_head, world-conditioned Q_head on top of the
transformer state encoder.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .auction import N_AUCTION_FEATURES
from .features import FEATURE_DIM, N_DOMINOES
from .tokenize import (
    N_PLAYER_REL,
    N_TRICK_POS,
    N_TRICK_SLOTS,
    N_TYPES,
    SEQ_LEN,
    TOKEN_VOCAB_SIZE,
)

DEFAULT_HIDDEN_DIM = 256


class StateEncoder(nn.Module):
    """Simple MLP state encoder. 2 hidden layers, GELU, layer-norm."""

    def __init__(self, in_dim: int = FEATURE_DIM, hidden_dim: int = DEFAULT_HIDDEN_DIM, out_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BeliefHead(nn.Module):
    """Per-domino 3-way classifier: P(domino ∈ {left_opp, partner, right_opp}).

    Output shape: [B, 28, 3] logits. Downstream cross-entropy loss masked to
    unseen dominoes.
    """

    def __init__(self, in_dim: int = DEFAULT_HIDDEN_DIM, n_dominoes: int = N_DOMINOES, n_seats: int = 3):
        super().__init__()
        self.n_dominoes = n_dominoes
        self.n_seats = n_seats
        self.proj = nn.Linear(in_dim, n_dominoes * n_seats)

    def forward(self, z: Tensor) -> Tensor:
        logits = self.proj(z)  # [B, 28*3]
        return logits.view(-1, self.n_dominoes, self.n_seats)  # [B, 28, 3]


class StudentV0(nn.Module):
    """State encoder + belief head. v0 sanity-check student."""

    def __init__(self, hidden_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.encoder = StateEncoder(hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.belief = BeliefHead(in_dim=hidden_dim)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        z = self.encoder(features)
        belief_logits = self.belief(z)
        return {"belief_logits": belief_logits, "z": z}


class PlayEmbedding(nn.Module):
    """Sum of embeddings for the five channels of a play token.

    Input: [B, L, 5] long
    Output: [B, L, d_model]

    Note: attribute names avoid shadowing nn.Module methods (`type`, `float`,
    etc.) — hence the `_emb` suffix.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.tok_emb = nn.Embedding(TOKEN_VOCAB_SIZE, d_model)
        self.type_emb = nn.Embedding(N_TYPES, d_model)
        self.trick_emb = nn.Embedding(N_TRICK_SLOTS, d_model)
        self.pos_emb = nn.Embedding(N_TRICK_POS, d_model)
        self.player_rel_emb = nn.Embedding(N_PLAYER_REL, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: [B, L, 5] where channels are (token, type, trick, pos, player_rel)
        t = tokens[..., 0]
        ty = tokens[..., 1]
        tr = tokens[..., 2]
        po = tokens[..., 3]
        pr = tokens[..., 4]
        emb = (
            self.tok_emb(t)
            + self.type_emb(ty)
            + self.trick_emb(tr)
            + self.pos_emb(po)
            + self.player_rel_emb(pr)
        )
        return self.norm(emb)


class TransformerEncoder(nn.Module):
    """Small transformer over the [L, 5] play-sequence tokens.

    Attention mask: PAD positions (token==PAD) are excluded from attention.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = PlayEmbedding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, tokens: Tensor, attention_mask: Tensor) -> Tensor:
        """
        tokens: [B, L, 5] long
        attention_mask: [B, L] bool — True where real
        Returns: [B, L, d_model]
        """
        x = self.embed(tokens)  # [B, L, d_model]
        # TransformerEncoder expects src_key_padding_mask as True where PAD.
        padding_mask = ~attention_mask  # True where pad
        x = self.blocks(x, src_key_padding_mask=padding_mask)
        return x


class StudentTransformer(nn.Module):
    """Transformer-encoder student. v1 — belief head only for now; heads for
    V/π_me/world-Q will be added after belief is proven to learn."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        # Use the CLS position (index 0) as the pooled state embedding.
        self.belief = BeliefHead(in_dim=d_model)

    def forward(self, tokens: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        h = self.encoder(tokens, attention_mask)  # [B, L, d_model]
        cls_h = h[:, 0, :]  # [B, d_model] — CLS token's encoding
        belief_logits = self.belief(cls_h)  # [B, 28, 3]
        return {"belief_logits": belief_logits, "z": cls_h, "hidden": h}


class WorldEncoder(nn.Module):
    """Encode a single world's [28, 3] seat-one-hot assignment into a latent."""

    def __init__(self, d_model: int, d_world: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_DOMINOES * 3, d_world),
            nn.GELU(),
            nn.LayerNorm(d_world),
            nn.Linear(d_world, d_world),
        )
        self.d_world = d_world

    def forward(self, world_assignment: Tensor) -> Tensor:
        """world_assignment: [B, 28, 3] float → [B, d_world] latent."""
        B = world_assignment.shape[0]
        flat = world_assignment.reshape(B, -1)
        return self.net(flat)


class QHead(nn.Module):
    """Q per action given fused (state_emb, world_emb). Output [B, 7]."""

    def __init__(self, state_dim: int, world_dim: int, hidden_dim: int = 256, n_actions: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + world_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state_emb: Tensor, world_emb: Tensor) -> Tensor:
        fused = torch.cat([state_emb, world_emb], dim=-1)
        return self.net(fused)


class StudentTransformerFull(nn.Module):
    """Five-head LAMIR-ready student (minus π_opp which is deferred to v2).

    Four heads sharing a transformer state encoder:
      - belief: [B, 28, 3] seat-per-domino logits (supervised against truth)
      - V:      [B] scalar (supervised against oracle E[Q] for action_taken)
      - π_me:   [B, 7] action softmax logits (supervised against oracle argmax)
      - Q:      [B, 7] world-conditioned Q per action (supervised against
                oracle per-world Q for one sampled world per forward pass)
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        d_world: int = 64,
        q_hidden: int = 256,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.belief = BeliefHead(in_dim=d_model)
        self.v_head = nn.Linear(d_model, 1)
        self.pi_me = nn.Linear(d_model, 7)
        self.world_encoder = WorldEncoder(d_model, d_world=d_world)
        self.q_head = QHead(state_dim=d_model, world_dim=d_world, hidden_dim=q_hidden)

    def forward(
        self,
        tokens: Tensor,
        attention_mask: Tensor,
        world_assignment: Tensor,
    ) -> dict[str, Tensor]:
        """
        tokens:            [B, L, 5] long
        attention_mask:    [B, L] bool
        world_assignment:  [B, 28, 3] float
        """
        h = self.encoder(tokens, attention_mask)  # [B, L, d_model]
        state_emb = h[:, 0, :]                    # [B, d_model]

        belief_logits = self.belief(state_emb)    # [B, 28, 3]
        v = self.v_head(state_emb).squeeze(-1)    # [B]
        pi_me_logits = self.pi_me(state_emb)      # [B, 7]

        world_emb = self.world_encoder(world_assignment)  # [B, d_world]
        q = self.q_head(state_emb, world_emb)     # [B, 7]

        return {
            "belief_logits": belief_logits,
            "v": v,
            "pi_me_logits": pi_me_logits,
            "q": q,
            "state_emb": state_emb,
            "world_emb": world_emb,
        }


def v1_full_loss(
    out: dict[str, Tensor],
    batch: dict[str, Tensor],
    weights: dict[str, float] | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Joint loss for the full student. Returns (loss_total, per_head_scalars)."""
    if weights is None:
        weights = {"belief": 1.0, "v": 0.5, "pi_me": 0.5, "q": 1.0}

    # --- Belief loss (masked CE per unseen domino slot) ---
    L_belief = belief_loss(out["belief_logits"], batch["belief_target"], batch["belief_mask"])

    # --- V loss: MSE against e_q[action_taken] ---
    B = batch["e_q"].shape[0]
    e_q_at_action = batch["e_q"][torch.arange(B, device=batch["e_q"].device), batch["action_taken"]]
    L_v = nn.functional.mse_loss(out["v"], e_q_at_action)

    # --- π_me loss: CE against action_taken, masked to legal actions ---
    # Use a large negative (not -inf) so label smoothing and gradients stay finite.
    pi_logits = out["pi_me_logits"].clone()
    pi_logits = pi_logits.masked_fill(~batch["legal_mask"], -1e9)
    L_pi = nn.functional.cross_entropy(pi_logits, batch["action_taken"])

    # --- Q loss: MSE against q_per_world, masked to legal actions ---
    q_pred = out["q"]
    q_target = batch["q_per_world"]
    legal_f = batch["legal_mask"].float()
    sq_err = (q_pred - q_target) ** 2 * legal_f
    n_legal = legal_f.sum().clamp(min=1.0)
    L_q = sq_err.sum() / n_legal

    total = (
        weights["belief"] * L_belief
        + weights["v"] * L_v
        + weights["pi_me"] * L_pi
        + weights["q"] * L_q
    )

    return total, {
        "L_total": float(total.item()),
        "L_belief": float(L_belief.item()),
        "L_v": float(L_v.item()),
        "L_pi": float(L_pi.item()),
        "L_q": float(L_q.item()),
    }


def pi_me_accuracy(pi_logits: Tensor, legal_mask: Tensor, action_taken: Tensor) -> tuple[int, int]:
    """Bot-match rate: fraction of legal-argmax(pi_me) == action_taken."""
    masked = pi_logits.clone()
    masked[~legal_mask] = float("-inf")
    preds = masked.argmax(dim=-1)
    return int((preds == action_taken).sum().item()), int(action_taken.numel())


def v3_consistency_loss(
    out: dict[str, Tensor],
    batch: dict[str, Tensor],
    weights: dict[str, float] | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Joint loss for v3: v1_full_loss + V/π consistency regularizer.

    Adds ``L_consistency = (V_head.detach() - policy_expected_Q)^2`` where
    ``policy_expected_Q = Σ_legal softmax(pi_me_logits) * e_q``. Penalizes
    the observed pathology where V_head predicts the oracle's max legal
    E[Q] correctly but π_me concentrates probability on suboptimal actions
    (blunder forensics receipt #4). Detaching V keeps the gradient flowing
    toward π only — don't let consistency pressure corrupt the value head.

    Returns (total_loss, per-head scalar dict).
    """
    if weights is None:
        weights = {"belief": 1.0, "v": 0.5, "pi_me": 0.5, "q": 1.0, "consistency": 0.3}

    # --- Belief loss (masked CE per unseen domino slot) ---
    L_belief = belief_loss(out["belief_logits"], batch["belief_target"], batch["belief_mask"])

    # --- V loss: MSE against e_q[action_taken] ---
    B = batch["e_q"].shape[0]
    e_q_at_action = batch["e_q"][torch.arange(B, device=batch["e_q"].device), batch["action_taken"]]
    L_v = nn.functional.mse_loss(out["v"], e_q_at_action)

    # --- π_me loss: CE against action_taken, legal-masked ---
    pi_logits = out["pi_me_logits"].masked_fill(~batch["legal_mask"], -1e9)
    L_pi = nn.functional.cross_entropy(pi_logits, batch["action_taken"])

    # --- Q loss: MSE against q_per_world on legal actions ---
    legal_f = batch["legal_mask"].float()
    sq_err = (out["q"] - batch["q_per_world"]) ** 2 * legal_f
    n_legal = legal_f.sum().clamp(min=1.0)
    L_q = sq_err.sum() / n_legal

    # --- Consistency loss: V_head.detach() vs policy-expected Q ---
    # softmax over legal actions only
    pi_softmax = torch.softmax(pi_logits, dim=-1)  # [B, 7]
    policy_expected_q = (pi_softmax * batch["e_q"]).sum(dim=-1)  # [B]
    L_consistency = nn.functional.mse_loss(policy_expected_q, out["v"].detach())

    total = (
        weights["belief"] * L_belief
        + weights["v"] * L_v
        + weights["pi_me"] * L_pi
        + weights["q"] * L_q
        + weights["consistency"] * L_consistency
    )

    return total, {
        "L_total": float(total.item()),
        "L_belief": float(L_belief.item()),
        "L_v": float(L_v.item()),
        "L_pi": float(L_pi.item()),
        "L_q": float(L_q.item()),
        "L_consistency": float(L_consistency.item()),
    }


# --- v2: voids-aware student --------------------------------------------------

N_VOIDS_FEATURES = 24  # 3 relative opponents × 8 suits


class VoidsEncoder(nn.Module):
    """Small MLP that projects the [24]-dim void indicator vector into d_model.

    The output is added to the pooled state embedding before the heads run —
    explicit void evidence gets mixed into whatever the transformer has
    inferred attentionally.
    """

    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_VOIDS_FEATURES, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, voids: Tensor) -> Tensor:
        return self.norm(self.net(voids))


class StudentTransformerFullVoids(nn.Module):
    """v2: transformer encoder + VoidsEncoder + four heads.

    Identical to StudentTransformerFull except the pooled state_emb is
    `cls_h + voids_encoder(voids)` before any head runs. The transformer
    has to learn voids implicitly from play tokens; this lets it ALSO see
    them explicitly, closing the sparse-signal belief gap.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        d_world: int = 64,
        q_hidden: int = 256,
        voids_hidden: int = 64,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.voids_encoder = VoidsEncoder(d_model, hidden_dim=voids_hidden)
        self.belief = BeliefHead(in_dim=d_model)
        self.v_head = nn.Linear(d_model, 1)
        self.pi_me = nn.Linear(d_model, 7)
        self.world_encoder = WorldEncoder(d_model, d_world=d_world)
        self.q_head = QHead(state_dim=d_model, world_dim=d_world, hidden_dim=q_hidden)

    def forward(
        self,
        tokens: Tensor,
        attention_mask: Tensor,
        world_assignment: Tensor,
        voids: Tensor,
    ) -> dict[str, Tensor]:
        h = self.encoder(tokens, attention_mask)
        cls_h = h[:, 0, :]
        state_emb = cls_h + self.voids_encoder(voids)

        belief_logits = self.belief(state_emb)
        v = self.v_head(state_emb).squeeze(-1)
        pi_me_logits = self.pi_me(state_emb)

        world_emb = self.world_encoder(world_assignment)
        q = self.q_head(state_emb, world_emb)

        return {
            "belief_logits": belief_logits,
            "v": v,
            "pi_me_logits": pi_me_logits,
            "q": q,
            "state_emb": state_emb,
            "world_emb": world_emb,
        }


# --- #24: auction-conditioned student -----------------------------------------


class BidsEncoder(nn.Module):
    """Project the [18]-dim auction feature into d_model.

    The output is added to the pooled state embedding alongside the void
    evidence — the same explicit-feature injection VoidsEncoder uses, applied to
    the completed auction (rung #24). The auction is the signal the play-evidence
    belief head was missing at the Bayes ceiling.
    """

    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_AUCTION_FEATURES, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, bids: Tensor) -> Tensor:
        return self.norm(self.net(bids))


class StudentTransformerFullVoidsAuction(nn.Module):
    """v2+auction: transformer encoder + VoidsEncoder + BidsEncoder + four heads.

    Identical to StudentTransformerFullVoids except the pooled state_emb is
    `cls_h + voids_encoder(voids) + bids_encoder(bids)` before any head runs.
    The transformer sees play tokens; voids inject explicit follow-failure
    evidence; the auction injects who-bid-what-and-won. Conditioning belief on
    the auction is rung #24 — the unlock #25's null pointed at.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        d_world: int = 64,
        q_hidden: int = 256,
        voids_hidden: int = 64,
        bids_hidden: int = 64,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.voids_encoder = VoidsEncoder(d_model, hidden_dim=voids_hidden)
        self.bids_encoder = BidsEncoder(d_model, hidden_dim=bids_hidden)
        self.belief = BeliefHead(in_dim=d_model)
        self.v_head = nn.Linear(d_model, 1)
        self.pi_me = nn.Linear(d_model, 7)
        self.world_encoder = WorldEncoder(d_model, d_world=d_world)
        self.q_head = QHead(state_dim=d_model, world_dim=d_world, hidden_dim=q_hidden)

    def forward(
        self,
        tokens: Tensor,
        attention_mask: Tensor,
        world_assignment: Tensor,
        voids: Tensor,
        bids: Tensor,
    ) -> dict[str, Tensor]:
        h = self.encoder(tokens, attention_mask)
        cls_h = h[:, 0, :]
        state_emb = cls_h + self.voids_encoder(voids) + self.bids_encoder(bids)

        belief_logits = self.belief(state_emb)
        v = self.v_head(state_emb).squeeze(-1)
        pi_me_logits = self.pi_me(state_emb)

        world_emb = self.world_encoder(world_assignment)
        q = self.q_head(state_emb, world_emb)

        return {
            "belief_logits": belief_logits,
            "v": v,
            "pi_me_logits": pi_me_logits,
            "q": q,
            "state_emb": state_emb,
            "world_emb": world_emb,
        }


def belief_loss(
    logits: Tensor,    # [B, 28, 3]
    target: Tensor,    # [B, 28] long in {0,1,2}
    mask: Tensor,      # [B, 28] bool
) -> Tensor:
    """Cross-entropy over unseen-domino slots only. Returns scalar mean."""
    # Flatten to [B*28, 3] and [B*28] / [B*28]
    B, D, S = logits.shape
    flat_logits = logits.reshape(B * D, S)
    flat_target = target.reshape(B * D)
    flat_mask = mask.reshape(B * D)
    if flat_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    # Select only masked positions
    sel_logits = flat_logits[flat_mask]
    sel_target = flat_target[flat_mask]
    return nn.functional.cross_entropy(sel_logits, sel_target)


def belief_accuracy(
    logits: Tensor,    # [B, 28, 3]
    target: Tensor,    # [B, 28]
    mask: Tensor,      # [B, 28]
) -> tuple[int, int]:
    """Return (n_correct, n_total) over unseen-domino slots."""
    preds = logits.argmax(dim=-1)  # [B, 28]
    sel = mask
    n_total = int(sel.sum().item())
    if n_total == 0:
        return 0, 0
    n_correct = int(((preds == target) & sel).sum().item())
    return n_correct, n_total
