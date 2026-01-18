# E[Q] Training MVP

**Goal:** Validate that Stage 2 (imperfect-info policy) can learn from E[logits] labels generated using the existing perfect-info model.

**Non-goal:** Production-quality data, optimal hyperparameters, true Q-values.

---

## The Core Insight

We already have Stage 1: `domino-large-817k-valuehead-acc97.8-qgap0.07.ckpt`

This model sees all 4 hands and outputs action logits. The logits are trained via `softmax(Q/T)` distillation, so:

```
logits ≈ Q/T + constant
argmax(logits) = argmax(Q)
```

**For E[Q] move selection, logits work identically to Q.** The constant cancels when comparing moves.

---

## Two-Stage Architecture

```
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: Existing Perfect-Info Model                  │
│                                                         │
│  Input:  All 4 hands + game state (32 tokens)          │
│  Output: 7 action logits + state value                 │
│  Role:   Fast oracle for E[Q] sampling                 │
│                                                         │
│  forge/models/domino-large-817k-valuehead-*.ckpt       │
└─────────────────────────────────────────────────────────┘
                          │
                          │ Query N=100 sampled worlds
                          ▼
┌─────────────────────────────────────────────────────────┐
│  DATA GENERATION                                        │
│                                                         │
│  For each decision:                                     │
│    1. Infer voids from play history                    │
│    2. Sample N consistent worlds                        │
│    3. Query Stage 1 for logits on each world           │
│    4. E[logits] = mean(logits across worlds)           │
│    5. Play argmax(E[logits])                           │
│    6. Record (transcript, E[logits])                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: Imperfect-Info Policy (TO BUILD)             │
│                                                         │
│  Input:  Transcript only (public information)          │
│  Output: E[logits] for each of 7 actions               │
│  Role:   Deployed player (single forward pass)         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## MVP Scope

| Component | MVP | Future |
|-----------|-----|--------|
| Games generated | 100-1000 | 100K |
| Samples per decision | 100 | 1000 |
| Stage 2 model | 2 layers, 64 dim | 4 layers, 128 dim |
| Training | Converges? | Hyperparameter sweep |
| Evaluation | Loss decreases, move agreement | Full metrics suite |

---

## Implementation

### 1. Void Inference

Players reveal voids by failing to follow suit. Critical for filtering consistent worlds.

```python
# forge/eq/voids.py

from forge.oracle.tables import can_follow, led_suit_for_lead_domino

def infer_voids(
    plays: list[tuple[int, int, int]],  # (player, domino_id, leader_domino_id)
    decl_id: int
) -> dict[int, set[int]]:
    """
    Infer which suits each player is void in.

    Args:
        plays: List of (player, domino_played, domino_that_led_trick)
        decl_id: Declaration (0-9)

    Returns:
        {player: {void_suits}} where suit is 0-6 or 7 (called suit)
    """
    voids = {0: set(), 1: set(), 2: set(), 3: set()}

    for player, domino_id, leader_domino_id in plays:
        led_suit = led_suit_for_lead_domino(leader_domino_id, decl_id)
        if not can_follow(domino_id, led_suit, decl_id):
            voids[player].add(led_suit)

    return voids
```

### 2. World Sampling

```python
# forge/eq/sampling.py

import random
from forge.oracle.tables import can_follow

def sample_consistent_worlds(
    my_player: int,
    my_hand: list[int],
    played: set[int],
    voids: dict[int, set[int]],
    decl_id: int,
    n_samples: int = 100
) -> list[list[list[int]]]:
    """
    Sample opponent hand configurations consistent with observed voids.

    Returns list of deals: [p0_hand, p1_hand, p2_hand, p3_hand]
    """
    remaining = [d for d in range(28) if d not in my_hand and d not in played]
    other_players = [p for p in range(4) if p != my_player]

    # How many dominoes each player still has
    # (simplified: assume 7 - len(played_by_player))

    worlds = []
    attempts = 0
    max_attempts = n_samples * 100

    while len(worlds) < n_samples and attempts < max_attempts:
        attempts += 1
        random.shuffle(remaining)

        # Distribute remaining dominoes to other players
        hands = {my_player: my_hand[:]}
        idx = 0
        valid = True

        for p in other_players:
            # For MVP: assume all others have same count as remaining/3
            size = len(remaining) // 3
            if p == other_players[-1]:
                size = len(remaining) - idx  # Last player gets remainder

            player_hand = remaining[idx:idx + size]
            idx += size

            # Check void constraints
            for domino in player_hand:
                for void_suit in voids[p]:
                    if can_follow(domino, void_suit, decl_id):
                        valid = False
                        break
                if not valid:
                    break

            if not valid:
                break
            hands[p] = player_hand

        if valid:
            worlds.append([hands.get(p, []) for p in range(4)])

    if len(worlds) < n_samples:
        print(f"Warning: only {len(worlds)} valid worlds (wanted {n_samples})")

    return worlds
```

### 3. Stage 1 Query Interface

```python
# forge/eq/oracle.py

import torch
from pathlib import Path
from forge.ml.module import DominoLightningModule
from forge.ml.tokenize import process_shard  # Reuse tokenization logic

class Stage1Oracle:
    """Wrapper for querying the perfect-info model."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cuda"):
        self.model = DominoLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location=device
        )
        self.model.eval()
        self.model.to(device)
        self.device = device

    def query_batch(
        self,
        worlds: list[dict],  # Each has: hands, remaining, trick, decl_id
        current_player: int
    ) -> torch.Tensor:
        """
        Query logits for a batch of complete worlds.

        Returns: (N, 7) logits tensor
        """
        # Tokenize each world using existing logic
        # This requires adapting process_shard or writing new tokenizer

        tokens, masks = self._tokenize_worlds(worlds)
        players = torch.full((len(worlds),), current_player, dtype=torch.long)

        with torch.no_grad():
            logits, _ = self.model(
                tokens.to(self.device),
                masks.to(self.device),
                players.to(self.device)
            )

        return logits.cpu()

    def _tokenize_worlds(self, worlds):
        """Convert world dicts to model input tensors."""
        # TODO: Implement - similar to process_shard but for arbitrary hands
        raise NotImplementedError("MVP: implement this")
```

### 4. Game Generation Loop

```python
# forge/eq/generate.py

def generate_eq_game(oracle: Stage1Oracle, n_samples: int = 100) -> GameRecord:
    """Generate one game using E[logits] policy."""

    # Random deal
    seed = random.randint(0, 2**32 - 1)
    hands = deal_from_seed(seed)
    decl_id = random.randint(0, 9)  # Random declaration for MVP

    game = GameState(hands, decl_id)
    record = GameRecord(seed=seed, decl_id=decl_id)

    while not game.is_terminal:
        cp = game.current_player

        # Infer voids from play history
        voids = infer_voids(game.play_history, decl_id)

        # Sample consistent worlds
        worlds = sample_consistent_worlds(
            my_player=cp,
            my_hand=game.hands[cp],
            played=game.played_dominoes,
            voids=voids,
            decl_id=decl_id,
            n_samples=n_samples
        )

        # Query oracle for each world
        world_dicts = [
            make_world_dict(w, game.trick_state, decl_id)
            for w in worlds
        ]
        logits = oracle.query_batch(world_dicts, cp)  # (N, 7)

        # E[logits] = mean across worlds
        e_logits = logits.mean(dim=0)  # (7,)

        # Mask illegal moves
        legal_mask = game.legal_mask  # (7,) bool
        e_logits_masked = e_logits.clone()
        e_logits_masked[~legal_mask] = float('-inf')

        # Select move (max for Team 0, min for Team 1)
        if cp % 2 == 0:
            action = e_logits_masked.argmax().item()
        else:
            e_logits_masked[~legal_mask] = float('inf')
            action = e_logits_masked.argmin().item()

        # Record decision
        record.add_decision(
            player=cp,
            action=action,
            e_logits=e_logits.numpy(),
            legal_mask=legal_mask
        )

        # Apply move
        game = game.apply_action(action)

    record.final_score = game.score
    return record
```

### 5. Stage 2 Model

```python
# forge/eq/stage2.py

import torch
import torch.nn as nn

class Stage2Model(nn.Module):
    """
    Imperfect-info policy: predicts E[logits] from transcript.

    Input: Sequence of (decl, player, domino, led_suit) tokens
    Output: E[logits] for 7 actions
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 128
    ):
        super().__init__()

        # Embeddings
        self.decl_embed = nn.Embedding(10, d_model)
        self.player_embed = nn.Embedding(4, d_model)
        self.domino_embed = nn.Embedding(29, d_model)  # 28 + PAD
        self.suit_embed = nn.Embedding(9, d_model)     # 8 + PAD
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        # Output: E[logits] for 7 actions
        self.head = nn.Linear(d_model, 7)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, T, 4) - each token is (decl, player, domino, led_suit)

        Returns:
            e_logits: (B, 7)
        """
        B, T, _ = tokens.shape

        # Embed each feature and sum
        x = (
            self.decl_embed(tokens[:, :, 0]) +
            self.player_embed(tokens[:, :, 1]) +
            self.domino_embed(tokens[:, :, 2]) +
            self.suit_embed(tokens[:, :, 3]) +
            self.pos_embed(torch.arange(T, device=tokens.device))
        )

        # Causal mask
        mask = torch.triu(
            torch.ones(T, T, device=x.device) * float('-inf'),
            diagonal=1
        )

        x = self.transformer(x, mask=mask)

        # Use last token for prediction
        return self.head(x[:, -1, :])
```

### 6. Training Loop

```python
# forge/eq/train_stage2.py

def train_stage2(
    train_data: list[GameRecord],
    val_data: list[GameRecord],
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4
):
    model = Stage2Model(d_model=64, n_layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in make_batches(train_data, batch_size):
            tokens = batch['tokens']        # (B, T, 4)
            targets = batch['e_logits']     # (B, 7)
            legal = batch['legal_mask']     # (B, 7)

            pred = model(tokens)

            # MSE on legal moves only
            mask = legal.float()
            loss = ((pred - targets) ** 2 * mask).sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        val_loss, val_agreement = evaluate(model, val_data)

        print(f"Epoch {epoch}: train_loss={total_loss/n_batches:.4f}, "
              f"val_loss={val_loss:.4f}, val_agreement={val_agreement:.3f}")
```

---

## Validation Criteria

**MVP Success = All three conditions met:**

1. **Loss decreases** — Training loss goes down over epochs
2. **Val loss tracks** — Validation loss follows (no overfitting)
3. **Move agreement > random** — Model picks same move as E[logits]-oracle more than 1/7 of the time

**If these pass:** Scale up (more data, bigger model, N=1000 samples)

**If these fail:** Debug tokenization, check logit magnitudes, verify data pipeline

---

## File Structure

```
forge/eq/
├── __init__.py
├── voids.py               # infer_voids()
├── sampling.py            # sample_consistent_worlds()
├── oracle.py              # Stage1Oracle (uses forge/ml/tokenize.py)
├── game.py                # GameState tracker
├── generate.py            # generate_eq_game()
├── transcript_tokenize.py # Stage 2 tokenizer (NEW format, public info only)
├── stage2.py              # Stage2Model (different architecture from Stage 1)
├── train_stage2.py        # Training loop
└── evaluate.py            # Move agreement metrics
```

**Note on tokenizers:**
- **Stage 1** (oracle.py): Uses existing `forge/ml/tokenize.py` — sees all 4 hands
- **Stage 2** (transcript_tokenize.py): New format — sees only public transcript with relative player IDs

---

## Quick Start

```bash
# 1. Generate 100 games (MVP)
python -m forge.eq.generate --n-games 100 --n-samples 100 --output data/eq-mvp/

# 2. Train Stage 2
python -m forge.eq.train_stage2 --data data/eq-mvp/ --epochs 10

# 3. Evaluate
python -m forge.eq.evaluate --checkpoint runs/eq-stage2/best.ckpt
```

---

## What We're Deferring

| Item | Why Defer |
|------|-----------|
| ~~True Q-values~~ | ✅ **DONE** - Q-value models now available in catalog |
| N=1000 samples | N=100 is enough to validate convergence |
| 100K games | 100-1000 games validates the pipeline |
| Bidding model | Focus on play first |
| Optimal architecture | Validate 2-layer works before scaling |
| Production tokenization | Get it working, then optimize |

### Note: Q-Value Models for Interpretable E[Q]

Q-value models (`domino-qval-*.ckpt`) are now available and produce **directly interpretable** E[Q] marginals:

| Model Type | Output | Range | Interpretability |
|------------|--------|-------|------------------|
| Logit-based | Arbitrary logits | ~[-2, 0] | Low (needs softmax) |
| Q-value | Expected points | ~[-42, +42] | High (direct points) |

For debugging and analysis, Q-value models are preferred because:
- E[Q] values are in points (e.g., -17.89 = "expect 17.89 pts behind")
- Meaningful deltas between options
- Easy to verify values are in valid game range

See `forge/eq/README.md` for details.

---

## Risk Checklist

- [ ] Logit magnitudes are reasonable (±10, not ±1000)
- [ ] World sampling terminates (void constraints not too restrictive)
- [ ] Tokenization matches Stage 1 format for oracle queries
- [ ] Legal mask applied correctly (Team 0 max, Team 1 min)
- [ ] No data leakage (Stage 2 only sees transcript, not hands)
