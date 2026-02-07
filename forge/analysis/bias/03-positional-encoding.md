# Positional Encoding Investigation

## Summary

**Finding: The DominoTransformer uses NO positional encoding at all.**

This rules out the common causes of position-0 degeneracy (sin(0)=0, learned embedding undertrained).

## Architecture Analysis

### Model Definition (`forge/ml/module.py`)

The `DominoTransformer` class creates a standard `nn.TransformerEncoder`:

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dim,
    nhead=n_heads,
    dim_feedforward=ff_dim,
    dropout=dropout,
    batch_first=True,
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
```

**Critical observation**: PyTorch's `TransformerEncoder` does **not** include positional encoding by default. The user must explicitly add it, and this model does not.

### Token Construction (`forge/ml/tokenize.py`)

Tokens are structured as:
- Token 0: Context token (decl_id, normalized_leader)
- Tokens 1-7: Player 0's hand (in deal order)
- Tokens 8-14: Player 1's hand
- Tokens 15-21: Player 2's hand
- Tokens 22-28: Player 3's hand
- Tokens 29-31: Current trick plays (if any)

Each token has 12 features including:
- `player_id_embed` (normalized: 0=current, 1=next, 2=partner, 3=prev)
- `token_type_embed` (distinguishes context vs hand vs trick tokens)
- Domino features (pips, trump rank, etc.)

### Position Awareness

Without explicit positional encoding, the model has **limited** position awareness:
1. **Token type embedding** distinguishes context, hand, and trick tokens
2. **Player ID embedding** knows which player owns each domino
3. **Relative position within a player's hand** is NOT encoded

The transformer is essentially **permutation-equivariant** over tokens of the same type.

## Output Extraction (The Real Issue?)

The key code in the forward pass (lines 111-118):

```python
# Extract hand representations for current player
# Player's 7 dominoes start at index 1 + player_id * 7
start_indices = 1 + current_player * 7
offsets = torch.arange(7, device=device)
gather_indices = start_indices.unsqueeze(1) + offsets.unsqueeze(0)

hand_repr = torch.gather(x, dim=1, index=gather_indices)
logits = self.output_proj(hand_repr).squeeze(-1)
```

**This is positionally hardcoded**: Output slot 0 is always the **first token** in the current player's hand section, which corresponds to the **first domino dealt** to that player.

## Implications for Slot 0 Bias

Since there's no positional encoding:
1. **Sinusoidal PE degeneracy (sin(0)=0)**: Not applicable - no sinusoidal PE exists
2. **Learned PE undertrained**: Not applicable - no learned PE exists
3. **Attention positional bias**: Not applicable - no position information in attention

### The Actual Mechanism

The model sees all tokens as essentially "unordered" except for:
- Token type (context/hand/trick)
- Player ownership (normalized to current player's perspective)

The slot 0 bias must therefore come from:

1. **First-dealt domino statistics**: Is there a systematic pattern in which domino ends up as the first dealt? (Answer: No, deal is RNG-based)

2. **Output projection asymmetry**: All 7 output slots share the same `output_proj` linear layer, so no architectural asymmetry exists.

3. **Gather operation ordering**: The `torch.gather` extracts representations in deal order. If transformer attention patterns differ by absolute input position (e.g., position 1 vs position 7), this could create a bias.

4. **Training data correlation**: Is there something special about the first domino in each player's hand in the training data?

## Key Insight

**Without positional encoding, the transformer should be permutation-equivariant over hand tokens.** The only reason slot 0 would behave differently is if:

1. The **absolute position** (token index 1 vs 7) matters to attention despite no PE (unlikely but possible - attention scores are computed from content, but layer norm and residual connections create subtle dependencies)

2. The **deal order** correlates with something meaningful (e.g., first-dealt domino has specific statistical properties)

3. There's a **gradient flow asymmetry** during training (first output slot experiences different gradient patterns)

## Recommendation

The absence of positional encoding means the transformer has no explicit "slot 0" concept for the input. The bias likely originates from:

1. **Output extraction order**: The model always projects the first dealt domino's representation to output slot 0
2. **Implicit attention order effects**: Even without PE, attention may have subtle ordering biases from the sequential nature of the implementation

### Experiment to Confirm

To definitively test, shuffle the order of hand tokens at inference time (preserving their features) and see if:
- Bias follows the **content** (shuffled features move, bias follows)
- Bias follows the **position** (bias stays at output slot 0 regardless of content)

If bias follows position, the issue is in the output extraction or attention patterns, not input features.

## Conclusion

**Type of PE: None (no positional encoding used)**

The slot 0 bias is NOT caused by positional encoding degeneracy. The model has no position encoding to create such a problem. The bias must originate from:
- The fixed output extraction mechanism (gather by position)
- Implicit ordering effects in attention computation
- Training data/gradient flow asymmetries

This is a significant finding - the architecture is intentionally position-agnostic for the input, but the output is position-dependent by design.
