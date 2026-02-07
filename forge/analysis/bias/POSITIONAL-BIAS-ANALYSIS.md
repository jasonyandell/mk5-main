POSITIONAL BIAS RESEARCH

# Issue
The test shows the model has a systematic bias against slot 0 (first position in hand), regardless of which domino     
  occupies it:                                                                                                           
  ┌────────────────────┬────────┬────────────────┐                                                                       
  │       Metric       │ Slot 0 │   Slots 1-6    │                                                                       
  ├────────────────────┼────────┼────────────────┤                                                                       
  │ r(model, oracle)   │ 0.81   │ 0.99+          │                                                                       
  ├────────────────────┼────────┼────────────────┤                                                                       
  │ Variance explained │ 65%    │ 98.7-99.0%     │                                                                       
  ├────────────────────┼────────┼────────────────┤                                                                       
  │ Mean bias          │ -0.26  │ -0.04 to +0.02 │                                                                       
  └────────────────────┴────────┴────────────────┘                                                                       
  This is not about the 0-0 domino being undervalued - it's about the first output position being systematically         
  degraded. The model predicts slot 0 with 19× worse accuracy than other slots.                                          
                                                                                                                         
  Possible causes:                                                                                                       
  - Positional encoding artifact in the transformer                                                                      
  - Attention pattern favoring later positions                                                                           
  - Training data tokenization putting context token at position 0                                                       
  - Causal masking or padding effects                                                                                    
                                                                                                                         
  This is architectural, not a learned strategy about specific dominoes.

## Investigation Results

### 01: Attention Mask (01-attention-mask.md)
**Causal masking is NOT the cause.** The model uses `nn.TransformerEncoder` with only `src_key_padding_mask` (padding mask) - no causal mask, no `is_causal=True`, no triangular mask. Position 0 has full bidirectional attention to all tokens. Investigate tokenization order or value head gradient competition next.

### 02: Tokenization Layout (02-tokenization-layout.md)
**No off-by-one bug in tokenization.** Token position 0 contains a context token (decl_id, leader), not a hand token. Hand tokens for player P occupy positions `1 + P*7` through `7 + P*7`. Model correctly extracts positions 1-7 for Q-value outputs 0-6. The indexing is aligned: output slot 0 = token position 1 = first domino in current player's hand. Root cause must be elsewhere (attention patterns, embedding edge effects, or training distribution).

### 03: Positional Encoding (03-positional-encoding.md)
**No positional encoding exists.** The DominoTransformer uses `nn.TransformerEncoder` without any PE (no sinusoidal, no learned, no RoPE). The model is permutation-equivariant over input tokens - the slot 0 bias must originate from the fixed output extraction mechanism (`torch.gather` by position) or implicit attention ordering effects, not PE degeneracy.

### 05: Padding Convention (05-padding-convention.md)
**Padding is NOT the cause.** Both Stage 1 and Stage 2 tokenization use right-padding (zeros appended to end). Position 0 always contains meaningful content (context/declaration token). Output slots 0-6 map to input positions 1-7+ (never padded). PAD tokens never appear at position 0.

### 04: Output Head Bias (04-output-head-bias.md)
**Output head has NO position-specific bias.** The Q-value projection uses `nn.Linear(256, 1)` applied identically to all 7 hand positions via `torch.gather`. The single scalar bias (-0.045) is uniform across positions. Slot 0 degradation must originate upstream in the transformer encoder or from tokenization ordering effects.

### 09: RoPE Position 0 (09-rope-position-0.md)
**RoPE is NOT used.** The model uses no positional encoding of any kind - not RoPE, not sinusoidal, not learned embeddings. The hypothesis that RoPE's degenerate rotation at position 0 (where theta=0 gives identity-like rotation with cos(0)=1, sin(0)=0) could cause the bias is ruled out. See 03-positional-encoding.md for full PE analysis.

### 07: Learned PE Weights (07-learned-pe-weights.md)
**NO learned positional embeddings exist.** Verified checkpoint contains no `pos_embed`, `positional_encoding`, or similar weights. The model uses `nn.TransformerEncoder` without any PE mechanism - within-hand position is completely indistinguishable. The model relies on token_type_embed (category) and player_id_embed (ownership) but has NO within-hand position signal. PE degeneracy CANNOT cause slot 0 bias.

### 08: BOS Token Ghost (08-bos-token-ghost.md)
**BOS token ghost is RULED OUT.** The model is trained from scratch with randomly initialized weights - no pretrained weights from HuggingFace or elsewhere. PyTorch's `nn.TransformerEncoder` has no built-in position 0 special handling. The architecture has no CLS/BOS token convention; the context token at position 0 is used only for value prediction, not Q-values (which extract from positions 1-7+).

### 12: Domino Ordering (12-domino-ordering.md)
**CONFIRMED: Sorting creates training distribution bias.** The `deal_from_seed()` function sorts hands by domino ID, causing slot 0 to always contain the minimum-ID domino (0-0, 1-0, 1-1, etc.). Slot 0 has 1.74-bit KL divergence from uniform, with blank-containing dominoes appearing 2.46x more often than expected. High-value dominoes (6-4, 6-5, 6-6) NEVER appear in slot 0. This creates systematic training asymmetry where slot 0 sees "easy" low-value dominoes while slots 5-6 see strategically complex dominoes.

### 10: Deep Attention Routing (10-attention-routing-deep.md)
**Attention routing does NOT explain the bias.** Analysis of attention weights from all 6 layers (8 heads each) across 1000 validation samples shows: (1) Hand slot 0 receives similar attention to slots 1-6 in the final layer (ratio=0.97), (2) No systematic isolation of position 0 or slot 0 in any layer, (3) All layers show balanced attention given/received ratios. The bias must originate from the training distribution asymmetry (see 12) or other factors, not attention routing.

### 13: Action Frequency Distribution (13-action-frequency.md)
**Training data frequency is NOT the cause.** Slot 0 is the oracle's MOST recommended action (24.4% vs 12.6% avg), not least. Legal action frequency is nearly uniform (~30% all slots). However, slot 0 has the highest tie rate: 69% of slot 0 recommendations are ties with other actions (vs 0-57% for slots 1-6). The model sees many examples where slot 0 is "best" but equally-good alternatives exist, which may confuse learning.

### 15: PyTorch TransformerEncoder Edge Effects (15-transformer-edge-effects.md)
**PyTorch TransformerEncoder has NO inherent edge effects.** Reviewed 2024-2025 research on attention sinks and position bias in transformers. The documented phenomena (attention sinks at position 0, LayerNorm recency bias) arise from causal masking and positional encodings - neither of which this model uses. With bidirectional attention and no PE, the architecture is permutation-equivariant. PyTorch's implementation has no position-specific code paths. The slot 0 bias must originate from training data distribution (fixed domino ordering), not architectural effects.

### 20: Proposed Fix - Shuffle Hand Ordering (20-proposed-fix-shuffle.md)
**PROPOSED FIX: Shuffle within-hand ordering during tokenization.** The simplest fix is to randomly permute each player's 7-domino hand using the existing per-shard deterministic RNG in `forge/ml/tokenize.py`. This eliminates the 1.74-bit KL divergence in slot 0's training distribution. Requires re-tokenization (~30 min) and retraining (~2 hours). Oracle parquet shards remain unchanged. Expected to raise slot 0 correlation from 0.81 to ~0.99.

### 17: Attention Head Specialization (17-attention-head-specialization.md)
**PARTIAL: Some heads suppress slot 0, but net effect is balanced.** Per-head analysis of 48 attention heads (6 layers x 8 heads) found 11 "suppressor" heads that significantly reduce attention to slot 0 (ratio < 0.8, p < 0.01), concentrated in layer_0 (6/8 heads). However, 17 "amplifier" heads increase attention to slot 0 in middle layers. Net effect is neutral. Attention patterns alone do not explain the bias; root cause remains training distribution (sorted domino ordering).

### 19: LayerNorm Statistics (19-layernorm-statistics.md)
**LayerNorm is NOT the cause.** Pre-normalization activation statistics (mean, std) are nearly identical for slot 0 vs slots 1-6 at the final layer (std ratio = 1.002). Early layers show minor differences (layer 0 has 17% higher std for slot 0), but this converges to parity by the final layers. LayerNorm applies equivalent normalization to all slots; the slot 0 bias cannot be attributed to differential LayerNorm scaling.

### 16: Gradient Flow Analysis (16-gradient-flow.md)
**Gradient interference from value head does NOT cause slot 0 bias.** Measured gradient magnitudes at each position during backprop. Value loss gradient is exactly zero at positions 1-7 (no spillover from position 0's context token). Q-loss shows position 1 (slot 0) receives 1.29x the gradient of positions 2-7, but this exists independently of the value head. The dual-head architecture does not create gradient competition - root cause remains training distribution bias from sorted domino ordering.

### 18: Layer-wise Degradation (18-layer-wise-degradation.md)
**Gap emerges in final layer, not gradual accumulation.** Extracted embeddings from all 6 transformer layers and computed per-slot correlations with oracle Q-values (499K validation samples). Key findings: (1) All slots start with near-zero correlation at input layer, (2) Both slot 0 and slots 1-6 improve steadily through layers 0-4 with only minor gap fluctuations (-0.01 to -0.07), (3) In final layer, slots 1-6 jump to r=0.995 while slot 0 only reaches r=0.959. The 3.5% correlation gap emerges because slot 0 fails to make the final convergence step. Root cause is training data distribution (sorted domino ordering) causing the model to learn slot 0 less precisely, not architectural bias.

### 14: Embedding Analysis (14-embedding-analysis.md)
**CONFIRMED: Embeddings are NOT the cause - data distribution is.** Analyzed 499K validation samples for embedding differences between slot 0 and slots 1-6. Embedding statistics (L2 norm, variance, cosine similarity) show negligible effect sizes (Cohen's d < 0.11) despite statistical significance. The actual root cause: dominoes are sorted by pip value within each hand. Slot 0 has mean high_pip=1.33 vs 4.45 for slots 1-6, and 46% doubles vs 21%. Slot 0 dominoes (0-0, 1-1, 2-2) have higher strategic variance and are inherently harder to predict. The correlation gap reflects intrinsic domino difficulty, not architectural bias.

### 11: Context Token Adjacency (11-context-adjacency.md)
**PARTIAL: Context adjacency contributes to slot 0 degradation.** Position 1 (slot 0 source for player 0) is directly adjacent to the context token at position 0. Layer 0 shows 3.51x attention bias from context toward position 1; layers 1-2 show 1.5x elevated attention from position 1 toward context. Player 0 (adjacent to context) has r=0.56 for slot 0, while player 3 (farthest from context) has r=0.72. The value head uses context token for state value prediction, creating gradient competition. However, the bias persists across all players, suggesting context adjacency is a contributing factor but not the sole cause - data distribution (sorted domino ordering) remains the primary root cause.

### 06: Attention Patterns (06-attention-patterns.md)
**CONFIRMED: Attention sink is NOT the cause.** Extracted attention weights from all 6 transformer layers. Final layer shows perfectly balanced attention (slot 0 receives 0.98x vs slots 1-6). Early layers show mild bias (1.86x) that fully resolves. The actual root cause: **hand sorting creates declaration-specific confounding**. For decl_ids 2, 3, 4 (twos/threes/fours), slot 0 correlation drops to 0.16-0.74 while ALL other declarations maintain r=0.99+. Data shows slot 0 contains the SAME domino (e.g., 0-0) for 96-100% of samples within each declaration type due to `sorted()` in `deal_from_seed()`. The model cannot learn position-independent representations when position is deterministically confounded with domino identity. **Fix: Shuffle hand ordering during tokenization.**

#