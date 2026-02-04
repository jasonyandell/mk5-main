# Zeb ML Overview

A guide for engineers with math backgrounds but no ML training.

---

## 1. What We're Building

**Goal**: Train a neural network to play Texas 42 dominoes competitively.

**The challenge**: Unlike chess or Go, 42 has imperfect information (you can't see opponents' hands), making optimal play impossible to compute. Instead, we want a network that learns good heuristics from experience.

**Current approach**: REINFORCE with self-play. The network plays games against itself (or a heuristic opponent), and we adjust its weights to favor actions that led to wins.

**Where we are**: The system is implemented and runnable. Early training shows the network learning *something* but not yet beating a simple heuristic opponent consistently.

---

## 2. The Core ML Concepts

### Policy Networks: "What Should I Do?"

A policy network takes a game state as input and outputs a probability distribution over possible actions.

```
State: [current hand, played cards, trump suit, ...]
                    |
                    v
            [Neural Network]
                    |
                    v
Action probabilities: [0.05, 0.15, 0.60, 0.10, 0.05, 0.05, 0.00]
                        ^           ^
                   play slot 0   play slot 2 (highest prob)
```

The network outputs a "logit" (raw score) for each of your 7 hand slots. We convert these to probabilities using softmax:

```python
probs = softmax(logits)  # e.g., [2.1, 3.5, 1.2, ...] -> [0.12, 0.45, 0.05, ...]
```

Higher probability = network thinks this is a better move.

### Value Networks: "How Am I Doing?"

A value network takes a game state and outputs a single number predicting the expected outcome.

```
State: [current hand, played cards, trump suit, ...]
                    |
                    v
            [Neural Network]
                    |
                    v
             Value: 0.35
             (Team 0 slightly ahead)
```

Output range is [-1, 1]:
- +1 = Team 0 definitely wins
- -1 = Team 1 definitely wins
- 0 = Even game

### Why We Need Both (Actor-Critic)

**Policy alone** tells us what to do but not whether we're doing well. We can't tell if a loss was due to bad decisions or just bad luck.

**Value alone** tells us about states but not which action to take. We'd need to evaluate every possible action (expensive).

**Together** they form an actor-critic system:
- **Actor** (policy): Chooses actions
- **Critic** (value): Evaluates how good states are

The critic helps the actor learn by providing a "baseline" - we'll explain this in Section 3.

In our code, both networks share the same transformer body but have different "heads":

```python
# model.py: One backbone, two heads
class ZebModel(nn.Module):
    def __init__(self):
        self.encoder = TransformerEncoder(...)  # Shared
        self.policy_proj = nn.Linear(...)       # Policy head
        self.value_head = nn.Sequential(...)    # Value head
```

---

## 3. REINFORCE Explained

### The Basic Idea

REINFORCE is gradient descent for decision-making:

1. Play a game using current policy
2. See what happened (win/loss)
3. Adjust weights to make winning actions more likely

**Intuition**: Imagine you're learning to shoot free throws. You don't know the physics, but you can adjust based on outcomes:
- Made the shot? Repeat that motion.
- Missed? Try something different.

That's REINFORCE. Actions that preceded good outcomes get "reinforced."

### The Math (Simplified)

The policy loss in our code is:

```python
# module.py line 69-73
advantage = outcomes - value.detach()
log_probs = log_softmax(policy)
action_log_probs = log_probs[chosen_action]
policy_loss = -(advantage * action_log_probs).mean()
```

Breaking this down:

1. **`log_probs`**: Log of the probability we assigned to each action before taking it. If we were 60% confident in action A, log_prob = log(0.6) = -0.51.

2. **`advantage`**: How much better/worse the outcome was than expected (explained below).

3. **The product**: `advantage * log_prob` is the "policy gradient."
   - Positive advantage + action we took = increase this action's probability
   - Negative advantage + action we took = decrease this action's probability

4. **Negative sign**: We minimize loss, so we negate to maximize expected reward.

### The Variance Problem

REINFORCE has a fundamental problem: **high variance**.

Consider: You play slot 3 and eventually win. Was slot 3 a good move, or did you just get lucky later? REINFORCE can't distinguish. Every action in a winning game gets reinforced, even bad ones.

**What this means practically**:
- Learning is noisy and slow
- You need many games to average out the luck
- Progress can seem random or plateaued

### The Value Baseline Trick

The "advantage" calculation is how we reduce variance:

```python
advantage = outcomes - value.detach()  # Outcome minus predicted value
```

Instead of asking "did we win?", we ask "did we do better than expected?"

**Example**:
- You're way ahead (value predicts +0.8), you win (+1.0)
- Advantage = 1.0 - 0.8 = +0.2 (slightly better than expected)
- Actions get slightly reinforced

Compare to:
- You're way behind (value predicts -0.8), you win (+1.0)
- Advantage = 1.0 - (-0.8) = +1.8 (much better than expected!)
- Actions get strongly reinforced

This makes sense: coming back from behind was impressive and should be reinforced more.

**Why `.detach()`?** We don't want to backpropagate through the value network when computing policy gradients. They train independently.

---

## 4. Self-Play: The AlphaZero Approach

### Why Play Against Yourself?

When you play against yourself:
- Every game generates training data for both sides
- No need to label "good" moves manually
- Difficulty automatically scales (opponent improves as you improve)

AlphaGo/AlphaZero famously used this to surpass human play without human training data.

### The Bootstrap Problem

**Problem**: An untrained network plays randomly. Random vs random produces random outcomes. How do you learn anything from noise?

**Hope**: Even random play generates *some* signal. A completely terrible move (like playing a domino that can't win anything) loses more often than a merely bad move. Gradient updates, averaged over many games, should pick up on this.

**Reality**: The signal is weak. Early training is mostly noise, and progress is slow until the network gets "good enough" to generate meaningful games.

### The Moving Target Problem

**Problem**: In supervised learning, the target (correct answer) is fixed. In self-play, the opponent keeps changing because it's you.

Imagine learning to box by fighting your reflection. Every time you improve, so does your opponent. Are you getting better or just staying even?

**What this causes**:
- Win rate against yourself is always ~50%
- You need external benchmarks (random player, heuristic) to measure progress
- Training can be unstable (you might get worse at some things while getting better at others)

Our code handles this with periodic evaluation:

```python
# run_laptop.py: Evaluate against fixed baselines
vs_random = evaluate_vs_random(model, n_games=100)
vs_heuristic = evaluate_vs_heuristic(model, n_games=100)
```

### Alternative: Training Against a Fixed Opponent

We also support training against a fixed heuristic (`--vs-heuristic` flag):

```python
# self_play.py: play_games_vs_heuristic()
# Model plays seats 0,2; heuristic plays seats 1,3
```

**Advantages**:
- Stable training target
- Clear progress signal (beating the heuristic)
- Lower variance

**Disadvantages**:
- Network learns to exploit heuristic weaknesses
- May not generalize to other play styles
- Ceiling limited by heuristic quality

---

## 5. Key Hyperparameters

### entropy_weight (default: 0.1)

**What it does**: Rewards the network for staying "uncertain" about which action to take.

```python
# module.py line 78-84, 90
entropy = -(probs * log(probs)).sum()  # Higher when probs are spread out
loss = policy_loss - entropy_weight * entropy  # Subtract: higher entropy = lower loss
```

**Too low** (e.g., 0.001):
- Network becomes overconfident too quickly
- Always picks the same action in similar states
- "Entropy collapse": stops exploring, gets stuck in local optima

**Too high** (e.g., 1.0):
- Network stays random forever
- Never commits to a strategy
- Learning is slow or nonexistent

**Healthy range**: 0.01 to 0.2. We use 0.1 to encourage exploration during early training.

### learning_rate (default: 3e-4)

**What it does**: How much to adjust weights after each batch.

**Too low** (e.g., 1e-6):
- Learning takes forever
- May never converge

**Too high** (e.g., 1e-2):
- Updates overshoot, undoing previous progress
- Training loss oscillates wildly
- May diverge entirely

**Safe range**: 1e-4 to 3e-4 for Adam/AdamW optimizers.

### games_per_epoch (default: 500)

**What it does**: How many games to play before each gradient update.

**Too few** (e.g., 50):
- High variance: each batch is mostly noise
- Unstable training

**Too many** (e.g., 10000):
- Slow iteration: takes forever per epoch
- Using stale policy to generate data

**Trade-off**: More games = cleaner signal, but slower feedback loop.

### temperature (default: 1.0)

**What it does**: Controls exploration during action selection.

```python
# model.py line 147
probs = softmax(policy / temperature)
```

**Temperature = 1.0**: Actions proportional to learned probabilities.
**Temperature > 1.0**: Flatter distribution, more random.
**Temperature < 1.0**: Sharper distribution, more greedy.

**During training**: Higher temperature (1.0+) encourages exploration.
**During evaluation**: Lower temperature (0.1-0.3) for best play.

### value_weight (default: 0.5)

**What it does**: How much to weight value loss vs policy loss.

```python
loss = policy_loss + value_weight * value_loss - entropy_weight * entropy
```

**Too low**: Value head undertrained, poor baselines for REINFORCE.
**Too high**: Value head dominates training, policy updates are too small.

**Typical**: 0.25 to 1.0.

---

## 6. Common Pitfalls and Solutions

### Entropy Collapse

**Symptoms**:
- Entropy drops to near zero early in training
- Network always picks the same action
- Win rate plateaus or drops

**Causes**:
- Entropy weight too low
- Learning rate too high
- Not enough exploration in early training

**Solutions**:
- Increase `entropy_weight` (we use 0.1)
- Use temperature > 1.0 during data generation
- Check initial entropy (should be ~1.5 for 7-action space)

We monitor this explicitly:

```python
# module.py line 102-106
if not self._logged_initial_entropy:
    print(f"Initial entropy: {entropy:.3f} (healthy range: 0.5-1.5)")
```

### Reward Sparsity

**Problem**: In 42, you don't know if you won until the game ends (28 actions later). Credit assignment is hard.

**What happens**: The network struggles to connect early decisions to final outcomes.

**Solutions we use**:
- Value baseline (reduces noise)
- Fill outcomes for all steps post-game:

```python
# self_play.py line 124-125
for step in game.steps:
    step.outcome = get_outcome(game.state, step.seat)
```

**Future options**:
- Intermediate rewards (points per trick)
- Temporal difference learning (TD(lambda))

### High Variance Gradients

**Problem**: REINFORCE gradients are noisy. A single lucky game can push weights in the wrong direction.

**Symptoms**:
- Loss oscillates wildly
- Metrics don't show clear trends
- "Training loss looks random"

**Solutions**:
- More games per epoch
- Gradient clipping (we use 1.0):

```python
# run_laptop.py line 201
gradient_clip_val=1.0
```

- Larger batch sizes

### Catastrophic Forgetting in Self-Play

**Problem**: Network "forgets" how to play against old versions of itself.

**Example**: Version 5 learns to beat Version 4 but loses to Version 2's strategy.

**Symptoms**:
- Performance oscillates against fixed baselines
- Wins against current self but loses to older checkpoints

**Solutions** (not yet implemented):
- Keep a pool of old checkpoints as training partners
- Regularization to prevent drastic weight changes
- Lower learning rate

---

## 7. What "More Training" Actually Means

### Sample Efficiency Expectations

Reinforcement learning is data-hungry. Rough benchmarks from similar systems:

| System | Game Complexity | Training Games | Training Time |
|--------|----------------|----------------|---------------|
| Simple Atari | Low | ~10 million frames | Hours |
| Connect Four | Medium | ~500K games | Hours |
| Go (AlphaZero) | Very High | ~5 million games | Days (TPU clusters) |
| 42 (us) | Medium | ??? | ??? |

**Our current settings**: 500 games/epoch x 20 epochs = 10,000 games. This is likely **not enough**.

**Expected needs**: 100K - 1M games for meaningful learning, based on game complexity.

### When to Expect Signal

**Epochs 1-5**: Mostly noise. Don't expect consistent improvement.

**Epochs 5-10**: Early signal. Win rate vs random should creep above 50%.

**Epochs 10-50**: Measurable learning. Should reliably beat random.

**Epochs 50+**: Where real progress happens. This is where AlphaZero-style training shines.

**Red flags** (something is broken):
- Entropy drops to 0 before epoch 5
- Win rate vs random goes *below* 50%
- Training loss increases steadily

### What a Healthy Training Curve Looks Like

**Policy loss**: Starts high, trends downward with noise. Plateaus are normal.

**Value loss**: Should decrease as value head learns to predict outcomes.

**Entropy**: Gradual decline from ~1.5 to ~0.5 over training. Sudden drops are bad.

**Win rate vs baselines**: Stepwise improvement with plateaus.

```
Epoch 1:   vs Random 48%, vs Heuristic 42%  (basically random)
Epoch 10:  vs Random 53%, vs Heuristic 45%  (slight signal)
Epoch 50:  vs Random 65%, vs Heuristic 52%  (learning)
Epoch 200: vs Random 80%, vs Heuristic 60%  (goal territory)
```

---

## 8. Alternative Approaches We Could Try

### PPO (Proximal Policy Optimization)

**What it is**: A more stable variant of policy gradient methods.

**Key idea**: Limit how much the policy can change in one update.

```
loss = min(ratio * advantage, clip(ratio, 1-epsilon, 1+epsilon) * advantage)
```

**Advantages over REINFORCE**:
- Lower variance
- More sample efficient
- Less sensitive to hyperparameters

**Why we're not using it (yet)**: More complex to implement correctly. REINFORCE is a good starting point to verify the system works.

### Imitation Learning from Oracle

**What it is**: Train the network to copy the oracle's decisions.

**How it would work**:
1. Run oracle solver on game states
2. Oracle outputs "best" action
3. Train network with supervised cross-entropy loss

**Advantages**:
- Fast, stable learning
- Clear signal (oracle is always "right")
- No reward sparsity issue

**Disadvantages**:
- Ceiling limited by oracle quality
- Oracle is slow to generate data
- Network may not generalize beyond oracle's training distribution

**Could combine with RL**: Pre-train with imitation, fine-tune with self-play.

### Curriculum Learning

**What it is**: Start with easier tasks, gradually increase difficulty.

**For 42, this could mean**:
1. Train on hands with obvious plays first
2. Gradually introduce more complex situations
3. Add opponent strength incrementally

**Why it might help**: The full game is too hard initially. Easier problems provide cleaner gradients.

### Reward Shaping

**What it is**: Give intermediate rewards, not just win/loss.

**Possible rewards for 42**:
- Points scored per trick (+1 per 5 points)
- Winning a trick (+0.1)
- Making bid (+0.5), setting bid (+0.5)

**Trade-off**: Easier credit assignment, but risks teaching suboptimal play (e.g., winning pointless tricks).

---

## 9. Current Status and Next Steps

### What We've Built

- **Complete training pipeline**: Model, data generation, training loop, evaluation
- **Baseline comparisons**: Random player (~50%), heuristic player (~57% vs random)
- **Monitoring**: W&B integration, entropy tracking, periodic evaluation
- **Two training modes**: Self-play and vs-heuristic

### Initial Observations

From BASELINES.md and early runs:

| Player | vs Random | vs Heuristic |
|--------|-----------|--------------|
| Random | 50% | ~43% |
| Heuristic | 57% | 50% |
| Untrained Neural | ~50% | ~43% |

The untrained network is essentially random, as expected.

### What Worked

- **Observation encoding**: Reuses Stage 2 tokenization, clean and efficient
- **Batched self-play**: ~12 games/sec with neural inference
- **Entropy monitoring**: Caught potential collapse early

### Open Questions

1. **Sample efficiency**: How many games do we really need? Current 10K games per run may be 10-100x too few.

2. **Heuristic vs self-play**: Which training target works better for 42 specifically?

3. **Model size**: Is 75K params (small) enough, or do we need 300K (medium)?

4. **Bidding phase**: Currently skipped (`skip_bidding=True`). Adding bidding adds complexity but is essential for real 42.

5. **Temperature schedule**: Should we anneal temperature during training?

### Suggested Next Steps

**Short-term**:
1. Run longer training (100+ epochs, 50K+ games)
2. Monitor full learning curves in W&B
3. Compare self-play vs heuristic training

**Medium-term**:
4. Implement PPO for lower variance training
5. Try imitation learning from oracle as a baseline
6. Add bidding phase to training

**Long-term**:
7. Curriculum learning from simple to complex situations
8. Multi-checkpoint self-play pool
9. Hyperparameter search at scale

---

## Glossary

| Term | Definition |
|------|------------|
| **Policy** | Probability distribution over actions |
| **Value** | Predicted expected outcome from a state |
| **Advantage** | Outcome minus value baseline |
| **Entropy** | Measure of policy uncertainty |
| **Logit** | Raw network output before softmax |
| **Softmax** | Converts logits to probabilities (exp(x)/sum(exp(x))) |
| **Epoch** | One pass through training data (for us: generate games, train once) |
| **Gradient clipping** | Cap gradient magnitude to prevent huge updates |
| **Baseline** | A reference player for evaluation (random, heuristic) |

---

*Last updated: February 2026*
