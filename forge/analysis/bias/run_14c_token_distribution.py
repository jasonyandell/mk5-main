"""
Investigation 14c: Token input distribution analysis by slot.

Check if slot 0 has different input token distributions than slots 1-6.
This could explain why the model learns different representations for slot 0.

Analyzes:
1. Distribution of each token feature (high_pip, low_pip, etc.) per slot
2. Distribution of legal mask per slot
3. Distribution of oracle Q-values per slot
4. Is slot 0 more likely to be the "best" action?
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
from scipy import stats

PROJECT_ROOT = Path("/home/jason/v2/mk5-tailwind")
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("=" * 70)
    print("Investigation 14c: Token Distribution Analysis by Slot")
    print("=" * 70)
    print()

    # Load validation data
    data_path = PROJECT_ROOT / "data/tokenized-full/val"
    print(f"Loading validation data from: {data_path}")

    tokens = np.load(data_path / "tokens.npy")
    masks = np.load(data_path / "masks.npy")
    players = np.load(data_path / "players.npy")
    qvals = np.load(data_path / "qvals.npy")
    legal = np.load(data_path / "legal.npy")
    teams = np.load(data_path / "teams.npy")
    targets = np.load(data_path / "targets.npy")

    n_samples = len(tokens)
    print(f"Total samples: {n_samples:,}")

    # Token features indices:
    # 0: high_pip (0-6)
    # 1: low_pip (0-6)
    # 2: is_double (0-1)
    # 3: count_value (0-2: 0pts, 5pts, 10pts)
    # 4: trump_rank (0-7)
    # 5: player_id (0-3)
    # 6: is_current (0-1)
    # 7: is_partner (0-1)
    # 8: is_remaining (0-1)
    # 9: token_type (1-4 for hands)
    # 10: decl_id (0-9)
    # 11: leader (0-3)

    feature_names = [
        "high_pip", "low_pip", "is_double", "count_value", "trump_rank",
        "player_id", "is_current", "is_partner", "is_remaining",
        "token_type", "decl_id", "leader"
    ]

    print("\n" + "=" * 70)
    print("PART 1: Extract hand tokens per slot")
    print("=" * 70)

    # For each sample, extract the current player's hand tokens
    # Hand position for current player: start = 1 + player * 7
    # Slot i = position start + i

    # Extract hand tokens for each slot
    hand_tokens = []  # Will be (n_samples, 7, 12)

    for slot in range(7):
        # For each sample, calculate position for this slot
        positions = 1 + players * 7 + slot
        # Extract the token at that position
        # tokens shape: (n_samples, 32, 12)
        # Need to gather along dim 1
        slot_tokens = tokens[np.arange(n_samples), positions.astype(int)]
        hand_tokens.append(slot_tokens)

    hand_tokens = np.stack(hand_tokens, axis=1)  # (n_samples, 7, 12)
    print(f"Extracted hand tokens shape: {hand_tokens.shape}")

    print("\n" + "=" * 70)
    print("PART 2: Feature distributions per slot")
    print("=" * 70)

    # Check if feature distributions differ by slot
    for feat_idx, feat_name in enumerate(feature_names):
        # Get values for each slot
        slot_values = [hand_tokens[:, slot, feat_idx] for slot in range(7)]

        # Chi-squared test: are the distributions different?
        # For continuous-like features, use Kruskal-Wallis instead
        try:
            stat, p = stats.kruskal(*slot_values)
            test_name = "Kruskal-Wallis"
        except ValueError:
            # All values identical
            stat, p = 0, 1.0
            test_name = "N/A (constant)"

        # Calculate means per slot
        slot_means = [np.mean(sv) for sv in slot_values]
        slot0_mean = slot_means[0]
        slots16_mean = np.mean(slot_means[1:])

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        print(f"\n{feat_name}:")
        print(f"  Slot 0 mean: {slot0_mean:.4f}")
        print(f"  Slots 1-6 mean: {slots16_mean:.4f}")
        print(f"  Difference: {slot0_mean - slots16_mean:+.4f}")
        print(f"  {test_name}: stat={stat:.2f}, p={p:.2e} {sig}")

    print("\n" + "=" * 70)
    print("PART 3: Legal mask distribution per slot")
    print("=" * 70)

    # Check if some slots are more likely to be legal
    legal_rates = []
    for slot in range(7):
        legal_rate = legal[:, slot].mean()
        legal_rates.append(legal_rate)
        marker = " <-- SLOT 0" if slot == 0 else ""
        print(f"  Slot {slot}: {legal_rate:.4f} ({legal_rate*100:.2f}%){marker}")

    # Chi-squared for legal rates
    stat, p = stats.kruskal(*[legal[:, slot] for slot in range(7)])
    print(f"\n  Kruskal-Wallis: stat={stat:.2f}, p={p:.2e}")

    print("\n" + "=" * 70)
    print("PART 4: Target action distribution (which slot is optimal?)")
    print("=" * 70)

    target_counts = Counter(targets)
    total = sum(target_counts.values())

    print("\nHow often is each slot the optimal action:")
    for slot in range(7):
        count = target_counts.get(slot, 0)
        pct = 100 * count / total
        marker = " <-- SLOT 0" if slot == 0 else ""
        print(f"  Slot {slot}: {count:,} ({pct:.2f}%){marker}")

    # Is slot 0 underrepresented as the optimal action?
    expected_uniform = total / 7
    slot0_observed = target_counts.get(0, 0)
    print(f"\n  Expected if uniform: {expected_uniform:.0f}")
    print(f"  Slot 0 observed: {slot0_observed}")
    print(f"  Ratio: {slot0_observed / expected_uniform:.4f}")

    # Chi-squared test for uniform distribution
    observed = [target_counts.get(s, 0) for s in range(7)]
    stat, p = stats.chisquare(observed)
    print(f"\n  Chi-squared test vs uniform: stat={stat:.2f}, p={p:.2e}")

    print("\n" + "=" * 70)
    print("PART 5: Oracle Q-value distribution per slot")
    print("=" * 70)

    # Convert Q-values to current player perspective
    team_sign = np.where(teams == 0, 1, -1).reshape(-1, 1)
    oracle_q = qvals * team_sign

    # Replace illegal with NaN for stats
    oracle_q_legal = np.where(legal > 0, oracle_q, np.nan)

    print("\nMean oracle Q-value per slot (legal actions only):")
    for slot in range(7):
        slot_q = oracle_q_legal[:, slot]
        valid = ~np.isnan(slot_q)
        mean_q = np.nanmean(slot_q)
        std_q = np.nanstd(slot_q)
        n_valid = valid.sum()
        marker = " <-- SLOT 0" if slot == 0 else ""
        print(f"  Slot {slot}: mean={mean_q:.2f}, std={std_q:.2f} (n={n_valid:,}){marker}")

    # Kruskal-Wallis on Q-values
    slot_qs = []
    for slot in range(7):
        slot_q = oracle_q_legal[:, slot]
        valid = ~np.isnan(slot_q)
        slot_qs.append(slot_q[valid])

    stat, p = stats.kruskal(*slot_qs)
    print(f"\n  Kruskal-Wallis: stat={stat:.2f}, p={p:.2e}")

    print("\n" + "=" * 70)
    print("PART 6: Domino sorting analysis")
    print("=" * 70)

    # Within each hand, how are dominoes sorted?
    # Check if slot 0 tends to have different domino properties

    # Trump rank distribution by slot
    print("\nMean trump_rank per slot (0=highest trump, 7=not trump):")
    for slot in range(7):
        trump_rank = hand_tokens[:, slot, 4]  # trump_rank feature
        mean_rank = np.mean(trump_rank)
        marker = " <-- SLOT 0" if slot == 0 else ""
        print(f"  Slot {slot}: {mean_rank:.4f}{marker}")

    # is_double distribution by slot
    print("\nProportion of doubles per slot:")
    for slot in range(7):
        is_double = hand_tokens[:, slot, 2]  # is_double feature
        pct_double = 100 * np.mean(is_double)
        marker = " <-- SLOT 0" if slot == 0 else ""
        print(f"  Slot {slot}: {pct_double:.2f}%{marker}")

    # High pip distribution
    print("\nMean high_pip per slot:")
    for slot in range(7):
        high_pip = hand_tokens[:, slot, 0]
        mean_pip = np.mean(high_pip)
        marker = " <-- SLOT 0" if slot == 0 else ""
        print(f"  Slot {slot}: {mean_pip:.4f}{marker}")

    print("\n" + "=" * 70)
    print("PART 7: Remaining dominoes distribution")
    print("=" * 70)

    # is_remaining shows whether the domino is still in hand
    print("\nProportion of remaining (in-hand) dominoes per slot:")
    for slot in range(7):
        is_remaining = hand_tokens[:, slot, 8]
        pct_remaining = 100 * np.mean(is_remaining)
        marker = " <-- SLOT 0" if slot == 0 else ""
        print(f"  Slot {slot}: {pct_remaining:.2f}%{marker}")

    # Statistical test
    stat, p = stats.kruskal(*[hand_tokens[:, slot, 8] for slot in range(7)])
    print(f"\n  Kruskal-Wallis: stat={stat:.2f}, p={p:.2e}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Summary of findings
    print("\nKey differences for slot 0 vs slots 1-6:")

    # Legal rate difference
    slot0_legal = legal_rates[0]
    slots16_legal = np.mean(legal_rates[1:])
    print(f"  1. Legal rate: {slot0_legal:.4f} vs {slots16_legal:.4f} (diff: {slot0_legal-slots16_legal:+.4f})")

    # Target rate difference
    slot0_target_rate = target_counts.get(0, 0) / total
    slots16_target_rate = sum(target_counts.get(s, 0) for s in range(1, 7)) / total / 6
    print(f"  2. Optimal action rate: {slot0_target_rate:.4f} vs {slots16_target_rate:.4f} (diff: {slot0_target_rate-slots16_target_rate:+.4f})")

    # Trump rank difference
    slot0_trump = np.mean(hand_tokens[:, 0, 4])
    slots16_trump = np.mean([np.mean(hand_tokens[:, slot, 4]) for slot in range(1, 7)])
    print(f"  3. Mean trump_rank: {slot0_trump:.4f} vs {slots16_trump:.4f} (diff: {slot0_trump-slots16_trump:+.4f})")


if __name__ == "__main__":
    main()
