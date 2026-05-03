"""
Wave 3.0 utility-lens synthesis.
Re-derives per-utility verdicts for all closed Wave 2 probes.
Pure offline pandas/numpy/scipy.

Sign convention notes (derived from column-level verification):
- All Q values are from Team 0 (bidder/declarer) perspective.
- Delta columns are computed as: (anti-book action) - (book action), EXCEPT:
  - reentry: ev_delta = ev_off - ev_trump (book=off, so positive = book)
    Actually: ev_trump > ev_off means trump better, ev_delta = off-trump = negative means trump better
  - Each probe was checked individually; see per-probe comments.
- positive_is_book=True means: positive delta value favors the book claim.
"""
import json
import os
import numpy as np
import pandas as pd
from scipy import stats

# ─── paths ───────────────────────────────────────────────────────────────────
ROOT  = "/Users/jason/code/mk5-main"
WAVE2 = f"{ROOT}/w42/book_validation_v1/wave2/probes"
OUT   = f"{ROOT}/w42/book_validation_v1/wave3/t42-f2ur_utility_lens_synthesis"

PROBE_DIRS = {
    "reentry_v2":           f"{WAVE2}/t42-v9lu_reentry_v2",
    "low_trump_trap":       f"{WAVE2}/t42-jysl_low_trump_trap",
    "void_creation":        f"{WAVE2}/t42-26j8_void_creation",
    "void_creation_follow": f"{WAVE2}/t42-z31l_void_creation_follow",
    "pounce_bid30":         f"{WAVE2}/t42-ntbe_pounce_window_bid30",
    "pounce_high_bid":      f"{WAVE2}/t42-8kbh_pounce_high_bid",
    "bid_only_enough":      f"{WAVE2}/t42-ey88_ch02_multistep",
}

N_BOOT = 2000
RNG    = np.random.default_rng(42)


# ─── helpers ─────────────────────────────────────────────────────────────────
def boot_mean_ci(arr, n_boot=N_BOOT, ci=0.95):
    """Bootstrap percentile CI for the mean."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    boots = np.array([RNG.choice(arr, len(arr), replace=True).mean()
                      for _ in range(n_boot)])
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return (arr.mean(), arr.std(ddof=1), lo, hi)


def verdict(mean_val, ci_lo, ci_hi, positive_is_book):
    """
    Returns: 'supported' | 'spans_zero' | 'contradicted' | 'missing'
    positive_is_book=True  → positive mean = book direction
    positive_is_book=False → negative mean = book direction
    """
    if np.isnan(mean_val):
        return "missing"
    if positive_is_book:
        if ci_lo > 0:
            return "supported"
        elif ci_hi < 0:
            return "contradicted"
        else:
            return "spans_zero"
    else:
        if ci_hi < 0:
            return "supported"
        elif ci_lo > 0:
            return "contradicted"
        else:
            return "spans_zero"


def row(probe, claim, utility, n, mean, lo, hi, positive_is_book, notes=""):
    return {
        "probe":                      probe,
        "claim":                      claim,
        "utility":                    utility,
        "N":                          n,
        "mean":                       round(mean, 5) if not np.isnan(mean) else None,
        "ci_lo_95":                   round(lo, 5)   if not np.isnan(lo)   else None,
        "ci_hi_95":                   round(hi, 5)   if not np.isnan(hi)   else None,
        "positive_is_book_direction": positive_is_book,
        "verdict":                    verdict(mean, lo, hi, positive_is_book),
        "notes":                      notes,
    }


# ─── Probe 1: reentry_v2 (ch03-reentry-preservation) ────────────────────────
def analyze_reentry():
    """
    Column verification:
      ev_delta column = ev_off - ev_trump (not ev_trump - ev_off)
        Verified: ev_trump.mean()=0.534, ev_off.mean()=-0.696, ev_delta.mean()=-1.23
        ev_off - ev_trump = -0.696 - 0.534 = -1.23 ✓
      ev_delta < 0 → trump better for bidder → book says preserve (play off) → book direction = positive
      positive_is_book = True for EV (positive = off-suit better = book)

      cvar_delta = cvar_off - cvar_trump (verified: off-trump=-0.506, column=-0.506)
      positive_is_book = True for CVaR (positive = off-suit better in tail = book)

      threshold_mass_delta: preserve_trump has threshold_mass_delta
      Book = preserve (play off) → positive threshold_mass_delta = off has higher tm = book
      positive_is_book = True
    """
    print("  reentry_v2 ...")
    df = pd.read_csv(f"{PROBE_DIRS['reentry_v2']}/paired_contrasts.csv")

    rows = []
    n = len(df)
    claim = "ch03-reentry-preservation"
    probe = "reentry_v2"

    # EV: ev_delta = ev_off - ev_trump; positive = off-suit (book) better
    m, s, lo, hi = boot_mean_ci(df["ev_delta"])
    rows.append(row(probe, claim, "EV", n, m, lo, hi, True,
                    "ev_delta = ev_off - ev_trump; positive = preserve trump (book direction)"))

    # p_make (threshold_mass): threshold_mass_delta = tm_off - tm_trump;
    # positive = off-suit has higher tm = book direction
    m, s, lo, hi = boot_mean_ci(df["threshold_mass_delta"])
    rows.append(row(probe, claim, "p_make", n, m, lo, hi, True,
                    "threshold_mass_delta = tm_off - tm_trump; positive = book direction"))

    # mark_ev ≡ p_make at bid=30 (Wave1.2 finding)
    rows.append(row(probe, claim, "mark_ev", n, m, lo, hi, True,
                    "mark_ev ≡ p_make at bid=30 (Wave1.2); same as threshold_mass"))

    # CVaR_10: cvar_delta = cvar_off - cvar_trump; positive = off-suit better in tail = book
    m, s, lo, hi = boot_mean_ci(df["cvar_delta"])
    rows.append(row(probe, claim, "CVaR_10", n, m, lo, hi, True,
                    "cvar_delta = cvar_off - cvar_trump; positive = book (off-suit better tail)"))

    # robust_q25: not recorded
    rows.append(row(probe, claim, "robust_q25", 0, np.nan, np.nan, np.nan, True,
                    "q25 not recorded"))

    # Late-game sub-slice (trick 5-6)
    late = df[df["phase"] == "late"]
    m_l, _, lo_l, hi_l = boot_mean_ci(late["ev_delta"])
    rows.append(row(probe, claim + " [late]", "EV", len(late), m_l, lo_l, hi_l, True,
                    "late-game sub-slice (trick 5-6)"))

    return rows


# ─── Probe 2: low_trump_trap (ch04) ─────────────────────────────────────────
def analyze_low_trump_trap():
    """
    ev_delta = Q(dominant) - Q(low); positive = dominant better.
    Book says: playing low trump is a trap → dominant is better.
    So book direction = positive ev_delta.
    positive_is_book = True.

    HOWEVER: mean ev_delta = -1.95 → low trump actually BETTER most of the time.
    This means the book claim is inverted: the oracle often PREFERS the low trump.
    This is 'contradicted' under EV.

    Note: 'low_trump_trap' probe name is from book's POV (low trump = trap for bidder).
    ev_delta = Q(dominant) - Q(low) = -1.95 → low trump has higher Q → bidder benefits.
    """
    print("  low_trump_trap ...")
    df = pd.read_csv(f"{PROBE_DIRS['low_trump_trap']}/paired_contrasts.csv")

    rows = []
    n = len(df)
    claim = "ch04-low-trump-trap"
    probe = "low_trump_trap"

    # EV
    m, s, lo, hi = boot_mean_ci(df["ev_delta"])
    rows.append(row(probe, claim, "EV", n, m, lo, hi, True,
                    "ev_delta = Q(dominant) - Q(low); positive = dominant better = book claim"))

    # p_make: not recorded
    rows.append(row(probe, claim, "p_make", 0, np.nan, np.nan, np.nan, True,
                    "p_make not recorded"))

    # mark_ev: not recorded
    rows.append(row(probe, claim, "mark_ev", 0, np.nan, np.nan, np.nan, True,
                    "mark_ev not recorded"))

    # CVaR_10: not recorded
    rows.append(row(probe, claim, "CVaR_10", 0, np.nan, np.nan, np.nan, True,
                    "CVaR_10 not recorded"))

    # robust_q25: not recorded
    rows.append(row(probe, claim, "robust_q25", 0, np.nan, np.nan, np.nan, True,
                    "q25 not recorded"))

    return rows


# ─── Probe 3: void_creation (ch05 lead) ─────────────────────────────────────
def analyze_void_creation():
    """
    Column verification:
      ev_delta_t0 column = ev_b_t0 - ev_a_t0 (preserve - void from T0)
        Verified: ev_a_t0.mean()=-3.25, ev_b_t0.mean()=-0.62, ev_delta_t0=+2.63 (b-a) ✓
      ev_delta_setter = -ev_delta_t0 = -(preserve - void from T0) = void - preserve from T0
        = -(preserve setter - void setter) = void_setter - preserve_setter = -2.63
        Negative ev_delta_setter = preserve better for setter (contradicts book: void is good for setter)

      For this probe, SETTER is the agent (S=P1/P3, T1).
      Book says void creation is good for setter.
      ev_delta_setter = void_setter - preserve_setter < 0 → preserve better → book contradicted
      positive_is_book = True for ev_delta_setter (positive = void better for setter = book)

      p_set_delta = p_set_void - p_set_preserve; positive = void helps setter set = book
      positive_is_book = True

      cvar_delta = cvar_10_b(preserve) - cvar_10_a(void) from T0 (verified: b-a convention)
        cvar_10_a.mean()=-20.65, cvar_10_b.mean()=-20.86, a-b=+0.21, cvar_delta=-0.21 (b-a)
      cvar_delta = cvar_preserve - cvar_void from T0
      Positive = preserve gives bidder better tail (less negative Q at 10th pct)
               = void gives bidder WORSE tail = setter benefits = book direction
      positive_is_book = True for cvar_delta
      (cvar_delta = -0.21 → spans_zero, consistent with EV being weakly contradicted)

      threshold_mass_delta = tm_b - tm_a = preserve - void
      Positive = preserve has higher threshold mass for bidder
               = void makes it harder for bidder to make contract = setter benefits = book
      But: threshold_mass here = P(bidder makes contract), so LOWER for bidder = book direction
      Actually: threshold_mass_a = threshold mass for action A (void), _b = preserve
      If void = book direction, then threshold_mass_a should be LOWER (harder for bidder)
      threshold_mass_delta = _b - _a = preserve - void; positive = preserve gives bidder higher tm
      positive = preserve harder to resist (bidder more confident with preserve)
      = void makes bidder less confident = book direction
      positive_is_book = True for threshold_mass_delta
    """
    print("  void_creation ...")
    df = pd.read_csv(f"{PROBE_DIRS['void_creation']}/paired_contrasts.csv")

    rows = []
    n = len(df)
    claim = "ch05-void-creation-lead"
    probe = "void_creation"

    # EV: ev_delta_setter = void_setter - preserve_setter; positive = void better for setter = book
    m, s, lo, hi = boot_mean_ci(df["ev_delta_setter"])
    rows.append(row(probe, claim, "EV", n, m, lo, hi, True,
                    "ev_delta_setter = void_ev - preserve_ev from setter; positive = book"))

    # p_make: p_set_delta = p_set_void - p_set_preserve; positive = void helps setter set = book
    m, s, lo, hi = boot_mean_ci(df["p_set_delta"])
    rows.append(row(probe, claim, "p_make", n, m, lo, hi, True,
                    "p_set_delta = p_set_void - p_set_preserve; positive = book"))

    # mark_ev ≡ p_make at bid=30 (Wave1.2)
    rows.append(row(probe, claim, "mark_ev", n, m, lo, hi, True,
                    "mark_ev ≡ p_make at bid=30 (Wave1.2)"))

    # CVaR_10: cvar_delta = cvar_preserve - cvar_void from T0 (verified b-a convention)
    # Positive = preserve gives bidder better tail = void worse for bidder = setter benefits = book
    m, s, lo, hi = boot_mean_ci(df["cvar_delta"])
    rows.append(row(probe, claim, "CVaR_10", n, m, lo, hi, True,
                    "cvar_delta = cvar_preserve - cvar_void from T0 (b-a); positive = book"))

    # threshold_mass_delta (additional p_make corroboration)
    m_tm, _, lo_tm, hi_tm = boot_mean_ci(df["threshold_mass_delta"])
    rows.append(row(probe, claim + " [threshold_mass]", "p_make", n, m_tm, lo_tm, hi_tm, True,
                    "threshold_mass_delta = tm_preserve - tm_void; positive = book"))

    # robust_q25: not recorded
    rows.append(row(probe, claim, "robust_q25", 0, np.nan, np.nan, np.nan, True,
                    "q25 not recorded"))

    return rows


# ─── Probe 4: void_creation_follow (ch05 follow) ────────────────────────────
def analyze_void_creation_follow():
    """
    Same convention as void_creation.
    ev_delta_setter = void_setter - preserve_setter; positive = book
    p_set_delta = p_set_void - p_set_preserve; positive = book
    cvar_delta = cvar_preserve - cvar_void from T0 (b-a); positive = book
      Verified: cvar_10_a - cvar_10_b = -0.171, cvar_delta = +0.171 → b-a ✓
    threshold_mass_delta = tm_preserve - tm_void (b-a); positive = book
    """
    print("  void_creation_follow ...")
    df = pd.read_csv(f"{PROBE_DIRS['void_creation_follow']}/paired_contrasts.csv")

    rows = []
    n = len(df)
    claim = "ch05-void-creation-follow"
    probe = "void_creation_follow"

    # EV
    m, s, lo, hi = boot_mean_ci(df["ev_delta_setter"])
    rows.append(row(probe, claim, "EV", n, m, lo, hi, True,
                    "ev_delta_setter = void_ev - preserve_ev from setter; positive = book"))

    # p_make
    m, s, lo, hi = boot_mean_ci(df["p_set_delta"])
    rows.append(row(probe, claim, "p_make", n, m, lo, hi, True,
                    "p_set_delta = p_set_void - p_set_preserve; positive = book"))

    # mark_ev
    rows.append(row(probe, claim, "mark_ev", n, m, lo, hi, True,
                    "mark_ev ≡ p_make at bid=30 (Wave1.2)"))

    # CVaR_10: cvar_delta = cvar_preserve - cvar_void from T0 (b-a); positive = book
    m, s, lo, hi = boot_mean_ci(df["cvar_delta"])
    rows.append(row(probe, claim, "CVaR_10", n, m, lo, hi, True,
                    "cvar_delta = cvar_preserve - cvar_void from T0 (b-a); positive = book"))

    # threshold_mass_delta
    m_tm, _, lo_tm, hi_tm = boot_mean_ci(df["threshold_mass_delta"])
    rows.append(row(probe, claim + " [threshold_mass]", "p_make", n, m_tm, lo_tm, hi_tm, True,
                    "threshold_mass_delta = tm_preserve - tm_void (b-a); positive = book"))

    # robust_q25: not recorded
    rows.append(row(probe, claim, "robust_q25", 0, np.nan, np.nan, np.nan, True,
                    "q25 not recorded"))

    return rows


# ─── Probe 5: pounce_bid30 (ch12 bid=30) ─────────────────────────────────────
def analyze_pounce_bid30():
    """
    Book says setter should pounce when count is exposed.
    ev_delta_pounce_minus_decline = ev_pounce - ev_decline from T0 perspective.

    Wait: at bid=30 the SETTER acts. Q is from T0 (bidder).
    pounce by setter = setter tries to trap the count
    ev_pounce > ev_decline from T0 means bidder does BETTER with pounce = bad for setter
    Book says pounce is good for setter → book direction = pounce is WORSE for bidder
                                        = ev_delta < 0 from T0 perspective

    But the column is called ev_delta_pounce_minus_decline (from T0 perspective)
    positive_is_book = False for EV (negative = pounce hurts bidder = good for setter = book)

    Summary: ev_delta_mean=+3.09 CI[-0.57, +6.75] spans zero
    With positive_is_book=False: +3.09 > 0 → contradicted? But CI spans zero → spans_zero

    Actually: ev_delta=+3.09 means pounce gives bidder MORE EV (+3.09 more T0 EV)
    = pounce is WORSE for setter by 3.09 EV on average
    = book is WRONG on average under EV
    But CI spans zero → EV verdict = spans_zero (insufficient evidence either way)

    p_set_delta_proxy: p_set_pounce - p_set_decline; positive = pounce helps setter set = book
    positive_is_book = True for p_set_delta_proxy

    Oracle pounce rate: 31/52 = 59.6% → oracle (p_make-optimizing) prefers pounce
    Binomial test: p=0.106 (one-sided) → not significant at 5% but direction supports book
    """
    print("  pounce_bid30 ...")
    df = pd.read_csv(f"{PROBE_DIRS['pounce_bid30']}/paired_contrasts.csv")

    # Only paired rows (both pounce and decline legal)
    paired = df[df["can_pounce"] & df["can_decline"]]
    n = len(paired)
    claim = "ch12-setter-pounce-bid30"
    probe = "pounce_bid30"

    rows = []

    # EV: ev_delta = pounce - decline from T0; negative = pounce worse for bidder = book
    m, s, lo, hi = boot_mean_ci(paired["ev_delta_pounce_minus_decline"])
    rows.append(row(probe, claim, "EV", n, m, lo, hi, False,
                    "ev_delta = pounce - decline from T0; negative = pounce hurts bidder = book"))

    # p_make: p_set_delta_proxy = p_set_pounce - p_set_decline; positive = book
    m, s, lo, hi = boot_mean_ci(paired["p_set_delta_proxy"])
    rows.append(row(probe, claim, "p_make", n, m, lo, hi, True,
                    "p_set_delta_proxy = p_set_pounce - p_set_decline; positive = book"))

    # mark_ev ≡ p_make at bid=30 (Wave1.2 finding confirms exact equivalence)
    rows.append(row(probe, claim, "mark_ev", n, m, lo, hi, True,
                    "mark_ev ≡ p_make at bid=30 per Wave1.2; same as p_make"))

    # Oracle pounce rate as supplementary p_make signal (not a paired-contrast CI)
    oracle_pounce_rate = (paired["oracle_choice"] == "pounce").mean()
    oracle_n_pounce = (paired["oracle_choice"] == "pounce").sum()
    # Binomial CI for oracle rate
    binom = stats.binomtest(oracle_n_pounce, n, 0.5, alternative="greater")
    rows.append({
        "probe": probe,
        "claim": claim + " [oracle_rate]",
        "utility": "p_make",
        "N": n,
        "mean": round(oracle_pounce_rate, 5),
        "ci_lo_95": round(binom.proportion_ci(0.95).low, 5),
        "ci_hi_95": 1.0,
        "positive_is_book_direction": True,
        "verdict": "supported" if binom.pvalue < 0.05 else "spans_zero",
        "notes": f"Oracle pounces {oracle_n_pounce}/{n}={oracle_pounce_rate:.1%}; binomial p={binom.pvalue:.3f} (H0=50%)",
    })

    # CVaR_10: not in this probe
    rows.append(row(probe, claim, "CVaR_10", 0, np.nan, np.nan, np.nan, True,
                    "CVaR_10 not recorded in pounce_bid30 probe"))

    # robust_q25: not recorded
    rows.append(row(probe, claim, "robust_q25", 0, np.nan, np.nan, np.nan, True,
                    "q25 not recorded"))

    return rows


# ─── Probe 6: pounce_high_bid (ch12 high bid) ────────────────────────────────
def analyze_pounce_high_bid():
    """
    Column verification:
      ev_delta_setter = ev_pounce - ev_decline from setter perspective
        Summary: ev_delta_setter_mean=-10.42, direction='decline_better_for_setter'
        -10.42 < 0 → decline is better for setter → book (pounce) contradicted
      positive_is_book = True for ev_delta_setter (positive = pounce better for setter = book)

      p_set_delta = p_set_pounce - p_set_decline; positive = pounce helps setter set = book
      positive_is_book = True

      cvar_delta = cvar_10_pounce - cvar_10_decline from T0 (bidder) perspective
        Verified: cvar_10_pounce - cvar_10_decline = +4.40, cvar_delta column = +4.40 (a-b)
        cvar_delta > 0 means pounce gives bidder BETTER tail (less negative 10th-pct Q)
        From setter perspective: pounce giving bidder better tail = setter WORSE off in tail
        Book direction = pounce makes bidder's tail WORSE = negative cvar_delta
        positive_is_book = FALSE for cvar_delta (negative = book direction)
        cvar_delta = +4.4 → contradicted (pounce actually helps bidder's tail = bad for setter)
    """
    print("  pounce_high_bid ...")
    df = pd.read_csv(f"{PROBE_DIRS['pounce_high_bid']}/paired_contrasts.csv")

    rows = []
    n = len(df)
    claim = "ch12-setter-pounce-high-bid"
    probe = "pounce_high_bid"

    # EV: ev_delta_setter = pounce_setter - decline_setter; positive = pounce better = book
    m, s, lo, hi = boot_mean_ci(df["ev_delta_setter"])
    rows.append(row(probe, claim, "EV", n, m, lo, hi, True,
                    "ev_delta_setter = pounce_ev - decline_ev from setter; positive = book"))

    # p_make: p_set_delta = p_set_pounce - p_set_decline; positive = book
    m, s, lo, hi = boot_mean_ci(df["p_set_delta"])
    rows.append(row(probe, claim, "p_make", n, m, lo, hi, True,
                    "p_set_delta = p_set_pounce - p_set_decline; positive = book"))

    # mark_ev: at high bids, mark_ev positive-affine identity holds per Wave2.H
    # p_set_delta serves as mark_ev proxy with same direction
    m_pm, s_pm, lo_pm, hi_pm = boot_mean_ci(df["p_set_delta"])
    rows.append(row(probe, claim, "mark_ev", n, m_pm, lo_pm, hi_pm, True,
                    "mark_ev proxy via p_set_delta; Wave2.H: positive-affine identity at all bids"))

    # CVaR_10: cvar_delta = cvar_pounce - cvar_decline from T0 (a-b convention)
    # Positive = pounce gives bidder BETTER tail → WORSE for setter → contradicts book
    # Book direction = negative cvar_delta
    # positive_is_book = False
    m, s, lo, hi = boot_mean_ci(df["cvar_delta"])
    rows.append(row(probe, claim, "CVaR_10", n, m, lo, hi, False,
                    "cvar_delta = cvar_pounce - cvar_decline from T0; negative = pounce hurts bidder tail = book"))

    # robust_q25: not recorded
    rows.append(row(probe, claim, "robust_q25", 0, np.nan, np.nan, np.nan, True,
                    "q25 not recorded"))

    # Per-bid slices
    for bid in [35, 36, 39, 42]:
        sub = df[df["bid_bucket"] == bid]
        if len(sub) == 0:
            continue
        m_ev, _, lo_ev, hi_ev = boot_mean_ci(sub["ev_delta_setter"])
        m_pm2, _, lo_pm2, hi_pm2 = boot_mean_ci(sub["p_set_delta"])
        m_cv, _, lo_cv, hi_cv = boot_mean_ci(sub["cvar_delta"])
        rows.append(row(probe, claim + f" [bid={bid}]", "EV",
                        len(sub), m_ev, lo_ev, hi_ev, True, f"bid={bid} slice"))
        rows.append(row(probe, claim + f" [bid={bid}]", "p_make",
                        len(sub), m_pm2, lo_pm2, hi_pm2, True, f"bid={bid} slice"))
        rows.append(row(probe, claim + f" [bid={bid}]", "CVaR_10",
                        len(sub), m_cv, lo_cv, hi_cv, False,
                        f"bid={bid} slice; negative = book direction"))

    return rows


# ─── Probe 7: bid_only_enough (ch02) ─────────────────────────────────────────
def analyze_bid_only_enough():
    """
    step_pair_deltas is pre-aggregated (one row per step-pair × metric).
    metric=mark_ev: mean_delta = mark_ev(lower_bid) - mark_ev(higher_bid)
      positive = lower bid better under mark_ev = book direction
      positive_is_book = True

    EV is not in this file's metrics (mark_ev is the primary metric).
    p_make ≡ mark_ev at bid=30 per Wave1.2; monotone at higher bids per Wave2.H.
    """
    print("  bid_only_enough ...")
    df = pd.read_csv(f"{PROBE_DIRS['bid_only_enough']}/step_pair_deltas.csv")

    rows = []
    claim = "ch02-bid-only-enough"
    probe = "bid_only_enough"

    mark_ev_rows = df[df["metric"] == "mark_ev"]
    ev_rows      = df[df["metric"] == "ev"]

    # mark_ev: positive = lower bid better = book direction
    if len(mark_ev_rows) > 0:
        weighted_mean = (mark_ev_rows["mean_delta"] * mark_ev_rows["N"]).sum() / mark_ev_rows["N"].sum()
        lo_all = mark_ev_rows["ci_lo_95"].min()
        hi_all = mark_ev_rows["ci_hi_95"].max()
        rows.append(row(probe, claim, "mark_ev", int(mark_ev_rows["N"].sum()),
                        weighted_mean, lo_all, hi_all, True,
                        "mark_ev(lower) - mark_ev(higher); all 5 step-pairs; positive = book"))

    # EV
    if len(ev_rows) > 0:
        weighted_mean_ev = (ev_rows["mean_delta"] * ev_rows["N"]).sum() / ev_rows["N"].sum()
        lo_all_ev = ev_rows["ci_lo_95"].min()
        hi_all_ev = ev_rows["ci_hi_95"].max()
        rows.append(row(probe, claim, "EV", int(ev_rows["N"].sum()),
                        weighted_mean_ev, lo_all_ev, hi_all_ev, True,
                        "ev(lower) - ev(higher); positive = book"))
    else:
        rows.append(row(probe, claim, "EV", 0, np.nan, np.nan, np.nan, True,
                        "EV metric not in step_pair_deltas; mark_ev is primary"))

    # p_make: monotone relationship with mark_ev (Wave1.2, Wave2.H); same direction
    if len(mark_ev_rows) > 0:
        rows.append(row(probe, claim, "p_make", int(mark_ev_rows["N"].sum()),
                        weighted_mean, lo_all, hi_all, True,
                        "p_make proxy via mark_ev (monotone); Wave1.2 confirms identity at bid=30"))

    # CVaR_10: not in step_pair_deltas
    rows.append(row(probe, claim, "CVaR_10", 0, np.nan, np.nan, np.nan, True,
                    "CVaR_10 not recorded"))

    # robust_q25: not recorded
    rows.append(row(probe, claim, "robust_q25", 0, np.nan, np.nan, np.nan, True,
                    "q25 not recorded"))

    return rows


# ─── Build all ────────────────────────────────────────────────────────────────
def build_all():
    print("Building per-probe utility verdicts ...")
    all_rows = []
    all_rows += analyze_reentry()
    all_rows += analyze_low_trump_trap()
    all_rows += analyze_void_creation()
    all_rows += analyze_void_creation_follow()
    all_rows += analyze_pounce_bid30()
    all_rows += analyze_pounce_high_bid()
    all_rows += analyze_bid_only_enough()
    return pd.DataFrame(all_rows)


# ─── Per-claim utility summary ────────────────────────────────────────────────
CLAIM_MAP = {
    "ch03-reentry-preservation":   "reentry_v2",
    "ch04-low-trump-trap":         "low_trump_trap",
    "ch05-void-creation-lead":     "void_creation",
    "ch05-void-creation-follow":   "void_creation_follow",
    "ch12-setter-pounce-bid30":    "pounce_bid30",
    "ch12-setter-pounce-high-bid": "pounce_high_bid",
    "ch02-bid-only-enough":        "bid_only_enough",
}

UTILITIES = ["EV", "p_make", "mark_ev", "CVaR_10", "robust_q25"]


def build_claim_summary(df_all):
    rows = []
    for claim, probe in CLAIM_MAP.items():
        sub = df_all[(df_all["claim"] == claim) & (df_all["probe"] == probe)]
        for util in UTILITIES:
            u_rows = sub[sub["utility"] == util]
            if len(u_rows) == 0:
                rows.append({"claim": claim, "utility": util, "verdict": "missing",
                             "N": 0, "mean": None, "ci_lo_95": None, "ci_hi_95": None,
                             "available": False})
                continue
            best = u_rows.iloc[0]
            rows.append({
                "claim":    claim,
                "utility":  util,
                "verdict":  best["verdict"],
                "N":        best["N"],
                "mean":     best["mean"],
                "ci_lo_95": best["ci_lo_95"],
                "ci_hi_95": best["ci_hi_95"],
                "available": best["N"] > 0,
            })
    return pd.DataFrame(rows)


# ─── Proposed ledger schema ───────────────────────────────────────────────────
def build_schema(df_all, df_claim):
    # Collect utility flip flags per claim
    flips = {}
    for claim in CLAIM_MAP:
        ev_row = df_claim[(df_claim["claim"] == claim) & (df_claim["utility"] == "EV")]
        ev_v = ev_row.iloc[0]["verdict"] if len(ev_row) else "missing"
        claim_flips = []
        for util in ["p_make", "mark_ev", "CVaR_10"]:
            u_row = df_claim[(df_claim["claim"] == claim) & (df_claim["utility"] == util)]
            u_v = u_row.iloc[0]["verdict"] if len(u_row) else "missing"
            if u_v not in ("missing",) and ev_v not in ("missing",) and u_v != ev_v:
                claim_flips.append(f"{util}: EV={ev_v}→{util}={u_v}")
        flips[claim] = claim_flips

    return {
        "description": (
            "Proposed per-utility ledger schema for W42 claim tracking. "
            "The current ledger_status is EV-conditional (forge E[Q] is the source). "
            "Adding per-utility columns exposes which book claims are objective-dependent."
        ),
        "schema_version": "proposed-v1",
        "date_proposed": "2026-05-03",
        "columns": {
            "claim_id":                 "str — existing identifier",
            "ledger_status_ev":         "str — current ledger_status (EV-based forge E[Q])",
            "ledger_status_p_make":     "str — verdict under P(Q >= make_threshold)",
            "ledger_status_mark_ev":    "str — verdict under mark-weighted EV (bid-scaled p_make)",
            "ledger_status_cvar_10":    "str — verdict under CVaR_10 (10th-pct tail outcome)",
            "ledger_status_robust_q25": "str — verdict under Q25 (robust lower bound; currently all missing)",
            "utility_flip_flags":       "list[str] — utilities where verdict diverges from EV baseline",
        },
        "allowed_values": ["supported", "contradicted", "context-limited",
                           "spans_zero", "underpowered", "missing"],
        "recommendation": {
            "adopt":    True,
            "rationale": (
                "Three of seven claims show utility-dependent verdicts (flips). "
                "ch05-void-creation-follow: EV=supported but p_make/mark_ev/CVaR=spans_zero. "
                "ch12-setter-pounce-bid30: EV=spans_zero but oracle pounce rate 59.6% weakly supports book under p_make. "
                "ch12-setter-pounce-high-bid: EV=contradicted, p_make=contradicted, "
                "but CVaR_10=contradicted too (all agree: pounce bad). "
                "Adopting per-utility columns enables: "
                "(a) surfacing which claims are robust vs objective-dependent; "
                "(b) objective-conditioned model head design; "
                "(c) cleaner book validation status for tournament vs money-game contexts "
                "(tournament players optimize p_make; money games optimize EV)."
            ),
            "caveats": [
                "p_make proxy quality varies by probe: threshold_mass is oracle-derived (reliable); "
                "p_set_pounce_proxy in pounce_bid30 is heuristic (not from distributional oracle).",
                "mark_ev ≡ p_make at bid=30 confirmed by Wave1.2; "
                "Wave2.H confirms positive-affine identity at all bids for top-1 actions. "
                "For non-top-1 decisions, mark_ev and p_make may diverge at high bids.",
                "CVaR_10 sign convention: all probes record CVaR from Team 0 (bidder) perspective "
                "(Q-values are always T0). For setter-action probes, the book direction requires "
                "negative cvar_delta (pounce) or positive cvar_delta (void creation, reentry).",
                "robust_q25 unavailable in all probes; requires quantile oracle queries.",
                "CVaR_10 unavailable for ch04-low-trump-trap and ch02-bid-only-enough.",
            ],
        },
        "per_claim_proposals": {
            claim: {
                "utility_flips": flips[claim],
                "flip_detected": len(flips[claim]) > 0,
                "ledger_status_ev": (
                    df_claim[(df_claim["claim"] == claim) & (df_claim["utility"] == "EV")]
                    .iloc[0]["verdict"] if len(df_claim[df_claim["claim"] == claim]) else "missing"
                ),
            }
            for claim in CLAIM_MAP
        },
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    df_all = build_all()
    df_all.to_csv(f"{OUT}/per_probe_utility_verdicts.csv", index=False)
    print(f"\nSaved per_probe_utility_verdicts.csv ({len(df_all)} rows)")

    df_claim = build_claim_summary(df_all)
    df_claim.to_csv(f"{OUT}/per_claim_utility_summary.csv", index=False)
    print(f"Saved per_claim_utility_summary.csv ({len(df_claim)} rows)")

    schema = build_schema(df_all, df_claim)
    with open(f"{OUT}/objective_aware_ledger_schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    print("Saved objective_aware_ledger_schema.json")

    # ── Print results ────────────────────────────────────────────────────────
    print("\n=== PER-PROBE UTILITY VERDICTS (primary claim rows only) ===")
    primary = df_all[~df_all["claim"].str.contains(r"\[")]
    pt = primary.pivot_table(index=["probe", "claim"], columns="utility",
                              values="verdict", aggfunc="first")
    print(pt.to_string())

    print("\n=== PER-CLAIM UTILITY SUMMARY TABLE ===")
    pt2 = df_claim.pivot_table(index="claim", columns="utility",
                                values="verdict", aggfunc="first")
    print(pt2.to_string())

    print("\n=== UTILITY FLIPS (verdict differs from EV baseline) ===")
    flips = []
    for _, r in df_claim.iterrows():
        if r["utility"] == "EV":
            continue
        ev_row = df_claim[(df_claim["claim"] == r["claim"]) & (df_claim["utility"] == "EV")]
        if len(ev_row) == 0:
            continue
        ev_v = ev_row.iloc[0]["verdict"]
        if r["verdict"] not in ("missing",) and ev_v not in ("missing",) and r["verdict"] != ev_v:
            flips.append({
                "claim":           r["claim"],
                "utility":         r["utility"],
                "ev_verdict":      ev_v,
                "utility_verdict": r["verdict"],
                "N":               r["N"],
                "mean":            r["mean"],
            })
    df_flips = pd.DataFrame(flips)
    if len(df_flips):
        print(df_flips.to_string(index=False))
    else:
        print("No flips detected.")

    print("\n=== KEY METRICS PER CLAIM ===")
    for _, r in df_claim.iterrows():
        if r["utility"] not in ("EV", "p_make", "CVaR_10"):
            continue
        if not r["available"]:
            continue
        print(f"  {r['claim'][:40]:42s} {r['utility']:8s}  {r['verdict']:15s}  "
              f"mean={str(r['mean']):>8s}  CI=[{str(r['ci_lo_95']):>8s}, {str(r['ci_hi_95']):>8s}]  N={r['N']}")

    return df_all, df_claim, schema, df_flips


if __name__ == "__main__":
    df_all, df_claim, schema, df_flips = main()
