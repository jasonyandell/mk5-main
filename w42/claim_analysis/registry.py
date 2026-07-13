"""Built-in claim-label registry for the reusable w42 harness."""

from __future__ import annotations

from .harness import ClaimSpec


GUS_TACTICAL_SPECS = (
    ClaimSpec(
        claim_id="ch05-pounce-count-before-certainty",
        family="setter defense",
        label="ch05_setter_pounce_count_before_certainty",
        description="Defender can take an offense-controlled trick with count before all later seats have played.",
        online_fields=("seat_role", "team", "trick_position", "current_winner_team_before", "candidate_count_points", "candidate_beats_current"),
        offline_label_fields=("mean", "threshold_mass", "lower_tail_mass", "q_per_world"),
        leakage_policy="Public/action-local detector; oracle and sampled-world values are offline labels only.",
    ),
    ClaimSpec(
        claim_id="ch05-pounce-count",
        family="setter defense",
        label="ch05_setter_pounce_count",
        description="Defender can take an offense-controlled trick with count.",
        online_fields=("seat_role", "team", "trick_position", "current_winner_team_before", "candidate_count_points", "candidate_beats_current"),
        offline_label_fields=("mean", "threshold_mass", "lower_tail_mass", "q_per_world"),
        leakage_policy="Public/action-local detector; oracle and sampled-world values are offline labels only.",
    ),
    ClaimSpec(
        claim_id="ch05-extra-count-to-set",
        family="setter defense",
        label="ch05_setter_pounce_count_sets_now",
        description="Pounce-count candidate would reach the set threshold if it wins the current trick.",
        online_fields=("bid_value", "defense_score_before", "current_trick_count_before", "candidate_count_points", "candidate_would_win_trick_now"),
        offline_label_fields=("mean", "threshold_mass", "lower_tail_mass", "q_per_world"),
        leakage_policy="Set-threshold arithmetic is public from bid/score/trick state; oracle values remain labels only.",
    ),
    ClaimSpec(
        claim_id="ch05-reckless-count-to-bidder",
        family="setter defense",
        label="ch05_reckless_count_to_bidder",
        description="Defender plays count into an offense-controlled trick without winning it now.",
        online_fields=("team", "current_winner_team_before", "candidate_count_points", "candidate_would_win_trick_now"),
        offline_label_fields=("mean", "threshold_mass", "lower_tail_mass", "q_per_world"),
        leakage_policy="Public/action-local negative-control detector; oracle values remain labels only.",
    ),
    ClaimSpec(
        claim_id="ch04-safe-partner-count-donation",
        family="partner support",
        label="ch04_partner_safe_count_donation_current_control",
        description="Bidder partner donates count while the bidder team is currently winning the trick.",
        online_fields=("seat_role", "current_winner_team_before", "candidate_count_points"),
        offline_label_fields=("mean", "threshold_mass", "lower_tail_mass", "q_per_world"),
        leakage_policy="Current-control is public; guaranteed future control and oracle values are offline labels.",
    ),
    ClaimSpec(
        claim_id="ch04-unsafe-partner-count-donation",
        family="partner support",
        label="ch04_partner_unsafe_count_to_defense",
        description="Bidder partner donates count into a defense-controlled trick the candidate cannot beat.",
        online_fields=("seat_role", "current_winner_team_before", "candidate_count_points", "candidate_beats_current"),
        offline_label_fields=("mean", "threshold_mass", "lower_tail_mass", "q_per_world"),
        leakage_policy="Public/action-local negative-control detector; oracle values remain labels only.",
    ),
)


BRANCH_ATLAS_SPECS = (
    ClaimSpec(
        claim_id="distribution-scalar-ev-omission",
        family="distribution-aware EV",
        label="scalar_ev_lying_by_omission",
        description="Top scalar mean disagrees with threshold, lower-tail, or high-variance branch evidence.",
        online_fields=("public decision context", "candidate action"),
        offline_label_fields=("mean", "threshold_mass", "lower_tail_mass_le_neg18", "q_per_world"),
        leakage_policy="Distribution labels are offline training/eval labels, not live hidden-truth features.",
    ),
    ClaimSpec(
        claim_id="distribution-large-lower-tail",
        family="distribution-aware EV",
        label="large_lower_tail",
        description="Action has a large lower-tail/disaster-mass component.",
        online_fields=("public decision context", "candidate action"),
        offline_label_fields=("lower_tail_mass_le_neg18", "cvar_low_10", "q_per_world"),
        leakage_policy="Tail labels are offline evaluation labels.",
    ),
    ClaimSpec(
        claim_id="hidden-threat-large-impact",
        family="hidden-threat belief impact",
        label="hidden_threat_large_impact",
        description="Offline hidden holder/domino attribution has large outcome impact.",
        online_fields=("public decision context", "candidate action"),
        offline_label_fields=("world_hands", "q_per_world", "top_hidden_impact_*"),
        leakage_policy="Hidden holder truth is report-only and must become learned belief before live use.",
    ),
    ClaimSpec(
        claim_id="position-first-setter-pounce-window",
        family="seat/position",
        label="first_setter_pounce_window",
        description="First setter response window in the position strategy map.",
        online_fields=("seat_role", "trick_position", "current_winner_team_before", "candidate action"),
        offline_label_fields=("mean", "threshold_mass", "lower_tail_mass_le_neg18"),
        leakage_policy="Position/context detector is public; outcomes are labels.",
    ),
    ClaimSpec(
        claim_id="position-partner-count-donation",
        family="partner support",
        label="partner_count_donation",
        description="Partner count-donation context tag from branch-atlas strategy surface.",
        online_fields=("seat_role", "trick_position", "current_winner_team_before", "candidate_count_points"),
        offline_label_fields=("mean", "threshold_mass", "lower_tail_mass_le_neg18"),
        leakage_policy="Public context only; hidden partner need is an offline diagnostic unless belief-derived.",
    ),
    ClaimSpec(
        claim_id="position-needs-real-bid-margin",
        family="bidding risk",
        label="needs_real_bid_margin",
        description="Claim slice that cannot be settled on fixed-bid branch-atlas data.",
        online_fields=("bid_value", "score", "public auction when available"),
        offline_label_fields=("make/set labels", "counterfactual bid outcomes"),
        leakage_policy="Future bid outcomes are labels only; current fixed bid value is not bid margin.",
    ),
)


CHAMPION_SPECS = (
    ClaimSpec(
        claim_id="ch05-pounce-count-before-certainty",
        family="setter defense",
        label="ch05_setter_pounce_count_before_certainty",
        description="Defender takes offense-controlled trick with count before all later seats play; champion slice.",
        online_fields=("seat_role", "team", "trick_position", "current_winner_team_before",
                       "candidate_count_points", "candidate_beats_current"),
        offline_label_fields=("mean", "threshold_mass", "lower_tail_mass", "q_per_world"),
        leakage_policy="Public/action-local detector; oracle and sampled-world values are offline labels only.",
    ),
    ClaimSpec(
        claim_id="ch05-pounce-count",
        family="setter defense",
        label="ch05_setter_pounce_count",
        description="Defender takes offense-controlled trick with count; champion slice.",
        online_fields=("seat_role", "team", "trick_position", "current_winner_team_before",
                       "candidate_count_points", "candidate_beats_current"),
        offline_label_fields=("mean", "threshold_mass", "lower_tail_mass", "q_per_world"),
        leakage_policy="Public/action-local detector; oracle and sampled-world values are offline labels only.",
    ),
    ClaimSpec(
        claim_id="ch05-extra-count-to-set",
        family="setter defense",
        label="ch05_setter_pounce_count_sets_now",
        description="Pounce-count candidate reaches set threshold if it wins the current trick; champion slice.",
        online_fields=("bid_value", "defense_score_before", "current_trick_count_before",
                       "candidate_count_points", "candidate_would_win_trick_now"),
        offline_label_fields=("mean", "threshold_mass", "lower_tail_mass", "q_per_world"),
        leakage_policy="Set-threshold arithmetic is public from bid/score/trick state; oracle values remain labels only.",
    ),
    ClaimSpec(
        claim_id="ch05-reckless-count-to-bidder",
        family="setter defense",
        label="ch05_reckless_count_to_bidder",
        description="Defender plays count into offense-controlled trick without winning it; negative control; champion slice.",
        online_fields=("team", "current_winner_team_before", "candidate_count_points",
                       "candidate_would_win_trick_now"),
        offline_label_fields=("mean", "threshold_mass", "lower_tail_mass", "q_per_world"),
        leakage_policy="Public/action-local negative-control detector; oracle values remain labels only.",
    ),
    ClaimSpec(
        claim_id="ch04-safe-partner-count-donation",
        family="partner support",
        label="ch04_partner_safe_count_donation_current_control",
        description="Bidder partner donates count while bidder team currently winning; champion slice.",
        online_fields=("seat_role", "current_winner_team_before", "candidate_count_points"),
        offline_label_fields=("mean", "threshold_mass", "lower_tail_mass", "q_per_world"),
        leakage_policy="Current-control is public; guaranteed future control and oracle values are offline labels.",
    ),
    ClaimSpec(
        claim_id="ch04-unsafe-partner-count-donation",
        family="partner support",
        label="ch04_partner_unsafe_count_to_defense",
        description="Bidder partner donates count into defense-controlled trick they cannot beat; negative control; champion slice.",
        online_fields=("seat_role", "current_winner_team_before", "candidate_count_points",
                       "candidate_beats_current"),
        offline_label_fields=("mean", "threshold_mass", "lower_tail_mass", "q_per_world"),
        leakage_policy="Public/action-local negative-control detector; oracle values remain labels only.",
    ),
)


def specs_for_source_kind(source_kind: str) -> tuple[ClaimSpec, ...]:
    if source_kind == "gus_claim_rows":
        return GUS_TACTICAL_SPECS
    if source_kind in {"branch_atlas_actions", "phase2_decision_actions"}:
        return BRANCH_ATLAS_SPECS
    if source_kind == "champion_play_rows":
        return CHAMPION_SPECS
    return ()
