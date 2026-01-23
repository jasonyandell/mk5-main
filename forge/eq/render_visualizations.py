#!/usr/bin/env python3
"""E[Q] Visualization Renderer.

Generates beautiful visualizations of the E[Q] pipeline:
- Static decision snapshots with domino graphics
- 3D animated trajectory through decision space
- 3D animated belief cloud showing uncertainty evolution

Outputs GIFs suitable for MMS sharing.

Usage:
    python -m forge.eq.render_visualizations
    python -m forge.eq.render_visualizations --game 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA

# === Configuration ===
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "forge/data/eq_v2.2_250g.pt"
OUTPUT_DIR = PROJECT_ROOT / "forge/eq/renders"

# === Domino Rendering ===
PIP_POSITIONS = {
    0: [],
    1: [(0.5, 0.5)],
    2: [(0.25, 0.75), (0.75, 0.25)],
    3: [(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)],
    4: [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)],
    5: [(0.25, 0.25), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (0.75, 0.75)],
    6: [(0.25, 0.2), (0.25, 0.5), (0.25, 0.8), (0.75, 0.2), (0.75, 0.5), (0.75, 0.8)],
}

FEAT_HIGH = 0
FEAT_LOW = 1
FEAT_PLAYER = 4
FEAT_DECL = 6
FEAT_TOKEN_TYPE = 7
TOKEN_TYPE_HAND = 1
TOKEN_TYPE_PLAY = 2

DECL_NAMES = {
    0: "Blanks", 1: "Ones", 2: "Twos", 3: "Threes", 4: "Fours",
    5: "Fives", 6: "Sixes", 7: "Doubles", 8: "Doubles (suit)", 9: "No Trump"
}
PLAYER_NAMES = ["ME", "LEFT", "PARTNER", "RIGHT"]


def draw_domino(ax, x, y, high, low, width=1.0, height=2.0,
                face_color='#1a1a2e', edge_color='#4a9eff', pip_color='white',
                highlight=False, alpha=1.0):
    """Draw a domino at (x, y) with given pips."""
    if highlight:
        edge_color = '#ffcc00'
        face_color = '#2a2a4e'

    rect = mpatches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=face_color, edgecolor=edge_color, linewidth=2,
        alpha=alpha
    )
    ax.add_patch(rect)

    ax.plot([x + 0.1, x + width - 0.1], [y + height/2, y + height/2],
            color=edge_color, linewidth=1.5, alpha=alpha)

    pip_radius = width * 0.08
    for px, py in PIP_POSITIONS[high]:
        circle = plt.Circle(
            (x + px * width, y + height/2 + py * height/2),
            pip_radius, color=pip_color, alpha=alpha
        )
        ax.add_patch(circle)

    for px, py in PIP_POSITIONS[low]:
        circle = plt.Circle(
            (x + px * width, y + py * height/2),
            pip_radius, color=pip_color, alpha=alpha
        )
        ax.add_patch(circle)


def domino_id_to_pips(domino_id):
    """Convert domino ID (0-27) to (high, low) pips."""
    hi = 0
    while (hi + 1) * (hi + 2) // 2 <= domino_id:
        hi += 1
    lo = domino_id - hi * (hi + 1) // 2
    return hi, lo


def decode_decision(tok, length):
    """Decode transcript tokens into game state."""
    tok = tok[:length]
    decl_id = tok[0, FEAT_DECL].item()

    hand = []
    plays = []

    for i in range(length):
        tt = tok[i, FEAT_TOKEN_TYPE].item()
        if tt == TOKEN_TYPE_HAND:
            hand.append((tok[i, FEAT_HIGH].item(), tok[i, FEAT_LOW].item()))
        elif tt == TOKEN_TYPE_PLAY:
            plays.append((
                tok[i, FEAT_PLAYER].item(),
                tok[i, FEAT_HIGH].item(),
                tok[i, FEAT_LOW].item()
            ))

    return {'decl_id': decl_id, 'hand': hand, 'plays': plays}


def render_decision_snapshot(data, idx, save_path=None):
    """Render a beautiful snapshot of a single decision."""
    tokens = data['transcript_tokens']
    lengths = data['transcript_lengths']
    e_q_mean = data['e_q_mean']
    e_q_var = data.get('e_q_var')
    legal_mask = data['legal_mask']
    action_taken = data['action_taken']
    game_idx = data['game_idx']
    decision_idx = data['decision_idx']
    ess = data.get('ess')

    tok = tokens[idx]
    length = lengths[idx].item()
    state = decode_decision(tok, length)

    eq = e_q_mean[idx]
    var = e_q_var[idx] if e_q_var is not None else None
    mask = legal_mask[idx]
    action = action_taken[idx].item()
    game = game_idx[idx].item()
    decision = decision_idx[idx].item()

    hand = state['hand']
    plays = state['plays']
    decl_id = state['decl_id']

    n_complete_tricks = len(plays) // 4
    trick_num = n_complete_tricks + 1
    current_trick = plays[-(len(plays) % 4):] if len(plays) % 4 != 0 else []

    # Setup style
    plt.style.use('dark_background')

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0d1117')

    gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 2], width_ratios=[1.5, 1],
                          hspace=0.3, wspace=0.2)

    # Header
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_facecolor('#0d1117')
    ax_header.axis('off')
    ax_header.text(0.5, 0.7, f"Game {game + 1}  •  Decision {decision + 1}/28  •  Trick {trick_num}/7",
                   ha='center', va='center', fontsize=20, color='white', weight='bold')
    ax_header.text(0.5, 0.2, f"Declaration: {DECL_NAMES[decl_id]}",
                   ha='center', va='center', fontsize=16, color='#4a9eff')

    # Hand Display
    ax_hand = fig.add_subplot(gs[1, 0])
    ax_hand.set_facecolor('#0d1117')
    ax_hand.set_xlim(-0.5, len(hand) * 1.5 + 0.5)
    ax_hand.set_ylim(-0.5, 3)
    ax_hand.set_aspect('equal')
    ax_hand.axis('off')
    ax_hand.set_title('My Hand', fontsize=14, color='white', pad=10)

    for i, (high, low) in enumerate(hand):
        is_selected = (i == action)
        is_legal = mask[i].item() if i < len(mask) else False
        alpha = 1.0 if is_legal else 0.4
        draw_domino(ax_hand, i * 1.5, 0.5, high, low, highlight=is_selected, alpha=alpha)
        if is_selected:
            ax_hand.text(i * 1.5 + 0.5, -0.3, "▲ PLAYED", ha='center',
                        fontsize=10, color='#ffcc00', weight='bold')

    # E[Q] Bar Chart
    ax_eq = fig.add_subplot(gs[1, 1])
    ax_eq.set_facecolor('#0d1117')

    legal_indices = [i for i in range(len(hand)) if mask[i].item()]
    sorted_legal = sorted(legal_indices, key=lambda i: eq[i].item(), reverse=True)

    cmap = LinearSegmentedColormap.from_list('eq', ['#ff4444', '#888888', '#44ff44'])
    eq_vals = [eq[i].item() for i in sorted_legal]
    eq_min, eq_max = min(eq_vals), max(eq_vals)
    eq_range = eq_max - eq_min if eq_max != eq_min else 1

    y_pos = np.arange(len(sorted_legal))
    colors = [cmap((eq[i].item() - eq_min) / eq_range) for i in sorted_legal]

    bars = ax_eq.barh(y_pos, [eq[i].item() for i in sorted_legal],
                      color=colors, edgecolor='white', linewidth=0.5)

    if var is not None:
        stds = [np.sqrt(max(0, var[i].item())) for i in sorted_legal]
        ax_eq.errorbar([eq[i].item() for i in sorted_legal], y_pos,
                       xerr=stds, fmt='none', color='white', alpha=0.5, capsize=3)

    labels = [f"[{hand[i][0]}:{hand[i][1]}]" for i in sorted_legal]
    ax_eq.set_yticks(y_pos)
    ax_eq.set_yticklabels(labels, fontsize=12, color='white')
    ax_eq.set_xlabel('E[Q] (expected points)', fontsize=12, color='white')
    ax_eq.set_title('Action Values', fontsize=14, color='white', pad=10)
    ax_eq.axvline(0, color='white', linestyle='--', alpha=0.3)
    ax_eq.tick_params(colors='white')

    for i, idx_action in enumerate(sorted_legal):
        if idx_action == action:
            bars[i].set_edgecolor('#ffcc00')
            bars[i].set_linewidth(3)
            ax_eq.text(eq[idx_action].item() + 1, i, '← PLAYED',
                      va='center', fontsize=10, color='#ffcc00')

    # Current Trick
    ax_trick = fig.add_subplot(gs[2, 0])
    ax_trick.set_facecolor('#0d1117')
    ax_trick.set_xlim(-0.5, 8)
    ax_trick.set_ylim(-0.5, 3)
    ax_trick.set_aspect('equal')
    ax_trick.axis('off')

    if current_trick:
        ax_trick.set_title(f'Current Trick ({len(current_trick)}/4 played)',
                          fontsize=14, color='white', pad=10)
        for i, (player, high, low) in enumerate(current_trick):
            draw_domino(ax_trick, i * 1.8, 0.5, high, low)
            ax_trick.text(i * 1.8 + 0.5, 2.8, PLAYER_NAMES[player],
                         ha='center', fontsize=10, color='#4a9eff')
        ax_trick.text(len(current_trick) * 1.8 + 0.5, 1.5, "?",
                     ha='center', fontsize=24, color='#ffcc00')
        ax_trick.text(len(current_trick) * 1.8 + 0.5, 2.8, "ME",
                     ha='center', fontsize=10, color='#ffcc00')
    else:
        ax_trick.set_title('Leading new trick...', fontsize=14, color='white', pad=10)
        ax_trick.text(2, 1.5, "Your lead!", ha='center', fontsize=16, color='#44ff44')

    # Stats Panel
    ax_stats = fig.add_subplot(gs[2, 1])
    ax_stats.set_facecolor('#0d1117')
    ax_stats.axis('off')

    stats_text = f"""Statistics
─────────────────
Legal actions: {len(legal_indices)}
Best E[Q]: {eq_max:+.1f} pts
Worst E[Q]: {eq_min:+.1f} pts
Gap: {eq_max - eq_min:.1f} pts
"""
    if var is not None:
        best_idx = sorted_legal[0]
        stats_text += f"Best σ: ±{np.sqrt(max(0, var[best_idx].item())):.1f} pts\n"
    if ess is not None:
        stats_text += f"\nESS: {ess[idx].item():.1f}"

    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                  fontsize=12, color='white', va='top', family='monospace',
                  bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#4a9eff'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def create_trajectory_animation(data, game_id=0, save_path=None, fps=6):
    """Create 3D trajectory animation for a single game."""
    tokens = data['transcript_tokens']
    lengths = data['transcript_lengths']
    e_q_mean = data['e_q_mean']
    e_q_var = data.get('e_q_var')
    legal_mask = data['legal_mask']
    game_idx_arr = data['game_idx']

    game_mask = (game_idx_arr == game_id)
    indices = torch.where(game_mask)[0].numpy()

    if len(indices) == 0:
        print(f"No data for game {game_id}")
        return None

    print(f"Game {game_id}: {len(indices)} decisions")

    game_tokens = tokens[indices].numpy()
    game_lengths = lengths[indices].numpy()
    game_eq = e_q_mean[indices].numpy()
    game_var = e_q_var[indices].numpy() if e_q_var is not None else None

    embeddings = []
    for tok, length in zip(game_tokens, game_lengths):
        emb = tok[:length].mean(axis=0)
        embeddings.append(emb)
    embeddings = np.array(embeddings)

    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(embeddings)

    best_eq = []
    for i in range(len(indices)):
        mask = legal_mask[indices[i]].numpy()
        eq_vals = game_eq[i]
        legal_eq = eq_vals[mask]
        best_eq.append(legal_eq.max() if len(legal_eq) > 0 else 0)
    best_eq = np.array(best_eq)

    if game_var is not None:
        uncertainties = []
        for i in range(len(indices)):
            mask = legal_mask[indices[i]].numpy()
            var_vals = game_var[i]
            legal_var = var_vals[mask]
            uncertainties.append(np.sqrt(legal_var.mean()) if len(legal_var) > 0 else 0)
        uncertainties = np.array(uncertainties)
        point_sizes = 50 + uncertainties * 10
    else:
        point_sizes = np.full(len(indices), 80)

    eq_norm = (best_eq - best_eq.min()) / (best_eq.max() - best_eq.min() + 1e-8)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor('#0d1117')
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.cm.plasma
    margin = 0.1

    def update(frame):
        ax.clear()

        ax.set_facecolor('#0d1117')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#333333')
        ax.yaxis.pane.set_edgecolor('#333333')
        ax.zaxis.pane.set_edgecolor('#333333')
        ax.tick_params(colors='#666666', labelsize=8)
        ax.set_xlabel('PC1', color='#666666', fontsize=9)
        ax.set_ylabel('PC2', color='#666666', fontsize=9)
        ax.set_zlabel('PC3', color='#666666', fontsize=9)
        ax.set_xlim(coords_3d[:, 0].min() - margin, coords_3d[:, 0].max() + margin)
        ax.set_ylim(coords_3d[:, 1].min() - margin, coords_3d[:, 1].max() + margin)
        ax.set_zlim(coords_3d[:, 2].min() - margin, coords_3d[:, 2].max() + margin)

        n = frame + 1
        xs, ys, zs = coords_3d[:n, 0], coords_3d[:n, 1], coords_3d[:n, 2]

        ax.plot(xs, ys, zs, color='#4a9eff', linewidth=2, alpha=0.6)

        colors = cmap(eq_norm[:n])
        ax.scatter(xs, ys, zs, c=colors, s=point_sizes[:n], alpha=0.8,
                   edgecolors='white', linewidth=0.5)

        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], c='#ffcc00', s=200,
                   marker='*', edgecolors='white', linewidth=2, zorder=10)

        ax.view_init(elev=20, azim=frame * 4)

        trick = (frame // 4) + 1
        ax.set_title(f'Game {game_id + 1} • Decision {frame + 1}/28 • Trick {trick}/7\n'
                    f'E[Q] = {best_eq[frame]:+.1f} pts',
                    fontsize=12, color='white', pad=10)

        return []

    n_frames = len(indices)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000//fps, blit=False)

    if save_path:
        print(f"Saving trajectory animation to {save_path}...")
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=80,
                  savefig_kwargs={'facecolor': '#0d1117'})
        print(f"Saved: {save_path}")

    plt.close()
    return anim


def create_belief_cloud_animation(data, game_id=0, n_particles=150, save_path=None, fps=5):
    """Create 3D belief cloud animation showing uncertainty evolution."""
    e_q_mean = data['e_q_mean']
    e_q_var = data.get('e_q_var')
    legal_mask = data['legal_mask']
    game_idx_arr = data['game_idx']
    ess = data.get('ess')

    game_mask = (game_idx_arr == game_id)
    indices = torch.where(game_mask)[0].numpy()

    if len(indices) == 0:
        print(f"No data for game {game_id}")
        return None

    print(f"Game {game_id}: {len(indices)} decisions")

    game_var = e_q_var[indices].numpy() if e_q_var is not None else None
    game_ess = ess[indices].numpy() if ess is not None else None
    game_eq = e_q_mean[indices].numpy()

    uncertainties = []
    for i in range(len(indices)):
        mask = legal_mask[indices[i]].numpy()
        if game_var is not None:
            var_vals = game_var[i][mask]
            uncertainties.append(np.sqrt(var_vals.mean()) if len(var_vals) > 0 else 5)
        else:
            uncertainties.append(5)
    uncertainties = np.array(uncertainties)

    max_spread = 2.0
    min_spread = 0.15
    u_norm = uncertainties / (uncertainties.max() + 1e-8)
    spreads = min_spread + u_norm * (max_spread - min_spread)

    centers = []
    for i in range(len(indices)):
        mask = legal_mask[indices[i]].numpy()
        eq_vals = game_eq[i][mask]
        centers.append(eq_vals.mean() if len(eq_vals) > 0 else 0)
    centers = np.array(centers)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor('#0d1117')
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()

        ax.set_facecolor('#0d1117')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#333333')
        ax.yaxis.pane.set_edgecolor('#333333')
        ax.zaxis.pane.set_edgecolor('#333333')
        ax.tick_params(colors='#666666', labelsize=8)
        ax.set_xlabel('Belief X', color='#666666', fontsize=9)
        ax.set_ylabel('Belief Y', color='#666666', fontsize=9)
        ax.set_zlabel('E[Q]', color='#666666', fontsize=9)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(centers.min() - 5, centers.max() + 5)

        spread = spreads[frame]
        center_eq = centers[frame]
        current_ess = game_ess[frame] if game_ess is not None else 100

        rng_frame = np.random.default_rng(42 + frame)
        xs = rng_frame.normal(0, spread, n_particles)
        ys = rng_frame.normal(0, spread, n_particles)
        zs = rng_frame.normal(center_eq, spread * 2, n_particles)

        dists = np.sqrt(xs**2 + ys**2 + (zs - center_eq)**2)
        weights = np.exp(-dists / (spread + 0.1))
        weights = weights / weights.max()

        colors = plt.cm.Blues(0.3 + weights * 0.7)
        sizes = 15 + weights * 50

        ax.scatter(xs, ys, zs, c=colors, s=sizes, alpha=0.6, edgecolors='none')

        ax.scatter([0], [0], [center_eq], c='#ffcc00', s=400, marker='*',
                   edgecolors='white', linewidth=2, zorder=10)

        if frame > 0:
            traj_z = centers[:frame+1]
            traj_x = np.zeros(frame+1)
            traj_y = np.zeros(frame+1)
            ax.plot(traj_x, traj_y, traj_z, color='#ffcc00', linewidth=3, alpha=0.8)

        ax.view_init(elev=25, azim=frame * 5 + 45)

        trick = (frame // 4) + 1
        ess_str = f"ESS={current_ess:.0f}" if game_ess is not None else ""
        ax.set_title(f'Belief Cloud • Decision {frame + 1}/28 • Trick {trick}/7\n'
                    f'Uncertainty: ±{uncertainties[frame]:.1f} pts  {ess_str}',
                    fontsize=12, color='white', pad=10)

        return []

    n_frames = len(indices)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000//fps, blit=False)

    if save_path:
        print(f"Saving belief cloud animation to {save_path}...")
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=80,
                  savefig_kwargs={'facecolor': '#0d1117'})
        print(f"Saved: {save_path}")

    plt.close()
    return anim


def render_all_dominoes(save_path=None):
    """Render all 28 dominoes in a grid."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 10.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('All 28 Dominoes', fontsize=20, color='white', pad=20)

    for domino_id in range(28):
        high, low = domino_id_to_pips(domino_id)
        row = domino_id // 7
        col = domino_id % 7
        draw_domino(ax, col * 2, (3 - row) * 2.5, high, low, highlight=(high == low))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        print(f"Saved: {save_path}")

    plt.close()


def render_data_overload(data, idx, save_path=None):
    """MAXIMUM DATA DENSITY - everything we know about one decision."""
    from forge.oracle.tables import can_follow, led_suit_for_lead_domino

    tokens = data['transcript_tokens']
    lengths = data['transcript_lengths']
    e_q_mean = data['e_q_mean']
    e_q_var = data.get('e_q_var')
    legal_mask = data['legal_mask']
    action_taken = data['action_taken']
    game_idx = data['game_idx']
    decision_idx = data['decision_idx']
    ess_arr = data.get('ess')
    max_w_arr = data.get('max_w')
    exploration_mode = data.get('exploration_mode')
    q_gap_arr = data.get('q_gap')

    tok = tokens[idx]
    length = lengths[idx].item()
    state = decode_decision(tok, length)

    eq = e_q_mean[idx]
    var = e_q_var[idx] if e_q_var is not None else None
    mask = legal_mask[idx]
    action = action_taken[idx].item()
    game = game_idx[idx].item()
    decision = decision_idx[idx].item()
    current_ess = ess_arr[idx].item() if ess_arr is not None else None
    current_max_w = max_w_arr[idx].item() if max_w_arr is not None else None
    current_exp_mode = exploration_mode[idx].item() if exploration_mode is not None else None
    current_q_gap = q_gap_arr[idx].item() if q_gap_arr is not None else None

    hand = state['hand']
    plays = state['plays']
    decl_id = state['decl_id']

    # Group plays into tricks
    tricks = [plays[i:i+4] for i in range(0, len(plays), 4)]
    n_complete = sum(1 for t in tricks if len(t) == 4)
    trick_num = n_complete + 1
    current_trick = plays[-(len(plays) % 4):] if len(plays) % 4 != 0 else []

    # Infer voids from play history
    def pips_to_id(h, l):
        hi, lo = max(h, l), min(h, l)
        return hi * (hi + 1) // 2 + lo

    voids = {p: set() for p in range(4)}  # player -> set of void suits
    for t_idx, trick in enumerate(tricks):
        if len(trick) < 2:
            continue
        lead_player, lead_h, lead_l = trick[0]
        lead_id = pips_to_id(lead_h, lead_l)
        led_suit = led_suit_for_lead_domino(lead_id, decl_id)

        for player, h, l in trick[1:]:
            dom_id = pips_to_id(h, l)
            if not can_follow(dom_id, led_suit, decl_id):
                voids[player].add(led_suit)

    # Setup figure - MASSIVE
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(24, 16))
    fig.patch.set_facecolor('#0a0a12')

    # Create grid: 4 rows, 4 columns
    gs = fig.add_gridspec(4, 4, height_ratios=[0.8, 2, 2, 1.5],
                          width_ratios=[1.2, 1.5, 1, 1],
                          hspace=0.25, wspace=0.2)

    # === ROW 0: HEADER ===
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_facecolor('#0a0a12')
    ax_header.axis('off')

    exp_modes = {0: "GREEDY", 1: "BOLTZMANN", 2: "EPSILON", 3: "BLUNDER"}
    exp_str = exp_modes.get(current_exp_mode, "?") if current_exp_mode is not None else "?"

    ax_header.text(0.5, 0.7,
        f"GAME {game + 1}  •  DECISION {decision + 1}/28  •  TRICK {trick_num}/7",
        ha='center', va='center', fontsize=28, color='white', weight='bold',
        family='monospace')
    ax_header.text(0.5, 0.25,
        f"Declaration: {DECL_NAMES[decl_id]}  |  Action: {exp_str}  |  Index: {idx}",
        ha='center', va='center', fontsize=16, color='#4a9eff', family='monospace')

    # === ROW 1, COL 0: MY HAND ===
    ax_hand = fig.add_subplot(gs[1, 0])
    ax_hand.set_facecolor('#0a0a12')
    ax_hand.set_xlim(-0.3, max(len(hand), 1) * 1.4 + 0.3)
    ax_hand.set_ylim(-0.8, 3)
    ax_hand.set_aspect('equal')
    ax_hand.axis('off')
    ax_hand.set_title('MY HAND', fontsize=14, color='#4a9eff', pad=10, weight='bold')

    for i, (high, low) in enumerate(hand):
        is_selected = (i == action)
        is_legal = mask[i].item() if i < len(mask) else False
        alpha = 1.0 if is_legal else 0.3
        draw_domino(ax_hand, i * 1.4, 0.5, high, low, highlight=is_selected, alpha=alpha)
        if is_selected:
            ax_hand.text(i * 1.4 + 0.5, -0.5, "▲ PLAYED", ha='center',
                        fontsize=9, color='#ffcc00', weight='bold')

    # === ROW 1, COL 1: E[Q] BAR CHART ===
    ax_eq = fig.add_subplot(gs[1, 1])
    ax_eq.set_facecolor('#0a0a12')

    legal_indices = [i for i in range(len(hand)) if mask[i].item()]
    sorted_legal = sorted(legal_indices, key=lambda i: eq[i].item(), reverse=True)

    if sorted_legal:
        cmap = LinearSegmentedColormap.from_list('eq', ['#ff4444', '#666666', '#44ff44'])
        eq_vals = [eq[i].item() for i in sorted_legal]
        eq_min, eq_max = min(eq_vals), max(eq_vals)
        eq_range = eq_max - eq_min if eq_max != eq_min else 1

        y_pos = np.arange(len(sorted_legal))
        colors = [cmap((eq[i].item() - eq_min) / eq_range) for i in sorted_legal]

        bars = ax_eq.barh(y_pos, eq_vals, color=colors, edgecolor='white', linewidth=0.5, height=0.7)

        if var is not None:
            stds = [np.sqrt(max(0, var[i].item())) for i in sorted_legal]
            ax_eq.errorbar(eq_vals, y_pos, xerr=stds, fmt='none',
                          color='white', alpha=0.6, capsize=4, capthick=1.5)

        labels = [f"[{hand[i][0]}:{hand[i][1]}]" for i in sorted_legal]
        ax_eq.set_yticks(y_pos)
        ax_eq.set_yticklabels(labels, fontsize=11, color='white', family='monospace')
        ax_eq.axvline(0, color='white', linestyle='--', alpha=0.3)

        for i, idx_action in enumerate(sorted_legal):
            val = eq[idx_action].item()
            if idx_action == action:
                bars[i].set_edgecolor('#ffcc00')
                bars[i].set_linewidth(3)
            ax_eq.text(val + 0.3, i, f'{val:+.1f}', va='center', fontsize=9,
                      color='#ffcc00' if idx_action == action else 'white')

    ax_eq.set_xlabel('E[Q] (expected points)', fontsize=11, color='white')
    ax_eq.set_title('ACTION VALUES', fontsize=14, color='#4a9eff', pad=10, weight='bold')
    ax_eq.tick_params(colors='white')
    ax_eq.spines['bottom'].set_color('#333')
    ax_eq.spines['left'].set_color('#333')

    # === ROW 1, COL 2: VARIANCE BREAKDOWN ===
    ax_var = fig.add_subplot(gs[1, 2])
    ax_var.set_facecolor('#0a0a12')

    if var is not None and sorted_legal:
        var_vals = [np.sqrt(max(0, var[i].item())) for i in sorted_legal]
        y_pos = np.arange(len(sorted_legal))

        var_colors = plt.cm.Oranges(np.array(var_vals) / (max(var_vals) + 1e-8) * 0.7 + 0.3)
        ax_var.barh(y_pos, var_vals, color=var_colors, edgecolor='white', linewidth=0.5, height=0.7)

        labels = [f"[{hand[i][0]}:{hand[i][1]}]" for i in sorted_legal]
        ax_var.set_yticks(y_pos)
        ax_var.set_yticklabels(labels, fontsize=11, color='white', family='monospace')
        ax_var.set_xlabel('σ (uncertainty pts)', fontsize=11, color='white')

        for i, v in enumerate(var_vals):
            ax_var.text(v + 0.1, i, f'±{v:.1f}', va='center', fontsize=9, color='white')
    else:
        ax_var.text(0.5, 0.5, 'No variance\ndata', ha='center', va='center',
                   fontsize=14, color='#666', transform=ax_var.transAxes)

    ax_var.set_title('UNCERTAINTY', fontsize=14, color='#ff9944', pad=10, weight='bold')
    ax_var.tick_params(colors='white')
    ax_var.spines['bottom'].set_color('#333')
    ax_var.spines['left'].set_color('#333')

    # === ROW 1, COL 3: ESS GAUGE ===
    ax_ess = fig.add_subplot(gs[1, 3])
    ax_ess.set_facecolor('#0a0a12')
    ax_ess.axis('off')
    ax_ess.set_title('POSTERIOR HEALTH', fontsize=14, color='#44ff44', pad=10, weight='bold')

    if current_ess is not None:
        # Draw a semicircle gauge
        theta = np.linspace(0, np.pi, 100)
        r_outer = 1.0
        r_inner = 0.6

        # Background arc
        ax_ess.fill_between(np.cos(theta) * r_outer, np.sin(theta) * r_outer,
                            np.cos(theta) * r_inner, np.sin(theta) * r_inner,
                            color='#1a1a2e', alpha=0.8)

        # ESS needle position (0-200 scale)
        ess_norm = min(current_ess / 200, 1.0)
        needle_angle = np.pi * (1 - ess_norm)

        # Color zones
        for start, end, color in [(0, 0.25, '#ff4444'), (0.25, 0.5, '#ffaa00'), (0.5, 1.0, '#44ff44')]:
            t = np.linspace(np.pi * (1 - end), np.pi * (1 - start), 50)
            ax_ess.fill_between(np.cos(t) * r_outer, np.sin(t) * r_outer,
                               np.cos(t) * r_inner, np.sin(t) * r_inner,
                               color=color, alpha=0.3)

        # Needle
        ax_ess.plot([0, np.cos(needle_angle) * 0.9], [0, np.sin(needle_angle) * 0.9],
                   color='white', linewidth=3)
        ax_ess.scatter([0], [0], color='white', s=100, zorder=10)

        ax_ess.set_xlim(-1.3, 1.3)
        ax_ess.set_ylim(-0.3, 1.3)

        ess_color = '#ff4444' if current_ess < 50 else ('#ffaa00' if current_ess < 100 else '#44ff44')
        ax_ess.text(0, -0.15, f'ESS: {current_ess:.0f}', ha='center', fontsize=16,
                   color=ess_color, weight='bold')

        if current_max_w is not None:
            ax_ess.text(0, -0.35, f'max_w: {current_max_w:.3f}', ha='center',
                       fontsize=11, color='#888')
    else:
        ax_ess.text(0.5, 0.5, 'No ESS\ndata', ha='center', va='center',
                   fontsize=14, color='#666', transform=ax_ess.transAxes)

    # === ROW 2, COL 0: CURRENT TRICK ===
    ax_trick = fig.add_subplot(gs[2, 0])
    ax_trick.set_facecolor('#0a0a12')
    ax_trick.set_xlim(-0.3, 7.5)
    ax_trick.set_ylim(-0.5, 3.2)
    ax_trick.set_aspect('equal')
    ax_trick.axis('off')
    ax_trick.set_title('CURRENT TRICK', fontsize=14, color='#4a9eff', pad=10, weight='bold')

    if current_trick:
        for i, (player, high, low) in enumerate(current_trick):
            draw_domino(ax_trick, i * 1.8, 0.5, high, low)
            ax_trick.text(i * 1.8 + 0.5, 2.8, PLAYER_NAMES[player],
                         ha='center', fontsize=10, color='#4a9eff')
        # Show pending slot
        ax_trick.text(len(current_trick) * 1.8 + 0.5, 1.5, "?",
                     ha='center', fontsize=28, color='#ffcc00')
        ax_trick.text(len(current_trick) * 1.8 + 0.5, 2.8, "ME",
                     ha='center', fontsize=10, color='#ffcc00')
    else:
        ax_trick.text(2, 1.5, "YOUR LEAD!", ha='center', fontsize=18, color='#44ff44')

    # === ROW 2, COL 1: TRICK HISTORY ===
    ax_history = fig.add_subplot(gs[2, 1])
    ax_history.set_facecolor('#0a0a12')
    ax_history.axis('off')
    ax_history.set_title('TRICK HISTORY', fontsize=14, color='#4a9eff', pad=10, weight='bold')

    complete_tricks = [t for t in tricks if len(t) == 4]
    if complete_tricks:
        for t_idx, trick in enumerate(complete_tricks[:6]):  # Show up to 6 tricks
            y_base = 0.85 - t_idx * 0.14
            trick_str = " → ".join(f"{PLAYER_NAMES[p][0]}[{h}:{l}]" for p, h, l in trick)
            ax_history.text(0.05, y_base, f"T{t_idx+1}: {trick_str}",
                           fontsize=10, color='white', family='monospace',
                           transform=ax_history.transAxes)
    else:
        ax_history.text(0.5, 0.5, "No tricks\ncompleted", ha='center', va='center',
                       fontsize=14, color='#666', transform=ax_history.transAxes)

    # === ROW 2, COL 2-3: VOID HEATMAP ===
    ax_voids = fig.add_subplot(gs[2, 2:])
    ax_voids.set_facecolor('#0a0a12')
    ax_voids.set_title('VOID INFERENCE MAP', fontsize=14, color='#ff6644', pad=10, weight='bold')

    # Create void matrix: 4 players × 8 suits
    suit_names = ['0s', '1s', '2s', '3s', '4s', '5s', '6s', 'T']
    void_matrix = np.zeros((4, 8))
    for player in range(4):
        for suit in voids[player]:
            if suit < 8:
                void_matrix[player, suit] = 1

    im = ax_voids.imshow(void_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    ax_voids.set_xticks(range(8))
    ax_voids.set_xticklabels(suit_names, fontsize=10, color='white')
    ax_voids.set_yticks(range(4))
    ax_voids.set_yticklabels(PLAYER_NAMES, fontsize=10, color='white')
    ax_voids.tick_params(colors='white')

    # Add text annotations
    for i in range(4):
        for j in range(8):
            color = 'white' if void_matrix[i, j] == 0 else 'black'
            text = '✓' if void_matrix[i, j] == 0 else 'VOID'
            ax_voids.text(j, i, text, ha='center', va='center', fontsize=8, color=color)

    # === ROW 3: TOKEN SEQUENCE & STATS ===
    ax_tokens = fig.add_subplot(gs[3, :2])
    ax_tokens.set_facecolor('#0a0a12')
    ax_tokens.set_title('TOKEN SEQUENCE (model input)', fontsize=14, color='#9966ff', pad=10, weight='bold')

    # Visualize token sequence as colored bars
    token_types = tok[:length, FEAT_TOKEN_TYPE].numpy()
    token_colors = {0: '#666666', 1: '#4a9eff', 2: '#ff9944'}  # decl, hand, play

    for i in range(min(length, 50)):  # Show up to 50 tokens
        tt = int(token_types[i])
        color = token_colors.get(tt, '#333333')
        ax_tokens.bar(i, 1, color=color, edgecolor='none', width=0.8)

    ax_tokens.set_xlim(-0.5, min(length, 50) + 0.5)
    ax_tokens.set_ylim(0, 1.2)
    ax_tokens.set_xlabel('Token position', fontsize=11, color='white')
    ax_tokens.tick_params(colors='white', labelleft=False)
    ax_tokens.spines['bottom'].set_color('#333')
    ax_tokens.spines['left'].set_color('#333')

    # Legend for token types
    ax_tokens.text(0.02, 0.95, '■ Decl  ■ Hand  ■ Play', transform=ax_tokens.transAxes,
                  fontsize=10, color='white', va='top')

    # === STATS PANEL ===
    ax_stats = fig.add_subplot(gs[3, 2:])
    ax_stats.set_facecolor('#0a0a12')
    ax_stats.axis('off')
    ax_stats.set_title('STATISTICS', fontsize=14, color='#ffcc00', pad=10, weight='bold')

    eq_vals = [eq[i].item() for i in legal_indices] if legal_indices else [0]
    eq_max, eq_min = max(eq_vals), min(eq_vals)

    stats_lines = [
        f"Legal actions:     {len(legal_indices)}",
        f"Best E[Q]:         {eq_max:+.2f} pts",
        f"Worst E[Q]:        {eq_min:+.2f} pts",
        f"Q-gap (best-2nd):  {current_q_gap:.2f} pts" if current_q_gap else "Q-gap: N/A",
        f"Total gap:         {eq_max - eq_min:.2f} pts",
        "",
        f"ESS:               {current_ess:.1f}" if current_ess else "ESS: N/A",
        f"Max weight:        {current_max_w:.4f}" if current_max_w else "Max weight: N/A",
        f"Plays so far:      {len(plays)}",
        f"Sequence length:   {length} tokens",
    ]

    stats_text = "\n".join(stats_lines)
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=12, color='white', va='top', family='monospace',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e',
                           edgecolor='#4a9eff', linewidth=2))

    # Add watermark
    fig.text(0.99, 0.01, 'E[Q] Stage 2 • Texas 42 AI', ha='right', va='bottom',
            fontsize=10, color='#333', style='italic')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0a12')
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def create_game_playthrough(data, game_id, output_dir, fps=2, player_only=True):
    """Create full game playthrough animation with data overload frames.

    Args:
        player_only: If True, show only player 0's 7 decisions (one per trick).
                     If False, show all 28 decisions from all players.
    """
    from PIL import Image
    import io

    game_idx_arr = data['game_idx']
    game_mask = (game_idx_arr == game_id)
    all_indices = torch.where(game_mask)[0].numpy()

    if len(all_indices) == 0:
        print(f"No data for game {game_id}")
        return None

    tokens = data['transcript_tokens']
    lengths = data['transcript_lengths']

    # Filter to just player 0's decisions (same absolute player throughout)
    if player_only:
        from forge.oracle.tables import resolve_trick

        def pips_to_id(h, l):
            hi, lo = max(h, l), min(h, l)
            return hi * (hi + 1) // 2 + lo

        def get_absolute_seat(plays, decl_id):
            """Determine which absolute seat (0-3) is making this decision."""
            n_plays = len(plays)
            trick_num = n_plays // 4
            position_in_trick = n_plays % 4

            if trick_num == 0:
                # First trick: seat = position in trick
                return position_in_trick

            # Track who leads each trick
            leader = 0  # Trick 1 leader is seat 0 by definition
            for t in range(trick_num):
                trick_plays = plays[t*4 : t*4+4]
                if len(trick_plays) < 4:
                    break

                domino_ids = tuple(pips_to_id(h, l) for _, h, l in trick_plays)
                leader_domino = domino_ids[0]

                outcome = resolve_trick(leader_domino, domino_ids, decl_id)
                # Winner's absolute seat
                leader = (leader + outcome.winner_offset) % 4

            return (leader + position_in_trick) % 4

        # Find all decisions made by absolute seat 0
        indices = []
        for idx in all_indices:
            tok = tokens[idx]
            length = lengths[idx].item()
            state = decode_decision(tok, length)

            abs_seat = get_absolute_seat(state['plays'], state['decl_id'])
            if abs_seat == 0:
                indices.append(idx)

        indices = np.array(indices[:7])  # Should be exactly 7
        print(f"Game {game_id}: showing player 0's 7 decisions")

        # Get P0's initial hand from the first decision (should have 7 dominoes)
        first_idx = indices[0]
        first_state = decode_decision(tokens[first_idx], lengths[first_idx].item())
        p0_initial_hand = first_state['hand']
        assert len(p0_initial_hand) == 7, f"First P0 decision should have 7 dominoes, got {len(p0_initial_hand)}"
    else:
        indices = all_indices
        p0_initial_hand = None  # Not applicable for all-players mode
        print(f"Game {game_id}: {len(indices)} decisions (all players)")
    print("Rendering frames...")

    frames = []
    for i, idx in enumerate(indices):
        # Render frame to memory
        fig = render_data_overload_frame(data, idx, initial_hand=p0_initial_hand)

        # Convert to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0a0a12')
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img.copy())
        buf.close()
        plt.close(fig)

        print(f"  Frame {i+1}/{len(indices)}", end='\r')

    print(f"\n  Generated {len(frames)} frames")

    # Save as GIF
    output_path = output_dir / f"game_{game_id}_playthrough.gif"
    print(f"Saving to {output_path}...")

    # Calculate duration in ms
    duration_ms = 1000 // fps

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.2f} MB)")

    return output_path


def get_all_initial_hands(data, game_id):
    """Get all 4 players' initial 7-domino hands for a game.

    Returns dict mapping player_position (0-3) -> initial hand.
    Player position is inferred from n_plays % 4 at each decision.
    """
    game_idx = data['game_idx']
    tokens = data['transcript_tokens']
    lengths = data['transcript_lengths']

    initial_hands = {}

    # Find first 7-domino decision for each player position
    for i in range((game_idx == game_id).sum().item()):
        idx = (game_idx == game_id).nonzero(as_tuple=True)[0][i].item()
        tok = tokens[idx]
        length = lengths[idx].item()
        state = decode_decision(tok, length)
        hand = state['hand']
        plays = state['plays']

        # Player position in the current trick (0=lead, 1=2nd, etc.)
        position = len(plays) % 4

        # Store first 7-domino hand we see for each position
        if len(hand) == 7 and position not in initial_hands:
            initial_hands[position] = hand

        if len(initial_hands) == 4:
            break

    return initial_hands


def find_matching_initial_hand(hand, initial_hands):
    """Find which player's initial hand matches the current hand.

    Returns (player_position, initial_hand) or (None, None) if no match.
    """
    hand_set = set(hand)

    for player_pos, init_hand in initial_hands.items():
        init_set = set(init_hand)
        # Current hand should be a subset of initial hand
        if hand_set.issubset(init_set):
            return player_pos, init_hand

    return None, None


def render_data_overload_frame(data, idx, initial_hand=None):
    """Render data overload frame and return figure (for animation).

    Args:
        data: Dataset dict with transcript_tokens, e_q_mean, etc.
        idx: Index of the decision to render.
        initial_hand: Optional 7-domino hand for fixed slot positions.
                      If provided, dominoes stay in their original slots.
                      If None, falls back to dynamic hand display.
    """
    from forge.oracle.tables import can_follow, led_suit_for_lead_domino

    tokens = data['transcript_tokens']
    lengths = data['transcript_lengths']
    e_q_mean = data['e_q_mean']
    e_q_var = data.get('e_q_var')
    legal_mask = data['legal_mask']
    action_taken = data['action_taken']
    game_idx = data['game_idx']
    decision_idx = data['decision_idx']
    ess_arr = data.get('ess')
    max_w_arr = data.get('max_w')
    exploration_mode = data.get('exploration_mode')
    q_gap_arr = data.get('q_gap')
    actual_outcome_arr = data.get('actual_outcome')

    tok = tokens[idx]
    length = lengths[idx].item()
    state = decode_decision(tok, length)

    eq = e_q_mean[idx]
    var = e_q_var[idx] if e_q_var is not None else None
    mask = legal_mask[idx]
    action = action_taken[idx].item()
    game = game_idx[idx].item()
    decision = decision_idx[idx].item()
    current_ess = ess_arr[idx].item() if ess_arr is not None else None
    current_max_w = max_w_arr[idx].item() if max_w_arr is not None else None
    current_exp_mode = exploration_mode[idx].item() if exploration_mode is not None else None
    current_q_gap = q_gap_arr[idx].item() if q_gap_arr is not None else None
    actual_outcome = actual_outcome_arr[idx].item() if actual_outcome_arr is not None else None

    hand = state['hand']
    plays = state['plays']
    decl_id = state['decl_id']

    # Use provided initial_hand for fixed positions, otherwise fall back to current hand
    if initial_hand is None:
        initial_hand = hand  # No fixed positions, just show current hand

    # Create mapping: original slot -> (domino, still_in_hand, original_slot)
    # NOTE: mask is indexed by original slot, not current hand position!
    hand_set = set(hand)  # Current remaining dominoes
    slot_info = []
    for orig_slot, domino in enumerate(initial_hand):
        if domino in hand_set:
            slot_info.append((domino, True, orig_slot))
        else:
            # Domino was played
            slot_info.append((domino, False, orig_slot))

    # Group plays into tricks
    tricks = [plays[i:i+4] for i in range(0, len(plays), 4)]
    n_complete = sum(1 for t in tricks if len(t) == 4)
    trick_num = n_complete + 1
    current_trick = plays[-(len(plays) % 4):] if len(plays) % 4 != 0 else []

    # Infer voids
    def pips_to_id(h, l):
        hi, lo = max(h, l), min(h, l)
        return hi * (hi + 1) // 2 + lo

    voids = {p: set() for p in range(4)}
    for t_idx, trick in enumerate(tricks):
        if len(trick) < 2:
            continue
        lead_player, lead_h, lead_l = trick[0]
        lead_id = pips_to_id(lead_h, lead_l)
        led_suit = led_suit_for_lead_domino(lead_id, decl_id)
        for player, h, l in trick[1:]:
            dom_id = pips_to_id(h, l)
            if not can_follow(dom_id, led_suit, decl_id):
                voids[player].add(led_suit)

    # Setup figure - NEW LAYOUT: 3 columns (left=context, center=hand+violins, right=metrics)
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('#0a0a12')

    # Grid: 3 rows x 3 columns
    # Row 0: Header (spans all)
    # Row 1: Current Trick | Action Values (violins) | Uncertainty + ESS
    # Row 2: History       | My Hand (dominoes)      | Void Map + Stats
    gs = fig.add_gridspec(3, 3, height_ratios=[0.4, 2.5, 2],
                          width_ratios=[1, 2.5, 1.5],
                          hspace=0.15, wspace=0.15)

    # Map decoded hand to original mask positions
    legal_original_positions = [i for i in range(7) if mask[i].item()]
    orig_to_decoded = {orig: dec for dec, orig in enumerate(legal_original_positions)}

    # === HEADER (row 0, spans all) ===
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_facecolor('#0a0a12')
    ax_header.axis('off')

    exp_modes = {0: "GREEDY", 1: "BOLTZMANN", 2: "EPSILON", 3: "BLUNDER"}
    exp_str = exp_modes.get(current_exp_mode, "?") if current_exp_mode is not None else "?"

    ax_header.text(0.5, 0.65,
        f"GAME {game + 1}  •  DECISION {decision + 1}/28  •  TRICK {trick_num}/7",
        ha='center', va='center', fontsize=22, color='white', weight='bold',
        family='monospace')
    ax_header.text(0.5, 0.15,
        f"Declaration: {DECL_NAMES[decl_id]}  |  Action: {exp_str}",
        ha='center', va='center', fontsize=13, color='#4a9eff', family='monospace')

    # === LEFT COLUMN ===

    # --- CURRENT TRICK (row 1, col 0) ---
    ax_trick = fig.add_subplot(gs[1, 0])
    ax_trick.set_facecolor('#0a0a12')
    ax_trick.axis('off')
    ax_trick.set_title('CURRENT TRICK', fontsize=11, color='#4a9eff', pad=6, weight='bold')

    # Vertical layout: 4 slots stacked (domino 0.6w x 1.2h = 1:2 ratio)
    dom_tw, dom_th = 0.6, 1.2
    trick_spacing = 1.5
    ax_trick.set_xlim(-0.2, 2.2)
    ax_trick.set_ylim(-0.3, 4 * trick_spacing + 0.3)
    ax_trick.set_aspect('equal')

    n_played = len(current_trick)
    for slot in range(4):
        y = (3 - slot) * trick_spacing
        if slot < n_played:
            player, high, low = current_trick[slot]
            draw_domino(ax_trick, 0.2, y, high, low, width=dom_tw, height=dom_th)
            ax_trick.text(1.1, y + dom_th/2, PLAYER_NAMES[player], ha='left', va='center',
                         fontsize=9, color='#4a9eff')
        elif slot == n_played:
            # Next to play (ME)
            ax_trick.text(0.2 + dom_tw/2, y + dom_th/2, "?", ha='center', va='center',
                         fontsize=20, color='#ffcc00', weight='bold')
            ax_trick.text(1.1, y + dom_th/2, "ME", ha='left', va='center',
                         fontsize=9, color='#ffcc00', weight='bold')
        else:
            # Empty slot
            ax_trick.text(0.2 + dom_tw/2, y + dom_th/2, "·", ha='center', va='center',
                         fontsize=16, color='#333')

    if n_played == 0:
        ax_trick.text(0.5 + dom_tw/2, 2.5, "YOUR\nLEAD!", ha='center', va='center',
                     fontsize=12, color='#44ff44', weight='bold')

    # --- HISTORY (row 2, col 0) ---
    ax_history = fig.add_subplot(gs[2, 0])
    ax_history.set_facecolor('#0a0a12')
    ax_history.axis('off')
    ax_history.set_title('HISTORY', fontsize=11, color='#4a9eff', pad=6, weight='bold')

    complete_tricks = [t for t in tricks if len(t) == 4]
    # Mini dominoes with proper 1:2 aspect ratio
    dom_hw, dom_hh = 0.3, 0.6
    hist_spacing_x = 0.38
    hist_spacing_y = 0.75

    ax_history.set_xlim(-0.5, 4 * hist_spacing_x + dom_hw + 0.1)
    ax_history.set_ylim(-0.2, 6 * hist_spacing_y + dom_hh + 0.1)
    ax_history.set_aspect('equal')

    for t_idx in range(6):
        y_base = (5 - t_idx) * hist_spacing_y
        has_trick = t_idx < len(complete_tricks)
        label_color = '#4a9eff' if has_trick else '#333'
        ax_history.text(-0.25, y_base + dom_hh/2, f"T{t_idx+1}",
                       fontsize=7, color=label_color, va='center', ha='right')
        if has_trick:
            trick = complete_tricks[t_idx]
            for d_idx, (player, high, low) in enumerate(trick):
                x = d_idx * hist_spacing_x
                draw_domino(ax_history, x, y_base, high, low,
                           width=dom_hw, height=dom_hh,
                           pip_color='#aaa' if player != 0 else 'white')

    # === CENTER COLUMN (the star!) ===

    # --- ACTION VALUES with VERTICAL VIOLINS (row 1, col 1) ---
    ax_eq = fig.add_subplot(gs[1, 1])
    ax_eq.set_facecolor('#0a0a12')
    ax_eq.set_title('ACTION VALUES', fontsize=11, color='#4a9eff', pad=6, weight='bold')

    # Fixed 7 slots for dominoes, violins above each
    n_slots = 7
    dom_width = 1.0
    dom_spacing = 1.4
    total_width = n_slots * dom_spacing

    ax_eq.set_xlim(-0.5, total_width - 0.4)
    ax_eq.set_ylim(-42, 42)  # E[Q] range

    # Draw zero line
    ax_eq.axhline(0, color='white', linestyle='--', alpha=0.3, linewidth=1)

    # Y-axis labels
    ax_eq.set_ylabel('E[Q] (pts)', fontsize=10, color='white')
    ax_eq.tick_params(colors='white', labelsize=9)
    ax_eq.set_yticks([-40, -20, 0, 20, 40])
    ax_eq.set_xticks([])  # No x ticks, dominoes serve as labels

    # Color map for E[Q]
    cmap = LinearSegmentedColormap.from_list('eq', ['#ff4444', '#666666', '#44ff44'])

    # Number of dominoes remaining in hand
    n_remaining = len(hand)

    # Draw 7 FIXED slots: violins only for LEGAL dominoes that are still in hand
    for slot in range(n_slots):
        x_center = slot * dom_spacing + dom_width / 2

        domino, still_in_hand, orig_slot = slot_info[slot]

        if still_in_hand:
            is_legal = mask[orig_slot].item()
            is_selected = (orig_slot == action)

            # Only draw violin for legal actions
            if is_legal:
                mean_val = eq[orig_slot].item()
                std_val = np.sqrt(max(0.1, var[orig_slot].item())) if var is not None else 3.0

                # Gaussian curve (vertical violin)
                y_range = np.linspace(mean_val - 3*std_val, mean_val + 3*std_val, 100)
                gaussian = np.exp(-0.5 * ((y_range - mean_val) / std_val) ** 2)
                gaussian = gaussian / gaussian.max() * 0.5  # Scale width

                # Color based on mean E[Q]
                color = cmap((mean_val + 42) / 84)  # Normalize to 0-1

                # Draw filled violin
                ax_eq.fill_betweenx(y_range, x_center - gaussian, x_center + gaussian,
                                   color=color, alpha=0.7, edgecolor='none')

                # White outline
                ax_eq.plot(x_center + gaussian, y_range, color='white', linewidth=0.5, alpha=0.5)
                ax_eq.plot(x_center - gaussian, y_range, color='white', linewidth=0.5, alpha=0.5)

                # Mean line
                ax_eq.plot([x_center - 0.4, x_center + 0.4], [mean_val, mean_val],
                          color='white', linewidth=2)

                # Highlight selected action
                if is_selected:
                    ax_eq.fill_betweenx(y_range, x_center - gaussian, x_center + gaussian,
                                       color='none', edgecolor='#ffcc00', linewidth=2.5)
                    ax_eq.plot([x_center - 0.45, x_center + 0.45], [mean_val, mean_val],
                              color='#ffcc00', linewidth=3)
        # else: slot is ghost (domino was played) - no violin

    # --- MY HAND (row 2, col 1) ---
    ax_hand = fig.add_subplot(gs[2, 1])
    ax_hand.set_facecolor('#0a0a12')
    ax_hand.axis('off')
    ax_hand.set_title('MY HAND', fontsize=11, color='#4a9eff', pad=6, weight='bold')

    # Match spacing with violin plot above
    ax_hand.set_xlim(-0.5, total_width - 0.4)
    ax_hand.set_ylim(-1, 3.5)

    # Draw 7 FIXED slots - dominoes stay in their original positions
    for slot in range(n_slots):
        x = slot * dom_spacing

        domino, still_in_hand, orig_slot = slot_info[slot]
        high, low = domino

        if still_in_hand:
            # Domino still in hand - draw it in its original position
            is_legal = mask[orig_slot].item()
            is_selected = (orig_slot == action)
            draw_domino(ax_hand, x, 0.5, high, low, width=dom_width, height=2.0,
                       highlight=is_selected, alpha=1.0 if is_legal else 0.4)
            if is_selected:
                ax_hand.text(x + dom_width/2, -0.3, "▲ PLAYED", ha='center',
                            fontsize=9, color='#ffcc00', weight='bold')
        else:
            # Domino was played - draw ghost slot
            ghost = mpatches.FancyBboxPatch(
                (x, 0.5), dom_width, 2.0,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor='#0a0a12', edgecolor='#333', linewidth=1,
                linestyle='--', alpha=0.5
            )
            ax_hand.add_patch(ghost)

    # === RIGHT COLUMN (3x3 grid) ===
    gs_right = gs[1:, 2].subgridspec(3, 3, hspace=0.3, wspace=0.2)

    # Compute derived metrics
    legal_indices = [i for i in range(n_remaining) if mask[i].item()]
    eq_vals = [eq[i].item() for i in legal_indices] if legal_indices else [0]
    eq_max, eq_min = max(eq_vals), min(eq_vals)

    # Confidence: softmax probability of best action
    if legal_indices:
        eq_legal = torch.tensor([eq[i].item() for i in legal_indices])
        probs = torch.nn.functional.softmax(eq_legal, dim=0)
        confidence = probs.max().item() * 100  # percentage
    else:
        confidence = 0

    # Hand strength: sum of pips
    hand_strength = sum(h + l for h, l in hand) if hand else 0

    # Position in trick (0=lead, 1=2nd, 2=3rd, 3=4th)
    position_in_trick = len(plays) % 4

    # --- ROW 0: UNCERTAINTY, ESS, Q-GAP ---

    # UNCERTAINTY
    ax_var = fig.add_subplot(gs_right[0, 0])
    ax_var.set_facecolor('#0a0a12')
    ax_var.set_title('UNCERTAINTY', fontsize=9, color='#ff9944', pad=3, weight='bold')
    if var is not None and legal_indices:
        sorted_legal = sorted(legal_indices, key=lambda i: eq[i].item(), reverse=True)
        var_vals = [np.sqrt(max(0, var[i].item())) for i in sorted_legal]
        y_pos = np.arange(len(sorted_legal))
        var_colors = plt.cm.Oranges(np.array(var_vals) / (max(var_vals) + 1e-8) * 0.7 + 0.3)
        ax_var.barh(y_pos, var_vals, color=var_colors, edgecolor='white', linewidth=0.5, height=0.6)
        ax_var.set_yticks([])
    ax_var.set_xlim(0, 30)
    ax_var.tick_params(colors='white', labelsize=7)

    # ESS GAUGE
    ax_ess = fig.add_subplot(gs_right[0, 1])
    ax_ess.set_facecolor('#0a0a12')
    ax_ess.axis('off')
    ax_ess.set_title('ESS', fontsize=9, color='#44ff44', pad=3, weight='bold')
    if current_ess is not None:
        theta = np.linspace(0, np.pi, 100)
        r_outer, r_inner = 1.0, 0.65
        ax_ess.fill_between(np.cos(theta) * r_outer, np.sin(theta) * r_outer,
                            np.cos(theta) * r_inner, np.sin(theta) * r_inner,
                            color='#1a1a2e', alpha=0.8)
        ess_norm = min(current_ess / 200, 1.0)
        needle_angle = np.pi * (1 - ess_norm)
        for start, end, color in [(0, 0.25, '#ff4444'), (0.25, 0.5, '#ffaa00'), (0.5, 1.0, '#44ff44')]:
            t = np.linspace(np.pi * (1 - end), np.pi * (1 - start), 50)
            ax_ess.fill_between(np.cos(t) * r_outer, np.sin(t) * r_outer,
                               np.cos(t) * r_inner, np.sin(t) * r_inner, color=color, alpha=0.3)
        ax_ess.plot([0, np.cos(needle_angle) * 0.85], [0, np.sin(needle_angle) * 0.85], color='white', linewidth=2)
        ax_ess.scatter([0], [0], color='white', s=40, zorder=10)
        ax_ess.set_xlim(-1.2, 1.2)
        ax_ess.set_ylim(-0.25, 1.2)
        ess_color = '#ff4444' if current_ess < 50 else ('#ffaa00' if current_ess < 100 else '#44ff44')
        ax_ess.text(0, -0.15, f'{current_ess:.0f}', ha='center', fontsize=10, color=ess_color, weight='bold')

    # Q-GAP GAUGE
    ax_qgap = fig.add_subplot(gs_right[0, 2])
    ax_qgap.set_facecolor('#0a0a12')
    ax_qgap.axis('off')
    ax_qgap.set_title('Q-GAP', fontsize=9, color='#9966ff', pad=3, weight='bold')
    if current_q_gap is not None:
        theta = np.linspace(0, np.pi, 100)
        r_outer, r_inner = 1.0, 0.65
        ax_qgap.fill_between(np.cos(theta) * r_outer, np.sin(theta) * r_outer,
                             np.cos(theta) * r_inner, np.sin(theta) * r_inner,
                             color='#1a1a2e', alpha=0.8)
        # Q-gap scale: 0-20 pts (big gap = clear decision)
        gap_norm = min(current_q_gap / 20, 1.0)
        needle_angle = np.pi * (1 - gap_norm)
        for start, end, color in [(0, 0.25, '#ff4444'), (0.25, 0.5, '#ffaa00'), (0.5, 1.0, '#44ff44')]:
            t = np.linspace(np.pi * (1 - end), np.pi * (1 - start), 50)
            ax_qgap.fill_between(np.cos(t) * r_outer, np.sin(t) * r_outer,
                                np.cos(t) * r_inner, np.sin(t) * r_inner, color=color, alpha=0.3)
        ax_qgap.plot([0, np.cos(needle_angle) * 0.85], [0, np.sin(needle_angle) * 0.85], color='white', linewidth=2)
        ax_qgap.scatter([0], [0], color='white', s=40, zorder=10)
        ax_qgap.set_xlim(-1.2, 1.2)
        ax_qgap.set_ylim(-0.25, 1.2)
        gap_color = '#ff4444' if current_q_gap < 2 else ('#ffaa00' if current_q_gap < 5 else '#44ff44')
        ax_qgap.text(0, -0.15, f'{current_q_gap:.1f}', ha='center', fontsize=10, color=gap_color, weight='bold')

    # --- ROW 1: VOIDS, CONFIDENCE, FINAL RESULT ---

    # VOIDS
    ax_voids = fig.add_subplot(gs_right[1, 0])
    ax_voids.set_facecolor('#0a0a12')
    ax_voids.set_title('VOIDS', fontsize=9, color='#ff6644', pad=3, weight='bold')
    suit_names = ['0', '1', '2', '3', '4', '5', '6', 'T']
    void_matrix = np.zeros((4, 8))
    for player in range(4):
        for suit in voids[player]:
            if suit < 8:
                void_matrix[player, suit] = 1
    ax_voids.imshow(void_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    ax_voids.set_xticks(range(8))
    ax_voids.set_xticklabels(suit_names, fontsize=6, color='white')
    ax_voids.set_yticks(range(4))
    ax_voids.set_yticklabels(['ME', 'LF', 'PR', 'RT'], fontsize=6, color='white')
    ax_voids.tick_params(colors='white', length=2)

    # CONFIDENCE GAUGE
    ax_conf = fig.add_subplot(gs_right[1, 1])
    ax_conf.set_facecolor('#0a0a12')
    ax_conf.axis('off')
    ax_conf.set_title('CONFIDENCE', fontsize=9, color='#00ccff', pad=3, weight='bold')
    theta = np.linspace(0, np.pi, 100)
    r_outer, r_inner = 1.0, 0.65
    ax_conf.fill_between(np.cos(theta) * r_outer, np.sin(theta) * r_outer,
                         np.cos(theta) * r_inner, np.sin(theta) * r_inner,
                         color='#1a1a2e', alpha=0.8)
    conf_norm = confidence / 100
    needle_angle = np.pi * (1 - conf_norm)
    for start, end, color in [(0, 0.33, '#ff4444'), (0.33, 0.66, '#ffaa00'), (0.66, 1.0, '#44ff44')]:
        t = np.linspace(np.pi * (1 - end), np.pi * (1 - start), 50)
        ax_conf.fill_between(np.cos(t) * r_outer, np.sin(t) * r_outer,
                            np.cos(t) * r_inner, np.sin(t) * r_inner, color=color, alpha=0.3)
    ax_conf.plot([0, np.cos(needle_angle) * 0.85], [0, np.sin(needle_angle) * 0.85], color='white', linewidth=2)
    ax_conf.scatter([0], [0], color='white', s=40, zorder=10)
    ax_conf.set_xlim(-1.2, 1.2)
    ax_conf.set_ylim(-0.25, 1.2)
    conf_color = '#ff4444' if confidence < 40 else ('#ffaa00' if confidence < 70 else '#44ff44')
    ax_conf.text(0, -0.15, f'{confidence:.0f}%', ha='center', fontsize=10, color=conf_color, weight='bold')

    # FINAL RESULT (spoiler!)
    ax_result = fig.add_subplot(gs_right[1, 2])
    ax_result.set_facecolor('#0a0a12')
    ax_result.axis('off')
    ax_result.set_title('RESULT', fontsize=9, color='#ff66cc', pad=3, weight='bold')
    if actual_outcome is not None:
        # actual_outcome is points for player 0's team (positive = win)
        if actual_outcome > 0:
            result_text = "WIN"
            result_color = '#44ff44'
        elif actual_outcome < 0:
            result_text = "LOSE"
            result_color = '#ff4444'
        else:
            result_text = "TIE"
            result_color = '#ffaa00'
        ax_result.text(0.5, 0.6, result_text, ha='center', va='center', fontsize=14,
                      color=result_color, weight='bold', transform=ax_result.transAxes)
        ax_result.text(0.5, 0.25, f'{actual_outcome:+.0f}', ha='center', va='center', fontsize=11,
                      color='white', transform=ax_result.transAxes)

    # --- ROW 2: STATS, POSITION, HAND STRENGTH ---

    # STATS
    ax_stats = fig.add_subplot(gs_right[2, 0])
    ax_stats.set_facecolor('#0a0a12')
    ax_stats.axis('off')
    ax_stats.set_title('STATS', fontsize=9, color='#ffcc00', pad=3, weight='bold')
    stats_lines = [f"Hand: {n_remaining}/7", f"Legal: {len(legal_indices)}", f"Best: {eq_max:+.1f}"]
    stats_text = "\n".join(stats_lines)
    ax_stats.text(0.1, 0.85, stats_text, transform=ax_stats.transAxes,
                  fontsize=9, color='white', va='top', family='monospace')

    # POSITION
    ax_pos = fig.add_subplot(gs_right[2, 1])
    ax_pos.set_facecolor('#0a0a12')
    ax_pos.axis('off')
    ax_pos.set_title('POSITION', fontsize=9, color='#66ffcc', pad=3, weight='bold')
    pos_labels = ['LEAD', '2ND', '3RD', '4TH']
    pos_colors = ['#44ff44', '#ffaa00', '#ff9944', '#ff6666']
    ax_pos.text(0.5, 0.55, pos_labels[position_in_trick], ha='center', va='center',
               fontsize=14, color=pos_colors[position_in_trick], weight='bold',
               transform=ax_pos.transAxes)
    # Mini dots showing position
    for i in range(4):
        color = pos_colors[position_in_trick] if i == position_in_trick else '#333'
        ax_pos.scatter([0.2 + i * 0.2], [0.2], s=60, color=color, transform=ax_pos.transAxes)

    # HAND STRENGTH
    ax_strength = fig.add_subplot(gs_right[2, 2])
    ax_strength.set_facecolor('#0a0a12')
    ax_strength.axis('off')
    ax_strength.set_title('PIPS', fontsize=9, color='#ffcc66', pad=3, weight='bold')
    # Max possible pips in 7 dominoes is around 77 (if you had all high doubles)
    # Typical range is maybe 20-50
    ax_strength.text(0.5, 0.55, f'{hand_strength}', ha='center', va='center',
                    fontsize=18, color='white', weight='bold', transform=ax_strength.transAxes)
    # Bar showing relative strength (0-60 scale)
    bar_width = min(hand_strength / 60, 1.0) * 0.8
    ax_strength.add_patch(mpatches.Rectangle((0.1, 0.15), 0.8, 0.15, facecolor='#333',
                                             transform=ax_strength.transAxes))
    strength_color = '#ff4444' if hand_strength < 20 else ('#ffaa00' if hand_strength < 35 else '#44ff44')
    ax_strength.add_patch(mpatches.Rectangle((0.1, 0.15), bar_width, 0.15, facecolor=strength_color,
                                             transform=ax_strength.transAxes))

    fig.text(0.99, 0.01, 'E[Q] Stage 2 • Texas 42 AI', ha='right', va='bottom',
            fontsize=9, color='#333', style='italic')

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Render E[Q] visualizations")
    parser.add_argument("--game", type=int, default=0, help="Game ID to visualize")
    parser.add_argument("--decision", type=int, default=100, help="Decision index for snapshot")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH), help="Dataset path")
    parser.add_argument("--skip-animations", action="store_true", help="Skip generating animations")
    parser.add_argument("--overload", action="store_true", help="Generate data overload view")
    parser.add_argument("--random", action="store_true", help="Pick a random decision")
    parser.add_argument("--playthrough", action="store_true", help="Generate full game playthrough GIF")
    parser.add_argument("--all-players", action="store_true", help="Show all 28 decisions (default: just 7 lead decisions)")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second for playthrough")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading {args.dataset}...")
    data = torch.load(args.dataset, weights_only=False)
    n_decisions = len(data['transcript_tokens'])
    print(f"Loaded {n_decisions:,} decisions")

    # Pick decision index
    decision_idx = args.decision
    if args.random:
        decision_idx = np.random.randint(0, n_decisions)
        print(f"Random decision selected: {decision_idx}")

    # Playthrough mode - animate entire game
    if args.playthrough:
        game_id = args.game
        if args.random:
            # Pick a random game
            unique_games = torch.unique(data['game_idx']).numpy()
            game_id = int(np.random.choice(unique_games))
            print(f"Random game selected: {game_id}")

        player_only = not args.all_players
        mode = "LEAD DECISIONS" if player_only else "ALL DECISIONS"
        print(f"\nGenerating {mode} PLAYTHROUGH for game {game_id}...")
        create_game_playthrough(data, game_id, output_dir, fps=args.fps, player_only=player_only)
        print("\n" + "=" * 50)
        print("Generated files:")
        for f in sorted(output_dir.glob("game_*_playthrough.gif")):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name:40} {size_mb:6.2f} MB")
        return

    # Data overload mode - just generate the massive view
    if args.overload:
        print(f"\nGenerating DATA OVERLOAD view for decision {decision_idx}...")
        render_data_overload(data, decision_idx, output_dir / f"data_overload_{decision_idx}.png")
        print("\n" + "=" * 50)
        print("Generated files:")
        for f in sorted(output_dir.glob("data_overload_*")):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name:40} {size_mb:6.2f} MB")
        return

    # Render all dominoes
    print("\n1. Rendering all dominoes...")
    render_all_dominoes(output_dir / "all_dominoes.png")

    # Render decision snapshot
    print(f"\n2. Rendering decision snapshot (idx={decision_idx})...")
    render_decision_snapshot(data, decision_idx, output_dir / "decision_snapshot.png")

    if not args.skip_animations:
        # Create trajectory animation
        print(f"\n3. Creating trajectory animation (game={args.game})...")
        create_trajectory_animation(data, args.game, output_dir / f"trajectory_game{args.game}.gif")

        # Create belief cloud animation
        print(f"\n4. Creating belief cloud animation (game={args.game})...")
        create_belief_cloud_animation(data, args.game, save_path=output_dir / f"belief_cloud_game{args.game}.gif")

    # Summary
    print("\n" + "=" * 50)
    print("Generated files:")
    for f in sorted(output_dir.glob("*")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:40} {size_mb:6.2f} MB")

    print("\n" + "=" * 50)
    print("MMS Tips:")
    print("  - GIFs under 3MB work well for MMS")
    print("  - PNGs can be sent as regular images")
    print(f"\nFiles ready at: {output_dir}")


if __name__ == "__main__":
    main()
