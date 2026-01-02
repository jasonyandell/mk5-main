"""
Generate visual bidding posters with domino tiles and P(make) heatmaps.

Usage:
    python -m forge.bidding.poster --hand "6-6,6-5,6-4,6-3,6-2,6-1,6-0" --output poster.pdf
    python -m forge.bidding.poster --hand "..." --samples 100  # More accurate
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


# Trump display order (matches evaluate.py)
TRUMP_ORDER = [
    "blanks", "ones", "twos", "threes", "fours",
    "fives", "sixes", "doubles-trump", "notrump"
]

# Bid columns
BID_LEVELS = list(range(30, 43))  # 30-42


def parse_hand(hand_str: str) -> list[tuple[int, int]]:
    """Parse hand string like '6-6,6-5,6-4' into list of (high, low) tuples."""
    dominoes = []
    for piece in hand_str.split(","):
        piece = piece.strip()
        high, low = piece.split("-")
        dominoes.append((int(high), int(low)))
    return dominoes


def get_pip_positions(n: int, size: float) -> list[tuple[float, float]]:
    """Get pip positions for a number 0-6 within a half-tile of given size.

    Returns list of (x, y) offsets from center of half-tile.
    """
    # Pip positions as fractions of half-tile size
    margin = size * 0.2
    center = size / 2
    offset = size * 0.25

    positions = {
        0: [],
        1: [(center, center)],
        2: [(center - offset, center - offset), (center + offset, center + offset)],
        3: [(center - offset, center - offset), (center, center), (center + offset, center + offset)],
        4: [(center - offset, center - offset), (center + offset, center - offset),
            (center - offset, center + offset), (center + offset, center + offset)],
        5: [(center - offset, center - offset), (center + offset, center - offset),
            (center, center),
            (center - offset, center + offset), (center + offset, center + offset)],
        6: [(center - offset, center - offset), (center + offset, center - offset),
            (center - offset, center), (center + offset, center),
            (center - offset, center + offset), (center + offset, center + offset)],
    }
    return positions.get(n, [])


def draw_domino(c: canvas.Canvas, x: float, y: float, high: int, low: int,
                tile_width: float = 0.8 * inch, tile_height: float = 0.4 * inch):
    """Draw a single domino tile at (x, y) with high|low pips.

    Domino is drawn horizontally: [high | low]
    """
    half_width = tile_width / 2
    pip_radius = tile_height * 0.08
    corner_radius = tile_height * 0.15

    # Draw tile background (rounded rectangle)
    c.setFillColor(colors.ivory)
    c.setStrokeColor(colors.black)
    c.setLineWidth(1.5)
    c.roundRect(x, y, tile_width, tile_height, corner_radius, fill=1, stroke=1)

    # Draw center dividing line
    c.setStrokeColor(colors.darkgray)
    c.setLineWidth(1)
    c.line(x + half_width, y + 2, x + half_width, y + tile_height - 2)

    # Draw pips for high side (left half)
    c.setFillColor(colors.black)
    for px, py in get_pip_positions(high, half_width):
        c.circle(x + px, y + py * (tile_height / half_width), pip_radius, fill=1, stroke=0)

    # Draw pips for low side (right half)
    for px, py in get_pip_positions(low, half_width):
        c.circle(x + half_width + px, y + py * (tile_height / half_width), pip_radius, fill=1, stroke=0)


def p_make_to_color(p: float) -> colors.Color:
    """Convert P(make) (0-1) to color on red-yellow-green scale."""
    if p < 0.5:
        # Red to Yellow (0 -> 0.5)
        t = p / 0.5
        return colors.Color(1.0, t, 0.0)
    else:
        # Yellow to Green (0.5 -> 1)
        t = (p - 0.5) / 0.5
        return colors.Color(1.0 - t, 1.0, 0.0)


def draw_heatmap(c: canvas.Canvas, x: float, y: float,
                 matrix: dict[str, dict[int, float]],
                 cell_width: float = 0.38 * inch,
                 cell_height: float = 0.28 * inch):
    """Draw P(make) heatmap table.

    matrix: {trump_name: {bid_level: p_make}}
    """
    header_height = cell_height * 1.2
    label_width = 1.2 * inch

    # Draw column headers (bid levels)
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.black)
    for i, bid in enumerate(BID_LEVELS):
        cx = x + label_width + i * cell_width + cell_width / 2
        cy = y + len(TRUMP_ORDER) * cell_height + header_height / 2 - 3
        c.drawCentredString(cx, cy, str(bid))

    # Draw rows
    for row_idx, trump in enumerate(TRUMP_ORDER):
        row_y = y + (len(TRUMP_ORDER) - 1 - row_idx) * cell_height

        # Row label
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.black)
        c.drawString(x + 5, row_y + cell_height / 2 - 3, trump)

        # Draw cells
        trump_data = matrix.get(trump, {})
        for col_idx, bid in enumerate(BID_LEVELS):
            cell_x = x + label_width + col_idx * cell_width
            p_make = trump_data.get(bid, 0)

            # Cell background color
            c.setFillColor(p_make_to_color(p_make))
            c.rect(cell_x, row_y, cell_width, cell_height, fill=1, stroke=0)

            # Cell border
            c.setStrokeColor(colors.gray)
            c.setLineWidth(0.5)
            c.rect(cell_x, row_y, cell_width, cell_height, fill=0, stroke=1)

            # Cell text
            pct = int(p_make * 100)
            if pct == 100:
                text = "100"
            elif pct == 0:
                text = "0"
            else:
                text = str(pct)

            # Use dark text on light colors, light text on dark colors
            brightness = p_make_to_color(p_make).red * 0.299 + p_make_to_color(p_make).green * 0.587
            if brightness > 0.6:
                c.setFillColor(colors.black)
            else:
                c.setFillColor(colors.white)

            c.setFont("Helvetica", 7)
            c.drawCentredString(cell_x + cell_width / 2, row_y + cell_height / 2 - 2, text)

    # Draw outer border
    total_width = label_width + len(BID_LEVELS) * cell_width
    total_height = len(TRUMP_ORDER) * cell_height
    c.setStrokeColor(colors.black)
    c.setLineWidth(1.5)
    c.rect(x + label_width, y, len(BID_LEVELS) * cell_width, total_height, fill=0, stroke=1)


def run_evaluation(hand: str, samples: int, seed: int | None = None) -> dict:
    """Run bidding evaluation and return parsed JSON."""
    cmd = [
        sys.executable, "-m", "forge.bidding.evaluate",
        "--hand", hand,
        "--samples", str(samples),
        "--json"
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Evaluation failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    return json.loads(result.stdout)


def build_matrix(eval_result: dict) -> dict[str, dict[int, float]]:
    """Convert evaluation result to matrix format for heatmap."""
    matrix = {}
    for trump_data in eval_result["trumps"]:
        trump_name = trump_data["trump"]
        matrix[trump_name] = {}
        for bid_data in trump_data["bids"]:
            matrix[trump_name][bid_data["bid"]] = bid_data["p_make"]
    return matrix


def generate_poster(hand: str, output_path: str, samples: int = 50, seed: int | None = None):
    """Generate a bidding poster PDF."""
    print(f"Evaluating hand: {hand}")
    eval_result = run_evaluation(hand, samples, seed)
    matrix = build_matrix(eval_result)

    # Parse dominoes
    dominoes = parse_hand(hand)

    # Create PDF
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 0.75 * inch, "Texas 42 Bidding Evaluation")

    # Draw dominoes centered at top
    tile_width = 0.85 * inch
    tile_height = 0.42 * inch
    tile_gap = 0.12 * inch
    total_tiles_width = len(dominoes) * tile_width + (len(dominoes) - 1) * tile_gap
    start_x = (width - total_tiles_width) / 2
    tiles_y = height - 1.5 * inch

    for i, (high, low) in enumerate(dominoes):
        x = start_x + i * (tile_width + tile_gap)
        draw_domino(c, x, tiles_y, high, low, tile_width, tile_height)

    # Hand string label
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.gray)
    c.drawCentredString(width / 2, tiles_y - 0.25 * inch, hand)

    # Section header
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.black)
    c.drawString(0.75 * inch, tiles_y - 0.7 * inch, "P(make) by Trump × Bid Level")

    # Draw heatmap
    heatmap_y = tiles_y - 3.5 * inch
    draw_heatmap(c, 0.75 * inch, heatmap_y, matrix)

    # Legend
    legend_y = heatmap_y - 0.6 * inch
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.black)
    c.drawString(0.75 * inch, legend_y, "Color scale:")

    # Draw gradient legend
    legend_width = 3 * inch
    legend_height = 0.2 * inch
    legend_x = 1.6 * inch
    for i in range(100):
        p = i / 100
        c.setFillColor(p_make_to_color(p))
        c.rect(legend_x + i * legend_width / 100, legend_y - 0.05 * inch,
               legend_width / 100 + 1, legend_height, fill=1, stroke=0)

    c.setFillColor(colors.black)
    c.drawString(legend_x - 0.3 * inch, legend_y, "0%")
    c.drawString(legend_x + legend_width + 0.1 * inch, legend_y, "100%")

    # Footer
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.gray)
    c.drawCentredString(width / 2, 0.5 * inch, f"Generated with {samples} samples per trump")

    c.save()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate bidding poster with domino tiles and P(make) heatmap"
    )
    parser.add_argument(
        "--hand", required=True,
        help='7 dominoes as "high-low" pairs, comma-separated'
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output PDF path"
    )
    parser.add_argument(
        "--samples", type=int, default=50,
        help="Simulations per trump (default 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()
    generate_poster(args.hand, args.output, args.samples, args.seed)


if __name__ == "__main__":
    main()
