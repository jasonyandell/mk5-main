# Oracle State Space Analysis

Analysis of state space characteristics across 1000 seeds (0-999) generated with `forge/cli/generate_continuous.py`.

## Summary Statistics

| Metric | Value |
|--------|-------|
| Seeds generated | 999 (1 skipped) |
| Total files | 1,089 |
| Total disk | 180 GB |
| Avg file size | ~165 MB |
| State count range | 3.2M - 115.0M |
| Skipped (>160M states) | seed 434 (~190M) |

### Split Distribution

| Split | Seeds | Files |
|-------|-------|-------|
| train | 0-899 | 899 |
| val | 900-949 | 95* |
| test | 950-999 | 95* |

*Extra files from legacy 10-decl-per-seed data for seeds 900-904, 950-954.

## By Average State Count (Descending)

| Decl | Name | Avg States | Min | Max | StdDev |
|------|------|-----------|-----|-----|--------|
| 7 | doubles-trump | **33.8M** | 5.8M | 111.6M | 28.2M |
| 2 | twos | 32.1M | 10.0M | 76.2M | 16.6M |
| 6 | sixes | 30.8M | 5.2M | 77.3M | 18.8M |
| 4 | fours | 29.2M | 3.1M | 102.2M | 23.2M |
| 0 | blanks | 27.4M | 3.2M | 73.1M | 16.1M |
| 5 | fives | 26.1M | 4.4M | 71.8M | 17.2M |
| 1 | ones | 25.6M | 5.8M | 115.0M | 18.5M |
| 9 | notrump | 24.5M | 4.0M | 102.9M | 20.7M |
| 3 | threes | 24.4M | 3.4M | 72.2M | 17.1M |
| 8 | doubles-suit | **23.1M** | 6.7M | 76.0M | 14.0M |

## By Variance (Descending)

| Decl | Name | Avg States | Min | Max | StdDev | Range |
|------|------|-----------|-----|-----|--------|-------|
| 7 | doubles-trump | 33.8M | 5.8M | 111.6M | **28.2M** | 19.3x |
| 4 | fours | 29.2M | 3.1M | 102.2M | 23.2M | 32.6x |
| 9 | notrump | 24.5M | 4.0M | 102.9M | 20.7M | 25.7x |
| 6 | sixes | 30.8M | 5.2M | 77.3M | 18.8M | 14.8x |
| 1 | ones | 25.6M | 5.8M | 115.0M | 18.5M | 19.9x |
| 5 | fives | 26.1M | 4.4M | 71.8M | 17.2M | 16.4x |
| 3 | threes | 24.4M | 3.4M | 72.2M | 17.1M | 21.2x |
| 2 | twos | 32.1M | 10.0M | 76.2M | 16.6M | 7.6x |
| 0 | blanks | 27.4M | 3.2M | 73.1M | 16.1M | 23.1x |
| 8 | doubles-suit | 23.1M | 6.7M | 76.0M | **14.0M** | 11.3x |

## By Disk Size (Descending)

| Decl | Name | Total Size | Files | Avg Size |
|------|------|-----------|-------|----------|
| 2 | twos | **23.1 GB** | 129 | 179 MB |
| 7 | doubles-trump | 22.2 GB | 130 | 171 MB |
| 6 | sixes | 22.0 GB | 130 | 169 MB |
| 5 | fives | 20.5 GB | 130 | 158 MB |
| 0 | blanks | 20.3 GB | 129 | 157 MB |
| 3 | threes | 20.2 GB | 128 | 158 MB |
| 1 | ones | 19.3 GB | 129 | 149 MB |
| 4 | fours | 18.8 GB | 127 | 148 MB |
| 8 | doubles-suit | 17.5 GB | 130 | 135 MB |
| 9 | notrump | **17.1 GB** | 130 | 131 MB |

## Key Observations

### Largest State Spaces

**Doubles-trump (decl=7)** consistently creates the largest state spaces:
- Highest average (33.8M)
- Highest variance (σ=28.2M)
- All 7 doubles are trump, creating complex branching

### Smallest/Most Predictable

**Doubles-suit (decl=8)** is the most constrained:
- Lowest average (23.1M)
- Lowest variance (σ=14.0M)
- Doubles form their own suit (not trump), may simplify branching

### Disk vs State Count Mismatch

**Twos** uses most disk despite not having highest state count:
- Highest min floor (10M states) - no small games
- Consistent large games = more total data

**Notrump** uses least disk despite mid-range state counts:
- High variance seeds balance out
- Possibly more efficient state representations

### Extreme Seeds

| Category | Seed | Decl | States |
|----------|------|------|--------|
| Largest computed | 727 | doubles-trump | 111.6M |
| Skipped (OOM) | 434 | fours | ~190M |
| Smallest | 930 | blanks | 3.2M |

### Skip Rate

With `--max-states 160000000`:
- 1 seed skipped out of 1000 (0.1%)
- Seed 434 had ~190M states (fours declaration)

## Generation Parameters

```bash
python -m forge.cli.generate_continuous \
  --output-dir /mnt/d/shards-standard \
  --max-states 160000000
```

- Device: CUDA (RTX 3050 Ti, 4GB VRAM)
- Rate: ~2-4 seeds/min (varies with state count)
- Time to 1000 seeds: ~4 hours
