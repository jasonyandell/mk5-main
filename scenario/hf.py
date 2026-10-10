"""A pour, published: the advantage dataset as a Hugging Face dataset repo.

One command turns pour directories (each a split) into a staged repo and
uploads it, so the repo on the Hub is always exactly what the arguments
say — a republish replaces every data file:

    python -u -m scenario.hf publish --repo jasonyandell/texas-42-walt-pour \\
        --split train=scratch/atlas/pour_s1001_n512_d5000 \\
        --split validation=scratch/atlas/pour_s1_n512

The staged repo (under `scratch/hf/` unless `--stage` says otherwise) holds
three Parquet configs read from the pour's own columns — `positions` (one
row per planned position: the record, the legal moves, their values and
advantages), `splats` (the same rows' per-scenario outcomes, worlds and
weights) and `deals` (each deal and how it really ended) — plus the raw
`pour.npz` + `manifest.json` per split for an exact round trip through
`scenario.atlas.load`, a dataset card, the CC0 legal code and SHA256SUMS.
Every list column is a pour column cut at the pour's own offsets, so a row
is its position's slice of the arrays, bit for bit.

Needs `pyarrow` and `huggingface_hub` (`scenario/requirements.txt`).
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scenario.atlas import N_PLAYS, Pour, calibration, load

HERE = Path(__file__).resolve().parent
SOURCE = "https://github.com/jasonyandell/mk5-main"
SHARD_ROWS = {"positions": 100_000, "splats": 5_000, "deals": 100_000}
GROUP_ROWS = {"positions": 10_000, "splats": 500, "deals": 10_000}
DECLS = ("blanks", "ones", "twos", "threes", "fours", "fives", "sixes", "doubles-trump", "doubles-suit", "notrump")


@dataclass(frozen=True)
class Split:
    name: str
    pour: Pour
    manifest: dict
    raw: Path            # the pour directory


def read_split(name: str, path: str | Path, commit: str | None = None) -> Split:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    commit = manifest.get("commit") or commit
    if commit is None:
        raise SystemExit(f"{path}/manifest.json records no commit; pass --commit")
    return Split(name, load(path), {**manifest, "commit": commit}, path)


# -- the tables ----------------------------------------------------------------------


def _lists(lengths: np.ndarray, values: pa.Array) -> pa.ListArray:
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    return pa.ListArray.from_arrays(pa.array(offsets, pa.int32()), values)


def _tiles(masks: np.ndarray) -> pa.ListArray:
    """Each tile mask as its ascending tile ids."""
    bits = (masks.astype(np.int64)[:, None] >> np.arange(N_PLAYS)) & 1 == 1
    return _lists(bits.sum(axis=1), pa.array(np.nonzero(bits)[1].astype(np.int8)))


def _seeds(p: Pour, manifest: dict) -> pa.Array:
    seeds = manifest.get("deal_seeds")
    return pa.array([None] * len(p.hands) if seeds is None else seeds, pa.int32())


def tables(p: Pour, manifest: dict) -> dict[str, pa.Table]:
    """The three tables of one pour. A position's row order is the pour's,
    and `position` is its index there (`Pour.position(i)`)."""
    d, k = p.deal.astype(np.int64), p.n_plays.astype(np.int64)
    na, nw = p.n_actions.astype(np.int64), p.n_worlds.astype(np.int64)
    seat = p.plays[d, k, 0].astype(np.int8)
    sign = np.where(seat % 2 == p.bidder % 2, 1, -1).astype(np.int8)
    before = np.arange(N_PLAYS)[None, :] < k[:, None]
    tiles = p.plays[:, :, 1].astype(np.int64)
    played = np.where(before, np.uint32(1) << tiles[d].astype(np.uint32), np.uint32(0))
    own = p.hands[d, seat] & ~np.bitwise_or.reduce(played, axis=1)
    group = np.repeat(np.arange(len(p)), na)
    mean = np.bincount(group, weights=p.q, minlength=len(p)) / na
    seeds = _seeds(p, manifest)
    key = {
        "position": pa.array(np.arange(len(p), dtype=np.int32)),
        "deal": pa.array(p.deal),
        "deal_seed": seeds.take(pa.array(d)),
        "n_plays": pa.array(p.n_plays),
    }
    history = pa.StructArray.from_arrays(
        [pa.array(p.plays[d][before][:, 0]), pa.array(p.plays[d][before][:, 1])], ["seat", "tile"])
    positions = pa.table({
        **key,
        "seat": pa.array(seat), "sign": pa.array(sign), "decl": pa.array(p.decl[d]),
        "hand": _tiles(own), "history": _lists(k, history),
        "actions": _lists(na, pa.array(p.actions)), "q": _lists(na, pa.array(p.q)),
        "advantage": _lists(na, pa.array(np.repeat(sign, na) * (p.q - mean[group]))),
        "chosen": pa.array(p.plays[d, k, 1]), "n_worlds": pa.array(p.n_worlds),
    })
    worlds = pa.FixedSizeListArray.from_arrays(pa.array(p.worlds.ravel()), 4)
    splats = pa.table({
        **key,
        "actions": _lists(na, pa.array(p.actions)),
        "weights": _lists(nw, pa.array(p.weights)),
        "worlds": _lists(nw, worlds),
        "outcomes": _lists(na, _lists(np.repeat(nw, na), pa.array(p.outcomes))),
    })
    bt = p.bidder % 2
    margin = p.banked[:, bt].astype(np.int8) - p.banked[:, 1 - bt]
    all_plays = pa.StructArray.from_arrays(
        [pa.array(p.plays[:, :, 0].ravel()), pa.array(p.plays[:, :, 1].ravel())], ["seat", "tile"])
    deals = pa.table({
        "deal": pa.array(np.arange(len(p.hands), dtype=np.int32)),
        "deal_seed": seeds,
        "decl": pa.array(p.decl),
        "hands": _lists(np.full(len(p.hands), 4), _tiles(p.hands.ravel())),
        "plays": _lists(np.full(len(p.hands), N_PLAYS), all_plays),
        "banked": _lists(np.full(len(p.hands), 2), pa.array(p.banked.ravel())),
        "margin": pa.array(margin),
        "made": pa.array(p.banked[:, bt] >= p.bid),
    })
    return {"positions": positions, "splats": splats, "deals": deals}


# -- the staged repo -----------------------------------------------------------------


def stage(splits: Sequence[Split], out: str | Path, repo: str) -> Path:
    """Write the whole repo for `splits` into `out` (emptied first)."""
    _check(splits)
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    rows = {}
    for s in splits:
        for config, table in tables(s.pour, s.manifest).items():
            rows[s.name, config] = table.num_rows
            for i, start in enumerate(range(0, max(table.num_rows, 1), SHARD_ROWS[config])):
                path = out / "data" / config / f"{s.name}-{i:04d}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table.slice(start, SHARD_ROWS[config]), path,
                               compression="zstd", row_group_size=GROUP_ROWS[config])
        raw = out / "raw" / s.name
        raw.mkdir(parents=True)
        for f in ("pour.npz", "manifest.json"):
            shutil.copy(s.raw / f, raw / f)
        (raw / "manifest.json").write_text(json.dumps(s.manifest, indent=2) + "\n")
    shutil.copy(HERE / "CC0-1.0.txt", out / "LICENSE")
    (out / "README.md").write_text(card(splits, rows, out, repo))
    sums = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(out)}"
            for f in sorted(out.rglob("*")) if f.is_file() and f.name != "README.md"]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return out


def _check(splits: Sequence[Split]) -> None:
    if not splits or len({s.name for s in splits}) != len(splits):
        raise SystemExit("name each split once")
    same = {(s.manifest["spec"], s.pour.bidder, s.pour.bid, s.pour.utility) for s in splits}
    if len(same) != 1:
        raise SystemExit(f"the splits were poured under different specs: {sorted(same)}")
    seeds = [x for s in splits for x in s.manifest.get("deal_seeds") or ()]
    if len(seeds) != len(set(seeds)):
        raise SystemExit("two splits share a deal")


def upload(staged: Path, repo: str, message: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=staged, commit_message=message,
                      delete_patterns=["data/**", "raw/**"])
    return f"https://huggingface.co/datasets/{repo}"


# -- the card ------------------------------------------------------------------------


def _mb(paths) -> str:
    return f"{sum(f.stat().st_size for f in paths) / 1e6:.1f}"


def _command(m: dict) -> str:
    seed = f"--seed {m['seed']}" + (" --fiber" if m.get("fiber") else "")
    return (f"python -u -m scenario.atlas pour --deals {m['deals']} --spec {m['spec']} "
            f"--bid {m['bid']} --bidder {m['bidder']} {seed} --out DIR")


def card(splits: Sequence[Split], rows: dict, out: Path, repo: str) -> str:
    s0 = splits[0]
    p, m = s0.pour, s0.manifest
    configs = "\n".join(
        f"- config_name: {c}\n" + ("  default: true\n" if c == "positions" else "") + "  data_files:\n"
        + "".join(f"  - split: {s.name}\n    path: data/{c}/{s.name}-*.parquet\n" for s in splits)
        for c in ("positions", "splats", "deals"))
    n = sum(rows[s.name, "positions"] for s in splits)
    size = next(c for b, c in ((1e3, "n<1K"), (1e4, "1K<n<10K"), (1e5, "10K<n<100K"), (1e6, "100K<n<1M"),
                               (1e7, "1M<n<10M"), (1e8, "10M<n<100M")) if n < b)
    split_rows = "\n".join(
        f"| `{s.name}` | {len(s.pour.hands):,} | {s.manifest['deal_seeds'][0] if s.manifest.get('deal_seeds') else '—'}"
        f"..{s.manifest['deal_seeds'][-1] if s.manifest.get('deal_seeds') else '—'} | {rows[s.name, 'positions']:,} | "
        f"{len(s.pour.actions):,} | {len(s.pour.worlds):,} | "
        f"{_mb((out / 'data').glob(f'*/{s.name}-*.parquet'))} | {_mb((out / 'raw' / s.name).glob('*'))} | "
        f"`{s.manifest['commit'][:8]}` | {s.manifest.get('wall_seconds', '—')} s |" for s in splits)
    cal = []
    for s in splits:
        c = calibration(s.pour)
        fm = "—" if c.forecast_margin is None else f"{c.forecast_margin:+.1f}"
        cal.append(f"| `{s.name}` | {c.n:,} | {fm} | {c.real_margin:+.1f} | {c.forecast_make:.0%} | {c.real_make:.0%} |")
    commands = "\n".join(f"# {s.name}\n{_command(s.manifest)}" for s in splits)
    names = ", ".join(f"{i} {x}" for i, x in enumerate(DECLS))
    every = np.concatenate([s.pour.n_worlds for s in splits])
    exact = float(np.mean(every < every.max()))
    weights = "1" if all(np.all(s.pour.weights == 1) for s in splits) else "not uniform"
    first = splits[0].name
    return f"""---
license: cc0-1.0
pretty_name: Texas 42 Walt pour (advantage dataset)
size_categories:
- {size}
task_categories:
- reinforcement-learning
tags:
- texas-42
- dominoes
- imperfect-information
- game-ai
- planning
- advantage
- self-play
configs:
{configs}---

# Texas 42: the Walt pour

An **advantage dataset** for the trick-taking domino game Texas 42. Walt, an imperfect-information
planner, plays all four seats of {sum(len(s.pour.hands) for s in splits):,} seeded deals. At every position
where the seat to move has a real choice, the whole outcome distribution of every legal move is kept, not
just the move Walt played. Public domain (CC0 1.0), by Jason Yandell; the code that made it is
[{SOURCE.removeprefix('https://github.com/')}]({SOURCE}) (`scenario/atlas.py`, `scenario/hf.py`).

## What a row is

A **position** is a point in a deal where the seat to move holds two or more legal tiles (a forced play
labels nothing and is not kept). The mover cannot see the other hands, so it samples **worlds**: deals of
the unseen tiles consistent with everything public (the plays so far and the voids they reveal). Under its plan
(the mover chooses once per cell of worlds it cannot tell apart; the other seats are modeled), each legal
move is played forward in every world to the end of the hand. The **splat** of a position is that
[actions × worlds] table of final outcomes, with the worlds and their weights.

**Units: bidder team.** Every outcome is the *bidding team's* final margin, bidder-team points minus
defender points, an integer in [-42, 42] (the hand has 42 points). `q[a]` is the weighted mean of move
`a`'s outcomes over the worlds. A defender wants these small. `sign` is +1 when the mover is on the
bidding team, -1 when it defends, and `advantage = sign * (q - mean(q))`, so a positive advantage is a
good move *for the seat to play*, with the legal moves as the group. The contract is made when the margin
is at least `2·bid − 42` (= {2 * p.bid - 42} for bid {p.bid}), so any bid's make probability is a tail of the splat.

**Paired across worlds.** Every move of a position is played out in the *same* worlds, so per-world
differences (`outcomes[a] − mean over moves`, times `sign`) remove the luck of the deal that all moves
share. That pairing is the point of keeping the splat: hundreds of paired outcomes per move per position.

## Configs and splits

Three tables, joined by `position` (the row's index in its split's pour) or by `deal`/`deal_seed`.
`(deal_seed, n_plays)` is unique across all splits.

- **`positions`** (default): one row per position, light enough to browse — the record, the moves, `q`, `advantage`.
- **`splats`**: the same rows' per-world outcomes, worlds and weights. Heavy; this is what a paired-advantage trainer reads.
- **`deals`**: one row per deal — the four real hands, all 28 plays, and how the hand really ended.

One row per position with list columns (not one row per action) keeps a group of legal moves together,
which is what a group-relative trainer consumes, and keeps a position's worlds stored once.

| split | deals | deal seeds | positions | actions | worlds | Parquet MB | raw MB | commit | pour wall |
|---|---|---|---|---|---|---|---|---|---|
{split_rows}

The splits are disjoint deals poured under the same spec: `{m['spec']}`, seat {p.bidder} bids {p.bid} and
declares the pip trump the arena's hand heuristic rates strongest (`arena.hand_metrics.best_trump`:
bid ceiling, then trump length), seat {p.bidder} leads,
utility `{p.utility}`. Deal *i* of a split is `scenario.atlas.deal(seed + i)`.

### Encodings

- **Tiles** are ids 0..27 ordered (0,0) (1,0) (1,1) (2,0) (2,1) (2,2) ... (6,6): id = hi·(hi+1)/2 + lo.
- **Tile masks** (`worlds`): bit t set means the seat holds tile t.
- **Seats** 0..3 in play order; teams are {{0, 2}} and {{1, 3}}; seat {p.bidder} is the bidder.
- **decl** (trump): {names}.

### `positions`

| column | type | meaning |
|---|---|---|
| `position` | int32 | index of the position in its split's pour (`scenario.atlas.Pour.position(i)`) |
| `deal`, `deal_seed` | int32 | the deal's index in the split and its seed |
| `n_plays` | int8 | plays before this position; the record is the deal's first `n_plays` plays |
| `seat` | int8 | the seat to move |
| `sign` | int8 | +1 if the mover is on the bidding team, -1 if defending |
| `decl` | int8 | trump |
| `hand` | list&lt;int8&gt; | the mover's remaining tiles |
| `history` | list&lt;struct&lt;seat, tile&gt;&gt; | the plays so far, in order |
| `actions` | list&lt;int8&gt; | the legal tiles, ascending |
| `q` | list&lt;float64&gt; | per action, the mean outcome over the worlds (bidder-team margin) |
| `advantage` | list&lt;float64&gt; | per action, `sign·(q − mean(q))`: positive is good for the mover |
| `chosen` | int8 | the tile Walt played (argmax of `sign·q`) |
| `n_worlds` | int32 | worlds sampled at this position |

### `splats`

| column | type | meaning |
|---|---|---|
| `position`, `deal`, `deal_seed`, `n_plays` | | as in `positions` |
| `actions` | list&lt;int8&gt; | the legal tiles, ascending |
| `weights` | list&lt;float64&gt; | per world, the belief weight (unnormalized) |
| `worlds` | list&lt;uint32[4]&gt; | per world, each seat's remaining tiles as a mask; the mover's entry is its real hand |
| `outcomes` | list&lt;list&lt;int8&gt;&gt; | `[action][world]`: the bidder-team final margin |

`q[a] = Σ_w weights[w]·outcomes[a][w] / Σ_w weights[w]`, exactly as the planner computed it.

### `deals`

| column | type | meaning |
|---|---|---|
| `deal`, `deal_seed` | int32 | as above |
| `decl` | int8 | trump |
| `hands` | list&lt;list&lt;int8&gt;&gt; | the four dealt hands, by seat |
| `plays` | list&lt;struct&lt;seat, tile&gt;&gt; | all 28 plays in order |
| `banked` | list&lt;int8&gt; | final points of team (0, 2) and team (1, 3) |
| `margin` | int8 | the bidding team's real final margin |
| `made` | bool | whether the bidding team made its bid |

## Loading

```python
from datasets import load_dataset

pos = load_dataset("{repo}", "positions", split="{first}")
row = pos[0]
best = row["actions"][max(range(len(row["actions"])), key=row["advantage"].__getitem__)]
```

```python
import numpy as np
import pandas as pd

base = "hf://datasets/{repo}/data"
pos = pd.read_parquet(f"{{base}}/positions/{first}-0000.parquet", columns=["position", "sign"])
splats = pd.read_parquet(f"{{base}}/splats/{first}-0000.parquet").merge(pos, on="position")
s = splats.iloc[0]
outcomes = np.stack(s["outcomes"]).astype(float)        # [actions, worlds], bidder-team margin
paired = s["sign"] * (outcomes - outcomes.mean(axis=0))  # per-world advantage, the mover's units
```

The raw pour round-trips exactly with the repo's own reader:

```python
from huggingface_hub import snapshot_download
from scenario.atlas import load, advantages     # github.com/jasonyandell/mk5-main

root = snapshot_download("{repo}", repo_type="dataset", allow_patterns="raw/{first}/*")
pour = load(f"{{root}}/raw/{first}")
advantages(pour.position(0))
```

## How it was made

```bash
{commands}
python -u -m scenario.hf publish --repo {repo} {' '.join(f'--split {s.name}=DIR' for s in splits)}
```

The `commit` column above is the mk5-main commit the pour ran at. Walt here is the scenario player of
[the mk5-main repo]({SOURCE}) at level 1 (`k1`): the mover plans over 512 root worlds (`n512`) while the
other seats are modeled by a level-0 Walt sampling 8 worlds of its own (`m8`), with the points utility and a
cell-size floor of 16 below the root (a cell of fewer than 16 worlds hands the decision to the rung below
with a fresh sample, so the plan never turns clairvoyant).
Walt in all four seats is honest self-play: every seat samples its own worlds at every decision.

**Calibration.** The bidder's opening forecast against how the deals really ended:

| split | deals | forecast margin | real margin | forecast make | real make |
|---|---|---|---|---|---|
{chr(10).join(cal)}

## What it is not

- **Not oracle labels.** No value here saw the hidden hands. Each is Walt's honest estimate from the
  mover's side; perfect-information ("double-dummy") values of these positions are a different object.
- **Worlds are samples, until they are not.** A position's worlds are drawn uniformly from its fiber, every
  completion of the hidden hands consistent with the record and the voids it has revealed. When the fiber
  has fewer than {p.n_worlds.max()} members it is enumerated outright and the splat is exact; that holds for
  {exact:.0%} of the positions here, all late in the hand. Every weight is {weights} in this release.
- **One line per deal.** Positions follow the line Walt actually played, not the deal's whole game tree.
- **One contract shape.** Seat {p.bidder} always bids {p.bid}; there is no auction and no nello, splash
  or plunge.

## Citation

```bibtex
@misc{{yandell2026waltpour,
  author = {{Jason Yandell}},
  title  = {{Texas 42: the Walt pour}},
  year   = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{repo}}}}},
  note   = {{Code: {SOURCE}}}
}}
```

## License

CC0 1.0 Universal (public domain dedication); the legal code is in `LICENSE`.
"""


# -- the command line ----------------------------------------------------------------


def _main(argv: Sequence[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="python -u -m scenario.hf")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("publish", help="stage pour directories as a dataset repo and upload it")
    pp.add_argument("--repo", required=True, help="a dataset repo id, e.g. jasonyandell/texas-42-walt-pour")
    pp.add_argument("--split", action="append", required=True, metavar="NAME=DIR", help="a pour directory as a split")
    pp.add_argument("--commit", help="the commit a pour ran at, when its manifest does not record one")
    pp.add_argument("--stage", help="the staging directory (default scratch/hf/<repo name>)")
    pp.add_argument("--no-upload", action="store_true", help="stage only")
    args = ap.parse_args(argv)

    splits = [read_split(*s.split("=", 1), args.commit) for s in args.split]
    out = Path(args.stage or HERE.parent / "scratch" / "hf" / args.repo.split("/")[-1])
    stage(splits, out, args.repo)
    print(f"staged {', '.join(f'{s.name} ({len(s.pour):,} positions)' for s in splits)} -> {out}", flush=True)
    if not args.no_upload:
        print(upload(out, args.repo, f"publish {', '.join(s.name for s in splits)}"), flush=True)


if __name__ == "__main__":
    _main()
