---
title: Publish data to Hugging Face — Playbook
kind: playbook
first_seen: 2026-10-12
last_updated: 2026-10-20
status: active
---

Generated data goes to Hugging Face, not GitHub and not only `scratch/`. Jason
(2026-10-12): "this is all public so anything at all that is a bunch of
generated data? straight to huggingface. costs nothing financially and might
save us from lost data." Git carries the code that made the data and the wiki
page that explains it; the Hub carries the bytes. Every dataset is public,
**CC0 1.0**, under the account `jasonyandell`.

## When

- **The moment a wiki page cites it.** If a page names a `scratch/` path, a
  number read off a file, or a net trained on a file, that file is published
  before the page is committed — or in the same session at the latest.
- **Any generated dataset:** a pour, an arena's per-deal or per-hand output
  (`*.jsonl`, `per_hand.csv`), a study's outputs (JSON, logs, the scripts that
  made them), a trained net that something ships or cites.
- **Before a worktree is removed.** `scratch/` is gitignored and per checkout;
  a worktree's scratch dies with it. The 2026-10-12 research archive found two
  hub-cited artifacts already gone (`scratch/stochastic-field/`,
  `scratch/book/wave1/overlay-price.md`). The 2026-10-14 recovery
  ([[huggingface-assets]] § What stayed local) found 28 cited paths gone,
  one per study or file family: 13 had been staged for the Hub first, 3 can
  be rebuilt from elsewhere, 12 are lost for good.

Not for: code (git), prose (the wiki), anything under the "Never" list below.

## Where

| what | repo pattern | example |
|---|---|---|
| a pour (one training dataset, all its chunks) | `texas-42-<player>-pour[-<size>]`, one repo per pour, one split per chunk | [texas-42-zeb-pour-250k](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-pour-250k) |
| a study's or a month's research artifacts | `texas-42-<family>-research-<yyyy-mm>`, one repo per month, one directory per study | [texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10) |
| an era's evidence recovered after the fact | `texas-42-<family>-research-<yyyy-mm>[-<mm>]`, one directory per scratch directory, many-file directories as `<name>.tar.gz` + `<name>.files.txt` | [texas-42-mk5-research-2026-04-07](https://huggingface.co/datasets/jasonyandell/texas-42-mk5-research-2026-04-07) (the oracle and LLM era), [texas-42-walt-lanes-2026-08-09](https://huggingface.co/datasets/jasonyandell/texas-42-walt-lanes-2026-08-09) (`~/data/texas-42`), [texas-42-walt-speed-2026-09](https://huggingface.co/datasets/jasonyandell/texas-42-walt-speed-2026-09) (texas-42 `experiments/`) |

A pour that grows (expert iteration's next round) is republished as the whole
pour under one repo, not scattered: the 250k pour first went out as two repos
(`texas-42-zeb-pour`, `texas-42-zeb-pour-200k`) whose cards described the
planner wrongly, and a later reader downloaded the wrong pour. Both were
deleted 2026-10-14 after every raw chunk was matched by SHA-256 in the 250k repo. A research repo
can take more directories in later sessions of the same month (`upload_folder`
adds; it does not delete unless told to).

## How

The token is the Keychain entry `huggingface-token` ([docs/SECRETS.md](../../docs/SECRETS.md)).
Either export it in the shell that uploads, or let `huggingface_hub` read its
own store (`~/.cache/huggingface/token`, rehydrated from the Keychain with
`hf auth login --token "$HF_TOKEN"`). Never write it into a file in the repo, a
log, a script, or a commit.

```bash
export HF_TOKEN=$(security find-generic-password -s huggingface-token -w)
```

**A pour** — `scenario/hf.py` stages Parquet configs (`positions`, `splats`,
`deals`), the raw `pour.npz` + `manifest.json` per split for an exact
`scenario.atlas.load`, the field nets the specs name (`nets/`, with their
training sidecars), the card, CC0 and SHA256SUMS, and uploads:

```bash
python -u -m scenario.hf publish --repo jasonyandell/texas-42-zeb-pour-250k \
    --title "Texas 42: the Zeb pour, 250,000 deals" \
    --notes scratch/hf-publish/zeb-pour-250k-notes.md \
    --net scratch/field_net/v3/margin_d250k_h128x64.npz \
    --split s100001=scratch/atlas/pour_make_netfield_s100001_d50000 \
    --split s200001=scratch/atlas/zeb200k/pour_s200001_d50000   # … one --split per chunk
```

Stage first with `--no-upload --stage scratch/hf/<repo>` and read the card.
Splits must share one planner and contract; they may differ in the field net
(rounds of an expert iteration), and the card's table names each split's net.

**A folder of study outputs** — copy the directories into a staging folder
with an explicit exclusion list, scan, write the card, upload. The script that
did it for 2026-10-12 ships in that repo as `publish.py`; the shape:

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo, repo_type="dataset", exist_ok=True)
api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=staged,
                  commit_message="publish …")
```

Put `*.jsonl` and `*.npz` in the staged `.gitattributes` as LFS; the Hub stores
anything over 10 MB as LFS regardless. Size it first (`du -sh`); over ~2 GB of
study outputs is a question for Jason before uploading, not after.

### The card

Every repo's `README.md`:

- YAML front matter: `license: cc0-1.0`, `pretty_name`, tags (`texas-42`, …),
  and for Parquet the `configs` block (the pour publisher writes it).
- What it is in one paragraph: who generated it (which player, spec, net by
  name **and sha256**), the utility and contract, the dates, the mk5-main
  commit(s) it ran at.
- What each column, file or directory means; for a research repo, a table
  mapping every directory to the wiki page that explains it.
- What it is the input of (a net trained on it, an arena that cites it).
- Links to the wiki hub and to [the GitHub repo](https://github.com/jasonyandell/mk5-main).
- What was left out, and why.

### Verify

**Count and name every file first.** A study folder must hold on the Hub exactly the files you
staged. A flattened upload (every run's ledger copied onto one `runs/ledger.json`) once
published one batch of 288 and one of 120, and the one-file checksum below passed both times
(the ch9 datasets, fixed 2026-10-20, #201). `scenario.hf publish` checks this itself after
uploading. For a study folder, run it yourself:

```bash
python -u -m scenario.hf verify --repo jasonyandell/texas-42-zeb-research-2026-10 \
    --staged scratch/<stage>/book/<study> --path book/<study>
```

It fails naming every missing or extra file, then matches one file's SHA-256 downloaded
fresh. Keep the spot check below as well when the source lives outside the staging folder.

Download one raw file into a fresh directory and match its SHA256 against the
local source (and against the repo's `SHA256SUMS`):

```python
from huggingface_hub import snapshot_download
root = snapshot_download(repo, repo_type="dataset", allow_patterns="raw/s100001/*",
                         local_dir="scratch/hf-verify/<repo>")
```

```bash
shasum -a 256 scratch/hf-verify/<repo>/raw/s100001/pour.npz <local chunk>/pour.npz
```

Then put the repo link where the data is described (the hub paragraph that
cites it) and a line in [[log]].

## Never

- **Secrets.** Tokens, keys, `.env` files, `~/.cache` contents. Scan staged
  text for anything credential-shaped before uploading.
- **The book's text.** Nothing from Roberson's *Winning 42* — page images, OCR,
  transcribed hands, bids or reasoning (`scratch/winning42*/`,
  `book_hands.json`, briefs that quote it). Our measurements of the book's
  conventions may go; they name a convention by number and cite a page.
- **Anything not ours.** Third-party datasets, downloaded models, other
  people's code.
- **Build output and copies of the repo.** Wheels, `site-packages`, `.lake/`,
  bytecode, checkouts of mk5-main at a commit (git already has them).

## Related

- [[scenario-player]] — the hub whose pours and research these repos hold
- [[research-night]] — the overnight procedure whose outputs land in a research repo
