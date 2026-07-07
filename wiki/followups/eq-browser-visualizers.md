# eq-browser-visualizers — audit 2026-07-07

## Corrections

- Page said the PDF/journey sample uses a "small checked-in E[Q] PDF tensor"; the tensor `forge/data/eq_pdf_s9200-9201_d10_10s.pt` is not in the repo and was never committed (evidence: `git log --all -- forge/data/eq_pdf*` is empty; `forge/data/` does not exist). Page updated to say the tensor is local-only and that without it the export script only rebuilds the 27a aggregate surface.

## Verified

- Script, server, three HTML pages, 27a CSV inputs, output filenames, port 8000, `--limit` default of 5, and the `../data/*.jsonl` fetch-with-dummy-fallback behavior all match the page.

## Follow-ups

- Consider actually checking in a tiny E[Q] PDF tensor (or a pre-exported `data/*.jsonl` sample) so the PDF-discs and game-journey visualizers work from a fresh clone. Note: the exported jsonl files do exist in the main checkout at `forge/analysis/results/data/` but the blanket `data/` rule (`.gitignore:103`) excludes them, so checking them in requires a gitignore exception.

## Review (second pass, 2026-07-07)

- Verified — corrections stand. Independently confirmed: no git history for `forge/data/eq_pdf*` in any ref; `forge/data/` absent from the tree (and empty in the main checkout); `.gitignore:103` (`data/`) would ignore the tensor anyway; `forge/analysis/CLAUDE.md` explicitly states `forge/data/` doesn't exist. The "only rebuilds the 27a aggregate surface" claim matches `forge/analysis/scripts/export_eq_visualizer_data.py` (`export_game_and_pdf` returns `None` when the tensor is missing, lines 244-246; `export_surface` always runs, line 273). All "Verified" items re-checked: port 8000 (`serve.py`), `--limit` default 5 (line 268), output filenames (lines 107, 256-257), `../data/*.jsonl` fetch + dummy fallback (`eq_surface_3d.html:432/467/491`). Only amendment: annotated the follow-up suggestion with the gitignore constraint discovered during review.
