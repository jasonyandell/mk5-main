# eq-browser-visualizers — audit 2026-07-07

## Corrections

- Page said the PDF/journey sample uses a "small checked-in E[Q] PDF tensor"; the tensor `forge/data/eq_pdf_s9200-9201_d10_10s.pt` is not in the repo and was never committed (evidence: `git log --all -- forge/data/eq_pdf*` is empty; `forge/data/` does not exist). Page updated to say the tensor is local-only and that without it the export script only rebuilds the 27a aggregate surface.

## Verified

- Script, server, three HTML pages, 27a CSV inputs, output filenames, port 8000, `--limit` default of 5, and the `../data/*.jsonl` fetch-with-dummy-fallback behavior all match the page.

## Follow-ups

- Consider actually checking in a tiny E[Q] PDF tensor (or a pre-exported `data/*.jsonl` sample) so the PDF-discs and game-journey visualizers work from a fresh clone.
