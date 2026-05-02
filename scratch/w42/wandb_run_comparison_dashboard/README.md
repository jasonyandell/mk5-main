# w42 W&B Run Comparison Dashboard Artifacts

Bead: `t42-csw6.26`

W&B project: <https://wandb.ai/jasonyandell-forge42/w42>

The local W&B API check succeeded through `/Users/jason/.netrc` and returned 10
project runs. Programmatic workspace/report creation is not available in this
environment because `wandb_workspaces` is not installed, so this artifact
captures a reusable view definition instead of a hosted report URL.

## Files

- `run_comparison.csv` - direct run links, grouping, status, headline metrics,
  HF links, and claim-ledger impact.
- `wandb_view_definition.json` - reusable W&B workspace/report view definition:
  filters, groupings, columns, and panels.
- `wiki/experiments/w42-wandb-run-comparison-dashboard.md` - narrative report
  and future tagging conventions.

## Headline Links

- Project: <https://wandb.ai/jasonyandell-forge42/w42>
- Rich primary run: <https://wandb.ai/jasonyandell-forge42/w42/runs/3xyy2dxr>
- 84 corrected run: <https://wandb.ai/jasonyandell-forge42/w42/runs/vmta9zew>
- Useful failed 84 run: <https://wandb.ai/jasonyandell-forge42/w42/runs/oizyjty5>

## API Commands

```bash
python -m wandb status
python - <<'PY'
import wandb
api = wandb.Api(timeout=20)
runs = list(api.runs("jasonyandell-forge42/w42", per_page=100))
print(len(runs))
for run in runs:
    print(run.id, run.name, run.state, run.group, run.url)
PY
```
