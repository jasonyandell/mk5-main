# W&B Reports: Documentation & Collaboration

Create interactive, shareable reports to document experiments, share findings, and collaborate with your team.

## Overview

Reports combine:
- **Visualizations**: Charts, tables, media from your runs
- **Narrative**: Markdown text explaining your experiments
- **Interactive elements**: Filters, run selectors, dynamic panels

## Creating Reports

### From the UI
1. Navigate to your project workspace
2. Click "Create report" in the upper right
3. Select charts to include
4. Add text and organize panels

### Programmatically (Python API)
```python
import wandb

api = wandb.Api()

# Create a new report
report = api.create_report(
    project="my-project",
    title="Model Comparison Report",
    description="Comparing ResNet architectures",
)

# Save the report
report.save()

print(f"Report URL: {report.url}")
```

### Using Report API (Preview)
```python
import wandb_workspaces.reports.v2 as wr

# Create report
report = wr.Report(
    project="my-project",
    title="Experiment Summary",
    description="Weekly experiment results",
)

# Add blocks
report.blocks = [
    wr.H1("Model Performance"),
    wr.P("This week we tested three new architectures..."),
    wr.PanelGrid(
        panels=[
            wr.LinePlot(x="epoch", y="val_accuracy"),
            wr.ScalarChart(metric="final_accuracy"),
        ]
    ),
]

report.save()
```

## Report Structure

### Blocks
Reports are composed of blocks:

| Block Type | Purpose |
|------------|---------|
| Headers (H1, H2, H3) | Section organization |
| Paragraph (P) | Narrative text |
| Panel Grid | Charts and visualizations |
| Run Selector | Filter runs interactively |
| Code Block | Display code snippets |
| LaTeX | Mathematical notation |

### Panel Types
Common visualization panels:

| Panel | Use Case |
|-------|----------|
| Line Plot | Training curves, metrics over time |
| Scatter Plot | Comparing two metrics |
| Bar Chart | Categorical comparisons |
| Parallel Coordinates | Hyperparameter analysis |
| Media Panel | Images, audio, video |
| Table | Tabular data, run comparisons |

## Sharing Reports

### Share with Team
1. Click "Share" button in upper right
2. Enter email address or username
3. Select permission level:
   - **Can view**: Read-only access
   - **Can edit**: Full editing rights

### Magic Link (No Login Required)
1. Click "Share" → "Copy magic link"
2. Anyone with the link can view (no W&B account needed)
3. Useful for stakeholders outside your organization

### Embedding Reports

#### In Notion
```markdown
Paste the report URL directly - Notion auto-embeds
```

#### In HTML/Websites
```html
<iframe
  src="https://wandb.ai/<entity>/<project>/reports/<report-id>/embed"
  width="100%"
  height="600"
  frameborder="0"
></iframe>
```

**Note**: Only public reports can be embedded.

## Report Templates

### Experiment Comparison Template
```
# Experiment: [Name]
## Overview
Brief description of experiment goals

## Methodology
- Model architecture
- Training data
- Hyperparameters

## Results
[Panel Grid with key metrics]

## Analysis
Key findings and insights

## Next Steps
Planned follow-up experiments
```

### Weekly Progress Template
```
# Week [N] Progress Report

## Summary
Brief overview of accomplishments

## Experiments Run
[Table of runs with key metrics]

## Best Performing Model
[Panels showing top model performance]

## Challenges
Issues encountered and solutions

## Next Week
Planned experiments
```

## Collaboration Features

### Comments
- Click on any panel or text to add comments
- Team members receive notifications
- Threaded discussions supported

### Version History
- Reports auto-save versions
- View and restore previous versions
- Track changes over time

### Filters and Run Selection
- Add run selectors to let viewers filter data
- Compare specific runs interactively
- Toggle run visibility

## Exporting Reports

### PDF Export
1. Open report in browser
2. Click "..." menu → "Download PDF"
3. Full report with all visualizations

### LaTeX Export
1. Click "..." menu → "Export LaTeX"
2. Download as .zip file
3. Includes figures and .tex file

## Best Practices

### Report Organization
```
1. Executive Summary (key findings first)
2. Background (brief context)
3. Methodology (reproducibility details)
4. Results (visualizations with explanations)
5. Discussion (insights, limitations)
6. Next Steps (actionable items)
```

### Effective Panels
- **Limit panels**: 4-6 per section maximum
- **Consistent styling**: Use same color schemes
- **Clear titles**: Descriptive panel names
- **Context**: Always explain what charts show

### Writing Tips
- Lead with conclusions, not process
- Use bullet points for key findings
- Reference specific runs by name
- Include hyperparameters in tables

### Common Anti-Patterns

| DON'T | DO |
|-------|-----|
| Too many panels | Focus on key metrics |
| No narrative | Explain what charts mean |
| Share raw workspace | Create curated report |
| Forget to update | Keep reports current |
| Embed private reports | Make public for embedding |

## Automating Reports

### Scheduled Reports (Enterprise)
Generate reports on schedule with latest runs.

### Programmatic Updates
```python
api = wandb.Api()
report = api.report("<entity>/<project>/<report-id>")

# Update title
report.title = "Updated Model Comparison"

# Add new section (API varies)
report.save()
```

## Report Gallery

W&B maintains a gallery of example reports:
- https://docs.wandb.ai/guides/reports/reports-gallery/

Use these as templates for your own reports.
