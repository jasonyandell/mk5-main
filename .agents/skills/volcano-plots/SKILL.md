---
name: volcano-plots
description: Volcano plot visualization for differential analysis. Use for visualizing significance vs effect size, identifying differentially expressed features, and highlighting statistically significant changes with large magnitudes.
---

# Volcano Plots Guide

Volcano plots visualize statistical significance (p-value) vs magnitude of change (fold change or effect size). They enable quick identification of features with both large effects and high statistical significance.

## Quick Start

### Basic Volcano Plot

```python
import numpy as np
import matplotlib.pyplot as plt

def volcano_plot(
    log2_fold_change,
    neg_log10_pvalue,
    labels=None,
    fc_threshold=1.0,
    pval_threshold=0.05,
    figsize=(10, 8),
    alpha=0.6
):
    """
    Create a volcano plot.

    Parameters:
    -----------
    log2_fold_change : array-like
        Log2 fold change values (x-axis)
    neg_log10_pvalue : array-like
        -log10(p-value) values (y-axis)
    labels : array-like, optional
        Labels for points (for annotation)
    fc_threshold : float
        Fold change threshold for significance (default: 1.0 = 2-fold)
    pval_threshold : float
        P-value threshold (will be converted to -log10)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert p-value threshold to -log10 scale
    neg_log10_threshold = -np.log10(pval_threshold)

    # Classify points
    up = (log2_fold_change > fc_threshold) & (neg_log10_pvalue > neg_log10_threshold)
    down = (log2_fold_change < -fc_threshold) & (neg_log10_pvalue > neg_log10_threshold)
    ns = ~(up | down)

    # Plot each category
    ax.scatter(log2_fold_change[ns], neg_log10_pvalue[ns],
               c='gray', alpha=alpha*0.5, s=20, label='Not significant')
    ax.scatter(log2_fold_change[up], neg_log10_pvalue[up],
               c='red', alpha=alpha, s=30, label='Up-regulated')
    ax.scatter(log2_fold_change[down], neg_log10_pvalue[down],
               c='blue', alpha=alpha, s=30, label='Down-regulated')

    # Add threshold lines
    ax.axhline(y=neg_log10_threshold, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(x=fc_threshold, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(x=-fc_threshold, color='black', linestyle='--', linewidth=0.5)

    # Labels
    ax.set_xlabel('Log2 Fold Change', fontsize=12)
    ax.set_ylabel('-Log10(P-value)', fontsize=12)
    ax.set_title('Volcano Plot', fontsize=14)
    ax.legend(loc='upper right')

    plt.tight_layout()
    return fig, ax

# Example usage
log2_fc = np.random.randn(1000) * 2
pvalues = 10 ** (-np.abs(log2_fc) * np.random.uniform(0.5, 2, 1000))
neg_log10_p = -np.log10(pvalues)

fig, ax = volcano_plot(log2_fc, neg_log10_p)
plt.show()
```

## Using Seaborn

```python
import seaborn as sns
import pandas as pd

def volcano_plot_seaborn(df, x='log2_fold_change', y='neg_log10_pvalue',
                         fc_threshold=1.0, pval_threshold=0.05):
    """
    Volcano plot using seaborn.

    Parameters:
    -----------
    df : DataFrame
        Must contain columns for fold change and p-value
    x : str
        Column name for log2 fold change
    y : str
        Column name for -log10(p-value)
    """
    # Add significance classification
    df = df.copy()
    neg_log10_thresh = -np.log10(pval_threshold)

    conditions = [
        (df[x] > fc_threshold) & (df[y] > neg_log10_thresh),
        (df[x] < -fc_threshold) & (df[y] > neg_log10_thresh),
    ]
    choices = ['Up', 'Down']
    df['significance'] = np.select(conditions, choices, default='NS')

    # Color palette
    palette = {'Up': 'red', 'Down': 'blue', 'NS': 'gray'}

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.scatterplot(
        data=df, x=x, y=y,
        hue='significance', palette=palette,
        alpha=0.6, s=30, ax=ax,
        hue_order=['Up', 'Down', 'NS']
    )

    # Threshold lines
    ax.axhline(y=neg_log10_thresh, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(x=fc_threshold, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(x=-fc_threshold, color='black', linestyle='--', linewidth=0.5)

    ax.set_xlabel('Log2 Fold Change')
    ax.set_ylabel('-Log10(P-value)')
    ax.set_title('Volcano Plot')

    return fig, ax
```

## Interactive with Plotly

```python
import plotly.express as px

def volcano_plot_interactive(df, x='log2_fold_change', y='neg_log10_pvalue',
                              label_col='gene', fc_threshold=1.0, pval_threshold=0.05):
    """
    Interactive volcano plot with hover labels.
    """
    df = df.copy()
    neg_log10_thresh = -np.log10(pval_threshold)

    # Classify significance
    conditions = [
        (df[x] > fc_threshold) & (df[y] > neg_log10_thresh),
        (df[x] < -fc_threshold) & (df[y] > neg_log10_thresh),
    ]
    df['significance'] = np.select(conditions, ['Up', 'Down'], default='NS')

    # Create interactive plot
    fig = px.scatter(
        df, x=x, y=y,
        color='significance',
        color_discrete_map={'Up': 'red', 'Down': 'blue', 'NS': 'lightgray'},
        hover_name=label_col,
        hover_data={x: ':.2f', y: ':.2f'},
        title='Volcano Plot (Interactive)'
    )

    # Add threshold lines
    fig.add_hline(y=neg_log10_thresh, line_dash='dash', line_color='black')
    fig.add_vline(x=fc_threshold, line_dash='dash', line_color='black')
    fig.add_vline(x=-fc_threshold, line_dash='dash', line_color='black')

    fig.update_layout(
        xaxis_title='Log2 Fold Change',
        yaxis_title='-Log10(P-value)'
    )

    return fig
```

## Annotating Top Hits

```python
def volcano_with_labels(df, x='log2_fc', y='neg_log10_p', label_col='gene',
                        n_top=10, fc_threshold=1.0, pval_threshold=0.05):
    """
    Volcano plot with top hits labeled.
    """
    from adjustText import adjust_text

    fig, ax = plt.subplots(figsize=(12, 10))

    neg_log10_thresh = -np.log10(pval_threshold)

    # Classify
    up = (df[x] > fc_threshold) & (df[y] > neg_log10_thresh)
    down = (df[x] < -fc_threshold) & (df[y] > neg_log10_thresh)
    ns = ~(up | down)

    # Plot
    ax.scatter(df.loc[ns, x], df.loc[ns, y], c='gray', alpha=0.3, s=20)
    ax.scatter(df.loc[up, x], df.loc[up, y], c='red', alpha=0.6, s=30)
    ax.scatter(df.loc[down, x], df.loc[down, y], c='blue', alpha=0.6, s=30)

    # Find top hits (highest -log10 p-value among significant)
    significant = df[up | down].copy()
    top_hits = significant.nlargest(n_top, y)

    # Add labels
    texts = []
    for _, row in top_hits.iterrows():
        texts.append(ax.text(row[x], row[y], row[label_col], fontsize=8))

    # Adjust text to avoid overlap (requires adjustText package)
    try:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    except:
        pass  # Works without adjustText, just may have overlaps

    # Threshold lines
    ax.axhline(y=neg_log10_thresh, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(x=fc_threshold, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(x=-fc_threshold, color='black', linestyle='--', linewidth=0.5)

    ax.set_xlabel('Log2 Fold Change')
    ax.set_ylabel('-Log10(P-value)')
    ax.set_title(f'Volcano Plot (top {n_top} labeled)')

    return fig, ax
```

## Data Preparation

### From Raw Data

```python
import pandas as pd
from scipy import stats

def prepare_volcano_data(group1, group2, feature_names):
    """
    Calculate fold changes and p-values for volcano plot.

    Parameters:
    -----------
    group1, group2 : array-like
        Data matrices (samples x features)
    feature_names : array-like
        Names for each feature
    """
    results = []

    for i, name in enumerate(feature_names):
        # Get values for this feature
        vals1 = group1[:, i]
        vals2 = group2[:, i]

        # Calculate means
        mean1 = np.mean(vals1)
        mean2 = np.mean(vals2)

        # Log2 fold change (handle zeros)
        if mean1 > 0 and mean2 > 0:
            log2_fc = np.log2(mean2 / mean1)
        else:
            log2_fc = np.nan

        # T-test
        _, pvalue = stats.ttest_ind(vals1, vals2)

        results.append({
            'feature': name,
            'mean_group1': mean1,
            'mean_group2': mean2,
            'log2_fold_change': log2_fc,
            'pvalue': pvalue,
            'neg_log10_pvalue': -np.log10(pvalue) if pvalue > 0 else np.inf
        })

    df = pd.DataFrame(results)

    # Add adjusted p-values (FDR)
    from statsmodels.stats.multitest import multipletests
    _, df['padj'], _, _ = multipletests(df['pvalue'].fillna(1), method='fdr_bh')
    df['neg_log10_padj'] = -np.log10(df['padj'])

    return df
```

### From DESeq2/EdgeR Output

```python
def load_deseq2_results(filepath):
    """
    Load DESeq2 results for volcano plot.
    Expects columns: log2FoldChange, pvalue, padj
    """
    df = pd.read_csv(filepath)

    # Rename columns if needed
    column_map = {
        'log2FoldChange': 'log2_fold_change',
        'pvalue': 'pvalue',
        'padj': 'padj'
    }
    df = df.rename(columns=column_map)

    # Calculate -log10 values
    df['neg_log10_pvalue'] = -np.log10(df['pvalue'])
    df['neg_log10_padj'] = -np.log10(df['padj'])

    return df
```

## Customization

### Color by Effect Size

```python
def volcano_gradient(df, x='log2_fc', y='neg_log10_p'):
    """
    Volcano with color gradient by fold change magnitude.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        df[x], df[y],
        c=df[x],
        cmap='RdBu_r',
        alpha=0.6,
        s=30,
        vmin=-df[x].abs().max(),
        vmax=df[x].abs().max()
    )

    plt.colorbar(scatter, label='Log2 Fold Change')
    ax.set_xlabel('Log2 Fold Change')
    ax.set_ylabel('-Log10(P-value)')

    return fig, ax
```

### Symmetric Axes

```python
def volcano_symmetric(df, x='log2_fc', y='neg_log10_p'):
    """
    Volcano plot with symmetric x-axis.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(df[x], df[y], alpha=0.5, s=20)

    # Make x-axis symmetric
    max_fc = df[x].abs().max() * 1.1
    ax.set_xlim(-max_fc, max_fc)

    ax.set_xlabel('Log2 Fold Change')
    ax.set_ylabel('-Log10(P-value)')

    return fig, ax
```

## Multiple Comparisons

### Multi-panel Volcano

```python
def multi_volcano(comparisons, fc_col='log2_fc', pval_col='neg_log10_p',
                  fc_threshold=1.0, pval_threshold=0.05):
    """
    Create multiple volcano plots for different comparisons.

    Parameters:
    -----------
    comparisons : dict
        {comparison_name: DataFrame}
    """
    n = len(comparisons)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axes = np.atleast_2d(axes).flatten()

    neg_log10_thresh = -np.log10(pval_threshold)

    for ax, (name, df) in zip(axes, comparisons.items()):
        up = (df[fc_col] > fc_threshold) & (df[pval_col] > neg_log10_thresh)
        down = (df[fc_col] < -fc_threshold) & (df[pval_col] > neg_log10_thresh)
        ns = ~(up | down)

        ax.scatter(df.loc[ns, fc_col], df.loc[ns, pval_col], c='gray', alpha=0.3, s=10)
        ax.scatter(df.loc[up, fc_col], df.loc[up, pval_col], c='red', alpha=0.5, s=15)
        ax.scatter(df.loc[down, fc_col], df.loc[down, pval_col], c='blue', alpha=0.5, s=15)

        ax.axhline(y=neg_log10_thresh, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(x=fc_threshold, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(x=-fc_threshold, color='black', linestyle='--', linewidth=0.5)

        ax.set_title(name)
        ax.set_xlabel('Log2 FC')
        ax.set_ylabel('-Log10(P)')

        # Count significant
        n_up = up.sum()
        n_down = down.sum()
        ax.text(0.02, 0.98, f'Up: {n_up}\nDown: {n_down}',
                transform=ax.transAxes, va='top', fontsize=8)

    # Hide unused axes
    for ax in axes[len(comparisons):]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig
```

## Best Practices

| DO | DON'T |
|----|-------|
| Use adjusted p-values (FDR) | Use raw p-values without correction |
| Log2 transform fold changes | Use raw fold changes |
| Set meaningful thresholds | Use arbitrary thresholds |
| Label only top hits | Label all significant points |
| Use symmetric x-axis | Allow asymmetric distortion |
| Consider effect size + significance | Focus only on p-values |

## Common Thresholds

| Context | Fold Change | P-value |
|---------|------------|---------|
| Exploratory | |log2FC| > 0.5 | padj < 0.1 |
| Standard | |log2FC| > 1.0 | padj < 0.05 |
| Stringent | |log2FC| > 1.5 | padj < 0.01 |
| Very stringent | |log2FC| > 2.0 | padj < 0.001 |

## Integration with Analysis

### Save Significant Features

```python
def get_significant(df, fc_col='log2_fc', pval_col='padj',
                    fc_threshold=1.0, pval_threshold=0.05):
    """
    Extract significant features from volcano data.
    """
    up = (df[fc_col] > fc_threshold) & (df[pval_col] < pval_threshold)
    down = (df[fc_col] < -fc_threshold) & (df[pval_col] < pval_threshold)

    up_features = df[up].sort_values(fc_col, ascending=False)
    down_features = df[down].sort_values(fc_col, ascending=True)

    return up_features, down_features
```

## Resources

- **EnhancedVolcano (R)**: https://bioconductor.org/packages/EnhancedVolcano/
- **adjustText**: https://github.com/Phlya/adjustText
- **Plotly Express**: https://plotly.com/python/plotly-express/
