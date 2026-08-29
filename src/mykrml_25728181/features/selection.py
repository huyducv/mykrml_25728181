"""Inspecting and pruning redundant features.

Aggregating a panel by several summaries at once multiplies the number of
near-duplicate columns, so the same correlation check gets run repeatedly -
before the filter, after it, and on the source data to justify the drop list.
"""

import numpy as np
import pandas as pd


def correlation_pairs(df, threshold=0.95, numeric_only=True, verbose=True, head=None):
    """List the feature pairs whose absolute correlation exceeds a threshold

    Only the upper triangle is kept, so each pair appears once and no feature
    is reported against itself.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of features
    threshold : float
        Absolute correlation above which a pair is reported (default: 0.95)
    numeric_only : bool
        Restrict the correlation matrix to numeric columns (default: True)
    verbose : bool
        Print the count and the pairs
    head : int
        When printing, show only this many pairs. None (default) shows all

    Returns
    -------
    pd.Series
        Absolute correlation of each pair above the threshold, sorted descending,
        indexed by the two feature names
    """
    corr = df.corr(numeric_only=numeric_only).abs()

    upper_tri_mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairs = corr.where(upper_tri_mask).unstack().dropna().sort_values(ascending=False)

    high_corr = pairs[pairs > threshold]

    if verbose:
        print(f"Feature pairs ({len(high_corr)}) with correlation > {threshold}:")
        shown = high_corr.head(head) if head else high_corr
        print(shown.to_string())

    return high_corr


def plot_correlation_heatmap(df, title="Correlation heatmap (multicollinearity check)",
                             figsize=(12, 10), numeric_only=True, cmap="coolwarm"):
    """Plot the absolute correlation matrix of a feature set

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of features
    title : str
        Plot title
    figsize : tuple
        Figure size in inches (default: (12, 10))
    numeric_only : bool
        Restrict the correlation matrix to numeric columns (default: True)
    cmap : str
        Matplotlib colourmap name (default: 'coolwarm')

    Returns
    -------
    pd.DataFrame
        The absolute correlation matrix that was plotted
    """
    import matplotlib.pyplot as plt

    corr = df.corr(numeric_only=numeric_only).abs()

    plt.figure(figsize=figsize)

    try:
        import seaborn as sns
        sns.heatmap(corr, annot=False, fmt=".2f", cmap=cmap,
                    vmin=-1, vmax=1, linewidths=0.5)
    except ImportError:
        plt.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        plt.colorbar()

    plt.title(title)
    plt.tight_layout()
    plt.show()

    return corr


def split_feature_types(df, cat_cols, verbose=True):
    """Separate the columns of a feature matrix into categorical and numeric

    The preprocessing pipelines route the two groups differently, so getting
    this split wrong silently one-hot encodes a numeric column or feeds a
    string to a model expecting a float.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix
    cat_cols : list
        Columns to treat as categorical
    verbose : bool
        Print how many columns fall into each group

    Returns
    -------
    list
        Numeric column names
    list
        Categorical column names, restricted to those actually present
    """
    cat_cols = [c for c in cat_cols if c in df.columns]
    num_cols = [c for c in df.columns if c not in cat_cols]

    if verbose:
        print(f"Numeric columns: {len(num_cols)}")
        print(f"Categorical columns: {len(cat_cols)}")

    return num_cols, cat_cols
