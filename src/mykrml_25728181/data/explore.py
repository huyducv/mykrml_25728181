"""Exploratory summaries that get repeated once per dataset.

These exist so that the same question - "how much is missing?", "how skewed is
the target?" - is answered the same way for the training and the testing frame,
rather than being re-typed with a subtle difference.
"""

import pandas as pd


def missing_summary(frames, as_pct=True, only_missing=True, sort_by=None, round_to=1):
    """Compare missingness across one or more dataframes, column by column

    Parameters
    ----------
    frames : dict or pd.DataFrame
        Mapping of dataset name to dataframe, e.g. ``{'train': df1, 'test': df2}``.
        A bare dataframe is treated as ``{'df': frame}``
    as_pct : bool
        Report percentages of rows (default) rather than raw counts
    only_missing : bool
        Keep only columns missing in at least one of the frames (default: True)
    sort_by : str
        Name of the frame to sort on. Defaults to the first one given
    round_to : int
        Decimal places for the returned figures (default: 1)

    Returns
    -------
    pd.DataFrame
        One row per column, one numeric column per input frame
    """
    if isinstance(frames, pd.DataFrame):
        frames = {"df": frames}

    suffix = "_pct" if as_pct else "_count"
    summary = pd.DataFrame({
        f"{name}{suffix}": (df.isna().mean() * 100 if as_pct else df.isna().sum())
        for name, df in frames.items()
    })

    if only_missing:
        summary = summary[(summary > 0).any(axis=1)]

    sort_col = f"{sort_by}{suffix}" if sort_by else summary.columns[0]

    return summary.sort_values(sort_col, ascending=False).round(round_to)


def target_summary(y, name="target", verbose=True):
    """Report the class balance of a binary target and the baselines it implies

    On an imbalanced target the no-skill AUPRC equals the positive rate, and the
    accuracy of always predicting the majority class equals one minus it - which
    is why accuracy is unusable in that setting and AUPRC is not.

    Parameters
    ----------
    y : pd.Series or Numpy Array
        Binary target
    name : str
        Name used in the printed report
    verbose : bool
        Print the summary (default: True)

    Returns
    -------
    dict
        ``n``, ``positives``, ``negatives``, ``positive_rate``,
        ``no_skill_auprc`` and ``majority_accuracy``
    """
    y = pd.Series(y)
    positive_rate = float(y.mean())

    stats = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "negatives": int(len(y) - y.sum()),
        "positive_rate": positive_rate,
        "no_skill_auprc": positive_rate,
        "majority_accuracy": 1 - positive_rate,
    }

    if verbose:
        print(f"{name}: {stats['n']} rows")
        print(f"  positive                  : {stats['positives']} ({positive_rate:.2%})")
        print(f"  negative                  : {stats['negatives']}")
        print(f"  no-skill AUPRC baseline   : {positive_rate:.4f}  (equals the positive rate)")
        print(f"  always-predict-0 accuracy : {1 - positive_rate:.4f}  (why accuracy is unusable)")

    return stats
