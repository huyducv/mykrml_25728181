"""Row-level cleaning steps applied identically to every dataset in a project,
typically a training frame and a testing frame.

Each function returns a new dataframe and leaves the input untouched, so a
cleaning sequence can be re-run in a notebook without accumulating state.
"""

import pandas as pd


def coerce_numeric(df, cols):
    """Cast the given columns to numeric, turning unparseable values into NaN

    Useful for columns read as ``object`` because a handful of values are
    stored as text. Normalising them also makes genuinely identical rows
    compare equal, which matters when de-duplication follows.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    cols : str or list
        Column name, or list of column names, to cast

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe with the requested columns cast to numeric
    """
    out = df.copy()
    if isinstance(cols, str):
        cols = [cols]

    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def drop_duplicate_rows(df, subset=None, name="dataset", keep="first", verbose=True):
    """Drop duplicated rows and report how many were removed

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    subset : list
        Columns defining a duplicate. None (default) compares whole rows
    name : str
        Name of the dataset, used in the printed report
    keep : str
        Which duplicate to keep, passed to ``drop_duplicates`` (default: 'first')
    verbose : bool
        Print the number of duplicates found and the resulting shape

    Returns
    -------
    pd.DataFrame
        Dataframe with duplicates removed
    """
    n_dupes = int(df.duplicated(subset=subset).sum())
    out = df.drop_duplicates(subset=subset, keep=keep)

    if verbose:
        scope = "identical rows" if subset is None else f"duplicates on {subset}"
        print(f"{name}: {n_dupes} {scope} removed -> shape {out.shape}")

    return out
