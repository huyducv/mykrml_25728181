"""Column-level transformations applied identically to every dataset.

Each of these repairs a column that arrives in a form a model cannot use: an
ordered category stored as a label, or a categorical dtype flattened back to
object by a ``ColumnTransformer``.
"""


def map_ordinal(df, col, mapping, out_col=None):
    """Map an ordered category onto an integer scale

    Use this rather than one-hot encoding when the categories have a natural
    order the model should be able to see. Values absent from the mapping
    become NaN, which is the wanted outcome for corrupted entries: they are
    removed without having to be enumerated.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    col : str
        Column to map
    mapping : dict
        Label to integer mapping, e.g. ``{'low': 1, 'medium': 2, 'high': 3}``
    out_col : str
        Destination column. Defaults to overwriting ``col``

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe with the column mapped
    """
    out = df.copy()
    out[out_col or col] = out[col].map(mapping)

    return out


def cast_categorical(df, cols):
    """Cast the given columns to the pandas ``category`` dtype

    Gradient boosting libraries use their native categorical handling only when
    the dtype says so; left as strings the same columns are either rejected or
    silently mishandled.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    cols : list
        Columns to cast

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe with those columns cast to 'category'
    """
    out = df.copy()
    present = [c for c in cols if c in out.columns]
    out[present] = out[present].astype("category")

    return out


def object_to_category(df):
    """Cast every object or string column of a dataframe to ``category``

    A ``ColumnTransformer`` returns a plain frame in which any ``category``
    dtype set upstream has been flattened back to object. This restores it. It
    is defined at module level rather than as a closure so that the transformer
    built by ``make_category_caster`` stays picklable, which parallel
    cross-validation requires.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe with object columns cast to 'category'
    """
    out = df.copy()
    object_cols = out.select_dtypes(include=["object", "string"]).columns
    out[object_cols] = out[object_cols].astype("category")

    return out


def make_category_caster():
    """Return ``object_to_category`` wrapped as a scikit-learn transformer

    Normally used as a pipeline step between a ``ColumnTransformer`` and a model
    that relies on native categorical handling.

    Returns
    -------
    sklearn.preprocessing.FunctionTransformer
        Transformer usable as a step inside a ``Pipeline``
    """
    from sklearn.preprocessing import FunctionTransformer

    return FunctionTransformer(object_to_category)
