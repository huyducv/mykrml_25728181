def convert_to_date(df, cols:list):
    """Convert specified columns from a Pandas dataframe into datetime

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    cols : list
        List of columns to be converted

    Returns
    -------
    pd.DataFrame
        Pandas dataframe with converted columns
    """
    import pandas as pd

    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df





def compute_age(df, dob_col='dob', year_col='year', ref_month_day='12-31',
                out_col='age', drop_dob=True):
    """Derive age at the end of each recorded season from a date of birth

    The reference date is the end of the season the row describes, not the
    date the notebook happens to run, so re-running later does not silently
    change the feature. A raw date of birth is not used directly because it
    lets a model key on birth cohort, which maps onto the train/test era split.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    dob_col : str
        Column holding the date of birth (default: 'dob')
    year_col : str
        Column holding the season year (default: 'year')
    ref_month_day : str
        Month-day within that year to measure the age at (default: '12-31')
    out_col : str
        Name of the derived column (default: 'age')
    drop_dob : bool
        Remove the date of birth column afterwards (default: True)

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe with the age column added
    """
    import pandas as pd

    out = df.copy()

    dob = pd.to_datetime(out[dob_col], errors='coerce')
    reference = pd.to_datetime(
        out[year_col].astype(str) + '-' + ref_month_day, errors='coerce'
    )

    out[out_col] = (reference - dob).dt.days / 365.25

    if drop_dob:
        out = out.drop(columns=[dob_col])

    return out
