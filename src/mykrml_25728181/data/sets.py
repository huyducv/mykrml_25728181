import pandas as pd

def pop_target(df, target_col):
    """Extract target variable from dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    target_col : str
        Name of the target variable

    Returns
    -------
    pd.DataFrame
        Subsetted Pandas dataframe containing all features
    pd.Series
        Subsetted Pandas dataframe containing the target
    """

    df_copy = df.copy()
    target = df_copy.pop(target_col)

    return df_copy, target


def save_sets(
      X_train=None,
      y_train=None,
      X_val=None,
      y_val=None,
      X_test=None,
      y_test=None,
      path='../data/processed/',
):
    """Save the different sets locally

    Parameters
    ----------
    X_train: Pandas DataFrame
        Features for the training set
    y_train: Pandas DataFrame
        Target for the training set
    X_val: Pandas DataFrame
        Features for the validation set
    y_val: Pandas DataFrame
        Target for the validation set
    X_test: Pandas DataFrame
        Features for the testing set
    y_test: Pandas DataFrame
        Target for the testing set
    path : str
        Path to the folder where the sets will be saved (default: '../data/processed/')

    Returns
    -------
    """
    import pandas as pd

    if X_train is not None:
      X_train.to_csv(f'{path}X_train.csv', index=False)
    if X_val is not None:
      X_val.to_csv(f'{path}X_val.csv', index=False)
    if X_test is not None:
      X_test.to_csv(f'{path}X_test.csv', index=False)
    if y_train is not None:
      y_train.to_csv(f'{path}y_train.csv', index=False)
    if y_val is not None:
      y_val.to_csv(f'{path}y_val.csv', index=False)
    if y_test is not None:
      y_test.to_csv(f'{path}y_test.csv', index=False)





def load_sets(path='../data/processed/'):
    """Load the different locally save sets

    Parameters
    ----------
    path : str
        Path to the folder where the sets are saved (default: '../data/processed/')

    Returns
    -------
    Pandas DataFrame
        Features for the training set
    Pandas DataFrame
        Target for the training set
    Pandas DataFrame
        Features for the validation set
    Pandas DataFrame
        Target for the validation set
    Pandas DataFrame
        Features for the testing set
    Pandas DataFrame
        Target for the testing set
    """
    import pandas as pd
    import os.path

    X_train = pd.read_csv(f'{path}X_train.csv') if os.path.isfile(f'{path}X_train.csv') else None
    X_val   = pd.read_csv(f'{path}X_val.csv')   if os.path.isfile(f'{path}X_val.csv')   else None
    X_test  = pd.read_csv(f'{path}X_test.csv')  if os.path.isfile(f'{path}X_test.csv')  else None
    y_train = pd.read_csv(f'{path}y_train.csv') if os.path.isfile(f'{path}y_train.csv') else None
    y_val   = pd.read_csv(f'{path}y_val.csv')   if os.path.isfile(f'{path}y_val.csv')   else None
    y_test  = pd.read_csv(f'{path}y_test.csv')  if os.path.isfile(f'{path}y_test.csv')  else None

    return X_train, y_train, X_val, y_val, X_test, y_test





def subset_x_y(target, features, start_index:int, end_index:int):
    """Keep only the rows for X and y (optional) sets from the specified indexes

    Parameters
    ----------
    target : pd.DataFrame
        Dataframe containing the target
    features : pd.DataFrame
        Dataframe containing all features
    features : int
        Index of the starting observation
    features : int
        Index of the ending observation

    Returns
    -------
    pd.DataFrame
        Subsetted Pandas dataframe containing the target
    pd.DataFrame
        Subsetted Pandas dataframe containing all features
    """

    return features[start_index:end_index], target[start_index:end_index]




def split_sets_by_time(df, target_col, test_ratio=0.2):
    """Split sets by indexes for an ordered dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    test_ratio : float
        Ratio used for the validation and testing sets (default: 0.2)

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    """

    df_copy = df.copy()
    target = df_copy.pop(target_col)
    cutoff = int(len(df_copy) / 5)

    X_train, y_train = subset_x_y(target=target, features=df_copy, start_index=0, end_index=-cutoff*2)
    X_val, y_val     = subset_x_y(target=target, features=df_copy, start_index=-cutoff*2, end_index=-cutoff)
    X_test, y_test   = subset_x_y(target=target, features=df_copy, start_index=-cutoff, end_index=len(df_copy))

    return X_train, y_train, X_val, y_val, X_test, y_test



def split_sets_random(features, target, test_ratio=0.2):
    """Split sets randomly

    Parameters
    ----------
    features : pd.DataFrame
        Input dataframe
    target : pd.Series
        Target column
    test_ratio : float
        Ratio used for the validation and testing sets (default: 0.2)

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    """
    from sklearn.model_selection import train_test_split

    val_ratio = test_ratio / (1 - test_ratio)
    X_data, X_test, y_data, y_test = train_test_split(features, target, test_size=test_ratio, random_state=8)
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=val_ratio, random_state=8)

    return X_train, y_train, X_val, y_val, X_test, y_test




def split_train_val(features, target, test_size=0.2, random_state=42, stratify=True):
    """Split into a training and a validation set, stratified by default

    Stratification matters whenever the positive class is rare: an unstratified
    split can vary the positive count enough between the two sides to make the
    scores incomparable.

    Parameters
    ----------
    features : pd.DataFrame
        Input features
    target : pd.Series
        Target column
    test_size : float
        Proportion held out for validation (default: 0.2)
    random_state : int
        Seed for the split (default: 42)
    stratify : bool
        Preserve the target distribution across both sides (default: True)

    Returns
    -------
    pd.DataFrame
        Features for the training set
    pd.DataFrame
        Features for the validation set
    pd.Series
        Target for the training set
    pd.Series
        Target for the validation set
    """
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target if stratify else None,
    )

    print(f'Training data size: {X_train.shape}')
    print(f'Validation data size: {X_val.shape}')

    return X_train, X_val, y_train, y_val
