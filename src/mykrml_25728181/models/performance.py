def print_regressor_scores(y_preds, y_actuals, set_name=None):
    """Print the RMSE and MAE for the provided data

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    from sklearn.metrics import root_mean_squared_error as rmse
    from sklearn.metrics import mean_absolute_error as mae

    print(f"RMSE {set_name}: {rmse(y_actuals, y_preds)}")
    print(f"MAE {set_name}: {mae(y_actuals, y_preds)}")




def print_classifier_scores(y_preds, y_actuals, set_name=None):
    """Print the Accuracy and F1 score for the provided data.
    The value of the 'average' parameter for F1 score will be determined according to the number of distinct values of the target variable: 'binary' for bianry classification' or 'weighted' for multi-classs classification

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed
    Returns
    -------
    """
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    import pandas as pd

    average = 'weighted' if pd.Series(y_actuals).nunique() > 2 else 'binary'

    print(f"Accuracy {set_name}: {accuracy_score(y_actuals, y_preds)}")
    print(f"F1 {set_name}: {f1_score(y_actuals, y_preds, average=average)}")





def assess_classifier_set(model, features, target, set_name=''):
    """Save the predictions from a trained model on a given set and print its accuracy and F1 scores

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Trained Sklearn model with set hyperparameters
    features : Numpy Array
        Features
    target : Numpy Array
        Target variable
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    preds = model.predict(features)
    print_classifier_scores(y_preds=preds, y_actuals=target, set_name=set_name)





def fit_assess_classifier(model, X_train, y_train, X_val, y_val):
    """Train a classifier model, print its accuracy and F1 scores on the training and validation set and return the trained model

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Instantiated Sklearn model with set hyperparameters
    X_train : Numpy Array
        Features for the training set
    y_train : Numpy Array
        Target for the training set
    X_train : Numpy Array
        Features for the validation set
    y_train : Numpy Array
        Target for the validation set

    Returns
    sklearn.base.BaseEstimator
        Trained model
    -------
    """
    model.fit(X_train, y_train)
    assess_classifier_set(model, X_train, y_train, set_name='Training')
    assess_classifier_set(model, X_val, y_val, set_name='Validation')
    return model








def plot_precision_recall(y_actuals, y_probs, title="Precision-Recall Curve",
                          figsize=(8, 6), set_name=None, show=True):
    """Plot the precision-recall curve and report the area under it

    AUPRC is the right summary on an imbalanced problem. ROC-AUC is not: the
    large pool of true negatives means the false-positive rate barely moves as
    the threshold changes, so a model can post a strong ROC-AUC while most of
    what it flags is wrong.

    Parameters
    ----------
    y_actuals : Numpy Array
        Actual target
    y_probs : Numpy Array
        Predicted probability of the positive class
    title : str
        Plot title
    figsize : tuple
        Figure size in inches (default: (8, 6))
    set_name : str
        Name of the set, printed alongside the score
    show : bool
        Draw the plot (default: True). Set to False for the score alone

    Returns
    -------
    float
        The AUPRC score
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_actuals, y_probs)
    auprc = average_precision_score(y_actuals, y_probs)

    print(f"{set_name or 'Val'} AUPRC: {auprc:.4f}")

    if show:
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, label=f"AUPRC = {auprc:.4f}")
        plt.fill_between(recall, precision, alpha=0.2)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.legend(loc="best")
        plt.show()

    return auprc


def assess_auprc(model, features, target, set_name="Validation",
                 title="Precision-Recall Curve", show=True):
    """Score a fitted model on a set and plot its precision-recall curve

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Trained model exposing ``predict_proba``
    features : pd.DataFrame
        Features
    target : Numpy Array
        Target variable
    set_name : str
        Name of the set, printed alongside the score
    title : str
        Plot title
    show : bool
        Draw the plot (default: True)

    Returns
    -------
    float
        The AUPRC score
    Numpy Array
        Predicted probability of the positive class
    """
    y_probs = model.predict_proba(features)[:, 1]
    auprc = plot_precision_recall(target, y_probs, title=title,
                                  set_name=set_name, show=show)

    return auprc, y_probs


def shortlist_report(y_actuals, y_probs, sizes=(25, 50, 100, 200, 500),
                     round_to=3, verbose=True):
    """Precision, recall and lift at the top of a ranking

    This is the operational view of a triage model. What matters is not a
    single threshold but how dense the top of the ranking is compared with the
    base rate, because that is what decides whether a shortlist is worth
    working down.

    Parameters
    ----------
    y_actuals : Numpy Array
        Actual target
    y_probs : Numpy Array
        Predicted probability of the positive class
    sizes : tuple
        Shortlist sizes to report (default: (25, 50, 100, 200, 500))
    round_to : int
        Decimal places for the returned figures (default: 3)
    verbose : bool
        Print the base rate under the table

    Returns
    -------
    pd.DataFrame
        One row per shortlist size, with the count found, precision, recall
        and lift against the base rate
    """
    import pandas as pd

    ranked = pd.DataFrame({
        "actual": pd.Series(y_actuals).reset_index(drop=True),
        "score": pd.Series(y_probs).reset_index(drop=True),
    }).sort_values("score", ascending=False)

    base_rate = ranked["actual"].mean()
    total_positives = ranked["actual"].sum()

    rows = []
    for k in sizes:
        top = ranked.head(k)
        rows.append({
            "shortlist_size": k,
            "true_positives_found": int(top["actual"].sum()),
            "precision": top["actual"].mean(),
            "recall": top["actual"].sum() / total_positives,
            "lift_vs_base_rate": top["actual"].mean() / base_rate,
        })

    if verbose:
        print(f"Base rate: {base_rate:.4f} "
              f"({int(total_positives)} positives among {len(ranked)} rows)")

    return pd.DataFrame(rows).round(round_to)
