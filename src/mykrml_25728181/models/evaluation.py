"""Cross-validation protocols that can resolve small differences.

A single 5-fold split is the default habit, but on a small positive class its
standard error is often wider than the effects being compared, so every result
sits inside the noise band and a comparison decides nothing. Repeating the
split and testing the folds pairwise is what makes small differences readable.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


def stratified_cv(pipeline, X, y, n_splits=5, n_repeats=5, random_state=42,
                  scoring="average_precision", baseline=None, n_jobs=-1,
                  verbose=True):
    """Repeated stratified cross-validation reporting the mean and standard error

    Pass ``baseline`` - the per-fold array of the incumbent, measured with the
    same arguments - to get a paired per-fold comparison. At small effect sizes
    that is the only way to tell a real gain from fold luck, because the two
    arrays share their folds and the difference cancels the fold-to-fold
    variation that otherwise swamps the effect.

    Note that any estimator inside the pipeline should be left single-threaded:
    the folds are already parallelised here, and an inner ``n_jobs=-1`` would
    oversubscribe the machine.

    Parameters
    ----------
    pipeline : sklearn.base.BaseEstimator
        Pipeline or estimator to evaluate
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    n_splits : int
        Folds per repeat (default: 5)
    n_repeats : int
        Number of repeats (default: 5)
    random_state : int
        Seed for the fold assignment (default: 42)
    scoring : str
        Scikit-learn scoring name (default: 'average_precision', i.e. AUPRC)
    baseline : Numpy Array
        Per-fold scores of the incumbent, for a paired comparison
    n_jobs : int
        Parallel jobs across folds (default: -1)
    verbose : bool
        Print the summary line and the paired comparison

    Returns
    -------
    Numpy Array
        One score per fold, n_splits * n_repeats of them
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    cv_score = cross_val_score(
        pipeline, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs
    )

    se = cv_score.std(ddof=1) / np.sqrt(len(cv_score))

    if verbose:
        print(f"AUPRC over {len(cv_score)} folds: "
              f"{cv_score.mean():.4f} +/- {se:.4f} (SE)")

    if baseline is not None:
        from scipy import stats

        d = cv_score - baseline
        _, p = stats.ttest_rel(cv_score, baseline)

        if verbose:
            print(f"Paired vs baseline: {d.mean():+.4f} "
                  f"(p={p:.3f}, wins {int((d > 0).sum())}/{len(d)})")

    return cv_score


def compare_models(models, X, y, n_splits=5, n_repeats=5, random_state=42,
                   scoring="average_precision", n_jobs=-1, verbose=True):
    """Score several candidate pipelines on identical folds and rank them

    Every model sees the same fold assignment, so the comparison is not
    confounded by which rows happened to land where.

    Parameters
    ----------
    models : dict
        Mapping of display name to pipeline
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    n_splits : int
        Folds per repeat (default: 5)
    n_repeats : int
        Number of repeats. Set to 1 for a plain single split (default: 5)
    random_state : int
        Seed for the fold assignment (default: 42)
    scoring : str
        Scikit-learn scoring name (default: 'average_precision')
    n_jobs : int
        Parallel jobs across folds (default: -1)
    verbose : bool
        Print progress and each model's summary line

    Returns
    -------
    pd.DataFrame
        One row per model - folds, mean, standard error, min and max - sorted
        by mean score descending
    dict
        Per-fold score array for each model, so a later candidate can be
        compared against one of them as a paired baseline
    """
    results = []
    fold_scores = {}

    for name, pipeline in models.items():
        if verbose:
            print("-" * 30)
            print(f"Running {name}...")

        scores = stratified_cv(
            pipeline, X, y,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        fold_scores[name] = scores

        results.append({
            "Model": name,
            "Folds": len(scores),
            "Mean AUPRC": scores.mean(),
            "SE AUPRC": scores.std(ddof=1) / np.sqrt(len(scores)),
            "Min AUPRC": scores.min(),
            "Max AUPRC": scores.max(),
        })

    results_df = pd.DataFrame(results).sort_values("Mean AUPRC", ascending=False)

    return results_df, fold_scores


def confirm_top_candidates(search, base_pipeline, X, y, top_n=3, baseline=None,
                           n_splits=5, n_repeats=5, random_state=42,
                           scoring="average_precision", n_jobs=-1):
    """Re-measure the best candidates of a search under the repeated protocol

    A hyperparameter search screens on a single split whose standard error is
    usually wider than the gaps between candidates, so ``best_params_`` is
    partly a draw from that noise and ``best_score_`` is optimistic. Re-scoring
    the top few on repeated cross-validation shows how much of each screening
    score was real - the shrinkage column - and picks the winner on the better
    estimate.

    Parameters
    ----------
    search : sklearn.model_selection.BaseSearchCV
        A fitted GridSearchCV or RandomizedSearchCV
    base_pipeline : sklearn.base.BaseEstimator
        The pipeline the search was run over; cloned for each candidate
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    top_n : int
        How many candidates to re-measure (default: 3)
    baseline : Numpy Array
        Per-fold scores of the incumbent, for the paired comparison. Computed
        from base_pipeline when not supplied
    n_splits : int
        Folds per repeat (default: 5)
    n_repeats : int
        Number of repeats (default: 5)
    random_state : int
        Seed for the fold assignment (default: 42)
    scoring : str
        Scikit-learn scoring name (default: 'average_precision')
    n_jobs : int
        Parallel jobs across folds (default: -1)

    Returns
    -------
    pd.DataFrame
        Screening score, confirmed score, shrinkage and parameters per candidate
    dict
        Parameters of the candidate that scored best after confirmation
    Numpy Array
        Per-fold scores of the incumbent
    """
    cv_kwargs = dict(n_splits=n_splits, n_repeats=n_repeats,
                     random_state=random_state, scoring=scoring, n_jobs=n_jobs)

    if baseline is None:
        print("Re-measuring the incumbent on the repeated protocol...")
        baseline = stratified_cv(base_pipeline, X, y, **cv_kwargs)

    cv_results = pd.DataFrame(search.cv_results_)
    top = cv_results.nlargest(top_n, "mean_test_score")[["mean_test_score", "params"]]

    confirmed = []

    for screen_score, params in top.itertuples(index=False):
        print(f"\nConfirming candidate screened at {screen_score:.4f} ...")

        candidate = clone(base_pipeline).set_params(**params)
        scores = stratified_cv(candidate, X, y, baseline=baseline, **cv_kwargs)

        confirmed.append({
            f"screen ({n_splits}-fold)": screen_score,
            f"confirmed ({n_splits}x{n_repeats})": scores.mean(),
            "shrinkage": scores.mean() - screen_score,
            "params": params,
        })

    confirm_df = pd.DataFrame(confirmed)
    best_params = max(
        confirmed, key=lambda r: r[f"confirmed ({n_splits}x{n_repeats})"]
    )["params"]

    return confirm_df, best_params, baseline


def holdout_comparison(models, X_val, y_val, scoring="average_precision",
                       fit_sets=None):
    """Score several fitted models on the same holdout set

    Parameters
    ----------
    models : dict
        Mapping of display name to a fitted estimator
    X_val : pd.DataFrame
        Holdout features
    y_val : pd.Series
        Holdout target
    scoring : str
        Either 'average_precision' (default) or 'roc_auc'
    fit_sets : tuple
        Optional ``(X_train, y_train)``. When given, each model is refit on it
        before being scored

    Returns
    -------
    pd.DataFrame
        One row per model, sorted by score descending
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    metric = average_precision_score if scoring == "average_precision" else roc_auc_score
    label = "Validation AUPRC" if scoring == "average_precision" else "Validation ROC-AUC"

    results = []

    for name, model in models.items():
        if fit_sets is not None:
            model.fit(*fit_sets)

        y_pred = model.predict_proba(X_val)[:, 1]
        results.append({"Model": name, label: metric(y_val, y_pred)})

    return pd.DataFrame(results).sort_values(label, ascending=False)
