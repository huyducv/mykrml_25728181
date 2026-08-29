"""Tests for the behaviours that are load-bearing or easy to break.

Deliberately not one test per function: a test that only restates a two-line
wrapper in a second syntax costs reading time and catches nothing. What is
covered here is the behaviour something downstream actually depends on.
"""

import numpy as np
import pandas as pd
import pytest

from mykrml_25728181.data.cleaning import drop_duplicate_rows
from mykrml_25728181.data.explore import target_summary
from mykrml_25728181.features.dates import compute_age
from mykrml_25728181.features.selection import correlation_pairs
from mykrml_25728181.features.transform import make_category_caster, map_ordinal
from mykrml_25728181.models.evaluation import stratified_cv
from mykrml_25728181.models.performance import shortlist_report


@pytest.fixture
def frame():
    return pd.DataFrame({
        "year": [2010, 2011, 2010, 2011],
        "yr": ["Fr", "So", "Jr", "Sr"],
        "dob": ["1990-10-15", "1990-10-15", "1989-10-15", "1989-10-15"],
    })


def test_compute_age_measures_against_the_row_year_not_today(frame):
    """The reference date must come from the data, not from the clock.

    Using the current date would make the feature change on every re-run and
    quietly break reproducibility.
    """
    out = compute_age(frame, ref_month_day="04-30")

    assert "dob" not in out.columns
    # born 1990-10-15, measured at 2010-04-30 -> 19.54 years
    assert out["age"].iloc[0] == pytest.approx(19.54, abs=0.01)
    # same player, next season -> exactly one year older
    assert out["age"].iloc[1] - out["age"].iloc[0] == pytest.approx(1.0, abs=0.01)


def test_map_ordinal_orders_known_labels_and_nulls_unknown_ones(frame):
    """Unknown labels must become NaN rather than raise or pass through.

    Corrupted category values are removed by not appearing in the mapping, so
    this behaviour is what keeps them out of the model.
    """
    mapping = {"Fr": 1, "So": 2, "Jr": 3, "Sr": 4}

    assert map_ordinal(frame, "yr", mapping)["yr"].tolist() == [1, 2, 3, 4]

    corrupted = pd.DataFrame({"yr": ["0", "57.1", "Fr"]})
    mapped = map_ordinal(corrupted, "yr", mapping)["yr"]

    assert mapped.isna().tolist() == [True, True, False]
    assert mapped.iloc[2] == 1


def test_make_category_caster_is_picklable():
    """Parallel cross-validation pickles every pipeline step.

    A closure or lambda would work in a notebook and then fail only under
    n_jobs=-1, which is the worst way to find out.
    """
    import pickle

    caster = pickle.loads(pickle.dumps(make_category_caster()))
    out = caster.fit_transform(pd.DataFrame({"a": ["x", "y"], "b": [1, 2]}))

    assert str(out["a"].dtype) == "category"
    assert str(out["b"].dtype) != "category"


def test_correlation_pairs_reports_each_pair_once():
    """Only the upper triangle counts - otherwise every pair is double-counted
    and each feature is reported as correlated with itself."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    df = pd.DataFrame({"a": a, "b": a * 2 + 1e-9, "c": rng.normal(size=200)})

    pairs = correlation_pairs(df, threshold=0.95, verbose=False)

    assert len(pairs) == 1
    assert set(pairs.index[0]) == {"a", "b"}


def test_drop_duplicate_rows_distinguishes_whole_rows_from_a_key():
    """Two different operations: identical records, versus a repeated key that
    would be double-counted by a later group-by."""
    df = pd.DataFrame({"pid": [1, 1, 2], "year": [2010, 2010, 2010], "x": [1, 2, 3]})

    assert len(drop_duplicate_rows(df, verbose=False)) == 3
    assert len(drop_duplicate_rows(df, subset=["pid", "year"], verbose=False)) == 2


def test_target_summary_reports_the_no_skill_baseline():
    """On an imbalanced target the no-skill AUPRC equals the positive rate.
    Every claim about lift is measured against this number."""
    stats = target_summary(pd.Series([0] * 97 + [1] * 3), verbose=False)

    assert stats["positives"] == 3
    assert stats["positive_rate"] == pytest.approx(0.03)
    assert stats["no_skill_auprc"] == stats["positive_rate"]
    assert stats["majority_accuracy"] == pytest.approx(0.97)


def test_stratified_cv_returns_one_score_per_repeated_fold():
    """n_splits x n_repeats scores, not n_splits - the whole point of the
    repeated protocol is the larger sample of folds."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("abc"))
    y = pd.Series([0, 1] * 60)

    scores = stratified_cv(LogisticRegression(), X, y, n_splits=3, n_repeats=2,
                           n_jobs=1, verbose=False)

    assert len(scores) == 6


def test_shortlist_report_ranks_before_taking_the_top_k():
    """If it did not sort descending, the top-k would be an arbitrary slice and
    every precision and lift figure would be meaningless."""
    y = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0])
    probs = np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1])

    out = shortlist_report(y, probs, sizes=(2, 5), verbose=False)

    assert out.loc[0, "true_positives_found"] == 2
    assert out.loc[0, "precision"] == 1.0
    assert out.loc[0, "recall"] == 1.0
    assert out.loc[0, "lift_vs_base_rate"] == 5.0
