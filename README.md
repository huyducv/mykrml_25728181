# mykrml_25728181

General-purpose helpers for a tabular machine learning workflow: cleaning,
building the preprocessing variants that different algorithms need, and
measuring models on a cross-validation protocol that can resolve small
differences.

Everything here is deliberately domain-neutral. Steps that only make sense for
one dataset stay in that project's notebook; what lives in this package is the
work that gets repeated across projects, and - more importantly - repeated once
per dataframe within a project, which is exactly where a training frame and a
testing frame drift apart.

## Installation

```bash
$ pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mykrml_25728181
```

## Usage

Three subpackages: `data`, `features`, `models`.

### Cleaning and exploration

```python
from mykrml_25728181.data.cleaning import coerce_numeric, drop_duplicate_rows
from mykrml_25728181.data.explore import missing_summary, target_summary

df = coerce_numeric(df, 'num')                          # text -> numeric, bad values to NaN
df = drop_duplicate_rows(df, name='train')              # reports what it removed
df = drop_duplicate_rows(df, subset=['id', 'year'], name='train')

missing_summary({'train': df_train, 'test': df_test})   # % missing, side by side
target_summary(y)                                       # class balance and its baselines
```

### Feature preparation

```python
from mykrml_25728181.features.dates import compute_age
from mykrml_25728181.features.transform import (
    map_ordinal, cast_categorical, make_category_caster
)

df = compute_age(df, dob_col='dob', year_col='year')    # age at year end, not at run time
df = map_ordinal(df, 'level', {'low': 1, 'medium': 2, 'high': 3})
df = cast_categorical(df, ['role', 'region'])
```

`compute_age` measures against a date built from each row's own year rather than
the date the notebook runs, so re-running it later does not silently change the
feature.

### Feature selection

```python
from mykrml_25728181.features.selection import (
    correlation_pairs, plot_correlation_heatmap, split_feature_types
)

correlation_pairs(X, threshold=0.95)                    # each pair once, sorted
plot_correlation_heatmap(X)                             # returns the matrix it plots
num_cols, cat_cols = split_feature_types(X, cat_cols=['role', 'region'])
```

### Preprocessing and modelling

```python
from mykrml_25728181.models.pipelines import build_preprocessors
from mykrml_25728181.models.evaluation import stratified_cv, compare_models

pre = build_preprocessors(num_cols, cat_cols)
pipeline = Pipeline([('preprocess', pre.native_missing),
                     ('cat_caster', make_category_caster()),
                     ('model', SomeClassifier())])

scores = stratified_cv(pipeline, X_train, y_train)                 # 5x5, mean +/- SE
other  = stratified_cv(variant, X_train, y_train, baseline=scores) # paired per-fold test
table, folds = compare_models({'A': pipeline, 'B': variant}, X_train, y_train)
```

`build_preprocessors` returns four variants - `ohe_scaler`, `ohe`, `basic` and
`native_missing` - because a linear model, a tree model and a booster that
handles NaN natively each need the data prepared differently. It is a named
tuple, so unpack it positionally or address it by name.

`stratified_cv` defaults to 5x5 repeated cross-validation rather than a single
split. On a small positive class a single 5-fold has a standard error wide
enough to swallow the effects being compared, so every comparison decides
nothing; passing `baseline` adds the paired per-fold test that makes small
differences readable.

`confirm_top_candidates` exists for the same reason: a search screens on one
split, so `best_params_` is partly a draw from noise. It re-scores the top few
candidates on the repeated protocol and reports how much of each screening score
was real.

### Scoring

```python
from mykrml_25728181.models.performance import assess_auprc, shortlist_report

auprc, probs = assess_auprc(model, X_val, y_val)     # curve plus AUPRC
shortlist_report(y_val, probs)                       # precision, recall and lift at top-k
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`mykrml_25728181` was created by huyducvu. It is licensed under the terms of the MIT license.

## Credits

`mykrml_25728181` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
