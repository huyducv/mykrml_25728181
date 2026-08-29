# Changelog

<!--next-version-placeholder-->

## v0.0.1 (27/08/2026)

- First release of `mykrml_25728181`!

## v0.0.2 (28/08/2026)

- Second release of `mykrml_25728181`: add pop_target() function

## v0.0.3 (29/08/2026)

- Third release of `mykrml_25728181`: add functionalities on scores and fit

## v0.1.0 (29/08/2026)

Fourth release of `mykrml_25728181`. Grows the package into the general-purpose
support library used by the AT1 experiment notebook.

Scope: only domain-neutral helpers belong here. Steps that make sense for one
dataset only - the season-relative standardisation, the player-level
aggregation, the submission writer - stay in the notebook that owns them.

Added:

- `data.cleaning`: `coerce_numeric`, `drop_duplicate_rows`
- `data.explore`: `missing_summary`, `target_summary`
- `data.sets`: `split_train_val`
- `features.dates`: `compute_age`
- `features.transform`: `map_ordinal`, `cast_categorical`, `object_to_category`,
  `make_category_caster`
- `features.selection`: `correlation_pairs`, `plot_correlation_heatmap`,
  `split_feature_types`
- `models.pipelines`: `build_preprocessors`
- `models.evaluation`: `stratified_cv`, `compare_models`,
  `confirm_top_candidates`, `holdout_comparison`
- `models.performance`: `plot_precision_recall`, `assess_auprc`,
  `shortlist_report`
- `__init__.py` files for the `data`, `features` and `models` subpackages, so
  each one re-exports its public functions

Changed:

- `requires-python` relaxed from `>=3.12.11` to `>=3.9`, and the runtime
  dependencies from exact pins to lower bounds, so the package installs into an
  existing analysis environment without downgrading it
- added `numpy`, `scipy` and `matplotlib` as dependencies

Removed:

- the `mykrml-25728181` console script, which pointed at a `main` that does not
  exist. This is a library, not a command-line tool
