"""Preprocessing pipelines for a mixed numeric / categorical feature matrix.

Different algorithms need the same data prepared differently - a linear model
needs complete scaled numbers, a tree model does not care about scale, and a
gradient booster that handles NaN natively is actively harmed by imputation.
Building all the variants from one call keeps the column lists in sync, which
is where this kind of code usually goes wrong.
"""

from collections import namedtuple

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#: The four preprocessing variants, returned together so they can be unpacked
#: positionally or addressed by name.
Preprocessors = namedtuple(
    "Preprocessors", ["ohe_scaler", "ohe", "basic", "native_missing"]
)


def build_preprocessors(num_cols, cat_cols, num_strategy="median",
                        cat_strategy="most_frequent"):
    """Build the four preprocessing variants the candidate algorithms need

    Note the ordering inside each pipeline: imputation runs *before* scaling
    and before encoding. In the reverse order a one-hot encoder turns NaN into
    its own column and the imputer that follows has nothing left to do.

    The variants are:

    - ``ohe_scaler`` - impute, scale, one-hot encode. For linear models, which
      need complete, scaled, numeric input.
    - ``ohe`` - impute and one-hot encode without scaling. For tree models,
      which are scale-invariant.
    - ``basic`` - impute only, leaving categoricals as categories for a model
      with native categorical handling.
    - ``native_missing`` - numeric passed through untouched so the model sees
      the NaNs itself, with only the categoricals imputed.

    Parameters
    ----------
    num_cols : list
        Numeric column names
    cat_cols : list
        Categorical column names
    num_strategy : str
        Imputation strategy for numeric columns (default: 'median')
    cat_strategy : str
        Imputation strategy for categorical columns (default: 'most_frequent')

    Returns
    -------
    Preprocessors
        Named tuple of four fitted-ready ColumnTransformers, in the order
        ohe_scaler, ohe, basic, native_missing
    """
    num_impute_scale = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=num_strategy)),
        ("scaler", StandardScaler()),
    ])

    num_impute = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=num_strategy)),
    ])

    cat_impute_encode = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=cat_strategy)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    cat_impute = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=cat_strategy)),
    ])

    def column_transformer(num_step, cat_step, remainder="drop"):
        return ColumnTransformer(
            transformers=[
                ("num", num_step, num_cols),
                ("cat", cat_step, cat_cols),
            ],
            remainder=remainder,
        ).set_output(transform="pandas")

    return Preprocessors(
        ohe_scaler=column_transformer(num_impute_scale, cat_impute_encode),
        ohe=column_transformer(num_impute, cat_impute_encode),
        basic=column_transformer(num_impute, cat_impute),
        native_missing=column_transformer("passthrough", cat_impute),
    )
