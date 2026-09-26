"""Feature engineering: dates, column repair and feature selection."""

from mykrml_25728181.features.dates import compute_age, convert_to_date
from mykrml_25728181.features.selection import (
    correlation_pairs,
    plot_correlation_heatmap,
    split_feature_types,
)
from mykrml_25728181.features.time_series import (
    add_future_targets,
    build_time_series_features,
)
from mykrml_25728181.features.transform import (
    cast_categorical,
    make_category_caster,
    map_ordinal,
    object_to_category,
)

__all__ = [
    "convert_to_date",
    "compute_age",
    "map_ordinal",
    "cast_categorical",
    "object_to_category",
    "make_category_caster",
    "correlation_pairs",
    "plot_correlation_heatmap",
    "split_feature_types",
    "add_future_targets",
    "build_time_series_features",
]
