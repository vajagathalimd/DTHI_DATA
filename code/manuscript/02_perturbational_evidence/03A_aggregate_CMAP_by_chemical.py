#!/usr/bin/env python3

from __future__ import annotations

from math import comb
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


PROJECT = Path(
    "."
)

INPUT_FILE = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6C/"
      "phase6C2B_CMAP_empirical_null_results.tsv.gz"
)

PROCESSED_DIR = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6C"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase6"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase6/"
      "phase6C3A_CMAP_chemical_aggregation.log"
)

FULL_OUTPUT = (
    PROCESSED_DIR
    / "phase6C3A_CMAP_chemical_level_aggregation.tsv.gz"
)

MATURATION_OUTPUT = (
    TABLE_DIR
    / "phase6C3A_maturation_supported_chemicals.tsv"
)

FETAL_OUTPUT = (
    TABLE_DIR
    / "phase6C3A_exploratory_fetal_supported_chemicals.tsv"
)

MIXED_OUTPUT = (
    TABLE_DIR
    / "phase6C3A_bidirectional_or_mixed_chemicals.tsv"
)

SUMMARY_OUTPUT = (
    TABLE_DIR
    / "phase6C3A_chemical_aggregation_summary.tsv"
)

COMPLETION_OUTPUT = (
    TABLE_DIR
    / "phase6C3A_completion_summary.tsv"
)

for directory in [
    PROCESSED_DIR,
    TABLE_DIR,
    LOG_FILE.parent,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def parse_boolean(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.fillna(
            False
        )

    normalized = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {
            "true",
            "t",
            "1",
            "yes",
            "y",
        }
    )


def normalize_text(
    value: object,
) -> str:
    if pd.isna(
        value
    ):
        return ""

    value = str(
        value
    ).strip()

    if value.lower() in {
        "",
        "nan",
        "none",
        "na",
    }:
        return ""

    return value


def normalize_chemical_name(
    value: object,
) -> str:
    value = normalize_text(
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def extract_concentration(
    parent: object,
    chemical_name: object,
) -> str:
    parent_text = normalize_text(
        parent
    )

    chemical_text = normalize_chemical_name(
        chemical_name
    )

    if not parent_text:
        return ""

    remainder = re.sub(
        r"^CMAP\s+",
        "",
        parent_text,
        flags=re.IGNORECASE,
    )

    if chemical_text and remainder.lower().startswith(
        chemical_text.lower()
    ):
        remainder = remainder[
            len(
                chemical_text
            ):
        ].strip()

    tokens = remainder.split()

    if not tokens:
        return ""

    candidate = tokens[0]

    try:
        float(
            candidate
        )
    except ValueError:
        return ""

    return candidate


def exact_upper_binomial_probability(
    successes: int,
    trials: int,
) -> float:
    if trials <= 0:
        return 1.0

    numerator = sum(
        comb(
            trials,
            count,
        )
        for count in range(
            successes,
            trials + 1,
        )
    )

    return numerator / (
        2 ** trials
    )


def bh_adjust(
    values: pd.Series,
) -> pd.Series:
    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    output = pd.Series(
        np.nan,
        index=values.index,
        dtype=float,
    )

    valid = values.notna()

    if not valid.any():
        return output

    valid_values = values.loc[
        valid
    ].to_numpy(
        dtype=float
    )

    order = np.argsort(
        valid_values
    )

    ranked = valid_values[
        order
    ]

    number = len(
        ranked
    )

    adjusted_ranked = (
        ranked
        * number
        / np.arange(
            1,
            number + 1,
        )
    )

    adjusted_ranked = np.minimum.accumulate(
        adjusted_ranked[::-1]
    )[::-1]

    adjusted_ranked = np.clip(
        adjusted_ranked,
        0,
        1,
    )

    restored = np.empty(
        number,
        dtype=float,
    )

    restored[
        order
    ] = adjusted_ranked

    output.loc[
        valid
    ] = restored

    return output


def join_unique(
    values: pd.Series,
    maximum: int | None = None,
) -> str:
    cleaned = sorted(
        {
            normalize_text(
                value
            )
            for value in values
            if normalize_text(
                value
            )
        }
    )

    if maximum is not None:
        cleaned = cleaned[
            :maximum
        ]

    return "|".join(
        cleaned
    )


def safe_min(
    series: pd.Series,
) -> float:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return float(
        values.min()
    )


def safe_median(
    series: pd.Series,
) -> float:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return float(
        values.median()
    )


def safe_max(
    series: pd.Series,
) -> float:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return float(
        values.max()
    )


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing Phase 6C2B input: {INPUT_FILE}"
        )

    data = pd.read_csv(
        INPUT_FILE,
        sep="\t",
        low_memory=False,
    )

    required_columns = {
        "CMAP_parent",
        "chemical_name",
        "DTXSID",
        "developmental_direction_score",
        "maturation_up_preference_log2OR",
        "fetal_down_preference_log2OR",
        "empirical_maturation_p",
        "empirical_maturation_FDR",
        "empirical_fetal_p",
        "empirical_fetal_FDR",
        "maxT_maturation_FWER_p",
        "maxT_fetal_FWER_p",
        "maturation_component_consistent",
        "fetal_component_consistent",
    }

    missing = sorted(
        required_columns
        - set(
            data.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(
                missing
            )
        )

    numeric_columns = [
        "developmental_direction_score",
        "maturation_up_preference_log2OR",
        "fetal_down_preference_log2OR",
        "empirical_maturation_p",
        "empirical_maturation_FDR",
        "empirical_fetal_p",
        "empirical_fetal_FDR",
        "maxT_maturation_FWER_p",
        "maxT_fetal_FWER_p",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data[
        "maturation_component_consistent"
    ] = parse_boolean(
        data[
            "maturation_component_consistent"
        ]
    )

    data[
        "fetal_component_consistent"
    ] = parse_boolean(
        data[
            "fetal_component_consistent"
        ]
    )

    data[
        "chemical_name_clean"
    ] = data[
        "chemical_name"
    ].map(
        normalize_chemical_name
    )

    data[
        "DTXSID_clean"
    ] = data[
        "DTXSID"
    ].map(
        normalize_text
    )

    data[
        "chemical_key"
    ] = np.where(
        data[
            "DTXSID_clean"
        ].ne(""),
        data[
            "DTXSID_clean"
        ],
        data[
            "chemical_name_clean"
        ].str.lower(),
    )

    data[
        "concentration_token"
    ] = [
        extract_concentration(
            parent,
            chemical,
        )
        for parent, chemical in zip(
            data[
                "CMAP_parent"
            ],
            data[
                "chemical_name_clean"
            ],
        )
    ]

    data[
        "positive_direction"
    ] = data[
        "developmental_direction_score"
    ].gt(0)

    data[
        "negative_direction"
    ] = data[
        "developmental_direction_score"
    ].lt(0)

    data[
        "maturation_empirical_parent"
    ] = (
        data[
            "positive_direction"
        ]
        & data[
            "maturation_component_consistent"
        ]
        & data[
            "empirical_maturation_FDR"
        ].lt(0.05)
    )

    data[
        "maturation_maxT_parent"
    ] = (
        data[
            "positive_direction"
        ]
        & data[
            "maturation_component_consistent"
        ]
        & data[
            "maxT_maturation_FWER_p"
        ].lt(0.05)
    )

    data[
        "fetal_empirical_parent"
    ] = (
        data[
            "negative_direction"
        ]
        & data[
            "fetal_component_consistent"
        ]
        & data[
            "empirical_fetal_FDR"
        ].lt(0.05)
    )

    data[
        "fetal_maxT_parent"
    ] = (
        data[
            "negative_direction"
        ]
        & data[
            "fetal_component_consistent"
        ]
        & data[
            "maxT_fetal_FWER_p"
        ].lt(0.05)
    )

    valid = data[
        "chemical_key"
    ].ne("")

    excluded_missing_key = int(
        (
            ~valid
        ).sum()
    )

    data = data.loc[
        valid
    ].copy()

    aggregation_rows = []

    for chemical_key, group in data.groupby(
        "chemical_key",
        sort=True,
        dropna=False,
    ):
        total = len(
            group
        )

        positive_count = int(
            group[
                "positive_direction"
            ].sum()
        )

        negative_count = int(
            group[
                "negative_direction"
            ].sum()
        )

        nonzero_count = (
            positive_count
            + negative_count
        )

        positive_fraction = (
            positive_count
            / nonzero_count
            if nonzero_count > 0
            else np.nan
        )

        negative_fraction = (
            negative_count
            / nonzero_count
            if nonzero_count > 0
            else np.nan
        )

        maturation_empirical_count = int(
            group[
                "maturation_empirical_parent"
            ].sum()
        )

        maturation_maxT_count = int(
            group[
                "maturation_maxT_parent"
            ].sum()
        )

        fetal_empirical_count = int(
            group[
                "fetal_empirical_parent"
            ].sum()
        )

        fetal_maxT_count = int(
            group[
                "fetal_maxT_parent"
            ].sum()
        )

        aggregation_rows.append(
            {
                "chemical_key":
                    chemical_key,

                "chemical_name":
                    join_unique(
                        group[
                            "chemical_name_clean"
                        ]
                    ),

                "DTXSID":
                    join_unique(
                        group[
                            "DTXSID_clean"
                        ]
                    ),

                "total_CMAP_parents":
                    total,

                "unique_concentrations":
                    len(
                        {
                            value
                            for value in group[
                                "concentration_token"
                            ]
                            if value
                        }
                    ),

                "concentration_tokens":
                    join_unique(
                        group[
                            "concentration_token"
                        ]
                    ),

                "positive_direction_parents":
                    positive_count,

                "negative_direction_parents":
                    negative_count,

                "zero_direction_parents":
                    total
                    - nonzero_count,

                "positive_direction_fraction":
                    positive_fraction,

                "negative_direction_fraction":
                    negative_fraction,

                "positive_sign_test_p":
                    exact_upper_binomial_probability(
                        positive_count,
                        nonzero_count,
                    ),

                "negative_sign_test_p":
                    exact_upper_binomial_probability(
                        negative_count,
                        nonzero_count,
                    ),

                "minimum_direction_score":
                    safe_min(
                        group[
                            "developmental_direction_score"
                        ]
                    ),

                "median_direction_score":
                    safe_median(
                        group[
                            "developmental_direction_score"
                        ]
                    ),

                "maximum_direction_score":
                    safe_max(
                        group[
                            "developmental_direction_score"
                        ]
                    ),

                "median_maturation_component_log2OR":
                    safe_median(
                        group[
                            "maturation_up_preference_log2OR"
                        ]
                    ),

                "median_fetal_down_component_log2OR":
                    safe_median(
                        group[
                            "fetal_down_preference_log2OR"
                        ]
                    ),

                "maturation_empirical_FDR_parent_count":
                    maturation_empirical_count,

                "maturation_maxT_FWER_parent_count":
                    maturation_maxT_count,

                "fetal_empirical_FDR_parent_count":
                    fetal_empirical_count,

                "fetal_maxT_FWER_parent_count":
                    fetal_maxT_count,

                "minimum_empirical_maturation_p":
                    safe_min(
                        group[
                            "empirical_maturation_p"
                        ]
                    ),

                "minimum_empirical_maturation_FDR":
                    safe_min(
                        group[
                            "empirical_maturation_FDR"
                        ]
                    ),

                "minimum_maxT_maturation_FWER_p":
                    safe_min(
                        group[
                            "maxT_maturation_FWER_p"
                        ]
                    ),

                "minimum_empirical_fetal_p":
                    safe_min(
                        group[
                            "empirical_fetal_p"
                        ]
                    ),

                "minimum_empirical_fetal_FDR":
                    safe_min(
                        group[
                            "empirical_fetal_FDR"
                        ]
                    ),

                "minimum_maxT_fetal_FWER_p":
                    safe_min(
                        group[
                            "maxT_fetal_FWER_p"
                        ]
                    ),

                "maturation_empirical_parents":
                    join_unique(
                        group.loc[
                            group[
                                "maturation_empirical_parent"
                            ],
                            "CMAP_parent",
                        ],
                        maximum=30,
                    ),

                "maturation_maxT_parents":
                    join_unique(
                        group.loc[
                            group[
                                "maturation_maxT_parent"
                            ],
                            "CMAP_parent",
                        ],
                        maximum=30,
                    ),

                "fetal_empirical_parents":
                    join_unique(
                        group.loc[
                            group[
                                "fetal_empirical_parent"
                            ],
                            "CMAP_parent",
                        ],
                        maximum=30,
                    ),
            }
        )

    chemical_table = pd.DataFrame(
        aggregation_rows
    )

    chemical_table[
        "positive_sign_test_FDR"
    ] = bh_adjust(
        chemical_table[
            "positive_sign_test_p"
        ]
    )

    chemical_table[
        "negative_sign_test_FDR"
    ] = bh_adjust(
        chemical_table[
            "negative_sign_test_p"
        ]
    )

    chemical_table[
        "support_class"
    ] = np.select(
        [
            chemical_table[
                "maturation_maxT_FWER_parent_count"
            ].ge(2),

            chemical_table[
                "maturation_maxT_FWER_parent_count"
            ].eq(1),

            chemical_table[
                "maturation_empirical_FDR_parent_count"
            ].ge(2),

            chemical_table[
                "fetal_empirical_FDR_parent_count"
            ].ge(2),
        ],
        [
            "maturation_maxT_replicated",
            "maturation_maxT_single",
            "maturation_empirical_replicated_only",
            "fetal_empirical_replicated_only",
        ],
        default="unresolved_or_single_empirical",
    )

    chemical_table[
        "bidirectional_empirical_support"
    ] = (
        chemical_table[
            "maturation_empirical_FDR_parent_count"
        ].gt(0)
        & chemical_table[
            "fetal_empirical_FDR_parent_count"
        ].gt(0)
    )

    chemical_table = chemical_table.sort_values(
        [
            "maturation_maxT_FWER_parent_count",
            "maturation_empirical_FDR_parent_count",
            "positive_direction_fraction",
            "median_direction_score",
            "chemical_name",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    chemical_table.to_csv(
        FULL_OUTPUT,
        sep="\t",
        index=False,
        compression="gzip",
    )

    maturation_table = chemical_table.loc[
        chemical_table[
            "maturation_maxT_FWER_parent_count"
        ].gt(0)
        | chemical_table[
            "maturation_empirical_FDR_parent_count"
        ].ge(2)
    ].copy()

    maturation_table = maturation_table.sort_values(
        [
            "maturation_maxT_FWER_parent_count",
            "maturation_empirical_FDR_parent_count",
            "positive_direction_fraction",
            "median_direction_score",
            "minimum_maxT_maturation_FWER_p",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
        ],
    )

    maturation_table.to_csv(
        MATURATION_OUTPUT,
        sep="\t",
        index=False,
    )

    fetal_table = chemical_table.loc[
        chemical_table[
            "fetal_empirical_FDR_parent_count"
        ].gt(0)
    ].copy()

    fetal_table = fetal_table.sort_values(
        [
            "fetal_empirical_FDR_parent_count",
            "negative_direction_fraction",
            "median_direction_score",
            "minimum_empirical_fetal_FDR",
        ],
        ascending=[
            False,
            False,
            True,
            True,
        ],
    )

    fetal_table.to_csv(
        FETAL_OUTPUT,
        sep="\t",
        index=False,
    )

    mixed_table = chemical_table.loc[
        chemical_table[
            "bidirectional_empirical_support"
        ]
    ].copy()

    mixed_table.to_csv(
        MIXED_OUTPUT,
        sep="\t",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "CMAP_parent_rows":
                    len(
                        data
                    ),

                "chemical_level_entities":
                    len(
                        chemical_table
                    ),

                "parents_excluded_missing_chemical_key":
                    excluded_missing_key,

                "chemicals_with_any_empirical_maturation_parent":
                    int(
                        chemical_table[
                            "maturation_empirical_FDR_parent_count"
                        ].gt(0).sum()
                    ),

                "chemicals_with_replicated_empirical_maturation_parents":
                    int(
                        chemical_table[
                            "maturation_empirical_FDR_parent_count"
                        ].ge(2).sum()
                    ),

                "chemicals_with_any_maxT_maturation_parent":
                    int(
                        chemical_table[
                            "maturation_maxT_FWER_parent_count"
                        ].gt(0).sum()
                    ),

                "chemicals_with_replicated_maxT_maturation_parents":
                    int(
                        chemical_table[
                            "maturation_maxT_FWER_parent_count"
                        ].ge(2).sum()
                    ),

                "chemicals_with_any_empirical_fetal_parent":
                    int(
                        chemical_table[
                            "fetal_empirical_FDR_parent_count"
                        ].gt(0).sum()
                    ),

                "chemicals_with_replicated_empirical_fetal_parents":
                    int(
                        chemical_table[
                            "fetal_empirical_FDR_parent_count"
                        ].ge(2).sum()
                    ),

                "chemicals_with_maxT_fetal_parent":
                    int(
                        chemical_table[
                            "fetal_maxT_FWER_parent_count"
                        ].gt(0).sum()
                    ),

                "chemicals_with_bidirectional_empirical_support":
                    int(
                        chemical_table[
                            "bidirectional_empirical_support"
                        ].sum()
                    ),
            }
        ]
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        sep="\t",
        index=False,
    )

    completion = pd.DataFrame(
        [
            {
                "input_CMAP_parents":
                    len(
                        data
                    ),

                "chemical_entities_created":
                    len(
                        chemical_table
                    ),

                "DTXSID_used_as_primary_identifier":
                    True,

                "chemical_name_fallback_used":
                    True,

                "condition_level_results_retained":
                    True,

                "chemical_level_sign_tests_calculated":
                    True,

                "Phase6C3A_status":
                    "completed",
            }
        ]
    )

    completion.to_csv(
        COMPLETION_OUTPUT,
        sep="\t",
        index=False,
    )

    log_text = "\n".join(
        [
            "===== PHASE 6C3A SUMMARY =====",
            summary.to_string(
                index=False
            ),
            "",
            "===== SUPPORT CLASSES =====",
            chemical_table[
                "support_class"
            ]
            .value_counts(
                dropna=False
            )
            .rename_axis(
                "support_class"
            )
            .reset_index(
                name="chemical_count"
            )
            .to_string(
                index=False
            ),
            "",
            "===== COMPLETION =====",
            completion.to_string(
                index=False
            ),
        ]
    )

    LOG_FILE.write_text(
        log_text + "\n",
        encoding="utf-8",
    )

    print(
        log_text
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            (
                "Phase 6C3A failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
