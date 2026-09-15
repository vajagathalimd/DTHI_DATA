#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT = Path(
    "."
)

COMPARISON_FILE = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6D/"
      "phase6D1_original_vs_leave_program_out_scores.tsv.gz"
)

MAXT_FILE = (
    PROJECT
    / "07_tables/main_tables/phase6/"
      "phase6C3D_maxT_maturation_parent_mechanism_summary.tsv"
)

PROCESSED_DIR = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6D"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase6"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase6/"
      "phase6D2_robust_residual_mechanism_classification.log"
)

PARENT_OUTPUT = (
    PROCESSED_DIR
    / "phase6D2_maxT_parent_robust_mechanism_classes.tsv.gz"
)

CHEMICAL_OUTPUT = (
    TABLE_DIR
    / "phase6D2_maxT_chemical_robustness_summary.tsv"
)

CLASS_OUTPUT = (
    TABLE_DIR
    / "phase6D2_robust_mechanism_class_summary.tsv"
)

CATEGORY_OUTPUT = (
    TABLE_DIR
    / "phase6D2_maxT_category_robustness_summary.tsv"
)

ROBUST_PARENT_OUTPUT = (
    TABLE_DIR
    / "phase6D2_globally_robust_maxT_parents.tsv"
)

RELATIVE_PARENT_OUTPUT = (
    TABLE_DIR
    / "phase6D2_relative_only_maxT_parents.tsv"
)

COMPLETION_OUTPUT = (
    TABLE_DIR
    / "phase6D2_completion_summary.tsv"
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


EARLY_CATEGORIES = [
    "early_proliferative_state",
    "fetal_neurodevelopmental_state",
]

LATE_CATEGORIES = [
    "late_synaptic_maturation",
    "mitochondrial_metabolic_maturation",
    "glial_myelin_maturation",
]

INJURY_CATEGORIES = [
    "DNA_damage_p53",
    "apoptosis_cytotoxicity",
    "oxidative_hypoxic_stress",
    "inflammatory_stress",
]

ALL_CATEGORIES = (
    EARLY_CATEGORIES
    + LATE_CATEGORIES
    + INJURY_CATEGORIES
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


def clean_text(
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


def chemical_key(
    row: pd.Series,
) -> str:
    dtxsid = clean_text(
        row.get(
            "DTXSID",
            "",
        )
    )

    if dtxsid:
        return dtxsid

    name = clean_text(
        row.get(
            "chemical_name",
            "",
        )
    ).lower()

    if name:
        return name

    return clean_text(
        row.get(
            "CMAP_parent",
            "",
        )
    ).lower()


def join_unique(
    values: pd.Series,
    maximum: int | None = None,
) -> str:
    cleaned = sorted(
        {
            clean_text(
                value
            )
            for value in values
            if clean_text(
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


def safe_median(
    values: pd.Series,
) -> float:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return float(
        values.median()
    )


def safe_minimum(
    values: pd.Series,
) -> float:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return float(
        values.min()
    )


def classify_parent(
    row: pd.Series,
) -> str:
    robust_early = int(
        row[
            "leaveout_global_early_count"
        ]
    )

    robust_late = int(
        row[
            "leaveout_global_late_count"
        ]
    )

    robust_injury = int(
        row[
            "leaveout_global_injury_count"
        ]
    )

    robust_global = (
        robust_early
        + robust_late
        + robust_injury
    )

    robust_relative = int(
        row[
            "leaveout_within_parent_total_count"
        ]
    )

    original_support = (
        int(
            row[
                "original_global_total_count"
            ]
        )
        + int(
            row[
                "original_within_parent_total_count"
            ]
        )
    )

    if robust_global > 0:
        if (
            robust_injury > 0
            and robust_early > 0
        ):
            return (
                "robust_mixed_early_and_injury"
            )

        if robust_injury > 0:
            return (
                "robust_injury_stress"
            )

        if robust_early > 0:
            return (
                "robust_early_program_suppression"
            )

        if robust_late > 0:
            return (
                "robust_late_maturation"
            )

        return (
            "robust_other_global"
        )

    if robust_relative > 0:
        return (
            "robust_relative_profile_only"
        )

    if original_support > 0:
        return (
            "program_overlap_dependent"
        )

    return (
        "no_fixed_panel_mechanism"
    )


def main() -> None:
    if not COMPARISON_FILE.exists():
        raise FileNotFoundError(
            f"Missing comparison file: {COMPARISON_FILE}"
        )

    if not MAXT_FILE.exists():
        raise FileNotFoundError(
            f"Missing maxT file: {MAXT_FILE}"
        )

    comparison = pd.read_csv(
        COMPARISON_FILE,
        sep="\t",
        low_memory=False,
    )

    max_t = pd.read_csv(
        MAXT_FILE,
        sep="\t",
        low_memory=False,
    )

    if len(
        max_t
    ) != 142:
        raise RuntimeError(
            (
                "Expected 142 maxT parents, "
                f"found {len(max_t)}."
            )
        )

    required_columns = {
        "CMAP_parent",
        "mechanism_category",
        "original_preferred_direction_log2OR",
        "leaveout_preferred_direction_log2OR",
        "original_aligned_category_FDR_significant",
        "leaveout_aligned_category_FDR_significant",
        "original_aligned_parent_profile_FDR_significant",
        "leaveout_aligned_parent_profile_FDR_significant",
    }

    missing = sorted(
        required_columns
        - set(
            comparison.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Missing comparison columns: "
            + ", ".join(
                missing
            )
        )

    boolean_columns = [
        "original_aligned_category_FDR_significant",
        "leaveout_aligned_category_FDR_significant",
        "original_aligned_parent_profile_FDR_significant",
        "leaveout_aligned_parent_profile_FDR_significant",
    ]

    for column in boolean_columns:
        comparison[column] = parse_boolean(
            comparison[column]
        )

    numeric_columns = [
        "original_preferred_direction_log2OR",
        "leaveout_preferred_direction_log2OR",
    ]

    for column in numeric_columns:
        comparison[column] = pd.to_numeric(
            comparison[column],
            errors="coerce",
        )

    selected = comparison.loc[
        comparison[
            "CMAP_parent"
        ].isin(
            max_t[
                "CMAP_parent"
            ]
        )
    ].copy()

    expected_rows = (
        142
        * 9
    )

    if len(
        selected
    ) != expected_rows:
        raise RuntimeError(
            (
                f"Expected {expected_rows} selected rows, "
                f"found {len(selected)}."
            )
        )

    if (
        selected[
            "mechanism_category"
        ].nunique()
        != 9
    ):
        raise RuntimeError(
            "Expected nine mechanism categories."
        )

    matrices = {}

    matrix_columns = {
        "original_global":
            "original_aligned_category_FDR_significant",

        "leaveout_global":
            "leaveout_aligned_category_FDR_significant",

        "original_within":
            "original_aligned_parent_profile_FDR_significant",

        "leaveout_within":
            "leaveout_aligned_parent_profile_FDR_significant",

        "original_score":
            "original_preferred_direction_log2OR",

        "leaveout_score":
            "leaveout_preferred_direction_log2OR",
    }

    for matrix_name, value_column in matrix_columns.items():
        matrices[
            matrix_name
        ] = selected.pivot(
            index="CMAP_parent",
            columns="mechanism_category",
            values=value_column,
        )

    result = max_t.copy().set_index(
        "CMAP_parent",
        drop=False,
    )

    for matrix_name, matrix in matrices.items():
        matrices[
            matrix_name
        ] = matrix.reindex(
            result.index
        )

    for category in ALL_CATEGORIES:
        if (
            category
            not in matrices[
                "leaveout_global"
            ].columns
        ):
            raise RuntimeError(
                f"Missing category: {category}"
            )

        result[
            f"original_global_{category}"
        ] = (
            matrices[
                "original_global"
            ][category]
            .fillna(
                False
            )
            .astype(
                bool
            )
        )

        result[
            f"leaveout_global_{category}"
        ] = (
            matrices[
                "leaveout_global"
            ][category]
            .fillna(
                False
            )
            .astype(
                bool
            )
        )

        result[
            f"original_within_parent_{category}"
        ] = (
            matrices[
                "original_within"
            ][category]
            .fillna(
                False
            )
            .astype(
                bool
            )
        )

        result[
            f"leaveout_within_parent_{category}"
        ] = (
            matrices[
                "leaveout_within"
            ][category]
            .fillna(
                False
            )
            .astype(
                bool
            )
        )

        result[
            f"original_{category}_log2OR"
        ] = matrices[
            "original_score"
        ][category]

        result[
            f"leaveout_{category}_log2OR"
        ] = matrices[
            "leaveout_score"
        ][category]

    result[
        "original_global_early_count"
    ] = sum(
        result[
            f"original_global_{category}"
        ].astype(
            int
        )
        for category in EARLY_CATEGORIES
    )

    result[
        "original_global_late_count"
    ] = sum(
        result[
            f"original_global_{category}"
        ].astype(
            int
        )
        for category in LATE_CATEGORIES
    )

    result[
        "original_global_injury_count"
    ] = sum(
        result[
            f"original_global_{category}"
        ].astype(
            int
        )
        for category in INJURY_CATEGORIES
    )

    result[
        "leaveout_global_early_count"
    ] = sum(
        result[
            f"leaveout_global_{category}"
        ].astype(
            int
        )
        for category in EARLY_CATEGORIES
    )

    result[
        "leaveout_global_late_count"
    ] = sum(
        result[
            f"leaveout_global_{category}"
        ].astype(
            int
        )
        for category in LATE_CATEGORIES
    )

    result[
        "leaveout_global_injury_count"
    ] = sum(
        result[
            f"leaveout_global_{category}"
        ].astype(
            int
        )
        for category in INJURY_CATEGORIES
    )

    result[
        "original_global_total_count"
    ] = (
        result[
            "original_global_early_count"
        ]
        + result[
            "original_global_late_count"
        ]
        + result[
            "original_global_injury_count"
        ]
    )

    result[
        "leaveout_global_total_count"
    ] = (
        result[
            "leaveout_global_early_count"
        ]
        + result[
            "leaveout_global_late_count"
        ]
        + result[
            "leaveout_global_injury_count"
        ]
    )

    result[
        "original_within_parent_total_count"
    ] = sum(
        result[
            f"original_within_parent_{category}"
        ].astype(
            int
        )
        for category in ALL_CATEGORIES
    )

    result[
        "leaveout_within_parent_total_count"
    ] = sum(
        result[
            f"leaveout_within_parent_{category}"
        ].astype(
            int
        )
        for category in ALL_CATEGORIES
    )

    result[
        "robust_mechanism_class"
    ] = result.apply(
        classify_parent,
        axis=1,
    )

    result[
        "chemical_key"
    ] = result.apply(
        chemical_key,
        axis=1,
    )

    result[
        "any_global_mechanism_retained"
    ] = result[
        "leaveout_global_total_count"
    ].gt(0)

    result[
        "any_within_parent_mechanism_retained"
    ] = result[
        "leaveout_within_parent_total_count"
    ].gt(0)

    result = result.reset_index(
        drop=True
    )

    result = result.sort_values(
        [
            "leaveout_global_total_count",
            "leaveout_within_parent_total_count",
            "leaveout_global_injury_count",
            "leaveout_global_early_count",
            "developmental_direction_score",
            "maxT_maturation_FWER_p",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            True,
        ],
    )

    result.to_csv(
        PARENT_OUTPUT,
        sep="\t",
        index=False,
        compression="gzip",
    )

    class_summary = (
        result.groupby(
            "robust_mechanism_class",
            dropna=False,
        )
        .agg(
            CMAP_parent_count=(
                "CMAP_parent",
                "size",
            ),

            chemical_entity_count=(
                "chemical_key",
                "nunique",
            ),

            median_developmental_direction_score=(
                "developmental_direction_score",
                "median",
            ),

            median_original_global_count=(
                "original_global_total_count",
                "median",
            ),

            median_leaveout_global_count=(
                "leaveout_global_total_count",
                "median",
            ),

            median_leaveout_within_parent_count=(
                "leaveout_within_parent_total_count",
                "median",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "CMAP_parent_count",
                "robust_mechanism_class",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    class_summary[
        "fraction_of_142"
    ] = (
        class_summary[
            "CMAP_parent_count"
        ]
        / len(
            result
        )
    )

    class_summary.to_csv(
        CLASS_OUTPUT,
        sep="\t",
        index=False,
    )

    category_rows = []

    for category in ALL_CATEGORIES:
        category_rows.append(
            {
                "mechanism_category":
                    category,

                "axis_group":
                    (
                        "early"
                        if category in EARLY_CATEGORIES
                        else (
                            "late"
                            if category in LATE_CATEGORIES
                            else "injury"
                        )
                    ),

                "maxT_parents":
                    len(
                        result
                    ),

                "original_global_count":
                    int(
                        result[
                            f"original_global_{category}"
                        ].sum()
                    ),

                "leaveout_global_count":
                    int(
                        result[
                            f"leaveout_global_{category}"
                        ].sum()
                    ),

                "retained_global_count":
                    int(
                        (
                            result[
                                f"original_global_{category}"
                            ]
                            & result[
                                f"leaveout_global_{category}"
                            ]
                        ).sum()
                    ),

                "lost_global_count":
                    int(
                        (
                            result[
                                f"original_global_{category}"
                            ]
                            & ~result[
                                f"leaveout_global_{category}"
                            ]
                        ).sum()
                    ),

                "gained_global_count":
                    int(
                        (
                            ~result[
                                f"original_global_{category}"
                            ]
                            & result[
                                f"leaveout_global_{category}"
                            ]
                        ).sum()
                    ),

                "leaveout_within_parent_count":
                    int(
                        result[
                            f"leaveout_within_parent_{category}"
                        ].sum()
                    ),

                "median_original_log2OR":
                    safe_median(
                        result[
                            f"original_{category}_log2OR"
                        ]
                    ),

                "median_leaveout_log2OR":
                    safe_median(
                        result[
                            f"leaveout_{category}_log2OR"
                        ]
                    ),
            }
        )

    category_summary = pd.DataFrame(
        category_rows
    )

    category_summary.to_csv(
        CATEGORY_OUTPUT,
        sep="\t",
        index=False,
    )

    chemical_rows = []

    for key, group in result.groupby(
        "chemical_key",
        sort=True,
        dropna=False,
    ):
        class_counts = (
            group[
                "robust_mechanism_class"
            ]
            .value_counts()
        )

        dominant_class = class_counts.index[
            0
        ]

        dominant_count = int(
            class_counts.iloc[
                0
            ]
        )

        chemical_rows.append(
            {
                "chemical_key":
                    key,

                "chemical_name":
                    join_unique(
                        group[
                            "chemical_name"
                        ]
                    ),

                "DTXSID":
                    join_unique(
                        group[
                            "DTXSID"
                        ]
                    ),

                "maxT_parent_count":
                    len(
                        group
                    ),

                "parents_with_global_robust_mechanism":
                    int(
                        group[
                            "leaveout_global_total_count"
                        ].gt(0).sum()
                    ),

                "parents_with_within_parent_robust_mechanism":
                    int(
                        group[
                            "leaveout_within_parent_total_count"
                        ].gt(0).sum()
                    ),

                "robust_injury_parent_count":
                    int(
                        group[
                            "robust_mechanism_class"
                        ].eq(
                            "robust_injury_stress"
                        ).sum()
                    ),

                "robust_mixed_parent_count":
                    int(
                        group[
                            "robust_mechanism_class"
                        ].eq(
                            "robust_mixed_early_and_injury"
                        ).sum()
                    ),

                "robust_early_suppression_parent_count":
                    int(
                        group[
                            "robust_mechanism_class"
                        ].eq(
                            "robust_early_program_suppression"
                        ).sum()
                    ),

                "robust_late_maturation_parent_count":
                    int(
                        group[
                            "robust_mechanism_class"
                        ].eq(
                            "robust_late_maturation"
                        ).sum()
                    ),

                "program_overlap_dependent_parent_count":
                    int(
                        group[
                            "robust_mechanism_class"
                        ].eq(
                            "program_overlap_dependent"
                        ).sum()
                    ),

                "no_fixed_panel_mechanism_parent_count":
                    int(
                        group[
                            "robust_mechanism_class"
                        ].eq(
                            "no_fixed_panel_mechanism"
                        ).sum()
                    ),

                "dominant_robust_mechanism_class":
                    dominant_class,

                "dominant_class_parent_count":
                    dominant_count,

                "dominant_class_fraction":
                    dominant_count
                    / len(
                        group
                    ),

                "robust_mechanism_classes":
                    "|".join(
                        (
                            class_counts.index
                            + ":"
                            + class_counts.astype(
                                str
                            )
                        ).tolist()
                    ),

                "median_developmental_direction_score":
                    safe_median(
                        group[
                            "developmental_direction_score"
                        ]
                    ),

                "minimum_maxT_maturation_FWER_p":
                    safe_minimum(
                        group[
                            "maxT_maturation_FWER_p"
                        ]
                    ),

                "CMAP_parents":
                    join_unique(
                        group[
                            "CMAP_parent"
                        ],
                        maximum=40,
                    ),
            }
        )

    chemical_summary = pd.DataFrame(
        chemical_rows
    )

    chemical_summary = chemical_summary.sort_values(
        [
            "parents_with_global_robust_mechanism",
            "parents_with_within_parent_robust_mechanism",
            "maxT_parent_count",
            "median_developmental_direction_score",
            "chemical_name",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
        ],
    )

    chemical_summary.to_csv(
        CHEMICAL_OUTPUT,
        sep="\t",
        index=False,
    )

    robust_parents = result.loc[
        result[
            "leaveout_global_total_count"
        ].gt(0)
    ].copy()

    robust_parents.to_csv(
        ROBUST_PARENT_OUTPUT,
        sep="\t",
        index=False,
    )

    relative_parents = result.loc[
        result[
            "leaveout_global_total_count"
        ].eq(0)
        & result[
            "leaveout_within_parent_total_count"
        ].gt(0)
    ].copy()

    relative_parents.to_csv(
        RELATIVE_PARENT_OUTPUT,
        sep="\t",
        index=False,
    )

    completion = pd.DataFrame(
        [
            {
                "maxT_parents_input":
                    len(
                        max_t
                    ),

                "maxT_parents_classified":
                    len(
                        result
                    ),

                "mechanism_categories":
                    len(
                        ALL_CATEGORIES
                    ),

                "parents_with_any_global_robust_mechanism":
                    int(
                        result[
                            "leaveout_global_total_count"
                        ].gt(0).sum()
                    ),

                "parents_with_any_within_parent_robust_mechanism":
                    int(
                        result[
                            "leaveout_within_parent_total_count"
                        ].gt(0).sum()
                    ),

                "parents_with_global_robust_late_maturation":
                    int(
                        result[
                            "leaveout_global_late_count"
                        ].gt(0).sum()
                    ),

                "chemical_entities":
                    len(
                        chemical_summary
                    ),

                "leave_program_out_results_used":
                    True,

                "original_overlap_dependent_support_flagged":
                    True,

                "Phase6D2_status":
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
            "===== PHASE 6D2 CLASS SUMMARY =====",
            class_summary.to_string(
                index=False
            ),
            "",
            "===== MAXT CATEGORY ROBUSTNESS =====",
            category_summary.to_string(
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
                "Phase 6D2 failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
