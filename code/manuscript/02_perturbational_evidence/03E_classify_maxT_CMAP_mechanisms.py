#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT = Path(
    "."
)

SCORE_FILE = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6C/"
      "phase6C3D_CMAP_parent_mechanism_scores.tsv.gz"
)

MAXT_FILE = (
    PROJECT
    / "07_tables/main_tables/phase6/"
      "phase6C3D_maxT_maturation_parent_mechanism_summary.tsv"
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
      "phase6C3E_maxT_mechanistic_classification.log"
)

PARENT_OUTPUT = (
    PROCESSED_DIR
    / "phase6C3E_maxT_parent_mechanistic_classification.tsv.gz"
)

CHEMICAL_OUTPUT = (
    TABLE_DIR
    / "phase6C3E_maxT_chemical_mechanistic_summary.tsv"
)

CLASS_SUMMARY_OUTPUT = (
    TABLE_DIR
    / "phase6C3E_parent_mechanism_class_summary.tsv"
)

CATEGORY_SUPPORT_OUTPUT = (
    TABLE_DIR
    / "phase6C3E_maxT_category_support_summary.tsv"
)

TOP_PARENT_OUTPUT = (
    TABLE_DIR
    / "phase6C3E_representative_maxT_parents.tsv"
)

COMPLETION_OUTPUT = (
    TABLE_DIR
    / "phase6C3E_completion_summary.tsv"
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
        "na",
        "none",
    }:
        return ""

    return value


def create_chemical_key(
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

    chemical_name = clean_text(
        row.get(
            "chemical_name",
            "",
        )
    ).lower()

    if chemical_name:
        return chemical_name

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
    early_proliferation_global = bool(
        row[
            "global_early_proliferative_state"
        ]
    )

    fetal_neurodevelopment_global = bool(
        row[
            "global_fetal_neurodevelopmental_state"
        ]
    )

    global_injury_count = int(
        row[
            "global_injury_category_count"
        ]
    )

    global_late_count = int(
        row[
            "global_late_category_count"
        ]
    )

    relative_late_count = int(
        row[
            "within_parent_late_category_count"
        ]
    )

    relative_injury_count = int(
        row[
            "within_parent_injury_category_count"
        ]
    )

    if (
        early_proliferation_global
        and global_injury_count > 0
    ):
        return (
            "cell_cycle_plus_injury_confounded"
        )

    if early_proliferation_global:
        return (
            "cell_cycle_suppression_dominant"
        )

    if global_injury_count > 0:
        return (
            "injury_stress_dominant"
        )

    if fetal_neurodevelopment_global:
        return (
            "early_neurodevelopmental_suppression"
        )

    if global_late_count > 0:
        return (
            "globally_supported_late_maturation"
        )

    if (
        relative_late_count > 0
        and relative_injury_count == 0
    ):
        return (
            "relative_late_support_exploratory"
        )

    if (
        relative_late_count > 0
        and relative_injury_count > 0
    ):
        return (
            "mixed_relative_late_and_injury"
        )

    if relative_injury_count > 0:
        return (
            "relative_injury_support_only"
        )

    return (
        "unresolved_developmental_state_shift"
    )


def main() -> None:
    if not SCORE_FILE.exists():
        raise FileNotFoundError(
            f"Missing mechanism-score file: {SCORE_FILE}"
        )

    if not MAXT_FILE.exists():
        raise FileNotFoundError(
            f"Missing maxT parent file: {MAXT_FILE}"
        )

    scores = pd.read_csv(
        SCORE_FILE,
        sep="\t",
        low_memory=False,
    )

    max_t = pd.read_csv(
        MAXT_FILE,
        sep="\t",
        low_memory=False,
    )

    required_score_columns = {
        "CMAP_parent",
        "mechanism_category",
        "preferred_direction_log2OR",
        "aligned_category_FDR_significant",
        "aligned_parent_profile_FDR_significant",
    }

    missing_score_columns = sorted(
        required_score_columns
        - set(
            scores.columns
        )
    )

    if missing_score_columns:
        raise RuntimeError(
            "Missing score columns: "
            + ", ".join(
                missing_score_columns
            )
        )

    if len(
        max_t
    ) != 142:
        raise RuntimeError(
            (
                "Expected 142 maxT-supported parents, "
                f"found {len(max_t)}."
            )
        )

    scores[
        "aligned_category_FDR_significant"
    ] = parse_boolean(
        scores[
            "aligned_category_FDR_significant"
        ]
    )

    scores[
        "aligned_parent_profile_FDR_significant"
    ] = parse_boolean(
        scores[
            "aligned_parent_profile_FDR_significant"
        ]
    )

    scores[
        "preferred_direction_log2OR"
    ] = pd.to_numeric(
        scores[
            "preferred_direction_log2OR"
        ],
        errors="coerce",
    )

    selected_scores = scores.loc[
        scores[
            "CMAP_parent"
        ].isin(
            max_t[
                "CMAP_parent"
            ]
        )
    ].copy()

    expected_rows = (
        len(
            max_t
        )
        * len(
            ALL_CATEGORIES
        )
    )

    if len(
        selected_scores
    ) != expected_rows:
        raise RuntimeError(
            (
                "Expected "
                f"{expected_rows} maxT parent-category rows, "
                f"found {len(selected_scores)}."
            )
        )

    if (
        selected_scores[
            "mechanism_category"
        ].nunique()
        != 9
    ):
        raise RuntimeError(
            "The maxT score subset does not contain nine categories."
        )

    global_matrix = selected_scores.pivot(
        index="CMAP_parent",
        columns="mechanism_category",
        values=(
            "aligned_category_FDR_significant"
        ),
    ).fillna(
        False
    ).astype(
        bool
    )

    parent_matrix = selected_scores.pivot(
        index="CMAP_parent",
        columns="mechanism_category",
        values=(
            "aligned_parent_profile_FDR_significant"
        ),
    ).fillna(
        False
    ).astype(
        bool
    )

    score_matrix = selected_scores.pivot(
        index="CMAP_parent",
        columns="mechanism_category",
        values="preferred_direction_log2OR",
    )

    for category in ALL_CATEGORIES:
        if category not in global_matrix.columns:
            raise RuntimeError(
                f"Missing mechanism category: {category}"
            )

    classification = max_t.copy()

    classification = classification.set_index(
        "CMAP_parent",
        drop=False,
    )

    global_matrix = global_matrix.reindex(
        classification.index
    )

    parent_matrix = parent_matrix.reindex(
        classification.index
    )

    score_matrix = score_matrix.reindex(
        classification.index
    )

    for category in ALL_CATEGORIES:
        classification[
            f"global_{category}"
        ] = global_matrix[
            category
        ]

        classification[
            f"within_parent_{category}"
        ] = parent_matrix[
            category
        ]

        classification[
            f"decomposition_{category}_log2OR"
        ] = score_matrix[
            category
        ]

    classification[
        "global_early_category_count"
    ] = global_matrix[
        EARLY_CATEGORIES
    ].sum(
        axis=1
    )

    classification[
        "global_late_category_count"
    ] = global_matrix[
        LATE_CATEGORIES
    ].sum(
        axis=1
    )

    classification[
        "global_injury_category_count"
    ] = global_matrix[
        INJURY_CATEGORIES
    ].sum(
        axis=1
    )

    classification[
        "within_parent_early_category_count"
    ] = parent_matrix[
        EARLY_CATEGORIES
    ].sum(
        axis=1
    )

    classification[
        "within_parent_late_category_count"
    ] = parent_matrix[
        LATE_CATEGORIES
    ].sum(
        axis=1
    )

    classification[
        "within_parent_injury_category_count"
    ] = parent_matrix[
        INJURY_CATEGORIES
    ].sum(
        axis=1
    )

    classification[
        "mechanistic_class"
    ] = classification.apply(
        classify_parent,
        axis=1,
    )

    classification[
        "chemical_key"
    ] = classification.apply(
        create_chemical_key,
        axis=1,
    )

    classification[
        "late_minus_injury_score"
    ] = (
        classification[
            "late_maturation_support_score"
        ]
        - classification[
            "injury_stress_confounder_score"
        ]
    )

    classification[
        "late_minus_early_score"
    ] = (
        classification[
            "late_maturation_support_score"
        ]
        - classification[
            "early_program_suppression_score"
        ]
    )

    classification = classification.reset_index(
        drop=True
    )

    classification = classification.sort_values(
        [
            "mechanistic_class",
            "global_injury_category_count",
            "global_early_category_count",
            "within_parent_late_category_count",
            "late_minus_injury_score",
            "developmental_direction_score",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            False,
            False,
        ],
    )

    classification.to_csv(
        PARENT_OUTPUT,
        sep="\t",
        index=False,
        compression="gzip",
    )

    class_summary = (
        classification.groupby(
            "mechanistic_class",
            dropna=False,
        )
        .agg(
            CMAP_parent_count=(
                "CMAP_parent",
                "size",
            ),

            unique_chemical_entities=(
                "chemical_key",
                "nunique",
            ),

            median_developmental_direction_score=(
                "developmental_direction_score",
                "median",
            ),

            median_early_program_suppression_score=(
                "early_program_suppression_score",
                "median",
            ),

            median_late_maturation_support_score=(
                "late_maturation_support_score",
                "median",
            ),

            median_injury_stress_confounder_score=(
                "injury_stress_confounder_score",
                "median",
            ),

            median_late_minus_injury_score=(
                "late_minus_injury_score",
                "median",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "CMAP_parent_count",
                "mechanistic_class",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    class_summary[
        "fraction_of_142_parents"
    ] = (
        class_summary[
            "CMAP_parent_count"
        ]
        / len(
            classification
        )
    )

    class_summary.to_csv(
        CLASS_SUMMARY_OUTPUT,
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
                        classification
                    ),

                "globally_FDR_aligned_parents":
                    int(
                        classification[
                            f"global_{category}"
                        ].sum()
                    ),

                "within_parent_FDR_aligned_parents":
                    int(
                        classification[
                            f"within_parent_{category}"
                        ].sum()
                    ),

                "median_log2OR":
                    safe_median(
                        classification[
                            f"decomposition_{category}_log2OR"
                        ]
                    ),

                "positive_log2OR_parents":
                    int(
                        classification[
                            f"decomposition_{category}_log2OR"
                        ].gt(0).sum()
                    ),

                "negative_log2OR_parents":
                    int(
                        classification[
                            f"decomposition_{category}_log2OR"
                        ].lt(0).sum()
                    ),
            }
        )

    category_support = pd.DataFrame(
        category_rows
    )

    category_support.to_csv(
        CATEGORY_SUPPORT_OUTPUT,
        sep="\t",
        index=False,
    )

    chemical_rows = []

    for chemical_key, group in classification.groupby(
        "chemical_key",
        sort=True,
        dropna=False,
    ):
        class_counts = (
            group[
                "mechanistic_class"
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
                    chemical_key,

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

                "dominant_mechanistic_class":
                    dominant_class,

                "dominant_class_parent_count":
                    dominant_count,

                "dominant_class_fraction":
                    dominant_count
                    / len(
                        group
                    ),

                "mechanistic_classes":
                    "|".join(
                        (
                            class_counts.index
                            + ":"
                            + class_counts.astype(
                                str
                            )
                        ).tolist()
                    ),

                "cell_cycle_plus_injury_parent_count":
                    int(
                        group[
                            "mechanistic_class"
                        ].eq(
                            "cell_cycle_plus_injury_confounded"
                        ).sum()
                    ),

                "cell_cycle_dominant_parent_count":
                    int(
                        group[
                            "mechanistic_class"
                        ].eq(
                            "cell_cycle_suppression_dominant"
                        ).sum()
                    ),

                "injury_dominant_parent_count":
                    int(
                        group[
                            "mechanistic_class"
                        ].eq(
                            "injury_stress_dominant"
                        ).sum()
                    ),

                "relative_late_support_parent_count":
                    int(
                        group[
                            "mechanistic_class"
                        ].isin(
                            {
                                "relative_late_support_exploratory",
                                "mixed_relative_late_and_injury",
                                "globally_supported_late_maturation",
                            }
                        ).sum()
                    ),

                "unresolved_parent_count":
                    int(
                        group[
                            "mechanistic_class"
                        ].eq(
                            "unresolved_developmental_state_shift"
                        ).sum()
                    ),

                "median_developmental_direction_score":
                    safe_median(
                        group[
                            "developmental_direction_score"
                        ]
                    ),

                "median_early_program_suppression_score":
                    safe_median(
                        group[
                            "early_program_suppression_score"
                        ]
                    ),

                "median_late_maturation_support_score":
                    safe_median(
                        group[
                            "late_maturation_support_score"
                        ]
                    ),

                "median_injury_stress_confounder_score":
                    safe_median(
                        group[
                            "injury_stress_confounder_score"
                        ]
                    ),

                "median_late_minus_injury_score":
                    safe_median(
                        group[
                            "late_minus_injury_score"
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
                        maximum=30,
                    ),
            }
        )

    chemical_summary = pd.DataFrame(
        chemical_rows
    )

    chemical_summary = chemical_summary.sort_values(
        [
            "maxT_parent_count",
            "dominant_class_fraction",
            "median_developmental_direction_score",
            "chemical_name",
        ],
        ascending=[
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

    representative_rows = []

    for mechanism_class, group in classification.groupby(
        "mechanistic_class",
        sort=True,
    ):
        group = group.sort_values(
            [
                "global_injury_category_count",
                "global_early_category_count",
                "within_parent_late_category_count",
                "late_minus_injury_score",
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
        ).head(
            25
        )

        group = group.copy()

        group[
            "within_class_rank"
        ] = range(
            1,
            len(
                group
            )
            + 1,
        )

        representative_rows.append(
            group
        )

    representatives = pd.concat(
        representative_rows,
        ignore_index=True,
    )

    representative_columns = [
        "mechanistic_class",
        "within_class_rank",
        "chemical_name",
        "DTXSID",
        "CMAP_parent",
        "developmental_direction_score",
        "maxT_maturation_FWER_p",
        "early_program_suppression_score",
        "late_maturation_support_score",
        "injury_stress_confounder_score",
        "late_minus_injury_score",
        "global_early_category_count",
        "global_late_category_count",
        "global_injury_category_count",
        "within_parent_early_category_count",
        "within_parent_late_category_count",
        "within_parent_injury_category_count",
        "decomposition_early_proliferative_state_log2OR",
        "decomposition_fetal_neurodevelopmental_state_log2OR",
        "decomposition_late_synaptic_maturation_log2OR",
        "decomposition_mitochondrial_metabolic_maturation_log2OR",
        "decomposition_glial_myelin_maturation_log2OR",
        "decomposition_DNA_damage_p53_log2OR",
        "decomposition_apoptosis_cytotoxicity_log2OR",
        "decomposition_oxidative_hypoxic_stress_log2OR",
        "decomposition_inflammatory_stress_log2OR",
    ]

    representatives[
        representative_columns
    ].to_csv(
        TOP_PARENT_OUTPUT,
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
                        classification
                    ),

                "mechanism_categories_used":
                    len(
                        ALL_CATEGORIES
                    ),

                "chemical_entities_created":
                    len(
                        chemical_summary
                    ),

                "globally_supported_late_maturation_parents":
                    int(
                        classification[
                            "global_late_category_count"
                        ].gt(0).sum()
                    ),

                "parents_with_global_cell_cycle_support":
                    int(
                        classification[
                            "global_early_proliferative_state"
                        ].sum()
                    ),

                "parents_with_any_global_injury_support":
                    int(
                        classification[
                            "global_injury_category_count"
                        ].gt(0).sum()
                    ),

                "classification_is_descriptive":
                    True,

                "late_support_without_global_FDR_is_exploratory":
                    True,

                "Phase6C3E_status":
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
            "===== PHASE 6C3E CLASS SUMMARY =====",
            class_summary.to_string(
                index=False
            ),
            "",
            "===== MAXT CATEGORY SUPPORT =====",
            category_support.to_string(
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
                "Phase 6C3E failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
