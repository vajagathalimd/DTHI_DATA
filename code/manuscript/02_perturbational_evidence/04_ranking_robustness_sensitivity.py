#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(
    "."
)

INPUT_FILE = (
    PROJECT
    / "07_tables/main_tables/phase6/"
      "phase6F2_dual_chemical_ranking.tsv"
)

CLASSIFICATION_FILE = (
    PROJECT
    / "07_tables/main_tables/phase6/"
      "phase6F3_final_biological_classification.tsv"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase6"
)

PROCESSED_DIR = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6F"
)

FIGURE_DIR = (
    PROJECT
    / "08_figures/supplementary_figures/phase6"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase6/"
      "phase6F4_ranking_robustness_sensitivity.log"
)


MECHANISTIC_LONG_OUTPUT = (
    TABLE_DIR
    / "phase6F4_mechanistic_weighting_sensitivity.tsv"
)

MECHANISTIC_STABILITY_OUTPUT = (
    TABLE_DIR
    / "phase6F4_mechanistic_rank_stability.tsv"
)

CONFIDENCE_LONG_OUTPUT = (
    TABLE_DIR
    / "phase6F4_confidence_leave_layer_out.tsv"
)

CONFIDENCE_STABILITY_OUTPUT = (
    TABLE_DIR
    / "phase6F4_confidence_rank_stability.tsv"
)

COMPARISON_OUTPUT = (
    TABLE_DIR
    / "phase6F4_scenario_comparison_summary.tsv"
)

TOPSET_OUTPUT = (
    TABLE_DIR
    / "phase6F4_top_set_overlap.tsv"
)

CHEMICAL_SUMMARY_OUTPUT = (
    TABLE_DIR
    / "phase6F4_chemical_robustness_summary.tsv"
)

COMPLETION_OUTPUT = (
    TABLE_DIR
    / "phase6F4_completion_summary.tsv"
)

FULL_OUTPUT_GZ = (
    PROCESSED_DIR
    / "phase6F4_ranking_robustness_results.tsv.gz"
)

FIGURE_PNG = (
    FIGURE_DIR
    / "Supplementary_Figure_6F4_ranking_robustness.png"
)

FIGURE_PDF = (
    FIGURE_DIR
    / "Supplementary_Figure_6F4_ranking_robustness.pdf"
)

FIGURE_CAPTION = (
    FIGURE_DIR
    / "Supplementary_Figure_6F4_caption.txt"
)


EXPECTED_CHEMICALS = 29
EXPECTED_MATCHED = 15
EXPECTED_UNMATCHED = 14
EXPECTED_REPRODUCIBILITY_EVALUABLE = 14

MECHANISTIC_TOP_K = 5
CONFIDENCE_TOP_K = 10


MECHANISTIC_SCENARIOS = {
    "baseline_CMAP40_LINCS60": (
        0.40,
        0.60,
    ),

    "equal_CMAP50_LINCS50": (
        0.50,
        0.50,
    ),

    "CMAP_heavy_CMAP60_LINCS40": (
        0.60,
        0.40,
    ),

    "LINCS_heavy_CMAP20_LINCS80": (
        0.20,
        0.80,
    ),

    "CMAP_only_matched": (
        1.00,
        0.00,
    ),

    "direct_LINCS_only": (
        0.00,
        1.00,
    ),
}


CONFIDENCE_COMPONENT_MAXIMA = {
    "confidence_CMAP_component":
        30.0,

    "confidence_direct_component":
        35.0,

    "confidence_reproducibility_component":
        25.0,

    "confidence_neural_component":
        10.0,
}


CONFIDENCE_SCENARIOS = {
    "baseline_all_layers":
        [],

    "leave_out_CMAP":
        [
            "confidence_CMAP_component",
        ],

    "leave_out_direct_LINCS":
        [
            "confidence_direct_component",
        ],

    "leave_out_reproducibility":
        [
            "confidence_reproducibility_component",
        ],

    "leave_out_neural":
        [
            "confidence_neural_component",
        ],
}


for directory in [
    TABLE_DIR,
    PROCESSED_DIR,
    FIGURE_DIR,
    LOG_FILE.parent,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def clean_text(
    value: object,
) -> str:
    if pd.isna(
        value
    ):
        return ""

    text = str(
        value
    ).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "na",
    }:
        return ""

    return text


def parse_bool(
    value: object,
) -> bool:
    if isinstance(
        value,
        bool,
    ):
        return value

    if pd.isna(
        value
    ):
        return False

    return str(
        value
    ).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def numeric(
    values: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        values,
        errors="coerce",
    )


def deterministic_ranks(
    table: pd.DataFrame,
    score_column: str,
) -> pd.Series:
    ordering = (
        table[
            [
                "chemical_name",
                score_column,
            ]
        ]
        .sort_values(
            [
                score_column,
                "chemical_name",
            ],
            ascending=[
                False,
                True,
            ],
            na_position="last",
        )
        .reset_index()
    )

    ordering[
        "deterministic_rank"
    ] = np.arange(
        1,
        len(
            ordering
        ) + 1,
    )

    rank_lookup = ordering.set_index(
        "index"
    )[
        "deterministic_rank"
    ]

    return rank_lookup.reindex(
        table.index
    ).astype(
        int
    )


def top_set(
    table: pd.DataFrame,
    score_column: str,
    k: int,
) -> set[str]:
    return set(
        table.sort_values(
            [
                score_column,
                "chemical_name",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .head(
            k
        )[
            "chemical_name"
        ]
        .astype(
            str
        )
    )


def jaccard(
    first: set[str],
    second: set[str],
) -> float:
    union = first | second

    if not union:
        return np.nan

    return len(
        first & second
    ) / len(
        union
    )


def rank_correlation(
    baseline: pd.DataFrame,
    scenario: pd.DataFrame,
) -> tuple[
    float,
    float,
]:
    merged = baseline[
        [
            "chemical_name",
            "scenario_score",
            "scenario_rank",
        ]
    ].merge(
        scenario[
            [
                "chemical_name",
                "scenario_score",
                "scenario_rank",
            ]
        ],
        on="chemical_name",
        how="inner",
        suffixes=(
            "_baseline",
            "_scenario",
        ),
        validate="one_to_one",
    )

    if len(
        merged
    ) < 3:
        return (
            np.nan,
            np.nan,
        )

    spearman = float(
        merged[
            [
                "scenario_rank_baseline",
                "scenario_rank_scenario",
            ]
        ].corr(
            method="spearman"
        ).iloc[
            0,
            1,
        ]
    )

    pearson = float(
        merged[
            [
                "scenario_score_baseline",
                "scenario_score_scenario",
            ]
        ].corr(
            method="pearson"
        ).iloc[
            0,
            1,
        ]
    )

    return (
        spearman,
        pearson,
    )


def main() -> None:
    for path in [
        INPUT_FILE,
        CLASSIFICATION_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(
                    path
                )
            )

    ranking = pd.read_csv(
        INPUT_FILE,
        sep="\t",
        low_memory=False,
    )

    classification = pd.read_csv(
        CLASSIFICATION_FILE,
        sep="\t",
        low_memory=False,
    )

    if len(
        ranking
    ) != EXPECTED_CHEMICALS:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_CHEMICALS} ranking rows, "
                f"found {len(ranking)}."
            )
        )

    if len(
        classification
    ) != EXPECTED_CHEMICALS:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_CHEMICALS} classification rows, "
                f"found {len(classification)}."
            )
        )

    required_columns = [
        "chemical_name",
        "mechanistic_priority_scope",
        "CMAP_provisional_priority_score",
        "direct_LINCS_hazard_priority_score",
        "integrated_mechanistic_priority_score",
        "evidence_confidence_score",
        "confidence_CMAP_component",
        "confidence_direct_component",
        "confidence_reproducibility_component",
        "confidence_neural_component",
        "replicate_concordant_sensitivity_class",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in ranking.columns
    ]

    if missing_columns:
        raise RuntimeError(
            (
                "Missing Phase 6F4 input columns: "
                + "|".join(
                    missing_columns
                )
            )
        )

    data = ranking.merge(
        classification[
            [
                "chemical_name",
                "final_biological_class",
                "final_evidence_strength",
                "classification_claim_scope",
            ]
        ],
        on="chemical_name",
        how="left",
        validate="one_to_one",
    )

    matched = data.loc[
        data[
            "mechanistic_priority_scope"
        ].eq(
            "integrated_CMAP_plus_direct_LINCS"
        )
    ].copy()

    unmatched = data.loc[
        data[
            "mechanistic_priority_scope"
        ].eq(
            "CMAP_only_provisional"
        )
    ].copy()

    if len(
        matched
    ) != EXPECTED_MATCHED:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_MATCHED} matched chemicals, "
                f"found {len(matched)}."
            )
        )

    if len(
        unmatched
    ) != EXPECTED_UNMATCHED:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_UNMATCHED} unmatched chemicals, "
                f"found {len(unmatched)}."
            )
        )

    cmap_score = numeric(
        matched[
            "CMAP_provisional_priority_score"
        ]
    )

    direct_score = numeric(
        matched[
            "direct_LINCS_hazard_priority_score"
        ]
    )

    if (
        cmap_score.isna().any()
        or direct_score.isna().any()
    ):
        raise RuntimeError(
            "Matched chemicals contain missing mechanistic scores."
        )

    # ================================================================
    # Mechanistic weighting sensitivity
    # ================================================================

    mechanistic_rows: list[
        dict[str, object]
    ] = []

    for scenario, (
        cmap_weight,
        direct_weight,
    ) in MECHANISTIC_SCENARIOS.items():
        scenario_table = matched[
            [
                "chemical_name",
                "final_biological_class",
                "final_evidence_strength",
            ]
        ].copy()

        scenario_table[
            "scenario_name"
        ] = scenario

        scenario_table[
            "CMAP_weight"
        ] = cmap_weight

        scenario_table[
            "direct_LINCS_weight"
        ] = direct_weight

        scenario_table[
            "scenario_score"
        ] = (
            cmap_weight
            * cmap_score
            + direct_weight
            * direct_score
        )

        scenario_table[
            "scenario_rank"
        ] = deterministic_ranks(
            scenario_table,
            "scenario_score",
        )

        mechanistic_rows.extend(
            scenario_table.to_dict(
                orient="records"
            )
        )

    mechanistic_long = pd.DataFrame(
        mechanistic_rows
    )

    baseline_mechanistic = mechanistic_long.loc[
        mechanistic_long[
            "scenario_name"
        ].eq(
            "baseline_CMAP40_LINCS60"
        )
    ].copy()

    mechanistic_stability = (
        mechanistic_long.groupby(
            "chemical_name",
            sort=True,
        )
        .agg(
            baseline_rank=(
                "scenario_rank",
                lambda values: int(
                    mechanistic_long.loc[
                        values.index
                    ].loc[
                        mechanistic_long.loc[
                            values.index,
                            "scenario_name",
                        ].eq(
                            "baseline_CMAP40_LINCS60"
                        ),
                        "scenario_rank",
                    ].iloc[
                        0
                    ]
                ),
            ),

            minimum_rank=(
                "scenario_rank",
                "min",
            ),

            maximum_rank=(
                "scenario_rank",
                "max",
            ),

            mean_rank=(
                "scenario_rank",
                "mean",
            ),

            rank_standard_deviation=(
                "scenario_rank",
                "std",
            ),

            minimum_score=(
                "scenario_score",
                "min",
            ),

            maximum_score=(
                "scenario_score",
                "max",
            ),

            final_biological_class=(
                "final_biological_class",
                "first",
            ),

            final_evidence_strength=(
                "final_evidence_strength",
                "first",
            ),
        )
        .reset_index()
    )

    mechanistic_stability[
        "rank_range"
    ] = (
        mechanistic_stability[
            "maximum_rank"
        ]
        - mechanistic_stability[
            "minimum_rank"
        ]
    )

    top5_by_scenario = {
        scenario: top_set(
            group,
            "scenario_score",
            MECHANISTIC_TOP_K,
        )
        for scenario, group in mechanistic_long.groupby(
            "scenario_name",
            sort=True,
        )
    }

    mechanistic_stability[
        "stable_top5_all_mechanistic_scenarios"
    ] = mechanistic_stability[
        "chemical_name"
    ].map(
        lambda chemical: all(
            chemical in members
            for members in top5_by_scenario.values()
        )
    )

    # ================================================================
    # Confidence leave-layer-out sensitivity
    # ================================================================

    confidence_rows: list[
        dict[str, object]
    ] = []

    for scenario, excluded_components in CONFIDENCE_SCENARIOS.items():
        retained_components = [
            component
            for component in CONFIDENCE_COMPONENT_MAXIMA
            if component not in excluded_components
        ]

        retained_maximum = sum(
            CONFIDENCE_COMPONENT_MAXIMA[
                component
            ]
            for component in retained_components
        )

        scenario_table = data[
            [
                "chemical_name",
                "final_biological_class",
                "final_evidence_strength",
                "classification_claim_scope",
            ]
        ].copy()

        scenario_table[
            "scenario_name"
        ] = scenario

        scenario_table[
            "excluded_components"
        ] = (
            "|".join(
                excluded_components
            )
            if excluded_components
            else "none"
        )

        scenario_table[
            "retained_maximum_points"
        ] = retained_maximum

        scenario_table[
            "scenario_score"
        ] = (
            data[
                retained_components
            ].apply(
                pd.to_numeric,
                errors="coerce",
            ).fillna(
                0
            ).sum(
                axis=1
            )
            / retained_maximum
            * 100.0
        )

        scenario_table[
            "scenario_rank"
        ] = deterministic_ranks(
            scenario_table,
            "scenario_score",
        )

        confidence_rows.extend(
            scenario_table.to_dict(
                orient="records"
            )
        )

    confidence_long = pd.DataFrame(
        confidence_rows
    )

    baseline_confidence = confidence_long.loc[
        confidence_long[
            "scenario_name"
        ].eq(
            "baseline_all_layers"
        )
    ].copy()

    confidence_stability = (
        confidence_long.groupby(
            "chemical_name",
            sort=True,
        )
        .agg(
            baseline_rank=(
                "scenario_rank",
                lambda values: int(
                    confidence_long.loc[
                        values.index
                    ].loc[
                        confidence_long.loc[
                            values.index,
                            "scenario_name",
                        ].eq(
                            "baseline_all_layers"
                        ),
                        "scenario_rank",
                    ].iloc[
                        0
                    ]
                ),
            ),

            minimum_rank=(
                "scenario_rank",
                "min",
            ),

            maximum_rank=(
                "scenario_rank",
                "max",
            ),

            mean_rank=(
                "scenario_rank",
                "mean",
            ),

            rank_standard_deviation=(
                "scenario_rank",
                "std",
            ),

            minimum_score=(
                "scenario_score",
                "min",
            ),

            maximum_score=(
                "scenario_score",
                "max",
            ),

            final_biological_class=(
                "final_biological_class",
                "first",
            ),

            final_evidence_strength=(
                "final_evidence_strength",
                "first",
            ),

            classification_claim_scope=(
                "classification_claim_scope",
                "first",
            ),
        )
        .reset_index()
    )

    confidence_stability[
        "rank_range"
    ] = (
        confidence_stability[
            "maximum_rank"
        ]
        - confidence_stability[
            "minimum_rank"
        ]
    )

    top10_by_scenario = {
        scenario: top_set(
            group,
            "scenario_score",
            CONFIDENCE_TOP_K,
        )
        for scenario, group in confidence_long.groupby(
            "scenario_name",
            sort=True,
        )
    }

    confidence_stability[
        "stable_top10_all_confidence_scenarios"
    ] = confidence_stability[
        "chemical_name"
    ].map(
        lambda chemical: all(
            chemical in members
            for members in top10_by_scenario.values()
        )
    )

    # ================================================================
    # Scenario comparisons
    # ================================================================

    comparison_rows: list[
        dict[str, object]
    ] = []

    baseline_mechanistic_top = top_set(
        baseline_mechanistic,
        "scenario_score",
        MECHANISTIC_TOP_K,
    )

    for scenario, group in mechanistic_long.groupby(
        "scenario_name",
        sort=True,
    ):
        spearman, pearson = rank_correlation(
            baseline_mechanistic,
            group,
        )

        scenario_top = top_set(
            group,
            "scenario_score",
            MECHANISTIC_TOP_K,
        )

        comparison_rows.append(
            {
                "analysis_family":
                    "mechanistic_priority",

                "baseline_scenario":
                    "baseline_CMAP40_LINCS60",

                "comparison_scenario":
                    scenario,

                "chemical_count":
                    len(
                        group
                    ),

                "spearman_rank_correlation":
                    spearman,

                "pearson_score_correlation":
                    pearson,

                "top_k":
                    MECHANISTIC_TOP_K,

                "top_k_overlap_count":
                    len(
                        baseline_mechanistic_top
                        & scenario_top
                    ),

                "top_k_jaccard":
                    jaccard(
                        baseline_mechanistic_top,
                        scenario_top,
                    ),
            }
        )

    baseline_confidence_top = top_set(
        baseline_confidence,
        "scenario_score",
        CONFIDENCE_TOP_K,
    )

    for scenario, group in confidence_long.groupby(
        "scenario_name",
        sort=True,
    ):
        spearman, pearson = rank_correlation(
            baseline_confidence,
            group,
        )

        scenario_top = top_set(
            group,
            "scenario_score",
            CONFIDENCE_TOP_K,
        )

        comparison_rows.append(
            {
                "analysis_family":
                    "evidence_confidence",

                "baseline_scenario":
                    "baseline_all_layers",

                "comparison_scenario":
                    scenario,

                "chemical_count":
                    len(
                        group
                    ),

                "spearman_rank_correlation":
                    spearman,

                "pearson_score_correlation":
                    pearson,

                "top_k":
                    CONFIDENCE_TOP_K,

                "top_k_overlap_count":
                    len(
                        baseline_confidence_top
                        & scenario_top
                    ),

                "top_k_jaccard":
                    jaccard(
                        baseline_confidence_top,
                        scenario_top,
                    ),
            }
        )

    comparison_summary = pd.DataFrame(
        comparison_rows
    )

    top_set_rows: list[
        dict[str, object]
    ] = []

    for scenario, members in top5_by_scenario.items():
        top_set_rows.append(
            {
                "analysis_family":
                    "mechanistic_priority",

                "scenario_name":
                    scenario,

                "top_k":
                    MECHANISTIC_TOP_K,

                "chemical_names":
                    "|".join(
                        sorted(
                            members
                        )
                    ),
            }
        )

    for scenario, members in top10_by_scenario.items():
        top_set_rows.append(
            {
                "analysis_family":
                    "evidence_confidence",

                "scenario_name":
                    scenario,

                "top_k":
                    CONFIDENCE_TOP_K,

                "chemical_names":
                    "|".join(
                        sorted(
                            members
                        )
                    ),
            }
        )

    top_set_table = pd.DataFrame(
        top_set_rows
    )

    # ================================================================
    # Replicate-concordant exclusion sensitivity
    # ================================================================

    reproducibility_evaluable = matched.loc[
        ~matched[
            "replicate_concordant_sensitivity_class"
        ].fillna(
            ""
        ).eq(
            "not_evaluable_no_replicate_concordant_profiles"
        )
    ].copy()

    if len(
        reproducibility_evaluable
    ) != EXPECTED_REPRODUCIBILITY_EVALUABLE:
        raise RuntimeError(
            (
                "Expected "
                f"{EXPECTED_REPRODUCIBILITY_EVALUABLE} "
                "reproducibility-evaluable matched chemicals, found "
                f"{len(reproducibility_evaluable)}."
            )
        )

    reproducibility_evaluable[
        "restricted_baseline_score"
    ] = numeric(
        reproducibility_evaluable[
            "integrated_mechanistic_priority_score"
        ]
    )

    reproducibility_evaluable[
        "restricted_baseline_rank"
    ] = deterministic_ranks(
        reproducibility_evaluable,
        "restricted_baseline_score",
    )

    baseline_without_non_evaluable = baseline_mechanistic.loc[
        baseline_mechanistic[
            "chemical_name"
        ].isin(
            reproducibility_evaluable[
                "chemical_name"
            ]
        )
    ].copy()

    restricted_top5 = top_set(
        reproducibility_evaluable.rename(
            columns={
                "restricted_baseline_score":
                    "scenario_score",
            }
        ),
        "scenario_score",
        MECHANISTIC_TOP_K,
    )

    baseline_restricted_top5 = top_set(
        baseline_without_non_evaluable,
        "scenario_score",
        MECHANISTIC_TOP_K,
    )

    comparison_summary = pd.concat(
        [
            comparison_summary,

            pd.DataFrame(
                [
                    {
                        "analysis_family":
                            "reproducibility_exclusion",

                        "baseline_scenario":
                            "baseline_CMAP40_LINCS60",

                        "comparison_scenario":
                            (
                                "exclude_no_replicate_"
                                "concordant_profiles"
                            ),

                        "chemical_count":
                            len(
                                reproducibility_evaluable
                            ),

                        "spearman_rank_correlation":
                            1.0,

                        "pearson_score_correlation":
                            1.0,

                        "top_k":
                            MECHANISTIC_TOP_K,

                        "top_k_overlap_count":
                            len(
                                baseline_restricted_top5
                                & restricted_top5
                            ),

                        "top_k_jaccard":
                            jaccard(
                                baseline_restricted_top5,
                                restricted_top5,
                            ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    # ================================================================
    # Combined chemical robustness summary
    # ================================================================

    mechanistic_stability_for_merge = (
        mechanistic_stability.rename(
            columns={
                "baseline_rank":
                    "baseline_rank_mechanistic",

                "minimum_rank":
                    "minimum_rank_mechanistic",

                "maximum_rank":
                    "maximum_rank_mechanistic",

                "mean_rank":
                    "mean_rank_mechanistic",

                "rank_standard_deviation":
                    "rank_standard_deviation_mechanistic",

                "minimum_score":
                    "minimum_score_mechanistic",

                "maximum_score":
                    "maximum_score_mechanistic",

                "rank_range":
                    "rank_range_mechanistic",
            }
        )
    )

    confidence_stability_for_merge = (
        confidence_stability.rename(
            columns={
                "baseline_rank":
                    "baseline_rank_confidence",

                "minimum_rank":
                    "minimum_rank_confidence",

                "maximum_rank":
                    "maximum_rank_confidence",

                "mean_rank":
                    "mean_rank_confidence",

                "rank_standard_deviation":
                    "rank_standard_deviation_confidence",

                "minimum_score":
                    "minimum_score_confidence",

                "maximum_score":
                    "maximum_score_confidence",

                "rank_range":
                    "rank_range_confidence",
            }
        )
    )

    chemical_summary = data[
        [
            "chemical_name",
            "final_biological_class",
            "final_evidence_strength",
            "classification_claim_scope",
            "mechanistic_priority_scope",
            "replicate_concordant_sensitivity_class",
        ]
    ].merge(
        mechanistic_stability_for_merge,
        on=[
            "chemical_name",
            "final_biological_class",
            "final_evidence_strength",
        ],
        how="left",
        validate="one_to_one",
    ).merge(
        confidence_stability_for_merge,
        on=[
            "chemical_name",
            "final_biological_class",
            "final_evidence_strength",
            "classification_claim_scope",
        ],
        how="left",
        validate="one_to_one",
    )

    chemical_summary[
        "mechanistic_rank_robustness_class"
    ] = np.select(
        [
            chemical_summary[
                "mechanistic_priority_scope"
            ].eq(
                "CMAP_only_provisional"
            ),

            chemical_summary[
                "rank_range_mechanistic"
            ].fillna(
                np.inf
            ).le(
                2
            ),

            chemical_summary[
                "rank_range_mechanistic"
            ].fillna(
                np.inf
            ).le(
                5
            ),
        ],
        [
            "not_applicable_CMAP_only_scope",
            "high_rank_stability",
            "moderate_rank_stability",
        ],
        default="weight_sensitive",
    )

    chemical_summary[
        "confidence_rank_robustness_class"
    ] = np.select(
        [
            chemical_summary[
                "rank_range_confidence"
            ].le(
                3
            ),

            chemical_summary[
                "rank_range_confidence"
            ].le(
                7
            ),
        ],
        [
            "high_rank_stability",
            "moderate_rank_stability",
        ],
        default="layer_sensitive",
    )

    chemical_summary[
        "Phase6F4_overall_robustness"
    ] = np.select(
        [
            chemical_summary[
                "mechanistic_rank_robustness_class"
            ].eq(
                "high_rank_stability"
            )
            & chemical_summary[
                "confidence_rank_robustness_class"
            ].eq(
                "high_rank_stability"
            ),

            chemical_summary[
                "mechanistic_rank_robustness_class"
            ].isin(
                [
                    "high_rank_stability",
                    "moderate_rank_stability",
                    "not_applicable_CMAP_only_scope",
                ]
            )
            & chemical_summary[
                "confidence_rank_robustness_class"
            ].isin(
                [
                    "high_rank_stability",
                    "moderate_rank_stability",
                ]
            ),
        ],
        [
            "high_overall_robustness",
            "moderate_overall_robustness",
        ],
        default="sensitivity_dependent",
    )

    # ================================================================
    # Supplementary figure
    # ================================================================

    mechanistic_pivot = mechanistic_long.pivot(
        index="chemical_name",
        columns="scenario_name",
        values="scenario_rank",
    )

    mechanistic_order = (
        baseline_mechanistic.sort_values(
            "scenario_rank"
        )[
            "chemical_name"
        ]
        .tolist()
    )

    mechanistic_pivot = mechanistic_pivot.reindex(
        mechanistic_order
    )

    confidence_top_names = (
        baseline_confidence.sort_values(
            "scenario_rank"
        )
        .head(
            15
        )[
            "chemical_name"
        ]
        .tolist()
    )

    confidence_pivot = confidence_long.pivot(
        index="chemical_name",
        columns="scenario_name",
        values="scenario_rank",
    ).reindex(
        confidence_top_names
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(
            20,
            10,
        ),
    )

    first = axes[
        0
    ].imshow(
        mechanistic_pivot.to_numpy(
            dtype=float
        ),
        aspect="auto",
        interpolation="nearest",
    )

    axes[
        0
    ].set_title(
        "Mechanistic-priority rank sensitivity"
    )

    axes[
        0
    ].set_yticks(
        np.arange(
            len(
                mechanistic_pivot.index
            )
        )
    )

    axes[
        0
    ].set_yticklabels(
        mechanistic_pivot.index,
        fontsize=8,
    )

    axes[
        0
    ].set_xticks(
        np.arange(
            len(
                mechanistic_pivot.columns
            )
        )
    )

    axes[
        0
    ].set_xticklabels(
        [
            column.replace(
                "_",
                "\n",
            )
            for column in mechanistic_pivot.columns
        ],
        rotation=45,
        ha="right",
        fontsize=8,
    )

    figure.colorbar(
        first,
        ax=axes[
            0
        ],
        label="Rank",
    )

    second = axes[
        1
    ].imshow(
        confidence_pivot.to_numpy(
            dtype=float
        ),
        aspect="auto",
        interpolation="nearest",
    )

    axes[
        1
    ].set_title(
        "Evidence-confidence leave-layer-out sensitivity"
    )

    axes[
        1
    ].set_yticks(
        np.arange(
            len(
                confidence_pivot.index
            )
        )
    )

    axes[
        1
    ].set_yticklabels(
        confidence_pivot.index,
        fontsize=8,
    )

    axes[
        1
    ].set_xticks(
        np.arange(
            len(
                confidence_pivot.columns
            )
        )
    )

    axes[
        1
    ].set_xticklabels(
        [
            column.replace(
                "_",
                "\n",
            )
            for column in confidence_pivot.columns
        ],
        rotation=45,
        ha="right",
        fontsize=8,
    )

    figure.colorbar(
        second,
        ax=axes[
            1
        ],
        label="Rank",
    )

    figure.tight_layout()

    figure.savefig(
        FIGURE_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_PDF,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    caption = (
        "Supplementary Figure. Robustness of chemical mechanistic-priority "
        "and evidence-confidence rankings. The left panel shows ranks of "
        "the 15 chemicals with direct quantitative LINCS profiles under "
        "the prespecified baseline weighting, equal weighting, CMAP-heavy, "
        "LINCS-heavy, CMAP-only and direct-LINCS-only scenarios. The right "
        "panel shows the 15 highest-confidence chemicals under the full "
        "evidence model and after separately removing CMAP, direct LINCS, "
        "replicate-concordant or neural-context components. Lower values "
        "indicate higher rank. These analyses assess weighting and evidence-"
        "layer dependence; they do not create a therapeutic or beneficial "
        "chemical ranking."
    )

    FIGURE_CAPTION.write_text(
        caption + "\n",
        encoding="utf-8",
    )

    # ================================================================
    # Save outputs
    # ================================================================

    mechanistic_long.to_csv(
        MECHANISTIC_LONG_OUTPUT,
        sep="\t",
        index=False,
    )

    mechanistic_stability.to_csv(
        MECHANISTIC_STABILITY_OUTPUT,
        sep="\t",
        index=False,
    )

    confidence_long.to_csv(
        CONFIDENCE_LONG_OUTPUT,
        sep="\t",
        index=False,
    )

    confidence_stability.to_csv(
        CONFIDENCE_STABILITY_OUTPUT,
        sep="\t",
        index=False,
    )

    comparison_summary.to_csv(
        COMPARISON_OUTPUT,
        sep="\t",
        index=False,
    )

    top_set_table.to_csv(
        TOPSET_OUTPUT,
        sep="\t",
        index=False,
    )

    chemical_summary.to_csv(
        CHEMICAL_SUMMARY_OUTPUT,
        sep="\t",
        index=False,
    )

    combined_long = pd.concat(
        [
            mechanistic_long.assign(
                analysis_family="mechanistic_priority"
            ),

            confidence_long.assign(
                analysis_family="evidence_confidence"
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    combined_long.to_csv(
        FULL_OUTPUT_GZ,
        sep="\t",
        index=False,
        compression="gzip",
    )

    mechanistic_nonbaseline = comparison_summary.loc[
        comparison_summary[
            "analysis_family"
        ].eq(
            "mechanistic_priority"
        )
        & ~comparison_summary[
            "comparison_scenario"
        ].eq(
            "baseline_CMAP40_LINCS60"
        )
    ]

    confidence_nonbaseline = comparison_summary.loc[
        comparison_summary[
            "analysis_family"
        ].eq(
            "evidence_confidence"
        )
        & ~comparison_summary[
            "comparison_scenario"
        ].eq(
            "baseline_all_layers"
        )
    ]

    minimum_mechanistic_spearman = float(
        mechanistic_nonbaseline[
            "spearman_rank_correlation"
        ].min()
    )

    minimum_confidence_spearman = float(
        confidence_nonbaseline[
            "spearman_rank_correlation"
        ].min()
    )

    minimum_mechanistic_top5_overlap = int(
        mechanistic_nonbaseline[
            "top_k_overlap_count"
        ].min()
    )

    minimum_confidence_top10_overlap = int(
        confidence_nonbaseline[
            "top_k_overlap_count"
        ].min()
    )

    stable_mechanistic_top5 = int(
        mechanistic_stability[
            "stable_top5_all_mechanistic_scenarios"
        ].sum()
    )

    stable_confidence_top10 = int(
        confidence_stability[
            "stable_top10_all_confidence_scenarios"
        ].sum()
    )

    figure_created = bool(
        FIGURE_PNG.exists()
        and FIGURE_PDF.exists()
        and FIGURE_CAPTION.exists()
    )

    status = (
        "completed"
        if (
            len(
                mechanistic_long
            )
            == (
                EXPECTED_MATCHED
                * len(
                    MECHANISTIC_SCENARIOS
                )
            )
            and len(
                confidence_long
            )
            == (
                EXPECTED_CHEMICALS
                * len(
                    CONFIDENCE_SCENARIOS
                )
            )
            and len(
                mechanistic_stability
            )
            == EXPECTED_MATCHED
            and len(
                confidence_stability
            )
            == EXPECTED_CHEMICALS
            and len(
                reproducibility_evaluable
            )
            == EXPECTED_REPRODUCIBILITY_EVALUABLE
            and figure_created
        )
        else "failed_validation"
    )

    completion = pd.DataFrame(
        [
            {
                "mechanistic_priority_chemicals":
                    EXPECTED_MATCHED,

                "mechanistic_weighting_scenarios":
                    len(
                        MECHANISTIC_SCENARIOS
                    ),

                "mechanistic_chemical_scenario_results":
                    len(
                        mechanistic_long
                    ),

                "confidence_ranked_chemicals":
                    EXPECTED_CHEMICALS,

                "confidence_leave_layer_out_scenarios":
                    len(
                        CONFIDENCE_SCENARIOS
                    ),

                "confidence_chemical_scenario_results":
                    len(
                        confidence_long
                    ),

                "replicate_concordant_evaluable_matched_chemicals":
                    len(
                        reproducibility_evaluable
                    ),

                "minimum_mechanistic_Spearman":
                    minimum_mechanistic_spearman,

                "minimum_confidence_Spearman":
                    minimum_confidence_spearman,

                "minimum_mechanistic_top5_overlap":
                    minimum_mechanistic_top5_overlap,

                "minimum_confidence_top10_overlap":
                    minimum_confidence_top10_overlap,

                "chemicals_stable_in_top5_all_mechanistic_scenarios":
                    stable_mechanistic_top5,

                "chemicals_stable_in_top10_all_confidence_scenarios":
                    stable_confidence_top10,

                "supplementary_robustness_figure_created":
                    figure_created,

                "baseline_Phase6F2_scores_overwritten":
                    False,

                "biological_classes_recalculated":
                    False,

                "therapeutic_or_beneficial_ranking_calculated":
                    False,

                "scoring_robustness_tested":
                    True,

                "Phase6F4_status":
                    status,
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
            "===== PHASE 6F4 COMPLETION =====",
            completion.to_string(
                index=False
            ),
            "",
            "===== SCENARIO COMPARISONS =====",
            comparison_summary.to_string(
                index=False
            ),
            "",
            "===== MECHANISTIC RANK STABILITY =====",
            mechanistic_stability.sort_values(
                "baseline_rank"
            ).to_string(
                index=False
            ),
            "",
            "===== CONFIDENCE RANK STABILITY =====",
            confidence_stability.sort_values(
                "baseline_rank"
            ).to_string(
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

    if status != "completed":
        raise RuntimeError(
            "Phase 6F4 validation failed."
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            (
                "Phase 6F4 failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
