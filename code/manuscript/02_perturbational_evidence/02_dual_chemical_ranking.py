#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT = Path(
    "."
)

INPUT_FILE = (
    PROJECT
    / "07_tables/main_tables/phase6/"
      "phase6F1_integrated_chemical_evidence_matrix.tsv"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase6"
)

PROCESSED_DIR = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6F"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase6/"
      "phase6F2_dual_chemical_ranking.log"
)


DUAL_OUTPUT = (
    TABLE_DIR
    / "phase6F2_dual_chemical_ranking.tsv"
)

DUAL_GZ_OUTPUT = (
    PROCESSED_DIR
    / "phase6F2_dual_chemical_ranking.tsv.gz"
)

INTEGRATED_PRIORITY_OUTPUT = (
    TABLE_DIR
    / "phase6F2_integrated_mechanistic_priority_ranking.tsv"
)

CMAP_PROVISIONAL_OUTPUT = (
    TABLE_DIR
    / "phase6F2_CMAP_only_provisional_priority_ranking.tsv"
)

CONFIDENCE_OUTPUT = (
    TABLE_DIR
    / "phase6F2_evidence_confidence_ranking.tsv"
)

SCORING_DICTIONARY_OUTPUT = (
    TABLE_DIR
    / "phase6F2_scoring_dictionary.tsv"
)

COMPLETION_OUTPUT = (
    TABLE_DIR
    / "phase6F2_completion_summary.tsv"
)


for directory in [
    TABLE_DIR,
    PROCESSED_DIR,
    LOG_FILE.parent,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


EXPECTED_CHEMICALS = 29
EXPECTED_DIRECT_MATCHED = 15
EXPECTED_DIRECT_UNMATCHED = 14
EXPECTED_CMAP_GLOBAL_RESIDUAL = 6
EXPECTED_REPLICATE_EVALUABLE = 14


def clean_text(
    value: object,
) -> str:
    if pd.isna(value):
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


def safe_fraction(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    numerator_numeric = numeric(
        numerator
    ).fillna(
        0
    )

    denominator_numeric = numeric(
        denominator
    )

    output = np.where(
        denominator_numeric > 0,
        numerator_numeric
        / denominator_numeric,
        0.0,
    )

    return pd.Series(
        output,
        index=numerator.index,
        dtype=float,
    ).clip(
        lower=0,
        upper=1,
    )


def evidence_multiplier(
    value: object,
) -> float:
    level = clean_text(
        value
    )

    if level == "global_maxT":
        return 1.0

    if level == "FDR_only":
        return 2.0 / 3.0

    return 0.0


def tier_points(
    value: object,
) -> float:
    tier = clean_text(
        value
    )

    if tier.startswith(
        "Tier_A"
    ):
        return 20.0

    if tier.startswith(
        "Tier_B"
    ):
        return 14.0

    if tier.startswith(
        "Tier_C"
    ):
        return 8.0

    if tier.startswith(
        "Tier_D"
    ):
        return 4.0

    return 0.0


def priority_class(
    value: object,
) -> str:
    if pd.isna(
        value
    ):
        return "not_calculated"

    score = float(
        value
    )

    if score >= 70:
        return "very_high_priority"

    if score >= 50:
        return "high_priority"

    if score >= 30:
        return "moderate_priority"

    return "low_priority"


def confidence_class(
    value: object,
) -> str:
    if pd.isna(
        value
    ):
        return "not_calculated"

    score = float(
        value
    )

    if score >= 75:
        return "very_high_confidence"

    if score >= 60:
        return "high_confidence"

    if score >= 40:
        return "moderate_confidence"

    if score >= 20:
        return "low_confidence"

    return "very_low_confidence"


def reproducibility_base_points(
    value: object,
) -> float:
    sensitivity_class = clean_text(
        value
    )

    mapping = {
        "all_primary_absolute_axes_bootstrap_replicated":
            12.0,

        "all_primary_absolute_axes_directionally_retained":
            9.0,

        "partial_primary_axis_retention":
            5.0,

        "primary_absolute_support_not_retained":
            1.0,

        "no_primary_absolute_axes_but_all_directions_stable":
            5.0,

        "no_primary_absolute_axes_mixed_directional_stability":
            3.0,

        "descriptive_only_single_cell_or_low_signature_count":
            1.0,

        "not_evaluable_no_replicate_concordant_profiles":
            0.0,
    }

    return mapping.get(
        sensitivity_class,
        0.0,
    )


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            str(
                INPUT_FILE
            )
        )

    data = pd.read_csv(
        INPUT_FILE,
        sep="\t",
        low_memory=False,
    )

    if len(
        data
    ) != EXPECTED_CHEMICALS:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_CHEMICALS} chemicals, "
                f"found {len(data)}."
            )
        )

    required_columns = [
        "chemical_name",
        "DTXSID",
        "evidence_tier",
        "maxT_parent_count",
        "parents_with_global_robust_mechanism",
        "parents_with_within_parent_robust_mechanism",
        "robust_injury_parent_count",
        "robust_mixed_parent_count",
        "robust_early_suppression_parent_count",
        "dominant_class_fraction",
        "direct_LINCS_quantitative_evaluated",
        "Level5_signature_count",
        "unique_cell_count",
        "primary_global_maxT_supported_axes",
        "primary_FDR_only_supported_axes",
        "primary_absolute_direction_supported_axes",
        "primary_secondary_replicated_absolute_axis_count",
        "neural_quantitative_data_available",
        "neural_absolute_supported_axis_count",
        "replicate_concordant_signatures",
        "replicate_concordant_cells",
        "replicate_concordant_sensitivity_class",
        "primary_absolute_direction_retained_count",
        "primary_absolute_bootstrap_replicated_count",
        "overall_sign_concordance_fraction",
        "CMAP_global_residual_mechanism",
        "CMAP_within_parent_residual_mechanism",
        "direct_injury_activation",
        "direct_injury_exceeds_late_program",
        "direct_early_program_suppression",
        "direct_late_maturation_depletion",
        "direct_negative_developmental_shift",
        "injury_stress_direct_evidence_level",
        "late_minus_injury_direct_evidence_level",
        "early_program_suppression_direct_evidence_level",
        "late_maturation_support_direct_evidence_level",
        "developmental_shift_direct_evidence_level",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise RuntimeError(
            (
                "Missing required Phase 6F2 columns: "
                + "|".join(
                    missing_columns
                )
            )
        )

    data = data.copy()

    boolean_columns = [
        "direct_LINCS_quantitative_evaluated",
        "CMAP_global_residual_mechanism",
        "CMAP_within_parent_residual_mechanism",
        "direct_injury_activation",
        "direct_injury_exceeds_late_program",
        "direct_early_program_suppression",
        "direct_late_maturation_depletion",
        "direct_negative_developmental_shift",
        "neural_quantitative_data_available",
    ]

    for column in boolean_columns:
        data[
            column
        ] = data[
            column
        ].map(
            parse_bool
        )

    matched = data[
        "direct_LINCS_quantitative_evaluated"
    ]

    if int(
        matched.sum()
    ) != EXPECTED_DIRECT_MATCHED:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_DIRECT_MATCHED} direct LINCS "
                f"chemicals, found {int(matched.sum())}."
            )
        )

    if int(
        (
            ~matched
        ).sum()
    ) != EXPECTED_DIRECT_UNMATCHED:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_DIRECT_UNMATCHED} chemicals "
                f"without direct LINCS, "
                f"found {int((~matched).sum())}."
            )
        )

    # ================================================================
    # A. CMAP provisional mechanistic-priority component
    # ================================================================

    maxT_parent_count = numeric(
        data[
            "maxT_parent_count"
        ]
    ).fillna(
        0
    )

    global_parent_fraction = safe_fraction(
        data[
            "parents_with_global_robust_mechanism"
        ],
        data[
            "maxT_parent_count"
        ],
    )

    injury_mixed_parent_fraction = safe_fraction(
        numeric(
            data[
                "robust_injury_parent_count"
            ]
        ).fillna(
            0
        )
        + numeric(
            data[
                "robust_mixed_parent_count"
            ]
        ).fillna(
            0
        ),
        data[
            "maxT_parent_count"
        ],
    )

    early_suppression_parent_fraction = safe_fraction(
        data[
            "robust_early_suppression_parent_count"
        ],
        data[
            "maxT_parent_count"
        ],
    )

    data[
        "CMAP_priority_global_residual_points"
    ] = np.where(
        data[
            "CMAP_global_residual_mechanism"
        ],
        25.0,
        0.0,
    )

    data[
        "CMAP_priority_within_parent_points"
    ] = np.where(
        data[
            "CMAP_within_parent_residual_mechanism"
        ],
        10.0,
        0.0,
    )

    data[
        "CMAP_priority_injury_mixed_fraction_points"
    ] = (
        injury_mixed_parent_fraction
        * 25.0
    )

    data[
        "CMAP_priority_early_suppression_fraction_points"
    ] = (
        early_suppression_parent_fraction
        * 15.0
    )

    data[
        "CMAP_priority_global_parent_fraction_points"
    ] = (
        global_parent_fraction
        * 15.0
    )

    data[
        "CMAP_priority_parent_replication_points"
    ] = (
        maxT_parent_count
        .div(
            5.0
        )
        .clip(
            lower=0,
            upper=1,
        )
        * 10.0
    )

    cmap_priority_columns = [
        "CMAP_priority_global_residual_points",
        "CMAP_priority_within_parent_points",
        "CMAP_priority_injury_mixed_fraction_points",
        "CMAP_priority_early_suppression_fraction_points",
        "CMAP_priority_global_parent_fraction_points",
        "CMAP_priority_parent_replication_points",
    ]

    data[
        "CMAP_provisional_priority_score"
    ] = data[
        cmap_priority_columns
    ].sum(
        axis=1
    ).clip(
        lower=0,
        upper=100,
    )

    # ================================================================
    # B. Direct quantitative LINCS mechanistic-hazard component
    # ================================================================

    direct_components = [
        (
            "direct_priority_injury_activation_points",
            "direct_injury_activation",
            "injury_stress_direct_evidence_level",
            30.0,
        ),
        (
            "direct_priority_injury_exceeds_late_points",
            "direct_injury_exceeds_late_program",
            "late_minus_injury_direct_evidence_level",
            25.0,
        ),
        (
            "direct_priority_early_suppression_points",
            "direct_early_program_suppression",
            "early_program_suppression_direct_evidence_level",
            15.0,
        ),
        (
            "direct_priority_late_depletion_points",
            "direct_late_maturation_depletion",
            "late_maturation_support_direct_evidence_level",
            15.0,
        ),
        (
            "direct_priority_negative_shift_points",
            "direct_negative_developmental_shift",
            "developmental_shift_direct_evidence_level",
            15.0,
        ),
    ]

    direct_priority_columns: list[
        str
    ] = []

    for (
        output_column,
        flag_column,
        evidence_column,
        maximum_points,
    ) in direct_components:
        direct_priority_columns.append(
            output_column
        )

        data[
            output_column
        ] = np.where(
            data[
                flag_column
            ],
            data[
                evidence_column
            ].map(
                evidence_multiplier
            )
            * maximum_points,
            0.0,
        )

        data.loc[
            ~matched,
            output_column,
        ] = np.nan

    data[
        "direct_LINCS_hazard_priority_score"
    ] = data[
        direct_priority_columns
    ].sum(
        axis=1,
        min_count=1,
    )

    data.loc[
        ~matched,
        "direct_LINCS_hazard_priority_score",
    ] = np.nan

    # Integrated score exists only for direct-LINCS matched chemicals.
    data[
        "integrated_mechanistic_priority_score"
    ] = np.where(
        matched,
        (
            0.40
            * data[
                "CMAP_provisional_priority_score"
            ]
            + 0.60
            * data[
                "direct_LINCS_hazard_priority_score"
            ]
        ),
        np.nan,
    )

    data[
        "mechanistic_priority_scope"
    ] = np.where(
        matched,
        "integrated_CMAP_plus_direct_LINCS",
        "CMAP_only_provisional",
    )

    data[
        "mechanistic_priority_score_within_scope"
    ] = np.where(
        matched,
        data[
            "integrated_mechanistic_priority_score"
        ],
        data[
            "CMAP_provisional_priority_score"
        ],
    )

    data[
        "mechanistic_priority_class"
    ] = data[
        "mechanistic_priority_score_within_scope"
    ].map(
        priority_class
    )

    data[
        "mechanistic_priority_rank_within_scope"
    ] = (
        data.groupby(
            "mechanistic_priority_scope",
            sort=False,
        )[
            "mechanistic_priority_score_within_scope"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(
            int
        )
    )

    # ================================================================
    # C. Evidence-confidence score
    # ================================================================

    # CMAP evidence confidence: maximum 30 points.
    data[
        "confidence_CMAP_tier_points"
    ] = data[
        "evidence_tier"
    ].map(
        tier_points
    )

    data[
        "confidence_CMAP_parent_replication_points"
    ] = (
        maxT_parent_count
        .div(
            5.0
        )
        .clip(
            lower=0,
            upper=1,
        )
        * 5.0
    )

    data[
        "confidence_CMAP_dominant_fraction_points"
    ] = (
        numeric(
            data[
                "dominant_class_fraction"
            ]
        )
        .fillna(
            0
        )
        .clip(
            lower=0,
            upper=1,
        )
        * 5.0
    )

    data[
        "confidence_CMAP_component"
    ] = data[
        [
            "confidence_CMAP_tier_points",
            "confidence_CMAP_parent_replication_points",
            "confidence_CMAP_dominant_fraction_points",
        ]
    ].sum(
        axis=1
    )

    # Direct quantitative confidence: maximum 35 points.
    signature_count = numeric(
        data[
            "Level5_signature_count"
        ]
    ).fillna(
        0
    )

    cell_count = numeric(
        data[
            "unique_cell_count"
        ]
    ).fillna(
        0
    )

    primary_global_count = numeric(
        data[
            "primary_global_maxT_supported_axes"
        ]
    ).fillna(
        0
    )

    primary_fdr_count = numeric(
        data[
            "primary_FDR_only_supported_axes"
        ]
    ).fillna(
        0
    )

    primary_secondary_replicated = numeric(
        data[
            "primary_secondary_replicated_absolute_axis_count"
        ]
    ).fillna(
        0
    )

    data[
        "confidence_direct_availability_points"
    ] = np.where(
        matched,
        8.0,
        0.0,
    )

    data[
        "confidence_direct_signature_depth_points"
    ] = np.where(
        matched,
        (
            np.log1p(
                signature_count
            )
            / np.log1p(
                345
            )
        ).clip(
            lower=0,
            upper=1,
        )
        * 7.0,
        0.0,
    )

    data[
        "confidence_direct_cell_breadth_points"
    ] = np.where(
        matched,
        cell_count.div(
            20.0
        ).clip(
            lower=0,
            upper=1,
        )
        * 6.0,
        0.0,
    )

    data[
        "confidence_primary_global_points"
    ] = np.where(
        matched,
        primary_global_count.div(
            3.0
        ).clip(
            lower=0,
            upper=1,
        )
        * 8.0,
        0.0,
    )

    data[
        "confidence_primary_FDR_points"
    ] = np.where(
        matched,
        primary_fdr_count.div(
            3.0
        ).clip(
            lower=0,
            upper=1,
        )
        * 2.0,
        0.0,
    )

    data[
        "confidence_cross_scope_replication_points"
    ] = np.where(
        matched,
        primary_secondary_replicated.div(
            3.0
        ).clip(
            lower=0,
            upper=1,
        )
        * 4.0,
        0.0,
    )

    data[
        "confidence_direct_component"
    ] = data[
        [
            "confidence_direct_availability_points",
            "confidence_direct_signature_depth_points",
            "confidence_direct_cell_breadth_points",
            "confidence_primary_global_points",
            "confidence_primary_FDR_points",
            "confidence_cross_scope_replication_points",
        ]
    ].sum(
        axis=1
    )

    # Replicate-concordant confidence: maximum 25 points.
    primary_absolute_count = numeric(
        data[
            "primary_absolute_direction_supported_axes"
        ]
    ).fillna(
        0
    )

    retained_count = numeric(
        data[
            "primary_absolute_direction_retained_count"
        ]
    ).fillna(
        0
    )

    bootstrap_count = numeric(
        data[
            "primary_absolute_bootstrap_replicated_count"
        ]
    ).fillna(
        0
    )

    overall_sign_concordance = numeric(
        data[
            "overall_sign_concordance_fraction"
        ]
    ).fillna(
        0
    ).clip(
        lower=0,
        upper=1,
    )

    retention_fraction = np.where(
        primary_absolute_count > 0,
        retained_count
        / primary_absolute_count,
        overall_sign_concordance
        * 0.5,
    )

    bootstrap_fraction = np.where(
        primary_absolute_count > 0,
        bootstrap_count
        / primary_absolute_count,
        0.0,
    )

    retention_fraction = pd.Series(
        retention_fraction,
        index=data.index,
    ).clip(
        lower=0,
        upper=1,
    )

    bootstrap_fraction = pd.Series(
        bootstrap_fraction,
        index=data.index,
    ).clip(
        lower=0,
        upper=1,
    )

    data[
        "confidence_reproducibility_class_points"
    ] = data[
        "replicate_concordant_sensitivity_class"
    ].map(
        reproducibility_base_points
    )

    data[
        "confidence_reproducibility_direction_points"
    ] = np.where(
        matched,
        retention_fraction
        * 6.0,
        0.0,
    )

    data[
        "confidence_reproducibility_bootstrap_points"
    ] = np.where(
        matched,
        bootstrap_fraction
        * 7.0,
        0.0,
    )

    data[
        "confidence_reproducibility_component"
    ] = data[
        [
            "confidence_reproducibility_class_points",
            "confidence_reproducibility_direction_points",
            "confidence_reproducibility_bootstrap_points",
        ]
    ].sum(
        axis=1
    ).clip(
        lower=0,
        upper=25,
    )

    # Neural-context confidence: maximum 10 points.
    neural_supported_count = numeric(
        data[
            "neural_absolute_supported_axis_count"
        ]
    ).fillna(
        0
    )

    data[
        "confidence_neural_availability_points"
    ] = np.where(
        data[
            "neural_quantitative_data_available"
        ],
        4.0,
        0.0,
    )

    data[
        "confidence_neural_absolute_support_points"
    ] = np.where(
        data[
            "neural_quantitative_data_available"
        ],
        neural_supported_count.div(
            2.0
        ).clip(
            lower=0,
            upper=1,
        )
        * 6.0,
        0.0,
    )

    data[
        "confidence_neural_component"
    ] = data[
        [
            "confidence_neural_availability_points",
            "confidence_neural_absolute_support_points",
        ]
    ].sum(
        axis=1
    )

    data[
        "evidence_confidence_score"
    ] = data[
        [
            "confidence_CMAP_component",
            "confidence_direct_component",
            "confidence_reproducibility_component",
            "confidence_neural_component",
        ]
    ].sum(
        axis=1
    ).clip(
        lower=0,
        upper=100,
    )

    data[
        "evidence_confidence_class"
    ] = data[
        "evidence_confidence_score"
    ].map(
        confidence_class
    )

    data[
        "evidence_confidence_rank"
    ] = data[
        "evidence_confidence_score"
    ].rank(
        method="min",
        ascending=False,
    ).astype(
        int
    )

    data[
        "quantitative_missingness_guardrail"
    ] = np.where(
        matched,
        (
            "Integrated priority incorporates direct quantitative "
            "LINCS evidence."
        ),
        (
            "Priority is provisional and CMAP-only; lower confidence "
            "reflects missing quantitative validation and does not "
            "indicate biological inactivity."
        ),
    )

    data[
        "beneficial_interpretation_guardrail"
    ] = (
        "Priority measures adverse developmental perturbation evidence; "
        "a positive developmental-shift or late-program score is not "
        "interpreted as therapeutic benefit."
    )

    # ================================================================
    # Output tables
    # ================================================================

    ranking_columns = [
        "chemical_name",
        "DTXSID",
        "evidence_tier",
        "mechanistic_priority_scope",
        "mechanistic_priority_rank_within_scope",
        "mechanistic_priority_score_within_scope",
        "mechanistic_priority_class",
        "CMAP_provisional_priority_score",
        "direct_LINCS_hazard_priority_score",
        "integrated_mechanistic_priority_score",
        "evidence_confidence_rank",
        "evidence_confidence_score",
        "evidence_confidence_class",
        "overall_primary_interpretation",
        "dominant_robust_mechanism_class",
        "CMAP_global_residual_mechanism",
        "direct_injury_activation",
        "direct_injury_exceeds_late_program",
        "direct_early_program_suppression",
        "direct_late_maturation_depletion",
        "direct_negative_developmental_shift",
        "primary_absolute_direction_supported_axes",
        "primary_secondary_replicated_absolute_axis_count",
        "primary_absolute_direction_retained_count",
        "primary_absolute_bootstrap_replicated_count",
        "neural_absolute_supported_axis_count",
        "replicate_concordant_sensitivity_class",
        "quantitative_missingness_guardrail",
        "beneficial_interpretation_guardrail",
    ]

    dual = data[
        ranking_columns
        + cmap_priority_columns
        + direct_priority_columns
        + [
            "confidence_CMAP_component",
            "confidence_direct_component",
            "confidence_reproducibility_component",
            "confidence_neural_component",
        ]
    ].copy()

    dual = dual.sort_values(
        [
            "mechanistic_priority_scope",
            "mechanistic_priority_rank_within_scope",
            "evidence_confidence_rank",
            "chemical_name",
        ],
        ascending=[
            True,
            True,
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    dual.to_csv(
        DUAL_OUTPUT,
        sep="\t",
        index=False,
    )

    dual.to_csv(
        DUAL_GZ_OUTPUT,
        sep="\t",
        index=False,
        compression="gzip",
    )

    integrated_priority = dual.loc[
        dual[
            "mechanistic_priority_scope"
        ].eq(
            "integrated_CMAP_plus_direct_LINCS"
        )
    ].sort_values(
        [
            "mechanistic_priority_rank_within_scope",
            "evidence_confidence_rank",
        ]
    )

    integrated_priority.to_csv(
        INTEGRATED_PRIORITY_OUTPUT,
        sep="\t",
        index=False,
    )

    provisional_priority = dual.loc[
        dual[
            "mechanistic_priority_scope"
        ].eq(
            "CMAP_only_provisional"
        )
    ].sort_values(
        [
            "mechanistic_priority_rank_within_scope",
            "evidence_confidence_rank",
        ]
    )

    provisional_priority.to_csv(
        CMAP_PROVISIONAL_OUTPUT,
        sep="\t",
        index=False,
    )

    confidence_ranking = dual.sort_values(
        [
            "evidence_confidence_rank",
            "mechanistic_priority_score_within_scope",
            "chemical_name",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )

    confidence_ranking.to_csv(
        CONFIDENCE_OUTPUT,
        sep="\t",
        index=False,
    )

    scoring_dictionary = pd.DataFrame(
        [
            {
                "ranking":
                    "CMAP provisional mechanistic priority",

                "component":
                    "Global residual mechanism",

                "maximum_points":
                    25,

                "rule":
                    "25 points when at least one global residual CMAP mechanism is present.",

                "interpretive_role":
                    "Strong independent CMAP mechanism evidence.",
            },
            {
                "ranking":
                    "CMAP provisional mechanistic priority",

                "component":
                    "Within-parent residual mechanism",

                "maximum_points":
                    10,

                "rule":
                    "10 points when at least one within-parent residual mechanism is present.",

                "interpretive_role":
                    "Additional within-signature mechanistic support.",
            },
            {
                "ranking":
                    "CMAP provisional mechanistic priority",

                "component":
                    "Injury or mixed-mechanism parent fraction",

                "maximum_points":
                    25,

                "rule":
                    "Fraction of maxT parents classified as injury/stress or mixed early-plus-injury multiplied by 25.",

                "interpretive_role":
                    "Prioritizes adverse injury-dominant mechanisms.",
            },
            {
                "ranking":
                    "CMAP provisional mechanistic priority",

                "component":
                    "Early-suppression parent fraction",

                "maximum_points":
                    15,

                "rule":
                    "Fraction of maxT parents with robust early-program suppression multiplied by 15.",

                "interpretive_role":
                    "Captures proliferative or developmental-state suppression.",
            },
            {
                "ranking":
                    "CMAP provisional mechanistic priority",

                "component":
                    "Global residual parent fraction",

                "maximum_points":
                    15,

                "rule":
                    "Fraction of maxT parents retaining global mechanisms after leave-program-out analysis multiplied by 15.",

                "interpretive_role":
                    "Measures consistency across parent signatures.",
            },
            {
                "ranking":
                    "CMAP provisional mechanistic priority",

                "component":
                    "Parent-signature replication",

                "maximum_points":
                    10,

                "rule":
                    "maxT parent count divided by five and capped at one, multiplied by 10.",

                "interpretive_role":
                    "Rewards replication across CMAP parent signatures.",
            },
            {
                "ranking":
                    "Direct LINCS hazard priority",

                "component":
                    "Injury/stress activation",

                "maximum_points":
                    30,

                "rule":
                    "30 points for global maxT support; 20 points for FDR-only support.",

                "interpretive_role":
                    "Largest direct adverse-mechanism contribution.",
            },
            {
                "ranking":
                    "Direct LINCS hazard priority",

                "component":
                    "Injury exceeds late program",

                "maximum_points":
                    25,

                "rule":
                    "25 points for global maxT support; 16.67 points for FDR-only support.",

                "interpretive_role":
                    "Prevents apparent maturation from masking injury.",
            },
            {
                "ranking":
                    "Direct LINCS hazard priority",

                "component":
                    "Early-program suppression",

                "maximum_points":
                    15,

                "rule":
                    "15 points for global maxT support; 10 points for FDR-only support.",

                "interpretive_role":
                    "Captures suppression of fetal or proliferative programs.",
            },
            {
                "ranking":
                    "Direct LINCS hazard priority",

                "component":
                    "Late-program depletion",

                "maximum_points":
                    15,

                "rule":
                    "15 points for global maxT support; 10 points for FDR-only support.",

                "interpretive_role":
                    "Captures loss of synaptic, metabolic or glial maturation programs.",
            },
            {
                "ranking":
                    "Direct LINCS hazard priority",

                "component":
                    "Negative developmental shift",

                "maximum_points":
                    15,

                "rule":
                    "15 points for global maxT support; 10 points for FDR-only support.",

                "interpretive_role":
                    "Captures absolute movement toward an earlier developmental state.",
            },
            {
                "ranking":
                    "Integrated mechanistic priority",

                "component":
                    "CMAP and direct LINCS combination",

                "maximum_points":
                    100,

                "rule":
                    "40% CMAP provisional priority plus 60% direct quantitative LINCS hazard priority.",

                "interpretive_role":
                    "Calculated only for the 15 direct-LINCS matched chemicals.",
            },
            {
                "ranking":
                    "Evidence confidence",

                "component":
                    "CMAP confidence",

                "maximum_points":
                    30,

                "rule":
                    "Evidence tier, parent replication and dominant-class consistency.",

                "interpretive_role":
                    "Strength of directional and residual CMAP evidence.",
            },
            {
                "ranking":
                    "Evidence confidence",

                "component":
                    "Direct quantitative confidence",

                "maximum_points":
                    35,

                "rule":
                    "Direct availability, signature depth, cell breadth, empirical support and cross-scope replication.",

                "interpretive_role":
                    "Strength and breadth of quantitative LINCS evidence.",
            },
            {
                "ranking":
                    "Evidence confidence",

                "component":
                    "Replicate-concordant confidence",

                "maximum_points":
                    25,

                "rule":
                    "Sensitivity class, retained primary directions and hierarchical-bootstrap replication.",

                "interpretive_role":
                    "Robustness after excluding weakly concordant signatures.",
            },
            {
                "ranking":
                    "Evidence confidence",

                "component":
                    "Neural-context confidence",

                "maximum_points":
                    10,

                "rule":
                    "Neural-profile availability and neural absolute-effect support.",

                "interpretive_role":
                    "Contextual relevance to neural lineage models.",
            },
        ]
    )

    scoring_dictionary.to_csv(
        SCORING_DICTIONARY_OUTPUT,
        sep="\t",
        index=False,
    )

    integrated_score_count = int(
        data[
            "integrated_mechanistic_priority_score"
        ].notna().sum()
    )

    provisional_scope_count = int(
        data[
            "mechanistic_priority_scope"
        ].eq(
            "CMAP_only_provisional"
        ).sum()
    )

    direct_score_missing_unmatched = bool(
        data.loc[
            ~matched,
            "direct_LINCS_hazard_priority_score",
        ].isna().all()
    )

    integrated_score_missing_unmatched = bool(
        data.loc[
            ~matched,
            "integrated_mechanistic_priority_score",
        ].isna().all()
    )

    confidence_complete = bool(
        data[
            "evidence_confidence_score"
        ].notna().all()
    )

    status = (
        "completed"
        if (
            len(
                data
            )
            == EXPECTED_CHEMICALS
            and integrated_score_count
            == EXPECTED_DIRECT_MATCHED
            and provisional_scope_count
            == EXPECTED_DIRECT_UNMATCHED
            and direct_score_missing_unmatched
            and integrated_score_missing_unmatched
            and confidence_complete
            and int(
                data[
                    "CMAP_global_residual_mechanism"
                ].sum()
            )
            == EXPECTED_CMAP_GLOBAL_RESIDUAL
            and int(
                data[
                    "replicate_concordant_signatures"
                ].fillna(
                    0
                ).gt(
                    0
                ).sum()
            )
            == EXPECTED_REPLICATE_EVALUABLE
        )
        else "failed_validation"
    )

    completion = pd.DataFrame(
        [
            {
                "ranked_chemicals":
                    len(
                        data
                    ),

                "integrated_priority_chemicals":
                    integrated_score_count,

                "CMAP_only_provisional_priority_chemicals":
                    provisional_scope_count,

                "evidence_confidence_ranked_chemicals":
                    int(
                        data[
                            "evidence_confidence_score"
                        ].notna().sum()
                    ),

                "CMAP_global_residual_chemicals":
                    int(
                        data[
                            "CMAP_global_residual_mechanism"
                        ].sum()
                    ),

                "replicate_concordant_evaluable_chemicals":
                    int(
                        data[
                            "replicate_concordant_signatures"
                        ].fillna(
                            0
                        ).gt(
                            0
                        ).sum()
                    ),

                "unmatched_direct_hazard_scores_preserved_as_NA":
                    direct_score_missing_unmatched,

                "unmatched_integrated_priority_scores_preserved_as_NA":
                    integrated_score_missing_unmatched,

                "beneficial_or_therapeutic_ranking_calculated":
                    False,

                "final_biological_classification_calculated":
                    False,

                "scoring_robustness_tested":
                    False,

                "Phase6F2_status":
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
            "===== PHASE 6F2 COMPLETION =====",
            completion.to_string(
                index=False
            ),
            "",
            "===== INTEGRATED MECHANISTIC PRIORITY =====",
            integrated_priority[
                [
                    "mechanistic_priority_rank_within_scope",
                    "chemical_name",
                    "integrated_mechanistic_priority_score",
                    "mechanistic_priority_class",
                    "evidence_confidence_score",
                    "evidence_confidence_class",
                    "overall_primary_interpretation",
                ]
            ].to_string(
                index=False
            ),
            "",
            "===== CMAP-ONLY PROVISIONAL PRIORITY =====",
            provisional_priority[
                [
                    "mechanistic_priority_rank_within_scope",
                    "chemical_name",
                    "CMAP_provisional_priority_score",
                    "mechanistic_priority_class",
                    "evidence_confidence_score",
                    "evidence_confidence_class",
                    "dominant_robust_mechanism_class",
                ]
            ].to_string(
                index=False
            ),
            "",
            "===== EVIDENCE CONFIDENCE RANKING =====",
            confidence_ranking[
                [
                    "evidence_confidence_rank",
                    "chemical_name",
                    "evidence_confidence_score",
                    "evidence_confidence_class",
                    "mechanistic_priority_scope",
                    "mechanistic_priority_score_within_scope",
                ]
            ].to_string(
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
            "Phase 6F2 validation failed."
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            (
                "Phase 6F2 failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
