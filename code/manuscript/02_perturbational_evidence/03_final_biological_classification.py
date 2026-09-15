#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT = Path(
    "."
)

PHASE6F1_FILE = (
    PROJECT
    / "07_tables/main_tables/phase6/"
      "phase6F1_integrated_chemical_evidence_matrix.tsv"
)

PHASE6F2_FILE = (
    PROJECT
    / "07_tables/main_tables/phase6/"
      "phase6F2_dual_chemical_ranking.tsv"
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
      "phase6F3_final_biological_classification.log"
)


CLASSIFICATION_OUTPUT = (
    TABLE_DIR
    / "phase6F3_final_biological_classification.tsv"
)

CLASSIFICATION_GZ_OUTPUT = (
    PROCESSED_DIR
    / "phase6F3_final_biological_classification.tsv.gz"
)

CLASS_SUMMARY_OUTPUT = (
    TABLE_DIR
    / "phase6F3_biological_class_summary.tsv"
)

EVIDENCE_SUMMARY_OUTPUT = (
    TABLE_DIR
    / "phase6F3_evidence_strength_summary.tsv"
)

DICTIONARY_OUTPUT = (
    TABLE_DIR
    / "phase6F3_classification_dictionary.tsv"
)

COMPLETION_OUTPUT = (
    TABLE_DIR
    / "phase6F3_completion_summary.tsv"
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
EXPECTED_DIRECT_LINCS = 15
EXPECTED_CMAP_ONLY = 14
EXPECTED_REPLICATE_NOT_EVALUABLE = 1


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


def numeric_value(
    value: object,
    default: float = 0.0,
) -> float:
    converted = pd.to_numeric(
        pd.Series(
            [
                value,
            ]
        ),
        errors="coerce",
    ).iloc[
        0
    ]

    if pd.isna(
        converted
    ):
        return default

    return float(
        converted
    )


def assert_unique(
    table: pd.DataFrame,
    column: str,
    label: str,
) -> None:
    values = table[
        column
    ].dropna().astype(
        str
    ).str.strip()

    values = values.loc[
        values.ne(
            ""
        )
    ]

    duplicated = values.loc[
        values.duplicated(
            keep=False
        )
    ]

    if not duplicated.empty:
        raise RuntimeError(
            (
                f"{label} contains duplicated {column}: "
                + "|".join(
                    duplicated.head(
                        20
                    )
                )
            )
        )


def build_descriptors(
    row: pd.Series,
) -> str:
    descriptors: list[
        str
    ] = []

    flag_map = [
        (
            "CMAP_global_residual_mechanism",
            "CMAP_global_residual",
        ),
        (
            "CMAP_within_parent_residual_mechanism",
            "CMAP_within_parent_residual",
        ),
        (
            "CMAP_injury_or_mixed_residual_mechanism",
            "CMAP_injury_or_mixed",
        ),
        (
            "CMAP_early_suppression_residual_mechanism",
            "CMAP_early_suppression",
        ),
        (
            "direct_injury_activation",
            "direct_injury_activation",
        ),
        (
            "direct_injury_exceeds_late_program",
            "injury_exceeds_late_program",
        ),
        (
            "direct_early_program_suppression",
            "direct_early_program_suppression",
        ),
        (
            "direct_late_maturation_depletion",
            "direct_late_program_depletion",
        ),
        (
            "direct_late_maturation_support",
            "direct_late_program_support",
        ),
        (
            "direct_negative_developmental_shift",
            "direct_negative_developmental_shift",
        ),
        (
            "direct_positive_developmental_shift",
            "direct_positive_developmental_shift",
        ),
        (
            "direct_relative_only_result_present",
            "null_relative_result_present",
        ),
        (
            "neural_quantitative_data_available",
            "neural_context_available",
        ),
    ]

    for column, descriptor in flag_map:
        if parse_bool(
            row.get(
                column
            )
        ):
            descriptors.append(
                descriptor
            )

    reproducibility_class = clean_text(
        row.get(
            "replicate_concordant_sensitivity_class"
        )
    )

    if reproducibility_class:
        descriptors.append(
            (
                "reproducibility:"
                + reproducibility_class
            )
        )

    dominant_class = clean_text(
        row.get(
            "dominant_robust_mechanism_class"
        )
    )

    if dominant_class:
        descriptors.append(
            (
                "CMAP_dominant:"
                + dominant_class
            )
        )

    return "|".join(
        descriptors
    )


def classify_direct_chemical(
    row: pd.Series,
) -> tuple[
    str,
    str,
]:
    injury = parse_bool(
        row.get(
            "direct_injury_activation"
        )
    )

    injury_dominant = parse_bool(
        row.get(
            "direct_injury_exceeds_late_program"
        )
    )

    early_suppression = parse_bool(
        row.get(
            "direct_early_program_suppression"
        )
    )

    late_depletion = parse_bool(
        row.get(
            "direct_late_maturation_depletion"
        )
    )

    late_support = parse_bool(
        row.get(
            "direct_late_maturation_support"
        )
    )

    negative_shift = parse_bool(
        row.get(
            "direct_negative_developmental_shift"
        )
    )

    positive_shift = parse_bool(
        row.get(
            "direct_positive_developmental_shift"
        )
    )

    relative_only = parse_bool(
        row.get(
            "direct_relative_only_result_present"
        )
    )

    absolute_count = numeric_value(
        row.get(
            "primary_absolute_direction_supported_axes"
        )
    )

    if (
        injury_dominant
        and early_suppression
    ):
        return (
            "early_program_suppression_with_injury_dominance",
            (
                "Absolute quantitative evidence supports suppression "
                "of early developmental programs together with an "
                "injury/stress response that exceeds any apparent "
                "late-program contribution."
            ),
        )

    if injury_dominant:
        return (
            "injury_dominant_developmental_perturbant",
            (
                "The dominant direct quantitative signal is injury or "
                "stress, and this effect exceeds apparent late-program "
                "support."
            ),
        )

    if (
        injury
        and late_depletion
    ):
        return (
            "injury_associated_late_program_depletion",
            (
                "Injury/stress activation occurs together with "
                "depletion of late synaptic, metabolic or glial "
                "maturation programs."
            ),
        )

    if (
        injury
        and late_support
    ):
        return (
            "late_program_support_with_injury_confounding",
            (
                "A late-program signal is present, but concurrent "
                "injury/stress prevents interpretation as coherent or "
                "beneficial maturation."
            ),
        )

    if injury:
        return (
            "injury_associated_developmental_perturbant",
            (
                "Direct quantitative evidence supports an injury or "
                "stress response without sufficient evidence for a "
                "more specific developmental mechanism."
            ),
        )

    if late_depletion:
        return (
            "late_program_depletion_without_detected_injury",
            (
                "Late synaptic, metabolic or glial maturation programs "
                "are depleted without a detected absolute injury/stress "
                "signal in the primary analysis."
            ),
        )

    if early_suppression:
        return (
            "early_program_suppression_without_detected_injury",
            (
                "Early proliferative or developmental programs are "
                "suppressed without a detected absolute injury/stress "
                "signal."
            ),
        )

    if negative_shift:
        return (
            "negative_developmental_state_shift_without_resolved_mechanism",
            (
                "The quantitative profile shifts toward an earlier "
                "developmental state, but the fixed mechanism panel "
                "does not resolve the dominant cause."
            ),
        )

    if (
        late_support
        and positive_shift
    ):
        return (
            "late_program_support_with_positive_developmental_shift",
            (
                "Late-program support and a positive developmental "
                "shift are detected without an absolute injury signal. "
                "This remains a perturbational signature and is not "
                "interpreted as therapeutic maturation."
            ),
        )

    if late_support:
        return (
            "isolated_late_program_support_without_detected_injury",
            (
                "A late-program signal is supported without detected "
                "absolute injury, but evidence is insufficient to infer "
                "coordinated physiological maturation or benefit."
            ),
        )

    if positive_shift:
        return (
            "positive_developmental_shift_without_coherent_late_support",
            (
                "A positive developmental-state shift is detected "
                "without coherent support across the late synaptic, "
                "metabolic and glial maturation components."
            ),
        )

    if (
        relative_only
        and absolute_count == 0
    ):
        return (
            "null_relative_deviation_without_absolute_effect",
            (
                "The chemical differs from its matched random-gene null "
                "but does not show a supported absolute biological "
                "effect in the primary quantitative analysis."
            ),
        )

    if absolute_count > 0:
        return (
            "absolute_developmental_effect_without_fixed_mechanistic_class",
            (
                "At least one absolute developmental-axis effect is "
                "supported, but it does not map to a more specific "
                "predefined mechanistic class."
            ),
        )

    return (
        "quantitatively_evaluated_without_supported_absolute_effect",
        (
            "Direct LINCS profiles were evaluated, but no absolute "
            "developmental or mechanistic effect met the primary "
            "support criteria."
        ),
    )


def classify_CMAP_only(
    row: pd.Series,
) -> tuple[
    str,
    str,
]:
    tier = clean_text(
        row.get(
            "evidence_tier"
        )
    )

    dominant = clean_text(
        row.get(
            "dominant_robust_mechanism_class"
        )
    )

    global_residual = parse_bool(
        row.get(
            "CMAP_global_residual_mechanism"
        )
    )

    if global_residual:
        if "injury" in dominant:
            return (
                "CMAP_only_global_residual_injury_or_mixed_mechanism",
                (
                    "CMAP leave-program-out analysis supports a global "
                    "residual injury/stress or mixed early-plus-injury "
                    "mechanism, but no direct quantitative LINCS profile "
                    "was available."
                ),
            )

        if "early" in dominant:
            return (
                "CMAP_only_global_residual_early_program_suppression",
                (
                    "CMAP leave-program-out analysis supports global "
                    "residual early-program suppression, but no direct "
                    "quantitative LINCS profile was available."
                ),
            )

        return (
            "CMAP_only_global_residual_mechanism",
            (
                "A global residual CMAP mechanism is supported, but "
                "direct quantitative LINCS validation was unavailable."
            ),
        )

    if tier.startswith(
        "Tier_B"
    ):
        return (
            "CMAP_only_relative_residual_profile",
            (
                "CMAP evidence supports a relative residual profile "
                "without a globally supported fixed-panel mechanism; "
                "direct quantitative validation was unavailable."
            ),
        )

    if tier.startswith(
        "Tier_C"
    ):
        return (
            "CMAP_only_program_overlap_dependent_or_unresolved",
            (
                "The CMAP developmental-state signal is dependent on "
                "program overlap or remains mechanistically unresolved, "
                "and direct quantitative LINCS validation was "
                "unavailable."
            ),
        )

    return (
        "CMAP_only_no_fixed_panel_mechanism",
        (
            "CMAP concordance was detected without a supported residual "
            "fixed-panel mechanism, and direct quantitative LINCS "
            "validation was unavailable."
        ),
    )


def reproducibility_level(
    row: pd.Series,
) -> str:
    direct = parse_bool(
        row.get(
            "direct_LINCS_quantitative_evaluated"
        )
    )

    if not direct:
        return "not_applicable_no_direct_LINCS"

    sensitivity_class = clean_text(
        row.get(
            "replicate_concordant_sensitivity_class"
        )
    )

    mapping = {
        "all_primary_absolute_axes_bootstrap_replicated":
            "bootstrap_replicated",

        "all_primary_absolute_axes_directionally_retained":
            "directionally_retained",

        "partial_primary_axis_retention":
            "partial_directional_retention",

        "primary_absolute_support_not_retained":
            "not_retained_after_reproducibility_filter",

        "no_primary_absolute_axes_but_all_directions_stable":
            "descriptive_profile_stability",

        "no_primary_absolute_axes_mixed_directional_stability":
            "mixed_descriptive_stability",

        "not_evaluable_no_replicate_concordant_profiles":
            "not_evaluable_no_replicate_concordant_profiles",
    }

    return mapping.get(
        sensitivity_class,
        "limited_or_unclassified_reproducibility",
    )


def evidence_strength(
    row: pd.Series,
) -> str:
    direct = parse_bool(
        row.get(
            "direct_LINCS_quantitative_evaluated"
        )
    )

    confidence_class = clean_text(
        row.get(
            "evidence_confidence_class"
        )
    )

    reproducibility = clean_text(
        row.get(
            "reproducibility_evidence_level"
        )
    )

    tier = clean_text(
        row.get(
            "evidence_tier"
        )
    )

    if not direct:
        if tier.startswith(
            "Tier_A"
        ):
            return "provisional_strong_CMAP_only"

        if tier.startswith(
            "Tier_B"
        ):
            return "provisional_moderate_CMAP_only"

        return "provisional_limited_CMAP_only"

    if (
        confidence_class
        in {
            "very_high_confidence",
            "high_confidence",
        }
        and reproducibility
        == "bootstrap_replicated"
    ):
        return "strong_multilayer_evidence"

    if (
        confidence_class
        in {
            "very_high_confidence",
            "high_confidence",
        }
        and reproducibility
        == "directionally_retained"
    ):
        return "strong_directional_multilayer_evidence"

    if (
        confidence_class
        in {
            "very_high_confidence",
            "high_confidence",
            "moderate_confidence",
        }
        and reproducibility
        in {
            "bootstrap_replicated",
            "directionally_retained",
            "partial_directional_retention",
            "descriptive_profile_stability",
        }
    ):
        return "moderate_multilayer_evidence"

    if reproducibility == (
        "not_evaluable_no_replicate_concordant_profiles"
    ):
        return "moderate_primary_evidence_reproducibility_not_evaluable"

    if confidence_class == "low_confidence":
        return "limited_direct_evidence"

    return "moderate_or_limited_direct_evidence"


def main() -> None:
    required_files = [
        PHASE6F1_FILE,
        PHASE6F2_FILE,
    ]

    missing = [
        str(
            path
        )
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing Phase 6F3 input files:\n"
            + "\n".join(
                missing
            )
        )

    evidence = pd.read_csv(
        PHASE6F1_FILE,
        sep="\t",
        low_memory=False,
    )

    ranking = pd.read_csv(
        PHASE6F2_FILE,
        sep="\t",
        low_memory=False,
    )

    if len(
        evidence
    ) != EXPECTED_CHEMICALS:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_CHEMICALS} Phase 6F1 rows, "
                f"found {len(evidence)}."
            )
        )

    if len(
        ranking
    ) != EXPECTED_CHEMICALS:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_CHEMICALS} Phase 6F2 rows, "
                f"found {len(ranking)}."
            )
        )

    assert_unique(
        evidence,
        "chemical_name",
        "Phase 6F1 evidence matrix",
    )

    assert_unique(
        ranking,
        "chemical_name",
        "Phase 6F2 ranking",
    )

    ranking_columns = [
        "chemical_name",
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
        "quantitative_missingness_guardrail",
        "beneficial_interpretation_guardrail",
    ]

    missing_ranking_columns = [
        column
        for column in ranking_columns
        if column not in ranking.columns
    ]

    if missing_ranking_columns:
        raise RuntimeError(
            (
                "Missing Phase 6F2 ranking columns: "
                + "|".join(
                    missing_ranking_columns
                )
            )
        )

    data = evidence.merge(
        ranking[
            ranking_columns
        ],
        on="chemical_name",
        how="left",
        validate="one_to_one",
    )

    if data[
        "evidence_confidence_score"
    ].isna().any():
        raise RuntimeError(
            "One or more chemicals lack Phase 6F2 confidence scores."
        )

    boolean_columns = [
        "direct_LINCS_quantitative_evaluated",
        "CMAP_global_residual_mechanism",
        "CMAP_within_parent_residual_mechanism",
        "CMAP_injury_or_mixed_residual_mechanism",
        "CMAP_early_suppression_residual_mechanism",
        "direct_injury_activation",
        "direct_injury_exceeds_late_program",
        "direct_early_program_suppression",
        "direct_late_maturation_depletion",
        "direct_late_maturation_support",
        "direct_negative_developmental_shift",
        "direct_positive_developmental_shift",
        "direct_relative_only_result_present",
        "neural_quantitative_data_available",
    ]

    for column in boolean_columns:
        if column not in data.columns:
            raise RuntimeError(
                f"Missing classification column: {column}"
            )

        data[
            column
        ] = data[
            column
        ].map(
            parse_bool
        )

    biological_classes: list[
        str
    ] = []

    interpretation_statements: list[
        str
    ] = []

    claim_scopes: list[
        str
    ] = []

    for _, row in data.iterrows():
        direct = parse_bool(
            row[
                "direct_LINCS_quantitative_evaluated"
            ]
        )

        if direct:
            biological_class, statement = (
                classify_direct_chemical(
                    row
                )
            )

            claim_scope = (
                "direct_quantitative_multilayer_classification"
            )

        else:
            biological_class, statement = (
                classify_CMAP_only(
                    row
                )
            )

            claim_scope = (
                "CMAP_only_provisional_classification"
            )

        biological_classes.append(
            biological_class
        )

        interpretation_statements.append(
            statement
        )

        claim_scopes.append(
            claim_scope
        )

    data[
        "final_biological_class"
    ] = biological_classes

    data[
        "biological_interpretation_statement"
    ] = interpretation_statements

    data[
        "classification_claim_scope"
    ] = claim_scopes

    data[
        "reproducibility_evidence_level"
    ] = data.apply(
        reproducibility_level,
        axis=1,
    )

    data[
        "final_evidence_strength"
    ] = data.apply(
        evidence_strength,
        axis=1,
    )

    data[
        "mechanistic_descriptors"
    ] = data.apply(
        build_descriptors,
        axis=1,
    )

    data[
        "therapeutic_or_beneficial_interpretation"
    ] = "not_assessed_and_not_inferred"

    data[
        "classification_guardrail"
    ] = (
        "Classes describe perturbational mechanism and evidence strength; "
        "they do not establish therapeutic benefit, clinical toxicity, "
        "causality or in-vivo exposure relevance."
    )

    data[
        "direct_quantitative_missingness_is_negative_evidence"
    ] = False

    data[
        "classification_requires_Phase6F4_robustness"
    ] = True

    unclassified = data[
        "final_biological_class"
    ].fillna(
        ""
    ).eq(
        ""
    )

    if unclassified.any():
        raise RuntimeError(
            (
                "Unclassified chemicals: "
                + "|".join(
                    data.loc[
                        unclassified,
                        "chemical_name",
                    ].astype(
                        str
                    )
                )
            )
        )

    preferred_columns = [
        "chemical_name",
        "DTXSID",
        "final_biological_class",
        "biological_interpretation_statement",
        "final_evidence_strength",
        "reproducibility_evidence_level",
        "classification_claim_scope",
        "mechanistic_descriptors",

        "mechanistic_priority_scope",
        "mechanistic_priority_rank_within_scope",
        "mechanistic_priority_score_within_scope",
        "mechanistic_priority_class",
        "evidence_confidence_rank",
        "evidence_confidence_score",
        "evidence_confidence_class",

        "evidence_tier",
        "dominant_robust_mechanism_class",
        "overall_primary_interpretation",

        "direct_LINCS_quantitative_evaluated",
        "CMAP_global_residual_mechanism",
        "CMAP_within_parent_residual_mechanism",
        "CMAP_injury_or_mixed_residual_mechanism",
        "CMAP_early_suppression_residual_mechanism",

        "direct_injury_activation",
        "direct_injury_exceeds_late_program",
        "direct_early_program_suppression",
        "direct_late_maturation_depletion",
        "direct_late_maturation_support",
        "direct_negative_developmental_shift",
        "direct_positive_developmental_shift",
        "direct_relative_only_result_present",

        "primary_absolute_direction_supported_axes",
        "primary_secondary_replicated_absolute_axis_count",
        "primary_absolute_direction_retained_count",
        "primary_absolute_bootstrap_replicated_count",
        "neural_absolute_supported_axis_count",

        "replicate_concordant_signatures",
        "replicate_concordant_cells",
        "replicate_concordant_sensitivity_class",

        "therapeutic_or_beneficial_interpretation",
        "classification_guardrail",
        "direct_quantitative_missingness_is_negative_evidence",
        "classification_requires_Phase6F4_robustness",
    ]

    ordered_columns = [
        column
        for column in preferred_columns
        if column in data.columns
    ]

    remaining_columns = [
        column
        for column in data.columns
        if column not in ordered_columns
    ]

    output = data[
        ordered_columns
        + remaining_columns
    ].sort_values(
        [
            "classification_claim_scope",
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

    output.to_csv(
        CLASSIFICATION_OUTPUT,
        sep="\t",
        index=False,
    )

    output.to_csv(
        CLASSIFICATION_GZ_OUTPUT,
        sep="\t",
        index=False,
        compression="gzip",
    )

    class_summary = (
        output.groupby(
            [
                "classification_claim_scope",
                "final_biological_class",
            ],
            dropna=False,
            sort=True,
        )
        .agg(
            chemical_count=(
                "chemical_name",
                "size",
            ),
            median_priority_score=(
                "mechanistic_priority_score_within_scope",
                "median",
            ),
            median_confidence_score=(
                "evidence_confidence_score",
                "median",
            ),
            bootstrap_replicated_chemicals=(
                "reproducibility_evidence_level",
                lambda values: int(
                    (
                        values
                        == "bootstrap_replicated"
                    ).sum()
                ),
            ),
            chemical_names=(
                "chemical_name",
                lambda values: "|".join(
                    sorted(
                        values.astype(
                            str
                        )
                    )
                ),
            ),
        )
        .reset_index()
    )

    class_summary.to_csv(
        CLASS_SUMMARY_OUTPUT,
        sep="\t",
        index=False,
    )

    evidence_summary = (
        output.groupby(
            [
                "classification_claim_scope",
                "final_evidence_strength",
            ],
            dropna=False,
            sort=True,
        )
        .agg(
            chemical_count=(
                "chemical_name",
                "size",
            ),
            median_confidence_score=(
                "evidence_confidence_score",
                "median",
            ),
            chemical_names=(
                "chemical_name",
                lambda values: "|".join(
                    sorted(
                        values.astype(
                            str
                        )
                    )
                ),
            ),
        )
        .reset_index()
    )

    evidence_summary.to_csv(
        EVIDENCE_SUMMARY_OUTPUT,
        sep="\t",
        index=False,
    )

    classification_dictionary = pd.DataFrame(
        [
            {
                "classification_group":
                    "Direct quantitative adverse mechanisms",

                "class_examples":
                    (
                        "injury_dominant_developmental_perturbant;"
                        "early_program_suppression_with_injury_dominance;"
                        "injury_associated_late_program_depletion"
                    ),

                "definition":
                    (
                        "Absolute gene-matched quantitative evidence "
                        "supports injury, developmental suppression or "
                        "late-program depletion."
                    ),

                "allowed_claim":
                    (
                        "Supported perturbational mechanism within the "
                        "analysed LINCS contexts."
                    ),

                "prohibited_claim":
                    (
                        "Clinical toxicity, in-vivo causality or "
                        "human exposure risk."
                    ),
            },
            {
                "classification_group":
                    "Direct quantitative developmental-state effects",

                "class_examples":
                    (
                        "negative_developmental_state_shift_without_"
                        "resolved_mechanism;"
                        "positive_developmental_shift_without_"
                        "coherent_late_support"
                    ),

                "definition":
                    (
                        "An absolute developmental-state score is "
                        "supported, but the fixed mechanism panel does "
                        "not establish a coherent adverse or maturation "
                        "mechanism."
                    ),

                "allowed_claim":
                    "Developmental-state perturbation.",

                "prohibited_claim":
                    "Therapeutic rejuvenation or beneficial maturation.",
            },
            {
                "classification_group":
                    "Null-relative or unsupported absolute effects",

                "class_examples":
                    (
                        "null_relative_deviation_without_absolute_effect;"
                        "quantitatively_evaluated_without_supported_"
                        "absolute_effect"
                    ),

                "definition":
                    (
                        "The profile differs from a matched null or was "
                        "quantitatively evaluated, but no supported "
                        "absolute effect was established."
                    ),

                "allowed_claim":
                    "Relative or descriptive profile deviation.",

                "prohibited_claim":
                    "Absolute pathway activation or suppression.",
            },
            {
                "classification_group":
                    "CMAP-only provisional classes",

                "class_examples":
                    (
                        "CMAP_only_global_residual_injury_or_mixed_"
                        "mechanism;"
                        "CMAP_only_program_overlap_dependent_or_"
                        "unresolved"
                    ),

                "definition":
                    (
                        "Directional CMAP evidence is available, but "
                        "direct quantitative LINCS validation is absent."
                    ),

                "allowed_claim":
                    "Provisional CMAP-supported mechanism.",

                "prohibited_claim":
                    (
                        "Direct quantitative validation or biological "
                        "inactivity when LINCS data are absent."
                    ),
            },
        ]
    )

    classification_dictionary.to_csv(
        DICTIONARY_OUTPUT,
        sep="\t",
        index=False,
    )

    direct_count = int(
        output[
            "direct_LINCS_quantitative_evaluated"
        ].sum()
    )

    cmap_only_count = int(
        (
            ~output[
                "direct_LINCS_quantitative_evaluated"
            ]
        ).sum()
    )

    injury_related_count = int(
        output[
            "final_biological_class"
        ].str.contains(
            "injury",
            case=False,
            na=False,
        ).sum()
    )

    late_depletion_count = int(
        output[
            "final_biological_class"
        ].str.contains(
            "late_program_depletion",
            case=False,
            na=False,
        ).sum()
    )

    late_support_count = int(
        output[
            "final_biological_class"
        ].str.contains(
            "late_program_support",
            case=False,
            na=False,
        ).sum()
    )

    null_relative_or_no_absolute_count = int(
        output[
            "final_biological_class"
        ].isin(
            [
                "null_relative_deviation_without_absolute_effect",
                "quantitatively_evaluated_without_supported_absolute_effect",
            ]
        ).sum()
    )

    replicate_not_evaluable_count = int(
        output[
            "reproducibility_evidence_level"
        ].eq(
            "not_evaluable_no_replicate_concordant_profiles"
        ).sum()
    )

    therapeutic_labels = int(
        output[
            "therapeutic_or_beneficial_interpretation"
        ].ne(
            "not_assessed_and_not_inferred"
        ).sum()
    )

    status = (
        "completed"
        if (
            len(
                output
            )
            == EXPECTED_CHEMICALS
            and direct_count
            == EXPECTED_DIRECT_LINCS
            and cmap_only_count
            == EXPECTED_CMAP_ONLY
            and replicate_not_evaluable_count
            == EXPECTED_REPLICATE_NOT_EVALUABLE
            and int(
                unclassified.sum()
            )
            == 0
            and therapeutic_labels
            == 0
        )
        else "failed_validation"
    )

    completion = pd.DataFrame(
        [
            {
                "classified_chemicals":
                    len(
                        output
                    ),

                "direct_quantitative_classifications":
                    direct_count,

                "CMAP_only_provisional_classifications":
                    cmap_only_count,

                "unique_biological_classes":
                    output[
                        "final_biological_class"
                    ].nunique(),

                "injury_related_classifications":
                    injury_related_count,

                "late_program_depletion_classifications":
                    late_depletion_count,

                "late_program_support_classifications":
                    late_support_count,

                "null_relative_or_no_absolute_effect_classifications":
                    null_relative_or_no_absolute_count,

                "replicate_concordant_not_evaluable_chemicals":
                    replicate_not_evaluable_count,

                "unclassified_chemicals":
                    int(
                        unclassified.sum()
                    ),

                "therapeutic_or_beneficial_labels_assigned":
                    therapeutic_labels,

                "final_biological_classification_calculated":
                    True,

                "scoring_robustness_tested":
                    False,

                "Phase6F3_status":
                    status,
            }
        ]
    )

    completion.to_csv(
        COMPLETION_OUTPUT,
        sep="\t",
        index=False,
    )

    compact_columns = [
        "chemical_name",
        "final_biological_class",
        "final_evidence_strength",
        "reproducibility_evidence_level",
        "mechanistic_priority_score_within_scope",
        "evidence_confidence_score",
        "classification_claim_scope",
    ]

    log_text = "\n".join(
        [
            "===== PHASE 6F3 COMPLETION =====",
            completion.to_string(
                index=False
            ),
            "",
            "===== FINAL BIOLOGICAL CLASSIFICATION =====",
            output[
                compact_columns
            ].to_string(
                index=False
            ),
            "",
            "===== BIOLOGICAL CLASS SUMMARY =====",
            class_summary.to_string(
                index=False
            ),
            "",
            "===== EVIDENCE-STRENGTH SUMMARY =====",
            evidence_summary.to_string(
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
            "Phase 6F3 validation failed."
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            (
                "Phase 6F3 failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
