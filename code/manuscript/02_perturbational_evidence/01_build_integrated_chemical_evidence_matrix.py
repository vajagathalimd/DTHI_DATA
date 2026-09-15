#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


PROJECT = Path(
    "."
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase6"
)

PROCESSED_DIR = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6F"
)

LOG_DIR = (
    PROJECT
    / "09_pipeline_logs/phase6"
)


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

CMAP_TIER_FILE = (
    TABLE_DIR
    / "phase6D3_chemical_evidence_tiers.tsv"
)

LINCS_COVERAGE_FILE = (
    TABLE_DIR
    / "phase6E1B_LINCS_chemical_coverage_summary.tsv"
)

LINCS_INTERPRETATION_FILE = (
    TABLE_DIR
    / "phase6E3C_chemical_level_interpretation.tsv"
)

CROSS_SCOPE_FILE = (
    TABLE_DIR
    / "phase6E3C_cross_scope_consistency.tsv"
)

QUALITY_FILE = (
    TABLE_DIR
    / "phase6E3D1_quality_summary_by_chemical.tsv"
)

GOLD_DECISION_FILE = (
    TABLE_DIR
    / "phase6E3D2_analysis_decision.tsv"
)

REPLICATE_CONCORDANT_FILE = (
    TABLE_DIR
    / "phase6E3D3_chemical_sensitivity_summary.tsv"
)

REPLICATE_CONCORDANT_COMPLETION_FILE = (
    TABLE_DIR
    / "phase6E3D3_completion_summary.tsv"
)


# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------

INTEGRATED_GZ_OUTPUT = (
    PROCESSED_DIR
    / "phase6F1_integrated_chemical_evidence_matrix.tsv.gz"
)

INTEGRATED_TABLE_OUTPUT = (
    TABLE_DIR
    / "phase6F1_integrated_chemical_evidence_matrix.tsv"
)

AVAILABILITY_OUTPUT = (
    TABLE_DIR
    / "phase6F1_evidence_availability_summary.tsv"
)

DIRECT_SUMMARY_OUTPUT = (
    TABLE_DIR
    / "phase6F1_direct_quantitative_evidence_summary.tsv"
)

MISSINGNESS_OUTPUT = (
    TABLE_DIR
    / "phase6F1_quantitative_missingness_audit.tsv"
)

LAYER_DICTIONARY_OUTPUT = (
    TABLE_DIR
    / "phase6F1_evidence_layer_dictionary.tsv"
)

COMPLETION_OUTPUT = (
    TABLE_DIR
    / "phase6F1_completion_summary.tsv"
)

LOG_FILE = (
    LOG_DIR
    / "phase6F1_integrated_chemical_evidence_matrix.log"
)


for directory in [
    TABLE_DIR,
    PROCESSED_DIR,
    LOG_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


EXPECTED_CHEMICALS = 29
EXPECTED_DIRECT_MATCHED = 15
EXPECTED_DIRECT_UNMATCHED = 14
EXPECTED_CROSS_SCOPE_ROWS = 90
EXPECTED_AXES = 6
EXPECTED_CMAP_GLOBAL_RESIDUAL = 6
EXPECTED_NEURAL_CHEMICALS = 6
EXPECTED_REPLICATE_CONCORDANT_EVALUABLE = 14
EXPECTED_REPLICATE_CONCORDANT_NOT_EVALUABLE = 1


AXES = [
    "developmental_shift",
    "early_program_suppression",
    "late_maturation_support",
    "injury_stress",
    "late_minus_injury",
    "late_minus_early",
]


def clean_text(
    value: object,
) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "na",
    }:
        return ""

    return text


def normalize_chemical(
    value: object,
) -> str:
    text = clean_text(
        value
    ).lower()

    text = re.sub(
        r"[-_/]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def parse_bool(
    value: object,
) -> bool:
    if isinstance(
        value,
        bool,
    ):
        return value

    if pd.isna(value):
        return False

    return str(
        value
    ).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def assert_unique(
    table: pd.DataFrame,
    column: str,
    label: str,
) -> None:
    series = table[
        column
    ]

    present = (
        series.notna()
        & series.astype(
            str
        ).str.strip().ne(
            ""
        )
        & series.astype(
            str
        ).str.lower().ne(
            "nan"
        )
    )

    present_values = series.loc[
        present
    ]

    duplicated = present_values.loc[
        present_values.duplicated(
            keep=False
        )
    ]

    if not duplicated.empty:
        raise RuntimeError(
            (
                f"{label} contains duplicated nonmissing values "
                f"in {column}: "
                + "|".join(
                    duplicated.astype(str).head(20)
                )
            )
        )


def starts_with(
    value: object,
    prefix: str,
) -> bool:
    return clean_text(
        value
    ).lower().startswith(
        prefix.lower()
    )


def contains_text(
    value: object,
    pattern: str,
) -> bool:
    return pattern.lower() in clean_text(
        value
    ).lower()


def evidence_level_from_interpretation(
    value: object,
) -> str:
    text = clean_text(
        value
    )

    if not text:
        return ""

    if text.endswith(
        "_global_maxT"
    ):
        return "global_maxT"

    if text.endswith(
        "_FDR_only"
    ):
        return "FDR_only"

    if text.startswith(
        "relative_"
    ):
        return "null_relative_only"

    if text == "not_empirically_supported":
        return "none"

    return "none"


def tier_rank(
    value: object,
) -> int:
    text = clean_text(
        value
    )

    if text.startswith(
        "Tier_A"
    ):
        return 1

    if text.startswith(
        "Tier_B"
    ):
        return 2

    if text.startswith(
        "Tier_C"
    ):
        return 3

    if text.startswith(
        "Tier_D"
    ):
        return 4

    return 9


def main() -> None:
    required_files = [
        CMAP_TIER_FILE,
        LINCS_COVERAGE_FILE,
        LINCS_INTERPRETATION_FILE,
        CROSS_SCOPE_FILE,
        QUALITY_FILE,
        GOLD_DECISION_FILE,
        REPLICATE_CONCORDANT_FILE,
        REPLICATE_CONCORDANT_COMPLETION_FILE,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required Phase 6F1 inputs:\n"
            + "\n".join(
                missing_files
            )
        )

    cmap = pd.read_csv(
        CMAP_TIER_FILE,
        sep="\t",
        low_memory=False,
    )

    coverage = pd.read_csv(
        LINCS_COVERAGE_FILE,
        sep="\t",
        low_memory=False,
    )

    interpretation = pd.read_csv(
        LINCS_INTERPRETATION_FILE,
        sep="\t",
        low_memory=False,
    )

    cross_scope = pd.read_csv(
        CROSS_SCOPE_FILE,
        sep="\t",
        low_memory=False,
    )

    quality = pd.read_csv(
        QUALITY_FILE,
        sep="\t",
        low_memory=False,
    )

    gold_decision = pd.read_csv(
        GOLD_DECISION_FILE,
        sep="\t",
        low_memory=False,
    )

    replicate_concordant = pd.read_csv(
        REPLICATE_CONCORDANT_FILE,
        sep="\t",
        low_memory=False,
    )

    replicate_completion = pd.read_csv(
        REPLICATE_CONCORDANT_COMPLETION_FILE,
        sep="\t",
        low_memory=False,
    )

    if len(
        cmap
    ) != EXPECTED_CHEMICALS:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_CHEMICALS} CMAP chemicals, "
                f"found {len(cmap)}."
            )
        )

    if len(
        coverage
    ) != EXPECTED_CHEMICALS:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_CHEMICALS} LINCS coverage rows, "
                f"found {len(coverage)}."
            )
        )

    if len(
        interpretation
    ) != EXPECTED_DIRECT_MATCHED:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_DIRECT_MATCHED} direct LINCS "
                f"interpretation rows, found {len(interpretation)}."
            )
        )

    if len(
        cross_scope
    ) != EXPECTED_CROSS_SCOPE_ROWS:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_CROSS_SCOPE_ROWS} cross-scope rows, "
                f"found {len(cross_scope)}."
            )
        )

    if cross_scope[
        "axis_name"
    ].nunique() != EXPECTED_AXES:
        raise RuntimeError(
            "Unexpected number of biological score axes."
        )

    if len(
        quality
    ) != EXPECTED_DIRECT_MATCHED:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_DIRECT_MATCHED} chemical-level "
                f"quality rows, found {len(quality)}."
            )
        )

    if len(
        replicate_concordant
    ) != EXPECTED_DIRECT_MATCHED:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_DIRECT_MATCHED} replicate-"
                f"concordant chemical rows, "
                f"found {len(replicate_concordant)}."
            )
        )

    if len(
        gold_decision
    ) != 1:
        raise RuntimeError(
            "Expected one Phase 6E3D2 decision row."
        )

    if len(
        replicate_completion
    ) != 1:
        raise RuntimeError(
            "Expected one Phase 6E3D3 completion row."
        )

    replicate_status = clean_text(
        replicate_completion[
            "Phase6E3D3_status"
        ].iloc[
            0
        ]
    )

    if replicate_status != "completed":
        raise RuntimeError(
            (
                "Phase 6E3D3 is not completed. "
                f"Recorded status: {replicate_status}"
            )
        )

    assert_unique(
        cmap,
        "DTXSID",
        "CMAP chemical tiers",
    )

    assert_unique(
        coverage,
        "DTXSID",
        "LINCS chemical coverage",
    )

    cmap = cmap.copy()

    cmap[
        "chemical_normalized"
    ] = cmap[
        "chemical_name"
    ].map(
        normalize_chemical
    )

    assert_unique(
        cmap,
        "chemical_normalized",
        "CMAP normalized chemical names",
    )

    coverage = coverage.copy()

    coverage[
        "chemical_normalized_coverage"
    ] = coverage[
        "chemical_query"
    ].map(
        normalize_chemical
    )

    interpretation = interpretation.copy()

    interpretation[
        "chemical_normalized"
    ] = interpretation[
        "chemical_query"
    ].map(
        normalize_chemical
    )

    quality = quality.copy()

    quality[
        "chemical_normalized"
    ] = quality[
        "chemical_query"
    ].map(
        normalize_chemical
    )

    replicate_concordant = replicate_concordant.copy()

    replicate_concordant[
        "chemical_normalized"
    ] = replicate_concordant[
        "chemical_query"
    ].map(
        normalize_chemical
    )

    cross_scope = cross_scope.copy()

    cross_scope[
        "chemical_normalized"
    ] = cross_scope[
        "chemical_query"
    ].map(
        normalize_chemical
    )

    assert_unique(
        interpretation,
        "chemical_normalized",
        "Phase 6E3C interpretation",
    )

    assert_unique(
        quality,
        "chemical_normalized",
        "Phase 6E3D1 quality summary",
    )

    assert_unique(
        replicate_concordant,
        "chemical_normalized",
        "Phase 6E3D3 sensitivity summary",
    )

    assert_unique(
        coverage,
        "chemical_normalized_coverage",
        "LINCS normalized chemical names",
    )

    coverage_merge = coverage[
        [
            "DTXSID",
            "chemical_query",
            "chemical_normalized_coverage",
            "match_status",
            "match_method",
            "matched_perturbagen_names",
            "matched_pert_ids",
            "matched_pert_id_count",
            "Level5_signature_count",
            "unique_cell_count",
            "cell_ids",
            "unique_dose_label_count",
            "dose_labels",
            "unique_time_label_count",
            "time_labels",
            "perturbation_types",
            "signature_metrics_found",
        ]
    ].copy()

    coverage_merge = coverage_merge.rename(
        columns={
            "DTXSID":
                "DTXSID_LINCS",
        }
    )

    integrated = cmap.merge(
        coverage_merge,
        left_on="chemical_normalized",
        right_on="chemical_normalized_coverage",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_LINCS",
        ),
    )

    cmap_dtxsid = integrated[
        "DTXSID"
    ].fillna(
        ""
    ).astype(
        str
    ).str.strip()

    lincs_dtxsid = integrated[
        "DTXSID_LINCS"
    ].fillna(
        ""
    ).astype(
        str
    ).str.strip()

    dtxsid_conflict = integrated.loc[
        cmap_dtxsid.ne(
            ""
        )
        & lincs_dtxsid.ne(
            ""
        )
        & cmap_dtxsid.ne(
            lincs_dtxsid
        )
    ]

    if not dtxsid_conflict.empty:
        raise RuntimeError(
            (
                "CMAP and LINCS DTXSID disagreement after "
                "chemical-name matching: "
                + "|".join(
                    dtxsid_conflict[
                        "chemical_name"
                    ].astype(
                        str
                    )
                )
            )
        )

    integrated[
        "DTXSID_match_status"
    ] = np.select(
        [
            cmap_dtxsid.ne(
                ""
            )
            & lincs_dtxsid.ne(
                ""
            )
            & cmap_dtxsid.eq(
                lincs_dtxsid
            ),

            cmap_dtxsid.eq(
                ""
            )
            & lincs_dtxsid.ne(
                ""
            ),

            cmap_dtxsid.ne(
                ""
            )
            & lincs_dtxsid.eq(
                ""
            ),

            cmap_dtxsid.eq(
                ""
            )
            & lincs_dtxsid.eq(
                ""
            ),
        ],
        [
            "concordant_in_both_sources",
            "missing_in_CMAP_available_in_LINCS",
            "available_in_CMAP_missing_in_LINCS",
            "missing_in_both_sources",
        ],
        default="unclassified",
    )


    if integrated[
        "match_status"
    ].isna().any():
        missing = integrated.loc[
            integrated[
                "match_status"
            ].isna(),
            "DTXSID",
        ].astype(
            str
        ).tolist()

        raise RuntimeError(
            (
                "Missing LINCS coverage records for: "
                + "|".join(
                    missing
                )
            )
        )

    matched_mask = (
        integrated[
            "match_status"
        ]
        .astype(str)
        .str.lower()
        .eq(
            "matched"
        )
    )

    if int(
        matched_mask.sum()
    ) != EXPECTED_DIRECT_MATCHED:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_DIRECT_MATCHED} direct LINCS "
                f"matches, found {int(matched_mask.sum())}."
            )
        )

    if int(
        (~matched_mask).sum()
    ) != EXPECTED_DIRECT_UNMATCHED:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_DIRECT_UNMATCHED} unmatched "
                f"chemicals, found {int((~matched_mask).sum())}."
            )
        )

    interpretation_columns = [
        column
        for column in interpretation.columns
        if column not in {
            "chemical_query",
            "chemical_normalized",
            "evidence_tier",
        }
    ]

    integrated = integrated.merge(
        interpretation[
            [
                "chemical_normalized",
            ]
            + interpretation_columns
        ],
        on="chemical_normalized",
        how="left",
        validate="one_to_one",
    )

    quality_columns = [
        column
        for column in quality.columns
        if column not in {
            "chemical_query",
            "chemical_normalized",
        }
    ]

    integrated = integrated.merge(
        quality[
            [
                "chemical_normalized",
            ]
            + quality_columns
        ],
        on="chemical_normalized",
        how="left",
        validate="one_to_one",
    )

    replicate_columns = [
        column
        for column in replicate_concordant.columns
        if column not in {
            "chemical_query",
            "chemical_normalized",
        }
    ]

    integrated = integrated.merge(
        replicate_concordant[
            [
                "chemical_normalized",
            ]
            + replicate_columns
        ],
        on="chemical_normalized",
        how="left",
        validate="one_to_one",
    )

    matched_without_interpretation = integrated.loc[
        matched_mask
        & integrated[
            "overall_primary_interpretation"
        ].isna()
    ]

    if not matched_without_interpretation.empty:
        raise RuntimeError(
            (
                "Direct LINCS-matched chemicals are missing "
                "Phase 6E3C interpretation: "
                + "|".join(
                    matched_without_interpretation[
                        "chemical_name"
                    ].astype(str)
                )
            )
        )

    unmatched_with_interpretation = integrated.loc[
        ~matched_mask
        & integrated[
            "overall_primary_interpretation"
        ].notna()
    ]

    if not unmatched_with_interpretation.empty:
        raise RuntimeError(
            (
                "Chemicals without direct LINCS profiles unexpectedly "
                "contain quantitative interpretations."
            )
        )

    # -----------------------------------------------------------------
    # Cross-scope and neural evidence
    # -----------------------------------------------------------------

    boolean_columns = [
        "primary_absolute_direction_supported",
        "secondary_absolute_direction_supported",
        "observed_sign_concordant",
        "empirical_direction_concordant",
        "absolute_support_replicated",
        "neural_absolute_direction_supported",
    ]

    for column in boolean_columns:
        cross_scope[
            column
        ] = cross_scope[
            column
        ].map(
            parse_bool
        )

    cross_rows: list[
        dict[str, object]
    ] = []

    for chemical, group in cross_scope.groupby(
        "chemical_normalized",
        sort=True,
    ):
        if len(
            group
        ) != EXPECTED_AXES:
            raise RuntimeError(
                (
                    f"Expected {EXPECTED_AXES} axes for {chemical}, "
                    f"found {len(group)}."
                )
            )

        row: dict[
            str,
            object
        ] = {
            "chemical_normalized":
                chemical,

            "cross_scope_axis_count":
                len(
                    group
                ),

            "primary_absolute_supported_axis_count":
                int(
                    group[
                        "primary_absolute_direction_supported"
                    ].sum()
                ),

            "secondary_absolute_supported_axis_count":
                int(
                    group[
                        "secondary_absolute_direction_supported"
                    ].sum()
                ),

            "primary_secondary_replicated_absolute_axis_count":
                int(
                    group[
                        "absolute_support_replicated"
                    ].sum()
                ),

            "primary_secondary_observed_sign_concordant_axis_count":
                int(
                    group[
                        "observed_sign_concordant"
                    ].sum()
                ),

            "primary_secondary_empirical_direction_concordant_axis_count":
                int(
                    group[
                        "empirical_direction_concordant"
                    ].sum()
                ),

            "neural_axis_count_available":
                int(
                    group[
                        "neural_observed_score"
                    ].notna().sum()
                ),

            "neural_absolute_supported_axis_count":
                int(
                    group[
                        "neural_absolute_direction_supported"
                    ].sum()
                ),
        }

        for axis in AXES:
            axis_group = group.loc[
                group[
                    "axis_name"
                ].eq(
                    axis
                )
            ]

            if len(
                axis_group
            ) != 1:
                raise RuntimeError(
                    (
                        f"Expected one {axis} row for {chemical}, "
                        f"found {len(axis_group)}."
                    )
                )

            axis_row = axis_group.iloc[
                0
            ]

            row[
                f"{axis}_primary_secondary_absolute_replicated"
            ] = bool(
                axis_row[
                    "absolute_support_replicated"
                ]
            )

            row[
                f"{axis}_neural_score"
            ] = axis_row[
                "neural_observed_score"
            ]

            row[
                f"{axis}_neural_evidence_level"
            ] = clean_text(
                axis_row[
                    "neural_evidence_level"
                ]
            )

            row[
                f"{axis}_neural_absolute_supported"
            ] = bool(
                axis_row[
                    "neural_absolute_direction_supported"
                ]
            )

            row[
                f"{axis}_neural_interpretation"
            ] = clean_text(
                axis_row[
                    "neural_biological_interpretation"
                ]
            )

        cross_rows.append(
            row
        )

    cross_summary = pd.DataFrame(
        cross_rows
    )

    assert_unique(
        cross_summary,
        "chemical_normalized",
        "Cross-scope summary",
    )

    integrated = integrated.merge(
        cross_summary,
        on="chemical_normalized",
        how="left",
        validate="one_to_one",
    )

    matched_without_cross_scope = integrated.loc[
        matched_mask
        & integrated[
            "cross_scope_axis_count"
        ].isna()
    ]

    if not matched_without_cross_scope.empty:
        raise RuntimeError(
            (
                "Direct LINCS-matched chemicals are missing "
                "cross-scope evidence."
            )
        )

    # -----------------------------------------------------------------
    # Core evidence flags
    # -----------------------------------------------------------------

    integrated[
        "direct_LINCS_quantitative_evaluated"
    ] = matched_mask

    integrated[
        "direct_LINCS_quantitative_status"
    ] = np.where(
        matched_mask,
        "quantitatively_evaluated",
        "not_evaluated_no_direct_LINCS_match",
    )

    integrated[
        "CMAP_global_residual_mechanism"
    ] = pd.to_numeric(
        integrated[
            "parents_with_global_robust_mechanism"
        ],
        errors="coerce",
    ).fillna(
        0
    ).gt(
        0
    )

    integrated[
        "CMAP_within_parent_residual_mechanism"
    ] = pd.to_numeric(
        integrated[
            "parents_with_within_parent_robust_mechanism"
        ],
        errors="coerce",
    ).fillna(
        0
    ).gt(
        0
    )

    integrated[
        "CMAP_injury_or_mixed_residual_mechanism"
    ] = (
        pd.to_numeric(
            integrated[
                "robust_injury_parent_count"
            ],
            errors="coerce",
        ).fillna(
            0
        ).gt(
            0
        )
        | pd.to_numeric(
            integrated[
                "robust_mixed_parent_count"
            ],
            errors="coerce",
        ).fillna(
            0
        ).gt(
            0
        )
    )

    integrated[
        "CMAP_early_suppression_residual_mechanism"
    ] = pd.to_numeric(
        integrated[
            "robust_early_suppression_parent_count"
        ],
        errors="coerce",
    ).fillna(
        0
    ).gt(
        0
    )

    integrated[
        "CMAP_robust_late_maturation_mechanism"
    ] = pd.to_numeric(
        integrated[
            "robust_late_maturation_parent_count"
        ],
        errors="coerce",
    ).fillna(
        0
    ).gt(
        0
    )

    integrated[
        "direct_injury_activation"
    ] = integrated[
        "injury_stress_interpretation"
    ].map(
        lambda value: starts_with(
            value,
            "injury_stress_activation",
        )
    )

    integrated[
        "direct_injury_exceeds_late_program"
    ] = integrated[
        "late_minus_injury_interpretation"
    ].map(
        lambda value: starts_with(
            value,
            "injury_exceeds_late_program",
        )
    )

    integrated[
        "direct_late_maturation_support"
    ] = integrated[
        "late_maturation_support_interpretation"
    ].map(
        lambda value: starts_with(
            value,
            "late_maturation_program_support",
        )
    )

    integrated[
        "direct_late_maturation_depletion"
    ] = integrated[
        "late_maturation_support_interpretation"
    ].map(
        lambda value: starts_with(
            value,
            "late_maturation_program_depletion",
        )
    )

    integrated[
        "direct_early_program_suppression"
    ] = integrated[
        "early_program_suppression_interpretation"
    ].map(
        lambda value: starts_with(
            value,
            "early_program_suppression",
        )
    )

    integrated[
        "direct_positive_developmental_shift"
    ] = integrated[
        "developmental_shift_interpretation"
    ].map(
        lambda value: starts_with(
            value,
            "positive_developmental_state_shift",
        )
    )

    integrated[
        "direct_negative_developmental_shift"
    ] = integrated[
        "developmental_shift_interpretation"
    ].map(
        lambda value: starts_with(
            value,
            "negative_developmental_state_shift",
        )
    )

    integrated[
        "direct_relative_only_result_present"
    ] = pd.to_numeric(
        integrated[
            "primary_null_relative_only_axes"
        ],
        errors="coerce",
    ).fillna(
        0
    ).gt(
        0
    )

    for axis in AXES:
        interpretation_column = (
            f"{axis}_interpretation"
        )

        integrated[
            f"{axis}_direct_evidence_level"
        ] = integrated[
            interpretation_column
        ].map(
            evidence_level_from_interpretation
        )

    integrated[
        "neural_quantitative_data_available"
    ] = pd.to_numeric(
        integrated[
            "neural_axis_count_available"
        ],
        errors="coerce",
    ).fillna(
        0
    ).gt(
        0
    )

    integrated[
        "replicate_concordant_profiles_available"
    ] = pd.to_numeric(
        integrated[
            "replicate_concordant_signatures"
        ],
        errors="coerce",
    ).fillna(
        0
    ).gt(
        0
    )

    integrated[
        "replicate_concordant_multicell_evaluable"
    ] = integrated[
        "coverage_class"
    ].fillna(
        ""
    ).eq(
        "adequate_multicell"
    )

    integrated[
        "replicate_concordant_not_evaluable"
    ] = integrated[
        "replicate_concordant_sensitivity_class"
    ].fillna(
        ""
    ).eq(
        "not_evaluable_no_replicate_concordant_profiles"
    )

    integrated[
        "replicate_concordant_full_bootstrap_replication"
    ] = integrated[
        "replicate_concordant_sensitivity_class"
    ].fillna(
        ""
    ).eq(
        "all_primary_absolute_axes_bootstrap_replicated"
    )

    integrated[
        "replicate_concordant_all_directions_retained"
    ] = integrated[
        "replicate_concordant_sensitivity_class"
    ].fillna(
        ""
    ).isin(
        [
            "all_primary_absolute_axes_bootstrap_replicated",
            "all_primary_absolute_axes_directionally_retained",
        ]
    )

    integrated[
        "replicate_concordant_partial_retention"
    ] = integrated[
        "replicate_concordant_sensitivity_class"
    ].fillna(
        ""
    ).eq(
        "partial_primary_axis_retention"
    )

    integrated[
        "replicate_concordant_descriptive_stability"
    ] = integrated[
        "replicate_concordant_sensitivity_class"
    ].fillna(
        ""
    ).str.startswith(
        "no_primary_absolute_axes"
    )

    gold_status = clean_text(
        gold_decision[
            "status"
        ].iloc[
            0
        ]
    )

    integrated[
        "gold_restricted_sensitivity_status"
    ] = gold_status

    integrated[
        "gold_restricted_sensitivity_used"
    ] = False

    integrated[
        "replicate_concordant_sensitivity_used"
    ] = matched_mask

    # -----------------------------------------------------------------
    # Availability and missingness
    # -----------------------------------------------------------------

    integrated[
        "evidence_availability_class"
    ] = np.select(
        [
            matched_mask
            & integrated[
                "neural_quantitative_data_available"
            ]
            & integrated[
                "replicate_concordant_profiles_available"
            ],

            matched_mask
            & ~integrated[
                "neural_quantitative_data_available"
            ]
            & integrated[
                "replicate_concordant_profiles_available"
            ],

            matched_mask
            & ~integrated[
                "replicate_concordant_profiles_available"
            ],

            ~matched_mask
            & integrated[
                "CMAP_global_residual_mechanism"
            ],

            ~matched_mask
            & ~integrated[
                "CMAP_global_residual_mechanism"
            ],
        ],
        [
            "CMAP_plus_direct_LINCS_plus_neural_plus_reproducibility",
            "CMAP_plus_direct_LINCS_plus_reproducibility_no_neural",
            "CMAP_plus_direct_LINCS_no_reproducibility_subset",
            "CMAP_only_global_residual_mechanism",
            "CMAP_only_without_global_residual_mechanism",
        ],
        default="unclassified",
    )

    integrated[
        "direct_quantitative_missingness_reason"
    ] = np.where(
        matched_mask,
        "",
        "No exact matched GSE70138 Level 5 perturbagen profile",
    )

    integrated[
        "direct_quantitative_missingness_treatment"
    ] = np.where(
        matched_mask,
        "Observed quantitative evidence retained",
        (
            "Quantitative fields remain NA; absence of a profile is "
            "not interpreted as absence of biological activity"
        ),
    )

    direct_quantitative_columns = [
        "overall_primary_interpretation",
        "developmental_shift_observed_score",
        "developmental_shift_empirical_z",
        "early_program_suppression_observed_score",
        "early_program_suppression_empirical_z",
        "late_maturation_support_observed_score",
        "late_maturation_support_empirical_z",
        "injury_stress_observed_score",
        "injury_stress_empirical_z",
        "late_minus_injury_observed_score",
        "late_minus_injury_empirical_z",
        "late_minus_early_observed_score",
        "late_minus_early_empirical_z",
    ]

    unmatched_quantitative = integrated.loc[
        ~matched_mask,
        direct_quantitative_columns,
    ]

    unmatched_quantitative_preserved_as_na = bool(
        not unmatched_quantitative.notna().any().any()
    )

    if not unmatched_quantitative_preserved_as_na:
        raise RuntimeError(
            (
                "One or more chemicals without direct LINCS coverage "
                "contain nonmissing quantitative values."
            )
        )

    # -----------------------------------------------------------------
    # Sorting and outputs
    # -----------------------------------------------------------------

    integrated[
        "evidence_tier_rank"
    ] = integrated[
        "evidence_tier"
    ].map(
        tier_rank
    )

    preferred_columns = [
        "chemical_key",
        "chemical_name",
        "chemical_display",
        "chemical_normalized",
        "DTXSID",

        "evidence_tier",
        "evidence_tier_label",
        "evidence_tier_rank",
        "maxT_parent_count",
        "parents_with_global_robust_mechanism",
        "parents_with_within_parent_robust_mechanism",
        "robust_injury_parent_count",
        "robust_mixed_parent_count",
        "robust_early_suppression_parent_count",
        "robust_late_maturation_parent_count",
        "program_overlap_dependent_parent_count",
        "no_fixed_panel_mechanism_parent_count",
        "dominant_robust_mechanism_class",
        "dominant_class_fraction",
        "median_developmental_direction_score",
        "minimum_maxT_maturation_FWER_p",
        "CMAP_global_residual_mechanism",
        "CMAP_within_parent_residual_mechanism",
        "CMAP_injury_or_mixed_residual_mechanism",
        "CMAP_early_suppression_residual_mechanism",
        "CMAP_robust_late_maturation_mechanism",

        "match_status",
        "match_method",
        "matched_perturbagen_names",
        "matched_pert_ids",
        "Level5_signature_count",
        "unique_cell_count",
        "unique_dose_label_count",
        "unique_time_label_count",
        "direct_LINCS_quantitative_status",
        "direct_LINCS_quantitative_evaluated",

        "overall_primary_interpretation",
        "primary_global_maxT_supported_axes",
        "primary_FDR_only_supported_axes",
        "primary_absolute_direction_supported_axes",
        "primary_null_relative_only_axes",

        "developmental_shift_observed_score",
        "developmental_shift_empirical_z",
        "developmental_shift_interpretation",
        "early_program_suppression_observed_score",
        "early_program_suppression_empirical_z",
        "early_program_suppression_interpretation",
        "late_maturation_support_observed_score",
        "late_maturation_support_empirical_z",
        "late_maturation_support_interpretation",
        "injury_stress_observed_score",
        "injury_stress_empirical_z",
        "injury_stress_interpretation",
        "late_minus_injury_observed_score",
        "late_minus_injury_empirical_z",
        "late_minus_injury_interpretation",
        "late_minus_early_observed_score",
        "late_minus_early_empirical_z",
        "late_minus_early_interpretation",

        "direct_injury_activation",
        "direct_injury_exceeds_late_program",
        "direct_late_maturation_support",
        "direct_late_maturation_depletion",
        "direct_early_program_suppression",
        "direct_positive_developmental_shift",
        "direct_negative_developmental_shift",
        "direct_relative_only_result_present",

        "primary_absolute_supported_axis_count",
        "secondary_absolute_supported_axis_count",
        "primary_secondary_replicated_absolute_axis_count",
        "primary_secondary_observed_sign_concordant_axis_count",
        "primary_secondary_empirical_direction_concordant_axis_count",
        "neural_axis_count_available",
        "neural_absolute_supported_axis_count",
        "neural_quantitative_data_available",

        "replicate_concordant_signatures",
        "replicate_concordant_cells",
        "coverage_class",
        "available_axis_count",
        "overall_sign_concordant_axis_count",
        "overall_sign_concordance_fraction",
        "primary_absolute_supported_axis_count",
        "primary_absolute_direction_retained_count",
        "primary_absolute_bootstrap_replicated_count",
        "replicate_concordant_sensitivity_class",
        "replicate_concordant_profiles_available",
        "replicate_concordant_multicell_evaluable",
        "replicate_concordant_not_evaluable",
        "replicate_concordant_full_bootstrap_replication",
        "replicate_concordant_all_directions_retained",
        "replicate_concordant_partial_retention",
        "replicate_concordant_descriptive_stability",

        "valid_gold_metric_count",
        "invalid_gold_metric_count",
        "reconstructed_gold_signature_count",
        "unique_cells_gold",
        "neural_gold_signature_count",
        "gold_restricted_sensitivity_status",
        "gold_restricted_sensitivity_used",
        "replicate_concordant_sensitivity_used",

        "evidence_availability_class",
        "direct_quantitative_missingness_reason",
        "direct_quantitative_missingness_treatment",
    ]

    ordered_columns: list[
        str
    ] = []

    for column in preferred_columns:
        if (
            column in integrated.columns
            and column not in ordered_columns
        ):
            ordered_columns.append(
                column
            )

    remaining_columns = [
        column
        for column in integrated.columns
        if column not in ordered_columns
    ]

    integrated = integrated[
        ordered_columns
        + remaining_columns
    ].sort_values(
        [
            "evidence_tier_rank",
            "parents_with_global_robust_mechanism",
            "chemical_name",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    integrated.to_csv(
        INTEGRATED_GZ_OUTPUT,
        sep="\t",
        index=False,
        compression="gzip",
    )

    integrated.to_csv(
        INTEGRATED_TABLE_OUTPUT,
        sep="\t",
        index=False,
    )

    availability_summary = (
        integrated.groupby(
            "evidence_availability_class",
            dropna=False,
            sort=True,
        )
        .agg(
            chemical_count=(
                "chemical_name",
                "size",
            ),

            CMAP_global_residual_chemicals=(
                "CMAP_global_residual_mechanism",
                "sum",
            ),

            direct_LINCS_evaluated_chemicals=(
                "direct_LINCS_quantitative_evaluated",
                "sum",
            ),

            neural_data_chemicals=(
                "neural_quantitative_data_available",
                "sum",
            ),

            replicate_concordant_profile_chemicals=(
                "replicate_concordant_profiles_available",
                "sum",
            ),

            total_Level5_signatures=(
                "Level5_signature_count",
                "sum",
            ),

            total_replicate_concordant_signatures=(
                "replicate_concordant_signatures",
                "sum",
            ),
        )
        .reset_index()
    )

    availability_summary.to_csv(
        AVAILABILITY_OUTPUT,
        sep="\t",
        index=False,
    )

    direct_summary_columns = [
        "chemical_name",
        "DTXSID",
        "evidence_tier",
        "Level5_signature_count",
        "unique_cell_count",
        "neural_quantitative_data_available",
        "replicate_concordant_signatures",
        "replicate_concordant_cells",
        "replicate_concordant_sensitivity_class",
        "overall_primary_interpretation",
        "primary_global_maxT_supported_axes",
        "primary_FDR_only_supported_axes",
        "primary_absolute_direction_supported_axes",
        "primary_null_relative_only_axes",
        "primary_secondary_replicated_absolute_axis_count",
        "neural_absolute_supported_axis_count",
        "primary_absolute_direction_retained_count",
        "primary_absolute_bootstrap_replicated_count",
        "direct_injury_activation",
        "direct_injury_exceeds_late_program",
        "direct_late_maturation_support",
        "direct_late_maturation_depletion",
        "direct_early_program_suppression",
        "direct_positive_developmental_shift",
        "direct_negative_developmental_shift",
    ]

    direct_summary = integrated.loc[
        integrated[
            "direct_LINCS_quantitative_evaluated"
        ],
        direct_summary_columns,
    ].copy()

    direct_summary.to_csv(
        DIRECT_SUMMARY_OUTPUT,
        sep="\t",
        index=False,
    )

    missingness = integrated.loc[
        ~integrated[
            "direct_LINCS_quantitative_evaluated"
        ],
        [
            "chemical_name",
            "DTXSID",
            "evidence_tier",
            "maxT_parent_count",
            "parents_with_global_robust_mechanism",
            "dominant_robust_mechanism_class",
            "match_status",
            "direct_quantitative_missingness_reason",
            "direct_quantitative_missingness_treatment",
        ],
    ].copy()

    missingness.to_csv(
        MISSINGNESS_OUTPUT,
        sep="\t",
        index=False,
    )

    layer_dictionary = pd.DataFrame(
        [
            {
                "evidence_layer":
                    "CMAP directional concordance",

                "source_output":
                    str(
                        CMAP_TIER_FILE.relative_to(
                            PROJECT
                        )
                    ),

                "chemical_coverage":
                    29,

                "role_in_Phase6F":
                    (
                        "Defines perturbational direction and residual "
                        "mechanism tier."
                    ),

                "missing_data_treatment":
                    "Complete for the 29 Phase 6 chemicals.",
            },

            {
                "evidence_layer":
                    "Direct quantitative LINCS Level 5",

                "source_output":
                    str(
                        LINCS_INTERPRETATION_FILE.relative_to(
                            PROJECT
                        )
                    ),

                "chemical_coverage":
                    15,

                "role_in_Phase6F":
                    (
                        "Primary quantitative chemical-axis evidence "
                        "after gene-matched empirical-null calibration."
                    ),

                "missing_data_treatment":
                    (
                        "Fourteen unmatched chemicals remain NA and are "
                        "not assigned zero biological evidence."
                    ),
            },

            {
                "evidence_layer":
                    "Cross-scope quantitative replication",

                "source_output":
                    str(
                        CROSS_SCOPE_FILE.relative_to(
                            PROJECT
                        )
                    ),

                "chemical_coverage":
                    15,

                "role_in_Phase6F":
                    (
                        "Compares cell-balanced and all-signature "
                        "aggregation and records neural sensitivity."
                    ),

                "missing_data_treatment":
                    "Available only for direct LINCS-matched chemicals.",
            },

            {
                "evidence_layer":
                    "Neural-lineage LINCS sensitivity",

                "source_output":
                    str(
                        CROSS_SCOPE_FILE.relative_to(
                            PROJECT
                        )
                    ),

                "chemical_coverage":
                    EXPECTED_NEURAL_CHEMICALS,

                "role_in_Phase6F":
                    (
                        "Provides neural-context support without "
                        "replacing the all-cell primary analysis."
                    ),

                "missing_data_treatment":
                    (
                        "No neural profile is treated as unavailable "
                        "context, not negative evidence."
                    ),
            },

            {
                "evidence_layer":
                    "Official reconstructed gold-signature audit",

                "source_output":
                    str(
                        GOLD_DECISION_FILE.relative_to(
                            PROJECT
                        )
                    ),

                "chemical_coverage":
                    1,

                "role_in_Phase6F":
                    (
                        "Documents that a gold-restricted analysis was "
                        "not inferentially feasible."
                    ),

                "missing_data_treatment":
                    "Not used as a ranking score.",
            },

            {
                "evidence_layer":
                    "Replicate-concordant LINCS sensitivity",

                "source_output":
                    str(
                        REPLICATE_CONCORDANT_FILE.relative_to(
                            PROJECT
                        )
                    ),

                "chemical_coverage":
                    EXPECTED_REPLICATE_CONCORDANT_EVALUABLE,

                "role_in_Phase6F":
                    (
                        "Measures reproducibility-filtered directional "
                        "retention and hierarchical-bootstrap support."
                    ),

                "missing_data_treatment":
                    (
                        "Trichostatin A remains not evaluable rather "
                        "than receiving zero evidence."
                    ),
            },
        ]
    )

    layer_dictionary.to_csv(
        LAYER_DICTIONARY_OUTPUT,
        sep="\t",
        index=False,
    )

    cmap_global_count = int(
        integrated[
            "CMAP_global_residual_mechanism"
        ].sum()
    )

    neural_count = int(
        integrated[
            "neural_quantitative_data_available"
        ].sum()
    )

    replicate_evaluable_count = int(
        integrated[
            "replicate_concordant_profiles_available"
        ].sum()
    )

    final_matched_mask = (
        integrated[
            "direct_LINCS_quantitative_evaluated"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )

    final_replicate_available_mask = (
        integrated[
            "replicate_concordant_profiles_available"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )

    replicate_not_evaluable_count = int(
        (
            final_matched_mask
            & ~final_replicate_available_mask
        ).sum()
    )

    status = (
        "completed"
        if (
            len(
                integrated
            )
            == EXPECTED_CHEMICALS
            and int(
                integrated[
                    "direct_LINCS_quantitative_evaluated"
                ].sum()
            )
            == EXPECTED_DIRECT_MATCHED
            and len(
                missingness
            )
            == EXPECTED_DIRECT_UNMATCHED
            and cmap_global_count
            == EXPECTED_CMAP_GLOBAL_RESIDUAL
            and neural_count
            == EXPECTED_NEURAL_CHEMICALS
            and replicate_evaluable_count
            == EXPECTED_REPLICATE_CONCORDANT_EVALUABLE
            and replicate_not_evaluable_count
            == EXPECTED_REPLICATE_CONCORDANT_NOT_EVALUABLE
            and unmatched_quantitative_preserved_as_na
        )
        else "failed_validation"
    )

    completion = pd.DataFrame(
        [
            {
                "integrated_chemicals":
                    len(
                        integrated
                    ),

                "CMAP_chemicals":
                    cmap[
                        "chemical_normalized"
                    ].nunique(),

                "direct_LINCS_matched_chemicals":
                    int(
                        integrated[
                            "direct_LINCS_quantitative_evaluated"
                        ].sum()
                    ),

                "direct_LINCS_unmatched_chemicals":
                    int(
                        (
                            ~integrated[
                                "direct_LINCS_quantitative_evaluated"
                            ]
                        ).sum()
                    ),

                "CMAP_global_residual_chemicals":
                    cmap_global_count,

                "direct_neural_data_chemicals":
                    neural_count,

                "replicate_concordant_evaluable_chemicals":
                    replicate_evaluable_count,

                "replicate_concordant_not_evaluable_chemicals":
                    replicate_not_evaluable_count,

                "direct_injury_activation_chemicals":
                    int(
                        integrated[
                            "direct_injury_activation"
                        ].sum()
                    ),

                "direct_injury_dominant_chemicals":
                    int(
                        integrated[
                            "direct_injury_exceeds_late_program"
                        ].sum()
                    ),

                "direct_late_support_chemicals":
                    int(
                        integrated[
                            "direct_late_maturation_support"
                        ].sum()
                    ),

                "direct_late_depletion_chemicals":
                    int(
                        integrated[
                            "direct_late_maturation_depletion"
                        ].sum()
                    ),

                "gold_restricted_analysis_used":
                    False,

                "replicate_concordant_sensitivity_integrated":
                    True,

                "unmatched_quantitative_fields_preserved_as_NA":
                    unmatched_quantitative_preserved_as_na,

                "final_ranking_calculated":
                    False,

                "Phase6F1_status":
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
            "===== PHASE 6F1 COMPLETION =====",
            completion.to_string(
                index=False
            ),
            "",
            "===== EVIDENCE AVAILABILITY =====",
            availability_summary.to_string(
                index=False
            ),
            "",
            "===== DIRECT QUANTITATIVE EVIDENCE =====",
            direct_summary.to_string(
                index=False
            ),
            "",
            "===== CHEMICALS NOT QUANTITATIVELY EVALUATED =====",
            missingness.to_string(
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
            "Phase 6F1 integration validation failed."
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            (
                "Phase 6F1 failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
