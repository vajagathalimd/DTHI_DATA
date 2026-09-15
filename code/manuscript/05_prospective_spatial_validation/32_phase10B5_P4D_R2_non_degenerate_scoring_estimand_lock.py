from __future__ import annotations

import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


project = Path(sys.argv[1])
status_path = Path(sys.argv[2])
module_sets_path = Path(sys.argv[3])
old_plan_path = Path(sys.argv[4])
module_gene_path = Path(sys.argv[5])
source_lock_path = Path(sys.argv[6])
out = Path(sys.argv[7])

audit_dir = (
    out
    / "01_estimand_audit"
)

core_dir = (
    out
    / "02_section_pseudobulk_core_plan"
)

expanded_dir = (
    out
    / "03_expanded_panel_descriptive_plan"
)

localization_dir = (
    out
    / "04_cell_annotation_localization_plan"
)

guardrail_dir = (
    out
    / "05_corrected_analysis_guardrails"
)

access_dir = (
    out
    / "06_expression_access_guard"
)

run_audit_dir = (
    out
    / "07_audit"
)

for directory in (
    audit_dir,
    core_dir,
    expanded_dir,
    localization_dir,
    guardrail_dir,
    access_dir,
    run_audit_dir,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def read_tsv(
    path: Path,
) -> tuple[
    list[dict[str, str]],
    list[str],
]:

    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        return (
            list(reader),
            reader.fieldnames or [],
        )


def write_tsv(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


status_rows, status_columns = read_tsv(
    status_path
)

module_rows, module_columns = read_tsv(
    module_sets_path
)

old_plan_rows, old_plan_columns = read_tsv(
    old_plan_path
)

module_gene_rows, module_gene_columns = read_tsv(
    module_gene_path
)

source_rows, source_columns = read_tsv(
    source_lock_path
)

if len(status_rows) != 1:

    raise SystemExit(
        "FAIL: P4D-R1 status must contain one row."
    )

status_source = status_rows[0]

if len(module_rows) != 9:

    raise SystemExit(
        f"FAIL: expected nine module gene sets; "
        f"observed {len(module_rows)}."
    )

if len(old_plan_rows) != 68:

    raise SystemExit(
        f"FAIL: expected 68 original scoring "
        f"definitions; observed "
        f"{len(old_plan_rows)}."
    )

if len(module_gene_rows) != 534:

    raise SystemExit(
        f"FAIL: expected 534 module-gene rows; "
        f"observed {len(module_gene_rows)}."
    )

if len(source_rows) != 6:

    raise SystemExit(
        f"FAIL: expected six expression-source "
        f"locks; observed {len(source_rows)}."
    )

if not all(
    row[
        "primary_expression_source"
    ] == "/X"
    for row in source_rows
):

    raise SystemExit(
        "FAIL: not all source locks use /X."
    )

core_modules = sorted(
    row[
        "module"
    ]
    for row in module_rows
    if row[
        "scoring_family"
    ] == "cross_panel_harmonized_core"
)

expanded_modules = sorted(
    row[
        "module"
    ]
    for row in module_rows
    if row[
        "scoring_family"
    ] == "expanded_panel_only"
)

if len(core_modules) != 5:

    raise SystemExit(
        f"FAIL: expected five cross-panel core "
        f"modules; observed {len(core_modules)}."
    )

if len(expanded_modules) != 4:

    raise SystemExit(
        f"FAIL: expected four expanded-panel "
        f"modules; observed {len(expanded_modules)}."
    )

core_old_rows = [
    row
    for row in old_plan_rows
    if row[
        "scoring_family"
    ] == "cross_panel_harmonized_core"
]

expanded_old_rows = [
    row
    for row in old_plan_rows
    if row[
        "scoring_family"
    ] == "expanded_panel_only"
]

if len(core_old_rows) != 60:

    raise SystemExit(
        f"FAIL: expected 60 cross-panel section-"
        f"module definitions; observed "
        f"{len(core_old_rows)}."
    )

if len(expanded_old_rows) != 8:

    raise SystemExit(
        f"FAIL: expected eight expanded-panel "
        f"definitions; observed "
        f"{len(expanded_old_rows)}."
    )

within_section_rows = [
    row
    for row in old_plan_rows
    if row[
        "standardization_scope"
    ] == "within_section_per_gene"
]

if len(within_section_rows) != 68:

    raise SystemExit(
        "FAIL: original 68-row scoring plan was "
        "not uniformly within-section standardized."
    )

old_estimand_rows = [
    {
        "original_scoring_family": row[
            "scoring_family"
        ],
        "original_module": row[
            "module"
        ],
        "original_section": row[
            "archive_name"
        ],
        "original_standardization_scope": row[
            "standardization_scope"
        ],
        "original_module_score_definition": row[
            "module_score_definition"
        ],
        "section_wide_mean_is_mathematically_zero": (
            True
        ),
        "valid_for_between_section_temporal_comparison": (
            False
        ),
        "valid_for_within_section_cell_localization": (
            True
        ),
        "mathematical_reason": (
            "each_gene_is_centered_to_zero_across_"
            "cells_within_the_same_section_so_the_"
            "section_wide_mean_of_the_module_score_"
            "is_zero"
        ),
    }
    for row in old_plan_rows
]

write_tsv(
    audit_dir
    / "phase10B5_P4D_R2_original_estimand_degeneracy_audit.tsv",
    old_estimand_rows,
    [
        "original_scoring_family",
        "original_module",
        "original_section",
        "original_standardization_scope",
        "original_module_score_definition",
        "section_wide_mean_is_mathematically_zero",
        "valid_for_between_section_temporal_comparison",
        "valid_for_within_section_cell_localization",
        "mathematical_reason",
    ],
)

proof_rows = [
    {
        "step": 1,
        "statement": (
            "For each section s and gene g define "
            "z_ig=(x_ig-mean_sg)/SD_sg."
        ),
    },
    {
        "step": 2,
        "statement": (
            "By construction the mean of z_ig over "
            "all selected cells i in section s is zero."
        ),
    },
    {
        "step": 3,
        "statement": (
            "A module score is the mean of its "
            "gene-specific z values for each cell."
        ),
    },
    {
        "step": 4,
        "statement": (
            "The section-wide module mean is therefore "
            "the mean of several zero-valued gene means "
            "and is also zero."
        ),
    },
    {
        "step": 5,
        "statement": (
            "Within-section z scoring is retained only "
            "for cell-class localization, not for "
            "between-section developmental comparison."
        ),
    },
]

write_tsv(
    audit_dir
    / "phase10B5_P4D_R2_mathematical_estimand_proof.tsv",
    proof_rows,
    [
        "step",
        "statement",
    ],
)

core_plan_rows: list[
    dict[str, Any]
] = []

for row in core_old_rows:

    analysis_set = row[
        "effective_analysis_set"
    ]

    if analysis_set == "primary":

        estimand_role = (
            "primary_reference_and_primary_result"
        )

        scaling_application = (
            "estimate_gene_center_and_scale_from_"
            "the_8_primary_sections_and_transform_"
            "this_primary_section"
        )

    elif analysis_set == "section_sensitivity":

        estimand_role = (
            "sensitivity_projection_only"
        )

        scaling_application = (
            "apply_gene_center_and_scale_estimated_"
            "from_the_8_primary_sections_without_"
            "refitting"
        )

    else:

        raise SystemExit(
            f"FAIL: unexpected analysis set: "
            f"{analysis_set}"
        )

    core_plan_rows.append(
        {
            "estimand_family": (
                "section_pseudobulk_cross_panel_core"
            ),
            "estimand_role": estimand_role,
            "effective_analysis_set": analysis_set,
            "archive_name": row[
                "archive_name"
            ],
            "donor_id": row[
                "donor_id"
            ],
            "gestational_week": row[
                "gestational_week"
            ],
            "panel_class": row[
                "panel_class"
            ],
            "processed_H5AD": row[
                "processed_H5AD"
            ],
            "locked_sample_value": row[
                "locked_sample_value"
            ],
            "locked_region_value": row[
                "locked_region_value"
            ],
            "selected_cell_count": row[
                "selected_cell_count"
            ],
            "module": row[
                "module"
            ],
            "prespecified_claim_class": row[
                "prespecified_claim_class"
            ],
            "gene_count": row[
                "gene_count"
            ],
            "gene_symbols": row[
                "gene_symbols"
            ],
            "expression_source": "/X",
            "cell_to_section_aggregation": (
                "arithmetic_mean_processed_expression_"
                "per_gene_across_selected_cells"
            ),
            "gene_standardization_reference": (
                "eight_independent_primary_sections_"
                "with_equal_section_weight"
            ),
            "gene_standardization_center": (
                "mean_of_primary_section_gene_means"
            ),
            "gene_standardization_scale": (
                "population_SD_across_8_primary_"
                "section_gene_means_ddof0"
            ),
            "scaling_application": (
                scaling_application
            ),
            "zero_variance_gene_rule": (
                "standardized_gene_value_zero_and_"
                "flag_gene"
            ),
            "module_score_definition": (
                "unweighted_mean_of_primary_reference_"
                "standardized_section_gene_means"
            ),
            "primary_statistical_unit": (
                "independent_donor_section"
            ),
            "cell_count_used_as_statistical_replication": (
                False
            ),
        }
    )

primary_core_rows = [
    row
    for row in core_plan_rows
    if row[
        "effective_analysis_set"
    ] == "primary"
]

sensitivity_core_rows = [
    row
    for row in core_plan_rows
    if row[
        "effective_analysis_set"
    ] == "section_sensitivity"
]

if len(primary_core_rows) != 40:

    raise SystemExit(
        f"FAIL: expected 40 primary core "
        f"section-module rows; observed "
        f"{len(primary_core_rows)}."
    )

if len(sensitivity_core_rows) != 20:

    raise SystemExit(
        f"FAIL: expected 20 sensitivity core "
        f"section-module rows; observed "
        f"{len(sensitivity_core_rows)}."
    )

write_tsv(
    core_dir
    / "phase10B5_P4D_R2_section_pseudobulk_core_plan.tsv",
    core_plan_rows,
    [
        "estimand_family",
        "estimand_role",
        "effective_analysis_set",
        "archive_name",
        "donor_id",
        "gestational_week",
        "panel_class",
        "processed_H5AD",
        "locked_sample_value",
        "locked_region_value",
        "selected_cell_count",
        "module",
        "prespecified_claim_class",
        "gene_count",
        "gene_symbols",
        "expression_source",
        "cell_to_section_aggregation",
        "gene_standardization_reference",
        "gene_standardization_center",
        "gene_standardization_scale",
        "scaling_application",
        "zero_variance_gene_rule",
        "module_score_definition",
        "primary_statistical_unit",
        "cell_count_used_as_statistical_replication",
    ],
)

expanded_plan_rows: list[
    dict[str, Any]
] = []

for row in expanded_old_rows:

    if row[
        "effective_analysis_set"
    ] != "primary":

        raise SystemExit(
            "FAIL: expanded-panel plan contains a "
            "non-primary section."
        )

    expanded_plan_rows.append(
        {
            "estimand_family": (
                "expanded_panel_two_section_descriptive"
            ),
            "effective_analysis_set": (
                "primary"
            ),
            "archive_name": row[
                "archive_name"
            ],
            "donor_id": row[
                "donor_id"
            ],
            "gestational_week": row[
                "gestational_week"
            ],
            "processed_H5AD": row[
                "processed_H5AD"
            ],
            "locked_sample_value": row[
                "locked_sample_value"
            ],
            "locked_region_value": row[
                "locked_region_value"
            ],
            "selected_cell_count": row[
                "selected_cell_count"
            ],
            "module": row[
                "module"
            ],
            "gene_count": row[
                "gene_count"
            ],
            "gene_symbols": row[
                "gene_symbols"
            ],
            "expression_source": "/X",
            "section_gene_summary": (
                "arithmetic_mean_processed_expression_"
                "per_gene_across_selected_cells"
            ),
            "comparison": (
                "GW20_minus_GW18_gene_level_difference"
            ),
            "module_summary": (
                "median_gene_difference_mean_gene_"
                "difference_and_direction_concordance"
            ),
            "formal_inferential_test_authorized": (
                False
            ),
            "reason_no_formal_inference": (
                "only_two_independent_sections_and_"
                "one_section_per_age"
            ),
            "claim_scope": (
                "descriptive_expanded_panel_two_age_"
                "validation_only"
            ),
        }
    )

if len(expanded_plan_rows) != 8:

    raise SystemExit(
        f"FAIL: expected eight expanded-panel "
        f"descriptive rows; observed "
        f"{len(expanded_plan_rows)}."
    )

expanded_ages = {
    int(
        row[
            "gestational_week"
        ]
    )
    for row in expanded_plan_rows
}

if expanded_ages != {
    18,
    20,
}:

    raise SystemExit(
        f"FAIL: expanded-panel ages are "
        f"{sorted(expanded_ages)}; expected 18 and 20."
    )

write_tsv(
    expanded_dir
    / "phase10B5_P4D_R2_expanded_panel_descriptive_plan.tsv",
    expanded_plan_rows,
    [
        "estimand_family",
        "effective_analysis_set",
        "archive_name",
        "donor_id",
        "gestational_week",
        "processed_H5AD",
        "locked_sample_value",
        "locked_region_value",
        "selected_cell_count",
        "module",
        "gene_count",
        "gene_symbols",
        "expression_source",
        "section_gene_summary",
        "comparison",
        "module_summary",
        "formal_inferential_test_authorized",
        "reason_no_formal_inference",
        "claim_scope",
    ],
)

localization_rows: list[
    dict[str, Any]
] = []

for row in old_plan_rows:

    localization_rows.append(
        {
            "estimand_family": (
                "within_section_cell_annotation_"
                "localization"
            ),
            "effective_analysis_set": row[
                "effective_analysis_set"
            ],
            "archive_name": row[
                "archive_name"
            ],
            "donor_id": row[
                "donor_id"
            ],
            "gestational_week": row[
                "gestational_week"
            ],
            "processed_H5AD": row[
                "processed_H5AD"
            ],
            "locked_sample_value": row[
                "locked_sample_value"
            ],
            "locked_region_value": row[
                "locked_region_value"
            ],
            "selected_cell_count": row[
                "selected_cell_count"
            ],
            "scoring_family": row[
                "scoring_family"
            ],
            "module": row[
                "module"
            ],
            "gene_count": row[
                "gene_count"
            ],
            "gene_symbols": row[
                "gene_symbols"
            ],
            "expression_source": "/X",
            "gene_standardization": (
                "within_section_per_gene_zscore_ddof0"
            ),
            "cell_module_score": (
                "unweighted_mean_of_within_section_"
                "standardized_genes"
            ),
            "primary_grouping": (
                "H1_annotation_within_section"
            ),
            "secondary_grouping": (
                "H2_annotation_within_section"
            ),
            "H3_grouping": (
                "not_authorized_for_primary_or_"
                "secondary_analysis"
            ),
            "section_wide_mean_interpretation": (
                "not_reportable_because_forced_to_zero"
            ),
            "between_section_temporal_claim_authorized": (
                False
            ),
            "valid_use": (
                "relative_cell_class_localization_"
                "within_each_section"
            ),
        }
    )

if len(localization_rows) != 68:

    raise SystemExit(
        f"FAIL: expected 68 localization rows; "
        f"observed {len(localization_rows)}."
    )

write_tsv(
    localization_dir
    / "phase10B5_P4D_R2_cell_annotation_localization_plan.tsv",
    localization_rows,
    [
        "estimand_family",
        "effective_analysis_set",
        "archive_name",
        "donor_id",
        "gestational_week",
        "processed_H5AD",
        "locked_sample_value",
        "locked_region_value",
        "selected_cell_count",
        "scoring_family",
        "module",
        "gene_count",
        "gene_symbols",
        "expression_source",
        "gene_standardization",
        "cell_module_score",
        "primary_grouping",
        "secondary_grouping",
        "H3_grouping",
        "section_wide_mean_interpretation",
        "between_section_temporal_claim_authorized",
        "valid_use",
    ],
)

guardrail_rows = [
    {
        "guardrail": (
            "expression_source"
        ),
        "locked_value": "/X",
        "reason": (
            "P4D_and_P4D_R1_consistently_identified_"
            "author_processed_continuous_expression"
        ),
    },
    {
        "guardrail": (
            "temporal_primary_unit"
        ),
        "locked_value": (
            "eight_independent_primary_donor_sections"
        ),
        "reason": (
            "prevents_cell_level_pseudoreplication"
        ),
    },
    {
        "guardrail": (
            "temporal_gene_input"
        ),
        "locked_value": (
            "section_pseudobulk_processed_expression_"
            "mean_per_gene"
        ),
        "reason": (
            "preserves_between_section_expression_"
            "differences"
        ),
    },
    {
        "guardrail": (
            "temporal_gene_standardization"
        ),
        "locked_value": (
            "across_eight_primary_sections_equal_"
            "section_weight_ddof0"
        ),
        "reason": (
            "creates_non_degenerate_section_level_"
            "module_scores"
        ),
    },
    {
        "guardrail": (
            "sensitivity_section_scaling"
        ),
        "locked_value": (
            "project_using_primary_section_center_"
            "and_scale_without_refitting"
        ),
        "reason": (
            "prevents_sensitivity_sections_from_"
            "altering_primary_reference"
        ),
    },
    {
        "guardrail": (
            "cell_localization_standardization"
        ),
        "locked_value": (
            "within_section_gene_zscore_ddof0"
        ),
        "reason": (
            "valid_for_relative_H1_and_H2_localization_"
            "only"
        ),
    },
    {
        "guardrail": (
            "within_section_section_mean"
        ),
        "locked_value": (
            "not_interpreted_or_tested"
        ),
        "reason": (
            "mathematically_forced_to_zero"
        ),
    },
    {
        "guardrail": (
            "expanded_panel_modules"
        ),
        "locked_value": (
            "descriptive_GW18_vs_GW20_gene_difference_"
            "and_direction_concordance"
        ),
        "reason": (
            "only_two_independent_sections_are_"
            "available"
        ),
    },
    {
        "guardrail": (
            "cell_count_weighting"
        ),
        "locked_value": (
            "no_cell_count_weighting_of_primary_"
            "section_level_statistics"
        ),
        "reason": (
            "each_independent_section_receives_equal_"
            "statistical_weight"
        ),
    },
    {
        "guardrail": (
            "H1_annotation"
        ),
        "locked_value": (
            "primary_cell_localization_grouping"
        ),
        "reason": (
            "complete_in_all_12_sections"
        ),
    },
    {
        "guardrail": (
            "H2_annotation"
        ),
        "locked_value": (
            "secondary_cell_localization_grouping"
        ),
        "reason": (
            "complete_but_more_granular_and_age_"
            "specific"
        ),
    },
    {
        "guardrail": (
            "H3_annotation"
        ),
        "locked_value": (
            "excluded_from_primary_and_secondary_"
            "localization"
        ),
        "reason": (
            "substantially_incomplete"
        ),
    },
]

write_tsv(
    guardrail_dir
    / "phase10B5_P4D_R2_corrected_analysis_guardrails.tsv",
    guardrail_rows,
    [
        "guardrail",
        "locked_value",
        "reason",
    ],
)

access_rows = [
    {
        "processed_H5AD": row[
            "processed_H5AD"
        ],
        "source_lock_preserved": True,
        "X_values_accessed": False,
        "raw_X_values_accessed": False,
        "complete_expression_matrix_loaded": False,
        "section_pseudobulk_computed": False,
        "cell_localization_scores_computed": False,
        "module_scores_computed": False,
    }
    for row in source_rows
]

write_tsv(
    access_dir
    / "phase10B5_P4D_R2_expression_access_guard.tsv",
    access_rows,
    [
        "processed_H5AD",
        "source_lock_preserved",
        "X_values_accessed",
        "raw_X_values_accessed",
        "complete_expression_matrix_loaded",
        "section_pseudobulk_computed",
        "cell_localization_scores_computed",
        "module_scores_computed",
    ],
)

selected_cells = int(
    status_source[
        "selected_cells_total"
    ]
)

primary_cells = int(
    status_source[
        "primary_selected_cells"
    ]
)

sensitivity_cells = int(
    status_source[
        "section_sensitivity_selected_cells"
    ]
)

technical_pass = (
    len(source_rows) == 6
    and len(core_plan_rows) == 60
    and len(primary_core_rows) == 40
    and len(sensitivity_core_rows) == 20
    and len(expanded_plan_rows) == 8
    and len(localization_rows) == 68
    and len(module_gene_rows) == 534
    and selected_cells == 6_062_942
    and primary_cells == 4_306_468
    and sensitivity_cells == 1_756_474
)

status_value = (
    "passed_phase10B5_P4D_R2_non_degenerate_"
    "section_pseudobulk_and_cell_localization_"
    "estimand_lock_ready_for_chunked_expression_"
    "aggregation"
    if technical_pass
    else (
        "phase10B5_P4D_R2_requires_manual_review"
    )
)

status = {
    "phase": "phase10B5_P4D_R2",
    "H5AD_source_locks_preserved": len(
        source_rows
    ),
    "selected_cells_total": selected_cells,
    "primary_selected_cells": primary_cells,
    "section_sensitivity_selected_cells": (
        sensitivity_cells
    ),
    "cross_panel_core_modules": len(
        core_modules
    ),
    "expanded_panel_only_modules": len(
        expanded_modules
    ),
    "primary_core_section_module_estimands": len(
        primary_core_rows
    ),
    "sensitivity_core_section_module_estimands": len(
        sensitivity_core_rows
    ),
    "total_core_section_module_estimands": len(
        core_plan_rows
    ),
    "expanded_panel_descriptive_estimands": len(
        expanded_plan_rows
    ),
    "cell_annotation_localization_estimands": len(
        localization_rows
    ),
    "original_within_section_temporal_estimand_degenerate": (
        True
    ),
    "P4D_R1_expression_source_lock_preserved": True,
    "P4D_R1_module_gene_sets_preserved": True,
    "temporal_primary_unit": (
        "independent_donor_section"
    ),
    "temporal_gene_input": (
        "section_pseudobulk_processed_X_mean"
    ),
    "temporal_standardization_reference": (
        "eight_primary_sections_equal_weight"
    ),
    "sensitivity_sections_projected_without_refit": (
        True
    ),
    "within_section_zscore_retained_for_cell_localization": (
        True
    ),
    "expanded_panel_formal_inference_authorized": (
        False
    ),
    "expression_values_accessed": False,
    "section_pseudobulk_computed": False,
    "module_scores_computed": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P4D_R2_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B5_P4D_R2_status.tsv",
    [status],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P4D-R2 NON-DEGENERATE "
    "SCORING ESTIMAND LOCK =====",
    "",
    (
        "H5AD source locks preserved: "
        f"{len(source_rows)}/6"
    ),
    (
        "Selected cells: "
        f"{selected_cells}"
    ),
    (
        "Primary core section-module estimands: "
        f"{len(primary_core_rows)}"
    ),
    (
        "Sensitivity core section-module estimands: "
        f"{len(sensitivity_core_rows)}"
    ),
    (
        "Expanded-panel descriptive estimands: "
        f"{len(expanded_plan_rows)}"
    ),
    (
        "Cell-annotation localization estimands: "
        f"{len(localization_rows)}"
    ),
    "",
    (
        "Original within-section temporal estimand "
        "degenerate: TRUE"
    ),
    "P4D-R1 expression-source lock preserved: TRUE",
    "P4D-R1 module gene sets preserved: TRUE",
    (
        "Temporal primary unit: "
        "independent donor/section"
    ),
    (
        "Temporal gene input: section-pseudobulk "
        "processed /X mean"
    ),
    (
        "Temporal standardization reference: "
        "eight primary sections with equal weight"
    ),
    (
        "Sensitivity sections projected without "
        "refitting: TRUE"
    ),
    (
        "Within-section z scores retained for "
        "cell localization: TRUE"
    ),
    (
        "Expanded-panel formal inference "
        "authorized: FALSE"
    ),
    "Expression values accessed: FALSE",
    "Section pseudobulk computed: FALSE",
    "Module scores computed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P4D-R2 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P4D_R2_report.txt"
).write_text(
    "\n".join(report)
    + "\n",
    encoding="utf-8",
)

print()
print(
    "\n".join(report)
)

print(
    "\n===== CORRECTED ESTIMAND COUNTS ====="
)

print(
    "primary_core_section_module_estimands"
    f"\t{len(primary_core_rows)}"
)

print(
    "sensitivity_core_section_module_estimands"
    f"\t{len(sensitivity_core_rows)}"
)

print(
    "expanded_panel_descriptive_estimands"
    f"\t{len(expanded_plan_rows)}"
)

print(
    "cell_annotation_localization_estimands"
    f"\t{len(localization_rows)}"
)

checksum_rows: list[
    dict[str, Any]
] = []

for path in sorted(
    out.rglob("*")
):

    if (
        path.is_file()
        and path.name
        != "phase10B5_P4D_R2_SHA256.tsv"
    ):

        checksum_rows.append(
            {
                "sha256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
                "size_bytes": (
                    path.stat().st_size
                ),
                "project_relative_path": (
                    path.relative_to(
                        project
                    ).as_posix()
                ),
            }
        )

write_tsv(
    out
    / "phase10B5_P4D_R2_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)

if not technical_pass:

    raise SystemExit(
        "Phase 10B5-P4D-R2 requires manual review."
    )
