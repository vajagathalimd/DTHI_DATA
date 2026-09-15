from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any


project = Path(sys.argv[1])
evidence_path = Path(sys.argv[2])
primary_path = Path(sys.argv[3])
lodo_path = Path(sys.argv[4])
section_path = Path(sys.argv[5])
adjusted_path = Path(sys.argv[6])
panel_path = Path(sys.argv[7])
score_path = Path(sys.argv[8])
expanded_path = Path(sys.argv[9])
out = Path(sys.argv[10])

canonical_dir = out / "01_canonical_temporal_evidence"
claim_dir = out / "02_claim_hierarchy"
results_dir = out / "03_manuscript_results"
figure_dir = out / "04_figure_data"
localization_dir = out / "05_cell_localization_scope"
guardrail_dir = out / "06_reporting_guardrails"
audit_dir = out / "07_audit"

for directory in (
    canonical_dir,
    claim_dir,
    results_dir,
    figure_dir,
    localization_dir,
    guardrail_dir,
    audit_dir,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def read_tsv(
    path: Path,
) -> tuple[list[dict[str, str]], list[str]]:

    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        return list(reader), reader.fieldnames or []


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


def as_bool(
    value: Any,
) -> bool:

    return str(value).strip().upper() in {
        "TRUE",
        "T",
        "1",
        "YES",
    }


def optional_float(
    value: Any,
) -> float | None:

    text = str(value).strip()

    if text == "":
        return None

    return float(text)


evidence_rows, _ = read_tsv(
    evidence_path
)

primary_rows, _ = read_tsv(
    primary_path
)

lodo_rows, _ = read_tsv(
    lodo_path
)

section_rows, _ = read_tsv(
    section_path
)

adjusted_rows, _ = read_tsv(
    adjusted_path
)

panel_rows, _ = read_tsv(
    panel_path
)

score_rows, _ = read_tsv(
    score_path
)

expanded_rows, _ = read_tsv(
    expanded_path
)

for label, rows, expected in (
    ("evidence", evidence_rows, 5),
    ("primary", primary_rows, 5),
    ("LODO", lodo_rows, 5),
    ("section sensitivity", section_rows, 5),
    ("panel adjusted", adjusted_rows, 5),
    ("panel audit", panel_rows, 5),
    ("primary scores", score_rows, 50),
    ("expanded", expanded_rows, 4),
):

    if len(rows) != expected:

        raise SystemExit(
            f"FAIL: {label} contains {len(rows)} "
            f"rows; expected {expected}."
        )

primary_lookup = {
    row["module"]: row
    for row in primary_rows
}

lodo_lookup = {
    row["module"]: row
    for row in lodo_rows
}

section_lookup = {
    row["module"]: row
    for row in section_rows
}

adjusted_lookup = {
    row["module"]: row
    for row in adjusted_rows
}

panel_lookup = {
    row["module"]: row
    for row in panel_rows
}

modules = sorted(
    primary_lookup
)

if set(modules) != set(
    lodo_lookup
) or set(modules) != set(
    section_lookup
) or set(modules) != set(
    adjusted_lookup
) or set(modules) != set(
    panel_lookup
):

    raise SystemExit(
        "FAIL: module sets differ across inputs."
    )

canonical_rows: list[dict[str, Any]] = []

stable_primary_modules = 0
unstable_sensitivity_modules = 0
corrected_hits = 0
nominal_primary_hits = 0

for module in modules:

    primary = primary_lookup[module]
    lodo = lodo_lookup[module]
    section = section_lookup[module]
    adjusted = adjusted_lookup[module]
    panel = panel_lookup[module]

    claim_class = primary[
        "prespecified_claim_class"
    ]

    primary_p = float(
        primary[
            "exact_two_sided_spearman_p"
        ]
    )

    primary_q = optional_float(
        primary[
            "BH_q_primary_four_module_family"
        ]
    )

    adjusted_p = float(
        adjusted[
            "exact_two_sided_age_p"
        ]
    )

    adjusted_q = float(
        adjusted[
            "BH_q_five_module_sensitivity_family"
        ]
    )

    lodo_fraction = float(
        lodo[
            "direction_concordance_fraction"
        ]
    )

    section_fraction = float(
        section[
            "direction_concordance_fraction"
        ]
    )

    directionally_stable = (
        lodo_fraction >= 5 / 6
        and section_fraction >= 6 / 8
    )

    primary_family = (
        claim_class
        == "broad_primary_multiage_multidonor"
    )

    primary_nominal = (
        primary_family
        and primary_p <= 0.05
    )

    primary_corrected = (
        primary_family
        and primary_q is not None
        and primary_q <= 0.05
    )

    adjusted_nominal = (
        adjusted_p <= 0.05
    )

    adjusted_corrected = (
        adjusted_q <= 0.05
    )

    if primary_corrected:
        corrected_hits += 1

    if primary_nominal:
        nominal_primary_hits += 1

    if primary_family:

        if primary_corrected and directionally_stable:

            evidence_tier = (
                "T1_corrected_temporal_support"
            )

        elif primary_nominal and directionally_stable:

            evidence_tier = (
                "T2_nominal_exact_stable"
            )

        elif directionally_stable:

            evidence_tier = (
                "T3_directionally_stable_descriptive"
            )

            stable_primary_modules += 1

        else:

            evidence_tier = (
                "T4_descriptive_or_unstable"
            )

    else:

        if directionally_stable:

            evidence_tier = (
                "S1_prespecified_sensitivity_stable"
            )

        else:

            evidence_tier = (
                "S2_prespecified_sensitivity_unstable"
            )

            unstable_sensitivity_modules += 1

    canonical_rows.append(
        {
            "module": module,
            "prespecified_claim_class": claim_class,
            "primary_panel": "300_gene_panel",
            "primary_independent_sections": 6,
            "primary_spearman_rho": float(
                primary[
                    "spearman_rho"
                ]
            ),
            "primary_exact_p": primary_p,
            "primary_BH_q": (
                primary_q
                if primary_q is not None
                else ""
            ),
            "primary_direction": primary[
                "direction"
            ],
            "primary_nominal_support": (
                primary_nominal
            ),
            "primary_corrected_support": (
                primary_corrected
            ),
            "LODO_direction_concordance": (
                lodo_fraction
            ),
            "section_selection_direction_concordance": (
                section_fraction
            ),
            "directionally_stable": (
                directionally_stable
            ),
            "all8_panel_adjusted_age_beta": float(
                adjusted[
                    "age_beta_per_GW"
                ]
            ),
            "all8_panel_adjusted_exact_p": (
                adjusted_p
            ),
            "all8_panel_adjusted_BH_q": (
                adjusted_q
            ),
            "all8_nominal_support": (
                adjusted_nominal
            ),
            "all8_corrected_support": (
                adjusted_corrected
            ),
            "panel_960_beta": float(
                adjusted[
                    "panel_960_beta"
                ]
            ),
            "panel_signature_flag": as_bool(
                panel[
                    "panel_signature_flag"
                ]
            ),
            "evidence_tier": evidence_tier,
            "statistically_confirmed_temporal_trajectory": (
                primary_corrected
            ),
            "reporting_scope": (
                "descriptive_temporal_trend"
                if not primary_corrected
                else "corrected_temporal_association"
            ),
        }
    )

write_tsv(
    canonical_dir
    / "phase10B5_P4E3_canonical_temporal_module_evidence.tsv",
    canonical_rows,
    [
        "module",
        "prespecified_claim_class",
        "primary_panel",
        "primary_independent_sections",
        "primary_spearman_rho",
        "primary_exact_p",
        "primary_BH_q",
        "primary_direction",
        "primary_nominal_support",
        "primary_corrected_support",
        "LODO_direction_concordance",
        "section_selection_direction_concordance",
        "directionally_stable",
        "all8_panel_adjusted_age_beta",
        "all8_panel_adjusted_exact_p",
        "all8_panel_adjusted_BH_q",
        "all8_nominal_support",
        "all8_corrected_support",
        "panel_960_beta",
        "panel_signature_flag",
        "evidence_tier",
        "statistically_confirmed_temporal_trajectory",
        "reporting_scope",
    ],
)

primary_modules = [
    row
    for row in canonical_rows
    if row[
        "prespecified_claim_class"
    ] == "broad_primary_multiage_multidonor"
]

sensitivity_modules = [
    row
    for row in canonical_rows
    if row[
        "prespecified_claim_class"
    ] == "broad_sensitivity_multiage_multidonor"
]

if len(primary_modules) != 4:
    raise SystemExit(
        "FAIL: expected four primary modules."
    )

if len(sensitivity_modules) != 1:
    raise SystemExit(
        "FAIL: expected one sensitivity module."
    )

strongest_primary = max(
    primary_modules,
    key=lambda row: abs(
        float(
            row[
                "primary_spearman_rho"
            ]
        )
    ),
)

astro = next(
    row
    for row in canonical_rows
    if row[
        "module"
    ]
    == "astrocyte_maturation_metabolic_support"
)

claim_rows = [
    {
        "claim_id": "TEMPORAL_C1",
        "claim_level": "primary_conclusion",
        "claim": (
            "No prespecified primary module showed "
            "a statistically confirmed temporal "
            "trajectory after exact testing and "
            "multiple-testing correction."
        ),
        "support": (
            "0_primary_nominal_exact_hits_and_"
            "0_primary_BH_FDR_hits"
        ),
        "authorized_language": (
            "no_statistically_confirmed_temporal_"
            "association_was_detected"
        ),
        "prohibited_language": (
            "modules_changed_significantly_with_age"
        ),
    },
    {
        "claim_id": "TEMPORAL_C2",
        "claim_level": "strongest_descriptive_trend",
        "claim": (
            "Astrocyte maturation and metabolic "
            "support showed the strongest increasing "
            "and directionally stable temporal trend."
        ),
        "support": (
            f"rho={astro['primary_spearman_rho']};"
            f"exact_p={astro['primary_exact_p']};"
            f"BH_q={astro['primary_BH_q']};"
            f"LODO={astro['LODO_direction_concordance']};"
            f"section_sensitivity="
            f"{astro['section_selection_direction_concordance']}"
        ),
        "authorized_language": (
            "strongest_stable_descriptive_trend"
        ),
        "prohibited_language": (
            "significant_astrocyte_maturation_increase"
        ),
    },
    {
        "claim_id": "TEMPORAL_C3",
        "claim_level": "stable_negative_trends",
        "claim": (
            "Neurogenesis/migration, patterning/"
            "arealization and progenitor/radial-glia "
            "modules showed directionally stable "
            "negative trends without corrected support."
        ),
        "support": (
            "stable_across_LODO_and_section_selection_"
            "but_primary_exact_p_gt_0.05"
        ),
        "authorized_language": (
            "directionally_stable_descriptive_decline"
        ),
        "prohibited_language": (
            "significant_developmental_decline"
        ),
    },
    {
        "claim_id": "TEMPORAL_C4",
        "claim_level": "sensitivity_only",
        "claim": (
            "Activity-dependent plasticity was "
            "prespecified as sensitivity-only and "
            "was not directionally robust."
        ),
        "support": (
            "LODO_concordance_4_of_6_and_"
            "section_selection_concordance_4_of_8"
        ),
        "authorized_language": (
            "unstable_sensitivity_pattern"
        ),
        "prohibited_language": (
            "validated_activity_dependent_trajectory"
        ),
    },
    {
        "claim_id": "TEMPORAL_C5",
        "claim_level": "technical_limitation",
        "claim": (
            "Processed expression showed a systematic "
            "panel-class shift across all five modules."
        ),
        "support": (
            "5_of_5_panel_signature_flags_and_both_"
            "960_panel_sections_ranked_bottom_two"
        ),
        "authorized_language": (
            "panel_homogeneous_primary_analysis_"
            "was_required"
        ),
        "prohibited_language": (
            "960_panel_negative_scores_are_biological"
        ),
    },
    {
        "claim_id": "TEMPORAL_C6",
        "claim_level": "expanded_panel",
        "claim": (
            "All four expanded-panel modules were "
            "descriptively higher at GW20 than GW18."
        ),
        "support": (
            "direction_concordance_0.818_to_1.000_"
            "with_one_independent_section_per_age"
        ),
        "authorized_language": (
            "descriptive_two_section_comparison"
        ),
        "prohibited_language": (
            "statistically_significant_GW20_increase"
        ),
    },
]

write_tsv(
    claim_dir
    / "phase10B5_P4E3_temporal_claim_hierarchy.tsv",
    claim_rows,
    [
        "claim_id",
        "claim_level",
        "claim",
        "support",
        "authorized_language",
        "prohibited_language",
    ],
)

tier_counts: dict[str, int] = {}

for row in canonical_rows:

    tier = row[
        "evidence_tier"
    ]

    tier_counts[
        tier
    ] = (
        tier_counts.get(
            tier,
            0,
        )
        + 1
    )

tier_rows = [
    {
        "evidence_tier": tier,
        "module_count": count,
        "modules": ";".join(
            sorted(
                row[
                    "module"
                ]
                for row in canonical_rows
                if row[
                    "evidence_tier"
                ] == tier
            )
        ),
    }
    for tier, count in sorted(
        tier_counts.items()
    )
]

write_tsv(
    claim_dir
    / "phase10B5_P4E3_temporal_evidence_tier_counts.tsv",
    tier_rows,
    [
        "evidence_tier",
        "module_count",
        "modules",
    ],
)

figure_rows = sorted(
    canonical_rows,
    key=lambda row: (
        float(
            row[
                "primary_spearman_rho"
            ]
        ),
        row[
            "module"
        ],
    ),
)

write_tsv(
    figure_dir
    / "phase10B5_P4E3_temporal_module_figure_data.tsv",
    figure_rows,
    [
        "module",
        "prespecified_claim_class",
        "primary_spearman_rho",
        "primary_exact_p",
        "primary_BH_q",
        "LODO_direction_concordance",
        "section_selection_direction_concordance",
        "all8_panel_adjusted_age_beta",
        "all8_panel_adjusted_exact_p",
        "all8_panel_adjusted_BH_q",
        "panel_960_beta",
        "evidence_tier",
    ],
)

write_tsv(
    figure_dir
    / "phase10B5_P4E3_primary_section_score_figure_data.tsv",
    score_rows,
    [
        "effective_analysis_set",
        "archive_name",
        "donor_id",
        "gestational_week",
        "panel_class",
        "processed_H5AD",
        "selected_cells",
        "module",
        "prespecified_claim_class",
        "gene_count",
        "module_score",
    ],
)

write_tsv(
    figure_dir
    / "phase10B5_P4E3_expanded_panel_figure_data.tsv",
    expanded_rows,
    list(
        expanded_rows[0].keys()
    ),
)

astro_rho = float(
    astro[
        "primary_spearman_rho"
    ]
)

astro_p = float(
    astro[
        "primary_exact_p"
    ]
)

astro_q = float(
    astro[
        "primary_BH_q"
    ]
)

astro_adjusted_beta = float(
    astro[
        "all8_panel_adjusted_age_beta"
    ]
)

astro_adjusted_p = float(
    astro[
        "all8_panel_adjusted_exact_p"
    ]
)

astro_adjusted_q = float(
    astro[
        "all8_panel_adjusted_BH_q"
    ]
)

results_text = f"""Phase 10B5 MERFISH temporal validation

Temporal associations were evaluated using six independent donor-sections assayed with the homogeneous 300-gene panel, spanning gestational weeks 15, 20, 22 and 34. Exact age-label permutation tests identified no nominally significant association among the four prespecified primary modules, and no association survived Benjamini-Hochberg correction.

The strongest temporal pattern involved the astrocyte maturation and metabolic-support module, which increased with gestational age (Spearman rho = {astro_rho:.3f}, exact two-sided p = {astro_p:.4f}, BH q = {astro_q:.4f}). Its direction was retained across all six leave-one-section-out analyses and all eight valid one-section-per-donor configurations. Nevertheless, this pattern did not meet the prespecified corrected significance threshold and is interpreted as a stable descriptive trend.

Neurogenesis/migration/layering, patterning/arealization and progenitor/radial-glia modules showed negative age directions that were generally stable across donor omission and alternate-section analyses, but their exact primary tests were not significant. Activity-dependent plasticity, which was prespecified as a sensitivity module, showed limited directional robustness.

In the all-eight-section panel-adjusted sensitivity model, the astrocyte maturation and metabolic-support module retained a positive age coefficient (beta = {astro_adjusted_beta:.4f} per gestational week; exact p = {astro_adjusted_p:.4f}; BH q = {astro_adjusted_q:.4f}), providing nominal but not multiplicity-corrected sensitivity evidence.

All five core modules exhibited a systematic panel-class shift, with the two 960-gene sections receiving the lowest scores. The homogeneous 300-gene analysis was therefore retained as the primary temporal analysis, while the combined panel-adjusted model was treated as sensitivity evidence.

The four expanded-panel modules were descriptively higher at gestational week 20 than week 18. These comparisons were not subjected to formal inference because only one independent section was available at each age.
"""

(
    results_dir
    / "phase10B5_P4E3_manuscript_ready_temporal_results.txt"
).write_text(
    results_text,
    encoding="utf-8",
)

localization_rows = [
    {
        "analysis_component": (
            "core_module_H1_localization"
        ),
        "modules": 5,
        "eligible_sections": 12,
        "annotation_column": (
            "H1_annotation"
        ),
        "role": (
            "primary_cell_class_localization"
        ),
        "standardization": (
            "within_section_per_gene_zscore_ddof0"
        ),
        "aggregation": (
            "section_by_H1_module_mean_median_SD_"
            "and_cell_count"
        ),
        "formal_cell_level_inference": False,
        "between_section_pooling": False,
    },
    {
        "analysis_component": (
            "core_module_H2_localization"
        ),
        "modules": 5,
        "eligible_sections": 12,
        "annotation_column": (
            "H2_annotation"
        ),
        "role": (
            "secondary_cell_subclass_localization"
        ),
        "standardization": (
            "within_section_per_gene_zscore_ddof0"
        ),
        "aggregation": (
            "section_by_H2_module_mean_median_SD_"
            "and_cell_count"
        ),
        "formal_cell_level_inference": False,
        "between_section_pooling": False,
    },
    {
        "analysis_component": (
            "expanded_module_localization"
        ),
        "modules": 4,
        "eligible_sections": 2,
        "annotation_column": (
            "H1_annotation_then_H2_annotation"
        ),
        "role": (
            "descriptive_expanded_panel_localization"
        ),
        "standardization": (
            "within_section_per_gene_zscore_ddof0"
        ),
        "aggregation": (
            "section_annotation_module_summary"
        ),
        "formal_cell_level_inference": False,
        "between_section_pooling": False,
    },
]

write_tsv(
    localization_dir
    / "phase10B5_P4E3_cell_localization_scope_lock.tsv",
    localization_rows,
    [
        "analysis_component",
        "modules",
        "eligible_sections",
        "annotation_column",
        "role",
        "standardization",
        "aggregation",
        "formal_cell_level_inference",
        "between_section_pooling",
    ],
)

guardrail_rows = [
    {
        "guardrail": (
            "statistical_conclusion"
        ),
        "locked_value": (
            "no_corrected_temporal_module_association"
        ),
    },
    {
        "guardrail": (
            "strongest_temporal_pattern"
        ),
        "locked_value": (
            "astrocyte_maturation_metabolic_support_"
            "stable_descriptive_increase"
        ),
    },
    {
        "guardrail": (
            "primary_panel"
        ),
        "locked_value": (
            "300_gene_panel"
        ),
    },
    {
        "guardrail": (
            "combined_panel_analysis"
        ),
        "locked_value": (
            "panel_adjusted_sensitivity_only"
        ),
    },
    {
        "guardrail": (
            "expanded_panel_modules"
        ),
        "locked_value": (
            "descriptive_only_no_formal_inference"
        ),
    },
    {
        "guardrail": (
            "cell_level_replication"
        ),
        "locked_value": (
            "not_authorized"
        ),
    },
    {
        "guardrail": (
            "H1_localization"
        ),
        "locked_value": (
            "primary_descriptive_cell_class_axis"
        ),
    },
    {
        "guardrail": (
            "H2_localization"
        ),
        "locked_value": (
            "secondary_descriptive_subclass_axis"
        ),
    },
    {
        "guardrail": (
            "H3_localization"
        ),
        "locked_value": (
            "excluded_from_primary_and_secondary_"
            "analyses"
        ),
    },
]

write_tsv(
    guardrail_dir
    / "phase10B5_P4E3_reporting_guardrails.tsv",
    guardrail_rows,
    [
        "guardrail",
        "locked_value",
    ],
)

expanded_all_higher = all(
    row[
        "dominant_direction"
    ] == "predominantly_higher_at_GW20"
    for row in expanded_rows
)

technical_pass = (
    len(canonical_rows) == 5
    and len(primary_modules) == 4
    and len(sensitivity_modules) == 1
    and corrected_hits == 0
    and nominal_primary_hits == 0
    and stable_primary_modules == 4
    and unstable_sensitivity_modules == 1
    and all(
        row[
            "panel_signature_flag"
        ]
        for row in canonical_rows
    )
    and expanded_all_higher
    and strongest_primary[
        "module"
    ]
    == "astrocyte_maturation_metabolic_support"
)

status_value = (
    "passed_phase10B5_P4E3_conservative_temporal_"
    "result_synthesis_and_claim_lock_ready_for_"
    "H1_H2_cell_localization_aggregation"
    if technical_pass
    else (
        "phase10B5_P4E3_requires_manual_review"
    )
)

status = {
    "phase": "phase10B5_P4E3",
    "canonical_temporal_module_rows": len(
        canonical_rows
    ),
    "prespecified_primary_modules": len(
        primary_modules
    ),
    "prespecified_sensitivity_modules": len(
        sensitivity_modules
    ),
    "primary_nominal_exact_hits": (
        nominal_primary_hits
    ),
    "primary_corrected_hits": (
        corrected_hits
    ),
    "all8_panel_adjusted_corrected_hits": sum(
        bool(
            row[
                "all8_corrected_support"
            ]
        )
        for row in canonical_rows
    ),
    "stable_descriptive_primary_modules": (
        stable_primary_modules
    ),
    "unstable_sensitivity_modules": (
        unstable_sensitivity_modules
    ),
    "panel_signature_flagged_modules": sum(
        bool(
            row[
                "panel_signature_flag"
            ]
        )
        for row in canonical_rows
    ),
    "strongest_descriptive_module": (
        strongest_primary[
            "module"
        ]
    ),
    "expanded_descriptive_modules": len(
        expanded_rows
    ),
    "expanded_modules_predominantly_higher_at_GW20": sum(
        row[
            "dominant_direction"
        ] == "predominantly_higher_at_GW20"
        for row in expanded_rows
    ),
    "statistically_confirmed_temporal_modules": (
        corrected_hits
    ),
    "manuscript_claims_locked": True,
    "H1_localization_authorized": True,
    "H2_localization_authorized": True,
    "H3_localization_authorized": False,
    "expression_values_accessed_in_this_phase": (
        False
    ),
    "H5AD_files_opened_in_this_phase": False,
    "new_hypothesis_tests_performed": False,
    "cell_level_inference_performed": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P4E3_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B5_P4E3_status.tsv",
    [status],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P4E3 TEMPORAL RESULT "
    "SYNTHESIS AND CLAIM LOCK =====",
    "",
    (
        "Canonical temporal modules: "
        f"{len(canonical_rows)}"
    ),
    (
        "Primary nominal exact hits: "
        f"{nominal_primary_hits}"
    ),
    (
        "Primary corrected hits: "
        f"{corrected_hits}"
    ),
    (
        "Stable descriptive primary modules: "
        f"{stable_primary_modules}/4"
    ),
    (
        "Unstable sensitivity modules: "
        f"{unstable_sensitivity_modules}/1"
    ),
    (
        "Panel-signature flagged modules: "
        f"{sum(row['panel_signature_flag'] for row in canonical_rows)}/5"
    ),
    (
        "Strongest descriptive module: "
        f"{strongest_primary['module']}"
    ),
    (
        "Expanded modules higher at GW20: "
        f"{sum(row['dominant_direction'] == 'predominantly_higher_at_GW20' for row in expanded_rows)}/4"
    ),
    "",
    "Statistically confirmed temporal modules: 0",
    "Manuscript claims locked: TRUE",
    "H1 localization authorized: TRUE",
    "H2 localization authorized: TRUE",
    "H3 localization authorized: FALSE",
    "Expression values accessed in this phase: FALSE",
    "H5AD files opened in this phase: FALSE",
    "New hypothesis tests performed: FALSE",
    "Cell-level inference performed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P4E3 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P4E3_report.txt"
).write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

print()
print(
    "\n".join(report)
)

print(
    "\n===== CANONICAL TEMPORAL EVIDENCE ====="
)

for row in canonical_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "module",
                "primary_spearman_rho",
                "primary_exact_p",
                "primary_BH_q",
                "LODO_direction_concordance",
                "section_selection_direction_concordance",
                "all8_panel_adjusted_exact_p",
                "all8_panel_adjusted_BH_q",
                "evidence_tier",
            )
        )
    )

if not technical_pass:

    raise SystemExit(
        "Phase 10B5-P4E3 requires manual review."
    )
