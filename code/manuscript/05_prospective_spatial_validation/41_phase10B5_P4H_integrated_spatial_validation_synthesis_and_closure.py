from __future__ import annotations

import csv
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


project = Path(sys.argv[1])

temporal_path = Path(sys.argv[2])
localization_path = Path(sys.argv[3])
spatial_path = Path(sys.argv[4])
figure_manifest_path = Path(sys.argv[5])
main_selection_path = Path(sys.argv[6])
core_scales_path = Path(sys.argv[7])
expanded_scales_path = Path(sys.argv[8])
figure_legends_path = Path(sys.argv[9])
p4g3_root = Path(sys.argv[10])

out = Path(sys.argv[11])

evidence_dir = out / "01_integrated_evidence"
claim_dir = out / "02_claim_hierarchy"
spatial_dir = out / "03_spatial_heterogeneity"
results_dir = out / "04_manuscript_results"
limitations_dir = out / "05_manuscript_limitations"
figure_dir = out / "06_figure_inventory"
guardrail_dir = out / "07_reporting_guardrails"
audit_dir = out / "08_audit"


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
        writer.writerows(
            rows
        )


def true_value(
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


temporal_rows, temporal_columns = read_tsv(
    temporal_path
)

localization_rows, localization_columns = read_tsv(
    localization_path
)

spatial_rows, spatial_columns = read_tsv(
    spatial_path
)

figure_rows, figure_columns = read_tsv(
    figure_manifest_path
)

main_selection_rows, _ = read_tsv(
    main_selection_path
)

core_scale_rows, _ = read_tsv(
    core_scales_path
)

expanded_scale_rows, _ = read_tsv(
    expanded_scales_path
)

if len(temporal_rows) != 5:
    raise SystemExit(
        f"FAIL: temporal rows={len(temporal_rows)}; "
        "expected 5."
    )

if len(localization_rows) != 9:
    raise SystemExit(
        f"FAIL: localization rows="
        f"{len(localization_rows)}; expected 9."
    )

if len(spatial_rows) != 68:
    raise SystemExit(
        f"FAIL: spatial map summaries="
        f"{len(spatial_rows)}; expected 68."
    )

if len(figure_rows) != 5:
    raise SystemExit(
        f"FAIL: figure-manifest rows="
        f"{len(figure_rows)}; expected 5."
    )

if len(main_selection_rows) != 4:
    raise SystemExit(
        f"FAIL: main-section rows="
        f"{len(main_selection_rows)}; expected 4."
    )

if len(core_scale_rows) != 5:
    raise SystemExit(
        "FAIL: expected five core color-scale locks."
    )

if len(expanded_scale_rows) != 4:
    raise SystemExit(
        "FAIL: expected four expanded color-scale locks."
    )

core_modules = [
    "activity_dependent_plasticity",
    "astrocyte_maturation_metabolic_support",
    "neurogenesis_migration_layering",
    "patterning_arealization",
    "progenitor_radial_glia",
]

expanded_modules = [
    "axon_guidance_neurite_outgrowth",
    "oligodendrocyte_myelination",
    "synaptic_assembly_receptor_trafficking",
    "synaptic_membrane_structural_candidates",
]

temporal_lookup = {
    row["module"]: row
    for row in temporal_rows
}

localization_lookup = {
    row["module"]: row
    for row in localization_rows
}

if set(temporal_lookup) != set(core_modules):
    raise SystemExit(
        "FAIL: unexpected temporal module set."
    )

if set(localization_lookup) != set(
    core_modules + expanded_modules
):
    raise SystemExit(
        "FAIL: unexpected localization module set."
    )

# ---------------------------------------------------------
# Primary 300-panel spatial summaries.
# ---------------------------------------------------------

primary_300_core = [
    row
    for row in spatial_rows
    if (
        row[
            "effective_analysis_set"
        ] == "primary"
        and row[
            "panel_class"
        ] == "300_gene_panel"
        and row[
            "module"
        ] in core_modules
    )
]

if len(primary_300_core) != 30:
    raise SystemExit(
        f"FAIL: expected 30 primary 300-panel "
        f"core maps; observed "
        f"{len(primary_300_core)}."
    )

spatial_by_module: dict[
    str,
    list[dict[str, str]],
] = defaultdict(list)

for row in primary_300_core:

    spatial_by_module[
        row[
            "module"
        ]
    ].append(
        row
    )

spatial_summary_rows: list[
    dict[str, Any]
] = []

spatial_summary_lookup: dict[
    str,
    dict[str, Any],
] = {}

for module in core_modules:

    rows = spatial_by_module[
        module
    ]

    if len(rows) != 6:
        raise SystemExit(
            f"FAIL: {module} has {len(rows)} "
            "primary 300-panel spatial maps."
        )

    SDs = sorted(
        float(
            row[
                "weighted_spatial_bin_mean_SD"
            ]
        )
        for row in rows
    )

    occupied = [
        int(
            row[
                "occupied_bins"
            ]
        )
        for row in rows
    ]

    display = [
        int(
            row[
                "figure_display_bins"
            ]
        )
        for row in rows
    ]

    all_positive = all(
        value > 0.0
        for value in SDs
    )

    row = {
        "module": module,
        "primary_300_sections": 6,
        "spatial_maps": 6,
        "minimum_weighted_spatial_bin_mean_SD": (
            min(
                SDs
            )
        ),
        "median_weighted_spatial_bin_mean_SD": (
            (
                SDs[2]
                + SDs[3]
            )
            / 2.0
        ),
        "maximum_weighted_spatial_bin_mean_SD": (
            max(
                SDs
            )
        ),
        "minimum_occupied_bins": min(
            occupied
        ),
        "maximum_occupied_bins": max(
            occupied
        ),
        "minimum_figure_display_bins": min(
            display
        ),
        "maximum_figure_display_bins": max(
            display
        ),
        "positive_within_section_spatial_dispersion_all_sections": (
            all_positive
        ),
        "formal_spatial_inference": False,
        "interpretation": (
            "descriptive_within_section_spatial_"
            "heterogeneity_only"
        ),
    }

    spatial_summary_rows.append(
        row
    )

    spatial_summary_lookup[
        module
    ] = row

write_tsv(
    spatial_dir
    / "phase10B5_P4H_primary_300_core_spatial_heterogeneity_summary.tsv",
    spatial_summary_rows,
    [
        "module",
        "primary_300_sections",
        "spatial_maps",
        "minimum_weighted_spatial_bin_mean_SD",
        "median_weighted_spatial_bin_mean_SD",
        "maximum_weighted_spatial_bin_mean_SD",
        "minimum_occupied_bins",
        "maximum_occupied_bins",
        "minimum_figure_display_bins",
        "maximum_figure_display_bins",
        "positive_within_section_spatial_dispersion_all_sections",
        "formal_spatial_inference",
        "interpretation",
    ],
)

# ---------------------------------------------------------
# Integrated core-module evidence.
# ---------------------------------------------------------

integrated_rows: list[
    dict[str, Any]
] = []

for module in core_modules:

    temporal = temporal_lookup[
        module
    ]

    localization = localization_lookup[
        module
    ]

    spatial = spatial_summary_lookup[
        module
    ]

    localization_group_field = (
        "expected_H2_localization_group"
        if "expected_H2_localization_group"
        in localization
        else "expected_H2_broad_compartment"
    )

    integrated_rows.append(
        {
            "module": module,
            "prespecified_claim_class": (
                temporal[
                    "prespecified_claim_class"
                ]
            ),
            "primary_temporal_panel": (
                temporal[
                    "primary_panel"
                ]
            ),
            "primary_temporal_sections": (
                temporal[
                    "primary_independent_sections"
                ]
            ),
            "primary_spearman_rho": (
                temporal[
                    "primary_spearman_rho"
                ]
            ),
            "primary_exact_p": (
                temporal[
                    "primary_exact_p"
                ]
            ),
            "primary_BH_q": (
                temporal[
                    "primary_BH_q"
                ]
            ),
            "temporal_direction": (
                temporal[
                    "primary_direction"
                ]
            ),
            "temporal_directionally_stable": (
                temporal[
                    "directionally_stable"
                ]
            ),
            "temporal_evidence_tier": (
                temporal[
                    "evidence_tier"
                ]
            ),
            "statistically_confirmed_temporal_trajectory": (
                temporal[
                    "statistically_confirmed_temporal_trajectory"
                ]
            ),
            "H2_expected_localization_group": (
                localization.get(
                    localization_group_field,
                    "",
                )
            ),
            "H2_support_count": (
                localization[
                    "expected_H2_support_count"
                ]
            ),
            "H2_support_fraction": (
                localization[
                    "expected_H2_support_fraction"
                ]
            ),
            "H2_localization_strength": (
                localization[
                    "localization_strength"
                ]
            ),
            "statistical_localization_claim_authorized": (
                localization[
                    "statistical_localization_claim_authorized"
                ]
            ),
            "primary_300_spatial_maps": (
                spatial[
                    "spatial_maps"
                ]
            ),
            "median_spatial_bin_mean_SD": (
                spatial[
                    "median_weighted_spatial_bin_mean_SD"
                ]
            ),
            "spatial_dispersion_positive_all_primary_300_sections": (
                spatial[
                    "positive_within_section_spatial_dispersion_all_sections"
                ]
            ),
            "formal_spatial_inference": False,
            "integrated_interpretation": (
                "temporal_evidence_plus_descriptive_"
                "cell_class_and_spatial_localization"
            ),
        }
    )

write_tsv(
    evidence_dir
    / "phase10B5_P4H_integrated_core_module_evidence.tsv",
    integrated_rows,
    [
        "module",
        "prespecified_claim_class",
        "primary_temporal_panel",
        "primary_temporal_sections",
        "primary_spearman_rho",
        "primary_exact_p",
        "primary_BH_q",
        "temporal_direction",
        "temporal_directionally_stable",
        "temporal_evidence_tier",
        "statistically_confirmed_temporal_trajectory",
        "H2_expected_localization_group",
        "H2_support_count",
        "H2_support_fraction",
        "H2_localization_strength",
        "statistical_localization_claim_authorized",
        "primary_300_spatial_maps",
        "median_spatial_bin_mean_SD",
        "spatial_dispersion_positive_all_primary_300_sections",
        "formal_spatial_inference",
        "integrated_interpretation",
    ],
)

# ---------------------------------------------------------
# Expanded-panel evidence.
# ---------------------------------------------------------

expanded_spatial = [
    row
    for row in spatial_rows
    if row[
        "scoring_family"
    ] == "expanded_panel_only"
]

if len(expanded_spatial) != 8:
    raise SystemExit(
        f"FAIL: expected eight expanded spatial "
        f"maps; observed {len(expanded_spatial)}."
    )

expanded_rows: list[
    dict[str, Any]
] = []

for module in expanded_modules:

    localization = localization_lookup[
        module
    ]

    module_spatial = [
        row
        for row in expanded_spatial
        if row[
            "module"
        ] == module
    ]

    if len(module_spatial) != 2:
        raise SystemExit(
            f"FAIL: {module} does not have two "
            "expanded spatial maps."
        )

    SDs = [
        float(
            row[
                "weighted_spatial_bin_mean_SD"
            ]
        )
        for row in module_spatial
    ]

    localization_group_field = (
        "expected_H2_localization_group"
        if "expected_H2_localization_group"
        in localization
        else "expected_H2_broad_compartment"
    )

    expanded_rows.append(
        {
            "module": module,
            "independent_sections": 2,
            "formal_temporal_inference": False,
            "H2_expected_localization_group": (
                localization.get(
                    localization_group_field,
                    "",
                )
            ),
            "H2_localization_strength": (
                localization[
                    "localization_strength"
                ]
            ),
            "spatial_maps": 2,
            "minimum_spatial_bin_mean_SD": min(
                SDs
            ),
            "maximum_spatial_bin_mean_SD": max(
                SDs
            ),
            "formal_spatial_inference": False,
            "reporting_scope": (
                "descriptive_expanded_panel_"
                "two_section_validation_only"
            ),
        }
    )

write_tsv(
    evidence_dir
    / "phase10B5_P4H_expanded_panel_evidence.tsv",
    expanded_rows,
    [
        "module",
        "independent_sections",
        "formal_temporal_inference",
        "H2_expected_localization_group",
        "H2_localization_strength",
        "spatial_maps",
        "minimum_spatial_bin_mean_SD",
        "maximum_spatial_bin_mean_SD",
        "formal_spatial_inference",
        "reporting_scope",
    ],
)

# ---------------------------------------------------------
# Final claim hierarchy.
# ---------------------------------------------------------

claim_rows = [
    {
        "claim_id": "10B5_FINAL_C1",
        "evidence_level": "primary_statistical",
        "claim": (
            "No prespecified primary developmental "
            "module showed a statistically confirmed "
            "temporal trajectory in MERFISH after "
            "exact testing and multiplicity correction."
        ),
        "authorized_language": (
            "no_statistically_confirmed_temporal_"
            "trajectory_was_detected"
        ),
        "prohibited_language": (
            "developmental_modules_changed_"
            "significantly_with_age"
        ),
    },
    {
        "claim_id": "10B5_FINAL_C2",
        "evidence_level": "stable_descriptive_temporal",
        "claim": (
            "Astrocyte maturation/metabolic support "
            "showed the strongest stable positive "
            "temporal trend, whereas neurogenesis/"
            "migration, patterning/arealization and "
            "progenitor/radial-glia modules showed "
            "stable negative descriptive trends."
        ),
        "authorized_language": (
            "stable_descriptive_temporal_trends"
        ),
        "prohibited_language": (
            "significant_developmental_changes"
        ),
    },
    {
        "claim_id": "10B5_FINAL_C3",
        "evidence_level": "descriptive_cell_localization",
        "claim": (
            "Activity-dependent plasticity localized "
            "to neuronal or migrating-excitatory "
            "subclasses across all primary sections."
        ),
        "authorized_language": (
            "consistent_descriptive_excitatory_"
            "lineage_localization"
        ),
        "prohibited_language": (
            "statistically_specific_to_neurons"
        ),
    },
    {
        "claim_id": "10B5_FINAL_C4",
        "evidence_level": "descriptive_cell_localization",
        "claim": (
            "Astrocyte maturation/metabolic support "
            "and progenitor/radial-glia modules "
            "localized consistently to glial-lineage "
            "subclasses."
        ),
        "authorized_language": (
            "consistent_descriptive_glial_"
            "lineage_localization"
        ),
        "prohibited_language": (
            "statistically_glial_specific"
        ),
    },
    {
        "claim_id": "10B5_FINAL_C5",
        "evidence_level": "descriptive_cell_localization",
        "claim": (
            "Neurogenesis/migration and patterning/"
            "arealization localized predominantly to "
            "progenitor or migratory developmental "
            "compartments."
        ),
        "authorized_language": (
            "strong_descriptive_progenitor_"
            "or_migratory_localization"
        ),
        "prohibited_language": (
            "exclusive_progenitor_specificity"
        ),
    },
    {
        "claim_id": "10B5_FINAL_C6",
        "evidence_level": "descriptive_spatial",
        "claim": (
            "All five core modules exhibited "
            "within-section spatial heterogeneity "
            "across all six primary 300-gene sections."
        ),
        "authorized_language": (
            "descriptive_within_section_spatial_"
            "heterogeneity"
        ),
        "prohibited_language": (
            "significant_spatial_autocorrelation_"
            "or_anatomical_gradient"
        ),
    },
    {
        "claim_id": "10B5_FINAL_C7",
        "evidence_level": "technical_limitation",
        "claim": (
            "A systematic panel-class shift required "
            "300-gene sections to define the primary "
            "temporal analysis and separate display "
            "of the 960-gene sections."
        ),
        "authorized_language": (
            "panel_class_effect_was_controlled_by_"
            "analysis_separation"
        ),
        "prohibited_language": (
            "960_panel_shift_is_biological"
        ),
    },
    {
        "claim_id": "10B5_FINAL_C8",
        "evidence_level": "expanded_descriptive",
        "claim": (
            "The four expanded-panel modules provide "
            "two-section descriptive localization "
            "evidence only."
        ),
        "authorized_language": (
            "descriptive_two_section_validation"
        ),
        "prohibited_language": (
            "independent_temporal_replication"
        ),
    },
    {
        "claim_id": "10B5_FINAL_C9",
        "evidence_level": "spatial_limitation",
        "claim": (
            "Native processed spatial coordinates "
            "were visualized without anatomical axis "
            "interpretation or cross-section spatial "
            "registration."
        ),
        "authorized_language": (
            "native_section_specific_spatial_maps"
        ),
        "prohibited_language": (
            "registered_cortical_axis_or_"
            "cross_section_spatial_gradient"
        ),
    },
]

write_tsv(
    claim_dir
    / "phase10B5_P4H_final_claim_hierarchy.tsv",
    claim_rows,
    [
        "claim_id",
        "evidence_level",
        "claim",
        "authorized_language",
        "prohibited_language",
    ],
)

# ---------------------------------------------------------
# Manuscript-ready integrated Results.
# ---------------------------------------------------------

astro = temporal_lookup[
    "astrocyte_maturation_metabolic_support"
]

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

astro_q = optional_float(
    astro[
        "primary_BH_q"
    ]
)

astro_q_text = (
    f"{astro_q:.4f}"
    if astro_q is not None
    else "not applicable"
)

results_text = f"""External MERFISH spatial validation

External spatial validation was performed using processed MERFISH datasets spanning gestational weeks 15–34. After panel harmonization and section-level pseudobulk scoring, no prespecified primary module showed a statistically confirmed temporal association after exact permutation testing and multiple-testing correction.

The strongest temporal pattern was observed for the astrocyte maturation/metabolic-support module (Spearman rho = {astro_rho:.3f}, exact two-sided p = {astro_p:.4f}, BH q = {astro_q_text}). This positive trajectory was directionally stable across donor-omission and alternate-section analyses but remained below the prespecified corrected significance threshold. Neurogenesis/migration/layering, patterning/arealization and progenitor/radial-glia modules showed stable negative descriptive age trends. Activity-dependent plasticity remained a sensitivity-only temporal signal.

Within-section H1/H2 localization provided stronger cell-class resolution than the temporal analysis. Activity-dependent plasticity localized consistently to neuronal or migrating-excitatory subclasses across all eight independent primary sections. Astrocyte maturation/metabolic support and progenitor/radial-glia localized to glial-lineage subclasses across all primary sections, whereas neurogenesis/migration/layering and patterning/arealization localized predominantly to progenitor or migratory developmental compartments.

Spatial mapping further demonstrated descriptive within-section heterogeneity for all five core modules across all six primary 300-gene sections. Module scores were summarized in 64 x 64 native spatial grids using bins containing at least 20 cells. These maps were generated without smoothing, interpolation, cross-section spatial registration or spatial hypothesis testing, and therefore represent section-specific localization rather than statistically inferred spatial gradients.

The two primary 960-gene sections were analyzed and displayed separately because processed expression showed a systematic panel-class shift. Four expanded-panel modules were consequently restricted to descriptive two-section validation. Axon-guidance/neurite-outgrowth localized to progenitor-related compartments, both synaptic modules localized to neuronal subclasses, and oligodendrocyte/myelination showed a mixed H2 pattern across the two sections.
"""

(
    results_dir
    / "phase10B5_P4H_manuscript_ready_integrated_results.txt"
).write_text(
    results_text,
    encoding="utf-8",
)

limitations_text = """Phase 10B5 MERFISH validation limitations

1. Temporal inference was based on a small number of independent donor-sections. The primary panel-homogeneous analysis contained six 300-gene sections, so absence of corrected significance should not be interpreted as evidence for absence of developmental change.

2. The processed 300-gene and 960-gene MERFISH objects showed a systematic panel-class shift. The primary temporal analysis was therefore restricted to the homogeneous 300-gene panel, and combined-panel analyses remained sensitivity analyses.

3. Cell-level observations were not treated as independent biological replicates. H1 and H2 results are descriptive within-section localization summaries.

4. Spatial module maps were generated from native processed coordinates. Coordinate units and anatomical axis orientation were not inferred, and no cross-section spatial registration was performed.

5. No spatial smoothing, interpolation, Moran's I, neighborhood significance test or other spatial hypothesis test was used. Spatial results therefore describe within-section heterogeneity rather than statistically confirmed spatial organization.

6. Alternate FB080 and FB123 sections are within-donor sensitivity specimens rather than independent biological replicates.

7. Expanded-panel modules were available in only two independent 960-gene sections, one at GW18 and one at GW20. Their temporal and spatial interpretations remain descriptive.

8. H3 annotations were not used for primary or secondary localization because annotation completeness was insufficient. H1 and H2 remained the locked localization levels.

9. UMB5900 BA18 was used as the processed GW34 section because the originally planned BA17 section was unavailable in the processed H5AD object. The substitution remains explicitly documented in the validation audit trail.
"""

(
    limitations_dir
    / "phase10B5_P4H_manuscript_ready_limitations.txt"
).write_text(
    limitations_text,
    encoding="utf-8",
)

# ---------------------------------------------------------
# Figure inventory + hash verification.
# ---------------------------------------------------------

figure_inventory_rows: list[
    dict[str, Any]
] = []

figure_file_count = 0

for row in figure_rows:

    for file_type in (
        "PDF",
        "PNG",
    ):

        relative = row[
            file_type
        ]

        path = (
            p4g3_root
            / relative
        )

        if (
            not path.is_file()
            or path.stat().st_size == 0
        ):

            raise SystemExit(
                f"FAIL: missing figure file: {path}"
            )

        digest = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

        figure_inventory_rows.append(
            {
                "figure_id": row[
                    "figure_id"
                ],
                "role": row[
                    "role"
                ],
                "file_type": file_type,
                "panel_count": row[
                    "panels"
                ],
                "source_relative_path": (
                    path.relative_to(
                        project
                    ).as_posix()
                ),
                "size_bytes": (
                    path.stat().st_size
                ),
                "sha256": digest,
                "formal_inference": (
                    row[
                        "formal_inference"
                    ]
                ),
            }
        )

        figure_file_count += 1

if figure_file_count != 10:
    raise SystemExit(
        f"FAIL: figure files={figure_file_count}; "
        "expected 10."
    )

write_tsv(
    figure_dir
    / "phase10B5_P4H_canonical_figure_inventory.tsv",
    figure_inventory_rows,
    [
        "figure_id",
        "role",
        "file_type",
        "panel_count",
        "source_relative_path",
        "size_bytes",
        "sha256",
        "formal_inference",
    ],
)

# Copy textual figure legend into closure output.
figure_legend_text = figure_legends_path.read_text(
    encoding="utf-8",
)

(
    figure_dir
    / "phase10B5_P4H_canonical_spatial_figure_legends.txt"
).write_text(
    figure_legend_text,
    encoding="utf-8",
)

# ---------------------------------------------------------
# Final guardrails.
# ---------------------------------------------------------

guardrail_rows = [
    {
        "guardrail": "primary_temporal_unit",
        "locked_value": (
            "independent_donor_section"
        ),
    },
    {
        "guardrail": "primary_temporal_panel",
        "locked_value": (
            "300_gene_panel"
        ),
    },
    {
        "guardrail": "temporal_significance_claim",
        "locked_value": (
            "none_after_prespecified_correction"
        ),
    },
    {
        "guardrail": "H1_H2_localization",
        "locked_value": (
            "descriptive_within_section_only"
        ),
    },
    {
        "guardrail": "cell_level_replication",
        "locked_value": (
            "not_authorized"
        ),
    },
    {
        "guardrail": "spatial_mapping",
        "locked_value": (
            "64x64_native_section_specific_bins"
        ),
    },
    {
        "guardrail": "minimum_spatial_display_cells",
        "locked_value": "20",
    },
    {
        "guardrail": "spatial_smoothing",
        "locked_value": "none",
    },
    {
        "guardrail": "spatial_interpolation",
        "locked_value": "none",
    },
    {
        "guardrail": "cross_section_registration",
        "locked_value": "none",
    },
    {
        "guardrail": "coordinate_orientation",
        "locked_value": (
            "not_anatomically_interpreted"
        ),
    },
    {
        "guardrail": "spatial_hypothesis_testing",
        "locked_value": "none",
    },
    {
        "guardrail": "expanded_panel",
        "locked_value": (
            "descriptive_two_section_validation_only"
        ),
    },
    {
        "guardrail": "H3",
        "locked_value": "excluded",
    },
]

write_tsv(
    guardrail_dir
    / "phase10B5_P4H_final_reporting_guardrails.tsv",
    guardrail_rows,
    [
        "guardrail",
        "locked_value",
    ],
)

# ---------------------------------------------------------
# Closure audit.
# ---------------------------------------------------------

confirmed_temporal = sum(
    true_value(
        row[
            "statistically_confirmed_temporal_trajectory"
        ]
    )
    for row in temporal_rows
)

stable_primary_temporal = sum(
    (
        row[
            "prespecified_claim_class"
        ] == "broad_primary_multiage_multidonor"
        and true_value(
            row[
                "directionally_stable"
            ]
        )
    )
    for row in temporal_rows
)

core_complete_localization = sum(
    (
        row[
            "module"
        ] in core_modules
        and row[
            "localization_strength"
        ] == "complete"
    )
    for row in localization_rows
)

core_strong_localization = sum(
    (
        row[
            "module"
        ] in core_modules
        and row[
            "localization_strength"
        ] == "strong"
    )
    for row in localization_rows
)

expanded_complete = sum(
    (
        row[
            "module"
        ] in expanded_modules
        and row[
            "localization_strength"
        ] == "complete"
    )
    for row in localization_rows
)

expanded_mixed = sum(
    (
        row[
            "module"
        ] in expanded_modules
        and row[
            "localization_strength"
        ] == "mixed_two_section_pattern"
    )
    for row in localization_rows
)

spatial_positive_all = all(
    bool(
        row[
            "positive_within_section_spatial_dispersion_all_sections"
        ]
    )
    for row in spatial_summary_rows
)

technical_pass = (
    confirmed_temporal == 0
    and stable_primary_temporal == 4
    and core_complete_localization == 3
    and core_strong_localization == 2
    and expanded_complete == 3
    and expanded_mixed == 1
    and len(
        integrated_rows
    ) == 5
    and len(
        expanded_rows
    ) == 4
    and len(
        claim_rows
    ) == 9
    and len(
        spatial_summary_rows
    ) == 5
    and spatial_positive_all
    and figure_file_count == 10
)

status_value = (
    "passed_phase10B5_P4H_integrated_spatial_"
    "validation_synthesis_and_canonical_closure_"
    "ready_for_phase10_continuation"
    if technical_pass
    else (
        "phase10B5_P4H_requires_manual_review"
    )
)

status = {
    "phase": "phase10B5_P4H",
    "core_temporal_modules": 5,
    "statistically_confirmed_temporal_modules": (
        confirmed_temporal
    ),
    "stable_descriptive_primary_temporal_modules": (
        stable_primary_temporal
    ),
    "cell_localization_modules": len(
        localization_rows
    ),
    "core_complete_H2_localization_modules": (
        core_complete_localization
    ),
    "core_strong_H2_localization_modules": (
        core_strong_localization
    ),
    "expanded_complete_H2_localization_modules": (
        expanded_complete
    ),
    "expanded_mixed_H2_localization_modules": (
        expanded_mixed
    ),
    "primary_300_core_spatial_maps": len(
        primary_300_core
    ),
    "core_modules_with_positive_spatial_dispersion_in_all_primary_300_sections": sum(
        bool(
            row[
                "positive_within_section_spatial_dispersion_all_sections"
            ]
        )
        for row in spatial_summary_rows
    ),
    "all_core_modules_descriptively_spatially_heterogeneous": (
        spatial_positive_all
    ),
    "canonical_claim_rows": len(
        claim_rows
    ),
    "canonical_figure_manifest_rows": len(
        figure_rows
    ),
    "canonical_figure_files_verified": (
        figure_file_count
    ),
    "new_expression_values_accessed": False,
    "H5AD_files_opened": 0,
    "new_cell_level_scores_computed": False,
    "new_hypothesis_tests_performed": False,
    "cell_level_inference_performed": False,
    "spatial_hypothesis_tests_performed": False,
    "cross_section_spatial_registration_performed": (
        False
    ),
    "coordinate_orientation_interpreted": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_closed": True,
    "phase10B5_P4H_status": status_value,
}

write_tsv(
    out
    / "phase10B5_P4H_status.tsv",
    [status],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P4H INTEGRATED "
    "SPATIAL VALIDATION CLOSURE =====",
    "",
    "Core temporal modules: 5",
    (
        "Statistically confirmed temporal modules: "
        f"{confirmed_temporal}"
    ),
    (
        "Stable descriptive primary temporal modules: "
        f"{stable_primary_temporal}/4"
    ),
    (
        "Cell-localization modules: "
        f"{len(localization_rows)}"
    ),
    (
        "Core complete H2 localization modules: "
        f"{core_complete_localization}/5"
    ),
    (
        "Core strong H2 localization modules: "
        f"{core_strong_localization}/5"
    ),
    (
        "Expanded complete H2 modules: "
        f"{expanded_complete}/4"
    ),
    (
        "Expanded mixed H2 modules: "
        f"{expanded_mixed}/4"
    ),
    (
        "Primary 300-panel core spatial maps: "
        f"{len(primary_300_core)}/30"
    ),
    (
        "Core modules with positive spatial "
        f"dispersion in all primary 300 sections: "
        f"{sum(bool(row['positive_within_section_spatial_dispersion_all_sections']) for row in spatial_summary_rows)}/5"
    ),
    (
        "Canonical claims: "
        f"{len(claim_rows)}"
    ),
    (
        "Canonical figure files verified: "
        f"{figure_file_count}/10"
    ),
    "",
    "New expression values accessed: FALSE",
    "H5AD files opened: 0",
    "New cell-level scores computed: FALSE",
    "New hypothesis tests performed: FALSE",
    "Cell-level inference performed: FALSE",
    "Spatial hypothesis tests performed: FALSE",
    (
        "Cross-section spatial registration "
        "performed: FALSE"
    ),
    "Coordinate orientation interpreted: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "Phase 10B5 closed: TRUE",
    "",
    (
        "PHASE 10B5-P4H STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P4H_report.txt"
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
    "\n===== INTEGRATED CORE EVIDENCE ====="
)

for row in integrated_rows:

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
                "H2_expected_localization_group",
                "H2_support_count",
                "H2_support_fraction",
                "H2_localization_strength",
                "median_spatial_bin_mean_SD",
            )
        )
    )

# ---------------------------------------------------------
# Output SHA256.
# ---------------------------------------------------------

checksum_rows: list[
    dict[str, Any]
] = []

for path in sorted(
    out.rglob("*")
):

    if (
        path.is_file()
        and path.name
        != "phase10B5_P4H_SHA256.tsv"
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
    / "phase10B5_P4H_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)

if not technical_pass:

    raise SystemExit(
        "Phase 10B5-P4H requires manual review."
    )
