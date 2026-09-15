from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


project = Path(sys.argv[1])

b4_module_path = Path(sys.argv[2])
b5_core_path = Path(sys.argv[3])
b5_expanded_path = Path(sys.argv[4])
b5_spatial_path = Path(sys.argv[5])
b5_claims_path = Path(sys.argv[6])
b5_figures_path = Path(sys.argv[7])

out = Path(sys.argv[8])

synthesis_dir = (
    out
    / "01_cross_platform_module_synthesis"
)

compatibility_dir = (
    out
    / "02_cellular_context_compatibility"
)

claim_dir = (
    out
    / "03_final_claim_hierarchy"
)

results_dir = (
    out
    / "04_manuscript_results"
)

limitations_dir = (
    out
    / "05_manuscript_limitations"
)

provenance_dir = (
    out
    / "06_provenance"
)

guardrail_dir = (
    out
    / "07_reporting_guardrails"
)

audit_dir = (
    out
    / "08_audit"
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
        writer.writerows(
            rows
        )


def parse_bool(
    value: Any,
) -> bool:

    return str(value).strip().upper() in {
        "TRUE",
        "T",
        "YES",
        "1",
    }


def identity_family(
    identity: str,
) -> str:

    text = identity.strip()
    upper = text.upper()

    if (
        upper == "VRG"
        or upper == "RG"
        or upper.startswith("RG-")
        or upper.startswith("VRG-")
        or upper == "IPC"
        or upper.startswith("IPC-")
        or upper.startswith(
            "DIVIDING PROGENITOR"
        )
        or upper == "OPC"
    ):

        return (
            "radial_glial_or_progenitor"
        )

    if upper.startswith(
        "EN-MIG"
    ):

        return (
            "migrating_excitatory"
        )

    if upper.startswith(
        "EN-"
    ):

        return "neuronal"

    if upper.startswith(
        "IN-"
    ):

        return (
            "inhibitory_neuronal"
        )

    if "MICROGLIA" in upper:

        return "microglia"

    if (
        upper == "EC"
        or upper.startswith("EC-")
    ):

        return "endothelial"

    if "ASTRO" in upper:

        return "astrocyte"

    return "other_or_unresolved"


def compatibility(
    b4_family: str,
    b5_group: str,
) -> tuple[str, bool | None]:

    if b5_group == "mixed":

        return (
            "mixed_MERFISH_context_"
            "not_classified_as_concordant_"
            "or_discordant",
            None,
        )

    accepted = {
        "glial_lineage": {
            "radial_glial_or_progenitor",
            "astrocyte",
        },
        "neural_progenitor_or_migratory": {
            "radial_glial_or_progenitor",
            "migrating_excitatory",
        },
        "neuronal": {
            "neuronal",
            "inhibitory_neuronal",
        },
        "neuronal_or_migrating_excitatory": {
            "neuronal",
            "inhibitory_neuronal",
            "migrating_excitatory",
        },
    }

    accepted_set = accepted.get(
        b5_group,
        set(),
    )

    matched = (
        b4_family
        in accepted_set
    )

    if matched:

        return (
            "descriptively_compatible_"
            "cellular_context",
            True,
        )

    return (
        "different_descriptive_"
        "cellular_context",
        False,
    )


b4_rows, _ = read_tsv(
    b4_module_path
)

b5_core_rows, _ = read_tsv(
    b5_core_path
)

b5_expanded_rows, _ = read_tsv(
    b5_expanded_path
)

b5_spatial_rows, _ = read_tsv(
    b5_spatial_path
)

b5_claim_rows, _ = read_tsv(
    b5_claims_path
)

b5_figure_rows, _ = read_tsv(
    b5_figures_path
)

if len(b4_rows) != 9:
    raise SystemExit(
        f"FAIL: Phase10B4 module rows="
        f"{len(b4_rows)}; expected 9."
    )

if len(b5_core_rows) != 5:
    raise SystemExit(
        f"FAIL: Phase10B5 core rows="
        f"{len(b5_core_rows)}; expected 5."
    )

if len(b5_expanded_rows) != 4:
    raise SystemExit(
        f"FAIL: Phase10B5 expanded rows="
        f"{len(b5_expanded_rows)}; expected 4."
    )

if len(b5_spatial_rows) != 5:
    raise SystemExit(
        f"FAIL: Phase10B5 spatial rows="
        f"{len(b5_spatial_rows)}; expected 5."
    )

if len(b5_claim_rows) != 9:
    raise SystemExit(
        f"FAIL: Phase10B5 claim rows="
        f"{len(b5_claim_rows)}; expected 9."
    )

if len(b5_figure_rows) != 10:
    raise SystemExit(
        f"FAIL: Phase10B5 figure inventory rows="
        f"{len(b5_figure_rows)}; expected 10."
    )

b4_lookup = {
    row[
        "module"
    ]: row
    for row in b4_rows
}

b5_core_lookup = {
    row[
        "module"
    ]: row
    for row in b5_core_rows
}

b5_expanded_lookup = {
    row[
        "module"
    ]: row
    for row in b5_expanded_rows
}

b5_spatial_lookup = {
    row[
        "module"
    ]: row
    for row in b5_spatial_rows
}

core_modules = set(
    b5_core_lookup
)

expanded_modules = set(
    b5_expanded_lookup
)

all_modules = (
    core_modules
    | expanded_modules
)

if set(
    b4_lookup
) != all_modules:

    raise SystemExit(
        "FAIL: Phase10B4 and Phase10B5 "
        "module sets differ."
    )

if core_modules & expanded_modules:

    raise SystemExit(
        "FAIL: core and expanded module "
        "sets overlap."
    )

if len(
    all_modules
) != 9:

    raise SystemExit(
        "FAIL: expected nine unique modules."
    )

# ---------------------------------------------------------
# Cross-platform synthesis.
# ---------------------------------------------------------

module_rows: list[
    dict[str, Any]
] = []

compatibility_rows: list[
    dict[str, Any]
] = []

compatible_count = 0
different_count = 0
mixed_count = 0

all_b4_geometry_supported = True
all_b5_spatial_supported = True

for module in sorted(
    all_modules
):

    b4 = b4_lookup[
        module
    ]

    robust_count = int(
        b4[
            "geometry_robust_signal_count"
        ]
    )

    b4_geometry_supported = (
        robust_count > 0
    )

    all_b4_geometry_supported &= (
        b4_geometry_supported
    )

    top_identity = b4[
        "highest_scoring_snRNAseq_identity"
    ]

    top_family = identity_family(
        top_identity
    )

    if module in core_modules:

        b5 = b5_core_lookup[
            module
        ]

        b5_group = b5[
            "H2_expected_localization_group"
        ]

        b5_strength = b5[
            "H2_localization_strength"
        ]

        temporal_scope = (
            "formal_exact_temporal_test_"
            "performed_in_phase10B5"
        )

        temporal_rho = b5[
            "primary_spearman_rho"
        ]

        temporal_p = b5[
            "primary_exact_p"
        ]

        temporal_q = b5[
            "primary_BH_q"
        ]

        temporal_tier = b5[
            "temporal_evidence_tier"
        ]

        temporal_confirmed = parse_bool(
            b5[
                "statistically_confirmed_temporal_trajectory"
            ]
        )

        b5_spatial = (
            b5_spatial_lookup[
                module
            ]
        )

        minimum_spatial_SD = float(
            b5_spatial[
                "minimum_weighted_spatial_bin_mean_SD"
            ]
        )

        maximum_spatial_SD = float(
            b5_spatial[
                "maximum_weighted_spatial_bin_mean_SD"
            ]
        )

        b5_spatial_sections = int(
            b5_spatial[
                "spatial_maps"
            ]
        )

        b5_spatial_scope = (
            "six_primary_300_panel_sections"
        )

    else:

        b5 = b5_expanded_lookup[
            module
        ]

        b5_group = b5[
            "H2_expected_localization_group"
        ]

        b5_strength = b5[
            "H2_localization_strength"
        ]

        temporal_scope = (
            "descriptive_two_section_only"
        )

        temporal_rho = ""
        temporal_p = ""
        temporal_q = ""
        temporal_tier = (
            "expanded_descriptive_only"
        )

        temporal_confirmed = False

        minimum_spatial_SD = float(
            b5[
                "minimum_spatial_bin_mean_SD"
            ]
        )

        maximum_spatial_SD = float(
            b5[
                "maximum_spatial_bin_mean_SD"
            ]
        )

        b5_spatial_sections = int(
            b5[
                "spatial_maps"
            ]
        )

        b5_spatial_scope = (
            "two_primary_960_panel_sections"
        )

    b5_spatial_supported = (
        minimum_spatial_SD > 0.0
        and maximum_spatial_SD > 0.0
    )

    all_b5_spatial_supported &= (
        b5_spatial_supported
    )

    compatibility_class, matched = (
        compatibility(
            top_family,
            b5_group,
        )
    )

    if matched is True:

        compatible_count += 1

    elif matched is False:

        different_count += 1

    else:

        mixed_count += 1

    if matched is True:

        if module in core_modules:

            synthesis_class = (
                "core_descriptive_cross_platform_"
                "convergence"
            )

        else:

            synthesis_class = (
                "expanded_descriptive_cross_platform_"
                "convergence"
            )

    elif matched is False:

        synthesis_class = (
            "cross_platform_spatial_context_with_"
            "cellular_context_difference"
        )

    else:

        synthesis_class = (
            "cross_platform_spatial_context_with_"
            "mixed_MERFISH_cellular_localization"
        )

    module_rows.append(
        {
            "module": module,
            "phase10B4_locked_genes": int(
                b4[
                    "locked_genes"
                ]
            ),
            "phase10B4_geometry_robust_signal_count": (
                robust_count
            ),
            "phase10B4_geometry_robust_depths": (
                b4[
                    "geometry_robust_depths"
                ]
            ),
            "phase10B4_geometry_robust_directions": (
                b4[
                    "geometry_robust_directions"
                ]
            ),
            "phase10B4_module_spatial_priority": (
                b4[
                    "module_spatial_priority"
                ]
            ),
            "phase10B4_highest_snRNAseq_identity": (
                top_identity
            ),
            "phase10B4_highest_identity_family": (
                top_family
            ),
            "phase10B5_temporal_scope": (
                temporal_scope
            ),
            "phase10B5_temporal_rho": (
                temporal_rho
            ),
            "phase10B5_temporal_exact_p": (
                temporal_p
            ),
            "phase10B5_temporal_BH_q": (
                temporal_q
            ),
            "phase10B5_temporal_evidence_tier": (
                temporal_tier
            ),
            "phase10B5_temporal_confirmed": (
                temporal_confirmed
            ),
            "phase10B5_H2_localization_group": (
                b5_group
            ),
            "phase10B5_H2_localization_strength": (
                b5_strength
            ),
            "cellular_context_compatibility": (
                compatibility_class
            ),
            "phase10B5_spatial_sections": (
                b5_spatial_sections
            ),
            "phase10B5_spatial_scope": (
                b5_spatial_scope
            ),
            "phase10B5_minimum_spatial_bin_mean_SD": (
                minimum_spatial_SD
            ),
            "phase10B5_maximum_spatial_bin_mean_SD": (
                maximum_spatial_SD
            ),
            "phase10B4_geometry_spatial_evidence": (
                b4_geometry_supported
            ),
            "phase10B5_descriptive_spatial_heterogeneity": (
                b5_spatial_supported
            ),
            "cross_platform_synthesis_class": (
                synthesis_class
            ),
            "direct_replication_claim_authorized": (
                False
            ),
            "population_level_inference_authorized": (
                False
            ),
        }
    )

    compatibility_rows.append(
        {
            "module": module,
            "phase10B4_highest_snRNAseq_identity": (
                top_identity
            ),
            "phase10B4_identity_family": (
                top_family
            ),
            "phase10B5_H2_localization_group": (
                b5_group
            ),
            "phase10B5_H2_localization_strength": (
                b5_strength
            ),
            "compatibility_class": (
                compatibility_class
            ),
            "compatible_boolean": (
                ""
                if matched is None
                else matched
            ),
            "comparison_role": (
                "descriptive_cellular_context_only"
            ),
            "gene_set_equivalence_assumed": (
                False
            ),
            "direct_replication_claim": (
                False
            ),
        }
    )

if compatible_count != 7:
    raise SystemExit(
        f"FAIL: expected 7 descriptively "
        f"compatible module contexts; observed "
        f"{compatible_count}."
    )

if different_count != 1:
    raise SystemExit(
        f"FAIL: expected 1 different cellular "
        f"context; observed {different_count}."
    )

if mixed_count != 1:
    raise SystemExit(
        f"FAIL: expected 1 mixed/unresolved "
        f"context; observed {mixed_count}."
    )

write_tsv(
    synthesis_dir
    / "phase10B6_cross_platform_module_synthesis.tsv",
    module_rows,
    [
        "module",
        "phase10B4_locked_genes",
        "phase10B4_geometry_robust_signal_count",
        "phase10B4_geometry_robust_depths",
        "phase10B4_geometry_robust_directions",
        "phase10B4_module_spatial_priority",
        "phase10B4_highest_snRNAseq_identity",
        "phase10B4_highest_identity_family",
        "phase10B5_temporal_scope",
        "phase10B5_temporal_rho",
        "phase10B5_temporal_exact_p",
        "phase10B5_temporal_BH_q",
        "phase10B5_temporal_evidence_tier",
        "phase10B5_temporal_confirmed",
        "phase10B5_H2_localization_group",
        "phase10B5_H2_localization_strength",
        "cellular_context_compatibility",
        "phase10B5_spatial_sections",
        "phase10B5_spatial_scope",
        "phase10B5_minimum_spatial_bin_mean_SD",
        "phase10B5_maximum_spatial_bin_mean_SD",
        "phase10B4_geometry_spatial_evidence",
        "phase10B5_descriptive_spatial_heterogeneity",
        "cross_platform_synthesis_class",
        "direct_replication_claim_authorized",
        "population_level_inference_authorized",
    ],
)

write_tsv(
    compatibility_dir
    / "phase10B6_cellular_context_compatibility.tsv",
    compatibility_rows,
    [
        "module",
        "phase10B4_highest_snRNAseq_identity",
        "phase10B4_identity_family",
        "phase10B5_H2_localization_group",
        "phase10B5_H2_localization_strength",
        "compatibility_class",
        "compatible_boolean",
        "comparison_role",
        "gene_set_equivalence_assumed",
        "direct_replication_claim",
    ],
)

# ---------------------------------------------------------
# Final cross-platform claim hierarchy.
# ---------------------------------------------------------

claim_rows = [
    {
        "claim_id": "10B6_C1",
        "evidence_level": (
            "cross_platform_spatial_context"
        ),
        "claim": (
            "All nine locked developmental modules "
            "showed geometry-robust section-specific "
            "Visium evidence in Phase 10B4 and "
            "non-zero within-section MERFISH spatial "
            "heterogeneity in their available Phase "
            "10B5 sections."
        ),
        "authorized_language": (
            "cross_platform_descriptive_spatial_"
            "convergence"
        ),
        "prohibited_language": (
            "independent_spatial_replication"
        ),
    },
    {
        "claim_id": "10B6_C2",
        "evidence_level": (
            "cellular_context_convergence"
        ),
        "claim": (
            "Seven of nine modules showed "
            "descriptively compatible cellular "
            "contexts between the GW20 snRNA-seq "
            "identity summary and MERFISH H2 "
            "localization."
        ),
        "authorized_language": (
            "compatible_cross_platform_cellular_"
            "context"
        ),
        "prohibited_language": (
            "replicated_cell_type_specificity"
        ),
    },
    {
        "claim_id": "10B6_C3",
        "evidence_level": (
            "cellular_context_difference"
        ),
        "claim": (
            "Activity-dependent plasticity showed "
            "different cellular contexts across "
            "platforms: Microglia was the highest "
            "GW20 snRNA-seq identity, whereas "
            "MERFISH localized the panel-limited "
            "module to neuronal or migrating-"
            "excitatory subclasses."
        ),
        "authorized_language": (
            "cross_platform_cellular_context_"
            "difference"
        ),
        "prohibited_language": (
            "biological_contradiction"
        ),
    },
    {
        "claim_id": "10B6_C4",
        "evidence_level": (
            "mixed_expanded_context"
        ),
        "claim": (
            "Oligodendrocyte/myelination retained "
            "spatial evidence across both branches "
            "but MERFISH H2 localization was mixed "
            "across the two expanded-panel sections."
        ),
        "authorized_language": (
            "mixed_descriptive_cross_platform_"
            "context"
        ),
        "prohibited_language": (
            "validated_OPC_specificity"
        ),
    },
    {
        "claim_id": "10B6_C5",
        "evidence_level": (
            "temporal_constraint"
        ),
        "claim": (
            "MERFISH did not provide a statistically "
            "confirmed temporal trajectory for any "
            "prespecified primary module after "
            "correction."
        ),
        "authorized_language": (
            "no_corrected_temporal_trajectory_"
            "detected"
        ),
        "prohibited_language": (
            "developmental_change_proven_absent"
        ),
    },
    {
        "claim_id": "10B6_C6",
        "evidence_level": (
            "platform_limitation"
        ),
        "claim": (
            "Visium, snRNA-seq and MERFISH differed "
            "in donor structure, assay design, "
            "available genes and inferential unit, "
            "so cross-platform agreement is "
            "contextual convergence rather than "
            "direct replication."
        ),
        "authorized_language": (
            "contextual_cross_platform_convergence"
        ),
        "prohibited_language": (
            "independent_replication"
        ),
    },
    {
        "claim_id": "10B6_C7",
        "evidence_level": (
            "spatial_direction_constraint"
        ),
        "claim": (
            "The V2-higher Visium effects were not "
            "directionally compared with MERFISH "
            "maps because MERFISH coordinates were "
            "not anatomically registered or assigned "
            "cross-section cortical axes."
        ),
        "authorized_language": (
            "section_specific_spatial_context"
        ),
        "prohibited_language": (
            "cross_platform_spatial_direction_"
            "concordance"
        ),
    },
]

write_tsv(
    claim_dir
    / "phase10B6_final_cross_platform_claim_hierarchy.tsv",
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
# Manuscript-ready synthesis.
# ---------------------------------------------------------

results_text = """Cross-platform fetal spatial validation

The GW20 snRNA-seq/Visium analysis and the independent multi-age MERFISH analysis provided complementary rather than directly replicative validation of the locked developmental modules. In the GW20 Visium section, all nine modules showed at least one geometry-robust section-specific spatial signal, while MERFISH demonstrated non-zero within-section spatial heterogeneity for all nine modules across the available primary sections.

Cellular-context comparison showed descriptive compatibility for seven of nine modules. Progenitor/radial-glia and astrocyte maturation/metabolic-support modules were associated with radial-glial or progenitor identities in the GW20 snRNA-seq data and glial-lineage subclasses in MERFISH. Neurogenesis/migration and axon-guidance modules aligned with migratory or progenitor-related identities, while the two synaptic modules aligned with neuronal identities across both modalities.

Activity-dependent plasticity showed a cross-platform cellular-context difference: Microglia was the highest-scoring GW20 snRNA-seq identity, whereas the MERFISH panel-limited module localized most strongly to neuronal or migrating-excitatory subclasses. This difference is retained as a platform/context sensitivity rather than interpreted as a biological contradiction because the assays and effective measured gene sets differ.

The oligodendrocyte/myelination module also remained heterogeneous across modalities. It retained geometry-robust Visium evidence, but MERFISH H2 localization differed between the two expanded-panel sections and therefore remained mixed.

MERFISH temporal analysis did not identify a multiplicity-corrected developmental trajectory for any prespecified primary module. Consequently, the integrated Phase 10B evidence supports reproducible spatial and cellular-context structure more strongly than a simple monotonic gestational-age effect.
"""

(
    results_dir
    / "phase10B6_manuscript_ready_cross_platform_results.txt"
).write_text(
    results_text,
    encoding="utf-8",
)

limitations_text = """Phase 10B cross-platform validation limitations

1. Phase 10B4 Visium spatial inference arose from a single fetal spatial section and therefore represents section-specific evidence rather than population-level replication.

2. The snRNA-seq regional contrast was donor-confounded and was retained only as descriptive cellular context.

3. MERFISH contained multiple independent donor-sections, but its 300-gene and 960-gene processed panels showed a systematic panel-class shift. Primary temporal inference therefore used the homogeneous 300-gene panel.

4. The effective measured gene sets differed across snRNA-seq, Visium and MERFISH. Agreement between highest-scoring cellular identities and MERFISH H2 localization is therefore contextual compatibility rather than gene-set-identical replication.

5. Activity-dependent plasticity showed different highest cellular contexts between the GW20 snRNA-seq summary and MERFISH. This difference should not be described as a mechanistic contradiction without matched-platform validation.

6. MERFISH coordinates were not anatomically registered across specimens. Visium V1/V2 directionality therefore cannot be directly compared with MERFISH x/y spatial patterns.

7. No new cross-platform statistical test was performed in Phase 10B6. The synthesis integrates already-locked evidence hierarchies.

8. Cell-level observations were never treated as independent biological replicates.

9. Expanded MERFISH modules were available in only two 960-gene sections and remain descriptive.
"""

(
    limitations_dir
    / "phase10B6_manuscript_ready_cross_platform_limitations.txt"
).write_text(
    limitations_text,
    encoding="utf-8",
)

# ---------------------------------------------------------
# Provenance.
# ---------------------------------------------------------

source_paths = [
    (
        "Phase10B4_A5_module_synthesis",
        b4_module_path,
    ),
    (
        "Phase10B5_P4H_core_evidence",
        b5_core_path,
    ),
    (
        "Phase10B5_P4H_expanded_evidence",
        b5_expanded_path,
    ),
    (
        "Phase10B5_P4H_spatial_summary",
        b5_spatial_path,
    ),
    (
        "Phase10B5_P4H_claim_hierarchy",
        b5_claims_path,
    ),
    (
        "Phase10B5_P4H_figure_inventory",
        b5_figures_path,
    ),
]

provenance_rows = []

for source_name, path in source_paths:

    provenance_rows.append(
        {
            "source_name": source_name,
            "project_relative_path": (
                path.relative_to(
                    project
                ).as_posix()
            ),
            "size_bytes": (
                path.stat().st_size
            ),
            "sha256": hashlib.sha256(
                path.read_bytes()
            ).hexdigest(),
            "source_used_for_new_inference": (
                False
            ),
        }
    )

write_tsv(
    provenance_dir
    / "phase10B6_source_provenance.tsv",
    provenance_rows,
    [
        "source_name",
        "project_relative_path",
        "size_bytes",
        "sha256",
        "source_used_for_new_inference",
    ],
)

# ---------------------------------------------------------
# Final Phase 10B guardrails.
# ---------------------------------------------------------

guardrail_rows = [
    {
        "guardrail": (
            "cross_platform_interpretation"
        ),
        "locked_value": (
            "contextual_convergence_not_replication"
        ),
    },
    {
        "guardrail": (
            "Phase10B4_Visium_scope"
        ),
        "locked_value": (
            "section_specific_not_population_level"
        ),
    },
    {
        "guardrail": (
            "snRNAseq_region_contrast"
        ),
        "locked_value": (
            "descriptive_context_only_donor_confounded"
        ),
    },
    {
        "guardrail": (
            "MERFISH_temporal_significance"
        ),
        "locked_value": (
            "no_corrected_primary_temporal_trajectory"
        ),
    },
    {
        "guardrail": (
            "cellular_context_comparison"
        ),
        "locked_value": (
            "descriptive_compatibility_only"
        ),
    },
    {
        "guardrail": (
            "gene_set_equivalence"
        ),
        "locked_value": (
            "not_assumed_across_platforms"
        ),
    },
    {
        "guardrail": (
            "activity_dependent_plasticity"
        ),
        "locked_value": (
            "cross_platform_cellular_context_"
            "difference_retained"
        ),
    },
    {
        "guardrail": (
            "oligodendrocyte_myelination"
        ),
        "locked_value": (
            "mixed_MERFISH_cellular_context"
        ),
    },
    {
        "guardrail": (
            "cross_platform_spatial_direction"
        ),
        "locked_value": (
            "not_compared"
        ),
    },
    {
        "guardrail": (
            "new_cross_platform_statistics"
        ),
        "locked_value": "none",
    },
    {
        "guardrail": (
            "independent_replication_claim"
        ),
        "locked_value": "not_authorized",
    },
    {
        "guardrail": (
            "causal_claim"
        ),
        "locked_value": "not_authorized",
    },
]

write_tsv(
    guardrail_dir
    / "phase10B6_final_reporting_guardrails.tsv",
    guardrail_rows,
    [
        "guardrail",
        "locked_value",
    ],
)

# ---------------------------------------------------------
# Technical closure.
# ---------------------------------------------------------

b4_robust_modules = sum(
    int(
        row[
            "phase10B4_geometry_robust_signal_count"
        ]
    ) > 0
    for row in module_rows
)

b5_spatial_modules = sum(
    bool(
        row[
            "phase10B5_descriptive_spatial_heterogeneity"
        ]
    )
    for row in module_rows
)

confirmed_temporal = sum(
    bool(
        row[
            "phase10B5_temporal_confirmed"
        ]
    )
    for row in module_rows
)

stable_primary_temporal = sum(
    (
        row[
            "phase10B5_temporal_evidence_tier"
        ]
        == "T3_directionally_stable_descriptive"
    )
    for row in module_rows
)

technical_pass = (
    len(
        module_rows
    ) == 9
    and b4_robust_modules == 9
    and b5_spatial_modules == 9
    and compatible_count == 7
    and different_count == 1
    and mixed_count == 1
    and confirmed_temporal == 0
    and stable_primary_temporal == 4
    and all_b4_geometry_supported
    and all_b5_spatial_supported
)

status_value = (
    "passed_phase10B6_cross_platform_spatial_"
    "validation_convergence_synthesis_and_"
    "phase10B_canonical_closure_ready_for_"
    "phase10_continuation"
    if technical_pass
    else (
        "phase10B6_requires_manual_review"
    )
)

status = {
    "phase": "phase10B6",
    "locked_modules_integrated": len(
        module_rows
    ),
    "core_modules": len(
        core_modules
    ),
    "expanded_modules": len(
        expanded_modules
    ),
    "phase10B4_modules_with_geometry_robust_spatial_evidence": (
        b4_robust_modules
    ),
    "phase10B5_modules_with_descriptive_spatial_heterogeneity": (
        b5_spatial_modules
    ),
    "cellular_context_descriptively_compatible_modules": (
        compatible_count
    ),
    "cellular_context_different_modules": (
        different_count
    ),
    "cellular_context_mixed_or_unresolved_modules": (
        mixed_count
    ),
    "statistically_confirmed_MERFISH_temporal_modules": (
        confirmed_temporal
    ),
    "stable_descriptive_primary_MERFISH_temporal_modules": (
        stable_primary_temporal
    ),
    "new_expression_values_accessed": False,
    "H5AD_files_opened": 0,
    "new_cell_level_scores_computed": False,
    "new_spatial_statistics_computed": False,
    "new_cross_platform_hypothesis_tests_performed": (
        False
    ),
    "direct_cross_platform_replication_claimed": (
        False
    ),
    "independent_biological_replication_claimed": (
        False
    ),
    "population_level_inference_claimed": (
        False
    ),
    "cross_platform_spatial_direction_compared": (
        False
    ),
    "gene_set_equivalence_assumed": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B_closed": True,
    "phase10B6_status": status_value,
}

write_tsv(
    out
    / "phase10B6_status.tsv",
    [status],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B6 CROSS-PLATFORM "
    "VALIDATION SYNTHESIS =====",
    "",
    (
        "Locked modules integrated: "
        f"{len(module_rows)}/9"
    ),
    (
        "Core modules: "
        f"{len(core_modules)}/5"
    ),
    (
        "Expanded modules: "
        f"{len(expanded_modules)}/4"
    ),
    (
        "Modules with Phase10B4 geometry-robust "
        f"Visium evidence: {b4_robust_modules}/9"
    ),
    (
        "Modules with Phase10B5 descriptive "
        f"MERFISH spatial heterogeneity: "
        f"{b5_spatial_modules}/9"
    ),
    (
        "Descriptively compatible cellular contexts: "
        f"{compatible_count}/9"
    ),
    (
        "Different cellular contexts: "
        f"{different_count}/9"
    ),
    (
        "Mixed/unresolved cellular contexts: "
        f"{mixed_count}/9"
    ),
    (
        "Statistically confirmed MERFISH temporal "
        f"modules: {confirmed_temporal}"
    ),
    (
        "Stable descriptive primary MERFISH temporal "
        f"modules: {stable_primary_temporal}/4"
    ),
    "",
    "New expression values accessed: FALSE",
    "H5AD files opened: 0",
    "New cell-level scores computed: FALSE",
    "New spatial statistics computed: FALSE",
    (
        "New cross-platform hypothesis tests "
        "performed: FALSE"
    ),
    (
        "Direct cross-platform replication "
        "claimed: FALSE"
    ),
    (
        "Independent biological replication "
        "claimed: FALSE"
    ),
    (
        "Population-level inference claimed: FALSE"
    ),
    (
        "Cross-platform spatial direction "
        "compared: FALSE"
    ),
    "Gene-set equivalence assumed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "Phase 10B closed: TRUE",
    "",
    (
        "PHASE 10B6 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B6_report.txt"
).write_text(
    "\n".join(
        report
    )
    + "\n",
    encoding="utf-8",
)

print()
print(
    "\n".join(
        report
    )
)

print(
    "\n===== CROSS-PLATFORM MODULE SYNTHESIS ====="
)

for row in module_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "module",
                "phase10B4_geometry_robust_signal_count",
                "phase10B4_highest_snRNAseq_identity",
                "phase10B5_H2_localization_group",
                "cellular_context_compatibility",
                "phase10B5_temporal_evidence_tier",
                "cross_platform_synthesis_class",
            )
        )
    )

# ---------------------------------------------------------
# Output checksum.
# ---------------------------------------------------------

checksum_rows = []

for path in sorted(
    out.rglob("*")
):

    if (
        path.is_file()
        and path.name
        != "phase10B6_SHA256.tsv"
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
    / "phase10B6_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)

if not technical_pass:

    raise SystemExit(
        "Phase 10B6 requires manual review."
    )
