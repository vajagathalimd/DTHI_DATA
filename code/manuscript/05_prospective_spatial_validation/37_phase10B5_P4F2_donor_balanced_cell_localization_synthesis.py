from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


project = Path(sys.argv[1])
h1_summary_path = Path(sys.argv[2])
h2_summary_path = Path(sys.argv[3])
h1_calls_path = Path(sys.argv[4])
h2_calls_path = Path(sys.argv[5])
design_path = Path(sys.argv[6])
temporal_path = Path(sys.argv[7])
out = Path(sys.argv[8])

mapping_dir = out / "01_label_harmonization"
consensus_dir = out / "02_top_call_consensus"
sensitivity_dir = out / "03_donor_section_sensitivity"
synthesis_dir = out / "04_module_localization_synthesis"
claim_dir = out / "05_claim_hierarchy"
results_dir = out / "06_manuscript_results"
guardrail_dir = out / "07_reporting_guardrails"
audit_dir = out / "08_audit"

for directory in (
    mapping_dir,
    consensus_dir,
    sensitivity_dir,
    synthesis_dir,
    claim_dir,
    results_dir,
    guardrail_dir,
    audit_dir,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

core_modules = (
    "activity_dependent_plasticity",
    "astrocyte_maturation_metabolic_support",
    "neurogenesis_migration_layering",
    "patterning_arealization",
    "progenitor_radial_glia",
)

expanded_modules = (
    "axon_guidance_neurite_outgrowth",
    "oligodendrocyte_myelination",
    "synaptic_assembly_receptor_trafficking",
    "synaptic_membrane_structural_candidates",
)

all_modules = set(
    core_modules
    + expanded_modules
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


def harmonize_label(
    exact_label: str,
) -> tuple[str, str, str]:

    label = exact_label.strip()
    upper = label.upper()

    if upper.startswith("ASTRO"):

        return (
            "astrocyte",
            "glial_lineage",
            "prefix_ASTRO",
        )

    if upper == "EC":

        return (
            "endothelial",
            "vascular_endothelial",
            "exact_EC",
        )

    if upper == "OPC":

        return (
            "oligodendrocyte_precursor",
            "glial_lineage",
            "exact_OPC",
        )

    if upper == "MG":

        return (
            "microglia",
            "microglia",
            "exact_MG",
        )

    if (
        "ORG" in upper
        or upper in {
            "RG",
            "VRG",
            "TRG",
        }
        or upper.startswith("RG-")
        or upper.endswith("-RG")
    ):

        return (
            "radial_glia",
            "glial_lineage",
            "radial_glia_pattern",
        )

    if (
        "IPC" in upper
        or upper.startswith("INP")
    ):

        return (
            "intermediate_progenitor",
            "neural_progenitor_or_migratory",
            "IPC_or_INP_pattern",
        )

    if (
        upper == "DP"
        or upper.startswith("DP-")
    ):

        return (
            "dorsal_pallial_progenitor",
            "neural_progenitor_or_migratory",
            "DP_pattern",
        )

    if (
        upper.startswith("EN-MIG")
        or upper.startswith("EN-IZ")
        or upper.startswith("EN-OSVZ")
        or upper.startswith("EN-SVZ")
        or upper.startswith("EN-L2/IZ")
    ):

        return (
            "migrating_transitional_excitatory",
            "neural_progenitor_or_migratory",
            "migratory_excitatory_pattern",
        )

    if upper.startswith("EN-ET"):

        return (
            "extratelencephalic_excitatory",
            "neuronal",
            "EN_ET_pattern",
        )

    if (
        upper.startswith("EN-IT")
        or upper.startswith("EN-L2")
    ):

        return (
            "intratelencephalic_excitatory",
            "neuronal",
            "EN_IT_or_EN_L2_pattern",
        )

    if (
        upper == "IN"
        or upper.startswith("IN-")
    ):

        return (
            "inhibitory_neuron",
            "neuronal",
            "IN_pattern",
        )

    return (
        "other_unmapped",
        "other_unmapped",
        "unmapped",
    )


def summarize_counter(
    values: list[str],
) -> dict[str, Any]:

    if not values:

        raise ValueError(
            "Cannot summarize an empty value list."
        )

    counts = Counter(
        values
    )

    maximum = max(
        counts.values()
    )

    winners = sorted(
        label
        for label, count
        in counts.items()
        if count == maximum
    )

    return {
        "dominant_value": winners[0],
        "dominant_count": maximum,
        "dominant_fraction": (
            maximum
            / len(values)
        ),
        "dominant_tied": (
            len(winners) > 1
        ),
        "tied_dominant_values": (
            ";".join(
                winners
            )
        ),
        "all_counts": json.dumps(
            dict(
                sorted(
                    counts.items()
                )
            ),
            sort_keys=True,
        ),
    }


def consensus_strength(
    fraction: float,
    tied: bool,
) -> str:

    if tied:

        return "mixed_tied"

    if fraction >= 1.0:

        return "complete"

    if fraction >= 0.75:

        return "strong"

    if fraction >= 0.50:

        return "moderate"

    return "mixed"


h1_summary_rows, _ = read_tsv(
    h1_summary_path
)

h2_summary_rows, _ = read_tsv(
    h2_summary_path
)

h1_call_rows, _ = read_tsv(
    h1_calls_path
)

h2_call_rows, _ = read_tsv(
    h2_calls_path
)

design_rows, _ = read_tsv(
    design_path
)

temporal_rows, _ = read_tsv(
    temporal_path
)

expected_counts = {
    "H1 summary": (
        len(h1_summary_rows),
        514,
    ),
    "H2 summary": (
        len(h2_summary_rows),
        2236,
    ),
    "H1 top/bottom calls": (
        len(h1_call_rows),
        136,
    ),
    "H2 top/bottom calls": (
        len(h2_call_rows),
        136,
    ),
    "revised design": (
        len(design_rows),
        12,
    ),
    "temporal modules": (
        len(temporal_rows),
        5,
    ),
}

for label, (
    observed,
    expected,
) in expected_counts.items():

    if observed != expected:

        raise SystemExit(
            f"FAIL: {label} contains {observed} "
            f"rows; expected {expected}."
        )

if {
    row[
        "module"
    ]
    for row in h1_summary_rows
} != all_modules:

    raise SystemExit(
        "FAIL: unexpected H1 module set."
    )

if {
    row[
        "module"
    ]
    for row in h2_summary_rows
} != all_modules:

    raise SystemExit(
        "FAIL: unexpected H2 module set."
    )

design_lookup = {
    row[
        "archive_name"
    ]: row
    for row in design_rows
}

if len(
    design_lookup
) != 12:

    raise SystemExit(
        "FAIL: revised design does not contain "
        "12 unique sections."
    )

if sum(
    row[
        "effective_analysis_set"
    ] == "primary"
    for row in design_rows
) != 8:

    raise SystemExit(
        "FAIL: revised design does not contain "
        "eight primary sections."
    )

if sum(
    row[
        "effective_analysis_set"
    ] == "section_sensitivity"
    for row in design_rows
) != 4:

    raise SystemExit(
        "FAIL: revised design does not contain "
        "four sensitivity sections."
    )

temporal_claim_lookup = {
    row[
        "module"
    ]: row[
        "prespecified_claim_class"
    ]
    for row in temporal_rows
}

if set(
    temporal_claim_lookup
) != set(
    core_modules
):

    raise SystemExit(
        "FAIL: temporal evidence does not contain "
        "the five core modules."
    )

h1_top_rows = [
    row
    for row in h1_call_rows
    if row[
        "call_type"
    ] == "top"
]

h2_top_rows = [
    row
    for row in h2_call_rows
    if row[
        "call_type"
    ] == "top"
]

if len(
    h1_top_rows
) != 68:

    raise SystemExit(
        f"FAIL: expected 68 H1 top calls; "
        f"observed {len(h1_top_rows)}."
    )

if len(
    h2_top_rows
) != 68:

    raise SystemExit(
        f"FAIL: expected 68 H2 top calls; "
        f"observed {len(h2_top_rows)}."
    )

for annotation_label, rows in (
    (
        "H1_annotation",
        h1_top_rows,
    ),
    (
        "H2_annotation",
        h2_top_rows,
    ),
):

    keys = [
        (
            row[
                "archive_name"
            ],
            row[
                "module"
            ],
        )
        for row in rows
    ]

    if len(
        keys
    ) != len(
        set(
            keys
        )
    ):

        raise SystemExit(
            f"FAIL: duplicate {annotation_label} "
            "top calls detected."
        )


def augment_top_rows(
    rows: list[dict[str, str]],
    annotation_level: str,
) -> list[dict[str, Any]]:

    output: list[
        dict[str, Any]
    ] = []

    for row in rows:

        archive_name = row[
            "archive_name"
        ]

        if archive_name not in design_lookup:

            raise SystemExit(
                f"FAIL: {archive_name} absent from "
                "the revised design."
            )

        design = design_lookup[
            archive_name
        ]

        canonical_family, broad_compartment, rule = (
            harmonize_label(
                row[
                    "annotation_label"
                ]
            )
        )

        module = row[
            "module"
        ]

        output.append(
            {
                **row,
                "annotation_level": (
                    annotation_level
                ),
                "effective_analysis_set": (
                    design[
                        "effective_analysis_set"
                    ]
                ),
                "panel_class": design[
                    "panel_class"
                ],
                "processed_H5AD": design[
                    "processed_H5AD"
                ],
                "scoring_family": (
                    "cross_panel_harmonized_core"
                    if module in core_modules
                    else "expanded_panel_only"
                ),
                "prespecified_claim_class": (
                    temporal_claim_lookup[
                        module
                    ]
                    if module in core_modules
                    else (
                        "expanded_panel_only_"
                        "two_section_validation"
                    )
                ),
                "canonical_family": (
                    canonical_family
                ),
                "broad_compartment": (
                    broad_compartment
                ),
                "harmonization_rule": (
                    rule
                ),
                "exact_label_retained": True,
                "cross_section_harmonization_role": (
                    "descriptive_only"
                ),
            }
        )

    return output


h1_augmented = augment_top_rows(
    h1_top_rows,
    "H1_annotation",
)

h2_augmented = augment_top_rows(
    h2_top_rows,
    "H2_annotation",
)

augmented_columns = [
    "effective_analysis_set",
    "archive_name",
    "donor_id",
    "gestational_week",
    "panel_class",
    "processed_H5AD",
    "scoring_family",
    "prespecified_claim_class",
    "module",
    "annotation_level",
    "annotation_label",
    "canonical_family",
    "broad_compartment",
    "harmonization_rule",
    "cell_count",
    "mean_module_score",
    "median_module_score",
    "exact_label_retained",
    "cross_section_harmonization_role",
]

write_tsv(
    consensus_dir
    / "phase10B5_P4F2_H1_augmented_top_calls.tsv",
    h1_augmented,
    augmented_columns,
)

write_tsv(
    consensus_dir
    / "phase10B5_P4F2_H2_augmented_top_calls.tsv",
    h2_augmented,
    augmented_columns,
)


def build_mapping_table(
    summary_rows: list[dict[str, str]],
    top_rows: list[dict[str, Any]],
    annotation_level: str,
) -> list[dict[str, Any]]:

    summary_counts = Counter(
        row[
            "annotation_label"
        ]
        for row in summary_rows
    )

    top_counts = Counter(
        row[
            "annotation_label"
        ]
        for row in top_rows
    )

    output: list[
        dict[str, Any]
    ] = []

    for exact_label in sorted(
        summary_counts
    ):

        family, compartment, rule = (
            harmonize_label(
                exact_label
            )
        )

        output.append(
            {
                "annotation_level": (
                    annotation_level
                ),
                "exact_annotation_label": (
                    exact_label
                ),
                "canonical_family": (
                    family
                ),
                "broad_compartment": (
                    compartment
                ),
                "harmonization_rule": (
                    rule
                ),
                "all_localization_summary_occurrences": (
                    summary_counts[
                        exact_label
                    ]
                ),
                "top_call_occurrences": (
                    top_counts[
                        exact_label
                    ]
                ),
                "exact_label_retained": True,
                "harmonization_for_formal_inference": (
                    False
                ),
            }
        )

    return output


h1_mapping_rows = build_mapping_table(
    h1_summary_rows,
    h1_augmented,
    "H1_annotation",
)

h2_mapping_rows = build_mapping_table(
    h2_summary_rows,
    h2_augmented,
    "H2_annotation",
)

mapping_columns = [
    "annotation_level",
    "exact_annotation_label",
    "canonical_family",
    "broad_compartment",
    "harmonization_rule",
    "all_localization_summary_occurrences",
    "top_call_occurrences",
    "exact_label_retained",
    "harmonization_for_formal_inference",
]

write_tsv(
    mapping_dir
    / "phase10B5_P4F2_H1_label_harmonization.tsv",
    h1_mapping_rows,
    mapping_columns,
)

write_tsv(
    mapping_dir
    / "phase10B5_P4F2_H2_label_harmonization.tsv",
    h2_mapping_rows,
    mapping_columns,
)


scope_definitions: tuple[
    tuple[
        str,
        Callable[
            [
                dict[str, Any],
            ],
            bool,
        ],
    ],
    ...,
] = (
    (
        "all_selected_sections",
        lambda row: True,
    ),
    (
        "primary_independent_sections",
        lambda row: (
            row[
                "effective_analysis_set"
            ] == "primary"
        ),
    ),
    (
        "primary_300_panel_sections",
        lambda row: (
            row[
                "effective_analysis_set"
            ] == "primary"
            and row[
                "panel_class"
            ] == "300_gene_panel"
        ),
    ),
)


def build_consensus(
    rows: list[dict[str, Any]],
    annotation_level: str,
) -> list[dict[str, Any]]:

    output: list[
        dict[str, Any]
    ] = []

    for scope_name, predicate in scope_definitions:

        scope_rows = [
            row
            for row in rows
            if predicate(
                row
            )
        ]

        modules = sorted(
            {
                row[
                    "module"
                ]
                for row in scope_rows
            }
        )

        for module in modules:

            module_rows = [
                row
                for row in scope_rows
                if row[
                    "module"
                ] == module
            ]

            family_summary = summarize_counter(
                [
                    row[
                        "canonical_family"
                    ]
                    for row in module_rows
                ]
            )

            compartment_summary = summarize_counter(
                [
                    row[
                        "broad_compartment"
                    ]
                    for row in module_rows
                ]
            )

            output.append(
                {
                    "annotation_level": (
                        annotation_level
                    ),
                    "scope": scope_name,
                    "module": module,
                    "scoring_family": (
                        module_rows[0][
                            "scoring_family"
                        ]
                    ),
                    "prespecified_claim_class": (
                        module_rows[0][
                            "prespecified_claim_class"
                        ]
                    ),
                    "sections": len(
                        module_rows
                    ),
                    "independent_donors": len(
                        {
                            row[
                                "donor_id"
                            ]
                            for row in module_rows
                        }
                    ),
                    "dominant_canonical_family": (
                        family_summary[
                            "dominant_value"
                        ]
                    ),
                    "dominant_canonical_family_count": (
                        family_summary[
                            "dominant_count"
                        ]
                    ),
                    "dominant_canonical_family_fraction": (
                        family_summary[
                            "dominant_fraction"
                        ]
                    ),
                    "canonical_family_tied": (
                        family_summary[
                            "dominant_tied"
                        ]
                    ),
                    "canonical_family_counts": (
                        family_summary[
                            "all_counts"
                        ]
                    ),
                    "dominant_broad_compartment": (
                        compartment_summary[
                            "dominant_value"
                        ]
                    ),
                    "dominant_broad_compartment_count": (
                        compartment_summary[
                            "dominant_count"
                        ]
                    ),
                    "dominant_broad_compartment_fraction": (
                        compartment_summary[
                            "dominant_fraction"
                        ]
                    ),
                    "broad_compartment_tied": (
                        compartment_summary[
                            "dominant_tied"
                        ]
                    ),
                    "broad_compartment_counts": (
                        compartment_summary[
                            "all_counts"
                        ]
                    ),
                    "broad_compartment_consensus_strength": (
                        consensus_strength(
                            compartment_summary[
                                "dominant_fraction"
                            ],
                            compartment_summary[
                                "dominant_tied"
                            ],
                        )
                    ),
                    "exact_labels_retained": True,
                    "formal_inference_performed": (
                        False
                    ),
                }
            )

    return output


h1_consensus_rows = build_consensus(
    h1_augmented,
    "H1_annotation",
)

h2_consensus_rows = build_consensus(
    h2_augmented,
    "H2_annotation",
)

if len(
    h1_consensus_rows
) != 23:

    raise SystemExit(
        f"FAIL: expected 23 H1 consensus rows; "
        f"observed {len(h1_consensus_rows)}."
    )

if len(
    h2_consensus_rows
) != 23:

    raise SystemExit(
        f"FAIL: expected 23 H2 consensus rows; "
        f"observed {len(h2_consensus_rows)}."
    )

consensus_columns = [
    "annotation_level",
    "scope",
    "module",
    "scoring_family",
    "prespecified_claim_class",
    "sections",
    "independent_donors",
    "dominant_canonical_family",
    "dominant_canonical_family_count",
    "dominant_canonical_family_fraction",
    "canonical_family_tied",
    "canonical_family_counts",
    "dominant_broad_compartment",
    "dominant_broad_compartment_count",
    "dominant_broad_compartment_fraction",
    "broad_compartment_tied",
    "broad_compartment_counts",
    "broad_compartment_consensus_strength",
    "exact_labels_retained",
    "formal_inference_performed",
]

write_tsv(
    consensus_dir
    / "phase10B5_P4F2_H1_top_call_consensus.tsv",
    h1_consensus_rows,
    consensus_columns,
)

write_tsv(
    consensus_dir
    / "phase10B5_P4F2_H2_top_call_consensus.tsv",
    h2_consensus_rows,
    consensus_columns,
)


def donor_sensitivity_rows(
    rows: list[dict[str, Any]],
    annotation_level: str,
) -> list[dict[str, Any]]:

    sections_by_donor: dict[
        str,
        list[str],
    ] = defaultdict(list)

    for design in design_rows:

        sections_by_donor[
            design[
                "donor_id"
            ]
        ].append(
            design[
                "archive_name"
            ]
        )

    repeated_donors = sorted(
        donor
        for donor, sections
        in sections_by_donor.items()
        if len(
            sections
        ) > 1
    )

    if repeated_donors != [
        "FB080",
        "FB123",
    ]:

        raise SystemExit(
            "FAIL: expected FB080 and FB123 as "
            "repeated-section donors."
        )

    lookup = {
        (
            row[
                "archive_name"
            ],
            row[
                "module"
            ],
        ): row
        for row in rows
    }

    output: list[
        dict[str, Any]
    ] = []

    for donor in repeated_donors:

        donor_sections = sorted(
            sections_by_donor[
                donor
            ]
        )

        primary_sections = [
            section
            for section in donor_sections
            if design_lookup[
                section
            ][
                "effective_analysis_set"
            ] == "primary"
        ]

        if len(
            primary_sections
        ) != 1:

            raise SystemExit(
                f"FAIL: donor {donor} does not "
                "have exactly one primary section."
            )

        primary_section = primary_sections[0]

        for module in core_modules:

            section_rows = [
                lookup[
                    (
                        section,
                        module,
                    )
                ]
                for section in donor_sections
            ]

            primary_row = lookup[
                (
                    primary_section,
                    module,
                )
            ]

            compartments = [
                row[
                    "broad_compartment"
                ]
                for row in section_rows
            ]

            summary = summarize_counter(
                compartments
            )

            alternate_rows = [
                row
                for row in section_rows
                if row[
                    "archive_name"
                ] != primary_section
            ]

            output.append(
                {
                    "annotation_level": (
                        annotation_level
                    ),
                    "donor_id": donor,
                    "module": module,
                    "sections_evaluated": len(
                        section_rows
                    ),
                    "primary_archive": (
                        primary_section
                    ),
                    "primary_broad_compartment": (
                        primary_row[
                            "broad_compartment"
                        ]
                    ),
                    "alternate_archives": ";".join(
                        row[
                            "archive_name"
                        ]
                        for row in alternate_rows
                    ),
                    "alternate_broad_compartments": (
                        ";".join(
                            row[
                                "broad_compartment"
                            ]
                            for row in alternate_rows
                        )
                    ),
                    "dominant_broad_compartment": (
                        summary[
                            "dominant_value"
                        ]
                    ),
                    "dominant_count": (
                        summary[
                            "dominant_count"
                        ]
                    ),
                    "dominant_fraction": (
                        summary[
                            "dominant_fraction"
                        ]
                    ),
                    "all_sections_concordant": (
                        len(
                            set(
                                compartments
                            )
                        ) == 1
                    ),
                    "all_alternates_match_primary": (
                        all(
                            row[
                                "broad_compartment"
                            ]
                            == primary_row[
                                "broad_compartment"
                            ]
                            for row in alternate_rows
                        )
                    ),
                    "formal_inference_performed": (
                        False
                    ),
                }
            )

    return output


h1_sensitivity_rows = donor_sensitivity_rows(
    h1_augmented,
    "H1_annotation",
)

h2_sensitivity_rows = donor_sensitivity_rows(
    h2_augmented,
    "H2_annotation",
)

if len(
    h1_sensitivity_rows
) != 10:

    raise SystemExit(
        f"FAIL: expected 10 H1 donor-sensitivity "
        f"rows; observed "
        f"{len(h1_sensitivity_rows)}."
    )

if len(
    h2_sensitivity_rows
) != 10:

    raise SystemExit(
        f"FAIL: expected 10 H2 donor-sensitivity "
        f"rows; observed "
        f"{len(h2_sensitivity_rows)}."
    )

sensitivity_columns = [
    "annotation_level",
    "donor_id",
    "module",
    "sections_evaluated",
    "primary_archive",
    "primary_broad_compartment",
    "alternate_archives",
    "alternate_broad_compartments",
    "dominant_broad_compartment",
    "dominant_count",
    "dominant_fraction",
    "all_sections_concordant",
    "all_alternates_match_primary",
    "formal_inference_performed",
]

write_tsv(
    sensitivity_dir
    / "phase10B5_P4F2_H1_donor_section_sensitivity.tsv",
    h1_sensitivity_rows,
    sensitivity_columns,
)

write_tsv(
    sensitivity_dir
    / "phase10B5_P4F2_H2_donor_section_sensitivity.tsv",
    h2_sensitivity_rows,
    sensitivity_columns,
)

h1_concordant_donor_modules = sum(
    row[
        "all_sections_concordant"
    ]
    for row in h1_sensitivity_rows
)

h2_concordant_donor_modules = sum(
    row[
        "all_sections_concordant"
    ]
    for row in h2_sensitivity_rows
)

primary_h1_lookup = {
    row[
        "module"
    ]: row
    for row in h1_consensus_rows
    if row[
        "scope"
    ] == "primary_independent_sections"
}

primary_h2_lookup = {
    row[
        "module"
    ]: row
    for row in h2_consensus_rows
    if row[
        "scope"
    ] == "primary_independent_sections"
}

if set(
    primary_h1_lookup
) != all_modules:

    raise SystemExit(
        "FAIL: primary H1 consensus does not "
        "contain all nine modules."
    )

if set(
    primary_h2_lookup
) != all_modules:

    raise SystemExit(
        "FAIL: primary H2 consensus does not "
        "contain all nine modules."
    )

expected_H2_compartment = {
    "activity_dependent_plasticity": (
        "neuronal_or_migrating_excitatory"
    ),
    "astrocyte_maturation_metabolic_support": (
        "glial_lineage"
    ),
    "neurogenesis_migration_layering": (
        "neural_progenitor_or_migratory"
    ),
    "patterning_arealization": (
        "neural_progenitor_or_migratory"
    ),
    "progenitor_radial_glia": (
        "glial_lineage"
    ),
    "axon_guidance_neurite_outgrowth": (
        "neural_progenitor_or_migratory"
    ),
    "oligodendrocyte_myelination": (
        "mixed"
    ),
    "synaptic_assembly_receptor_trafficking": (
        "neuronal"
    ),
    "synaptic_membrane_structural_candidates": (
        "neuronal"
    ),
}

authorized_interpretation = {
    "activity_dependent_plasticity": (
        "relative_localization_to_neuronal_"
        "subclasses_across_primary_sections"
    ),
    "astrocyte_maturation_metabolic_support": (
        "relative_localization_to_radial_glial_"
        "and_astrocytic_subclasses"
    ),
    "neurogenesis_migration_layering": (
        "relative_localization_to_neurogenic_"
        "progenitor_and_migratory_compartments"
    ),
    "patterning_arealization": (
        "relative_localization_to_pallial_and_"
        "intermediate_progenitor_compartments"
    ),
    "progenitor_radial_glia": (
        "relative_localization_to_radial_glial_"
        "lineage_compartments"
    ),
    "axon_guidance_neurite_outgrowth": (
        "descriptive_localization_to_SVZ_and_"
        "progenitor_related_compartments"
    ),
    "oligodendrocyte_myelination": (
        "mixed_two_section_localization_without_"
        "a_cross_section_consensus"
    ),
    "synaptic_assembly_receptor_trafficking": (
        "descriptive_localization_to_neuronal_"
        "subclasses"
    ),
    "synaptic_membrane_structural_candidates": (
        "descriptive_localization_to_neuronal_"
        "subclasses"
    ),
}

h2_primary_top_lookup = {
    module: [
        row
        for row in h2_augmented
        if (
            row[
                "module"
            ] == module
            and row[
                "effective_analysis_set"
            ] == "primary"
        )
    ]
    for module in all_modules
}

module_synthesis_rows: list[
    dict[str, Any]
] = []

for module in sorted(
    all_modules
):

    h1 = primary_h1_lookup[
        module
    ]

    h2 = primary_h2_lookup[
        module
    ]

    expected_compartment = (
        expected_H2_compartment[
            module
        ]
    )

    h2_primary_rows = (
        h2_primary_top_lookup[
            module
        ]
    )

    if expected_compartment == "mixed":

        expected_count: Any = ""
        expected_fraction: Any = ""

        expected_supported = (
            bool(
                h2[
                    "broad_compartment_tied"
                ]
            )
            or float(
                h2[
                    "dominant_broad_compartment_fraction"
                ]
            ) <= 0.5
        )

        localization_strength = (
            "mixed_two_section_pattern"
        )

    elif module == "activity_dependent_plasticity":

        accepted_activity_families = {
            "extratelencephalic_excitatory",
            "intratelencephalic_excitatory",
            "migrating_transitional_excitatory",
            "inhibitory_neuron",
        }

        expected_count = sum(
            row[
                "canonical_family"
            ] in accepted_activity_families
            for row in h2_primary_rows
        )

        expected_fraction = (
            expected_count
            / len(
                h2_primary_rows
            )
        )

        expected_supported = (
            expected_fraction
            >= 0.75
        )

        localization_strength = (
            "complete"
            if expected_fraction == 1.0
            else (
                "strong"
                if expected_fraction >= 0.75
                else "mixed"
            )
        )

    else:

        expected_count = sum(
            row[
                "broad_compartment"
            ] == expected_compartment
            for row in h2_primary_rows
        )

        expected_fraction = (
            expected_count
            / len(
                h2_primary_rows
            )
        )

        expected_supported = (
            expected_fraction
            >= 0.75
        )

        localization_strength = (
            "complete"
            if expected_fraction == 1.0
            else (
                "strong"
                if expected_fraction >= 0.75
                else "mixed"
            )
        )

    module_synthesis_rows.append(
        {
            "module": module,
            "scoring_family": (
                "cross_panel_harmonized_core"
                if module in core_modules
                else "expanded_panel_only"
            ),
            "prespecified_claim_class": (
                temporal_claim_lookup[
                    module
                ]
                if module in core_modules
                else (
                    "expanded_panel_only_"
                    "two_section_validation"
                )
            ),
            "primary_sections": len(
                h2_primary_rows
            ),
            "H1_dominant_broad_compartment": (
                h1[
                    "dominant_broad_compartment"
                ]
            ),
            "H1_dominant_count": (
                h1[
                    "dominant_broad_compartment_count"
                ]
            ),
            "H1_dominant_fraction": (
                h1[
                    "dominant_broad_compartment_fraction"
                ]
            ),
            "H1_consensus_strength": (
                h1[
                    "broad_compartment_consensus_strength"
                ]
            ),
            "H2_dominant_broad_compartment": (
                h2[
                    "dominant_broad_compartment"
                ]
            ),
            "H2_dominant_count": (
                h2[
                    "dominant_broad_compartment_count"
                ]
            ),
            "H2_dominant_fraction": (
                h2[
                    "dominant_broad_compartment_fraction"
                ]
            ),
            "H2_consensus_strength": (
                h2[
                    "broad_compartment_consensus_strength"
                ]
            ),
            "expected_H2_broad_compartment": (
                expected_compartment
            ),
            "expected_H2_support_count": (
                expected_count
            ),
            "expected_H2_support_fraction": (
                expected_fraction
            ),
            "expected_H2_pattern_supported": (
                expected_supported
            ),
            "localization_strength": (
                localization_strength
            ),
            "authorized_interpretation": (
                authorized_interpretation[
                    module
                ]
            ),
            "statistical_localization_claim_authorized": (
                False
            ),
        }
    )

if len(
    module_synthesis_rows
) != 9:

    raise SystemExit(
        f"FAIL: expected nine module synthesis "
        f"rows; observed "
        f"{len(module_synthesis_rows)}."
    )

write_tsv(
    synthesis_dir
    / "phase10B5_P4F2_module_localization_synthesis.tsv",
    module_synthesis_rows,
    [
        "module",
        "scoring_family",
        "prespecified_claim_class",
        "primary_sections",
        "H1_dominant_broad_compartment",
        "H1_dominant_count",
        "H1_dominant_fraction",
        "H1_consensus_strength",
        "H2_dominant_broad_compartment",
        "H2_dominant_count",
        "H2_dominant_fraction",
        "H2_consensus_strength",
        "expected_H2_broad_compartment",
        "expected_H2_support_count",
        "expected_H2_support_fraction",
        "expected_H2_pattern_supported",
        "localization_strength",
        "authorized_interpretation",
        "statistical_localization_claim_authorized",
    ],
)

synthesis_lookup = {
    row[
        "module"
    ]: row
    for row in module_synthesis_rows
}

required_support = {
    "activity_dependent_plasticity": 8,
    "astrocyte_maturation_metabolic_support": 8,
    "neurogenesis_migration_layering": 7,
    "patterning_arealization": 7,
    "progenitor_radial_glia": 8,
    "axon_guidance_neurite_outgrowth": 2,
    "synaptic_assembly_receptor_trafficking": 2,
    "synaptic_membrane_structural_candidates": 2,
}

for module, expected_count in required_support.items():

    observed_count = int(
        synthesis_lookup[
            module
        ][
            "expected_H2_support_count"
        ]
    )

    if observed_count != expected_count:

        raise SystemExit(
            f"FAIL: {module} H2 support count "
            f"is {observed_count}; expected "
            f"{expected_count}."
        )

oligo = synthesis_lookup[
    "oligodendrocyte_myelination"
]

if oligo[
    "localization_strength"
] != "mixed_two_section_pattern":

    raise SystemExit(
        "FAIL: oligodendrocyte/myelination "
        "localization was not classified as mixed."
    )

claim_rows = [
    {
        "claim_id": "LOCALIZATION_C1",
        "claim_level": "primary_methodological",
        "claim": (
            "H1 and H2 localization represents "
            "relative within-section enrichment and "
            "not a between-section temporal effect."
        ),
        "support": (
            "within_section_gene_standardization_"
            "with_zero_section_wide_module_means"
        ),
        "authorized_language": (
            "relatively_localized_or_enriched_"
            "within_each_section"
        ),
        "prohibited_language": (
            "significantly_higher_between_cell_types"
        ),
    },
    {
        "claim_id": "LOCALIZATION_C2",
        "claim_level": "excitatory_lineage_localization",
        "claim": (
            "Activity-dependent plasticity localized "
            "most strongly to neuronal or migrating-"
            "excitatory subclasses in all eight "
            "independent primary sections."
        ),
        "support": (
            "H2_neuronal_or_migrating_excitatory_"
            "top_call_8_of_8_primary_sections"
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
        "claim_id": "LOCALIZATION_C3",
        "claim_level": "glial_localization",
        "claim": (
            "Astrocyte maturation/metabolic support "
            "and progenitor/radial-glia modules "
            "localized to radial-glial or astrocytic "
            "subclasses in all eight primary sections."
        ),
        "support": (
            "H2_glial_lineage_top_call_8_of_8_for_"
            "both_modules"
        ),
        "authorized_language": (
            "consistent_descriptive_glial_lineage_"
            "localization"
        ),
        "prohibited_language": (
            "cell_type_specific_statistical_enrichment"
        ),
    },
    {
        "claim_id": "LOCALIZATION_C4",
        "claim_level": "neurogenic_localization",
        "claim": (
            "Neurogenesis/migration and patterning/"
            "arealization localized predominantly to "
            "progenitor or migratory compartments."
        ),
        "support": (
            "H2_neural_progenitor_or_migratory_"
            "top_call_7_of_8_for_both_modules"
        ),
        "authorized_language": (
            "strong_descriptive_developmental_"
            "compartment_localization"
        ),
        "prohibited_language": (
            "exclusive_progenitor_specificity"
        ),
    },
    {
        "claim_id": "LOCALIZATION_C5",
        "claim_level": "section_sensitivity",
        "claim": (
            "Broad-compartment localization was "
            "concordant across all repeated-section "
            "evaluations at H2 and all but one at H1."
        ),
        "support": (
            "H2_10_of_10_and_H1_9_of_10_"
            "donor_module_evaluations"
        ),
        "authorized_language": (
            "robust_to_available_within_donor_"
            "section_choice"
        ),
        "prohibited_language": (
            "independently_replicated_by_each_section"
        ),
    },
    {
        "claim_id": "LOCALIZATION_C6",
        "claim_level": "expanded_panel",
        "claim": (
            "Expanded-panel axon-guidance localized "
            "to SVZ/progenitor compartments, whereas "
            "both synaptic modules localized to "
            "neuronal subclasses in both sections."
        ),
        "support": (
            "two_of_two_expanded_sections_for_each_"
            "stated_pattern"
        ),
        "authorized_language": (
            "descriptive_two_section_localization"
        ),
        "prohibited_language": (
            "generalized_cell_type_specificity"
        ),
    },
    {
        "claim_id": "LOCALIZATION_C7",
        "claim_level": "expanded_panel_limitation",
        "claim": (
            "The oligodendrocyte/myelination module "
            "showed different top subclasses in the "
            "two expanded-panel sections."
        ),
        "support": (
            "GW18_inhibitory_neuronal_and_GW20_OPC_"
            "top_H2_calls"
        ),
        "authorized_language": (
            "mixed_descriptive_localization"
        ),
        "prohibited_language": (
            "validated_OPC_specific_localization"
        ),
    },
]

write_tsv(
    claim_dir
    / "phase10B5_P4F2_cell_localization_claim_hierarchy.tsv",
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

results_text = """Phase 10B5 MERFISH cell-class localization

Cell-class localization was evaluated using module scores standardized within each individual section. These scores describe relative localization among annotated cells within the same section; they do not represent between-section temporal differences, and no cell-level hypothesis tests were performed.

At the H2 subclass level, the activity-dependent plasticity module localized most strongly to neuronal or migrating-excitatory subclasses in all eight independent primary sections. The astrocyte maturation/metabolic-support and progenitor/radial-glia modules localized to radial-glial or astrocytic subclasses in all eight primary sections.

The neurogenesis/migration/layering and patterning/arealization modules localized predominantly to neural-progenitor or migratory compartments in seven of eight primary sections. In the gestational-week-34 section, their highest H2 localization shifted toward neuronal subclasses, consistent with the later developmental composition of this specimen.

The broad-compartment assignments were concordant across all ten repeated-section donor-module evaluations at the H2 level. At H1, nine of ten evaluations were concordant; the only discordance involved the astrocyte maturation/metabolic-support module across FB080 sections, where the highest broad H1 class varied between radial glia and endothelial cells. Exact annotation labels were retained in all outputs, and the broader mapping was used only for descriptive cross-section synthesis.

Among the expanded-panel modules, axon-guidance/neurite-outgrowth localized to SVZ or progenitor-related subclasses in both sections. Synaptic assembly/receptor trafficking and synaptic-membrane structural modules localized to neuronal subclasses in both sections. The oligodendrocyte/myelination module showed a mixed pattern, with an inhibitory-neuronal top subclass at gestational week 18 and an OPC top subclass at gestational week 20. Because only two expanded-panel sections were available, these patterns remain descriptive.
"""

(
    results_dir
    / "phase10B5_P4F2_manuscript_ready_cell_localization_results.txt"
).write_text(
    results_text,
    encoding="utf-8",
)

guardrail_rows = [
    {
        "guardrail": (
            "localization_estimand"
        ),
        "locked_value": (
            "relative_within_section_module_"
            "localization"
        ),
        "reason": (
            "within_section_gene_z_scores_force_"
            "section_wide_means_to_zero"
        ),
    },
    {
        "guardrail": (
            "primary_annotation_resolution"
        ),
        "locked_value": (
            "H1_broad_and_H2_subclass_with_H2_"
            "preferred_for_biological_synthesis"
        ),
        "reason": (
            "H2_resolves_subclasses_hidden_by_H1_"
            "broad_class_averaging"
        ),
    },
    {
        "guardrail": (
            "exact_labels"
        ),
        "locked_value": (
            "retained_in_all_canonical_outputs"
        ),
        "reason": (
            "annotation_taxonomies_vary_across_ages"
        ),
    },
    {
        "guardrail": (
            "cross_section_harmonization"
        ),
        "locked_value": (
            "descriptive_biological_family_mapping_"
            "only"
        ),
        "reason": (
            "harmonized_labels_are_not_original_"
            "author_annotations"
        ),
    },
    {
        "guardrail": (
            "primary_section_weighting"
        ),
        "locked_value": (
            "one_primary_section_per_independent_"
            "donor"
        ),
        "reason": (
            "prevents_FB080_and_FB123_sensitivity_"
            "sections_from_overweighting_consensus"
        ),
    },
    {
        "guardrail": (
            "repeated_sections"
        ),
        "locked_value": (
            "within_donor_sensitivity_only"
        ),
        "reason": (
            "alternate_sections_are_not_independent_"
            "biological_replicates"
        ),
    },
    {
        "guardrail": (
            "cell_level_hypothesis_tests"
        ),
        "locked_value": (
            "not_authorized"
        ),
        "reason": (
            "cells_are_not_independent_biological_"
            "replicates"
        ),
    },
    {
        "guardrail": (
            "expanded_panel"
        ),
        "locked_value": (
            "two_section_descriptive_localization"
        ),
        "reason": (
            "only_one_section_per_age_is_available"
        ),
    },
    {
        "guardrail": (
            "H3_annotations"
        ),
        "locked_value": (
            "excluded"
        ),
        "reason": (
            "H3_annotations_are_substantially_"
            "incomplete"
        ),
    },
]

write_tsv(
    guardrail_dir
    / "phase10B5_P4F2_reporting_guardrails.tsv",
    guardrail_rows,
    [
        "guardrail",
        "locked_value",
        "reason",
    ],
)

core_complete_H2_modules = sum(
    (
        row[
            "module"
        ] in core_modules
        and row[
            "localization_strength"
        ] == "complete"
    )
    for row in module_synthesis_rows
)

core_strong_H2_modules = sum(
    (
        row[
            "module"
        ] in core_modules
        and row[
            "localization_strength"
        ] == "strong"
    )
    for row in module_synthesis_rows
)

expanded_complete_modules = sum(
    (
        row[
            "module"
        ] in expanded_modules
        and row[
            "localization_strength"
        ] == "complete"
    )
    for row in module_synthesis_rows
)

expanded_mixed_modules = sum(
    (
        row[
            "module"
        ] in expanded_modules
        and row[
            "localization_strength"
        ] == "mixed_two_section_pattern"
    )
    for row in module_synthesis_rows
)

technical_pass = (
    len(
        h1_augmented
    ) == 68
    and len(
        h2_augmented
    ) == 68
    and len(
        h1_consensus_rows
    ) == 23
    and len(
        h2_consensus_rows
    ) == 23
    and len(
        h1_sensitivity_rows
    ) == 10
    and len(
        h2_sensitivity_rows
    ) == 10
    and h1_concordant_donor_modules == 9
    and h2_concordant_donor_modules == 10
    and len(
        module_synthesis_rows
    ) == 9
    and core_complete_H2_modules == 3
    and core_strong_H2_modules == 2
    and expanded_complete_modules == 3
    and expanded_mixed_modules == 1
)

status_value = (
    "passed_phase10B5_P4F2_donor_balanced_"
    "H1_H2_descriptive_localization_synthesis_"
    "and_claim_lock_ready_for_spatial_"
    "coordinate_mapping"
    if technical_pass
    else (
        "phase10B5_P4F2_requires_manual_review"
    )
)

status = {
    "phase": "phase10B5_P4F2",
    "H1_localization_summary_rows_consumed": len(
        h1_summary_rows
    ),
    "H2_localization_summary_rows_consumed": len(
        h2_summary_rows
    ),
    "H1_top_calls_consumed": len(
        h1_augmented
    ),
    "H2_top_calls_consumed": len(
        h2_augmented
    ),
    "H1_unique_exact_labels_harmonized": len(
        h1_mapping_rows
    ),
    "H2_unique_exact_labels_harmonized": len(
        h2_mapping_rows
    ),
    "H1_consensus_rows": len(
        h1_consensus_rows
    ),
    "H2_consensus_rows": len(
        h2_consensus_rows
    ),
    "primary_independent_sections": 8,
    "primary_300_panel_sections": 6,
    "repeated_section_donors": 2,
    "H1_donor_module_sensitivity_rows": len(
        h1_sensitivity_rows
    ),
    "H2_donor_module_sensitivity_rows": len(
        h2_sensitivity_rows
    ),
    "H1_fully_concordant_donor_module_evaluations": (
        h1_concordant_donor_modules
    ),
    "H2_fully_concordant_donor_module_evaluations": (
        h2_concordant_donor_modules
    ),
    "module_localization_synthesis_rows": len(
        module_synthesis_rows
    ),
    "core_complete_H2_consensus_modules": (
        core_complete_H2_modules
    ),
    "core_strong_H2_consensus_modules": (
        core_strong_H2_modules
    ),
    "expanded_complete_H2_consensus_modules": (
        expanded_complete_modules
    ),
    "expanded_mixed_H2_modules": (
        expanded_mixed_modules
    ),
    "exact_annotation_labels_retained": True,
    "descriptive_cross_section_label_harmonization_performed": (
        True
    ),
    "harmonization_used_for_formal_inference": (
        False
    ),
    "expression_values_accessed_in_this_phase": (
        False
    ),
    "H5AD_files_opened_in_this_phase": False,
    "cell_level_scores_accessed_in_this_phase": (
        False
    ),
    "new_hypothesis_tests_performed": False,
    "cell_level_inference_performed": False,
    "between_section_pooling_performed": False,
    "spatial_coordinates_accessed": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P4F2_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B5_P4F2_status.tsv",
    [status],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P4F2 DONOR-BALANCED "
    "H1/H2 LOCALIZATION SYNTHESIS =====",
    "",
    (
        "H1 top calls synthesized: "
        f"{len(h1_augmented)}"
    ),
    (
        "H2 top calls synthesized: "
        f"{len(h2_augmented)}"
    ),
    (
        "H1 consensus rows: "
        f"{len(h1_consensus_rows)}"
    ),
    (
        "H2 consensus rows: "
        f"{len(h2_consensus_rows)}"
    ),
    (
        "Primary independent sections: "
        "8"
    ),
    (
        "H1 fully concordant donor-module "
        f"evaluations: {h1_concordant_donor_modules}/10"
    ),
    (
        "H2 fully concordant donor-module "
        f"evaluations: {h2_concordant_donor_modules}/10"
    ),
    (
        "Core complete H2 consensus modules: "
        f"{core_complete_H2_modules}/5"
    ),
    (
        "Core strong H2 consensus modules: "
        f"{core_strong_H2_modules}/5"
    ),
    (
        "Expanded complete H2 consensus modules: "
        f"{expanded_complete_modules}/4"
    ),
    (
        "Expanded mixed H2 modules: "
        f"{expanded_mixed_modules}/4"
    ),
    "",
    "Exact annotation labels retained: TRUE",
    (
        "Descriptive cross-section label "
        "harmonization performed: TRUE"
    ),
    (
        "Harmonization used for formal inference: "
        "FALSE"
    ),
    "Expression values accessed in this phase: FALSE",
    "H5AD files opened in this phase: FALSE",
    "Cell-level scores accessed in this phase: FALSE",
    "New hypothesis tests performed: FALSE",
    "Cell-level inference performed: FALSE",
    "Between-section pooling performed: FALSE",
    "Spatial coordinates accessed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P4F2 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P4F2_report.txt"
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
    "\n===== MODULE LOCALIZATION SYNTHESIS ====="
)

for row in module_synthesis_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "module",
                "primary_sections",
                "H1_dominant_broad_compartment",
                "H1_dominant_fraction",
                "H2_dominant_broad_compartment",
                "H2_dominant_fraction",
                "expected_H2_broad_compartment",
                "expected_H2_support_count",
                "expected_H2_support_fraction",
                "localization_strength",
            )
        )
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
        != "phase10B5_P4F2_SHA256.tsv"
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
    / "phase10B5_P4F2_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)

if not technical_pass:

    raise SystemExit(
        "Phase 10B5-P4F2 requires manual review."
    )
