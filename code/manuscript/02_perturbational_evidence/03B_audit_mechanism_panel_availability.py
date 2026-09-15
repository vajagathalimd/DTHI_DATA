#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import re
import sys

import pandas as pd


PROJECT = Path(
    "."
)

INPUT_FILE = (
    PROJECT
    / "07_tables/main_tables/phase6/"
      "phase6B4B_all_signature_program_overlap.tsv.gz"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase6"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase6/"
      "phase6C3B_mechanism_panel_availability.log"
)

EXACT_MATCH_FILE = (
    TABLE_DIR
    / "phase6C3B_exact_mechanism_signature_matches.tsv"
)

MISSING_FILE = (
    TABLE_DIR
    / "phase6C3B_missing_exact_mechanism_signatures.tsv"
)

CANDIDATE_FILE = (
    TABLE_DIR
    / "phase6C3B_mechanism_signature_candidates.tsv"
)

CATEGORY_SUMMARY_FILE = (
    TABLE_DIR
    / "phase6C3B_mechanism_category_summary.tsv"
)

COMPLETION_FILE = (
    TABLE_DIR
    / "phase6C3B_completion_summary.tsv"
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


EXACT_PANEL = [
    # Cell-cycle and proliferation confounders
    (
        "cell_cycle_proliferation",
        "maturation_confounder",
        "HALLMARK_E2F_TARGETS",
        1,
    ),
    (
        "cell_cycle_proliferation",
        "maturation_confounder",
        "HALLMARK_G2M_CHECKPOINT",
        1,
    ),
    (
        "cell_cycle_proliferation",
        "maturation_confounder",
        "HALLMARK_MITOTIC_SPINDLE",
        1,
    ),
    (
        "cell_cycle_proliferation",
        "maturation_confounder",
        "HALLMARK_MYC_TARGETS_V1",
        2,
    ),
    (
        "cell_cycle_proliferation",
        "maturation_confounder",
        "HALLMARK_MYC_TARGETS_V2",
        2,
    ),

    # DNA damage, p53 and death
    (
        "DNA_damage_p53",
        "maturation_confounder",
        "HALLMARK_P53_PATHWAY",
        1,
    ),
    (
        "DNA_damage_p53",
        "maturation_confounder",
        "HALLMARK_DNA_REPAIR",
        1,
    ),
    (
        "apoptosis_cytotoxicity",
        "maturation_confounder",
        "HALLMARK_APOPTOSIS",
        1,
    ),

    # Broad stress responses
    (
        "oxidative_hypoxic_stress",
        "maturation_confounder",
        "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY",
        1,
    ),
    (
        "oxidative_hypoxic_stress",
        "maturation_confounder",
        "HALLMARK_HYPOXIA",
        1,
    ),
    (
        "inflammatory_stress",
        "maturation_confounder",
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
        1,
    ),
    (
        "inflammatory_stress",
        "maturation_confounder",
        "HALLMARK_INFLAMMATORY_RESPONSE",
        1,
    ),

    # Metabolic maturation
    (
        "mitochondrial_metabolic_maturation",
        "developmental_supportive",
        "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
        1,
    ),

    # Neuronal maturation
    (
        "neuronal_differentiation",
        "developmental_supportive",
        "GO_NEURON_DIFFERENTIATION",
        1,
    ),
    (
        "neuronal_differentiation",
        "developmental_supportive",
        "GO_NEUROGENESIS",
        1,
    ),
    (
        "neuronal_differentiation",
        "developmental_supportive",
        "GO_NEURON_PROJECTION_DEVELOPMENT",
        2,
    ),

    # Synaptic maturation
    (
        "synaptic_maturation",
        "developmental_supportive",
        "GO_SYNAPTIC_SIGNALING",
        1,
    ),
    (
        "synaptic_maturation",
        "developmental_supportive",
        "GO_CHEMICAL_SYNAPTIC_TRANSMISSION",
        1,
    ),
    (
        "synaptic_maturation",
        "developmental_supportive",
        "GO_REGULATION_OF_SYNAPTIC_TRANSMISSION",
        2,
    ),

    # Axonal development
    (
        "axon_neurite_development",
        "developmental_supportive",
        "GO_AXONOGENESIS",
        1,
    ),
    (
        "axon_neurite_development",
        "developmental_supportive",
        "GO_AXON_GUIDANCE",
        1,
    ),
    (
        "axon_neurite_development",
        "developmental_supportive",
        "GO_NEURITE_DEVELOPMENT",
        2,
    ),

    # Glial and myelin maturation
    (
        "glial_myelin_maturation",
        "developmental_supportive",
        "GO_MYELINATION",
        1,
    ),
    (
        "glial_myelin_maturation",
        "developmental_supportive",
        "GO_OLIGODENDROCYTE_DIFFERENTIATION",
        1,
    ),
    (
        "glial_myelin_maturation",
        "developmental_supportive",
        "GO_ASTROCYTE_DIFFERENTIATION",
        2,
    ),
    (
        "glial_myelin_maturation",
        "developmental_supportive",
        "GO_GLIAL_CELL_DIFFERENTIATION",
        2,
    ),
]


CATEGORY_PATTERNS = {
    "cell_cycle_proliferation": (
        r"CELL_CYCLE|MITOTIC|MITOSIS|G2M|G2_M|"
        r"E2F_TARGET|DNA_REPLICATION|CELL_DIVISION|"
        r"CHROMOSOME_SEGREGATION|MYC_TARGET"
    ),

    "DNA_damage_p53": (
        r"DNA_DAMAGE|DNA_REPAIR|P53|TP53|"
        r"DOUBLE_STRAND_BREAK|GENOTOX"
    ),

    "apoptosis_cytotoxicity": (
        r"APOPTOS|PROGRAMMED_CELL_DEATH|"
        r"CYTOTOX|CELL_DEATH"
    ),

    "oxidative_hypoxic_stress": (
        r"OXIDATIVE_STRESS|REACTIVE_OXYGEN|"
        r"HYPOXIA|HYPOXIC|OXIDANT"
    ),

    "inflammatory_stress": (
        r"INFLAMMATORY|INFLAMMATION|"
        r"TNFA_SIGNALING|NF_KB|NFKB|"
        r"CYTOKINE_MEDIATED"
    ),

    "mitochondrial_metabolic_maturation": (
        r"OXIDATIVE_PHOSPHORYLATION|"
        r"MITOCHONDRIAL_RESPIRATORY|"
        r"ELECTRON_TRANSPORT_CHAIN|"
        r"ATP_SYNTHESIS_COUPLED"
    ),

    "neuronal_differentiation": (
        r"NEURON_DIFFERENTIATION|NEUROGENESIS|"
        r"NEURON_DEVELOPMENT|"
        r"NEURON_PROJECTION_DEVELOPMENT"
    ),

    "synaptic_maturation": (
        r"SYNAPTIC_SIGNALING|SYNAPTIC_TRANSMISSION|"
        r"CHEMICAL_SYNAPTIC|SYNAPSE_ORGANIZATION|"
        r"NEUROTRANSMITTER"
    ),

    "axon_neurite_development": (
        r"AXONOGENESIS|AXON_GUIDANCE|"
        r"AXON_DEVELOPMENT|NEURITE|"
        r"NEURON_PROJECTION_GUIDANCE"
    ),

    "glial_myelin_maturation": (
        r"MYELINATION|OLIGODENDROCYTE|ASTROCYTE|"
        r"GLIAL_CELL_DIFFERENTIATION|"
        r"GLIOGENESIS|MYELIN"
    ),
}


ROLE_MAP = {
    "cell_cycle_proliferation":
        "maturation_confounder",

    "DNA_damage_p53":
        "maturation_confounder",

    "apoptosis_cytotoxicity":
        "maturation_confounder",

    "oxidative_hypoxic_stress":
        "maturation_confounder",

    "inflammatory_stress":
        "maturation_confounder",

    "mitochondrial_metabolic_maturation":
        "developmental_supportive",

    "neuronal_differentiation":
        "developmental_supportive",

    "synaptic_maturation":
        "developmental_supportive",

    "axon_neurite_development":
        "developmental_supportive",

    "glial_myelin_maturation":
        "developmental_supportive",
}


def normalize_signature(
    value: object,
) -> str:
    if pd.isna(
        value
    ):
        return ""

    value = str(
        value
    ).strip()

    value = re.sub(
        r"[\s\-]+",
        "_",
        value,
    )

    value = re.sub(
        r"_+",
        "_",
        value,
    )

    return value.upper()


def source_priority(
    source: object,
) -> int:
    source_text = str(
        source
    ).strip().lower()

    priorities = {
        "msigdb": 1,
        "bioplanet": 2,
        "dorothea": 3,
        "bryant stress signatures": 4,
    }

    return priorities.get(
        source_text,
        99,
    )


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing signature audit input: {INPUT_FILE}"
        )

    data = pd.read_csv(
        INPUT_FILE,
        sep="\t",
        low_memory=False,
    )

    required_columns = {
        "signature",
        "parent",
        "source",
        "subsource",
        "type",
        "direction",
        "ngene_parsed",
        "maturation_overlap",
        "maturation_overlap_fraction",
        "fetal_overlap",
        "fetal_overlap_fraction",
        "description",
    }

    missing_columns = sorted(
        required_columns
        - set(
            data.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(
                missing_columns
            )
        )

    numeric_columns = [
        "ngene_parsed",
        "maturation_overlap",
        "maturation_overlap_fraction",
        "fetal_overlap",
        "fetal_overlap_fraction",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data[
        "signature_normalized"
    ] = data[
        "signature"
    ].map(
        normalize_signature
    )

    data[
        "source_priority"
    ] = data[
        "source"
    ].map(
        source_priority
    )

    data[
        "preferred_reference_source"
    ] = data[
        "source"
    ].astype(str).str.lower().isin(
        {
            "msigdb",
            "bioplanet",
            "dorothea",
            "bryant stress signatures",
        }
    )

    data[
        "primary_size_eligible"
    ] = data[
        "ngene_parsed"
    ].between(
        10,
        500,
        inclusive="both",
    )

    panel = pd.DataFrame(
        EXACT_PANEL,
        columns=[
            "mechanism_category",
            "interpretation_role",
            "target_signature",
            "panel_priority",
        ],
    )

    panel[
        "target_signature_normalized"
    ] = panel[
        "target_signature"
    ].map(
        normalize_signature
    )

    exact_matches = panel.merge(
        data,
        how="left",
        left_on="target_signature_normalized",
        right_on="signature_normalized",
        suffixes=(
            "_panel",
            "_library",
        ),
    )

    exact_matches[
        "exact_signature_available"
    ] = exact_matches[
        "signature"
    ].notna()

    exact_matches = exact_matches.sort_values(
        [
            "mechanism_category",
            "panel_priority",
            "target_signature",
            "source_priority",
        ],
        ascending=[
            True,
            True,
            True,
            True,
        ],
    )

    exact_matches.to_csv(
        EXACT_MATCH_FILE,
        sep="\t",
        index=False,
    )

    available_targets = set(
        exact_matches.loc[
            exact_matches[
                "exact_signature_available"
            ],
            "target_signature",
        ]
    )

    missing_targets = panel.loc[
        ~panel[
            "target_signature"
        ].isin(
            available_targets
        )
    ].copy()

    missing_targets.to_csv(
        MISSING_FILE,
        sep="\t",
        index=False,
    )

    candidate_rows = []

    preferred_data = data.loc[
        data[
            "preferred_reference_source"
        ]
        & data[
            "primary_size_eligible"
        ]
        & data[
            "type"
        ].astype(str).str.lower().eq(
            "nondirectional"
        )
    ].copy()

    for category, pattern in CATEGORY_PATTERNS.items():
        matched = preferred_data.loc[
            preferred_data[
                "signature_normalized"
            ].str.contains(
                pattern,
                regex=True,
                na=False,
            )
        ].copy()

        matched[
            "mechanism_category"
        ] = category

        matched[
            "interpretation_role"
        ] = ROLE_MAP[
            category
        ]

        matched[
            "exact_panel_target"
        ] = matched[
            "signature_normalized"
        ].isin(
            panel.loc[
                panel[
                    "mechanism_category"
                ].eq(category),
                "target_signature_normalized",
            ]
        )

        matched[
            "developmental_program_overlap"
        ] = matched[
            [
                "maturation_overlap_fraction",
                "fetal_overlap_fraction",
            ]
        ].max(
            axis=1
        )

        matched = matched.sort_values(
            [
                "exact_panel_target",
                "source_priority",
                "developmental_program_overlap",
                "ngene_parsed",
                "signature",
            ],
            ascending=[
                False,
                True,
                False,
                True,
                True,
            ],
        )

        matched[
            "within_category_candidate_rank"
        ] = range(
            1,
            len(
                matched
            )
            + 1,
        )

        candidate_rows.append(
            matched.head(
                100
            )
        )

    candidate_table = pd.concat(
        candidate_rows,
        ignore_index=True,
    )

    candidate_columns = [
        "mechanism_category",
        "interpretation_role",
        "within_category_candidate_rank",
        "exact_panel_target",
        "signature",
        "parent",
        "source",
        "subsource",
        "type",
        "direction",
        "ngene_parsed",
        "maturation_overlap",
        "maturation_overlap_fraction",
        "fetal_overlap",
        "fetal_overlap_fraction",
        "description",
    ]

    candidate_table[
        candidate_columns
    ].to_csv(
        CANDIDATE_FILE,
        sep="\t",
        index=False,
    )

    summary_rows = []

    for category in CATEGORY_PATTERNS:
        category_panel = panel.loc[
            panel[
                "mechanism_category"
            ].eq(
                category
            )
        ]

        category_exact = exact_matches.loc[
            exact_matches[
                "mechanism_category"
            ].eq(
                category
            )
        ]

        category_candidates = candidate_table.loc[
            candidate_table[
                "mechanism_category"
            ].eq(
                category
            )
        ]

        summary_rows.append(
            {
                "mechanism_category":
                    category,

                "interpretation_role":
                    ROLE_MAP[
                        category
                    ],

                "exact_targets_requested":
                    len(
                        category_panel
                    ),

                "exact_targets_available":
                    category_exact.loc[
                        category_exact[
                            "exact_signature_available"
                        ],
                        "target_signature",
                    ].nunique(),

                "exact_targets_missing":
                    (
                        len(
                            category_panel
                        )
                        - category_exact.loc[
                            category_exact[
                                "exact_signature_available"
                            ],
                            "target_signature",
                        ].nunique()
                    ),

                "candidate_signatures_exported":
                    len(
                        category_candidates
                    ),

                "MSigDB_candidates":
                    int(
                        category_candidates[
                            "source"
                        ].astype(str).str.lower().eq(
                            "msigdb"
                        ).sum()
                    ),

                "Bioplanet_candidates":
                    int(
                        category_candidates[
                            "source"
                        ].astype(str).str.lower().eq(
                            "bioplanet"
                        ).sum()
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        CATEGORY_SUMMARY_FILE,
        sep="\t",
        index=False,
    )

    completion = pd.DataFrame(
        [
            {
                "mechanism_categories":
                    len(
                        CATEGORY_PATTERNS
                    ),

                "exact_targets_requested":
                    len(
                        panel
                    ),

                "exact_targets_available":
                    len(
                        available_targets
                    ),

                "exact_targets_missing":
                    len(
                        missing_targets
                    ),

                "candidate_signatures_exported":
                    len(
                        candidate_table
                    ),

                "chemical_results_used_for_panel_selection":
                    False,

                "outcome_independent_panel":
                    True,

                "Phase6C3B_status":
                    "completed",
            }
        ]
    )

    completion.to_csv(
        COMPLETION_FILE,
        sep="\t",
        index=False,
    )

    log_text = "\n".join(
        [
            "===== PHASE 6C3B MECHANISM PANEL SUMMARY =====",
            summary.to_string(
                index=False
            ),
            "",
            "===== MISSING EXACT TARGETS =====",
            (
                missing_targets.to_string(
                    index=False
                )
                if not missing_targets.empty
                else "None"
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
                "Phase 6C3B failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
