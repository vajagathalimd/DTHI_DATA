from __future__ import annotations

import csv
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


project = Path(sys.argv[1])
module_summary_path = Path(sys.argv[2])
identity_summary_path = Path(sys.argv[3])
region_delta_path = Path(sys.argv[4])
primary_path = Path(sys.argv[5])
robustness_path = Path(sys.argv[6])
out = Path(sys.argv[7])

evidence_dir = (
    out
    / "01_spatial_evidence_hierarchy"
)

identity_dir = (
    out
    / "02_snRNAseq_identity_context"
)

module_dir = (
    out
    / "03_module_level_synthesis"
)

audit_dir = (
    out
    / "04_audit"
)

for directory in (
    evidence_dir,
    identity_dir,
    module_dir,
    audit_dir,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

minimum_identity_nuclei = 100
top_identity_count = 5

depth_order = (
    "SP",
    "superficial_CP",
    "L4",
    "deep_CP",
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

        rows = list(reader)
        columns = reader.fieldnames or []

    return rows, columns


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


def as_float(
    value: Any,
    label: str,
) -> float:

    try:
        result = float(value)
    except (
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(
            f"FAIL: invalid numeric value for {label}: "
            f"{value!r}; {error}"
        )

    if not math.isfinite(result):
        raise SystemExit(
            f"FAIL: nonfinite numeric value for {label}."
        )

    return result


def require_columns(
    columns: list[str],
    required: set[str],
    label: str,
) -> None:

    missing = sorted(
        required - set(columns)
    )

    if missing:
        raise SystemExit(
            f"FAIL: {label} lacks columns: "
            + ";".join(missing)
        )


module_rows, module_columns = read_tsv(
    module_summary_path
)

require_columns(
    module_columns,
    {
        "module",
        "locked_genes",
    },
    "module summary",
)

locked_gene_counts = {
    row["module"]: int(
        row["locked_genes"]
    )
    for row in module_rows
}

modules = sorted(
    locked_gene_counts
)

if len(modules) != 9:
    raise SystemExit(
        f"FAIL: expected nine locked modules, "
        f"observed {len(modules)}."
    )

identity_rows, identity_columns = read_tsv(
    identity_summary_path
)

require_columns(
    identity_columns,
    {
        "author_identity",
        "module",
        "units",
        "mean_score",
        "median_score",
    },
    "snRNA-seq identity summary",
)

identity_by_module: dict[
    str,
    list[dict[str, Any]],
] = defaultdict(list)

for row in identity_rows:

    module = row["module"]

    if module not in locked_gene_counts:
        continue

    units = int(
        float(
            row["units"]
        )
    )

    identity_by_module[
        module
    ].append(
        {
            "author_identity": (
                row["author_identity"]
            ),
            "units": units,
            "mean_score": as_float(
                row["mean_score"],
                "identity mean_score",
            ),
            "median_score": as_float(
                row["median_score"],
                "identity median_score",
            ),
        }
    )

identity_context_rows: list[
    dict[str, Any]
] = []

identity_context_lookup: dict[
    str,
    dict[str, Any],
] = {}

for module in modules:

    eligible = [
        row
        for row in identity_by_module[
            module
        ]
        if row["units"]
        >= minimum_identity_nuclei
    ]

    if not eligible:
        raise SystemExit(
            f"FAIL: no eligible identity context "
            f"for module {module}."
        )

    positive = sorted(
        eligible,
        key=lambda row: (
            -row["mean_score"],
            -row["units"],
            row["author_identity"],
        ),
    )[
        :top_identity_count
    ]

    negative = sorted(
        eligible,
        key=lambda row: (
            row["mean_score"],
            -row["units"],
            row["author_identity"],
        ),
    )[
        :top_identity_count
    ]

    top_positive_names = ";".join(
        row["author_identity"]
        for row in positive
    )

    top_positive_values = ";".join(
        f"{row['mean_score']:.6g}"
        for row in positive
    )

    top_negative_names = ";".join(
        row["author_identity"]
        for row in negative
    )

    top_negative_values = ";".join(
        f"{row['mean_score']:.6g}"
        for row in negative
    )

    context = {
        "module": module,
        "eligible_author_identities": len(
            eligible
        ),
        "minimum_nuclei_per_identity": (
            minimum_identity_nuclei
        ),
        "top_positive_identities": (
            top_positive_names
        ),
        "top_positive_mean_scores": (
            top_positive_values
        ),
        "top_negative_identities": (
            top_negative_names
        ),
        "top_negative_mean_scores": (
            top_negative_values
        ),
        "highest_identity": positive[0][
            "author_identity"
        ],
        "highest_identity_mean_score": (
            positive[0]["mean_score"]
        ),
        "lowest_identity": negative[0][
            "author_identity"
        ],
        "lowest_identity_mean_score": (
            negative[0]["mean_score"]
        ),
        "identity_context_role": (
            "cellular_context_only_not_independent_replication"
        ),
    }

    identity_context_rows.append(
        context
    )

    identity_context_lookup[
        module
    ] = context

write_tsv(
    identity_dir
    / "phase10B4_A5_snRNAseq_identity_context.tsv",
    identity_context_rows,
    [
        "module",
        "eligible_author_identities",
        "minimum_nuclei_per_identity",
        "top_positive_identities",
        "top_positive_mean_scores",
        "top_negative_identities",
        "top_negative_mean_scores",
        "highest_identity",
        "highest_identity_mean_score",
        "lowest_identity",
        "lowest_identity_mean_score",
        "identity_context_role",
    ],
)

region_rows, region_columns = read_tsv(
    region_delta_path
)

require_columns(
    region_columns,
    {
        "module",
        "frontal",
        "occipital",
        "frontal_minus_occipital",
        "comparison_interpretation",
        "inferential_test_performed",
    },
    "snRNA-seq region descriptive delta",
)

region_lookup = {
    row["module"]: row
    for row in region_rows
}

if set(region_lookup) != set(modules):
    raise SystemExit(
        "FAIL: snRNA-seq region delta does not contain "
        "exactly the nine locked modules."
    )

for module, row in region_lookup.items():

    if as_bool(
        row[
            "inferential_test_performed"
        ]
    ):
        raise SystemExit(
            "FAIL: snRNA-seq region table unexpectedly "
            "contains inferential testing."
        )

    if (
        row[
            "comparison_interpretation"
        ]
        != (
            "descriptive_only_region_"
            "fully_confounded_by_donor"
        )
    ):
        raise SystemExit(
            f"FAIL: missing donor-confounding annotation "
            f"for {module}."
        )

primary_rows, primary_columns = read_tsv(
    primary_path
)

require_columns(
    primary_columns,
    {
        "depth_stratum",
        "module",
        "direction",
        "V1_minus_V2",
        "Hedges_g",
        "permutation_p_two_sided",
        "within_depth_maxT_p",
        "within_depth_BH_FDR",
        "global_BH_FDR",
        "global_Holm_FWER",
        "section_specific_primary_signal",
    },
    "A3 primary results",
)

if len(primary_rows) != 36:
    raise SystemExit(
        f"FAIL: expected 36 A3 primary rows, "
        f"observed {len(primary_rows)}."
    )

primary_lookup = {
    (
        row["depth_stratum"],
        row["module"],
    ): row
    for row in primary_rows
}

robustness_rows, robustness_columns = read_tsv(
    robustness_path
)

require_columns(
    robustness_columns,
    {
        "depth_stratum",
        "module",
        "A3_section_specific_primary_signal",
        "geometry_candidates",
        "unique_grid_widths",
        "direction_concordance_fraction",
        "width_median_direction_concordance",
        "median_V1_minus_V2",
        "median_Hedges_g",
        "geometry_robust_effect",
        "geometry_robust_A3_primary_signal",
    },
    "A4 geometry robustness",
)

if len(robustness_rows) != 36:
    raise SystemExit(
        f"FAIL: expected 36 A4 robustness rows, "
        f"observed {len(robustness_rows)}."
    )

robustness_lookup = {
    (
        row["depth_stratum"],
        row["module"],
    ): row
    for row in robustness_rows
}

if set(primary_lookup) != set(
    robustness_lookup
):
    raise SystemExit(
        "FAIL: A3 and A4 test keys do not match."
    )

evidence_rows: list[
    dict[str, Any]
] = []

for depth in depth_order:

    for module in modules:

        key = (
            depth,
            module,
        )

        primary = primary_lookup[key]
        robust = robustness_lookup[key]
        identity = identity_context_lookup[
            module
        ]
        region = region_lookup[module]

        primary_signal = as_bool(
            primary[
                "section_specific_primary_signal"
            ]
        )

        geometry_robust = as_bool(
            robust[
                "geometry_robust_effect"
            ]
        )

        robust_primary = as_bool(
            robust[
                "geometry_robust_A3_primary_signal"
            ]
        )

        global_bh = as_float(
            primary["global_BH_FDR"],
            "global BH-FDR",
        )

        global_holm = as_float(
            primary["global_Holm_FWER"],
            "global Holm-FWER",
        )

        within_max_t = as_float(
            primary["within_depth_maxT_p"],
            "within-depth maxT p",
        )

        raw_p = as_float(
            primary[
                "permutation_p_two_sided"
            ],
            "raw permutation p",
        )

        if robust_primary:
            evidence_class = (
                "S1_geometry_robust_section_specific_corrected"
            )

        elif primary_signal:
            evidence_class = (
                "S2_section_specific_corrected_"
                "not_geometry_robust"
            )

        elif global_bh < 0.05:
            evidence_class = (
                "S3_global_FDR_only"
            )

        elif raw_p < 0.05:
            evidence_class = (
                "S4_nominal_section_specific"
            )

        else:
            evidence_class = (
                "S5_descriptive_no_spatial_signal"
            )

        evidence_rows.append(
            {
                "depth_stratum": depth,
                "module": module,
                "locked_genes": (
                    locked_gene_counts[
                        module
                    ]
                ),
                "spatial_evidence_class": (
                    evidence_class
                ),
                "A3_section_specific_primary_signal": (
                    primary_signal
                ),
                "A4_geometry_robust_effect": (
                    geometry_robust
                ),
                "A4_geometry_robust_primary_signal": (
                    robust_primary
                ),
                "direction": primary[
                    "direction"
                ],
                "V1_minus_V2": as_float(
                    primary["V1_minus_V2"],
                    "V1-minus-V2",
                ),
                "A3_Hedges_g": as_float(
                    primary["Hedges_g"],
                    "A3 Hedges g",
                ),
                "permutation_p_two_sided": (
                    raw_p
                ),
                "within_depth_maxT_p": (
                    within_max_t
                ),
                "global_BH_FDR": global_bh,
                "global_Holm_FWER": (
                    global_holm
                ),
                "geometry_candidates": int(
                    robust[
                        "geometry_candidates"
                    ]
                ),
                "unique_grid_widths": int(
                    robust[
                        "unique_grid_widths"
                    ]
                ),
                "direction_concordance_fraction": (
                    as_float(
                        robust[
                            "direction_concordance_fraction"
                        ],
                        "direction concordance",
                    )
                ),
                "width_median_direction_concordance": (
                    as_float(
                        robust[
                            "width_median_direction_concordance"
                        ],
                        "width direction concordance",
                    )
                ),
                "geometry_median_V1_minus_V2": (
                    as_float(
                        robust[
                            "median_V1_minus_V2"
                        ],
                        "geometry median contrast",
                    )
                ),
                "geometry_median_Hedges_g": (
                    as_float(
                        robust[
                            "median_Hedges_g"
                        ],
                        "geometry median Hedges g",
                    )
                ),
                "top_positive_snRNAseq_identities": (
                    identity[
                        "top_positive_identities"
                    ]
                ),
                "top_negative_snRNAseq_identities": (
                    identity[
                        "top_negative_identities"
                    ]
                ),
                "snRNAseq_frontal_minus_occipital_descriptive": (
                    as_float(
                        region[
                            "frontal_minus_occipital"
                        ],
                        "snRNA descriptive delta",
                    )
                ),
                "snRNAseq_region_comparison_role": (
                    "descriptive_context_only_"
                    "different_contrast_and_donor_confounded"
                ),
                "single_Visium_section": True,
                "independent_biological_replication": (
                    False
                ),
                "population_level_inference": (
                    False
                ),
            }
        )

evidence_rows.sort(
    key=lambda row: (
        {
            "S1_geometry_robust_section_specific_corrected": 1,
            (
                "S2_section_specific_corrected_"
                "not_geometry_robust"
            ): 2,
            "S3_global_FDR_only": 3,
            "S4_nominal_section_specific": 4,
            "S5_descriptive_no_spatial_signal": 5,
        }[
            row[
                "spatial_evidence_class"
            ]
        ],
        float(
            row["global_BH_FDR"]
        ),
        -abs(
            float(
                row[
                    "geometry_median_Hedges_g"
                ]
            )
        ),
        depth_order.index(
            row["depth_stratum"]
        ),
        row["module"],
    )
)

evidence_columns = [
    "depth_stratum",
    "module",
    "locked_genes",
    "spatial_evidence_class",
    "A3_section_specific_primary_signal",
    "A4_geometry_robust_effect",
    "A4_geometry_robust_primary_signal",
    "direction",
    "V1_minus_V2",
    "A3_Hedges_g",
    "permutation_p_two_sided",
    "within_depth_maxT_p",
    "global_BH_FDR",
    "global_Holm_FWER",
    "geometry_candidates",
    "unique_grid_widths",
    "direction_concordance_fraction",
    "width_median_direction_concordance",
    "geometry_median_V1_minus_V2",
    "geometry_median_Hedges_g",
    "top_positive_snRNAseq_identities",
    "top_negative_snRNAseq_identities",
    "snRNAseq_frontal_minus_occipital_descriptive",
    "snRNAseq_region_comparison_role",
    "single_Visium_section",
    "independent_biological_replication",
    "population_level_inference",
]

write_tsv(
    evidence_dir
    / "phase10B4_A5_all_spatial_evidence.tsv",
    evidence_rows,
    evidence_columns,
)

robust_signal_rows = [
    row
    for row in evidence_rows
    if row[
        "A4_geometry_robust_primary_signal"
    ]
]

write_tsv(
    evidence_dir
    / "phase10B4_A5_geometry_robust_spatial_signals.tsv",
    robust_signal_rows,
    evidence_columns,
)

module_synthesis_rows: list[
    dict[str, Any]
] = []

for module in modules:

    module_evidence = [
        row
        for row in evidence_rows
        if row["module"] == module
    ]

    robust_rows = [
        row
        for row in module_evidence
        if row[
            "A4_geometry_robust_primary_signal"
        ]
    ]

    robust_depths = [
        depth
        for depth in depth_order
        if any(
            row["depth_stratum"] == depth
            for row in robust_rows
        )
    ]

    robust_directions = sorted(
        {
            row["direction"]
            for row in robust_rows
        }
    )

    minimum_bh = min(
        float(
            row["global_BH_FDR"]
        )
        for row in module_evidence
    )

    maximum_absolute_geometry_g = max(
        abs(
            float(
                row[
                    "geometry_median_Hedges_g"
                ]
            )
        )
        for row in module_evidence
    )

    if len(robust_depths) >= 2:
        module_priority = (
            "high_section_specific_spatial_priority"
        )

    elif len(robust_depths) == 1:
        module_priority = (
            "moderate_section_specific_spatial_priority"
        )

    elif minimum_bh < 0.05:
        module_priority = (
            "corrected_spatial_signal_not_primary_robust"
        )

    else:
        module_priority = (
            "descriptive_spatial_context_only"
        )

    identity = identity_context_lookup[
        module
    ]

    region = region_lookup[module]

    module_synthesis_rows.append(
        {
            "module": module,
            "locked_genes": (
                locked_gene_counts[
                    module
                ]
            ),
            "spatial_tests": len(
                module_evidence
            ),
            "geometry_robust_signal_count": len(
                robust_rows
            ),
            "geometry_robust_depths": (
                ";".join(
                    robust_depths
                )
            ),
            "geometry_robust_directions": (
                ";".join(
                    robust_directions
                )
            ),
            "minimum_global_BH_FDR": (
                minimum_bh
            ),
            "maximum_absolute_geometry_median_Hedges_g": (
                maximum_absolute_geometry_g
            ),
            "module_spatial_priority": (
                module_priority
            ),
            "highest_scoring_snRNAseq_identity": (
                identity[
                    "highest_identity"
                ]
            ),
            "highest_identity_mean_score": (
                identity[
                    "highest_identity_mean_score"
                ]
            ),
            "lowest_scoring_snRNAseq_identity": (
                identity[
                    "lowest_identity"
                ]
            ),
            "lowest_identity_mean_score": (
                identity[
                    "lowest_identity_mean_score"
                ]
            ),
            "snRNAseq_frontal_minus_occipital_descriptive": (
                as_float(
                    region[
                        "frontal_minus_occipital"
                    ],
                    "module region delta",
                )
            ),
            "cross_modal_interpretation": (
                "Visium_section_specific_spatial_evidence_"
                "with_snRNAseq_cellular_context_not_replication"
            ),
        }
    )

module_synthesis_rows.sort(
    key=lambda row: (
        -int(
            row[
                "geometry_robust_signal_count"
            ]
        ),
        float(
            row["minimum_global_BH_FDR"]
        ),
        -float(
            row[
                "maximum_absolute_geometry_median_Hedges_g"
            ]
        ),
        row["module"],
    )
)

write_tsv(
    module_dir
    / "phase10B4_A5_module_level_cross_modal_synthesis.tsv",
    module_synthesis_rows,
    [
        "module",
        "locked_genes",
        "spatial_tests",
        "geometry_robust_signal_count",
        "geometry_robust_depths",
        "geometry_robust_directions",
        "minimum_global_BH_FDR",
        "maximum_absolute_geometry_median_Hedges_g",
        "module_spatial_priority",
        "highest_scoring_snRNAseq_identity",
        "highest_identity_mean_score",
        "lowest_scoring_snRNAseq_identity",
        "lowest_identity_mean_score",
        "snRNAseq_frontal_minus_occipital_descriptive",
        "cross_modal_interpretation",
    ],
)

evidence_class_counts: dict[
    str,
    int,
] = defaultdict(int)

for row in evidence_rows:
    evidence_class_counts[
        row["spatial_evidence_class"]
    ] += 1

evidence_class_rows = [
    {
        "spatial_evidence_class": key,
        "tests": value,
    }
    for key, value in sorted(
        evidence_class_counts.items()
    )
]

write_tsv(
    evidence_dir
    / "phase10B4_A5_spatial_evidence_class_counts.tsv",
    evidence_class_rows,
    [
        "spatial_evidence_class",
        "tests",
    ],
)

robust_depths = sorted(
    {
        row["depth_stratum"]
        for row in robust_signal_rows
    },
    key=depth_order.index,
)

robust_modules = sorted(
    {
        row["module"]
        for row in robust_signal_rows
    }
)

all_robust_v2_higher = all(
    row["direction"] == "V2_higher"
    for row in robust_signal_rows
)

status_value = (
    "passed_phase10B4_A5_conservative_cross_modal_"
    "section_specific_synthesis_ready_for_"
    "manuscript_figures_and_multiage_MERFISH_panel_audit"
    if (
        len(evidence_rows) == 36
        and len(robust_signal_rows) == 17
        and len(module_synthesis_rows) == 9
        and len(identity_context_rows) == 9
        and all_robust_v2_higher
    )
    else (
        "phase10B4_A5_requires_manual_review"
    )
)

status = {
    "phase": "phase10B4_A5",
    "locked_modules": len(
        modules
    ),
    "spatial_tests_synthesised": len(
        evidence_rows
    ),
    "geometry_robust_spatial_signals": len(
        robust_signal_rows
    ),
    "modules_with_geometry_robust_signals": len(
        robust_modules
    ),
    "geometry_robust_depth_strata": (
        ";".join(
            robust_depths
        )
    ),
    "all_geometry_robust_signals_V2_higher": (
        all_robust_v2_higher
    ),
    "snRNAseq_identity_context_modules": len(
        identity_context_rows
    ),
    "snRNAseq_region_contrast_role": (
        "descriptive_context_only_donor_confounded"
    ),
    "new_inferential_tests_performed": (
        False
    ),
    "direct_cross_modal_replication_claimed": (
        False
    ),
    "independent_biological_replication": (
        False
    ),
    "population_level_inference": (
        False
    ),
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B4_A5_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B4_A5_status.tsv",
    [status],
    list(status.keys()),
)

report = [
    "===== PHASE 10B4-A5 CONSERVATIVE "
    "CROSS-MODAL SYNTHESIS =====",
    "",
    (
        "Locked modules: "
        f"{len(modules)}"
    ),
    (
        "Spatial tests synthesised: "
        f"{len(evidence_rows)}"
    ),
    (
        "Geometry-robust spatial signals: "
        f"{len(robust_signal_rows)}"
    ),
    (
        "Modules with robust spatial signals: "
        f"{len(robust_modules)}"
    ),
    (
        "Robust depth strata: "
        f"{';'.join(robust_depths)}"
    ),
    (
        "All geometry-robust signals V2-higher: "
        f"{all_robust_v2_higher}"
    ),
    (
        "snRNA-seq identity-context modules: "
        f"{len(identity_context_rows)}"
    ),
    "",
    (
        "snRNA-seq regional contrast role: "
        "descriptive context only; donor-confounded"
    ),
    "New inferential tests performed: FALSE",
    "Direct cross-modal replication claimed: FALSE",
    "Independent biological replication: FALSE",
    "Population-level inference: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B4-A5 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B4_A5_report.txt"
).write_text(
    "\n".join(report)
    + "\n",
    encoding="utf-8",
)

print(
    "\n".join(report)
)

print(
    "\n===== MODULE-LEVEL SYNTHESIS ====="
)

module_columns = [
    "module",
    "locked_genes",
    "geometry_robust_signal_count",
    "geometry_robust_depths",
    "geometry_robust_directions",
    "minimum_global_BH_FDR",
    "maximum_absolute_geometry_median_Hedges_g",
    "module_spatial_priority",
    "highest_scoring_snRNAseq_identity",
    "lowest_scoring_snRNAseq_identity",
]

print(
    "\t".join(
        module_columns
    )
)

for row in module_synthesis_rows:

    print(
        "\t".join(
            str(row[column])
            for column in module_columns
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
        != "phase10B4_A5_SHA256.tsv"
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
    / "phase10B4_A5_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)
