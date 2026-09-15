from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


project = Path(sys.argv[1])
sample_map_path = Path(sys.argv[2])
eligibility_path = Path(sys.argv[3])
out = Path(sys.argv[4])

primary_dir = (
    out
    / "01_primary_fetal_design"
)

sensitivity_dir = (
    out
    / "02_section_sensitivity_design"
)

claim_dir = (
    out
    / "03_module_claim_boundaries"
)

exclusion_dir = (
    out
    / "04_exclusions"
)

audit_dir = (
    out
    / "05_audit"
)

for directory in (
    primary_dir,
    sensitivity_dir,
    claim_dir,
    exclusion_dir,
    audit_dir,
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


def as_bool(
    value: Any,
) -> bool:

    return str(value).strip().upper() in {
        "TRUE",
        "T",
        "1",
        "YES",
    }


sample_rows, sample_columns = read_tsv(
    sample_map_path
)

required_sample_columns = {
    "archive_name",
    "age_type",
    "age_value",
    "gestational_week",
    "primary_cortical_area",
    "measured_panel_class",
    "mapping_status",
}

missing = sorted(
    required_sample_columns
    - set(
        sample_columns
    )
)

if missing:
    raise SystemExit(
        "FAIL: sample map lacks columns: "
        + ";".join(
            missing
        )
    )

if len(sample_rows) != 45:
    raise SystemExit(
        f"FAIL: expected 45 sample rows, "
        f"observed {len(sample_rows)}."
    )

sample_lookup = {
    row["archive_name"]: row
    for row in sample_rows
}

eligibility_rows, eligibility_columns = read_tsv(
    eligibility_path
)

required_eligibility_columns = {
    "archive_name",
    "corrected_panel_class",
    "module",
    "matched_module_genes",
    "module_coverage_fraction",
    "primary_scoring_eligible",
    "sensitivity_scoring_eligible",
    "matched_genes",
}

missing = sorted(
    required_eligibility_columns
    - set(
        eligibility_columns
    )
)

if missing:
    raise SystemExit(
        "FAIL: eligibility table lacks columns: "
        + ";".join(
            missing
        )
    )

if len(eligibility_rows) != 405:
    raise SystemExit(
        f"FAIL: expected 405 eligibility rows, "
        f"observed {len(eligibility_rows)}."
    )

eligibility_lookup: dict[
    str,
    list[dict[str, str]],
] = defaultdict(list)

for row in eligibility_rows:

    eligibility_lookup[
        row["archive_name"]
    ].append(
        row
    )


primary_selection = [
    {
        "archive_name": "UMB1367_O1.zip",
        "selection_role": (
            "GW15_independent_occipital_donor_anchor"
        ),
    },
    {
        "archive_name": "UMB1117_O1.zip",
        "selection_role": (
            "GW15_independent_occipital_donor_anchor"
        ),
    },
    {
        "archive_name": "UMB1759_O1.zip",
        "selection_role": (
            "GW18_expanded_panel_occipital_anchor"
        ),
    },
    {
        "archive_name": "FB080_O1c.zip",
        "selection_role": (
            "GW20_same_donor_cross_technology_anchor"
        ),
    },
    {
        "archive_name": "FB121_O1.zip",
        "selection_role": (
            "GW20_independent_occipital_donor_anchor"
        ),
    },
    {
        "archive_name": "UMB1031_O1.zip",
        "selection_role": (
            "GW20_expanded_panel_independent_donor_anchor"
        ),
    },
    {
        "archive_name": "FB123_O1.zip",
        "selection_role": (
            "GW22_independent_occipital_donor_anchor"
        ),
    },
    {
        "archive_name": "UMB5900_BA17.zip",
        "selection_role": (
            "GW34_independent_occipital_donor_anchor"
        ),
    },
]


section_sensitivity_selection = [
    {
        "archive_name": "FB080_O1a.zip",
        "selection_role": (
            "GW20_FB080_within_donor_section_sensitivity"
        ),
    },
    {
        "archive_name": "FB080_O1b.zip",
        "selection_role": (
            "GW20_FB080_within_donor_section_sensitivity"
        ),
    },
    {
        "archive_name": "FB080_O1d.zip",
        "selection_role": (
            "GW20_FB080_within_donor_section_sensitivity"
        ),
    },
    {
        "archive_name": "FB123_O2.zip",
        "selection_role": (
            "GW22_FB123_within_donor_section_sensitivity"
        ),
    },
    {
        "archive_name": "UMB5900_BA18.zip",
        "selection_role": (
            "GW34_UMB5900_within_donor_area_sensitivity"
        ),
    },
]


def infer_donor(
    archive_name: str,
) -> str:

    return Path(
        archive_name
    ).stem.split(
        "_",
        1,
    )[0]


def build_design_rows(
    selections: list[dict[str, str]],
    design_class: str,
) -> list[dict[str, Any]]:

    rows: list[
        dict[str, Any]
    ] = []

    for order, selection in enumerate(
        selections,
        start=1,
    ):

        archive_name = selection[
            "archive_name"
        ]

        if archive_name not in sample_lookup:
            raise SystemExit(
                f"FAIL: selected archive is absent "
                f"from sample map: {archive_name}"
            )

        sample = sample_lookup[
            archive_name
        ]

        if sample[
            "age_type"
        ] != "fetal":
            raise SystemExit(
                f"FAIL: non-fetal archive selected: "
                f"{archive_name}"
            )

        if sample[
            "mapping_status"
        ] == (
            "not_resolved_in_public_experiment_table"
        ):
            raise SystemExit(
                f"FAIL: unresolved sample selected: "
                f"{archive_name}"
            )

        module_rows = eligibility_lookup[
            archive_name
        ]

        if len(module_rows) != 9:
            raise SystemExit(
                f"FAIL: expected nine module rows for "
                f"{archive_name}; observed "
                f"{len(module_rows)}."
            )

        primary_modules = sorted(
            row["module"]
            for row in module_rows
            if as_bool(
                row[
                    "primary_scoring_eligible"
                ]
            )
        )

        sensitivity_modules = sorted(
            row["module"]
            for row in module_rows
            if as_bool(
                row[
                    "sensitivity_scoring_eligible"
                ]
            )
        )

        rows.append(
            {
                "design_order": order,
                "design_class": design_class,
                "archive_name": archive_name,
                "age_type": sample[
                    "age_type"
                ],
                "donor_id": infer_donor(
                    archive_name
                ),
                "age_value": sample[
                    "age_value"
                ],
                "gestational_week": int(
                    sample[
                        "gestational_week"
                    ]
                ),
                "primary_cortical_area": (
                    sample[
                        "primary_cortical_area"
                    ]
                ),
                "panel_class": (
                    sample[
                        "measured_panel_class"
                    ]
                ),
                "selection_role": (
                    selection[
                        "selection_role"
                    ]
                ),
                "primary_eligible_module_count": len(
                    primary_modules
                ),
                "primary_eligible_modules": ";".join(
                    primary_modules
                ),
                "sensitivity_eligible_module_count": len(
                    sensitivity_modules
                ),
                "sensitivity_eligible_modules": ";".join(
                    sensitivity_modules
                ),
                "expression_payload_downloaded": (
                    False
                ),
            }
        )

    return rows


primary_rows = build_design_rows(
    primary_selection,
    "primary_independent_fetal_occipital",
)

sensitivity_rows = build_design_rows(
    section_sensitivity_selection,
    "within_donor_section_sensitivity",
)

primary_columns = [
    "design_order",
    "design_class",
    "archive_name",
    "age_type",
    "donor_id",
    "age_value",
    "gestational_week",
    "primary_cortical_area",
    "panel_class",
    "selection_role",
    "primary_eligible_module_count",
    "primary_eligible_modules",
    "sensitivity_eligible_module_count",
    "sensitivity_eligible_modules",
    "expression_payload_downloaded",
]

write_tsv(
    primary_dir
    / "phase10B5_P3_R2_primary_fetal_occipital_design.tsv",
    primary_rows,
    primary_columns,
)

write_tsv(
    sensitivity_dir
    / "phase10B5_P3_R2_within_donor_section_sensitivity_design.tsv",
    sensitivity_rows,
    primary_columns,
)

primary_archives = {
    row[
        "archive_name"
    ]
    for row in primary_rows
}

sensitivity_archives = {
    row[
        "archive_name"
    ]
    for row in sensitivity_rows
}

if (
    primary_archives
    & sensitivity_archives
):
    raise SystemExit(
        "FAIL: primary and sensitivity archive "
        "sets overlap."
    )

primary_module_rows: dict[
    str,
    list[dict[str, str]],
] = defaultdict(list)

for archive_name in primary_archives:

    for row in eligibility_lookup[
        archive_name
    ]:

        primary_module_rows[
            row["module"]
        ].append(
            row
        )

if len(
    primary_module_rows
) != 9:
    raise SystemExit(
        "FAIL: expected nine modules in primary design."
    )

claim_rows: list[
    dict[str, Any]
] = []

for module in sorted(
    primary_module_rows
):

    rows = primary_module_rows[
        module
    ]

    primary_count = sum(
        as_bool(
            row[
                "primary_scoring_eligible"
            ]
        )
        for row in rows
    )

    sensitivity_count = sum(
        as_bool(
            row[
                "sensitivity_scoring_eligible"
            ]
        )
        for row in rows
    )

    panel_primary_counts = Counter(
        row[
            "corrected_panel_class"
        ]
        for row in rows
        if as_bool(
            row[
                "primary_scoring_eligible"
            ]
        )
    )

    if primary_count == len(
        primary_rows
    ):

        claim_class = (
            "broad_primary_multiage_multidonor"
        )

    elif sensitivity_count == len(
        primary_rows
    ):

        claim_class = (
            "broad_sensitivity_multiage_multidonor"
        )

    elif (
        primary_count == 2
        and panel_primary_counts.get(
            "960_gene_panel",
            0,
        ) == 2
    ):

        claim_class = (
            "expanded_panel_only_two_age_validation"
        )

    else:

        claim_class = (
            "insufficient_for_prespecified_validation"
        )

    matched_counts = sorted(
        {
            int(
                row[
                    "matched_module_genes"
                ]
            )
            for row in rows
        }
    )

    fractions = sorted(
        {
            float(
                row[
                    "module_coverage_fraction"
                ]
            )
            for row in rows
        }
    )

    claim_rows.append(
        {
            "module": module,
            "primary_archives": len(
                primary_rows
            ),
            "primary_eligible_archives": (
                primary_count
            ),
            "sensitivity_eligible_archives": (
                sensitivity_count
            ),
            "300_panel_primary_archives": (
                panel_primary_counts.get(
                    "300_gene_panel",
                    0,
                )
            ),
            "960_panel_primary_archives": (
                panel_primary_counts.get(
                    "960_gene_panel",
                    0,
                )
            ),
            "matched_gene_counts": ";".join(
                str(value)
                for value in matched_counts
            ),
            "coverage_fractions": ";".join(
                f"{value:.6g}"
                for value in fractions
            ),
            "prespecified_claim_class": (
                claim_class
            ),
        }
    )

write_tsv(
    claim_dir
    / "phase10B5_P3_R2_module_claim_boundaries.tsv",
    claim_rows,
    [
        "module",
        "primary_archives",
        "primary_eligible_archives",
        "sensitivity_eligible_archives",
        "300_panel_primary_archives",
        "960_panel_primary_archives",
        "matched_gene_counts",
        "coverage_fractions",
        "prespecified_claim_class",
    ],
)

selected_archives = (
    primary_archives
    | sensitivity_archives
)

exclusion_rows: list[
    dict[str, Any]
] = []

for sample in sample_rows:

    archive_name = sample[
        "archive_name"
    ]

    if archive_name in selected_archives:
        continue

    if sample[
        "mapping_status"
    ] == (
        "not_resolved_in_public_experiment_table"
    ):

        exclusion_reason = (
            "unresolved_sample_identity"
        )

    elif sample[
        "age_type"
    ] != "fetal":

        exclusion_reason = (
            "adult_not_fetal_replication"
        )

    elif sample[
        "primary_cortical_area"
    ] != "Occi":

        exclusion_reason = (
            "non_occipital_area_not_in_primary_temporal_design"
        )

    else:

        exclusion_reason = (
            "additional_same_donor_or_same_age_section_"
            "held_to_avoid_pseudoreplication"
        )

    exclusion_rows.append(
        {
            "archive_name": archive_name,
            "age_type": sample[
                "age_type"
            ],
            "age_value": sample[
                "age_value"
            ],
            "primary_cortical_area": (
                sample[
                    "primary_cortical_area"
                ]
            ),
            "panel_class": (
                sample[
                    "measured_panel_class"
                ]
            ),
            "exclusion_reason": (
                exclusion_reason
            ),
            "expression_payload_downloaded": (
                False
            ),
        }
    )

write_tsv(
    exclusion_dir
    / "phase10B5_P3_R2_nonselected_archive_register.tsv",
    exclusion_rows,
    [
        "archive_name",
        "age_type",
        "age_value",
        "primary_cortical_area",
        "panel_class",
        "exclusion_reason",
        "expression_payload_downloaded",
    ],
)

primary_donors = {
    row["donor_id"]
    for row in primary_rows
}

primary_ages = {
    row["gestational_week"]
    for row in primary_rows
}

primary_panel_counts = Counter(
    row["panel_class"]
    for row in primary_rows
)

claim_counts = Counter(
    row[
        "prespecified_claim_class"
    ]
    for row in claim_rows
)

technical_pass = (
    len(primary_rows) == 8
    and len(primary_donors) == 8
    and primary_ages
    == {
        15,
        18,
        20,
        22,
        34,
    }
    and primary_panel_counts.get(
        "300_gene_panel",
        0,
    ) == 6
    and primary_panel_counts.get(
        "960_gene_panel",
        0,
    ) == 2
    and len(sensitivity_rows) == 5
    and claim_counts.get(
        "broad_primary_multiage_multidonor",
        0,
    ) == 4
    and claim_counts.get(
        "broad_sensitivity_multiage_multidonor",
        0,
    ) == 1
    and claim_counts.get(
        "expanded_panel_only_two_age_validation",
        0,
    ) == 4
)

status_value = (
    "passed_phase10B5_P3_R2_donor_balanced_"
    "occipital_fetal_validation_design_ready_for_"
    "targeted_expression_streaming"
    if technical_pass
    else (
        "phase10B5_P3_R2_requires_manual_review"
    )
)

status = {
    "phase": "phase10B5_P3_R2",
    "primary_fetal_archives": len(
        primary_rows
    ),
    "primary_independent_donors": len(
        primary_donors
    ),
    "primary_gestational_ages": ";".join(
        str(value)
        for value in sorted(
            primary_ages
        )
    ),
    "primary_300_panel_archives": (
        primary_panel_counts.get(
            "300_gene_panel",
            0,
        )
    ),
    "primary_960_panel_archives": (
        primary_panel_counts.get(
            "960_gene_panel",
            0,
        )
    ),
    "within_donor_section_sensitivity_archives": len(
        sensitivity_rows
    ),
    "broad_primary_modules": (
        claim_counts.get(
            "broad_primary_multiage_multidonor",
            0,
        )
    ),
    "broad_sensitivity_modules": (
        claim_counts.get(
            "broad_sensitivity_multiage_multidonor",
            0,
        )
    ),
    "expanded_panel_only_modules": (
        claim_counts.get(
            "expanded_panel_only_two_age_validation",
            0,
        )
    ),
    "adult_archives_selected": 0,
    "unresolved_archives_selected": 0,
    "new_inferential_tests_performed": False,
    "network_requests_performed": False,
    "expression_payload_downloaded": False,
    "expression_values_accessed": False,
    "module_scores_computed": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P3_R2_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B5_P3_R2_status.tsv",
    [status],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P3-R2 DONOR-BALANCED "
    "FETAL VALIDATION DESIGN =====",
    "",
    (
        "Primary fetal archives: "
        f"{len(primary_rows)}"
    ),
    (
        "Primary independent donors: "
        f"{len(primary_donors)}"
    ),
    (
        "Primary gestational ages: "
        + ";".join(
            str(value)
            for value in sorted(
                primary_ages
            )
        )
    ),
    (
        "Primary 300-panel archives: "
        f"{primary_panel_counts.get('300_gene_panel', 0)}"
    ),
    (
        "Primary 960-panel archives: "
        f"{primary_panel_counts.get('960_gene_panel', 0)}"
    ),
    (
        "Within-donor section-sensitivity archives: "
        f"{len(sensitivity_rows)}"
    ),
    (
        "Broad primary modules: "
        f"{claim_counts.get('broad_primary_multiage_multidonor', 0)}"
    ),
    (
        "Broad sensitivity modules: "
        f"{claim_counts.get('broad_sensitivity_multiage_multidonor', 0)}"
    ),
    (
        "Expanded-panel-only modules: "
        f"{claim_counts.get('expanded_panel_only_two_age_validation', 0)}"
    ),
    "",
    "Adult archives selected: 0",
    "Unresolved archives selected: 0",
    "New inferential tests performed: FALSE",
    "Network requests performed: FALSE",
    "Expression payload downloaded: FALSE",
    "Expression values accessed: FALSE",
    "Module scores computed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P3-R2 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P3_R2_report.txt"
).write_text(
    "\n".join(report)
    + "\n",
    encoding="utf-8",
)

print(
    "\n".join(report)
)

print(
    "\n===== PRIMARY FETAL DESIGN ====="
)

for row in primary_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "design_order",
                "archive_name",
                "donor_id",
                "gestational_week",
                "primary_cortical_area",
                "panel_class",
                "primary_eligible_module_count",
                "sensitivity_eligible_module_count",
            )
        )
    )

print(
    "\n===== MODULE CLAIM BOUNDARIES ====="
)

for row in claim_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "module",
                "primary_eligible_archives",
                "sensitivity_eligible_archives",
                "300_panel_primary_archives",
                "960_panel_primary_archives",
                "prespecified_claim_class",
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
        != "phase10B5_P3_R2_SHA256.tsv"
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
    / "phase10B5_P3_R2_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)
