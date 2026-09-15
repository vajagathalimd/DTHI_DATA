from __future__ import annotations

import csv
import gzip
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


project = Path(sys.argv[1])
matrix_audit_path = Path(sys.argv[2])
panel_gene_path = Path(sys.argv[3])
coverage_path = Path(sys.argv[4])
out = Path(sys.argv[5])

panel_dir = (
    out
    / "01_corrected_panel_lock"
)

mapping_dir = (
    out
    / "02_external_sample_mapping"
)

eligibility_dir = (
    out
    / "03_module_scoring_eligibility"
)

design_dir = (
    out
    / "04_targeted_validation_design"
)

exception_dir = (
    out
    / "05_exceptions"
)

audit_dir = (
    out
    / "06_audit"
)

for directory in (
    panel_dir,
    mapping_dir,
    eligibility_dir,
    design_dir,
    exception_dir,
    audit_dir,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

primary_minimum_genes = 5
primary_minimum_fraction = 0.20

sensitivity_minimum_genes = 4
sensitivity_minimum_fraction = 0.15


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


def write_tsv_gz(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:

    with gzip.open(
        path,
        "wt",
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


def normalize(
    value: str,
) -> str:

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        value.strip().lower(),
    ).strip(
        "_"
    )


def is_blank_barcode(
    value: str,
) -> bool:

    normalized = normalize(
        value
    )

    patterns = (
        r"^blank(?:_|$)",
        r"^blankbarcode(?:_|$)",
        r"^negative_control(?:_|$)",
        r"^neg_control(?:_|$)",
        r"^no_target(?:_|$)",
        r"^notarget(?:_|$)",
        r"^unused(?:_|$)",
    )

    return any(
        re.search(
            pattern,
            normalized,
        )
        is not None
        for pattern in patterns
    )


matrix_rows, matrix_columns = read_tsv(
    matrix_audit_path
)

if len(matrix_rows) != 45:
    raise SystemExit(
        f"FAIL: expected 45 matrix-audit rows, "
        f"observed {len(matrix_rows)}."
    )

matrix_lookup = {
    row["archive_name"]: row
    for row in matrix_rows
}

with gzip.open(
    panel_gene_path,
    "rt",
    encoding="utf-8-sig",
    newline="",
) as handle:

    reader = csv.DictReader(
        handle,
        delimiter="\t",
    )

    panel_gene_rows = list(reader)

panel_by_archive: dict[
    str,
    list[dict[str, str]],
] = defaultdict(list)

for row in panel_gene_rows:
    panel_by_archive[
        row["archive_name"]
    ].append(
        row
    )

if set(panel_by_archive) != set(
    matrix_lookup
):
    raise SystemExit(
        "FAIL: panel-gene archives do not match "
        "matrix-audit archives."
    )

corrected_panel_rows: list[
    dict[str, Any]
] = []

corrected_panel_genes: list[
    dict[str, Any]
] = []

panel_exception_rows: list[
    dict[str, Any]
] = []

for archive_name in sorted(
    panel_by_archive
):

    rows = sorted(
        panel_by_archive[
            archive_name
        ],
        key=lambda row: int(
            row["gene_order"]
        ),
    )

    raw_fields = [
        row["gene_symbol"]
        for row in rows
    ]

    blank_fields = [
        field
        for field in raw_fields
        if is_blank_barcode(
            field
        )
    ]

    biological_fields = [
        field
        for field in raw_fields
        if not is_blank_barcode(
            field
        )
    ]

    raw_count = len(
        raw_fields
    )

    blank_count = len(
        blank_fields
    )

    biological_count = len(
        biological_fields
    )

    if (
        raw_count == 315
        and blank_count == 15
        and biological_count == 300
    ):
        corrected_panel = (
            "300_gene_panel"
        )
        panel_resolution = (
            "resolved_300_genes_plus_15_blanks"
        )

    elif (
        raw_count == 1000
        and blank_count == 40
        and biological_count == 960
    ):
        corrected_panel = (
            "960_gene_panel"
        )
        panel_resolution = (
            "resolved_960_genes_plus_40_blanks"
        )

    elif (
        raw_count == 300
        and blank_count == 0
        and biological_count == 300
    ):
        corrected_panel = (
            "300_gene_panel"
        )
        panel_resolution = (
            "resolved_300_gene_export_without_blank_columns"
        )

    else:
        corrected_panel = (
            "unresolved_panel"
        )
        panel_resolution = (
            "manual_review"
        )

        panel_exception_rows.append(
            {
                "archive_name": archive_name,
                "raw_panel_columns": (
                    raw_count
                ),
                "blank_barcode_columns": (
                    blank_count
                ),
                "biological_gene_columns": (
                    biological_count
                ),
                "exception": (
                    "panel_does_not_match_"
                    "300_plus_15_or_960_plus_40_design"
                ),
            }
        )

    matrix = matrix_lookup[
        archive_name
    ]

    corrected_panel_rows.append(
        {
            "archive_name": archive_name,
            "donor_id": (
                matrix["donor_id"]
            ),
            "section_token": (
                matrix["section_token"]
            ),
            "raw_panel_columns": (
                raw_count
            ),
            "blank_barcode_columns": (
                blank_count
            ),
            "biological_gene_columns": (
                biological_count
            ),
            "corrected_panel_class": (
                corrected_panel
            ),
            "panel_resolution": (
                panel_resolution
            ),
            "blank_barcode_names": (
                ";".join(
                    blank_fields
                )
            ),
            "duplicate_biological_gene_names": (
                biological_count
                - len(
                    {
                        gene.upper()
                        for gene
                        in biological_fields
                    }
                )
            ),
        }
    )

    biological_order = 0

    for row in rows:

        gene = row[
            "gene_symbol"
        ]

        blank = is_blank_barcode(
            gene
        )

        if not blank:
            biological_order += 1

        corrected_panel_genes.append(
            {
                "archive_name": (
                    archive_name
                ),
                "corrected_panel_class": (
                    corrected_panel
                ),
                "raw_column_order": int(
                    row[
                        "gene_order"
                    ]
                ),
                "raw_column_name": (
                    gene
                ),
                "blank_barcode": (
                    blank
                ),
                "biological_gene_order": (
                    biological_order
                    if not blank
                    else ""
                ),
                "normalized_gene_symbol": (
                    gene.upper()
                    if not blank
                    else ""
                ),
            }
        )

panel_columns = [
    "archive_name",
    "donor_id",
    "section_token",
    "raw_panel_columns",
    "blank_barcode_columns",
    "biological_gene_columns",
    "corrected_panel_class",
    "panel_resolution",
    "blank_barcode_names",
    "duplicate_biological_gene_names",
]

write_tsv(
    panel_dir
    / "phase10B5_P3_R1_corrected_archive_panel_lock.tsv",
    corrected_panel_rows,
    panel_columns,
)

write_tsv_gz(
    panel_dir
    / "phase10B5_P3_R1_corrected_panel_gene_inventory.tsv.gz",
    corrected_panel_genes,
    [
        "archive_name",
        "corrected_panel_class",
        "raw_column_order",
        "raw_column_name",
        "blank_barcode",
        "biological_gene_order",
        "normalized_gene_symbol",
    ],
)

write_tsv(
    exception_dir
    / "phase10B5_P3_R1_panel_resolution_exceptions.tsv",
    panel_exception_rows,
    [
        "archive_name",
        "raw_panel_columns",
        "blank_barcode_columns",
        "biological_gene_columns",
        "exception",
    ],
)

panel_lookup = {
    row["archive_name"]: row
    for row in corrected_panel_rows
}


def sample(
    archive: str,
    age_type: str,
    age_value: str,
    gestational_week: str,
    area: str,
    expected_panel: str,
    mapping_status: str = "official_external_mapping",
) -> dict[str, str]:

    return {
        "archive_name": archive,
        "age_type": age_type,
        "age_value": age_value,
        "gestational_week": (
            gestational_week
        ),
        "primary_cortical_area": area,
        "expected_panel_class": (
            expected_panel
        ),
        "mapping_status": (
            mapping_status
        ),
    }


sample_rows = [
    sample("UMB1367_P1.zip", "fetal", "GW15", "15", "Par", "300_gene_panel"),
    sample("UMB1367_O1.zip", "fetal", "GW15", "15", "Occi", "300_gene_panel"),

    sample("UMB1117_F1a.zip", "fetal", "GW15", "15", "PFC", "300_gene_panel"),
    sample("UMB1117_F1b.zip", "fetal", "GW15", "15", "PFC", "300_gene_panel"),
    sample("UMB1117_F2a.zip", "fetal", "GW15", "15", "PMC/M1", "300_gene_panel"),
    sample("UMB1117_F2b.zip", "fetal", "GW15", "15", "PMC/M1", "300_gene_panel"),
    sample("UMB1117_P1.zip", "fetal", "GW15", "15", "Par", "300_gene_panel"),
    sample("UMB1117_T1.zip", "fetal", "GW15", "15", "Temp", "300_gene_panel"),
    sample("UMB1117_O1.zip", "fetal", "GW15", "15", "Occi", "300_gene_panel"),

    sample("UMB1759_O1.zip", "fetal", "GW18", "18", "Occi", "960_gene_panel"),

    sample("FB080_F1.zip", "fetal", "GW20", "20", "PFC", "300_gene_panel"),
    sample("FB080_F2a.zip", "fetal", "GW20", "20", "PFC", "300_gene_panel"),
    sample("FB080_F2b.zip", "fetal", "GW20", "20", "PFC", "300_gene_panel"),
    sample("FB080_P1a.zip", "fetal", "GW20", "20", "Par", "300_gene_panel"),
    sample("FB080_P1b.zip", "fetal", "GW20", "20", "Par", "300_gene_panel"),
    sample("FB080_P2.zip", "fetal", "GW20", "20", "Par", "300_gene_panel"),
    sample("FB080_T1.zip", "fetal", "GW20", "20", "Temp", "300_gene_panel"),
    sample("FB080_O1a.zip", "fetal", "GW20", "20", "Occi", "300_gene_panel"),
    sample("FB080_O1b.zip", "fetal", "GW20", "20", "Occi", "300_gene_panel"),
    sample("FB080_O1c.zip", "fetal", "GW20", "20", "Occi", "300_gene_panel"),
    sample("FB080_O1d.zip", "fetal", "GW20", "20", "Occi", "300_gene_panel"),

    sample("FB121_F1.zip", "fetal", "GW20", "20", "PFC", "300_gene_panel"),
    sample("FB121_F2.zip", "fetal", "GW20", "20", "PMC/M1", "300_gene_panel"),
    sample("FB121_P1.zip", "fetal", "GW20", "20", "Par", "300_gene_panel"),
    sample("FB121_P2.zip", "fetal", "GW20", "20", "Par", "300_gene_panel"),
    sample("FB121_T1.zip", "fetal", "GW20", "20", "Temp", "300_gene_panel"),
    sample("FB121_O1.zip", "fetal", "GW20", "20", "Occi", "300_gene_panel"),

    sample("UMB1031_O1.zip", "fetal", "GW20", "20", "Occi", "960_gene_panel"),

    sample("FB123_F1.zip", "fetal", "GW22", "22", "PFC", "300_gene_panel"),
    sample("FB123_F2.zip", "fetal", "GW22", "22", "PFC", "300_gene_panel"),
    sample("FB123_F3.zip", "fetal", "GW22", "22", "PMC/M1", "300_gene_panel"),
    sample("FB123_P1.zip", "fetal", "GW22", "22", "Par", "300_gene_panel"),
    sample(
        "FB123_P1_2.zip",
        "fetal",
        "GW22",
        "22",
        "Par",
        "300_gene_panel",
        "inferred_same_donor_and_P1_section_replication",
    ),
    sample("FB123_O1.zip", "fetal", "GW22", "22", "Occi", "300_gene_panel"),
    sample("FB123_O2.zip", "fetal", "GW22", "22", "Occi", "300_gene_panel"),

    sample("UMB5900_BA9.zip", "fetal", "GW34", "34", "PFC", "300_gene_panel"),
    sample("UMB5900_BA4.zip", "fetal", "GW34", "34", "PMC/M1", "300_gene_panel"),
    sample("UMB5900_BA123.zip", "fetal", "GW34", "34", "Par", "300_gene_panel"),
    sample("UMB5900_BA40a.zip", "fetal", "GW34", "34", "Par", "300_gene_panel"),
    sample("UMB5900_BA40b.zip", "fetal", "GW34", "34", "Par", "300_gene_panel"),
    sample("UMB5900_BA22.zip", "fetal", "GW34", "34", "Temp", "300_gene_panel"),
    sample("UMB5900_BA18.zip", "fetal", "GW34", "34", "Occi", "300_gene_panel"),
    sample("UMB5900_BA17.zip", "fetal", "GW34", "34", "Occi", "300_gene_panel"),

    sample(
        "UMB5859_BA17.zip",
        "unresolved",
        "",
        "",
        "Occi_or_BA17_from_archive_name",
        "",
        "not_resolved_in_public_experiment_table",
    ),

    sample(
        "UMB5958_BA17.zip",
        "adult",
        "22_years",
        "",
        "Occi",
        "300_gene_panel",
    ),
]

if len(sample_rows) != 45:
    raise SystemExit(
        f"FAIL: external sample map contains "
        f"{len(sample_rows)} rows; expected 45."
    )

if {
    row["archive_name"]
    for row in sample_rows
} != set(
    panel_lookup
):
    missing_from_map = sorted(
        set(panel_lookup)
        - {
            row["archive_name"]
            for row in sample_rows
        }
    )

    extra_in_map = sorted(
        {
            row["archive_name"]
            for row in sample_rows
        }
        - set(panel_lookup)
    )

    raise SystemExit(
        "FAIL: sample-map/archive mismatch. "
        f"Missing={missing_from_map}; "
        f"extra={extra_in_map}"
    )

for row in sample_rows:

    measured_panel = panel_lookup[
        row["archive_name"]
    ][
        "corrected_panel_class"
    ]

    expected_panel = row[
        "expected_panel_class"
    ]

    row[
        "measured_panel_class"
    ] = measured_panel

    row[
        "panel_assignment_agreement"
    ] = (
        measured_panel == expected_panel
        if expected_panel
        else ""
    )

write_tsv(
    mapping_dir
    / "phase10B5_P3_R1_external_sample_age_area_panel_lock.tsv",
    sample_rows,
    [
        "archive_name",
        "age_type",
        "age_value",
        "gestational_week",
        "primary_cortical_area",
        "expected_panel_class",
        "measured_panel_class",
        "panel_assignment_agreement",
        "mapping_status",
    ],
)

coverage_rows, coverage_columns = read_tsv(
    coverage_path
)

if len(coverage_rows) != 405:
    raise SystemExit(
        f"FAIL: expected 405 module coverage rows, "
        f"observed {len(coverage_rows)}."
    )

eligibility_rows: list[
    dict[str, Any]
] = []

for row in coverage_rows:

    archive_name = row[
        "archive_name"
    ]

    corrected_panel = panel_lookup[
        archive_name
    ][
        "corrected_panel_class"
    ]

    matched = int(
        row[
            "matched_module_genes"
        ]
    )

    fraction = float(
        row[
            "module_coverage_fraction"
        ]
    )

    primary_eligible = (
        matched
        >= primary_minimum_genes
        and fraction
        >= primary_minimum_fraction
    )

    sensitivity_eligible = (
        matched
        >= sensitivity_minimum_genes
        and fraction
        >= sensitivity_minimum_fraction
    )

    eligibility_rows.append(
        {
            "archive_name": archive_name,
            "corrected_panel_class": (
                corrected_panel
            ),
            "module": row[
                "module"
            ],
            "locked_module_genes": int(
                row[
                    "locked_module_genes"
                ]
            ),
            "matched_module_genes": (
                matched
            ),
            "module_coverage_fraction": (
                fraction
            ),
            "primary_scoring_eligible": (
                primary_eligible
            ),
            "sensitivity_scoring_eligible": (
                sensitivity_eligible
            ),
            "primary_minimum_genes": (
                primary_minimum_genes
            ),
            "primary_minimum_fraction": (
                primary_minimum_fraction
            ),
            "sensitivity_minimum_genes": (
                sensitivity_minimum_genes
            ),
            "sensitivity_minimum_fraction": (
                sensitivity_minimum_fraction
            ),
            "matched_genes": row[
                "matched_genes"
            ],
        }
    )

eligibility_columns = [
    "archive_name",
    "corrected_panel_class",
    "module",
    "locked_module_genes",
    "matched_module_genes",
    "module_coverage_fraction",
    "primary_scoring_eligible",
    "sensitivity_scoring_eligible",
    "primary_minimum_genes",
    "primary_minimum_fraction",
    "sensitivity_minimum_genes",
    "sensitivity_minimum_fraction",
    "matched_genes",
]

write_tsv(
    eligibility_dir
    / "phase10B5_P3_R1_archive_module_scoring_eligibility.tsv",
    eligibility_rows,
    eligibility_columns,
)

summary_groups: dict[
    tuple[str, str],
    list[dict[str, Any]],
] = defaultdict(list)

for row in eligibility_rows:

    summary_groups[
        (
            row[
                "corrected_panel_class"
            ],
            row["module"],
        )
    ].append(
        row
    )

eligibility_summary_rows: list[
    dict[str, Any]
] = []

for (
    panel_class,
    module,
), rows in sorted(
    summary_groups.items()
):

    eligibility_summary_rows.append(
        {
            "corrected_panel_class": (
                panel_class
            ),
            "module": module,
            "archives": len(
                rows
            ),
            "matched_module_genes": (
                int(
                    rows[0][
                        "matched_module_genes"
                    ]
                )
            ),
            "module_coverage_fraction": (
                float(
                    rows[0][
                        "module_coverage_fraction"
                    ]
                )
            ),
            "primary_eligible_archives": sum(
                bool(
                    row[
                        "primary_scoring_eligible"
                    ]
                )
                for row in rows
            ),
            "sensitivity_eligible_archives": sum(
                bool(
                    row[
                        "sensitivity_scoring_eligible"
                    ]
                )
                for row in rows
            ),
        }
    )

write_tsv(
    eligibility_dir
    / "phase10B5_P3_R1_panel_module_eligibility_summary.tsv",
    eligibility_summary_rows,
    [
        "corrected_panel_class",
        "module",
        "archives",
        "matched_module_genes",
        "module_coverage_fraction",
        "primary_eligible_archives",
        "sensitivity_eligible_archives",
    ],
)

temporal_roles = {
    "UMB1117_O1.zip": (
        "prespecified_GW15_occipital_temporal_comparator"
    ),
    "UMB1759_O1.zip": (
        "prespecified_GW18_expanded_panel_occipital_comparator"
    ),
    "FB080_O1c.zip": (
        "prespecified_GW20_same_donor_cross_technology_section"
    ),
    "UMB1031_O1.zip": (
        "prespecified_GW20_independent_donor_expanded_panel"
    ),
    "UMB5900_BA17.zip": (
        "prespecified_GW34_occipital_temporal_comparator"
    ),
    "UMB5958_BA17.zip": (
        "adult_occipital_temporal_comparator_not_fetal_replication"
    ),
}

sample_lookup = {
    row["archive_name"]: row
    for row in sample_rows
}

eligibility_lookup: dict[
    str,
    list[dict[str, Any]],
] = defaultdict(list)

for row in eligibility_rows:
    eligibility_lookup[
        row["archive_name"]
    ].append(
        row
    )

target_rows: list[
    dict[str, Any]
] = []

for priority, archive_name in enumerate(
    temporal_roles,
    start=1,
):

    sample_row = sample_lookup[
        archive_name
    ]

    module_rows = eligibility_lookup[
        archive_name
    ]

    primary_modules = sorted(
        row["module"]
        for row in module_rows
        if row[
            "primary_scoring_eligible"
        ]
    )

    sensitivity_modules = sorted(
        row["module"]
        for row in module_rows
        if row[
            "sensitivity_scoring_eligible"
        ]
    )

    target_rows.append(
        {
            "download_priority": priority,
            "archive_name": archive_name,
            "validation_role": (
                temporal_roles[
                    archive_name
                ]
            ),
            "age_type": sample_row[
                "age_type"
            ],
            "age_value": sample_row[
                "age_value"
            ],
            "gestational_week": (
                sample_row[
                    "gestational_week"
                ]
            ),
            "primary_cortical_area": (
                sample_row[
                    "primary_cortical_area"
                ]
            ),
            "corrected_panel_class": (
                sample_row[
                    "measured_panel_class"
                ]
            ),
            "primary_eligible_module_count": len(
                primary_modules
            ),
            "primary_eligible_modules": (
                ";".join(
                    primary_modules
                )
            ),
            "sensitivity_eligible_module_count": len(
                sensitivity_modules
            ),
            "sensitivity_eligible_modules": (
                ";".join(
                    sensitivity_modules
                )
            ),
            "fetal_replication_candidate": (
                sample_row[
                    "age_type"
                ] == "fetal"
            ),
            "expression_payload_downloaded": (
                False
            ),
        }
    )

write_tsv(
    design_dir
    / "phase10B5_P3_R1_targeted_temporal_MERFISH_design.tsv",
    target_rows,
    [
        "download_priority",
        "archive_name",
        "validation_role",
        "age_type",
        "age_value",
        "gestational_week",
        "primary_cortical_area",
        "corrected_panel_class",
        "primary_eligible_module_count",
        "primary_eligible_modules",
        "sensitivity_eligible_module_count",
        "sensitivity_eligible_modules",
        "fetal_replication_candidate",
        "expression_payload_downloaded",
    ],
)

corrected_panel_counts = Counter(
    row[
        "corrected_panel_class"
    ]
    for row in corrected_panel_rows
)

resolved_sample_mappings = sum(
    row[
        "mapping_status"
    ]
    != (
        "not_resolved_in_public_experiment_table"
    )
    for row in sample_rows
)

unresolved_sample_mappings = (
    len(sample_rows)
    - resolved_sample_mappings
)

known_panel_agreements = [
    row[
        "panel_assignment_agreement"
    ]
    for row in sample_rows
    if row[
        "expected_panel_class"
    ]
]

panel_assignment_disagreements = sum(
    agreement is not True
    for agreement
    in known_panel_agreements
)

technical_pass = (
    len(
        corrected_panel_rows
    ) == 45
    and corrected_panel_counts.get(
        "300_gene_panel",
        0,
    ) == 42
    and corrected_panel_counts.get(
        "960_gene_panel",
        0,
    ) == 3
    and not panel_exception_rows
    and len(
        eligibility_rows
    ) == 405
    and resolved_sample_mappings == 44
    and unresolved_sample_mappings == 1
    and panel_assignment_disagreements == 0
)

status_value = (
    "passed_phase10B5_P3_R1_corrected_panel_blank_"
    "classification_and_external_sample_mapping_ready_for_"
    "targeted_MERFISH_expression_streaming_design"
    if technical_pass
    else (
        "phase10B5_P3_R1_requires_manual_review"
    )
)

status = {
    "phase": "phase10B5_P3_R1",
    "archives": len(
        corrected_panel_rows
    ),
    "corrected_300_gene_panel_archives": (
        corrected_panel_counts.get(
            "300_gene_panel",
            0,
        )
    ),
    "corrected_960_gene_panel_archives": (
        corrected_panel_counts.get(
            "960_gene_panel",
            0,
        )
    ),
    "panel_resolution_exceptions": len(
        panel_exception_rows
    ),
    "resolved_or_inferred_sample_mappings": (
        resolved_sample_mappings
    ),
    "unresolved_sample_mappings": (
        unresolved_sample_mappings
    ),
    "known_panel_assignment_disagreements": (
        panel_assignment_disagreements
    ),
    "archive_module_eligibility_rows": len(
        eligibility_rows
    ),
    "targeted_temporal_archives": len(
        target_rows
    ),
    "primary_minimum_module_genes": (
        primary_minimum_genes
    ),
    "primary_minimum_module_fraction": (
        primary_minimum_fraction
    ),
    "sensitivity_minimum_module_genes": (
        sensitivity_minimum_genes
    ),
    "sensitivity_minimum_module_fraction": (
        sensitivity_minimum_fraction
    ),
    "P3_numeric_area_values_promoted": (
        False
    ),
    "expression_payload_downloaded": (
        False
    ),
    "expression_values_accessed": (
        False
    ),
    "module_scores_computed": (
        False
    ),
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P3_R1_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B5_P3_R1_status.tsv",
    [
        status
    ],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P3-R1 CORRECTED PANEL "
    "AND EXTERNAL SAMPLE LOCK =====",
    "",
    (
        "Archives: "
        f"{len(corrected_panel_rows)}"
    ),
    (
        "Corrected 300-gene panel archives: "
        f"{corrected_panel_counts.get('300_gene_panel', 0)}"
    ),
    (
        "Corrected 960-gene panel archives: "
        f"{corrected_panel_counts.get('960_gene_panel', 0)}"
    ),
    (
        "Panel resolution exceptions: "
        f"{len(panel_exception_rows)}"
    ),
    (
        "Resolved or inferred sample mappings: "
        f"{resolved_sample_mappings}"
    ),
    (
        "Unresolved sample mappings: "
        f"{unresolved_sample_mappings}"
    ),
    (
        "Known panel assignment disagreements: "
        f"{panel_assignment_disagreements}"
    ),
    (
        "Archive-module eligibility rows: "
        f"{len(eligibility_rows)}"
    ),
    (
        "Targeted temporal archives: "
        f"{len(target_rows)}"
    ),
    "",
    (
        "Primary scoring threshold: "
        f">={primary_minimum_genes} genes and "
        f">={primary_minimum_fraction:.0%} coverage"
    ),
    (
        "Sensitivity threshold: "
        f">={sensitivity_minimum_genes} genes and "
        f">={sensitivity_minimum_fraction:.0%} coverage"
    ),
    "",
    "P3 numeric area values promoted: FALSE",
    "Expression payload downloaded: FALSE",
    "Expression values accessed: FALSE",
    "Module scores computed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P3-R1 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P3_R1_report.txt"
).write_text(
    "\n".join(report)
    + "\n",
    encoding="utf-8",
)

print(
    "\n".join(report)
)

print(
    "\n===== CORRECTED PANEL COUNTS ====="
)

for panel_class, count in sorted(
    corrected_panel_counts.items()
):
    print(
        f"{panel_class}\t{count}"
    )

print(
    "\n===== PANEL-MODULE ELIGIBILITY ====="
)

for row in eligibility_summary_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "corrected_panel_class",
                "module",
                "archives",
                "matched_module_genes",
                "module_coverage_fraction",
                "primary_eligible_archives",
                "sensitivity_eligible_archives",
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
        != "phase10B5_P3_R1_SHA256.tsv"
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
    / "phase10B5_P3_R1_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)
