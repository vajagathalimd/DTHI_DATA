from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


project = Path(sys.argv[1])
record_json_path = Path(sys.argv[2])
primary_path = Path(sys.argv[3])
sensitivity_path = Path(sys.argv[4])
raw_schema_path = Path(sys.argv[5])
out = Path(sys.argv[6])

record_dir = (
    out
    / "01_record_inventory"
)

mapping_dir = (
    out
    / "02_archive_object_mapping"
)

limitation_dir = (
    out
    / "03_raw_metadata_limitation"
)

decision_dir = (
    out
    / "04_source_decision"
)

audit_dir = (
    out
    / "05_audit"
)

for directory in (
    record_dir,
    mapping_dir,
    limitation_dir,
    decision_dir,
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


def human_size(
    size_bytes: int,
) -> str:

    value = float(
        size_bytes
    )

    for unit in (
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ):

        if (
            value < 1024
            or unit == "TB"
        ):
            return f"{value:.2f}_{unit}"

        value /= 1024

    return f"{size_bytes}_B"


record = json.loads(
    record_json_path.read_text(
        encoding="utf-8"
    )
)

record_id = str(
    record.get(
        "id",
        "",
    )
)

record_title = str(
    record.get(
        "metadata",
        {},
    ).get(
        "title",
        "",
    )
)

if record_id != "14422018":
    raise SystemExit(
        f"FAIL: unexpected processed-record ID: "
        f"{record_id}"
    )

expected_title_token = (
    "Spatial Transcriptomics Reveals Human "
    "Cortical Layer and Area Specification"
)

if expected_title_token.lower() not in record_title.lower():
    raise SystemExit(
        "FAIL: processed-record title does not match "
        f"the expected study: {record_title}"
    )

record_files = record.get(
    "files",
    []
)

if not isinstance(
    record_files,
    list,
):
    raise SystemExit(
        "FAIL: Zenodo files field is not a list."
    )

all_file_rows: list[
    dict[str, Any]
] = []

file_lookup: dict[
    str,
    dict[str, Any],
] = {}

for file_record in record_files:

    file_name = str(
        file_record.get(
            "key",
            "",
        )
    )

    size_bytes = int(
        file_record.get(
            "size",
            0,
        )
        or 0
    )

    checksum = str(
        file_record.get(
            "checksum",
            "",
        )
    )

    links = file_record.get(
        "links",
        {},
    )

    content_url = str(
        links.get(
            "content",
            "",
        )
    )

    row = {
        "file_name": file_name,
        "size_bytes": size_bytes,
        "size_human": human_size(
            size_bytes
        ),
        "checksum": checksum,
        "content_url": content_url,
        "is_H5AD": (
            file_name.lower().endswith(
                ".h5ad"
            )
        ),
    }

    all_file_rows.append(
        row
    )

    file_lookup[
        file_name
    ] = row

write_tsv(
    record_dir
    / "phase10B5_P4A_R1_complete_processed_record_manifest.tsv",
    all_file_rows,
    [
        "file_name",
        "size_bytes",
        "size_human",
        "checksum",
        "content_url",
        "is_H5AD",
    ],
)

target_h5ad_files = [
    "gw15.h5ad",
    "gw18_umb1759.h5ad",
    "gw20.h5ad",
    "gw20_umb1031.h5ad",
    "gw22.h5ad",
    "gw34.h5ad",
]

missing_target_files = [
    file_name
    for file_name in target_h5ad_files
    if file_name not in file_lookup
]

if missing_target_files:
    raise SystemExit(
        "FAIL: target processed H5AD files are absent: "
        + ";".join(
            missing_target_files
        )
    )

target_file_rows = [
    {
        **file_lookup[
            file_name
        ],
        "selection_reason": {
            "gw15.h5ad": (
                "contains_selected_GW15_occipital_donors"
            ),
            "gw18_umb1759.h5ad": (
                "selected_GW18_expanded_panel_donor"
            ),
            "gw20.h5ad": (
                "contains_selected_GW20_primary_and_"
                "within_donor_sections"
            ),
            "gw20_umb1031.h5ad": (
                "selected_GW20_expanded_panel_donor"
            ),
            "gw22.h5ad": (
                "contains_selected_GW22_primary_and_"
                "section_sensitivity_samples"
            ),
            "gw34.h5ad": (
                "contains_selected_GW34_BA17_primary_"
                "and_BA18_sensitivity_samples"
            ),
        }[
            file_name
        ],
        "payload_downloaded": False,
        "obs_schema_audited": False,
        "expression_values_accessed": False,
    }
    for file_name in target_h5ad_files
]

write_tsv(
    record_dir
    / "phase10B5_P4A_R1_selected_processed_H5AD_manifest.tsv",
    target_file_rows,
    [
        "file_name",
        "size_bytes",
        "size_human",
        "checksum",
        "content_url",
        "is_H5AD",
        "selection_reason",
        "payload_downloaded",
        "obs_schema_audited",
        "expression_values_accessed",
    ],
)

primary_rows, primary_columns = read_tsv(
    primary_path
)

sensitivity_rows, sensitivity_columns = read_tsv(
    sensitivity_path
)

if len(primary_rows) != 8:
    raise SystemExit(
        f"FAIL: expected eight primary samples, "
        f"observed {len(primary_rows)}."
    )

if len(sensitivity_rows) != 5:
    raise SystemExit(
        f"FAIL: expected five sensitivity samples, "
        f"observed {len(sensitivity_rows)}."
    )

selected_rows: list[
    dict[str, str]
] = []

for row in primary_rows:

    copied = dict(
        row
    )

    copied[
        "analysis_set"
    ] = "primary"

    selected_rows.append(
        copied
    )

for row in sensitivity_rows:

    copied = dict(
        row
    )

    copied[
        "analysis_set"
    ] = "section_sensitivity"

    selected_rows.append(
        copied
    )

archive_to_processed_object = {
    "UMB1367_O1.zip": "gw15.h5ad",
    "UMB1117_O1.zip": "gw15.h5ad",

    "UMB1759_O1.zip": "gw18_umb1759.h5ad",

    "FB080_O1c.zip": "gw20.h5ad",
    "FB121_O1.zip": "gw20.h5ad",
    "FB080_O1a.zip": "gw20.h5ad",
    "FB080_O1b.zip": "gw20.h5ad",
    "FB080_O1d.zip": "gw20.h5ad",

    "UMB1031_O1.zip": "gw20_umb1031.h5ad",

    "FB123_O1.zip": "gw22.h5ad",
    "FB123_O2.zip": "gw22.h5ad",

    "UMB5900_BA17.zip": "gw34.h5ad",
    "UMB5900_BA18.zip": "gw34.h5ad",
}

selected_archive_names = {
    row[
        "archive_name"
    ]
    for row in selected_rows
}

if selected_archive_names != set(
    archive_to_processed_object
):
    missing_mapping = sorted(
        selected_archive_names
        - set(
            archive_to_processed_object
        )
    )

    extra_mapping = sorted(
        set(
            archive_to_processed_object
        )
        - selected_archive_names
    )

    raise SystemExit(
        "FAIL: selected archive/object map mismatch. "
        f"Missing={missing_mapping}; "
        f"extra={extra_mapping}"
    )

mapping_rows: list[
    dict[str, Any]
] = []

for selected_order, row in enumerate(
    selected_rows,
    start=1,
):

    archive_name = row[
        "archive_name"
    ]

    processed_object = (
        archive_to_processed_object[
            archive_name
        ]
    )

    mapping_rows.append(
        {
            "selected_order": selected_order,
            "analysis_set": row[
                "analysis_set"
            ],
            "archive_name": archive_name,
            "donor_id": row[
                "donor_id"
            ],
            "gestational_week": row[
                "gestational_week"
            ],
            "panel_class": row[
                "panel_class"
            ],
            "candidate_processed_H5AD": (
                processed_object
            ),
            "mapping_basis": (
                "age_specific_object_name_and_"
                "prespecified_sample_mapping"
            ),
            "requires_obs_sample_identifier_confirmation": (
                True
            ),
            "processed_object_payload_downloaded": (
                False
            ),
            "expression_values_accessed": (
                False
            ),
        }
    )

write_tsv(
    mapping_dir
    / "phase10B5_P4A_R1_archive_to_processed_H5AD_candidate_mapping.tsv",
    mapping_rows,
    [
        "selected_order",
        "analysis_set",
        "archive_name",
        "donor_id",
        "gestational_week",
        "panel_class",
        "candidate_processed_H5AD",
        "mapping_basis",
        "requires_obs_sample_identifier_confirmation",
        "processed_object_payload_downloaded",
        "expression_values_accessed",
    ],
)

with gzip.open(
    raw_schema_path,
    "rt",
    encoding="utf-8-sig",
    newline="",
) as handle:

    reader = csv.DictReader(
        handle,
        delimiter="\t",
    )

    raw_schema_rows = list(
        reader
    )

if not raw_schema_rows:
    raise SystemExit(
        "FAIL: raw metadata schema table is empty."
    )

selected_schema_rows = [
    row
    for row in raw_schema_rows
    if row[
        "archive_name"
    ] in selected_archive_names
]

schema_signatures = {
    row[
        "schema_signature"
    ]
    for row in selected_schema_rows
}

if len(schema_signatures) != 1:
    raise SystemExit(
        "FAIL: selected raw metadata does not have "
        "one common schema signature."
    )

common_schema: dict[
    tuple[int, str],
    dict[str, Any],
] = {}

for row in selected_schema_rows:

    key = (
        int(
            row[
                "column_index"
            ]
        ),
        row[
            "column_name"
        ],
    )

    target = common_schema.setdefault(
        key,
        {
            "column_index": key[0],
            "column_name": key[1],
            "normalized_column_name": row[
                "normalized_column_name"
            ],
            "inferred_roles": set(),
            "prefix_value_types": set(),
            "archives": set(),
        },
    )

    target[
        "inferred_roles"
    ].update(
        role
        for role in row[
            "inferred_roles"
        ].split(";")
        if role
    )

    target[
        "prefix_value_types"
    ].add(
        row[
            "prefix_value_type"
        ]
    )

    target[
        "archives"
    ].add(
        row[
            "archive_name"
        ]
    )

common_schema_rows: list[
    dict[str, Any]
] = []

for (
    column_index,
    column_name,
), row in sorted(
    common_schema.items()
):

    common_schema_rows.append(
        {
            "column_index": column_index,
            "column_name": column_name,
            "normalized_column_name": row[
                "normalized_column_name"
            ],
            "inferred_roles": ";".join(
                sorted(
                    row[
                        "inferred_roles"
                    ]
                )
            ),
            "prefix_value_types": ";".join(
                sorted(
                    row[
                        "prefix_value_types"
                    ]
                )
            ),
            "archives_represented": len(
                row[
                    "archives"
                ]
            ),
            "promoted_for_annotation_dependent_analysis": (
                False
                if (
                    "cell_type_annotation"
                    not in row[
                        "inferred_roles"
                    ]
                    and "cluster_annotation"
                    not in row[
                        "inferred_roles"
                    ]
                    and "layer_or_depth"
                    not in row[
                        "inferred_roles"
                    ]
                )
                else True
            ),
        }
    )

write_tsv(
    limitation_dir
    / "phase10B5_P4A_R1_raw_metadata_common_schema.tsv",
    common_schema_rows,
    [
        "column_index",
        "column_name",
        "normalized_column_name",
        "inferred_roles",
        "prefix_value_types",
        "archives_represented",
        "promoted_for_annotation_dependent_analysis",
    ],
)

annotation_columns = [
    row
    for row in common_schema_rows
    if (
        "cell_type_annotation"
        in row[
            "inferred_roles"
        ]
        or "cluster_annotation"
        in row[
            "inferred_roles"
        ]
    )
]

layer_columns = [
    row
    for row in common_schema_rows
    if "layer_or_depth"
    in row[
        "inferred_roles"
    ]
]

coordinate_columns = [
    row
    for row in common_schema_rows
    if (
        "spatial_x"
        in row[
            "inferred_roles"
        ]
        or "spatial_y"
        in row[
            "inferred_roles"
        ]
    )
]

source_decision_rows = [
    {
        "source": (
            "Zenodo_15127709_raw_MERFISH_outputs"
        ),
        "retained_role": (
            "raw_count_and_segmentation_coordinate_"
            "audit_source"
        ),
        "suitable_for_expression_validation": (
            True
        ),
        "suitable_for_annotation_dependent_validation": (
            False
        ),
        "suitable_for_layer_dependent_validation": (
            False
        ),
        "superseded": False,
        "decision": (
            "retain_as_raw_audit_and_fallback_source"
        ),
    },
    {
        "source": (
            "Zenodo_14422018_processed_MERFISH_H5AD"
        ),
        "retained_role": (
            "primary_source_for_cell_annotation_layer_"
            "and_processed_spatial_validation"
        ),
        "suitable_for_expression_validation": (
            "pending_H5AD_schema_audit"
        ),
        "suitable_for_annotation_dependent_validation": (
            "pending_H5AD_obs_audit"
        ),
        "suitable_for_layer_dependent_validation": (
            "pending_H5AD_obs_audit"
        ),
        "superseded": False,
        "decision": (
            "select_for_targeted_download_and_schema_audit"
        ),
    },
]

write_tsv(
    decision_dir
    / "phase10B5_P4A_R1_data_source_decision.tsv",
    source_decision_rows,
    [
        "source",
        "retained_role",
        "suitable_for_expression_validation",
        "suitable_for_annotation_dependent_validation",
        "suitable_for_layer_dependent_validation",
        "superseded",
        "decision",
    ],
)

total_target_bytes = sum(
    int(
        row[
            "size_bytes"
        ]
    )
    for row in target_file_rows
)

technical_pass = (
    len(target_file_rows) == 6
    and len(mapping_rows) == 13
    and len(
        {
            row[
                "candidate_processed_H5AD"
            ]
            for row in mapping_rows
        }
    ) == 6
    and len(schema_signatures) == 1
    and len(annotation_columns) == 0
    and len(layer_columns) == 0
    and len(coordinate_columns) == 2
)

status_value = (
    "passed_phase10B5_P4A_R1_processed_H5AD_"
    "source_substitution_preflight_ready_for_"
    "selective_processed_object_download_and_obs_schema_audit"
    if technical_pass
    else (
        "phase10B5_P4A_R1_requires_manual_review"
    )
)

status = {
    "phase": "phase10B5_P4A_R1",
    "processed_Zenodo_record_ID": (
        record_id
    ),
    "processed_record_files": len(
        all_file_rows
    ),
    "selected_processed_H5AD_files": len(
        target_file_rows
    ),
    "selected_archive_mappings": len(
        mapping_rows
    ),
    "unique_processed_objects_selected": len(
        {
            row[
                "candidate_processed_H5AD"
            ]
            for row in mapping_rows
        }
    ),
    "selected_processed_payload_bytes": (
        total_target_bytes
    ),
    "selected_processed_payload_size_human": (
        human_size(
            total_target_bytes
        )
    ),
    "raw_metadata_schema_signatures": len(
        schema_signatures
    ),
    "raw_annotation_or_cluster_columns": len(
        annotation_columns
    ),
    "raw_layer_or_depth_columns": len(
        layer_columns
    ),
    "raw_coordinate_columns": len(
        coordinate_columns
    ),
    "raw_CSV_source_retained_as_audit": True,
    "raw_CSV_source_invalidated": False,
    "processed_H5AD_required_for_annotation_layer_audit": (
        True
    ),
    "network_record_metadata_request_performed": True,
    "processed_H5AD_payload_downloaded": False,
    "complete_ZIP_archives_downloaded": False,
    "expression_values_accessed": False,
    "module_scores_computed": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P4A_R1_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B5_P4A_R1_status.tsv",
    [status],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P4A-R1 PROCESSED H5AD "
    "SOURCE-SUBSTITUTION PREFLIGHT =====",
    "",
    (
        "Processed Zenodo record ID: "
        f"{record_id}"
    ),
    (
        "Processed-record files: "
        f"{len(all_file_rows)}"
    ),
    (
        "Selected processed H5AD files: "
        f"{len(target_file_rows)}"
    ),
    (
        "Selected archive mappings: "
        f"{len(mapping_rows)}"
    ),
    (
        "Unique processed objects selected: "
        f"{len(set(row['candidate_processed_H5AD'] for row in mapping_rows))}"
    ),
    (
        "Selected processed payload: "
        f"{human_size(total_target_bytes)}"
    ),
    (
        "Raw metadata schema signatures: "
        f"{len(schema_signatures)}"
    ),
    (
        "Raw annotation or cluster columns: "
        f"{len(annotation_columns)}"
    ),
    (
        "Raw layer or depth columns: "
        f"{len(layer_columns)}"
    ),
    (
        "Raw coordinate columns: "
        f"{len(coordinate_columns)}"
    ),
    "",
    "Raw CSV source retained as audit: TRUE",
    "Raw CSV source invalidated: FALSE",
    (
        "Processed H5AD required for annotation/layer "
        "audit: TRUE"
    ),
    "Network record-metadata request performed: TRUE",
    "Processed H5AD payload downloaded: FALSE",
    "Complete ZIP archives downloaded: FALSE",
    "Expression values accessed: FALSE",
    "Module scores computed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P4A-R1 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P4A_R1_report.txt"
).write_text(
    "\n".join(report)
    + "\n",
    encoding="utf-8",
)

print(
    "\n".join(report)
)

print(
    "\n===== SELECTED PROCESSED H5AD FILES ====="
)

for row in target_file_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "file_name",
                "size_human",
                "checksum",
                "selection_reason",
            )
        )
    )

print(
    "\n===== ARCHIVE TO PROCESSED-OBJECT MAP ====="
)

for row in mapping_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "analysis_set",
                "archive_name",
                "gestational_week",
                "candidate_processed_H5AD",
                "requires_obs_sample_identifier_confirmation",
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
        != "phase10B5_P4A_R1_SHA256.tsv"
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
    / "phase10B5_P4A_R1_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)
