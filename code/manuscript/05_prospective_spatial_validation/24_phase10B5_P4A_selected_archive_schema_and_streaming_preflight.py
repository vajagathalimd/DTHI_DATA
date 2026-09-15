from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


project = Path(sys.argv[1])
primary_path = Path(sys.argv[2])
sensitivity_path = Path(sys.argv[3])
resolution_path = Path(sys.argv[4])
member_manifest_path = Path(sys.argv[5])
remote_manifest_path = Path(sys.argv[6])
eligibility_path = Path(sys.argv[7])
raw_header_dir = Path(sys.argv[8])
raw_prefix_dir = Path(sys.argv[9])
out = Path(sys.argv[10])

metadata_dir = (
    out
    / "01_metadata_schema"
)

matrix_dir = (
    out
    / "02_matrix_schema"
)

streaming_dir = (
    out
    / "03_streaming_plan"
)

gene_dir = (
    out
    / "04_selected_gene_inventory"
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
    metadata_dir,
    matrix_dir,
    streaming_dir,
    gene_dir,
    exception_dir,
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


def as_bool(
    value: Any,
) -> bool:

    return str(value).strip().upper() in {
        "TRUE",
        "T",
        "1",
        "YES",
    }


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


def infer_column_type(
    values: list[str],
) -> str:

    nonempty = [
        value
        for value in values
        if value.strip()
    ]

    if not nonempty:
        return "empty_in_prefix"

    numeric = 0
    integer = 0

    for value in nonempty:

        try:
            parsed = float(
                value
            )

            numeric += 1

            if parsed.is_integer():
                integer += 1

        except ValueError:
            pass

    if numeric == len(
        nonempty
    ):

        if integer == len(
            nonempty
        ):
            return "integer"

        return "numeric"

    if numeric > 0:
        return "mixed"

    return "text"


def classify_column(
    column_name: str,
) -> list[str]:

    normalized = normalize(
        column_name
    )

    roles: list[str] = []

    exact_identifier_names = {
        "cell",
        "cell_id",
        "cellid",
        "barcode",
        "barcode_id",
        "object_id",
        "segmentation_id",
        "index",
        "unnamed_0",
    }

    if (
        normalized in exact_identifier_names
        or normalized.endswith(
            "_cell_id"
        )
    ):
        roles.append(
            "cell_identifier"
        )

    if any(
        token in normalized
        for token in (
            "cell_type",
            "celltype",
            "cell_annotation",
            "annotation",
            "subclass",
            "sub_class",
            "major_type",
            "broad_type",
            "broad_class",
            "cell_class",
        )
    ):
        roles.append(
            "cell_type_annotation"
        )

    if any(
        token in normalized
        for token in (
            "cluster",
            "leiden",
            "louvain",
            "community",
        )
    ):
        roles.append(
            "cluster_annotation"
        )

    x_exact = {
        "x",
        "x_coord",
        "x_coordinate",
        "center_x",
        "centroid_x",
        "global_x",
        "spatial_x",
        "x_centroid",
    }

    y_exact = {
        "y",
        "y_coord",
        "y_coordinate",
        "center_y",
        "centroid_y",
        "global_y",
        "spatial_y",
        "y_centroid",
    }

    if normalized in x_exact:
        roles.append(
            "spatial_x"
        )

    if normalized in y_exact:
        roles.append(
            "spatial_y"
        )

    if any(
        token in normalized
        for token in (
            "cortical_layer",
            "layer_label",
            "layer_annotation",
            "normalized_depth",
            "relative_depth",
            "cortical_depth",
            "distance_to_pia",
            "pia_distance",
            "wm_distance",
            "white_matter_distance",
        )
    ) or normalized in {
        "layer",
        "depth",
    }:
        roles.append(
            "layer_or_depth"
        )

    if any(
        token in normalized
        for token in (
            "sample_id",
            "section_id",
            "slice_id",
            "fov",
            "field_of_view",
            "brain_id",
            "donor_id",
            "subject_id",
        )
    ):
        roles.append(
            "sample_or_section_identifier"
        )

    if (
        normalized == "area"
        or any(
            token in normalized
            for token in (
                "cell_area",
                "nucleus_area",
                "volume",
                "perimeter",
                "width",
                "height",
                "major_axis",
                "minor_axis",
                "eccentricity",
            )
        )
    ):
        roles.append(
            "morphology_not_cortical_area"
        )

    if any(
        token in normalized
        for token in (
            "transcript_count",
            "total_counts",
            "n_counts",
            "n_genes",
            "gene_count",
            "quality",
            "qc",
            "blank_count",
        )
    ):
        roles.append(
            "quality_control"
        )

    if not roles:
        roles.append(
            "unclassified"
        )

    return roles


def choose_candidate(
    columns: list[str],
    roles_by_column: dict[str, list[str]],
    target_role: str,
    priority_names: tuple[str, ...],
) -> str:

    normalized_lookup = {
        normalize(
            column
        ): column
        for column in columns
    }

    for priority in priority_names:

        if priority in normalized_lookup:

            column = normalized_lookup[
                priority
            ]

            if target_role in roles_by_column[
                column
            ]:
                return column

    candidates = [
        column
        for column in columns
        if target_role
        in roles_by_column[
            column
        ]
    ]

    return (
        candidates[0]
        if len(candidates) == 1
        else ""
    )


primary_rows, primary_columns = read_tsv(
    primary_path
)

sensitivity_rows, sensitivity_columns = read_tsv(
    sensitivity_path
)

if len(primary_rows) != 8:
    raise SystemExit(
        f"FAIL: expected eight primary rows, "
        f"observed {len(primary_rows)}."
    )

if len(sensitivity_rows) != 5:
    raise SystemExit(
        f"FAIL: expected five sensitivity rows, "
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

selected_archives = [
    row[
        "archive_name"
    ]
    for row in selected_rows
]

if len(
    selected_archives
) != len(
    set(
        selected_archives
    )
):
    raise SystemExit(
        "FAIL: selected archive list contains duplicates."
    )

if len(selected_archives) != 13:
    raise SystemExit(
        f"FAIL: expected 13 selected archives, "
        f"observed {len(selected_archives)}."
    )

selected_lookup = {
    row["archive_name"]: row
    for row in selected_rows
}

resolution_rows, resolution_columns = read_tsv(
    resolution_path
)

resolution_lookup = {
    row["archive_name"]: row
    for row in resolution_rows
}

with gzip.open(
    member_manifest_path,
    "rt",
    encoding="utf-8-sig",
    newline="",
) as handle:

    reader = csv.DictReader(
        handle,
        delimiter="\t",
    )

    member_rows = list(
        reader
    )

member_lookup = {
    (
        row["archive_name"],
        row["member_name"],
    ): row
    for row in member_rows
}

remote_rows, remote_columns = read_tsv(
    remote_manifest_path
)

remote_lookup = {
    row["file_name"]: row
    for row in remote_rows
}

eligibility_rows, eligibility_columns = read_tsv(
    eligibility_path
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

metadata_column_rows: list[
    dict[str, Any]
] = []

metadata_archive_rows: list[
    dict[str, Any]
] = []

matrix_schema_rows: list[
    dict[str, Any]
] = []

streaming_rows: list[
    dict[str, Any]
] = []

selected_gene_rows: list[
    dict[str, Any]
] = []

exception_rows: list[
    dict[str, Any]
] = []

metadata_signatures = Counter()
role_archive_counts = Counter()

total_planned_compressed_bytes = 0

for selected_order, selected in enumerate(
    selected_rows,
    start=1,
):

    archive_name = selected[
        "archive_name"
    ]

    archive_stem = Path(
        archive_name
    ).stem

    analysis_set = selected[
        "analysis_set"
    ]

    resolution = resolution_lookup.get(
        archive_name
    )

    remote = remote_lookup.get(
        archive_name
    )

    if resolution is None:

        exception_rows.append(
            {
                "archive_name": archive_name,
                "exception_type": (
                    "missing_archive_resolution"
                ),
                "details": "",
            }
        )

        continue

    if remote is None:

        exception_rows.append(
            {
                "archive_name": archive_name,
                "exception_type": (
                    "missing_remote_manifest_record"
                ),
                "details": "",
            }
        )

        continue

    matrix_member_name = resolution[
        "selected_cell_by_gene_member"
    ]

    metadata_member_name = resolution[
        "selected_metadata_member"
    ]

    matrix_member = member_lookup.get(
        (
            archive_name,
            matrix_member_name,
        )
    )

    metadata_member = member_lookup.get(
        (
            archive_name,
            metadata_member_name,
        )
    )

    if matrix_member is None:

        exception_rows.append(
            {
                "archive_name": archive_name,
                "exception_type": (
                    "missing_matrix_member_record"
                ),
                "details": (
                    matrix_member_name
                ),
            }
        )

        continue

    if metadata_member is None:

        exception_rows.append(
            {
                "archive_name": archive_name,
                "exception_type": (
                    "missing_metadata_member_record"
                ),
                "details": (
                    metadata_member_name
                ),
            }
        )

        continue

    matrix_header_path = (
        raw_header_dir
        / (
            f"{archive_stem}_"
            "cell_by_gene_header.csv"
        )
    )

    metadata_prefix_path = (
        raw_prefix_dir
        / (
            f"{archive_stem}_"
            "cell_metadata_prefix.csv"
        )
    )

    if not matrix_header_path.is_file():

        exception_rows.append(
            {
                "archive_name": archive_name,
                "exception_type": (
                    "missing_local_matrix_header"
                ),
                "details": str(
                    matrix_header_path
                ),
            }
        )

        continue

    if not metadata_prefix_path.is_file():

        exception_rows.append(
            {
                "archive_name": archive_name,
                "exception_type": (
                    "missing_local_metadata_prefix"
                ),
                "details": str(
                    metadata_prefix_path
                ),
            }
        )

        continue

    with matrix_header_path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        matrix_header = next(
            csv.reader(
                handle
            )
        )

    matrix_first_field = (
        matrix_header[0]
        if matrix_header
        else ""
    )

    matrix_identifier_present = (
        normalize(
            matrix_first_field
        )
        in {
            "",
            "cell",
            "cell_id",
            "cellid",
            "barcode",
            "index",
            "unnamed_0",
        }
    )

    matrix_schema_rows.append(
        {
            "selected_order": selected_order,
            "analysis_set": analysis_set,
            "archive_name": archive_name,
            "matrix_member": (
                matrix_member_name
            ),
            "raw_header_fields": len(
                matrix_header
            ),
            "first_header_field": (
                matrix_first_field
            ),
            "first_field_is_identifier_or_blank": (
                matrix_identifier_present
            ),
            "matrix_compression_method": int(
                matrix_member[
                    "compression_method"
                ]
            ),
            "matrix_compressed_size_bytes": int(
                matrix_member[
                    "compressed_size_bytes"
                ]
            ),
            "matrix_uncompressed_size_bytes": int(
                matrix_member[
                    "uncompressed_size_bytes"
                ]
            ),
            "matrix_local_header_offset": int(
                matrix_member[
                    "local_header_offset"
                ]
            ),
        }
    )

    with metadata_prefix_path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.reader(
            handle
        )

        prefix_rows = list(
            reader
        )

    if len(prefix_rows) < 2:

        exception_rows.append(
            {
                "archive_name": archive_name,
                "exception_type": (
                    "metadata_prefix_has_no_data_rows"
                ),
                "details": str(
                    metadata_prefix_path
                ),
            }
        )

        continue

    metadata_header = prefix_rows[0]

    metadata_data = [
        row
        for row in prefix_rows[1:]
        if row
    ]

    signature_source = "\t".join(
        normalize(
            column
        )
        for column in metadata_header
    )

    schema_signature = hashlib.sha256(
        signature_source.encode(
            "utf-8"
        )
    ).hexdigest()

    metadata_signatures[
        schema_signature
    ] += 1

    roles_by_column: dict[
        str,
        list[str],
    ] = {}

    for column_index, column_name in enumerate(
        metadata_header
    ):

        values = [
            row[column_index]
            for row in metadata_data
            if column_index < len(
                row
            )
        ]

        roles = classify_column(
            column_name
        )

        roles_by_column[
            column_name
        ] = roles

        unique_values: list[str] = []

        seen: set[str] = set()

        for value in values:

            stripped = value.strip()

            if (
                stripped
                and stripped not in seen
            ):

                seen.add(
                    stripped
                )

                unique_values.append(
                    stripped
                )

            if len(unique_values) >= 10:
                break

        metadata_column_rows.append(
            {
                "archive_name": archive_name,
                "analysis_set": analysis_set,
                "schema_signature": (
                    schema_signature
                ),
                "column_index": (
                    column_index
                ),
                "column_name": (
                    column_name
                ),
                "normalized_column_name": (
                    normalize(
                        column_name
                    )
                ),
                "inferred_roles": ";".join(
                    roles
                ),
                "prefix_value_type": (
                    infer_column_type(
                        values
                    )
                ),
                "nonempty_prefix_values": sum(
                    bool(
                        value.strip()
                    )
                    for value in values
                ),
                "example_values": ";".join(
                    unique_values
                ),
            }
        )

    cell_id_column = choose_candidate(
        metadata_header,
        roles_by_column,
        "cell_identifier",
        (
            "cell_id",
            "cellid",
            "cell",
            "barcode",
            "index",
            "unnamed_0",
        ),
    )

    cell_type_column = choose_candidate(
        metadata_header,
        roles_by_column,
        "cell_type_annotation",
        (
            "cell_type",
            "celltype",
            "annotation",
            "subclass",
            "cell_class",
            "major_type",
        ),
    )

    cluster_column = choose_candidate(
        metadata_header,
        roles_by_column,
        "cluster_annotation",
        (
            "cluster",
            "leiden",
            "louvain",
        ),
    )

    x_column = choose_candidate(
        metadata_header,
        roles_by_column,
        "spatial_x",
        (
            "center_x",
            "centroid_x",
            "global_x",
            "spatial_x",
            "x",
        ),
    )

    y_column = choose_candidate(
        metadata_header,
        roles_by_column,
        "spatial_y",
        (
            "center_y",
            "centroid_y",
            "global_y",
            "spatial_y",
            "y",
        ),
    )

    layer_depth_column = choose_candidate(
        metadata_header,
        roles_by_column,
        "layer_or_depth",
        (
            "cortical_layer",
            "layer",
            "normalized_depth",
            "relative_depth",
            "cortical_depth",
            "distance_to_pia",
            "depth",
        ),
    )

    morphology_columns = [
        column
        for column in metadata_header
        if (
            "morphology_not_cortical_area"
            in roles_by_column[
                column
            ]
        )
    ]

    annotation_ready = bool(
        cell_type_column
        or cluster_column
    )

    coordinate_ready = bool(
        x_column
        and y_column
    )

    identifier_ready = bool(
        cell_id_column
    )

    for role_name, role_ready in (
        (
            "cell_identifier",
            identifier_ready,
        ),
        (
            "annotation_or_cluster",
            annotation_ready,
        ),
        (
            "spatial_coordinates",
            coordinate_ready,
        ),
        (
            "layer_or_depth",
            bool(
                layer_depth_column
            ),
        ),
    ):

        if role_ready:
            role_archive_counts[
                role_name
            ] += 1

    metadata_archive_rows.append(
        {
            "selected_order": selected_order,
            "analysis_set": analysis_set,
            "archive_name": archive_name,
            "schema_signature": (
                schema_signature
            ),
            "metadata_columns": len(
                metadata_header
            ),
            "prefix_data_rows": len(
                metadata_data
            ),
            "selected_cell_id_column": (
                cell_id_column
            ),
            "selected_cell_type_column": (
                cell_type_column
            ),
            "selected_cluster_column": (
                cluster_column
            ),
            "selected_spatial_x_column": (
                x_column
            ),
            "selected_spatial_y_column": (
                y_column
            ),
            "selected_layer_or_depth_column": (
                layer_depth_column
            ),
            "morphology_columns_not_cortical_area": (
                ";".join(
                    morphology_columns
                )
            ),
            "identifier_ready": (
                identifier_ready
            ),
            "annotation_ready": (
                annotation_ready
            ),
            "coordinate_ready": (
                coordinate_ready
            ),
            "layer_or_depth_ready": bool(
                layer_depth_column
            ),
        }
    )

    archive_eligibility = eligibility_lookup.get(
        archive_name,
        []
    )

    if len(archive_eligibility) != 9:

        exception_rows.append(
            {
                "archive_name": archive_name,
                "exception_type": (
                    "eligibility_row_count_not_nine"
                ),
                "details": str(
                    len(
                        archive_eligibility
                    )
                ),
            }
        )

        continue

    selected_modules: list[
        dict[str, str]
    ] = []

    selected_genes: set[str] = set()

    for eligibility in archive_eligibility:

        sensitivity_eligible = as_bool(
            eligibility[
                "sensitivity_scoring_eligible"
            ]
        )

        if not sensitivity_eligible:
            continue

        selected_modules.append(
            eligibility
        )

        matched_genes = [
            gene.strip()
            for gene in eligibility[
                "matched_genes"
            ].split(";")
            if gene.strip()
        ]

        selected_genes.update(
            matched_genes
        )

        for gene in matched_genes:

            selected_gene_rows.append(
                {
                    "archive_name": archive_name,
                    "analysis_set": analysis_set,
                    "module": eligibility[
                        "module"
                    ],
                    "primary_scoring_eligible": (
                        as_bool(
                            eligibility[
                                "primary_scoring_eligible"
                            ]
                        )
                    ),
                    "sensitivity_scoring_eligible": (
                        sensitivity_eligible
                    ),
                    "gene_symbol": gene,
                }
            )

    matrix_compressed_size = int(
        matrix_member[
            "compressed_size_bytes"
        ]
    )

    metadata_compressed_size = int(
        metadata_member[
            "compressed_size_bytes"
        ]
    )

    planned_compressed_bytes = (
        matrix_compressed_size
        + metadata_compressed_size
    )

    total_planned_compressed_bytes += (
        planned_compressed_bytes
    )

    streaming_rows.append(
        {
            "selected_order": selected_order,
            "analysis_set": analysis_set,
            "archive_name": archive_name,
            "donor_id": selected[
                "donor_id"
            ],
            "gestational_week": selected[
                "gestational_week"
            ],
            "panel_class": selected[
                "panel_class"
            ],
            "remote_content_url": remote[
                "content_url"
            ],
            "remote_archive_size_bytes": int(
                remote[
                    "size_bytes"
                ]
            ),
            "matrix_member": (
                matrix_member_name
            ),
            "matrix_compression_method": int(
                matrix_member[
                    "compression_method"
                ]
            ),
            "matrix_local_header_offset": int(
                matrix_member[
                    "local_header_offset"
                ]
            ),
            "matrix_compressed_size_bytes": (
                matrix_compressed_size
            ),
            "matrix_uncompressed_size_bytes": int(
                matrix_member[
                    "uncompressed_size_bytes"
                ]
            ),
            "metadata_member": (
                metadata_member_name
            ),
            "metadata_compression_method": int(
                metadata_member[
                    "compression_method"
                ]
            ),
            "metadata_local_header_offset": int(
                metadata_member[
                    "local_header_offset"
                ]
            ),
            "metadata_compressed_size_bytes": (
                metadata_compressed_size
            ),
            "metadata_uncompressed_size_bytes": int(
                metadata_member[
                    "uncompressed_size_bytes"
                ]
            ),
            "selected_cell_id_column": (
                cell_id_column
            ),
            "selected_cell_type_column": (
                cell_type_column
            ),
            "selected_cluster_column": (
                cluster_column
            ),
            "selected_spatial_x_column": (
                x_column
            ),
            "selected_spatial_y_column": (
                y_column
            ),
            "selected_layer_or_depth_column": (
                layer_depth_column
            ),
            "sensitivity_eligible_modules": ";".join(
                sorted(
                    eligibility[
                        "module"
                    ]
                    for eligibility
                    in selected_modules
                )
            ),
            "selected_unique_gene_count": len(
                selected_genes
            ),
            "selected_unique_genes": ";".join(
                sorted(
                    selected_genes
                )
            ),
            "planned_compressed_bytes": (
                planned_compressed_bytes
            ),
            "planned_compressed_size_human": (
                human_size(
                    planned_compressed_bytes
                )
            ),
            "complete_ZIP_archive_required": (
                False
            ),
            "expression_values_accessed": (
                False
            ),
        }
    )

metadata_column_columns = [
    "archive_name",
    "analysis_set",
    "schema_signature",
    "column_index",
    "column_name",
    "normalized_column_name",
    "inferred_roles",
    "prefix_value_type",
    "nonempty_prefix_values",
    "example_values",
]

write_tsv_gz(
    metadata_dir
    / "phase10B5_P4A_metadata_column_schema.tsv.gz",
    metadata_column_rows,
    metadata_column_columns,
)

metadata_archive_columns = [
    "selected_order",
    "analysis_set",
    "archive_name",
    "schema_signature",
    "metadata_columns",
    "prefix_data_rows",
    "selected_cell_id_column",
    "selected_cell_type_column",
    "selected_cluster_column",
    "selected_spatial_x_column",
    "selected_spatial_y_column",
    "selected_layer_or_depth_column",
    "morphology_columns_not_cortical_area",
    "identifier_ready",
    "annotation_ready",
    "coordinate_ready",
    "layer_or_depth_ready",
]

write_tsv(
    metadata_dir
    / "phase10B5_P4A_archive_metadata_schema_resolution.tsv",
    metadata_archive_rows,
    metadata_archive_columns,
)

write_tsv(
    matrix_dir
    / "phase10B5_P4A_matrix_schema_resolution.tsv",
    matrix_schema_rows,
    [
        "selected_order",
        "analysis_set",
        "archive_name",
        "matrix_member",
        "raw_header_fields",
        "first_header_field",
        "first_field_is_identifier_or_blank",
        "matrix_compression_method",
        "matrix_compressed_size_bytes",
        "matrix_uncompressed_size_bytes",
        "matrix_local_header_offset",
    ],
)

write_tsv(
    streaming_dir
    / "phase10B5_P4A_targeted_member_streaming_plan.tsv",
    streaming_rows,
    [
        "selected_order",
        "analysis_set",
        "archive_name",
        "donor_id",
        "gestational_week",
        "panel_class",
        "remote_content_url",
        "remote_archive_size_bytes",
        "matrix_member",
        "matrix_compression_method",
        "matrix_local_header_offset",
        "matrix_compressed_size_bytes",
        "matrix_uncompressed_size_bytes",
        "metadata_member",
        "metadata_compression_method",
        "metadata_local_header_offset",
        "metadata_compressed_size_bytes",
        "metadata_uncompressed_size_bytes",
        "selected_cell_id_column",
        "selected_cell_type_column",
        "selected_cluster_column",
        "selected_spatial_x_column",
        "selected_spatial_y_column",
        "selected_layer_or_depth_column",
        "sensitivity_eligible_modules",
        "selected_unique_gene_count",
        "selected_unique_genes",
        "planned_compressed_bytes",
        "planned_compressed_size_human",
        "complete_ZIP_archive_required",
        "expression_values_accessed",
    ],
)

write_tsv(
    gene_dir
    / "phase10B5_P4A_selected_module_gene_inventory.tsv",
    selected_gene_rows,
    [
        "archive_name",
        "analysis_set",
        "module",
        "primary_scoring_eligible",
        "sensitivity_scoring_eligible",
        "gene_symbol",
    ],
)

write_tsv(
    exception_dir
    / "phase10B5_P4A_schema_preflight_exceptions.tsv",
    exception_rows,
    [
        "archive_name",
        "exception_type",
        "details",
    ],
)

signature_rows = [
    {
        "schema_signature": signature,
        "archives": count,
        "archive_names": ";".join(
            sorted(
                row[
                    "archive_name"
                ]
                for row in metadata_archive_rows
                if row[
                    "schema_signature"
                ] == signature
            )
        ),
    }
    for signature, count in sorted(
        metadata_signatures.items()
    )
]

write_tsv(
    metadata_dir
    / "phase10B5_P4A_metadata_schema_signature_summary.tsv",
    signature_rows,
    [
        "schema_signature",
        "archives",
        "archive_names",
    ],
)

role_summary_rows = [
    {
        "required_role": role,
        "archives_ready": (
            role_archive_counts.get(
                role,
                0,
            )
        ),
        "selected_archives": 13,
        "all_selected_archives_ready": (
            role_archive_counts.get(
                role,
                0,
            ) == 13
        ),
    }
    for role in (
        "cell_identifier",
        "annotation_or_cluster",
        "spatial_coordinates",
        "layer_or_depth",
    )
]

write_tsv(
    metadata_dir
    / "phase10B5_P4A_required_role_summary.tsv",
    role_summary_rows,
    [
        "required_role",
        "archives_ready",
        "selected_archives",
        "all_selected_archives_ready",
    ],
)

primary_streaming_rows = [
    row
    for row in streaming_rows
    if row[
        "analysis_set"
    ] == "primary"
]

sensitivity_streaming_rows = [
    row
    for row in streaming_rows
    if row[
        "analysis_set"
    ] == "section_sensitivity"
]

technical_pass = (
    len(streaming_rows) == 13
    and len(primary_streaming_rows) == 8
    and len(sensitivity_streaming_rows) == 5
    and len(metadata_archive_rows) == 13
    and len(matrix_schema_rows) == 13
    and not exception_rows
)

identifier_ready_archives = (
    role_archive_counts.get(
        "cell_identifier",
        0,
    )
)

annotation_ready_archives = (
    role_archive_counts.get(
        "annotation_or_cluster",
        0,
    )
)

coordinate_ready_archives = (
    role_archive_counts.get(
        "spatial_coordinates",
        0,
    )
)

layer_depth_ready_archives = (
    role_archive_counts.get(
        "layer_or_depth",
        0,
    )
)

if (
    technical_pass
    and identifier_ready_archives == 13
    and annotation_ready_archives == 13
    and coordinate_ready_archives == 13
):

    status_value = (
        "passed_phase10B5_P4A_selected_archive_schema_"
        "and_streaming_preflight_ready_for_targeted_"
        "matrix_metadata_streaming"
    )

elif technical_pass:

    status_value = (
        "completed_phase10B5_P4A_technical_streaming_"
        "preflight_requires_manual_metadata_column_lock"
    )

else:

    status_value = (
        "phase10B5_P4A_requires_manual_review"
    )

status = {
    "phase": "phase10B5_P4A",
    "selected_archives": len(
        streaming_rows
    ),
    "primary_archives": len(
        primary_streaming_rows
    ),
    "section_sensitivity_archives": len(
        sensitivity_streaming_rows
    ),
    "metadata_schema_signatures": len(
        metadata_signatures
    ),
    "archives_with_cell_identifier": (
        identifier_ready_archives
    ),
    "archives_with_annotation_or_cluster": (
        annotation_ready_archives
    ),
    "archives_with_spatial_coordinates": (
        coordinate_ready_archives
    ),
    "archives_with_layer_or_depth": (
        layer_depth_ready_archives
    ),
    "selected_module_gene_rows": len(
        selected_gene_rows
    ),
    "planned_remote_compressed_bytes": (
        total_planned_compressed_bytes
    ),
    "planned_remote_compressed_size_human": (
        human_size(
            total_planned_compressed_bytes
        )
    ),
    "schema_preflight_exceptions": len(
        exception_rows
    ),
    "network_requests_performed": False,
    "complete_ZIP_archives_downloaded": False,
    "expression_payload_downloaded": False,
    "expression_values_accessed": False,
    "module_scores_computed": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P4A_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B5_P4A_status.tsv",
    [
        status
    ],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P4A SELECTED-ARCHIVE "
    "SCHEMA AND STREAMING PREFLIGHT =====",
    "",
    (
        "Selected archives: "
        f"{len(streaming_rows)}"
    ),
    (
        "Primary archives: "
        f"{len(primary_streaming_rows)}"
    ),
    (
        "Section-sensitivity archives: "
        f"{len(sensitivity_streaming_rows)}"
    ),
    (
        "Metadata schema signatures: "
        f"{len(metadata_signatures)}"
    ),
    (
        "Archives with cell identifier: "
        f"{identifier_ready_archives}/13"
    ),
    (
        "Archives with annotation or cluster: "
        f"{annotation_ready_archives}/13"
    ),
    (
        "Archives with spatial coordinates: "
        f"{coordinate_ready_archives}/13"
    ),
    (
        "Archives with layer or depth: "
        f"{layer_depth_ready_archives}/13"
    ),
    (
        "Selected module-gene rows: "
        f"{len(selected_gene_rows)}"
    ),
    (
        "Planned remote compressed payload: "
        f"{human_size(total_planned_compressed_bytes)}"
    ),
    (
        "Schema-preflight exceptions: "
        f"{len(exception_rows)}"
    ),
    "",
    "Network requests performed: FALSE",
    "Complete ZIP archives downloaded: FALSE",
    "Expression payload downloaded: FALSE",
    "Expression values accessed: FALSE",
    "Module scores computed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P4A STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P4A_report.txt"
).write_text(
    "\n".join(report)
    + "\n",
    encoding="utf-8",
)

print(
    "\n".join(report)
)

print(
    "\n===== REQUIRED METADATA ROLES ====="
)

for row in role_summary_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "required_role",
                "archives_ready",
                "selected_archives",
                "all_selected_archives_ready",
            )
        )
    )

print(
    "\n===== ARCHIVE METADATA SCHEMA ====="
)

for row in metadata_archive_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "archive_name",
                "selected_cell_id_column",
                "selected_cell_type_column",
                "selected_cluster_column",
                "selected_spatial_x_column",
                "selected_spatial_y_column",
                "selected_layer_or_depth_column",
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
        != "phase10B5_P4A_SHA256.tsv"
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
    / "phase10B5_P4A_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)
