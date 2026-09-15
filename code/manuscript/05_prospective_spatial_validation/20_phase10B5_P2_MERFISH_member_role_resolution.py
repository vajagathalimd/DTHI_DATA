from __future__ import annotations

import csv
import gzip
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


project = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
out = Path(sys.argv[3])

role_dir = (
    out
    / "01_member_roles"
)

resolution_dir = (
    out
    / "02_archive_resolution"
)

exception_dir = (
    out
    / "03_exceptions"
)

audit_dir = (
    out
    / "04_audit"
)

for directory in (
    role_dir,
    resolution_dir,
    exception_dir,
    audit_dir,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
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


def classify_role(
    member_name: str,
    is_directory: bool,
) -> tuple[
    str,
    bool,
    str,
]:

    normalized = member_name.replace(
        "\\",
        "/",
    )

    lower = normalized.lower()

    basename = Path(
        normalized
    ).name

    basename_lower = basename.lower()

    path_parts = {
        part.lower()
        for part in Path(
            normalized
        ).parts
    }

    if is_directory:
        return (
            "directory",
            False,
            "directory_entry",
        )

    if (
        "__macosx" in path_parts
        or basename_lower.startswith(
            "._"
        )
        or basename_lower
        in {
            ".ds_store",
            "thumbs.db",
        }
    ):
        return (
            "filesystem_artifact",
            False,
            "excluded_operating_system_artifact",
        )

    if basename_lower in {
        "cell_metadata.csv",
        "cell_metadata.tsv",
    }:
        return (
            "cell_metadata",
            True,
            "candidate_for_metadata_prefix_audit",
        )

    if (
        "cell_by_gene" in basename_lower
        or "cellbygene" in basename_lower
        or "gene_by_cell" in basename_lower
        or "genebycell" in basename_lower
    ):
        return (
            "cell_by_gene_matrix",
            True,
            "candidate_for_header_only_panel_audit",
        )

    if (
        "detected_transcript" in basename_lower
        or (
            "transcript" in basename_lower
            and "metadata" not in basename_lower
        )
    ):
        return (
            "detected_transcripts",
            False,
            "held_not_required_for_panel_preflight",
        )

    if basename_lower in {
        "readme",
        "readme.txt",
        "readme.md",
        "license",
        "license.txt",
    }:
        return (
            "documentation",
            True,
            "candidate_for_small_document_review",
        )

    if re.search(
        r"(expression|counts?|matrix)",
        basename_lower,
    ):
        return (
            "other_expression_matrix",
            False,
            "manual_review_expression_member",
        )

    return (
        "other_member",
        False,
        "manual_review",
    )


with gzip.open(
    manifest_path,
    "rt",
    encoding="utf-8-sig",
    newline="",
) as handle:

    reader = csv.DictReader(
        handle,
        delimiter="\t",
    )

    columns = reader.fieldnames or []
    raw_rows = list(reader)

required_columns = {
    "archive_name",
    "member_name",
    "member_basename",
    "is_directory",
    "compressed_size_bytes",
    "uncompressed_size_bytes",
    "compression_method",
    "encrypted",
    "local_header_offset",
    "crc32_hex",
}

missing = sorted(
    required_columns
    - set(
        columns
    )
)

if missing:
    raise SystemExit(
        "FAIL: member manifest lacks columns: "
        + ";".join(
            missing
        )
    )

if len(raw_rows) != 156:
    raise SystemExit(
        f"FAIL: expected 156 total member entries, "
        f"observed {len(raw_rows)}."
    )

role_rows: list[
    dict[str, Any]
] = []

for row in raw_rows:

    is_directory = as_bool(
        row[
            "is_directory"
        ]
    )

    role, usable, action = classify_role(
        row[
            "member_name"
        ],
        is_directory,
    )

    compressed_size = int(
        row[
            "compressed_size_bytes"
        ]
    )

    uncompressed_size = int(
        row[
            "uncompressed_size_bytes"
        ]
    )

    role_rows.append(
        {
            "archive_name": (
                row[
                    "archive_name"
                ]
            ),
            "member_name": (
                row[
                    "member_name"
                ]
            ),
            "member_basename": (
                row[
                    "member_basename"
                ]
            ),
            "resolved_role": role,
            "usable_for_next_phase": (
                usable
                and not is_directory
                and not as_bool(
                    row["encrypted"]
                )
                and uncompressed_size > 0
            ),
            "recommended_action": (
                action
            ),
            "is_directory": (
                is_directory
            ),
            "encrypted": as_bool(
                row[
                    "encrypted"
                ]
            ),
            "compression_method": int(
                row[
                    "compression_method"
                ]
            ),
            "compressed_size_bytes": (
                compressed_size
            ),
            "uncompressed_size_bytes": (
                uncompressed_size
            ),
            "compressed_size_human": (
                human_size(
                    compressed_size
                )
            ),
            "uncompressed_size_human": (
                human_size(
                    uncompressed_size
                )
            ),
            "local_header_offset": int(
                row[
                    "local_header_offset"
                ]
            ),
            "crc32_hex": (
                row[
                    "crc32_hex"
                ]
            ),
        }
    )

role_columns = [
    "archive_name",
    "member_name",
    "member_basename",
    "resolved_role",
    "usable_for_next_phase",
    "recommended_action",
    "is_directory",
    "encrypted",
    "compression_method",
    "compressed_size_bytes",
    "uncompressed_size_bytes",
    "compressed_size_human",
    "uncompressed_size_human",
    "local_header_offset",
    "crc32_hex",
]

write_tsv_gz(
    role_dir
    / "phase10B5_P2_resolved_member_roles.tsv.gz",
    role_rows,
    role_columns,
)

non_directory_rows = [
    row
    for row in role_rows
    if not row[
        "is_directory"
    ]
]

role_counts = Counter(
    row[
        "resolved_role"
    ]
    for row in non_directory_rows
)

role_summary_rows = [
    {
        "resolved_role": role,
        "members": count,
        "archives_represented": len(
            {
                row["archive_name"]
                for row in non_directory_rows
                if row[
                    "resolved_role"
                ] == role
            }
        ),
        "total_compressed_size_bytes": sum(
            int(
                row[
                    "compressed_size_bytes"
                ]
            )
            for row in non_directory_rows
            if row[
                "resolved_role"
            ] == role
        ),
        "total_uncompressed_size_bytes": sum(
            int(
                row[
                    "uncompressed_size_bytes"
                ]
            )
            for row in non_directory_rows
            if row[
                "resolved_role"
            ] == role
        ),
    }
    for role, count in sorted(
        role_counts.items()
    )
]

write_tsv(
    role_dir
    / "phase10B5_P2_member_role_summary.tsv",
    role_summary_rows,
    [
        "resolved_role",
        "members",
        "archives_represented",
        "total_compressed_size_bytes",
        "total_uncompressed_size_bytes",
    ],
)

basename_counts = Counter(
    row[
        "member_basename"
    ]
    for row in non_directory_rows
)

basename_summary_rows = [
    {
        "member_basename": basename,
        "members": count,
        "archives_represented": len(
            {
                row["archive_name"]
                for row in non_directory_rows
                if row[
                    "member_basename"
                ] == basename
            }
        ),
        "resolved_roles": ";".join(
            sorted(
                {
                    row[
                        "resolved_role"
                    ]
                    for row in non_directory_rows
                    if row[
                        "member_basename"
                    ] == basename
                }
            )
        ),
    }
    for basename, count in sorted(
        basename_counts.items()
    )
]

write_tsv(
    role_dir
    / "phase10B5_P2_unique_member_basenames.tsv",
    basename_summary_rows,
    [
        "member_basename",
        "members",
        "archives_represented",
        "resolved_roles",
    ],
)

archives = sorted(
    {
        row[
            "archive_name"
        ]
        for row in role_rows
    }
)

if len(archives) != 45:
    raise SystemExit(
        f"FAIL: expected 45 archives, observed "
        f"{len(archives)}."
    )

resolution_rows: list[
    dict[str, Any]
] = []

exception_rows: list[
    dict[str, Any]
] = []

for archive_name in archives:

    archive_rows = [
        row
        for row in role_rows
        if row[
            "archive_name"
        ] == archive_name
        and not row[
            "is_directory"
        ]
    ]

    metadata_candidates = [
        row
        for row in archive_rows
        if row[
            "resolved_role"
        ] == "cell_metadata"
        and row[
            "usable_for_next_phase"
        ]
    ]

    zero_size_metadata = [
        row
        for row in archive_rows
        if row[
            "resolved_role"
        ] == "cell_metadata"
        and int(
            row[
                "uncompressed_size_bytes"
            ]
        ) == 0
    ]

    matrix_candidates = [
        row
        for row in archive_rows
        if row[
            "resolved_role"
        ] == "cell_by_gene_matrix"
        and row[
            "usable_for_next_phase"
        ]
    ]

    transcript_members = [
        row
        for row in archive_rows
        if row[
            "resolved_role"
        ] == "detected_transcripts"
    ]

    artifacts = [
        row
        for row in archive_rows
        if row[
            "resolved_role"
        ] == "filesystem_artifact"
    ]

    metadata_candidates.sort(
        key=lambda row: (
            len(
                Path(
                    row[
                        "member_name"
                    ]
                ).parts
            ),
            -int(
                row[
                    "uncompressed_size_bytes"
                ]
            ),
            row[
                "member_name"
            ],
        )
    )

    matrix_candidates.sort(
        key=lambda row: (
            len(
                Path(
                    row[
                        "member_name"
                    ]
                ).parts
            ),
            -int(
                row[
                    "uncompressed_size_bytes"
                ]
            ),
            row[
                "member_name"
            ],
        )
    )

    selected_metadata = (
        metadata_candidates[0]
        if len(
            metadata_candidates
        ) == 1
        else None
    )

    selected_matrix = (
        matrix_candidates[0]
        if len(
            matrix_candidates
        ) == 1
        else None
    )

    metadata_status = (
        "resolved"
        if selected_metadata
        else (
            "zero_byte_metadata_only"
            if (
                len(
                    metadata_candidates
                ) == 0
                and len(
                    zero_size_metadata
                ) > 0
            )
            else (
                "missing"
                if len(
                    metadata_candidates
                ) == 0
                else "multiple_candidates"
            )
        )
    )

    matrix_status = (
        "resolved"
        if selected_matrix
        else (
            "missing"
            if len(
                matrix_candidates
            ) == 0
            else "multiple_candidates"
        )
    )

    archive_stem = Path(
        archive_name
    ).stem

    donor_id = archive_stem.split(
        "_",
        1,
    )[0]

    section_token = (
        archive_stem.split(
            "_",
            1,
        )[1]
        if "_" in archive_stem
        else ""
    )

    archive_resolution = (
        "ready_for_prefix_audit"
        if (
            selected_metadata
            and selected_matrix
        )
        else (
            "matrix_ready_metadata_exception"
            if (
                selected_matrix
                and not selected_metadata
            )
            else "manual_review"
        )
    )

    resolution_rows.append(
        {
            "archive_name": archive_name,
            "donor_id_from_archive": (
                donor_id
            ),
            "section_token_from_archive": (
                section_token
            ),
            "selected_metadata_member": (
                selected_metadata[
                    "member_name"
                ]
                if selected_metadata
                else ""
            ),
            "selected_metadata_compressed_size_bytes": (
                selected_metadata[
                    "compressed_size_bytes"
                ]
                if selected_metadata
                else 0
            ),
            "selected_metadata_uncompressed_size_bytes": (
                selected_metadata[
                    "uncompressed_size_bytes"
                ]
                if selected_metadata
                else 0
            ),
            "metadata_resolution_status": (
                metadata_status
            ),
            "selected_cell_by_gene_member": (
                selected_matrix[
                    "member_name"
                ]
                if selected_matrix
                else ""
            ),
            "selected_matrix_compressed_size_bytes": (
                selected_matrix[
                    "compressed_size_bytes"
                ]
                if selected_matrix
                else 0
            ),
            "selected_matrix_uncompressed_size_bytes": (
                selected_matrix[
                    "uncompressed_size_bytes"
                ]
                if selected_matrix
                else 0
            ),
            "matrix_resolution_status": (
                matrix_status
            ),
            "detected_transcript_members": len(
                transcript_members
            ),
            "filesystem_artifacts_excluded": len(
                artifacts
            ),
            "archive_resolution": (
                archive_resolution
            ),
        }
    )

    if metadata_status != "resolved":

        exception_rows.append(
            {
                "archive_name": archive_name,
                "exception_type": (
                    "metadata_resolution"
                ),
                "exception_status": (
                    metadata_status
                ),
                "candidate_members": ";".join(
                    row[
                        "member_name"
                    ]
                    for row in (
                        metadata_candidates
                        + zero_size_metadata
                    )
                ),
                "recommended_action": (
                    "retain_archive_for_matrix_panel_audit_"
                    "and_resolve_age_metadata_externally"
                ),
            }
        )

    if matrix_status != "resolved":

        exception_rows.append(
            {
                "archive_name": archive_name,
                "exception_type": (
                    "matrix_resolution"
                ),
                "exception_status": (
                    matrix_status
                ),
                "candidate_members": ";".join(
                    row[
                        "member_name"
                    ]
                    for row in matrix_candidates
                ),
                "recommended_action": (
                    "manual_review_before_any_expression_download"
                ),
            }
        )

resolution_columns = [
    "archive_name",
    "donor_id_from_archive",
    "section_token_from_archive",
    "selected_metadata_member",
    "selected_metadata_compressed_size_bytes",
    "selected_metadata_uncompressed_size_bytes",
    "metadata_resolution_status",
    "selected_cell_by_gene_member",
    "selected_matrix_compressed_size_bytes",
    "selected_matrix_uncompressed_size_bytes",
    "matrix_resolution_status",
    "detected_transcript_members",
    "filesystem_artifacts_excluded",
    "archive_resolution",
]

write_tsv(
    resolution_dir
    / "phase10B5_P2_archive_member_resolution.tsv",
    resolution_rows,
    resolution_columns,
)

write_tsv(
    exception_dir
    / "phase10B5_P2_member_resolution_exceptions.tsv",
    exception_rows,
    [
        "archive_name",
        "exception_type",
        "exception_status",
        "candidate_members",
        "recommended_action",
    ],
)

artifact_rows = [
    row
    for row in role_rows
    if row[
        "resolved_role"
    ] == "filesystem_artifact"
]

write_tsv(
    exception_dir
    / "phase10B5_P2_excluded_filesystem_artifacts.tsv",
    artifact_rows,
    role_columns,
)

metadata_resolved = sum(
    row[
        "metadata_resolution_status"
    ] == "resolved"
    for row in resolution_rows
)

matrix_resolved = sum(
    row[
        "matrix_resolution_status"
    ] == "resolved"
    for row in resolution_rows
)

matrix_ready_metadata_exception = sum(
    row[
        "archive_resolution"
    ] == "matrix_ready_metadata_exception"
    for row in resolution_rows
)

archives_ready = sum(
    row[
        "archive_resolution"
    ] == "ready_for_prefix_audit"
    for row in resolution_rows
)

transcript_archives = sum(
    int(
        row[
            "detected_transcript_members"
        ]
    ) >= 1
    for row in resolution_rows
)

status_value = (
    "passed_phase10B5_P2_member_role_resolution_"
    "ready_for_remote_matrix_header_and_metadata_prefix_audit"
    if (
        matrix_resolved == 45
        and metadata_resolved >= 44
        and transcript_archives == 45
    )
    else (
        "phase10B5_P2_requires_manual_review"
    )
)

status = {
    "phase": "phase10B5_P2",
    "remote_archives": len(
        archives
    ),
    "ZIP_file_members_resolved": len(
        non_directory_rows
    ),
    "cell_metadata_members_total": (
        role_counts.get(
            "cell_metadata",
            0,
        )
    ),
    "usable_metadata_members_resolved": (
        metadata_resolved
    ),
    "cell_by_gene_members_total": (
        role_counts.get(
            "cell_by_gene_matrix",
            0,
        )
    ),
    "cell_by_gene_archives_resolved": (
        matrix_resolved
    ),
    "detected_transcript_members_total": (
        role_counts.get(
            "detected_transcripts",
            0,
        )
    ),
    "archives_with_detected_transcripts": (
        transcript_archives
    ),
    "filesystem_artifacts_excluded": len(
        artifact_rows
    ),
    "archives_ready_for_metadata_and_header_prefix_audit": (
        archives_ready
    ),
    "matrix_ready_archives_with_metadata_exception": (
        matrix_ready_metadata_exception
    ),
    "network_requests_performed": False,
    "archive_payload_downloaded": False,
    "expression_values_accessed": False,
    "module_scores_computed": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P2_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B5_P2_status.tsv",
    [
        status
    ],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P2 MERFISH "
    "MEMBER-ROLE RESOLUTION =====",
    "",
    (
        "Remote archives: "
        f"{len(archives)}"
    ),
    (
        "ZIP file members resolved: "
        f"{len(non_directory_rows)}"
    ),
    (
        "Usable metadata members resolved: "
        f"{metadata_resolved}"
    ),
    (
        "Cell-by-gene archives resolved: "
        f"{matrix_resolved}"
    ),
    (
        "Archives with detected-transcript members: "
        f"{transcript_archives}"
    ),
    (
        "Filesystem artifacts excluded: "
        f"{len(artifact_rows)}"
    ),
    (
        "Archives ready for metadata and matrix-header "
        f"prefix audit: {archives_ready}"
    ),
    (
        "Matrix-ready archives with metadata exception: "
        f"{matrix_ready_metadata_exception}"
    ),
    "",
    "Network requests performed: FALSE",
    "Archive payload downloaded: FALSE",
    "Expression values accessed: FALSE",
    "Module scores computed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P2 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P2_report.txt"
).write_text(
    "\n".join(report)
    + "\n",
    encoding="utf-8",
)

print(
    "\n".join(report)
)

print(
    "\n===== MEMBER ROLE SUMMARY ====="
)

for row in role_summary_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "resolved_role",
                "members",
                "archives_represented",
                "total_compressed_size_bytes",
                "total_uncompressed_size_bytes",
            )
        )
    )

print(
    "\n===== ARCHIVE RESOLUTION EXCEPTIONS ====="
)

for row in exception_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "archive_name",
                "exception_type",
                "exception_status",
                "candidate_members",
                "recommended_action",
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
        != "phase10B5_P2_SHA256.tsv"
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
    / "phase10B5_P2_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)
