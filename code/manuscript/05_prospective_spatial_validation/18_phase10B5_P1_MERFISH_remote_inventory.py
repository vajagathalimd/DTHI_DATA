from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


project = Path(sys.argv[1])
record_id = int(sys.argv[2])
api_url = sys.argv[3]
out = Path(sys.argv[4])

metadata_dir = (
    out
    / "01_record_metadata"
)

inventory_dir = (
    out
    / "02_file_inventory"
)

download_dir = (
    out
    / "03_targeted_download_plan"
)

audit_dir = (
    out
    / "04_audit"
)

for directory in (
    metadata_dir,
    inventory_dir,
    download_dir,
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


def fetch_json(
    url: str,
    attempts: int = 3,
) -> dict[str, Any]:

    last_error: Exception | None = None

    for attempt in range(
        1,
        attempts + 1,
    ):

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "DTHI-Transcriptomics-Validation/"
                    "phase10B5-P1"
                ),
                "Accept": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=60,
            ) as response:

                payload = response.read()

            result = json.loads(
                payload.decode(
                    "utf-8"
                )
            )

            if not isinstance(
                result,
                dict,
            ):
                raise ValueError(
                    "Zenodo response was not a JSON object."
                )

            return result

        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            json.JSONDecodeError,
            ValueError,
        ) as error:

            last_error = error

            if attempt < attempts:
                time.sleep(
                    3 * attempt
                )

    raise SystemExit(
        "FAIL: could not retrieve Zenodo record after "
        f"{attempts} attempts: {last_error}"
    )


def human_size(
    size_bytes: int,
) -> str:

    value = float(
        size_bytes
    )

    units = (
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    )

    for unit in units:

        if (
            value < 1024
            or unit == units[-1]
        ):
            return (
                f"{value:.2f}_{unit}"
            )

        value /= 1024

    return f"{size_bytes}_B"


def compressed_extension(
    filename: str,
) -> str:

    lower = filename.lower()

    compound_extensions = (
        ".tar.gz",
        ".tar.bz2",
        ".tar.xz",
        ".csv.gz",
        ".tsv.gz",
        ".txt.gz",
        ".json.gz",
        ".zarr.zip",
    )

    for extension in compound_extensions:

        if lower.endswith(
            extension
        ):
            return extension

    suffixes = Path(
        filename
    ).suffixes

    return (
        "".join(suffixes[-2:])
        if len(suffixes) >= 2
        else (
            suffixes[-1]
            if suffixes
            else ""
        )
    )


def classify_file(
    filename: str,
) -> tuple[
    str,
    str,
]:

    lower = filename.lower()

    panel_terms = (
        "panel",
        "gene_list",
        "genelist",
        "codebook",
        "barcode",
        "probe",
        "target_gene",
        "targetgene",
    )

    metadata_terms = (
        "metadata",
        "meta_data",
        "sample",
        "annotation",
        "cluster",
        "celltype",
        "cell_type",
        "label",
        "manifest",
        "readme",
    )

    expression_terms = (
        "expression",
        "matrix",
        "cell_by_gene",
        "cellbygene",
        "counts",
        "transcript",
        "h5ad",
        "loom",
        "zarr",
    )

    geometry_terms = (
        "coordinate",
        "spatial",
        "boundary",
        "polygon",
        "image",
        "dapi",
        "mosaic",
        "segmentation",
    )

    if any(
        term in lower
        for term in panel_terms
    ):
        return (
            "panel_or_gene_definition",
            "candidate_for_small_targeted_download",
        )

    if any(
        term in lower
        for term in metadata_terms
    ):
        return (
            "sample_or_annotation_metadata",
            "candidate_for_small_targeted_download",
        )

    if any(
        term in lower
        for term in expression_terms
    ):
        return (
            "processed_expression_or_transcript_data",
            "hold_until_panel_coverage_and_sample_review",
        )

    if any(
        term in lower
        for term in geometry_terms
    ):
        return (
            "image_or_spatial_geometry",
            "hold_unless_required_for_followup_analysis",
        )

    return (
        "unclassified_manual_review",
        "manual_review_before_download",
    )


def age_tokens(
    filename: str,
) -> list[str]:

    tokens = re.findall(
        r"(?i)\bgw[_-]?(\d{1,2})\b",
        filename,
    )

    return sorted(
        {
            f"GW{int(token)}"
            for token in tokens
        },
        key=lambda value: int(
            value[2:]
        ),
    )


def panel_tokens(
    filename: str,
) -> list[str]:

    lower = filename.lower()

    tokens: list[str] = []

    if re.search(
        r"(?<!\d)300(?!\d)",
        lower,
    ):
        tokens.append(
            "300_gene_panel"
        )

    if re.search(
        r"(?<!\d)960(?!\d)",
        lower,
    ):
        tokens.append(
            "960_gene_panel"
        )

    return tokens


record = fetch_json(
    api_url
)

observed_id = int(
    record.get(
        "id",
        -1,
    )
)

if observed_id != record_id:
    raise SystemExit(
        f"FAIL: retrieved record ID {observed_id}; "
        f"expected {record_id}."
    )

metadata = record.get(
    "metadata",
    {}
)

title = str(
    metadata.get(
        "title",
        ""
    )
)

if (
    "MERFISH" not in title.upper()
    and "SPATIAL TRANSCRIPTOMICS" not in title.upper()
):
    raise SystemExit(
        "FAIL: retrieved Zenodo title does not appear "
        "to be the expected MERFISH dataset."
    )

files = record.get(
    "files",
    []
)

if not isinstance(
    files,
    list,
) or not files:
    raise SystemExit(
        "FAIL: Zenodo record contains no file inventory."
    )

raw_json_path = (
    metadata_dir
    / "phase10B5_P1_Zenodo_record_15127709.json"
)

raw_json_path.write_text(
    json.dumps(
        record,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

record_summary = {
    "record_id": observed_id,
    "title": title,
    "publication_date": metadata.get(
        "publication_date",
        "",
    ),
    "upload_type": metadata.get(
        "upload_type",
        "",
    ),
    "version": metadata.get(
        "version",
        "",
    ),
    "doi": metadata.get(
        "doi",
        "",
    ),
    "file_count": len(
        files
    ),
    "total_size_bytes": sum(
        int(
            file.get(
                "size",
                0,
            )
        )
        for file in files
    ),
    "record_api_url": api_url,
}

record_summary[
    "total_size_human"
] = human_size(
    int(
        record_summary[
            "total_size_bytes"
        ]
    )
)

write_tsv(
    metadata_dir
    / "phase10B5_P1_record_summary.tsv",
    [
        record_summary
    ],
    list(
        record_summary.keys()
    ),
)

manifest_rows: list[
    dict[str, Any]
] = []

for file_index, file_entry in enumerate(
    files,
    start=1,
):

    filename = str(
        file_entry.get(
            "key",
            "",
        )
    )

    if not filename:
        raise SystemExit(
            "FAIL: Zenodo file entry lacks a filename."
        )

    size_bytes = int(
        file_entry.get(
            "size",
            0,
        )
    )

    checksum = str(
        file_entry.get(
            "checksum",
            "",
        )
    )

    links = file_entry.get(
        "links",
        {}
    )

    category, recommendation = classify_file(
        filename
    )

    ages = age_tokens(
        filename
    )

    panels = panel_tokens(
        filename
    )

    manifest_rows.append(
        {
            "file_index": file_index,
            "file_name": filename,
            "extension": compressed_extension(
                filename
            ),
            "size_bytes": size_bytes,
            "size_human": human_size(
                size_bytes
            ),
            "checksum": checksum,
            "category": category,
            "download_recommendation": (
                recommendation
            ),
            "age_tokens_from_filename": (
                ";".join(
                    ages
                )
            ),
            "panel_tokens_from_filename": (
                ";".join(
                    panels
                )
            ),
            "small_file_le_50MB": (
                size_bytes
                <= 50
                * 1024
                * 1024
            ),
            "small_file_le_250MB": (
                size_bytes
                <= 250
                * 1024
                * 1024
            ),
            "content_url": links.get(
                "content",
                links.get(
                    "self",
                    "",
                ),
            ),
        }
    )

manifest_columns = [
    "file_index",
    "file_name",
    "extension",
    "size_bytes",
    "size_human",
    "checksum",
    "category",
    "download_recommendation",
    "age_tokens_from_filename",
    "panel_tokens_from_filename",
    "small_file_le_50MB",
    "small_file_le_250MB",
    "content_url",
]

write_tsv(
    inventory_dir
    / "phase10B5_P1_complete_Zenodo_file_manifest.tsv",
    manifest_rows,
    manifest_columns,
)

category_counts = Counter(
    row["category"]
    for row in manifest_rows
)

category_summary_rows = [
    {
        "category": category,
        "files": count,
        "total_size_bytes": sum(
            int(row["size_bytes"])
            for row in manifest_rows
            if row["category"] == category
        ),
        "total_size_human": human_size(
            sum(
                int(row["size_bytes"])
                for row in manifest_rows
                if row["category"] == category
            )
        ),
    }
    for category, count in sorted(
        category_counts.items()
    )
]

write_tsv(
    inventory_dir
    / "phase10B5_P1_file_category_summary.tsv",
    category_summary_rows,
    [
        "category",
        "files",
        "total_size_bytes",
        "total_size_human",
    ],
)

all_age_tokens = sorted(
    {
        age
        for row in manifest_rows
        for age in str(
            row[
                "age_tokens_from_filename"
            ]
        ).split(";")
        if age
    },
    key=lambda value: int(
        value[2:]
    ),
)

all_panel_tokens = sorted(
    {
        panel
        for row in manifest_rows
        for panel in str(
            row[
                "panel_tokens_from_filename"
            ]
        ).split(";")
        if panel
    }
)

age_panel_rows = [
    {
        "age_or_panel_token_type": "age",
        "token": age,
        "files_containing_token": sum(
            age
            in str(
                row[
                    "age_tokens_from_filename"
                ]
            ).split(";")
            for row in manifest_rows
        ),
    }
    for age in all_age_tokens
]

age_panel_rows.extend(
    {
        "age_or_panel_token_type": "panel",
        "token": panel,
        "files_containing_token": sum(
            panel
            in str(
                row[
                    "panel_tokens_from_filename"
                ]
            ).split(";")
            for row in manifest_rows
        ),
    }
    for panel in all_panel_tokens
)

write_tsv(
    inventory_dir
    / "phase10B5_P1_filename_age_and_panel_tokens.tsv",
    age_panel_rows,
    [
        "age_or_panel_token_type",
        "token",
        "files_containing_token",
    ],
)

small_review_candidates = [
    row
    for row in manifest_rows
    if (
        row["category"]
        in {
            "panel_or_gene_definition",
            "sample_or_annotation_metadata",
        }
        and bool(
            row["small_file_le_250MB"]
        )
    )
]

small_review_candidates.sort(
    key=lambda row: (
        0
        if row["category"]
        == "panel_or_gene_definition"
        else 1,
        int(
            row["size_bytes"]
        ),
        str(
            row["file_name"]
        ),
    )
)

for priority, row in enumerate(
    small_review_candidates,
    start=1,
):
    row[
        "download_priority"
    ] = priority

candidate_columns = [
    "download_priority",
    *manifest_columns,
]

write_tsv(
    download_dir
    / "phase10B5_P1_small_panel_and_metadata_candidates.tsv",
    small_review_candidates,
    candidate_columns,
)

held_expression_files = [
    row
    for row in manifest_rows
    if row["category"]
    == (
        "processed_expression_or_transcript_data"
    )
]

write_tsv(
    download_dir
    / "phase10B5_P1_expression_files_held_from_download.tsv",
    held_expression_files,
    manifest_columns,
)

file_count_matches_previous_preflight = (
    len(files) == 45
)

status_value = (
    "passed_phase10B5_P1_MERFISH_remote_inventory_"
    "ready_for_targeted_panel_and_metadata_download"
    if (
        observed_id == record_id
        and len(files) > 0
    )
    else (
        "phase10B5_P1_requires_manual_review"
    )
)

status = {
    "phase": "phase10B5_P1",
    "Zenodo_record_id": record_id,
    "Zenodo_record_title": title,
    "record_files": len(
        files
    ),
    "record_total_size_bytes": (
        record_summary[
            "total_size_bytes"
        ]
    ),
    "record_file_count_matches_previous_P0_inventory_45": (
        file_count_matches_previous_preflight
    ),
    "panel_or_gene_definition_files": (
        category_counts.get(
            "panel_or_gene_definition",
            0,
        )
    ),
    "sample_or_annotation_metadata_files": (
        category_counts.get(
            "sample_or_annotation_metadata",
            0,
        )
    ),
    "processed_expression_files_held": len(
        held_expression_files
    ),
    "small_panel_or_metadata_candidates": len(
        small_review_candidates
    ),
    "age_tokens_detected": (
        ";".join(
            all_age_tokens
        )
    ),
    "panel_tokens_detected": (
        ";".join(
            all_panel_tokens
        )
    ),
    "large_remote_files_downloaded": False,
    "expression_values_accessed": False,
    "module_scores_computed": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P1_status": status_value,
}

write_tsv(
    out
    / "phase10B5_P1_status.tsv",
    [
        status
    ],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P1 MERFISH "
    "REMOTE-INVENTORY PREFLIGHT =====",
    "",
    (
        "Zenodo record: "
        f"{record_id}"
    ),
    (
        "Record title: "
        f"{title}"
    ),
    (
        "Record files: "
        f"{len(files)}"
    ),
    (
        "Record total size: "
        f"{record_summary['total_size_human']}"
    ),
    (
        "File count matches previous P0 inventory "
        f"of 45: {file_count_matches_previous_preflight}"
    ),
    (
        "Panel/gene-definition files: "
        f"{category_counts.get('panel_or_gene_definition', 0)}"
    ),
    (
        "Sample/annotation metadata files: "
        f"{category_counts.get('sample_or_annotation_metadata', 0)}"
    ),
    (
        "Expression/transcript files held: "
        f"{len(held_expression_files)}"
    ),
    (
        "Small targeted review candidates: "
        f"{len(small_review_candidates)}"
    ),
    (
        "Age tokens detected: "
        + (
            ";".join(
                all_age_tokens
            )
            or "NONE"
        )
    ),
    (
        "Panel tokens detected: "
        + (
            ";".join(
                all_panel_tokens
            )
            or "NONE"
        )
    ),
    "",
    "Large remote files downloaded: FALSE",
    "Expression values accessed: FALSE",
    "Module scores computed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P1 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P1_report.txt"
).write_text(
    "\n".join(report)
    + "\n",
    encoding="utf-8",
)

print(
    "\n".join(report)
)

print(
    "\n===== COMPLETE FILE MANIFEST ====="
)

print(
    "\t".join(
        [
            "file_index",
            "file_name",
            "size_human",
            "category",
            "age_tokens",
            "panel_tokens",
            "recommendation",
        ]
    )
)

for row in manifest_rows:

    print(
        "\t".join(
            [
                str(
                    row["file_index"]
                ),
                str(
                    row["file_name"]
                ),
                str(
                    row["size_human"]
                ),
                str(
                    row["category"]
                ),
                str(
                    row[
                        "age_tokens_from_filename"
                    ]
                ),
                str(
                    row[
                        "panel_tokens_from_filename"
                    ]
                ),
                str(
                    row[
                        "download_recommendation"
                    ]
                ),
            ]
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
        != "phase10B5_P1_SHA256.tsv"
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
    / "phase10B5_P1_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)
