from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import urllib.request
from pathlib import Path


PROJECT = Path(
    "."
)

SPATIAL_ROOT = (
    PROJECT
    / "12_validation_package"
    / "phase10B_spatial_replication"
)

OUTPUT = (
    SPATIAL_ROOT
    / "phase10B4_P0_preflight"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


def write_tsv(
    path: Path,
    rows: list[dict],
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
            extrasaction="ignore",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)


def open_text(
    path: Path,
):

    if path.suffix.lower() == ".gz":
        return gzip.open(
            path,
            "rt",
            encoding="utf-8",
            errors="replace",
        )

    return path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    )


def md5_file(
    path: Path,
) -> str:

    digest = hashlib.md5()

    with path.open("rb") as handle:

        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


name_pattern = re.compile(
    r"(dthi|module|gene[_-]?set|geneset|seed)",
    flags=re.IGNORECASE,
)

accepted_gene_columns = {
    "gene",
    "genes",
    "gene_symbol",
    "gene_symbols",
    "symbol",
    "symbols",
    "hgnc_symbol",
    "module_gene",
    "member_gene",
}

excluded_parts = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "01_raw_data",
    "archive",
    "archives",
    "phase10B4_P0_preflight",
}

module_candidates = []

for path in PROJECT.rglob("*"):

    if not path.is_file():
        continue

    relative_path = path.relative_to(
        PROJECT
    )

    if (
        set(relative_path.parts)
        & excluded_parts
    ):
        continue

    if not name_pattern.search(
        path.name
    ):
        continue

    logical_suffix = path.suffix.lower()

    if logical_suffix == ".gz":
        logical_suffix = Path(
            path.stem
        ).suffix.lower()

    if logical_suffix not in {
        ".tsv",
        ".csv",
        ".txt",
    }:
        continue

    if path.stat().st_size > (
        100 * 1024 * 1024
    ):
        continue

    try:

        with open_text(path) as handle:
            header = handle.readline().rstrip(
                "\r\n"
            )

        delimiter = (
            "\t"
            if header.count("\t")
            >= header.count(",")
            else ","
        )

        columns = [
            value.strip().strip('"')
            for value in header.split(
                delimiter
            )
        ]

        normalized_columns = {
            re.sub(
                r"[^a-z0-9]+",
                "_",
                value.lower(),
            ).strip("_")
            for value in columns
        }

        matched_gene_columns = sorted(
            normalized_columns
            & accepted_gene_columns
        )

        confidence = (
            "high"
            if matched_gene_columns
            else "filename_only"
        )

        module_candidates.append(
            {
                "confidence": confidence,
                "file_name": path.name,
                "size_bytes": path.stat().st_size,
                "matched_gene_columns": ";".join(
                    matched_gene_columns
                ),
                "header_columns": ";".join(
                    columns[:30]
                ),
                "project_relative_path": (
                    relative_path.as_posix()
                ),
            }
        )

    except Exception as error:

        module_candidates.append(
            {
                "confidence": "unreadable_header",
                "file_name": path.name,
                "size_bytes": path.stat().st_size,
                "matched_gene_columns": "",
                "header_columns": (
                    f"ERROR:{type(error).__name__}"
                ),
                "project_relative_path": (
                    relative_path.as_posix()
                ),
            }
        )


module_candidates.sort(
    key=lambda row: (
        {
            "high": 0,
            "filename_only": 1,
            "unreadable_header": 2,
        }.get(
            row["confidence"],
            9,
        ),
        row["project_relative_path"],
    )
)

write_tsv(
    OUTPUT
    / "phase10B4_P0_module_candidate_inventory.tsv",
    module_candidates,
    [
        "confidence",
        "file_name",
        "size_bytes",
        "matched_gene_columns",
        "header_columns",
        "project_relative_path",
    ],
)


record_rows = []
remote_file_rows = []
zenodo_errors = []

for requested_record_id in (
    "14422018",
    "15127709",
):

    api_url = (
        "https://zenodo.org/api/records/"
        f"{requested_record_id}"
    )

    try:

        request = urllib.request.Request(
            api_url,
            headers={
                "User-Agent": (
                    "DTHI-Struct-Phase10B4-"
                    "Preflight/1.0"
                )
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=60,
        ) as response:

            record = json.load(
                response
            )

        metadata = record.get(
            "metadata",
            {},
        )

        files = record.get(
            "files",
            [],
        )

        resolved_record_id = record.get(
            "id",
            requested_record_id,
        )

        record_rows.append(
            {
                "requested_record_id": (
                    requested_record_id
                ),
                "resolved_record_id": (
                    resolved_record_id
                ),
                "title": metadata.get(
                    "title",
                    "",
                ),
                "doi": metadata.get(
                    "doi",
                    "",
                ),
                "file_count": len(files),
                "api_reached": True,
            }
        )

        for item in files:

            file_name = item.get(
                "key",
                "",
            )

            file_name_lower = (
                file_name.lower()
            )

            gestational_ages = sorted(
                set(
                    re.findall(
                        r"GW[_ -]?(\d{1,2})",
                        file_name,
                        flags=re.IGNORECASE,
                    )
                )
            )

            remote_file_rows.append(
                {
                    "record_id": (
                        resolved_record_id
                    ),
                    "file_name": file_name,
                    "size_bytes": item.get(
                        "size",
                        "",
                    ),
                    "checksum": item.get(
                        "checksum",
                        "",
                    ),
                    "possible_GW_from_filename": (
                        ";".join(
                            gestational_ages
                        )
                    ),
                    "metadata_like": bool(
                        re.search(
                            (
                                r"meta|sample|annotation|"
                                r"cluster|cell[_-]?type|"
                                r"manifest"
                            ),
                            file_name_lower,
                        )
                    ),
                    "MERFISH_like": (
                        "merfish"
                        in file_name_lower
                        or "merscope"
                        in file_name_lower
                    ),
                    "download_url": (
                        item.get(
                            "links",
                            {},
                        ).get(
                            "content",
                            "",
                        )
                        or item.get(
                            "links",
                            {},
                        ).get(
                            "self",
                            "",
                        )
                    ),
                }
            )

    except Exception as error:

        zenodo_errors.append(
            (
                f"{requested_record_id}: "
                f"{type(error).__name__}: "
                f"{error}"
            )
        )

        record_rows.append(
            {
                "requested_record_id": (
                    requested_record_id
                ),
                "resolved_record_id": "",
                "title": "",
                "doi": "",
                "file_count": 0,
                "api_reached": False,
            }
        )


write_tsv(
    OUTPUT
    / "phase10B4_P0_Zenodo_record_inventory.tsv",
    record_rows,
    [
        "requested_record_id",
        "resolved_record_id",
        "title",
        "doi",
        "file_count",
        "api_reached",
    ],
)

write_tsv(
    OUTPUT
    / "phase10B4_P0_Zenodo_file_inventory.tsv",
    remote_file_rows,
    [
        "record_id",
        "file_name",
        "size_bytes",
        "checksum",
        "possible_GW_from_filename",
        "metadata_like",
        "MERFISH_like",
        "download_url",
    ],
)

(
    OUTPUT
    / "phase10B4_P0_Zenodo_errors.txt"
).write_text(
    "\n".join(
        zenodo_errors
    )
    + (
        "\n"
        if zenodo_errors
        else ""
    ),
    encoding="utf-8",
)


assay_inventory_path = (
    OUTPUT
    / "phase10B4_P0_object_assay_inventory.tsv"
)

assay_rows = []

if assay_inventory_path.exists():

    with assay_inventory_path.open(
        encoding="utf-8",
        newline="",
    ) as handle:

        assay_rows = list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )



structurally_accessible = sum(
    str(
        row.get(
            "structural_access",
            "",
        )
    ).upper()
    == "TRUE"
    for row in assay_rows
)

high_confidence_modules = sum(
    row["confidence"] == "high"
    for row in module_candidates
)

records_reached = sum(
    bool(
        row["api_reached"]
    )
    for row in record_rows
)

object_files = [
    SPATIAL_ROOT
    / "phase10B2_primary_object_preflight"
    / "01_raw_objects"
    / "Final_Integration_MDM_100323.rds",
    SPATIAL_ROOT
    / "phase10B2_primary_object_preflight"
    / "01_raw_objects"
    / "Visium_A1_brain_011124.rds",
]

objects_present = sum(
    path.exists()
    for path in object_files
)

if (
    objects_present == 2
    and structurally_accessible >= 2
    and records_reached == 2
    and high_confidence_modules >= 1
):
    preflight_status = (
        "passed_phase10B4_P0_ready_for_"
        "module_lock_selection_and_targeted_"
        "multiage_MERFISH_download"
    )
elif (
    objects_present == 2
    and structurally_accessible >= 2
    and records_reached == 2
):
    preflight_status = (
        "completed_phase10B4_P0_requires_"
        "module_table_resolution_before_scoring"
    )
else:
    preflight_status = (
        "phase10B4_P0_requires_manual_review"
    )

status_rows = [
    {
        "phase": "phase10B4_P0",
        "phase10B3_R3_status_verified": True,
        "raw_objects_present": objects_present,
        "structurally_accessible_assay_layers_or_slots": (
            structurally_accessible
        ),
        "module_table_candidates": len(
            module_candidates
        ),
        "high_confidence_module_table_candidates": (
            high_confidence_modules
        ),
        "Zenodo_records_reached": (
            records_reached
        ),
        "Zenodo_files_inventoried": len(
            remote_file_rows
        ),
        "expression_values_summarised": False,
        "module_scores_computed": False,
        "large_remote_files_downloaded": False,
        "candidate_TFs_changed": False,
        "validation_hypotheses_changed": False,
        "phase10B4_P0_status": (
            preflight_status
        ),
    }
]

write_tsv(
    OUTPUT
    / "phase10B4_P0_status.tsv",
    status_rows,
    list(
        status_rows[0].keys()
    ),
)

report_lines = [
    "===== PHASE 10B4-P0 GUARDED PREFLIGHT =====",
    "",
    "Phase 10B3-R3 status verified: TRUE",
    f"Raw objects present: {objects_present}/2",
    (
        "Structurally accessible assay layers/slots: "
        f"{structurally_accessible}"
    ),
    (
        "Module-table candidates: "
        f"{len(module_candidates)}"
    ),
    (
        "High-confidence module-table candidates: "
        f"{high_confidence_modules}"
    ),
    (
        "Zenodo records reached: "
        f"{records_reached}/2"
    ),
    (
        "Zenodo files inventoried: "
        f"{len(remote_file_rows)}"
    ),
    "",
    "Expression values summarised: FALSE",
    "Module scores computed: FALSE",
    "Large remote files downloaded: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B4-P0 STATUS: "
        f"{preflight_status}"
    ),
]

(
    OUTPUT
    / "phase10B4_P0_report.txt"
).write_text(
    "\n".join(
        report_lines
    )
    + "\n",
    encoding="utf-8",
)

checksum_rows = []

for output_path in sorted(
    OUTPUT.rglob("*")
):

    if (
        output_path.is_file()
        and output_path.name
        != "phase10B4_P0_MD5.tsv"
    ):

        checksum_rows.append(
            {
                "md5": md5_file(
                    output_path
                ),
                "size_bytes": (
                    output_path.stat().st_size
                ),
                "project_relative_path": (
                    output_path.relative_to(
                        PROJECT
                    ).as_posix()
                ),
            }
        )

write_tsv(
    OUTPUT
    / "phase10B4_P0_MD5.tsv",
    checksum_rows,
    [
        "md5",
        "size_bytes",
        "project_relative_path",
    ],
)

print(
    "\n".join(
        report_lines
    )
)

print(
    "\n===== TOP MODULE-TABLE CANDIDATES ====="
)

if module_candidates:

    for row in module_candidates[:25]:

        print(
            f"{row['confidence']}\t"
            f"{row['matched_gene_columns'] or '-'}\t"
            f"{row['project_relative_path']}"
        )

else:
    print("NONE")

print(
    "\n===== ZENODO RECORDS ====="
)

for row in record_rows:

    print(
        f"{row['requested_record_id']}\t"
        f"api_reached={row['api_reached']}\t"
        f"files={row['file_count']}\t"
        f"{row['title']}"
    )

print(
    "\n===== METADATA-LIKE REMOTE FILES ====="
)

metadata_rows = [
    row
    for row in remote_file_rows
    if row["metadata_like"]
]

if metadata_rows:

    for row in metadata_rows[:40]:

        print(
            f"{row['record_id']}\t"
            f"{row['size_bytes']}\t"
            f"{row['file_name']}"
        )

else:
    print(
        "NONE IDENTIFIED FROM FILENAMES"
    )
