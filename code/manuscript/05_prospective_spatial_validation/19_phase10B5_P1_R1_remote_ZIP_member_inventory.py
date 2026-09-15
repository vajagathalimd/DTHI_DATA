from __future__ import annotations

import csv
import gzip
import math
import re
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


manifest_path = Path(sys.argv[1])
out = Path(sys.argv[2])

archive_dir = (
    out
    / "01_archive_audit"
)

member_dir = (
    out
    / "02_member_inventory"
)

target_dir = (
    out
    / "03_targeted_member_plan"
)

audit_dir = (
    out
    / "04_audit"
)

for directory in (
    archive_dir,
    member_dir,
    target_dir,
    audit_dir,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

tail_request_bytes = 131_072
maximum_central_directory_bytes = (
    256
    * 1024
    * 1024
)

maximum_member_entries = 500_000
maximum_target_member_size = (
    250
    * 1024
    * 1024
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


def fetch_range(
    url: str,
    start: int,
    end: int,
    attempts: int = 3,
) -> bytes:

    if start < 0 or end < start:
        raise ValueError(
            f"Invalid byte range: {start}-{end}"
        )

    expected_length = (
        end - start + 1
    )

    maximum_download_bytes = max(
        expected_length + 65_536,
        131_072,
    )

    endpoint_candidates = [
        url
    ]

    api_match = re.fullmatch(
        r"https://zenodo\.org/api/records/"
        r"(?P<record_id>\d+)/files/"
        r"(?P<filename>.+)/content",
        url,
    )

    if api_match:

        record_id = api_match.group(
            "record_id"
        )

        decoded_filename = urllib.parse.unquote(
            api_match.group(
                "filename"
            )
        )

        public_url = (
            f"https://zenodo.org/records/{record_id}/files/"
            f"{urllib.parse.quote(decoded_filename)}"
            "?download=1"
        )

        endpoint_candidates.append(
            public_url
        )

    errors: list[str] = []

    for attempt in range(
        1,
        attempts + 1,
    ):

        for endpoint_url in endpoint_candidates:

            with tempfile.TemporaryDirectory(
                prefix="dthi_zenodo_range_"
            ) as temporary_directory:

                temporary_path = Path(
                    temporary_directory
                )

                header_path = (
                    temporary_path
                    / "headers.txt"
                )

                payload_path = (
                    temporary_path
                    / "payload.bin"
                )

                command = [
                    "curl",
                    "--silent",
                    "--show-error",
                    "--location",
                    "--http1.1",
                    "--fail",
                    "--max-redirs",
                    "10",
                    "--connect-timeout",
                    "30",
                    "--max-time",
                    "120",
                    "--max-filesize",
                    str(
                        maximum_download_bytes
                    ),
                    "--user-agent",
                    (
                        "DTHI-MERFISH-RemoteZIP/"
                        "phase10B5-P1-R1"
                    ),
                    "--header",
                    "Accept: */*",
                    "--range",
                    f"{start}-{end}",
                    "--dump-header",
                    str(
                        header_path
                    ),
                    "--output",
                    str(
                        payload_path
                    ),
                    endpoint_url,
                ]

                completed = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )

                header_text = (
                    header_path.read_text(
                        encoding="iso-8859-1",
                        errors="replace",
                    )
                    if header_path.exists()
                    else ""
                )

                response_blocks = re.split(
                    r"\r?\n\r?\n",
                    header_text.strip(),
                )

                status = 0
                response_headers: dict[
                    str,
                    str,
                ] = {}

                for block in reversed(
                    response_blocks
                ):

                    lines = [
                        line
                        for line in block.splitlines()
                        if line.strip()
                    ]

                    if (
                        not lines
                        or not lines[0].startswith(
                            "HTTP/"
                        )
                    ):
                        continue

                    parts = lines[0].split()

                    if len(parts) < 2:
                        continue

                    try:
                        status = int(
                            parts[1]
                        )
                    except ValueError:
                        continue

                    for line in lines[1:]:

                        if ":" not in line:
                            continue

                        key, value = line.split(
                            ":",
                            1,
                        )

                        response_headers[
                            key.strip().lower()
                        ] = value.strip()

                    break

                content_range = (
                    response_headers.get(
                        "content-range",
                        "",
                    )
                )

                range_match = re.fullmatch(
                    r"bytes\s+"
                    r"(?P<start>\d+)-"
                    r"(?P<end>\d+)/"
                    r"(?P<total>\d+|\*)",
                    content_range,
                    flags=re.IGNORECASE,
                )

                exact_content_range = bool(
                    range_match
                    and int(
                        range_match.group(
                            "start"
                        )
                    ) == start
                    and int(
                        range_match.group(
                            "end"
                        )
                    ) == end
                )

                payload_size = (
                    payload_path.stat().st_size
                    if payload_path.exists()
                    else 0
                )

                if (
                    completed.returncode == 0
                    and status == 206
                    and exact_content_range
                    and payload_size
                    == expected_length
                ):

                    return payload_path.read_bytes()

                errors.append(
                    (
                        f"attempt={attempt}; "
                        f"endpoint={endpoint_url}; "
                        f"curl_status="
                        f"{completed.returncode}; "
                        f"HTTP_status={status}; "
                        f"Content-Range="
                        f"{content_range!r}; "
                        f"bytes={payload_size}; "
                        f"stderr="
                        f"{completed.stderr.strip()!r}"
                    )
                )

        if attempt < attempts:
            time.sleep(
                attempt * 2
            )

    raise RuntimeError(
        "All safe curl byte-range attempts failed: "
        + " || ".join(
            errors
        )
    )



def parse_zip64_extra(
    extra: bytes,
    compressed_size: int,
    uncompressed_size: int,
    local_offset: int,
) -> tuple[
    int,
    int,
    int,
]:

    position = 0

    while position + 4 <= len(extra):

        header_id, data_size = struct.unpack_from(
            "<HH",
            extra,
            position,
        )

        position += 4

        data = extra[
            position:position + data_size
        ]

        position += data_size

        if header_id != 0x0001:
            continue

        data_position = 0

        if uncompressed_size == 0xFFFFFFFF:

            if data_position + 8 > len(data):
                raise ValueError(
                    "Malformed ZIP64 uncompressed-size field."
                )

            uncompressed_size = struct.unpack_from(
                "<Q",
                data,
                data_position,
            )[0]

            data_position += 8

        if compressed_size == 0xFFFFFFFF:

            if data_position + 8 > len(data):
                raise ValueError(
                    "Malformed ZIP64 compressed-size field."
                )

            compressed_size = struct.unpack_from(
                "<Q",
                data,
                data_position,
            )[0]

            data_position += 8

        if local_offset == 0xFFFFFFFF:

            if data_position + 8 > len(data):
                raise ValueError(
                    "Malformed ZIP64 local-offset field."
                )

            local_offset = struct.unpack_from(
                "<Q",
                data,
                data_position,
            )[0]

        break

    return (
        compressed_size,
        uncompressed_size,
        local_offset,
    )


def decode_filename(
    value: bytes,
    flags: int,
) -> str:

    encoding = (
        "utf-8"
        if flags & 0x0800
        else "cp437"
    )

    return value.decode(
        encoding,
        errors="replace",
    )


def classify_member(
    member_name: str,
) -> tuple[
    str,
    str,
]:

    lower = member_name.lower()
    basename = Path(lower).name

    panel_terms = (
        "codebook",
        "code_book",
        "gene_panel",
        "genepanel",
        "gene_list",
        "genelist",
        "target_gene",
        "targetgene",
        "target_list",
        "targetlist",
        "probe",
        "barcode",
        "bit_list",
        "bitlist",
    )

    expression_terms = (
        "cell_by_gene",
        "cellbygene",
        "gene_by_cell",
        "genebycell",
        "expression",
        "count_matrix",
        "countmatrix",
        "counts",
        "detected_transcript",
        "detected_transcripts",
        "transcript",
        "transcripts",
        ".h5ad",
        ".loom",
        ".zarr",
    )

    metadata_terms = (
        "cell_metadata",
        "cellmetadata",
        "metadata",
        "annotation",
        "annotations",
        "cell_type",
        "celltype",
        "cluster",
        "sample_info",
        "sampleinfo",
        "manifest",
    )

    geometry_terms = (
        "coordinate",
        "coordinates",
        "centroid",
        "boundary",
        "boundaries",
        "polygon",
        "segmentation",
        "cellpose",
        "spatial",
    )

    image_extensions = (
        ".tif",
        ".tiff",
        ".dax",
        ".png",
        ".jpg",
        ".jpeg",
        ".ome.tif",
        ".ome.tiff",
    )

    documentation_names = (
        "readme",
        "license",
        "methods",
        "description",
    )

    archive_extensions = (
        ".zip",
        ".tar",
        ".tar.gz",
        ".tgz",
        ".7z",
    )

    if any(
        term in lower
        for term in panel_terms
    ):
        return (
            "panel_or_gene_definition",
            "priority_target_member",
        )

    if any(
        term in lower
        for term in expression_terms
    ):
        return (
            "processed_expression_or_transcript_data",
            "hold_until_panel_and_sample_mapping",
        )

    if any(
        term in lower
        for term in metadata_terms
    ):
        return (
            "sample_or_cell_metadata",
            "priority_target_member",
        )

    if any(
        term in lower
        for term in geometry_terms
    ):
        return (
            "spatial_geometry_or_segmentation",
            "secondary_target_if_required",
        )

    if basename.endswith(
        image_extensions
    ):
        return (
            "microscopy_or_segmentation_image",
            "hold_large_image_payload",
        )

    if basename.endswith(
        archive_extensions
    ):
        return (
            "nested_archive",
            "manual_review_nested_archive",
        )

    if any(
        term in basename
        for term in documentation_names
    ):
        return (
            "documentation",
            "priority_target_member",
        )

    return (
        "other_member",
        "manual_review",
    )


def extract_tokens(
    archive_name: str,
    member_name: str,
) -> tuple[
    list[str],
    list[str],
]:

    combined = (
        f"{archive_name} {member_name}"
    )

    ages = sorted(
        {
            f"GW{int(value)}"
            for value in re.findall(
                r"(?i)\bgw[_-]?(\d{1,2})\b",
                combined,
            )
        },
        key=lambda value: int(
            value[2:]
        ),
    )

    panels: list[str] = []

    if re.search(
        r"(?<!\d)300(?!\d)",
        combined,
    ):
        panels.append(
            "300_gene_panel"
        )

    if re.search(
        r"(?<!\d)960(?!\d)",
        combined,
    ):
        panels.append(
            "960_gene_panel"
        )

    return (
        ages,
        panels,
    )


def inspect_remote_zip(
    archive_name: str,
    archive_size: int,
    content_url: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:

    tail_start = max(
        0,
        archive_size
        - tail_request_bytes
    )

    tail = fetch_range(
        content_url,
        tail_start,
        archive_size - 1,
    )

    eocd_signature = b"PK\x05\x06"

    search_position = len(
        tail
    )

    eocd_position = -1
    eocd_values: tuple[
        int,
        int,
        int,
        int,
        int,
        int,
        int,
    ] | None = None

    while True:

        candidate = tail.rfind(
            eocd_signature,
            0,
            search_position,
        )

        if candidate < 0:
            break

        if candidate + 22 <= len(tail):

            values = struct.unpack_from(
                "<4H2LH",
                tail,
                candidate + 4,
            )

            (
                disk_number,
                central_disk_number,
                entries_on_disk,
                total_entries,
                central_size,
                central_offset,
                comment_length,
            ) = values

            absolute_eocd = (
                tail_start
                + candidate
            )

            expected_end = (
                absolute_eocd
                + 22
                + comment_length
            )

            if expected_end == archive_size:

                eocd_position = candidate

                eocd_values = (
                    disk_number,
                    central_disk_number,
                    entries_on_disk,
                    total_entries,
                    central_size,
                    central_offset,
                    comment_length,
                )

                break

        search_position = candidate

    if (
        eocd_position < 0
        or eocd_values is None
    ):
        raise RuntimeError(
            "Could not locate a valid ZIP end-of-central-"
            "directory record."
        )

    (
        disk_number,
        central_disk_number,
        entries_on_disk,
        total_entries,
        central_size,
        central_offset,
        comment_length,
    ) = eocd_values

    if (
        disk_number != 0
        or central_disk_number != 0
        or entries_on_disk != total_entries
    ):
        raise RuntimeError(
            "Multi-disk ZIP archives are unsupported."
        )

    if (
        total_entries == 0xFFFF
        or central_size == 0xFFFFFFFF
        or central_offset == 0xFFFFFFFF
    ):
        raise RuntimeError(
            "ZIP64 end-of-directory structure requires "
            "a separate parser."
        )

    if total_entries > maximum_member_entries:
        raise RuntimeError(
            "Archive member count exceeds safety limit: "
            f"{total_entries}"
        )

    if central_size > maximum_central_directory_bytes:
        raise RuntimeError(
            "Central directory exceeds safety limit: "
            f"{central_size} bytes"
        )

    if (
        central_offset < 0
        or central_size < 0
        or central_offset
        + central_size
        > archive_size
    ):
        raise RuntimeError(
            "Central-directory offsets are invalid."
        )

    central_data = fetch_range(
        content_url,
        central_offset,
        central_offset
        + central_size
        - 1,
    )

    members: list[
        dict[str, Any]
    ] = []

    position = 0

    for member_index in range(
        1,
        total_entries + 1,
    ):

        if position + 46 > len(
            central_data
        ):
            raise RuntimeError(
                "Central directory ended before all "
                "declared members were parsed."
            )

        values = struct.unpack_from(
            "<4s6H3L5H2L",
            central_data,
            position,
        )

        (
            signature,
            version_made,
            version_needed,
            flags,
            compression_method,
            modification_time,
            modification_date,
            crc32,
            compressed_size,
            uncompressed_size,
            filename_length,
            extra_length,
            comment_length_member,
            disk_start,
            internal_attributes,
            external_attributes,
            local_header_offset,
        ) = values

        if signature != b"PK\x01\x02":
            raise RuntimeError(
                "Unexpected central-directory signature "
                f"at member {member_index}."
            )

        filename_start = (
            position + 46
        )

        filename_end = (
            filename_start
            + filename_length
        )

        extra_end = (
            filename_end
            + extra_length
        )

        comment_end = (
            extra_end
            + comment_length_member
        )

        if comment_end > len(
            central_data
        ):
            raise RuntimeError(
                "Central-directory member exceeds "
                "retrieved byte range."
            )

        filename_bytes = central_data[
            filename_start:filename_end
        ]

        extra = central_data[
            filename_end:extra_end
        ]

        member_name = decode_filename(
            filename_bytes,
            flags,
        )

        (
            compressed_size,
            uncompressed_size,
            local_header_offset,
        ) = parse_zip64_extra(
            extra,
            compressed_size,
            uncompressed_size,
            local_header_offset,
        )

        category, recommendation = classify_member(
            member_name
        )

        ages, panels = extract_tokens(
            archive_name,
            member_name,
        )

        is_directory = member_name.endswith(
            "/"
        )

        members.append(
            {
                "archive_name": archive_name,
                "archive_size_bytes": archive_size,
                "member_index": member_index,
                "member_name": member_name,
                "member_basename": Path(
                    member_name
                ).name,
                "member_depth": (
                    len(
                        [
                            part
                            for part in Path(
                                member_name
                            ).parts
                            if part not in {
                                "",
                                ".",
                            }
                        ]
                    )
                ),
                "is_directory": is_directory,
                "compression_method": (
                    compression_method
                ),
                "encrypted": bool(
                    flags & 0x0001
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
                "local_header_offset": (
                    local_header_offset
                ),
                "crc32_hex": (
                    f"{crc32:08x}"
                ),
                "category": category,
                "download_recommendation": (
                    recommendation
                ),
                "age_tokens": ";".join(
                    ages
                ),
                "panel_tokens": ";".join(
                    panels
                ),
                "target_size_le_250MB": (
                    uncompressed_size
                    <= maximum_target_member_size
                ),
            }
        )

        position = comment_end

    if len(members) != total_entries:
        raise RuntimeError(
            "Parsed member count does not match "
            "the EOCD declaration."
        )

    archive_audit = {
        "archive_name": archive_name,
        "archive_size_bytes": archive_size,
        "archive_size_human": human_size(
            archive_size
        ),
        "range_requests_supported": True,
        "tail_metadata_bytes_retrieved": len(
            tail
        ),
        "central_directory_bytes_retrieved": (
            len(
                central_data
            )
        ),
        "total_remote_metadata_bytes_retrieved": (
            len(tail)
            + len(central_data)
        ),
        "declared_members": total_entries,
        "parsed_members": len(
            members
        ),
        "central_directory_offset": (
            central_offset
        ),
        "central_directory_size": (
            central_size
        ),
        "ZIP_comment_bytes": (
            comment_length
        ),
        "archive_payload_downloaded": False,
        "inspection_status": (
            "passed_remote_central_directory_parse"
        ),
        "inspection_error": "",
    }

    return (
        archive_audit,
        members,
    )


remote_rows, remote_columns = read_tsv(
    manifest_path
)

required_columns = {
    "file_name",
    "size_bytes",
    "content_url",
    "extension",
}

missing = sorted(
    required_columns
    - set(
        remote_columns
    )
)

if missing:
    raise SystemExit(
        "FAIL: P1 manifest lacks columns: "
        + ";".join(
            missing
        )
    )

zip_rows = [
    row
    for row in remote_rows
    if row.get(
        "extension",
        ""
    ).lower() == ".zip"
]

if len(zip_rows) != 45:
    raise SystemExit(
        "FAIL: expected 45 remote ZIP archives, "
        f"observed {len(zip_rows)}."
    )

archive_audit_rows: list[
    dict[str, Any]
] = []

member_rows: list[
    dict[str, Any]
] = []

for archive_number, row in enumerate(
    zip_rows,
    start=1,
):

    archive_name = row[
        "file_name"
    ]

    archive_size = int(
        row[
            "size_bytes"
        ]
    )

    content_url = row[
        "content_url"
    ]

    print(
        f"[{archive_number:02d}/45] "
        f"Inspecting {archive_name}"
    )

    try:

        archive_audit, members = (
            inspect_remote_zip(
                archive_name,
                archive_size,
                content_url,
            )
        )

        archive_audit_rows.append(
            archive_audit
        )

        member_rows.extend(
            members
        )

        print(
            "  PASS: "
            f"{len(members)} members; "
            f"central directory "
            f"{human_size(archive_audit['central_directory_size'])}"
        )

    except Exception as error:

        archive_audit_rows.append(
            {
                "archive_name": archive_name,
                "archive_size_bytes": (
                    archive_size
                ),
                "archive_size_human": human_size(
                    archive_size
                ),
                "range_requests_supported": False,
                "tail_metadata_bytes_retrieved": 0,
                "central_directory_bytes_retrieved": 0,
                "total_remote_metadata_bytes_retrieved": 0,
                "declared_members": 0,
                "parsed_members": 0,
                "central_directory_offset": "",
                "central_directory_size": "",
                "ZIP_comment_bytes": "",
                "archive_payload_downloaded": False,
                "inspection_status": (
                    "failed_remote_central_directory_parse"
                ),
                "inspection_error": str(
                    error
                ),
            }
        )

        print(
            f"  FAIL: {error}"
        )

archive_columns = [
    "archive_name",
    "archive_size_bytes",
    "archive_size_human",
    "range_requests_supported",
    "tail_metadata_bytes_retrieved",
    "central_directory_bytes_retrieved",
    "total_remote_metadata_bytes_retrieved",
    "declared_members",
    "parsed_members",
    "central_directory_offset",
    "central_directory_size",
    "ZIP_comment_bytes",
    "archive_payload_downloaded",
    "inspection_status",
    "inspection_error",
]

write_tsv(
    archive_dir
    / "phase10B5_P1_R1_remote_archive_audit.tsv",
    archive_audit_rows,
    archive_columns,
)

member_columns = [
    "archive_name",
    "archive_size_bytes",
    "member_index",
    "member_name",
    "member_basename",
    "member_depth",
    "is_directory",
    "compression_method",
    "encrypted",
    "compressed_size_bytes",
    "uncompressed_size_bytes",
    "compressed_size_human",
    "uncompressed_size_human",
    "local_header_offset",
    "crc32_hex",
    "category",
    "download_recommendation",
    "age_tokens",
    "panel_tokens",
    "target_size_le_250MB",
]

write_tsv_gz(
    member_dir
    / "phase10B5_P1_R1_complete_remote_ZIP_member_manifest.tsv.gz",
    member_rows,
    member_columns,
)

non_directory_members = [
    row
    for row in member_rows
    if not row[
        "is_directory"
    ]
]

category_counts = Counter(
    row["category"]
    for row in non_directory_members
)

category_summary_rows = [
    {
        "category": category,
        "members": count,
        "archives_represented": len(
            {
                row["archive_name"]
                for row in non_directory_members
                if row["category"] == category
            }
        ),
        "total_compressed_size_bytes": sum(
            int(
                row[
                    "compressed_size_bytes"
                ]
            )
            for row in non_directory_members
            if row["category"] == category
        ),
        "total_uncompressed_size_bytes": sum(
            int(
                row[
                    "uncompressed_size_bytes"
                ]
            )
            for row in non_directory_members
            if row["category"] == category
        ),
    }
    for category, count in sorted(
        category_counts.items()
    )
]

write_tsv(
    member_dir
    / "phase10B5_P1_R1_member_category_summary.tsv",
    category_summary_rows,
    [
        "category",
        "members",
        "archives_represented",
        "total_compressed_size_bytes",
        "total_uncompressed_size_bytes",
    ],
)

target_categories = {
    "panel_or_gene_definition",
    "sample_or_cell_metadata",
    "documentation",
}

target_members = [
    row
    for row in non_directory_members
    if (
        row["category"]
        in target_categories
        and row[
            "target_size_le_250MB"
        ]
        and not row[
            "encrypted"
        ]
    )
]

target_members.sort(
    key=lambda row: (
        {
            "panel_or_gene_definition": 0,
            "sample_or_cell_metadata": 1,
            "documentation": 2,
        }.get(
            row["category"],
            9,
        ),
        int(
            row[
                "uncompressed_size_bytes"
            ]
        ),
        row[
            "archive_name"
        ],
        row[
            "member_name"
        ],
    )
)

for priority, row in enumerate(
    target_members,
    start=1,
):
    row[
        "target_priority"
    ] = priority

write_tsv(
    target_dir
    / "phase10B5_P1_R1_targeted_member_candidates.tsv",
    target_members,
    [
        "target_priority",
        *member_columns,
    ],
)

expression_members = [
    row
    for row in non_directory_members
    if row["category"]
    == (
        "processed_expression_or_transcript_data"
    )
]

write_tsv(
    target_dir
    / "phase10B5_P1_R1_expression_members_held.tsv",
    expression_members,
    member_columns,
)

age_tokens = sorted(
    {
        token
        for row in non_directory_members
        for token in str(
            row["age_tokens"]
        ).split(";")
        if token
    },
    key=lambda value: int(
        value[2:]
    ),
)

panel_tokens = sorted(
    {
        token
        for row in non_directory_members
        for token in str(
            row["panel_tokens"]
        ).split(";")
        if token
    }
)

token_rows = [
    {
        "token_type": "age",
        "token": token,
        "members_containing_token": sum(
            token
            in str(
                row["age_tokens"]
            ).split(";")
            for row in non_directory_members
        ),
        "archives_containing_token": len(
            {
                row["archive_name"]
                for row in non_directory_members
                if token
                in str(
                    row["age_tokens"]
                ).split(";")
            }
        ),
    }
    for token in age_tokens
]

token_rows.extend(
    {
        "token_type": "panel",
        "token": token,
        "members_containing_token": sum(
            token
            in str(
                row["panel_tokens"]
            ).split(";")
            for row in non_directory_members
        ),
        "archives_containing_token": len(
            {
                row["archive_name"]
                for row in non_directory_members
                if token
                in str(
                    row["panel_tokens"]
                ).split(";")
            }
        ),
    }
    for token in panel_tokens
)

write_tsv(
    member_dir
    / "phase10B5_P1_R1_internal_age_and_panel_tokens.tsv",
    token_rows,
    [
        "token_type",
        "token",
        "members_containing_token",
        "archives_containing_token",
    ],
)

archives_parsed = sum(
    row[
        "inspection_status"
    ]
    == (
        "passed_remote_central_directory_parse"
    )
    for row in archive_audit_rows
)

archives_failed = (
    len(
        archive_audit_rows
    )
    - archives_parsed
)

metadata_bytes_retrieved = sum(
    int(
        row[
            "total_remote_metadata_bytes_retrieved"
        ]
    )
    for row in archive_audit_rows
)

panel_member_count = category_counts.get(
    "panel_or_gene_definition",
    0,
)

metadata_member_count = category_counts.get(
    "sample_or_cell_metadata",
    0,
)

if (
    archives_parsed == 45
    and len(
        non_directory_members
    ) > 0
    and len(
        target_members
    ) > 0
):

    status_value = (
        "passed_phase10B5_P1_R1_remote_ZIP_member_"
        "inventory_ready_for_selective_panel_metadata_"
        "member_extraction_and_sample_age_mapping"
    )

elif (
    archives_parsed == 45
    and len(
        non_directory_members
    ) > 0
):

    status_value = (
        "completed_phase10B5_P1_R1_remote_ZIP_member_"
        "inventory_requires_manual_member_classification"
    )

else:

    status_value = (
        "phase10B5_P1_R1_requires_manual_review_"
        "because_not_all_remote_ZIP_directories_were_parsed"
    )

status = {
    "phase": "phase10B5_P1_R1",
    "remote_ZIP_archives_expected": 45,
    "remote_ZIP_archives_parsed": (
        archives_parsed
    ),
    "remote_ZIP_archives_failed": (
        archives_failed
    ),
    "ZIP_members_total_including_directories": len(
        member_rows
    ),
    "ZIP_file_members": len(
        non_directory_members
    ),
    "panel_or_gene_definition_members": (
        panel_member_count
    ),
    "sample_or_cell_metadata_members": (
        metadata_member_count
    ),
    "processed_expression_members_held": len(
        expression_members
    ),
    "targeted_small_member_candidates": len(
        target_members
    ),
    "age_tokens_detected": ";".join(
        age_tokens
    ),
    "panel_tokens_detected": ";".join(
        panel_tokens
    ),
    "remote_metadata_bytes_retrieved": (
        metadata_bytes_retrieved
    ),
    "remote_metadata_size_human": human_size(
        metadata_bytes_retrieved
    ),
    "archive_payload_downloaded": False,
    "complete_ZIP_archives_downloaded": False,
    "expression_values_accessed": False,
    "module_scores_computed": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P1_R1_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B5_P1_R1_status.tsv",
    [
        status
    ],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P1-R1 REMOTE ZIP-MEMBER "
    "INVENTORY =====",
    "",
    (
        "Remote ZIP archives expected: "
        "45"
    ),
    (
        "Remote ZIP archives parsed: "
        f"{archives_parsed}"
    ),
    (
        "Remote ZIP archives failed: "
        f"{archives_failed}"
    ),
    (
        "ZIP members including directories: "
        f"{len(member_rows)}"
    ),
    (
        "ZIP file members: "
        f"{len(non_directory_members)}"
    ),
    (
        "Panel/gene-definition members: "
        f"{panel_member_count}"
    ),
    (
        "Sample/cell-metadata members: "
        f"{metadata_member_count}"
    ),
    (
        "Expression/transcript members held: "
        f"{len(expression_members)}"
    ),
    (
        "Small targeted member candidates: "
        f"{len(target_members)}"
    ),
    (
        "Age tokens detected: "
        + (
            ";".join(
                age_tokens
            )
            or "NONE"
        )
    ),
    (
        "Panel tokens detected: "
        + (
            ";".join(
                panel_tokens
            )
            or "NONE"
        )
    ),
    (
        "Remote ZIP metadata retrieved: "
        f"{human_size(metadata_bytes_retrieved)}"
    ),
    "",
    "Archive payload downloaded: FALSE",
    "Complete ZIP archives downloaded: FALSE",
    "Expression values accessed: FALSE",
    "Module scores computed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P1-R1 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P1_R1_report.txt"
).write_text(
    "\n".join(report)
    + "\n",
    encoding="utf-8",
)

print(
    "\n".join(report)
)

print(
    "\n===== MEMBER CATEGORY SUMMARY ====="
)

for row in category_summary_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "category",
                "members",
                "archives_represented",
                "total_compressed_size_bytes",
                "total_uncompressed_size_bytes",
            )
        )
    )

print(
    "\n===== TARGETED MEMBER CANDIDATES ====="
)

for row in target_members:

    print(
        "\t".join(
            [
                str(
                    row[
                        "target_priority"
                    ]
                ),
                str(
                    row[
                        "archive_name"
                    ]
                ),
                str(
                    row[
                        "member_name"
                    ]
                ),
                str(
                    row[
                        "uncompressed_size_human"
                    ]
                ),
                str(
                    row[
                        "category"
                    ]
                ),
            ]
        )
    )
