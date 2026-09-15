from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import re
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


project = Path(sys.argv[1])
remote_manifest_path = Path(sys.argv[2])
member_manifest_path = Path(sys.argv[3])
resolution_path = Path(sys.argv[4])
module_lock_path = Path(sys.argv[5])
out = Path(sys.argv[6])

matrix_dir = (
    out
    / "01_matrix_header_audit"
)

matrix_raw_dir = (
    matrix_dir
    / "raw_headers"
)

metadata_dir = (
    out
    / "02_metadata_prefix_audit"
)

metadata_raw_dir = (
    metadata_dir
    / "raw_prefixes"
)

coverage_dir = (
    out
    / "03_module_panel_coverage"
)

mapping_dir = (
    out
    / "04_sample_age_area_mapping"
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
    matrix_dir,
    matrix_raw_dir,
    metadata_dir,
    metadata_raw_dir,
    coverage_dir,
    mapping_dir,
    exception_dir,
    audit_dir,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

matrix_required_complete_lines = 1
metadata_required_complete_lines = 30

range_chunk_bytes = 262_144

matrix_max_compressed_bytes = (
    4
    * 1024
    * 1024
)

metadata_max_compressed_bytes = (
    8
    * 1024
    * 1024
)

global_remote_byte_limit = (
    512
    * 1024
    * 1024
)

network_bytes_retrieved = 0


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


def parse_final_headers(
    header_text: str,
) -> tuple[
    int,
    dict[str, str],
]:

    blocks = re.split(
        r"\r?\n\r?\n",
        header_text.strip(),
    )

    for block in reversed(
        blocks
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

        headers: dict[str, str] = {}

        for line in lines[1:]:

            if ":" not in line:
                continue

            key, value = line.split(
                ":",
                1,
            )

            headers[
                key.strip().lower()
            ] = value.strip()

        return (
            status,
            headers,
        )

    return (
        0,
        {},
    )


def endpoint_candidates(
    url: str,
) -> list[str]:

    candidates = [
        url
    ]

    match = re.fullmatch(
        r"https://zenodo\.org/api/records/"
        r"(?P<record_id>\d+)/files/"
        r"(?P<filename>.+)/content",
        url,
    )

    if match:

        record_id = match.group(
            "record_id"
        )

        filename = urllib.parse.unquote(
            match.group(
                "filename"
            )
        )

        public_url = (
            f"https://zenodo.org/records/{record_id}/files/"
            f"{urllib.parse.quote(filename)}"
            "?download=1"
        )

        candidates.append(
            public_url
        )

    return candidates


def fetch_range(
    url: str,
    start: int,
    end: int,
    attempts: int = 3,
) -> bytes:

    global network_bytes_retrieved

    if start < 0 or end < start:
        raise ValueError(
            f"Invalid byte range: {start}-{end}"
        )

    expected_length = (
        end - start + 1
    )

    errors: list[str] = []

    for attempt in range(
        1,
        attempts + 1,
    ):

        for endpoint_url in endpoint_candidates(
            url
        ):

            with tempfile.TemporaryDirectory(
                prefix="dthi_merfish_prefix_"
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
                    "--limit-rate",
                    "4M",
                    "--max-filesize",
                    str(
                        expected_length
                        + 65_536
                    ),
                    "--user-agent",
                    (
                        "DTHI-MERFISH-PrefixAudit/"
                        "phase10B5-P3"
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

                status, headers = parse_final_headers(
                    header_text
                )

                content_range = headers.get(
                    "content-range",
                    "",
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

                    payload = payload_path.read_bytes()

                    network_bytes_retrieved += len(
                        payload
                    )

                    if (
                        network_bytes_retrieved
                        > global_remote_byte_limit
                    ):
                        raise RuntimeError(
                            "Global remote-byte safety limit "
                            "was exceeded."
                        )

                    return payload

                errors.append(
                    (
                        f"attempt={attempt}; "
                        f"HTTP={status}; "
                        f"range={content_range!r}; "
                        f"bytes={payload_size}; "
                        f"curl={completed.returncode}; "
                        f"stderr="
                        f"{completed.stderr.strip()!r}"
                    )
                )

        if attempt < attempts:
            time.sleep(
                attempt * 2
            )

    raise RuntimeError(
        "All byte-range attempts failed: "
        + " || ".join(
            errors
        )
    )


def member_data_start(
    content_url: str,
    local_header_offset: int,
    expected_method: int,
) -> tuple[
    int,
    int,
]:

    local_header = fetch_range(
        content_url,
        local_header_offset,
        local_header_offset + 29,
    )

    if len(local_header) != 30:
        raise RuntimeError(
            "Local ZIP header length was not 30 bytes."
        )

    (
        signature,
        version_needed,
        flags,
        method,
        modification_time,
        modification_date,
        crc32,
        compressed_size_local,
        uncompressed_size_local,
        filename_length,
        extra_length,
    ) = struct.unpack(
        "<4s5H3L2H",
        local_header,
    )

    if signature != b"PK\x03\x04":
        raise RuntimeError(
            "Invalid ZIP local-file-header signature."
        )

    if method != expected_method:
        raise RuntimeError(
            "Compression method differs between local "
            "and central headers."
        )

    data_start = (
        local_header_offset
        + 30
        + filename_length
        + extra_length
    )

    return (
        data_start,
        flags,
    )


def extract_text_prefix(
    content_url: str,
    local_header_offset: int,
    compression_method: int,
    compressed_size: int,
    required_complete_lines: int,
    maximum_compressed_bytes: int,
) -> tuple[
    bytes,
    int,
    bool,
]:

    if compressed_size <= 0:
        return (
            b"",
            0,
            True,
        )

    data_start, flags = member_data_start(
        content_url,
        local_header_offset,
        compression_method,
    )

    encrypted = bool(
        flags & 0x0001
    )

    if encrypted:
        raise RuntimeError(
            "Encrypted ZIP member is unsupported."
        )

    if compression_method == 8:
        decompressor = zlib.decompressobj(
            -15
        )
    elif compression_method == 0:
        decompressor = None
    else:
        raise RuntimeError(
            "Unsupported ZIP compression method: "
            f"{compression_method}"
        )

    output = bytearray()
    compressed_bytes_retrieved = 0

    allowed_compressed_bytes = min(
        compressed_size,
        maximum_compressed_bytes,
    )

    while (
        compressed_bytes_retrieved
        < allowed_compressed_bytes
    ):

        current_size = min(
            range_chunk_bytes,
            allowed_compressed_bytes
            - compressed_bytes_retrieved,
        )

        start = (
            data_start
            + compressed_bytes_retrieved
        )

        end = (
            start
            + current_size
            - 1
        )

        compressed_chunk = fetch_range(
            content_url,
            start,
            end,
        )

        compressed_bytes_retrieved += len(
            compressed_chunk
        )

        if decompressor is None:
            decoded_chunk = compressed_chunk
        else:
            decoded_chunk = decompressor.decompress(
                compressed_chunk
            )

        output.extend(
            decoded_chunk
        )

        complete_lines = output.count(
            b"\n"
        )

        if complete_lines >= required_complete_lines:
            break

        if len(output) > (
            32
            * 1024
            * 1024
        ):
            raise RuntimeError(
                "Decompressed prefix exceeded safety limit."
            )

    complete_member_retrieved = (
        compressed_bytes_retrieved
        >= compressed_size
    )

    if (
        complete_member_retrieved
        and decompressor is not None
    ):
        output.extend(
            decompressor.flush()
        )

    return (
        bytes(output),
        compressed_bytes_retrieved,
        complete_member_retrieved,
    )


def complete_text_lines(
    payload: bytes,
) -> str:

    text = payload.decode(
        "utf-8-sig",
        errors="replace",
    )

    if not text:
        return ""

    if text.endswith(
        (
            "\n",
            "\r",
        )
    ):
        return text

    lines = text.splitlines(
        keepends=True
    )

    complete = [
        line
        for line in lines
        if line.endswith(
            (
                "\n",
                "\r",
            )
        )
    ]

    return "".join(
        complete
    )


def normalize_column(
    value: str,
) -> str:

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        value.strip().lower(),
    ).strip(
        "_"
    )


def choose_column(
    columns: list[str],
    candidates: tuple[str, ...],
) -> str | None:

    normalized = {
        normalize_column(
            column
        ): column
        for column in columns
    }

    for candidate in candidates:

        if candidate in normalized:
            return normalized[
                candidate
            ]

    return None


remote_rows, remote_columns = read_tsv(
    remote_manifest_path
)

required_remote_columns = {
    "file_name",
    "content_url",
    "size_bytes",
}

missing = sorted(
    required_remote_columns
    - set(
        remote_columns
    )
)

if missing:
    raise SystemExit(
        "FAIL: remote manifest lacks columns: "
        + ";".join(
            missing
        )
    )

remote_lookup = {
    row["file_name"]: row
    for row in remote_rows
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

    member_rows = list(reader)
    member_columns = reader.fieldnames or []

required_member_columns = {
    "archive_name",
    "member_name",
    "compression_method",
    "compressed_size_bytes",
    "uncompressed_size_bytes",
    "local_header_offset",
}

missing = sorted(
    required_member_columns
    - set(
        member_columns
    )
)

if missing:
    raise SystemExit(
        "FAIL: member manifest lacks columns: "
        + ";".join(
            missing
        )
    )

member_lookup = {
    (
        row["archive_name"],
        row["member_name"],
    ): row
    for row in member_rows
}

resolution_rows, resolution_columns = read_tsv(
    resolution_path
)

required_resolution_columns = {
    "archive_name",
    "donor_id_from_archive",
    "section_token_from_archive",
    "selected_metadata_member",
    "metadata_resolution_status",
    "selected_cell_by_gene_member",
    "matrix_resolution_status",
}

missing = sorted(
    required_resolution_columns
    - set(
        resolution_columns
    )
)

if missing:
    raise SystemExit(
        "FAIL: resolution table lacks columns: "
        + ";".join(
            missing
        )
    )

if len(resolution_rows) != 45:
    raise SystemExit(
        f"FAIL: expected 45 archive-resolution rows, "
        f"observed {len(resolution_rows)}."
    )

module_rows, module_columns = read_tsv(
    module_lock_path
)

module_column = choose_column(
    module_columns,
    (
        "module",
        "module_id",
        "module_name",
    ),
)

gene_column = choose_column(
    module_columns,
    (
        "gene",
        "gene_symbol",
        "symbol",
        "gene_name",
    ),
)

if (
    module_column is None
    or gene_column is None
):
    raise SystemExit(
        "FAIL: could not identify module/gene columns "
        "in corrected module lock. Columns: "
        + ";".join(
            module_columns
        )
    )

module_genes: dict[
    str,
    set[str],
] = defaultdict(set)

for row in module_rows:

    module = row[
        module_column
    ].strip()

    gene = row[
        gene_column
    ].strip().upper()

    if module and gene:
        module_genes[
            module
        ].add(
            gene
        )

if len(module_genes) != 9:
    raise SystemExit(
        f"FAIL: expected nine locked modules, "
        f"observed {len(module_genes)}."
    )

locked_pairs = sum(
    len(genes)
    for genes in module_genes.values()
)

if locked_pairs != 244:
    raise SystemExit(
        f"FAIL: expected 244 locked module-gene pairs, "
        f"observed {locked_pairs}."
    )

matrix_audit_rows: list[
    dict[str, Any]
] = []

panel_gene_rows: list[
    dict[str, Any]
] = []

coverage_rows: list[
    dict[str, Any]
] = []

metadata_audit_rows: list[
    dict[str, Any]
] = []

sample_mapping_rows: list[
    dict[str, Any]
] = []

exception_rows: list[
    dict[str, Any]
] = []

non_gene_identifier_fields = {
    "",
    "cell",
    "cell_id",
    "cellid",
    "barcode",
    "index",
    "unnamed_0",
    "unnamed",
}

age_column_terms = (
    "age",
    "gestational",
    "gestation",
    "developmental_age",
    "development_stage",
    "gw",
    "week",
)

donor_column_terms = (
    "donor",
    "subject",
    "brain_id",
    "specimen",
)

area_column_terms = (
    "area",
    "region",
    "cortical_area",
    "brain_region",
    "structure",
    "section",
    "slice",
    "sample",
)

for archive_index, resolution in enumerate(
    resolution_rows,
    start=1,
):

    archive_name = resolution[
        "archive_name"
    ]

    donor_id = resolution[
        "donor_id_from_archive"
    ]

    section_token = resolution[
        "section_token_from_archive"
    ]

    print(
        f"[{archive_index:02d}/45] "
        f"Auditing {archive_name}"
    )

    remote = remote_lookup.get(
        archive_name
    )

    if remote is None:
        raise SystemExit(
            f"FAIL: archive absent from remote manifest: "
            f"{archive_name}"
        )

    content_url = remote[
        "content_url"
    ]

    matrix_member_name = resolution[
        "selected_cell_by_gene_member"
    ]

    matrix_member = member_lookup.get(
        (
            archive_name,
            matrix_member_name,
        )
    )

    if matrix_member is None:
        raise SystemExit(
            f"FAIL: unresolved matrix member for "
            f"{archive_name}: {matrix_member_name}"
        )

    try:

        matrix_prefix, matrix_compressed_bytes, matrix_complete = (
            extract_text_prefix(
                content_url,
                int(
                    matrix_member[
                        "local_header_offset"
                    ]
                ),
                int(
                    matrix_member[
                        "compression_method"
                    ]
                ),
                int(
                    matrix_member[
                        "compressed_size_bytes"
                    ]
                ),
                matrix_required_complete_lines,
                matrix_max_compressed_bytes,
            )
        )

        matrix_text = complete_text_lines(
            matrix_prefix
        )

        if not matrix_text:
            raise RuntimeError(
                "No complete matrix-header line was retrieved."
            )

        matrix_header_line = matrix_text.splitlines()[0]

        header_fields = next(
            csv.reader(
                [
                    matrix_header_line
                ]
            )
        )

        raw_header_count = len(
            header_fields
        )

        panel_fields = list(
            header_fields
        )

        identifier_fields_removed: list[str] = []

        while panel_fields:

            normalized_first = normalize_column(
                panel_fields[0]
            )

            if normalized_first not in non_gene_identifier_fields:
                break

            identifier_fields_removed.append(
                panel_fields.pop(0)
            )

        panel_genes = [
            field.strip()
            for field in panel_fields
            if field.strip()
        ]

        panel_gene_count = len(
            panel_genes
        )

        if panel_gene_count == 300:
            panel_class = (
                "300_gene_panel"
            )
        elif panel_gene_count == 960:
            panel_class = (
                "960_gene_panel"
            )
        else:
            panel_class = (
                f"other_{panel_gene_count}_gene_panel"
            )

        safe_archive = archive_name.replace(
            ".zip",
            ""
        )

        (
            matrix_raw_dir
            / f"{safe_archive}_cell_by_gene_header.csv"
        ).write_text(
            matrix_header_line
            + "\n",
            encoding="utf-8",
        )

        normalized_panel_genes = {
            gene.upper()
            for gene in panel_genes
        }

        matrix_audit_rows.append(
            {
                "archive_name": archive_name,
                "donor_id": donor_id,
                "section_token": section_token,
                "matrix_member": (
                    matrix_member_name
                ),
                "raw_header_columns": (
                    raw_header_count
                ),
                "identifier_columns_removed": len(
                    identifier_fields_removed
                ),
                "identifier_column_names": ";".join(
                    identifier_fields_removed
                ),
                "panel_gene_count": (
                    panel_gene_count
                ),
                "panel_class": panel_class,
                "unique_panel_genes": len(
                    normalized_panel_genes
                ),
                "duplicate_panel_gene_names": (
                    panel_gene_count
                    - len(
                        normalized_panel_genes
                    )
                ),
                "compressed_prefix_bytes_retrieved": (
                    matrix_compressed_bytes
                ),
                "complete_matrix_member_downloaded": (
                    matrix_complete
                ),
                "matrix_header_retrieved": True,
                "matrix_data_rows_accessed": False,
            }
        )

        for gene_order, gene in enumerate(
            panel_genes,
            start=1,
        ):

            panel_gene_rows.append(
                {
                    "archive_name": archive_name,
                    "donor_id": donor_id,
                    "section_token": section_token,
                    "panel_class": panel_class,
                    "panel_gene_count": (
                        panel_gene_count
                    ),
                    "gene_order": gene_order,
                    "gene_symbol": gene,
                    "normalized_gene_symbol": (
                        gene.upper()
                    ),
                }
            )

        for module in sorted(
            module_genes
        ):

            locked = module_genes[
                module
            ]

            matched = sorted(
                locked
                & normalized_panel_genes
            )

            missing_genes = sorted(
                locked
                - normalized_panel_genes
            )

            coverage_rows.append(
                {
                    "archive_name": archive_name,
                    "donor_id": donor_id,
                    "section_token": section_token,
                    "panel_class": panel_class,
                    "panel_gene_count": (
                        panel_gene_count
                    ),
                    "module": module,
                    "locked_module_genes": len(
                        locked
                    ),
                    "matched_module_genes": len(
                        matched
                    ),
                    "module_coverage_fraction": (
                        len(matched)
                        / len(locked)
                    ),
                    "matched_genes": ";".join(
                        matched
                    ),
                    "missing_genes": ";".join(
                        missing_genes
                    ),
                    "zero_module_coverage": (
                        len(matched) == 0
                    ),
                }
            )

    except Exception as error:

        exception_rows.append(
            {
                "archive_name": archive_name,
                "member_type": (
                    "cell_by_gene_matrix"
                ),
                "member_name": (
                    matrix_member_name
                ),
                "exception": str(
                    error
                ),
            }
        )

        print(
            f"  MATRIX FAIL: {error}"
        )

        continue

    metadata_status = resolution[
        "metadata_resolution_status"
    ]

    metadata_member_name = resolution[
        "selected_metadata_member"
    ]

    if metadata_status != "resolved":

        metadata_audit_rows.append(
            {
                "archive_name": archive_name,
                "donor_id": donor_id,
                "section_token": section_token,
                "metadata_member": "",
                "metadata_resolution_status": (
                    metadata_status
                ),
                "metadata_header_columns": 0,
                "metadata_columns": "",
                "prefix_data_rows_parsed": 0,
                "age_columns_detected": "",
                "age_values_in_prefix": "",
                "donor_columns_detected": "",
                "donor_values_in_prefix": "",
                "area_columns_detected": "",
                "area_values_in_prefix": "",
                "compressed_prefix_bytes_retrieved": 0,
                "complete_metadata_member_downloaded": (
                    False
                ),
                "metadata_prefix_retrieved": False,
            }
        )

        sample_mapping_rows.append(
            {
                "archive_name": archive_name,
                "donor_id_from_archive": (
                    donor_id
                ),
                "section_token_from_archive": (
                    section_token
                ),
                "age_columns_detected": "",
                "age_values_in_prefix": "",
                "donor_values_in_prefix": "",
                "area_values_in_prefix": "",
                "age_resolution_status": (
                    "metadata_exception_requires_external_mapping"
                ),
                "metadata_available": False,
            }
        )

        continue

    metadata_member = member_lookup.get(
        (
            archive_name,
            metadata_member_name,
        )
    )

    if metadata_member is None:
        raise SystemExit(
            f"FAIL: metadata member absent for "
            f"{archive_name}: {metadata_member_name}"
        )

    try:

        (
            metadata_prefix,
            metadata_compressed_bytes,
            metadata_complete,
        ) = extract_text_prefix(
            content_url,
            int(
                metadata_member[
                    "local_header_offset"
                ]
            ),
            int(
                metadata_member[
                    "compression_method"
                ]
            ),
            int(
                metadata_member[
                    "compressed_size_bytes"
                ]
            ),
            metadata_required_complete_lines,
            metadata_max_compressed_bytes,
        )

        metadata_text = complete_text_lines(
            metadata_prefix
        )

        if not metadata_text:
            raise RuntimeError(
                "No complete metadata lines were retrieved."
            )

        reader = csv.reader(
            io.StringIO(
                metadata_text
            )
        )

        parsed_rows = list(
            reader
        )

        if len(parsed_rows) < 2:
            raise RuntimeError(
                "Metadata prefix contains fewer than "
                "two complete CSV rows."
            )

        metadata_header = parsed_rows[0]

        metadata_data_rows = [
            row
            for row in parsed_rows[1:]
            if row
        ][
            :metadata_required_complete_lines - 1
        ]

        safe_archive = archive_name.replace(
            ".zip",
            ""
        )

        (
            metadata_raw_dir
            / f"{safe_archive}_cell_metadata_prefix.csv"
        ).write_text(
            metadata_text,
            encoding="utf-8",
        )

        normalized_headers = [
            normalize_column(
                column
            )
            for column in metadata_header
        ]

        age_indices = [
            index
            for index, column in enumerate(
                normalized_headers
            )
            if any(
                term in column
                for term in age_column_terms
            )
        ]

        donor_indices = [
            index
            for index, column in enumerate(
                normalized_headers
            )
            if any(
                term in column
                for term in donor_column_terms
            )
        ]

        area_indices = [
            index
            for index, column in enumerate(
                normalized_headers
            )
            if any(
                term in column
                for term in area_column_terms
            )
        ]

        def values_for_indices(
            indices: list[int],
        ) -> list[str]:

            values: set[str] = set()

            for row in metadata_data_rows:

                for index in indices:

                    if index >= len(row):
                        continue

                    value = row[
                        index
                    ].strip()

                    if value:
                        values.add(
                            value
                        )

            return sorted(
                values
            )

        age_values = values_for_indices(
            age_indices
        )

        donor_values = values_for_indices(
            donor_indices
        )

        area_values = values_for_indices(
            area_indices
        )

        if not age_values:

            age_token_values = sorted(
                {
                    match.group(0).upper().replace(
                        "_",
                        ""
                    ).replace(
                        "-",
                        ""
                    )
                    for row in metadata_data_rows
                    for value in row
                    for match in re.finditer(
                        r"(?i)\bGW[_-]?\d{1,2}\b",
                        value,
                    )
                }
            )

            if age_token_values:
                age_values = age_token_values

        if age_indices and age_values:
            age_resolution_status = (
                "age_values_detected_in_metadata_prefix"
            )
        elif age_values:
            age_resolution_status = (
                "GW_token_detected_in_metadata_prefix"
            )
        elif age_indices:
            age_resolution_status = (
                "age_column_detected_but_prefix_value_absent"
            )
        else:
            age_resolution_status = (
                "age_not_detected_in_metadata_prefix"
            )

        metadata_audit_rows.append(
            {
                "archive_name": archive_name,
                "donor_id": donor_id,
                "section_token": section_token,
                "metadata_member": (
                    metadata_member_name
                ),
                "metadata_resolution_status": (
                    metadata_status
                ),
                "metadata_header_columns": len(
                    metadata_header
                ),
                "metadata_columns": ";".join(
                    metadata_header
                ),
                "prefix_data_rows_parsed": len(
                    metadata_data_rows
                ),
                "age_columns_detected": ";".join(
                    metadata_header[index]
                    for index in age_indices
                ),
                "age_values_in_prefix": ";".join(
                    age_values
                ),
                "donor_columns_detected": ";".join(
                    metadata_header[index]
                    for index in donor_indices
                ),
                "donor_values_in_prefix": ";".join(
                    donor_values
                ),
                "area_columns_detected": ";".join(
                    metadata_header[index]
                    for index in area_indices
                ),
                "area_values_in_prefix": ";".join(
                    area_values
                ),
                "compressed_prefix_bytes_retrieved": (
                    metadata_compressed_bytes
                ),
                "complete_metadata_member_downloaded": (
                    metadata_complete
                ),
                "metadata_prefix_retrieved": True,
            }
        )

        sample_mapping_rows.append(
            {
                "archive_name": archive_name,
                "donor_id_from_archive": (
                    donor_id
                ),
                "section_token_from_archive": (
                    section_token
                ),
                "age_columns_detected": ";".join(
                    metadata_header[index]
                    for index in age_indices
                ),
                "age_values_in_prefix": ";".join(
                    age_values
                ),
                "donor_values_in_prefix": ";".join(
                    donor_values
                ),
                "area_values_in_prefix": ";".join(
                    area_values
                ),
                "age_resolution_status": (
                    age_resolution_status
                ),
                "metadata_available": True,
            }
        )

    except Exception as error:

        exception_rows.append(
            {
                "archive_name": archive_name,
                "member_type": (
                    "cell_metadata"
                ),
                "member_name": (
                    metadata_member_name
                ),
                "exception": str(
                    error
                ),
            }
        )

        print(
            f"  METADATA FAIL: {error}"
        )

matrix_columns = [
    "archive_name",
    "donor_id",
    "section_token",
    "matrix_member",
    "raw_header_columns",
    "identifier_columns_removed",
    "identifier_column_names",
    "panel_gene_count",
    "panel_class",
    "unique_panel_genes",
    "duplicate_panel_gene_names",
    "compressed_prefix_bytes_retrieved",
    "complete_matrix_member_downloaded",
    "matrix_header_retrieved",
    "matrix_data_rows_accessed",
]

write_tsv(
    matrix_dir
    / "phase10B5_P3_matrix_header_audit.tsv",
    matrix_audit_rows,
    matrix_columns,
)

write_tsv_gz(
    matrix_dir
    / "phase10B5_P3_panel_gene_inventory.tsv.gz",
    panel_gene_rows,
    [
        "archive_name",
        "donor_id",
        "section_token",
        "panel_class",
        "panel_gene_count",
        "gene_order",
        "gene_symbol",
        "normalized_gene_symbol",
    ],
)

write_tsv(
    metadata_dir
    / "phase10B5_P3_metadata_prefix_audit.tsv",
    metadata_audit_rows,
    [
        "archive_name",
        "donor_id",
        "section_token",
        "metadata_member",
        "metadata_resolution_status",
        "metadata_header_columns",
        "metadata_columns",
        "prefix_data_rows_parsed",
        "age_columns_detected",
        "age_values_in_prefix",
        "donor_columns_detected",
        "donor_values_in_prefix",
        "area_columns_detected",
        "area_values_in_prefix",
        "compressed_prefix_bytes_retrieved",
        "complete_metadata_member_downloaded",
        "metadata_prefix_retrieved",
    ],
)

write_tsv(
    coverage_dir
    / "phase10B5_P3_archive_module_panel_coverage.tsv",
    coverage_rows,
    [
        "archive_name",
        "donor_id",
        "section_token",
        "panel_class",
        "panel_gene_count",
        "module",
        "locked_module_genes",
        "matched_module_genes",
        "module_coverage_fraction",
        "matched_genes",
        "missing_genes",
        "zero_module_coverage",
    ],
)

write_tsv(
    mapping_dir
    / "phase10B5_P3_sample_age_area_mapping_preflight.tsv",
    sample_mapping_rows,
    [
        "archive_name",
        "donor_id_from_archive",
        "section_token_from_archive",
        "age_columns_detected",
        "age_values_in_prefix",
        "donor_values_in_prefix",
        "area_values_in_prefix",
        "age_resolution_status",
        "metadata_available",
    ],
)

write_tsv(
    exception_dir
    / "phase10B5_P3_prefix_audit_exceptions.tsv",
    exception_rows,
    [
        "archive_name",
        "member_type",
        "member_name",
        "exception",
    ],
)

panel_class_counts = Counter(
    row[
        "panel_class"
    ]
    for row in matrix_audit_rows
)

panel_summary_rows = [
    {
        "panel_class": panel_class,
        "archives": count,
        "donors": len(
            {
                row["donor_id"]
                for row in matrix_audit_rows
                if row[
                    "panel_class"
                ] == panel_class
            }
        ),
        "section_tokens": ";".join(
            sorted(
                {
                    row[
                        "section_token"
                    ]
                    for row in matrix_audit_rows
                    if row[
                        "panel_class"
                    ] == panel_class
                }
            )
        ),
    }
    for panel_class, count in sorted(
        panel_class_counts.items()
    )
]

write_tsv(
    matrix_dir
    / "phase10B5_P3_panel_class_summary.tsv",
    panel_summary_rows,
    [
        "panel_class",
        "archives",
        "donors",
        "section_tokens",
    ],
)

coverage_by_panel_module: dict[
    tuple[str, str],
    list[dict[str, Any]],
] = defaultdict(list)

for row in coverage_rows:

    coverage_by_panel_module[
        (
            row["panel_class"],
            row["module"],
        )
    ].append(
        row
    )

coverage_summary_rows: list[
    dict[str, Any]
] = []

for (
    panel_class,
    module,
), rows in sorted(
    coverage_by_panel_module.items()
):

    matched_counts = [
        int(
            row[
                "matched_module_genes"
            ]
        )
        for row in rows
    ]

    fractions = [
        float(
            row[
                "module_coverage_fraction"
            ]
        )
        for row in rows
    ]

    coverage_summary_rows.append(
        {
            "panel_class": panel_class,
            "module": module,
            "archives": len(
                rows
            ),
            "locked_module_genes": int(
                rows[0][
                    "locked_module_genes"
                ]
            ),
            "minimum_matched_genes": min(
                matched_counts
            ),
            "maximum_matched_genes": max(
                matched_counts
            ),
            "minimum_coverage_fraction": min(
                fractions
            ),
            "maximum_coverage_fraction": max(
                fractions
            ),
            "archives_with_zero_coverage": sum(
                count == 0
                for count in matched_counts
            ),
        }
    )

write_tsv(
    coverage_dir
    / "phase10B5_P3_panel_class_module_coverage_summary.tsv",
    coverage_summary_rows,
    [
        "panel_class",
        "module",
        "archives",
        "locked_module_genes",
        "minimum_matched_genes",
        "maximum_matched_genes",
        "minimum_coverage_fraction",
        "maximum_coverage_fraction",
        "archives_with_zero_coverage",
    ],
)

matrix_headers_audited = len(
    matrix_audit_rows
)

metadata_prefixes_audited = sum(
    as_bool(
        row[
            "metadata_prefix_retrieved"
        ]
    )
    for row in metadata_audit_rows
)

matrix_data_rows_accessed = any(
    as_bool(
        row[
            "matrix_data_rows_accessed"
        ]
    )
    for row in matrix_audit_rows
)

complete_matrices_downloaded = sum(
    as_bool(
        row[
            "complete_matrix_member_downloaded"
        ]
    )
    for row in matrix_audit_rows
)

complete_metadata_members_downloaded = sum(
    as_bool(
        row[
            "complete_metadata_member_downloaded"
        ]
    )
    for row in metadata_audit_rows
)

age_mappings_detected = sum(
    row[
        "age_resolution_status"
    ]
    in {
        "age_values_detected_in_metadata_prefix",
        "GW_token_detected_in_metadata_prefix",
    }
    for row in sample_mapping_rows
)

zero_coverage_rows = sum(
    as_bool(
        row[
            "zero_module_coverage"
        ]
    )
    for row in coverage_rows
)

technical_pass = (
    matrix_headers_audited == 45
    and metadata_prefixes_audited == 44
    and len(
        coverage_rows
    ) == 405
    and not matrix_data_rows_accessed
    and complete_matrices_downloaded == 0
)

status_value = (
    "passed_phase10B5_P3_remote_header_prefix_audit_"
    "ready_for_panel_coverage_lock_and_sample_age_mapping_review"
    if technical_pass
    else (
        "phase10B5_P3_requires_manual_review"
    )
)

status = {
    "phase": "phase10B5_P3",
    "remote_archives": 45,
    "matrix_headers_audited": (
        matrix_headers_audited
    ),
    "metadata_prefixes_audited": (
        metadata_prefixes_audited
    ),
    "metadata_exception_archives": (
        45 - metadata_prefixes_audited
    ),
    "locked_modules": len(
        module_genes
    ),
    "locked_module_gene_pairs": (
        locked_pairs
    ),
    "archive_module_coverage_rows": len(
        coverage_rows
    ),
    "panel_classes_detected": ";".join(
        sorted(
            panel_class_counts
        )
    ),
    "age_mappings_detected_from_prefix": (
        age_mappings_detected
    ),
    "zero_module_coverage_rows": (
        zero_coverage_rows
    ),
    "remote_bytes_retrieved": (
        network_bytes_retrieved
    ),
    "remote_size_retrieved_human": (
        human_size(
            network_bytes_retrieved
        )
    ),
    "partial_archive_member_payload_retrieved": (
        True
    ),
    "complete_ZIP_archives_downloaded": (
        False
    ),
    "complete_expression_matrices_downloaded": (
        complete_matrices_downloaded
    ),
    "complete_metadata_members_downloaded": (
        complete_metadata_members_downloaded
    ),
    "matrix_headers_accessed": True,
    "matrix_data_rows_accessed": (
        matrix_data_rows_accessed
    ),
    "expression_values_accessed": (
        False
    ),
    "metadata_values_accessed": (
        metadata_prefixes_audited > 0
    ),
    "module_scores_computed": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P3_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B5_P3_status.tsv",
    [
        status
    ],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P3 REMOTE MATRIX-HEADER "
    "AND METADATA-PREFIX AUDIT =====",
    "",
    (
        "Matrix headers audited: "
        f"{matrix_headers_audited}/45"
    ),
    (
        "Metadata prefixes audited: "
        f"{metadata_prefixes_audited}/44 usable"
    ),
    (
        "Archive-module coverage rows: "
        f"{len(coverage_rows)}"
    ),
    (
        "Panel classes detected: "
        + (
            ";".join(
                sorted(
                    panel_class_counts
                )
            )
            or "NONE"
        )
    ),
    (
        "Age mappings detected from prefixes: "
        f"{age_mappings_detected}"
    ),
    (
        "Zero module-coverage rows: "
        f"{zero_coverage_rows}"
    ),
    (
        "Remote bytes retrieved: "
        f"{human_size(network_bytes_retrieved)}"
    ),
    "",
    "Partial archive-member payload retrieved: TRUE",
    "Complete ZIP archives downloaded: FALSE",
    (
        "Complete expression matrices downloaded: "
        f"{complete_matrices_downloaded}"
    ),
    (
        "Complete metadata members downloaded: "
        f"{complete_metadata_members_downloaded}"
    ),
    "Matrix headers accessed: TRUE",
    "Matrix data rows accessed: FALSE",
    "Expression values accessed: FALSE",
    "Metadata values accessed: TRUE",
    "Module scores computed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P3 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P3_report.txt"
).write_text(
    "\n".join(report)
    + "\n",
    encoding="utf-8",
)

print(
    "\n".join(report)
)

print(
    "\n===== PANEL CLASS SUMMARY ====="
)

for row in panel_summary_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "panel_class",
                "archives",
                "donors",
                "section_tokens",
            )
        )
    )

print(
    "\n===== SAMPLE AGE/AREA MAPPING PREFLIGHT ====="
)

for row in sample_mapping_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "archive_name",
                "donor_id_from_archive",
                "section_token_from_archive",
                "age_values_in_prefix",
                "area_values_in_prefix",
                "age_resolution_status",
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
        != "phase10B5_P3_SHA256.tsv"
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
    / "phase10B5_P3_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)
