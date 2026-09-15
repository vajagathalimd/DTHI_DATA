from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any


project = Path(sys.argv[1])
input_manifest = Path(sys.argv[2])
out = Path(sys.argv[3])
record_id = sys.argv[4]
probe_bytes = int(sys.argv[5])

corrected_dir = (
    out
    / "01_corrected_manifest"
)

probe_dir = (
    out
    / "02_range_probes"
)

header_dir = (
    probe_dir
    / "headers"
)

payload_dir = (
    probe_dir
    / "payloads"
)

audit_dir = (
    out
    / "03_audit"
)

for directory in (
    corrected_dir,
    probe_dir,
    header_dir,
    payload_dir,
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


def parse_headers(
    text: str,
) -> tuple[
    int,
    dict[str, str],
]:

    blocks = re.split(
        r"\r?\n\r?\n",
        text.strip(),
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

        try:
            status = int(
                parts[1]
            )
        except (
            IndexError,
            ValueError,
        ):
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


def safe_file_token(
    value: str,
) -> str:

    return re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value,
    )


def probe_endpoint(
    file_name: str,
    endpoint_type: str,
    url: str,
    file_size: int,
) -> dict[str, Any]:

    start = 0

    end = min(
        probe_bytes,
        file_size,
    ) - 1

    expected_bytes = (
        end - start + 1
    )

    safe_name = safe_file_token(
        f"{file_name}.{endpoint_type}"
    )

    header_path = (
        header_dir
        / f"{safe_name}.headers.txt"
    )

    payload_path = (
        payload_dir
        / f"{safe_name}.probe.bin"
    )

    command = [
        "curl",
        "--silent",
        "--show-error",
        "--location",
        "--http1.1",
        "--max-redirs",
        "10",
        "--connect-timeout",
        "30",
        "--max-time",
        "180",
        "--retry",
        "3",
        "--retry-all-errors",
        "--limit-rate",
        "2M",
        "--max-filesize",
        str(
            expected_bytes
            + 65_536
        ),
        "--user-agent",
        (
            "DTHI-H5AD-Range-Probe/"
            "phase10B5-P4A-R2"
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
        url,
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

    http_status, headers = parse_headers(
        header_text
    )

    content_range = headers.get(
        "content-range",
        "",
    )

    content_length = headers.get(
        "content-length",
        "",
    )

    content_type = headers.get(
        "content-type",
        "",
    )

    payload_size = (
        payload_path.stat().st_size
        if payload_path.exists()
        else 0
    )

    match = re.fullmatch(
        r"bytes\s+"
        r"(?P<start>\d+)-"
        r"(?P<end>\d+)/"
        r"(?P<total>\d+|\*)",
        content_range,
        flags=re.IGNORECASE,
    )

    exact_content_range = bool(
        match
        and int(
            match.group(
                "start"
            )
        ) == start
        and int(
            match.group(
                "end"
            )
        ) == end
    )

    probe_passed = (
        completed.returncode == 0
        and http_status == 206
        and exact_content_range
        and payload_size == expected_bytes
    )

    return {
        "file_name": file_name,
        "endpoint_type": endpoint_type,
        "url": url,
        "requested_range": (
            f"{start}-{end}"
        ),
        "requested_bytes": (
            expected_bytes
        ),
        "curl_exit_status": (
            completed.returncode
        ),
        "HTTP_status": (
            http_status
        ),
        "Content_Range": (
            content_range
        ),
        "Content_Length": (
            content_length
        ),
        "Content_Type": (
            content_type
        ),
        "bytes_retrieved": (
            payload_size
        ),
        "exact_Content_Range": (
            exact_content_range
        ),
        "probe_passed": (
            probe_passed
        ),
        "stderr": (
            completed.stderr.strip()
        ),
        "complete_H5AD_downloaded": (
            payload_size == file_size
        ),
    }


rows, columns = read_tsv(
    input_manifest
)

required_columns = {
    "file_name",
    "size_bytes",
    "size_human",
    "checksum",
    "selection_reason",
}

missing = sorted(
    required_columns
    - set(
        columns
    )
)

if missing:
    raise SystemExit(
        "FAIL: input manifest lacks columns: "
        + ";".join(
            missing
        )
    )

if len(rows) != 6:
    raise SystemExit(
        f"FAIL: expected six selected files, "
        f"observed {len(rows)}."
    )

expected_files = {
    "gw15.h5ad",
    "gw18_umb1759.h5ad",
    "gw20.h5ad",
    "gw20_umb1031.h5ad",
    "gw22.h5ad",
    "gw34.h5ad",
}

if {
    row["file_name"]
    for row in rows
} != expected_files:

    raise SystemExit(
        "FAIL: selected H5AD file set differs "
        "from the locked six-file design."
    )

probe_rows: list[
    dict[str, Any]
] = []

corrected_rows: list[
    dict[str, Any]
] = []

for index, row in enumerate(
    rows,
    start=1,
):

    file_name = row[
        "file_name"
    ]

    size_bytes = int(
        row[
            "size_bytes"
        ]
    )

    encoded_name = urllib.parse.quote(
        file_name,
        safe="",
    )

    api_url = (
        f"https://zenodo.org/api/records/"
        f"{record_id}/files/"
        f"{encoded_name}/content"
    )

    public_url = (
        f"https://zenodo.org/records/"
        f"{record_id}/files/"
        f"{encoded_name}?download=1"
    )

    print(
        f"[{index:02d}/06] Probing {file_name}",
        flush=True,
    )

    api_result = probe_endpoint(
        file_name,
        "Zenodo_API_content",
        api_url,
        size_bytes,
    )

    probe_rows.append(
        api_result
    )

    print(
        "  API:"
        f" HTTP={api_result['HTTP_status']};"
        f" bytes={api_result['bytes_retrieved']};"
        f" exact={api_result['exact_Content_Range']};"
        f" pass={api_result['probe_passed']}",
        flush=True,
    )

    public_result = probe_endpoint(
        file_name,
        "Zenodo_public_download",
        public_url,
        size_bytes,
    )

    probe_rows.append(
        public_result
    )

    print(
        "  Public:"
        f" HTTP={public_result['HTTP_status']};"
        f" bytes={public_result['bytes_retrieved']};"
        f" exact={public_result['exact_Content_Range']};"
        f" pass={public_result['probe_passed']}",
        flush=True,
    )

    selected_result = None

    for candidate in (
        api_result,
        public_result,
    ):

        if candidate[
            "probe_passed"
        ]:

            selected_result = candidate
            break

    if selected_result is None:

        corrected_rows.append(
            {
                **row,
                "derived_API_content_url": (
                    api_url
                ),
                "derived_public_download_url": (
                    public_url
                ),
                "selected_download_endpoint_type": "",
                "selected_download_url": "",
                "range_probe_passed": False,
                "range_probe_bytes": 0,
                "complete_H5AD_downloaded": False,
            }
        )

        continue

    corrected_rows.append(
        {
            **row,
            "derived_API_content_url": (
                api_url
            ),
            "derived_public_download_url": (
                public_url
            ),
            "selected_download_endpoint_type": (
                selected_result[
                    "endpoint_type"
                ]
            ),
            "selected_download_url": (
                selected_result[
                    "url"
                ]
            ),
            "range_probe_passed": True,
            "range_probe_bytes": (
                selected_result[
                    "bytes_retrieved"
                ]
            ),
            "complete_H5AD_downloaded": False,
        }
    )

probe_columns = [
    "file_name",
    "endpoint_type",
    "url",
    "requested_range",
    "requested_bytes",
    "curl_exit_status",
    "HTTP_status",
    "Content_Range",
    "Content_Length",
    "Content_Type",
    "bytes_retrieved",
    "exact_Content_Range",
    "probe_passed",
    "stderr",
    "complete_H5AD_downloaded",
]

write_tsv(
    probe_dir
    / "phase10B5_P4A_R2_H5AD_range_probe_results.tsv",
    probe_rows,
    probe_columns,
)

corrected_columns = [
    "file_name",
    "size_bytes",
    "size_human",
    "checksum",
    "selection_reason",
    "derived_API_content_url",
    "derived_public_download_url",
    "selected_download_endpoint_type",
    "selected_download_url",
    "range_probe_passed",
    "range_probe_bytes",
    "complete_H5AD_downloaded",
    "payload_downloaded",
    "obs_schema_audited",
    "expression_values_accessed",
]

write_tsv(
    corrected_dir
    / "phase10B5_P4A_R2_corrected_selected_H5AD_manifest.tsv",
    corrected_rows,
    corrected_columns,
)

files_with_working_endpoint = sum(
    bool(
        row[
            "range_probe_passed"
        ]
    )
    for row in corrected_rows
)

api_endpoints_passing = sum(
    bool(
        row[
            "probe_passed"
        ]
    )
    for row in probe_rows
    if row[
        "endpoint_type"
    ] == "Zenodo_API_content"
)

public_endpoints_passing = sum(
    bool(
        row[
            "probe_passed"
        ]
    )
    for row in probe_rows
    if row[
        "endpoint_type"
    ] == "Zenodo_public_download"
)

total_probe_bytes = sum(
    int(
        row[
            "bytes_retrieved"
        ]
    )
    for row in probe_rows
)

complete_files_downloaded = sum(
    bool(
        row[
            "complete_H5AD_downloaded"
        ]
    )
    for row in probe_rows
)

technical_pass = (
    len(corrected_rows) == 6
    and files_with_working_endpoint == 6
    and complete_files_downloaded == 0
)

status_value = (
    "passed_phase10B5_P4A_R2_processed_H5AD_"
    "URL_repair_and_range_probe_ready_for_"
    "resumable_verified_download"
    if technical_pass
    else (
        "phase10B5_P4A_R2_requires_manual_review"
    )
)

status = {
    "phase": "phase10B5_P4A_R2",
    "selected_H5AD_files": len(
        corrected_rows
    ),
    "files_with_working_download_endpoint": (
        files_with_working_endpoint
    ),
    "API_content_endpoints_passing": (
        api_endpoints_passing
    ),
    "public_download_endpoints_passing": (
        public_endpoints_passing
    ),
    "range_probe_requests": len(
        probe_rows
    ),
    "range_probe_bytes_retrieved": (
        total_probe_bytes
    ),
    "complete_H5AD_files_downloaded": (
        complete_files_downloaded
    ),
    "H5AD_expression_values_accessed": (
        False
    ),
    "H5AD_obs_values_accessed": False,
    "module_scores_computed": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P4A_R2_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B5_P4A_R2_status.tsv",
    [status],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P4A-R2 PROCESSED-H5AD "
    "URL REPAIR AND RANGE PROBE =====",
    "",
    (
        "Selected H5AD files: "
        f"{len(corrected_rows)}"
    ),
    (
        "Files with working endpoint: "
        f"{files_with_working_endpoint}/6"
    ),
    (
        "API-content endpoints passing: "
        f"{api_endpoints_passing}/6"
    ),
    (
        "Public-download endpoints passing: "
        f"{public_endpoints_passing}/6"
    ),
    (
        "Range-probe requests: "
        f"{len(probe_rows)}"
    ),
    (
        "Range-probe bytes retrieved: "
        f"{total_probe_bytes}"
    ),
    (
        "Complete H5AD files downloaded: "
        f"{complete_files_downloaded}"
    ),
    "",
    "H5AD expression values accessed: FALSE",
    "H5AD obs values accessed: FALSE",
    "Module scores computed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P4A-R2 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P4A_R2_report.txt"
).write_text(
    "\n".join(report)
    + "\n",
    encoding="utf-8",
)

print(
    "\n".join(report)
)

print(
    "\n===== CORRECTED H5AD MANIFEST ====="
)

for row in corrected_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "file_name",
                "selected_download_endpoint_type",
                "range_probe_passed",
                "range_probe_bytes",
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
        != "phase10B5_P4A_R2_SHA256.tsv"
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
    / "phase10B5_P4A_R2_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)
