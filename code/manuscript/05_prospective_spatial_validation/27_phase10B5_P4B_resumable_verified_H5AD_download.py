from __future__ import annotations

import csv
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


project = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
out = Path(sys.argv[3])

download_dir = (
    out
    / "01_H5AD_payloads"
)

audit_dir = (
    out
    / "02_download_audit"
)

validation_dir = (
    out
    / "03_validation"
)

for directory in (
    download_dir,
    audit_dir,
    validation_dir,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

expected_hdf5_signature = (
    b"\x89HDF\r\n\x1a\n"
)

minimum_free_space_margin = (
    5
    * 1024**3
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
    value: int,
) -> str:

    number = float(
        value
    )

    for unit in (
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ):

        if (
            number < 1024
            or unit == "TB"
        ):
            return f"{number:.2f}_{unit}"

        number /= 1024

    return f"{value}_B"


def calculate_md5(
    path: Path,
) -> str:

    digest = hashlib.md5()

    with path.open(
        "rb",
    ) as handle:

        while True:

            chunk = handle.read(
                16
                * 1024
                * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def inspect_file(
    path: Path,
    expected_size: int,
    expected_md5: str,
) -> dict[str, Any]:

    exists = path.is_file()

    observed_size = (
        path.stat().st_size
        if exists
        else 0
    )

    size_matches = (
        exists
        and observed_size
        == expected_size
    )

    observed_md5 = ""

    md5_matches = False

    hdf5_signature_matches = False

    if size_matches:

        observed_md5 = calculate_md5(
            path
        )

        md5_matches = (
            observed_md5.lower()
            == expected_md5.lower()
        )

        with path.open(
            "rb",
        ) as handle:

            signature = handle.read(
                8
            )

        hdf5_signature_matches = (
            signature
            == expected_hdf5_signature
        )

    fully_valid = (
        size_matches
        and md5_matches
        and hdf5_signature_matches
    )

    return {
        "exists": exists,
        "observed_size_bytes": observed_size,
        "size_matches": size_matches,
        "observed_md5": observed_md5,
        "md5_matches": md5_matches,
        "HDF5_signature_matches": (
            hdf5_signature_matches
        ),
        "fully_valid": fully_valid,
    }


rows, columns = read_tsv(
    manifest_path
)

required_columns = {
    "file_name",
    "size_bytes",
    "checksum",
    "selected_download_url",
    "range_probe_passed",
}

missing = sorted(
    required_columns
    - set(
        columns
    )
)

if missing:
    raise SystemExit(
        "FAIL: corrected manifest lacks columns: "
        + ";".join(
            missing
        )
    )

if len(rows) != 6:
    raise SystemExit(
        f"FAIL: expected six H5AD files, "
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

observed_files = {
    row[
        "file_name"
    ]
    for row in rows
}

if observed_files != expected_files:
    raise SystemExit(
        "FAIL: selected H5AD set differs from "
        "the locked six-file design."
    )

for row in rows:

    if row[
        "range_probe_passed"
    ].strip().upper() != "TRUE":

        raise SystemExit(
            "FAIL: range probe did not pass for "
            f"{row['file_name']}."
        )

    if not row[
        "selected_download_url"
    ].strip():

        raise SystemExit(
            "FAIL: selected download URL is blank for "
            f"{row['file_name']}."
        )

expected_total_bytes = sum(
    int(
        row[
            "size_bytes"
        ]
    )
    for row in rows
)

remaining_download_bytes = 0

for row in rows:

    file_name = row[
        "file_name"
    ]

    expected_size = int(
        row[
            "size_bytes"
        ]
    )

    final_path = (
        download_dir
        / file_name
    )

    part_path = (
        download_dir
        / f"{file_name}.part"
    )

    if (
        final_path.is_file()
        and final_path.stat().st_size
        == expected_size
    ):
        continue

    partial_size = (
        part_path.stat().st_size
        if part_path.is_file()
        else 0
    )

    if partial_size > expected_size:
        partial_size = 0

    remaining_download_bytes += max(
        expected_size
        - partial_size,
        0,
    )

disk_usage = shutil.disk_usage(
    download_dir
)

required_free_bytes = (
    remaining_download_bytes
    + minimum_free_space_margin
)

print(
    "===== DISK-SPACE PREFLIGHT ====="
)

print(
    "Expected complete payload: "
    f"{human_size(expected_total_bytes)}"
)

print(
    "Remaining download estimate: "
    f"{human_size(remaining_download_bytes)}"
)

print(
    "Available disk space: "
    f"{human_size(disk_usage.free)}"
)

print(
    "Required free space including margin: "
    f"{human_size(required_free_bytes)}"
)

if disk_usage.free < required_free_bytes:

    raise SystemExit(
        "FAIL: insufficient free disk space for "
        "the remaining downloads plus the "
        "5-GB safety margin."
    )

print(
    "PASS: disk-space preflight completed."
)

download_results: list[
    dict[str, Any]
] = []

for file_index, row in enumerate(
    rows,
    start=1,
):

    file_name = row[
        "file_name"
    ]

    expected_size = int(
        row[
            "size_bytes"
        ]
    )

    checksum = row[
        "checksum"
    ].strip()

    if not checksum.lower().startswith(
        "md5:"
    ):
        raise SystemExit(
            f"FAIL: unsupported checksum for "
            f"{file_name}: {checksum}"
        )

    expected_md5 = checksum.split(
        ":",
        1,
    )[1].strip().lower()

    selected_url = row[
        "selected_download_url"
    ].strip()

    final_path = (
        download_dir
        / file_name
    )

    part_path = (
        download_dir
        / f"{file_name}.part"
    )

    print()
    print(
        f"===== [{file_index:02d}/06] "
        f"{file_name} =====",
        flush=True,
    )

    existing_validation = inspect_file(
        final_path,
        expected_size,
        expected_md5,
    )

    if existing_validation[
        "fully_valid"
    ]:

        print(
            "PASS: existing final file is already "
            "size-, MD5- and HDF5-verified."
        )

        download_results.append(
            {
                "file_name": file_name,
                "expected_size_bytes": (
                    expected_size
                ),
                "observed_size_bytes": (
                    existing_validation[
                        "observed_size_bytes"
                    ]
                ),
                "expected_md5": (
                    expected_md5
                ),
                "observed_md5": (
                    existing_validation[
                        "observed_md5"
                    ]
                ),
                "size_matches": True,
                "md5_matches": True,
                "HDF5_signature_matches": True,
                "download_attempts_this_run": 0,
                "download_action": (
                    "existing_verified_file_reused"
                ),
                "curl_exit_status": 0,
                "complete_payload_verified": True,
                "expression_values_accessed": False,
                "obs_values_accessed": False,
            }
        )

        continue

    if final_path.exists():

        quarantine = (
            download_dir
            / (
                f"{file_name}.invalid_final_"
                f"{int(time.time())}"
            )
        )

        final_path.rename(
            quarantine
        )

        print(
            "WARNING: invalid existing final file "
            f"moved to {quarantine.name}."
        )

    if (
        part_path.exists()
        and part_path.stat().st_size
        > expected_size
    ):

        quarantine = (
            download_dir
            / (
                f"{file_name}.oversized_part_"
                f"{int(time.time())}"
            )
        )

        part_path.rename(
            quarantine
        )

        print(
            "WARNING: oversized partial file moved "
            f"to {quarantine.name}."
        )

    attempts_used = 0
    final_curl_status = -1
    final_validation: dict[str, Any] = {}
    download_action = ""

    for attempt in (
        1,
        2,
    ):

        attempts_used = attempt

        existing_part_bytes = (
            part_path.stat().st_size
            if part_path.exists()
            else 0
        )

        if existing_part_bytes > 0:

            print(
                "Resuming from "
                f"{human_size(existing_part_bytes)}."
            )

            download_action = (
                "resumed_partial_download"
            )

        else:

            print(
                "Starting a fresh download."
            )

            download_action = (
                "fresh_download"
            )

        command = [
            "curl",
            "--fail",
            "--location",
            "--http1.1",
            "--continue-at",
            "-",
            "--output",
            str(
                part_path
            ),
            "--retry",
            "20",
            "--retry-all-errors",
            "--retry-delay",
            "10",
            "--connect-timeout",
            "30",
            "--speed-time",
            "180",
            "--speed-limit",
            "1024",
            "--progress-bar",
            "--user-agent",
            (
                "DTHI-H5AD-Verified-Download/"
                "phase10B5-P4B"
            ),
            selected_url,
        ]

        completed = subprocess.run(
            command,
            check=False,
        )

        final_curl_status = (
            completed.returncode
        )

        if final_curl_status != 0:

            print(
                "WARNING: curl exited with status "
                f"{final_curl_status}."
            )

            if attempt == 1:
                continue

            break

        observed_part_size = (
            part_path.stat().st_size
            if part_path.exists()
            else 0
        )

        print(
            "Downloaded file size: "
            f"{human_size(observed_part_size)}"
        )

        if observed_part_size != expected_size:

            print(
                "WARNING: downloaded size does not "
                "match the official manifest."
            )

            if attempt == 1:
                continue

            break

        print(
            "Calculating MD5 checksum..."
        )

        final_validation = inspect_file(
            part_path,
            expected_size,
            expected_md5,
        )

        if final_validation[
            "fully_valid"
        ]:

            os.replace(
                part_path,
                final_path,
            )

            print(
                "PASS: exact size, official MD5 and "
                "HDF5 signature verified."
            )

            break

        print(
            "WARNING: verification failed."
        )

        print(
            "  Size matches: "
            f"{final_validation.get('size_matches')}"
        )

        print(
            "  MD5 matches: "
            f"{final_validation.get('md5_matches')}"
        )

        print(
            "  HDF5 signature matches: "
            f"{final_validation.get('HDF5_signature_matches')}"
        )

        if attempt == 1:

            quarantine = (
                download_dir
                / (
                    f"{file_name}.failed_verification_"
                    f"{int(time.time())}"
                )
            )

            if part_path.exists():
                part_path.rename(
                    quarantine
                )

            print(
                "Retrying once from a clean file."
            )

    verified = final_path.is_file()

    if verified:

        final_validation = inspect_file(
            final_path,
            expected_size,
            expected_md5,
        )

        verified = bool(
            final_validation[
                "fully_valid"
            ]
        )

    download_results.append(
        {
            "file_name": file_name,
            "expected_size_bytes": (
                expected_size
            ),
            "observed_size_bytes": (
                final_validation.get(
                    "observed_size_bytes",
                    0,
                )
            ),
            "expected_md5": expected_md5,
            "observed_md5": (
                final_validation.get(
                    "observed_md5",
                    "",
                )
            ),
            "size_matches": (
                final_validation.get(
                    "size_matches",
                    False,
                )
            ),
            "md5_matches": (
                final_validation.get(
                    "md5_matches",
                    False,
                )
            ),
            "HDF5_signature_matches": (
                final_validation.get(
                    "HDF5_signature_matches",
                    False,
                )
            ),
            "download_attempts_this_run": (
                attempts_used
            ),
            "download_action": (
                download_action
            ),
            "curl_exit_status": (
                final_curl_status
            ),
            "complete_payload_verified": (
                verified
            ),
            "expression_values_accessed": (
                False
            ),
            "obs_values_accessed": (
                False
            ),
        }
    )

    if not verified:

        print(
            f"FAIL: {file_name} was not verified."
        )

        break

result_columns = [
    "file_name",
    "expected_size_bytes",
    "observed_size_bytes",
    "expected_md5",
    "observed_md5",
    "size_matches",
    "md5_matches",
    "HDF5_signature_matches",
    "download_attempts_this_run",
    "download_action",
    "curl_exit_status",
    "complete_payload_verified",
    "expression_values_accessed",
    "obs_values_accessed",
]

write_tsv(
    audit_dir
    / "phase10B5_P4B_download_and_MD5_validation.tsv",
    download_results,
    result_columns,
)

verified_files = sum(
    bool(
        row[
            "complete_payload_verified"
        ]
    )
    for row in download_results
)

verified_bytes = sum(
    int(
        row[
            "observed_size_bytes"
        ]
    )
    for row in download_results
    if row[
        "complete_payload_verified"
    ]
)

partial_files_remaining = len(
    list(
        download_dir.glob(
            "*.part"
        )
    )
)

technical_pass = (
    len(download_results) == 6
    and verified_files == 6
    and verified_bytes
    == expected_total_bytes
    and partial_files_remaining == 0
)

status_value = (
    "passed_phase10B5_P4B_resumable_MD5_verified_"
    "processed_H5AD_download_ready_for_metadata_only_"
    "H5AD_schema_audit"
    if technical_pass
    else (
        "phase10B5_P4B_download_incomplete_or_"
        "requires_rerun"
    )
)

status = {
    "phase": "phase10B5_P4B",
    "selected_H5AD_files": len(
        rows
    ),
    "verified_complete_H5AD_files": (
        verified_files
    ),
    "expected_total_payload_bytes": (
        expected_total_bytes
    ),
    "verified_total_payload_bytes": (
        verified_bytes
    ),
    "verified_total_payload_size_human": (
        human_size(
            verified_bytes
        )
    ),
    "partial_files_remaining": (
        partial_files_remaining
    ),
    "official_MD5_checksums_verified": (
        verified_files
    ),
    "HDF5_signatures_verified": (
        verified_files
    ),
    "processed_H5AD_payload_downloaded": (
        verified_files > 0
    ),
    "H5AD_structure_opened": False,
    "H5AD_obs_values_accessed": False,
    "H5AD_var_values_accessed": False,
    "H5AD_expression_values_accessed": False,
    "module_scores_computed": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P4B_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B5_P4B_status.tsv",
    [status],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P4B RESUMABLE VERIFIED "
    "PROCESSED-H5AD DOWNLOAD =====",
    "",
    (
        "Selected H5AD files: "
        f"{len(rows)}"
    ),
    (
        "Verified complete H5AD files: "
        f"{verified_files}/6"
    ),
    (
        "Verified payload: "
        f"{human_size(verified_bytes)}"
    ),
    (
        "Official MD5 checksums verified: "
        f"{verified_files}/6"
    ),
    (
        "HDF5 signatures verified: "
        f"{verified_files}/6"
    ),
    (
        "Partial files remaining: "
        f"{partial_files_remaining}"
    ),
    "",
    "H5AD structure opened: FALSE",
    "H5AD obs values accessed: FALSE",
    "H5AD var values accessed: FALSE",
    "H5AD expression values accessed: FALSE",
    "Module scores computed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P4B STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P4B_report.txt"
).write_text(
    "\n".join(report)
    + "\n",
    encoding="utf-8",
)

print()
print(
    "\n".join(report)
)

small_output_paths = [
    audit_dir
    / "phase10B5_P4B_download_and_MD5_validation.tsv",
    out
    / "phase10B5_P4B_status.tsv",
    out
    / "phase10B5_P4B_report.txt",
]

checksum_rows: list[
    dict[str, Any]
] = []

for path in small_output_paths:

    if not path.is_file():
        continue

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
    validation_dir
    / "phase10B5_P4B_small_output_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)

if not technical_pass:
    raise SystemExit(
        "Phase 10B5-P4B is incomplete. "
        "Rerun the same wrapper to resume any "
        "remaining .part download."
    )
