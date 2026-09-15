#!/usr/bin/env python3

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import mimetypes
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request

import pandas as pd


PROJECT = Path(
    "."
)

BASE_URL = (
    "https://clowder.edap-cluster.com"
)

INPUT_MANIFEST = (
    PROJECT
    / "07_tables/main_tables/phase6/"
      "phase6B2C_target_folder_file_manifest.tsv"
)

DOWNLOAD_DIR = (
    PROJECT
    / "01_raw_data/perturbational_validation/"
      "EPA_HTTr"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase6"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase6/"
      "phase6B3_EPA_HTTr_download_verify.log"
)

DOWNLOAD_MANIFEST = (
    TABLE_DIR
    / "phase6B3_EPA_HTTr_download_manifest.tsv"
)

COMPLETION_FILE = (
    TABLE_DIR
    / "phase6B3_completion_summary.tsv"
)

CHUNK_SIZE = 1024 * 1024

MAXIMUM_FILE_BYTES = (
    500
    * 1024
    * 1024
)

ALLOWED_FILENAMES = {
    "HTTr Signatures.RData",
    (
        "HTTr signature catalog for dashboard "
        "2022-06-24_firstSheetOnly.xlsx"
    ),
}

for directory in [
    DOWNLOAD_DIR,
    TABLE_DIR,
    LOG_FILE.parent,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def log(message: str = "") -> None:
    print(
        message,
        flush=True,
    )

    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            message + "\n"
        )


def safe_filename(
    filename: str,
) -> str:
    clean = Path(
        filename
    ).name

    if clean != filename:
        raise ValueError(
            f"Unsafe filename: {filename}"
        )

    return clean


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                CHUNK_SIZE
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def inspect_magic_bytes(
    path: Path,
) -> dict[str, str]:
    with path.open(
        "rb"
    ) as handle:
        first_32 = handle.read(
            32
        )

    hex_signature = (
        first_32.hex()
    )

    text_signature = (
        first_32.decode(
            "ascii",
            errors="replace",
        )
        .replace(
            "\r",
            "\\r",
        )
        .replace(
            "\n",
            "\\n",
        )
    )

    detected_type = (
        "unknown"
    )

    if first_32.startswith(
        b"PK\x03\x04"
    ):
        detected_type = (
            "ZIP_container_probable_XLSX"
        )

    elif first_32.startswith(
        b"\x1f\x8b"
    ):
        detected_type = (
            "gzip_compressed"
        )

    elif first_32.startswith(
        b"BZh"
    ):
        detected_type = (
            "bzip2_compressed"
        )

    elif first_32.startswith(
        b"\xfd7zXZ\x00"
    ):
        detected_type = (
            "xz_compressed"
        )

    elif (
        first_32.startswith(
            b"RDX"
        )
        or first_32.startswith(
            b"RDA"
        )
        or first_32.startswith(
            b"RDB"
        )
    ):
        detected_type = (
            "R_serialization"
        )

    elif first_32.startswith(
        b"X\n"
    ):
        detected_type = (
            "R_XDR_serialization"
        )

    return {
        "first_32_bytes_hex":
            hex_signature,

        "first_32_bytes_text":
            text_signature,

        "detected_file_signature":
            detected_type,
    }


def candidate_urls(
    file_id: str,
) -> list[str]:
    return [
        (
            f"{BASE_URL}/api/files/"
            f"{file_id}/blob"
        ),
        (
            f"{BASE_URL}/api/files/"
            f"{file_id}/download"
        ),
        (
            f"{BASE_URL}/files/"
            f"{file_id}/download"
        ),
    ]


def probe_candidate(
    url: str,
) -> dict[str, object]:
    request = urllib.request.Request(
        url=url,
        headers={
            "Accept":
                "application/octet-stream,*/*",

            "Range":
                "bytes=0-255",

            "Accept-Encoding":
                "identity",

            "User-Agent":
                (
                    "DTHI-Struct-Phase6/"
                    "1.0 EPA-HTTr-download"
                ),
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=90,
        ) as response:
            body = response.read(
                256
            )

            content_type = (
                response.headers.get(
                    "Content-Type",
                    ""
                )
            )

            final_url = (
                response.geturl()
            )

            looks_like_error_page = (
                "text/html"
                in content_type.lower()
                or "application/json"
                in content_type.lower()
                or body.lstrip().startswith(
                    b"<"
                )
                or body.lstrip().startswith(
                    b"{"
                )
            )

            return {
                "url":
                    url,

                "working":
                    bool(
                        response.status
                        in {
                            200,
                            206,
                        }
                        and body
                        and not looks_like_error_page
                    ),

                "http_status":
                    int(
                        response.status
                    ),

                "content_type":
                    content_type,

                "content_length":
                    response.headers.get(
                        "Content-Length",
                        ""
                    ),

                "content_range":
                    response.headers.get(
                        "Content-Range",
                        ""
                    ),

                "content_disposition":
                    response.headers.get(
                        "Content-Disposition",
                        ""
                    ),

                "final_url":
                    final_url,

                "probe_bytes":
                    len(
                        body
                    ),

                "error":
                    "",
            }

    except Exception as error:
        return {
            "url":
                url,

            "working":
                False,

            "http_status":
                "",

            "content_type":
                "",

            "content_length":
                "",

            "content_range":
                "",

            "content_disposition":
                "",

            "final_url":
                "",

            "probe_bytes":
                0,

            "error":
                (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
        }


def resolve_download_url(
    file_id: str,
) -> tuple[str, list[dict[str, object]]]:
    probe_results = []

    for url in candidate_urls(
        file_id
    ):
        result = probe_candidate(
            url
        )

        probe_results.append(
            result
        )

        if result[
            "working"
        ]:
            return (
                url,
                probe_results,
            )

    raise RuntimeError(
        (
            "No public binary-download route "
            f"worked for file {file_id}. "
            f"Probe results: {probe_results}"
        )
    )


def download_file(
    url: str,
    output_path: Path,
    expected_size: int,
) -> dict[str, object]:
    temporary_path = Path(
        str(
            output_path
        )
        + ".part"
    )

    temporary_path.unlink(
        missing_ok=True
    )

    request = urllib.request.Request(
        url=url,
        headers={
            "Accept":
                "application/octet-stream,*/*",

            "Accept-Encoding":
                "identity",

            "User-Agent":
                (
                    "DTHI-Struct-Phase6/"
                    "1.0 EPA-HTTr-download"
                ),
        },
        method="GET",
    )

    started = time.time()

    downloaded_bytes = 0

    digest = hashlib.sha256()

    with urllib.request.urlopen(
        request,
        timeout=180,
    ) as response:
        content_type = (
            response.headers.get(
                "Content-Type",
                ""
            )
        )

        content_disposition = (
            response.headers.get(
                "Content-Disposition",
                ""
            )
        )

        declared_length_text = (
            response.headers.get(
                "Content-Length",
                ""
            )
        )

        try:
            declared_length = int(
                declared_length_text
            )

        except (
            TypeError,
            ValueError,
        ):
            declared_length = 0

        if (
            declared_length
            > MAXIMUM_FILE_BYTES
        ):
            raise RuntimeError(
                (
                    "Server-declared file size exceeds "
                    "the 500 MB safety limit: "
                    f"{declared_length}"
                )
            )

        with temporary_path.open(
            "wb"
        ) as output_handle:
            while True:
                chunk = response.read(
                    CHUNK_SIZE
                )

                if not chunk:
                    break

                downloaded_bytes += len(
                    chunk
                )

                if (
                    downloaded_bytes
                    > MAXIMUM_FILE_BYTES
                ):
                    raise RuntimeError(
                        (
                            "Downloaded file exceeded the "
                            "500 MB safety limit."
                        )
                    )

                digest.update(
                    chunk
                )

                output_handle.write(
                    chunk
                )

    if downloaded_bytes == 0:
        raise RuntimeError(
            "Downloaded file is empty."
        )

    if (
        expected_size > 0
        and downloaded_bytes
        != expected_size
    ):
        raise RuntimeError(
            (
                "Downloaded size does not match "
                "the Clowder manifest: "
                f"expected={expected_size}, "
                f"downloaded={downloaded_bytes}"
            )
        )

    os.replace(
        temporary_path,
        output_path,
    )

    return {
        "downloaded_bytes":
            downloaded_bytes,

        "sha256":
            digest.hexdigest(),

        "HTTP_content_type":
            content_type,

        "HTTP_content_disposition":
            content_disposition,

        "HTTP_declared_length":
            declared_length,

        "download_elapsed_seconds":
            time.time()
            - started,

        "final_download_url":
            response.geturl(),
    }


def existing_file_valid(
    path: Path,
    expected_size: int,
) -> bool:
    return bool(
        path.exists()
        and path.is_file()
        and path.stat().st_size
        == expected_size
        and expected_size > 0
    )


def main() -> None:
    LOG_FILE.write_text(
        "",
        encoding="utf-8",
    )

    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing input manifest: {INPUT_MANIFEST}"
        )

    manifest = pd.read_csv(
        INPUT_MANIFEST,
        sep="\t",
        dtype="string",
        low_memory=False,
    )

    if len(
        manifest
    ) != 2:
        raise RuntimeError(
            (
                "Expected exactly two files in the "
                "resolved EPA HTTr folder, found "
                f"{len(manifest)}."
            )
        )

    found_filenames = set(
        manifest[
            "filename"
        ].astype(
            str
        )
    )

    if found_filenames != ALLOWED_FILENAMES:
        raise RuntimeError(
            (
                "Resolved filenames do not match "
                "the expected EPA HTTr files. "
                f"Found: {sorted(found_filenames)}"
            )
        )

    log(
        "===== Phase 6B3 started ====="
    )

    output_rows = []

    for _, record in manifest.iterrows():
        file_id = str(
            record[
                "file_id"
            ]
        ).strip()

        filename = safe_filename(
            str(
                record[
                    "filename"
                ]
            )
        )

        expected_size = int(
            float(
                record[
                    "size_bytes"
                ]
            )
        )

        output_path = (
            DOWNLOAD_DIR
            / filename
        )

        log()
        log(
            f"File: {filename}"
        )

        log(
            f"File ID: {file_id}"
        )

        log(
            (
                "Expected size: "
                f"{expected_size:,} bytes"
            )
        )

        (
            selected_url,
            probe_results,
        ) = resolve_download_url(
            file_id
        )

        log(
            f"Selected route: {selected_url}"
        )

        if existing_file_valid(
            output_path,
            expected_size,
        ):
            log(
                "Existing local file has the expected size; reusing it."
            )

            download_status = (
                "reused_existing_verified_size"
            )

            download_summary = {
                "downloaded_bytes":
                    output_path.stat().st_size,

                "sha256":
                    sha256_file(
                        output_path
                    ),

                "HTTP_content_type":
                    "",

                "HTTP_content_disposition":
                    "",

                "HTTP_declared_length":
                    "",

                "download_elapsed_seconds":
                    0,

                "final_download_url":
                    selected_url,
            }

        else:
            output_path.unlink(
                missing_ok=True
            )

            download_summary = (
                download_file(
                    selected_url,
                    output_path,
                    expected_size,
                )
            )

            download_status = (
                "downloaded_and_size_verified"
            )

        magic = inspect_magic_bytes(
            output_path
        )

        actual_size = (
            output_path.stat().st_size
        )

        size_matches = (
            actual_size
            == expected_size
        )

        log(
            f"Actual size: {actual_size:,} bytes"
        )

        log(
            f"SHA256: {download_summary['sha256']}"
        )

        log(
            (
                "Detected signature: "
                f"{magic['detected_file_signature']}"
            )
        )

        output_rows.append(
            {
                "file_id":
                    file_id,

                "filename":
                    filename,

                "expected_size_bytes":
                    expected_size,

                "actual_size_bytes":
                    actual_size,

                "size_matches_manifest":
                    size_matches,

                "local_relative_path":
                    str(
                        output_path.relative_to(
                            PROJECT
                        )
                    ),

                "selected_download_url":
                    selected_url,

                "final_download_url":
                    download_summary[
                        "final_download_url"
                    ],

                "HTTP_content_type":
                    download_summary[
                        "HTTP_content_type"
                    ],

                "HTTP_content_disposition":
                    download_summary[
                        "HTTP_content_disposition"
                    ],

                "HTTP_declared_length":
                    download_summary[
                        "HTTP_declared_length"
                    ],

                "sha256":
                    download_summary[
                        "sha256"
                    ],

                **magic,

                "download_elapsed_seconds":
                    download_summary[
                        "download_elapsed_seconds"
                    ],

                "download_status":
                    download_status,

                "route_probe_results":
                    str(
                        probe_results
                    ),
            }
        )

    output_manifest = pd.DataFrame(
        output_rows
    )

    output_manifest.to_csv(
        DOWNLOAD_MANIFEST,
        sep="\t",
        index=False,
    )

    all_size_valid = bool(
        output_manifest[
            "size_matches_manifest"
        ].eq(
            True
        ).all()
    )

    xlsx_valid = bool(
        output_manifest.loc[
            output_manifest[
                "filename"
            ].str.endswith(
                ".xlsx"
            ),
            "detected_file_signature",
        ].eq(
            "ZIP_container_probable_XLSX"
        ).all()
    )

    rdata_present = bool(
        output_manifest.loc[
            output_manifest[
                "filename"
            ].str.endswith(
                ".RData"
            ),
            "actual_size_bytes",
        ].gt(
            0
        ).all()
    )

    completion_status = (
        "completed"
        if (
            len(
                output_manifest
            )
            == 2
            and all_size_valid
            and xlsx_valid
            and rdata_present
        )
        else "failed_validation"
    )

    completion = pd.DataFrame(
        [
            {
                "files_expected":
                    2,

                "files_present":
                    len(
                        output_manifest
                    ),

                "total_expected_bytes":
                    int(
                        output_manifest[
                            "expected_size_bytes"
                        ].sum()
                    ),

                "total_actual_bytes":
                    int(
                        output_manifest[
                            "actual_size_bytes"
                        ].sum()
                    ),

                "all_sizes_match_manifest":
                    all_size_valid,

                "XLSX_signature_valid":
                    xlsx_valid,

                "RData_file_nonempty":
                    rdata_present,

                "downloaded_utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                "Phase6B3_status":
                    completion_status,
            }
        ]
    )

    completion.to_csv(
        COMPLETION_FILE,
        sep="\t",
        index=False,
    )

    log()
    log(
        "===== DOWNLOAD MANIFEST ====="
    )

    log(
        output_manifest[
            [
                "filename",
                "expected_size_bytes",
                "actual_size_bytes",
                "size_matches_manifest",
                "detected_file_signature",
                "sha256",
                "download_status",
            ]
        ].to_string(
            index=False
        )
    )

    log()
    log(
        "===== PHASE 6B3 COMPLETION ====="
    )

    log(
        completion.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        log(
            (
                "Phase 6B3 failed: "
                f"{type(error).__name__}: "
                f"{error}"
            )
        )

        raise
