#!/usr/bin/env python3

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request

import ijson
import pandas as pd


PROJECT = Path(
    "."
)

BASE_URL = "https://clowder.edap-cluster.com"

DATASET_ID = "61147fefe4b0856fdc65639b"

TARGET_FOLDER_ID = "63652dc3e4b04f6bb140b088"

MANIFEST_URL = (
    f"{BASE_URL}/api/datasets/"
    f"{DATASET_ID}/files"
)

METADATA_DIR = (
    PROJECT
    / "02_metadata/phase6/clowder_complete_manifest"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase6"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase6/"
      "phase6B2C_complete_manifest_filter.log"
)

RAW_MANIFEST_GZ = (
    METADATA_DIR
    / (
        "EPA_HTTr_Clowder_dataset_"
        f"{DATASET_ID}_files.json.gz"
    )
)

RESPONSE_HEADERS_FILE = (
    METADATA_DIR
    / "phase6B2C_HTTP_response_headers.json"
)

SAMPLE_RECORDS_FILE = (
    METADATA_DIR
    / "phase6B2C_first_five_file_records.json"
)

TARGET_MANIFEST_FILE = (
    TABLE_DIR
    / "phase6B2C_target_folder_file_manifest.tsv"
)

KEYWORD_MANIFEST_FILE = (
    TABLE_DIR
    / "phase6B2C_HTTr_keyword_candidate_file_manifest.tsv"
)

FOLDER_TOKEN_FILE = (
    TABLE_DIR
    / "phase6B2C_folder_token_counts.tsv"
)

RECORD_KEY_FILE = (
    TABLE_DIR
    / "phase6B2C_file_record_key_counts.tsv"
)

COMPLETION_FILE = (
    TABLE_DIR
    / "phase6B2C_completion_summary.tsv"
)

for directory in [
    METADATA_DIR,
    TABLE_DIR,
    LOG_FILE.parent,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


MAXIMUM_RAW_BYTES = 2 * 1024 ** 3

CHUNK_SIZE = 1024 * 1024

HEX24_PATTERN = re.compile(
    r"\b[a-fA-F0-9]{24}\b"
)

KEYWORD_PATTERN = re.compile(
    (
        r"httr|"
        r"high[\s_-]*throughput[\s_-]*transcript|"
        r"transcriptomic|"
        r"temposeq|"
        r"tempo[\s_-]*seq|"
        r"gene[\s_-]*expression|"
        r"bmd|"
        r"benchmark[\s_-]*dose"
    ),
    flags=re.IGNORECASE,
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


def safe_text(value: object) -> str:
    if value is None:
        return ""

    if isinstance(
        value,
        (
            dict,
            list,
        ),
    ):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    return str(value)


def safe_integer(value: object) -> int:
    try:
        return int(value)

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return 0


def collect_folder_information(
    value: object,
) -> tuple[set[str], set[str]]:
    folder_ids: set[str] = set()
    folder_names: set[str] = set()

    def walk(
        item: object,
        parent_key: str = "",
    ) -> None:
        if isinstance(
            item,
            dict,
        ):
            for key, nested_value in item.items():
                normalized_key = (
                    str(key)
                    .strip()
                    .lower()
                    .replace(
                        "-",
                        "_",
                    )
                )

                if normalized_key in {
                    "id",
                    "_id",
                    "folderid",
                    "folder_id",
                }:
                    text = safe_text(
                        nested_value
                    ).strip()

                    if text:
                        folder_ids.add(
                            text
                        )

                if normalized_key in {
                    "name",
                    "foldername",
                    "folder_name",
                    "path",
                    "label",
                    "title",
                }:
                    text = safe_text(
                        nested_value
                    ).strip()

                    if text:
                        folder_names.add(
                            text
                        )

                walk(
                    nested_value,
                    normalized_key,
                )

        elif isinstance(
            item,
            list,
        ):
            for nested_value in item:
                walk(
                    nested_value,
                    parent_key,
                )

        elif isinstance(
            item,
            str,
        ):
            for possible_id in HEX24_PATTERN.findall(
                item
            ):
                folder_ids.add(
                    possible_id
                )

            if parent_key in {
                "name",
                "foldername",
                "folder_name",
                "path",
                "label",
                "title",
            }:
                text = item.strip()

                if text:
                    folder_names.add(
                        text
                    )

    walk(
        value
    )

    return (
        folder_ids,
        folder_names,
    )


def identify_filename(
    record: dict[str, object],
) -> str:
    for key in [
        "filename",
        "fileName",
        "name",
    ]:
        value = safe_text(
            record.get(
                key
            )
        ).strip()

        if value:
            return value

    return ""


def identify_file_id(
    record: dict[str, object],
) -> str:
    for key in [
        "id",
        "_id",
        "fileId",
        "file_id",
    ]:
        value = safe_text(
            record.get(
                key
            )
        ).strip()

        if value:
            return value

    return ""


def identify_size(
    record: dict[str, object],
) -> int:
    for key in [
        "size",
        "length",
        "fileSize",
        "file_size",
    ]:
        if key in record:
            return safe_integer(
                record.get(
                    key
                )
            )

    return 0


def identify_content_type(
    record: dict[str, object],
) -> str:
    for key in [
        "contentType",
        "content_type",
        "mimeType",
        "mime_type",
    ]:
        value = safe_text(
            record.get(
                key
            )
        ).strip()

        if value:
            return value

    return ""


def identify_created_date(
    record: dict[str, object],
) -> str:
    for key in [
        "date-created",
        "date_created",
        "created",
        "uploadDate",
        "upload_date",
    ]:
        value = safe_text(
            record.get(
                key
            )
        ).strip()

        if value:
            return value

    return ""


def download_complete_manifest() -> dict[str, object]:
    temporary_file = Path(
        str(
            RAW_MANIFEST_GZ
        )
        + ".part"
    )

    temporary_file.unlink(
        missing_ok=True
    )

    headers = {
        "Accept":
            "application/json",

        "Accept-Encoding":
            "identity",

        "User-Agent":
            (
                "DTHI-Struct-Phase6/"
                "1.0 Clowder-metadata-inventory"
            ),
    }

    request = urllib.request.Request(
        url=MANIFEST_URL,
        headers=headers,
        method="GET",
    )

    started = time.time()

    raw_bytes = 0

    digest = hashlib.sha256()

    with urllib.request.urlopen(
        request,
        timeout=180,
    ) as response:
        response_headers = dict(
            response.headers.items()
        )

        response_headers[
            "HTTP_status"
        ] = int(
            response.status
        )

        RESPONSE_HEADERS_FILE.write_text(
            json.dumps(
                response_headers,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        declared_length = safe_integer(
            response.headers.get(
                "Content-Length"
            )
        )

        if (
            declared_length > 0
            and declared_length
            > MAXIMUM_RAW_BYTES
        ):
            raise RuntimeError(
                (
                    "Clowder metadata response exceeds "
                    "the configured 2 GB safety limit: "
                    f"{declared_length} bytes"
                )
            )

        with gzip.open(
            temporary_file,
            mode="wb",
            compresslevel=6,
        ) as output_handle:
            while True:
                chunk = response.read(
                    CHUNK_SIZE
                )

                if not chunk:
                    break

                raw_bytes += len(
                    chunk
                )

                if raw_bytes > MAXIMUM_RAW_BYTES:
                    raise RuntimeError(
                        (
                            "Clowder metadata response exceeded "
                            "the configured 2 GB safety limit."
                        )
                    )

                digest.update(
                    chunk
                )

                output_handle.write(
                    chunk
                )

                if (
                    raw_bytes
                    % (
                        100
                        * 1024
                        * 1024
                    )
                    < CHUNK_SIZE
                ):
                    log(
                        (
                            "Downloaded metadata: "
                            f"{raw_bytes / 1024 ** 2:.1f} MB"
                        )
                    )

    os.replace(
        temporary_file,
        RAW_MANIFEST_GZ,
    )

    return {
        "manifest_reused":
            False,

        "raw_manifest_bytes":
            raw_bytes,

        "compressed_manifest_bytes":
            RAW_MANIFEST_GZ.stat().st_size,

        "raw_manifest_sha256":
            digest.hexdigest(),

        "download_elapsed_seconds":
            time.time()
            - started,
    }


def reuse_existing_manifest() -> dict[str, object]:
    return {
        "manifest_reused":
            True,

        "raw_manifest_bytes":
            "",

        "compressed_manifest_bytes":
            RAW_MANIFEST_GZ.stat().st_size,

        "raw_manifest_sha256":
            "",

        "download_elapsed_seconds":
            0,
    }


def parse_manifest() -> dict[str, object]:
    target_rows: list[
        dict[str, object]
    ] = []

    keyword_rows: list[
        dict[str, object]
    ] = []

    first_records: list[
        dict[str, object]
    ] = []

    folder_token_counts: Counter[str] = (
        Counter()
    )

    record_key_counts: Counter[str] = (
        Counter()
    )

    total_records = 0

    records_with_folders = 0

    target_id_occurrences = 0

    with gzip.open(
        RAW_MANIFEST_GZ,
        mode="rb",
    ) as input_handle:
        for record in ijson.items(
            input_handle,
            "item",
        ):
            if not isinstance(
                record,
                dict,
            ):
                continue

            total_records += 1

            record_key_counts.update(
                str(key)
                for key in record.keys()
            )

            if len(
                first_records
            ) < 5:
                first_records.append(
                    record
                )

            filename = identify_filename(
                record
            )

            file_id = identify_file_id(
                record
            )

            size_bytes = identify_size(
                record
            )

            content_type = (
                identify_content_type(
                    record
                )
            )

            created_date = (
                identify_created_date(
                    record
                )
            )

            folder_value = record.get(
                "folders",
                [],
            )

            if folder_value not in [
                None,
                [],
                {},
                "",
            ]:
                records_with_folders += 1

            (
                folder_ids,
                folder_names,
            ) = collect_folder_information(
                folder_value
            )

            folder_token_counts.update(
                folder_ids
            )

            serialized_folders = json.dumps(
                folder_value,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )

            target_match = (
                TARGET_FOLDER_ID
                in folder_ids
                or TARGET_FOLDER_ID
                in serialized_folders
            )

            if target_match:
                target_id_occurrences += 1

            common_row = {
                "file_id":
                    file_id,

                "filename":
                    filename,

                "size_bytes":
                    size_bytes,

                "size_megabytes":
                    size_bytes
                    / 1024
                    / 1024,

                "content_type":
                    content_type,

                "created_date":
                    created_date,

                "folder_ids":
                    "|".join(
                        sorted(
                            folder_ids
                        )
                    ),

                "folder_names":
                    "|".join(
                        sorted(
                            folder_names
                        )
                    ),

                "folder_metadata_json":
                    serialized_folders,
            }

            if target_match:
                target_rows.append(
                    common_row
                )

            keyword_text = " ".join(
                [
                    filename,
                    " ".join(
                        sorted(
                            folder_names
                        )
                    ),
                    serialized_folders,
                ]
            )

            if KEYWORD_PATTERN.search(
                keyword_text
            ):
                keyword_rows.append(
                    {
                        **common_row,

                        "matched_keyword_context":
                            keyword_text[
                                :1000
                            ],
                    }
                )

    SAMPLE_RECORDS_FILE.write_text(
        json.dumps(
            first_records,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    target_manifest = pd.DataFrame(
        target_rows
    )

    keyword_manifest = pd.DataFrame(
        keyword_rows
    )

    manifest_columns = [
        "file_id",
        "filename",
        "size_bytes",
        "size_megabytes",
        "content_type",
        "created_date",
        "folder_ids",
        "folder_names",
        "folder_metadata_json",
    ]

    if target_manifest.empty:
        target_manifest = pd.DataFrame(
            columns=manifest_columns
        )

    else:
        target_manifest = (
            target_manifest[
                manifest_columns
            ]
            .sort_values(
                [
                    "size_bytes",
                    "filename",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(
                drop=True
            )
        )

    target_manifest.to_csv(
        TARGET_MANIFEST_FILE,
        sep="\t",
        index=False,
    )

    keyword_columns = (
        manifest_columns
        + [
            "matched_keyword_context",
        ]
    )

    if keyword_manifest.empty:
        keyword_manifest = pd.DataFrame(
            columns=keyword_columns
        )

    else:
        keyword_manifest = (
            keyword_manifest[
                keyword_columns
            ]
            .sort_values(
                [
                    "size_bytes",
                    "filename",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(
                drop=True
            )
        )

    keyword_manifest.to_csv(
        KEYWORD_MANIFEST_FILE,
        sep="\t",
        index=False,
    )

    folder_table = pd.DataFrame(
        [
            {
                "folder_token":
                    token,

                "file_record_count":
                    count,

                "is_target_folder_ID":
                    token
                    == TARGET_FOLDER_ID,
            }
            for token, count
            in folder_token_counts.items()
        ]
    )

    if folder_table.empty:
        folder_table = pd.DataFrame(
            columns=[
                "folder_token",
                "file_record_count",
                "is_target_folder_ID",
            ]
        )

    else:
        folder_table = (
            folder_table.sort_values(
                [
                    "file_record_count",
                    "folder_token",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(
                drop=True
            )
        )

    folder_table.to_csv(
        FOLDER_TOKEN_FILE,
        sep="\t",
        index=False,
    )

    key_table = pd.DataFrame(
        [
            {
                "record_key":
                    key,

                "file_record_count":
                    count,
            }
            for key, count
            in record_key_counts.items()
        ]
    )

    if not key_table.empty:
        key_table = (
            key_table.sort_values(
                [
                    "file_record_count",
                    "record_key",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(
                drop=True
            )
        )

    key_table.to_csv(
        RECORD_KEY_FILE,
        sep="\t",
        index=False,
    )

    return {
        "total_file_records":
            total_records,

        "records_with_folder_metadata":
            records_with_folders,

        "unique_folder_tokens":
            len(
                folder_token_counts
            ),

        "target_folder_file_records":
            len(
                target_manifest
            ),

        "target_folder_ID_occurrences":
            target_id_occurrences,

        "keyword_candidate_file_records":
            len(
                keyword_manifest
            ),

        "target_folder_total_size_bytes":
            int(
                target_manifest[
                    "size_bytes"
                ].sum()
            )
            if not target_manifest.empty
            else 0,
    }


def main() -> None:
    LOG_FILE.write_text(
        "",
        encoding="utf-8",
    )

    log(
        "===== Phase 6B2C started ====="
    )

    log(
        f"Manifest endpoint: {MANIFEST_URL}"
    )

    log(
        f"Target folder ID: {TARGET_FOLDER_ID}"
    )

    if (
        RAW_MANIFEST_GZ.exists()
        and RAW_MANIFEST_GZ.stat().st_size
        > 0
    ):
        log(
            "Reusing existing compressed metadata manifest."
        )

        download_summary = (
            reuse_existing_manifest()
        )

    else:
        log(
            (
                "Downloading complete Clowder JSON "
                "metadata manifest only..."
            )
        )

        download_summary = (
            download_complete_manifest()
        )

    log(
        (
            "Compressed manifest size: "
            f"{RAW_MANIFEST_GZ.stat().st_size / 1024 ** 2:.2f} MB"
        )
    )

    log(
        "Streaming and filtering file records..."
    )

    parsing_summary = parse_manifest()

    target_count = int(
        parsing_summary[
            "target_folder_file_records"
        ]
    )

    keyword_count = int(
        parsing_summary[
            "keyword_candidate_file_records"
        ]
    )

    if target_count > 0:
        status = (
            "completed_target_folder_resolved"
        )

    elif keyword_count > 0:
        status = (
            "completed_target_not_resolved_"
            "keyword_candidates_found"
        )

    else:
        status = (
            "completed_target_not_resolved_"
            "no_keyword_candidates"
        )

    completion = pd.DataFrame(
        [
            {
                "dataset_id":
                    DATASET_ID,

                "target_folder_id":
                    TARGET_FOLDER_ID,

                "retrieved_utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                **download_summary,
                **parsing_summary,

                "HTTr_data_files_downloaded":
                    0,

                "Phase6B2C_status":
                    status,
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
        "===== PHASE 6B2C COMPLETION ====="
    )

    log(
        completion.to_string(
            index=False
        )
    )

    log()
    log(
        "===== TARGET-FOLDER FILES ====="
    )

    target_manifest = pd.read_csv(
        TARGET_MANIFEST_FILE,
        sep="\t",
        dtype="string",
    )

    if target_manifest.empty:
        log(
            "No file records directly matched the target folder ID."
        )

    else:
        log(
            target_manifest[
                [
                    "file_id",
                    "filename",
                    "size_megabytes",
                    "content_type",
                    "folder_names",
                ]
            ]
            .head(
                100
            )
            .to_string(
                index=False
            )
        )

    log()
    log(
        "===== Phase 6B2C completed ====="
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        log(
            (
                "Phase 6B2C failed: "
                f"{type(error).__name__}: "
                f"{error}"
            )
        )

        raise
