#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

import pandas as pd


PROJECT = Path(
    "."
)

PREVIEW_DIR = (
    PROJECT
    / "02_metadata/phase6/"
      "clowder_endpoint_probe"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase6"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase6/"
      "phase6B2B_partial_response_inspection.log"
)

OUTPUT_FILE = (
    TABLE_DIR
    / "phase6B2B_partial_response_inspection.tsv"
)

KEY_FILE = (
    TABLE_DIR
    / "phase6B2B_detected_JSON_keys.tsv"
)

COMPLETION_FILE = (
    TABLE_DIR
    / "phase6B2B_completion_summary.tsv"
)

TARGET_FOLDER_ID = (
    "63652dc3e4b04f6bb140b088"
)

PREVIEW_FILES = sorted(
    PREVIEW_DIR.glob(
        "dataset_files_folder_query_*.txt"
    )
)

for directory in [
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


def sha256_bytes(
    data: bytes,
) -> str:
    return hashlib.sha256(
        data
    ).hexdigest()


def escaped_preview(
    text: str,
    length: int = 800,
) -> str:
    return (
        text[:length]
        .replace(
            "\r",
            "\\r",
        )
        .replace(
            "\n",
            "\\n",
        )
        .replace(
            "\t",
            "\\t",
        )
    )


def detect_json_keys(
    text: str,
) -> list[str]:
    keys = re.findall(
        r'"([^"\\]{1,100})"\s*:',
        text,
    )

    return sorted(
        set(
            keys
        )
    )


def decode_first_array_item(
    text: str,
) -> tuple[bool, object | None, str]:
    stripped = text.lstrip()

    if not stripped.startswith(
        "["
    ):
        return (
            False,
            None,
            (
                "Top-level response does not "
                "begin with a JSON array."
            ),
        )

    array_start = text.find(
        "["
    )

    item_start = array_start + 1

    while (
        item_start < len(
            text
        )
        and text[
            item_start
        ].isspace()
    ):
        item_start += 1

    try:
        item, _ = (
            json.JSONDecoder()
            .raw_decode(
                text,
                item_start,
            )
        )

        return (
            True,
            item,
            "",
        )

    except json.JSONDecodeError as error:
        return (
            False,
            None,
            (
                f"Unable to decode first array "
                f"item: {error}"
            ),
        )


def flatten_selected_fields(
    value: object,
    prefix: str = "",
) -> dict[str, str]:
    selected = {}

    target_fragments = {
        "id",
        "_id",
        "name",
        "filename",
        "fileName",
        "folder",
        "folderId",
        "folder_id",
        "dataset",
        "datasetId",
        "size",
        "contentType",
        "uploadDate",
        "created",
    }

    if isinstance(
        value,
        dict,
    ):
        for key, item in value.items():
            path = (
                f"{prefix}.{key}"
                if prefix
                else key
            )

            if key in target_fragments:
                selected[
                    path
                ] = str(
                    item
                )

            if isinstance(
                item,
                (
                    dict,
                    list,
                ),
            ):
                selected.update(
                    flatten_selected_fields(
                        item,
                        path,
                    )
                )

    elif isinstance(
        value,
        list,
    ):
        for index, item in enumerate(
            value[:5]
        ):
            path = (
                f"{prefix}[{index}]"
            )

            selected.update(
                flatten_selected_fields(
                    item,
                    path,
                )
            )

    return selected


def main() -> None:
    LOG_FILE.write_text(
        "",
        encoding="utf-8",
    )

    if not PREVIEW_FILES:
        raise FileNotFoundError(
            (
                "No dataset-files response "
                f"previews found in {PREVIEW_DIR}"
            )
        )

    rows = []
    key_rows = []
    hashes = []

    log(
        "===== Phase 6B2B started ====="
    )

    for path in PREVIEW_FILES:
        raw = path.read_bytes()

        text = raw.decode(
            "utf-8",
            errors="replace",
        )

        digest = sha256_bytes(
            raw
        )

        hashes.append(
            digest
        )

        keys = detect_json_keys(
            text
        )

        (
            first_item_decoded,
            first_item,
            first_item_error,
        ) = decode_first_array_item(
            text
        )

        selected_fields = {}

        first_item_type = ""

        if first_item_decoded:
            first_item_type = type(
                first_item
            ).__name__

            selected_fields = (
                flatten_selected_fields(
                    first_item
                )
            )

        target_occurrences = text.count(
            TARGET_FOLDER_ID
        )

        folder_key_occurrences = len(
            re.findall(
                (
                    r'"(?:folder|folderId|'
                    r'folder_id)"\s*:'
                ),
                text,
                flags=re.IGNORECASE,
            )
        )

        filename_key_occurrences = len(
            re.findall(
                (
                    r'"(?:filename|fileName|name)"'
                    r'\s*:'
                ),
                text,
                flags=re.IGNORECASE,
            )
        )

        id_candidates = re.findall(
            r'"(?:id|_id)"\s*:\s*"([^"]+)"',
            text,
        )

        filename_candidates = re.findall(
            (
                r'"(?:filename|fileName|name)"'
                r'\s*:\s*"([^"]+)"'
            ),
            text,
            flags=re.IGNORECASE,
        )

        rows.append(
            {
                "preview_file":
                    str(
                        path.relative_to(
                            PROJECT
                        )
                    ),

                "size_bytes":
                    len(
                        raw
                    ),

                "sha256":
                    digest,

                "opening_character":
                    text.lstrip()[:1],

                "opening_preview":
                    escaped_preview(
                        text
                    ),

                "detected_unique_JSON_keys":
                    len(
                        keys
                    ),

                "target_folder_ID_occurrences":
                    target_occurrences,

                "folder_key_occurrences":
                    folder_key_occurrences,

                "filename_key_occurrences":
                    filename_key_occurrences,

                "first_item_decoded":
                    first_item_decoded,

                "first_item_type":
                    first_item_type,

                "first_item_selected_fields":
                    json.dumps(
                        selected_fields,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),

                "first_item_decode_error":
                    first_item_error,

                "first_10_ID_candidates":
                    "|".join(
                        id_candidates[:10]
                    ),

                "first_10_filename_candidates":
                    "|".join(
                        filename_candidates[:10]
                    ),
            }
        )

        for key in keys:
            key_rows.append(
                {
                    "preview_file":
                        path.name,

                    "JSON_key":
                        key,
                }
            )

        log()
        log(
            f"===== {path.name} ====="
        )

        log(
            f"Size: {len(raw)} bytes"
        )

        log(
            f"SHA256: {digest}"
        )

        log(
            (
                "Target folder ID occurrences: "
                f"{target_occurrences}"
            )
        )

        log(
            (
                "Folder-related key occurrences: "
                f"{folder_key_occurrences}"
            )
        )

        log(
            (
                "First array item decoded: "
                f"{first_item_decoded}"
            )
        )

        log(
            "Opening preview:"
        )

        log(
            escaped_preview(
                text
            )
        )

        if selected_fields:
            log(
                "First item selected fields:"
            )

            log(
                json.dumps(
                    selected_fields,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )

        if first_item_error:
            log(
                first_item_error
            )

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
    )

    key_table = pd.DataFrame(
        key_rows
    )

    key_table.to_csv(
        KEY_FILE,
        sep="\t",
        index=False,
    )

    all_previews_identical = (
        len(
            set(
                hashes
            )
        )
        == 1
    )

    target_folder_present = bool(
        results[
            "target_folder_ID_occurrences"
        ].gt(
            0
        ).any()
    )

    folder_keys_present = bool(
        results[
            "folder_key_occurrences"
        ].gt(
            0
        ).any()
    )

    first_item_decoded_any = bool(
        results[
            "first_item_decoded"
        ].eq(
            True
        ).any()
    )

    completion = pd.DataFrame(
        [
            {
                "preview_files_inspected":
                    len(
                        results
                    ),

                "all_query_responses_identical":
                    all_previews_identical,

                "target_folder_ID_present":
                    target_folder_present,

                "folder_related_keys_present":
                    folder_keys_present,

                "first_complete_file_record_decoded":
                    first_item_decoded_any,

                "full_data_files_downloaded":
                    0,

                "Phase6B2B_status":
                    "completed",
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
        "===== QUERY RESPONSE COMPARISON ====="
    )

    log(
        completion.to_string(
            index=False
        )
    )

    log()
    log(
        "===== DETECTED JSON KEYS ====="
    )

    if key_table.empty:
        log(
            "No JSON-style keys detected."
        )

    else:
        log(
            (
                key_table[
                    "JSON_key"
                ]
                .value_counts()
                .rename_axis(
                    "JSON_key"
                )
                .reset_index(
                    name="preview_count"
                )
                .to_string(
                    index=False
                )
            )
        )

    log()
    log(
        "===== Phase 6B2B completed ====="
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        log(
            (
                "Phase 6B2B failed: "
                f"{type(error).__name__}: "
                f"{error}"
            )
        )

        raise
