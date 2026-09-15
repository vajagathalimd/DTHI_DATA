#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd


PROJECT = Path(
    "."
)

BASE_URL = (
    "https://clowder.edap-cluster.com"
)

DATASET_ID = (
    "61147fefe4b0856fdc65639b"
)

TARGET_FOLDER_ID = (
    "63652dc3e4b04f6bb140b088"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase6"
)

METADATA_DIR = (
    PROJECT
    / "02_metadata/phase6/"
      "clowder_endpoint_probe"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase6/"
      "phase6B2A_clowder_endpoint_probe.log"
)

OUTPUT_FILE = (
    TABLE_DIR
    / "phase6B2A_clowder_endpoint_probe.tsv"
)

COMPLETION_FILE = (
    TABLE_DIR
    / "phase6B2A_completion_summary.tsv"
)

for directory in [
    TABLE_DIR,
    METADATA_DIR,
    LOG_FILE.parent,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


MAX_RESPONSE_BYTES = 262_144


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


def response_structure(
    payload: object,
) -> tuple[str, int, str]:
    if isinstance(
        payload,
        list,
    ):
        preview = ""

        if payload:
            first = payload[0]

            if isinstance(
                first,
                dict,
            ):
                preview = "|".join(
                    sorted(
                        first.keys()
                    )
                )

            else:
                preview = type(
                    first
                ).__name__

        return (
            "list",
            len(
                payload
            ),
            preview,
        )

    if isinstance(
        payload,
        dict,
    ):
        return (
            "dict",
            len(
                payload
            ),
            "|".join(
                sorted(
                    payload.keys()
                )
            ),
        )

    return (
        type(
            payload
        ).__name__,
        1,
        "",
    )


def probe_url(
    probe_name: str,
    url: str,
) -> dict[str, object]:
    headers = {
        "Accept":
            "application/json,text/plain,*/*",

        "Accept-Encoding":
            "identity",

        "User-Agent":
            (
                "DTHI-Struct-Phase6/"
                "1.0 EPA-Clowder-inventory"
            ),
    }

    request = urllib.request.Request(
        url=url,
        headers=headers,
        method="GET",
    )

    started = time.time()

    try:
        with urllib.request.urlopen(
            request,
            timeout=90,
        ) as response:
            status = int(
                response.status
            )

            content_type = (
                response.headers.get(
                    "Content-Type",
                    "",
                )
            )

            declared_length = (
                response.headers.get(
                    "Content-Length",
                    "",
                )
            )

            body = response.read(
                MAX_RESPONSE_BYTES
            )

            elapsed = (
                time.time()
                - started
            )

            text = body.decode(
                "utf-8",
                errors="replace",
            )

            preview_file = (
                METADATA_DIR
                / f"{probe_name}.txt"
            )

            preview_file.write_text(
                text,
                encoding="utf-8",
            )

            json_valid = False
            json_type = ""
            json_items = 0
            json_keys = ""

            try:
                payload = json.loads(
                    text
                )

                json_valid = True

                (
                    json_type,
                    json_items,
                    json_keys,
                ) = response_structure(
                    payload
                )

                json_file = (
                    METADATA_DIR
                    / f"{probe_name}.json"
                )

                json_file.write_text(
                    json.dumps(
                        payload,
                        indent=2,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )

            except json.JSONDecodeError:
                pass

            return {
                "probe_name":
                    probe_name,

                "url":
                    url,

                "http_status":
                    status,

                "content_type":
                    content_type,

                "declared_content_length":
                    declared_length,

                "bytes_read":
                    len(
                        body
                    ),

                "response_truncated_at_limit":
                    len(
                        body
                    )
                    >= MAX_RESPONSE_BYTES,

                "json_valid":
                    json_valid,

                "json_type":
                    json_type,

                "json_item_or_key_count":
                    json_items,

                "top_level_or_first_item_keys":
                    json_keys,

                "elapsed_seconds":
                    elapsed,

                "error":
                    "",

                "preview_file":
                    str(
                        preview_file.relative_to(
                            PROJECT
                        )
                    ),
            }

    except urllib.error.HTTPError as error:
        try:
            body = error.read(
                8192
            )

            error_preview = body.decode(
                "utf-8",
                errors="replace",
            )

        except Exception:
            error_preview = ""

        return {
            "probe_name":
                probe_name,

            "url":
                url,

            "http_status":
                int(
                    error.code
                ),

            "content_type":
                error.headers.get(
                    "Content-Type",
                    "",
                ),

            "declared_content_length":
                error.headers.get(
                    "Content-Length",
                    "",
                ),

            "bytes_read":
                len(
                    error_preview.encode(
                        "utf-8"
                    )
                ),

            "response_truncated_at_limit":
                False,

            "json_valid":
                False,

            "json_type":
                "",

            "json_item_or_key_count":
                0,

            "top_level_or_first_item_keys":
                "",

            "elapsed_seconds":
                time.time()
                - started,

            "error":
                (
                    f"HTTPError: {error}; "
                    f"{error_preview[:300]}"
                ),

            "preview_file":
                "",
        }

    except Exception as error:
        return {
            "probe_name":
                probe_name,

            "url":
                url,

            "http_status":
                "",

            "content_type":
                "",

            "declared_content_length":
                "",

            "bytes_read":
                0,

            "response_truncated_at_limit":
                False,

            "json_valid":
                False,

            "json_type":
                "",

            "json_item_or_key_count":
                0,

            "top_level_or_first_item_keys":
                "",

            "elapsed_seconds":
                time.time()
                - started,

            "error":
                (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),

            "preview_file":
                "",
        }


def build_probe_urls() -> list[tuple[str, str]]:
    folder_query_variants = [
        {
            "folderId":
                TARGET_FOLDER_ID,

            "page":
                0,

            "limit":
                10,
        },
        {
            "folder_id":
                TARGET_FOLDER_ID,

            "page":
                0,

            "limit":
                10,
        },
        {
            "folder":
                TARGET_FOLDER_ID,

            "page":
                0,

            "limit":
                10,
        },
    ]

    probes = [
        (
            "dataset_metadata",
            (
                f"{BASE_URL}/api/datasets/"
                f"{DATASET_ID}"
            ),
        ),
        (
            "folder_metadata",
            (
                f"{BASE_URL}/api/folders/"
                f"{TARGET_FOLDER_ID}"
            ),
        ),
        (
            "folder_files",
            (
                f"{BASE_URL}/api/folders/"
                f"{TARGET_FOLDER_ID}/files"
            ),
        ),
        (
            "folder_subfolders",
            (
                f"{BASE_URL}/api/folders/"
                f"{TARGET_FOLDER_ID}/folders"
            ),
        ),
        (
            "dataset_folders",
            (
                f"{BASE_URL}/api/datasets/"
                f"{DATASET_ID}/folders"
            ),
        ),
    ]

    for index, query in enumerate(
        folder_query_variants,
        start=1,
    ):
        encoded = urllib.parse.urlencode(
            query
        )

        probes.append(
            (
                (
                    "dataset_files_"
                    f"folder_query_{index}"
                ),
                (
                    f"{BASE_URL}/api/datasets/"
                    f"{DATASET_ID}/files?"
                    f"{encoded}"
                ),
            )
        )

        probes.append(
            (
                (
                    "dataset_list_"
                    f"folder_query_{index}"
                ),
                (
                    f"{BASE_URL}/api/datasets/"
                    f"{DATASET_ID}/list?"
                    f"{encoded}"
                ),
            )
        )

    return probes


def main() -> None:
    LOG_FILE.write_text(
        "",
        encoding="utf-8",
    )

    log(
        "===== Phase 6B2A started ====="
    )

    log(
        f"Dataset ID: {DATASET_ID}"
    )

    log(
        f"Target folder ID: {TARGET_FOLDER_ID}"
    )

    rows = []

    for probe_name, url in build_probe_urls():
        log()
        log(
            f"Probing: {probe_name}"
        )

        result = probe_url(
            probe_name,
            url,
        )

        rows.append(
            result
        )

        log(
            (
                f"HTTP={result['http_status']} "
                f"JSON={result['json_valid']} "
                f"type={result['json_type']} "
                f"items={result['json_item_or_key_count']} "
                f"bytes={result['bytes_read']}"
            )
        )

        if result[
            "error"
        ]:
            log(
                f"Error: {result['error']}"
            )

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
    )

    successful_json = results.loc[
        (
            pd.to_numeric(
                results[
                    "http_status"
                ],
                errors="coerce",
            )
            == 200
        )
        & results[
            "json_valid"
        ].eq(
            True
        )
    ].copy()

    target_folder_routes = successful_json.loc[
        successful_json[
            "probe_name"
        ].str.contains(
            (
                "folder_metadata|"
                "folder_files|"
                "folder_subfolders|"
                "folder_query"
            ),
            regex=True,
            na=False,
        )
    ]

    completion = pd.DataFrame(
        [
            {
                "dataset_id":
                    DATASET_ID,

                "target_folder_id":
                    TARGET_FOLDER_ID,

                "API_routes_tested":
                    len(
                        results
                    ),

                "HTTP_200_routes":
                    int(
                        (
                            pd.to_numeric(
                                results[
                                    "http_status"
                                ],
                                errors="coerce",
                            )
                            == 200
                        ).sum()
                    ),

                "valid_JSON_routes":
                    len(
                        successful_json
                    ),

                "target_folder_JSON_routes":
                    len(
                        target_folder_routes
                    ),

                "data_files_downloaded":
                    0,

                "Phase6B2A_status":
                    (
                        "completed_working_folder_route_found"
                        if len(
                            target_folder_routes
                        ) > 0
                        else
                        "completed_probe_requires_route_review"
                    ),
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
        "===== PROBE RESULTS ====="
    )

    display_columns = [
        "probe_name",
        "http_status",
        "content_type",
        "bytes_read",
        "response_truncated_at_limit",
        "json_valid",
        "json_type",
        "json_item_or_key_count",
        "error",
    ]

    log(
        results[
            display_columns
        ].to_string(
            index=False
        )
    )

    log()
    log(
        "===== PHASE 6B2A COMPLETION ====="
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
                "Phase 6B2A failed: "
                f"{type(error).__name__}: "
                f"{error}"
            )
        )

        raise
