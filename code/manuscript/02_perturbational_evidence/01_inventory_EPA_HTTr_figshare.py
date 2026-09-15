#!/usr/bin/env python3

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

import pandas as pd


PROJECT = Path(
    "."
)

ARTICLE_ID = 21502758

ARTICLE_API = (
    f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
)

FILES_API = (
    f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/files"
)

METADATA_DIR = (
    PROJECT
    / "02_metadata/phase6"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase6"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase6/"
      "phase6B1_EPA_HTTr_figshare_inventory.log"
)

ARTICLE_JSON_FILE = (
    METADATA_DIR
    / "EPA_HTTr_figshare_article_21502758.json"
)

FILE_MANIFEST = (
    TABLE_DIR
    / "phase6B1_EPA_HTTr_figshare_file_manifest.tsv"
)

COMPLETION_FILE = (
    TABLE_DIR
    / "phase6B1_completion_summary.tsv"
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


def fetch_json(
    url: str,
    attempts: int = 4,
) -> object:
    headers = {
        "Accept":
            "application/json",

        "User-Agent":
            (
                "DTHI-Struct-Phase6/"
                "1.0 scientific-data-inventory"
            ),
    }

    last_error: Exception | None = None

    for attempt in range(
        1,
        attempts + 1,
    ):
        request = urllib.request.Request(
            url=url,
            headers=headers,
            method="GET",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=60,
            ) as response:
                status = response.status

                if status != 200:
                    raise RuntimeError(
                        f"HTTP status {status}"
                    )

                payload = response.read()

                return json.loads(
                    payload.decode(
                        "utf-8"
                    )
                )

        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            RuntimeError,
            json.JSONDecodeError,
        ) as error:
            last_error = error

            log(
                f"Attempt {attempt}/{attempts} failed "
                f"for {url}: {error}"
            )

            if attempt < attempts:
                time.sleep(
                    3 * attempt
                )

    raise RuntimeError(
        f"Unable to retrieve {url}: {last_error}"
    )


def text_value(
    value: object,
) -> str:
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
        )

    return str(
        value
    )


def main() -> None:
    LOG_FILE.write_text(
        "",
        encoding="utf-8",
    )

    log(
        "===== Phase 6B1 started ====="
    )

    log(
        f"Retrieving Figshare article {ARTICLE_ID}..."
    )

    article = fetch_json(
        ARTICLE_API
    )

    if not isinstance(
        article,
        dict,
    ):
        raise TypeError(
            "Figshare article response is not a JSON object."
        )

    article_files = article.get(
        "files",
        [],
    )

    if not article_files:
        log(
            "No files embedded in article metadata; "
            "querying the files endpoint."
        )

        article_files = fetch_json(
            FILES_API
        )

    if not isinstance(
        article_files,
        list,
    ):
        raise TypeError(
            "Figshare files response is not a JSON list."
        )

    ARTICLE_JSON_FILE.write_text(
        json.dumps(
            article,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_rows = []

    for file_record in article_files:
        if not isinstance(
            file_record,
            dict,
        ):
            continue

        size_bytes = pd.to_numeric(
            file_record.get(
                "size"
            ),
            errors="coerce",
        )

        if pd.isna(
            size_bytes
        ):
            size_bytes = 0

        size_bytes = int(
            size_bytes
        )

        filename = text_value(
            file_record.get(
                "name"
            )
        )

        suffixes = "".join(
            Path(
                filename
            ).suffixes
        )

        supplied_md5 = text_value(
            file_record.get(
                "supplied_md5"
            )
        )

        computed_md5 = text_value(
            file_record.get(
                "computed_md5"
            )
        )

        manifest_rows.append(
            {
                "article_id":
                    ARTICLE_ID,

                "article_version":
                    article.get(
                        "version",
                        "",
                    ),

                "file_id":
                    file_record.get(
                        "id",
                        "",
                    ),

                "filename":
                    filename,

                "file_suffixes":
                    suffixes,

                "size_bytes":
                    size_bytes,

                "size_megabytes":
                    size_bytes
                    / 1024
                    / 1024,

                "size_gigabytes":
                    size_bytes
                    / 1024
                    / 1024
                    / 1024,

                "is_link_only":
                    file_record.get(
                        "is_link_only",
                        False,
                    ),

                "download_url":
                    text_value(
                        file_record.get(
                            "download_url"
                        )
                    ),

                "supplied_md5":
                    supplied_md5,

                "computed_md5":
                    computed_md5,

                "md5_metadata_consistent":
                    bool(
                        supplied_md5
                        and computed_md5
                        and supplied_md5
                        == computed_md5
                    ),

                "download_status":
                    "not_downloaded_inventory_only",
            }
        )

    file_manifest = pd.DataFrame(
        manifest_rows
    )

    expected_columns = [
        "article_id",
        "article_version",
        "file_id",
        "filename",
        "file_suffixes",
        "size_bytes",
        "size_megabytes",
        "size_gigabytes",
        "is_link_only",
        "download_url",
        "supplied_md5",
        "computed_md5",
        "md5_metadata_consistent",
        "download_status",
    ]

    if file_manifest.empty:
        file_manifest = pd.DataFrame(
            columns=expected_columns
        )

    else:
        file_manifest = (
            file_manifest[
                expected_columns
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

    file_manifest.to_csv(
        FILE_MANIFEST,
        sep="\t",
        index=False,
    )

    total_size_bytes = int(
        file_manifest[
            "size_bytes"
        ].sum()
    ) if not file_manifest.empty else 0

    retrieval_time = datetime.now(
        timezone.utc
    ).isoformat()

    completion = pd.DataFrame(
        [
            {
                "figshare_article_id":
                    ARTICLE_ID,

                "article_title":
                    text_value(
                        article.get(
                            "title"
                        )
                    ),

                "article_version":
                    article.get(
                        "version",
                        "",
                    ),

                "article_doi":
                    text_value(
                        article.get(
                            "doi"
                        )
                    ),

                "article_published_date":
                    text_value(
                        article.get(
                            "published_date"
                        )
                    ),

                "article_modified_date":
                    text_value(
                        article.get(
                            "modified_date"
                        )
                    ),

                "retrieved_utc":
                    retrieval_time,

                "file_count":
                    len(
                        file_manifest
                    ),

                "total_size_bytes":
                    total_size_bytes,

                "total_size_megabytes":
                    total_size_bytes
                    / 1024
                    / 1024,

                "total_size_gigabytes":
                    total_size_bytes
                    / 1024
                    / 1024
                    / 1024,

                "files_downloaded":
                    0,

                "inventory_only":
                    True,

                "Phase6B1_status":
                    (
                        "completed"
                        if len(
                            file_manifest
                        ) > 0
                        else "failed_no_files_found"
                    ),
            }
        ]
    )

    completion.to_csv(
        COMPLETION_FILE,
        sep="\t",
        index=False,
    )

    log("")
    log(
        "===== ARTICLE SUMMARY ====="
    )

    log(
        completion.to_string(
            index=False
        )
    )

    log("")
    log(
        "===== FILE MANIFEST ====="
    )

    if file_manifest.empty:
        log(
            "No files found."
        )

    else:
        display_columns = [
            "file_id",
            "filename",
            "size_megabytes",
            "is_link_only",
            "md5_metadata_consistent",
        ]

        log(
            file_manifest[
                display_columns
            ].to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.3f}"
                ),
            )
        )

    log("")
    log(
        "===== Phase 6B1 completed ====="
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        log(
            f"Phase 6B1 failed: {error}"
        )

        raise
