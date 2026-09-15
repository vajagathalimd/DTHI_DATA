#!/usr/bin/env python3

from pathlib import Path
import re
import sys

import pandas as pd


PROJECT = Path(
    "."
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase5"
)

FIGURE_DIR = (
    PROJECT
    / "06_figures/main_figures/phase5"
)

OUTPUT_FILE = (
    TABLE_DIR
    / "phase5E1_final_completion_manifest.tsv"
)

FIGURE_MANIFEST_FILE = (
    TABLE_DIR
    / "phase5E1_figure_manifest.tsv"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase5/"
      "phase5E1_final_completion_manifest.log"
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


def scalar_text(value: object) -> str:
    if pd.isna(value):
        return ""

    return str(value)


def inspect_completion_file(
    path: Path,
) -> dict[str, object]:
    try:
        table = pd.read_csv(
            path,
            sep="\t",
            low_memory=False,
        )

    except Exception as error:
        return {
            "component":
                path.stem,

            "completion_file":
                str(
                    path.relative_to(
                        PROJECT
                    )
                ),

            "status":
                "read_error",

            "reported_figures":
                "",

            "key_summary":
                f"{type(error).__name__}: {error}",
        }

    if table.empty:
        return {
            "component":
                path.stem,

            "completion_file":
                str(
                    path.relative_to(
                        PROJECT
                    )
                ),

            "status":
                "empty",

            "reported_figures":
                "",

            "key_summary":
                "",
        }

    first_row = table.iloc[0]

    status_columns = [
        column
        for column in table.columns
        if column.lower().endswith(
            "status"
        )
    ]

    figure_columns = [
        column
        for column in table.columns
        if "figure" in column.lower()
    ]

    if status_columns:
        status = scalar_text(
            first_row[
                status_columns[0]
            ]
        )
    else:
        status = "status_column_not_found"

    reported_figures = "; ".join(
        (
            f"{column}="
            f"{scalar_text(first_row[column])}"
        )
        for column in figure_columns
    )

    excluded_columns = set(
        status_columns
        + figure_columns
    )

    summary_fields = []

    for column in table.columns:
        if column in excluded_columns:
            continue

        value = first_row[
            column
        ]

        if pd.isna(value):
            continue

        summary_fields.append(
            f"{column}={scalar_text(value)}"
        )

    return {
        "component":
            path.stem,

        "completion_file":
            str(
                path.relative_to(
                    PROJECT
                )
            ),

        "status":
            status,

        "reported_figures":
            reported_figures,

        "key_summary":
            "; ".join(
                summary_fields
            ),
    }


def extract_figure_number(
    filename: str,
) -> int | None:
    match = re.match(
        r"Figure(\d+)",
        filename,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return int(
        match.group(1)
    )


def build_figure_manifest() -> pd.DataFrame:
    rows = []

    for path in sorted(
        FIGURE_DIR.glob(
            "Figure*"
        )
    ):
        if not path.is_file():
            continue

        figure_number = extract_figure_number(
            path.name
        )

        if figure_number is None:
            continue

        if not (
            34
            <= figure_number
            <= 44
        ):
            continue

        rows.append(
            {
                "figure_number":
                    figure_number,

                "filename":
                    path.name,

                "file_format":
                    path.suffix.lower().lstrip(
                        "."
                    ),

                "size_bytes":
                    path.stat().st_size,

                "size_megabytes":
                    path.stat().st_size
                    / 1024
                    / 1024,

                "relative_path":
                    str(
                        path.relative_to(
                            PROJECT
                        )
                    ),

                "exists":
                    True,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "figure_number",
                "filename",
                "file_format",
                "size_bytes",
                "size_megabytes",
                "relative_path",
                "exists",
            ]
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "figure_number",
                "file_format",
                "filename",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def main() -> None:
    completion_files = sorted(
        TABLE_DIR.glob(
            "phase5*completion_summary.tsv"
        )
    )

    manifest_rows = [
        inspect_completion_file(
            path
        )
        for path in completion_files
    ]

    manifest = pd.DataFrame(
        manifest_rows,
        columns=[
            "component",
            "completion_file",
            "status",
            "reported_figures",
            "key_summary",
        ],
    )

    if not manifest.empty:
        manifest = manifest.sort_values(
            "component"
        ).reset_index(
            drop=True
        )

    manifest.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
    )

    figure_manifest = (
        build_figure_manifest()
    )

    figure_manifest.to_csv(
        FIGURE_MANIFEST_FILE,
        sep="\t",
        index=False,
    )

    figure_numbers_present = sorted(
        figure_manifest[
            "figure_number"
        ].unique()
    ) if not figure_manifest.empty else []

    expected_figure_numbers = list(
        range(
            34,
            45,
        )
    )

    missing_figure_numbers = [
        number
        for number in expected_figure_numbers
        if number
        not in figure_numbers_present
    ]

    completed_components = 0
    incomplete_components = 0

    if not manifest.empty:
        normalized_status = (
            manifest[
                "status"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        completed_components = int(
            normalized_status.eq(
                "completed"
            ).sum()
        )

        incomplete_components = int(
            (
                ~normalized_status.eq(
                    "completed"
                )
            ).sum()
        )

    log_lines = [
        "Phase 5E1 final completion audit",
        "",
        (
            "Completion-summary files: "
            f"{len(completion_files)}"
        ),
        (
            "Completed components: "
            f"{completed_components}"
        ),
        (
            "Non-completed or unclassified components: "
            f"{incomplete_components}"
        ),
        (
            "Figure files detected for Figures 34–44: "
            f"{len(figure_manifest)}"
        ),
        (
            "Figure numbers present: "
            + (
                ", ".join(
                    str(number)
                    for number
                    in figure_numbers_present
                )
                if figure_numbers_present
                else "none"
            )
        ),
        (
            "Missing figure numbers: "
            + (
                ", ".join(
                    str(number)
                    for number
                    in missing_figure_numbers
                )
                if missing_figure_numbers
                else "none"
            )
        ),
        "",
        (
            "Completion manifest: "
            + str(
                OUTPUT_FILE.relative_to(
                    PROJECT
                )
            )
        ),
        (
            "Figure manifest: "
            + str(
                FIGURE_MANIFEST_FILE.relative_to(
                    PROJECT
                )
            )
        ),
    ]

    LOG_FILE.write_text(
        "\n".join(
            log_lines
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "===== PHASE 5 COMPLETION MANIFEST ====="
    )

    if manifest.empty:
        print(
            "No Phase 5 completion summaries found."
        )
    else:
        print(
            manifest[
                [
                    "component",
                    "status",
                    "reported_figures",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "===== PHASE 5 FIGURES 34–44 ====="
    )

    if figure_manifest.empty:
        print(
            "No Figure 34–44 files found."
        )
    else:
        print(
            figure_manifest[
                [
                    "figure_number",
                    "filename",
                    "file_format",
                    "size_megabytes",
                ]
            ].to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.3f}"
                ),
            )
        )

    print()
    print(
        "===== AUDIT SUMMARY ====="
    )

    for line in log_lines[2:]:
        print(
            line
        )

    print()
    print(
        "===== Phase 5E1 completed ====="
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            f"Phase 5E1 failed: {error}",
            file=sys.stderr,
        )

        raise
