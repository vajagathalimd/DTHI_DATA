#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import csv
import gzip
import re
import sys


PROJECT = Path(
    "."
)

SEARCH_ROOTS = [
    PROJECT / "02_metadata",
    PROJECT / "03_processed_data",
    PROJECT / "07_tables",
]

OUTPUT_FILE = (
    PROJECT
    / "07_tables/main_tables/phase5/"
      "phase5D7A_adult_hierarchy_input_inventory.tsv"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase5/"
      "phase5D7A_adult_hierarchy_inventory.log"
)

VALID_SUFFIXES = {
    ".tsv",
    ".csv",
    ".txt",
    ".gz",
    ".xlsx",
    ".rds",
    ".pkl",
    ".parquet",
}

NAME_PATTERNS = {
    "AHBA_expression": re.compile(
        r"ahba.*expression|expression.*ahba|"
        r"cortical_expression|gene_symbol_expression",
        re.IGNORECASE,
    ),
    "hierarchy_mapping": re.compile(
        r"hierarchy|tier|transmodal|sensorimotor|"
        r"unimodal|heteromodal|paralimbic",
        re.IGNORECASE,
    ),
    "parcel_mapping": re.compile(
        r"schaefer|parcel|parcellation|region_map|"
        r"atlas_mapping",
        re.IGNORECASE,
    ),
    "external_map": re.compile(
        r"neuromap|myelin|thickness|sydnor|gradient|"
        r"external.*map|spatial.*map",
        re.IGNORECASE,
    ),
    "phase3_results": re.compile(
        r"phase3|hierarchy.*result|module.*hierarchy|"
        r"spatial.*correlation",
        re.IGNORECASE,
    ),
    "sample_metadata": re.compile(
        r"sample.*metadata|metadata.*sample|"
        r"strict.*cortical|probable.*cortical",
        re.IGNORECASE,
    ),
}

HEADER_PATTERNS = {
    "sample_metadata": re.compile(
        r"sample_id|donor_id|structure_acronym|"
        r"structure_name|mni_x|mni_y|mni_z",
        re.IGNORECASE,
    ),
    "hierarchy_mapping": re.compile(
        r"hierarchy_tier|hierarchy|tier_label|"
        r"cortical_hierarchy",
        re.IGNORECASE,
    ),
    "parcel_mapping": re.compile(
        r"parcel|schaefer|hemisphere|roi|region_id",
        re.IGNORECASE,
    ),
    "external_map": re.compile(
        r"myelin|thickness|gradient|sydnor|neuromap",
        re.IGNORECASE,
    ),
    "expression_matrix": re.compile(
        r"gene_symbol|gene_id|sample_id",
        re.IGNORECASE,
    ),
}


def read_header(path: Path) -> str:
    try:
        if path.suffix == ".gz":
            with gzip.open(
                path,
                "rt",
                encoding="utf-8",
                errors="replace",
            ) as handle:
                return handle.readline().strip()

        if path.suffix in {
            ".tsv",
            ".csv",
            ".txt",
        }:
            with path.open(
                "r",
                encoding="utf-8",
                errors="replace",
            ) as handle:
                return handle.readline().strip()

    except Exception as error:
        return f"READ_ERROR: {error}"

    return ""


def classify_candidate(
    path: Path,
    header: str,
) -> tuple[list[str], int]:
    categories: list[str] = []
    priority = 0

    text = f"{path.name} {path.parent}".lower()

    for category, pattern in NAME_PATTERNS.items():
        if pattern.search(text):
            categories.append(category)

            if category == "AHBA_expression":
                priority += 10
            elif category == "hierarchy_mapping":
                priority += 9
            elif category == "sample_metadata":
                priority += 8
            elif category == "parcel_mapping":
                priority += 7
            elif category == "external_map":
                priority += 7
            elif category == "phase3_results":
                priority += 6

    for category, pattern in HEADER_PATTERNS.items():
        if pattern.search(header):
            categories.append(category)

            if category == "hierarchy_mapping":
                priority += 8
            elif category == "sample_metadata":
                priority += 7
            elif category == "parcel_mapping":
                priority += 6
            elif category == "external_map":
                priority += 6
            elif category == "expression_matrix":
                priority += 4

    if "phase3" in str(path).lower():
        priority += 4

    if "ahba" in str(path).lower():
        priority += 4

    return sorted(set(categories)), priority


def main() -> None:
    records: list[dict[str, object]] = []

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            suffixes = set(path.suffixes)

            if not suffixes.intersection(
                VALID_SUFFIXES
            ):
                continue

            header = read_header(path)

            categories, priority = classify_candidate(
                path,
                header,
            )

            if not categories:
                continue

            records.append(
                {
                    "priority_score": priority,
                    "candidate_categories": "|".join(
                        categories
                    ),
                    "filename": path.name,
                    "size_bytes": path.stat().st_size,
                    "relative_path": str(
                        path.relative_to(PROJECT)
                    ),
                    "first_line": header[:1000],
                }
            )

    records.sort(
        key=lambda row: (
            -int(row["priority_score"]),
            str(row["relative_path"]),
        )
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "priority_score",
                "candidate_categories",
                "filename",
                "size_bytes",
                "relative_path",
                "first_line",
            ],
            delimiter="\t",
        )

        writer.writeheader()
        writer.writerows(records)

    with LOG_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "Phase 5D7A adult hierarchy input inventory\n"
        )
        handle.write(
            f"Candidate files: {len(records)}\n"
        )
        handle.write(
            f"Inventory file: "
            f"{OUTPUT_FILE.relative_to(PROJECT)}\n"
        )

    print(
        "===== Phase 5D7A inventory completed ====="
    )

    print(
        f"Candidate files: {len(records)}"
    )

    print()

    for row in records[:80]:
        print(
            f"{int(row['priority_score']):2d}  "
            f"{int(row['size_bytes']):12d}  "
            f"{row['candidate_categories']:<55}  "
            f"{row['relative_path']}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"Phase 5D7A failed: {error}",
            file=sys.stderr,
        )
        raise
