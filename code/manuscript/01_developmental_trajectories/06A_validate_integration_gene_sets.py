#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT = Path(
    "."
)

FILES = {
    "brainspan_seed": (
        PROJECT
        / "03_processed_data/transcriptomics/BrainSpan/"
          "DTHI_seed_modules/dthi_seed_gene_matching.tsv"
    ),
    "ahba_seed": (
        PROJECT
        / "03_processed_data/transcriptomics/AHBA/"
          "DTHI_seed_modules/ahba_dthi_seed_gene_matching.tsv"
    ),
    "brainspan_celltype": (
        PROJECT
        / "03_processed_data/single_cell_marker_validation/"
          "phase2A/brainspan_celltype_marker_matching.tsv"
    ),
    "ahba_celltype": (
        PROJECT
        / "03_processed_data/single_cell_marker_validation/"
          "phase2A/ahba_celltype_marker_matching.tsv"
    ),
}

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase5"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase5/"
      "phase5D6A_selected_gene_set_validation.log"
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


def normalize_gene(values: pd.Series) -> pd.Series:
    result = (
        values.astype("string")
        .str.strip()
        .str.upper()
    )

    return result.mask(
        result.isin(
            [
                "",
                "NA",
                "NAN",
                "NONE",
                "<NA>",
            ]
        )
    )


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    label: str,
) -> str:
    normalized = {
        column.lower().strip(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    raise ValueError(
        f"Could not identify {label}. "
        f"Available columns: {list(dataframe.columns)}"
    )


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    dataframe = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        low_memory=False,
    )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    return dataframe


def prepare_seed_table(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str]:
    module_column = find_column(
        dataframe,
        [
            "module",
            "module_name",
            "dthi_module",
        ],
        "DTHI module column",
    )

    gene_column = find_column(
        dataframe,
        [
            "seed_gene",
            "gene_symbol",
            "gene",
            "marker_gene",
        ],
        "seed-gene column",
    )

    result = dataframe[
        [
            module_column,
            gene_column,
        ]
    ].copy()

    result.columns = [
        "gene_set",
        "gene_symbol",
    ]

    result["gene_set"] = (
        result["gene_set"]
        .astype("string")
        .str.strip()
    )

    result["gene_symbol"] = normalize_gene(
        result["gene_symbol"]
    )

    result = (
        result.dropna(
            subset=[
                "gene_set",
                "gene_symbol",
            ]
        )
        .drop_duplicates()
        .sort_values(
            [
                "gene_set",
                "gene_symbol",
            ]
        )
        .reset_index(drop=True)
    )

    return result, module_column, gene_column


def prepare_celltype_table(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str]:
    celltype_column = find_column(
        dataframe,
        [
            "celltype_module",
            "cell_type_module",
            "celltype",
            "cell_type",
            "marker_set",
        ],
        "cell-type marker-set column",
    )

    gene_column = find_column(
        dataframe,
        [
            "marker_gene",
            "gene_symbol",
            "gene",
            "seed_gene",
        ],
        "marker-gene column",
    )

    result = dataframe[
        [
            celltype_column,
            gene_column,
        ]
    ].copy()

    result.columns = [
        "gene_set",
        "gene_symbol",
    ]

    result["gene_set"] = (
        result["gene_set"]
        .astype("string")
        .str.strip()
    )

    result["gene_symbol"] = normalize_gene(
        result["gene_symbol"]
    )

    result = (
        result.dropna(
            subset=[
                "gene_set",
                "gene_symbol",
            ]
        )
        .drop_duplicates()
        .sort_values(
            [
                "gene_set",
                "gene_symbol",
            ]
        )
        .reset_index(drop=True)
    )

    return result, celltype_column, gene_column


def set_pairs(dataframe: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        dataframe[
            [
                "gene_set",
                "gene_symbol",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    )


def write_difference(
    left: set[tuple[str, str]],
    right: set[tuple[str, str]],
    output_path: Path,
    difference_label: str,
) -> None:
    difference = sorted(
        left - right
    )

    dataframe = pd.DataFrame(
        difference,
        columns=[
            "gene_set",
            "gene_symbol",
        ],
    )

    dataframe.insert(
        0,
        "difference_type",
        difference_label,
    )

    dataframe.to_csv(
        output_path,
        sep="\t",
        index=False,
    )


def summarize_gene_sets(
    dataframe: pd.DataFrame,
    source: str,
    gene_set_type: str,
) -> pd.DataFrame:
    summary = (
        dataframe.groupby(
            "gene_set",
            observed=True,
        )
        .agg(
            unique_genes=(
                "gene_symbol",
                "nunique",
            )
        )
        .reset_index()
    )

    summary.insert(
        0,
        "gene_set_type",
        gene_set_type,
    )

    summary.insert(
        0,
        "source",
        source,
    )

    return summary


def main() -> None:
    print(
        "===== Phase 5D6A selected-input validation started ====="
    )

    raw_tables: dict[str, pd.DataFrame] = {}
    schema_rows: list[dict[str, object]] = []

    for label, path in FILES.items():
        dataframe = read_table(path)
        raw_tables[label] = dataframe

        schema_rows.append(
            {
                "input_label": label,
                "relative_path": str(
                    path.relative_to(PROJECT)
                ),
                "rows": len(dataframe),
                "columns": len(dataframe.columns),
                "column_names": "|".join(
                    dataframe.columns
                ),
            }
        )

        print()
        print(f"===== {label} =====")
        print(
            path.relative_to(PROJECT)
        )
        print(
            f"Shape: {dataframe.shape[0]} rows × "
            f"{dataframe.shape[1]} columns"
        )
        print(
            "Columns:",
            ", ".join(dataframe.columns),
        )

    (
        brainspan_seed,
        brainspan_seed_set_column,
        brainspan_seed_gene_column,
    ) = prepare_seed_table(
        raw_tables["brainspan_seed"]
    )

    (
        ahba_seed,
        ahba_seed_set_column,
        ahba_seed_gene_column,
    ) = prepare_seed_table(
        raw_tables["ahba_seed"]
    )

    (
        brainspan_celltype,
        brainspan_celltype_set_column,
        brainspan_celltype_gene_column,
    ) = prepare_celltype_table(
        raw_tables["brainspan_celltype"]
    )

    (
        ahba_celltype,
        ahba_celltype_set_column,
        ahba_celltype_gene_column,
    ) = prepare_celltype_table(
        raw_tables["ahba_celltype"]
    )

    selected_columns = {
        "brainspan_seed": (
            brainspan_seed_set_column,
            brainspan_seed_gene_column,
        ),
        "ahba_seed": (
            ahba_seed_set_column,
            ahba_seed_gene_column,
        ),
        "brainspan_celltype": (
            brainspan_celltype_set_column,
            brainspan_celltype_gene_column,
        ),
        "ahba_celltype": (
            ahba_celltype_set_column,
            ahba_celltype_gene_column,
        ),
    }

    for row in schema_rows:
        set_column, gene_column = selected_columns[
            str(row["input_label"])
        ]

        row["selected_gene_set_column"] = set_column
        row["selected_gene_column"] = gene_column

    schema_table = pd.DataFrame(
        schema_rows
    )

    schema_table.to_csv(
        TABLE_DIR
        / "phase5D6A_selected_input_schema_audit.tsv",
        sep="\t",
        index=False,
    )

    seed_summary = pd.concat(
        [
            summarize_gene_sets(
                brainspan_seed,
                "BrainSpan",
                "DTHI_seed_module",
            ),
            summarize_gene_sets(
                ahba_seed,
                "AHBA",
                "DTHI_seed_module",
            ),
        ],
        ignore_index=True,
    )

    celltype_summary = pd.concat(
        [
            summarize_gene_sets(
                brainspan_celltype,
                "BrainSpan",
                "celltype_marker_set",
            ),
            summarize_gene_sets(
                ahba_celltype,
                "AHBA",
                "celltype_marker_set",
            ),
        ],
        ignore_index=True,
    )

    seed_summary.to_csv(
        TABLE_DIR
        / "phase5D6A_seed_gene_set_counts.tsv",
        sep="\t",
        index=False,
    )

    celltype_summary.to_csv(
        TABLE_DIR
        / "phase5D6A_celltype_marker_set_counts.tsv",
        sep="\t",
        index=False,
    )

    brainspan_seed_pairs = set_pairs(
        brainspan_seed
    )

    ahba_seed_pairs = set_pairs(
        ahba_seed
    )

    brainspan_celltype_pairs = set_pairs(
        brainspan_celltype
    )

    ahba_celltype_pairs = set_pairs(
        ahba_celltype
    )

    consistency = pd.DataFrame(
        [
            {
                "gene_set_type": "DTHI_seed_module",
                "brainspan_pairs": len(
                    brainspan_seed_pairs
                ),
                "ahba_pairs": len(
                    ahba_seed_pairs
                ),
                "shared_pairs": len(
                    brainspan_seed_pairs
                    & ahba_seed_pairs
                ),
                "brainspan_only_pairs": len(
                    brainspan_seed_pairs
                    - ahba_seed_pairs
                ),
                "ahba_only_pairs": len(
                    ahba_seed_pairs
                    - brainspan_seed_pairs
                ),
                "definitions_identical": (
                    brainspan_seed_pairs
                    == ahba_seed_pairs
                ),
            },
            {
                "gene_set_type": "celltype_marker_set",
                "brainspan_pairs": len(
                    brainspan_celltype_pairs
                ),
                "ahba_pairs": len(
                    ahba_celltype_pairs
                ),
                "shared_pairs": len(
                    brainspan_celltype_pairs
                    & ahba_celltype_pairs
                ),
                "brainspan_only_pairs": len(
                    brainspan_celltype_pairs
                    - ahba_celltype_pairs
                ),
                "ahba_only_pairs": len(
                    ahba_celltype_pairs
                    - brainspan_celltype_pairs
                ),
                "definitions_identical": (
                    brainspan_celltype_pairs
                    == ahba_celltype_pairs
                ),
            },
        ]
    )

    consistency.to_csv(
        TABLE_DIR
        / "phase5D6A_cross_dataset_gene_set_consistency.tsv",
        sep="\t",
        index=False,
    )

    write_difference(
        brainspan_seed_pairs,
        ahba_seed_pairs,
        TABLE_DIR
        / "phase5D6A_seed_pairs_brainspan_only.tsv",
        "BrainSpan_only",
    )

    write_difference(
        ahba_seed_pairs,
        brainspan_seed_pairs,
        TABLE_DIR
        / "phase5D6A_seed_pairs_ahba_only.tsv",
        "AHBA_only",
    )

    write_difference(
        brainspan_celltype_pairs,
        ahba_celltype_pairs,
        TABLE_DIR
        / "phase5D6A_celltype_pairs_brainspan_only.tsv",
        "BrainSpan_only",
    )

    write_difference(
        ahba_celltype_pairs,
        brainspan_celltype_pairs,
        TABLE_DIR
        / "phase5D6A_celltype_pairs_ahba_only.tsv",
        "AHBA_only",
    )

    with LOG_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "Phase 5D6A selected integration gene-set validation\n\n"
        )

        handle.write(
            consistency.to_string(
                index=False
            )
        )

        handle.write("\n")

    print()
    print(
        "===== DTHI SEED-MODULE COUNTS ====="
    )

    print(
        seed_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== CELL-TYPE MARKER COUNTS ====="
    )

    print(
        celltype_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== CROSS-DATASET CONSISTENCY ====="
    )

    print(
        consistency.to_string(
            index=False
        )
    )

    print()
    print(
        "===== Phase 5D6A completed ====="
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"Phase 5D6A failed: {error}",
            file=sys.stderr,
        )
        raise
