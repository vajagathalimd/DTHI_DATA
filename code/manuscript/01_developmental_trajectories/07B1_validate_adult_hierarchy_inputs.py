#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import csv
import gzip
import sys

import pandas as pd


PROJECT = Path(
    "."
)

EXPRESSION_FILE = (
    PROJECT
    / "03_processed_data/transcriptomics/AHBA/"
      "ahba_5donor_microarray_strict_cortical_expression.tsv.gz"
)

METADATA_FILE = (
    PROJECT
    / "03_processed_data/transcriptomics/AHBA/"
      "ahba_5donor_microarray_strict_cortical_sample_metadata.tsv"
)

ASSIGNMENT_FILE = (
    PROJECT
    / "03_processed_data/spatial_hierarchy/phase3C/"
      "AHBA_sample_to_Schaefer100_assignment.tsv"
)

HIERARCHY_FILE = (
    PROJECT
    / "03_processed_data/spatial_hierarchy/phase3B/"
      "AHBA_regions_with_anatomical_hierarchy_proxy.tsv"
)

EXTERNAL_MAP_FILE = (
    PROJECT
    / "03_processed_data/spatial_hierarchy/phase3C/"
      "Schaefer100_external_maps_wide.tsv"
)

MATURATION_FILE = (
    PROJECT
    / "03_processed_data/developmental_trajectory/phase5D/"
      "enrichment/phase5D4_maturation_high_increasing_genes.tsv"
)

FETAL_FILE = (
    PROJECT
    / "03_processed_data/developmental_trajectory/phase5D/"
      "enrichment/phase5D4_fetal_high_decreasing_genes.tsv"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase5"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase5/"
      "phase5D7B1_adult_input_validation.log"
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


def normalize_gene(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().upper()

    if text in {
        "",
        "NA",
        "NAN",
        "NONE",
        "<NA>",
    }:
        return ""

    return text


def normalize_id(value: object) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_parcel_id(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if not text:
        return ""

    try:
        number = float(text)

        if number.is_integer():
            return str(int(number))

        return str(number)

    except ValueError:
        return text


def parse_boolean(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "t",
                "1",
                "yes",
                "y",
            }
        )
    )


def require_columns(
    dataframe: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{label} is missing columns: {missing}. "
            f"Available columns: {list(dataframe.columns)}"
        )


def read_expression_header(
    path: Path,
) -> tuple[str, list[str]]:
    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        header = handle.readline().rstrip("\n\r")

    columns = header.split("\t")

    if len(columns) < 2:
        raise ValueError(
            "AHBA expression matrix has fewer than two columns."
        )

    return columns[0], columns[1:]


def read_expression_genes(
    path: Path,
) -> set[str]:
    genes: set[str] = set()

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as handle:
        reader = csv.reader(
            handle,
            delimiter="\t",
        )

        next(reader, None)

        for row in reader:
            if not row:
                continue

            gene = normalize_gene(
                row[0]
            )

            if gene:
                genes.add(gene)

    return genes


def read_program_genes(
    path: Path,
) -> set[str]:
    dataframe = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
    )

    candidates = [
        "gene_symbol",
        "gene",
        "symbol",
    ]

    gene_column = None

    for candidate in candidates:
        if candidate in dataframe.columns:
            gene_column = candidate
            break

    if gene_column is None:
        gene_column = dataframe.columns[0]

    return {
        gene
        for gene in (
            normalize_gene(value)
            for value in dataframe[gene_column]
        )
        if gene
    }


def set_summary(
    label: str,
    reference: set[str],
    comparison: set[str],
) -> dict[str, object]:
    shared = reference & comparison

    return {
        "comparison": label,
        "reference_count": len(reference),
        "comparison_count": len(comparison),
        "shared_count": len(shared),
        "reference_only_count": len(
            reference - comparison
        ),
        "comparison_only_count": len(
            comparison - reference
        ),
        "reference_coverage_fraction": (
            len(shared) / len(reference)
            if reference
            else float("nan")
        ),
        "comparison_coverage_fraction": (
            len(shared) / len(comparison)
            if comparison
            else float("nan")
        ),
    }


def main() -> None:
    required_files = [
        EXPRESSION_FILE,
        METADATA_FILE,
        ASSIGNMENT_FILE,
        HIERARCHY_FILE,
        EXTERNAL_MAP_FILE,
        MATURATION_FILE,
        FETAL_FILE,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(
                str(path)
                for path in missing_files
            )
        )

    print(
        "===== Phase 5D7B1 adult-input validation started ====="
    )

    gene_column, expression_samples = read_expression_header(
        EXPRESSION_FILE
    )

    expression_sample_set = {
        normalize_id(value)
        for value in expression_samples
        if normalize_id(value)
    }

    expression_genes = read_expression_genes(
        EXPRESSION_FILE
    )

    metadata = pd.read_csv(
        METADATA_FILE,
        sep="\t",
        dtype="string",
        low_memory=False,
    )

    assignment = pd.read_csv(
        ASSIGNMENT_FILE,
        sep="\t",
        dtype="string",
        low_memory=False,
    )

    hierarchy = pd.read_csv(
        HIERARCHY_FILE,
        sep="\t",
        dtype="string",
        low_memory=False,
    )

    external = pd.read_csv(
        EXTERNAL_MAP_FILE,
        sep="\t",
        dtype="string",
        low_memory=False,
    )

    require_columns(
        metadata,
        [
            "sample_id",
            "donor_id",
            "structure_acronym",
            "structure_name",
        ],
        "AHBA strict cortical metadata",
    )

    require_columns(
        assignment,
        [
            "sample_id",
            "donor_id",
            "parcel_id",
            "usable_assignment",
        ],
        "AHBA Schaefer100 assignment",
    )

    require_columns(
        hierarchy,
        [
            "structure_acronym",
            "hierarchy_tier",
        ],
        "AHBA anatomical hierarchy table",
    )

    require_columns(
        external,
        [
            "parcel_id",
        ],
        "Schaefer100 external maps table",
    )

    metadata["sample_id"] = (
        metadata["sample_id"]
        .map(normalize_id)
    )

    metadata["donor_id"] = (
        metadata["donor_id"]
        .map(normalize_id)
    )

    metadata["structure_acronym"] = (
        metadata["structure_acronym"]
        .map(normalize_id)
    )

    assignment["sample_id"] = (
        assignment["sample_id"]
        .map(normalize_id)
    )

    assignment["donor_id"] = (
        assignment["donor_id"]
        .map(normalize_id)
    )

    usable_assignment = assignment.loc[
        parse_boolean(
            assignment["usable_assignment"]
        )
    ].copy()

    metadata_sample_set = set(
        metadata["sample_id"]
    ) - {""}

    assignment_sample_set = set(
        assignment["sample_id"]
    ) - {""}

    usable_assignment_sample_set = set(
        usable_assignment["sample_id"]
    ) - {""}

    sample_audit = pd.DataFrame(
        [
            set_summary(
                "expression_vs_strict_metadata",
                expression_sample_set,
                metadata_sample_set,
            ),
            set_summary(
                "expression_vs_all_Schaefer_assignments",
                expression_sample_set,
                assignment_sample_set,
            ),
            set_summary(
                "expression_vs_usable_Schaefer_assignments",
                expression_sample_set,
                usable_assignment_sample_set,
            ),
        ]
    )

    sample_audit.to_csv(
        TABLE_DIR
        / "phase5D7B1_sample_identifier_audit.tsv",
        sep="\t",
        index=False,
    )

    metadata_structures = set(
        metadata["structure_acronym"]
    ) - {""}

    hierarchy_structures = set(
        hierarchy["structure_acronym"]
        .map(normalize_id)
    ) - {""}

    mapped_structures = (
        metadata_structures
        & hierarchy_structures
    )

    hierarchy_sample_count = metadata[
        "structure_acronym"
    ].isin(
        hierarchy_structures
    ).sum()

    hierarchy_audit = pd.DataFrame(
        [
            {
                "strict_metadata_samples":
                    len(metadata),

                "strict_metadata_structures":
                    len(metadata_structures),

                "hierarchy_mapping_structures":
                    len(hierarchy_structures),

                "shared_structures":
                    len(mapped_structures),

                "samples_with_hierarchy_mapping":
                    int(hierarchy_sample_count),

                "sample_hierarchy_coverage_fraction":
                    (
                        hierarchy_sample_count
                        / len(metadata)
                        if len(metadata)
                        else float("nan")
                    ),

                "hierarchy_tiers":
                    hierarchy[
                        "hierarchy_tier"
                    ].nunique(
                        dropna=True
                    ),
            }
        ]
    )

    hierarchy_audit.to_csv(
        TABLE_DIR
        / "phase5D7B1_hierarchy_coverage_audit.tsv",
        sep="\t",
        index=False,
    )

    assignment_parcels = set(
        usable_assignment["parcel_id"]
        .map(normalize_parcel_id)
    ) - {""}

    external_parcels = set(
        external["parcel_id"]
        .map(normalize_parcel_id)
    ) - {""}

    parcel_audit = pd.DataFrame(
        [
            {
                "usable_assigned_samples":
                    len(
                        usable_assignment_sample_set
                    ),

                "usable_assignment_parcels":
                    len(
                        assignment_parcels
                    ),

                "external_map_parcels":
                    len(
                        external_parcels
                    ),

                "shared_parcels":
                    len(
                        assignment_parcels
                        & external_parcels
                    ),

                "assignment_parcel_coverage_fraction":
                    (
                        len(
                            assignment_parcels
                            & external_parcels
                        )
                        / len(
                            assignment_parcels
                        )
                        if assignment_parcels
                        else float("nan")
                    ),

                "external_map_columns":
                    "|".join(
                        external.columns
                    ),
            }
        ]
    )

    parcel_audit.to_csv(
        TABLE_DIR
        / "phase5D7B1_parcel_external_map_audit.tsv",
        sep="\t",
        index=False,
    )

    maturation_genes = read_program_genes(
        MATURATION_FILE
    )

    fetal_genes = read_program_genes(
        FETAL_FILE
    )

    gene_mapping = pd.DataFrame(
        [
            {
                "developmental_program":
                    "maturation_high_increasing",

                "program_genes":
                    len(maturation_genes),

                "AHBA_expression_genes":
                    len(expression_genes),

                "mapped_genes":
                    len(
                        maturation_genes
                        & expression_genes
                    ),

                "mapping_fraction":
                    (
                        len(
                            maturation_genes
                            & expression_genes
                        )
                        / len(
                            maturation_genes
                        )
                        if maturation_genes
                        else float("nan")
                    ),

                "missing_genes":
                    "/".join(
                        sorted(
                            maturation_genes
                            - expression_genes
                        )
                    ),
            },
            {
                "developmental_program":
                    "fetal_high_decreasing",

                "program_genes":
                    len(fetal_genes),

                "AHBA_expression_genes":
                    len(expression_genes),

                "mapped_genes":
                    len(
                        fetal_genes
                        & expression_genes
                    ),

                "mapping_fraction":
                    (
                        len(
                            fetal_genes
                            & expression_genes
                        )
                        / len(
                            fetal_genes
                        )
                        if fetal_genes
                        else float("nan")
                    ),

                "missing_genes":
                    "/".join(
                        sorted(
                            fetal_genes
                            - expression_genes
                        )
                    ),
            },
        ]
    )

    gene_mapping.to_csv(
        TABLE_DIR
        / "phase5D7B1_program_gene_AHBA_mapping.tsv",
        sep="\t",
        index=False,
    )

    input_summary = pd.DataFrame(
        [
            {
                "expression_gene_column":
                    gene_column,

                "expression_genes":
                    len(expression_genes),

                "expression_samples":
                    len(expression_sample_set),

                "strict_metadata_samples":
                    len(metadata_sample_set),

                "strict_metadata_donors":
                    metadata["donor_id"].nunique(),

                "strict_metadata_structures":
                    metadata[
                        "structure_acronym"
                    ].nunique(),

                "usable_Schaefer_samples":
                    len(
                        usable_assignment_sample_set
                    ),

                "usable_Schaefer_parcels":
                    len(
                        assignment_parcels
                    ),

                "external_map_parcels":
                    len(
                        external_parcels
                    ),

                "Phase5D7B1_status":
                    "completed",
            }
        ]
    )

    input_summary.to_csv(
        TABLE_DIR
        / "phase5D7B1_completion_summary.tsv",
        sep="\t",
        index=False,
    )

    with LOG_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "Phase 5D7B1 adult validation input audit\n\n"
        )

        handle.write(
            "Input summary\n"
        )

        handle.write(
            input_summary.to_string(
                index=False
            )
        )

        handle.write(
            "\n\nSample identifier audit\n"
        )

        handle.write(
            sample_audit.to_string(
                index=False
            )
        )

        handle.write(
            "\n\nHierarchy coverage\n"
        )

        handle.write(
            hierarchy_audit.to_string(
                index=False
            )
        )

        handle.write(
            "\n\nParcel/external-map coverage\n"
        )

        handle.write(
            parcel_audit.to_string(
                index=False
            )
        )

        handle.write(
            "\n\nProgram-gene AHBA mapping\n"
        )

        handle.write(
            gene_mapping.to_string(
                index=False
            )
        )

        handle.write("\n")

    print()
    print(
        "===== INPUT SUMMARY ====="
    )

    print(
        input_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== SAMPLE IDENTIFIER AUDIT ====="
    )

    print(
        sample_audit.to_string(
            index=False
        )
    )

    print()
    print(
        "===== HIERARCHY COVERAGE ====="
    )

    print(
        hierarchy_audit.to_string(
            index=False
        )
    )

    print()
    print(
        "===== PARCEL AND EXTERNAL-MAP COVERAGE ====="
    )

    print(
        parcel_audit.to_string(
            index=False
        )
    )

    print()
    print(
        "===== PROGRAM-GENE AHBA MAPPING ====="
    )

    print(
        gene_mapping[
            [
                "developmental_program",
                "program_genes",
                "AHBA_expression_genes",
                "mapped_genes",
                "mapping_fraction",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "===== Phase 5D7B1 completed ====="
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"Phase 5D7B1 failed: {error}",
            file=sys.stderr,
        )
        raise
