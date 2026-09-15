#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import gzip
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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

HIERARCHY_FILE = (
    PROJECT
    / "03_processed_data/spatial_hierarchy/phase3B/"
      "AHBA_regions_with_anatomical_hierarchy_proxy.tsv"
)

ASSIGNMENT_FILE = (
    PROJECT
    / "03_processed_data/spatial_hierarchy/phase3C/"
      "AHBA_sample_to_Schaefer100_assignment.tsv"
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

PROCESSED_DIR = (
    PROJECT
    / "03_processed_data/developmental_trajectory/phase5D7"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase5"
)

FIGURE_DIR = (
    PROJECT
    / "06_figures/main_figures/phase5"
)

SOURCE_DIR = (
    PROJECT
    / "06_figures/source_data/phase5"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase5/"
      "phase5D7B2_AHBA_program_hierarchy.log"
)

for directory in [
    PROCESSED_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    SOURCE_DIR,
    LOG_FILE.parent,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


PROGRAM_FILES = {
    "maturation_high_increasing": MATURATION_FILE,
    "fetal_high_decreasing": FETAL_FILE,
}

OUTCOMES = {
    "maturation_high_score_z":
        "Maturation-high program",

    "fetal_high_score_z":
        "Fetal-high program",

    "maturation_minus_fetal_balance":
        "Maturation minus fetal balance",
}

CHUNK_SIZE = 256
N_PERMUTATIONS = 10000
RANDOM_SEED = 20260720


def log(message: str = "") -> None:
    print(message, flush=True)

    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            message + "\n"
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
            [
                "true",
                "t",
                "1",
                "yes",
                "y",
            ]
        )
    )


def read_program_genes(path: Path) -> set[str]:
    table = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        low_memory=False,
    )

    candidate_columns = [
        "gene_symbol",
        "gene",
        "symbol",
    ]

    gene_column = next(
        (
            column
            for column in candidate_columns
            if column in table.columns
        ),
        table.columns[0],
    )

    genes = {
        normalize_gene(value)
        for value in table[gene_column]
    }

    return {
        gene
        for gene in genes
        if gene
    }


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=float,
    )

    mean = np.nanmean(values)

    standard_deviation = np.nanstd(
        values,
        ddof=1,
    )

    if (
        not np.isfinite(standard_deviation)
        or standard_deviation <= 0
    ):
        return np.full(
            values.shape,
            np.nan,
            dtype=float,
        )

    return (
        values - mean
    ) / standard_deviation


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=float,
    )

    output = np.full(
        values.shape,
        np.nan,
        dtype=float,
    )

    finite = np.isfinite(values)

    if not finite.any():
        return output

    valid = values[finite]

    ordering = np.argsort(
        valid
    )

    ordered = valid[ordering]

    number_of_tests = len(
        ordered
    )

    adjusted = (
        ordered
        * number_of_tests
        / np.arange(
            1,
            number_of_tests + 1,
        )
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.minimum(
        adjusted,
        1.0,
    )

    restored = np.empty_like(
        adjusted
    )

    restored[ordering] = adjusted

    output[finite] = restored

    return output


def combined_fisher_r(
    correlations: list[float],
) -> float:
    valid = np.asarray(
        [
            value
            for value in correlations
            if np.isfinite(value)
        ],
        dtype=float,
    )

    if len(valid) == 0:
        return float("nan")

    valid = np.clip(
        valid,
        -0.999999,
        0.999999,
    )

    return float(
        np.tanh(
            np.mean(
                np.arctanh(
                    valid
                )
            )
        )
    )


def main() -> None:
    LOG_FILE.write_text(
        "",
        encoding="utf-8",
    )

    log(
        "===== Phase 5D7B2 started ====="
    )

    required_files = [
        EXPRESSION_FILE,
        METADATA_FILE,
        HIERARCHY_FILE,
        ASSIGNMENT_FILE,
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

    program_genes = {
        program: read_program_genes(path)
        for program, path in PROGRAM_FILES.items()
    }

    maturation_genes = program_genes[
        "maturation_high_increasing"
    ]

    fetal_genes = program_genes[
        "fetal_high_decreasing"
    ]

    shared_program_genes = (
        maturation_genes
        & fetal_genes
    )

    if shared_program_genes:
        raise ValueError(
            "The two developmental programs unexpectedly "
            f"share {len(shared_program_genes)} genes."
        )

    log(
        f"Maturation-high genes requested: "
        f"{len(maturation_genes)}"
    )

    log(
        f"Fetal-high genes requested: "
        f"{len(fetal_genes)}"
    )

    with gzip.open(
        EXPRESSION_FILE,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        header = (
            handle.readline()
            .rstrip("\n\r")
            .split("\t")
        )

    if (
        len(header) < 2
        or header[0] != "gene_symbol"
    ):
        raise ValueError(
            "Unexpected AHBA expression-matrix header."
        )

    sample_ids = [
        normalize_id(value)
        for value in header[1:]
    ]

    number_of_samples = len(
        sample_ids
    )

    if len(set(sample_ids)) != number_of_samples:
        raise ValueError(
            "Duplicate sample IDs detected in AHBA expression matrix."
        )

    log(
        f"AHBA strict cortical samples: {number_of_samples}"
    )

    target_union = (
        maturation_genes
        | fetal_genes
    )

    score_sums = {
        program: np.zeros(
            number_of_samples,
            dtype=np.float64,
        )
        for program in program_genes
    }

    score_counts = {
        program: np.zeros(
            number_of_samples,
            dtype=np.int32,
        )
        for program in program_genes
    }

    retained_genes = {
        program: set()
        for program in program_genes
    }

    raw_mapped_genes = {
        program: set()
        for program in program_genes
    }

    reader = pd.read_csv(
        EXPRESSION_FILE,
        sep="\t",
        compression="gzip",
        chunksize=CHUNK_SIZE,
        low_memory=False,
        dtype={
            "gene_symbol": "string",
        },
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):
        chunk[
            "gene_symbol"
        ] = chunk[
            "gene_symbol"
        ].map(
            normalize_gene
        )

        chunk = chunk.loc[
            chunk[
                "gene_symbol"
            ].isin(
                target_union
            )
        ].copy()

        if chunk.empty:
            continue

        chunk = chunk.drop_duplicates(
            subset=[
                "gene_symbol",
            ],
            keep="first",
        )

        for program, genes in program_genes.items():
            program_chunk = chunk.loc[
                chunk[
                    "gene_symbol"
                ].isin(
                    genes
                )
            ].copy()

            if program_chunk.empty:
                continue

            mapped_gene_names = set(
                program_chunk[
                    "gene_symbol"
                ]
            )

            raw_mapped_genes[
                program
            ].update(
                mapped_gene_names
            )

            values = (
                program_chunk.iloc[
                    :,
                    1:,
                ]
                .apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                .to_numpy(
                    dtype=np.float64
                )
            )

            finite_fraction = np.mean(
                np.isfinite(values),
                axis=1,
            )

            gene_means = np.nanmean(
                values,
                axis=1,
            )

            gene_standard_deviations = np.nanstd(
                values,
                axis=1,
                ddof=1,
            )

            valid_rows = (
                finite_fraction >= 0.90
            ) & (
                np.isfinite(
                    gene_standard_deviations
                )
            ) & (
                gene_standard_deviations > 0
            )

            if not valid_rows.any():
                continue

            valid_values = values[
                valid_rows,
                :
            ]

            valid_means = gene_means[
                valid_rows
            ]

            valid_standard_deviations = (
                gene_standard_deviations[
                    valid_rows
                ]
            )

            standardized = (
                valid_values
                - valid_means[
                    :,
                    None,
                ]
            ) / valid_standard_deviations[
                :,
                None,
            ]

            score_sums[
                program
            ] += np.nansum(
                standardized,
                axis=0,
            )

            score_counts[
                program
            ] += np.sum(
                np.isfinite(
                    standardized
                ),
                axis=0,
            ).astype(
                np.int32
            )

            retained_names = (
                program_chunk.loc[
                    valid_rows,
                    "gene_symbol",
                ]
            )

            retained_genes[
                program
            ].update(
                retained_names
            )

        if chunk_number % 20 == 0:
            log(
                f"Expression chunks processed: "
                f"{chunk_number}"
            )

    raw_scores = {}

    mapping_rows = []

    for program in program_genes:
        denominator = score_counts[
            program
        ].astype(float)

        score = np.divide(
            score_sums[
                program
            ],
            denominator,
            out=np.full(
                denominator.shape,
                np.nan,
                dtype=float,
            ),
            where=denominator > 0,
        )

        raw_scores[
            program
        ] = score

        mapping_rows.append(
            {
                "developmental_program":
                    program,

                "requested_program_genes":
                    len(
                        program_genes[
                            program
                        ]
                    ),

                "raw_AHBA_mapped_genes":
                    len(
                        raw_mapped_genes[
                            program
                        ]
                    ),

                "retained_nonconstant_genes":
                    len(
                        retained_genes[
                            program
                        ]
                    ),

                "retained_mapping_fraction":
                    (
                        len(
                            retained_genes[
                                program
                            ]
                        )
                        / len(
                            program_genes[
                                program
                            ]
                        )
                    ),

                "minimum_genes_per_sample":
                    int(
                        score_counts[
                            program
                        ].min()
                    ),

                "maximum_genes_per_sample":
                    int(
                        score_counts[
                            program
                        ].max()
                    ),
            }
        )

        log(
            f"{program}: "
            f"{len(retained_genes[program])} "
            f"nonconstant AHBA genes retained"
        )

    mapping_audit = pd.DataFrame(
        mapping_rows
    )

    mapping_audit.to_csv(
        TABLE_DIR
        / "phase5D7B2_program_gene_mapping_audit.tsv",
        sep="\t",
        index=False,
    )

    sample_scores = pd.DataFrame(
        {
            "sample_id":
                sample_ids,

            "maturation_high_score_mean_gene_z":
                raw_scores[
                    "maturation_high_increasing"
                ],

            "fetal_high_score_mean_gene_z":
                raw_scores[
                    "fetal_high_decreasing"
                ],
        }
    )

    sample_scores[
        "maturation_high_score_z"
    ] = zscore(
        sample_scores[
            "maturation_high_score_mean_gene_z"
        ].to_numpy(
            dtype=float
        )
    )

    sample_scores[
        "fetal_high_score_z"
    ] = zscore(
        sample_scores[
            "fetal_high_score_mean_gene_z"
        ].to_numpy(
            dtype=float
        )
    )

    sample_scores[
        "maturation_minus_fetal_balance"
    ] = (
        sample_scores[
            "maturation_high_score_z"
        ]
        -
        sample_scores[
            "fetal_high_score_z"
        ]
    )

    metadata = pd.read_csv(
        METADATA_FILE,
        sep="\t",
        dtype="string",
        low_memory=False,
    )

    required_metadata_columns = [
        "sample_id",
        "donor_id",
        "structure_acronym",
        "structure_name",
    ]

    missing_metadata_columns = [
        column
        for column in required_metadata_columns
        if column not in metadata.columns
    ]

    if missing_metadata_columns:
        raise ValueError(
            "Metadata columns missing: "
            + ", ".join(
                missing_metadata_columns
            )
        )

    for column in [
        "sample_id",
        "donor_id",
        "structure_acronym",
    ]:
        metadata[
            column
        ] = metadata[
            column
        ].map(
            normalize_id
        )

    if metadata[
        "sample_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate sample IDs in strict cortical metadata."
        )

    sample_scores = sample_scores.merge(
        metadata[
            required_metadata_columns
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    if sample_scores[
        "donor_id"
    ].isna().any():
        raise ValueError(
            "Some AHBA expression samples did not map to metadata."
        )

    hierarchy = pd.read_csv(
        HIERARCHY_FILE,
        sep="\t",
        dtype="string",
        low_memory=False,
    )

    required_hierarchy_columns = [
        "structure_acronym",
        "hierarchy_tier",
        "hierarchy_proxy_0_1",
    ]

    missing_hierarchy_columns = [
        column
        for column in required_hierarchy_columns
        if column not in hierarchy.columns
    ]

    if missing_hierarchy_columns:
        raise ValueError(
            "Hierarchy columns missing: "
            + ", ".join(
                missing_hierarchy_columns
            )
        )

    hierarchy[
        "structure_acronym"
    ] = hierarchy[
        "structure_acronym"
    ].map(
        normalize_id
    )

    hierarchy[
        "hierarchy_tier"
    ] = pd.to_numeric(
        hierarchy[
            "hierarchy_tier"
        ],
        errors="coerce",
    )

    hierarchy[
        "hierarchy_proxy_0_1"
    ] = pd.to_numeric(
        hierarchy[
            "hierarchy_proxy_0_1"
        ],
        errors="coerce",
    )

    hierarchy_conflicts = (
        hierarchy.groupby(
            "structure_acronym",
            observed=True,
        )[
            [
                "hierarchy_tier",
                "hierarchy_proxy_0_1",
            ]
        ]
        .nunique(
            dropna=True
        )
    )

    conflicting_structures = hierarchy_conflicts.loc[
        (
            hierarchy_conflicts[
                "hierarchy_tier"
            ] > 1
        )
        |
        (
            hierarchy_conflicts[
                "hierarchy_proxy_0_1"
            ] > 1
        )
    ]

    if not conflicting_structures.empty:
        raise ValueError(
            "Conflicting hierarchy assignments detected for: "
            + ", ".join(
                conflicting_structures.index.astype(str)
            )
        )

    hierarchy_mapping = (
        hierarchy[
            required_hierarchy_columns
        ]
        .sort_values(
            [
                "structure_acronym",
                "hierarchy_tier",
            ],
            na_position="last",
        )
        .drop_duplicates(
            subset=[
                "structure_acronym",
            ],
            keep="first",
        )
    )

    sample_scores = sample_scores.merge(
        hierarchy_mapping,
        on="structure_acronym",
        how="left",
        validate="many_to_one",
    )

    missing_hierarchy_mask = (
        sample_scores[
            "hierarchy_tier"
        ].isna()
        |
        sample_scores[
            "hierarchy_proxy_0_1"
        ].isna()
    )

    hierarchy_exclusions = sample_scores.loc[
        missing_hierarchy_mask,
        [
            "sample_id",
            "donor_id",
            "structure_acronym",
            "structure_name",
            "maturation_high_score_z",
            "fetal_high_score_z",
            "maturation_minus_fetal_balance",
        ],
    ].copy()

    hierarchy_exclusions[
        "exclusion_reason"
    ] = (
        "Cortical structure outside predefined "
        "neocortical hierarchy"
    )

    hierarchy_exclusions.to_csv(
        TABLE_DIR
        / "phase5D7B2_hierarchy_excluded_samples.tsv",
        sep="\t",
        index=False,
    )

    log(
        "Samples excluded from hierarchy analysis: "
        f"{len(hierarchy_exclusions)}"
    )

    if not hierarchy_exclusions.empty:
        exclusion_summary = (
            hierarchy_exclusions.groupby(
                [
                    "structure_acronym",
                    "structure_name",
                ],
                observed=True,
                dropna=False,
            )
            .agg(
                n_samples=(
                    "sample_id",
                    "nunique",
                ),

                n_donors=(
                    "donor_id",
                    "nunique",
                ),
            )
            .reset_index()
        )

        log(
            exclusion_summary.to_string(
                index=False
            )
        )

    sample_scores = sample_scores.loc[
        ~missing_hierarchy_mask
    ].copy()

    if sample_scores.empty:
        raise ValueError(
            "No AHBA samples remain after hierarchy filtering."
        )

    if (
        sample_scores[
            "hierarchy_tier"
        ].nunique()
        != 4
    ):
        raise ValueError(
            "Expected four hierarchy tiers after filtering."
        )

    log(
        "AHBA hierarchy-analysis samples retained: "
        f"{len(sample_scores)}"
    )

    assignment = pd.read_csv(
        ASSIGNMENT_FILE,
        sep="\t",
        dtype="string",
        low_memory=False,
    )

    assignment[
        "sample_id"
    ] = assignment[
        "sample_id"
    ].map(
        normalize_id
    )

    assignment[
        "usable_assignment"
    ] = parse_boolean(
        assignment[
            "usable_assignment"
        ]
    )

    assignment[
        "parcel_id_normalized"
    ] = assignment[
        "parcel_id"
    ].map(
        normalize_parcel_id
    )

    assignment_columns = [
        column
        for column in [
            "sample_id",
            "usable_assignment",
            "parcel_id_normalized",
            "parcel_label",
            "hemisphere",
            "network",
        ]
        if column in assignment.columns
    ]

    assignment_subset = (
        assignment[
            assignment_columns
        ]
        .drop_duplicates(
            subset=[
                "sample_id",
            ]
        )
    )

    sample_scores = sample_scores.merge(
        assignment_subset,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    sample_scores.to_csv(
        PROCESSED_DIR
        / "phase5D7B2_AHBA_sample_program_scores.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    donor_region = (
        sample_scores.groupby(
            [
                "donor_id",
                "structure_acronym",
                "structure_name",
                "hierarchy_tier",
                "hierarchy_proxy_0_1",
            ],
            observed=True,
        )
        .agg(
            n_samples=(
                "sample_id",
                "nunique",
            ),

            maturation_high_score_z=(
                "maturation_high_score_z",
                "mean",
            ),

            fetal_high_score_z=(
                "fetal_high_score_z",
                "mean",
            ),

            maturation_minus_fetal_balance=(
                "maturation_minus_fetal_balance",
                "mean",
            ),
        )
        .reset_index()
    )

    donor_region.to_csv(
        PROCESSED_DIR
        / "phase5D7B2_AHBA_donor_region_program_scores.tsv",
        sep="\t",
        index=False,
    )

    donor_tier = (
        donor_region.groupby(
            [
                "donor_id",
                "hierarchy_tier",
            ],
            observed=True,
        )
        .agg(
            n_regions=(
                "structure_acronym",
                "nunique",
            ),

            maturation_high_score_z=(
                "maturation_high_score_z",
                "mean",
            ),

            fetal_high_score_z=(
                "fetal_high_score_z",
                "mean",
            ),

            maturation_minus_fetal_balance=(
                "maturation_minus_fetal_balance",
                "mean",
            ),
        )
        .reset_index()
    )

    donor_tier.to_csv(
        PROCESSED_DIR
        / "phase5D7B2_AHBA_donor_tier_program_scores.tsv",
        sep="\t",
        index=False,
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    individual_rows = []
    blocked_rows = []

    for outcome, outcome_label in OUTCOMES.items():
        donor_blocks = []

        for donor_id, donor_table in donor_region.groupby(
            "donor_id",
            observed=True,
        ):
            analysis = donor_table[
                [
                    outcome,
                    "hierarchy_proxy_0_1",
                    "hierarchy_tier",
                ]
            ].dropna()

            if len(analysis) < 8:
                continue

            observed_r = spearmanr(
                analysis[
                    outcome
                ].to_numpy(
                    dtype=float
                ),
                analysis[
                    "hierarchy_proxy_0_1"
                ].to_numpy(
                    dtype=float
                ),
            ).statistic

            donor_blocks.append(
                {
                    "donor_id":
                        str(donor_id),

                    "score":
                        analysis[
                            outcome
                        ].to_numpy(
                            dtype=float
                        ),

                    "hierarchy":
                        analysis[
                            "hierarchy_proxy_0_1"
                        ].to_numpy(
                            dtype=float
                        ),

                    "observed_r":
                        float(
                            observed_r
                        ),

                    "n_regions":
                        len(
                            analysis
                        ),

                    "n_hierarchy_tiers":
                        int(
                            analysis[
                                "hierarchy_tier"
                            ].nunique()
                        ),
                }
            )

            individual_rows.append(
                {
                    "outcome":
                        outcome,

                    "outcome_label":
                        outcome_label,

                    "donor_id":
                        str(donor_id),

                    "n_regions":
                        len(
                            analysis
                        ),

                    "n_hierarchy_tiers":
                        int(
                            analysis[
                                "hierarchy_tier"
                            ].nunique()
                        ),

                    "spearman_r":
                        float(
                            observed_r
                        ),
                }
            )

        observed_correlations = [
            block[
                "observed_r"
            ]
            for block in donor_blocks
        ]

        observed_combined_r = combined_fisher_r(
            observed_correlations
        )

        observed_statistic = np.arctanh(
            np.clip(
                observed_combined_r,
                -0.999999,
                0.999999,
            )
        )

        null_statistics = np.empty(
            N_PERMUTATIONS,
            dtype=float,
        )

        for permutation_index in range(
            N_PERMUTATIONS
        ):
            permutation_correlations = []

            for block in donor_blocks:
                permuted_hierarchy = rng.permutation(
                    block[
                        "hierarchy"
                    ]
                )

                permutation_r = spearmanr(
                    block[
                        "score"
                    ],
                    permuted_hierarchy,
                ).statistic

                permutation_correlations.append(
                    float(
                        permutation_r
                    )
                )

            permutation_combined_r = combined_fisher_r(
                permutation_correlations
            )

            null_statistics[
                permutation_index
            ] = np.arctanh(
                np.clip(
                    permutation_combined_r,
                    -0.999999,
                    0.999999,
                )
            )

        permutation_p = (
            1
            + np.sum(
                np.abs(
                    null_statistics
                )
                >= abs(
                    observed_statistic
                )
            )
        ) / (
            N_PERMUTATIONS
            + 1
        )

        blocked_rows.append(
            {
                "outcome":
                    outcome,

                "outcome_label":
                    outcome_label,

                "n_donors":
                    len(
                        donor_blocks
                    ),

                "minimum_regions_per_donor":
                    min(
                        block[
                            "n_regions"
                        ]
                        for block in donor_blocks
                    ),

                "maximum_regions_per_donor":
                    max(
                        block[
                            "n_regions"
                        ]
                        for block in donor_blocks
                    ),

                "mean_donor_spearman_r":
                    float(
                        np.mean(
                            observed_correlations
                        )
                    ),

                "median_donor_spearman_r":
                    float(
                        np.median(
                            observed_correlations
                        )
                    ),

                "combined_fisher_z_r":
                    observed_combined_r,

                "positive_donor_correlations":
                    int(
                        np.sum(
                            np.asarray(
                                observed_correlations
                            ) > 0
                        )
                    ),

                "negative_donor_correlations":
                    int(
                        np.sum(
                            np.asarray(
                                observed_correlations
                            ) < 0
                        )
                    ),

                "permutation_p_two_sided":
                    permutation_p,

                "n_permutations":
                    N_PERMUTATIONS,
            }
        )

        log(
            f"{outcome}: combined r = "
            f"{observed_combined_r:.4f}; "
            f"permutation p = "
            f"{permutation_p:.6g}"
        )

    individual_results = pd.DataFrame(
        individual_rows
    )

    blocked_results = pd.DataFrame(
        blocked_rows
    )

    blocked_results[
        "fdr_bh"
    ] = bh_adjust(
        blocked_results[
            "permutation_p_two_sided"
        ].to_numpy(
            dtype=float
        )
    )

    blocked_results[
        "FDR_significant"
    ] = (
        blocked_results[
            "fdr_bh"
        ] < 0.05
    )

    individual_results.to_csv(
        TABLE_DIR
        / "phase5D7B2_individual_donor_hierarchy_correlations.tsv",
        sep="\t",
        index=False,
    )

    blocked_results.to_csv(
        TABLE_DIR
        / "phase5D7B2_blocked_hierarchy_permutation_results.tsv",
        sep="\t",
        index=False,
    )

    figure_source_rows = []

    for outcome, outcome_label in OUTCOMES.items():
        temporary = donor_tier[
            [
                "donor_id",
                "hierarchy_tier",
                outcome,
            ]
        ].copy()

        temporary = temporary.rename(
            columns={
                outcome:
                    "program_score",
            }
        )

        temporary[
            "outcome"
        ] = outcome

        temporary[
            "outcome_label"
        ] = outcome_label

        figure_source_rows.append(
            temporary
        )

    figure_source = pd.concat(
        figure_source_rows,
        ignore_index=True,
    )

    figure_source.to_csv(
        SOURCE_DIR
        / "Figure43_source_AHBA_program_hierarchy.tsv",
        sep="\t",
        index=False,
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(
            15,
            5.5,
        ),
        sharex=True,
    )

    for axis, (
        outcome,
        outcome_label,
    ) in zip(
        axes,
        OUTCOMES.items(),
    ):
        for donor_id, donor_table in donor_tier.groupby(
            "donor_id",
            observed=True,
        ):
            donor_table = donor_table.sort_values(
                "hierarchy_tier"
            )

            axis.plot(
                donor_table[
                    "hierarchy_tier"
                ],
                donor_table[
                    outcome
                ],
                marker="o",
                linewidth=1,
                alpha=0.45,
            )

        tier_mean = (
            donor_tier.groupby(
                "hierarchy_tier",
                observed=True,
            )[outcome]
            .agg(
                [
                    "mean",
                    "sem",
                ]
            )
            .reset_index()
        )

        axis.errorbar(
            tier_mean[
                "hierarchy_tier"
            ],
            tier_mean[
                "mean"
            ],
            yerr=tier_mean[
                "sem"
            ],
            marker="o",
            linewidth=2.5,
            capsize=4,
        )

        result = blocked_results.loc[
            blocked_results[
                "outcome"
            ] == outcome
        ].iloc[0]

        axis.set_title(
            outcome_label
            + "\n"
            + (
                f"blocked r = "
                f"{result['combined_fisher_z_r']:.2f}; "
                f"FDR = "
                f"{result['fdr_bh']:.3g}"
            )
        )

        axis.set_xlabel(
            "Anatomical hierarchy tier"
        )

        axis.set_xticks(
            [
                1,
                2,
                3,
                4,
            ]
        )

        axis.grid(
            axis="y",
            alpha=0.25,
        )

    axes[0].set_ylabel(
        "Adult AHBA program score"
    )

    figure.suptitle(
        "Figure 43. BrainSpan-derived developmental programs "
        "across the adult cortical hierarchy",
        fontsize=14,
        fontweight="bold",
    )

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.92,
        ]
    )

    figure.savefig(
        FIGURE_DIR
        / "Figure43_AHBA_developmental_program_hierarchy.pdf",
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "Figure43_AHBA_developmental_program_hierarchy.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    usable_assignment = sample_scores[
        "usable_assignment"
    ].fillna(
        False
    )

    completion = pd.DataFrame(
        [
            {
                "AHBA_expression_samples":
                    number_of_samples,

                "AHBA_hierarchy_analysis_samples":
                    len(
                        sample_scores
                    ),

                "hierarchy_excluded_samples":
                    number_of_samples
                    -
                    len(
                        sample_scores
                    ),

                "AHBA_strict_cortical_donors":
                    sample_scores[
                        "donor_id"
                    ].nunique(),

                "AHBA_hierarchy_structures":
                    sample_scores[
                        "structure_acronym"
                    ].nunique(),

                "hierarchy_tiers":
                    sample_scores[
                        "hierarchy_tier"
                    ].nunique(),

                "maturation_retained_genes":
                    len(
                        retained_genes[
                            "maturation_high_increasing"
                        ]
                    ),

                "fetal_retained_genes":
                    len(
                        retained_genes[
                            "fetal_high_decreasing"
                        ]
                    ),

                "hierarchy_outcomes_tested":
                    len(
                        OUTCOMES
                    ),

                "hierarchy_FDR_significant":
                    int(
                        blocked_results[
                            "FDR_significant"
                        ].sum()
                    ),

                "usable_Schaefer_samples":
                    int(
                        usable_assignment.sum()
                    ),

                "normalized_Schaefer_parcels":
                    sample_scores.loc[
                        usable_assignment,
                        "parcel_id_normalized",
                    ].nunique(),

                "figure43_generated":
                    (
                        FIGURE_DIR
                        / "Figure43_AHBA_developmental_program_hierarchy.pdf"
                    ).exists(),

                "Phase5D7B2_status":
                    "completed",
            }
        ]
    )

    completion.to_csv(
        TABLE_DIR
        / "phase5D7B2_completion_summary.tsv",
        sep="\t",
        index=False,
    )

    log("")
    log(
        "===== Phase 5D7B2 completed ====="
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
            f"Phase 5D7B2 failed: {error}"
        )

        raise
