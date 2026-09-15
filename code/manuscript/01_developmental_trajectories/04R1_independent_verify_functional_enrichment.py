#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


PROJECT = Path(
    "."
)

TABLE_DIR = (
    PROJECT
    / "07_tables"
    / "main_tables"
    / "phase5"
    / "phase5D4_R1"
)

ENRICHMENT_DIR = (
    PROJECT
    / "03_processed_data"
    / "developmental_trajectory"
    / "phase5D"
    / "phase5D4_R1"
    / "enrichment"
)

HISTORICAL_DIR = (
    PROJECT
    / "03_processed_data"
    / "developmental_trajectory"
    / "phase5D"
    / "enrichment"
)

FIGURE_DIR = (
    PROJECT
    / "06_figures"
    / "main_figures"
    / "phase5"
    / "phase5D4_R1"
)

SOURCE_DIR = (
    PROJECT
    / "06_figures"
    / "source_data"
    / "phase5"
    / "phase5D4_R1"
)

METADATA_DIR = (
    PROJECT
    / "02_metadata"
    / "phase5"
    / "phase5D4_R1"
)

LOG_DIR = (
    PROJECT
    / "09_pipeline_logs"
    / "phase5"
    / "phase5D4_R1"
)

RESULT_FILE = (
    ENRICHMENT_DIR
    / "phase5D4_R1_all_functional_enrichment_results.tsv.gz"
)

ASSIGNMENT_FILE = (
    ENRICHMENT_DIR
    / "phase5D4_R1_program_gene_assignments.tsv.gz"
)

BACKGROUND_FILE = (
    ENRICHMENT_DIR
    / "phase5D4_R1_trajectory_model_background_genes.tsv"
)

PROGRAM_COUNTS_FILE = (
    TABLE_DIR
    / "phase5D4_R1_program_gene_counts.tsv"
)

DISCORDANT_FILE = (
    TABLE_DIR
    / "phase5D4_R1_endpoint_direction_discordant_genes.tsv"
)

MAPPING_FILE = (
    TABLE_DIR
    / "phase5D4_R1_gene_set_background_mapping_audit.tsv"
)

SUMMARY_FILE = (
    TABLE_DIR
    / "phase5D4_R1_functional_enrichment_summary.tsv"
)

TOP20_FILE = (
    TABLE_DIR
    / "phase5D4_R1_top20_terms_per_program_and_source.tsv"
)

COMPLETION_FILE = (
    TABLE_DIR
    / "phase5D4_R1_completion_summary.tsv"
)

INTERNAL_AUDIT_FILE = (
    TABLE_DIR
    / "phase5D4_R1_internal_validation_audit.tsv"
)

TEST_COUNT_FILE = (
    TABLE_DIR
    / "phase5D4_R1_program_source_test_counts.tsv"
)

RESOURCE_AUDIT_FILE = (
    TABLE_DIR
    / "phase5D4_R1_msigdbr_resource_freeze_audit.tsv"
)

FIGURE_SOURCE_FILE = (
    SOURCE_DIR
    / "phase5D4_R1_Figure40_source_data.tsv"
)

FIGURE_PNG = (
    FIGURE_DIR
    / "Figure40_R1_developmental_program_functional_enrichment.png"
)

FIGURE_PDF = (
    FIGURE_DIR
    / "Figure40_R1_developmental_program_functional_enrichment.pdf"
)

METHODOLOGY_FILE = (
    METADATA_DIR
    / "phase5D4_R1_enrichment_methodology.txt"
)

SESSION_FILE = (
    METADATA_DIR
    / "phase5D4_R1_sessionInfo.txt"
)

VERIFICATION_TABLE = (
    TABLE_DIR
    / "phase5D4_R1_independent_verification_audit.tsv"
)

VERIFICATION_SUMMARY = (
    TABLE_DIR
    / "phase5D4_R1_independent_verification_summary.tsv"
)

SHA256_FILE = (
    TABLE_DIR
    / "phase5D4_R1_verified_output_SHA256.tsv"
)


EXPECTED_PROGRAMS = {
    "maturation_high_increasing": 3333,
    "fetal_high_decreasing": 5412,
}

EXPECTED_SOURCES = {
    "GO_BP",
    "GO_CC",
    "GO_MF",
    "Reactome",
    "Hallmark",
}


checks: list[dict[str, object]] = []


def add_check(
    section: str,
    item: str,
    value: object,
    passed: bool,
    detail: str,
) -> None:
    checks.append(
        {
            "section": section,
            "item": item,
            "value": str(value),
            "passed": bool(passed),
            "detail": detail,
        }
    )


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file is missing: {path}"
        )

    if path.stat().st_size == 0:
        raise RuntimeError(
            f"Required file is empty: {path}"
        )


def read_tsv(path: Path) -> pd.DataFrame:
    require_file(path)

    return pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
    )


def normalize_bool(value: object) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "passed",
    }


def normalize_symbols(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.upper()
    )


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=float,
    )

    number_of_tests = values.size

    order = np.argsort(
        values,
        kind="mergesort",
    )

    ordered_values = values[order]

    ranks = np.arange(
        1,
        number_of_tests + 1,
        dtype=float,
    )

    ordered_adjusted = (
        ordered_values
        * number_of_tests
        / ranks
    )

    ordered_adjusted = np.minimum.accumulate(
        ordered_adjusted[::-1]
    )[::-1]

    ordered_adjusted = np.clip(
        ordered_adjusted,
        0.0,
        1.0,
    )

    adjusted = np.empty_like(
        ordered_adjusted
    )

    adjusted[order] = ordered_adjusted

    return adjusted


def count_gene_ids(value: object) -> int:
    if pd.isna(value):
        return 0

    text = str(value).strip()

    if text == "":
        return 0

    genes = [
        gene.strip().upper()
        for gene in text.split("/")
        if gene.strip()
    ]

    return len(set(genes))


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_historical_program(
    path: Path,
    program: str,
) -> pd.DataFrame:
    historical = read_tsv(path)

    candidate_columns = [
        column
        for column in (
            "gene_symbol",
            "gene",
            "symbol",
        )
        if column in historical.columns
    ]

    if not candidate_columns and historical.shape[1] == 1:
        candidate_columns = [
            historical.columns[0]
        ]

    if not candidate_columns:
        raise RuntimeError(
            f"Gene column not found in {path}"
        )

    return pd.DataFrame(
        {
            "gene_symbol": normalize_symbols(
                historical[
                    candidate_columns[0]
                ]
            ),
            "developmental_program": program,
        }
    ).dropna()


required_files = [
    RESULT_FILE,
    ASSIGNMENT_FILE,
    BACKGROUND_FILE,
    PROGRAM_COUNTS_FILE,
    DISCORDANT_FILE,
    MAPPING_FILE,
    SUMMARY_FILE,
    TOP20_FILE,
    COMPLETION_FILE,
    INTERNAL_AUDIT_FILE,
    TEST_COUNT_FILE,
    RESOURCE_AUDIT_FILE,
    FIGURE_SOURCE_FILE,
    FIGURE_PNG,
    FIGURE_PDF,
    METHODOLOGY_FILE,
    SESSION_FILE,
]

for required_file in required_files:
    require_file(required_file)


results = read_tsv(
    RESULT_FILE
)

assignments = read_tsv(
    ASSIGNMENT_FILE
)

background = read_tsv(
    BACKGROUND_FILE
)

program_counts = read_tsv(
    PROGRAM_COUNTS_FILE
)

discordant = read_tsv(
    DISCORDANT_FILE
)

mapping = read_tsv(
    MAPPING_FILE
)

completion = read_tsv(
    COMPLETION_FILE
)

internal_audit = read_tsv(
    INTERNAL_AUDIT_FILE
)

resource_audit = read_tsv(
    RESOURCE_AUDIT_FILE
)


required_result_columns = {
    "developmental_program",
    "enrichment_source",
    "ID",
    "mapped_background_genes",
    "mapped_foreground_genes",
    "background_gene_count",
    "Count",
    "GeneRatio",
    "BgRatio",
    "enrichment_fold",
    "odds_ratio_haldane",
    "pvalue",
    "p_adjust",
    "FDR_significant",
    "geneID",
}

missing_result_columns = (
    required_result_columns
    - set(results.columns)
)

add_check(
    "schema",
    "result_columns_present",
    len(missing_result_columns),
    len(missing_result_columns) == 0,
    (
        "Missing columns: "
        + ", ".join(
            sorted(missing_result_columns)
        )
        if missing_result_columns
        else "All required enrichment columns are present."
    ),
)

if missing_result_columns:
    raise RuntimeError(
        "Required enrichment columns are missing."
    )


completion_row = completion.iloc[0]

add_check(
    "completion",
    "internal_status",
    completion_row[
        "Phase5D4_R1_status"
    ],
    completion_row[
        "Phase5D4_R1_status"
    ]
    == "completed_pending_independent_verification",
    "Expected status before independent verification.",
)

add_check(
    "completion",
    "all_internal_audits_passed",
    completion_row[
        "all_internal_audits_passed"
    ],
    normalize_bool(
        completion_row[
            "all_internal_audits_passed"
        ]
    ),
    "The R workflow must pass all internal audits.",
)

add_check(
    "completion",
    "historical_outputs_used_as_inputs",
    completion_row[
        "historical_outputs_used_as_inputs"
    ],
    not normalize_bool(
        completion_row[
            "historical_outputs_used_as_inputs"
        ]
    ),
    "Historical enrichment outputs must not be analytical inputs.",
)

add_check(
    "completion",
    "statistics_recomputed_from_phase5D3",
    completion_row[
        "statistics_recomputed_from_phase5D3"
    ],
    normalize_bool(
        completion_row[
            "statistics_recomputed_from_phase5D3"
        ]
    ),
    "Statistics must originate from verified Phase 5D3 assignments.",
)

add_check(
    "completion",
    "preverification_validity_flag",
    completion_row[
        "Phase5D4_outputs_valid"
    ],
    not normalize_bool(
        completion_row[
            "Phase5D4_outputs_valid"
        ]
    ),
    "The original validity flag should remain false before this audit.",
)


internal_pass = internal_audit[
    "passed"
].map(
    normalize_bool
)

add_check(
    "internal_audit",
    "all_internal_rows_passed",
    f"{internal_pass.sum()}/{len(internal_pass)}",
    bool(
        internal_pass.all()
    ),
    "Every internal audit row must be TRUE.",
)


assignments[
    "gene_symbol"
] = normalize_symbols(
    assignments[
        "gene_symbol"
    ]
)

duplicate_assignment_count = int(
    assignments[
        "gene_symbol"
    ].duplicated().sum()
)

add_check(
    "programs",
    "duplicate_program_genes",
    duplicate_assignment_count,
    duplicate_assignment_count == 0,
    "Each confident gene must occur once.",
)

observed_program_counts = (
    assignments.groupby(
        "developmental_program"
    )[
        "gene_symbol"
    ]
    .nunique()
    .to_dict()
)

for program, expected_count in (
    EXPECTED_PROGRAMS.items()
):
    observed_count = int(
        observed_program_counts.get(
            program,
            0,
        )
    )

    add_check(
        "programs",
        f"{program}_gene_count",
        observed_count,
        observed_count == expected_count,
        f"Expected {expected_count} genes.",
    )


add_check(
    "programs",
    "total_confident_program_genes",
    assignments[
        "gene_symbol"
    ].nunique(),
    assignments[
        "gene_symbol"
    ].nunique() == 8745,
    "Expected 8,745 unique confident genes.",
)

program_gene_sets = {
    program: set(
        assignments.loc[
            assignments[
                "developmental_program"
            ]
            == program,
            "gene_symbol",
        ]
    )
    for program in EXPECTED_PROGRAMS
}

program_overlap = (
    program_gene_sets[
        "maturation_high_increasing"
    ]
    & program_gene_sets[
        "fetal_high_decreasing"
    ]
)

add_check(
    "programs",
    "program_gene_overlap",
    len(program_overlap),
    len(program_overlap) == 0,
    "The two programs must be mutually exclusive.",
)


endpoint_flags = assignments[
    "endpoint_direction_consistent"
].map(
    normalize_bool
)

independent_discordant = assignments.loc[
    ~endpoint_flags
].copy()

add_check(
    "programs",
    "endpoint_direction_discordant_genes",
    independent_discordant[
        "gene_symbol"
    ].nunique(),
    independent_discordant[
        "gene_symbol"
    ].nunique() == 27,
    "Expected 27 retained endpoint-discordant genes.",
)

discordant_definition_passed = bool(
    (
        independent_discordant[
            "developmental_program"
        ]
        == "fetal_high_decreasing"
    ).all()
    and (
        independent_discordant[
            "assignment_confidence"
        ]
        .astype(str)
        .str.lower()
        == "moderate"
    ).all()
)

add_check(
    "programs",
    "discordant_gene_definition",
    discordant_definition_passed,
    discordant_definition_passed,
    "All 27 must be moderate-confidence fetal-high genes.",
)


background[
    "gene_symbol"
] = normalize_symbols(
    background[
        "gene_symbol"
    ]
)

add_check(
    "background",
    "trajectory_background_genes",
    background[
        "gene_symbol"
    ].nunique(),
    background[
        "gene_symbol"
    ].nunique() == 17762,
    "Expected 17,762 trajectory-model background genes.",
)


observed_sources = set(
    results[
        "enrichment_source"
    ].unique()
)

add_check(
    "resources",
    "five_expected_sources",
    ";".join(
        sorted(observed_sources)
    ),
    observed_sources
    == EXPECTED_SOURCES,
    "Expected GO_BP, GO_CC, GO_MF, Reactome and Hallmark.",
)

add_check(
    "resources",
    "resource_database_version",
    ";".join(
        sorted(
            resource_audit[
                "database_version"
            ]
            .astype(str)
            .unique()
        )
    ),
    set(
        resource_audit[
            "database_version"
        ].astype(str)
    )
    == {"2026.1.Hs"},
    "All frozen resources must use MSigDB 2026.1.Hs.",
)

add_check(
    "resources",
    "resource_duplicate_pairs",
    int(
        resource_audit[
            "duplicate_term_gene_pairs"
        ].sum()
    ),
    int(
        resource_audit[
            "duplicate_term_gene_pairs"
        ].sum()
    )
    == 0,
    "No duplicate frozen term-gene pairs are permitted.",
)


block_groups = results.groupby(
    [
        "developmental_program",
        "enrichment_source",
    ],
    sort=False,
)

add_check(
    "statistics",
    "program_source_blocks",
    block_groups.ngroups,
    block_groups.ngroups == 10,
    "Two programs × five enrichment sources.",
)


recomputed_pvalues = np.empty(
    len(results),
    dtype=float,
)

recomputed_adjusted = np.empty(
    len(results),
    dtype=float,
)

for _, indexes in block_groups.groups.items():
    indexes = np.asarray(
        list(indexes),
        dtype=int,
    )

    block = results.loc[indexes]

    population_size = block[
        "mapped_background_genes"
    ].to_numpy(
        dtype=int
    )

    set_size = block[
        "background_gene_count"
    ].to_numpy(
        dtype=int
    )

    foreground_size = block[
        "mapped_foreground_genes"
    ].to_numpy(
        dtype=int
    )

    overlap_size = block[
        "Count"
    ].to_numpy(
        dtype=int
    )

    block_pvalues = hypergeom.sf(
        overlap_size - 1,
        population_size,
        set_size,
        foreground_size,
    )

    recomputed_pvalues[
        indexes
    ] = block_pvalues

    recomputed_adjusted[
        indexes
    ] = bh_adjust(
        block_pvalues
    )


maximum_p_difference = float(
    np.max(
        np.abs(
            results[
                "pvalue"
            ].to_numpy(
                dtype=float
            )
            - recomputed_pvalues
        )
    )
)

maximum_adjusted_difference = float(
    np.max(
        np.abs(
            results[
                "p_adjust"
            ].to_numpy(
                dtype=float
            )
            - recomputed_adjusted
        )
    )
)

add_check(
    "statistics",
    "maximum_hypergeometric_P_difference",
    f"{maximum_p_difference:.6e}",
    maximum_p_difference < 1e-12,
    "Independent SciPy hypergeometric survival probabilities.",
)

add_check(
    "statistics",
    "maximum_BH_adjusted_P_difference",
    f"{maximum_adjusted_difference:.6e}",
    maximum_adjusted_difference < 1e-12,
    "Independent Python BH implementation within each block.",
)


gene_id_counts = results[
    "geneID"
].map(
    count_gene_ids
).to_numpy(
    dtype=int
)

stored_counts = results[
    "Count"
].to_numpy(
    dtype=int
)

add_check(
    "statistics",
    "geneID_overlap_count_consistency",
    int(
        np.sum(
            gene_id_counts
            != stored_counts
        )
    ),
    bool(
        np.array_equal(
            gene_id_counts,
            stored_counts,
        )
    ),
    "Unique genes encoded in geneID must equal Count.",
)


expected_gene_ratio = (
    results[
        "Count"
    ].astype(int).astype(str)
    + "/"
    + results[
        "mapped_foreground_genes"
    ].astype(int).astype(str)
)

expected_background_ratio = (
    results[
        "background_gene_count"
    ].astype(int).astype(str)
    + "/"
    + results[
        "mapped_background_genes"
    ].astype(int).astype(str)
)

add_check(
    "statistics",
    "GeneRatio_consistency",
    int(
        (
            results[
                "GeneRatio"
            ].astype(str)
            != expected_gene_ratio
        ).sum()
    ),
    bool(
        (
            results[
                "GeneRatio"
            ].astype(str)
            == expected_gene_ratio
        ).all()
    ),
    "GeneRatio must equal Count/mapped foreground.",
)

add_check(
    "statistics",
    "BgRatio_consistency",
    int(
        (
            results[
                "BgRatio"
            ].astype(str)
            != expected_background_ratio
        ).sum()
    ),
    bool(
        (
            results[
                "BgRatio"
            ].astype(str)
            == expected_background_ratio
        ).all()
    ),
    "BgRatio must equal set size/mapped background.",
)


count = results[
    "Count"
].to_numpy(
    dtype=float
)

foreground_size = results[
    "mapped_foreground_genes"
].to_numpy(
    dtype=float
)

set_size = results[
    "background_gene_count"
].to_numpy(
    dtype=float
)

universe_size = results[
    "mapped_background_genes"
].to_numpy(
    dtype=float
)

expected_fold = (
    count / foreground_size
) / (
    set_size / universe_size
)

expected_odds = (
    (count + 0.5)
    * (
        universe_size
        - set_size
        - foreground_size
        + count
        + 0.5
    )
    / (
        (
            foreground_size
            - count
            + 0.5
        )
        * (
            set_size
            - count
            + 0.5
        )
    )
)

fold_difference = float(
    np.max(
        np.abs(
            results[
                "enrichment_fold"
            ].to_numpy(
                dtype=float
            )
            - expected_fold
        )
    )
)

odds_difference = float(
    np.max(
        np.abs(
            results[
                "odds_ratio_haldane"
            ].to_numpy(
                dtype=float
            )
            - expected_odds
        )
    )
)

add_check(
    "statistics",
    "maximum_enrichment_fold_difference",
    f"{fold_difference:.6e}",
    fold_difference < 1e-12,
    "Independent enrichment-fold recomputation.",
)

add_check(
    "statistics",
    "maximum_Haldane_odds_difference",
    f"{odds_difference:.6e}",
    odds_difference < 1e-10,
    "Independent Haldane-corrected odds-ratio recomputation.",
)


stored_fdr = results[
    "FDR_significant"
].map(
    normalize_bool
).to_numpy()

expected_fdr = (
    results[
        "p_adjust"
    ].to_numpy(
        dtype=float
    )
    < 0.05
)

add_check(
    "statistics",
    "FDR_flag_consistency",
    int(
        np.sum(
            stored_fdr
            != expected_fdr
        )
    ),
    bool(
        np.array_equal(
            stored_fdr,
            expected_fdr,
        )
    ),
    "FDR_significant must equal adjusted P < 0.05.",
)


set_size_passed = bool(
    (
        results[
            "background_gene_count"
        ]
        .between(
            10,
            500,
            inclusive="both",
        )
    ).all()
)

add_check(
    "statistics",
    "gene_set_size_range",
    (
        f"{results['background_gene_count'].min()}"
        f"–{results['background_gene_count'].max()}"
    ),
    set_size_passed,
    "All tested sets must contain 10–500 mapped background genes.",
)


duplicate_terms = int(
    results.duplicated(
        subset=[
            "developmental_program",
            "enrichment_source",
            "ID",
        ]
    ).sum()
)

add_check(
    "statistics",
    "duplicate_program_source_terms",
    duplicate_terms,
    duplicate_terms == 0,
    "Each program-source-term combination must be unique.",
)


mapping_keys = [
    "developmental_program",
    "enrichment_source",
]

observed_test_counts = (
    results.groupby(
        mapping_keys
    )
    .size()
    .rename(
        "observed_tested_terms"
    )
    .reset_index()
)

mapping_comparison = mapping.merge(
    observed_test_counts,
    on=mapping_keys,
    how="outer",
    validate="one_to_one",
)

mapping_term_match = bool(
    (
        mapping_comparison[
            "eligible_terms_after_size_filter"
        ]
        == mapping_comparison[
            "observed_tested_terms"
        ]
    ).all()
)

add_check(
    "mapping",
    "eligible_terms_equal_result_rows",
    mapping_term_match,
    mapping_term_match,
    "Each block must contain exactly its eligible source terms.",
)


mapping_value_match = True

for keys, block in block_groups:
    program, source = keys

    audit_row = mapping.loc[
        (
            mapping[
                "developmental_program"
            ]
            == program
        )
        & (
            mapping[
                "enrichment_source"
            ]
            == source
        )
    ]

    if len(audit_row) != 1:
        mapping_value_match = False
        continue

    audit_row = audit_row.iloc[0]

    if (
        block[
            "mapped_background_genes"
        ].nunique()
        != 1
        or block[
            "mapped_foreground_genes"
        ].nunique()
        != 1
    ):
        mapping_value_match = False
        continue

    if int(
        block[
            "mapped_background_genes"
        ].iloc[0]
    ) != int(
        audit_row[
            "mapped_background_genes"
        ]
    ):
        mapping_value_match = False

    if int(
        block[
            "mapped_foreground_genes"
        ].iloc[0]
    ) != int(
        audit_row[
            "mapped_foreground_genes"
        ]
    ):
        mapping_value_match = False


add_check(
    "mapping",
    "mapped_background_foreground_consistency",
    mapping_value_match,
    mapping_value_match,
    "Stored ORA universes must equal the mapping audit.",
)


historical_maturation = read_historical_program(
    HISTORICAL_DIR
    / "phase5D4_maturation_high_increasing_genes.tsv",
    "maturation_high_increasing",
)

historical_fetal = read_historical_program(
    HISTORICAL_DIR
    / "phase5D4_fetal_high_decreasing_genes.tsv",
    "fetal_high_decreasing",
)

historical_assignments = pd.concat(
    [
        historical_maturation,
        historical_fetal,
    ],
    ignore_index=True,
).drop_duplicates()

current_keys = set(
    zip(
        assignments[
            "gene_symbol"
        ],
        assignments[
            "developmental_program"
        ],
    )
)

historical_keys = set(
    zip(
        historical_assignments[
            "gene_symbol"
        ],
        historical_assignments[
            "developmental_program"
        ],
    )
)

add_check(
    "compatibility",
    "historical_program_assignments_exact_match",
    (
        f"current={len(current_keys)};"
        f" historical={len(historical_keys)}"
    ),
    current_keys == historical_keys,
    "Confirms Phase 6 used the same gene-to-program assignments.",
)


canonical_outputs = [
    RESULT_FILE,
    ASSIGNMENT_FILE,
    BACKGROUND_FILE,
    PROGRAM_COUNTS_FILE,
    DISCORDANT_FILE,
    MAPPING_FILE,
    SUMMARY_FILE,
    TOP20_FILE,
    COMPLETION_FILE,
    INTERNAL_AUDIT_FILE,
    TEST_COUNT_FILE,
    RESOURCE_AUDIT_FILE,
    FIGURE_SOURCE_FILE,
    FIGURE_PNG,
    FIGURE_PDF,
    METHODOLOGY_FILE,
    SESSION_FILE,
]

all_outputs_present = all(
    path.is_file()
    for path in canonical_outputs
)

all_outputs_nonempty = all(
    path.stat().st_size > 100
    for path in canonical_outputs
)

add_check(
    "outputs",
    "canonical_outputs_present",
    len(canonical_outputs),
    all_outputs_present,
    "All expected Phase 5D4-R1 outputs must exist.",
)

add_check(
    "outputs",
    "canonical_outputs_nonempty",
    all_outputs_nonempty,
    all_outputs_nonempty,
    "Every expected output must exceed 100 bytes.",
)


hash_rows = []

for output_path in canonical_outputs:
    hash_rows.append(
        {
            "sha256": sha256sum(
                output_path
            ),
            "size_bytes": output_path.stat().st_size,
            "path": str(
                output_path.relative_to(
                    PROJECT
                )
            ),
        }
    )

pd.DataFrame(
    hash_rows
).to_csv(
    SHA256_FILE,
    sep="\t",
    index=False,
)


audit = pd.DataFrame(
    checks
)

audit.to_csv(
    VERIFICATION_TABLE,
    sep="\t",
    index=False,
)


all_checks_passed = bool(
    audit[
        "passed"
    ].all()
)


verification_summary = pd.DataFrame(
    [
        {
            "independent_verifier": (
                "Python scipy.stats.hypergeom "
                "plus custom Python BH"
            ),
            "python_version": platform.python_version(),
            "pandas_version": version(
                "pandas"
            ),
            "numpy_version": version(
                "numpy"
            ),
            "scipy_version": version(
                "scipy"
            ),
            "audit_checks": len(
                audit
            ),
            "audit_checks_passed": int(
                audit[
                    "passed"
                ].sum()
            ),
            "maximum_hypergeometric_P_difference": (
                maximum_p_difference
            ),
            "maximum_BH_adjusted_P_difference": (
                maximum_adjusted_difference
            ),
            "historical_program_assignments_exact_match": (
                current_keys
                == historical_keys
            ),
            "all_independent_checks_passed": (
                all_checks_passed
            ),
            "Phase5D4_R1_outputs_valid": (
                all_checks_passed
            ),
            "Phase5D4_outputs_valid": (
                all_checks_passed
            ),
            "ready_for_visual_review_and_canonical_closure": (
                all_checks_passed
            ),
            "ready_for_phase8E_manuscript_integration": False,
            "Phase5D4_R1_status": (
                "independently_verified"
                if all_checks_passed
                else "independent_verification_failed"
            ),
        }
    ]
)

verification_summary.to_csv(
    VERIFICATION_SUMMARY,
    sep="\t",
    index=False,
)


print(
    "===== PHASE 5D4-R1 INDEPENDENT VERIFICATION ====="
)

print(
    audit.to_string(
        index=False
    )
)

print()

print(
    "===== INDEPENDENT VERIFICATION SUMMARY ====="
)

print(
    verification_summary.to_string(
        index=False
    )
)

print()

print(
    f"Verification audit: {VERIFICATION_TABLE}"
)

print(
    f"Verification summary: {VERIFICATION_SUMMARY}"
)

print(
    f"SHA256 manifest: {SHA256_FILE}"
)


if not all_checks_passed:
    failed = audit.loc[
        ~audit[
            "passed"
        ]
    ]

    print(
        "\nFAILED CHECKS:",
        file=sys.stderr,
    )

    print(
        failed.to_string(
            index=False
        ),
        file=sys.stderr,
    )

    raise SystemExit(1)


print(
    "PHASE 5D4-R1 INDEPENDENT VERIFICATION: PASSED"
)
