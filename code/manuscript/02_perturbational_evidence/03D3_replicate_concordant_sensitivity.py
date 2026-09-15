#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(
    "."
)

QC_FILE = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6E/"
      "phase6E3D1_signature_quality_flags.tsv.gz"
)

SIGNATURE_SCORE_FILE = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6E/"
      "phase6E3A_quantitative_signature_scores.tsv.gz"
)

FULL_PRIMARY_FILE = (
    PROJECT
    / "07_tables/main_tables/phase6/"
      "phase6E3C_cross_scope_consistency.tsv"
)

PROCESSED_DIR = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6E"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase6"
)

FIGURE_DIR = (
    PROJECT
    / "08_figures/supplementary_figures/phase6"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase6/"
      "phase6E3D3_replicate_concordant_sensitivity.log"
)

FILTERED_SIGNATURE_FILE = (
    PROCESSED_DIR
    / "phase6E3D3_replicate_concordant_signature_scores.tsv.gz"
)

SCORE_MAPPING_FILE = (
    TABLE_DIR
    / "phase6E3D3_score_column_mapping.tsv"
)

COVERAGE_FILE = (
    TABLE_DIR
    / "phase6E3D3_replicate_concordant_coverage_by_chemical.tsv"
)

BOOTSTRAP_FILE = (
    TABLE_DIR
    / "phase6E3D3_hierarchical_bootstrap_results.tsv"
)

COMPARISON_FILE = (
    TABLE_DIR
    / "phase6E3D3_full_vs_replicate_concordant_comparison.tsv"
)

AXIS_CONCORDANCE_FILE = (
    TABLE_DIR
    / "phase6E3D3_axis_concordance_summary.tsv"
)

CHEMICAL_SUMMARY_FILE = (
    TABLE_DIR
    / "phase6E3D3_chemical_sensitivity_summary.tsv"
)

COMPLETION_FILE = (
    TABLE_DIR
    / "phase6E3D3_completion_summary.tsv"
)

FIGURE_PNG = (
    FIGURE_DIR
    / "Supplementary_Figure_6E3D3_replicate_concordant_sensitivity.png"
)

FIGURE_PDF = (
    FIGURE_DIR
    / "Supplementary_Figure_6E3D3_replicate_concordant_sensitivity.pdf"
)

FIGURE_CAPTION = (
    FIGURE_DIR
    / "Supplementary_Figure_6E3D3_caption.txt"
)


EXPECTED_SIGNATURES = 1232
EXPECTED_CHEMICALS = 15
EXPECTED_AXES = 6
EXPECTED_COMPARISON_ROWS = 90

CC_THRESHOLD = 0.20
MIN_REPLICATES = 3
INVALID_SENTINEL = -666.0

BOOTSTRAP_ITERATIONS = 5000
RANDOM_SEED = 20260721

EPSILON = 1e-12


AXES = [
    "developmental_shift",
    "early_program_suppression",
    "late_maturation_support",
    "injury_stress",
    "late_minus_injury",
    "late_minus_early",
]


for directory in [
    PROCESSED_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    LOG_FILE.parent,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def clean_text(
    value: object,
) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "na",
    }:
        return ""

    return text


def normalize_text(
    value: object,
) -> str:
    text = clean_text(
        value
    ).lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )

    return text.strip(
        "_"
    )


def normalize_chemical(
    value: object,
) -> str:
    text = clean_text(
        value
    ).lower()

    text = re.sub(
        r"[-_/]+",
        " ",
        text,
    )

    return " ".join(
        text.split()
    )


def parse_bool(
    value: object,
) -> bool:
    if isinstance(
        value,
        bool,
    ):
        return value

    if pd.isna(value):
        return False

    return str(
        value
    ).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def find_column(
    table: pd.DataFrame,
    candidates: list[str],
    label: str,
) -> str:
    normalized_lookup = {
        normalize_text(column): column
        for column in table.columns
    }

    for candidate in candidates:
        normalized_candidate = normalize_text(
            candidate
        )

        if normalized_candidate in normalized_lookup:
            return normalized_lookup[
                normalized_candidate
            ]

    raise RuntimeError(
        (
            f"Could not identify {label}. "
            f"Expected one of: {candidates}. "
            f"Available columns: {list(table.columns)}"
        )
    )


def choose_axis_score_column(
    table: pd.DataFrame,
    axis: str,
) -> str:
    normalized_columns = {
        column: normalize_text(
            column
        )
        for column in table.columns
    }

    exact_candidates = [
        axis,
        f"{axis}_score",
        f"{axis}_full_gene",
        f"{axis}_full_gene_score",
        f"{axis}_score_full_gene",
        f"full_gene_{axis}",
        f"full_gene_{axis}_score",
        f"score_{axis}_full_gene",
    ]

    normalized_exact = {
        normalize_text(candidate)
        for candidate in exact_candidates
    }

    for column, normalized in normalized_columns.items():
        if normalized in normalized_exact:
            return column

    forbidden_tokens = [
        "landmark",
        "empirical",
        "null",
        "pvalue",
        "p_value",
        "fdr",
        "qvalue",
        "q_value",
        "neural",
        "cell_balanced",
        "chemical_summary",
        "median_summary",
        "leave_program_out",
        "leaveprogramout",
        "leave_out",
        "leaveout",
    ]

    candidates: list[
        tuple[int, str]
    ] = []

    normalized_axis = normalize_text(
        axis
    )

    for column, normalized in normalized_columns.items():
        if normalized_axis not in normalized:
            continue

        if any(
            token in normalized
            for token in forbidden_tokens
        ):
            continue

        score = 0

        if normalized == normalized_axis:
            score += 100

        if normalized.startswith(
            normalized_axis
        ):
            score += 30

        if normalized.endswith(
            normalized_axis
        ):
            score += 25

        if "full_gene" in normalized:
            score += 20

        if "score" in normalized:
            score += 10

        if "observed" in normalized:
            score += 5

        candidates.append(
            (
                score,
                column,
            )
        )

    if not candidates:
        raise RuntimeError(
            (
                f"Could not identify signature-level score column "
                f"for axis '{axis}'. Available columns:\n"
                + "\n".join(
                    table.columns.astype(str)
                )
            )
        )

    candidates.sort(
        key=lambda item: (
            -item[0],
            item[1],
        )
    )

    best_score = candidates[
        0
    ][
        0
    ]

    tied = [
        column
        for score, column in candidates
        if score == best_score
    ]

    if len(
        tied
    ) > 1:
        raise RuntimeError(
            (
                f"Ambiguous score columns for axis '{axis}': "
                + "|".join(
                    tied
                )
            )
        )

    return candidates[
        0
    ][
        1
    ]


def direction_from_score(
    value: object,
) -> str:
    if pd.isna(value):
        return "not_available"

    numeric = float(
        value
    )

    if numeric > EPSILON:
        return "positive"

    if numeric < -EPSILON:
        return "negative"

    return "zero"


def benjamini_hochberg(
    p_values: pd.Series,
) -> pd.Series:
    result = pd.Series(
        np.nan,
        index=p_values.index,
        dtype=float,
    )

    valid = p_values.dropna()

    if valid.empty:
        return result

    order = np.argsort(
        valid.to_numpy()
    )

    ordered_p = valid.to_numpy()[
        order
    ]

    number = len(
        ordered_p
    )

    adjusted = (
        ordered_p
        * number
        / np.arange(
            1,
            number + 1,
        )
    )

    adjusted = np.minimum.accumulate(
        adjusted[
            ::-1
        ]
    )[
        ::-1
    ]

    adjusted = np.clip(
        adjusted,
        0,
        1,
    )

    ordered_index = valid.index.to_numpy()[
        order
    ]

    result.loc[
        ordered_index
    ] = adjusted

    return result


def hierarchical_bootstrap(
    group: pd.DataFrame,
    score_column: str,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    cell_arrays: dict[
        str,
        np.ndarray,
    ] = {}

    for cell_id, cell_group in group.groupby(
        "cell_id_D3",
        sort=True,
    ):
        values = pd.to_numeric(
            cell_group[
                score_column
            ],
            errors="coerce",
        ).dropna().to_numpy(
            dtype=float
        )

        if len(
            values
        ):
            cell_arrays[
                str(
                    cell_id
                )
            ] = values

    cells = list(
        cell_arrays
    )

    if not cells:
        return np.array(
            [],
            dtype=float,
        )

    output = np.empty(
        iterations,
        dtype=float,
    )

    number_of_cells = len(
        cells
    )

    for iteration in range(
        iterations
    ):
        selected_cells = rng.choice(
            cells,
            size=number_of_cells,
            replace=True,
        )

        cell_statistics: list[
            float
        ] = []

        for cell in selected_cells:
            values = cell_arrays[
                str(
                    cell
                )
            ]

            resampled = rng.choice(
                values,
                size=len(
                    values
                ),
                replace=True,
            )

            cell_statistics.append(
                float(
                    np.median(
                        resampled
                    )
                )
            )

        output[
            iteration
        ] = float(
            np.mean(
                cell_statistics
            )
        )

    return output


def equal_cell_weight_score(
    group: pd.DataFrame,
    score_column: str,
) -> tuple[
    float,
    int,
    int,
]:
    usable = group.loc[
        pd.to_numeric(
            group[
                score_column
            ],
            errors="coerce",
        ).notna()
    ].copy()

    if usable.empty:
        return (
            np.nan,
            0,
            0,
        )

    usable[
        score_column
    ] = pd.to_numeric(
        usable[
            score_column
        ],
        errors="coerce",
    )

    cell_medians = (
        usable.groupby(
            "cell_id_D3",
            sort=True,
        )[
            score_column
        ]
        .median()
    )

    if cell_medians.empty:
        return (
            np.nan,
            0,
            0,
        )

    return (
        float(
            cell_medians.mean()
        ),
        int(
            len(
                usable
            )
        ),
        int(
            len(
                cell_medians
            )
        ),
    )


def main() -> None:
    required_files = [
        QC_FILE,
        SIGNATURE_SCORE_FILE,
        FULL_PRIMARY_FILE,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required Phase 6E3D3 inputs:\n"
            + "\n".join(
                missing_files
            )
        )

    quality = pd.read_csv(
        QC_FILE,
        sep="\t",
        low_memory=False,
    )

    signature_scores = pd.read_csv(
        SIGNATURE_SCORE_FILE,
        sep="\t",
        low_memory=False,
    )

    full_primary = pd.read_csv(
        FULL_PRIMARY_FILE,
        sep="\t",
        low_memory=False,
    )

    if len(
        quality
    ) != EXPECTED_SIGNATURES:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_SIGNATURES} QC rows, "
                f"found {len(quality)}."
            )
        )

    if len(
        signature_scores
    ) != EXPECTED_SIGNATURES:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_SIGNATURES} signature-score rows, "
                f"found {len(signature_scores)}."
            )
        )

    if len(
        full_primary
    ) != EXPECTED_COMPARISON_ROWS:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_COMPARISON_ROWS} primary comparison "
                f"rows, found {len(full_primary)}."
            )
        )

    qc_signature_id = find_column(
        quality,
        [
            "sig_id",
            "signature_id",
            "signature",
            "id",
        ],
        "QC signature identifier",
    )

    score_signature_id = find_column(
        signature_scores,
        [
            "sig_id",
            "signature_id",
            "signature",
            "id",
        ],
        "signature-score identifier",
    )

    chemical_column = find_column(
        quality,
        [
            "chemical_query_QC",
            "chemical_query",
            "pert_iname_metric",
            "pert_iname",
        ],
        "chemical name",
    )

    cell_column = find_column(
        quality,
        [
            "cell_id_QC",
            "cell_id",
            "cell",
        ],
        "cell identifier",
    )

    required_qc_columns = [
        "distil_cc_q75",
        "distil_nsample",
    ]

    missing_qc_columns = [
        column
        for column in required_qc_columns
        if column not in quality.columns
    ]

    if missing_qc_columns:
        raise RuntimeError(
            "Missing QC columns: "
            + "|".join(
                missing_qc_columns
            )
        )

    score_mapping_rows: list[
        dict[str, str]
    ] = []

    axis_score_columns: dict[
        str,
        str,
    ] = {}

    for axis in AXES:
        selected_column = choose_axis_score_column(
            signature_scores,
            axis,
        )

        axis_score_columns[
            axis
        ] = selected_column

        score_mapping_rows.append(
            {
                "axis_name":
                    axis,

                "selected_signature_score_column":
                    selected_column,

                "selection_rule":
                    (
                        "Primary full-gene signature-level score; "
                        "landmark and leave-program-out columns excluded."
                    ),
            }
        )

    score_mapping = pd.DataFrame(
        score_mapping_rows
    )

    score_mapping.to_csv(
        SCORE_MAPPING_FILE,
        sep="\t",
        index=False,
    )

    quality_subset = quality[
        [
            qc_signature_id,
            chemical_column,
            cell_column,
            "distil_cc_q75",
            "distil_nsample",
        ]
    ].copy()

    quality_subset = quality_subset.rename(
        columns={
            qc_signature_id:
                "signature_id_D3",

            chemical_column:
                "chemical_query_D3",

            cell_column:
                "cell_id_D3",
        }
    )

    score_subset_columns = [
        score_signature_id,
    ] + list(
        dict.fromkeys(
            axis_score_columns.values()
        )
    )

    score_subset = signature_scores[
        score_subset_columns
    ].copy()

    score_subset = score_subset.rename(
        columns={
            score_signature_id:
                "signature_id_D3",
        }
    )

    if quality_subset[
        "signature_id_D3"
    ].duplicated().any():
        raise RuntimeError(
            "QC signature identifiers are not unique."
        )

    if score_subset[
        "signature_id_D3"
    ].duplicated().any():
        raise RuntimeError(
            "Signature-score identifiers are not unique."
        )

    merged = score_subset.merge(
        quality_subset,
        on="signature_id_D3",
        how="inner",
        validate="one_to_one",
    )

    if len(
        merged
    ) != EXPECTED_SIGNATURES:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_SIGNATURES} merged signatures, "
                f"found {len(merged)}."
            )
        )

    merged[
        "chemical_normalized_D3"
    ] = merged[
        "chemical_query_D3"
    ].map(
        normalize_chemical
    )

    merged[
        "cell_id_D3"
    ] = merged[
        "cell_id_D3"
    ].map(
        clean_text
    )

    merged[
        "distil_cc_q75"
    ] = pd.to_numeric(
        merged[
            "distil_cc_q75"
        ],
        errors="coerce",
    )

    merged[
        "distil_nsample"
    ] = pd.to_numeric(
        merged[
            "distil_nsample"
        ],
        errors="coerce",
    )

    merged[
        "replicate_concordant_D3"
    ] = (
        merged[
            "distil_cc_q75"
        ].notna()
        & merged[
            "distil_cc_q75"
        ].ne(
            INVALID_SENTINEL
        )
        & merged[
            "distil_cc_q75"
        ].ge(
            CC_THRESHOLD
        )
        & merged[
            "distil_nsample"
        ].notna()
        & merged[
            "distil_nsample"
        ].ge(
            MIN_REPLICATES
        )
    )

    filtered = merged.loc[
        merged[
            "replicate_concordant_D3"
        ]
    ].copy()

    invalid_selected = filtered.loc[
        filtered[
            "distil_cc_q75"
        ].eq(
            INVALID_SENTINEL
        )
    ]

    if not invalid_selected.empty:
        raise RuntimeError(
            "The −666 sentinel entered the replicate-concordant subset."
        )

    filtered.to_csv(
        FILTERED_SIGNATURE_FILE,
        sep="\t",
        index=False,
        compression="gzip",
    )

    matched_chemicals = (
        full_primary[
            "chemical_query"
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if len(
        matched_chemicals
    ) != EXPECTED_CHEMICALS:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_CHEMICALS} matched chemicals, "
                f"found {len(matched_chemicals)}."
            )
        )

    coverage_rows: list[
        dict[str, object]
    ] = []

    for chemical in matched_chemicals:
        chemical_normalized = normalize_chemical(
            chemical
        )

        all_group = merged.loc[
            merged[
                "chemical_normalized_D3"
            ].eq(
                chemical_normalized
            )
        ]

        filtered_group = filtered.loc[
            filtered[
                "chemical_normalized_D3"
            ].eq(
                chemical_normalized
            )
        ]

        total_signatures = len(
            all_group
        )

        retained_signatures = len(
            filtered_group
        )

        retained_cells = filtered_group[
            "cell_id_D3"
        ].nunique()

        coverage_class = (
            "adequate_multicell"
            if (
                retained_signatures >= 3
                and retained_cells >= 2
            )
            else (
                "descriptive_single_cell"
                if retained_signatures >= 1
                else "no_replicate_concordant_signatures"
            )
        )

        coverage_rows.append(
            {
                "chemical_query":
                    chemical,

                "chemical_normalized":
                    chemical_normalized,

                "total_Level5_signatures":
                    total_signatures,

                "total_cells":
                    all_group[
                        "cell_id_D3"
                    ].nunique(),

                "replicate_concordant_signatures":
                    retained_signatures,

                "replicate_concordant_cells":
                    retained_cells,

                "retained_signature_fraction":
                    (
                        retained_signatures
                        / total_signatures
                        if total_signatures
                        else np.nan
                    ),

                "median_retained_distil_cc_q75":
                    (
                        float(
                            filtered_group[
                                "distil_cc_q75"
                            ].median()
                        )
                        if retained_signatures
                        else np.nan
                    ),

                "median_retained_distil_nsample":
                    (
                        float(
                            filtered_group[
                                "distil_nsample"
                            ].median()
                        )
                        if retained_signatures
                        else np.nan
                    ),

                "coverage_class":
                    coverage_class,

                "bootstrap_inference_eligible":
                    bool(
                        retained_signatures >= 3
                        and retained_cells >= 2
                    ),
            }
        )

    coverage = pd.DataFrame(
        coverage_rows
    )

    coverage.to_csv(
        COVERAGE_FILE,
        sep="\t",
        index=False,
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    bootstrap_rows: list[
        dict[str, object]
    ] = []

    for chemical in matched_chemicals:
        chemical_normalized = normalize_chemical(
            chemical
        )

        chemical_group = filtered.loc[
            filtered[
                "chemical_normalized_D3"
            ].eq(
                chemical_normalized
            )
        ]

        for axis in AXES:
            score_column = axis_score_columns[
                axis
            ]

            observed_score, signature_count, cell_count = (
                equal_cell_weight_score(
                    chemical_group,
                    score_column,
                )
            )

            inference_eligible = bool(
                signature_count >= 3
                and cell_count >= 2
            )

            bootstrap_values = np.array(
                [],
                dtype=float,
            )

            lower_ci = np.nan
            upper_ci = np.nan
            bootstrap_median = np.nan
            bootstrap_p_two_sided = np.nan

            if inference_eligible:
                bootstrap_values = hierarchical_bootstrap(
                    chemical_group,
                    score_column,
                    BOOTSTRAP_ITERATIONS,
                    rng,
                )

                if len(
                    bootstrap_values
                ) != BOOTSTRAP_ITERATIONS:
                    raise RuntimeError(
                        (
                            f"Bootstrap failed for {chemical}, {axis}."
                        )
                    )

                lower_ci = float(
                    np.quantile(
                        bootstrap_values,
                        0.025,
                    )
                )

                upper_ci = float(
                    np.quantile(
                        bootstrap_values,
                        0.975,
                    )
                )

                bootstrap_median = float(
                    np.median(
                        bootstrap_values
                    )
                )

                probability_nonpositive = (
                    1
                    + np.sum(
                        bootstrap_values <= 0
                    )
                ) / (
                    BOOTSTRAP_ITERATIONS
                    + 1
                )

                probability_nonnegative = (
                    1
                    + np.sum(
                        bootstrap_values >= 0
                    )
                ) / (
                    BOOTSTRAP_ITERATIONS
                    + 1
                )

                bootstrap_p_two_sided = min(
                    1.0,
                    2.0
                    * min(
                        probability_nonpositive,
                        probability_nonnegative,
                    ),
                )

            bootstrap_rows.append(
                {
                    "chemical_query":
                        chemical,

                    "chemical_normalized":
                        chemical_normalized,

                    "axis_name":
                        axis,

                    "selected_signature_score_column":
                        score_column,

                    "replicate_concordant_signature_count":
                        signature_count,

                    "replicate_concordant_cell_count":
                        cell_count,

                    "equal_cell_weight_observed_score":
                        observed_score,

                    "observed_direction":
                        direction_from_score(
                            observed_score
                        ),

                    "bootstrap_iterations":
                        (
                            BOOTSTRAP_ITERATIONS
                            if inference_eligible
                            else 0
                        ),

                    "bootstrap_inference_eligible":
                        inference_eligible,

                    "bootstrap_median":
                        bootstrap_median,

                    "bootstrap_CI_lower_95":
                        lower_ci,

                    "bootstrap_CI_upper_95":
                        upper_ci,

                    "bootstrap_p_two_sided":
                        bootstrap_p_two_sided,
                }
            )

    bootstrap = pd.DataFrame(
        bootstrap_rows
    )

    bootstrap[
        "bootstrap_FDR"
    ] = benjamini_hochberg(
        bootstrap[
            "bootstrap_p_two_sided"
        ]
    )

    bootstrap[
        "bootstrap_CI_excludes_zero"
    ] = (
        (
            bootstrap[
                "bootstrap_CI_lower_95"
            ] > 0
        )
        | (
            bootstrap[
                "bootstrap_CI_upper_95"
            ] < 0
        )
    )

    bootstrap[
        "bootstrap_absolute_support"
    ] = (
        bootstrap[
            "bootstrap_inference_eligible"
        ]
        & bootstrap[
            "bootstrap_FDR"
        ].lt(
            0.05
        )
        & bootstrap[
            "bootstrap_CI_excludes_zero"
        ]
    )

    bootstrap.to_csv(
        BOOTSTRAP_FILE,
        sep="\t",
        index=False,
    )

    full = full_primary.copy()

    full[
        "chemical_normalized"
    ] = full[
        "chemical_query"
    ].map(
        normalize_chemical
    )

    full[
        "primary_absolute_direction_supported"
    ] = full[
        "primary_absolute_direction_supported"
    ].map(
        parse_bool
    )

    comparison = full.merge(
        bootstrap,
        on=[
            "chemical_normalized",
            "axis_name",
        ],
        how="left",
        validate="one_to_one",
        suffixes=(
            "_full",
            "_replicate_concordant",
        ),
    )

    if len(
        comparison
    ) != EXPECTED_COMPARISON_ROWS:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_COMPARISON_ROWS} comparison rows, "
                f"found {len(comparison)}."
            )
        )

    comparison[
        "replicate_concordant_score_available"
    ] = comparison[
        "equal_cell_weight_observed_score"
    ].notna()

    comparison[
        "full_replicate_concordant_sign_concordant"
    ] = (
        comparison[
            "replicate_concordant_score_available"
        ]
        & comparison[
            "primary_observed_direction"
        ].eq(
            comparison[
                "observed_direction"
            ]
        )
    )

    comparison[
        "full_absolute_support_direction_retained"
    ] = (
        comparison[
            "primary_absolute_direction_supported"
        ]
        & comparison[
            "full_replicate_concordant_sign_concordant"
        ]
    )

    comparison[
        "full_absolute_support_bootstrap_replicated"
    ] = (
        comparison[
            "primary_absolute_direction_supported"
        ]
        & comparison[
            "full_replicate_concordant_sign_concordant"
        ]
        & comparison[
            "bootstrap_absolute_support"
        ].fillna(
            False
        )
    )

    comparison[
        "absolute_magnitude_ratio_filtered_to_full"
    ] = np.where(
        comparison[
            "primary_observed_score"
        ].abs() > EPSILON,
        (
            comparison[
                "equal_cell_weight_observed_score"
            ].abs()
            / comparison[
                "primary_observed_score"
            ].abs()
        ),
        np.nan,
    )

    comparison.to_csv(
        COMPARISON_FILE,
        sep="\t",
        index=False,
    )

    concordance_rows: list[
        dict[str, object]
    ] = []

    for axis, group in comparison.groupby(
        "axis_name",
        sort=True,
    ):
        usable = group.loc[
            group[
                "primary_observed_score"
            ].notna()
            & group[
                "equal_cell_weight_observed_score"
            ].notna()
        ].copy()

        pearson = np.nan
        spearman = np.nan

        if len(
            usable
        ) >= 3:
            pearson = float(
                usable[
                    [
                        "primary_observed_score",
                        "equal_cell_weight_observed_score",
                    ]
                ].corr(
                    method="pearson"
                ).iloc[
                    0,
                    1,
                ]
            )

            spearman = float(
                usable[
                    [
                        "primary_observed_score",
                        "equal_cell_weight_observed_score",
                    ]
                ].corr(
                    method="spearman"
                ).iloc[
                    0,
                    1,
                ]
            )

        full_supported = group.loc[
            group[
                "primary_absolute_direction_supported"
            ]
        ]

        concordance_rows.append(
            {
                "axis_name":
                    axis,

                "chemicals_with_replicate_concordant_score":
                    len(
                        usable
                    ),

                "pearson_full_vs_replicate_concordant":
                    pearson,

                "spearman_full_vs_replicate_concordant":
                    spearman,

                "overall_sign_concordance_fraction":
                    (
                        usable[
                            "full_replicate_concordant_sign_concordant"
                        ].mean()
                        if len(
                            usable
                        )
                        else np.nan
                    ),

                "full_absolute_supported_chemical_count":
                    len(
                        full_supported
                    ),

                "full_absolute_support_direction_retained_count":
                    int(
                        full_supported[
                            "full_absolute_support_direction_retained"
                        ].sum()
                    ),

                "full_absolute_support_bootstrap_replicated_count":
                    int(
                        full_supported[
                            "full_absolute_support_bootstrap_replicated"
                        ].sum()
                    ),
            }
        )

    axis_concordance = pd.DataFrame(
        concordance_rows
    )

    axis_concordance.to_csv(
        AXIS_CONCORDANCE_FILE,
        sep="\t",
        index=False,
    )

    chemical_rows: list[
        dict[str, object]
    ] = []

    coverage_lookup = coverage.set_index(
        "chemical_normalized"
    )

    for chemical, group in comparison.groupby(
        "chemical_query_full",
        sort=True,
    ):
        chemical_normalized = normalize_chemical(
            chemical
        )

        coverage_row = coverage_lookup.loc[
            chemical_normalized
        ]

        full_supported = group.loc[
            group[
                "primary_absolute_direction_supported"
            ]
        ]

        available = group.loc[
            group[
                "replicate_concordant_score_available"
            ]
        ]

        full_supported_count = len(
            full_supported
        )

        direction_retained_count = int(
            full_supported[
                "full_absolute_support_direction_retained"
            ].sum()
        )

        bootstrap_replicated_count = int(
            full_supported[
                "full_absolute_support_bootstrap_replicated"
            ].sum()
        )

        overall_sign_concordance_count = int(
            available[
                "full_replicate_concordant_sign_concordant"
            ].sum()
        )

        available_axis_count = len(
            available
        )

        coverage_class = clean_text(
            coverage_row[
                "coverage_class"
            ]
        )

        if coverage_class == "no_replicate_concordant_signatures":
            sensitivity_class = (
                "not_evaluable_no_replicate_concordant_profiles"
            )

        elif coverage_class == "descriptive_single_cell":
            sensitivity_class = (
                "descriptive_only_single_cell_or_low_signature_count"
            )

        elif (
            full_supported_count > 0
            and bootstrap_replicated_count
            == full_supported_count
        ):
            sensitivity_class = (
                "all_primary_absolute_axes_bootstrap_replicated"
            )

        elif (
            full_supported_count > 0
            and direction_retained_count
            == full_supported_count
        ):
            sensitivity_class = (
                "all_primary_absolute_axes_directionally_retained"
            )

        elif (
            full_supported_count > 0
            and direction_retained_count > 0
        ):
            sensitivity_class = (
                "partial_primary_axis_retention"
            )

        elif full_supported_count > 0:
            sensitivity_class = (
                "primary_absolute_support_not_retained"
            )

        elif (
            available_axis_count > 0
            and overall_sign_concordance_count
            == available_axis_count
        ):
            sensitivity_class = (
                "no_primary_absolute_axes_but_all_directions_stable"
            )

        else:
            sensitivity_class = (
                "no_primary_absolute_axes_mixed_directional_stability"
            )

        chemical_rows.append(
            {
                "chemical_query":
                    chemical,

                "replicate_concordant_signatures":
                    int(
                        coverage_row[
                            "replicate_concordant_signatures"
                        ]
                    ),

                "replicate_concordant_cells":
                    int(
                        coverage_row[
                            "replicate_concordant_cells"
                        ]
                    ),

                "coverage_class":
                    coverage_class,

                "available_axis_count":
                    available_axis_count,

                "overall_sign_concordant_axis_count":
                    overall_sign_concordance_count,

                "overall_sign_concordance_fraction":
                    (
                        overall_sign_concordance_count
                        / available_axis_count
                        if available_axis_count
                        else np.nan
                    ),

                "primary_absolute_supported_axis_count":
                    full_supported_count,

                "primary_absolute_direction_retained_count":
                    direction_retained_count,

                "primary_absolute_bootstrap_replicated_count":
                    bootstrap_replicated_count,

                "replicate_concordant_sensitivity_class":
                    sensitivity_class,
            }
        )

    chemical_summary = pd.DataFrame(
        chemical_rows
    ).sort_values(
        [
            "coverage_class",
            "primary_absolute_bootstrap_replicated_count",
            "overall_sign_concordance_fraction",
            "chemical_query",
        ],
        ascending=[
            True,
            False,
            False,
            True,
        ],
    )

    chemical_summary.to_csv(
        CHEMICAL_SUMMARY_FILE,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------
    # Supplementary figure
    # ------------------------------------------------------------

    figure_order = (
        chemical_summary.sort_values(
            [
                "replicate_concordant_cells",
                "replicate_concordant_signatures",
                "chemical_query",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )[
            "chemical_query"
        ]
        .tolist()
    )

    matrix = np.full(
        (
            len(
                figure_order
            ),
            len(
                AXES
            ),
        ),
        np.nan,
        dtype=float,
    )

    for row_index, chemical in enumerate(
        figure_order
    ):
        group = comparison.loc[
            comparison[
                "chemical_query_full"
            ].eq(
                chemical
            )
        ]

        for column_index, axis in enumerate(
            AXES
        ):
            axis_row = group.loc[
                group[
                    "axis_name"
                ].eq(
                    axis
                )
            ]

            if len(
                axis_row
            ) != 1:
                continue

            axis_row = axis_row.iloc[
                0
            ]

            if not bool(
                axis_row[
                    "replicate_concordant_score_available"
                ]
            ):
                matrix[
                    row_index,
                    column_index,
                ] = np.nan

            elif bool(
                axis_row[
                    "full_absolute_support_bootstrap_replicated"
                ]
            ):
                matrix[
                    row_index,
                    column_index,
                ] = 3

            elif bool(
                axis_row[
                    "full_absolute_support_direction_retained"
                ]
            ):
                matrix[
                    row_index,
                    column_index,
                ] = 2

            elif bool(
                axis_row[
                    "full_replicate_concordant_sign_concordant"
                ]
            ):
                matrix[
                    row_index,
                    column_index,
                ] = 1

            else:
                matrix[
                    row_index,
                    column_index,
                ] = 0

    plt.figure(
        figsize=(
            14,
            9,
        )
    )

    image = plt.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=3,
    )

    plt.xticks(
        np.arange(
            len(
                AXES
            )
        ),
        [
            axis.replace(
                "_",
                "\n",
            )
            for axis in AXES
        ],
        rotation=0,
        fontsize=9,
    )

    plt.yticks(
        np.arange(
            len(
                figure_order
            )
        ),
        figure_order,
        fontsize=9,
    )

    plt.xlabel(
        "Developmental and mechanistic score axis"
    )

    plt.ylabel(
        "LINCS chemical"
    )

    plt.title(
        (
            "Replicate-concordant sensitivity of quantitative "
            "LINCS conclusions"
        )
    )

    colorbar = plt.colorbar(
        image,
        ticks=[
            0,
            1,
            2,
            3,
        ],
    )

    colorbar.ax.set_yticklabels(
        [
            "Discordant",
            "Sign concordant",
            "Primary direction retained",
            "Bootstrap replicated",
        ]
    )

    for row_index in range(
        matrix.shape[
            0
        ]
    ):
        for column_index in range(
            matrix.shape[
                1
            ]
        ):
            value = matrix[
                row_index,
                column_index,
            ]

            if np.isnan(
                value
            ):
                label = "NA"
            else:
                label = str(
                    int(
                        value
                    )
                )

            plt.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=7,
            )

    plt.tight_layout()

    plt.savefig(
        FIGURE_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        FIGURE_PDF,
        bbox_inches="tight",
    )

    plt.close()

    caption = (
        "Supplementary Figure. Replicate-concordant sensitivity analysis "
        "of quantitative LINCS Level 5 signatures. Signatures were retained "
        "when replicate-profile concordance was at least 0.20, at least "
        "three replicate profiles contributed to the Level 5 consensus, "
        "and invalid −666 concordance values were excluded. Signature-level "
        "scores were aggregated by taking the median within each cell and "
        "then assigning equal weight to each represented cell. Cells and "
        "signatures were resampled hierarchically over 5,000 bootstrap "
        "iterations. Matrix values indicate discordance (0), concordant "
        "score direction without primary absolute support retention (1), "
        "retention of a primary absolute-effect direction (2), or retention "
        "with bootstrap FDR below 0.05 and a 95% confidence interval "
        "excluding zero (3). NA indicates that no replicate-concordant "
        "score was available. This sensitivity analysis assesses profile "
        "reproducibility and does not replace the stratified gene-matched "
        "empirical-null analysis used for the primary quantitative results."
    )

    FIGURE_CAPTION.write_text(
        caption + "\n",
        encoding="utf-8",
    )

    chemicals_with_profiles = int(
        coverage[
            "replicate_concordant_signatures"
        ].gt(
            0
        ).sum()
    )

    chemicals_with_multicell_profiles = int(
        coverage[
            "bootstrap_inference_eligible"
        ].sum()
    )

    eligible_tests = int(
        bootstrap[
            "bootstrap_inference_eligible"
        ].sum()
    )

    bootstrap_supported_tests = int(
        bootstrap[
            "bootstrap_absolute_support"
        ].sum()
    )

    primary_absolute_tests = int(
        comparison[
            "primary_absolute_direction_supported"
        ].sum()
    )

    primary_direction_retained = int(
        comparison[
            "full_absolute_support_direction_retained"
        ].sum()
    )

    primary_bootstrap_replicated = int(
        comparison[
            "full_absolute_support_bootstrap_replicated"
        ].sum()
    )

    figure_created = (
        FIGURE_PNG.exists()
        and FIGURE_PDF.exists()
        and FIGURE_CAPTION.exists()
    )

    status = (
        "completed"
        if (
            len(
                comparison
            )
            == EXPECTED_COMPARISON_ROWS
            and chemicals_with_profiles >= 5
            and chemicals_with_multicell_profiles >= 3
            and eligible_tests > 0
            and figure_created
        )
        else "not_performed_insufficient_coverage"
    )

    completion = pd.DataFrame(
        [
            {
                "source_Level5_signatures":
                    len(
                        merged
                    ),

                "replicate_concordant_definition":
                    (
                        "distil_cc_q75>=0.20;"
                        "distil_nsample>=3;"
                        "exclude_distil_cc_q75_minus666"
                    ),

                "replicate_concordant_signatures":
                    len(
                        filtered
                    ),

                "replicate_concordant_fraction":
                    (
                        len(
                            filtered
                        )
                        / len(
                            merged
                        )
                    ),

                "chemicals_with_replicate_concordant_profiles":
                    chemicals_with_profiles,

                "chemicals_with_multicell_inference_eligible_profiles":
                    chemicals_with_multicell_profiles,

                "chemical_axis_results":
                    len(
                        comparison
                    ),

                "bootstrap_inference_eligible_tests":
                    eligible_tests,

                "bootstrap_FDR_supported_tests":
                    bootstrap_supported_tests,

                "primary_absolute_supported_tests":
                    primary_absolute_tests,

                "primary_absolute_direction_retained_tests":
                    primary_direction_retained,

                "primary_absolute_bootstrap_replicated_tests":
                    primary_bootstrap_replicated,

                "bootstrap_iterations":
                    BOOTSTRAP_ITERATIONS,

                "gold_definition_relaxed":
                    False,

                "gold_analysis_replaced":
                    False,

                "primary_gene_matched_empirical_null_replaced":
                    False,

                "supplementary_figure_created":
                    figure_created,

                "Phase6E3D3_status":
                    status,
            }
        ]
    )

    completion.to_csv(
        COMPLETION_FILE,
        sep="\t",
        index=False,
    )

    log_text = "\n".join(
        [
            "===== PHASE 6E3D3 COMPLETION =====",
            completion.to_string(
                index=False
            ),
            "",
            "===== SCORE COLUMN MAPPING =====",
            score_mapping.to_string(
                index=False
            ),
            "",
            "===== REPLICATE-CONCORDANT COVERAGE =====",
            coverage.to_string(
                index=False
            ),
            "",
            "===== AXIS CONCORDANCE =====",
            axis_concordance.to_string(
                index=False
            ),
            "",
            "===== CHEMICAL SENSITIVITY SUMMARY =====",
            chemical_summary.to_string(
                index=False
            ),
        ]
    )

    LOG_FILE.write_text(
        log_text + "\n",
        encoding="utf-8",
    )

    print(
        log_text
    )

    if status != "completed":
        print(
            (
                "\nPhase 6E3D3 did not meet the prespecified minimum "
                "coverage requirements. Outputs were retained as an "
                "insufficient-coverage audit."
            ),
            file=sys.stderr,
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            (
                "Phase 6E3D3 failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
