#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT = Path(
    "."
)

INPUT_FILE = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6E/"
      "phase6E2C_matched_signature_metadata.tsv.gz"
)

PROCESSED_DIR = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6E"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase6"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase6/"
      "phase6E3D1_reconstruct_LINCS_gold_QC.log"
)

FLAGGED_OUTPUT = (
    PROCESSED_DIR
    / "phase6E3D1_signature_quality_flags.tsv.gz"
)

CHEMICAL_OUTPUT = (
    TABLE_DIR
    / "phase6E3D1_quality_summary_by_chemical.tsv"
)

METRIC_OUTPUT = (
    TABLE_DIR
    / "phase6E3D1_quality_metric_distribution.tsv"
)

COMPLETION_OUTPUT = (
    TABLE_DIR
    / "phase6E3D1_completion_summary.tsv"
)

for directory in [
    PROCESSED_DIR,
    TABLE_DIR,
    LOG_FILE.parent,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


EXPECTED_SIGNATURES = 1232
EXPECTED_CHEMICALS = 15

INVALID_SENTINEL = -666.0

GOLD_CC_THRESHOLD = 0.20
GOLD_SELF_RANK_THRESHOLD = 0.05


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


def find_column(
    table: pd.DataFrame,
    candidates: list[str],
) -> str:
    lookup = {
        str(column).lower(): column
        for column in table.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[
                candidate.lower()
            ]

    raise RuntimeError(
        "Could not identify any expected column: "
        + ", ".join(
            candidates
        )
    )


def classify_cell(
    value: object,
) -> str:
    cell = clean_text(
        value
    ).upper()

    neural_cells = {
        "NPC",
        "NPC.CAS9",
        "NPC.TAK",
        "NEU",
        "MNEU.E",
    }

    if cell in neural_cells:
        return "neural_lineage"

    return "non_neural"


def safe_median(
    values: pd.Series,
) -> float:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return np.nan

    return float(
        numeric.median()
    )


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            str(
                INPUT_FILE
            )
        )

    metadata = pd.read_csv(
        INPUT_FILE,
        sep="\t",
        low_memory=False,
    ).reset_index(
        drop=True
    )

    if len(metadata) != EXPECTED_SIGNATURES:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_SIGNATURES} signatures, "
                f"found {len(metadata)}."
            )
        )

    chemical_column = find_column(
        metadata,
        [
            "chemical_query",
            "pert_iname",
        ],
    )

    cell_column = find_column(
        metadata,
        [
            "cell_id",
            "cell",
        ],
    )

    required_metric_columns = [
        "distil_cc_q75",
        "distil_ss",
        "pct_self_rank_q25",
        "distil_nsample",
    ]

    missing_metrics = [
        column
        for column in required_metric_columns
        if column not in metadata.columns
    ]

    if missing_metrics:
        raise RuntimeError(
            "Missing required QC columns: "
            + ", ".join(
                missing_metrics
            )
        )

    output = metadata.copy()

    output[
        "chemical_query_QC"
    ] = output[
        chemical_column
    ].map(
        clean_text
    )

    output[
        "cell_id_QC"
    ] = output[
        cell_column
    ].map(
        clean_text
    )

    output[
        "cell_class_QC"
    ] = output[
        "cell_id_QC"
    ].map(
        classify_cell
    )

    for column in required_metric_columns:
        output[
            column
        ] = pd.to_numeric(
            output[
                column
            ],
            errors="coerce",
        )

    output[
        "distil_cc_q75_valid"
    ] = (
        output[
            "distil_cc_q75"
        ].notna()
        & output[
            "distil_cc_q75"
        ].ne(
            INVALID_SENTINEL
        )
    )

    output[
        "pct_self_rank_q25_valid"
    ] = (
        output[
            "pct_self_rank_q25"
        ].notna()
        & output[
            "pct_self_rank_q25"
        ].ne(
            INVALID_SENTINEL
        )
    )

    output[
        "distil_ss_valid"
    ] = (
        output[
            "distil_ss"
        ].notna()
        & output[
            "distil_ss"
        ].ge(
            0
        )
    )

    output[
        "distil_nsample_valid"
    ] = (
        output[
            "distil_nsample"
        ].notna()
        & output[
            "distil_nsample"
        ].ge(
            1
        )
    )

    output[
        "gold_metrics_valid"
    ] = (
        output[
            "distil_cc_q75_valid"
        ]
        & output[
            "pct_self_rank_q25_valid"
        ]
    )

    output[
        "is_gold_reconstructed"
    ] = (
        output[
            "gold_metrics_valid"
        ]
        & output[
            "distil_cc_q75"
        ].ge(
            GOLD_CC_THRESHOLD
        )
        & output[
            "pct_self_rank_q25"
        ].le(
            GOLD_SELF_RANK_THRESHOLD
        )
    )

    output[
        "gold_failure_reason"
    ] = np.select(
        [
            ~output[
                "gold_metrics_valid"
            ],

            output[
                "distil_cc_q75"
            ].lt(
                GOLD_CC_THRESHOLD
            ),

            output[
                "pct_self_rank_q25"
            ].gt(
                GOLD_SELF_RANK_THRESHOLD
            ),
        ],
        [
            "invalid_or_missing_gold_metrics",
            "replicate_correlation_below_0.20",
            "self_rank_above_0.05",
        ],
        default="passes_reconstructed_is_gold",
    )

    invalid_selected = output.loc[
        output[
            "is_gold_reconstructed"
        ]
        & (
            output[
                "distil_cc_q75"
            ].eq(
                INVALID_SENTINEL
            )
            | output[
                "pct_self_rank_q25"
            ].eq(
                INVALID_SENTINEL
            )
        )
    ]

    if not invalid_selected.empty:
        raise RuntimeError(
            "Invalid −666 sentinel values entered the gold subset."
        )

    output.to_csv(
        FLAGGED_OUTPUT,
        sep="\t",
        index=False,
        compression="gzip",
    )

    metric_rows: list[
        dict[str, object]
    ] = []

    for metric in required_metric_columns:
        numeric = pd.to_numeric(
            output[
                metric
            ],
            errors="coerce",
        )

        if metric in {
            "distil_cc_q75",
            "pct_self_rank_q25",
        }:
            valid = numeric.loc[
                numeric.ne(
                    INVALID_SENTINEL
                )
                & numeric.notna()
            ]
        else:
            valid = numeric.dropna()

        metric_rows.append(
            {
                "metric":
                    metric,

                "total_signature_count":
                    len(
                        output
                    ),

                "valid_value_count":
                    len(
                        valid
                    ),

                "invalid_or_missing_count":
                    len(
                        output
                    )
                    - len(
                        valid
                    ),

                "minimum_valid":
                    (
                        float(
                            valid.min()
                        )
                        if not valid.empty
                        else np.nan
                    ),

                "q05_valid":
                    (
                        float(
                            valid.quantile(
                                0.05
                            )
                        )
                        if not valid.empty
                        else np.nan
                    ),

                "q25_valid":
                    (
                        float(
                            valid.quantile(
                                0.25
                            )
                        )
                        if not valid.empty
                        else np.nan
                    ),

                "median_valid":
                    (
                        float(
                            valid.median()
                        )
                        if not valid.empty
                        else np.nan
                    ),

                "q75_valid":
                    (
                        float(
                            valid.quantile(
                                0.75
                            )
                        )
                        if not valid.empty
                        else np.nan
                    ),

                "q95_valid":
                    (
                        float(
                            valid.quantile(
                                0.95
                            )
                        )
                        if not valid.empty
                        else np.nan
                    ),

                "maximum_valid":
                    (
                        float(
                            valid.max()
                        )
                        if not valid.empty
                        else np.nan
                    ),
            }
        )

    metric_summary = pd.DataFrame(
        metric_rows
    )

    metric_summary.to_csv(
        METRIC_OUTPUT,
        sep="\t",
        index=False,
    )

    chemical_rows: list[
        dict[str, object]
    ] = []

    for chemical, group in output.groupby(
        "chemical_query_QC",
        sort=True,
    ):
        gold = group.loc[
            group[
                "is_gold_reconstructed"
            ]
        ]

        valid = group.loc[
            group[
                "gold_metrics_valid"
            ]
        ]

        neural = group.loc[
            group[
                "cell_class_QC"
            ].eq(
                "neural_lineage"
            )
        ]

        neural_gold = neural.loc[
            neural[
                "is_gold_reconstructed"
            ]
        ]

        chemical_rows.append(
            {
                "chemical_query":
                    chemical,

                "total_signature_count":
                    len(
                        group
                    ),

                "valid_gold_metric_count":
                    len(
                        valid
                    ),

                "invalid_gold_metric_count":
                    len(
                        group
                    )
                    - len(
                        valid
                    ),

                "reconstructed_gold_signature_count":
                    len(
                        gold
                    ),

                "gold_fraction_of_all_signatures":
                    len(
                        gold
                    )
                    / len(
                        group
                    ),

                "gold_fraction_of_valid_signatures":
                    (
                        len(
                            gold
                        )
                        / len(
                            valid
                        )
                        if len(
                            valid
                        )
                        else np.nan
                    ),

                "unique_cells_all":
                    group[
                        "cell_id_QC"
                    ].nunique(),

                "unique_cells_gold":
                    gold[
                        "cell_id_QC"
                    ].nunique(),

                "neural_signature_count":
                    len(
                        neural
                    ),

                "neural_gold_signature_count":
                    len(
                        neural_gold
                    ),

                "neural_cells_all":
                    neural[
                        "cell_id_QC"
                    ].nunique(),

                "neural_cells_gold":
                    neural_gold[
                        "cell_id_QC"
                    ].nunique(),

                "median_distil_cc_q75_valid":
                    safe_median(
                        valid[
                            "distil_cc_q75"
                        ]
                    ),

                "median_distil_ss_all":
                    safe_median(
                        group[
                            "distil_ss"
                        ]
                    ),

                "median_distil_ss_gold":
                    safe_median(
                        gold[
                            "distil_ss"
                        ]
                    ),

                "median_pct_self_rank_q25_valid":
                    safe_median(
                        valid[
                            "pct_self_rank_q25"
                        ]
                    ),

                "median_distil_nsample_all":
                    safe_median(
                        group[
                            "distil_nsample"
                        ]
                    ),

                "median_distil_nsample_gold":
                    safe_median(
                        gold[
                            "distil_nsample"
                        ]
                    ),
            }
        )

    chemical_summary = pd.DataFrame(
        chemical_rows
    ).sort_values(
        [
            "reconstructed_gold_signature_count",
            "chemical_query",
        ],
        ascending=[
            False,
            True,
        ],
    )

    chemical_summary.to_csv(
        CHEMICAL_OUTPUT,
        sep="\t",
        index=False,
    )

    total_gold = int(
        output[
            "is_gold_reconstructed"
        ].sum()
    )

    valid_gold_metrics = int(
        output[
            "gold_metrics_valid"
        ].sum()
    )

    gold_chemicals = int(
        chemical_summary[
            "reconstructed_gold_signature_count"
        ].gt(
            0
        ).sum()
    )

    gold_multicell_chemicals = int(
        chemical_summary[
            "unique_cells_gold"
        ].ge(
            2
        ).sum()
    )

    neural_gold_signatures = int(
        (
            output[
                "is_gold_reconstructed"
            ]
            & output[
                "cell_class_QC"
            ].eq(
                "neural_lineage"
            )
        ).sum()
    )

    neural_gold_chemicals = int(
        chemical_summary[
            "neural_gold_signature_count"
        ].gt(
            0
        ).sum()
    )

    status = (
        "completed"
        if (
            len(
                output
            )
            == EXPECTED_SIGNATURES
            and output[
                "chemical_query_QC"
            ].nunique()
            == EXPECTED_CHEMICALS
            and len(
                invalid_selected
            )
            == 0
        )
        else "failed_validation"
    )

    completion = pd.DataFrame(
        [
            {
                "source_signatures":
                    len(
                        output
                    ),

                "source_chemicals":
                    output[
                        "chemical_query_QC"
                    ].nunique(),

                "gold_definition":
                    (
                        "distil_cc_q75>=0.20_and_"
                        "pct_self_rank_q25<=0.05"
                    ),

                "invalid_sentinel":
                    INVALID_SENTINEL,

                "signatures_with_valid_gold_metrics":
                    valid_gold_metrics,

                "signatures_with_invalid_gold_metrics":
                    len(
                        output
                    )
                    - valid_gold_metrics,

                "reconstructed_gold_signatures":
                    total_gold,

                "reconstructed_gold_fraction":
                    total_gold
                    / len(
                        output
                    ),

                "chemicals_with_gold_signatures":
                    gold_chemicals,

                "chemicals_with_gold_signatures_in_at_least_2_cells":
                    gold_multicell_chemicals,

                "neural_gold_signatures":
                    neural_gold_signatures,

                "chemicals_with_neural_gold_signatures":
                    neural_gold_chemicals,

                "invalid_sentinel_selected_as_gold":
                    len(
                        invalid_selected
                    ),

                "Phase6E3D1_status":
                    status,
            }
        ]
    )

    completion.to_csv(
        COMPLETION_OUTPUT,
        sep="\t",
        index=False,
    )

    log_text = "\n".join(
        [
            "===== PHASE 6E3D1 COMPLETION =====",
            completion.to_string(
                index=False
            ),
            "",
            "===== QUALITY SUMMARY BY CHEMICAL =====",
            chemical_summary.to_string(
                index=False
            ),
            "",
            "===== VALID QUALITY-METRIC DISTRIBUTIONS =====",
            metric_summary.to_string(
                index=False
            ),
            "",
            "===== GOLD FAILURE REASONS =====",
            output[
                "gold_failure_reason"
            ].value_counts(
                dropna=False
            ).to_string(),
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
        raise RuntimeError(
            "Phase 6E3D1 validation failed."
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            (
                "Phase 6E3D1 failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
