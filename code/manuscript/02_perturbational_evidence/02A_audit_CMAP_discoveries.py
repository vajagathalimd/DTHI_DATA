#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT = Path(
    "."
)

CMAP_FILE = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6C/"
      "phase6C1_CMAP_directional_concordance.tsv.gz"
)

REFERENCE_SUMMARY_FILE = (
    PROJECT
    / "07_tables/main_tables/phase6/"
      "phase6C1_reference_enrichment_source_summary.tsv"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase6"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase6/"
      "phase6C2A_CMAP_discovery_diagnostic.log"
)

SUMMARY_FILE = (
    TABLE_DIR
    / "phase6C2A_CMAP_discovery_diagnostic_summary.tsv"
)

EFFECT_THRESHOLD_FILE = (
    TABLE_DIR
    / "phase6C2A_CMAP_effect_threshold_summary.tsv"
)

QUANTILE_FILE = (
    TABLE_DIR
    / "phase6C2A_CMAP_score_quantiles.tsv"
)

CHEMICAL_MULTIPLICITY_FILE = (
    TABLE_DIR
    / "phase6C2A_CMAP_chemical_multiplicity.tsv"
)

TOP_CANDIDATE_FILE = (
    TABLE_DIR
    / "phase6C2A_CMAP_top_directional_candidates.tsv"
)

COMPLETION_FILE = (
    TABLE_DIR
    / "phase6C2A_completion_summary.tsv"
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


def normalize_text(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )


def write_log(
    lines: list[str],
) -> None:
    text = "\n".join(
        lines
    ) + "\n"

    LOG_FILE.write_text(
        text,
        encoding="utf-8",
    )

    print(
        text,
        end="",
    )


def count_nonempty_unique(
    series: pd.Series,
) -> int:
    values = normalize_text(
        series
    )

    values = values.loc[
        values.ne("")
        & values.str.lower().ne(
            "nan"
        )
    ]

    return int(
        values.nunique()
    )


def safe_quantiles(
    series: pd.Series,
    probabilities: list[float],
) -> pd.Series:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    return values.quantile(
        probabilities
    )


def main() -> None:
    if not CMAP_FILE.exists():
        raise FileNotFoundError(
            f"Missing CMAP result file: {CMAP_FILE}"
        )

    cmap = pd.read_csv(
        CMAP_FILE,
        sep="\t",
        low_memory=False,
    )

    required_columns = {
        "CMAP_parent",
        "chemical_name",
        "DTXSID",
        "up_signature_size",
        "down_signature_size",
        "maturation_genes_up",
        "maturation_genes_down",
        "fetal_genes_up",
        "fetal_genes_down",
        "maturation_up_preference_log2OR",
        "fetal_down_preference_log2OR",
        "developmental_direction_score",
        "maturation_up_preference_p",
        "fetal_down_preference_p",
        "fetal_up_preference_p",
        "maturation_down_preference_p",
        "maturation_like_FDR",
        "fetal_like_FDR",
        "directional_state",
    }

    missing_columns = sorted(
        required_columns
        - set(
            cmap.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Missing CMAP columns: "
            + ", ".join(
                missing_columns
            )
        )

    numeric_columns = [
        "up_signature_size",
        "down_signature_size",
        "maturation_genes_up",
        "maturation_genes_down",
        "fetal_genes_up",
        "fetal_genes_down",
        "maturation_up_preference_log2OR",
        "fetal_down_preference_log2OR",
        "developmental_direction_score",
        "maturation_up_preference_p",
        "fetal_down_preference_p",
        "fetal_up_preference_p",
        "maturation_down_preference_p",
        "maturation_like_FDR",
        "fetal_like_FDR",
    ]

    for column in numeric_columns:
        cmap[column] = pd.to_numeric(
            cmap[column],
            errors="coerce",
        )

    cmap[
        "maturation_component_direction_consistent"
    ] = (
        cmap[
            "maturation_up_preference_log2OR"
        ].gt(0)
        & cmap[
            "fetal_down_preference_log2OR"
        ].gt(0)
    )

    cmap[
        "fetal_component_direction_consistent"
    ] = (
        cmap[
            "maturation_up_preference_log2OR"
        ].lt(0)
        & cmap[
            "fetal_down_preference_log2OR"
        ].lt(0)
    )

    cmap[
        "maturation_components_nominal_p05"
    ] = (
        cmap[
            "maturation_up_preference_p"
        ].lt(0.05)
        & cmap[
            "fetal_down_preference_p"
        ].lt(0.05)
    )

    cmap[
        "fetal_components_nominal_p05"
    ] = (
        cmap[
            "fetal_up_preference_p"
        ].lt(0.05)
        & cmap[
            "maturation_down_preference_p"
        ].lt(0.05)
    )

    maturation_fdr = cmap[
        "maturation_like_FDR"
    ].lt(0.05)

    fetal_fdr = cmap[
        "fetal_like_FDR"
    ].lt(0.05)

    maturation_consistent = cmap[
        "maturation_component_direction_consistent"
    ]

    fetal_consistent = cmap[
        "fetal_component_direction_consistent"
    ]

    maturation_both_nominal = cmap[
        "maturation_components_nominal_p05"
    ]

    fetal_both_nominal = cmap[
        "fetal_components_nominal_p05"
    ]

    chemical_names = normalize_text(
        cmap[
            "chemical_name"
        ]
    )

    dtxsids = normalize_text(
        cmap[
            "DTXSID"
        ]
    )

    valid_chemical_name = (
        chemical_names.ne("")
        & chemical_names.str.lower().ne(
            "nan"
        )
    )

    valid_dtxsid = (
        dtxsids.ne("")
        & dtxsids.str.lower().ne(
            "nan"
        )
    )

    summary = pd.DataFrame(
        [
            {
                "CMAP_parents":
                    len(
                        cmap
                    ),

                "unique_nonempty_chemical_names":
                    count_nonempty_unique(
                        cmap[
                            "chemical_name"
                        ]
                    ),

                "parents_with_nonempty_chemical_name":
                    int(
                        valid_chemical_name.sum()
                    ),

                "unique_nonempty_DTXSIDs":
                    count_nonempty_unique(
                        cmap[
                            "DTXSID"
                        ]
                    ),

                "parents_with_DTXSID":
                    int(
                        valid_dtxsid.sum()
                    ),

                "positive_direction_scores":
                    int(
                        cmap[
                            "developmental_direction_score"
                        ].gt(0).sum()
                    ),

                "negative_direction_scores":
                    int(
                        cmap[
                            "developmental_direction_score"
                        ].lt(0).sum()
                    ),

                "zero_direction_scores":
                    int(
                        cmap[
                            "developmental_direction_score"
                        ].eq(0).sum()
                    ),

                "maturation_like_FDR_lt_0_05":
                    int(
                        maturation_fdr.sum()
                    ),

                "fetal_like_FDR_lt_0_05":
                    int(
                        fetal_fdr.sum()
                    ),

                "maturation_FDR_and_component_direction_consistent":
                    int(
                        (
                            maturation_fdr
                            & maturation_consistent
                        ).sum()
                    ),

                "fetal_FDR_and_component_direction_consistent":
                    int(
                        (
                            fetal_fdr
                            & fetal_consistent
                        ).sum()
                    ),

                "maturation_FDR_and_both_components_nominal_p05":
                    int(
                        (
                            maturation_fdr
                            & maturation_both_nominal
                        ).sum()
                    ),

                "fetal_FDR_and_both_components_nominal_p05":
                    int(
                        (
                            fetal_fdr
                            & fetal_both_nominal
                        ).sum()
                    ),

                "median_up_signature_size":
                    float(
                        cmap[
                            "up_signature_size"
                        ].median()
                    ),

                "median_down_signature_size":
                    float(
                        cmap[
                            "down_signature_size"
                        ].median()
                    ),

                "Phase6C2A_interpretation":
                    (
                        "diagnostic_only_pending_empirical_null"
                    ),
            }
        ]
    )

    summary.to_csv(
        SUMMARY_FILE,
        sep="\t",
        index=False,
    )

    effect_rows = []

    for threshold in [
        0.00,
        0.25,
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
    ]:
        maturation_effect = cmap[
            "developmental_direction_score"
        ].ge(
            threshold
        )

        fetal_effect = cmap[
            "developmental_direction_score"
        ].le(
            -threshold
        )

        effect_rows.append(
            {
                "absolute_direction_score_threshold":
                    threshold,

                "maturation_direction_count":
                    int(
                        maturation_effect.sum()
                    ),

                "fetal_direction_count":
                    int(
                        fetal_effect.sum()
                    ),

                "maturation_FDR_and_effect":
                    int(
                        (
                            maturation_fdr
                            & maturation_effect
                        ).sum()
                    ),

                "fetal_FDR_and_effect":
                    int(
                        (
                            fetal_fdr
                            & fetal_effect
                        ).sum()
                    ),

                "maturation_FDR_effect_and_component_consistency":
                    int(
                        (
                            maturation_fdr
                            & maturation_effect
                            & maturation_consistent
                        ).sum()
                    ),

                "fetal_FDR_effect_and_component_consistency":
                    int(
                        (
                            fetal_fdr
                            & fetal_effect
                            & fetal_consistent
                        ).sum()
                    ),

                "maturation_FDR_effect_and_both_component_p05":
                    int(
                        (
                            maturation_fdr
                            & maturation_effect
                            & maturation_both_nominal
                        ).sum()
                    ),

                "fetal_FDR_effect_and_both_component_p05":
                    int(
                        (
                            fetal_fdr
                            & fetal_effect
                            & fetal_both_nominal
                        ).sum()
                    ),
            }
        )

    effect_table = pd.DataFrame(
        effect_rows
    )

    effect_table.to_csv(
        EFFECT_THRESHOLD_FILE,
        sep="\t",
        index=False,
    )

    quantile_rows = []

    probabilities = [
        0.00,
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00,
    ]

    for variable in [
        "developmental_direction_score",
        "maturation_up_preference_log2OR",
        "fetal_down_preference_log2OR",
        "maturation_like_FDR",
        "fetal_like_FDR",
    ]:
        quantiles = safe_quantiles(
            cmap[
                variable
            ],
            probabilities,
        )

        for probability, value in quantiles.items():
            quantile_rows.append(
                {
                    "variable":
                        variable,

                    "quantile":
                        probability,

                    "value":
                        value,
                }
            )

    quantile_table = pd.DataFrame(
        quantile_rows
    )

    quantile_table.to_csv(
        QUANTILE_FILE,
        sep="\t",
        index=False,
    )

    multiplicity_source = pd.DataFrame(
        {
            "chemical_name":
                chemical_names,

            "DTXSID":
                dtxsids,

            "maturation_FDR":
                maturation_fdr,

            "fetal_FDR":
                fetal_fdr,

            "direction_score":
                cmap[
                    "developmental_direction_score"
                ],
        }
    )

    multiplicity_source = multiplicity_source.loc[
        valid_chemical_name
    ].copy()

    multiplicity_rows = []

    for chemical_name, group in (
        multiplicity_source.groupby(
            "chemical_name",
            sort=True,
            dropna=False,
        )
    ):
        valid_group_dtxsid = group[
            "DTXSID"
        ].loc[
            group[
                "DTXSID"
            ].ne("")
            & group[
                "DTXSID"
            ].str.lower().ne(
                "nan"
            )
        ]

        multiplicity_rows.append(
            {
                "chemical_name":
                    chemical_name,

                "CMAP_parent_count":
                    len(
                        group
                    ),

                "unique_DTXSIDs":
                    int(
                        valid_group_dtxsid.nunique()
                    ),

                "DTXSIDs":
                    "|".join(
                        sorted(
                            valid_group_dtxsid.unique()
                        )
                    ),

                "maturation_like_FDR_parent_count":
                    int(
                        group[
                            "maturation_FDR"
                        ].sum()
                    ),

                "fetal_like_FDR_parent_count":
                    int(
                        group[
                            "fetal_FDR"
                        ].sum()
                    ),

                "minimum_direction_score":
                    group[
                        "direction_score"
                    ].min(),

                "median_direction_score":
                    group[
                        "direction_score"
                    ].median(),

                "maximum_direction_score":
                    group[
                        "direction_score"
                    ].max(),
            }
        )

    multiplicity = pd.DataFrame(
        multiplicity_rows
    )

    if not multiplicity.empty:
        multiplicity = multiplicity.sort_values(
            [
                "CMAP_parent_count",
                "chemical_name",
            ],
            ascending=[
                False,
                True,
            ],
        ).reset_index(
            drop=True
        )

    multiplicity.to_csv(
        CHEMICAL_MULTIPLICITY_FILE,
        sep="\t",
        index=False,
    )

    maturation_rank = cmap.loc[
        maturation_fdr
        & maturation_consistent
    ].copy()

    maturation_rank[
        "ranked_state"
    ] = "maturation_like"

    maturation_rank = maturation_rank.sort_values(
        [
            "maturation_like_FDR",
            "developmental_direction_score",
            "CMAP_parent",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).head(
        100
    )

    fetal_rank = cmap.loc[
        fetal_fdr
        & fetal_consistent
    ].copy()

    fetal_rank[
        "ranked_state"
    ] = "fetal_like"

    fetal_rank = fetal_rank.sort_values(
        [
            "fetal_like_FDR",
            "developmental_direction_score",
            "CMAP_parent",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    ).head(
        100
    )

    top_candidates = pd.concat(
        [
            maturation_rank,
            fetal_rank,
        ],
        ignore_index=True,
    )

    top_candidates[
        "state_rank"
    ] = (
        top_candidates.groupby(
            "ranked_state"
        ).cumcount()
        + 1
    )

    ordered_columns = [
        "ranked_state",
        "state_rank",
        "chemical_name",
        "DTXSID",
        "CMAP_parent",
        "description",
        "up_signature_size",
        "down_signature_size",
        "maturation_genes_up",
        "maturation_genes_down",
        "fetal_genes_up",
        "fetal_genes_down",
        "maturation_up_preference_log2OR",
        "fetal_down_preference_log2OR",
        "developmental_direction_score",
        "maturation_up_preference_p",
        "fetal_down_preference_p",
        "fetal_up_preference_p",
        "maturation_down_preference_p",
        "maturation_like_FDR",
        "fetal_like_FDR",
        "directional_state",
    ]

    top_candidates[
        ordered_columns
    ].to_csv(
        TOP_CANDIDATE_FILE,
        sep="\t",
        index=False,
    )

    completion = pd.DataFrame(
        [
            {
                "CMAP_parents_audited":
                    len(
                        cmap
                    ),

                "effect_thresholds_tested":
                    len(
                        effect_table
                    ),

                "unique_chemical_names":
                    count_nonempty_unique(
                        cmap[
                            "chemical_name"
                        ]
                    ),

                "top_candidates_exported":
                    len(
                        top_candidates
                    ),

                "empirical_null_completed":
                    False,

                "Phase6C2A_status":
                    "completed",
            }
        ]
    )

    completion.to_csv(
        COMPLETION_FILE,
        sep="\t",
        index=False,
    )

    log_lines = [
        "===== PHASE 6C2A DIAGNOSTIC SUMMARY =====",
        summary.to_string(
            index=False
        ),
        "",
        "===== EFFECT-THRESHOLD SUMMARY =====",
        effect_table.to_string(
            index=False
        ),
        "",
        "===== COMPLETION =====",
        completion.to_string(
            index=False
        ),
    ]

    write_log(
        log_lines
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            (
                "Phase 6C2A failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
