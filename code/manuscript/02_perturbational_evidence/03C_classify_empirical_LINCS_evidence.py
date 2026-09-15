#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(
    "."
)

RESULT_FILE = (
    PROJECT
    / "07_tables/main_tables/phase6/"
      "phase6E3B_empirical_calibration_results.tsv"
)

PHASE6D3_CHEMICAL_FILE = (
    PROJECT
    / "07_tables/main_tables/phase6/"
      "phase6D3_chemical_evidence_tiers.tsv"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase6"
)

FIGURE_DIR = (
    PROJECT
    / "08_figures/main_figures/phase6"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase6/"
      "phase6E3C_empirical_evidence_classification.log"
)

PRIMARY_OUTPUT = (
    TABLE_DIR
    / "phase6E3C_primary_biological_interpretation.tsv"
)

CROSS_SCOPE_OUTPUT = (
    TABLE_DIR
    / "phase6E3C_cross_scope_consistency.tsv"
)

CHEMICAL_OUTPUT = (
    TABLE_DIR
    / "phase6E3C_chemical_level_interpretation.tsv"
)

SUMMARY_OUTPUT = (
    TABLE_DIR
    / "phase6E3C_evidence_class_summary.tsv"
)

COMPLETION_OUTPUT = (
    TABLE_DIR
    / "phase6E3C_completion_summary.tsv"
)

FIGURE_PNG = (
    FIGURE_DIR
    / "Figure46_quantitative_LINCS_empirical_validation.png"
)

FIGURE_PDF = (
    FIGURE_DIR
    / "Figure46_quantitative_LINCS_empirical_validation.pdf"
)

FIGURE_CAPTION = (
    FIGURE_DIR
    / "Figure46_caption.txt"
)

for directory in [
    TABLE_DIR,
    FIGURE_DIR,
    LOG_FILE.parent,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


PRIMARY_SCOPE = "all_cells_cell_balanced"
SECONDARY_SCOPE = "all_signatures_median"
NEURAL_SCOPE = "neural_lineage_cell_balanced"

EXPECTED_CHEMICALS = 15

AXIS_ORDER = [
    "developmental_shift",
    "early_program_suppression",
    "late_maturation_support",
    "injury_stress",
    "late_minus_injury",
    "late_minus_early",
]

AXIS_LABELS = {
    "developmental_shift":
        "Developmental\nshift",

    "early_program_suppression":
        "Early-program\nsuppression",

    "late_maturation_support":
        "Late-maturation\nsupport",

    "injury_stress":
        "Injury/\nstress",

    "late_minus_injury":
        "Late minus\ninjury",

    "late_minus_early":
        "Late minus\nearly",
}


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


def observed_direction(
    value: float,
    tolerance: float = 1e-12,
) -> str:
    if value > tolerance:
        return "positive"

    if value < -tolerance:
        return "negative"

    return "zero"


def evidence_direction(
    row: pd.Series,
) -> tuple[str, str]:
    global_class = clean_text(
        row[
            "global_maxT_evidence_class"
        ]
    )

    fdr_class = clean_text(
        row[
            "directional_FDR_evidence_class"
        ]
    )

    if global_class == "positive_global_maxT_supported":
        return (
            "positive",
            "global_maxT",
        )

    if global_class == "negative_global_maxT_supported":
        return (
            "negative",
            "global_maxT",
        )

    if fdr_class == "positive_FDR_supported":
        return (
            "positive",
            "FDR_only",
        )

    if fdr_class == "negative_FDR_supported":
        return (
            "negative",
            "FDR_only",
        )

    return (
        "none",
        "none",
    )


def generic_evidence_class(
    observed: str,
    empirical: str,
    evidence_level: str,
) -> str:
    if empirical == "none":
        return "not_empirically_supported"

    if observed == empirical:
        return (
            f"absolute_{observed}_"
            f"{evidence_level}_supported"
        )

    if empirical == "positive" and observed == "negative":
        return (
            "above_null_but_observed_negative_"
            f"{evidence_level}"
        )

    if empirical == "negative" and observed == "positive":
        return (
            "below_null_but_observed_positive_"
            f"{evidence_level}"
        )

    return (
        "null_relative_support_without_"
        "absolute_direction"
    )


def biological_interpretation(
    axis_name: str,
    observed: str,
    empirical: str,
    evidence_level: str,
) -> str:
    if empirical == "none":
        return "not_empirically_supported"

    if observed != empirical:
        if empirical == "positive":
            return (
                "relative_preservation_or_above_null_"
                "without_absolute_activation"
            )

        return (
            "relative_depletion_or_below_null_"
            "without_absolute_suppression"
        )

    interpretation_map = {
        (
            "developmental_shift",
            "positive",
        ):
            "positive_developmental_state_shift",

        (
            "developmental_shift",
            "negative",
        ):
            "negative_developmental_state_shift",

        (
            "early_program_suppression",
            "positive",
        ):
            "early_program_suppression",

        (
            "early_program_suppression",
            "negative",
        ):
            "early_program_retention",

        (
            "late_maturation_support",
            "positive",
        ):
            "late_maturation_program_support",

        (
            "late_maturation_support",
            "negative",
        ):
            "late_maturation_program_depletion",

        (
            "injury_stress",
            "positive",
        ):
            "injury_stress_activation",

        (
            "injury_stress",
            "negative",
        ):
            "injury_stress_attenuation",

        (
            "late_minus_injury",
            "positive",
        ):
            "late_program_exceeds_injury",

        (
            "late_minus_injury",
            "negative",
        ):
            "injury_exceeds_late_program",

        (
            "late_minus_early",
            "positive",
        ):
            "late_program_exceeds_early_suppression",

        (
            "late_minus_early",
            "negative",
        ):
            "early_component_exceeds_late_program",
    }

    base = interpretation_map.get(
        (
            axis_name,
            observed,
        ),
        "directionally_supported_response",
    )

    return (
        f"{base}_{evidence_level}"
    )


def assign_overall_chemical_class(
    axis_rows: pd.DataFrame,
) -> str:
    lookup = {
        row[
            "axis_name"
        ]: row
        for _, row in axis_rows.iterrows()
    }

    def interpretation(
        axis: str,
    ) -> str:
        if axis not in lookup:
            return ""

        return clean_text(
            lookup[
                axis
            ][
                "biological_interpretation"
            ]
        )

    injury = interpretation(
        "injury_stress"
    )

    late = interpretation(
        "late_maturation_support"
    )

    late_minus_injury = interpretation(
        "late_minus_injury"
    )

    developmental = interpretation(
        "developmental_shift"
    )

    if (
        injury.startswith(
            "injury_stress_activation"
        )
        and late_minus_injury.startswith(
            "injury_exceeds_late_program"
        )
    ):
        return "injury_dominant_developmental_perturbation"

    if (
        late.startswith(
            "late_maturation_program_support"
        )
        and injury.startswith(
            "injury_stress_activation"
        )
    ):
        return "late_support_with_injury_confounding"

    if (
        late.startswith(
            "late_maturation_program_support"
        )
        and not injury.startswith(
            "injury_stress_activation"
        )
    ):
        return "late_program_support_without_supported_injury"

    if late.startswith(
        "relative_preservation"
    ):
        return "relative_late_program_preservation_only"

    if late.startswith(
        "late_maturation_program_depletion"
    ):
        return "late_program_depletion"

    if developmental.startswith(
        "positive_developmental_state_shift"
    ):
        return "developmental_shift_without_coherent_late_support"

    if developmental.startswith(
        "negative_developmental_state_shift"
    ):
        return "negative_developmental_state_shift"

    return "no_coherent_empirically_supported_pattern"


def main() -> None:
    required_files = [
        RESULT_FILE,
        PHASE6D3_CHEMICAL_FILE,
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(
                missing
            )
        )

    results = pd.read_csv(
        RESULT_FILE,
        sep="\t",
        low_memory=False,
    )

    phase6d3 = pd.read_csv(
        PHASE6D3_CHEMICAL_FILE,
        sep="\t",
        low_memory=False,
    )

    required_columns = {
        "analysis_scope",
        "chemical_query",
        "evidence_tier",
        "axis_name",
        "observed_score",
        "empirical_z",
        "directional_FDR_evidence_class",
        "global_maxT_evidence_class",
    }

    missing_columns = (
        required_columns
        - set(
            results.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Missing result columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    results = results.copy()

    empirical_directions = []
    evidence_levels = []

    for _, row in results.iterrows():
        empirical_direction, evidence_level = (
            evidence_direction(
                row
            )
        )

        empirical_directions.append(
            empirical_direction
        )

        evidence_levels.append(
            evidence_level
        )

    results[
        "observed_direction"
    ] = results[
        "observed_score"
    ].map(
        observed_direction
    )

    results[
        "empirical_direction"
    ] = empirical_directions

    results[
        "evidence_level"
    ] = evidence_levels

    results[
        "generic_evidence_class"
    ] = [
        generic_evidence_class(
            observed,
            empirical,
            level,
        )
        for observed, empirical, level
        in zip(
            results[
                "observed_direction"
            ],
            results[
                "empirical_direction"
            ],
            results[
                "evidence_level"
            ],
        )
    ]

    results[
        "biological_interpretation"
    ] = [
        biological_interpretation(
            axis,
            observed,
            empirical,
            level,
        )
        for axis, observed, empirical, level
        in zip(
            results[
                "axis_name"
            ],
            results[
                "observed_direction"
            ],
            results[
                "empirical_direction"
            ],
            results[
                "evidence_level"
            ],
        )
    ]

    results[
        "absolute_direction_supported"
    ] = (
        results[
            "empirical_direction"
        ].ne(
            "none"
        )
        & results[
            "observed_direction"
        ].eq(
            results[
                "empirical_direction"
            ]
        )
    )

    results[
        "null_relative_only"
    ] = (
        results[
            "empirical_direction"
        ].ne(
            "none"
        )
        & ~results[
            "absolute_direction_supported"
        ]
    )

    primary = results.loc[
        results[
            "analysis_scope"
        ].eq(
            PRIMARY_SCOPE
        )
    ].copy()

    if len(
        primary
    ) != (
        EXPECTED_CHEMICALS
        * len(
            AXIS_ORDER
        )
    ):
        raise RuntimeError(
            (
                "Expected 90 primary chemical-axis results, "
                f"found {len(primary)}."
            )
        )

    if primary[
        "chemical_query"
    ].nunique() != EXPECTED_CHEMICALS:
        raise RuntimeError(
            "Expected 15 primary chemicals."
        )

    if set(
        primary[
            "axis_name"
        ]
    ) != set(
        AXIS_ORDER
    ):
        raise RuntimeError(
            "Unexpected primary axis set."
        )

    primary = primary.sort_values(
        [
            "chemical_query",
            "axis_name",
        ]
    )

    primary.to_csv(
        PRIMARY_OUTPUT,
        sep="\t",
        index=False,
    )

    secondary = results.loc[
        results[
            "analysis_scope"
        ].eq(
            SECONDARY_SCOPE
        )
    ].copy()

    neural = results.loc[
        results[
            "analysis_scope"
        ].eq(
            NEURAL_SCOPE
        )
    ].copy()

    primary_join = primary[
        [
            "chemical_query",
            "axis_name",
            "observed_score",
            "observed_direction",
            "empirical_direction",
            "evidence_level",
            "absolute_direction_supported",
            "biological_interpretation",
        ]
    ].rename(
        columns={
            column:
                f"primary_{column}"
            for column in [
                "observed_score",
                "observed_direction",
                "empirical_direction",
                "evidence_level",
                "absolute_direction_supported",
                "biological_interpretation",
            ]
        }
    )

    secondary_join = secondary[
        [
            "chemical_query",
            "axis_name",
            "observed_score",
            "observed_direction",
            "empirical_direction",
            "evidence_level",
            "absolute_direction_supported",
            "biological_interpretation",
        ]
    ].rename(
        columns={
            column:
                f"secondary_{column}"
            for column in [
                "observed_score",
                "observed_direction",
                "empirical_direction",
                "evidence_level",
                "absolute_direction_supported",
                "biological_interpretation",
            ]
        }
    )

    cross_scope = primary_join.merge(
        secondary_join,
        on=[
            "chemical_query",
            "axis_name",
        ],
        how="left",
        validate="one_to_one",
    )

    cross_scope[
        "observed_sign_concordant"
    ] = (
        cross_scope[
            "primary_observed_direction"
        ]
        == cross_scope[
            "secondary_observed_direction"
        ]
    )

    cross_scope[
        "empirical_direction_concordant"
    ] = (
        cross_scope[
            "primary_empirical_direction"
        ]
        == cross_scope[
            "secondary_empirical_direction"
        ]
    )

    cross_scope[
        "absolute_support_replicated"
    ] = (
        cross_scope[
            "primary_absolute_direction_supported"
        ].fillna(
            False
        )
        & cross_scope[
            "secondary_absolute_direction_supported"
        ].fillna(
            False
        )
        & cross_scope[
            "primary_empirical_direction"
        ].eq(
            cross_scope[
                "secondary_empirical_direction"
            ]
        )
    )

    if not neural.empty:
        neural_join = neural[
            [
                "chemical_query",
                "axis_name",
                "observed_score",
                "observed_direction",
                "empirical_direction",
                "evidence_level",
                "absolute_direction_supported",
                "biological_interpretation",
            ]
        ].rename(
            columns={
                column:
                    f"neural_{column}"
                for column in [
                    "observed_score",
                    "observed_direction",
                    "empirical_direction",
                    "evidence_level",
                    "absolute_direction_supported",
                    "biological_interpretation",
                ]
            }
        )

        cross_scope = cross_scope.merge(
            neural_join,
            on=[
                "chemical_query",
                "axis_name",
            ],
            how="left",
            validate="one_to_one",
        )

    cross_scope.to_csv(
        CROSS_SCOPE_OUTPUT,
        sep="\t",
        index=False,
    )

    summary = (
        primary.groupby(
            [
                "axis_name",
                "evidence_level",
                "generic_evidence_class",
                "biological_interpretation",
            ],
            dropna=False,
            sort=True,
        )
        .size()
        .reset_index(
            name="primary_test_count"
        )
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        sep="\t",
        index=False,
    )

    chemical_rows = []

    for chemical, group in primary.groupby(
        "chemical_query",
        sort=True,
    ):
        axis_lookup = {
            row[
                "axis_name"
            ]: row
            for _, row in group.iterrows()
        }

        row = {
            "chemical_query":
                chemical,

            "evidence_tier":
                clean_text(
                    group[
                        "evidence_tier"
                    ].iloc[0]
                ),

            "primary_global_maxT_supported_axes":
                int(
                    group[
                        "evidence_level"
                    ].eq(
                        "global_maxT"
                    ).sum()
                ),

            "primary_FDR_only_supported_axes":
                int(
                    group[
                        "evidence_level"
                    ].eq(
                        "FDR_only"
                    ).sum()
                ),

            "primary_absolute_direction_supported_axes":
                int(
                    group[
                        "absolute_direction_supported"
                    ].sum()
                ),

            "primary_null_relative_only_axes":
                int(
                    group[
                        "null_relative_only"
                    ].sum()
                ),

            "overall_primary_interpretation":
                assign_overall_chemical_class(
                    group
                ),
        }

        for axis in AXIS_ORDER:
            axis_row = axis_lookup[
                axis
            ]

            row[
                f"{axis}_observed_score"
            ] = axis_row[
                "observed_score"
            ]

            row[
                f"{axis}_empirical_z"
            ] = axis_row[
                "empirical_z"
            ]

            row[
                f"{axis}_interpretation"
            ] = axis_row[
                "biological_interpretation"
            ]

        chemical_rows.append(
            row
        )

    chemical_summary = pd.DataFrame(
        chemical_rows
    )

    replication_summary = (
        cross_scope.groupby(
            "chemical_query",
            sort=True,
        )[
            "absolute_support_replicated"
        ]
        .sum()
        .reset_index(
            name=(
                "primary_secondary_"
                "replicated_absolute_axes"
            )
        )
    )

    chemical_summary = chemical_summary.merge(
        replication_summary,
        on="chemical_query",
        how="left",
        validate="one_to_one",
    )

    chemical_summary.to_csv(
        CHEMICAL_OUTPUT,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------
    # Figure 46
    # ------------------------------------------------------------

    heatmap = primary.pivot(
        index="chemical_query",
        columns="axis_name",
        values="empirical_z",
    ).reindex(
        columns=AXIS_ORDER
    )

    evidence_table = primary.pivot(
        index="chemical_query",
        columns="axis_name",
        values="evidence_level",
    ).reindex(
        index=heatmap.index,
        columns=AXIS_ORDER,
    )

    absolute_table = primary.pivot(
        index="chemical_query",
        columns="axis_name",
        values="absolute_direction_supported",
    ).reindex(
        index=heatmap.index,
        columns=AXIS_ORDER,
    )

    observed_table = primary.pivot(
        index="chemical_query",
        columns="axis_name",
        values="observed_score",
    ).reindex(
        index=heatmap.index,
        columns=AXIS_ORDER,
    )

    sorting = pd.DataFrame(
        {
            "injury_z":
                heatmap[
                    "injury_stress"
                ],

            "late_minus_injury_z":
                heatmap[
                    "late_minus_injury"
                ],
        }
    ).sort_values(
        [
            "injury_z",
            "late_minus_injury_z",
        ],
        ascending=[
            False,
            True,
        ],
    )

    chemical_order = sorting.index.tolist()

    heatmap = heatmap.reindex(
        chemical_order
    )

    evidence_table = evidence_table.reindex(
        chemical_order
    )

    absolute_table = absolute_table.reindex(
        chemical_order
    )

    observed_table = observed_table.reindex(
        chemical_order
    )

    values = heatmap.to_numpy(
        dtype=float
    )

    finite_max = float(
        np.nanmax(
            np.abs(
                values
            )
        )
    )

    color_limit = max(
        3.0,
        min(
            8.0,
            finite_max,
        ),
    )

    figure = plt.figure(
        figsize=(
            16,
            12,
        )
    )

    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=[
            1.65,
            1.0,
        ],
        wspace=0.35,
    )

    axis_heatmap = figure.add_subplot(
        grid[
            0,
            0,
        ]
    )

    image = axis_heatmap.imshow(
        np.clip(
            values,
            -color_limit,
            color_limit,
        ),
        aspect="auto",
        vmin=-color_limit,
        vmax=color_limit,
        cmap="coolwarm",
    )

    axis_heatmap.set_xticks(
        np.arange(
            len(
                AXIS_ORDER
            )
        )
    )

    axis_heatmap.set_xticklabels(
        [
            AXIS_LABELS[
                axis
            ]
            for axis in AXIS_ORDER
        ],
        rotation=35,
        ha="right",
    )

    axis_heatmap.set_yticks(
        np.arange(
            len(
                chemical_order
            )
        )
    )

    axis_heatmap.set_yticklabels(
        chemical_order
    )

    axis_heatmap.set_title(
        "A. Empirical z-scores in cell-balanced LINCS profiles"
    )

    for row_index, chemical in enumerate(
        chemical_order
    ):
        for column_index, axis in enumerate(
            AXIS_ORDER
        ):
            evidence_level = evidence_table.loc[
                chemical,
                axis,
            ]

            absolute = bool(
                absolute_table.loc[
                    chemical,
                    axis,
                ]
            )

            observed = float(
                observed_table.loc[
                    chemical,
                    axis,
                ]
            )

            if evidence_level == "global_maxT":
                symbol = (
                    "G"
                    if absolute
                    else "g"
                )

            elif evidence_level == "FDR_only":
                symbol = (
                    "F"
                    if absolute
                    else "f"
                )

            else:
                symbol = ""

            if symbol:
                axis_heatmap.text(
                    column_index,
                    row_index,
                    symbol,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )

            if (
                evidence_level != "none"
                and np.sign(
                    observed
                )
                != np.sign(
                    values[
                        row_index,
                        column_index,
                    ]
                )
            ):
                axis_heatmap.text(
                    column_index + 0.28,
                    row_index - 0.25,
                    "†",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    colorbar = figure.colorbar(
        image,
        ax=axis_heatmap,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Empirical z-score"
    )

    axis_scatter = figure.add_subplot(
        grid[
            0,
            1,
        ]
    )

    scatter_data = chemical_summary.copy()

    x = scatter_data[
        "late_maturation_support_observed_score"
    ].to_numpy(
        dtype=float
    )

    y = scatter_data[
        "injury_stress_observed_score"
    ].to_numpy(
        dtype=float
    )

    sizes = (
        55
        + 35
        * scatter_data[
            "primary_global_maxT_supported_axes"
        ].to_numpy(
            dtype=float
        )
    )

    axis_scatter.scatter(
        x,
        y,
        s=sizes,
        alpha=0.8,
    )

    axis_scatter.axhline(
        0,
        linewidth=1,
    )

    axis_scatter.axvline(
        0,
        linewidth=1,
    )

    for _, row in scatter_data.iterrows():
        axis_scatter.annotate(
            row[
                "chemical_query"
            ],
            (
                row[
                    "late_maturation_support_observed_score"
                ],
                row[
                    "injury_stress_observed_score"
                ],
            ),
            xytext=(
                4,
                4,
            ),
            textcoords="offset points",
            fontsize=8,
        )

    axis_scatter.set_xlabel(
        "Observed late-maturation support score"
    )

    axis_scatter.set_ylabel(
        "Observed injury/stress score"
    )

    axis_scatter.set_title(
        "B. Absolute late-program and injury responses"
    )

    figure.text(
        0.01,
        0.01,
        (
            "G/F: absolute-direction global maxT/FDR support; "
            "g/f: null-relative support without matching absolute direction; "
            "†: empirical direction differs from the absolute-score sign."
        ),
        fontsize=9,
    )

    figure.savefig(
        FIGURE_PNG,
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_PDF,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    caption = (
        "Figure 46. Quantitative LINCS validation of developmental-state "
        "and mechanistic responses. (A) Empirical z-scores from 5,000 "
        "landmark-, mean-response- and variability-stratified gene-set "
        "permutations applied to cell-balanced chemical profiles. Uppercase "
        "G and F denote global maxT and FDR support whose empirical direction "
        "agreed with the sign of the observed biological score. Lowercase g "
        "and f denote statistically supported displacement from the matched "
        "null without absolute activation or suppression in the corresponding "
        "direction. (B) Absolute leave-program-out late-maturation and "
        "injury/stress scores. Point size reflects the number of globally "
        "supported axes. Positive developmental shifts were interpreted as "
        "maturation-related only when accompanied by positive absolute "
        "late-program support and not dominated by injury/stress."
    )

    FIGURE_CAPTION.write_text(
        caption + "\n",
        encoding="utf-8",
    )

    completion = pd.DataFrame(
        [
            {
                "primary_chemical_axis_results":
                    len(
                        primary
                    ),

                "primary_chemicals":
                    primary[
                        "chemical_query"
                    ].nunique(),

                "axes_classified":
                    primary[
                        "axis_name"
                    ].nunique(),

                "primary_global_maxT_supported_tests":
                    int(
                        primary[
                            "evidence_level"
                        ].eq(
                            "global_maxT"
                        ).sum()
                    ),

                "primary_FDR_only_supported_tests":
                    int(
                        primary[
                            "evidence_level"
                        ].eq(
                            "FDR_only"
                        ).sum()
                    ),

                "primary_absolute_direction_supported_tests":
                    int(
                        primary[
                            "absolute_direction_supported"
                        ].sum()
                    ),

                "primary_null_relative_only_tests":
                    int(
                        primary[
                            "null_relative_only"
                        ].sum()
                    ),

                "primary_secondary_cross_scope_tests":
                    len(
                        cross_scope
                    ),

                "primary_secondary_replicated_absolute_tests":
                    int(
                        cross_scope[
                            "absolute_support_replicated"
                        ].sum()
                    ),

                "neural_sensitivity_chemicals":
                    neural[
                        "chemical_query"
                    ].nunique(),

                "figure46_png_created":
                    FIGURE_PNG.exists(),

                "figure46_pdf_created":
                    FIGURE_PDF.exists(),

                "figure46_caption_created":
                    FIGURE_CAPTION.exists(),

                "Phase6E3C_status":
                    "completed",
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
            "===== PHASE 6E3C COMPLETION =====",
            completion.to_string(
                index=False
            ),
            "",
            "===== CHEMICAL-LEVEL INTERPRETATION =====",
            chemical_summary[
                [
                    "chemical_query",
                    "evidence_tier",
                    "primary_global_maxT_supported_axes",
                    "primary_FDR_only_supported_axes",
                    "primary_absolute_direction_supported_axes",
                    "primary_null_relative_only_axes",
                    "primary_secondary_replicated_absolute_axes",
                    "overall_primary_interpretation",
                ]
            ].to_string(
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


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            (
                "Phase 6E3C failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
