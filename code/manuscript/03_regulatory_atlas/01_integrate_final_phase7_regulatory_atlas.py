#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(
    "."
)

PHASE7E_ATLAS_FILE = (
    PROJECT
    / "07_tables/main_tables/phase7/"
      "phase7E3_final_TF_perturbation_atlas.tsv"
)

PHASE7D_ATLAS_FILE = (
    PROJECT
    / "07_tables/main_tables/phase7/"
      "phase7D4_final_primary_TF_cis_regulatory_atlas.tsv"
)

PHASE7E_COMPLETION_FILE = (
    PROJECT
    / "07_tables/main_tables/phase7/"
      "phase7E_completion_summary.tsv"
)

PHASE7D_COMPLETION_FILE = (
    PROJECT
    / "07_tables/main_tables/phase7/"
      "phase7D_completion_summary.tsv"
)

PHASE7C4_COMPLETION_FILE = (
    PROJECT
    / "07_tables/main_tables/phase7/"
      "phase7C4_completion_summary.tsv"
)

PHASE7B6_COMPLETION_FILE = (
    PROJECT
    / "07_tables/main_tables/phase7/"
      "phase7B6_completion_summary.tsv"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase7"
)

FIGURE_DIR = (
    PROJECT
    / "06_figures/main_figures/phase7"
)

LOG_DIR = (
    PROJECT
    / "09_pipeline_logs/phase7"
)


FINAL_ATLAS_OUTPUT = (
    TABLE_DIR
    / "phase7F1_final_integrated_regulatory_atlas.tsv"
)

EVIDENCE_CLASS_OUTPUT = (
    TABLE_DIR
    / "phase7F1_integrated_evidence_class_summary.tsv"
)

MECHANISTIC_SUMMARY_OUTPUT = (
    TABLE_DIR
    / "phase7F1_mechanistic_regulator_summary.tsv"
)

COMPLETION_OUTPUT = (
    TABLE_DIR
    / "phase7F1_completion_summary.tsv"
)

FIGURE_PNG = (
    FIGURE_DIR
    / "Figure54_final_integrated_developmental_regulatory_atlas.png"
)

FIGURE_PDF = (
    FIGURE_DIR
    / "Figure54_final_integrated_developmental_regulatory_atlas.pdf"
)

LOG_FILE = (
    LOG_DIR
    / "phase7F1_final_integrated_regulatory_atlas.log"
)


EXPECTED_TFS = 10
EXPECTED_DIRECT_CIS_TFS = 6
EXPECTED_PRIMARY_PERTURBATION_TFS = 8
EXPECTED_SENSITIVITY_TFS = 2
EXPECTED_TEMPORAL_PROGRAM_SUPPORTED_TFS = 4
EXPECTED_DTHI_SUPPORTED_TFS = 1


for directory in [
    TABLE_DIR,
    FIGURE_DIR,
    LOG_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def read_table(
    path: Path,
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        compression="infer",
        low_memory=False,
    )


def parse_boolean(
    values: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        values
    ):
        return values.fillna(
            False
        )

    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
                "y",
            ]
        )
    )


def require_columns(
    table: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = [
        column
        for column in required
        if column not in table.columns
    ]

    if missing:
        raise RuntimeError(
            (
                f"{label} is missing required columns: "
                + "|".join(
                    missing
                )
            )
        )


def main() -> None:
    required_files = [
        PHASE7E_ATLAS_FILE,
        PHASE7D_ATLAS_FILE,
        PHASE7E_COMPLETION_FILE,
        PHASE7D_COMPLETION_FILE,
        PHASE7C4_COMPLETION_FILE,
        PHASE7B6_COMPLETION_FILE,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing Phase 7F1 inputs:\n"
            + "\n".join(
                missing_files
            )
        )

    atlas = read_table(
        PHASE7E_ATLAS_FILE
    )

    phase7d_context = read_table(
        PHASE7D_ATLAS_FILE
    )

    context_columns = [
        "regulator",
        "adult_favored_neuronal_class",
        "highest_mean_adult_broad_cell_class",
        "fetal_directionally_robust_exploratory_pair_count",
    ]

    require_columns(
        phase7d_context,
        context_columns,
        "Phase 7D final cis-regulatory atlas",
    )

    if len(
        phase7d_context
    ) != EXPECTED_TFS:
        raise RuntimeError(
            (
                "Unexpected Phase 7D context TF count: "
                f"{len(phase7d_context)}."
            )
        )

    if phase7d_context[
        "regulator"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicated regulators in Phase 7D context atlas."
        )

    # Remove any partially retained versions before restoring the
    # canonical Phase 7D context fields.
    restore_columns = context_columns[
        1:
    ]

    atlas = atlas.drop(
        columns=[
            column
            for column in restore_columns
            if column in atlas.columns
        ]
    )

    atlas = atlas.merge(
        phase7d_context[
            context_columns
        ],
        on="regulator",
        how="left",
        validate="one_to_one",
    )

    missing_context_rows = int(
        atlas[
            restore_columns
        ].isna().all(
            axis=1
        ).sum()
    )

    if missing_context_rows != 0:
        raise RuntimeError(
            (
                "Phase 7D context restoration failed for "
                f"{missing_context_rows} regulators."
            )
        )

    required_columns = [
        "regulator",
        "final_regulatory_rank",
        "regulatory_evidence_tier",
        "developmental_program_direction",
        "integrated_celltype_evidence_class",
        "adult_favored_neuronal_class",
        "highest_mean_adult_broad_cell_class",
        "fetal_directionally_robust_exploratory_pair_count",
        "robust_cis_validation",
        "final_cis_regulatory_evidence_class",
        "model_family",
        "robust_primary_perturbation_prediction",
        "robust_small_regulon_sensitivity_prediction",
        "primary_temporal_program_shift_support",
        "small_regulon_temporal_shift_sensitivity_support",
        "primary_supported_DTHI_modules",
        "sensitivity_supported_DTHI_modules",
        "primary_supported_celltype_modules",
        "sensitivity_supported_celltype_modules",
        "cis_subset_enriched_vs_random_targets",
        "final_perturbation_evidence_class",
    ]

    require_columns(
        atlas,
        required_columns,
        "Phase 7E final atlas",
    )

    if len(
        atlas
    ) != EXPECTED_TFS:
        raise RuntimeError(
            (
                "Unexpected primary regulator count: "
                f"{len(atlas)}."
            )
        )

    if atlas[
        "regulator"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicated regulators in Phase 7E atlas."
        )

    boolean_columns = [
        "robust_cis_validation",
        "robust_primary_perturbation_prediction",
        "robust_small_regulon_sensitivity_prediction",
        "primary_temporal_program_shift_support",
        "small_regulon_temporal_shift_sensitivity_support",
        "cis_subset_enriched_vs_random_targets",
    ]

    for column in boolean_columns:
        atlas[
            column
        ] = parse_boolean(
            atlas[
                column
            ]
        )

    numeric_fill_columns = [
        "fetal_directionally_robust_exploratory_pair_count",
        "primary_supported_DTHI_modules",
        "sensitivity_supported_DTHI_modules",
        "primary_supported_celltype_modules",
        "sensitivity_supported_celltype_modules",
    ]

    for column in numeric_fill_columns:
        atlas[
            column
        ] = pd.to_numeric(
            atlas[
                column
            ],
            errors="coerce",
        ).fillna(
            0
        )

    atlas[
        "temporal_mechanistic_score"
    ] = np.where(
        atlas[
            "regulatory_evidence_tier"
        ].eq(
            "Tier_A_stringent_temporal_mechanistic"
        ),
        3,
        np.where(
            atlas[
                "regulatory_evidence_tier"
            ].eq(
                "Tier_B_robust_temporal_mechanistic"
            ),
            2,
            0,
        ),
    )

    primary_celltype_classes = {
        "adult_primary_localization_only",
        "adult_primary_plus_exploratory_fetal_directional_robustness",
    }

    secondary_celltype_classes = {
        "adult_secondary_localization_only",
        "adult_secondary_plus_exploratory_fetal_directional_robustness",
        "exploratory_fetal_directional_robustness_only",
    }

    atlas[
        "celltype_localization_score"
    ] = np.where(
        atlas[
            "integrated_celltype_evidence_class"
        ].isin(
            primary_celltype_classes
        ),
        2,
        np.where(
            atlas[
                "integrated_celltype_evidence_class"
            ].isin(
                secondary_celltype_classes
            ),
            1,
            0,
        ),
    )

    atlas[
        "cis_regulatory_score"
    ] = np.where(
        atlas[
            "robust_cis_validation"
        ],
        3,
        np.where(
            atlas[
                "final_cis_regulatory_evidence_class"
            ].eq(
                "Tier_D3_motif_testable_without_target_enrichment"
            ),
            1,
            0,
        ),
    )

    atlas[
        "perturbation_score"
    ] = np.where(
        atlas[
            "robust_primary_perturbation_prediction"
        ],
        3,
        np.where(
            atlas[
                "robust_small_regulon_sensitivity_prediction"
            ],
            1,
            0,
        ),
    )

    atlas[
        "temporal_program_shift_supported"
    ] = (
        atlas[
            "primary_temporal_program_shift_support"
        ]
        | atlas[
            "small_regulon_temporal_shift_sensitivity_support"
        ]
    )

    atlas[
        "supported_DTHI_module_count"
    ] = (
        atlas[
            "primary_supported_DTHI_modules"
        ]
        + atlas[
            "sensitivity_supported_DTHI_modules"
        ]
    ).astype(
        int
    )

    atlas[
        "supported_celltype_module_count"
    ] = (
        atlas[
            "primary_supported_celltype_modules"
        ]
        + atlas[
            "sensitivity_supported_celltype_modules"
        ]
    ).astype(
        int
    )

    atlas[
        "program_impact_score"
    ] = (
        atlas[
            "temporal_program_shift_supported"
        ].astype(
            int
        )
        * 2
        + atlas[
            "supported_DTHI_module_count"
        ].gt(
            0
        ).astype(
            int
        )
        + atlas[
            "supported_celltype_module_count"
        ].gt(
            0
        ).astype(
            int
        )
    )

    atlas[
        "cis_subset_specificity_score"
    ] = atlas[
        "cis_subset_enriched_vs_random_targets"
    ].astype(
        int
    )

    atlas[
        "descriptive_multilayer_evidence_score"
    ] = (
        atlas[
            "temporal_mechanistic_score"
        ]
        + atlas[
            "celltype_localization_score"
        ]
        + atlas[
            "cis_regulatory_score"
        ]
        + atlas[
            "perturbation_score"
        ]
        + atlas[
            "program_impact_score"
        ]
        + atlas[
            "cis_subset_specificity_score"
        ]
    )

    def final_evidence_class(
        row: pd.Series,
    ) -> str:
        if (
            bool(
                row[
                    "robust_primary_perturbation_prediction"
                ]
            )
            and bool(
                row[
                    "robust_cis_validation"
                ]
            )
        ):
            return (
                "Tier_F1_convergent_temporal_"
                "cis_perturbational_regulator"
            )

        if bool(
            row[
                "robust_primary_perturbation_prediction"
            ]
        ):
            return (
                "Tier_F2_primary_perturbational_"
                "regulator_without_direct_cis_validation"
            )

        if bool(
            row[
                "robust_small_regulon_sensitivity_prediction"
            ]
        ):
            return (
                "Tier_F3_small_regulon_"
                "perturbational_sensitivity_regulator"
            )

        return (
            "Tier_F4_no_robust_perturbational_support"
        )

    atlas[
        "final_phase7_evidence_class"
    ] = atlas.apply(
        final_evidence_class,
        axis=1,
    )

    def interpretation(
        row: pd.Series,
    ) -> str:
        direction = str(
            row[
                "developmental_program_direction"
            ]
        )

        if direction == "fetal_high":
            direction_text = (
                "fetal-high developmental regulator"
            )

        elif direction == "maturation_high":
            direction_text = (
                "maturation-high developmental regulator"
            )

        else:
            direction_text = (
                "developmental regulator"
            )

        evidence_parts = [
            direction_text,
        ]

        if bool(
            row[
                "robust_cis_validation"
            ]
        ):
            evidence_parts.append(
                "direct promoter-cis validated"
            )

        if bool(
            row[
                "robust_primary_perturbation_prediction"
            ]
        ):
            evidence_parts.append(
                "primary cross-fitted perturbational support"
            )

        elif bool(
            row[
                "robust_small_regulon_sensitivity_prediction"
            ]
        ):
            evidence_parts.append(
                "small-regulon perturbational sensitivity support"
            )

        if bool(
            row[
                "temporal_program_shift_supported"
            ]
        ):
            evidence_parts.append(
                "trajectory-concordant temporal-program impact"
            )

        if int(
            row[
                "supported_DTHI_module_count"
            ]
        ) > 0:
            evidence_parts.append(
                "DTHI-module impact"
            )

        if bool(
            row[
                "cis_subset_enriched_vs_random_targets"
            ]
        ):
            evidence_parts.append(
                "cis-subset specificity"
            )

        return "; ".join(
            evidence_parts
        )

    atlas[
        "final_integrated_interpretation"
    ] = atlas.apply(
        interpretation,
        axis=1,
    )

    class_order = {
        "Tier_F1_convergent_temporal_cis_perturbational_regulator":
            1,

        "Tier_F2_primary_perturbational_regulator_without_direct_cis_validation":
            2,

        "Tier_F3_small_regulon_perturbational_sensitivity_regulator":
            3,

        "Tier_F4_no_robust_perturbational_support":
            4,
    }

    atlas[
        "final_phase7_evidence_class_rank"
    ] = atlas[
        "final_phase7_evidence_class"
    ].map(
        class_order
    )

    atlas = atlas.sort_values(
        [
            "final_phase7_evidence_class_rank",
            "descriptive_multilayer_evidence_score",
            "final_regulatory_rank",
            "regulator",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    atlas[
        "final_integrated_rank"
    ] = np.arange(
        1,
        len(
            atlas
        )
        + 1,
    )

    final_columns = [
        "final_integrated_rank",
        "regulator",
        "final_phase7_evidence_class",
        "descriptive_multilayer_evidence_score",
        "regulatory_evidence_tier",
        "developmental_program_direction",
        "integrated_celltype_evidence_class",
        "adult_favored_neuronal_class",
        "highest_mean_adult_broad_cell_class",
        "final_cis_regulatory_evidence_class",
        "robust_cis_validation",
        "cis_subset_enriched_vs_random_targets",
        "model_family",
        "robust_primary_perturbation_prediction",
        "robust_small_regulon_sensitivity_prediction",
        "temporal_program_shift_supported",
        "supported_DTHI_module_count",
        "supported_celltype_module_count",
        "temporal_mechanistic_score",
        "celltype_localization_score",
        "cis_regulatory_score",
        "perturbation_score",
        "program_impact_score",
        "cis_subset_specificity_score",
        "final_integrated_interpretation",
    ]

    atlas[
        final_columns
    ].to_csv(
        FINAL_ATLAS_OUTPUT,
        sep="\t",
        index=False,
    )

    evidence_summary = (
        atlas.groupby(
            [
                "final_phase7_evidence_class",
                "final_phase7_evidence_class_rank",
            ],
            sort=True,
        )
        .agg(
            TF_count=(
                "regulator",
                "size",
            ),

            TFs=(
                "regulator",
                lambda values:
                    "|".join(
                        values.astype(str)
                    ),
            ),

            maturation_high_TFs=(
                "developmental_program_direction",
                lambda values:
                    int(
                        values.eq(
                            "maturation_high"
                        ).sum()
                    ),
            ),

            fetal_high_TFs=(
                "developmental_program_direction",
                lambda values:
                    int(
                        values.eq(
                            "fetal_high"
                        ).sum()
                    ),
            ),

            median_multilayer_score=(
                "descriptive_multilayer_evidence_score",
                "median",
            ),
        )
        .reset_index()
        .sort_values(
            "final_phase7_evidence_class_rank"
        )
    )

    evidence_summary.to_csv(
        EVIDENCE_CLASS_OUTPUT,
        sep="\t",
        index=False,
    )

    mechanistic_summary = atlas[
        [
            "regulator",
            "developmental_program_direction",
            "adult_favored_neuronal_class",
            "highest_mean_adult_broad_cell_class",
            "robust_cis_validation",
            "cis_subset_enriched_vs_random_targets",
            "temporal_program_shift_supported",
            "supported_DTHI_module_count",
            "final_phase7_evidence_class",
            "final_integrated_interpretation",
        ]
    ].copy()

    mechanistic_summary.to_csv(
        MECHANISTIC_SUMMARY_OUTPUT,
        sep="\t",
        index=False,
    )

    # ================================================================
    # Figure 54
    # ================================================================

    plot_atlas = atlas.sort_values(
        "final_integrated_rank"
    ).reset_index(
        drop=True
    )

    regulator_order = plot_atlas[
        "regulator"
    ].tolist()

    plot_atlas_by_regulator = (
        plot_atlas.set_index(
            "regulator",
            drop=False,
        )
        .loc[
            regulator_order
        ]
    )

    evidence_matrix = pd.DataFrame(
        {
            "Stringent temporal":
                plot_atlas_by_regulator[
                    "regulatory_evidence_tier"
                ]
                .eq(
                    "Tier_A_stringent_temporal_mechanistic"
                )
                .astype(
                    int
                ),

            "Cell-type localization":
                pd.to_numeric(
                    plot_atlas_by_regulator[
                        "celltype_localization_score"
                    ],
                    errors="coerce",
                )
                .fillna(
                    0
                )
                .gt(
                    0
                )
                .astype(
                    int
                ),

            "Direct cis validation":
                plot_atlas_by_regulator[
                    "robust_cis_validation"
                ]
                .fillna(
                    False
                )
                .astype(
                    bool
                )
                .astype(
                    int
                ),

            "Cis-subset specificity":
                plot_atlas_by_regulator[
                    "cis_subset_enriched_vs_random_targets"
                ]
                .fillna(
                    False
                )
                .astype(
                    bool
                )
                .astype(
                    int
                ),

            "Primary perturbation":
                plot_atlas_by_regulator[
                    "robust_primary_perturbation_prediction"
                ]
                .fillna(
                    False
                )
                .astype(
                    bool
                )
                .astype(
                    int
                ),

            "Small-regulon sensitivity":
                plot_atlas_by_regulator[
                    "robust_small_regulon_sensitivity_prediction"
                ]
                .fillna(
                    False
                )
                .astype(
                    bool
                )
                .astype(
                    int
                ),

            "Temporal-program impact":
                plot_atlas_by_regulator[
                    "temporal_program_shift_supported"
                ]
                .fillna(
                    False
                )
                .astype(
                    bool
                )
                .astype(
                    int
                ),

            "DTHI-module impact":
                pd.to_numeric(
                    plot_atlas_by_regulator[
                        "supported_DTHI_module_count"
                    ],
                    errors="coerce",
                )
                .fillna(
                    0
                )
                .gt(
                    0
                )
                .astype(
                    int
                ),
        },
        index=regulator_order,
    )

    if evidence_matrix.isna().any().any():
        missing_cells = (
            evidence_matrix.isna()
            .stack()
            .loc[
                lambda values:
                    values
            ]
            .index
            .tolist()
        )

        raise RuntimeError(
            (
                "Figure 54 evidence matrix still contains "
                f"missing values: {missing_cells}"
            )
        )

    if not evidence_matrix.isin(
        [
            0,
            1,
        ]
    ).all().all():
        raise RuntimeError(
            "Figure 54 evidence matrix contains values other than 0 or 1."
        )

    figure = plt.figure(
        figsize=(
            20,
            15,
        ),
        constrained_layout=True,
    )

    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[
            1.45,
            1,
        ],
    )

    ax_matrix = figure.add_subplot(
        grid[
            0,
            0,
        ]
    )

    ax_score = figure.add_subplot(
        grid[
            0,
            1,
        ]
    )

    ax_table = figure.add_subplot(
        grid[
            1,
            0,
        ]
    )

    ax_classes = figure.add_subplot(
        grid[
            1,
            1,
        ]
    )

    ax_matrix.imshow(
        evidence_matrix.to_numpy(
            dtype=float
        ),
        aspect="auto",
        interpolation="nearest",
        cmap="Greys",
        vmin=0,
        vmax=1,
    )

    ax_matrix.set_xticks(
        np.arange(
            evidence_matrix.shape[
                1
            ]
        )
    )

    ax_matrix.set_xticklabels(
        evidence_matrix.columns,
        rotation=38,
        ha="right",
        fontsize=9,
    )

    ax_matrix.set_yticks(
        np.arange(
            len(
                regulator_order
            )
        )
    )

    ax_matrix.set_yticklabels(
        regulator_order
    )

    ax_matrix.set_title(
        "A  Multilayer regulatory evidence "
        "(black = supported)"
    )

    for row_index in range(
        evidence_matrix.shape[
            0
        ]
    ):
        for column_index in range(
            evidence_matrix.shape[
                1
            ]
        ):
            value = int(
                evidence_matrix.iloc[
                    row_index,
                    column_index,
                ]
            )

            ax_matrix.text(
                column_index,
                row_index,
                "✓" if value else "–",
                ha="center",
                va="center",
                fontsize=11,
                color=(
                    "white"
                    if value
                    else "black"
                ),
                fontweight=(
                    "bold"
                    if value
                    else "normal"
                ),
            )

    y_positions = np.arange(
        len(
            regulator_order
        )
    )

    scores = plot_atlas[
        "descriptive_multilayer_evidence_score"
    ].to_numpy(
        dtype=float
    )

    ax_score.barh(
        y_positions,
        scores,
    )

    ax_score.set_yticks(
        y_positions
    )

    ax_score.set_yticklabels(
        regulator_order
    )

    ax_score.invert_yaxis()

    ax_score.set_xlabel(
        "Descriptive multilayer evidence score"
    )

    ax_score.set_title(
        "B  Integrated evidence ranking"
    )

    for index, score in enumerate(
        scores
    ):
        ax_score.text(
            score
            + 0.10,
            index,
            str(
                int(
                    score
                )
            ),
            va="center",
            fontsize=9,
        )

    table_rows = []

    for row in plot_atlas.itertuples(
        index=False
    ):
        class_label = str(
            row.final_phase7_evidence_class
        ).split(
            "_"
        )[
            1
        ]

        table_rows.append(
            [
                row.developmental_program_direction.replace(
                    "_",
                    " ",
                ),

                str(
                    row.adult_favored_neuronal_class
                ).replace(
                    "_",
                    " ",
                ),

                str(
                    row.highest_mean_adult_broad_cell_class
                ).replace(
                    "_",
                    " ",
                ),

                class_label,
            ]
        )

    ax_table.axis(
        "off"
    )

    summary_table = ax_table.table(
        cellText=table_rows,
        rowLabels=regulator_order,
        colLabels=[
            "Developmental\nprogram",
            "Favored neuronal\nclass",
            "Highest adult\nbroad class",
            "Final\ntier",
        ],
        cellLoc="center",
        rowLoc="center",
        loc="center",
        bbox=[
            0,
            0,
            1,
            0.95,
        ],
    )

    summary_table.auto_set_font_size(
        False
    )

    summary_table.set_fontsize(
        8
    )

    for (
        row_index,
        column_index,
    ), cell in summary_table.get_celld().items():
        if row_index == 0:
            cell.set_text_props(
                fontweight="bold"
            )

    ax_table.set_title(
        "C  Developmental and cell-type context",
        pad=10,
    )

    class_plot = evidence_summary.sort_values(
        "final_phase7_evidence_class_rank"
    )

    class_labels = [
        str(value).split(
            "_"
        )[
            1
        ]
        for value in class_plot[
            "final_phase7_evidence_class"
        ]
    ]

    class_counts = class_plot[
        "TF_count"
    ].to_numpy(
        dtype=int
    )

    class_positions = np.arange(
        len(
            class_counts
        )
    )

    ax_classes.bar(
        class_positions,
        class_counts,
    )

    ax_classes.set_xticks(
        class_positions
    )

    ax_classes.set_xticklabels(
        class_labels
    )

    ax_classes.set_ylabel(
        "Number of regulators"
    )

    ax_classes.set_title(
        "D  Final Phase 7 evidence classes",
        pad=12,
    )

    for index, row in class_plot.reset_index(
        drop=True
    ).iterrows():
        class_count = int(
            row[
                "TF_count"
            ]
        )

        TF_label = str(
            row[
                "TFs"
            ]
        ).replace(
            "|",
            "\n",
        )

        ax_classes.text(
            index,
            class_count
            / 2,
            TF_label,
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )

        ax_classes.text(
            index,
            class_count
            + 0.08,
            str(
                class_count
            ),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax_classes.set_ylim(
        0,
        max(
            class_counts
        )
        + 0.65,
    )

    figure.suptitle(
        (
            "Figure 54. Final integrated developmental "
            "transcription-factor regulatory atlas"
        ),
        fontsize=17,
    )

    figure.text(
        0.5,
        0.010,
        (
            "Black tiles indicate supported evidence; dashes "
            "indicate absence of that evidence layer. The multilayer "
            "score is descriptive and is not a probability, causal "
            "estimate or therapeutic ranking. Small-regulon RFXAP "
            "and SRSF2 findings remain sensitivity evidence."
        ),
        ha="center",
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

    figure_valid = (
        FIGURE_PNG.exists()
        and FIGURE_PNG.stat().st_size
        > 50000
        and FIGURE_PDF.exists()
        and FIGURE_PDF.stat().st_size
        > 10000
    )

    class_counts_observed = (
        atlas[
            "final_phase7_evidence_class"
        ].value_counts()
    )

    tier_f1_count = int(
        class_counts_observed.get(
            "Tier_F1_convergent_temporal_cis_perturbational_regulator",
            0,
        )
    )

    tier_f2_count = int(
        class_counts_observed.get(
            "Tier_F2_primary_perturbational_regulator_without_direct_cis_validation",
            0,
        )
    )

    tier_f3_count = int(
        class_counts_observed.get(
            "Tier_F3_small_regulon_perturbational_sensitivity_regulator",
            0,
        )
    )

    status = (
        "completed"
        if (
            len(
                atlas
            )
            == EXPECTED_TFS
            and int(
                atlas[
                    "robust_cis_validation"
                ].sum()
            )
            == EXPECTED_DIRECT_CIS_TFS
            and int(
                atlas[
                    "robust_primary_perturbation_prediction"
                ].sum()
            )
            == EXPECTED_PRIMARY_PERTURBATION_TFS
            and int(
                atlas[
                    "robust_small_regulon_sensitivity_prediction"
                ].sum()
            )
            == EXPECTED_SENSITIVITY_TFS
            and int(
                atlas[
                    "temporal_program_shift_supported"
                ].sum()
            )
            == EXPECTED_TEMPORAL_PROGRAM_SUPPORTED_TFS
            and int(
                atlas[
                    "supported_DTHI_module_count"
                ].gt(
                    0
                ).sum()
            )
            == EXPECTED_DTHI_SUPPORTED_TFS
            and tier_f1_count
            == 6
            and tier_f2_count
            == 2
            and tier_f3_count
            == 2
            and figure_valid
            and FINAL_ATLAS_OUTPUT.exists()
            and EVIDENCE_CLASS_OUTPUT.exists()
            and MECHANISTIC_SUMMARY_OUTPUT.exists()
        )
        else "failed_validation"
    )

    completion = pd.DataFrame(
        [
            {
                "primary_mechanistic_TFs":
                    len(
                        atlas
                    ),

                "Tier_F1_convergent_regulators":
                    tier_f1_count,

                "Tier_F2_primary_without_direct_cis":
                    tier_f2_count,

                "Tier_F3_small_regulon_sensitivity":
                    tier_f3_count,

                "direct_cis_validated_TFs":
                    int(
                        atlas[
                            "robust_cis_validation"
                        ].sum()
                    ),

                "primary_robust_perturbation_TFs":
                    int(
                        atlas[
                            "robust_primary_perturbation_prediction"
                        ].sum()
                    ),

                "small_regulon_sensitivity_TFs":
                    int(
                        atlas[
                            "robust_small_regulon_sensitivity_prediction"
                        ].sum()
                    ),

                "temporal_program_supported_TFs":
                    int(
                        atlas[
                            "temporal_program_shift_supported"
                        ].sum()
                    ),

                "DTHI_module_supported_TFs":
                    int(
                        atlas[
                            "supported_DTHI_module_count"
                        ].gt(
                            0
                        ).sum()
                    ),

                "celltype_module_supported_TFs":
                    int(
                        atlas[
                            "supported_celltype_module_count"
                        ].gt(
                            0
                        ).sum()
                    ),

                "cis_subset_specificity_TFs":
                    int(
                        atlas[
                            "cis_subset_enriched_vs_random_targets"
                        ].sum()
                    ),

                "descriptive_score_used":
                    True,

                "score_interpreted_as_causal_or_probabilistic":
                    False,

                "Figure54_PNG_created":
                    FIGURE_PNG.exists(),

                "Figure54_PDF_created":
                    FIGURE_PDF.exists(),

                "causal_interpretation_permitted":
                    False,

                "Phase7F1_status":
                    status,
            }
        ]
    )

    completion.to_csv(
        COMPLETION_OUTPUT,
        sep="\t",
        index=False,
    )

    display_columns = [
        "final_integrated_rank",
        "regulator",
        "developmental_program_direction",
        "descriptive_multilayer_evidence_score",
        "final_phase7_evidence_class",
        "final_integrated_interpretation",
    ]

    log_text = "\n".join(
        [
            "===== PHASE 7F1 COMPLETION =====",
            completion.to_string(
                index=False
            ),
            "",
            "===== FINAL INTEGRATED REGULATORY ATLAS =====",
            atlas[
                display_columns
            ].to_string(
                index=False
            ),
            "",
            "===== INTEGRATED EVIDENCE CLASS SUMMARY =====",
            evidence_summary.to_string(
                index=False
            ),
            "",
            "===== FIGURE 54 =====",
            (
                f"PNG\t{FIGURE_PNG}\t"
                f"{FIGURE_PNG.stat().st_size}"
            ),
            (
                f"PDF\t{FIGURE_PDF}\t"
                f"{FIGURE_PDF.stat().st_size}"
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
        raise RuntimeError(
            "Phase 7F1 final integration validation failed."
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            (
                "Phase 7F1 failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
