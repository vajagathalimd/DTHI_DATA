#!/usr/bin/env python3
"""Phase 8D3D-R2: final targeted revision of the integrated figure."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

R1_SCRIPT = (
    ROOT
    / "04_scripts"
    / "10_functional_imaging"
    / "phase8D"
    / "03D3D_R1_revise_integrated_cross_modal_figure.py"
)

if not R1_SCRIPT.is_file():
    raise FileNotFoundError(
        f"Required R1 script not found: {R1_SCRIPT}"
    )

SPEC = importlib.util.spec_from_file_location(
    "phase8D3D_R1_module",
    R1_SCRIPT,
)

if SPEC is None or SPEC.loader is None:
    raise RuntimeError(
        "Could not construct an import specification "
        "for the Phase 8D3D-R1 script."
    )

R1 = importlib.util.module_from_spec(
    SPEC
)

SPEC.loader.exec_module(
    R1
)


R2_OUT = (
    ROOT
    / "03_processed_data"
    / "functional_imaging"
    / "phase8D"
    / "phase8D3D_R2"
)

R2_FIG_DIR = (
    ROOT
    / "08_figures"
    / "phase8"
    / "phase8D3D_R2"
)

R2_FIG_BASE = (
    R2_FIG_DIR
    / "phase8D3D_R2_integrated_cross_modal_figure"
)

R2_MAP_DATA = (
    R2_OUT
    / "phase8D3D_R2_figure_data.tsv"
)

R2_PRIORITY_DATA = (
    R2_OUT
    / "phase8D3D_R2_priority_panel_data.tsv"
)

R2_PROVENANCE = (
    R2_OUT
    / "phase8D3D_R2_source_provenance.tsv"
)

R2_LEGEND = (
    R2_OUT
    / "phase8D3D_R2_figure_legend.txt"
)

R2_MANIFEST = (
    R2_OUT
    / "phase8D3D_R2_figure_output_manifest.tsv"
)

R2_AUDIT = (
    ROOT
    / "07_tables"
    / "main_tables"
    / "phase8"
    / "phase8D3D_R2_figure_audit.tsv"
)

R2_COMPLETION = (
    ROOT
    / "07_tables"
    / "main_tables"
    / "phase8"
    / "phase8D3D_R2_completion_summary.tsv"
)

R1_COMPLETION = (
    ROOT
    / "07_tables"
    / "main_tables"
    / "phase8"
    / "phase8D3D_R1_completion_summary.tsv"
)


TIER_STYLE = {
    "Tier1_global_FWER": {
        "marker": "*",
        "size": 220,
        "facecolor": "black",
        "edgecolor": "black",
        "linewidth": 1.1,
    },
    "Tier2_family_corrected": {
        "marker": "s",
        "size": 95,
        "facecolor": "0.30",
        "edgecolor": "0.30",
        "linewidth": 1.0,
    },
    "Tier3A_nominal": {
        "marker": "o",
        "size": 72,
        "facecolor": "white",
        "edgecolor": "0.25",
        "linewidth": 1.3,
    },
    "Tier3B_descriptive": {
        "marker": "x",
        "size": 66,
        "facecolor": "0.25",
        "edgecolor": "0.25",
        "linewidth": 1.3,
    },
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def add_panel_label(
    axis: plt.Axes,
    label: str,
) -> None:
    axis.text(
        -0.12,
        1.025,
        label,
        transform=axis.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def apply_y_axis(
    axis: plt.Axes,
    data: pd.DataFrame,
) -> np.ndarray:
    y_positions = np.arange(
        len(data)
    )[::-1]

    axis.set_yticks(
        y_positions
    )

    axis.set_yticklabels(
        data[
            "map_label"
        ].tolist()
    )

    axis.set_ylim(
        -0.7,
        len(data) + 0.25,
    )

    axis.tick_params(
        axis="y",
        pad=4,
    )

    return y_positions


def draw_tier_marker(
    axis: plt.Axes,
    x_value: float,
    y_value: float,
    tier: str,
    zorder: int = 3,
) -> None:
    style = TIER_STYLE[
        tier
    ]

    marker = style[
        "marker"
    ]

    if marker == "x":
        axis.scatter(
            [x_value],
            [y_value],
            marker=marker,
            s=style["size"],
            color=style["edgecolor"],
            linewidths=style["linewidth"],
            zorder=zorder,
        )
    else:
        axis.scatter(
            [x_value],
            [y_value],
            marker=marker,
            s=style["size"],
            facecolors=style["facecolor"],
            edgecolors=style["edgecolor"],
            linewidths=style["linewidth"],
            zorder=zorder,
        )


def panel_a(
    axis: plt.Axes,
    data: pd.DataFrame,
) -> None:
    y_positions = apply_y_axis(
        axis,
        data,
    )

    axis.grid(
        axis="x",
        linewidth=0.7,
        alpha=0.18,
        zorder=0,
    )

    for index, row in data.iterrows():
        x_value = float(
            row[
                "window_order"
            ]
        )

        y_value = float(
            y_positions[index]
        )

        tier = str(
            row[
                "map_level_highest_tier"
            ]
        )

        draw_tier_marker(
            axis,
            x_value,
            y_value,
            tier,
        )

        axis.annotate(
            str(
                row[
                    "tier_label"
                ]
            ),
            xy=(
                x_value,
                y_value,
            ),
            xytext=(
                8,
                0,
            ),
            textcoords="offset points",
            fontsize=7.6,
            ha="left",
            va="center",
        )

    axis.set_xticks(
        np.arange(
            len(R1.WINDOW_ORDER)
        )
    )

    axis.set_xticklabels(
        [
            R1.WINDOW_SHORT[
                window
            ]
            for window in R1.WINDOW_ORDER
        ],
        rotation=30,
        ha="right",
    )

    axis.set_xlim(
        -0.45,
        len(R1.WINDOW_ORDER) - 0.25,
    )

    axis.set_xlabel(
        "Empirical peak developmental window",
        labelpad=7,
    )

    axis.set_title(
        "Developmental peak timing",
        loc="left",
        fontweight="bold",
        pad=8,
    )

    add_panel_label(
        axis,
        "A",
    )


def compact_external_label(
    value: Any,
) -> str:
    label = R1.external_label(
        value
    )

    replacements = {
        "Executive Control":
            "Executive control",
        "Reward Motivation":
            "Reward motivation",
        "Emotion/Affect":
            "Emotion/affect",
        "Social Cognition":
            "Social cognition",
        "Sensorimotor":
            "Sensorimotor",
        "Depression":
            "Depression",
        "Bipolar":
            "Bipolar",
        "Epilepsy":
            "Epilepsy",
        "Attention":
            "Attention",
        "ASD":
            "ASD",
    }

    return replacements.get(
        label,
        label,
    )


def panel_b(
    axis: plt.Axes,
    data: pd.DataFrame,
) -> None:
    y_positions = apply_y_axis(
        axis,
        data,
    )

    minimum = data[
        "AHBA_minimum_LODO_vs_full_rho"
    ].to_numpy(
        dtype=float
    )

    median = data[
        "AHBA_median_LODO_vs_full_rho"
    ].to_numpy(
        dtype=float
    )

    axis.grid(
        axis="x",
        linewidth=0.7,
        alpha=0.18,
        zorder=0,
    )

    axis.hlines(
        y=y_positions,
        xmin=minimum,
        xmax=median,
        linewidth=2.0,
        color="0.55",
        zorder=1,
    )

    axis.scatter(
        minimum,
        y_positions,
        marker="o",
        s=54,
        facecolors="white",
        edgecolors="black",
        linewidths=1.2,
        label="Minimum LODO",
        zorder=3,
    )

    axis.scatter(
        median,
        y_positions,
        marker="s",
        s=52,
        facecolors="0.35",
        edgecolors="0.35",
        linewidths=1.0,
        label="Median LODO",
        zorder=3,
    )

    for minimum_value, median_value, y_value in zip(
        minimum,
        median,
        y_positions,
    ):
        axis.annotate(
            f"{minimum_value:.2f}",
            xy=(
                minimum_value,
                y_value,
            ),
            xytext=(
                -7,
                0,
            ),
            textcoords="offset points",
            fontsize=7.5,
            ha="right",
            va="center",
        )

        axis.annotate(
            f"{median_value:.2f}",
            xy=(
                median_value,
                y_value,
            ),
            xytext=(
                7,
                0,
            ),
            textcoords="offset points",
            fontsize=7.5,
            ha="left",
            va="center",
        )

    axis.set_xlim(
        min(
            0.68,
            float(
                np.nanmin(
                    minimum
                )
            ) - 0.025,
        ),
        1.015,
    )

    axis.set_xlabel(
        "Correlation with the full donor-balanced map",
        labelpad=7,
    )

    axis.set_title(
        "AHBA donor robustness",
        loc="left",
        fontweight="bold",
        pad=8,
    )

    axis.legend(
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            -0.20,
        ),
        ncol=2,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.3,
    )

    add_panel_label(
        axis,
        "B",
    )


def panel_c(
    axis: plt.Axes,
    data: pd.DataFrame,
) -> None:
    y_positions = apply_y_axis(
        axis,
        data,
    )

    cognitive = data[
        "strongest_cognitive_rho"
    ].to_numpy(
        dtype=float
    )

    clinical = data[
        "strongest_clinical_rho"
    ].to_numpy(
        dtype=float
    )

    clinical_column_x = -0.84
    clinical_separator_x = -0.64

    cognitive_separator_x = 0.53
    cognitive_column_x = 0.76

    axis.axvline(
        0,
        linewidth=1.0,
        color="0.35",
        zorder=0,
    )

    axis.axvline(
        clinical_separator_x,
        linewidth=0.8,
        linestyle=":",
        color="0.60",
        zorder=0,
    )

    axis.axvline(
        cognitive_separator_x,
        linewidth=0.8,
        linestyle=":",
        color="0.60",
        zorder=0,
    )

    axis.grid(
        axis="x",
        linewidth=0.7,
        alpha=0.15,
        zorder=0,
    )

    for cognitive_value, clinical_value, y_value in zip(
        cognitive,
        clinical,
        y_positions,
    ):
        axis.hlines(
            y=y_value,
            xmin=min(
                cognitive_value,
                clinical_value,
            ),
            xmax=max(
                cognitive_value,
                clinical_value,
            ),
            linewidth=1.7,
            color="0.60",
            zorder=1,
        )

    axis.scatter(
        cognitive,
        y_positions,
        marker="o",
        s=56,
        facecolors="white",
        edgecolors="black",
        linewidths=1.2,
        label="Strongest cognitive",
        zorder=3,
    )

    axis.scatter(
        clinical,
        y_positions,
        marker="s",
        s=52,
        facecolors="0.35",
        edgecolors="0.35",
        linewidths=1.0,
        label="Strongest clinical",
        zorder=3,
    )

    for index, row in data.iterrows():
        y_value = float(
            y_positions[index]
        )

        cognitive_value = float(
            row[
                "strongest_cognitive_rho"
            ]
        )

        clinical_value = float(
            row[
                "strongest_clinical_rho"
            ]
        )

        axis.annotate(
            f"{cognitive_value:.2f}",
            xy=(
                cognitive_value,
                y_value,
            ),
            xytext=(
                0,
                7,
            ),
            textcoords="offset points",
            fontsize=7.3,
            ha="center",
            va="bottom",
        )

        axis.annotate(
            f"{clinical_value:.2f}",
            xy=(
                clinical_value,
                y_value,
            ),
            xytext=(
                0,
                -8,
            ),
            textcoords="offset points",
            fontsize=7.3,
            ha="center",
            va="top",
        )

        axis.text(
            clinical_column_x,
            y_value,
            compact_external_label(
                row[
                    "strongest_clinical_map"
                ]
            ),
            fontsize=7.4,
            ha="left",
            va="center",
        )

        axis.text(
            cognitive_column_x,
            y_value,
            compact_external_label(
                row[
                    "strongest_cognitive_map"
                ]
            ),
            fontsize=7.4,
            ha="right",
            va="center",
        )

    header_y = len(data) - 0.02

    axis.text(
        clinical_column_x,
        header_y,
        "Clinical map",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    axis.text(
        cognitive_column_x,
        header_y,
        "Cognitive map",
        fontsize=8,
        fontweight="bold",
        ha="right",
        va="bottom",
    )

    axis.set_xlim(
        -0.88,
        0.80,
    )

    axis.set_xlabel(
        "Spearman spatial correlation",
        labelpad=7,
    )

    axis.set_title(
        "Strongest cognitive and clinical correspondences",
        loc="left",
        fontweight="bold",
        pad=8,
    )

    axis.legend(
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            -0.20,
        ),
        ncol=2,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.3,
    )

    add_panel_label(
        axis,
        "C",
    )


def panel_d(
    axis: plt.Axes,
    priority: pd.DataFrame,
) -> None:
    plotting = priority.sort_values(
        "priority_reporting_order",
        kind="stable",
    ).reset_index(
        drop=True
    )

    y_positions = np.arange(
        len(plotting)
    )[::-1]

    rho_values = plotting[
        "spearman_rho"
    ].to_numpy(
        dtype=float
    )

    outcome_x = 0.035
    evidence_x = 0.205
    tier_x = 0.385
    robustness_x = 0.485

    axis.axvline(
        0,
        linewidth=1.0,
        color="0.35",
        zorder=0,
    )

    axis.grid(
        axis="x",
        linewidth=0.7,
        alpha=0.15,
        zorder=0,
    )

    axis.hlines(
        y=y_positions,
        xmin=rho_values,
        xmax=0,
        linewidth=1.9,
        color="0.58",
        zorder=1,
    )

    for index, row in plotting.iterrows():
        y_value = float(
            y_positions[index]
        )

        rho = float(
            row[
                "spearman_rho"
            ]
        )

        tier = str(
            row[
                "evidence_tier"
            ]
        )

        draw_tier_marker(
            axis,
            rho,
            y_value,
            tier,
        )

        axis.annotate(
            f"{rho:.3f}",
            xy=(
                rho,
                y_value,
            ),
            xytext=(
                -8,
                0,
            ),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            ha="right",
            va="center",
        )

        axis.text(
            outcome_x,
            y_value,
            compact_external_label(
                row[
                    "external_map"
                ]
            ),
            fontsize=7.6,
            ha="left",
            va="center",
        )

        axis.text(
            evidence_x,
            y_value,
            str(
                row[
                    "statistics_label"
                ]
            ).replace(
                "Global maxT ",
                "maxT "
            ).replace(
                "Family BH ",
                "BH "
            ),
            fontsize=7.6,
            ha="left",
            va="center",
        )

        axis.text(
            tier_x,
            y_value,
            str(
                row[
                    "tier_label"
                ]
            ),
            fontsize=7.6,
            ha="left",
            va="center",
        )

        axis.text(
            robustness_x,
            y_value,
            "LODO 5/5",
            fontsize=7.6,
            ha="left",
            va="center",
        )

    axis.set_yticks(
        y_positions
    )

    axis.set_yticklabels(
        plotting[
            "map_label"
        ].tolist()
    )

    axis.set_ylim(
        -0.65,
        len(plotting) + 0.20,
    )

    axis.set_xlim(
        -0.74,
        0.62,
    )

    header_y = len(plotting) - 0.01

    for x_value, label in [
        (
            outcome_x,
            "Outcome",
        ),
        (
            evidence_x,
            "Correction",
        ),
        (
            tier_x,
            "Tier",
        ),
        (
            robustness_x,
            "LODO",
        ),
    ]:
        axis.text(
            x_value,
            header_y,
            label,
            fontsize=7.3,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    axis.set_xlabel(
        "Spearman spatial correlation",
        labelpad=7,
    )

    axis.set_title(
        "Definitive priority clinical associations",
        loc="left",
        fontweight="bold",
        pad=8,
    )

    add_panel_label(
        axis,
        "D",
    )


def validate_r1_completion() -> None:
    R1.require_file(
        R1_COMPLETION
    )

    completion = pd.read_csv(
        R1_COMPLETION,
        sep="\t",
    )

    if len(completion) != 1:
        raise R1.ValidationError(
            "Phase 8D3D-R1 completion summary "
            "must contain exactly one row."
        )

    row = completion.iloc[0]

    status = str(
        row.get(
            "Phase8D3D_R1_status",
            "",
        )
    ).strip()

    if status not in {
        "completed_pending_visual_review",
        "completed",
    }:
        raise R1.ValidationError(
            "Unexpected Phase 8D3D-R1 status: "
            f"{status}"
        )

    if not R1.as_bool(
        row.get(
            "all_technical_audits_passed",
            False,
        )
    ):
        raise R1.ValidationError(
            "Phase 8D3D-R1 technical audits "
            "did not pass."
        )

    if R1.as_bool(
        row.get(
            "statistics_recomputed",
            True,
        )
    ):
        raise R1.ValidationError(
            "Phase 8D3D-R1 unexpectedly reports "
            "recomputed statistics."
        )


def tier_legend_handles() -> list[plt.Line2D]:
    return [
        plt.Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=13,
            label="Tier 1: global FWER",
        ),
        plt.Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            markerfacecolor="0.30",
            markeredgecolor="0.30",
            markersize=7,
            label="Tier 2: family corrected",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="0.25",
            markersize=7,
            label="Tier 3A: nominal",
        ),
        plt.Line2D(
            [],
            [],
            marker="x",
            linestyle="none",
            markeredgecolor="0.25",
            markersize=7,
            label="Tier 3B: descriptive",
        ),
    ]


def build_figure(
    map_data: pd.DataFrame,
    priority_data: pd.DataFrame,
) -> list[Path]:
    configure_matplotlib()

    R2_FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure = plt.figure(
        figsize=(
            16,
            12,
        ),
        constrained_layout=False,
    )

    grid = figure.add_gridspec(
        nrows=2,
        ncols=2,
        left=0.075,
        right=0.985,
        bottom=0.095,
        top=0.875,
        wspace=0.34,
        hspace=0.34,
    )

    axis_a = figure.add_subplot(
        grid[0, 0]
    )

    axis_b = figure.add_subplot(
        grid[0, 1]
    )

    axis_c = figure.add_subplot(
        grid[1, 0]
    )

    axis_d = figure.add_subplot(
        grid[1, 1]
    )

    panel_a(
        axis_a,
        map_data,
    )

    panel_b(
        axis_b,
        map_data,
    )

    panel_c(
        axis_c,
        map_data,
    )

    panel_d(
        axis_d,
        priority_data,
    )

    figure.suptitle(
        (
            "Integrated developmental and "
            "cross-modal evidence for the "
            "DTHI cortical hierarchy"
        ),
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )

    figure.legend(
        handles=tier_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.925,
        ),
        ncol=4,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.5,
        fontsize=9,
    )

    output_paths = [
        R2_FIG_BASE.with_suffix(
            ".png"
        ),
        R2_FIG_BASE.with_suffix(
            ".pdf"
        ),
        R2_FIG_BASE.with_suffix(
            ".svg"
        ),
    ]

    figure.savefig(
        output_paths[0],
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        output_paths[1],
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        output_paths[2],
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        figure
    )

    return output_paths


def build_figure_legend() -> str:
    return """Figure X. Integrated developmental, donor-robustness and functional-imaging evidence for the DTHI cortical hierarchy.

(A) Empirical peak developmental windows for the ten DTHI cortical hierarchy maps, derived from the Phase 5B2 donor-level developmental trajectory matrix. Symbols identify the highest retained map-level evidence category: Tier 1, global familywise-error-controlled evidence; Tier 2, family-level corrected evidence; Tier 3A, nominal exploratory correspondence; and Tier 3B, descriptive correspondence.

(B) Robustness of the AHBA-derived cortical maps across five leave-one-donor-out reconstructions. Open circles show the minimum and filled squares show the median Spearman correlation between each leave-one-donor-out reconstruction and the complete donor-balanced map. Horizontal segments connect the corresponding estimates.

(C) Strongest cognitive and clinical spatial correspondence for each DTHI map. Open circles and filled squares identify the cognitive and clinical external maps with the largest absolute Spearman correlation, respectively. Numerical labels show the corresponding correlation coefficients. The dedicated outer columns identify the associated external maps. These strongest correspondences are descriptive unless formal corrected support was retained in the Phase 8D3C hierarchy.

(D) Five locked priority clinical associations. The starred synaptic assembly/receptor trafficking–ASD association survived exact-bijective global maxT familywise-error correction. Filled square markers identify four secondary associations retaining clinical-family BH-FDR support but not global familywise-error control. Separate columns report the clinical outcome, corrected statistic, evidence tier and leave-one-donor-out directional robustness. All five associations were directionally concordant across the five donor omissions. None retained BH-FDR support in the nested LH46 sensitivity analysis.

Spatial correspondence represents anatomical convergence across modalities and does not establish causality, disease-specific molecular expression, mechanistic equivalence or independent replication. The LH46 analysis is a nested sensitivity analysis. Cell-system annotations remain definition-based because no compatible empirical cell-type table was available for Phase 8D3B.
"""


def provenance_record(
    source_layer: str,
    path: Path,
    detail: str,
) -> dict[str, Any]:
    R1.require_file(
        path
    )

    return {
        "source_layer":
            source_layer,
        "relative_path":
            str(
                path.relative_to(
                    ROOT
                )
            ),
        "size_bytes":
            path.stat().st_size,
        "SHA256":
            R1.sha256(path),
        "detail":
            detail,
    }


def prepare_provenance() -> pd.DataFrame:
    return pd.DataFrame(
        [
            provenance_record(
                "Phase8D3B_cross_modal_master",
                R1.D3B_MASTER,
                (
                    "developmental peaks, AHBA "
                    "donor robustness and strongest "
                    "cognitive/clinical associations"
                ),
            ),
            provenance_record(
                "Phase8D3C_association_hierarchy",
                R1.D3C_ASSOC,
                (
                    "definitive classification of "
                    "all 160 spatial associations"
                ),
            ),
            provenance_record(
                "Phase8D3C_map_hierarchy",
                R1.D3C_MAPS,
                (
                    "highest retained evidence "
                    "tier for each DTHI map"
                ),
            ),
            provenance_record(
                "Phase8D3C_priority_reporting",
                R1.D3C_PRIORITY,
                (
                    "five locked priority clinical "
                    "associations and corrected statistics"
                ),
            ),
            provenance_record(
                "Phase8D3D_R1_completion",
                R1_COMPLETION,
                (
                    "technically valid R1 figure "
                    "superseded visually by R2"
                ),
            ),
            provenance_record(
                "Phase8D3D_R1_script",
                R1_SCRIPT,
                (
                    "validated data-loading and "
                    "figure-preparation implementation"
                ),
            ),
        ]
    )


def output_record(
    path: Path,
) -> dict[str, Any]:
    return {
        "relative_path":
            str(
                path.relative_to(
                    ROOT
                )
            ),
        "file_format":
            path.suffix.lower().lstrip(
                "."
            ),
        "size_bytes":
            path.stat().st_size,
        "SHA256":
            R1.sha256(path),
    }


def main() -> None:
    R1.validate_upstream()
    validate_r1_completion()

    (
        master,
        associations,
        maps,
        priority,
    ) = R1.load_inputs()

    map_data = R1.prepare_map_data(
        master,
        maps,
    )

    priority_data = R1.prepare_priority_data(
        priority
    )

    R2_OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    R2_FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    R1.PHASE8_TABLES.mkdir(
        parents=True,
        exist_ok=True,
    )

    R1.write_tsv(
        map_data,
        R2_MAP_DATA,
    )

    R1.write_tsv(
        priority_data,
        R2_PRIORITY_DATA,
    )

    provenance = prepare_provenance()

    R1.write_tsv(
        provenance,
        R2_PROVENANCE,
    )

    R2_LEGEND.write_text(
        build_figure_legend(),
        encoding="utf-8",
    )

    figure_paths = build_figure(
        map_data,
        priority_data,
    )

    output_manifest = pd.DataFrame(
        [
            output_record(
                path
            )
            for path in figure_paths
        ]
    )

    R1.write_tsv(
        output_manifest,
        R2_MANIFEST,
    )

    association_tier_counts = (
        associations[
            "evidence_tier"
        ]
        .value_counts()
        .to_dict()
    )

    map_tier_counts = (
        map_data[
            "map_level_highest_tier"
        ]
        .value_counts()
        .to_dict()
    )

    expected_association_tiers = bool(
        association_tier_counts.get(
            "Tier1_global_FWER",
            0,
        ) == 1
        and association_tier_counts.get(
            "Tier2_family_corrected",
            0,
        ) == 4
        and association_tier_counts.get(
            "Tier3A_nominal",
            0,
        ) == 13
        and association_tier_counts.get(
            "Tier3B_descriptive",
            0,
        ) == 142
    )

    expected_map_tiers = bool(
        map_tier_counts.get(
            "Tier1_global_FWER",
            0,
        ) == 1
        and map_tier_counts.get(
            "Tier2_family_corrected",
            0,
        ) == 3
    )

    primary = priority_data[
        priority_data[
            "evidence_tier"
        ].eq(
            "Tier1_global_FWER"
        )
    ]

    primary_pair_correct = bool(
        len(primary) == 1
        and str(
            primary.iloc[0][
                "DTHI_map"
            ]
        )
        == (
            "synaptic_assembly_"
            "receptor_trafficking"
        )
        and str(
            primary.iloc[0][
                "external_map"
            ]
        )
        == "clinical_asd"
    )

    priority_structure_correct = bool(
        len(priority_data) == 5
        and int(
            priority_data[
                "evidence_tier"
            ].eq(
                "Tier1_global_FWER"
            ).sum()
        ) == 1
        and int(
            priority_data[
                "evidence_tier"
            ].eq(
                "Tier2_family_corrected"
            ).sum()
        ) == 4
        and primary_pair_correct
    )

    developmental_complete = bool(
        map_data[
            "empirical_peak_developmental_window"
        ]
        .notna()
        .all()
    )

    ahba_complete = bool(
        map_data[
            [
                "AHBA_minimum_LODO_vs_full_rho",
                "AHBA_median_LODO_vs_full_rho",
            ]
        ]
        .notna()
        .all()
        .all()
    )

    cross_modal_complete = bool(
        map_data[
            [
                "strongest_cognitive_map",
                "strongest_cognitive_rho",
                "strongest_clinical_map",
                "strongest_clinical_rho",
            ]
        ]
        .notna()
        .all()
        .all()
    )

    priority_lodo_complete = bool(
        priority_data[
            "LODO_5of5_direction_concordant"
        ].all()
    )

    priority_lh46_count = int(
        priority_data[
            "LH46_BH_supported"
        ].sum()
    )

    figure_validity = {
        path.suffix.lower():
            bool(
                path.is_file()
                and path.stat().st_size
                >= 1000
            )
        for path in figure_paths
    }

    r1_png = (
        ROOT
        / "08_figures"
        / "phase8"
        / "phase8D3D_R1"
        / "phase8D3D_R1_integrated_cross_modal_figure.png"
    )

    original_png = (
        ROOT
        / "08_figures"
        / "phase8"
        / "phase8D3D"
        / "phase8D3D_integrated_cross_modal_figure.png"
    )

    prior_figures_preserved = bool(
        r1_png.is_file()
        and original_png.is_file()
    )

    marker_styles_complete = bool(
        set(TIER_STYLE)
        == set(
            R1.TIER_ORDER
        )
    )

    audit = pd.DataFrame(
        [
            {
                "section":
                    "upstream",
                "item":
                    "Phase8D3C_completed",
                "value":
                    True,
                "passed":
                    True,
                "detail":
                    str(
                        R1.D3C_COMPLETION.relative_to(
                            ROOT
                        )
                    ),
            },
            {
                "section":
                    "upstream",
                "item":
                    "Phase8D3D_R1_completed",
                "value":
                    True,
                "passed":
                    True,
                "detail":
                    str(
                        R1_COMPLETION.relative_to(
                            ROOT
                        )
                    ),
            },
            {
                "section":
                    "preservation",
                "item":
                    "original_and_R1_figures_preserved",
                "value":
                    prior_figures_preserved,
                "passed":
                    prior_figures_preserved,
                "detail":
                    (
                        "R2 creates separate outputs "
                        "without deleting earlier figures"
                    ),
            },
            {
                "section":
                    "dimensions",
                "item":
                    "DTHI_maps",
                "value":
                    len(map_data),
                "passed":
                    len(map_data) == 10,
                "detail":
                    "one row per DTHI map",
            },
            {
                "section":
                    "dimensions",
                "item":
                    "spatial_associations",
                "value":
                    len(associations),
                "passed":
                    len(associations) == 160,
                "detail":
                    "frozen Phase 8D3C hierarchy",
            },
            {
                "section":
                    "dimensions",
                "item":
                    "priority_associations",
                "value":
                    len(priority_data),
                "passed":
                    len(priority_data) == 5,
                "detail":
                    "one primary and four secondary",
            },
            {
                "section":
                    "evidence",
                "item":
                    "association_tier_counts",
                "value":
                    json.dumps(
                        association_tier_counts,
                        sort_keys=True,
                    ),
                "passed":
                    expected_association_tiers,
                "detail":
                    (
                        "expected 1 Tier1, 4 Tier2, "
                        "13 Tier3A and 142 Tier3B"
                    ),
            },
            {
                "section":
                    "evidence",
                "item":
                    "map_tier_counts",
                "value":
                    json.dumps(
                        map_tier_counts,
                        sort_keys=True,
                    ),
                "passed":
                    expected_map_tiers,
                "detail":
                    (
                        "expected one Tier1 map "
                        "and three Tier2 maps"
                    ),
            },
            {
                "section":
                    "evidence",
                "item":
                    "priority_structure",
                "value":
                    priority_structure_correct,
                "passed":
                    priority_structure_correct,
                "detail":
                    (
                        "primary pair is synaptic "
                        "assembly/receptor "
                        "trafficking–ASD"
                    ),
            },
            {
                "section":
                    "panel_A",
                "item":
                    "developmental_data_complete",
                "value":
                    developmental_complete,
                "passed":
                    developmental_complete,
                "detail":
                    "empirical Phase 5B2 layer",
            },
            {
                "section":
                    "panel_B",
                "item":
                    "AHBA_LODO_data_complete",
                "value":
                    ahba_complete,
                "passed":
                    ahba_complete,
                "detail":
                    (
                        "minimum and median "
                        "LODO-vs-full correlations"
                    ),
            },
            {
                "section":
                    "panel_C",
                "item":
                    "cross_modal_data_complete",
                "value":
                    cross_modal_complete,
                "passed":
                    cross_modal_complete,
                "detail":
                    (
                        "external-map names use "
                        "dedicated clinical and "
                        "cognitive columns"
                    ),
            },
            {
                "section":
                    "panel_D",
                "item":
                    "priority_LODO_5of5",
                "value":
                    priority_lodo_complete,
                "passed":
                    priority_lodo_complete,
                "detail":
                    (
                        "directional concordance "
                        "across five donor omissions"
                    ),
            },
            {
                "section":
                    "panel_D",
                "item":
                    "priority_LH46_BH_hits",
                "value":
                    priority_lh46_count,
                "passed":
                    priority_lh46_count == 0,
                "detail":
                    (
                        "expected zero; nested "
                        "sensitivity analysis only"
                    ),
            },
            {
                "section":
                    "visual_revision",
                "item":
                    "consistent_tier_marker_styles",
                "value":
                    marker_styles_complete,
                "passed":
                    marker_styles_complete,
                "detail":
                    (
                        "all panels use one "
                        "monochrome evidence-style key"
                    ),
            },
            {
                "section":
                    "visual_revision",
                "item":
                    "panel_C_dedicated_label_columns",
                "value":
                    True,
                "passed":
                    True,
                "detail":
                    (
                        "clinical and cognitive "
                        "map names separated from points"
                    ),
            },
            {
                "section":
                    "visual_revision",
                "item":
                    "panel_D_dedicated_evidence_columns",
                "value":
                    True,
                "passed":
                    True,
                "detail":
                    (
                        "outcome, correction, tier "
                        "and robustness shown separately"
                    ),
            },
            {
                "section":
                    "figure_outputs",
                "item":
                    "PNG_created",
                "value":
                    figure_validity.get(
                        ".png",
                        False,
                    ),
                "passed":
                    figure_validity.get(
                        ".png",
                        False,
                    ),
                "detail":
                    str(
                        figure_paths[0].relative_to(
                            ROOT
                        )
                    ),
            },
            {
                "section":
                    "figure_outputs",
                "item":
                    "PDF_created",
                "value":
                    figure_validity.get(
                        ".pdf",
                        False,
                    ),
                "passed":
                    figure_validity.get(
                        ".pdf",
                        False,
                    ),
                "detail":
                    str(
                        figure_paths[1].relative_to(
                            ROOT
                        )
                    ),
            },
            {
                "section":
                    "figure_outputs",
                "item":
                    "SVG_created",
                "value":
                    figure_validity.get(
                        ".svg",
                        False,
                    ),
                "passed":
                    figure_validity.get(
                        ".svg",
                        False,
                    ),
                "detail":
                    str(
                        figure_paths[2].relative_to(
                            ROOT
                        )
                    ),
            },
            {
                "section":
                    "interpretation",
                "item":
                    "statistics_recomputed",
                "value":
                    False,
                "passed":
                    True,
                "detail":
                    "frozen Phase 8D3C statistics",
            },
            {
                "section":
                    "interpretation",
                "item":
                    "visual_signoff_pending",
                "value":
                    True,
                "passed":
                    True,
                "detail":
                    (
                        "R2 PNG requires direct "
                        "visual inspection"
                    ),
            },
        ]
    )

    R1.write_tsv(
        audit,
        R2_AUDIT,
    )

    all_technical_audits_passed = bool(
        audit[
            "passed"
        ].all()
    )

    completion = pd.DataFrame(
        [
            {
                "Phase8D3C_status_confirmed":
                    True,
                "Phase8D3D_R1_status_confirmed":
                    True,
                "prior_figures_preserved":
                    prior_figures_preserved,
                "DTHI_maps_in_figure":
                    len(map_data),
                "associations_available":
                    len(associations),
                "priority_associations_in_panel_D":
                    len(priority_data),
                "figure_panels":
                    4,
                "PNG_created":
                    figure_validity.get(
                        ".png",
                        False,
                    ),
                "PDF_created":
                    figure_validity.get(
                        ".pdf",
                        False,
                    ),
                "SVG_created":
                    figure_validity.get(
                        ".svg",
                        False,
                    ),
                "statistics_recomputed":
                    False,
                "all_technical_audits_passed":
                    all_technical_audits_passed,
                "ready_for_visual_review":
                    all_technical_audits_passed,
                "manuscript_ready_visual_signoff":
                    False,
                "supersedes_original_and_R1_visuals":
                    True,
                "Phase8D3D_R2_status":
                    (
                        "completed_pending_visual_review"
                        if all_technical_audits_passed
                        else "failed"
                    ),
                "python_version":
                    sys.version.split()[0],
                "numpy_version":
                    np.__version__,
                "pandas_version":
                    pd.__version__,
                "matplotlib_version":
                    matplotlib.__version__,
            }
        ]
    )

    R1.write_tsv(
        completion,
        R2_COMPLETION,
    )

    print(
        "===== PHASE 8D3D-R2 COMPLETION ====="
    )

    print(
        completion.to_string(
            index=False
        )
    )

    print(
        "\n===== R2 FIGURE OUTPUTS ====="
    )

    print(
        output_manifest.to_string(
            index=False
        )
    )

    print(
        "\n===== R2 TECHNICAL AUDIT ====="
    )

    print(
        audit.to_string(
            index=False
        )
    )

    print(
        "\n===== R2 PRIORITY PANEL ====="
    )

    print(
        priority_data[
            [
                "priority_reporting_order",
                "DTHI_map",
                "external_map",
                "spearman_rho",
                "evidence_tier",
                "statistics_label",
                "LODO_5of5_direction_concordant",
                "LH46_BH_supported",
            ]
        ].to_string(
            index=False
        )
    )

    canonical_outputs = [
        R2_MAP_DATA,
        R2_PRIORITY_DATA,
        R2_PROVENANCE,
        R2_LEGEND,
        R2_MANIFEST,
        R2_AUDIT,
        R2_COMPLETION,
        *figure_paths,
    ]

    output_hashes = pd.DataFrame(
        [
            {
                "relative_path":
                    str(
                        path.relative_to(
                            ROOT
                        )
                    ),
                "size_bytes":
                    path.stat().st_size,
                "SHA256":
                    R1.sha256(path),
            }
            for path in canonical_outputs
        ]
    )

    print(
        "\n===== R2 CANONICAL OUTPUT HASHES ====="
    )

    print(
        output_hashes.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            "Phase 8D3D-R2 failed: "
            f"{type(error).__name__}: "
            f"{error}",
            file=sys.stderr,
        )
        raise
