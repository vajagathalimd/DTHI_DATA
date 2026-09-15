#!/usr/bin/env python3
"""Phase 8D3D-R1: revised integrated cross-modal figure."""

from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

PHASE8_TABLES = ROOT / "07_tables" / "main_tables" / "phase8"
PHASE8_FIGURES = ROOT / "08_figures" / "phase8"

D3C_COMPLETION = PHASE8_TABLES / "phase8D3C_completion_summary.tsv"
D3D_COMPLETION = PHASE8_TABLES / "phase8D3D_completion_summary.tsv"

D3B_MASTER = (
    ROOT
    / "03_processed_data"
    / "functional_imaging"
    / "phase8D"
    / "phase8D3B_cross_modal_synthesis"
    / "phase8D3B_cross_modal_master_matrix.tsv"
)

D3C_ASSOC = (
    ROOT
    / "03_processed_data"
    / "functional_imaging"
    / "phase8D"
    / "phase8D3C_evidence_hierarchy"
    / "phase8D3C_association_evidence_hierarchy.tsv"
)

D3C_MAPS = (
    ROOT
    / "03_processed_data"
    / "functional_imaging"
    / "phase8D"
    / "phase8D3C_evidence_hierarchy"
    / "phase8D3C_map_level_evidence_hierarchy.tsv"
)

D3C_PRIORITY = (
    ROOT
    / "03_processed_data"
    / "functional_imaging"
    / "phase8D"
    / "phase8D3C_evidence_hierarchy"
    / "phase8D3C_priority_reporting_table.tsv"
)

R1_OUT = (
    ROOT
    / "03_processed_data"
    / "functional_imaging"
    / "phase8D"
    / "phase8D3D_R1"
)

R1_FIG_DIR = PHASE8_FIGURES / "phase8D3D_R1"
R1_FIG_BASE = R1_FIG_DIR / "phase8D3D_R1_integrated_cross_modal_figure"

R1_FIG_DATA = R1_OUT / "phase8D3D_R1_figure_data.tsv"
R1_PRIORITY_DATA = R1_OUT / "phase8D3D_R1_priority_panel_data.tsv"
R1_PROVENANCE = R1_OUT / "phase8D3D_R1_source_provenance.tsv"
R1_LEGEND = R1_OUT / "phase8D3D_R1_figure_legend.txt"
R1_MANIFEST = R1_OUT / "phase8D3D_R1_figure_output_manifest.tsv"

R1_AUDIT = PHASE8_TABLES / "phase8D3D_R1_figure_audit.tsv"
R1_COMPLETION = PHASE8_TABLES / "phase8D3D_R1_completion_summary.tsv"

DTHI_ORDER = [
    "patterning_arealization",
    "progenitor_radial_glia",
    "neurogenesis_migration_layering",
    "axon_guidance_neurite_outgrowth",
    "synaptic_assembly_receptor_trafficking",
    "astrocyte_maturation_metabolic_support",
    "oligodendrocyte_myelination",
    "activity_dependent_plasticity",
    "synaptic_membrane_structural_candidates",
    "preliminary_maturation_balance",
]

DTHI_SHORT = {
    "patterning_arealization": "Patterning",
    "progenitor_radial_glia": "Prog./RG",
    "neurogenesis_migration_layering": "Neurogenesis/\nMigration",
    "axon_guidance_neurite_outgrowth": "Axon guidance/\nNeurites",
    "synaptic_assembly_receptor_trafficking": "Synaptic assembly/\nReceptor trafficking",
    "astrocyte_maturation_metabolic_support": "Astrocyte maturation/\nMetabolic support",
    "oligodendrocyte_myelination": "Oligodendrocyte/\nMyelination",
    "activity_dependent_plasticity": "Activity-dependent\nPlasticity",
    "synaptic_membrane_structural_candidates": "Synaptic-membrane\nCandidates",
    "preliminary_maturation_balance": "Preliminary\nMaturation balance",
}

WINDOW_ORDER = [
    "early_fetal_patterning",
    "mid_fetal_neurogenesis",
    "late_fetal_synaptogenesis",
    "infancy_postnatal_circuit_formation",
    "childhood_circuit_refinement",
    "adolescence_myelination_refinement",
    "adult_stabilization",
]

WINDOW_SHORT = {
    "early_fetal_patterning": "Early fetal",
    "mid_fetal_neurogenesis": "Mid-fetal",
    "late_fetal_synaptogenesis": "Late fetal",
    "infancy_postnatal_circuit_formation": "Infancy",
    "childhood_circuit_refinement": "Childhood",
    "adolescence_myelination_refinement": "Adolescence",
    "adult_stabilization": "Adult",
}

TIER_ORDER = {
    "Tier1_global_FWER": 1,
    "Tier2_family_corrected": 2,
    "Tier3A_nominal": 3,
    "Tier3B_descriptive": 4,
}

TIER_LABEL = {
    "Tier1_global_FWER": "Tier 1",
    "Tier2_family_corrected": "Tier 2",
    "Tier3A_nominal": "Tier 3A",
    "Tier3B_descriptive": "Tier 3B",
}


class ValidationError(RuntimeError):
    pass


def require_file(path: Path) -> None:
    if not path.is_file():
        raise ValidationError(f"Required file not found: {path}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, sep="\t", index=False)
    tmp.replace(path)


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def wrap_text(value: Any, width: int = 24) -> str:
    return "\n".join(
        textwrap.wrap(
            str(value),
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def external_label(value: Any) -> str:
    text = str(value)
    text = text.replace("cognitive_", "")
    text = text.replace("clinical_", "")
    text = text.replace("_", " ")

    special = {
        "asd": "ASD",
        "adhd": "ADHD",
        "ocd": "OCD",
        "parkinsons": "Parkinson's disease",
        "reward motivation": "Reward motivation",
        "executive control": "Executive control",
        "social cognition": "Social cognition",
        "emotion affect": "Emotion/affect",
        "sensorimotor": "Sensorimotor",
    }

    if text in special:
        return special[text]

    return text.title()


def validate_completion_table(
    path: Path,
    status_column: str,
    expected_status: set[str],
) -> pd.Series:
    require_file(path)

    frame = pd.read_csv(
        path,
        sep="\t",
    )

    if len(frame) != 1:
        raise ValidationError(
            f"{path.name} must contain exactly one row."
        )

    row = frame.iloc[0]

    status = str(
        row.get(
            status_column,
            "",
        )
    ).strip()

    if status not in expected_status:
        raise ValidationError(
            f"Unexpected {status_column}: {status}"
        )

    return row


def validate_upstream() -> None:
    row_3c = validate_completion_table(
        D3C_COMPLETION,
        "Phase8D3C_status",
        {"completed"},
    )

    if not as_bool(
        row_3c.get(
            "all_evidence_hierarchy_audits_passed",
            False,
        )
    ):
        raise ValidationError(
            "Phase 8D3C evidence-hierarchy audits did not pass."
        )

    if not as_bool(
        row_3c.get(
            "ready_for_phase8D3D_integrated_figure",
            False,
        )
    ):
        raise ValidationError(
            "Phase 8D3C is not ready for figure generation."
        )

    row_3d = validate_completion_table(
        D3D_COMPLETION,
        "Phase8D3D_status",
        {
            "completed_pending_visual_review",
            "completed",
        },
    )

    if not as_bool(
        row_3d.get(
            "all_technical_audits_passed",
            False,
        )
    ):
        raise ValidationError(
            "Original Phase 8D3D technical audits did not pass."
        )

    if as_bool(
        row_3d.get(
            "statistics_recomputed",
            True,
        )
    ):
        raise ValidationError(
            "Original Phase 8D3D unexpectedly reports recomputed statistics."
        )


def numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in frame.columns:
        raise ValidationError(
            f"Required numeric column not found: {column}"
        )

    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    )

    if values.isna().any():
        raise ValidationError(
            f"Column contains missing or nonnumeric values: {column}"
        )

    return values


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    input_paths = [
        D3B_MASTER,
        D3C_ASSOC,
        D3C_MAPS,
        D3C_PRIORITY,
    ]

    for path in input_paths:
        require_file(path)

    master = pd.read_csv(
        D3B_MASTER,
        sep="\t",
        low_memory=False,
    )

    associations = pd.read_csv(
        D3C_ASSOC,
        sep="\t",
        low_memory=False,
    )

    maps = pd.read_csv(
        D3C_MAPS,
        sep="\t",
        low_memory=False,
    )

    priority = pd.read_csv(
        D3C_PRIORITY,
        sep="\t",
        low_memory=False,
    )

    expected_sizes = {
        "master": (
            len(master),
            10,
        ),
        "associations": (
            len(associations),
            160,
        ),
        "maps": (
            len(maps),
            10,
        ),
        "priority": (
            len(priority),
            5,
        ),
    }

    for name, (
        observed,
        expected,
    ) in expected_sizes.items():
        if observed != expected:
            raise ValidationError(
                f"Expected {expected} {name} rows; found {observed}."
            )

    observed_maps = set(
        master[
            "DTHI_map"
        ].astype(str)
    )

    if observed_maps != set(DTHI_ORDER):
        raise ValidationError(
            "The Phase 8D3B master matrix does not contain "
            "the expected ten DTHI maps."
        )

    if set(
        maps[
            "DTHI_map"
        ].astype(str)
    ) != set(DTHI_ORDER):
        raise ValidationError(
            "The Phase 8D3C map hierarchy does not contain "
            "the expected ten DTHI maps."
        )

    return (
        master,
        associations,
        maps,
        priority,
    )


def prepare_map_data(
    master: pd.DataFrame,
    maps: pd.DataFrame,
) -> pd.DataFrame:
    master_columns = {
        "DTHI_map",
        "AHBA_minimum_LODO_vs_full_rho",
        "AHBA_median_LODO_vs_full_rho",
        "strongest_cognitive_map",
        "strongest_cognitive_rho",
        "strongest_clinical_map",
        "strongest_clinical_rho",
        "empirical_peak_developmental_window",
        "developmental_empirical_layer_status",
    }

    missing_master = (
        master_columns
        - set(master.columns)
    )

    if missing_master:
        raise ValidationError(
            "Phase 8D3B master matrix is missing columns: "
            f"{sorted(missing_master)}"
        )

    map_columns = {
        "DTHI_map",
        "map_level_highest_tier",
        "priority_association_count",
        "cross_modal_convergence_class",
    }

    missing_maps = (
        map_columns
        - set(maps.columns)
    )

    if missing_maps:
        raise ValidationError(
            "Phase 8D3C map hierarchy is missing columns: "
            f"{sorted(missing_maps)}"
        )

    map_subset = maps[
        [
            "DTHI_map",
            "map_level_highest_tier",
            "priority_association_count",
            "cross_modal_convergence_class",
        ]
    ].copy()

    data = master.merge(
        map_subset,
        on="DTHI_map",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_D3C",
        ),
    )

    for column in [
        "AHBA_minimum_LODO_vs_full_rho",
        "AHBA_median_LODO_vs_full_rho",
        "strongest_cognitive_rho",
        "strongest_clinical_rho",
    ]:
        data[column] = numeric(
            data,
            column,
        )

    data[
        "priority_association_count"
    ] = pd.to_numeric(
        data[
            "priority_association_count_D3C"
        ]
        if "priority_association_count_D3C"
        in data.columns
        else data[
            "priority_association_count"
        ],
        errors="coerce",
    )

    if data[
        "priority_association_count"
    ].isna().any():
        raise ValidationError(
            "Priority-association counts could not be parsed."
        )

    data[
        "map_order"
    ] = data[
        "DTHI_map"
    ].map(
        {
            name: index
            for index, name
            in enumerate(DTHI_ORDER)
        }
    )

    data[
        "window_order"
    ] = data[
        "empirical_peak_developmental_window"
    ].map(
        {
            name: index
            for index, name
            in enumerate(WINDOW_ORDER)
        }
    )

    data[
        "tier_order"
    ] = data[
        "map_level_highest_tier"
    ].map(
        TIER_ORDER
    )

    required_nonmissing = [
        "map_order",
        "window_order",
        "tier_order",
    ]

    if data[
        required_nonmissing
    ].isna().any().any():
        problem = data.loc[
            data[
                required_nonmissing
            ].isna().any(axis=1),
            [
                "DTHI_map",
                "empirical_peak_developmental_window",
                "map_level_highest_tier",
            ],
        ]

        raise ValidationError(
            "Unrecognized map, developmental window, or evidence tier:\n"
            + problem.to_string(index=False)
        )

    data[
        "map_label"
    ] = data[
        "DTHI_map"
    ].map(
        DTHI_SHORT
    )

    data[
        "window_label"
    ] = data[
        "empirical_peak_developmental_window"
    ].map(
        WINDOW_SHORT
    )

    data[
        "tier_label"
    ] = data[
        "map_level_highest_tier"
    ].map(
        TIER_LABEL
    )

    data[
        "cognitive_label"
    ] = data[
        "strongest_cognitive_map"
    ].map(
        external_label
    )

    data[
        "clinical_label"
    ] = data[
        "strongest_clinical_map"
    ].map(
        external_label
    )

    data = data.sort_values(
        "map_order",
        kind="stable",
    ).reset_index(
        drop=True
    )

    return data


def prepare_priority_data(
    priority: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "priority_reporting_order",
        "priority_role",
        "DTHI_map",
        "external_map",
        "spearman_rho",
        "evidence_tier",
        "bijective_p_spin_two_sided",
        "bijective_q_BH_family80",
        "bijective_p_maxT_global160",
        "bijective_p_maxT_family80",
        "LODO_5of5_direction_concordant",
        "LH46_BH_supported",
    }

    missing = (
        required
        - set(priority.columns)
    )

    if missing:
        raise ValidationError(
            "Priority table is missing columns: "
            f"{sorted(missing)}"
        )

    data = priority.copy()

    for column in [
        "priority_reporting_order",
        "spearman_rho",
        "bijective_p_spin_two_sided",
        "bijective_q_BH_family80",
        "bijective_p_maxT_global160",
        "bijective_p_maxT_family80",
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    if data[
        [
            "priority_reporting_order",
            "spearman_rho",
            "bijective_p_spin_two_sided",
            "bijective_q_BH_family80",
            "bijective_p_maxT_global160",
            "bijective_p_maxT_family80",
        ]
    ].isna().any().any():
        raise ValidationError(
            "Priority statistics contain missing or nonnumeric values."
        )

    data[
        "LODO_5of5_direction_concordant"
    ] = data[
        "LODO_5of5_direction_concordant"
    ].map(
        as_bool
    )

    data[
        "LH46_BH_supported"
    ] = data[
        "LH46_BH_supported"
    ].map(
        as_bool
    )

    data[
        "map_label"
    ] = data[
        "DTHI_map"
    ].map(
        DTHI_SHORT
    )

    data[
        "external_label"
    ] = data[
        "external_map"
    ].map(
        external_label
    )

    data[
        "tier_label"
    ] = data[
        "evidence_tier"
    ].map(
        TIER_LABEL
    )

    data[
        "statistics_label"
    ] = np.where(
        data[
            "evidence_tier"
        ].eq(
            "Tier1_global_FWER"
        ),
        data[
            "bijective_p_maxT_global160"
        ].map(
            lambda value:
                f"Global maxT P={value:.3f}"
        ),
        data[
            "bijective_q_BH_family80"
        ].map(
            lambda value:
                f"Family BH q={value:.3f}"
        ),
    )

    data = data.sort_values(
        "priority_reporting_order",
        kind="stable",
    ).reset_index(
        drop=True
    )

    return data


TIER_MARKER = {
    "Tier1_global_FWER": "*",
    "Tier2_family_corrected": "s",
    "Tier3A_nominal": "o",
    "Tier3B_descriptive": "x",
}

TIER_SIZE = {
    "Tier1_global_FWER": 210,
    "Tier2_family_corrected": 100,
    "Tier3A_nominal": 72,
    "Tier3B_descriptive": 58,
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
            "figure.titlesize": 15,
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
        1.05,
        label,
        transform=axis.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def apply_common_y_axis(
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
        len(data) - 0.3,
    )

    axis.tick_params(
        axis="y",
        pad=4,
    )

    return y_positions


def panel_a_developmental_timing(
    axis: plt.Axes,
    data: pd.DataFrame,
) -> None:
    y_positions = apply_common_y_axis(
        axis,
        data,
    )

    axis.grid(
        axis="x",
        linewidth=0.7,
        alpha=0.20,
        zorder=0,
    )

    for tier in TIER_ORDER:
        subset = data[
            data[
                "map_level_highest_tier"
            ].eq(tier)
        ]

        if subset.empty:
            continue

        subset_positions = [
            y_positions[
                int(index)
            ]
            for index in subset.index
        ]

        axis.scatter(
            subset[
                "window_order"
            ],
            subset_positions,
            marker=TIER_MARKER[
                tier
            ],
            s=TIER_SIZE[
                tier
            ],
            linewidths=1.2,
            zorder=3,
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

        axis.annotate(
            row[
                "tier_label"
            ],
            xy=(
                x_value,
                y_value,
            ),
            xytext=(
                8,
                0,
            ),
            textcoords="offset points",
            fontsize=7.5,
            va="center",
            ha="left",
        )

    axis.set_xticks(
        np.arange(
            len(WINDOW_ORDER)
        )
    )

    axis.set_xticklabels(
        [
            WINDOW_SHORT[
                window
            ]
            for window in WINDOW_ORDER
        ],
        rotation=30,
        ha="right",
    )

    axis.set_xlim(
        -0.45,
        len(WINDOW_ORDER) - 0.25,
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


def panel_b_ahba_stability(
    axis: plt.Axes,
    data: pd.DataFrame,
) -> None:
    y_positions = apply_common_y_axis(
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
        alpha=0.20,
        zorder=0,
    )

    axis.hlines(
        y=y_positions,
        xmin=minimum,
        xmax=median,
        linewidth=2.2,
        alpha=0.65,
        zorder=1,
    )

    axis.scatter(
        minimum,
        y_positions,
        marker="o",
        s=52,
        label="Minimum LODO",
        zorder=3,
    )

    axis.scatter(
        median,
        y_positions,
        marker="s",
        s=52,
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

    lower_limit = min(
        0.68,
        float(
            np.nanmin(
                minimum
            )
        ) - 0.03,
    )

    axis.set_xlim(
        lower_limit,
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
            -0.21,
        ),
        ncol=2,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.2,
    )

    add_panel_label(
        axis,
        "B",
    )


def panel_c_cross_modal(
    axis: plt.Axes,
    data: pd.DataFrame,
) -> None:
    y_positions = apply_common_y_axis(
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

    axis.axvline(
        0,
        linewidth=1.0,
        alpha=0.65,
        zorder=0,
    )

    axis.grid(
        axis="x",
        linewidth=0.7,
        alpha=0.18,
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
            linewidth=1.8,
            alpha=0.60,
            zorder=1,
        )

    axis.scatter(
        cognitive,
        y_positions,
        marker="o",
        s=55,
        label="Strongest cognitive",
        zorder=3,
    )

    axis.scatter(
        clinical,
        y_positions,
        marker="s",
        s=55,
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
            fontsize=7.2,
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
            fontsize=7.2,
            ha="center",
            va="top",
        )

        axis.text(
            -0.82,
            y_value,
            (
                "Clinical: "
                + str(
                    row[
                        "clinical_label"
                    ]
                )
            ),
            fontsize=7.4,
            ha="left",
            va="center",
        )

        axis.text(
            0.68,
            y_value,
            (
                "Cognitive: "
                + str(
                    row[
                        "cognitive_label"
                    ]
                )
            ),
            fontsize=7.4,
            ha="right",
            va="center",
        )

    axis.axvline(
        -0.70,
        linewidth=0.7,
        linestyle=":",
        alpha=0.45,
    )

    axis.axvline(
        0.54,
        linewidth=0.7,
        linestyle=":",
        alpha=0.45,
    )

    axis.set_xlim(
        -0.86,
        0.72,
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
            -0.21,
        ),
        ncol=2,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.2,
    )

    add_panel_label(
        axis,
        "C",
    )


def panel_d_priority(
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

    axis.axvline(
        0,
        linewidth=1.0,
        alpha=0.70,
        zorder=0,
    )

    axis.grid(
        axis="x",
        linewidth=0.7,
        alpha=0.18,
        zorder=0,
    )

    axis.hlines(
        y=y_positions,
        xmin=rho_values,
        xmax=0,
        linewidth=2.0,
        alpha=0.60,
        zorder=1,
    )

    for index, row in plotting.iterrows():
        tier = str(
            row[
                "evidence_tier"
            ]
        )

        y_value = float(
            y_positions[index]
        )

        rho = float(
            row[
                "spearman_rho"
            ]
        )

        axis.scatter(
            [rho],
            [y_value],
            marker=TIER_MARKER[
                tier
            ],
            s=TIER_SIZE[
                tier
            ],
            linewidths=1.2,
            zorder=3,
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

        evidence_text = (
            f"{row['external_label']}  |  "
            f"{row['statistics_label']}  |  "
            f"{row['tier_label']}  |  "
            "LODO 5/5"
        )

        axis.text(
            0.035,
            y_value,
            evidence_text,
            fontsize=8,
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
        len(plotting) - 0.15,
    )

    axis.set_xlim(
        -0.74,
        0.44,
    )

    axis.text(
        0.035,
        len(plotting) - 0.03,
        "Outcome | corrected evidence | tier | robustness",
        fontsize=8,
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

    tier_handles = [
        plt.Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            markersize=13,
            label="Tier 1: global FWER",
        ),
        plt.Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            markersize=7,
            label="Tier 2: family corrected",
        ),
    ]

    axis.legend(
        handles=tier_handles,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            -0.25,
        ),
        ncol=2,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.4,
    )

    add_panel_label(
        axis,
        "D",
    )


def build_figure(
    map_data: pd.DataFrame,
    priority_data: pd.DataFrame,
) -> list[Path]:
    configure_matplotlib()

    R1_FIG_DIR.mkdir(
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
        bottom=0.10,
        top=0.88,
        wspace=0.34,
        hspace=0.37,
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

    panel_a_developmental_timing(
        axis_a,
        map_data,
    )

    panel_b_ahba_stability(
        axis_b,
        map_data,
    )

    panel_c_cross_modal(
        axis_c,
        map_data,
    )

    panel_d_priority(
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

    tier_key_handles = [
        plt.Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            markersize=13,
            label="Tier 1: global FWER",
        ),
        plt.Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            markersize=7,
            label="Tier 2: family corrected",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=6,
            label="Tier 3A: nominal",
        ),
        plt.Line2D(
            [],
            [],
            marker="x",
            linestyle="none",
            markersize=7,
            label="Tier 3B: descriptive",
        ),
    ]

    figure.legend(
        handles=tier_key_handles,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.925,
        ),
        ncol=4,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.4,
        fontsize=9,
    )

    output_paths = [
        R1_FIG_BASE.with_suffix(
            ".png"
        ),
        R1_FIG_BASE.with_suffix(
            ".pdf"
        ),
        R1_FIG_BASE.with_suffix(
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

(A) Empirical peak developmental windows for the ten DTHI cortical hierarchy maps, derived from the Phase 5B2 donor-level developmental trajectory matrix. Symbols identify the highest retained evidence category for each map: Tier 1, global familywise-error-controlled evidence; Tier 2, family-level corrected evidence; Tier 3A, nominal exploratory correspondence; and Tier 3B, descriptive correspondence.

(B) Robustness of the AHBA-derived cortical maps across leave-one-donor-out reconstructions. Circles show the minimum and squares show the median Spearman correlation between each leave-one-donor-out reconstruction and the complete donor-balanced cortical map. Horizontal segments connect the corresponding minimum and median estimates.

(C) Strongest cognitive and clinical spatial correspondence for each DTHI map. Circles and squares indicate the cognitive and clinical external maps with the largest absolute Spearman correlation, respectively. Numerical labels show the corresponding correlations. These strongest correspondences are descriptive unless formal corrected support was retained in the definitive Phase 8D3C evidence hierarchy.

(D) Five locked priority clinical associations. The starred synaptic assembly/receptor trafficking–ASD association survived exact-bijective global maxT familywise-error correction. Square markers show four secondary associations retaining clinical-family BH-FDR support but not global familywise-error control. All five associations were directionally concordant across the five leave-one-donor-out reconstructions. None retained BH-FDR support in the nested LH46 sensitivity analysis.

Spatial correspondence represents anatomical convergence across modalities and does not establish causality, disease-specific molecular expression, mechanistic equivalence or independent replication. The LH46 analysis is a nested sensitivity analysis. Cell-system annotations remain definition-based because no compatible empirical cell-type table was available for Phase 8D3B.
"""


def provenance_record(
    source_layer: str,
    path: Path,
    detail: str,
) -> dict[str, Any]:
    require_file(path)

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
            sha256(path),
        "detail":
            detail,
    }


def prepare_provenance() -> pd.DataFrame:
    records = [
        provenance_record(
            "Phase8D3B_cross_modal_master",
            D3B_MASTER,
            (
                "developmental peaks, AHBA donor "
                "robustness and strongest cognitive "
                "and clinical associations"
            ),
        ),
        provenance_record(
            "Phase8D3C_association_hierarchy",
            D3C_ASSOC,
            (
                "definitive four-tier classification "
                "of 160 spatial associations"
            ),
        ),
        provenance_record(
            "Phase8D3C_map_hierarchy",
            D3C_MAPS,
            (
                "highest retained evidence tier "
                "for each DTHI map"
            ),
        ),
        provenance_record(
            "Phase8D3C_priority_reporting",
            D3C_PRIORITY,
            (
                "five locked priority clinical "
                "associations and corrected statistics"
            ),
        ),
        provenance_record(
            "Phase8D3C_completion",
            D3C_COMPLETION,
            (
                "validated evidence-hierarchy "
                "completion state"
            ),
        ),
        provenance_record(
            "Original_Phase8D3D_completion",
            D3D_COMPLETION,
            (
                "technically valid original figure "
                "retained and superseded visually by R1"
            ),
        ),
    ]

    return pd.DataFrame(
        records
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
            sha256(path),
    }


def main() -> None:
    validate_upstream()

    (
        master,
        associations,
        maps,
        priority,
    ) = load_inputs()

    map_data = prepare_map_data(
        master,
        maps,
    )

    priority_data = prepare_priority_data(
        priority
    )

    R1_OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    R1_FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PHASE8_TABLES.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_tsv(
        map_data,
        R1_FIG_DATA,
    )

    write_tsv(
        priority_data,
        R1_PRIORITY_DATA,
    )

    provenance = prepare_provenance()

    write_tsv(
        provenance,
        R1_PROVENANCE,
    )

    R1_LEGEND.write_text(
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

    write_tsv(
        output_manifest,
        R1_MANIFEST,
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

    tier_counts_correct = bool(
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

    map_counts_correct = bool(
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

    lodo_complete = bool(
        priority_data[
            "LODO_5of5_direction_concordant"
        ].all()
    )

    lh46_supported_count = int(
        priority_data[
            "LH46_BH_supported"
        ].sum()
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

    figure_file_validity = {
        path.suffix.lower():
            bool(
                path.is_file()
                and path.stat().st_size
                >= 1000
            )
        for path in figure_paths
    }

    original_png = (
        PHASE8_FIGURES
        / "phase8D3D"
        / "phase8D3D_integrated_cross_modal_figure.png"
    )

    original_figure_preserved = bool(
        original_png.is_file()
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
                        D3C_COMPLETION.relative_to(
                            ROOT
                        )
                    ),
            },
            {
                "section":
                    "upstream",
                "item":
                    "original_Phase8D3D_preserved",
                "value":
                    original_figure_preserved,
                "passed":
                    original_figure_preserved,
                "detail":
                    str(
                        original_png.relative_to(
                            ROOT
                        )
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
                    tier_counts_correct,
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
                    map_counts_correct,
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
                        "strongest cognitive and "
                        "clinical correspondence"
                    ),
            },
            {
                "section":
                    "panel_D",
                "item":
                    "priority_LODO_5of5",
                "value":
                    lodo_complete,
                "passed":
                    lodo_complete,
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
                    lh46_supported_count,
                "passed":
                    lh46_supported_count == 0,
                "detail":
                    (
                        "expected zero; nested "
                        "sensitivity analysis only"
                    ),
            },
            {
                "section":
                    "figure_outputs",
                "item":
                    "PNG_created",
                "value":
                    figure_file_validity.get(
                        ".png",
                        False,
                    ),
                "passed":
                    figure_file_validity.get(
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
                    figure_file_validity.get(
                        ".pdf",
                        False,
                    ),
                "passed":
                    figure_file_validity.get(
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
                    figure_file_validity.get(
                        ".svg",
                        False,
                    ),
                "passed":
                    figure_file_validity.get(
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
                    "revision",
                "item":
                    "landscape_layout",
                "value":
                    "16x12_inches",
                "passed":
                    True,
                "detail":
                    "compact two-by-two layout",
            },
            {
                "section":
                    "revision",
                "item":
                    "original_figure_superseded_visually",
                "value":
                    True,
                "passed":
                    True,
                "detail":
                    (
                        "R1 supersedes only the "
                        "visual presentation; "
                        "upstream statistics unchanged"
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
                        "R1 PNG requires direct "
                        "visual inspection"
                    ),
            },
        ]
    )

    write_tsv(
        audit,
        R1_AUDIT,
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
                "original_Phase8D3D_preserved":
                    original_figure_preserved,
                "DTHI_maps_in_figure":
                    len(map_data),
                "associations_available":
                    len(associations),
                "priority_associations_in_panel_D":
                    len(priority_data),
                "figure_panels":
                    4,
                "PNG_created":
                    figure_file_validity.get(
                        ".png",
                        False,
                    ),
                "PDF_created":
                    figure_file_validity.get(
                        ".pdf",
                        False,
                    ),
                "SVG_created":
                    figure_file_validity.get(
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
                "supersedes_original_visual":
                    True,
                "Phase8D3D_R1_status":
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

    write_tsv(
        completion,
        R1_COMPLETION,
    )

    print(
        "===== PHASE 8D3D-R1 COMPLETION ====="
    )

    print(
        completion.to_string(
            index=False
        )
    )

    print(
        "\n===== R1 FIGURE OUTPUTS ====="
    )

    print(
        output_manifest.to_string(
            index=False
        )
    )

    print(
        "\n===== R1 TECHNICAL AUDIT ====="
    )

    print(
        audit.to_string(
            index=False
        )
    )

    print(
        "\n===== R1 PRIORITY PANEL ====="
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
        R1_FIG_DATA,
        R1_PRIORITY_DATA,
        R1_PROVENANCE,
        R1_LEGEND,
        R1_MANIFEST,
        R1_AUDIT,
        R1_COMPLETION,
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
                    sha256(path),
            }
            for path in canonical_outputs
        ]
    )

    print(
        "\n===== R1 CANONICAL OUTPUT HASHES ====="
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
            "Phase 8D3D-R1 failed: "
            f"{type(error).__name__}: "
            f"{error}",
            file=sys.stderr,
        )
        raise
