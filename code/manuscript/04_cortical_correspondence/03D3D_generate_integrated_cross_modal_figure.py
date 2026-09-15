#!/usr/bin/env python3
"""Phase 8D3D: final integrated cross-modal figure."""

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
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]

D3B_DIR = (
    ROOT
    / "03_processed_data"
    / "functional_imaging"
    / "phase8D"
    / "phase8D3B_cross_modal_synthesis"
)

D3C_DIR = (
    ROOT
    / "03_processed_data"
    / "functional_imaging"
    / "phase8D"
    / "phase8D3C_evidence_hierarchy"
)

TABLE_DIR = (
    ROOT
    / "07_tables"
    / "main_tables"
    / "phase8"
)

FIGURE_DIR = (
    ROOT
    / "08_figures"
    / "phase8"
    / "phase8D3D"
)

OUT_DIR = (
    ROOT
    / "03_processed_data"
    / "functional_imaging"
    / "phase8D"
    / "phase8D3D_integrated_figure"
)

D3C_COMPLETION = (
    TABLE_DIR
    / "phase8D3C_completion_summary.tsv"
)

D3B_MASTER = (
    D3B_DIR
    / "phase8D3B_cross_modal_master_matrix.tsv"
)

D3C_ASSOCIATIONS = (
    D3C_DIR
    / "phase8D3C_association_evidence_hierarchy.tsv"
)

D3C_MAPS = (
    D3C_DIR
    / "phase8D3C_map_level_evidence_hierarchy.tsv"
)

D3C_PRIORITY = (
    D3C_DIR
    / "phase8D3C_priority_reporting_table.tsv"
)

D3C_LABELS = (
    D3C_DIR
    / "phase8D3C_manuscript_reporting_labels.tsv"
)

FIGURE_BASENAME = (
    FIGURE_DIR
    / "phase8D3D_integrated_cross_modal_figure"
)

FIGURE_DATA_OUT = (
    OUT_DIR
    / "phase8D3D_integrated_figure_data.tsv"
)

PRIORITY_DATA_OUT = (
    OUT_DIR
    / "phase8D3D_priority_panel_data.tsv"
)

SOURCE_OUT = (
    OUT_DIR
    / "phase8D3D_source_provenance.tsv"
)

LEGEND_OUT = (
    OUT_DIR
    / "phase8D3D_figure_legend.txt"
)

AUDIT_OUT = (
    TABLE_DIR
    / "phase8D3D_figure_audit.tsv"
)

COMPLETION_OUT = (
    TABLE_DIR
    / "phase8D3D_completion_summary.tsv"
)


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


DTHI_LABELS = {
    "patterning_arealization":
        "Patterning / arealization",
    "progenitor_radial_glia":
        "Progenitor / radial glia",
    "neurogenesis_migration_layering":
        "Neurogenesis / migration / layering",
    "axon_guidance_neurite_outgrowth":
        "Axon guidance / neurite outgrowth",
    "synaptic_assembly_receptor_trafficking":
        "Synaptic assembly / receptor trafficking",
    "astrocyte_maturation_metabolic_support":
        "Astrocyte maturation / metabolic support",
    "oligodendrocyte_myelination":
        "Oligodendrocyte / myelination",
    "activity_dependent_plasticity":
        "Activity-dependent plasticity",
    "synaptic_membrane_structural_candidates":
        "Synaptic-membrane structural candidates",
    "preliminary_maturation_balance":
        "Preliminary maturation balance",
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


WINDOW_LABELS = {
    "early_fetal_patterning":
        "Early fetal\npatterning",
    "mid_fetal_neurogenesis":
        "Mid-fetal\nneurogenesis",
    "late_fetal_synaptogenesis":
        "Late fetal\nsynaptogenesis",
    "infancy_postnatal_circuit_formation":
        "Infancy /\npostnatal circuits",
    "childhood_circuit_refinement":
        "Childhood\nrefinement",
    "adolescence_myelination_refinement":
        "Adolescent\nrefinement",
    "adult_stabilization":
        "Adult\nstabilization",
}


TIER_ORDER = {
    "Tier1_global_FWER": 1,
    "Tier2_family_corrected": 2,
    "Tier3A_nominal": 3,
    "Tier3B_descriptive": 4,
}


TIER_DISPLAY = {
    "Tier1_global_FWER":
        "Tier 1: global FWER",
    "Tier2_family_corrected":
        "Tier 2: family corrected",
    "Tier3A_nominal":
        "Tier 3A: nominal",
    "Tier3B_descriptive":
        "Tier 3B: descriptive",
}


TIER_MARKERS = {
    "Tier1_global_FWER": "*",
    "Tier2_family_corrected": "s",
    "Tier3A_nominal": "o",
    "Tier3B_descriptive": "x",
}


class ValidationError(RuntimeError):
    pass


def require_file(
    path: Path,
) -> None:
    if not path.is_file():
        raise ValidationError(
            f"Required file not found: {path}"
        )


def sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_tsv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    frame.to_csv(
        temporary,
        sep="\t",
        index=False,
    )

    temporary.replace(path)


def as_bool(
    value: Any,
) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def wrap_text(
    value: Any,
    width: int = 34,
) -> str:
    return "\n".join(
        textwrap.wrap(
            str(value),
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def validate_upstream() -> None:
    require_file(
        D3C_COMPLETION
    )

    completion = pd.read_csv(
        D3C_COMPLETION,
        sep="\t",
    )

    if len(completion) != 1:
        raise ValidationError(
            "Phase 8D3C completion summary "
            "must contain exactly one row."
        )

    row = completion.iloc[0]

    if str(
        row["Phase8D3C_status"]
    ).strip() != "completed":
        raise ValidationError(
            "Phase 8D3C is not completed."
        )

    if not as_bool(
        row[
            "ready_for_phase8D3D_integrated_figure"
        ]
    ):
        raise ValidationError(
            "Phase 8D3C is not ready "
            "for Phase 8D3D."
        )

    if as_bool(
        row["statistics_recomputed"]
    ):
        raise ValidationError(
            "Phase 8D3C unexpectedly reports "
            "that statistics were recomputed."
        )


def external_label(
    value: Any,
) -> str:
    text = str(value)

    replacements = {
        "cognitive_":
            "",
        "clinical_":
            "",
        "_":
            " ",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new,
        )

    special = {
        "asd":
            "ASD",
        "adhd":
            "ADHD",
        "ocd":
            "OCD",
        "parkinson s disease":
            "Parkinson's disease",
    }

    if text in special:
        return special[text]

    return text.title()


def short_map_label(
    value: Any,
) -> str:
    mapping = {
        "patterning_arealization":
            "Patterning /\narealization",
        "progenitor_radial_glia":
            "Progenitor /\nradial glia",
        "neurogenesis_migration_layering":
            "Neurogenesis /\nmigration",
        "axon_guidance_neurite_outgrowth":
            "Axon guidance /\nneurite outgrowth",
        "synaptic_assembly_receptor_trafficking":
            "Synaptic assembly /\nreceptor trafficking",
        "astrocyte_maturation_metabolic_support":
            "Astrocyte maturation /\nmetabolic support",
        "oligodendrocyte_myelination":
            "Oligodendrocyte /\nmyelination",
        "activity_dependent_plasticity":
            "Activity-dependent\nplasticity",
        "synaptic_membrane_structural_candidates":
            "Synaptic-membrane\nstructural candidates",
        "preliminary_maturation_balance":
            "Preliminary\nmaturation balance",
    }

    return mapping.get(
        str(value),
        str(value).replace(
            "_",
            " ",
        ),
    )


def numeric_column(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in frame.columns:
        raise ValidationError(
            f"Required numeric column "
            f"not found: {column}"
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    required_paths = [
        D3B_MASTER,
        D3C_ASSOCIATIONS,
        D3C_MAPS,
        D3C_PRIORITY,
        D3C_LABELS,
    ]

    for path in required_paths:
        require_file(path)

    master = pd.read_csv(
        D3B_MASTER,
        sep="\t",
        low_memory=False,
    )

    associations = pd.read_csv(
        D3C_ASSOCIATIONS,
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

    labels = pd.read_csv(
        D3C_LABELS,
        sep="\t",
        low_memory=False,
    )

    if len(master) != 10:
        raise ValidationError(
            "Expected ten Phase 8D3B "
            f"master rows; found {len(master)}."
        )

    if len(associations) != 160:
        raise ValidationError(
            "Expected 160 association rows; "
            f"found {len(associations)}."
        )

    if len(maps) != 10:
        raise ValidationError(
            "Expected ten map hierarchy rows; "
            f"found {len(maps)}."
        )

    if len(priority) != 5:
        raise ValidationError(
            "Expected five priority rows; "
            f"found {len(priority)}."
        )

    if len(labels) != 4:
        raise ValidationError(
            "Expected four manuscript "
            f"reporting tiers; found {len(labels)}."
        )

    observed_maps = set(
        master["DTHI_map"]
        .astype(str)
    )

    if observed_maps != set(DTHI_ORDER):
        raise ValidationError(
            "Phase 8D3B master matrix does "
            "not contain the expected ten maps."
        )

    return (
        master,
        associations,
        maps,
        priority,
    )


def prepare_figure_data(
    master: pd.DataFrame,
    maps: pd.DataFrame,
) -> pd.DataFrame:
    required_master_columns = {
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
        required_master_columns
        - set(master.columns)
    )

    if missing_master:
        raise ValidationError(
            "Phase 8D3B master matrix is "
            "missing columns: "
            f"{sorted(missing_master)}"
        )

    required_map_columns = {
        "DTHI_map",
        "map_level_highest_tier",
        "priority_association_count",
        "cross_modal_convergence_class",
    }

    missing_maps = (
        required_map_columns
        - set(maps.columns)
    )

    if missing_maps:
        raise ValidationError(
            "Phase 8D3C map hierarchy is "
            "missing columns: "
            f"{sorted(missing_maps)}"
        )

    data = master.merge(
        maps[
            [
                "DTHI_map",
                "map_level_highest_tier",
                "priority_association_count",
                "cross_modal_convergence_class",
            ]
        ],
        on="DTHI_map",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_D3C",
        ),
    )

    data[
        "AHBA_minimum_LODO_vs_full_rho"
    ] = numeric_column(
        data,
        "AHBA_minimum_LODO_vs_full_rho",
    )

    data[
        "AHBA_median_LODO_vs_full_rho"
    ] = numeric_column(
        data,
        "AHBA_median_LODO_vs_full_rho",
    )

    data[
        "strongest_cognitive_rho"
    ] = numeric_column(
        data,
        "strongest_cognitive_rho",
    )

    data[
        "strongest_clinical_rho"
    ] = numeric_column(
        data,
        "strongest_clinical_rho",
    )

    data[
        "priority_association_count"
    ] = pd.to_numeric(
        data[
            "priority_association_count_D3C"
        ],
        errors="coerce",
    ).fillna(
        pd.to_numeric(
            data[
                "priority_association_count"
            ],
            errors="coerce",
        )
    )

    data[
        "DTHI_map_order"
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
        "developmental_window_order"
    ] = data[
        "empirical_peak_developmental_window"
    ].map(
        {
            name: index
            for index, name
            in enumerate(WINDOW_ORDER)
        }
    )

    if data[
        "developmental_window_order"
    ].isna().any():
        missing_windows = data.loc[
            data[
                "developmental_window_order"
            ].isna(),
            [
                "DTHI_map",
                "empirical_peak_developmental_window",
            ],
        ]

        raise ValidationError(
            "Unrecognized developmental "
            "window values:\n"
            + missing_windows.to_string(
                index=False
            )
        )

    data[
        "tier_order"
    ] = data[
        "map_level_highest_tier"
    ].map(
        TIER_ORDER
    )

    if data[
        "tier_order"
    ].isna().any():
        raise ValidationError(
            "One or more maps have an "
            "unrecognized evidence tier."
        )

    data[
        "DTHI_label"
    ] = data[
        "DTHI_map"
    ].map(
        DTHI_LABELS
    )

    data[
        "short_DTHI_label"
    ] = data[
        "DTHI_map"
    ].map(
        short_map_label
    )

    data[
        "strongest_cognitive_label"
    ] = data[
        "strongest_cognitive_map"
    ].map(
        external_label
    )

    data[
        "strongest_clinical_label"
    ] = data[
        "strongest_clinical_map"
    ].map(
        external_label
    )

    data = data.sort_values(
        "DTHI_map_order",
        kind="stable",
    ).reset_index(
        drop=True
    )

    return data


def prepare_priority_data(
    priority: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "priority_reporting_order",
        "priority_role",
        "DTHI_map",
        "external_map",
        "spearman_rho",
        "evidence_tier",
        "bijective_p_spin_two_sided",
        "bijective_p_maxT_global160",
        "bijective_p_maxT_family80",
        "LODO_5of5_direction_concordant",
        "LH46_BH_supported",
    }

    missing = (
        required_columns
        - set(priority.columns)
    )

    if missing:
        raise ValidationError(
            "Priority table is missing "
            f"columns: {sorted(missing)}"
        )

    data = priority.copy()

    numeric_columns = [
        "priority_reporting_order",
        "spearman_rho",
        "bijective_p_spin_two_sided",
        "bijective_p_maxT_global160",
        "bijective_p_maxT_family80",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
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
        "association_label"
    ] = (
        data["DTHI_map"]
        .map(DTHI_LABELS)
        + " — "
        + data["external_map"]
        .map(external_label)
    )

    data = data.sort_values(
        "priority_reporting_order",
        kind="stable",
    ).reset_index(
        drop=True
    )

    return data


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family":
                "DejaVu Sans",
            "font.size":
                9,
            "axes.titlesize":
                11,
            "axes.labelsize":
                9,
            "xtick.labelsize":
                8,
            "ytick.labelsize":
                8,
            "legend.fontsize":
                8,
            "figure.titlesize":
                14,
            "axes.spines.top":
                False,
            "axes.spines.right":
                False,
            "pdf.fonttype":
                42,
            "ps.fonttype":
                42,
            "svg.fonttype":
                "none",
            "savefig.dpi":
                600,
        }
    )


def add_panel_label(
    axis: plt.Axes,
    label: str,
) -> None:
    axis.text(
        -0.14,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def tier_marker_size(
    tier: Any,
) -> float:
    sizes = {
        "Tier1_global_FWER":
            190,
        "Tier2_family_corrected":
            120,
        "Tier3A_nominal":
            75,
        "Tier3B_descriptive":
            45,
    }

    return float(
        sizes.get(
            str(tier),
            45,
        )
    )


def tier_short_label(
    tier: Any,
) -> str:
    labels = {
        "Tier1_global_FWER":
            "T1",
        "Tier2_family_corrected":
            "T2",
        "Tier3A_nominal":
            "T3A",
        "Tier3B_descriptive":
            "T3B",
    }

    return labels.get(
        str(tier),
        "",
    )


def panel_developmental_timing(
    axis: plt.Axes,
    data: pd.DataFrame,
) -> None:
    y_positions = np.arange(
        len(data)
    )[::-1]

    x_positions = data[
        "developmental_window_order"
    ].to_numpy(
        dtype=float
    )

    marker_sizes = data[
        "map_level_highest_tier"
    ].map(
        tier_marker_size
    ).to_numpy(
        dtype=float
    )

    axis.scatter(
        x_positions,
        y_positions,
        s=marker_sizes,
        zorder=3,
    )

    for x_value, y_value, tier in zip(
        x_positions,
        y_positions,
        data[
            "map_level_highest_tier"
        ],
    ):
        axis.annotate(
            tier_short_label(
                tier
            ),
            xy=(
                x_value,
                y_value,
            ),
            xytext=(
                0,
                9,
            ),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
        )

    axis.set_yticks(
        y_positions
    )

    axis.set_yticklabels(
        data[
            "DTHI_label"
        ].tolist()
    )

    axis.set_xticks(
        np.arange(
            len(WINDOW_ORDER)
        )
    )

    axis.set_xticklabels(
        [
            WINDOW_LABELS[
                window
            ]
            for window in WINDOW_ORDER
        ],
        rotation=35,
        ha="right",
    )

    axis.set_xlim(
        -0.45,
        len(WINDOW_ORDER) - 0.55,
    )

    axis.set_ylim(
        -0.8,
        len(data) - 0.15,
    )

    axis.grid(
        axis="x",
        alpha=0.22,
        linewidth=0.7,
    )

    axis.set_title(
        "Developmental timing of DTHI cortical hierarchy maps",
        loc="left",
        fontweight="bold",
        pad=10,
    )

    axis.set_xlabel(
        "Empirical peak developmental window"
    )

    axis.text(
        0.01,
        -0.30,
        (
            "Peak windows were derived from the "
            "Phase 5B2 donor-level developmental "
            "trajectory matrix. Marker size and "
            "labels denote the highest retained "
            "evidence tier for each DTHI map."
        ),
        transform=axis.transAxes,
        fontsize=7.5,
        va="top",
        ha="left",
    )

    add_panel_label(
        axis,
        "A",
    )


def panel_ahba_stability(
    axis: plt.Axes,
    data: pd.DataFrame,
) -> None:
    y_positions = np.arange(
        len(data)
    )[::-1]

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

    lower = np.minimum(
        minimum,
        median,
    )

    upper = np.maximum(
        minimum,
        median,
    )

    axis.hlines(
        y=y_positions,
        xmin=lower,
        xmax=upper,
        linewidth=2.0,
        alpha=0.65,
        zorder=1,
    )

    axis.scatter(
        minimum,
        y_positions,
        marker="o",
        s=42,
        label="Minimum LODO-vs-full rho",
        zorder=3,
    )

    axis.scatter(
        median,
        y_positions,
        marker="s",
        s=42,
        label="Median LODO-vs-full rho",
        zorder=3,
    )

    for x_value, y_value in zip(
        minimum,
        y_positions,
    ):
        axis.annotate(
            f"{x_value:.2f}",
            xy=(
                x_value,
                y_value,
            ),
            xytext=(
                -5,
                0,
            ),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=6.5,
        )

    axis.set_yticks(
        y_positions
    )

    axis.set_yticklabels(
        data[
            "DTHI_label"
        ].tolist()
    )

    axis.set_xlim(
        0.65,
        1.01,
    )

    axis.set_ylim(
        -0.8,
        len(data) - 0.15,
    )

    axis.set_xticks(
        np.arange(
            0.65,
            1.01,
            0.05,
        )
    )

    axis.grid(
        axis="x",
        alpha=0.22,
        linewidth=0.7,
    )

    axis.set_title(
        "AHBA donor robustness of the cortical maps",
        loc="left",
        fontweight="bold",
        pad=10,
    )

    axis.set_xlabel(
        "Spearman correlation with the full donor-balanced map"
    )

    axis.legend(
        loc="lower right",
        frameon=False,
        borderaxespad=0.4,
    )

    axis.text(
        0.01,
        -0.21,
        (
            "Horizontal segments connect the minimum "
            "and median correlations obtained across "
            "five leave-one-donor-out reconstructions."
        ),
        transform=axis.transAxes,
        fontsize=7.5,
        va="top",
        ha="left",
    )

    add_panel_label(
        axis,
        "B",
    )


def panel_cross_modal_correspondence(
    axis: plt.Axes,
    data: pd.DataFrame,
) -> None:
    y_positions = np.arange(
        len(data)
    )[::-1]

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
        linewidth=0.9,
        alpha=0.65,
        zorder=0,
    )

    for index, row in data.iterrows():
        y_value = y_positions[index]

        x_start = min(
            float(
                row[
                    "strongest_cognitive_rho"
                ]
            ),
            float(
                row[
                    "strongest_clinical_rho"
                ]
            ),
        )

        x_end = max(
            float(
                row[
                    "strongest_cognitive_rho"
                ]
            ),
            float(
                row[
                    "strongest_clinical_rho"
                ]
            ),
        )

        axis.hlines(
            y=y_value,
            xmin=x_start,
            xmax=x_end,
            linewidth=1.7,
            alpha=0.55,
            zorder=1,
        )

    axis.scatter(
        cognitive,
        y_positions,
        marker="o",
        s=48,
        label="Strongest cognitive map",
        zorder=3,
    )

    axis.scatter(
        clinical,
        y_positions,
        marker="s",
        s=48,
        label="Strongest clinical map",
        zorder=3,
    )

    for index, row in data.iterrows():
        y_value = y_positions[index]

        cognitive_rho = float(
            row[
                "strongest_cognitive_rho"
            ]
        )

        clinical_rho = float(
            row[
                "strongest_clinical_rho"
            ]
        )

        cognitive_offset = (
            6
            if cognitive_rho >= 0
            else -6
        )

        clinical_offset = (
            6
            if clinical_rho >= 0
            else -6
        )

        axis.annotate(
            str(
                row[
                    "strongest_cognitive_label"
                ]
            ),
            xy=(
                cognitive_rho,
                y_value,
            ),
            xytext=(
                cognitive_offset,
                6,
            ),
            textcoords="offset points",
            ha=(
                "left"
                if cognitive_offset > 0
                else "right"
            ),
            va="bottom",
            fontsize=6.1,
        )

        axis.annotate(
            str(
                row[
                    "strongest_clinical_label"
                ]
            ),
            xy=(
                clinical_rho,
                y_value,
            ),
            xytext=(
                clinical_offset,
                -6,
            ),
            textcoords="offset points",
            ha=(
                "left"
                if clinical_offset > 0
                else "right"
            ),
            va="top",
            fontsize=6.1,
        )

    axis.set_yticks(
        y_positions
    )

    axis.set_yticklabels(
        data[
            "DTHI_label"
        ].tolist()
    )

    combined_values = np.concatenate(
        [
            cognitive,
            clinical,
        ]
    )

    lower_limit = min(
        -0.70,
        float(
            np.nanmin(
                combined_values
            )
        ) - 0.08,
    )

    upper_limit = max(
        0.55,
        float(
            np.nanmax(
                combined_values
            )
        ) + 0.08,
    )

    axis.set_xlim(
        lower_limit,
        upper_limit,
    )

    axis.set_ylim(
        -0.8,
        len(data) - 0.15,
    )

    axis.grid(
        axis="x",
        alpha=0.22,
        linewidth=0.7,
    )

    axis.set_title(
        "Strongest cognitive and clinical spatial correspondences",
        loc="left",
        fontweight="bold",
        pad=10,
    )

    axis.set_xlabel(
        "Spearman spatial correlation"
    )

    axis.legend(
        loc="lower right",
        frameon=False,
        borderaxespad=0.4,
    )

    axis.text(
        0.01,
        -0.22,
        (
            "For each DTHI map, circles and squares "
            "identify the largest absolute cognitive "
            "and clinical spatial correlations, "
            "respectively. These values do not by "
            "themselves indicate corrected support."
        ),
        transform=axis.transAxes,
        fontsize=7.5,
        va="top",
        ha="left",
    )

    add_panel_label(
        axis,
        "C",
    )


def panel_priority_associations(
    axis: plt.Axes,
    priority: pd.DataFrame,
) -> None:
    plotting = priority.copy()

    plotting = plotting.sort_values(
        "priority_reporting_order",
        ascending=False,
        kind="stable",
    ).reset_index(
        drop=True
    )

    y_positions = np.arange(
        len(plotting)
    )

    rho_values = plotting[
        "spearman_rho"
    ].to_numpy(
        dtype=float
    )

    axis.axvline(
        0,
        linewidth=0.9,
        alpha=0.65,
        zorder=0,
    )

    axis.hlines(
        y=y_positions,
        xmin=0,
        xmax=rho_values,
        linewidth=2.2,
        alpha=0.55,
        zorder=1,
    )

    for tier in [
        "Tier1_global_FWER",
        "Tier2_family_corrected",
    ]:
        subset = plotting[
            plotting[
                "evidence_tier"
            ].eq(tier)
        ]

        if subset.empty:
            continue

        positions = subset.index.to_numpy()

        axis.scatter(
            subset[
                "spearman_rho"
            ],
            positions,
            marker=TIER_MARKERS[
                tier
            ],
            s=[
                tier_marker_size(
                    tier
                )
                for _ in range(
                    len(subset)
                )
            ],
            label=TIER_DISPLAY[
                tier
            ],
            zorder=3,
        )

    for index, row in plotting.iterrows():
        rho = float(
            row[
                "spearman_rho"
            ]
        )

        if str(
            row[
                "evidence_tier"
            ]
        ) == "Tier1_global_FWER":
            correction_text = (
                "global maxT "
                f"P={float(row['bijective_p_maxT_global160']):.3f}"
            )
        else:
            family_q = pd.to_numeric(
                pd.Series(
                    [
                        row.get(
                            "bijective_q_BH_family80",
                            np.nan,
                        )
                    ]
                ),
                errors="coerce",
            ).iloc[0]

            correction_text = (
                "family BH "
                f"q={family_q:.3f}"
            )

        robustness_text = (
            "LODO 5/5"
            if as_bool(
                row[
                    "LODO_5of5_direction_concordant"
                ]
            )
            else "LODO discordant"
        )

        annotation = (
            f"{external_label(row['external_map'])}; "
            f"{correction_text}; "
            f"{robustness_text}"
        )

        axis.annotate(
            annotation,
            xy=(
                rho,
                index,
            ),
            xytext=(
                7,
                0,
            ),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.6,
        )

        axis.annotate(
            f"{rho:.3f}",
            xy=(
                rho,
                index,
            ),
            xytext=(
                -7,
                0,
            ),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=6.6,
            fontweight="bold",
        )

    axis.set_yticks(
        y_positions
    )

    axis.set_yticklabels(
        [
            short_map_label(
                value
            )
            for value in plotting[
                "DTHI_map"
            ]
        ]
    )

    axis.set_xlim(
        -0.72,
        0.18,
    )

    axis.set_ylim(
        -0.65,
        len(plotting) - 0.35,
    )

    axis.grid(
        axis="x",
        alpha=0.22,
        linewidth=0.7,
    )

    axis.set_title(
        "Definitive priority clinical associations",
        loc="left",
        fontweight="bold",
        pad=10,
    )

    axis.set_xlabel(
        "Spearman spatial correlation"
    )

    axis.legend(
        loc="lower right",
        frameon=False,
        borderaxespad=0.4,
    )

    axis.text(
        0.01,
        -0.27,
        (
            "The starred association survived global "
            "maxT familywise-error correction. Square "
            "markers denote secondary associations with "
            "family-level BH-FDR support. All five pairs "
            "were directionally concordant across the "
            "five leave-one-donor-out reconstructions; "
            "none had LH46 BH-FDR support."
        ),
        transform=axis.transAxes,
        fontsize=7.5,
        va="top",
        ha="left",
    )

    add_panel_label(
        axis,
        "D",
    )


def build_figure(
    data: pd.DataFrame,
    priority: pd.DataFrame,
) -> list[Path]:
    configure_matplotlib()

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure = plt.figure(
        figsize=(
            18,
            19,
        ),
        constrained_layout=False,
    )

    grid = figure.add_gridspec(
        nrows=2,
        ncols=2,
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.93,
        wspace=0.48,
        hspace=0.50,
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

    panel_developmental_timing(
        axis_a,
        data,
    )

    panel_ahba_stability(
        axis_b,
        data,
    )

    panel_cross_modal_correspondence(
        axis_c,
        data,
    )

    panel_priority_associations(
        axis_d,
        priority,
    )

    figure.suptitle(
        (
            "Integrated developmental, "
            "transcriptomic and functional-imaging "
            "evidence for the DTHI cortical hierarchy"
        ),
        fontsize=16,
        fontweight="bold",
        y=0.975,
    )

    figure.text(
        0.5,
        0.025,
        (
            "Spatial correspondence is interpreted "
            "as cross-modal anatomical convergence, "
            "not as causality, mechanistic equivalence "
            "or independent replication. The LH46 "
            "analysis is a nested sensitivity analysis."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

    output_paths = [
        FIGURE_BASENAME.with_suffix(
            ".png"
        ),
        FIGURE_BASENAME.with_suffix(
            ".pdf"
        ),
        FIGURE_BASENAME.with_suffix(
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

(A) Empirical developmental peak windows for the ten DTHI cortical hierarchy maps. Developmental peaks were derived from the Phase 5B2 donor-level developmental trajectory matrix. Marker size and tier labels indicate the highest retained evidence category for each map: Tier 1, global familywise-error-controlled evidence; Tier 2, family-level corrected evidence; Tier 3A, nominal exploratory correspondence; and Tier 3B, descriptive correspondence.

(B) Robustness of the AHBA-derived cortical maps across leave-one-donor-out reconstructions. Circles represent the minimum and squares represent the median Spearman correlation between each leave-one-donor-out map and the complete donor-balanced map. Horizontal segments connect the minimum and median estimates.

(C) Strongest cognitive and clinical spatial correspondences for each DTHI map. Circles and squares denote the cognitive and clinical external maps with the largest absolute Spearman spatial correlation, respectively. These strongest correlations are descriptive unless the corresponding association retained formal corrected support in the definitive evidence hierarchy.

(D) Five priority clinical associations identified in the locked Phase 8D2C analysis and classified in Phase 8D3C. The starred synaptic assembly/receptor trafficking–ASD association survived exact-bijective global maxT familywise-error correction. Square markers indicate four secondary clinical associations retaining family-level BH-FDR support but not global familywise-error control. All five priority associations showed directionally concordant effects across the five leave-one-donor-out reconstructions. None retained BH-FDR support in the nested LH46 sensitivity analysis.

Spatial correspondence indicates anatomical convergence across modalities and does not demonstrate causality, disease-specific molecular expression, mechanistic equivalence or independent replication. Cell-system annotations remain definition-based because no compatible empirical cell-type table was available for Phase 8D3B.
"""


def provenance_record(
    layer: str,
    path: Path,
    detail: str,
) -> dict[str, Any]:
    require_file(
        path
    )

    return {
        "source_layer":
            layer,
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
                "developmental peaks, AHBA "
                "LODO stability and strongest "
                "cognitive/clinical associations"
            ),
        ),
        provenance_record(
            "Phase8D3C_association_hierarchy",
            D3C_ASSOCIATIONS,
            (
                "definitive classification of "
                "all 160 spatial associations"
            ),
        ),
        provenance_record(
            "Phase8D3C_map_hierarchy",
            D3C_MAPS,
            (
                "highest retained evidence tier "
                "for each of ten DTHI maps"
            ),
        ),
        provenance_record(
            "Phase8D3C_priority_table",
            D3C_PRIORITY,
            (
                "five locked priority clinical "
                "associations"
            ),
        ),
        provenance_record(
            "Phase8D3C_reporting_labels",
            D3C_LABELS,
            (
                "manuscript-safe evidence "
                "terminology"
            ),
        ),
        provenance_record(
            "Phase8D3C_completion",
            D3C_COMPLETION,
            (
                "validated upstream completion "
                "and readiness state"
            ),
        ),
    ]

    return pd.DataFrame(
        records
    )


def figure_output_record(
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

    figure_data = prepare_figure_data(
        master,
        maps,
    )

    priority_data = prepare_priority_data(
        priority
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_tsv(
        figure_data,
        FIGURE_DATA_OUT,
    )

    write_tsv(
        priority_data,
        PRIORITY_DATA_OUT,
    )

    provenance = prepare_provenance()

    write_tsv(
        provenance,
        SOURCE_OUT,
    )

    legend_text = build_figure_legend()

    LEGEND_OUT.write_text(
        legend_text,
        encoding="utf-8",
    )

    figure_paths = build_figure(
        figure_data,
        priority_data,
    )

    figure_outputs = pd.DataFrame(
        [
            figure_output_record(
                path
            )
            for path in figure_paths
        ]
    )

    figure_outputs_path = (
        OUT_DIR
        / "phase8D3D_figure_output_manifest.tsv"
    )

    write_tsv(
        figure_outputs,
        figure_outputs_path,
    )


def output_file_valid(
    path: Path,
    minimum_bytes: int = 100,
) -> bool:
    return (
        path.is_file()
        and path.stat().st_size
        >= minimum_bytes
    )


def expected_priority_structure(
    priority: pd.DataFrame,
) -> bool:
    if len(priority) != 5:
        return False

    tier_counts = (
        priority[
            "evidence_tier"
        ]
        .value_counts()
        .to_dict()
    )

    primary = priority[
        priority[
            "evidence_tier"
        ].eq(
            "Tier1_global_FWER"
        )
    ]

    return bool(
        tier_counts.get(
            "Tier1_global_FWER",
            0,
        ) == 1
        and tier_counts.get(
            "Tier2_family_corrected",
            0,
        ) == 4
        and len(primary) == 1
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


def build_audit(
    figure_data: pd.DataFrame,
    priority_data: pd.DataFrame,
    associations: pd.DataFrame,
    figure_paths: list[Path],
    figure_outputs_path: Path,
) -> pd.DataFrame:
    association_tier_counts = (
        associations[
            "evidence_tier"
        ]
        .value_counts()
        .to_dict()
    )

    expected_association_counts = bool(
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

    map_tier_counts = (
        figure_data[
            "map_level_highest_tier"
        ]
        .value_counts()
        .to_dict()
    )

    expected_map_counts = bool(
        map_tier_counts.get(
            "Tier1_global_FWER",
            0,
        ) == 1
        and map_tier_counts.get(
            "Tier2_family_corrected",
            0,
        ) == 3
    )

    developmental_complete = bool(
        figure_data[
            "empirical_peak_developmental_window"
        ]
        .notna()
        .all()
    )

    ahba_complete = bool(
        figure_data[
            [
                "AHBA_minimum_LODO_vs_full_rho",
                "AHBA_median_LODO_vs_full_rho",
            ]
        ]
        .notna()
        .all()
        .all()
    )

    external_complete = bool(
        figure_data[
            [
                "strongest_cognitive_rho",
                "strongest_clinical_rho",
                "strongest_cognitive_map",
                "strongest_clinical_map",
            ]
        ]
        .notna()
        .all()
        .all()
    )

    lodo_complete = bool(
        priority_data[
            "LODO_5of5_direction_concordant"
        ].all()
    )

    lh46_zero = bool(
        ~priority_data[
            "LH46_BH_supported"
        ].any()
    )

    figure_file_checks = {
        path.suffix.lower():
            output_file_valid(
                path
            )
        for path in figure_paths
    }

    records = [
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
                "input_dimensions",
            "item":
                "association_rows",
            "value":
                len(associations),
            "passed":
                len(associations) == 160,
            "detail":
                "definitive Phase 8D3C hierarchy",
        },
        {
            "section":
                "input_dimensions",
            "item":
                "DTHI_map_rows",
            "value":
                len(figure_data),
            "passed":
                len(figure_data) == 10,
            "detail":
                "one row per DTHI map",
        },
        {
            "section":
                "input_dimensions",
            "item":
                "priority_rows",
            "value":
                len(priority_data),
            "passed":
                len(priority_data) == 5,
            "detail":
                "one primary plus four secondary",
        },
        {
            "section":
                "evidence_structure",
            "item":
                "association_tier_counts",
            "value":
                json.dumps(
                    association_tier_counts,
                    sort_keys=True,
                ),
            "passed":
                expected_association_counts,
            "detail":
                (
                    "expected 1 Tier1, 4 Tier2, "
                    "13 Tier3A and 142 Tier3B"
                ),
        },
        {
            "section":
                "evidence_structure",
            "item":
                "map_level_tier_counts",
            "value":
                json.dumps(
                    map_tier_counts,
                    sort_keys=True,
                ),
            "passed":
                expected_map_counts,
            "detail":
                (
                    "expected one Tier1 map "
                    "and three Tier2 maps"
                ),
        },
        {
            "section":
                "evidence_structure",
            "item":
                "priority_structure",
            "value":
                expected_priority_structure(
                    priority_data
                ),
            "passed":
                expected_priority_structure(
                    priority_data
                ),
            "detail":
                (
                    "primary pair must be "
                    "synaptic assembly/receptor "
                    "trafficking–ASD"
                ),
        },
        {
            "section":
                "panel_A",
            "item":
                "developmental_peak_data_complete",
            "value":
                developmental_complete,
            "passed":
                developmental_complete,
            "detail":
                (
                    "empirical Phase 5B2 "
                    "developmental layer"
                ),
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
                "cognitive_clinical_data_complete",
            "value":
                external_complete,
            "passed":
                external_complete,
            "detail":
                (
                    "strongest cognitive and "
                    "clinical correspondence "
                    "for each map"
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
                int(
                    priority_data[
                        "LH46_BH_supported"
                    ].sum()
                ),
            "passed":
                lh46_zero,
            "detail":
                (
                    "expected zero; LH46 is a "
                    "nested sensitivity analysis"
                ),
        },
        {
            "section":
                "figure_outputs",
            "item":
                "PNG_created",
            "value":
                figure_file_checks.get(
                    ".png",
                    False,
                ),
            "passed":
                figure_file_checks.get(
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
                figure_file_checks.get(
                    ".pdf",
                    False,
                ),
            "passed":
                figure_file_checks.get(
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
                figure_file_checks.get(
                    ".svg",
                    False,
                ),
            "passed":
                figure_file_checks.get(
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
                "figure_outputs",
            "item":
                "output_manifest_created",
            "value":
                output_file_valid(
                    figure_outputs_path
                ),
            "passed":
                output_file_valid(
                    figure_outputs_path
                ),
            "detail":
                str(
                    figure_outputs_path.relative_to(
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
                (
                    "figure generated from "
                    "frozen upstream statistics"
                ),
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
                    "technical completion does "
                    "not replace visual inspection"
                ),
        },
    ]

    return pd.DataFrame(
        records
    )




def main() -> None:
    validate_upstream()

    (
        master,
        associations,
        maps,
        priority,
    ) = load_inputs()

    figure_data = prepare_figure_data(
        master,
        maps,
    )

    priority_data = prepare_priority_data(
        priority
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_tsv(
        figure_data,
        FIGURE_DATA_OUT,
    )

    write_tsv(
        priority_data,
        PRIORITY_DATA_OUT,
    )

    provenance = prepare_provenance()

    write_tsv(
        provenance,
        SOURCE_OUT,
    )

    LEGEND_OUT.write_text(
        build_figure_legend(),
        encoding="utf-8",
    )

    figure_paths = build_figure(
        figure_data,
        priority_data,
    )

    figure_outputs = pd.DataFrame(
        [
            figure_output_record(
                path
            )
            for path in figure_paths
        ]
    )

    figure_outputs_path = (
        OUT_DIR
        / "phase8D3D_figure_output_manifest.tsv"
    )

    write_tsv(
        figure_outputs,
        figure_outputs_path,
    )

    audit = build_audit(
        figure_data,
        priority_data,
        associations,
        figure_paths,
        figure_outputs_path,
    )

    write_tsv(
        audit,
        AUDIT_OUT,
    )

    all_audits_passed = bool(
        audit[
            "passed"
        ].all()
    )

    completion = pd.DataFrame(
        [
            {
                "Phase8D3C_status_confirmed":
                    True,
                "DTHI_maps_in_figure":
                    len(figure_data),
                "associations_available":
                    len(associations),
                "priority_associations_in_panel_D":
                    len(priority_data),
                "figure_panels":
                    4,
                "PNG_created":
                    output_file_valid(
                        figure_paths[0]
                    ),
                "PDF_created":
                    output_file_valid(
                        figure_paths[1]
                    ),
                "SVG_created":
                    output_file_valid(
                        figure_paths[2]
                    ),
                "statistics_recomputed":
                    False,
                "all_technical_audits_passed":
                    all_audits_passed,
                "ready_for_visual_review":
                    all_audits_passed,
                "manuscript_ready_visual_signoff":
                    False,
                "Phase8D3D_status":
                    (
                        "completed_pending_visual_review"
                        if all_audits_passed
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
        COMPLETION_OUT,
    )

    print(
        "===== PHASE 8D3D COMPLETION ====="
    )

    print(
        completion.to_string(
            index=False
        )
    )

    print(
        "\n===== FIGURE OUTPUTS ====="
    )

    print(
        figure_outputs.to_string(
            index=False
        )
    )

    print(
        "\n===== FIGURE AUDIT ====="
    )

    print(
        audit.to_string(
            index=False
        )
    )

    print(
        "\n===== PRIORITY PANEL DATA ====="
    )

    priority_display = [
        "priority_reporting_order",
        "priority_role",
        "DTHI_map",
        "external_map",
        "spearman_rho",
        "evidence_tier",
        "LODO_5of5_direction_concordant",
        "LH46_BH_supported",
    ]

    print(
        priority_data[
            priority_display
        ].to_string(
            index=False
        )
    )

    canonical_outputs = [
        FIGURE_DATA_OUT,
        PRIORITY_DATA_OUT,
        SOURCE_OUT,
        LEGEND_OUT,
        figure_outputs_path,
        AUDIT_OUT,
        COMPLETION_OUT,
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
        "\n===== CANONICAL OUTPUT HASHES ====="
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
            "Phase 8D3D failed: "
            f"{type(error).__name__}: "
            f"{error}",
            file=sys.stderr,
        )
        raise
