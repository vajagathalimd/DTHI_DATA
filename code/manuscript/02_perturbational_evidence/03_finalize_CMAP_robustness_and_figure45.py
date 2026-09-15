#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import textwrap

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


PROJECT = Path(
    "."
)

PARENT_FILE = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6D/"
      "phase6D2_maxT_parent_robust_mechanism_classes.tsv.gz"
)

CHEMICAL_FILE = (
    PROJECT
    / "07_tables/main_tables/phase6/"
      "phase6D2_maxT_chemical_robustness_summary.tsv"
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
      "phase6D3_finalize_CMAP_robustness.log"
)

PARENT_TIER_OUTPUT = (
    TABLE_DIR
    / "phase6D3_parent_evidence_tiers.tsv"
)

CHEMICAL_TIER_OUTPUT = (
    TABLE_DIR
    / "phase6D3_chemical_evidence_tiers.tsv"
)

TIER_SUMMARY_OUTPUT = (
    TABLE_DIR
    / "phase6D3_evidence_tier_summary.tsv"
)

ROBUST_CHEMICAL_OUTPUT = (
    TABLE_DIR
    / "phase6D3_globally_robust_chemicals.tsv"
)

COMPLETION_OUTPUT = (
    TABLE_DIR
    / "phase6D3_completion_summary.tsv"
)

FIGURE_PNG = (
    FIGURE_DIR
    / "Figure45_CMAP_robustness_funnel_and_residual_mechanisms.png"
)

FIGURE_PDF = (
    FIGURE_DIR
    / "Figure45_CMAP_robustness_funnel_and_residual_mechanisms.pdf"
)

CAPTION_FILE = (
    FIGURE_DIR
    / "Figure45_caption.txt"
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


CATEGORIES = [
    "early_proliferative_state",
    "fetal_neurodevelopmental_state",
    "late_synaptic_maturation",
    "mitochondrial_metabolic_maturation",
    "glial_myelin_maturation",
    "DNA_damage_p53",
    "apoptosis_cytotoxicity",
    "oxidative_hypoxic_stress",
    "inflammatory_stress",
]

CATEGORY_LABELS = {
    "early_proliferative_state":
        "Early proliferation\nsuppression",

    "fetal_neurodevelopmental_state":
        "Fetal neural-program\nsuppression",

    "late_synaptic_maturation":
        "Late synaptic\ninduction",

    "mitochondrial_metabolic_maturation":
        "Mitochondrial\ninduction",

    "glial_myelin_maturation":
        "Glial/myelin\ninduction",

    "DNA_damage_p53":
        "DNA damage/\np53",

    "apoptosis_cytotoxicity":
        "Apoptosis/\ncytotoxicity",

    "oxidative_hypoxic_stress":
        "Oxidative/\nhypoxic stress",

    "inflammatory_stress":
        "Inflammatory\nstress",
}

TIER_LABELS = {
    "Tier_A_global_residual_mechanism":
        "Tier A: global residual mechanism",

    "Tier_B_relative_residual_profile":
        "Tier B: relative residual profile",

    "Tier_C_program_overlap_dependent":
        "Tier C: program-overlap dependent",

    "Tier_D_no_fixed_panel_mechanism":
        "Tier D: no fixed-panel mechanism",
}

TIER_ORDER = [
    "Tier_A_global_residual_mechanism",
    "Tier_B_relative_residual_profile",
    "Tier_C_program_overlap_dependent",
    "Tier_D_no_fixed_panel_mechanism",
]


def assign_parent_tier(
    row: pd.Series,
) -> str:
    if int(
        row[
            "leaveout_global_total_count"
        ]
    ) > 0:
        return (
            "Tier_A_global_residual_mechanism"
        )

    if int(
        row[
            "leaveout_within_parent_total_count"
        ]
    ) > 0:
        return (
            "Tier_B_relative_residual_profile"
        )

    original_support = (
        int(
            row[
                "original_global_total_count"
            ]
        )
        + int(
            row[
                "original_within_parent_total_count"
            ]
        )
    )

    if original_support > 0:
        return (
            "Tier_C_program_overlap_dependent"
        )

    return (
        "Tier_D_no_fixed_panel_mechanism"
    )


def assign_chemical_tier(
    row: pd.Series,
) -> str:
    if int(
        row[
            "parents_with_global_robust_mechanism"
        ]
    ) > 0:
        return (
            "Tier_A_global_residual_mechanism"
        )

    if int(
        row[
            "parents_with_within_parent_robust_mechanism"
        ]
    ) > 0:
        return (
            "Tier_B_relative_residual_profile"
        )

    if int(
        row[
            "program_overlap_dependent_parent_count"
        ]
    ) > 0:
        return (
            "Tier_C_program_overlap_dependent"
        )

    return (
        "Tier_D_no_fixed_panel_mechanism"
    )


def clean_text(
    value: object,
) -> str:
    if pd.isna(
        value
    ):
        return ""

    value = str(
        value
    ).strip()

    if value.lower() in {
        "",
        "nan",
        "none",
        "na",
    }:
        return ""

    return value


def main() -> None:
    if not PARENT_FILE.exists():
        raise FileNotFoundError(
            f"Missing parent file: {PARENT_FILE}"
        )

    if not CHEMICAL_FILE.exists():
        raise FileNotFoundError(
            f"Missing chemical file: {CHEMICAL_FILE}"
        )

    parents = pd.read_csv(
        PARENT_FILE,
        sep="\t",
        low_memory=False,
    )

    chemicals = pd.read_csv(
        CHEMICAL_FILE,
        sep="\t",
        low_memory=False,
    )

    if len(
        parents
    ) != 142:
        raise RuntimeError(
            (
                "Expected 142 maxT parents, "
                f"found {len(parents)}."
            )
        )

    if len(
        chemicals
    ) != 29:
        raise RuntimeError(
            (
                "Expected 29 maxT chemical entities, "
                f"found {len(chemicals)}."
            )
        )

    required_parent_columns = {
        "CMAP_parent",
        "chemical_name",
        "DTXSID",
        "developmental_direction_score",
        "maxT_maturation_FWER_p",
        "robust_mechanism_class",
        "original_global_total_count",
        "leaveout_global_total_count",
        "original_within_parent_total_count",
        "leaveout_within_parent_total_count",
        "leaveout_global_early_count",
        "leaveout_global_late_count",
        "leaveout_global_injury_count",
    }

    for category in CATEGORIES:
        required_parent_columns.add(
            f"leaveout_{category}_log2OR"
        )

    missing_parent_columns = sorted(
        required_parent_columns
        - set(
            parents.columns
        )
    )

    if missing_parent_columns:
        raise RuntimeError(
            "Missing parent columns: "
            + ", ".join(
                missing_parent_columns
            )
        )

    required_chemical_columns = {
        "chemical_name",
        "DTXSID",
        "maxT_parent_count",
        "parents_with_global_robust_mechanism",
        "parents_with_within_parent_robust_mechanism",
        "program_overlap_dependent_parent_count",
        "no_fixed_panel_mechanism_parent_count",
        "robust_injury_parent_count",
        "robust_mixed_parent_count",
        "robust_early_suppression_parent_count",
        "robust_late_maturation_parent_count",
        "median_developmental_direction_score",
        "minimum_maxT_maturation_FWER_p",
    }

    missing_chemical_columns = sorted(
        required_chemical_columns
        - set(
            chemicals.columns
        )
    )

    if missing_chemical_columns:
        raise RuntimeError(
            "Missing chemical columns: "
            + ", ".join(
                missing_chemical_columns
            )
        )

    parents[
        "evidence_tier"
    ] = parents.apply(
        assign_parent_tier,
        axis=1,
    )

    parents[
        "evidence_tier_label"
    ] = parents[
        "evidence_tier"
    ].map(
        TIER_LABELS
    )

    parents[
        "chemical_display"
    ] = parents[
        "chemical_name"
    ].map(
        clean_text
    )

    missing_names = parents[
        "chemical_display"
    ].eq("")

    parents.loc[
        missing_names,
        "chemical_display",
    ] = parents.loc[
        missing_names,
        "CMAP_parent",
    ].astype(
        str
    )

    parent_tier_counts = (
        parents[
            "evidence_tier"
        ]
        .value_counts()
        .reindex(
            TIER_ORDER,
            fill_value=0,
        )
    )

    expected_parent_counts = {
        "Tier_A_global_residual_mechanism":
            15,

        "Tier_B_relative_residual_profile":
            4,

        "Tier_C_program_overlap_dependent":
            42,

        "Tier_D_no_fixed_panel_mechanism":
            81,
    }

    observed_parent_counts = (
        parent_tier_counts.to_dict()
    )

    if (
        observed_parent_counts
        != expected_parent_counts
    ):
        raise RuntimeError(
            (
                "Unexpected parent-tier distribution: "
                f"{observed_parent_counts}"
            )
        )

    if int(
        parents[
            "leaveout_global_late_count"
        ].gt(0).sum()
    ) != 0:
        raise RuntimeError(
            "Unexpected robust late-maturation parent detected."
        )

    chemicals[
        "evidence_tier"
    ] = chemicals.apply(
        assign_chemical_tier,
        axis=1,
    )

    chemicals[
        "evidence_tier_label"
    ] = chemicals[
        "evidence_tier"
    ].map(
        TIER_LABELS
    )

    chemicals[
        "chemical_display"
    ] = chemicals[
        "chemical_name"
    ].map(
        clean_text
    )

    missing_names = chemicals[
        "chemical_display"
    ].eq("")

    chemicals.loc[
        missing_names,
        "chemical_display",
    ] = chemicals.loc[
        missing_names,
        "DTXSID",
    ].map(
        clean_text
    )

    robust_chemicals = chemicals.loc[
        chemicals[
            "evidence_tier"
        ].eq(
            "Tier_A_global_residual_mechanism"
        )
    ].copy()

    if len(
        robust_chemicals
    ) != 6:
        raise RuntimeError(
            (
                "Expected six globally robust chemicals, "
                f"found {len(robust_chemicals)}."
            )
        )

    parents = parents.sort_values(
        [
            "evidence_tier",
            "leaveout_global_total_count",
            "leaveout_within_parent_total_count",
            "developmental_direction_score",
            "maxT_maturation_FWER_p",
            "CMAP_parent",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            True,
            True,
        ],
    )

    chemicals = chemicals.sort_values(
        [
            "evidence_tier",
            "parents_with_global_robust_mechanism",
            "parents_with_within_parent_robust_mechanism",
            "maxT_parent_count",
            "median_developmental_direction_score",
            "chemical_display",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            False,
            True,
        ],
    )

    robust_chemicals = robust_chemicals.sort_values(
        [
            "parents_with_global_robust_mechanism",
            "parents_with_within_parent_robust_mechanism",
            "maxT_parent_count",
            "chemical_display",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    )

    parents.to_csv(
        PARENT_TIER_OUTPUT,
        sep="\t",
        index=False,
    )

    chemicals.to_csv(
        CHEMICAL_TIER_OUTPUT,
        sep="\t",
        index=False,
    )

    robust_chemicals.to_csv(
        ROBUST_CHEMICAL_OUTPUT,
        sep="\t",
        index=False,
    )

    parent_summary = pd.DataFrame(
        {
            "entity_level":
                "CMAP_parent",

            "evidence_tier":
                TIER_ORDER,

            "evidence_tier_label":
                [
                    TIER_LABELS[
                        tier
                    ]
                    for tier in TIER_ORDER
                ],

            "entity_count":
                [
                    int(
                        parent_tier_counts[
                            tier
                        ]
                    )
                    for tier in TIER_ORDER
                ],
        }
    )

    parent_summary[
        "fraction_within_entity_level"
    ] = (
        parent_summary[
            "entity_count"
        ]
        / len(
            parents
        )
    )

    chemical_tier_counts = (
        chemicals[
            "evidence_tier"
        ]
        .value_counts()
        .reindex(
            TIER_ORDER,
            fill_value=0,
        )
    )

    chemical_summary = pd.DataFrame(
        {
            "entity_level":
                "chemical",

            "evidence_tier":
                TIER_ORDER,

            "evidence_tier_label":
                [
                    TIER_LABELS[
                        tier
                    ]
                    for tier in TIER_ORDER
                ],

            "entity_count":
                [
                    int(
                        chemical_tier_counts[
                            tier
                        ]
                    )
                    for tier in TIER_ORDER
                ],
        }
    )

    chemical_summary[
        "fraction_within_entity_level"
    ] = (
        chemical_summary[
            "entity_count"
        ]
        / len(
            chemicals
        )
    )

    tier_summary = pd.concat(
        [
            parent_summary,
            chemical_summary,
        ],
        ignore_index=True,
    )

    tier_summary.to_csv(
        TIER_SUMMARY_OUTPUT,
        sep="\t",
        index=False,
    )

    globally_robust_parents = parents.loc[
        parents[
            "evidence_tier"
        ].eq(
            "Tier_A_global_residual_mechanism"
        )
    ].copy()

    globally_robust_parents = (
        globally_robust_parents.sort_values(
            [
                "robust_mechanism_class",
                "chemical_display",
                "CMAP_parent",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
    )

    heatmap_columns = [
        f"leaveout_{category}_log2OR"
        for category in CATEGORIES
    ]

    heatmap = (
        globally_robust_parents[
            heatmap_columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    heatmap_labels = []

    occurrence_counter: dict[str, int] = {}

    for _, row in globally_robust_parents.iterrows():
        name = row[
            "chemical_display"
        ]

        occurrence_counter[
            name
        ] = occurrence_counter.get(
            name,
            0,
        ) + 1

        heatmap_labels.append(
            (
                f"{name} "
                f"#{occurrence_counter[name]} "
                f"[{row['robust_mechanism_class']}]"
            )
        )

    figure = plt.figure(
        figsize=(
            22,
            17,
        )
    )

    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[
            0.92,
            1.45,
        ],
        height_ratios=[
            0.85,
            1.35,
        ],
        hspace=0.34,
        wspace=0.30,
    )

    # Panel A: evidence funnel
    axis_a = figure.add_subplot(
        grid[
            0,
            0,
        ]
    )

    funnel_labels = [
        "Paired CMAP\nparents",
        "maxT developmental-\nstate shifts",
        "Global residual\nmechanisms",
        "maxT chemical\nentities",
        "Chemicals with global\nresidual support",
    ]

    funnel_counts = [
        9273,
        len(
            parents
        ),
        int(
            parent_tier_counts[
                "Tier_A_global_residual_mechanism"
            ]
        ),
        len(
            chemicals
        ),
        len(
            robust_chemicals
        ),
    ]

    funnel_positions = np.arange(
        len(
            funnel_labels
        )
    )

    funnel_bars = axis_a.bar(
        funnel_positions,
        funnel_counts,
    )

    axis_a.set_yscale(
        "log"
    )

    axis_a.set_ylabel(
        "Entity count, logarithmic scale",
        fontsize=11,
    )

    axis_a.set_xticks(
        funnel_positions
    )

    axis_a.set_xticklabels(
        funnel_labels,
        rotation=18,
        ha="right",
        fontsize=9,
    )

    axis_a.set_title(
        "A  Perturbational-validation evidence funnel",
        loc="left",
        fontweight="bold",
        fontsize=13,
    )

    axis_a.axvline(
        2.5,
        linestyle="--",
        linewidth=1,
    )

    for bar, count in zip(
        funnel_bars,
        funnel_counts,
    ):
        axis_a.text(
            bar.get_x()
            + bar.get_width()
            / 2,
            count
            * 1.12,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Panel B: parent evidence tiers
    axis_b = figure.add_subplot(
        grid[
            0,
            1,
        ]
    )

    tier_display_labels = [
        TIER_LABELS[
            tier
        ]
        for tier in TIER_ORDER
    ]

    tier_values = [
        int(
            parent_tier_counts[
                tier
            ]
        )
        for tier in TIER_ORDER
    ]

    y_positions = np.arange(
        len(
            tier_display_labels
        )
    )

    tier_bars = axis_b.barh(
        y_positions,
        tier_values,
    )

    axis_b.set_yticks(
        y_positions
    )

    axis_b.set_yticklabels(
        tier_display_labels,
        fontsize=10,
    )

    axis_b.invert_yaxis()

    axis_b.set_xlabel(
        "maxT-supported CMAP parents",
        fontsize=11,
    )

    axis_b.set_title(
        "B  Evidence hierarchy after developmental-program removal",
        loc="left",
        fontweight="bold",
        fontsize=13,
    )

    for bar, count in zip(
        tier_bars,
        tier_values,
    ):
        percentage = (
            count
            / len(
                parents
            )
            * 100
        )

        axis_b.text(
            count + 1,
            bar.get_y()
            + bar.get_height()
            / 2,
            f"{count} ({percentage:.1f}%)",
            va="center",
            fontsize=10,
        )

    axis_b.set_xlim(
        0,
        max(
            tier_values
        )
        * 1.23,
    )

    # Panel C: residual mechanism heatmap
    axis_c = figure.add_subplot(
        grid[
            1,
            1,
        ]
    )

    matrix_minimum = float(
        np.nanmin(
            heatmap
        )
    )

    matrix_maximum = float(
        np.nanmax(
            heatmap
        )
    )

    absolute_limit = max(
        abs(
            matrix_minimum
        ),
        abs(
            matrix_maximum
        ),
        0.1,
    )

    image = axis_c.imshow(
        heatmap,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(
            vmin=-absolute_limit,
            vcenter=0,
            vmax=absolute_limit,
        ),
    )

    axis_c.set_xticks(
        np.arange(
            len(
                CATEGORIES
            )
        )
    )

    axis_c.set_xticklabels(
        [
            CATEGORY_LABELS[
                category
            ]
            for category in CATEGORIES
        ],
        rotation=45,
        ha="right",
        fontsize=8,
    )

    axis_c.set_yticks(
        np.arange(
            len(
                heatmap_labels
            )
        )
    )

    axis_c.set_yticklabels(
        heatmap_labels,
        fontsize=7,
    )

    axis_c.set_title(
        (
            "C  Leave-program-out mechanism scores "
            "for 15 globally robust parents"
        ),
        loc="left",
        fontweight="bold",
        fontsize=13,
    )

    colorbar = figure.colorbar(
        image,
        ax=axis_c,
        fraction=0.025,
        pad=0.018,
    )

    colorbar.set_label(
        "Preferred-direction corrected log2 odds ratio",
        fontsize=9,
    )

    # Panel D: chemical replication
    axis_d = figure.add_subplot(
        grid[
            1,
            0,
        ]
    )

    chemical_names = (
        robust_chemicals[
            "chemical_display"
        ].tolist()
    )

    global_counts = (
        robust_chemicals[
            "parents_with_global_robust_mechanism"
        ]
        .astype(
            int
        )
        .to_numpy()
    )

    within_counts = (
        robust_chemicals[
            "parents_with_within_parent_robust_mechanism"
        ]
        .astype(
            int
        )
        .to_numpy()
    )

    chemical_positions = np.arange(
        len(
            chemical_names
        )
    )

    bar_width = 0.38

    global_bars = axis_d.bar(
        chemical_positions
        - bar_width / 2,
        global_counts,
        width=bar_width,
        label="Global residual support",
    )

    within_bars = axis_d.bar(
        chemical_positions
        + bar_width / 2,
        within_counts,
        width=bar_width,
        label="Any within-parent residual support",
    )

    axis_d.set_xticks(
        chemical_positions
    )

    axis_d.set_xticklabels(
        [
            textwrap.fill(
                name,
                width=14,
            )
            for name in chemical_names
        ],
        rotation=28,
        ha="right",
        fontsize=9,
    )

    axis_d.set_ylabel(
        "maxT-supported parent signatures",
        fontsize=11,
    )

    axis_d.set_title(
        "D  Chemical-level replication of residual mechanisms",
        loc="left",
        fontweight="bold",
        fontsize=13,
    )

    axis_d.legend(
        frameon=False,
        fontsize=9,
    )

    axis_d.set_ylim(
        0,
        max(
            within_counts
        )
        + 1.2,
    )

    for bars in [
        global_bars,
        within_bars,
    ]:
        for bar in bars:
            height = bar.get_height()

            axis_d.text(
                bar.get_x()
                + bar.get_width()
                / 2,
                height + 0.08,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    figure.suptitle(
        (
            "CMAP developmental-state concordance is dominated by "
            "program overlap and residual injury/proliferative mechanisms"
        ),
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )

    figure.savefig(
        FIGURE_PNG,
        dpi=300,
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
        "Figure 45. Robustness and mechanistic decomposition of CMAP "
        "developmental-state concordance. "
        "(A) Evidence funnel from 9,273 paired CMAP parent signatures to "
        "142 maxT-supported developmental-state shifts, 15 parent "
        "signatures retaining globally significant mechanisms after "
        "removal of all fetal-high and maturation-high program genes, "
        "29 represented chemical entities, and six chemicals with at "
        "least one globally robust residual mechanism. "
        "(B) Evidence-tier classification of the 142 maxT-supported "
        "parents. Tier A retained a globally significant residual "
        "mechanism; Tier B retained only within-parent relative support; "
        "Tier C depended on direct developmental-program overlap; and "
        "Tier D showed no support from the fixed mechanism panel. "
        "(C) Leave-program-out preferred-direction log2 odds ratios for "
        "the 15 globally robust parents across early proliferative, fetal "
        "neurodevelopmental, late synaptic, mitochondrial, glial/myelin, "
        "DNA-damage, apoptosis, oxidative/hypoxic, and inflammatory "
        "mechanism categories. "
        "(D) Chemical-level replication of global and within-parent "
        "residual mechanism support. No parent signature retained "
        "globally significant late synaptic, mitochondrial, or "
        "glial/myelin maturation support."
    )

    CAPTION_FILE.write_text(
        caption + "\n",
        encoding="utf-8",
    )

    completion = pd.DataFrame(
        [
            {
                "maxT_parent_signatures":
                    len(
                        parents
                    ),

                "maxT_chemical_entities":
                    len(
                        chemicals
                    ),

                "tier_A_global_residual_parents":
                    expected_parent_counts[
                        "Tier_A_global_residual_mechanism"
                    ],

                "tier_B_relative_residual_parents":
                    expected_parent_counts[
                        "Tier_B_relative_residual_profile"
                    ],

                "tier_C_program_overlap_dependent_parents":
                    expected_parent_counts[
                        "Tier_C_program_overlap_dependent"
                    ],

                "tier_D_no_fixed_panel_mechanism_parents":
                    expected_parent_counts[
                        "Tier_D_no_fixed_panel_mechanism"
                    ],

                "globally_robust_chemical_entities":
                    len(
                        robust_chemicals
                    ),

                "globally_robust_late_maturation_parents":
                    int(
                        parents[
                            "leaveout_global_late_count"
                        ].gt(0).sum()
                    ),

                "figure45_png_created":
                    FIGURE_PNG.exists(),

                "figure45_pdf_created":
                    FIGURE_PDF.exists(),

                "figure45_caption_created":
                    CAPTION_FILE.exists(),

                "Phase6D3_status":
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
            "===== PHASE 6D3 PARENT EVIDENCE TIERS =====",
            parent_summary.to_string(
                index=False
            ),
            "",
            "===== PHASE 6D3 CHEMICAL EVIDENCE TIERS =====",
            chemical_summary.to_string(
                index=False
            ),
            "",
            "===== GLOBALLY ROBUST CHEMICALS =====",
            robust_chemicals[
                [
                    "chemical_name",
                    "DTXSID",
                    "maxT_parent_count",
                    "parents_with_global_robust_mechanism",
                    "parents_with_within_parent_robust_mechanism",
                    "robust_injury_parent_count",
                    "robust_mixed_parent_count",
                    "robust_early_suppression_parent_count",
                    "robust_late_maturation_parent_count",
                ]
            ].to_string(
                index=False
            ),
            "",
            "===== COMPLETION =====",
            completion.to_string(
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
                "Phase 6D3 failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
