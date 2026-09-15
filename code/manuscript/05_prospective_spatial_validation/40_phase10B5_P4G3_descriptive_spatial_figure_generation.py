from __future__ import annotations

import csv
import gzip
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


project = Path(sys.argv[1])
bin_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
reconciliation_path = Path(sys.argv[4])
out = Path(sys.argv[5])

scale_dir = out / "01_color_scale_lock"
main_dir = out / "02_main_figure"
supp_dir = out / "03_supplementary_figures"
manifest_dir = out / "04_figure_manifest"
legend_dir = out / "05_figure_legends"
guardrail_dir = out / "06_reporting_guardrails"
audit_dir = out / "07_audit"

for directory in (
    scale_dir,
    main_dir,
    supp_dir,
    manifest_dir,
    legend_dir,
    guardrail_dir,
    audit_dir,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

GRID = 64
DISPLAY_THRESHOLD = 20

core_modules = [
    "activity_dependent_plasticity",
    "astrocyte_maturation_metabolic_support",
    "neurogenesis_migration_layering",
    "patterning_arealization",
    "progenitor_radial_glia",
]

expanded_modules = [
    "axon_guidance_neurite_outgrowth",
    "oligodendrocyte_myelination",
    "synaptic_assembly_receptor_trafficking",
    "synaptic_membrane_structural_candidates",
]

module_titles = {
    "activity_dependent_plasticity":
        "Activity-dependent\nplasticity",
    "astrocyte_maturation_metabolic_support":
        "Astrocyte maturation /\nmetabolic support",
    "neurogenesis_migration_layering":
        "Neurogenesis /\nmigration / layering",
    "patterning_arealization":
        "Patterning /\narealization",
    "progenitor_radial_glia":
        "Progenitor /\nradial glia",
    "axon_guidance_neurite_outgrowth":
        "Axon guidance /\nneurite outgrowth",
    "oligodendrocyte_myelination":
        "Oligodendrocyte /\nmyelination",
    "synaptic_assembly_receptor_trafficking":
        "Synaptic assembly /\nreceptor trafficking",
    "synaptic_membrane_structural_candidates":
        "Synaptic membrane /\nstructural candidates",
}


def read_tsv(
    path: Path,
) -> tuple[list[dict[str, str]], list[str]]:

    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        return list(reader), reader.fieldnames or []


def write_tsv(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def bool_text(
    value: str,
) -> bool:

    return value.strip().upper() == "TRUE"


summary_rows, _ = read_tsv(
    summary_path
)

reconciliation_rows, _ = read_tsv(
    reconciliation_path
)

if len(
    summary_rows
) != 68:

    raise SystemExit(
        f"FAIL: expected 68 map summaries; "
        f"observed {len(summary_rows)}."
    )

if len(
    reconciliation_rows
) != 12:

    raise SystemExit(
        f"FAIL: expected 12 section "
        f"reconciliation rows; observed "
        f"{len(reconciliation_rows)}."
    )

if not all(
    bool_text(
        row[
            "cell_count_matches"
        ]
    )
    for row in reconciliation_rows
):

    raise SystemExit(
        "FAIL: a section did not reconcile "
        "before figure generation."
    )

section_metadata: dict[
    str,
    dict[str, Any],
] = {}

map_summary_lookup: dict[
    tuple[str, str],
    dict[str, str],
] = {}

for row in summary_rows:

    archive = row[
        "archive_name"
    ]

    module = row[
        "module"
    ]

    metadata = {
        "archive_name": archive,
        "donor_id": row[
            "donor_id"
        ],
        "gestational_week": int(
            row[
                "gestational_week"
            ]
        ),
        "effective_analysis_set": row[
            "effective_analysis_set"
        ],
        "panel_class": row[
            "panel_class"
        ],
        "processed_H5AD": row[
            "processed_H5AD"
        ],
        "selected_cells": int(
            row[
                "selected_cells"
            ]
        ),
    }

    if archive in section_metadata:

        if (
            section_metadata[
                archive
            ]
            != metadata
        ):

            raise SystemExit(
                f"FAIL: inconsistent section "
                f"metadata for {archive}."
            )

    else:

        section_metadata[
            archive
        ] = metadata

    key = (
        archive,
        module,
    )

    if key in map_summary_lookup:

        raise SystemExit(
            f"FAIL: duplicate map summary: {key}"
        )

    map_summary_lookup[
        key
    ] = row

if len(
    section_metadata
) != 12:

    raise SystemExit(
        "FAIL: expected 12 unique sections."
    )

bin_rows_by_map: dict[
    tuple[str, str],
    list[tuple[int, int, int, float]],
] = defaultdict(list)

source_bin_rows = 0

with gzip.open(
    bin_path,
    "rt",
    encoding="utf-8",
    newline="",
) as handle:

    reader = csv.DictReader(
        handle,
        delimiter="\t",
    )

    for row in reader:

        source_bin_rows += 1

        archive = row[
            "archive_name"
        ]

        module = row[
            "module"
        ]

        key = (
            archive,
            module,
        )

        if key not in map_summary_lookup:

            raise SystemExit(
                f"FAIL: bin row references "
                f"unknown map: {key}"
            )

        x_bin = int(
            row[
                "x_bin"
            ]
        )

        y_bin = int(
            row[
                "y_bin"
            ]
        )

        cell_count = int(
            row[
                "cell_count"
            ]
        )

        score = float(
            row[
                "mean_module_score"
            ]
        )

        if not (
            0 <= x_bin < GRID
            and 0 <= y_bin < GRID
        ):

            raise SystemExit(
                f"FAIL: bin outside 64x64 grid: "
                f"{key}, {x_bin}, {y_bin}"
            )

        if not math.isfinite(
            score
        ):

            raise SystemExit(
                f"FAIL: nonfinite spatial-bin "
                f"score in {key}."
            )

        bin_rows_by_map[
            key
        ].append(
            (
                x_bin,
                y_bin,
                cell_count,
                score,
            )
        )

if source_bin_rows != 186_418:

    raise SystemExit(
        f"FAIL: expected 186418 source "
        f"bin rows; observed {source_bin_rows}."
    )

if len(
    bin_rows_by_map
) != 68:

    raise SystemExit(
        f"FAIL: expected 68 maps in compressed "
        f"bin table; observed "
        f"{len(bin_rows_by_map)}."
    )

for key, rows in bin_rows_by_map.items():

    expected_occupied = int(
        map_summary_lookup[
            key
        ][
            "occupied_bins"
        ]
    )

    if len(
        rows
    ) != expected_occupied:

        raise SystemExit(
            f"FAIL: occupied-bin mismatch for "
            f"{key}: {len(rows)} versus "
            f"{expected_occupied}."
        )


def map_matrix(
    archive: str,
    module: str,
) -> np.ndarray:

    matrix = np.full(
        (
            GRID,
            GRID,
        ),
        np.nan,
        dtype=np.float64,
    )

    for (
        x_bin,
        y_bin,
        cell_count,
        score,
    ) in bin_rows_by_map[
        (
            archive,
            module,
        )
    ]:

        if (
            cell_count
            >= DISPLAY_THRESHOLD
        ):

            matrix[
                y_bin,
                x_bin,
            ] = score

    return matrix


def collect_display_scores(
    archives: list[str],
    module: str,
) -> np.ndarray:

    values: list[
        float
    ] = []

    for archive in archives:

        for (
            x_bin,
            y_bin,
            cell_count,
            score,
        ) in bin_rows_by_map[
            (
                archive,
                module,
            )
        ]:

            if (
                cell_count
                >= DISPLAY_THRESHOLD
            ):

                values.append(
                    score
                )

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.size == 0:

        raise SystemExit(
            f"FAIL: no figure-eligible bins for "
            f"{module}."
        )

    return array


def symmetric_limit(
    values: np.ndarray,
) -> float:

    absolute = np.abs(
        values
    )

    limit = float(
        np.quantile(
            absolute,
            0.98,
        )
    )

    if (
        not math.isfinite(
            limit
        )
        or limit <= 0.0
    ):

        limit = float(
            absolute.max()
        )

    if limit <= 0.0:

        limit = 1.0

    return limit


primary_300_sections = sorted(
    [
        archive
        for archive, metadata
        in section_metadata.items()
        if (
            metadata[
                "effective_analysis_set"
            ] == "primary"
            and metadata[
                "panel_class"
            ] == "300_gene_panel"
        )
    ],
    key=lambda archive: (
        section_metadata[
            archive
        ][
            "gestational_week"
        ],
        section_metadata[
            archive
        ][
            "donor_id"
        ],
        archive,
    ),
)

primary_960_sections = sorted(
    [
        archive
        for archive, metadata
        in section_metadata.items()
        if (
            metadata[
                "effective_analysis_set"
            ] == "primary"
            and metadata[
                "panel_class"
            ] == "960_gene_panel"
        )
    ],
    key=lambda archive: (
        section_metadata[
            archive
        ][
            "gestational_week"
        ],
        archive,
    ),
)

sensitivity_sections = sorted(
    [
        archive
        for archive, metadata
        in section_metadata.items()
        if metadata[
            "effective_analysis_set"
        ] == "section_sensitivity"
    ],
    key=lambda archive: (
        section_metadata[
            archive
        ][
            "gestational_week"
        ],
        section_metadata[
            archive
        ][
            "donor_id"
        ],
        archive,
    ),
)

if len(
    primary_300_sections
) != 6:

    raise SystemExit(
        f"FAIL: expected six primary 300-panel "
        f"sections; observed "
        f"{len(primary_300_sections)}."
    )

if len(
    primary_960_sections
) != 2:

    raise SystemExit(
        f"FAIL: expected two primary 960-panel "
        f"sections; observed "
        f"{len(primary_960_sections)}."
    )

if len(
    sensitivity_sections
) != 4:

    raise SystemExit(
        f"FAIL: expected four sensitivity "
        f"sections; observed "
        f"{len(sensitivity_sections)}."
    )

representative_by_age: dict[
    int,
    str,
] = {}

for archive in primary_300_sections:

    metadata = section_metadata[
        archive
    ]

    age = metadata[
        "gestational_week"
    ]

    if (
        age not in representative_by_age
        or metadata[
            "selected_cells"
        ]
        > section_metadata[
            representative_by_age[
                age
            ]
        ][
            "selected_cells"
        ]
        or (
            metadata[
                "selected_cells"
            ]
            == section_metadata[
                representative_by_age[
                    age
                ]
            ][
                "selected_cells"
            ]
            and archive
            < representative_by_age[
                age
            ]
        )
    ):

        representative_by_age[
            age
        ] = archive

expected_ages = [
    15,
    20,
    22,
    34,
]

if sorted(
    representative_by_age
) != expected_ages:

    raise SystemExit(
        f"FAIL: unexpected primary 300-panel "
        f"age set: "
        f"{sorted(representative_by_age)}"
    )

representative_sections = [
    representative_by_age[
        age
    ]
    for age in expected_ages
]

core_scale_rows: list[
    dict[str, Any]
] = []

core_scale_lookup: dict[
    str,
    float,
] = {}

for module in core_modules:

    values = collect_display_scores(
        primary_300_sections,
        module,
    )

    limit = symmetric_limit(
        values
    )

    core_scale_lookup[
        module
    ] = limit

    core_scale_rows.append(
        {
            "scoring_family": (
                "cross_panel_harmonized_core"
            ),
            "module": module,
            "reference_scope": (
                "six_primary_300_gene_sections"
            ),
            "figure_eligible_bins": int(
                values.size
            ),
            "absolute_score_q98": (
                limit
            ),
            "vmin": (
                -limit
            ),
            "vmax": (
                limit
            ),
            "scale_center": 0.0,
            "scale_type": (
                "symmetric_module_specific"
            ),
        }
    )

expanded_scale_rows: list[
    dict[str, Any]
] = []

expanded_scale_lookup: dict[
    str,
    float,
] = {}

for module in expanded_modules:

    values = collect_display_scores(
        primary_960_sections,
        module,
    )

    limit = symmetric_limit(
        values
    )

    expanded_scale_lookup[
        module
    ] = limit

    expanded_scale_rows.append(
        {
            "scoring_family": (
                "expanded_panel_only"
            ),
            "module": module,
            "reference_scope": (
                "two_primary_960_gene_sections"
            ),
            "figure_eligible_bins": int(
                values.size
            ),
            "absolute_score_q98": (
                limit
            ),
            "vmin": (
                -limit
            ),
            "vmax": (
                limit
            ),
            "scale_center": 0.0,
            "scale_type": (
                "symmetric_module_specific"
            ),
        }
    )

write_tsv(
    scale_dir
    / "phase10B5_P4G3_core_300_panel_color_scale_lock.tsv",
    core_scale_rows,
    [
        "scoring_family",
        "module",
        "reference_scope",
        "figure_eligible_bins",
        "absolute_score_q98",
        "vmin",
        "vmax",
        "scale_center",
        "scale_type",
    ],
)

write_tsv(
    scale_dir
    / "phase10B5_P4G3_expanded_960_panel_color_scale_lock.tsv",
    expanded_scale_rows,
    [
        "scoring_family",
        "module",
        "reference_scope",
        "figure_eligible_bins",
        "absolute_score_q98",
        "vmin",
        "vmax",
        "scale_center",
        "scale_type",
    ],
)

representative_rows = [
    {
        "gestational_week": age,
        "archive_name": representative_by_age[
            age
        ],
        "donor_id": section_metadata[
            representative_by_age[
                age
            ]
        ][
            "donor_id"
        ],
        "selected_cells": section_metadata[
            representative_by_age[
                age
            ]
        ][
            "selected_cells"
        ],
        "selection_rule": (
            "largest_selected_cell_count_within_"
            "gestational_week_among_primary_"
            "300_gene_sections"
        ),
        "selection_used_for_inference": False,
        "all_primary_sections_shown_in_supplement": (
            True
        ),
    }
    for age in expected_ages
]

write_tsv(
    manifest_dir
    / "phase10B5_P4G3_main_figure_section_selection.tsv",
    representative_rows,
    [
        "gestational_week",
        "archive_name",
        "donor_id",
        "selected_cells",
        "selection_rule",
        "selection_used_for_inference",
        "all_primary_sections_shown_in_supplement",
    ],
)


def row_label(
    archive: str,
) -> str:

    metadata = section_metadata[
        archive
    ]

    return (
        f"GW{metadata['gestational_week']} | "
        f"{metadata['donor_id']}\n"
        f"{archive.replace('.zip', '')}"
    )


def render_grid(
    archives: list[str],
    modules: list[str],
    scale_lookup: dict[str, float],
    output_stem: Path,
    figure_title: str,
    panel_class_note: str,
) -> int:

    rows = len(
        archives
    )

    columns = len(
        modules
    )

    fig_width = max(
        10.0,
        2.6 * columns,
    )

    fig_height = max(
        4.0,
        2.45 * rows
        + 1.0,
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            fig_width,
            fig_height,
        ),
        squeeze=False,
        constrained_layout=True,
    )

    image_handles: dict[
        str,
        Any,
    ] = {}

    panels = 0

    for row_index, archive in enumerate(
        archives
    ):

        for column_index, module in enumerate(
            modules
        ):

            key = (
                archive,
                module,
            )

            if key not in bin_rows_by_map:

                raise SystemExit(
                    f"FAIL: missing map required "
                    f"for figure: {key}"
                )

            matrix = map_matrix(
                archive,
                module,
            )

            if not np.isfinite(
                matrix
            ).any():

                raise SystemExit(
                    f"FAIL: no display bins for "
                    f"{key}."
                )

            limit = scale_lookup[
                module
            ]

            axis = axes[
                row_index,
                column_index,
            ]

            image = axis.imshow(
                matrix,
                origin="lower",
                interpolation="none",
                vmin=-limit,
                vmax=limit,
                cmap="RdBu_r",
                aspect="equal",
            )

            image_handles[
                module
            ] = image

            axis.set_xticks(
                []
            )

            axis.set_yticks(
                []
            )

            if row_index == 0:

                axis.set_title(
                    module_titles[
                        module
                    ],
                    fontsize=8,
                )

            if column_index == 0:

                axis.set_ylabel(
                    row_label(
                        archive
                    ),
                    fontsize=8,
                )

            panels += 1

    for column_index, module in enumerate(
        modules
    ):

        colorbar = fig.colorbar(
            image_handles[
                module
            ],
            ax=axes[
                :,
                column_index
            ].tolist(),
            orientation="horizontal",
            fraction=0.035,
            pad=0.025,
        )

        colorbar.ax.tick_params(
            labelsize=6,
        )

        colorbar.set_label(
            "Within-section module score",
            fontsize=7,
        )

    fig.suptitle(
        figure_title
        + "\n"
        + panel_class_note
        + " | native processed spatial orientation; "
        + "no anatomical x/y interpretation",
        fontsize=11,
    )

    pdf_path = output_stem.with_suffix(
        ".pdf"
    )

    png_path = output_stem.with_suffix(
        ".png"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return panels


main_panels = render_grid(
    representative_sections,
    core_modules,
    core_scale_lookup,
    main_dir
    / "phase10B5_P4G3_main_primary_300_core_spatial_maps",
    (
        "Representative primary spatial localization "
        "of cortical developmental modules"
    ),
    (
        "one deterministic primary 300-gene section "
        "per represented gestational age"
    ),
)

supp_primary_300_panels = render_grid(
    primary_300_sections,
    core_modules,
    core_scale_lookup,
    supp_dir
    / "phase10B5_P4G3_S1_all_primary_300_core_spatial_maps",
    (
        "All primary 300-gene sections: "
        "core spatial module maps"
    ),
    (
        "six independent primary donor-sections"
    ),
)

supp_primary_960_panels = render_grid(
    primary_960_sections,
    core_modules,
    {
        module: symmetric_limit(
            collect_display_scores(
                primary_960_sections,
                module,
            )
        )
        for module in core_modules
    },
    supp_dir
    / "phase10B5_P4G3_S2_primary_960_core_spatial_maps",
    (
        "Primary 960-gene sections: "
        "core spatial module maps"
    ),
    (
        "shown separately because of locked "
        "panel-class differences"
    ),
)

supp_sensitivity_panels = render_grid(
    sensitivity_sections,
    core_modules,
    core_scale_lookup,
    supp_dir
    / "phase10B5_P4G3_S3_sensitivity_core_spatial_maps",
    (
        "Within-donor section-sensitivity "
        "spatial maps"
    ),
    (
        "core modules using the primary 300-panel "
        "module color-scale lock"
    ),
)

supp_expanded_panels = render_grid(
    primary_960_sections,
    expanded_modules,
    expanded_scale_lookup,
    supp_dir
    / "phase10B5_P4G3_S4_expanded_960_spatial_maps",
    (
        "Expanded-panel developmental module "
        "spatial maps"
    ),
    (
        "two primary 960-gene sections; "
        "descriptive only"
    ),
)

expected_panel_counts = {
    "main": 20,
    "S1": 30,
    "S2": 10,
    "S3": 20,
    "S4": 8,
}

observed_panel_counts = {
    "main": main_panels,
    "S1": supp_primary_300_panels,
    "S2": supp_primary_960_panels,
    "S3": supp_sensitivity_panels,
    "S4": supp_expanded_panels,
}

if (
    observed_panel_counts
    != expected_panel_counts
):

    raise SystemExit(
        "FAIL: figure panel count mismatch: "
        f"{observed_panel_counts}"
    )

figure_manifest_rows = [
    {
        "figure_id": "main",
        "role": "main_manuscript_candidate",
        "section_scope": (
            "one_deterministic_primary_300_"
            "section_per_represented_age"
        ),
        "sections": 4,
        "modules": 5,
        "panels": 20,
        "PDF": (
            "02_main_figure/"
            "phase10B5_P4G3_main_primary_300_"
            "core_spatial_maps.pdf"
        ),
        "PNG": (
            "02_main_figure/"
            "phase10B5_P4G3_main_primary_300_"
            "core_spatial_maps.png"
        ),
        "formal_inference": False,
    },
    {
        "figure_id": "S1",
        "role": "supplementary",
        "section_scope": (
            "all_primary_300_gene_sections"
        ),
        "sections": 6,
        "modules": 5,
        "panels": 30,
        "PDF": (
            "03_supplementary_figures/"
            "phase10B5_P4G3_S1_all_primary_300_"
            "core_spatial_maps.pdf"
        ),
        "PNG": (
            "03_supplementary_figures/"
            "phase10B5_P4G3_S1_all_primary_300_"
            "core_spatial_maps.png"
        ),
        "formal_inference": False,
    },
    {
        "figure_id": "S2",
        "role": "supplementary",
        "section_scope": (
            "two_primary_960_gene_sections"
        ),
        "sections": 2,
        "modules": 5,
        "panels": 10,
        "PDF": (
            "03_supplementary_figures/"
            "phase10B5_P4G3_S2_primary_960_"
            "core_spatial_maps.pdf"
        ),
        "PNG": (
            "03_supplementary_figures/"
            "phase10B5_P4G3_S2_primary_960_"
            "core_spatial_maps.png"
        ),
        "formal_inference": False,
    },
    {
        "figure_id": "S3",
        "role": "supplementary",
        "section_scope": (
            "four_section_sensitivity_sections"
        ),
        "sections": 4,
        "modules": 5,
        "panels": 20,
        "PDF": (
            "03_supplementary_figures/"
            "phase10B5_P4G3_S3_sensitivity_"
            "core_spatial_maps.pdf"
        ),
        "PNG": (
            "03_supplementary_figures/"
            "phase10B5_P4G3_S3_sensitivity_"
            "core_spatial_maps.png"
        ),
        "formal_inference": False,
    },
    {
        "figure_id": "S4",
        "role": "supplementary",
        "section_scope": (
            "expanded_panel_two_section_validation"
        ),
        "sections": 2,
        "modules": 4,
        "panels": 8,
        "PDF": (
            "03_supplementary_figures/"
            "phase10B5_P4G3_S4_expanded_960_"
            "spatial_maps.pdf"
        ),
        "PNG": (
            "03_supplementary_figures/"
            "phase10B5_P4G3_S4_expanded_960_"
            "spatial_maps.png"
        ),
        "formal_inference": False,
    },
]

write_tsv(
    manifest_dir
    / "phase10B5_P4G3_figure_manifest.tsv",
    figure_manifest_rows,
    [
        "figure_id",
        "role",
        "section_scope",
        "sections",
        "modules",
        "panels",
        "PDF",
        "PNG",
        "formal_inference",
    ],
)

legend_text = """Phase 10B5 spatial-module figure guidance

Main figure.
Spatial distributions of five prespecified core developmental modules across one deterministic primary 300-gene section per represented gestational age. For ages represented by more than one primary 300-gene section, the section with the largest selected-cell count was used for visualization only; all primary sections are shown in Supplementary Figure S1. Module scores were computed from processed /X expression after within-section gene standardization and summarized in a 64 x 64 spatial grid. Only bins containing at least 20 cells are displayed. Color scales are centered at zero and locked separately for each module using the 98th percentile of the absolute bin-score distribution across the six primary 300-gene sections. Coordinates retain the native processed orientation and are not interpreted as anatomical axes. No smoothing, cross-section spatial registration, or spatial hypothesis testing was performed.

Supplementary Figure S1.
Spatial maps for all six independent primary 300-gene sections and all five core modules, using the same module-specific color scales as the main figure.

Supplementary Figure S2.
Core-module spatial maps for the two primary 960-gene sections. These are displayed separately because the processed datasets showed a locked panel-class shift. The maps are descriptive and are not used to infer temporal effects across panel classes.

Supplementary Figure S3.
Core-module spatial maps for the four within-donor section-sensitivity specimens. These sections are not independent biological replicates and are shown only to visualize robustness to section choice.

Supplementary Figure S4.
Spatial localization of the four expanded-panel-only modules in the GW18 and GW20 960-gene sections. These two-section comparisons are descriptive only.
"""

(
    legend_dir
    / "phase10B5_P4G3_spatial_figure_legends.txt"
).write_text(
    legend_text,
    encoding="utf-8",
)

guardrail_rows = [
    {
        "guardrail": "input_source",
        "locked_value": (
            "P4G2_compressed_spatial_bin_table_only"
        ),
    },
    {
        "guardrail": "H5AD_reopening",
        "locked_value": "prohibited_and_not_performed",
    },
    {
        "guardrail": "expression_access",
        "locked_value": "not_performed",
    },
    {
        "guardrail": "main_section_selection",
        "locked_value": (
            "largest_selected_cell_count_per_age_"
            "among_primary_300_sections"
        ),
    },
    {
        "guardrail": "main_selection_inference_role",
        "locked_value": "visualization_only",
    },
    {
        "guardrail": "grid_resolution",
        "locked_value": "64_by_64",
    },
    {
        "guardrail": "minimum_display_bin_cells",
        "locked_value": "20",
    },
    {
        "guardrail": "spatial_smoothing",
        "locked_value": "none",
    },
    {
        "guardrail": "interpolation",
        "locked_value": "none",
    },
    {
        "guardrail": "cross_section_registration",
        "locked_value": "none",
    },
    {
        "guardrail": "coordinate_orientation",
        "locked_value": (
            "native_processed_not_anatomically_"
            "interpreted"
        ),
    },
    {
        "guardrail": "spatial_inference",
        "locked_value": "none",
    },
]

write_tsv(
    guardrail_dir
    / "phase10B5_P4G3_figure_reporting_guardrails.tsv",
    guardrail_rows,
    [
        "guardrail",
        "locked_value",
    ],
)

expected_figure_paths = []

for row in figure_manifest_rows:

    expected_figure_paths.extend(
        [
            out
            / row[
                "PDF"
            ],
            out
            / row[
                "PNG"
            ],
        ]
    )

for path in expected_figure_paths:

    if (
        not path.is_file()
        or path.stat().st_size == 0
    ):

        raise SystemExit(
            f"FAIL: missing/empty figure: {path}"
        )

technical_pass = (
    source_bin_rows == 186_418
    and len(
        bin_rows_by_map
    ) == 68
    and len(
        representative_sections
    ) == 4
    and main_panels == 20
    and supp_primary_300_panels == 30
    and supp_primary_960_panels == 10
    and supp_sensitivity_panels == 20
    and supp_expanded_panels == 8
    and len(
        expected_figure_paths
    ) == 10
    and all(
        path.is_file()
        and path.stat().st_size > 0
        for path in expected_figure_paths
    )
)

status_value = (
    "passed_phase10B5_P4G3_manuscript_quality_"
    "descriptive_spatial_figures_and_scale_lock_"
    "ready_for_phase10B5_spatial_validation_"
    "synthesis"
    if technical_pass
    else (
        "phase10B5_P4G3_requires_manual_review"
    )
)

status = {
    "phase": "phase10B5_P4G3",
    "source_spatial_bin_rows_consumed": (
        source_bin_rows
    ),
    "source_spatial_maps_verified": len(
        bin_rows_by_map
    ),
    "primary_300_sections": len(
        primary_300_sections
    ),
    "primary_960_sections": len(
        primary_960_sections
    ),
    "sensitivity_sections": len(
        sensitivity_sections
    ),
    "main_representative_sections": len(
        representative_sections
    ),
    "main_figure_panels": (
        main_panels
    ),
    "supplement_S1_panels": (
        supp_primary_300_panels
    ),
    "supplement_S2_panels": (
        supp_primary_960_panels
    ),
    "supplement_S3_panels": (
        supp_sensitivity_panels
    ),
    "supplement_S4_panels": (
        supp_expanded_panels
    ),
    "composite_figure_files_created": len(
        expected_figure_paths
    ),
    "core_color_scale_locks": len(
        core_scale_rows
    ),
    "expanded_color_scale_locks": len(
        expanded_scale_rows
    ),
    "grid_bins_per_axis": GRID,
    "minimum_cells_per_display_bin": (
        DISPLAY_THRESHOLD
    ),
    "H5AD_files_opened": 0,
    "expression_values_accessed": False,
    "cell_level_scores_accessed": False,
    "spatial_smoothing_performed": False,
    "spatial_interpolation_performed": False,
    "cross_section_spatial_registration_performed": (
        False
    ),
    "coordinate_orientation_interpreted": (
        False
    ),
    "formal_spatial_hypothesis_tests_performed": (
        False
    ),
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P4G3_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B5_P4G3_status.tsv",
    [status],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P4G3 DESCRIPTIVE "
    "SPATIAL FIGURE GENERATION =====",
    "",
    (
        "Source spatial-bin rows consumed: "
        f"{source_bin_rows}"
    ),
    (
        "Source maps verified: "
        f"{len(bin_rows_by_map)}/68"
    ),
    (
        "Primary 300-gene sections: "
        f"{len(primary_300_sections)}/6"
    ),
    (
        "Primary 960-gene sections: "
        f"{len(primary_960_sections)}/2"
    ),
    (
        "Sensitivity sections: "
        f"{len(sensitivity_sections)}/4"
    ),
    (
        "Main representative sections: "
        f"{len(representative_sections)}/4"
    ),
    (
        "Main figure panels: "
        f"{main_panels}/20"
    ),
    (
        "Supplement S1 panels: "
        f"{supp_primary_300_panels}/30"
    ),
    (
        "Supplement S2 panels: "
        f"{supp_primary_960_panels}/10"
    ),
    (
        "Supplement S3 panels: "
        f"{supp_sensitivity_panels}/20"
    ),
    (
        "Supplement S4 panels: "
        f"{supp_expanded_panels}/8"
    ),
    (
        "Composite PDF/PNG files created: "
        f"{len(expected_figure_paths)}/10"
    ),
    "",
    "H5AD files opened: 0",
    "Expression values accessed: FALSE",
    "Cell-level scores accessed: FALSE",
    "Spatial smoothing performed: FALSE",
    "Spatial interpolation performed: FALSE",
    (
        "Cross-section spatial registration "
        "performed: FALSE"
    ),
    "Coordinate orientation interpreted: FALSE",
    (
        "Formal spatial hypothesis tests "
        "performed: FALSE"
    ),
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P4G3 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P4G3_report.txt"
).write_text(
    "\n".join(
        report
    )
    + "\n",
    encoding="utf-8",
)

print()
print(
    "\n".join(
        report
    )
)

print(
    "\n===== MAIN FIGURE SECTION SELECTION ====="
)

for row in representative_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "gestational_week",
                "archive_name",
                "donor_id",
                "selected_cells",
                "selection_rule",
            )
        )
    )

checksum_rows: list[
    dict[str, Any]
] = []

for path in sorted(
    out.rglob("*")
):

    if (
        path.is_file()
        and path.name
        != "phase10B5_P4G3_SHA256.tsv"
    ):

        checksum_rows.append(
            {
                "sha256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
                "size_bytes": (
                    path.stat().st_size
                ),
                "project_relative_path": (
                    path.relative_to(
                        project
                    ).as_posix()
                ),
            }
        )

write_tsv(
    out
    / "phase10B5_P4G3_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)

if not technical_pass:

    raise SystemExit(
        "Phase 10B5-P4G3 requires manual review."
    )
