from __future__ import annotations

import csv
import gzip
import hashlib
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


project = Path(sys.argv[1])
visium_path = Path(sys.argv[2])
geometry_path = Path(sys.argv[3])
eligibility_path = Path(sys.argv[4])
primary_path = Path(sys.argv[5])
out = Path(sys.argv[6])

effect_dir = (
    out
    / "01_reconstructed_geometry_effects"
)

robustness_dir = (
    out
    / "02_geometry_robustness"
)

design_dir = (
    out
    / "03_design_audit"
)

audit_dir = (
    out
    / "04_audit"
)

for directory in (
    effect_dir,
    robustness_dir,
    design_dir,
    audit_dir,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

minimum_spots_per_block = 5

direction_threshold = 0.80
width_direction_threshold = 0.75
minimum_unique_widths = 3
minimum_absolute_median_g = 0.50

depth_order = (
    "SP",
    "superficial_CP",
    "L4",
    "deep_CP",
)


def read_tsv(
    path: Path,
) -> list[dict[str, str]]:

    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:

        return list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )


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


def write_tsv_gz(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:

    with gzip.open(
        path,
        "wt",
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


def as_bool(
    value: str,
) -> bool:

    return str(value).strip().upper() in {
        "TRUE",
        "T",
        "1",
        "YES",
    }


def sign_value(
    value: float,
) -> int:

    if value > 0:
        return 1

    if value < 0:
        return -1

    return 0


def direction_name(
    value: float,
) -> str:

    if value > 0:
        return "V1_higher"

    if value < 0:
        return "V2_higher"

    return "equal"


def trimmed_mean(
    values: list[float],
    proportion: float = 0.10,
) -> float:

    ordered = sorted(values)

    trim_count = int(
        math.floor(
            len(ordered)
            * proportion
        )
    )

    if (
        trim_count > 0
        and 2 * trim_count < len(ordered)
    ):
        ordered = ordered[
            trim_count:-trim_count
        ]

    return statistics.fmean(
        ordered
    )


def quantile(
    values: list[float],
    probability: float,
) -> float:

    ordered = sorted(values)

    if not ordered:
        return math.nan

    if len(ordered) == 1:
        return ordered[0]

    position = (
        probability
        * (len(ordered) - 1)
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower

    return (
        ordered[lower]
        * (1 - fraction)
        + ordered[upper]
        * fraction
    )


def depth_stratum(
    compartment: str,
) -> str:

    if compartment == "SP":
        return "SP"

    if compartment in {
        "CP_layer2",
        "CP_layer3",
        "CP_layers2_3",
    }:
        return "superficial_CP"

    if compartment == "CP_layer4":
        return "L4"

    if compartment == "CP_layers5_6":
        return "deep_CP"

    return "not_testable"


def block_statistics(
    v1: list[float],
    v2: list[float],
) -> tuple[
    float,
    float,
    float,
    float,
    float,
]:

    if len(v1) < 2 or len(v2) < 2:
        raise ValueError(
            "At least two blocks per area are required."
        )

    mean_v1 = statistics.fmean(v1)
    mean_v2 = statistics.fmean(v2)

    difference = mean_v1 - mean_v2

    variance_v1 = statistics.variance(v1)
    variance_v2 = statistics.variance(v2)

    standard_error = math.sqrt(
        variance_v1 / len(v1)
        + variance_v2 / len(v2)
    )

    welch_t = (
        difference / standard_error
        if standard_error > 0
        else 0.0
    )

    pooled_variance = (
        (
            (len(v1) - 1)
            * variance_v1
            + (len(v2) - 1)
            * variance_v2
        )
        / (
            len(v1)
            + len(v2)
            - 2
        )
    )

    pooled_sd = math.sqrt(
        max(
            pooled_variance,
            0.0,
        )
    )

    correction = (
        1
        - 3
        / (
            4
            * (
                len(v1)
                + len(v2)
            )
            - 9
        )
    )

    hedges_g = (
        correction
        * difference
        / pooled_sd
        if pooled_sd > 0
        else 0.0
    )

    return (
        mean_v1,
        mean_v2,
        difference,
        welch_t,
        hedges_g,
    )


geometry_rows = read_tsv(
    geometry_path
)

eligible_candidates = [
    row
    for row in geometry_rows
    if int(
        row.get(
            "eligible_depth_strata",
            "0",
        )
    ) == 4
]

if not eligible_candidates:
    raise SystemExit(
        "FAIL: no geometry candidate supports all four strata."
    )

eligible_candidates.sort(
    key=lambda row: (
        int(row["grid_width"]),
        int(row["row_offset"]),
        int(row["col_offset"]),
    )
)

eligibility_rows = read_tsv(
    eligibility_path
)

expected_counts: dict[
    tuple[str, str],
    tuple[int, int],
] = {}

for row in eligibility_rows:

    candidate_id = row[
        "candidate_id"
    ]

    if candidate_id not in {
        candidate["candidate_id"]
        for candidate in eligible_candidates
    }:
        continue

    if not as_bool(
        row.get(
            "eligible_for_area_permutation",
            "",
        )
    ):
        raise SystemExit(
            "FAIL: a selected sensitivity candidate "
            "does not support all four strata."
        )

    expected_counts[
        (
            candidate_id,
            row["depth_stratum"],
        )
    ] = (
        int(row["V1_blocks"]),
        int(row["V2_blocks"]),
    )

primary_rows = read_tsv(
    primary_path
)

if len(primary_rows) != 36:
    raise SystemExit(
        f"FAIL: expected 36 A3 results, observed {len(primary_rows)}"
    )

modules = sorted(
    {
        row["module"]
        for row in primary_rows
    }
)

if len(modules) != 9:
    raise SystemExit(
        f"FAIL: expected 9 modules, observed {len(modules)}"
    )

primary_lookup = {
    (
        row["depth_stratum"],
        row["module"],
    ): row
    for row in primary_rows
}

primary_signal_count = sum(
    as_bool(
        row.get(
            "section_specific_primary_signal",
            "",
        )
    )
    for row in primary_rows
)

spots: list[dict[str, Any]] = []

with gzip.open(
    visium_path,
    "rt",
    encoding="utf-8",
    newline="",
) as handle:

    reader = csv.DictReader(
        handle,
        delimiter="\t",
    )

    fields = reader.fieldnames or []

    required = {
        "unit_id",
        "area_assignment",
        "broad_compartment",
        "cluster_annotation_mapped",
        "array_row",
        "array_col",
        *modules,
    }

    missing = sorted(
        required - set(fields)
    )

    if missing:
        raise SystemExit(
            "FAIL: missing Visium columns: "
            + ";".join(missing)
        )

    for row in reader:

        area = row["area_assignment"]

        depth = depth_stratum(
            row["broad_compartment"]
        )

        if (
            area not in {"V1", "V2"}
            or depth == "not_testable"
            or not as_bool(
                row["cluster_annotation_mapped"]
            )
        ):
            continue

        scores = {
            module: float(
                row[module]
            )
            for module in modules
        }

        if not all(
            math.isfinite(value)
            for value in scores.values()
        ):
            raise SystemExit(
                "FAIL: nonfinite Visium score detected."
            )

        spots.append(
            {
                "unit_id": row["unit_id"],
                "area": area,
                "depth": depth,
                "array_row": float(
                    row["array_row"]
                ),
                "array_col": float(
                    row["array_col"]
                ),
                "scores": scores,
            }
        )

if len(spots) != 1386:
    raise SystemExit(
        "FAIL: expected 1386 eligible mapped V1/V2 "
        f"cortical spots, observed {len(spots)}."
    )

minimum_row = min(
    spot["array_row"]
    for spot in spots
)

minimum_col = min(
    spot["array_col"]
    for spot in spots
)

effect_rows: list[dict[str, Any]] = []
candidate_audit_rows: list[dict[str, Any]] = []

all_count_matches = True

for candidate in eligible_candidates:

    candidate_id = candidate[
        "candidate_id"
    ]

    width = int(
        candidate["grid_width"]
    )

    row_offset = int(
        candidate["row_offset"]
    )

    col_offset = int(
        candidate["col_offset"]
    )

    physical_blocks: dict[
        tuple[int, int, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for spot in spots:

        row_bin = math.floor(
            (
                spot["array_row"]
                - minimum_row
                - row_offset
            )
            / width
        )

        col_bin = math.floor(
            (
                spot["array_col"]
                - minimum_col
                - col_offset
            )
            / width
        )

        physical_blocks[
            (
                row_bin,
                col_bin,
                spot["depth"],
            )
        ].append(
            spot
        )

    retained_blocks: list[dict[str, Any]] = []

    mixed_blocks = 0
    low_count_blocks = 0

    for (
        row_bin,
        col_bin,
        depth,
    ), block_spots in physical_blocks.items():

        areas = {
            spot["area"]
            for spot in block_spots
        }

        if len(areas) != 1:
            mixed_blocks += 1
            continue

        if len(block_spots) < minimum_spots_per_block:
            low_count_blocks += 1
            continue

        area = next(iter(areas))

        block_scores = {
            module: trimmed_mean(
                [
                    spot["scores"][module]
                    for spot in block_spots
                ]
            )
            for module in modules
        }

        retained_blocks.append(
            {
                "block_id": (
                    f"{candidate_id}"
                    f"_r{row_bin:+04d}"
                    f"_c{col_bin:+04d}"
                ),
                "area": area,
                "depth": depth,
                "spots": len(
                    block_spots
                ),
                "scores": block_scores,
            }
        )

    count_matches = True

    for depth in depth_order:

        depth_blocks = [
            block
            for block in retained_blocks
            if block["depth"] == depth
        ]

        v1_blocks = [
            block
            for block in depth_blocks
            if block["area"] == "V1"
        ]

        v2_blocks = [
            block
            for block in depth_blocks
            if block["area"] == "V2"
        ]

        observed_counts = (
            len(v1_blocks),
            len(v2_blocks),
        )

        expected = expected_counts.get(
            (
                candidate_id,
                depth,
            )
        )

        if expected != observed_counts:
            count_matches = False
            all_count_matches = False

        for module in modules:

            v1_values = [
                block["scores"][module]
                for block in v1_blocks
            ]

            v2_values = [
                block["scores"][module]
                for block in v2_blocks
            ]

            (
                mean_v1,
                mean_v2,
                difference,
                welch_t,
                hedges_g,
            ) = block_statistics(
                v1_values,
                v2_values,
            )

            primary = primary_lookup[
                (
                    depth,
                    module,
                )
            ]

            primary_difference = float(
                primary["V1_minus_V2"]
            )

            effect_rows.append(
                {
                    "candidate_id": candidate_id,
                    "grid_width": width,
                    "row_offset": row_offset,
                    "col_offset": col_offset,
                    "selected_A3_geometry": (
                        candidate_id
                        == "w04_ro00_co00"
                    ),
                    "depth_stratum": depth,
                    "module": module,
                    "V1_blocks": len(v1_blocks),
                    "V2_blocks": len(v2_blocks),
                    "V1_block_mean": mean_v1,
                    "V2_block_mean": mean_v2,
                    "V1_minus_V2": difference,
                    "Welch_studentized_statistic": (
                        welch_t
                    ),
                    "Hedges_g": hedges_g,
                    "direction": direction_name(
                        difference
                    ),
                    "A3_primary_direction": (
                        primary["direction"]
                    ),
                    "direction_concordant_with_A3": (
                        sign_value(difference)
                        == sign_value(
                            primary_difference
                        )
                    ),
                    "A3_section_specific_primary_signal": (
                        as_bool(
                            primary[
                                "section_specific_primary_signal"
                            ]
                        )
                    ),
                    "A3_global_BH_FDR": float(
                        primary[
                            "global_BH_FDR"
                        ]
                    ),
                    "A3_within_depth_maxT_p": float(
                        primary[
                            "within_depth_maxT_p"
                        ]
                    ),
                }
            )

    candidate_audit_rows.append(
        {
            "candidate_id": candidate_id,
            "grid_width": width,
            "row_offset": row_offset,
            "col_offset": col_offset,
            "retained_blocks": len(
                retained_blocks
            ),
            "mixed_blocks_excluded": (
                mixed_blocks
            ),
            "low_count_blocks_excluded": (
                low_count_blocks
            ),
            "block_counts_match_R1": (
                count_matches
            ),
        }
    )

if not all_count_matches:
    raise SystemExit(
        "FAIL: reconstructed block counts do not "
        "match Phase 10B4-A2-R1."
    )

effect_columns = [
    "candidate_id",
    "grid_width",
    "row_offset",
    "col_offset",
    "selected_A3_geometry",
    "depth_stratum",
    "module",
    "V1_blocks",
    "V2_blocks",
    "V1_block_mean",
    "V2_block_mean",
    "V1_minus_V2",
    "Welch_studentized_statistic",
    "Hedges_g",
    "direction",
    "A3_primary_direction",
    "direction_concordant_with_A3",
    "A3_section_specific_primary_signal",
    "A3_global_BH_FDR",
    "A3_within_depth_maxT_p",
]

write_tsv_gz(
    effect_dir
    / "phase10B4_A4_all_geometry_effects.tsv.gz",
    effect_rows,
    effect_columns,
)

write_tsv(
    design_dir
    / "phase10B4_A4_candidate_reconstruction_audit.tsv",
    candidate_audit_rows,
    [
        "candidate_id",
        "grid_width",
        "row_offset",
        "col_offset",
        "retained_blocks",
        "mixed_blocks_excluded",
        "low_count_blocks_excluded",
        "block_counts_match_R1",
    ],
)

effects_by_test: dict[
    tuple[str, str],
    list[dict[str, Any]],
] = defaultdict(list)

for row in effect_rows:

    effects_by_test[
        (
            row["depth_stratum"],
            row["module"],
        )
    ].append(
        row
    )

robustness_rows: list[dict[str, Any]] = []

for depth in depth_order:

    for module in modules:

        rows = effects_by_test[
            (
                depth,
                module,
            )
        ]

        primary = primary_lookup[
            (
                depth,
                module,
            )
        ]

        primary_difference = float(
            primary["V1_minus_V2"]
        )

        primary_sign = sign_value(
            primary_difference
        )

        differences = [
            float(row["V1_minus_V2"])
            for row in rows
        ]

        hedges_values = [
            float(row["Hedges_g"])
            for row in rows
        ]

        concordant = [
            sign_value(value)
            == primary_sign
            for value in differences
        ]

        width_groups: dict[
            int,
            list[float],
        ] = defaultdict(list)

        for row in rows:

            width_groups[
                int(row["grid_width"])
            ].append(
                float(
                    row["V1_minus_V2"]
                )
            )

        width_median_concordance = []

        for width in sorted(
            width_groups
        ):

            median_difference = statistics.median(
                width_groups[width]
            )

            width_median_concordance.append(
                sign_value(
                    median_difference
                )
                == primary_sign
            )

        direction_concordance = (
            sum(concordant)
            / len(concordant)
        )

        width_direction_concordance = (
            sum(width_median_concordance)
            / len(width_median_concordance)
        )

        median_difference = statistics.median(
            differences
        )

        median_g = statistics.median(
            hedges_values
        )

        geometry_robust = (
            direction_concordance
            >= direction_threshold
            and width_direction_concordance
            >= width_direction_threshold
            and len(width_groups)
            >= minimum_unique_widths
            and sign_value(
                median_difference
            )
            == primary_sign
            and abs(median_g)
            >= minimum_absolute_median_g
        )

        primary_signal = as_bool(
            primary[
                "section_specific_primary_signal"
            ]
        )

        robustness_rows.append(
            {
                "depth_stratum": depth,
                "module": module,
                "A3_primary_direction": (
                    primary["direction"]
                ),
                "A3_V1_minus_V2": (
                    primary_difference
                ),
                "A3_Hedges_g": float(
                    primary["Hedges_g"]
                ),
                "A3_global_BH_FDR": float(
                    primary["global_BH_FDR"]
                ),
                "A3_within_depth_maxT_p": float(
                    primary[
                        "within_depth_maxT_p"
                    ]
                ),
                "A3_section_specific_primary_signal": (
                    primary_signal
                ),
                "geometry_candidates": len(
                    rows
                ),
                "unique_grid_widths": len(
                    width_groups
                ),
                "direction_concordant_candidates": sum(
                    concordant
                ),
                "direction_concordance_fraction": (
                    direction_concordance
                ),
                "width_median_direction_concordance": (
                    width_direction_concordance
                ),
                "median_V1_minus_V2": (
                    median_difference
                ),
                "q25_V1_minus_V2": quantile(
                    differences,
                    0.25,
                ),
                "q75_V1_minus_V2": quantile(
                    differences,
                    0.75,
                ),
                "minimum_V1_minus_V2": min(
                    differences
                ),
                "maximum_V1_minus_V2": max(
                    differences
                ),
                "median_Hedges_g": median_g,
                "q25_Hedges_g": quantile(
                    hedges_values,
                    0.25,
                ),
                "q75_Hedges_g": quantile(
                    hedges_values,
                    0.75,
                ),
                "geometry_robust_effect": (
                    geometry_robust
                ),
                "geometry_robust_A3_primary_signal": (
                    primary_signal
                    and geometry_robust
                ),
            }
        )

robustness_rows.sort(
    key=lambda row: (
        not bool(
            row[
                "A3_section_specific_primary_signal"
            ]
        ),
        not bool(
            row[
                "geometry_robust_A3_primary_signal"
            ]
        ),
        -float(
            row[
                "direction_concordance_fraction"
            ]
        ),
        -abs(
            float(
                row["median_Hedges_g"]
            )
        ),
        depth_order.index(
            row["depth_stratum"]
        ),
        row["module"],
    )
)

robustness_columns = [
    "depth_stratum",
    "module",
    "A3_primary_direction",
    "A3_V1_minus_V2",
    "A3_Hedges_g",
    "A3_global_BH_FDR",
    "A3_within_depth_maxT_p",
    "A3_section_specific_primary_signal",
    "geometry_candidates",
    "unique_grid_widths",
    "direction_concordant_candidates",
    "direction_concordance_fraction",
    "width_median_direction_concordance",
    "median_V1_minus_V2",
    "q25_V1_minus_V2",
    "q75_V1_minus_V2",
    "minimum_V1_minus_V2",
    "maximum_V1_minus_V2",
    "median_Hedges_g",
    "q25_Hedges_g",
    "q75_Hedges_g",
    "geometry_robust_effect",
    "geometry_robust_A3_primary_signal",
]

write_tsv(
    robustness_dir
    / "phase10B4_A4_geometry_robustness_summary.tsv",
    robustness_rows,
    robustness_columns,
)

primary_signal_rows = [
    row
    for row in robustness_rows
    if row[
        "A3_section_specific_primary_signal"
    ]
]

write_tsv(
    robustness_dir
    / "phase10B4_A4_A3_primary_signal_geometry_robustness.tsv",
    primary_signal_rows,
    robustness_columns,
)

geometry_robust_primary_signals = sum(
    bool(
        row[
            "geometry_robust_A3_primary_signal"
        ]
    )
    for row in robustness_rows
)

all_effects_finite = all(
    math.isfinite(
        float(
            row["V1_minus_V2"]
        )
    )
    and math.isfinite(
        float(
            row["Hedges_g"]
        )
    )
    for row in effect_rows
)

expected_effect_rows = (
    len(eligible_candidates)
    * 4
    * 9
)

status_value = (
    "passed_phase10B4_A4_geometry_robustness_"
    "ready_for_cross_modal_section_specific_synthesis"
    if (
        len(primary_rows) == 36
        and len(robustness_rows) == 36
        and len(effect_rows)
        == expected_effect_rows
        and all_effects_finite
        and all_count_matches
    )
    else (
        "phase10B4_A4_requires_manual_review"
    )
)

status = {
    "phase": "phase10B4_A4",
    "A3_primary_tests": len(
        primary_rows
    ),
    "A3_section_specific_primary_signals": (
        primary_signal_count
    ),
    "eligible_geometry_candidates": len(
        eligible_candidates
    ),
    "unique_grid_widths": len(
        {
            int(row["grid_width"])
            for row in eligible_candidates
        }
    ),
    "geometry_effect_rows": len(
        effect_rows
    ),
    "geometry_robust_A3_primary_signals": (
        geometry_robust_primary_signals
    ),
    "direction_concordance_threshold": (
        direction_threshold
    ),
    "width_direction_concordance_threshold": (
        width_direction_threshold
    ),
    "minimum_absolute_median_Hedges_g": (
        minimum_absolute_median_g
    ),
    "minimum_unique_grid_widths": (
        minimum_unique_widths
    ),
    "block_counts_match_R1": (
        all_count_matches
    ),
    "significance_tests_repeated_across_geometries": (
        False
    ),
    "geometry_candidates_treated_as_independent_replication": (
        False
    ),
    "single_Visium_section": True,
    "independent_biological_replication": False,
    "population_level_inference": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B4_A4_status": status_value,
}

write_tsv(
    out
    / "phase10B4_A4_status.tsv",
    [status],
    list(status.keys()),
)

report = [
    "===== PHASE 10B4-A4 GEOMETRY ROBUSTNESS =====",
    "",
    (
        "A3 primary tests: "
        f"{len(primary_rows)}"
    ),
    (
        "A3 section-specific primary signals: "
        f"{primary_signal_count}"
    ),
    (
        "Eligible geometry candidates: "
        f"{len(eligible_candidates)}"
    ),
    (
        "Unique grid widths: "
        f"{status['unique_grid_widths']}"
    ),
    (
        "Geometry-effect rows: "
        f"{len(effect_rows)}"
    ),
    (
        "Geometry-robust A3 primary signals: "
        f"{geometry_robust_primary_signals}"
    ),
    (
        "Direction-concordance threshold: "
        f"{direction_threshold}"
    ),
    (
        "Width-direction threshold: "
        f"{width_direction_threshold}"
    ),
    (
        "Minimum absolute median Hedges g: "
        f"{minimum_absolute_median_g}"
    ),
    "",
    "Block counts match R1: TRUE",
    "Significance tests repeated across geometries: FALSE",
    (
        "Geometry candidates treated as independent "
        "replication: FALSE"
    ),
    "Single Visium section: TRUE",
    "Independent biological replication: FALSE",
    "Population-level inference: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B4-A4 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B4_A4_report.txt"
).write_text(
    "\n".join(report)
    + "\n",
    encoding="utf-8",
)

print(
    "\n".join(report)
)

print(
    "\n===== A3 PRIMARY-SIGNAL GEOMETRY ROBUSTNESS ====="
)

for row in primary_signal_rows:

    print(
        "\t".join(
            str(row[column])
            for column in robustness_columns
        )
    )

checksum_rows: list[dict[str, Any]] = []

for path in sorted(
    out.rglob("*")
):

    if (
        path.is_file()
        and path.name
        != "phase10B4_A4_SHA256.tsv"
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
    / "phase10B4_A4_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)
