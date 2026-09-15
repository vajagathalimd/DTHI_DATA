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
module_summary_path = Path(sys.argv[3])
out = Path(sys.argv[4])

geometry_dir = out / "01_geometry_audit"
selected_dir = out / "02_selected_design"
audit_dir = out / "03_audit"

for directory in (
    geometry_dir,
    selected_dir,
    audit_dir,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
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


def trimmed_mean(
    values: list[float],
    proportion: float = 0.10,
) -> float:

    ordered = sorted(values)

    trim_count = int(
        math.floor(
            len(ordered) * proportion
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


module_rows = read_tsv(
    module_summary_path
)

modules = sorted(
    {
        row["module"]
        for row in module_rows
        if row.get("module")
    }
)

if len(modules) != 9:
    raise SystemExit(
        f"FAIL: expected 9 modules, observed {len(modules)}"
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

    required_fields = {
        "unit_id",
        "area_assignment",
        "broad_compartment",
        "cluster_annotation_mapped",
        "author_cluster_id",
        "array_row",
        "array_col",
        *modules,
    }

    missing = sorted(
        required_fields - set(fields)
    )

    if missing:
        raise SystemExit(
            "FAIL: missing Visium score columns: "
            + ";".join(missing)
        )

    for row in reader:

        area = row["area_assignment"]
        depth = depth_stratum(
            row["broad_compartment"]
        )

        mapped = as_bool(
            row["cluster_annotation_mapped"]
        )

        if (
            not mapped
            or area not in {"V1", "V2"}
            or depth == "not_testable"
        ):
            continue

        try:
            array_row = float(
                row["array_row"]
            )
            array_col = float(
                row["array_col"]
            )

            scores = {
                module: float(row[module])
                for module in modules
            }

        except ValueError as error:
            raise SystemExit(
                "FAIL: nonnumeric coordinate or score: "
                f"{error}"
            )

        if not all(
            math.isfinite(value)
            for value in scores.values()
        ):
            raise SystemExit(
                "FAIL: nonfinite module score detected."
            )

        spots.append(
            {
                "unit_id": row["unit_id"],
                "area": area,
                "depth": depth,
                "array_row": array_row,
                "array_col": array_col,
                "scores": scores,
            }
        )

if not spots:
    raise SystemExit(
        "FAIL: no eligible mapped V1/V2 cortical spots."
    )

minimum_row = min(
    spot["array_row"]
    for spot in spots
)

minimum_col = min(
    spot["array_col"]
    for spot in spots
)

candidate_widths = (
    4,
    6,
    8,
    10,
    12,
    16,
)

minimum_spots_per_block = 5
minimum_blocks_per_area = 4

depth_levels = (
    "SP",
    "superficial_CP",
    "L4",
    "deep_CP",
)

candidate_summaries: list[dict[str, Any]] = []
candidate_depth_rows: list[dict[str, Any]] = []
candidate_units: dict[
    str,
    list[dict[str, Any]],
] = {}

for width in candidate_widths:

    offsets = sorted(
        {
            0,
            width // 2,
        }
    )

    for row_offset in offsets:
        for col_offset in offsets:

            candidate_id = (
                f"w{width:02d}"
                f"_ro{row_offset:02d}"
                f"_co{col_offset:02d}"
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

            units: list[dict[str, Any]] = []
            mixed_blocks = 0
            low_count_pure_blocks = 0

            for (
                row_bin,
                col_bin,
                depth,
            ), block_spots in sorted(
                physical_blocks.items()
            ):

                areas = sorted(
                    {
                        spot["area"]
                        for spot in block_spots
                    }
                )

                if len(areas) != 1:
                    mixed_blocks += 1
                    continue

                area = areas[0]
                spot_count = len(
                    block_spots
                )

                if spot_count < minimum_spots_per_block:
                    low_count_pure_blocks += 1
                    continue

                unit: dict[str, Any] = {
                    "candidate_id": candidate_id,
                    "grid_width": width,
                    "row_offset": row_offset,
                    "col_offset": col_offset,
                    "block_id": (
                        f"{candidate_id}"
                        f"_r{row_bin:+04d}"
                        f"_c{col_bin:+04d}"
                    ),
                    "row_bin": row_bin,
                    "col_bin": col_bin,
                    "area": area,
                    "depth_stratum": depth,
                    "spots": spot_count,
                    "centroid_array_row": (
                        statistics.fmean(
                            spot["array_row"]
                            for spot in block_spots
                        )
                    ),
                    "centroid_array_col": (
                        statistics.fmean(
                            spot["array_col"]
                            for spot in block_spots
                        )
                    ),
                }

                for module in modules:

                    values = [
                        spot["scores"][module]
                        for spot in block_spots
                    ]

                    unit[
                        f"{module}__mean"
                    ] = statistics.fmean(
                        values
                    )

                    unit[
                        f"{module}__median"
                    ] = statistics.median(
                        values
                    )

                    unit[
                        f"{module}__trimmed_mean"
                    ] = trimmed_mean(
                        values
                    )

                units.append(
                    unit
                )

            candidate_units[
                candidate_id
            ] = units

            eligible_depths = 0
            common_blocks_total = 0
            common_blocks_values: list[int] = []

            for depth in depth_levels:

                depth_units = [
                    unit
                    for unit in units
                    if unit["depth_stratum"] == depth
                ]

                v1_units = [
                    unit
                    for unit in depth_units
                    if unit["area"] == "V1"
                ]

                v2_units = [
                    unit
                    for unit in depth_units
                    if unit["area"] == "V2"
                ]

                v1_blocks = len(
                    v1_units
                )

                v2_blocks = len(
                    v2_units
                )

                common_blocks = min(
                    v1_blocks,
                    v2_blocks,
                )

                eligible = (
                    v1_blocks
                    >= minimum_blocks_per_area
                    and v2_blocks
                    >= minimum_blocks_per_area
                )

                if eligible:
                    eligible_depths += 1
                    common_blocks_total += common_blocks
                    common_blocks_values.append(
                        common_blocks
                    )

                permutation_count = (
                    math.comb(
                        v1_blocks + v2_blocks,
                        v1_blocks,
                    )
                    if v1_blocks > 0
                    and v2_blocks > 0
                    else 0
                )

                candidate_depth_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "grid_width": width,
                        "row_offset": row_offset,
                        "col_offset": col_offset,
                        "depth_stratum": depth,
                        "V1_blocks": v1_blocks,
                        "V2_blocks": v2_blocks,
                        "V1_spots": sum(
                            unit["spots"]
                            for unit in v1_units
                        ),
                        "V2_spots": sum(
                            unit["spots"]
                            for unit in v2_units
                        ),
                        "common_area_blocks": (
                            common_blocks
                        ),
                        "exact_label_permutations": (
                            permutation_count
                        ),
                        "eligible_for_area_permutation": (
                            eligible
                        ),
                    }
                )

            total_physical_blocks = len(
                physical_blocks
            )

            mixed_fraction = (
                mixed_blocks
                / total_physical_blocks
                if total_physical_blocks
                else 1.0
            )

            candidate_summaries.append(
                {
                    "candidate_id": candidate_id,
                    "grid_width": width,
                    "row_offset": row_offset,
                    "col_offset": col_offset,
                    "eligible_depth_strata": (
                        eligible_depths
                    ),
                    "minimum_common_blocks_among_eligible": (
                        min(common_blocks_values)
                        if common_blocks_values
                        else 0
                    ),
                    "total_common_blocks_among_eligible": (
                        common_blocks_total
                    ),
                    "retained_pure_block_units": len(
                        units
                    ),
                    "mixed_physical_blocks_excluded": (
                        mixed_blocks
                    ),
                    "low_count_pure_blocks_excluded": (
                        low_count_pure_blocks
                    ),
                    "total_physical_blocks": (
                        total_physical_blocks
                    ),
                    "mixed_block_fraction": round(
                        mixed_fraction,
                        6,
                    ),
                    "distance_from_default_width": abs(
                        width - 8
                    ),
                    "offset_penalty": (
                        row_offset + col_offset
                    ),
                }
            )

candidate_summaries.sort(
    key=lambda row: (
        -int(
            row["eligible_depth_strata"]
        ),
        -int(
            row[
                "minimum_common_blocks_among_eligible"
            ]
        ),
        -int(
            row[
                "total_common_blocks_among_eligible"
            ]
        ),
        float(
            row["mixed_block_fraction"]
        ),
        int(
            row["distance_from_default_width"]
        ),
        int(
            row["offset_penalty"]
        ),
        int(
            row["grid_width"]
        ),
        int(
            row["row_offset"]
        ),
        int(
            row["col_offset"]
        ),
    )
)

selected = candidate_summaries[0]
selected_id = str(
    selected["candidate_id"]
)

for row in candidate_summaries:
    row["selected_primary_design"] = (
        row["candidate_id"] == selected_id
    )

selected_depth_rows = [
    row
    for row in candidate_depth_rows
    if row["candidate_id"] == selected_id
]

selected_units = candidate_units[
    selected_id
]

candidate_columns = [
    "candidate_id",
    "grid_width",
    "row_offset",
    "col_offset",
    "eligible_depth_strata",
    "minimum_common_blocks_among_eligible",
    "total_common_blocks_among_eligible",
    "retained_pure_block_units",
    "mixed_physical_blocks_excluded",
    "low_count_pure_blocks_excluded",
    "total_physical_blocks",
    "mixed_block_fraction",
    "distance_from_default_width",
    "offset_penalty",
    "selected_primary_design",
]

depth_columns = [
    "candidate_id",
    "grid_width",
    "row_offset",
    "col_offset",
    "depth_stratum",
    "V1_blocks",
    "V2_blocks",
    "V1_spots",
    "V2_spots",
    "common_area_blocks",
    "exact_label_permutations",
    "eligible_for_area_permutation",
]

unit_columns = [
    "candidate_id",
    "grid_width",
    "row_offset",
    "col_offset",
    "block_id",
    "row_bin",
    "col_bin",
    "area",
    "depth_stratum",
    "spots",
    "centroid_array_row",
    "centroid_array_col",
]

for module in modules:
    unit_columns.extend(
        [
            f"{module}__mean",
            f"{module}__median",
            f"{module}__trimmed_mean",
        ]
    )

write_tsv(
    geometry_dir
    / "phase10B4_A2_R1_candidate_geometry_summary.tsv",
    candidate_summaries,
    candidate_columns,
)

write_tsv(
    geometry_dir
    / "phase10B4_A2_R1_all_candidate_depth_eligibility.tsv",
    candidate_depth_rows,
    depth_columns,
)

write_tsv(
    selected_dir
    / "phase10B4_A2_R1_selected_depth_eligibility.tsv",
    selected_depth_rows,
    depth_columns,
)

write_tsv_gz(
    selected_dir
    / "phase10B4_A2_R1_selected_block_module_units.tsv.gz",
    selected_units,
    unit_columns,
)

eligible_selected = [
    row
    for row in selected_depth_rows
    if row[
        "eligible_for_area_permutation"
    ]
]

if len(eligible_selected) >= 2:

    status_value = (
        "passed_phase10B4_A2_R1_fixed_width_geometry_"
        "ready_for_multistratum_section_specific_"
        "block_permutation_testing"
    )

elif len(eligible_selected) == 1:

    eligible_name = eligible_selected[0][
        "depth_stratum"
    ]

    status_value = (
        "completed_phase10B4_A2_R1_fixed_width_geometry_"
        f"ready_for_{eligible_name}_only_section_specific_"
        "block_permutation_testing"
    )

else:

    status_value = (
        "phase10B4_A2_R1_no_depth_stratum_meets_"
        "prespecified_block_eligibility"
    )

status = {
    "phase": "phase10B4_A2_R1",
    "eligible_mapped_V1_V2_cortical_spots": len(
        spots
    ),
    "locked_modules": len(
        modules
    ),
    "candidate_grid_widths": (
        ";".join(
            str(value)
            for value in candidate_widths
        )
    ),
    "candidate_designs_evaluated": len(
        candidate_summaries
    ),
    "minimum_spots_per_retained_block": (
        minimum_spots_per_block
    ),
    "minimum_blocks_per_area": (
        minimum_blocks_per_area
    ),
    "selected_candidate_id": selected_id,
    "selected_grid_width": selected[
        "grid_width"
    ],
    "selected_row_offset": selected[
        "row_offset"
    ],
    "selected_col_offset": selected[
        "col_offset"
    ],
    "selected_eligible_depth_strata": len(
        eligible_selected
    ),
    "selected_eligible_depth_names": (
        ";".join(
            row["depth_stratum"]
            for row in eligible_selected
        )
    ),
    "grid_selected_using_geometry_only": True,
    "module_values_used_for_grid_selection": False,
    "permutation_tests_performed": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B4_A2_R1_status": status_value,
}

write_tsv(
    out / "phase10B4_A2_R1_status.tsv",
    [status],
    list(status.keys()),
)

report = [
    "===== PHASE 10B4-A2-R1 FIXED-WIDTH "
    "SPATIAL-BLOCK RESOLUTION =====",
    "",
    (
        "Eligible mapped V1/V2 cortical spots: "
        f"{len(spots)}"
    ),
    (
        "Candidate designs evaluated: "
        f"{len(candidate_summaries)}"
    ),
    (
        "Selected candidate: "
        f"{selected_id}"
    ),
    (
        "Selected grid width: "
        f"{selected['grid_width']}"
    ),
    (
        "Selected offsets: row="
        f"{selected['row_offset']}; "
        "column="
        f"{selected['col_offset']}"
    ),
    (
        "Eligible depth strata: "
        f"{len(eligible_selected)}"
    ),
    (
        "Eligible depth names: "
        + (
            ";".join(
                row["depth_stratum"]
                for row in eligible_selected
            )
            or "NONE"
        )
    ),
    "",
    "Grid selected using geometry only: TRUE",
    "Module values used for grid selection: FALSE",
    "Permutation tests performed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B4-A2-R1 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B4_A2_R1_report.txt"
).write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

print(
    "\n".join(report)
)

print(
    "\n===== SELECTED DEPTH ELIGIBILITY ====="
)

for row in selected_depth_rows:

    print(
        "\t".join(
            str(row[column])
            for column in depth_columns
        )
    )

checksum_rows: list[dict[str, Any]] = []

for path in sorted(
    out.rglob("*")
):

    if (
        path.is_file()
        and path.name
        != "phase10B4_A2_R1_SHA256.tsv"
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
    out / "phase10B4_A2_R1_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)
