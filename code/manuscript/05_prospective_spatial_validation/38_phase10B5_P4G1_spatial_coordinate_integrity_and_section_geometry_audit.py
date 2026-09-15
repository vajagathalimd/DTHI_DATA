from __future__ import annotations

import csv
import gzip
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


project = Path(sys.argv[1])
payload_dir = Path(sys.argv[2])
design_path = Path(sys.argv[3])
obs_lock_path = Path(sys.argv[4])
out = Path(sys.argv[5])

geometry_dir = out / "01_section_geometry"
quality_dir = out / "02_coordinate_quality"
sample_dir = out / "03_deterministic_geometry_samples"
processing_dir = out / "04_processing_audit"
guard_dir = out / "05_access_guard"
audit_dir = out / "06_audit"

for directory in (
    geometry_dir,
    quality_dir,
    sample_dir,
    processing_dir,
    guard_dir,
    audit_dir,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

expected_files = (
    "gw15.h5ad",
    "gw18_umb1759.h5ad",
    "gw20.h5ad",
    "gw20_umb1031.h5ad",
    "gw22.h5ad",
    "gw34.h5ad",
)

row_chunk_size = 250_000
sample_cells_per_section = 5_000


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


def decode(
    value: Any,
) -> str:

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    if isinstance(value, np.bytes_):
        return bytes(value).decode(
            "utf-8",
            errors="replace",
        )

    if isinstance(value, np.generic):
        return str(
            value.item()
        )

    return str(value)


def attr_text(
    value: Any,
) -> str:

    array = np.asarray(
        value
    )

    if array.ndim == 0:
        return decode(
            array.item()
        )

    return ";".join(
        decode(item)
        for item in array.reshape(-1)
    )


def categorical(
    obs: h5py.Group,
    column: str,
) -> tuple[list[str], h5py.Dataset]:

    if column not in obs:
        raise KeyError(
            f"Missing obs column: {column}"
        )

    group = obs[column]

    if not isinstance(
        group,
        h5py.Group,
    ):
        raise TypeError(
            f"obs/{column} is not categorical."
        )

    if (
        "categories" not in group
        or "codes" not in group
    ):
        raise KeyError(
            f"obs/{column} lacks categories/codes."
        )

    categories = [
        decode(value)
        for value in group[
            "categories"
        ][:]
    ]

    return categories, group["codes"]


def resolve_spatial_dataset(
    handle: h5py.File,
) -> h5py.Dataset:

    if "obsm" not in handle:
        raise KeyError(
            "Missing /obsm."
        )

    obsm = handle["obsm"]

    if "spatial" not in obsm:
        raise KeyError(
            "Missing /obsm/spatial."
        )

    spatial = obsm["spatial"]

    if isinstance(
        spatial,
        h5py.Dataset,
    ):
        dataset = spatial

    elif isinstance(
        spatial,
        h5py.Group,
    ) and "values" in spatial and isinstance(
        spatial["values"],
        h5py.Dataset,
    ):
        dataset = spatial[
            "values"
        ]

    else:
        raise TypeError(
            "Unsupported /obsm/spatial encoding."
        )

    if (
        dataset.ndim != 2
        or dataset.shape[1] != 2
    ):
        raise ValueError(
            f"Unexpected spatial shape: "
            f"{dataset.shape}"
        )

    return dataset


design_rows, design_columns = read_tsv(
    design_path
)

obs_lock_rows, obs_lock_columns = read_tsv(
    obs_lock_path
)

if len(design_rows) != 12:
    raise SystemExit(
        f"FAIL: expected 12 revised sections; "
        f"observed {len(design_rows)}."
    )

if sum(
    row[
        "effective_analysis_set"
    ] == "primary"
    for row in design_rows
) != 8:
    raise SystemExit(
        "FAIL: expected eight primary sections."
    )

if sum(
    row[
        "effective_analysis_set"
    ] == "section_sensitivity"
    for row in design_rows
) != 4:
    raise SystemExit(
        "FAIL: expected four sensitivity sections."
    )

if len(obs_lock_rows) != 6:
    raise SystemExit(
        f"FAIL: expected six obs-lock rows; "
        f"observed {len(obs_lock_rows)}."
    )

for row in obs_lock_rows:

    if row[
        "sample_column"
    ] != "sample":
        raise SystemExit(
            "FAIL: sample-column lock changed."
        )

    if row[
        "region_column"
    ] != "region":
        raise SystemExit(
            "FAIL: region-column lock changed."
        )

    if row[
        "spatial_embedding"
    ] != "obsm/spatial":
        raise SystemExit(
            "FAIL: spatial-embedding lock changed."
        )

section_metadata = {
    row[
        "archive_name"
    ]: {
        "effective_analysis_set": row[
            "effective_analysis_set"
        ],
        "archive_name": row[
            "archive_name"
        ],
        "donor_id": row[
            "donor_id"
        ],
        "gestational_week": int(
            row[
                "gestational_week"
            ]
        ),
        "panel_class": row[
            "panel_class"
        ],
        "processed_H5AD": row[
            "processed_H5AD"
        ],
        "sample_value": row[
            "locked_sample_value"
        ],
        "region_value": row[
            "locked_region_value"
        ],
        "selected_cell_count": int(
            row[
                "selected_cell_count"
            ]
        ),
    }
    for row in design_rows
}

if len(section_metadata) != 12:
    raise SystemExit(
        "FAIL: revised design contains duplicate "
        "archive names."
    )

sections_by_file: dict[
    str,
    list[str],
] = defaultdict(list)

for archive_name, metadata in section_metadata.items():

    sections_by_file[
        metadata[
            "processed_H5AD"
        ]
    ].append(
        archive_name
    )

for file_name in sections_by_file:

    sections_by_file[
        file_name
    ].sort()

if set(
    sections_by_file
) != set(
    expected_files
):
    raise SystemExit(
        "FAIL: revised design does not cover "
        "all six processed H5AD objects."
    )

geometry_rows: list[
    dict[str, Any]
] = []

quality_rows: list[
    dict[str, Any]
] = []

processing_rows: list[
    dict[str, Any]
] = []

guard_rows: list[
    dict[str, Any]
] = []

sample_rows: list[
    dict[str, Any]
] = []

total_selected_cells = 0
total_finite_cells = 0
total_nonfinite_cells = 0
all_sections_non_degenerate = True
all_sections_finite = True

for file_index, file_name in enumerate(
    expected_files,
    start=1,
):

    h5ad_path = (
        payload_dir
        / file_name
    )

    if not h5ad_path.is_file():
        raise SystemExit(
            f"FAIL: missing H5AD: {h5ad_path}"
        )

    archives = sections_by_file[
        file_name
    ]

    print(
        f"[{file_index:02d}/06] "
        f"Auditing spatial coordinates: "
        f"{file_name}",
        flush=True,
    )

    with h5py.File(
        h5ad_path,
        "r",
    ) as handle:

        obs = handle["obs"]

        sample_categories, sample_codes = (
            categorical(
                obs,
                "sample",
            )
        )

        region_categories, region_codes = (
            categorical(
                obs,
                "region",
            )
        )

        sample_lookup = {
            value: index
            for index, value
            in enumerate(
                sample_categories
            )
        }

        region_lookup = {
            value: index
            for index, value
            in enumerate(
                region_categories
            )
        }

        spatial = resolve_spatial_dataset(
            handle
        )

        spatial_rows = int(
            spatial.shape[0]
        )

        if int(
            sample_codes.shape[0]
        ) != spatial_rows:
            raise SystemExit(
                f"FAIL: sample rows differ from "
                f"spatial rows in {file_name}."
            )

        if int(
            region_codes.shape[0]
        ) != spatial_rows:
            raise SystemExit(
                f"FAIL: region rows differ from "
                f"spatial rows in {file_name}."
            )

        section_specs: list[
            dict[str, Any]
        ] = []

        for archive_name in archives:

            metadata = section_metadata[
                archive_name
            ]

            sample_value = metadata[
                "sample_value"
            ]

            region_value = metadata[
                "region_value"
            ]

            if sample_value not in sample_lookup:
                raise SystemExit(
                    f"FAIL: sample {sample_value} "
                    f"absent in {file_name}."
                )

            if region_value not in region_lookup:
                raise SystemExit(
                    f"FAIL: region {region_value} "
                    f"absent in {file_name}."
                )

            section_specs.append(
                {
                    "archive_name": archive_name,
                    "sample_code": sample_lookup[
                        sample_value
                    ],
                    "region_code": region_lookup[
                        region_value
                    ],
                    "expected_cells": metadata[
                        "selected_cell_count"
                    ],
                }
            )

        coordinate_chunks: dict[
            str,
            list[np.ndarray],
        ] = {
            archive_name: []
            for archive_name in archives
        }

        selected_counts = {
            archive_name: 0
            for archive_name in archives
        }

        spatial_values_read = 0

        for start in range(
            0,
            spatial_rows,
            row_chunk_size,
        ):

            end = min(
                start + row_chunk_size,
                spatial_rows,
            )

            sample_chunk = sample_codes[
                start:end
            ]

            region_chunk = region_codes[
                start:end
            ]

            any_selected = np.zeros(
                end - start,
                dtype=bool,
            )

            masks: dict[
                str,
                np.ndarray,
            ] = {}

            for spec in section_specs:

                archive_name = spec[
                    "archive_name"
                ]

                mask = (
                    sample_chunk
                    == spec[
                        "sample_code"
                    ]
                ) & (
                    region_chunk
                    == spec[
                        "region_code"
                    ]
                )

                if np.any(
                    any_selected
                    & mask
                ):
                    raise SystemExit(
                        f"FAIL: overlapping section "
                        f"mappings in {file_name}."
                    )

                any_selected |= mask

                masks[
                    archive_name
                ] = mask

            if not np.any(
                any_selected
            ):
                continue

            coordinate_chunk = np.asarray(
                spatial[
                    start:end,
                    :
                ],
                dtype=np.float64,
            )

            spatial_values_read += int(
                coordinate_chunk.size
            )

            for archive_name, mask in masks.items():

                if not np.any(
                    mask
                ):
                    continue

                selected_coordinates = (
                    coordinate_chunk[
                        mask,
                        :
                    ]
                )

                coordinate_chunks[
                    archive_name
                ].append(
                    selected_coordinates
                )

                selected_counts[
                    archive_name
                ] += int(
                    selected_coordinates.shape[0]
                )

        file_selected_cells = 0

        for spec in section_specs:

            archive_name = spec[
                "archive_name"
            ]

            expected_cells = int(
                spec[
                    "expected_cells"
                ]
            )

            observed_cells = int(
                selected_counts[
                    archive_name
                ]
            )

            if observed_cells != expected_cells:
                raise SystemExit(
                    f"FAIL: {archive_name} spatial "
                    f"cell count={observed_cells}; "
                    f"expected={expected_cells}."
                )

            coordinates = np.concatenate(
                coordinate_chunks[
                    archive_name
                ],
                axis=0,
            )

            if coordinates.shape != (
                expected_cells,
                2,
            ):
                raise SystemExit(
                    f"FAIL: unexpected coordinate "
                    f"shape for {archive_name}: "
                    f"{coordinates.shape}"
                )

            finite_mask = np.isfinite(
                coordinates
            ).all(
                axis=1
            )

            finite_cells = int(
                np.count_nonzero(
                    finite_mask
                )
            )

            nonfinite_cells = (
                expected_cells
                - finite_cells
            )

            finite_fraction = (
                finite_cells
                / expected_cells
            )

            if finite_cells == 0:
                raise SystemExit(
                    f"FAIL: no finite coordinates "
                    f"for {archive_name}."
                )

            finite_coordinates = (
                coordinates[
                    finite_mask,
                    :
                ]
            )

            x = finite_coordinates[
                :,
                0
            ]

            y = finite_coordinates[
                :,
                1
            ]

            x_min = float(
                x.min()
            )

            x_max = float(
                x.max()
            )

            y_min = float(
                y.min()
            )

            y_max = float(
                y.max()
            )

            x_span = (
                x_max
                - x_min
            )

            y_span = (
                y_max
                - y_min
            )

            non_degenerate = (
                x_span > 0.0
                and y_span > 0.0
            )

            all_sections_non_degenerate &= (
                non_degenerate
            )

            all_sections_finite &= (
                nonfinite_cells == 0
            )

            total_selected_cells += (
                expected_cells
            )

            total_finite_cells += (
                finite_cells
            )

            total_nonfinite_cells += (
                nonfinite_cells
            )

            file_selected_cells += (
                expected_cells
            )

            x_quantiles = np.quantile(
                x,
                [
                    0.01,
                    0.05,
                    0.25,
                    0.50,
                    0.75,
                    0.95,
                    0.99,
                ],
            )

            y_quantiles = np.quantile(
                y,
                [
                    0.01,
                    0.05,
                    0.25,
                    0.50,
                    0.75,
                    0.95,
                    0.99,
                ],
            )

            centered = (
                finite_coordinates
                - finite_coordinates.mean(
                    axis=0,
                    keepdims=True,
                )
            )

            covariance = np.cov(
                centered,
                rowvar=False,
                ddof=0,
            )

            eigenvalues = np.linalg.eigvalsh(
                covariance
            )

            eigenvalues = np.sort(
                np.maximum(
                    eigenvalues,
                    0.0,
                )
            )[::-1]

            principal_axis_ratio = (
                float(
                    math.sqrt(
                        eigenvalues[0]
                        / eigenvalues[1]
                    )
                )
                if eigenvalues[1] > 0
                else float("inf")
            )

            unique_coordinates = int(
                np.unique(
                    finite_coordinates,
                    axis=0,
                ).shape[0]
            )

            duplicate_cells = (
                finite_cells
                - unique_coordinates
            )

            duplicate_fraction = (
                duplicate_cells
                / finite_cells
            )

            metadata = section_metadata[
                archive_name
            ]

            geometry_rows.append(
                {
                    "effective_analysis_set": (
                        metadata[
                            "effective_analysis_set"
                        ]
                    ),
                    "archive_name": archive_name,
                    "donor_id": metadata[
                        "donor_id"
                    ],
                    "gestational_week": (
                        metadata[
                            "gestational_week"
                        ]
                    ),
                    "panel_class": metadata[
                        "panel_class"
                    ],
                    "processed_H5AD": file_name,
                    "selected_cells": (
                        expected_cells
                    ),
                    "finite_coordinate_cells": (
                        finite_cells
                    ),
                    "finite_fraction": (
                        finite_fraction
                    ),
                    "x_min": x_min,
                    "x_max": x_max,
                    "x_span": x_span,
                    "y_min": y_min,
                    "y_max": y_max,
                    "y_span": y_span,
                    "x_mean": float(
                        x.mean()
                    ),
                    "y_mean": float(
                        y.mean()
                    ),
                    "x_population_SD": float(
                        x.std(
                            ddof=0
                        )
                    ),
                    "y_population_SD": float(
                        y.std(
                            ddof=0
                        )
                    ),
                    "principal_axis_ratio": (
                        principal_axis_ratio
                    ),
                    "coordinate_units": (
                        "native_processed_coordinate_units"
                    ),
                    "coordinate_orientation_interpreted": (
                        False
                    ),
                }
            )

            quality_rows.append(
                {
                    "archive_name": archive_name,
                    "processed_H5AD": file_name,
                    "selected_cells": (
                        expected_cells
                    ),
                    "finite_coordinate_cells": (
                        finite_cells
                    ),
                    "nonfinite_coordinate_cells": (
                        nonfinite_cells
                    ),
                    "finite_fraction": (
                        finite_fraction
                    ),
                    "unique_coordinate_pairs": (
                        unique_coordinates
                    ),
                    "duplicate_coordinate_cells": (
                        duplicate_cells
                    ),
                    "duplicate_coordinate_fraction": (
                        duplicate_fraction
                    ),
                    "x_span_positive": (
                        x_span > 0.0
                    ),
                    "y_span_positive": (
                        y_span > 0.0
                    ),
                    "non_degenerate_2D_geometry": (
                        non_degenerate
                    ),
                    "x_q01": float(
                        x_quantiles[0]
                    ),
                    "x_q05": float(
                        x_quantiles[1]
                    ),
                    "x_q25": float(
                        x_quantiles[2]
                    ),
                    "x_q50": float(
                        x_quantiles[3]
                    ),
                    "x_q75": float(
                        x_quantiles[4]
                    ),
                    "x_q95": float(
                        x_quantiles[5]
                    ),
                    "x_q99": float(
                        x_quantiles[6]
                    ),
                    "y_q01": float(
                        y_quantiles[0]
                    ),
                    "y_q05": float(
                        y_quantiles[1]
                    ),
                    "y_q25": float(
                        y_quantiles[2]
                    ),
                    "y_q50": float(
                        y_quantiles[3]
                    ),
                    "y_q75": float(
                        y_quantiles[4]
                    ),
                    "y_q95": float(
                        y_quantiles[5]
                    ),
                    "y_q99": float(
                        y_quantiles[6]
                    ),
                }
            )

            sample_size = min(
                sample_cells_per_section,
                finite_cells,
            )

            sample_indices = np.linspace(
                0,
                finite_cells - 1,
                num=sample_size,
                dtype=np.int64,
            )

            sampled = finite_coordinates[
                sample_indices,
                :
            ]

            for sample_rank, (
                x_value,
                y_value,
            ) in enumerate(
                sampled,
                start=1,
            ):

                sample_rows.append(
                    {
                        "archive_name": archive_name,
                        "donor_id": metadata[
                            "donor_id"
                        ],
                        "gestational_week": (
                            metadata[
                                "gestational_week"
                            ]
                        ),
                        "processed_H5AD": (
                            file_name
                        ),
                        "sample_rank": (
                            sample_rank
                        ),
                        "x_native": float(
                            x_value
                        ),
                        "y_native": float(
                            y_value
                        ),
                    }
                )

        processing_rows.append(
            {
                "processed_H5AD": file_name,
                "selected_sections": len(
                    section_specs
                ),
                "selected_cells": (
                    file_selected_cells
                ),
                "spatial_embedding": (
                    "obsm/spatial"
                ),
                "spatial_rows": (
                    spatial_rows
                ),
                "spatial_columns": 2,
                "spatial_values_read": (
                    spatial_values_read
                ),
                "expression_values_accessed": (
                    False
                ),
                "H1_metadata_accessed": (
                    False
                ),
                "H2_metadata_accessed": (
                    False
                ),
            }
        )

        guard_rows.append(
            {
                "processed_H5AD": file_name,
                "sample_metadata_accessed": True,
                "region_metadata_accessed": True,
                "spatial_coordinate_values_accessed": (
                    True
                ),
                "H1_metadata_accessed": False,
                "H2_metadata_accessed": False,
                "H3_metadata_accessed": False,
                "X_values_accessed": False,
                "raw_X_values_accessed": False,
                "module_scores_computed": False,
                "spatial_module_statistics_computed": (
                    False
                ),
                "formal_spatial_hypothesis_tests_performed": (
                    False
                ),
            }
        )

if len(
    geometry_rows
) != 12:
    raise SystemExit(
        f"FAIL: expected 12 geometry rows; "
        f"observed {len(geometry_rows)}."
    )

if len(
    quality_rows
) != 12:
    raise SystemExit(
        f"FAIL: expected 12 coordinate-quality "
        f"rows; observed {len(quality_rows)}."
    )

if len(
    processing_rows
) != 6:
    raise SystemExit(
        f"FAIL: expected six processing rows; "
        f"observed {len(processing_rows)}."
    )

expected_sample_rows = (
    12
    * sample_cells_per_section
)

if len(
    sample_rows
) != expected_sample_rows:
    raise SystemExit(
        f"FAIL: expected {expected_sample_rows} "
        f"sampled-coordinate rows; observed "
        f"{len(sample_rows)}."
    )

write_tsv(
    geometry_dir
    / "phase10B5_P4G1_section_spatial_geometry.tsv",
    geometry_rows,
    [
        "effective_analysis_set",
        "archive_name",
        "donor_id",
        "gestational_week",
        "panel_class",
        "processed_H5AD",
        "selected_cells",
        "finite_coordinate_cells",
        "finite_fraction",
        "x_min",
        "x_max",
        "x_span",
        "y_min",
        "y_max",
        "y_span",
        "x_mean",
        "y_mean",
        "x_population_SD",
        "y_population_SD",
        "principal_axis_ratio",
        "coordinate_units",
        "coordinate_orientation_interpreted",
    ],
)

write_tsv(
    quality_dir
    / "phase10B5_P4G1_coordinate_quality.tsv",
    quality_rows,
    [
        "archive_name",
        "processed_H5AD",
        "selected_cells",
        "finite_coordinate_cells",
        "nonfinite_coordinate_cells",
        "finite_fraction",
        "unique_coordinate_pairs",
        "duplicate_coordinate_cells",
        "duplicate_coordinate_fraction",
        "x_span_positive",
        "y_span_positive",
        "non_degenerate_2D_geometry",
        "x_q01",
        "x_q05",
        "x_q25",
        "x_q50",
        "x_q75",
        "x_q95",
        "x_q99",
        "y_q01",
        "y_q05",
        "y_q25",
        "y_q50",
        "y_q75",
        "y_q95",
        "y_q99",
    ],
)

sample_path = (
    sample_dir
    / "phase10B5_P4G1_deterministic_spatial_geometry_sample.tsv.gz"
)

with gzip.open(
    sample_path,
    "wt",
    encoding="utf-8",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "archive_name",
            "donor_id",
            "gestational_week",
            "processed_H5AD",
            "sample_rank",
            "x_native",
            "y_native",
        ],
        delimiter="\t",
        lineterminator="\n",
    )

    writer.writeheader()
    writer.writerows(
        sample_rows
    )

write_tsv(
    processing_dir
    / "phase10B5_P4G1_file_processing_audit.tsv",
    processing_rows,
    [
        "processed_H5AD",
        "selected_sections",
        "selected_cells",
        "spatial_embedding",
        "spatial_rows",
        "spatial_columns",
        "spatial_values_read",
        "expression_values_accessed",
        "H1_metadata_accessed",
        "H2_metadata_accessed",
    ],
)

write_tsv(
    guard_dir
    / "phase10B5_P4G1_access_guard.tsv",
    guard_rows,
    [
        "processed_H5AD",
        "sample_metadata_accessed",
        "region_metadata_accessed",
        "spatial_coordinate_values_accessed",
        "H1_metadata_accessed",
        "H2_metadata_accessed",
        "H3_metadata_accessed",
        "X_values_accessed",
        "raw_X_values_accessed",
        "module_scores_computed",
        "spatial_module_statistics_computed",
        "formal_spatial_hypothesis_tests_performed",
    ],
)

technical_pass = (
    len(
        geometry_rows
    ) == 12
    and len(
        quality_rows
    ) == 12
    and len(
        processing_rows
    ) == 6
    and total_selected_cells
    == 6_062_942
    and total_finite_cells
    == 6_062_942
    and total_nonfinite_cells
    == 0
    and all_sections_non_degenerate
    and all_sections_finite
    and len(
        sample_rows
    ) == 60_000
)

status_value = (
    "passed_phase10B5_P4G1_selected_section_"
    "spatial_coordinate_integrity_and_geometry_"
    "audit_ready_for_descriptive_module_"
    "spatial_mapping"
    if technical_pass
    else (
        "phase10B5_P4G1_requires_manual_review"
    )
)

status = {
    "phase": "phase10B5_P4G1",
    "H5AD_files_opened": len(
        processing_rows
    ),
    "selected_sections_audited": len(
        geometry_rows
    ),
    "selected_cells_audited": (
        total_selected_cells
    ),
    "finite_coordinate_cells": (
        total_finite_cells
    ),
    "nonfinite_coordinate_cells": (
        total_nonfinite_cells
    ),
    "sections_with_fully_finite_coordinates": sum(
        int(
            float(
                row[
                    "finite_fraction"
                ]
            ) == 1.0
        )
        for row in quality_rows
    ),
    "sections_with_non_degenerate_2D_geometry": sum(
        bool(
            row[
                "non_degenerate_2D_geometry"
            ]
        )
        for row in quality_rows
    ),
    "deterministic_geometry_sample_rows": len(
        sample_rows
    ),
    "spatial_embedding": "obsm/spatial",
    "coordinate_units": (
        "native_processed_coordinate_units"
    ),
    "coordinate_units_inferred": False,
    "coordinate_orientation_interpreted": False,
    "spatial_coordinate_values_accessed": True,
    "expression_values_accessed": False,
    "H1_metadata_accessed": False,
    "H2_metadata_accessed": False,
    "H3_metadata_accessed": False,
    "module_scores_computed": False,
    "spatial_module_statistics_computed": False,
    "formal_spatial_hypothesis_tests_performed": (
        False
    ),
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P4G1_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B5_P4G1_status.tsv",
    [status],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P4G1 SPATIAL "
    "COORDINATE INTEGRITY AND GEOMETRY AUDIT =====",
    "",
    (
        "H5AD files opened: "
        f"{len(processing_rows)}/6"
    ),
    (
        "Selected sections audited: "
        f"{len(geometry_rows)}/12"
    ),
    (
        "Selected cells audited: "
        f"{total_selected_cells}"
    ),
    (
        "Finite coordinate cells: "
        f"{total_finite_cells}"
    ),
    (
        "Nonfinite coordinate cells: "
        f"{total_nonfinite_cells}"
    ),
    (
        "Sections with fully finite coordinates: "
        f"{sum(float(row['finite_fraction']) == 1.0 for row in quality_rows)}/12"
    ),
    (
        "Sections with non-degenerate 2D geometry: "
        f"{sum(bool(row['non_degenerate_2D_geometry']) for row in quality_rows)}/12"
    ),
    (
        "Deterministic geometry sample rows: "
        f"{len(sample_rows)}"
    ),
    "",
    "Spatial embedding: obsm/spatial",
    (
        "Coordinate units: native processed "
        "coordinate units"
    ),
    "Coordinate units inferred: FALSE",
    "Coordinate orientation interpreted: FALSE",
    "Spatial coordinate values accessed: TRUE",
    "Expression values accessed: FALSE",
    "H1 metadata accessed: FALSE",
    "H2 metadata accessed: FALSE",
    "H3 metadata accessed: FALSE",
    "Module scores computed: FALSE",
    "Spatial module statistics computed: FALSE",
    (
        "Formal spatial hypothesis tests "
        "performed: FALSE"
    ),
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P4G1 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P4G1_report.txt"
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
    "\n===== SECTION GEOMETRY ====="
)

for row in geometry_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "archive_name",
                "gestational_week",
                "selected_cells",
                "finite_fraction",
                "x_span",
                "y_span",
                "principal_axis_ratio",
            )
        )
    )

if not technical_pass:
    raise SystemExit(
        "Phase 10B5-P4G1 requires manual review."
    )
