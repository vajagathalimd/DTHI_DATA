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
geometry_path = Path(sys.argv[3])
plan_path = Path(sys.argv[4])
moment_path = Path(sys.argv[5])
design_path = Path(sys.argv[6])
out = Path(sys.argv[7])

map_dir = out / "01_spatial_bin_maps"
summary_dir = out / "02_map_summary"
reconciliation_dir = out / "03_reconciliation"
processing_dir = out / "04_processing_audit"
guard_dir = out / "05_access_guard"
audit_dir = out / "06_audit"

for directory in (
    map_dir,
    summary_dir,
    reconciliation_dir,
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

core_modules = {
    "activity_dependent_plasticity",
    "astrocyte_maturation_metabolic_support",
    "neurogenesis_migration_layering",
    "patterning_arealization",
    "progenitor_radial_glia",
}

grid_bins_per_axis = 64
total_grid_bins = (
    grid_bins_per_axis
    * grid_bins_per_axis
)

minimum_cells_for_display = 20
row_chunk_size = 100_000
variance_tolerance = 1e-12
mean_tolerance = 5e-5


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


def read_string_object(
    obj: h5py.Group | h5py.Dataset,
) -> list[str]:

    if isinstance(
        obj,
        h5py.Dataset,
    ):

        return [
            decode(value)
            for value in np.asarray(
                obj[:]
            ).reshape(-1)
        ]

    if "categories" in obj:

        return read_string_object(
            obj["categories"]
        )

    if "values" in obj:

        return read_string_object(
            obj["values"]
        )

    raise ValueError(
        f"Unsupported string object: {obj.name}"
    )


def matrix_shape(
    obj: h5py.Group | h5py.Dataset,
) -> tuple[int, int]:

    if isinstance(
        obj,
        h5py.Dataset,
    ):

        if obj.ndim != 2:

            raise ValueError(
                f"Non-2D matrix: {obj.name}"
            )

        return (
            int(
                obj.shape[0]
            ),
            int(
                obj.shape[1]
            ),
        )

    shape = np.asarray(
        obj.attrs.get(
            "shape",
            [],
        )
    ).reshape(-1)

    if shape.size != 2:

        raise ValueError(
            f"Missing matrix shape: {obj.name}"
        )

    return (
        int(
            shape[0]
        ),
        int(
            shape[1]
        ),
    )


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

    elif (
        isinstance(
            spatial,
            h5py.Group,
        )
        and "values" in spatial
        and isinstance(
            spatial["values"],
            h5py.Dataset,
        )
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


geometry_rows, _ = read_tsv(
    geometry_path
)

plan_rows, _ = read_tsv(
    plan_path
)

moment_rows, _ = read_tsv(
    moment_path
)

design_rows, _ = read_tsv(
    design_path
)

if len(
    geometry_rows
) != 12:

    raise SystemExit(
        "FAIL: expected 12 geometry rows."
    )

if len(
    plan_rows
) != 68:

    raise SystemExit(
        "FAIL: expected 68 localization "
        "estimands."
    )

if len(
    moment_rows
) != 488:

    raise SystemExit(
        "FAIL: expected 488 section-gene "
        "moment rows."
    )

if len(
    design_rows
) != 12:

    raise SystemExit(
        "FAIL: expected 12 revised sections."
    )

geometry_lookup = {
    row[
        "archive_name"
    ]: row
    for row in geometry_rows
}

if len(
    geometry_lookup
) != 12:

    raise SystemExit(
        "FAIL: duplicate geometry section."
    )

if not all(
    float(
        row[
            "finite_fraction"
        ]
    ) == 1.0
    for row in geometry_rows
):

    raise SystemExit(
        "FAIL: a section lacks fully finite "
        "coordinates."
    )

if not all(
    float(
        row[
            "x_span"
        ]
    ) > 0.0
    and float(
        row[
            "y_span"
        ]
    ) > 0.0
    for row in geometry_rows
):

    raise SystemExit(
        "FAIL: a section has degenerate geometry."
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

if len(
    section_metadata
) != 12:

    raise SystemExit(
        "FAIL: duplicate revised section."
    )

module_genes: dict[
    tuple[str, str],
    list[str],
] = {}

section_modules: dict[
    str,
    list[str],
] = defaultdict(list)

for row in plan_rows:

    archive_name = row[
        "archive_name"
    ]

    module = row[
        "module"
    ]

    genes = [
        gene
        for gene in row[
            "gene_symbols"
        ].split(";")
        if gene
    ]

    if len(
        genes
    ) != int(
        row[
            "gene_count"
        ]
    ):

        raise SystemExit(
            f"FAIL: gene-count mismatch for "
            f"{archive_name}, {module}."
        )

    module_genes[
        (
            archive_name,
            module,
        )
    ] = genes

    section_modules[
        archive_name
    ].append(
        module
    )

for archive_name in section_modules:

    section_modules[
        archive_name
    ] = sorted(
        set(
            section_modules[
                archive_name
            ]
        )
    )

if len(
    section_modules
) != 12:

    raise SystemExit(
        "FAIL: localization plan does not "
        "cover all 12 sections."
    )

core_map_definitions = sum(
    module in core_modules
    for modules in section_modules.values()
    for module in modules
)

expanded_map_definitions = sum(
    module not in core_modules
    for modules in section_modules.values()
    for module in modules
)

primary_map_definitions = sum(
    len(
        section_modules[
            archive_name
        ]
    )
    for archive_name, metadata
    in section_metadata.items()
    if metadata[
        "effective_analysis_set"
    ] == "primary"
)

sensitivity_map_definitions = sum(
    len(
        section_modules[
            archive_name
        ]
    )
    for archive_name, metadata
    in section_metadata.items()
    if metadata[
        "effective_analysis_set"
    ] == "section_sensitivity"
)

if core_map_definitions != 60:

    raise SystemExit(
        f"FAIL: expected 60 core maps; "
        f"observed {core_map_definitions}."
    )

if expanded_map_definitions != 8:

    raise SystemExit(
        f"FAIL: expected 8 expanded maps; "
        f"observed {expanded_map_definitions}."
    )

if primary_map_definitions != 48:

    raise SystemExit(
        f"FAIL: expected 48 primary maps; "
        f"observed {primary_map_definitions}."
    )

if sensitivity_map_definitions != 20:

    raise SystemExit(
        f"FAIL: expected 20 sensitivity maps; "
        f"observed {sensitivity_map_definitions}."
    )

moment_lookup: dict[
    tuple[str, str],
    tuple[float, float],
] = {}

for row in moment_rows:

    moment_lookup[
        (
            row[
                "archive_name"
            ],
            row[
                "gene_symbol"
            ],
        )
    ] = (
        float(
            row[
                "section_gene_mean"
            ]
        ),
        float(
            row[
                "section_gene_population_SD"
            ]
        ),
    )

for (
    archive_name,
    module,
), genes in module_genes.items():

    for gene in genes:

        if (
            archive_name,
            gene,
        ) not in moment_lookup:

            raise SystemExit(
                f"FAIL: missing moment for "
                f"{archive_name}, {gene}."
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
        "FAIL: revised sections do not cover "
        "all six H5AD files."
    )

bin_map_path = (
    map_dir
    / "phase10B5_P4G2_section_module_spatial_bins.tsv.gz"
)

bin_columns = [
    "effective_analysis_set",
    "archive_name",
    "donor_id",
    "gestational_week",
    "panel_class",
    "processed_H5AD",
    "scoring_family",
    "module",
    "grid_bins_per_axis",
    "x_bin",
    "y_bin",
    "bin_id",
    "x_center_native",
    "y_center_native",
    "x_center_normalized",
    "y_center_normalized",
    "cell_count",
    "mean_module_score",
    "population_SD_module_score",
    "figure_display_eligible",
    "minimum_cells_for_display",
    "coordinate_orientation_interpreted",
    "cross_section_registration_performed",
]

map_summary_rows: list[
    dict[str, Any]
] = []

reconciliation_rows: list[
    dict[str, Any]
] = []

processing_rows: list[
    dict[str, Any]
] = []

guard_rows: list[
    dict[str, Any]
] = []

files_processed = 0
sections_processed = 0
maps_processed = 0
bin_rows_written = 0

selected_cells_processed = 0
transient_cell_module_scores = 0
cell_module_bin_contributions = 0

nonfinite_score_values = 0
maximum_absolute_map_mean = 0.0

maps_with_display_bins = 0

with gzip.open(
    bin_map_path,
    "wt",
    encoding="utf-8",
    newline="",
) as bin_handle:

    bin_writer = csv.DictWriter(
        bin_handle,
        fieldnames=bin_columns,
        delimiter="\t",
        lineterminator="\n",
    )

    bin_writer.writeheader()

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
                f"FAIL: missing H5AD: "
                f"{h5ad_path}"
            )

        archives = sections_by_file[
            file_name
        ]

        file_modules = sorted(
            set().union(
                *(
                    set(
                        section_modules[
                            archive_name
                        ]
                    )
                    for archive_name in archives
                )
            )
        )

        for archive_name in archives:

            if set(
                section_modules[
                    archive_name
                ]
            ) != set(
                file_modules
            ):

                raise SystemExit(
                    f"FAIL: incompatible module sets "
                    f"within {file_name}."
                )

        required_genes = sorted(
            set().union(
                *(
                    set(
                        module_genes[
                            (
                                archive_name,
                                module,
                            )
                        ]
                    )
                    for archive_name in archives
                    for module in section_modules[
                        archive_name
                    ]
                )
            )
        )

        print(
            f"[{file_index:02d}/06] "
            f"Spatial module binning: "
            f"{file_name}",
            flush=True,
        )

        with h5py.File(
            h5ad_path,
            "r",
        ) as handle:

            files_processed += 1

            if "X" not in handle:

                raise SystemExit(
                    f"FAIL: /X absent from "
                    f"{file_name}."
                )

            x_object = handle[
                "X"
            ]

            encoding = attr_text(
                x_object.attrs.get(
                    "encoding-type",
                    "",
                )
            ).lower()

            if "csr" not in encoding:

                raise SystemExit(
                    f"FAIL: /X is not CSR in "
                    f"{file_name}: {encoding}"
                )

            if not all(
                key in x_object
                for key in (
                    "data",
                    "indices",
                    "indptr",
                )
            ):

                raise SystemExit(
                    f"FAIL: incomplete CSR object "
                    f"in {file_name}."
                )

            matrix_rows, matrix_columns = (
                matrix_shape(
                    x_object
                )
            )

            spatial = resolve_spatial_dataset(
                handle
            )

            if int(
                spatial.shape[0]
            ) != matrix_rows:

                raise SystemExit(
                    f"FAIL: spatial rows differ "
                    f"from /X rows in {file_name}."
                )

            obs = handle[
                "obs"
            ]

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

            if (
                int(
                    sample_codes.shape[0]
                )
                != matrix_rows
                or int(
                    region_codes.shape[0]
                )
                != matrix_rows
            ):

                raise SystemExit(
                    f"FAIL: obs row count mismatch "
                    f"in {file_name}."
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

            var = handle[
                "var"
            ]

            var_index_key = attr_text(
                var.attrs.get(
                    "_index",
                    "_index",
                )
            )

            matrix_genes = [
                gene.strip().upper()
                for gene in read_string_object(
                    var[
                        var_index_key
                    ]
                )
            ]

            if len(
                matrix_genes
            ) != matrix_columns:

                raise SystemExit(
                    f"FAIL: var/X dimension "
                    f"mismatch in {file_name}."
                )

            matrix_gene_position = {
                gene: index
                for index, gene
                in enumerate(
                    matrix_genes
                )
            }

            missing_genes = [
                gene
                for gene in required_genes
                if gene not in matrix_gene_position
            ]

            if missing_genes:

                raise SystemExit(
                    f"FAIL: required genes absent "
                    f"from {file_name}: "
                    + ";".join(
                        missing_genes
                    )
                )

            local_gene_position = {
                gene: index
                for index, gene
                in enumerate(
                    required_genes
                )
            }

            target_lookup = np.full(
                matrix_columns,
                -1,
                dtype=np.int32,
            )

            for gene, local_index in (
                local_gene_position.items()
            ):

                target_lookup[
                    matrix_gene_position[
                        gene
                    ]
                ] = local_index

            module_position = {
                module: index
                for index, module
                in enumerate(
                    file_modules
                )
            }

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
                        f"FAIL: sample "
                        f"{sample_value} absent "
                        f"from {file_name}."
                    )

                if region_value not in region_lookup:

                    raise SystemExit(
                        f"FAIL: region "
                        f"{region_value} absent "
                        f"from {file_name}."
                    )

                geometry = geometry_lookup[
                    archive_name
                ]

                section_specs.append(
                    {
                        "archive_name": (
                            archive_name
                        ),
                        "sample_code": (
                            sample_lookup[
                                sample_value
                            ]
                        ),
                        "region_code": (
                            region_lookup[
                                region_value
                            ]
                        ),
                        "expected_cells": (
                            metadata[
                                "selected_cell_count"
                            ]
                        ),
                        "x_min": float(
                            geometry[
                                "x_min"
                            ]
                        ),
                        "x_span": float(
                            geometry[
                                "x_span"
                            ]
                        ),
                        "y_min": float(
                            geometry[
                                "y_min"
                            ]
                        ),
                        "y_span": float(
                            geometry[
                                "y_span"
                            ]
                        ),
                    }
                )

            section_count = len(
                section_specs
            )

            module_count = len(
                file_modules
            )

            gene_count = len(
                required_genes
            )

            baseline = np.zeros(
                (
                    section_count,
                    module_count,
                ),
                dtype=np.float64,
            )

            coefficients = np.zeros(
                (
                    section_count,
                    gene_count,
                    module_count,
                ),
                dtype=np.float64,
            )

            for section_index, spec in enumerate(
                section_specs
            ):

                archive_name = spec[
                    "archive_name"
                ]

                for module in file_modules:

                    module_index = (
                        module_position[
                            module
                        ]
                    )

                    genes = module_genes[
                        (
                            archive_name,
                            module,
                        )
                    ]

                    denominator = float(
                        len(
                            genes
                        )
                    )

                    for gene in genes:

                        mean_value, SD_value = (
                            moment_lookup[
                                (
                                    archive_name,
                                    gene,
                                )
                            ]
                        )

                        if (
                            SD_value
                            <= variance_tolerance
                        ):

                            continue

                        gene_index = (
                            local_gene_position[
                                gene
                            ]
                        )

                        baseline[
                            section_index,
                            module_index,
                        ] += (
                            -mean_value
                            / SD_value
                            / denominator
                        )

                        coefficients[
                            section_index,
                            gene_index,
                            module_index,
                        ] += (
                            1.0
                            / SD_value
                            / denominator
                        )

            bin_counts = np.zeros(
                (
                    section_count,
                    total_grid_bins,
                ),
                dtype=np.int64,
            )

            bin_score_sums = np.zeros(
                (
                    section_count,
                    module_count,
                    total_grid_bins,
                ),
                dtype=np.float64,
            )

            bin_score_sumsq = np.zeros(
                (
                    section_count,
                    module_count,
                    total_grid_bins,
                ),
                dtype=np.float64,
            )

            selected_counts = np.zeros(
                section_count,
                dtype=np.int64,
            )

            data_dataset = x_object[
                "data"
            ]

            indices_dataset = x_object[
                "indices"
            ]

            indptr_dataset = x_object[
                "indptr"
            ]

            CSR_values_read_from_selected_chunks = 0
            selected_CSR_values = 0
            target_CSR_values = 0
            spatial_values_read_from_selected_chunks = 0

            for start in range(
                0,
                matrix_rows,
                row_chunk_size,
            ):

                end = min(
                    start + row_chunk_size,
                    matrix_rows,
                )

                sample_chunk = sample_codes[
                    start:end
                ]

                region_chunk = region_codes[
                    start:end
                ]

                row_assignment = np.full(
                    end - start,
                    -1,
                    dtype=np.int16,
                )

                for section_index, spec in enumerate(
                    section_specs
                ):

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
                        (
                            row_assignment
                            >= 0
                        )
                        & mask
                    ):

                        raise SystemExit(
                            f"FAIL: overlapping "
                            f"section mappings in "
                            f"{file_name}."
                        )

                    row_assignment[
                        mask
                    ] = section_index

                selected_rows = (
                    row_assignment
                    >= 0
                )

                if not np.any(
                    selected_rows
                ):

                    continue

                selected_section_indices = (
                    row_assignment[
                        selected_rows
                    ]
                )

                selected_counts += np.bincount(
                    selected_section_indices,
                    minlength=section_count,
                ).astype(
                    np.int64
                )

                coordinate_chunk = np.asarray(
                    spatial[
                        start:end,
                        :
                    ],
                    dtype=np.float64,
                )

                spatial_values_read_from_selected_chunks += int(
                    coordinate_chunk.size
                )

                selected_coordinates = (
                    coordinate_chunk[
                        selected_rows,
                        :
                    ]
                )

                if not np.isfinite(
                    selected_coordinates
                ).all():

                    raise SystemExit(
                        f"FAIL: nonfinite selected "
                        f"coordinates in {file_name}."
                    )

                scores = baseline[
                    selected_section_indices
                ].copy()

                selected_positions = np.full(
                    end - start,
                    -1,
                    dtype=np.int64,
                )

                selected_positions[
                    selected_rows
                ] = np.arange(
                    scores.shape[0],
                    dtype=np.int64,
                )

                indptr_chunk = np.asarray(
                    indptr_dataset[
                        start:end + 1
                    ],
                    dtype=np.int64,
                )

                data_start = int(
                    indptr_chunk[0]
                )

                data_end = int(
                    indptr_chunk[-1]
                )

                if data_end > data_start:

                    CSR_values_read_from_selected_chunks += (
                        data_end
                        - data_start
                    )

                    indices_chunk = np.asarray(
                        indices_dataset[
                            data_start:data_end
                        ],
                        dtype=np.int64,
                    )

                    values_chunk = np.asarray(
                        data_dataset[
                            data_start:data_end
                        ],
                        dtype=np.float64,
                    )

                    local_rows = np.repeat(
                        np.arange(
                            end - start,
                            dtype=np.int64,
                        ),
                        np.diff(
                            indptr_chunk
                        ),
                    )

                    entry_sections = (
                        row_assignment[
                            local_rows
                        ]
                    )

                    selected_entry_mask = (
                        entry_sections
                        >= 0
                    )

                    selected_CSR_values += int(
                        np.count_nonzero(
                            selected_entry_mask
                        )
                    )

                    if np.any(
                        selected_entry_mask
                    ):

                        selected_columns = (
                            indices_chunk[
                                selected_entry_mask
                            ]
                        )

                        selected_values = (
                            values_chunk[
                                selected_entry_mask
                            ]
                        )

                        selected_entry_sections = (
                            entry_sections[
                                selected_entry_mask
                            ]
                        )

                        selected_entry_rows = (
                            local_rows[
                                selected_entry_mask
                            ]
                        )

                        target_gene_indices = (
                            target_lookup[
                                selected_columns
                            ]
                        )

                        target_mask = (
                            target_gene_indices
                            >= 0
                        )

                        target_CSR_values += int(
                            np.count_nonzero(
                                target_mask
                            )
                        )

                        if np.any(
                            target_mask
                        ):

                            target_values = (
                                selected_values[
                                    target_mask
                                ]
                            )

                            target_sections = (
                                selected_entry_sections[
                                    target_mask
                                ]
                            )

                            target_genes = (
                                target_gene_indices[
                                    target_mask
                                ]
                            )

                            target_row_positions = (
                                selected_positions[
                                    selected_entry_rows[
                                        target_mask
                                    ]
                                ]
                            )

                            for module_index in range(
                                module_count
                            ):

                                module_coefficients = (
                                    coefficients[
                                        target_sections,
                                        target_genes,
                                        module_index,
                                    ]
                                )

                                active = (
                                    module_coefficients
                                    != 0.0
                                )

                                if not np.any(
                                    active
                                ):

                                    continue

                                additions = np.bincount(
                                    target_row_positions[
                                        active
                                    ],
                                    weights=(
                                        target_values[
                                            active
                                        ]
                                        * module_coefficients[
                                            active
                                        ]
                                    ),
                                    minlength=scores.shape[0],
                                )

                                scores[
                                    :,
                                    module_index,
                                ] += additions

                if not np.isfinite(
                    scores
                ).all():

                    nonfinite_score_values += int(
                        np.count_nonzero(
                            ~np.isfinite(
                                scores
                            )
                        )
                    )

                    raise SystemExit(
                        f"FAIL: nonfinite module "
                        f"scores in {file_name}."
                    )

                transient_cell_module_scores += int(
                    scores.size
                )

                for section_index, spec in enumerate(
                    section_specs
                ):

                    section_mask = (
                        selected_section_indices
                        == section_index
                    )

                    if not np.any(
                        section_mask
                    ):

                        continue

                    section_coordinates = (
                        selected_coordinates[
                            section_mask,
                            :
                        ]
                    )

                    section_scores = scores[
                        section_mask,
                        :
                    ]

                    x_normalized = (
                        section_coordinates[
                            :,
                            0
                        ]
                        - spec[
                            "x_min"
                        ]
                    ) / spec[
                        "x_span"
                    ]

                    y_normalized = (
                        section_coordinates[
                            :,
                            1
                        ]
                        - spec[
                            "y_min"
                        ]
                    ) / spec[
                        "y_span"
                    ]

                    x_bin = np.floor(
                        x_normalized
                        * grid_bins_per_axis
                    ).astype(
                        np.int64
                    )

                    y_bin = np.floor(
                        y_normalized
                        * grid_bins_per_axis
                    ).astype(
                        np.int64
                    )

                    x_bin = np.clip(
                        x_bin,
                        0,
                        grid_bins_per_axis - 1,
                    )

                    y_bin = np.clip(
                        y_bin,
                        0,
                        grid_bins_per_axis - 1,
                    )

                    bin_id = (
                        y_bin
                        * grid_bins_per_axis
                        + x_bin
                    )

                    counts = np.bincount(
                        bin_id,
                        minlength=total_grid_bins,
                    ).astype(
                        np.int64
                    )

                    bin_counts[
                        section_index,
                        :,
                    ] += counts

                    for module_index in range(
                        module_count
                    ):

                        values = section_scores[
                            :,
                            module_index
                        ]

                        bin_score_sums[
                            section_index,
                            module_index,
                            :,
                        ] += np.bincount(
                            bin_id,
                            weights=values,
                            minlength=total_grid_bins,
                        )

                        bin_score_sumsq[
                            section_index,
                            module_index,
                            :,
                        ] += np.bincount(
                            bin_id,
                            weights=(
                                values
                                * values
                            ),
                            minlength=total_grid_bins,
                        )

            expected_counts = np.asarray(
                [
                    spec[
                        "expected_cells"
                    ]
                    for spec in section_specs
                ],
                dtype=np.int64,
            )

            if not np.array_equal(
                selected_counts,
                expected_counts,
            ):

                details = "; ".join(
                    (
                        f"{section_specs[index]['archive_name']}:"
                        f"observed={selected_counts[index]},"
                        f"expected={expected_counts[index]}"
                    )
                    for index in range(
                        section_count
                    )
                )

                raise SystemExit(
                    f"FAIL: selected-cell "
                    f"count mismatch: {details}"
                )

            file_cell_module_contributions = 0

            for section_index, spec in enumerate(
                section_specs
            ):

                archive_name = spec[
                    "archive_name"
                ]

                metadata = section_metadata[
                    archive_name
                ]

                counts = bin_counts[
                    section_index,
                    :
                ]

                occupied = (
                    counts > 0
                )

                occupied_indices = np.flatnonzero(
                    occupied
                )

                occupied_count = int(
                    occupied_indices.size
                )

                if occupied_count == 0:

                    raise SystemExit(
                        f"FAIL: no occupied spatial "
                        f"bins for {archive_name}."
                    )

                section_bin_cell_total = int(
                    counts.sum()
                )

                expected_cells = int(
                    spec[
                        "expected_cells"
                    ]
                )

                if (
                    section_bin_cell_total
                    != expected_cells
                ):

                    raise SystemExit(
                        f"FAIL: bin-cell "
                        f"reconciliation failed for "
                        f"{archive_name}."
                    )

                sections_processed += 1
                selected_cells_processed += (
                    expected_cells
                )

                display_bins = (
                    counts
                    >= minimum_cells_for_display
                )

                display_bin_count = int(
                    np.count_nonzero(
                        display_bins
                    )
                )

                reconciliation_rows.append(
                    {
                        "archive_name": (
                            archive_name
                        ),
                        "donor_id": metadata[
                            "donor_id"
                        ],
                        "gestational_week": (
                            metadata[
                                "gestational_week"
                            ]
                        ),
                        "effective_analysis_set": (
                            metadata[
                                "effective_analysis_set"
                            ]
                        ),
                        "processed_H5AD": (
                            file_name
                        ),
                        "expected_selected_cells": (
                            expected_cells
                        ),
                        "binned_selected_cells": (
                            section_bin_cell_total
                        ),
                        "cell_count_matches": (
                            section_bin_cell_total
                            == expected_cells
                        ),
                        "occupied_spatial_bins": (
                            occupied_count
                        ),
                        "figure_display_bins": (
                            display_bin_count
                        ),
                        "grid_bins_per_axis": (
                            grid_bins_per_axis
                        ),
                    }
                )

                for module in file_modules:

                    module_index = (
                        module_position[
                            module
                        ]
                    )

                    sums = bin_score_sums[
                        section_index,
                        module_index,
                        :
                    ]

                    sumsq = bin_score_sumsq[
                        section_index,
                        module_index,
                        :
                    ]

                    means = np.zeros(
                        total_grid_bins,
                        dtype=np.float64,
                    )

                    SDs = np.zeros(
                        total_grid_bins,
                        dtype=np.float64,
                    )

                    means[
                        occupied
                    ] = (
                        sums[
                            occupied
                        ]
                        / counts[
                            occupied
                        ]
                    )

                    variances = np.maximum(
                        (
                            sumsq[
                                occupied
                            ]
                            / counts[
                                occupied
                            ]
                        )
                        - (
                            means[
                                occupied
                            ]
                            ** 2
                        ),
                        0.0,
                    )

                    SDs[
                        occupied
                    ] = np.sqrt(
                        variances
                    )

                    map_mean = float(
                        sums.sum()
                        / expected_cells
                    )

                    maximum_absolute_map_mean = max(
                        maximum_absolute_map_mean,
                        abs(
                            map_mean
                        ),
                    )

                    full_cell_variance = max(
                        float(
                            sumsq.sum()
                            / expected_cells
                            - map_mean
                            * map_mean
                        ),
                        0.0,
                    )

                    weighted_bin_mean_variance = (
                        float(
                            np.sum(
                                counts[
                                    occupied
                                ]
                                * (
                                    means[
                                        occupied
                                    ]
                                    - map_mean
                                )
                                ** 2
                            )
                            / expected_cells
                        )
                    )

                    figure_mask = (
                        display_bins
                        & occupied
                    )

                    if np.any(
                        figure_mask
                    ):

                        maps_with_display_bins += 1

                        figure_means = means[
                            figure_mask
                        ]

                    else:

                        figure_means = means[
                            occupied
                        ]

                    q02, q50, q98 = np.quantile(
                        figure_means,
                        [
                            0.02,
                            0.50,
                            0.98,
                        ],
                    )

                    map_summary_rows.append(
                        {
                            "effective_analysis_set": (
                                metadata[
                                    "effective_analysis_set"
                                ]
                            ),
                            "archive_name": (
                                archive_name
                            ),
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
                            "processed_H5AD": (
                                file_name
                            ),
                            "scoring_family": (
                                "cross_panel_harmonized_core"
                                if module in core_modules
                                else "expanded_panel_only"
                            ),
                            "module": module,
                            "selected_cells": (
                                expected_cells
                            ),
                            "grid_bins_per_axis": (
                                grid_bins_per_axis
                            ),
                            "occupied_bins": (
                                occupied_count
                            ),
                            "figure_display_bins": (
                                display_bin_count
                            ),
                            "minimum_cells_for_display": (
                                minimum_cells_for_display
                            ),
                            "median_cells_per_occupied_bin": float(
                                np.median(
                                    counts[
                                        occupied
                                    ]
                                )
                            ),
                            "maximum_cells_per_bin": int(
                                counts[
                                    occupied
                                ].max()
                            ),
                            "weighted_map_mean": (
                                map_mean
                            ),
                            "absolute_weighted_map_mean": abs(
                                map_mean
                            ),
                            "cell_module_population_SD": math.sqrt(
                                full_cell_variance
                            ),
                            "weighted_spatial_bin_mean_SD": math.sqrt(
                                max(
                                    weighted_bin_mean_variance,
                                    0.0,
                                )
                            ),
                            "figure_bin_mean_q02": float(
                                q02
                            ),
                            "figure_bin_mean_q50": float(
                                q50
                            ),
                            "figure_bin_mean_q98": float(
                                q98
                            ),
                            "formal_spatial_inference": (
                                False
                            ),
                        }
                    )

                    maps_processed += 1

                    cell_module_bin_contributions += (
                        expected_cells
                    )

                    file_cell_module_contributions += (
                        expected_cells
                    )

                    for bin_index in occupied_indices:

                        y_bin_value = int(
                            bin_index
                            // grid_bins_per_axis
                        )

                        x_bin_value = int(
                            bin_index
                            % grid_bins_per_axis
                        )

                        x_center_normalized = (
                            x_bin_value
                            + 0.5
                        ) / grid_bins_per_axis

                        y_center_normalized = (
                            y_bin_value
                            + 0.5
                        ) / grid_bins_per_axis

                        x_center_native = (
                            spec[
                                "x_min"
                            ]
                            + x_center_normalized
                            * spec[
                                "x_span"
                            ]
                        )

                        y_center_native = (
                            spec[
                                "y_min"
                            ]
                            + y_center_normalized
                            * spec[
                                "y_span"
                            ]
                        )

                        count = int(
                            counts[
                                bin_index
                            ]
                        )

                        bin_writer.writerow(
                            {
                                "effective_analysis_set": (
                                    metadata[
                                        "effective_analysis_set"
                                    ]
                                ),
                                "archive_name": (
                                    archive_name
                                ),
                                "donor_id": (
                                    metadata[
                                        "donor_id"
                                    ]
                                ),
                                "gestational_week": (
                                    metadata[
                                        "gestational_week"
                                    ]
                                ),
                                "panel_class": (
                                    metadata[
                                        "panel_class"
                                    ]
                                ),
                                "processed_H5AD": (
                                    file_name
                                ),
                                "scoring_family": (
                                    "cross_panel_harmonized_core"
                                    if module in core_modules
                                    else "expanded_panel_only"
                                ),
                                "module": (
                                    module
                                ),
                                "grid_bins_per_axis": (
                                    grid_bins_per_axis
                                ),
                                "x_bin": (
                                    x_bin_value
                                ),
                                "y_bin": (
                                    y_bin_value
                                ),
                                "bin_id": int(
                                    bin_index
                                ),
                                "x_center_native": (
                                    x_center_native
                                ),
                                "y_center_native": (
                                    y_center_native
                                ),
                                "x_center_normalized": (
                                    x_center_normalized
                                ),
                                "y_center_normalized": (
                                    y_center_normalized
                                ),
                                "cell_count": (
                                    count
                                ),
                                "mean_module_score": float(
                                    means[
                                        bin_index
                                    ]
                                ),
                                "population_SD_module_score": float(
                                    SDs[
                                        bin_index
                                    ]
                                ),
                                "figure_display_eligible": (
                                    count
                                    >= minimum_cells_for_display
                                ),
                                "minimum_cells_for_display": (
                                    minimum_cells_for_display
                                ),
                                "coordinate_orientation_interpreted": (
                                    False
                                ),
                                "cross_section_registration_performed": (
                                    False
                                ),
                            }
                        )

                        bin_rows_written += 1

            processing_rows.append(
                {
                    "processed_H5AD": (
                        file_name
                    ),
                    "selected_sections": (
                        section_count
                    ),
                    "modules_per_selected_section": (
                        module_count
                    ),
                    "required_target_genes": (
                        gene_count
                    ),
                    "selected_cells": int(
                        selected_counts.sum()
                    ),
                    "cell_module_contributions": (
                        file_cell_module_contributions
                    ),
                    "CSR_values_read_from_selected_chunks": (
                        CSR_values_read_from_selected_chunks
                    ),
                    "selected_CSR_values": (
                        selected_CSR_values
                    ),
                    "target_CSR_values": (
                        target_CSR_values
                    ),
                    "spatial_values_read_from_selected_chunks": (
                        spatial_values_read_from_selected_chunks
                    ),
                    "complete_X_matrix_materialized": (
                        False
                    ),
                    "cell_level_score_matrix_retained": (
                        False
                    ),
                }
            )

            guard_rows.append(
                {
                    "processed_H5AD": (
                        file_name
                    ),
                    "sample_metadata_accessed": (
                        True
                    ),
                    "region_metadata_accessed": (
                        True
                    ),
                    "spatial_coordinate_values_accessed": (
                        True
                    ),
                    "X_values_accessed": (
                        True
                    ),
                    "raw_X_values_accessed": (
                        False
                    ),
                    "H1_metadata_accessed": (
                        False
                    ),
                    "H2_metadata_accessed": (
                        False
                    ),
                    "H3_metadata_accessed": (
                        False
                    ),
                    "cell_level_module_scores_transiently_computed": (
                        True
                    ),
                    "cell_level_score_matrix_retained": (
                        False
                    ),
                    "descriptive_spatial_bin_summaries_computed": (
                        True
                    ),
                    "spatial_smoothing_performed": (
                        False
                    ),
                    "cross_section_spatial_registration_performed": (
                        False
                    ),
                    "coordinate_orientation_interpreted": (
                        False
                    ),
                    "formal_spatial_hypothesis_tests_performed": (
                        False
                    ),
                }
            )

expected_transient_scores = sum(
    int(
        row[
            "selected_cell_count"
        ]
    )
    for row in plan_rows
)

if expected_transient_scores != 36_733_794:

    raise SystemExit(
        f"FAIL: expected score workload "
        f"changed: {expected_transient_scores}"
    )

if len(
    reconciliation_rows
) != 12:

    raise SystemExit(
        f"FAIL: expected 12 reconciliation "
        f"rows; observed "
        f"{len(reconciliation_rows)}."
    )

if len(
    map_summary_rows
) != 68:

    raise SystemExit(
        f"FAIL: expected 68 map summaries; "
        f"observed "
        f"{len(map_summary_rows)}."
    )

write_tsv(
    summary_dir
    / "phase10B5_P4G2_section_module_spatial_map_summary.tsv",
    map_summary_rows,
    [
        "effective_analysis_set",
        "archive_name",
        "donor_id",
        "gestational_week",
        "panel_class",
        "processed_H5AD",
        "scoring_family",
        "module",
        "selected_cells",
        "grid_bins_per_axis",
        "occupied_bins",
        "figure_display_bins",
        "minimum_cells_for_display",
        "median_cells_per_occupied_bin",
        "maximum_cells_per_bin",
        "weighted_map_mean",
        "absolute_weighted_map_mean",
        "cell_module_population_SD",
        "weighted_spatial_bin_mean_SD",
        "figure_bin_mean_q02",
        "figure_bin_mean_q50",
        "figure_bin_mean_q98",
        "formal_spatial_inference",
    ],
)

write_tsv(
    reconciliation_dir
    / "phase10B5_P4G2_section_spatial_bin_reconciliation.tsv",
    reconciliation_rows,
    [
        "archive_name",
        "donor_id",
        "gestational_week",
        "effective_analysis_set",
        "processed_H5AD",
        "expected_selected_cells",
        "binned_selected_cells",
        "cell_count_matches",
        "occupied_spatial_bins",
        "figure_display_bins",
        "grid_bins_per_axis",
    ],
)

write_tsv(
    processing_dir
    / "phase10B5_P4G2_file_processing_audit.tsv",
    processing_rows,
    [
        "processed_H5AD",
        "selected_sections",
        "modules_per_selected_section",
        "required_target_genes",
        "selected_cells",
        "cell_module_contributions",
        "CSR_values_read_from_selected_chunks",
        "selected_CSR_values",
        "target_CSR_values",
        "spatial_values_read_from_selected_chunks",
        "complete_X_matrix_materialized",
        "cell_level_score_matrix_retained",
    ],
)

write_tsv(
    guard_dir
    / "phase10B5_P4G2_access_and_inference_guard.tsv",
    guard_rows,
    [
        "processed_H5AD",
        "sample_metadata_accessed",
        "region_metadata_accessed",
        "spatial_coordinate_values_accessed",
        "X_values_accessed",
        "raw_X_values_accessed",
        "H1_metadata_accessed",
        "H2_metadata_accessed",
        "H3_metadata_accessed",
        "cell_level_module_scores_transiently_computed",
        "cell_level_score_matrix_retained",
        "descriptive_spatial_bin_summaries_computed",
        "spatial_smoothing_performed",
        "cross_section_spatial_registration_performed",
        "coordinate_orientation_interpreted",
        "formal_spatial_hypothesis_tests_performed",
    ],
)

technical_pass = (
    files_processed == 6
    and sections_processed == 12
    and maps_processed == 68
    and core_map_definitions == 60
    and expanded_map_definitions == 8
    and primary_map_definitions == 48
    and sensitivity_map_definitions == 20
    and selected_cells_processed == 6_062_942
    and transient_cell_module_scores
    == 36_733_794
    and cell_module_bin_contributions
    == 36_733_794
    and nonfinite_score_values == 0
    and maximum_absolute_map_mean
    <= mean_tolerance
    and maps_with_display_bins == 68
    and all(
        row[
            "cell_count_matches"
        ]
        for row in reconciliation_rows
    )
)

status_value = (
    "passed_phase10B5_P4G2_chunked_within_section_"
    "module_spatial_bin_mapping_ready_for_"
    "descriptive_spatial_figure_generation"
    if technical_pass
    else (
        "phase10B5_P4G2_requires_manual_review"
    )
)

status = {
    "phase": "phase10B5_P4G2",
    "H5AD_files_processed": (
        files_processed
    ),
    "selected_sections_processed": (
        sections_processed
    ),
    "spatial_map_definitions": (
        maps_processed
    ),
    "core_spatial_map_definitions": (
        core_map_definitions
    ),
    "expanded_spatial_map_definitions": (
        expanded_map_definitions
    ),
    "primary_spatial_map_definitions": (
        primary_map_definitions
    ),
    "sensitivity_spatial_map_definitions": (
        sensitivity_map_definitions
    ),
    "selected_cells_processed": (
        selected_cells_processed
    ),
    "transient_cell_module_scores_computed": (
        transient_cell_module_scores
    ),
    "cell_module_bin_contributions": (
        cell_module_bin_contributions
    ),
    "spatial_bin_map_rows": (
        bin_rows_written
    ),
    "grid_bins_per_axis": (
        grid_bins_per_axis
    ),
    "minimum_cells_per_figure_bin": (
        minimum_cells_for_display
    ),
    "maps_with_figure_display_bins": (
        maps_with_display_bins
    ),
    "nonfinite_module_score_values": (
        nonfinite_score_values
    ),
    "maximum_absolute_section_module_map_mean": (
        maximum_absolute_map_mean
    ),
    "all_map_means_approximately_zero": (
        maximum_absolute_map_mean
        <= mean_tolerance
    ),
    "spatial_coordinate_values_accessed": (
        True
    ),
    "expression_values_accessed": (
        True
    ),
    "cell_level_module_scores_transiently_computed": (
        True
    ),
    "cell_level_score_matrix_retained": (
        False
    ),
    "descriptive_spatial_bin_summaries_computed": (
        True
    ),
    "spatial_smoothing_performed": (
        False
    ),
    "cross_section_spatial_registration_performed": (
        False
    ),
    "coordinate_units_inferred": (
        False
    ),
    "coordinate_orientation_interpreted": (
        False
    ),
    "H1_metadata_accessed": (
        False
    ),
    "H2_metadata_accessed": (
        False
    ),
    "H3_metadata_accessed": (
        False
    ),
    "formal_spatial_hypothesis_tests_performed": (
        False
    ),
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P4G2_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B5_P4G2_status.tsv",
    [status],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P4G2 CHUNKED "
    "DESCRIPTIVE MODULE SPATIAL MAPPING =====",
    "",
    (
        "H5AD files processed: "
        f"{files_processed}/6"
    ),
    (
        "Selected sections processed: "
        f"{sections_processed}/12"
    ),
    (
        "Spatial map definitions: "
        f"{maps_processed}/68"
    ),
    (
        "Core spatial maps: "
        f"{core_map_definitions}/60"
    ),
    (
        "Expanded spatial maps: "
        f"{expanded_map_definitions}/8"
    ),
    (
        "Primary spatial maps: "
        f"{primary_map_definitions}/48"
    ),
    (
        "Sensitivity spatial maps: "
        f"{sensitivity_map_definitions}/20"
    ),
    (
        "Selected cells processed: "
        f"{selected_cells_processed}"
    ),
    (
        "Transient cell-module scores: "
        f"{transient_cell_module_scores}"
    ),
    (
        "Cell-module bin contributions: "
        f"{cell_module_bin_contributions}"
    ),
    (
        "Spatial-bin output rows: "
        f"{bin_rows_written}"
    ),
    (
        "Maps with figure-ready bins: "
        f"{maps_with_display_bins}/68"
    ),
    (
        "Maximum absolute map mean: "
        f"{maximum_absolute_map_mean}"
    ),
    "",
    (
        "Grid resolution: "
        f"{grid_bins_per_axis} x "
        f"{grid_bins_per_axis}"
    ),
    (
        "Minimum cells per figure bin: "
        f"{minimum_cells_for_display}"
    ),
    "Spatial coordinate values accessed: TRUE",
    "Expression values accessed: TRUE",
    (
        "Cell-level module scores transiently "
        "computed: TRUE"
    ),
    "Cell-level score matrix retained: FALSE",
    (
        "Descriptive spatial-bin summaries "
        "computed: TRUE"
    ),
    "Spatial smoothing performed: FALSE",
    (
        "Cross-section spatial registration "
        "performed: FALSE"
    ),
    "Coordinate units inferred: FALSE",
    "Coordinate orientation interpreted: FALSE",
    "H1 metadata accessed: FALSE",
    "H2 metadata accessed: FALSE",
    "H3 metadata accessed: FALSE",
    (
        "Formal spatial hypothesis tests "
        "performed: FALSE"
    ),
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P4G2 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P4G2_report.txt"
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
    "\n===== SPATIAL MAP SUMMARY ====="
)

for row in map_summary_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "archive_name",
                "module",
                "selected_cells",
                "occupied_bins",
                "figure_display_bins",
                "weighted_map_mean",
                "weighted_spatial_bin_mean_SD",
                "figure_bin_mean_q02",
                "figure_bin_mean_q98",
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
        != "phase10B5_P4G2_SHA256.tsv"
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
    / "phase10B5_P4G2_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)

if not technical_pass:

    raise SystemExit(
        "Phase 10B5-P4G2 requires manual review."
    )
