from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import h5py
import numpy as np


project = Path(sys.argv[1])
payload_dir = Path(sys.argv[2])
core_plan_path = Path(sys.argv[3])
expanded_plan_path = Path(sys.argv[4])
module_sets_path = Path(sys.argv[5])
source_lock_path = Path(sys.argv[6])
out = Path(sys.argv[7])

checkpoint_dir = out / "01_file_accumulators"
moment_dir = out / "02_section_gene_moments"
reference_dir = out / "03_primary_gene_reference"
module_dir = out / "04_core_module_scores"
expanded_dir = out / "05_expanded_panel_results"
processing_dir = out / "06_processing_audit"
guard_dir = out / "07_expression_access_guard"
audit_dir = out / "08_audit"

for directory in (
    checkpoint_dir,
    moment_dir,
    reference_dir,
    module_dir,
    expanded_dir,
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

row_chunk_size = 100_000
zero_tolerance = 1e-12
variance_tolerance = 1e-12


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


def as_bool(
    value: Any,
) -> bool:

    return str(value).strip().upper() in {
        "TRUE",
        "T",
        "1",
        "YES",
    }


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

        return str(value.item())

    return str(value)


def attr_text(
    value: Any,
) -> str:

    array = np.asarray(value)

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

    if not isinstance(group, h5py.Group):

        raise TypeError(
            f"obs/{column} is not categorical."
        )

    if (
        "categories" not in group
        or "codes" not in group
    ):

        raise KeyError(
            f"obs/{column} lacks categories or codes."
        )

    categories = [
        decode(value)
        for value in group["categories"][:]
    ]

    return categories, group["codes"]


def read_string_object(
    obj: h5py.Group | h5py.Dataset,
) -> list[str]:

    if isinstance(obj, h5py.Dataset):

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

    if isinstance(obj, h5py.Dataset):

        if obj.ndim != 2:

            raise ValueError(
                f"Non-two-dimensional matrix: "
                f"{obj.name}"
            )

        return (
            int(obj.shape[0]),
            int(obj.shape[1]),
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

    return int(shape[0]), int(shape[1])


def safe_stem(
    file_name: str,
) -> str:

    return (
        file_name
        .replace(".h5ad", "")
        .replace("/", "_")
    )


core_plan_rows, core_plan_columns = read_tsv(
    core_plan_path
)

expanded_plan_rows, expanded_plan_columns = read_tsv(
    expanded_plan_path
)

module_set_rows, module_set_columns = read_tsv(
    module_sets_path
)

source_lock_rows, source_lock_columns = read_tsv(
    source_lock_path
)

if len(core_plan_rows) != 60:

    raise SystemExit(
        f"FAIL: expected 60 core estimands; "
        f"observed {len(core_plan_rows)}."
    )

if len(expanded_plan_rows) != 8:

    raise SystemExit(
        f"FAIL: expected eight expanded estimands; "
        f"observed {len(expanded_plan_rows)}."
    )

if len(module_set_rows) != 9:

    raise SystemExit(
        f"FAIL: expected nine module sets; "
        f"observed {len(module_set_rows)}."
    )

if len(source_lock_rows) != 6:

    raise SystemExit(
        f"FAIL: expected six source locks; "
        f"observed {len(source_lock_rows)}."
    )

if not all(
    row[
        "primary_expression_source"
    ] == "/X"
    for row in source_lock_rows
):

    raise SystemExit(
        "FAIL: not all files are locked to /X."
    )

if not all(
    not as_bool(
        row[
            "expression_values_accessed"
        ]
    )
    for row in source_lock_rows
):

    raise SystemExit(
        "FAIL: source lock indicates previous "
        "expression access."
    )

section_metadata: dict[
    str,
    dict[str, str],
] = {}

core_modules: dict[
    str,
    list[str],
] = {}

expanded_modules: dict[
    str,
    list[str],
] = {}

for row in module_set_rows:

    genes = [
        gene
        for gene in row[
            "gene_symbols"
        ].split(";")
        if gene
    ]

    if row[
        "scoring_family"
    ] == "cross_panel_harmonized_core":

        core_modules[
            row[
                "module"
            ]
        ] = genes

    elif row[
        "scoring_family"
    ] == "expanded_panel_only":

        expanded_modules[
            row[
                "module"
            ]
        ] = genes

if len(core_modules) != 5:

    raise SystemExit(
        "FAIL: expected five core modules."
    )

if len(expanded_modules) != 4:

    raise SystemExit(
        "FAIL: expected four expanded modules."
    )

for row in core_plan_rows:

    archive_name = row[
        "archive_name"
    ]

    candidate = {
        "effective_analysis_set": row[
            "effective_analysis_set"
        ],
        "archive_name": archive_name,
        "donor_id": row[
            "donor_id"
        ],
        "gestational_week": row[
            "gestational_week"
        ],
        "panel_class": row[
            "panel_class"
        ],
        "processed_H5AD": row[
            "processed_H5AD"
        ],
        "locked_sample_value": row[
            "locked_sample_value"
        ],
        "locked_region_value": row[
            "locked_region_value"
        ],
        "selected_cell_count": row[
            "selected_cell_count"
        ],
    }

    if archive_name in section_metadata:

        if section_metadata[
            archive_name
        ] != candidate:

            raise SystemExit(
                f"FAIL: inconsistent metadata for "
                f"{archive_name}."
            )

    else:

        section_metadata[
            archive_name
        ] = candidate

if len(section_metadata) != 12:

    raise SystemExit(
        f"FAIL: expected 12 unique sections; "
        f"observed {len(section_metadata)}."
    )

required_genes_by_section: dict[
    str,
    set[str],
] = defaultdict(set)

for row in core_plan_rows:

    required_genes_by_section[
        row[
            "archive_name"
        ]
    ].update(
        gene
        for gene in row[
            "gene_symbols"
        ].split(";")
        if gene
    )

for row in expanded_plan_rows:

    required_genes_by_section[
        row[
            "archive_name"
        ]
    ].update(
        gene
        for gene in row[
            "gene_symbols"
        ].split(";")
        if gene
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
        "FAIL: section design does not cover the "
        "six locked H5AD files."
    )

file_processing_rows: list[
    dict[str, Any]
] = []

accumulator_lookup: dict[
    str,
    dict[str, Any],
] = {}

for file_index, file_name in enumerate(
    expected_files,
    start=1,
):

    h5ad_path = payload_dir / file_name

    if not h5ad_path.is_file():

        raise SystemExit(
            f"FAIL: missing H5AD payload: "
            f"{h5ad_path}"
        )

    archives = sections_by_file[
        file_name
    ]

    required_genes = sorted(
        set().union(
            *(
                required_genes_by_section[
                    archive_name
                ]
                for archive_name in archives
            )
        )
    )

    expected_counts = np.asarray(
        [
            int(
                section_metadata[
                    archive_name
                ][
                    "selected_cell_count"
                ]
            )
            for archive_name in archives
        ],
        dtype=np.int64,
    )

    checkpoint_path = (
        checkpoint_dir
        / (
            f"phase10B5_P4E1_"
            f"{safe_stem(file_name)}_"
            "accumulator.npz"
        )
    )

    h5ad_stat = h5ad_path.stat()

    checkpoint_reused = False

    if checkpoint_path.is_file():

        try:

            with np.load(
                checkpoint_path,
                allow_pickle=False,
            ) as checkpoint:

                checkpoint_archives = [
                    str(value)
                    for value in checkpoint[
                        "archive_names"
                    ]
                ]

                checkpoint_genes = [
                    str(value)
                    for value in checkpoint[
                        "genes"
                    ]
                ]

                valid_checkpoint = (
                    checkpoint_archives == archives
                    and checkpoint_genes
                    == required_genes
                    and np.array_equal(
                        checkpoint[
                            "expected_cell_counts"
                        ],
                        expected_counts,
                    )
                    and int(
                        checkpoint[
                            "h5ad_size_bytes"
                        ][0]
                    )
                    == h5ad_stat.st_size
                    and int(
                        checkpoint[
                            "h5ad_mtime_ns"
                        ][0]
                    )
                    == h5ad_stat.st_mtime_ns
                )

                if valid_checkpoint:

                    accumulator_lookup[
                        file_name
                    ] = {
                        "archive_names": (
                            checkpoint_archives
                        ),
                        "genes": checkpoint_genes,
                        "observed_cell_counts": (
                            checkpoint[
                                "observed_cell_counts"
                            ].astype(
                                np.int64
                            )
                        ),
                        "expression_sums": (
                            checkpoint[
                                "expression_sums"
                            ].astype(
                                np.float64
                            )
                        ),
                        "expression_sumsq": (
                            checkpoint[
                                "expression_sumsq"
                            ].astype(
                                np.float64
                            )
                        ),
                        "detected_cell_counts": (
                            checkpoint[
                                "detected_cell_counts"
                            ].astype(
                                np.int64
                            )
                        ),
                        "selected_CSR_values_read": int(
                            checkpoint[
                                "selected_CSR_values_read"
                            ][0]
                        ),
                        "target_CSR_values_aggregated": int(
                            checkpoint[
                                "target_CSR_values_aggregated"
                            ][0]
                        ),
                        "non_target_CSR_values_transient": int(
                            checkpoint[
                                "non_target_CSR_values_transient"
                            ][0]
                        ),
                    }

                    checkpoint_reused = True

        except Exception:

            checkpoint_reused = False

    if checkpoint_reused:

        print(
            f"[{file_index:02d}/06] "
            f"Reusing verified accumulator: "
            f"{file_name}",
            flush=True,
        )

    else:

        print(
            f"[{file_index:02d}/06] "
            f"Streaming selected rows from /X: "
            f"{file_name}",
            flush=True,
        )

        with h5py.File(
            h5ad_path,
            "r",
        ) as handle:

            if "X" not in handle:

                raise SystemExit(
                    f"FAIL: /X missing from "
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
                    f"FAIL: incomplete CSR structure "
                    f"in {file_name}."
                )

            matrix_rows, matrix_columns = (
                matrix_shape(
                    x_object
                )
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
                    f"FAIL: obs and /X row counts "
                    f"differ in {file_name}."
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

            if var_index_key not in var:

                raise SystemExit(
                    f"FAIL: var index missing in "
                    f"{file_name}."
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
                    f"FAIL: var and /X column counts "
                    f"differ in {file_name}."
                )

            gene_position = {
                gene: index
                for index, gene
                in enumerate(
                    matrix_genes
                )
            }

            missing_genes = [
                gene
                for gene in required_genes
                if gene not in gene_position
            ]

            if missing_genes:

                raise SystemExit(
                    f"FAIL: required genes missing "
                    f"from {file_name}: "
                    + ";".join(
                        missing_genes
                    )
                )

            target_lookup = np.full(
                matrix_columns,
                -1,
                dtype=np.int32,
            )

            for local_gene_index, gene in enumerate(
                required_genes
            ):

                target_lookup[
                    gene_position[
                        gene
                    ]
                ] = local_gene_index

            section_specs: list[
                dict[str, Any]
            ] = []

            for archive_name in archives:

                metadata = section_metadata[
                    archive_name
                ]

                sample_value = metadata[
                    "locked_sample_value"
                ]

                region_value = metadata[
                    "locked_region_value"
                ]

                if sample_value not in sample_lookup:

                    raise SystemExit(
                        f"FAIL: sample {sample_value} "
                        f"missing in {file_name}."
                    )

                if region_value not in region_lookup:

                    raise SystemExit(
                        f"FAIL: region {region_value} "
                        f"missing in {file_name}."
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
                    }
                )

            section_count = len(
                section_specs
            )

            gene_count = len(
                required_genes
            )

            observed_cell_counts = np.zeros(
                section_count,
                dtype=np.int64,
            )

            expression_sums = np.zeros(
                (
                    section_count,
                    gene_count,
                ),
                dtype=np.float64,
            )

            expression_sumsq = np.zeros(
                (
                    section_count,
                    gene_count,
                ),
                dtype=np.float64,
            )

            detected_cell_counts = np.zeros(
                (
                    section_count,
                    gene_count,
                ),
                dtype=np.int64,
            )

            selected_CSR_values_read = 0
            target_CSR_values_aggregated = 0
            non_target_CSR_values_transient = 0

            data_dataset = x_object[
                "data"
            ]

            indices_dataset = x_object[
                "indices"
            ]

            indptr_dataset = x_object[
                "indptr"
            ]

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
                            f"FAIL: overlapping section "
                            f"assignments in {file_name}."
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

                observed_cell_counts += np.bincount(
                    row_assignment[
                        selected_rows
                    ],
                    minlength=section_count,
                ).astype(
                    np.int64
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

                if data_end <= data_start:

                    continue

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

                row_counts = np.diff(
                    indptr_chunk
                )

                local_rows = np.repeat(
                    np.arange(
                        end - start,
                        dtype=np.int64,
                    ),
                    row_counts,
                )

                entry_sections = row_assignment[
                    local_rows
                ]

                selected_entries = (
                    entry_sections
                    >= 0
                )

                selected_CSR_values_read += int(
                    np.count_nonzero(
                        selected_entries
                    )
                )

                if not np.any(
                    selected_entries
                ):

                    continue

                selected_indices = indices_chunk[
                    selected_entries
                ]

                selected_values = values_chunk[
                    selected_entries
                ]

                selected_sections_for_entry = (
                    entry_sections[
                        selected_entries
                    ]
                )

                selected_rows_for_entry = local_rows[
                    selected_entries
                ]

                target_gene_indices = target_lookup[
                    selected_indices
                ]

                target_entries = (
                    target_gene_indices
                    >= 0
                )

                target_CSR_values_aggregated += int(
                    np.count_nonzero(
                        target_entries
                    )
                )

                non_target_CSR_values_transient += int(
                    np.count_nonzero(
                        ~target_entries
                    )
                )

                if not np.any(
                    target_entries
                ):

                    continue

                target_sections = (
                    selected_sections_for_entry[
                        target_entries
                    ]
                )

                target_genes = target_gene_indices[
                    target_entries
                ]

                target_values = selected_values[
                    target_entries
                ]

                flat_indices = (
                    target_sections
                    * gene_count
                    + target_genes
                )

                expression_sums.reshape(
                    -1
                )[:] += np.bincount(
                    flat_indices,
                    weights=target_values,
                    minlength=(
                        section_count
                        * gene_count
                    ),
                )

                expression_sumsq.reshape(
                    -1
                )[:] += np.bincount(
                    flat_indices,
                    weights=(
                        target_values
                        * target_values
                    ),
                    minlength=(
                        section_count
                        * gene_count
                    ),
                )

                nonzero_target = (
                    np.abs(
                        target_values
                    )
                    > zero_tolerance
                )

                if np.any(
                    nonzero_target
                ):

                    row_gene_keys = (
                        selected_rows_for_entry[
                            target_entries
                        ][
                            nonzero_target
                        ]
                        * gene_count
                        + target_genes[
                            nonzero_target
                        ]
                    )

                    unique_row_gene_keys = np.unique(
                        row_gene_keys
                    )

                    unique_local_rows = (
                        unique_row_gene_keys
                        // gene_count
                    )

                    unique_genes = (
                        unique_row_gene_keys
                        % gene_count
                    )

                    unique_sections = row_assignment[
                        unique_local_rows
                    ]

                    detection_flat = (
                        unique_sections
                        * gene_count
                        + unique_genes
                    )

                    detected_cell_counts.reshape(
                        -1
                    )[:] += np.bincount(
                        detection_flat,
                        minlength=(
                            section_count
                            * gene_count
                        ),
                    ).astype(
                        np.int64
                    )

            if not np.array_equal(
                observed_cell_counts,
                expected_counts,
            ):

                details = "; ".join(
                    (
                        f"{archives[index]}:"
                        f"observed="
                        f"{observed_cell_counts[index]},"
                        f"expected="
                        f"{expected_counts[index]}"
                    )
                    for index in range(
                        len(archives)
                    )
                )

                raise SystemExit(
                    f"FAIL: selected-cell count "
                    f"mismatch in {file_name}: "
                    f"{details}"
                )

            accumulator_lookup[
                file_name
            ] = {
                "archive_names": archives,
                "genes": required_genes,
                "observed_cell_counts": (
                    observed_cell_counts
                ),
                "expression_sums": (
                    expression_sums
                ),
                "expression_sumsq": (
                    expression_sumsq
                ),
                "detected_cell_counts": (
                    detected_cell_counts
                ),
                "selected_CSR_values_read": (
                    selected_CSR_values_read
                ),
                "target_CSR_values_aggregated": (
                    target_CSR_values_aggregated
                ),
                "non_target_CSR_values_transient": (
                    non_target_CSR_values_transient
                ),
            }

            temporary_path = (
                checkpoint_path
                .with_suffix(
                    ".npz.tmp"
                )
            )

            with temporary_path.open(
                "wb",
            ) as handle_out:

                np.savez_compressed(
                    handle_out,
                    archive_names=np.asarray(
                        archives,
                        dtype="U",
                    ),
                    genes=np.asarray(
                        required_genes,
                        dtype="U",
                    ),
                    expected_cell_counts=(
                        expected_counts
                    ),
                    observed_cell_counts=(
                        observed_cell_counts
                    ),
                    expression_sums=(
                        expression_sums
                    ),
                    expression_sumsq=(
                        expression_sumsq
                    ),
                    detected_cell_counts=(
                        detected_cell_counts
                    ),
                    selected_CSR_values_read=np.asarray(
                        [
                            selected_CSR_values_read
                        ],
                        dtype=np.int64,
                    ),
                    target_CSR_values_aggregated=np.asarray(
                        [
                            target_CSR_values_aggregated
                        ],
                        dtype=np.int64,
                    ),
                    non_target_CSR_values_transient=np.asarray(
                        [
                            non_target_CSR_values_transient
                        ],
                        dtype=np.int64,
                    ),
                    h5ad_size_bytes=np.asarray(
                        [
                            h5ad_stat.st_size
                        ],
                        dtype=np.int64,
                    ),
                    h5ad_mtime_ns=np.asarray(
                        [
                            h5ad_stat.st_mtime_ns
                        ],
                        dtype=np.int64,
                    ),
                )

            os.replace(
                temporary_path,
                checkpoint_path,
            )

    accumulator = accumulator_lookup[
        file_name
    ]

    file_processing_rows.append(
        {
            "processed_H5AD": file_name,
            "selected_sections": len(
                accumulator[
                    "archive_names"
                ]
            ),
            "required_target_genes": len(
                accumulator[
                    "genes"
                ]
            ),
            "selected_cells_confirmed": int(
                accumulator[
                    "observed_cell_counts"
                ].sum()
            ),
            "checkpoint_reused": (
                checkpoint_reused
            ),
            "selected_CSR_values_read": (
                accumulator[
                    "selected_CSR_values_read"
                ]
            ),
            "target_CSR_values_aggregated": (
                accumulator[
                    "target_CSR_values_aggregated"
                ]
            ),
            "non_target_CSR_values_read_transiently": (
                accumulator[
                    "non_target_CSR_values_transient"
                ]
            ),
            "complete_X_matrix_materialized": (
                False
            ),
            "raw_X_values_accessed": (
                False
            ),
        }
    )

section_gene_rows: list[
    dict[str, Any]
] = []

moment_lookup: dict[
    tuple[str, str],
    dict[str, Any],
] = {}

cell_count_audit_rows: list[
    dict[str, Any]
] = []

for file_name in expected_files:

    accumulator = accumulator_lookup[
        file_name
    ]

    archive_position = {
        archive_name: index
        for index, archive_name
        in enumerate(
            accumulator[
                "archive_names"
            ]
        )
    }

    gene_position = {
        gene: index
        for index, gene
        in enumerate(
            accumulator[
                "genes"
            ]
        )
    }

    for archive_name in accumulator[
        "archive_names"
    ]:

        metadata = section_metadata[
            archive_name
        ]

        section_index = archive_position[
            archive_name
        ]

        observed_cells = int(
            accumulator[
                "observed_cell_counts"
            ][
                section_index
            ]
        )

        expected_cells = int(
            metadata[
                "selected_cell_count"
            ]
        )

        cell_count_audit_rows.append(
            {
                "archive_name": archive_name,
                "effective_analysis_set": (
                    metadata[
                        "effective_analysis_set"
                    ]
                ),
                "processed_H5AD": file_name,
                "expected_selected_cells": (
                    expected_cells
                ),
                "observed_selected_cells": (
                    observed_cells
                ),
                "cell_count_matches": (
                    observed_cells
                    == expected_cells
                ),
            }
        )

        for gene in sorted(
            required_genes_by_section[
                archive_name
            ]
        ):

            gene_index = gene_position[
                gene
            ]

            expression_sum = float(
                accumulator[
                    "expression_sums"
                ][
                    section_index,
                    gene_index,
                ]
            )

            expression_sumsq = float(
                accumulator[
                    "expression_sumsq"
                ][
                    section_index,
                    gene_index,
                ]
            )

            mean_expression = (
                expression_sum
                / observed_cells
            )

            variance = max(
                (
                    expression_sumsq
                    / observed_cells
                )
                - (
                    mean_expression
                    * mean_expression
                ),
                0.0,
            )

            population_sd = math.sqrt(
                variance
            )

            detected_cells = int(
                accumulator[
                    "detected_cell_counts"
                ][
                    section_index,
                    gene_index,
                ]
            )

            row = {
                "effective_analysis_set": (
                    metadata[
                        "effective_analysis_set"
                    ]
                ),
                "archive_name": archive_name,
                "donor_id": metadata[
                    "donor_id"
                ],
                "gestational_week": metadata[
                    "gestational_week"
                ],
                "panel_class": metadata[
                    "panel_class"
                ],
                "processed_H5AD": file_name,
                "selected_cells": (
                    observed_cells
                ),
                "gene_symbol": gene,
                "processed_X_sum": (
                    expression_sum
                ),
                "processed_X_sumsq": (
                    expression_sumsq
                ),
                "section_gene_mean": (
                    mean_expression
                ),
                "section_gene_population_SD": (
                    population_sd
                ),
                "detected_cell_count": (
                    detected_cells
                ),
                "detected_cell_fraction": (
                    detected_cells
                    / observed_cells
                ),
                "expression_source": "/X",
            }

            section_gene_rows.append(
                row
            )

            moment_lookup[
                (
                    archive_name,
                    gene,
                )
            ] = row

if len(section_gene_rows) != 488:

    raise SystemExit(
        f"FAIL: expected 488 section-gene "
        f"moment rows; observed "
        f"{len(section_gene_rows)}."
    )

if not all(
    row[
        "cell_count_matches"
    ]
    for row in cell_count_audit_rows
):

    raise SystemExit(
        "FAIL: at least one selected-cell count "
        "does not match the locked design."
    )

write_tsv(
    moment_dir
    / "phase10B5_P4E1_section_gene_moments.tsv",
    section_gene_rows,
    [
        "effective_analysis_set",
        "archive_name",
        "donor_id",
        "gestational_week",
        "panel_class",
        "processed_H5AD",
        "selected_cells",
        "gene_symbol",
        "processed_X_sum",
        "processed_X_sumsq",
        "section_gene_mean",
        "section_gene_population_SD",
        "detected_cell_count",
        "detected_cell_fraction",
        "expression_source",
    ],
)

write_tsv(
    processing_dir
    / "phase10B5_P4E1_selected_cell_count_audit.tsv",
    cell_count_audit_rows,
    [
        "archive_name",
        "effective_analysis_set",
        "processed_H5AD",
        "expected_selected_cells",
        "observed_selected_cells",
        "cell_count_matches",
    ],
)

write_tsv(
    processing_dir
    / "phase10B5_P4E1_file_processing_audit.tsv",
    file_processing_rows,
    [
        "processed_H5AD",
        "selected_sections",
        "required_target_genes",
        "selected_cells_confirmed",
        "checkpoint_reused",
        "selected_CSR_values_read",
        "target_CSR_values_aggregated",
        "non_target_CSR_values_read_transiently",
        "complete_X_matrix_materialized",
        "raw_X_values_accessed",
    ],
)

primary_sections = sorted(
    archive_name
    for archive_name, metadata
    in section_metadata.items()
    if metadata[
        "effective_analysis_set"
    ] == "primary"
)

sensitivity_sections = sorted(
    archive_name
    for archive_name, metadata
    in section_metadata.items()
    if metadata[
        "effective_analysis_set"
    ] == "section_sensitivity"
)

if len(primary_sections) != 8:

    raise SystemExit(
        "FAIL: expected eight primary sections."
    )

if len(sensitivity_sections) != 4:

    raise SystemExit(
        "FAIL: expected four sensitivity sections."
    )

core_genes = sorted(
    set().union(
        *(
            set(genes)
            for genes in core_modules.values()
        )
    )
)

if len(core_genes) != 36:

    raise SystemExit(
        f"FAIL: expected 36 harmonized core "
        f"genes; observed {len(core_genes)}."
    )

reference_rows: list[
    dict[str, Any]
] = []

reference_lookup: dict[
    str,
    dict[str, Any],
] = {}

for gene in core_genes:

    primary_values = np.asarray(
        [
            float(
                moment_lookup[
                    (
                        archive_name,
                        gene,
                    )
                ][
                    "section_gene_mean"
                ]
            )
            for archive_name in primary_sections
        ],
        dtype=np.float64,
    )

    center = float(
        primary_values.mean()
    )

    scale = float(
        primary_values.std(
            ddof=0
        )
    )

    zero_variance = (
        scale
        <= variance_tolerance
    )

    row = {
        "gene_symbol": gene,
        "primary_sections": 8,
        "primary_reference_center": center,
        "primary_reference_population_SD": (
            scale
        ),
        "zero_variance_gene": (
            zero_variance
        ),
        "standardization_reference": (
            "eight_primary_sections_equal_weight"
        ),
    }

    reference_rows.append(
        row
    )

    reference_lookup[
        gene
    ] = row

write_tsv(
    reference_dir
    / "phase10B5_P4E1_primary_gene_reference.tsv",
    reference_rows,
    [
        "gene_symbol",
        "primary_sections",
        "primary_reference_center",
        "primary_reference_population_SD",
        "zero_variance_gene",
        "standardization_reference",
    ],
)

standardized_gene_rows: list[
    dict[str, Any]
] = []

standardized_lookup: dict[
    tuple[str, str],
    float,
] = {}

for archive_name in sorted(
    section_metadata
):

    metadata = section_metadata[
        archive_name
    ]

    for gene in core_genes:

        raw_mean = float(
            moment_lookup[
                (
                    archive_name,
                    gene,
                )
            ][
                "section_gene_mean"
            ]
        )

        reference = reference_lookup[
            gene
        ]

        center = float(
            reference[
                "primary_reference_center"
            ]
        )

        scale = float(
            reference[
                "primary_reference_population_SD"
            ]
        )

        zero_variance = bool(
            reference[
                "zero_variance_gene"
            ]
        )

        standardized_value = (
            0.0
            if zero_variance
            else (
                raw_mean
                - center
            )
            / scale
        )

        standardized_lookup[
            (
                archive_name,
                gene,
            )
        ] = standardized_value

        standardized_gene_rows.append(
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
                "gestational_week": metadata[
                    "gestational_week"
                ],
                "panel_class": metadata[
                    "panel_class"
                ],
                "gene_symbol": gene,
                "section_gene_mean": (
                    raw_mean
                ),
                "primary_reference_center": (
                    center
                ),
                "primary_reference_population_SD": (
                    scale
                ),
                "standardized_section_gene_value": (
                    standardized_value
                ),
                "zero_variance_gene": (
                    zero_variance
                ),
                "sensitivity_projected_without_refit": (
                    metadata[
                        "effective_analysis_set"
                    ]
                    == "section_sensitivity"
                ),
            }
        )

if len(standardized_gene_rows) != 432:

    raise SystemExit(
        f"FAIL: expected 432 standardized "
        f"section-gene rows; observed "
        f"{len(standardized_gene_rows)}."
    )

write_tsv(
    reference_dir
    / "phase10B5_P4E1_standardized_core_section_gene_values.tsv",
    standardized_gene_rows,
    [
        "effective_analysis_set",
        "archive_name",
        "donor_id",
        "gestational_week",
        "panel_class",
        "gene_symbol",
        "section_gene_mean",
        "primary_reference_center",
        "primary_reference_population_SD",
        "standardized_section_gene_value",
        "zero_variance_gene",
        "sensitivity_projected_without_refit",
    ],
)

core_module_score_rows: list[
    dict[str, Any]
] = []

for row in core_plan_rows:

    archive_name = row[
        "archive_name"
    ]

    genes = [
        gene
        for gene in row[
            "gene_symbols"
        ].split(";")
        if gene
    ]

    values = [
        standardized_lookup[
            (
                archive_name,
                gene,
            )
        ]
        for gene in genes
    ]

    module_score = float(
        np.mean(
            values
        )
    )

    core_module_score_rows.append(
        {
            "effective_analysis_set": row[
                "effective_analysis_set"
            ],
            "estimand_role": row[
                "estimand_role"
            ],
            "archive_name": archive_name,
            "donor_id": row[
                "donor_id"
            ],
            "gestational_week": row[
                "gestational_week"
            ],
            "panel_class": row[
                "panel_class"
            ],
            "processed_H5AD": row[
                "processed_H5AD"
            ],
            "selected_cell_count": row[
                "selected_cell_count"
            ],
            "module": row[
                "module"
            ],
            "prespecified_claim_class": row[
                "prespecified_claim_class"
            ],
            "gene_count": len(
                genes
            ),
            "module_score": module_score,
            "temporal_primary_unit": (
                "independent_donor_section"
            ),
            "cell_count_used_as_replication": (
                False
            ),
            "sensitivity_projected_without_refit": (
                row[
                    "effective_analysis_set"
                ]
                == "section_sensitivity"
            ),
        }
    )

if len(core_module_score_rows) != 60:

    raise SystemExit(
        f"FAIL: expected 60 core module-score "
        f"rows; observed "
        f"{len(core_module_score_rows)}."
    )

write_tsv(
    module_dir
    / "phase10B5_P4E1_core_section_module_scores.tsv",
    core_module_score_rows,
    [
        "effective_analysis_set",
        "estimand_role",
        "archive_name",
        "donor_id",
        "gestational_week",
        "panel_class",
        "processed_H5AD",
        "selected_cell_count",
        "module",
        "prespecified_claim_class",
        "gene_count",
        "module_score",
        "temporal_primary_unit",
        "cell_count_used_as_replication",
        "sensitivity_projected_without_refit",
    ],
)

module_variance_rows: list[
    dict[str, Any]
] = []

nondegenerate_modules = 0

for module_name in sorted(
    core_modules
):

    primary_scores = np.asarray(
        [
            float(
                row[
                    "module_score"
                ]
            )
            for row in core_module_score_rows
            if (
                row[
                    "module"
                ] == module_name
                and row[
                    "effective_analysis_set"
                ] == "primary"
            )
        ],
        dtype=np.float64,
    )

    sensitivity_scores = np.asarray(
        [
            float(
                row[
                    "module_score"
                ]
            )
            for row in core_module_score_rows
            if (
                row[
                    "module"
                ] == module_name
                and row[
                    "effective_analysis_set"
                ]
                == "section_sensitivity"
            )
        ],
        dtype=np.float64,
    )

    primary_mean = float(
        primary_scores.mean()
    )

    primary_sd = float(
        primary_scores.std(
            ddof=0
        )
    )

    nondegenerate = (
        primary_sd
        > variance_tolerance
    )

    if nondegenerate:
        nondegenerate_modules += 1

    module_variance_rows.append(
        {
            "module": module_name,
            "primary_sections": int(
                primary_scores.size
            ),
            "sensitivity_sections": int(
                sensitivity_scores.size
            ),
            "primary_score_mean": (
                primary_mean
            ),
            "primary_score_population_SD": (
                primary_sd
            ),
            "primary_mean_approximately_zero": (
                abs(
                    primary_mean
                )
                <= 1e-10
            ),
            "nondegenerate_primary_variation": (
                nondegenerate
            ),
            "primary_score_minimum": float(
                primary_scores.min()
            ),
            "primary_score_maximum": float(
                primary_scores.max()
            ),
            "sensitivity_score_minimum": float(
                sensitivity_scores.min()
            ),
            "sensitivity_score_maximum": float(
                sensitivity_scores.max()
            ),
        }
    )

write_tsv(
    module_dir
    / "phase10B5_P4E1_core_module_variance_audit.tsv",
    module_variance_rows,
    [
        "module",
        "primary_sections",
        "sensitivity_sections",
        "primary_score_mean",
        "primary_score_population_SD",
        "primary_mean_approximately_zero",
        "nondegenerate_primary_variation",
        "primary_score_minimum",
        "primary_score_maximum",
        "sensitivity_score_minimum",
        "sensitivity_score_maximum",
    ],
)

expanded_gene_difference_rows: list[
    dict[str, Any]
] = []

expanded_module_summary_rows: list[
    dict[str, Any]
] = []

for module_name, genes in sorted(
    expanded_modules.items()
):

    module_plan = [
        row
        for row in expanded_plan_rows
        if row[
            "module"
        ] == module_name
    ]

    if len(module_plan) != 2:

        raise SystemExit(
            f"FAIL: expanded module {module_name} "
            "does not have two sections."
        )

    row_by_age = {
        int(
            row[
                "gestational_week"
            ]
        ): row
        for row in module_plan
    }

    if set(
        row_by_age
    ) != {
        18,
        20,
    }:

        raise SystemExit(
            f"FAIL: expanded module {module_name} "
            "does not contain GW18 and GW20."
        )

    gw18_archive = row_by_age[
        18
    ][
        "archive_name"
    ]

    gw20_archive = row_by_age[
        20
    ][
        "archive_name"
    ]

    differences: list[float] = []

    positive = 0
    negative = 0
    zero = 0

    for gene in genes:

        gw18_mean = float(
            moment_lookup[
                (
                    gw18_archive,
                    gene,
                )
            ][
                "section_gene_mean"
            ]
        )

        gw20_mean = float(
            moment_lookup[
                (
                    gw20_archive,
                    gene,
                )
            ][
                "section_gene_mean"
            ]
        )

        difference = (
            gw20_mean
            - gw18_mean
        )

        differences.append(
            difference
        )

        if difference > zero_tolerance:

            direction = "higher_at_GW20"
            positive += 1

        elif difference < -zero_tolerance:

            direction = "lower_at_GW20"
            negative += 1

        else:

            direction = "approximately_equal"
            zero += 1

        expanded_gene_difference_rows.append(
            {
                "module": module_name,
                "gene_symbol": gene,
                "GW18_archive": gw18_archive,
                "GW20_archive": gw20_archive,
                "GW18_section_gene_mean": (
                    gw18_mean
                ),
                "GW20_section_gene_mean": (
                    gw20_mean
                ),
                "GW20_minus_GW18_difference": (
                    difference
                ),
                "direction": direction,
                "formal_inference_authorized": (
                    False
                ),
            }
        )

    if positive > negative:

        dominant_direction = (
            "predominantly_higher_at_GW20"
        )

    elif negative > positive:

        dominant_direction = (
            "predominantly_lower_at_GW20"
        )

    else:

        dominant_direction = (
            "no_dominant_direction"
        )

    concordance_fraction = (
        max(
            positive,
            negative,
        )
        / len(
            differences
        )
    )

    expanded_module_summary_rows.append(
        {
            "module": module_name,
            "gene_count": len(
                differences
            ),
            "GW18_archive": gw18_archive,
            "GW20_archive": gw20_archive,
            "mean_gene_difference": float(
                np.mean(
                    differences
                )
            ),
            "median_gene_difference": float(
                median(
                    differences
                )
            ),
            "genes_higher_at_GW20": positive,
            "genes_lower_at_GW20": negative,
            "genes_approximately_equal": zero,
            "dominant_direction": (
                dominant_direction
            ),
            "direction_concordance_fraction": (
                concordance_fraction
            ),
            "formal_inference_authorized": (
                False
            ),
            "claim_scope": (
                "descriptive_two_section_"
                "expanded_panel_validation"
            ),
        }
    )

if len(expanded_gene_difference_rows) != 33:

    raise SystemExit(
        f"FAIL: expected 33 expanded gene "
        f"difference rows; observed "
        f"{len(expanded_gene_difference_rows)}."
    )

if len(expanded_module_summary_rows) != 4:

    raise SystemExit(
        f"FAIL: expected four expanded module "
        f"summaries; observed "
        f"{len(expanded_module_summary_rows)}."
    )

write_tsv(
    expanded_dir
    / "phase10B5_P4E1_expanded_panel_gene_differences.tsv",
    expanded_gene_difference_rows,
    [
        "module",
        "gene_symbol",
        "GW18_archive",
        "GW20_archive",
        "GW18_section_gene_mean",
        "GW20_section_gene_mean",
        "GW20_minus_GW18_difference",
        "direction",
        "formal_inference_authorized",
    ],
)

write_tsv(
    expanded_dir
    / "phase10B5_P4E1_expanded_panel_module_summary.tsv",
    expanded_module_summary_rows,
    [
        "module",
        "gene_count",
        "GW18_archive",
        "GW20_archive",
        "mean_gene_difference",
        "median_gene_difference",
        "genes_higher_at_GW20",
        "genes_lower_at_GW20",
        "genes_approximately_equal",
        "dominant_direction",
        "direction_concordance_fraction",
        "formal_inference_authorized",
        "claim_scope",
    ],
)

selected_cells_total = sum(
    int(
        row[
            "observed_selected_cells"
        ]
    )
    for row in cell_count_audit_rows
)

primary_selected_cells = sum(
    int(
        row[
            "observed_selected_cells"
        ]
    )
    for row in cell_count_audit_rows
    if row[
        "effective_analysis_set"
    ] == "primary"
)

sensitivity_selected_cells = sum(
    int(
        row[
            "observed_selected_cells"
        ]
    )
    for row in cell_count_audit_rows
    if row[
        "effective_analysis_set"
    ] == "section_sensitivity"
)

zero_variance_core_genes = sum(
    bool(
        row[
            "zero_variance_gene"
        ]
    )
    for row in reference_rows
)

all_finite = all(
    math.isfinite(
        float(
            row[
                "module_score"
            ]
        )
    )
    for row in core_module_score_rows
)

guard_rows = [
    {
        "processed_H5AD": row[
            "processed_H5AD"
        ],
        "expression_source": "/X",
        "chunked_CSR_payload_accessed": True,
        "selected_section_rows_aggregated": True,
        "non_target_values_read_transiently": (
            int(
                row[
                    "non_target_CSR_values_"
                    "read_transiently"
                ]
            )
            > 0
        ),
        "complete_X_matrix_materialized": False,
        "raw_X_values_accessed": False,
        "cell_level_module_scores_computed": False,
        "section_pseudobulk_computed": True,
        "section_level_module_scores_computed": True,
        "spatial_coordinate_values_accessed": False,
    }
    for row in file_processing_rows
]

write_tsv(
    guard_dir
    / "phase10B5_P4E1_expression_access_guard.tsv",
    guard_rows,
    [
        "processed_H5AD",
        "expression_source",
        "chunked_CSR_payload_accessed",
        "selected_section_rows_aggregated",
        "non_target_values_read_transiently",
        "complete_X_matrix_materialized",
        "raw_X_values_accessed",
        "cell_level_module_scores_computed",
        "section_pseudobulk_computed",
        "section_level_module_scores_computed",
        "spatial_coordinate_values_accessed",
    ],
)

technical_pass = (
    len(
        accumulator_lookup
    ) == 6
    and len(
        cell_count_audit_rows
    ) == 12
    and selected_cells_total
    == 6_062_942
    and primary_selected_cells
    == 4_306_468
    and sensitivity_selected_cells
    == 1_756_474
    and len(
        section_gene_rows
    ) == 488
    and len(
        reference_rows
    ) == 36
    and len(
        standardized_gene_rows
    ) == 432
    and len(
        core_module_score_rows
    ) == 60
    and len(
        expanded_gene_difference_rows
    ) == 33
    and len(
        expanded_module_summary_rows
    ) == 4
    and all_finite
)

if (
    technical_pass
    and nondegenerate_modules == 5
):

    status_value = (
        "passed_phase10B5_P4E1_chunked_processed_X_"
        "section_pseudobulk_and_non_degenerate_core_"
        "module_scoring_ready_for_temporal_"
        "association_analysis"
    )

elif technical_pass:

    status_value = (
        "completed_phase10B5_P4E1_section_pseudobulk_"
        "aggregation_requires_core_module_variance_"
        "review"
    )

else:

    status_value = (
        "phase10B5_P4E1_requires_manual_review"
    )

status = {
    "phase": "phase10B5_P4E1",
    "H5AD_files_processed": len(
        accumulator_lookup
    ),
    "file_accumulators_completed": len(
        accumulator_lookup
    ),
    "revised_sections_processed": len(
        cell_count_audit_rows
    ),
    "selected_cells_confirmed": (
        selected_cells_total
    ),
    "primary_selected_cells_confirmed": (
        primary_selected_cells
    ),
    "section_sensitivity_cells_confirmed": (
        sensitivity_selected_cells
    ),
    "section_gene_moment_rows": len(
        section_gene_rows
    ),
    "core_primary_gene_reference_rows": len(
        reference_rows
    ),
    "zero_variance_core_genes": (
        zero_variance_core_genes
    ),
    "core_standardized_section_gene_rows": len(
        standardized_gene_rows
    ),
    "core_section_module_score_rows": len(
        core_module_score_rows
    ),
    "primary_core_module_score_rows": sum(
        row[
            "effective_analysis_set"
        ] == "primary"
        for row in core_module_score_rows
    ),
    "sensitivity_core_module_score_rows": sum(
        row[
            "effective_analysis_set"
        ] == "section_sensitivity"
        for row in core_module_score_rows
    ),
    "nondegenerate_primary_core_modules": (
        nondegenerate_modules
    ),
    "expanded_gene_difference_rows": len(
        expanded_gene_difference_rows
    ),
    "expanded_module_summary_rows": len(
        expanded_module_summary_rows
    ),
    "expression_source": "/X",
    "chunked_expression_values_accessed": True,
    "complete_X_matrix_materialized": False,
    "raw_X_values_accessed": False,
    "cell_level_module_scores_computed": False,
    "section_pseudobulk_computed": True,
    "section_level_module_scores_computed": True,
    "spatial_coordinate_values_accessed": False,
    "primary_statistical_unit": (
        "independent_donor_section"
    ),
    "cell_count_used_as_replication": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P4E1_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B5_P4E1_status.tsv",
    [status],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P4E1 CHUNKED SECTION "
    "PSEUDOBULK AGGREGATION =====",
    "",
    (
        "H5AD files processed: "
        f"{len(accumulator_lookup)}/6"
    ),
    (
        "Revised sections processed: "
        f"{len(cell_count_audit_rows)}/12"
    ),
    (
        "Selected cells confirmed: "
        f"{selected_cells_total}"
    ),
    (
        "Section-gene moment rows: "
        f"{len(section_gene_rows)}"
    ),
    (
        "Core primary-reference genes: "
        f"{len(reference_rows)}"
    ),
    (
        "Zero-variance core genes: "
        f"{zero_variance_core_genes}"
    ),
    (
        "Core section-module scores: "
        f"{len(core_module_score_rows)}"
    ),
    (
        "Nondegenerate primary core modules: "
        f"{nondegenerate_modules}/5"
    ),
    (
        "Expanded-panel gene differences: "
        f"{len(expanded_gene_difference_rows)}"
    ),
    (
        "Expanded-panel module summaries: "
        f"{len(expanded_module_summary_rows)}"
    ),
    "",
    "Expression source: /X",
    "Chunked expression values accessed: TRUE",
    "Complete X matrix materialized: FALSE",
    "Raw X values accessed: FALSE",
    "Cell-level module scores computed: FALSE",
    "Section pseudobulk computed: TRUE",
    "Section-level module scores computed: TRUE",
    "Spatial-coordinate values accessed: FALSE",
    (
        "Primary statistical unit: "
        "independent donor/section"
    ),
    "Cell count used as replication: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P4E1 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P4E1_report.txt"
).write_text(
    "\n".join(report)
    + "\n",
    encoding="utf-8",
)

print()
print(
    "\n".join(report)
)

print(
    "\n===== CORE MODULE VARIANCE AUDIT ====="
)

for row in module_variance_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "module",
                "primary_score_mean",
                "primary_score_population_SD",
                "nondegenerate_primary_variation",
                "primary_score_minimum",
                "primary_score_maximum",
            )
        )
    )

print(
    "\n===== EXPANDED-PANEL MODULE SUMMARY ====="
)

for row in expanded_module_summary_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "module",
                "mean_gene_difference",
                "median_gene_difference",
                "dominant_direction",
                "direction_concordance_fraction",
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
        != "phase10B5_P4E1_SHA256.tsv"
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
    / "phase10B5_P4E1_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)

if not technical_pass:

    raise SystemExit(
        "Phase 10B5-P4E1 requires manual review."
    )
