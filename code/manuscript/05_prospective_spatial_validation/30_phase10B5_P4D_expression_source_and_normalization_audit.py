from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


project = Path(sys.argv[1])
payload_dir = Path(sys.argv[2])
design_path = Path(sys.argv[3])
out = Path(sys.argv[4])

structure_dir = out / "01_source_structure"
sampling_dir = out / "02_deterministic_sampling"
diagnostic_dir = out / "03_numerical_diagnostics"
comparison_dir = out / "04_source_comparisons"
decision_dir = out / "05_source_decision"
guard_dir = out / "06_expression_access_guard"
audit_dir = out / "07_audit"

for directory in (
    structure_dir,
    sampling_dir,
    diagnostic_dir,
    comparison_dir,
    decision_dir,
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

rows_per_section = 64
metadata_chunk_size = 500_000
maximum_columns_for_diagnostics = 1000
maximum_CSC_columns_for_diagnostics = 128
integer_tolerance = 1e-7
zero_tolerance = 1e-12


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


def decode(
    value: Any,
) -> Any:

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

        return value.item()

    if isinstance(value, np.ndarray):

        return [
            decode(item)
            for item in value.tolist()
        ]

    return value


def text(
    value: Any,
) -> str:

    decoded = decode(value)

    if isinstance(decoded, list):

        return ";".join(
            str(item)
            for item in decoded
        )

    return str(decoded)


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
        str(decode(value))
        for value in group["categories"][:]
    ]

    codes = group["codes"]

    return categories, codes


def matrix_encoding(
    obj: h5py.Dataset | h5py.Group,
) -> str:

    if isinstance(obj, h5py.Dataset):

        return text(
            obj.attrs.get(
                "encoding-type",
                "dense_array",
            )
        )

    return text(
        obj.attrs.get(
            "encoding-type",
            "",
        )
    )


def matrix_shape(
    obj: h5py.Dataset | h5py.Group,
) -> tuple[int, int]:

    if isinstance(obj, h5py.Dataset):

        if obj.ndim != 2:

            raise ValueError(
                f"Dense matrix is not two-dimensional: "
                f"{obj.name}"
            )

        return int(obj.shape[0]), int(obj.shape[1])

    shape_value = obj.attrs.get(
        "shape",
        None,
    )

    if shape_value is None:

        raise ValueError(
            f"Sparse matrix lacks shape attribute: "
            f"{obj.name}"
        )

    values = np.asarray(
        shape_value
    ).reshape(-1)

    if values.size != 2:

        raise ValueError(
            f"Unexpected matrix shape attribute: "
            f"{obj.name}"
        )

    return int(values[0]), int(values[1])


def matrix_dtype(
    obj: h5py.Dataset | h5py.Group,
) -> str:

    if isinstance(obj, h5py.Dataset):

        return str(obj.dtype)

    if "data" in obj:

        return str(obj["data"].dtype)

    return ""


def matrix_compression(
    obj: h5py.Dataset | h5py.Group,
) -> str:

    if isinstance(obj, h5py.Dataset):

        return str(obj.compression or "")

    if "data" in obj:

        return str(
            obj["data"].compression or ""
        )

    return ""


def object_attributes(
    obj: h5py.Dataset | h5py.Group,
) -> str:

    result = {
        str(key): decode(value)
        for key, value in obj.attrs.items()
    }

    return json.dumps(
        result,
        sort_keys=True,
        ensure_ascii=False,
    )


def find_expression_sources(
    handle: h5py.File,
) -> list[tuple[str, h5py.Dataset | h5py.Group]]:

    sources: list[
        tuple[
            str,
            h5py.Dataset | h5py.Group,
        ]
    ] = []

    if "X" in handle:

        sources.append(
            (
                "/X",
                handle["X"],
            )
        )

    if (
        "layers" in handle
        and isinstance(
            handle["layers"],
            h5py.Group,
        )
    ):

        for key in sorted(
            handle["layers"].keys()
        ):

            sources.append(
                (
                    f"/layers/{key}",
                    handle["layers"][key],
                )
            )

    if (
        "raw" in handle
        and isinstance(
            handle["raw"],
            h5py.Group,
        )
        and "X" in handle["raw"]
    ):

        sources.append(
            (
                "/raw/X",
                handle["raw"]["X"],
            )
        )

    return sources


def read_matrix_rows(
    obj: h5py.Dataset | h5py.Group,
    requested_rows: list[int],
) -> tuple[np.ndarray, int, bool]:

    row_indices = np.asarray(
        sorted(set(requested_rows)),
        dtype=np.int64,
    )

    n_rows, n_columns = matrix_shape(obj)

    if row_indices.size == 0:

        raise ValueError(
            f"No diagnostic rows supplied for {obj.name}"
        )

    if row_indices[-1] >= n_rows:

        raise IndexError(
            f"Diagnostic row exceeds matrix shape: "
            f"{obj.name}"
        )

    encoding = matrix_encoding(obj).lower()

    columns_truncated = False

    if (
        "csc" in encoding
        and n_columns
        > maximum_CSC_columns_for_diagnostics
    ):

        sampled_columns = (
            maximum_CSC_columns_for_diagnostics
        )

        columns_truncated = True

    else:

        sampled_columns = min(
            n_columns,
            maximum_columns_for_diagnostics,
        )

        columns_truncated = (
            sampled_columns < n_columns
        )

    if isinstance(obj, h5py.Dataset):

        values = np.asarray(
            obj[
                row_indices,
                :sampled_columns,
            ],
            dtype=np.float64,
        )

        return (
            values,
            sampled_columns,
            columns_truncated,
        )

    if (
        "data" not in obj
        or "indices" not in obj
        or "indptr" not in obj
    ):

        raise ValueError(
            f"Unsupported matrix group structure: "
            f"{obj.name}"
        )

    data = obj["data"]
    indices = obj["indices"]
    indptr = obj["indptr"]

    output = np.zeros(
        (
            row_indices.size,
            sampled_columns,
        ),
        dtype=np.float64,
    )

    if "csr" in encoding:

        for output_row, matrix_row in enumerate(
            row_indices
        ):

            start = int(
                indptr[
                    matrix_row
                ]
            )

            end = int(
                indptr[
                    matrix_row + 1
                ]
            )

            if end <= start:

                continue

            row_columns = np.asarray(
                indices[start:end],
                dtype=np.int64,
            )

            row_values = np.asarray(
                data[start:end],
                dtype=np.float64,
            )

            retained = (
                row_columns
                < sampled_columns
            )

            output[
                output_row,
                row_columns[retained],
            ] = row_values[retained]

    elif "csc" in encoding:

        row_position = {
            int(matrix_row): output_row
            for output_row, matrix_row
            in enumerate(row_indices)
        }

        for column in range(
            sampled_columns
        ):

            start = int(
                indptr[column]
            )

            end = int(
                indptr[column + 1]
            )

            if end <= start:

                continue

            column_rows = np.asarray(
                indices[start:end],
                dtype=np.int64,
            )

            column_values = np.asarray(
                data[start:end],
                dtype=np.float64,
            )

            for matrix_row, value in zip(
                column_rows,
                column_values,
            ):

                output_row = row_position.get(
                    int(matrix_row)
                )

                if output_row is not None:

                    output[
                        output_row,
                        column,
                    ] = float(value)

    else:

        raise ValueError(
            f"Unsupported sparse encoding "
            f"{encoding}: {obj.name}"
        )

    return (
        output,
        sampled_columns,
        columns_truncated,
    )


def classify_values(
    array: np.ndarray,
) -> dict[str, Any]:

    flattened = np.asarray(
        array,
        dtype=np.float64,
    ).reshape(-1)

    finite = flattened[
        np.isfinite(flattened)
    ]

    if finite.size == 0:

        return {
            "finite_values": 0,
            "nonzero_values": 0,
            "zero_fraction": "",
            "nonnegative_fraction": "",
            "integer_like_fraction_nonzero": "",
            "minimum": "",
            "maximum": "",
            "mean": "",
            "median": "",
            "p95": "",
            "p99": "",
            "median_row_sum": "",
            "classification": (
                "no_finite_values"
            ),
        }

    nonzero = finite[
        np.abs(finite)
        > zero_tolerance
    ]

    zero_fraction = float(
        1.0
        - (
            nonzero.size
            / finite.size
        )
    )

    nonnegative_fraction = float(
        np.mean(
            finite
            >= -integer_tolerance
        )
    )

    if nonzero.size:

        integer_like_fraction = float(
            np.mean(
                np.abs(
                    nonzero
                    - np.rint(nonzero)
                )
                <= integer_tolerance
            )
        )

        median_value = float(
            np.median(nonzero)
        )

        p95 = float(
            np.quantile(
                nonzero,
                0.95,
            )
        )

        p99 = float(
            np.quantile(
                nonzero,
                0.99,
            )
        )

    else:

        integer_like_fraction = 1.0
        median_value = 0.0
        p95 = 0.0
        p99 = 0.0

    if nonzero.size == 0:

        classification = (
            "all_zero_in_diagnostic_subset"
        )

    elif (
        nonnegative_fraction >= 0.999999
        and integer_like_fraction >= 0.999
    ):

        classification = (
            "integer_count_like"
        )

    elif nonnegative_fraction >= 0.999999:

        classification = (
            "continuous_nonnegative"
        )

    else:

        classification = (
            "continuous_signed"
        )

    row_sums = np.nansum(
        array,
        axis=1,
    )

    return {
        "finite_values": int(
            finite.size
        ),
        "nonzero_values": int(
            nonzero.size
        ),
        "zero_fraction": (
            zero_fraction
        ),
        "nonnegative_fraction": (
            nonnegative_fraction
        ),
        "integer_like_fraction_nonzero": (
            integer_like_fraction
        ),
        "minimum": float(
            finite.min()
        ),
        "maximum": float(
            finite.max()
        ),
        "mean": float(
            finite.mean()
        ),
        "median": median_value,
        "p95": p95,
        "p99": p99,
        "median_row_sum": float(
            np.median(row_sums)
        ),
        "classification": classification,
    }


design_rows, design_columns = read_tsv(
    design_path
)

if len(design_rows) != 12:

    raise SystemExit(
        f"FAIL: expected 12 revised sections, "
        f"observed {len(design_rows)}."
    )

design_by_file: dict[
    str,
    list[dict[str, str]],
] = defaultdict(list)

for row in design_rows:

    design_by_file[
        row[
            "processed_H5AD"
        ]
    ].append(row)

if set(design_by_file) != set(
    expected_files
):

    raise SystemExit(
        "FAIL: revised design does not cover the "
        "locked six H5AD files."
    )

structure_rows: list[dict[str, Any]] = []
sampled_row_rows: list[dict[str, Any]] = []
section_diagnostic_rows: list[dict[str, Any]] = []
file_diagnostic_rows: list[dict[str, Any]] = []
comparison_rows: list[dict[str, Any]] = []
guard_rows: list[dict[str, Any]] = []

file_metric_lookup: dict[
    tuple[str, str],
    dict[str, Any],
] = {}

files_opened = 0
sections_sampled = 0
sources_audited = 0
diagnostic_expression_values = 0

for file_index, file_name in enumerate(
    expected_files,
    start=1,
):

    path = payload_dir / file_name

    if not path.is_file():

        raise SystemExit(
            f"FAIL: missing H5AD payload: {path}"
        )

    print(
        f"[{file_index:02d}/06] "
        f"Auditing expression sources: "
        f"{file_name}",
        flush=True,
    )

    with h5py.File(
        path,
        "r",
    ) as handle:

        files_opened += 1

        obs = handle["obs"]

        sample_categories, sample_codes = categorical(
            obs,
            "sample",
        )

        region_categories, region_codes = categorical(
            obs,
            "region",
        )

        sample_lookup = {
            value: index
            for index, value
            in enumerate(sample_categories)
        }

        region_lookup = {
            value: index
            for index, value
            in enumerate(region_categories)
        }

        target_rows = design_by_file[
            file_name
        ]

        sampling_specs: dict[
            str,
            dict[str, Any],
        ] = {}

        for target in target_rows:

            archive_name = target[
                "archive_name"
            ]

            expected_sample = target[
                "locked_sample_value"
            ]

            expected_region = target[
                "locked_region_value"
            ]

            if expected_sample not in sample_lookup:

                raise SystemExit(
                    f"FAIL: missing sample category "
                    f"{expected_sample} in {file_name}."
                )

            if expected_region not in region_lookup:

                raise SystemExit(
                    f"FAIL: missing region category "
                    f"{expected_region} in {file_name}."
                )

            sampling_specs[
                archive_name
            ] = {
                "sample_code": sample_lookup[
                    expected_sample
                ],
                "region_code": region_lookup[
                    expected_region
                ],
                "row_indices": [],
            }

        n_obs = int(
            sample_codes.shape[0]
        )

        for start in range(
            0,
            n_obs,
            metadata_chunk_size,
        ):

            if all(
                len(spec["row_indices"])
                >= rows_per_section
                for spec in sampling_specs.values()
            ):

                break

            end = min(
                start + metadata_chunk_size,
                n_obs,
            )

            sample_chunk = sample_codes[
                start:end
            ]

            region_chunk = region_codes[
                start:end
            ]

            for archive_name, spec in (
                sampling_specs.items()
            ):

                remaining = (
                    rows_per_section
                    - len(
                        spec[
                            "row_indices"
                        ]
                    )
                )

                if remaining <= 0:

                    continue

                matched = np.flatnonzero(
                    (
                        sample_chunk
                        == spec[
                            "sample_code"
                        ]
                    )
                    & (
                        region_chunk
                        == spec[
                            "region_code"
                        ]
                    )
                )

                if matched.size:

                    selected = matched[
                        :remaining
                    ] + start

                    spec[
                        "row_indices"
                    ].extend(
                        int(value)
                        for value in selected
                    )

        for target in target_rows:

            archive_name = target[
                "archive_name"
            ]

            rows_found = sampling_specs[
                archive_name
            ][
                "row_indices"
            ]

            if len(rows_found) != rows_per_section:

                raise SystemExit(
                    f"FAIL: selected only "
                    f"{len(rows_found)} diagnostic rows "
                    f"for {archive_name}; expected "
                    f"{rows_per_section}."
                )

            sections_sampled += 1

            for sampling_order, row_index in enumerate(
                rows_found,
                start=1,
            ):

                sampled_row_rows.append(
                    {
                        "archive_name": archive_name,
                        "effective_analysis_set": target[
                            "effective_analysis_set"
                        ],
                        "processed_H5AD": file_name,
                        "sample_value": target[
                            "locked_sample_value"
                        ],
                        "region_value": target[
                            "locked_region_value"
                        ],
                        "sampling_order": sampling_order,
                        "obs_row_index": row_index,
                        "sampling_purpose": (
                            "expression_source_"
                            "classification_only"
                        ),
                    }
                )

        sources = find_expression_sources(
            handle
        )

        if not sources:

            raise SystemExit(
                f"FAIL: no expression sources found "
                f"in {file_name}."
            )

        arrays_by_source: dict[
            str,
            list[np.ndarray],
        ] = defaultdict(list)

        for source_path, source_object in sources:

            source_rows, source_columns = (
                matrix_shape(
                    source_object
                )
            )

            if source_rows != n_obs:

                raise SystemExit(
                    f"FAIL: expression source "
                    f"{source_path} in {file_name} "
                    f"contains {source_rows} rows; "
                    f"obs contains {n_obs}."
                )

            structure_rows.append(
                {
                    "file_name": file_name,
                    "source_path": source_path,
                    "encoding_type": matrix_encoding(
                        source_object
                    ),
                    "matrix_rows": source_rows,
                    "matrix_columns": source_columns,
                    "dtype": matrix_dtype(
                        source_object
                    ),
                    "compression": matrix_compression(
                        source_object
                    ),
                    "attributes": object_attributes(
                        source_object
                    ),
                    "source_name_suggests_counts": (
                        "count"
                        in source_path.lower()
                    ),
                    "source_name_suggests_normalized": any(
                        token in source_path.lower()
                        for token in (
                            "norm",
                            "log",
                            "scale",
                        )
                    ),
                }
            )

            sources_audited += 1

            for target in target_rows:

                archive_name = target[
                    "archive_name"
                ]

                diagnostic_rows = sampling_specs[
                    archive_name
                ][
                    "row_indices"
                ]

                values, sampled_columns, truncated = (
                    read_matrix_rows(
                        source_object,
                        diagnostic_rows,
                    )
                )

                diagnostic_expression_values += int(
                    values.size
                )

                arrays_by_source[
                    source_path
                ].append(
                    values
                )

                metrics = classify_values(
                    values
                )

                section_diagnostic_rows.append(
                    {
                        "archive_name": archive_name,
                        "effective_analysis_set": target[
                            "effective_analysis_set"
                        ],
                        "processed_H5AD": file_name,
                        "source_path": source_path,
                        "rows_sampled": (
                            values.shape[0]
                        ),
                        "columns_sampled": (
                            sampled_columns
                        ),
                        "matrix_columns": (
                            source_columns
                        ),
                        "columns_truncated": (
                            truncated
                        ),
                        **metrics,
                        "diagnostic_only": True,
                        "module_scores_computed": False,
                    }
                )

        file_arrays: dict[
            str,
            np.ndarray,
        ] = {}

        for source_path, arrays in (
            arrays_by_source.items()
        ):

            minimum_columns = min(
                array.shape[1]
                for array in arrays
            )

            combined = np.concatenate(
                [
                    array[
                        :,
                        :minimum_columns,
                    ]
                    for array in arrays
                ],
                axis=0,
            )

            file_arrays[
                source_path
            ] = combined

            metrics = classify_values(
                combined
            )

            row = {
                "processed_H5AD": file_name,
                "source_path": source_path,
                "sections_represented": len(
                    arrays
                ),
                "rows_sampled": (
                    combined.shape[0]
                ),
                "columns_sampled": (
                    combined.shape[1]
                ),
                **metrics,
                "diagnostic_only": True,
                "module_scores_computed": False,
            }

            file_diagnostic_rows.append(
                row
            )

            file_metric_lookup[
                (
                    file_name,
                    source_path,
                )
            ] = row

        if "/X" in file_arrays:

            reference = file_arrays[
                "/X"
            ]

            for source_path, candidate in (
                file_arrays.items()
            ):

                if source_path == "/X":

                    continue

                shared_rows = min(
                    reference.shape[0],
                    candidate.shape[0],
                )

                shared_columns = min(
                    reference.shape[1],
                    candidate.shape[1],
                )

                first = reference[
                    :shared_rows,
                    :shared_columns,
                ].reshape(-1)

                second = candidate[
                    :shared_rows,
                    :shared_columns,
                ].reshape(-1)

                finite = (
                    np.isfinite(first)
                    & np.isfinite(second)
                )

                first = first[finite]
                second = second[finite]

                if first.size:

                    differences = np.abs(
                        first - second
                    )

                    equal_fraction = float(
                        np.mean(
                            differences
                            <= integer_tolerance
                        )
                    )

                    maximum_difference = float(
                        differences.max()
                    )

                else:

                    equal_fraction = ""
                    maximum_difference = ""

                if (
                    first.size >= 2
                    and np.std(first) > 0
                    and np.std(second) > 0
                ):

                    correlation = float(
                        np.corrcoef(
                            first,
                            second,
                        )[0, 1]
                    )

                else:

                    correlation = ""

                comparison_rows.append(
                    {
                        "processed_H5AD": file_name,
                        "reference_source": "/X",
                        "comparison_source": source_path,
                        "shared_rows": shared_rows,
                        "shared_columns": (
                            shared_columns
                        ),
                        "finite_values_compared": int(
                            first.size
                        ),
                        "equal_fraction_tolerance_1e_7": (
                            equal_fraction
                        ),
                        "maximum_absolute_difference": (
                            maximum_difference
                        ),
                        "pearson_correlation": (
                            correlation
                        ),
                        "diagnostic_only": True,
                    }
                )

        guard_rows.append(
            {
                "processed_H5AD": file_name,
                "selected_sections": len(
                    target_rows
                ),
                "rows_sampled_per_section": (
                    rows_per_section
                ),
                "expression_sources_audited": len(
                    sources
                ),
                "maximum_columns_sampled": (
                    maximum_columns_for_diagnostics
                ),
                "targeted_expression_values_accessed": (
                    True
                ),
                "complete_expression_matrix_loaded": (
                    False
                ),
                "all_cells_expression_accessed": (
                    False
                ),
                "all_genes_expression_accessed": (
                    False
                ),
                "module_scores_computed": (
                    False
                ),
                "spatial_values_accessed": (
                    False
                ),
            }
        )

write_tsv(
    structure_dir
    / "phase10B5_P4D_expression_source_structure.tsv",
    structure_rows,
    [
        "file_name",
        "source_path",
        "encoding_type",
        "matrix_rows",
        "matrix_columns",
        "dtype",
        "compression",
        "attributes",
        "source_name_suggests_counts",
        "source_name_suggests_normalized",
    ],
)

write_tsv_gz(
    sampling_dir
    / "phase10B5_P4D_deterministic_diagnostic_rows.tsv.gz",
    sampled_row_rows,
    [
        "archive_name",
        "effective_analysis_set",
        "processed_H5AD",
        "sample_value",
        "region_value",
        "sampling_order",
        "obs_row_index",
        "sampling_purpose",
    ],
)

diagnostic_columns = [
    "archive_name",
    "effective_analysis_set",
    "processed_H5AD",
    "source_path",
    "rows_sampled",
    "columns_sampled",
    "matrix_columns",
    "columns_truncated",
    "finite_values",
    "nonzero_values",
    "zero_fraction",
    "nonnegative_fraction",
    "integer_like_fraction_nonzero",
    "minimum",
    "maximum",
    "mean",
    "median",
    "p95",
    "p99",
    "median_row_sum",
    "classification",
    "diagnostic_only",
    "module_scores_computed",
]

write_tsv(
    diagnostic_dir
    / "phase10B5_P4D_section_source_diagnostics.tsv",
    section_diagnostic_rows,
    diagnostic_columns,
)

write_tsv(
    diagnostic_dir
    / "phase10B5_P4D_file_source_diagnostics.tsv",
    file_diagnostic_rows,
    [
        "processed_H5AD",
        "source_path",
        "sections_represented",
        "rows_sampled",
        "columns_sampled",
        "finite_values",
        "nonzero_values",
        "zero_fraction",
        "nonnegative_fraction",
        "integer_like_fraction_nonzero",
        "minimum",
        "maximum",
        "mean",
        "median",
        "p95",
        "p99",
        "median_row_sum",
        "classification",
        "diagnostic_only",
        "module_scores_computed",
    ],
)

write_tsv(
    comparison_dir
    / "phase10B5_P4D_X_source_comparisons.tsv",
    comparison_rows,
    [
        "processed_H5AD",
        "reference_source",
        "comparison_source",
        "shared_rows",
        "shared_columns",
        "finite_values_compared",
        "equal_fraction_tolerance_1e_7",
        "maximum_absolute_difference",
        "pearson_correlation",
        "diagnostic_only",
    ],
)

source_summary_rows: list[
    dict[str, Any]
] = []

all_source_paths = sorted(
    {
        row[
            "source_path"
        ]
        for row in file_diagnostic_rows
    }
)

for source_path in all_source_paths:

    rows = [
        row
        for row in file_diagnostic_rows
        if row[
            "source_path"
        ] == source_path
    ]

    classifications = Counter(
        row[
            "classification"
        ]
        for row in rows
    )

    source_summary_rows.append(
        {
            "source_path": source_path,
            "files_present": len(rows),
            "all_six_files_present": (
                len(rows) == 6
            ),
            "integer_count_like_files": (
                classifications[
                    "integer_count_like"
                ]
            ),
            "continuous_nonnegative_files": (
                classifications[
                    "continuous_nonnegative"
                ]
            ),
            "continuous_signed_files": (
                classifications[
                    "continuous_signed"
                ]
            ),
            "all_zero_files": (
                classifications[
                    "all_zero_in_diagnostic_subset"
                ]
            ),
            "classification_profile": ";".join(
                f"{key}:{value}"
                for key, value in sorted(
                    classifications.items()
                )
            ),
        }
    )

write_tsv(
    decision_dir
    / "phase10B5_P4D_cross_file_source_summary.tsv",
    source_summary_rows,
    [
        "source_path",
        "files_present",
        "all_six_files_present",
        "integer_count_like_files",
        "continuous_nonnegative_files",
        "continuous_signed_files",
        "all_zero_files",
        "classification_profile",
    ],
)

count_candidates = [
    row[
        "source_path"
    ]
    for row in source_summary_rows
    if (
        row[
            "all_six_files_present"
        ]
        and row[
            "integer_count_like_files"
        ] == 6
    )
]

continuous_candidates = [
    row[
        "source_path"
    ]
    for row in source_summary_rows
    if (
        row[
            "all_six_files_present"
        ]
        and (
            row[
                "continuous_nonnegative_files"
            ]
            + row[
                "continuous_signed_files"
            ]
        ) == 6
    )
]


def count_rank(
    source_path: str,
) -> tuple[int, str]:

    lowered = source_path.lower()

    if (
        source_path.startswith(
            "/layers/"
        )
        and "count" in lowered
    ):
        return 0, source_path

    if source_path == "/raw/X":
        return 1, source_path

    if source_path == "/X":
        return 2, source_path

    if source_path.startswith(
        "/layers/"
    ):
        return 3, source_path

    return 4, source_path


def continuous_rank(
    source_path: str,
) -> tuple[int, str]:

    lowered = source_path.lower()

    if source_path == "/X":
        return 0, source_path

    if (
        source_path.startswith(
            "/layers/"
        )
        and any(
            token in lowered
            for token in (
                "norm",
                "log",
                "scale",
            )
        )
    ):
        return 1, source_path

    if source_path.startswith(
        "/layers/"
    ):
        return 2, source_path

    if source_path == "/raw/X":
        return 3, source_path

    return 4, source_path


count_candidates.sort(
    key=count_rank
)

continuous_candidates.sort(
    key=continuous_rank
)

recommended_count_source = (
    count_candidates[0]
    if count_candidates
    else ""
)

recommended_processed_source = (
    continuous_candidates[0]
    if continuous_candidates
    else ""
)

if recommended_processed_source:

    recommended_scoring_strategy = (
        "use_consistent_continuous_processed_source_"
        "after_prespecified_gene_wise_standardization_"
        "within_each_section"
    )

elif recommended_count_source:

    recommended_scoring_strategy = (
        "derive_library_size_normalized_log1p_values_"
        "from_consistent_integer_count_source_then_"
        "perform_gene_wise_standardization_within_"
        "each_section"
    )

else:

    recommended_scoring_strategy = (
        "manual_source_review_required_before_"
        "normalization_or_scoring"
    )

automatic_source_recommendation = bool(
    recommended_processed_source
    or recommended_count_source
)

decision = {
    "files_audited": files_opened,
    "revised_sections_sampled": (
        sections_sampled
    ),
    "diagnostic_rows_per_section": (
        rows_per_section
    ),
    "expression_sources_audited": (
        sources_audited
    ),
    "diagnostic_expression_values_accessed": (
        diagnostic_expression_values
    ),
    "consistent_integer_count_candidates": ";".join(
        count_candidates
    ),
    "consistent_continuous_candidates": ";".join(
        continuous_candidates
    ),
    "recommended_count_source": (
        recommended_count_source
    ),
    "recommended_processed_source": (
        recommended_processed_source
    ),
    "recommended_scoring_strategy": (
        recommended_scoring_strategy
    ),
    "automatic_source_recommendation_available": (
        automatic_source_recommendation
    ),
    "source_lock_finalized": False,
    "module_scoring_authorized": False,
    "module_scores_computed": False,
}

write_tsv(
    decision_dir
    / "phase10B5_P4D_expression_source_decision.tsv",
    [decision],
    list(
        decision.keys()
    ),
)

write_tsv(
    guard_dir
    / "phase10B5_P4D_expression_access_guard.tsv",
    guard_rows,
    [
        "processed_H5AD",
        "selected_sections",
        "rows_sampled_per_section",
        "expression_sources_audited",
        "maximum_columns_sampled",
        "targeted_expression_values_accessed",
        "complete_expression_matrix_loaded",
        "all_cells_expression_accessed",
        "all_genes_expression_accessed",
        "module_scores_computed",
        "spatial_values_accessed",
    ],
)

guard_clean = all(
    not row[
        "complete_expression_matrix_loaded"
    ]
    and not row[
        "all_cells_expression_accessed"
    ]
    and not row[
        "module_scores_computed"
    ]
    for row in guard_rows
)

technical_pass = (
    files_opened == 6
    and sections_sampled == 12
    and len(sampled_row_rows)
    == (
        12
        * rows_per_section
    )
    and len(guard_rows) == 6
    and guard_clean
)

if (
    technical_pass
    and automatic_source_recommendation
):

    status_value = (
        "passed_phase10B5_P4D_expression_source_and_"
        "normalization_audit_with_consistent_source_"
        "recommendation_ready_for_prespecified_"
        "module_scoring_source_lock"
    )

elif technical_pass:

    status_value = (
        "completed_phase10B5_P4D_expression_source_"
        "and_normalization_audit_requires_manual_"
        "source_lock"
    )

else:

    status_value = (
        "phase10B5_P4D_requires_manual_review"
    )

status = {
    "phase": "phase10B5_P4D",
    "H5AD_files_audited": files_opened,
    "revised_sections_sampled": (
        sections_sampled
    ),
    "diagnostic_rows_per_section": (
        rows_per_section
    ),
    "deterministic_diagnostic_rows": len(
        sampled_row_rows
    ),
    "expression_sources_audited": (
        sources_audited
    ),
    "consistent_integer_count_candidates": len(
        count_candidates
    ),
    "consistent_continuous_candidates": len(
        continuous_candidates
    ),
    "automatic_source_recommendation_available": (
        automatic_source_recommendation
    ),
    "targeted_expression_diagnostic_values_accessed": (
        True
    ),
    "complete_expression_matrix_loaded": (
        False
    ),
    "all_cells_expression_accessed": (
        False
    ),
    "spatial_coordinate_values_accessed": (
        False
    ),
    "module_scores_computed": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P4D_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B5_P4D_status.tsv",
    [status],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P4D EXPRESSION-SOURCE "
    "AND NORMALIZATION AUDIT =====",
    "",
    (
        "H5AD files audited: "
        f"{files_opened}/6"
    ),
    (
        "Revised sections sampled: "
        f"{sections_sampled}/12"
    ),
    (
        "Diagnostic rows per section: "
        f"{rows_per_section}"
    ),
    (
        "Expression sources audited: "
        f"{sources_audited}"
    ),
    (
        "Consistent integer-count candidates: "
        f"{len(count_candidates)}"
    ),
    (
        "Consistent continuous candidates: "
        f"{len(continuous_candidates)}"
    ),
    (
        "Recommended count source: "
        f"{recommended_count_source or 'NONE'}"
    ),
    (
        "Recommended processed source: "
        f"{recommended_processed_source or 'NONE'}"
    ),
    (
        "Recommended scoring strategy: "
        f"{recommended_scoring_strategy}"
    ),
    "",
    "Targeted diagnostic expression values accessed: TRUE",
    "Complete expression matrix loaded: FALSE",
    "All cells expression accessed: FALSE",
    "Spatial-coordinate values accessed: FALSE",
    "Module scores computed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P4D STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P4D_report.txt"
).write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

print()
print(
    "\n".join(report)
)

print(
    "\n===== CROSS-FILE SOURCE SUMMARY ====="
)

for row in source_summary_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "source_path",
                "files_present",
                "integer_count_like_files",
                "continuous_nonnegative_files",
                "continuous_signed_files",
                "classification_profile",
            )
        )
    )

print(
    "\n===== SOURCE DECISION ====="
)

for key, value in decision.items():

    print(
        f"{key}\t{value}"
    )

checksum_rows: list[dict[str, Any]] = []

for path in sorted(
    out.rglob("*")
):

    if (
        path.is_file()
        and path.name
        != "phase10B5_P4D_SHA256.tsv"
    ):

        checksum_rows.append(
            {
                "sha256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
                "size_bytes": path.stat().st_size,
                "project_relative_path": (
                    path.relative_to(
                        project
                    ).as_posix()
                ),
            }
        )

write_tsv(
    out
    / "phase10B5_P4D_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)

if not technical_pass:

    raise SystemExit(
        "Phase 10B5-P4D requires manual review."
    )
