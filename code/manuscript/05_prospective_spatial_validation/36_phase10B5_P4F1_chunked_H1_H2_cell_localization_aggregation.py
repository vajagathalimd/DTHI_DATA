from __future__ import annotations

import csv
import hashlib
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


project = Path(sys.argv[1])
payload_dir = Path(sys.argv[2])
localization_plan_path = Path(sys.argv[3])
moment_path = Path(sys.argv[4])
annotation_path = Path(sys.argv[5])
out = Path(sys.argv[6])

section_dir = out / "01_section_module_audit"
h1_dir = out / "02_H1_localization"
h2_dir = out / "03_H2_localization"
call_dir = out / "04_top_bottom_calls"
processing_dir = out / "05_processing_audit"
guard_dir = out / "06_expression_access_guard"
audit_dir = out / "07_audit"

for directory in (
    section_dir,
    h1_dir,
    h2_dir,
    call_dir,
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
variance_tolerance = 1e-12
mean_tolerance = 5e-5
reconciliation_tolerance = 5e-5


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


def section_sort_key(
    metadata: dict[str, Any],
) -> tuple[Any, ...]:

    return (
        int(
            metadata[
                "gestational_week"
            ]
        ),
        metadata[
            "donor_id"
        ],
        metadata[
            "archive_name"
        ],
    )


def descriptive_statistics(
    values: np.ndarray,
) -> dict[str, float]:

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.size == 0:

        raise ValueError(
            "Cannot summarize an empty group."
        )

    quantiles = np.quantile(
        array,
        [
            0.25,
            0.50,
            0.75,
        ],
    )

    return {
        "mean_module_score": float(
            array.mean()
        ),
        "median_module_score": float(
            quantiles[1]
        ),
        "population_SD": float(
            array.std(
                ddof=0
            )
        ),
        "minimum_module_score": float(
            array.min()
        ),
        "Q1_module_score": float(
            quantiles[0]
        ),
        "Q3_module_score": float(
            quantiles[2]
        ),
        "maximum_module_score": float(
            array.max()
        ),
    }


plan_rows, plan_columns = read_tsv(
    localization_plan_path
)

moment_rows, moment_columns = read_tsv(
    moment_path
)

annotation_rows, annotation_columns = read_tsv(
    annotation_path
)

if len(plan_rows) != 68:

    raise SystemExit(
        f"FAIL: expected 68 localization "
        f"estimands; observed {len(plan_rows)}."
    )

if len(moment_rows) != 488:

    raise SystemExit(
        f"FAIL: expected 488 section-gene "
        f"moment rows; observed "
        f"{len(moment_rows)}."
    )

if len(annotation_rows) != 36:

    raise SystemExit(
        f"FAIL: expected 36 annotation-"
        f"completeness rows; observed "
        f"{len(annotation_rows)}."
    )

if not all(
    row[
        "expression_source"
    ] == "/X"
    for row in plan_rows
):

    raise SystemExit(
        "FAIL: localization plan is not uniformly "
        "locked to /X."
    )

if not all(
    row[
        "gene_standardization"
    ] == "within_section_per_gene_zscore_ddof0"
    for row in plan_rows
):

    raise SystemExit(
        "FAIL: unexpected localization "
        "standardization rule."
    )

if any(
    as_bool(
        row[
            "between_section_temporal_claim_authorized"
        ]
    )
    for row in plan_rows
):

    raise SystemExit(
        "FAIL: localization plan authorizes a "
        "between-section temporal claim."
    )

section_metadata: dict[
    str,
    dict[str, Any],
] = {}

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

    metadata = {
        "effective_analysis_set": row[
            "effective_analysis_set"
        ],
        "archive_name": archive_name,
        "donor_id": row[
            "donor_id"
        ],
        "gestational_week": int(
            row[
                "gestational_week"
            ]
        ),
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
        "scoring_family": row[
            "scoring_family"
        ],
    }

    if archive_name in section_metadata:

        previous = dict(
            section_metadata[
                archive_name
            ]
        )

        previous.pop(
            "scoring_family",
            None,
        )

        comparable = dict(
            metadata
        )

        comparable.pop(
            "scoring_family",
            None,
        )

        if previous != comparable:

            raise SystemExit(
                f"FAIL: inconsistent section "
                f"metadata for {archive_name}."
            )

    else:

        section_metadata[
            archive_name
        ] = metadata

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

    if len(genes) != int(
        row[
            "gene_count"
        ]
    ):

        raise SystemExit(
            f"FAIL: gene count mismatch for "
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

if len(section_metadata) != 12:

    raise SystemExit(
        f"FAIL: expected 12 sections; "
        f"observed {len(section_metadata)}."
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

core_sections = [
    archive_name
    for archive_name, modules
    in section_modules.items()
    if len(modules) == 5
]

expanded_sections = [
    archive_name
    for archive_name, modules
    in section_modules.items()
    if len(modules) == 9
]

if len(core_sections) != 10:

    raise SystemExit(
        f"FAIL: expected ten five-module sections; "
        f"observed {len(core_sections)}."
    )

if len(expanded_sections) != 2:

    raise SystemExit(
        f"FAIL: expected two nine-module sections; "
        f"observed {len(expanded_sections)}."
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
                f"FAIL: section moment absent for "
                f"{archive_name}, {gene}."
            )

annotation_expected: dict[
    tuple[str, str],
    int,
] = {}

for row in annotation_rows:

    annotation_column = row[
        "annotation_column"
    ]

    if annotation_column not in {
        "H1_annotation",
        "H2_annotation",
        "H3_annotation",
    }:

        raise SystemExit(
            f"FAIL: unexpected annotation column: "
            f"{annotation_column}"
        )

    if annotation_column in {
        "H1_annotation",
        "H2_annotation",
    }:

        if float(
            row[
                "nonmissing_fraction"
            ]
        ) < 0.999999:

            raise SystemExit(
                f"FAIL: incomplete {annotation_column} "
                f"for {row['archive_name']}."
            )

        annotation_expected[
            (
                row[
                    "archive_name"
                ],
                annotation_column,
            )
        ] = int(
            row[
                "observed_annotation_categories"
            ]
        )

if len(annotation_expected) != 24:

    raise SystemExit(
        f"FAIL: expected 24 H1/H2 annotation "
        f"locks; observed "
        f"{len(annotation_expected)}."
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
    ].sort(
        key=lambda archive: section_sort_key(
            section_metadata[
                archive
            ]
        )
    )

if set(
    sections_by_file
) != set(
    expected_files
):

    raise SystemExit(
        "FAIL: localization design does not cover "
        "the six locked H5AD objects."
    )

expected_H1_rows = sum(
    annotation_expected[
        (
            archive_name,
            "H1_annotation",
        )
    ]
    * len(
        section_modules[
            archive_name
        ]
    )
    for archive_name in section_metadata
)

expected_H2_rows = sum(
    annotation_expected[
        (
            archive_name,
            "H2_annotation",
        )
    ]
    * len(
        section_modules[
            archive_name
        ]
    )
    for archive_name in section_metadata
)

expected_transient_scores = sum(
    metadata[
        "selected_cell_count"
    ]
    * len(
        section_modules[
            archive_name
        ]
    )
    for archive_name, metadata
    in section_metadata.items()
)

section_audit_rows: list[
    dict[str, Any]
] = []

H1_rows: list[
    dict[str, Any]
] = []

H2_rows: list[
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
selected_cells_processed = 0
transient_cell_module_scores = 0
missing_H1_cells = 0
missing_H2_cells = 0

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

    module_position = {
        module: index
        for index, module
        in enumerate(
            file_modules
        )
    }

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
        f"Streaming H1/H2 localization: "
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
                f"FAIL: incomplete CSR object in "
                f"{file_name}."
            )

        matrix_rows, matrix_columns = matrix_shape(
            x_object
        )

        obs = handle[
            "obs"
        ]

        sample_categories, sample_codes = categorical(
            obs,
            "sample",
        )

        region_categories, region_codes = categorical(
            obs,
            "region",
        )

        H1_categories, H1_codes = categorical(
            obs,
            "H1_annotation",
        )

        H2_categories, H2_codes = categorical(
            obs,
            "H2_annotation",
        )

        for codes, label in (
            (sample_codes, "sample"),
            (region_codes, "region"),
            (H1_codes, "H1"),
            (H2_codes, "H2"),
        ):

            if int(
                codes.shape[0]
            ) != matrix_rows:

                raise SystemExit(
                    f"FAIL: {label} length differs "
                    f"from /X rows in {file_name}."
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
                f"FAIL: var index absent from "
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
                f"FAIL: required genes absent from "
                f"{file_name}: "
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
                matrix_gene_position[
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

            if metadata[
                "sample_value"
            ] not in sample_lookup:

                raise SystemExit(
                    f"FAIL: sample "
                    f"{metadata['sample_value']} absent "
                    f"from {file_name}."
                )

            if metadata[
                "region_value"
            ] not in region_lookup:

                raise SystemExit(
                    f"FAIL: region "
                    f"{metadata['region_value']} absent "
                    f"from {file_name}."
                )

            section_specs.append(
                {
                    "archive_name": archive_name,
                    "sample_code": sample_lookup[
                        metadata[
                            "sample_value"
                        ]
                    ],
                    "region_code": region_lookup[
                        metadata[
                            "region_value"
                        ]
                    ],
                    "expected_cells": metadata[
                        "selected_cell_count"
                    ],
                }
            )

        section_count = len(
            section_specs
        )

        gene_count = len(
            required_genes
        )

        module_count = len(
            file_modules
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

        local_gene_position = {
            gene: index
            for index, gene
            in enumerate(
                required_genes
            )
        }

        for section_index, spec in enumerate(
            section_specs
        ):

            archive_name = spec[
                "archive_name"
            ]

            if set(
                section_modules[
                    archive_name
                ]
            ) != set(
                file_modules
            ):

                raise SystemExit(
                    f"FAIL: sections within "
                    f"{file_name} have incompatible "
                    "module sets."
                )

            for module in section_modules[
                archive_name
            ]:

                module_index = module_position[
                    module
                ]

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

                    gene_index = local_gene_position[
                        gene
                    ]

                    if SD_value <= variance_tolerance:

                        continue

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

        score_chunks: dict[
            str,
            list[np.ndarray],
        ] = {
            archive_name: []
            for archive_name in archives
        }

        H1_chunks: dict[
            str,
            list[np.ndarray],
        ] = {
            archive_name: []
            for archive_name in archives
        }

        H2_chunks: dict[
            str,
            list[np.ndarray],
        ] = {
            archive_name: []
            for archive_name in archives
        }

        selected_counts = np.zeros(
            section_count,
            dtype=np.int64,
        )

        selected_CSR_values_read = 0
        target_CSR_values_used = 0
        file_transient_scores = 0

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

            H1_chunk = H1_codes[
                start:end
            ]

            H2_chunk = H2_codes[
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
                        f"assignment in {file_name}."
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

            selected_H1 = np.asarray(
                H1_chunk[
                    selected_rows
                ],
                dtype=np.int16,
            )

            selected_H2 = np.asarray(
                H2_chunk[
                    selected_rows
                ],
                dtype=np.int16,
            )

            missing_H1_cells += int(
                np.count_nonzero(
                    selected_H1 < 0
                )
            )

            missing_H2_cells += int(
                np.count_nonzero(
                    selected_H2 < 0
                )
            )

            selected_counts += np.bincount(
                selected_section_indices,
                minlength=section_count,
            ).astype(
                np.int64
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
                int(
                    np.count_nonzero(
                        selected_rows
                    )
                ),
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

                entry_sections = row_assignment[
                    local_rows
                ]

                selected_entry_mask = (
                    entry_sections
                    >= 0
                )

                selected_CSR_values_read += int(
                    np.count_nonzero(
                        selected_entry_mask
                    )
                )

                if np.any(
                    selected_entry_mask
                ):

                    selected_matrix_columns = (
                        indices_chunk[
                            selected_entry_mask
                        ]
                    )

                    selected_values = values_chunk[
                        selected_entry_mask
                    ]

                    selected_entry_sections = (
                        entry_sections[
                            selected_entry_mask
                        ]
                    )

                    selected_entry_rows = local_rows[
                        selected_entry_mask
                    ]

                    target_gene_indices = target_lookup[
                        selected_matrix_columns
                    ]

                    target_mask = (
                        target_gene_indices
                        >= 0
                    )

                    target_CSR_values_used += int(
                        np.count_nonzero(
                            target_mask
                        )
                    )

                    if np.any(
                        target_mask
                    ):

                        target_values = selected_values[
                            target_mask
                        ]

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

                            nonzero_coefficients = (
                                module_coefficients
                                != 0.0
                            )

                            if not np.any(
                                nonzero_coefficients
                            ):

                                continue

                            additions = np.bincount(
                                target_row_positions[
                                    nonzero_coefficients
                                ],
                                weights=(
                                    target_values[
                                        nonzero_coefficients
                                    ]
                                    * module_coefficients[
                                        nonzero_coefficients
                                    ]
                                ),
                                minlength=scores.shape[0],
                            )

                            scores[
                                :,
                                module_index,
                            ] += additions

            transient_cell_module_scores += int(
                scores.size
            )

            file_transient_scores += int(
                scores.size
            )

            for section_index, spec in enumerate(
                section_specs
            ):

                archive_name = spec[
                    "archive_name"
                ]

                section_mask = (
                    selected_section_indices
                    == section_index
                )

                if not np.any(
                    section_mask
                ):

                    continue

                score_chunks[
                    archive_name
                ].append(
                    scores[
                        section_mask
                    ].astype(
                        np.float32,
                        copy=False,
                    )
                )

                H1_chunks[
                    archive_name
                ].append(
                    selected_H1[
                        section_mask
                    ]
                )

                H2_chunks[
                    archive_name
                ].append(
                    selected_H2[
                        section_mask
                    ]
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
                f"FAIL: selected-cell count mismatch "
                f"in {file_name}: {details}"
            )

        for section_index, spec in enumerate(
            section_specs
        ):

            archive_name = spec[
                "archive_name"
            ]

            section_scores = np.concatenate(
                score_chunks[
                    archive_name
                ],
                axis=0,
            ).astype(
                np.float64,
                copy=False,
            )

            section_H1 = np.concatenate(
                H1_chunks[
                    archive_name
                ],
                axis=0,
            )

            section_H2 = np.concatenate(
                H2_chunks[
                    archive_name
                ],
                axis=0,
            )

            expected_cells = spec[
                "expected_cells"
            ]

            if (
                section_scores.shape[0]
                != expected_cells
                or section_H1.shape[0]
                != expected_cells
                or section_H2.shape[0]
                != expected_cells
            ):

                raise SystemExit(
                    f"FAIL: retained localization "
                    f"rows differ from the expected "
                    f"cell count for {archive_name}."
                )

            if section_scores.shape[1] != len(
                file_modules
            ):

                raise SystemExit(
                    f"FAIL: module-column mismatch "
                    f"for {archive_name}."
                )

            sections_processed += 1
            selected_cells_processed += (
                expected_cells
            )

            observed_H1_codes = sorted(
                int(value)
                for value in np.unique(
                    section_H1
                )
                if int(value) >= 0
            )

            observed_H2_codes = sorted(
                int(value)
                for value in np.unique(
                    section_H2
                )
                if int(value) >= 0
            )

            expected_H1_categories = (
                annotation_expected[
                    (
                        archive_name,
                        "H1_annotation",
                    )
                ]
            )

            expected_H2_categories = (
                annotation_expected[
                    (
                        archive_name,
                        "H2_annotation",
                    )
                ]
            )

            if len(
                observed_H1_codes
            ) != expected_H1_categories:

                raise SystemExit(
                    f"FAIL: {archive_name} has "
                    f"{len(observed_H1_codes)} observed "
                    f"H1 categories; expected "
                    f"{expected_H1_categories}."
                )

            if len(
                observed_H2_codes
            ) != expected_H2_categories:

                raise SystemExit(
                    f"FAIL: {archive_name} has "
                    f"{len(observed_H2_codes)} observed "
                    f"H2 categories; expected "
                    f"{expected_H2_categories}."
                )

            for module_index, module in enumerate(
                file_modules
            ):

                values = section_scores[
                    :,
                    module_index,
                ]

                section_mean = float(
                    values.mean()
                )

                section_audit_rows.append(
                    {
                        "effective_analysis_set": (
                            section_metadata[
                                archive_name
                            ][
                                "effective_analysis_set"
                            ]
                        ),
                        "archive_name": archive_name,
                        "donor_id": section_metadata[
                            archive_name
                        ][
                            "donor_id"
                        ],
                        "gestational_week": (
                            section_metadata[
                                archive_name
                            ][
                                "gestational_week"
                            ]
                        ),
                        "processed_H5AD": file_name,
                        "module": module,
                        "selected_cells": expected_cells,
                        "section_wide_mean": (
                            section_mean
                        ),
                        "section_wide_population_SD": float(
                            values.std(
                                ddof=0
                            )
                        ),
                        "section_wide_minimum": float(
                            values.min()
                        ),
                        "section_wide_maximum": float(
                            values.max()
                        ),
                        "absolute_section_mean": abs(
                            section_mean
                        ),
                        "mean_expected_to_be_zero": True,
                        "valid_between_section_temporal_estimand": (
                            False
                        ),
                        "valid_within_section_localization_estimand": (
                            True
                        ),
                    }
                )

                for code in observed_H1_codes:

                    mask = (
                        section_H1
                        == code
                    )

                    category_values = values[
                        mask
                    ]

                    statistics = descriptive_statistics(
                        category_values
                    )

                    H1_rows.append(
                        {
                            "effective_analysis_set": (
                                section_metadata[
                                    archive_name
                                ][
                                    "effective_analysis_set"
                                ]
                            ),
                            "archive_name": archive_name,
                            "donor_id": section_metadata[
                                archive_name
                            ][
                                "donor_id"
                            ],
                            "gestational_week": (
                                section_metadata[
                                    archive_name
                                ][
                                    "gestational_week"
                                ]
                            ),
                            "processed_H5AD": file_name,
                            "scoring_family": (
                                "expanded_panel_only"
                                if module not in {
                                    "activity_dependent_plasticity",
                                    "astrocyte_maturation_metabolic_support",
                                    "neurogenesis_migration_layering",
                                    "patterning_arealization",
                                    "progenitor_radial_glia",
                                }
                                else "cross_panel_harmonized_core"
                            ),
                            "module": module,
                            "annotation_level": (
                                "H1_annotation"
                            ),
                            "annotation_code": code,
                            "annotation_label": (
                                H1_categories[
                                    code
                                ]
                            ),
                            "cell_count": int(
                                category_values.size
                            ),
                            "cell_fraction_within_section": (
                                category_values.size
                                / expected_cells
                            ),
                            **statistics,
                            "formal_cell_level_inference": (
                                False
                            ),
                            "between_section_pooling": (
                                False
                            ),
                        }
                    )

                for code in observed_H2_codes:

                    mask = (
                        section_H2
                        == code
                    )

                    category_values = values[
                        mask
                    ]

                    statistics = descriptive_statistics(
                        category_values
                    )

                    H2_rows.append(
                        {
                            "effective_analysis_set": (
                                section_metadata[
                                    archive_name
                                ][
                                    "effective_analysis_set"
                                ]
                            ),
                            "archive_name": archive_name,
                            "donor_id": section_metadata[
                                archive_name
                            ][
                                "donor_id"
                            ],
                            "gestational_week": (
                                section_metadata[
                                    archive_name
                                ][
                                    "gestational_week"
                                ]
                            ),
                            "processed_H5AD": file_name,
                            "scoring_family": (
                                "expanded_panel_only"
                                if module not in {
                                    "activity_dependent_plasticity",
                                    "astrocyte_maturation_metabolic_support",
                                    "neurogenesis_migration_layering",
                                    "patterning_arealization",
                                    "progenitor_radial_glia",
                                }
                                else "cross_panel_harmonized_core"
                            ),
                            "module": module,
                            "annotation_level": (
                                "H2_annotation"
                            ),
                            "annotation_code": code,
                            "annotation_label": (
                                H2_categories[
                                    code
                                ]
                            ),
                            "cell_count": int(
                                category_values.size
                            ),
                            "cell_fraction_within_section": (
                                category_values.size
                                / expected_cells
                            ),
                            **statistics,
                            "formal_cell_level_inference": (
                                False
                            ),
                            "between_section_pooling": (
                                False
                            ),
                        }
                    )

        processing_rows.append(
            {
                "processed_H5AD": file_name,
                "selected_sections": (
                    section_count
                ),
                "modules_localized": (
                    module_count
                ),
                "required_target_genes": (
                    gene_count
                ),
                "selected_cells_processed": int(
                    selected_counts.sum()
                ),
                "transient_cell_module_scores": (
                    file_transient_scores
                ),
                "selected_CSR_values_read": (
                    selected_CSR_values_read
                ),
                "target_CSR_values_used": (
                    target_CSR_values_used
                ),
                "complete_X_matrix_materialized": (
                    False
                ),
                "cell_level_score_matrix_retained": (
                    False
                ),
                "raw_X_values_accessed": (
                    False
                ),
            }
        )

        guard_rows.append(
            {
                "processed_H5AD": file_name,
                "expression_source": "/X",
                "H1_metadata_accessed": True,
                "H2_metadata_accessed": True,
                "H3_metadata_accessed": False,
                "chunked_expression_values_accessed": (
                    True
                ),
                "cell_level_module_scores_transiently_computed": (
                    True
                ),
                "cell_level_score_matrix_retained": (
                    False
                ),
                "section_H1_aggregates_computed": (
                    True
                ),
                "section_H2_aggregates_computed": (
                    True
                ),
                "cell_level_hypothesis_tests_performed": (
                    False
                ),
                "between_section_pooling_performed": (
                    False
                ),
                "spatial_coordinates_accessed": (
                    False
                ),
                "raw_X_values_accessed": (
                    False
                ),
            }
        )


def add_local_ranks(
    rows: list[dict[str, Any]],
) -> None:

    grouped: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:

        grouped[
            (
                row[
                    "archive_name"
                ],
                row[
                    "module"
                ],
            )
        ].append(
            row
        )

    for group_rows in grouped.values():

        ordered = sorted(
            group_rows,
            key=lambda row: (
                -float(
                    row[
                        "mean_module_score"
                    ]
                ),
                row[
                    "annotation_label"
                ],
            ),
        )

        group_size = len(
            ordered
        )

        for rank, row in enumerate(
            ordered,
            start=1,
        ):

            row[
                "descending_mean_rank"
            ] = rank

            row[
                "annotation_groups_in_section_module"
            ] = group_size

            row[
                "top_localization_group"
            ] = (
                rank == 1
            )

            row[
                "bottom_localization_group"
            ] = (
                rank == group_size
            )


add_local_ranks(
    H1_rows
)

add_local_ranks(
    H2_rows
)

summary_columns = [
    "effective_analysis_set",
    "archive_name",
    "donor_id",
    "gestational_week",
    "processed_H5AD",
    "scoring_family",
    "module",
    "annotation_level",
    "annotation_code",
    "annotation_label",
    "cell_count",
    "cell_fraction_within_section",
    "mean_module_score",
    "median_module_score",
    "population_SD",
    "minimum_module_score",
    "Q1_module_score",
    "Q3_module_score",
    "maximum_module_score",
    "descending_mean_rank",
    "annotation_groups_in_section_module",
    "top_localization_group",
    "bottom_localization_group",
    "formal_cell_level_inference",
    "between_section_pooling",
]

write_tsv(
    section_dir
    / "phase10B5_P4F1_section_module_zero_mean_audit.tsv",
    section_audit_rows,
    [
        "effective_analysis_set",
        "archive_name",
        "donor_id",
        "gestational_week",
        "processed_H5AD",
        "module",
        "selected_cells",
        "section_wide_mean",
        "section_wide_population_SD",
        "section_wide_minimum",
        "section_wide_maximum",
        "absolute_section_mean",
        "mean_expected_to_be_zero",
        "valid_between_section_temporal_estimand",
        "valid_within_section_localization_estimand",
    ],
)

write_tsv(
    h1_dir
    / "phase10B5_P4F1_section_H1_module_localization.tsv",
    H1_rows,
    summary_columns,
)

write_tsv(
    h2_dir
    / "phase10B5_P4F1_section_H2_module_localization.tsv",
    H2_rows,
    summary_columns,
)

H1_top_bottom_rows = [
    {
        "archive_name": row[
            "archive_name"
        ],
        "donor_id": row[
            "donor_id"
        ],
        "gestational_week": row[
            "gestational_week"
        ],
        "module": row[
            "module"
        ],
        "annotation_level": (
            "H1_annotation"
        ),
        "call_type": (
            "top"
            if row[
                "top_localization_group"
            ]
            else "bottom"
        ),
        "annotation_label": row[
            "annotation_label"
        ],
        "cell_count": row[
            "cell_count"
        ],
        "mean_module_score": row[
            "mean_module_score"
        ],
        "median_module_score": row[
            "median_module_score"
        ],
        "cross_section_label_harmonization_performed": (
            False
        ),
    }
    for row in H1_rows
    if (
        row[
            "top_localization_group"
        ]
        or row[
            "bottom_localization_group"
        ]
    )
]

H2_top_bottom_rows = [
    {
        "archive_name": row[
            "archive_name"
        ],
        "donor_id": row[
            "donor_id"
        ],
        "gestational_week": row[
            "gestational_week"
        ],
        "module": row[
            "module"
        ],
        "annotation_level": (
            "H2_annotation"
        ),
        "call_type": (
            "top"
            if row[
                "top_localization_group"
            ]
            else "bottom"
        ),
        "annotation_label": row[
            "annotation_label"
        ],
        "cell_count": row[
            "cell_count"
        ],
        "mean_module_score": row[
            "mean_module_score"
        ],
        "median_module_score": row[
            "median_module_score"
        ],
        "cross_section_label_harmonization_performed": (
            False
        ),
    }
    for row in H2_rows
    if (
        row[
            "top_localization_group"
        ]
        or row[
            "bottom_localization_group"
        ]
    )
]

call_columns = [
    "archive_name",
    "donor_id",
    "gestational_week",
    "module",
    "annotation_level",
    "call_type",
    "annotation_label",
    "cell_count",
    "mean_module_score",
    "median_module_score",
    "cross_section_label_harmonization_performed",
]

write_tsv(
    call_dir
    / "phase10B5_P4F1_H1_top_bottom_localization_calls.tsv",
    H1_top_bottom_rows,
    call_columns,
)

write_tsv(
    call_dir
    / "phase10B5_P4F1_H2_top_bottom_localization_calls.tsv",
    H2_top_bottom_rows,
    call_columns,
)

H1_top_tally: Counter[
    tuple[str, str]
] = Counter()

module_section_counts: Counter[
    str
] = Counter()

for row in H1_rows:

    if row[
        "top_localization_group"
    ]:

        H1_top_tally[
            (
                row[
                    "module"
                ],
                row[
                    "annotation_label"
                ],
            )
        ] += 1

        module_section_counts[
            row[
                "module"
            ]
        ] += 1

H1_tally_rows = [
    {
        "module": module,
        "exact_H1_annotation_label": label,
        "sections_called_top": count,
        "eligible_sections": (
            module_section_counts[
                module
            ]
        ),
        "top_call_fraction": (
            count
            / module_section_counts[
                module
            ]
        ),
        "cross_section_label_harmonization_performed": (
            False
        ),
        "interpretation": (
            "descriptive_exact_label_tally_only"
        ),
    }
    for (
        module,
        label,
    ), count in sorted(
        H1_top_tally.items()
    )
]

write_tsv(
    call_dir
    / "phase10B5_P4F1_exact_H1_top_call_tally.tsv",
    H1_tally_rows,
    [
        "module",
        "exact_H1_annotation_label",
        "sections_called_top",
        "eligible_sections",
        "top_call_fraction",
        "cross_section_label_harmonization_performed",
        "interpretation",
    ],
)

write_tsv(
    processing_dir
    / "phase10B5_P4F1_file_processing_audit.tsv",
    processing_rows,
    [
        "processed_H5AD",
        "selected_sections",
        "modules_localized",
        "required_target_genes",
        "selected_cells_processed",
        "transient_cell_module_scores",
        "selected_CSR_values_read",
        "target_CSR_values_used",
        "complete_X_matrix_materialized",
        "cell_level_score_matrix_retained",
        "raw_X_values_accessed",
    ],
)

write_tsv(
    guard_dir
    / "phase10B5_P4F1_expression_and_inference_guard.tsv",
    guard_rows,
    [
        "processed_H5AD",
        "expression_source",
        "H1_metadata_accessed",
        "H2_metadata_accessed",
        "H3_metadata_accessed",
        "chunked_expression_values_accessed",
        "cell_level_module_scores_transiently_computed",
        "cell_level_score_matrix_retained",
        "section_H1_aggregates_computed",
        "section_H2_aggregates_computed",
        "cell_level_hypothesis_tests_performed",
        "between_section_pooling_performed",
        "spatial_coordinates_accessed",
        "raw_X_values_accessed",
    ],
)

maximum_absolute_section_mean = max(
    float(
        row[
            "absolute_section_mean"
        ]
    )
    for row in section_audit_rows
)

H1_reconciliation_errors: list[float] = []
H2_reconciliation_errors: list[float] = []

section_mean_lookup = {
    (
        row[
            "archive_name"
        ],
        row[
            "module"
        ],
    ): float(
        row[
            "section_wide_mean"
        ]
    )
    for row in section_audit_rows
}

for rows, error_store in (
    (H1_rows, H1_reconciliation_errors),
    (H2_rows, H2_reconciliation_errors),
):

    grouped: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:

        grouped[
            (
                row[
                    "archive_name"
                ],
                row[
                    "module"
                ],
            )
        ].append(
            row
        )

    for key, group_rows in grouped.items():

        total_cells = sum(
            int(
                row[
                    "cell_count"
                ]
            )
            for row in group_rows
        )

        weighted_mean = sum(
            int(
                row[
                    "cell_count"
                ]
            )
            * float(
                row[
                    "mean_module_score"
                ]
            )
            for row in group_rows
        ) / total_cells

        error_store.append(
            abs(
                weighted_mean
                - section_mean_lookup[
                    key
                ]
            )
        )

maximum_H1_reconciliation_error = max(
    H1_reconciliation_errors
)

maximum_H2_reconciliation_error = max(
    H2_reconciliation_errors
)

technical_pass = (
    files_processed == 6
    and sections_processed == 12
    and selected_cells_processed == 6_062_942
    and transient_cell_module_scores
    == expected_transient_scores
    and len(
        section_audit_rows
    ) == 68
    and len(
        H1_rows
    ) == expected_H1_rows
    and len(
        H2_rows
    ) == expected_H2_rows
    and len(
        H1_top_bottom_rows
    ) == 136
    and len(
        H2_top_bottom_rows
    ) == 136
    and missing_H1_cells == 0
    and missing_H2_cells == 0
    and maximum_absolute_section_mean
    <= mean_tolerance
    and maximum_H1_reconciliation_error
    <= reconciliation_tolerance
    and maximum_H2_reconciliation_error
    <= reconciliation_tolerance
)

status_value = (
    "passed_phase10B5_P4F1_chunked_within_section_"
    "H1_H2_module_localization_aggregation_ready_"
    "for_descriptive_cell_class_synthesis"
    if technical_pass
    else (
        "phase10B5_P4F1_requires_manual_review"
    )
)

status = {
    "phase": "phase10B5_P4F1",
    "H5AD_files_processed": (
        files_processed
    ),
    "revised_sections_processed": (
        sections_processed
    ),
    "selected_cells_processed": (
        selected_cells_processed
    ),
    "section_module_localization_estimands": len(
        section_audit_rows
    ),
    "transient_cell_module_scores_computed": (
        transient_cell_module_scores
    ),
    "expected_transient_cell_module_scores": (
        expected_transient_scores
    ),
    "H1_localization_summary_rows": len(
        H1_rows
    ),
    "expected_H1_localization_summary_rows": (
        expected_H1_rows
    ),
    "H2_localization_summary_rows": len(
        H2_rows
    ),
    "expected_H2_localization_summary_rows": (
        expected_H2_rows
    ),
    "H1_top_bottom_call_rows": len(
        H1_top_bottom_rows
    ),
    "H2_top_bottom_call_rows": len(
        H2_top_bottom_rows
    ),
    "selected_cells_missing_H1": (
        missing_H1_cells
    ),
    "selected_cells_missing_H2": (
        missing_H2_cells
    ),
    "maximum_absolute_section_module_mean": (
        maximum_absolute_section_mean
    ),
    "maximum_H1_weighted_mean_reconciliation_error": (
        maximum_H1_reconciliation_error
    ),
    "maximum_H2_weighted_mean_reconciliation_error": (
        maximum_H2_reconciliation_error
    ),
    "within_section_module_means_approximately_zero": (
        maximum_absolute_section_mean
        <= mean_tolerance
    ),
    "expression_source": "/X",
    "chunked_expression_values_accessed": True,
    "cell_level_module_scores_transiently_computed": (
        True
    ),
    "cell_level_score_matrix_retained": False,
    "section_H1_aggregates_computed": True,
    "section_H2_aggregates_computed": True,
    "H3_metadata_accessed": False,
    "cell_level_hypothesis_tests_performed": False,
    "between_section_pooling_performed": False,
    "cross_section_label_harmonization_performed": (
        False
    ),
    "spatial_coordinates_accessed": False,
    "raw_X_values_accessed": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P4F1_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B5_P4F1_status.tsv",
    [status],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P4F1 CHUNKED H1/H2 "
    "CELL-LOCALIZATION AGGREGATION =====",
    "",
    (
        "H5AD files processed: "
        f"{files_processed}/6"
    ),
    (
        "Revised sections processed: "
        f"{sections_processed}/12"
    ),
    (
        "Selected cells processed: "
        f"{selected_cells_processed}"
    ),
    (
        "Section-module localization estimands: "
        f"{len(section_audit_rows)}"
    ),
    (
        "Transient cell-module scores computed: "
        f"{transient_cell_module_scores}"
    ),
    (
        "H1 localization summary rows: "
        f"{len(H1_rows)}"
    ),
    (
        "H2 localization summary rows: "
        f"{len(H2_rows)}"
    ),
    (
        "Selected cells missing H1: "
        f"{missing_H1_cells}"
    ),
    (
        "Selected cells missing H2: "
        f"{missing_H2_cells}"
    ),
    (
        "Maximum absolute section-module mean: "
        f"{maximum_absolute_section_mean}"
    ),
    (
        "Maximum H1 reconciliation error: "
        f"{maximum_H1_reconciliation_error}"
    ),
    (
        "Maximum H2 reconciliation error: "
        f"{maximum_H2_reconciliation_error}"
    ),
    "",
    "Expression source: /X",
    "Chunked expression values accessed: TRUE",
    (
        "Cell-level module scores transiently "
        "computed: TRUE"
    ),
    "Cell-level score matrix retained: FALSE",
    "Section H1 aggregates computed: TRUE",
    "Section H2 aggregates computed: TRUE",
    "H3 metadata accessed: FALSE",
    "Cell-level hypothesis tests performed: FALSE",
    "Between-section pooling performed: FALSE",
    (
        "Cross-section label harmonization "
        "performed: FALSE"
    ),
    "Spatial coordinates accessed: FALSE",
    "Raw X values accessed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P4F1 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P4F1_report.txt"
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
    "\n===== EXACT H1 TOP-CALL TALLY ====="
)

for row in H1_tally_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "module",
                "exact_H1_annotation_label",
                "sections_called_top",
                "eligible_sections",
                "top_call_fraction",
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
        != "phase10B5_P4F1_SHA256.tsv"
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
    / "phase10B5_P4F1_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)

if not technical_pass:

    raise SystemExit(
        "Phase 10B5-P4F1 requires manual review."
    )
