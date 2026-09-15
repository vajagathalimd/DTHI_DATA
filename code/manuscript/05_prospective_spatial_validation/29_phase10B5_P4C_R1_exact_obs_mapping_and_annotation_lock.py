from __future__ import annotations

import csv
import gzip
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


project = Path(sys.argv[1])
payload_dir = Path(sys.argv[2])
mapping_path = Path(sys.argv[3])
out = Path(sys.argv[4])

column_dir = out / "01_obs_column_lock"
original_dir = out / "02_original_mapping_audit"
design_dir = out / "03_revised_processed_design"
annotation_dir = out / "04_annotation_completeness"
layer_dir = out / "05_layer_depth_availability"
category_dir = out / "06_category_counts"
guard_dir = out / "07_expression_access_guard"
audit_dir = out / "08_audit"

for directory in (
    column_dir,
    original_dir,
    design_dir,
    annotation_dir,
    layer_dir,
    category_dir,
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

chunk_size = 500_000


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

    codes = group["codes"]

    return categories, codes


def section_token(
    archive_name: str,
) -> str:

    stem = Path(
        archive_name
    ).stem

    return stem.split(
        "_",
        1,
    )[1]


mapping_rows, mapping_columns = read_tsv(
    mapping_path
)

if len(mapping_rows) != 13:
    raise SystemExit(
        f"FAIL: expected 13 mapping rows, "
        f"observed {len(mapping_rows)}."
    )

region_overrides = {
    "UMB1759_O1.zip": "Occi",
    "UMB1031_O1.zip": "OP",
}

for row in mapping_rows:

    row["expected_sample"] = row[
        "donor_id"
    ]

    row["expected_region"] = region_overrides.get(
        row["archive_name"],
        section_token(
            row["archive_name"]
        ),
    )

mapping_by_file: dict[
    str,
    list[dict[str, str]],
] = defaultdict(list)

for row in mapping_rows:

    mapping_by_file[
        row[
            "candidate_processed_H5AD"
        ]
    ].append(row)

# Processed-data design:
# BA17 is absent from gw34.h5ad.
# BA18 is therefore promoted from sensitivity to
# the primary GW34 processed-H5AD representative.
final_targets: list[dict[str, str]] = []

for row in mapping_rows:

    archive_name = row[
        "archive_name"
    ]

    if archive_name in {
        "UMB5900_BA17.zip",
        "UMB5900_BA18.zip",
    }:
        continue

    target = dict(row)

    target["effective_analysis_set"] = row[
        "analysis_set"
    ]

    target["source_substitution"] = ""

    final_targets.append(target)

ba18_source = next(
    row
    for row in mapping_rows
    if row["archive_name"]
    == "UMB5900_BA18.zip"
)

ba18_target = dict(
    ba18_source
)

ba18_target[
    "effective_analysis_set"
] = "primary"

ba18_target[
    "source_substitution"
] = (
    "promoted_to_primary_processed_H5AD_"
    "representative_because_BA17_is_absent_"
    "from_gw34_H5AD"
)

final_targets.append(
    ba18_target
)

if len(final_targets) != 12:
    raise SystemExit(
        f"FAIL: revised processed design contains "
        f"{len(final_targets)} rows; expected 12."
    )

final_by_file: dict[
    str,
    list[dict[str, str]],
] = defaultdict(list)

for row in final_targets:

    final_by_file[
        row[
            "candidate_processed_H5AD"
        ]
    ].append(row)

pair_counts: Counter[
    tuple[str, str, str]
] = Counter()

annotation_counts: dict[
    tuple[str, str],
    Counter[int],
] = defaultdict(Counter)

annotation_nonmissing: Counter[
    tuple[str, str]
] = Counter()

annotation_selected: Counter[str] = Counter()

layer_counts: dict[
    str,
    Counter[int],
] = defaultdict(Counter)

layer_nonmissing: Counter[str] = Counter()

depth_finite: Counter[str] = Counter()
depth_sum: Counter[str] = Counter()
depth_min: dict[str, float] = {}
depth_max: dict[str, float] = {}

column_lock_rows: list[dict[str, Any]] = []
category_count_rows: list[dict[str, Any]] = []
guard_rows: list[dict[str, Any]] = []

annotation_categories_by_file: dict[
    tuple[str, str],
    list[str],
] = {}

layer_categories_by_file: dict[
    str,
    list[str],
] = {}

files_opened = 0

for file_index, file_name in enumerate(
    expected_files,
    start=1,
):

    path = payload_dir / file_name

    print(
        f"[{file_index:02d}/06] "
        f"Scanning obs metadata: {file_name}",
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

        annotation_objects: dict[
            str,
            tuple[list[str], h5py.Dataset],
        ] = {}

        for annotation_column in (
            "H1_annotation",
            "H2_annotation",
            "H3_annotation",
        ):

            categories, codes = categorical(
                obs,
                annotation_column,
            )

            annotation_objects[
                annotation_column
            ] = (
                categories,
                codes,
            )

            annotation_categories_by_file[
                (
                    file_name,
                    annotation_column,
                )
            ] = categories

        layer_column = ""
        depth_column = ""

        if "layer" in obs:

            layer_column = "layer"

            categories, codes = categorical(
                obs,
                layer_column,
            )

            layer_categories_by_file[
                file_name
            ] = categories

            layer_codes = codes

        else:

            layer_codes = None

        if "cortical_depth" in obs:

            depth_column = "cortical_depth"

            depth_dataset = obs[
                depth_column
            ]

            if not isinstance(
                depth_dataset,
                h5py.Dataset,
            ):
                raise SystemExit(
                    f"FAIL: {file_name} cortical_depth "
                    "is not a dataset."
                )

        else:

            depth_dataset = None

        spatial_shape = ""

        if (
            "obsm" in handle
            and "spatial" in handle["obsm"]
        ):

            spatial = handle[
                "obsm"
            ][
                "spatial"
            ]

            spatial_shape = "x".join(
                str(value)
                for value in spatial.shape
            )

        n_obs = int(
            sample_codes.shape[0]
        )

        if int(
            region_codes.shape[0]
        ) != n_obs:

            raise SystemExit(
                f"FAIL: sample/region length mismatch "
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

        original_specs = []

        for row in mapping_by_file.get(
            file_name,
            [],
        ):

            original_specs.append(
                {
                    "archive_name": row[
                        "archive_name"
                    ],
                    "sample": row[
                        "expected_sample"
                    ],
                    "region": row[
                        "expected_region"
                    ],
                    "sample_code": sample_lookup.get(
                        row[
                            "expected_sample"
                        ]
                    ),
                    "region_code": region_lookup.get(
                        row[
                            "expected_region"
                        ]
                    ),
                }
            )

        final_specs = []

        for row in final_by_file.get(
            file_name,
            [],
        ):

            final_specs.append(
                {
                    "archive_name": row[
                        "archive_name"
                    ],
                    "sample": row[
                        "expected_sample"
                    ],
                    "region": row[
                        "expected_region"
                    ],
                    "sample_code": sample_lookup.get(
                        row[
                            "expected_sample"
                        ]
                    ),
                    "region_code": region_lookup.get(
                        row[
                            "expected_region"
                        ]
                    ),
                }
            )

        for start in range(
            0,
            n_obs,
            chunk_size,
        ):

            end = min(
                start + chunk_size,
                n_obs,
            )

            sample_chunk = sample_codes[
                start:end
            ]

            region_chunk = region_codes[
                start:end
            ]

            annotation_chunks = {
                column: codes[start:end]
                for column, (
                    categories,
                    codes,
                ) in annotation_objects.items()
            }

            layer_chunk = (
                layer_codes[start:end]
                if layer_codes is not None
                else None
            )

            depth_chunk = (
                depth_dataset[start:end]
                if depth_dataset is not None
                else None
            )

            for spec in original_specs:

                if (
                    spec["sample_code"] is None
                    or spec["region_code"] is None
                ):
                    continue

                mask = (
                    sample_chunk
                    == spec["sample_code"]
                ) & (
                    region_chunk
                    == spec["region_code"]
                )

                pair_counts[
                    (
                        file_name,
                        spec["sample"],
                        spec["region"],
                    )
                ] += int(
                    np.count_nonzero(mask)
                )

            for spec in final_specs:

                if (
                    spec["sample_code"] is None
                    or spec["region_code"] is None
                ):
                    continue

                mask = (
                    sample_chunk
                    == spec["sample_code"]
                ) & (
                    region_chunk
                    == spec["region_code"]
                )

                selected_count = int(
                    np.count_nonzero(mask)
                )

                if selected_count == 0:
                    continue

                archive_name = spec[
                    "archive_name"
                ]

                annotation_selected[
                    archive_name
                ] += selected_count

                for (
                    annotation_column,
                    annotation_chunk,
                ) in annotation_chunks.items():

                    selected_codes = (
                        annotation_chunk[
                            mask
                        ]
                    )

                    valid_codes = selected_codes[
                        selected_codes >= 0
                    ]

                    annotation_nonmissing[
                        (
                            archive_name,
                            annotation_column,
                        )
                    ] += int(
                        valid_codes.size
                    )

                    if valid_codes.size:

                        counts = np.bincount(
                            valid_codes.astype(
                                np.int64,
                                copy=False,
                            )
                        )

                        for category_code, count in enumerate(
                            counts
                        ):

                            if count:

                                annotation_counts[
                                    (
                                        archive_name,
                                        annotation_column,
                                    )
                                ][
                                    category_code
                                ] += int(count)

                if layer_chunk is not None:

                    selected_codes = layer_chunk[
                        mask
                    ]

                    valid_codes = selected_codes[
                        selected_codes >= 0
                    ]

                    layer_nonmissing[
                        archive_name
                    ] += int(
                        valid_codes.size
                    )

                    if valid_codes.size:

                        counts = np.bincount(
                            valid_codes.astype(
                                np.int64,
                                copy=False,
                            )
                        )

                        for category_code, count in enumerate(
                            counts
                        ):

                            if count:

                                layer_counts[
                                    archive_name
                                ][
                                    category_code
                                ] += int(count)

                if depth_chunk is not None:

                    selected_depth = np.asarray(
                        depth_chunk[
                            mask
                        ],
                        dtype=float,
                    )

                    finite = selected_depth[
                        np.isfinite(
                            selected_depth
                        )
                    ]

                    if finite.size:

                        depth_finite[
                            archive_name
                        ] += int(
                            finite.size
                        )

                        depth_sum[
                            archive_name
                        ] += float(
                            finite.sum()
                        )

                        current_min = float(
                            finite.min()
                        )

                        current_max = float(
                            finite.max()
                        )

                        depth_min[
                            archive_name
                        ] = min(
                            depth_min.get(
                                archive_name,
                                current_min,
                            ),
                            current_min,
                        )

                        depth_max[
                            archive_name
                        ] = max(
                            depth_max.get(
                                archive_name,
                                current_max,
                            ),
                            current_max,
                        )

        column_lock_rows.append(
            {
                "processed_H5AD": file_name,
                "obs_rows": n_obs,
                "sample_column": "sample",
                "region_column": "region",
                "primary_annotation_column": (
                    "H1_annotation"
                ),
                "secondary_annotation_column": (
                    "H2_annotation"
                ),
                "fine_annotation_column": (
                    "H3_annotation"
                ),
                "explicit_layer_column": (
                    layer_column
                ),
                "continuous_depth_column": (
                    depth_column
                ),
                "spatial_embedding": (
                    "obsm/spatial"
                ),
                "spatial_shape": spatial_shape,
                "gene_identifier_source": (
                    "var/_index"
                ),
                "H5AD_X_selected_for_scoring": (
                    False
                ),
            }
        )

        guard_rows.append(
            {
                "processed_H5AD": file_name,
                "obs_sample_codes_accessed": True,
                "obs_region_codes_accessed": True,
                "obs_annotation_codes_accessed": True,
                "obs_layer_or_depth_values_accessed": bool(
                    layer_column
                    or depth_column
                ),
                "spatial_values_accessed": False,
                "H5AD_X_values_accessed": False,
                "H5AD_layers_expression_accessed": False,
                "H5AD_raw_expression_accessed": False,
                "module_scores_computed": False,
            }
        )

original_mapping_rows: list[
    dict[str, Any]
] = []

original_mapped = 0

for row in mapping_rows:

    count = pair_counts[
        (
            row[
                "candidate_processed_H5AD"
            ],
            row[
                "expected_sample"
            ],
            row[
                "expected_region"
            ],
        )
    ]

    if count > 0:
        original_mapped += 1

    mapping_status = (
        "exact_sample_region_mapping_confirmed"
        if count > 0
        else (
            "processed_H5AD_region_absent"
            if row[
                "archive_name"
            ] == "UMB5900_BA17.zip"
            else "exact_mapping_not_found"
        )
    )

    original_mapping_rows.append(
        {
            "analysis_set": row[
                "analysis_set"
            ],
            "archive_name": row[
                "archive_name"
            ],
            "donor_id": row[
                "donor_id"
            ],
            "gestational_week": row[
                "gestational_week"
            ],
            "processed_H5AD": row[
                "candidate_processed_H5AD"
            ],
            "locked_sample_value": row[
                "expected_sample"
            ],
            "locked_region_value": row[
                "expected_region"
            ],
            "selected_cell_count": count,
            "mapping_status": mapping_status,
            "expression_values_accessed": False,
        }
    )

write_tsv(
    original_dir
    / "phase10B5_P4C_R1_original_archive_mapping_audit.tsv",
    original_mapping_rows,
    [
        "analysis_set",
        "archive_name",
        "donor_id",
        "gestational_week",
        "processed_H5AD",
        "locked_sample_value",
        "locked_region_value",
        "selected_cell_count",
        "mapping_status",
        "expression_values_accessed",
    ],
)

final_design_rows: list[
    dict[str, Any]
] = []

for target in final_targets:

    count = pair_counts[
        (
            target[
                "candidate_processed_H5AD"
            ],
            target[
                "expected_sample"
            ],
            target[
                "expected_region"
            ],
        )
    ]

    final_design_rows.append(
        {
            "effective_analysis_set": target[
                "effective_analysis_set"
            ],
            "archive_name": target[
                "archive_name"
            ],
            "donor_id": target[
                "donor_id"
            ],
            "gestational_week": target[
                "gestational_week"
            ],
            "panel_class": target[
                "panel_class"
            ],
            "processed_H5AD": target[
                "candidate_processed_H5AD"
            ],
            "locked_sample_value": target[
                "expected_sample"
            ],
            "locked_region_value": target[
                "expected_region"
            ],
            "selected_cell_count": count,
            "source_substitution": target[
                "source_substitution"
            ],
            "processed_mapping_confirmed": (
                count > 0
            ),
            "expression_values_accessed": (
                False
            ),
        }
    )

write_tsv(
    design_dir
    / "phase10B5_P4C_R1_revised_processed_validation_design.tsv",
    final_design_rows,
    [
        "effective_analysis_set",
        "archive_name",
        "donor_id",
        "gestational_week",
        "panel_class",
        "processed_H5AD",
        "locked_sample_value",
        "locked_region_value",
        "selected_cell_count",
        "source_substitution",
        "processed_mapping_confirmed",
        "expression_values_accessed",
    ],
)

annotation_rows: list[
    dict[str, Any]
] = []

for target in final_targets:

    archive_name = target[
        "archive_name"
    ]

    selected = annotation_selected[
        archive_name
    ]

    file_name = target[
        "candidate_processed_H5AD"
    ]

    for annotation_column in (
        "H1_annotation",
        "H2_annotation",
        "H3_annotation",
    ):

        nonmissing = annotation_nonmissing[
            (
                archive_name,
                annotation_column,
            )
        ]

        counts = annotation_counts[
            (
                archive_name,
                annotation_column,
            )
        ]

        categories = (
            annotation_categories_by_file[
                (
                    file_name,
                    annotation_column,
                )
            ]
        )

        observed_categories = [
            categories[code]
            for code, count in sorted(
                counts.items()
            )
            if (
                count > 0
                and code < len(categories)
            )
        ]

        annotation_rows.append(
            {
                "archive_name": archive_name,
                "processed_H5AD": file_name,
                "annotation_column": (
                    annotation_column
                ),
                "selected_cells": selected,
                "nonmissing_annotation_cells": (
                    nonmissing
                ),
                "nonmissing_fraction": (
                    nonmissing / selected
                    if selected
                    else 0.0
                ),
                "observed_annotation_categories": len(
                    observed_categories
                ),
                "annotation_categories": ";".join(
                    observed_categories
                ),
                "primary_use": {
                    "H1_annotation": (
                        "primary_broad_cell_class"
                    ),
                    "H2_annotation": (
                        "secondary_cell_subclass"
                    ),
                    "H3_annotation": (
                        "exploratory_fine_annotation"
                    ),
                }[
                    annotation_column
                ],
            }
        )

        for code, count in sorted(
            counts.items()
        ):

            if code >= len(
                categories
            ):
                continue

            category_count_rows.append(
                {
                    "archive_name": archive_name,
                    "processed_H5AD": file_name,
                    "metadata_column": (
                        annotation_column
                    ),
                    "category_code": code,
                    "category_value": (
                        categories[code]
                    ),
                    "cell_count": count,
                }
            )

write_tsv(
    annotation_dir
    / "phase10B5_P4C_R1_annotation_completeness.tsv",
    annotation_rows,
    [
        "archive_name",
        "processed_H5AD",
        "annotation_column",
        "selected_cells",
        "nonmissing_annotation_cells",
        "nonmissing_fraction",
        "observed_annotation_categories",
        "annotation_categories",
        "primary_use",
    ],
)

layer_rows: list[
    dict[str, Any]
] = []

for target in final_targets:

    archive_name = target[
        "archive_name"
    ]

    file_name = target[
        "candidate_processed_H5AD"
    ]

    selected = annotation_selected[
        archive_name
    ]

    locked_row = next(
        row
        for row in column_lock_rows
        if row[
            "processed_H5AD"
        ] == file_name
    )

    layer_column = locked_row[
        "explicit_layer_column"
    ]

    depth_column = locked_row[
        "continuous_depth_column"
    ]

    if layer_column:

        nonmissing = layer_nonmissing[
            archive_name
        ]

        source_type = (
            "explicit_categorical_layer"
        )

        categories = layer_categories_by_file[
            file_name
        ]

        observed = []

        for code, count in sorted(
            layer_counts[
                archive_name
            ].items()
        ):

            if code >= len(categories):
                continue

            observed.append(
                categories[code]
            )

            category_count_rows.append(
                {
                    "archive_name": archive_name,
                    "processed_H5AD": file_name,
                    "metadata_column": (
                        layer_column
                    ),
                    "category_code": code,
                    "category_value": (
                        categories[code]
                    ),
                    "cell_count": count,
                }
            )

        mean_depth = ""
        minimum_depth = ""
        maximum_depth = ""

    elif depth_column:

        nonmissing = depth_finite[
            archive_name
        ]

        source_type = (
            "continuous_cortical_depth"
        )

        observed = []

        mean_depth = (
            depth_sum[
                archive_name
            ] / nonmissing
            if nonmissing
            else ""
        )

        minimum_depth = depth_min.get(
            archive_name,
            "",
        )

        maximum_depth = depth_max.get(
            archive_name,
            "",
        )

    else:

        nonmissing = 0
        source_type = (
            "no_explicit_layer_or_depth"
        )

        observed = []
        mean_depth = ""
        minimum_depth = ""
        maximum_depth = ""

    layer_rows.append(
        {
            "archive_name": archive_name,
            "processed_H5AD": file_name,
            "selected_cells": selected,
            "layer_depth_source_type": (
                source_type
            ),
            "locked_metadata_column": (
                layer_column
                or depth_column
            ),
            "nonmissing_layer_depth_cells": (
                nonmissing
            ),
            "nonmissing_fraction": (
                nonmissing / selected
                if selected
                else 0.0
            ),
            "observed_layer_categories": ";".join(
                observed
            ),
            "mean_cortical_depth": (
                mean_depth
            ),
            "minimum_cortical_depth": (
                minimum_depth
            ),
            "maximum_cortical_depth": (
                maximum_depth
            ),
            "eligible_for_explicit_layer_analysis": (
                nonmissing > 0
            ),
            "H2_annotation_may_encode_laminar_identity": (
                True
            ),
        }
    )

write_tsv(
    layer_dir
    / "phase10B5_P4C_R1_layer_depth_availability.tsv",
    layer_rows,
    [
        "archive_name",
        "processed_H5AD",
        "selected_cells",
        "layer_depth_source_type",
        "locked_metadata_column",
        "nonmissing_layer_depth_cells",
        "nonmissing_fraction",
        "observed_layer_categories",
        "mean_cortical_depth",
        "minimum_cortical_depth",
        "maximum_cortical_depth",
        "eligible_for_explicit_layer_analysis",
        "H2_annotation_may_encode_laminar_identity",
    ],
)

write_tsv_gz(
    category_dir
    / "phase10B5_P4C_R1_selected_metadata_category_counts.tsv.gz",
    category_count_rows,
    [
        "archive_name",
        "processed_H5AD",
        "metadata_column",
        "category_code",
        "category_value",
        "cell_count",
    ],
)

write_tsv(
    column_dir
    / "phase10B5_P4C_R1_obs_column_lock.tsv",
    column_lock_rows,
    [
        "processed_H5AD",
        "obs_rows",
        "sample_column",
        "region_column",
        "primary_annotation_column",
        "secondary_annotation_column",
        "fine_annotation_column",
        "explicit_layer_column",
        "continuous_depth_column",
        "spatial_embedding",
        "spatial_shape",
        "gene_identifier_source",
        "H5AD_X_selected_for_scoring",
    ],
)

write_tsv(
    guard_dir
    / "phase10B5_P4C_R1_expression_access_guard.tsv",
    guard_rows,
    [
        "processed_H5AD",
        "obs_sample_codes_accessed",
        "obs_region_codes_accessed",
        "obs_annotation_codes_accessed",
        "obs_layer_or_depth_values_accessed",
        "spatial_values_accessed",
        "H5AD_X_values_accessed",
        "H5AD_layers_expression_accessed",
        "H5AD_raw_expression_accessed",
        "module_scores_computed",
    ],
)

final_mapped = sum(
    bool(
        row[
            "processed_mapping_confirmed"
        ]
    )
    for row in final_design_rows
)

primary_count = sum(
    row[
        "effective_analysis_set"
    ] == "primary"
    for row in final_design_rows
)

sensitivity_count = sum(
    row[
        "effective_analysis_set"
    ] == "section_sensitivity"
    for row in final_design_rows
)

explicit_layer_sections = sum(
    bool(
        row[
            "eligible_for_explicit_layer_analysis"
        ]
    )
    for row in layer_rows
)

h1_complete_sections = sum(
    row[
        "annotation_column"
    ] == "H1_annotation"
    and row[
        "nonmissing_fraction"
    ] >= 0.99
    for row in annotation_rows
)

expression_guard_clean = all(
    not row[
        "H5AD_X_values_accessed"
    ]
    and not row[
        "H5AD_layers_expression_accessed"
    ]
    and not row[
        "H5AD_raw_expression_accessed"
    ]
    for row in guard_rows
)

technical_pass = (
    files_opened == 6
    and original_mapped == 12
    and final_mapped == 12
    and primary_count == 8
    and sensitivity_count == 4
    and h1_complete_sections == 12
    and expression_guard_clean
)

status_value = (
    "passed_phase10B5_P4C_R1_exact_sample_region_"
    "annotation_lock_with_GW34_BA18_processed_"
    "substitution_ready_for_expression_source_and_"
    "normalization_audit"
    if technical_pass
    else (
        "phase10B5_P4C_R1_requires_manual_review"
    )
)

status = {
    "phase": "phase10B5_P4C_R1",
    "H5AD_files_scanned": files_opened,
    "original_archive_mappings": 13,
    "original_exact_mappings_confirmed": (
        original_mapped
    ),
    "original_unavailable_processed_sections": (
        13 - original_mapped
    ),
    "revised_processed_design_sections": len(
        final_design_rows
    ),
    "revised_primary_sections": primary_count,
    "revised_section_sensitivity_sections": (
        sensitivity_count
    ),
    "revised_exact_mappings_confirmed": (
        final_mapped
    ),
    "GW34_BA18_promoted_to_primary": True,
    "GW34_BA17_retained_in_raw_audit_only": True,
    "sections_with_complete_H1_annotation": (
        h1_complete_sections
    ),
    "sections_with_explicit_layer_or_depth": (
        explicit_layer_sections
    ),
    "sample_codes_accessed": True,
    "region_codes_accessed": True,
    "annotation_metadata_accessed": True,
    "layer_depth_metadata_accessed": True,
    "spatial_coordinate_values_accessed": False,
    "H5AD_X_values_accessed": False,
    "H5AD_layers_expression_accessed": False,
    "H5AD_raw_expression_accessed": False,
    "module_scores_computed": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P4C_R1_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B5_P4C_R1_status.tsv",
    [status],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P4C-R1 EXACT OBS "
    "MAPPING AND ANNOTATION LOCK =====",
    "",
    (
        "H5AD files scanned: "
        f"{files_opened}/6"
    ),
    (
        "Original exact mappings confirmed: "
        f"{original_mapped}/13"
    ),
    (
        "Original unavailable processed sections: "
        f"{13 - original_mapped}"
    ),
    (
        "Revised processed-design sections: "
        f"{len(final_design_rows)}"
    ),
    (
        "Revised primary sections: "
        f"{primary_count}"
    ),
    (
        "Revised section-sensitivity sections: "
        f"{sensitivity_count}"
    ),
    (
        "Revised exact mappings confirmed: "
        f"{final_mapped}/{len(final_design_rows)}"
    ),
    (
        "Sections with complete H1 annotation: "
        f"{h1_complete_sections}/12"
    ),
    (
        "Sections with explicit layer/depth: "
        f"{explicit_layer_sections}/12"
    ),
    "",
    "GW34 BA18 promoted to processed primary: TRUE",
    "GW34 BA17 retained in raw audit only: TRUE",
    "Spatial-coordinate values accessed: FALSE",
    "H5AD X values accessed: FALSE",
    "H5AD expression layers accessed: FALSE",
    "H5AD raw expression accessed: FALSE",
    "Module scores computed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P4C-R1 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P4C_R1_report.txt"
).write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

print()
print(
    "\n".join(report)
)

print(
    "\n===== ORIGINAL MAPPING AUDIT ====="
)

for row in original_mapping_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "archive_name",
                "processed_H5AD",
                "locked_sample_value",
                "locked_region_value",
                "selected_cell_count",
                "mapping_status",
            )
        )
    )

print(
    "\n===== REVISED PROCESSED DESIGN ====="
)

for row in final_design_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "effective_analysis_set",
                "archive_name",
                "processed_H5AD",
                "locked_sample_value",
                "locked_region_value",
                "selected_cell_count",
                "source_substitution",
            )
        )
    )

print(
    "\n===== LAYER/DEPTH AVAILABILITY ====="
)

for row in layer_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "archive_name",
                "processed_H5AD",
                "layer_depth_source_type",
                "locked_metadata_column",
                "nonmissing_layer_depth_cells",
                "nonmissing_fraction",
                "eligible_for_explicit_layer_analysis",
            )
        )
    )

checksum_rows: list[dict[str, Any]] = []

for path in sorted(
    out.rglob("*")
):

    if (
        path.is_file()
        and path.name
        != "phase10B5_P4C_R1_SHA256.tsv"
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
    / "phase10B5_P4C_R1_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)

if not technical_pass:
    raise SystemExit(
        "Phase 10B5-P4C-R1 requires manual review."
    )
