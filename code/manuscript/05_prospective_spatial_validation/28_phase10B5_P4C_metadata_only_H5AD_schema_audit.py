from __future__ import annotations

import csv
import gzip
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


project = Path(sys.argv[1])
payload_dir = Path(sys.argv[2])
mapping_path = Path(sys.argv[3])
selected_gene_path = Path(sys.argv[4])
out = Path(sys.argv[5])

structure_dir = (
    out
    / "01_H5AD_structure"
)

obs_dir = (
    out
    / "02_obs_schema"
)

var_dir = (
    out
    / "03_var_schema"
)

obsm_dir = (
    out
    / "04_obsm_and_spatial"
)

mapping_dir = (
    out
    / "05_archive_mapping"
)

candidate_dir = (
    out
    / "06_role_candidates"
)

guard_dir = (
    out
    / "07_expression_access_guard"
)

audit_dir = (
    out
    / "08_audit"
)

for directory in (
    structure_dir,
    obs_dir,
    var_dir,
    obsm_dir,
    mapping_dir,
    candidate_dir,
    guard_dir,
    audit_dir,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

expected_files = [
    "gw15.h5ad",
    "gw18_umb1759.h5ad",
    "gw20.h5ad",
    "gw20_umb1031.h5ad",
    "gw22.h5ad",
    "gw34.h5ad",
]

category_export_limit = 10000
preview_value_limit = 20


def read_tsv(
    path: Path,
) -> tuple[
    list[dict[str, str]],
    list[str],
]:

    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        return (
            list(reader),
            reader.fieldnames or [],
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


def decode_value(
    value: Any,
) -> Any:

    if isinstance(
        value,
        bytes,
    ):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    if isinstance(
        value,
        np.bytes_,
    ):
        return bytes(
            value
        ).decode(
            "utf-8",
            errors="replace",
        )

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        np.ndarray,
    ):
        return [
            decode_value(
                item
            )
            for item in value.tolist()
        ]

    return value


def attr_text(
    value: Any,
) -> str:

    decoded = decode_value(
        value
    )

    if isinstance(
        decoded,
        list,
    ):
        return ";".join(
            str(
                item
            )
            for item in decoded
        )

    return str(
        decoded
    )


def normalize_name(
    value: str,
) -> str:

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        value.strip().lower(),
    ).strip(
        "_"
    )


def normalize_token(
    value: str,
) -> str:

    return re.sub(
        r"[^a-z0-9]+",
        "",
        value.strip().lower(),
    )


def object_encoding(
    obj: h5py.Group | h5py.Dataset,
) -> str:

    value = obj.attrs.get(
        "encoding-type",
        "",
    )

    return attr_text(
        value
    )


def object_shape(
    obj: h5py.Group | h5py.Dataset,
) -> str:

    if isinstance(
        obj,
        h5py.Dataset,
    ):
        return "x".join(
            str(
                dimension
            )
            for dimension in obj.shape
        )

    return ""


def object_dtype(
    obj: h5py.Group | h5py.Dataset,
) -> str:

    if isinstance(
        obj,
        h5py.Dataset,
    ):
        return str(
            obj.dtype
        )

    return ""


def read_dataset_preview(
    dataset: h5py.Dataset,
    limit: int = preview_value_limit,
) -> list[str]:

    if dataset.ndim == 0:

        values = [
            dataset[()]
        ]

    elif dataset.shape[0] == 0:

        values = []

    else:

        values = dataset[
            : min(
                dataset.shape[0],
                limit,
            )
        ]

    array = np.asarray(
        values
    ).reshape(
        -1
    )

    return [
        str(
            decode_value(
                value
            )
        )
        for value in array
    ]


def dataframe_columns(
    group: h5py.Group,
) -> tuple[
    list[str],
    str,
]:

    index_key = attr_text(
        group.attrs.get(
            "_index",
            "_index",
        )
    )

    column_order = group.attrs.get(
        "column-order",
        None,
    )

    if column_order is not None:

        decoded = decode_value(
            column_order
        )

        if isinstance(
            decoded,
            list,
        ):

            columns = [
                str(
                    column
                )
                for column in decoded
            ]

        else:

            columns = [
                str(
                    decoded
                )
            ]

    else:

        columns = [
            key
            for key in group.keys()
            if key != index_key
        ]

    return (
        columns,
        index_key,
    )


def categorical_levels(
    obj: h5py.Group | h5py.Dataset,
) -> list[str]:

    if not isinstance(
        obj,
        h5py.Group,
    ):
        return []

    if "categories" not in obj:
        return []

    categories = obj[
        "categories"
    ]

    if not isinstance(
        categories,
        h5py.Dataset,
    ):
        return []

    values = read_dataset_preview(
        categories,
        limit=min(
            categories.shape[0]
            if categories.ndim
            else 1,
            category_export_limit,
        ),
    )

    return values


def preview_object_values(
    obj: h5py.Group | h5py.Dataset,
) -> list[str]:

    if isinstance(
        obj,
        h5py.Dataset,
    ):
        return read_dataset_preview(
            obj
        )

    if "categories" in obj:

        categories = obj[
            "categories"
        ]

        if isinstance(
            categories,
            h5py.Dataset,
        ):
            return read_dataset_preview(
                categories
            )

    if "values" in obj:

        values = obj[
            "values"
        ]

        if isinstance(
            values,
            h5py.Dataset,
        ):
            return read_dataset_preview(
                values
            )

    return []


def classify_role(
    column_name: str,
) -> list[str]:

    normalized = normalize_name(
        column_name
    )

    roles: list[str] = []

    sample_terms = (
        "sample",
        "section",
        "slice",
        "brain",
        "donor",
        "subject",
        "specimen",
        "library",
        "batch",
        "orig_ident",
        "region",
        "area_label",
        "fov",
    )

    annotation_terms = (
        "cell_type",
        "celltype",
        "cell_annotation",
        "annotation",
        "subclass",
        "supertype",
        "major_type",
        "broad_type",
        "cell_class",
        "class_label",
    )

    cluster_terms = (
        "cluster",
        "leiden",
        "louvain",
        "subcluster",
        "community",
    )

    layer_terms = (
        "layer",
        "laminar",
        "cortical_zone",
        "zone_label",
        "cortical_depth",
        "normalized_depth",
        "relative_depth",
        "distance_to_pia",
        "pia_distance",
        "white_matter_distance",
    )

    if any(
        term in normalized
        for term in sample_terms
    ):
        roles.append(
            "sample_or_section"
        )

    if any(
        term in normalized
        for term in annotation_terms
    ):
        roles.append(
            "cell_annotation"
        )

    if any(
        term in normalized
        for term in cluster_terms
    ):
        roles.append(
            "cluster"
        )

    if any(
        term in normalized
        for term in layer_terms
    ) or normalized in {
        "layer",
        "depth",
    }:
        roles.append(
            "layer_or_depth"
        )

    if normalized in {
        "x",
        "center_x",
        "centroid_x",
        "spatial_x",
        "global_x",
    }:
        roles.append(
            "spatial_x"
        )

    if normalized in {
        "y",
        "center_y",
        "centroid_y",
        "spatial_y",
        "global_y",
    }:
        roles.append(
            "spatial_y"
        )

    if normalized in {
        "cell",
        "cell_id",
        "cellid",
        "barcode",
        "entityid",
        "entity_id",
    }:
        roles.append(
            "cell_identifier"
        )

    if not roles:
        roles.append(
            "unclassified"
        )

    return roles


def read_all_text_values(
    obj: h5py.Group | h5py.Dataset,
    maximum_values: int = 20000,
) -> list[str]:

    if isinstance(
        obj,
        h5py.Group,
    ):

        if "categories" in obj:

            categories = obj[
                "categories"
            ]

            if isinstance(
                categories,
                h5py.Dataset,
            ):
                return read_all_text_values(
                    categories,
                    maximum_values,
                )

        if "values" in obj:

            values = obj[
                "values"
            ]

            if isinstance(
                values,
                h5py.Dataset,
            ):
                return read_all_text_values(
                    values,
                    maximum_values,
                )

        return []

    if obj.ndim == 0:

        raw = [
            obj[()]
        ]

    elif obj.shape[0] > maximum_values:

        raw = obj[
            :maximum_values
        ]

    else:

        raw = obj[:]

    return [
        str(
            decode_value(
                value
            )
        )
        for value in np.asarray(
            raw
        ).reshape(
            -1
        )
    ]


mapping_rows, mapping_columns = read_tsv(
    mapping_path
)

if len(mapping_rows) != 13:
    raise SystemExit(
        f"FAIL: expected 13 archive/object mappings, "
        f"observed {len(mapping_rows)}."
    )

mapping_by_object: dict[
    str,
    list[dict[str, str]],
] = defaultdict(list)

for row in mapping_rows:

    mapping_by_object[
        row[
            "candidate_processed_H5AD"
        ]
    ].append(
        row
    )

selected_gene_rows, selected_gene_columns = read_tsv(
    selected_gene_path
)

genes_by_archive: dict[
    str,
    set[str],
] = defaultdict(set)

for row in selected_gene_rows:

    gene = row[
        "gene_symbol"
    ].strip().upper()

    if gene:

        genes_by_archive[
            row[
                "archive_name"
            ]
        ].add(
            gene
        )

structure_rows: list[
    dict[str, Any]
] = []

obs_schema_rows: list[
    dict[str, Any]
] = []

category_rows: list[
    dict[str, Any]
] = []

var_schema_rows: list[
    dict[str, Any]
] = []

gene_inventory_rows: list[
    dict[str, Any]
] = []

gene_coverage_rows: list[
    dict[str, Any]
] = []

obsm_rows: list[
    dict[str, Any]
] = []

role_candidate_rows: list[
    dict[str, Any]
] = []

archive_mapping_audit_rows: list[
    dict[str, Any]
] = []

guard_rows: list[
    dict[str, Any]
] = []

opened_files = 0
obs_groups_found = 0
var_groups_found = 0

archive_tokens_matched = 0

annotation_candidate_files = set()
cluster_candidate_files = set()
layer_candidate_files = set()
spatial_candidate_files = set()
sample_candidate_files = set()

for file_index, file_name in enumerate(
    expected_files,
    start=1,
):

    path = (
        payload_dir
        / file_name
    )

    if not path.is_file():
        raise SystemExit(
            f"FAIL: missing H5AD file: {path}"
        )

    print(
        f"[{file_index:02d}/06] Auditing {file_name}",
        flush=True,
    )

    with h5py.File(
        path,
        "r",
    ) as handle:

        opened_files += 1

        root_keys = sorted(
            handle.keys()
        )

        for key, value in handle.attrs.items():

            structure_rows.append(
                {
                    "file_name": file_name,
                    "object_path": "/",
                    "object_type": "root_attribute",
                    "shape": "",
                    "dtype": "",
                    "encoding_type": "",
                    "attribute_name": key,
                    "attribute_value": attr_text(
                        value
                    ),
                    "expression_value_read": False,
                }
            )

        def structure_visitor(
            object_name: str,
            obj: h5py.Group | h5py.Dataset,
        ) -> None:

            structure_rows.append(
                {
                    "file_name": file_name,
                    "object_path": (
                        "/"
                        + object_name
                    ),
                    "object_type": (
                        "dataset"
                        if isinstance(
                            obj,
                            h5py.Dataset,
                        )
                        else "group"
                    ),
                    "shape": object_shape(
                        obj
                    ),
                    "dtype": object_dtype(
                        obj
                    ),
                    "encoding_type": object_encoding(
                        obj
                    ),
                    "attribute_name": "",
                    "attribute_value": "",
                    "expression_value_read": False,
                }
            )

        handle.visititems(
            structure_visitor
        )

        expression_paths_present = [
            expression_path
            for expression_path in (
                "/X",
                "/layers",
                "/raw",
                "/raw/X",
            )
            if expression_path.lstrip(
                "/"
            )
            in handle
            or (
                expression_path.startswith(
                    "/raw/"
                )
                and "raw" in handle
                and expression_path.split(
                    "/"
                )[-1]
                in handle[
                    "raw"
                ]
            )
        ]

        guard_rows.append(
            {
                "file_name": file_name,
                "root_keys": ";".join(
                    root_keys
                ),
                "expression_paths_present": ";".join(
                    expression_paths_present
                ),
                "H5AD_structure_opened": True,
                "obs_metadata_values_accessed": (
                    "obs" in handle
                ),
                "var_metadata_values_accessed": (
                    "var" in handle
                ),
                "X_values_accessed": False,
                "layer_values_accessed": False,
                "raw_expression_values_accessed": False,
                "module_scores_computed": False,
            }
        )

        if "obs" not in handle:

            continue

        obs = handle[
            "obs"
        ]

        if not isinstance(
            obs,
            h5py.Group,
        ):

            continue

        obs_groups_found += 1

        obs_columns, obs_index_key = dataframe_columns(
            obs
        )

        source_values: dict[
            str,
            list[str],
        ] = {}

        if obs_index_key in obs:

            index_values = preview_object_values(
                obs[
                    obs_index_key
                ]
            )

            source_values[
                f"obs_index:{obs_index_key}"
            ] = index_values

            role_candidate_rows.append(
                {
                    "file_name": file_name,
                    "source": "obs_index",
                    "column_or_key": (
                        obs_index_key
                    ),
                    "candidate_role": (
                        "cell_identifier_or_sample_token"
                    ),
                    "encoding_type": object_encoding(
                        obs[
                            obs_index_key
                        ]
                    ),
                    "preview_or_categories": ";".join(
                        index_values
                    ),
                }
            )

        for column_index, column_name in enumerate(
            obs_columns
        ):

            if column_name not in obs:
                continue

            obj = obs[
                column_name
            ]

            roles = classify_role(
                column_name
            )

            levels = categorical_levels(
                obj
            )

            preview_values = preview_object_values(
                obj
            )

            if levels:

                source_values[
                    f"obs:{column_name}"
                ] = levels

                for level_index, level in enumerate(
                    levels
                ):

                    category_rows.append(
                        {
                            "file_name": file_name,
                            "column_name": (
                                column_name
                            ),
                            "category_index": (
                                level_index
                            ),
                            "category_value": (
                                level
                            ),
                            "categories_truncated": (
                                len(levels)
                                >= category_export_limit
                            ),
                        }
                    )

            elif (
                "sample_or_section"
                in roles
                or "cell_annotation"
                in roles
                or "cluster" in roles
                or "layer_or_depth"
                in roles
            ):

                source_values[
                    f"obs:{column_name}"
                ] = preview_values

            obs_schema_rows.append(
                {
                    "file_name": file_name,
                    "column_index": (
                        column_index
                    ),
                    "column_name": (
                        column_name
                    ),
                    "normalized_column_name": (
                        normalize_name(
                            column_name
                        )
                    ),
                    "object_type": (
                        "dataset"
                        if isinstance(
                            obj,
                            h5py.Dataset,
                        )
                        else "group"
                    ),
                    "encoding_type": object_encoding(
                        obj
                    ),
                    "shape": object_shape(
                        obj
                    ),
                    "dtype": object_dtype(
                        obj
                    ),
                    "inferred_roles": ";".join(
                        roles
                    ),
                    "category_count_exported": len(
                        levels
                    ),
                    "preview_values": ";".join(
                        preview_values
                    ),
                }
            )

            for role in roles:

                if role == "unclassified":
                    continue

                role_candidate_rows.append(
                    {
                        "file_name": file_name,
                        "source": "obs",
                        "column_or_key": (
                            column_name
                        ),
                        "candidate_role": role,
                        "encoding_type": object_encoding(
                            obj
                        ),
                        "preview_or_categories": ";".join(
                            levels[
                                :preview_value_limit
                            ]
                            if levels
                            else preview_values
                        ),
                    }
                )

                if role == "sample_or_section":
                    sample_candidate_files.add(
                        file_name
                    )

                elif role == "cell_annotation":
                    annotation_candidate_files.add(
                        file_name
                    )

                elif role == "cluster":
                    cluster_candidate_files.add(
                        file_name
                    )

                elif role == "layer_or_depth":
                    layer_candidate_files.add(
                        file_name
                    )

                elif role in {
                    "spatial_x",
                    "spatial_y",
                }:
                    spatial_candidate_files.add(
                        file_name
                    )

        expected_mappings = mapping_by_object.get(
            file_name,
            []
        )

        for expected in expected_mappings:

            archive_name = expected[
                "archive_name"
            ]

            archive_stem = Path(
                archive_name
            ).stem

            expected_token = normalize_token(
                archive_stem
            )

            matches: list[
                tuple[
                    int,
                    str,
                    str,
                ]
            ] = []

            for source_name, values in source_values.items():

                for value in values:

                    normalized_value = normalize_token(
                        value
                    )

                    if not normalized_value:
                        continue

                    if normalized_value == expected_token:

                        quality = 3

                    elif normalized_value.startswith(
                        expected_token
                    ):

                        quality = 2

                    elif expected_token in normalized_value:

                        quality = 1

                    else:

                        continue

                    matches.append(
                        (
                            quality,
                            source_name,
                            value,
                        )
                    )

            matches.sort(
                key=lambda item: (
                    -item[0],
                    item[1],
                    item[2],
                )
            )

            if matches:

                best_quality, best_source, best_value = (
                    matches[0]
                )

                mapping_status = (
                    "matched_in_metadata_category_or_preview"
                )

                archive_tokens_matched += 1

            else:

                best_quality = 0
                best_source = ""
                best_value = ""

                mapping_status = (
                    "not_matched_in_schema_preview_"
                    "requires_targeted_obs_value_scan"
                )

            archive_mapping_audit_rows.append(
                {
                    "analysis_set": expected[
                        "analysis_set"
                    ],
                    "archive_name": archive_name,
                    "donor_id": expected[
                        "donor_id"
                    ],
                    "gestational_week": expected[
                        "gestational_week"
                    ],
                    "processed_H5AD": file_name,
                    "matched_source": (
                        best_source
                    ),
                    "matched_value": (
                        best_value
                    ),
                    "match_quality": (
                        best_quality
                    ),
                    "mapping_status": (
                        mapping_status
                    ),
                    "expression_values_accessed": (
                        False
                    ),
                }
            )

        if "var" in handle:

            var = handle[
                "var"
            ]

            if isinstance(
                var,
                h5py.Group,
            ):

                var_groups_found += 1

                var_columns, var_index_key = dataframe_columns(
                    var
                )

                for column_index, column_name in enumerate(
                    var_columns
                ):

                    if column_name not in var:
                        continue

                    obj = var[
                        column_name
                    ]

                    var_schema_rows.append(
                        {
                            "file_name": file_name,
                            "column_index": (
                                column_index
                            ),
                            "column_name": (
                                column_name
                            ),
                            "encoding_type": object_encoding(
                                obj
                            ),
                            "shape": object_shape(
                                obj
                            ),
                            "dtype": object_dtype(
                                obj
                            ),
                            "preview_values": ";".join(
                                preview_object_values(
                                    obj
                                )
                            ),
                        }
                    )

                gene_source = ""

                gene_values: list[str] = []

                gene_priority = (
                    "gene_symbol",
                    "gene_symbols",
                    "symbol",
                    "feature_name",
                    "gene_name",
                )

                normalized_var_lookup = {
                    normalize_name(
                        column
                    ): column
                    for column in var_columns
                    if column in var
                }

                for priority in gene_priority:

                    if priority in normalized_var_lookup:

                        candidate_column = normalized_var_lookup[
                            priority
                        ]

                        candidate_values = read_all_text_values(
                            var[
                                candidate_column
                            ],
                            maximum_values=100000,
                        )

                        if candidate_values:

                            gene_source = (
                                f"var:{candidate_column}"
                            )

                            gene_values = (
                                candidate_values
                            )

                            break

                if (
                    not gene_values
                    and var_index_key in var
                ):

                    gene_source = (
                        f"var_index:{var_index_key}"
                    )

                    gene_values = read_all_text_values(
                        var[
                            var_index_key
                        ],
                        maximum_values=100000,
                    )

                normalized_gene_values = {
                    gene.strip().upper()
                    for gene in gene_values
                    if gene.strip()
                }

                for gene_order, gene in enumerate(
                    gene_values,
                    start=1,
                ):

                    gene_inventory_rows.append(
                        {
                            "file_name": file_name,
                            "gene_source": (
                                gene_source
                            ),
                            "gene_order": (
                                gene_order
                            ),
                            "gene_identifier": (
                                gene
                            ),
                            "normalized_gene_identifier": (
                                gene.strip().upper()
                            ),
                        }
                    )

                expected_genes: set[str] = set()

                for expected in mapping_by_object.get(
                    file_name,
                    [],
                ):

                    expected_genes.update(
                        genes_by_archive.get(
                            expected[
                                "archive_name"
                            ],
                            set(),
                        )
                    )

                matched_genes = sorted(
                    expected_genes
                    & normalized_gene_values
                )

                missing_genes = sorted(
                    expected_genes
                    - normalized_gene_values
                )

                gene_coverage_rows.append(
                    {
                        "file_name": file_name,
                        "gene_source": (
                            gene_source
                        ),
                        "H5AD_gene_identifiers": len(
                            normalized_gene_values
                        ),
                        "prespecified_selected_genes": len(
                            expected_genes
                        ),
                        "matched_selected_genes": len(
                            matched_genes
                        ),
                        "missing_selected_genes": len(
                            missing_genes
                        ),
                        "matched_gene_symbols": ";".join(
                            matched_genes
                        ),
                        "missing_gene_symbols": ";".join(
                            missing_genes
                        ),
                        "expression_values_accessed": (
                            False
                        ),
                    }
                )

        if "obsm" in handle:

            obsm = handle[
                "obsm"
            ]

            if isinstance(
                obsm,
                h5py.Group,
            ):

                for key in sorted(
                    obsm.keys()
                ):

                    obj = obsm[
                        key
                    ]

                    normalized_key = normalize_name(
                        key
                    )

                    spatial_candidate = any(
                        token in normalized_key
                        for token in (
                            "spatial",
                            "coord",
                            "xy",
                            "center",
                        )
                    )

                    if spatial_candidate:
                        spatial_candidate_files.add(
                            file_name
                        )

                    obsm_rows.append(
                        {
                            "file_name": file_name,
                            "obsm_key": key,
                            "object_type": (
                                "dataset"
                                if isinstance(
                                    obj,
                                    h5py.Dataset,
                                )
                                else "group"
                            ),
                            "encoding_type": object_encoding(
                                obj
                            ),
                            "shape": object_shape(
                                obj
                            ),
                            "dtype": object_dtype(
                                obj
                            ),
                            "candidate_spatial_embedding": (
                                spatial_candidate
                            ),
                            "values_accessed": False,
                        }
                    )

                    if spatial_candidate:

                        role_candidate_rows.append(
                            {
                                "file_name": file_name,
                                "source": "obsm",
                                "column_or_key": key,
                                "candidate_role": (
                                    "spatial_embedding"
                                ),
                                "encoding_type": object_encoding(
                                    obj
                                ),
                                "preview_or_categories": "",
                            }
                        )

structure_columns = [
    "file_name",
    "object_path",
    "object_type",
    "shape",
    "dtype",
    "encoding_type",
    "attribute_name",
    "attribute_value",
    "expression_value_read",
]

write_tsv_gz(
    structure_dir
    / "phase10B5_P4C_H5AD_structure_inventory.tsv.gz",
    structure_rows,
    structure_columns,
)

write_tsv(
    obs_dir
    / "phase10B5_P4C_obs_column_schema.tsv",
    obs_schema_rows,
    [
        "file_name",
        "column_index",
        "column_name",
        "normalized_column_name",
        "object_type",
        "encoding_type",
        "shape",
        "dtype",
        "inferred_roles",
        "category_count_exported",
        "preview_values",
    ],
)

write_tsv_gz(
    obs_dir
    / "phase10B5_P4C_obs_categorical_levels.tsv.gz",
    category_rows,
    [
        "file_name",
        "column_name",
        "category_index",
        "category_value",
        "categories_truncated",
    ],
)

write_tsv(
    var_dir
    / "phase10B5_P4C_var_column_schema.tsv",
    var_schema_rows,
    [
        "file_name",
        "column_index",
        "column_name",
        "encoding_type",
        "shape",
        "dtype",
        "preview_values",
    ],
)

write_tsv_gz(
    var_dir
    / "phase10B5_P4C_gene_identifier_inventory.tsv.gz",
    gene_inventory_rows,
    [
        "file_name",
        "gene_source",
        "gene_order",
        "gene_identifier",
        "normalized_gene_identifier",
    ],
)

write_tsv(
    var_dir
    / "phase10B5_P4C_selected_gene_coverage.tsv",
    gene_coverage_rows,
    [
        "file_name",
        "gene_source",
        "H5AD_gene_identifiers",
        "prespecified_selected_genes",
        "matched_selected_genes",
        "missing_selected_genes",
        "matched_gene_symbols",
        "missing_gene_symbols",
        "expression_values_accessed",
    ],
)

write_tsv(
    obsm_dir
    / "phase10B5_P4C_obsm_schema.tsv",
    obsm_rows,
    [
        "file_name",
        "obsm_key",
        "object_type",
        "encoding_type",
        "shape",
        "dtype",
        "candidate_spatial_embedding",
        "values_accessed",
    ],
)

write_tsv(
    mapping_dir
    / "phase10B5_P4C_archive_metadata_mapping_audit.tsv",
    archive_mapping_audit_rows,
    [
        "analysis_set",
        "archive_name",
        "donor_id",
        "gestational_week",
        "processed_H5AD",
        "matched_source",
        "matched_value",
        "match_quality",
        "mapping_status",
        "expression_values_accessed",
    ],
)

write_tsv(
    candidate_dir
    / "phase10B5_P4C_obs_role_candidates.tsv",
    role_candidate_rows,
    [
        "file_name",
        "source",
        "column_or_key",
        "candidate_role",
        "encoding_type",
        "preview_or_categories",
    ],
)

write_tsv(
    guard_dir
    / "phase10B5_P4C_expression_access_guard.tsv",
    guard_rows,
    [
        "file_name",
        "root_keys",
        "expression_paths_present",
        "H5AD_structure_opened",
        "obs_metadata_values_accessed",
        "var_metadata_values_accessed",
        "X_values_accessed",
        "layer_values_accessed",
        "raw_expression_values_accessed",
        "module_scores_computed",
    ],
)

candidate_summary_rows = [
    {
        "candidate_role": role,
        "files_with_candidate": len(
            files
        ),
        "selected_H5AD_files": 6,
        "all_files_have_candidate": (
            len(files) == 6
        ),
    }
    for role, files in (
        (
            "sample_or_section",
            sample_candidate_files,
        ),
        (
            "cell_annotation",
            annotation_candidate_files,
        ),
        (
            "cluster",
            cluster_candidate_files,
        ),
        (
            "layer_or_depth",
            layer_candidate_files,
        ),
        (
            "spatial",
            spatial_candidate_files,
        ),
    )
]

write_tsv(
    candidate_dir
    / "phase10B5_P4C_role_candidate_summary.tsv",
    candidate_summary_rows,
    [
        "candidate_role",
        "files_with_candidate",
        "selected_H5AD_files",
        "all_files_have_candidate",
    ],
)

all_expression_guards_clean = all(
    not bool(
        row[
            "X_values_accessed"
        ]
    )
    and not bool(
        row[
            "layer_values_accessed"
        ]
    )
    and not bool(
        row[
            "raw_expression_values_accessed"
        ]
    )
    for row in guard_rows
)

sample_mapping_complete = (
    archive_tokens_matched == 13
)

annotation_available = (
    len(
        annotation_candidate_files
        | cluster_candidate_files
    ) == 6
)

layer_available = (
    len(
        layer_candidate_files
    ) == 6
)

spatial_available = (
    len(
        spatial_candidate_files
    ) == 6
)

technical_pass = (
    opened_files == 6
    and obs_groups_found == 6
    and var_groups_found == 6
    and len(
        archive_mapping_audit_rows
    ) == 13
    and len(
        guard_rows
    ) == 6
    and all_expression_guards_clean
)

if (
    technical_pass
    and sample_mapping_complete
    and annotation_available
    and spatial_available
):

    status_value = (
        "passed_phase10B5_P4C_metadata_only_H5AD_"
        "schema_audit_ready_for_obs_column_lock_and_"
        "targeted_expression_scoring"
    )

elif technical_pass:

    status_value = (
        "completed_phase10B5_P4C_metadata_only_H5AD_"
        "schema_audit_requires_targeted_obs_mapping_or_"
        "manual_column_lock"
    )

else:

    status_value = (
        "phase10B5_P4C_requires_manual_review"
    )

status = {
    "phase": "phase10B5_P4C",
    "selected_H5AD_files": 6,
    "H5AD_files_opened": (
        opened_files
    ),
    "obs_groups_found": (
        obs_groups_found
    ),
    "var_groups_found": (
        var_groups_found
    ),
    "obs_schema_rows": len(
        obs_schema_rows
    ),
    "obs_categorical_level_rows": len(
        category_rows
    ),
    "obsm_schema_rows": len(
        obsm_rows
    ),
    "archive_mappings_expected": 13,
    "archive_tokens_matched_in_categories_or_previews": (
        archive_tokens_matched
    ),
    "files_with_sample_or_section_candidate": len(
        sample_candidate_files
    ),
    "files_with_annotation_candidate": len(
        annotation_candidate_files
    ),
    "files_with_cluster_candidate": len(
        cluster_candidate_files
    ),
    "files_with_layer_or_depth_candidate": len(
        layer_candidate_files
    ),
    "files_with_spatial_candidate": len(
        spatial_candidate_files
    ),
    "selected_gene_coverage_rows": len(
        gene_coverage_rows
    ),
    "H5AD_structure_opened": True,
    "H5AD_obs_metadata_values_accessed": True,
    "H5AD_var_metadata_values_accessed": True,
    "H5AD_X_values_accessed": False,
    "H5AD_layer_values_accessed": False,
    "H5AD_raw_expression_values_accessed": False,
    "module_scores_computed": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P4C_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B5_P4C_status.tsv",
    [status],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P4C METADATA-ONLY "
    "H5AD SCHEMA AUDIT =====",
    "",
    (
        "H5AD files opened: "
        f"{opened_files}/6"
    ),
    (
        "obs groups found: "
        f"{obs_groups_found}/6"
    ),
    (
        "var groups found: "
        f"{var_groups_found}/6"
    ),
    (
        "obs schema rows: "
        f"{len(obs_schema_rows)}"
    ),
    (
        "Archive tokens matched in categories/previews: "
        f"{archive_tokens_matched}/13"
    ),
    (
        "Files with sample/section candidate: "
        f"{len(sample_candidate_files)}/6"
    ),
    (
        "Files with annotation candidate: "
        f"{len(annotation_candidate_files)}/6"
    ),
    (
        "Files with cluster candidate: "
        f"{len(cluster_candidate_files)}/6"
    ),
    (
        "Files with layer/depth candidate: "
        f"{len(layer_candidate_files)}/6"
    ),
    (
        "Files with spatial candidate: "
        f"{len(spatial_candidate_files)}/6"
    ),
    "",
    "H5AD structure opened: TRUE",
    "H5AD obs metadata values accessed: TRUE",
    "H5AD var metadata values accessed: TRUE",
    "H5AD X values accessed: FALSE",
    "H5AD layer values accessed: FALSE",
    "H5AD raw expression values accessed: FALSE",
    "Module scores computed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P4C STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P4C_report.txt"
).write_text(
    "\n".join(report)
    + "\n",
    encoding="utf-8",
)

print(
    "\n".join(report)
)

print(
    "\n===== ROLE CANDIDATE SUMMARY ====="
)

for row in candidate_summary_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "candidate_role",
                "files_with_candidate",
                "selected_H5AD_files",
                "all_files_have_candidate",
            )
        )
    )

print(
    "\n===== ARCHIVE-MAPPING AUDIT ====="
)

for row in archive_mapping_audit_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "archive_name",
                "processed_H5AD",
                "matched_source",
                "matched_value",
                "mapping_status",
            )
        )
    )

print(
    "\n===== SELECTED-GENE COVERAGE ====="
)

for row in gene_coverage_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "file_name",
                "gene_source",
                "H5AD_gene_identifiers",
                "prespecified_selected_genes",
                "matched_selected_genes",
                "missing_selected_genes",
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
        != "phase10B5_P4C_SHA256.tsv"
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
    / "phase10B5_P4C_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)
