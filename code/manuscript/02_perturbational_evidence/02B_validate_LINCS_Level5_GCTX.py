#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd


PROJECT = Path(
    "."
)

MATRIX_FILE = (
    PROJECT
    / "01_raw_data/perturbational_validation/LINCS_L1000/"
      "GSE70138/matrix/"
      "GSE70138_Broad_LINCS_Level5_COMPZ_"
      "n118050x12328_2017-03-06.gctx"
)

GENE_INFO_FILE = (
    PROJECT
    / "01_raw_data/perturbational_validation/LINCS_L1000/"
      "GSE70138/metadata/"
      "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz"
)

SIG_INFO_FILE = (
    PROJECT
    / "01_raw_data/perturbational_validation/LINCS_L1000/"
      "GSE70138/metadata/"
      "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz"
)

MATCHED_SIGNATURE_FILE = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6E/"
      "phase6E1B_LINCS_matched_signature_metadata.tsv.gz"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase6"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase6/"
      "phase6E2B_validate_Level5_GCTX.log"
)

SCHEMA_OUTPUT = (
    TABLE_DIR
    / "phase6E2B_Level5_GCTX_schema.tsv"
)

DATASET_INVENTORY_OUTPUT = (
    TABLE_DIR
    / "phase6E2B_Level5_GCTX_dataset_inventory.tsv"
)

SAMPLE_STATISTICS_OUTPUT = (
    TABLE_DIR
    / "phase6E2B_Level5_GCTX_sample_statistics.tsv"
)

ID_ALIGNMENT_OUTPUT = (
    TABLE_DIR
    / "phase6E2B_Level5_GCTX_ID_alignment.tsv"
)

COMPLETION_OUTPUT = (
    TABLE_DIR
    / "phase6E2B_completion_summary.tsv"
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


EXPECTED_GENE_COUNT = 12328
EXPECTED_SIGNATURE_COUNT = 118050
EXPECTED_MATCHED_SIGNATURE_COUNT = 1232


def clean_identifier(
    value: object,
) -> str:
    if isinstance(
        value,
        bytes,
    ):
        value = value.decode(
            "utf-8",
            errors="replace",
        )

    text = str(
        value
    ).strip()

    if text.endswith(
        ".0"
    ):
        integer_part = text[
            :-2
        ]

        if integer_part.lstrip(
            "-"
        ).isdigit():
            text = integer_part

    return text


def read_identifier_dataset(
    dataset: h5py.Dataset,
) -> list[str]:
    values = np.asarray(
        dataset[()]
    ).reshape(
        -1
    )

    return [
        clean_identifier(
            value
        )
        for value in values
    ]


def find_column(
    table: pd.DataFrame,
    candidates: list[str],
) -> str:
    lookup = {
        str(column).lower(): column
        for column in table.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[
                candidate.lower()
            ]

    raise RuntimeError(
        "Could not locate any expected column: "
        + ", ".join(
            candidates
        )
    )


def locate_matrix_dataset(
    handle: h5py.File,
    row_count: int,
    column_count: int,
) -> tuple[str, h5py.Dataset]:
    known_paths = [
        "/0/DATA/0/matrix",
        "0/DATA/0/matrix",
    ]

    for path in known_paths:
        if path in handle:
            dataset = handle[
                path
            ]

            if (
                isinstance(
                    dataset,
                    h5py.Dataset,
                )
                and dataset.ndim == 2
            ):
                return (
                    dataset.name,
                    dataset,
                )

    candidate_paths: list[str] = []

    def visitor(
        name: str,
        object_: object,
    ) -> None:
        if not isinstance(
            object_,
            h5py.Dataset,
        ):
            return

        if object_.ndim != 2:
            return

        shape = tuple(
            int(value)
            for value in object_.shape
        )

        valid_shapes = {
            (
                row_count,
                column_count,
            ),
            (
                column_count,
                row_count,
            ),
        }

        if shape in valid_shapes:
            candidate_paths.append(
                object_.name
            )

    handle.visititems(
        visitor
    )

    if len(
        candidate_paths
    ) != 1:
        raise RuntimeError(
            "Expected one matrix dataset, found: "
            + repr(
                candidate_paths
            )
        )

    path = candidate_paths[
        0
    ]

    return (
        path,
        handle[path],
    )


def locate_identifier_dataset(
    handle: h5py.File,
    axis_name: str,
) -> tuple[str, h5py.Dataset]:
    axis_name = axis_name.upper()

    known_paths = [
        f"/0/META/{axis_name}/id",
        f"0/META/{axis_name}/id",
    ]

    for path in known_paths:
        if path in handle:
            dataset = handle[
                path
            ]

            if isinstance(
                dataset,
                h5py.Dataset,
            ):
                return (
                    dataset.name,
                    dataset,
                )

    candidates: list[str] = []

    def visitor(
        name: str,
        object_: object,
    ) -> None:
        if not isinstance(
            object_,
            h5py.Dataset,
        ):
            return

        normalized_name = (
            "/"
            + name.strip(
                "/"
            ).upper()
            + "/"
        )

        leaf_name = name.split(
            "/"
        )[-1].lower()

        if (
            f"/META/{axis_name}/"
            in normalized_name
            and leaf_name
            in {
                "id",
                "rid",
                "cid",
            }
        ):
            candidates.append(
                object_.name
            )

    handle.visititems(
        visitor
    )

    if len(
        candidates
    ) != 1:
        raise RuntimeError(
            (
                f"Expected one {axis_name} identifier dataset, "
                f"found: {candidates}"
            )
        )

    path = candidates[
        0
    ]

    return (
        path,
        handle[path],
    )


def dataset_inventory(
    handle: h5py.File,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def visitor(
        name: str,
        object_: object,
    ) -> None:
        if not isinstance(
            object_,
            h5py.Dataset,
        ):
            return

        rows.append(
            {
                "dataset_path":
                    object_.name,

                "dimensions":
                    "x".join(
                        str(value)
                        for value in object_.shape
                    ),

                "ndim":
                    object_.ndim,

                "dtype":
                    str(
                        object_.dtype
                    ),

                "chunk_shape":
                    (
                        "x".join(
                            str(value)
                            for value in object_.chunks
                        )
                        if object_.chunks
                        else ""
                    ),

                "internal_compression":
                    (
                        clean_identifier(
                            object_.compression
                        )
                        if object_.compression
                        else ""
                    ),
            }
        )

    handle.visititems(
        visitor
    )

    return pd.DataFrame(
        rows
    ).sort_values(
        "dataset_path"
    )


def sample_matrix_blocks(
    matrix: h5py.Dataset,
) -> pd.DataFrame:
    row_count = int(
        matrix.shape[
            0
        ]
    )

    column_count = int(
        matrix.shape[
            1
        ]
    )

    block_size = 64

    row_starts = [
        0,
        max(
            0,
            row_count // 2
            - block_size // 2,
        ),
        max(
            0,
            row_count
            - block_size,
        ),
    ]

    column_starts = [
        0,
        max(
            0,
            column_count // 2
            - block_size // 2,
        ),
        max(
            0,
            column_count
            - block_size,
        ),
    ]

    labels = [
        "start",
        "middle",
        "end",
    ]

    rows: list[dict[str, object]] = []

    for label, row_start, column_start in zip(
        labels,
        row_starts,
        column_starts,
    ):
        block = np.asarray(
            matrix[
                row_start:
                    min(
                        row_start
                        + block_size,
                        row_count,
                    ),
                column_start:
                    min(
                        column_start
                        + block_size,
                        column_count,
                    ),
            ],
            dtype=float,
        )

        finite_mask = np.isfinite(
            block
        )

        finite_values = block[
            finite_mask
        ]

        rows.append(
            {
                "sample_block":
                    label,

                "row_start":
                    row_start,

                "column_start":
                    column_start,

                "sampled_values":
                    block.size,

                "finite_values":
                    int(
                        finite_mask.sum()
                    ),

                "finite_fraction":
                    float(
                        finite_mask.mean()
                    ),

                "minimum":
                    (
                        float(
                            finite_values.min()
                        )
                        if finite_values.size
                        else np.nan
                    ),

                "maximum":
                    (
                        float(
                            finite_values.max()
                        )
                        if finite_values.size
                        else np.nan
                    ),

                "mean":
                    (
                        float(
                            finite_values.mean()
                        )
                        if finite_values.size
                        else np.nan
                    ),

                "standard_deviation":
                    (
                        float(
                            finite_values.std()
                        )
                        if finite_values.size
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    required_files = [
        MATRIX_FILE,
        GENE_INFO_FILE,
        SIG_INFO_FILE,
        MATCHED_SIGNATURE_FILE,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(
                missing_files
            )
        )

    gene_info = pd.read_csv(
        GENE_INFO_FILE,
        sep="\t",
        low_memory=False,
    )

    sig_info = pd.read_csv(
        SIG_INFO_FILE,
        sep="\t",
        low_memory=False,
    )

    matched_signatures = pd.read_csv(
        MATCHED_SIGNATURE_FILE,
        sep="\t",
        low_memory=False,
    )

    gene_id_column = find_column(
        gene_info,
        [
            "pr_gene_id",
            "gene_id",
            "id",
        ],
    )

    signature_id_column = find_column(
        sig_info,
        [
            "sig_id",
            "signature_id",
            "id",
        ],
    )

    matched_signature_id_column = find_column(
        matched_signatures,
        [
            "sig_id",
            "signature_id",
        ],
    )

    metadata_gene_ids = [
        clean_identifier(
            value
        )
        for value in gene_info[
            gene_id_column
        ]
    ]

    metadata_signature_ids = [
        clean_identifier(
            value
        )
        for value in sig_info[
            signature_id_column
        ]
    ]

    matched_signature_ids = {
        clean_identifier(
            value
        )
        for value in matched_signatures[
            matched_signature_id_column
        ]
    }

    if len(
        gene_info
    ) != EXPECTED_GENE_COUNT:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_GENE_COUNT} gene rows, "
                f"found {len(gene_info)}."
            )
        )

    if len(
        sig_info
    ) != EXPECTED_SIGNATURE_COUNT:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_SIGNATURE_COUNT} signature rows, "
                f"found {len(sig_info)}."
            )
        )

    if len(
        matched_signature_ids
    ) != EXPECTED_MATCHED_SIGNATURE_COUNT:
        raise RuntimeError(
            (
                "Expected "
                f"{EXPECTED_MATCHED_SIGNATURE_COUNT} matched signatures, "
                f"found {len(matched_signature_ids)}."
            )
        )

    with h5py.File(
        MATRIX_FILE,
        mode="r",
    ) as handle:
        inventory = dataset_inventory(
            handle
        )

        row_id_path, row_id_dataset = (
            locate_identifier_dataset(
                handle,
                "ROW",
            )
        )

        column_id_path, column_id_dataset = (
            locate_identifier_dataset(
                handle,
                "COL",
            )
        )

        row_ids = read_identifier_dataset(
            row_id_dataset
        )

        column_ids = read_identifier_dataset(
            column_id_dataset
        )

        matrix_path, matrix = locate_matrix_dataset(
            handle,
            row_count=len(
                row_ids
            ),
            column_count=len(
                column_ids
            ),
        )

        matrix_shape = tuple(
            int(value)
            for value in matrix.shape
        )

        if matrix_shape == (
            len(row_ids),
            len(column_ids),
        ):
            matrix_orientation = (
                "ROW_by_COL"
            )

            matrix_gene_axis = 0
            matrix_signature_axis = 1

        elif matrix_shape == (
            len(column_ids),
            len(row_ids),
        ):
            matrix_orientation = (
                "COL_by_ROW"
            )

            matrix_gene_axis = 1
            matrix_signature_axis = 0

        else:
            raise RuntimeError(
                (
                    "Matrix dimensions do not agree with "
                    "ROW and COL identifier counts."
                )
            )

        sample_statistics = sample_matrix_blocks(
            matrix
        )

        schema = pd.DataFrame(
            [
                {
                    "GCTX_filename":
                        MATRIX_FILE.name,

                    "GCTX_bytes":
                        MATRIX_FILE.stat().st_size,

                    "matrix_dataset_path":
                        matrix_path,

                    "matrix_dimensions":
                        "x".join(
                            str(value)
                            for value in matrix_shape
                        ),

                    "matrix_dtype":
                        str(
                            matrix.dtype
                        ),

                    "matrix_chunk_shape":
                        (
                            "x".join(
                                str(value)
                                for value in matrix.chunks
                            )
                            if matrix.chunks
                            else ""
                        ),

                    "matrix_internal_compression":
                        (
                            clean_identifier(
                                matrix.compression
                            )
                            if matrix.compression
                            else ""
                        ),

                    "matrix_orientation":
                        matrix_orientation,

                    "matrix_gene_axis":
                        matrix_gene_axis,

                    "matrix_signature_axis":
                        matrix_signature_axis,

                    "ROW_identifier_path":
                        row_id_path,

                    "ROW_identifier_count":
                        len(row_ids),

                    "COL_identifier_path":
                        column_id_path,

                    "COL_identifier_count":
                        len(column_ids),
                }
            ]
        )

    inventory.to_csv(
        DATASET_INVENTORY_OUTPUT,
        sep="\t",
        index=False,
    )

    sample_statistics.to_csv(
        SAMPLE_STATISTICS_OUTPUT,
        sep="\t",
        index=False,
    )

    schema.to_csv(
        SCHEMA_OUTPUT,
        sep="\t",
        index=False,
    )

    row_id_set = set(
        row_ids
    )

    column_id_set = set(
        column_ids
    )

    metadata_gene_id_set = set(
        metadata_gene_ids
    )

    metadata_signature_id_set = set(
        metadata_signature_ids
    )

    row_gene_set_match = (
        row_id_set
        == metadata_gene_id_set
    )

    column_signature_set_match = (
        column_id_set
        == metadata_signature_id_set
    )

    row_gene_order_match = (
        row_ids
        == metadata_gene_ids
    )

    column_signature_order_match = (
        column_ids
        == metadata_signature_ids
    )

    matched_signatures_present = (
        matched_signature_ids.issubset(
            column_id_set
        )
    )

    id_alignment = pd.DataFrame(
        [
            {
                "identifier_type":
                    "GCTX_ROW_vs_gene_info",

                "GCTX_ID_count":
                    len(row_ids),

                "metadata_ID_count":
                    len(metadata_gene_ids),

                "GCTX_unique_ID_count":
                    len(row_id_set),

                "metadata_unique_ID_count":
                    len(metadata_gene_id_set),

                "exact_set_match":
                    row_gene_set_match,

                "exact_order_match":
                    row_gene_order_match,

                "GCTX_only_IDs":
                    len(
                        row_id_set
                        - metadata_gene_id_set
                    ),

                "metadata_only_IDs":
                    len(
                        metadata_gene_id_set
                        - row_id_set
                    ),
            },
            {
                "identifier_type":
                    "GCTX_COL_vs_sig_info",

                "GCTX_ID_count":
                    len(column_ids),

                "metadata_ID_count":
                    len(metadata_signature_ids),

                "GCTX_unique_ID_count":
                    len(column_id_set),

                "metadata_unique_ID_count":
                    len(metadata_signature_id_set),

                "exact_set_match":
                    column_signature_set_match,

                "exact_order_match":
                    column_signature_order_match,

                "GCTX_only_IDs":
                    len(
                        column_id_set
                        - metadata_signature_id_set
                    ),

                "metadata_only_IDs":
                    len(
                        metadata_signature_id_set
                        - column_id_set
                    ),
            },
            {
                "identifier_type":
                    "matched_Phase6E1B_signatures",

                "GCTX_ID_count":
                    len(column_ids),

                "metadata_ID_count":
                    len(matched_signature_ids),

                "GCTX_unique_ID_count":
                    len(column_id_set),

                "metadata_unique_ID_count":
                    len(matched_signature_ids),

                "exact_set_match":
                    matched_signatures_present,

                "exact_order_match":
                    False,

                "GCTX_only_IDs":
                    len(
                        column_id_set
                        - matched_signature_ids
                    ),

                "metadata_only_IDs":
                    len(
                        matched_signature_ids
                        - column_id_set
                    ),
            },
        ]
    )

    id_alignment.to_csv(
        ID_ALIGNMENT_OUTPUT,
        sep="\t",
        index=False,
    )

    all_sampled_values_finite = bool(
        sample_statistics[
            "finite_fraction"
        ].eq(
            1.0
        ).all()
    )

    matrix_shape_valid = (
        schema[
            "matrix_dimensions"
        ].iloc[0]
        in {
            "12328x118050",
            "118050x12328",
        }
    )

    required_alignment_valid = bool(
        row_gene_set_match
        and column_signature_set_match
        and matched_signatures_present
    )

    status = (
        "completed"
        if (
            matrix_shape_valid
            and required_alignment_valid
            and all_sampled_values_finite
        )
        else "failed_validation"
    )

    completion = pd.DataFrame(
        [
            {
                "GCTX_file_exists":
                    MATRIX_FILE.exists(),

                "GCTX_file_bytes":
                    MATRIX_FILE.stat().st_size,

                "HDF5_open_successful":
                    True,

                "matrix_dataset_count":
                    1,

                "matrix_dimensions":
                    schema[
                        "matrix_dimensions"
                    ].iloc[0],

                "expected_gene_dimension":
                    EXPECTED_GENE_COUNT,

                "expected_signature_dimension":
                    EXPECTED_SIGNATURE_COUNT,

                "matrix_dimensions_valid":
                    matrix_shape_valid,

                "GCTX_ROW_ID_count":
                    len(row_ids),

                "GCTX_COL_ID_count":
                    len(column_ids),

                "ROW_gene_ID_exact_set_match":
                    row_gene_set_match,

                "COL_signature_ID_exact_set_match":
                    column_signature_set_match,

                "matched_signature_IDs_expected":
                    EXPECTED_MATCHED_SIGNATURE_COUNT,

                "matched_signature_IDs_present":
                    len(
                        matched_signature_ids
                        & column_id_set
                    ),

                "all_matched_signature_IDs_present":
                    matched_signatures_present,

                "sampled_matrix_values_all_finite":
                    all_sampled_values_finite,

                "compressed_archive_retained":
                    MATRIX_FILE.with_suffix(
                        MATRIX_FILE.suffix
                        + ".gz"
                    ).exists(),

                "Phase6E2B_status":
                    status,
            }
        ]
    )

    completion.to_csv(
        COMPLETION_OUTPUT,
        sep="\t",
        index=False,
    )

    log_text = "\n".join(
        [
            "===== PHASE 6E2B GCTX SCHEMA =====",
            schema.to_string(
                index=False
            ),
            "",
            "===== IDENTIFIER ALIGNMENT =====",
            id_alignment.to_string(
                index=False
            ),
            "",
            "===== MATRIX SAMPLE STATISTICS =====",
            sample_statistics.to_string(
                index=False
            ),
            "",
            "===== COMPLETION =====",
            completion.to_string(
                index=False
            ),
        ]
    )

    LOG_FILE.write_text(
        log_text + "\n",
        encoding="utf-8",
    )

    print(
        log_text
    )

    if status != "completed":
        raise RuntimeError(
            "GCTX structural validation failed."
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            (
                "Phase 6E2B failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
