#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd


PROJECT = Path(
    "."
)

GCTX_FILE = (
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

MATCHED_METADATA_FILE = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6E/"
      "phase6E1B_LINCS_matched_signature_metadata.tsv.gz"
)

PROCESSED_DIR = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6E"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase6"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase6/"
      "phase6E2C_extract_matched_Level5_profiles.log"
)

SUBSET_H5 = (
    PROCESSED_DIR
    / "phase6E2C_matched_Level5_subset.h5"
)

SIGNATURE_METADATA_OUTPUT = (
    PROCESSED_DIR
    / "phase6E2C_matched_signature_metadata.tsv.gz"
)

GENE_METADATA_OUTPUT = (
    PROCESSED_DIR
    / "phase6E2C_LINCS_gene_metadata.tsv.gz"
)

SIGNATURE_QC_OUTPUT = (
    TABLE_DIR
    / "phase6E2C_signature_value_QC.tsv.gz"
)

CHEMICAL_SUMMARY_OUTPUT = (
    TABLE_DIR
    / "phase6E2C_extracted_signature_coverage_by_chemical.tsv"
)

COMPLETION_OUTPUT = (
    TABLE_DIR
    / "phase6E2C_completion_summary.tsv"
)

for directory in [
    PROCESSED_DIR,
    TABLE_DIR,
    LOG_FILE.parent,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


EXPECTED_SIGNATURES = 1232
EXPECTED_GENES = 12328
EXPECTED_LANDMARKS = 978


def clean_identifier(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode(
            "utf-8",
            errors="replace",
        )

    text = str(value).strip()

    if text.endswith(".0"):
        integer_part = text[:-2]

        if integer_part.lstrip("-").isdigit():
            text = integer_part

    return text


def read_hdf5_strings(
    dataset: h5py.Dataset,
) -> list[str]:
    values = np.asarray(
        dataset[()]
    ).reshape(-1)

    return [
        clean_identifier(value)
        for value in values
    ]


def find_column(
    table: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    lookup = {
        str(column).lower(): column
        for column in table.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    if required:
        raise RuntimeError(
            "Could not locate any expected column: "
            + ", ".join(candidates)
        )

    return None


def parse_landmark(
    series: pd.Series,
) -> pd.Series:
    normalized = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {
            "1",
            "1.0",
            "true",
            "t",
            "yes",
            "y",
        }
    )


def calculate_sha256(
    path: Path,
    block_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def main() -> None:
    required_files = [
        GCTX_FILE,
        GENE_INFO_FILE,
        MATCHED_METADATA_FILE,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(missing_files)
        )

    gene_info = pd.read_csv(
        GENE_INFO_FILE,
        sep="\t",
        low_memory=False,
    )

    matched_metadata = pd.read_csv(
        MATCHED_METADATA_FILE,
        sep="\t",
        low_memory=False,
    )

    signature_id_column = find_column(
        matched_metadata,
        [
            "sig_id",
            "signature_id",
        ],
    )

    gene_id_column = find_column(
        gene_info,
        [
            "pr_gene_id",
            "gene_id",
            "id",
        ],
    )

    gene_symbol_column = find_column(
        gene_info,
        [
            "pr_gene_symbol",
            "gene_symbol",
            "symbol",
        ],
    )

    landmark_column = find_column(
        gene_info,
        [
            "pr_is_lm",
            "is_landmark",
            "landmark",
        ],
    )

    if len(matched_metadata) != EXPECTED_SIGNATURES:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_SIGNATURES} metadata rows, "
                f"found {len(matched_metadata)}."
            )
        )

    matched_metadata = matched_metadata.copy()

    matched_metadata[
        "sig_id"
    ] = matched_metadata[
        signature_id_column
    ].map(
        clean_identifier
    )

    if matched_metadata["sig_id"].duplicated().any():
        duplicated = matched_metadata.loc[
            matched_metadata["sig_id"].duplicated(
                keep=False
            ),
            "sig_id",
        ]

        raise RuntimeError(
            (
                "Matched metadata contains duplicated signature IDs: "
                + "|".join(
                    duplicated.astype(str).head(20)
                )
            )
        )

    gene_metadata = pd.DataFrame(
        {
            "gene_id":
                gene_info[
                    gene_id_column
                ].map(
                    clean_identifier
                ),

            "gene_symbol":
                gene_info[
                    gene_symbol_column
                ]
                .fillna("")
                .astype(str)
                .str.strip(),

            "is_landmark":
                parse_landmark(
                    gene_info[
                        landmark_column
                    ]
                ),
        }
    )

    if len(gene_metadata) != EXPECTED_GENES:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_GENES} gene rows, "
                f"found {len(gene_metadata)}."
            )
        )

    if gene_metadata["gene_id"].duplicated().any():
        raise RuntimeError(
            "gene_info contains duplicated gene IDs."
        )

    landmark_count = int(
        gene_metadata["is_landmark"].sum()
    )

    if landmark_count != EXPECTED_LANDMARKS:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_LANDMARKS} landmarks, "
                f"found {landmark_count}."
            )
        )

    with h5py.File(
        GCTX_FILE,
        mode="r",
    ) as source:
        matrix = source[
            "/0/DATA/0/matrix"
        ]

        row_ids = read_hdf5_strings(
            source[
                "/0/META/ROW/id"
            ]
        )

        column_ids = read_hdf5_strings(
            source[
                "/0/META/COL/id"
            ]
        )

        if tuple(matrix.shape) != (
            len(column_ids),
            len(row_ids),
        ):
            raise RuntimeError(
                (
                    "Expected signatures × genes matrix orientation, "
                    f"found matrix shape {matrix.shape}, "
                    f"{len(column_ids)} COL IDs and "
                    f"{len(row_ids)} ROW IDs."
                )
            )

        if row_ids != gene_metadata[
            "gene_id"
        ].tolist():
            raise RuntimeError(
                "GCTX gene order does not match gene_info."
            )

        signature_index = {
            signature_id: index
            for index, signature_id
            in enumerate(column_ids)
        }

        missing_signature_ids = [
            signature_id
            for signature_id
            in matched_metadata["sig_id"]
            if signature_id not in signature_index
        ]

        if missing_signature_ids:
            raise RuntimeError(
                (
                    "Matched signatures absent from GCTX: "
                    + "|".join(
                        missing_signature_ids[:20]
                    )
                )
            )

        matched_metadata[
            "GCTX_signature_index"
        ] = matched_metadata[
            "sig_id"
        ].map(
            signature_index
        ).astype(
            int
        )

        matched_metadata = matched_metadata.sort_values(
            [
                "GCTX_signature_index",
                "sig_id",
            ]
        ).reset_index(
            drop=True
        )

        matched_metadata[
            "subset_matrix_row"
        ] = np.arange(
            len(matched_metadata),
            dtype=int,
        )

        selected_positions = matched_metadata[
            "GCTX_signature_index"
        ].to_numpy(
            dtype=int
        )

        if not np.all(
            np.diff(selected_positions) > 0
        ):
            raise RuntimeError(
                "Selected GCTX positions are not strictly increasing."
            )

        if SUBSET_H5.exists():
            SUBSET_H5.unlink()

        string_dtype = h5py.string_dtype(
            encoding="utf-8"
        )

        qc_rows: list[dict[str, object]] = []

        with h5py.File(
            SUBSET_H5,
            mode="w",
        ) as destination:
            data_dataset = destination.create_dataset(
                "matrix",
                shape=(
                    EXPECTED_SIGNATURES,
                    EXPECTED_GENES,
                ),
                dtype="float32",
                chunks=(
                    32,
                    1024,
                ),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            destination.create_dataset(
                "sig_id",
                data=np.asarray(
                    matched_metadata[
                        "sig_id"
                    ].tolist(),
                    dtype=object,
                ),
                dtype=string_dtype,
            )

            destination.create_dataset(
                "gene_id",
                data=np.asarray(
                    gene_metadata[
                        "gene_id"
                    ].tolist(),
                    dtype=object,
                ),
                dtype=string_dtype,
            )

            destination.create_dataset(
                "gene_symbol",
                data=np.asarray(
                    gene_metadata[
                        "gene_symbol"
                    ].tolist(),
                    dtype=object,
                ),
                dtype=string_dtype,
            )

            destination.create_dataset(
                "is_landmark",
                data=gene_metadata[
                    "is_landmark"
                ].astype(
                    np.uint8
                ).to_numpy(),
                dtype="uint8",
            )

            destination.attrs[
                "source_GCTX"
            ] = GCTX_FILE.name

            destination.attrs[
                "matrix_orientation"
            ] = "signatures_by_genes"

            destination.attrs[
                "signature_count"
            ] = EXPECTED_SIGNATURES

            destination.attrs[
                "gene_count"
            ] = EXPECTED_GENES

            destination.attrs[
                "landmark_gene_count"
            ] = landmark_count

            batch_size = 32

            for start in range(
                0,
                EXPECTED_SIGNATURES,
                batch_size,
            ):
                end = min(
                    start + batch_size,
                    EXPECTED_SIGNATURES,
                )

                batch_positions = selected_positions[
                    start:end
                ]

                block = np.asarray(
                    matrix[
                        batch_positions,
                        :,
                    ],
                    dtype=np.float32,
                )

                expected_shape = (
                    end - start,
                    EXPECTED_GENES,
                )

                if block.shape != expected_shape:
                    raise RuntimeError(
                        (
                            f"Unexpected extraction block shape "
                            f"{block.shape}; expected {expected_shape}."
                        )
                    )

                finite_mask = np.isfinite(
                    block
                )

                if not finite_mask.all():
                    raise RuntimeError(
                        (
                            "Non-finite values detected in extracted "
                            f"rows {start}:{end}."
                        )
                    )

                data_dataset[
                    start:end,
                    :,
                ] = block

                for local_index in range(
                    block.shape[0]
                ):
                    values = block[
                        local_index,
                        :,
                    ]

                    metadata_row = matched_metadata.iloc[
                        start + local_index
                    ]

                    qc_rows.append(
                        {
                            "subset_matrix_row":
                                int(
                                    start
                                    + local_index
                                ),

                            "GCTX_signature_index":
                                int(
                                    metadata_row[
                                        "GCTX_signature_index"
                                    ]
                                ),

                            "sig_id":
                                metadata_row[
                                    "sig_id"
                                ],

                            "chemical_query":
                                metadata_row.get(
                                    "chemical_query",
                                    "",
                                ),

                            "cell_id":
                                metadata_row.get(
                                    "cell_id",
                                    "",
                                ),

                            "finite_value_count":
                                int(
                                    np.isfinite(
                                        values
                                    ).sum()
                                ),

                            "minimum":
                                float(
                                    values.min()
                                ),

                            "maximum":
                                float(
                                    values.max()
                                ),

                            "mean":
                                float(
                                    values.mean()
                                ),

                            "standard_deviation":
                                float(
                                    values.std(
                                        ddof=0
                                    )
                                ),

                            "median":
                                float(
                                    np.median(
                                        values
                                    )
                                ),

                            "zero_value_count":
                                int(
                                    np.count_nonzero(
                                        values == 0
                                    )
                                ),

                            "absolute_value_ge_9_99_count":
                                int(
                                    np.count_nonzero(
                                        np.abs(
                                            values
                                        ) >= 9.99
                                    )
                                ),
                        }
                    )

                print(
                    (
                        f"Extracted signatures "
                        f"{start + 1}-{end} "
                        f"of {EXPECTED_SIGNATURES}"
                    ),
                    flush=True,
                )

    signature_qc = pd.DataFrame(
        qc_rows
    )

    if len(signature_qc) != EXPECTED_SIGNATURES:
        raise RuntimeError(
            (
                "Unexpected signature QC row count: "
                f"{len(signature_qc)}"
            )
        )

    if not signature_qc[
        "finite_value_count"
    ].eq(
        EXPECTED_GENES
    ).all():
        raise RuntimeError(
            "One or more extracted signatures contain non-finite values."
        )

    matched_metadata.to_csv(
        SIGNATURE_METADATA_OUTPUT,
        sep="\t",
        index=False,
        compression="gzip",
    )

    gene_metadata.to_csv(
        GENE_METADATA_OUTPUT,
        sep="\t",
        index=False,
        compression="gzip",
    )

    signature_qc.to_csv(
        SIGNATURE_QC_OUTPUT,
        sep="\t",
        index=False,
        compression="gzip",
    )

    chemical_column = find_column(
        matched_metadata,
        [
            "chemical_query",
            "pert_iname",
        ],
    )

    cell_column = find_column(
        matched_metadata,
        [
            "cell_id",
            "cell",
        ],
        required=False,
    )

    dose_column = find_column(
        matched_metadata,
        [
            "pert_idose",
            "_dose_label",
        ],
        required=False,
    )

    time_column = find_column(
        matched_metadata,
        [
            "pert_itime",
            "_time_label",
        ],
        required=False,
    )

    chemical_summary_rows: list[dict[str, object]] = []

    for chemical, group in matched_metadata.groupby(
        chemical_column,
        sort=True,
    ):
        chemical_summary_rows.append(
            {
                "chemical_query":
                    chemical,

                "extracted_signature_count":
                    len(group),

                "unique_cell_count":
                    (
                        group[
                            cell_column
                        ].nunique(
                            dropna=True
                        )
                        if cell_column
                        else 0
                    ),

                "unique_dose_count":
                    (
                        group[
                            dose_column
                        ].nunique(
                            dropna=True
                        )
                        if dose_column
                        else 0
                    ),

                "unique_time_count":
                    (
                        group[
                            time_column
                        ].nunique(
                            dropna=True
                        )
                        if time_column
                        else 0
                    ),

                "minimum_signature_mean":
                    float(
                        signature_qc.loc[
                            signature_qc[
                                "chemical_query"
                            ].eq(
                                chemical
                            ),
                            "mean",
                        ].min()
                    ),

                "maximum_signature_mean":
                    float(
                        signature_qc.loc[
                            signature_qc[
                                "chemical_query"
                            ].eq(
                                chemical
                            ),
                            "mean",
                        ].max()
                    ),

                "median_signature_standard_deviation":
                    float(
                        signature_qc.loc[
                            signature_qc[
                                "chemical_query"
                            ].eq(
                                chemical
                            ),
                            "standard_deviation",
                        ].median()
                    ),
            }
        )

    chemical_summary = pd.DataFrame(
        chemical_summary_rows
    ).sort_values(
        [
            "extracted_signature_count",
            "chemical_query",
        ],
        ascending=[
            False,
            True,
        ],
    )

    chemical_summary.to_csv(
        CHEMICAL_SUMMARY_OUTPUT,
        sep="\t",
        index=False,
    )

    subset_sha256 = calculate_sha256(
        SUBSET_H5
    )

    with h5py.File(
        SUBSET_H5,
        mode="r",
    ) as subset:
        subset_shape = tuple(
            int(value)
            for value in subset[
                "matrix"
            ].shape
        )

        stored_signature_ids = read_hdf5_strings(
            subset[
                "sig_id"
            ]
        )

        stored_gene_ids = read_hdf5_strings(
            subset[
                "gene_id"
            ]
        )

        stored_landmark_count = int(
            np.asarray(
                subset[
                    "is_landmark"
                ][()]
            ).sum()
        )

    signature_order_valid = (
        stored_signature_ids
        == matched_metadata[
            "sig_id"
        ].tolist()
    )

    gene_order_valid = (
        stored_gene_ids
        == gene_metadata[
            "gene_id"
        ].tolist()
    )

    status = (
        "completed"
        if (
            subset_shape
            == (
                EXPECTED_SIGNATURES,
                EXPECTED_GENES,
            )
            and signature_order_valid
            and gene_order_valid
            and stored_landmark_count
            == EXPECTED_LANDMARKS
            and signature_qc[
                "finite_value_count"
            ].eq(
                EXPECTED_GENES
            ).all()
        )
        else "failed_validation"
    )

    completion = pd.DataFrame(
        [
            {
                "source_GCTX_signatures":
                    118050,

                "source_GCTX_genes":
                    EXPECTED_GENES,

                "matched_signatures_expected":
                    EXPECTED_SIGNATURES,

                "matched_signatures_extracted":
                    subset_shape[0],

                "genes_expected":
                    EXPECTED_GENES,

                "genes_extracted":
                    subset_shape[1],

                "landmark_genes_expected":
                    EXPECTED_LANDMARKS,

                "landmark_genes_extracted":
                    stored_landmark_count,

                "unique_matched_chemicals":
                    matched_metadata[
                        chemical_column
                    ].nunique(),

                "all_extracted_values_finite":
                    bool(
                        signature_qc[
                            "finite_value_count"
                        ].eq(
                            EXPECTED_GENES
                        ).all()
                    ),

                "signature_ID_order_valid":
                    signature_order_valid,

                "gene_ID_order_valid":
                    gene_order_valid,

                "subset_HDF5_bytes":
                    SUBSET_H5.stat().st_size,

                "subset_HDF5_SHA256":
                    subset_sha256,

                "full_GCTX_retained":
                    GCTX_FILE.exists(),

                "Phase6E2C_status":
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
            "===== PHASE 6E2C COMPLETION =====",
            completion.to_string(
                index=False
            ),
            "",
            "===== CHEMICAL EXTRACTION SUMMARY =====",
            chemical_summary.to_string(
                index=False
            ),
            "",
            "===== SIGNATURE VALUE QC SUMMARY =====",
            signature_qc[
                [
                    "minimum",
                    "maximum",
                    "mean",
                    "standard_deviation",
                    "median",
                    "zero_value_count",
                    "absolute_value_ge_9_99_count",
                ]
            ].describe().to_string(),
        ]
    )

    LOG_FILE.write_text(
        log_text + "\n",
        encoding="utf-8",
    )

    print()
    print(
        log_text
    )

    if status != "completed":
        raise RuntimeError(
            "Phase 6E2C validation failed."
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            (
                "Phase 6E2C failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
