#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


PROJECT = Path(
    "."
)

SUBSET_H5 = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6E/"
      "phase6E2C_matched_Level5_subset.h5"
)

SIGNATURE_METADATA_FILE = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6E/"
      "phase6E2C_matched_signature_metadata.tsv.gz"
)

GENE_METADATA_FILE = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6E/"
      "phase6E2C_LINCS_gene_metadata.tsv.gz"
)

MATURATION_FILE = (
    PROJECT
    / "03_processed_data/developmental_trajectory/phase5D/enrichment/"
      "phase5D4_maturation_high_increasing_genes.tsv"
)

FETAL_FILE = (
    PROJECT
    / "03_processed_data/developmental_trajectory/phase5D/enrichment/"
      "phase5D4_fetal_high_decreasing_genes.tsv"
)

MECHANISM_FILE = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6C/"
      "phase6C3C_mechanism_category_gene_unions.tsv.gz"
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
      "phase6E3A_quantitative_LINCS_scoring.log"
)

SIGNATURE_SCORE_OUTPUT = (
    PROCESSED_DIR
    / "phase6E3A_quantitative_signature_scores.tsv.gz"
)

GENE_SET_MAPPING_OUTPUT = (
    TABLE_DIR
    / "phase6E3A_gene_set_mapping.tsv"
)

CHEMICAL_SUMMARY_OUTPUT = (
    TABLE_DIR
    / "phase6E3A_quantitative_score_summary_by_chemical.tsv"
)

CHEMICAL_CELLCLASS_OUTPUT = (
    TABLE_DIR
    / "phase6E3A_quantitative_score_summary_by_chemical_cellclass.tsv"
)

QUALITY_METRIC_OUTPUT = (
    TABLE_DIR
    / "phase6E3A_quality_metric_distribution.tsv"
)

CONCORDANCE_OUTPUT = (
    TABLE_DIR
    / "phase6E3A_full_vs_landmark_score_concordance.tsv"
)

COMPLETION_OUTPUT = (
    TABLE_DIR
    / "phase6E3A_completion_summary.tsv"
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
EXPECTED_CHEMICALS = 15


EARLY_CATEGORIES = [
    "early_proliferative_state",
    "fetal_neurodevelopmental_state",
]

LATE_CATEGORIES = [
    "late_synaptic_maturation",
    "mitochondrial_metabolic_maturation",
    "glial_myelin_maturation",
]

INJURY_CATEGORIES = [
    "DNA_damage_p53",
    "apoptosis_cytotoxicity",
    "oxidative_hypoxic_stress",
    "inflammatory_stress",
]

ALL_MECHANISM_CATEGORIES = (
    EARLY_CATEGORIES
    + LATE_CATEGORIES
    + INJURY_CATEGORIES
)


EXPECTED_GENE_SET_COUNTS = {
    "maturation_high_increasing": {
        "input": 3333,
        "mapped": 1834,
        "landmark": 164,
    },

    "fetal_high_decreasing": {
        "input": 5412,
        "mapped": 2771,
        "landmark": 291,
    },

    "DNA_damage_p53": {
        "input": 340,
        "mapped": 322,
        "landmark": 58,
    },

    "apoptosis_cytotoxicity": {
        "input": 161,
        "mapped": 157,
        "landmark": 40,
    },

    "early_proliferative_state": {
        "input": 637,
        "mapped": 600,
        "landmark": 102,
    },

    "fetal_neurodevelopmental_state": {
        "input": 480,
        "mapped": 401,
        "landmark": 59,
    },

    "glial_myelin_maturation": {
        "input": 290,
        "mapped": 258,
        "landmark": 36,
    },

    "inflammatory_stress": {
        "input": 349,
        "mapped": 338,
        "landmark": 46,
    },

    "late_synaptic_maturation": {
        "input": 493,
        "mapped": 405,
        "landmark": 28,
    },

    "mitochondrial_metabolic_maturation": {
        "input": 240,
        "mapped": 219,
        "landmark": 26,
    },

    "oxidative_hypoxic_stress": {
        "input": 233,
        "mapped": 222,
        "landmark": 49,
    },
}


def decode_strings(
    values: Iterable[object],
) -> list[str]:
    decoded: list[str] = []

    for value in values:
        if isinstance(value, bytes):
            value = value.decode(
                "utf-8",
                errors="replace",
            )

        decoded.append(
            str(value).strip()
        )

    return decoded


def clean_text(
    value: object,
) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "na",
    }:
        return ""

    return text


def normalize_gene(
    value: object,
) -> str:
    return clean_text(value).upper()


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


def read_gene_set(
    path: Path,
) -> set[str]:
    table = pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
    )

    gene_column = find_column(
        table,
        [
            "gene_symbol",
            "gene",
            "symbol",
            "external_gene_name",
        ],
        required=False,
    )

    if gene_column is None:
        gene_column = table.columns[0]

    return {
        normalize_gene(value)
        for value in table[gene_column]
        if normalize_gene(value)
    }


def classify_cell(
    value: object,
) -> str:
    cell = clean_text(value).upper()

    neural_lineage = {
        "NPC",
        "NPC.CAS9",
        "NPC.TAK",
        "NEU",
        "MNEU.E",
    }

    pluripotent = {
        "HUES3",
    }

    hematopoietic = {
        "CD34",
        "JURKAT",
        "SKL",
        "SKL.C",
    }

    if cell in neural_lineage:
        return "neural_lineage"

    if cell in pluripotent:
        return "pluripotent_stem"

    if cell in hematopoietic:
        return "hematopoietic"

    return "other_non_neural"


def score_gene_indices(
    matrix: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    if indices.size == 0:
        return np.full(
            matrix.shape[0],
            np.nan,
            dtype=float,
        )

    return matrix[
        :,
        indices,
    ].mean(
        axis=1,
        dtype=np.float64,
    )


def summarize_values(
    values: pd.Series,
) -> dict[str, float]:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return {
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
            "positive_fraction": np.nan,
            "negative_fraction": np.nan,
        }

    return {
        "median":
            float(
                numeric.median()
            ),

        "q25":
            float(
                numeric.quantile(
                    0.25
                )
            ),

        "q75":
            float(
                numeric.quantile(
                    0.75
                )
            ),

        "minimum":
            float(
                numeric.min()
            ),

        "maximum":
            float(
                numeric.max()
            ),

        "positive_fraction":
            float(
                numeric.gt(0).mean()
            ),

        "negative_fraction":
            float(
                numeric.lt(0).mean()
            ),
    }


def summarize_groups(
    table: pd.DataFrame,
    group_columns: list[str],
    score_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    grouped = table.groupby(
        group_columns,
        sort=True,
        dropna=False,
    )

    for group_key, group in grouped:
        if not isinstance(
            group_key,
            tuple,
        ):
            group_key = (
                group_key,
            )

        row: dict[str, object] = {
            column: value
            for column, value
            in zip(
                group_columns,
                group_key,
            )
        }

        row[
            "signature_count"
        ] = len(group)

        row[
            "unique_cell_count"
        ] = (
            group[
                "cell_id"
            ].nunique(
                dropna=True
            )
            if "cell_id" in group.columns
            else 0
        )

        row[
            "neural_lineage_signature_count"
        ] = int(
            group[
                "cell_class"
            ].eq(
                "neural_lineage"
            ).sum()
        )

        row[
            "unique_dose_count"
        ] = (
            group[
                "pert_idose"
            ].nunique(
                dropna=True
            )
            if "pert_idose" in group.columns
            else 0
        )

        row[
            "unique_time_count"
        ] = (
            group[
                "pert_itime"
            ].nunique(
                dropna=True
            )
            if "pert_itime" in group.columns
            else 0
        )

        for score_column in score_columns:
            summary = summarize_values(
                group[
                    score_column
                ]
            )

            for statistic, value in summary.items():
                row[
                    f"{score_column}_{statistic}"
                ] = value

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    required_files = [
        SUBSET_H5,
        SIGNATURE_METADATA_FILE,
        GENE_METADATA_FILE,
        MATURATION_FILE,
        FETAL_FILE,
        MECHANISM_FILE,
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

    signature_metadata = pd.read_csv(
        SIGNATURE_METADATA_FILE,
        sep="\t",
        low_memory=False,
    )

    gene_metadata = pd.read_csv(
        GENE_METADATA_FILE,
        sep="\t",
        low_memory=False,
    )

    mechanism_table = pd.read_csv(
        MECHANISM_FILE,
        sep="\t",
        low_memory=False,
    )

    if len(signature_metadata) != EXPECTED_SIGNATURES:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_SIGNATURES} signatures, "
                f"found {len(signature_metadata)}."
            )
        )

    if len(gene_metadata) != EXPECTED_GENES:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_GENES} genes, "
                f"found {len(gene_metadata)}."
            )
        )

    signature_id_column = find_column(
        signature_metadata,
        [
            "sig_id",
            "signature_id",
        ],
    )

    gene_id_column = find_column(
        gene_metadata,
        [
            "gene_id",
            "pr_gene_id",
        ],
    )

    gene_symbol_column = find_column(
        gene_metadata,
        [
            "gene_symbol",
            "pr_gene_symbol",
        ],
    )

    landmark_column = find_column(
        gene_metadata,
        [
            "is_landmark",
            "pr_is_lm",
        ],
    )

    chemical_column = find_column(
        signature_metadata,
        [
            "chemical_query",
            "pert_iname",
        ],
    )

    cell_column = find_column(
        signature_metadata,
        [
            "cell_id",
            "cell",
        ],
        required=False,
    )

    evidence_tier_column = find_column(
        signature_metadata,
        [
            "evidence_tier_request",
            "evidence_tier",
        ],
        required=False,
    )

    with h5py.File(
        SUBSET_H5,
        mode="r",
    ) as handle:
        matrix = np.asarray(
            handle[
                "matrix"
            ][()],
            dtype=np.float32,
        )

        h5_signature_ids = decode_strings(
            handle[
                "sig_id"
            ][()]
        )

        h5_gene_ids = decode_strings(
            handle[
                "gene_id"
            ][()]
        )

        h5_gene_symbols = decode_strings(
            handle[
                "gene_symbol"
            ][()]
        )

        h5_landmarks = np.asarray(
            handle[
                "is_landmark"
            ][()],
            dtype=bool,
        )

    if matrix.shape != (
        EXPECTED_SIGNATURES,
        EXPECTED_GENES,
    ):
        raise RuntimeError(
            f"Unexpected matrix shape: {matrix.shape}"
        )

    if not np.isfinite(
        matrix
    ).all():
        raise RuntimeError(
            "The extracted LINCS matrix contains non-finite values."
        )

    metadata_signature_ids = (
        signature_metadata[
            signature_id_column
        ]
        .astype(str)
        .str.strip()
        .tolist()
    )

    metadata_gene_ids = (
        gene_metadata[
            gene_id_column
        ]
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.strip()
        .tolist()
    )

    metadata_gene_symbols = (
        gene_metadata[
            gene_symbol_column
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )

    metadata_landmarks = (
        gene_metadata[
            landmark_column
        ]
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )

    if h5_signature_ids != metadata_signature_ids:
        raise RuntimeError(
            "HDF5 and metadata signature order differ."
        )

    if h5_gene_ids != metadata_gene_ids:
        raise RuntimeError(
            "HDF5 and metadata gene-ID order differ."
        )

    if h5_gene_symbols != metadata_gene_symbols:
        raise RuntimeError(
            "HDF5 and metadata gene-symbol order differ."
        )

    if not np.array_equal(
        h5_landmarks,
        metadata_landmarks,
    ):
        raise RuntimeError(
            "HDF5 and metadata landmark masks differ."
        )

    if int(
        h5_landmarks.sum()
    ) != EXPECTED_LANDMARKS:
        raise RuntimeError(
            "Unexpected landmark-gene count."
        )

    normalized_symbols = [
        normalize_gene(symbol)
        for symbol in h5_gene_symbols
    ]

    if len(
        set(
            normalized_symbols
        )
    ) != EXPECTED_GENES:
        raise RuntimeError(
            "LINCS gene symbols are not unique."
        )

    symbol_to_index = {
        symbol: index
        for index, symbol
        in enumerate(
            normalized_symbols
        )
    }

    maturation_genes = read_gene_set(
        MATURATION_FILE
    )

    fetal_genes = read_gene_set(
        FETAL_FILE
    )

    if maturation_genes & fetal_genes:
        raise RuntimeError(
            "Maturation and fetal programs unexpectedly overlap."
        )

    developmental_union = (
        maturation_genes
        | fetal_genes
    )

    mechanism_category_column = find_column(
        mechanism_table,
        [
            "mechanism_category",
        ],
    )

    mechanism_gene_column = find_column(
        mechanism_table,
        [
            "gene_symbol",
            "gene",
            "symbol",
        ],
    )

    mechanism_axis_column = find_column(
        mechanism_table,
        [
            "axis_role",
        ],
    )

    mechanism_direction_column = find_column(
        mechanism_table,
        [
            "expected_CMAP_direction",
        ],
    )

    mechanism_sets: dict[str, set[str]] = {}

    mechanism_metadata: dict[
        str,
        dict[str, str],
    ] = {}

    for category, group in mechanism_table.groupby(
        mechanism_category_column,
        sort=True,
    ):
        category_name = clean_text(
            category
        )

        mechanism_sets[
            category_name
        ] = {
            normalize_gene(value)
            for value in group[
                mechanism_gene_column
            ]
            if normalize_gene(value)
        }

        mechanism_metadata[
            category_name
        ] = {
            "axis_role":
                clean_text(
                    group[
                        mechanism_axis_column
                    ].iloc[0]
                ),

            "expected_direction":
                clean_text(
                    group[
                        mechanism_direction_column
                    ].iloc[0]
                ).lower(),
        }

    if set(
        mechanism_sets
    ) != set(
        ALL_MECHANISM_CATEGORIES
    ):
        raise RuntimeError(
            (
                "Unexpected mechanism-category set: "
                + repr(
                    sorted(
                        mechanism_sets
                    )
                )
            )
        )

    mapping_rows: list[
        dict[str, object]
    ] = []

    mapped_indices: dict[
        str,
        np.ndarray
    ] = {}

    landmark_indices: dict[
        str,
        np.ndarray
    ] = {}

    leaveout_indices: dict[
        str,
        np.ndarray
    ] = {}

    leaveout_landmark_indices: dict[
        str,
        np.ndarray
    ] = {}

    def register_gene_set(
        gene_set_name: str,
        source_type: str,
        genes: set[str],
        axis_role: str,
        expected_direction: str,
        apply_leaveout: bool,
    ) -> None:
        mapped_genes = sorted(
            genes
            & set(
                symbol_to_index
            )
        )

        mapped = np.asarray(
            [
                symbol_to_index[
                    gene
                ]
                for gene in mapped_genes
            ],
            dtype=int,
        )

        landmark = mapped[
            h5_landmarks[
                mapped
            ]
        ]

        if apply_leaveout:
            leaveout_genes = (
                genes
                - developmental_union
            )
        else:
            leaveout_genes = set(
                genes
            )

        mapped_leaveout_genes = sorted(
            leaveout_genes
            & set(
                symbol_to_index
            )
        )

        leaveout = np.asarray(
            [
                symbol_to_index[
                    gene
                ]
                for gene in mapped_leaveout_genes
            ],
            dtype=int,
        )

        leaveout_landmark = leaveout[
            h5_landmarks[
                leaveout
            ]
        ]

        expected_counts = EXPECTED_GENE_SET_COUNTS[
            gene_set_name
        ]

        observed_counts = {
            "input":
                len(
                    genes
                ),

            "mapped":
                mapped.size,

            "landmark":
                landmark.size,
        }

        if observed_counts != expected_counts:
            raise RuntimeError(
                (
                    f"Gene-set count mismatch for {gene_set_name}: "
                    f"expected {expected_counts}, "
                    f"observed {observed_counts}"
                )
            )

        mapped_indices[
            gene_set_name
        ] = mapped

        landmark_indices[
            gene_set_name
        ] = landmark

        leaveout_indices[
            gene_set_name
        ] = leaveout

        leaveout_landmark_indices[
            gene_set_name
        ] = leaveout_landmark

        mapping_rows.append(
            {
                "source_type":
                    source_type,

                "gene_set_name":
                    gene_set_name,

                "axis_role":
                    axis_role,

                "expected_direction":
                    expected_direction,

                "input_gene_count":
                    len(
                        genes
                    ),

                "LINCS_mapped_gene_count":
                    mapped.size,

                "LINCS_mapping_fraction":
                    mapped.size
                    / len(
                        genes
                    ),

                "LINCS_landmark_gene_count":
                    landmark.size,

                "LINCS_landmark_fraction":
                    landmark.size
                    / len(
                        genes
                    ),

                "developmental_program_genes_removed":
                    len(
                        genes
                        & developmental_union
                    )
                    if apply_leaveout
                    else 0,

                "leave_program_out_gene_count":
                    len(
                        leaveout_genes
                    ),

                "leave_program_out_LINCS_gene_count":
                    leaveout.size,

                "leave_program_out_landmark_gene_count":
                    leaveout_landmark.size,
            }
        )

    register_gene_set(
        gene_set_name=(
            "maturation_high_increasing"
        ),
        source_type=(
            "developmental_program"
        ),
        genes=maturation_genes,
        axis_role=(
            "late_maturation"
        ),
        expected_direction=(
            "up"
        ),
        apply_leaveout=False,
    )

    register_gene_set(
        gene_set_name=(
            "fetal_high_decreasing"
        ),
        source_type=(
            "developmental_program"
        ),
        genes=fetal_genes,
        axis_role=(
            "early_fetal"
        ),
        expected_direction=(
            "dn"
        ),
        apply_leaveout=False,
    )

    for category in sorted(
        mechanism_sets
    ):
        register_gene_set(
            gene_set_name=category,
            source_type=(
                "mechanism_category"
            ),
            genes=mechanism_sets[
                category
            ],
            axis_role=(
                mechanism_metadata[
                    category
                ][
                    "axis_role"
                ]
            ),
            expected_direction=(
                mechanism_metadata[
                    category
                ][
                    "expected_direction"
                ]
            ),
            apply_leaveout=True,
        )

    mapping_table = pd.DataFrame(
        mapping_rows
    )

    mapping_table.to_csv(
        GENE_SET_MAPPING_OUTPUT,
        sep="\t",
        index=False,
    )

    scores = signature_metadata.copy()

    scores[
        "chemical_query"
    ] = scores[
        chemical_column
    ].map(
        clean_text
    )

    if cell_column:
        scores[
            "cell_id"
        ] = scores[
            cell_column
        ].map(
            clean_text
        )
    else:
        scores[
            "cell_id"
        ] = ""

    scores[
        "cell_class"
    ] = scores[
        "cell_id"
    ].map(
        classify_cell
    )

    if evidence_tier_column:
        scores[
            "evidence_tier"
        ] = scores[
            evidence_tier_column
        ].map(
            clean_text
        )
    else:
        scores[
            "evidence_tier"
        ] = ""

    maturation_all = score_gene_indices(
        matrix,
        mapped_indices[
            "maturation_high_increasing"
        ],
    )

    fetal_all = score_gene_indices(
        matrix,
        mapped_indices[
            "fetal_high_decreasing"
        ],
    )

    maturation_landmark = score_gene_indices(
        matrix,
        landmark_indices[
            "maturation_high_increasing"
        ],
    )

    fetal_landmark = score_gene_indices(
        matrix,
        landmark_indices[
            "fetal_high_decreasing"
        ],
    )

    scores[
        "maturation_program_mean_z_all"
    ] = maturation_all

    scores[
        "fetal_program_mean_z_all"
    ] = fetal_all

    scores[
        "developmental_shift_all"
    ] = (
        maturation_all
        - fetal_all
    )

    scores[
        "maturation_program_mean_z_landmark"
    ] = maturation_landmark

    scores[
        "fetal_program_mean_z_landmark"
    ] = fetal_landmark

    scores[
        "developmental_shift_landmark"
    ] = (
        maturation_landmark
        - fetal_landmark
    )

    for category in ALL_MECHANISM_CATEGORIES:
        direction = mechanism_metadata[
            category
        ][
            "expected_direction"
        ]

        multiplier = (
            -1.0
            if direction == "dn"
            else 1.0
        )

        raw_all = score_gene_indices(
            matrix,
            mapped_indices[
                category
            ],
        )

        raw_landmark = score_gene_indices(
            matrix,
            landmark_indices[
                category
            ],
        )

        leaveout_all = score_gene_indices(
            matrix,
            leaveout_indices[
                category
            ],
        )

        leaveout_landmark = score_gene_indices(
            matrix,
            leaveout_landmark_indices[
                category
            ],
        )

        scores[
            f"mechanism_{category}_raw_mean_z_all"
        ] = raw_all

        scores[
            f"mechanism_{category}_oriented_mean_z_all"
        ] = (
            multiplier
            * raw_all
        )

        scores[
            f"mechanism_{category}_raw_mean_z_landmark"
        ] = raw_landmark

        scores[
            f"mechanism_{category}_oriented_mean_z_landmark"
        ] = (
            multiplier
            * raw_landmark
        )

        scores[
            f"mechanism_{category}_leaveout_raw_mean_z_all"
        ] = leaveout_all

        scores[
            f"mechanism_{category}_leaveout_oriented_mean_z_all"
        ] = (
            multiplier
            * leaveout_all
        )

        scores[
            f"mechanism_{category}_leaveout_raw_mean_z_landmark"
        ] = leaveout_landmark

        scores[
            f"mechanism_{category}_leaveout_oriented_mean_z_landmark"
        ] = (
            multiplier
            * leaveout_landmark
        )

    def composite(
        categories: list[str],
        suffix: str,
    ) -> np.ndarray:
        component_matrix = np.vstack(
            [
                scores[
                    (
                        f"mechanism_{category}_"
                        f"{suffix}"
                    )
                ].to_numpy(
                    dtype=float
                )
                for category in categories
            ]
        )

        return np.nanmean(
            component_matrix,
            axis=0,
        )

    for data_scope in [
        "all",
        "landmark",
    ]:
        suffix = (
            f"leaveout_oriented_mean_z_{data_scope}"
        )

        scores[
            f"early_program_suppression_index_{data_scope}"
        ] = composite(
            EARLY_CATEGORIES,
            suffix,
        )

        scores[
            f"late_maturation_support_index_{data_scope}"
        ] = composite(
            LATE_CATEGORIES,
            suffix,
        )

        scores[
            f"injury_stress_index_{data_scope}"
        ] = composite(
            INJURY_CATEGORIES,
            suffix,
        )

        scores[
            f"late_minus_injury_index_{data_scope}"
        ] = (
            scores[
                f"late_maturation_support_index_{data_scope}"
            ]
            - scores[
                f"injury_stress_index_{data_scope}"
            ]
        )

        scores[
            f"late_minus_early_index_{data_scope}"
        ] = (
            scores[
                f"late_maturation_support_index_{data_scope}"
            ]
            - scores[
                f"early_program_suppression_index_{data_scope}"
            ]
        )

    primary_score_columns = [
        "developmental_shift_all",
        "developmental_shift_landmark",
        "early_program_suppression_index_all",
        "late_maturation_support_index_all",
        "injury_stress_index_all",
        "late_minus_injury_index_all",
        "early_program_suppression_index_landmark",
        "late_maturation_support_index_landmark",
        "injury_stress_index_landmark",
        "late_minus_injury_index_landmark",
    ]

    if not np.isfinite(
        scores[
            primary_score_columns
        ].to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "One or more primary quantitative scores are non-finite."
        )

    if scores[
        "chemical_query"
    ].nunique() != EXPECTED_CHEMICALS:
        raise RuntimeError(
            "Unexpected number of matched chemicals."
        )

    scores.to_csv(
        SIGNATURE_SCORE_OUTPUT,
        sep="\t",
        index=False,
        compression="gzip",
    )

    chemical_summary = summarize_groups(
        scores,
        group_columns=[
            "chemical_query",
            "evidence_tier",
        ],
        score_columns=primary_score_columns,
    )

    chemical_summary = chemical_summary.sort_values(
        [
            "signature_count",
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

    chemical_cellclass_summary = summarize_groups(
        scores,
        group_columns=[
            "chemical_query",
            "evidence_tier",
            "cell_class",
        ],
        score_columns=primary_score_columns,
    )

    chemical_cellclass_summary = (
        chemical_cellclass_summary.sort_values(
            [
                "chemical_query",
                "cell_class",
            ]
        )
    )

    chemical_cellclass_summary.to_csv(
        CHEMICAL_CELLCLASS_OUTPUT,
        sep="\t",
        index=False,
    )

    metric_columns = [
        column
        for column in scores.columns
        if column.startswith(
            "metric_"
        )
    ]

    metric_rows: list[
        dict[str, object]
    ] = []

    for column in metric_columns:
        numeric = pd.to_numeric(
            scores[
                column
            ],
            errors="coerce",
        )

        nonmissing = numeric.dropna()

        metric_rows.append(
            {
                "metric_column":
                    column,

                "metadata_dtype":
                    str(
                        scores[
                            column
                        ].dtype
                    ),

                "total_rows":
                    len(
                        scores
                    ),

                "numeric_nonmissing_rows":
                    len(
                        nonmissing
                    ),

                "numeric_nonmissing_fraction":
                    len(
                        nonmissing
                    )
                    / len(
                        scores
                    ),

                "minimum":
                    (
                        float(
                            nonmissing.min()
                        )
                        if not nonmissing.empty
                        else np.nan
                    ),

                "q25":
                    (
                        float(
                            nonmissing.quantile(
                                0.25
                            )
                        )
                        if not nonmissing.empty
                        else np.nan
                    ),

                "median":
                    (
                        float(
                            nonmissing.median()
                        )
                        if not nonmissing.empty
                        else np.nan
                    ),

                "q75":
                    (
                        float(
                            nonmissing.quantile(
                                0.75
                            )
                        )
                        if not nonmissing.empty
                        else np.nan
                    ),

                "maximum":
                    (
                        float(
                            nonmissing.max()
                        )
                        if not nonmissing.empty
                        else np.nan
                    ),

                "unique_raw_values":
                    scores[
                        column
                    ].nunique(
                        dropna=True
                    ),
            }
        )

    quality_metric_summary = pd.DataFrame(
        metric_rows
    )

    quality_metric_summary.to_csv(
        QUALITY_METRIC_OUTPUT,
        sep="\t",
        index=False,
    )

    concordance_pairs = [
        (
            "maturation_program_mean_z",
            "maturation_program_mean_z_all",
            "maturation_program_mean_z_landmark",
        ),
        (
            "fetal_program_mean_z",
            "fetal_program_mean_z_all",
            "fetal_program_mean_z_landmark",
        ),
        (
            "developmental_shift",
            "developmental_shift_all",
            "developmental_shift_landmark",
        ),
        (
            "early_program_suppression_index",
            "early_program_suppression_index_all",
            "early_program_suppression_index_landmark",
        ),
        (
            "late_maturation_support_index",
            "late_maturation_support_index_all",
            "late_maturation_support_index_landmark",
        ),
        (
            "injury_stress_index",
            "injury_stress_index_all",
            "injury_stress_index_landmark",
        ),
        (
            "late_minus_injury_index",
            "late_minus_injury_index_all",
            "late_minus_injury_index_landmark",
        ),
    ]

    for category in ALL_MECHANISM_CATEGORIES:
        concordance_pairs.append(
            (
                f"{category}_leaveout_oriented",
                (
                    f"mechanism_{category}_"
                    "leaveout_oriented_mean_z_all"
                ),
                (
                    f"mechanism_{category}_"
                    "leaveout_oriented_mean_z_landmark"
                ),
            )
        )

    concordance_rows: list[
        dict[str, object]
    ] = []

    for score_name, all_column, landmark_column_name in concordance_pairs:
        all_values = pd.to_numeric(
            scores[
                all_column
            ],
            errors="coerce",
        )

        landmark_values = pd.to_numeric(
            scores[
                landmark_column_name
            ],
            errors="coerce",
        )

        valid = (
            all_values.notna()
            & landmark_values.notna()
        )

        concordance_rows.append(
            {
                "score_name":
                    score_name,

                "signature_count":
                    int(
                        valid.sum()
                    ),

                "pearson_r":
                    float(
                        all_values[
                            valid
                        ].corr(
                            landmark_values[
                                valid
                            ],
                            method="pearson",
                        )
                    ),

                "spearman_r":
                    float(
                        all_values[
                            valid
                        ].corr(
                            landmark_values[
                                valid
                            ],
                            method="spearman",
                        )
                    ),

                "sign_concordance_fraction":
                    float(
                        (
                            np.sign(
                                all_values[
                                    valid
                                ]
                            )
                            == np.sign(
                                landmark_values[
                                    valid
                                ]
                            )
                        ).mean()
                    ),

                "median_absolute_difference":
                    float(
                        (
                            all_values[
                                valid
                            ]
                            - landmark_values[
                                valid
                            ]
                        ).abs().median()
                    ),
            }
        )

    concordance = pd.DataFrame(
        concordance_rows
    )

    concordance.to_csv(
        CONCORDANCE_OUTPUT,
        sep="\t",
        index=False,
    )

    completion = pd.DataFrame(
        [
            {
                "LINCS_signatures_scored":
                    len(
                        scores
                    ),

                "LINCS_genes_used":
                    matrix.shape[
                        1
                    ],

                "LINCS_landmark_genes":
                    int(
                        h5_landmarks.sum()
                    ),

                "matched_chemicals":
                    scores[
                        "chemical_query"
                    ].nunique(),

                "unique_cells":
                    scores[
                        "cell_id"
                    ].nunique(),

                "neural_lineage_signatures":
                    int(
                        scores[
                            "cell_class"
                        ].eq(
                            "neural_lineage"
                        ).sum()
                    ),

                "maturation_genes_mapped":
                    mapped_indices[
                        "maturation_high_increasing"
                    ].size,

                "fetal_genes_mapped":
                    mapped_indices[
                        "fetal_high_decreasing"
                    ].size,

                "maturation_landmark_genes":
                    landmark_indices[
                        "maturation_high_increasing"
                    ].size,

                "fetal_landmark_genes":
                    landmark_indices[
                        "fetal_high_decreasing"
                    ].size,

                "mechanism_categories_scored":
                    len(
                        ALL_MECHANISM_CATEGORIES
                    ),

                "leave_program_out_scores_calculated":
                    True,

                "full_gene_scores_calculated":
                    True,

                "landmark_only_scores_calculated":
                    True,

                "all_primary_scores_finite":
                    bool(
                        np.isfinite(
                            scores[
                                primary_score_columns
                            ].to_numpy(
                                dtype=float
                            )
                        ).all()
                    ),

                "inferential_significance_calculated":
                    False,

                "Phase6E3A_status":
                    "completed",
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
            "===== PHASE 6E3A COMPLETION =====",
            completion.to_string(
                index=False
            ),
            "",
            "===== GENE-SET MAPPING =====",
            mapping_table.to_string(
                index=False
            ),
            "",
            "===== FULL-GENE VS LANDMARK CONCORDANCE =====",
            concordance.to_string(
                index=False
            ),
            "",
            "===== CHEMICAL-LEVEL SCORE SUMMARY =====",
            chemical_summary.to_string(
                index=False
            ),
            "",
            "===== QUALITY-METRIC DISTRIBUTION =====",
            (
                quality_metric_summary.to_string(
                    index=False
                )
                if not quality_metric_summary.empty
                else "No metric_ columns detected."
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


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            (
                "Phase 6E3A failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
