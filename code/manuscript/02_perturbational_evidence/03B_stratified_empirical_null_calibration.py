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
      "phase6E3B_stratified_empirical_null.log"
)

PROFILE_H5_OUTPUT = (
    PROCESSED_DIR
    / "phase6E3B_aggregated_chemical_profiles.h5"
)

PROFILE_METADATA_OUTPUT = (
    TABLE_DIR
    / "phase6E3B_aggregated_profile_metadata.tsv"
)

GENE_SET_MAPPING_OUTPUT = (
    TABLE_DIR
    / "phase6E3B_empirical_gene_set_mapping.tsv"
)

STRATA_OUTPUT = (
    TABLE_DIR
    / "phase6E3B_gene_permutation_strata.tsv"
)

RESULT_OUTPUT = (
    TABLE_DIR
    / "phase6E3B_empirical_calibration_results.tsv"
)

PRIMARY_OUTPUT = (
    TABLE_DIR
    / "phase6E3B_primary_cell_balanced_results.tsv"
)

SUPPORTED_OUTPUT = (
    TABLE_DIR
    / "phase6E3B_empirically_supported_results.tsv"
)

NULL_SUMMARY_OUTPUT = (
    TABLE_DIR
    / "phase6E3B_null_distribution_summary.tsv"
)

NULL_ARRAY_OUTPUT = (
    PROCESSED_DIR
    / "phase6E3B_empirical_null_scores.npz"
)

COMPLETION_OUTPUT = (
    TABLE_DIR
    / "phase6E3B_completion_summary.tsv"
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

N_PERMUTATIONS = 5000
RANDOM_SEED = 20260721

PRIMARY_SCOPE = "all_cells_cell_balanced"

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

ALL_CATEGORIES = (
    EARLY_CATEGORIES
    + LATE_CATEGORIES
    + INJURY_CATEGORIES
)

EXPECTED_DIRECTION = {
    "early_proliferative_state":
        "dn",

    "fetal_neurodevelopmental_state":
        "dn",

    "late_synaptic_maturation":
        "up",

    "mitochondrial_metabolic_maturation":
        "up",

    "glial_myelin_maturation":
        "up",

    "DNA_damage_p53":
        "up",

    "apoptosis_cytotoxicity":
        "up",

    "oxidative_hypoxic_stress":
        "up",

    "inflammatory_stress":
        "up",
}

AXIS_NAMES = [
    "developmental_shift",
    "early_program_suppression",
    "late_maturation_support",
    "injury_stress",
    "late_minus_injury",
    "late_minus_early",
]


def decode_strings(
    values: np.ndarray,
) -> list[str]:
    output: list[str] = []

    for value in values:
        if isinstance(value, bytes):
            value = value.decode(
                "utf-8",
                errors="replace",
            )

        output.append(
            str(value).strip()
        )

    return output


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
    return clean_text(
        value
    ).upper()


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
            return lookup[
                candidate.lower()
            ]

    if required:
        raise RuntimeError(
            "Could not identify any expected column: "
            + ", ".join(
                candidates
            )
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
        for value in table[
            gene_column
        ]
        if normalize_gene(value)
    }


def classify_cell(
    value: object,
) -> str:
    cell = clean_text(
        value
    ).upper()

    neural_cells = {
        "NPC",
        "NPC.CAS9",
        "NPC.TAK",
        "NEU",
        "MNEU.E",
    }

    if cell in neural_cells:
        return "neural_lineage"

    return "non_neural"


def parse_boolean(
    values: pd.Series,
) -> pd.Series:
    normalized = (
        values
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


def rank_bins(
    values: np.ndarray,
    number_of_bins: int,
) -> np.ndarray:
    if values.size == 0:
        return np.asarray(
            [],
            dtype=int,
        )

    order = np.argsort(
        values,
        kind="mergesort",
    )

    output = np.empty(
        values.size,
        dtype=int,
    )

    output[
        order
    ] = (
        np.arange(
            values.size,
            dtype=int,
        )
        * number_of_bins
        // values.size
    )

    output = np.minimum(
        output,
        number_of_bins - 1,
    )

    return output


def bh_adjust(
    p_values: np.ndarray,
) -> np.ndarray:
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    output = np.full(
        p_values.shape,
        np.nan,
        dtype=float,
    )

    valid = np.isfinite(
        p_values
    )

    if not valid.any():
        return output

    valid_values = p_values[
        valid
    ]

    order = np.argsort(
        valid_values
    )

    ranked = valid_values[
        order
    ]

    number = ranked.size

    adjusted = (
        ranked
        * number
        / np.arange(
            1,
            number + 1,
            dtype=float,
        )
    )

    adjusted = np.minimum.accumulate(
        adjusted[
            ::-1
        ]
    )[
        ::-1
    ]

    adjusted = np.minimum(
        adjusted,
        1.0,
    )

    inverse = np.empty_like(
        order
    )

    inverse[
        order
    ] = np.arange(
        number
    )

    output[
        valid
    ] = adjusted[
        inverse
    ]

    return output


def empirical_p_upper(
    null_values: np.ndarray,
    observed: float,
) -> float:
    return float(
        (
            1
            + np.count_nonzero(
                null_values >= observed
            )
        )
        / (
            null_values.size
            + 1
        )
    )


def empirical_p_lower(
    null_values: np.ndarray,
    observed: float,
) -> float:
    return float(
        (
            1
            + np.count_nonzero(
                null_values <= observed
            )
        )
        / (
            null_values.size
            + 1
        )
    )


def median_profile(
    matrix: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    if indices.size == 0:
        raise RuntimeError(
            "Cannot aggregate an empty signature group."
        )

    return np.median(
        matrix[
            indices,
            :,
        ],
        axis=0,
    ).astype(
        np.float32
    )


def cell_balanced_profile(
    matrix: np.ndarray,
    metadata: pd.DataFrame,
    indices: np.ndarray,
) -> tuple[np.ndarray, int]:
    subset = metadata.iloc[
        indices
    ]

    cell_profiles: list[
        np.ndarray
    ] = []

    for _, group in subset.groupby(
        "cell_id",
        sort=True,
        dropna=False,
    ):
        row_indices = group.index.to_numpy(
            dtype=int
        )

        cell_profiles.append(
            median_profile(
                matrix,
                row_indices,
            )
        )

    if not cell_profiles:
        raise RuntimeError(
            "No cell profiles were generated."
        )

    stacked = np.stack(
        cell_profiles,
        axis=0,
    )

    profile = np.median(
        stacked,
        axis=0,
    ).astype(
        np.float32
    )

    return (
        profile,
        len(
            cell_profiles
        ),
    )


def calculate_axes(
    profiles: np.ndarray,
    gene_sets: dict[str, np.ndarray],
    mapping: np.ndarray | None = None,
) -> np.ndarray:
    def indices(
        name: str,
    ) -> np.ndarray:
        original = gene_sets[
            name
        ]

        if mapping is None:
            return original

        return mapping[
            original
        ]

    maturation = profiles[
        :,
        indices(
            "maturation_high_increasing"
        ),
    ].mean(
        axis=1,
        dtype=np.float64,
    )

    fetal = profiles[
        :,
        indices(
            "fetal_high_decreasing"
        ),
    ].mean(
        axis=1,
        dtype=np.float64,
    )

    oriented: dict[
        str,
        np.ndarray
    ] = {}

    for category in ALL_CATEGORIES:
        values = profiles[
            :,
            indices(
                category
            ),
        ].mean(
            axis=1,
            dtype=np.float64,
        )

        multiplier = (
            -1.0
            if EXPECTED_DIRECTION[
                category
            ] == "dn"
            else 1.0
        )

        oriented[
            category
        ] = (
            multiplier
            * values
        )

    early = np.mean(
        np.vstack(
            [
                oriented[
                    category
                ]
                for category
                in EARLY_CATEGORIES
            ]
        ),
        axis=0,
    )

    late = np.mean(
        np.vstack(
            [
                oriented[
                    category
                ]
                for category
                in LATE_CATEGORIES
            ]
        ),
        axis=0,
    )

    injury = np.mean(
        np.vstack(
            [
                oriented[
                    category
                ]
                for category
                in INJURY_CATEGORIES
            ]
        ),
        axis=0,
    )

    return np.column_stack(
        [
            maturation
            - fetal,
            early,
            late,
            injury,
            late
            - injury,
            late
            - early,
        ]
    )


def main() -> None:
    required_files = [
        SUBSET_H5,
        SIGNATURE_METADATA_FILE,
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

    metadata = pd.read_csv(
        SIGNATURE_METADATA_FILE,
        sep="\t",
        low_memory=False,
    ).reset_index(
        drop=True
    )

    if len(metadata) != EXPECTED_SIGNATURES:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_SIGNATURES} signatures, "
                f"found {len(metadata)}."
            )
        )

    signature_id_column = find_column(
        metadata,
        [
            "sig_id",
            "signature_id",
        ],
    )

    chemical_column = find_column(
        metadata,
        [
            "chemical_query",
            "pert_iname",
        ],
    )

    cell_column = find_column(
        metadata,
        [
            "cell_id",
            "cell",
        ],
    )

    evidence_tier_column = find_column(
        metadata,
        [
            "evidence_tier_request",
            "evidence_tier",
        ],
        required=False,
    )

    dose_column = find_column(
        metadata,
        [
            "pert_idose",
            "_dose_label",
        ],
        required=False,
    )

    time_column = find_column(
        metadata,
        [
            "pert_itime",
            "_time_label",
        ],
        required=False,
    )

    metadata[
        "sig_id"
    ] = metadata[
        signature_id_column
    ].map(
        clean_text
    )

    metadata[
        "chemical_query"
    ] = metadata[
        chemical_column
    ].map(
        clean_text
    )

    metadata[
        "cell_id"
    ] = metadata[
        cell_column
    ].map(
        clean_text
    )

    metadata[
        "cell_class"
    ] = metadata[
        "cell_id"
    ].map(
        classify_cell
    )

    if evidence_tier_column:
        metadata[
            "evidence_tier"
        ] = metadata[
            evidence_tier_column
        ].map(
            clean_text
        )
    else:
        metadata[
            "evidence_tier"
        ] = ""

    high_quality_candidates = [
        column
        for column in metadata.columns
        if "is_hiq" in column.lower()
    ]

    high_quality_column = (
        high_quality_candidates[0]
        if high_quality_candidates
        else None
    )

    if high_quality_column:
        metadata[
            "_is_high_quality"
        ] = parse_boolean(
            metadata[
                high_quality_column
            ]
        )
    else:
        metadata[
            "_is_high_quality"
        ] = False

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

        signature_ids = decode_strings(
            handle[
                "sig_id"
            ][()]
        )

        gene_symbols = [
            normalize_gene(
                value
            )
            for value in decode_strings(
                handle[
                    "gene_symbol"
                ][()]
            )
        ]

        landmark_mask = np.asarray(
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

    if signature_ids != metadata[
        "sig_id"
    ].tolist():
        raise RuntimeError(
            "Signature order differs between HDF5 and metadata."
        )

    if int(
        landmark_mask.sum()
    ) != EXPECTED_LANDMARKS:
        raise RuntimeError(
            "Unexpected landmark-gene count."
        )

    if not np.isfinite(
        matrix
    ).all():
        raise RuntimeError(
            "Non-finite values were detected in the LINCS matrix."
        )

    if len(
        set(
            gene_symbols
        )
    ) != EXPECTED_GENES:
        raise RuntimeError(
            "LINCS gene symbols are not unique."
        )

    symbol_to_index = {
        symbol: index
        for index, symbol
        in enumerate(
            gene_symbols
        )
    }

    lincs_gene_set = set(
        symbol_to_index
    )

    maturation_genes = read_gene_set(
        MATURATION_FILE
    )

    fetal_genes = read_gene_set(
        FETAL_FILE
    )

    if maturation_genes & fetal_genes:
        raise RuntimeError(
            "Developmental programs unexpectedly overlap."
        )

    developmental_union = (
        maturation_genes
        | fetal_genes
    )

    mechanism_table = pd.read_csv(
        MECHANISM_FILE,
        sep="\t",
        low_memory=False,
    )

    category_column = find_column(
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

    mechanism_sets: dict[
        str,
        set[str]
    ] = {}

    for category, group in mechanism_table.groupby(
        category_column,
        sort=True,
    ):
        category_name = clean_text(
            category
        )

        mechanism_sets[
            category_name
        ] = {
            normalize_gene(
                value
            )
            for value in group[
                mechanism_gene_column
            ]
            if normalize_gene(
                value
            )
        }

    if set(
        mechanism_sets
    ) != set(
        ALL_CATEGORIES
    ):
        raise RuntimeError(
            (
                "Unexpected mechanism categories: "
                + repr(
                    sorted(
                        mechanism_sets
                    )
                )
            )
        )

    gene_sets: dict[
        str,
        np.ndarray
    ] = {}

    mapping_rows: list[
        dict[str, object]
    ] = []

    def register_set(
        name: str,
        source_type: str,
        genes: set[str],
        remove_developmental_programs: bool,
    ) -> None:
        original_count = len(
            genes
        )

        working_genes = (
            genes
            - developmental_union
            if remove_developmental_programs
            else set(
                genes
            )
        )

        mapped_genes = sorted(
            working_genes
            & lincs_gene_set
        )

        mapped_indices = np.asarray(
            [
                symbol_to_index[
                    gene
                ]
                for gene in mapped_genes
            ],
            dtype=int,
        )

        if mapped_indices.size == 0:
            raise RuntimeError(
                f"No LINCS genes mapped for {name}."
            )

        gene_sets[
            name
        ] = mapped_indices

        mapping_rows.append(
            {
                "source_type":
                    source_type,

                "gene_set_name":
                    name,

                "expected_direction":
                    (
                        EXPECTED_DIRECTION[
                            name
                        ]
                        if name
                        in EXPECTED_DIRECTION
                        else (
                            "up"
                            if name
                            == "maturation_high_increasing"
                            else "dn"
                        )
                    ),

                "original_gene_count":
                    original_count,

                "developmental_program_genes_removed":
                    (
                        len(
                            genes
                            & developmental_union
                        )
                        if remove_developmental_programs
                        else 0
                    ),

                "post_removal_gene_count":
                    len(
                        working_genes
                    ),

                "LINCS_mapped_gene_count":
                    mapped_indices.size,

                "LINCS_landmark_gene_count":
                    int(
                        landmark_mask[
                            mapped_indices
                        ].sum()
                    ),
            }
        )

    register_set(
        "maturation_high_increasing",
        "developmental_program",
        maturation_genes,
        False,
    )

    register_set(
        "fetal_high_decreasing",
        "developmental_program",
        fetal_genes,
        False,
    )

    for category in ALL_CATEGORIES:
        register_set(
            category,
            "mechanism_category_leave_program_out",
            mechanism_sets[
                category
            ],
            True,
        )

    mapping_table = pd.DataFrame(
        mapping_rows
    )

    mapping_table.to_csv(
        GENE_SET_MAPPING_OUTPUT,
        sep="\t",
        index=False,
    )

    profile_arrays: list[
        np.ndarray
    ] = []

    profile_rows: list[
        dict[str, object]
    ] = []

    def add_scope(
        scope_name: str,
        subset_mask: pd.Series,
        cell_balanced: bool,
    ) -> None:
        subset_metadata = metadata.loc[
            subset_mask
        ]

        for chemical, chemical_group in subset_metadata.groupby(
            "chemical_query",
            sort=True,
        ):
            row_indices = chemical_group.index.to_numpy(
                dtype=int
            )

            if cell_balanced:
                profile, contributing_cells = (
                    cell_balanced_profile(
                        matrix,
                        metadata,
                        row_indices,
                    )
                )
            else:
                profile = median_profile(
                    matrix,
                    row_indices,
                )

                contributing_cells = (
                    chemical_group[
                        "cell_id"
                    ].nunique()
                )

            profile_index = len(
                profile_arrays
            )

            profile_id = (
                f"{scope_name}|{chemical}"
            )

            profile_arrays.append(
                profile
            )

            profile_rows.append(
                {
                    "profile_index":
                        profile_index,

                    "profile_id":
                        profile_id,

                    "analysis_scope":
                        scope_name,

                    "is_primary_scope":
                        scope_name
                        == PRIMARY_SCOPE,

                    "chemical_query":
                        chemical,

                    "evidence_tier":
                        clean_text(
                            chemical_group[
                                "evidence_tier"
                            ].iloc[0]
                        ),

                    "signature_count":
                        len(
                            chemical_group
                        ),

                    "unique_cell_count":
                        chemical_group[
                            "cell_id"
                        ].nunique(),

                    "contributing_cell_profiles":
                        contributing_cells,

                    "neural_lineage_signature_count":
                        int(
                            chemical_group[
                                "cell_class"
                            ].eq(
                                "neural_lineage"
                            ).sum()
                        ),

                    "unique_dose_count":
                        (
                            chemical_group[
                                dose_column
                            ].nunique(
                                dropna=True
                            )
                            if dose_column
                            else 0
                        ),

                    "unique_time_count":
                        (
                            chemical_group[
                                time_column
                            ].nunique(
                                dropna=True
                            )
                            if time_column
                            else 0
                        ),

                    "high_quality_signature_count":
                        int(
                            chemical_group[
                                "_is_high_quality"
                            ].sum()
                        ),

                    "high_quality_flag_column":
                        (
                            high_quality_column
                            if high_quality_column
                            else ""
                        ),
                }
            )

    all_mask = pd.Series(
        True,
        index=metadata.index,
    )

    add_scope(
        "all_cells_cell_balanced",
        all_mask,
        True,
    )

    add_scope(
        "all_signatures_median",
        all_mask,
        False,
    )

    neural_mask = metadata[
        "cell_class"
    ].eq(
        "neural_lineage"
    )

    add_scope(
        "neural_lineage_cell_balanced",
        neural_mask,
        True,
    )

    if high_quality_column:
        high_quality_mask = metadata[
            "_is_high_quality"
        ]

        if high_quality_mask.any():
            add_scope(
                "high_quality_cell_balanced",
                high_quality_mask,
                True,
            )

            high_quality_neural_mask = (
                high_quality_mask
                & neural_mask
            )

            if high_quality_neural_mask.any():
                add_scope(
                    "high_quality_neural_cell_balanced",
                    high_quality_neural_mask,
                    True,
                )

    profile_matrix = np.stack(
        profile_arrays,
        axis=0,
    ).astype(
        np.float32
    )

    profile_metadata = pd.DataFrame(
        profile_rows
    )

    primary_profiles = profile_metadata.loc[
        profile_metadata[
            "analysis_scope"
        ].eq(
            PRIMARY_SCOPE
        )
    ]

    if primary_profiles[
        "chemical_query"
    ].nunique() != EXPECTED_CHEMICALS:
        raise RuntimeError(
            (
                "Primary cell-balanced scope does not contain "
                f"all {EXPECTED_CHEMICALS} chemicals."
            )
        )

    if not np.isfinite(
        profile_matrix
    ).all():
        raise RuntimeError(
            "Aggregated profiles contain non-finite values."
        )

    profile_metadata.to_csv(
        PROFILE_METADATA_OUTPUT,
        sep="\t",
        index=False,
    )

    string_dtype = h5py.string_dtype(
        encoding="utf-8"
    )

    if PROFILE_H5_OUTPUT.exists():
        PROFILE_H5_OUTPUT.unlink()

    with h5py.File(
        PROFILE_H5_OUTPUT,
        mode="w",
    ) as handle:
        handle.create_dataset(
            "matrix",
            data=profile_matrix,
            dtype="float32",
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=(
                min(
                    16,
                    profile_matrix.shape[
                        0
                    ],
                ),
                1024,
            ),
        )

        handle.create_dataset(
            "profile_id",
            data=np.asarray(
                profile_metadata[
                    "profile_id"
                ].tolist(),
                dtype=object,
            ),
            dtype=string_dtype,
        )

        handle.create_dataset(
            "gene_symbol",
            data=np.asarray(
                gene_symbols,
                dtype=object,
            ),
            dtype=string_dtype,
        )

        handle.attrs[
            "matrix_orientation"
        ] = "aggregated_profiles_by_genes"

        handle.attrs[
            "primary_scope"
        ] = PRIMARY_SCOPE

    gene_mean = matrix.mean(
        axis=0,
        dtype=np.float64,
    )

    gene_sd = matrix.std(
        axis=0,
        dtype=np.float64,
    )

    number_of_mean_bins = 5
    number_of_sd_bins = 5

    mean_bins = np.full(
        EXPECTED_GENES,
        -1,
        dtype=int,
    )

    sd_bins = np.full(
        EXPECTED_GENES,
        -1,
        dtype=int,
    )

    for landmark_value in [
        False,
        True,
    ]:
        group_indices = np.where(
            landmark_mask
            == landmark_value
        )[0]

        mean_bins[
            group_indices
        ] = rank_bins(
            gene_mean[
                group_indices
            ],
            number_of_mean_bins,
        )

        sd_bins[
            group_indices
        ] = rank_bins(
            gene_sd[
                group_indices
            ],
            number_of_sd_bins,
        )

    stratum_labels = (
        landmark_mask.astype(
            int
        )
        * (
            number_of_mean_bins
            * number_of_sd_bins
        )
        + mean_bins
        * number_of_sd_bins
        + sd_bins
    )

    unique_strata = np.unique(
        stratum_labels
    )

    stratum_indices = [
        np.where(
            stratum_labels
            == stratum
        )[0]
        for stratum in unique_strata
    ]

    strata_rows: list[
        dict[str, object]
    ] = []

    for stratum, indices in zip(
        unique_strata,
        stratum_indices,
    ):
        strata_rows.append(
            {
                "stratum_id":
                    int(
                        stratum
                    ),

                "is_landmark":
                    bool(
                        stratum
                        // (
                            number_of_mean_bins
                            * number_of_sd_bins
                        )
                    ),

                "mean_bin":
                    int(
                        (
                            stratum
                            % (
                                number_of_mean_bins
                                * number_of_sd_bins
                            )
                        )
                        // number_of_sd_bins
                    ),

                "standard_deviation_bin":
                    int(
                        stratum
                        % number_of_sd_bins
                    ),

                "gene_count":
                    len(
                        indices
                    ),

                "minimum_gene_mean":
                    float(
                        gene_mean[
                            indices
                        ].min()
                    ),

                "maximum_gene_mean":
                    float(
                        gene_mean[
                            indices
                        ].max()
                    ),

                "minimum_gene_standard_deviation":
                    float(
                        gene_sd[
                            indices
                        ].min()
                    ),

                "maximum_gene_standard_deviation":
                    float(
                        gene_sd[
                            indices
                        ].max()
                    ),
            }
        )

    strata_table = pd.DataFrame(
        strata_rows
    )

    strata_table.to_csv(
        STRATA_OUTPUT,
        sep="\t",
        index=False,
    )

    observed_scores = calculate_axes(
        profile_matrix,
        gene_sets,
        mapping=None,
    )

    if not np.isfinite(
        observed_scores
    ).all():
        raise RuntimeError(
            "Observed profile scores contain non-finite values."
        )

    null_scores = np.empty(
        (
            N_PERMUTATIONS,
            profile_matrix.shape[
                0
            ],
            len(
                AXIS_NAMES
            ),
        ),
        dtype=np.float32,
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    identity_mapping = np.arange(
        EXPECTED_GENES,
        dtype=int,
    )

    for permutation_index in range(
        N_PERMUTATIONS
    ):
        mapping = identity_mapping.copy()

        for indices in stratum_indices:
            mapping[
                indices
            ] = rng.permutation(
                indices
            )

        null_scores[
            permutation_index,
            :,
            :,
        ] = calculate_axes(
            profile_matrix,
            gene_sets,
            mapping=mapping,
        ).astype(
            np.float32
        )

        if (
            permutation_index
            == 0
            or (
                permutation_index
                + 1
            )
            % 250
            == 0
        ):
            print(
                (
                    "Completed empirical permutation "
                    f"{permutation_index + 1}/"
                    f"{N_PERMUTATIONS}"
                ),
                flush=True,
            )

    if not np.isfinite(
        null_scores
    ).all():
        raise RuntimeError(
            "Empirical-null scores contain non-finite values."
        )

    np.savez_compressed(
        NULL_ARRAY_OUTPUT,
        null_scores=null_scores,
        observed_scores=observed_scores.astype(
            np.float32
        ),
        profile_ids=np.asarray(
            profile_metadata[
                "profile_id"
            ].tolist()
        ),
        axis_names=np.asarray(
            AXIS_NAMES
        ),
        random_seed=np.asarray(
            [
                RANDOM_SEED
            ],
            dtype=np.int64,
        ),
        permutations=np.asarray(
            [
                N_PERMUTATIONS
            ],
            dtype=np.int64,
        ),
    )

    null_mean = null_scores.mean(
        axis=0,
        dtype=np.float64,
    )

    null_sd = null_scores.std(
        axis=0,
        dtype=np.float64,
    )

    if np.any(
        null_sd <= 0
    ):
        raise RuntimeError(
            "A null distribution has zero variance."
        )

    observed_z = (
        observed_scores
        - null_mean
    ) / null_sd

    null_z = (
        null_scores.astype(
            np.float64
        )
        - null_mean[
            np.newaxis,
            :,
            :,
        ]
    ) / null_sd[
        np.newaxis,
        :,
        :,
    ]

    result_rows: list[
        dict[str, object]
    ] = []

    result_lookup: dict[
        tuple[int, int],
        int
    ] = {}

    null_summary_rows: list[
        dict[str, object]
    ] = []

    for profile_index, profile_row in profile_metadata.iterrows():
        for axis_index, axis_name in enumerate(
            AXIS_NAMES
        ):
            null_values = null_scores[
                :,
                profile_index,
                axis_index,
            ].astype(
                np.float64
            )

            observed = float(
                observed_scores[
                    profile_index,
                    axis_index,
                ]
            )

            p_upper = empirical_p_upper(
                null_values,
                observed,
            )

            p_lower = empirical_p_lower(
                null_values,
                observed,
            )

            p_two_sided = min(
                1.0,
                2.0
                * min(
                    p_upper,
                    p_lower,
                ),
            )

            result_lookup[
                (
                    profile_index,
                    axis_index,
                )
            ] = len(
                result_rows
            )

            result_rows.append(
                {
                    "profile_index":
                        profile_index,

                    "profile_id":
                        profile_row[
                            "profile_id"
                        ],

                    "analysis_scope":
                        profile_row[
                            "analysis_scope"
                        ],

                    "is_primary_scope":
                        profile_row[
                            "is_primary_scope"
                        ],

                    "chemical_query":
                        profile_row[
                            "chemical_query"
                        ],

                    "evidence_tier":
                        profile_row[
                            "evidence_tier"
                        ],

                    "signature_count":
                        profile_row[
                            "signature_count"
                        ],

                    "unique_cell_count":
                        profile_row[
                            "unique_cell_count"
                        ],

                    "neural_lineage_signature_count":
                        profile_row[
                            "neural_lineage_signature_count"
                        ],

                    "axis_name":
                        axis_name,

                    "observed_score":
                        observed,

                    "null_mean":
                        float(
                            null_mean[
                                profile_index,
                                axis_index,
                            ]
                        ),

                    "null_standard_deviation":
                        float(
                            null_sd[
                                profile_index,
                                axis_index,
                            ]
                        ),

                    "empirical_z":
                        float(
                            observed_z[
                                profile_index,
                                axis_index,
                            ]
                        ),

                    "empirical_p_upper":
                        p_upper,

                    "empirical_p_lower":
                        p_lower,

                    "empirical_p_two_sided":
                        p_two_sided,
                }
            )

            null_summary_rows.append(
                {
                    "profile_id":
                        profile_row[
                            "profile_id"
                        ],

                    "analysis_scope":
                        profile_row[
                            "analysis_scope"
                        ],

                    "chemical_query":
                        profile_row[
                            "chemical_query"
                        ],

                    "axis_name":
                        axis_name,

                    "null_mean":
                        float(
                            np.mean(
                                null_values
                            )
                        ),

                    "null_standard_deviation":
                        float(
                            np.std(
                                null_values
                            )
                        ),

                    "null_q001":
                        float(
                            np.quantile(
                                null_values,
                                0.001,
                            )
                        ),

                    "null_q025":
                        float(
                            np.quantile(
                                null_values,
                                0.025,
                            )
                        ),

                    "null_median":
                        float(
                            np.median(
                                null_values
                            )
                        ),

                    "null_q975":
                        float(
                            np.quantile(
                                null_values,
                                0.975,
                            )
                        ),

                    "null_q999":
                        float(
                            np.quantile(
                                null_values,
                                0.999,
                            )
                        ),
                }
            )

    results = pd.DataFrame(
        result_rows
    )

    null_summary = pd.DataFrame(
        null_summary_rows
    )

    for p_column in [
        "empirical_p_upper",
        "empirical_p_lower",
        "empirical_p_two_sided",
    ]:
        within_axis_column = (
            p_column
            + "_FDR_within_axis"
        )

        within_scope_column = (
            p_column
            + "_FDR_within_scope_all_axes"
        )

        results[
            within_axis_column
        ] = np.nan

        results[
            within_scope_column
        ] = np.nan

        for _, indices in results.groupby(
            [
                "analysis_scope",
                "axis_name",
            ],
            sort=False,
        ).groups.items():
            index_array = np.asarray(
                list(
                    indices
                ),
                dtype=int,
            )

            results.loc[
                index_array,
                within_axis_column,
            ] = bh_adjust(
                results.loc[
                    index_array,
                    p_column,
                ].to_numpy(
                    dtype=float
                )
            )

        for _, indices in results.groupby(
            "analysis_scope",
            sort=False,
        ).groups.items():
            index_array = np.asarray(
                list(
                    indices
                ),
                dtype=int,
            )

            results.loc[
                index_array,
                within_scope_column,
            ] = bh_adjust(
                results.loc[
                    index_array,
                    p_column,
                ].to_numpy(
                    dtype=float
                )
            )

    for column in [
        "maxT_p_upper_within_axis",
        "maxT_p_lower_within_axis",
        "maxT_p_two_sided_within_axis",
        "maxT_p_upper_within_scope_all_axes",
        "maxT_p_lower_within_scope_all_axes",
        "maxT_p_two_sided_within_scope_all_axes",
    ]:
        results[
            column
        ] = np.nan

    for scope, scope_group in profile_metadata.groupby(
        "analysis_scope",
        sort=False,
    ):
        profile_indices = scope_group[
            "profile_index"
        ].to_numpy(
            dtype=int
        )

        scope_null_z = null_z[
            :,
            profile_indices,
            :,
        ]

        global_max = scope_null_z.max(
            axis=(
                1,
                2,
            )
        )

        global_min = scope_null_z.min(
            axis=(
                1,
                2,
            )
        )

        global_max_abs = np.abs(
            scope_null_z
        ).max(
            axis=(
                1,
                2,
            )
        )

        for axis_index, axis_name in enumerate(
            AXIS_NAMES
        ):
            axis_null_z = null_z[
                :,
                profile_indices,
                axis_index,
            ]

            axis_max = axis_null_z.max(
                axis=1
            )

            axis_min = axis_null_z.min(
                axis=1
            )

            axis_max_abs = np.abs(
                axis_null_z
            ).max(
                axis=1
            )

            for profile_index in profile_indices:
                result_index = result_lookup[
                    (
                        int(
                            profile_index
                        ),
                        axis_index,
                    )
                ]

                z_value = float(
                    observed_z[
                        profile_index,
                        axis_index,
                    ]
                )

                results.loc[
                    result_index,
                    "maxT_p_upper_within_axis",
                ] = empirical_p_upper(
                    axis_max,
                    z_value,
                )

                results.loc[
                    result_index,
                    "maxT_p_lower_within_axis",
                ] = empirical_p_lower(
                    axis_min,
                    z_value,
                )

                results.loc[
                    result_index,
                    "maxT_p_two_sided_within_axis",
                ] = empirical_p_upper(
                    axis_max_abs,
                    abs(
                        z_value
                    ),
                )

                results.loc[
                    result_index,
                    "maxT_p_upper_within_scope_all_axes",
                ] = empirical_p_upper(
                    global_max,
                    z_value,
                )

                results.loc[
                    result_index,
                    "maxT_p_lower_within_scope_all_axes",
                ] = empirical_p_lower(
                    global_min,
                    z_value,
                )

                results.loc[
                    result_index,
                    "maxT_p_two_sided_within_scope_all_axes",
                ] = empirical_p_upper(
                    global_max_abs,
                    abs(
                        z_value
                    ),
                )

    upper_fdr_column = (
        "empirical_p_upper_"
        "FDR_within_scope_all_axes"
    )

    lower_fdr_column = (
        "empirical_p_lower_"
        "FDR_within_scope_all_axes"
    )

    results[
        "directional_FDR_evidence_class"
    ] = np.select(
        [
            results[
                upper_fdr_column
            ].lt(
                0.05
            ),
            results[
                lower_fdr_column
            ].lt(
                0.05
            ),
        ],
        [
            "positive_FDR_supported",
            "negative_FDR_supported",
        ],
        default="not_FDR_supported",
    )

    results[
        "global_maxT_evidence_class"
    ] = np.select(
        [
            results[
                "maxT_p_upper_within_scope_all_axes"
            ].lt(
                0.05
            ),
            results[
                "maxT_p_lower_within_scope_all_axes"
            ].lt(
                0.05
            ),
        ],
        [
            "positive_global_maxT_supported",
            "negative_global_maxT_supported",
        ],
        default="not_global_maxT_supported",
    )

    results = results.sort_values(
        [
            "analysis_scope",
            "axis_name",
            "empirical_p_two_sided",
            "chemical_query",
        ]
    ).reset_index(
        drop=True
    )

    results.to_csv(
        RESULT_OUTPUT,
        sep="\t",
        index=False,
    )

    null_summary.to_csv(
        NULL_SUMMARY_OUTPUT,
        sep="\t",
        index=False,
    )

    primary_results = results.loc[
        results[
            "analysis_scope"
        ].eq(
            PRIMARY_SCOPE
        )
    ].copy()

    primary_results.to_csv(
        PRIMARY_OUTPUT,
        sep="\t",
        index=False,
    )

    supported_results = results.loc[
        (
            results[
                "directional_FDR_evidence_class"
            ].ne(
                "not_FDR_supported"
            )
            |
            results[
                "global_maxT_evidence_class"
            ].ne(
                "not_global_maxT_supported"
            )
        )
    ].copy()

    supported_results.to_csv(
        SUPPORTED_OUTPUT,
        sep="\t",
        index=False,
    )

    primary_positive_fdr = int(
        primary_results[
            "directional_FDR_evidence_class"
        ].eq(
            "positive_FDR_supported"
        ).sum()
    )

    primary_negative_fdr = int(
        primary_results[
            "directional_FDR_evidence_class"
        ].eq(
            "negative_FDR_supported"
        ).sum()
    )

    primary_positive_global_maxt = int(
        primary_results[
            "global_maxT_evidence_class"
        ].eq(
            "positive_global_maxT_supported"
        ).sum()
    )

    primary_negative_global_maxt = int(
        primary_results[
            "global_maxT_evidence_class"
        ].eq(
            "negative_global_maxT_supported"
        ).sum()
    )

    all_results_finite = bool(
        np.isfinite(
            results[
                [
                    "observed_score",
                    "null_mean",
                    "null_standard_deviation",
                    "empirical_z",
                    "empirical_p_upper",
                    "empirical_p_lower",
                    "empirical_p_two_sided",
                    "maxT_p_upper_within_axis",
                    "maxT_p_lower_within_axis",
                    "maxT_p_two_sided_within_axis",
                    "maxT_p_upper_within_scope_all_axes",
                    "maxT_p_lower_within_scope_all_axes",
                    "maxT_p_two_sided_within_scope_all_axes",
                ]
            ].to_numpy(
                dtype=float
            )
        ).all()
    )

    completion = pd.DataFrame(
        [
            {
                "source_signatures":
                    len(
                        metadata
                    ),

                "source_genes":
                    matrix.shape[
                        1
                    ],

                "landmark_genes":
                    int(
                        landmark_mask.sum()
                    ),

                "matched_chemicals":
                    metadata[
                        "chemical_query"
                    ].nunique(),

                "primary_analysis_scope":
                    PRIMARY_SCOPE,

                "primary_scope_profiles":
                    len(
                        primary_profiles
                    ),

                "total_aggregated_profiles":
                    len(
                        profile_metadata
                    ),

                "analysis_scopes":
                    profile_metadata[
                        "analysis_scope"
                    ].nunique(),

                "neural_lineage_chemicals":
                    profile_metadata.loc[
                        profile_metadata[
                            "analysis_scope"
                        ].eq(
                            "neural_lineage_cell_balanced"
                        ),
                        "chemical_query",
                    ].nunique(),

                "high_quality_flag_detected":
                    high_quality_column
                    is not None,

                "high_quality_flag_column":
                    (
                        high_quality_column
                        if high_quality_column
                        else ""
                    ),

                "gene_permutation_strata":
                    len(
                        unique_strata
                    ),

                "mean_response_bins_per_landmark_class":
                    number_of_mean_bins,

                "variability_bins_per_landmark_class":
                    number_of_sd_bins,

                "empirical_permutations":
                    N_PERMUTATIONS,

                "random_seed":
                    RANDOM_SEED,

                "shared_within_stratum_gene_permutation":
                    True,

                "gene_set_overlap_preserved":
                    True,

                "mechanism_leave_program_out":
                    True,

                "axes_tested":
                    len(
                        AXIS_NAMES
                    ),

                "primary_positive_FDR_supported_tests":
                    primary_positive_fdr,

                "primary_negative_FDR_supported_tests":
                    primary_negative_fdr,

                "primary_positive_global_maxT_supported_tests":
                    primary_positive_global_maxt,

                "primary_negative_global_maxT_supported_tests":
                    primary_negative_global_maxt,

                "all_empirical_results_finite":
                    all_results_finite,

                "inferential_significance_calculated":
                    True,

                "Phase6E3B_status":
                    (
                        "completed"
                        if all_results_finite
                        else "failed_validation"
                    ),
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
            "===== PHASE 6E3B COMPLETION =====",
            completion.to_string(
                index=False
            ),
            "",
            "===== PRIMARY CELL-BALANCED RESULTS =====",
            primary_results[
                [
                    "chemical_query",
                    "evidence_tier",
                    "axis_name",
                    "observed_score",
                    "empirical_z",
                    "empirical_p_upper",
                    "empirical_p_lower",
                    "empirical_p_two_sided",
                    upper_fdr_column,
                    lower_fdr_column,
                    "maxT_p_two_sided_within_axis",
                    "maxT_p_two_sided_within_scope_all_axes",
                    "directional_FDR_evidence_class",
                    "global_maxT_evidence_class",
                ]
            ].to_string(
                index=False
            ),
            "",
            "===== EMPIRICALLY SUPPORTED RESULTS =====",
            (
                supported_results[
                    [
                        "analysis_scope",
                        "chemical_query",
                        "axis_name",
                        "observed_score",
                        "empirical_z",
                        "directional_FDR_evidence_class",
                        "global_maxT_evidence_class",
                    ]
                ].to_string(
                    index=False
                )
                if not supported_results.empty
                else "No empirically supported results."
            ),
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

    if not all_results_finite:
        raise RuntimeError(
            "Phase 6E3B empirical-result validation failed."
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            (
                "Phase 6E3B failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
