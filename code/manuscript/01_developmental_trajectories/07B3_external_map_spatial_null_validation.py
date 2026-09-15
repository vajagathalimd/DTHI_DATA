#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from neuromaps.nulls import alexander_bloch


PROJECT = Path(
    "."
)

RAW_MAP_DIR = (
    PROJECT
    / "01_raw_data/external_cortical_maps/neuromaps"
)

SAMPLE_SCORE_FILE = (
    PROJECT
    / "03_processed_data/developmental_trajectory/phase5D7/"
      "phase5D7B2_AHBA_sample_program_scores.tsv.gz"
)

EXTERNAL_MAP_FILE = (
    PROJECT
    / "03_processed_data/spatial_hierarchy/phase3C/"
      "Schaefer100_external_maps_wide.tsv"
)

LEFT_PARCELLATION = (
    PROJECT
    / "03_processed_data/spatial_hierarchy/phase3C/"
      "Schaefer100_fsLR32k_hemi-L.label.gii"
)

RIGHT_PARCELLATION = (
    PROJECT
    / "03_processed_data/spatial_hierarchy/phase3C/"
      "Schaefer100_fsLR32k_hemi-R.label.gii"
)

PROCESSED_DIR = (
    PROJECT
    / "03_processed_data/developmental_trajectory/phase5D7"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase5"
)

FIGURE_DIR = (
    PROJECT
    / "06_figures/main_figures/phase5"
)

SOURCE_DIR = (
    PROJECT
    / "06_figures/source_data/phase5"
)

METADATA_DIR = (
    PROJECT
    / "02_metadata/phase5"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase5/"
      "phase5D7B3_external_map_spatial_null.log"
)

for directory in [
    PROCESSED_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    SOURCE_DIR,
    METADATA_DIR,
    LOG_FILE.parent,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

os.environ[
    "NEUROMAPS_DATA"
] = str(
    RAW_MAP_DIR
)

N_PERMUTATIONS = 10000
RANDOM_SEED = 20260720

PROGRAMS = [
    "maturation_high_score_z",
    "fetal_high_score_z",
    "maturation_minus_fetal_balance",
]

PRIMARY_MAPS = [
    "margulies_fc_gradient1_association_oriented",
    "sydnor_sensory_association_axis_association_oriented",
    "hcp_t1w_t2w_myelin_association_oriented",
]

SECONDARY_MAPS = [
    "hcp_cortical_thickness_association_oriented",
]

ALL_MAPS = (
    PRIMARY_MAPS
    + SECONDARY_MAPS
)

PRETTY_PROGRAMS = {
    "maturation_high_score_z":
        "Maturation-high program",

    "fetal_high_score_z":
        "Fetal-high program",

    "maturation_minus_fetal_balance":
        "Maturation minus fetal balance",
}

PRETTY_MAPS = {
    "margulies_fc_gradient1_association_oriented":
        "Margulies FC gradient",

    "sydnor_sensory_association_axis_association_oriented":
        "Sydnor sensory–association axis",

    "hcp_t1w_t2w_myelin_association_oriented":
        "Inverse HCP myelin",

    "hcp_cortical_thickness_association_oriented":
        "HCP cortical thickness",
}


def log(message: str = "") -> None:
    print(
        message,
        flush=True,
    )

    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            str(message) + "\n"
        )


def normalize_id(
    value: object,
) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_parcel_id(
    value: object,
) -> int | None:
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    try:
        number = float(text)

        if not number.is_integer():
            return None

        integer = int(
            number
        )

        if integer < 1 or integer > 100:
            return None

        return integer

    except ValueError:
        return None


def parse_boolean(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.fillna(
            False
        )

    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "t",
                "1",
                "yes",
                "y",
            ]
        )
    )


def bh_adjust(
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=float,
    )

    adjusted = np.full(
        values.shape,
        np.nan,
        dtype=float,
    )

    valid_indices = np.where(
        np.isfinite(
            values
        )
    )[0]

    if len(valid_indices) == 0:
        return adjusted

    valid_values = values[
        valid_indices
    ]

    ordering = np.argsort(
        valid_values
    )

    ordered_values = valid_values[
        ordering
    ]

    number_of_tests = len(
        ordered_values
    )

    corrected = (
        ordered_values
        * number_of_tests
        / np.arange(
            1,
            number_of_tests + 1,
        )
    )

    corrected = np.minimum.accumulate(
        corrected[::-1]
    )[::-1]

    corrected = np.clip(
        corrected,
        0,
        1,
    )

    restored = np.empty_like(
        corrected
    )

    restored[
        ordering
    ] = corrected

    adjusted[
        valid_indices
    ] = restored

    return adjusted


def test_family(
    external_map: str,
) -> str:
    if external_map in PRIMARY_MAPS:
        return "primary"

    return "secondary"


def vectorized_spearman_null_test(
    observed_map: np.ndarray,
    target_map: np.ndarray,
    null_maps: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    observed_map = np.asarray(
        observed_map,
        dtype=float,
    )

    target_map = np.asarray(
        target_map,
        dtype=float,
    )

    null_maps = np.asarray(
        null_maps,
        dtype=float,
    )

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    valid = (
        mask
        & np.isfinite(
            observed_map
        )
        & np.isfinite(
            target_map
        )
    )

    observed_values = observed_map[
        valid
    ]

    target_values = target_map[
        valid
    ]

    if len(
        observed_values
    ) < 10:
        return {
            "n_parcels":
                len(
                    observed_values
                ),

            "spearman_r":
                np.nan,

            "parametric_p":
                np.nan,

            "spatial_p":
                np.nan,

            "null_mean_r":
                np.nan,

            "null_sd_r":
                np.nan,

            "null_lower_025":
                np.nan,

            "null_upper_975":
                np.nan,
        }

    observed_test = spearmanr(
        observed_values,
        target_values,
    )

    observed_r = float(
        observed_test.statistic
    )

    parametric_p = float(
        observed_test.pvalue
    )

    null_subset = null_maps[
        valid,
        :
    ]

    target_ranks = rankdata(
        target_values
    )

    target_centered = (
        target_ranks
        - target_ranks.mean()
    )

    target_norm = np.sqrt(
        np.sum(
            target_centered ** 2
        )
    )

    null_ranks = rankdata(
        null_subset,
        axis=0,
    )

    null_centered = (
        null_ranks
        - null_ranks.mean(
            axis=0,
            keepdims=True,
        )
    )

    null_norms = np.sqrt(
        np.sum(
            null_centered ** 2,
            axis=0,
        )
    )

    denominators = (
        null_norms
        * target_norm
    )

    null_correlations = np.divide(
        np.sum(
            null_centered
            * target_centered[
                :,
                None,
            ],
            axis=0,
        ),
        denominators,
        out=np.full(
            denominators.shape,
            np.nan,
            dtype=float,
        ),
        where=denominators > 0,
    )

    finite_nulls = null_correlations[
        np.isfinite(
            null_correlations
        )
    ]

    spatial_p = (
        1
        + np.sum(
            np.abs(
                finite_nulls
            )
            >= abs(
                observed_r
            )
        )
    ) / (
        1
        + len(
            finite_nulls
        )
    )

    return {
        "n_parcels":
            len(
                observed_values
            ),

        "spearman_r":
            observed_r,

        "parametric_p":
            parametric_p,

        "spatial_p":
            spatial_p,

        "null_mean_r":
            float(
                np.mean(
                    finite_nulls
                )
            ),

        "null_sd_r":
            float(
                np.std(
                    finite_nulls
                )
            ),

        "null_lower_025":
            float(
                np.quantile(
                    finite_nulls,
                    0.025,
                )
            ),

        "null_upper_975":
            float(
                np.quantile(
                    finite_nulls,
                    0.975,
                )
            ),
    }


def donor_balanced_parcel_table(
    donor_parcel: pd.DataFrame,
    external_maps: pd.DataFrame,
) -> pd.DataFrame:
    parcel_scores = (
        donor_parcel.groupby(
            [
                "parcel_id",
                "parcel_label",
                "hemisphere",
                "network",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            maturation_high_score_z=(
                "maturation_high_score_z",
                "mean",
            ),

            fetal_high_score_z=(
                "fetal_high_score_z",
                "mean",
            ),

            maturation_minus_fetal_balance=(
                "maturation_minus_fetal_balance",
                "mean",
            ),

            n_donors=(
                "donor_id",
                "nunique",
            ),

            n_samples=(
                "n_samples_in_donor_parcel",
                "sum",
            ),
        )
        .reset_index()
    )

    integrated = external_maps.merge(
        parcel_scores,
        on="parcel_id",
        how="left",
        suffixes=(
            "_external",
            "_AHBA",
        ),
        validate="one_to_one",
    )

    for metadata_column in [
        "parcel_label",
        "hemisphere",
        "network",
    ]:
        external_column = (
            metadata_column
            + "_external"
        )

        ahba_column = (
            metadata_column
            + "_AHBA"
        )

        if (
            external_column
            in integrated.columns
        ):
            integrated[
                metadata_column
            ] = integrated[
                external_column
            ]

        elif (
            metadata_column
            not in integrated.columns
            and ahba_column
            in integrated.columns
        ):
            integrated[
                metadata_column
            ] = integrated[
                ahba_column
            ]

    drop_columns = [
        column
        for column in integrated.columns
        if column.endswith(
            "_external"
        )
        or column.endswith(
            "_AHBA"
        )
    ]

    integrated = integrated.drop(
        columns=drop_columns,
        errors="ignore",
    )

    integrated[
        "n_donors"
    ] = integrated[
        "n_donors"
    ].fillna(
        0
    ).astype(
        int
    )

    integrated[
        "n_samples"
    ] = integrated[
        "n_samples"
    ].fillna(
        0
    ).astype(
        int
    )

    return integrated.sort_values(
        "parcel_id"
    ).reset_index(
        drop=True
    )


def calculate_plain_spearman(
    table: pd.DataFrame,
    program: str,
    external_map: str,
    mask: np.ndarray,
) -> tuple[int, float, float]:
    valid = (
        np.asarray(
            mask,
            dtype=bool,
        )
        & np.isfinite(
            table[
                program
            ].to_numpy(
                dtype=float
            )
        )
        & np.isfinite(
            table[
                external_map
            ].to_numpy(
                dtype=float
            )
        )
    )

    number_of_parcels = int(
        valid.sum()
    )

    if number_of_parcels < 10:
        return (
            number_of_parcels,
            np.nan,
            np.nan,
        )

    test = spearmanr(
        table.loc[
            valid,
            program,
        ].to_numpy(
            dtype=float
        ),
        table.loc[
            valid,
            external_map,
        ].to_numpy(
            dtype=float
        ),
    )

    return (
        number_of_parcels,
        float(
            test.statistic
        ),
        float(
            test.pvalue
        ),
    )


def main() -> None:
    LOG_FILE.write_text(
        "",
        encoding="utf-8",
    )

    log(
        "===== Phase 5D7B3 started ====="
    )

    required_files = [
        SAMPLE_SCORE_FILE,
        EXTERNAL_MAP_FILE,
        LEFT_PARCELLATION,
        RIGHT_PARCELLATION,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(
                str(path)
                for path in missing_files
            )
        )

    scores = pd.read_csv(
        SAMPLE_SCORE_FILE,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    required_score_columns = [
        "sample_id",
        "donor_id",
        "usable_assignment",
        "parcel_id_normalized",
        "parcel_label",
        "hemisphere",
        "network",
    ] + PROGRAMS

    missing_score_columns = [
        column
        for column in required_score_columns
        if column not in scores.columns
    ]

    if missing_score_columns:
        raise ValueError(
            "Sample-score columns missing: "
            + ", ".join(
                missing_score_columns
            )
        )

    scores[
        "sample_id"
    ] = scores[
        "sample_id"
    ].map(
        normalize_id
    )

    scores[
        "donor_id"
    ] = scores[
        "donor_id"
    ].map(
        normalize_id
    )

    scores[
        "usable_assignment"
    ] = parse_boolean(
        scores[
            "usable_assignment"
        ]
    )

    scores[
        "parcel_id"
    ] = scores[
        "parcel_id_normalized"
    ].map(
        normalize_parcel_id
    )

    usable_scores = scores.loc[
        scores[
            "usable_assignment"
        ]
        & scores[
            "parcel_id"
        ].notna()
    ].copy()

    usable_scores[
        "parcel_id"
    ] = usable_scores[
        "parcel_id"
    ].astype(
        int
    )

    for column in PROGRAMS:
        usable_scores[
            column
        ] = pd.to_numeric(
            usable_scores[
                column
            ],
            errors="coerce",
        )

    if usable_scores[
        PROGRAMS
    ].isna().any().any():
        missing_counts = (
            usable_scores[
                PROGRAMS
            ]
            .isna()
            .sum()
        )

        raise ValueError(
            "Missing program scores among usable samples:\n"
            + missing_counts.to_string()
        )

    log(
        "Usable AHBA samples: "
        f"{len(usable_scores)}"
    )

    log(
        "AHBA donors: "
        f"{usable_scores['donor_id'].nunique()}"
    )

    log(
        "Schaefer parcels represented: "
        f"{usable_scores['parcel_id'].nunique()}"
    )

    donor_parcel = (
        usable_scores.groupby(
            [
                "donor_id",
                "parcel_id",
                "parcel_label",
                "hemisphere",
                "network",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            maturation_high_score_z=(
                "maturation_high_score_z",
                "mean",
            ),

            fetal_high_score_z=(
                "fetal_high_score_z",
                "mean",
            ),

            maturation_minus_fetal_balance=(
                "maturation_minus_fetal_balance",
                "mean",
            ),

            n_samples_in_donor_parcel=(
                "sample_id",
                "nunique",
            ),
        )
        .reset_index()
    )

    donor_parcel.to_csv(
        PROCESSED_DIR
        / "phase5D7B3_AHBA_donor_Schaefer100_program_scores.tsv",
        sep="\t",
        index=False,
    )

    external_maps = pd.read_csv(
        EXTERNAL_MAP_FILE,
        sep="\t",
        low_memory=False,
    )

    if "parcel_id" not in external_maps.columns:
        raise ValueError(
            "External-map table lacks parcel_id."
        )

    external_maps[
        "parcel_id"
    ] = external_maps[
        "parcel_id"
    ].map(
        normalize_parcel_id
    )

    if external_maps[
        "parcel_id"
    ].isna().any():
        raise ValueError(
            "Invalid parcel IDs in external-map table."
        )

    external_maps[
        "parcel_id"
    ] = external_maps[
        "parcel_id"
    ].astype(
        int
    )

    if external_maps[
        "parcel_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate external-map parcel IDs."
        )

    external_maps = external_maps.sort_values(
        "parcel_id"
    ).reset_index(
        drop=True
    )

    if external_maps[
        "parcel_id"
    ].tolist() != list(
        range(
            1,
            101,
        )
    ):
        raise ValueError(
            "External-map table is not ordered as parcels 1–100."
        )

    for external_map in ALL_MAPS:
        if external_map not in external_maps.columns:
            raise ValueError(
                f"External map missing: {external_map}"
            )

        external_maps[
            external_map
        ] = pd.to_numeric(
            external_maps[
                external_map
            ],
            errors="coerce",
        )

        if external_maps[
            external_map
        ].isna().any():
            raise ValueError(
                "External map contains missing values: "
                f"{external_map}"
            )

    integrated = donor_balanced_parcel_table(
        donor_parcel,
        external_maps,
    )

    if integrated[
        "parcel_id"
    ].tolist() != list(
        range(
            1,
            101,
        )
    ):
        raise ValueError(
            "Integrated table is not ordered as parcels 1–100."
        )

    for program in PROGRAMS:
        if integrated[
            program
        ].isna().any():
            missing_parcels = integrated.loc[
                integrated[
                    program
                ].isna(),
                "parcel_id",
            ].tolist()

            raise ValueError(
                f"{program} lacks data for parcels: "
                f"{missing_parcels}"
            )

    integrated[
        "AHBA_support_class"
    ] = np.select(
        [
            integrated[
                "n_donors"
            ] >= 4,

            integrated[
                "n_donors"
            ] >= 3,

            integrated[
                "n_donors"
            ] >= 2,

            integrated[
                "n_donors"
            ] >= 1,
        ],
        [
            "high",
            "moderate",
            "limited",
            "minimal",
        ],
        default="none",
    )

    integrated.to_csv(
        PROCESSED_DIR
        / "phase5D7B3_Schaefer100_program_external_maps_integrated.tsv",
        sep="\t",
        index=False,
    )

    analysis_masks = {
        "primary_LH_n3_donors": (
            (
                integrated[
                    "hemisphere"
                ]
                == "LH"
            )
            & (
                integrated[
                    "n_donors"
                ]
                >= 3
            )
        ).to_numpy(),

        "sensitivity_LH_n4_donors": (
            (
                integrated[
                    "hemisphere"
                ]
                == "LH"
            )
            & (
                integrated[
                    "n_donors"
                ]
                >= 4
            )
        ).to_numpy(),

        "sensitivity_bilateral_n2_donors": (
            integrated[
                "n_donors"
            ]
            >= 2
        ).to_numpy(),

        "sensitivity_bilateral_all_parcels": (
            integrated[
                "n_donors"
            ]
            >= 1
        ).to_numpy(),
    }

    subset_rows = []

    for subset_name, mask in analysis_masks.items():
        selected = integrated.loc[
            mask
        ]

        subset_rows.append(
            {
                "analysis_subset":
                    subset_name,

                "n_parcels":
                    int(
                        mask.sum()
                    ),

                "left_parcels":
                    int(
                        (
                            selected[
                                "hemisphere"
                            ]
                            == "LH"
                        ).sum()
                    ),

                "right_parcels":
                    int(
                        (
                            selected[
                                "hemisphere"
                            ]
                            == "RH"
                        ).sum()
                    ),

                "minimum_donors":
                    int(
                        selected[
                            "n_donors"
                        ].min()
                    ),

                "maximum_donors":
                    int(
                        selected[
                            "n_donors"
                        ].max()
                    ),

                "median_donors":
                    float(
                        selected[
                            "n_donors"
                        ].median()
                    ),
            }
        )

    subset_summary = pd.DataFrame(
        subset_rows
    )

    subset_summary.to_csv(
        TABLE_DIR
        / "phase5D7B3_analysis_subset_summary.tsv",
        sep="\t",
        index=False,
    )

    log("")
    log(
        "Analysis subsets:"
    )

    log(
        subset_summary.to_string(
            index=False
        )
    )

    parcellation = (
        str(
            LEFT_PARCELLATION
        ),
        str(
            RIGHT_PARCELLATION
        ),
    )

    null_maps_by_program = {}

    for program_index, program in enumerate(
        PROGRAMS
    ):
        log("")
        log(
            f"Generating {N_PERMUTATIONS:,} "
            f"Alexander-Bloch nulls for {program}..."
        )

        null_maps = alexander_bloch(
            integrated[
                program
            ].to_numpy(
                dtype=float
            ),
            atlas="fsLR",
            density="32k",
            parcellation=parcellation,
            n_perm=N_PERMUTATIONS,
            seed=(
                RANDOM_SEED
                + program_index
            ),
        )

        null_maps = np.asarray(
            null_maps,
            dtype=float,
        )

        if (
            null_maps.shape[0] != 100
            and null_maps.ndim == 2
            and null_maps.shape[1] == 100
        ):
            null_maps = null_maps.T

        if null_maps.shape != (
            100,
            N_PERMUTATIONS,
        ):
            raise ValueError(
                f"Unexpected null shape for {program}: "
                f"{null_maps.shape}"
            )

        null_maps_by_program[
            program
        ] = null_maps

    result_rows = []

    for subset_name, mask in analysis_masks.items():
        for program in PROGRAMS:
            observed_map = integrated[
                program
            ].to_numpy(
                dtype=float
            )

            null_maps = null_maps_by_program[
                program
            ]

            for external_map in ALL_MAPS:
                test_result = (
                    vectorized_spearman_null_test(
                        observed_map=
                            observed_map,

                        target_map=
                            integrated[
                                external_map
                            ].to_numpy(
                                dtype=float
                            ),

                        null_maps=
                            null_maps,

                        mask=
                            mask,
                    )
                )

                result_rows.append(
                    {
                        "analysis_subset":
                            subset_name,

                        "test_family":
                            test_family(
                                external_map
                            ),

                        "developmental_program":
                            program,

                        "program_label":
                            PRETTY_PROGRAMS[
                                program
                            ],

                        "external_map":
                            external_map,

                        "external_map_label":
                            PRETTY_MAPS[
                                external_map
                            ],

                        **test_result,

                        "direction":
                            (
                                "positive"
                                if (
                                    np.isfinite(
                                        test_result[
                                            "spearman_r"
                                        ]
                                    )
                                    and test_result[
                                        "spearman_r"
                                    ] > 0
                                )
                                else (
                                    "negative"
                                    if (
                                        np.isfinite(
                                            test_result[
                                                "spearman_r"
                                            ]
                                        )
                                        and test_result[
                                            "spearman_r"
                                        ] < 0
                                    )
                                    else "none"
                                )
                            ),
                    }
                )

    results = pd.DataFrame(
        result_rows
    )

    results[
        "parametric_fdr_bh"
    ] = (
        results.groupby(
            [
                "analysis_subset",
                "test_family",
            ],
            observed=True,
        )[
            "parametric_p"
        ]
        .transform(
            lambda values: bh_adjust(
                values.to_numpy(
                    dtype=float
                )
            )
        )
    )

    results[
        "spatial_fdr_bh"
    ] = (
        results.groupby(
            [
                "analysis_subset",
                "test_family",
            ],
            observed=True,
        )[
            "spatial_p"
        ]
        .transform(
            lambda values: bh_adjust(
                values.to_numpy(
                    dtype=float
                )
            )
        )
    )

    results[
        "spatial_significant_fdr05"
    ] = (
        results[
            "spatial_fdr_bh"
        ] < 0.05
    )

    results = results.sort_values(
        [
            "analysis_subset",
            "test_family",
            "developmental_program",
            "external_map",
        ]
    ).reset_index(
        drop=True
    )

    results.to_csv(
        TABLE_DIR
        / "phase5D7B3_external_map_spatial_null_results.tsv",
        sep="\t",
        index=False,
    )

    # ========================================================
    # Leave-one-donor-out sensitivity analysis
    # ========================================================

    full_primary = results.loc[
        results[
            "analysis_subset"
        ]
        == "primary_LH_n3_donors"
    ].copy()

    full_primary_lookup = {
        (
            row[
                "developmental_program"
            ],
            row[
                "external_map"
            ],
        ): row[
            "spearman_r"
        ]
        for _, row in full_primary.iterrows()
    }

    donors = sorted(
        donor_parcel[
            "donor_id"
        ].unique()
    )

    leaveout_rows = []

    for excluded_donor in donors:
        reduced_donor_parcel = donor_parcel.loc[
            donor_parcel[
                "donor_id"
            ]
            != excluded_donor
        ].copy()

        reduced_integrated = (
            donor_balanced_parcel_table(
                reduced_donor_parcel,
                external_maps,
            )
        )

        reduced_mask = (
            (
                reduced_integrated[
                    "hemisphere"
                ]
                == "LH"
            )
            & (
                reduced_integrated[
                    "n_donors"
                ]
                >= 3
            )
        ).to_numpy()

        for program in PROGRAMS:
            for external_map in ALL_MAPS:
                (
                    number_of_parcels,
                    correlation,
                    parametric_p,
                ) = calculate_plain_spearman(
                    table=
                        reduced_integrated,

                    program=
                        program,

                    external_map=
                        external_map,

                    mask=
                        reduced_mask,
                )

                full_correlation = (
                    full_primary_lookup[
                        (
                            program,
                            external_map,
                        )
                    ]
                )

                leaveout_rows.append(
                    {
                        "excluded_donor":
                            excluded_donor,

                        "developmental_program":
                            program,

                        "external_map":
                            external_map,

                        "test_family":
                            test_family(
                                external_map
                            ),

                        "n_parcels":
                            number_of_parcels,

                        "spearman_r":
                            correlation,

                        "parametric_p":
                            parametric_p,

                        "full_primary_spearman_r":
                            full_correlation,

                        "same_direction_as_full":
                            (
                                np.sign(
                                    correlation
                                )
                                == np.sign(
                                    full_correlation
                                )
                                if (
                                    np.isfinite(
                                        correlation
                                    )
                                    and np.isfinite(
                                        full_correlation
                                    )
                                )
                                else False
                            ),
                    }
                )

    leaveout = pd.DataFrame(
        leaveout_rows
    )

    leaveout.to_csv(
        TABLE_DIR
        / "phase5D7B3_leave_one_donor_out_correlations.tsv",
        sep="\t",
        index=False,
    )

    leaveout_summary = (
        leaveout.groupby(
            [
                "developmental_program",
                "external_map",
                "test_family",
            ],
            observed=True,
        )
        .agg(
            n_leaveout_donors=(
                "excluded_donor",
                "nunique",
            ),

            minimum_n_parcels=(
                "n_parcels",
                "min",
            ),

            maximum_n_parcels=(
                "n_parcels",
                "max",
            ),

            minimum_spearman_r=(
                "spearman_r",
                "min",
            ),

            maximum_spearman_r=(
                "spearman_r",
                "max",
            ),

            median_spearman_r=(
                "spearman_r",
                "median",
            ),

            same_direction_fraction=(
                "same_direction_as_full",
                "mean",
            ),
        )
        .reset_index()
    )

    leaveout_summary.to_csv(
        TABLE_DIR
        / "phase5D7B3_leave_one_donor_out_summary.tsv",
        sep="\t",
        index=False,
    )

    # ========================================================
    # Figure 44
    # ========================================================

    primary_plot = results.loc[
        results[
            "analysis_subset"
        ]
        == "primary_LH_n3_donors"
    ].copy()

    correlation_matrix = (
        primary_plot.pivot(
            index=
                "developmental_program",

            columns=
                "external_map",

            values=
                "spearman_r",
        )
        .reindex(
            index=
                PROGRAMS,

            columns=
                ALL_MAPS,
        )
    )

    qvalue_matrix = (
        primary_plot.pivot(
            index=
                "developmental_program",

            columns=
                "external_map",

            values=
                "spatial_fdr_bh",
        )
        .reindex(
            index=
                PROGRAMS,

            columns=
                ALL_MAPS,
        )
    )

    figure, axis = plt.subplots(
        figsize=(
            11,
            5.8,
        )
    )

    image = axis.imshow(
        correlation_matrix.to_numpy(
            dtype=float
        ),
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    figure.colorbar(
        image,
        ax=axis,
        label="Spearman correlation",
    )

    axis.set_yticks(
        range(
            len(
                PROGRAMS
            )
        )
    )

    axis.set_yticklabels(
        [
            PRETTY_PROGRAMS[
                program
            ]
            for program in PROGRAMS
        ]
    )

    axis.set_xticks(
        range(
            len(
                ALL_MAPS
            )
        )
    )

    axis.set_xticklabels(
        [
            PRETTY_MAPS[
                external_map
            ]
            for external_map in ALL_MAPS
        ],
        rotation=30,
        ha="right",
    )

    for row_index, program in enumerate(
        PROGRAMS
    ):
        for column_index, external_map in enumerate(
            ALL_MAPS
        ):
            correlation = correlation_matrix.loc[
                program,
                external_map,
            ]

            q_value = qvalue_matrix.loc[
                program,
                external_map,
            ]

            significance = (
                "**"
                if (
                    np.isfinite(
                        q_value
                    )
                    and q_value < 0.01
                )
                else (
                    "*"
                    if (
                        np.isfinite(
                            q_value
                        )
                        and q_value < 0.05
                    )
                    else ""
                )
            )

            axis.text(
                column_index,
                row_index,
                f"{correlation:.2f}{significance}",
                ha="center",
                va="center",
                fontsize=10,
            )

    axis.set_title(
        "Figure 44. Spatial-null validation of developmental "
        "programs in adult cortex"
    )

    axis.set_xlabel(
        "Association-oriented external cortical map"
    )

    axis.set_ylabel(
        "BrainSpan-derived developmental program"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURE_DIR
        / "Figure44_developmental_program_external_map_spatial_null.pdf",
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "Figure44_developmental_program_external_map_spatial_null.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    primary_plot.to_csv(
        SOURCE_DIR
        / "Figure44_source_developmental_program_external_map_spatial_null.tsv",
        sep="\t",
        index=False,
    )

    primary_results = results.loc[
        (
            results[
                "analysis_subset"
            ]
            == "primary_LH_n3_donors"
        )
        & (
            results[
                "test_family"
            ]
            == "primary"
        )
    ]

    secondary_results = results.loc[
        (
            results[
                "analysis_subset"
            ]
            == "primary_LH_n3_donors"
        )
        & (
            results[
                "test_family"
            ]
            == "secondary"
        )
    ]

    completion = pd.DataFrame(
        [
            {
                "usable_AHBA_samples":
                    len(
                        usable_scores
                    ),

                "AHBA_donors":
                    donor_parcel[
                        "donor_id"
                    ].nunique(),

                "donor_parcel_rows":
                    len(
                        donor_parcel
                    ),

                "represented_parcels":
                    int(
                        (
                            integrated[
                                "n_donors"
                            ]
                            >= 1
                        ).sum()
                    ),

                "primary_LH_n3_parcels":
                    int(
                        analysis_masks[
                            "primary_LH_n3_donors"
                        ].sum()
                    ),

                "spatial_null_permutations":
                    N_PERMUTATIONS,

                "minimum_possible_spatial_p":
                    (
                        1
                        / (
                            N_PERMUTATIONS
                            + 1
                        )
                    ),

                "primary_tests":
                    len(
                        primary_results
                    ),

                "primary_spatial_FDR_significant":
                    int(
                        primary_results[
                            "spatial_significant_fdr05"
                        ].sum()
                    ),

                "secondary_tests":
                    len(
                        secondary_results
                    ),

                "secondary_spatial_FDR_significant":
                    int(
                        secondary_results[
                            "spatial_significant_fdr05"
                        ].sum()
                    ),

                "leave_one_donor_out_donors":
                    len(
                        donors
                    ),

                "figure44_generated":
                    (
                        FIGURE_DIR
                        / "Figure44_developmental_program_external_map_spatial_null.pdf"
                    ).exists(),

                "Phase5D7B3_status":
                    "completed",
            }
        ]
    )

    completion.to_csv(
        TABLE_DIR
        / "phase5D7B3_completion_summary.tsv",
        sep="\t",
        index=False,
    )

    report_file = (
        METADATA_DIR
        / "PHASE5D7B3_EXTERNAL_MAP_SPATIAL_NULL_REPORT.txt"
    )

    with report_file.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "Phase 5D7B3 external cortical-map validation\n\n"
        )

        handle.write(
            "Primary subset: left-hemisphere Schaefer100 "
            "parcels represented by at least three AHBA donors.\n"
        )

        handle.write(
            "Spatial null: Alexander-Bloch surface rotation.\n"
        )

        handle.write(
            f"Permutations: {N_PERMUTATIONS}\n\n"
        )

        handle.write(
            "Completion summary\n"
        )

        handle.write(
            completion.to_string(
                index=False
            )
        )

        handle.write(
            "\n\nPrimary results\n"
        )

        handle.write(
            primary_plot.to_string(
                index=False
            )
        )

        handle.write(
            "\n\nLeave-one-donor-out summary\n"
        )

        handle.write(
            leaveout_summary.to_string(
                index=False
            )
        )

        handle.write("\n")

    log("")
    log(
        "===== Phase 5D7B3 completed ====="
    )

    log(
        completion.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        log(
            f"Phase 5D7B3 failed: {error}"
        )

        raise
