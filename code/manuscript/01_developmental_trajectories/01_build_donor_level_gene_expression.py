#!/usr/bin/env python3

from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd


PROJECT = Path(
    "."
)

EXPRESSION_FILE = (
    PROJECT
    / "03_processed_data/transcriptomics/BrainSpan/"
      "brainspan_cortical_expression_gene_symbol.tsv.gz"
)

SAMPLE_METADATA_FILE = (
    PROJECT
    / "03_processed_data/transcriptomics/BrainSpan/"
      "brainspan_cortical_sample_metadata.tsv"
)

DONOR_METADATA_FILE = (
    PROJECT
    / "03_processed_data/developmental_trajectory/phase5B2/"
      "phase5B2_donor_level_module_matrix.tsv"
)

OUT_DIR = (
    PROJECT
    / "03_processed_data/developmental_trajectory/phase5D"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase5"
)

META_DIR = (
    PROJECT
    / "02_metadata/phase5"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase5/"
      "phase5D1_donor_level_gene_expression.log"
)

for directory in [
    OUT_DIR,
    TABLE_DIR,
    META_DIR,
    LOG_FILE.parent,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

LOG_FILE.write_text("")


def log(message=""):
    print(message, flush=True)

    with LOG_FILE.open("a") as handle:
        handle.write(
            str(message) + "\n"
        )


def normalize_gene_symbol(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return value.upper()


for required_file in [
    EXPRESSION_FILE,
    SAMPLE_METADATA_FILE,
    DONOR_METADATA_FILE,
]:
    if not required_file.exists():
        raise FileNotFoundError(
            f"Missing required input: {required_file}"
        )


log("===== Phase 5D1 started =====")
log(
    f"Expression file: "
    f"{EXPRESSION_FILE.relative_to(PROJECT)}"
)


# ============================================================
# Read sample and donor metadata
# ============================================================

sample_metadata = pd.read_csv(
    SAMPLE_METADATA_FILE,
    sep="\t",
    dtype=str
)

required_sample_columns = {
    "sample_id",
    "donor_clean",
    "structure_acronym",
}

missing_sample_columns = (
    required_sample_columns.difference(
        sample_metadata.columns
    )
)

if missing_sample_columns:
    raise ValueError(
        "Missing sample metadata columns: "
        + ", ".join(
            sorted(missing_sample_columns)
        )
    )


sample_metadata[
    "sample_id"
] = sample_metadata[
    "sample_id"
].astype(str).str.strip()

sample_metadata[
    "donor_clean"
] = sample_metadata[
    "donor_clean"
].astype(str).str.strip()


if sample_metadata[
    "sample_id"
].duplicated().any():
    duplicated = sample_metadata.loc[
        sample_metadata[
            "sample_id"
        ].duplicated(
            keep=False
        ),
        "sample_id",
    ].tolist()

    raise ValueError(
        "Duplicated sample IDs in metadata: "
        + ", ".join(
            duplicated[:20]
        )
    )


donor_metadata = pd.read_csv(
    DONOR_METADATA_FILE,
    sep="\t",
    low_memory=False
)

required_donor_columns = {
    "donor_clean",
    "sex_clean",
    "developmental_window",
    "developmental_age_years_from_birth",
    "developmental_time_transformed",
}

missing_donor_columns = (
    required_donor_columns.difference(
        donor_metadata.columns
    )
)

if missing_donor_columns:
    raise ValueError(
        "Missing donor metadata columns: "
        + ", ".join(
            sorted(missing_donor_columns)
        )
    )


donor_metadata[
    "donor_clean"
] = donor_metadata[
    "donor_clean"
].astype(str).str.strip()

donor_metadata = donor_metadata.drop_duplicates(
    subset="donor_clean"
).copy()


# ============================================================
# Read cortical gene-expression matrix
# ============================================================

log("Reading cortical gene-expression matrix...")

try:
    expression = pd.read_csv(
        EXPRESSION_FILE,
        sep="\t",
        index_col=0,
        compression="gzip",
        dtype=np.float32,
    )

except Exception:
    expression = pd.read_csv(
        EXPRESSION_FILE,
        sep="\t",
        index_col=0,
        compression="gzip",
        low_memory=False,
    )

    expression = expression.apply(
        pd.to_numeric,
        errors="coerce"
    ).astype(
        np.float32
    )


expression.columns = (
    expression.columns
    .astype(str)
    .str.strip()
)


log(
    f"Raw matrix dimensions: "
    f"{expression.shape[0]} genes × "
    f"{expression.shape[1]} samples"
)


if expression.columns.duplicated().any():
    duplicated_columns = expression.columns[
        expression.columns.duplicated(
            keep=False
        )
    ].tolist()

    raise ValueError(
        "Duplicated expression sample columns: "
        + ", ".join(
            duplicated_columns[:20]
        )
    )


metadata_samples = set(
    sample_metadata[
        "sample_id"
    ]
)

expression_samples = set(
    expression.columns
)

missing_from_metadata = sorted(
    expression_samples.difference(
        metadata_samples
    )
)

missing_from_expression = sorted(
    metadata_samples.difference(
        expression_samples
    )
)


if missing_from_metadata:
    raise ValueError(
        "Expression samples missing from metadata: "
        + ", ".join(
            missing_from_metadata[:20]
        )
    )


if missing_from_expression:
    raise ValueError(
        "Metadata samples missing from expression matrix: "
        + ", ".join(
            missing_from_expression[:20]
        )
    )


sample_metadata_indexed = sample_metadata.set_index(
    "sample_id",
    drop=True,
)

sample_metadata_indexed = sample_metadata_indexed.reindex(
    expression.columns
)

sample_metadata_indexed.index.name = "sample_id"

sample_metadata = sample_metadata_indexed.reset_index()

if "sample_id" not in sample_metadata.columns:
    raise RuntimeError(
        "Sample alignment failed to restore the sample_id column."
    )


# ============================================================
# Normalize and collapse duplicate gene symbols
# ============================================================

original_symbols = pd.Series(
    expression.index.astype(str),
    name="original_gene_symbol"
)

normalized_symbols = original_symbols.map(
    normalize_gene_symbol
)

valid_symbol_mask = normalized_symbols.notna()

invalid_symbol_count = int(
    (~valid_symbol_mask).sum()
)


duplicate_frame = pd.DataFrame({
    "original_gene_symbol":
        original_symbols.values,

    "normalized_gene_symbol":
        normalized_symbols.values,
})

duplicate_counts = (
    duplicate_frame[
        "normalized_gene_symbol"
    ]
    .value_counts(
        dropna=True
    )
)

duplicated_normalized_symbols = set(
    duplicate_counts[
        duplicate_counts > 1
    ].index
)

duplicate_audit = (
    duplicate_frame.loc[
        duplicate_frame[
            "normalized_gene_symbol"
        ].isin(
            duplicated_normalized_symbols
        )
    ]
    .groupby(
        "normalized_gene_symbol",
        as_index=False
    )
    .agg(
        n_original_rows=(
            "original_gene_symbol",
            "size",
        ),

        original_symbols=(
            "original_gene_symbol",
            lambda values: ";".join(
                sorted(
                    set(
                        map(
                            str,
                            values
                        )
                    )
                )
            ),
        ),
    )
)


duplicate_audit.to_csv(
    TABLE_DIR
    / "phase5D1_duplicate_gene_symbol_audit.tsv",
    sep="\t",
    index=False
)


expression = expression.loc[
    valid_symbol_mask.values
].copy()

expression.index = normalized_symbols.loc[
    valid_symbol_mask
].values


raw_valid_gene_rows = int(
    expression.shape[0]
)


if expression.index.duplicated().any():
    expression = (
        expression.groupby(
            level=0,
            sort=True
        )
        .mean()
        .astype(
            np.float32
        )
    )


log(
    f"Gene rows after symbol normalization: "
    f"{raw_valid_gene_rows}"
)

log(
    f"Unique normalized genes: "
    f"{expression.shape[0]}"
)

log(
    f"Duplicated normalized symbols collapsed: "
    f"{len(duplicate_audit)}"
)


# ============================================================
# Sample-to-donor alignment
# ============================================================

sample_alignment = sample_metadata[
    [
        "sample_id",
        "donor_clean",
        "structure_acronym",
    ]
].copy()


sample_alignment[
    "expression_column_order"
] = np.arange(
    1,
    len(sample_alignment) + 1
)


donor_sample_counts = (
    sample_alignment.groupby(
        "donor_clean",
        as_index=False
    )
    .agg(
        n_cortical_samples=(
            "sample_id",
            "size",
        ),

        n_cortical_regions=(
            "structure_acronym",
            "nunique",
        ),

        cortical_regions=(
            "structure_acronym",
            lambda values: ";".join(
                sorted(
                    set(
                        map(
                            str,
                            values
                        )
                    )
                )
            ),
        ),
    )
)


sample_alignment.to_csv(
    TABLE_DIR
    / "phase5D1_sample_to_donor_alignment.tsv",
    sep="\t",
    index=False
)


sample_donors = set(
    donor_sample_counts[
        "donor_clean"
    ]
)

metadata_donors = set(
    donor_metadata[
        "donor_clean"
    ]
)


missing_donor_metadata = sorted(
    sample_donors.difference(
        metadata_donors
    )
)

if missing_donor_metadata:
    raise ValueError(
        "Donors missing from Phase 5B2 metadata: "
        + ", ".join(
            missing_donor_metadata
        )
    )


donor_metadata = donor_metadata.merge(
    donor_sample_counts,
    on="donor_clean",
    how="inner",
    suffixes=(
        "",
        "_phase5D1"
    ),
    validate="one_to_one",
)


donor_metadata = donor_metadata.sort_values(
    [
        "developmental_time_transformed",
        "donor_clean",
    ]
).reset_index(
    drop=True
)


donor_order = donor_metadata[
    "donor_clean"
].tolist()


# ============================================================
# Average cortical regions within each donor
# ============================================================

log(
    f"Aggregating expression across "
    f"{len(donor_order)} independent donors..."
)


donor_expression = pd.DataFrame(
    index=expression.index
)


sample_metadata_indexed = (
    sample_metadata.set_index(
        "sample_id"
    )
)


for donor in donor_order:
    donor_samples = sample_metadata_indexed.index[
        sample_metadata_indexed[
            "donor_clean"
        ] == donor
    ].tolist()

    if not donor_samples:
        raise ValueError(
            f"No expression samples found for donor "
            f"{donor}"
        )

    donor_expression[
        donor
    ] = expression[
        donor_samples
    ].mean(
        axis=1,
        skipna=True
    ).astype(
        np.float32
    )


donor_expression.index.name = (
    "gene_symbol"
)


donor_expression.to_csv(
    OUT_DIR
    / "phase5D1_donor_mean_gene_expression.tsv.gz",
    sep="\t",
    compression="gzip",
    float_format="%.7g",
)


donor_metadata.to_csv(
    OUT_DIR
    / "phase5D1_donor_metadata.tsv",
    sep="\t",
    index=False
)


log(
    f"Donor-level matrix dimensions: "
    f"{donor_expression.shape[0]} genes × "
    f"{donor_expression.shape[1]} donors"
)


# ============================================================
# Expression-scale audit
# ============================================================

array = donor_expression.to_numpy(
    dtype=np.float64
)

finite_values = array[
    np.isfinite(array)
]


quantile_probabilities = np.array([
    0.00,
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00,
])

quantile_values = np.quantile(
    finite_values,
    quantile_probabilities
)


scale_audit = pd.DataFrame({
    "quantile":
        quantile_probabilities,

    "expression_value":
        quantile_values,
})


scale_audit.to_csv(
    TABLE_DIR
    / "phase5D1_expression_scale_quantiles.tsv",
    sep="\t",
    index=False
)


global_scale_summary = {
    "finite_values":
        int(
            finite_values.size
        ),

    "fraction_exact_zero":
        float(
            np.mean(
                finite_values == 0
            )
        ),

    "fraction_negative":
        float(
            np.mean(
                finite_values < 0
            )
        ),

    "fraction_greater_than_one":
        float(
            np.mean(
                finite_values > 1
            )
        ),

    "global_minimum":
        float(
            np.min(
                finite_values
            )
        ),

    "global_median":
        float(
            np.median(
                finite_values
            )
        ),

    "global_maximum":
        float(
            np.max(
                finite_values
            )
        ),
}


# ============================================================
# Per-gene QC
# ============================================================

log("Calculating gene-level quality metrics...")

finite_mask = np.isfinite(
    array
)

n_finite = finite_mask.sum(
    axis=1
)

with warnings.catch_warnings():
    warnings.simplefilter(
        "ignore",
        category=RuntimeWarning
    )

    gene_mean = np.nanmean(
        array,
        axis=1
    )

    gene_median = np.nanmedian(
        array,
        axis=1
    )

    gene_sd = np.nanstd(
        array,
        axis=1,
        ddof=1
    )

    gene_minimum = np.nanmin(
        array,
        axis=1
    )

    gene_maximum = np.nanmax(
        array,
        axis=1
    )

    gene_q25 = np.nanquantile(
        array,
        0.25,
        axis=1
    )

    gene_q75 = np.nanquantile(
        array,
        0.75,
        axis=1
    )

    gene_mad = np.nanmedian(
        np.abs(
            array -
            gene_median[:, None]
        ),
        axis=1
    )


finite_denominator = np.maximum(
    n_finite,
    1
)

fraction_above_zero = (
    (
        (array > 0) &
        finite_mask
    ).sum(
        axis=1
    )
    /
    finite_denominator
)

fraction_above_one = (
    (
        (array > 1) &
        finite_mask
    ).sum(
        axis=1
    )
    /
    finite_denominator
)


unique_value_count = donor_expression.nunique(
    axis=1,
    dropna=True
).to_numpy()


minimum_required_donors = math.ceil(
    donor_expression.shape[1] *
    0.90
)


gene_qc = pd.DataFrame({
    "gene_symbol":
        donor_expression.index,

    "n_finite_donors":
        n_finite,

    "fraction_finite_donors":
        n_finite /
        donor_expression.shape[1],

    "mean_expression":
        gene_mean,

    "median_expression":
        gene_median,

    "standard_deviation":
        gene_sd,

    "median_absolute_deviation":
        gene_mad,

    "minimum_expression":
        gene_minimum,

    "maximum_expression":
        gene_maximum,

    "expression_range":
        gene_maximum -
        gene_minimum,

    "q25":
        gene_q25,

    "q75":
        gene_q75,

    "interquartile_range":
        gene_q75 -
        gene_q25,

    "fraction_above_zero":
        fraction_above_zero,

    "fraction_above_one":
        fraction_above_one,

    "unique_donor_values":
        unique_value_count,
})


gene_qc[
    "coverage_pass"
] = (
    gene_qc[
        "n_finite_donors"
    ] >= minimum_required_donors
)


gene_qc[
    "nonconstant_pass"
] = (
    gene_qc[
        "standard_deviation"
    ] > 0
) & (
    gene_qc[
        "unique_donor_values"
    ] >= 5
)


gene_qc[
    "detectable_pass"
] = (
    gene_qc[
        "maximum_expression"
    ] > 0
)


gene_qc[
    "basic_trajectory_eligible"
] = (
    gene_qc[
        "coverage_pass"
    ]
    &
    gene_qc[
        "nonconstant_pass"
    ]
    &
    gene_qc[
        "detectable_pass"
    ]
)


eligible_sd = gene_qc.loc[
    gene_qc[
        "basic_trajectory_eligible"
    ],
    "standard_deviation",
]


if len(eligible_sd) > 0:
    sd_median = float(
        eligible_sd.median()
    )

    sd_q75 = float(
        eligible_sd.quantile(
            0.75
        )
    )

else:
    sd_median = np.nan
    sd_q75 = np.nan


gene_qc[
    "above_eligible_SD_median"
] = (
    gene_qc[
        "basic_trajectory_eligible"
    ]
    &
    (
        gene_qc[
            "standard_deviation"
        ] >= sd_median
    )
)


gene_qc[
    "above_eligible_SD_q75"
] = (
    gene_qc[
        "basic_trajectory_eligible"
    ]
    &
    (
        gene_qc[
            "standard_deviation"
        ] >= sd_q75
    )
)


gene_qc[
    "SD_percentile_among_eligible"
] = np.nan


eligible_indices = gene_qc[
    "basic_trajectory_eligible"
]


gene_qc.loc[
    eligible_indices,
    "SD_percentile_among_eligible",
] = (
    gene_qc.loc[
        eligible_indices,
        "standard_deviation",
    ]
    .rank(
        method="average",
        pct=True
    )
)


gene_qc = gene_qc.sort_values(
    [
        "basic_trajectory_eligible",
        "standard_deviation",
    ],
    ascending=[
        False,
        False,
    ]
)


gene_qc.to_csv(
    TABLE_DIR
    / "phase5D1_gene_quality_metrics.tsv.gz",
    sep="\t",
    index=False,
    compression="gzip",
)


basic_gene_list = gene_qc.loc[
    gene_qc[
        "basic_trajectory_eligible"
    ],
    "gene_symbol",
]


basic_gene_list.to_csv(
    OUT_DIR
    / "phase5D1_basic_trajectory_eligible_genes.txt",
    index=False,
    header=False,
)


# ============================================================
# Completion summary
# ============================================================

summary = {
    "raw_expression_gene_rows":
        int(
            len(
                original_symbols
            )
        ),

    "invalid_gene_symbols_removed":
        invalid_symbol_count,

    "valid_gene_rows_before_collapsing":
        raw_valid_gene_rows,

    "unique_normalized_genes":
        int(
            donor_expression.shape[0]
        ),

    "duplicate_symbol_groups_collapsed":
        int(
            len(
                duplicate_audit
            )
        ),

    "cortical_samples_aligned":
        int(
            expression.shape[1]
        ),

    "independent_donors":
        int(
            donor_expression.shape[1]
        ),

    "minimum_samples_per_donor":
        int(
            donor_sample_counts[
                "n_cortical_samples"
            ].min()
        ),

    "maximum_samples_per_donor":
        int(
            donor_sample_counts[
                "n_cortical_samples"
            ].max()
        ),

    "minimum_required_finite_donors":
        int(
            minimum_required_donors
        ),

    "basic_trajectory_eligible_genes":
        int(
            gene_qc[
                "basic_trajectory_eligible"
            ].sum()
        ),

    "genes_above_eligible_SD_median":
        int(
            gene_qc[
                "above_eligible_SD_median"
            ].sum()
        ),

    "genes_above_eligible_SD_q75":
        int(
            gene_qc[
                "above_eligible_SD_q75"
            ].sum()
        ),

    "eligible_gene_SD_median":
        sd_median,

    "eligible_gene_SD_q75":
        sd_q75,

    "minimum_developmental_age":
        float(
            donor_metadata[
                "developmental_age_years_from_birth"
            ].min()
        ),

    "maximum_developmental_age":
        float(
            donor_metadata[
                "developmental_age_years_from_birth"
            ].max()
        ),

    "Phase5D1_status":
        "completed",
}


summary.update(
    global_scale_summary
)


pd.DataFrame([
    summary
]).to_csv(
    TABLE_DIR
    / "phase5D1_completion_summary.tsv",
    sep="\t",
    index=False,
)


with (
    META_DIR
    / "phase5D1_donor_level_expression_summary.json"
).open("w") as handle:
    json.dump(
        summary,
        handle,
        indent=2
    )


log("")
log("===== Phase 5D1 completed =====")
log(
    json.dumps(
        summary,
        indent=2
    )
)

log("")
log("Expression-scale quantiles:")
log(
    scale_audit.to_string(
        index=False
    )
)

log("")
log("Top 20 variable eligible genes:")
log(
    gene_qc.loc[
        gene_qc[
            "basic_trajectory_eligible"
        ],
        [
            "gene_symbol",
            "mean_expression",
            "standard_deviation",
            "median_absolute_deviation",
            "fraction_above_zero",
        ],
    ]
    .head(20)
    .to_string(
        index=False
    )
)
