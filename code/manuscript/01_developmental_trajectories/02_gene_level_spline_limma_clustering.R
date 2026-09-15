#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(data.table)
    library(limma)
    library(ggplot2)
    library(cluster)
    library(splines)
})

PROJECT <- "."

EXPRESSION_FILE <- file.path(
    PROJECT,
    "03_processed_data/developmental_trajectory/phase5D",
    "phase5D1_donor_mean_gene_expression.tsv.gz"
)

METADATA_FILE <- file.path(
    PROJECT,
    "03_processed_data/developmental_trajectory/phase5D",
    "phase5D1_donor_metadata.tsv"
)

OUT_DIR <- file.path(
    PROJECT,
    "03_processed_data/developmental_trajectory/phase5D"
)

MODEL_DIR <- file.path(
    OUT_DIR,
    "models"
)

TABLE_DIR <- file.path(
    PROJECT,
    "07_tables/main_tables/phase5"
)

FIGURE_DIR <- file.path(
    PROJECT,
    "06_figures/main_figures/phase5"
)

SOURCE_DIR <- file.path(
    PROJECT,
    "06_figures/source_data/phase5"
)

META_DIR <- file.path(
    PROJECT,
    "02_metadata/phase5"
)

LOG_FILE <- file.path(
    PROJECT,
    "09_pipeline_logs/phase5",
    "phase5D2_gene_level_spline_limma.log"
)

for (directory in c(
    OUT_DIR,
    MODEL_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    SOURCE_DIR,
    META_DIR,
    dirname(LOG_FILE)
)) {
    dir.create(
        directory,
        recursive = TRUE,
        showWarnings = FALSE
    )
}

cat("", file = LOG_FILE)

log_message <- function(...) {
    message <- paste0(...)

    cat(message, "\n")

    cat(
        message,
        "\n",
        file = LOG_FILE,
        append = TRUE
    )
}

set.seed(20260714)

SPLINE_DF <- 4L
PREDICTION_POINTS <- 120L
MINIMUM_RAW_EXPRESSION <- 1
MINIMUM_DETECTED_FRACTION <- 0.20
MINIMUM_LOG2_SD <- 0.25
MINIMUM_LOG2_RANGE <- 0.75
MINIMUM_PREDICTED_RANGE <- 0.50
MAXIMUM_CLUSTER_GENES <- 2500L
MAXIMUM_SILHOUETTE_GENES <- 800L
MAXIMUM_CLUSTERS <- 10L

signed_log_transform <- function(
    values,
    scale = 0.05
) {
    sign(values) *
        log1p(
            abs(values) / scale
        )
}

inverse_signed_log <- function(
    values,
    scale = 0.05
) {
    sign(values) *
        scale *
        (
            exp(abs(values)) - 1
        )
}

format_age <- function(value) {
    if (!is.finite(value)) {
        return(NA_character_)
    }

    if (value < 0) {
        return(
            paste0(
                round(
                    40 + value * 52,
                    1
                ),
                " pcw"
            )
        )
    }

    if (value < 1) {
        return(
            paste0(
                round(
                    value * 12,
                    1
                ),
                " months"
            )
        )
    }

    paste0(
        round(value, 1),
        " years"
    )
}

safe_spearman <- function(first, second) {
    if (
        length(first) < 3 ||
        sd(first) == 0 ||
        sd(second) == 0
    ) {
        return(NA_real_)
    }

    suppressWarnings(
        cor(
            first,
            second,
            method = "spearman"
        )
    )
}

if (!file.exists(EXPRESSION_FILE)) {
    stop(
        "Missing donor expression matrix: ",
        EXPRESSION_FILE
    )
}

if (!file.exists(METADATA_FILE)) {
    stop(
        "Missing donor metadata: ",
        METADATA_FILE
    )
}

log_message("===== Phase 5D2 started =====")

# ============================================================
# Read and align data
# ============================================================

expression_table <- fread(
    cmd = paste(
        "gzip -dc",
        shQuote(EXPRESSION_FILE)
    ),
    sep = "\t",
    header = TRUE,
    check.names = FALSE,
    data.table = TRUE
)

if (
    all(
        grepl(
            "^V[0-9]+$",
            names(expression_table)[-1]
        )
    )
) {
    stop(
        "Expression header was not read correctly; artificial V-column names remain."
    )
}

gene_column <- names(expression_table)[1]

gene_symbols <- as.character(
    expression_table[[gene_column]]
)

expression_matrix <- as.matrix(
    expression_table[
        ,
        setdiff(
            names(expression_table),
            gene_column
        ),
        with = FALSE
    ]
)

storage.mode(expression_matrix) <- "double"

rownames(expression_matrix) <- gene_symbols

metadata <- fread(
    METADATA_FILE,
    colClasses = c(
        donor_clean = "character"
    )
)

required_metadata <- c(
    "donor_clean",
    "sex_clean",
    "developmental_age_years_from_birth",
    "developmental_time_transformed"
)

missing_metadata <- setdiff(
    required_metadata,
    names(metadata)
)

if (length(missing_metadata) > 0) {
    stop(
        "Missing metadata columns: ",
        paste(
            missing_metadata,
            collapse = ", "
        )
    )
}

normalize_donor_id <- function(values) {
    values <- trimws(
        as.character(values)
    )

    values <- sub(
        "^\\ufeff",
        "",
        values
    )

    values
}

donor_columns <- normalize_donor_id(
    colnames(
        expression_matrix
    )
)

metadata[
    ,
    donor_clean := normalize_donor_id(
        donor_clean
    )
]

colnames(
    expression_matrix
) <- donor_columns

if (
    anyDuplicated(
        donor_columns
    ) > 0
) {
    duplicated_expression_donors <- unique(
        donor_columns[
            duplicated(
                donor_columns
            )
        ]
    )

    stop(
        "Duplicated donor columns in expression matrix: ",
        paste(
            duplicated_expression_donors,
            collapse = ", "
        )
    )
}

if (
    anyDuplicated(
        metadata$donor_clean
    ) > 0
) {
    duplicated_metadata_donors <- unique(
        metadata$donor_clean[
            duplicated(
                metadata$donor_clean
            )
        ]
    )

    stop(
        "Duplicated donors in metadata: ",
        paste(
            duplicated_metadata_donors,
            collapse = ", "
        )
    )
}

all_donor_ids <- sort(
    unique(
        c(
            donor_columns,
            metadata$donor_clean
        )
    )
)

donor_alignment_audit <- data.table(
    donor_clean = all_donor_ids,

    in_expression_matrix =
        all_donor_ids %in%
        donor_columns,

    in_metadata =
        all_donor_ids %in%
        metadata$donor_clean
)

fwrite(
    donor_alignment_audit,
    file.path(
        TABLE_DIR,
        "phase5D2_donor_alignment_audit.tsv"
    ),
    sep = "\t"
)

missing_from_metadata <- setdiff(
    donor_columns,
    metadata$donor_clean
)

missing_from_expression <- setdiff(
    metadata$donor_clean,
    donor_columns
)

if (
    length(
        missing_from_metadata
    ) > 0 ||
    length(
        missing_from_expression
    ) > 0
) {
    stop(
        paste0(
            "Expression and metadata donors genuinely differ. ",
            "Missing from metadata: ",
            ifelse(
                length(
                    missing_from_metadata
                ) == 0,
                "none",
                paste(
                    missing_from_metadata,
                    collapse = ", "
                )
            ),
            "; missing from expression: ",
            ifelse(
                length(
                    missing_from_expression
                ) == 0,
                "none",
                paste(
                    missing_from_expression,
                    collapse = ", "
                )
            ),
            ". See phase5D2_donor_alignment_audit.tsv."
        )
    )
}

metadata <- metadata[
    match(
        donor_columns,
        donor_clean
    )
]

if (
    any(
        is.na(
            metadata$donor_clean
        )
    )
) {
    stop(
        "NA values appeared during donor metadata alignment."
    )
}

if (
    !identical(
        donor_columns,
        metadata$donor_clean
    )
) {
    stop(
        "Donor order alignment failed after normalization."
    )
}

log_message(
    "Expression donors aligned with metadata: ",
    length(
        donor_columns
    )
)

metadata[
    ,
    sex_clean := droplevels(
        factor(
            fifelse(
                is.na(sex_clean) |
                sex_clean == "",
                "unknown",
                as.character(sex_clean)
            )
        )
    )
]

developmental_time <- as.numeric(
    metadata$
        developmental_time_transformed
)

developmental_age <- as.numeric(
    metadata$
        developmental_age_years_from_birth
)

log_message(
    "Input genes: ",
    nrow(expression_matrix)
)

log_message(
    "Independent donors: ",
    ncol(expression_matrix)
)

log_message(
    "Developmental range: ",
    format_age(
        min(developmental_age)
    ),
    " to ",
    format_age(
        max(developmental_age)
    )
)

# ============================================================
# Log transformation and gene filtering
# ============================================================

if (
    any(
        expression_matrix < 0,
        na.rm = TRUE
    )
) {
    stop(
        "Negative expression values detected; "
    )
}

log_expression <- log2(
    expression_matrix + 1
)

minimum_detected_donors <- ceiling(
    ncol(expression_matrix) *
    MINIMUM_DETECTED_FRACTION
)

finite_donors <- rowSums(
    is.finite(
        log_expression
    )
)

detected_donors <- rowSums(
    expression_matrix >=
        MINIMUM_RAW_EXPRESSION,
    na.rm = TRUE
)

log2_sd <- apply(
    log_expression,
    1,
    sd,
    na.rm = TRUE
)

log2_minimum <- apply(
    log_expression,
    1,
    min,
    na.rm = TRUE
)

log2_maximum <- apply(
    log_expression,
    1,
    max,
    na.rm = TRUE
)

log2_range <- (
    log2_maximum -
    log2_minimum
)

filter_table <- data.table(
    gene_symbol = rownames(
        expression_matrix
    ),

    finite_donors =
        finite_donors,

    detected_donors =
        detected_donors,

    detected_fraction =
        detected_donors /
        ncol(
            expression_matrix
        ),

    log2_mean =
        rowMeans(
            log_expression,
            na.rm = TRUE
        ),

    log2_SD =
        log2_sd,

    log2_range =
        log2_range
)

filter_table[
    ,
    finite_pass :=
        finite_donors ==
        ncol(expression_matrix)
]

filter_table[
    ,
    detection_pass :=
        detected_donors >=
        minimum_detected_donors
]

filter_table[
    ,
    variability_pass :=
        log2_SD >=
        MINIMUM_LOG2_SD
        &
        log2_range >=
        MINIMUM_LOG2_RANGE
]

filter_table[
    ,
    trajectory_model_pass :=
        finite_pass
        &
        detection_pass
        &
        variability_pass
]

fwrite(
    filter_table,
    file.path(
        TABLE_DIR,
        "phase5D2_gene_filtering_summary.tsv.gz"
    ),
    sep = "\t",
    compress = "gzip"
)

eligible_genes <- filter_table[
    trajectory_model_pass == TRUE,
    gene_symbol
]

if (length(eligible_genes) < 100) {
    stop(
        "Too few genes passed trajectory filtering: ",
        length(eligible_genes)
    )
}

filtered_expression <- log_expression[
    eligible_genes,
    ,
    drop = FALSE
]

log_message(
    "Minimum detected donors: ",
    minimum_detected_donors
)

log_message(
    "Genes retained for spline modelling: ",
    nrow(filtered_expression)
)

fwrite(
    data.table(
        gene_symbol = rownames(
            filtered_expression
        ),
        filtered_expression
    ),
    file.path(
        OUT_DIR,
        "phase5D2_filtered_log2_expression.tsv.gz"
    ),
    sep = "\t",
    compress = "gzip"
)

# ============================================================
# Natural-spline design matrix
# ============================================================

sex_design <- model.matrix(
    ~ sex_clean,
    data = metadata
)

spline_basis <- ns(
    developmental_time,
    df = SPLINE_DF
)

colnames(spline_basis) <- paste0(
    "developmental_spline_",
    seq_len(
        ncol(spline_basis)
    )
)

design <- cbind(
    sex_design,
    spline_basis
)

design <- design[
    ,
    !duplicated(
        colnames(design)
    ),
    drop = FALSE
]

spline_columns <- grep(
    "^developmental_spline_",
    colnames(design)
)

if (length(spline_columns) != SPLINE_DF) {
    stop(
        "Unexpected number of spline coefficients."
    )
}

design_table <- data.table(
    donor_clean = metadata$donor_clean,
    design
)

fwrite(
    design_table,
    file.path(
        TABLE_DIR,
        "phase5D2_spline_limma_design_matrix.tsv"
    ),
    sep = "\t"
)

# ============================================================
# Limma empirical-Bayes model
# ============================================================

fit <- lmFit(
    filtered_expression,
    design
)

fit <- eBayes(
    fit,
    trend = TRUE,
    robust = TRUE
)

saveRDS(
    fit,
    file.path(
        MODEL_DIR,
        "phase5D2_spline_limma_fit.rds"
    )
)

limma_results <- topTable(
    fit,
    coef = spline_columns,
    number = Inf,
    sort.by = "F"
)

limma_results[["gene_symbol"]] <- rownames(
    limma_results
)

limma_results <- as.data.table(
    limma_results
)

setnames(
    limma_results,
    old = intersect(
        c(
            "P.Value",
            "adj.P.Val",
            "F"
        ),
        names(limma_results)
    ),
    new = c(
        "spline_P",
        "spline_FDR",
        "moderated_F"
    )[
        match(
            intersect(
                c(
                    "P.Value",
                    "adj.P.Val",
                    "F"
                ),
                names(limma_results)
            ),
            c(
                "P.Value",
                "adj.P.Val",
                "F"
            )
        )
    ]
)

# ============================================================
# Marginal developmental predictions
# ============================================================

prediction_time <- seq(
    min(developmental_time),
    max(developmental_time),
    length.out = PREDICTION_POINTS
)

prediction_age <- inverse_signed_log(
    prediction_time
)

prediction_spline <- predict(
    spline_basis,
    prediction_time
)

sex_levels <- levels(
    metadata$sex_clean
)

prediction_sex_data <- data.frame(
    sex_clean = factor(
        sex_levels,
        levels = sex_levels
    )
)

prediction_sex_design <- model.matrix(
    ~ sex_clean,
    data = prediction_sex_data
)

mean_sex_design <- colMeans(
    prediction_sex_design
)

prediction_design <- cbind(
    matrix(
        rep(
            mean_sex_design,
            each = PREDICTION_POINTS
        ),
        nrow = PREDICTION_POINTS,
        byrow = FALSE
    ),
    prediction_spline
)

colnames(prediction_design) <- colnames(
    design
)

predicted_expression <- (
    fit$coefficients %*%
    t(prediction_design)
)

predicted_range <- apply(
    predicted_expression,
    1,
    function(values) {
        max(values) -
            min(values)
    }
)

predicted_earliest <- predicted_expression[
    ,
    1
]

predicted_latest <- predicted_expression[
    ,
    ncol(predicted_expression)
]

predicted_change <- (
    predicted_latest -
    predicted_earliest
)

predicted_age_spearman <- apply(
    predicted_expression,
    1,
    function(values) {
        safe_spearman(
            prediction_age,
            values
        )
    }
)

peak_index <- max.col(
    predicted_expression,
    ties.method = "first"
)

trough_index <- max.col(
    -predicted_expression,
    ties.method = "first"
)

prediction_features <- data.table(
    gene_symbol = rownames(
        predicted_expression
    ),

    predicted_log2_range =
        predicted_range,

    predicted_earliest_log2 =
        predicted_earliest,

    predicted_latest_log2 =
        predicted_latest,

    predicted_earliest_to_latest_change =
        predicted_change,

    predicted_age_spearman_r =
        predicted_age_spearman,

    peak_age_years_from_birth =
        prediction_age[
            peak_index
        ],

    peak_age_label =
        vapply(
            prediction_age[
                peak_index
            ],
            format_age,
            character(1)
        ),

    trough_age_years_from_birth =
        prediction_age[
            trough_index
        ],

    trough_age_label =
        vapply(
            prediction_age[
                trough_index
            ],
            format_age,
            character(1)
        )
)

gene_results <- merge(
    limma_results,
    prediction_features,
    by = "gene_symbol",
    all.x = TRUE
)

gene_results[
    ,
    FDR_significant :=
        spline_FDR < 0.05
]

gene_results[
    ,
    effect_size_pass :=
        predicted_log2_range >=
        MINIMUM_PREDICTED_RANGE
]

gene_results[
    ,
    clustering_candidate :=
        FDR_significant
        &
        effect_size_pass
]

setorder(
    gene_results,
    spline_FDR,
    -moderated_F
)

fwrite(
    gene_results,
    file.path(
        TABLE_DIR,
        "phase5D2_gene_level_spline_limma_results.tsv.gz"
    ),
    sep = "\t",
    compress = "gzip"
)

# ============================================================
# Data-driven trajectory clustering
# ============================================================

cluster_candidates <- gene_results[
    clustering_candidate == TRUE
]

if (
    nrow(cluster_candidates) >
    MAXIMUM_CLUSTER_GENES
) {
    cluster_candidates <- cluster_candidates[
        order(
            spline_FDR,
            -moderated_F,
            -predicted_log2_range
        )
    ][
        seq_len(
            MAXIMUM_CLUSTER_GENES
        )
    ]
}

cluster_genes <- cluster_candidates[["gene_symbol"]]

cluster_prediction <- predicted_expression[
    cluster_genes,
    ,
    drop = FALSE
]

cluster_row_mean <- rowMeans(
    cluster_prediction
)

cluster_row_sd <- apply(
    cluster_prediction,
    1,
    sd
)

valid_cluster_rows <- (
    is.finite(cluster_row_sd)
    &
    cluster_row_sd > 0
)

cluster_prediction <- cluster_prediction[
    valid_cluster_rows,
    ,
    drop = FALSE
]

cluster_row_mean <- cluster_row_mean[
    valid_cluster_rows
]

cluster_row_sd <- cluster_row_sd[
    valid_cluster_rows
]

cluster_z <- (
    cluster_prediction -
    cluster_row_mean
) / cluster_row_sd

number_cluster_genes <- nrow(
    cluster_z
)

if (number_cluster_genes < 20) {
    stop(
        "Too few significant effect-size-filtered genes "
    )
}

silhouette_sample_size <- min(
    MAXIMUM_SILHOUETTE_GENES,
    number_cluster_genes
)

silhouette_sample_indices <- sample(
    seq_len(
        number_cluster_genes
    ),
    silhouette_sample_size
)

silhouette_matrix <- cluster_z[
    silhouette_sample_indices,
    ,
    drop = FALSE
]

silhouette_distance <- dist(
    silhouette_matrix
)

maximum_tested_clusters <- min(
    MAXIMUM_CLUSTERS,
    max(
        2L,
        floor(
            sqrt(
                silhouette_sample_size
            )
        )
    )
)

candidate_k <- seq.int(
    2L,
    maximum_tested_clusters
)

silhouette_rows <- vector(
    "list",
    length(candidate_k)
)

for (
    k_index
    in seq_along(candidate_k)
) {
    current_k <- candidate_k[
        k_index
    ]

    current_kmeans <- kmeans(
        silhouette_matrix,
        centers = current_k,
        nstart = 50,
        iter.max = 200
    )

    current_silhouette <- silhouette(
        current_kmeans$cluster,
        silhouette_distance
    )

    silhouette_rows[[k_index]] <- data.table(
        k = current_k,

        average_silhouette =
            mean(
                current_silhouette[
                    ,
                    "sil_width"
                ]
            )
    )
}

silhouette_summary <- rbindlist(
    silhouette_rows
)

selected_k <- silhouette_summary[
    which.max(
        average_silhouette
    ),
    k
]

final_kmeans <- kmeans(
    cluster_z,
    centers = selected_k,
    nstart = 100,
    iter.max = 500
)

cluster_assignments <- data.table(
    gene_symbol = rownames(
        cluster_z
    ),

    trajectory_cluster =
        as.integer(
            final_kmeans$cluster
        )
)

cluster_assignments <- merge(
    cluster_assignments,
    gene_results[
        ,
        .(
            gene_symbol,
            spline_P,
            spline_FDR,
            moderated_F,
            predicted_log2_range,
            predicted_earliest_to_latest_change,
            predicted_age_spearman_r,
            peak_age_label,
            trough_age_label
        )
    ],
    by = "gene_symbol",
    all.x = TRUE
)

setorder(
    cluster_assignments,
    trajectory_cluster,
    spline_FDR
)

fwrite(
    silhouette_summary,
    file.path(
        TABLE_DIR,
        "phase5D2_cluster_number_silhouette_summary.tsv"
    ),
    sep = "\t"
)

fwrite(
    cluster_assignments,
    file.path(
        TABLE_DIR,
        "phase5D2_data_driven_trajectory_cluster_assignments.tsv"
    ),
    sep = "\t"
)

# ============================================================
# Cluster centroid and interval summaries
# ============================================================

cluster_long <- data.table(
    gene_symbol = rep(
        rownames(cluster_z),
        times = ncol(cluster_z)
    ),

    prediction_index = rep(
        seq_len(
            ncol(cluster_z)
        ),
        each = nrow(cluster_z)
    ),

    trajectory_z = as.vector(
        cluster_z
    )
)

if (
    nrow(cluster_long) !=
    nrow(cluster_z) * ncol(cluster_z)
) {
    stop(
        "Cluster trajectory long-table construction failed."
    )
}

if (
    any(
        !is.finite(
            cluster_long$prediction_index
        )
    )
) {
    stop(
        "Non-finite prediction indices detected."
    )
}

cluster_long <- merge(
    cluster_long,
    cluster_assignments[
        ,
        .(
            gene_symbol,
            trajectory_cluster
        )
    ],
    by = "gene_symbol",
    all.x = TRUE
)

cluster_long[
    ,
    developmental_time_transformed :=
        prediction_time[
            prediction_index
        ]
]

cluster_long[
    ,
    developmental_age_years_from_birth :=
        prediction_age[
            prediction_index
        ]
]

cluster_centroids <- cluster_long[
    ,
    .(
        mean_z = mean(
            trajectory_z
        ),

        median_z = median(
            trajectory_z
        ),

        q25_z = quantile(
            trajectory_z,
            0.25
        ),

        q75_z = quantile(
            trajectory_z,
            0.75
        ),

        n_genes = uniqueN(
            gene_symbol
        )
    ),
    by = .(
        trajectory_cluster,
        prediction_index,
        developmental_time_transformed,
        developmental_age_years_from_birth
    )
]

cluster_summary_rows <- vector(
    "list",
    selected_k
)

for (
    cluster_id
    in seq_len(selected_k)
) {
    current_centroid <- cluster_centroids[
        trajectory_cluster ==
            cluster_id
    ]

    peak_row <- current_centroid[
        which.max(mean_z)
    ]

    trough_row <- current_centroid[
        which.min(mean_z)
    ]

    earliest_value <- current_centroid[
        1,
        mean_z
    ]

    latest_value <- current_centroid[
        .N,
        mean_z
    ]

    cluster_summary_rows[[cluster_id]] <- data.table(
        trajectory_cluster =
            cluster_id,

        n_genes =
            unique(
                current_centroid$n_genes
            ),

        peak_age_years_from_birth =
            peak_row$
                developmental_age_years_from_birth,

        peak_age_label =
            format_age(
                peak_row$
                    developmental_age_years_from_birth
            ),

        trough_age_years_from_birth =
            trough_row$
                developmental_age_years_from_birth,

        trough_age_label =
            format_age(
                trough_row$
                    developmental_age_years_from_birth
            ),

        earliest_centroid_z =
            earliest_value,

        latest_centroid_z =
            latest_value,

        earliest_to_latest_centroid_change =
            latest_value -
            earliest_value,

        centroid_age_spearman_r =
            safe_spearman(
                current_centroid$
                    developmental_age_years_from_birth,
                current_centroid$
                    mean_z
            )
    )
}

cluster_summary <- rbindlist(
    cluster_summary_rows
)

cluster_summary[
    ,
    broad_trajectory_pattern :=
        fifelse(
            centroid_age_spearman_r >= 0.80,
            "progressive_increase",
            fifelse(
                centroid_age_spearman_r <= -0.80,
                "progressive_decrease",
                fifelse(
                    peak_age_years_from_birth > -0.10
                    &
                    peak_age_years_from_birth < 2,
                    "perinatal_or_infant_peak",
                    "nonmonotonic_trajectory"
                )
            )
        )
]

fwrite(
    cluster_summary,
    file.path(
        TABLE_DIR,
        "phase5D2_data_driven_trajectory_cluster_summary.tsv"
    ),
    sep = "\t"
)

fwrite(
    cluster_centroids,
    file.path(
        SOURCE_DIR,
        "Figure38_source_cluster_trajectory_centroids.tsv.gz"
    ),
    sep = "\t",
    compress = "gzip"
)

# ============================================================
# Figure 37: significance–effect landscape
# ============================================================

plot_results <- copy(
    gene_results
)

minimum_positive_FDR <- min(
    plot_results[
        spline_FDR > 0,
        spline_FDR
    ],
    na.rm = TRUE
)

plot_results[
    ,
    plotting_FDR :=
        pmax(
            spline_FDR,
            minimum_positive_FDR
        )
]

plot_results[
    ,
    evidence_class :=
        fifelse(
            clustering_candidate,
            "FDR and effect-size supported",
            fifelse(
                FDR_significant,
                "FDR supported; small trajectory range",
                "Not FDR supported"
            )
        )
]

figure_37 <- ggplot(
    plot_results,
    aes(
        x = predicted_log2_range,
        y = -log10(
            plotting_FDR
        ),
        shape = evidence_class
    )
) +
    geom_point(
        alpha = 0.45,
        size = 1
    ) +
    geom_vline(
        xintercept =
            MINIMUM_PREDICTED_RANGE,
        linetype = 2,
        linewidth = 0.35
    ) +
    geom_hline(
        yintercept =
            -log10(0.05),
        linetype = 2,
        linewidth = 0.35
    ) +
    labs(
        title = paste0(
            "Figure 37. Gene-level developmental ",
            "trajectory discovery"
        ),
        subtitle = paste0(
            "Natural-spline limma model across ",
            ncol(expression_matrix),
            " independent BrainSpan donors"
        ),
        x = "Predicted developmental range, log2(expression + 1)",
        y = "-log10 spline FDR",
        shape = "Evidence"
    ) +
    theme_bw(
        base_size = 11
    ) +
    theme(
        plot.title = element_text(
            size = 14,
            face = "bold"
        )
    )

ggsave(
    filename = file.path(
        FIGURE_DIR,
        "Figure37_gene_level_developmental_trajectory_discovery.pdf"
    ),
    plot = figure_37,
    width = 10,
    height = 7,
    units = "in"
)

ggsave(
    filename = file.path(
        FIGURE_DIR,
        "Figure37_gene_level_developmental_trajectory_discovery.png"
    ),
    plot = figure_37,
    width = 10,
    height = 7,
    units = "in",
    dpi = 300
)

fwrite(
    plot_results[
        ,
        .(
            gene_symbol,
            spline_FDR,
            predicted_log2_range,
            evidence_class
        )
    ],
    file.path(
        SOURCE_DIR,
        "Figure37_source_gene_significance_effect.tsv.gz"
    ),
    sep = "\t",
    compress = "gzip"
)

# ============================================================
# Figure 38: data-driven trajectory clusters
# ============================================================

cluster_centroids[
    ,
    trajectory_cluster_label :=
        factor(
            paste0(
                "Cluster ",
                trajectory_cluster,
                " (n=",
                n_genes,
                ")"
            ),
            levels = paste0(
                "Cluster ",
                cluster_summary$
                    trajectory_cluster,
                " (n=",
                cluster_summary$n_genes,
                ")"
            )
        )
]

axis_age_values <- c(
    -0.60,
    -0.40,
    -0.20,
    0,
    0.5,
    1,
    5,
    10,
    20,
    40
)

axis_positions <- signed_log_transform(
    axis_age_values
)

axis_labels <- c(
    "Prenatal\n−0.6 y",
    "−0.4 y",
    "−0.2 y",
    "Birth",
    "6 mo",
    "1 y",
    "5 y",
    "10 y",
    "20 y",
    "40 y"
)

figure_38 <- ggplot(
    cluster_centroids,
    aes(
        x =
            developmental_time_transformed,
        y = mean_z
    )
) +
    geom_hline(
        yintercept = 0,
        linetype = 2,
        linewidth = 0.3
    ) +
    geom_ribbon(
        aes(
            ymin = q25_z,
            ymax = q75_z
        ),
        alpha = 0.20
    ) +
    geom_line(
        linewidth = 0.9
    ) +
    facet_wrap(
        ~ trajectory_cluster_label,
        ncol = 2,
        scales = "free_y"
    ) +
    scale_x_continuous(
        breaks = axis_positions,
        labels = axis_labels
    ) +
    labs(
        title = paste0(
            "Figure 38. Data-driven developmental ",
            "gene-trajectory clusters"
        ),
        subtitle = paste0(
            "Lines show cluster means; ribbons show ",
            "interquartile ranges of standardized trajectories"
        ),
        x = "Developmental age relative to birth",
        y = "Standardized predicted expression"
    ) +
    theme_bw(
        base_size = 10
    ) +
    theme(
        plot.title = element_text(
            size = 14,
            face = "bold"
        ),
        strip.text = element_text(
            size = 9,
            face = "bold"
        ),
        axis.text.x = element_text(
            size = 7.5
        )
    )

ggsave(
    filename = file.path(
        FIGURE_DIR,
        "Figure38_data_driven_gene_trajectory_clusters.pdf"
    ),
    plot = figure_38,
    width = 13,
    height = 4 + selected_k * 1.7,
    units = "in",
    limitsize = FALSE
)

ggsave(
    filename = file.path(
        FIGURE_DIR,
        "Figure38_data_driven_gene_trajectory_clusters.png"
    ),
    plot = figure_38,
    width = 13,
    height = 4 + selected_k * 1.7,
    units = "in",
    dpi = 300,
    limitsize = FALSE
)

# ============================================================
# Completion summary
# ============================================================

completion <- data.table(
    input_genes =
        nrow(expression_matrix),

    independent_donors =
        ncol(expression_matrix),

    minimum_detected_donors =
        minimum_detected_donors,

    trajectory_model_genes =
        nrow(filtered_expression),

    spline_FDR_significant_genes =
        sum(
            gene_results$
                FDR_significant,
            na.rm = TRUE
        ),

    FDR_and_effect_supported_genes =
        sum(
            gene_results$
                clustering_candidate,
            na.rm = TRUE
        ),

    genes_entering_clustering =
        nrow(cluster_z),

    selected_number_of_clusters =
        selected_k,

    selected_average_silhouette =
        silhouette_summary[
            k == selected_k,
            average_silhouette
        ],

    spline_degrees_of_freedom =
        SPLINE_DF,

    minimum_predicted_log2_range =
        MINIMUM_PREDICTED_RANGE,

    Phase5D2_status =
        "completed"
)

fwrite(
    completion,
    file.path(
        TABLE_DIR,
        "phase5D2_completion_summary.tsv"
    ),
    sep = "\t"
)

report_file <- file.path(
    META_DIR,
    "PHASE5D2_GENE_LEVEL_TRAJECTORY_REPORT.txt"
)

sink(report_file)

cat(
    "Phase 5D2 gene-level developmental trajectory discovery\n\n"
)

cat(
    "Expression was transformed using log2(expression + 1). "
)

cat(
    "Genes were required to reach raw expression >= 1 in at least ",
    minimum_detected_donors,
    " of ",
    ncol(expression_matrix),
    " donors and to meet log-scale variability thresholds.\n\n",
    sep = ""
)

cat("Completion summary\n")

print(completion)

cat("\nCluster-number assessment\n")

print(silhouette_summary)

cat("\nCluster summaries\n")

print(cluster_summary)

cat(
    "\nAll statistical tests used independent donors as the unit of inference. "
)

cat(
    "Standardization was applied only after modelling and only for trajectory clustering.\n"
)

sink()

log_message("")
log_message("===== Phase 5D2 completed =====")

log_message(
    paste(
        capture.output(
            print(completion)
        ),
        collapse = "\n"
    )
)
