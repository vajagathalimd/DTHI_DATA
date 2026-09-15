#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(data.table)
    library(ggplot2)
    library(splines)
})

PROJECT <- "."

FIT_FILE <- file.path(
    PROJECT,
    "03_processed_data/developmental_trajectory/phase5D/models",
    "phase5D2_spline_limma_fit.rds"
)

METADATA_FILE <- file.path(
    PROJECT,
    "03_processed_data/developmental_trajectory/phase5D",
    "phase5D1_donor_metadata.tsv"
)

GENE_RESULTS_FILE <- file.path(
    PROJECT,
    "07_tables/main_tables/phase5",
    "phase5D2_gene_level_spline_limma_results.tsv.gz"
)

SEED_ASSIGNMENT_FILE <- file.path(
    PROJECT,
    "07_tables/main_tables/phase5",
    "phase5D2_data_driven_trajectory_cluster_assignments.tsv"
)

CENTROID_FILE <- file.path(
    PROJECT,
    "06_figures/source_data/phase5",
    "Figure38_source_cluster_trajectory_centroids.tsv.gz"
)

OUT_DIR <- file.path(
    PROJECT,
    "03_processed_data/developmental_trajectory/phase5D"
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
    "phase5D3_full_centroid_assignment.log"
)

for (directory in c(
    OUT_DIR,
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

required_files <- c(
    FIT_FILE,
    METADATA_FILE,
    GENE_RESULTS_FILE,
    SEED_ASSIGNMENT_FILE,
    CENTROID_FILE
)

missing_files <- required_files[
    !file.exists(required_files)
]

if (length(missing_files) > 0) {
    stop(
        "Missing required files: ",
        paste(
            missing_files,
            collapse = "; "
        )
    )
}

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

row_zscore <- function(matrix_value) {
    row_means <- rowMeans(
        matrix_value,
        na.rm = TRUE
    )

    row_sds <- apply(
        matrix_value,
        1,
        sd,
        na.rm = TRUE
    )

    valid <- (
        is.finite(row_sds) &
        row_sds > 0
    )

    result <- matrix(
        NA_real_,
        nrow = nrow(matrix_value),
        ncol = ncol(matrix_value),
        dimnames = dimnames(matrix_value)
    )

    result[valid, ] <- (
        matrix_value[valid, , drop = FALSE] -
        row_means[valid]
    ) / row_sds[valid]

    result
}

log_message("===== Phase 5D3 started =====")

# ============================================================
# Load model and metadata
# ============================================================

fit <- readRDS(
    FIT_FILE
)

metadata <- fread(
    METADATA_FILE,
    colClasses = c(
        donor_clean = "character"
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
    metadata$developmental_time_transformed
)

prediction_time <- seq(
    min(developmental_time),
    max(developmental_time),
    length.out = PREDICTION_POINTS
)

prediction_age <- inverse_signed_log(
    prediction_time
)

# ============================================================
# Reconstruct the Phase 5D2 prediction design
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

if (
    ncol(prediction_design) !=
    ncol(fit$coefficients)
) {
    stop(
        "Prediction design does not match the saved limma model."
    )
}

predicted_expression <- (
    fit$coefficients %*%
    t(prediction_design)
)

rownames(predicted_expression) <- rownames(
    fit$coefficients
)

colnames(predicted_expression) <- paste0(
    "T",
    seq_len(PREDICTION_POINTS)
)

# ============================================================
# Select all supported genes
# ============================================================

gene_results <- fread(
    cmd = paste(
        "gzip -dc",
        shQuote(GENE_RESULTS_FILE)
    ),
    sep = "\t",
    header = TRUE
)

supported_gene_mask <- (
    gene_results$clustering_candidate == TRUE
)

supported_genes <- as.character(
    gene_results$gene_symbol[
        supported_gene_mask
    ]
)

supported_genes <- intersect(
    supported_genes,
    rownames(predicted_expression)
)

if (length(supported_genes) < 100) {
    stop(
        "Too few supported genes were recovered."
    )
}

supported_prediction <- predicted_expression[
    supported_genes,
    ,
    drop = FALSE
]

supported_z <- row_zscore(
    supported_prediction
)

valid_supported <- rowSums(
    is.finite(supported_z)
) == ncol(supported_z)

supported_z <- supported_z[
    valid_supported,
    ,
    drop = FALSE
]

supported_genes <- rownames(
    supported_z
)

log_message(
    "Supported genes reconstructed: ",
    length(supported_genes)
)

# ============================================================
# Load the original learned centroids
# ============================================================

centroid_table <- fread(
    cmd = paste(
        "gzip -dc",
        shQuote(CENTROID_FILE)
    ),
    sep = "\t",
    header = TRUE
)

required_centroid_columns <- c(
    "trajectory_cluster",
    "prediction_index",
    "mean_z"
)

missing_centroid_columns <- setdiff(
    required_centroid_columns,
    names(centroid_table)
)

if (length(missing_centroid_columns) > 0) {
    stop(
        "Missing centroid columns: ",
        paste(
            missing_centroid_columns,
            collapse = ", "
        )
    )
}

cluster_ids <- sort(
    unique(
        centroid_table$trajectory_cluster
    )
)

if (length(cluster_ids) != 2) {
    stop(
        "Expected exactly two learned trajectory clusters."
    )
}

centroid_matrix <- do.call(
    rbind,
    lapply(
        cluster_ids,
        function(cluster_id) {
            current <- centroid_table[
                trajectory_cluster == cluster_id
            ]

            setorder(
                current,
                prediction_index
            )

            if (nrow(current) != PREDICTION_POINTS) {
                stop(
                    "Cluster ",
                    cluster_id,
                    " does not contain ",
                    PREDICTION_POINTS,
                    " prediction points."
                )
            }

            current$mean_z
        }
    )
)

rownames(centroid_matrix) <- paste0(
    "Cluster_",
    cluster_ids
)

centroid_z <- row_zscore(
    centroid_matrix
)

# ============================================================
# Correlation-based full assignment
# ============================================================

correlation_matrix <- (
    supported_z %*%
    t(centroid_z)
) / (
    ncol(supported_z) - 1
)

colnames(correlation_matrix) <- rownames(
    centroid_z
)

best_index <- max.col(
    correlation_matrix,
    ties.method = "first"
)

best_correlation <- correlation_matrix[
    cbind(
        seq_len(
            nrow(correlation_matrix)
        ),
        best_index
    )
]

second_index <- ifelse(
    best_index == 1,
    2,
    1
)

second_correlation <- correlation_matrix[
    cbind(
        seq_len(
            nrow(correlation_matrix)
        ),
        second_index
    )
]

correlation_margin <- (
    best_correlation -
    second_correlation
)

nearest_cluster <- cluster_ids[
    best_index
]

assignment_confidence <- fifelse(
    best_correlation >= 0.80 &
    correlation_margin >= 0.20,
    "high",

    fifelse(
        best_correlation >= 0.60 &
        correlation_margin >= 0.10,
        "moderate",

        fifelse(
            best_correlation >= 0.40 &
            correlation_margin >= 0.05,
            "low",
            "ambiguous"
        )
    )
)

confident_cluster <- ifelse(
    assignment_confidence %in%
        c(
            "high",
            "moderate"
        ),
    nearest_cluster,
    NA_integer_
)

assignment_table <- data.table(
    gene_symbol =
        supported_genes,

    correlation_cluster_1 =
        correlation_matrix[
            ,
            1
        ],

    correlation_cluster_2 =
        correlation_matrix[
            ,
            2
        ],

    nearest_cluster =
        nearest_cluster,

    best_centroid_correlation =
        best_correlation,

    second_centroid_correlation =
        second_correlation,

    correlation_margin =
        correlation_margin,

    assignment_confidence =
        assignment_confidence,

    confident_cluster =
        confident_cluster
)

assignment_table <- merge(
    assignment_table,
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
    assignment_table,
    nearest_cluster,
    -best_centroid_correlation,
    spline_FDR
)

fwrite(
    assignment_table,
    file.path(
        TABLE_DIR,
        "phase5D3_all_supported_gene_centroid_assignments.tsv.gz"
    ),
    sep = "\t",
    compress = "gzip"
)

# ============================================================
# Validate against original 2,500-gene clustering
# ============================================================

seed_assignments <- fread(
    SEED_ASSIGNMENT_FILE
)

validation <- merge(
    seed_assignments[
        ,
        .(
            gene_symbol,
            original_trajectory_cluster =
                trajectory_cluster
        )
    ],
    assignment_table[
        ,
        .(
            gene_symbol,
            nearest_cluster,
            best_centroid_correlation,
            correlation_margin,
            assignment_confidence
        )
    ],
    by = "gene_symbol",
    all.x = TRUE
)

validation[
    ,
    assignment_agreement :=
        original_trajectory_cluster ==
        nearest_cluster
]

validation_summary <- data.table(
    original_clustered_genes =
        nrow(validation),

    recovered_clustered_genes =
        sum(
            !is.na(
                validation$nearest_cluster
            )
        ),

    assignment_agreement_count =
        sum(
            validation$assignment_agreement,
            na.rm = TRUE
        ),

    assignment_agreement_fraction =
        mean(
            validation$assignment_agreement,
            na.rm = TRUE
        ),

    median_best_correlation =
        median(
            validation$best_centroid_correlation,
            na.rm = TRUE
        ),

    median_correlation_margin =
        median(
            validation$correlation_margin,
            na.rm = TRUE
        )
)

fwrite(
    validation,
    file.path(
        TABLE_DIR,
        "phase5D3_seed_cluster_assignment_validation.tsv"
    ),
    sep = "\t"
)

fwrite(
    validation_summary,
    file.path(
        TABLE_DIR,
        "phase5D3_seed_cluster_validation_summary.tsv"
    ),
    sep = "\t"
)

confusion_table <- as.data.table(
    table(
        original_cluster =
            validation$original_trajectory_cluster,

        centroid_assignment =
            validation$nearest_cluster
    )
)

fwrite(
    confusion_table,
    file.path(
        TABLE_DIR,
        "phase5D3_seed_cluster_confusion_matrix.tsv"
    ),
    sep = "\t"
)

# ============================================================
# Assignment summaries
# ============================================================

assignment_summary <- assignment_table[
    ,
    .(
        n_genes = .N,

        median_best_correlation =
            median(
                best_centroid_correlation,
                na.rm = TRUE
            ),

        median_correlation_margin =
            median(
                correlation_margin,
                na.rm = TRUE
            ),

        median_spline_FDR =
            median(
                spline_FDR,
                na.rm = TRUE
            ),

        median_predicted_change =
            median(
                predicted_earliest_to_latest_change,
                na.rm = TRUE
            ),

        median_age_spearman =
            median(
                predicted_age_spearman_r,
                na.rm = TRUE
            )
    ),
    by = .(
        nearest_cluster,
        assignment_confidence
    )
]

setorder(
    assignment_summary,
    nearest_cluster,
    assignment_confidence
)

fwrite(
    assignment_summary,
    file.path(
        TABLE_DIR,
        "phase5D3_centroid_assignment_summary.tsv"
    ),
    sep = "\t"
)

# ============================================================
# Full-program centroid summaries
# High and moderate confidence genes only
# ============================================================

centroid_rows <- list()

for (cluster_id in cluster_ids) {
    selected_genes <- assignment_table[
        confident_cluster == cluster_id,
        gene_symbol
    ]

    selected_matrix <- supported_z[
        selected_genes,
        ,
        drop = FALSE
    ]

    if (nrow(selected_matrix) < 10) {
        stop(
            "Too few confidently assigned genes for cluster ",
            cluster_id
        )
    }

    centroid_rows[[as.character(cluster_id)]] <- data.table(
        trajectory_cluster =
            cluster_id,

        prediction_index =
            seq_len(PREDICTION_POINTS),

        developmental_time_transformed =
            prediction_time,

        developmental_age_years_from_birth =
            prediction_age,

        mean_z =
            colMeans(
                selected_matrix,
                na.rm = TRUE
            ),

        median_z =
            apply(
                selected_matrix,
                2,
                median,
                na.rm = TRUE
            ),

        q25_z =
            apply(
                selected_matrix,
                2,
                quantile,
                probs = 0.25,
                na.rm = TRUE
            ),

        q75_z =
            apply(
                selected_matrix,
                2,
                quantile,
                probs = 0.75,
                na.rm = TRUE
            ),

        n_confident_genes =
            nrow(selected_matrix)
    )
}

full_centroids <- rbindlist(
    centroid_rows
)

fwrite(
    full_centroids,
    file.path(
        SOURCE_DIR,
        "Figure39_source_full_program_centroids.tsv.gz"
    ),
    sep = "\t",
    compress = "gzip"
)

# ============================================================
# Figure 39
# ============================================================

cluster_labels <- full_centroids[
    ,
    .(
        n_confident_genes =
            unique(
                n_confident_genes
            )
    ),
    by = trajectory_cluster
]

cluster_labels[
    ,
    panel_label := paste0(
        "Program ",
        trajectory_cluster,
        " (n=",
        n_confident_genes,
        ")"
    )
]

full_centroids <- merge(
    full_centroids,
    cluster_labels,
    by = "trajectory_cluster",
    all.x = TRUE
)

full_centroids[
    ,
    panel_label := factor(
        panel_label,
        levels = cluster_labels[
            order(
                trajectory_cluster
            ),
            panel_label
        ]
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
    "Prenatal\n-0.6 y",
    "-0.4 y",
    "-0.2 y",
    "Birth",
    "6 mo",
    "1 y",
    "5 y",
    "10 y",
    "20 y",
    "40 y"
)

figure_39 <- ggplot(
    full_centroids,
    aes(
        x =
            developmental_time_transformed,
        y = mean_z
    )
) +
    geom_hline(
        yintercept = 0,
        linetype = 2,
        linewidth = 0.35
    ) +
    geom_ribbon(
        aes(
            ymin = q25_z,
            ymax = q75_z
        ),
        alpha = 0.20
    ) +
    geom_line(
        linewidth = 1
    ) +
    facet_wrap(
        ~ panel_label,
        ncol = 2,
        scales = "free_y"
    ) +
    scale_x_continuous(
        breaks = axis_positions,
        labels = axis_labels
    ) +
    labs(
        title = paste0(
            "Figure 39. Full developmental gene programs ",
            "defined by centroid assignment"
        ),
        subtitle = paste0(
            "High- and moderate-confidence assignments among ",
            nrow(assignment_table),
            " developmentally supported genes"
        ),
        x = "Developmental age relative to birth",
        y = "Standardized predicted expression"
    ) +
    theme_bw(
        base_size = 11
    ) +
    theme(
        plot.title = element_text(
            size = 14,
            face = "bold"
        ),
        strip.text = element_text(
            size = 10,
            face = "bold"
        ),
        axis.text.x = element_text(
            size = 8
        )
    )

ggsave(
    filename = file.path(
        FIGURE_DIR,
        "Figure39_full_developmental_gene_programs.pdf"
    ),
    plot = figure_39,
    width = 14,
    height = 7,
    units = "in"
)

ggsave(
    filename = file.path(
        FIGURE_DIR,
        "Figure39_full_developmental_gene_programs.png"
    ),
    plot = figure_39,
    width = 14,
    height = 7,
    units = "in",
    dpi = 300
)

# ============================================================
# Completion summary
# ============================================================

completion <- data.table(
    supported_genes_available =
        nrow(assignment_table),

    high_confidence_assignments =
        sum(
            assignment_table$assignment_confidence ==
                "high"
        ),

    moderate_confidence_assignments =
        sum(
            assignment_table$assignment_confidence ==
                "moderate"
        ),

    low_confidence_assignments =
        sum(
            assignment_table$assignment_confidence ==
                "low"
        ),

    ambiguous_assignments =
        sum(
            assignment_table$assignment_confidence ==
                "ambiguous"
        ),

    confidently_assigned_genes =
        sum(
            !is.na(
                assignment_table$confident_cluster
            )
        ),

    original_seed_cluster_genes =
        validation_summary$original_clustered_genes,

    seed_assignment_agreement_fraction =
        validation_summary$assignment_agreement_fraction,

    learned_clusters =
        length(cluster_ids),

    Phase5D3_status =
        "completed"
)

fwrite(
    completion,
    file.path(
        TABLE_DIR,
        "phase5D3_completion_summary.tsv"
    ),
    sep = "\t"
)

report_file <- file.path(
    META_DIR,
    "PHASE5D3_FULL_GENE_PROGRAM_ASSIGNMENT_REPORT.txt"
)

sink(report_file)

cat(
    "Phase 5D3 full developmental gene-program assignment\n\n"
)

cat(
    "All genes passing both spline-FDR and developmental-range "
)

cat(
    "thresholds were assigned to the two learned trajectory "
)

cat(
    "centroids by Pearson trajectory correlation.\n\n"
)

cat("Completion summary\n")

print(completion)

cat("\nSeed-cluster validation\n")

print(validation_summary)

cat("\nAssignment summary\n")

print(assignment_summary)

cat(
    "\nHigh- and moderate-confidence assignments were used "
)

cat(
    "to calculate the full-program centroids shown in Figure 39.\n"
)

sink()

log_message("")
log_message("===== Phase 5D3 completed =====")

log_message(
    paste(
        capture.output(
            print(completion)
        ),
        collapse = "\n"
    )
)
