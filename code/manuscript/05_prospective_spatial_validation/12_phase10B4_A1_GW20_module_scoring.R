options(
    stringsAsFactors = FALSE,
    warn = 1
)

args <- commandArgs(
    trailingOnly = TRUE
)

module_lock_path <- args[1]
sn_object_path <- args[2]
visium_object_path <- args[3]
sn_metadata_path <- args[4]
visium_metadata_path <- args[5]
out_dir <- args[6]

suppressPackageStartupMessages(
    library(Matrix)
)

suppressPackageStartupMessages(
    library(SeuratObject)
)

suppressPackageStartupMessages(
    library(data.table)
)

score_dir <- file.path(
    out_dir,
    "01_scores"
)

summary_dir <- file.path(
    out_dir,
    "02_summaries"
)

design_dir <- file.path(
    out_dir,
    "03_design_audit"
)

audit_dir <- file.path(
    out_dir,
    "04_audit"
)

for (
    directory in c(
        score_dir,
        summary_dir,
        design_dir,
        audit_dir
    )
) {
    dir.create(
        directory,
        recursive = TRUE,
        showWarnings = FALSE
    )
}

normalize_name <- function(x) {
    gsub(
        "(^_+|_+$)",
        "",
        gsub(
            "[^a-z0-9]+",
            "_",
            tolower(x)
        )
    )
}

lock <- read.delim(
    module_lock_path,
    check.names = FALSE
)

normalized <- setNames(
    colnames(lock),
    normalize_name(
        colnames(lock)
    )
)

gene_candidates <- c(
    "gene_symbol",
    "gene",
    "symbol",
    "hgnc_symbol",
    "module_gene",
    "member_gene"
)

module_candidates <- c(
    "module_id",
    "module",
    "module_name",
    "dthi_module",
    "program",
    "seed_module",
    "gene_set",
    "geneset"
)

gene_key <- gene_candidates[
    gene_candidates %in% names(normalized)
][1]

module_key <- module_candidates[
    module_candidates %in% names(normalized)
][1]

if (
    is.na(gene_key)
    || is.na(module_key)
) {
    stop(
        "Could not identify module and gene columns in the locked source."
    )
}

gene_col <- normalized[[gene_key]]
module_col <- normalized[[module_key]]

module_map <- unique(
    data.frame(
        module = trimws(
            as.character(
                lock[[module_col]]
            )
        ),
        gene_symbol = toupper(
            trimws(
                as.character(
                    lock[[gene_col]]
                )
            )
        ),
        stringsAsFactors = FALSE
    )
)

module_map <- module_map[
    nzchar(module_map$module)
    & nzchar(module_map$gene_symbol),
    ,
    drop = FALSE
]

modules <- sort(
    unique(
        module_map$module
    )
)

if (length(modules) != 9L) {
    stop(
        "Expected exactly nine locked DTHI modules."
    )
}

read_metadata <- function(path) {
    read.delim(
        gzfile(path),
        check.names = FALSE
    )
}

write_gz_table <- function(
    x,
    path
) {
    connection <- gzfile(
        path,
        open = "wt"
    )

    on.exit(
        close(connection),
        add = TRUE
    )

    write.table(
        x,
        connection,
        sep = "\t",
        quote = FALSE,
        row.names = FALSE
    )
}

get_data_matrix <- function(
    object,
    assay_name
) {
    result <- tryCatch(
        GetAssayData(
            object = object,
            assay = assay_name,
            layer = "data"
        ),
        error = function(error) {
            NULL
        }
    )

    if (is.null(result)) {
        result <- tryCatch(
            GetAssayData(
                object = object,
                assay = assay_name,
                slot = "data"
            ),
            error = function(error) {
                NULL
            }
        )
    }

    if (is.null(result)) {
        stop(
            paste0(
                "Could not access normalized data for assay ",
                assay_name,
                "."
            )
        )
    }

    if (
        nrow(result) == 0L
        || ncol(result) == 0L
    ) {
        stop(
            paste0(
                "Normalized data matrix is empty for assay ",
                assay_name,
                "."
            )
        )
    }

    result
}

compute_scores <- function(
    expression_matrix,
    module_map,
    modules
) {
    feature_upper <- toupper(
        rownames(expression_matrix)
    )

    if (anyDuplicated(feature_upper)) {
        stop(
            "Upper-cased assay feature names are duplicated."
        )
    }

    score_matrix <- matrix(
        NA_real_,
        nrow = ncol(expression_matrix),
        ncol = length(modules),
        dimnames = list(
            colnames(expression_matrix),
            modules
        )
    )

    qc_rows <- vector(
        "list",
        length(modules)
    )

    for (
        module_index in seq_along(modules)
    ) {
        module_name <- modules[module_index]

        requested <- sort(
            unique(
                module_map$gene_symbol[
                    module_map$module == module_name
                ]
            )
        )

        feature_index <- match(
            requested,
            feature_upper
        )

        matched <- !is.na(
            feature_index
        )

        submatrix <- expression_matrix[
            feature_index[matched],
            ,
            drop = FALSE
        ]

        gene_means <- Matrix::rowMeans(
            submatrix
        )

        squared <- submatrix
        squared@x <- squared@x ^ 2

        gene_variances <- pmax(
            Matrix::rowMeans(squared)
            - gene_means ^ 2,
            0
        )

        gene_sds <- sqrt(
            gene_variances
        )

        valid <- is.finite(gene_sds) &
            gene_sds > 0

        if (!any(valid)) {
            stop(
                paste0(
                    "No variable genes remained for module ",
                    module_name,
                    "."
                )
            )
        }

        scaled_sparse <- Diagonal(
            x = 1 / gene_sds[valid]
        ) %*% submatrix[
            valid,
            ,
            drop = FALSE
        ]

        score <- as.numeric(
            Matrix::colMeans(
                scaled_sparse
            )
        ) - mean(
            gene_means[valid]
            / gene_sds[valid]
        )

        score_matrix[
            ,
            module_index
        ] <- score

        qc_rows[[module_index]] <- data.frame(
            module = module_name,
            locked_genes = length(requested),
            assay_genes_matched = sum(matched),
            variable_genes_scored = sum(valid),
            constant_genes_excluded = (
                sum(matched) - sum(valid)
            ),
            score_mean = mean(score),
            score_sd = stats::sd(score),
            score_min = min(score),
            score_max = max(score),
            all_scores_finite = all(
                is.finite(score)
            ),
            stringsAsFactors = FALSE
        )
    }

    list(
        scores = score_matrix,
        qc = do.call(
            rbind,
            qc_rows
        )
    )
}

summarize_scores <- function(
    score_table,
    group_columns,
    modules
) {
    dt <- as.data.table(
        score_table
    )

    long <- melt(
        dt,
        id.vars = group_columns,
        measure.vars = modules,
        variable.name = "module",
        value.name = "score",
        variable.factor = FALSE
    )

    long[
        ,
        .(
            units = .N,
            mean_score = mean(score),
            sd_score = stats::sd(score),
            median_score = stats::median(score),
            q25_score = as.numeric(
                stats::quantile(
                    score,
                    0.25,
                    names = FALSE
                )
            ),
            q75_score = as.numeric(
                stats::quantile(
                    score,
                    0.75,
                    names = FALSE
                )
            )
        ),
        by = c(
            group_columns,
            "module"
        )
    ]
}

message(
    "Reading snRNA-seq object and RNA normalized data"
)

sn_object <- readRDS(
    sn_object_path
)

sn_matrix <- get_data_matrix(
    sn_object,
    "RNA"
)

sn_metadata <- read_metadata(
    sn_metadata_path
)

sn_match <- match(
    colnames(sn_matrix),
    sn_metadata$unit_id
)

if (any(is.na(sn_match))) {
    stop(
        "Not all snRNA-seq matrix columns matched frozen metadata."
    )
}

sn_metadata <- sn_metadata[
    sn_match,
    ,
    drop = FALSE
]

if (
    !identical(
        as.character(sn_metadata$unit_id),
        colnames(sn_matrix)
    )
) {
    stop(
        "snRNA-seq metadata order does not match the expression matrix."
    )
}

sn_result <- compute_scores(
    sn_matrix,
    module_map,
    modules
)

sn_score_table <- cbind(
    sn_metadata[
        ,
        c(
            "object_label",
            "unit_id",
            "analysis_sample_id",
            "donor_id",
            "developmental_age_weeks",
            "cortical_lobe",
            "cortical_area_context",
            "author_identity",
            "seurat_cluster"
        ),
        drop = FALSE
    ],
    as.data.frame(
        sn_result$scores,
        check.names = FALSE
    )
)

write_gz_table(
    sn_score_table,
    file.path(
        score_dir,
        "phase10B4_A1_snRNAseq_module_scores.tsv.gz"
    )
)

write.table(
    sn_result$qc,
    file.path(
        audit_dir,
        "phase10B4_A1_snRNAseq_score_QC.tsv"
    ),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)

write.table(
    summarize_scores(
        sn_score_table,
        c(
            "analysis_sample_id",
            "donor_id",
            "cortical_lobe",
            "cortical_area_context"
        ),
        modules
    ),
    file.path(
        summary_dir,
        "phase10B4_A1_snRNAseq_sample_module_summary.tsv"
    ),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)

write.table(
    summarize_scores(
        sn_score_table,
        c("author_identity"),
        modules
    ),
    file.path(
        summary_dir,
        "phase10B4_A1_snRNAseq_identity_module_summary.tsv"
    ),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)

sn_all_finite <- all(
    sn_result$qc$all_scores_finite
)

sn_score_module_count <- ncol(
    sn_result$scores
)

rm(
    sn_object,
    sn_matrix,
    sn_result
)

invisible(
    gc(
        verbose = FALSE
    )
)

message(
    "Reading Visium object and Spatial normalized data"
)

visium_object <- readRDS(
    visium_object_path
)

visium_matrix <- get_data_matrix(
    visium_object,
    "Spatial"
)

visium_metadata <- read_metadata(
    visium_metadata_path
)

visium_match <- match(
    colnames(visium_matrix),
    visium_metadata$unit_id
)

if (any(is.na(visium_match))) {
    stop(
        "Not all Visium matrix columns matched frozen metadata."
    )
}

visium_metadata <- visium_metadata[
    visium_match,
    ,
    drop = FALSE
]

if (
    !identical(
        as.character(visium_metadata$unit_id),
        colnames(visium_matrix)
    )
) {
    stop(
        "Visium metadata order does not match the expression matrix."
    )
}

visium_result <- compute_scores(
    visium_matrix,
    module_map,
    modules
)

visium_score_table <- cbind(
    visium_metadata[
        ,
        c(
            "object_label",
            "unit_id",
            "analysis_sample_id",
            "donor_id",
            "developmental_age_weeks",
            "cortical_lobe",
            "cortical_area_context",
            "author_cluster_id",
            "author_cluster_label",
            "broad_compartment",
            "area_assignment",
            "cluster_annotation_mapped",
            "array_row",
            "array_col",
            "image_row",
            "image_col"
        ),
        drop = FALSE
    ],
    as.data.frame(
        visium_result$scores,
        check.names = FALSE
    )
)

write_gz_table(
    visium_score_table,
    file.path(
        score_dir,
        "phase10B4_A1_Visium_module_scores.tsv.gz"
    )
)

write.table(
    visium_result$qc,
    file.path(
        audit_dir,
        "phase10B4_A1_Visium_score_QC.tsv"
    ),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)

write.table(
    summarize_scores(
        visium_score_table,
        c("area_assignment"),
        modules
    ),
    file.path(
        summary_dir,
        "phase10B4_A1_Visium_area_module_summary.tsv"
    ),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)

write.table(
    summarize_scores(
        visium_score_table,
        c("broad_compartment"),
        modules
    ),
    file.path(
        summary_dir,
        "phase10B4_A1_Visium_compartment_module_summary.tsv"
    ),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)

write.table(
    summarize_scores(
        visium_score_table,
        c(
            "author_cluster_id",
            "author_cluster_label",
            "cluster_annotation_mapped"
        ),
        modules
    ),
    file.path(
        summary_dir,
        "phase10B4_A1_Visium_cluster_module_summary.tsv"
    ),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)

visium_all_finite <- all(
    visium_result$qc$all_scores_finite
)

visium_score_module_count <- ncol(
    visium_result$scores
)

frontal_donors <- sort(
    unique(
        sn_score_table$donor_id[
            sn_score_table$cortical_lobe == "frontal"
        ]
    )
)

occipital_donors <- sort(
    unique(
        sn_score_table$donor_id[
            sn_score_table$cortical_lobe == "occipital"
        ]
    )
)

shared_region_donors <- intersect(
    frontal_donors,
    occipital_donors
)

design_audit <- data.frame(
    snRNAseq_independent_donors = length(
        unique(sn_score_table$donor_id)
    ),
    snRNAseq_analysis_samples = length(
        unique(sn_score_table$analysis_sample_id)
    ),
    frontal_donors = paste(
        frontal_donors,
        collapse = ";"
    ),
    occipital_donors = paste(
        occipital_donors,
        collapse = ";"
    ),
    donors_shared_between_frontal_and_occipital = paste(
        shared_region_donors,
        collapse = ";"
    ),
    frontal_occipital_comparison_confounded_by_donor = (
        length(shared_region_donors) == 0L
    ),
    Visium_independent_donors = length(
        unique(visium_score_table$donor_id)
    ),
    Visium_analysis_samples = length(
        unique(visium_score_table$analysis_sample_id)
    ),
    unresolved_cluster_14_retained = any(
        visium_score_table$author_cluster_id == 14L
        & !visium_score_table$cluster_annotation_mapped
    ),
    inferential_tests_performed = FALSE,
    stringsAsFactors = FALSE
)

write.table(
    design_audit,
    file.path(
        design_dir,
        "phase10B4_A1_design_limitation_audit.tsv"
    ),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)

all_finite <- (
    sn_all_finite
    && visium_all_finite
)

status_value <- if (
    all_finite
    && sn_score_module_count == 9L
    && visium_score_module_count == 9L
) {
    "passed_phase10B4_A1_deterministic_GW20_module_scores_ready_for_descriptive_spatial_summaries_and_block_aware_testing"
} else {
    "phase10B4_A1_requires_manual_review"
}

status <- data.frame(
    phase = "phase10B4_A1",
    locked_modules_scored = length(modules),
    snRNAseq_nuclei_scored = nrow(sn_score_table),
    Visium_spots_scored = nrow(visium_score_table),
    score_definition = "mean_gene_zscore_within_modality",
    primary_snRNAseq_assay = "RNA_data",
    primary_Visium_assay = "Spatial_data",
    all_scores_finite = all_finite,
    inferential_tests_performed = FALSE,
    region_donor_confounding_flag = (
        design_audit$frontal_occipital_comparison_confounded_by_donor
    ),
    unresolved_Visium_cluster_14_retained = (
        design_audit$unresolved_cluster_14_retained
    ),
    candidate_TFs_changed = FALSE,
    validation_hypotheses_changed = FALSE,
    phase10B4_A1_status = status_value,
    stringsAsFactors = FALSE
)

write.table(
    status,
    file.path(
        out_dir,
        "phase10B4_A1_status.tsv"
    ),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)

report <- c(
    "===== PHASE 10B4-A1 DETERMINISTIC GW20 MODULE SCORING =====",
    "",
    paste0(
        "Locked modules scored: ",
        length(modules)
    ),
    paste0(
        "snRNA-seq nuclei scored: ",
        nrow(sn_score_table)
    ),
    paste0(
        "Visium spots scored: ",
        nrow(visium_score_table)
    ),
    "Score definition: mean gene-wise z-score within each modality",
    "Primary snRNA-seq assay: RNA/data",
    "Primary Visium assay: Spatial/data",
    paste0(
        "All scores finite: ",
        all_finite
    ),
    paste0(
        "Frontal-occipital comparison confounded by donor: ",
        design_audit$frontal_occipital_comparison_confounded_by_donor
    ),
    paste0(
        "Unresolved Visium cluster 14 retained: ",
        design_audit$unresolved_cluster_14_retained
    ),
    "Inferential tests performed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    paste0(
        "PHASE 10B4-A1 STATUS: ",
        status_value
    )
)

writeLines(
    report,
    file.path(
        out_dir,
        "phase10B4_A1_report.txt"
    )
)

cat(
    paste(
        report,
        collapse = "\n"
    ),
    "\n"
)
