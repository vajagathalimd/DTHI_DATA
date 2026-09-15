#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(data.table)
    library(ggplot2)
})


PROJECT <- normalizePath(
    ".",
    mustWork = TRUE
)


ASSIGNMENT_FILE <- file.path(
    PROJECT,
    "07_tables",
    "main_tables",
    "phase5",
    "phase5D3_all_supported_gene_centroid_assignments.tsv.gz"
)


FILTER_FILE <- file.path(
    PROJECT,
    "07_tables",
    "main_tables",
    "phase5",
    "phase5D2_gene_filtering_summary.tsv.gz"
)


RESOURCE_FILE <- file.path(
    PROJECT,
    "03_processed_data",
    "developmental_trajectory",
    "phase5D",
    "phase5D4_R1",
    "resources",
    "phase5D4_R1_all_five_sources_gene_sets.rds"
)


RESOURCE_AUDIT_FILE <- file.path(
    PROJECT,
    "07_tables",
    "main_tables",
    "phase5",
    "phase5D4_R1",
    "phase5D4_R1_msigdbr_resource_freeze_audit.tsv"
)


PROCESSED_DIR <- file.path(
    PROJECT,
    "03_processed_data",
    "developmental_trajectory",
    "phase5D",
    "phase5D4_R1",
    "enrichment"
)


TABLE_DIR <- file.path(
    PROJECT,
    "07_tables",
    "main_tables",
    "phase5",
    "phase5D4_R1"
)


FIGURE_DIR <- file.path(
    PROJECT,
    "06_figures",
    "main_figures",
    "phase5",
    "phase5D4_R1"
)


SOURCE_DATA_DIR <- file.path(
    PROJECT,
    "06_figures",
    "source_data",
    "phase5",
    "phase5D4_R1"
)


METADATA_DIR <- file.path(
    PROJECT,
    "02_metadata",
    "phase5",
    "phase5D4_R1"
)


LOG_FILE <- file.path(
    PROJECT,
    "09_pipeline_logs",
    "phase5",
    "phase5D4_R1",
    "phase5D4_R1_functional_enrichment.log"
)


for (directory in c(
    PROCESSED_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    SOURCE_DATA_DIR,
    METADATA_DIR,
    dirname(LOG_FILE)
)) {
    dir.create(
        directory,
        recursive = TRUE,
        showWarnings = FALSE
    )
}


cat(
    "",
    file = LOG_FILE
)


log_message <- function(...) {
    text <- paste0(...)

    cat(
        text,
        "\n"
    )

    cat(
        text,
        "\n",
        file = LOG_FILE,
        append = TRUE
    )
}


normalize_symbol <- function(values) {
    values <- toupper(
        trimws(
            as.character(values)
        )
    )

    values[
        is.na(values)
        |
        values == ""
    ] <- NA_character_

    values
}


as_logical_flag <- function(values) {
    normalized <- tolower(
        trimws(
            as.character(values)
        )
    )

    normalized %in% c(
        "true",
        "1",
        "yes",
        "passed"
    )
}


clean_term_name <- function(values) {
    values <- as.character(values)

    prefixes <- c(
        "^HALLMARK_",
        "^REACTOME_",
        "^GOBP_",
        "^GOCC_",
        "^GOMF_"
    )

    for (prefix in prefixes) {
        values <- sub(
            prefix,
            "",
            values
        )
    }

    values <- gsub(
        "_",
        " ",
        values,
        fixed = TRUE
    )

    values <- tolower(values)

    tools::toTitleCase(values)
}


require_file <- function(path) {
    if (!file.exists(path)) {
        stop(
            paste0(
                "Required file not found: ",
                path
            )
        )
    }
}


read_input_table <- function(path) {
    require_file(path)

    result <- fread(
        path,
        sep = "\t",
        header = TRUE,
        na.strings = c(
            "",
            "NA",
            "NaN"
        ),
        showProgress = FALSE
    )

    if (nrow(result) == 0L) {
        stop(
            paste0(
                "Input table is empty: ",
                path
            )
        )
    }

    result
}


validate_columns <- function(
    table,
    required_columns,
    table_name
) {
    missing_columns <- setdiff(
        required_columns,
        names(table)
    )

    if (length(missing_columns) > 0L) {
        stop(
            paste0(
                table_name,
                " is missing required columns: ",
                paste(
                    missing_columns,
                    collapse = ", "
                )
            )
        )
    }
}


prepare_background <- function(filter_table) {
    validate_columns(
        filter_table,
        c(
            "gene_symbol",
            "trajectory_model_pass"
        ),
        "Phase 5D2 filtering table"
    )

    filter_table[
        ,
        gene_symbol := normalize_symbol(
            gene_symbol
        )
    ]

    filter_table[
        ,
        trajectory_model_pass_flag :=
            as_logical_flag(
                trajectory_model_pass
            )
    ]

    background <- sort(
        unique(
            filter_table[
                trajectory_model_pass_flag == TRUE
                &
                !is.na(gene_symbol),
                gene_symbol
            ]
        )
    )

    if (length(background) != 17762L) {
        stop(
            paste0(
                "Unexpected trajectory-model background size: ",
                length(background),
                "; expected 17762."
            )
        )
    }

    background
}


prepare_program_assignments <- function(
    assignments,
    trajectory_background
) {
    validate_columns(
        assignments,
        c(
            "gene_symbol",
            "assignment_confidence",
            "confident_cluster",
            "spline_FDR",
            "predicted_log2_range",
            "predicted_earliest_to_latest_change",
            "predicted_age_spearman_r",
            "peak_age_label",
            "trough_age_label"
        ),
        "Phase 5D3 assignment table"
    )

    assignments[
        ,
        gene_symbol := normalize_symbol(
            gene_symbol
        )
    ]

    assignments[
        ,
        assignment_confidence :=
            tolower(
                trimws(
                    as.character(
                        assignment_confidence
                    )
                )
            )
    ]

    assignments[
        ,
        confident_cluster :=
            as.integer(
                confident_cluster
            )
    ]

    foreground <- assignments[
        assignment_confidence %chin% c(
            "high",
            "moderate"
        )
        &
        confident_cluster %in% c(
            1L,
            2L
        )
        &
        !is.na(gene_symbol)
    ]

    foreground[
        ,
        developmental_program :=
            fifelse(
                confident_cluster == 1L,
                "maturation_high_increasing",
                "fetal_high_decreasing"
            )
    ]

    foreground[
        ,
        endpoint_direction_consistent :=
            fifelse(
                developmental_program ==
                    "maturation_high_increasing",
                predicted_earliest_to_latest_change > 0,
                predicted_earliest_to_latest_change < 0
            )
    ]

    foreground <- unique(
        foreground[
            gene_symbol %chin%
                trajectory_background,
            .(
                gene_symbol,
                developmental_program,
                assignment_confidence,
                confident_cluster,
                spline_FDR,
                predicted_log2_range,
                predicted_earliest_to_latest_change,
                predicted_age_spearman_r,
                peak_age_label,
                trough_age_label,
                endpoint_direction_consistent
            )
        ]
    )

    duplicate_genes <- foreground[
        ,
        .N,
        by = gene_symbol
    ][
        N > 1L
    ]

    if (nrow(duplicate_genes) > 0L) {
        stop(
            paste0(
                "Duplicate confident program genes detected: ",
                nrow(duplicate_genes)
            )
        )
    }

    program_counts <- foreground[
        ,
        .(
            n_genes = uniqueN(
                gene_symbol
            )
        ),
        by = developmental_program
    ]

    expected_counts <- data.table(
        developmental_program = c(
            "maturation_high_increasing",
            "fetal_high_decreasing"
        ),
        expected_n_genes = c(
            3333L,
            5412L
        )
    )

    count_check <- merge(
        expected_counts,
        program_counts,
        by = "developmental_program",
        all.x = TRUE,
        sort = FALSE
    )

    count_check[
        is.na(n_genes),
        n_genes := 0L
    ]

    if (
        any(
            count_check$n_genes !=
                count_check$expected_n_genes
        )
    ) {
        stop(
            paste0(
                "Program-gene counts do not match Phase 5D3. ",
                paste(
                    paste0(
                        count_check$developmental_program,
                        "=",
                        count_check$n_genes,
                        " expected ",
                        count_check$expected_n_genes
                    ),
                    collapse = "; "
                )
            )
        )
    }

    if (uniqueN(foreground$gene_symbol) != 8745L) {
        stop(
            paste0(
                "Expected 8745 confident program genes; observed ",
                uniqueN(foreground$gene_symbol),
                "."
            )
        )
    }

    overlap <- intersect(
        foreground[
            developmental_program ==
                "maturation_high_increasing",
            gene_symbol
        ],
        foreground[
            developmental_program ==
                "fetal_high_decreasing",
            gene_symbol
        ]
    )

    if (length(overlap) != 0L) {
        stop(
            paste0(
                "Developmental-program overlap detected: ",
                length(overlap)
            )
        )
    }

    endpoint_discordant <- foreground[
        endpoint_direction_consistent == FALSE
    ]

    if (nrow(endpoint_discordant) != 27L) {
        stop(
            paste0(
                "Expected 27 endpoint-direction-discordant genes; observed ",
                nrow(endpoint_discordant),
                "."
            )
        )
    }

    if (
        any(
            endpoint_discordant$developmental_program !=
                "fetal_high_decreasing"
        )
        ||
        any(
            endpoint_discordant$assignment_confidence !=
                "moderate"
        )
    ) {
        stop(
            paste0(
                "The 27 endpoint-discordant genes do not match ",
                "the validated Phase 5D4-R1 derivation audit."
            )
        )
    }

    setorder(
        foreground,
        developmental_program,
        gene_symbol
    )

    foreground
}


perform_ora <- function(
    foreground_genes,
    trajectory_background,
    term_to_gene,
    program_name,
    source_name,
    minimum_set_size = 10L,
    maximum_set_size = 500L
) {
    source_table <- unique(
        term_to_gene[
            enrichment_source == source_name
            &
            !is.na(gene_symbol)
            &
            !is.na(ID)
            &
            ID != "",
            .(
                ID,
                Description,
                term_name,
                gene_symbol
            )
        ]
    )

    if (nrow(source_table) == 0L) {
        stop(
            paste0(
                "Frozen resource is empty for ",
                source_name,
                "."
            )
        )
    }

    source_genes <- unique(
        source_table$gene_symbol
    )

    mapped_background <- sort(
        intersect(
            trajectory_background,
            source_genes
        )
    )

    mapped_foreground <- sort(
        intersect(
            foreground_genes,
            mapped_background
        )
    )

    universe_size <- length(
        mapped_background
    )

    foreground_size <- length(
        mapped_foreground
    )

    if (
        universe_size == 0L
        ||
        foreground_size == 0L
    ) {
        stop(
            paste0(
                "No mapped background or foreground genes for ",
                program_name,
                " / ",
                source_name,
                "."
            )
        )
    }

    mapped_term_to_gene <- source_table[
        gene_symbol %chin%
            mapped_background
    ]

    set_sizes <- mapped_term_to_gene[
        ,
        .(
            Description =
                Description[[1L]],
            term_name =
                term_name[[1L]],
            background_gene_count =
                uniqueN(
                    gene_symbol
                )
        ),
        by = ID
    ]

    set_sizes <- set_sizes[
        background_gene_count >=
            minimum_set_size
        &
        background_gene_count <=
            maximum_set_size
    ]

    if (nrow(set_sizes) == 0L) {
        stop(
            paste0(
                "No eligible gene sets remain for ",
                source_name,
                "."
            )
        )
    }

    eligible_term_to_gene <- mapped_term_to_gene[
        ID %chin% set_sizes$ID
    ]

    overlap_table <- eligible_term_to_gene[
        gene_symbol %chin%
            mapped_foreground,
        .(
            Count =
                uniqueN(
                    gene_symbol
                ),
            geneID =
                paste(
                    sort(
                        unique(
                            gene_symbol
                        )
                    ),
                    collapse = "/"
                )
        ),
        by = ID
    ]

    result <- merge(
        set_sizes,
        overlap_table,
        by = "ID",
        all.x = TRUE,
        sort = FALSE
    )

    result[
        is.na(Count),
        Count := 0L
    ]

    result[
        is.na(geneID),
        geneID := ""
    ]

    result[
        ,
        pvalue := phyper(
            q = Count - 1L,
            m = background_gene_count,
            n = universe_size -
                background_gene_count,
            k = foreground_size,
            lower.tail = FALSE
        )
    ]

    result[
        ,
        p_adjust := stats::p.adjust(
            pvalue,
            method = "BH"
        )
    ]

    result[
        ,
        foreground_fraction :=
            Count /
            foreground_size
    ]

    result[
        ,
        background_fraction :=
            background_gene_count /
            universe_size
    ]

    result[
        ,
        enrichment_fold :=
            foreground_fraction /
            background_fraction
    ]

    result[
        ,
        odds_ratio_haldane :=
            (
                Count + 0.5
            )
            *
            (
                universe_size
                -
                background_gene_count
                -
                foreground_size
                +
                Count
                +
                0.5
            )
            /
            (
                (
                    foreground_size
                    -
                    Count
                    +
                    0.5
                )
                *
                (
                    background_gene_count
                    -
                    Count
                    +
                    0.5
                )
            )
    ]

    result[
        ,
        `:=`(
            developmental_program =
                program_name,
            enrichment_source =
                source_name,
            mapped_background_genes =
                universe_size,
            mapped_foreground_genes =
                foreground_size,
            GeneRatio =
                paste0(
                    Count,
                    "/",
                    foreground_size
                ),
            BgRatio =
                paste0(
                    background_gene_count,
                    "/",
                    universe_size
                ),
            FDR_significant =
                p_adjust < 0.05,
            cleaned_term =
                clean_term_name(
                    term_name
                )
        )
    ]

    setorder(
        result,
        p_adjust,
        -Count,
        ID
    )

    result[
        ,
        .(
            developmental_program,
            enrichment_source,
            ID,
            term_name,
            cleaned_term,
            Description,
            mapped_background_genes,
            mapped_foreground_genes,
            background_gene_count,
            Count,
            GeneRatio,
            BgRatio,
            foreground_fraction,
            background_fraction,
            enrichment_fold,
            odds_ratio_haldane,
            pvalue,
            p_adjust,
            FDR_significant,
            geneID
        )
    ]
}


run_phase5D4_R1 <- function() {
    log_message(
        "===== Phase 5D4-R1 functional enrichment started ====="
    )

    log_message(
        "Method: source-specific one-sided hypergeometric ORA"
    )

    log_message(
        "Multiple-testing correction: BH within each program and source"
    )

    log_message(
        "Gene-set size limits: 10–500 mapped trajectory-background genes"
    )

    log_message(
        "MSigDB resources: frozen local 2026.1.Hs release"
    )


    for (path in c(
        ASSIGNMENT_FILE,
        FILTER_FILE,
        RESOURCE_FILE,
        RESOURCE_AUDIT_FILE
    )) {
        require_file(path)
    }


    assignments <- read_input_table(
        ASSIGNMENT_FILE
    )

    filter_table <- read_input_table(
        FILTER_FILE
    )

    resource_audit <- read_input_table(
        RESOURCE_AUDIT_FILE
    )

    frozen_resources <- as.data.table(
        readRDS(
            RESOURCE_FILE
        )
    )


    validate_columns(
        frozen_resources,
        c(
            "enrichment_source",
            "collection",
            "subcollection",
            "ID",
            "term_name",
            "Description",
            "gene_symbol",
            "database_version"
        ),
        "Frozen MSigDB resource"
    )


    validate_columns(
        resource_audit,
        c(
            "enrichment_source",
            "collection",
            "subcollection",
            "msigdbr_version",
            "database_version",
            "resource_rows",
            "unique_terms",
            "unique_gene_symbols",
            "duplicate_term_gene_pairs"
        ),
        "Frozen-resource audit"
    )


    expected_sources <- c(
        "GO_BP",
        "GO_CC",
        "GO_MF",
        "Reactome",
        "Hallmark"
    )


    observed_sources <- unique(
        resource_audit$enrichment_source
    )


    if (
        !identical(
            observed_sources,
            expected_sources
        )
    ) {
        stop(
            paste0(
                "Frozen-resource sources are missing or misordered: ",
                paste(
                    observed_sources,
                    collapse = ", "
                )
            )
        )
    }


    if (
        any(
            resource_audit$duplicate_term_gene_pairs != 0L
        )
    ) {
        stop(
            "Duplicate term-gene pairs were reported by the resource audit."
        )
    }


    if (
        uniqueN(
            frozen_resources$enrichment_source
        ) != 5L
    ) {
        stop(
            "Expected five sources in the frozen combined resource."
        )
    }


    frozen_resources[
        ,
        gene_symbol := normalize_symbol(
            gene_symbol
        )
    ]


    frozen_resources <- unique(
        frozen_resources[
            !is.na(gene_symbol)
            &
            !is.na(ID)
            &
            ID != ""
        ]
    )


    duplicate_resource_pairs <- frozen_resources[
        ,
        .N,
        by = .(
            enrichment_source,
            ID,
            gene_symbol
        )
    ][
        N > 1L
    ]


    if (nrow(duplicate_resource_pairs) > 0L) {
        stop(
            paste0(
                "Duplicate frozen term-gene pairs detected: ",
                nrow(duplicate_resource_pairs)
            )
        )
    }


    trajectory_background <- prepare_background(
        filter_table
    )


    foreground <- prepare_program_assignments(
        assignments,
        trajectory_background
    )


    program_order <- c(
        "maturation_high_increasing",
        "fetal_high_decreasing"
    )


    program_counts <- foreground[
        ,
        .(
            n_genes =
                uniqueN(
                    gene_symbol
                ),
            high_confidence_genes =
                uniqueN(
                    gene_symbol[
                        assignment_confidence == "high"
                    ]
                ),
            moderate_confidence_genes =
                uniqueN(
                    gene_symbol[
                        assignment_confidence == "moderate"
                    ]
                ),
            endpoint_direction_discordant_genes =
                uniqueN(
                    gene_symbol[
                        endpoint_direction_consistent == FALSE
                    ]
                )
        ),
        by = developmental_program
    ]


    program_counts[
        ,
        program_order :=
            match(
                developmental_program,
                program_order
            )
    ]

    setorder(
        program_counts,
        program_order
    )

    program_counts[
        ,
        program_order := NULL
    ]


    program_assignment_file <- file.path(
        PROCESSED_DIR,
        "phase5D4_R1_program_gene_assignments.tsv.gz"
    )


    program_count_file <- file.path(
        TABLE_DIR,
        "phase5D4_R1_program_gene_counts.tsv"
    )


    discordant_file <- file.path(
        TABLE_DIR,
        "phase5D4_R1_endpoint_direction_discordant_genes.tsv"
    )


    fwrite(
        foreground,
        program_assignment_file,
        sep = "\t",
        compress = "gzip"
    )


    fwrite(
        program_counts,
        program_count_file,
        sep = "\t"
    )


    fwrite(
        foreground[
            endpoint_direction_consistent == FALSE
        ],
        discordant_file,
        sep = "\t"
    )


    for (program_name in program_order) {
        gene_file <- file.path(
            PROCESSED_DIR,
            paste0(
                "phase5D4_R1_",
                program_name,
                "_genes.tsv"
            )
        )

        fwrite(
            foreground[
                developmental_program ==
                    program_name,
                .(
                    gene_symbol
                )
            ],
            gene_file,
            sep = "\t"
        )
    }


    background_file <- file.path(
        PROCESSED_DIR,
        "phase5D4_R1_trajectory_model_background_genes.tsv"
    )


    fwrite(
        data.table(
            gene_symbol =
                trajectory_background
        ),
        background_file,
        sep = "\t"
    )


    log_message(
        "Trajectory-model background genes: ",
        length(
            trajectory_background
        )
    )

    log_message(
        "Confident program genes: ",
        uniqueN(
            foreground$gene_symbol
        )
    )

    log_message(
        "Maturation-high genes: ",
        foreground[
            developmental_program ==
                "maturation_high_increasing",
            uniqueN(
                gene_symbol
            )
        ]
    )

    log_message(
        "Fetal-high genes: ",
        foreground[
            developmental_program ==
                "fetal_high_decreasing",
            uniqueN(
                gene_symbol
            )
        ]
    )

    log_message(
        "Endpoint-direction-discordant genes retained: ",
        foreground[
            endpoint_direction_consistent == FALSE,
            uniqueN(
                gene_symbol
            )
        ]
    )


    enrichment_results <- list()
    mapping_audit_rows <- list()

    enrichment_index <- 0L
    mapping_index <- 0L


    for (source_name in expected_sources) {
        source_table <- frozen_resources[
            enrichment_source ==
                source_name
        ]

        source_genes <- unique(
            source_table$gene_symbol
        )

        mapped_background <- intersect(
            trajectory_background,
            source_genes
        )

        mapped_source_table <- source_table[
            gene_symbol %chin%
                mapped_background
        ]

        source_set_sizes <- mapped_source_table[
            ,
            .(
                background_gene_count =
                    uniqueN(
                        gene_symbol
                    )
            ),
            by = ID
        ]

        eligible_source_sets <- source_set_sizes[
            background_gene_count >= 10L
            &
            background_gene_count <= 500L
        ]


        for (program_name in program_order) {
            program_genes <- foreground[
                developmental_program ==
                    program_name,
                gene_symbol
            ]

            mapped_foreground <- intersect(
                program_genes,
                mapped_background
            )

            mapping_index <- mapping_index + 1L

            mapping_audit_rows[[mapping_index]] <- data.table(
                developmental_program =
                    program_name,
                enrichment_source =
                    source_name,
                trajectory_background_genes =
                    length(
                        trajectory_background
                    ),
                source_available_gene_symbols =
                    uniqueN(
                        source_genes
                    ),
                mapped_background_genes =
                    length(
                        mapped_background
                    ),
                background_mapping_fraction =
                    length(
                        mapped_background
                    )
                    /
                    length(
                        trajectory_background
                    ),
                total_program_genes =
                    length(
                        program_genes
                    ),
                mapped_foreground_genes =
                    length(
                        mapped_foreground
                    ),
                foreground_mapping_fraction =
                    length(
                        mapped_foreground
                    )
                    /
                    length(
                        program_genes
                    ),
                source_terms_before_size_filter =
                    uniqueN(
                        source_table$ID
                    ),
                eligible_terms_after_size_filter =
                    nrow(
                        eligible_source_sets
                    ),
                minimum_set_size =
                    10L,
                maximum_set_size =
                    500L
            )


            current_result <- perform_ora(
                foreground_genes =
                    program_genes,
                trajectory_background =
                    trajectory_background,
                term_to_gene =
                    frozen_resources,
                program_name =
                    program_name,
                source_name =
                    source_name,
                minimum_set_size =
                    10L,
                maximum_set_size =
                    500L
            )


            if (nrow(current_result) == 0L) {
                stop(
                    paste0(
                        "No enrichment results for ",
                        program_name,
                        " / ",
                        source_name,
                        "."
                    )
                )
            }


            enrichment_index <- enrichment_index + 1L

            enrichment_results[[enrichment_index]] <- current_result


            log_message(
                program_name,
                " | ",
                source_name,
                " | mapped foreground: ",
                unique(
                    current_result$mapped_foreground_genes
                ),
                " | terms tested: ",
                nrow(
                    current_result
                ),
                " | FDR significant: ",
                sum(
                    current_result$FDR_significant,
                    na.rm = TRUE
                )
            )
        }
    }


    if (length(enrichment_results) != 10L) {
        stop(
            paste0(
                "Expected ten program-source result blocks; observed ",
                length(enrichment_results),
                "."
            )
        )
    }


    all_enrichment <- rbindlist(
        enrichment_results,
        use.names = TRUE,
        fill = TRUE
    )


    mapping_audit <- rbindlist(
        mapping_audit_rows,
        use.names = TRUE,
        fill = TRUE
    )


    all_enrichment[
        ,
        program_order :=
            match(
                developmental_program,
                program_order
            )
    ]


    all_enrichment[
        ,
        source_order :=
            match(
                enrichment_source,
                expected_sources
            )
    ]


    setorder(
        all_enrichment,
        program_order,
        source_order,
        p_adjust,
        -Count,
        ID
    )


    all_enrichment[
        ,
        c(
            "program_order",
            "source_order"
        ) := NULL
    ]


    mapping_audit[
        ,
        program_order :=
            match(
                developmental_program,
                program_order
            )
    ]


    mapping_audit[
        ,
        source_order :=
            match(
                enrichment_source,
                expected_sources
            )
    ]


    setorder(
        mapping_audit,
        program_order,
        source_order
    )


    mapping_audit[
        ,
        c(
            "program_order",
            "source_order"
        ) := NULL
    ]


    if (
        any(
            !is.finite(
                all_enrichment$pvalue
            )
        )
        ||
        any(
            !is.finite(
                all_enrichment$p_adjust
            )
        )
    ) {
        stop(
            "Non-finite enrichment P values were detected."
        )
    }


    if (
        any(
            all_enrichment$pvalue < 0
            |
            all_enrichment$pvalue > 1
        )
        ||
        any(
            all_enrichment$p_adjust < 0
            |
            all_enrichment$p_adjust > 1
        )
    ) {
        stop(
            "Enrichment P values fall outside [0, 1]."
        )
    }


    if (
        any(
            all_enrichment$Count >
                all_enrichment$mapped_foreground_genes
        )
        ||
        any(
            all_enrichment$Count >
                all_enrichment$background_gene_count
        )
    ) {
        stop(
            "Impossible enrichment overlap counts were detected."
        )
    }


    if (
        any(
            all_enrichment$background_gene_count < 10L
        )
        ||
        any(
            all_enrichment$background_gene_count > 500L
        )
    ) {
        stop(
            "Gene-set size filtering was not applied correctly."
        )
    }


    all_results_file <- file.path(
        PROCESSED_DIR,
        "phase5D4_R1_all_functional_enrichment_results.tsv.gz"
    )


    mapping_audit_file <- file.path(
        TABLE_DIR,
        "phase5D4_R1_gene_set_background_mapping_audit.tsv"
    )


    fwrite(
        all_enrichment,
        all_results_file,
        sep = "\t",
        compress = "gzip"
    )


    fwrite(
        mapping_audit,
        mapping_audit_file,
        sep = "\t"
    )


    enrichment_summary <- all_enrichment[
        ,
        {
            ordered_result <- .SD[
                order(
                    p_adjust,
                    -Count,
                    ID
                )
            ]

            top_result <- ordered_result[
                1L
            ]

            list(
                mapped_background_genes =
                    unique(
                        mapped_background_genes
                    ),
                mapped_foreground_genes =
                    unique(
                        mapped_foreground_genes
                    ),
                tested_terms =
                    .N,
                terms_with_overlap =
                    sum(
                        Count > 0L
                    ),
                FDR_significant_terms =
                    sum(
                        FDR_significant,
                        na.rm = TRUE
                    ),
                minimum_pvalue =
                    min(
                        pvalue,
                        na.rm = TRUE
                    ),
                minimum_p_adjust =
                    min(
                        p_adjust,
                        na.rm = TRUE
                    ),
                top_term_ID =
                    top_result$ID,
                top_term =
                    top_result$cleaned_term,
                top_term_overlap =
                    top_result$Count,
                top_term_enrichment_fold =
                    top_result$enrichment_fold,
                top_term_adjusted_P =
                    top_result$p_adjust
            )
        },
        by = .(
            developmental_program,
            enrichment_source
        )
    ]


    enrichment_summary[
        ,
        program_order :=
            match(
                developmental_program,
                program_order
            )
    ]


    enrichment_summary[
        ,
        source_order :=
            match(
                enrichment_source,
                expected_sources
            )
    ]


    setorder(
        enrichment_summary,
        program_order,
        source_order
    )


    enrichment_summary[
        ,
        c(
            "program_order",
            "source_order"
        ) := NULL
    ]


    summary_file <- file.path(
        TABLE_DIR,
        "phase5D4_R1_functional_enrichment_summary.tsv"
    )


    fwrite(
        enrichment_summary,
        summary_file,
        sep = "\t"
    )


    top20 <- all_enrichment[
        order(
            developmental_program,
            enrichment_source,
            p_adjust,
            -Count,
            ID
        ),
        head(
            .SD,
            20L
        ),
        by = .(
            developmental_program,
            enrichment_source
        )
    ]


    top20[
        ,
        rank_within_program_and_source :=
            seq_len(
                .N
            ),
        by = .(
            developmental_program,
            enrichment_source
        )
    ]


    setcolorder(
        top20,
        c(
            "developmental_program",
            "enrichment_source",
            "rank_within_program_and_source",
            setdiff(
                names(top20),
                c(
                    "developmental_program",
                    "enrichment_source",
                    "rank_within_program_and_source"
                )
            )
        )
    )


    top20_file <- file.path(
        TABLE_DIR,
        "phase5D4_R1_top20_terms_per_program_and_source.tsv"
    )


    fwrite(
        top20,
        top20_file,
        sep = "\t"
    )


    figure_source_file <- file.path(
        SOURCE_DATA_DIR,
        "phase5D4_R1_Figure40_source_data.tsv"
    )


    plot_data <- all_enrichment[
        FDR_significant == TRUE
        &
        Count > 0L
    ][
        order(
            developmental_program,
            enrichment_source,
            p_adjust,
            -Count,
            ID
        ),
        head(
            .SD,
            5L
        ),
        by = .(
            developmental_program,
            enrichment_source
        )
    ]


    figure_generated <- FALSE

    figure_png <- file.path(
        FIGURE_DIR,
        "Figure40_R1_developmental_program_functional_enrichment.png"
    )


    figure_pdf <- file.path(
        FIGURE_DIR,
        "Figure40_R1_developmental_program_functional_enrichment.pdf"
    )


    if (nrow(plot_data) > 0L) {
        plot_data[
            ,
            negative_log10_FDR :=
                -log10(
                    pmax(
                        p_adjust,
                        .Machine$double.xmin
                    )
                )
        ]

        plot_data[
            ,
            term_key :=
                paste(
                    developmental_program,
                    enrichment_source,
                    ID,
                    sep = "::"
                )
        ]

        plot_data[
            ,
            term_key :=
                factor(
                    term_key,
                    levels = rev(
                        unique(
                            term_key
                        )
                    )
                )
        ]

        label_map <- setNames(
            plot_data$cleaned_term,
            as.character(
                plot_data$term_key
            )
        )

        fwrite(
            plot_data,
            figure_source_file,
            sep = "\t"
        )

        figure_40_R1 <- ggplot(
            plot_data,
            aes(
                x = negative_log10_FDR,
                y = term_key,
                size = Count,
                colour = enrichment_source
            )
        ) +
            geom_point(
                alpha = 0.85
            ) +
            facet_grid(
                enrichment_source ~
                    developmental_program,
                scales = "free_y",
                space = "free_y"
            ) +
            scale_y_discrete(
                labels = label_map
            ) +
            labs(
                title = paste0(
                    "Functional enrichment of developmental ",
                    "cortical trajectory programs"
                ),
                subtitle = paste0(
                    "One-sided source-specific hypergeometric ORA; ",
                    "BH correction within each program and source"
                ),
                x = expression(
                    -log[10](
                        adjusted~italic(P)
                    )
                ),
                y = NULL,
                size = "Overlap genes",
                colour = "Source"
            ) +
            theme_bw(
                base_size = 10
            ) +
            theme(
                strip.text = element_text(
                    face = "bold"
                ),
                axis.text.y = element_text(
                    size = 6.8
                ),
                legend.position = "bottom",
                panel.grid.major.y = element_blank(),
                plot.title = element_text(
                    face = "bold"
                )
            )

        ggsave(
            filename = figure_png,
            plot = figure_40_R1,
            width = 15,
            height = 13,
            units = "in",
            dpi = 600,
            bg = "white"
        )

        ggsave(
            filename = figure_pdf,
            plot = figure_40_R1,
            width = 15,
            height = 13,
            units = "in",
            bg = "white"
        )

        figure_generated <- TRUE
    }


    list(
        trajectory_background =
            trajectory_background,
        foreground =
            foreground,
        program_counts =
            program_counts,
        mapping_audit =
            mapping_audit,
        all_enrichment =
            all_enrichment,
        enrichment_summary =
            enrichment_summary,
        top20 =
            top20,
        plot_data =
            plot_data,
        figure_generated =
            figure_generated,
        files = list(
            program_assignments =
                program_assignment_file,
            program_counts =
                program_count_file,
            discordant_genes =
                discordant_file,
            background_genes =
                background_file,
            all_results =
                all_results_file,
            mapping_audit =
                mapping_audit_file,
            enrichment_summary =
                summary_file,
            top20 =
                top20_file,
            figure_source =
                figure_source_file,
            figure_png =
                figure_png,
            figure_pdf =
                figure_pdf
        )
    )
}


write_methodology_record <- function() {
    methodology_file <- file.path(
        METADATA_DIR,
        "phase5D4_R1_enrichment_methodology.txt"
    )

    methodology_text <- c(
        "PHASE 5D4-R1 FUNCTIONAL ENRICHMENT",
        "",
        "Input developmental programs:",
        "High- and moderate-confidence Phase 5D3 centroid assignments.",
        "",
        "Program definitions:",
        "Cluster 1: maturation_high_increasing",
        "Cluster 2: fetal_high_decreasing",
        "",
        "Background:",
        "Genes passing Phase 5D2 trajectory modelling.",
        "",
        "Gene-set resource:",
        "MSigDB 2026.1.Hs, frozen locally through msigdbr 26.1.0.",
        "",
        "Sources:",
        "GO Biological Process",
        "GO Cellular Component",
        "GO Molecular Function",
        "Reactome",
        "Hallmark",
        "",
        "Statistical universe:",
        "Source-specific mapped trajectory-model background.",
        "",
        "Gene-set eligibility:",
        "10 to 500 mapped background genes.",
        "",
        "Test:",
        "One-sided hypergeometric over-representation analysis.",
        "",
        "Multiple-testing correction:",
        "Benjamini-Hochberg separately within each developmental program and source.",
        "",
        "Endpoint-direction diagnostic:",
        paste0(
            "Twenty-seven moderate-confidence fetal-high genes with an ",
            "opposite earliest-to-latest endpoint sign are retained because ",
            "program assignment is based on the complete trajectory centroid."
        ),
        "",
        "Historical-output policy:",
        paste0(
            "Historical Phase 5D4 enrichment outputs were not used as ",
            "analytical inputs and remain invalid pending independent ",
            "verification of Phase 5D4-R1."
        )
    )

    writeLines(
        methodology_text,
        methodology_file
    )

    methodology_file
}


build_internal_audit <- function(
    analysis
) {
    all_enrichment <- analysis$all_enrichment
    mapping_audit <- analysis$mapping_audit
    foreground <- analysis$foreground

    expected_sources <- c(
        "GO_BP",
        "GO_CC",
        "GO_MF",
        "Reactome",
        "Hallmark"
    )

    expected_programs <- c(
        "maturation_high_increasing",
        "fetal_high_decreasing"
    )


    recomputed <- copy(
        all_enrichment
    )

    recomputed[
        ,
        recomputed_pvalue := phyper(
            q = Count - 1L,
            m = background_gene_count,
            n = mapped_background_genes -
                background_gene_count,
            k = mapped_foreground_genes,
            lower.tail = FALSE
        )
    ]

    recomputed[
        ,
        recomputed_p_adjust :=
            stats::p.adjust(
                recomputed_pvalue,
                method = "BH"
            ),
        by = .(
            developmental_program,
            enrichment_source
        )
    ]


    maximum_pvalue_difference <- max(
        abs(
            recomputed$pvalue -
                recomputed$recomputed_pvalue
        ),
        na.rm = TRUE
    )


    maximum_adjusted_P_difference <- max(
        abs(
            recomputed$p_adjust -
                recomputed$recomputed_p_adjust
        ),
        na.rm = TRUE
    )


    block_counts <- all_enrichment[
        ,
        .(
            tested_terms =
                .N
        ),
        by = .(
            developmental_program,
            enrichment_source
        )
    ]


    term_count_check <- merge(
        block_counts,
        mapping_audit[
            ,
            .(
                developmental_program,
                enrichment_source,
                eligible_terms_after_size_filter
            )
        ],
        by = c(
            "developmental_program",
            "enrichment_source"
        ),
        all = TRUE,
        sort = FALSE
    )


    historical_program_files <- c(
        maturation_high_increasing =
            file.path(
                PROJECT,
                "03_processed_data",
                "developmental_trajectory",
                "phase5D",
                "enrichment",
                "phase5D4_maturation_high_increasing_genes.tsv"
            ),
        fetal_high_decreasing =
            file.path(
                PROJECT,
                "03_processed_data",
                "developmental_trajectory",
                "phase5D",
                "enrichment",
                "phase5D4_fetal_high_decreasing_genes.tsv"
            )
    )


    historical_assignment_exact_match <- FALSE


    if (
        all(
            file.exists(
                historical_program_files
            )
        )
    ) {
        historical_program_tables <- lapply(
            names(
                historical_program_files
            ),
            function(program_name) {
                historical_table <- fread(
                    historical_program_files[[program_name]],
                    sep = "\t",
                    showProgress = FALSE
                )

                candidate_gene_columns <- intersect(
                    c(
                        "gene_symbol",
                        "gene",
                        "symbol"
                    ),
                    names(
                        historical_table
                    )
                )

                if (
                    length(
                        candidate_gene_columns
                    ) == 0L
                    &&
                    ncol(
                        historical_table
                    ) == 1L
                ) {
                    candidate_gene_columns <-
                        names(
                            historical_table
                        )[[1L]]
                }

                if (
                    length(
                        candidate_gene_columns
                    ) == 0L
                ) {
                    stop(
                        paste0(
                            "Could not identify the gene-symbol column in ",
                            historical_program_files[[program_name]],
                            "."
                        )
                    )
                }

                data.table(
                    gene_symbol =
                        normalize_symbol(
                            historical_table[[candidate_gene_columns[[1L]]]]
                        ),
                    developmental_program =
                        program_name
                )
            }
        )

        historical <- rbindlist(
            historical_program_tables,
            use.names = TRUE,
            fill = TRUE
        )

        historical <- unique(
            historical[
                !is.na(
                    gene_symbol
                )
            ]
        )

        historical_keys <- sort(
            paste(
                historical$gene_symbol,
                historical$developmental_program,
                sep = "::"
            )
        )

        current_keys <- sort(
            paste(
                foreground$gene_symbol,
                foreground$developmental_program,
                sep = "::"
            )
        )

        historical_assignment_exact_match <-
            identical(
                historical_keys,
                current_keys
            )
    }


    source_term_balance <- block_counts[
        ,
        .(
            distinct_tested_term_counts =
                uniqueN(
                    tested_terms
                )
        ),
        by = enrichment_source
    ]


    duplicate_result_rows <- all_enrichment[
        ,
        .N,
        by = .(
            developmental_program,
            enrichment_source,
            ID
        )
    ][
        N > 1L
    ]


    output_files <- unlist(
        analysis$files,
        use.names = TRUE
    )


    required_output_files <- output_files[
        names(output_files) !=
            "figure_source"
    ]


    output_files_present <- all(
        file.exists(
            required_output_files
        )
    )


    output_files_nonempty <- all(
        file.info(
            required_output_files
        )$size > 100
    )


    audit_rows <- list()


    add_audit <- function(
        section,
        item,
        value,
        passed,
        detail
    ) {
        audit_rows[[length(audit_rows) + 1L]] <<- data.table(
            section = section,
            item = item,
            value = as.character(value),
            passed = isTRUE(passed),
            detail = detail
        )
    }


    add_audit(
        "inputs",
        "trajectory_background_genes",
        length(
            analysis$trajectory_background
        ),
        length(
            analysis$trajectory_background
        ) == 17762L,
        "Expected Phase 5D2 trajectory-model background."
    )


    add_audit(
        "inputs",
        "confident_program_genes",
        uniqueN(
            foreground$gene_symbol
        ),
        uniqueN(
            foreground$gene_symbol
        ) == 8745L,
        "High- and moderate-confidence Phase 5D3 assignments."
    )


    add_audit(
        "inputs",
        "maturation_high_genes",
        foreground[
            developmental_program ==
                "maturation_high_increasing",
            uniqueN(
                gene_symbol
            )
        ],
        foreground[
            developmental_program ==
                "maturation_high_increasing",
            uniqueN(
                gene_symbol
            )
        ] == 3333L,
        "Expected validated maturation-high program size."
    )


    add_audit(
        "inputs",
        "fetal_high_genes",
        foreground[
            developmental_program ==
                "fetal_high_decreasing",
            uniqueN(
                gene_symbol
            )
        ],
        foreground[
            developmental_program ==
                "fetal_high_decreasing",
            uniqueN(
                gene_symbol
            )
        ] == 5412L,
        "Expected validated fetal-high program size."
    )


    add_audit(
        "inputs",
        "shared_program_genes",
        length(
            intersect(
                foreground[
                    developmental_program ==
                        "maturation_high_increasing",
                    gene_symbol
                ],
                foreground[
                    developmental_program ==
                        "fetal_high_decreasing",
                    gene_symbol
                ]
            )
        ),
        length(
            intersect(
                foreground[
                    developmental_program ==
                        "maturation_high_increasing",
                    gene_symbol
                ],
                foreground[
                    developmental_program ==
                        "fetal_high_decreasing",
                    gene_symbol
                ]
            )
        ) == 0L,
        "The two trajectory programs must be mutually exclusive."
    )


    add_audit(
        "inputs",
        "endpoint_direction_discordant_genes",
        foreground[
            endpoint_direction_consistent == FALSE,
            uniqueN(
                gene_symbol
            )
        ],
        foreground[
            endpoint_direction_consistent == FALSE,
            uniqueN(
                gene_symbol
            )
        ] == 27L,
        paste0(
            "Retained because full-trajectory centroid assignment, ",
            "not endpoint direction, defines the program."
        )
    )


    add_audit(
        "compatibility",
        "historical_program_assignments_exact_match",
        historical_assignment_exact_match,
        historical_assignment_exact_match,
        paste0(
            "Confirms that Phase 6 used the same 8,745 ",
            "gene-to-program assignments."
        )
    )


    add_audit(
        "resources",
        "enrichment_sources",
        uniqueN(
            all_enrichment$enrichment_source
        ),
        identical(
            sort(
                unique(
                    all_enrichment$enrichment_source
                )
            ),
            sort(
                expected_sources
            )
        ),
        "Expected five frozen MSigDB sources."
    )


    add_audit(
        "statistics",
        "program_source_blocks",
        nrow(
            block_counts
        ),
        nrow(
            block_counts
        ) == 10L,
        "Two developmental programs tested against five sources."
    )


    add_audit(
        "statistics",
        "all_blocks_nonempty",
        all(
            block_counts$tested_terms > 0L
        ),
        all(
            block_counts$tested_terms > 0L
        ),
        "Every program-source block contains eligible tested terms."
    )


    add_audit(
        "statistics",
        "tested_terms_match_mapping_audit",
        all(
            term_count_check$tested_terms ==
                term_count_check$eligible_terms_after_size_filter
        ),
        all(
            term_count_check$tested_terms ==
                term_count_check$eligible_terms_after_size_filter
        ),
        "ORA result rows equal eligible source-specific gene sets."
    )


    add_audit(
        "statistics",
        "source_term_counts_equal_between_programs",
        all(
            source_term_balance$
                distinct_tested_term_counts == 1L
        ),
        all(
            source_term_balance$
                distinct_tested_term_counts == 1L
        ),
        "Both programs use the same eligible terms within each source."
    )


    add_audit(
        "statistics",
        "maximum_hypergeometric_P_difference",
        format(
            maximum_pvalue_difference,
            scientific = TRUE,
            digits = 6
        ),
        maximum_pvalue_difference < 1e-14,
        "Stored P values independently recomputed from contingency counts."
    )


    add_audit(
        "statistics",
        "maximum_BH_adjusted_P_difference",
        format(
            maximum_adjusted_P_difference,
            scientific = TRUE,
            digits = 6
        ),
        maximum_adjusted_P_difference < 1e-14,
        "BH correction independently recomputed within each block."
    )


    add_audit(
        "statistics",
        "FDR_flag_consistency",
        all(
            all_enrichment$FDR_significant ==
                (
                    all_enrichment$p_adjust < 0.05
                )
        ),
        all(
            all_enrichment$FDR_significant ==
                (
                    all_enrichment$p_adjust < 0.05
                )
        ),
        "FDR significance flag matches adjusted P < 0.05."
    )


    add_audit(
        "statistics",
        "duplicate_program_source_terms",
        nrow(
            duplicate_result_rows
        ),
        nrow(
            duplicate_result_rows
        ) == 0L,
        "Each program-source-term combination occurs once."
    )


    add_audit(
        "statistics",
        "gene_set_size_range",
        paste0(
            min(
                all_enrichment$
                    background_gene_count
            ),
            "–",
            max(
                all_enrichment$
                    background_gene_count
            )
        ),
        all(
            all_enrichment$
                background_gene_count >= 10L
            &
            all_enrichment$
                background_gene_count <= 500L
        ),
        "All tested terms satisfy the prespecified 10–500 size range."
    )


    add_audit(
        "statistics",
        "valid_probability_range",
        all(
            all_enrichment$pvalue >= 0
            &
            all_enrichment$pvalue <= 1
            &
            all_enrichment$p_adjust >= 0
            &
            all_enrichment$p_adjust <= 1
        ),
        all(
            all_enrichment$pvalue >= 0
            &
            all_enrichment$pvalue <= 1
            &
            all_enrichment$p_adjust >= 0
            &
            all_enrichment$p_adjust <= 1
        ),
        "All raw and adjusted P values lie within [0,1]."
    )


    add_audit(
        "outputs",
        "required_output_files_present",
        output_files_present,
        output_files_present,
        "All expected tables and definitive figure formats were created."
    )


    add_audit(
        "outputs",
        "required_output_files_nonempty",
        output_files_nonempty,
        output_files_nonempty,
        "All expected outputs exceed 100 bytes."
    )


    add_audit(
        "outputs",
        "figure40_R1_generated",
        analysis$figure_generated,
        analysis$figure_generated,
        "PNG and PDF functional-enrichment figure generated."
    )


    audit <- rbindlist(
        audit_rows,
        use.names = TRUE,
        fill = TRUE
    )


    list(
        audit = audit,
        recomputed = recomputed,
        block_counts = block_counts,
        term_count_check = term_count_check
    )
}


finalize_phase5D4_R1 <- function() {
    analysis <- run_phase5D4_R1()

    methodology_file <- write_methodology_record()

    internal_validation <- build_internal_audit(
        analysis
    )

    audit <- internal_validation$audit

    audit_file <- file.path(
        TABLE_DIR,
        "phase5D4_R1_internal_validation_audit.tsv"
    )

    fwrite(
        audit,
        audit_file,
        sep = "\t"
    )


    block_count_file <- file.path(
        TABLE_DIR,
        "phase5D4_R1_program_source_test_counts.tsv"
    )

    fwrite(
        internal_validation$term_count_check,
        block_count_file,
        sep = "\t"
    )


    all_audits_passed <- all(
        audit$passed
    )


    total_tested_terms <- nrow(
        analysis$all_enrichment
    )


    total_FDR_significant_terms <- sum(
        analysis$all_enrichment$
            FDR_significant,
        na.rm = TRUE
    )


    completion <- data.table(
        trajectory_model_background_genes =
            length(
                analysis$trajectory_background
            ),
        confident_program_genes =
            uniqueN(
                analysis$foreground$gene_symbol
            ),
        maturation_high_genes =
            analysis$foreground[
                developmental_program ==
                    "maturation_high_increasing",
                uniqueN(
                    gene_symbol
                )
            ],
        fetal_high_genes =
            analysis$foreground[
                developmental_program ==
                    "fetal_high_decreasing",
                uniqueN(
                    gene_symbol
                )
            ],
        endpoint_direction_discordant_genes =
            analysis$foreground[
                endpoint_direction_consistent == FALSE,
                uniqueN(
                    gene_symbol
                )
            ],
        enrichment_sources =
            uniqueN(
                analysis$all_enrichment$
                    enrichment_source
            ),
        program_source_blocks =
            nrow(
                unique(
                    analysis$all_enrichment[
                        ,
                        .(
                            developmental_program,
                            enrichment_source
                        )
                    ]
                )
            ),
        total_tested_terms =
            total_tested_terms,
        total_FDR_significant_terms =
            total_FDR_significant_terms,
        enrichment_method =
            paste0(
                "source-specific one-sided hypergeometric ",
                "ORA with within-program/source BH correction"
            ),
        gene_set_resource =
            "MSigDB 2026.1.Hs via frozen msigdbr 26.1.0 resources",
        minimum_gene_set_size =
            10L,
        maximum_gene_set_size =
            500L,
        figure40_R1_generated =
            analysis$figure_generated,
        historical_outputs_used_as_inputs =
            FALSE,
        statistics_recomputed_from_phase5D3 =
            TRUE,
        all_internal_audits_passed =
            all_audits_passed,
        Phase5D4_R1_outputs_valid_internal =
            all_audits_passed,
        Phase5D4_outputs_valid =
            FALSE,
        ready_for_independent_verification =
            all_audits_passed,
        ready_for_phase8E_manuscript_integration =
            FALSE,
        Phase5D4_R1_status =
            if (
                all_audits_passed
            ) {
                "completed_pending_independent_verification"
            } else {
                "failed"
            },
        R_version =
            R.version.string,
        data_table_version =
            as.character(
                packageVersion(
                    "data.table"
                )
            ),
        ggplot2_version =
            as.character(
                packageVersion(
                    "ggplot2"
                )
            )
    )


    completion_file <- file.path(
        TABLE_DIR,
        "phase5D4_R1_completion_summary.tsv"
    )


    fwrite(
        completion,
        completion_file,
        sep = "\t"
    )


    session_file <- file.path(
        METADATA_DIR,
        "phase5D4_R1_sessionInfo.txt"
    )


    writeLines(
        capture.output(
            sessionInfo()
        ),
        session_file
    )


    canonical_outputs <- c(
        analysis$files$program_assignments,
        analysis$files$program_counts,
        analysis$files$discordant_genes,
        analysis$files$background_genes,
        analysis$files$all_results,
        analysis$files$mapping_audit,
        analysis$files$enrichment_summary,
        analysis$files$top20,
        analysis$files$figure_source,
        analysis$files$figure_png,
        analysis$files$figure_pdf,
        audit_file,
        block_count_file,
        completion_file,
        methodology_file,
        session_file
    )


    missing_outputs <- canonical_outputs[
        !file.exists(
            canonical_outputs
        )
    ]


    if (length(missing_outputs) > 0L) {
        stop(
            paste0(
                "Canonical outputs are missing: ",
                paste(
                    missing_outputs,
                    collapse = "; "
                )
            )
        )
    }


    log_message("")
    log_message(
        "===== PHASE 5D4-R1 COMPLETION ====="
    )

    print(
        completion
    )


    log_message("")
    log_message(
        "===== INTERNAL VALIDATION AUDIT ====="
    )

    print(
        audit
    )


    log_message("")
    log_message(
        "===== FUNCTIONAL ENRICHMENT SUMMARY ====="
    )

    print(
        analysis$enrichment_summary
    )


    log_message("")
    log_message(
        "===== TOP TERM PER PROGRAM AND SOURCE ====="
    )

    print(
        analysis$enrichment_summary[
            ,
            .(
                developmental_program,
                enrichment_source,
                tested_terms,
                FDR_significant_terms,
                top_term,
                top_term_overlap,
                top_term_enrichment_fold,
                top_term_adjusted_P
            )
        ]
    )


    log_message("")
    log_message(
        "Completion summary: ",
        completion_file
    )

    log_message(
        "Internal audit: ",
        audit_file
    )

    log_message(
        "All enrichment results: ",
        analysis$files$all_results
    )

    log_message(
        "Figure 40-R1 PNG: ",
        analysis$files$figure_png
    )

    log_message(
        "Figure 40-R1 PDF: ",
        analysis$files$figure_pdf
    )


    if (!all_audits_passed) {
        stop(
            "One or more Phase 5D4-R1 internal audits failed."
        )
    }


    log_message(
        "PHASE 5D4-R1 INTERNAL STATUS: PASSED"
    )

    invisible(
        completion
    )
}


if (sys.nframe() == 0L) {
    tryCatch(
        {
            finalize_phase5D4_R1()
        },
        error = function(error) {
            message(
                paste0(
                    "Phase 5D4-R1 failed: ",
                    conditionMessage(error)
                )
            )

            quit(
                save = "no",
                status = 1L,
                runLast = FALSE
            )
        }
    )
}
