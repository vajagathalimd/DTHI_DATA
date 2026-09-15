#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(data.table)
    library(ggplot2)
    library(igraph)
})

PROJECT <- "."

INPUT_FILE <- file.path(
    PROJECT,
    "07_tables/main_tables/phase5",
    "phase5D4_all_functional_enrichment_results.tsv.gz"
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

METADATA_DIR <- file.path(
    PROJECT,
    "02_metadata/phase5"
)

LOG_FILE <- file.path(
    PROJECT,
    "09_pipeline_logs/phase5",
    "phase5D5_redundancy_reduction.log"
)

for (directory in c(
    TABLE_DIR,
    FIGURE_DIR,
    SOURCE_DIR,
    METADATA_DIR,
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
    text <- paste0(...)

    cat(text, "\n")

    cat(
        text,
        "\n",
        file = LOG_FILE,
        append = TRUE
    )
}

MINIMUM_SHARED_GENES <- 3L
MINIMUM_JACCARD <- 0.25
FIGURE_THEMES_PER_SOURCE <- 5L

split_gene_ids <- function(value) {
    if (
        is.na(value) ||
        value == ""
    ) {
        return(character())
    }

    genes <- strsplit(
        value,
        "/",
        fixed = TRUE
    )[[1]]

    genes <- trimws(
        genes
    )

    sort(
        unique(
            genes[
                genes != ""
            ]
        )
    )
}

analyse_group <- function(
    group_table,
    program_name,
    source_name
) {
    group_table <- copy(
        group_table
    )

    group_table <- unique(
        group_table,
        by = "ID"
    )

    setorder(
        group_table,
        p_adjust,
        -Count,
        -enrichment_fold
    )

    gene_sets <- setNames(
        lapply(
            group_table$geneID,
            split_gene_ids
        ),
        group_table$ID
    )

    edge_rows <- list()
    edge_index <- 0L

    number_of_terms <- nrow(
        group_table
    )

    if (number_of_terms >= 2L) {
        for (
            first_index
            in seq_len(
                number_of_terms - 1L
            )
        ) {
            first_id <- group_table$ID[
                first_index
            ]

            first_genes <- gene_sets[[first_id]]

            for (
                second_index
                in seq.int(
                    first_index + 1L,
                    number_of_terms
                )
            ) {
                second_id <- group_table$ID[
                    second_index
                ]

                second_genes <- gene_sets[[second_id]]

                shared_genes <- intersect(
                    first_genes,
                    second_genes
                )

                shared_count <- length(
                    shared_genes
                )

                if (
                    shared_count <
                    MINIMUM_SHARED_GENES
                ) {
                    next
                }

                union_count <- length(
                    union(
                        first_genes,
                        second_genes
                    )
                )

                if (union_count == 0L) {
                    next
                }

                jaccard_similarity <-
                    shared_count /
                    union_count

                if (
                    jaccard_similarity <
                    MINIMUM_JACCARD
                ) {
                    next
                }

                edge_index <- edge_index + 1L

                edge_rows[[edge_index]] <- data.table(
                    from = first_id,
                    to = second_id,
                    shared_genes =
                        shared_count,
                    jaccard_similarity =
                        jaccard_similarity
                )
            }
        }
    }

    if (length(edge_rows) > 0L) {
        edge_table <- rbindlist(
            edge_rows
        )
    } else {
        edge_table <- data.table(
            from = character(),
            to = character(),
            shared_genes = integer(),
            jaccard_similarity = numeric()
        )
    }

    vertex_table <- data.table(
        name = group_table$ID
    )

    graph_object <- graph_from_data_frame(
        d = edge_table,
        directed = FALSE,
        vertices = vertex_table
    )

    if (ecount(graph_object) > 0L) {
        communities <- cluster_louvain(
            graph_object,
            weights =
                E(graph_object)$jaccard_similarity
        )

        raw_membership <- membership(
            communities
        )

        membership_table <- data.table(
            ID = names(
                raw_membership
            ),
            raw_theme =
                as.integer(
                    raw_membership
                )
        )
    } else {
        membership_table <- data.table(
            ID = group_table$ID,
            raw_theme =
                seq_len(
                    nrow(group_table)
                )
        )
    }

    membership_table <- merge(
        membership_table,
        group_table,
        by = "ID",
        all.x = TRUE
    )

    representative_order <- membership_table[
        order(
            raw_theme,
            p_adjust,
            -Count,
            -enrichment_fold
        ),
        .SD[1],
        by = raw_theme
    ]

    setorder(
        representative_order,
        p_adjust,
        -Count,
        -enrichment_fold
    )

    representative_order[
        ,
        theme_number :=
            seq_len(.N)
    ]

    theme_mapping <- representative_order[
        ,
        .(
            raw_theme,
            theme_number
        )
    ]

    membership_table <- merge(
        membership_table,
        theme_mapping,
        by = "raw_theme",
        all.x = TRUE
    )

    membership_table[
        ,
        functional_theme_id :=
            sprintf(
                "%s__%s__T%03d",
                program_name,
                source_name,
                theme_number
            )
    ]

    membership_table[
        ,
        is_representative :=
            frank(
                p_adjust,
                ties.method = "first"
            ) == 1L,
        by = functional_theme_id
    ]

    theme_summaries <- membership_table[
        ,
        {
            member_ids <- ID

            member_gene_lists <- gene_sets[
                member_ids
            ]

            union_genes <- sort(
                unique(
                    unlist(
                        member_gene_lists,
                        use.names = FALSE
                    )
                )
            )

            gene_frequency <- table(
                unlist(
                    member_gene_lists,
                    use.names = FALSE
                )
            )

            core_threshold <- ceiling(
                .N / 2
            )

            core_genes <- sort(
                names(
                    gene_frequency[
                        gene_frequency >=
                            core_threshold
                    ]
                )
            )

            representative_row <- .SD[
                order(
                    p_adjust,
                    -Count,
                    -enrichment_fold
                )
            ][1]

            .(
                representative_ID =
                    representative_row$ID,

                representative_description =
                    representative_row$Description,

                representative_FDR =
                    representative_row$p_adjust,

                representative_overlap =
                    representative_row$Count,

                representative_enrichment_fold =
                    representative_row$enrichment_fold,

                member_terms =
                    .N,

                union_gene_count =
                    length(
                        union_genes
                    ),

                union_genes =
                    paste(
                        union_genes,
                        collapse = "/"
                    ),

                core_gene_count =
                    length(
                        core_genes
                    ),

                core_genes =
                    paste(
                        core_genes,
                        collapse = "/"
                    )
            )
        },
        by = functional_theme_id
    ]

    theme_summaries[
        ,
        `:=`(
            developmental_program =
                program_name,

            enrichment_source =
                source_name
        )
    ]

    membership_table[
        ,
        `:=`(
            developmental_program =
                program_name,

            enrichment_source =
                source_name
        )
    ]

    edge_table[
        ,
        `:=`(
            developmental_program =
                program_name,

            enrichment_source =
                source_name
        )
    ]

    list(
        membership = membership_table,
        themes = theme_summaries,
        edges = edge_table
    )
}

if (!file.exists(INPUT_FILE)) {
    stop(
        "Missing Phase 5D4 enrichment file: ",
        INPUT_FILE
    )
}

log_message(
    "===== Phase 5D5 started ====="
)

log_message(
    "Minimum shared genes: ",
    MINIMUM_SHARED_GENES
)

log_message(
    "Minimum Jaccard similarity: ",
    MINIMUM_JACCARD
)

enrichment <- fread(
    cmd = paste(
        "gzip -dc",
        shQuote(INPUT_FILE)
    ),
    sep = "\t",
    header = TRUE
)

required_columns <- c(
    "developmental_program",
    "enrichment_source",
    "ID",
    "Description",
    "Count",
    "enrichment_fold",
    "p_adjust",
    "FDR_significant",
    "geneID"
)

missing_columns <- setdiff(
    required_columns,
    names(enrichment)
)

if (length(missing_columns) > 0L) {
    stop(
        "Missing enrichment columns: ",
        paste(
            missing_columns,
            collapse = ", "
        )
    )
}

significant_terms <- enrichment[
    FDR_significant == TRUE
    &
    is.finite(p_adjust)
    &
    p_adjust < 0.05
    &
    Count > 0
]

if (nrow(significant_terms) == 0L) {
    stop(
        "No FDR-significant Phase 5D4 terms were found."
    )
}

group_combinations <- unique(
    significant_terms[
        ,
        .(
            developmental_program,
            enrichment_source
        )
    ]
)

membership_results <- list()
theme_results <- list()
edge_results <- list()

for (
    group_index
    in seq_len(
        nrow(group_combinations)
    )
) {
    program_name <-
        group_combinations$developmental_program[
            group_index
        ]

    source_name <-
        group_combinations$enrichment_source[
            group_index
        ]

    current_group <- significant_terms[
        developmental_program ==
            program_name
        &
        enrichment_source ==
            source_name
    ]

    log_message(
        program_name,
        " | ",
        source_name,
        " | significant terms: ",
        nrow(current_group)
    )

    result <- analyse_group(
        group_table =
            current_group,

        program_name =
            program_name,

        source_name =
            source_name
    )

    membership_results[[group_index]] <- result$membership

    theme_results[[group_index]] <- result$themes

    edge_results[[group_index]] <- result$edges
}

term_membership <- rbindlist(
    membership_results,
    fill = TRUE
)

representative_themes <- rbindlist(
    theme_results,
    fill = TRUE
)

similarity_edges <- rbindlist(
    edge_results,
    fill = TRUE
)

setorder(
    term_membership,
    developmental_program,
    enrichment_source,
    functional_theme_id,
    p_adjust
)

setorder(
    representative_themes,
    developmental_program,
    enrichment_source,
    representative_FDR
)

fwrite(
    term_membership,
    file.path(
        TABLE_DIR,
        "phase5D5_functional_theme_membership.tsv.gz"
    ),
    sep = "\t",
    compress = "gzip"
)

fwrite(
    representative_themes,
    file.path(
        TABLE_DIR,
        "phase5D5_representative_functional_themes.tsv"
    ),
    sep = "\t"
)

fwrite(
    similarity_edges,
    file.path(
        TABLE_DIR,
        "phase5D5_term_similarity_edges.tsv.gz"
    ),
    sep = "\t",
    compress = "gzip"
)

reduction_summary <- term_membership[
    ,
    .(
        original_significant_terms =
            uniqueN(ID),

        redundancy_reduced_themes =
            uniqueN(
                functional_theme_id
            ),

        multi_term_themes =
            uniqueN(
                functional_theme_id[
                    functional_theme_id %chin%
                        functional_theme_id[
                            duplicated(
                                functional_theme_id
                            )
                        ]
                ]
            ),

        largest_theme_terms =
            max(
                tabulate(
                    match(
                        functional_theme_id,
                        unique(
                            functional_theme_id
                        )
                    )
                )
            )
    ),
    by = .(
        developmental_program,
        enrichment_source
    )
]

reduction_summary[
    ,
    terms_removed_as_redundant :=
        original_significant_terms -
        redundancy_reduced_themes
]

reduction_summary[
    ,
    reduction_fraction :=
        terms_removed_as_redundant /
        original_significant_terms
]

setorder(
    reduction_summary,
    developmental_program,
    enrichment_source
)

fwrite(
    reduction_summary,
    file.path(
        TABLE_DIR,
        "phase5D5_redundancy_reduction_summary.tsv"
    ),
    sep = "\t"
)

plot_data <- copy(
    representative_themes
)

plot_data[
    ,
    rank_within_source :=
        frank(
            representative_FDR,
            ties.method = "first"
        ),
    by = .(
        developmental_program,
        enrichment_source
    )
]

plot_data <- plot_data[
    rank_within_source <=
        FIGURE_THEMES_PER_SOURCE
]

plot_data[
    ,
    program_label := fifelse(
        developmental_program ==
            "maturation_high_increasing",
        "Maturation-high / increasing",
        "Fetal-high / decreasing"
    )
]

plot_data[
    ,
    negative_log10_FDR :=
        -log10(
            pmax(
                representative_FDR,
                .Machine$double.xmin
            )
        )
]

plot_data[
    ,
    shortened_description := fifelse(
        nchar(
            representative_description
        ) > 72,
        paste0(
            substr(
                representative_description,
                1,
                69
            ),
            "..."
        ),
        representative_description
    )
]

plot_data[
    ,
    theme_key := paste(
        shortened_description,
        developmental_program,
        enrichment_source,
        functional_theme_id,
        sep = "___"
    )
]

theme_order <- plot_data[
    order(
        program_label,
        enrichment_source,
        negative_log10_FDR
    ),
    theme_key
]

plot_data[
    ,
    theme_key := factor(
        theme_key,
        levels = unique(
            theme_order
        )
    )
]

plot_data[
    ,
    enrichment_source := factor(
        enrichment_source,
        levels = c(
            "GO_BP",
            "GO_CC",
            "GO_MF",
            "Reactome",
            "Hallmark"
        )
    )
]

figure_41 <- ggplot(
    plot_data,
    aes(
        x = negative_log10_FDR,
        y = theme_key,
        size = member_terms
    )
) +
    geom_point(
        alpha = 0.82
    ) +
    facet_grid(
        program_label ~
            enrichment_source,
        scales = "free_y",
        space = "free_y"
    ) +
    scale_y_discrete(
        labels = function(values) {
            sub(
                "___.*$",
                "",
                values
            )
        }
    ) +
    labs(
        title = paste0(
            "Figure 41. Redundancy-reduced functional themes ",
            "of cortical development"
        ),

        subtitle = paste0(
            "Terms grouped by shared foreground genes; ",
            "Jaccard >= ",
            MINIMUM_JACCARD,
            " and shared genes >= ",
            MINIMUM_SHARED_GENES
        ),

        x = "-log10 adjusted P value",
        y = NULL,
        size = "Terms in theme"
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
            size = 8.5,
            face = "bold"
        ),

        axis.text.y = element_text(
            size = 6.8
        ),

        panel.spacing = grid::unit(
            7,
            "points"
        )
    )

ggsave(
    filename = file.path(
        FIGURE_DIR,
        "Figure41_redundancy_reduced_functional_themes.pdf"
    ),
    plot = figure_41,
    width = 20,
    height = 12,
    units = "in"
)

ggsave(
    filename = file.path(
        FIGURE_DIR,
        "Figure41_redundancy_reduced_functional_themes.png"
    ),
    plot = figure_41,
    width = 20,
    height = 12,
    units = "in",
    dpi = 300
)

fwrite(
    plot_data,
    file.path(
        SOURCE_DIR,
        "Figure41_source_redundancy_reduced_themes.tsv"
    ),
    sep = "\t"
)

completion <- data.table(
    input_FDR_significant_terms =
        nrow(
            significant_terms
        ),

    redundancy_reduced_themes =
        nrow(
            representative_themes
        ),

    terms_removed_as_redundant =
        nrow(
            significant_terms
        ) -
        nrow(
            representative_themes
        ),

    overall_reduction_fraction =
        (
            nrow(
                significant_terms
            ) -
            nrow(
                representative_themes
            )
        ) /
        nrow(
            significant_terms
        ),

    similarity_edges =
        nrow(
            similarity_edges
        ),

    minimum_shared_genes =
        MINIMUM_SHARED_GENES,

    minimum_jaccard =
        MINIMUM_JACCARD,

    figure41_generated =
        file.exists(
            file.path(
                FIGURE_DIR,
                "Figure41_redundancy_reduced_functional_themes.pdf"
            )
        ),

    Phase5D5_status =
        "completed"
)

fwrite(
    completion,
    file.path(
        TABLE_DIR,
        "phase5D5_completion_summary.tsv"
    ),
    sep = "\t"
)

report_file <- file.path(
    METADATA_DIR,
    "PHASE5D5_REDUNDANCY_REDUCTION_REPORT.txt"
)

sink(report_file)

cat(
    "Phase 5D5 redundancy reduction of functional enrichment terms\n\n"
)

cat(
    "Similarity basis: shared foreground genes.\n"
)

cat(
    "Minimum shared genes: ",
    MINIMUM_SHARED_GENES,
    "\n",
    sep = ""
)

cat(
    "Minimum Jaccard similarity: ",
    MINIMUM_JACCARD,
    "\n\n",
    sep = ""
)

cat("Completion summary\n")

print(completion)

cat("\nProgram/source reduction summary\n")

print(reduction_summary)

sink()

log_message("")
log_message(
    "===== Phase 5D5 completed ====="
)

log_message(
    paste(
        capture.output(
            print(completion)
        ),
        collapse = "\n"
    )
)
