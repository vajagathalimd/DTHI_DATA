#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(data.table)
    library(ggplot2)
})

PROJECT <- "."

ASSIGNMENT_FILE <- file.path(
    PROJECT,
    "07_tables/main_tables/phase5",
    "phase5D3_all_supported_gene_centroid_assignments.tsv.gz"
)

FILTER_FILE <- file.path(
    PROJECT,
    "07_tables/main_tables/phase5",
    "phase5D2_gene_filtering_summary.tsv.gz"
)

SEED_FILE <- file.path(
    PROJECT,
    "03_processed_data/transcriptomics/BrainSpan/DTHI_seed_modules",
    "dthi_seed_gene_matching.tsv"
)

CELLTYPE_FILE <- file.path(
    PROJECT,
    "03_processed_data/single_cell_marker_validation/phase2A",
    "brainspan_celltype_marker_matching.tsv"
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
    "phase5D6B_program_module_celltype_integration.log"
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

scalar_character <- function(values) {
    if (is.list(values)) {
        return(
            vapply(
                values,
                function(value) {
                    value <- as.character(value)

                    value <- value[
                        !is.na(value)
                        &
                        value != ""
                    ]

                    if (length(value) == 0L) {
                        return(NA_character_)
                    }

                    paste(
                        value,
                        collapse = ";"
                    )
                },
                character(1)
            )
        )
    }

    as.character(values)
}

normalize_gene <- function(values) {
    values <- toupper(
        trimws(
            scalar_character(values)
        )
    )

    values[
        is.na(values) |
        values == "" |
        values == "NA" |
        values == "NAN"
    ] <- NA_character_

    values
}

read_gzip_table <- function(path) {
    fread(
        cmd = paste(
            "gzip -dc",
            shQuote(path)
        ),
        sep = "\t",
        header = TRUE
    )
}

safe_fisher <- function(
    contingency_table,
    alternative = "greater"
) {
    result <- fisher.test(
        contingency_table,
        alternative = alternative
    )

    list(
        p_value = as.numeric(
            result$p.value
        ),

        fisher_odds_ratio = if (
            length(result$estimate) == 0
        ) {
            NA_real_
        } else {
            as.numeric(
                result$estimate
            )
        }
    )
}

make_display_name <- function(values) {
    mapping <- c(
        activity_dependent_plasticity =
            "Activity-dependent plasticity",

        astrocyte_maturation_metabolic_support =
            "Astrocyte maturation/support",

        axon_guidance_neurite_outgrowth =
            "Axon guidance/neurite growth",

        neurogenesis_migration_layering =
            "Neurogenesis/migration/layering",

        oligodendrocyte_myelination =
            "Oligodendrocyte/myelination",

        patterning_arealization =
            "Patterning/arealization",

        progenitor_radial_glia =
            "Progenitor/radial glia",

        synaptic_assembly_receptor_trafficking =
            "Synaptic assembly/receptor trafficking",

        synaptic_membrane_structural_candidates =
            "Synaptic membrane structure",

        astrocyte =
            "Astrocyte",

        deep_layer_neuron =
            "Deep-layer neuron",

        endothelial =
            "Endothelial",

        excitatory_projection_neuron =
            "Excitatory projection neuron",

        inhibitory_interneuron =
            "Inhibitory interneuron",

        intermediate_progenitor =
            "Intermediate progenitor",

        microglia =
            "Microglia",

        oligodendrocyte =
            "Oligodendrocyte",

        opc =
            "OPC",

        pericyte_vascular_smooth_muscle =
            "Pericyte/vascular smooth muscle",

        radial_glia_progenitor =
            "Radial glia/progenitor",

        upper_layer_neuron =
            "Upper-layer neuron"
    )

    result <- unname(
        mapping[
            values
        ]
    )

    missing <- is.na(result)

    result[missing] <- tools::toTitleCase(
        gsub(
            "_",
            " ",
            values[missing],
            fixed = TRUE
        )
    )

    result
}

required_files <- c(
    ASSIGNMENT_FILE,
    FILTER_FILE,
    SEED_FILE,
    CELLTYPE_FILE
)

missing_files <- required_files[
    !file.exists(required_files)
]

if (length(missing_files) > 0) {
    stop(
        "Missing required inputs: ",
        paste(
            missing_files,
            collapse = "; "
        )
    )
}

log_message(
    "===== Phase 5D6B started ====="
)

# ============================================================
# Load developmental program assignments and background
# ============================================================

assignments <- read_gzip_table(
    ASSIGNMENT_FILE
)

filter_table <- read_gzip_table(
    FILTER_FILE
)

assignments[
    ,
    gene_symbol := normalize_gene(
        gene_symbol
    )
]

assignments[
    ,
    confident_cluster :=
        as.integer(
            confident_cluster
        )
]

filter_table[
    ,
    gene_symbol := normalize_gene(
        gene_symbol
    )
]

trajectory_background <- unique(
    filter_table[
        trajectory_model_pass == TRUE &
        !is.na(gene_symbol),
        gene_symbol
    ]
)

program_assignments <- assignments[
    assignment_confidence %chin%
        c(
            "high",
            "moderate"
        )
    &
    !is.na(confident_cluster)
    &
    !is.na(gene_symbol)
]

program_assignments[
    ,
    developmental_program := fifelse(
        confident_cluster == 1L,
        "maturation_high_increasing",
        fifelse(
            confident_cluster == 2L,
            "fetal_high_decreasing",
            NA_character_
        )
    )
]

program_assignments <- program_assignments[
    !is.na(developmental_program)
    &
    gene_symbol %chin%
        trajectory_background,
    .(
        gene_symbol,
        developmental_program
    )
]

program_assignments[
    ,
    gene_symbol :=
        normalize_gene(
            gene_symbol
        )
]

program_assignments[
    ,
    developmental_program :=
        scalar_character(
            developmental_program
        )
]

program_assignments <- unique(
    program_assignments,
    by = c(
        "gene_symbol",
        "developmental_program"
    )
)

program_names <- c(
    "maturation_high_increasing",
    "fetal_high_decreasing"
)

program_sizes <- program_assignments[
    ,
    .(
        program_genes =
            uniqueN(
                gene_symbol
            )
    ),
    by = developmental_program
]

if (
    program_sizes[
        developmental_program ==
            "maturation_high_increasing",
        program_genes
    ] != 3333L
) {
    stop(
        "Unexpected maturation-high gene count."
    )
}

if (
    program_sizes[
        developmental_program ==
            "fetal_high_decreasing",
        program_genes
    ] != 5412L
) {
    stop(
        "Unexpected fetal-high gene count."
    )
}

# ============================================================
# Load canonical DTHI and cell-type definitions
# ============================================================

seed_table <- fread(
    SEED_FILE,
    sep = "\t",
    header = TRUE
)

celltype_table <- fread(
    CELLTYPE_FILE,
    sep = "\t",
    header = TRUE
)

required_seed_columns <- c(
    "module",
    "seed_gene"
)

required_celltype_columns <- c(
    "celltype_module",
    "marker_gene"
)

if (
    !all(
        required_seed_columns %in%
            names(seed_table)
    )
) {
    stop(
        "Required DTHI seed columns are missing."
    )
}

if (
    !all(
        required_celltype_columns %in%
            names(celltype_table)
    )
) {
    stop(
        "Required cell-type marker columns are missing."
    )
}

seed_gene_sets <- data.table(
    gene_set_type = rep(
        "DTHI_seed_module",
        nrow(seed_table)
    ),

    gene_set = trimws(
        scalar_character(
            seed_table[["module"]]
        )
    ),

    gene_symbol = normalize_gene(
        seed_table[["seed_gene"]]
    )
)

seed_gene_sets <- unique(
    seed_gene_sets,
    by = c(
        "gene_set_type",
        "gene_set",
        "gene_symbol"
    )
)

celltype_gene_sets <- data.table(
    gene_set_type = rep(
        "celltype_marker_set",
        nrow(celltype_table)
    ),

    gene_set = trimws(
        scalar_character(
            celltype_table[["celltype_module"]]
        )
    ),

    gene_symbol = normalize_gene(
        celltype_table[["marker_gene"]]
    )
)

celltype_gene_sets <- unique(
    celltype_gene_sets,
    by = c(
        "gene_set_type",
        "gene_set",
        "gene_symbol"
    )
)

log_message(
    "Program-assignment column classes: ",
    paste(
        names(program_assignments),
        vapply(
            program_assignments,
            function(column) {
                paste(
                    class(column),
                    collapse = "/"
                )
            },
            character(1)
        ),
        sep = "=",
        collapse = "; "
    )
)

log_message(
    "Seed gene-set column classes: ",
    paste(
        names(seed_gene_sets),
        vapply(
            seed_gene_sets,
            function(column) {
                paste(
                    class(column),
                    collapse = "/"
                )
            },
            character(1)
        ),
        sep = "=",
        collapse = "; "
    )
)

log_message(
    "Cell-type gene-set column classes: ",
    paste(
        names(celltype_gene_sets),
        vapply(
            celltype_gene_sets,
            function(column) {
                paste(
                    class(column),
                    collapse = "/"
                )
            },
            character(1)
        ),
        sep = "=",
        collapse = "; "
    )
)

gene_sets <- rbindlist(
    list(
        seed_gene_sets,
        celltype_gene_sets
    ),
    use.names = TRUE,
    fill = FALSE
)

gene_sets <- gene_sets[
    !is.na(gene_set)
    &
    !is.na(gene_symbol)
]

display_name_values <- make_display_name(
    as.character(
        gene_sets$gene_set
    )
)

if (is.list(display_name_values)) {
    display_name_values <- unlist(
        display_name_values,
        recursive = TRUE,
        use.names = FALSE
    )
}

display_name_values <- as.character(
    display_name_values
)

if (
    length(display_name_values) !=
    nrow(gene_sets)
) {
    stop(
        "Display-name generation returned ",
        length(display_name_values),
        " values for ",
        nrow(gene_sets),
        " gene-set rows."
    )
}

gene_sets[
    ,
    display_name :=
        display_name_values
]

list_columns_after_labeling <- names(
    gene_sets
)[
    vapply(
        gene_sets,
        is.list,
        logical(1)
    )
]

if (
    length(list_columns_after_labeling) > 0L
) {
    stop(
        "List columns remain in gene_sets after label generation: ",
        paste(
            list_columns_after_labeling,
            collapse = ", "
        )
    )
}

log_message(
    "Combined gene-set column classes: ",
    paste(
        names(gene_sets),
        vapply(
            gene_sets,
            function(column) {
                paste(
                    class(column),
                    collapse = "/"
                )
            },
            character(1)
        ),
        sep = "=",
        collapse = "; "
    )
)

# ============================================================
# Gene-set background mapping audit
# ============================================================

confident_union <- unique(
    program_assignments$gene_symbol
)

mapping_audit <- gene_sets[
    ,
    {
        original_genes <- unique(
            gene_symbol
        )

        mapped_background <- intersect(
            original_genes,
            trajectory_background
        )

        mapped_confident <- intersect(
            original_genes,
            confident_union
        )

        .(
            display_name =
                display_name[1],

            original_gene_count =
                length(
                    original_genes
                ),

            mapped_background_gene_count =
                length(
                    mapped_background
                ),

            background_mapping_fraction =
                length(
                    mapped_background
                ) /
                length(
                    original_genes
                ),

            mapped_confident_gene_count =
                length(
                    mapped_confident
                ),

            mapped_background_genes =
                paste(
                    sort(
                        mapped_background
                    ),
                    collapse = "/"
                )
        )
    },
    by = .(
        gene_set_type,
        gene_set
    )
]

setorder(
    mapping_audit,
    gene_set_type,
    gene_set
)

fwrite(
    mapping_audit,
    file.path(
        TABLE_DIR,
        "phase5D6B_gene_set_background_mapping_audit.tsv"
    ),
    sep = "\t"
)

# ============================================================
# Program-specific enrichment against trajectory background
# ============================================================

background_size <- length(
    trajectory_background
)

enrichment_results <- list()

for (
    gene_set_index
    in seq_len(
        nrow(
            mapping_audit
        )
    )
) {
    gene_set_type_value <-
        mapping_audit$gene_set_type[
            gene_set_index
        ]

    gene_set_value <-
        mapping_audit$gene_set[
            gene_set_index
        ]

    display_name_value <-
        mapping_audit$display_name[
            gene_set_index
        ]

    original_genes <- unique(
        gene_sets[
            gene_set_type ==
                gene_set_type_value
            &
            gene_set ==
                gene_set_value,
            gene_symbol
        ]
    )

    mapped_genes <- intersect(
        original_genes,
        trajectory_background
    )

    mapped_set_size <- length(
        mapped_genes
    )

    for (program_name in program_names) {
        program_genes <- unique(
            program_assignments[
                developmental_program ==
                    program_name,
                gene_symbol
            ]
        )

        program_size <- length(
            program_genes
        )

        overlap_genes <- sort(
            intersect(
                mapped_genes,
                program_genes
            )
        )

        overlap_count <- length(
            overlap_genes
        )

        contingency <- matrix(
            c(
                overlap_count,
                program_size -
                    overlap_count,

                mapped_set_size -
                    overlap_count,

                background_size -
                    program_size -
                    mapped_set_size +
                    overlap_count
            ),
            nrow = 2,
            byrow = TRUE
        )

        if (any(contingency < 0)) {
            stop(
                "Negative contingency-table cell for ",
                gene_set_value,
                " and ",
                program_name
            )
        }

        fisher_result <- safe_fisher(
            contingency,
            alternative = "greater"
        )

        expected_overlap <-
            program_size *
            mapped_set_size /
            background_size

        enrichment_fold <- if (
            mapped_set_size == 0
        ) {
            NA_real_
        } else {
            (
                overlap_count /
                program_size
            ) /
            (
                mapped_set_size /
                background_size
            )
        }

        haldane_odds_ratio <- (
            (
                contingency[1, 1] +
                0.5
            ) *
            (
                contingency[2, 2] +
                0.5
            )
        ) /
        (
            (
                contingency[1, 2] +
                0.5
            ) *
            (
                contingency[2, 1] +
                0.5
            )
        )

        enrichment_results[[length(
                    enrichment_results
                ) + 1L]] <- data.table(
            developmental_program =
                program_name,

            gene_set_type =
                gene_set_type_value,

            gene_set =
                gene_set_value,

            display_name =
                display_name_value,

            trajectory_background_genes =
                background_size,

            program_gene_count =
                program_size,

            original_gene_set_size =
                length(
                    original_genes
                ),

            mapped_gene_set_size =
                mapped_set_size,

            overlap_gene_count =
                overlap_count,

            expected_overlap =
                expected_overlap,

            enrichment_fold =
                enrichment_fold,

            fisher_odds_ratio =
                fisher_result$fisher_odds_ratio,

            haldane_odds_ratio =
                haldane_odds_ratio,

            p_value =
                fisher_result$p_value,

            overlap_genes =
                paste(
                    overlap_genes,
                    collapse = "/"
                )
        )
    }
}

program_enrichment <- rbindlist(
    enrichment_results
)

program_enrichment[
    ,
    fdr_bh := p.adjust(
        p_value,
        method = "BH"
    ),
    by = .(
        developmental_program,
        gene_set_type
    )
]

program_enrichment[
    ,
    FDR_significant :=
        fdr_bh < 0.05
]

setorder(
    program_enrichment,
    gene_set_type,
    gene_set,
    developmental_program
)

fwrite(
    program_enrichment,
    file.path(
        TABLE_DIR,
        "phase5D6B_gene_set_enrichment_by_program.tsv"
    ),
    sep = "\t"
)

# ============================================================
# Direct maturation-high versus fetal-high preference
# ============================================================

maturation_genes <- unique(
    program_assignments[
        developmental_program ==
            "maturation_high_increasing",
        gene_symbol
    ]
)

fetal_genes <- unique(
    program_assignments[
        developmental_program ==
            "fetal_high_decreasing",
        gene_symbol
    ]
)

maturation_size <- length(
    maturation_genes
)

fetal_size <- length(
    fetal_genes
)

preference_results <- list()

for (
    gene_set_index
    in seq_len(
        nrow(
            mapping_audit
        )
    )
) {
    gene_set_type_value <-
        mapping_audit$gene_set_type[
            gene_set_index
        ]

    gene_set_value <-
        mapping_audit$gene_set[
            gene_set_index
        ]

    display_name_value <-
        mapping_audit$display_name[
            gene_set_index
        ]

    mapped_genes <- intersect(
        unique(
            gene_sets[
                gene_set_type ==
                    gene_set_type_value
                &
                gene_set ==
                    gene_set_value,
                gene_symbol
            ]
        ),
        trajectory_background
    )

    maturation_overlap <- intersect(
        mapped_genes,
        maturation_genes
    )

    fetal_overlap <- intersect(
        mapped_genes,
        fetal_genes
    )

    maturation_count <- length(
        maturation_overlap
    )

    fetal_count <- length(
        fetal_overlap
    )

    contingency <- matrix(
        c(
            maturation_count,
            maturation_size -
                maturation_count,

            fetal_count,
            fetal_size -
                fetal_count
        ),
        nrow = 2,
        byrow = TRUE
    )

    fisher_result <- safe_fisher(
        contingency,
        alternative = "two.sided"
    )

    haldane_odds_ratio <- (
        (
            maturation_count +
            0.5
        ) *
        (
            fetal_size -
            fetal_count +
            0.5
        )
    ) /
    (
        (
            maturation_size -
            maturation_count +
            0.5
        ) *
        (
            fetal_count +
            0.5
        )
    )

    preference_results[[length(
                preference_results
            ) + 1L]] <- data.table(
        gene_set_type =
            gene_set_type_value,

        gene_set =
            gene_set_value,

        display_name =
            display_name_value,

        maturation_program_genes =
            maturation_size,

        fetal_program_genes =
            fetal_size,

        maturation_overlap =
            maturation_count,

        fetal_overlap =
            fetal_count,

        maturation_overlap_fraction =
            maturation_count /
            maturation_size,

        fetal_overlap_fraction =
            fetal_count /
            fetal_size,

        fisher_odds_ratio =
            fisher_result$fisher_odds_ratio,

        haldane_odds_ratio =
            haldane_odds_ratio,

        log2_preference_odds_ratio =
            log2(
                haldane_odds_ratio
            ),

        p_value =
            fisher_result$p_value,

        maturation_overlap_genes =
            paste(
                sort(
                    maturation_overlap
                ),
                collapse = "/"
            ),

        fetal_overlap_genes =
            paste(
                sort(
                    fetal_overlap
                ),
                collapse = "/"
            )
    )
}

program_preference <- rbindlist(
    preference_results
)

program_preference[
    ,
    fdr_bh := p.adjust(
        p_value,
        method = "BH"
    ),
    by = gene_set_type
]

program_preference[
    ,
    preferred_program := fifelse(
        log2_preference_odds_ratio > 0,
        "maturation_high_increasing",
        fifelse(
            log2_preference_odds_ratio < 0,
            "fetal_high_decreasing",
            "no_direction"
        )
    )
]

program_preference[
    ,
    preference_FDR_significant :=
        fdr_bh < 0.05
]

setorder(
    program_preference,
    gene_set_type,
    fdr_bh
)

fwrite(
    program_preference,
    file.path(
        TABLE_DIR,
        "phase5D6B_direct_program_preference.tsv"
    ),
    sep = "\t"
)

# ============================================================
# Figure 42
# ============================================================

module_order <- c(
    "patterning_arealization",
    "progenitor_radial_glia",
    "neurogenesis_migration_layering",
    "axon_guidance_neurite_outgrowth",
    "synaptic_assembly_receptor_trafficking",
    "synaptic_membrane_structural_candidates",
    "activity_dependent_plasticity",
    "astrocyte_maturation_metabolic_support",
    "oligodendrocyte_myelination"
)

celltype_order <- c(
    "radial_glia_progenitor",
    "intermediate_progenitor",
    "excitatory_projection_neuron",
    "inhibitory_interneuron",
    "deep_layer_neuron",
    "upper_layer_neuron",
    "astrocyte",
    "opc",
    "oligodendrocyte",
    "microglia",
    "endothelial",
    "pericyte_vascular_smooth_muscle"
)

ordered_gene_sets <- c(
    module_order,
    celltype_order
)

display_lookup <- gene_sets[
    ,
    .(
        display_name =
            as.character(
                display_name[1L]
            )
    ),
    by = gene_set
]

display_lookup[
    ,
    gene_set :=
        as.character(
            gene_set
        )
]

display_lookup[
    ,
    display_name :=
        as.character(
            display_name
        )
]

ordered_display_names <- display_lookup[
    match(
        ordered_gene_sets,
        gene_set
    ),
    display_name
]

plot_data <- copy(
    program_enrichment
)

plot_data[
    ,
    program_label := fifelse(
        developmental_program ==
            "maturation_high_increasing",
        "Maturation-high",
        "Fetal-high"
    )
]

plot_data[
    ,
    gene_set_type_label := fifelse(
        gene_set_type ==
            "DTHI_seed_module",
        "DTHI seed modules",
        "Cell-type marker sets"
    )
]

plot_data[
    ,
    plot_key := paste(
        display_name,
        gene_set_type_label,
        sep = "___"
    )
]

ordered_plot_keys <- c(
    paste(
        ordered_display_names[
            seq_along(
                module_order
            )
        ],
        "DTHI seed modules",
        sep = "___"
    ),

    paste(
        ordered_display_names[
            length(
                module_order
            ) +
            seq_along(
                celltype_order
            )
        ],
        "Cell-type marker sets",
        sep = "___"
    )
)

plot_data[
    ,
    plot_key := factor(
        plot_key,
        levels = rev(
            ordered_plot_keys
        )
    )
]

plot_data[
    ,
    program_label := factor(
        program_label,
        levels = c(
            "Fetal-high",
            "Maturation-high"
        )
    )
]

plot_data[
    ,
    negative_log10_FDR :=
        -log10(
            pmax(
                fdr_bh,
                .Machine$double.xmin
            )
        )
]

plot_data[
    ,
    fold_label := sprintf(
        "%.2fx",
        enrichment_fold
    )
]

figure_42 <- ggplot(
    plot_data,
    aes(
        x = program_label,
        y = plot_key
    )
) +
    geom_point(
        aes(
            size =
                overlap_gene_count,
            fill =
                negative_log10_FDR
        ),
        shape = 21,
        stroke = 0.35
    ) +
    geom_text(
        aes(
            label =
                fold_label
        ),
        size = 2.25
    ) +
    facet_grid(
        gene_set_type_label ~ .,
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
    scale_size_continuous(
        range = c(
            4,
            12
        )
    ) +
    labs(
        title = paste0(
            "Figure 42. Developmental gene programs align with ",
            "DTHI modules and cortical cell classes"
        ),

        subtitle = paste0(
            "Point size = overlapping genes; ",
            "label = enrichment fold; ",
            "fill = -log10 FDR"
        ),

        x = NULL,
        y = NULL,
        size = "Overlap genes",
        fill = "-log10 FDR"
    ) +
    theme_bw(
        base_size = 10
    ) +
    theme(
        plot.title = element_text(
            size = 14,
            face = "bold"
        ),

        strip.text.y = element_text(
            size = 10,
            face = "bold"
        ),

        axis.text.y = element_text(
            size = 8
        ),

        axis.text.x = element_text(
            size = 10,
            face = "bold"
        ),

        panel.spacing = grid::unit(
            8,
            "points"
        )
    )

ggsave(
    filename = file.path(
        FIGURE_DIR,
        "Figure42_program_module_celltype_integration.pdf"
    ),
    plot = figure_42,
    width = 11,
    height = 13,
    units = "in"
)

ggsave(
    filename = file.path(
        FIGURE_DIR,
        "Figure42_program_module_celltype_integration.png"
    ),
    plot = figure_42,
    width = 11,
    height = 13,
    units = "in",
    dpi = 300
)

fwrite(
    plot_data,
    file.path(
        SOURCE_DIR,
        "Figure42_source_program_module_celltype_integration.tsv"
    ),
    sep = "\t"
)

# ============================================================
# Completion summary
# ============================================================

completion <- data.table(
    trajectory_background_genes =
        background_size,

    maturation_high_genes =
        maturation_size,

    fetal_high_genes =
        fetal_size,

    DTHI_seed_modules =
        uniqueN(
            seed_gene_sets$gene_set
        ),

    celltype_marker_sets =
        uniqueN(
            celltype_gene_sets$gene_set
        ),

    program_specific_tests =
        nrow(
            program_enrichment
        ),

    program_specific_FDR_significant =
        sum(
            program_enrichment$FDR_significant,
            na.rm = TRUE
        ),

    direct_preference_tests =
        nrow(
            program_preference
        ),

    direct_preference_FDR_significant =
        sum(
            program_preference$preference_FDR_significant,
            na.rm = TRUE
        ),

    figure42_generated =
        file.exists(
            file.path(
                FIGURE_DIR,
                "Figure42_program_module_celltype_integration.pdf"
            )
        ),

    Phase5D6B_status =
        "completed"
)

fwrite(
    completion,
    file.path(
        TABLE_DIR,
        "phase5D6B_completion_summary.tsv"
    ),
    sep = "\t"
)

report_file <- file.path(
    METADATA_DIR,
    "PHASE5D6B_PROGRAM_MODULE_CELLTYPE_INTEGRATION_REPORT.txt"
)

sink(report_file)

cat(
    "Phase 5D6B developmental-program integration\n\n"
)

cat(
    "Canonical definitions: BrainSpan DTHI modules and cell-type marker sets.\n"
)

cat(
    "Phase 5D6A confirmed that BrainSpan and AHBA definitions were identical.\n"
)

cat(
    "Background: genes passing Phase 5D2 trajectory-model filtering.\n"
)

cat(
    "Foregrounds: high- and moderate-confidence Phase 5D3 program assignments.\n\n"
)

cat("Completion summary\n")

print(completion)

cat("\nGene-set mapping audit\n")

print(mapping_audit)

cat("\nProgram-specific enrichment\n")

print(program_enrichment)

cat("\nDirect program preference\n")

print(program_preference)

sink()

log_message("")
log_message(
    "===== Phase 5D6B completed ====="
)

log_message(
    paste(
        capture.output(
            print(completion)
        ),
        collapse = "\n"
    )
)
