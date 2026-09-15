#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(data.table)
    library(msigdbr)
})


PROJECT <- normalizePath(
    ".",
    mustWork = TRUE
)

RESOURCE_DIR <- file.path(
    PROJECT,
    "03_processed_data",
    "developmental_trajectory",
    "phase5D",
    "phase5D4_R1",
    "resources"
)

TABLE_DIR <- file.path(
    PROJECT,
    "07_tables",
    "main_tables",
    "phase5",
    "phase5D4_R1"
)

LOG_FILE <- file.path(
    PROJECT,
    "09_pipeline_logs",
    "phase5",
    "phase5D4_R1",
    "phase5D4_R1_msigdbr_resource_freeze_internal.log"
)


for (directory in c(
    RESOURCE_DIR,
    TABLE_DIR,
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


find_column <- function(
    table,
    candidates,
    required = TRUE
) {
    available <- intersect(
        candidates,
        names(table)
    )

    if (length(available) > 0L) {
        return(
            available[[1L]]
        )
    }

    if (required) {
        stop(
            paste0(
                "Required column was not found. Candidates: ",
                paste(
                    candidates,
                    collapse = ", "
                )
            )
        )
    }

    NA_character_
}


retrieve_msigdbr <- function(
    collection,
    subcollection = NULL
) {
    available_arguments <- names(
        formals(
            msigdbr::msigdbr
        )
    )

    call_arguments <- list(
        species = "Homo sapiens"
    )

    if ("db_species" %in% available_arguments) {
        call_arguments$db_species <- "HS"
    }

    if ("collection" %in% available_arguments) {
        call_arguments$collection <- collection
    } else if ("category" %in% available_arguments) {
        call_arguments$category <- collection
    } else {
        stop(
            paste0(
                "The installed msigdbr version exposes ",
                "neither collection nor category arguments."
            )
        )
    }

    if (!is.null(subcollection)) {
        if ("subcollection" %in% available_arguments) {
            call_arguments$subcollection <- subcollection
        } else if ("subcategory" %in% available_arguments) {
            call_arguments$subcategory <- subcollection
        } else {
            stop(
                paste0(
                    "The installed msigdbr version exposes ",
                    "neither subcollection nor subcategory arguments."
                )
            )
        }
    }

    as.data.table(
        do.call(
            msigdbr::msigdbr,
            call_arguments
        )
    )
}


specifications <- data.table(
    enrichment_source = c(
        "GO_BP",
        "GO_CC",
        "GO_MF",
        "Reactome",
        "Hallmark"
    ),
    collection = c(
        "C5",
        "C5",
        "C5",
        "C2",
        "H"
    ),
    subcollection = c(
        "GO:BP",
        "GO:CC",
        "GO:MF",
        "CP:REACTOME",
        NA_character_
    )
)


all_resources <- vector(
    mode = "list",
    length = nrow(specifications)
)

audit_rows <- vector(
    mode = "list",
    length = nrow(specifications)
)


log_message(
    "===== Phase 5D4-R1 MSigDB resource freeze started ====="
)

log_message(
    "R version: ",
    R.version.string
)

log_message(
    "msigdbr version: ",
    as.character(
        packageVersion("msigdbr")
    )
)


for (index in seq_len(nrow(specifications))) {
    source_name <- specifications[
        index,
        enrichment_source
    ]

    collection_name <- specifications[
        index,
        collection
    ]

    subcollection_name <- specifications[
        index,
        subcollection
    ]

    log_message("")
    log_message(
        "Retrieving ",
        source_name,
        "..."
    )

    resource <- retrieve_msigdbr(
        collection = collection_name,
        subcollection = if (
            is.na(subcollection_name)
        ) {
            NULL
        } else {
            subcollection_name
        }
    )

    if (nrow(resource) == 0L) {
        stop(
            paste0(
                "No rows were returned for ",
                source_name,
                "."
            )
        )
    }

    gene_column <- find_column(
        resource,
        c(
            "gene_symbol",
            "human_gene_symbol"
        )
    )

    id_column <- find_column(
        resource,
        c(
            "gs_id",
            "gs_name"
        )
    )

    name_column <- find_column(
        resource,
        c(
            "gs_name",
            "gs_id"
        )
    )

    description_column <- find_column(
        resource,
        c(
            "gs_description",
            "gs_name"
        ),
        required = FALSE
    )

    version_column <- find_column(
        resource,
        c(
            "db_version"
        ),
        required = FALSE
    )

    standardized <- data.table(
        enrichment_source = source_name,
        collection = collection_name,
        subcollection = subcollection_name,
        ID = as.character(
            resource[[id_column]]
        ),
        term_name = as.character(
            resource[[name_column]]
        ),
        Description = if (
            is.na(description_column)
        ) {
            as.character(
                resource[[name_column]]
            )
        } else {
            as.character(
                resource[[description_column]]
            )
        },
        gene_symbol = normalize_symbol(
            resource[[gene_column]]
        ),
        database_version = if (
            is.na(version_column)
        ) {
            NA_character_
        } else {
            as.character(
                resource[[version_column]]
            )
        }
    )

    standardized <- unique(
        standardized[
            !is.na(gene_symbol)
            &
            !is.na(ID)
            &
            ID != ""
        ]
    )

    if (nrow(standardized) == 0L) {
        stop(
            paste0(
                "Standardization removed all rows for ",
                source_name,
                "."
            )
        )
    }

    duplicate_pairs <- (
        nrow(standardized)
        -
        nrow(
            unique(
                standardized[
                    ,
                    .(
                        ID,
                        gene_symbol
                    )
                ]
            )
        )
    )

    if (duplicate_pairs != 0L) {
        stop(
            paste0(
                "Duplicate term-gene pairs remain for ",
                source_name,
                ": ",
                duplicate_pairs
            )
        )
    }

    source_tsv <- file.path(
        RESOURCE_DIR,
        paste0(
            "phase5D4_R1_",
            source_name,
            "_gene_sets.tsv.gz"
        )
    )

    source_rds <- file.path(
        RESOURCE_DIR,
        paste0(
            "phase5D4_R1_",
            source_name,
            "_gene_sets.rds"
        )
    )

    fwrite(
        standardized,
        source_tsv,
        sep = "\t",
        compress = "gzip"
    )

    saveRDS(
        standardized,
        source_rds,
        compress = "xz"
    )

    all_resources[[index]] <- standardized

    database_versions <- sort(
        unique(
            na.omit(
                standardized$database_version
            )
        )
    )

    audit_rows[[index]] <- data.table(
        enrichment_source = source_name,
        collection = collection_name,
        subcollection = subcollection_name,
        msigdbr_version = as.character(
            packageVersion("msigdbr")
        ),
        database_version = if (
            length(database_versions) == 0L
        ) {
            NA_character_
        } else {
            paste(
                database_versions,
                collapse = ";"
            )
        },
        resource_rows = nrow(standardized),
        unique_terms = uniqueN(
            standardized$ID
        ),
        unique_gene_symbols = uniqueN(
            standardized$gene_symbol
        ),
        duplicate_term_gene_pairs = duplicate_pairs,
        resource_tsv = source_tsv,
        resource_rds = source_rds
    )

    log_message(
        source_name,
        ": ",
        uniqueN(standardized$ID),
        " terms; ",
        uniqueN(standardized$gene_symbol),
        " genes; ",
        nrow(standardized),
        " term-gene rows"
    )
}


combined <- rbindlist(
    all_resources,
    use.names = TRUE,
    fill = TRUE
)

combined <- unique(
    combined
)

combined_tsv <- file.path(
    RESOURCE_DIR,
    "phase5D4_R1_all_five_sources_gene_sets.tsv.gz"
)

combined_rds <- file.path(
    RESOURCE_DIR,
    "phase5D4_R1_all_five_sources_gene_sets.rds"
)

fwrite(
    combined,
    combined_tsv,
    sep = "\t",
    compress = "gzip"
)

saveRDS(
    combined,
    combined_rds,
    compress = "xz"
)


audit <- rbindlist(
    audit_rows,
    use.names = TRUE,
    fill = TRUE
)

audit[
    ,
    source_order := match(
        enrichment_source,
        specifications$enrichment_source
    )
]

setorder(
    audit,
    source_order
)

audit[
    ,
    source_order := NULL
]

audit_file <- file.path(
    TABLE_DIR,
    "phase5D4_R1_msigdbr_resource_freeze_audit.tsv"
)

fwrite(
    audit,
    audit_file,
    sep = "\t"
)


combined_summary <- data.table(
    enrichment_sources = uniqueN(
        combined$enrichment_source
    ),
    combined_term_gene_rows = nrow(
        combined
    ),
    combined_unique_source_terms = nrow(
        unique(
            combined[
                ,
                .(
                    enrichment_source,
                    ID
                )
            ]
        )
    ),
    combined_unique_gene_symbols = uniqueN(
        combined$gene_symbol
    ),
    msigdbr_version = as.character(
        packageVersion("msigdbr")
    ),
    R_version = R.version.string,
    resource_freeze_status = "completed"
)

summary_file <- file.path(
    TABLE_DIR,
    "phase5D4_R1_resource_freeze_completion.tsv"
)

fwrite(
    combined_summary,
    summary_file,
    sep = "\t"
)


if (nrow(audit) != 5L) {
    stop(
        paste0(
            "Expected five audit rows; observed ",
            nrow(audit),
            "."
        )
    )
}

if (
    !identical(
        audit$enrichment_source,
        specifications$enrichment_source
    )
) {
    stop(
        "The five enrichment sources are missing or misordered."
    )
}

if (
    any(audit$unique_terms <= 0L)
    ||
    any(audit$unique_gene_symbols <= 0L)
    ||
    any(audit$resource_rows <= 0L)
) {
    stop(
        "At least one frozen source is empty."
    )
}

if (
    any(
        audit$duplicate_term_gene_pairs != 0L
    )
) {
    stop(
        "Duplicate term-gene pairs remain."
    )
}


log_message("")
log_message(
    "===== PHASE 5D4-R1 RESOURCE FREEZE ====="
)

print(
    audit
)

log_message(
    "Combined rows: ",
    nrow(combined)
)

log_message(
    "Combined source-specific terms: ",
    nrow(
        unique(
            combined[
                ,
                .(
                    enrichment_source,
                    ID
                )
            ]
        )
    )
)

log_message(
    "Combined genes: ",
    uniqueN(combined$gene_symbol)
)

log_message(
    "Combined TSV: ",
    combined_tsv
)

log_message(
    "Combined RDS: ",
    combined_rds
)

log_message(
    "Resource audit: ",
    audit_file
)

log_message(
    "Completion summary: ",
    summary_file
)

log_message(
    "RESOURCE FREEZE STATUS: PASSED"
)
