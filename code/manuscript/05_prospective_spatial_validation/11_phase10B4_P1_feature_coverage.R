options(
    stringsAsFactors = FALSE,
    warn = 1
)

args <- commandArgs(
    trailingOnly = TRUE
)

lock_path <- args[1]
sn_path <- args[2]
visium_path <- args[3]
out_path <- args[4]

suppressPackageStartupMessages(
    library(
        SeuratObject
    )
)

lock <- read.delim(
    lock_path,
    check.names = FALSE
)

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
        "Could not identify module/gene columns in locked source."
    )
}

gene_col <- normalized[[gene_key]]
module_col <- normalized[[module_key]]

locked <- unique(
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

locked <- locked[
    nzchar(locked$module)
    & nzchar(locked$gene_symbol),
    ,
    drop = FALSE
]

message(
    "Reading snRNA-seq object for feature-name audit only"
)

sn <- readRDS(
    sn_path
)

sn_features <- toupper(
    rownames(
        sn[["RNA"]]
    )
)

rm(sn)

invisible(
    gc(
        verbose = FALSE
    )
)

message(
    "Reading Visium object for feature-name audit only"
)

visium <- readRDS(
    visium_path
)

visium_features <- toupper(
    rownames(
        visium[["Spatial"]]
    )
)

rm(visium)

invisible(
    gc(
        verbose = FALSE
    )
)

modules <- sort(
    unique(
        locked$module
    )
)

coverage_rows <- lapply(
    modules,
    function(module_name) {

        genes <- sort(
            unique(
                locked$gene_symbol[
                    locked$module == module_name
                ]
            )
        )

        sn_match <- genes %in% sn_features
        visium_match <- genes %in% visium_features

        data.frame(
            module = module_name,
            locked_genes = length(genes),
            snRNAseq_RNA_features_matched = sum(
                sn_match
            ),
            snRNAseq_RNA_coverage_pct = round(
                100 * mean(sn_match),
                3
            ),
            Visium_Spatial_features_matched = sum(
                visium_match
            ),
            Visium_Spatial_coverage_pct = round(
                100 * mean(visium_match),
                3
            ),
            matched_in_both = sum(
                sn_match
                & visium_match
            ),
            zero_coverage_snRNAseq = (
                sum(sn_match) == 0
            ),
            zero_coverage_Visium = (
                sum(visium_match) == 0
            ),
            stringsAsFactors = FALSE
        )
    }
)

coverage <- do.call(
    rbind,
    coverage_rows
)

write.table(
    coverage,
    out_path,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)

cat(
    "===== FEATURE-NAME COVERAGE AUDIT =====\n"
)

print(
    coverage,
    row.names = FALSE
)

cat(
    "Expression values accessed: FALSE\n"
)

cat(
    "Module scores computed: FALSE\n"
)
