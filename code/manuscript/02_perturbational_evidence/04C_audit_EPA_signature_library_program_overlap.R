#!/usr/bin/env Rscript

options(
  stringsAsFactors = FALSE,
  width = 220
)

project <- "."

rdata_file <- file.path(
  project,
  "01_raw_data/perturbational_validation/EPA_HTTr",
  "HTTr Signatures.RData"
)

maturation_file <- file.path(
  project,
  "03_processed_data/developmental_trajectory/phase5D/enrichment",
  "phase5D4_maturation_high_increasing_genes.tsv"
)

fetal_file <- file.path(
  project,
  "03_processed_data/developmental_trajectory/phase5D/enrichment",
  "phase5D4_fetal_high_decreasing_genes.tsv"
)

table_dir <- file.path(
  project,
  "07_tables/main_tables/phase6"
)

log_file <- file.path(
  project,
  "09_pipeline_logs/phase6",
  "phase6B4B_signature_library_program_overlap.log"
)

source_summary_file <- file.path(
  table_dir,
  "phase6B4B_signature_source_summary.tsv"
)

type_summary_file <- file.path(
  table_dir,
  "phase6B4B_signature_type_summary.tsv"
)

direction_summary_file <- file.path(
  table_dir,
  "phase6B4B_signature_direction_summary.tsv"
)

program_coverage_file <- file.path(
  table_dir,
  "phase6B4B_program_gene_coverage.tsv"
)

signature_overlap_file <- file.path(
  table_dir,
  "phase6B4B_all_signature_program_overlap.tsv.gz"
)

top_overlap_file <- file.path(
  table_dir,
  "phase6B4B_top_signature_program_overlaps.tsv"
)

keyword_source_file <- file.path(
  table_dir,
  "phase6B4B_HTTr_CMAP_keyword_signature_summary.tsv"
)

completion_file <- file.path(
  table_dir,
  "phase6B4B_completion_summary.tsv"
)

dir.create(
  table_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  dirname(log_file),
  recursive = TRUE,
  showWarnings = FALSE
)

writeLines(
  character(),
  log_file
)

log_message <- function(...) {
  message <- paste0(...)

  cat(
    message,
    "\n",
    sep = ""
  )

  cat(
    message,
    "\n",
    sep = "",
    file = log_file,
    append = TRUE
  )
}

normalize_gene <- function(values) {
  values <- toupper(
    trimws(
      as.character(values)
    )
  )

  values[
    is.na(values) |
    values %in% c(
      "",
      "NA",
      "NAN",
      "NONE",
      "<NA>",
      "-"
    )
  ] <- ""

  values
}

read_program_genes <- function(path) {
  if (!file.exists(path)) {
    stop(
      paste(
        "Missing program file:",
        path
      )
    )
  }

  connection <- if (
    grepl(
      "\\.gz$",
      path,
      ignore.case = TRUE
    )
  ) {
    gzfile(
      path,
      open = "rt"
    )
  } else {
    file(
      path,
      open = "rt"
    )
  }

  on.exit(
    close(connection),
    add = TRUE
  )

  table <- read.delim(
    connection,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )

  candidate_columns <- c(
    "gene_symbol",
    "gene",
    "symbol"
  )

  matched_columns <- intersect(
    candidate_columns,
    colnames(table)
  )

  gene_column <- if (
    length(matched_columns) > 0L
  ) {
    matched_columns[[1L]]
  } else {
    colnames(table)[[1L]]
  }

  genes <- unique(
    normalize_gene(
      table[[gene_column]]
    )
  )

  genes[
    genes != ""
  ]
}

summarize_group <- function(
  database,
  group_column
) {
  groups <- sort(
    unique(
      as.character(
        database[[group_column]]
      )
    )
  )

  rows <- lapply(
    groups,
    function(group_value) {
      subset <- database[
        as.character(
          database[[group_column]]
        ) == group_value,
        ,
        drop = FALSE
      ]

      directional_flag <- (
        tolower(
          as.character(
            subset$type
          )
        ) == "directional"
        |
        !tolower(
          as.character(
            subset$direction
          )
        ) %in% c(
          "",
          "-",
          "nondirectional"
        )
      )

      data.frame(
        group = group_value,
        signature_rows = nrow(subset),
        unique_signatures = length(
          unique(
            subset$signature
          )
        ),
        unique_parents = length(
          unique(
            subset$parent
          )
        ),
        directional_rows = sum(
          directional_flag,
          na.rm = TRUE
        ),
        nondirectional_rows = sum(
          !directional_flag,
          na.rm = TRUE
        ),
        minimum_ngene = suppressWarnings(
          min(
            subset$ngene,
            na.rm = TRUE
          )
        ),
        median_ngene = suppressWarnings(
          median(
            subset$ngene,
            na.rm = TRUE
          )
        ),
        maximum_ngene = suppressWarnings(
          max(
            subset$ngene,
            na.rm = TRUE
          )
        ),
        stringsAsFactors = FALSE
      )
    }
  )

  do.call(
    rbind,
    rows
  )
}

log_message(
  "===== Phase 6B4B started ====="
)

workspace <- new.env(
  parent = emptyenv()
)

loaded_objects <- load(
  rdata_file,
  envir = workspace
)

if (!"sigdb" %in% loaded_objects) {
  stop(
    paste(
      "Expected object 'sigdb' was not found.",
      "Loaded objects:",
      paste(
        loaded_objects,
        collapse = ", "
      )
    )
  )
}

sigdb <- get(
  "sigdb",
  envir = workspace,
  inherits = FALSE
)

required_columns <- c(
  "signature",
  "parent",
  "source",
  "subsource",
  "type",
  "direction",
  "ngene",
  "description",
  "gene.list"
)

missing_columns <- setdiff(
  required_columns,
  colnames(sigdb)
)

if (length(missing_columns) > 0L) {
  stop(
    paste(
      "Missing sigdb columns:",
      paste(
        missing_columns,
        collapse = ", "
      )
    )
  )
}

maturation_genes <- read_program_genes(
  maturation_file
)

fetal_genes <- read_program_genes(
  fetal_file
)

if (length(maturation_genes) != 3333L) {
  stop(
    paste(
      "Unexpected maturation-high count:",
      length(maturation_genes)
    )
  )
}

if (length(fetal_genes) != 5412L) {
  stop(
    paste(
      "Unexpected fetal-high count:",
      length(fetal_genes)
    )
  )
}

log_message(
  "Signature rows: ",
  nrow(sigdb)
)

log_message(
  "Parsing gene lists..."
)

gene_lists <- strsplit(
  as.character(
    sigdb$gene.list
  ),
  split = "|",
  fixed = TRUE
)

gene_lists <- lapply(
  gene_lists,
  function(genes) {
    genes <- unique(
      normalize_gene(
        genes
      )
    )

    genes[
      genes != ""
    ]
  }
)

signature_gene_counts <- vapply(
  gene_lists,
  length,
  integer(1)
)

signature_gene_count_matches <- (
  signature_gene_counts ==
  as.integer(
    sigdb$ngene
  )
)

gene_universe <- sort(
  unique(
    unlist(
      gene_lists,
      use.names = FALSE
    )
  )
)

maturation_overlap <- vapply(
  gene_lists,
  function(genes) {
    sum(
      genes %in% maturation_genes
    )
  },
  integer(1)
)

fetal_overlap <- vapply(
  gene_lists,
  function(genes) {
    sum(
      genes %in% fetal_genes
    )
  },
  integer(1)
)

maturation_overlap_fraction <- ifelse(
  signature_gene_counts > 0L,
  maturation_overlap /
    signature_gene_counts,
  NA_real_
)

fetal_overlap_fraction <- ifelse(
  signature_gene_counts > 0L,
  fetal_overlap /
    signature_gene_counts,
  NA_real_
)

source_summary <- summarize_group(
  sigdb,
  "source"
)

colnames(source_summary)[
  colnames(source_summary) == "group"
] <- "source"

source_summary <- source_summary[
  order(
    -source_summary$signature_rows,
    source_summary$source
  ),
  ,
  drop = FALSE
]

type_summary <- summarize_group(
  sigdb,
  "type"
)

colnames(type_summary)[
  colnames(type_summary) == "group"
] <- "type"

direction_summary <- summarize_group(
  sigdb,
  "direction"
)

colnames(direction_summary)[
  colnames(direction_summary) == "group"
] <- "direction"

write.table(
  source_summary,
  source_summary_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

write.table(
  type_summary,
  type_summary_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

write.table(
  direction_summary,
  direction_summary_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

program_coverage <- data.frame(
  developmental_program = c(
    "maturation_high",
    "fetal_high"
  ),
  program_gene_count = c(
    length(maturation_genes),
    length(fetal_genes)
  ),
  genes_present_in_signature_universe = c(
    sum(
      maturation_genes %in%
      gene_universe
    ),
    sum(
      fetal_genes %in%
      gene_universe
    )
  ),
  genes_absent_from_signature_universe = c(
    sum(
      !maturation_genes %in%
      gene_universe
    ),
    sum(
      !fetal_genes %in%
      gene_universe
    )
  ),
  signature_universe_gene_count = length(
    gene_universe
  ),
  stringsAsFactors = FALSE
)

program_coverage$coverage_fraction <- (
  program_coverage$
    genes_present_in_signature_universe /
  program_coverage$
    program_gene_count
)

write.table(
  program_coverage,
  program_coverage_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

overlap_table <- data.frame(
  signature_index = seq_len(
    nrow(sigdb)
  ),
  signature = as.character(
    sigdb$signature
  ),
  parent = as.character(
    sigdb$parent
  ),
  source = as.character(
    sigdb$source
  ),
  subsource = as.character(
    sigdb$subsource
  ),
  type = as.character(
    sigdb$type
  ),
  direction = as.character(
    sigdb$direction
  ),
  ngene_reported = as.integer(
    sigdb$ngene
  ),
  ngene_parsed = signature_gene_counts,
  gene_count_matches_reported = (
    signature_gene_count_matches
  ),
  maturation_overlap = maturation_overlap,
  maturation_overlap_fraction = (
    maturation_overlap_fraction
  ),
  fetal_overlap = fetal_overlap,
  fetal_overlap_fraction = (
    fetal_overlap_fraction
  ),
  description = as.character(
    sigdb$description
  ),
  stringsAsFactors = FALSE
)

connection <- gzfile(
  signature_overlap_file,
  open = "wt"
)

write.table(
  overlap_table,
  connection,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

close(
  connection
)

top_maturation <- overlap_table[
  order(
    -overlap_table$maturation_overlap,
    -overlap_table$maturation_overlap_fraction,
    overlap_table$signature
  ),
  ,
  drop = FALSE
]

top_maturation <- head(
  top_maturation,
  100L
)

top_maturation$ranked_program <- (
  "maturation_high"
)

top_fetal <- overlap_table[
  order(
    -overlap_table$fetal_overlap,
    -overlap_table$fetal_overlap_fraction,
    overlap_table$signature
  ),
  ,
  drop = FALSE
]

top_fetal <- head(
  top_fetal,
  100L
)

top_fetal$ranked_program <- (
  "fetal_high"
)

top_overlap <- rbind(
  top_maturation,
  top_fetal
)

top_overlap <- top_overlap[
  ,
  c(
    "ranked_program",
    setdiff(
      colnames(top_overlap),
      "ranked_program"
    )
  ),
  drop = FALSE
]

write.table(
  top_overlap,
  top_overlap_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

search_text <- paste(
  sigdb$signature,
  sigdb$parent,
  sigdb$source,
  sigdb$subsource,
  sigdb$description
)

explicit_httr_flag <- grepl(
  "HTTR|HIGH[ _-]*THROUGHPUT[ _-]*TRANSCRIPT|TEMPOSEQ|TEMPO[ _-]*SEQ|TOXCAST",
  search_text,
  ignore.case = TRUE
)

cmap_flag <- grepl(
  "CMAP|CONNECTIVITY[ _-]*MAP",
  search_text,
  ignore.case = TRUE
)

keyword_summary <- data.frame(
  category = c(
    "explicit_HTTr_TempoSeq_ToxCast_text",
    "CMAP_connectivity_map_text"
  ),
  signature_rows = c(
    sum(
      explicit_httr_flag,
      na.rm = TRUE
    ),
    sum(
      cmap_flag,
      na.rm = TRUE
    )
  ),
  unique_signatures = c(
    length(
      unique(
        sigdb$signature[
          explicit_httr_flag
        ]
      )
    ),
    length(
      unique(
        sigdb$signature[
          cmap_flag
        ]
      )
    )
  ),
  unique_sources = c(
    length(
      unique(
        sigdb$source[
          explicit_httr_flag
        ]
      )
    ),
    length(
      unique(
        sigdb$source[
          cmap_flag
        ]
      )
    )
  ),
  source_names = c(
    paste(
      sort(
        unique(
          sigdb$source[
            explicit_httr_flag
          ]
        )
      ),
      collapse = "|"
    ),
    paste(
      sort(
        unique(
          sigdb$source[
            cmap_flag
          ]
        )
      ),
      collapse = "|"
    )
  ),
  stringsAsFactors = FALSE
)

write.table(
  keyword_summary,
  keyword_source_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

directional_flag <- (
  tolower(
    as.character(
      sigdb$type
    )
  ) == "directional"
  |
  !tolower(
    as.character(
      sigdb$direction
    )
  ) %in% c(
    "",
    "-",
    "nondirectional"
  )
)

completion <- data.frame(
  signature_rows = nrow(sigdb),
  unique_signature_names = length(
    unique(
      sigdb$signature
    )
  ),
  unique_parent_names = length(
    unique(
      sigdb$parent
    )
  ),
  unique_sources = length(
    unique(
      sigdb$source
    )
  ),
  signature_gene_universe = length(
    gene_universe
  ),
  directional_signature_rows = sum(
    directional_flag,
    na.rm = TRUE
  ),
  nondirectional_signature_rows = sum(
    !directional_flag,
    na.rm = TRUE
  ),
  gene_count_matching_rows = sum(
    signature_gene_count_matches,
    na.rm = TRUE
  ),
  gene_count_mismatching_rows = sum(
    !signature_gene_count_matches,
    na.rm = TRUE
  ),
  explicit_HTTr_keyword_rows = sum(
    explicit_httr_flag,
    na.rm = TRUE
  ),
  CMAP_keyword_rows = sum(
    cmap_flag,
    na.rm = TRUE
  ),
  maturation_program_coverage = (
    program_coverage$
      coverage_fraction[
        program_coverage$
          developmental_program ==
          "maturation_high"
      ]
  ),
  fetal_program_coverage = (
    program_coverage$
      coverage_fraction[
        program_coverage$
          developmental_program ==
          "fetal_high"
      ]
  ),
  Phase6B4B_status = "completed",
  stringsAsFactors = FALSE
)

write.table(
  completion,
  completion_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

log_message("")
log_message(
  "===== SOURCE SUMMARY ====="
)

capture <- capture.output(
  print(
    source_summary,
    row.names = FALSE
  )
)

for (line in capture) {
  log_message(line)
}

log_message("")
log_message(
  "===== TYPE SUMMARY ====="
)

capture <- capture.output(
  print(
    type_summary,
    row.names = FALSE
  )
)

for (line in capture) {
  log_message(line)
}

log_message("")
log_message(
  "===== DIRECTION SUMMARY ====="
)

capture <- capture.output(
  print(
    direction_summary,
    row.names = FALSE
  )
)

for (line in capture) {
  log_message(line)
}

log_message("")
log_message(
  "===== PROGRAM COVERAGE ====="
)

capture <- capture.output(
  print(
    program_coverage,
    row.names = FALSE
  )
)

for (line in capture) {
  log_message(line)
}

log_message("")
log_message(
  "===== HTTr / CMAP KEYWORD SUMMARY ====="
)

capture <- capture.output(
  print(
    keyword_summary,
    row.names = FALSE
  )
)

for (line in capture) {
  log_message(line)
}

log_message("")
log_message(
  "===== PHASE 6B4B COMPLETION ====="
)

capture <- capture.output(
  print(
    completion,
    row.names = FALSE
  )
)

for (line in capture) {
  log_message(line)
}
