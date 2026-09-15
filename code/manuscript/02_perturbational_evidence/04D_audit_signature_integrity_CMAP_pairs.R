#!/usr/bin/env Rscript

options(
  stringsAsFactors = FALSE,
  width = 220
)

project <- "."

input_file <- file.path(
  project,
  "01_raw_data/perturbational_validation/EPA_HTTr",
  "HTTr Signatures.RData"
)

table_dir <- file.path(
  project,
  "07_tables/main_tables/phase6"
)

log_file <- file.path(
  project,
  "09_pipeline_logs/phase6",
  "phase6B4C_signature_integrity_CMAP_pairs.log"
)

mismatch_source_file <- file.path(
  table_dir,
  "phase6B4C_gene_count_mismatch_by_source.tsv"
)

difference_distribution_file <- file.path(
  table_dir,
  "phase6B4C_gene_count_difference_distribution.tsv"
)

mismatch_examples_file <- file.path(
  table_dir,
  "phase6B4C_gene_count_mismatch_examples.tsv"
)

explicit_httr_file <- file.path(
  table_dir,
  "phase6B4C_explicit_HTTr_keyword_rows.tsv"
)

cmap_parent_file <- file.path(
  table_dir,
  "phase6B4C_CMAP_parent_pair_audit.tsv"
)

cmap_summary_file <- file.path(
  table_dir,
  "phase6B4C_CMAP_pair_summary.tsv"
)

completion_file <- file.path(
  table_dir,
  "phase6B4C_completion_summary.tsv"
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

safe_range <- function(values) {
  values <- values[
    is.finite(values)
  ]

  if (length(values) == 0L) {
    return(
      c(
        minimum = NA_real_,
        median = NA_real_,
        maximum = NA_real_
      )
    )
  }

  c(
    minimum = min(values),
    median = median(values),
    maximum = max(values)
  )
}

log_message(
  "===== Phase 6B4C started ====="
)

workspace <- new.env(
  parent = emptyenv()
)

loaded_objects <- load(
  input_file,
  envir = workspace
)

if (!"sigdb" %in% loaded_objects) {
  stop(
    "Expected object 'sigdb' was not found."
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
      "Missing columns:",
      paste(
        missing_columns,
        collapse = ", "
      )
    )
  )
}

raw_gene_lists <- strsplit(
  as.character(
    sigdb$gene.list
  ),
  split = "|",
  fixed = TRUE
)

raw_token_count <- vapply(
  raw_gene_lists,
  length,
  integer(1)
)

normalized_gene_lists <- lapply(
  raw_gene_lists,
  normalize_gene
)

nonempty_gene_lists <- lapply(
  normalized_gene_lists,
  function(genes) {
    genes[
      genes != ""
    ]
  }
)

nonempty_token_count <- vapply(
  nonempty_gene_lists,
  length,
  integer(1)
)

unique_gene_lists <- lapply(
  nonempty_gene_lists,
  unique
)

unique_gene_count <- vapply(
  unique_gene_lists,
  length,
  integer(1)
)

duplicate_tokens_removed <- (
  nonempty_token_count -
  unique_gene_count
)

reported_ngene <- as.integer(
  sigdb$ngene
)

reported_minus_raw <- (
  reported_ngene -
  raw_token_count
)

reported_minus_nonempty <- (
  reported_ngene -
  nonempty_token_count
)

reported_minus_unique <- (
  reported_ngene -
  unique_gene_count
)

integrity_table <- data.frame(
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
  reported_ngene = reported_ngene,
  raw_token_count = raw_token_count,
  nonempty_token_count = nonempty_token_count,
  unique_gene_count = unique_gene_count,
  duplicate_tokens_removed = duplicate_tokens_removed,
  reported_matches_raw = (
    reported_ngene ==
    raw_token_count
  ),
  reported_matches_nonempty = (
    reported_ngene ==
    nonempty_token_count
  ),
  reported_matches_unique = (
    reported_ngene ==
    unique_gene_count
  ),
  reported_minus_raw = reported_minus_raw,
  reported_minus_nonempty = (
    reported_minus_nonempty
  ),
  reported_minus_unique = (
    reported_minus_unique
  ),
  description = as.character(
    sigdb$description
  ),
  gene_list = as.character(
    sigdb$gene.list
  ),
  stringsAsFactors = FALSE
)

sources <- sort(
  unique(
    integrity_table$source
  )
)

source_rows <- lapply(
  sources,
  function(source_name) {
    current <- integrity_table[
      integrity_table$source ==
        source_name,
      ,
      drop = FALSE
    ]

    difference_range <- safe_range(
      current$reported_minus_unique
    )

    data.frame(
      source = source_name,
      signature_rows = nrow(current),
      reported_matches_raw_rows = sum(
        current$reported_matches_raw,
        na.rm = TRUE
      ),
      reported_matches_nonempty_rows = sum(
        current$reported_matches_nonempty,
        na.rm = TRUE
      ),
      reported_matches_unique_rows = sum(
        current$reported_matches_unique,
        na.rm = TRUE
      ),
      mismatching_unique_rows = sum(
        !current$reported_matches_unique,
        na.rm = TRUE
      ),
      rows_with_duplicate_tokens = sum(
        current$duplicate_tokens_removed > 0L,
        na.rm = TRUE
      ),
      duplicate_tokens_removed_total = sum(
        current$duplicate_tokens_removed,
        na.rm = TRUE
      ),
      minimum_reported_minus_unique = (
        difference_range[
          "minimum"
        ]
      ),
      median_reported_minus_unique = (
        difference_range[
          "median"
        ]
      ),
      maximum_reported_minus_unique = (
        difference_range[
          "maximum"
        ]
      ),
      stringsAsFactors = FALSE
    )
  }
)

source_summary <- do.call(
  rbind,
  source_rows
)

source_summary <- source_summary[
  order(
    -source_summary$mismatching_unique_rows,
    source_summary$source
  ),
  ,
  drop = FALSE
]

write.table(
  source_summary,
  mismatch_source_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

difference_values <- sort(
  unique(
    reported_minus_unique
  )
)

difference_rows <- lapply(
  difference_values,
  function(difference_value) {
    current <- integrity_table[
      integrity_table$
        reported_minus_unique ==
        difference_value,
      ,
      drop = FALSE
    ]

    data.frame(
      reported_minus_unique = (
        difference_value
      ),
      signature_rows = nrow(current),
      unique_sources = length(
        unique(
          current$source
        )
      ),
      source_names = paste(
        sort(
          unique(
            current$source
          )
        ),
        collapse = "|"
      ),
      stringsAsFactors = FALSE
    )
  }
)

difference_distribution <- do.call(
  rbind,
  difference_rows
)

difference_distribution <- (
  difference_distribution[
    order(
      -difference_distribution$
        signature_rows,
      difference_distribution$
        reported_minus_unique
    ),
    ,
    drop = FALSE
  ]
)

write.table(
  difference_distribution,
  difference_distribution_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

mismatch_examples <- integrity_table[
  !integrity_table$
    reported_matches_unique,
  ,
  drop = FALSE
]

mismatch_examples$absolute_difference <- abs(
  mismatch_examples$
    reported_minus_unique
)

mismatch_examples <- mismatch_examples[
  order(
    -mismatch_examples$
      absolute_difference,
    -mismatch_examples$
      duplicate_tokens_removed,
    mismatch_examples$source,
    mismatch_examples$signature
  ),
  ,
  drop = FALSE
]

mismatch_examples <- head(
  mismatch_examples,
  200L
)

write.table(
  mismatch_examples,
  mismatch_examples_file,
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

explicit_httr_rows <- integrity_table[
  explicit_httr_flag,
  ,
  drop = FALSE
]

write.table(
  explicit_httr_rows,
  explicit_httr_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

cmap <- integrity_table[
  toupper(
    integrity_table$source
  ) == "CMAP",
  ,
  drop = FALSE
]

if (nrow(cmap) == 0L) {
  stop(
    "No CMAP rows were detected."
  )
}

cmap_parents <- sort(
  unique(
    cmap$parent
  )
)

cmap_pair_rows <- lapply(
  cmap_parents,
  function(parent_name) {
    current <- cmap[
      cmap$parent ==
        parent_name,
      ,
      drop = FALSE
    ]

    directions <- tolower(
      trimws(
        current$direction
      )
    )

    up_rows <- sum(
      directions == "up"
    )

    down_rows <- sum(
      directions == "dn"
    )

    other_rows <- sum(
      !directions %in% c(
        "up",
        "dn"
      )
    )

    data.frame(
      parent = parent_name,
      total_rows = nrow(current),
      up_rows = up_rows,
      down_rows = down_rows,
      other_direction_rows = other_rows,
      pair_complete = (
        nrow(current) == 2L &&
        up_rows == 1L &&
        down_rows == 1L &&
        other_rows == 0L
      ),
      unique_signature_names = length(
        unique(
          current$signature
        )
      ),
      unique_reported_ngene_values = paste(
        sort(
          unique(
            current$reported_ngene
          )
        ),
        collapse = "|"
      ),
      unique_parsed_ngene_values = paste(
        sort(
          unique(
            current$unique_gene_count
          )
        ),
        collapse = "|"
      ),
      all_gene_counts_match_unique = all(
        current$reported_matches_unique
      ),
      stringsAsFactors = FALSE
    )
  }
)

cmap_pair_audit <- do.call(
  rbind,
  cmap_pair_rows
)

cmap_pair_audit <- cmap_pair_audit[
  order(
    cmap_pair_audit$pair_complete,
    cmap_pair_audit$parent
  ),
  ,
  drop = FALSE
]

write.table(
  cmap_pair_audit,
  cmap_parent_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

cmap_summary <- data.frame(
  CMAP_signature_rows = nrow(cmap),
  CMAP_unique_parents = nrow(
    cmap_pair_audit
  ),
  complete_up_down_pairs = sum(
    cmap_pair_audit$pair_complete
  ),
  incomplete_pairs = sum(
    !cmap_pair_audit$pair_complete
  ),
  parents_with_exactly_two_rows = sum(
    cmap_pair_audit$total_rows == 2L
  ),
  parents_with_one_up_one_down = sum(
    cmap_pair_audit$up_rows == 1L &
    cmap_pair_audit$down_rows == 1L
  ),
  CMAP_gene_count_matching_rows = sum(
    cmap$reported_matches_unique
  ),
  CMAP_gene_count_mismatching_rows = sum(
    !cmap$reported_matches_unique
  ),
  stringsAsFactors = FALSE
)

write.table(
  cmap_summary,
  cmap_summary_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

mismatching_rows <- sum(
  !integrity_table$
    reported_matches_unique
)

duplicate_explained_rows <- sum(
  !integrity_table$
    reported_matches_unique &
  integrity_table$
    reported_matches_nonempty
)

completion <- data.frame(
  signature_rows = nrow(
    integrity_table
  ),
  reported_matches_raw_rows = sum(
    integrity_table$
      reported_matches_raw
  ),
  reported_matches_nonempty_rows = sum(
    integrity_table$
      reported_matches_nonempty
  ),
  reported_matches_unique_rows = sum(
    integrity_table$
      reported_matches_unique
  ),
  gene_count_mismatching_rows = (
    mismatching_rows
  ),
  mismatches_explained_by_duplicate_collapse = (
    duplicate_explained_rows
  ),
  rows_with_duplicate_tokens = sum(
    integrity_table$
      duplicate_tokens_removed > 0L
  ),
  explicit_HTTr_keyword_rows = nrow(
    explicit_httr_rows
  ),
  CMAP_signature_rows = nrow(cmap),
  CMAP_unique_parents = nrow(
    cmap_pair_audit
  ),
  CMAP_complete_up_down_pairs = sum(
    cmap_pair_audit$pair_complete
  ),
  CMAP_incomplete_pairs = sum(
    !cmap_pair_audit$pair_complete
  ),
  Phase6B4C_status = "completed",
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
  "===== GENE-COUNT MISMATCH BY SOURCE ====="
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
  "===== EXPLICIT HTTr KEYWORD ROWS ====="
)

capture <- capture.output(
  print(
    explicit_httr_rows[
      ,
      c(
        "signature",
        "parent",
        "source",
        "subsource",
        "type",
        "direction",
        "reported_ngene",
        "unique_gene_count",
        "description"
      ),
      drop = FALSE
    ],
    row.names = FALSE
  )
)

for (line in capture) {
  log_message(line)
}

log_message("")
log_message(
  "===== CMAP PAIR SUMMARY ====="
)

capture <- capture.output(
  print(
    cmap_summary,
    row.names = FALSE
  )
)

for (line in capture) {
  log_message(line)
}

log_message("")
log_message(
  "===== PHASE 6B4C COMPLETION ====="
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
