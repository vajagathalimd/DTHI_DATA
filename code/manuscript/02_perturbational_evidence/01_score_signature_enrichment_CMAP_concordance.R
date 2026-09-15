#!/usr/bin/env Rscript

options(
  stringsAsFactors = FALSE,
  width = 240,
  scipen = 999
)

project <- "."

signature_file <- file.path(
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

processed_dir <- file.path(
  project,
  "03_processed_data/perturbational_validation/phase6C"
)

table_dir <- file.path(
  project,
  "07_tables/main_tables/phase6"
)

log_file <- file.path(
  project,
  "09_pipeline_logs/phase6",
  "phase6C1_signature_enrichment_CMAP_concordance.log"
)

reference_output <- file.path(
  processed_dir,
  "phase6C1_reference_signature_program_enrichment.tsv.gz"
)

reference_top_output <- file.path(
  table_dir,
  "phase6C1_top_reference_signature_enrichments.tsv"
)

reference_summary_output <- file.path(
  table_dir,
  "phase6C1_reference_enrichment_source_summary.tsv"
)

cmap_output <- file.path(
  processed_dir,
  "phase6C1_CMAP_directional_concordance.tsv.gz"
)

cmap_top_output <- file.path(
  table_dir,
  "phase6C1_top_CMAP_directional_candidates.tsv"
)

cmap_gene_coverage_output <- file.path(
  table_dir,
  "phase6C1_CMAP_program_gene_coverage.tsv"
)

cmap_summary_output <- file.path(
  table_dir,
  "phase6C1_CMAP_directional_summary.tsv"
)

completion_output <- file.path(
  table_dir,
  "phase6C1_completion_summary.tsv"
)

dir.create(
  processed_dir,
  recursive = TRUE,
  showWarnings = FALSE
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
        "Missing developmental-program file:",
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

  matched <- intersect(
    candidate_columns,
    colnames(table)
  )

  gene_column <- if (
    length(matched) > 0L
  ) {
    matched[[1L]]
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

canonical_source <- function(source) {
  source <- as.character(source)

  output <- source

  output[
    tolower(source) == "dorothea"
  ] <- "DoRothEA"

  output
}

safe_hypergeometric_p <- function(
  overlap,
  program_in_universe,
  universe_size,
  signature_size
) {
  if (
    universe_size <= 0L ||
    program_in_universe <= 0L ||
    signature_size <= 0L ||
    overlap <= 0L
  ) {
    return(1)
  }

  phyper(
    q = overlap - 1L,
    m = program_in_universe,
    n = universe_size -
      program_in_universe,
    k = signature_size,
    lower.tail = FALSE
  )
}

safe_fold_enrichment <- function(
  overlap,
  signature_size,
  program_in_universe,
  universe_size
) {
  if (
    signature_size <= 0L ||
    program_in_universe <= 0L ||
    universe_size <= 0L
  ) {
    return(NA_real_)
  }

  observed_fraction <- overlap /
    signature_size

  expected_fraction <- program_in_universe /
    universe_size

  if (expected_fraction <= 0) {
    return(NA_real_)
  }

  observed_fraction /
    expected_fraction
}

safe_log2_odds_ratio <- function(
  first_hits,
  first_nonhits,
  second_hits,
  second_nonhits
) {
  numerator <- (
    (first_hits + 0.5) /
    (first_nonhits + 0.5)
  )

  denominator <- (
    (second_hits + 0.5) /
    (second_nonhits + 0.5)
  )

  log2(
    numerator /
    denominator
  )
}

one_sided_preference_p <- function(
  preferred_hits,
  preferred_size,
  comparison_hits,
  comparison_size
) {
  total_hits <- preferred_hits +
    comparison_hits

  total_size <- preferred_size +
    comparison_size

  if (
    total_size <= 0L ||
    total_hits <= 0L
  ) {
    return(1)
  }

  total_nonhits <- total_size -
    total_hits

  phyper(
    q = preferred_hits - 1L,
    m = total_hits,
    n = total_nonhits,
    k = preferred_size,
    lower.tail = FALSE
  )
}

combine_two_pvalues <- function(
  first_p,
  second_p
) {
  first_p <- max(
    first_p,
    .Machine$double.xmin
  )

  second_p <- max(
    second_p,
    .Machine$double.xmin
  )

  statistic <- -2 * (
    log(first_p) +
    log(second_p)
  )

  pchisq(
    statistic,
    df = 4,
    lower.tail = FALSE
  )
}

extract_dtxsid <- function(description) {
  description <- as.character(description)

  match <- regexpr(
    "DTXSID[0-9]+",
    description,
    perl = TRUE,
    ignore.case = TRUE
  )

  if (match[[1L]] < 0L) {
    return("")
  }

  regmatches(
    description,
    match
  )
}

extract_chemical_name <- function(description) {
  description <- trimws(
    as.character(description)
  )

  description <- sub(
    "\\s+DTXSID[0-9]+.*$",
    "",
    description,
    ignore.case = TRUE,
    perl = TRUE
  )

  trimws(
    description
  )
}

write_gzip_table <- function(
  table,
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
    table,
    connection,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE,
    na = ""
  )
}

log_message(
  "===== Phase 6C1 started ====="
)

maturation_genes <- read_program_genes(
  maturation_file
)

fetal_genes <- read_program_genes(
  fetal_file
)

if (length(maturation_genes) != 3333L) {
  stop(
    paste(
      "Unexpected maturation-high gene count:",
      length(maturation_genes)
    )
  )
}

if (length(fetal_genes) != 5412L) {
  stop(
    paste(
      "Unexpected fetal-high gene count:",
      length(fetal_genes)
    )
  )
}

if (
  length(
    intersect(
      maturation_genes,
      fetal_genes
    )
  ) != 0L
) {
  stop(
    "The developmental programs are not disjoint."
  )
}

workspace <- new.env(
  parent = emptyenv()
)

loaded_objects <- load(
  signature_file,
  envir = workspace
)

if (!"sigdb" %in% loaded_objects) {
  stop(
    "Expected RData object 'sigdb' was not found."
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

sigdb$source_original <- as.character(
  sigdb$source
)

sigdb$source_canonical <- canonical_source(
  sigdb$source_original
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

parsed_sizes <- vapply(
  gene_lists,
  length,
  integer(1)
)

log_message(
  "Signature rows loaded: ",
  nrow(sigdb)
)

log_message(
  "Parsed signature gene lists: ",
  length(gene_lists)
)

# ------------------------------------------------------------------
# Part A: source-specific reference-signature enrichment
# ------------------------------------------------------------------

reference_indices <- which(
  toupper(
    sigdb$source_original
  ) != "CMAP"
)

reference_sources <- sort(
  unique(
    sigdb$source_canonical[
      reference_indices
    ]
  )
)

program_names <- c(
  "maturation_high",
  "fetal_high"
)

program_gene_sets <- list(
  maturation_high = maturation_genes,
  fetal_high = fetal_genes
)

reference_rows <- list()

reference_row_index <- 0L

log_message(
  "Scoring non-CMAP reference signatures..."
)

for (source_name in reference_sources) {
  source_indices <- reference_indices[
    sigdb$source_canonical[
      reference_indices
    ] == source_name
  ]

  source_universe <- sort(
    unique(
      unlist(
        gene_lists[
          source_indices
        ],
        use.names = FALSE
      )
    )
  )

  universe_size <- length(
    source_universe
  )

  for (program_name in program_names) {
    program_genes <- program_gene_sets[[program_name]]

    program_in_universe <- intersect(
      program_genes,
      source_universe
    )

    program_background_size <- length(
      program_in_universe
    )

    for (signature_index in source_indices) {
      genes <- gene_lists[[signature_index]]

      signature_size <- length(
        genes
      )

      overlapping_genes <- intersect(
        genes,
        program_in_universe
      )

      overlap <- length(
        overlapping_genes
      )

      p_value <- safe_hypergeometric_p(
        overlap = overlap,
        program_in_universe = program_background_size,
        universe_size = universe_size,
        signature_size = signature_size
      )

      fold_enrichment <- safe_fold_enrichment(
        overlap = overlap,
        signature_size = signature_size,
        program_in_universe = program_background_size,
        universe_size = universe_size
      )

      primary_size_eligible <- (
        signature_size >= 10L &&
        signature_size <= 1000L
      )

      reference_row_index <- (
        reference_row_index +
        1L
      )

      reference_rows[[reference_row_index]] <- data.frame(
        signature_index = signature_index,
        developmental_program = program_name,
        signature = as.character(
          sigdb$signature[
            signature_index
          ]
        ),
        parent = as.character(
          sigdb$parent[
            signature_index
          ]
        ),
        source_original = as.character(
          sigdb$source_original[
            signature_index
          ]
        ),
        source_canonical = source_name,
        subsource = as.character(
          sigdb$subsource[
            signature_index
          ]
        ),
        type = as.character(
          sigdb$type[
            signature_index
          ]
        ),
        direction = as.character(
          sigdb$direction[
            signature_index
          ]
        ),
        signature_size_parsed = signature_size,
        signature_size_reported = as.integer(
          sigdb$ngene[
            signature_index
          ]
        ),
        source_universe_size = universe_size,
        program_genes_in_source_universe = (
          program_background_size
        ),
        overlap_genes = overlap,
        overlap_fraction_of_signature = if (
          signature_size > 0L
        ) {
          overlap /
            signature_size
        } else {
          NA_real_
        },
        overlap_fraction_of_program = if (
          program_background_size > 0L
        ) {
          overlap /
            program_background_size
        } else {
          NA_real_
        },
        fold_enrichment = fold_enrichment,
        hypergeometric_p = p_value,
        primary_size_eligible = (
          primary_size_eligible
        ),
        overlap_gene_symbols = paste(
          sort(
            overlapping_genes
          ),
          collapse = "|"
        ),
        description = as.character(
          sigdb$description[
            signature_index
          ]
        ),
        stringsAsFactors = FALSE
      )
    }
  }
}

reference_table <- do.call(
  rbind,
  reference_rows
)

reference_table$FDR_within_source_program_all <- (
  NA_real_
)

group_key <- interaction(
  reference_table$source_canonical,
  reference_table$developmental_program,
  drop = TRUE
)

for (group in levels(group_key)) {
  indices <- which(
    group_key == group
  )

  reference_table$
    FDR_within_source_program_all[
      indices
    ] <- p.adjust(
      reference_table$
        hypergeometric_p[
          indices
        ],
      method = "BH"
    )
}

reference_table$
  FDR_within_source_program_primary_size <- (
    NA_real_
  )

for (group in levels(group_key)) {
  indices <- which(
    group_key == group &
    reference_table$
      primary_size_eligible
  )

  if (length(indices) == 0L) {
    next
  }

  reference_table$
    FDR_within_source_program_primary_size[
      indices
    ] <- p.adjust(
      reference_table$
        hypergeometric_p[
          indices
        ],
      method = "BH"
    )
}

reference_table$primary_significant <- (
  reference_table$primary_size_eligible &
  !is.na(
    reference_table$
      FDR_within_source_program_primary_size
  ) &
  reference_table$
    FDR_within_source_program_primary_size <
    0.05 &
  reference_table$overlap_genes >= 3L &
  reference_table$fold_enrichment >= 1.5
)

reference_table$reference_role <- ifelse(
  reference_table$source_canonical ==
    "Random",
  "negative_control_reference",
  "mechanistic_reference"
)

reference_table <- reference_table[
  order(
    reference_table$
      developmental_program,
    reference_table$
      source_canonical,
    reference_table$
      FDR_within_source_program_primary_size,
    -reference_table$
      fold_enrichment,
    reference_table$signature
  ),
  ,
  drop = FALSE
]

write_gzip_table(
  reference_table,
  reference_output
)

summary_groups <- unique(
  reference_table[
    ,
    c(
      "developmental_program",
      "source_canonical",
      "reference_role"
    ),
    drop = FALSE
  ]
)

reference_summary_rows <- lapply(
  seq_len(
    nrow(summary_groups)
  ),
  function(index) {
    current_group <- summary_groups[
      index,
      ,
      drop = FALSE
    ]

    current <- reference_table[
      reference_table$
        developmental_program ==
        current_group$
          developmental_program &
      reference_table$
        source_canonical ==
        current_group$
          source_canonical,
      ,
      drop = FALSE
    ]

    data.frame(
      developmental_program = (
        current_group$
          developmental_program
      ),
      source_canonical = (
        current_group$
          source_canonical
      ),
      reference_role = (
        current_group$
          reference_role
      ),
      tested_signatures = nrow(current),
      primary_size_eligible_signatures = sum(
        current$
          primary_size_eligible
      ),
      FDR_lt_0_05_primary_size = sum(
        !is.na(
          current$
            FDR_within_source_program_primary_size
        ) &
        current$
          FDR_within_source_program_primary_size <
          0.05
      ),
      primary_significant_effect_filtered = sum(
        current$
          primary_significant
      ),
      minimum_primary_FDR = if (
        all(
          is.na(
            current$
              FDR_within_source_program_primary_size
          )
        )
      ) {
        NA_real_
      } else {
        min(
          current$
            FDR_within_source_program_primary_size,
          na.rm = TRUE
        )
      },
      maximum_fold_enrichment = max(
        current$fold_enrichment,
        na.rm = TRUE
      ),
      stringsAsFactors = FALSE
    )
  }
)

reference_summary <- do.call(
  rbind,
  reference_summary_rows
)

reference_summary <- reference_summary[
  order(
    reference_summary$
      developmental_program,
    -reference_summary$
      primary_significant_effect_filtered,
    reference_summary$
      source_canonical
  ),
  ,
  drop = FALSE
]

write.table(
  reference_summary,
  reference_summary_output,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

top_reference_rows <- list()

top_reference_index <- 0L

for (program_name in program_names) {
  for (source_name in reference_sources) {
    current <- reference_table[
      reference_table$
        developmental_program ==
        program_name &
      reference_table$
        source_canonical ==
        source_name &
      reference_table$
        primary_size_eligible,
      ,
      drop = FALSE
    ]

    if (nrow(current) == 0L) {
      next
    }

    current <- current[
      order(
        current$
          FDR_within_source_program_primary_size,
        -current$
          fold_enrichment,
        -current$
          overlap_genes,
        current$signature
      ),
      ,
      drop = FALSE
    ]

    current <- head(
      current,
      30L
    )

    current$within_source_rank <- seq_len(
      nrow(current)
    )

    top_reference_index <- (
      top_reference_index +
      1L
    )

    top_reference_rows[[top_reference_index]] <- current
  }
}

top_reference <- do.call(
  rbind,
  top_reference_rows
)

write.table(
  top_reference,
  reference_top_output,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

# ------------------------------------------------------------------
# Part B: paired CMAP directional concordance
# ------------------------------------------------------------------

log_message(
  "Scoring paired CMAP directional concordance..."
)

cmap_indices <- which(
  toupper(
    sigdb$source_original
  ) == "CMAP"
)

cmap_universe <- sort(
  unique(
    unlist(
      gene_lists[
        cmap_indices
      ],
      use.names = FALSE
    )
  )
)

cmap_universe_size <- length(
  cmap_universe
)

maturation_cmap <- intersect(
  maturation_genes,
  cmap_universe
)

fetal_cmap <- intersect(
  fetal_genes,
  cmap_universe
)

cmap_coverage <- data.frame(
  developmental_program = c(
    "maturation_high",
    "fetal_high"
  ),
  total_program_genes = c(
    length(maturation_genes),
    length(fetal_genes)
  ),
  genes_in_CMAP_universe = c(
    length(maturation_cmap),
    length(fetal_cmap)
  ),
  genes_absent_from_CMAP_universe = c(
    length(
      setdiff(
        maturation_genes,
        cmap_universe
      )
    ),
    length(
      setdiff(
        fetal_genes,
        cmap_universe
      )
    )
  ),
  CMAP_universe_size = cmap_universe_size,
  coverage_fraction = c(
    length(maturation_cmap) /
      length(maturation_genes),
    length(fetal_cmap) /
      length(fetal_genes)
  ),
  stringsAsFactors = FALSE
)

write.table(
  cmap_coverage,
  cmap_gene_coverage_output,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

cmap_parents <- sort(
  unique(
    as.character(
      sigdb$parent[
        cmap_indices
      ]
    )
  )
)

cmap_rows <- vector(
  "list",
  length(cmap_parents)
)

for (parent_index in seq_along(cmap_parents)) {
  parent_name <- cmap_parents[[parent_index]]

  parent_indices <- cmap_indices[
    as.character(
      sigdb$parent[
        cmap_indices
      ]
    ) == parent_name
  ]

  directions <- tolower(
    trimws(
      as.character(
        sigdb$direction[
          parent_indices
        ]
      )
    )
  )

  up_indices <- parent_indices[
    directions == "up"
  ]

  down_indices <- parent_indices[
    directions == "dn"
  ]

  if (
    length(up_indices) != 1L ||
    length(down_indices) != 1L
  ) {
    stop(
      paste(
        "Invalid CMAP pair:",
        parent_name,
        "up rows=",
        length(up_indices),
        "down rows=",
        length(down_indices)
      )
    )
  }

  up_index <- up_indices[[1L]]
  down_index <- down_indices[[1L]]

  up_genes <- intersect(
    gene_lists[[up_index]],
    cmap_universe
  )

  down_genes <- intersect(
    gene_lists[[down_index]],
    cmap_universe
  )

  up_size <- length(
    up_genes
  )

  down_size <- length(
    down_genes
  )

  up_maturation_genes <- intersect(
    up_genes,
    maturation_cmap
  )

  down_maturation_genes <- intersect(
    down_genes,
    maturation_cmap
  )

  up_fetal_genes <- intersect(
    up_genes,
    fetal_cmap
  )

  down_fetal_genes <- intersect(
    down_genes,
    fetal_cmap
  )

  up_maturation <- length(
    up_maturation_genes
  )

  down_maturation <- length(
    down_maturation_genes
  )

  up_fetal <- length(
    up_fetal_genes
  )

  down_fetal <- length(
    down_fetal_genes
  )

  p_maturation_up <- one_sided_preference_p(
    preferred_hits = up_maturation,
    preferred_size = up_size,
    comparison_hits = down_maturation,
    comparison_size = down_size
  )

  p_fetal_down <- one_sided_preference_p(
    preferred_hits = down_fetal,
    preferred_size = down_size,
    comparison_hits = up_fetal,
    comparison_size = up_size
  )

  p_fetal_up <- one_sided_preference_p(
    preferred_hits = up_fetal,
    preferred_size = up_size,
    comparison_hits = down_fetal,
    comparison_size = down_size
  )

  p_maturation_down <- one_sided_preference_p(
    preferred_hits = down_maturation,
    preferred_size = down_size,
    comparison_hits = up_maturation,
    comparison_size = up_size
  )

  maturation_log2_or <- safe_log2_odds_ratio(
    first_hits = up_maturation,
    first_nonhits = up_size -
      up_maturation,
    second_hits = down_maturation,
    second_nonhits = down_size -
      down_maturation
  )

  fetal_down_log2_or <- safe_log2_odds_ratio(
    first_hits = down_fetal,
    first_nonhits = down_size -
      down_fetal,
    second_hits = up_fetal,
    second_nonhits = up_size -
      up_fetal
  )

  developmental_direction_score <- mean(
    c(
      maturation_log2_or,
      fetal_down_log2_or
    )
  )

  maturation_like_p <- combine_two_pvalues(
    p_maturation_up,
    p_fetal_down
  )

  fetal_like_p <- combine_two_pvalues(
    p_fetal_up,
    p_maturation_down
  )

  description_values <- unique(
    as.character(
      sigdb$description[
        parent_indices
      ]
    )
  )

  description_values <- description_values[
    !is.na(description_values) &
    description_values != ""
  ]

  description <- if (
    length(description_values) == 0L
  ) {
    ""
  } else {
    description_values[[1L]]
  }

  cmap_rows[[parent_index]] <- data.frame(
    CMAP_parent = parent_name,
    chemical_name = extract_chemical_name(
      description
    ),
    DTXSID = extract_dtxsid(
      description
    ),
    description = description,
    up_signature = as.character(
      sigdb$signature[
        up_index
      ]
    ),
    down_signature = as.character(
      sigdb$signature[
        down_index
      ]
    ),
    up_signature_size = up_size,
    down_signature_size = down_size,
    up_down_shared_genes = length(
      intersect(
        up_genes,
        down_genes
      )
    ),
    maturation_genes_up = up_maturation,
    maturation_genes_down = (
      down_maturation
    ),
    fetal_genes_up = up_fetal,
    fetal_genes_down = down_fetal,
    maturation_up_preference_log2OR = (
      maturation_log2_or
    ),
    fetal_down_preference_log2OR = (
      fetal_down_log2_or
    ),
    developmental_direction_score = (
      developmental_direction_score
    ),
    maturation_up_preference_p = (
      p_maturation_up
    ),
    fetal_down_preference_p = (
      p_fetal_down
    ),
    fetal_up_preference_p = (
      p_fetal_up
    ),
    maturation_down_preference_p = (
      p_maturation_down
    ),
    maturation_like_combined_p = (
      maturation_like_p
    ),
    fetal_like_combined_p = fetal_like_p,
    maturation_up_gene_symbols = paste(
      sort(
        up_maturation_genes
      ),
      collapse = "|"
    ),
    maturation_down_gene_symbols = paste(
      sort(
        down_maturation_genes
      ),
      collapse = "|"
    ),
    fetal_up_gene_symbols = paste(
      sort(
        up_fetal_genes
      ),
      collapse = "|"
    ),
    fetal_down_gene_symbols = paste(
      sort(
        down_fetal_genes
      ),
      collapse = "|"
    ),
    stringsAsFactors = FALSE
  )
}

cmap_table <- do.call(
  rbind,
  cmap_rows
)

cmap_table$maturation_like_FDR <- p.adjust(
  cmap_table$maturation_like_combined_p,
  method = "BH"
)

cmap_table$fetal_like_FDR <- p.adjust(
  cmap_table$fetal_like_combined_p,
  method = "BH"
)

cmap_table$directional_state <- ifelse(
  cmap_table$maturation_like_FDR < 0.05 &
  cmap_table$fetal_like_FDR >= 0.05 &
  cmap_table$developmental_direction_score > 0,
  "maturation_like",
  ifelse(
    cmap_table$fetal_like_FDR < 0.05 &
    cmap_table$maturation_like_FDR >= 0.05 &
    cmap_table$developmental_direction_score < 0,
    "fetal_like",
    ifelse(
      cmap_table$maturation_like_FDR < 0.05 &
      cmap_table$fetal_like_FDR < 0.05,
      "bidirectional_or_ambiguous",
      "not_FDR_significant"
    )
  )
)

cmap_table$absolute_direction_score <- abs(
  cmap_table$developmental_direction_score
)

cmap_table <- cmap_table[
  order(
    pmin(
      cmap_table$maturation_like_FDR,
      cmap_table$fetal_like_FDR
    ),
    -cmap_table$absolute_direction_score,
    cmap_table$CMAP_parent
  ),
  ,
  drop = FALSE
]

write_gzip_table(
  cmap_table,
  cmap_output
)

maturation_rank <- cmap_table[
  order(
    cmap_table$maturation_like_FDR,
    -cmap_table$developmental_direction_score,
    cmap_table$CMAP_parent
  ),
  ,
  drop = FALSE
]

maturation_rank <- head(
  maturation_rank,
  150L
)

maturation_rank$ranked_state <- (
  "maturation_like"
)

maturation_rank$state_rank <- seq_len(
  nrow(maturation_rank)
)

fetal_rank <- cmap_table[
  order(
    cmap_table$fetal_like_FDR,
    cmap_table$developmental_direction_score,
    cmap_table$CMAP_parent
  ),
  ,
  drop = FALSE
]

fetal_rank <- head(
  fetal_rank,
  150L
)

fetal_rank$ranked_state <- (
  "fetal_like"
)

fetal_rank$state_rank <- seq_len(
  nrow(fetal_rank)
)

top_cmap <- rbind(
  maturation_rank,
  fetal_rank
)

top_cmap <- top_cmap[
  ,
  c(
    "ranked_state",
    "state_rank",
    setdiff(
      colnames(top_cmap),
      c(
        "ranked_state",
        "state_rank"
      )
    )
  ),
  drop = FALSE
]

write.table(
  top_cmap,
  cmap_top_output,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

cmap_summary <- data.frame(
  CMAP_parents_tested = nrow(
    cmap_table
  ),
  CMAP_universe_genes = cmap_universe_size,
  maturation_program_genes_in_CMAP = length(
    maturation_cmap
  ),
  fetal_program_genes_in_CMAP = length(
    fetal_cmap
  ),
  parents_with_up_down_gene_overlap = sum(
    cmap_table$up_down_shared_genes > 0L
  ),
  maturation_like_FDR_lt_0_05 = sum(
    cmap_table$maturation_like_FDR <
      0.05
  ),
  fetal_like_FDR_lt_0_05 = sum(
    cmap_table$fetal_like_FDR <
      0.05
  ),
  classified_maturation_like = sum(
    cmap_table$directional_state ==
      "maturation_like"
  ),
  classified_fetal_like = sum(
    cmap_table$directional_state ==
      "fetal_like"
  ),
  classified_bidirectional_or_ambiguous = sum(
    cmap_table$directional_state ==
      "bidirectional_or_ambiguous"
  ),
  classified_not_FDR_significant = sum(
    cmap_table$directional_state ==
      "not_FDR_significant"
  ),
  maximum_direction_score = max(
    cmap_table$
      developmental_direction_score
  ),
  minimum_direction_score = min(
    cmap_table$
      developmental_direction_score
  ),
  stringsAsFactors = FALSE
)

write.table(
  cmap_summary,
  cmap_summary_output,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

random_reference <- reference_summary[
  reference_summary$source_canonical ==
    "Random",
  ,
  drop = FALSE
]

completion <- data.frame(
  developmental_programs = 2L,
  non_CMAP_signature_rows = length(
    reference_indices
  ),
  reference_program_tests = nrow(
    reference_table
  ),
  reference_primary_significant_tests = sum(
    reference_table$primary_significant
  ),
  random_negative_control_tests = sum(
    reference_table$source_canonical ==
      "Random"
  ),
  random_negative_control_primary_significant = sum(
    reference_table$source_canonical ==
      "Random" &
    reference_table$primary_significant
  ),
  CMAP_signature_rows = length(
    cmap_indices
  ),
  CMAP_parents_tested = nrow(
    cmap_table
  ),
  CMAP_maturation_like_FDR_lt_0_05 = sum(
    cmap_table$maturation_like_FDR <
      0.05
  ),
  CMAP_fetal_like_FDR_lt_0_05 = sum(
    cmap_table$fetal_like_FDR <
      0.05
  ),
  source_specific_backgrounds_used = TRUE,
  parsed_unique_gene_counts_used = TRUE,
  full_ranked_expression_profiles_available = FALSE,
  Phase6C1_status = "completed",
  stringsAsFactors = FALSE
)

write.table(
  completion,
  completion_output,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

log_message("")
log_message(
  "===== REFERENCE ENRICHMENT SUMMARY ====="
)

capture <- capture.output(
  print(
    reference_summary,
    row.names = FALSE
  )
)

for (line in capture) {
  log_message(line)
}

log_message("")
log_message(
  "===== CMAP PROGRAM COVERAGE ====="
)

capture <- capture.output(
  print(
    cmap_coverage,
    row.names = FALSE
  )
)

for (line in capture) {
  log_message(line)
}

log_message("")
log_message(
  "===== CMAP DIRECTIONAL SUMMARY ====="
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
  "===== PHASE 6C1 COMPLETION ====="
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
