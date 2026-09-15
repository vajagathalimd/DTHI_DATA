#!/usr/bin/env Rscript

options(
  stringsAsFactors = FALSE,
  width = 240,
  scipen = 999
)

suppressPackageStartupMessages(
  library(Matrix)
)

project <- "."

signature_file <- file.path(
  project,
  "01_raw_data/perturbational_validation/EPA_HTTr",
  "HTTr Signatures.RData"
)

observed_file <- file.path(
  project,
  "03_processed_data/perturbational_validation/phase6C",
  "phase6C1_CMAP_directional_concordance.tsv.gz"
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
  "phase6C2B_frequency_matched_empirical_null.log"
)

result_file <- file.path(
  processed_dir,
  "phase6C2B_CMAP_empirical_null_results.tsv.gz"
)

top_file <- file.path(
  table_dir,
  "phase6C2B_top_empirically_calibrated_CMAP_candidates.tsv"
)

strata_file <- file.path(
  table_dir,
  "phase6C2B_frequency_strata_matching.tsv"
)

null_threshold_file <- file.path(
  table_dir,
  "phase6C2B_null_direction_score_thresholds.tsv"
)

null_count_file <- file.path(
  table_dir,
  "phase6C2B_null_discovery_count_summary.tsv"
)

summary_file <- file.path(
  table_dir,
  "phase6C2B_empirical_null_summary.tsv"
)

completion_file <- file.path(
  table_dir,
  "phase6C2B_completion_summary.tsv"
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

number_of_permutations <- 5000L
batch_size <- 100L
number_of_strata <- 10L
random_seed <- 20260720L

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

  candidates <- c(
    "gene_symbol",
    "gene",
    "symbol"
  )

  matched <- intersect(
    candidates,
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

parse_gene_lists <- function(values) {
  lists <- strsplit(
    as.character(values),
    split = "|",
    fixed = TRUE
  )

  lapply(
    lists,
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
}

build_sparse_incidence <- function(
  gene_lists,
  universe
) {
  lengths_vector <- lengths(
    gene_lists
  )

  gene_vector <- unlist(
    gene_lists,
    use.names = FALSE
  )

  gene_indices <- match(
    gene_vector,
    universe
  )

  signature_indices <- rep(
    seq_along(gene_lists),
    times = lengths_vector
  )

  valid <- !is.na(
    gene_indices
  )

  sparseMatrix(
    i = gene_indices[valid],
    j = signature_indices[valid],
    x = 1,
    dims = c(
      length(universe),
      length(gene_lists)
    ),
    giveCsparse = TRUE
  )
}

corrected_log2_odds <- function(
  preferred_hits,
  preferred_size,
  comparison_hits,
  comparison_size
) {
  log2(
    (
      (preferred_hits + 0.5) /
      (preferred_size - preferred_hits + 0.5)
    ) /
    (
      (comparison_hits + 0.5) /
      (comparison_size - comparison_hits + 0.5)
    )
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
  "===== Phase 6C2B started ====="
)

log_message(
  "Permutations: ",
  number_of_permutations
)

log_message(
  "Batch size: ",
  batch_size
)

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

cmap <- sigdb[
  toupper(
    as.character(
      sigdb$source
    )
  ) == "CMAP",
  ,
  drop = FALSE
]

cmap$direction <- tolower(
  trimws(
    as.character(
      cmap$direction
    )
  )
)

up_table <- cmap[
  cmap$direction == "up",
  ,
  drop = FALSE
]

down_table <- cmap[
  cmap$direction == "dn",
  ,
  drop = FALSE
]

up_table <- up_table[
  order(
    up_table$parent
  ),
  ,
  drop = FALSE
]

down_table <- down_table[
  order(
    down_table$parent
  ),
  ,
  drop = FALSE
]

if (
  nrow(up_table) != 9273L ||
  nrow(down_table) != 9273L
) {
  stop(
    "Unexpected number of CMAP up/down signatures."
  )
}

if (
  !identical(
    as.character(
      up_table$parent
    ),
    as.character(
      down_table$parent
    )
  )
) {
  stop(
    "CMAP up/down parent ordering is inconsistent."
  )
}

parents <- as.character(
  up_table$parent
)

up_gene_lists <- parse_gene_lists(
  up_table$gene.list
)

down_gene_lists <- parse_gene_lists(
  down_table$gene.list
)

cmap_universe <- sort(
  unique(
    c(
      unlist(
        up_gene_lists,
        use.names = FALSE
      ),
      unlist(
        down_gene_lists,
        use.names = FALSE
      )
    )
  )
)

if (
  length(cmap_universe) != 13470L
) {
  stop(
    paste(
      "Unexpected CMAP universe size:",
      length(cmap_universe)
    )
  )
}

log_message(
  "CMAP universe genes: ",
  length(cmap_universe)
)

log_message(
  "Building sparse up/down incidence matrices..."
)

up_matrix <- build_sparse_incidence(
  up_gene_lists,
  cmap_universe
)

down_matrix <- build_sparse_incidence(
  down_gene_lists,
  cmap_universe
)

up_sizes <- lengths(
  up_gene_lists
)

down_sizes <- lengths(
  down_gene_lists
)

gene_frequency <- as.numeric(
  rowSums(
    up_matrix
  ) +
  rowSums(
    down_matrix
  )
)

frequency_order <- order(
  gene_frequency,
  cmap_universe
)

frequency_stratum <- integer(
  length(cmap_universe)
)

frequency_stratum[
  frequency_order
] <- ceiling(
  seq_along(
    frequency_order
  ) /
  length(
    frequency_order
  ) *
  number_of_strata
)

frequency_stratum[
  frequency_stratum < 1L
] <- 1L

frequency_stratum[
  frequency_stratum >
  number_of_strata
] <- number_of_strata

maturation_genes <- intersect(
  read_program_genes(
    maturation_file
  ),
  cmap_universe
)

fetal_genes <- intersect(
  read_program_genes(
    fetal_file
  ),
  cmap_universe
)

if (
  length(maturation_genes) != 1863L ||
  length(fetal_genes) != 2829L
) {
  stop(
    paste(
      "Unexpected CMAP program sizes:",
      length(maturation_genes),
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
    "CMAP developmental programs are not disjoint."
  )
}

maturation_indices <- match(
  maturation_genes,
  cmap_universe
)

fetal_indices <- match(
  fetal_genes,
  cmap_universe
)

stratum_indices <- split(
  seq_along(
    cmap_universe
  ),
  frequency_stratum
)

maturation_stratum_counts <- tabulate(
  frequency_stratum[
    maturation_indices
  ],
  nbins = number_of_strata
)

fetal_stratum_counts <- tabulate(
  frequency_stratum[
    fetal_indices
  ],
  nbins = number_of_strata
)

strata_table <- data.frame(
  frequency_stratum = seq_len(
    number_of_strata
  ),
  genes_in_stratum = vapply(
    stratum_indices,
    length,
    integer(1)
  ),
  minimum_CMAP_frequency = vapply(
    stratum_indices,
    function(indices) {
      min(
        gene_frequency[
          indices
        ]
      )
    },
    numeric(1)
  ),
  median_CMAP_frequency = vapply(
    stratum_indices,
    function(indices) {
      median(
        gene_frequency[
          indices
        ]
      )
    },
    numeric(1)
  ),
  maximum_CMAP_frequency = vapply(
    stratum_indices,
    function(indices) {
      max(
        gene_frequency[
          indices
        ]
      )
    },
    numeric(1)
  ),
  maturation_genes = (
    maturation_stratum_counts
  ),
  fetal_genes = fetal_stratum_counts,
  combined_program_genes = (
    maturation_stratum_counts +
    fetal_stratum_counts
  ),
  available_after_combined_sampling = (
    vapply(
      stratum_indices,
      length,
      integer(1)
    ) -
    maturation_stratum_counts -
    fetal_stratum_counts
  ),
  stringsAsFactors = FALSE
)

if (
  any(
    strata_table$
      available_after_combined_sampling <
      0L
  )
) {
  stop(
    "Frequency-stratum program matching is infeasible."
  )
}

write.table(
  strata_table,
  strata_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

observed <- read.delim(
  gzfile(
    observed_file,
    open = "rt"
  ),
  check.names = FALSE,
  stringsAsFactors = FALSE
)

observed <- observed[
  match(
    parents,
    observed$CMAP_parent
  ),
  ,
  drop = FALSE
]

if (
  any(
    is.na(
      observed$CMAP_parent
    )
  )
) {
  stop(
    "Observed CMAP table could not be aligned to all parents."
  )
}

observed_score <- as.numeric(
  observed$developmental_direction_score
)

observed_maturation_component <- as.numeric(
  observed$maturation_up_preference_log2OR
)

observed_fetal_component <- as.numeric(
  observed$fetal_down_preference_log2OR
)

maturation_indicator <- numeric(
  length(cmap_universe)
)

fetal_indicator <- numeric(
  length(cmap_universe)
)

maturation_indicator[
  maturation_indices
] <- 1

fetal_indicator[
  fetal_indices
] <- 1

observed_maturation_up <- as.numeric(
  crossprod(
    up_matrix,
    maturation_indicator
  )
)

observed_maturation_down <- as.numeric(
  crossprod(
    down_matrix,
    maturation_indicator
  )
)

observed_fetal_up <- as.numeric(
  crossprod(
    up_matrix,
    fetal_indicator
  )
)

observed_fetal_down <- as.numeric(
  crossprod(
    down_matrix,
    fetal_indicator
  )
)

recalculated_maturation_component <- (
  corrected_log2_odds(
    observed_maturation_up,
    up_sizes,
    observed_maturation_down,
    down_sizes
  )
)

recalculated_fetal_component <- (
  corrected_log2_odds(
    observed_fetal_down,
    down_sizes,
    observed_fetal_up,
    up_sizes
  )
)

recalculated_score <- (
  recalculated_maturation_component +
  recalculated_fetal_component
) / 2

maximum_score_difference <- max(
  abs(
    recalculated_score -
    observed_score
  )
)

if (
  maximum_score_difference >
  1e-8
) {
  stop(
    paste(
      "Observed-score reconstruction failed; maximum difference:",
      maximum_score_difference
    )
  )
}

log_message(
  "Observed score reconstruction: VALID"
)

set.seed(
  random_seed
)

positive_exceedances <- integer(
  length(parents)
)

negative_exceedances <- integer(
  length(parents)
)

null_maximum_scores <- numeric(
  number_of_permutations
)

null_minimum_scores <- numeric(
  number_of_permutations
)

thresholds <- c(
  0.25,
  0.50,
  0.75,
  1.00,
  1.25,
  1.50,
  2.00
)

null_positive_counts <- matrix(
  0L,
  nrow = number_of_permutations,
  ncol = length(thresholds)
)

null_negative_counts <- matrix(
  0L,
  nrow = number_of_permutations,
  ncol = length(thresholds)
)

colnames(
  null_positive_counts
) <- paste0(
  "score_ge_",
  thresholds
)

colnames(
  null_negative_counts
) <- paste0(
  "score_le_minus_",
  thresholds
)

completed_permutations <- 0L

while (
  completed_permutations <
  number_of_permutations
) {
  current_batch <- min(
    batch_size,
    number_of_permutations -
      completed_permutations
  )

  maturation_random <- matrix(
    0,
    nrow = length(cmap_universe),
    ncol = current_batch
  )

  fetal_random <- matrix(
    0,
    nrow = length(cmap_universe),
    ncol = current_batch
  )

  for (
    batch_column in seq_len(
      current_batch
    )
  ) {
    for (
      stratum in seq_len(
        number_of_strata
      )
    ) {
      pool <- stratum_indices[[as.character(stratum)]]

      maturation_count <- (
        maturation_stratum_counts[
          stratum
        ]
      )

      fetal_count <- (
        fetal_stratum_counts[
          stratum
        ]
      )

      maturation_sample <- if (
        maturation_count > 0L
      ) {
        sample(
          pool,
          size = maturation_count,
          replace = FALSE
        )
      } else {
        integer()
      }

      remaining_pool <- pool[
        !pool %in%
        maturation_sample
      ]

      fetal_sample <- if (
        fetal_count > 0L
      ) {
        sample(
          remaining_pool,
          size = fetal_count,
          replace = FALSE
        )
      } else {
        integer()
      }

      maturation_random[
        maturation_sample,
        batch_column
      ] <- 1

      fetal_random[
        fetal_sample,
        batch_column
      ] <- 1
    }
  }

  maturation_up <- as.matrix(
    crossprod(
      up_matrix,
      maturation_random
    )
  )

  maturation_down <- as.matrix(
    crossprod(
      down_matrix,
      maturation_random
    )
  )

  fetal_up <- as.matrix(
    crossprod(
      up_matrix,
      fetal_random
    )
  )

  fetal_down <- as.matrix(
    crossprod(
      down_matrix,
      fetal_random
    )
  )

  maturation_component <- (
    corrected_log2_odds(
      maturation_up,
      up_sizes,
      maturation_down,
      down_sizes
    )
  )

  fetal_component <- (
    corrected_log2_odds(
      fetal_down,
      down_sizes,
      fetal_up,
      up_sizes
    )
  )

  null_score <- (
    maturation_component +
    fetal_component
  ) / 2

  positive_exceedances <- (
    positive_exceedances +
    rowSums(
      null_score >= observed_score
    )
  )

  negative_exceedances <- (
    negative_exceedances +
    rowSums(
      null_score <= observed_score
    )
  )

  permutation_indices <- (
    completed_permutations +
    seq_len(
      current_batch
    )
  )

  null_maximum_scores[
    permutation_indices
  ] <- apply(
    null_score,
    2,
    max
  )

  null_minimum_scores[
    permutation_indices
  ] <- apply(
    null_score,
    2,
    min
  )

  for (
    threshold_index in seq_along(
      thresholds
    )
  ) {
    threshold <- thresholds[
      threshold_index
    ]

    null_positive_counts[
      permutation_indices,
      threshold_index
    ] <- colSums(
      null_score >= threshold
    )

    null_negative_counts[
      permutation_indices,
      threshold_index
    ] <- colSums(
      null_score <= -threshold
    )
  }

  completed_permutations <- (
    completed_permutations +
    current_batch
  )

  if (
    completed_permutations %% 500L == 0L ||
    completed_permutations ==
    number_of_permutations
  ) {
    log_message(
      "Completed permutations: ",
      completed_permutations,
      "/",
      number_of_permutations
    )
  }
}

empirical_maturation_p <- (
  positive_exceedances + 1
) / (
  number_of_permutations + 1
)

empirical_fetal_p <- (
  negative_exceedances + 1
) / (
  number_of_permutations + 1
)

empirical_maturation_FDR <- p.adjust(
  empirical_maturation_p,
  method = "BH"
)

empirical_fetal_FDR <- p.adjust(
  empirical_fetal_p,
  method = "BH"
)

maxT_maturation_p <- vapply(
  observed_score,
  function(score) {
    (
      1 +
      sum(
        null_maximum_scores >= score
      )
    ) /
    (
      number_of_permutations + 1
    )
  },
  numeric(1)
)

maxT_fetal_p <- vapply(
  observed_score,
  function(score) {
    (
      1 +
      sum(
        null_minimum_scores <= score
      )
    ) /
    (
      number_of_permutations + 1
    )
  },
  numeric(1)
)

component_maturation_consistent <- (
  observed_maturation_component > 0 &
  observed_fetal_component > 0
)

component_fetal_consistent <- (
  observed_maturation_component < 0 &
  observed_fetal_component < 0
)

empirical_state <- ifelse(
  observed_score > 0 &
  component_maturation_consistent &
  empirical_maturation_FDR < 0.05,
  "maturation_like",
  ifelse(
    observed_score < 0 &
    component_fetal_consistent &
    empirical_fetal_FDR < 0.05,
    "fetal_like",
    "not_empirically_significant"
  )
)

result <- observed

result$frequency_matched_permutations <- (
  number_of_permutations
)

result$empirical_maturation_p <- (
  empirical_maturation_p
)

result$empirical_maturation_FDR <- (
  empirical_maturation_FDR
)

result$empirical_fetal_p <- (
  empirical_fetal_p
)

result$empirical_fetal_FDR <- (
  empirical_fetal_FDR
)

result$maxT_maturation_FWER_p <- (
  maxT_maturation_p
)

result$maxT_fetal_FWER_p <- (
  maxT_fetal_p
)

result$maturation_component_consistent <- (
  component_maturation_consistent
)

result$fetal_component_consistent <- (
  component_fetal_consistent
)

result$empirical_directional_state <- (
  empirical_state
)

result$minimum_empirical_p <- pmin(
  empirical_maturation_p,
  empirical_fetal_p
)

result$minimum_empirical_FDR <- pmin(
  empirical_maturation_FDR,
  empirical_fetal_FDR
)

result <- result[
  order(
    result$minimum_empirical_FDR,
    -abs(
      result$developmental_direction_score
    ),
    result$CMAP_parent
  ),
  ,
  drop = FALSE
]

write_gzip_table(
  result,
  result_file
)

maturation_top <- result[
  result$empirical_directional_state ==
    "maturation_like",
  ,
  drop = FALSE
]

maturation_top <- maturation_top[
  order(
    maturation_top$
      empirical_maturation_FDR,
    maturation_top$
      maxT_maturation_FWER_p,
    -maturation_top$
      developmental_direction_score,
    maturation_top$CMAP_parent
  ),
  ,
  drop = FALSE
]

maturation_top <- head(
  maturation_top,
  100L
)

maturation_top$empirical_rank <- seq_len(
  nrow(maturation_top)
)

fetal_top <- result[
  result$empirical_directional_state ==
    "fetal_like",
  ,
  drop = FALSE
]

fetal_top <- fetal_top[
  order(
    fetal_top$
      empirical_fetal_FDR,
    fetal_top$
      maxT_fetal_FWER_p,
    fetal_top$
      developmental_direction_score,
    fetal_top$CMAP_parent
  ),
  ,
  drop = FALSE
]

fetal_top <- head(
  fetal_top,
  100L
)

fetal_top$empirical_rank <- seq_len(
  nrow(fetal_top)
)

top_table <- rbind(
  maturation_top,
  fetal_top
)

write.table(
  top_table,
  top_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

quantile_probabilities <- c(
  0,
  0.001,
  0.005,
  0.01,
  0.025,
  0.05,
  0.50,
  0.95,
  0.975,
  0.99,
  0.995,
  0.999,
  1
)

threshold_table <- rbind(
  data.frame(
    null_distribution = (
      "permutation_maximum_score"
    ),
    quantile = quantile_probabilities,
    score = as.numeric(
      quantile(
        null_maximum_scores,
        probs = quantile_probabilities
      )
    ),
    stringsAsFactors = FALSE
  ),
  data.frame(
    null_distribution = (
      "permutation_minimum_score"
    ),
    quantile = quantile_probabilities,
    score = as.numeric(
      quantile(
        null_minimum_scores,
        probs = quantile_probabilities
      )
    ),
    stringsAsFactors = FALSE
  )
)

write.table(
  threshold_table,
  null_threshold_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

count_rows <- list()

count_index <- 0L

for (
  threshold_index in seq_along(
    thresholds
  )
) {
  threshold <- thresholds[
    threshold_index
  ]

  for (
    direction_name in c(
      "positive",
      "negative"
    )
  ) {
    values <- if (
      direction_name == "positive"
    ) {
      null_positive_counts[
        ,
        threshold_index
      ]
    } else {
      null_negative_counts[
        ,
        threshold_index
      ]
    }

    count_index <- count_index + 1L

    count_rows[[count_index]] <- data.frame(
      direction = direction_name,
      absolute_score_threshold = threshold,
      minimum_null_count = min(values),
      median_null_count = median(values),
      mean_null_count = mean(values),
      percentile_95_null_count = as.numeric(
        quantile(
          values,
          0.95
        )
      ),
      percentile_99_null_count = as.numeric(
        quantile(
          values,
          0.99
        )
      ),
      maximum_null_count = max(values),
      observed_count = if (
        direction_name == "positive"
      ) {
        sum(
          observed_score >= threshold
        )
      } else {
        sum(
          observed_score <= -threshold
        )
      },
      stringsAsFactors = FALSE
    )
  }
}

null_count_summary <- do.call(
  rbind,
  count_rows
)

write.table(
  null_count_summary,
  null_count_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

summary <- data.frame(
  CMAP_parents_tested = length(
    parents
  ),
  frequency_matched_permutations = (
    number_of_permutations
  ),
  frequency_strata = number_of_strata,
  maturation_program_genes = length(
    maturation_genes
  ),
  fetal_program_genes = length(
    fetal_genes
  ),
  maximum_observed_score = max(
    observed_score
  ),
  minimum_observed_score = min(
    observed_score
  ),
  empirical_maturation_FDR_lt_0_05 = sum(
    empirical_maturation_FDR < 0.05 &
    observed_score > 0
  ),
  empirical_fetal_FDR_lt_0_05 = sum(
    empirical_fetal_FDR < 0.05 &
    observed_score < 0
  ),
  classified_maturation_like = sum(
    empirical_state ==
      "maturation_like"
  ),
  classified_fetal_like = sum(
    empirical_state ==
      "fetal_like"
  ),
  maxT_maturation_FWER_lt_0_05 = sum(
    maxT_maturation_p < 0.05 &
    observed_score > 0
  ),
  maxT_fetal_FWER_lt_0_05 = sum(
    maxT_fetal_p < 0.05 &
    observed_score < 0
  ),
  maximum_score_reconstruction_difference = (
    maximum_score_difference
  ),
  stringsAsFactors = FALSE
)

write.table(
  summary,
  summary_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

completion <- data.frame(
  permutations_requested = (
    number_of_permutations
  ),
  permutations_completed = (
    completed_permutations
  ),
  CMAP_parents_tested = length(
    parents
  ),
  exact_program_sizes_preserved = TRUE,
  program_disjointness_preserved = TRUE,
  CMAP_frequency_strata_preserved = TRUE,
  empirical_FDR_calculated = TRUE,
  maxT_FWER_calculated = TRUE,
  Phase6C2B_status = "completed",
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
  "===== EMPIRICAL NULL SUMMARY ====="
)

capture <- capture.output(
  print(
    summary,
    row.names = FALSE
  )
)

for (line in capture) {
  log_message(line)
}

log_message("")
log_message(
  "===== PHASE 6C2B COMPLETION ====="
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
