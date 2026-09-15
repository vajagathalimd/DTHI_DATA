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

category_union_file <- file.path(
  project,
  "03_processed_data/perturbational_validation/phase6C",
  "phase6C3C_mechanism_category_gene_unions.tsv.gz"
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

original_score_file <- file.path(
  project,
  "03_processed_data/perturbational_validation/phase6C",
  "phase6C3D_CMAP_parent_mechanism_scores.tsv.gz"
)

maxT_file <- file.path(
  project,
  "07_tables/main_tables/phase6",
  "phase6C3D_maxT_maturation_parent_mechanism_summary.tsv"
)

processed_dir <- file.path(
  project,
  "03_processed_data/perturbational_validation/phase6D"
)

table_dir <- file.path(
  project,
  "07_tables/main_tables/phase6"
)

log_file <- file.path(
  project,
  "09_pipeline_logs/phase6",
  "phase6D1_leave_program_out_robustness.log"
)

score_output <- file.path(
  processed_dir,
  "phase6D1_leave_program_out_parent_category_scores.tsv.gz"
)

comparison_output <- file.path(
  processed_dir,
  "phase6D1_original_vs_leave_program_out_scores.tsv.gz"
)

coverage_output <- file.path(
  table_dir,
  "phase6D1_leave_program_out_category_coverage.tsv"
)

category_summary_output <- file.path(
  table_dir,
  "phase6D1_leave_program_out_category_robustness.tsv"
)

maxT_summary_output <- file.path(
  table_dir,
  "phase6D1_maxT_parent_category_robustness.tsv"
)

completion_output <- file.path(
  table_dir,
  "phase6D1_completion_summary.tsv"
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

  table <- read.delim(
    connection,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )

  close(
    connection
  )

  candidate_columns <- intersect(
    c(
      "gene_symbol",
      "gene",
      "symbol"
    ),
    colnames(table)
  )

  gene_column <- if (
    length(candidate_columns) > 0L
  ) {
    candidate_columns[1L]
  } else {
    colnames(table)[1L]
  }

  genes <- unique(
    normalize_gene(
      table[, gene_column]
    )
  )

  genes[
    genes != ""
  ]
}

parse_gene_lists <- function(values) {
  raw_lists <- strsplit(
    as.character(values),
    split = "|",
    fixed = TRUE
  )

  lapply(
    raw_lists,
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

  write.table(
    table,
    connection,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE,
    na = ""
  )

  close(
    connection
  )
}

safe_spearman <- function(
  first,
  second
) {
  valid <- is.finite(first) &
    is.finite(second)

  if (
    sum(valid) < 3L ||
    length(
      unique(
        first[valid]
      )
    ) < 2L ||
    length(
      unique(
        second[valid]
      )
    ) < 2L
  ) {
    return(
      NA_real_
    )
  }

  suppressWarnings(
    cor(
      first[valid],
      second[valid],
      method = "spearman"
    )
  )
}

log_message(
  "===== Phase 6D1 started ====="
)

maturation_genes <- read_program_genes(
  maturation_file
)

fetal_genes <- read_program_genes(
  fetal_file
)

program_union <- union(
  maturation_genes,
  fetal_genes
)

if (
  length(
    intersect(
      maturation_genes,
      fetal_genes
    )
  ) != 0L
) {
  stop(
    "Maturation and fetal programs are unexpectedly overlapping."
  )
}

log_message(
  "Maturation program genes: ",
  length(maturation_genes)
)

log_message(
  "Fetal program genes: ",
  length(fetal_genes)
)

log_message(
  "Combined program genes removed: ",
  length(program_union)
)

category_connection <- gzfile(
  category_union_file,
  open = "rt"
)

category_unions <- read.delim(
  category_connection,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

close(
  category_connection
)

required_category_columns <- c(
  "mechanism_category",
  "axis_role",
  "expected_CMAP_direction",
  "gene_symbol"
)

missing_category_columns <- setdiff(
  required_category_columns,
  colnames(category_unions)
)

if (
  length(missing_category_columns) > 0L
) {
  stop(
    paste(
      "Missing category columns:",
      paste(
        missing_category_columns,
        collapse = ", "
      )
    )
  )
}

category_unions$gene_symbol <- normalize_gene(
  category_unions$gene_symbol
)

category_unions <- category_unions[
  category_unions$gene_symbol != "",
  ,
  drop = FALSE
]

category_meta <- unique(
  category_unions[
    ,
    c(
      "mechanism_category",
      "axis_role",
      "expected_CMAP_direction"
    ),
    drop = FALSE
  ]
)

category_meta <- category_meta[
  order(
    category_meta$mechanism_category
  ),
  ,
  drop = FALSE
]

if (
  nrow(category_meta) != 9L
) {
  stop(
    paste(
      "Expected nine mechanism categories, found",
      nrow(category_meta)
    )
  )
}

workspace <- new.env(
  parent = emptyenv()
)

loaded_objects <- load(
  signature_file,
  envir = workspace
)

if (
  !"sigdb" %in%
  loaded_objects
) {
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
    trimws(
      as.character(
        sigdb$source
      )
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
    "Unexpected CMAP up/down signature counts."
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
    "CMAP parent pairing is inconsistent."
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

up_sizes <- lengths(
  up_gene_lists
)

down_sizes <- lengths(
  down_gene_lists
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

log_message(
  "CMAP parents: ",
  length(parents)
)

log_message(
  "CMAP universe genes: ",
  length(cmap_universe)
)

leaveout_scores <- NULL
coverage_table <- NULL

for (
  category_index in seq_len(
    nrow(category_meta)
  )
) {
  category <- category_meta$
    mechanism_category[
      category_index
    ]

  axis_role <- category_meta$
    axis_role[
      category_index
    ]

  expected_direction <- category_meta$
    expected_CMAP_direction[
      category_index
    ]

  original_genes <- sort(
    unique(
      category_unions$
        gene_symbol[
          category_unions$
            mechanism_category ==
            category
        ]
    )
  )

  removed_maturation <- intersect(
    original_genes,
    maturation_genes
  )

  removed_fetal <- intersect(
    original_genes,
    fetal_genes
  )

  leaveout_genes <- setdiff(
    original_genes,
    program_union
  )

  leaveout_genes_in_CMAP <- intersect(
    leaveout_genes,
    cmap_universe
  )

  if (
    length(leaveout_genes_in_CMAP) < 10L
  ) {
    stop(
      paste(
        "Insufficient leave-program-out genes for",
        category,
        ":",
        length(leaveout_genes_in_CMAP)
      )
    )
  }

  up_hits <- vapply(
    up_gene_lists,
    function(genes) {
      sum(
        genes %in%
          leaveout_genes_in_CMAP
      )
    },
    integer(1)
  )

  down_hits <- vapply(
    down_gene_lists,
    function(genes) {
      sum(
        genes %in%
          leaveout_genes_in_CMAP
      )
    },
    integer(1)
  )

  if (
    expected_direction == "up"
  ) {
    preferred_hits <- up_hits
    opposite_hits <- down_hits
    preferred_sizes <- up_sizes
    opposite_sizes <- down_sizes
  } else {
    preferred_hits <- down_hits
    opposite_hits <- up_hits
    preferred_sizes <- down_sizes
    opposite_sizes <- up_sizes
  }

  total_hits <- preferred_hits +
    opposite_hits

  total_sizes <- preferred_sizes +
    opposite_sizes

  preference_p <- rep(
    1,
    length(parents)
  )

  valid_tests <- total_hits > 0L &
    total_sizes > 0L

  preference_p[
    valid_tests
  ] <- phyper(
    q = preferred_hits[
      valid_tests
    ] - 1L,
    m = total_hits[
      valid_tests
    ],
    n = total_sizes[
      valid_tests
    ] - total_hits[
      valid_tests
    ],
    k = preferred_sizes[
      valid_tests
    ],
    lower.tail = FALSE
  )

  preference_log2OR <- corrected_log2_odds(
    preferred_hits = preferred_hits,
    preferred_size = preferred_sizes,
    comparison_hits = opposite_hits,
    comparison_size = opposite_sizes
  )

  current_scores <- data.frame(
    CMAP_parent = parents,
    mechanism_category = category,
    axis_role = axis_role,
    expected_CMAP_direction = (
      expected_direction
    ),
    original_category_gene_count = (
      length(original_genes)
    ),
    maturation_program_genes_removed = (
      length(removed_maturation)
    ),
    fetal_program_genes_removed = (
      length(removed_fetal)
    ),
    total_program_genes_removed = (
      length(
        intersect(
          original_genes,
          program_union
        )
      )
    ),
    leaveout_category_gene_count = (
      length(leaveout_genes)
    ),
    leaveout_genes_in_CMAP_universe = (
      length(leaveout_genes_in_CMAP)
    ),
    up_signature_size = up_sizes,
    down_signature_size = down_sizes,
    leaveout_category_genes_up = up_hits,
    leaveout_category_genes_down = (
      down_hits
    ),
    preferred_direction_hits = (
      preferred_hits
    ),
    opposite_direction_hits = (
      opposite_hits
    ),
    preferred_direction_log2OR = (
      preference_log2OR
    ),
    directional_preference_p = (
      preference_p
    ),
    stringsAsFactors = FALSE
  )

  leaveout_scores <- rbind(
    leaveout_scores,
    current_scores
  )

  current_coverage <- data.frame(
    mechanism_category = category,
    axis_role = axis_role,
    expected_CMAP_direction = (
      expected_direction
    ),
    original_category_gene_count = (
      length(original_genes)
    ),
    maturation_program_genes_removed = (
      length(removed_maturation)
    ),
    fetal_program_genes_removed = (
      length(removed_fetal)
    ),
    total_program_genes_removed = (
      length(
        intersect(
          original_genes,
          program_union
        )
      )
    ),
    leaveout_category_gene_count = (
      length(leaveout_genes)
    ),
    leaveout_fraction = (
      length(leaveout_genes) /
      length(original_genes)
    ),
    leaveout_genes_in_CMAP_universe = (
      length(leaveout_genes_in_CMAP)
    ),
    CMAP_coverage_fraction = (
      length(leaveout_genes_in_CMAP) /
      length(leaveout_genes)
    ),
    stringsAsFactors = FALSE
  )

  coverage_table <- rbind(
    coverage_table,
    current_coverage
  )

  log_message(
    "Scored category ",
    category_index,
    "/",
    nrow(category_meta),
    ": ",
    category,
    " | retained ",
    length(leaveout_genes),
    "/",
    length(original_genes),
    " genes"
  )
}

leaveout_scores$
  FDR_within_mechanism_category <- ave(
    leaveout_scores$
      directional_preference_p,
    leaveout_scores$
      mechanism_category,
    FUN = function(values) {
      p.adjust(
        values,
        method = "BH"
      )
    }
  )

leaveout_scores$
  FDR_within_CMAP_parent <- ave(
    leaveout_scores$
      directional_preference_p,
    leaveout_scores$
      CMAP_parent,
    FUN = function(values) {
      p.adjust(
        values,
        method = "BH"
      )
    }
  )

leaveout_scores$
  aligned_category_FDR_significant <- (
    leaveout_scores$
      preferred_direction_log2OR > 0 &
    leaveout_scores$
      FDR_within_mechanism_category <
      0.05
  )

leaveout_scores$
  aligned_parent_profile_FDR_significant <- (
    leaveout_scores$
      preferred_direction_log2OR > 0 &
    leaveout_scores$
      FDR_within_CMAP_parent <
      0.05
  )

write_gzip_table(
  leaveout_scores,
  score_output
)

write.table(
  coverage_table,
  coverage_output,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

original_connection <- gzfile(
  original_score_file,
  open = "rt"
)

original_scores <- read.delim(
  original_connection,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

close(
  original_connection
)

original_subset <- original_scores[
  ,
  c(
    "CMAP_parent",
    "mechanism_category",
    "preferred_direction_log2OR",
    "FDR_within_mechanism_category",
    "FDR_within_CMAP_parent",
    "aligned_category_FDR_significant",
    "aligned_parent_profile_FDR_significant"
  ),
  drop = FALSE
]

colnames(original_subset)[
  3:7
] <- c(
  "original_preferred_direction_log2OR",
  "original_FDR_within_mechanism_category",
  "original_FDR_within_CMAP_parent",
  "original_aligned_category_FDR_significant",
  "original_aligned_parent_profile_FDR_significant"
)

leaveout_subset <- leaveout_scores[
  ,
  c(
    "CMAP_parent",
    "mechanism_category",
    "preferred_direction_log2OR",
    "FDR_within_mechanism_category",
    "FDR_within_CMAP_parent",
    "aligned_category_FDR_significant",
    "aligned_parent_profile_FDR_significant"
  ),
  drop = FALSE
]

colnames(leaveout_subset)[
  3:7
] <- c(
  "leaveout_preferred_direction_log2OR",
  "leaveout_FDR_within_mechanism_category",
  "leaveout_FDR_within_CMAP_parent",
  "leaveout_aligned_category_FDR_significant",
  "leaveout_aligned_parent_profile_FDR_significant"
)

comparison <- merge(
  original_subset,
  leaveout_subset,
  by = c(
    "CMAP_parent",
    "mechanism_category"
  ),
  all = TRUE,
  sort = FALSE
)

if (
  nrow(comparison) != 83457L
) {
  stop(
    paste(
      "Unexpected comparison row count:",
      nrow(comparison)
    )
  )
}

comparison$global_support_status <- ifelse(
  comparison$
    original_aligned_category_FDR_significant &
  comparison$
    leaveout_aligned_category_FDR_significant,
  "retained_after_program_removal",
  ifelse(
    comparison$
      original_aligned_category_FDR_significant &
    !comparison$
      leaveout_aligned_category_FDR_significant,
    "lost_after_program_removal",
    ifelse(
      !comparison$
        original_aligned_category_FDR_significant &
      comparison$
        leaveout_aligned_category_FDR_significant,
      "gained_after_program_removal",
      "not_globally_supported"
    )
  )
)

write_gzip_table(
  comparison,
  comparison_output
)

maxT_table <- read.delim(
  maxT_file,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

if (
  nrow(maxT_table) != 142L
) {
  stop(
    paste(
      "Expected 142 maxT parents, found",
      nrow(maxT_table)
    )
  )
}

maxT_parents <- unique(
  as.character(
    maxT_table$CMAP_parent
  )
)

comparison$is_maxT_parent <- (
  comparison$CMAP_parent %in%
    maxT_parents
)

category_summary <- NULL

for (
  category_index in seq_len(
    nrow(category_meta)
  )
) {
  category <- category_meta$
    mechanism_category[
      category_index
    ]

  current <- comparison[
    comparison$mechanism_category ==
      category,
    ,
    drop = FALSE
  ]

  current_maxT <- current[
    current$is_maxT_parent,
    ,
    drop = FALSE
  ]

  original_global <- sum(
    current$
      original_aligned_category_FDR_significant
  )

  leaveout_global <- sum(
    current$
      leaveout_aligned_category_FDR_significant
  )

  retained_global <- sum(
    current$global_support_status ==
      "retained_after_program_removal"
  )

  category_row <- data.frame(
    mechanism_category = category,
    axis_role = category_meta$
      axis_role[
        category_index
      ],
    expected_CMAP_direction = (
      category_meta$
        expected_CMAP_direction[
          category_index
        ]
    ),
    all_parent_score_spearman = safe_spearman(
      current$
        original_preferred_direction_log2OR,
      current$
        leaveout_preferred_direction_log2OR
    ),
    maxT_parent_score_spearman = safe_spearman(
      current_maxT$
        original_preferred_direction_log2OR,
      current_maxT$
        leaveout_preferred_direction_log2OR
    ),
    original_global_FDR_parent_count = (
      original_global
    ),
    leaveout_global_FDR_parent_count = (
      leaveout_global
    ),
    retained_global_FDR_parent_count = (
      retained_global
    ),
    lost_global_FDR_parent_count = sum(
      current$global_support_status ==
        "lost_after_program_removal"
    ),
    gained_global_FDR_parent_count = sum(
      current$global_support_status ==
        "gained_after_program_removal"
    ),
    global_support_retention_fraction = if (
      original_global > 0L
    ) {
      retained_global /
        original_global
    } else {
      NA_real_
    },
    maxT_original_global_FDR_count = sum(
      current_maxT$
        original_aligned_category_FDR_significant
    ),
    maxT_leaveout_global_FDR_count = sum(
      current_maxT$
        leaveout_aligned_category_FDR_significant
    ),
    maxT_retained_global_FDR_count = sum(
      current_maxT$global_support_status ==
        "retained_after_program_removal"
    ),
    maxT_lost_global_FDR_count = sum(
      current_maxT$global_support_status ==
        "lost_after_program_removal"
    ),
    maxT_gained_global_FDR_count = sum(
      current_maxT$global_support_status ==
        "gained_after_program_removal"
    ),
    stringsAsFactors = FALSE
  )

  category_summary <- rbind(
    category_summary,
    category_row
  )
}

write.table(
  category_summary,
  category_summary_output,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

maxT_comparison <- comparison[
  comparison$is_maxT_parent,
  ,
  drop = FALSE
]

maxT_robustness <- aggregate(
  cbind(
    original_global = as.integer(
      maxT_comparison$
        original_aligned_category_FDR_significant
    ),
    leaveout_global = as.integer(
      maxT_comparison$
        leaveout_aligned_category_FDR_significant
    ),
    original_within_parent = as.integer(
      maxT_comparison$
        original_aligned_parent_profile_FDR_significant
    ),
    leaveout_within_parent = as.integer(
      maxT_comparison$
        leaveout_aligned_parent_profile_FDR_significant
    )
  ),
  by = list(
    CMAP_parent = maxT_comparison$CMAP_parent
  ),
  FUN = sum
)

maxT_robustness <- merge(
  maxT_table[
    ,
    c(
      "CMAP_parent",
      "chemical_name",
      "DTXSID",
      "developmental_direction_score",
      "maxT_maturation_FWER_p"
    ),
    drop = FALSE
  ],
  maxT_robustness,
  by = "CMAP_parent",
  all.x = TRUE,
  sort = FALSE
)

maxT_robustness <- maxT_robustness[
  order(
    -maxT_robustness$leaveout_global,
    -maxT_robustness$
      leaveout_within_parent,
    maxT_robustness$
      maxT_maturation_FWER_p,
    maxT_robustness$CMAP_parent
  ),
  ,
  drop = FALSE
]

write.table(
  maxT_robustness,
  maxT_summary_output,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

completion <- data.frame(
  CMAP_parents = length(
    parents
  ),
  mechanism_categories = nrow(
    category_meta
  ),
  parent_category_tests = nrow(
    leaveout_scores
  ),
  expected_parent_category_tests = (
    length(parents) *
    nrow(category_meta)
  ),
  maxT_parents_evaluated = length(
    maxT_parents
  ),
  maturation_genes_removed = length(
    maturation_genes
  ),
  fetal_genes_removed = length(
    fetal_genes
  ),
  mechanism_gene_sets_recomputed = TRUE,
  within_category_FDR_recomputed = TRUE,
  within_parent_FDR_recomputed = TRUE,
  original_vs_leaveout_comparison_completed = TRUE,
  Phase6D1_status = "completed",
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
  "===== LEAVE-PROGRAM-OUT COVERAGE ====="
)

for (
  line in capture.output(
    print(
      coverage_table,
      row.names = FALSE
    )
  )
) {
  log_message(line)
}

log_message("")
log_message(
  "===== CATEGORY ROBUSTNESS ====="
)

for (
  line in capture.output(
    print(
      category_summary,
      row.names = FALSE
    )
  )
) {
  log_message(line)
}

log_message("")
log_message(
  "===== COMPLETION ====="
)

for (
  line in capture.output(
    print(
      completion,
      row.names = FALSE
    )
  )
) {
  log_message(line)
}
