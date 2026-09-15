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

observed_file <- file.path(
  project,
  "03_processed_data/perturbational_validation/phase6C",
  "phase6C2B_CMAP_empirical_null_results.tsv.gz"
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
  "phase6C3D_CMAP_mechanism_decomposition.log"
)

score_file <- file.path(
  processed_dir,
  "phase6C3D_CMAP_parent_mechanism_scores.tsv.gz"
)

parent_summary_file <- file.path(
  processed_dir,
  "phase6C3D_CMAP_parent_mechanism_summary.tsv.gz"
)

coverage_file <- file.path(
  table_dir,
  "phase6C3D_mechanism_category_CMAP_coverage.tsv"
)

global_summary_file <- file.path(
  table_dir,
  "phase6C3D_mechanism_category_global_summary.tsv"
)

maxT_summary_file <- file.path(
  table_dir,
  "phase6C3D_maxT_maturation_parent_mechanism_summary.tsv"
)

completion_file <- file.path(
  table_dir,
  "phase6C3D_completion_summary.tsv"
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

log_message(
  "===== Phase 6C3D started ====="
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
    paste(
      "Unexpected CMAP up/down counts:",
      nrow(up_table),
      nrow(down_table)
    )
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
    "CMAP up/down parent pairing is inconsistent."
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
      "Missing category-union columns:",
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
  nrow(category_meta) != 9L ||
  anyDuplicated(
    category_meta$mechanism_category
  )
) {
  stop(
    "Expected exactly nine unique mechanism categories."
  )
}

if (
  !all(
    category_meta$expected_CMAP_direction %in%
      c(
        "up",
        "dn"
      )
  )
) {
  stop(
    "Unexpected category direction."
  )
}

coverage_rows <- vector(
  "list",
  nrow(category_meta)
)

score_rows <- vector(
  "list",
  nrow(category_meta)
)

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

  category_genes <- sort(
    unique(
      category_unions$
        gene_symbol[
          category_unions$
            mechanism_category ==
            category
        ]
    )
  )

  category_genes_in_CMAP <- intersect(
    category_genes,
    cmap_universe
  )

  up_hits <- vapply(
    up_gene_lists,
    function(genes) {
      sum(
        genes %in%
          category_genes_in_CMAP
      )
    },
    integer(1)
  )

  down_hits <- vapply(
    down_gene_lists,
    function(genes) {
      sum(
        genes %in%
          category_genes_in_CMAP
      )
    },
    integer(1)
  )

  if (
    expected_direction == "up"
  ) {
    preferred_hits <- up_hits
    comparison_hits <- down_hits
    preferred_sizes <- up_sizes
    comparison_sizes <- down_sizes
  } else {
    preferred_hits <- down_hits
    comparison_hits <- up_hits
    preferred_sizes <- down_sizes
    comparison_sizes <- up_sizes
  }

  total_hits <- (
    preferred_hits +
    comparison_hits
  )

  total_sizes <- (
    preferred_sizes +
    comparison_sizes
  )

  preference_p <- rep(
    1,
    length(parents)
  )

  valid_tests <- (
    total_hits > 0L &
    total_sizes > 0L
  )

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
    comparison_hits = comparison_hits,
    comparison_size = comparison_sizes
  )

  score_rows[[category_index]] <- data.frame(
    CMAP_parent = parents,
    mechanism_category = category,
    axis_role = axis_role,
    expected_CMAP_direction = (
      expected_direction
    ),
    category_union_genes = length(
      category_genes
    ),
    category_genes_in_CMAP_universe = (
      length(
        category_genes_in_CMAP
      )
    ),
    up_signature_size = up_sizes,
    down_signature_size = down_sizes,
    category_genes_up = up_hits,
    category_genes_down = down_hits,
    preferred_direction_hits = (
      preferred_hits
    ),
    opposite_direction_hits = (
      comparison_hits
    ),
    total_category_hits = total_hits,
    preferred_direction_log2OR = (
      preference_log2OR
    ),
    directional_preference_p = (
      preference_p
    ),
    stringsAsFactors = FALSE
  )

  coverage_rows[[category_index]] <- data.frame(
    mechanism_category = category,
    axis_role = axis_role,
    expected_CMAP_direction = (
      expected_direction
    ),
    category_union_genes = length(
      category_genes
    ),
    category_genes_in_CMAP_universe = (
      length(
        category_genes_in_CMAP
      )
    ),
    genes_absent_from_CMAP_universe = (
      length(
        setdiff(
          category_genes,
          cmap_universe
        )
      )
    ),
    CMAP_coverage_fraction = (
      length(
        category_genes_in_CMAP
      ) /
      length(
        category_genes
      )
    ),
    stringsAsFactors = FALSE
  )

  log_message(
    "Scored category ",
    category_index,
    "/",
    nrow(category_meta),
    ": ",
    category
  )
}

scores <- do.call(
  rbind,
  score_rows
)

coverage <- do.call(
  rbind,
  coverage_rows
)

scores$FDR_within_mechanism_category <- ave(
  scores$directional_preference_p,
  scores$mechanism_category,
  FUN = function(values) {
    p.adjust(
      values,
      method = "BH"
    )
  }
)

scores$FDR_within_CMAP_parent <- ave(
  scores$directional_preference_p,
  scores$CMAP_parent,
  FUN = function(values) {
    p.adjust(
      values,
      method = "BH"
    )
  }
)

scores$FDR_across_all_parent_category_tests <- p.adjust(
  scores$directional_preference_p,
  method = "BH"
)

scores$aligned_category_FDR_significant <- (
  scores$preferred_direction_log2OR > 0 &
  scores$FDR_within_mechanism_category <
    0.05
)

scores$aligned_parent_profile_FDR_significant <- (
  scores$preferred_direction_log2OR > 0 &
  scores$FDR_within_CMAP_parent <
    0.05
)

scores <- scores[
  order(
    scores$CMAP_parent,
    scores$mechanism_category
  ),
  ,
  drop = FALSE
]

write_gzip_table(
  scores,
  score_file
)

write.table(
  coverage,
  coverage_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

global_summary_rows <- lapply(
  seq_len(
    nrow(category_meta)
  ),
  function(category_index) {
    category <- category_meta$
      mechanism_category[
        category_index
      ]

    current <- scores[
      scores$mechanism_category ==
        category,
      ,
      drop = FALSE
    ]

    data.frame(
      mechanism_category = category,
      axis_role = unique(
        current$axis_role
      ),
      expected_CMAP_direction = unique(
        current$
          expected_CMAP_direction
      ),
      CMAP_parents_tested = nrow(
        current
      ),
      parents_with_any_category_overlap = sum(
        current$total_category_hits >
          0L
      ),
      parents_with_positive_direction_score = sum(
        current$
          preferred_direction_log2OR >
          0
      ),
      parents_with_negative_direction_score = sum(
        current$
          preferred_direction_log2OR <
          0
      ),
      aligned_category_FDR_lt_0_05 = sum(
        current$
          aligned_category_FDR_significant
      ),
      aligned_parent_profile_FDR_lt_0_05 = sum(
        current$
          aligned_parent_profile_FDR_significant
      ),
      median_preferred_direction_log2OR = median(
        current$
          preferred_direction_log2OR
      ),
      percentile_95_preferred_direction_log2OR = as.numeric(
        quantile(
          current$
            preferred_direction_log2OR,
          0.95
        )
      ),
      maximum_preferred_direction_log2OR = max(
        current$
          preferred_direction_log2OR
      ),
      stringsAsFactors = FALSE
    )
  }
)

global_summary <- do.call(
  rbind,
  global_summary_rows
)

write.table(
  global_summary,
  global_summary_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

score_matrix <- xtabs(
  preferred_direction_log2OR ~
    CMAP_parent +
    mechanism_category,
  data = scores
)

category_FDR_matrix <- xtabs(
  as.integer(
    aligned_category_FDR_significant
  ) ~
    CMAP_parent +
    mechanism_category,
  data = scores
)

parent_FDR_matrix <- xtabs(
  as.integer(
    aligned_parent_profile_FDR_significant
  ) ~
    CMAP_parent +
    mechanism_category,
  data = scores
)

required_matrix_categories <- c(
  "early_proliferative_state",
  "fetal_neurodevelopmental_state",
  "late_synaptic_maturation",
  "mitochondrial_metabolic_maturation",
  "glial_myelin_maturation",
  "DNA_damage_p53",
  "apoptosis_cytotoxicity",
  "oxidative_hypoxic_stress",
  "inflammatory_stress"
)

if (
  !all(
    required_matrix_categories %in%
      colnames(score_matrix)
  )
) {
  stop(
    "One or more mechanism categories are missing from the score matrix."
  )
}

early_categories <- c(
  "early_proliferative_state",
  "fetal_neurodevelopmental_state"
)

late_categories <- c(
  "late_synaptic_maturation",
  "mitochondrial_metabolic_maturation",
  "glial_myelin_maturation"
)

injury_categories <- c(
  "DNA_damage_p53",
  "apoptosis_cytotoxicity",
  "oxidative_hypoxic_stress",
  "inflammatory_stress"
)

parent_summary <- data.frame(
  CMAP_parent = rownames(
    score_matrix
  ),
  early_program_suppression_score = rowMeans(
    score_matrix[
      ,
      early_categories,
      drop = FALSE
    ]
  ),
  late_maturation_support_score = rowMeans(
    score_matrix[
      ,
      late_categories,
      drop = FALSE
    ]
  ),
  injury_stress_confounder_score = rowMeans(
    score_matrix[
      ,
      injury_categories,
      drop = FALSE
    ]
  ),
  early_aligned_category_FDR_count = rowSums(
    category_FDR_matrix[
      ,
      early_categories,
      drop = FALSE
    ]
  ),
  late_aligned_category_FDR_count = rowSums(
    category_FDR_matrix[
      ,
      late_categories,
      drop = FALSE
    ]
  ),
  injury_aligned_category_FDR_count = rowSums(
    category_FDR_matrix[
      ,
      injury_categories,
      drop = FALSE
    ]
  ),
  early_aligned_parent_profile_FDR_count = rowSums(
    parent_FDR_matrix[
      ,
      early_categories,
      drop = FALSE
    ]
  ),
  late_aligned_parent_profile_FDR_count = rowSums(
    parent_FDR_matrix[
      ,
      late_categories,
      drop = FALSE
    ]
  ),
  injury_aligned_parent_profile_FDR_count = rowSums(
    parent_FDR_matrix[
      ,
      injury_categories,
      drop = FALSE
    ]
  ),
  stringsAsFactors = FALSE
)

for (
  category in required_matrix_categories
) {
  output_name <- paste0(
    category,
    "_log2OR"
  )

  parent_summary[
    ,
    output_name
  ] <- score_matrix[
    ,
    category
  ]
}

observed_connection <- gzfile(
  observed_file,
  open = "rt"
)

observed <- read.delim(
  observed_connection,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

close(
  observed_connection
)

required_observed_columns <- c(
  "CMAP_parent",
  "chemical_name",
  "DTXSID",
  "developmental_direction_score",
  "empirical_maturation_FDR",
  "empirical_fetal_FDR",
  "maxT_maturation_FWER_p",
  "maxT_fetal_FWER_p",
  "maturation_component_consistent",
  "fetal_component_consistent",
  "empirical_directional_state"
)

missing_observed_columns <- setdiff(
  required_observed_columns,
  colnames(observed)
)

if (
  length(missing_observed_columns) > 0L
) {
  stop(
    paste(
      "Missing empirical-result columns:",
      paste(
        missing_observed_columns,
        collapse = ", "
      )
    )
  )
}

observed_subset <- observed[
  ,
  required_observed_columns,
  drop = FALSE
]

parent_summary <- merge(
  observed_subset,
  parent_summary,
  by = "CMAP_parent",
  all.x = TRUE,
  sort = FALSE
)

if (
  nrow(parent_summary) != 9273L ||
  any(
    is.na(
      parent_summary$
        late_maturation_support_score
    )
  )
) {
  stop(
    "Parent-level mechanism summary merge failed."
  )
}

parent_summary$maxT_maturation_parent <- (
  parent_summary$
    developmental_direction_score >
    0 &
  parent_summary$
    maturation_component_consistent &
  parent_summary$
    maxT_maturation_FWER_p <
    0.05
)

parent_summary$empirical_maturation_parent <- (
  parent_summary$
    developmental_direction_score >
    0 &
  parent_summary$
    maturation_component_consistent &
  parent_summary$
    empirical_maturation_FDR <
    0.05
)

parent_summary$empirical_fetal_parent <- (
  parent_summary$
    developmental_direction_score <
    0 &
  parent_summary$
    fetal_component_consistent &
  parent_summary$
    empirical_fetal_FDR <
    0.05
)

parent_summary <- parent_summary[
  order(
    parent_summary$
      maxT_maturation_FWER_p,
    -parent_summary$
      developmental_direction_score,
    parent_summary$CMAP_parent
  ),
  ,
  drop = FALSE
]

write_gzip_table(
  parent_summary,
  parent_summary_file
)

maxT_summary <- parent_summary[
  parent_summary$
    maxT_maturation_parent,
  ,
  drop = FALSE
]

maxT_summary <- maxT_summary[
  order(
    -maxT_summary$
      late_aligned_category_FDR_count,
    maxT_summary$
      injury_aligned_category_FDR_count,
    -maxT_summary$
      late_maturation_support_score,
    maxT_summary$
      injury_stress_confounder_score,
    maxT_summary$
      maxT_maturation_FWER_p,
    maxT_summary$CMAP_parent
  ),
  ,
  drop = FALSE
]

write.table(
  maxT_summary,
  maxT_summary_file,
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
    scores
  ),
  expected_parent_category_tests = (
    length(parents) *
    nrow(category_meta)
  ),
  maxT_maturation_parents_retained = nrow(
    maxT_summary
  ),
  category_union_scoring_used = TRUE,
  within_category_FDR_calculated = TRUE,
  within_parent_FDR_calculated = TRUE,
  early_late_injury_axes_summarized = TRUE,
  Phase6C3D_status = "completed",
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
  "===== CATEGORY GLOBAL SUMMARY ====="
)

for (
  line in capture.output(
    print(
      global_summary,
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
