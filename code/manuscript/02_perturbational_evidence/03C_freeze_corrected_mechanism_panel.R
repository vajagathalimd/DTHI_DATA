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

table_dir <- file.path(
  project,
  "07_tables/main_tables/phase6"
)

processed_dir <- file.path(
  project,
  "03_processed_data/perturbational_validation/phase6C"
)

log_file <- file.path(
  project,
  "09_pipeline_logs/phase6",
  "phase6C3C_fixed_mechanism_panel.log"
)

manifest_file <- file.path(
  table_dir,
  "phase6C3C_fixed_mechanism_panel_manifest.tsv"
)

long_gene_file <- file.path(
  processed_dir,
  "phase6C3C_fixed_mechanism_panel_genes.tsv.gz"
)

category_union_file <- file.path(
  processed_dir,
  "phase6C3C_mechanism_category_gene_unions.tsv.gz"
)

jaccard_file <- file.path(
  table_dir,
  "phase6C3C_mechanism_panel_pairwise_jaccard.tsv"
)

category_summary_file <- file.path(
  table_dir,
  "phase6C3C_mechanism_category_summary.tsv"
)

completion_file <- file.path(
  table_dir,
  "phase6C3C_completion_summary.tsv"
)

dir.create(
  table_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  processed_dir,
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

normalize_signature <- function(values) {
  values <- toupper(
    trimws(
      as.character(values)
    )
  )

  values <- gsub(
    "[[:space:]-]+",
    "_",
    values
  )

  values <- gsub(
    "_+",
    "_",
    values
  )

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
      "rt"
    )
  } else {
    file(
      path,
      "rt"
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

  matched_columns <- intersect(
    c(
      "gene_symbol",
      "gene",
      "symbol"
    ),
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

parse_gene_list <- function(value) {
  genes <- strsplit(
    as.character(value),
    "|",
    fixed = TRUE
  )[[1L]]

  genes <- unique(
    normalize_gene(
      genes
    )
  )

  genes[
    genes != ""
  ]
}

write_gzip_table <- function(
  table,
  path
) {
  connection <- gzfile(
    path,
    "wt"
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

panel <- data.frame(
  mechanism_category = c(
    rep(
      "early_proliferative_state",
      4
    ),
    rep(
      "fetal_neurodevelopmental_state",
      4
    ),
    rep(
      "late_synaptic_maturation",
      3
    ),
    rep(
      "mitochondrial_metabolic_maturation",
      3
    ),
    rep(
      "glial_myelin_maturation",
      4
    ),
    rep(
      "DNA_damage_p53",
      2
    ),
    "apoptosis_cytotoxicity",
    rep(
      "oxidative_hypoxic_stress",
      2
    ),
    rep(
      "inflammatory_stress",
      2
    )
  ),

  axis_role = c(
    rep(
      "early_program_suppression",
      4
    ),
    rep(
      "early_program_suppression_ambiguous",
      4
    ),
    rep(
      "late_program_induction",
      3
    ),
    rep(
      "late_program_induction",
      3
    ),
    rep(
      "late_program_induction",
      4
    ),
    rep(
      "injury_stress_confounder",
      2
    ),
    "injury_stress_confounder",
    rep(
      "injury_stress_confounder",
      2
    ),
    rep(
      "injury_stress_confounder",
      2
    )
  ),

  expected_CMAP_direction = c(
    rep(
      "dn",
      4
    ),
    rep(
      "dn",
      4
    ),
    rep(
      "up",
      3
    ),
    rep(
      "up",
      3
    ),
    rep(
      "up",
      4
    ),
    rep(
      "up",
      2
    ),
    "up",
    rep(
      "up",
      2
    ),
    rep(
      "up",
      2
    )
  ),

  panel_signature = c(
    "HALLMARK_E2F_TARGETS",
    "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_MYC_TARGETS_V1",
    "HALLMARK_MITOTIC_SPINDLE",

    "GO_CENTRAL_NERVOUS_SYSTEM_NEURON_DIFFERENTIATION",
    "GO_NEURON_PROJECTION_GUIDANCE",
    "KEGG_AXON_GUIDANCE",
    "GO_REGULATION_OF_AXONOGENESIS",

    "GO_SYNAPTIC_SIGNALING",
    "REACTOME_NEUROTRANSMITTER_RELEASE_CYCLE",
    "GO_POSITIVE_REGULATION_OF_SYNAPTIC_TRANSMISSION",

    "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
    "GO_ELECTRON_TRANSPORT_CHAIN",
    "GO_OXIDATIVE_PHOSPHORYLATION",

    "GO_OLIGODENDROCYTE_DIFFERENTIATION",
    "GO_GLIAL_CELL_DIFFERENTIATION",
    "GO_MYELIN_SHEATH",
    "GO_ASTROCYTE_DIFFERENTIATION",

    "HALLMARK_P53_PATHWAY",
    "HALLMARK_DNA_REPAIR",

    "HALLMARK_APOPTOSIS",

    "HALLMARK_HYPOXIA",
    "WEIGEL_OXIDATIVE_STRESS_RESPONSE",

    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_INFLAMMATORY_RESPONSE"
  ),

  panel_priority = c(
    1, 1, 2, 2,
    1, 1, 2, 2,
    1, 1, 2,
    1, 2, 2,
    1, 1, 1, 2,
    1, 1,
    1,
    1, 2,
    1, 1
  ),

  interpretation_note = c(
    rep(
      paste(
        "Down-regulation supports cell-cycle withdrawal",
        "but may reflect cytostasis or toxicity."
      ),
      4
    ),

    rep(
      paste(
        "Down-regulation may reflect temporal progression",
        "or harmful suppression of early neurodevelopment."
      ),
      4
    ),

    rep(
      paste(
        "Up-regulation supports late neuronal",
        "or synaptic maturation."
      ),
      3
    ),

    rep(
      paste(
        "Up-regulation supports oxidative",
        "metabolic maturation."
      ),
      3
    ),

    rep(
      paste(
        "Up-regulation supports glial differentiation",
        "or myelin-associated maturation."
      ),
      4
    ),

    rep(
      paste(
        "Up-regulation indicates DNA-damage",
        "or p53 stress confounding."
      ),
      2
    ),

    paste(
      "Up-regulation indicates apoptotic",
      "or cytotoxic confounding."
    ),

    rep(
      paste(
        "Up-regulation indicates oxidative",
        "or hypoxic stress confounding."
      ),
      2
    ),

    rep(
      paste(
        "Up-regulation indicates inflammatory",
        "stress confounding."
      ),
      2
    )
  ),

  stringsAsFactors = FALSE
)

if (
  nrow(panel) != 25L ||
  anyDuplicated(
    panel$panel_signature
  )
) {
  stop(
    "Fixed panel definition is invalid."
  )
}

panel$panel_signature_normalized <- normalize_signature(
  panel$panel_signature
)

log_message(
  "===== Phase 6C3C started ====="
)

log_message(
  "Fixed panel signatures requested: ",
  nrow(panel)
)

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

sigdb$signature_normalized <- normalize_signature(
  sigdb$signature
)

sigdb$source_lower <- tolower(
  trimws(
    as.character(
      sigdb$source
    )
  )
)

sigdb$subsource_text <- as.character(
  sigdb$subsource
)

maturation_genes <- read_program_genes(
  maturation_file
)

fetal_genes <- read_program_genes(
  fetal_file
)

manifest_rows <- vector(
  "list",
  nrow(panel)
)

gene_rows <- vector(
  "list",
  nrow(panel)
)

panel_gene_sets <- vector(
  "list",
  nrow(panel)
)

for (
  index in seq_len(
    nrow(panel)
  )
) {
  candidates <- sigdb[
    sigdb$source_lower ==
      "msigdb" &
    sigdb$signature_normalized ==
      panel$
        panel_signature_normalized[
          index
        ],
    ,
    drop = FALSE
  ]

  non_archived <- candidates[
    !grepl(
      "ARCHIVED",
      candidates$subsource_text,
      ignore.case = TRUE
    ),
    ,
    drop = FALSE
  ]

  if (
    nrow(non_archived) > 0L
  ) {
    candidates <- non_archived
  }

  if (
    nrow(candidates) != 1L
  ) {
    stop(
      paste(
        "Expected exactly one MSigDB match for",
        panel$panel_signature[index],
        "but found",
        nrow(candidates)
      )
    )
  }

  selected <- candidates[
    1L,
    ,
    drop = FALSE
  ]

  genes <- parse_gene_list(
    selected$gene.list
  )

  panel_gene_sets[[index]] <- genes

  maturation_overlap <- length(
    intersect(
      genes,
      maturation_genes
    )
  )

  fetal_overlap <- length(
    intersect(
      genes,
      fetal_genes
    )
  )

  manifest_rows[[index]] <- data.frame(
    mechanism_category = (
      panel$mechanism_category[index]
    ),
    axis_role = panel$axis_role[index],
    expected_CMAP_direction = (
      panel$expected_CMAP_direction[index]
    ),
    panel_priority = panel$panel_priority[index],
    panel_signature_requested = (
      panel$panel_signature[index]
    ),
    signature_library_name = as.character(
      selected$signature
    ),
    source = as.character(
      selected$source
    ),
    subsource = as.character(
      selected$subsource
    ),
    signature_type = as.character(
      selected$type
    ),
    signature_direction = as.character(
      selected$direction
    ),
    reported_gene_count = as.integer(
      selected$ngene
    ),
    parsed_unique_gene_count = length(
      genes
    ),
    maturation_overlap = maturation_overlap,
    maturation_overlap_fraction = (
      maturation_overlap /
      length(genes)
    ),
    fetal_overlap = fetal_overlap,
    fetal_overlap_fraction = (
      fetal_overlap /
      length(genes)
    ),
    interpretation_note = (
      panel$interpretation_note[index]
    ),
    stringsAsFactors = FALSE
  )

  gene_rows[[index]] <- data.frame(
    mechanism_category = (
      panel$mechanism_category[index]
    ),
    axis_role = panel$axis_role[index],
    expected_CMAP_direction = (
      panel$expected_CMAP_direction[index]
    ),
    panel_signature = (
      panel$panel_signature[index]
    ),
    gene_symbol = genes,
    stringsAsFactors = FALSE
  )
}

manifest <- do.call(
  rbind,
  manifest_rows
)

long_genes <- do.call(
  rbind,
  gene_rows
)

write.table(
  manifest,
  manifest_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

write_gzip_table(
  long_genes,
  long_gene_file
)

categories <- unique(
  panel$mechanism_category
)

category_union_rows <- list()
category_summary_rows <- list()

for (
  category_index in seq_along(
    categories
  )
) {
  category <- categories[[category_index]]

  term_indices <- which(
    panel$mechanism_category ==
      category
  )

  union_genes <- sort(
    unique(
      unlist(
        panel_gene_sets[
          term_indices
        ],
        use.names = FALSE
      )
    )
  )

  category_union_rows[[category_index]] <- data.frame(
    mechanism_category = category,
    axis_role = panel$axis_role[
      term_indices[[1L]]
    ],
    expected_CMAP_direction = (
      panel$expected_CMAP_direction[
        term_indices[[1L]]
      ]
    ),
    gene_symbol = union_genes,
    stringsAsFactors = FALSE
  )

  total_memberships <- sum(
    lengths(
      panel_gene_sets[
        term_indices
      ]
    )
  )

  maturation_overlap <- length(
    intersect(
      union_genes,
      maturation_genes
    )
  )

  fetal_overlap <- length(
    intersect(
      union_genes,
      fetal_genes
    )
  )

  category_summary_rows[[category_index]] <- data.frame(
    mechanism_category = category,
    axis_role = panel$axis_role[
      term_indices[[1L]]
    ],
    expected_CMAP_direction = (
      panel$expected_CMAP_direction[
        term_indices[[1L]]
      ]
    ),
    panel_terms = length(
      term_indices
    ),
    total_gene_memberships = (
      total_memberships
    ),
    union_gene_count = length(
      union_genes
    ),
    redundancy_fraction = (
      1 -
      length(union_genes) /
      total_memberships
    ),
    maturation_overlap = maturation_overlap,
    maturation_overlap_fraction = (
      maturation_overlap /
      length(union_genes)
    ),
    fetal_overlap = fetal_overlap,
    fetal_overlap_fraction = (
      fetal_overlap /
      length(union_genes)
    ),
    stringsAsFactors = FALSE
  )
}

category_unions <- do.call(
  rbind,
  category_union_rows
)

category_summary <- do.call(
  rbind,
  category_summary_rows
)

write_gzip_table(
  category_unions,
  category_union_file
)

pair_indices <- combn(
  seq_len(
    nrow(panel)
  ),
  2L
)

jaccard_rows <- vector(
  "list",
  ncol(pair_indices)
)

for (
  pair_index in seq_len(
    ncol(pair_indices)
  )
) {
  first <- pair_indices[
    1L,
    pair_index
  ]

  second <- pair_indices[
    2L,
    pair_index
  ]

  intersection_size <- length(
    intersect(
      panel_gene_sets[[first]],
      panel_gene_sets[[second]]
    )
  )

  union_size <- length(
    union(
      panel_gene_sets[[first]],
      panel_gene_sets[[second]]
    )
  )

  jaccard_rows[[pair_index]] <- data.frame(
    first_category = (
      panel$mechanism_category[first]
    ),
    first_signature = (
      panel$panel_signature[first]
    ),
    second_category = (
      panel$mechanism_category[second]
    ),
    second_signature = (
      panel$panel_signature[second]
    ),
    same_category = (
      panel$mechanism_category[first] ==
      panel$mechanism_category[second]
    ),
    intersection_genes = (
      intersection_size
    ),
    union_genes = union_size,
    jaccard = if (
      union_size > 0L
    ) {
      intersection_size /
        union_size
    } else {
      NA_real_
    },
    stringsAsFactors = FALSE
  )
}

jaccard_table <- do.call(
  rbind,
  jaccard_rows
)

write.table(
  jaccard_table,
  jaccard_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

category_summary$
  maximum_within_category_jaccard <- vapply(
    category_summary$
      mechanism_category,
    function(category) {
      values <- jaccard_table$jaccard[
        jaccard_table$same_category &
        jaccard_table$first_category ==
          category
      ]

      if (
        length(values) == 0L
      ) {
        NA_real_
      } else {
        max(
          values,
          na.rm = TRUE
        )
      }
    },
    numeric(1)
  )

write.table(
  category_summary,
  category_summary_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

completion <- data.frame(
  fixed_panel_signatures_requested = nrow(
    panel
  ),
  fixed_panel_signatures_resolved = nrow(
    manifest
  ),
  mechanism_categories = nrow(
    category_summary
  ),
  panel_gene_memberships = nrow(
    long_genes
  ),
  category_union_gene_memberships = nrow(
    category_unions
  ),
  pairwise_jaccard_tests = nrow(
    jaccard_table
  ),
  outcome_independent_panel = TRUE,
  early_and_late_developmental_axes_separated = TRUE,
  injury_confounder_axis_separated = TRUE,
  Phase6C3C_status = "completed",
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
  "===== FIXED PANEL MANIFEST ====="
)

for (
  line in capture.output(
    print(
      manifest,
      row.names = FALSE
    )
  )
) {
  log_message(line)
}

log_message("")
log_message(
  "===== CATEGORY SUMMARY ====="
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
