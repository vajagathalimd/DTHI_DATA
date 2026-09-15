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

metadata_dir <- file.path(
  project,
  "02_metadata/phase6/EPA_HTTr_schema"
)

log_file <- file.path(
  project,
  "09_pipeline_logs/phase6",
  "phase6B4A_RData_schema_inspection.log"
)

object_manifest_file <- file.path(
  table_dir,
  "phase6B4A_RData_object_manifest.tsv"
)

column_manifest_file <- file.path(
  table_dir,
  "phase6B4A_RData_column_manifest.tsv"
)

list_manifest_file <- file.path(
  table_dir,
  "phase6B4A_RData_list_element_manifest.tsv"
)

structure_report_file <- file.path(
  metadata_dir,
  "phase6B4A_RData_structure_report.txt"
)

dir.create(
  table_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  metadata_dir,
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

clean_text <- function(value) {
  if (length(value) == 0L) {
    return("")
  }

  value <- paste(
    as.character(value),
    collapse = "|"
  )

  value <- gsub(
    "[\t\r\n]+",
    " ",
    value
  )

  substr(
    value,
    1L,
    2000L
  )
}

dimension_text <- function(object) {
  dimensions <- dim(object)

  if (is.null(dimensions)) {
    return("")
  }

  paste(
    dimensions,
    collapse = "x"
  )
}

preview_values <- function(values, maximum = 8L) {
  if (length(values) == 0L) {
    return("")
  }

  values <- head(
    values,
    maximum
  )

  values <- vapply(
    values,
    function(value) {
      if (length(value) == 0L || is.na(value)) {
        return("<NA>")
      }

      clean_text(value)
    },
    character(1)
  )

  paste(
    values,
    collapse = "|"
  )
}

empty_object_manifest <- data.frame(
  object_name = character(),
  object_class = character(),
  object_type = character(),
  object_mode = character(),
  object_length = numeric(),
  object_dimensions = character(),
  number_of_rows = numeric(),
  number_of_columns = numeric(),
  object_size_bytes = numeric(),
  object_size_megabytes = numeric(),
  is_data_frame = logical(),
  is_matrix = logical(),
  is_list = logical(),
  is_environment = logical(),
  is_S4 = logical(),
  top_level_names_count = numeric(),
  top_level_names_preview = character(),
  row_names_preview = character(),
  stringsAsFactors = FALSE
)

empty_column_manifest <- data.frame(
  object_name = character(),
  column_index = integer(),
  column_name = character(),
  column_class = character(),
  column_type = character(),
  column_length = numeric(),
  missing_values = numeric(),
  nonmissing_values = numeric(),
  first_values = character(),
  numeric_minimum = numeric(),
  numeric_maximum = numeric(),
  stringsAsFactors = FALSE
)

empty_list_manifest <- data.frame(
  object_name = character(),
  element_index = integer(),
  element_name = character(),
  element_class = character(),
  element_type = character(),
  element_length = numeric(),
  element_dimensions = character(),
  element_size_bytes = numeric(),
  nested_names_preview = character(),
  list_elements_total = numeric(),
  list_elements_reported = numeric(),
  list_manifest_truncated = logical(),
  stringsAsFactors = FALSE
)

if (!file.exists(input_file)) {
  stop(
    paste(
      "Missing RData source:",
      input_file
    )
  )
}

log_message(
  "===== Phase 6B4A RData inspection started ====="
)

log_message(
  "Input: ",
  input_file
)

log_message(
  "Compressed file size: ",
  file.info(input_file)$size,
  " bytes"
)

workspace <- new.env(
  parent = emptyenv()
)

loaded_names <- load(
  input_file,
  envir = workspace
)

loaded_names <- sort(
  unique(
    loaded_names
  )
)

if (length(loaded_names) == 0L) {
  stop(
    "The RData file loaded but contained no objects."
  )
}

log_message(
  "Objects loaded: ",
  length(loaded_names)
)

log_message(
  "Object names: ",
  paste(
    loaded_names,
    collapse = ", "
  )
)

object_rows <- list()
column_rows <- list()
list_rows <- list()
structure_lines <- character()

for (object_index in seq_along(loaded_names)) {
  object_name <- loaded_names[[object_index]]

  object <- get(
    object_name,
    envir = workspace,
    inherits = FALSE
  )

  object_dimensions <- dim(object)

  number_of_rows <- if (
    is.null(object_dimensions)
  ) {
    NA_real_
  } else {
    as.numeric(object_dimensions[[1L]])
  }

  number_of_columns <- if (
    is.null(object_dimensions) ||
    length(object_dimensions) < 2L
  ) {
    NA_real_
  } else {
    as.numeric(object_dimensions[[2L]])
  }

  top_names <- names(object)

  object_rows[[length(object_rows) + 1L]] <- data.frame(
    object_name = object_name,
    object_class = clean_text(class(object)),
    object_type = typeof(object),
    object_mode = mode(object),
    object_length = length(object),
    object_dimensions = dimension_text(object),
    number_of_rows = number_of_rows,
    number_of_columns = number_of_columns,
    object_size_bytes = as.numeric(object.size(object)),
    object_size_megabytes = (
      as.numeric(object.size(object)) /
      1024^2
    ),
    is_data_frame = is.data.frame(object),
    is_matrix = is.matrix(object),
    is_list = is.list(object),
    is_environment = is.environment(object),
    is_S4 = isS4(object),
    top_level_names_count = if (
      is.null(top_names)
    ) {
      0L
    } else {
      length(top_names)
    },
    top_level_names_preview = if (
      is.null(top_names)
    ) {
      ""
    } else {
      preview_values(
        top_names,
        maximum = 30L
      )
    },
    row_names_preview = if (
      is.null(dim(object))
    ) {
      ""
    } else {
      preview_values(
        rownames(object),
        maximum = 10L
      )
    },
    stringsAsFactors = FALSE
  )

  structure_lines <- c(
    structure_lines,
    paste0(
      "============================================================"
    ),
    paste0(
      "OBJECT: ",
      object_name
    ),
    paste0(
      "CLASS: ",
      clean_text(class(object))
    ),
    paste0(
      "TYPE: ",
      typeof(object)
    ),
    paste0(
      "DIMENSIONS: ",
      dimension_text(object)
    ),
    paste0(
      "SIZE BYTES: ",
      as.numeric(object.size(object))
    ),
    "",
    capture.output(
      str(
        object,
        max.level = 3L,
        list.len = 30L,
        vec.len = 10L,
        give.attr = FALSE
      )
    ),
    ""
  )

  if (
    is.data.frame(object) ||
    is.matrix(object)
  ) {
    column_names <- colnames(object)

    if (is.null(column_names)) {
      column_names <- paste0(
        "V",
        seq_len(ncol(object))
      )
    }

    for (
      column_index in seq_len(ncol(object))
    ) {
      column <- object[
        ,
        column_index,
        drop = TRUE
      ]

      missing_values <- tryCatch(
        sum(is.na(column)),
        error = function(error) {
          NA_real_
        }
      )

      numeric_minimum <- NA_real_
      numeric_maximum <- NA_real_

      if (
        is.numeric(column) &&
        any(!is.na(column))
      ) {
        numeric_minimum <- suppressWarnings(
          min(
            column,
            na.rm = TRUE
          )
        )

        numeric_maximum <- suppressWarnings(
          max(
            column,
            na.rm = TRUE
          )
        )
      }

      column_rows[[length(column_rows) + 1L]] <- data.frame(
        object_name = object_name,
        column_index = column_index,
        column_name = clean_text(
          column_names[[column_index]]
        ),
        column_class = clean_text(
          class(column)
        ),
        column_type = typeof(column),
        column_length = length(column),
        missing_values = missing_values,
        nonmissing_values = if (
          is.na(missing_values)
        ) {
          NA_real_
        } else {
          length(column) - missing_values
        },
        first_values = preview_values(
          column,
          maximum = 8L
        ),
        numeric_minimum = numeric_minimum,
        numeric_maximum = numeric_maximum,
        stringsAsFactors = FALSE
      )
    }
  }

  if (
    is.list(object) &&
    !is.data.frame(object)
  ) {
    total_elements <- length(object)

    reporting_limit <- min(
      total_elements,
      2000L
    )

    element_names <- names(object)

    for (
      element_index in seq_len(reporting_limit)
    ) {
      element <- object[[element_index]]

      element_name <- if (
        is.null(element_names) ||
        is.na(element_names[[element_index]]) ||
        element_names[[element_index]] == ""
      ) {
        paste0(
          "[[",
          element_index,
          "]]"
        )
      } else {
        element_names[[element_index]]
      }

      nested_names <- names(element)

      list_rows[[length(list_rows) + 1L]] <- data.frame(
        object_name = object_name,
        element_index = element_index,
        element_name = clean_text(element_name),
        element_class = clean_text(class(element)),
        element_type = typeof(element),
        element_length = length(element),
        element_dimensions = dimension_text(element),
        element_size_bytes = as.numeric(
          object.size(element)
        ),
        nested_names_preview = if (
          is.null(nested_names)
        ) {
          ""
        } else {
          preview_values(
            nested_names,
            maximum = 20L
          )
        },
        list_elements_total = total_elements,
        list_elements_reported = reporting_limit,
        list_manifest_truncated = (
          total_elements >
          reporting_limit
        ),
        stringsAsFactors = FALSE
      )
    }
  }
}

object_manifest <- if (
  length(object_rows) == 0L
) {
  empty_object_manifest
} else {
  do.call(
    rbind,
    object_rows
  )
}

column_manifest <- if (
  length(column_rows) == 0L
) {
  empty_column_manifest
} else {
  do.call(
    rbind,
    column_rows
  )
}

list_manifest <- if (
  length(list_rows) == 0L
) {
  empty_list_manifest
} else {
  do.call(
    rbind,
    list_rows
  )
}

write.table(
  object_manifest,
  object_manifest_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

write.table(
  column_manifest,
  column_manifest_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

write.table(
  list_manifest,
  list_manifest_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

writeLines(
  structure_lines,
  structure_report_file
)

log_message("")
log_message(
  "===== RDATA OBJECT MANIFEST ====="
)

capture <- capture.output(
  print(
    object_manifest,
    row.names = FALSE
  )
)

for (line in capture) {
  log_message(line)
}

log_message("")
log_message(
  "Data-frame or matrix columns inventoried: ",
  nrow(column_manifest)
)

log_message(
  "Top-level list elements inventoried: ",
  nrow(list_manifest)
)

log_message(
  "Structure report: ",
  structure_report_file
)

log_message("")
log_message(
  "===== Phase 6B4A RData inspection completed ====="
)
