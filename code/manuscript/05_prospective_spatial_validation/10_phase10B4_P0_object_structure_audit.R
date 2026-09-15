options(
    stringsAsFactors = FALSE,
    warn = 1
)

project <- "."

spatial_root <- file.path(
    project,
    "12_validation_package",
    "phase10B_spatial_replication"
)

output <- file.path(
    spatial_root,
    "phase10B4_P0_preflight"
)

object_paths <- c(
    file.path(
        spatial_root,
        "phase10B2_primary_object_preflight",
        "01_raw_objects",
        "Final_Integration_MDM_100323.rds"
    ),
    file.path(
        spatial_root,
        "phase10B2_primary_object_preflight",
        "01_raw_objects",
        "Visium_A1_brain_011124.rds"
    )
)

safe_attr <- function(
    object,
    attribute_name
) {
    tryCatch(
        attr(
            object,
            attribute_name,
            exact = TRUE
        ),
        error = function(error) {
            NULL
        }
    )
}

safe_slot <- function(
    object,
    slot_name
) {
    tryCatch(
        {
            if (
                isS4(object) &&
                slot_name %in% methods::slotNames(object)
            ) {
                methods::slot(
                    object,
                    slot_name
                )
            } else {
                NULL
            }
        },
        error = function(error) {
            NULL
        }
    )
}

get_component <- function(
    object,
    component_name
) {
    value <- safe_attr(
        object,
        component_name
    )

    if (is.null(value)) {
        value <- safe_slot(
            object,
            component_name
        )
    }

    value
}

inventory_rows <- list()

for (object_path in object_paths) {

    object_label <- sub(
        "\\.rds$",
        "",
        basename(object_path)
    )

    if (!file.exists(object_path)) {

        inventory_rows[[length(inventory_rows) + 1L]] <- data.frame(
            object_label = object_label,
            assay = NA_character_,
            layer_or_slot = NA_character_,
            n_features = NA_integer_,
            n_units = NA_integer_,
            matrix_class = NA_character_,
            structural_access = FALSE,
            stringsAsFactors = FALSE
        )

        next
    }

    message(
        "Reading object structure: ",
        object_label
    )

    object <- readRDS(
        object_path
    )

    assays <- get_component(
        object,
        "assays"
    )

    assay_names <- tryCatch(
        names(assays),
        error = function(error) {
            character()
        }
    )

    if (length(assay_names) == 0L) {

        inventory_rows[[length(inventory_rows) + 1L]] <- data.frame(
            object_label = object_label,
            assay = NA_character_,
            layer_or_slot = NA_character_,
            n_features = NA_integer_,
            n_units = NA_integer_,
            matrix_class = class(object)[1],
            structural_access = FALSE,
            stringsAsFactors = FALSE
        )

    } else {

        for (assay_name in assay_names) {

            assay <- tryCatch(
                assays[[assay_name]],
                error = function(error) {
                    NULL
                }
            )

            candidates <- list()

            layers <- get_component(
                assay,
                "layers"
            )

            layer_names <- tryCatch(
                names(layers),
                error = function(error) {
                    character()
                }
            )

            if (length(layer_names) > 0L) {

                for (layer_name in layer_names) {

                    layer_value <- tryCatch(
                        layers[[layer_name]],
                        error = function(error) {
                            NULL
                        }
                    )

                    if (!is.null(layer_value)) {
                        candidates[[paste0("layer:", layer_name)]] <- layer_value
                    }
                }
            }

            for (
                slot_name in c(
                    "counts",
                    "data",
                    "scale.data"
                )
            ) {

                slot_value <- get_component(
                    assay,
                    slot_name
                )

                if (!is.null(slot_value)) {
                    candidates[[paste0("slot:", slot_name)]] <- slot_value
                }
            }

            if (length(candidates) == 0L) {

                inventory_rows[[length(inventory_rows) + 1L]] <- data.frame(
                    object_label = object_label,
                    assay = assay_name,
                    layer_or_slot = NA_character_,
                    n_features = NA_integer_,
                    n_units = NA_integer_,
                    matrix_class = class(assay)[1],
                    structural_access = FALSE,
                    stringsAsFactors = FALSE
                )

            } else {

                for (
                    candidate_name in names(candidates)
                ) {

                    matrix_object <- candidates[[candidate_name]]

                    matrix_dimensions <- tryCatch(
                        dim(matrix_object),
                        error = function(error) {
                            NULL
                        }
                    )

                    inventory_rows[[length(inventory_rows) + 1L]] <- data.frame(
                        object_label = object_label,
                        assay = assay_name,
                        layer_or_slot = candidate_name,
                        n_features = if (
                            length(matrix_dimensions) >= 1L
                        ) {
                            matrix_dimensions[1]
                        } else {
                            NA_integer_
                        },
                        n_units = if (
                            length(matrix_dimensions) >= 2L
                        ) {
                            matrix_dimensions[2]
                        } else {
                            NA_integer_
                        },
                        matrix_class = paste(
                            class(matrix_object),
                            collapse = ";"
                        ),
                        structural_access = (
                            length(matrix_dimensions) >= 2L
                        ),
                        stringsAsFactors = FALSE
                    )
                }
            }
        }
    }

    rm(
        object,
        assays
    )

    invisible(
        gc(
            verbose = FALSE
        )
    )
}

inventory <- do.call(
    rbind,
    inventory_rows
)

write.table(
    inventory,
    file.path(
        output,
        "phase10B4_P0_object_assay_inventory.tsv"
    ),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)

required_packages <- c(
    "Matrix",
    "SeuratObject",
    "Seurat",
    "data.table",
    "ggplot2"
)

package_inventory <- data.frame(
    package = required_packages,
    available = vapply(
        required_packages,
        requireNamespace,
        logical(1),
        quietly = TRUE
    ),
    version = vapply(
        required_packages,
        function(package_name) {

            if (
                requireNamespace(
                    package_name,
                    quietly = TRUE
                )
            ) {
                as.character(
                    utils::packageVersion(
                        package_name
                    )
                )
            } else {
                NA_character_
            }
        },
        character(1)
    ),
    stringsAsFactors = FALSE
)

write.table(
    package_inventory,
    file.path(
        output,
        "phase10B4_P0_R_package_inventory.tsv"
    ),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)

message(
    "Object structural audit completed."
)
