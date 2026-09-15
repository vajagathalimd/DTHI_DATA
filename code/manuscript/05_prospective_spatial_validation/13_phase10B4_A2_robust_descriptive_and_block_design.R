options(
    stringsAsFactors = FALSE,
    warn = 1
)

args <- commandArgs(
    trailingOnly = TRUE
)

module_summary_path <- args[1]
sn_score_path <- args[2]
visium_score_path <- args[3]
out_dir <- args[4]

suppressPackageStartupMessages(
    library(data.table)
)

sn_dir <- file.path(
    out_dir,
    "01_snRNAseq_descriptive"
)

visium_dir <- file.path(
    out_dir,
    "02_Visium_descriptive"
)

block_dir <- file.path(
    out_dir,
    "03_spatial_block_design"
)

audit_dir <- file.path(
    out_dir,
    "04_audit"
)

for (
    directory in c(
        sn_dir,
        visium_dir,
        block_dir,
        audit_dir
    )
) {
    dir.create(
        directory,
        recursive = TRUE,
        showWarnings = FALSE
    )
}

safe_trimmed_mean <- function(x) {
    mean(
        x,
        trim = 0.1,
        na.rm = TRUE
    )
}

safe_quantile <- function(
    x,
    probability
) {
    as.numeric(
        stats::quantile(
            x,
            probability,
            na.rm = TRUE,
            names = FALSE
        )
    )
}

make_quantile_bins <- function(
    values,
    requested_bins
) {
    breaks <- unique(
        as.numeric(
            stats::quantile(
                values,
                probabilities = seq(
                    0,
                    1,
                    length.out = requested_bins + 1L
                ),
                na.rm = TRUE,
                names = FALSE,
                type = 7
            )
        )
    )

    if (length(breaks) < 3L) {
        stop(
            paste0(
                "Insufficient unique coordinate breaks for ",
                requested_bins,
                " requested bins."
            )
        )
    }

    as.integer(
        cut(
            values,
            breaks = breaks,
            include.lowest = TRUE,
            labels = FALSE
        )
    )
}

cast_design_metric <- function(
    design_long,
    metric,
    suffix
) {
    formula <- stats::as.formula(
        paste0(
            "grid_k + depth_stratum ~ area_class"
        )
    )

    result <- dcast(
        design_long,
        formula,
        value.var = metric,
        fill = 0
    )

    for (
        area_name in c(
            "V1",
            "V2"
        )
    ) {
        if (!area_name %in% colnames(result)) {
            result[
                ,
                (area_name) := 0
            ]
        }
    }

    setnames(
        result,
        c(
            "V1",
            "V2"
        ),
        c(
            paste0(
                "V1_",
                suffix
            ),
            paste0(
                "V2_",
                suffix
            )
        )
    )

    result
}

module_summary <- fread(
    module_summary_path
)

if (!"module" %in% colnames(module_summary)) {
    stop(
        "The selected-module summary lacks a module column."
    )
}

modules <- sort(
    unique(
        as.character(
            module_summary$module
        )
    )
)

if (length(modules) != 9L) {
    stop(
        "Expected exactly nine locked DTHI modules."
    )
}

message(
    "Reading snRNA-seq module scores"
)

sn <- fread(
    sn_score_path
)

message(
    "Reading Visium module scores"
)

visium <- fread(
    visium_score_path
)

if (nrow(sn) != 91898L) {
    stop(
        paste0(
            "Unexpected snRNA-seq row count: ",
            nrow(sn)
        )
    )
}

if (nrow(visium) != 3591L) {
    stop(
        paste0(
            "Unexpected Visium row count: ",
            nrow(visium)
        )
    )
}

if (!all(modules %in% colnames(sn))) {
    stop(
        "At least one locked module is absent from the snRNA-seq score table."
    )
}

if (!all(modules %in% colnames(visium))) {
    stop(
        "At least one locked module is absent from the Visium score table."
    )
}

if (
    !all(
        is.finite(
            as.matrix(
                sn[
                    ,
                    ..modules
                ]
            )
        )
    )
) {
    stop(
        "Non-finite snRNA-seq module scores detected."
    )
}

if (
    !all(
        is.finite(
            as.matrix(
                visium[
                    ,
                    ..modules
                ]
            )
        )
    )
) {
    stop(
        "Non-finite Visium module scores detected."
    )
}

sn_id_columns <- c(
    "unit_id",
    "analysis_sample_id",
    "donor_id",
    "cortical_lobe",
    "cortical_area_context",
    "author_identity"
)

sn_long <- melt(
    sn,
    id.vars = sn_id_columns,
    measure.vars = modules,
    variable.name = "module",
    value.name = "score",
    variable.factor = FALSE
)

sn_sample_summary <- sn_long[
    ,
    .(
        nuclei = .N,
        mean_score = mean(score),
        sd_score = stats::sd(score),
        median_score = stats::median(score),
        trimmed_mean_score = safe_trimmed_mean(score),
        q25_score = safe_quantile(
            score,
            0.25
        ),
        q75_score = safe_quantile(
            score,
            0.75
        )
    ),
    by = .(
        analysis_sample_id,
        donor_id,
        cortical_lobe,
        cortical_area_context,
        module
    )
]

fwrite(
    sn_sample_summary,
    file.path(
        sn_dir,
        "phase10B4_A2_snRNAseq_sample_robust_summary.tsv"
    ),
    sep = "\t",
    quote = FALSE
)

sn_donor_region_summary <- sn_sample_summary[
    ,
    .(
        analysis_samples = .N,
        nuclei = sum(nuclei),
        equal_weight_sample_mean = mean(mean_score),
        equal_weight_sample_median = mean(median_score),
        equal_weight_trimmed_mean = mean(trimmed_mean_score)
    ),
    by = .(
        donor_id,
        cortical_lobe,
        cortical_area_context,
        module
    )
]

fwrite(
    sn_donor_region_summary,
    file.path(
        sn_dir,
        "phase10B4_A2_snRNAseq_donor_region_summary.tsv"
    ),
    sep = "\t",
    quote = FALSE
)

sn_region_for_cast <- sn_donor_region_summary[
    ,
    .(
        descriptive_score = mean(
            equal_weight_trimmed_mean
        ),
        donor_count = uniqueN(
            donor_id
        )
    ),
    by = .(
        module,
        cortical_lobe
    )
]

sn_region_scores <- dcast(
    sn_region_for_cast,
    module ~ cortical_lobe,
    value.var = "descriptive_score"
)

sn_region_donors <- dcast(
    sn_region_for_cast,
    module ~ cortical_lobe,
    value.var = "donor_count"
)

if (
    !"frontal" %in% colnames(sn_region_scores)
    || !"occipital" %in% colnames(sn_region_scores)
) {
    stop(
        "Both frontal and occipital descriptive summaries are required."
    )
}

setnames(
    sn_region_donors,
    c(
        "frontal",
        "occipital"
    ),
    c(
        "frontal_donors",
        "occipital_donors"
    )
)

sn_region_delta <- merge(
    sn_region_scores,
    sn_region_donors,
    by = "module",
    all = TRUE
)

sn_region_delta[
    ,
    frontal_minus_occipital := (
        frontal - occipital
    )
]

sn_region_delta[
    ,
    comparison_interpretation :=
        "descriptive_only_region_fully_confounded_by_donor"
]

sn_region_delta[
    ,
    inferential_test_performed := FALSE
]

fwrite(
    sn_region_delta,
    file.path(
        sn_dir,
        "phase10B4_A2_snRNAseq_region_descriptive_delta.tsv"
    ),
    sep = "\t",
    quote = FALSE
)

sn_identity_sample <- sn_long[
    ,
    .(
        nuclei = .N,
        mean_score = mean(score),
        median_score = stats::median(score),
        trimmed_mean_score = safe_trimmed_mean(score)
    ),
    by = .(
        analysis_sample_id,
        donor_id,
        cortical_lobe,
        author_identity,
        module
    )
]

sn_identity_sample_eligible <- sn_identity_sample[
    nuclei >= 50L
]

sn_identity_region <- sn_identity_sample_eligible[
    ,
    .(
        analysis_samples = .N,
        nuclei = sum(nuclei),
        equal_weight_trimmed_mean = mean(
            trimmed_mean_score
        )
    ),
    by = .(
        donor_id,
        cortical_lobe,
        author_identity,
        module
    )
]

sn_identity_cast <- dcast(
    sn_identity_region,
    author_identity + module ~ cortical_lobe,
    value.var = "equal_weight_trimmed_mean"
)

if (
    "frontal" %in% colnames(sn_identity_cast)
    && "occipital" %in% colnames(sn_identity_cast)
) {
    sn_identity_cast[
        ,
        frontal_minus_occipital := (
            frontal - occipital
        )
    ]

    sn_identity_cast[
        ,
        comparison_interpretation :=
            "identity_stratified_descriptive_only_donor_confounded"
    ]
}

fwrite(
    sn_identity_cast,
    file.path(
        sn_dir,
        "phase10B4_A2_snRNAseq_identity_region_descriptive_delta.tsv"
    ),
    sep = "\t",
    quote = FALSE
)

frontal_donors <- sort(
    unique(
        sn$cortical_lobe[
            sn$cortical_lobe == "frontal"
        ]
    )
)

frontal_donor_ids <- sort(
    unique(
        sn$donor_id[
            sn$cortical_lobe == "frontal"
        ]
    )
)

occipital_donor_ids <- sort(
    unique(
        sn$donor_id[
            sn$cortical_lobe == "occipital"
        ]
    )
)

shared_region_donors <- intersect(
    frontal_donor_ids,
    occipital_donor_ids
)

region_donor_confounding <- (
    length(shared_region_donors) == 0L
)

visium[
    ,
    area_class := fifelse(
        area_assignment %chin% c(
            "V1",
            "V2",
            "shared"
        ),
        area_assignment,
        "unresolved"
    )
]

visium[
    ,
    depth_stratum := fcase(
        broad_compartment == "VZ",
        "VZ",

        broad_compartment == "iSVZ",
        "iSVZ",

        broad_compartment == "oSVZ",
        "oSVZ",

        broad_compartment == "IZ",
        "IZ",

        broad_compartment == "SP",
        "SP",

        broad_compartment %chin% c(
            "CP_layer2",
            "CP_layer3",
            "CP_layers2_3"
        ),
        "superficial_CP",

        broad_compartment == "CP_layer4",
        "L4",

        broad_compartment == "CP_layers5_6",
        "deep_CP",

        default = "unresolved"
    )
]

visium_id_columns <- c(
    "unit_id",
    "donor_id",
    "author_cluster_id",
    "author_cluster_label",
    "broad_compartment",
    "depth_stratum",
    "area_assignment",
    "area_class",
    "cluster_annotation_mapped",
    "image_row",
    "image_col"
)

visium_long <- melt(
    visium,
    id.vars = visium_id_columns,
    measure.vars = modules,
    variable.name = "module",
    value.name = "score",
    variable.factor = FALSE
)

visium_area_depth_summary <- visium_long[
    ,
    .(
        spots = .N,
        mean_score = mean(score),
        sd_score = stats::sd(score),
        median_score = stats::median(score),
        trimmed_mean_score = safe_trimmed_mean(score),
        q25_score = safe_quantile(
            score,
            0.25
        ),
        q75_score = safe_quantile(
            score,
            0.75
        )
    ),
    by = .(
        area_class,
        depth_stratum,
        module
    )
]

fwrite(
    visium_area_depth_summary,
    file.path(
        visium_dir,
        "phase10B4_A2_Visium_area_depth_robust_summary.tsv"
    ),
    sep = "\t",
    quote = FALSE
)

visium_sensitivity <- rbindlist(
    list(
        visium_long[
            ,
            .(
                spots = .N,
                mean_score = mean(score),
                median_score = stats::median(score),
                trimmed_mean_score = safe_trimmed_mean(score)
            ),
            by = .(
                area_class,
                depth_stratum,
                module
            )
        ][
            ,
            analysis_set := "all_spots_cluster14_retained"
        ],

        visium_long[
            !(
                author_cluster_id == 14L
                & !cluster_annotation_mapped
            ),
            .(
                spots = .N,
                mean_score = mean(score),
                median_score = stats::median(score),
                trimmed_mean_score = safe_trimmed_mean(score)
            ),
            by = .(
                area_class,
                depth_stratum,
                module
            )
        ][
            ,
            analysis_set := "cluster14_excluded"
        ]
    ),
    use.names = TRUE
)

fwrite(
    visium_sensitivity,
    file.path(
        visium_dir,
        "phase10B4_A2_Visium_cluster14_sensitivity_summary.tsv"
    ),
    sep = "\t",
    quote = FALSE
)

block_unit_list <- list()
block_design_list <- list()

for (
    grid_k in c(
        6L,
        8L,
        10L
    )
) {
    current <- copy(
        visium
    )

    current[
        ,
        row_bin := make_quantile_bins(
            image_row,
            grid_k
        )
    ]

    current[
        ,
        col_bin := make_quantile_bins(
            image_col,
            grid_k
        )
    ]

    current[
        ,
        block_id := sprintf(
            "k%02d_r%02d_c%02d",
            grid_k,
            row_bin,
            col_bin
        )
    ]

    current[
        ,
        grid_k := grid_k
    ]

    current_long <- melt(
        current,
        id.vars = c(
            "unit_id",
            "grid_k",
            "block_id",
            "area_class",
            "depth_stratum",
            "author_cluster_id",
            "cluster_annotation_mapped"
        ),
        measure.vars = modules,
        variable.name = "module",
        value.name = "score",
        variable.factor = FALSE
    )

    units <- current_long[
        area_class %chin% c(
            "V1",
            "V2"
        )
        & cluster_annotation_mapped == TRUE
        & depth_stratum %chin% c(
            "SP",
            "superficial_CP",
            "L4",
            "deep_CP"
        ),
        .(
            spots = .N,
            mean_score = mean(score),
            median_score = stats::median(score),
            trimmed_mean_score = safe_trimmed_mean(score)
        ),
        by = .(
            grid_k,
            block_id,
            area_class,
            depth_stratum,
            module
        )
    ]

    block_unit_list[[length(block_unit_list) + 1L]] <- units

    unit_design <- unique(
        units[
            ,
            .(
                grid_k,
                block_id,
                area_class,
                depth_stratum,
                spots
            )
        ]
    )

    design <- unit_design[
        ,
        .(
            blocks = as.numeric(
                uniqueN(block_id)
            ),
            total_spots = as.numeric(
                sum(
                    as.numeric(spots)
                )
            ),
            median_spots_per_block = as.numeric(
                stats::median(
                    as.numeric(spots)
                )
            ),
            minimum_spots_per_block = as.numeric(
                min(
                    as.numeric(spots)
                )
            )
        ),
        by = .(
            grid_k,
            depth_stratum,
            area_class
        )
    ]

    block_design_list[[length(block_design_list) + 1L]] <- design
}

all_block_units <- rbindlist(
    block_unit_list,
    use.names = TRUE
)

block_design_long <- rbindlist(
    block_design_list,
    use.names = TRUE
)

blocks_wide <- cast_design_metric(
    block_design_long,
    "blocks",
    "blocks"
)

total_spots_wide <- cast_design_metric(
    block_design_long,
    "total_spots",
    "total_spots"
)

median_spots_wide <- cast_design_metric(
    block_design_long,
    "median_spots_per_block",
    "median_spots_per_block"
)

minimum_spots_wide <- cast_design_metric(
    block_design_long,
    "minimum_spots_per_block",
    "minimum_spots_per_block"
)

block_design_wide <- Reduce(
    function(x, y) {
        merge(
            x,
            y,
            by = c(
                "grid_k",
                "depth_stratum"
            ),
            all = TRUE
        )
    },
    list(
        blocks_wide,
        total_spots_wide,
        median_spots_wide,
        minimum_spots_wide
    )
)

block_design_wide[
    ,
    eligible_for_area_permutation := (
        V1_blocks >= 4L
        & V2_blocks >= 4L
        & V1_median_spots_per_block >= 5
        & V2_median_spots_per_block >= 5
    )
]

block_design_wide[
    ,
    common_area_blocks := pmin(
        V1_blocks,
        V2_blocks
    )
]

fwrite(
    block_design_wide,
    file.path(
        block_dir,
        "phase10B4_A2_grid_depth_eligibility.tsv"
    ),
    sep = "\t",
    quote = FALSE
)

grid_summary <- block_design_wide[
    ,
    .(
        eligible_depth_strata = sum(
            eligible_for_area_permutation
        ),
        minimum_common_area_blocks = min(
            common_area_blocks
        ),
        total_common_area_blocks = sum(
            common_area_blocks
        )
    ),
    by = grid_k
]

grid_summary[
    ,
    distance_from_default_grid := abs(
        grid_k - 8L
    )
]

setorder(
    grid_summary,
    -eligible_depth_strata,
    distance_from_default_grid,
    -minimum_common_area_blocks,
    -total_common_area_blocks
)

selected_grid_k <- grid_summary$grid_k[1]

grid_summary[
    ,
    selected_primary_grid := (
        grid_k == selected_grid_k
    )
]

fwrite(
    grid_summary,
    file.path(
        block_dir,
        "phase10B4_A2_grid_selection_summary.tsv"
    ),
    sep = "\t",
    quote = FALSE
)

selected_units <- all_block_units[
    grid_k == selected_grid_k
]

fwrite(
    selected_units,
    file.path(
        block_dir,
        "phase10B4_A2_selected_spatial_block_units.tsv"
    ),
    sep = "\t",
    quote = FALSE
)

selected_eligible_strata <- block_design_wide[
    grid_k == selected_grid_k
    & eligible_for_area_permutation == TRUE
]

status_value <- if (
    length(modules) == 9L
    && nrow(sn) == 91898L
    && nrow(visium) == 3591L
    && region_donor_confounding
    && nrow(selected_eligible_strata) >= 2L
) {
    "passed_phase10B4_A2_robust_descriptive_summaries_and_spatial_block_design_ready_for_within_section_permutation_testing"
} else {
    "phase10B4_A2_requires_manual_review"
}

status <- data.frame(
    phase = "phase10B4_A2",
    locked_modules = length(modules),
    snRNAseq_nuclei_summarised = nrow(sn),
    Visium_spots_summarised = nrow(visium),
    snRNAseq_region_comparison_donor_confounded = region_donor_confounding,
    snRNAseq_inferential_tests_performed = FALSE,
    Visium_cluster14_retained_in_descriptive_summary = TRUE,
    Visium_cluster14_excluded_from_block_testing_design = TRUE,
    candidate_grid_sizes = "6;8;10",
    selected_primary_grid = selected_grid_k,
    eligible_depth_strata_for_selected_grid = nrow(
        selected_eligible_strata
    ),
    spatial_permutation_tests_performed = FALSE,
    candidate_TFs_changed = FALSE,
    validation_hypotheses_changed = FALSE,
    phase10B4_A2_status = status_value,
    stringsAsFactors = FALSE
)

fwrite(
    status,
    file.path(
        out_dir,
        "phase10B4_A2_status.tsv"
    ),
    sep = "\t",
    quote = FALSE
)

report <- c(
    "===== PHASE 10B4-A2 ROBUST DESCRIPTIVE AND BLOCK-DESIGN PREFLIGHT =====",
    "",
    paste0(
        "Locked modules summarised: ",
        length(modules)
    ),
    paste0(
        "snRNA-seq nuclei summarised: ",
        nrow(sn)
    ),
    paste0(
        "Visium spots summarised: ",
        nrow(visium)
    ),
    paste0(
        "snRNA-seq region comparison donor-confounded: ",
        region_donor_confounding
    ),
    "snRNA-seq inferential tests performed: FALSE",
    "Visium cluster 14 retained in descriptive summaries: TRUE",
    "Visium cluster 14 excluded from block-testing design: TRUE",
    "Candidate grid sizes: 6;8;10",
    paste0(
        "Selected primary grid: ",
        selected_grid_k
    ),
    paste0(
        "Eligible depth strata for selected grid: ",
        nrow(selected_eligible_strata)
    ),
    "Spatial permutation tests performed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    paste0(
        "PHASE 10B4-A2 STATUS: ",
        status_value
    )
)

writeLines(
    report,
    file.path(
        out_dir,
        "phase10B4_A2_report.txt"
    )
)

cat(
    paste(
        report,
        collapse = "\n"
    ),
    "\n"
)
