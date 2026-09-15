#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import hashlib
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(
    "."
)

F3_FILE = (
    PROJECT
    / "07_tables/main_tables/phase6/"
      "phase6F3_final_biological_classification.tsv"
)

F3_COMPLETION_FILE = (
    PROJECT
    / "07_tables/main_tables/phase6/"
      "phase6F3_completion_summary.tsv"
)

F4_FILE = (
    PROJECT
    / "07_tables/main_tables/phase6/"
      "phase6F4_chemical_robustness_summary.tsv"
)

F4_COMPLETION_FILE = (
    PROJECT
    / "07_tables/main_tables/phase6/"
      "phase6F4_completion_summary.tsv"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase6"
)

PROCESSED_DIR = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6F"
)

FIGURE_DIR = (
    PROJECT
    / "08_figures/main_figures/phase6"
)

LOG_DIR = (
    PROJECT
    / "09_pipeline_logs/phase6"
)


ATLAS_OUTPUT = (
    TABLE_DIR
    / "phase6F5_final_chemical_evidence_atlas.tsv"
)

ATLAS_GZ_OUTPUT = (
    PROCESSED_DIR
    / "phase6F5_final_chemical_evidence_atlas.tsv.gz"
)

REPORTING_OUTPUT = (
    TABLE_DIR
    / "phase6F5_manuscript_reporting_table.tsv"
)

CLASS_SUMMARY_OUTPUT = (
    TABLE_DIR
    / "phase6F5_final_reporting_group_summary.tsv"
)

KEY_FINDINGS_OUTPUT = (
    TABLE_DIR
    / "phase6F5_phase6_key_findings.tsv"
)

METHODS_OUTPUT = (
    TABLE_DIR
    / "phase6F5_methods_summary.txt"
)

RESULTS_OUTPUT = (
    TABLE_DIR
    / "phase6F5_results_summary.txt"
)

FIGURE_PNG = (
    FIGURE_DIR
    / "Figure_47_integrated_perturbational_evidence_atlas.png"
)

FIGURE_PDF = (
    FIGURE_DIR
    / "Figure_47_integrated_perturbational_evidence_atlas.pdf"
)

FIGURE_CAPTION = (
    FIGURE_DIR
    / "Figure_47_caption.txt"
)

SUPPLEMENT_LEGEND = (
    TABLE_DIR
    / "phase6F5_supplementary_table_legend.txt"
)

MANIFEST_OUTPUT = (
    TABLE_DIR
    / "phase6F5_core_output_manifest.tsv"
)

COMPLETION_OUTPUT = (
    TABLE_DIR
    / "phase6F5_completion_summary.tsv"
)

LOG_FILE = (
    LOG_DIR
    / "phase6F5_finalize_phase6_evidence_atlas.log"
)


for directory in [
    TABLE_DIR,
    PROCESSED_DIR,
    FIGURE_DIR,
    LOG_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


EXPECTED_TOTAL = 29
EXPECTED_DIRECT = 15
EXPECTED_CMAP_ONLY = 14
EXPECTED_GLOBAL_RESIDUAL = 6
EXPECTED_REPLICATE_EVALUABLE = 14


def parse_bool(
    value: object,
) -> bool:
    if isinstance(
        value,
        bool,
    ):
        return value

    if pd.isna(
        value
    ):
        return False

    return str(
        value
    ).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def assert_unique(
    table: pd.DataFrame,
    column: str,
    label: str,
) -> None:
    values = (
        table[
            column
        ]
        .dropna()
        .astype(
            str
        )
        .str.strip()
    )

    duplicated = values.loc[
        values.duplicated(
            keep=False
        )
    ]

    if not duplicated.empty:
        raise RuntimeError(
            (
                f"{label} contains duplicate {column}: "
                + "|".join(
                    duplicated.unique()
                )
            )
        )


def reporting_group(
    row: pd.Series,
) -> str:
    direct = parse_bool(
        row[
            "direct_LINCS_quantitative_evaluated"
        ]
    )

    biological_class = str(
        row[
            "final_biological_class"
        ]
    ).lower()

    global_residual = parse_bool(
        row[
            "CMAP_global_residual_mechanism"
        ]
    )

    if not direct:
        if global_residual:
            return "CMAP_only_global_residual_mechanism"

        return "CMAP_only_provisional_or_unresolved"

    if "injury" in biological_class:
        return "direct_injury_or_stress_perturbation"

    if (
        "late_program_depletion" in biological_class
        or "early_program_suppression" in biological_class
        or "negative_developmental" in biological_class
    ):
        return "direct_developmental_program_disruption"

    if (
        "late_program_support" in biological_class
        or "positive_developmental" in biological_class
    ):
        return "direct_late_or_positive_shift_without_benefit_inference"

    if (
        "null_relative" in biological_class
        or "without_supported_absolute_effect" in biological_class
    ):
        return "direct_no_supported_absolute_effect"

    return "direct_other_developmental_effect"


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def main() -> None:
    required_inputs = [
        F3_FILE,
        F3_COMPLETION_FILE,
        F4_FILE,
        F4_COMPLETION_FILE,
    ]

    missing = [
        str(
            path
        )
        for path in required_inputs
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing Phase 6F5 inputs:\n"
            + "\n".join(
                missing
            )
        )

    classification = pd.read_csv(
        F3_FILE,
        sep="\t",
        low_memory=False,
    )

    robustness = pd.read_csv(
        F4_FILE,
        sep="\t",
        low_memory=False,
    )

    f3_completion = pd.read_csv(
        F3_COMPLETION_FILE,
        sep="\t",
    )

    f4_completion = pd.read_csv(
        F4_COMPLETION_FILE,
        sep="\t",
    )

    if len(
        classification
    ) != EXPECTED_TOTAL:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_TOTAL} classified chemicals, "
                f"found {len(classification)}."
            )
        )

    if len(
        robustness
    ) != EXPECTED_TOTAL:
        raise RuntimeError(
            (
                f"Expected {EXPECTED_TOTAL} robustness rows, "
                f"found {len(robustness)}."
            )
        )

    assert_unique(
        classification,
        "chemical_name",
        "Phase 6F3 classification",
    )

    assert_unique(
        robustness,
        "chemical_name",
        "Phase 6F4 robustness",
    )

    required_classification_columns = [
        "chemical_name",
        "DTXSID",
        "final_biological_class",
        "biological_interpretation_statement",
        "final_evidence_strength",
        "reproducibility_evidence_level",
        "classification_claim_scope",
        "mechanistic_priority_scope",
        "mechanistic_priority_rank_within_scope",
        "mechanistic_priority_score_within_scope",
        "mechanistic_priority_class",
        "CMAP_provisional_priority_score",
        "direct_LINCS_hazard_priority_score",
        "integrated_mechanistic_priority_score",
        "evidence_confidence_rank",
        "evidence_confidence_score",
        "evidence_confidence_class",
        "evidence_tier",
        "dominant_robust_mechanism_class",
        "overall_primary_interpretation",
        "direct_LINCS_quantitative_evaluated",
        "CMAP_global_residual_mechanism",
        "neural_quantitative_data_available",
        "replicate_concordant_signatures",
        "primary_absolute_bootstrap_replicated_count",
        "direct_injury_activation",
        "direct_injury_exceeds_late_program",
        "direct_early_program_suppression",
        "direct_late_maturation_depletion",
        "direct_late_maturation_support",
        "direct_negative_developmental_shift",
        "direct_positive_developmental_shift",
    ]

    missing_columns = [
        column
        for column in required_classification_columns
        if column not in classification.columns
    ]

    if missing_columns:
        raise RuntimeError(
            (
                "Missing classification columns: "
                + "|".join(
                    missing_columns
                )
            )
        )

    robustness_columns = [
        "chemical_name",
        "mechanistic_rank_robustness_class",
        "confidence_rank_robustness_class",
        "Phase6F4_overall_robustness",
        "rank_range_mechanistic",
        "rank_range_confidence",
        "stable_top5_all_mechanistic_scenarios",
        "stable_top10_all_confidence_scenarios",
    ]

    missing_robustness_columns = [
        column
        for column in robustness_columns
        if column not in robustness.columns
    ]

    if missing_robustness_columns:
        raise RuntimeError(
            (
                "Missing robustness columns: "
                + "|".join(
                    missing_robustness_columns
                )
            )
        )

    atlas = classification.merge(
        robustness[
            robustness_columns
        ],
        on="chemical_name",
        how="left",
        validate="one_to_one",
    )

    boolean_columns = [
        "direct_LINCS_quantitative_evaluated",
        "CMAP_global_residual_mechanism",
        "neural_quantitative_data_available",
        "direct_injury_activation",
        "direct_injury_exceeds_late_program",
        "direct_early_program_suppression",
        "direct_late_maturation_depletion",
        "direct_late_maturation_support",
        "direct_negative_developmental_shift",
        "direct_positive_developmental_shift",
        "stable_top5_all_mechanistic_scenarios",
        "stable_top10_all_confidence_scenarios",
    ]

    for column in boolean_columns:
        atlas[
            column
        ] = atlas[
            column
        ].map(
            parse_bool
        )

    atlas[
        "replicate_concordant_evaluable"
    ] = (
        pd.to_numeric(
            atlas[
                "replicate_concordant_signatures"
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
        .gt(
            0
        )
    )

    atlas[
        "bootstrap_replicated_absolute_effect"
    ] = (
        pd.to_numeric(
            atlas[
                "primary_absolute_bootstrap_replicated_count"
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
        .gt(
            0
        )
    )

    atlas[
        "final_reporting_group"
    ] = atlas.apply(
        reporting_group,
        axis=1,
    )

    atlas[
        "priority_scope_warning"
    ] = np.where(
        atlas[
            "direct_LINCS_quantitative_evaluated"
        ],
        (
            "Integrated priority combines CMAP and direct quantitative "
            "LINCS evidence."
        ),
        (
            "CMAP-only provisional priority is not directly comparable "
            "with the integrated priority ranking."
        ),
    )

    atlas[
        "final_interpretive_guardrail"
    ] = (
        "Results describe transcriptomic perturbational mechanisms and "
        "evidence strength; they do not establish therapeutic benefit, "
        "clinical toxicity, causal human risk or exposure relevance."
    )

    direct_mask = atlas[
        "direct_LINCS_quantitative_evaluated"
    ]

    cmap_only_mask = ~direct_mask

    if not atlas.loc[
        cmap_only_mask,
        "direct_LINCS_hazard_priority_score",
    ].isna().all():
        raise RuntimeError(
            "CMAP-only chemicals contain direct hazard scores."
        )

    if not atlas.loc[
        cmap_only_mask,
        "integrated_mechanistic_priority_score",
    ].isna().all():
        raise RuntimeError(
            "CMAP-only chemicals contain integrated priority scores."
        )

    scope_order = {
        "integrated_CMAP_plus_direct_LINCS":
            0,

        "CMAP_only_provisional":
            1,
    }

    atlas[
        "_scope_order"
    ] = atlas[
        "mechanistic_priority_scope"
    ].map(
        scope_order
    )

    atlas = atlas.sort_values(
        [
            "_scope_order",
            "mechanistic_priority_rank_within_scope",
            "evidence_confidence_rank",
            "chemical_name",
        ],
        ascending=[
            True,
            True,
            True,
            True,
        ],
    ).drop(
        columns=[
            "_scope_order",
        ]
    ).reset_index(
        drop=True
    )

    preferred_columns = [
        "chemical_name",
        "DTXSID",
        "final_reporting_group",
        "final_biological_class",
        "biological_interpretation_statement",
        "final_evidence_strength",
        "reproducibility_evidence_level",
        "Phase6F4_overall_robustness",

        "mechanistic_priority_scope",
        "mechanistic_priority_rank_within_scope",
        "mechanistic_priority_score_within_scope",
        "mechanistic_priority_class",
        "evidence_confidence_rank",
        "evidence_confidence_score",
        "evidence_confidence_class",

        "CMAP_provisional_priority_score",
        "direct_LINCS_hazard_priority_score",
        "integrated_mechanistic_priority_score",

        "evidence_tier",
        "dominant_robust_mechanism_class",
        "overall_primary_interpretation",

        "direct_LINCS_quantitative_evaluated",
        "CMAP_global_residual_mechanism",
        "neural_quantitative_data_available",
        "replicate_concordant_evaluable",
        "bootstrap_replicated_absolute_effect",

        "direct_injury_activation",
        "direct_injury_exceeds_late_program",
        "direct_early_program_suppression",
        "direct_late_maturation_depletion",
        "direct_late_maturation_support",
        "direct_negative_developmental_shift",
        "direct_positive_developmental_shift",

        "mechanistic_rank_robustness_class",
        "confidence_rank_robustness_class",
        "rank_range_mechanistic",
        "rank_range_confidence",
        "stable_top5_all_mechanistic_scenarios",
        "stable_top10_all_confidence_scenarios",

        "classification_claim_scope",
        "priority_scope_warning",
        "final_interpretive_guardrail",
    ]

    atlas_columns = [
        column
        for column in preferred_columns
        if column in atlas.columns
    ]

    atlas = atlas[
        atlas_columns
    ]

    atlas.to_csv(
        ATLAS_OUTPUT,
        sep="\t",
        index=False,
    )

    atlas.to_csv(
        ATLAS_GZ_OUTPUT,
        sep="\t",
        index=False,
        compression="gzip",
    )

    reporting_columns = [
        "chemical_name",
        "DTXSID",
        "final_reporting_group",
        "final_biological_class",
        "final_evidence_strength",
        "Phase6F4_overall_robustness",
        "mechanistic_priority_scope",
        "mechanistic_priority_rank_within_scope",
        "mechanistic_priority_score_within_scope",
        "evidence_confidence_rank",
        "evidence_confidence_score",
        "evidence_confidence_class",
        "biological_interpretation_statement",
        "priority_scope_warning",
        "final_interpretive_guardrail",
    ]

    atlas[
        reporting_columns
    ].to_csv(
        REPORTING_OUTPUT,
        sep="\t",
        index=False,
    )

    class_summary = (
        atlas.groupby(
            [
                "final_reporting_group",
                "mechanistic_priority_scope",
            ],
            dropna=False,
            sort=True,
        )
        .agg(
            chemical_count=(
                "chemical_name",
                "size",
            ),
            median_priority_score=(
                "mechanistic_priority_score_within_scope",
                "median",
            ),
            median_confidence_score=(
                "evidence_confidence_score",
                "median",
            ),
            bootstrap_replicated_chemicals=(
                "bootstrap_replicated_absolute_effect",
                "sum",
            ),
            chemical_names=(
                "chemical_name",
                lambda values: "|".join(
                    sorted(
                        values.astype(
                            str
                        )
                    )
                ),
            ),
        )
        .reset_index()
    )

    class_summary.to_csv(
        CLASS_SUMMARY_OUTPUT,
        sep="\t",
        index=False,
    )

    direct_count = int(
        direct_mask.sum()
    )

    cmap_only_count = int(
        cmap_only_mask.sum()
    )

    global_residual_count = int(
        atlas[
            "CMAP_global_residual_mechanism"
        ].sum()
    )

    replicate_evaluable_count = int(
        atlas[
            "replicate_concordant_evaluable"
        ].sum()
    )

    injury_count = int(
        f3_completion.loc[
            0,
            "injury_related_classifications",
        ]
    )

    late_depletion_count = int(
        f3_completion.loc[
            0,
            "late_program_depletion_classifications",
        ]
    )

    late_support_count = int(
        f3_completion.loc[
            0,
            "late_program_support_classifications",
        ]
    )

    null_or_no_absolute_count = int(
        f3_completion.loc[
            0,
            (
                "null_relative_or_no_absolute_effect_"
                "classifications"
            ),
        ]
    )

    minimum_mechanistic_spearman = float(
        f4_completion.loc[
            0,
            "minimum_mechanistic_Spearman",
        ]
    )

    minimum_confidence_spearman = float(
        f4_completion.loc[
            0,
            "minimum_confidence_Spearman",
        ]
    )

    stable_top5_count = int(
        f4_completion.loc[
            0,
            (
                "chemicals_stable_in_top5_all_"
                "mechanistic_scenarios"
            ),
        ]
    )

    stable_top10_count = int(
        f4_completion.loc[
            0,
            (
                "chemicals_stable_in_top10_all_"
                "confidence_scenarios"
            ),
        ]
    )

    key_findings = pd.DataFrame(
        [
            {
                "finding":
                    "Total chemicals integrated",

                "value":
                    EXPECTED_TOTAL,

                "interpretation":
                    (
                        "Directional CMAP and quantitative LINCS evidence "
                        "were integrated without treating missing profiles "
                        "as negative biological evidence."
                    ),
            },
            {
                "finding":
                    "Direct quantitative LINCS chemicals",

                "value":
                    direct_count,

                "interpretation":
                    (
                        "These chemicals received integrated mechanistic "
                        "priority scores."
                    ),
            },
            {
                "finding":
                    "CMAP-only provisional chemicals",

                "value":
                    cmap_only_count,

                "interpretation":
                    (
                        "These chemicals were retained in a separate "
                        "provisional ranking."
                    ),
            },
            {
                "finding":
                    "CMAP global residual chemicals",

                "value":
                    global_residual_count,

                "interpretation":
                    (
                        "Residual fixed-panel mechanisms remained after "
                        "controlling for developmental-program overlap."
                    ),
            },
            {
                "finding":
                    "Injury-related classifications",

                "value":
                    injury_count,

                "interpretation":
                    (
                        "Injury or stress was the most frequent resolved "
                        "adverse biological interpretation."
                    ),
            },
            {
                "finding":
                    "Late-program depletion classifications",

                "value":
                    late_depletion_count,

                "interpretation":
                    (
                        "Absolute depletion of late maturation programs "
                        "was uncommon but detectable."
                    ),
            },
            {
                "finding":
                    "Late-program support classifications",

                "value":
                    late_support_count,

                "interpretation":
                    (
                        "Late-program support was not interpreted as "
                        "therapeutic or beneficial maturation."
                    ),
            },
            {
                "finding":
                    "Replicate-concordant evaluable chemicals",

                "value":
                    replicate_evaluable_count,

                "interpretation":
                    (
                        "These chemicals retained sufficient profiles for "
                        "the reproducibility-filtered sensitivity analysis."
                    ),
            },
            {
                "finding":
                    "Minimum mechanistic rank Spearman",

                "value":
                    minimum_mechanistic_spearman,

                "interpretation":
                    (
                        "The minimum occurred under an extreme single-layer "
                        "weighting scenario."
                    ),
            },
            {
                "finding":
                    "Minimum confidence rank Spearman",

                "value":
                    minimum_confidence_spearman,

                "interpretation":
                    (
                        "Confidence rankings remained strongly correlated "
                        "after removing individual evidence layers."
                    ),
            },
            {
                "finding":
                    "Stable top-five mechanistic chemicals",

                "value":
                    stable_top5_count,

                "interpretation":
                    (
                        "These chemicals remained in the top five across "
                        "all mechanistic weighting scenarios."
                    ),
            },
            {
                "finding":
                    "Stable top-ten confidence chemicals",

                "value":
                    stable_top10_count,

                "interpretation":
                    (
                        "These chemicals remained in the top ten after "
                        "all leave-one-layer-out confidence analyses."
                    ),
            },
        ]
    )

    key_findings.to_csv(
        KEY_FINDINGS_OUTPUT,
        sep="\t",
        index=False,
    )

    # ================================================================
    # Figure 47
    # ================================================================

    direct = atlas.loc[
        atlas[
            "direct_LINCS_quantitative_evaluated"
        ]
    ].copy()

    provisional = atlas.loc[
        ~atlas[
            "direct_LINCS_quantitative_evaluated"
        ]
    ].copy()

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(
            20,
            16,
        ),
    )

    ax_a = axes[
        0,
        0,
    ]

    ax_a.scatter(
        direct[
            "integrated_mechanistic_priority_score"
        ],
        direct[
            "evidence_confidence_score"
        ],
        s=90,
    )

    label_direct = set(
        direct.nlargest(
            5,
            "integrated_mechanistic_priority_score",
        )[
            "chemical_name"
        ]
    ) | set(
        direct.nlargest(
            5,
            "evidence_confidence_score",
        )[
            "chemical_name"
        ]
    )

    for _, row in direct.iterrows():
        if row[
            "chemical_name"
        ] in label_direct:
            ax_a.annotate(
                row[
                    "chemical_name"
                ],
                (
                    row[
                        "integrated_mechanistic_priority_score"
                    ],
                    row[
                        "evidence_confidence_score"
                    ],
                ),
                xytext=(
                    4,
                    4,
                ),
                textcoords="offset points",
                fontsize=8,
            )

    ax_a.set_xlabel(
        "Integrated mechanistic perturbation priority"
    )

    ax_a.set_ylabel(
        "Evidence confidence"
    )

    ax_a.set_title(
        "A. Direct quantitative LINCS chemicals"
    )

    ax_a.set_xlim(
        left=0
    )

    ax_a.set_ylim(
        0,
        100,
    )

    ax_b = axes[
        0,
        1,
    ]

    ax_b.scatter(
        provisional[
            "CMAP_provisional_priority_score"
        ],
        provisional[
            "evidence_confidence_score"
        ],
        s=90,
    )

    label_provisional = set(
        provisional.nlargest(
            5,
            "CMAP_provisional_priority_score",
        )[
            "chemical_name"
        ]
    )

    for _, row in provisional.iterrows():
        if row[
            "chemical_name"
        ] in label_provisional:
            ax_b.annotate(
                row[
                    "chemical_name"
                ],
                (
                    row[
                        "CMAP_provisional_priority_score"
                    ],
                    row[
                        "evidence_confidence_score"
                    ],
                ),
                xytext=(
                    4,
                    4,
                ),
                textcoords="offset points",
                fontsize=8,
            )

    ax_b.set_xlabel(
        "CMAP-only provisional priority"
    )

    ax_b.set_ylabel(
        "Evidence confidence"
    )

    ax_b.set_title(
        "B. Chemicals without direct quantitative LINCS profiles"
    )

    ax_b.set_xlim(
        left=0
    )

    ax_b.set_ylim(
        0,
        100,
    )

    evidence_columns = [
        "CMAP_global_residual_mechanism",
        "direct_LINCS_quantitative_evaluated",
        "neural_quantitative_data_available",
        "replicate_concordant_evaluable",
        "bootstrap_replicated_absolute_effect",
    ]

    evidence_labels = [
        "CMAP residual",
        "Direct LINCS",
        "Neural context",
        "Replicate subset",
        "Bootstrap effect",
    ]

    evidence_matrix = (
        atlas[
            evidence_columns
        ]
        .astype(
            int
        )
        .to_numpy()
    )

    ax_c = axes[
        1,
        0,
    ]

    image = ax_c.imshow(
        evidence_matrix,
        aspect="auto",
        interpolation="nearest",
    )

    ax_c.set_xticks(
        np.arange(
            len(
                evidence_labels
            )
        )
    )

    ax_c.set_xticklabels(
        evidence_labels,
        rotation=35,
        ha="right",
    )

    ax_c.set_yticks(
        np.arange(
            len(
                atlas
            )
        )
    )

    ax_c.set_yticklabels(
        atlas[
            "chemical_name"
        ],
        fontsize=7,
    )

    ax_c.set_title(
        "C. Evidence-layer availability and replication"
    )

    figure.colorbar(
        image,
        ax=ax_c,
        ticks=[
            0,
            1,
        ],
        label="Evidence absent/present",
    )

    class_counts = (
        atlas[
            "final_reporting_group"
        ]
        .value_counts()
        .sort_values(
            ascending=True
        )
    )

    ax_d = axes[
        1,
        1,
    ]

    ax_d.barh(
        class_counts.index,
        class_counts.values,
    )

    ax_d.set_xlabel(
        "Number of chemicals"
    )

    ax_d.set_title(
        "D. Final manuscript reporting groups"
    )

    for position, value in enumerate(
        class_counts.values
    ):
        ax_d.text(
            value + 0.1,
            position,
            str(
                value
            ),
            va="center",
        )

    figure.suptitle(
        (
            "Figure 47. Integrated perturbational evidence atlas for "
            "developmental transcriptomic programs"
        ),
        fontsize=16,
    )

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97,
        ]
    )

    figure.savefig(
        FIGURE_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_PDF,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    caption = (
        "Figure 47. Integrated perturbational evidence atlas for "
        "developmental transcriptomic programs. (A) Integrated mechanistic "
        "perturbation priority versus evidence confidence for the 15 "
        "chemicals with direct quantitative LINCS Level 5 profiles. "
        "(B) Separate CMAP-only provisional priority versus evidence "
        "confidence for the 14 chemicals lacking exact matched quantitative "
        "profiles; these scores are not directly comparable with panel A. "
        "(C) Availability of global residual CMAP evidence, direct LINCS "
        "profiles, neural-context profiles, replicate-concordant profiles "
        "and bootstrap-replicated absolute effects. Chemicals are ordered "
        "first by evidence scope and then by mechanistic-priority rank. "
        "(D) Distribution of final manuscript reporting groups. Priority "
        "represents developmental perturbation rather than therapeutic "
        "benefit, and absence of a quantitative profile is not interpreted "
        "as absence of biological activity."
    )

    FIGURE_CAPTION.write_text(
        caption + "\n",
        encoding="utf-8",
    )

    methods_text = (
        "Phase 6 integrated directional CMAP gene-set concordance with "
        "quantitative GSE70138 LINCS Level 5 perturbational profiles. "
        "Mechanistic perturbation priority and evidence confidence were "
        "calculated independently. Chemicals lacking an exact matched "
        "quantitative LINCS profile were retained in a separate CMAP-only "
        "provisional ranking, with quantitative fields preserved as missing "
        "rather than assigned zero. Biological classes were based on "
        "absolute injury/stress, early-program suppression, late-program "
        "depletion, developmental-state shift and null-relative evidence. "
        "Robustness was evaluated using alternative CMAP–LINCS weightings, "
        "leave-one-evidence-layer-out confidence models and exclusion of "
        "profiles lacking replicate-concordant support."
    )

    METHODS_OUTPUT.write_text(
        methods_text + "\n",
        encoding="utf-8",
    )

    results_text = (
        f"Twenty-nine chemicals were integrated, including {direct_count} "
        f"with direct quantitative LINCS evidence and {cmap_only_count} "
        f"retained as CMAP-only provisional chemicals. "
        f"{global_residual_count} chemicals showed a global residual CMAP "
        f"mechanism. Final classifications included {injury_count} "
        f"injury-related profiles, {late_depletion_count} late-program "
        f"depletion profiles, {late_support_count} late-program support "
        f"profiles and {null_or_no_absolute_count} null-relative or "
        f"unsupported absolute-effect profiles. Balanced mechanistic "
        f"weighting scenarios were highly concordant, while the minimum "
        f"Spearman correlation of {minimum_mechanistic_spearman:.3f} arose "
        f"under an extreme single-layer scenario. Evidence-confidence "
        f"rankings remained stable across leave-one-layer-out analyses "
        f"(minimum Spearman {minimum_confidence_spearman:.3f}). No chemical "
        f"was classified as therapeutic or beneficial."
    )

    RESULTS_OUTPUT.write_text(
        results_text + "\n",
        encoding="utf-8",
    )

    supplement_legend = (
        "Supplementary Table. Final chemical-level perturbational evidence "
        "atlas. The table reports biological class, evidence strength, "
        "mechanistic-priority scope, evidence confidence, robustness, "
        "quantitative evidence availability and interpretation guardrails "
        "for all 29 chemicals. Integrated and CMAP-only provisional priority "
        "scores must be interpreted within their respective scopes."
    )

    SUPPLEMENT_LEGEND.write_text(
        supplement_legend + "\n",
        encoding="utf-8",
    )

    figure_created = bool(
        FIGURE_PNG.exists()
        and FIGURE_PDF.exists()
        and FIGURE_CAPTION.exists()
    )

    status = (
        "completed"
        if (
            len(
                atlas
            )
            == EXPECTED_TOTAL
            and direct_count
            == EXPECTED_DIRECT
            and cmap_only_count
            == EXPECTED_CMAP_ONLY
            and global_residual_count
            == EXPECTED_GLOBAL_RESIDUAL
            and replicate_evaluable_count
            == EXPECTED_REPLICATE_EVALUABLE
            and figure_created
            and atlas[
                "final_biological_class"
            ].notna().all()
        )
        else "failed_validation"
    )

    completion = pd.DataFrame(
        [
            {
                "final_atlas_chemicals":
                    len(
                        atlas
                    ),

                "direct_quantitative_chemicals":
                    direct_count,

                "CMAP_only_provisional_chemicals":
                    cmap_only_count,

                "CMAP_global_residual_chemicals":
                    global_residual_count,

                "replicate_concordant_evaluable_chemicals":
                    replicate_evaluable_count,

                "unique_biological_classes":
                    atlas[
                        "final_biological_class"
                    ].nunique(),

                "unique_reporting_groups":
                    atlas[
                        "final_reporting_group"
                    ].nunique(),

                "injury_related_classifications":
                    injury_count,

                "late_program_depletion_classifications":
                    late_depletion_count,

                "late_program_support_classifications":
                    late_support_count,

                "stable_top5_all_mechanistic_scenarios":
                    stable_top5_count,

                "stable_top10_all_confidence_scenarios":
                    stable_top10_count,

                "Figure47_created":
                    figure_created,

                "integrated_and_provisional_rankings_kept_separate":
                    True,

                "missing_quantitative_profiles_interpreted_as_zero":
                    False,

                "therapeutic_or_beneficial_ranking_calculated":
                    False,

                "Phase6F5_status":
                    status,
            }
        ]
    )

    completion.to_csv(
        COMPLETION_OUTPUT,
        sep="\t",
        index=False,
    )

    manifest_paths = [
        ATLAS_OUTPUT,
        ATLAS_GZ_OUTPUT,
        REPORTING_OUTPUT,
        CLASS_SUMMARY_OUTPUT,
        KEY_FINDINGS_OUTPUT,
        METHODS_OUTPUT,
        RESULTS_OUTPUT,
        FIGURE_PNG,
        FIGURE_PDF,
        FIGURE_CAPTION,
        SUPPLEMENT_LEGEND,
        COMPLETION_OUTPUT,
        Path(
            __file__
        ).resolve(),
    ]

    manifest_rows = []

    for path in manifest_paths:
        manifest_rows.append(
            {
                "relative_path":
                    str(
                        path.relative_to(
                            PROJECT
                        )
                    ),

                "size_bytes":
                    path.stat().st_size,

                "sha256":
                    sha256_file(
                        path
                    ),
            }
        )

    pd.DataFrame(
        manifest_rows
    ).to_csv(
        MANIFEST_OUTPUT,
        sep="\t",
        index=False,
    )

    log_text = "\n".join(
        [
            "===== PHASE 6F5 COMPLETION =====",
            completion.to_string(
                index=False
            ),
            "",
            "===== FINAL REPORTING GROUPS =====",
            class_summary.to_string(
                index=False
            ),
            "",
            "===== RESULTS SUMMARY =====",
            results_text,
        ]
    )

    LOG_FILE.write_text(
        log_text + "\n",
        encoding="utf-8",
    )

    print(
        log_text
    )

    if status != "completed":
        raise RuntimeError(
            "Phase 6F5 validation failed."
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            (
                "Phase 6F5 failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
