#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from enigmatoolbox.datasets import load_summary_stats


# BEGIN PHASE8D2B1 SEMICOLON TABLE FALLBACK
_PD_READ_CSV_ORIGINAL = pd.read_csv
_PD_READ_TABLE_ORIGINAL = pd.read_table


def _phase8d_retry_semicolon_table(
    reader: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Retry genuinely semicolon-delimited one-column tables safely."""

    explicit_separator = (
        len(args) >= 2
        or "sep" in kwargs
        or "delimiter" in kwargs
    )

    if explicit_separator or frame.shape[1] != 1:
        return frame

    sole_header = str(frame.columns[0])

    if ";" not in sole_header:
        return frame

    retry_kwargs = dict(kwargs)
    retry_kwargs["sep"] = ";"

    retry = reader(*args, **retry_kwargs)

    required_columns = {"Structure", "d_icv"}
    recovered_columns = {str(column) for column in retry.columns}

    if (
        retry.shape[1] > 1
        and required_columns.issubset(recovered_columns)
    ):
        return retry

    return frame


def _phase8d_read_csv(
    *args: Any,
    **kwargs: Any,
) -> pd.DataFrame:
    frame = _PD_READ_CSV_ORIGINAL(*args, **kwargs)

    return _phase8d_retry_semicolon_table(
        _PD_READ_CSV_ORIGINAL,
        args,
        kwargs,
        frame,
    )


def _phase8d_read_table(
    *args: Any,
    **kwargs: Any,
) -> pd.DataFrame:
    frame = _PD_READ_TABLE_ORIGINAL(*args, **kwargs)

    return _phase8d_retry_semicolon_table(
        _PD_READ_TABLE_ORIGINAL,
        args,
        kwargs,
        frame,
    )


pd.read_csv = _phase8d_read_csv
pd.read_table = _phase8d_read_table
# END PHASE8D2B1 SEMICOLON TABLE FALLBACK



PROJECT = Path(
    "."
)

RESOURCE_DIR = (
    PROJECT
    / "05_external_resources/phase8D"
)

ENIGMA_DIR = (
    RESOURCE_DIR
    / "ENIGMA"
)

NEUROSYNTH_DIR = (
    RESOURCE_DIR
    / "neurosynth-data"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase8"
)

PROCESSED_DIR = (
    PROJECT
    / "03_processed_data/functional_imaging/phase8D"
)

LOG_DIR = (
    PROJECT
    / "09_pipeline_logs/phase8"
)


PHASE8D2A_COMPLETION = (
    TABLE_DIR
    / "phase8D2A_completion_summary.tsv"
)

VOCABULARY_FILE = (
    NEUROSYNTH_DIR
    / "data-neurosynth_version-7_vocab-terms_vocabulary.txt"
)

FEATURE_MATRIX_FILE = (
    NEUROSYNTH_DIR
    / (
        "data-neurosynth_version-7_vocab-terms_"
        "source-abstract_type-tfidf_features.npz"
    )
)

SCHIZOPHRENIA_FALLBACK = (
    ENIGMA_DIR
    / "enigmatoolbox/datasets/summary_statistics/"
      "Schizophrenia_case-controls_CortThick.csv"
)


COGNITIVE_MANIFEST_OUTPUT = (
    PROCESSED_DIR
    / "phase8D2B1_fixed_cognitive_term_manifest.tsv"
)

CLINICAL_MANIFEST_OUTPUT = (
    PROCESSED_DIR
    / "phase8D2B1_fixed_clinical_map_manifest.tsv"
)

CLINICAL_SCHEMA_OUTPUT = (
    TABLE_DIR
    / "phase8D2B1_clinical_table_schema_audit.tsv"
)

CLINICAL_PREVIEW_OUTPUT = (
    TABLE_DIR
    / "phase8D2B1_selected_clinical_table_preview.tsv"
)

COGNITIVE_MATRIX_OUTPUT = (
    TABLE_DIR
    / "phase8D2B1_Neurosynth_feature_matrix_QC.tsv"
)

COMPLETION_OUTPUT = (
    TABLE_DIR
    / "phase8D2B1_completion_summary.tsv"
)

HASH_OUTPUT = (
    TABLE_DIR
    / "phase8D2B1_canonical_output_SHA256.tsv"
)

LOG_FILE = (
    LOG_DIR
    / "phase8D2B1_selection_and_schema_audit.log"
)


COGNITIVE_SELECTIONS = [
    {
        "cognitive_domain":
            "attention",

        "selected_term":
            "attention",

        "selection_rationale":
            "canonical_domain_term",
    },

    {
        "cognitive_domain":
            "executive_control",

        "selected_term":
            "cognitive control",

        "selection_rationale":
            "specific_executive_control_construct",
    },

    {
        "cognitive_domain":
            "memory",

        "selected_term":
            "episodic memory",

        "selection_rationale":
            "specific_declarative_memory_construct",
    },

    {
        "cognitive_domain":
            "language",

        "selected_term":
            "language",

        "selection_rationale":
            "canonical_domain_term",
    },

    {
        "cognitive_domain":
            "social_cognition",

        "selected_term":
            "mentalizing",

        "selection_rationale":
            "specific_social_inference_construct",
    },

    {
        "cognitive_domain":
            "emotion_affect",

        "selected_term":
            "emotion",

        "selection_rationale":
            "canonical_affective_processing_term",
    },

    {
        "cognitive_domain":
            "reward_motivation",

        "selected_term":
            "reward",

        "selection_rationale":
            "canonical_reward_processing_term",
    },

    {
        "cognitive_domain":
            "sensorimotor",

        "selected_term":
            "motor",

        "selection_rationale":
            "broad_primary_sensorimotor_construct",
    },
]


CLINICAL_SELECTIONS = [
    {
        "disorder":
            "adhd",

        "selected_dataset_key":
            "CortThick_case_vs_controls_allages",

        "clinical_scope":
            "all_ages",

        "selection_rationale":
            "broad_primary_case_control_contrast",

        "loading_mode":
            "ENIGMA_loader",
    },

    {
        "disorder":
            "asd",

        "selected_dataset_key":
            "CortThick_case_vs_controls_meta_analysis",

        "clinical_scope":
            "meta_analysis",

        "selection_rationale":
            "primary_published_meta_analytic_contrast",

        "loading_mode":
            "ENIGMA_loader",
    },

    {
        "disorder":
            "bipolar",

        "selected_dataset_key":
            "CortThick_case_vs_controls_adult",

        "clinical_scope":
            "adult",

        "selection_rationale":
            "adult_primary_case_control_contrast",

        "loading_mode":
            "ENIGMA_loader",
    },

    {
        "disorder":
            "depression",

        "selected_dataset_key":
            "CortThick_case_vs_controls_adult",

        "clinical_scope":
            "adult",

        "selection_rationale":
            "adult_primary_case_control_contrast",

        "loading_mode":
            "ENIGMA_loader",
    },

    {
        "disorder":
            "epilepsy",

        "selected_dataset_key":
            "CortThick_case_vs_controls_allepilepsy",

        "clinical_scope":
            "all_epilepsy",

        "selection_rationale":
            "broad_primary_case_control_contrast",

        "loading_mode":
            "ENIGMA_loader",
    },

    {
        "disorder":
            "ocd",

        "selected_dataset_key":
            "CortThick_case_vs_controls_adult",

        "clinical_scope":
            "adult",

        "selection_rationale":
            "adult_primary_case_control_contrast",

        "loading_mode":
            "ENIGMA_loader",
    },

    {
        "disorder":
            "parkinsons",

        "selected_dataset_key":
            "CortThick_PDvsCN",

        "clinical_scope":
            "PD_vs_controls",

        "selection_rationale":
            "overall_PD_case_control_contrast",

        "loading_mode":
            "ENIGMA_loader",
    },

    {
        "disorder":
            "schizophrenia",

        "selected_dataset_key":
            "CortThick_case_vs_controls_direct_file_fallback",

        "clinical_scope":
            "case_vs_controls",

        "selection_rationale":
            (
                "primary_case_control_cortical_thickness_"
                "file_with_documented_loader_fallback"
            ),

        "loading_mode":
            "direct_repository_file",
    },
]


for directory in [
    TABLE_DIR,
    PROCESSED_DIR,
    LOG_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def normalize_term(
    value: str,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(
            value
        )
        .strip()
        .lower()
        .replace(
            "_",
            " ",
        )
        .replace(
            "-",
            " ",
        ),
    )


def load_selected_clinical_table(
    disorder: str,
    dataset_key: str,
    loading_mode: str,
) -> pd.DataFrame:
    if loading_mode == "direct_repository_file":
        if disorder != "schizophrenia":
            raise RuntimeError(
                "Direct loading was requested for an unexpected disorder."
            )

        if not SCHIZOPHRENIA_FALLBACK.exists():
            raise FileNotFoundError(
                f"Missing schizophrenia fallback: "
                f"{SCHIZOPHRENIA_FALLBACK}"
            )

        return pd.read_csv(
            SCHIZOPHRENIA_FALLBACK,
            low_memory=False,
        )

    datasets = load_summary_stats(
        disorder
    )

    if not isinstance(
        datasets,
        dict,
    ):
        raise RuntimeError(
            f"{disorder}: ENIGMA loader did not return a dictionary."
        )

    if dataset_key not in datasets:
        available = "|".join(
            sorted(
                map(
                    str,
                    datasets.keys(),
                )
            )
        )

        raise KeyError(
            (
                f"{disorder}: selected key '{dataset_key}' "
                f"was not found. Available: {available}"
            )
        )

    table = datasets[
        dataset_key
    ]

    if not isinstance(
        table,
        pd.DataFrame,
    ):
        raise TypeError(
            f"{disorder}: selected ENIGMA object is not a DataFrame."
        )

    return table.copy()


def likely_region_columns(
    columns: list[str],
) -> list[str]:
    patterns = [
        r"region",
        r"structure",
        r"parcel",
        r"label",
        r"roi",
        r"area",
        r"name",
    ]

    matches = []

    for column in columns:
        normalized = str(
            column
        ).lower()

        if any(
            re.search(
                pattern,
                normalized,
            )
            for pattern in patterns
        ):
            matches.append(
                str(
                    column
                )
            )

    return matches


def likely_effect_columns(
    table: pd.DataFrame,
) -> list[str]:
    numeric_columns = [
        str(
            column
        )
        for column in table.select_dtypes(
            include=[
                np.number,
            ]
        ).columns
    ]

    exact_names = {
        "d",
        "cohen_d",
        "cohens_d",
        "effect_size",
        "effectsize",
        "beta",
        "estimate",
        "z",
        "zscore",
    }

    patterns = [
        r"cohen",
        r"effect",
        r"hedges",
        r"beta",
        r"estimate",
        r"zscore",
        r"(^|_)d($|_)",
    ]

    exact_matches = [
        column
        for column in numeric_columns
        if (
            str(
                column
            )
            .strip()
            .lower()
            in exact_names
        )
    ]

    pattern_matches = [
        column
        for column in numeric_columns
        if any(
            re.search(
                pattern,
                str(
                    column
                ).lower(),
            )
            for pattern in patterns
        )
    ]

    ordered = []

    for column in (
        exact_matches
        + pattern_matches
    ):
        if column not in ordered:
            ordered.append(
                column
            )

    return ordered


def main() -> None:
    required = [
        PHASE8D2A_COMPLETION,
        VOCABULARY_FILE,
        FEATURE_MATRIX_FILE,
        SCHIZOPHRENIA_FALLBACK,
    ]

    missing = [
        str(
            path
        )
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing Phase 8D2B1 prerequisites:\n"
            + "\n".join(
                missing
            )
        )

    phase8d2a = pd.read_csv(
        PHASE8D2A_COMPLETION,
        sep="\t",
        low_memory=False,
    )

    phase8d2a_completed = bool(
        "Phase8D2A_status"
        in phase8d2a.columns
        and str(
            phase8d2a[
                "Phase8D2A_status"
            ].iloc[
                0
            ]
        ).strip().lower()
        == "completed"
    )

    ready_from_previous_phase = bool(
        "ready_for_phase8D2B"
        in phase8d2a.columns
        and str(
            phase8d2a[
                "ready_for_phase8D2B"
            ].iloc[
                0
            ]
        ).strip().lower()
        == "true"
    )

    # ============================================================
    # Neurosynth cognitive-term audit
    # ============================================================

    vocabulary = [
        line.strip()
        for line in VOCABULARY_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    normalized_to_indices: dict[
        str,
        list[int],
    ] = {}

    for index, term in enumerate(
        vocabulary
    ):
        normalized_to_indices.setdefault(
            normalize_term(
                term
            ),
            [],
        ).append(
            index
        )

    feature_matrix = sparse.load_npz(
        FEATURE_MATRIX_FILE
    ).tocsr()

    if feature_matrix.shape[
        1
    ] == len(
        vocabulary
    ):
        term_axis = 1
        study_axis = 0

    elif feature_matrix.shape[
        0
    ] == len(
        vocabulary
    ):
        term_axis = 0
        study_axis = 1

    else:
        raise RuntimeError(
            (
                "Neurosynth feature-matrix dimensions do not align "
                f"with the {len(vocabulary)}-term vocabulary: "
                f"{feature_matrix.shape}"
            )
        )

    cognitive_rows = []

    for rank, selection in enumerate(
        COGNITIVE_SELECTIONS,
        start=1,
    ):
        normalized = normalize_term(
            selection[
                "selected_term"
            ]
        )

        indices = normalized_to_indices.get(
            normalized,
            [],
        )

        exact_match_unique = (
            len(
                indices
            )
            == 1
        )

        vocabulary_index = (
            indices[
                0
            ]
            if exact_match_unique
            else -1
        )

        if exact_match_unique:
            if term_axis == 1:
                nonzero_studies = int(
                    feature_matrix[
                        :,
                        vocabulary_index,
                    ].count_nonzero()
                )

            else:
                nonzero_studies = int(
                    feature_matrix[
                        vocabulary_index,
                        :,
                    ].count_nonzero()
                )

        else:
            nonzero_studies = 0

        cognitive_rows.append(
            {
                **selection,

                "fixed_rank":
                    rank,

                "Neurosynth_version":
                    "7",

                "exact_vocabulary_match":
                    exact_match_unique,

                "vocabulary_index_zero_based":
                    vocabulary_index,

                "nonzero_studies":
                    nonzero_studies,

                "minimum_study_support_met":
                    nonzero_studies
                    >= 20,

                "planned_map_type":
                    "NiMARE_term_meta_analytic_map",

                "harmonization_target":
                    "Schaefer100_LH50",

                "selection_frozen":
                    True,
            }
        )

    cognitive_manifest = pd.DataFrame(
        cognitive_rows
    )

    cognitive_manifest.to_csv(
        COGNITIVE_MANIFEST_OUTPUT,
        sep="\t",
        index=False,
    )

    cognitive_matrix_QC = pd.DataFrame(
        [
            {
                "feature_matrix_rows":
                    feature_matrix.shape[
                        0
                    ],

                "feature_matrix_columns":
                    feature_matrix.shape[
                        1
                    ],

                "feature_matrix_nonzero":
                    feature_matrix.nnz,

                "vocabulary_terms":
                    len(
                        vocabulary
                    ),

                "term_axis":
                    term_axis,

                "study_axis":
                    study_axis,

                "selected_terms":
                    len(
                        cognitive_manifest
                    ),

                "selected_terms_exactly_matched":
                    int(
                        cognitive_manifest[
                            "exact_vocabulary_match"
                        ].sum()
                    ),

                "selected_terms_with_at_least_20_studies":
                    int(
                        cognitive_manifest[
                            "minimum_study_support_met"
                        ].sum()
                    ),

                "matrix_vocabulary_alignment_valid":
                    True,
            }
        ]
    )

    cognitive_matrix_QC.to_csv(
        COGNITIVE_MATRIX_OUTPUT,
        sep="\t",
        index=False,
    )

    # ============================================================
    # ENIGMA clinical schema audit
    # ============================================================

    clinical_manifest_rows = []
    schema_rows = []
    preview_rows = []

    for rank, selection in enumerate(
        CLINICAL_SELECTIONS,
        start=1,
    ):
        table = load_selected_clinical_table(
            disorder=selection[
                "disorder"
            ],

            dataset_key=selection[
                "selected_dataset_key"
            ],

            loading_mode=selection[
                "loading_mode"
            ],
        )

        columns = [
            str(
                column
            )
            for column in table.columns
        ]

        numeric_columns = [
            str(
                column
            )
            for column in table.select_dtypes(
                include=[
                    np.number,
                ]
            ).columns
        ]

        region_candidates = (
            likely_region_columns(
                columns
            )
        )

        effect_candidates = (
            likely_effect_columns(
                table
            )
        )

        clinical_manifest_rows.append(
            {
                **selection,

                "fixed_rank":
                    rank,

                "source":
                    "ENIGMA_Toolbox",

                "table_rows":
                    len(
                        table
                    ),

                "table_columns":
                    len(
                        columns
                    ),

                "candidate_effect_columns":
                    "|".join(
                        effect_candidates
                    ),

                "candidate_effect_column_count":
                    len(
                        effect_candidates
                    ),

                "candidate_region_columns":
                    "|".join(
                        region_candidates
                    ),

                "candidate_region_column_count":
                    len(
                        region_candidates
                    ),

                "harmonization_target":
                    (
                        "DK68_to_fsaverage5_to_"
                        "Schaefer100_LH50"
                    ),

                "effect_direction":
                    "verify_from_selected_table_schema",

                "selection_frozen":
                    True,
            }
        )

        schema_rows.append(
            {
                "disorder":
                    selection[
                        "disorder"
                    ],

                "selected_dataset_key":
                    selection[
                        "selected_dataset_key"
                    ],

                "loading_mode":
                    selection[
                        "loading_mode"
                    ],

                "rows":
                    len(
                        table
                    ),

                "columns":
                    len(
                        columns
                    ),

                "all_column_names":
                    "|".join(
                        columns
                    ),

                "numeric_column_names":
                    "|".join(
                        numeric_columns
                    ),

                "region_column_candidates":
                    "|".join(
                        region_candidates
                    ),

                "effect_column_candidates":
                    "|".join(
                        effect_candidates
                    ),

                "all_values_missing_columns":
                    "|".join(
                        [
                            str(
                                column
                            )
                            for column in table.columns
                            if table[
                                column
                            ].isna().all()
                        ]
                    ),

                "duplicate_rows":
                    int(
                        table.duplicated().sum()
                    ),

                "schema_audit_passed":
                    bool(
                        len(
                            table
                        )
                        >= 68
                        and len(
                            numeric_columns
                        )
                        >= 1
                    ),
            }
        )

        preview = table.head(
            3
        ).copy()

        preview.insert(
            0,
            "selected_dataset_key",
            selection[
                "selected_dataset_key"
            ],
        )

        preview.insert(
            0,
            "disorder",
            selection[
                "disorder"
            ],
        )

        preview_rows.append(
            preview
        )

    clinical_manifest = pd.DataFrame(
        clinical_manifest_rows
    )

    clinical_schema = pd.DataFrame(
        schema_rows
    )

    clinical_preview = pd.concat(
        preview_rows,
        ignore_index=True,
        sort=False,
    )

    clinical_manifest.to_csv(
        CLINICAL_MANIFEST_OUTPUT,
        sep="\t",
        index=False,
    )

    clinical_schema.to_csv(
        CLINICAL_SCHEMA_OUTPUT,
        sep="\t",
        index=False,
    )

    clinical_preview.to_csv(
        CLINICAL_PREVIEW_OUTPUT,
        sep="\t",
        index=False,
    )

    cognitive_terms_ready = bool(
        len(
            cognitive_manifest
        )
        == 8
        and cognitive_manifest[
            "exact_vocabulary_match"
        ].all()
        and cognitive_manifest[
            "minimum_study_support_met"
        ].all()
    )

    clinical_tables_ready = bool(
        len(
            clinical_manifest
        )
        == 8
        and clinical_schema[
            "schema_audit_passed"
        ].all()
    )

    all_effect_columns_detected = bool(
        clinical_manifest[
            "candidate_effect_column_count"
        ].gt(
            0
        ).all()
    )

    status = (
        "completed"
        if (
            phase8d2a_completed
            and ready_from_previous_phase
            and cognitive_terms_ready
            and clinical_tables_ready
        )
        else "failed_validation"
    )

    completion = pd.DataFrame(
        [
            {
                "Phase8D2A_status_confirmed":
                    phase8d2a_completed,

                "Phase8D2A_ready_flag_confirmed":
                    ready_from_previous_phase,

                "cognitive_domains_selected":
                    len(
                        cognitive_manifest
                    ),

                "cognitive_terms_exactly_matched":
                    int(
                        cognitive_manifest[
                            "exact_vocabulary_match"
                        ].sum()
                    ),

                "cognitive_terms_with_at_least_20_studies":
                    int(
                        cognitive_manifest[
                            "minimum_study_support_met"
                        ].sum()
                    ),

                "clinical_disorders_selected":
                    len(
                        clinical_manifest
                    ),

                "clinical_tables_loaded":
                    int(
                        clinical_schema[
                            "schema_audit_passed"
                        ].sum()
                    ),

                "clinical_tables_with_effect_column_candidates":
                    int(
                        clinical_manifest[
                            "candidate_effect_column_count"
                        ].gt(
                            0
                        ).sum()
                    ),

                "all_clinical_effect_columns_automatically_resolved":
                    all_effect_columns_detected,

                "selections_frozen_before_map_generation":
                    True,

                "ready_for_phase8D2B2":
                    bool(
                        status
                        == "completed"
                    ),

                "Phase8D2B1_status":
                    status,
            }
        ]
    )

    completion.to_csv(
        COMPLETION_OUTPUT,
        sep="\t",
        index=False,
    )

    canonical_outputs = [
        COGNITIVE_MANIFEST_OUTPUT,
        CLINICAL_MANIFEST_OUTPUT,
        CLINICAL_SCHEMA_OUTPUT,
        CLINICAL_PREVIEW_OUTPUT,
        COGNITIVE_MATRIX_OUTPUT,
        COMPLETION_OUTPUT,
    ]

    hashes = pd.DataFrame(
        [
            {
                "relative_path":
                    str(
                        path.relative_to(
                            PROJECT
                        )
                    ),

                "size_bytes":
                    path.stat().st_size,

                "SHA256":
                    sha256_file(
                        path
                    ),
            }
            for path in canonical_outputs
        ]
    )

    hashes.to_csv(
        HASH_OUTPUT,
        sep="\t",
        index=False,
    )

    log_text = "\n".join(
        [
            "===== PHASE 8D2B1 COMPLETION =====",
            completion.to_string(
                index=False
            ),
            "",
            "===== FIXED COGNITIVE TERMS =====",
            cognitive_manifest.to_string(
                index=False
            ),
            "",
            "===== FIXED CLINICAL MAPS =====",
            clinical_manifest.to_string(
                index=False
            ),
            "",
            "===== CLINICAL SCHEMA AUDIT =====",
            clinical_schema.to_string(
                index=False
            ),
            "",
            "===== NEUROSYNTH FEATURE MATRIX QC =====",
            cognitive_matrix_QC.to_string(
                index=False
            ),
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
            "Phase 8D2B1 selection/schema validation failed."
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            (
                "Phase 8D2B1 failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
