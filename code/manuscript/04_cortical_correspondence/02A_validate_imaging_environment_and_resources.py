#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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


PHASE8D1_COMPLETION = (
    TABLE_DIR
    / "phase8D1_completion_summary.tsv"
)

PACKAGE_OUTPUT = (
    TABLE_DIR
    / "phase8D2A_package_version_manifest.tsv"
)

REPOSITORY_OUTPUT = (
    TABLE_DIR
    / "phase8D2A_repository_commit_manifest.tsv"
)

ENIGMA_INVENTORY_OUTPUT = (
    TABLE_DIR
    / "phase8D2A_ENIGMA_summary_statistics_inventory.tsv"
)

ENIGMA_DISORDER_OUTPUT = (
    TABLE_DIR
    / "phase8D2A_ENIGMA_disorder_availability_summary.tsv"
)

CROSSWALK_OUTPUT = (
    TABLE_DIR
    / "phase8D2A_ENIGMA_DK68_to_Schaefer100_crosswalk_QC.tsv"
)

NEUROSYNTH_RESOURCE_OUTPUT = (
    TABLE_DIR
    / "phase8D2A_Neurosynth_v7_resource_inventory.tsv"
)

COGNITIVE_TERM_OUTPUT = (
    TABLE_DIR
    / "phase8D2A_Neurosynth_cognitive_term_vocabulary_audit.tsv"
)

SELECTED_SOURCE_OUTPUT = (
    PROCESSED_DIR
    / "phase8D2A_selected_resource_sources.tsv"
)

HASH_OUTPUT = (
    TABLE_DIR
    / "phase8D2A_canonical_output_SHA256.tsv"
)

COMPLETION_OUTPUT = (
    TABLE_DIR
    / "phase8D2A_completion_summary.tsv"
)

LOG_FILE = (
    LOG_DIR
    / "phase8D2A_environment_and_resource_validation.log"
)


PACKAGE_IMPORTS = {
    "numpy":
        "numpy",

    "pandas":
        "pandas",

    "scipy":
        "scipy",

    "matplotlib":
        "matplotlib",

    "scikit-learn":
        "sklearn",

    "nibabel":
        "nibabel",

    "nilearn":
        "nilearn",

    "neuromaps":
        "neuromaps",

    "nimare":
        "nimare",

    "enigmatoolbox":
        "enigmatoolbox",
}


TARGET_DISORDERS = [
    "adhd",
    "asd",
    "bipolar",
    "depression",
    "epilepsy",
    "ocd",
    "parkinsons",
    "schizophrenia",
]


COGNITIVE_DOMAINS = {
    "attention": [
        "attention",
        "attentional",
    ],

    "executive_control": [
        "executive",
        "cognitive control",
        "working memory",
    ],

    "memory": [
        "memory",
        "episodic memory",
        "recognition memory",
    ],

    "language": [
        "language",
        "semantic",
        "speech",
    ],

    "social_cognition": [
        "social",
        "theory of mind",
        "mentalizing",
    ],

    "emotion_affect": [
        "emotion",
        "emotional",
        "fear",
        "affective",
    ],

    "reward_motivation": [
        "reward",
        "motivation",
        "reinforcement",
    ],

    "sensorimotor": [
        "motor",
        "visual",
        "auditory",
        "somatosensory",
    ],
}


EXPECTED_SCHAEFER_PARCELS = 100
EXPECTED_DK_PARCELS = 68


for directory in [
    TABLE_DIR,
    PROCESSED_DIR,
    LOG_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def read_table(
    path: Path,
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        compression="infer",
        low_memory=False,
    )


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def git_value(
    repository: Path,
    arguments: list[str],
) -> str:
    return (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(repository),
                *arguments,
            ],
            text=True,
        )
        .strip()
    )


def package_version(
    distribution_name: str,
) -> str:
    try:
        return importlib.metadata.version(
            distribution_name
        )

    except importlib.metadata.PackageNotFoundError:
        return ""


def normalize_term(
    value: str,
) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def main() -> None:
    required_files = [
        PHASE8D1_COMPLETION,
        ENIGMA_DIR / ".git",
        NEUROSYNTH_DIR / ".git",
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing Phase 8D2A prerequisites:\n"
            + "\n".join(missing)
        )

    phase8d1 = read_table(
        PHASE8D1_COMPLETION
    )

    phase8d1_completed = bool(
        "Phase8D1_status"
        in phase8d1.columns
        and str(
            phase8d1[
                "Phase8D1_status"
            ].iloc[0]
        ).strip().lower()
        == "completed"
    )

    # ============================================================
    # Package audit
    # ============================================================

    package_rows = []

    for distribution, module_name in (
        PACKAGE_IMPORTS.items()
    ):
        try:
            module = importlib.import_module(
                module_name
            )

            import_successful = True

            module_file = str(
                getattr(
                    module,
                    "__file__",
                    "",
                )
            )

            import_error = ""

        except Exception as error:
            import_successful = False
            module_file = ""
            import_error = (
                f"{type(error).__name__}: "
                f"{error}"
            )

        package_rows.append(
            {
                "distribution":
                    distribution,

                "module":
                    module_name,

                "version":
                    package_version(
                        distribution
                    ),

                "import_successful":
                    import_successful,

                "module_file":
                    module_file,

                "import_error":
                    import_error,
            }
        )

    packages = pd.DataFrame(
        package_rows
    )

    packages.to_csv(
        PACKAGE_OUTPUT,
        sep="\t",
        index=False,
    )

    # ============================================================
    # Repository audit
    # ============================================================

    repository_rows = []

    for resource_name, repository in [
        (
            "ENIGMA_Toolbox",
            ENIGMA_DIR,
        ),
        (
            "Neurosynth_data",
            NEUROSYNTH_DIR,
        ),
    ]:
        repository_rows.append(
            {
                "resource":
                    resource_name,

                "relative_path":
                    str(
                        repository.relative_to(
                            PROJECT
                        )
                    ),

                "commit_SHA":
                    git_value(
                        repository,
                        [
                            "rev-parse",
                            "HEAD",
                        ],
                    ),

                "commit_date":
                    git_value(
                        repository,
                        [
                            "show",
                            "-s",
                            "--format=%cI",
                            "HEAD",
                        ],
                    ),

                "branch":
                    git_value(
                        repository,
                        [
                            "rev-parse",
                            "--abbrev-ref",
                            "HEAD",
                        ],
                    ),

                "working_tree_clean":
                    git_value(
                        repository,
                        [
                            "status",
                            "--porcelain",
                        ],
                    )
                    == "",

                "remote_origin":
                    git_value(
                        repository,
                        [
                            "remote",
                            "get-url",
                            "origin",
                        ],
                    ),
            }
        )

    repositories = pd.DataFrame(
        repository_rows
    )

    repositories.to_csv(
        REPOSITORY_OUTPUT,
        sep="\t",
        index=False,
    )

    # ============================================================
    # ENIGMA summary-statistics inventory
    # ============================================================

    from enigmatoolbox.datasets import (
        load_summary_stats,
    )

    from enigmatoolbox.utils.parcellation import (
        parcel_to_surface,
        surface_to_parcel,
    )

    enigma_rows = []
    disorder_rows = []

    for disorder in TARGET_DISORDERS:
        try:
            datasets = load_summary_stats(
                disorder
            )

            if not isinstance(
                datasets,
                dict,
            ):
                raise RuntimeError(
                    "load_summary_stats did not return a dictionary."
                )

            cortical_thickness_keys = []

            for dataset_key, dataset in (
                datasets.items()
            ):
                if not isinstance(
                    dataset,
                    pd.DataFrame,
                ):
                    continue

                lower_key = str(
                    dataset_key
                ).lower()

                is_cortical_thickness = (
                    "cortthick"
                    in lower_key
                    or (
                        "cortical"
                        in lower_key
                        and "thick"
                        in lower_key
                    )
                )

                is_case_control = (
                    "case"
                    in lower_key
                    and (
                        "control"
                        in lower_key
                        or "controls"
                        in lower_key
                    )
                )

                if (
                    is_cortical_thickness
                    and is_case_control
                ):
                    cortical_thickness_keys.append(
                        str(dataset_key)
                    )

                enigma_rows.append(
                    {
                        "disorder":
                            disorder,

                        "dataset_key":
                            str(dataset_key),

                        "rows":
                            len(dataset),

                        "columns":
                            len(dataset.columns),

                        "column_names":
                            "|".join(
                                map(
                                    str,
                                    dataset.columns,
                                )
                            ),

                        "is_cortical_thickness":
                            is_cortical_thickness,

                        "is_case_control":
                            is_case_control,

                        "candidate_primary_clinical_map":
                            bool(
                                is_cortical_thickness
                                and is_case_control
                            ),
                    }
                )

            disorder_rows.append(
                {
                    "disorder":
                        disorder,

                    "load_successful":
                        True,

                    "available_tables":
                        len(datasets),

                    "cortical_thickness_case_control_tables":
                        len(
                            cortical_thickness_keys
                        ),

                    "candidate_table_keys":
                        "|".join(
                            cortical_thickness_keys
                        ),

                    "load_error":
                        "",
                }
            )

        except Exception as error:
            disorder_rows.append(
                {
                    "disorder":
                        disorder,

                    "load_successful":
                        False,

                    "available_tables":
                        0,

                    "cortical_thickness_case_control_tables":
                        0,

                    "candidate_table_keys":
                        "",

                    "load_error":
                        (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                }
            )

    enigma_inventory = pd.DataFrame(
        enigma_rows
    )

    disorder_summary = pd.DataFrame(
        disorder_rows
    )

    enigma_inventory.to_csv(
        ENIGMA_INVENTORY_OUTPUT,
        sep="\t",
        index=False,
    )

    disorder_summary.to_csv(
        ENIGMA_DISORDER_OUTPUT,
        sep="\t",
        index=False,
    )

    # ============================================================
    # ENIGMA DK68 -> surface -> Schaefer100 compatibility test
    # ============================================================

    DK_test_vector = np.arange(
        EXPECTED_DK_PARCELS,
        dtype=float,
    )

    DK_surface = parcel_to_surface(
        DK_test_vector,
        "aparc_fsa5",
    )

    Schaefer100_test = surface_to_parcel(
        DK_surface,
        "schaefer_100_fsa5",
    )

    DK_surface = np.asarray(
        DK_surface,
        dtype=float,
    )

    Schaefer100_test = np.asarray(
        Schaefer100_test,
        dtype=float,
    )

    crosswalk_QC = pd.DataFrame(
        [
            {
                "source_parcellation":
                    "Desikan_Killiany_68",

                "intermediate_surface":
                    "fsaverage5",

                "target_parcellation":
                    "Schaefer_100",

                "source_values":
                    len(
                        DK_test_vector
                    ),

                "surface_vertices":
                    DK_surface.size,

                "target_values":
                    Schaefer100_test.size,

                "finite_surface_fraction":
                    float(
                        np.isfinite(
                            DK_surface
                        ).mean()
                    ),

                "finite_target_fraction":
                    float(
                        np.isfinite(
                            Schaefer100_test
                        ).mean()
                    ),

                "target_constant":
                    bool(
                        np.unique(
                            Schaefer100_test[
                                np.isfinite(
                                    Schaefer100_test
                                )
                            ]
                        ).size
                        <= 1
                    ),

                "crosswalk_ready":
                    bool(
                        len(
                            DK_test_vector
                        )
                        == EXPECTED_DK_PARCELS
                        and Schaefer100_test.size
                        == EXPECTED_SCHAEFER_PARCELS
                        and np.isfinite(
                            DK_surface
                        ).all()
                        and np.isfinite(
                            Schaefer100_test
                        ).all()
                        and np.unique(
                            Schaefer100_test
                        ).size
                        > 1
                    ),
            }
        ]
    )

    crosswalk_QC.to_csv(
        CROSSWALK_OUTPUT,
        sep="\t",
        index=False,
    )

    # ============================================================
    # Neurosynth v7 inventory
    # ============================================================

    expected_patterns = {
        "coordinates":
            "*version-7*coordinates.tsv.gz",

        "metadata":
            "*version-7*metadata.tsv.gz",

        "terms_features":
            (
                "*version-7_vocab-terms_"
                "source-abstract_type-tfidf_features.npz"
            ),

        "terms_vocabulary":
            "*version-7_vocab-terms_vocabulary.txt",
    }

    resource_rows = []
    selected_files = {}

    for resource_type, pattern in (
        expected_patterns.items()
    ):
        matches = sorted(
            NEUROSYNTH_DIR.glob(
                pattern
            )
        )

        selected_path = (
            matches[0]
            if len(matches) == 1
            else None
        )

        selected_files[
            resource_type
        ] = selected_path

        resource_rows.append(
            {
                "resource_type":
                    resource_type,

                "glob_pattern":
                    pattern,

                "matching_files":
                    len(matches),

                "selected_relative_path":
                    (
                        str(
                            selected_path.relative_to(
                                PROJECT
                            )
                        )
                        if selected_path is not None
                        else ""
                    ),

                "size_bytes":
                    (
                        selected_path.stat().st_size
                        if selected_path is not None
                        else 0
                    ),

                "SHA256":
                    (
                        sha256_file(
                            selected_path
                        )
                        if selected_path is not None
                        else ""
                    ),

                "resource_ready":
                    selected_path
                    is not None,
            }
        )

    neurosynth_resources = pd.DataFrame(
        resource_rows
    )

    neurosynth_resources.to_csv(
        NEUROSYNTH_RESOURCE_OUTPUT,
        sep="\t",
        index=False,
    )

    vocabulary_path = selected_files[
        "terms_vocabulary"
    ]

    if vocabulary_path is None:
        vocabulary = []

    else:
        vocabulary = [
            line.strip()
            for line in vocabulary_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

    normalized_vocabulary = {
        normalize_term(term):
            term
        for term in vocabulary
    }

    cognitive_rows = []

    for domain, requested_terms in (
        COGNITIVE_DOMAINS.items()
    ):
        exact_matches = []

        partial_matches = []

        for requested_term in requested_terms:
            normalized_requested = normalize_term(
                requested_term
            )

            if normalized_requested in (
                normalized_vocabulary
            ):
                exact_matches.append(
                    normalized_vocabulary[
                        normalized_requested
                    ]
                )

            for normalized_term, original_term in (
                normalized_vocabulary.items()
            ):
                if (
                    normalized_requested
                    in normalized_term
                    or normalized_term
                    in normalized_requested
                ):
                    partial_matches.append(
                        original_term
                    )

        cognitive_rows.append(
            {
                "cognitive_domain":
                    domain,

                "requested_terms":
                    "|".join(
                        requested_terms
                    ),

                "exact_vocabulary_matches":
                    "|".join(
                        sorted(
                            set(
                                exact_matches
                            )
                        )
                    ),

                "partial_vocabulary_matches":
                    "|".join(
                        sorted(
                            set(
                                partial_matches
                            )
                        )[
                            :30
                        ]
                    ),

                "exact_matches":
                    len(
                        set(
                            exact_matches
                        )
                    ),

                "partial_matches":
                    len(
                        set(
                            partial_matches
                        )
                    ),

                "domain_has_candidate_term":
                    bool(
                        exact_matches
                        or partial_matches
                    ),
            }
        )

    cognitive_audit = pd.DataFrame(
        cognitive_rows
    )

    cognitive_audit.to_csv(
        COGNITIVE_TERM_OUTPUT,
        sep="\t",
        index=False,
    )

    selected_sources = pd.DataFrame(
        [
            {
                "resource_family":
                    "cognitive_activation_maps",

                "primary_source":
                    "Neurosynth_version_7",

                "planned_map_type":
                    (
                        "term_based_meta_analytic_"
                        "association_or_uniformity_map"
                    ),

                "harmonization_target":
                    "Schaefer100_LH50",

                "analysis_role":
                    "primary_cognitive_spatial_validation",
            },

            {
                "resource_family":
                    "clinical_cortical_maps",

                "primary_source":
                    "ENIGMA_Toolbox",

                "planned_map_type":
                    (
                        "case_control_cortical_thickness_"
                        "effect_size"
                    ),

                "harmonization_target":
                    (
                        "DK68_to_fsaverage5_to_"
                        "Schaefer100_LH50"
                    ),

                "analysis_role":
                    "primary_clinical_spatial_validation",
            },

            {
                "resource_family":
                    "predictive_text_to_brain_maps",

                "primary_source":
                    "NeuroQuery",

                "planned_map_type":
                    "predictive_encoding_map",

                "harmonization_target":
                    "Schaefer100_LH50",

                "analysis_role":
                    "optional_sensitivity_only",
            },
        ]
    )

    selected_sources.to_csv(
        SELECTED_SOURCE_OUTPUT,
        sep="\t",
        index=False,
    )

    packages_ready = bool(
        packages[
            "import_successful"
        ].all()
    )

    repositories_ready = bool(
        len(repositories) == 2
        and repositories[
            "working_tree_clean"
        ].all()
    )

    disorders_available = int(
        disorder_summary[
            "load_successful"
        ].sum()
    )

    disorders_with_CT_maps = int(
        disorder_summary[
            "cortical_thickness_case_control_tables"
        ].gt(0).sum()
    )

    neurosynth_ready = bool(
        neurosynth_resources[
            "resource_ready"
        ].all()
    )

    cognitive_domains_with_terms = int(
        cognitive_audit[
            "domain_has_candidate_term"
        ].sum()
    )

    crosswalk_ready = bool(
        crosswalk_QC[
            "crosswalk_ready"
        ].iloc[0]
    )

    status = (
        "completed"
        if (
            phase8d1_completed
            and packages_ready
            and repositories_ready
            and disorders_available
            == len(
                TARGET_DISORDERS
            )
            and disorders_with_CT_maps
            >= 6
            and neurosynth_ready
            and cognitive_domains_with_terms
            == len(
                COGNITIVE_DOMAINS
            )
            and crosswalk_ready
        )
        else "failed_validation"
    )

    completion = pd.DataFrame(
        [
            {
                "Phase8D1_status_confirmed":
                    phase8d1_completed,

                "packages_audited":
                    len(packages),

                "packages_import_successful":
                    int(
                        packages[
                            "import_successful"
                        ].sum()
                    ),

                "repositories_audited":
                    len(repositories),

                "repositories_clean":
                    int(
                        repositories[
                            "working_tree_clean"
                        ].sum()
                    ),

                "ENIGMA_disorders_requested":
                    len(
                        TARGET_DISORDERS
                    ),

                "ENIGMA_disorders_loaded":
                    disorders_available,

                "ENIGMA_disorders_with_case_control_cortical_thickness":
                    disorders_with_CT_maps,

                "DK68_to_Schaefer100_crosswalk_ready":
                    crosswalk_ready,

                "Neurosynth_core_resources_ready":
                    neurosynth_ready,

                "Neurosynth_vocabulary_terms":
                    len(vocabulary),

                "cognitive_domains_requested":
                    len(
                        COGNITIVE_DOMAINS
                    ),

                "cognitive_domains_with_candidate_terms":
                    cognitive_domains_with_terms,

                "primary_cognitive_source":
                    "Neurosynth_version_7",

                "primary_clinical_source":
                    "ENIGMA_Toolbox",

                "NeuroQuery_analysis_role":
                    "optional_sensitivity_only",

                "ready_for_phase8D2B":
                    status
                    == "completed",

                "Phase8D2A_status":
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
        PACKAGE_OUTPUT,
        REPOSITORY_OUTPUT,
        ENIGMA_INVENTORY_OUTPUT,
        ENIGMA_DISORDER_OUTPUT,
        CROSSWALK_OUTPUT,
        NEUROSYNTH_RESOURCE_OUTPUT,
        COGNITIVE_TERM_OUTPUT,
        SELECTED_SOURCE_OUTPUT,
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
            "===== PHASE 8D2A COMPLETION =====",
            completion.to_string(
                index=False
            ),
            "",
            "===== PACKAGE MANIFEST =====",
            packages.to_string(
                index=False
            ),
            "",
            "===== REPOSITORY MANIFEST =====",
            repositories.to_string(
                index=False
            ),
            "",
            "===== ENIGMA DISORDER AVAILABILITY =====",
            disorder_summary.to_string(
                index=False
            ),
            "",
            "===== ENIGMA CROSSWALK QC =====",
            crosswalk_QC.to_string(
                index=False
            ),
            "",
            "===== NEUROSYNTH RESOURCE INVENTORY =====",
            neurosynth_resources.to_string(
                index=False
            ),
            "",
            "===== COGNITIVE TERM AUDIT =====",
            cognitive_audit.to_string(
                index=False
            ),
            "",
            "===== SELECTED SOURCES =====",
            selected_sources.to_string(
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
            "Phase 8D2A validation failed."
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            (
                "Phase 8D2A failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
