#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT = Path(
    "."
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


PHASE8C_COMPLETION = (
    TABLE_DIR
    / "phase8C_final_completion_summary.tsv"
)

PHASE8C_HASH_MANIFEST = (
    TABLE_DIR
    / "phase8C4_canonical_output_SHA256.tsv"
)

FIGURE56_PNG = (
    PROJECT
    / "06_figures/main_figures/phase8/"
      "Figure56_developmental_program_and_network_localization_atlas.png"
)


CANDIDATE_OUTPUT = (
    TABLE_DIR
    / "phase8D1_cognitive_clinical_resource_candidate_inventory.tsv"
)

PACKAGE_OUTPUT = (
    TABLE_DIR
    / "phase8D1_neuroimaging_package_audit.tsv"
)

CATEGORY_OUTPUT = (
    TABLE_DIR
    / "phase8D1_candidate_category_summary.tsv"
)

ACQUISITION_PLAN_OUTPUT = (
    TABLE_DIR
    / "phase8D1_resource_acquisition_plan.tsv"
)

COMPLETION_OUTPUT = (
    TABLE_DIR
    / "phase8D1_completion_summary.tsv"
)

LOG_FILE = (
    LOG_DIR
    / "phase8D1_cognitive_clinical_resource_audit.log"
)


SEARCH_ROOTS = [
    PROJECT / "02_raw_data",
    PROJECT / "03_processed_data",
    PROJECT / "05_external_resources",
    PROJECT / "08_supplementary",
    PROJECT / "10_resources",
]

EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "14_compressed_archives",
    "__pycache__",
    "node_modules",
}


COGNITIVE_KEYWORDS = {
    "attention": [
        "attention",
        "attentional",
    ],

    "executive_control": [
        "executive",
        "cognitive_control",
        "working_memory",
        "working-memory",
    ],

    "memory": [
        "memory",
        "episodic",
        "recognition",
    ],

    "language": [
        "language",
        "semantic",
        "speech",
        "verbal",
    ],

    "social_cognition": [
        "social",
        "theory_of_mind",
        "mentalizing",
    ],

    "emotion_affect": [
        "emotion",
        "affect",
        "fear",
        "valence",
    ],

    "reward_motivation": [
        "reward",
        "motivation",
        "reinforcement",
    ],

    "sensorimotor": [
        "motor",
        "somatosensory",
        "visual",
        "auditory",
    ],

    "general_cognition": [
        "cognition",
        "cognitive",
        "intelligence",
        "iq",
    ],
}


CLINICAL_KEYWORDS = {
    "autism": [
        "autism",
        "asd",
    ],

    "ADHD": [
        "adhd",
        "attention_deficit",
    ],

    "schizophrenia": [
        "schizophrenia",
        "psychosis",
        "scz",
    ],

    "major_depression": [
        "depression",
        "mdd",
    ],

    "bipolar_disorder": [
        "bipolar",
        "bd",
    ],

    "OCD": [
        "ocd",
        "obsessive",
    ],

    "epilepsy": [
        "epilepsy",
        "seizure",
    ],

    "Parkinson_disease": [
        "parkinson",
        "pd_",
    ],

    "Alzheimer_disease": [
        "alzheimer",
        "dementia",
        "ad_",
    ],

    "developmental_disorder": [
        "developmental_disorder",
        "neurodevelopmental",
        "intellectual_disability",
    ],
}


SOURCE_KEYWORDS = {
    "Neurosynth": [
        "neurosynth",
    ],

    "NeuroQuery": [
        "neuroquery",
    ],

    "NiMARE": [
        "nimare",
    ],

    "ENIGMA": [
        "enigma",
    ],

    "neuromaps": [
        "neuromaps",
    ],

    "BrainMap": [
        "brainmap",
    ],

    "HCP": [
        "hcp",
    ],
}


SUPPORTED_SUFFIXES = [
    ".tsv",
    ".tsv.gz",
    ".csv",
    ".csv.gz",
    ".txt",
    ".npy",
    ".npz",
    ".nii",
    ".nii.gz",
    ".gii",
    ".func.gii",
    ".shape.gii",
    ".dscalar.nii",
    ".dtseries.nii",
]


PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "nibabel",
    "nilearn",
    "neuromaps",
    "netneurotools",
    "brainstat",
    "enigmatoolbox",
    "nimare",
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

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def keyword_present(
    text: str,
    keyword: str,
) -> bool:
    text = text.lower()
    keyword = keyword.lower()

    # Short abbreviations such as IQ, ASD, OCD, SCZ and BD
    # must occur as standalone path/name tokens. This prevents
    # false matches such as "iq" inside "unique".
    if (
        len(keyword) <= 3
        and keyword.isalnum()
    ):
        pattern = (
            rf"(?<![a-z0-9])"
            rf"{re.escape(keyword)}"
            rf"(?![a-z0-9])"
        )

        return (
            re.search(
                pattern,
                text,
            )
            is not None
        )

    return keyword in text


def classify_keywords(
    text: str,
    vocabulary: dict[str, list[str]],
) -> list[str]:
    matches = []

    for category, keywords in vocabulary.items():
        if any(
            keyword_present(
                text,
                keyword,
            )
            for keyword in keywords
        ):
            matches.append(category)

    return matches


def supported_file(
    path: Path,
) -> bool:
    lower_name = path.name.lower()

    return any(
        lower_name.endswith(suffix)
        for suffix in SUPPORTED_SUFFIXES
    )


def inspect_tabular_header(
    path: Path,
) -> tuple[int, str]:
    lower_name = path.name.lower()

    if not lower_name.endswith(
        (
            ".tsv",
            ".tsv.gz",
            ".csv",
            ".csv.gz",
            ".txt",
        )
    ):
        return 0, ""

    separator = (
        ","
        if lower_name.endswith(
            (
                ".csv",
                ".csv.gz",
            )
        )
        else "\t"
    )

    try:
        header = pd.read_csv(
            path,
            sep=separator,
            compression="infer",
            nrows=0,
            low_memory=False,
        )

        columns = [
            str(column)
            for column in header.columns
        ]

        return (
            len(columns),
            "|".join(columns),
        )

    except Exception:
        return -1, ""


def main() -> None:
    required = [
        PHASE8C_COMPLETION,
        PHASE8C_HASH_MANIFEST,
        FIGURE56_PNG,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing Phase 8D1 prerequisites:\n"
            + "\n".join(missing)
        )

    phase8c = pd.read_csv(
        PHASE8C_COMPLETION,
        sep="\t",
        low_memory=False,
    )

    phase8c_completed = bool(
        "Phase8C_status"
        in phase8c.columns
        and str(
            phase8c[
                "Phase8C_status"
            ].iloc[0]
        ).strip().lower()
        == "completed"
    )

    manifest = pd.read_csv(
        PHASE8C_HASH_MANIFEST,
        sep="\t",
        low_memory=False,
    )

    figure_relative_path = str(
        FIGURE56_PNG.relative_to(PROJECT)
    )

    manifest_match = manifest.loc[
        manifest[
            "relative_path"
        ].eq(figure_relative_path)
    ]

    figure_hash = sha256_file(
        FIGURE56_PNG
    )

    figure_hash_matches_manifest = bool(
        len(manifest_match) == 1
        and str(
            manifest_match[
                "SHA256"
            ].iloc[0]
        )
        == figure_hash
    )

    candidate_rows = []

    scanned_files = 0

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if any(
                part in EXCLUDED_PARTS
                for part in path.parts
            ):
                continue

            if not supported_file(path):
                continue

            scanned_files += 1

            relative_path = str(
                path.relative_to(PROJECT)
            )

            searchable = relative_path.lower()

            cognitive = classify_keywords(
                searchable,
                COGNITIVE_KEYWORDS,
            )

            clinical = classify_keywords(
                searchable,
                CLINICAL_KEYWORDS,
            )

            sources = classify_keywords(
                searchable,
                SOURCE_KEYWORDS,
            )

            if not (
                cognitive
                or clinical
                or sources
            ):
                continue

            column_count, columns = (
                inspect_tabular_header(path)
            )

            candidate_rows.append(
                {
                    "relative_path":
                        relative_path,

                    "filename":
                        path.name,

                    "size_bytes":
                        path.stat().st_size,

                    "cognitive_categories":
                        "|".join(cognitive),

                    "clinical_categories":
                        "|".join(clinical),

                    "candidate_sources":
                        "|".join(sources),

                    "candidate_type":
                        (
                            "cognitive_and_clinical"
                            if cognitive and clinical
                            else (
                                "cognitive"
                                if cognitive
                                else (
                                    "clinical"
                                    if clinical
                                    else "source_named"
                                )
                            )
                        ),

                    "tabular_column_count":
                        column_count,

                    "tabular_columns":
                        columns,

                    "likely_surface_or_volume_map":
                        path.name.lower().endswith(
                            (
                                ".nii",
                                ".nii.gz",
                                ".gii",
                                ".func.gii",
                                ".shape.gii",
                                ".dscalar.nii",
                                ".dtseries.nii",
                            )
                        ),

                    "SHA256":
                        sha256_file(path),
                }
            )

    candidates = pd.DataFrame(
        candidate_rows
    )

    if candidates.empty:
        candidates = pd.DataFrame(
            columns=[
                "relative_path",
                "filename",
                "size_bytes",
                "cognitive_categories",
                "clinical_categories",
                "candidate_sources",
                "candidate_type",
                "tabular_column_count",
                "tabular_columns",
                "likely_surface_or_volume_map",
                "SHA256",
            ]
        )

    else:
        candidates = candidates.sort_values(
            [
                "candidate_type",
                "candidate_sources",
                "relative_path",
            ]
        ).reset_index(drop=True)

    candidates.to_csv(
        CANDIDATE_OUTPUT,
        sep="\t",
        index=False,
    )

    package_rows = []

    for package in PACKAGES:
        specification = importlib.util.find_spec(
            package
        )

        package_rows.append(
            {
                "package":
                    package,

                "installed":
                    specification is not None,

                "module_origin":
                    (
                        str(specification.origin)
                        if specification is not None
                        else ""
                    ),
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

    category_rows = []

    for category_type, vocabulary in [
        (
            "cognitive",
            COGNITIVE_KEYWORDS,
        ),
        (
            "clinical",
            CLINICAL_KEYWORDS,
        ),
        (
            "source",
            SOURCE_KEYWORDS,
        ),
    ]:
        column = (
            "cognitive_categories"
            if category_type == "cognitive"
            else (
                "clinical_categories"
                if category_type == "clinical"
                else "candidate_sources"
            )
        )

        for category in vocabulary:
            count = int(
                candidates[
                    column
                ]
                .fillna("")
                .str.split("|")
                .apply(
                    lambda values:
                        category in values
                )
                .sum()
            )

            category_rows.append(
                {
                    "category_type":
                        category_type,

                    "category":
                        category,

                    "candidate_files":
                        count,
                }
            )

    category_summary = pd.DataFrame(
        category_rows
    )

    category_summary.to_csv(
        CATEGORY_OUTPUT,
        sep="\t",
        index=False,
    )

    cognitive_candidates = int(
        candidates[
            "candidate_type"
        ].isin(
            [
                "cognitive",
                "cognitive_and_clinical",
            ]
        ).sum()
    )

    clinical_candidates = int(
        candidates[
            "candidate_type"
        ].isin(
            [
                "clinical",
                "cognitive_and_clinical",
            ]
        ).sum()
    )

    imaging_packages_installed = int(
        packages.loc[
            packages[
                "package"
            ].isin(
                [
                    "nibabel",
                    "nilearn",
                    "neuromaps",
                    "netneurotools",
                    "brainstat",
                    "enigmatoolbox",
                    "nimare",
                ]
            ),
            "installed",
        ].sum()
    )

    acquisition_plan = pd.DataFrame(
        [
            {
                "resource_family":
                    "cognitive_activation_maps",

                "preferred_sources":
                    "Neurosynth|NeuroQuery|NiMARE",

                "target_domains":
                    (
                        "attention|executive_control|memory|"
                        "language|social_cognition|emotion_affect|"
                        "reward_motivation|sensorimotor"
                    ),

                "preferred_space":
                    "MNI152_volume_or_fsaverage_surface",

                "required_processing":
                    (
                        "parcellate_or_transform_to_"
                        "Schaefer100_and_retain_LH50"
                    ),

                "local_candidates":
                    cognitive_candidates,
            },

            {
                "resource_family":
                    "clinical_cortical_maps",

                "preferred_sources":
                    "ENIGMA_or_primary_consortium_maps",

                "target_domains":
                    (
                        "autism|ADHD|schizophrenia|major_depression|"
                        "bipolar_disorder|OCD|epilepsy|"
                        "Parkinson_disease|Alzheimer_disease"
                    ),

                "preferred_space":
                    "surface_parcellated_or_vertexwise",

                "required_processing":
                    (
                        "transform_or_crosswalk_to_"
                        "Schaefer100_with_explicit_atlas_QC"
                    ),

                "local_candidates":
                    clinical_candidates,
            },
        ]
    )

    acquisition_plan.to_csv(
        ACQUISITION_PLAN_OUTPUT,
        sep="\t",
        index=False,
    )

    status = (
        "completed"
        if (
            phase8c_completed
            and figure_hash_matches_manifest
            and scanned_files >= 0
            and len(packages) == len(PACKAGES)
        )
        else "failed_validation"
    )

    completion = pd.DataFrame(
        [
            {
                "Phase8C_status_confirmed":
                    phase8c_completed,

                "canonical_Figure56_hash_matches_manifest":
                    figure_hash_matches_manifest,

                "search_roots_requested":
                    len(SEARCH_ROOTS),

                "search_roots_present":
                    sum(
                        root.exists()
                        for root in SEARCH_ROOTS
                    ),

                "supported_files_scanned":
                    scanned_files,

                "candidate_files":
                    len(candidates),

                "cognitive_candidate_files":
                    cognitive_candidates,

                "clinical_candidate_files":
                    clinical_candidates,

                "surface_or_volume_map_candidates":
                    int(
                        candidates[
                            "likely_surface_or_volume_map"
                        ].sum()
                    )
                    if not candidates.empty
                    else 0,

                "packages_audited":
                    len(packages),

                "specialized_imaging_packages_installed":
                    imaging_packages_installed,

                "local_resources_sufficient":
                    bool(
                        cognitive_candidates > 0
                        and clinical_candidates > 0
                    ),

                "external_resource_acquisition_likely_required":
                    not (
                        cognitive_candidates > 0
                        and clinical_candidates > 0
                    ),

                "ready_for_phase8D2":
                    status == "completed",

                "Phase8D1_status":
                    status,
            }
        ]
    )

    completion.to_csv(
        COMPLETION_OUTPUT,
        sep="\t",
        index=False,
    )

    log_text = "\n".join(
        [
            "===== PHASE 8D1 COMPLETION =====",
            completion.to_string(index=False),
            "",
            "===== PACKAGE AUDIT =====",
            packages.to_string(index=False),
            "",
            "===== CANDIDATE CATEGORY SUMMARY =====",
            category_summary.to_string(index=False),
            "",
            "===== TOP RESOURCE CANDIDATES =====",
            (
                candidates.head(40).to_string(index=False)
                if not candidates.empty
                else "No local cognitive or clinical map candidates found."
            ),
            "",
            "===== ACQUISITION PLAN =====",
            acquisition_plan.to_string(index=False),
        ]
    )

    LOG_FILE.write_text(
        log_text + "\n",
        encoding="utf-8",
    )

    print(log_text)

    if status != "completed":
        raise RuntimeError(
            "Phase 8D1 resource audit failed validation."
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            (
                "Phase 8D1 failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
