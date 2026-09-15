#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import platform
import sys
import tarfile
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import scipy
import statsmodels


PROJECT = Path(
    "."
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase7"
)

FIGURE_DIR = (
    PROJECT
    / "06_figures/main_figures/phase7"
)

LOG_DIR = (
    PROJECT
    / "09_pipeline_logs/phase7"
)

ARCHIVE_DIR = (
    PROJECT
    / "14_compressed_archives"
)

ARCHIVE_PATH = (
    ARCHIVE_DIR
    / "phase7_FINAL_core_outputs_2026-07-23.tar.gz"
)

ARCHIVE_SHA256_PATH = Path(
    str(
        ARCHIVE_PATH
    )
    + ".sha256"
)

AUDIT_OUTPUT = (
    TABLE_DIR
    / "phase7F2_final_output_audit.tsv"
)

MILESTONE_OUTPUT = (
    TABLE_DIR
    / "phase7F2_milestone_completion_audit.tsv"
)

FIGURE_AUDIT_OUTPUT = (
    TABLE_DIR
    / "phase7F2_figure_audit.tsv"
)

ENVIRONMENT_OUTPUT = (
    LOG_DIR
    / "phase7F2_environment_snapshot.tsv"
)

MANIFEST_OUTPUT = (
    LOG_DIR
    / "phase7F2_core_archive_manifest.tsv"
)

EXCLUSION_OUTPUT = (
    LOG_DIR
    / "phase7F2_archive_excluded_files.tsv"
)

README_OUTPUT = (
    TABLE_DIR
    / "phase7_FINAL_readme.txt"
)

COMPLETION_OUTPUT = (
    TABLE_DIR
    / "phase7_FINAL_completion_summary.tsv"
)

LOG_FILE = (
    LOG_DIR
    / "phase7F2_final_audit_archive_and_closure.log"
)


FINAL_ATLAS_FILE = (
    TABLE_DIR
    / "phase7F1_final_integrated_regulatory_atlas.tsv"
)

FINAL_CLASS_FILE = (
    TABLE_DIR
    / "phase7F1_integrated_evidence_class_summary.tsv"
)

FINAL_MECHANISTIC_FILE = (
    TABLE_DIR
    / "phase7F1_mechanistic_regulator_summary.tsv"
)

PHASE7F1_COMPLETION_FILE = (
    TABLE_DIR
    / "phase7F1_completion_summary.tsv"
)


EXPECTED_TFS = {
    "CIITA",
    "NRF1",
    "E2F1",
    "MYC",
    "E2F4",
    "REST",
    "RFX5",
    "RFXAP",
    "SRSF2",
    "E2F3",
}

EXPECTED_FIGURE_NUMBERS = [
    48,
    49,
    51,
    52,
    53,
    54,
]

EXPECTED_F1_COUNT = 6
EXPECTED_F2_COUNT = 2
EXPECTED_F3_COUNT = 2

MAX_ARCHIVE_FILE_BYTES = (
    300
    * 1024
    * 1024
)


for directory in [
    TABLE_DIR,
    FIGURE_DIR,
    LOG_DIR,
    ARCHIVE_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def read_table(
    path: Path,
    **kwargs: Any,
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        compression="infer",
        low_memory=False,
        **kwargs,
    )


def parse_boolean(
    value: Any,
) -> bool:
    return str(
        value
    ).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


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


def status_from_completion_file(
    path: Path,
) -> tuple[str, list[str]]:
    table = read_table(
        path
    )

    status_columns = [
        column
        for column in table.columns
        if str(
            column
        ).lower().endswith(
            "_status"
        )
    ]

    if not status_columns:
        return (
            "missing_status_column",
            [],
        )

    values = []

    for column in status_columns:
        values.extend(
            table[
                column
            ]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .tolist()
        )

    if (
        values
        and all(
            value
            == "completed"
            for value in values
        )
    ):
        return (
            "completed",
            status_columns,
        )

    return (
        "|".join(
            values
        )
        if values
        else "missing_status_value",
        status_columns,
    )


def resolve_completion_file(
    exact_names: list[str],
    fallback_pattern: str,
) -> Path:
    for name in exact_names:
        candidate = (
            TABLE_DIR
            / name
        )

        if candidate.exists():
            return candidate

    fallback = sorted(
        TABLE_DIR.glob(
            fallback_pattern
        )
    )

    if not fallback:
        raise FileNotFoundError(
            (
                "No completion summary found for "
                f"{fallback_pattern}."
            )
        )

    completed_candidates = []

    for candidate in fallback:
        status, _ = (
            status_from_completion_file(
                candidate
            )
        )

        if status == "completed":
            completed_candidates.append(
                candidate
            )

    if completed_candidates:
        return sorted(
            completed_candidates
        )[
            -1
        ]

    return fallback[
        -1
    ]


def validate_pdf(
    path: Path,
) -> tuple[bool, int | None, str]:
    if not path.exists():
        return (
            False,
            None,
            "missing",
        )

    with path.open(
        "rb"
    ) as handle:
        header = handle.read(
            5
        )

        handle.seek(
            max(
                0,
                path.stat().st_size
                - 2048,
            )
        )

        trailer = handle.read()

    binary_valid = (
        header
        == b"%PDF-"
        and b"%%EOF"
        in trailer
    )

    page_count = None
    validator = (
        "binary_header_and_EOF"
    )

    try:
        from pypdf import PdfReader

        reader = PdfReader(
            str(
                path
            )
        )

        page_count = len(
            reader.pages
        )

        binary_valid = (
            binary_valid
            and page_count
            >= 1
        )

        validator = "pypdf"

    except ImportError:
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(
                str(
                    path
                )
            )

            page_count = len(
                reader.pages
            )

            binary_valid = (
                binary_valid
                and page_count
                >= 1
            )

            validator = "PyPDF2"

        except ImportError:
            pass

    return (
        binary_valid,
        page_count,
        validator,
    )


def select_figure_file(
    figure_number: int,
    extension: str,
) -> Path:
    exact_preference = {
        52:
            FIGURE_DIR
            / (
                "Figure52_cross_fitted_TF_"
                "program_impact_atlas."
                + extension
            ),

        53:
            FIGURE_DIR
            / (
                "Figure53_cross_fitted_TF_"
                "perturbation_robustness."
                + extension
            ),

        54:
            FIGURE_DIR
            / (
                "Figure54_final_integrated_"
                "developmental_regulatory_atlas."
                + extension
            ),
    }

    preferred = exact_preference.get(
        figure_number
    )

    if (
        preferred is not None
        and preferred.exists()
    ):
        return preferred

    candidates = sorted(
        FIGURE_DIR.glob(
            f"Figure{figure_number}*.{extension}"
        )
    )

    canonical = [
        candidate
        for candidate in candidates
        if not any(
            token
            in candidate.name.lower()
            for token in [
                "_original",
                "_revised",
                "_before_",
                "backup",
            ]
        )
    ]

    if canonical:
        return canonical[
            0
        ]

    if candidates:
        return candidates[
            0
        ]

    return (
        FIGURE_DIR
        / f"Figure{figure_number}_MISSING.{extension}"
    )


def should_exclude_from_archive(
    path: Path,
) -> tuple[bool, str]:
    lower_name = path.name.lower()

    excluded_tokens = [
        "_before_",
        "_original",
        "_revised",
        ".bak",
        "~",
    ]

    if any(
        token
        in lower_name
        for token in excluded_tokens
    ):
        return (
            True,
            "superseded_or_backup_file",
        )

    if path.stat().st_size > MAX_ARCHIVE_FILE_BYTES:
        return (
            True,
            "larger_than_300_MB_core_archive_limit",
        )

    if path == ARCHIVE_PATH:
        return (
            True,
            "archive_cannot_include_itself",
        )

    return (
        False,
        "",
    )


def create_archive(
    archive_path: Path,
    files: list[Path],
) -> None:
    temporary_path = Path(
        str(
            archive_path
        )
        + ".partial"
    )

    temporary_path.unlink(
        missing_ok=True
    )

    with tarfile.open(
        temporary_path,
        mode="w:gz",
        compresslevel=6,
    ) as archive:
        for file_path in files:
            archive.add(
                file_path,
                arcname=str(
                    file_path.relative_to(
                        PROJECT
                    )
                ),
                recursive=False,
            )

    temporary_path.replace(
        archive_path
    )


def validate_archive(
    archive_path: Path,
    required_members: list[str],
) -> tuple[bool, int, list[str]]:
    with tarfile.open(
        archive_path,
        mode="r:gz",
    ) as archive:
        members = archive.getmembers()

        names = {
            member.name
            for member in members
        }

        missing_required = [
            member
            for member in required_members
            if member not in names
        ]

        for member in members:
            if member.isfile():
                extracted = archive.extractfile(
                    member
                )

                if extracted is not None:
                    extracted.read(
                        1
                    )

    return (
        len(
            missing_required
        )
        == 0,
        len(
            members
        ),
        missing_required,
    )


def main() -> None:
    # ================================================================
    # Milestone completion audit
    # ================================================================

    milestone_definitions = {
        "Phase7A": (
            [
                "phase7A2_completion_summary.tsv",
                "phase7A_completion_summary.tsv",
                "phase7A1_completion_summary.tsv",
            ],
            "phase7A*completion_summary.tsv",
        ),

        "Phase7B": (
            [
                "phase7B6_completion_summary.tsv",
                "phase7B_completion_summary.tsv",
            ],
            "phase7B*completion_summary.tsv",
        ),

        "Phase7C": (
            [
                "phase7C4_completion_summary.tsv",
                "phase7C_completion_summary.tsv",
            ],
            "phase7C*completion_summary.tsv",
        ),

        "Phase7D": (
            [
                "phase7D_completion_summary.tsv",
                "phase7D4_completion_summary.tsv",
            ],
            "phase7D*completion_summary.tsv",
        ),

        "Phase7E": (
            [
                "phase7E_completion_summary.tsv",
            ],
            "phase7E_completion_summary.tsv",
        ),

        "Phase7F1": (
            [
                "phase7F1_completion_summary.tsv",
            ],
            "phase7F1_completion_summary.tsv",
        ),
    }

    milestone_rows = []

    for milestone, (
        exact_names,
        fallback_pattern,
    ) in milestone_definitions.items():
        completion_file = (
            resolve_completion_file(
                exact_names,
                fallback_pattern,
            )
        )

        status, status_columns = (
            status_from_completion_file(
                completion_file
            )
        )

        milestone_rows.append(
            {
                "milestone":
                    milestone,

                "completion_file":
                    str(
                        completion_file.relative_to(
                            PROJECT
                        )
                    ),

                "status_columns":
                    "|".join(
                        status_columns
                    ),

                "status":
                    status,

                "completed":
                    status
                    == "completed",
            }
        )

    milestone_audit = pd.DataFrame(
        milestone_rows
    )

    milestone_audit.to_csv(
        MILESTONE_OUTPUT,
        sep="\t",
        index=False,
    )

    # ================================================================
    # Final atlas validation
    # ================================================================

    required_core_files = [
        FINAL_ATLAS_FILE,
        FINAL_CLASS_FILE,
        FINAL_MECHANISTIC_FILE,
        PHASE7F1_COMPLETION_FILE,
        TABLE_DIR
        / "phase7D4_final_primary_TF_cis_regulatory_atlas.tsv",
        TABLE_DIR
        / "phase7E3_final_TF_perturbation_atlas.tsv",
        TABLE_DIR
        / "phase7B6_final_BrainSpan_TF_regulatory_atlas.tsv.gz",
        TABLE_DIR
        / "phase7B6_primary_mechanistic_TF_shortlist.tsv",
    ]

    missing_core_files = [
        str(
            path
        )
        for path in required_core_files
        if not path.exists()
    ]

    if missing_core_files:
        raise FileNotFoundError(
            "Missing required final Phase 7 files:\n"
            + "\n".join(
                missing_core_files
            )
        )

    final_atlas = read_table(
        FINAL_ATLAS_FILE
    )

    final_classes = read_table(
        FINAL_CLASS_FILE
    )

    completion_f1 = read_table(
        PHASE7F1_COMPLETION_FILE
    )

    required_final_columns = [
        "final_integrated_rank",
        "regulator",
        "final_phase7_evidence_class",
        "descriptive_multilayer_evidence_score",
        "developmental_program_direction",
        "final_integrated_interpretation",
    ]

    missing_final_columns = [
        column
        for column in required_final_columns
        if column not in final_atlas.columns
    ]

    if missing_final_columns:
        raise RuntimeError(
            (
                "Final atlas missing columns: "
                + "|".join(
                    missing_final_columns
                )
            )
        )

    observed_tfs = set(
        final_atlas[
            "regulator"
        ].astype(str)
    )

    class_counts = (
        final_atlas[
            "final_phase7_evidence_class"
        ]
        .value_counts()
    )

    f1_count = int(
        class_counts.get(
            (
                "Tier_F1_convergent_temporal_"
                "cis_perturbational_regulator"
            ),
            0,
        )
    )

    f2_count = int(
        class_counts.get(
            (
                "Tier_F2_primary_perturbational_"
                "regulator_without_direct_cis_validation"
            ),
            0,
        )
    )

    f3_count = int(
        class_counts.get(
            (
                "Tier_F3_small_regulon_"
                "perturbational_sensitivity_regulator"
            ),
            0,
        )
    )

    atlas_checks = {
        "final_atlas_has_10_rows":
            len(
                final_atlas
            )
            == 10,

        "final_atlas_has_10_unique_TFs":
            final_atlas[
                "regulator"
            ].nunique()
            == 10,

        "expected_TF_set_exact":
            observed_tfs
            == EXPECTED_TFS,

        "integrated_rank_is_1_to_10":
            final_atlas[
                "final_integrated_rank"
            ].astype(
                int
            ).tolist()
            == list(
                range(
                    1,
                    11,
                )
            ),

        "critical_fields_complete":
            not final_atlas[
                required_final_columns
            ].isna().any().any(),

        "Tier_F1_count_is_6":
            f1_count
            == EXPECTED_F1_COUNT,

        "Tier_F2_count_is_2":
            f2_count
            == EXPECTED_F2_COUNT,

        "Tier_F3_count_is_2":
            f3_count
            == EXPECTED_F3_COUNT,

        "Phase7F1_status_completed":
            (
                "Phase7F1_status"
                in completion_f1.columns
                and str(
                    completion_f1[
                        "Phase7F1_status"
                    ].iloc[
                        0
                    ]
                ).strip().lower()
                == "completed"
            ),
    }

    # ================================================================
    # Figure audit
    # ================================================================

    figure_rows = []

    for figure_number in EXPECTED_FIGURE_NUMBERS:
        png_path = select_figure_file(
            figure_number,
            "png",
        )

        pdf_path = select_figure_file(
            figure_number,
            "pdf",
        )

        pdf_valid, (
            pdf_pages
        ), pdf_validator = validate_pdf(
            pdf_path
        )

        png_valid = (
            png_path.exists()
            and png_path.stat().st_size
            > 50000
        )

        figure_rows.append(
            {
                "figure_number":
                    figure_number,

                "PNG_path":
                    (
                        str(
                            png_path.relative_to(
                                PROJECT
                            )
                        )
                        if png_path.exists()
                        else str(
                            png_path
                        )
                    ),

                "PNG_size_bytes":
                    (
                        png_path.stat().st_size
                        if png_path.exists()
                        else 0
                    ),

                "PNG_valid":
                    png_valid,

                "PDF_path":
                    (
                        str(
                            pdf_path.relative_to(
                                PROJECT
                            )
                        )
                        if pdf_path.exists()
                        else str(
                            pdf_path
                        )
                    ),

                "PDF_size_bytes":
                    (
                        pdf_path.stat().st_size
                        if pdf_path.exists()
                        else 0
                    ),

                "PDF_pages":
                    pdf_pages,

                "PDF_validator":
                    pdf_validator,

                "PDF_valid":
                    (
                        pdf_valid
                        and pdf_path.stat().st_size
                        > 10000
                        if pdf_path.exists()
                        else False
                    ),

                "figure_pair_valid":
                    (
                        png_valid
                        and pdf_valid
                        and pdf_path.exists()
                        and pdf_path.stat().st_size
                        > 10000
                    ),
            }
        )

    figure_audit = pd.DataFrame(
        figure_rows
    )

    figure_audit.to_csv(
        FIGURE_AUDIT_OUTPUT,
        sep="\t",
        index=False,
    )

    # ================================================================
    # Environment snapshot
    # ================================================================

    environment = pd.DataFrame(
        [
            {
                "component":
                    "Python",

                "version":
                    platform.python_version(),
            },

            {
                "component":
                    "pandas",

                "version":
                    pd.__version__,
            },

            {
                "component":
                    "numpy",

                "version":
                    np.__version__,
            },

            {
                "component":
                    "scipy",

                "version":
                    scipy.__version__,
            },

            {
                "component":
                    "statsmodels",

                "version":
                    statsmodels.__version__,
            },

            {
                "component":
                    "matplotlib",

                "version":
                    matplotlib.__version__,
            },

            {
                "component":
                    "platform",

                "version":
                    platform.platform(),
            },
        ]
    )

    environment.to_csv(
        ENVIRONMENT_OUTPUT,
        sep="\t",
        index=False,
    )

    # ================================================================
    # Consolidated audit
    # ================================================================

    audit_rows = []

    for check_name, passed in (
        atlas_checks.items()
    ):
        audit_rows.append(
            {
                "audit_category":
                    "final_atlas",

                "audit_item":
                    check_name,

                "passed":
                    bool(
                        passed
                    ),

                "details":
                    "",
            }
        )

    for row in milestone_audit.itertuples(
        index=False
    ):
        audit_rows.append(
            {
                "audit_category":
                    "milestone_completion",

                "audit_item":
                    row.milestone,

                "passed":
                    bool(
                        row.completed
                    ),

                "details":
                    (
                        f"{row.status}; "
                        f"{row.completion_file}"
                    ),
            }
        )

    for row in figure_audit.itertuples(
        index=False
    ):
        audit_rows.append(
            {
                "audit_category":
                    "canonical_figure_pair",

                "audit_item":
                    f"Figure{row.figure_number}",

                "passed":
                    bool(
                        row.figure_pair_valid
                    ),

                "details":
                    (
                        f"PNG={row.PNG_size_bytes}; "
                        f"PDF={row.PDF_size_bytes}; "
                        f"pages={row.PDF_pages}; "
                        f"validator={row.PDF_validator}"
                    ),
            }
        )

    final_audit = pd.DataFrame(
        audit_rows
    )

    final_audit.to_csv(
        AUDIT_OUTPUT,
        sep="\t",
        index=False,
    )

    prearchive_checks_passed = bool(
        final_audit[
            "passed"
        ].all()
    )

    if not prearchive_checks_passed:
        failed = final_audit.loc[
            ~final_audit[
                "passed"
            ]
        ]

        raise RuntimeError(
            (
                "Phase 7 final audit failed:\n"
                + failed.to_string(
                    index=False
                )
            )
        )

    # ================================================================
    # README and preliminary completion summary
    # ================================================================

    readme_text = f"""PHASE 7 FINAL CORE OUTPUTS

Project:
Developmental transcriptome-to-structure mechanisms underlying
human cortical temporal hierarchy

Phase:
Phase 7 — Regulatory and cellular mechanisms

Final primary transcription factors:
{", ".join(final_atlas["regulator"].astype(str).tolist())}

Final evidence classes:
F1 convergent temporal/cis/perturbational regulators: {f1_count}
F2 primary perturbational regulators without direct cis validation: {f2_count}
F3 small-regulon perturbational sensitivity regulators: {f3_count}

Key interpretation:
The integrated evidence score is descriptive. It is not a probability,
causal estimate, therapeutic ranking, or proof of experimental TF
perturbation. RFXAP and SRSF2 remain small-regulon sensitivity results.

Canonical final table:
07_tables/main_tables/phase7/
phase7F1_final_integrated_regulatory_atlas.tsv

Canonical final figure:
06_figures/main_figures/phase7/
Figure54_final_integrated_developmental_regulatory_atlas.png
Figure54_final_integrated_developmental_regulatory_atlas.pdf

Archive:
14_compressed_archives/
{ARCHIVE_PATH.name}

Large reference resources exceeding 300 MB and superseded backup files
are intentionally omitted from the core archive. Their processed
derivatives, resource audits, checksums and pipeline logs are retained.
"""

    README_OUTPUT.write_text(
        readme_text,
        encoding="utf-8",
    )

    completion = pd.DataFrame(
        [
            {
                "Phase7_milestones_audited":
                    len(
                        milestone_audit
                    ),

                "Phase7_milestones_completed":
                    int(
                        milestone_audit[
                            "completed"
                        ].sum()
                    ),

                "canonical_figure_pairs_audited":
                    len(
                        figure_audit
                    ),

                "canonical_figure_pairs_valid":
                    int(
                        figure_audit[
                            "figure_pair_valid"
                        ].sum()
                    ),

                "final_primary_TFs":
                    len(
                        final_atlas
                    ),

                "Tier_F1_regulators":
                    f1_count,

                "Tier_F2_regulators":
                    f2_count,

                "Tier_F3_regulators":
                    f3_count,

                "final_atlas_valid":
                    all(
                        atlas_checks.values()
                    ),

                "all_prearchive_audits_passed":
                    prearchive_checks_passed,

                "large_raw_resources_in_core_archive":
                    False,

                "archive_filename":
                    ARCHIVE_PATH.name,

                "archive_validation_status":
                    "pending",

                "archive_entries":
                    0,

                "archive_size_bytes":
                    0,

                "archive_SHA256_generated":
                    False,

                "causal_interpretation_permitted":
                    False,

                "Phase7_status":
                    "completed",
            }
        ]
    )

    completion.to_csv(
        COMPLETION_OUTPUT,
        sep="\t",
        index=False,
    )

    # ================================================================
    # Build curated archive file list
    # ================================================================

    candidate_files: set[Path] = set()

    explicit_roots = [
        PROJECT
        / "04_scripts/09_regulatory_mechanisms",

        FIGURE_DIR,
        TABLE_DIR,
        LOG_DIR,
    ]

    for root in explicit_roots:
        if root.exists():
            for path in root.rglob(
                "*"
            ):
                if path.is_file():
                    candidate_files.add(
                        path
                    )

    for broad_root in [
        PROJECT
        / "02_raw_data",

        PROJECT
        / "03_processed_data",
    ]:
        if not broad_root.exists():
            continue

        for path in broad_root.rglob(
            "*"
        ):
            if not path.is_file():
                continue

            relative_lower = str(
                path.relative_to(
                    PROJECT
                )
            ).lower()

            if "phase7" in relative_lower:
                candidate_files.add(
                    path
                )

    candidate_files.update(
        [
            AUDIT_OUTPUT,
            MILESTONE_OUTPUT,
            FIGURE_AUDIT_OUTPUT,
            ENVIRONMENT_OUTPUT,
            README_OUTPUT,
            COMPLETION_OUTPUT,
        ]
    )

    included_files = []
    excluded_rows = []

    for path in sorted(
        candidate_files,
        key=lambda item:
            str(
                item.relative_to(
                    PROJECT
                )
            ),
    ):
        excluded, reason = (
            should_exclude_from_archive(
                path
            )
        )

        if excluded:
            excluded_rows.append(
                {
                    "relative_path":
                        str(
                            path.relative_to(
                                PROJECT
                            )
                        ),

                    "size_bytes":
                        path.stat().st_size,

                    "exclusion_reason":
                        reason,
                }
            )

        else:
            included_files.append(
                path
            )

    manifest_rows = []

    for path in included_files:
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

                "SHA256":
                    sha256_file(
                        path
                    ),
            }
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest.to_csv(
        MANIFEST_OUTPUT,
        sep="\t",
        index=False,
    )

    excluded = pd.DataFrame(
        excluded_rows,
        columns=[
            "relative_path",
            "size_bytes",
            "exclusion_reason",
        ],
    )

    excluded.to_csv(
        EXCLUSION_OUTPUT,
        sep="\t",
        index=False,
    )

    archive_files = sorted(
        set(
            included_files
            + [
                MANIFEST_OUTPUT,
                EXCLUSION_OUTPUT,
            ]
        ),
        key=lambda item:
            str(
                item.relative_to(
                    PROJECT
                )
            ),
    )

    required_archive_members = [
        str(
            FINAL_ATLAS_FILE.relative_to(
                PROJECT
            )
        ),

        str(
            FINAL_CLASS_FILE.relative_to(
                PROJECT
            )
        ),

        str(
            COMPLETION_OUTPUT.relative_to(
                PROJECT
            )
        ),

        str(
            README_OUTPUT.relative_to(
                PROJECT
            )
        ),

        str(
            (
                FIGURE_DIR
                / (
                    "Figure54_final_integrated_"
                    "developmental_regulatory_atlas.png"
                )
            ).relative_to(
                PROJECT
            )
        ),

        str(
            (
                FIGURE_DIR
                / (
                    "Figure54_final_integrated_"
                    "developmental_regulatory_atlas.pdf"
                )
            ).relative_to(
                PROJECT
            )
        ),

        str(
            MANIFEST_OUTPUT.relative_to(
                PROJECT
            )
        ),
    ]

    # First archive pass
    create_archive(
        ARCHIVE_PATH,
        archive_files,
    )

    archive_valid, (
        archive_entries
    ), missing_members = validate_archive(
        ARCHIVE_PATH,
        required_archive_members,
    )

    if not archive_valid:
        raise RuntimeError(
            (
                "Archive validation failed. Missing members: "
                + "|".join(
                    missing_members
                )
            )
        )

    # Update completion summary and rebuild archive so the archived
    # summary records final archive validation.
    completion.loc[
        0,
        "archive_validation_status",
    ] = "completed"

    completion.loc[
        0,
        "archive_entries",
    ] = archive_entries

    completion.loc[
        0,
        "archive_size_bytes",
    ] = ARCHIVE_PATH.stat().st_size

    completion.loc[
        0,
        "archive_SHA256_generated",
    ] = True

    completion.to_csv(
        COMPLETION_OUTPUT,
        sep="\t",
        index=False,
    )

    create_archive(
        ARCHIVE_PATH,
        archive_files,
    )

    final_archive_valid, (
        final_archive_entries
    ), final_missing_members = validate_archive(
        ARCHIVE_PATH,
        required_archive_members,
    )

    if not final_archive_valid:
        raise RuntimeError(
            (
                "Final archive validation failed. Missing members: "
                + "|".join(
                    final_missing_members
                )
            )
        )

    archive_sha256 = sha256_file(
        ARCHIVE_PATH
    )

    ARCHIVE_SHA256_PATH.write_text(
        (
            f"{archive_sha256}  "
            f"{ARCHIVE_PATH.name}\n"
        ),
        encoding="utf-8",
    )

    # Final external summary update. The version contained in the
    # archive already records completed validation; this update only
    # refreshes final compressed size and entry count.
    completion.loc[
        0,
        "archive_entries",
    ] = final_archive_entries

    completion.loc[
        0,
        "archive_size_bytes",
    ] = ARCHIVE_PATH.stat().st_size

    completion.to_csv(
        COMPLETION_OUTPUT,
        sep="\t",
        index=False,
    )

    log_text = "\n".join(
        [
            "===== PHASE 7 FINAL COMPLETION =====",
            completion.to_string(
                index=False
            ),
            "",
            "===== MILESTONE AUDIT =====",
            milestone_audit.to_string(
                index=False
            ),
            "",
            "===== FIGURE AUDIT =====",
            figure_audit.to_string(
                index=False
            ),
            "",
            "===== FINAL INTEGRATED TF ATLAS =====",
            final_atlas[
                [
                    "final_integrated_rank",
                    "regulator",
                    "final_phase7_evidence_class",
                    "descriptive_multilayer_evidence_score",
                    "developmental_program_direction",
                ]
            ].to_string(
                index=False
            ),
            "",
            "===== ARCHIVE =====",
            f"Path\t{ARCHIVE_PATH}",
            f"Size_bytes\t{ARCHIVE_PATH.stat().st_size}",
            f"Entries\t{final_archive_entries}",
            f"SHA256\t{archive_sha256}",
            f"Excluded_files\t{len(excluded)}",
        ]
    )

    LOG_FILE.write_text(
        log_text + "\n",
        encoding="utf-8",
    )

    print(
        log_text
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            (
                "Phase 7F2 failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
