#!/usr/bin/env python3
"""Phase 8D3E: canonical closure of Phase 8D3."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]

PHASE8D = (
    ROOT
    / "03_processed_data"
    / "functional_imaging"
    / "phase8D"
)

TABLES = (
    ROOT
    / "07_tables"
    / "main_tables"
    / "phase8"
)

SCRIPTS = (
    ROOT
    / "04_scripts"
    / "10_functional_imaging"
    / "phase8D"
)

LOGS = (
    ROOT
    / "09_pipeline_logs"
    / "phase8"
)

FIGURES = (
    ROOT
    / "08_figures"
    / "phase8"
)


D3A_DIR = (
    PHASE8D
    / "phase8D3A_canonical_lock"
)

D3A_MANIFEST = (
    D3A_DIR
    / "phase8D3A_locked_inputs_manifest.tsv"
)

D3A_ARCHIVE = (
    D3A_DIR
    / "phase8D3A_canonical_lock.tar.gz"
)

D3A_COMPLETION = (
    TABLES
    / "phase8D3A_completion_summary.tsv"
)


D3B_DIR = (
    PHASE8D
    / "phase8D3B_cross_modal_synthesis"
)

D3B_COMPLETION = (
    TABLES
    / "phase8D3B_completion_summary.tsv"
)


D3C_DIR = (
    PHASE8D
    / "phase8D3C_evidence_hierarchy"
)

D3C_AUDIT = (
    TABLES
    / "phase8D3C_evidence_hierarchy_audit.tsv"
)

D3C_COMPLETION = (
    TABLES
    / "phase8D3C_completion_summary.tsv"
)


D3D_R1_DIR = (
    PHASE8D
    / "phase8D3D_R1"
)

D3D_R1_COMPLETION = (
    TABLES
    / "phase8D3D_R1_completion_summary.tsv"
)


D3D_R2_DIR = (
    PHASE8D
    / "phase8D3D_R2"
)

D3D_R2_FIGURES = (
    FIGURES
    / "phase8D3D_R2"
)

D3D_R2_AUDIT = (
    TABLES
    / "phase8D3D_R2_figure_audit.tsv"
)

D3D_R2_COMPLETION = (
    TABLES
    / "phase8D3D_R2_completion_summary.tsv"
)

D3D_R2_SIGNOFF = (
    TABLES
    / "phase8D3D_R2_visual_signoff.tsv"
)


D3C_SCRIPT = (
    SCRIPTS
    / "03D3C_build_definitive_evidence_hierarchy.py"
)

D3D_R1_SCRIPT = (
    SCRIPTS
    / "03D3D_R1_revise_integrated_cross_modal_figure.py"
)

D3D_R2_SCRIPT = (
    SCRIPTS
    / "03D3D_R2_finalize_integrated_cross_modal_figure.py"
)

D3E_SCRIPT = Path(__file__).resolve()


D3C_LOG = (
    LOGS
    / "phase8D3C_evidence_hierarchy_console.log"
)

D3D_R1_LOG = (
    LOGS
    / "phase8D3D_R1_console.log"
)

D3D_R2_LOG = (
    LOGS
    / "phase8D3D_R2_final_console.log"
)


OUT = (
    PHASE8D
    / "phase8D3E_canonical_closure"
)

SNAPSHOT = (
    OUT
    / "snapshot"
)

MANIFEST_OUT = (
    OUT
    / "phase8D3E_canonical_manifest.tsv"
)

UPSTREAM_ANCHORS_OUT = (
    OUT
    / "phase8D3E_upstream_archive_anchors.tsv"
)

ARCHIVE_OUT = (
    OUT
    / "phase8D3E_canonical_closure.tar.gz"
)

ARCHIVE_SHA_OUT = (
    OUT
    / "phase8D3E_canonical_closure.tar.gz.sha256"
)

AUDIT_OUT = (
    TABLES
    / "phase8D3E_closure_audit.tsv"
)

COMPLETION_OUT = (
    TABLES
    / "phase8D3E_completion_summary.tsv"
)


EXPECTED_D3A_ARCHIVE_SHA256 = (
    "94831a7e8cb13494472f5dd705f4b3af"
    "4cb1e90e81f2cb75c7a4d4ea4579c067"
)


class ValidationError(RuntimeError):
    pass


def require_file(path: Path) -> None:
    if not path.is_file():
        raise ValidationError(
            f"Required file not found: {path}"
        )


def require_directory(path: Path) -> None:
    if not path.is_dir():
        raise ValidationError(
            f"Required directory not found: {path}"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def read_tsv(path: Path) -> tuple[
    list[str],
    list[dict[str, str]],
]:
    require_file(path)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        rows = list(reader)
        fields = list(
            reader.fieldnames
            or []
        )

    return fields, rows


def write_tsv(
    path: Path,
    fields: list[str],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: str(
                        row.get(
                            key,
                            "",
                        )
                    )
                    for key in fields
                }
            )

    temporary.replace(path)


def require_single_row(
    path: Path,
) -> dict[str, str]:
    _, rows = read_tsv(path)

    if len(rows) != 1:
        raise ValidationError(
            f"{path.name} must contain exactly one row."
        )

    return rows[0]


ORIGINAL_D3D_FIGURES = (
    FIGURES
    / "phase8D3D"
)

R1_FIGURES = (
    FIGURES
    / "phase8D3D_R1"
)

R2_FIGURES = (
    FIGURES
    / "phase8D3D_R2"
)


def validate_passed_audit(
    path: Path,
) -> int:
    fields, rows = read_tsv(
        path
    )

    if "passed" not in fields:
        raise ValidationError(
            f"Audit table has no 'passed' column: {path}"
        )

    if not rows:
        raise ValidationError(
            f"Audit table is empty: {path}"
        )

    failed_rows = [
        row
        for row in rows
        if not as_bool(
            row.get(
                "passed",
                False,
            )
        )
    ]

    if failed_rows:
        failed_items = [
            row.get(
                "item",
                "unlabelled_item",
            )
            for row in failed_rows
        ]

        raise ValidationError(
            f"Audit failures detected in {path.name}: "
            f"{failed_items}"
        )

    return len(rows)


def validate_status(
    row: dict[str, str],
    column: str,
    allowed: set[str],
    source: Path,
) -> str:
    if column not in row:
        raise ValidationError(
            f"Missing status column {column} "
            f"in {source.name}"
        )

    status = str(
        row[
            column
        ]
    ).strip()

    if status not in allowed:
        raise ValidationError(
            f"Unexpected {column} in {source.name}: "
            f"{status}"
        )

    return status


def validate_phase8D3A() -> dict[str, Any]:
    require_directory(
        D3A_DIR
    )

    require_file(
        D3A_MANIFEST
    )

    require_file(
        D3A_ARCHIVE
    )

    require_file(
        D3A_COMPLETION
    )

    completion = require_single_row(
        D3A_COMPLETION
    )

    validate_status(
        completion,
        "Phase8D3A_status",
        {
            "completed",
        },
        D3A_COMPLETION,
    )

    if (
        "ready_for_phase8D3B_cross_modal_synthesis"
        in completion
        and not as_bool(
            completion[
                "ready_for_phase8D3B_cross_modal_synthesis"
            ]
        )
    ):
        raise ValidationError(
            "Phase 8D3A is not marked ready "
            "for Phase 8D3B."
        )

    fields, manifest_rows = read_tsv(
        D3A_MANIFEST
    )

    if len(manifest_rows) != 75:
        raise ValidationError(
            "Expected 75 Phase 8D3A locked files; "
            f"found {len(manifest_rows)}."
        )

    if "hash_match" not in fields:
        raise ValidationError(
            "Phase 8D3A manifest does not contain "
            "the hash_match column."
        )

    failed_hash_rows = [
        row
        for row in manifest_rows
        if not as_bool(
            row.get(
                "hash_match",
                False,
            )
        )
    ]

    if failed_hash_rows:
        raise ValidationError(
            "One or more Phase 8D3A manifest "
            "hash checks failed."
        )

    observed_archive_hash = sha256(
        D3A_ARCHIVE
    )

    if (
        observed_archive_hash
        != EXPECTED_D3A_ARCHIVE_SHA256
    ):
        raise ValidationError(
            "Phase 8D3A archive SHA256 mismatch. "
            f"Observed: {observed_archive_hash}"
        )

    return {
        "locked_files":
            len(manifest_rows),
        "archive_size_bytes":
            D3A_ARCHIVE.stat().st_size,
        "archive_SHA256":
            observed_archive_hash,
    }


def validate_phase8D3B() -> dict[str, Any]:
    require_directory(
        D3B_DIR
    )

    require_file(
        D3B_COMPLETION
    )

    completion = require_single_row(
        D3B_COMPLETION
    )

    validate_status(
        completion,
        "Phase8D3B_status",
        {
            "completed",
        },
        D3B_COMPLETION,
    )

    if not as_bool(
        completion.get(
            "ready_for_phase8D3C_evidence_hierarchy",
            False,
        )
    ):
        raise ValidationError(
            "Phase 8D3B is not marked ready "
            "for Phase 8D3C."
        )

    return {
        "DTHI_maps":
            int(
                completion.get(
                    "DTHI_maps_synthesized",
                    completion.get(
                        "DTHI_maps",
                        10,
                    ),
                )
            ),
        "priority_pairs":
            int(
                completion.get(
                    "priority_associations",
                    completion.get(
                        "priority_pairs",
                        5,
                    ),
                )
            ),
    }


def validate_phase8D3C() -> dict[str, Any]:
    require_directory(
        D3C_DIR
    )

    require_file(
        D3C_COMPLETION
    )

    require_file(
        D3C_AUDIT
    )

    completion = require_single_row(
        D3C_COMPLETION
    )

    validate_status(
        completion,
        "Phase8D3C_status",
        {
            "completed",
        },
        D3C_COMPLETION,
    )

    if not as_bool(
        completion.get(
            "all_evidence_hierarchy_audits_passed",
            False,
        )
    ):
        raise ValidationError(
            "Phase 8D3C completion summary "
            "does not report passed audits."
        )

    if not as_bool(
        completion.get(
            "ready_for_phase8D3D_integrated_figure",
            False,
        )
    ):
        raise ValidationError(
            "Phase 8D3C is not marked ready "
            "for Phase 8D3D."
        )

    if as_bool(
        completion.get(
            "statistics_recomputed",
            True,
        )
    ):
        raise ValidationError(
            "Phase 8D3C unexpectedly reports "
            "recomputed statistics."
        )

    audit_rows = validate_passed_audit(
        D3C_AUDIT
    )

    expected_counts = {
        "associations_classified":
            160,
        "DTHI_maps_classified":
            10,
        "priority_associations_reported":
            5,
        "Tier1_global_FWER_associations":
            1,
        "Tier2_family_corrected_associations":
            4,
        "Tier3A_nominal_associations":
            13,
        "Tier3B_descriptive_associations":
            142,
    }

    for column, expected in expected_counts.items():
        observed = int(
            completion.get(
                column,
                -1,
            )
        )

        if observed != expected:
            raise ValidationError(
                f"Unexpected Phase 8D3C {column}: "
                f"{observed}; expected {expected}."
            )

    return {
        **expected_counts,
        "audit_rows":
            audit_rows,
    }


def validate_phase8D3D_R1() -> dict[str, Any]:
    require_file(
        D3D_R1_COMPLETION
    )

    completion = require_single_row(
        D3D_R1_COMPLETION
    )

    validate_status(
        completion,
        "Phase8D3D_R1_status",
        {
            "completed_pending_visual_review",
            "completed",
        },
        D3D_R1_COMPLETION,
    )

    if not as_bool(
        completion.get(
            "all_technical_audits_passed",
            False,
        )
    ):
        raise ValidationError(
            "Phase 8D3D-R1 technical audits "
            "did not pass."
        )

    if as_bool(
        completion.get(
            "statistics_recomputed",
            True,
        )
    ):
        raise ValidationError(
            "Phase 8D3D-R1 unexpectedly reports "
            "recomputed statistics."
        )

    return {
        "status":
            completion[
                "Phase8D3D_R1_status"
            ],
        "superseded_by":
            "Phase8D3D_R2",
    }


def validate_phase8D3D_R2() -> dict[str, Any]:
    require_directory(
        D3D_R2_DIR
    )

    require_directory(
        D3D_R2_FIGURES
    )

    require_file(
        D3D_R2_COMPLETION
    )

    require_file(
        D3D_R2_AUDIT
    )

    require_file(
        D3D_R2_SIGNOFF
    )

    completion = require_single_row(
        D3D_R2_COMPLETION
    )

    validate_status(
        completion,
        "Phase8D3D_R2_status",
        {
            "completed",
        },
        D3D_R2_COMPLETION,
    )

    required_true_fields = [
        "all_technical_audits_passed",
        "manuscript_ready_visual_signoff",
        "definitive_manuscript_figure",
        "supersedes_original_and_R1_visuals",
    ]

    for column in required_true_fields:
        if not as_bool(
            completion.get(
                column,
                False,
            )
        ):
            raise ValidationError(
                f"Phase 8D3D-R2 field is not True: "
                f"{column}"
            )

    if str(
        completion.get(
            "visual_review_status",
            "",
        )
    ).strip() != "passed":
        raise ValidationError(
            "Phase 8D3D-R2 visual review "
            "is not marked passed."
        )

    if str(
        completion.get(
            "definitive_figure_version",
            "",
        )
    ).strip() != "Phase8D3D_R2":
        raise ValidationError(
            "Unexpected definitive figure version."
        )

    if as_bool(
        completion.get(
            "statistics_recomputed",
            True,
        )
    ):
        raise ValidationError(
            "Phase 8D3D-R2 unexpectedly reports "
            "recomputed statistics."
        )

    audit_rows = validate_passed_audit(
        D3D_R2_AUDIT
    )

    signoff_rows = validate_passed_audit(
        D3D_R2_SIGNOFF
    )

    definitive_files = [
        D3D_R2_FIGURES
        / "phase8D3D_R2_integrated_cross_modal_figure.png",
        D3D_R2_FIGURES
        / "phase8D3D_R2_integrated_cross_modal_figure.pdf",
        D3D_R2_FIGURES
        / "phase8D3D_R2_integrated_cross_modal_figure.svg",
    ]

    for path in definitive_files:
        require_file(
            path
        )

        if path.stat().st_size < 1000:
            raise ValidationError(
                f"Definitive figure appears too small: {path}"
            )

    return {
        "audit_rows":
            audit_rows,
        "visual_signoff_rows":
            signoff_rows,
        "definitive_figure_version":
            "Phase8D3D_R2",
        "definitive_files":
            [
                str(
                    path.relative_to(
                        ROOT
                    )
                )
                for path in definitive_files
            ],
    }


def add_tree_files(
    collection: dict[str, Path],
    directory: Path,
) -> None:
    if not directory.is_dir():
        return

    for path in sorted(
        directory.rglob("*")
    ):
        if not path.is_file():
            continue

        relative = str(
            path.relative_to(
                ROOT
            )
        )

        collection[
            relative
        ] = path


def add_file(
    collection: dict[str, Path],
    path: Path,
    required: bool = True,
) -> None:
    if path.is_file():
        relative = str(
            path.relative_to(
                ROOT
            )
        )

        collection[
            relative
        ] = path

    elif required:
        raise ValidationError(
            f"Canonical source file not found: {path}"
        )


def collect_canonical_files() -> list[Path]:
    files: dict[str, Path] = {}

    # Phase 8D3A is represented by its manifest,
    # completion summary and external archive anchor.
    # The 2.3-GB archive itself is not duplicated.
    add_file(
        files,
        D3A_MANIFEST,
    )

    add_file(
        files,
        D3A_COMPLETION,
    )

    # Cross-modal synthesis and evidence hierarchy.
    add_tree_files(
        files,
        D3B_DIR,
    )

    add_tree_files(
        files,
        D3C_DIR,
    )

    # R1 metadata are retained because R1 is a
    # technically valid but visually superseded version.
    add_tree_files(
        files,
        D3D_R1_DIR,
    )

    # R2 is the definitive manuscript figure version.
    add_tree_files(
        files,
        D3D_R2_DIR,
    )

    add_tree_files(
        files,
        D3D_R2_FIGURES,
    )

    # Preserve earlier visual versions as provenance.
    add_tree_files(
        files,
        ORIGINAL_D3D_FIGURES,
    )

    add_tree_files(
        files,
        R1_FIGURES,
    )

    # Completion summaries, audits and visual sign-off.
    for path in [
        D3B_COMPLETION,
        D3C_AUDIT,
        D3C_COMPLETION,
        D3D_R1_COMPLETION,
        D3D_R2_AUDIT,
        D3D_R2_COMPLETION,
        D3D_R2_SIGNOFF,
    ]:
        add_file(
            files,
            path,
        )

    # Canonical scripts.
    for path in [
        D3C_SCRIPT,
        D3D_R1_SCRIPT,
        D3D_R2_SCRIPT,
        D3E_SCRIPT,
    ]:
        add_file(
            files,
            path,
        )

    # Execution logs are included when present.
    for path in [
        D3C_LOG,
        D3D_R1_LOG,
        D3D_R2_LOG,
    ]:
        add_file(
            files,
            path,
            required=False,
        )

    selected = [
        files[key]
        for key in sorted(
            files
        )
    ]

    if not selected:
        raise ValidationError(
            "No files were selected for the "
            "Phase 8D3E closure snapshot."
        )

    return selected


def classify_file(
    path: Path,
) -> tuple[str, str]:
    relative = str(
        path.relative_to(
            ROOT
        )
    )

    if "phase8D3A" in relative:
        return (
            "Phase8D3A_anchor_metadata",
            (
                "Phase 8D3A canonical lock metadata; "
                "the 2.3-GB archive is externally anchored "
                "and is not duplicated."
            ),
        )

    if "phase8D3B" in relative:
        return (
            "Phase8D3B_cross_modal_synthesis",
            "Validated cross-modal synthesis output.",
        )

    if "phase8D3C" in relative:
        return (
            "Phase8D3C_evidence_hierarchy",
            (
                "Definitive four-tier evidence hierarchy "
                "and reporting metadata."
            ),
        )

    if "phase8D3D_R2" in relative:
        return (
            "Phase8D3D_R2_definitive_visual",
            (
                "Definitive manuscript-ready integrated "
                "cross-modal figure or associated metadata."
            ),
        )

    if "phase8D3D_R1" in relative:
        return (
            "Phase8D3D_R1_superseded_visual",
            (
                "Technically valid but visually superseded "
                "R1 figure or metadata."
            ),
        )

    if "phase8D3D" in relative:
        return (
            "Phase8D3D_original_superseded_visual",
            (
                "Original technically valid figure retained "
                "for provenance and visually superseded by R2."
            ),
        )

    if relative.endswith(
        ".py"
    ):
        return (
            "canonical_script",
            "Executable Phase 8D3 workflow script.",
        )

    if relative.endswith(
        ".log"
    ):
        return (
            "execution_log",
            "Console execution record.",
        )

    return (
        "supporting_output",
        "Supporting Phase 8D3 canonical output.",
    )


def remove_existing_snapshot() -> None:
    if SNAPSHOT.exists():
        shutil.rmtree(
            SNAPSHOT
        )

    SNAPSHOT.mkdir(
        parents=True,
        exist_ok=True,
    )


def snapshot_destination(
    source: Path,
) -> Path:
    relative = source.relative_to(
        ROOT
    )

    return (
        SNAPSHOT
        / relative
    )


def copy_to_snapshot(
    source: Path,
) -> Path:
    destination = snapshot_destination(
        source
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        destination,
    )

    return destination


def build_snapshot(
    source_files: list[Path],
) -> list[dict[str, Any]]:
    remove_existing_snapshot()

    manifest_rows: list[
        dict[str, Any]
    ] = []

    for source in source_files:
        require_file(
            source
        )

        destination = copy_to_snapshot(
            source
        )

        source_hash = sha256(
            source
        )

        snapshot_hash = sha256(
            destination
        )

        if (
            source_hash
            != snapshot_hash
        ):
            raise ValidationError(
                "Snapshot hash mismatch: "
                f"{source}"
            )

        category, detail = classify_file(
            source
        )

        manifest_rows.append(
            {
                "canonical_order":
                    len(
                        manifest_rows
                    ) + 1,
                "category":
                    category,
                "source_relative_path":
                    str(
                        source.relative_to(
                            ROOT
                        )
                    ),
                "snapshot_relative_path":
                    str(
                        destination.relative_to(
                            OUT
                        )
                    ),
                "size_bytes":
                    source.stat().st_size,
                "source_SHA256":
                    source_hash,
                "snapshot_SHA256":
                    snapshot_hash,
                "hash_match":
                    True,
                "definitive_status":
                    (
                        "definitive"
                        if (
                            "Phase8D3D_R2"
                            in category
                            or category
                            in {
                                "Phase8D3C_evidence_hierarchy",
                                "Phase8D3B_cross_modal_synthesis",
                                "canonical_script",
                            }
                        )
                        else "retained_provenance"
                    ),
                "detail":
                    detail,
            }
        )

    if not manifest_rows:
        raise ValidationError(
            "The canonical snapshot contains no files."
        )

    return manifest_rows


def write_manifest(
    rows: list[dict[str, Any]],
) -> None:
    fields = [
        "canonical_order",
        "category",
        "source_relative_path",
        "snapshot_relative_path",
        "size_bytes",
        "source_SHA256",
        "snapshot_SHA256",
        "hash_match",
        "definitive_status",
        "detail",
    ]

    write_tsv(
        MANIFEST_OUT,
        fields,
        rows,
    )


def write_upstream_archive_anchors(
    phase8D3A: dict[str, Any],
) -> list[dict[str, Any]]:
    anchor_rows = [
        {
            "upstream_phase":
                "Phase8D3A",
            "archive_role":
                (
                    "external_canonical_input_archive_"
                    "not_duplicated_in_Phase8D3E"
                ),
            "relative_path":
                str(
                    D3A_ARCHIVE.relative_to(
                        ROOT
                    )
                ),
            "size_bytes":
                phase8D3A[
                    "archive_size_bytes"
                ],
            "SHA256":
                phase8D3A[
                    "archive_SHA256"
                ],
            "expected_SHA256":
                EXPECTED_D3A_ARCHIVE_SHA256,
            "hash_verified":
                True,
            "locked_input_files":
                phase8D3A[
                    "locked_files"
                ],
            "detail":
                (
                    "The Phase 8D3A archive remains the "
                    "canonical frozen-input anchor. Its "
                    "contents are referenced by hash rather "
                    "than duplicated into the Phase 8D3E archive."
                ),
        }
    ]

    write_tsv(
        UPSTREAM_ANCHORS_OUT,
        [
            "upstream_phase",
            "archive_role",
            "relative_path",
            "size_bytes",
            "SHA256",
            "expected_SHA256",
            "hash_verified",
            "locked_input_files",
            "detail",
        ],
        anchor_rows,
    )

    return anchor_rows


def add_generated_file_to_snapshot(
    path: Path,
) -> Path:
    require_file(
        path
    )

    destination = snapshot_destination(
        path
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        path,
        destination,
    )

    if sha256(
        path
    ) != sha256(
        destination
    ):
        raise ValidationError(
            "Generated-file snapshot hash mismatch: "
            f"{path}"
        )

    return destination


def normalized_tar_info(
    source: Path,
    arcname: str,
) -> tarfile.TarInfo:
    info = tarfile.TarInfo(
        name=arcname
    )

    stat = source.stat()

    info.size = stat.st_size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    info.type = tarfile.REGTYPE

    return info


def create_deterministic_archive(
    archive_path: Path,
) -> tuple[int, str]:
    if archive_path.exists():
        archive_path.unlink()

    snapshot_files = sorted(
        path
        for path in SNAPSHOT.rglob("*")
        if path.is_file()
    )

    if not snapshot_files:
        raise ValidationError(
            "No snapshot files are available "
            "for deterministic archiving."
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        dir=OUT
    ) as temporary_directory:
        temporary_tar = (
            Path(
                temporary_directory
            )
            / "phase8D3E_canonical_closure.tar"
        )

        with tarfile.open(
            temporary_tar,
            mode="w",
            format=tarfile.PAX_FORMAT,
        ) as archive:
            for path in snapshot_files:
                arcname = str(
                    Path(
                        "phase8D3E_canonical_closure"
                    )
                    / path.relative_to(
                        SNAPSHOT
                    )
                )

                info = normalized_tar_info(
                    path,
                    arcname,
                )

                with path.open(
                    "rb"
                ) as handle:
                    archive.addfile(
                        info,
                        handle,
                    )

        with temporary_tar.open(
            "rb"
        ) as source_handle:
            with archive_path.open(
                "wb"
            ) as destination_handle:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=destination_handle,
                    mtime=0,
                ) as gzip_handle:
                    shutil.copyfileobj(
                        source_handle,
                        gzip_handle,
                        length=1024 * 1024,
                    )

    archive_hash = sha256(
        archive_path
    )

    ARCHIVE_SHA_OUT.write_text(
        (
            f"{archive_hash}  "
            f"{archive_path.name}\n"
        ),
        encoding="utf-8",
    )

    return (
        len(
            snapshot_files
        ),
        archive_hash,
    )


def verify_archive_members(
    archive_path: Path,
) -> int:
    require_file(
        archive_path
    )

    with tarfile.open(
        archive_path,
        mode="r:gz",
    ) as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile()
        ]

        member_names = [
            member.name
            for member in members
        ]

        if len(
            member_names
        ) != len(
            set(
                member_names
            )
        ):
            raise ValidationError(
                "Duplicate member names were found "
                "inside the Phase 8D3E archive."
            )

        for member in members:
            if member.mtime != 0:
                raise ValidationError(
                    "Archive member has a nonzero "
                    f"timestamp: {member.name}"
                )

            if member.uid != 0 or member.gid != 0:
                raise ValidationError(
                    "Archive member has noncanonical "
                    f"ownership: {member.name}"
                )

    return len(
        members
    )


def snapshot_hash_matches() -> bool:
    _, rows = read_tsv(
        MANIFEST_OUT
    )

    for row in rows:
        snapshot_path = (
            OUT
            / row[
                "snapshot_relative_path"
            ]
        )

        if not snapshot_path.is_file():
            return False

        if sha256(
            snapshot_path
        ) != row[
            "snapshot_SHA256"
        ]:
            return False

    return True


def count_snapshot_files() -> int:
    return sum(
        1
        for path in SNAPSHOT.rglob("*")
        if path.is_file()
    )


def write_closure_readme(
    source_file_count: int,
) -> Path:
    path = (
        OUT
        / "phase8D3E_closure_readme.txt"
    )

    text = f"""PHASE 8D3E CANONICAL CLOSURE

PROJECT
DTHI_Struct_Cortical_Hierarchy

CLOSED SCOPE
Phase 8D3A canonical input lock
Phase 8D3B cross-modal synthesis
Phase 8D3C definitive evidence hierarchy
Phase 8D3D-R2 definitive integrated cross-modal figure

DEFINITIVE VISUAL VERSION
Phase8D3D_R2

DEFINITIVE PRIMARY RESULT
synaptic_assembly_receptor_trafficking–clinical_asd

EVIDENCE HIERARCHY
Tier 1 global-FWER associations: 1
Tier 2 family-corrected associations: 4
Tier 3A nominal associations: 13
Tier 3B descriptive associations: 142

CANONICAL SOURCE FILES
{source_file_count}

UPSTREAM LARGE-ARCHIVE POLICY
The 2.3-GB Phase 8D3A canonical input archive is not duplicated inside
the Phase 8D3E archive. It is linked through its verified SHA256 digest:

{EXPECTED_D3A_ARCHIVE_SHA256}

INTERPRETATION LIMITS
Spatial correspondence does not establish causality, disease-specific
molecular expression, mechanistic equivalence, or independent replication.
The LH46 analysis is a nested sensitivity analysis.

PROJECT-WIDE REMAINING BLOCKER
The latest Phase 5D4 functional-enrichment attempt failed. Its outputs are
invalid and must not be used. Phase 5D4 must be rerun and verified before
final manuscript integration or final project-wide closure.

STATISTICS RECOMPUTED
False
"""

    path.write_text(
        text,
        encoding="utf-8",
    )

    return path


def verify_archive_sha_record(
    expected_hash: str,
) -> bool:
    require_file(
        ARCHIVE_SHA_OUT
    )

    content = (
        ARCHIVE_SHA_OUT
        .read_text(
            encoding="utf-8"
        )
        .strip()
    )

    if not content:
        return False

    recorded_hash = content.split()[0]

    return bool(
        recorded_hash
        == expected_hash
        == sha256(
            ARCHIVE_OUT
        )
    )


def build_audit(
    phase8D3A: dict[str, Any],
    phase8D3B: dict[str, Any],
    phase8D3C: dict[str, Any],
    phase8D3D_R1: dict[str, Any],
    phase8D3D_R2: dict[str, Any],
    source_file_count: int,
    manifest_row_count: int,
    snapshot_file_count: int,
    archive_member_count: int,
    archive_hash: str,
) -> list[dict[str, Any]]:
    definitive_files_exist = all(
        (
            ROOT
            / relative
        ).is_file()
        for relative
        in phase8D3D_R2[
            "definitive_files"
        ]
    )

    archive_metadata_canonical = bool(
        archive_member_count
        == snapshot_file_count
    )

    return [
        {
            "section":
                "Phase8D3A",
            "item":
                "locked_input_files",
            "value":
                phase8D3A[
                    "locked_files"
                ],
            "passed":
                (
                    phase8D3A[
                        "locked_files"
                    ]
                    == 75
                ),
            "detail":
                (
                    "All Phase 8D3A source-to-"
                    "snapshot hashes matched."
                ),
        },
        {
            "section":
                "Phase8D3A",
            "item":
                "upstream_archive_SHA256",
            "value":
                phase8D3A[
                    "archive_SHA256"
                ],
            "passed":
                (
                    phase8D3A[
                        "archive_SHA256"
                    ]
                    == EXPECTED_D3A_ARCHIVE_SHA256
                ),
            "detail":
                (
                    "The 2.3-GB canonical input "
                    "archive is externally anchored "
                    "and not duplicated."
                ),
        },
        {
            "section":
                "Phase8D3B",
            "item":
                "cross_modal_synthesis",
            "value":
                "completed",
            "passed":
                (
                    phase8D3B[
                        "DTHI_maps"
                    ]
                    == 10
                ),
            "detail":
                (
                    f"{phase8D3B['DTHI_maps']} "
                    "DTHI maps synthesized."
                ),
        },
        {
            "section":
                "Phase8D3C",
            "item":
                "associations_classified",
            "value":
                phase8D3C[
                    "associations_classified"
                ],
            "passed":
                (
                    phase8D3C[
                        "associations_classified"
                    ]
                    == 160
                ),
            "detail":
                (
                    "Mutually exclusive definitive "
                    "evidence hierarchy."
                ),
        },
        {
            "section":
                "Phase8D3C",
            "item":
                "evidence_tier_structure",
            "value":
                json.dumps(
                    {
                        "Tier1":
                            phase8D3C[
                                "Tier1_global_FWER_associations"
                            ],
                        "Tier2":
                            phase8D3C[
                                "Tier2_family_corrected_associations"
                            ],
                        "Tier3A":
                            phase8D3C[
                                "Tier3A_nominal_associations"
                            ],
                        "Tier3B":
                            phase8D3C[
                                "Tier3B_descriptive_associations"
                            ],
                    },
                    sort_keys=True,
                ),
            "passed":
                bool(
                    phase8D3C[
                        "Tier1_global_FWER_associations"
                    ] == 1
                    and phase8D3C[
                        "Tier2_family_corrected_associations"
                    ] == 4
                    and phase8D3C[
                        "Tier3A_nominal_associations"
                    ] == 13
                    and phase8D3C[
                        "Tier3B_descriptive_associations"
                    ] == 142
                ),
            "detail":
                (
                    "One global-FWER result, four "
                    "family-corrected results."
                ),
        },
        {
            "section":
                "Phase8D3D_R1",
            "item":
                "superseded_visual_preserved",
            "value":
                phase8D3D_R1[
                    "status"
                ],
            "passed":
                (
                    phase8D3D_R1[
                        "superseded_by"
                    ]
                    == "Phase8D3D_R2"
                ),
            "detail":
                (
                    "R1 is technically valid but "
                    "visually superseded by R2."
                ),
        },
        {
            "section":
                "Phase8D3D_R2",
            "item":
                "definitive_visual_version",
            "value":
                phase8D3D_R2[
                    "definitive_figure_version"
                ],
            "passed":
                (
                    phase8D3D_R2[
                        "definitive_figure_version"
                    ]
                    == "Phase8D3D_R2"
                ),
            "detail":
                (
                    "R2 passed direct manuscript-"
                    "readiness visual review."
                ),
        },
        {
            "section":
                "Phase8D3D_R2",
            "item":
                "definitive_figure_files",
            "value":
                len(
                    phase8D3D_R2[
                        "definitive_files"
                    ]
                ),
            "passed":
                definitive_files_exist,
            "detail":
                (
                    "PNG, PDF and SVG definitive "
                    "outputs are present."
                ),
        },
        {
            "section":
                "snapshot",
            "item":
                "canonical_source_files",
            "value":
                source_file_count,
            "passed":
                (
                    source_file_count
                    > 0
                ),
            "detail":
                (
                    "Files selected from completed "
                    "Phase 8D3 components."
                ),
        },
        {
            "section":
                "snapshot",
            "item":
                "manifest_rows",
            "value":
                manifest_row_count,
            "passed":
                (
                    manifest_row_count
                    == source_file_count
                ),
            "detail":
                (
                    "One canonical manifest row "
                    "per source file."
                ),
        },
        {
            "section":
                "snapshot",
            "item":
                "all_manifest_hashes_match",
            "value":
                snapshot_hash_matches(),
            "passed":
                snapshot_hash_matches(),
            "detail":
                (
                    "Every source file matches its "
                    "snapshot SHA256."
                ),
        },
        {
            "section":
                "snapshot",
            "item":
                "snapshot_files",
            "value":
                snapshot_file_count,
            "passed":
                (
                    snapshot_file_count
                    >= source_file_count + 3
                ),
            "detail":
                (
                    "Includes canonical sources, "
                    "manifest, upstream anchors "
                    "and closure README."
                ),
        },
        {
            "section":
                "archive",
            "item":
                "archive_created",
            "value":
                ARCHIVE_OUT.is_file(),
            "passed":
                ARCHIVE_OUT.is_file(),
            "detail":
                str(
                    ARCHIVE_OUT.relative_to(
                        ROOT
                    )
                ),
        },
        {
            "section":
                "archive",
            "item":
                "archive_members",
            "value":
                archive_member_count,
            "passed":
                archive_metadata_canonical,
            "detail":
                (
                    "Archive member count matches "
                    "the canonical snapshot."
                ),
        },
        {
            "section":
                "archive",
            "item":
                "archive_SHA256_record",
            "value":
                archive_hash,
            "passed":
                verify_archive_sha_record(
                    archive_hash
                ),
            "detail":
                (
                    "Archive and SHA256 sidecar "
                    "contain the same digest."
                ),
        },
        {
            "section":
                "interpretation",
            "item":
                "statistics_recomputed",
            "value":
                False,
            "passed":
                True,
            "detail":
                (
                    "Phase 8D3E is closure and "
                    "archiving only."
                ),
        },
        {
            "section":
                "project_dependency",
            "item":
                "Phase5D4_functional_enrichment",
            "value":
                "rerun_required",
            "passed":
                True,
            "detail":
                (
                    "Latest Phase 5D4 attempt failed; "
                    "its outputs remain invalid and "
                    "are excluded from this closure."
                ),
        },
    ]


def main() -> None:
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES.mkdir(
        parents=True,
        exist_ok=True,
    )

    phase8D3A = validate_phase8D3A()
    phase8D3B = validate_phase8D3B()
    phase8D3C = validate_phase8D3C()
    phase8D3D_R1 = validate_phase8D3D_R1()
    phase8D3D_R2 = validate_phase8D3D_R2()

    source_files = collect_canonical_files()

    manifest_rows = build_snapshot(
        source_files
    )

    write_manifest(
        manifest_rows
    )

    write_upstream_archive_anchors(
        phase8D3A
    )

    closure_readme = write_closure_readme(
        len(
            source_files
        )
    )

    for generated_path in [
        MANIFEST_OUT,
        UPSTREAM_ANCHORS_OUT,
        closure_readme,
    ]:
        add_generated_file_to_snapshot(
            generated_path
        )

    snapshot_file_count = count_snapshot_files()

    (
        archive_input_count,
        archive_hash,
    ) = create_deterministic_archive(
        ARCHIVE_OUT
    )

    archive_member_count = verify_archive_members(
        ARCHIVE_OUT
    )

    if (
        archive_input_count
        != snapshot_file_count
    ):
        raise ValidationError(
            "Archive input count does not match "
            "the snapshot file count."
        )

    audit_rows = build_audit(
        phase8D3A,
        phase8D3B,
        phase8D3C,
        phase8D3D_R1,
        phase8D3D_R2,
        len(
            source_files
        ),
        len(
            manifest_rows
        ),
        snapshot_file_count,
        archive_member_count,
        archive_hash,
    )

    write_tsv(
        AUDIT_OUT,
        [
            "section",
            "item",
            "value",
            "passed",
            "detail",
        ],
        audit_rows,
    )

    all_audits_passed = all(
        as_bool(
            row[
                "passed"
            ]
        )
        for row in audit_rows
    )

    completion_rows = [
        {
            "Phase8D3A_status_confirmed":
                True,
            "Phase8D3B_status_confirmed":
                True,
            "Phase8D3C_status_confirmed":
                True,
            "Phase8D3D_R2_status_confirmed":
                True,
            "definitive_figure_version":
                "Phase8D3D_R2",
            "canonical_source_files":
                len(
                    source_files
                ),
            "canonical_snapshot_files":
                snapshot_file_count,
            "archive_members":
                archive_member_count,
            "Phase8D3A_external_anchor_SHA256":
                phase8D3A[
                    "archive_SHA256"
                ],
            "Phase8D3E_archive_SHA256":
                archive_hash,
            "statistics_recomputed":
                False,
            "all_closure_audits_passed":
                all_audits_passed,
            "Phase8D3_status":
                (
                    "completed"
                    if all_audits_passed
                    else "failed"
                ),
            "Phase8D3E_status":
                (
                    "completed"
                    if all_audits_passed
                    else "failed"
                ),
            "Phase5D4_outputs_valid":
                False,
            "ready_for_Phase5D4_rerun":
                all_audits_passed,
            "ready_for_phase8E_manuscript_integration":
                False,
            "manuscript_integration_blocker":
                (
                    "Phase5D4 functional-enrichment "
                    "rerun and verification"
                ),
            "python_version":
                sys.version.split()[0],
        }
    ]

    write_tsv(
        COMPLETION_OUT,
        [
            "Phase8D3A_status_confirmed",
            "Phase8D3B_status_confirmed",
            "Phase8D3C_status_confirmed",
            "Phase8D3D_R2_status_confirmed",
            "definitive_figure_version",
            "canonical_source_files",
            "canonical_snapshot_files",
            "archive_members",
            "Phase8D3A_external_anchor_SHA256",
            "Phase8D3E_archive_SHA256",
            "statistics_recomputed",
            "all_closure_audits_passed",
            "Phase8D3_status",
            "Phase8D3E_status",
            "Phase5D4_outputs_valid",
            "ready_for_Phase5D4_rerun",
            "ready_for_phase8E_manuscript_integration",
            "manuscript_integration_blocker",
            "python_version",
        ],
        completion_rows,
    )

    print(
        "===== PHASE 8D3E COMPLETION ====="
    )

    for key, value in completion_rows[0].items():
        print(
            f"{key}: {value}"
        )

    print(
        "\n===== PHASE 8D3E AUDIT ====="
    )

    for row in audit_rows:
        print(
            f"[{row['section']}] "
            f"{row['item']}: "
            f"value={row['value']} | "
            f"passed={row['passed']}"
        )

    print(
        "\n===== CLOSURE OUTPUTS ====="
    )

    closure_outputs = [
        MANIFEST_OUT,
        UPSTREAM_ANCHORS_OUT,
        closure_readme,
        ARCHIVE_OUT,
        ARCHIVE_SHA_OUT,
        AUDIT_OUT,
        COMPLETION_OUT,
    ]

    for path in closure_outputs:
        print(
            f"{path.relative_to(ROOT)}\t"
            f"{path.stat().st_size}\t"
            f"{sha256(path)}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            "Phase 8D3E failed: "
            f"{type(error).__name__}: "
            f"{error}",
            file=sys.stderr,
        )
        raise
