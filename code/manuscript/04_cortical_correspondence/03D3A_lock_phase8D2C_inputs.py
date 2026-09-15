#!/usr/bin/env python3
"""Phase 8D3A: canonical lock, provenance manifest, and reproducible archive."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_2C = ROOT / "03_processed_data" / "functional_imaging" / "phase8D" / "phase8D2C"
TABLES = ROOT / "07_tables" / "main_tables" / "phase8"
FIGURES_2C4B = ROOT / "08_figures" / "phase8" / "phase8D2C4B"
LOGS = ROOT / "09_pipeline_logs" / "phase8"
SCRIPTS = ROOT / "04_scripts" / "10_functional_imaging" / "phase8D"
LOCK_ROOT = ROOT / "03_processed_data" / "functional_imaging" / "phase8D" / "phase8D3A_canonical_lock"
SNAPSHOT_ROOT = LOCK_ROOT / "snapshot"
MANIFEST = LOCK_ROOT / "phase8D3A_locked_inputs_manifest.tsv"
PROVENANCE = LOCK_ROOT / "phase8D3A_provenance.json"
SUPERSESSION = LOCK_ROOT / "phase8D3A_supersession_record.tsv"
README = LOCK_ROOT / "README_PHASE8D3A_LOCK.txt"
ARCHIVE = LOCK_ROOT / "phase8D3A_canonical_lock.tar.gz"
ARCHIVE_RECORD = LOCK_ROOT / "phase8D3A_archive_record.tsv"
AUDIT = TABLES / "phase8D3A_lock_audit.tsv"
COMPLETION = TABLES / "phase8D3A_completion_summary.tsv"
LOCK_MARKER = LOCK_ROOT / "PHASE8D3A_LOCK_COMPLETE.txt"
PHASE_COMPONENTS = ["phase8D2C1", "phase8D2C2A", "phase8D2C2B", "phase8D2C2C", "phase8D2C3", "phase8D2C4B"]
REQUIRED_EXACT = [
    PROCESSED_2C / "phase8D2C2C_bijective_primary_spatial_correspondence.tsv",
    PROCESSED_2C / "phase8D2C2C_bijective_high_support_sensitivity.tsv",
    PROCESSED_2C / "phase8D2C2C_bijective_vs_nonbijective_comparison.tsv",
    PROCESSED_2C / "phase8D2C2C_primary_supported_bijective_summary.tsv",
    PROCESSED_2C / "phase8D2C3_integrated_evidence.tsv",
    PROCESSED_2C / "phase8D2C3_priority_associations.tsv",
    PROCESSED_2C / "phase8D2C3_DTHI_external_spearman_matrix.tsv",
    PROCESSED_2C / "phase8D2C3_evidence_class_matrix.tsv",
    FIGURES_2C4B / "phase8D2C4B_all_revised_figures.pdf",
    FIGURES_2C4B / "phase8D2C4B_figure_legends.txt",
    TABLES / "phase8D2C2C_completion_summary.tsv",
    TABLES / "phase8D2C3_completion_summary.tsv",
    TABLES / "phase8D2C4B_completion_summary.tsv",
]

class ValidationError(RuntimeError):
    pass

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def atomic_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)

def require_file(path: Path) -> None:
    if not path.is_file():
        raise ValidationError(f"Required canonical file not found: {path}")

def validate_completion(component: str) -> Path:
    matches = sorted(TABLES.glob(f"{component}*completion*summary*.tsv"))
    if not matches:
        matches = sorted(TABLES.glob(f"{component}*completion*.tsv"))
    if len(matches) != 1:
        raise ValidationError(f"Expected one completion summary for {component}; found {len(matches)}: {[str(item) for item in matches]}")
    frame = pd.read_csv(matches[0], sep="\t")
    if len(frame) != 1:
        raise ValidationError(f"Completion summary must have one row: {matches[0]}")
    status_columns = [column for column in frame.columns if str(column).lower().endswith("_status")]
    if not status_columns:
        raise ValidationError(f"No status column found in completion summary: {matches[0]}")
    values = {str(frame.loc[0, column]).strip().lower() for column in status_columns}
    if "completed" not in values:
        raise ValidationError(f"{component} is not marked completed in {matches[0]}; status values={sorted(values)}")
    return matches[0]

def is_superseded_original_2c4(path: Path) -> bool:
    name = path.name
    return ("phase8D2C4" in name and "phase8D2C4B" not in name) or (name.startswith("03C4_") and not name.startswith("03C4B"))

def collect_sources() -> list[tuple[Path, str, str]]:
    records: list[tuple[Path, str, str]] = []
    for path in sorted(PROCESSED_2C.rglob("*")):
        if path.is_file():
            records.append((path, "phase8D2C_processed", "canonical_analysis_output"))
    for path in sorted(TABLES.glob("phase8D2C*.tsv")):
        if path.is_file() and not is_superseded_original_2c4(path):
            records.append((path, "phase8D2C_tables", "canonical_table_or_audit"))
    for path in sorted(FIGURES_2C4B.rglob("*")):
        if path.is_file():
            records.append((path, "phase8D2C4B_figures", "final_approved_figure"))
    for path in sorted(LOGS.glob("phase8D2C*")):
        if path.is_file() and not is_superseded_original_2c4(path):
            records.append((path, "phase8D2C_logs", "execution_log"))
    for path in sorted(SCRIPTS.glob("03C*.py")):
        if path.is_file() and path.name != Path(__file__).name and not is_superseded_original_2c4(path):
            records.append((path, "phase8D2C_scripts", "analysis_source"))
    unique: dict[str, tuple[Path, str, str]] = {}
    for path, group, role in records:
        unique[str(path.relative_to(ROOT))] = (path, group, role)
    return [unique[key] for key in sorted(unique)]

def normalized_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    info.mode = 0o555 if info.isdir() else 0o444
    return info

def build_reproducible_archive(archive_path: Path, members: list[Path]) -> int:
    with archive_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path in sorted(members, key=lambda item: str(item.relative_to(LOCK_ROOT))):
                    tar.add(path, arcname=str(path.relative_to(LOCK_ROOT)), recursive=False, filter=normalized_tar_info)
    return len(members)

def main() -> None:
    for path in REQUIRED_EXACT:
        require_file(path)
    completion_files = {component: validate_completion(component) for component in PHASE_COMPONENTS}
    source_records = collect_sources()
    if not source_records:
        raise ValidationError("No canonical Phase 8D2C files were collected.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if LOCK_ROOT.exists():
        LOCK_ROOT.rename(LOCK_ROOT.with_name(f"{LOCK_ROOT.name}.backup_{timestamp}"))
    LOCK_ROOT.mkdir(parents=True, exist_ok=False)
    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=False)
    manifest_rows = []
    for source, group, role in source_records:
        relative = source.relative_to(ROOT)
        snapshot = SNAPSHOT_ROOT / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, snapshot)
        source_hash = sha256(source)
        snapshot_hash = sha256(snapshot)
        if source_hash != snapshot_hash:
            raise ValidationError(f"Snapshot hash mismatch for {relative}")
        os.chmod(snapshot, 0o444)
        manifest_rows.append({
            "phase_group": group,
            "analysis_role": role,
            "source_relative_path": str(relative),
            "snapshot_relative_path": str(snapshot.relative_to(LOCK_ROOT)),
            "size_bytes": source.stat().st_size,
            "source_SHA256": source_hash,
            "snapshot_SHA256": snapshot_hash,
            "hash_match": True,
            "locked_read_only": True,
        })
    manifest = pd.DataFrame(manifest_rows)
    atomic_tsv(manifest, MANIFEST)
    supersession = pd.DataFrame([{
        "superseded_stage": "Phase8D2C4",
        "superseding_stage": "Phase8D2C4B",
        "canonical_figure_directory": str(FIGURES_2C4B.relative_to(ROOT)),
        "canonical_combined_pdf": str((FIGURES_2C4B / "phase8D2C4B_all_revised_figures.pdf").relative_to(ROOT)),
        "reason": "figure_only_revision_and_final_visual_signoff",
        "original_phase8D2C4_excluded_from_lock": True,
    }])
    atomic_tsv(supersession, SUPERSESSION)
    provenance_payload = {
        "phase": "Phase8D3A",
        "status": "canonical_lock",
        "created_utc": utc_now(),
        "project_root": str(ROOT),
        "locked_upstream_phase": "Phase8D2C",
        "completed_components": PHASE_COMPONENTS,
        "completion_summary_files": {component: str(path.relative_to(ROOT)) for component, path in completion_files.items()},
        "canonical_primary_result": {
            "DTHI_map": "synaptic_assembly_receptor_trafficking",
            "external_map": "clinical_asd",
            "spearman_rho": -0.654250,
            "exact_bijective_spin_p_two_sided": 0.0015,
            "exact_bijective_global_maxT_p": 0.018698,
            "exact_bijective_clinical_family_maxT_p": 0.013899,
            "global_BH_q": 0.059994,
            "LODO_direction_concordance": "5_of_5",
            "LH46_support": "nominal_only",
        },
        "secondary_reporting_class": "four_clinical_family_BH_FDR_supported_associations",
        "interpretation_constraints": [
            "spatial_correspondence_not_causality",
            "LH46_is_nested_sensitivity_not_independent_replication",
            "exact_bijective_inference_is_definitive_spatial_null_sensitivity",
            "Phase8D2C4B_supersedes_Phase8D2C4",
        ],
        "software": {"python": sys.version.split()[0], "pandas": pd.__version__, "platform": platform.platform()},
        "locked_file_count": int(len(manifest)),
        "manifest_SHA256": sha256(MANIFEST),
    }
    PROVENANCE.write_text(json.dumps(provenance_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    README.write_text(
        "PHASE 8D3A CANONICAL LOCK\n\n"
        "This directory is the immutable snapshot of the completed Phase 8D2C functional-imaging analysis.\n\n"
        "Canonical interpretation:\n"
        "1. The sole exact-bijective global maxT association is synaptic assembly/receptor trafficking–ASD.\n"
        "2. Four additional clinical associations have secondary clinical-family BH-FDR support.\n"
        "3. No association survives nested LH46 BH-FDR.\n"
        "4. Spatial correspondence does not establish causality.\n"
        "5. Phase 8D2C4B supersedes the original Phase 8D2C4 figures.\n\n"
        "Downstream Phase 8D3 analyses must read from snapshot/ and must not silently replace or selectively modify these locked inputs.\n",
        encoding="utf-8",
    )
    archive_members = [path for path in LOCK_ROOT.rglob("*") if path.is_file() and path != ARCHIVE]
    archive_member_count = build_reproducible_archive(ARCHIVE, archive_members)
    archive_record = pd.DataFrame([{
        "relative_path": str(ARCHIVE.relative_to(ROOT)),
        "size_bytes": ARCHIVE.stat().st_size,
        "SHA256": sha256(ARCHIVE),
        "archive_member_count": archive_member_count,
        "deterministic_metadata": True,
    }])
    atomic_tsv(archive_record, ARCHIVE_RECORD)
    all_snapshot_hashes_match = bool(manifest["hash_match"].all())
    all_snapshot_files_exist = all((LOCK_ROOT / item).is_file() for item in manifest["snapshot_relative_path"])
    audit = pd.DataFrame([
        {"section": "upstream", "item": "Phase8D2C_components_completed", "value": len(completion_files), "passed": len(completion_files) == 6, "detail": ";".join(PHASE_COMPONENTS)},
        {"section": "inputs", "item": "locked_file_count", "value": len(manifest), "passed": len(manifest) > 0, "detail": "deduplicated_by_project_relative_path"},
        {"section": "integrity", "item": "snapshot_hashes_match_sources", "value": all_snapshot_hashes_match, "passed": all_snapshot_hashes_match, "detail": "SHA256"},
        {"section": "integrity", "item": "all_snapshot_files_exist", "value": all_snapshot_files_exist, "passed": all_snapshot_files_exist, "detail": ""},
        {"section": "provenance", "item": "Phase8D2C4B_supersession_recorded", "value": True, "passed": True, "detail": "original_Phase8D2C4_excluded"},
        {"section": "archive", "item": "canonical_archive_created", "value": ARCHIVE.is_file(), "passed": ARCHIVE.is_file(), "detail": sha256(ARCHIVE)},
        {"section": "archive", "item": "archive_member_count", "value": archive_member_count, "passed": archive_member_count > 0, "detail": "deterministic_tar_gzip_metadata"},
    ])
    atomic_tsv(audit, AUDIT)
    all_passed = bool(audit["passed"].all())
    completion = pd.DataFrame([{
        "Phase8D2C_status_confirmed": True,
        "phase_components_locked": len(completion_files),
        "locked_files": len(manifest),
        "snapshot_hashes_match_sources": all_snapshot_hashes_match,
        "all_snapshot_files_present": all_snapshot_files_exist,
        "supersession_recorded": True,
        "canonical_archive_created": ARCHIVE.is_file(),
        "archive_member_count": archive_member_count,
        "all_lock_audits_passed": all_passed,
        "ready_for_phase8D3B_cross_modal_synthesis": all_passed,
        "Phase8D3A_status": "completed" if all_passed else "failed",
        "python_version": sys.version.split()[0],
        "pandas_version": pd.__version__,
    }])
    atomic_tsv(completion, COMPLETION)
    LOCK_MARKER.write_text(
        "PHASE 8D3A CANONICAL LOCK COMPLETE\n"
        f"created_utc={utc_now()}\n"
        f"locked_files={len(manifest)}\n"
        f"manifest_SHA256={sha256(MANIFEST)}\n"
        f"archive_SHA256={sha256(ARCHIVE)}\n"
        f"ready_for_phase8D3B_cross_modal_synthesis={str(all_passed)}\n",
        encoding="utf-8",
    )
    print("===== PHASE 8D3A COMPLETION =====")
    print(completion.to_string(index=False))
    print("\n===== LOCK AUDIT =====")
    print(audit.to_string(index=False))
    print("\n===== CANONICAL ARCHIVE =====")
    print(archive_record.to_string(index=False))
    print("\n===== LOCKED FILE GROUPS =====")
    grouped = manifest.groupby(["phase_group", "analysis_role"]).size().reset_index(name="file_count")
    print(grouped.to_string(index=False))
    print("\n===== KEY HASHES =====")
    key_hashes = pd.DataFrame([
        {"relative_path": str(MANIFEST.relative_to(ROOT)), "size_bytes": MANIFEST.stat().st_size, "SHA256": sha256(MANIFEST)},
        {"relative_path": str(PROVENANCE.relative_to(ROOT)), "size_bytes": PROVENANCE.stat().st_size, "SHA256": sha256(PROVENANCE)},
        {"relative_path": str(ARCHIVE.relative_to(ROOT)), "size_bytes": ARCHIVE.stat().st_size, "SHA256": sha256(ARCHIVE)},
        {"relative_path": str(AUDIT.relative_to(ROOT)), "size_bytes": AUDIT.stat().st_size, "SHA256": sha256(AUDIT)},
        {"relative_path": str(COMPLETION.relative_to(ROOT)), "size_bytes": COMPLETION.stat().st_size, "SHA256": sha256(COMPLETION)},
    ])
    print(key_hashes.to_string(index=False))

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Phase 8D3A failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise
