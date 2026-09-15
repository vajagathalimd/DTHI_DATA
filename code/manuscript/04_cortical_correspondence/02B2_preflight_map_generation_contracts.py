#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import nilearn
import nimare
import numpy as np
import pandas as pd
from scipy import sparse

from enigmatoolbox.datasets import load_summary_stats
from enigmatoolbox.utils.parcellation import parcel_to_surface, surface_to_parcel

PROJECT = Path('.')
PHASE_TABLES = PROJECT / '07_tables/main_tables/phase8'
PROC = PROJECT / '03_processed_data/functional_imaging/phase8D'
ENIGMA_ROOT = PROJECT / '05_external_resources/phase8D/ENIGMA'
NEUROSYNTH_ROOT = PROJECT / '05_external_resources/phase8D/neurosynth-data'
OUT = PHASE_TABLES / 'phase8D2B2_preflight_audit.tsv'
SUMMARY = PHASE_TABLES / 'phase8D2B2_preflight_completion_summary.tsv'
PHASE8D2B2_REPAIR_VERSION = '2026-07-23-final-v3'


def add(rows: list[dict[str, Any]], section: str, item: str, value: Any, passed: bool | None = None, detail: str = '') -> None:
    rows.append({
        'section': section,
        'item': item,
        'value': str(value),
        'passed': '' if passed is None else bool(passed),
        'detail': detail,
    })


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def normalize_region(value: Any) -> str:
    """Normalize ENIGMA/FreeSurfer DK labels to hemisphere+region."""

    if pd.isna(value):
        return ""

    label = str(value).strip().lower()

    # Handle full paths and filename-based schizophrenia labels.
    label = label.replace("\\", "/").rsplit("/", 1)[-1]
    label = re.sub(r"\.csv$", "", label)
    label = re.sub(
        r"(?:[_-]asy)?[_-](?:thick|thickness)$",
        "",
        label,
    )

    # Remove optional FreeSurfer cortical prefix.
    label = re.sub(r"^ctx[_-]*", "", label)

    # Standardize hemisphere prefixes.
    label = re.sub(
        r"^(?:left|lh|l)[_\-\s]+",
        "l_",
        label,
    )
    label = re.sub(
        r"^(?:right|rh|r)[_\-\s]+",
        "r_",
        label,
    )

    return re.sub(r"[^a-z0-9]+", "", label)


def finite_mean(
    values: Any,
    weights: Any = None,
) -> float:
    """Calculate a mean using only finite DK-mapped vertices."""

    array = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    valid = np.isfinite(array)

    if not valid.any():
        return float("nan")

    if weights is None:
        return float(array[valid].mean())

    weight_array = np.asarray(
        weights,
        dtype=float,
    ).reshape(-1)

    weighted_valid = (
        valid
        & np.isfinite(weight_array)
        & (weight_array > 0)
    )

    if not weighted_valid.any():
        return float(array[valid].mean())

    return float(
        np.average(
            array[weighted_valid],
            weights=weight_array[weighted_valid],
        )
    )


def read_delimited(path: Path) -> pd.DataFrame:
    with path.open('r', encoding='utf-8-sig', errors='replace') as handle:
        header = handle.readline()
    if header.count(';') > header.count(',') and header.count(';') > header.count('\t'):
        return pd.read_csv(path, sep=';')
    if header.count('\t') > header.count(','):
        return pd.read_csv(path, sep='\t')
    return pd.read_csv(path)


def parse_bool(value: Any) -> bool:
    """Convert stored TSV boolean values without treating 'False' as true."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def find_schema_table(
    required: set[str],
    prefix: str = "phase8D2B1",
) -> Path:
    """Find the strongest matching frozen input table by schema."""

    search_roots = [
        PHASE_TABLES,
        PROC,
        PROJECT / "07_tables",
        PROJECT / "03_processed_data",
    ]

    candidate_records: list[tuple[tuple[int, int, float], Path]] = []
    seen: set[Path] = set()

    for root in search_roots:
        if not root.exists():
            continue

        for path in root.rglob("*.tsv"):
            path = path.resolve()

            if path in seen:
                continue

            seen.add(path)

            if path in {OUT.resolve(), SUMMARY.resolve()}:
                continue

            try:
                frame = pd.read_csv(path, sep="\t")
            except Exception:
                continue

            if not required.issubset(set(frame.columns)):
                continue

            validity_score = 0

            if "selection_frozen" in required:
                frozen = frame["selection_frozen"].map(parse_bool).all()

                if len(frame) == 8 and frozen:
                    validity_score += 100
                elif frozen:
                    validity_score += 20

            if {
                "ready_for_phase8D2B2",
                "Phase8D2B1_status",
            }.issubset(required):
                ready_value = (
                    parse_bool(frame.loc[0, "ready_for_phase8D2B2"])
                    if len(frame) > 0
                    else False
                )

                completed_value = (
                    str(frame.loc[0, "Phase8D2B1_status"]).strip()
                    == "completed"
                    if len(frame) > 0
                    else False
                )

                if ready_value and completed_value:
                    validity_score += 200
                elif completed_value:
                    validity_score += 50

            prefix_score = (
                20
                if prefix.lower() in path.name.lower()
                else 0
            )

            candidate_records.append(
                (
                    (
                        validity_score,
                        prefix_score,
                        path.stat().st_mtime,
                    ),
                    path,
                )
            )

    if not candidate_records:
        raise RuntimeError(
            "No TSV table was found with required columns "
            f"{sorted(required)} under the approved Phase 8 "
            "table and processed-data directories."
        )

    candidate_records.sort(
        key=lambda record: record[0],
        reverse=True,
    )

    selected = candidate_records[0][1]

    print(
        "Resolved frozen input table for "
        f"{sorted(required)}:\n  {selected}"
    )

    if len(candidate_records) > 1:
        print("Other schema-compatible candidates:")

        for score, path in candidate_records[1:6]:
            print(f"  score={score}: {path}")

    return selected


def get_selected_table(stats: Any, key: str) -> pd.DataFrame:
    if hasattr(stats, key):
        return getattr(stats, key).copy()
    if isinstance(stats, dict) and key in stats:
        return stats[key].copy()
    if hasattr(stats, 'keys') and key in list(stats.keys()):
        return stats[key].copy()
    available = [name for name in dir(stats) if not name.startswith('_')]
    raise KeyError(f'Could not resolve {key}. Available public attributes: {available}')


def locate_schizophrenia_file() -> Path:
    """Resolve the canonical ENIGMA schizophrenia fallback table."""

    filename = "Schizophrenia_case-controls_CortThick.csv"

    matches = sorted(
        ENIGMA_ROOT.rglob(filename),
        key=lambda path: str(path),
    )

    if not matches:
        raise RuntimeError(
            "The schizophrenia cortical-thickness fallback file "
            "was not found in the ENIGMA repository."
        )

    canonical = (
        ENIGMA_ROOT
        / "enigmatoolbox"
        / "datasets"
        / "summary_statistics"
        / filename
    )

    if canonical.is_file():
        selected = canonical
        selection_reason = "canonical_python_ENIGMA_Toolbox_copy"

    elif len(matches) == 1:
        selected = matches[0]
        selection_reason = "only_available_repository_copy"

    else:
        hash_groups: dict[str, list[Path]] = {}

        for path in matches:
            digest = sha256(path)
            hash_groups.setdefault(digest, []).append(path)

        if len(hash_groups) == 1:
            selected = matches[0]
            selection_reason = "byte_identical_repository_copies"
        else:
            details = {
                digest: [str(path) for path in paths]
                for digest, paths in hash_groups.items()
            }

            raise RuntimeError(
                "Multiple non-identical schizophrenia fallback "
                f"files were found: {details}"
            )

    table = read_delimited(selected)
    required_columns = {"Structure", "d_icv"}

    if not required_columns.issubset(set(table.columns)):
        raise RuntimeError(
            "Selected schizophrenia file does not contain the "
            f"required columns {sorted(required_columns)}: "
            f"{selected}; columns={table.columns.tolist()}"
        )

    print("Resolved schizophrenia fallback file:")
    print(f"  selected: {selected}")
    print(f"  reason: {selection_reason}")
    print(f"  shape: {table.shape[0]}x{table.shape[1]}")
    print(f"  SHA256: {sha256(selected)}")

    alternatives = [
        path
        for path in matches
        if path.resolve() != selected.resolve()
    ]

    if alternatives:
        print("  additional repository copies not selected:")

        selected_hash = sha256(selected)

        for path in alternatives:
            alternate_hash = sha256(path)
            identical = alternate_hash == selected_hash

            print(
                f"    {path} "
                f"[byte_identical={identical}; "
                f"SHA256={alternate_hash}]"
            )

    return selected


def inspect_neurosynth(rows: list[dict[str, Any]]) -> dict[str, Any]:
    files = [p for p in NEUROSYNTH_ROOT.rglob('*') if p.is_file() and '.git' not in p.parts]
    add(rows, 'neurosynth', 'resource_file_count', len(files), len(files) > 0)

    roles: dict[str, list[Path]] = {
        'legacy_database': [],
        'legacy_features': [],
        'coordinates': [],
        'metadata': [],
        'sparse_features': [],
        'vocabulary': [],
    }
    for p in files:
        low = p.name.lower()
        if low == 'database.txt' or ('database' in low and p.suffix in {'.txt', '.gz'}):
            roles['legacy_database'].append(p)
        if low == 'features.txt':
            roles['legacy_features'].append(p)
        if 'coordinate' in low and (low.endswith('.tsv.gz') or low.endswith('.csv.gz') or low.endswith('.tsv') or low.endswith('.csv')):
            roles['coordinates'].append(p)
        if 'metadata' in low and (low.endswith('.tsv.gz') or low.endswith('.csv.gz') or low.endswith('.tsv') or low.endswith('.csv')):
            roles['metadata'].append(p)
        if low.endswith('.npz') and ('feature' in low or 'tfidf' in low):
            roles['sparse_features'].append(p)
        if ('vocab' in low or 'vocabulary' in low) and (low.endswith('.txt') or low.endswith('.tsv')):
            roles['vocabulary'].append(p)

    for role, paths in roles.items():
        add(rows, 'neurosynth', f'{role}_count', len(paths), None, '|'.join(str(p.relative_to(PROJECT)) for p in paths[:20]))

    # Record all plausible small manifests and matrix shapes without modifying anything.
    for p in sorted(roles['sparse_features']):
        try:
            matrix = sparse.load_npz(p)
            add(rows, 'neurosynth_matrix', str(p.relative_to(PROJECT)), f'{matrix.shape[0]}x{matrix.shape[1]};nnz={matrix.nnz}', matrix.shape == (14371, 3228))
        except Exception as exc:
            add(rows, 'neurosynth_matrix', str(p.relative_to(PROJECT)), type(exc).__name__, False, str(exc))

    for p in sorted(roles['vocabulary']):
        try:
            vocab = [line.rstrip('\n\r') for line in p.open('r', encoding='utf-8', errors='replace') if line.strip()]
            add(rows, 'neurosynth_vocabulary', str(p.relative_to(PROJECT)), len(vocab), len(vocab) == 3228, '|'.join(vocab[:5]))
        except Exception as exc:
            add(rows, 'neurosynth_vocabulary', str(p.relative_to(PROJECT)), type(exc).__name__, False, str(exc))

    return roles


def main() -> None:
    PHASE_TABLES.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    add(rows, 'versions', 'python', sys.version.split()[0], True)
    add(rows, 'versions', 'numpy', np.__version__, True)
    add(rows, 'versions', 'pandas', pd.__version__, True)
    add(rows, 'versions', 'nibabel', nib.__version__, True)
    add(rows, 'versions', 'nilearn', nilearn.__version__, True)
    add(rows, 'versions', 'nimare', nimare.__version__, True)

    from nimare.io import convert_neurosynth_to_dataset
    from nimare.meta.cbma.mkda import MKDAChi2
    from nilearn.surface import vol_to_surf

    add(rows, 'api', 'convert_neurosynth_to_dataset_signature', inspect.signature(convert_neurosynth_to_dataset), True)
    add(rows, 'api', 'MKDAChi2_fit_signature', inspect.signature(MKDAChi2.fit), True)
    add(rows, 'api', 'vol_to_surf_signature', inspect.signature(vol_to_surf), True)

    cognitive_path = find_schema_table({'cognitive_domain', 'selected_term', 'selection_frozen'})
    clinical_path = find_schema_table({'disorder', 'selected_dataset_key', 'loading_mode', 'selection_frozen'})
    completion_path = find_schema_table({'ready_for_phase8D2B2', 'Phase8D2B1_status'})

    cognitive = pd.read_csv(cognitive_path, sep='\t')
    clinical = pd.read_csv(clinical_path, sep='\t')
    completion = pd.read_csv(completion_path, sep='\t')

    add(rows, 'frozen_inputs', 'cognitive_table', cognitive_path.relative_to(PROJECT), len(cognitive) == 8, sha256(cognitive_path))
    add(rows, 'frozen_inputs', 'clinical_table', clinical_path.relative_to(PROJECT), len(clinical) == 8, sha256(clinical_path))
    ready = parse_bool(completion.loc[0, 'ready_for_phase8D2B2']) and str(completion.loc[0, 'Phase8D2B1_status']).strip() == 'completed'
    add(rows, 'frozen_inputs', 'phase8D2B1_ready', ready, ready, sha256(completion_path))
    add(rows, 'frozen_inputs', 'cognitive_selection_frozen', cognitive['selection_frozen'].map(parse_bool).all(), cognitive['selection_frozen'].map(parse_bool).all())
    add(rows, 'frozen_inputs', 'clinical_selection_frozen', clinical['selection_frozen'].map(parse_bool).all(), clinical['selection_frozen'].map(parse_bool).all())

    roles = inspect_neurosynth(rows)

    # Surface/parcellation contract tests.
    #
    # Vertices outside the DK cortical projection remain NaN.
    # Schaefer parcels are reduced using finite DK vertices only,
    # preventing NaN propagation without introducing zero-fill bias.
    dk_probe = np.arange(
        1,
        69,
        dtype=float,
    )

    surf = np.asarray(
        parcel_to_surface(
            dk_probe,
            "aparc_fsa5",
            fill=np.nan,
        ),
        dtype=float,
    ).reshape(-1)

    sch = np.asarray(
        surface_to_parcel(
            surf,
            "schaefer_100_fsa5",
            red_op=finite_mean,
        ),
        dtype=float,
    ).reshape(-1)

    coverage = np.asarray(
        surface_to_parcel(
            np.isfinite(surf).astype(float),
            "schaefer_100_fsa5",
            red_op="mean",
        ),
        dtype=float,
    ).reshape(-1)

    schaefer_finite = (
        sch.size == 101
        and np.isfinite(sch[1:]).all()
    )

    schaefer_coverage = (
        coverage.size == 101
        and np.isfinite(coverage[1:]).all()
        and (coverage[1:] > 0).all()
    )

    add(
        rows,
        "parcellation",
        "aparc_fsa5_surface_vertices",
        surf.size,
        surf.size == 20484,
    )

    add(
        rows,
        "parcellation",
        "schaefer100_raw_values",
        sch.size,
        sch.size == 101,
    )

    add(
        rows,
        "parcellation",
        "schaefer100_background_index0_nan_or_fill",
        sch[0] if sch.size else "missing",
        sch.size == 101,
    )

    add(
        rows,
        "parcellation",
        "schaefer100_cortical_values_finite",
        (
            int(np.isfinite(sch[1:]).sum())
            if sch.size == 101
            else -1
        ),
        schaefer_finite,
    )

    add(
        rows,
        "parcellation",
        "schaefer100_parcels_with_DK_coverage",
        (
            int((coverage[1:] > 0).sum())
            if coverage.size == 101
            else -1
        ),
        schaefer_coverage,
        (
            "minimum_DK_vertex_fraction="
            f"{float(np.min(coverage[1:])):.12g}"
            if coverage.size == 101
            else "invalid_coverage_vector"
        ),
    )

    # Clinical table contract and canonical region order.
    loaded: dict[str, pd.DataFrame] = {}
    for row in clinical.sort_values('fixed_rank').itertuples(index=False):
        disorder = str(row.disorder)
        key = str(row.selected_dataset_key)
        mode = str(row.loading_mode)
        if mode == 'direct_repository_file':
            table = read_delimited(locate_schizophrenia_file())
        else:
            table = get_selected_table(load_summary_stats(disorder), key)
        loaded[disorder] = table
        required = {'Structure', 'd_icv'}
        add(rows, 'clinical_load', disorder, f'{table.shape[0]}x{table.shape[1]}', required.issubset(table.columns), '|'.join(map(str, table.columns)))

    canonical = loaded['adhd'][['Structure', 'd_icv']].copy()
    canonical['norm'] = canonical['Structure'].map(normalize_region)
    canonical_ok = len(canonical) == 68 and canonical['norm'].nunique() == 68 and canonical['d_icv'].notna().all()
    add(rows, 'clinical_regions', 'canonical_adhd_DK68', f'rows={len(canonical)};unique={canonical.norm.nunique()}', canonical_ok)
    canonical_set = set(canonical['norm'])

    for disorder, table in loaded.items():
        work = table[['Structure', 'd_icv']].copy()
        work['norm'] = work['Structure'].map(normalize_region)
        work['d_icv'] = pd.to_numeric(work['d_icv'], errors='coerce')
        matched = work[work['norm'].isin(canonical_set) & work['d_icv'].notna()].copy()
        extra = work.loc[~work['norm'].isin(canonical_set), 'Structure'].astype(str).tolist()
        counts = matched.groupby('norm').size()
        dup_regions = counts[counts > 1].index.tolist()
        conflicting: list[str] = []
        for region in dup_regions:
            values = matched.loc[matched['norm'] == region, 'd_icv'].to_numpy(dtype=float)
            if not np.allclose(values, values[0], rtol=0, atol=1e-12, equal_nan=True):
                conflicting.append(region)
        missing = sorted(canonical_set - set(matched['norm']))
        resolvable = not missing and not conflicting and matched['norm'].nunique() == 68
        detail = json.dumps({'extra_rows': extra, 'duplicate_regions': dup_regions, 'conflicting_regions': conflicting, 'missing_regions': missing})
        add(rows, 'clinical_regions', disorder, f'matched_rows={len(matched)};unique={matched.norm.nunique()}', resolvable, detail)

    # Existing project atlas/mask metadata useful for later LH50 alignment.
    manifest_candidates: list[Path] = []
    for path in list(PROC.rglob('*.tsv')) + list(PHASE_TABLES.rglob('*.tsv')):
        low = path.name.lower()
        if 'schaefer' in low and any(token in low for token in ('parcel', 'label', 'manifest', 'crosswalk', 'mask')):
            manifest_candidates.append(path)
    add(rows, 'existing_atlas_metadata', 'candidate_count', len(manifest_candidates), len(manifest_candidates) > 0, '|'.join(str(p.relative_to(PROJECT)) for p in manifest_candidates[:30]))

    audit = pd.DataFrame(rows)
    audit.to_csv(OUT, sep='\t', index=False)

    required_checks = {
        'Phase8D2B1_ready': ready,
        'frozen_cognitive_rows_8': len(cognitive) == 8,
        'frozen_clinical_rows_8': len(clinical) == 8,
        'DK_surface_vertices_20484': surf.size == 20484,
        'Schaefer_raw_values_101': sch.size == 101,
        'Schaefer_cortical_values_100_finite': schaefer_finite,
        'Schaefer_cortical_parcels_all_have_DK_coverage': schaefer_coverage,
        'all_clinical_DK68_resolvable': all(
            bool(v) for v in audit.loc[(audit.section == 'clinical_regions') & (audit.item != 'canonical_adhd_DK68'), 'passed'].tolist()
        ),
        'neurosynth_resources_present': any(len(v) > 0 for v in roles.values()),
    }
    ready_for_generation = all(required_checks.values())
    summary = pd.DataFrame([{**required_checks, 'ready_for_phase8D2B2_map_generation': ready_for_generation, 'Phase8D2B2_preflight_status': 'completed' if ready_for_generation else 'failed_validation'}])
    summary.to_csv(SUMMARY, sep='\t', index=False)

    print('===== PHASE 8D2B2 PREFLIGHT COMPLETION =====')
    print(summary.to_string(index=False))
    print('\n===== KEY API CONTRACTS =====')
    print(audit[audit.section.isin(['versions', 'api'])].to_string(index=False))
    print('\n===== NEUROSYNTH RESOURCE DISCOVERY =====')
    print(audit[audit.section.str.startswith('neurosynth')].to_string(index=False))
    print('\n===== PARCELLATION CONTRACT =====')
    print(audit[audit.section == 'parcellation'].to_string(index=False))
    print('\n===== CLINICAL DK68 RESOLUTION =====')
    print(audit[audit.section == 'clinical_regions'].to_string(index=False))
    print('\n===== EXISTING ATLAS METADATA =====')
    print(audit[audit.section == 'existing_atlas_metadata'].to_string(index=False))

    if not ready_for_generation:
        failed = [
            name
            for name, passed in required_checks.items()
            if not passed
        ]
        raise RuntimeError(
            f'Phase 8D2B2 preflight validation failed: {failed}'
        )


if __name__ == '__main__':
    main()
