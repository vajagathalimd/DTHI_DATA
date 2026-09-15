#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT = Path('.')
TABLES = PROJECT / '07_tables/main_tables/phase8'
PROC8D = PROJECT / '03_processed_data/functional_imaging/phase8D'
PROC8A = PROJECT / '03_processed_data/functional_imaging/phase8A'
PROC3C = PROJECT / '03_processed_data/spatial_hierarchy/phase3C'
OUT = PROC8D / 'phase8D2C'

B2_SUMMARY = TABLES / 'phase8D2B2_completion_summary.tsv'
A3_SUMMARY = TABLES / 'phase8A3_completion_summary.tsv'
MAPS = PROC8D / 'phase8D2B2_maps/matrices/phase8D2B2_combined_Schaefer100_LH50_matrix.tsv'
MAPS_Z = PROC8D / 'phase8D2B2_maps/matrices/phase8D2B2_combined_Schaefer100_LH50_zscored.tsv'
DTHI = PROC3C / 'AHBA_Schaefer100_donor_balanced_DTHI_scores.tsv'
DTHI_DONOR = PROC3C / 'AHBA_donor_Schaefer100_DTHI_scores.tsv'
SPINS = PROC8A / 'phase8A3_Schaefer100_LH_spin_resampling_indices_10000.npy'
FROZEN_MASK_RESOURCE = PROC8A / 'phase8A2_Schaefer100_analysis_masks.tsv'

DTHI_OUT = OUT / 'phase8D2C1_DTHI_Schaefer100_LH50_matrix.tsv'
MASK_OUT = OUT / 'phase8D2C1_analysis_masks_LH50.tsv'
INPUT_OUT = OUT / 'phase8D2C1_selected_inputs.tsv'
PLAN_OUT = TABLES / 'phase8D2C1_frozen_statistical_plan.tsv'
AUDIT_OUT = TABLES / 'phase8D2C1_input_audit.tsv'
SUMMARY_OUT = TABLES / 'phase8D2C1_completion_summary.tsv'
HASH_OUT = TABLES / 'phase8D2C1_canonical_output_hashes.tsv'

DTHI_COLUMNS = [
    'patterning_arealization',
    'progenitor_radial_glia',
    'neurogenesis_migration_layering',
    'axon_guidance_neurite_outgrowth',
    'synaptic_assembly_receptor_trafficking',
    'astrocyte_maturation_metabolic_support',
    'oligodendrocyte_myelination',
    'activity_dependent_plasticity',
    'synaptic_membrane_structural_candidates',
    'preliminary_maturation_balance',
]


class ValidationError(RuntimeError):
    pass


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temp, sep='\t', index=False)
    temp.replace(path)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise ValidationError(f'Missing required file: {path}')


def require_completed(path: Path, status: str, ready: str | None = None) -> None:
    frame = pd.read_csv(path, sep='\t')
    if len(frame) != 1 or status not in frame.columns:
        raise ValidationError(f'Invalid completion summary: {path}')
    if str(frame.loc[0, status]).strip() != 'completed':
        raise ValidationError(f'{status} is not completed: {path}')
    if ready is not None:
        if ready not in frame.columns or not parse_bool(frame.loc[0, ready]):
            raise ValidationError(f'{ready} is not true: {path}')


def parcel_column(frame: pd.DataFrame) -> str:
    matches = [
        column for column in ['parcel_id', 'parcel_index_full', 'parcel']
        if column in frame.columns
    ]
    if len(matches) != 1:
        raise ValidationError(
            f'Expected one parcel ID column, found {matches}; '
            f'columns={frame.columns.tolist()}'
        )
    return matches[0]


def canonical_spins(array: np.ndarray) -> tuple[np.ndarray, str]:
    spins = np.asarray(array)
    if spins.shape == (10000, 50):
        orientation = 'rotations_by_parcels'
    elif spins.shape == (50, 10000):
        spins = spins.T
        orientation = 'transposed_from_parcels_by_rotations'
    else:
        raise ValidationError(f'Unexpected LH spin shape: {spins.shape}')
    if not np.all(np.isfinite(spins)):
        raise ValidationError('LH spin resource contains non-finite indices.')
    if not np.allclose(spins, np.rint(spins), rtol=0, atol=0):
        raise ValidationError('LH spin resource contains non-integer indices.')
    spins = np.rint(spins).astype(np.int64)
    if int(spins.min()) < 0 or int(spins.max()) > 49:
        raise ValidationError(
            f'LH spin index range must be 0-49, found {spins.min()}-{spins.max()}.'
        )
    return spins, orientation


def add(rows: list[dict[str, Any]], section: str, item: str, value: Any,
        passed: bool, detail: str = '') -> None:
    rows.append({
        'section': section,
        'item': item,
        'value': str(value),
        'passed': bool(passed),
        'detail': detail,
    })


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    required = [
        B2_SUMMARY, A3_SUMMARY, MAPS, MAPS_Z, DTHI, DTHI_DONOR,
        SPINS, FROZEN_MASK_RESOURCE,
    ]
    for path in required:
        require_file(path)

    require_completed(B2_SUMMARY, 'Phase8D2B2_status', 'ready_for_phase8D2C')
    require_completed(A3_SUMMARY, 'Phase8A3_status')

    audit: list[dict[str, Any]] = []
    add(audit, 'upstream', 'Phase8D2B2_ready', True, True)
    add(audit, 'upstream', 'Phase8A3_spatial_null_ready', True, True)

    maps = pd.read_csv(MAPS, sep='\t')
    maps_z = pd.read_csv(MAPS_Z, sep='\t')
    cognitive = [c for c in maps.columns if c.startswith('cognitive_')]
    clinical = [c for c in maps.columns if c.startswith('clinical_')]
    external = cognitive + clinical
    if len(maps) != 50 or len(cognitive) != 8 or len(clinical) != 8:
        raise ValidationError(
            f'Expected 50x16 LH map matrix with 8 cognitive and 8 clinical maps; '
            f'found rows={len(maps)}, cognitive={len(cognitive)}, clinical={len(clinical)}.'
        )
    if not np.isfinite(maps[external].to_numpy(dtype=float)).all():
        raise ValidationError('Combined Phase8D2B2 map matrix contains non-finite values.')
    z_external = [
        c for c in maps_z.columns
        if c.startswith('cognitive_') or c.startswith('clinical_')
    ]
    if external != z_external:
        raise ValidationError('Raw and z-scored Phase8D2B2 map columns are not identical.')

    add(audit, 'external_maps', 'cognitive_maps', len(cognitive), len(cognitive) == 8,
        '|'.join(cognitive))
    add(audit, 'external_maps', 'clinical_maps', len(clinical), len(clinical) == 8,
        '|'.join(clinical))
    add(audit, 'external_maps', 'LH_parcels', len(maps), len(maps) == 50)
    add(audit, 'external_maps', 'all_values_finite', True, True)

    dthi = pd.read_csv(DTHI, sep='\t')
    donor = pd.read_csv(DTHI_DONOR, sep='\t')
    missing_balanced = [c for c in DTHI_COLUMNS if c not in dthi.columns]
    missing_donor = [c for c in DTHI_COLUMNS if c not in donor.columns]
    if missing_balanced or missing_donor:
        raise ValidationError(
            f'Missing DTHI columns: balanced={missing_balanced}; donor={missing_donor}'
        )

    dthi_parcel = parcel_column(dthi)
    donor_parcel = parcel_column(donor)
    donor_id_matches = [c for c in ['donor_id', 'donor', 'donor_name'] if c in donor.columns]
    if len(donor_id_matches) != 1:
        raise ValidationError(f'Expected one donor ID column, found {donor_id_matches}.')
    donor_id = donor_id_matches[0]

    dthi[dthi_parcel] = pd.to_numeric(dthi[dthi_parcel], errors='coerce')
    dthi_lh = dthi[dthi[dthi_parcel].between(1, 50, inclusive='both')].copy()
    dthi_lh = dthi_lh.sort_values(dthi_parcel).drop_duplicates(dthi_parcel)
    if len(dthi_lh) != 50:
        raise ValidationError(f'Expected 50 donor-balanced LH parcels; found {len(dthi_lh)}.')
    if not np.array_equal(dthi_lh[dthi_parcel].to_numpy(dtype=int), np.arange(1, 51)):
        raise ValidationError('Donor-balanced DTHI rows are not ordered as parcel IDs 1-50.')
    if not np.isfinite(dthi_lh[DTHI_COLUMNS].to_numpy(dtype=float)).all():
        raise ValidationError('Donor-balanced DTHI matrix contains non-finite values.')
    if np.any(np.std(dthi_lh[DTHI_COLUMNS].to_numpy(dtype=float), axis=0) <= 0):
        raise ValidationError('At least one DTHI map is constant across LH50.')

    if 'n_donors' in dthi_lh.columns:
        n_donors = pd.to_numeric(dthi_lh['n_donors'], errors='coerce')
        primary = (n_donors >= 3).to_numpy(dtype=bool)
        high = (n_donors >= 4).to_numpy(dtype=bool)
        mask_source = 'n_donors_thresholds'
    elif 'AHBA_support_class' in dthi_lh.columns:
        support = dthi_lh['AHBA_support_class'].astype(str).str.lower().str.strip()
        primary = support.isin({'high', 'moderate'}).to_numpy(dtype=bool)
        high = support.eq('high').to_numpy(dtype=bool)
        mask_source = 'AHBA_support_class'
    else:
        raise ValidationError('DTHI table has neither n_donors nor AHBA_support_class.')

    if int(primary.sum()) != 49 or int(high.sum()) != 46:
        raise ValidationError(
            f'Expected primary LH49 and high-support LH46; found {primary.sum()} and {high.sum()}.'
        )

    donor[donor_parcel] = pd.to_numeric(donor[donor_parcel], errors='coerce')
    donor_lh = donor[donor[donor_parcel].between(1, 50, inclusive='both')].copy()
    donors = sorted(donor_lh[donor_id].dropna().astype(str).unique().tolist())
    duplicates = int(donor_lh.duplicated([donor_id, donor_parcel]).sum())
    if len(donors) != 5 or duplicates != 0:
        raise ValidationError(
            f'Donor DTHI contract failed: donors={len(donors)}, duplicate donor-parcels={duplicates}.'
        )

    add(audit, 'DTHI', 'DTHI_maps', len(DTHI_COLUMNS), len(DTHI_COLUMNS) == 10,
        '|'.join(DTHI_COLUMNS))
    add(audit, 'DTHI', 'balanced_LH_parcels', len(dthi_lh), len(dthi_lh) == 50)
    add(audit, 'DTHI', 'primary_LH_parcels', int(primary.sum()), int(primary.sum()) == 49,
        mask_source)
    add(audit, 'DTHI', 'high_support_LH_parcels', int(high.sum()), int(high.sum()) == 46,
        mask_source)
    add(audit, 'DTHI', 'AHBA_donors_for_LODO', len(donors), len(donors) == 5,
        '|'.join(donors))

    spins, orientation = canonical_spins(np.load(SPINS))
    unique_counts = np.asarray([np.unique(row).size for row in spins], dtype=int)
    add(audit, 'spatial_null', 'LH_spin_shape', f'{spins.shape[0]}x{spins.shape[1]}',
        spins.shape == (10000, 50), orientation)
    add(audit, 'spatial_null', 'LH_spin_index_range', f'{spins.min()}-{spins.max()}', True)
    add(audit, 'spatial_null', 'nearest_neighbour_collisions_present',
        f'min={unique_counts.min()};median={np.median(unique_counts):.1f};max={unique_counts.max()}',
        int(unique_counts.min()) < 50,
        'hemisphere_preserving_nonbijective_spin_resampling')

    maps_parcel = parcel_column(maps)
    maps_ids = pd.to_numeric(maps[maps_parcel], errors='coerce').to_numpy(dtype=int)
    if not np.array_equal(maps_ids, np.arange(1, 51)):
        raise ValidationError('Phase8D2B2 combined map rows are not parcel IDs 1-50.')

    harmonized = pd.DataFrame({'parcel_id': np.arange(1, 51)})
    for column in ['parcel_label', 'hemisphere', 'network', 'atlas']:
        if column in maps.columns:
            harmonized[column] = maps[column].values
        elif column in dthi_lh.columns:
            harmonized[column] = dthi_lh[column].values
    for column in DTHI_COLUMNS:
        harmonized[column] = dthi_lh[column].to_numpy(dtype=float)
    atomic_tsv(harmonized, DTHI_OUT)

    mask_frame = pd.DataFrame({
        'parcel_id': np.arange(1, 51),
        'primary_LH49': primary,
        'high_support_LH46': high,
    })
    if 'parcel_label' in harmonized.columns:
        mask_frame.insert(1, 'parcel_label', harmonized['parcel_label'])
    atomic_tsv(mask_frame, MASK_OUT)

    inputs = pd.DataFrame([
        {'resource': 'Phase8D2B2_combined_LH50_maps', 'relative_path': str(MAPS.relative_to(PROJECT)),
         'rows': len(maps), 'analysis_columns': 16, 'SHA256': sha256(MAPS)},
        {'resource': 'AHBA_donor_balanced_DTHI', 'relative_path': str(DTHI.relative_to(PROJECT)),
         'rows': len(dthi), 'analysis_columns': 10, 'SHA256': sha256(DTHI)},
        {'resource': 'AHBA_donor_parcel_DTHI_for_LODO', 'relative_path': str(DTHI_DONOR.relative_to(PROJECT)),
         'rows': len(donor), 'analysis_columns': 10, 'SHA256': sha256(DTHI_DONOR)},
        {'resource': 'Phase8A3_LH_spins', 'relative_path': str(SPINS.relative_to(PROJECT)),
         'rows': 10000, 'analysis_columns': 50, 'SHA256': sha256(SPINS)},
        {'resource': 'Phase8A2_frozen_mask_resource', 'relative_path': str(FROZEN_MASK_RESOURCE.relative_to(PROJECT)),
         'rows': len(pd.read_csv(FROZEN_MASK_RESOURCE, sep='\t')), 'analysis_columns': '',
         'SHA256': sha256(FROZEN_MASK_RESOURCE)},
    ])
    atomic_tsv(inputs, INPUT_OUT)

    plan = pd.DataFrame([
        {
            'analysis_family': 'primary_spatial_correspondence',
            'scope': 'fixed_primary_LH49',
            'DTHI_maps': 10,
            'external_maps': 16,
            'tests': 160,
            'statistic': 'Spearman_rho',
            'spatial_null': 'rotate_complete_external_LH50_then_apply_fixed_LH49_mask',
            'rotations': 10000,
            'multiple_testing': 'BH_global160;BH_cognitive80;BH_clinical80;global_maxT160;family_maxT80',
            'interpretation': 'primary_inference',
        },
        {
            'analysis_family': 'high_support_sensitivity',
            'scope': 'fixed_high_support_LH46',
            'DTHI_maps': 10,
            'external_maps': 16,
            'tests': 160,
            'statistic': 'Spearman_rho',
            'spatial_null': 'same_10000_LH_rotations_then_apply_fixed_LH46_mask',
            'rotations': 10000,
            'multiple_testing': 'BH_within_sensitivity160',
            'interpretation': 'nested_support_sensitivity_not_independent_replication',
        },
        {
            'analysis_family': 'leave_one_donor_out_robustness',
            'scope': 'primary_LH49_pairwise_finite',
            'DTHI_maps': 10,
            'external_maps': 16,
            'tests': 160,
            'statistic': 'Spearman_rho_after_each_of_5_donor_omissions',
            'spatial_null': 'not_retested_per_LODO',
            'rotations': 0,
            'multiple_testing': 'not_applicable',
            'interpretation': 'direction_and_effect_size_stability_not_replication',
        },
    ])
    atomic_tsv(plan, PLAN_OUT)
    atomic_tsv(pd.DataFrame(audit), AUDIT_OUT)

    ready = (
        len(cognitive) == 8 and len(clinical) == 8 and len(DTHI_COLUMNS) == 10
        and int(primary.sum()) == 49 and int(high.sum()) == 46
        and len(donors) == 5 and spins.shape == (10000, 50)
    )
    summary = pd.DataFrame([{
        'Phase8D2B2_status_confirmed': True,
        'Phase8A3_status_confirmed': True,
        'DTHI_maps': 10,
        'cognitive_maps': 8,
        'clinical_maps': 8,
        'primary_spatial_tests': 160,
        'primary_LH_parcels': int(primary.sum()),
        'high_support_LH_parcels': int(high.sum()),
        'AHBA_donors_for_LODO': len(donors),
        'spatial_null_rotations': spins.shape[0],
        'hemisphere_preserved': True,
        'nearest_neighbour_collisions_present': bool(int(unique_counts.min()) < 50),
        'ready_for_phase8D2C2': ready,
        'Phase8D2C1_status': 'completed' if ready else 'failed_validation',
    }])
    atomic_tsv(summary, SUMMARY_OUT)

    hash_paths = [DTHI_OUT, MASK_OUT, INPUT_OUT, PLAN_OUT, AUDIT_OUT, SUMMARY_OUT]
    hashes = pd.DataFrame([
        {'relative_path': str(path.relative_to(PROJECT)),
         'size_bytes': path.stat().st_size, 'SHA256': sha256(path)}
        for path in hash_paths
    ])
    atomic_tsv(hashes, HASH_OUT)

    print('===== PHASE 8D2C1 COMPLETION =====')
    print(summary.to_string(index=False))
    print('\n===== FROZEN STATISTICAL PLAN =====')
    print(plan.to_string(index=False))
    print('\n===== INPUT AUDIT =====')
    print(pd.DataFrame(audit).to_string(index=False))
    print('\n===== CANONICAL OUTPUT HASHES =====')
    print(hashes.to_string(index=False))

    if not ready:
        raise RuntimeError('Phase 8D2C1 validation failed.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Phase 8D2C1 failed: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise
