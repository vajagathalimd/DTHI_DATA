#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import statsmodels
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

PROJECT = Path('.')
TABLES = PROJECT / '07_tables/main_tables/phase8'
PROC8D = PROJECT / '03_processed_data/functional_imaging/phase8D'
PROC8A = PROJECT / '03_processed_data/functional_imaging/phase8A'
OUT = PROC8D / 'phase8D2C'

C1_SUMMARY = TABLES / 'phase8D2C1_completion_summary.tsv'
DTHI_FILE = OUT / 'phase8D2C1_DTHI_Schaefer100_LH50_matrix.tsv'
MASK_FILE = OUT / 'phase8D2C1_analysis_masks_LH50.tsv'
EXTERNAL_FILE = (
    PROC8D / 'phase8D2B2_maps/matrices/'
    'phase8D2B2_combined_Schaefer100_LH50_matrix.tsv'
)
SPIN_FILE = PROC8A / 'phase8A3_Schaefer100_LH_spin_resampling_indices_10000.npy'

PRIMARY_OUT = OUT / 'phase8D2C2A_primary_spatial_correspondence.tsv'
SENSITIVITY_OUT = OUT / 'phase8D2C2A_high_support_sensitivity.tsv'
SIGNIFICANT_OUT = OUT / 'phase8D2C2A_primary_significant_results.tsv'
NULL_OUT = OUT / 'phase8D2C2A_spatial_null_distributions.npz'
AUDIT_OUT = TABLES / 'phase8D2C2A_spatial_correspondence_audit.tsv'
SUMMARY_OUT = TABLES / 'phase8D2C2A_completion_summary.tsv'
HASH_OUT = TABLES / 'phase8D2C2A_canonical_output_hashes.tsv'

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

COGNITIVE_COLUMNS = [
    'cognitive_attention',
    'cognitive_executive_control',
    'cognitive_memory',
    'cognitive_language',
    'cognitive_social_cognition',
    'cognitive_emotion_affect',
    'cognitive_reward_motivation',
    'cognitive_sensorimotor',
]

CLINICAL_COLUMNS = [
    'clinical_adhd',
    'clinical_asd',
    'clinical_bipolar',
    'clinical_depression',
    'clinical_epilepsy',
    'clinical_ocd',
    'clinical_parkinsons',
    'clinical_schizophrenia',
]

EXTERNAL_COLUMNS = COGNITIVE_COLUMNS + CLINICAL_COLUMNS
N_ROTATIONS = 10000


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


def require_c1_completed() -> None:
    frame = pd.read_csv(C1_SUMMARY, sep='\t')
    if len(frame) != 1:
        raise ValidationError('Phase 8D2C1 completion summary must contain one row.')
    if str(frame.loc[0, 'Phase8D2C1_status']).strip() != 'completed':
        raise ValidationError('Phase 8D2C1 is not completed.')
    if not parse_bool(frame.loc[0, 'ready_for_phase8D2C2']):
        raise ValidationError('Phase 8D2C1 is not ready for Phase 8D2C2.')


def parcel_column(frame: pd.DataFrame) -> str:
    matches = [
        column
        for column in ['parcel_id', 'parcel_index_full', 'parcel']
        if column in frame.columns
    ]
    if len(matches) != 1:
        raise ValidationError(
            f'Expected one parcel identifier column; found {matches}. '
            f'Columns={frame.columns.tolist()}'
        )
    return matches[0]


def bool_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        raise ValidationError(f'Missing mask column: {name}')
    return frame[name].map(parse_bool).to_numpy(dtype=bool)


def canonical_spins(array: np.ndarray) -> np.ndarray:
    spins = np.asarray(array)
    if spins.shape == (50, N_ROTATIONS):
        spins = spins.T
    if spins.shape != (N_ROTATIONS, 50):
        raise ValidationError(f'Unexpected spin matrix shape: {spins.shape}')
    if not np.isfinite(spins).all():
        raise ValidationError('Spin matrix contains non-finite indices.')
    if not np.allclose(spins, np.rint(spins), rtol=0, atol=0):
        raise ValidationError('Spin matrix contains non-integer indices.')
    spins = np.rint(spins).astype(np.int16)
    if int(spins.min()) < 0 or int(spins.max()) > 49:
        raise ValidationError(
            f'Spin indices must range from 0 to 49; found {spins.min()}-{spins.max()}.'
        )
    return spins


def unit_rank_vector(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.asarray(values, dtype=float), method='average')
    centered = ranks - ranks.mean()
    norm = float(np.sqrt(np.sum(centered * centered)))
    if not np.isfinite(norm) or norm <= 0:
        raise ValidationError('A map is constant after rank transformation.')
    return centered / norm


def unit_rank_rows(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.asarray(values, dtype=float), method='average', axis=1)
    centered = ranks - ranks.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(centered * centered, axis=1, keepdims=True))
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        bad = np.flatnonzero((~np.isfinite(norms[:, 0])) | (norms[:, 0] <= 0))
        raise ValidationError(
            f'{bad.size} rotated maps are constant or invalid after ranking; '
            f'first indices={bad[:10].tolist()}'
        )
    return centered / norms


def bh(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValidationError('Non-finite P values supplied to BH correction.')
    return multipletests(array, alpha=0.05, method='fdr_bh')[1]


def run_scope(
    scope_name: str,
    mask: np.ndarray,
    dthi_values: np.ndarray,
    external_values: np.ndarray,
    spins: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    n_parcels = int(mask.sum())
    dthi_ranked = np.column_stack(
        [unit_rank_vector(dthi_values[mask, index]) for index in range(dthi_values.shape[1])]
    )

    null = np.empty((spins.shape[0], len(EXTERNAL_COLUMNS) * len(DTHI_COLUMNS)), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    test_index = 0

    print(f'===== {scope_name}: {n_parcels} PARCELS =====', flush=True)

    for external_index, external_name in enumerate(EXTERNAL_COLUMNS):
        external_full = external_values[:, external_index]
        observed_rank = unit_rank_vector(external_full[mask])
        observed_all = observed_rank @ dthi_ranked

        rotated_full = external_full[spins]
        rotated_scope = rotated_full[:, mask]
        rotated_rank = unit_rank_rows(rotated_scope)
        null_all = rotated_rank @ dthi_ranked

        family = 'cognitive' if external_name.startswith('cognitive_') else 'clinical'

        for dthi_index, dthi_name in enumerate(DTHI_COLUMNS):
            observed = float(observed_all[dthi_index])
            null_vector = np.asarray(null_all[:, dthi_index], dtype=float)
            empirical_p = float(
                (1 + np.count_nonzero(np.abs(null_vector) >= abs(observed)))
                / (len(null_vector) + 1)
            )

            null[:, test_index] = null_vector.astype(np.float32)
            rows.append(
                {
                    'scope': scope_name,
                    'test_index': test_index,
                    'DTHI_map': dthi_name,
                    'external_map': external_name,
                    'external_family': family,
                    'n_parcels': n_parcels,
                    'spearman_rho': observed,
                    'abs_spearman_rho': abs(observed),
                    'p_spin_two_sided': empirical_p,
                    'null_mean': float(np.mean(null_vector)),
                    'null_sd': float(np.std(null_vector, ddof=1)),
                    'null_q025': float(np.quantile(null_vector, 0.025)),
                    'null_q975': float(np.quantile(null_vector, 0.975)),
                }
            )
            test_index += 1

        print(
            f'[{external_index + 1:02d}/16] {external_name}: '
            f'10 DTHI correlations and {spins.shape[0]} spatial nulls',
            flush=True,
        )

    frame = pd.DataFrame(rows)
    if len(frame) != 160 or null.shape != (N_ROTATIONS, 160):
        raise ValidationError(
            f'{scope_name} produced rows={len(frame)} and null shape={null.shape}; '
            'expected 160 and 10000x160.'
        )
    return frame, null


def add_primary_corrections(frame: pd.DataFrame, null: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    raw_p = result['p_spin_two_sided'].to_numpy(dtype=float)
    result['q_BH_global160'] = bh(raw_p)

    result['q_BH_family80'] = np.nan
    result['p_maxT_family80'] = np.nan
    result['p_maxT_global160'] = np.nan

    max_global = np.max(np.abs(null), axis=1)
    observed_abs = result['abs_spearman_rho'].to_numpy(dtype=float)
    result['p_maxT_global160'] = [
        (1 + np.count_nonzero(max_global >= value)) / (len(max_global) + 1)
        for value in observed_abs
    ]

    for family in ['cognitive', 'clinical']:
        family_mask = result['external_family'].eq(family).to_numpy()
        if int(family_mask.sum()) != 80:
            raise ValidationError(f'{family} family does not contain 80 tests.')
        result.loc[family_mask, 'q_BH_family80'] = bh(raw_p[family_mask])
        max_family = np.max(np.abs(null[:, family_mask]), axis=1)
        result.loc[family_mask, 'p_maxT_family80'] = [
            (1 + np.count_nonzero(max_family >= value)) / (len(max_family) + 1)
            for value in observed_abs[family_mask]
        ]

    result['significant_BH_global160'] = result['q_BH_global160'] < 0.05
    result['significant_BH_family80'] = result['q_BH_family80'] < 0.05
    result['significant_maxT_global160'] = result['p_maxT_global160'] < 0.05
    result['significant_maxT_family80'] = result['p_maxT_family80'] < 0.05
    result['effect_direction'] = np.where(result['spearman_rho'] >= 0, 'positive', 'negative')
    return result.sort_values(
        ['p_maxT_global160', 'q_BH_global160', 'p_spin_two_sided', 'abs_spearman_rho'],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def add_sensitivity_corrections(
    frame: pd.DataFrame,
    primary: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    result['q_BH_sensitivity160'] = bh(result['p_spin_two_sided'].to_numpy(dtype=float))
    result['significant_BH_sensitivity160'] = result['q_BH_sensitivity160'] < 0.05

    primary_effects = primary[
        ['DTHI_map', 'external_map', 'spearman_rho']
    ].rename(columns={'spearman_rho': 'primary_LH49_rho'})
    result = result.merge(primary_effects, on=['DTHI_map', 'external_map'], how='left', validate='one_to_one')
    result['delta_rho_high_support_minus_primary'] = (
        result['spearman_rho'] - result['primary_LH49_rho']
    )
    result['same_direction_as_primary'] = (
        np.sign(result['spearman_rho']) == np.sign(result['primary_LH49_rho'])
    )
    return result.sort_values(
        ['q_BH_sensitivity160', 'p_spin_two_sided', 'abs_spearman_rho'],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def add_audit(
    rows: list[dict[str, Any]],
    section: str,
    item: str,
    value: Any,
    passed: bool,
    detail: str = '',
) -> None:
    rows.append(
        {
            'section': section,
            'item': item,
            'value': str(value),
            'passed': bool(passed),
            'detail': detail,
        }
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    for path in [C1_SUMMARY, DTHI_FILE, MASK_FILE, EXTERNAL_FILE, SPIN_FILE]:
        require_file(path)
    require_c1_completed()

    dthi = pd.read_csv(DTHI_FILE, sep='\t')
    masks = pd.read_csv(MASK_FILE, sep='\t')
    external = pd.read_csv(EXTERNAL_FILE, sep='\t')

    missing_dthi = [column for column in DTHI_COLUMNS if column not in dthi.columns]
    missing_external = [column for column in EXTERNAL_COLUMNS if column not in external.columns]
    if missing_dthi or missing_external:
        raise ValidationError(
            f'Missing analysis columns: DTHI={missing_dthi}; external={missing_external}'
        )

    dthi_parcel = parcel_column(dthi)
    mask_parcel = parcel_column(masks)
    external_parcel = parcel_column(external)

    dthi_ids = pd.to_numeric(dthi[dthi_parcel], errors='coerce').to_numpy(dtype=int)
    mask_ids = pd.to_numeric(masks[mask_parcel], errors='coerce').to_numpy(dtype=int)
    external_ids = pd.to_numeric(external[external_parcel], errors='coerce').to_numpy(dtype=int)
    expected_ids = np.arange(1, 51)
    if not (
        np.array_equal(dthi_ids, expected_ids)
        and np.array_equal(mask_ids, expected_ids)
        and np.array_equal(external_ids, expected_ids)
    ):
        raise ValidationError('DTHI, mask, and external matrices are not aligned as LH parcel IDs 1-50.')

    primary_mask = bool_column(masks, 'primary_LH49')
    high_mask = bool_column(masks, 'high_support_LH46')
    if int(primary_mask.sum()) != 49 or int(high_mask.sum()) != 46:
        raise ValidationError(
            f'Expected LH49/LH46 masks; found {primary_mask.sum()}/{high_mask.sum()}.'
        )
    if not np.all(high_mask <= primary_mask):
        raise ValidationError('High-support LH46 is not nested within primary LH49.')

    dthi_values = dthi[DTHI_COLUMNS].to_numpy(dtype=float)
    external_values = external[EXTERNAL_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(dthi_values).all() or not np.isfinite(external_values).all():
        raise ValidationError('Input map matrices contain non-finite values.')

    spins = canonical_spins(np.load(SPIN_FILE))
    unique_counts = np.asarray([np.unique(row).size for row in spins], dtype=int)
    identity_count = int(np.sum(np.all(spins == np.arange(50), axis=1)))

    primary_raw, primary_null = run_scope(
        'primary_LH49', primary_mask, dthi_values, external_values, spins
    )
    primary = add_primary_corrections(primary_raw, primary_null)

    sensitivity_raw, sensitivity_null = run_scope(
        'high_support_LH46', high_mask, dthi_values, external_values, spins
    )
    sensitivity = add_sensitivity_corrections(sensitivity_raw, primary)

    if not np.isfinite(
        primary[
            [
                'spearman_rho', 'p_spin_two_sided', 'q_BH_global160',
                'q_BH_family80', 'p_maxT_global160', 'p_maxT_family80',
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValidationError('Primary results contain non-finite inferential values.')
    if not np.isfinite(
        sensitivity[
            [
                'spearman_rho', 'p_spin_two_sided', 'q_BH_sensitivity160',
                'primary_LH49_rho', 'delta_rho_high_support_minus_primary',
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValidationError('Sensitivity results contain non-finite inferential values.')

    atomic_tsv(primary, PRIMARY_OUT)
    atomic_tsv(sensitivity, SENSITIVITY_OUT)

    significant = primary[
        primary['significant_BH_global160']
        | primary['significant_maxT_global160']
        | primary['significant_BH_family80']
        | primary['significant_maxT_family80']
    ].copy()
    atomic_tsv(significant, SIGNIFICANT_OUT)

    test_ids = np.asarray(
        [
            f'{row.DTHI_map}__{row.external_map}'
            for row in primary_raw.itertuples(index=False)
        ],
        dtype='U160',
    )
    np.savez_compressed(
        NULL_OUT,
        primary_null=primary_null,
        sensitivity_null=sensitivity_null,
        test_ids=test_ids,
        spin_unique_counts=unique_counts.astype(np.int16),
    )

    audit: list[dict[str, Any]] = []
    add_audit(audit, 'upstream', 'Phase8D2C1_ready', True, True)
    add_audit(audit, 'inputs', 'DTHI_matrix', f'{dthi_values.shape[0]}x{dthi_values.shape[1]}', dthi_values.shape == (50, 10), sha256(DTHI_FILE))
    add_audit(audit, 'inputs', 'external_matrix', f'{external_values.shape[0]}x{external_values.shape[1]}', external_values.shape == (50, 16), sha256(EXTERNAL_FILE))
    add_audit(audit, 'masks', 'primary_LH49', int(primary_mask.sum()), int(primary_mask.sum()) == 49)
    add_audit(audit, 'masks', 'high_support_LH46', int(high_mask.sum()), int(high_mask.sum()) == 46)
    add_audit(audit, 'spatial_null', 'rotations', spins.shape[0], spins.shape[0] == N_ROTATIONS)
    add_audit(
        audit,
        'spatial_null',
        'unique_assignments_per_rotation',
        f'min={unique_counts.min()};median={np.median(unique_counts):.1f};max={unique_counts.max()}',
        int(unique_counts.min()) >= 1,
        'frozen_Phase8A3_nonbijective_nearest-neighbour_resampling',
    )
    add_audit(audit, 'spatial_null', 'identity_rotations', identity_count, True, 'reported_not_used_as_exclusion_criterion')
    add_audit(audit, 'primary', 'tests_completed', len(primary), len(primary) == 160)
    add_audit(audit, 'primary', 'global_BH_applied', 160, primary['q_BH_global160'].notna().all())
    add_audit(audit, 'primary', 'family_BH_applied', '80+80', primary['q_BH_family80'].notna().all())
    add_audit(audit, 'primary', 'global_maxT_applied', 160, primary['p_maxT_global160'].notna().all())
    add_audit(audit, 'primary', 'family_maxT_applied', '80+80', primary['p_maxT_family80'].notna().all())
    add_audit(audit, 'sensitivity', 'tests_completed', len(sensitivity), len(sensitivity) == 160)
    add_audit(audit, 'outputs', 'all_inferential_values_finite', True, True)

    ready = (
        len(primary) == 160
        and len(sensitivity) == 160
        and primary['q_BH_global160'].notna().all()
        and primary['p_maxT_global160'].notna().all()
        and sensitivity['q_BH_sensitivity160'].notna().all()
    )

    summary = pd.DataFrame(
        [
            {
                'Phase8D2C1_status_confirmed': True,
                'primary_tests_completed': len(primary),
                'sensitivity_tests_completed': len(sensitivity),
                'spatial_null_rotations': spins.shape[0],
                'primary_LH_parcels': int(primary_mask.sum()),
                'high_support_LH_parcels': int(high_mask.sum()),
                'primary_BH_global_significant': int(primary['significant_BH_global160'].sum()),
                'primary_maxT_global_significant': int(primary['significant_maxT_global160'].sum()),
                'primary_BH_family_significant': int(primary['significant_BH_family80'].sum()),
                'primary_maxT_family_significant': int(primary['significant_maxT_family80'].sum()),
                'sensitivity_BH_significant': int(sensitivity['significant_BH_sensitivity160'].sum()),
                'all_inferential_values_finite': True,
                'ready_for_phase8D2C2B_LODO': bool(ready),
                'Phase8D2C2A_status': 'completed' if ready else 'failed_validation',
                'python_version': platform.python_version(),
                'numpy_version': np.__version__,
                'pandas_version': pd.__version__,
                'scipy_version': scipy.__version__,
                'statsmodels_version': statsmodels.__version__,
            }
        ]
    )

    atomic_tsv(pd.DataFrame(audit), AUDIT_OUT)
    atomic_tsv(summary, SUMMARY_OUT)

    hash_paths = [
        PRIMARY_OUT,
        SENSITIVITY_OUT,
        SIGNIFICANT_OUT,
        NULL_OUT,
        AUDIT_OUT,
        SUMMARY_OUT,
    ]
    hashes = pd.DataFrame(
        [
            {
                'relative_path': str(path.relative_to(PROJECT)),
                'size_bytes': path.stat().st_size,
                'SHA256': sha256(path),
            }
            for path in hash_paths
        ]
    )
    atomic_tsv(hashes, HASH_OUT)

    print('\n===== PHASE 8D2C2A COMPLETION =====')
    print(summary.to_string(index=False))
    print('\n===== TOP PRIMARY RESULTS =====')
    print(
        primary[
            [
                'DTHI_map', 'external_map', 'external_family', 'spearman_rho',
                'p_spin_two_sided', 'q_BH_global160', 'q_BH_family80',
                'p_maxT_global160', 'p_maxT_family80',
            ]
        ].head(20).to_string(index=False)
    )
    print('\n===== SIGNIFICANT PRIMARY RESULTS =====')
    if significant.empty:
        print('No primary results passed the frozen BH/maxT thresholds.')
    else:
        print(
            significant[
                [
                    'DTHI_map', 'external_map', 'spearman_rho',
                    'q_BH_global160', 'q_BH_family80',
                    'p_maxT_global160', 'p_maxT_family80',
                ]
            ].to_string(index=False)
        )
    print('\n===== AUDIT =====')
    print(pd.DataFrame(audit).to_string(index=False))
    print('\n===== CANONICAL OUTPUT HASHES =====')
    print(hashes.to_string(index=False))

    if not ready:
        raise RuntimeError('Phase 8D2C2A validation failed.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(
            f'Phase 8D2C2A failed: {type(exc).__name__}: {exc}',
            file=sys.stderr,
            flush=True,
        )
        raise
