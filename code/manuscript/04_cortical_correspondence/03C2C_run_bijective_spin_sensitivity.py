#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests


PROJECT = Path(".")
TABLES = PROJECT / "07_tables/main_tables/phase8"
PROC8D = PROJECT / "03_processed_data/functional_imaging/phase8D"
OUT = PROC8D / "phase8D2C"
ENIGMA = PROJECT / "05_external_resources/phase8D/ENIGMA"

C2B_SUMMARY = TABLES / "phase8D2C2B_completion_summary.tsv"
DTHI_MATRIX = OUT / "phase8D2C1_DTHI_Schaefer100_LH50_matrix.tsv"
MASKS = OUT / "phase8D2C1_analysis_masks_LH50.tsv"
EXTERNAL_MATRIX = (
    PROC8D
    / "phase8D2B2_maps/matrices"
    / "phase8D2B2_combined_Schaefer100_LH50_matrix.tsv"
)
PRIMARY_OLD = OUT / "phase8D2C2A_primary_spatial_correspondence.tsv"
SENS_OLD = OUT / "phase8D2C2A_high_support_sensitivity.tsv"
SUPPORTED_LODO = OUT / "phase8D2C2B_primary_supported_LODO_summary.tsv"

SPHERE_LH = (
    ENIGMA
    / "enigmatoolbox/datasets/surfaces/fsa5_sphere_lh.gii"
)
PARCELLATION = (
    ENIGMA
    / "enigmatoolbox/datasets/parcellations/schaefer_100_fsa5.csv"
)

SPINS_OUT = OUT / "phase8D2C2C_Schaefer100_LH_bijective_spins_10000.npy"
CENTROIDS_OUT = OUT / "phase8D2C2C_Schaefer100_LH_spherical_centroids.tsv"
PRIMARY_OUT = OUT / "phase8D2C2C_bijective_primary_spatial_correspondence.tsv"
SENS_OUT = OUT / "phase8D2C2C_bijective_high_support_sensitivity.tsv"
COMPARE_OUT = OUT / "phase8D2C2C_bijective_vs_nonbijective_comparison.tsv"
SUPPORTED_OUT = OUT / "phase8D2C2C_primary_supported_bijective_summary.tsv"
NULL_OUT = OUT / "phase8D2C2C_bijective_null_distributions.npz"
AUDIT_OUT = TABLES / "phase8D2C2C_bijective_spin_audit.tsv"
SUMMARY_OUT = TABLES / "phase8D2C2C_completion_summary.tsv"
HASH_OUT = TABLES / "phase8D2C2C_canonical_output_hashes.tsv"

N_ROTATIONS = 10000
SEED = 20260723

DTHI_COLUMNS = [
    "patterning_arealization",
    "progenitor_radial_glia",
    "neurogenesis_migration_layering",
    "axon_guidance_neurite_outgrowth",
    "synaptic_assembly_receptor_trafficking",
    "astrocyte_maturation_metabolic_support",
    "oligodendrocyte_myelination",
    "activity_dependent_plasticity",
    "synaptic_membrane_structural_candidates",
    "preliminary_maturation_balance",
]


class ValidationError(RuntimeError):
    pass


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, sep="\t", index=False)
    temp.replace(path)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise ValidationError(f"Missing required file: {path}")


def require_completed(path: Path, status: str, ready: str | None = None) -> None:
    frame = pd.read_csv(path, sep="\t")
    if len(frame) != 1 or status not in frame.columns:
        raise ValidationError(f"Invalid completion summary: {path}")
    if str(frame.loc[0, status]).strip() != "completed":
        raise ValidationError(f"{status} is not completed: {path}")
    if ready is not None:
        if ready not in frame.columns or not parse_bool(frame.loc[0, ready]):
            raise ValidationError(f"{ready} is not true: {path}")


def parcel_column(frame: pd.DataFrame) -> str:
    matches = [
        column
        for column in ["parcel_id", "parcel_index_full", "parcel"]
        if column in frame.columns
    ]
    if len(matches) != 1:
        raise ValidationError(
            f"Expected one parcel ID column, found {matches}; "
            f"columns={frame.columns.tolist()}"
        )
    return matches[0]


def load_parcellation(path: Path) -> np.ndarray:
    attempts: list[tuple[str, dict[str, Any]]] = [
        ("comma", {"delimiter": ","}),
        ("whitespace", {}),
    ]
    errors: list[str] = []
    for name, kwargs in attempts:
        try:
            labels = np.loadtxt(path, dtype=float, **kwargs).reshape(-1)
            if labels.size == 20484:
                if not np.all(np.isfinite(labels)):
                    raise ValueError("non-finite labels")
                if not np.allclose(labels, np.rint(labels), rtol=0, atol=0):
                    raise ValueError("non-integer labels")
                return np.rint(labels).astype(np.int64)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise ValidationError(
        f"Could not load 20,484-vertex Schaefer100 parcellation: {path}; "
        f"attempts={errors}"
    )


def load_lh_centroids() -> tuple[np.ndarray, pd.DataFrame]:
    sphere_img = nib.load(str(SPHERE_LH))

    # A GIFTI surface contains both an N x 3 POINTSET array holding
    # vertex coordinates and an M x 3 TRIANGLE array holding faces.
    # Select coordinates by GIFTI intent rather than by array shape.
    nifti_intent_pointset = 1008
    pointset_arrays = [
        np.asarray(darray.data)
        for darray in sphere_img.darrays
        if int(darray.intent) == nifti_intent_pointset
    ]

    if len(pointset_arrays) != 1:
        intent_shapes = [
            (
                int(darray.intent),
                tuple(np.asarray(darray.data).shape),
            )
            for darray in sphere_img.darrays
        ]
        raise ValidationError(
            f"Expected one GIFTI POINTSET array in {SPHERE_LH}; "
            f"found intent/shape pairs={intent_shapes}"
        )

    vertices = pointset_arrays[0].astype(float)
    if vertices.shape != (10242, 3):
        raise ValidationError(
            f"Expected fsaverage5 LH sphere shape 10242x3; found {vertices.shape}"
        )

    labels = load_parcellation(PARCELLATION)
    labels_lh = labels[:10242]
    labels_rh = labels[10242:]
    lh_unique = set(np.unique(labels_lh).tolist())
    rh_unique = set(np.unique(labels_rh).tolist())
    expected_lh = set(range(0, 51))
    expected_rh = {0, *range(51, 101)}
    if lh_unique != expected_lh:
        raise ValidationError(
            f"Unexpected LH Schaefer labels: {sorted(lh_unique)}"
        )
    if rh_unique != expected_rh:
        raise ValidationError(
            f"Unexpected RH Schaefer labels: {sorted(rh_unique)}"
        )

    centroids = np.empty((50, 3), dtype=float)
    records: list[dict[str, Any]] = []
    for parcel_id in range(1, 51):
        mask = labels_lh == parcel_id
        count = int(mask.sum())
        if count <= 0:
            raise ValidationError(f"Empty LH Schaefer parcel {parcel_id}.")
        vector = vertices[mask].mean(axis=0)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            raise ValidationError(
                f"Invalid spherical centroid for parcel {parcel_id}."
            )
        vector = vector / norm
        centroids[parcel_id - 1] = vector
        records.append(
            {
                "parcel_id": parcel_id,
                "vertex_count": count,
                "sphere_x": vector[0],
                "sphere_y": vector[1],
                "sphere_z": vector[2],
                "unit_norm": float(np.linalg.norm(vector)),
            }
        )
    if not np.isfinite(centroids).all():
        raise ValidationError("Spherical centroids contain non-finite values.")
    return centroids, pd.DataFrame(records)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = q @ np.diag(signs)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def enigma_greedy_bijection(
    reference: np.ndarray,
    rotated: np.ndarray,
) -> np.ndarray:
    distances = np.linalg.norm(
        reference[:, None, :] - rotated[None, :, :],
        axis=2,
    )
    active_rows = list(range(reference.shape[0]))
    active_cols = list(range(rotated.shape[0]))
    mapping = np.full(reference.shape[0], -1, dtype=np.int64)

    while active_rows:
        sub = distances[np.ix_(active_rows, active_cols)]
        nearest = sub.min(axis=1)
        row_pos = int(np.argmax(nearest))
        row = active_rows[row_pos]
        col_pos = int(np.argmin(sub[row_pos]))
        col = active_cols[col_pos]
        mapping[row] = col
        active_rows.pop(row_pos)
        active_cols.pop(col_pos)

    if not np.array_equal(np.sort(mapping), np.arange(reference.shape[0])):
        raise ValidationError("Greedy rotation assignment was not bijective.")
    return mapping


def generate_bijective_spins(centroids: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    spins = np.empty((N_ROTATIONS, 50), dtype=np.int16)
    accepted = 0
    identity_rejections = 0
    started = time.time()

    while accepted < N_ROTATIONS:
        rotation = random_rotation(rng)
        rotated = centroids @ rotation
        mapping = enigma_greedy_bijection(centroids, rotated)
        if np.array_equal(mapping, np.arange(50)):
            identity_rejections += 1
            continue
        spins[accepted] = mapping.astype(np.int16)
        accepted += 1
        if accepted % 1000 == 0:
            elapsed = (time.time() - started) / 60.0
            print(
                f"  generated {accepted}/{N_ROTATIONS} exact bijections "
                f"({elapsed:.1f} min)",
                flush=True,
            )

    print(f"Identity mappings rejected: {identity_rejections}", flush=True)
    return spins


def validate_spins(spins: np.ndarray) -> dict[str, Any]:
    if spins.shape != (N_ROTATIONS, 50):
        raise ValidationError(f"Unexpected bijective spin shape: {spins.shape}")
    if int(spins.min()) < 0 or int(spins.max()) > 49:
        raise ValidationError(
            f"Spin index range must be 0-49; found {spins.min()}-{spins.max()}."
        )
    unique_counts = np.asarray(
        [np.unique(row).size for row in spins],
        dtype=int,
    )
    identities = int(
        np.sum(np.all(spins == np.arange(50)[None, :], axis=1))
    )
    if not np.all(unique_counts == 50):
        raise ValidationError(
            f"Not all rotations are bijective: "
            f"minimum unique assignments={unique_counts.min()}"
        )
    if identities != 0:
        raise ValidationError(f"Identity rotations present: {identities}")
    unique_permutations = int(np.unique(spins, axis=0).shape[0])
    return {
        "minimum_unique_assignments": int(unique_counts.min()),
        "median_unique_assignments": float(np.median(unique_counts)),
        "maximum_unique_assignments": int(unique_counts.max()),
        "identity_rotations": identities,
        "unique_permutations": unique_permutations,
    }


def zscore_rows(array: np.ndarray) -> np.ndarray:
    means = array.mean(axis=1, keepdims=True)
    centered = array - means
    scales = np.sqrt(np.sum(centered * centered, axis=1, keepdims=True))
    if np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValidationError("At least one ranked null vector is constant.")
    return centered / scales


def ranked_standardized_vector(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, method="average")
    return zscore_rows(ranks[None, :])[0]


def ranked_standardized_rows(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, method="average", axis=1)
    return zscore_rows(ranks)


def empirical_two_sided(observed: float, nulls: np.ndarray) -> float:
    return float(
        (1 + np.sum(np.abs(nulls) >= abs(observed)))
        / (nulls.size + 1)
    )


def add_primary_corrections(
    frame: pd.DataFrame,
    null_matrix: np.ndarray,
) -> pd.DataFrame:
    result = frame.copy()
    pvalues = result["p_spin_two_sided"].to_numpy(dtype=float)
    result["q_BH_global160"] = multipletests(
        pvalues, method="fdr_bh"
    )[1]

    result["q_BH_family80"] = np.nan
    result["p_maxT_family80"] = np.nan
    global_max = np.max(np.abs(null_matrix), axis=1)
    result["p_maxT_global160"] = [
        (1 + np.sum(global_max >= abs(value))) / (len(global_max) + 1)
        for value in result["spearman_rho"].to_numpy(dtype=float)
    ]

    for family in ["cognitive", "clinical"]:
        mask = result["external_family"].eq(family).to_numpy()
        result.loc[mask, "q_BH_family80"] = multipletests(
            pvalues[mask], method="fdr_bh"
        )[1]
        family_max = np.max(np.abs(null_matrix[:, mask]), axis=1)
        result.loc[mask, "p_maxT_family80"] = [
            (1 + np.sum(family_max >= abs(value)))
            / (len(family_max) + 1)
            for value in result.loc[mask, "spearman_rho"].to_numpy(dtype=float)
        ]

    result["BH_global160_significant"] = result["q_BH_global160"] < 0.05
    result["BH_family80_significant"] = result["q_BH_family80"] < 0.05
    result["maxT_global160_significant"] = result["p_maxT_global160"] < 0.05
    result["maxT_family80_significant"] = result["p_maxT_family80"] < 0.05
    return result


def add_sensitivity_correction(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    pvalues = result["p_spin_two_sided"].to_numpy(dtype=float)
    result["q_BH_sensitivity160"] = multipletests(
        pvalues, method="fdr_bh"
    )[1]
    result["BH_sensitivity160_significant"] = (
        result["q_BH_sensitivity160"] < 0.05
    )
    return result


def run_scope(
    dthi: pd.DataFrame,
    external: pd.DataFrame,
    dthi_columns: list[str],
    external_columns: list[str],
    mask: np.ndarray,
    spins: np.ndarray,
    scope: str,
    correction_mode: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    null_columns: list[np.ndarray] = []
    total = len(dthi_columns) * len(external_columns)
    completed = 0

    dthi_ranked = np.vstack(
        [
            ranked_standardized_vector(
                dthi[column].to_numpy(dtype=float)[mask]
            )
            for column in dthi_columns
        ]
    )

    for external_map in external_columns:
        family = (
            "cognitive"
            if external_map.startswith("cognitive_")
            else "clinical"
        )
        y = external[external_map].to_numpy(dtype=float)
        observed_y = ranked_standardized_vector(y[mask])
        observed_values = dthi_ranked @ observed_y

        rotated = y[spins][:, mask]
        rotated_ranked = ranked_standardized_rows(rotated)
        null_block = rotated_ranked @ dthi_ranked.T

        if null_block.shape != (N_ROTATIONS, len(dthi_columns)):
            raise ValidationError(
                f"Unexpected null block shape for {external_map}: "
                f"{null_block.shape}"
            )

        for dthi_index, dthi_map in enumerate(dthi_columns):
            observed = float(observed_values[dthi_index])
            nulls = null_block[:, dthi_index].astype(np.float32)
            pvalue = empirical_two_sided(observed, nulls)
            rows.append(
                {
                    "analysis_scope": scope,
                    "DTHI_map": dthi_map,
                    "external_map": external_map,
                    "external_family": family,
                    "parcels": int(mask.sum()),
                    "spearman_rho": observed,
                    "p_spin_two_sided": pvalue,
                    "rotations": spins.shape[0],
                    "spin_assignment": "ENIGMA_style_greedy_exact_bijection",
                    "rotation_seed": SEED,
                }
            )
            null_columns.append(nulls)
            completed += 1

        print(
            f"  {scope}: external map {external_map} "
            f"({completed}/{total} tests)",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    null_matrix = np.column_stack(null_columns)
    if null_matrix.shape != (N_ROTATIONS, total):
        raise ValidationError(
            f"Unexpected null matrix shape for {scope}: {null_matrix.shape}"
        )
    if correction_mode == "primary":
        frame = add_primary_corrections(frame, null_matrix)
    elif correction_mode == "sensitivity":
        frame = add_sensitivity_correction(frame)
    else:
        raise ValidationError(
            f"Unknown correction mode: {correction_mode}"
        )
    return frame, null_matrix


def merge_comparison(
    old: pd.DataFrame,
    new: pd.DataFrame,
    scope: str,
    correction_mode: str,
) -> pd.DataFrame:
    keys = ["DTHI_map", "external_map", "external_family"]

    if correction_mode == "primary":
        metrics = [
            "spearman_rho",
            "p_spin_two_sided",
            "q_BH_global160",
            "q_BH_family80",
            "p_maxT_global160",
            "p_maxT_family80",
        ]
        new_extra = [
            "BH_global160_significant",
            "BH_family80_significant",
            "maxT_global160_significant",
            "maxT_family80_significant",
        ]
    elif correction_mode == "sensitivity":
        metrics = [
            "spearman_rho",
            "p_spin_two_sided",
            "q_BH_sensitivity160",
        ]
        new_extra = ["BH_sensitivity160_significant"]
    else:
        raise ValidationError(
            f"Unknown comparison correction mode: {correction_mode}"
        )

    old_available = [
        column for column in keys + metrics
        if column in old.columns
    ]
    old_sub = old[old_available].copy()
    old_sub = old_sub.rename(
        columns={
            column: f"nonbijective_{column}"
            for column in old_available
            if column not in keys
        }
    )

    new_columns = keys + metrics + new_extra
    missing_new = [column for column in new_columns if column not in new.columns]
    if missing_new:
        raise ValidationError(
            f"Missing bijective comparison columns: {missing_new}"
        )
    new_sub = new[new_columns].copy()
    new_sub = new_sub.rename(
        columns={
            column: f"bijective_{column}"
            for column in new_columns
            if column not in keys
        }
    )

    merged = old_sub.merge(
        new_sub,
        on=keys,
        how="outer",
        validate="one_to_one",
    )
    merged.insert(0, "analysis_scope", scope)
    for metric in metrics:
        left = f"nonbijective_{metric}"
        right = f"bijective_{metric}"
        if left in merged.columns and right in merged.columns:
            merged[f"delta_bijective_minus_nonbijective_{metric}"] = (
                merged[right] - merged[left]
            )
    return merged


def main() -> None:
    started = time.time()
    required = [
        C2B_SUMMARY,
        DTHI_MATRIX,
        MASKS,
        EXTERNAL_MATRIX,
        PRIMARY_OLD,
        SENS_OLD,
        SUPPORTED_LODO,
        SPHERE_LH,
        PARCELLATION,
    ]
    for path in required:
        require_file(path)
    require_completed(
        C2B_SUMMARY,
        "Phase8D2C2B_status",
        "ready_for_phase8D2C2C_bijective_spin_sensitivity",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    dthi = pd.read_csv(DTHI_MATRIX, sep="\t")
    masks = pd.read_csv(MASKS, sep="\t")
    external = pd.read_csv(EXTERNAL_MATRIX, sep="\t")
    old_primary = pd.read_csv(PRIMARY_OLD, sep="\t")
    old_sens = pd.read_csv(SENS_OLD, sep="\t")
    supported_lodo = pd.read_csv(SUPPORTED_LODO, sep="\t")

    dthi_parcel = parcel_column(dthi)
    external_parcel = parcel_column(external)
    mask_parcel = parcel_column(masks)
    for frame, column, name in [
        (dthi, dthi_parcel, "DTHI"),
        (external, external_parcel, "external"),
        (masks, mask_parcel, "mask"),
    ]:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=int)
        if not np.array_equal(values, np.arange(1, 51)):
            raise ValidationError(
                f"{name} rows are not ordered as LH parcel IDs 1-50."
            )

    external_columns = [
        column
        for column in external.columns
        if column.startswith("cognitive_") or column.startswith("clinical_")
    ]
    if len(external_columns) != 16:
        raise ValidationError(
            f"Expected 16 external maps; found {len(external_columns)}."
        )
    missing_dthi = [column for column in DTHI_COLUMNS if column not in dthi.columns]
    if missing_dthi:
        raise ValidationError(f"Missing DTHI columns: {missing_dthi}")

    primary_mask = masks["primary_LH49"].map(parse_bool).to_numpy(dtype=bool)
    high_mask = masks["high_support_LH46"].map(parse_bool).to_numpy(dtype=bool)
    if int(primary_mask.sum()) != 49 or int(high_mask.sum()) != 46:
        raise ValidationError(
            f"Expected LH49/LH46 masks; found "
            f"{primary_mask.sum()}/{high_mask.sum()}."
        )

    print("===== BUILDING EXACT BIJECTIVE LH SPINS =====", flush=True)
    centroids, centroid_frame = load_lh_centroids()
    atomic_tsv(centroid_frame, CENTROIDS_OUT)
    spins = generate_bijective_spins(centroids)
    spin_qc = validate_spins(spins)
    np.save(SPINS_OUT, spins)

    print("\n===== RUNNING BIJECTIVE PRIMARY LH49 TESTS =====", flush=True)
    primary, primary_nulls = run_scope(
        dthi,
        external,
        DTHI_COLUMNS,
        external_columns,
        primary_mask,
        spins,
        "bijective_primary_LH49",
        "primary",
    )
    atomic_tsv(primary, PRIMARY_OUT)

    print("\n===== RUNNING BIJECTIVE HIGH-SUPPORT LH46 TESTS =====", flush=True)
    sensitivity, sensitivity_nulls = run_scope(
        dthi,
        external,
        DTHI_COLUMNS,
        external_columns,
        high_mask,
        spins,
        "bijective_high_support_LH46",
        "sensitivity",
    )
    atomic_tsv(sensitivity, SENS_OUT)

    np.savez_compressed(
        NULL_OUT,
        primary_nulls=primary_nulls,
        sensitivity_nulls=sensitivity_nulls,
        DTHI_maps=np.asarray(DTHI_COLUMNS, dtype="U"),
        external_maps=np.asarray(external_columns, dtype="U"),
        rotation_seed=np.asarray([SEED], dtype=np.int64),
    )

    primary_comparison = merge_comparison(
        old_primary,
        primary,
        "primary_LH49",
        "primary",
    )
    sensitivity_comparison = merge_comparison(
        old_sens,
        sensitivity,
        "high_support_LH46",
        "sensitivity",
    )
    comparison = pd.concat(
        [primary_comparison, sensitivity_comparison],
        ignore_index=True,
        sort=False,
    )
    atomic_tsv(comparison, COMPARE_OUT)

    supported_keys = supported_lodo[["DTHI_map", "external_map"]].drop_duplicates()
    supported = supported_keys.merge(
        primary,
        on=["DTHI_map", "external_map"],
        how="left",
        validate="one_to_one",
    ).merge(
        supported_lodo,
        on=["DTHI_map", "external_map"],
        how="left",
        suffixes=("_bijective", "_LODO"),
        validate="one_to_one",
    )
    if len(supported) != 4:
        raise ValidationError(
            f"Expected four primary-supported pairs; found {len(supported)}."
        )
    supported["bijective_global_BH_retained"] = (
        supported["q_BH_global160"] < 0.05
    )
    supported["bijective_family_maxT_retained"] = (
        supported["p_maxT_family80"] < 0.05
    )
    supported["LODO_all_five_direction_concordant"] = (
        supported["sign_concordant_omissions"] == 5
    )
    atomic_tsv(supported, SUPPORTED_OUT)

    primary_inferential_columns = [
        "spearman_rho",
        "p_spin_two_sided",
        "q_BH_global160",
        "q_BH_family80",
        "p_maxT_global160",
        "p_maxT_family80",
    ]
    sensitivity_inferential_columns = [
        "spearman_rho",
        "p_spin_two_sided",
        "q_BH_sensitivity160",
    ]
    all_finite = bool(
        np.isfinite(
            primary[primary_inferential_columns].to_numpy(dtype=float)
        ).all()
        and np.isfinite(
            sensitivity[sensitivity_inferential_columns].to_numpy(dtype=float)
        ).all()
    )

    audit = pd.DataFrame(
        [
            {
                "section": "upstream",
                "item": "Phase8D2C2B_ready",
                "value": True,
                "passed": True,
                "detail": "",
            },
            {
                "section": "geometry",
                "item": "fsaverage5_LH_sphere_vertices",
                "value": 10242,
                "passed": centroids.shape == (50, 3),
                "detail": str(SPHERE_LH.relative_to(PROJECT)),
            },
            {
                "section": "geometry",
                "item": "Schaefer100_LH_spherical_centroids",
                "value": len(centroid_frame),
                "passed": len(centroid_frame) == 50,
                "detail": str(PARCELLATION.relative_to(PROJECT)),
            },
            {
                "section": "spatial_null",
                "item": "bijective_rotations",
                "value": spins.shape[0],
                "passed": spins.shape == (N_ROTATIONS, 50),
                "detail": f"seed={SEED}",
            },
            {
                "section": "spatial_null",
                "item": "unique_assignments_per_rotation",
                "value": (
                    f"min={spin_qc['minimum_unique_assignments']};"
                    f"median={spin_qc['median_unique_assignments']:.1f};"
                    f"max={spin_qc['maximum_unique_assignments']}"
                ),
                "passed": spin_qc["minimum_unique_assignments"] == 50,
                "detail": "ENIGMA_style_greedy_exact_bijection",
            },
            {
                "section": "spatial_null",
                "item": "identity_rotations",
                "value": spin_qc["identity_rotations"],
                "passed": spin_qc["identity_rotations"] == 0,
                "detail": "",
            },
            {
                "section": "spatial_null",
                "item": "unique_permutation_rows",
                "value": spin_qc["unique_permutations"],
                "passed": spin_qc["unique_permutations"] > 0,
                "detail": "duplicate_random_permutations_allowed",
            },
            {
                "section": "inference",
                "item": "primary_tests_completed",
                "value": len(primary),
                "passed": len(primary) == 160,
                "detail": "",
            },
            {
                "section": "inference",
                "item": "sensitivity_tests_completed",
                "value": len(sensitivity),
                "passed": len(sensitivity) == 160,
                "detail": "",
            },
            {
                "section": "inference",
                "item": "all_inferential_values_finite",
                "value": all_finite,
                "passed": all_finite,
                "detail": "",
            },
            {
                "section": "interpretation",
                "item": "analysis_role",
                "value": "sensitivity_only",
                "passed": True,
                "detail": (
                    "exact_bijective_rotation_check_for_original_"
                    "nearest_neighbour_collision_resource"
                ),
            },
        ]
    )
    atomic_tsv(audit, AUDIT_OUT)

    ready = bool(
        len(primary) == 160
        and len(sensitivity) == 160
        and len(supported) == 4
        and all_finite
        and spin_qc["minimum_unique_assignments"] == 50
        and spin_qc["identity_rotations"] == 0
    )
    summary = pd.DataFrame(
        [
            {
                "Phase8D2C2B_status_confirmed": True,
                "bijective_spin_rotations": N_ROTATIONS,
                "bijective_spin_parcels": 50,
                "exact_bijective_permutations": int(
                    spin_qc["minimum_unique_assignments"] == 50
                )
                * N_ROTATIONS,
                "primary_tests_completed": len(primary),
                "sensitivity_tests_completed": len(sensitivity),
                "primary_supported_pairs_reviewed": len(supported),
                "supported_pairs_global_BH_retained": int(
                    supported["bijective_global_BH_retained"].sum()
                ),
                "supported_pairs_family_maxT_retained": int(
                    supported["bijective_family_maxT_retained"].sum()
                ),
                "all_inferential_values_finite": all_finite,
                "ready_for_phase8D2C3_evidence_synthesis": ready,
                "Phase8D2C2C_status": (
                    "completed" if ready else "failed_validation"
                ),
                "runtime_minutes": (time.time() - started) / 60.0,
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "nibabel_version": nib.__version__,
            }
        ]
    )
    atomic_tsv(summary, SUMMARY_OUT)

    hash_paths = [
        SPINS_OUT,
        CENTROIDS_OUT,
        PRIMARY_OUT,
        SENS_OUT,
        COMPARE_OUT,
        SUPPORTED_OUT,
        NULL_OUT,
        AUDIT_OUT,
        SUMMARY_OUT,
    ]
    hashes = pd.DataFrame(
        [
            {
                "relative_path": str(path.relative_to(PROJECT)),
                "size_bytes": path.stat().st_size,
                "SHA256": sha256(path),
            }
            for path in hash_paths
        ]
    )
    atomic_tsv(hashes, HASH_OUT)

    print("\n===== PHASE 8D2C2C COMPLETION =====")
    print(summary.to_string(index=False))
    print("\n===== PRIMARY-SUPPORTED PAIRS UNDER BIJECTIVE SPINS =====")
    display_columns = [
        "DTHI_map",
        "external_map",
        "spearman_rho",
        "p_spin_two_sided",
        "q_BH_global160",
        "p_maxT_global160",
        "p_maxT_family80",
        "bijective_global_BH_retained",
        "bijective_family_maxT_retained",
        "LODO_all_five_direction_concordant",
    ]
    print(supported[display_columns].to_string(index=False))
    print("\n===== BIJECTIVE SPIN AUDIT =====")
    print(audit.to_string(index=False))
    print("\n===== CANONICAL OUTPUT HASHES =====")
    print(hashes.to_string(index=False))

    if not ready:
        raise RuntimeError("Phase 8D2C2C validation failed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"Phase 8D2C2C failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise
