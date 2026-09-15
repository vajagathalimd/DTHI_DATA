#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT = Path(".")
TABLES = PROJECT / "07_tables/main_tables/phase8"
PROC8D = PROJECT / "03_processed_data/functional_imaging/phase8D"
PROC3C = PROJECT / "03_processed_data/spatial_hierarchy/phase3C"
OUT = PROC8D / "phase8D2C"

C1_SUMMARY = TABLES / "phase8D2C1_completion_summary.tsv"
C2A_SUMMARY = TABLES / "phase8D2C2A_completion_summary.tsv"
PRIMARY_RESULTS = OUT / "phase8D2C2A_primary_spatial_correspondence.tsv"
SIGNIFICANT_RESULTS = OUT / "phase8D2C2A_primary_significant_results.tsv"
DTHI_BALANCED = OUT / "phase8D2C1_DTHI_Schaefer100_LH50_matrix.tsv"
DTHI_DONOR = PROC3C / "AHBA_donor_Schaefer100_DTHI_scores.tsv"
EXTERNAL = PROC8D / "phase8D2B2_maps/matrices/phase8D2B2_combined_Schaefer100_LH50_matrix.tsv"
MASKS = OUT / "phase8D2C1_analysis_masks_LH50.tsv"

LODO_DETAIL_OUT = OUT / "phase8D2C2B_LODO_correlations.tsv"
LODO_SUMMARY_OUT = OUT / "phase8D2C2B_LODO_summary.tsv"
LODO_SIG_OUT = OUT / "phase8D2C2B_primary_supported_LODO_summary.tsv"
MAP_STABILITY_OUT = OUT / "phase8D2C2B_LODO_DTHI_map_stability.tsv"
RECON_AUDIT_OUT = TABLES / "phase8D2C2B_pooled_reconstruction_audit.tsv"
AUDIT_OUT = TABLES / "phase8D2C2B_LODO_audit.tsv"
SUMMARY_OUT = TABLES / "phase8D2C2B_completion_summary.tsv"
HASH_OUT = TABLES / "phase8D2C2B_canonical_output_hashes.tsv"

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

def require_completed(path: Path, status_col: str, ready_col: str | None = None) -> None:
    if not path.is_file():
        raise ValidationError(f"Missing completion summary: {path}")
    frame = pd.read_csv(path, sep="\t")
    if len(frame) != 1 or status_col not in frame.columns:
        raise ValidationError(f"Invalid completion summary: {path}")
    if str(frame.loc[0, status_col]).strip() != "completed":
        raise ValidationError(f"{status_col} is not completed.")
    if ready_col is not None:
        if ready_col not in frame.columns or not parse_bool(frame.loc[0, ready_col]):
            raise ValidationError(f"{ready_col} is not true.")

def parcel_column(frame: pd.DataFrame) -> str:
    matches = [c for c in ["parcel_id", "parcel_index_full", "parcel"] if c in frame.columns]
    if len(matches) != 1:
        raise ValidationError(f"Expected one parcel column, found {matches}.")
    return matches[0]

def donor_column(frame: pd.DataFrame) -> str:
    matches = [c for c in ["donor_id", "donor", "donor_name"] if c in frame.columns]
    if len(matches) != 1:
        raise ValidationError(f"Expected one donor column, found {matches}.")
    return matches[0]

def finite_pair(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = mask & np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]

def correlation_pair(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[int, float, float]:
    xv, yv = finite_pair(x, y, mask)
    if len(xv) < 10:
        return len(xv), np.nan, np.nan
    if np.std(xv) <= 0 or np.std(yv) <= 0:
        return len(xv), np.nan, np.nan
    rho = float(spearmanr(xv, yv).statistic)
    r = float(pearsonr(xv, yv).statistic)
    return len(xv), rho, r

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    require_completed(C1_SUMMARY, "Phase8D2C1_status", "ready_for_phase8D2C2")
    require_completed(C2A_SUMMARY, "Phase8D2C2A_status", "ready_for_phase8D2C2B_LODO")

    for path in [PRIMARY_RESULTS, DTHI_BALANCED, DTHI_DONOR, EXTERNAL, MASKS]:
        if not path.is_file():
            raise ValidationError(f"Missing required input: {path}")

    primary_results = pd.read_csv(PRIMARY_RESULTS, sep="\t")
    balanced = pd.read_csv(DTHI_BALANCED, sep="\t")
    donor = pd.read_csv(DTHI_DONOR, sep="\t")
    external = pd.read_csv(EXTERNAL, sep="\t")
    masks = pd.read_csv(MASKS, sep="\t")

    external_columns = [
        c for c in external.columns
        if c.startswith("cognitive_") or c.startswith("clinical_")
    ]
    if len(external_columns) != 16:
        raise ValidationError(f"Expected 16 external maps, found {len(external_columns)}.")
    if any(c not in balanced.columns for c in DTHI_COLUMNS):
        raise ValidationError("Balanced DTHI table is missing required maps.")
    if any(c not in donor.columns for c in DTHI_COLUMNS):
        raise ValidationError("Donor DTHI table is missing required maps.")

    bp = parcel_column(balanced)
    ep = parcel_column(external)
    mp = parcel_column(masks)
    dp = parcel_column(donor)
    dc = donor_column(donor)

    balanced[bp] = pd.to_numeric(balanced[bp], errors="coerce")
    external[ep] = pd.to_numeric(external[ep], errors="coerce")
    masks[mp] = pd.to_numeric(masks[mp], errors="coerce")
    donor[dp] = pd.to_numeric(donor[dp], errors="coerce")
    donor[dc] = donor[dc].astype(str)

    balanced = balanced[balanced[bp].between(1, 50)].sort_values(bp).drop_duplicates(bp)
    external = external[external[ep].between(1, 50)].sort_values(ep).drop_duplicates(ep)
    masks = masks[masks[mp].between(1, 50)].sort_values(mp).drop_duplicates(mp)
    donor = donor[donor[dp].between(1, 50)].copy()

    expected_ids = np.arange(1, 51)
    for name, frame, column in [
        ("balanced", balanced, bp),
        ("external", external, ep),
        ("masks", masks, mp),
    ]:
        ids = frame[column].to_numpy(dtype=int)
        if len(frame) != 50 or not np.array_equal(ids, expected_ids):
            raise ValidationError(f"{name} is not ordered as parcel IDs 1-50.")

    primary_mask = masks["primary_LH49"].map(parse_bool).to_numpy(dtype=bool)
    if int(primary_mask.sum()) != 49:
        raise ValidationError(f"Expected primary LH49 mask, found {primary_mask.sum()} parcels.")

    donors = sorted(donor[dc].dropna().astype(str).unique().tolist())
    if len(donors) != 5:
        raise ValidationError(f"Expected five AHBA donors, found {donors}.")
    if int(donor.duplicated([dc, dp]).sum()) != 0:
        raise ValidationError("Duplicate donor-parcel rows are present.")

    for column in DTHI_COLUMNS:
        donor[column] = pd.to_numeric(donor[column], errors="coerce")

    pooled = (
        donor.groupby(dp, sort=True)[DTHI_COLUMNS]
        .mean()
        .reindex(expected_ids)
    )
    pooled.index.name = "parcel_id"

    reconstruction_rows = []
    reconstruction_ok = True
    for dthi_map in DTHI_COLUMNS:
        expected = balanced[dthi_map].to_numpy(dtype=float)
        observed = pooled[dthi_map].to_numpy(dtype=float)
        valid = np.isfinite(expected) & np.isfinite(observed)
        if int(valid.sum()) != 50:
            reconstruction_ok = False
        diff = np.abs(expected[valid] - observed[valid])
        max_diff = float(diff.max()) if len(diff) else np.nan
        mean_diff = float(diff.mean()) if len(diff) else np.nan
        rho = float(spearmanr(expected[valid], observed[valid]).statistic) if len(diff) >= 10 else np.nan
        passed = bool(len(diff) == 50 and max_diff <= 5e-10 and np.isclose(rho, 1.0, atol=1e-12))
        reconstruction_ok &= passed
        reconstruction_rows.append({
            "DTHI_map": dthi_map,
            "finite_parcels": int(valid.sum()),
            "maximum_absolute_difference": max_diff,
            "mean_absolute_difference": mean_diff,
            "spearman_rho_pooled_vs_frozen": rho,
            "passed": passed,
        })

    reconstruction = pd.DataFrame(reconstruction_rows)
    atomic_tsv(reconstruction, RECON_AUDIT_OUT)
    if not reconstruction_ok:
        raise ValidationError(
            "Donor-level means do not reproduce the frozen donor-balanced DTHI maps. "
            "Inspect phase8D2C2B_pooled_reconstruction_audit.tsv before continuing."
        )

    required_primary_columns = {"DTHI_map", "external_map", "spearman_rho"}
    if not required_primary_columns.issubset(primary_results.columns):
        raise ValidationError(
            f"Primary result columns are incompatible: {primary_results.columns.tolist()}"
        )
    primary_lookup = primary_results.set_index(["DTHI_map", "external_map"], drop=False)
    if len(primary_lookup) != 160 or primary_lookup.index.duplicated().any():
        raise ValidationError("Primary result table is not a unique 10x16 grid.")

    external_values = {
        column: external[column].to_numpy(dtype=float)
        for column in external_columns
    }
    balanced_values = {
        column: balanced[column].to_numpy(dtype=float)
        for column in DTHI_COLUMNS
    }

    lodo_rows = []
    stability_rows = []

    print("===== PHASE 8D2C2B: FIVE-DONOR LODO =====", flush=True)
    for omitted in donors:
        remaining = donor[donor[dc] != omitted]
        lodo = (
            remaining.groupby(dp, sort=True)[DTHI_COLUMNS]
            .mean()
            .reindex(expected_ids)
        )
        if not np.isfinite(lodo.to_numpy(dtype=float)[primary_mask, :]).all():
            raise ValidationError(
                f"LODO map after omitting donor {omitted} has non-finite values in LH49."
            )

        for dthi_map in DTHI_COLUMNS:
            n_map, rho_map, r_map = correlation_pair(
                balanced_values[dthi_map],
                lodo[dthi_map].to_numpy(dtype=float),
                primary_mask,
            )
            diff = np.abs(
                balanced_values[dthi_map][primary_mask]
                - lodo[dthi_map].to_numpy(dtype=float)[primary_mask]
            )
            stability_rows.append({
                "omitted_donor": omitted,
                "DTHI_map": dthi_map,
                "n_parcels": n_map,
                "spearman_rho_LODO_vs_full": rho_map,
                "pearson_r_LODO_vs_full": r_map,
                "mean_absolute_difference": float(np.mean(diff)),
                "maximum_absolute_difference": float(np.max(diff)),
            })

            for external_map in external_columns:
                n, rho, _ = correlation_pair(
                    lodo[dthi_map].to_numpy(dtype=float),
                    external_values[external_map],
                    primary_mask,
                )
                primary_row = primary_lookup.loc[(dthi_map, external_map)]
                primary_rho = float(primary_row["spearman_rho"])
                sign_concordant = bool(
                    np.sign(rho) == np.sign(primary_rho)
                ) if np.isfinite(rho) and primary_rho != 0 else bool(np.isclose(rho, primary_rho))
                lodo_rows.append({
                    "omitted_donor": omitted,
                    "DTHI_map": dthi_map,
                    "external_map": external_map,
                    "external_family": (
                        "cognitive" if external_map.startswith("cognitive_") else "clinical"
                    ),
                    "n_parcels": n,
                    "primary_spearman_rho": primary_rho,
                    "LODO_spearman_rho": rho,
                    "delta_from_primary": rho - primary_rho,
                    "absolute_delta_from_primary": abs(rho - primary_rho),
                    "sign_concordant_with_primary": sign_concordant,
                })

        print(f"Completed omission {omitted}: 10 DTHI maps x 16 external maps", flush=True)

    lodo_detail = pd.DataFrame(lodo_rows)
    map_stability = pd.DataFrame(stability_rows)
    atomic_tsv(lodo_detail, LODO_DETAIL_OUT)
    atomic_tsv(map_stability, MAP_STABILITY_OUT)

    if len(lodo_detail) != 800 or len(map_stability) != 50:
        raise ValidationError(
            f"Unexpected LODO row counts: correlations={len(lodo_detail)}, "
            f"map_stability={len(map_stability)}."
        )

    summaries = []
    for (dthi_map, external_map), group in lodo_detail.groupby(
        ["DTHI_map", "external_map"], sort=False
    ):
        if len(group) != 5:
            raise ValidationError(f"Expected five omissions for {dthi_map} vs {external_map}.")
        primary_row = primary_lookup.loc[(dthi_map, external_map)]
        values = group["LODO_spearman_rho"].to_numpy(dtype=float)
        deltas = group["absolute_delta_from_primary"].to_numpy(dtype=float)
        sign_count = int(group["sign_concordant_with_primary"].sum())
        max_delta = float(np.max(deltas))
        if sign_count == 5 and max_delta <= 0.10:
            stability_class = "direction_and_effect_size_stable"
        elif sign_count == 5 and max_delta <= 0.20:
            stability_class = "direction_stable_moderate_effect_variation"
        elif sign_count == 5:
            stability_class = "direction_stable_high_effect_variation"
        elif sign_count == 4:
            stability_class = "mostly_direction_stable"
        else:
            stability_class = "donor_sensitive"

        summaries.append({
            "DTHI_map": dthi_map,
            "external_map": external_map,
            "external_family": group["external_family"].iloc[0],
            "primary_spearman_rho": float(primary_row["spearman_rho"]),
            "primary_p_spin_two_sided": float(primary_row["p_spin_two_sided"]),
            "primary_q_BH_global160": float(primary_row["q_BH_global160"]),
            "primary_q_BH_family80": float(primary_row["q_BH_family80"]),
            "primary_p_maxT_global160": float(primary_row["p_maxT_global160"]),
            "primary_p_maxT_family80": float(primary_row["p_maxT_family80"]),
            "LODO_mean_rho": float(np.mean(values)),
            "LODO_median_rho": float(np.median(values)),
            "LODO_minimum_rho": float(np.min(values)),
            "LODO_maximum_rho": float(np.max(values)),
            "LODO_standard_deviation": float(np.std(values, ddof=1)),
            "LODO_range": float(np.ptp(values)),
            "minimum_absolute_LODO_rho": float(np.min(np.abs(values))),
            "median_absolute_delta_from_primary": float(np.median(deltas)),
            "maximum_absolute_delta_from_primary": max_delta,
            "sign_concordant_omissions": sign_count,
            "sign_concordant_fraction": sign_count / 5.0,
            "all_five_direction_concordant": sign_count == 5,
            "LODO_stability_class": stability_class,
        })

    lodo_summary = pd.DataFrame(summaries)
    if len(lodo_summary) != 160:
        raise ValidationError(f"Expected 160 LODO summaries, found {len(lodo_summary)}.")

    supported = lodo_summary[
        (lodo_summary["primary_q_BH_global160"] < 0.05)
        | (lodo_summary["primary_p_maxT_family80"] < 0.05)
        | (lodo_summary["primary_p_maxT_global160"] < 0.05)
    ].copy()
    supported = supported.sort_values(
        ["primary_p_maxT_global160", "primary_q_BH_global160", "DTHI_map", "external_map"]
    )

    atomic_tsv(lodo_summary, LODO_SUMMARY_OUT)
    atomic_tsv(supported, LODO_SIG_OUT)

    all_finite = bool(
        np.isfinite(
            lodo_detail[
                ["primary_spearman_rho", "LODO_spearman_rho",
                 "delta_from_primary", "absolute_delta_from_primary"]
            ].to_numpy(dtype=float)
        ).all()
        and np.isfinite(
            lodo_summary[
                ["primary_spearman_rho", "LODO_mean_rho", "LODO_median_rho",
                 "LODO_minimum_rho", "LODO_maximum_rho",
                 "maximum_absolute_delta_from_primary"]
            ].to_numpy(dtype=float)
        ).all()
        and np.isfinite(
            map_stability[
                ["spearman_rho_LODO_vs_full", "pearson_r_LODO_vs_full",
                 "mean_absolute_difference", "maximum_absolute_difference"]
            ].to_numpy(dtype=float)
        ).all()
    )

    audit = pd.DataFrame([
        {"section": "upstream", "item": "Phase8D2C1_ready", "value": True, "passed": True, "detail": ""},
        {"section": "upstream", "item": "Phase8D2C2A_ready_for_LODO", "value": True, "passed": True, "detail": ""},
        {"section": "inputs", "item": "AHBA_donors", "value": len(donors), "passed": len(donors) == 5, "detail": "|".join(donors)},
        {"section": "inputs", "item": "primary_LH_parcels", "value": int(primary_mask.sum()), "passed": int(primary_mask.sum()) == 49, "detail": ""},
        {"section": "reconstruction", "item": "pooled_donor_means_reproduce_frozen_maps", "value": reconstruction_ok, "passed": reconstruction_ok, "detail": "tolerance_max_abs_diff<=5e-10"},
        {"section": "LODO", "item": "correlation_rows", "value": len(lodo_detail), "passed": len(lodo_detail) == 800, "detail": "5_donors_x_10_DTHI_x_16_external"},
        {"section": "LODO", "item": "pair_summaries", "value": len(lodo_summary), "passed": len(lodo_summary) == 160, "detail": ""},
        {"section": "LODO", "item": "DTHI_map_stability_rows", "value": len(map_stability), "passed": len(map_stability) == 50, "detail": "5_donors_x_10_DTHI"},
        {"section": "LODO", "item": "all_values_finite", "value": all_finite, "passed": all_finite, "detail": ""},
        {"section": "interpretation", "item": "LODO_role", "value": "robustness_only", "passed": True, "detail": "no_spatial_retesting_no_replication_claim"},
    ])
    atomic_tsv(audit, AUDIT_OUT)

    ready = bool(
        reconstruction_ok
        and len(lodo_detail) == 800
        and len(lodo_summary) == 160
        and len(map_stability) == 50
        and all_finite
    )
    summary = pd.DataFrame([{
        "Phase8D2C1_status_confirmed": True,
        "Phase8D2C2A_status_confirmed": True,
        "AHBA_donors": len(donors),
        "DTHI_maps": len(DTHI_COLUMNS),
        "external_maps": len(external_columns),
        "LODO_correlations_completed": len(lodo_detail),
        "LODO_pair_summaries": len(lodo_summary),
        "primary_supported_pairs_reviewed": len(supported),
        "supported_pairs_all_five_direction_concordant": int(supported["all_five_direction_concordant"].sum()),
        "all_LODO_values_finite": all_finite,
        "ready_for_phase8D2C2C_bijective_spin_sensitivity": ready,
        "Phase8D2C2B_status": "completed" if ready else "failed_validation",
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }])
    atomic_tsv(summary, SUMMARY_OUT)

    hash_paths = [
        LODO_DETAIL_OUT, LODO_SUMMARY_OUT, LODO_SIG_OUT, MAP_STABILITY_OUT,
        RECON_AUDIT_OUT, AUDIT_OUT, SUMMARY_OUT,
    ]
    hashes = pd.DataFrame([
        {
            "relative_path": str(path.relative_to(PROJECT)),
            "size_bytes": path.stat().st_size,
            "SHA256": sha256(path),
        }
        for path in hash_paths
    ])
    atomic_tsv(hashes, HASH_OUT)

    print("\n===== PHASE 8D2C2B COMPLETION =====")
    print(summary.to_string(index=False))
    print("\n===== PRIMARY-SUPPORTED PAIR LODO ROBUSTNESS =====")
    if supported.empty:
        print("No primary-supported pairs were identified.")
    else:
        columns = [
            "DTHI_map", "external_map", "primary_spearman_rho",
            "primary_q_BH_global160", "primary_p_maxT_family80",
            "LODO_minimum_rho", "LODO_maximum_rho",
            "maximum_absolute_delta_from_primary",
            "sign_concordant_omissions", "LODO_stability_class",
        ]
        print(supported[columns].to_string(index=False))
    print("\n===== LODO MAP-STABILITY SUMMARY =====")
    map_summary = (
        map_stability.groupby("DTHI_map", as_index=False)
        .agg(
            minimum_LODO_vs_full_spearman=("spearman_rho_LODO_vs_full", "min"),
            median_LODO_vs_full_spearman=("spearman_rho_LODO_vs_full", "median"),
            maximum_LODO_map_difference=("maximum_absolute_difference", "max"),
        )
        .sort_values("minimum_LODO_vs_full_spearman")
    )
    print(map_summary.to_string(index=False))
    print("\n===== AUDIT =====")
    print(audit.to_string(index=False))
    print("\n===== CANONICAL OUTPUT HASHES =====")
    print(hashes.to_string(index=False))

    if not ready:
        raise RuntimeError("Phase 8D2C2B validation failed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"Phase 8D2C2B failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise
