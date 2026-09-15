#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT = Path(".")
OUT = PROJECT / "03_processed_data/functional_imaging/phase8D/phase8D2C"
TABLES = PROJECT / "07_tables/main_tables/phase8"

C2C_SUMMARY = TABLES / "phase8D2C2C_completion_summary.tsv"
NONBIJ_PRIMARY = OUT / "phase8D2C2A_primary_spatial_correspondence.tsv"
NONBIJ_SENS = OUT / "phase8D2C2A_high_support_sensitivity.tsv"
LODO_SUMMARY = OUT / "phase8D2C2B_LODO_summary.tsv"
LODO_MAP_STABILITY = OUT / "phase8D2C2B_LODO_DTHI_map_stability.tsv"
BIJ_PRIMARY = OUT / "phase8D2C2C_bijective_primary_spatial_correspondence.tsv"
BIJ_SENS = OUT / "phase8D2C2C_bijective_high_support_sensitivity.tsv"

INTEGRATED_OUT = OUT / "phase8D2C3_integrated_evidence.tsv"
PRIORITY_OUT = OUT / "phase8D2C3_priority_associations.tsv"
RHO_MATRIX_OUT = OUT / "phase8D2C3_DTHI_external_spearman_matrix.tsv"
EVIDENCE_MATRIX_OUT = OUT / "phase8D2C3_evidence_class_matrix.tsv"
MAP_STABILITY_OUT = OUT / "phase8D2C3_DTHI_map_stability_summary.tsv"
NARRATIVE_OUT = OUT / "phase8D2C3_evidence_interpretation.txt"

AUDIT_OUT = TABLES / "phase8D2C3_evidence_synthesis_audit.tsv"
COUNTS_OUT = TABLES / "phase8D2C3_evidence_counts.tsv"
SUMMARY_OUT = TABLES / "phase8D2C3_completion_summary.tsv"
HASH_OUT = TABLES / "phase8D2C3_canonical_output_hashes.tsv"

KEYS = ["DTHI_map", "external_map", "external_family"]


class ValidationError(RuntimeError):
    pass


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def require_file(path: Path) -> None:
    if not path.is_file():
        raise ValidationError(f"Missing required file: {path}")


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValidationError(f"{label} is missing columns: {missing}")


def assert_unique_pairs(frame: pd.DataFrame, label: str) -> None:
    require_columns(frame, KEYS, label)
    if frame.duplicated(KEYS).any():
        examples = frame.loc[frame.duplicated(KEYS, keep=False), KEYS].head(10)
        raise ValidationError(
            f"{label} contains duplicate map pairs:\n{examples.to_string(index=False)}"
        )


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


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def renamed_metrics(
    frame: pd.DataFrame,
    metrics: list[str],
    prefix: str,
    label: str,
) -> pd.DataFrame:
    require_columns(frame, KEYS + metrics, label)
    result = frame[KEYS + metrics].copy()
    result = result.rename(
        columns={metric: f"{prefix}{metric}" for metric in metrics}
    )
    return result


def corrected_any(frame: pd.DataFrame, prefix: str) -> pd.Series:
    return (
        frame[f"{prefix}q_BH_global160"].lt(0.05)
        | frame[f"{prefix}q_BH_family80"].lt(0.05)
        | frame[f"{prefix}p_maxT_global160"].lt(0.05)
        | frame[f"{prefix}p_maxT_family80"].lt(0.05)
    )


def evidence_class(row: pd.Series) -> str:
    if row["bijective_p_maxT_global160"] < 0.05:
        return "global_FWER_bijective"
    if row["bijective_p_maxT_family80"] < 0.05:
        return "family_FWER_bijective"
    if row["bijective_q_BH_global160"] < 0.05:
        return "global_FDR_bijective"
    if row["bijective_q_BH_family80"] < 0.05:
        return "family_FDR_bijective"
    if row["nonbijective_any_corrected"]:
        return "nonbijective_only_corrected"
    if row["bijective_p_spin_two_sided"] < 0.05:
        return "bijective_nominal_only"
    return "no_corrected_spatial_support"


def high_support_class(row: pd.Series) -> str:
    if row["bijective_high_support_q_BH_sensitivity160"] < 0.05:
        return "LH46_BH_retained"
    if row["bijective_high_support_p_spin_two_sided"] < 0.05:
        return "LH46_nominal_only"
    if np.sign(row["spearman_rho"]) == np.sign(
        row["bijective_high_support_spearman_rho"]
    ):
        return "LH46_direction_concordant_not_nominal"
    return "LH46_direction_discordant"


def integrated_statement(row: pd.Series) -> str:
    main = row["evidence_class"]
    donor = (
        "LODO_5of5_direction"
        if row["all_five_direction_concordant"]
        else f"LODO_{int(row['sign_concordant_omissions'])}of5_direction"
    )
    support = row["high_support_class"]
    return f"{main};{donor};{support}"


def main() -> None:
    required = [
        C2C_SUMMARY,
        NONBIJ_PRIMARY,
        NONBIJ_SENS,
        LODO_SUMMARY,
        LODO_MAP_STABILITY,
        BIJ_PRIMARY,
        BIJ_SENS,
    ]
    for path in required:
        require_file(path)

    completion = pd.read_csv(C2C_SUMMARY, sep="\t")
    if len(completion) != 1:
        raise ValidationError("Phase 8D2C2C completion summary must contain one row.")
    if str(completion.loc[0, "Phase8D2C2C_status"]).strip() != "completed":
        raise ValidationError("Phase 8D2C2C is not completed.")
    if not parse_bool(
        completion.loc[0, "ready_for_phase8D2C3_evidence_synthesis"]
    ):
        raise ValidationError("Phase 8D2C2C is not ready for evidence synthesis.")

    nonbij_primary = pd.read_csv(NONBIJ_PRIMARY, sep="\t")
    nonbij_sens = pd.read_csv(NONBIJ_SENS, sep="\t")
    lodo = pd.read_csv(LODO_SUMMARY, sep="\t")
    map_stability = pd.read_csv(LODO_MAP_STABILITY, sep="\t")
    bij_primary = pd.read_csv(BIJ_PRIMARY, sep="\t")
    bij_sens = pd.read_csv(BIJ_SENS, sep="\t")

    for frame, label, expected in [
        (nonbij_primary, "non-bijective primary table", 160),
        (nonbij_sens, "non-bijective sensitivity table", 160),
        (lodo, "LODO summary", 160),
        (bij_primary, "bijective primary table", 160),
        (bij_sens, "bijective sensitivity table", 160),
    ]:
        if len(frame) != expected:
            raise ValidationError(
                f"{label} must contain {expected} rows; found {len(frame)}."
            )
        assert_unique_pairs(frame, label)

    primary_metrics = [
        "spearman_rho",
        "p_spin_two_sided",
        "q_BH_global160",
        "q_BH_family80",
        "p_maxT_global160",
        "p_maxT_family80",
    ]
    sens_metrics = [
        "spearman_rho",
        "p_spin_two_sided",
        "q_BH_sensitivity160",
    ]
    lodo_metrics = [
        "LODO_mean_rho",
        "LODO_median_rho",
        "LODO_minimum_rho",
        "LODO_maximum_rho",
        "LODO_standard_deviation",
        "minimum_absolute_LODO_rho",
        "maximum_absolute_delta_from_primary",
        "sign_concordant_omissions",
        "sign_concordant_fraction",
        "all_five_direction_concordant",
        "LODO_stability_class",
    ]

    integrated = renamed_metrics(
        bij_primary, primary_metrics, "bijective_", "bijective primary"
    )
    integrated = integrated.rename(
        columns={"bijective_spearman_rho": "spearman_rho"}
    )

    nonbij = renamed_metrics(
        nonbij_primary, primary_metrics, "nonbijective_", "non-bijective primary"
    )
    integrated = integrated.merge(
        nonbij, on=KEYS, how="inner", validate="one_to_one"
    )

    bij_sens_sub = renamed_metrics(
        bij_sens, sens_metrics, "bijective_high_support_", "bijective sensitivity"
    )
    nonbij_sens_sub = renamed_metrics(
        nonbij_sens,
        sens_metrics,
        "nonbijective_high_support_",
        "non-bijective sensitivity",
    )
    integrated = integrated.merge(
        bij_sens_sub, on=KEYS, how="inner", validate="one_to_one"
    ).merge(
        nonbij_sens_sub, on=KEYS, how="inner", validate="one_to_one"
    )

    require_columns(lodo, KEYS + lodo_metrics, "LODO summary")
    integrated = integrated.merge(
        lodo[KEYS + lodo_metrics],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )

    if len(integrated) != 160:
        raise ValidationError(
            f"Integrated table must contain 160 pairs; found {len(integrated)}."
        )

    integrated["observed_rho_matches_nonbijective"] = np.isclose(
        integrated["spearman_rho"],
        integrated["nonbijective_spearman_rho"],
        rtol=0,
        atol=1e-12,
    )
    if not integrated["observed_rho_matches_nonbijective"].all():
        raise ValidationError(
            "Observed rho differs between bijective and non-bijective analyses."
        )

    integrated["nonbijective_any_corrected"] = corrected_any(
        integrated, "nonbijective_"
    )
    integrated["bijective_any_corrected"] = corrected_any(
        integrated, "bijective_"
    )
    integrated["bijective_global_maxT_significant"] = integrated[
        "bijective_p_maxT_global160"
    ].lt(0.05)
    integrated["bijective_family_maxT_significant"] = integrated[
        "bijective_p_maxT_family80"
    ].lt(0.05)
    integrated["bijective_global_BH_significant"] = integrated[
        "bijective_q_BH_global160"
    ].lt(0.05)
    integrated["bijective_family_BH_significant"] = integrated[
        "bijective_q_BH_family80"
    ].lt(0.05)
    integrated["bijective_high_support_BH_significant"] = integrated[
        "bijective_high_support_q_BH_sensitivity160"
    ].lt(0.05)
    integrated["LH46_direction_concordant"] = (
        np.sign(integrated["spearman_rho"])
        == np.sign(integrated["bijective_high_support_spearman_rho"])
    )
    integrated["LH46_rho_delta_from_LH49"] = (
        integrated["bijective_high_support_spearman_rho"]
        - integrated["spearman_rho"]
    )
    integrated["evidence_class"] = integrated.apply(evidence_class, axis=1)
    integrated["high_support_class"] = integrated.apply(
        high_support_class, axis=1
    )
    integrated["integrated_evidence_statement"] = integrated.apply(
        integrated_statement, axis=1
    )

    integrated["priority_for_reporting"] = (
        integrated["nonbijective_any_corrected"]
        | integrated["bijective_any_corrected"]
    )
    integrated["rank_global_maxT"] = integrated[
        "bijective_p_maxT_global160"
    ].rank(method="min")
    integrated = integrated.sort_values(
        [
            "bijective_p_maxT_global160",
            "bijective_p_maxT_family80",
            "bijective_q_BH_global160",
            "DTHI_map",
            "external_map",
        ]
    ).reset_index(drop=True)

    priority = integrated.loc[
        integrated["priority_for_reporting"]
    ].copy().reset_index(drop=True)

    rho_matrix = integrated.pivot(
        index="DTHI_map",
        columns="external_map",
        values="spearman_rho",
    )
    evidence_matrix = integrated.pivot(
        index="DTHI_map",
        columns="external_map",
        values="evidence_class",
    )

    require_columns(
        map_stability,
        [
            "DTHI_map",
            "spearman_rho_LODO_vs_full",
            "maximum_absolute_difference",
        ],
        "LODO map stability",
    )
    stability_summary = (
        map_stability.groupby("DTHI_map", as_index=False)
        .agg(
            minimum_LODO_vs_full_spearman=(
                "spearman_rho_LODO_vs_full",
                "min",
            ),
            median_LODO_vs_full_spearman=(
                "spearman_rho_LODO_vs_full",
                "median",
            ),
            maximum_LODO_map_difference=(
                "maximum_absolute_difference",
                "max",
            ),
        )
        .sort_values("minimum_LODO_vs_full_spearman")
    )

    counts = pd.DataFrame(
        [
            {
                "metric": "integrated_pairs",
                "count": len(integrated),
            },
            {
                "metric": "bijective_global_maxT_significant",
                "count": int(
                    integrated["bijective_global_maxT_significant"].sum()
                ),
            },
            {
                "metric": "bijective_family_maxT_significant",
                "count": int(
                    integrated["bijective_family_maxT_significant"].sum()
                ),
            },
            {
                "metric": "bijective_global_BH_significant",
                "count": int(
                    integrated["bijective_global_BH_significant"].sum()
                ),
            },
            {
                "metric": "bijective_family_BH_significant",
                "count": int(
                    integrated["bijective_family_BH_significant"].sum()
                ),
            },
            {
                "metric": "bijective_LH46_BH_significant",
                "count": int(
                    integrated["bijective_high_support_BH_significant"].sum()
                ),
            },
            {
                "metric": "nonbijective_any_corrected",
                "count": int(integrated["nonbijective_any_corrected"].sum()),
            },
            {
                "metric": "priority_for_reporting",
                "count": len(priority),
            },
            {
                "metric": "priority_LODO_5of5_direction",
                "count": int(
                    priority["all_five_direction_concordant"].sum()
                ),
            },
        ]
    )

    finite_columns = [
        "spearman_rho",
        "bijective_p_spin_two_sided",
        "bijective_q_BH_global160",
        "bijective_q_BH_family80",
        "bijective_p_maxT_global160",
        "bijective_p_maxT_family80",
        "bijective_high_support_spearman_rho",
        "bijective_high_support_p_spin_two_sided",
        "bijective_high_support_q_BH_sensitivity160",
        "LODO_mean_rho",
        "LODO_minimum_rho",
        "LODO_maximum_rho",
        "maximum_absolute_delta_from_primary",
    ]
    all_finite = bool(
        np.isfinite(integrated[finite_columns].to_numpy(dtype=float)).all()
    )

    global_hits = integrated.loc[
        integrated["bijective_global_maxT_significant"]
    ].copy()
    family_hits = integrated.loc[
        integrated["bijective_family_maxT_significant"]
        & ~integrated["bijective_global_maxT_significant"]
    ].copy()

    lines = [
        "PHASE 8D2C3 EVIDENCE INTERPRETATION",
        "",
        "Primary inferential hierarchy:",
        "1. Exact-bijective global maxT is the strongest whole-analysis familywise criterion.",
        "2. Exact-bijective family maxT is family-specific support.",
        "3. BH-FDR, LH46 sensitivity, and LODO provide complementary—not interchangeable—evidence.",
        "",
        f"Integrated map pairs: {len(integrated)}",
        f"Exact-bijective global maxT significant: {len(global_hits)}",
        f"Exact-bijective family maxT significant: "
        f"{int(integrated['bijective_family_maxT_significant'].sum())}",
        f"Exact-bijective global BH significant: "
        f"{int(integrated['bijective_global_BH_significant'].sum())}",
        f"LH46 BH significant: "
        f"{int(integrated['bijective_high_support_BH_significant'].sum())}",
        "",
    ]

    if len(global_hits):
        lines.append("Globally familywise-corrected exact-bijective associations:")
        for _, row in global_hits.iterrows():
            lines.append(
                f"- {row['DTHI_map']} vs {row['external_map']}: "
                f"rho={row['spearman_rho']:.6f}, "
                f"p_global_maxT={row['bijective_p_maxT_global160']:.6f}, "
                f"p_family_maxT={row['bijective_p_maxT_family80']:.6f}, "
                f"q_global_BH={row['bijective_q_BH_global160']:.6f}, "
                f"LODO_direction={int(row['sign_concordant_omissions'])}/5, "
                f"LH46={row['high_support_class']}."
            )
    else:
        lines.append(
            "No association met exact-bijective global maxT correction."
        )

    if len(family_hits):
        lines.extend(["", "Additional family-maxT-only associations:"])
        for _, row in family_hits.iterrows():
            lines.append(
                f"- {row['DTHI_map']} vs {row['external_map']}: "
                f"rho={row['spearman_rho']:.6f}, "
                f"p_family_maxT={row['bijective_p_maxT_family80']:.6f}."
            )

    lines.extend(
        [
            "",
            "Interpretive guardrails:",
            "- A global maxT result may coexist with q_BH > 0.05 because the methods use different correction principles and different null information.",
            "- LODO assesses donor-direction and effect-size stability; it is not an independent replication or a new significance test.",
            "- LH46 is a nested support sensitivity analysis; failure to retain BH significance weakens generalizability across the highest-support parcels but does not mathematically invalidate the LH49 result.",
            "- Results supported only by the original non-bijective spins are exploratory after exact-bijective reassessment.",
            "",
        ]
    )
    narrative = "\n".join(lines)

    atomic_tsv(integrated, INTEGRATED_OUT)
    atomic_tsv(priority, PRIORITY_OUT)
    atomic_tsv(rho_matrix.reset_index(), RHO_MATRIX_OUT)
    atomic_tsv(evidence_matrix.reset_index(), EVIDENCE_MATRIX_OUT)
    atomic_tsv(stability_summary, MAP_STABILITY_OUT)
    atomic_tsv(counts, COUNTS_OUT)
    atomic_text(narrative, NARRATIVE_OUT)

    audit = pd.DataFrame(
        [
            {
                "section": "upstream",
                "item": "Phase8D2C2C_completed",
                "value": True,
                "passed": True,
                "detail": "",
            },
            {
                "section": "integration",
                "item": "primary_pairs",
                "value": len(integrated),
                "passed": len(integrated) == 160,
                "detail": "10_DTHI_x_16_external",
            },
            {
                "section": "integration",
                "item": "observed_rho_identical_between_null_models",
                "value": bool(
                    integrated[
                        "observed_rho_matches_nonbijective"
                    ].all()
                ),
                "passed": bool(
                    integrated[
                        "observed_rho_matches_nonbijective"
                    ].all()
                ),
                "detail": "expected_because_only_null_assignment_changed",
            },
            {
                "section": "inference",
                "item": "exact_bijective_global_maxT_hits",
                "value": len(global_hits),
                "passed": True,
                "detail": "",
            },
            {
                "section": "sensitivity",
                "item": "LH46_BH_hits",
                "value": int(
                    integrated[
                        "bijective_high_support_BH_significant"
                    ].sum()
                ),
                "passed": True,
                "detail": "nested_not_independent_replication",
            },
            {
                "section": "robustness",
                "item": "priority_pairs_LODO_5of5",
                "value": int(
                    priority[
                        "all_five_direction_concordant"
                    ].sum()
                ),
                "passed": True,
                "detail": f"of_{len(priority)}_priority_pairs",
            },
            {
                "section": "outputs",
                "item": "all_integrated_numeric_values_finite",
                "value": all_finite,
                "passed": all_finite,
                "detail": "",
            },
        ]
    )
    atomic_tsv(audit, AUDIT_OUT)

    ready = bool(
        len(integrated) == 160
        and all_finite
        and integrated["observed_rho_matches_nonbijective"].all()
    )
    summary = pd.DataFrame(
        [
            {
                "Phase8D2C2C_status_confirmed": True,
                "integrated_pairs": len(integrated),
                "priority_pairs": len(priority),
                "bijective_global_maxT_significant": int(
                    integrated["bijective_global_maxT_significant"].sum()
                ),
                "bijective_family_maxT_significant": int(
                    integrated["bijective_family_maxT_significant"].sum()
                ),
                "bijective_global_BH_significant": int(
                    integrated["bijective_global_BH_significant"].sum()
                ),
                "bijective_family_BH_significant": int(
                    integrated["bijective_family_BH_significant"].sum()
                ),
                "bijective_LH46_BH_significant": int(
                    integrated["bijective_high_support_BH_significant"].sum()
                ),
                "priority_pairs_LODO_5of5_direction": int(
                    priority["all_five_direction_concordant"].sum()
                ),
                "all_integrated_values_finite": all_finite,
                "ready_for_phase8D2C4_figures": ready,
                "Phase8D2C3_status": "completed" if ready else "failed",
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
            }
        ]
    )
    atomic_tsv(summary, SUMMARY_OUT)

    canonical = [
        INTEGRATED_OUT,
        PRIORITY_OUT,
        RHO_MATRIX_OUT,
        EVIDENCE_MATRIX_OUT,
        MAP_STABILITY_OUT,
        NARRATIVE_OUT,
        COUNTS_OUT,
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
            for path in canonical
        ]
    )
    atomic_tsv(hashes, HASH_OUT)

    print("===== PHASE 8D2C3 COMPLETION =====")
    print(summary.to_string(index=False))
    print()
    print("===== EXACT-BIJECTIVE GLOBAL maxT RESULTS =====")
    display_columns = [
        "DTHI_map",
        "external_map",
        "external_family",
        "spearman_rho",
        "bijective_p_spin_two_sided",
        "bijective_q_BH_global160",
        "bijective_p_maxT_global160",
        "bijective_p_maxT_family80",
        "all_five_direction_concordant",
        "LODO_minimum_rho",
        "LODO_maximum_rho",
        "high_support_class",
        "integrated_evidence_statement",
    ]
    if len(global_hits):
        print(global_hits[display_columns].to_string(index=False))
    else:
        print("None")
    print()
    print("===== PRIORITY ASSOCIATIONS =====")
    print(priority[display_columns].to_string(index=False))
    print()
    print("===== EVIDENCE COUNTS =====")
    print(counts.to_string(index=False))
    print()
    print("===== AUDIT =====")
    print(audit.to_string(index=False))
    print()
    print("===== CANONICAL OUTPUT HASHES =====")
    print(hashes.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"Phase 8D2C3 failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise
