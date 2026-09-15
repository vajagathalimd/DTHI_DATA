#!/usr/bin/env python3
"""Phase 8D3B: cross-modal convergence synthesis from the Phase 8D3A lock."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

LOCK_ROOT = (
    ROOT
    / "03_processed_data"
    / "functional_imaging"
    / "phase8D"
    / "phase8D3A_canonical_lock"
)
LOCK_MANIFEST = LOCK_ROOT / "phase8D3A_locked_inputs_manifest.tsv"
D3A_COMPLETION = (
    ROOT
    / "07_tables"
    / "main_tables"
    / "phase8"
    / "phase8D3A_completion_summary.tsv"
)

OUT_DIR = (
    ROOT
    / "03_processed_data"
    / "functional_imaging"
    / "phase8D"
    / "phase8D3B_cross_modal_synthesis"
)
TABLE_DIR = ROOT / "07_tables" / "main_tables" / "phase8"

MASTER_OUT = OUT_DIR / "phase8D3B_cross_modal_master_matrix.tsv"
ASSOCIATION_OUT = OUT_DIR / "phase8D3B_map_external_association_summary.tsv"
PRIORITY_OUT = OUT_DIR / "phase8D3B_priority_evidence_hierarchy.tsv"
SOURCE_OUT = OUT_DIR / "phase8D3B_source_provenance.tsv"
INTERPRETATION_OUT = OUT_DIR / "phase8D3B_interpretation.txt"
AUDIT_OUT = TABLE_DIR / "phase8D3B_cross_modal_audit.tsv"
COMPLETION_OUT = TABLE_DIR / "phase8D3B_completion_summary.tsv"

DTHI_ORDER = [
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

DEFINITION_ANNOTATIONS = {
    "patterning_arealization": {
        "map_type": "primary_seed_module",
        "developmental_role": "early_fetal_patterning_and_cortical_arealization",
        "dominant_cell_systems": "radial_glia;early_neural_progenitors",
        "biological_process": "morphogen_signalling;cortical_identity;areal_specification",
    },
    "progenitor_radial_glia": {
        "map_type": "primary_seed_module",
        "developmental_role": "early_to_mid_fetal_progenitor_expansion",
        "dominant_cell_systems": "radial_glia;outer_radial_glia;intermediate_progenitors",
        "biological_process": "progenitor_maintenance;cell_cycle;neurogenic_transition",
    },
    "neurogenesis_migration_layering": {
        "map_type": "primary_seed_module",
        "developmental_role": "mid_fetal_neurogenesis_migration_and_layer_formation",
        "dominant_cell_systems": "intermediate_progenitors;newborn_excitatory_neurons;radial_glia",
        "biological_process": "neuronal_differentiation;migration;cortical_layering",
    },
    "axon_guidance_neurite_outgrowth": {
        "map_type": "primary_seed_module",
        "developmental_role": "mid_to_late_fetal_projection_and_neurite_organization",
        "dominant_cell_systems": "projection_neurons;developing_interneurons",
        "biological_process": "axon_guidance;neurite_outgrowth;target_selection",
    },
    "synaptic_assembly_receptor_trafficking": {
        "map_type": "primary_seed_module",
        "developmental_role": "late_fetal_to_postnatal_synaptic_assembly",
        "dominant_cell_systems": "excitatory_neurons;inhibitory_neurons",
        "biological_process": "synaptic_adhesion;receptor_trafficking;synapse_organization",
    },
    "astrocyte_maturation_metabolic_support": {
        "map_type": "primary_seed_module",
        "developmental_role": "late_fetal_to_postnatal_astroglial_maturation",
        "dominant_cell_systems": "astrocytes",
        "biological_process": "metabolic_support;neurotransmitter_homeostasis;gliotransmission",
    },
    "oligodendrocyte_myelination": {
        "map_type": "primary_seed_module",
        "developmental_role": "postnatal_childhood_myelination_and_circuit_stabilization",
        "dominant_cell_systems": "oligodendrocyte_precursors;oligodendrocytes",
        "biological_process": "oligodendrocyte_maturation;myelination;axon_glia_support",
    },
    "activity_dependent_plasticity": {
        "map_type": "primary_seed_module",
        "developmental_role": "postnatal_to_adolescent_activity_dependent_maturation",
        "dominant_cell_systems": "excitatory_neurons;inhibitory_neurons",
        "biological_process": "calcium_signalling;immediate_early_response;synaptic_plasticity",
    },
    "synaptic_membrane_structural_candidates": {
        "map_type": "derived_candidate_map",
        "developmental_role": "derived_synaptic_membrane_architecture_priority",
        "dominant_cell_systems": "neurons;axon_glia_interfaces",
        "biological_process": "membrane_adhesion;receptor_scaffolding;structural_interactions",
    },
    "preliminary_maturation_balance": {
        "map_type": "derived_composite_map",
        "developmental_role": "derived_balance_of_early_and_late_maturation_programs",
        "dominant_cell_systems": "mixed_neuronal_and_glial_systems",
        "biological_process": "developmental_timing_balance;circuit_maturation",
    },
}

ALIASES = {
    "patterning": "patterning_arealization",
    "arealization": "patterning_arealization",
    "progenitor": "progenitor_radial_glia",
    "radial_glia": "progenitor_radial_glia",
    "neurogenesis": "neurogenesis_migration_layering",
    "migration_layering": "neurogenesis_migration_layering",
    "axon_guidance": "axon_guidance_neurite_outgrowth",
    "neurite_outgrowth": "axon_guidance_neurite_outgrowth",
    "synaptic_assembly": "synaptic_assembly_receptor_trafficking",
    "receptor_trafficking": "synaptic_assembly_receptor_trafficking",
    "astrocyte_maturation": "astrocyte_maturation_metabolic_support",
    "metabolic_support": "astrocyte_maturation_metabolic_support",
    "oligodendrocyte": "oligodendrocyte_myelination",
    "myelination": "oligodendrocyte_myelination",
    "activity_dependent": "activity_dependent_plasticity",
    "plasticity": "activity_dependent_plasticity",
    "synaptic_membrane": "synaptic_membrane_structural_candidates",
    "maturation_balance": "preliminary_maturation_balance",
}

WINDOW_ORDER = {
    "early_fetal_patterning": 1,
    "mid_fetal_neurogenesis": 2,
    "late_fetal_synaptogenesis": 3,
    "infancy_postnatal_circuit_formation": 4,
    "childhood_circuit_refinement": 5,
    "adolescence_myelination_refinement": 6,
    "adult_stabilization": 7,
}


class ValidationError(RuntimeError):
    pass


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


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def normalize_token(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def resolve_map(value: Any) -> str | None:
    token = normalize_token(value)
    if token in DTHI_ORDER:
        return token
    if token in ALIASES:
        return ALIASES[token]
    for alias, canonical in ALIASES.items():
        if alias in token:
            return canonical
    for canonical in DTHI_ORDER:
        if canonical in token or token in canonical:
            return canonical
    return None


def read_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator, nrows=nrows, low_memory=False)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise ValidationError(f"Required file not found: {path}")


def validate_d3a() -> None:
    require_file(D3A_COMPLETION)
    frame = pd.read_csv(D3A_COMPLETION, sep="\t")
    if len(frame) != 1:
        raise ValidationError("Phase 8D3A completion summary must contain one row.")
    if str(frame.loc[0, "Phase8D3A_status"]).strip() != "completed":
        raise ValidationError("Phase 8D3A is not completed.")
    if not parse_bool(frame.loc[0, "ready_for_phase8D3B_cross_modal_synthesis"]):
        raise ValidationError("Phase 8D3A is not ready for Phase 8D3B.")


def load_manifest() -> pd.DataFrame:
    require_file(LOCK_MANIFEST)
    manifest = pd.read_csv(LOCK_MANIFEST, sep="\t")
    required = {
        "source_relative_path",
        "snapshot_relative_path",
        "snapshot_SHA256",
        "hash_match",
    }
    if not required.issubset(manifest.columns):
        raise ValidationError(
            f"Lock manifest is missing columns: {sorted(required - set(manifest.columns))}"
        )
    if not manifest["hash_match"].map(parse_bool).all():
        raise ValidationError("The Phase 8D3A manifest contains a failed hash match.")
    return manifest


def locked_path(manifest: pd.DataFrame, filename: str) -> Path:
    hits = manifest[
        manifest["source_relative_path"].astype(str).str.endswith("/" + filename)
        | manifest["source_relative_path"].astype(str).eq(filename)
    ]
    if len(hits) != 1:
        raise ValidationError(
            f"Expected one locked file named {filename}; found {len(hits)}."
        )
    path = LOCK_ROOT / str(hits.iloc[0]["snapshot_relative_path"])
    require_file(path)
    if sha256(path) != str(hits.iloc[0]["snapshot_SHA256"]):
        raise ValidationError(f"Locked snapshot hash mismatch: {path}")
    return path


def best_association(subset: pd.DataFrame) -> pd.Series:
    index = subset["spearman_rho"].abs().idxmax()
    return subset.loc[index]


def evidence_tier(rows: pd.DataFrame) -> str:
    if (rows["bijective_p_maxT_global160"] < 0.05).any():
        return "Tier1_global_FWER"
    if (rows["bijective_p_maxT_family80"] < 0.05).any():
        return "Tier2_family_FWER"
    if (rows["bijective_q_BH_family80"] < 0.05).any():
        return "Tier2_family_FDR"
    if (rows["bijective_p_spin_two_sided"] < 0.05).any():
        return "Tier3_nominal_spatial_correspondence"
    return "Tier3_descriptive_only"


def candidate_files() -> list[Path]:
    roots = [
        ROOT / "03_processed_data",
        ROOT / "07_tables",
    ]
    files: list[Path] = []
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".tsv", ".csv", ".txt"}:
                continue
            relative = str(path.relative_to(ROOT))
            if "phase8D3A_canonical_lock" in relative:
                continue
            if "phase8D3B_cross_modal_synthesis" in relative:
                continue
            if path.stat().st_size > 80 * 1024 * 1024:
                continue
            files.append(path)
    return sorted(set(files))


def header_candidate(path: Path, kind: str) -> dict[str, Any] | None:
    try:
        head = read_table(path, nrows=5)
    except Exception:
        return None
    if head.empty and len(head.columns) == 0:
        return None

    columns = {normalize_token(column): column for column in head.columns}
    map_matches = [column for token, column in columns.items() if resolve_map(token)]
    map_col = next(
        (
            columns[token]
            for token in [
                "dthi_map",
                "module",
                "module_name",
                "program",
                "signature",
                "component",
            ]
            if token in columns
        ),
        None,
    )

    if kind == "developmental":
        category_tokens = [
            "developmental_window",
            "window",
            "stage",
            "age_window",
            "period",
        ]
        keywords = ["development", "brainspan", "trajectory", "window", "temporal"]
    else:
        category_tokens = [
            "cell_type",
            "celltype",
            "cell_class",
            "cluster",
            "broad_cell_type",
        ]
        keywords = ["cell_type", "celltype", "single_cell", "pseudobulk", "marker"]

    category_col = next(
        (columns[token] for token in category_tokens if token in columns),
        None,
    )

    preferred_scores = [
        "module_score",
        "mean_score",
        "mean_expression",
        "average_expression",
        "enrichment_score",
        "specificity_score",
        "effect_size",
        "z_score",
        "score",
        "value",
    ]
    score_col = next(
        (columns[token] for token in preferred_scores if token in columns),
        None,
    )

    path_token = normalize_token(str(path.relative_to(ROOT)))
    keyword_score = sum(2 for keyword in keywords if keyword in path_token)

    long_form = map_col is not None and category_col is not None
    wide_form = len(map_matches) >= 5 and category_col is not None
    if not long_form and not wide_form:
        return None

    return {
        "path": path,
        "kind": kind,
        "map_col": map_col,
        "category_col": category_col,
        "score_col": score_col,
        "wide_map_columns": map_matches,
        "matched_maps": len(map_matches),
        "format": "long" if long_form else "wide",
        "selection_score": (
            keyword_score
            + (20 if long_form else 10)
            + len(map_matches)
            + (5 if score_col is not None else 0)
        ),
    }


def discover_source(kind: str) -> dict[str, Any] | None:
    candidates = [
        item
        for path in candidate_files()
        if (item := header_candidate(path, kind)) is not None
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            item["selection_score"],
            item["matched_maps"],
            -item["path"].stat().st_size,
        ),
        reverse=True,
    )
    return candidates[0]


def numeric_score_column(frame: pd.DataFrame, excluded: set[str]) -> str | None:
    candidates = []
    for column in frame.columns:
        if column in excluded:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        coverage = numeric.notna().mean()
        if coverage >= 0.70:
            candidates.append((coverage, column))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def summarize_developmental(source: dict[str, Any] | None) -> tuple[pd.DataFrame, str]:
    columns = [
        "DTHI_map",
        "empirical_peak_developmental_window",
        "empirical_peak_window_score",
        "empirical_early_to_late_delta",
        "empirical_developmental_windows_observed",
        "developmental_empirical_source",
    ]
    if source is None:
        return pd.DataFrame(columns=columns), "not_found"

    frame = read_table(source["path"])
    if source["format"] == "long":
        map_col = source["map_col"]
        category_col = source["category_col"]
        score_col = source["score_col"] or numeric_score_column(
            frame, {map_col, category_col}
        )
        if score_col is None:
            return pd.DataFrame(columns=columns), "found_but_no_numeric_score"
        long = frame[[map_col, category_col, score_col]].copy()
        long.columns = ["raw_map", "developmental_window", "score"]
        long["DTHI_map"] = long["raw_map"].map(resolve_map)
    else:
        category_col = source["category_col"]
        wide_columns = source["wide_map_columns"]
        long = frame[[category_col] + wide_columns].melt(
            id_vars=[category_col],
            var_name="raw_map",
            value_name="score",
        )
        long = long.rename(columns={category_col: "developmental_window"})
        long["DTHI_map"] = long["raw_map"].map(resolve_map)

    long["score"] = pd.to_numeric(long["score"], errors="coerce")
    long = long.dropna(subset=["DTHI_map", "score"])
    if long.empty:
        return pd.DataFrame(columns=columns), "found_but_unusable"

    long["window_token"] = long["developmental_window"].map(normalize_token)
    long["window_order"] = long["window_token"].map(WINDOW_ORDER)
    rows = []
    for map_name, group in long.groupby("DTHI_map"):
        grouped = (
            group.groupby(["developmental_window", "window_token", "window_order"], dropna=False)["score"]
            .mean()
            .reset_index()
        )
        peak = grouped.loc[grouped["score"].idxmax()]
        ordered = grouped.dropna(subset=["window_order"]).sort_values("window_order")
        delta = np.nan
        if len(ordered) >= 2:
            delta = float(ordered.iloc[-1]["score"] - ordered.iloc[0]["score"])
        rows.append(
            {
                "DTHI_map": map_name,
                "empirical_peak_developmental_window": str(peak["developmental_window"]),
                "empirical_peak_window_score": float(peak["score"]),
                "empirical_early_to_late_delta": delta,
                "empirical_developmental_windows_observed": int(len(grouped)),
                "developmental_empirical_source": str(source["path"].relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows, columns=columns), "used"


def summarize_cell_type(source: dict[str, Any] | None) -> tuple[pd.DataFrame, str]:
    columns = [
        "DTHI_map",
        "empirical_top_cell_type",
        "empirical_top_cell_type_score",
        "empirical_cell_types_observed",
        "cell_type_empirical_source",
    ]
    if source is None:
        return pd.DataFrame(columns=columns), "not_found"

    frame = read_table(source["path"])
    if source["format"] == "long":
        map_col = source["map_col"]
        category_col = source["category_col"]
        score_col = source["score_col"] or numeric_score_column(
            frame, {map_col, category_col}
        )
        if score_col is None:
            return pd.DataFrame(columns=columns), "found_but_no_numeric_score"
        long = frame[[map_col, category_col, score_col]].copy()
        long.columns = ["raw_map", "cell_type", "score"]
        long["DTHI_map"] = long["raw_map"].map(resolve_map)
    else:
        category_col = source["category_col"]
        wide_columns = source["wide_map_columns"]
        long = frame[[category_col] + wide_columns].melt(
            id_vars=[category_col],
            var_name="raw_map",
            value_name="score",
        )
        long = long.rename(columns={category_col: "cell_type"})
        long["DTHI_map"] = long["raw_map"].map(resolve_map)

    long["score"] = pd.to_numeric(long["score"], errors="coerce")
    long = long.dropna(subset=["DTHI_map", "score"])
    if long.empty:
        return pd.DataFrame(columns=columns), "found_but_unusable"

    rows = []
    for map_name, group in long.groupby("DTHI_map"):
        grouped = group.groupby("cell_type", dropna=False)["score"].mean().reset_index()
        index = grouped["score"].abs().idxmax()
        top = grouped.loc[index]
        rows.append(
            {
                "DTHI_map": map_name,
                "empirical_top_cell_type": str(top["cell_type"]),
                "empirical_top_cell_type_score": float(top["score"]),
                "empirical_cell_types_observed": int(len(grouped)),
                "cell_type_empirical_source": str(source["path"].relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows, columns=columns), "used"


def source_record(
    layer: str,
    path: Path | None,
    status: str,
    locked: bool,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "evidence_layer": layer,
        "source_status": status,
        "locked_Phase8D3A_input": locked,
        "relative_path": str(path.relative_to(ROOT)) if path else "",
        "size_bytes": path.stat().st_size if path and path.is_file() else np.nan,
        "SHA256": sha256(path) if path and path.is_file() else "",
        "detail": detail,
    }


def main() -> None:
    validate_d3a()
    manifest = load_manifest()

    integrated_path = locked_path(
        manifest, "phase8D2C3_integrated_evidence.tsv"
    )
    priority_path = locked_path(
        manifest, "phase8D2C3_priority_associations.tsv"
    )
    stability_path = locked_path(
        manifest, "phase8D2C3_DTHI_map_stability_summary.tsv"
    )

    integrated = pd.read_csv(integrated_path, sep="\t")
    priority = pd.read_csv(priority_path, sep="\t")
    stability = pd.read_csv(stability_path, sep="\t")

    required_integrated = {
        "DTHI_map",
        "external_map",
        "external_family",
        "spearman_rho",
        "bijective_p_spin_two_sided",
        "bijective_q_BH_global160",
        "bijective_q_BH_family80",
        "bijective_p_maxT_global160",
        "bijective_p_maxT_family80",
    }
    missing = required_integrated - set(integrated.columns)
    if missing:
        raise ValidationError(
            f"Integrated evidence table is missing columns: {sorted(missing)}"
        )
    if len(integrated) != 160:
        raise ValidationError(
            f"Expected 160 integrated associations; found {len(integrated)}."
        )
    if set(integrated["DTHI_map"].unique()) != set(DTHI_ORDER):
        raise ValidationError("Integrated evidence does not contain the frozen 10-map set.")
    if len(priority) != 5:
        raise ValidationError(
            f"Expected five priority associations; found {len(priority)}."
        )

    developmental_source = discover_source("developmental")
    cell_source = discover_source("cell_type")
    developmental, developmental_status = summarize_developmental(developmental_source)
    cell_type, cell_status = summarize_cell_type(cell_source)

    association_rows = []
    master_rows = []

    stability_index = stability.set_index("DTHI_map")
    priority_by_map = {
        name: group.copy()
        for name, group in priority.groupby("DTHI_map")
    }

    for map_name in DTHI_ORDER:
        rows = integrated[integrated["DTHI_map"] == map_name].copy()
        cognitive = rows[rows["external_family"] == "cognitive"]
        clinical = rows[rows["external_family"] == "clinical"]
        if len(cognitive) != 8 or len(clinical) != 8:
            raise ValidationError(
                f"{map_name} does not have 8 cognitive and 8 clinical associations."
            )

        best_cognitive = best_association(cognitive)
        best_clinical = best_association(clinical)
        tier = evidence_tier(rows)

        priority_rows = priority_by_map.get(map_name, pd.DataFrame())
        priority_external = (
            ";".join(priority_rows["external_map"].astype(str).tolist())
            if not priority_rows.empty
            else ""
        )
        lodo_concordant = np.nan
        if not priority_rows.empty and "all_five_direction_concordant" in priority_rows.columns:
            lodo_concordant = bool(
                priority_rows["all_five_direction_concordant"].map(parse_bool).all()
            )

        stability_row = stability_index.loc[map_name]
        annotation = DEFINITION_ANNOTATIONS[map_name]

        association_rows.append(
            {
                "DTHI_map": map_name,
                "strongest_cognitive_map": best_cognitive["external_map"],
                "strongest_cognitive_rho": float(best_cognitive["spearman_rho"]),
                "strongest_cognitive_spin_p": float(best_cognitive["bijective_p_spin_two_sided"]),
                "strongest_clinical_map": best_clinical["external_map"],
                "strongest_clinical_rho": float(best_clinical["spearman_rho"]),
                "strongest_clinical_spin_p": float(best_clinical["bijective_p_spin_two_sided"]),
                "global_maxT_hits": int((rows["bijective_p_maxT_global160"] < 0.05).sum()),
                "family_maxT_hits": int((rows["bijective_p_maxT_family80"] < 0.05).sum()),
                "global_BH_hits": int((rows["bijective_q_BH_global160"] < 0.05).sum()),
                "family_BH_hits": int((rows["bijective_q_BH_family80"] < 0.05).sum()),
                "nominal_spin_hits": int((rows["bijective_p_spin_two_sided"] < 0.05).sum()),
                "final_evidence_tier": tier,
                "priority_association_count": int(len(priority_rows)),
                "priority_external_maps": priority_external,
            }
        )

        master_rows.append(
            {
                "DTHI_map": map_name,
                "map_type": annotation["map_type"],
                "annotation_basis": "frozen_module_definition",
                "developmental_role_definition": annotation["developmental_role"],
                "dominant_cell_systems_definition": annotation["dominant_cell_systems"],
                "biological_process_definition": annotation["biological_process"],
                "AHBA_minimum_LODO_vs_full_rho": float(
                    stability_row["minimum_LODO_vs_full_spearman"]
                ),
                "AHBA_median_LODO_vs_full_rho": float(
                    stability_row["median_LODO_vs_full_spearman"]
                ),
                "strongest_cognitive_map": best_cognitive["external_map"],
                "strongest_cognitive_rho": float(best_cognitive["spearman_rho"]),
                "strongest_clinical_map": best_clinical["external_map"],
                "strongest_clinical_rho": float(best_clinical["spearman_rho"]),
                "global_maxT_hits": int((rows["bijective_p_maxT_global160"] < 0.05).sum()),
                "family_maxT_hits": int((rows["bijective_p_maxT_family80"] < 0.05).sum()),
                "family_BH_hits": int((rows["bijective_q_BH_family80"] < 0.05).sum()),
                "priority_association_count": int(len(priority_rows)),
                "priority_external_maps": priority_external,
                "priority_LODO_5of5_direction": lodo_concordant,
                "cross_modal_evidence_tier": tier,
            }
        )

    association_summary = pd.DataFrame(association_rows)
    master = pd.DataFrame(master_rows)

    master = master.merge(developmental, on="DTHI_map", how="left")
    master = master.merge(cell_type, on="DTHI_map", how="left")

    master["developmental_empirical_layer_status"] = developmental_status
    master["cell_type_empirical_layer_status"] = cell_status
    master["cross_modal_convergence_class"] = np.select(
        [
            master["cross_modal_evidence_tier"].eq("Tier1_global_FWER")
            & master["priority_LODO_5of5_direction"].eq(True),
            master["cross_modal_evidence_tier"].isin(
                ["Tier2_family_FWER", "Tier2_family_FDR"]
            )
            & master["priority_LODO_5of5_direction"].eq(True),
        ],
        [
            "global_spatial_evidence_with_AHBA_donor_robustness",
            "family_spatial_evidence_with_AHBA_donor_robustness",
        ],
        default="descriptive_or_nominal_cross_modal_correspondence",
    )

    priority_hierarchy = priority.copy()
    priority_hierarchy["reporting_tier"] = np.select(
        [
            priority_hierarchy["bijective_p_maxT_global160"] < 0.05,
            priority_hierarchy["bijective_p_maxT_family80"] < 0.05,
            priority_hierarchy["bijective_q_BH_family80"] < 0.05,
        ],
        [
            "Tier1_global_FWER",
            "Tier2_family_FWER",
            "Tier2_family_FDR",
        ],
        default="Tier3_exploratory",
    )
    priority_hierarchy["cross_modal_interpretation"] = np.where(
        priority_hierarchy["reporting_tier"].eq("Tier1_global_FWER"),
        "primary_global_spatial_result_with_direction_stable_AHBA_LODO",
        "secondary_family_level_result_with_direction_stable_AHBA_LODO",
    )

    source_rows = [
        source_record(
            "Phase8D2C3_integrated_spatial_evidence",
            integrated_path,
            "used",
            True,
        ),
        source_record(
            "Phase8D2C3_priority_associations",
            priority_path,
            "used",
            True,
        ),
        source_record(
            "AHBA_LODO_map_stability",
            stability_path,
            "used",
            True,
        ),
        source_record(
            "developmental_trajectory_empirical_layer",
            developmental_source["path"] if developmental_source else None,
            developmental_status,
            False,
            (
                f"selection_score={developmental_source['selection_score']};"
                f"format={developmental_source['format']}"
                if developmental_source
                else "no compatible table discovered"
            ),
        ),
        source_record(
            "cell_type_empirical_layer",
            cell_source["path"] if cell_source else None,
            cell_status,
            False,
            (
                f"selection_score={cell_source['selection_score']};"
                f"format={cell_source['format']}"
                if cell_source
                else "no compatible table discovered"
            ),
        ),
    ]
    sources = pd.DataFrame(source_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_tsv(master, MASTER_OUT)
    atomic_tsv(association_summary, ASSOCIATION_OUT)
    atomic_tsv(priority_hierarchy, PRIORITY_OUT)
    atomic_tsv(sources, SOURCE_OUT)

    tier_counts = master["cross_modal_evidence_tier"].value_counts().to_dict()
    interpretation = f"""PHASE 8D3B CROSS-MODAL CONVERGENCE SYNTHESIS

The synthesis integrates ten frozen DTHI module maps with:
1. definition-based developmental and cell-system annotations;
2. AHBA five-donor leave-one-donor-out map stability;
3. exact-bijective cognitive-map spatial correspondence;
4. exact-bijective clinical-map spatial correspondence; and
5. the locked Phase 8D2C evidence hierarchy.

Primary result:
Synaptic assembly/receptor trafficking–ASD remains the sole Tier 1 global-FWER
association and is direction-concordant across all five AHBA donor omissions.

Secondary results:
Four clinical associations retain Tier 2 family-level support and five-of-five
LODO directional concordance. They are not equivalent to the global-FWER result.

Empirical developmental source status: {developmental_status}
Empirical cell-type source status: {cell_status}

The definition-based developmental and cell-system columns are annotations from
the frozen DTHI module definitions. They must not be described as empirical
cell-type enrichment unless the corresponding empirical source status is 'used'.

Tier counts across the ten DTHI maps:
{json.dumps(tier_counts, sort_keys=True)}

Interpretive constraint:
Cross-modal spatial correspondence does not demonstrate causality, direct
cellular equivalence, or disease mechanism. The LH46 analysis is a nested
sensitivity analysis rather than independent replication.
"""
    INTERPRETATION_OUT.write_text(interpretation, encoding="utf-8")

    numeric_columns = master.select_dtypes(include=[np.number]).columns
    all_numeric_finite = bool(
        np.isfinite(master[numeric_columns].to_numpy(dtype=float), where=~pd.isna(master[numeric_columns].to_numpy(dtype=float))).all()
    )

    audit = pd.DataFrame(
        [
            {
                "section": "upstream",
                "item": "Phase8D3A_completed",
                "value": True,
                "passed": True,
                "detail": str(D3A_COMPLETION.relative_to(ROOT)),
            },
            {
                "section": "locked_inputs",
                "item": "integrated_associations",
                "value": len(integrated),
                "passed": len(integrated) == 160,
                "detail": "10_DTHI_x_16_external_maps",
            },
            {
                "section": "locked_inputs",
                "item": "priority_associations",
                "value": len(priority),
                "passed": len(priority) == 5,
                "detail": "",
            },
            {
                "section": "synthesis",
                "item": "DTHI_maps_synthesized",
                "value": len(master),
                "passed": len(master) == 10,
                "detail": ";".join(DTHI_ORDER),
            },
            {
                "section": "synthesis",
                "item": "Tier1_global_FWER_maps",
                "value": int(
                    master["cross_modal_evidence_tier"].eq("Tier1_global_FWER").sum()
                ),
                "passed": int(
                    master["cross_modal_evidence_tier"].eq("Tier1_global_FWER").sum()
                ) == 1,
                "detail": "synaptic_assembly_receptor_trafficking",
            },
            {
                "section": "optional_empirical_layer",
                "item": "developmental_source_status",
                "value": developmental_status,
                "passed": True,
                "detail": "nonblocking_but_explicitly_recorded",
            },
            {
                "section": "optional_empirical_layer",
                "item": "cell_type_source_status",
                "value": cell_status,
                "passed": True,
                "detail": "nonblocking_but_explicitly_recorded",
            },
            {
                "section": "outputs",
                "item": "all_numeric_values_valid",
                "value": all_numeric_finite,
                "passed": all_numeric_finite,
                "detail": "NaN_allowed_only_for_unavailable_optional_layers",
            },
        ]
    )
    atomic_tsv(audit, AUDIT_OUT)

    all_passed = bool(audit["passed"].all())
    completion = pd.DataFrame(
        [
            {
                "Phase8D3A_status_confirmed": True,
                "DTHI_maps_synthesized": len(master),
                "locked_spatial_associations_integrated": len(integrated),
                "priority_associations_integrated": len(priority),
                "cross_modal_core_layers": 5,
                "developmental_empirical_layer_status": developmental_status,
                "cell_type_empirical_layer_status": cell_status,
                "Tier1_global_FWER_maps": int(
                    master["cross_modal_evidence_tier"].eq("Tier1_global_FWER").sum()
                ),
                "Tier2_family_supported_maps": int(
                    master["cross_modal_evidence_tier"].isin(
                        ["Tier2_family_FWER", "Tier2_family_FDR"]
                    ).sum()
                ),
                "all_cross_modal_audits_passed": all_passed,
                "ready_for_phase8D3C_evidence_hierarchy": all_passed,
                "Phase8D3B_status": "completed" if all_passed else "failed",
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
            }
        ]
    )
    atomic_tsv(completion, COMPLETION_OUT)

    print("===== PHASE 8D3B COMPLETION =====")
    print(completion.to_string(index=False))

    print("\n===== CROSS-MODAL MASTER MATRIX =====")
    display_columns = [
        "DTHI_map",
        "cross_modal_evidence_tier",
        "strongest_cognitive_map",
        "strongest_cognitive_rho",
        "strongest_clinical_map",
        "strongest_clinical_rho",
        "priority_association_count",
        "AHBA_minimum_LODO_vs_full_rho",
        "cross_modal_convergence_class",
    ]
    print(master[display_columns].to_string(index=False))

    print("\n===== SOURCE PROVENANCE =====")
    print(sources.to_string(index=False))

    print("\n===== AUDIT =====")
    print(audit.to_string(index=False))

    print("\n===== CANONICAL OUTPUT HASHES =====")
    outputs = [
        MASTER_OUT,
        ASSOCIATION_OUT,
        PRIORITY_OUT,
        SOURCE_OUT,
        INTERPRETATION_OUT,
        AUDIT_OUT,
        COMPLETION_OUT,
    ]
    hashes = pd.DataFrame(
        [
            {
                "relative_path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "SHA256": sha256(path),
            }
            for path in outputs
        ]
    )
    print(hashes.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"Phase 8D3B failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise
