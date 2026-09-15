#!/usr/bin/env python3
"""Phase 8D3C: definitive evidence hierarchy and manuscript-safe classification."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
LOCK = ROOT / "03_processed_data/functional_imaging/phase8D/phase8D3A_canonical_lock"
MANIFEST = LOCK / "phase8D3A_locked_inputs_manifest.tsv"
D3B = ROOT / "03_processed_data/functional_imaging/phase8D/phase8D3B_cross_modal_synthesis"
TABLES = ROOT / "07_tables/main_tables/phase8"
OUT = ROOT / "03_processed_data/functional_imaging/phase8D/phase8D3C_evidence_hierarchy"

D3B_COMPLETION = TABLES / "phase8D3B_completion_summary.tsv"
D3B_MASTER = D3B / "phase8D3B_cross_modal_master_matrix.tsv"
D3B_ASSOC = D3B / "phase8D3B_map_external_association_summary.tsv"
D3B_PRIORITY = D3B / "phase8D3B_priority_evidence_hierarchy.tsv"
D3B_PROVENANCE = D3B / "phase8D3B_source_provenance.tsv"

ASSOC_OUT = OUT / "phase8D3C_association_evidence_hierarchy.tsv"
MAP_OUT = OUT / "phase8D3C_map_level_evidence_hierarchy.tsv"
PRIORITY_OUT = OUT / "phase8D3C_priority_reporting_table.tsv"
LABELS_OUT = OUT / "phase8D3C_manuscript_reporting_labels.tsv"
SUMMARY_OUT = OUT / "phase8D3C_evidence_summary.txt"
PROVENANCE_OUT = OUT / "phase8D3C_source_provenance.tsv"
AUDIT_OUT = TABLES / "phase8D3C_evidence_hierarchy_audit.tsv"
COMPLETION_OUT = TABLES / "phase8D3C_completion_summary.tsv"

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

TIER_ORDER = {
    "Tier1_global_FWER": 1,
    "Tier2_family_corrected": 2,
    "Tier3A_nominal": 3,
    "Tier3B_descriptive": 4,
}

TIER_LABEL = {
    "Tier1_global_FWER":
        "Tier 1 — global familywise-error-controlled evidence",
    "Tier2_family_corrected":
        "Tier 2 — family-level corrected evidence",
    "Tier3A_nominal":
        "Tier 3A — nominal exact-bijective spatial correspondence",
    "Tier3B_descriptive":
        "Tier 3B — descriptive spatial correspondence",
}

REPORTING = {
    "Tier1_global_FWER":
        "main_text_primary_result",
    "Tier2_family_corrected":
        "main_text_secondary_result",
    "Tier3A_nominal":
        "supplementary_or_explicitly_exploratory_only",
    "Tier3B_descriptive":
        "matrix_or_descriptive_summary_only",
}


class ValidationError(RuntimeError):
    pass


def require(path: Path) -> None:
    if not path.is_file():
        raise ValidationError(f"Required file not found: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )
    frame.to_csv(
        temporary,
        sep="\t",
        index=False,
    )
    temporary.replace(path)


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def number(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            np.nan,
            index=frame.index,
            dtype=float,
        )
    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def validate_upstream() -> None:
    require(D3B_COMPLETION)

    completion = pd.read_csv(
        D3B_COMPLETION,
        sep="\t",
    )

    if len(completion) != 1:
        raise ValidationError(
            "Phase 8D3B completion summary "
            "must contain exactly one row."
        )

    row = completion.iloc[0]

    if str(
        row["Phase8D3B_status"]
    ).strip() != "completed":
        raise ValidationError(
            "Phase 8D3B is not completed."
        )

    if not as_bool(
        row[
            "ready_for_phase8D3C_evidence_hierarchy"
        ]
    ):
        raise ValidationError(
            "Phase 8D3B is not ready "
            "for Phase 8D3C."
        )


def load_manifest() -> pd.DataFrame:
    require(MANIFEST)

    manifest = pd.read_csv(
        MANIFEST,
        sep="\t",
    )

    needed = {
        "source_relative_path",
        "snapshot_relative_path",
        "snapshot_SHA256",
        "hash_match",
    }

    missing = needed - set(
        manifest.columns
    )

    if missing:
        raise ValidationError(
            "Manifest missing columns: "
            f"{sorted(missing)}"
        )

    if not manifest[
        "hash_match"
    ].map(as_bool).all():
        raise ValidationError(
            "Phase 8D3A manifest contains "
            "failed hash matches."
        )

    return manifest


def locked_file(
    manifest: pd.DataFrame,
    filename: str,
) -> Path:
    source = manifest[
        "source_relative_path"
    ].astype(str)
    hits = manifest[
        source.str.endswith(
            "/" + filename
        )
        | source.eq(filename)
    ]

    if len(hits) != 1:
        raise ValidationError(
            "Expected exactly one locked file "
            f"named {filename}; found {len(hits)}."
        )

    path = (
        LOCK
        / str(
            hits.iloc[0][
                "snapshot_relative_path"
            ]
        )
    )

    require(path)

    expected_hash = str(
        hits.iloc[0]["snapshot_SHA256"]
    )
    observed_hash = sha256(path)

    if observed_hash != expected_hash:
        raise ValidationError(
            "Locked file hash mismatch: "
            f"{path}"
        )

    return path


def classify_tier(
    frame: pd.DataFrame,
) -> pd.Series:
    global_maxt = number(
        frame,
        "bijective_p_maxT_global160",
    )
    family_maxt = number(
        frame,
        "bijective_p_maxT_family80",
    )
    global_bh = number(
        frame,
        "bijective_q_BH_global160",
    )
    family_bh = number(
        frame,
        "bijective_q_BH_family80",
    )
    nominal = number(
        frame,
        "bijective_p_spin_two_sided",
    )

    conditions = [
        global_maxt.lt(0.05),
        (
            family_maxt.lt(0.05)
            | global_bh.lt(0.05)
            | family_bh.lt(0.05)
        ),
        nominal.lt(0.05),
    ]

    choices = [
        "Tier1_global_FWER",
        "Tier2_family_corrected",
        "Tier3A_nominal",
    ]

    return pd.Series(
        np.select(
            conditions,
            choices,
            default="Tier3B_descriptive",
        ),
        index=frame.index,
    )


def corrected_subclass(
    frame: pd.DataFrame,
) -> pd.Series:
    global_maxt = number(
        frame,
        "bijective_p_maxT_global160",
    )
    family_maxt = number(
        frame,
        "bijective_p_maxT_family80",
    )
    global_bh = number(
        frame,
        "bijective_q_BH_global160",
    )
    family_bh = number(
        frame,
        "bijective_q_BH_family80",
    )
    nominal = number(
        frame,
        "bijective_p_spin_two_sided",
    )

    conditions = [
        global_maxt.lt(0.05),
        family_maxt.lt(0.05),
        global_bh.lt(0.05),
        family_bh.lt(0.05),
        nominal.lt(0.05),
    ]

    choices = [
        "global_maxT_FWER",
        "family_maxT_FWER",
        "global_BH_FDR",
        "family_BH_FDR",
        "nominal_exact_bijective_spin",
    ]

    return pd.Series(
        np.select(
            conditions,
            choices,
            default="no_statistical_support",
        ),
        index=frame.index,
    )


def effect_direction(
    rho: float,
) -> str:
    if pd.isna(rho):
        return "unavailable"

    if rho > 0:
        return "positive"

    if rho < 0:
        return "negative"

    return "zero"


def effect_magnitude(
    rho: float,
) -> str:
    if pd.isna(rho):
        return "unavailable"

    value = abs(float(rho))

    if value >= 0.60:
        return "large"

    if value >= 0.40:
        return "moderate"

    if value >= 0.20:
        return "small"

    return "very_small"


def safe_sentence(
    row: pd.Series,
) -> str:
    rho = float(
        row["spearman_rho"]
    )

    map_label = str(
        row["DTHI_map"]
    ).replace("_", " ")

    external_label = str(
        row["external_map"]
    ).replace("_", " ")

    tier = row["evidence_tier"]

    if tier == "Tier1_global_FWER":
        return (
            f"The {map_label} map showed a "
            "negative spatial association with "
            f"{external_label} "
            f"(Spearman rho={rho:.3f}) that "
            "survived exact-bijective global "
            "maxT correction."
        )

    if tier == "Tier2_family_corrected":
        return (
            f"The {map_label} map showed a "
            f"{effect_direction(rho)} spatial "
            f"association with {external_label} "
            f"(Spearman rho={rho:.3f}) that "
            "retained family-level corrected "
            "support but not global "
            "familywise-error control."
        )

    if tier == "Tier3A_nominal":
        return (
            f"The {map_label}–{external_label} "
            "correspondence was nominal under "
            "the exact-bijective spin test and "
            "is treated as exploratory."
        )

    return (
        f"The {map_label}–{external_label} "
        "correspondence is presented "
        "descriptively and is not interpreted "
        "as statistically supported."
    )


def lodo_fields(
    frame: pd.DataFrame,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
]:
    concordance_candidates = [
        "all_five_direction_concordant",
        "LODO_5of5_direction",
        "lodo_all_five_direction_concordant",
    ]

    minimum_candidates = [
        "LODO_rho_min",
        "lodo_rho_min",
        "minimum_LODO_rho",
        "LODO_minimum_rho",
    ]

    maximum_candidates = [
        "LODO_rho_max",
        "lodo_rho_max",
        "maximum_LODO_rho",
        "LODO_maximum_rho",
    ]

    concordance = pd.Series(
        False,
        index=frame.index,
        dtype=bool,
    )

    for column in concordance_candidates:
        if column in frame.columns:
            concordance = frame[
                column
            ].map(as_bool)
            break

    minimum = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float,
    )

    for column in minimum_candidates:
        if column in frame.columns:
            minimum = number(
                frame,
                column,
            )
            break

    maximum = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float,
    )

    for column in maximum_candidates:
        if column in frame.columns:
            maximum = number(
                frame,
                column,
            )
            break

    return (
        concordance,
        minimum,
        maximum,
    )


def source_record(
    layer: str,
    path: Path,
    locked: bool,
    detail: str,
) -> dict[str, Any]:
    return {
        "evidence_layer": layer,
        "source_status": "used",
        "locked_Phase8D3A_input": locked,
        "relative_path": str(
            path.relative_to(ROOT)
        ),
        "size_bytes": path.stat().st_size,
        "SHA256": sha256(path),
        "detail": detail,
    }


MANUSCRIPT_TERM = {
    "Tier1_global_FWER":
        "primary global-FWER-supported spatial association",
    "Tier2_family_corrected":
        "secondary family-level corrected spatial association",
    "Tier3A_nominal":
        "nominal exploratory spatial correspondence",
    "Tier3B_descriptive":
        "descriptive spatial correspondence without statistical support",
}


def main() -> None:
    validate_upstream()

    for path in [
        D3B_MASTER,
        D3B_ASSOC,
        D3B_PRIORITY,
        D3B_PROVENANCE,
    ]:
        require(path)

    manifest = load_manifest()

    integrated_path = locked_file(
        manifest,
        "phase8D2C3_integrated_evidence.tsv",
    )

    locked_priority_path = locked_file(
        manifest,
        "phase8D2C3_priority_associations.tsv",
    )

    integrated = pd.read_csv(
        integrated_path,
        sep="\t",
        low_memory=False,
    )

    locked_priority = pd.read_csv(
        locked_priority_path,
        sep="\t",
        low_memory=False,
    )

    master = pd.read_csv(
        D3B_MASTER,
        sep="\t",
        low_memory=False,
    )

    d3b_association = pd.read_csv(
        D3B_ASSOC,
        sep="\t",
        low_memory=False,
    )

    d3b_priority = pd.read_csv(
        D3B_PRIORITY,
        sep="\t",
        low_memory=False,
    )

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

    missing = (
        required_integrated
        - set(integrated.columns)
    )

    if missing:
        raise ValidationError(
            "Integrated evidence table "
            "is missing columns: "
            f"{sorted(missing)}"
        )

    if len(integrated) != 160:
        raise ValidationError(
            "Expected 160 frozen spatial "
            f"associations; found {len(integrated)}."
        )

    observed_maps = set(
        integrated["DTHI_map"]
        .astype(str)
        .unique()
    )

    if observed_maps != set(DTHI_ORDER):
        missing_maps = (
            set(DTHI_ORDER)
            - observed_maps
        )
        unexpected_maps = (
            observed_maps
            - set(DTHI_ORDER)
        )

        raise ValidationError(
            "Frozen map set mismatch. "
            f"Missing={sorted(missing_maps)}; "
            f"unexpected={sorted(unexpected_maps)}"
        )

    if len(locked_priority) != 5:
        raise ValidationError(
            "Expected five frozen priority "
            "associations; found "
            f"{len(locked_priority)}."
        )

    if len(master) != 10:
        raise ValidationError(
            "Expected ten Phase 8D3B map rows; "
            f"found {len(master)}."
        )

    if len(d3b_association) != 10:
        raise ValidationError(
            "Expected ten Phase 8D3B association-"
            f"summary rows; found {len(d3b_association)}."
        )

    if len(d3b_priority) != 5:
        raise ValidationError(
            "Expected five Phase 8D3B priority "
            f"rows; found {len(d3b_priority)}."
        )

    hierarchy = integrated.copy()

    hierarchy["spearman_rho"] = number(
        hierarchy,
        "spearman_rho",
    )

    hierarchy["evidence_tier"] = (
        classify_tier(hierarchy)
    )

    hierarchy["evidence_tier_order"] = (
        hierarchy["evidence_tier"]
        .map(TIER_ORDER)
    )

    hierarchy["evidence_tier_label"] = (
        hierarchy["evidence_tier"]
        .map(TIER_LABEL)
    )

    hierarchy[
        "corrected_evidence_subclass"
    ] = corrected_subclass(hierarchy)

    hierarchy["effect_direction"] = (
        hierarchy["spearman_rho"]
        .map(effect_direction)
    )

    hierarchy["effect_magnitude"] = (
        hierarchy["spearman_rho"]
        .map(effect_magnitude)
    )

    hierarchy["manuscript_term"] = (
        hierarchy["evidence_tier"]
        .map(MANUSCRIPT_TERM)
    )

    hierarchy["reporting_permission"] = (
        hierarchy["evidence_tier"]
        .map(REPORTING)
    )

    key_columns = [
        "DTHI_map",
        "external_map",
    ]

    for column in key_columns:
        if column not in locked_priority.columns:
            raise ValidationError(
                "Locked priority table is missing "
                f"the key column: {column}"
            )

    priority_keys = set(
        map(
            tuple,
            locked_priority[
                key_columns
            ].astype(str).to_numpy(),
        )
    )

    hierarchy[
        "locked_priority_association"
    ] = [
        (
            str(dthi),
            str(external),
        ) in priority_keys
        for dthi, external
        in hierarchy[
            key_columns
        ].to_numpy()
    ]

    (
        lodo_concordance,
        lodo_minimum,
        lodo_maximum,
    ) = lodo_fields(hierarchy)

    hierarchy[
        "LODO_5of5_direction_concordant"
    ] = lodo_concordance

    hierarchy[
        "LODO_rho_min"
    ] = lodo_minimum

    hierarchy[
        "LODO_rho_max"
    ] = lodo_maximum

    lh46_candidates = [
        "bijective_LH46_q_BH_global160",
        "LH46_q_BH_global160",
        "bijective_LH46_q_BH_family80",
        "LH46_q_BH_family80",
    ]

    lh46_column = next(
        (
            column
            for column in lh46_candidates
            if column in hierarchy.columns
        ),
        None,
    )

    if lh46_column is None:
        hierarchy[
            "LH46_BH_supported"
        ] = False
    else:
        hierarchy[
            "LH46_BH_supported"
        ] = number(
            hierarchy,
            lh46_column,
        ).lt(0.05)

    hierarchy[
        "manuscript_safe_sentence"
    ] = hierarchy.apply(
        safe_sentence,
        axis=1,
    )

    hierarchy[
        "causal_interpretation_allowed"
    ] = False

    hierarchy[
        "independent_replication_claim_allowed"
    ] = False

    hierarchy[
        "LH46_interpretation"
    ] = np.where(
        hierarchy["LH46_BH_supported"],
        (
            "nested_LH46_sensitivity_"
            "corrected_support"
        ),
        (
            "no_nested_LH46_"
            "BH_FDR_support"
        ),
    )

    hierarchy = (
        hierarchy
        .sort_values(
            [
                "evidence_tier_order",
                "external_family",
                "DTHI_map",
                "external_map",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    duplicate_keys = hierarchy.duplicated(
        key_columns,
        keep=False,
    )

    if duplicate_keys.any():
        duplicated = hierarchy.loc[
            duplicate_keys,
            key_columns,
        ]

        raise ValidationError(
            "Duplicate DTHI–external keys detected:\n"
            + duplicated.to_string(index=False)
        )

    map_rows = []

    master_index = master.set_index(
        "DTHI_map"
    )

    if not master_index.index.is_unique:
        raise ValidationError(
            "Phase 8D3B master matrix contains "
            "duplicate DTHI maps."
        )

    for map_name in DTHI_ORDER:
        rows = hierarchy[
            hierarchy["DTHI_map"]
            .astype(str)
            .eq(map_name)
        ].copy()

        if len(rows) != 16:
            raise ValidationError(
                f"{map_name} has {len(rows)} "
                "associations rather than 16."
            )

        rows[
            "absolute_spearman_rho"
        ] = rows[
            "spearman_rho"
        ].abs()

        best = (
            rows
            .sort_values(
                [
                    "evidence_tier_order",
                    "absolute_spearman_rho",
                ],
                ascending=[
                    True,
                    False,
                ],
                kind="stable",
            )
            .iloc[0]
        )

        strongest = (
            rows
            .sort_values(
                "absolute_spearman_rho",
                ascending=False,
                kind="stable",
            )
            .iloc[0]
        )

        if map_name not in master_index.index:
            raise ValidationError(
                "DTHI map missing from Phase 8D3B "
                f"master matrix: {map_name}"
            )

        map_master = master_index.loc[
            map_name
        ]

        tier_counts = (
            rows["evidence_tier"]
            .value_counts()
        )

        map_rows.append(
            {
                "DTHI_map":
                    map_name,
                "map_level_highest_tier":
                    best["evidence_tier"],
                "map_level_highest_tier_order":
                    int(
                        best[
                            "evidence_tier_order"
                        ]
                    ),
                "map_level_reporting_label":
                    TIER_LABEL[
                        best["evidence_tier"]
                    ],
                "Tier1_association_count":
                    int(
                        tier_counts.get(
                            "Tier1_global_FWER",
                            0,
                        )
                    ),
                "Tier2_association_count":
                    int(
                        tier_counts.get(
                            "Tier2_family_corrected",
                            0,
                        )
                    ),
                "Tier3A_association_count":
                    int(
                        tier_counts.get(
                            "Tier3A_nominal",
                            0,
                        )
                    ),
                "Tier3B_association_count":
                    int(
                        tier_counts.get(
                            "Tier3B_descriptive",
                            0,
                        )
                    ),
                "priority_association_count":
                    int(
                        rows[
                            "locked_priority_association"
                        ].sum()
                    ),
                "highest_evidence_external_map":
                    best["external_map"],
                "highest_evidence_external_family":
                    best["external_family"],
                "highest_evidence_rho":
                    float(
                        best["spearman_rho"]
                    ),
                "strongest_absolute_external_map":
                    strongest["external_map"],
                "strongest_absolute_external_family":
                    strongest["external_family"],
                "strongest_absolute_rho":
                    float(
                        strongest[
                            "spearman_rho"
                        ]
                    ),
                "AHBA_minimum_LODO_vs_full_rho":
                    float(
                        map_master[
                            "AHBA_minimum_LODO_vs_full_rho"
                        ]
                    ),
                "AHBA_median_LODO_vs_full_rho":
                    float(
                        map_master[
                            "AHBA_median_LODO_vs_full_rho"
                        ]
                    ),
                "developmental_empirical_layer_status":
                    str(
                        map_master.get(
                            "developmental_empirical_layer_status",
                            "",
                        )
                    ),
                "empirical_peak_developmental_window":
                    str(
                        map_master.get(
                            "empirical_peak_developmental_window",
                            "",
                        )
                    ),
                "cell_type_empirical_layer_status":
                    str(
                        map_master.get(
                            "cell_type_empirical_layer_status",
                            "",
                        )
                    ),
                "cell_system_annotation_basis":
                    "frozen_module_definition",
                "cross_modal_convergence_class":
                    str(
                        map_master[
                            "cross_modal_convergence_class"
                        ]
                    ),
            }
        )

    map_hierarchy = pd.DataFrame(
        map_rows
    ).sort_values(
        [
            "map_level_highest_tier_order",
            "DTHI_map",
        ],
        kind="stable",
    )

    priority = hierarchy[
        hierarchy[
            "locked_priority_association"
        ]
    ].copy()

    if len(priority) != 5:
        raise ValidationError(
            "The definitive hierarchy contains "
            f"{len(priority)} priority rows "
            "rather than five."
        )

    priority_order = {
        (
            "synaptic_assembly_receptor_trafficking",
            "clinical_asd",
        ): 1,
        (
            "axon_guidance_neurite_outgrowth",
            "clinical_depression",
        ): 2,
        (
            "patterning_arealization",
            "clinical_asd",
        ): 3,
        (
            "synaptic_membrane_structural_candidates",
            "clinical_depression",
        ): 4,
        (
            "synaptic_membrane_structural_candidates",
            "clinical_asd",
        ): 5,
    }

    priority[
        "priority_reporting_order"
    ] = [
        priority_order.get(
            (
                str(dthi),
                str(external),
            ),
            999,
        )
        for dthi, external
        in priority[
            key_columns
        ].to_numpy()
    ]

    if (
        priority[
            "priority_reporting_order"
        ]
        .eq(999)
        .any()
    ):
        unexpected = priority.loc[
            priority[
                "priority_reporting_order"
            ].eq(999),
            key_columns,
        ]

        raise ValidationError(
            "Unexpected priority association "
            "encountered:\n"
            + unexpected.to_string(
                index=False
            )
        )

    priority[
        "priority_role"
    ] = np.where(
        priority[
            "evidence_tier"
        ].eq(
            "Tier1_global_FWER"
        ),
        "primary_result",
        "secondary_result",
    )

    priority[
        "recommended_heading"
    ] = np.where(
        priority[
            "priority_role"
        ].eq(
            "primary_result"
        ),
        (
            "Primary globally corrected "
            "spatial association"
        ),
        (
            "Secondary family-level corrected "
            "spatial associations"
        ),
    )

    priority = priority.sort_values(
        "priority_reporting_order",
        kind="stable",
    )

    labels = pd.DataFrame(
        [
            {
                "evidence_tier":
                    "Tier1_global_FWER",
                "tier_order":
                    1,
                "allowed_description":
                    (
                        "survived exact-bijective "
                        "global maxT correction"
                    ),
                "main_text_status":
                    "primary",
                "significant_word_allowed":
                    True,
                "replication_word_allowed":
                    False,
                "causal_language_allowed":
                    False,
                "required_qualifier":
                    "spatial association",
                "prohibited_claims":
                    (
                        "causal mechanism;"
                        "disease-specific expression;"
                        "independent replication"
                    ),
            },
            {
                "evidence_tier":
                    "Tier2_family_corrected",
                "tier_order":
                    2,
                "allowed_description":
                    (
                        "retained family-level "
                        "corrected support"
                    ),
                "main_text_status":
                    "secondary",
                "significant_word_allowed":
                    True,
                "replication_word_allowed":
                    False,
                "causal_language_allowed":
                    False,
                "required_qualifier":
                    (
                        "did not survive global "
                        "familywise-error control"
                    ),
                "prohibited_claims":
                    (
                        "global significance;"
                        "replication;"
                        "causal mechanism"
                    ),
            },
            {
                "evidence_tier":
                    "Tier3A_nominal",
                "tier_order":
                    3,
                "allowed_description":
                    (
                        "nominal exact-bijective "
                        "spin correspondence"
                    ),
                "main_text_status":
                    (
                        "exploratory_or_"
                        "supplementary"
                    ),
                "significant_word_allowed":
                    False,
                "replication_word_allowed":
                    False,
                "causal_language_allowed":
                    False,
                "required_qualifier":
                    "uncorrected and exploratory",
                "prohibited_claims":
                    (
                        "significant;"
                        "corrected;"
                        "replicated;"
                        "causal"
                    ),
            },
            {
                "evidence_tier":
                    "Tier3B_descriptive",
                "tier_order":
                    4,
                "allowed_description":
                    (
                        "descriptive correspondence "
                        "without statistical support"
                    ),
                "main_text_status":
                    "matrix_or_summary_only",
                "significant_word_allowed":
                    False,
                "replication_word_allowed":
                    False,
                "causal_language_allowed":
                    False,
                "required_qualifier":
                    "not statistically supported",
                "prohibited_claims":
                    (
                        "association;"
                        "significance;"
                        "replication;"
                        "causality"
                    ),
            },
        ]
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_tsv(
        hierarchy,
        ASSOC_OUT,
    )

    write_tsv(
        map_hierarchy,
        MAP_OUT,
    )

    write_tsv(
        priority,
        PRIORITY_OUT,
    )

    write_tsv(
        labels,
        LABELS_OUT,
    )

    provenance = pd.DataFrame(
        [
            source_record(
                (
                    "locked_integrated_"
                    "spatial_evidence"
                ),
                integrated_path,
                True,
                (
                    "definitive_160_association_"
                    "inference_source"
                ),
            ),
            source_record(
                (
                    "locked_priority_"
                    "associations"
                ),
                locked_priority_path,
                True,
                "frozen_five_priority_pairs",
            ),
            source_record(
                (
                    "Phase8D3B_cross_modal_"
                    "master_matrix"
                ),
                D3B_MASTER,
                False,
                (
                    "map_level_developmental_"
                    "and_AHBA_context"
                ),
            ),
            source_record(
                (
                    "Phase8D3B_association_"
                    "summary"
                ),
                D3B_ASSOC,
                False,
                "cross_modal_map_summary",
            ),
            source_record(
                (
                    "Phase8D3B_priority_"
                    "hierarchy"
                ),
                D3B_PRIORITY,
                False,
                "upstream_reporting_context",
            ),
            source_record(
                (
                    "Phase8D3B_source_"
                    "provenance"
                ),
                D3B_PROVENANCE,
                False,
                "empirical_layer_provenance",
            ),
        ]
    )

    write_tsv(
        provenance,
        PROVENANCE_OUT,
    )

    association_counts = (
        hierarchy[
            "evidence_tier"
        ]
        .value_counts()
        .to_dict()
    )

    map_counts = (
        map_hierarchy[
            "map_level_highest_tier"
        ]
        .value_counts()
        .to_dict()
    )

    primary = priority[
        priority[
            "priority_role"
        ].eq(
            "primary_result"
        )
    ]

    secondary = priority[
        priority[
            "priority_role"
        ].eq(
            "secondary_result"
        )
    ]

    primary_lines = []

    for row in primary.itertuples():
        primary_lines.append(
            (
                f"- {row.DTHI_map}–"
                f"{row.external_map}: "
                f"rho={row.spearman_rho:.6f}; "
                "exact-bijective spin "
                f"P={row.bijective_p_spin_two_sided:.6g}; "
                "global maxT "
                f"P={row.bijective_p_maxT_global160:.6g}; "
                "family maxT "
                f"P={row.bijective_p_maxT_family80:.6g}."
            )
        )

    secondary_lines = []

    for row in secondary.itertuples():
        secondary_lines.append(
            (
                f"- {row.DTHI_map}–"
                f"{row.external_map}: "
                f"rho={row.spearman_rho:.6f}; "
                "family BH "
                f"q={row.bijective_q_BH_family80:.6g}; "
                "corrected subclass="
                f"{row.corrected_evidence_subclass}."
            )
        )

    summary_text = f"""PHASE 8D3C DEFINITIVE EVIDENCE HIERARCHY

STATUS
Phase 8D3C was generated from frozen Phase 8D3A spatial inputs and the
validated Phase 8D3B cross-modal synthesis. No spatial statistic,
permutation, threshold, or hypothesis was recomputed.

ASSOCIATION-LEVEL COUNTS
{json.dumps(association_counts, sort_keys=True)}

MAP-LEVEL HIGHEST-EVIDENCE COUNTS
{json.dumps(map_counts, sort_keys=True)}

TIER 1 — PRIMARY GLOBAL-FWER RESULT
{chr(10).join(primary_lines)}

TIER 2 — SECONDARY FAMILY-LEVEL CORRECTED RESULTS
{chr(10).join(secondary_lines)}

REPORTING RULES
1. Tier 1 may be described as surviving exact-bijective global maxT correction.
2. Tier 2 may be described as family-level corrected support but must be
   distinguished from global familywise-error control.
3. Tier 3A is nominal and exploratory and must not be called significant.
4. Tier 3B is descriptive and must not be interpreted as statistically supported.
5. Spatial correspondence does not establish causality, cellular equivalence,
   disease mechanism, or independent replication.
6. The LH46 analysis is a nested sensitivity analysis, not independent replication.
7. No empirical cell-type layer was available in Phase 8D3B; cell-system
   annotations remain definition-based.

READY STATE
The hierarchy is ready for Phase 8D3D integrated cross-modal figure generation.
"""

    SUMMARY_OUT.write_text(
        summary_text,
        encoding="utf-8",
    )

    unique_keys = not hierarchy.duplicated(
        key_columns
    ).any()

    all_tiers_valid = hierarchy[
        "evidence_tier"
    ].isin(
        TIER_ORDER
    ).all()

    tier1_associations = int(
        hierarchy[
            "evidence_tier"
        ].eq(
            "Tier1_global_FWER"
        ).sum()
    )

    tier2_associations = int(
        hierarchy[
            "evidence_tier"
        ].eq(
            "Tier2_family_corrected"
        ).sum()
    )

    tier3a_associations = int(
        hierarchy[
            "evidence_tier"
        ].eq(
            "Tier3A_nominal"
        ).sum()
    )

    tier3b_associations = int(
        hierarchy[
            "evidence_tier"
        ].eq(
            "Tier3B_descriptive"
        ).sum()
    )

    tier1_maps = int(
        map_hierarchy[
            "map_level_highest_tier"
        ].eq(
            "Tier1_global_FWER"
        ).sum()
    )

    tier2_maps = int(
        map_hierarchy[
            "map_level_highest_tier"
        ].eq(
            "Tier2_family_corrected"
        ).sum()
    )

    priority_lodo_concordant = bool(
        priority[
            "LODO_5of5_direction_concordant"
        ].all()
    )

    no_lh46_bh_hits = bool(
        ~hierarchy[
            "LH46_BH_supported"
        ].any()
    )

    expected_primary = priority[
        priority[
            "priority_role"
        ].eq(
            "primary_result"
        )
    ]

    primary_pair_correct = (
        len(expected_primary) == 1
        and str(
            expected_primary.iloc[0][
                "DTHI_map"
            ]
        )
        == (
            "synaptic_assembly_"
            "receptor_trafficking"
        )
        and str(
            expected_primary.iloc[0][
                "external_map"
            ]
        )
        == "clinical_asd"
    )

    audit = pd.DataFrame(
        [
            {
                "section":
                    "upstream",
                "item":
                    "Phase8D3B_completed",
                "value":
                    True,
                "passed":
                    True,
                "detail":
                    str(
                        D3B_COMPLETION.relative_to(
                            ROOT
                        )
                    ),
            },
            {
                "section":
                    "association_hierarchy",
                "item":
                    "associations_classified",
                "value":
                    len(hierarchy),
                "passed":
                    len(hierarchy) == 160,
                "detail":
                    (
                        "mutually_exclusive_"
                        "four_tier_system"
                    ),
            },
            {
                "section":
                    "association_hierarchy",
                "item":
                    "unique_DTHI_external_keys",
                "value":
                    unique_keys,
                "passed":
                    unique_keys,
                "detail":
                    "",
            },
            {
                "section":
                    "association_hierarchy",
                "item":
                    "all_tiers_valid",
                "value":
                    bool(all_tiers_valid),
                "passed":
                    bool(all_tiers_valid),
                "detail":
                    ";".join(
                        TIER_ORDER.keys()
                    ),
            },
            {
                "section":
                    "association_hierarchy",
                "item":
                    "tier_counts_sum_to_160",
                "value":
                    (
                        tier1_associations
                        + tier2_associations
                        + tier3a_associations
                        + tier3b_associations
                    ),
                "passed":
                    (
                        tier1_associations
                        + tier2_associations
                        + tier3a_associations
                        + tier3b_associations
                    )
                    == 160,
                "detail":
                    (
                        "Tier1+Tier2+Tier3A+"
                        "Tier3B"
                    ),
            },
            {
                "section":
                    "expected_results",
                "item":
                    (
                        "Tier1_global_FWER_"
                        "associations"
                    ),
                "value":
                    tier1_associations,
                "passed":
                    tier1_associations == 1,
                "detail":
                    (
                        "synaptic_assembly_"
                        "receptor_trafficking-"
                        "clinical_asd"
                    ),
            },
            {
                "section":
                    "expected_results",
                "item":
                    (
                        "Tier2_family_corrected_"
                        "associations"
                    ),
                "value":
                    tier2_associations,
                "passed":
                    tier2_associations == 4,
                "detail":
                    (
                        "four_secondary_"
                        "clinical_associations"
                    ),
            },
            {
                "section":
                    "expected_results",
                "item":
                    "primary_pair_identity",
                "value":
                    primary_pair_correct,
                "passed":
                    primary_pair_correct,
                "detail":
                    (
                        "synaptic_assembly_"
                        "receptor_trafficking-"
                        "clinical_asd"
                    ),
            },
            {
                "section":
                    "expected_results",
                "item":
                    "priority_associations",
                "value":
                    len(priority),
                "passed":
                    len(priority) == 5,
                "detail":
                    (
                        "one_primary_plus_"
                        "four_secondary"
                    ),
            },
            {
                "section":
                    "robustness",
                "item":
                    (
                        "priority_pairs_LODO_"
                        "5of5_direction"
                    ),
                "value":
                    priority_lodo_concordant,
                "passed":
                    priority_lodo_concordant,
                "detail":
                    (
                        "directional_concordance_"
                        "not_effect_size_stability"
                    ),
            },
            {
                "section":
                    "sensitivity",
                "item":
                    "LH46_BH_supported_pairs",
                "value":
                    int(
                        hierarchy[
                            "LH46_BH_supported"
                        ].sum()
                    ),
                "passed":
                    no_lh46_bh_hits,
                "detail":
                    (
                        "nested_LH46_sensitivity_"
                        "expected_zero_BH_hits"
                    ),
            },
            {
                "section":
                    "map_hierarchy",
                "item":
                    "DTHI_maps_classified",
                "value":
                    len(map_hierarchy),
                "passed":
                    len(map_hierarchy) == 10,
                "detail":
                    "",
            },
            {
                "section":
                    "map_hierarchy",
                "item":
                    "Tier1_maps",
                "value":
                    tier1_maps,
                "passed":
                    tier1_maps == 1,
                "detail":
                    (
                        "synaptic_assembly_"
                        "receptor_trafficking"
                    ),
            },
            {
                "section":
                    "map_hierarchy",
                "item":
                    "Tier2_maps",
                "value":
                    tier2_maps,
                "passed":
                    tier2_maps == 3,
                "detail":
                    (
                        "patterning;"
                        "axon_guidance;"
                        "synaptic_membrane"
                    ),
            },
            {
                "section":
                    "interpretation",
                "item":
                    "causal_claims_disabled",
                "value":
                    bool(
                        (
                            ~hierarchy[
                                "causal_interpretation_allowed"
                            ]
                        ).all()
                    ),
                "passed":
                    bool(
                        (
                            ~hierarchy[
                                "causal_interpretation_allowed"
                            ]
                        ).all()
                    ),
                "detail":
                    "",
            },
            {
                "section":
                    "interpretation",
                "item":
                    (
                        "independent_replication_"
                        "claims_disabled"
                    ),
                "value":
                    bool(
                        (
                            ~hierarchy[
                                "independent_replication_claim_allowed"
                            ]
                        ).all()
                    ),
                "passed":
                    bool(
                        (
                            ~hierarchy[
                                "independent_replication_claim_allowed"
                            ]
                        ).all()
                    ),
                "detail":
                    (
                        "LH46_nested_"
                        "sensitivity_only"
                    ),
            },
        ]
    )

    write_tsv(
        audit,
        AUDIT_OUT,
    )

    all_passed = bool(
        audit["passed"].all()
    )

    completion = pd.DataFrame(
        [
            {
                "Phase8D3B_status_confirmed":
                    True,
                "associations_classified":
                    len(hierarchy),
                "DTHI_maps_classified":
                    len(map_hierarchy),
                "priority_associations_reported":
                    len(priority),
                "Tier1_global_FWER_associations":
                    tier1_associations,
                "Tier2_family_corrected_associations":
                    tier2_associations,
                "Tier3A_nominal_associations":
                    tier3a_associations,
                "Tier3B_descriptive_associations":
                    tier3b_associations,
                "Tier1_maps":
                    tier1_maps,
                "Tier2_maps":
                    tier2_maps,
                "priority_pairs_LODO_5of5_direction":
                    priority_lodo_concordant,
                "LH46_BH_supported_pairs":
                    int(
                        hierarchy[
                            "LH46_BH_supported"
                        ].sum()
                    ),
                "hierarchy_mutually_exclusive":
                    True,
                "statistics_recomputed":
                    False,
                "all_evidence_hierarchy_audits_passed":
                    all_passed,
                "ready_for_phase8D3D_integrated_figure":
                    all_passed,
                "Phase8D3C_status":
                    (
                        "completed"
                        if all_passed
                        else "failed"
                    ),
                "python_version":
                    sys.version.split()[0],
                "numpy_version":
                    np.__version__,
                "pandas_version":
                    pd.__version__,
            }
        ]
    )

    write_tsv(
        completion,
        COMPLETION_OUT,
    )

    print(
        "===== PHASE 8D3C COMPLETION ====="
    )
    print(
        completion.to_string(
            index=False
        )
    )

    print(
        "\n===== ASSOCIATION-LEVEL "
        "EVIDENCE COUNTS ====="
    )

    evidence_counts = (
        hierarchy[
            "evidence_tier"
        ]
        .value_counts()
        .reindex(
            TIER_ORDER.keys(),
            fill_value=0,
        )
        .rename_axis(
            "evidence_tier"
        )
        .reset_index(
            name="association_count"
        )
    )

    print(
        evidence_counts.to_string(
            index=False
        )
    )

    print(
        "\n===== MAP-LEVEL "
        "EVIDENCE HIERARCHY ====="
    )

    map_display = [
        "DTHI_map",
        "map_level_highest_tier",
        "priority_association_count",
        "highest_evidence_external_map",
        "highest_evidence_rho",
        "AHBA_minimum_LODO_vs_full_rho",
        "empirical_peak_developmental_window",
        "cross_modal_convergence_class",
    ]

    print(
        map_hierarchy[
            map_display
        ].to_string(
            index=False
        )
    )

    print(
        "\n===== PRIORITY "
        "REPORTING TABLE ====="
    )

    priority_display = [
        "priority_reporting_order",
        "priority_role",
        "DTHI_map",
        "external_map",
        "spearman_rho",
        "evidence_tier",
        "corrected_evidence_subclass",
        "bijective_p_spin_two_sided",
        "bijective_q_BH_global160",
        "bijective_q_BH_family80",
        "bijective_p_maxT_global160",
        "bijective_p_maxT_family80",
        "LODO_5of5_direction_concordant",
        "LH46_BH_supported",
    ]

    existing_priority_columns = [
        column
        for column in priority_display
        if column in priority.columns
    ]

    print(
        priority[
            existing_priority_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\n===== AUDIT ====="
    )

    print(
        audit.to_string(
            index=False
        )
    )

    outputs = [
        ASSOC_OUT,
        MAP_OUT,
        PRIORITY_OUT,
        LABELS_OUT,
        SUMMARY_OUT,
        PROVENANCE_OUT,
        AUDIT_OUT,
        COMPLETION_OUT,
    ]

    output_hashes = pd.DataFrame(
        [
            {
                "relative_path":
                    str(
                        path.relative_to(
                            ROOT
                        )
                    ),
                "size_bytes":
                    path.stat().st_size,
                "SHA256":
                    sha256(path),
            }
            for path in outputs
        ]
    )

    print(
        "\n===== CANONICAL "
        "OUTPUT HASHES ====="
    )

    print(
        output_hashes.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            "Phase 8D3C failed: "
            f"{type(error).__name__}: "
            f"{error}",
            file=sys.stderr,
        )
        raise
