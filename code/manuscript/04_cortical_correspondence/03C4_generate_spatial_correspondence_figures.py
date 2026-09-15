#!/usr/bin/env python3
"""Phase 8D2C4: generate manuscript-ready spatial-correspondence figures."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "03_processed_data/functional_imaging/phase8D/phase8D2C"
TABLES = ROOT / "07_tables/main_tables/phase8"
FIGURES = ROOT / "08_figures/phase8/phase8D2C4"

SUMMARY_IN = TABLES / "phase8D2C3_completion_summary.tsv"
INTEGRATED_IN = DATA / "phase8D2C3_integrated_evidence.tsv"
PRIORITY_IN = DATA / "phase8D2C3_priority_associations.tsv"
STABILITY_IN = DATA / "phase8D2C3_DTHI_map_stability_summary.tsv"
COUNTS_IN = TABLES / "phase8D2C3_evidence_counts.tsv"

HEATMAP_BASE = FIGURES / "phase8D2C4_Figure1_DTHI_external_spatial_correspondence"
FOREST_BASE = FIGURES / "phase8D2C4_Figure2_priority_association_robustness"
STABILITY_BASE = FIGURES / "phase8D2C4_Figure3_DTHI_LODO_map_stability"
COUNTS_BASE = FIGURES / "phase8D2C4_Figure4_evidence_counts"
CLASS_BASE = FIGURES / "phase8D2C4_SupplementaryFigure1_evidence_class_matrix"
COMBINED_PDF = FIGURES / "phase8D2C4_all_figures.pdf"
LEGENDS_OUT = FIGURES / "phase8D2C4_figure_legends.txt"
MANIFEST_OUT = TABLES / "phase8D2C4_figure_manifest.tsv"
AUDIT_OUT = TABLES / "phase8D2C4_figure_audit.tsv"
COMPLETION_OUT = TABLES / "phase8D2C4_completion_summary.tsv"

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
EXTERNAL_ORDER = [
    "cognitive_attention",
    "cognitive_executive_control",
    "cognitive_memory",
    "cognitive_language",
    "cognitive_social_cognition",
    "cognitive_emotion_affect",
    "cognitive_reward_motivation",
    "cognitive_sensorimotor",
    "clinical_adhd",
    "clinical_asd",
    "clinical_bipolar",
    "clinical_depression",
    "clinical_epilepsy",
    "clinical_ocd",
    "clinical_parkinsons",
    "clinical_schizophrenia",
]

DTHI_LABELS = {
    "patterning_arealization": "Patterning / arealization",
    "progenitor_radial_glia": "Progenitor / radial glia",
    "neurogenesis_migration_layering": "Neurogenesis / migration / layering",
    "axon_guidance_neurite_outgrowth": "Axon guidance / neurite outgrowth",
    "synaptic_assembly_receptor_trafficking": "Synaptic assembly / receptor trafficking",
    "astrocyte_maturation_metabolic_support": "Astrocyte maturation / metabolic support",
    "oligodendrocyte_myelination": "Oligodendrocyte / myelination",
    "activity_dependent_plasticity": "Activity-dependent plasticity",
    "synaptic_membrane_structural_candidates": "Synaptic-membrane structural candidates",
    "preliminary_maturation_balance": "Preliminary maturation balance",
}
EXTERNAL_LABELS = {
    "cognitive_attention": "Attention",
    "cognitive_executive_control": "Executive control",
    "cognitive_memory": "Episodic memory",
    "cognitive_language": "Language",
    "cognitive_social_cognition": "Social cognition",
    "cognitive_emotion_affect": "Emotion / affect",
    "cognitive_reward_motivation": "Reward / motivation",
    "cognitive_sensorimotor": "Sensorimotor",
    "clinical_adhd": "ADHD",
    "clinical_asd": "ASD",
    "clinical_bipolar": "Bipolar disorder",
    "clinical_depression": "Depression",
    "clinical_epilepsy": "Epilepsy",
    "clinical_ocd": "OCD",
    "clinical_parkinsons": "Parkinson's disease",
    "clinical_schizophrenia": "Schizophrenia",
}

CLASS_ORDER = [
    "no_corrected_spatial_support",
    "bijective_nominal_only",
    "nonbijective_only_corrected",
    "family_FDR_bijective",
    "global_FDR_bijective",
    "family_FWER_bijective",
    "global_FWER_bijective",
]


class ValidationError(RuntimeError):
    pass


def require_file(path: Path) -> None:
    if not path.is_file():
        raise ValidationError(f"Required file not found: {path}")


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValidationError(f"{label} is missing columns: {missing}")


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_tsv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)


def save_figure(fig: plt.Figure, base: Path) -> list[Path]:
    paths = [
        base.with_suffix(".png"),
        base.with_suffix(".pdf"),
        base.with_suffix(".svg"),
    ]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], bbox_inches="tight")
    return paths


def marker_for(row: pd.Series) -> str:
    if row["bijective_p_maxT_global160"] < 0.05:
        return "★"
    if row["bijective_p_maxT_family80"] < 0.05:
        return "◆"
    if row["bijective_q_BH_family80"] < 0.05:
        return "●"
    return ""


def build_heatmap(integrated: pd.DataFrame) -> plt.Figure:
    pivot = integrated.pivot(
        index="DTHI_map", columns="external_map", values="spearman_rho"
    ).reindex(index=DTHI_ORDER, columns=EXTERNAL_ORDER)
    if pivot.isna().any().any():
        raise ValidationError("Correlation heatmap contains missing values.")

    fig, ax = plt.subplots(figsize=(15.8, 8.8))
    image = ax.imshow(
        pivot.to_numpy(dtype=float),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-0.70,
        vmax=0.70,
    )
    ax.set_xticks(np.arange(len(EXTERNAL_ORDER)))
    ax.set_xticklabels(
        [EXTERNAL_LABELS[item] for item in EXTERNAL_ORDER],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.set_yticks(np.arange(len(DTHI_ORDER)))
    ax.set_yticklabels([DTHI_LABELS[item] for item in DTHI_ORDER], fontsize=9)
    ax.axvline(7.5, linewidth=1.2)
    ax.set_xlabel("External cognitive and clinical maps")
    ax.set_ylabel("DTHI cortical hierarchy maps")
    ax.set_title(
        "Spatial correspondence between DTHI maps and external cortical maps\n"
        "Spearman correlations across the fixed LH49 analysis scope"
    )

    lookup = integrated.set_index(["DTHI_map", "external_map"])
    for row_index, dthi_map in enumerate(DTHI_ORDER):
        for column_index, external_map in enumerate(EXTERNAL_ORDER):
            marker = marker_for(lookup.loc[(dthi_map, external_map)])
            if marker:
                ax.text(
                    column_index,
                    row_index,
                    marker,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )

    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Spearman ρ")
    ax.text(
        0.0,
        -0.25,
        "Marker hierarchy: ★ global maxT P<0.05; ◆ family maxT P<0.05; "
        "● family BH-FDR q<0.05. Exact-bijective spatial nulls.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )
    fig.tight_layout()
    return fig


def priority_label(row: pd.Series) -> str:
    dthi = DTHI_LABELS.get(row["DTHI_map"], row["DTHI_map"])
    external = EXTERNAL_LABELS.get(row["external_map"], row["external_map"])
    if row["bijective_p_maxT_global160"] < 0.05:
        support = "global maxT"
    elif row["bijective_p_maxT_family80"] < 0.05:
        support = "family maxT"
    elif row["bijective_q_BH_family80"] < 0.05:
        support = "family BH-FDR"
    else:
        support = "exploratory"
    return f"{dthi} — {external} [{support}]"


def build_priority_forest(priority: pd.DataFrame) -> plt.Figure:
    ordered = priority.sort_values("spearman_rho").reset_index(drop=True)
    y = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    for index, row in ordered.iterrows():
        ax.plot(
            [row["LODO_minimum_rho"], row["LODO_maximum_rho"]],
            [index, index],
            linewidth=3,
            solid_capstyle="round",
        )
    ax.scatter(
        ordered["spearman_rho"],
        y,
        marker="o",
        s=75,
        label="Full five-donor LH49 ρ",
        zorder=3,
    )
    ax.scatter(
        ordered["bijective_high_support_spearman_rho"],
        y,
        marker="x",
        s=85,
        label="Nested LH46 ρ",
        zorder=4,
    )
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([priority_label(row) for _, row in ordered.iterrows()])
    ax.set_xlabel("Spearman ρ")
    ax.set_title(
        "Priority spatial associations and leave-one-donor-out robustness\n"
        "Horizontal segments show the five-donor LODO range, not confidence intervals"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(
        min(-0.80, float(ordered["LODO_minimum_rho"].min()) - 0.05),
        max(0.10, float(ordered["LODO_maximum_rho"].max()) + 0.05),
    )
    fig.tight_layout()
    return fig


def build_stability(stability: pd.DataFrame) -> plt.Figure:
    ordered = stability.sort_values(
        "minimum_LODO_vs_full_spearman"
    ).reset_index(drop=True)
    y = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(12.5, 7.3))
    ax.scatter(
        ordered["minimum_LODO_vs_full_spearman"],
        y,
        marker="o",
        s=65,
        label="Minimum LODO-vs-full ρ",
    )
    ax.scatter(
        ordered["median_LODO_vs_full_spearman"],
        y,
        marker="s",
        s=55,
        label="Median LODO-vs-full ρ",
    )
    for index, row in ordered.iterrows():
        ax.plot(
            [
                row["minimum_LODO_vs_full_spearman"],
                row["median_LODO_vs_full_spearman"],
            ],
            [index, index],
            linewidth=2,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(
        [DTHI_LABELS.get(item, item) for item in ordered["DTHI_map"]]
    )
    ax.set_xlabel("Spearman correlation with the full donor-balanced DTHI map")
    ax.set_title("DTHI map stability across five leave-one-donor-out reconstructions")
    ax.set_xlim(0.65, 1.01)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def build_counts(counts: pd.DataFrame) -> plt.Figure:
    wanted = [
        "bijective_global_maxT_significant",
        "bijective_family_maxT_significant",
        "bijective_global_BH_significant",
        "bijective_family_BH_significant",
        "bijective_LH46_BH_significant",
        "priority_for_reporting",
    ]
    labels = {
        "bijective_global_maxT_significant": "Global maxT",
        "bijective_family_maxT_significant": "Family maxT",
        "bijective_global_BH_significant": "Global BH-FDR",
        "bijective_family_BH_significant": "Family BH-FDR",
        "bijective_LH46_BH_significant": "LH46 BH-FDR",
        "priority_for_reporting": "Priority associations",
    }
    selected = (
        counts.loc[counts["metric"].isin(wanted)]
        .set_index("metric")
        .reindex(wanted)
        .reset_index()
    )
    if selected["count"].isna().any():
        raise ValidationError("Evidence-count table is missing required metrics.")

    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    x = np.arange(len(selected))
    bars = ax.bar(x, selected["count"].astype(int))
    ax.set_xticks(x)
    ax.set_xticklabels(
        [labels[item] for item in selected["metric"]],
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("Number of DTHI–external map pairs")
    ax.set_title("Integrated evidence counts across 160 spatial associations")
    ax.set_ylim(0, max(6, int(selected["count"].max()) + 1))
    for bar, value in zip(bars, selected["count"].astype(int)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.08,
            str(value),
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    return fig


def build_class_matrix(integrated: pd.DataFrame) -> plt.Figure:
    coded = integrated.copy()
    code_lookup = {name: index for index, name in enumerate(CLASS_ORDER)}
    coded["class_code"] = coded["evidence_class"].map(code_lookup)
    if coded["class_code"].isna().any():
        unknown = sorted(coded.loc[coded["class_code"].isna(), "evidence_class"].unique())
        raise ValidationError(f"Unknown evidence classes: {unknown}")

    pivot = coded.pivot(
        index="DTHI_map", columns="external_map", values="class_code"
    ).reindex(index=DTHI_ORDER, columns=EXTERNAL_ORDER)

    fig, ax = plt.subplots(figsize=(15.8, 8.8))
    image = ax.imshow(
        pivot.to_numpy(dtype=float),
        aspect="auto",
        cmap="tab20",
        vmin=-0.5,
        vmax=len(CLASS_ORDER) - 0.5,
    )
    ax.set_xticks(np.arange(len(EXTERNAL_ORDER)))
    ax.set_xticklabels(
        [EXTERNAL_LABELS[item] for item in EXTERNAL_ORDER],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.set_yticks(np.arange(len(DTHI_ORDER)))
    ax.set_yticklabels([DTHI_LABELS[item] for item in DTHI_ORDER], fontsize=9)
    ax.axvline(7.5, linewidth=1.2)
    ax.set_title("Integrated exact-bijective evidence classification")
    cbar = fig.colorbar(
        image,
        ax=ax,
        ticks=np.arange(len(CLASS_ORDER)),
        fraction=0.025,
        pad=0.02,
    )
    cbar.ax.set_yticklabels(
        [
            "No corrected support",
            "Nominal only",
            "Non-bijective only",
            "Family FDR",
            "Global FDR",
            "Family FWER",
            "Global FWER",
        ]
    )
    fig.tight_layout()
    return fig


def main() -> None:
    for path in [SUMMARY_IN, INTEGRATED_IN, PRIORITY_IN, STABILITY_IN, COUNTS_IN]:
        require_file(path)

    summary = pd.read_csv(SUMMARY_IN, sep="\t")
    if len(summary) != 1:
        raise ValidationError("Phase 8D2C3 completion summary must have one row.")
    if str(summary.loc[0, "Phase8D2C3_status"]).strip() != "completed":
        raise ValidationError("Phase 8D2C3 is not completed.")
    if not parse_bool(summary.loc[0, "ready_for_phase8D2C4_figures"]):
        raise ValidationError("Phase 8D2C3 is not ready for figure generation.")

    integrated = pd.read_csv(INTEGRATED_IN, sep="\t")
    priority = pd.read_csv(PRIORITY_IN, sep="\t")
    stability = pd.read_csv(STABILITY_IN, sep="\t")
    counts = pd.read_csv(COUNTS_IN, sep="\t")

    require_columns(
        integrated,
        [
            "DTHI_map",
            "external_map",
            "spearman_rho",
            "bijective_q_BH_family80",
            "bijective_p_maxT_global160",
            "bijective_p_maxT_family80",
            "evidence_class",
        ],
        "integrated evidence",
    )
    require_columns(
        priority,
        [
            "DTHI_map",
            "external_map",
            "spearman_rho",
            "bijective_p_maxT_global160",
            "bijective_p_maxT_family80",
            "bijective_q_BH_family80",
            "bijective_high_support_spearman_rho",
            "LODO_minimum_rho",
            "LODO_maximum_rho",
        ],
        "priority associations",
    )
    require_columns(
        stability,
        [
            "DTHI_map",
            "minimum_LODO_vs_full_spearman",
            "median_LODO_vs_full_spearman",
        ],
        "DTHI map stability",
    )
    require_columns(counts, ["metric", "count"], "evidence counts")

    if len(integrated) != 160:
        raise ValidationError(f"Expected 160 integrated pairs; found {len(integrated)}.")
    if len(priority) != 5:
        raise ValidationError(f"Expected 5 priority associations; found {len(priority)}.")
    if set(DTHI_ORDER) != set(integrated["DTHI_map"].unique()):
        raise ValidationError("DTHI map set does not match the frozen 10-map order.")
    if set(EXTERNAL_ORDER) != set(integrated["external_map"].unique()):
        raise ValidationError("External map set does not match the frozen 16-map order.")

    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figures = [
        ("Figure 1", HEATMAP_BASE, build_heatmap(integrated)),
        ("Figure 2", FOREST_BASE, build_priority_forest(priority)),
        ("Figure 3", STABILITY_BASE, build_stability(stability)),
        ("Figure 4", COUNTS_BASE, build_counts(counts)),
        ("Supplementary Figure 1", CLASS_BASE, build_class_matrix(integrated)),
    ]

    generated: list[tuple[str, Path]] = []
    with PdfPages(COMBINED_PDF) as pdf:
        for label, base, figure in figures:
            for path in save_figure(figure, base):
                generated.append((label, path))
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    generated.append(("Combined multipage PDF", COMBINED_PDF))

    legends = """PHASE 8D2C4 FIGURE LEGENDS

Figure 1. Spatial correspondence between DTHI cortical hierarchy maps and external cognitive and clinical maps. Heatmap values are Spearman correlations across the fixed left-hemisphere 49-parcel analysis scope. Symbols indicate the strongest exact-bijective inferential support for each pair: star, global maxT P<0.05; diamond, family maxT P<0.05; circle, family BH-FDR q<0.05. Cognitive maps are shown first and clinical maps second.

Figure 2. Donor robustness of priority DTHI–clinical spatial associations. Circles show full five-donor LH49 Spearman correlations, horizontal segments show the range obtained after each of five leave-one-donor-out reconstructions, and crosses show the nested high-support LH46 correlation. The LODO range is a robustness range and not a confidence interval.

Figure 3. Stability of the ten DTHI cortical maps after leave-one-donor-out reconstruction. For each DTHI map, circles show the minimum and squares show the median Spearman correlation between the full donor-balanced map and its five LODO reconstructions.

Figure 4. Number of spatial associations meeting each integrated evidence criterion across the 160 tested DTHI–external map pairs. Exact-bijective global and family maxT, global and family BH-FDR, nested LH46 BH-FDR, and the final priority-reporting set are shown separately.

Supplementary Figure 1. Integrated evidence classification for all 160 DTHI–external map pairs. Evidence classes follow the prespecified hierarchy from no corrected spatial support through exact-bijective global familywise-error control.
"""
    LEGENDS_OUT.write_text(legends, encoding="utf-8")

    manifest_rows = []
    for label, path in generated:
        manifest_rows.append(
            {
                "figure": label,
                "relative_path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "SHA256": sha256(path),
            }
        )
    manifest_rows.append(
        {
            "figure": "Figure legends",
            "relative_path": str(LEGENDS_OUT.relative_to(ROOT)),
            "size_bytes": LEGENDS_OUT.stat().st_size,
            "SHA256": sha256(LEGENDS_OUT),
        }
    )
    manifest = pd.DataFrame(manifest_rows)
    atomic_tsv(manifest, MANIFEST_OUT)

    audit = pd.DataFrame(
        [
            {"section": "upstream", "item": "Phase8D2C3_completed", "value": True, "passed": True},
            {"section": "inputs", "item": "integrated_pairs", "value": len(integrated), "passed": len(integrated) == 160},
            {"section": "inputs", "item": "priority_pairs", "value": len(priority), "passed": len(priority) == 5},
            {"section": "figures", "item": "standalone_figures", "value": len(figures), "passed": len(figures) == 5},
            {"section": "figures", "item": "combined_pdf_pages", "value": len(figures), "passed": len(figures) == 5},
            {"section": "outputs", "item": "all_manifest_files_exist", "value": all((ROOT / p).is_file() for p in manifest["relative_path"]), "passed": all((ROOT / p).is_file() for p in manifest["relative_path"])},
        ]
    )
    atomic_tsv(audit, AUDIT_OUT)

    completion = pd.DataFrame(
        [
            {
                "Phase8D2C3_status_confirmed": True,
                "integrated_pairs_visualized": len(integrated),
                "priority_pairs_visualized": len(priority),
                "standalone_figure_designs": len(figures),
                "raster_vector_pdf_outputs": 15,
                "combined_pdf_pages": len(figures),
                "all_figure_files_present": bool(audit.loc[audit["item"] == "all_manifest_files_exist", "passed"].iloc[0]),
                "ready_for_spatial_results_reporting": True,
                "Phase8D2C4_status": "completed",
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "matplotlib_version": matplotlib.__version__,
            }
        ]
    )
    atomic_tsv(completion, COMPLETION_OUT)

    print("===== PHASE 8D2C4 COMPLETION =====")
    print(completion.to_string(index=False))
    print("\n===== FIGURE MANIFEST =====")
    print(manifest.to_string(index=False))
    print("\n===== AUDIT =====")
    print(audit.to_string(index=False))
    print("\n===== CANONICAL OUTPUT HASHES =====")
    canonical = pd.concat(
        [
            manifest,
            pd.DataFrame(
                [
                    {
                        "figure": "Figure audit",
                        "relative_path": str(AUDIT_OUT.relative_to(ROOT)),
                        "size_bytes": AUDIT_OUT.stat().st_size,
                        "SHA256": sha256(AUDIT_OUT),
                    },
                    {
                        "figure": "Completion summary",
                        "relative_path": str(COMPLETION_OUT.relative_to(ROOT)),
                        "size_bytes": COMPLETION_OUT.stat().st_size,
                        "SHA256": sha256(COMPLETION_OUT),
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    print(canonical[["relative_path", "size_bytes", "SHA256"]].to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Phase 8D2C4 failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise
