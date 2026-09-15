#!/usr/bin/env python3
"""Phase 8D2C4B: figure-only revision for spatial-correspondence results."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "03_processed_data/functional_imaging/phase8D/phase8D2C"
TABLES = ROOT / "07_tables/main_tables/phase8"
FIGURES = ROOT / "08_figures/phase8/phase8D2C4B"

SUMMARY_IN = TABLES / "phase8D2C4_completion_summary.tsv"
INTEGRATED_IN = DATA / "phase8D2C3_integrated_evidence.tsv"
PRIORITY_IN = DATA / "phase8D2C3_priority_associations.tsv"
STABILITY_IN = DATA / "phase8D2C3_DTHI_map_stability_summary.tsv"
COUNTS_IN = TABLES / "phase8D2C3_evidence_counts.tsv"

FIG1_BASE = FIGURES / "phase8D2C4B_Figure1_DTHI_external_spatial_correspondence_revised"
FIG2_BASE = FIGURES / "phase8D2C4B_Figure2_priority_association_robustness_revised"
FIG3_BASE = FIGURES / "phase8D2C4B_Figure3_DTHI_LODO_map_stability_revised"
SUPP1_BASE = FIGURES / "phase8D2C4B_SupplementaryFigure1_overlapping_evidence_counts"
SUPP2_BASE = FIGURES / "phase8D2C4B_SupplementaryFigure2_evidence_class_matrix_revised"
COMBINED_PDF = FIGURES / "phase8D2C4B_all_revised_figures.pdf"
LEGENDS_OUT = FIGURES / "phase8D2C4B_figure_legends.txt"
MANIFEST_OUT = TABLES / "phase8D2C4B_figure_manifest.tsv"
AUDIT_OUT = TABLES / "phase8D2C4B_figure_revision_audit.tsv"
COMPLETION_OUT = TABLES / "phase8D2C4B_completion_summary.tsv"

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

# Only classes actually present after synthesis.
CLASS_PRESENT_ORDER = [
    "global_FWER_bijective",
    "family_FDR_bijective",
    "bijective_nominal_only",
]
CLASS_LABELS = {
    "global_FWER_bijective": "Global FWER",
    "family_FDR_bijective": "Family FDR",
    "bijective_nominal_only": "Nominal only",
}
CLASS_COLORS = {
    "global_FWER_bijective": "#0b4f8a",
    "family_FDR_bijective": "#8c564b",
    "bijective_nominal_only": "#7fbf7b",
}
BACKGROUND_COLOR = "#ffffff"


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
    paths = [base.with_suffix(".png"), base.with_suffix(".pdf"), base.with_suffix(".svg")]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], bbox_inches="tight")
    return paths


def evidence_marker(row: pd.Series) -> str:
    # strongest retained support only
    if float(row["bijective_p_maxT_global160"]) < 0.05:
        return "★"
    if float(row["bijective_p_maxT_family80"]) < 0.05:
        return "◆"
    if float(row["bijective_q_BH_family80"]) < 0.05:
        return "●"
    return ""


def build_heatmap(integrated: pd.DataFrame) -> plt.Figure:
    pivot = integrated.pivot(index="DTHI_map", columns="external_map", values="spearman_rho").reindex(
        index=DTHI_ORDER, columns=EXTERNAL_ORDER
    )
    if pivot.isna().any().any():
        raise ValidationError("Correlation heatmap contains missing values.")

    fig, ax = plt.subplots(figsize=(15.8, 8.9))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-0.70, vmax=0.70)
    ax.set_xticks(np.arange(len(EXTERNAL_ORDER)))
    ax.set_xticklabels([EXTERNAL_LABELS[item] for item in EXTERNAL_ORDER], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(DTHI_ORDER)))
    ax.set_yticklabels([DTHI_LABELS[item] for item in DTHI_ORDER], fontsize=9)
    ax.axvline(7.5, color="0.5", linewidth=1.1)
    ax.set_xlabel("External cognitive and clinical maps")
    ax.set_ylabel("DTHI cortical hierarchy maps")
    ax.set_title(
        "Spatial correspondence between DTHI maps and external cortical maps\n"
        "Spearman correlations across the fixed LH49 analysis scope"
    )

    lookup = integrated.set_index(["DTHI_map", "external_map"])
    for row_idx, dthi_map in enumerate(DTHI_ORDER):
        for col_idx, external_map in enumerate(EXTERNAL_ORDER):
            marker = evidence_marker(lookup.loc[(dthi_map, external_map)])
            if marker:
                txt = ax.text(
                    col_idx, row_idx, marker, ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold", zorder=5
                )
                txt.set_path_effects([pe.Stroke(linewidth=2.0, foreground="black"), pe.Normal()])

    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Spearman ρ")

    handles = [
        Line2D([0], [0], marker=None, linestyle="None", label="Strongest retained exact-bijective evidence"),
        Line2D([0], [0], marker=r"$★$", color="white", markeredgecolor="black", markersize=12,
               linestyle="None", label="Global maxT P<0.05"),
        Line2D([0], [0], marker=r"$◆$", color="white", markeredgecolor="black", markersize=11,
               linestyle="None", label="Family maxT P<0.05"),
        Line2D([0], [0], marker=r"$●$", color="white", markeredgecolor="black", markersize=10,
               linestyle="None", label="Family BH-FDR q<0.05"),
    ]
    legend = ax.legend(handles=handles[1:], loc="upper center", bbox_to_anchor=(0.5, -0.20),
                       ncol=3, frameon=False, fontsize=9, title="Strongest retained evidence category")
    legend._legend_box.align = "left"
    fig.tight_layout()
    return fig


def priority_support_label(row: pd.Series) -> str:
    if float(row["bijective_p_maxT_global160"]) < 0.05:
        return "global maxT"
    if float(row["bijective_p_maxT_family80"]) < 0.05:
        return "family maxT"
    if float(row["bijective_q_BH_family80"]) < 0.05:
        return "family BH-FDR"
    return "exploratory"


def priority_label(row: pd.Series) -> str:
    dthi = DTHI_LABELS.get(row["DTHI_map"], row["DTHI_map"])
    external = EXTERNAL_LABELS.get(row["external_map"], row["external_map"])
    return f"{dthi} — {external} [{priority_support_label(row)}]"


def build_priority_forest(priority: pd.DataFrame) -> plt.Figure:
    ordered = priority.sort_values("spearman_rho").reset_index(drop=True)
    y = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(13.3, 6.9))
    ax.axvline(0, color="0.6", linestyle="--", linewidth=1)

    primary_mask = ordered["bijective_p_maxT_global160"] < 0.05
    for idx, row in ordered.iterrows():
        line_color = "0.35"
        line_width = 2.2
        if primary_mask.iloc[idx]:
            line_color = "black"
            line_width = 2.8
        ax.plot([row["LODO_minimum_rho"], row["LODO_maximum_rho"]], [idx, idx],
                color=line_color, linewidth=line_width, solid_capstyle="round", zorder=1)

    ax.scatter(ordered["spearman_rho"], y, marker="o", s=70, color="#1f77b4", zorder=3, label="Full five-donor LH49 ρ")
    ax.scatter(ordered["bijective_high_support_spearman_rho"], y, marker="x", s=85, color="#ff7f0e", zorder=4, label="Nested LH46 ρ")

    # subtle emphasis for the primary association
    for idx, is_primary in enumerate(primary_mask):
        if bool(is_primary):
            ax.axhspan(idx - 0.35, idx + 0.35, color="0.96", zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels([priority_label(row) for _, row in ordered.iterrows()], fontsize=9)
    ax.set_xlabel("Spearman ρ")
    ax.set_title(
        "Priority spatial associations and leave-one-donor-out robustness\n"
        "Horizontal segments show the five-donor LODO range, not confidence intervals"
    )
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(min(-0.80, float(ordered["LODO_minimum_rho"].min()) - 0.05), 0.06)

    handles = [
        Line2D([0], [0], color="0.35", linewidth=2.2, label="LODO range (family BH-FDR priority associations)"),
        Line2D([0], [0], color="black", linewidth=2.8, label="LODO range (global maxT association)"),
        Line2D([0], [0], marker="o", color="#1f77b4", linestyle="None", markersize=7, label="Full five-donor LH49 ρ"),
        Line2D([0], [0], marker="x", color="#ff7f0e", linestyle="None", markersize=8, label="Nested LH46 ρ"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def build_stability(stability: pd.DataFrame) -> plt.Figure:
    ordered = stability.sort_values("minimum_LODO_vs_full_spearman").reset_index(drop=True)
    y = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(12.5, 7.3))
    for idx, row in ordered.iterrows():
        ax.plot(
            [row["minimum_LODO_vs_full_spearman"], row["median_LODO_vs_full_spearman"]],
            [idx, idx],
            color="0.55", linewidth=2.2, zorder=1
        )
    ax.scatter(ordered["minimum_LODO_vs_full_spearman"], y, marker="o", s=65, color="#1f77b4",
               label="Minimum LODO-vs-full ρ", zorder=3)
    ax.scatter(ordered["median_LODO_vs_full_spearman"], y, marker="s", s=55, color="#ff7f0e",
               label="Median LODO-vs-full ρ", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([DTHI_LABELS.get(item, item) for item in ordered["DTHI_map"]], fontsize=9)
    ax.set_xlabel("Spearman correlation with the full donor-balanced DTHI map")
    ax.set_title("DTHI map stability across five leave-one-donor-out reconstructions")
    ax.set_xlim(0.65, 1.01)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
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
    selected = counts.loc[counts["metric"].isin(wanted)].set_index("metric").reindex(wanted).reset_index()
    if selected["count"].isna().any():
        raise ValidationError("Evidence-count table is missing required metrics.")

    fig, ax = plt.subplots(figsize=(9.8, 5.9))
    x = np.arange(len(selected))
    bars = ax.bar(x, selected["count"].astype(int), color="0.35")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[item] for item in selected["metric"]], rotation=30, ha="right")
    ax.set_ylabel("Number of DTHI–external map pairs")
    ax.set_title("Overlapping evidence-count summary across 160 spatial associations")
    ax.set_ylim(0, max(6, int(selected["count"].max()) + 1))
    for bar, value in zip(bars, selected["count"].astype(int)):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.06, str(value), ha="center", va="bottom")

    ax.text(
        0.02, -0.25,
        "Categories overlap and should not be summed.\n"
        "For example, the global maxT association is also counted within family maxT,\n"
        "family BH-FDR, and the priority-reporting set.",
        transform=ax.transAxes, fontsize=9, va="top"
    )
    fig.tight_layout()
    return fig


def derive_display_class(row: pd.Series) -> str | None:
    if float(row["bijective_p_maxT_global160"]) < 0.05:
        return "global_FWER_bijective"
    if float(row["bijective_q_BH_family80"]) < 0.05:
        return "family_FDR_bijective"
    if float(row["bijective_p_spin_two_sided"]) < 0.05:
        return "bijective_nominal_only"
    return None


def build_class_matrix(integrated: pd.DataFrame) -> plt.Figure:
    display = integrated.copy()
    display["display_class"] = display.apply(derive_display_class, axis=1)
    value_lookup = {None: 0}
    for idx, item in enumerate(CLASS_PRESENT_ORDER, start=1):
        value_lookup[item] = idx
    display["display_code"] = display["display_class"].map(value_lookup)

    pivot = display.pivot(index="DTHI_map", columns="external_map", values="display_code").reindex(
        index=DTHI_ORDER, columns=EXTERNAL_ORDER
    )

    # background + three actual classes
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([BACKGROUND_COLOR] + [CLASS_COLORS[item] for item in CLASS_PRESENT_ORDER])
    norm = BoundaryNorm(np.arange(-0.5, len(CLASS_PRESENT_ORDER) + 1.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(15.8, 8.8))
    ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(EXTERNAL_ORDER)))
    ax.set_xticklabels([EXTERNAL_LABELS[item] for item in EXTERNAL_ORDER], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(DTHI_ORDER)))
    ax.set_yticklabels([DTHI_LABELS[item] for item in DTHI_ORDER], fontsize=9)
    ax.axvline(7.5, color="0.5", linewidth=1.1)
    ax.set_title("Integrated exact-bijective evidence classification")
    ax.set_xlabel("External cognitive and clinical maps")
    ax.set_ylabel("DTHI cortical hierarchy maps")

    legend_handles = [Patch(facecolor=CLASS_COLORS[item], edgecolor="none", label=CLASS_LABELS[item]) for item in CLASS_PRESENT_ORDER]
    legend_handles.insert(0, Patch(facecolor=BACKGROUND_COLOR, edgecolor="0.7", label="No corrected support"))
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)

    ax.text(
        0.0, -0.18,
        "Only classes present in the integrated evidence table are displayed.\n"
        "Nominal-only cells denote uncorrected exact-bijective support and should be interpreted cautiously.",
        transform=ax.transAxes, fontsize=9, va="top"
    )
    fig.tight_layout()
    return fig


def main() -> None:
    for path in [SUMMARY_IN, INTEGRATED_IN, PRIORITY_IN, STABILITY_IN, COUNTS_IN]:
        require_file(path)

    summary = pd.read_csv(SUMMARY_IN, sep="\t")
    if len(summary) != 1:
        raise ValidationError("Phase 8D2C4 completion summary must have one row.")
    if str(summary.loc[0, "Phase8D2C4_status"]).strip() != "completed":
        raise ValidationError("Phase 8D2C4 is not completed.")
    if not parse_bool(summary.loc[0, "ready_for_spatial_results_reporting"]):
        raise ValidationError("Phase 8D2C4 is not ready for figure revision.")

    integrated = pd.read_csv(INTEGRATED_IN, sep="\t")
    priority = pd.read_csv(PRIORITY_IN, sep="\t")
    stability = pd.read_csv(STABILITY_IN, sep="\t")
    counts = pd.read_csv(COUNTS_IN, sep="\t")

    require_columns(
        integrated,
        [
            "DTHI_map", "external_map", "spearman_rho",
            "bijective_p_spin_two_sided", "bijective_q_BH_family80",
            "bijective_p_maxT_global160", "bijective_p_maxT_family80"
        ],
        "integrated evidence",
    )
    require_columns(
        priority,
        [
            "DTHI_map", "external_map", "spearman_rho",
            "bijective_p_maxT_global160", "bijective_q_BH_family80",
            "bijective_high_support_spearman_rho",
            "LODO_minimum_rho", "LODO_maximum_rho"
        ],
        "priority associations",
    )
    require_columns(stability, ["DTHI_map", "minimum_LODO_vs_full_spearman", "median_LODO_vs_full_spearman"], "DTHI map stability")
    require_columns(counts, ["metric", "count"], "evidence counts")

    if len(integrated) != 160:
        raise ValidationError(f"Expected 160 integrated pairs; found {len(integrated)}.")
    if len(priority) != 5:
        raise ValidationError(f"Expected 5 priority associations; found {len(priority)}.")

    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    figures = [
        ("Figure 1", FIG1_BASE, build_heatmap(integrated)),
        ("Figure 2", FIG2_BASE, build_priority_forest(priority)),
        ("Figure 3", FIG3_BASE, build_stability(stability)),
        ("Supplementary Figure 1", SUPP1_BASE, build_counts(counts)),
        ("Supplementary Figure 2", SUPP2_BASE, build_class_matrix(integrated)),
    ]

    generated = []
    with PdfPages(COMBINED_PDF) as pdf:
        for label, base, fig in figures:
            for path in save_figure(fig, base):
                generated.append((label, path))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    generated.append(("Combined multipage PDF", COMBINED_PDF))

    legends = """PHASE 8D2C4B REVISED FIGURE LEGENDS

Figure 1. Spatial correspondence between DTHI cortical hierarchy maps and external cognitive and clinical maps. Heatmap values are Spearman correlations across the fixed left-hemisphere 49-parcel analysis scope. Symbols indicate the strongest retained exact-bijective evidence category for each pair: star, global maxT P<0.05; diamond, family maxT P<0.05; circle, family BH-FDR q<0.05. Cognitive maps are shown first and clinical maps second.

Figure 2. Donor robustness of the five priority DTHI–clinical spatial associations. Circles show full five-donor LH49 Spearman correlations, horizontal segments show the range obtained after each of five leave-one-donor-out reconstructions, and crosses show the nested high-support LH46 correlation. The single exact-bijective global maxT association is highlighted separately from the four family-BH-FDR priority associations. The LODO range is a robustness range and not a confidence interval.

Figure 3. Stability of the ten DTHI cortical maps after leave-one-donor-out reconstruction. For each DTHI map, circles show the minimum and squares show the median Spearman correlation between the full donor-balanced map and its five LODO reconstructions.

Supplementary Figure 1. Overlapping evidence-count summary across the 160 tested DTHI–external map pairs. Exact-bijective global and family maxT, global and family BH-FDR, nested LH46 BH-FDR, and the final priority-reporting set are shown separately. Categories overlap and should not be summed.

Supplementary Figure 2. Integrated exact-bijective evidence classification for all 160 DTHI–external map pairs. Only evidence classes actually present in the integrated results are displayed. Blank cells indicate no corrected spatial support, while coloured cells indicate exact-bijective global FWER support, family FDR support, or nominal-only exact-bijective support.
"""
    LEGENDS_OUT.write_text(legends, encoding="utf-8")

    manifest_rows = []
    for label, path in generated:
        manifest_rows.append({
            "figure": label,
            "relative_path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "SHA256": sha256(path),
        })
    manifest_rows.append({
        "figure": "Figure legends",
        "relative_path": str(LEGENDS_OUT.relative_to(ROOT)),
        "size_bytes": LEGENDS_OUT.stat().st_size,
        "SHA256": sha256(LEGENDS_OUT),
    })
    manifest = pd.DataFrame(manifest_rows)
    atomic_tsv(manifest, MANIFEST_OUT)

    audit = pd.DataFrame([
        {"section": "upstream", "item": "Phase8D2C4_completed", "value": True, "passed": True},
        {"section": "inputs", "item": "integrated_pairs", "value": len(integrated), "passed": len(integrated) == 160},
        {"section": "inputs", "item": "priority_pairs", "value": len(priority), "passed": len(priority) == 5},
        {"section": "figures", "item": "revised_figure_designs", "value": len(figures), "passed": len(figures) == 5},
        {"section": "figures", "item": "main_figures", "value": 3, "passed": True},
        {"section": "figures", "item": "supplementary_figures", "value": 2, "passed": True},
        {"section": "outputs", "item": "all_manifest_files_exist", "value": all((ROOT / p).is_file() for p in manifest["relative_path"]), "passed": all((ROOT / p).is_file() for p in manifest["relative_path"])},
    ])
    atomic_tsv(audit, AUDIT_OUT)

    completion = pd.DataFrame([{
        "Phase8D2C4_status_confirmed": True,
        "integrated_pairs_visualized": len(integrated),
        "priority_pairs_visualized": len(priority),
        "revised_figure_designs": len(figures),
        "main_figures": 3,
        "supplementary_figures": 2,
        "raster_vector_pdf_outputs": 15,
        "combined_pdf_pages": len(figures),
        "all_revised_figure_files_present": bool(audit.loc[audit["item"] == "all_manifest_files_exist", "passed"].iloc[0]),
        "ready_for_manuscript_figure_lock": True,
        "Phase8D2C4B_status": "completed",
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
    }])
    atomic_tsv(completion, COMPLETION_OUT)

    print("===== PHASE 8D2C4B COMPLETION =====")
    print(completion.to_string(index=False))
    print("\n===== FIGURE MANIFEST =====")
    print(manifest.to_string(index=False))
    print("\n===== AUDIT =====")
    print(audit.to_string(index=False))
    print("\n===== CANONICAL OUTPUT HASHES =====")
    extra = pd.DataFrame([
        {"figure": "Figure revision audit", "relative_path": str(AUDIT_OUT.relative_to(ROOT)), "size_bytes": AUDIT_OUT.stat().st_size, "SHA256": sha256(AUDIT_OUT)},
        {"figure": "Completion summary", "relative_path": str(COMPLETION_OUT.relative_to(ROOT)), "size_bytes": COMPLETION_OUT.stat().st_size, "SHA256": sha256(COMPLETION_OUT)},
    ])
    canonical = pd.concat([manifest, extra], ignore_index=True)
    print(canonical[["relative_path", "size_bytes", "SHA256"]].to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Phase 8D2C4B failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise
