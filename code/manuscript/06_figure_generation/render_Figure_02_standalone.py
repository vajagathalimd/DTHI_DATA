from pathlib import Path
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT = Path(".")

TABLE_DIR = PROJECT / "07_tables/main_tables/phase7"

ATLAS_FILE = TABLE_DIR / "phase7F1_final_integrated_regulatory_atlas.tsv"
SUMMARY_FILE = TABLE_DIR / "phase7F1_integrated_evidence_class_summary.tsv"

OUT_DIR = PROJECT / "Publication_manuscript/Figure_corrections/Figure_02"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Figure_02_FINAL.png"
OUT_PDF = OUT_DIR / "Figure_02_FINAL.pdf"

# ============================================================
# HELPERS
# ============================================================

def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

def find_first(df, exact=None, token_groups=None, required=True, label="column"):
    exact = exact or []
    token_groups = token_groups or []

    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    for c in exact:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    for tokens in token_groups:
        for c in cols:
            lc = c.lower()
            if all(tok.lower() in lc for tok in tokens):
                return c

    if required:
        raise RuntimeError(
            f"Could not find {label}. Available columns:\n"
            + "\n".join(cols)
        )
    return None

def parse_bool_series(series: pd.Series) -> pd.Series:
    mapping = {
        "true": True,
        "t": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "f": False,
        "0": False,
        "no": False,
        "n": False,
    }

    def _one(x):
        if pd.isna(x):
            return False
        if isinstance(x, bool):
            return x
        s = str(x).strip().lower()
        if s in mapping:
            return mapping[s]
        return False

    return series.map(_one)

def support_from_column(df, exact=None, token_groups=None, label="support"):
    col = find_first(
        df,
        exact=exact,
        token_groups=token_groups,
        required=False,
        label=label,
    )
    if col is None:
        return None, None

    numeric = pd.to_numeric(df[col], errors="coerce")
    if numeric.notna().sum() >= max(3, len(df) // 3):
        return col, numeric.fillna(0).gt(0).astype(int)

    boolean = parse_bool_series(df[col]).astype(int)
    return col, boolean

def short_class_label(x):
    if pd.isna(x):
        return "-"
    s = str(x)
    parts = re.split(r"[_\s]+", s)
    for p in parts:
        if re.fullmatch(r"F\d+", p):
            return p
    return s.replace("_", " ")

def prettify_program(x):
    if pd.isna(x):
        return "-"
    s = str(x).strip()
    mapping = {
        "fetal_high": "Fetal-high",
        "maturation_high": "Maturation-high",
        "decreasing": "Fetal-high",
        "increasing": "Maturation-high",
    }
    return mapping.get(s, s.replace("_", " ").title())

def wrap_text(x, width):
    if pd.isna(x):
        return "-"
    s = str(x)
    return "\n".join(textwrap.wrap(s, width=width)) if len(s) > width else s

# ============================================================
# LOAD
# ============================================================

require_file(ATLAS_FILE)
require_file(SUMMARY_FILE)

atlas = pd.read_csv(ATLAS_FILE, sep="\t")
summary = pd.read_csv(SUMMARY_FILE, sep="\t")

print("Loaded atlas:", ATLAS_FILE)
print("Rows:", len(atlas), "Columns:", len(atlas.columns))
print("Loaded summary:", SUMMARY_FILE)
print("Rows:", len(summary), "Columns:", len(summary.columns))

# ============================================================
# REQUIRED COLUMNS
# ============================================================

reg_col = find_first(atlas, exact=["regulator"], label="regulator")
rank_col = find_first(atlas, exact=["final_integrated_rank"], label="final integrated rank")
score_col = find_first(
    atlas,
    exact=["descriptive_multilayer_evidence_score"],
    token_groups=[("multilayer", "score")],
    label="descriptive multilayer evidence score",
)

program_col = find_first(
    atlas,
    exact=["developmental_program_direction"],
    token_groups=[("developmental", "program"), ("program", "direction")],
    label="developmental program direction",
)

fav_neuron_col = find_first(
    atlas,
    exact=["adult_favored_neuronal_class"],
    token_groups=[("favored", "neuronal", "class")],
    label="adult favored neuronal class",
)

adult_broad_col = find_first(
    atlas,
    exact=["highest_mean_adult_broad_cell_class"],
    token_groups=[("highest", "adult", "broad", "class")],
    label="highest adult broad class",
)

class_col = find_first(
    atlas,
    exact=["final_phase7_evidence_class"],
    token_groups=[("final", "evidence", "class")],
    label="final evidence class",
)

tier_source_col = class_col

# ============================================================
# SORT CANONICAL ORDER
# ============================================================

atlas[rank_col] = pd.to_numeric(atlas[rank_col], errors="coerce")
atlas[score_col] = pd.to_numeric(atlas[score_col], errors="coerce")

plot_atlas = (
    atlas
    .sort_values(rank_col, kind="stable")
    .reset_index(drop=True)
)

regulator_order = plot_atlas[reg_col].astype(str).tolist()

# ============================================================
# PANEL A — EVIDENCE MATRIX
# ============================================================

tier_col = find_first(
    plot_atlas,
    exact=["regulatory_evidence_tier"],
    token_groups=[("regulatory", "evidence", "tier")],
    required=False,
    label="regulatory evidence tier",
)

matrix = pd.DataFrame(index=regulator_order)

if tier_col is not None:
    matrix["Stringent temporal"] = (
        plot_atlas[tier_col]
        .astype(str)
        .str.contains("stringent", case=False, na=False)
        .astype(int)
        .to_numpy()
    )

col, vals = support_from_column(
    plot_atlas,
    exact=["celltype_localization_score"],
    token_groups=[("celltype", "localization"), ("cell", "type", "localization")],
    label="cell-type localization",
)
if vals is not None:
    matrix["Cell-type localization"] = vals.to_numpy()

col, vals = support_from_column(
    plot_atlas,
    exact=["cis_subset_specificity_score"],
    token_groups=[("cis", "specificity"), ("direct", "cis"), ("cis", "score")],
    label="direct cis-regulation",
)
if vals is not None:
    matrix["Direct cis-regulation"] = vals.to_numpy()

col, vals = support_from_column(
    plot_atlas,
    exact=[
        "primary_perturbation_support_score",
        "perturbation_support_score",
        "perturbational_support_score",
    ],
    token_groups=[("perturb", "support"), ("perturbation", "score"), ("primary", "perturb")],
    label="perturbational support",
)
if vals is not None:
    matrix["Perturbational support"] = vals.to_numpy()

col, vals = support_from_column(
    plot_atlas,
    exact=[
        "DTHI_support_score",
        "dthi_support_score",
        "developmental_hierarchy_support_score",
    ],
    token_groups=[("dthi", "support"), ("dthi", "score"), ("hierarchy", "support")],
    label="DTHI support",
)
if vals is not None:
    matrix["DTHI support"] = vals.to_numpy()

col, vals = support_from_column(
    plot_atlas,
    exact=[
        "developmental_program_impact_score",
        "program_impact_score",
        "broad_program_impact_score",
    ],
    token_groups=[("program", "impact"), ("developmental", "impact"), ("broad", "impact")],
    label="program impact",
)
if vals is not None:
    matrix["Program impact"] = vals.to_numpy()

col, vals = support_from_column(
    plot_atlas,
    exact=[
        "sensitivity_support_score",
        "small_regulon_sensitivity_score",
    ],
    token_groups=[("sensitivity", "score"), ("small", "regulon"), ("sensitivity", "evidence")],
    label="sensitivity evidence",
)
if vals is not None:
    matrix["Sensitivity evidence"] = vals.to_numpy()
else:
    sensitivity_from_text = (
        plot_atlas.apply(
            lambda r:
                (
                    ("sensitivity" in str(r).lower())
                    or
                    (str(r[reg_col]) in {"RFXAP", "SRSF2"})
                ),
            axis=1,
        )
        .astype(int)
    )
    if sensitivity_from_text.sum() > 0:
        matrix["Sensitivity evidence"] = sensitivity_from_text.to_numpy()

if matrix.shape[1] < 4:
    raise RuntimeError(
        "Too few evidence columns were reconstructed for Panel A.\n"
        f"Recovered columns: {list(matrix.columns)}"
    )

print("\nPanel A reconstructed columns:")
for c in matrix.columns:
    print(" -", c)

# ============================================================
# PANEL C/D PREP
# ============================================================

plot_atlas["final_tier_short"] = plot_atlas[tier_source_col].map(short_class_label)

table_df = pd.DataFrame({
    "Regulator": plot_atlas[reg_col].astype(str),
    "Developmental\nprogram": plot_atlas[program_col].map(prettify_program),
    "Favored neuronal\nclass": plot_atlas[fav_neuron_col].map(lambda x: wrap_text(x, 16)),
    "Highest adult\nbroad class": plot_atlas[adult_broad_col].map(lambda x: wrap_text(x, 16)),
    "Final\ntier": plot_atlas["final_tier_short"],
})

summary_class_col = find_first(
    summary,
    exact=["final_phase7_evidence_class"],
    token_groups=[("final", "evidence", "class")],
    label="summary final evidence class",
)

summary_rank_col = find_first(
    summary,
    exact=["final_phase7_evidence_class_rank"],
    token_groups=[("evidence", "class", "rank")],
    label="summary evidence class rank",
)

summary_count_col = find_first(
    summary,
    exact=["TF_count"],
    token_groups=[("tf", "count")],
    label="summary TF count",
)

summary_tfs_col = find_first(
    summary,
    exact=["TFs"],
    token_groups=[("tf",)],
    label="summary TF list",
)

class_plot = (
    summary
    .sort_values(summary_rank_col, kind="stable")
    .reset_index(drop=True)
)

class_plot["class_short"] = class_plot[summary_class_col].map(short_class_label)
class_plot[summary_count_col] = pd.to_numeric(class_plot[summary_count_col], errors="coerce").fillna(0).astype(int)

# ============================================================
# BUILD FIGURE
# ============================================================

fig = plt.figure(figsize=(16.5, 11.5))
gs = fig.add_gridspec(
    2,
    2,
    width_ratios=[1.35, 1.0],
    height_ratios=[1.0, 1.0],
    left=0.05,
    right=0.98,
    top=0.96,
    bottom=0.06,
    wspace=0.24,
    hspace=0.25,
)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])

# ------------------------------------------------------------
# Panel A
# ------------------------------------------------------------

mat = matrix.to_numpy(dtype=int)
cmap = ListedColormap(["white", "black"])
axA.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)

axA.set_xticks(np.arange(matrix.shape[1]))
axA.set_xticklabels(
    list(matrix.columns),
    rotation=32,
    ha="right",
    fontsize=10,
)

axA.set_yticks(np.arange(matrix.shape[0]))
axA.set_yticklabels(regulator_order, fontsize=10)

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        axA.text(
            j,
            i,
            "✓" if v == 1 else "–",
            ha="center",
            va="center",
            fontsize=11,
            color="white" if v == 1 else "black",
            fontweight="bold",
        )

axA.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
axA.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
axA.grid(which="minor", color="0.75", linestyle="-", linewidth=0.8)
axA.tick_params(which="minor", bottom=False, left=False)

axA.set_title(
    "A. Multilayer regulatory evidence",
    fontsize=13,
    fontweight="bold",
    pad=10,
)

# ------------------------------------------------------------
# Panel B
# ------------------------------------------------------------

bar_y = np.arange(len(plot_atlas))
axB.barh(bar_y, plot_atlas[score_col].to_numpy(), edgecolor="black", linewidth=0.7)
axB.set_yticks(bar_y)
axB.set_yticklabels(regulator_order, fontsize=10)
axB.invert_yaxis()
axB.set_xlabel("Descriptive multilayer evidence score", fontsize=10)
axB.set_title(
    "B. Integrated evidence ranking",
    fontsize=13,
    fontweight="bold",
    pad=10,
)

for i, value in enumerate(plot_atlas[score_col].to_numpy()):
    if pd.notna(value):
        axB.text(
            value + 0.03,
            i,
            f"{value:.1f}" if abs(value - round(value)) > 1e-9 else f"{int(round(value))}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

# ------------------------------------------------------------
# Panel C
# ------------------------------------------------------------

axC.axis("off")

table = axC.table(
    cellText=table_df.values.tolist(),
    colLabels=table_df.columns.tolist(),
    cellLoc="center",
    rowLoc="center",
    loc="center",
    bbox=[0.0, 0.0, 1.0, 0.94],
)

table.auto_set_font_size(False)
table.set_fontsize(8)

for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_text_props(fontweight="bold")
        cell.set_linewidth(0.8)
    else:
        cell.set_linewidth(0.5)
    if c == 0 and r > 0:
        cell.set_text_props(fontweight="bold")

axC.set_title(
    "C. Developmental and cell-type context",
    fontsize=13,
    fontweight="bold",
    pad=10,
)

# ------------------------------------------------------------
# Panel D
# ------------------------------------------------------------

x = np.arange(len(class_plot))
counts = class_plot[summary_count_col].to_numpy(dtype=int)

axD.bar(x, counts, edgecolor="black", linewidth=0.8)
axD.set_xticks(x)
axD.set_xticklabels(class_plot["class_short"].tolist(), fontsize=10)
axD.set_ylabel("Number of regulators", fontsize=10)
axD.set_title(
    "D. Final evidence classes",
    fontsize=13,
    fontweight="bold",
    pad=10,
)

for i, (_, row) in enumerate(class_plot.iterrows()):
    count = int(row[summary_count_col])
    tf_text = str(row[summary_tfs_col]).replace("|", "\n")
    axD.text(
        i,
        count / 2 if count > 0 else 0.05,
        tf_text,
        ha="center",
        va="center",
        fontsize=8,
        color="white" if count > 0 else "black",
        fontweight="bold",
    )
    axD.text(
        i,
        count + 0.08,
        str(count),
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

axD.set_ylim(0, max(counts.max() + 0.8, 1.0))

# ============================================================
# SAVE
# ============================================================

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
plt.close(fig)

print("\n============================================================")
print("PASS: Figure 2 standalone rebuild completed.")
print("PNG:", OUT_PNG)
print("PDF:", OUT_PDF)
print("============================================================")
