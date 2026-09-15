from pathlib import Path
import pymupdf as fitz

ROOT = Path(
    "./"
    "Publication_manuscript/Figure_corrections/Figure_03/individual_clean"
)

PANELS = {
    "A": ROOT / "Figure3A_CMAP_evidence_funnel.pdf",
    "B": ROOT / "Figure3B_evidence_hierarchy.pdf",
    "C": ROOT / "Figure3C_chemical_replication.pdf",
    "D": ROOT / "Figure3D_leave_program_out_heatmap.pdf",
    "E": ROOT / "Figure3E_empirical_LINCS_zscores.pdf",
    "F": ROOT / "Figure3F_late_program_vs_injury.pdf",
    "G": ROOT / "Figure3G_direct_quantitative_LINCS.pdf",
    "H": ROOT / "Figure3H_CMAP_only_provisional.pdf",
    "I": ROOT / "Figure3I_evidence_layer_availability.pdf",
    "J": ROOT / "Figure3J_final_reporting_groups.pdf",
}

PAIRS = [
    ("A", "B"),
    ("C", "D"),
    ("E", "F"),
    ("G", "H"),
    ("I", "J"),
]

# ============================================================
# VERIFY INPUTS
# ============================================================

missing = [
    str(path)
    for path in PANELS.values()
    if not path.exists()
]

if missing:
    raise FileNotFoundError(
        "Missing clean panel PDFs:\n" + "\n".join(missing)
    )

docs = {
    label: fitz.open(path)
    for label, path in PANELS.items()
}

# ============================================================
# PUBLICATION LAYOUT
# ============================================================

PAGE_WIDTH = 1400.0

LEFT_MARGIN = 34.0
RIGHT_MARGIN = 34.0
TOP_MARGIN = 32.0
BOTTOM_MARGIN = 34.0

COLUMN_GUTTER = 30.0
ROW_GUTTER = 28.0

COLUMN_WIDTH = (
    PAGE_WIDTH
    - LEFT_MARGIN
    - RIGHT_MARGIN
    - COLUMN_GUTTER
) / 2.0


def scaled_height(doc, width):
    r = doc[0].rect
    return width * r.height / r.width


row_heights = []

for left, right in PAIRS:
    lh = scaled_height(docs[left], COLUMN_WIDTH)
    rh = scaled_height(docs[right], COLUMN_WIDTH)
    row_heights.append(max(lh, rh))


PAGE_HEIGHT = (
    TOP_MARGIN
    + BOTTOM_MARGIN
    + sum(row_heights)
    + ROW_GUTTER * (len(PAIRS) - 1)
)

# ============================================================
# BUILD VECTOR COLLAGE
# ============================================================

final_doc = fitz.open()

page = final_doc.new_page(
    width=PAGE_WIDTH,
    height=PAGE_HEIGHT
)

current_y = TOP_MARGIN

for row_number, ((left, right), row_height) in enumerate(
    zip(PAIRS, row_heights),
    start=1
):
    left_doc = docs[left]
    right_doc = docs[right]

    left_h = scaled_height(left_doc, COLUMN_WIDTH)
    right_h = scaled_height(right_doc, COLUMN_WIDTH)

    left_rect = fitz.Rect(
        LEFT_MARGIN,
        current_y,
        LEFT_MARGIN + COLUMN_WIDTH,
        current_y + left_h,
    )

    right_x = (
        LEFT_MARGIN
        + COLUMN_WIDTH
        + COLUMN_GUTTER
    )

    right_rect = fitz.Rect(
        right_x,
        current_y,
        right_x + COLUMN_WIDTH,
        current_y + right_h,
    )

    page.show_pdf_page(
        left_rect,
        left_doc,
        0,
        keep_proportion=True
    )

    page.show_pdf_page(
        right_rect,
        right_doc,
        0,
        keep_proportion=True
    )

    print(
        f"PASS row {row_number}: {left} | {right}"
    )

    current_y += row_height + ROW_GUTTER


PDF_OUT = (
    ROOT
    / "Figure_03_perturbational_transcriptomic_evidence_hierarchy_FINAL_COLLAGE.pdf"
)

PNG_OUT = (
    ROOT
    / "Figure_03_perturbational_transcriptomic_evidence_hierarchy_FINAL_COLLAGE.png"
)

final_doc.save(
    PDF_OUT,
    garbage=4,
    deflate=True
)

final_doc.close()

for doc in docs.values():
    doc.close()

# ============================================================
# HIGH-RESOLUTION PNG
#
# 3× rendering gives ~4200 px width and >500 dpi-equivalent
# at a standard journal two-column print width.
# ============================================================

check = fitz.open(PDF_OUT)

pix = check[0].get_pixmap(
    matrix=fitz.Matrix(3.0, 3.0),
    alpha=False
)

pix.save(PNG_OUT)

check.close()

print()
print("============================================================")
print("FINAL FIGURE 3 COLLAGE COMPLETE")
print("============================================================")
print("Layout:")
print("  A | B")
print("  C | D")
print("  E | F")
print("  G | H")
print("  I | J")
print()
print("Panel cropping: NO")
print("Scientific analysis rerun: NO")
print("Panel values modified: NO")
print("Source: clean individual vector PDFs")
print()
print("PDF:", PDF_OUT)
print("PNG:", PNG_OUT)
print("============================================================")
