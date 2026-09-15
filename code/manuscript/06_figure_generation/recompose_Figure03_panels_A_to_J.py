from pathlib import Path
import fitz
import sys

PROJECT = Path(".")
OUT = PROJECT / "Publication_manuscript/Figure_corrections/Figure_03"
PANEL_DIR = OUT / "panels"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

F45 = OUT / "Figure45_clean_flattened.pdf"
F46 = OUT / "Figure46_clean_flattened.pdf"
F47 = OUT / "Figure47_clean_flattened.pdf"

FINAL_PDF = OUT / "Figure_03_perturbational_transcriptomic_evidence_hierarchy_FINAL.pdf"
FINAL_PNG = OUT / "Figure_03_perturbational_transcriptomic_evidence_hierarchy_FINAL.png"

for f in (F45, F46, F47):
    if not f.exists():
        print("ERROR: missing cleaned source:", f)
        sys.exit(1)


# ============================================================
# Utility functions
# ============================================================

def nr(page, x0, y0, x1, y1):
    """Normalized rectangle -> page coordinates."""
    W = page.rect.width
    H = page.rect.height
    return fitz.Rect(
        x0 * W,
        y0 * H,
        x1 * W,
        y1 * H,
    )


def remove_text_blocks(pdf_path, phrases, output_path):
    """
    Remove only specified figure-level notes.
    No plotted data are modified.
    """
    doc = fitz.open(pdf_path)
    page = doc[0]

    removed = 0

    for phrase in phrases:
        hits = page.search_for(phrase)

        for r in hits:
            erase = fitz.Rect(
                max(0, r.x0 - 4),
                max(0, r.y0 - 3),
                min(page.rect.width, r.x1 + 4),
                min(page.rect.height, r.y1 + 3),
            )
            page.add_redact_annot(
                erase,
                fill=(1, 1, 1)
            )
            removed += 1

    if removed:
        page.apply_redactions()

    doc.save(output_path)
    doc.close()

    print(
        f"Footnote cleanup: {removed} matched text region(s) removed"
    )


def save_panel(source_doc, clip, panel_name):
    """
    Save each extracted panel as an independent vector PDF
    plus a high-resolution PNG for visual QA.
    """
    src_page = source_doc[0]

    panel_pdf = PANEL_DIR / f"Panel_{panel_name}.pdf"
    panel_png = PANEL_DIR / f"Panel_{panel_name}.png"

    pdoc = fitz.open()
    ppage = pdoc.new_page(
        width=clip.width,
        height=clip.height
    )

    ppage.show_pdf_page(
        ppage.rect,
        source_doc,
        0,
        clip=clip,
        keep_proportion=True
    )

    pdoc.save(panel_pdf)
    pdoc.close()

    check = fitz.open(panel_pdf)
    pix = check[0].get_pixmap(
        matrix=fitz.Matrix(2.5, 2.5),
        alpha=False
    )
    pix.save(panel_png)
    check.close()

    return panel_pdf


# ============================================================
# Remove the Figure 46 explanatory note from graphic.
# The information belongs in the manuscript legend.
# ============================================================

F46_NO_NOTE = OUT / "Figure46_clean_flattened_no_note.pdf"

remove_text_blocks(
    F46,
    [
        "G/F:",
        "empirical direction differs from the absolute-score sign",
        "null-relative support with matching absolute direction",
    ],
    F46_NO_NOTE
)


# ============================================================
# Open cleaned canonical source figures.
# ============================================================

d45 = fitz.open(F45)
d46 = fitz.open(F46_NO_NOTE)
d47 = fitz.open(F47)

p45 = d45[0]
p46 = d46[0]
p47 = d47[0]


# ============================================================
# PANEL EXTRACTION
#
# These clips deliberately include axis labels and titles but
# exclude adjacent panels. B and D receive additional left
# margin because their long y-axis labels caused the overlap.
# ============================================================

clips = {
    # Figure 45
    "A": (d45, nr(p45, 0.000, 0.045, 0.410, 0.505)),
    "B": (d45, nr(p45, 0.410, 0.045, 1.000, 0.505)),
    "C": (d45, nr(p45, 0.000, 0.490, 0.410, 0.990)),
    "D": (d45, nr(p45, 0.410, 0.490, 1.000, 0.990)),

    # Figure 46
    "E": (d46, nr(p46, 0.000, 0.025, 0.500, 0.935)),
    "F": (d46, nr(p46, 0.500, 0.025, 1.000, 0.935)),

    # Figure 47
    "G": (d47, nr(p47, 0.000, 0.040, 0.500, 0.515)),
    "H": (d47, nr(p47, 0.500, 0.040, 1.000, 0.515)),
    "I": (d47, nr(p47, 0.000, 0.495, 0.500, 0.995)),
    "J": (d47, nr(p47, 0.500, 0.495, 1.000, 0.995)),
}


# ============================================================
# Save A–J individually first.
# ============================================================

panel_paths = {}

for label in "ABCDEFGHIJ":
    src_doc, clip = clips[label]

    panel_paths[label] = save_panel(
        src_doc,
        clip,
        label
    )

    print(
        f"PASS panel {label}: "
        f"{clip.width:.1f} × {clip.height:.1f} pt"
    )


d45.close()
d46.close()
d47.close()


# ============================================================
# FINAL PUBLICATION LAYOUT
#
# 2 columns × 5 rows:
#
# A | B
# C | D
# E | F
# G | H
# I | J
#
# Right column receives slightly more width because B/D/F/J
# carry longer labels. Large horizontal gutter prevents
# labels from crossing into neighbouring panels.
# ============================================================

pairs = [
    ("A", "B"),
    ("C", "D"),
    ("E", "F"),
    ("G", "H"),
    ("I", "J"),
]

opened = {
    k: fitz.open(v)
    for k, v in panel_paths.items()
}

PAGE_WIDTH = 1320.0

LEFT_MARGIN = 35.0
RIGHT_MARGIN = 35.0

# Deliberately generous inter-panel spacing.
COLUMN_GUTTER = 120.0

TOP_MARGIN = 28.0
BOTTOM_MARGIN = 32.0
ROW_GUTTER = 42.0

available_width = (
    PAGE_WIDTH
    - LEFT_MARGIN
    - RIGHT_MARGIN
    - COLUMN_GUTTER
)

LEFT_WIDTH = available_width * 0.45
RIGHT_WIDTH = available_width * 0.55


def scaled_height(doc, target_width):
    r = doc[0].rect
    return r.height * target_width / r.width


row_heights = []

for left, right in pairs:
    lh = scaled_height(
        opened[left],
        LEFT_WIDTH
    )
    rh = scaled_height(
        opened[right],
        RIGHT_WIDTH
    )

    row_heights.append(
        max(lh, rh)
    )


PAGE_HEIGHT = (
    TOP_MARGIN
    + BOTTOM_MARGIN
    + sum(row_heights)
    + ROW_GUTTER * (len(pairs) - 1)
)

final = fitz.open()
page = final.new_page(
    width=PAGE_WIDTH,
    height=PAGE_HEIGHT
)

current_y = TOP_MARGIN

for row_index, ((left, right), row_height) in enumerate(
    zip(pairs, row_heights),
    start=1
):
    left_doc = opened[left]
    right_doc = opened[right]

    left_h = scaled_height(
        left_doc,
        LEFT_WIDTH
    )

    right_h = scaled_height(
        right_doc,
        RIGHT_WIDTH
    )

    # Top-align both panels in each row.
    left_rect = fitz.Rect(
        LEFT_MARGIN,
        current_y,
        LEFT_MARGIN + LEFT_WIDTH,
        current_y + left_h,
    )

    right_x = (
        LEFT_MARGIN
        + LEFT_WIDTH
        + COLUMN_GUTTER
    )

    right_rect = fitz.Rect(
        right_x,
        current_y,
        right_x + RIGHT_WIDTH,
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
        f"ROW {row_index}: "
        f"{left} | {right}"
    )

    current_y += (
        row_height
        + ROW_GUTTER
    )


# ============================================================
# Save vector PDF.
# ============================================================

final.save(
    FINAL_PDF,
    garbage=4,
    deflate=True
)

final.close()

for doc in opened.values():
    doc.close()


# ============================================================
# High-resolution PNG for manuscript/visual QA.
# ============================================================

check = fitz.open(FINAL_PDF)
pix = check[0].get_pixmap(
    matrix=fitz.Matrix(3.2, 3.2),
    alpha=False
)
pix.save(FINAL_PNG)
check.close()


print()
print("============================================================")
print("FINAL FIGURE 3 RECOMPOSITION COMPLETE")
print("============================================================")
print("Scientific analyses recomputed: NO")
print("Canonical statistical values modified: NO")
print("Nested panel lettering: REMOVED")
print("Legacy Figure 47 heading: REMOVED")
print("E/F explanatory footnote: REMOVED FROM GRAPHIC")
print("Final sequence: A–J")
print()
print("PDF:", FINAL_PDF)
print("PNG:", FINAL_PNG)
print("Individual panels:", PANEL_DIR)
print("============================================================")
