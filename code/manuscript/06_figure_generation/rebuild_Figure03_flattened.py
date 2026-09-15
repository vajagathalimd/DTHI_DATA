from pathlib import Path
import sys

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF/fitz is not installed in this Python environment.")
    print("Please stop here and paste this message back to ChatGPT.")
    sys.exit(1)

PROJECT = Path(".")

SRC = PROJECT / "08_figures/main_figures/phase6"
OUT = PROJECT / "Publication_manuscript/Figure_corrections/Figure_03"

FIG45 = SRC / "Figure45_CMAP_robustness_funnel_and_residual_mechanisms.pdf"
FIG46 = SRC / "Figure46_quantitative_LINCS_empirical_validation.pdf"
FIG47 = SRC / "Figure_47_integrated_perturbational_evidence_atlas.pdf"

CLEAN45 = OUT / "Figure45_clean_flattened.pdf"
CLEAN46 = OUT / "Figure46_clean_flattened.pdf"
CLEAN47 = OUT / "Figure47_clean_flattened.pdf"

FINAL_PDF = OUT / "Figure_03_perturbational_transcriptomic_evidence_hierarchy_CORRECTED.pdf"
FINAL_PNG = OUT / "Figure_03_perturbational_transcriptomic_evidence_hierarchy_CORRECTED.png"

for f in [FIG45, FIG46, FIG47]:
    if not f.exists():
        raise FileNotFoundError(f)

def union_rect(rects):
    r = fitz.Rect(rects[0])
    for rr in rects[1:]:
        r |= rr
    return r

def replace_text(page, old_candidates, new_text, fontsize=11):
    """
    Replace a plot title using the existing vector PDF.
    Stops if no candidate string can be found.
    """
    hits = []
    matched = None

    for old in old_candidates:
        found = page.search_for(old)
        if found:
            hits = found
            matched = old
            break

    if not hits:
        print("WARNING: title not found:")
        for old in old_candidates:
            print("   ", repr(old))
        return False

    r = union_rect(hits)

    # Slight enlargement to erase complete old glyphs.
    erase = fitz.Rect(
        max(0, r.x0 - 3),
        max(0, r.y0 - 2),
        min(page.rect.width, r.x1 + 4),
        min(page.rect.height, r.y1 + 3),
    )

    page.add_redact_annot(erase, fill=(1, 1, 1))
    page.apply_redactions()

    # Give the replacement enough horizontal room.
    write_rect = fitz.Rect(
        erase.x0,
        erase.y0 - 1,
        min(page.rect.width - 3, erase.x0 + max(erase.width * 1.40, 260)),
        erase.y1 + 5,
    )

    size = fontsize

    while size >= 7:
        rc = page.insert_textbox(
            write_rect,
            new_text,
            fontsize=size,
            fontname="helv",
            color=(0, 0, 0),
            align=0,
        )
        if rc >= 0:
            break
        size -= 0.5

    print(f"PASS: {matched!r}")
    print(f"   -> {new_text!r}")
    return True


def remove_title(page, candidates):
    for text in candidates:
        hits = page.search_for(text)
        if hits:
            r = union_rect(hits)

            # Numbered title occupies the central top margin.
            erase = fitz.Rect(
                0,
                max(0, r.y0 - 5),
                page.rect.width,
                min(page.rect.height, r.y1 + 8),
            )

            page.add_redact_annot(
                erase,
                fill=(1, 1, 1),
            )
            page.apply_redactions()

            print(f"PASS: removed legacy heading {text!r}")
            return True

    print("WARNING: legacy heading not found:", candidates)
    return False


# ============================================================
# FIGURE 45 → final panels A–D
# ============================================================

doc = fitz.open(FIG45)
page = doc[0]

replace_text(
    page,
    [
        "A  Perturbational-validation evidence funnel",
        "A Perturbational-validation evidence funnel",
        "A. Perturbational-validation evidence funnel",
    ],
    "A. Perturbational-validation evidence funnel",
    fontsize=11,
)

replace_text(
    page,
    [
        "B  Evidence hierarchy after developmental-program removal",
        "B Evidence hierarchy after developmental-program removal",
        "B. Evidence hierarchy after developmental-program removal",
    ],
    "B. Evidence hierarchy after developmental-program removal",
    fontsize=11,
)

# Original source placed D bottom-left and C bottom-right.
# Relabel them in normal reading order.

replace_text(
    page,
    [
        "D  Chemical-level replication of residual mechanisms",
        "D Chemical-level replication of residual mechanisms",
        "D. Chemical-level replication of residual mechanisms",
    ],
    "C. Chemical-level replication of residual mechanisms",
    fontsize=11,
)

replace_text(
    page,
    [
        "C  Leave-program-out mechanism scores for 15 globally robust parents",
        "C Leave-program-out mechanism scores for 15 globally robust parents",
        "C. Leave-program-out mechanism scores for 15 globally robust parents",
    ],
    "D. Leave-program-out mechanism scores for globally robust parents",
    fontsize=10.5,
)

# Remove source-level heading so Figure 3 has no competing embedded title.
remove_title(
    page,
    [
        "CMAP developmental-state concordance is dominated by program overlap and residual injury/proliferative mechanisms",
        "CMAP developmental-state concordance is dominated by programme overlap and residual injury/proliferative mechanisms",
    ],
)

doc.save(CLEAN45)
doc.close()


# ============================================================
# FIGURE 46 → final panels E–F
# ============================================================

doc = fitz.open(FIG46)
page = doc[0]

replace_text(
    page,
    [
        "A. Empirical z-scores in cell-balanced LINCS profiles",
        "A Empirical z-scores in cell-balanced LINCS profiles",
    ],
    "E. Empirical z-scores in cell-balanced LINCS profiles",
    fontsize=11,
)

replace_text(
    page,
    [
        "B. Absolute late-program and injury responses",
        "B Absolute late-program and injury responses",
    ],
    "F. Late-program support versus injury/stress",
    fontsize=11,
)

doc.save(CLEAN46)
doc.close()


# ============================================================
# FIGURE 47 → final panels G–J
# ============================================================

doc = fitz.open(FIG47)
page = doc[0]

replace_text(
    page,
    ["A. Direct quantitative LINCS chemicals"],
    "G. Direct quantitative LINCS chemicals",
    fontsize=11,
)

replace_text(
    page,
    ["B. Chemicals without direct quantitative LINCS profiles"],
    "H. CMap-only provisional chemicals",
    fontsize=11,
)

replace_text(
    page,
    ["C. Evidence-layer availability and replication"],
    "I. Evidence-layer availability and replication",
    fontsize=11,
)

replace_text(
    page,
    ["D. Final manuscript reporting groups"],
    "J. Final perturbational reporting groups",
    fontsize=11,
)

remove_title(
    page,
    [
        "Figure 47. Integrated perturbational evidence atlas for developmental transcriptomic programs",
        "Figure 47. Integrated perturbational evidence atlas for developmental transcriptomic programmes",
    ],
)

doc.save(CLEAN47)
doc.close()


# ============================================================
# Assemble clean publication Figure 3
# No outer A/B/C block lettering.
# ============================================================

sources = [
    fitz.open(CLEAN45),
    fitz.open(CLEAN46),
    fitz.open(CLEAN47),
]

# Use common output width and retain each source aspect ratio.
target_width = max(d[0].rect.width for d in sources)
gap = 14

scaled_heights = [
    d[0].rect.height * target_width / d[0].rect.width
    for d in sources
]

total_height = sum(scaled_heights) + gap * (len(sources) - 1)

out = fitz.open()
page = out.new_page(
    width=target_width,
    height=total_height,
)

y = 0

for source, h in zip(sources, scaled_heights):
    rect = fitz.Rect(
        0,
        y,
        target_width,
        y + h,
    )

    page.show_pdf_page(
        rect,
        source,
        0,
        keep_proportion=True,
    )

    y += h + gap

out.save(FINAL_PDF)

for source in sources:
    source.close()

out.close()


# ============================================================
# Render high-resolution PNG from final vector PDF
# ============================================================

doc = fitz.open(FINAL_PDF)
page = doc[0]

# ~4× PDF point resolution; comparable to the previous
# publication PNG while retaining readable panel text.
pix = page.get_pixmap(
    matrix=fitz.Matrix(4, 4),
    alpha=False,
)

pix.save(FINAL_PNG)
doc.close()

print()
print("============================================================")
print("FIGURE 3 FLATTENING COMPLETE")
print("============================================================")
print("No CMap/LINCS statistics were recomputed.")
print("Original Phase 6 PDFs remain untouched.")
print("Final panel sequence: A–J")
print("PDF:", FINAL_PDF)
print("PNG:", FINAL_PNG)
print("============================================================")
