from __future__ import annotations

import csv
import hashlib
import re
import shutil
import sys
from collections import Counter
from pathlib import Path


project = Path(sys.argv[1])
source_a = Path(sys.argv[2])
source_b = Path(sys.argv[3])
remote_inventory = Path(sys.argv[4])
out = Path(sys.argv[5])

lock_dir = out / "01_frozen_module_lock"
audit_dir = out / "02_module_audit"
remote_dir = out / "04_remote_download_plan"

for directory in (
    lock_dir,
    audit_dir,
    remote_dir,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

gene_candidates = (
    "gene_symbol",
    "gene",
    "symbol",
    "hgnc_symbol",
    "module_gene",
    "member_gene",
)

module_candidates = (
    "module_id",
    "module",
    "module_name",
    "dthi_module",
    "program",
    "seed_module",
    "gene_set",
    "geneset",
)


def read_tsv(
    path: Path,
) -> tuple[list[dict[str, str]], list[str]]:

    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        rows = list(reader)
        fields = reader.fieldnames or []

    return rows, fields


def normalized_map(
    fields: list[str],
) -> dict[str, str]:

    result: dict[str, str] = {}

    for field in fields:

        key = re.sub(
            r"[^a-z0-9]+",
            "_",
            field.lower(),
        ).strip("_")

        result[key] = field

    return result


def detect_column(
    fields: list[str],
    candidates: tuple[str, ...],
    label: str,
    path: Path,
) -> str:

    mapping = normalized_map(
        fields
    )

    for candidate in candidates:

        if candidate in mapping:
            return mapping[candidate]

    raise SystemExit(
        f"FAIL: could not identify {label} column "
        f"in {path}; columns={fields}"
    )


def canonical_pairs(
    path: Path,
) -> tuple[
    list[dict[str, str]],
    list[str],
    str,
    str,
    set[tuple[str, str]],
]:

    rows, fields = read_tsv(
        path
    )

    gene_col = detect_column(
        fields,
        gene_candidates,
        "gene",
        path,
    )

    module_col = detect_column(
        fields,
        module_candidates,
        "module",
        path,
    )

    pairs: set[tuple[str, str]] = set()

    for row in rows:

        module = str(
            row.get(
                module_col,
                "",
            )
        ).strip()

        gene = str(
            row.get(
                gene_col,
                "",
            )
        ).strip().upper()

        if (
            module
            and gene
            and gene not in {
                "NA",
                "NAN",
                "NONE",
            }
        ):
            pairs.add(
                (
                    module,
                    gene,
                )
            )

    if not pairs:
        raise SystemExit(
            f"FAIL: no valid module-gene pairs in {path}"
        )

    return (
        rows,
        fields,
        module_col,
        gene_col,
        pairs,
    )


(
    rows_a,
    fields_a,
    module_a,
    gene_a,
    pairs_a,
) = canonical_pairs(
    source_a
)

(
    rows_b,
    fields_b,
    module_b,
    gene_b,
    pairs_b,
) = canonical_pairs(
    source_b
)

selected_path = (
    lock_dir
    / "phase10B4_P1_DTHI_module_gene_lock.tsv"
)

shutil.copy2(
    source_b,
    selected_path,
)

manifest_path = (
    lock_dir
    / "phase10B4_P1_module_source_manifest.tsv"
)

with manifest_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:

    fields = [
        "role",
        "project_relative_path",
        "sha256",
        "rows",
        "valid_unique_pairs",
        "modules",
        "module_column",
        "gene_column",
    ]

    writer = csv.DictWriter(
        handle,
        fieldnames=fields,
        delimiter="\t",
        lineterminator="\n",
    )

    writer.writeheader()

    source_records = (
        (
            "comparator_phase10A_frozen",
            source_a,
            rows_a,
            pairs_a,
            module_a,
            gene_a,
        ),
        (
            "selected_phase10B_corrected",
            source_b,
            rows_b,
            pairs_b,
            module_b,
            gene_b,
        ),
    )

    for (
        role,
        path,
        rows,
        pairs,
        module_col,
        gene_col,
    ) in source_records:

        writer.writerow(
            {
                "role": role,
                "project_relative_path": (
                    path.relative_to(
                        project
                    ).as_posix()
                ),
                "sha256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
                "rows": len(rows),
                "valid_unique_pairs": len(pairs),
                "modules": len(
                    {
                        module
                        for module, _ in pairs
                    }
                ),
                "module_column": module_col,
                "gene_column": gene_col,
            }
        )

summary_counter = Counter(
    module
    for module, _ in pairs_b
)

summary_path = (
    audit_dir
    / "phase10B4_P1_selected_module_summary.tsv"
)

with summary_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "module",
            "locked_genes",
        ],
        delimiter="\t",
        lineterminator="\n",
    )

    writer.writeheader()

    for module in sorted(
        summary_counter
    ):
        writer.writerow(
            {
                "module": module,
                "locked_genes": (
                    summary_counter[module]
                ),
            }
        )

only_a = sorted(
    pairs_a - pairs_b
)

only_b = sorted(
    pairs_b - pairs_a
)

difference_path = (
    audit_dir
    / "phase10B4_P1_module_pair_differences.tsv"
)

with difference_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "difference",
            "module",
            "gene_symbol",
        ],
        delimiter="\t",
        lineterminator="\n",
    )

    writer.writeheader()

    for module, gene in only_a:

        writer.writerow(
            {
                "difference": "phase10A_only",
                "module": module,
                "gene_symbol": gene,
            }
        )

    for module, gene in only_b:

        writer.writerow(
            {
                "difference": (
                    "phase10B_corrected_only"
                ),
                "module": module,
                "gene_symbol": gene,
            }
        )

with remote_inventory.open(
    encoding="utf-8",
    newline="",
) as handle:

    remote_rows = list(
        csv.DictReader(
            handle,
            delimiter="\t",
        )
    )

triage_rows: list[dict[str, str]] = []

for row in remote_rows:

    name = row.get(
        "file_name",
        "",
    )

    lower = name.lower()

    record_id = row.get(
        "record_id",
        "",
    )

    age_tokens = sorted(
        set(
            re.findall(
                r"gw[_ -]?(\d{1,2})",
                name,
                flags=re.I,
            )
        )
    )

    is_macaque = bool(
        re.search(
            r"macaque|monkey",
            lower,
        )
    )

    is_image = bool(
        re.search(
            (
                r"\.(tif|tiff|png|jpg|jpeg|pdf)$"
                r"|image|dapi"
            ),
            lower,
        )
    )

    is_metadata = bool(
        re.search(
            (
                r"meta|annot|cluster|cell[_ -]?type"
                r"|sample|manifest|obs|label"
            ),
            lower,
        )
    )

    is_panel = bool(
        re.search(
            (
                r"gene[_ -]?panel|codebook"
                r"|genes?[_ -]?list|probe"
            ),
            lower,
        )
    )

    is_expression = bool(
        re.search(
            (
                r"count|matrix|expression"
                r"|\.h5ad$|\.rds$|\.h5$"
                r"|\.csv(\.gz)?$|\.tsv(\.gz)?$"
                r"|\.parquet$|\.feather$|\.zarr"
            ),
            lower,
        )
    )

    human_signal = (
        bool(
            re.search(
                r"human|fetal|cortex|gw[_ -]?\d",
                lower,
            )
        )
        or record_id == "15127709"
    )

    if is_macaque:

        decision = "exclude_nonhuman"
        priority = "X"

    elif (
        is_image
        and not (
            is_metadata
            or is_panel
        )
    ):

        decision = "exclude_image_only"
        priority = "X"

    elif (
        human_signal
        and (
            is_metadata
            or is_panel
        )
    ):

        decision = (
            "candidate_metadata_or_gene_panel"
        )
        priority = "1"

    elif (
        human_signal
        and is_expression
    ):

        decision = (
            "candidate_expression_or_processed_object"
        )
        priority = "2"

    elif record_id == "15127709":

        decision = (
            "manual_review_human_MERFISH_record"
        )
        priority = "3"

    else:

        decision = (
            "manual_review_other_record"
        )
        priority = "4"

    triage = dict(row)

    triage.update(
        {
            "priority": priority,
            "decision": decision,
            "age_tokens_from_filename": (
                ";".join(age_tokens)
            ),
            "nonhuman_signal": str(
                is_macaque
            ),
            "image_signal": str(
                is_image
            ),
            "metadata_signal": str(
                is_metadata
            ),
            "gene_panel_signal": str(
                is_panel
            ),
            "expression_signal": str(
                is_expression
            ),
        }
    )

    triage_rows.append(
        triage
    )

triage_fields = list(
    remote_rows[0].keys()
) + [
    "priority",
    "decision",
    "age_tokens_from_filename",
    "nonhuman_signal",
    "image_signal",
    "metadata_signal",
    "gene_panel_signal",
    "expression_signal",
]

triage_path = (
    remote_dir
    / "phase10B4_P1_Zenodo_file_triage.tsv"
)

with triage_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=triage_fields,
        delimiter="\t",
        lineterminator="\n",
    )

    writer.writeheader()

    writer.writerows(
        sorted(
            triage_rows,
            key=lambda row: (
                row["priority"],
                row.get(
                    "record_id",
                    "",
                ),
                row.get(
                    "file_name",
                    "",
                ),
            ),
        )
    )

candidate_rows = [
    row
    for row in triage_rows
    if row["priority"] in {
        "1",
        "2",
    }
]

candidate_path = (
    remote_dir
    / "phase10B4_P1_targeted_download_candidates.tsv"
)

with candidate_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:

    fields = [
        "priority",
        "decision",
        "record_id",
        "file_name",
        "size_bytes",
        "checksum",
        "download_url",
        "age_tokens_from_filename",
    ]

    writer = csv.DictWriter(
        handle,
        fieldnames=fields,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="ignore",
    )

    writer.writeheader()

    writer.writerows(
        sorted(
            candidate_rows,
            key=lambda row: (
                row["priority"],
                int(
                    row.get(
                        "size_bytes",
                    )
                    or 0
                ),
                row.get(
                    "file_name",
                    "",
                ),
            ),
        )
    )

print(
    "===== MODULE SOURCE RESOLUTION ====="
)

print(
    f"Comparator Phase 10A pairs: {len(pairs_a)}"
)

print(
    "Selected corrected Phase 10B pairs: "
    f"{len(pairs_b)}"
)

print(
    f"Selected modules: {len(summary_counter)}"
)

print(
    f"Phase 10A-only pairs: {len(only_a)}"
)

print(
    "Phase 10B-corrected-only pairs: "
    f"{len(only_b)}"
)

print(
    "Selected source: "
    "phase10B1_R1_corrected_module_gene_lock.tsv"
)

print(
    "\n===== REMOTE FILE TRIAGE ====="
)

for decision, count in sorted(
    Counter(
        row["decision"]
        for row in triage_rows
    ).items()
):

    print(
        f"{decision}: {count}"
    )

print(
    f"Targeted candidate files: {len(candidate_rows)}"
)
