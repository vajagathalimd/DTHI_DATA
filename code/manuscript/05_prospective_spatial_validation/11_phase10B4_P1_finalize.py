from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path


project = Path(sys.argv[1])
out = Path(sys.argv[2])

coverage_path = (
    out
    / "03_feature_coverage"
    / "phase10B4_P1_module_feature_coverage.tsv"
)

candidates_path = (
    out
    / "04_remote_download_plan"
    / "phase10B4_P1_targeted_download_candidates.tsv"
)

summary_path = (
    out
    / "02_module_audit"
    / "phase10B4_P1_selected_module_summary.tsv"
)

with coverage_path.open(
    encoding="utf-8",
    newline="",
) as handle:

    coverage = list(
        csv.DictReader(
            handle,
            delimiter="\t",
        )
    )

with candidates_path.open(
    encoding="utf-8",
    newline="",
) as handle:

    candidates = list(
        csv.DictReader(
            handle,
            delimiter="\t",
        )
    )

with summary_path.open(
    encoding="utf-8",
    newline="",
) as handle:

    modules = list(
        csv.DictReader(
            handle,
            delimiter="\t",
        )
    )

zero_sn = [
    row["module"]
    for row in coverage
    if row[
        "zero_coverage_snRNAseq"
    ].upper() == "TRUE"
]

zero_vis = [
    row["module"]
    for row in coverage
    if row[
        "zero_coverage_Visium"
    ].upper() == "TRUE"
]

if (
    modules
    and not zero_sn
    and not zero_vis
    and candidates
):

    status_value = (
        "passed_phase10B4_P1_module_lock_and_"
        "feature_coverage_ready_for_GW20_scoring_"
        "and_targeted_MERFISH_download_review"
    )

else:

    status_value = (
        "phase10B4_P1_requires_manual_review"
    )

status = {
    "phase": "phase10B4_P1",
    "selected_module_source": (
        "phase10B1_R1_corrected_module_gene_lock.tsv"
    ),
    "locked_modules": len(modules),
    "modules_with_zero_snRNAseq_coverage": (
        ";".join(zero_sn)
    ),
    "modules_with_zero_Visium_coverage": (
        ";".join(zero_vis)
    ),
    "targeted_remote_candidate_files": len(
        candidates
    ),
    "expression_values_accessed": False,
    "module_scores_computed": False,
    "large_remote_files_downloaded": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B4_P1_status": status_value,
}

with (
    out
    / "phase10B4_P1_status.tsv"
).open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=list(
            status.keys()
        ),
        delimiter="\t",
        lineterminator="\n",
    )

    writer.writeheader()
    writer.writerow(status)

report = [
    "===== PHASE 10B4-P1 MODULE LOCK "
    "AND DOWNLOAD PLAN =====",
    "",
    (
        "Selected module source: "
        f"{status['selected_module_source']}"
    ),
    (
        "Locked modules: "
        f"{status['locked_modules']}"
    ),
    (
        "Modules with zero snRNA-seq "
        "feature coverage: "
        f"{status['modules_with_zero_snRNAseq_coverage'] or 'NONE'}"
    ),
    (
        "Modules with zero Visium "
        "feature coverage: "
        f"{status['modules_with_zero_Visium_coverage'] or 'NONE'}"
    ),
    (
        "Targeted remote candidate files: "
        f"{status['targeted_remote_candidate_files']}"
    ),
    "",
    "Expression values accessed: FALSE",
    "Module scores computed: FALSE",
    "Large remote files downloaded: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B4-P1 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B4_P1_report.txt"
).write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

checksum_rows = []

for path in sorted(
    out.rglob("*")
):

    if (
        path.is_file()
        and path.name
        != "phase10B4_P1_SHA256.tsv"
    ):

        checksum_rows.append(
            {
                "sha256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
                "size_bytes": (
                    path.stat().st_size
                ),
                "project_relative_path": (
                    path.relative_to(
                        project
                    ).as_posix()
                ),
            }
        )

with (
    out
    / "phase10B4_P1_SHA256.tsv"
).open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "sha256",
            "size_bytes",
            "project_relative_path",
        ],
        delimiter="\t",
        lineterminator="\n",
    )

    writer.writeheader()
    writer.writerows(
        checksum_rows
    )

print(
    "\n".join(report)
)
