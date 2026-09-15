from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path
from typing import Any


project = Path(sys.argv[1])
gates_path = Path(sys.argv[2])
endpoints_path = Path(sys.argv[3])
b6_status_path = Path(sys.argv[4])
b6_claims_path = Path(sys.argv[5])
expected_b6 = sys.argv[6]
out = Path(sys.argv[7])


def read_tsv(path):
    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )
        return list(reader)


def write_tsv(path, rows, columns):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


gates = read_tsv(
    gates_path
)

endpoints = read_tsv(
    endpoints_path
)

b6_status_rows = read_tsv(
    b6_status_path
)

claims = read_tsv(
    b6_claims_path
)

if len(b6_status_rows) != 1:
    raise SystemExit(
        "FAIL: Phase10B6 status must contain one row."
    )

b6 = b6_status_rows[0]

if b6.get(
    "phase10B6_status",
    "",
) != expected_b6:

    raise SystemExit(
        "FAIL: Phase10B6 is not in the required "
        "closed state."
    )

if b6.get(
    "phase10B_closed",
    "",
).upper() != "TRUE":

    raise SystemExit(
        "FAIL: Phase10B is not closed."
    )

if int(
    b6.get(
        "locked_modules_integrated",
        "-1",
    )
) != 9:

    raise SystemExit(
        "FAIL: Phase10B6 did not integrate 9 modules."
    )

if b6.get(
    "direct_cross_platform_replication_claimed",
    "",
).upper() != "FALSE":

    raise SystemExit(
        "FAIL: unexpected direct replication claim."
    )

if b6.get(
    "independent_biological_replication_claimed",
    "",
).upper() != "FALSE":

    raise SystemExit(
        "FAIL: unexpected independent replication claim."
    )

if b6.get(
    "population_level_inference_claimed",
    "",
).upper() != "FALSE":

    raise SystemExit(
        "FAIL: unexpected population-level inference."
    )

if b6.get(
    "cross_platform_spatial_direction_compared",
    "",
).upper() != "FALSE":

    raise SystemExit(
        "FAIL: cross-platform spatial direction "
        "was unexpectedly compared."
    )

if b6.get(
    "new_cross_platform_hypothesis_tests_performed",
    "",
).upper() != "FALSE":

    raise SystemExit(
        "FAIL: unexpected new cross-platform tests."
    )

gate_lookup = {
    row["gate_id"]: row
    for row in gates
}

for gate_id in (
    "G01",
    "G02",
    "G03",
    "G04",
):
    if gate_id not in gate_lookup:
        raise SystemExit(
            f"FAIL: missing prospective gate {gate_id}."
        )

gate_rows = [
    {
        "gate_id": "G01",
        "original_layer":
            gate_lookup["G01"]["layer"],
        "original_criterion":
            gate_lookup["G01"]["criterion"],
        "phase10B_final_status":
            "NOT_PASSED_AS_PRESPECIFIED",
        "gate_passed": False,
        "reason":
            "Final Phase10B explicitly did not perform "
            "cross-platform spatial-direction comparison "
            "or claim independent replication; therefore "
            "prespecified directional reproduction of >=6/9 "
            "modules cannot be asserted.",
        "biological_failure_implied": False,
        "protocol_deviation_or_estimand_change": True,
    },
    {
        "gate_id": "G02",
        "original_layer":
            gate_lookup["G02"]["layer"],
        "original_criterion":
            gate_lookup["G02"]["criterion"],
        "phase10B_final_status":
            "NOT_PASSED_AS_PRESPECIFIED",
        "gate_passed": False,
        "reason":
            "Progenitor-radial-glia and synaptic-assembly "
            "retained descriptive spatial support, but the "
            "prospective criterion referred to successful "
            "independent spatial replication, which was not "
            "the final authorized Phase10B estimand.",
        "biological_failure_implied": False,
        "protocol_deviation_or_estimand_change": True,
    },
    {
        "gate_id": "G03",
        "original_layer":
            gate_lookup["G03"]["layer"],
        "original_criterion":
            gate_lookup["G03"]["criterion"],
        "phase10B_final_status":
            "NOT_PASSED_AS_PRESPECIFIED",
        "gate_passed": False,
        "reason":
            "Phase10B did not produce the required "
            "donor-level spatial BH-adjusted P<0.05 plus "
            ">=80% leave-one-out directional retention "
            "criterion. Visium evidence was section-specific "
            "and MERFISH temporal tests yielded no corrected "
            "primary temporal associations.",
        "biological_failure_implied": False,
        "protocol_deviation_or_estimand_change": True,
    },
    {
        "gate_id": "G04",
        "original_layer":
            gate_lookup["G04"]["layer"],
        "original_criterion":
            gate_lookup["G04"]["criterion"],
        "phase10B_final_status":
            "NOT_PASSED_AS_PRESPECIFIED",
        "gate_passed": False,
        "reason":
            "The final Phase10B closure did not authorize "
            "a prespecified module-size- and expression-"
            "matched random-gene-set replication gate test "
            "across the independent validation branch.",
        "biological_failure_implied": False,
        "protocol_deviation_or_estimand_change": True,
    },
]

write_tsv(
    out
    / "01_gate_reconciliation"
    / "phase10B7_G01_G04_reconciliation.tsv",
    gate_rows,
    [
        "gate_id",
        "original_layer",
        "original_criterion",
        "phase10B_final_status",
        "gate_passed",
        "reason",
        "biological_failure_implied",
        "protocol_deviation_or_estimand_change",
    ],
)

endpoint_lookup = {
    row["endpoint_id"]: row
    for row in endpoints
}

endpoint_rows = [
    {
        "endpoint_id": "E01",
        "endpoint":
            endpoint_lookup["E01"]["endpoint"],
        "phase10B_resolution":
            "NOT_DIRECTLY_EVALUATED_AS_PRESPECIFIED",
        "interpretation":
            "Phase10B focused on locked DTHI module "
            "localization rather than a donor-level global "
            "maturation-minus-fetal programme endpoint.",
    },
    {
        "endpoint_id": "E02",
        "endpoint":
            endpoint_lookup["E02"]["endpoint"],
        "phase10B_resolution":
            "PARTIALLY_ADDRESSED_DESCRIPTIVELY",
        "interpretation":
            "Cell-class localization recovered progenitor, "
            "glial and neuronal contexts, but no formal "
            "donor-age-area-cell-state pseudobulk contrast "
            "matching E02 was used as the final inferential "
            "endpoint.",
    },
    {
        "endpoint_id": "E03",
        "endpoint":
            endpoint_lookup["E03"]["endpoint"],
        "phase10B_resolution":
            "PARTIALLY_ADDRESSED_SECTION_SPECIFICALLY",
        "interpretation":
            "Visium supplied section-specific spatial "
            "geometry/layer evidence, but the original "
            "donor-age-area-compartment pseudobulk endpoint "
            "was not achieved as prespecified.",
    },
    {
        "endpoint_id": "E04",
        "endpoint":
            endpoint_lookup["E04"]["endpoint"],
        "phase10B_resolution":
            "EVALUATED_WITH_MODIFIED_PANEL_HOMOGENEOUS_ESTIMAND",
        "interpretation":
            "MERFISH age association was evaluated with "
            "panel-homogeneous exact donor-section tests "
            "rather than the originally specified donor-aware "
            "mixed-effects model; no corrected primary "
            "temporal trajectory was detected.",
    },
]

write_tsv(
    out
    / "02_endpoint_reconciliation"
    / "phase10B7_E01_E04_reconciliation.tsv",
    endpoint_rows,
    [
        "endpoint_id",
        "endpoint",
        "phase10B_resolution",
        "interpretation",
    ],
)

deviation_text = """PHASE 10B7 PROSPECTIVE GATE RECONCILIATION

Phase 10B completed successfully as a conservative external
spatial-validation analysis, but it did not satisfy the original
Phase 10A independent-spatial-replication gates G01-G04 as written.

This is not recorded as a biological failure of the developmental
modules. It is an estimand and data-structure limitation.

The available external data required the final evidence hierarchy to
use:
  - section-specific Visium spatial evidence;
  - descriptive snRNA-seq cellular context;
  - panel-homogeneous MERFISH donor-section temporal analysis;
  - descriptive H1/H2 cellular localization; and
  - native-coordinate within-section MERFISH spatial heterogeneity.

Direct independent replication, population-level Visium inference,
cross-platform spatial-direction concordance and a new matched-random
cross-platform gate test were explicitly not claimed.

The original prospective gates remain in the audit trail and are not
rewritten after outcome inspection.

Candidate TFs, module identities and validation hypotheses remain
unchanged.

Phase 10C may proceed to the separately prespecified E05/G05
multiome regulatory-validation layer.
"""

(
    out
    / "03_protocol_deviation"
    / "phase10B7_protocol_deviation_declaration.txt"
).write_text(
    deviation_text,
    encoding="utf-8",
)

guardrail_rows = [
    {
        "guardrail":
            "Phase10B_technical_completion",
        "locked_value":
            "TRUE",
    },
    {
        "guardrail":
            "Phase10A_G01_G04_prospective_gate_success",
        "locked_value":
            "FALSE",
    },
    {
        "guardrail":
            "independent_spatial_replication_claim",
        "locked_value":
            "NOT_AUTHORIZED",
    },
    {
        "guardrail":
            "biological_failure_inferred_from_gate_nonpass",
        "locked_value":
            "FALSE",
    },
    {
        "guardrail":
            "candidate_reselection",
        "locked_value":
            "PROHIBITED",
    },
    {
        "guardrail":
            "module_reselection",
        "locked_value":
            "PROHIBITED",
    },
    {
        "guardrail":
            "next_prespecified_layer",
        "locked_value":
            "E05_G05_MULTIOME_REGULATORY_VALIDATION",
    },
]

write_tsv(
    out
    / "04_reporting_guardrails"
    / "phase10B7_reporting_guardrails.tsv",
    guardrail_rows,
    [
        "guardrail",
        "locked_value",
    ],
)

passed_gates = sum(
    bool(row["gate_passed"])
    for row in gate_rows
)

technical_pass = (
    len(gate_rows) == 4
    and passed_gates == 0
    and len(endpoint_rows) == 4
)

status_value = (
    "passed_phase10B7_prospective_spatial_gate_"
    "reconciliation_with_required_gate_nonpass_"
    "locked_ready_for_phase10C_multiome_validation"
    if technical_pass
    else
    "phase10B7_requires_manual_review"
)

status = {
    "phase": "phase10B7",
    "phase10B_technical_closure_verified": True,
    "prospective_spatial_gates_audited": 4,
    "prospective_spatial_gates_passed": (
        passed_gates
    ),
    "prospective_spatial_gates_not_passed": (
        4 - passed_gates
    ),
    "endpoints_E01_E04_reconciled": 4,
    "independent_spatial_replication_claim_authorized":
        False,
    "biological_failure_implied":
        False,
    "candidate_TFs_changed":
        False,
    "module_identities_changed":
        False,
    "validation_hypotheses_changed":
        False,
    "protocol_deviation_locked":
        True,
    "ready_for_phase10C_multiome_validation":
        True,
    "phase10B7_status":
        status_value,
}

write_tsv(
    out
    / "phase10B7_status.tsv",
    [status],
    list(status.keys()),
)

report = [
    "===== PHASE 10B7 PROSPECTIVE GATE RECONCILIATION =====",
    "",
    "Phase 10B technical closure verified: TRUE",
    "Prospective spatial gates audited: 4/4",
    f"Prospective spatial gates passed: {passed_gates}/4",
    f"Prospective spatial gates not passed: {4-passed_gates}/4",
    "Endpoints E01-E04 reconciled: 4/4",
    "",
    "Independent spatial replication claim authorized: FALSE",
    "Biological failure implied: FALSE",
    "Candidate TFs changed: FALSE",
    "Module identities changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "Protocol deviation locked: TRUE",
    "Ready for Phase 10C multiome validation: TRUE",
    "",
    f"PHASE 10B7 STATUS: {status_value}",
]

(
    out
    / "phase10B7_report.txt"
).write_text(
    "\n".join(report)
    + "\n",
    encoding="utf-8",
)

print(
    "\n".join(report)
)

checksum_rows = []

for path in sorted(
    out.rglob("*")
):
    if (
        path.is_file()
        and path.name
        != "phase10B7_SHA256.tsv"
    ):
        checksum_rows.append(
            {
                "sha256":
                    hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest(),
                "size_bytes":
                    path.stat().st_size,
                "project_relative_path":
                    path.relative_to(
                        project
                    ).as_posix(),
            }
        )

write_tsv(
    out
    / "phase10B7_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)

if not technical_pass:
    raise SystemExit(
        "Phase 10B7 requires manual review."
    )
