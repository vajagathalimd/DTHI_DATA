# Prospective validation lock chronology

## Purpose

This directory documents the prospective validation framework used in the
Developmental Transcriptomic Hierarchy (DTHI) study.

The public GitHub repository was created after the analyses were performed.
Therefore, the GitHub commit date must **not** be interpreted as an external
preregistration timestamp.

Instead, this deposit preserves the project's original internal lock files,
audit material, hashes, frozen inputs, branch-specific method locks, and final
validation closure.

## Original Phase 10A master lock

The original master validation lock is stored in:

`validation_lock/original_phase10A_lock/`

The lock declaration records:

- internal lock time: **2026-07-29T22:32:53.121938+00:00**
- random seed: **20260730**
- frozen developmental programme assignments
- frozen DTHI module definitions
- frozen candidate transcription-factor shortlist
- frozen expected module directions
- frozen primary endpoints
- frozen negative-control selection rule
- frozen success gates
- donor or donor-level pseudobulk as the primary inferential unit

The declaration explicitly states that MERFISH, Visium, external snRNA-seq,
multiome, CRISPR and virtual-perturbation results were not inspected during
candidate selection.

It also states that failed prespecified hypotheses must remain in the final
report and that candidate addition, removal or reordering after validation-data
inspection was prohibited unless versioned and reported as a protocol deviation.

## Frozen-source chronology

The retained filesystem metadata of the original project shows that the source
material underlying the Phase 10A lock accumulated before the final lock:

- 2026-07-14: frozen DTHI module-definition source
- 2026-07-21: frozen supported-TF trajectory sources
- 2026-07-24: frozen developmental-program assignments
- 2026-07-29: final frozen selected-file manifest and status material
- 2026-07-30: candidate, module-direction, endpoint, control and gate tables

A later file labelled `R1` records a module-source-resolution correction.
It is retained alongside the original material rather than replacing or hiding
the earlier audit history.

Filesystem modification times are provided only as project provenance and are
not claimed to constitute independent preregistration evidence.

## Branch-specific method locks

Additional outcome-neutral method specifications were created as later
validation branches became operational. These are stored separately from the
original Phase 10A master lock.

### G05 multiome regulatory branch

`validation_lock/branch_method_locks/G05_multiome/`

The retained source file was created on 2026-08-04 and specifies the global G05
success rule.

### G06 virtual-perturbation branch

`validation_lock/branch_method_locks/G06_virtual_perturbation/`

The retained source file was created on 2026-08-07 and documents the causal /
virtual-perturbation analysis method lock.

### G07 external CRISPRi branch

`validation_lock/branch_method_locks/G07_external_CRISPRi/`

The retained source file was created on 2026-08-08 and documents the external
validation/testability method lock.

These later branch-specific locks are not represented as having been part of
the original 29–30 July Phase 10A master lock.

## Final closure

Final gate-level results are stored separately in:

`validation_lock/final_closure/`

The original gates were retained through project closure. Gates not satisfying
their original inferential requirements were classified as
`NOT_PASSED_AS_PRESPECIFIED`.

This classification should not automatically be interpreted as a biological
null because some gates were limited by unavailable estimands, directional
signatures, multiple-testing criteria, or cross-method requirements.

## Public deposit

The audit material was publicly deposited to GitHub on **15 September 2026**.

The public deposit preserves the project audit trail but is not claimed as a
prospective external registration.

## Historical filesystem paths

Some frozen manifests or audit files may contain historical local filesystem
paths such as `/mnt/d/...`.

These strings are retained intentionally where modifying them would alter the
original frozen audit material. They document historical project provenance
and are not required public execution paths.

Public-facing scripts use portable project-root configuration instead.
