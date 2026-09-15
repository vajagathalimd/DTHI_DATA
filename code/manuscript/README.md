# Manuscript analysis code

This directory contains the manuscript-relevant analysis and figure-generation
scripts for the DTHI study.

## Organization

- `01_developmental_trajectories/`
  BrainSpan gene trajectories, developmental programme assignment,
  functional enrichment and downstream programme integration.

- `02_perturbational_evidence/`
  EPA HTTr/CMAP and LINCS processing, empirical calibration,
  robustness analysis and final perturbational evidence synthesis.

- `03_regulatory_atlas/`
  Final integration and closure of the developmental regulatory atlas.

- `04_cortical_correspondence/`
  Adult cortical phenotype-map harmonization, spatial correspondence,
  spin-based inference, donor sensitivity and definitive evidence hierarchy.

- `05_prospective_spatial_validation/`
  Prospective Phase 10B4-B7 spatial-validation workflow, including GW20
  Visium analysis, MERFISH analysis, cross-platform synthesis and gate
  reconciliation.

- `06_figure_generation/`
  Final manuscript-specific Figure 2 and Figure 3 rendering/composition code.

## Portability

The deposited copies preserve the scientific analysis logic of the original
scripts. Where an original script contained the author's absolute project root

`/mnt/d/2026_Work/DTHI_Struct_Cortical_Hierarchy`

that prefix was replaced with `.` in the public copy so that paths are relative
to the project working directory.

`CODE_MANIFEST.tsv` records the SHA256 checksum of both the original local
script and the deposited public copy, and records whether this project-root
normalization was applied.

No statistical analyses were rerun and no scientific parameters, thresholds,
models, tests or result values were changed while preparing this code package.

Historical backup, failed, pre-fix and superseded scripts are intentionally
excluded from this manuscript-facing package.
