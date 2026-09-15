# DTHI_DATA

Reproducibility and derived-data repository for the Developmental
Transcriptomic Hierarchy (DTHI) study of human cortical development.

## Scope

The study integrates developmental transcriptomics, adult cortical
transcriptomic maps, regulatory inference, perturbational transcriptomics,
fetal spatial references, single-cell multiome data, virtual perturbation,
and external cortical phenotype maps.

This repository contains derived source-data tables, analysis code,
prospective-analysis lock/audit material, validation-closure tables,
and publication-supporting reproducibility files.

Raw third-party datasets are not redistributed here and should be obtained
from their original repositories and publications.

## Evidence levels

The study explicitly distinguishes:

1. discovery-level associations;
2. supportive cross-modal or computational evidence; and
3. strict prospective validation.

The final prospective validation closure retained the original G01-G07
criteria without candidate reselection or post-result threshold relaxation.

At project closure:

- prospective gates passed: 0/7;
- strict convergent regulators: 0;
- CellOracle findings for NKX2-2, RFX5 and RFXANK are supportive
  computational evidence rather than validated causal effects;
- external CRISPRi/Perturb-seq testing was limited by availability of the
  candidate-specific directional signatures required by the locked tests.

A gate classified as `NOT_PASSED_AS_PRESPECIFIED` should not automatically
be interpreted as a biological null.

## Repository structure

- `source_data/` — derived manuscript-supporting source tables
- `validation_lock/` — lock, gate-closure and candidate-evidence files
- `code/` — analysis and figure-generation scripts
- `figures/` — final publication figures (added after figure lock)
- `environments/` — software/environment documentation
- `documentation/` — reproducibility and dataset documentation

## Prospective-lock documentation

The validation plan was locked within the project workflow before the
corresponding validation analyses were interpreted. The lock and audit files
are deposited here to document that workflow.

This GitHub repository itself was created later and should not be interpreted
as an externally time-stamped preregistration.

## Raw datasets

Primary datasets remain available from their original sources, including
BrainSpan, the Allen Human Brain Atlas, GEO/LINCS, ENIGMA, Neurosynth,
fetal spatial transcriptomic resources, developing-brain multiome resources,
and Perturb-seq/CRISPRi resources.

Exact dataset descriptions and analytical roles are reported in the manuscript.

## Citation

A formal citation file and archived DOI will be added with the manuscript
release.
