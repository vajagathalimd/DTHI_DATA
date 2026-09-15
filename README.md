# DTHI_DATA

Reproducibility, derived-data and manuscript-support repository for the
**Developmental Transcriptomic Hierarchy (DTHI)** study of human cortical
development.

## Study scope

The study integrates developmental transcriptomics, adult cortical
transcriptomic maps, regulatory inference, perturbational transcriptomics,
fetal spatial transcriptomics, single-cell and multiome evidence,
virtual perturbation, and external cortical phenotype maps.

The repository contains manuscript-supporting derived data, final figures,
figure source data, analysis code, prospective-analysis lock/audit material,
validation-closure outputs, software information and reproducibility
documentation.

Raw third-party datasets are **not redistributed**. They should be obtained
from the original repositories and publications described in the manuscript.

## Evidence framework

The study explicitly distinguishes:

1. discovery-level associations;
2. supportive cross-modal or computational evidence; and
3. strict prospective validation.

The final prospective validation closure retained the original G01-G07
criteria without candidate reselection, post-result threshold relaxation or
post-hoc rescue.

At project closure:

- prospective gates passed: **0/7**;
- strict convergent regulators: **0**;
- CellOracle findings for **NKX2-2, RFX5 and RFXANK** are supportive
  computational evidence rather than validated causal effects;
- external CRISPRi/Perturb-seq testing was limited by availability of the
  candidate-specific directional signatures required by the locked tests.

A gate classified as `NOT_PASSED_AS_PRESPECIFIED` should not automatically
be interpreted as a biological null.

## Repository structure

- `supplementary_tables/`
  - manuscript Supplementary Tables S1-S15

- `figures/main/`
  - final manuscript Figures 1-6 in PNG/PDF format
  - SHA256 figure manifest

- `figures/source_data/`
  - numerical source data supporting Figures 1-6
  - figure-to-source manifest and checksums

- `code/manuscript/`
  - manuscript-relevant analysis and figure-generation code
  - developmental trajectories
  - perturbational evidence
  - developmental regulatory atlas
  - cortical phenotype correspondence
  - prospective spatial validation
  - manuscript figure rendering

- `validation_lock/`
  - original prospective validation lock
  - frozen candidate/end-point/gate definitions
  - validation chronology and audit material

- `source_data/`
  - additional derived analysis and validation outputs

- `environments/`
  - software/environment documentation

- `documentation/`
  - reproducibility and provenance documentation

## Prospective-lock documentation

Candidate transcription factors, module directions, primary endpoints,
negative-control rules and validation gates were frozen within the project
workflow before the corresponding independent validation analyses were
interpreted.

The original lock declaration, frozen inputs, audit files and final closure
are deposited in `validation_lock/`.

The GitHub repository itself was created **after** these analyses and should
therefore not be interpreted as an externally time-stamped preregistration.

## Main-figure source data

Source data corresponding to manuscript Figures 1-6 are deposited under:

`figures/source_data/`

The file

`figures/source_data/FIGURE_SOURCE_DATA_MANIFEST.tsv`

records the original project-relative source, public repository location,
file size and SHA256 checksum for each deposited source-data file.

No statistical tests, effect sizes, P values or module scores were recomputed
during preparation of the public figure-source package.

## Analysis code

Manuscript-relevant scripts are deposited under:

`code/manuscript/`

The file

`code/manuscript/CODE_MANIFEST.tsv`

records the original project-relative source, original SHA256 checksum,
public-copy checksum and whether a local project-root path was normalized
for portability.

Historical backup, failed, superseded and pre-fix scripts are intentionally
excluded from the manuscript-facing code package.

## Raw datasets

Primary datasets remain available from their original sources, including
BrainSpan, the Allen Human Brain Atlas, GEO, LINCS/CMAP, ENIGMA,
Neurosynth and the fetal/developing-brain spatial, single-cell and multiome
resources described in the manuscript.

Users should obtain third-party datasets directly from those original
repositories and comply with their respective licenses and terms.

## Reproducibility

Additional documentation is available in:

`documentation/REPRODUCIBILITY_PATHS.md`

Software and environment information is also provided in:

`supplementary_tables/Supplementary_Table_S15_software_and_environment_manifest.tsv`

## Citation

If using this repository before publication of the associated manuscript,
please cite the repository metadata in `CITATION.cff`.

A versioned archival DOI will be added after creation of the manuscript
release.

## Licensing

Original analysis code in this repository is released under the **MIT
License**; see `LICENSE`.

Repository-generated derived tables, figures and documentation are made
available under **Creative Commons Attribution 4.0 International
(CC BY 4.0)**; see `DATA_LICENSE.md`.

These licenses do not override licenses, terms or attribution requirements
of third-party datasets, software or resources.
