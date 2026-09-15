from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


project = Path(sys.argv[1])
payload_dir = Path(sys.argv[2])
source_decision_path = Path(sys.argv[3])
design_path = Path(sys.argv[4])
gene_inventory_path = Path(sys.argv[5])
claim_boundary_path = Path(sys.argv[6])
out = Path(sys.argv[7])

source_dir = out / "01_expression_source_lock"
module_dir = out / "02_harmonized_module_gene_sets"
plan_dir = out / "03_section_scoring_plan"
position_dir = out / "04_gene_position_lock"
resource_dir = out / "05_resource_plan"
guardrail_dir = out / "06_analysis_guardrails"
access_dir = out / "07_expression_access_guard"
audit_dir = out / "08_audit"

for directory in (
    source_dir,
    module_dir,
    plan_dir,
    position_dir,
    resource_dir,
    guardrail_dir,
    access_dir,
    audit_dir,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

expected_files = (
    "gw15.h5ad",
    "gw18_umb1759.h5ad",
    "gw20.h5ad",
    "gw20_umb1031.h5ad",
    "gw22.h5ad",
    "gw34.h5ad",
)

expected_core_counts = {
    "activity_dependent_plasticity": 4,
    "astrocyte_maturation_metabolic_support": 7,
    "neurogenesis_migration_layering": 13,
    "patterning_arealization": 6,
    "progenitor_radial_glia": 9,
}

expected_expanded_counts = {
    "axon_guidance_neurite_outgrowth": 11,
    "oligodendrocyte_myelination": 5,
    "synaptic_assembly_receptor_trafficking": 7,
    "synaptic_membrane_structural_candidates": 10,
}


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

        return list(reader), reader.fieldnames or []


def write_tsv(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:

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


def as_bool(
    value: Any,
) -> bool:

    return str(value).strip().upper() in {
        "TRUE",
        "T",
        "1",
        "YES",
    }


def decode(
    value: Any,
) -> str:

    if isinstance(value, bytes):

        return value.decode(
            "utf-8",
            errors="replace",
        )

    if isinstance(value, np.bytes_):

        return bytes(value).decode(
            "utf-8",
            errors="replace",
        )

    if isinstance(value, np.generic):

        return str(value.item())

    return str(value)


def attr_text(
    value: Any,
) -> str:

    array = np.asarray(value)

    if array.ndim == 0:

        return decode(
            array.item()
        )

    return ";".join(
        decode(item)
        for item in array.reshape(-1)
    )


def matrix_shape(
    obj: h5py.Group | h5py.Dataset,
) -> tuple[int, int]:

    if isinstance(obj, h5py.Dataset):

        if obj.ndim != 2:

            raise ValueError(
                f"Non-two-dimensional matrix: "
                f"{obj.name}"
            )

        return (
            int(obj.shape[0]),
            int(obj.shape[1]),
        )

    shape = np.asarray(
        obj.attrs.get(
            "shape",
            [],
        )
    ).reshape(-1)

    if shape.size != 2:

        raise ValueError(
            f"Matrix shape is unavailable: "
            f"{obj.name}"
        )

    return int(shape[0]), int(shape[1])


def matrix_encoding(
    obj: h5py.Group | h5py.Dataset,
) -> str:

    return attr_text(
        obj.attrs.get(
            "encoding-type",
            (
                "dense_array"
                if isinstance(
                    obj,
                    h5py.Dataset,
                )
                else ""
            ),
        )
    )


def read_text_object(
    obj: h5py.Group | h5py.Dataset,
) -> list[str]:

    if isinstance(obj, h5py.Dataset):

        return [
            decode(value)
            for value in np.asarray(
                obj[:]
            ).reshape(-1)
        ]

    if "categories" in obj:

        return read_text_object(
            obj["categories"]
        )

    if "values" in obj:

        return read_text_object(
            obj["values"]
        )

    raise ValueError(
        f"Unsupported string object: {obj.name}"
    )


source_rows, source_columns = read_tsv(
    source_decision_path
)

if len(source_rows) != 1:

    raise SystemExit(
        "FAIL: source-decision table must contain "
        "one row."
    )

source_decision = source_rows[0]

if source_decision.get(
    "recommended_processed_source",
    "",
) != "/X":

    raise SystemExit(
        "FAIL: P4D did not recommend /X as the "
        "processed source."
    )

if source_decision.get(
    "recommended_count_source",
    "",
) != "/raw/X":

    raise SystemExit(
        "FAIL: P4D did not recommend /raw/X as "
        "the count source."
    )

if as_bool(
    source_decision.get(
        "module_scores_computed",
        "",
    )
):

    raise SystemExit(
        "FAIL: P4D unexpectedly computed module "
        "scores."
    )

design_rows, design_columns = read_tsv(
    design_path
)

if len(design_rows) != 12:

    raise SystemExit(
        f"FAIL: expected 12 revised sections; "
        f"observed {len(design_rows)}."
    )

if sum(
    row[
        "effective_analysis_set"
    ] == "primary"
    for row in design_rows
) != 8:

    raise SystemExit(
        "FAIL: revised design does not contain "
        "eight primary sections."
    )

if sum(
    row[
        "effective_analysis_set"
    ] == "section_sensitivity"
    for row in design_rows
) != 4:

    raise SystemExit(
        "FAIL: revised design does not contain "
        "four section-sensitivity sections."
    )

if not all(
    as_bool(
        row[
            "processed_mapping_confirmed"
        ]
    )
    for row in design_rows
):

    raise SystemExit(
        "FAIL: not all revised processed mappings "
        "are confirmed."
    )

selected_archives = {
    row[
        "archive_name"
    ]
    for row in design_rows
}

gene_rows, gene_columns = read_tsv(
    gene_inventory_path
)

genes_by_archive_module: dict[
    tuple[str, str],
    set[str],
] = defaultdict(set)

for row in gene_rows:

    archive_name = row[
        "archive_name"
    ]

    if archive_name not in selected_archives:

        continue

    if not as_bool(
        row[
            "sensitivity_scoring_eligible"
        ]
    ):

        continue

    gene = row[
        "gene_symbol"
    ].strip().upper()

    if gene:

        genes_by_archive_module[
            (
                archive_name,
                row[
                    "module"
                ],
            )
        ].add(gene)

boundary_rows, boundary_columns = read_tsv(
    claim_boundary_path
)

if len(boundary_rows) != 9:

    raise SystemExit(
        f"FAIL: expected nine module-boundary rows; "
        f"observed {len(boundary_rows)}."
    )

boundary_by_module = {
    row[
        "module"
    ]: row
    for row in boundary_rows
}

core_modules = sorted(
    module
    for module, row
    in boundary_by_module.items()
    if row[
        "prespecified_claim_class"
    ] in {
        "broad_primary_multiage_multidonor",
        "broad_sensitivity_multiage_multidonor",
    }
)

expanded_modules = sorted(
    module
    for module, row
    in boundary_by_module.items()
    if row[
        "prespecified_claim_class"
    ]
    == "expanded_panel_only_two_age_validation"
)

if set(core_modules) != set(
    expected_core_counts
):

    raise SystemExit(
        "FAIL: broad cross-panel module set differs "
        "from the locked design."
    )

if set(expanded_modules) != set(
    expected_expanded_counts
):

    raise SystemExit(
        "FAIL: expanded-panel module set differs "
        "from the locked design."
    )

expanded_design_rows = [
    row
    for row in design_rows
    if (
        row[
            "effective_analysis_set"
        ] == "primary"
        and row[
            "panel_class"
        ] == "960_gene_panel"
    )
]

if len(expanded_design_rows) != 2:

    raise SystemExit(
        "FAIL: expected two expanded-panel primary "
        "sections."
    )

core_gene_sets: dict[
    str,
    set[str],
] = {}

expanded_gene_sets: dict[
    str,
    set[str],
] = {}

for module in core_modules:

    section_sets = [
        genes_by_archive_module[
            (
                row[
                    "archive_name"
                ],
                module,
            )
        ]
        for row in design_rows
    ]

    if any(
        not genes
        for genes in section_sets
    ):

        raise SystemExit(
            f"FAIL: core module {module} is absent "
            "from at least one revised section."
        )

    intersection = set.intersection(
        *section_sets
    )

    expected_count = expected_core_counts[
        module
    ]

    if len(intersection) != expected_count:

        raise SystemExit(
            f"FAIL: {module} cross-panel core contains "
            f"{len(intersection)} genes; expected "
            f"{expected_count}."
        )

    core_gene_sets[
        module
    ] = intersection

for module in expanded_modules:

    section_sets = [
        genes_by_archive_module[
            (
                row[
                    "archive_name"
                ],
                module,
            )
        ]
        for row in expanded_design_rows
    ]

    if any(
        not genes
        for genes in section_sets
    ):

        raise SystemExit(
            f"FAIL: expanded module {module} is absent "
            "from one of the two expanded objects."
        )

    intersection = set.intersection(
        *section_sets
    )

    expected_count = expected_expanded_counts[
        module
    ]

    if len(intersection) != expected_count:

        raise SystemExit(
            f"FAIL: {module} expanded-panel set "
            f"contains {len(intersection)} genes; "
            f"expected {expected_count}."
        )

    expanded_gene_sets[
        module
    ] = intersection

module_set_rows: list[
    dict[str, Any]
] = []

for module in core_modules:

    claim_class = boundary_by_module[
        module
    ][
        "prespecified_claim_class"
    ]

    module_set_rows.append(
        {
            "scoring_family": (
                "cross_panel_harmonized_core"
            ),
            "module": module,
            "prespecified_claim_class": (
                claim_class
            ),
            "eligible_sections": 12,
            "eligible_primary_sections": 8,
            "eligible_section_sensitivity_sections": 4,
            "gene_count": len(
                core_gene_sets[
                    module
                ]
            ),
            "gene_symbols": ";".join(
                sorted(
                    core_gene_sets[
                        module
                    ]
                )
            ),
            "gene_set_derivation": (
                "intersection_across_all_12_revised_"
                "processed_sections"
            ),
            "primary_use": (
                "multiage_multidonor_validation"
                if claim_class
                == "broad_primary_multiage_multidonor"
                else "prespecified_sensitivity_validation"
            ),
        }
    )

for module in expanded_modules:

    module_set_rows.append(
        {
            "scoring_family": (
                "expanded_panel_only"
            ),
            "module": module,
            "prespecified_claim_class": (
                boundary_by_module[
                    module
                ][
                    "prespecified_claim_class"
                ]
            ),
            "eligible_sections": 2,
            "eligible_primary_sections": 2,
            "eligible_section_sensitivity_sections": 0,
            "gene_count": len(
                expanded_gene_sets[
                    module
                ]
            ),
            "gene_symbols": ";".join(
                sorted(
                    expanded_gene_sets[
                        module
                    ]
                )
            ),
            "gene_set_derivation": (
                "intersection_across_the_two_960_gene_"
                "primary_processed_objects"
            ),
            "primary_use": (
                "expanded_panel_two_age_validation"
            ),
        }
    )

write_tsv(
    module_dir
    / "phase10B5_P4D_R1_harmonized_module_gene_sets.tsv",
    module_set_rows,
    [
        "scoring_family",
        "module",
        "prespecified_claim_class",
        "eligible_sections",
        "eligible_primary_sections",
        "eligible_section_sensitivity_sections",
        "gene_count",
        "gene_symbols",
        "gene_set_derivation",
        "primary_use",
    ],
)

section_plan_rows: list[
    dict[str, Any]
] = []

module_gene_rows: list[
    dict[str, Any]
] = []

for row in design_rows:

    archive_name = row[
        "archive_name"
    ]

    for module in core_modules:

        genes = sorted(
            core_gene_sets[
                module
            ]
        )

        section_plan_rows.append(
            {
                "effective_analysis_set": row[
                    "effective_analysis_set"
                ],
                "archive_name": archive_name,
                "donor_id": row[
                    "donor_id"
                ],
                "gestational_week": row[
                    "gestational_week"
                ],
                "panel_class": row[
                    "panel_class"
                ],
                "processed_H5AD": row[
                    "processed_H5AD"
                ],
                "locked_sample_value": row[
                    "locked_sample_value"
                ],
                "locked_region_value": row[
                    "locked_region_value"
                ],
                "selected_cell_count": row[
                    "selected_cell_count"
                ],
                "scoring_family": (
                    "cross_panel_harmonized_core"
                ),
                "module": module,
                "prespecified_claim_class": (
                    boundary_by_module[
                        module
                    ][
                        "prespecified_claim_class"
                    ]
                ),
                "gene_count": len(genes),
                "gene_symbols": ";".join(
                    genes
                ),
                "expression_source": "/X",
                "standardization_scope": (
                    "within_section_per_gene"
                ),
                "standardization_center": (
                    "section_gene_mean"
                ),
                "standardization_scale": (
                    "section_gene_population_SD_ddof0"
                ),
                "zero_variance_gene_rule": (
                    "set_standardized_gene_value_to_zero"
                ),
                "module_score_definition": (
                    "unweighted_mean_of_standardized_"
                    "eligible_genes"
                ),
            }
        )

        for gene in genes:

            module_gene_rows.append(
                {
                    "archive_name": archive_name,
                    "processed_H5AD": row[
                        "processed_H5AD"
                    ],
                    "scoring_family": (
                        "cross_panel_harmonized_core"
                    ),
                    "module": module,
                    "gene_symbol": gene,
                }
            )

    if row[
        "panel_class"
    ] == "960_gene_panel":

        for module in expanded_modules:

            genes = sorted(
                expanded_gene_sets[
                    module
                ]
            )

            section_plan_rows.append(
                {
                    "effective_analysis_set": row[
                        "effective_analysis_set"
                    ],
                    "archive_name": archive_name,
                    "donor_id": row[
                        "donor_id"
                    ],
                    "gestational_week": row[
                        "gestational_week"
                    ],
                    "panel_class": row[
                        "panel_class"
                    ],
                    "processed_H5AD": row[
                        "processed_H5AD"
                    ],
                    "locked_sample_value": row[
                        "locked_sample_value"
                    ],
                    "locked_region_value": row[
                        "locked_region_value"
                    ],
                    "selected_cell_count": row[
                        "selected_cell_count"
                    ],
                    "scoring_family": (
                        "expanded_panel_only"
                    ),
                    "module": module,
                    "prespecified_claim_class": (
                        boundary_by_module[
                            module
                        ][
                            "prespecified_claim_class"
                        ]
                    ),
                    "gene_count": len(genes),
                    "gene_symbols": ";".join(
                        genes
                    ),
                    "expression_source": "/X",
                    "standardization_scope": (
                        "within_section_per_gene"
                    ),
                    "standardization_center": (
                        "section_gene_mean"
                    ),
                    "standardization_scale": (
                        "section_gene_population_SD_ddof0"
                    ),
                    "zero_variance_gene_rule": (
                        "set_standardized_gene_value_to_zero"
                    ),
                    "module_score_definition": (
                        "unweighted_mean_of_standardized_"
                        "eligible_genes"
                    ),
                }
            )

            for gene in genes:

                module_gene_rows.append(
                    {
                        "archive_name": archive_name,
                        "processed_H5AD": row[
                            "processed_H5AD"
                        ],
                        "scoring_family": (
                            "expanded_panel_only"
                        ),
                        "module": module,
                        "gene_symbol": gene,
                    }
                )

if len(section_plan_rows) != 68:

    raise SystemExit(
        f"FAIL: expected 68 section-module scoring "
        f"definitions; observed "
        f"{len(section_plan_rows)}."
    )

if len(module_gene_rows) != 534:

    raise SystemExit(
        f"FAIL: expected 534 section-module-gene "
        f"rows; observed {len(module_gene_rows)}."
    )

write_tsv(
    plan_dir
    / "phase10B5_P4D_R1_section_module_scoring_plan.tsv",
    section_plan_rows,
    [
        "effective_analysis_set",
        "archive_name",
        "donor_id",
        "gestational_week",
        "panel_class",
        "processed_H5AD",
        "locked_sample_value",
        "locked_region_value",
        "selected_cell_count",
        "scoring_family",
        "module",
        "prespecified_claim_class",
        "gene_count",
        "gene_symbols",
        "expression_source",
        "standardization_scope",
        "standardization_center",
        "standardization_scale",
        "zero_variance_gene_rule",
        "module_score_definition",
    ],
)

write_tsv(
    plan_dir
    / "phase10B5_P4D_R1_section_module_gene_plan.tsv",
    module_gene_rows,
    [
        "archive_name",
        "processed_H5AD",
        "scoring_family",
        "module",
        "gene_symbol",
    ],
)

plans_by_file: dict[
    str,
    list[dict[str, Any]],
] = defaultdict(list)

for row in section_plan_rows:

    plans_by_file[
        row[
            "processed_H5AD"
        ]
    ].append(row)

source_lock_rows: list[
    dict[str, Any]
] = []

position_rows: list[
    dict[str, Any]
] = []

for file_name in expected_files:

    path = payload_dir / file_name

    if not path.is_file():

        raise SystemExit(
            f"FAIL: missing H5AD file: {path}"
        )

    with h5py.File(
        path,
        "r",
    ) as handle:

        if "X" not in handle:

            raise SystemExit(
                f"FAIL: /X missing from {file_name}."
            )

        if (
            "raw" not in handle
            or "X" not in handle[
                "raw"
            ]
        ):

            raise SystemExit(
                f"FAIL: /raw/X missing from "
                f"{file_name}."
            )

        x_object = handle["X"]
        raw_x_object = handle["raw"]["X"]

        x_rows, x_columns = matrix_shape(
            x_object
        )

        raw_rows, raw_columns = matrix_shape(
            raw_x_object
        )

        if (
            x_rows != raw_rows
            or x_columns != raw_columns
        ):

            raise SystemExit(
                f"FAIL: /X and /raw/X shapes differ "
                f"in {file_name}."
            )

        if "obs" not in handle:

            raise SystemExit(
                f"FAIL: /obs missing from {file_name}."
            )

        obs = handle["obs"]

        obs_index_key = attr_text(
            obs.attrs.get(
                "_index",
                "_index",
            )
        )

        if obs_index_key not in obs:

            raise SystemExit(
                f"FAIL: obs index missing in "
                f"{file_name}."
            )

        obs_rows = int(
            obs[
                obs_index_key
            ].shape[0]
        )

        if x_rows != obs_rows:

            raise SystemExit(
                f"FAIL: /X row count differs from "
                f"obs in {file_name}."
            )

        if "var" not in handle:

            raise SystemExit(
                f"FAIL: /var missing from {file_name}."
            )

        var = handle["var"]

        var_index_key = attr_text(
            var.attrs.get(
                "_index",
                "_index",
            )
        )

        if var_index_key not in var:

            raise SystemExit(
                f"FAIL: var index missing in "
                f"{file_name}."
            )

        genes = [
            gene.strip().upper()
            for gene in read_text_object(
                var[
                    var_index_key
                ]
            )
        ]

        if len(genes) != x_columns:

            raise SystemExit(
                f"FAIL: var gene count differs from "
                f"/X columns in {file_name}."
            )

        gene_position = {
            gene: index
            for index, gene
            in enumerate(genes)
        }

        required_genes = sorted(
            {
                gene
                for plan in plans_by_file[
                    file_name
                ]
                for gene in plan[
                    "gene_symbols"
                ].split(";")
                if gene
            }
        )

        missing_genes = [
            gene
            for gene in required_genes
            if gene not in gene_position
        ]

        if missing_genes:

            raise SystemExit(
                f"FAIL: required genes missing from "
                f"{file_name}: "
                + ";".join(
                    missing_genes
                )
            )

        for gene in required_genes:

            position_rows.append(
                {
                    "processed_H5AD": file_name,
                    "gene_symbol": gene,
                    "zero_based_column_index": (
                        gene_position[
                            gene
                        ]
                    ),
                    "expression_source": "/X",
                    "gene_identifier_source": (
                        f"/var/{var_index_key}"
                    ),
                }
            )

        source_lock_rows.append(
            {
                "processed_H5AD": file_name,
                "primary_expression_source": "/X",
                "primary_source_interpretation": (
                    "author_processed_continuous_"
                    "nonnegative_expression"
                ),
                "exact_author_normalization_formula_proven": (
                    False
                ),
                "count_reference_source": "/raw/X",
                "count_reference_interpretation": (
                    "integer_count_like_expression"
                ),
                "X_encoding": matrix_encoding(
                    x_object
                ),
                "raw_X_encoding": matrix_encoding(
                    raw_x_object
                ),
                "matrix_rows": x_rows,
                "matrix_columns": x_columns,
                "obs_rows": obs_rows,
                "var_genes": len(genes),
                "required_scoring_genes": len(
                    required_genes
                ),
                "all_required_genes_present": True,
                "expression_values_accessed": False,
                "module_scores_computed": False,
            }
        )

write_tsv(
    source_dir
    / "phase10B5_P4D_R1_expression_source_lock.tsv",
    source_lock_rows,
    [
        "processed_H5AD",
        "primary_expression_source",
        "primary_source_interpretation",
        "exact_author_normalization_formula_proven",
        "count_reference_source",
        "count_reference_interpretation",
        "X_encoding",
        "raw_X_encoding",
        "matrix_rows",
        "matrix_columns",
        "obs_rows",
        "var_genes",
        "required_scoring_genes",
        "all_required_genes_present",
        "expression_values_accessed",
        "module_scores_computed",
    ],
)

write_tsv(
    position_dir
    / "phase10B5_P4D_R1_gene_position_lock.tsv",
    position_rows,
    [
        "processed_H5AD",
        "gene_symbol",
        "zero_based_column_index",
        "expression_source",
        "gene_identifier_source",
    ],
)

selected_cells = sum(
    int(
        row[
            "selected_cell_count"
        ]
    )
    for row in design_rows
)

primary_cells = sum(
    int(
        row[
            "selected_cell_count"
        ]
    )
    for row in design_rows
    if row[
        "effective_analysis_set"
    ] == "primary"
)

sensitivity_cells = sum(
    int(
        row[
            "selected_cell_count"
        ]
    )
    for row in design_rows
    if row[
        "effective_analysis_set"
    ] == "section_sensitivity"
)

planned_score_values = sum(
    int(
        row[
            "selected_cell_count"
        ]
    )
    for row in section_plan_rows
)

float32_score_bytes = (
    planned_score_values
    * 4
)

resource_rows = [
    {
        "metric": (
            "selected_cells_total"
        ),
        "value": selected_cells,
        "unit": "cells",
    },
    {
        "metric": (
            "primary_selected_cells"
        ),
        "value": primary_cells,
        "unit": "cells",
    },
    {
        "metric": (
            "section_sensitivity_selected_cells"
        ),
        "value": sensitivity_cells,
        "unit": "cells",
    },
    {
        "metric": (
            "section_module_score_definitions"
        ),
        "value": len(
            section_plan_rows
        ),
        "unit": "definitions",
    },
    {
        "metric": (
            "section_module_gene_rows"
        ),
        "value": len(
            module_gene_rows
        ),
        "unit": "rows",
    },
    {
        "metric": (
            "planned_cell_module_score_values"
        ),
        "value": planned_score_values,
        "unit": "float_values",
    },
    {
        "metric": (
            "minimum_float32_score_payload_bytes"
        ),
        "value": float32_score_bytes,
        "unit": "bytes",
    },
]

write_tsv(
    resource_dir
    / "phase10B5_P4D_R1_scoring_resource_plan.tsv",
    resource_rows,
    [
        "metric",
        "value",
        "unit",
    ],
)

guardrail_rows = [
    {
        "guardrail": (
            "primary_expression_source"
        ),
        "locked_value": "/X",
        "rationale": (
            "consistent_continuous_processed_source_"
            "across_all_six_H5AD_objects"
        ),
    },
    {
        "guardrail": (
            "count_reference_source"
        ),
        "locked_value": "/raw/X",
        "rationale": (
            "consistent_integer_count_like_source_"
            "retained_for_QC_or_sensitivity_only"
        ),
    },
    {
        "guardrail": (
            "cross_panel_gene_harmonization"
        ),
        "locked_value": (
            "intersection_across_all_12_sections"
        ),
        "rationale": (
            "prevents_panel_gene_composition_from_"
            "confounding_multiage_comparisons"
        ),
    },
    {
        "guardrail": (
            "expanded_panel_claim_scope"
        ),
        "locked_value": (
            "two_960_gene_primary_sections_only"
        ),
        "rationale": (
            "expanded_modules_are_not_supported_"
            "in_300_gene_objects"
        ),
    },
    {
        "guardrail": (
            "gene_standardization"
        ),
        "locked_value": (
            "within_section_gene_zscore_ddof0"
        ),
        "rationale": (
            "removes_section_specific_location_and_"
            "scale_before_module_averaging"
        ),
    },
    {
        "guardrail": (
            "module_score"
        ),
        "locked_value": (
            "unweighted_mean_of_standardized_genes"
        ),
        "rationale": (
            "matches_prespecified_module_scoring_"
            "framework"
        ),
    },
    {
        "guardrail": (
            "primary_statistical_unit"
        ),
        "locked_value": (
            "independent_donor_section"
        ),
        "rationale": (
            "millions_of_cells_are_not_independent_"
            "biological_replicates"
        ),
    },
    {
        "guardrail": (
            "cell_level_use"
        ),
        "locked_value": (
            "descriptive_distribution_and_"
            "annotation_aggregation_only"
        ),
        "rationale": (
            "prevents_cell_level_pseudoreplication"
        ),
    },
    {
        "guardrail": (
            "within_donor_sections"
        ),
        "locked_value": (
            "section_sensitivity_not_independent_"
            "donor_replication"
        ),
        "rationale": (
            "FB080_and_FB123_repeated_sections_share_"
            "donors"
        ),
    },
    {
        "guardrail": (
            "H1_annotation"
        ),
        "locked_value": (
            "primary_broad_cell_class"
        ),
        "rationale": (
            "complete_in_all_12_selected_sections"
        ),
    },
    {
        "guardrail": (
            "H2_annotation"
        ),
        "locked_value": (
            "secondary_cell_subclass"
        ),
        "rationale": (
            "complete_but_more_granular_and_"
            "age_specific"
        ),
    },
    {
        "guardrail": (
            "H3_annotation"
        ),
        "locked_value": (
            "exploratory_only"
        ),
        "rationale": (
            "substantially_incomplete_across_sections"
        ),
    },
]

write_tsv(
    guardrail_dir
    / "phase10B5_P4D_R1_analysis_guardrails.tsv",
    guardrail_rows,
    [
        "guardrail",
        "locked_value",
        "rationale",
    ],
)

access_rows = [
    {
        "processed_H5AD": row[
            "processed_H5AD"
        ],
        "H5AD_structure_accessed": True,
        "var_gene_identifiers_accessed": True,
        "X_values_accessed": False,
        "raw_X_values_accessed": False,
        "all_cells_expression_accessed": False,
        "module_scores_computed": False,
    }
    for row in source_lock_rows
]

write_tsv(
    access_dir
    / "phase10B5_P4D_R1_expression_access_guard.tsv",
    access_rows,
    [
        "processed_H5AD",
        "H5AD_structure_accessed",
        "var_gene_identifiers_accessed",
        "X_values_accessed",
        "raw_X_values_accessed",
        "all_cells_expression_accessed",
        "module_scores_computed",
    ],
)

technical_pass = (
    len(source_lock_rows) == 6
    and len(module_set_rows) == 9
    and len(section_plan_rows) == 68
    and len(module_gene_rows) == 534
    and selected_cells == 6_062_942
    and primary_cells == 4_306_468
    and sensitivity_cells == 1_756_474
    and planned_score_values == 36_733_794
    and all(
        row[
            "all_required_genes_present"
        ]
        for row in source_lock_rows
    )
)

status_value = (
    "passed_phase10B5_P4D_R1_processed_X_"
    "cross_panel_core_and_expanded_panel_"
    "source_lock_ready_for_chunked_"
    "sectionwise_module_scoring"
    if technical_pass
    else (
        "phase10B5_P4D_R1_requires_manual_review"
    )
)

status = {
    "phase": "phase10B5_P4D_R1",
    "H5AD_files_locked": len(
        source_lock_rows
    ),
    "revised_processed_sections": len(
        design_rows
    ),
    "primary_sections": 8,
    "section_sensitivity_sections": 4,
    "selected_cells_total": selected_cells,
    "primary_selected_cells": primary_cells,
    "section_sensitivity_selected_cells": (
        sensitivity_cells
    ),
    "cross_panel_core_modules": len(
        core_modules
    ),
    "expanded_panel_only_modules": len(
        expanded_modules
    ),
    "harmonized_module_gene_sets": len(
        module_set_rows
    ),
    "section_module_score_definitions": len(
        section_plan_rows
    ),
    "section_module_gene_rows": len(
        module_gene_rows
    ),
    "planned_cell_module_score_values": (
        planned_score_values
    ),
    "primary_expression_source": "/X",
    "count_reference_source": "/raw/X",
    "source_lock_finalized": True,
    "module_scoring_authorized": True,
    "expression_values_accessed": False,
    "module_scores_computed": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P4D_R1_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B5_P4D_R1_status.tsv",
    [status],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P4D-R1 EXPRESSION "
    "SOURCE AND HARMONIZED SCORING LOCK =====",
    "",
    (
        "H5AD files locked: "
        f"{len(source_lock_rows)}/6"
    ),
    (
        "Revised processed sections: "
        f"{len(design_rows)}"
    ),
    (
        "Selected cells: "
        f"{selected_cells}"
    ),
    (
        "Cross-panel core modules: "
        f"{len(core_modules)}"
    ),
    (
        "Expanded-panel-only modules: "
        f"{len(expanded_modules)}"
    ),
    (
        "Section-module score definitions: "
        f"{len(section_plan_rows)}"
    ),
    (
        "Section-module-gene rows: "
        f"{len(module_gene_rows)}"
    ),
    (
        "Planned cell-module score values: "
        f"{planned_score_values}"
    ),
    "",
    "Primary expression source: /X",
    "Count reference source: /raw/X",
    (
        "Cross-panel broad scores use harmonized "
        "core genes: TRUE"
    ),
    (
        "Primary statistical unit is donor/section: "
        "TRUE"
    ),
    "Source lock finalized: TRUE",
    "Module scoring authorized: TRUE",
    "Expression values accessed: FALSE",
    "Module scores computed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P4D-R1 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P4D_R1_report.txt"
).write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

print()
print(
    "\n".join(report)
)

print(
    "\n===== HARMONIZED MODULE GENE SETS ====="
)

for row in module_set_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "scoring_family",
                "module",
                "prespecified_claim_class",
                "gene_count",
                "gene_symbols",
            )
        )
    )

print(
    "\n===== RESOURCE PLAN ====="
)

for row in resource_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "metric",
                "value",
                "unit",
            )
        )
    )

checksum_rows: list[
    dict[str, Any]
] = []

for path in sorted(
    out.rglob("*")
):

    if (
        path.is_file()
        and path.name
        != "phase10B5_P4D_R1_SHA256.tsv"
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

write_tsv(
    out
    / "phase10B5_P4D_R1_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)

if not technical_pass:

    raise SystemExit(
        "Phase 10B5-P4D-R1 requires manual review."
    )
