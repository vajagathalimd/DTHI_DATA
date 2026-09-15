from __future__ import annotations

import csv
import hashlib
import itertools
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


project = Path(sys.argv[1])
moment_path = Path(sys.argv[2])
all8_score_path = Path(sys.argv[3])
module_set_path = Path(sys.argv[4])
out = Path(sys.argv[5])

panel_dir = out / "01_panel_class_audit"
primary_dir = out / "02_primary_300_panel_scores"
test_dir = out / "03_primary_exact_temporal_tests"
lodo_dir = out / "04_leave_one_section_out"
config_dir = out / "05_section_selection_sensitivity"
adjusted_dir = out / "06_all8_panel_adjusted_sensitivity"
evidence_dir = out / "07_temporal_evidence_hierarchy"
guardrail_dir = out / "08_analysis_guardrails"
audit_dir = out / "09_audit"

for directory in (
    panel_dir,
    primary_dir,
    test_dir,
    lodo_dir,
    config_dir,
    adjusted_dir,
    evidence_dir,
    guardrail_dir,
    audit_dir,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

variance_tolerance = 1e-12
comparison_tolerance = 1e-12


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


def average_ranks(
    values: Iterable[float],
) -> np.ndarray:

    array = np.asarray(
        list(values),
        dtype=np.float64,
    )

    order = np.argsort(
        array,
        kind="mergesort",
    )

    ranks = np.empty(
        array.size,
        dtype=np.float64,
    )

    start = 0

    while start < array.size:

        end = start + 1

        while (
            end < array.size
            and array[
                order[end]
            ] == array[
                order[start]
            ]
        ):

            end += 1

        average_rank = (
            start
            + end
            - 1
        ) / 2.0 + 1.0

        ranks[
            order[start:end]
        ] = average_rank

        start = end

    return ranks


def spearman_rho(
    first: Iterable[float],
    second: Iterable[float],
) -> float:

    first_ranks = average_ranks(
        first
    )

    second_ranks = average_ranks(
        second
    )

    if (
        np.std(
            first_ranks
        )
        <= variance_tolerance
        or np.std(
            second_ranks
        )
        <= variance_tolerance
    ):

        return float("nan")

    return float(
        np.corrcoef(
            first_ranks,
            second_ranks,
        )[0, 1]
    )


def simple_ols_slope(
    age: np.ndarray,
    score: np.ndarray,
) -> float:

    design = np.column_stack(
        [
            np.ones(
                age.size,
                dtype=np.float64,
            ),
            age.astype(
                np.float64
            ),
        ]
    )

    beta = np.linalg.lstsq(
        design,
        score,
        rcond=None,
    )[0]

    return float(
        beta[1]
    )


def age_panel_ols(
    age: np.ndarray,
    panel: np.ndarray,
    score: np.ndarray,
) -> dict[str, float]:

    design = np.column_stack(
        [
            np.ones(
                age.size,
                dtype=np.float64,
            ),
            age.astype(
                np.float64
            ),
            panel.astype(
                np.float64
            ),
        ]
    )

    beta = np.linalg.lstsq(
        design,
        score,
        rcond=None,
    )[0]

    fitted = design @ beta

    residual = (
        score
        - fitted
    )

    degrees_of_freedom = (
        age.size
        - design.shape[1]
    )

    residual_sum_squares = float(
        residual
        @ residual
    )

    if degrees_of_freedom <= 0:

        age_t = float("nan")

    else:

        residual_variance = (
            residual_sum_squares
            / degrees_of_freedom
        )

        covariance = (
            residual_variance
            * np.linalg.pinv(
                design.T
                @ design
            )
        )

        age_standard_error = math.sqrt(
            max(
                float(
                    covariance[1, 1]
                ),
                0.0,
            )
        )

        age_t = (
            float(
                beta[1]
                / age_standard_error
            )
            if age_standard_error
            > variance_tolerance
            else float("nan")
        )

    total_sum_squares = float(
        np.sum(
            (
                score
                - score.mean()
            )
            ** 2
        )
    )

    r_squared = (
        1.0
        - residual_sum_squares
        / total_sum_squares
        if total_sum_squares
        > variance_tolerance
        else 0.0
    )

    return {
        "age_beta": float(
            beta[1]
        ),
        "panel_beta": float(
            beta[2]
        ),
        "age_t": age_t,
        "r_squared": float(
            r_squared
        ),
    }


def unique_permutations(
    values: Iterable[int],
) -> list[tuple[int, ...]]:

    counts = Counter(
        int(value)
        for value in values
    )

    total_length = sum(
        counts.values()
    )

    output: list[
        tuple[int, ...]
    ] = []

    def recurse(
        prefix: list[int],
    ) -> None:

        if len(
            prefix
        ) == total_length:

            output.append(
                tuple(
                    prefix
                )
            )

            return

        for value in sorted(
            counts
        ):

            if counts[
                value
            ] == 0:

                continue

            counts[
                value
            ] -= 1

            prefix.append(
                value
            )

            recurse(
                prefix
            )

            prefix.pop()

            counts[
                value
            ] += 1

    recurse(
        []
    )

    return output


def exact_spearman_test(
    ages: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:

    observed_rho = spearman_rho(
        ages,
        scores,
    )

    observed_slope = simple_ols_slope(
        ages,
        scores,
    )

    permutations = unique_permutations(
        ages.astype(
            int
        )
    )

    permutation_rhos: list[float] = []

    permutation_slopes: list[float] = []

    for permutation in permutations:

        permuted_age = np.asarray(
            permutation,
            dtype=np.float64,
        )

        permutation_rhos.append(
            spearman_rho(
                permuted_age,
                scores,
            )
        )

        permutation_slopes.append(
            simple_ols_slope(
                permuted_age,
                scores,
            )
        )

    rho_extreme = sum(
        abs(value)
        >= abs(
            observed_rho
        ) - comparison_tolerance
        for value in permutation_rhos
    )

    slope_extreme = sum(
        abs(value)
        >= abs(
            observed_slope
        ) - comparison_tolerance
        for value in permutation_slopes
    )

    return {
        "spearman_rho": (
            observed_rho
        ),
        "OLS_slope_per_GW": (
            observed_slope
        ),
        "exact_two_sided_spearman_p": (
            rho_extreme
            / len(
                permutations
            )
        ),
        "exact_two_sided_slope_p": (
            slope_extreme
            / len(
                permutations
            )
        ),
        "exact_permutations": len(
            permutations
        ),
        "minimum_attainable_exact_p": (
            1.0
            / len(
                permutations
            )
        ),
    }


def bh_adjust(
    p_values: dict[str, float],
) -> dict[str, float]:

    ordered = sorted(
        p_values.items(),
        key=lambda item: item[1],
    )

    count = len(
        ordered
    )

    adjusted: dict[
        str,
        float
    ] = {}

    running = 1.0

    for reverse_index in range(
        count - 1,
        -1,
        -1,
    ):

        key, value = ordered[
            reverse_index
        ]

        rank = (
            reverse_index
            + 1
        )

        candidate = min(
            1.0,
            value
            * count
            / rank,
        )

        running = min(
            running,
            candidate,
        )

        adjusted[
            key
        ] = running

    return adjusted


def direction_label(
    value: float,
) -> str:

    if value > comparison_tolerance:

        return "increasing_with_age"

    if value < -comparison_tolerance:

        return "decreasing_with_age"

    return "approximately_flat"


def direction_concordant(
    first: float,
    second: float,
) -> bool:

    if (
        abs(
            first
        )
        <= comparison_tolerance
        and abs(
            second
        )
        <= comparison_tolerance
    ):

        return True

    return (
        first
        * second
        > 0
    )


moment_rows, moment_columns = read_tsv(
    moment_path
)

all8_rows, all8_columns = read_tsv(
    all8_score_path
)

module_rows, module_columns = read_tsv(
    module_set_path
)

if len(
    moment_rows
) != 488:

    raise SystemExit(
        f"FAIL: expected 488 section-gene "
        f"moment rows; observed "
        f"{len(moment_rows)}."
    )

if len(
    all8_rows
) != 60:

    raise SystemExit(
        f"FAIL: expected 60 core score rows; "
        f"observed {len(all8_rows)}."
    )

core_module_rows = [
    row
    for row in module_rows
    if row[
        "scoring_family"
    ] == "cross_panel_harmonized_core"
]

if len(
    core_module_rows
) != 5:

    raise SystemExit(
        f"FAIL: expected five core modules; "
        f"observed {len(core_module_rows)}."
    )

module_genes = {
    row[
        "module"
    ]: [
        gene
        for gene in row[
            "gene_symbols"
        ].split(";")
        if gene
    ]
    for row in core_module_rows
}

claim_class = {
    row[
        "module"
    ]: row[
        "prespecified_claim_class"
    ]
    for row in core_module_rows
}

core_modules = sorted(
    module_genes
)

primary_claim_modules = sorted(
    module
    for module in core_modules
    if claim_class[
        module
    ] == (
        "broad_primary_multiage_"
        "multidonor"
    )
)

sensitivity_claim_modules = sorted(
    module
    for module in core_modules
    if claim_class[
        module
    ] == (
        "broad_sensitivity_multiage_"
        "multidonor"
    )
)

if len(
    primary_claim_modules
) != 4:

    raise SystemExit(
        "FAIL: expected four primary temporal "
        "modules."
    )

if len(
    sensitivity_claim_modules
) != 1:

    raise SystemExit(
        "FAIL: expected one prespecified "
        "sensitivity module."
    )

section_metadata: dict[
    str,
    dict[str, Any],
] = {}

moment_lookup: dict[
    tuple[str, str],
    float,
] = {}

for row in moment_rows:

    archive_name = row[
        "archive_name"
    ]

    metadata = {
        "effective_analysis_set": row[
            "effective_analysis_set"
        ],
        "archive_name": archive_name,
        "donor_id": row[
            "donor_id"
        ],
        "gestational_week": int(
            row[
                "gestational_week"
            ]
        ),
        "panel_class": row[
            "panel_class"
        ],
        "processed_H5AD": row[
            "processed_H5AD"
        ],
        "selected_cells": int(
            row[
                "selected_cells"
            ]
        ),
    }

    if archive_name in section_metadata:

        comparable = dict(
            section_metadata[
                archive_name
            ]
        )

        if comparable != metadata:

            raise SystemExit(
                f"FAIL: inconsistent section "
                f"metadata for {archive_name}."
            )

    else:

        section_metadata[
            archive_name
        ] = metadata

    moment_lookup[
        (
            archive_name,
            row[
                "gene_symbol"
            ],
        )
    ] = float(
        row[
            "section_gene_mean"
        ]
    )

if len(
    section_metadata
) != 12:

    raise SystemExit(
        f"FAIL: expected 12 sections; "
        f"observed {len(section_metadata)}."
    )

core_genes = sorted(
    set().union(
        *(
            set(
                genes
            )
            for genes in module_genes.values()
        )
    )
)

if len(
    core_genes
) != 36:

    raise SystemExit(
        f"FAIL: expected 36 core genes; "
        f"observed {len(core_genes)}."
    )

for archive_name in section_metadata:

    for gene in core_genes:

        if (
            archive_name,
            gene,
        ) not in moment_lookup:

            raise SystemExit(
                f"FAIL: {gene} missing for "
                f"{archive_name}."
            )


def section_sort_key(
    archive_name: str,
) -> tuple[Any, ...]:

    metadata = section_metadata[
        archive_name
    ]

    return (
        metadata[
            "gestational_week"
        ],
        metadata[
            "donor_id"
        ],
        archive_name,
    )


def calculate_scores(
    target_sections: list[str],
    reference_sections: list[str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, str], float],
]:

    reference_rows: list[
        dict[str, Any]
    ] = []

    standardized_rows: list[
        dict[str, Any]
    ] = []

    module_score_rows: list[
        dict[str, Any]
    ] = []

    standardized_lookup: dict[
        tuple[str, str],
        float,
    ] = {}

    reference_lookup: dict[
        str,
        tuple[float, float, bool],
    ] = {}

    for gene in core_genes:

        values = np.asarray(
            [
                moment_lookup[
                    (
                        archive_name,
                        gene,
                    )
                ]
                for archive_name in reference_sections
            ],
            dtype=np.float64,
        )

        center = float(
            values.mean()
        )

        scale = float(
            values.std(
                ddof=0
            )
        )

        zero_variance = (
            scale
            <= variance_tolerance
        )

        reference_lookup[
            gene
        ] = (
            center,
            scale,
            zero_variance,
        )

        reference_rows.append(
            {
                "gene_symbol": gene,
                "reference_sections": len(
                    reference_sections
                ),
                "reference_center": (
                    center
                ),
                "reference_population_SD": (
                    scale
                ),
                "zero_variance_gene": (
                    zero_variance
                ),
            }
        )

    for archive_name in target_sections:

        metadata = section_metadata[
            archive_name
        ]

        for gene in core_genes:

            value = moment_lookup[
                (
                    archive_name,
                    gene,
                )
            ]

            center, scale, zero_variance = (
                reference_lookup[
                    gene
                ]
            )

            standardized = (
                0.0
                if zero_variance
                else (
                    value
                    - center
                )
                / scale
            )

            standardized_lookup[
                (
                    archive_name,
                    gene,
                )
            ] = standardized

            standardized_rows.append(
                {
                    "effective_analysis_set": (
                        metadata[
                            "effective_analysis_set"
                        ]
                    ),
                    "archive_name": archive_name,
                    "donor_id": metadata[
                        "donor_id"
                    ],
                    "gestational_week": (
                        metadata[
                            "gestational_week"
                        ]
                    ),
                    "panel_class": metadata[
                        "panel_class"
                    ],
                    "gene_symbol": gene,
                    "section_gene_mean": (
                        value
                    ),
                    "reference_center": (
                        center
                    ),
                    "reference_population_SD": (
                        scale
                    ),
                    "standardized_gene_value": (
                        standardized
                    ),
                    "zero_variance_gene": (
                        zero_variance
                    ),
                }
            )

        for module in core_modules:

            genes = module_genes[
                module
            ]

            module_score = float(
                np.mean(
                    [
                        standardized_lookup[
                            (
                                archive_name,
                                gene,
                            )
                        ]
                        for gene in genes
                    ]
                )
            )

            module_score_rows.append(
                {
                    "effective_analysis_set": (
                        metadata[
                            "effective_analysis_set"
                        ]
                    ),
                    "archive_name": archive_name,
                    "donor_id": metadata[
                        "donor_id"
                    ],
                    "gestational_week": (
                        metadata[
                            "gestational_week"
                        ]
                    ),
                    "panel_class": metadata[
                        "panel_class"
                    ],
                    "processed_H5AD": metadata[
                        "processed_H5AD"
                    ],
                    "selected_cells": metadata[
                        "selected_cells"
                    ],
                    "module": module,
                    "prespecified_claim_class": (
                        claim_class[
                            module
                        ]
                    ),
                    "gene_count": len(
                        genes
                    ),
                    "module_score": (
                        module_score
                    ),
                }
            )

    return (
        reference_rows,
        standardized_rows,
        module_score_rows,
        standardized_lookup,
    )


primary_all8_rows = [
    row
    for row in all8_rows
    if row[
        "effective_analysis_set"
    ] == "primary"
]

if len(
    primary_all8_rows
) != 40:

    raise SystemExit(
        f"FAIL: expected 40 all-eight primary "
        f"score rows; observed "
        f"{len(primary_all8_rows)}."
    )

all8_score_lookup = {
    (
        row[
            "archive_name"
        ],
        row[
            "module"
        ],
    ): float(
        row[
            "module_score"
        ]
    )
    for row in primary_all8_rows
}

all8_primary_sections = sorted(
    {
        row[
            "archive_name"
        ]
        for row in primary_all8_rows
    },
    key=section_sort_key,
)

if len(
    all8_primary_sections
) != 8:

    raise SystemExit(
        "FAIL: expected eight independent "
        "primary sections."
    )

primary_300_sections = sorted(
    [
        archive_name
        for archive_name in all8_primary_sections
        if section_metadata[
            archive_name
        ][
            "panel_class"
        ] == "300_gene_panel"
    ],
    key=section_sort_key,
)

primary_960_sections = sorted(
    [
        archive_name
        for archive_name in all8_primary_sections
        if section_metadata[
            archive_name
        ][
            "panel_class"
        ] == "960_gene_panel"
    ],
    key=section_sort_key,
)

sensitivity_300_sections = sorted(
    [
        archive_name
        for archive_name, metadata
        in section_metadata.items()
        if (
            metadata[
                "effective_analysis_set"
            ] == "section_sensitivity"
            and metadata[
                "panel_class"
            ] == "300_gene_panel"
        )
    ],
    key=section_sort_key,
)

if len(
    primary_300_sections
) != 6:

    raise SystemExit(
        f"FAIL: expected six 300-panel primary "
        f"sections; observed "
        f"{len(primary_300_sections)}."
    )

if len(
    primary_960_sections
) != 2:

    raise SystemExit(
        f"FAIL: expected two 960-panel primary "
        f"sections; observed "
        f"{len(primary_960_sections)}."
    )

if len(
    sensitivity_300_sections
) != 4:

    raise SystemExit(
        f"FAIL: expected four 300-panel "
        f"sensitivity sections; observed "
        f"{len(sensitivity_300_sections)}."
    )

primary_300_ages = sorted(
    section_metadata[
        archive_name
    ][
        "gestational_week"
    ]
    for archive_name in primary_300_sections
)

if primary_300_ages != [
    15,
    15,
    20,
    20,
    22,
    34,
]:

    raise SystemExit(
        f"FAIL: unexpected 300-panel age design: "
        f"{primary_300_ages}"
    )

panel_section_rows: list[
    dict[str, Any]
] = []

for archive_name in all8_primary_sections:

    module_values = [
        all8_score_lookup[
            (
                archive_name,
                module,
            )
        ]
        for module in core_modules
    ]

    panel_section_rows.append(
        {
            "archive_name": archive_name,
            "donor_id": section_metadata[
                archive_name
            ][
                "donor_id"
            ],
            "gestational_week": (
                section_metadata[
                    archive_name
                ][
                    "gestational_week"
                ]
            ),
            "panel_class": section_metadata[
                archive_name
            ][
                "panel_class"
            ],
            "mean_across_five_core_modules": float(
                np.mean(
                    module_values
                )
            ),
            "minimum_core_module_score": float(
                np.min(
                    module_values
                )
            ),
            "maximum_core_module_score": float(
                np.max(
                    module_values
                )
            ),
        }
    )

write_tsv(
    panel_dir
    / "phase10B5_P4E2_primary_section_global_score_audit.tsv",
    panel_section_rows,
    [
        "archive_name",
        "donor_id",
        "gestational_week",
        "panel_class",
        "mean_across_five_core_modules",
        "minimum_core_module_score",
        "maximum_core_module_score",
    ],
)

panel_module_rows: list[
    dict[str, Any]
] = []

panel_flagged_modules = 0

for module in core_modules:

    score_rows = [
        row
        for row in primary_all8_rows
        if row[
            "module"
        ] == module
    ]

    values_300 = np.asarray(
        [
            float(
                row[
                    "module_score"
                ]
            )
            for row in score_rows
            if row[
                "panel_class"
            ] == "300_gene_panel"
        ],
        dtype=np.float64,
    )

    values_960 = np.asarray(
        [
            float(
                row[
                    "module_score"
                ]
            )
            for row in score_rows
            if row[
                "panel_class"
            ] == "960_gene_panel"
        ],
        dtype=np.float64,
    )

    all_values = np.asarray(
        [
            float(
                row[
                    "module_score"
                ]
            )
            for row in score_rows
        ],
        dtype=np.float64,
    )

    score_by_section = {
        row[
            "archive_name"
        ]: float(
            row[
                "module_score"
            ]
        )
        for row in score_rows
    }

    ordered_sections = sorted(
        score_by_section,
        key=lambda section: (
            score_by_section[
                section
            ],
            section,
        ),
    )

    ascending_rank = {
        section: index + 1
        for index, section
        in enumerate(
            ordered_sections
        )
    }

    ranks_960 = [
        ascending_rank[
            section
        ]
        for section in primary_960_sections
    ]

    both_960_bottom_two = (
        sorted(
            ranks_960
        ) == [
            1,
            2,
        ]
    )

    gw20_300_values = np.asarray(
        [
            score_by_section[
                section
            ]
            for section in primary_300_sections
            if section_metadata[
                section
            ][
                "gestational_week"
            ] == 20
        ],
        dtype=np.float64,
    )

    gw20_960_values = np.asarray(
        [
            score_by_section[
                section
            ]
            for section in primary_960_sections
            if section_metadata[
                section
            ][
                "gestational_week"
            ] == 20
        ],
        dtype=np.float64,
    )

    overall_sd = float(
        all_values.std(
            ddof=0
        )
    )

    panel_difference = float(
        values_960.mean()
        - values_300.mean()
    )

    standardized_panel_difference = (
        panel_difference
        / overall_sd
        if overall_sd
        > variance_tolerance
        else 0.0
    )

    panel_signature_flag = (
        both_960_bottom_two
        or abs(
            standardized_panel_difference
        ) >= 0.75
    )

    if panel_signature_flag:
        panel_flagged_modules += 1

    panel_module_rows.append(
        {
            "module": module,
            "prespecified_claim_class": (
                claim_class[
                    module
                ]
            ),
            "mean_300_panel_score": float(
                values_300.mean()
            ),
            "mean_960_panel_score": float(
                values_960.mean()
            ),
            "mean_960_minus_300": (
                panel_difference
            ),
            "overall_primary_score_SD": (
                overall_sd
            ),
            "standardized_960_minus_300": (
                standardized_panel_difference
            ),
            "960_section_rank_positions_ascending": (
                ";".join(
                    str(value)
                    for value in sorted(
                        ranks_960
                    )
                )
            ),
            "both_960_sections_are_bottom_two": (
                both_960_bottom_two
            ),
            "GW20_300_panel_mean": float(
                gw20_300_values.mean()
            ),
            "GW20_960_panel_value": float(
                gw20_960_values.mean()
            ),
            "GW20_960_minus_300": float(
                gw20_960_values.mean()
                - gw20_300_values.mean()
            ),
            "panel_signature_flag": (
                panel_signature_flag
            ),
        }
    )

write_tsv(
    panel_dir
    / "phase10B5_P4E2_module_panel_class_audit.tsv",
    panel_module_rows,
    [
        "module",
        "prespecified_claim_class",
        "mean_300_panel_score",
        "mean_960_panel_score",
        "mean_960_minus_300",
        "overall_primary_score_SD",
        "standardized_960_minus_300",
        "960_section_rank_positions_ascending",
        "both_960_sections_are_bottom_two",
        "GW20_300_panel_mean",
        "GW20_960_panel_value",
        "GW20_960_minus_300",
        "panel_signature_flag",
    ],
)

panel_signature_detected = (
    panel_flagged_modules
    >= 4
)

all_300_sections = sorted(
    primary_300_sections
    + sensitivity_300_sections,
    key=section_sort_key,
)

(
    primary_reference_rows,
    primary_standardized_rows,
    primary_300_score_rows,
    primary_300_standardized_lookup,
) = calculate_scores(
    all_300_sections,
    primary_300_sections,
)

if len(
    primary_reference_rows
) != 36:

    raise SystemExit(
        "FAIL: expected 36 panel-homogeneous "
        "gene-reference rows."
    )

if len(
    primary_standardized_rows
) != 360:

    raise SystemExit(
        f"FAIL: expected 360 standardized "
        f"300-panel section-gene rows; observed "
        f"{len(primary_standardized_rows)}."
    )

if len(
    primary_300_score_rows
) != 50:

    raise SystemExit(
        f"FAIL: expected 50 300-panel module-score "
        f"rows; observed "
        f"{len(primary_300_score_rows)}."
    )

write_tsv(
    primary_dir
    / "phase10B5_P4E2_300_panel_gene_reference.tsv",
    primary_reference_rows,
    [
        "gene_symbol",
        "reference_sections",
        "reference_center",
        "reference_population_SD",
        "zero_variance_gene",
    ],
)

write_tsv(
    primary_dir
    / "phase10B5_P4E2_300_panel_standardized_section_genes.tsv",
    primary_standardized_rows,
    [
        "effective_analysis_set",
        "archive_name",
        "donor_id",
        "gestational_week",
        "panel_class",
        "gene_symbol",
        "section_gene_mean",
        "reference_center",
        "reference_population_SD",
        "standardized_gene_value",
        "zero_variance_gene",
    ],
)

write_tsv(
    primary_dir
    / "phase10B5_P4E2_300_panel_section_module_scores.tsv",
    primary_300_score_rows,
    [
        "effective_analysis_set",
        "archive_name",
        "donor_id",
        "gestational_week",
        "panel_class",
        "processed_H5AD",
        "selected_cells",
        "module",
        "prespecified_claim_class",
        "gene_count",
        "module_score",
    ],
)

primary_score_lookup = {
    (
        row[
            "archive_name"
        ],
        row[
            "module"
        ],
    ): float(
        row[
            "module_score"
        ]
    )
    for row in primary_300_score_rows
}

primary_test_rows: list[
    dict[str, Any]
] = []

primary_test_lookup: dict[
    str,
    dict[str, Any],
] = {}

ages_300 = np.asarray(
    [
        section_metadata[
            archive_name
        ][
            "gestational_week"
        ]
        for archive_name in primary_300_sections
    ],
    dtype=np.float64,
)

for module in core_modules:

    scores = np.asarray(
        [
            primary_score_lookup[
                (
                    archive_name,
                    module,
                )
            ]
            for archive_name in primary_300_sections
        ],
        dtype=np.float64,
    )

    result = exact_spearman_test(
        ages_300,
        scores,
    )

    row = {
        "module": module,
        "prespecified_claim_class": (
            claim_class[
                module
            ]
        ),
        "primary_BH_family_included": (
            module
            in primary_claim_modules
        ),
        "primary_sections": 6,
        "independent_donors": 6,
        "panel_class": (
            "300_gene_panel"
        ),
        "gestational_weeks": (
            "15;15;20;20;22;34"
        ),
        **result,
        "direction": direction_label(
            result[
                "spearman_rho"
            ]
        ),
    }

    primary_test_rows.append(
        row
    )

    primary_test_lookup[
        module
    ] = row

primary_q = bh_adjust(
    {
        module: float(
            primary_test_lookup[
                module
            ][
                "exact_two_sided_spearman_p"
            ]
        )
        for module in primary_claim_modules
    }
)

for row in primary_test_rows:

    row[
        "BH_q_primary_four_module_family"
    ] = (
        primary_q[
            row[
                "module"
            ]
        ]
        if row[
            "module"
        ] in primary_q
        else ""
    )

write_tsv(
    test_dir
    / "phase10B5_P4E2_primary_300_panel_exact_temporal_tests.tsv",
    primary_test_rows,
    [
        "module",
        "prespecified_claim_class",
        "primary_BH_family_included",
        "primary_sections",
        "independent_donors",
        "panel_class",
        "gestational_weeks",
        "spearman_rho",
        "OLS_slope_per_GW",
        "exact_two_sided_spearman_p",
        "exact_two_sided_slope_p",
        "exact_permutations",
        "minimum_attainable_exact_p",
        "direction",
        "BH_q_primary_four_module_family",
    ],
)

lodo_rows: list[
    dict[str, Any]
] = []

for omitted_section in primary_300_sections:

    retained_sections = [
        section
        for section in primary_300_sections
        if section != omitted_section
    ]

    (
        lodo_reference,
        lodo_standardized,
        lodo_scores,
        lodo_lookup,
    ) = calculate_scores(
        retained_sections,
        retained_sections,
    )

    lodo_score_lookup = {
        (
            row[
                "archive_name"
            ],
            row[
                "module"
            ],
        ): float(
            row[
                "module_score"
            ]
        )
        for row in lodo_scores
    }

    retained_ages = np.asarray(
        [
            section_metadata[
                section
            ][
                "gestational_week"
            ]
            for section in retained_sections
        ],
        dtype=np.float64,
    )

    for module in core_modules:

        retained_scores = np.asarray(
            [
                lodo_score_lookup[
                    (
                        section,
                        module,
                    )
                ]
                for section in retained_sections
            ],
            dtype=np.float64,
        )

        result = exact_spearman_test(
            retained_ages,
            retained_scores,
        )

        full_rho = float(
            primary_test_lookup[
                module
            ][
                "spearman_rho"
            ]
        )

        lodo_rows.append(
            {
                "module": module,
                "omitted_archive": (
                    omitted_section
                ),
                "omitted_donor": (
                    section_metadata[
                        omitted_section
                    ][
                        "donor_id"
                    ]
                ),
                "omitted_GW": (
                    section_metadata[
                        omitted_section
                    ][
                        "gestational_week"
                    ]
                ),
                "retained_sections": 5,
                "spearman_rho": (
                    result[
                        "spearman_rho"
                    ]
                ),
                "exact_two_sided_spearman_p": (
                    result[
                        "exact_two_sided_spearman_p"
                    ]
                ),
                "exact_permutations": (
                    result[
                        "exact_permutations"
                    ]
                ),
                "full_analysis_rho": (
                    full_rho
                ),
                "direction_concordant_with_full": (
                    direction_concordant(
                        full_rho,
                        result[
                            "spearman_rho"
                        ],
                    )
                ),
                "GW34_BA18_substitution_removed": (
                    omitted_section
                    == "UMB5900_BA18.zip"
                ),
            }
        )

if len(
    lodo_rows
) != 30:

    raise SystemExit(
        f"FAIL: expected 30 LODO rows; "
        f"observed {len(lodo_rows)}."
    )

write_tsv(
    lodo_dir
    / "phase10B5_P4E2_primary_300_panel_LODO_results.tsv",
    lodo_rows,
    [
        "module",
        "omitted_archive",
        "omitted_donor",
        "omitted_GW",
        "retained_sections",
        "spearman_rho",
        "exact_two_sided_spearman_p",
        "exact_permutations",
        "full_analysis_rho",
        "direction_concordant_with_full",
        "GW34_BA18_substitution_removed",
    ],
)

lodo_summary_rows: list[
    dict[str, Any]
] = []

lodo_summary_lookup: dict[
    str,
    dict[str, Any],
] = {}

for module in core_modules:

    rows = [
        row
        for row in lodo_rows
        if row[
            "module"
        ] == module
    ]

    rho_values = np.asarray(
        [
            float(
                row[
                    "spearman_rho"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    concordant = sum(
        as_bool(
            row[
                "direction_concordant_with_full"
            ]
        )
        for row in rows
    )

    row = {
        "module": module,
        "LODO_omissions": 6,
        "direction_concordant_omissions": (
            concordant
        ),
        "direction_concordance_fraction": (
            concordant
            / 6.0
        ),
        "minimum_LODO_rho": float(
            rho_values.min()
        ),
        "median_LODO_rho": float(
            np.median(
                rho_values
            )
        ),
        "maximum_LODO_rho": float(
            rho_values.max()
        ),
        "all_LODO_rhos_finite": bool(
            np.all(
                np.isfinite(
                    rho_values
                )
            )
        ),
    }

    lodo_summary_rows.append(
        row
    )

    lodo_summary_lookup[
        module
    ] = row

write_tsv(
    lodo_dir
    / "phase10B5_P4E2_primary_300_panel_LODO_summary.tsv",
    lodo_summary_rows,
    [
        "module",
        "LODO_omissions",
        "direction_concordant_omissions",
        "direction_concordance_fraction",
        "minimum_LODO_rho",
        "median_LODO_rho",
        "maximum_LODO_rho",
        "all_LODO_rhos_finite",
    ],
)

sections_300_by_donor: dict[
    str,
    list[str],
] = defaultdict(list)

for archive_name in all_300_sections:

    sections_300_by_donor[
        section_metadata[
            archive_name
        ][
            "donor_id"
        ]
    ].append(
        archive_name
    )

for donor in sections_300_by_donor:

    sections_300_by_donor[
        donor
    ].sort(
        key=section_sort_key
    )

variable_donors = sorted(
    donor
    for donor, sections
    in sections_300_by_donor.items()
    if len(
        sections
    ) > 1
)

if set(
    variable_donors
) != {
    "FB080",
    "FB123",
}:

    raise SystemExit(
        "FAIL: expected FB080 and FB123 as the "
        "two section-selection sensitivity donors."
    )

if len(
    sections_300_by_donor[
        "FB080"
    ]
) != 4:

    raise SystemExit(
        "FAIL: expected four FB080 section choices."
    )

if len(
    sections_300_by_donor[
        "FB123"
    ]
) != 2:

    raise SystemExit(
        "FAIL: expected two FB123 section choices."
    )

fixed_donors = sorted(
    donor
    for donor, sections
    in sections_300_by_donor.items()
    if len(
        sections
    ) == 1
)

fixed_sections = [
    sections_300_by_donor[
        donor
    ][0]
    for donor in fixed_donors
]

configuration_section_rows: list[
    dict[str, Any]
] = []

configuration_test_rows: list[
    dict[str, Any]
] = []

default_configuration_id = ""

configuration_counter = 0

for fb080_section, fb123_section in itertools.product(
    sections_300_by_donor[
        "FB080"
    ],
    sections_300_by_donor[
        "FB123"
    ],
):

    configuration_counter += 1

    configuration_id = (
        f"C{configuration_counter:02d}"
    )

    sections = sorted(
        fixed_sections
        + [
            fb080_section,
            fb123_section,
        ],
        key=section_sort_key,
    )

    if len(
        sections
    ) != 6:

        raise SystemExit(
            f"FAIL: {configuration_id} does not "
            "contain six independent donors."
        )

    donors = {
        section_metadata[
            section
        ][
            "donor_id"
        ]
        for section in sections
    }

    if len(
        donors
    ) != 6:

        raise SystemExit(
            f"FAIL: {configuration_id} repeats "
            "a donor."
        )

    configuration_ages = sorted(
        section_metadata[
            section
        ][
            "gestational_week"
        ]
        for section in sections
    )

    if configuration_ages != [
        15,
        15,
        20,
        20,
        22,
        34,
    ]:

        raise SystemExit(
            f"FAIL: {configuration_id} has an "
            f"unexpected age design: "
            f"{configuration_ages}"
        )

    default_configuration = all(
        section_metadata[
            section
        ][
            "effective_analysis_set"
        ] == "primary"
        for section in sections
    )

    if default_configuration:

        default_configuration_id = (
            configuration_id
        )

    for section in sections:

        configuration_section_rows.append(
            {
                "configuration_id": (
                    configuration_id
                ),
                "default_primary_configuration": (
                    default_configuration
                ),
                "archive_name": section,
                "donor_id": (
                    section_metadata[
                        section
                    ][
                        "donor_id"
                    ]
                ),
                "gestational_week": (
                    section_metadata[
                        section
                    ][
                        "gestational_week"
                    ]
                ),
                "original_analysis_set": (
                    section_metadata[
                        section
                    ][
                        "effective_analysis_set"
                    ]
                ),
            }
        )

    (
        configuration_reference,
        configuration_standardized,
        configuration_scores,
        configuration_lookup,
    ) = calculate_scores(
        sections,
        sections,
    )

    score_lookup = {
        (
            row[
                "archive_name"
            ],
            row[
                "module"
            ],
        ): float(
            row[
                "module_score"
            ]
        )
        for row in configuration_scores
    }

    ages = np.asarray(
        [
            section_metadata[
                section
            ][
                "gestational_week"
            ]
            for section in sections
        ],
        dtype=np.float64,
    )

    for module in core_modules:

        scores = np.asarray(
            [
                score_lookup[
                    (
                        section,
                        module,
                    )
                ]
                for section in sections
            ],
            dtype=np.float64,
        )

        result = exact_spearman_test(
            ages,
            scores,
        )

        configuration_test_rows.append(
            {
                "configuration_id": (
                    configuration_id
                ),
                "default_primary_configuration": (
                    default_configuration
                ),
                "FB080_section": (
                    fb080_section
                ),
                "FB123_section": (
                    fb123_section
                ),
                "module": module,
                "spearman_rho": (
                    result[
                        "spearman_rho"
                    ]
                ),
                "exact_two_sided_spearman_p": (
                    result[
                        "exact_two_sided_spearman_p"
                    ]
                ),
                "exact_permutations": (
                    result[
                        "exact_permutations"
                    ]
                ),
                "direction": direction_label(
                    result[
                        "spearman_rho"
                    ]
                ),
            }
        )

if configuration_counter != 8:

    raise SystemExit(
        f"FAIL: expected eight valid section "
        f"configurations; observed "
        f"{configuration_counter}."
    )

if not default_configuration_id:

    raise SystemExit(
        "FAIL: default primary configuration "
        "was not identified."
    )

if len(
    configuration_section_rows
) != 48:

    raise SystemExit(
        f"FAIL: expected 48 configuration-section "
        f"rows; observed "
        f"{len(configuration_section_rows)}."
    )

if len(
    configuration_test_rows
) != 40:

    raise SystemExit(
        f"FAIL: expected 40 configuration-test "
        f"rows; observed "
        f"{len(configuration_test_rows)}."
    )

write_tsv(
    config_dir
    / "phase10B5_P4E2_section_selection_configurations.tsv",
    configuration_section_rows,
    [
        "configuration_id",
        "default_primary_configuration",
        "archive_name",
        "donor_id",
        "gestational_week",
        "original_analysis_set",
    ],
)

write_tsv(
    config_dir
    / "phase10B5_P4E2_section_selection_exact_tests.tsv",
    configuration_test_rows,
    [
        "configuration_id",
        "default_primary_configuration",
        "FB080_section",
        "FB123_section",
        "module",
        "spearman_rho",
        "exact_two_sided_spearman_p",
        "exact_permutations",
        "direction",
    ],
)

default_configuration_rho = {
    row[
        "module"
    ]: float(
        row[
            "spearman_rho"
        ]
    )
    for row in configuration_test_rows
    if as_bool(
        row[
            "default_primary_configuration"
        ]
    )
}

if len(
    default_configuration_rho
) != 5:

    raise SystemExit(
        "FAIL: default configuration does not "
        "contain five module tests."
    )

configuration_summary_rows: list[
    dict[str, Any]
] = []

configuration_summary_lookup: dict[
    str,
    dict[str, Any],
] = {}

for module in core_modules:

    rows = [
        row
        for row in configuration_test_rows
        if row[
            "module"
        ] == module
    ]

    rho_values = np.asarray(
        [
            float(
                row[
                    "spearman_rho"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    p_values = np.asarray(
        [
            float(
                row[
                    "exact_two_sided_spearman_p"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    reference_rho = (
        default_configuration_rho[
            module
        ]
    )

    concordant = sum(
        direction_concordant(
            reference_rho,
            float(
                row[
                    "spearman_rho"
                ]
            ),
        )
        for row in rows
    )

    row = {
        "module": module,
        "configurations": 8,
        "default_configuration_id": (
            default_configuration_id
        ),
        "default_configuration_rho": (
            reference_rho
        ),
        "direction_concordant_configurations": (
            concordant
        ),
        "direction_concordance_fraction": (
            concordant
            / 8.0
        ),
        "minimum_configuration_rho": float(
            rho_values.min()
        ),
        "median_configuration_rho": float(
            np.median(
                rho_values
            )
        ),
        "maximum_configuration_rho": float(
            rho_values.max()
        ),
        "minimum_exact_p": float(
            p_values.min()
        ),
        "maximum_exact_p": float(
            p_values.max()
        ),
        "configurations_with_nominal_p_le_0_05": int(
            np.count_nonzero(
                p_values
                <= 0.05
            )
        ),
    }

    configuration_summary_rows.append(
        row
    )

    configuration_summary_lookup[
        module
    ] = row

write_tsv(
    config_dir
    / "phase10B5_P4E2_section_selection_sensitivity_summary.tsv",
    configuration_summary_rows,
    [
        "module",
        "configurations",
        "default_configuration_id",
        "default_configuration_rho",
        "direction_concordant_configurations",
        "direction_concordance_fraction",
        "minimum_configuration_rho",
        "median_configuration_rho",
        "maximum_configuration_rho",
        "minimum_exact_p",
        "maximum_exact_p",
        "configurations_with_nominal_p_le_0_05",
    ],
)

all8_adjusted_rows: list[
    dict[str, Any]
] = []

panel_indicator = np.asarray(
    [
        (
            1
            if section_metadata[
                section
            ][
                "panel_class"
            ] == "960_gene_panel"
            else 0
        )
        for section in all8_primary_sections
    ],
    dtype=np.int64,
)

all8_ages = np.asarray(
    [
        section_metadata[
            section
        ][
            "gestational_week"
        ]
        for section in all8_primary_sections
    ],
    dtype=np.float64,
)

indices_300 = np.flatnonzero(
    panel_indicator
    == 0
)

indices_960 = np.flatnonzero(
    panel_indicator
    == 1
)

permutations_300 = unique_permutations(
    all8_ages[
        indices_300
    ].astype(
        int
    )
)

permutations_960 = unique_permutations(
    all8_ages[
        indices_960
    ].astype(
        int
    )
)

if (
    len(
        permutations_300
    ) != 180
    or len(
        permutations_960
    ) != 2
):

    raise SystemExit(
        "FAIL: unexpected within-panel "
        "permutation counts."
    )

for module in core_modules:

    scores = np.asarray(
        [
            all8_score_lookup[
                (
                    section,
                    module,
                )
            ]
            for section in all8_primary_sections
        ],
        dtype=np.float64,
    )

    observed = age_panel_ols(
        all8_ages,
        panel_indicator,
        scores,
    )

    permuted_t_values: list[
        float
    ] = []

    for permutation_300 in permutations_300:

        for permutation_960 in permutations_960:

            permuted_age = np.asarray(
                all8_ages,
                dtype=np.float64,
            ).copy()

            permuted_age[
                indices_300
            ] = np.asarray(
                permutation_300,
                dtype=np.float64,
            )

            permuted_age[
                indices_960
            ] = np.asarray(
                permutation_960,
                dtype=np.float64,
            )

            permuted = age_panel_ols(
                permuted_age,
                panel_indicator,
                scores,
            )

            permuted_t_values.append(
                float(
                    permuted[
                        "age_t"
                    ]
                )
            )

    extreme = sum(
        abs(
            value
        )
        >= abs(
            observed[
                "age_t"
            ]
        ) - comparison_tolerance
        for value in permuted_t_values
    )

    exact_p = (
        extreme
        / len(
            permuted_t_values
        )
    )

    all8_adjusted_rows.append(
        {
            "module": module,
            "prespecified_claim_class": (
                claim_class[
                    module
                ]
            ),
            "sections": 8,
            "panel_300_sections": 6,
            "panel_960_sections": 2,
            "model": (
                "module_score_intercept_plus_"
                "gestational_week_plus_panel_class"
            ),
            "age_beta_per_GW": (
                observed[
                    "age_beta"
                ]
            ),
            "panel_960_beta": (
                observed[
                    "panel_beta"
                ]
            ),
            "age_t_statistic": (
                observed[
                    "age_t"
                ]
            ),
            "model_R_squared": (
                observed[
                    "r_squared"
                ]
            ),
            "exact_within_panel_permutations": len(
                permuted_t_values
            ),
            "exact_two_sided_age_p": (
                exact_p
            ),
            "direction": direction_label(
                observed[
                    "age_beta"
                ]
            ),
            "analysis_role": (
                "panel_adjusted_sensitivity"
            ),
        }
    )

all8_q = bh_adjust(
    {
        row[
            "module"
        ]: float(
            row[
                "exact_two_sided_age_p"
            ]
        )
        for row in all8_adjusted_rows
    }
)

for row in all8_adjusted_rows:

    row[
        "BH_q_five_module_sensitivity_family"
    ] = all8_q[
        row[
            "module"
        ]
    ]

write_tsv(
    adjusted_dir
    / "phase10B5_P4E2_all8_panel_adjusted_exact_tests.tsv",
    all8_adjusted_rows,
    [
        "module",
        "prespecified_claim_class",
        "sections",
        "panel_300_sections",
        "panel_960_sections",
        "model",
        "age_beta_per_GW",
        "panel_960_beta",
        "age_t_statistic",
        "model_R_squared",
        "exact_within_panel_permutations",
        "exact_two_sided_age_p",
        "BH_q_five_module_sensitivity_family",
        "direction",
        "analysis_role",
    ],
)

all8_adjusted_lookup = {
    row[
        "module"
    ]: row
    for row in all8_adjusted_rows
}

evidence_rows: list[
    dict[str, Any]
] = []

for module in core_modules:

    primary = primary_test_lookup[
        module
    ]

    lodo = lodo_summary_lookup[
        module
    ]

    configuration = (
        configuration_summary_lookup[
            module
        ]
    )

    adjusted = (
        all8_adjusted_lookup[
            module
        ]
    )

    primary_p = float(
        primary[
            "exact_two_sided_spearman_p"
        ]
    )

    primary_q_value = primary.get(
        "BH_q_primary_four_module_family",
        "",
    )

    lodo_stable = (
        int(
            lodo[
                "direction_concordant_omissions"
            ]
        )
        >= 5
    )

    configuration_stable = (
        int(
            configuration[
                "direction_concordant_configurations"
            ]
        )
        >= 6
    )

    stable = (
        lodo_stable
        and configuration_stable
    )

    if module in primary_claim_modules:

        if (
            primary_q_value != ""
            and float(
                primary_q_value
            )
            <= 0.05
            and stable
        ):

            evidence_class = (
                "primary_exact_FDR_supported_"
                "and_directionally_stable"
            )

        elif (
            primary_p
            <= 0.05
            and stable
        ):

            evidence_class = (
                "primary_nominal_exact_supported_"
                "and_directionally_stable"
            )

        elif stable:

            evidence_class = (
                "primary_directionally_stable_"
                "descriptive"
            )

        else:

            evidence_class = (
                "primary_descriptive_or_unstable"
            )

    else:

        if (
            primary_p
            <= 0.05
            and stable
        ):

            evidence_class = (
                "prespecified_sensitivity_exact_"
                "supported_and_stable"
            )

        elif stable:

            evidence_class = (
                "prespecified_sensitivity_"
                "directionally_stable_descriptive"
            )

        else:

            evidence_class = (
                "prespecified_sensitivity_"
                "descriptive_or_unstable"
            )

    evidence_rows.append(
        {
            "module": module,
            "prespecified_claim_class": (
                claim_class[
                    module
                ]
            ),
            "primary_analysis": (
                "six_independent_300_panel_sections"
            ),
            "primary_spearman_rho": (
                primary[
                    "spearman_rho"
                ]
            ),
            "primary_exact_p": (
                primary_p
            ),
            "primary_BH_q": (
                primary_q_value
            ),
            "primary_direction": (
                primary[
                    "direction"
                ]
            ),
            "LODO_direction_concordance": (
                lodo[
                    "direction_concordance_fraction"
                ]
            ),
            "section_selection_direction_concordance": (
                configuration[
                    "direction_concordance_fraction"
                ]
            ),
            "all8_panel_adjusted_age_beta": (
                adjusted[
                    "age_beta_per_GW"
                ]
            ),
            "all8_panel_adjusted_exact_p": (
                adjusted[
                    "exact_two_sided_age_p"
                ]
            ),
            "all8_panel_adjusted_BH_q": (
                adjusted[
                    "BH_q_five_module_sensitivity_family"
                ]
            ),
            "all8_adjusted_direction": (
                adjusted[
                    "direction"
                ]
            ),
            "panel_signature_flag": next(
                row[
                    "panel_signature_flag"
                ]
                for row in panel_module_rows
                if row[
                    "module"
                ] == module
            ),
            "evidence_class": (
                evidence_class
            ),
            "cell_level_replication_used": (
                False
            ),
        }
    )

write_tsv(
    evidence_dir
    / "phase10B5_P4E2_temporal_evidence_hierarchy.tsv",
    evidence_rows,
    [
        "module",
        "prespecified_claim_class",
        "primary_analysis",
        "primary_spearman_rho",
        "primary_exact_p",
        "primary_BH_q",
        "primary_direction",
        "LODO_direction_concordance",
        "section_selection_direction_concordance",
        "all8_panel_adjusted_age_beta",
        "all8_panel_adjusted_exact_p",
        "all8_panel_adjusted_BH_q",
        "all8_adjusted_direction",
        "panel_signature_flag",
        "evidence_class",
        "cell_level_replication_used",
    ],
)

guardrail_rows = [
    {
        "guardrail": (
            "primary_temporal_panel"
        ),
        "locked_value": (
            "300_gene_panel_only"
        ),
        "reason": (
            "both_960_gene_sections_showed_a_"
            "systematic_negative_shift_across_"
            "the_core_modules"
        ),
    },
    {
        "guardrail": (
            "primary_independent_sections"
        ),
        "locked_value": "6",
        "reason": (
            "one_section_per_independent_donor"
        ),
    },
    {
        "guardrail": (
            "primary_age_test"
        ),
        "locked_value": (
            "exact_two_sided_Spearman_age_"
            "permutation_test"
        ),
        "reason": (
            "small_sample_and_tied_age_design"
        ),
    },
    {
        "guardrail": (
            "primary_multiple_testing"
        ),
        "locked_value": (
            "BH_across_four_broad_primary_modules"
        ),
        "reason": (
            "activity_dependent_plasticity_was_"
            "prespecified_as_sensitivity_only"
        ),
    },
    {
        "guardrail": (
            "within_donor_section_sensitivity"
        ),
        "locked_value": (
            "eight_one_section_per_donor_"
            "configurations"
        ),
        "reason": (
            "alternate_FB080_and_FB123_sections_"
            "are_not_independent_replicates"
        ),
    },
    {
        "guardrail": (
            "all8_analysis"
        ),
        "locked_value": (
            "panel_adjusted_exact_within_panel_"
            "age_permutation_sensitivity"
        ),
        "reason": (
            "retains_GW18_without_treating_"
            "panel_class_as_biological_signal"
        ),
    },
    {
        "guardrail": (
            "GW34_BA18_substitution"
        ),
        "locked_value": (
            "retained_in_primary_with_LODO_"
            "substitution_removal_audit"
        ),
        "reason": (
            "processed_BA17_section_was_unavailable"
        ),
    },
    {
        "guardrail": (
            "expanded_panel_modules"
        ),
        "locked_value": (
            "descriptive_GW18_vs_GW20_only"
        ),
        "reason": (
            "one_independent_section_per_age"
        ),
    },
    {
        "guardrail": (
            "cell_level_inference"
        ),
        "locked_value": "not_authorized",
        "reason": (
            "millions_of_cells_are_not_independent_"
            "biological_replicates"
        ),
    },
]

write_tsv(
    guardrail_dir
    / "phase10B5_P4E2_analysis_guardrails.tsv",
    guardrail_rows,
    [
        "guardrail",
        "locked_value",
        "reason",
    ],
)

primary_nominal_hits = sum(
    (
        row[
            "module"
        ] in primary_claim_modules
        and float(
            row[
                "exact_two_sided_spearman_p"
            ]
        )
        <= 0.05
    )
    for row in primary_test_rows
)

primary_FDR_hits = sum(
    (
        row[
            "module"
        ] in primary_claim_modules
        and row[
            "BH_q_primary_four_module_family"
        ] != ""
        and float(
            row[
                "BH_q_primary_four_module_family"
            ]
        )
        <= 0.05
    )
    for row in primary_test_rows
)

all8_adjusted_FDR_hits = sum(
    float(
        row[
            "BH_q_five_module_sensitivity_family"
        ]
    )
    <= 0.05
    for row in all8_adjusted_rows
)

technical_pass = (
    len(
        panel_module_rows
    ) == 5
    and len(
        primary_reference_rows
    ) == 36
    and len(
        primary_standardized_rows
    ) == 360
    and len(
        primary_300_score_rows
    ) == 50
    and len(
        primary_test_rows
    ) == 5
    and len(
        lodo_rows
    ) == 30
    and len(
        lodo_summary_rows
    ) == 5
    and configuration_counter
    == 8
    and len(
        configuration_test_rows
    ) == 40
    and len(
        configuration_summary_rows
    ) == 5
    and len(
        all8_adjusted_rows
    ) == 5
    and len(
        evidence_rows
    ) == 5
)

status_value = (
    "passed_phase10B5_P4E2_panel_homogeneous_"
    "exact_temporal_and_panel_adjusted_"
    "sensitivity_analysis_ready_for_"
    "temporal_result_synthesis"
    if technical_pass
    else (
        "phase10B5_P4E2_requires_manual_review"
    )
)

status = {
    "phase": "phase10B5_P4E2",
    "all8_primary_sections_audited": 8,
    "primary_300_panel_sections": 6,
    "primary_960_panel_sections": 2,
    "primary_300_panel_independent_donors": 6,
    "primary_300_panel_age_permutations": 180,
    "panel_audit_module_rows": len(
        panel_module_rows
    ),
    "panel_signature_flagged_modules": (
        panel_flagged_modules
    ),
    "panel_signature_detected": (
        panel_signature_detected
    ),
    "primary_300_gene_reference_rows": len(
        primary_reference_rows
    ),
    "primary_300_standardized_section_gene_rows": len(
        primary_standardized_rows
    ),
    "primary_300_section_module_score_rows": len(
        primary_300_score_rows
    ),
    "primary_exact_temporal_test_rows": len(
        primary_test_rows
    ),
    "primary_BH_test_family_modules": len(
        primary_claim_modules
    ),
    "primary_nominal_exact_hits": (
        primary_nominal_hits
    ),
    "primary_BH_FDR_hits": (
        primary_FDR_hits
    ),
    "LODO_result_rows": len(
        lodo_rows
    ),
    "LODO_summary_rows": len(
        lodo_summary_rows
    ),
    "section_selection_configurations": (
        configuration_counter
    ),
    "section_selection_test_rows": len(
        configuration_test_rows
    ),
    "all8_panel_adjusted_test_rows": len(
        all8_adjusted_rows
    ),
    "all8_within_panel_exact_permutations": 360,
    "all8_panel_adjusted_BH_FDR_hits": (
        all8_adjusted_FDR_hits
    ),
    "temporal_evidence_hierarchy_rows": len(
        evidence_rows
    ),
    "expression_values_accessed_in_this_phase": (
        False
    ),
    "H5AD_files_opened_in_this_phase": False,
    "cell_level_hypothesis_tests_performed": (
        False
    ),
    "primary_statistical_unit": (
        "independent_donor_section"
    ),
    "expanded_panel_formal_inference_performed": (
        False
    ),
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B5_P4E2_status": (
        status_value
    ),
}

write_tsv(
    out
    / "phase10B5_P4E2_status.tsv",
    [status],
    list(
        status.keys()
    ),
)

report = [
    "===== PHASE 10B5-P4E2 PANEL-AWARE "
    "EXACT TEMPORAL ANALYSIS =====",
    "",
    "All-eight primary sections audited: 8",
    "Primary homogeneous 300-panel sections: 6",
    "Primary 960-panel sections: 2",
    (
        "Panel-signature flagged modules: "
        f"{panel_flagged_modules}/5"
    ),
    (
        "Panel signature detected: "
        f"{panel_signature_detected}"
    ),
    (
        "Primary exact age permutations: "
        "180"
    ),
    (
        "Primary temporal module tests: "
        f"{len(primary_test_rows)}"
    ),
    (
        "Primary four-module nominal hits: "
        f"{primary_nominal_hits}"
    ),
    (
        "Primary four-module BH-FDR hits: "
        f"{primary_FDR_hits}"
    ),
    (
        "LODO analyses: "
        f"{len(lodo_rows)}"
    ),
    (
        "One-section-per-donor configurations: "
        f"{configuration_counter}"
    ),
    (
        "All-eight panel-adjusted exact tests: "
        f"{len(all8_adjusted_rows)}"
    ),
    (
        "All-eight within-panel permutations: "
        "360"
    ),
    (
        "All-eight adjusted BH-FDR hits: "
        f"{all8_adjusted_FDR_hits}"
    ),
    "",
    "Expression values accessed in this phase: FALSE",
    "H5AD files opened in this phase: FALSE",
    "Cell-level hypothesis tests performed: FALSE",
    (
        "Primary statistical unit: "
        "independent donor/section"
    ),
    "Expanded-panel formal inference performed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B5-P4E2 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B5_P4E2_report.txt"
).write_text(
    "\n".join(report)
    + "\n",
    encoding="utf-8",
)

print()
print(
    "\n".join(report)
)

print(
    "\n===== PANEL-CLASS AUDIT ====="
)

for row in panel_module_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "module",
                "mean_960_minus_300",
                "standardized_960_minus_300",
                "960_section_rank_positions_ascending",
                "both_960_sections_are_bottom_two",
                "panel_signature_flag",
            )
        )
    )

print(
    "\n===== PRIMARY EXACT TEMPORAL TESTS ====="
)

for row in primary_test_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "module",
                "spearman_rho",
                "exact_two_sided_spearman_p",
                "BH_q_primary_four_module_family",
                "direction",
            )
        )
    )

print(
    "\n===== TEMPORAL EVIDENCE HIERARCHY ====="
)

for row in evidence_rows:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in (
                "module",
                "primary_spearman_rho",
                "primary_exact_p",
                "primary_BH_q",
                "LODO_direction_concordance",
                "section_selection_direction_concordance",
                "all8_panel_adjusted_exact_p",
                "evidence_class",
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
        != "phase10B5_P4E2_SHA256.tsv"
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
    / "phase10B5_P4E2_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)

if not technical_pass:

    raise SystemExit(
        "Phase 10B5-P4E2 requires manual review."
    )
