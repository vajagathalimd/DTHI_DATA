from __future__ import annotations

import csv
import gzip
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


project = Path(sys.argv[1])
block_path = Path(sys.argv[2])
eligibility_path = Path(sys.argv[3])
out = Path(sys.argv[4])

primary_dir = out / "01_primary_results"
sensitivity_dir = out / "02_metric_sensitivity"
design_dir = out / "03_design_and_multiplicity"
audit_dir = out / "04_audit"

for directory in (
    primary_dir,
    sensitivity_dir,
    design_dir,
    audit_dir,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

permutation_count = 100_000
permutation_batch_size = 5_000
random_seed = 10_204_003
primary_metric = "trimmed_mean"

depth_order = (
    "SP",
    "superficial_CP",
    "L4",
    "deep_CP",
)

metric_order = (
    "trimmed_mean",
    "mean",
    "median",
)


def read_tsv(
    path: Path,
) -> list[dict[str, str]]:

    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:

        return list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )


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
    value: str,
) -> bool:

    return str(value).strip().upper() in {
        "TRUE",
        "T",
        "1",
        "YES",
    }


def bh_adjust(
    values: list[float],
) -> list[float]:

    p = np.asarray(
        values,
        dtype=float,
    )

    count = p.size

    order = np.argsort(
        p,
        kind="mergesort",
    )

    ranked = p[order]

    adjusted_ranked = (
        ranked
        * count
        / np.arange(
            1,
            count + 1,
            dtype=float,
        )
    )

    adjusted_ranked = np.minimum.accumulate(
        adjusted_ranked[::-1]
    )[::-1]

    adjusted_ranked = np.minimum(
        adjusted_ranked,
        1.0,
    )

    adjusted = np.empty(
        count,
        dtype=float,
    )

    adjusted[order] = adjusted_ranked

    return adjusted.tolist()


def holm_adjust(
    values: list[float],
) -> list[float]:

    p = np.asarray(
        values,
        dtype=float,
    )

    count = p.size

    order = np.argsort(
        p,
        kind="mergesort",
    )

    ranked = p[order]

    multipliers = np.arange(
        count,
        0,
        -1,
        dtype=float,
    )

    adjusted_ranked = ranked * multipliers

    adjusted_ranked = np.maximum.accumulate(
        adjusted_ranked
    )

    adjusted_ranked = np.minimum(
        adjusted_ranked,
        1.0,
    )

    adjusted = np.empty(
        count,
        dtype=float,
    )

    adjusted[order] = adjusted_ranked

    return adjusted.tolist()


def welch_statistics(
    matrix: np.ndarray,
    group_one_mask: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    group_one = matrix[
        group_one_mask,
        :,
    ]

    group_two = matrix[
        ~group_one_mask,
        :,
    ]

    n_one = group_one.shape[0]
    n_two = group_two.shape[0]

    mean_one = group_one.mean(
        axis=0,
    )

    mean_two = group_two.mean(
        axis=0,
    )

    difference = mean_one - mean_two

    variance_one = group_one.var(
        axis=0,
        ddof=1,
    )

    variance_two = group_two.var(
        axis=0,
        ddof=1,
    )

    denominator = np.sqrt(
        variance_one / n_one
        + variance_two / n_two
    )

    statistic = np.divide(
        difference,
        denominator,
        out=np.zeros_like(
            difference,
            dtype=float,
        ),
        where=denominator > 0,
    )

    pooled_denominator = (
        n_one + n_two - 2
    )

    pooled_variance = (
        (
            (n_one - 1) * variance_one
            + (n_two - 1) * variance_two
        )
        / pooled_denominator
    )

    pooled_sd = np.sqrt(
        np.maximum(
            pooled_variance,
            0,
        )
    )

    correction = (
        1
        - 3
        / (
            4 * (n_one + n_two)
            - 9
        )
    )

    hedges_g = np.divide(
        correction * difference,
        pooled_sd,
        out=np.full_like(
            difference,
            np.nan,
            dtype=float,
        ),
        where=pooled_sd > 0,
    )

    return (
        statistic,
        difference,
        mean_one,
        mean_two,
        hedges_g,
    )


def permutation_statistics(
    matrix: np.ndarray,
    membership: np.ndarray,
) -> np.ndarray:

    n_permutations = membership.shape[0]
    n_units = membership.shape[1]

    n_one = membership.sum(
        axis=1,
    ).astype(float)

    n_two = (
        n_units - n_one
    )

    total = matrix.sum(
        axis=0,
    )

    squared = matrix ** 2

    total_squared = squared.sum(
        axis=0,
    )

    sum_one = membership @ matrix
    sum_two = (
        total[np.newaxis, :]
        - sum_one
    )

    square_sum_one = membership @ squared
    square_sum_two = (
        total_squared[np.newaxis, :]
        - square_sum_one
    )

    mean_one = (
        sum_one
        / n_one[:, np.newaxis]
    )

    mean_two = (
        sum_two
        / n_two[:, np.newaxis]
    )

    variance_one = (
        square_sum_one
        - (
            sum_one ** 2
            / n_one[:, np.newaxis]
        )
    ) / (
        n_one[:, np.newaxis]
        - 1
    )

    variance_two = (
        square_sum_two
        - (
            sum_two ** 2
            / n_two[:, np.newaxis]
        )
    ) / (
        n_two[:, np.newaxis]
        - 1
    )

    variance_one = np.maximum(
        variance_one,
        0,
    )

    variance_two = np.maximum(
        variance_two,
        0,
    )

    denominator = np.sqrt(
        variance_one
        / n_one[:, np.newaxis]
        + variance_two
        / n_two[:, np.newaxis]
    )

    statistic = np.divide(
        mean_one - mean_two,
        denominator,
        out=np.zeros(
            (
                n_permutations,
                matrix.shape[1],
            ),
            dtype=float,
        ),
        where=denominator > 0,
    )

    return statistic


eligibility = read_tsv(
    eligibility_path
)

eligible_depths = [
    row["depth_stratum"]
    for row in eligibility
    if as_bool(
        row.get(
            "eligible_for_area_permutation",
            "",
        )
    )
]

if set(eligible_depths) != set(depth_order):
    raise SystemExit(
        "FAIL: expected SP, superficial_CP, L4, "
        "and deep_CP to be eligible."
    )

with gzip.open(
    block_path,
    "rt",
    encoding="utf-8",
    newline="",
) as handle:

    reader = csv.DictReader(
        handle,
        delimiter="\t",
    )

    fields = reader.fieldnames or []

    trimmed_suffix = "__trimmed_mean"

    modules = sorted(
        field[
            :-len(trimmed_suffix)
        ]
        for field in fields
        if field.endswith(
            trimmed_suffix
        )
    )

    if len(modules) != 9:
        raise SystemExit(
            f"FAIL: expected 9 modules, observed {len(modules)}"
        )

    required = {
        "candidate_id",
        "block_id",
        "area",
        "depth_stratum",
        "spots",
    }

    for module in modules:
        for metric in metric_order:
            required.add(
                f"{module}__{metric}"
            )

    missing = sorted(
        required - set(fields)
    )

    if missing:
        raise SystemExit(
            "FAIL: selected block table lacks columns: "
            + ";".join(missing)
        )

    rows = list(reader)

if not rows:
    raise SystemExit(
        "FAIL: selected block table is empty."
    )

candidate_ids = {
    row["candidate_id"]
    for row in rows
}

if candidate_ids != {
    "w04_ro00_co00",
}:
    raise SystemExit(
        "FAIL: unexpected selected candidate IDs: "
        + ";".join(
            sorted(candidate_ids)
        )
    )

rows_by_depth: dict[
    str,
    list[dict[str, str]],
] = defaultdict(list)

for row in rows:

    depth = row["depth_stratum"]

    if depth in depth_order:
        rows_by_depth[depth].append(
            row
        )

primary_results: list[dict[str, Any]] = []
sensitivity_results: list[dict[str, Any]] = []
depth_design_rows: list[dict[str, Any]] = []

seed_sequence = np.random.SeedSequence(
    random_seed
)

depth_seed_sequences = seed_sequence.spawn(
    len(depth_order)
)

for depth_index, depth in enumerate(
    depth_order
):

    depth_rows = rows_by_depth[
        depth
    ]

    block_ids = [
        row["block_id"]
        for row in depth_rows
    ]

    if len(block_ids) != len(
        set(block_ids)
    ):
        raise SystemExit(
            f"FAIL: duplicated block IDs within {depth}."
        )

    areas = np.asarray(
        [
            row["area"]
            for row in depth_rows
        ],
        dtype=object,
    )

    if not set(areas).issubset(
        {
            "V1",
            "V2",
        }
    ):
        raise SystemExit(
            f"FAIL: non-V1/V2 area in {depth}."
        )

    v1_mask = areas == "V1"

    n_v1 = int(
        v1_mask.sum()
    )

    n_v2 = int(
        (~v1_mask).sum()
    )

    if n_v1 < 4 or n_v2 < 4:
        raise SystemExit(
            f"FAIL: insufficient blocks in {depth}: "
            f"V1={n_v1}; V2={n_v2}"
        )

    matrices: dict[str, np.ndarray] = {}

    observed: dict[
        str,
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
    ] = {}

    for metric in metric_order:

        matrix = np.asarray(
            [
                [
                    float(
                        row[
                            f"{module}__{metric}"
                        ]
                    )
                    for module in modules
                ]
                for row in depth_rows
            ],
            dtype=float,
        )

        if not np.isfinite(
            matrix
        ).all():
            raise SystemExit(
                f"FAIL: nonfinite {metric} scores in {depth}."
            )

        matrices[metric] = matrix

        observed[metric] = welch_statistics(
            matrix,
            v1_mask,
        )

    raw_exceedance = {
        metric: np.zeros(
            len(modules),
            dtype=np.int64,
        )
        for metric in metric_order
    }

    family_exceedance = np.zeros(
        len(modules),
        dtype=np.int64,
    )

    primary_abs_observed = np.abs(
        observed[primary_metric][0]
    )

    rng = np.random.default_rng(
        depth_seed_sequences[
            depth_index
        ]
    )

    completed = 0

    while completed < permutation_count:

        current_batch = min(
            permutation_batch_size,
            permutation_count - completed,
        )

        random_values = rng.random(
            (
                current_batch,
                len(depth_rows),
            )
        )

        selected_indices = np.argpartition(
            random_values,
            kth=n_v1 - 1,
            axis=1,
        )[
            :,
            :n_v1,
        ]

        membership = np.zeros(
            (
                current_batch,
                len(depth_rows),
            ),
            dtype=float,
        )

        membership[
            np.arange(
                current_batch
            )[:, np.newaxis],
            selected_indices,
        ] = 1.0

        primary_permuted_abs = None

        for metric in metric_order:

            permuted = permutation_statistics(
                matrices[metric],
                membership,
            )

            permuted_abs = np.abs(
                permuted
            )

            observed_abs = np.abs(
                observed[metric][0]
            )

            raw_exceedance[metric] += (
                permuted_abs
                >= (
                    observed_abs[
                        np.newaxis,
                        :
                    ]
                    - 1e-12
                )
            ).sum(
                axis=0
            )

            if metric == primary_metric:
                primary_permuted_abs = (
                    permuted_abs
                )

        if primary_permuted_abs is None:
            raise SystemExit(
                "FAIL: primary permutation matrix absent."
            )

        family_maximum = primary_permuted_abs.max(
            axis=1
        )

        family_exceedance += (
            family_maximum[
                :,
                np.newaxis,
            ]
            >= (
                primary_abs_observed[
                    np.newaxis,
                    :
                ]
                - 1e-12
            )
        ).sum(
            axis=0
        )

        completed += current_batch

    exact_permutations = math.comb(
        n_v1 + n_v2,
        n_v1,
    )

    depth_design_rows.append(
        {
            "depth_stratum": depth,
            "total_blocks": (
                n_v1 + n_v2
            ),
            "V1_blocks": n_v1,
            "V2_blocks": n_v2,
            "exact_label_permutations": (
                exact_permutations
            ),
            "Monte_Carlo_permutations": (
                permutation_count
            ),
            "permutation_seed_entropy": (
                random_seed
            ),
            "permutation_unit": (
                "equal_weight_spatial_block"
            ),
            "permutation_scope": (
                "within_depth_stratum"
            ),
        }
    )

    primary_raw_p = (
        raw_exceedance[primary_metric]
        + 1
    ) / (
        permutation_count + 1
    )

    primary_family_p = (
        family_exceedance + 1
    ) / (
        permutation_count + 1
    )

    for module_index, module in enumerate(
        modules
    ):

        (
            observed_statistic,
            observed_difference,
            mean_v1,
            mean_v2,
            hedges_g,
        ) = observed[
            primary_metric
        ]

        p_value = float(
            primary_raw_p[
                module_index
            ]
        )

        primary_results.append(
            {
                "depth_stratum": depth,
                "module": module,
                "primary_block_metric": (
                    primary_metric
                ),
                "V1_blocks": n_v1,
                "V2_blocks": n_v2,
                "V1_block_mean": float(
                    mean_v1[
                        module_index
                    ]
                ),
                "V2_block_mean": float(
                    mean_v2[
                        module_index
                    ]
                ),
                "V1_minus_V2": float(
                    observed_difference[
                        module_index
                    ]
                ),
                "Welch_studentized_statistic": float(
                    observed_statistic[
                        module_index
                    ]
                ),
                "Hedges_g": float(
                    hedges_g[
                        module_index
                    ]
                ),
                "direction": (
                    "V1_higher"
                    if observed_difference[
                        module_index
                    ] > 0
                    else (
                        "V2_higher"
                        if observed_difference[
                            module_index
                        ] < 0
                        else "equal"
                    )
                ),
                "permutation_p_two_sided": p_value,
                "permutation_p_Monte_Carlo_SE": (
                    math.sqrt(
                        p_value
                        * (
                            1 - p_value
                        )
                        / (
                            permutation_count + 1
                        )
                    )
                ),
                "within_depth_maxT_p": float(
                    primary_family_p[
                        module_index
                    ]
                ),
                "Monte_Carlo_permutations": (
                    permutation_count
                ),
                "exact_label_permutations": (
                    exact_permutations
                ),
            }
        )

        for metric in metric_order:

            (
                metric_statistic,
                metric_difference,
                metric_mean_v1,
                metric_mean_v2,
                metric_hedges_g,
            ) = observed[metric]

            metric_p = (
                raw_exceedance[metric][
                    module_index
                ]
                + 1
            ) / (
                permutation_count + 1
            )

            sensitivity_results.append(
                {
                    "depth_stratum": depth,
                    "module": module,
                    "block_metric": metric,
                    "V1_blocks": n_v1,
                    "V2_blocks": n_v2,
                    "V1_block_mean": float(
                        metric_mean_v1[
                            module_index
                        ]
                    ),
                    "V2_block_mean": float(
                        metric_mean_v2[
                            module_index
                        ]
                    ),
                    "V1_minus_V2": float(
                        metric_difference[
                            module_index
                        ]
                    ),
                    "Welch_studentized_statistic": float(
                        metric_statistic[
                            module_index
                        ]
                    ),
                    "Hedges_g": float(
                        metric_hedges_g[
                            module_index
                        ]
                    ),
                    "direction": (
                        "V1_higher"
                        if metric_difference[
                            module_index
                        ] > 0
                        else (
                            "V2_higher"
                            if metric_difference[
                                module_index
                            ] < 0
                            else "equal"
                        )
                    ),
                    "permutation_p_two_sided": float(
                        metric_p
                    ),
                    "Monte_Carlo_permutations": (
                        permutation_count
                    ),
                }
            )

global_raw_p = [
    float(
        row["permutation_p_two_sided"]
    )
    for row in primary_results
]

global_bh = bh_adjust(
    global_raw_p
)

global_holm = holm_adjust(
    global_raw_p
)

for index, row in enumerate(
    primary_results
):

    row["global_BH_FDR"] = float(
        global_bh[index]
    )

    row["global_Holm_FWER"] = float(
        global_holm[index]
    )

for depth in depth_order:

    indices = [
        index
        for index, row in enumerate(
            primary_results
        )
        if row["depth_stratum"] == depth
    ]

    depth_p = [
        float(
            primary_results[index][
                "permutation_p_two_sided"
            ]
        )
        for index in indices
    ]

    depth_bh = bh_adjust(
        depth_p
    )

    for local_index, global_index in enumerate(
        indices
    ):
        primary_results[
            global_index
        ][
            "within_depth_BH_FDR"
        ] = float(
            depth_bh[
                local_index
            ]
        )

for row in primary_results:

    row[
        "passes_global_BH_0_05"
    ] = (
        row["global_BH_FDR"] < 0.05
    )

    row[
        "passes_global_Holm_0_05"
    ] = (
        row["global_Holm_FWER"] < 0.05
    )

    row[
        "passes_within_depth_maxT_0_05"
    ] = (
        row["within_depth_maxT_p"] < 0.05
    )

    row[
        "passes_within_depth_BH_0_05"
    ] = (
        row["within_depth_BH_FDR"] < 0.05
    )

    row[
        "section_specific_primary_signal"
    ] = (
        row["passes_global_BH_0_05"]
        and row[
            "passes_within_depth_maxT_0_05"
        ]
    )

primary_results.sort(
    key=lambda row: (
        float(
            row["global_BH_FDR"]
        ),
        float(
            row["within_depth_maxT_p"]
        ),
        -abs(
            float(
                row["Welch_studentized_statistic"]
            )
        ),
        depth_order.index(
            row["depth_stratum"]
        ),
        row["module"],
    )
)

primary_columns = [
    "depth_stratum",
    "module",
    "primary_block_metric",
    "V1_blocks",
    "V2_blocks",
    "V1_block_mean",
    "V2_block_mean",
    "V1_minus_V2",
    "Welch_studentized_statistic",
    "Hedges_g",
    "direction",
    "permutation_p_two_sided",
    "permutation_p_Monte_Carlo_SE",
    "within_depth_maxT_p",
    "within_depth_BH_FDR",
    "global_BH_FDR",
    "global_Holm_FWER",
    "passes_within_depth_maxT_0_05",
    "passes_within_depth_BH_0_05",
    "passes_global_BH_0_05",
    "passes_global_Holm_0_05",
    "section_specific_primary_signal",
    "Monte_Carlo_permutations",
    "exact_label_permutations",
]

write_tsv(
    primary_dir
    / "phase10B4_A3_primary_block_permutation_results.tsv",
    primary_results,
    primary_columns,
)

primary_lookup = {
    (
        row["depth_stratum"],
        row["module"],
    ): row
    for row in primary_results
}

for row in sensitivity_results:

    primary = primary_lookup[
        (
            row["depth_stratum"],
            row["module"],
        )
    ]

    row[
        "direction_concordant_with_primary"
    ] = (
        row["direction"]
        == primary["direction"]
    )

    row[
        "primary_global_BH_FDR"
    ] = primary[
        "global_BH_FDR"
    ]

    row[
        "primary_within_depth_maxT_p"
    ] = primary[
        "within_depth_maxT_p"
    ]

sensitivity_columns = [
    "depth_stratum",
    "module",
    "block_metric",
    "V1_blocks",
    "V2_blocks",
    "V1_block_mean",
    "V2_block_mean",
    "V1_minus_V2",
    "Welch_studentized_statistic",
    "Hedges_g",
    "direction",
    "permutation_p_two_sided",
    "direction_concordant_with_primary",
    "primary_global_BH_FDR",
    "primary_within_depth_maxT_p",
    "Monte_Carlo_permutations",
]

write_tsv(
    sensitivity_dir
    / "phase10B4_A3_block_metric_sensitivity.tsv",
    sensitivity_results,
    sensitivity_columns,
)

write_tsv(
    design_dir
    / "phase10B4_A3_depth_permutation_design.tsv",
    depth_design_rows,
    [
        "depth_stratum",
        "total_blocks",
        "V1_blocks",
        "V2_blocks",
        "exact_label_permutations",
        "Monte_Carlo_permutations",
        "permutation_seed_entropy",
        "permutation_unit",
        "permutation_scope",
    ],
)

multiplicity_rows = [
    {
        "family": "within_depth_modules",
        "tests_per_family": 9,
        "number_of_families": 4,
        "method": "Westfall_Young_style_maxT_from_shared_within_depth_permutations",
        "primary_role": (
            "within_depth_familywise_error_control"
        ),
    },
    {
        "family": "all_module_depth_tests",
        "tests_per_family": 36,
        "number_of_families": 1,
        "method": "Benjamini_Hochberg",
        "primary_role": (
            "global_false_discovery_rate_control"
        ),
    },
    {
        "family": "all_module_depth_tests",
        "tests_per_family": 36,
        "number_of_families": 1,
        "method": "Holm",
        "primary_role": (
            "global_familywise_error_sensitivity"
        ),
    },
]

write_tsv(
    design_dir
    / "phase10B4_A3_multiplicity_plan.tsv",
    multiplicity_rows,
    [
        "family",
        "tests_per_family",
        "number_of_families",
        "method",
        "primary_role",
    ],
)

global_bh_hits = sum(
    bool(
        row["passes_global_BH_0_05"]
    )
    for row in primary_results
)

global_holm_hits = sum(
    bool(
        row["passes_global_Holm_0_05"]
    )
    for row in primary_results
)

within_depth_maxT_hits = sum(
    bool(
        row[
            "passes_within_depth_maxT_0_05"
        ]
    )
    for row in primary_results
)

primary_signals = sum(
    bool(
        row[
            "section_specific_primary_signal"
        ]
    )
    for row in primary_results
)

sensitivity_concordance = sum(
    bool(
        row[
            "direction_concordant_with_primary"
        ]
    )
    for row in sensitivity_results
)

all_primary_finite = all(
    math.isfinite(
        float(
            row[
                "Welch_studentized_statistic"
            ]
        )
    )
    and math.isfinite(
        float(
            row[
                "permutation_p_two_sided"
            ]
        )
    )
    and math.isfinite(
        float(
            row[
                "global_BH_FDR"
            ]
        )
    )
    for row in primary_results
)

status_value = (
    "passed_phase10B4_A3_section_specific_"
    "multistratum_block_permutation_testing_"
    "ready_for_geometry_sensitivity_and_"
    "cross_modal_synthesis"
    if (
        len(primary_results) == 36
        and len(sensitivity_results) == 108
        and all_primary_finite
    )
    else (
        "phase10B4_A3_requires_manual_review"
    )
)

status = {
    "phase": "phase10B4_A3",
    "selected_geometry": "w04_ro00_co00",
    "eligible_depth_strata_tested": 4,
    "locked_modules_tested": 9,
    "primary_tests": len(
        primary_results
    ),
    "primary_block_metric": (
        primary_metric
    ),
    "permutations_per_depth": (
        permutation_count
    ),
    "permutation_unit": (
        "equal_weight_spatial_block"
    ),
    "within_depth_maxT_hits_0_05": (
        within_depth_maxT_hits
    ),
    "global_BH_hits_0_05": (
        global_bh_hits
    ),
    "global_Holm_hits_0_05": (
        global_holm_hits
    ),
    "section_specific_primary_signals": (
        primary_signals
    ),
    "metric_sensitivity_direction_concordance": (
        f"{sensitivity_concordance}/"
        f"{len(sensitivity_results)}"
    ),
    "single_Visium_section": True,
    "independent_biological_replication": False,
    "population_level_inference": False,
    "snRNAseq_inferential_tests_performed": False,
    "candidate_TFs_changed": False,
    "validation_hypotheses_changed": False,
    "phase10B4_A3_status": status_value,
}

write_tsv(
    out / "phase10B4_A3_status.tsv",
    [status],
    list(status.keys()),
)

report = [
    "===== PHASE 10B4-A3 SECTION-SPECIFIC "
    "BLOCK PERMUTATION TESTING =====",
    "",
    "Selected geometry: w04_ro00_co00",
    "Eligible depth strata tested: 4",
    "Locked modules tested: 9",
    (
        "Primary tests: "
        f"{len(primary_results)}"
    ),
    (
        "Primary block metric: "
        f"{primary_metric}"
    ),
    (
        "Monte Carlo permutations per depth: "
        f"{permutation_count}"
    ),
    (
        "Within-depth maxT hits at 0.05: "
        f"{within_depth_maxT_hits}"
    ),
    (
        "Global BH-FDR hits at 0.05: "
        f"{global_bh_hits}"
    ),
    (
        "Global Holm-FWER hits at 0.05: "
        f"{global_holm_hits}"
    ),
    (
        "Section-specific primary signals: "
        f"{primary_signals}"
    ),
    (
        "Metric-sensitivity direction concordance: "
        f"{sensitivity_concordance}/"
        f"{len(sensitivity_results)}"
    ),
    "",
    "Single Visium section: TRUE",
    "Independent biological replication: FALSE",
    "Population-level inference: FALSE",
    "snRNA-seq inferential tests performed: FALSE",
    "Candidate TFs changed: FALSE",
    "Validation hypotheses changed: FALSE",
    "",
    (
        "PHASE 10B4-A3 STATUS: "
        f"{status_value}"
    ),
]

(
    out
    / "phase10B4_A3_report.txt"
).write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

print(
    "\n".join(report)
)

print(
    "\n===== TOP PRIMARY RESULTS ====="
)

for row in primary_results[:15]:

    print(
        "\t".join(
            str(
                row[column]
            )
            for column in primary_columns
        )
    )

checksum_rows: list[dict[str, Any]] = []

for path in sorted(
    out.rglob("*")
):

    if (
        path.is_file()
        and path.name
        != "phase10B4_A3_SHA256.tsv"
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
    out / "phase10B4_A3_SHA256.tsv",
    checksum_rows,
    [
        "sha256",
        "size_bytes",
        "project_relative_path",
    ],
)
