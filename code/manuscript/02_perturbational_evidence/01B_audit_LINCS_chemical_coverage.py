#!/usr/bin/env python3

from __future__ import annotations

from difflib import SequenceMatcher
import gzip
from pathlib import Path
import re
import sys

import pandas as pd


PROJECT = Path(
    "."
)

METADATA_DIR = (
    PROJECT
    / "01_raw_data/perturbational_validation/"
      "LINCS_L1000/GSE70138/metadata"
)

CHEMICAL_FILE = (
    PROJECT
    / "07_tables/main_tables/phase6/"
      "phase6D3_chemical_evidence_tiers.tsv"
)

PERT_FILE = (
    METADATA_DIR
    / "GSE70138_Broad_LINCS_pert_info_2017-03-06.txt.gz"
)

SIG_FILE = (
    METADATA_DIR
    / "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz"
)

METRIC_FILE = (
    METADATA_DIR
    / "GSE70138_Broad_LINCS_sig_metrics_2017-03-06.txt.gz"
)

CELL_FILE = (
    METADATA_DIR
    / "GSE70138_Broad_LINCS_cell_info_2017-04-28.txt.gz"
)

GENE_FILE = (
    METADATA_DIR
    / "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz"
)

CHECKSUM_FILE = (
    METADATA_DIR
    / "GSE70138_SHA512SUMS.txt.gz"
)

PROCESSED_DIR = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6E"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase6"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase6/"
      "phase6E1B_LINCS_chemical_coverage.log"
)

SCHEMA_OUTPUT = (
    TABLE_DIR
    / "phase6E1B_LINCS_metadata_schema.tsv"
)

COVERAGE_OUTPUT = (
    TABLE_DIR
    / "phase6E1B_LINCS_chemical_coverage_summary.tsv"
)

PERT_MATCH_OUTPUT = (
    TABLE_DIR
    / "phase6E1B_LINCS_perturbagen_matches.tsv"
)

FUZZY_OUTPUT = (
    TABLE_DIR
    / "phase6E1B_LINCS_unmatched_candidate_matches.tsv"
)

SIGNATURE_OUTPUT = (
    PROCESSED_DIR
    / "phase6E1B_LINCS_matched_signature_metadata.tsv.gz"
)

QC_OUTPUT = (
    TABLE_DIR
    / "phase6E1B_LINCS_quality_metric_columns.tsv"
)

MATRIX_OUTPUT = (
    TABLE_DIR
    / "phase6E1B_LINCS_matrix_inventory.tsv"
)

COMPLETION_OUTPUT = (
    TABLE_DIR
    / "phase6E1B_completion_summary.tsv"
)

for directory in [
    PROCESSED_DIR,
    TABLE_DIR,
    LOG_FILE.parent,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    if text in {
        "",
        "nan",
        "na",
        "none",
    }:
        return ""

    return re.sub(
        r"[^a-z0-9]+",
        "",
        text,
    )


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "na",
        "none",
    }:
        return ""

    return text


def find_column(
    table: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    lookup = {
        str(column).lower(): column
        for column in table.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    if required:
        raise RuntimeError(
            "Missing expected column. Candidates: "
            + ", ".join(candidates)
        )

    return None


def join_unique(
    values: pd.Series,
    maximum: int | None = None,
) -> str:
    cleaned = sorted(
        {
            clean_text(value)
            for value in values
            if clean_text(value)
        }
    )

    if maximum is not None:
        cleaned = cleaned[:maximum]

    return "|".join(cleaned)


def count_unique(values: pd.Series) -> int:
    return len(
        {
            clean_text(value)
            for value in values
            if clean_text(value)
        }
    )


def combine_columns(
    table: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    available = [
        column
        for column in columns
        if column in table.columns
    ]

    if not available:
        return pd.Series(
            "",
            index=table.index,
            dtype=str,
        )

    def combine_row(row: pd.Series) -> str:
        values = [
            clean_text(row[column])
            for column in available
            if clean_text(row[column])
        ]

        return "|".join(values)

    return table[available].apply(
        combine_row,
        axis=1,
    )


def parse_checksum_manifest(
    path: Path,
) -> pd.DataFrame:
    columns = [
        "SHA512",
        "filename",
        "is_expression_matrix",
        "is_Level5_matrix",
    ]

    if not path.exists():
        return pd.DataFrame(
            columns=columns
        )

    rows: list[dict[str, object]] = []

    with gzip.open(
        path,
        mode="rt",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            fields = line.split(
                maxsplit=1
            )

            if len(fields) != 2:
                continue

            checksum, filename = fields
            filename = filename.lstrip("*")
            lowered = filename.lower()

            expression_matrix = bool(
                re.search(
                    r"\.(gctx|gct)(\.gz)?$",
                    lowered,
                )
            )

            level5_matrix = bool(
                expression_matrix
                and (
                    "level5" in lowered
                    or "level_5" in lowered
                    or "level-5" in lowered
                    or "compz" in lowered
                )
            )

            rows.append(
                {
                    "SHA512":
                        checksum,

                    "filename":
                        filename,

                    "is_expression_matrix":
                        expression_matrix,

                    "is_Level5_matrix":
                        level5_matrix,
                }
            )

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def main() -> None:
    required_files = [
        CHEMICAL_FILE,
        PERT_FILE,
        SIG_FILE,
        METRIC_FILE,
        CELL_FILE,
        GENE_FILE,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required input files:\n"
            + "\n".join(missing_files)
        )

    chemicals = pd.read_csv(
        CHEMICAL_FILE,
        sep="\t",
        low_memory=False,
    )

    pert = pd.read_csv(
        PERT_FILE,
        sep="\t",
        low_memory=False,
    )

    signatures = pd.read_csv(
        SIG_FILE,
        sep="\t",
        low_memory=False,
    )

    metrics = pd.read_csv(
        METRIC_FILE,
        sep="\t",
        low_memory=False,
    )

    cells = pd.read_csv(
        CELL_FILE,
        sep="\t",
        low_memory=False,
    )

    genes = pd.read_csv(
        GENE_FILE,
        sep="\t",
        low_memory=False,
    )

    expected_dimensions = {
        "chemical_entities":
            29,

        "perturbagens":
            2170,

        "signatures":
            118050,

        "signature_metrics":
            118050,

        "cells":
            98,

        "genes":
            12328,
    }

    observed_dimensions = {
        "chemical_entities":
            len(chemicals),

        "perturbagens":
            len(pert),

        "signatures":
            len(signatures),

        "signature_metrics":
            len(metrics),

        "cells":
            len(cells),

        "genes":
            len(genes),
    }

    for name, expected in expected_dimensions.items():
        observed = observed_dimensions[name]

        if observed != expected:
            raise RuntimeError(
                f"{name}: expected {expected}, found {observed}"
            )

    schema_rows = []

    for table_name, table, path in [
        ("pert_info", pert, PERT_FILE),
        ("sig_info", signatures, SIG_FILE),
        ("sig_metrics", metrics, METRIC_FILE),
        ("cell_info", cells, CELL_FILE),
        ("gene_info", genes, GENE_FILE),
    ]:
        schema_rows.append(
            {
                "table_name":
                    table_name,

                "filename":
                    path.name,

                "rows":
                    len(table),

                "columns":
                    len(table.columns),

                "column_names":
                    "|".join(
                        map(str, table.columns)
                    ),
            }
        )

    pd.DataFrame(
        schema_rows
    ).to_csv(
        SCHEMA_OUTPUT,
        sep="\t",
        index=False,
    )

    chemical_name_column = find_column(
        chemicals,
        [
            "chemical_name",
            "chemical_display",
        ],
    )

    dtxsid_column = find_column(
        chemicals,
        ["DTXSID"],
        required=False,
    )

    tier_column = find_column(
        chemicals,
        ["evidence_tier"],
        required=False,
    )

    pert_id_column = find_column(
        pert,
        [
            "pert_id",
            "perturbagen_id",
        ],
    )

    pert_name_column = find_column(
        pert,
        [
            "pert_iname",
            "pert_name",
            "compound_name",
            "cmap_name",
        ],
    )

    sig_id_column = find_column(
        signatures,
        [
            "sig_id",
            "signature_id",
        ],
    )

    sig_pert_id_column = find_column(
        signatures,
        [
            "pert_id",
            "perturbagen_id",
        ],
    )

    sig_pert_name_column = find_column(
        signatures,
        [
            "pert_iname",
            "pert_name",
            "compound_name",
        ],
        required=False,
    )

    metric_sig_id_column = find_column(
        metrics,
        [
            "sig_id",
            "signature_id",
        ],
    )

    cell_column = find_column(
        signatures,
        [
            "cell_id",
            "cell",
        ],
        required=False,
    )

    perturbation_type_column = find_column(
        signatures,
        [
            "pert_type",
            "perturbation_type",
        ],
        required=False,
    )

    if signatures[
        sig_id_column
    ].duplicated().any():
        raise RuntimeError(
            "sig_info contains duplicated signature identifiers."
        )

    if metrics[
        metric_sig_id_column
    ].duplicated().any():
        raise RuntimeError(
            "sig_metrics contains duplicated signature identifiers."
        )

    pert = pert.copy()

    pert[
        "_normalized_name"
    ] = pert[
        pert_name_column
    ].map(
        normalize_name
    )

    signatures = signatures.copy()

    signatures[
        "_dose_label"
    ] = combine_columns(
        signatures,
        [
            "pert_idose",
            "pert_dose",
            "pert_dose_unit",
        ],
    )

    signatures[
        "_time_label"
    ] = combine_columns(
        signatures,
        [
            "pert_itime",
            "pert_time",
            "pert_time_unit",
        ],
    )

    if sig_pert_name_column:
        signatures[
            "_normalized_pert_name"
        ] = signatures[
            sig_pert_name_column
        ].map(
            normalize_name
        )

    metric_columns = [
        column
        for column in metrics.columns
        if column != metric_sig_id_column
    ]

    qc_rows = []

    for column in metric_columns:
        qc_rows.append(
            {
                "metric_column":
                    column,

                "dtype":
                    str(metrics[column].dtype),

                "nonmissing_values":
                    int(
                        metrics[column]
                        .notna()
                        .sum()
                    ),
            }
        )

    pd.DataFrame(
        qc_rows
    ).to_csv(
        QC_OUTPUT,
        sep="\t",
        index=False,
    )

    metric_subset = metrics.rename(
        columns={
            metric_sig_id_column:
                sig_id_column
        }
    )

    candidate_names = (
        pert[
            [
                pert_id_column,
                pert_name_column,
                "_normalized_name",
            ]
        ]
        .drop_duplicates()
        .loc[
            lambda frame:
                frame[
                    "_normalized_name"
                ].ne("")
        ]
        .copy()
    )

    match_rows: list[dict[str, object]] = []
    fuzzy_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    matched_signature_parts: list[pd.DataFrame] = []

    for _, chemical in chemicals.iterrows():
        query_name = clean_text(
            chemical[
                chemical_name_column
            ]
        )

        query_normalized = normalize_name(
            query_name
        )

        dtxsid = (
            clean_text(
                chemical[dtxsid_column]
            )
            if dtxsid_column
            else ""
        )

        evidence_tier = (
            clean_text(
                chemical[tier_column]
            )
            if tier_column
            else ""
        )

        exact = candidate_names.loc[
            candidate_names[
                "_normalized_name"
            ].eq(
                query_normalized
            )
        ].copy()

        matched_ids = sorted(
            {
                clean_text(value)
                for value in exact[
                    pert_id_column
                ]
                if clean_text(value)
            }
        )

        matched_names = sorted(
            {
                clean_text(value)
                for value in exact[
                    pert_name_column
                ]
                if clean_text(value)
            }
        )

        match_status = (
            "matched"
            if matched_ids
            else "unmatched"
        )

        match_method = (
            "normalized_exact"
            if matched_ids
            else ""
        )

        for _, matched in exact.iterrows():
            match_rows.append(
                {
                    "chemical_query":
                        query_name,

                    "DTXSID":
                        dtxsid,

                    "evidence_tier":
                        evidence_tier,

                    "match_method":
                        match_method,

                    "LINCS_pert_id":
                        clean_text(
                            matched[
                                pert_id_column
                            ]
                        ),

                    "LINCS_pert_iname":
                        clean_text(
                            matched[
                                pert_name_column
                            ]
                        ),
                }
            )

        if not matched_ids:
            unique_candidate_names = (
                candidate_names[
                    [
                        pert_name_column,
                        "_normalized_name",
                    ]
                ]
                .drop_duplicates()
            )

            scored_candidates = []

            for _, candidate in unique_candidate_names.iterrows():
                candidate_normalized = candidate[
                    "_normalized_name"
                ]

                similarity = SequenceMatcher(
                    None,
                    query_normalized,
                    candidate_normalized,
                ).ratio()

                scored_candidates.append(
                    (
                        similarity,
                        clean_text(
                            candidate[
                                pert_name_column
                            ]
                        ),
                        candidate_normalized,
                    )
                )

            scored_candidates.sort(
                key=lambda item: (
                    -item[0],
                    item[1].lower(),
                )
            )

            for rank, (
                similarity,
                candidate_name,
                candidate_normalized,
            ) in enumerate(
                scored_candidates[:5],
                start=1,
            ):
                candidate_ids = candidate_names.loc[
                    candidate_names[
                        "_normalized_name"
                    ].eq(
                        candidate_normalized
                    ),
                    pert_id_column,
                ]

                fuzzy_rows.append(
                    {
                        "chemical_query":
                            query_name,

                        "DTXSID":
                            dtxsid,

                        "evidence_tier":
                            evidence_tier,

                        "candidate_rank":
                            rank,

                        "candidate_similarity":
                            similarity,

                        "candidate_pert_iname":
                            candidate_name,

                        "candidate_pert_ids":
                            join_unique(
                                candidate_ids
                            ),
                    }
                )

        if matched_ids:
            signature_subset = signatures.loc[
                signatures[
                    sig_pert_id_column
                ].astype(str).isin(
                    matched_ids
                )
            ].copy()

            if (
                signature_subset.empty
                and sig_pert_name_column
            ):
                signature_subset = signatures.loc[
                    signatures[
                        "_normalized_pert_name"
                    ].eq(
                        query_normalized
                    )
                ].copy()

        else:
            signature_subset = signatures.iloc[
                0:0
            ].copy()

        if not signature_subset.empty:
            signature_subset[
                "chemical_query"
            ] = query_name

            signature_subset[
                "DTXSID_request"
            ] = dtxsid

            signature_subset[
                "evidence_tier_request"
            ] = evidence_tier

            signature_subset[
                "match_method"
            ] = match_method

            signature_subset = signature_subset.merge(
                metric_subset,
                on=sig_id_column,
                how="left",
                validate="one_to_one",
                suffixes=(
                    "",
                    "_metric",
                ),
                indicator="_metric_merge",
            )

            matched_signature_parts.append(
                signature_subset
            )

            metric_rows_found = int(
                signature_subset[
                    "_metric_merge"
                ].eq(
                    "both"
                ).sum()
            )

        else:
            metric_rows_found = 0

        coverage_rows.append(
            {
                "chemical_query":
                    query_name,

                "DTXSID":
                    dtxsid,

                "evidence_tier":
                    evidence_tier,

                "match_status":
                    match_status,

                "match_method":
                    match_method,

                "matched_perturbagen_names":
                    "|".join(
                        matched_names
                    ),

                "matched_pert_ids":
                    "|".join(
                        matched_ids
                    ),

                "matched_pert_id_count":
                    len(
                        matched_ids
                    ),

                "Level5_signature_count":
                    len(
                        signature_subset
                    ),

                "unique_cell_count":
                    (
                        count_unique(
                            signature_subset[
                                cell_column
                            ]
                        )
                        if (
                            cell_column
                            and not signature_subset.empty
                        )
                        else 0
                    ),

                "cell_ids":
                    (
                        join_unique(
                            signature_subset[
                                cell_column
                            ],
                            maximum=100,
                        )
                        if (
                            cell_column
                            and not signature_subset.empty
                        )
                        else ""
                    ),

                "unique_dose_label_count":
                    (
                        count_unique(
                            signature_subset[
                                "_dose_label"
                            ]
                        )
                        if not signature_subset.empty
                        else 0
                    ),

                "dose_labels":
                    (
                        join_unique(
                            signature_subset[
                                "_dose_label"
                            ],
                            maximum=100,
                        )
                        if not signature_subset.empty
                        else ""
                    ),

                "unique_time_label_count":
                    (
                        count_unique(
                            signature_subset[
                                "_time_label"
                            ]
                        )
                        if not signature_subset.empty
                        else 0
                    ),

                "time_labels":
                    (
                        join_unique(
                            signature_subset[
                                "_time_label"
                            ],
                            maximum=100,
                        )
                        if not signature_subset.empty
                        else ""
                    ),

                "perturbation_types":
                    (
                        join_unique(
                            signature_subset[
                                perturbation_type_column
                            ],
                            maximum=20,
                        )
                        if (
                            perturbation_type_column
                            and not signature_subset.empty
                        )
                        else ""
                    ),

                "signature_metrics_found":
                    metric_rows_found,
            }
        )

    coverage = pd.DataFrame(
        coverage_rows
    ).sort_values(
        [
            "match_status",
            "Level5_signature_count",
            "chemical_query",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )

    perturbagen_matches = pd.DataFrame(
        match_rows,
        columns=[
            "chemical_query",
            "DTXSID",
            "evidence_tier",
            "match_method",
            "LINCS_pert_id",
            "LINCS_pert_iname",
        ],
    )

    fuzzy_candidates = pd.DataFrame(
        fuzzy_rows,
        columns=[
            "chemical_query",
            "DTXSID",
            "evidence_tier",
            "candidate_rank",
            "candidate_similarity",
            "candidate_pert_iname",
            "candidate_pert_ids",
        ],
    )

    coverage.to_csv(
        COVERAGE_OUTPUT,
        sep="\t",
        index=False,
    )

    perturbagen_matches.to_csv(
        PERT_MATCH_OUTPUT,
        sep="\t",
        index=False,
    )

    fuzzy_candidates.to_csv(
        FUZZY_OUTPUT,
        sep="\t",
        index=False,
    )

    if matched_signature_parts:
        matched_signatures = pd.concat(
            matched_signature_parts,
            ignore_index=True,
        )
    else:
        matched_signatures = pd.DataFrame()

    matched_signatures.to_csv(
        SIGNATURE_OUTPUT,
        sep="\t",
        index=False,
        compression="gzip",
    )

    matrix_inventory = parse_checksum_manifest(
        CHECKSUM_FILE
    )

    matrix_inventory.to_csv(
        MATRIX_OUTPUT,
        sep="\t",
        index=False,
    )

    matched_chemical_count = int(
        coverage[
            "match_status"
        ].eq(
            "matched"
        ).sum()
    )

    chemicals_with_signatures = int(
        coverage[
            "Level5_signature_count"
        ].gt(0).sum()
    )

    tier_a_mask = coverage[
        "evidence_tier"
    ].eq(
        "Tier_A_global_residual_mechanism"
    )

    completion = pd.DataFrame(
        [
            {
                "Phase6D3_chemical_entities":
                    len(chemicals),

                "LINCS_perturbagens":
                    len(pert),

                "LINCS_Level5_signature_metadata_rows":
                    len(signatures),

                "LINCS_signature_metric_rows":
                    len(metrics),

                "normalized_exact_matched_chemicals":
                    matched_chemical_count,

                "unmatched_chemicals":
                    len(chemicals)
                    - matched_chemical_count,

                "chemicals_with_Level5_signatures":
                    chemicals_with_signatures,

                "matched_Level5_signatures":
                    len(matched_signatures),

                "matched_unique_cells":
                    (
                        count_unique(
                            matched_signatures[
                                cell_column
                            ]
                        )
                        if (
                            cell_column
                            and not matched_signatures.empty
                        )
                        else 0
                    ),

                "Tier_A_chemicals_total":
                    int(
                        tier_a_mask.sum()
                    ),

                "Tier_A_chemicals_with_Level5_signatures":
                    int(
                        (
                            tier_a_mask
                            & coverage[
                                "Level5_signature_count"
                            ].gt(0)
                        ).sum()
                    ),

                "quality_metric_columns_detected":
                    len(metric_columns),

                "Level5_matrix_candidates_in_checksum_manifest":
                    (
                        int(
                            matrix_inventory[
                                "is_Level5_matrix"
                            ].sum()
                        )
                        if not matrix_inventory.empty
                        else 0
                    ),

                "large_expression_matrix_downloaded":
                    False,

                "Phase6E1B_status":
                    "completed",
            }
        ]
    )

    completion.to_csv(
        COMPLETION_OUTPUT,
        sep="\t",
        index=False,
    )

    unmatched = coverage.loc[
        coverage[
            "match_status"
        ].eq(
            "unmatched"
        ),
        [
            "chemical_query",
            "DTXSID",
            "evidence_tier",
        ],
    ]

    level5_candidates = (
        matrix_inventory.loc[
            matrix_inventory[
                "is_Level5_matrix"
            ]
        ]
        if not matrix_inventory.empty
        else matrix_inventory
    )

    log_text = "\n".join(
        [
            "===== PHASE 6E1B COMPLETION =====",
            completion.to_string(
                index=False
            ),
            "",
            "===== CHEMICAL COVERAGE SUMMARY =====",
            coverage[
                [
                    "chemical_query",
                    "DTXSID",
                    "evidence_tier",
                    "match_status",
                    "matched_perturbagen_names",
                    "matched_pert_id_count",
                    "Level5_signature_count",
                    "unique_cell_count",
                    "unique_dose_label_count",
                    "unique_time_label_count",
                    "signature_metrics_found",
                ]
            ].to_string(
                index=False
            ),
            "",
            "===== UNMATCHED CHEMICALS =====",
            (
                unmatched.to_string(
                    index=False
                )
                if not unmatched.empty
                else "None"
            ),
            "",
            "===== LEVEL 5 MATRIX CANDIDATES =====",
            (
                level5_candidates.to_string(
                    index=False
                )
                if not level5_candidates.empty
                else "No Level 5 candidate detected."
            ),
        ]
    )

    LOG_FILE.write_text(
        log_text + "\n",
        encoding="utf-8",
    )

    print(
        log_text
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            (
                "Phase 6E1B failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
