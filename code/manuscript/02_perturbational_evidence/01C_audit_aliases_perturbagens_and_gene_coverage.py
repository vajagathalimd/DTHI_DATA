#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


PROJECT = Path(
    "."
)

METADATA_DIR = (
    PROJECT
    / "01_raw_data/perturbational_validation/"
      "LINCS_L1000/GSE70138/metadata"
)

PERT_FILE = (
    METADATA_DIR
    / "GSE70138_Broad_LINCS_pert_info_2017-03-06.txt.gz"
)

SIG_FILE = (
    METADATA_DIR
    / "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz"
)

GENE_FILE = (
    METADATA_DIR
    / "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz"
)

COVERAGE_FILE = (
    PROJECT
    / "07_tables/main_tables/phase6/"
      "phase6E1B_LINCS_chemical_coverage_summary.tsv"
)

PERT_MATCH_FILE = (
    PROJECT
    / "07_tables/main_tables/phase6/"
      "phase6E1B_LINCS_perturbagen_matches.tsv"
)

MATRIX_INVENTORY_FILE = (
    PROJECT
    / "07_tables/main_tables/phase6/"
      "phase6E1B_LINCS_matrix_inventory.tsv"
)

MATURATION_FILE = (
    PROJECT
    / "03_processed_data/developmental_trajectory/phase5D/enrichment/"
      "phase5D4_maturation_high_increasing_genes.tsv"
)

FETAL_FILE = (
    PROJECT
    / "03_processed_data/developmental_trajectory/phase5D/enrichment/"
      "phase5D4_fetal_high_decreasing_genes.tsv"
)

CATEGORY_UNION_FILE = (
    PROJECT
    / "03_processed_data/perturbational_validation/phase6C/"
      "phase6C3C_mechanism_category_gene_unions.tsv.gz"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase6"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase6/"
      "phase6E1C_alias_perturbagen_gene_coverage.log"
)

ALIAS_OUTPUT = (
    TABLE_DIR
    / "phase6E1C_LINCS_unmatched_alias_candidates.tsv"
)

PERT_AUDIT_OUTPUT = (
    TABLE_DIR
    / "phase6E1C_LINCS_matched_perturbagen_ID_audit.tsv"
)

GENE_COVERAGE_OUTPUT = (
    TABLE_DIR
    / "phase6E1C_LINCS_program_and_mechanism_gene_coverage.tsv"
)

MATRIX_OUTPUT = (
    TABLE_DIR
    / "phase6E1C_Level5_matrix_candidate.tsv"
)

COMPLETION_OUTPUT = (
    TABLE_DIR
    / "phase6E1C_completion_summary.tsv"
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value.lower() in {
        "",
        "nan",
        "none",
        "na",
    }:
        return ""

    return value


def normalize_name(value: object) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        clean_text(value).lower(),
    )


def normalize_gene(value: object) -> str:
    value = clean_text(value).upper()

    if value in {
        "",
        "NA",
        "NAN",
        "NONE",
        "-",
    }:
        return ""

    return value


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
            "Could not identify column from: "
            + ", ".join(candidates)
        )

    return None


def read_gene_set(path: Path) -> set[str]:
    table = pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
    )

    gene_column = find_column(
        table,
        [
            "gene_symbol",
            "gene",
            "symbol",
            "external_gene_name",
        ],
        required=False,
    )

    if gene_column is None:
        gene_column = table.columns[0]

    return {
        normalize_gene(value)
        for value in table[gene_column]
        if normalize_gene(value)
    }


def truthy_landmark(series: pd.Series) -> pd.Series:
    normalized = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {
            "1",
            "true",
            "t",
            "yes",
            "y",
        }
    )


def join_unique(
    values: pd.Series,
    maximum: int | None = None,
) -> str:
    unique_values = sorted(
        {
            clean_text(value)
            for value in values
            if clean_text(value)
        }
    )

    if maximum is not None:
        unique_values = unique_values[:maximum]

    return "|".join(unique_values)


def combine_labels(
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

    return table[available].apply(
        lambda row: "|".join(
            clean_text(row[column])
            for column in available
            if clean_text(row[column])
        ),
        axis=1,
    )


def main() -> None:
    required_files = [
        PERT_FILE,
        SIG_FILE,
        GENE_FILE,
        COVERAGE_FILE,
        PERT_MATCH_FILE,
        MATRIX_INVENTORY_FILE,
        MATURATION_FILE,
        FETAL_FILE,
        CATEGORY_UNION_FILE,
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(missing)
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

    gene_info = pd.read_csv(
        GENE_FILE,
        sep="\t",
        low_memory=False,
    )

    coverage = pd.read_csv(
        COVERAGE_FILE,
        sep="\t",
        low_memory=False,
    )

    pert_matches = pd.read_csv(
        PERT_MATCH_FILE,
        sep="\t",
        low_memory=False,
    )

    matrix_inventory = pd.read_csv(
        MATRIX_INVENTORY_FILE,
        sep="\t",
        low_memory=False,
    )

    category_unions = pd.read_csv(
        CATEGORY_UNION_FILE,
        sep="\t",
        low_memory=False,
    )

    if len(coverage) != 29:
        raise RuntimeError(
            f"Expected 29 chemicals, found {len(coverage)}."
        )

    if len(pert) != 2170:
        raise RuntimeError(
            f"Expected 2170 perturbagens, found {len(pert)}."
        )

    if len(signatures) != 118050:
        raise RuntimeError(
            f"Expected 118050 signatures, found {len(signatures)}."
        )

    if len(gene_info) != 12328:
        raise RuntimeError(
            f"Expected 12328 genes, found {len(gene_info)}."
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
        ],
    )

    sig_pert_id_column = find_column(
        signatures,
        [
            "pert_id",
            "perturbagen_id",
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

    gene_symbol_column = find_column(
        gene_info,
        [
            "pr_gene_symbol",
            "gene_symbol",
            "symbol",
        ],
    )

    landmark_column = find_column(
        gene_info,
        [
            "pr_is_lm",
            "is_landmark",
            "landmark",
        ],
        required=False,
    )

    signatures = signatures.copy()

    signatures["_dose_label"] = combine_labels(
        signatures,
        [
            "pert_idose",
            "pert_dose",
            "pert_dose_unit",
        ],
    )

    signatures["_time_label"] = combine_labels(
        signatures,
        [
            "pert_itime",
            "pert_time",
            "pert_time_unit",
        ],
    )

    # ------------------------------------------------------------
    # A. Matched perturbagen-ID audit
    # ------------------------------------------------------------

    pert_audit_rows: list[dict[str, object]] = []

    optional_pert_columns = [
        column
        for column in [
            "pert_type",
            "is_touchstone",
            "inchi_key",
            "inchi_key_prefix",
            "canonical_smiles",
            "pubchem_cid",
        ]
        if column in pert.columns
    ]

    for _, match in pert_matches.iterrows():
        chemical_query = clean_text(
            match["chemical_query"]
        )

        dtxsid = clean_text(
            match.get(
                "DTXSID",
                "",
            )
        )

        lincs_pert_id = clean_text(
            match["LINCS_pert_id"]
        )

        lincs_pert_name = clean_text(
            match["LINCS_pert_iname"]
        )

        pert_rows = pert.loc[
            pert[pert_id_column]
            .astype(str)
            .eq(lincs_pert_id)
        ]

        signature_rows = signatures.loc[
            signatures[sig_pert_id_column]
            .astype(str)
            .eq(lincs_pert_id)
        ]

        output_row: dict[str, object] = {
            "chemical_query":
                chemical_query,

            "DTXSID":
                dtxsid,

            "LINCS_pert_id":
                lincs_pert_id,

            "LINCS_pert_iname":
                lincs_pert_name,

            "pert_info_rows":
                len(pert_rows),

            "Level5_signature_count":
                len(signature_rows),

            "unique_cell_count":
                (
                    signature_rows[cell_column]
                    .nunique(dropna=True)
                    if cell_column
                    else 0
                ),

            "cell_ids":
                (
                    join_unique(
                        signature_rows[cell_column],
                        maximum=100,
                    )
                    if cell_column
                    else ""
                ),

            "dose_labels":
                join_unique(
                    signature_rows["_dose_label"],
                    maximum=100,
                ),

            "time_labels":
                join_unique(
                    signature_rows["_time_label"],
                    maximum=100,
                ),
        }

        for column in optional_pert_columns:
            output_row[column] = join_unique(
                pert_rows[column]
            )

        pert_audit_rows.append(
            output_row
        )

    perturbagen_audit = pd.DataFrame(
        pert_audit_rows
    ).sort_values(
        [
            "chemical_query",
            "LINCS_pert_id",
        ]
    )

    perturbagen_audit.to_csv(
        PERT_AUDIT_OUTPUT,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------
    # B. Conservative unmatched alias search
    # ------------------------------------------------------------

    alias_map = {
        "0175029-0000": [
            "0175029-0000",
            "0175029",
        ],

        "GW-8510": [
            "GW-8510",
            "GW8510",
        ],

        "H-7": [
            "H-7",
            "H7",
        ],

        "HC toxin": [
            "HC toxin",
            "HC-toxin",
            "helminthosporium carbonum toxin",
        ],

        "SC-58125": [
            "SC-58125",
            "SC58125",
        ],

        "alsterpaullone": [
            "alsterpaullone",
        ],

        "ellipticine": [
            "ellipticine",
        ],

        "fenoprofen": [
            "fenoprofen",
            "fenoprofen calcium",
        ],

        "hexetidine": [
            "hexetidine",
        ],

        "methotrexate": [
            "methotrexate",
            "amethopterin",
        ],

        "monobenzone": [
            "monobenzone",
            "4-benzyloxyphenol",
            "hydroquinone monobenzyl ether",
        ],

        "natamycin": [
            "natamycin",
            "pimaricin",
        ],

        "scriptaid": [
            "scriptaid",
        ],

        "suloctidil": [
            "suloctidil",
        ],
    }

    string_columns = [
        column
        for column in pert.columns
        if (
            pd.api.types.is_object_dtype(
                pert[column]
            )
            or pd.api.types.is_string_dtype(
                pert[column]
            )
        )
    ]

    normalized_columns = {
        column: pert[column].map(
            normalize_name
        )
        for column in string_columns
    }

    alias_rows: list[dict[str, object]] = []

    unmatched = coverage.loc[
        coverage["match_status"].eq(
            "unmatched"
        )
    ]

    for _, chemical in unmatched.iterrows():
        chemical_query = clean_text(
            chemical["chemical_query"]
        )

        dtxsid = clean_text(
            chemical.get(
                "DTXSID",
                "",
            )
        )

        aliases = alias_map.get(
            chemical_query,
            [chemical_query],
        )

        candidate_indices: set[int] = set()

        for alias in aliases:
            alias_normalized = normalize_name(
                alias
            )

            for column in string_columns:
                exact_mask = normalized_columns[
                    column
                ].eq(
                    alias_normalized
                )

                for index in pert.index[
                    exact_mask
                ]:
                    candidate_indices.add(
                        int(index)
                    )

                    alias_rows.append(
                        {
                            "chemical_query":
                                chemical_query,

                            "DTXSID":
                                dtxsid,

                            "alias_tested":
                                alias,

                            "match_mode":
                                "exact_normalized",

                            "matched_column":
                                column,

                            "matched_value":
                                clean_text(
                                    pert.loc[
                                        index,
                                        column,
                                    ]
                                ),

                            "LINCS_pert_id":
                                clean_text(
                                    pert.loc[
                                        index,
                                        pert_id_column,
                                    ]
                                ),

                            "LINCS_pert_iname":
                                clean_text(
                                    pert.loc[
                                        index,
                                        pert_name_column,
                                    ]
                                ),
                        }
                    )

        if not candidate_indices:
            alias_rows.append(
                {
                    "chemical_query":
                        chemical_query,

                    "DTXSID":
                        dtxsid,

                    "alias_tested":
                        "|".join(aliases),

                    "match_mode":
                        "no_exact_alias_candidate",

                    "matched_column":
                        "",

                    "matched_value":
                        "",

                    "LINCS_pert_id":
                        "",

                    "LINCS_pert_iname":
                        "",
                }
            )

    alias_candidates = pd.DataFrame(
        alias_rows
    ).drop_duplicates()

    alias_candidates = alias_candidates.sort_values(
        [
            "chemical_query",
            "match_mode",
            "LINCS_pert_iname",
            "LINCS_pert_id",
        ]
    )

    alias_candidates.to_csv(
        ALIAS_OUTPUT,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------
    # C. Developmental-program and mechanism-panel gene coverage
    # ------------------------------------------------------------

    lincs_genes = {
        normalize_gene(value)
        for value in gene_info[
            gene_symbol_column
        ]
        if normalize_gene(value)
    }

    if landmark_column:
        landmark_mask = truthy_landmark(
            gene_info[landmark_column]
        )

        landmark_genes = {
            normalize_gene(value)
            for value in gene_info.loc[
                landmark_mask,
                gene_symbol_column,
            ]
            if normalize_gene(value)
        }

    else:
        landmark_genes = set()

    maturation_genes = read_gene_set(
        MATURATION_FILE
    )

    fetal_genes = read_gene_set(
        FETAL_FILE
    )

    category_gene_column = find_column(
        category_unions,
        [
            "gene_symbol",
            "gene",
            "symbol",
        ],
    )

    category_name_column = find_column(
        category_unions,
        [
            "mechanism_category",
        ],
    )

    axis_role_column = find_column(
        category_unions,
        [
            "axis_role",
        ],
        required=False,
    )

    gene_set_rows: list[dict[str, object]] = []

    def add_gene_set(
        source_type: str,
        gene_set_name: str,
        axis_role: str,
        genes: set[str],
    ) -> None:
        mapped = genes & lincs_genes
        landmark_mapped = genes & landmark_genes

        gene_set_rows.append(
            {
                "source_type":
                    source_type,

                "gene_set_name":
                    gene_set_name,

                "axis_role":
                    axis_role,

                "input_gene_count":
                    len(genes),

                "LINCS_gene_count":
                    len(mapped),

                "LINCS_gene_coverage_fraction":
                    (
                        len(mapped) / len(genes)
                        if genes
                        else np.nan
                    ),

                "LINCS_landmark_gene_count":
                    len(landmark_mapped),

                "LINCS_landmark_coverage_fraction":
                    (
                        len(landmark_mapped)
                        / len(genes)
                        if genes
                        else np.nan
                    ),

                "LINCS_inferred_or_landmark_count":
                    len(mapped),

                "genes_absent_from_LINCS":
                    len(
                        genes - lincs_genes
                    ),
            }
        )

    add_gene_set(
        "developmental_program",
        "maturation_high_increasing",
        "late_maturation",
        maturation_genes,
    )

    add_gene_set(
        "developmental_program",
        "fetal_high_decreasing",
        "early_fetal",
        fetal_genes,
    )

    for category, group in category_unions.groupby(
        category_name_column,
        sort=True,
    ):
        genes = {
            normalize_gene(value)
            for value in group[
                category_gene_column
            ]
            if normalize_gene(value)
        }

        axis_role = (
            clean_text(
                group[
                    axis_role_column
                ].iloc[0]
            )
            if axis_role_column
            else ""
        )

        add_gene_set(
            "mechanism_category",
            clean_text(category),
            axis_role,
            genes,
        )

    gene_coverage = pd.DataFrame(
        gene_set_rows
    )

    gene_coverage.to_csv(
        GENE_COVERAGE_OUTPUT,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------
    # D. Matrix inventory
    # ------------------------------------------------------------

    if "is_Level5_matrix" in matrix_inventory.columns:
        level5_mask = (
            matrix_inventory[
                "is_Level5_matrix"
            ]
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin(
                {
                    "true",
                    "1",
                    "yes",
                }
            )
        )

    else:
        level5_mask = pd.Series(
            False,
            index=matrix_inventory.index,
        )

    matrix_candidate = matrix_inventory.loc[
        level5_mask
    ].copy()

    matrix_candidate.to_csv(
        MATRIX_OUTPUT,
        sep="\t",
        index=False,
    )

    exact_alias_chemicals = alias_candidates.loc[
        alias_candidates[
            "match_mode"
        ].eq(
            "exact_normalized"
        ),
        "chemical_query",
    ].nunique()

    completion = pd.DataFrame(
        [
            {
                "chemical_entities":
                    len(coverage),

                "exact_matched_chemicals_from_Phase6E1B":
                    int(
                        coverage[
                            "match_status"
                        ].eq(
                            "matched"
                        ).sum()
                    ),

                "unmatched_chemicals_audited":
                    len(unmatched),

                "unmatched_chemicals_with_exact_alias_candidate":
                    exact_alias_chemicals,

                "matched_perturbagen_IDs_audited":
                    pert_matches[
                        "LINCS_pert_id"
                    ].nunique(),

                "LINCS_gene_rows":
                    len(gene_info),

                "unique_LINCS_gene_symbols":
                    len(lincs_genes),

                "LINCS_landmark_gene_symbols":
                    len(landmark_genes),

                "developmental_programs_audited":
                    2,

                "mechanism_categories_audited":
                    category_unions[
                        category_name_column
                    ].nunique(),

                "Level5_matrix_candidates":
                    len(matrix_candidate),

                "large_expression_matrix_downloaded":
                    False,

                "Phase6E1C_status":
                    "completed",
            }
        ]
    )

    completion.to_csv(
        COMPLETION_OUTPUT,
        sep="\t",
        index=False,
    )

    log_text = "\n".join(
        [
            "===== PHASE 6E1C COMPLETION =====",
            completion.to_string(
                index=False
            ),
            "",
            "===== PROGRAM AND MECHANISM GENE COVERAGE =====",
            gene_coverage.to_string(
                index=False
            ),
            "",
            "===== UNMATCHED ALIAS CANDIDATES =====",
            alias_candidates.to_string(
                index=False
            ),
            "",
            "===== MATCHED PERTURBAGEN-ID AUDIT =====",
            perturbagen_audit.to_string(
                index=False
            ),
            "",
            "===== LEVEL 5 MATRIX CANDIDATE =====",
            (
                matrix_candidate.to_string(
                    index=False
                )
                if not matrix_candidate.empty
                else "No candidate detected."
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
                "Phase 6E1C failed: "
                f"{type(error).__name__}: "
                f"{error}"
            ),
            file=sys.stderr,
        )

        raise
