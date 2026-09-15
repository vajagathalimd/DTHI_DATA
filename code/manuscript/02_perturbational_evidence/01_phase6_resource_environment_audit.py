#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import csv
import gzip
import json
import os
import shutil
import subprocess
import sys

import pandas as pd


PROJECT = Path(
    "."
)

MATURATION_FILE = (
    PROJECT
    / "03_processed_data/developmental_trajectory/phase5D/"
      "enrichment/phase5D4_maturation_high_increasing_genes.tsv"
)

FETAL_FILE = (
    PROJECT
    / "03_processed_data/developmental_trajectory/phase5D/"
      "enrichment/phase5D4_fetal_high_decreasing_genes.tsv"
)

SEARCH_ROOTS = [
    PROJECT / "01_raw_data",
    PROJECT / "02_metadata",
    PROJECT / "03_processed_data",
    PROJECT / "07_tables",
]

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase6"
)

METADATA_DIR = (
    PROJECT
    / "02_metadata/phase6"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase6/"
      "phase6A_resource_environment_audit.log"
)

for directory in [
    TABLE_DIR,
    METADATA_DIR,
    LOG_FILE.parent,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


KEYWORDS = [
    "httr",
    "high_throughput_transcript",
    "high-throughput_transcript",
    "temposeq",
    "tempo_seq",
    "toxcast",
    "comptox",
    "lincs",
    "l1000",
    "cmap",
    "connectivity_map",
    "perturb",
    "signature",
    "sci_plex",
    "sciplex",
]

VALID_SUFFIXES = {
    ".tsv",
    ".csv",
    ".txt",
    ".gz",
    ".zip",
    ".tar",
    ".tgz",
    ".gctx",
    ".gct",
    ".h5",
    ".hdf5",
    ".h5ad",
    ".rds",
    ".rdata",
    ".xlsx",
    ".json",
    ".parquet",
    ".feather",
}

PYTHON_CANDIDATES = [
    Path(sys.executable),
    Path("/usr/bin/python3"),
    Path("/usr/local/bin/python"),
    Path("/root/miniconda/bin/python"),
    Path("/root/miniconda/envs/dthi_spatial/bin/python"),
]

PYTHON_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "h5py",
    "pyarrow",
    "anndata",
    "scanpy",
    "cmapPy",
    "gseapy",
    "statsmodels",
    "sklearn",
]

SOURCE_STRATEGY = [
    {
        "source_layer": "EPA_HTTr",
        "analysis_role": "primary_chemical_hazard_layer",
        "priority": 1,
        "planned_content": (
            "chemical identifiers, cell context, concentration, "
            "time, active signatures, BMD values and available "
            "gene/pathway response information"
        ),
        "planned_analysis": (
            "chemical concordance with fetal-high and "
            "maturation-high programs; potency-aware ranking"
        ),
        "status": "planned",
    },
    {
        "source_layer": "LINCS_L1000_Level5",
        "analysis_role": "secondary_perturbational_validation",
        "priority": 2,
        "planned_content": (
            "compound and genetic perturbation Level 5 signatures, "
            "signature metadata and perturbagen annotations"
        ),
        "planned_analysis": (
            "program induction, reversal, dose/time/cell-context "
            "consistency and mechanism-of-action aggregation"
        ),
        "status": "planned",
    },
    {
        "source_layer": "GEO_neural_validation",
        "analysis_role": "neurodevelopmental_context_validation",
        "priority": 3,
        "planned_content": (
            "selected neural, glial, organoid or developmental "
            "chemical perturbation transcriptomes"
        ),
        "planned_analysis": (
            "independent validation of prioritized chemical and "
            "mechanistic signatures"
        ),
        "status": "planned",
    },
]


def normalize_gene(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().upper()

    if text in {
        "",
        "NA",
        "NAN",
        "NONE",
        "<NA>",
    }:
        return ""

    return text


def read_program_genes(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing program-gene file: {path}"
        )

    table = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        low_memory=False,
    )

    candidate_columns = [
        "gene_symbol",
        "gene",
        "symbol",
    ]

    gene_column = next(
        (
            column
            for column in candidate_columns
            if column in table.columns
        ),
        table.columns[0],
    )

    genes = {
        normalize_gene(value)
        for value in table[gene_column]
    }

    return {
        gene
        for gene in genes
        if gene
    }


def file_matches(path: Path) -> bool:
    path_text = str(
        path.relative_to(PROJECT)
    ).lower()

    return any(
        keyword in path_text
        for keyword in KEYWORDS
    )


def inventory_local_assets() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            suffixes = {
                suffix.lower()
                for suffix in path.suffixes
            }

            if (
                not suffixes.intersection(
                    VALID_SUFFIXES
                )
                and not file_matches(path)
            ):
                continue

            if not file_matches(path):
                continue

            matched_keywords = [
                keyword
                for keyword in KEYWORDS
                if keyword
                in str(
                    path.relative_to(PROJECT)
                ).lower()
            ]

            rows.append(
                {
                    "relative_path":
                        str(
                            path.relative_to(
                                PROJECT
                            )
                        ),

                    "filename":
                        path.name,

                    "size_bytes":
                        path.stat().st_size,

                    "size_megabytes":
                        path.stat().st_size
                        / 1024
                        / 1024,

                    "matched_keywords":
                        "|".join(
                            matched_keywords
                        ),

                    "file_suffixes":
                        "|".join(
                            path.suffixes
                        ),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "relative_path",
                "filename",
                "size_bytes",
                "size_megabytes",
                "matched_keywords",
                "file_suffixes",
            ]
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "size_bytes",
                "relative_path",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


def inspect_python(
    interpreter: Path,
) -> dict[str, object]:
    if not interpreter.exists():
        return {
            "python_interpreter":
                str(interpreter),

            "exists":
                False,

            "python_version":
                "",

            "available_packages":
                "",

            "missing_packages":
                "|".join(
                    PYTHON_PACKAGES
                ),

            "status":
                "missing_interpreter",
        }

    probe_code = """
import importlib
import json
import sys

packages = %s

available = {}
missing = []

for package in packages:
    try:
        module = importlib.import_module(package)
        available[package] = getattr(
            module,
            "__version__",
            "available"
        )
    except Exception:
        missing.append(package)

print(
    json.dumps(
        {
            "executable": sys.executable,
            "python_version": sys.version.split()[0],
            "available": available,
            "missing": missing
        }
    )
)
""" % repr(PYTHON_PACKAGES)

    try:
        result = subprocess.run(
            [
                str(interpreter),
                "-c",
                probe_code,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

    except Exception as error:
        return {
            "python_interpreter":
                str(interpreter),

            "exists":
                True,

            "python_version":
                "",

            "available_packages":
                "",

            "missing_packages":
                "",

            "status":
                f"probe_error:{type(error).__name__}",
        }

    if result.returncode != 0:
        return {
            "python_interpreter":
                str(interpreter),

            "exists":
                True,

            "python_version":
                "",

            "available_packages":
                "",

            "missing_packages":
                "",

            "status":
                "probe_failed",
        }

    payload = json.loads(
        result.stdout.strip()
    )

    available_text = "|".join(
        (
            f"{package}="
            f"{version}"
        )
        for package, version
        in sorted(
            payload[
                "available"
            ].items()
        )
    )

    return {
        "python_interpreter":
            payload[
                "executable"
            ],

        "exists":
            True,

        "python_version":
            payload[
                "python_version"
            ],

        "available_packages":
            available_text,

        "missing_packages":
            "|".join(
                payload[
                    "missing"
                ]
            ),

        "status":
            "valid",
    }


def inventory_environments() -> pd.DataFrame:
    seen: set[str] = set()
    rows = []

    for interpreter in PYTHON_CANDIDATES:
        interpreter_text = str(
            interpreter
        )

        if interpreter_text in seen:
            continue

        seen.add(
            interpreter_text
        )

        rows.append(
            inspect_python(
                interpreter
            )
        )

    return pd.DataFrame(
        rows
    )


def disk_summary() -> pd.DataFrame:
    usage = shutil.disk_usage(
        PROJECT
    )

    return pd.DataFrame(
        [
            {
                "filesystem_path":
                    str(
                        PROJECT
                    ),

                "total_bytes":
                    usage.total,

                "used_bytes":
                    usage.used,

                "free_bytes":
                    usage.free,

                "total_gigabytes":
                    usage.total
                    / 1024 ** 3,

                "used_gigabytes":
                    usage.used
                    / 1024 ** 3,

                "free_gigabytes":
                    usage.free
                    / 1024 ** 3,

                "free_fraction":
                    usage.free
                    / usage.total,
            }
        ]
    )


def main() -> None:
    LOG_FILE.write_text(
        "",
        encoding="utf-8",
    )

    maturation_genes = read_program_genes(
        MATURATION_FILE
    )

    fetal_genes = read_program_genes(
        FETAL_FILE
    )

    shared_genes = (
        maturation_genes
        & fetal_genes
    )

    program_audit = pd.DataFrame(
        [
            {
                "developmental_program":
                    "maturation_high_increasing",

                "program_genes":
                    len(
                        maturation_genes
                    ),

                "expected_genes":
                    3333,

                "count_matches_expected":
                    len(
                        maturation_genes
                    ) == 3333,

                "program_file":
                    str(
                        MATURATION_FILE.relative_to(
                            PROJECT
                        )
                    ),
            },
            {
                "developmental_program":
                    "fetal_high_decreasing",

                "program_genes":
                    len(
                        fetal_genes
                    ),

                "expected_genes":
                    5412,

                "count_matches_expected":
                    len(
                        fetal_genes
                    ) == 5412,

                "program_file":
                    str(
                        FETAL_FILE.relative_to(
                            PROJECT
                        )
                    ),
            },
        ]
    )

    program_audit[
        "cross_program_shared_genes"
    ] = len(
        shared_genes
    )

    program_audit.to_csv(
        TABLE_DIR
        / "phase6A_input_program_gene_audit.tsv",
        sep="\t",
        index=False,
    )

    local_assets = (
        inventory_local_assets()
    )

    local_assets.to_csv(
        TABLE_DIR
        / "phase6A_local_perturbational_asset_inventory.tsv",
        sep="\t",
        index=False,
    )

    environments = (
        inventory_environments()
    )

    environments.to_csv(
        TABLE_DIR
        / "phase6A_python_environment_inventory.tsv",
        sep="\t",
        index=False,
    )

    storage = disk_summary()

    storage.to_csv(
        TABLE_DIR
        / "phase6A_storage_audit.tsv",
        sep="\t",
        index=False,
    )

    source_strategy = pd.DataFrame(
        SOURCE_STRATEGY
    )

    source_strategy.to_csv(
        METADATA_DIR
        / "phase6_source_strategy.tsv",
        sep="\t",
        index=False,
    )

    input_valid = bool(
        program_audit[
            "count_matches_expected"
        ].all()
        and len(
            shared_genes
        ) == 0
    )

    completion = pd.DataFrame(
        [
            {
                "maturation_high_genes":
                    len(
                        maturation_genes
                    ),

                "fetal_high_genes":
                    len(
                        fetal_genes
                    ),

                "shared_program_genes":
                    len(
                        shared_genes
                    ),

                "program_inputs_valid":
                    input_valid,

                "existing_local_perturbational_files":
                    len(
                        local_assets
                    ),

                "python_environments_tested":
                    len(
                        environments
                    ),

                "free_storage_gigabytes":
                    float(
                        storage.iloc[0][
                            "free_gigabytes"
                        ]
                    ),

                "source_layers_planned":
                    len(
                        source_strategy
                    ),

                "Phase6A_status":
                    (
                        "completed"
                        if input_valid
                        else "failed_input_validation"
                    ),
            }
        ]
    )

    completion.to_csv(
        TABLE_DIR
        / "phase6A_completion_summary.tsv",
        sep="\t",
        index=False,
    )

    report_lines = [
        "PHASE 6A RESOURCE AND ENVIRONMENT AUDIT",
        "",
        "Program-gene inputs:",
        program_audit.to_string(
            index=False
        ),
        "",
        "Storage:",
        storage.to_string(
            index=False
        ),
        "",
        "Python environments:",
        environments.to_string(
            index=False
        ),
        "",
        "Existing perturbational assets:",
        (
            local_assets.head(
                100
            ).to_string(
                index=False
            )
            if not local_assets.empty
            else "No existing perturbational files detected."
        ),
        "",
        "Planned source layers:",
        source_strategy.to_string(
            index=False
        ),
        "",
        "Completion:",
        completion.to_string(
            index=False
        ),
    ]

    report_file = (
        METADATA_DIR
        / "PHASE6A_RESOURCE_ENVIRONMENT_AUDIT_REPORT.txt"
    )

    report_file.write_text(
        "\n".join(
            report_lines
        )
        + "\n",
        encoding="utf-8",
    )

    LOG_FILE.write_text(
        completion.to_string(
            index=False
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "===== PHASE 6A PROGRAM INPUTS ====="
    )

    print(
        program_audit.to_string(
            index=False
        )
    )

    print()
    print(
        "===== STORAGE ====="
    )

    print(
        storage[
            [
                "total_gigabytes",
                "used_gigabytes",
                "free_gigabytes",
                "free_fraction",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    print()
    print(
        "===== PYTHON ENVIRONMENTS ====="
    )

    print(
        environments[
            [
                "python_interpreter",
                "python_version",
                "missing_packages",
                "status",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "===== EXISTING PERTURBATIONAL ASSETS ====="
    )

    if local_assets.empty:
        print(
            "No existing perturbational files detected."
        )
    else:
        print(
            local_assets[
                [
                    "size_megabytes",
                    "matched_keywords",
                    "relative_path",
                ]
            ].head(
                80
            ).to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.3f}"
                ),
            )
        )

    print()
    print(
        "===== PHASE 6A COMPLETION ====="
    )

    print(
        completion.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            f"Phase 6A failed: {error}",
            file=sys.stderr,
        )

        raise
