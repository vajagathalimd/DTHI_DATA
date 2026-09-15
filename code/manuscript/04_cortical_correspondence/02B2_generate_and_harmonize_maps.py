#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import nilearn
import nimare
import numpy as np
import pandas as pd
from nilearn.surface import vol_to_surf
from scipy import sparse

from enigmatoolbox.datasets import load_fc, load_summary_stats
from enigmatoolbox.utils.parcellation import parcel_to_surface, surface_to_parcel
from nimare.correct import FDRCorrector
from nimare.io import convert_neurosynth_to_dataset
from nimare.meta.cbma.mkda import MKDAChi2
from nimare.meta.kernel import MKDAKernel


PROJECT = Path(".")
PHASE_TABLES = PROJECT / "07_tables/main_tables/phase8"
PROC = PROJECT / "03_processed_data/functional_imaging/phase8D"
MAP_ROOT = PROC / "phase8D2B2_maps"
COG_VOL_DIR = MAP_ROOT / "cognitive_voxelwise"
COG_SURF_DIR = MAP_ROOT / "cognitive_surface"
COG_RESULT_DIR = MAP_ROOT / "cognitive_meta_results"
CLIN_SURF_DIR = MAP_ROOT / "clinical_surface"
MATRIX_DIR = MAP_ROOT / "matrices"
CACHE_DIR = MAP_ROOT / "cache"
LOG_DIR = PROJECT / "09_pipeline_logs/phase8"
ENIGMA_ROOT = PROJECT / "05_external_resources/phase8D/ENIGMA"
NEUROSYNTH_ROOT = PROJECT / "05_external_resources/phase8D/neurosynth-data"

PREFLIGHT_SUMMARY = PHASE_TABLES / "phase8D2B2_preflight_completion_summary.tsv"
COG_MANIFEST_OUT = PHASE_TABLES / "phase8D2B2_cognitive_map_manifest.tsv"
CLIN_MANIFEST_OUT = PHASE_TABLES / "phase8D2B2_clinical_map_manifest.tsv"
PARCEL_MANIFEST_OUT = PHASE_TABLES / "phase8D2B2_Schaefer100_parcel_manifest.tsv"
AUDIT_OUT = PHASE_TABLES / "phase8D2B2_map_generation_audit.tsv"
COMPLETION_OUT = PHASE_TABLES / "phase8D2B2_completion_summary.tsv"
HASH_OUT = PHASE_TABLES / "phase8D2B2_canonical_output_hashes.tsv"

COG_FULL_OUT = MATRIX_DIR / "phase8D2B2_cognitive_Schaefer100_matrix.tsv"
COG_LH_OUT = MATRIX_DIR / "phase8D2B2_cognitive_Schaefer100_LH50_matrix.tsv"
COG_LH_Z_OUT = MATRIX_DIR / "phase8D2B2_cognitive_Schaefer100_LH50_zscored.tsv"
CLIN_FULL_OUT = MATRIX_DIR / "phase8D2B2_clinical_Schaefer100_matrix.tsv"
CLIN_LH_OUT = MATRIX_DIR / "phase8D2B2_clinical_Schaefer100_LH50_matrix.tsv"
CLIN_LH_Z_OUT = MATRIX_DIR / "phase8D2B2_clinical_Schaefer100_LH50_zscored.tsv"
COMBINED_LH_OUT = MATRIX_DIR / "phase8D2B2_combined_Schaefer100_LH50_matrix.tsv"
COMBINED_LH_Z_OUT = MATRIX_DIR / "phase8D2B2_combined_Schaefer100_LH50_zscored.tsv"

COORDINATES = NEUROSYNTH_ROOT / "data-neurosynth_version-7_coordinates.tsv.gz"
METADATA = NEUROSYNTH_ROOT / "data-neurosynth_version-7_metadata.tsv.gz"
FEATURES = NEUROSYNTH_ROOT / "data-neurosynth_version-7_vocab-terms_source-abstract_type-tfidf_features.npz"
VOCABULARY = NEUROSYNTH_ROOT / "data-neurosynth_version-7_vocab-terms_vocabulary.txt"

FEATURE_GROUP = "Neurosynth_TFIDF"
LABEL_THRESHOLD = 0.001
MIN_SELECTED_STUDIES = 20
MKDA_RADIUS_MM = 10
PRIMARY_MAP = "z_desc-association"

MA_CACHE = CACHE_DIR / "neurosynth_v7_MKDA_r10_full_ma_maps.npz"
MA_IDS_CACHE = CACHE_DIR / "neurosynth_v7_MKDA_r10_full_ma_ids.tsv"
MA_CONTRACT_CACHE = CACHE_DIR / "neurosynth_v7_MKDA_r10_full_ma_contract.json"


class ValidationError(RuntimeError):
    pass


def log(message: str) -> None:
    print(message, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_strings(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def slugify(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_region(value: Any) -> str:
    if pd.isna(value):
        return ""
    label = str(value).strip().lower()
    label = label.replace("\\", "/").rsplit("/", 1)[-1]
    label = re.sub(r"\.csv$", "", label)
    label = re.sub(r"(?:[_-]asy)?[_-](?:thick|thickness)$", "", label)
    label = re.sub(r"^ctx[_-]*", "", label)
    label = re.sub(r"^(?:left|lh|l)[_\-\s]+", "l_", label)
    label = re.sub(r"^(?:right|rh|r)[_\-\s]+", "r_", label)
    return re.sub(r"[^a-z0-9]+", "", label)


def finite_mean(values: Any, weights: Any = None) -> float:
    array = np.asarray(values, dtype=float).reshape(-1)
    valid = np.isfinite(array)
    if not valid.any():
        return float("nan")
    if weights is None:
        return float(array[valid].mean())
    weight_array = np.asarray(weights, dtype=float).reshape(-1)
    weighted_valid = valid & np.isfinite(weight_array) & (weight_array > 0)
    if not weighted_valid.any():
        return float(array[valid].mean())
    return float(np.average(array[weighted_valid], weights=weight_array[weighted_valid]))


def read_delimited(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        header = handle.readline()
    if header.count(";") > max(header.count(","), header.count("\t")):
        return pd.read_csv(path, sep=";")
    if header.count("\t") > header.count(","):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def atomic_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)


def atomic_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def add_audit(
    rows: list[dict[str, Any]],
    section: str,
    item: str,
    value: Any,
    passed: bool,
    detail: str = "",
) -> None:
    rows.append(
        {
            "section": section,
            "item": item,
            "value": str(value),
            "passed": bool(passed),
            "detail": detail,
        }
    )


def find_schema_table(required: set[str], exclude: set[Path] | None = None) -> Path:
    roots = [PHASE_TABLES, PROC, PROJECT / "07_tables", PROJECT / "03_processed_data"]
    exclude_resolved = {path.resolve() for path in (exclude or set())}
    candidates: list[tuple[tuple[int, float], Path]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.tsv"):
            resolved = path.resolve()
            if resolved in seen or resolved in exclude_resolved:
                continue
            seen.add(resolved)
            try:
                frame = pd.read_csv(path, sep="\t")
            except Exception:
                continue
            if not required.issubset(set(frame.columns)):
                continue
            score = 0
            if "selection_frozen" in required:
                if len(frame) == 8 and frame["selection_frozen"].map(parse_bool).all():
                    score += 100
            if "fixed_rank" in frame.columns:
                score += 20
            candidates.append(((score, path.stat().st_mtime), path))
    if not candidates:
        raise ValidationError(f"No table found with required columns: {sorted(required)}")
    candidates.sort(key=lambda record: record[0], reverse=True)
    return candidates[0][1]


def get_selected_table(stats: Any, key: str) -> pd.DataFrame:
    if hasattr(stats, key):
        return getattr(stats, key).copy()
    if isinstance(stats, dict) and key in stats:
        return stats[key].copy()
    if hasattr(stats, "keys") and key in list(stats.keys()):
        return stats[key].copy()
    raise KeyError(f"Could not resolve ENIGMA table key: {key}")


def locate_schizophrenia_file() -> Path:
    filename = "Schizophrenia_case-controls_CortThick.csv"
    canonical = ENIGMA_ROOT / "enigmatoolbox/datasets/summary_statistics" / filename
    if canonical.is_file():
        return canonical
    matches = sorted(ENIGMA_ROOT.rglob(filename))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValidationError("Schizophrenia fallback file was not found.")
    hashes = {sha256(path) for path in matches}
    if len(hashes) == 1:
        return matches[0]
    raise ValidationError("Multiple non-identical schizophrenia fallback files were found.")


def validate_preflight() -> None:
    if not PREFLIGHT_SUMMARY.is_file():
        raise ValidationError(f"Missing preflight summary: {PREFLIGHT_SUMMARY}")
    summary = pd.read_csv(PREFLIGHT_SUMMARY, sep="\t")
    required = {
        "ready_for_phase8D2B2_map_generation",
        "Phase8D2B2_preflight_status",
    }
    if not required.issubset(summary.columns) or len(summary) != 1:
        raise ValidationError("Invalid Phase 8D2B2 preflight summary schema.")
    if not parse_bool(summary.loc[0, "ready_for_phase8D2B2_map_generation"]):
        raise ValidationError("Phase 8D2B2 preflight is not ready for map generation.")
    if str(summary.loc[0, "Phase8D2B2_preflight_status"]).strip() != "completed":
        raise ValidationError("Phase 8D2B2 preflight status is not completed.")


def load_frozen_manifests() -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    cognitive_path = find_schema_table(
        {"cognitive_domain", "selected_term", "fixed_rank", "selection_frozen", "nonzero_studies"},
        exclude={COG_MANIFEST_OUT, CLIN_MANIFEST_OUT},
    )
    clinical_path = find_schema_table(
        {"disorder", "selected_dataset_key", "loading_mode", "fixed_rank", "selection_frozen"},
        exclude={COG_MANIFEST_OUT, CLIN_MANIFEST_OUT},
    )
    cognitive = pd.read_csv(cognitive_path, sep="\t").sort_values("fixed_rank").reset_index(drop=True)
    clinical = pd.read_csv(clinical_path, sep="\t").sort_values("fixed_rank").reset_index(drop=True)
    if len(cognitive) != 8 or not cognitive["selection_frozen"].map(parse_bool).all():
        raise ValidationError("Frozen cognitive manifest failed the 8-row/frozen contract.")
    if len(clinical) != 8 or not clinical["selection_frozen"].map(parse_bool).all():
        raise ValidationError("Frozen clinical manifest failed the 8-row/frozen contract.")
    return cognitive, clinical, cognitive_path, clinical_path


def create_parcel_manifest() -> pd.DataFrame:
    _, labels, _, _ = load_fc(parcellation="schaefer_100")
    labels = np.asarray(labels, dtype=str).reshape(-1)
    if labels.size != 100:
        raise ValidationError(f"Expected 100 Schaefer labels, found {labels.size}.")

    atlas_surface = np.asarray(
        parcel_to_surface(np.arange(1, 101, dtype=float), "schaefer_100_fsa5", fill=0),
        dtype=float,
    ).reshape(-1)
    if atlas_surface.size != 20484:
        raise ValidationError(f"Expected 20484 atlas vertices, found {atlas_surface.size}.")
    lh_unique = set(np.unique(atlas_surface[:10242]).astype(int).tolist())
    rh_unique = set(np.unique(atlas_surface[10242:]).astype(int).tolist())
    if lh_unique != set(range(0, 51)):
        raise ValidationError(f"Unexpected LH Schaefer label set: {sorted(lh_unique)}")
    if rh_unique != {0, *range(51, 101)}:
        raise ValidationError(f"Unexpected RH Schaefer label set: {sorted(rh_unique)}")

    frame = pd.DataFrame(
        {
            "parcel_index_full": np.arange(1, 101),
            "hemisphere": ["LH"] * 50 + ["RH"] * 50,
            "hemisphere_parcel_index": list(range(1, 51)) + list(range(1, 51)),
            "parcel_label": labels,
            "atlas": "Schaefer100_fsa5",
        }
    )
    atomic_tsv(frame, PARCEL_MANIFEST_OUT)
    return frame


def load_clinical_tables(clinical_manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    loaded: dict[str, pd.DataFrame] = {}
    schizophrenia_path = locate_schizophrenia_file()
    for row in clinical_manifest.itertuples(index=False):
        disorder = str(row.disorder)
        mode = str(row.loading_mode)
        key = str(row.selected_dataset_key)
        if mode == "direct_repository_file":
            table = read_delimited(schizophrenia_path)
        else:
            table = get_selected_table(load_summary_stats(disorder), key)
        if not {"Structure", "d_icv"}.issubset(table.columns):
            raise ValidationError(
                f"Clinical table {disorder} lacks Structure/d_icv: {table.columns.tolist()}"
            )
        loaded[disorder] = table
    return loaded


def resolve_clinical_vector(
    table: pd.DataFrame,
    canonical_order: list[str],
    disorder: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    work = table[["Structure", "d_icv"]].copy()
    work["norm"] = work["Structure"].map(normalize_region)
    work["d_icv"] = pd.to_numeric(work["d_icv"], errors="coerce")
    canonical_set = set(canonical_order)
    matched = work[work["norm"].isin(canonical_set) & work["d_icv"].notna()].copy()
    values_by_region: dict[str, float] = {}
    duplicates: list[str] = []
    for region in canonical_order:
        values = matched.loc[matched["norm"] == region, "d_icv"].to_numpy(dtype=float)
        if values.size == 0:
            raise ValidationError(f"{disorder}: missing canonical DK region {region}.")
        if values.size > 1:
            duplicates.append(region)
            if not np.allclose(values, values[0], rtol=0, atol=1e-12, equal_nan=False):
                raise ValidationError(f"{disorder}: conflicting duplicate values for {region}: {values}")
        values_by_region[region] = float(values[0])
    vector = np.asarray([values_by_region[region] for region in canonical_order], dtype=float)
    if vector.size != 68 or not np.isfinite(vector).all():
        raise ValidationError(f"{disorder}: invalid DK68 vector.")
    extra_rows = work.loc[~work["norm"].isin(canonical_set) & work["norm"].ne(""), "Structure"].astype(str).tolist()
    return vector, {"duplicate_regions": duplicates, "extra_rows": extra_rows}


def dk68_to_schaefer100(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    surface = np.asarray(
        parcel_to_surface(np.asarray(vector, dtype=float), "aparc_fsa5", fill=np.nan),
        dtype=float,
    ).reshape(-1)
    parcels_raw = np.asarray(
        surface_to_parcel(surface, "schaefer_100_fsa5", red_op=finite_mean),
        dtype=float,
    ).reshape(-1)
    coverage = np.asarray(
        surface_to_parcel(np.isfinite(surface).astype(float), "schaefer_100_fsa5", red_op="mean"),
        dtype=float,
    ).reshape(-1)
    if surface.size != 20484 or parcels_raw.size != 101 or coverage.size != 101:
        raise ValidationError("Unexpected DK68-to-Schaefer100 dimensions.")
    parcels = parcels_raw[1:]
    if parcels.size != 100 or not np.isfinite(parcels).all():
        raise ValidationError("DK68-to-Schaefer100 produced non-finite cortical parcels.")
    if not np.isfinite(coverage[1:]).all() or not (coverage[1:] > 0).all():
        raise ValidationError("At least one Schaefer parcel lacks DK cortical coverage.")
    return surface, parcels, float(np.min(coverage[1:]))


def generate_clinical_maps(
    clinical_manifest: pd.DataFrame,
    parcel_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    log("\n===== GENERATING 8 CLINICAL MAPS =====")
    tables = load_clinical_tables(clinical_manifest)
    adhd = tables["adhd"]
    canonical = adhd[["Structure", "d_icv"]].copy()
    canonical["norm"] = canonical["Structure"].map(normalize_region)
    canonical["d_icv"] = pd.to_numeric(canonical["d_icv"], errors="coerce")
    canonical = canonical[canonical["norm"].ne("") & canonical["d_icv"].notna()].copy()
    if len(canonical) != 68 or canonical["norm"].nunique() != 68:
        raise ValidationError("ADHD table could not establish exactly 68 canonical DK regions.")
    canonical_order = canonical["norm"].tolist()

    map_columns: dict[str, np.ndarray] = {}
    manifest_rows: list[dict[str, Any]] = []
    for row in clinical_manifest.itertuples(index=False):
        disorder = str(row.disorder)
        slug = slugify(disorder)
        vector, resolution = resolve_clinical_vector(tables[disorder], canonical_order, disorder)
        surface, parcels, min_coverage = dk68_to_schaefer100(vector)

        dk_path = CLIN_SURF_DIR / f"clinical_{slug}_DK68.tsv"
        surf_path = CLIN_SURF_DIR / f"clinical_{slug}_fsaverage5_20484.npy"
        parcel_path = CLIN_SURF_DIR / f"clinical_{slug}_Schaefer100.tsv"
        atomic_tsv(
            pd.DataFrame(
                {
                    "dk_order": np.arange(1, 69),
                    "normalized_region": canonical_order,
                    "d_icv": vector,
                }
            ),
            dk_path,
        )
        np.save(surf_path, surface)
        atomic_tsv(
            pd.DataFrame(
                {
                    **parcel_manifest.to_dict(orient="list"),
                    "d_icv": parcels,
                }
            ),
            parcel_path,
        )

        map_columns[f"clinical_{slug}"] = parcels
        manifest_rows.append(
            {
                "fixed_rank": int(row.fixed_rank),
                "disorder": disorder,
                "selected_dataset_key": str(row.selected_dataset_key),
                "loading_mode": str(row.loading_mode),
                "source_effect_column": "d_icv",
                "DK_regions": 68,
                "Schaefer100_parcels": 100,
                "LH_parcels": 50,
                "minimum_DK_vertex_fraction": min_coverage,
                "duplicate_regions_resolved": "|".join(resolution["duplicate_regions"]),
                "nonregional_rows_excluded": len(resolution["extra_rows"]),
                "DK68_file": str(dk_path.relative_to(PROJECT)),
                "surface_file": str(surf_path.relative_to(PROJECT)),
                "Schaefer100_file": str(parcel_path.relative_to(PROJECT)),
                "status": "completed",
            }
        )
        log(f"[{int(row.fixed_rank)}/8] clinical {disorder}: 68 DK -> 100 Schaefer parcels")

    matrix = parcel_manifest.copy()
    for column, values in map_columns.items():
        matrix[column] = values
    lh = matrix[matrix["hemisphere"] == "LH"].reset_index(drop=True)
    numeric_columns = list(map_columns)
    atomic_tsv(matrix, CLIN_FULL_OUT)
    atomic_tsv(lh, CLIN_LH_OUT)
    atomic_tsv(zscore_matrix(lh, numeric_columns), CLIN_LH_Z_OUT)
    manifest = pd.DataFrame(manifest_rows).sort_values("fixed_rank")
    atomic_tsv(manifest, CLIN_MANIFEST_OUT)
    return matrix, manifest


def annotations_with_ids(dataset: Any) -> pd.DataFrame:
    annotations = dataset.annotations.copy()
    ids = [str(value) for value in dataset.ids]
    if "id" in annotations.columns:
        annotations = annotations.set_index("id", drop=False)
    annotations.index = annotations.index.map(str)
    missing = [identifier for identifier in ids if identifier not in annotations.index]
    if missing:
        raise ValidationError(f"Annotations are missing {len(missing)} Dataset IDs.")
    return annotations.loc[ids]


def resolve_feature_column(annotations: pd.DataFrame, term: str) -> str:
    preferred = f"{FEATURE_GROUP}__{term}"
    if preferred in annotations.columns:
        return preferred
    matches = [
        str(column)
        for column in annotations.columns
        if str(column) == term or str(column).endswith(f"__{term}")
    ]
    if len(matches) != 1:
        raise ValidationError(f"Expected one annotation column for '{term}', found {matches}")
    return matches[0]


def scipy_csr(value: Any) -> sparse.csr_matrix:
    if sparse.issparse(value):
        return value.tocsr()
    if hasattr(value, "to_scipy_sparse"):
        return value.to_scipy_sparse().tocsr()
    return sparse.csr_matrix(np.asarray(value))


def build_ma_contract(dataset_ids: list[str]) -> dict[str, Any]:
    return {
        "coordinates_sha256": sha256(COORDINATES),
        "metadata_sha256": sha256(METADATA),
        "features_sha256": sha256(FEATURES),
        "vocabulary_sha256": sha256(VOCABULARY),
        "dataset_ids_sha256": hash_strings(dataset_ids),
        "dataset_id_count": len(dataset_ids),
        "kernel": "MKDAKernel",
        "kernel_radius_mm": MKDA_RADIUS_MM,
        "nimare_version": nimare.__version__,
        "numpy_version": np.__version__,
    }


def load_or_compute_full_ma_maps(dataset: Any) -> sparse.csr_matrix:
    ids = [str(value) for value in dataset.ids]
    contract = build_ma_contract(ids)
    cache_valid = False
    if MA_CACHE.is_file() and MA_IDS_CACHE.is_file() and MA_CONTRACT_CACHE.is_file():
        try:
            saved_contract = json.loads(MA_CONTRACT_CACHE.read_text(encoding="utf-8"))
            saved_ids = pd.read_csv(MA_IDS_CACHE, sep="\t")["id"].astype(str).tolist()
            cache_valid = saved_contract == contract and saved_ids == ids
        except Exception:
            cache_valid = False
    if cache_valid:
        log(f"Loading validated full Neurosynth MA cache: {MA_CACHE.relative_to(PROJECT)}")
        matrix = sparse.load_npz(MA_CACHE).tocsr()
        if matrix.shape[0] != len(ids):
            raise ValidationError("Cached MA map row count does not match Dataset IDs.")
        return matrix

    log("Computing full Neurosynth v7 MKDA modeled-activation matrix once.")
    log("This is the longest one-time operation; the sparse result will be cached.")
    kernel = MKDAKernel(
        r=MKDA_RADIUS_MM,
        memory=str(CACHE_DIR / "joblib"),
        memory_level=1,
    )
    matrix = scipy_csr(kernel.transform(dataset, return_type="sparse"))
    if matrix.shape[0] != len(ids):
        raise ValidationError(
            f"MA matrix rows ({matrix.shape[0]}) do not match Dataset IDs ({len(ids)})."
        )
    sparse.save_npz(MA_CACHE, matrix, compressed=True)
    atomic_tsv(pd.DataFrame({"row_index": np.arange(len(ids)), "id": ids}), MA_IDS_CACHE)
    atomic_json(contract, MA_CONTRACT_CACHE)
    return matrix


def select_map_key(keys: list[str], required: str) -> str:
    if required in keys:
        return required
    matches = [key for key in keys if key.startswith(required)]
    if len(matches) == 1:
        return matches[0]
    raise ValidationError(f"Required NiMARE map '{required}' not found. Available: {keys}")


def save_optional_fdr_association(result: Any, output_path: Path) -> tuple[str, str]:
    corrected = FDRCorrector(method="indep", alpha=0.05).transform(result)
    keys = list(corrected.maps.keys())
    candidates = [
        key
        for key in keys
        if key.startswith("z_desc-association_level-voxel") and "corr-FDR" in key
    ]
    if len(candidates) != 1:
        return "", "|".join(keys)
    image = corrected.get_map(candidates[0], return_type="image")
    nib.save(image, output_path)
    return candidates[0], "|".join(keys)


def cognitive_volume_to_schaefer100(image: nib.spatialimages.SpatialImage) -> tuple[np.ndarray, np.ndarray, int]:
    lh_surface = ENIGMA_ROOT / "enigmatoolbox/datasets/surfaces/fsa5_lh.gii"
    rh_surface = ENIGMA_ROOT / "enigmatoolbox/datasets/surfaces/fsa5_rh.gii"
    if not lh_surface.is_file() or not rh_surface.is_file():
        raise ValidationError("ENIGMA fsaverage5 GIFTI surfaces were not found.")
    lh = np.asarray(
        vol_to_surf(
            image,
            str(lh_surface),
            radius=3.0,
            interpolation="linear",
            kind="auto",
        ),
        dtype=float,
    ).reshape(-1)
    rh = np.asarray(
        vol_to_surf(
            image,
            str(rh_surface),
            radius=3.0,
            interpolation="linear",
            kind="auto",
        ),
        dtype=float,
    ).reshape(-1)
    if lh.size != 10242 or rh.size != 10242:
        raise ValidationError(f"Unexpected fsaverage5 surface lengths: LH={lh.size}, RH={rh.size}")
    surface = np.concatenate([lh, rh])
    raw = np.asarray(
        surface_to_parcel(surface, "schaefer_100_fsa5", red_op=finite_mean),
        dtype=float,
    ).reshape(-1)
    if raw.size != 101:
        raise ValidationError(f"Expected 101 raw Schaefer values, found {raw.size}.")
    parcels = raw[1:]
    if parcels.size != 100 or not np.isfinite(parcels).all():
        raise ValidationError("Cognitive volume projection produced non-finite Schaefer cortical values.")
    return surface, parcels, int(np.isfinite(surface).sum())


def generate_cognitive_maps(
    cognitive_manifest: pd.DataFrame,
    parcel_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    log("\n===== BUILDING NEUROSYNTH v7 DATASET =====")
    for path in (COORDINATES, METADATA, FEATURES, VOCABULARY):
        if not path.is_file():
            raise ValidationError(f"Missing Neurosynth resource: {path}")

    dataset = convert_neurosynth_to_dataset(
        coordinates_file=str(COORDINATES),
        metadata_file=str(METADATA),
        annotations_files={"features": str(FEATURES), "vocabulary": str(VOCABULARY)},
        feature_groups=[FEATURE_GROUP],
        target="mni152_2mm",
    )
    all_ids = [str(value) for value in dataset.ids]
    if len(all_ids) != 14371:
        raise ValidationError(f"Expected 14371 Neurosynth IDs, found {len(all_ids)}.")
    annotations = annotations_with_ids(dataset)
    ma_full = load_or_compute_full_ma_maps(dataset)
    id_to_row = {identifier: index for index, identifier in enumerate(all_ids)}

    log("\n===== GENERATING 8 COGNITIVE MKDA-CHI2 MAPS =====")
    map_columns: dict[str, np.ndarray] = {}
    manifest_rows: list[dict[str, Any]] = []

    for row in cognitive_manifest.itertuples(index=False):
        start = time.time()
        domain = str(row.cognitive_domain)
        term = str(row.selected_term)
        rank = int(row.fixed_rank)
        domain_slug = slugify(domain)
        term_slug = slugify(term)
        column = resolve_feature_column(annotations, term)
        values = pd.to_numeric(annotations[column], errors="coerce").fillna(0.0)
        nonzero_count = int((values > 0).sum())
        expected_nonzero = int(row.nonzero_studies)
        if nonzero_count != expected_nonzero:
            raise ValidationError(
                f"{term}: nonzero annotation count {nonzero_count} != frozen count {expected_nonzero}."
            )
        selected_mask = values > LABEL_THRESHOLD
        selected_ids = [identifier for identifier in all_ids if bool(selected_mask.loc[identifier])]
        selected_set = set(selected_ids)
        unselected_ids = [identifier for identifier in all_ids if identifier not in selected_set]
        if len(selected_ids) < MIN_SELECTED_STUDIES:
            raise ValidationError(
                f"{term}: only {len(selected_ids)} studies exceed threshold {LABEL_THRESHOLD}."
            )
        if len(unselected_ids) < MIN_SELECTED_STUDIES:
            raise ValidationError(f"{term}: insufficient unselected studies ({len(unselected_ids)}).")

        dset_selected = dataset.slice(selected_ids)
        dset_unselected = dataset.slice(unselected_ids)
        selected_rows = [id_to_row[identifier] for identifier in selected_ids]
        unselected_rows = [id_to_row[identifier] for identifier in unselected_ids]
        ma_selected = ma_full[selected_rows, :]
        ma_unselected = ma_full[unselected_rows, :]

        estimator = MKDAChi2(
            prior=0.5,
            kernel__r=MKDA_RADIUS_MM,
            memory=str(CACHE_DIR / "joblib"),
            memory_level=1,
        )
        result = estimator.fit(
            dset_selected,
            dset_unselected,
            drop_invalid=True,
            ma_maps1=ma_selected,
            ma_maps2=ma_unselected,
        )
        map_keys = list(result.maps.keys())
        primary_key = select_map_key(map_keys, PRIMARY_MAP)
        primary_image = result.get_map(primary_key, return_type="image")

        stem = f"cognitive_{rank:02d}_{domain_slug}_{term_slug}"
        primary_path = COG_VOL_DIR / f"{stem}_{primary_key}.nii.gz"
        fdr_path = COG_VOL_DIR / f"{stem}_z_desc-association_level-voxel_corr-FDR_method-indep.nii.gz"
        result_path = COG_RESULT_DIR / f"{stem}_MKDAChi2_result.pkl.gz"
        surface_path = COG_SURF_DIR / f"{stem}_fsaverage5_20484.npy"
        parcel_path = COG_SURF_DIR / f"{stem}_Schaefer100.tsv"
        nib.save(primary_image, primary_path)
        fdr_key, corrected_keys = save_optional_fdr_association(result, fdr_path)
        result.save(result_path)
        surface, parcels, finite_vertices = cognitive_volume_to_schaefer100(primary_image)
        np.save(surface_path, surface)
        atomic_tsv(
            pd.DataFrame(
                {
                    **parcel_manifest.to_dict(orient="list"),
                    "association_z": parcels,
                }
            ),
            parcel_path,
        )

        map_columns[f"cognitive_{domain_slug}"] = parcels
        elapsed = time.time() - start
        manifest_rows.append(
            {
                "fixed_rank": rank,
                "cognitive_domain": domain,
                "selected_term": term,
                "annotation_column": column,
                "annotation_threshold": LABEL_THRESHOLD,
                "nonzero_studies": nonzero_count,
                "selected_studies": len(selected_ids),
                "unselected_studies": len(unselected_ids),
                "MKDA_radius_mm": MKDA_RADIUS_MM,
                "primary_map_key": primary_key,
                "FDR_map_key": fdr_key,
                "Schaefer100_parcels": 100,
                "LH_parcels": 50,
                "finite_surface_vertices": finite_vertices,
                "runtime_seconds": round(elapsed, 3),
                "primary_NIfTI": str(primary_path.relative_to(PROJECT)),
                "FDR_NIfTI": str(fdr_path.relative_to(PROJECT)) if fdr_key else "",
                "MetaResult_file": str(result_path.relative_to(PROJECT)),
                "surface_file": str(surface_path.relative_to(PROJECT)),
                "Schaefer100_file": str(parcel_path.relative_to(PROJECT)),
                "available_uncorrected_maps": "|".join(map_keys),
                "available_corrected_maps": corrected_keys,
                "status": "completed",
            }
        )
        log(
            f"[{rank}/8] cognitive {domain} <- '{term}': "
            f"selected={len(selected_ids)}, Schaefer=100, runtime={elapsed/60:.1f} min"
        )

    matrix = parcel_manifest.copy()
    for column, values in map_columns.items():
        matrix[column] = values
    lh = matrix[matrix["hemisphere"] == "LH"].reset_index(drop=True)
    numeric_columns = list(map_columns)
    atomic_tsv(matrix, COG_FULL_OUT)
    atomic_tsv(lh, COG_LH_OUT)
    atomic_tsv(zscore_matrix(lh, numeric_columns), COG_LH_Z_OUT)
    manifest = pd.DataFrame(manifest_rows).sort_values("fixed_rank")
    atomic_tsv(manifest, COG_MANIFEST_OUT)
    return matrix, manifest


def zscore_matrix(frame: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in numeric_columns:
        values = pd.to_numeric(output[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValidationError(f"Non-finite values in matrix column {column}.")
        standard_deviation = float(values.std(ddof=0))
        if standard_deviation <= 0:
            raise ValidationError(f"Zero variance in matrix column {column}.")
        output[column] = (values - float(values.mean())) / standard_deviation
    return output


def combine_lh_matrices(cognitive: pd.DataFrame, clinical: pd.DataFrame) -> None:
    cog_lh = cognitive[cognitive["hemisphere"] == "LH"].reset_index(drop=True)
    clin_lh = clinical[clinical["hemisphere"] == "LH"].reset_index(drop=True)
    key_columns = ["parcel_index_full", "hemisphere", "hemisphere_parcel_index", "parcel_label", "atlas"]
    if not cog_lh[key_columns].equals(clin_lh[key_columns]):
        raise ValidationError("Cognitive and clinical LH50 parcel metadata do not align.")
    cog_columns = [column for column in cog_lh.columns if column.startswith("cognitive_")]
    clin_columns = [column for column in clin_lh.columns if column.startswith("clinical_")]
    combined = cog_lh[key_columns + cog_columns].copy()
    for column in clin_columns:
        combined[column] = clin_lh[column].to_numpy(dtype=float)
    atomic_tsv(combined, COMBINED_LH_OUT)
    atomic_tsv(zscore_matrix(combined, cog_columns + clin_columns), COMBINED_LH_Z_OUT)


def collect_hashes(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in sorted({item.resolve() for item in paths if item.is_file()}):
        rows.append(
            {
                "relative_path": str(path.relative_to(PROJECT)),
                "size_bytes": path.stat().st_size,
                "SHA256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    for directory in (
        PHASE_TABLES,
        PROC,
        MAP_ROOT,
        COG_VOL_DIR,
        COG_SURF_DIR,
        COG_RESULT_DIR,
        CLIN_SURF_DIR,
        MATRIX_DIR,
        CACHE_DIR,
        LOG_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    audit_rows: list[dict[str, Any]] = []
    try:
        validate_preflight()
        add_audit(audit_rows, "inputs", "phase8D2B2_preflight_completed", True, True)

        cognitive_manifest, clinical_manifest, cognitive_input, clinical_input = load_frozen_manifests()
        add_audit(
            audit_rows,
            "inputs",
            "frozen_cognitive_manifest",
            cognitive_input.relative_to(PROJECT),
            len(cognitive_manifest) == 8,
            sha256(cognitive_input),
        )
        add_audit(
            audit_rows,
            "inputs",
            "frozen_clinical_manifest",
            clinical_input.relative_to(PROJECT),
            len(clinical_manifest) == 8,
            sha256(clinical_input),
        )

        parcel_manifest = create_parcel_manifest()
        atlas_ok = (
            len(parcel_manifest) == 100
            and (parcel_manifest["hemisphere"] == "LH").sum() == 50
            and (parcel_manifest["hemisphere"] == "RH").sum() == 50
        )
        add_audit(audit_rows, "atlas", "Schaefer100_partition_50LH_50RH", "100=50+50", atlas_ok)

        clinical_matrix, clinical_map_manifest = generate_clinical_maps(clinical_manifest, parcel_manifest)
        add_audit(
            audit_rows,
            "clinical",
            "clinical_maps_completed",
            len(clinical_map_manifest),
            len(clinical_map_manifest) == 8 and (clinical_map_manifest["status"] == "completed").all(),
        )
        clinical_columns = [column for column in clinical_matrix.columns if column.startswith("clinical_")]
        clinical_finite = len(clinical_columns) == 8 and np.isfinite(clinical_matrix[clinical_columns].to_numpy(dtype=float)).all()
        add_audit(audit_rows, "clinical", "clinical_Schaefer100_all_finite", clinical_finite, clinical_finite)

        cognitive_matrix, cognitive_map_manifest = generate_cognitive_maps(cognitive_manifest, parcel_manifest)
        add_audit(
            audit_rows,
            "cognitive",
            "cognitive_maps_completed",
            len(cognitive_map_manifest),
            len(cognitive_map_manifest) == 8 and (cognitive_map_manifest["status"] == "completed").all(),
        )
        cognitive_columns = [column for column in cognitive_matrix.columns if column.startswith("cognitive_")]
        cognitive_finite = len(cognitive_columns) == 8 and np.isfinite(cognitive_matrix[cognitive_columns].to_numpy(dtype=float)).all()
        add_audit(audit_rows, "cognitive", "cognitive_Schaefer100_all_finite", cognitive_finite, cognitive_finite)

        combine_lh_matrices(cognitive_matrix, clinical_matrix)
        combined = pd.read_csv(COMBINED_LH_OUT, sep="\t")
        combined_map_columns = [
            column for column in combined.columns if column.startswith("cognitive_") or column.startswith("clinical_")
        ]
        combined_ok = (
            len(combined) == 50
            and len(combined_map_columns) == 16
            and np.isfinite(combined[combined_map_columns].to_numpy(dtype=float)).all()
        )
        add_audit(audit_rows, "combined", "combined_LH50_16_maps", f"{len(combined)}x{len(combined_map_columns)}", combined_ok)

        canonical_paths = [
            PARCEL_MANIFEST_OUT,
            COG_MANIFEST_OUT,
            CLIN_MANIFEST_OUT,
            COG_FULL_OUT,
            COG_LH_OUT,
            COG_LH_Z_OUT,
            CLIN_FULL_OUT,
            CLIN_LH_OUT,
            CLIN_LH_Z_OUT,
            COMBINED_LH_OUT,
            COMBINED_LH_Z_OUT,
        ]
        canonical_paths.extend(COG_VOL_DIR.glob("*.nii.gz"))
        canonical_paths.extend(COG_SURF_DIR.glob("*"))
        canonical_paths.extend(CLIN_SURF_DIR.glob("*"))
        canonical_paths.extend(COG_RESULT_DIR.glob("*.pkl.gz"))

        audit = pd.DataFrame(audit_rows)
        all_passed = bool(audit["passed"].all())
        atomic_tsv(audit, AUDIT_OUT)

        completion = pd.DataFrame(
            [
                {
                    "Phase8D2B2_preflight_confirmed": True,
                    "cognitive_maps_requested": 8,
                    "cognitive_maps_completed": len(cognitive_map_manifest),
                    "clinical_maps_requested": 8,
                    "clinical_maps_completed": len(clinical_map_manifest),
                    "Schaefer100_parcels": 100,
                    "Schaefer100_LH_parcels": 50,
                    "combined_LH50_maps": len(combined_map_columns),
                    "all_map_values_finite": cognitive_finite and clinical_finite and combined_ok,
                    "ready_for_phase8D2C": all_passed,
                    "Phase8D2B2_status": "completed" if all_passed else "failed_validation",
                    "python_version": sys.version.split()[0],
                    "nimare_version": nimare.__version__,
                    "nilearn_version": nilearn.__version__,
                    "numpy_version": np.__version__,
                    "pandas_version": pd.__version__,
                }
            ]
        )
        atomic_tsv(completion, COMPLETION_OUT)

        canonical_paths.extend([AUDIT_OUT, COMPLETION_OUT])
        hashes = collect_hashes(canonical_paths)
        atomic_tsv(hashes, HASH_OUT)

        log("\n===== PHASE 8D2B2 COMPLETION =====")
        log(completion.to_string(index=False))
        log("\n===== MAP GENERATION AUDIT =====")
        log(audit.to_string(index=False))
        log("\n===== CANONICAL OUTPUT HASHES =====")
        log(hashes.to_string(index=False))

        if not all_passed:
            failed = audit.loc[~audit["passed"], "item"].tolist()
            raise ValidationError(f"Phase 8D2B2 validation failed: {failed}")

    except Exception as exc:
        if audit_rows:
            atomic_tsv(pd.DataFrame(audit_rows), AUDIT_OUT)
        failure = pd.DataFrame(
            [
                {
                    "Phase8D2B2_status": "failed",
                    "ready_for_phase8D2C": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            ]
        )
        atomic_tsv(failure, COMPLETION_OUT)
        log(f"Phase 8D2B2 failed: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
