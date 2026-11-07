#!/usr/bin/env python3
"""Benchmark input files and checksums."""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from script.setup.verify_benchmark_inputs import check_dataset
REF_HASHES = {
    "a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308",
    "80ac07373c3256d4fbd2112f8705f37ea761549d593549a740e5524e11e9f68b",
}

# These IDs had no cache in the original run; replacements would change its inputs.
HISTORICAL_FALLBACK_IDS = frozenset({
    "F7YBW8_MESOW_Ding_2023",
    "PSAE_PICP2_Tsuboyama_2023_1PSE",
    "Q6WV12_9MAXI_Somermeyer_2022",
})


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_evaluator(directory, problems):
    snapshot = json.loads((ROOT / "external/ProteinGym-main/SNAPSHOT.json").read_text())
    for name, expected in snapshot["sha256"].items():
        path = directory / name
        if not path.is_file() or sha256(path) != expected:
            problems.append(f"Frozen ProteinGym evaluator missing or changed: {path}")


def check_models(directory, problems):
    lock = json.loads((ROOT / "model/SAPROT_LOCK.json").read_text())
    for family, info in lock["models"].items():
        for name, expected in info["sha256"].items():
            path = directory / family / name
            if not path.is_file() or sha256(path) != expected:
                problems.append(f"Pinned SaProt file missing or changed: {path}")
    alternate = directory / "SaProt_650M_AF2/model.safetensors"
    if alternate.exists():
        problems.append(f"Unexpected alternate weight file could override pinned weights: {alternate}")


def check_caches(rows, msa, teacher, problems):
    hashes = {}
    for line in (ROOT / "data/proteingym_local/CACHES.sha256").read_text().splitlines():
        if line.strip():
            digest, relative = line.split(maxsplit=1)
            hashes[relative.lstrip("*")] = digest
    for row in rows:
        assay, length = row["DMS_id"], len(row["target_seq"])
        specs = (
            (msa, "msa_cache_dms_v1", "log_probs_msa_full",
             ("msa_cov_full", "msa_entropy_full", "msa_has_full", "msa_reliable_full")),
            (teacher, "teacher_siterm_dms_s4096_e100_q64_reg05_v2t1",
             "log_probs_teacher_full", ("teacher_has_full", "teacher_reliable_full")),
        )
        for directory, source_name, matrix, vectors in specs:
            path = directory / f"{assay}.npz"
            if assay in HISTORICAL_FALLBACK_IDS:
                if path.exists():
                    problems.append(f"Historical no-cache assay must retain its fallback: {path}")
                continue
            if not path.is_file():
                problems.append(f"Required historical cache missing: {path}")
                continue
            expected_hash = hashes.get(f"data/proteingym_local/{source_name}/{assay}.npz")
            if expected_hash is None or sha256(path) != expected_hash:
                problems.append(f"Historical cache checksum mismatch: {path}")
                continue
            try:
                with np.load(path, allow_pickle=False) as cache:
                    for key in (matrix,) + vectors:
                        values = cache[key]
                        expected = (length, 20) if key == matrix else (length,)
                        if values.shape != expected or not np.isfinite(values).all():
                            problems.append(f"Invalid {key} in {path}: expected {expected}")
                    if directory == msa and not np.isfinite(cache["log_neff_global"]).all():
                        problems.append(f"Invalid log_neff_global in {path}")
            except Exception as exc:
                problems.append(f"Cannot read {path}: {exc}")


def configured(name, default):
    return Path(os.environ.get(name, str(ROOT / default)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--evaluator-only", action="store_true")
    group.add_argument("--models-only", action="store_true")
    args = parser.parse_args()
    problems = []
    pg = configured("PROTEINGYM_ROOT", "external/ProteinGym-main")
    if not args.models_only:
        check_evaluator(pg, problems)
    if not args.evaluator_only:
        check_models(configured("MODEL_ROOT", "model"), problems)
    if args.evaluator_only or args.models_only:
        for problem in problems:
            print("[FAIL]", problem)
        if problems:
            raise SystemExit(2)
        print("[PASS] Pinned external asset checksums.")
        return
    def require(path):
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"Missing or empty: {path}")
            return False
        return True

    ref_path = configured("REF_CSV", "data/proteingym_local/reference/DMS_substitutions.csv")
    dataset = configured("DATASET_ROOT", "data/ProteinGym_v1_SaProt_AF2_v2")
    flat = configured("FLAT_DMS_DIR", "data/proteingym_local/DMS_ProteinGym_substitutions_v2")
    folds = configured("FOLDS_DIR", "data/proteingym_local/folds/PG_AF2_by_protein_k5_seed3407")
    msa = configured("MSA_CACHE_DIR", "data/proteingym_local/msa_cache_dms_v1")
    teacher = configured("TEACHER_CACHE_DIR", "data/proteingym_local/teacher_siterm_dms_s4096_e100_q64_reg05_v2t1")
    if require(ref_path):
        if hashlib.sha256(ref_path.read_bytes()).hexdigest() not in REF_HASHES:
            problems.append("Reference checksum does not match the frozen 217-assay release.")
        with ref_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 217:
            problems.append(f"Expected 217 reference rows, got {len(rows)}.")
        if require(folds / "proteins.txt"):
            ids = set((folds / "proteins.txt").read_text().splitlines())
            if ids != {r["DMS_id"] for r in rows}:
                problems.append("Reference and frozen assay IDs differ.")
        problems.extend(check_dataset(rows, dataset, flat))
        check_caches(rows, msa, teacher, problems)
    foldseek = os.environ.get("FOLDSEEK_BIN", "foldseek")
    if not shutil.which(foldseek):
        problems.append(f"Foldseek not executable: {foldseek}")
    if problems:
        for problem in problems[:25]:
            print("[FAIL]", problem)
        print(f"[FAIL] {len(problems)} input problems in total. See docs/REPRODUCE_0546.md.")
        raise SystemExit(2)
    print("[PASS] External file coverage, cache schemas and frozen reference.")
    print("[INFO] Three documented historical no-cache assays retain uniform/zero fallback inputs.")
    print("This is an input check, not proof of numeric reproduction or cache provenance.")


if __name__ == "__main__":
    main()
