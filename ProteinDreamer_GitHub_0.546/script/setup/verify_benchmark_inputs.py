#!/usr/bin/env python3
"""Check the structures and mutation tables used by the 0.546 run."""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(directory):
    manifest = json.loads((Path(directory) / "PROVENANCE.json").read_text())
    records = {}
    for item in manifest["assays"]:
        assay = item["DMS_id"]
        if not assay or Path(assay).name != assay or "\\" in assay:
            raise ValueError(f"Unsafe assay ID: {assay}")
        if assay in records or item["file"] != f"{assay}.pdb":
            raise ValueError(f"Invalid structure manifest entry: {assay}")
        records[assay] = item
    if not records:
        raise ValueError("Empty structure manifest")
    return manifest, records


def check_bundle(directory):
    directory = Path(directory)
    manifest, records = read_manifest(directory)
    archive = directory / "structures.zip"
    if sha256(archive) != manifest["archive_sha256"]:
        raise ValueError(f"Structure archive checksum mismatch: {archive}")
    with ZipFile(archive) as bundle:
        names = [item["file"] for item in records.values()]
        if sorted(bundle.namelist()) != sorted(names):
            raise ValueError("Structure archive members differ from the manifest")
        for item in records.values():
            if hashlib.sha256(bundle.read(item["file"])).hexdigest() != item["sha256"]:
                raise ValueError(f"Structure checksum mismatch: {item['DMS_id']}")
    return records


def check_structure(row, pdb, record):
    from src.saprot_dataset import load_structure_sequence

    if sha256(pdb) != record["sha256"]:
        raise ValueError(f"Historical structure checksum mismatch: {pdb}")
    sequence = load_structure_sequence(pdb)
    if (sequence != row["target_seq"] or len(sequence) != record["length"]
            or hashlib.sha256(sequence.encode()).hexdigest() != record["sequence_sha256"]):
        raise ValueError(f"Structure/reference sequence mismatch: {row['DMS_id']}")


def check_table(path, record):
    # This checksum pins all rows, their order, mutation coordinates and measurements.
    if sha256(path) != record["dms_sha256"]:
        raise ValueError(f"Frozen mutation table checksum mismatch: {path}")


def check_dataset(rows, dataset, flat=None, structures=None):
    directory = structures or ROOT / "data/structures_0546"
    _, records = read_manifest(directory)
    problems = []
    for row in rows:
        assay = row["DMS_id"]
        try:
            record = records[assay]
            base = Path(dataset) / "DATASET" / assay
            check_structure(row, base / f"{assay}.pdb", record)
            check_table(base / f"{assay}.csv", record)
            if flat is not None:
                check_table(Path(flat) / f"{assay}.csv", record)
        except (KeyError, OSError, ValueError) as exc:
            problems.append(f"{assay}: {exc}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(os.environ.get(
        "DATASET_ROOT", ROOT / "data/ProteinGym_v1_SaProt_AF2_v2")))
    parser.add_argument("--reference", type=Path, default=Path(os.environ.get(
        "REF_CSV", ROOT / "data/proteingym_local/reference/DMS_substitutions.csv")))
    parser.add_argument("--msa-cache", type=Path, default=Path(os.environ.get(
        "MSA_CACHE_DIR", ROOT / "data/proteingym_local/msa_cache_dms_v1")))
    parser.add_argument("--teacher-cache", type=Path, default=Path(os.environ.get(
        "TEACHER_CACHE_DIR", ROOT / "data/proteingym_local/teacher_siterm_dms_s4096_e100_q64_reg05_v2t1")))
    args = parser.parse_args()
    from script.setup.verify_external_assets import REF_HASHES, check_caches

    if sha256(args.reference) not in REF_HASHES:
        raise SystemExit("Reference checksum does not match the frozen 217-assay release.")
    with args.reference.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    problems = check_dataset(rows, args.dataset)
    check_caches(rows, args.msa_cache, args.teacher_cache, problems)
    if problems:
        for problem in problems:
            print("[FAIL]", problem)
        raise SystemExit(2)
    print(f"[PASS] Historical structures, mutation tables and caches: {len(rows)} assays.")


if __name__ == "__main__":
    main()
