#!/usr/bin/env python3
"""ProteinGym CSV/PDB files arranged by the 217-assay reference."""
import argparse
import csv
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resolve_unique(root, name):
    paths = sorted(root.rglob(name))
    if not paths:
        raise FileNotFoundError(f"Missing {name} under {root}")
    hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if len(hashes) != 1:
        raise ValueError(f"Multiple different copies of {name} under {root}")
    return paths[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "data/proteingym_local/reference/DMS_substitutions.csv")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/ProteinGym_v1_SaProt_AF2_v2")
    parser.add_argument("--flat-dms", type=Path, default=ROOT / "data/proteingym_local/DMS_ProteinGym_substitutions_v2")
    args = parser.parse_args()
    if args.dataset.exists() or args.flat_dms.exists():
        raise SystemExit("Output already exists. Select new --dataset and --flat-dms directories.")
    with args.reference.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    plan = []
    # Validate all source files before creating any output.
    for row in rows:
        assay = row["DMS_id"]
        if Path(assay).name != assay:
            raise ValueError(f"Unsafe assay ID: {assay}")
        csv_path = resolve_unique(args.raw_dir, row["DMS_filename"])
        pdb_path = resolve_unique(args.raw_dir, row["pdb_file"])
        with csv_path.open(newline="") as handle:
            columns = csv.DictReader(handle).fieldnames or []
        if not {"mutant", "mutated_sequence", "DMS_score"}.issubset(columns):
            raise ValueError(f"Invalid ProteinGym CSV columns: {csv_path}")
        plan.append((assay, csv_path, pdb_path))
    args.flat_dms.mkdir(parents=True)
    for assay, source_csv, source_pdb in plan:
        out = args.dataset / "DATASET" / assay
        out.mkdir(parents=True)
        shutil.copy2(source_csv, out / f"{assay}.csv")
        shutil.copy2(source_pdb, out / f"{assay}.pdb")
        shutil.copy2(source_csv, args.flat_dms / f"{assay}.csv")
    print(f"[PASS] Prepared {len(plan)} assays using reference DMS_filename/pdb_file mappings.")


if __name__ == "__main__":
    main()
