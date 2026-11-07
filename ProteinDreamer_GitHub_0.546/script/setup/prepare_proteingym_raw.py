#!/usr/bin/env python3
"""ProteinGym CSV/PDB files arranged by the 217-assay reference."""
import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from script.setup.verify_benchmark_inputs import check_bundle, check_table


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
    parser.add_argument("--structures", type=Path, default=ROOT / "data/structures_0546",
                        help="Pinned historical structure archive and manifest.")
    parser.add_argument("--reference", type=Path, default=ROOT / "data/proteingym_local/reference/DMS_substitutions.csv")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/ProteinGym_v1_SaProt_AF2_v2")
    parser.add_argument("--flat-dms", type=Path, default=ROOT / "data/proteingym_local/DMS_ProteinGym_substitutions_v2")
    args = parser.parse_args()
    if args.dataset.exists() or args.flat_dms.exists():
        raise SystemExit("Output already exists. Select new --dataset and --flat-dms directories.")
    with args.reference.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = check_bundle(args.structures)
    plan = []
    # Validate all source files before creating any output.
    for row in rows:
        assay = row["DMS_id"]
        if Path(assay).name != assay:
            raise ValueError(f"Unsafe assay ID: {assay}")
        csv_path = resolve_unique(args.raw_dir, row["DMS_filename"])
        if assay not in records:
            raise ValueError(f"No historical structure for {assay}")
        record = records[assay]
        sequence = row["target_seq"]
        if (len(sequence) != record["length"]
                or hashlib.sha256(sequence.encode()).hexdigest() != record["sequence_sha256"]):
            raise ValueError(f"Historical structure/reference mismatch: {assay}")
        with csv_path.open(newline="") as handle:
            columns = csv.DictReader(handle).fieldnames or []
        if not {"mutant", "mutated_sequence", "DMS_score"}.issubset(columns):
            raise ValueError(f"Invalid ProteinGym CSV columns: {csv_path}")
        check_table(csv_path, record)
        plan.append((assay, csv_path, record["file"]))
    args.flat_dms.mkdir(parents=True)
    with ZipFile(args.structures / "structures.zip") as bundle:
        for assay, source_csv, member in plan:
            out = args.dataset / "DATASET" / assay
            out.mkdir(parents=True)
            shutil.copy2(source_csv, out / f"{assay}.csv")
            (out / f"{assay}.pdb").write_bytes(bundle.read(member))
            shutil.copy2(source_csv, args.flat_dms / f"{assay}.csv")
    print(f"[PASS] Prepared {len(plan)} assays with the pinned historical structures.")


if __name__ == "__main__":
    main()
