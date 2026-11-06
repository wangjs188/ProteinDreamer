#!/usr/bin/env python3
"""Per-variant predictions and measurements for metric recalculation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCORE_COLUMN = "ProteinDreamer_WMZ_SiteRM_SaProt_add_neffgate_wmz_saprot_plmens2_AF2only_b130"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scores-dir", "measurements-dir", "reference", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Output exists; choose a new export directory.")
    reference = pd.read_csv(args.reference)
    if len(reference) != 217 or not reference.DMS_id.is_unique:
        raise ValueError("Expected the frozen 217-assay reference.")
    args.out.mkdir(parents=True)
    manifest = {
        "version": 1,
        "scope": "Archived per-variant predictions and experimental measurements; not a new model inference run.",
        "prediction_column_in_source": SCORE_COLUMN,
        "source_reference_sha256": sha256(args.reference),
        "transformation": "Exact one-to-one mutant join. Original numeric CSV strings retained; no filtering, imputation, score correction or new prediction.",
        "assays": [],
    }
    for row in reference.to_dict("records"):
        assay = row["DMS_id"]
        if Path(assay).name != assay:
            raise ValueError("Unsafe assay identifier")
        scores_path = args.scores_dir / (assay + ".csv")
        labels_path = args.measurements_dir / (assay + ".csv")
        scores = pd.read_csv(scores_path, usecols=["mutant", SCORE_COLUMN], dtype=str, keep_default_na=False)
        labels = pd.read_csv(labels_path, usecols=["mutant", "DMS_score"], dtype=str, keep_default_na=False)
        if not scores.mutant.is_unique or not labels.mutant.is_unique:
            raise ValueError(f"Duplicate mutant identifiers: {assay}")
        if set(scores.mutant) != set(labels.mutant):
            raise ValueError(f"Prediction/measurement coverage differs: {assay}")
        frame = labels.merge(scores, on="mutant", validate="one_to_one", sort=False)
        frame = frame.rename(columns={SCORE_COLUMN: "ProteinDreamer_score"})
        if not np.isfinite(frame[["DMS_score", "ProteinDreamer_score"]].astype(float).to_numpy()).all():
            raise ValueError(f"Non-finite values: {assay}")
        target = args.out / (assay + ".csv.gz")
        frame.to_csv(target, index=False, lineterminator="\n",
                     compression={"method": "gzip", "mtime": 0, "compresslevel": 6})
        manifest["assays"].append({
            "DMS_id": assay, "UniProt_ID": row["UniProt_ID"],
            "Selection_Type": row["coarse_selection_type"], "variants": len(frame),
            "file": target.name, "bytes": target.stat().st_size, "sha256": sha256(target),
            "prediction_source": args.scores_dir.name + "/" + scores_path.name,
            "prediction_source_sha256": sha256(scores_path),
            "measurement_source": args.measurements_dir.name + "/" + labels_path.name,
            "measurement_source_sha256": sha256(labels_path),
        })
        if len(manifest["assays"]) % 50 == 0:
            print(f"[export] {len(manifest['assays'])}/217 assays", flush=True)
    manifest["total_variants"] = sum(x["variants"] for x in manifest["assays"])
    (args.out / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"assays": 217, "variants": manifest["total_variants"],
                      "compressed_bytes": sum(x["bytes"] for x in manifest["assays"])}), flush=True)


if __name__ == "__main__":
    main()
