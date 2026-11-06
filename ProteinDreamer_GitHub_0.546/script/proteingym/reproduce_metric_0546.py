#!/usr/bin/env python3
"""Spearman recalculation from archived predictions and measurements (217 assays)."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
MODEL_COLUMN = "ProteinDreamer_WMZ_SiteRM_SaProt_add_neffgate_wmz_saprot_plmens2_AF2only_b130"
EXPECTED_AVERAGE = 0.5455137705176768


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "data/metric_0546")
    parser.add_argument("--out", type=Path, default=ROOT / "result/metric_0546")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Output already exists; select a new --out directory.")
    manifest_path = args.inputs / "PROVENANCE.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = pd.read_csv(ROOT / "data/proteingym_local/reference/DMS_substitutions.csv").set_index("DMS_id")
    expected = pd.read_csv(ROOT / "artifacts/proteingym_0.546/dms_level_spearman.csv").set_index("DMS ID")
    records = manifest["assays"]
    ids = [x["DMS_id"] for x in records]
    if len(ids) != 217 or len(set(ids)) != 217 or set(ids) != set(reference.index) or set(ids) != set(expected.index):
        raise ValueError("The inputs must cover all 217 frozen assays exactly once.")
    results = []
    for item in records:
        assay = item["DMS_id"]
        if item["file"] != assay + ".csv.gz" or Path(item["file"]).name != item["file"]:
            raise ValueError(f"Unexpected input filename: {assay}")
        path = args.inputs / item["file"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Input checksum mismatch: {assay}")
        row = reference.loc[assay]
        if item["UniProt_ID"] != row.UniProt_ID or item["Selection_Type"] != row.coarse_selection_type:
            raise ValueError(f"Reference grouping changed: {assay}")
        frame = pd.read_csv(path)
        if len(frame) != item["variants"] or not frame.mutant.is_unique:
            raise ValueError(f"Variant coverage changed: {assay}")
        values = frame[["DMS_score", "ProteinDreamer_score"]].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite input: {assay}")
        rho = float(spearmanr(values[:, 0], values[:, 1]).statistic)
        if not np.isfinite(rho):
            raise ValueError(f"Undefined Spearman correlation: {assay}")
        results.append({"DMS_id": assay, "UniProt_ID": item["UniProt_ID"],
                        "Selection_Type": item["Selection_Type"], "variants": len(frame),
                        "Spearman_unrounded": rho})
    scores = pd.DataFrame(results)
    # The archived ProteinGym evaluator rounds assay metrics before aggregation.
    scores["Spearman"] = scores.Spearman_unrounded.round(3)
    frozen = expected.loc[scores.DMS_id, MODEL_COLUMN].to_numpy(dtype=float)
    scores["matches_frozen_assay"] = scores.Spearman.to_numpy() == frozen
    grouped = scores.groupby(["UniProt_ID", "Selection_Type"]).Spearman.mean()
    category_means = grouped.groupby("Selection_Type").mean()
    average = float(category_means.mean())
    total_variants = int(scores.variants.sum())
    passed = (bool(scores.matches_frozen_assay.all())
              and total_variants == manifest["total_variants"]
              and abs(average - EXPECTED_AVERAGE) <= 1e-12)
    report = {
        "scope": "Metric re-evaluation of archived predictions; model inference was not rerun by this command.",
        "assays": len(scores), "variants": total_variants,
        "assays_matching_frozen_spearman": int(scores.matches_frozen_assay.sum()),
        "official_average": average, "displayed_average": f"{average:.3f}",
        "expected_average": EXPECTED_AVERAGE,
        "absolute_difference": abs(average - EXPECTED_AVERAGE),
        "aggregation": "Round assay Spearman to 3 decimals; mean within UniProt x selection type; mean within selection type; equal mean over selection types.",
        "category_means": category_means.to_dict(),
        "input_manifest_sha256": sha256(manifest_path),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "scipy": scipy.__version__},
        "pass": passed,
    }
    args.out.mkdir(parents=True)
    scores.to_csv(args.out / "assay_spearman.csv", index=False)
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Assays: {len(scores)}; variants: {total_variants}")
    print(f"Official average Spearman: {average:.16f} (displayed: {average:.3f})")
    print(f"Frozen assay matches: {int(scores.matches_frozen_assay.sum())}/217")
    print("PASS" if passed else "FAIL")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
