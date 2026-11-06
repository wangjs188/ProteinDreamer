#!/usr/bin/env python3
"""Prediction coverage and finite-score checks."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--flat-dms", required=True, type=Path)
    p.add_argument("--scores-root", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args()
    models = json.loads(args.config.read_text())["model_list_zero_shot_substitutions_DMS"]
    ref = pd.read_csv(args.reference)
    for assay in ref["DMS_id"]:
        truth = pd.read_csv(args.flat_dms / f"{assay}.csv", usecols=["mutant"])
        if truth["mutant"].isna().any() or truth["mutant"].duplicated().any():
            raise ValueError(f"{assay}: invalid/duplicated input mutant keys")
        for name, model in models.items():
            path = args.scores_root / model["location"] / f"{assay}.csv"
            col = model["input_score_name"]
            scores = pd.read_csv(path, usecols=["mutant", col])
            if scores["mutant"].duplicated().any() or set(scores["mutant"]) != set(truth["mutant"]):
                raise ValueError(f"{name}/{assay}: mutant coverage mismatch")
            if not np.isfinite(pd.to_numeric(scores[col], errors="coerce")).all():
                raise ValueError(f"{name}/{assay}: nonfinite prediction")
    print(f"[PASS] Exact mutant coverage and finite scores: {len(ref)} assays × {len(models)} models")


if __name__ == "__main__":
    main()
