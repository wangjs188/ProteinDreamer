#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _parse_float_list(text: str) -> List[float]:
    if not text:
        return []
    parts: List[str] = []
    for p in str(text).replace(",", ":").split(":"):
        p = p.strip()
        if p:
            parts.append(p)
    out: List[float] = []
    for p in parts:
        for q in p.split():
            q = q.strip()
            if q:
                out.append(float(q))
    return out


def _beta_tag(beta: float) -> str:
    v = int(round(float(beta) * 1000))
    return f"b{v:03d}"


def _read_ref_dms_ids(ref_csv: Path) -> list[str]:
    df = pd.read_csv(ref_csv)
    if "DMS_id" in df.columns:
        return df["DMS_id"].astype(str).tolist()
    if "DMS ID" in df.columns:
        return df["DMS ID"].astype(str).tolist()
    raise ValueError(f"Could not find DMS_id column in ref_csv: {ref_csv}")


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build a single ProteinGym score directory containing a beta sweep for baseline-add fusion:\n\n"
            "score(beta) = wmz_score + beta * baseline_score\n\n"
            "This avoids creating one directory per beta; instead each per-assay CSV contains multiple "
            "output columns (one per beta), and ProteinGym config can point multiple models to the same location."
        )
    )
    p.add_argument("--wmz_dir", type=str, required=True, help="Input score dir containing <DMS_id>.csv with wmz_col.")
    p.add_argument(
        "--baseline_dir",
        type=str,
        required=True,
        help="Baseline score dir containing <DMS_id>.csv with baseline_col (e.g., SaProt_ensemble).",
    )
    p.add_argument("--out_dir", type=str, required=True, help="Output score dir (will contain <DMS_id>.csv).")
    p.add_argument("--ref_csv", type=str, required=True, help="ProteinGym reference file with DMS_id list.")
    p.add_argument("--wmz_col", type=str, required=True)
    p.add_argument("--baseline_col", type=str, required=True)
    p.add_argument("--betas", type=str, required=True, help="Beta list (':'/','/whitespace-separated).")
    p.add_argument(
        "--out_prefix",
        type=str,
        required=True,
        help="Output column prefix; final col is <out_prefix>_<bXXX>.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--include_base_cols", action="store_true", help="Include wmz_col and baseline_col in output files.")
    return p


def main() -> None:
    args = create_parser().parse_args()
    wmz_dir = Path(args.wmz_dir)
    baseline_dir = Path(args.baseline_dir)
    out_dir = Path(args.out_dir)
    ref_csv = Path(args.ref_csv)

    if not wmz_dir.is_dir():
        raise FileNotFoundError(f"--wmz_dir does not exist: {wmz_dir}")
    if not baseline_dir.is_dir():
        raise FileNotFoundError(f"--baseline_dir does not exist: {baseline_dir}")
    if not ref_csv.is_file():
        raise FileNotFoundError(f"--ref_csv does not exist: {ref_csv}")

    betas = _parse_float_list(str(args.betas))
    if not betas:
        raise ValueError("--betas is empty")

    dms_ids = _read_ref_dms_ids(ref_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = 0
    for dms_id in dms_ids:
        p_wmz = wmz_dir / f"{dms_id}.csv"
        p_base = baseline_dir / f"{dms_id}.csv"
        if (not p_wmz.exists()) or (not p_base.exists()):
            missing += 1
            continue

        p_out = out_dir / f"{dms_id}.csv"
        if p_out.exists() and not args.overwrite:
            continue

        df_w = pd.read_csv(p_wmz, usecols=["mutant", str(args.wmz_col)])
        df_b = pd.read_csv(p_base, usecols=["mutant", str(args.baseline_col)])

        if set(df_w["mutant"]) != set(df_b["mutant"]):
            raise ValueError(f"Mutant coverage mismatch: {p_wmz} vs {p_base}")
        df = df_w.merge(df_b, on="mutant", how="inner", validate="one_to_one")
        mutant = df["mutant"].astype(str)
        wmz = df[str(args.wmz_col)].to_numpy(dtype=np.float32)
        base = df[str(args.baseline_col)].to_numpy(dtype=np.float32)

        out = pd.DataFrame({"mutant": mutant})
        if args.include_base_cols:
            out[str(args.wmz_col)] = wmz.astype(np.float32)
            out[str(args.baseline_col)] = base.astype(np.float32)

        for beta in betas:
            col = f"{str(args.out_prefix)}_{_beta_tag(beta)}"
            out[col] = (wmz + float(beta) * base).astype(np.float32)

        out.to_csv(p_out, index=False)

    if missing:
        raise RuntimeError(f"Missing {missing}/{len(dms_ids)} input files (wmz/base)")
    print(f"[baseline_add_sweep] wrote: {out_dir} (files={len(list(out_dir.glob('*.csv')))}, betas={len(betas)})")


if __name__ == "__main__":
    main()
