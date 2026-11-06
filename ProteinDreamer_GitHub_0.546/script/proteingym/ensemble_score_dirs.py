#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _parse_float_list(text: str) -> list[float]:
    if not text:
        return []
    parts: list[str] = []
    for p in str(text).replace(",", ":").split(":"):
        p = p.strip()
        if p:
            parts.append(p)
    out: list[float] = []
    for p in parts:
        for q in p.split():
            q = q.strip()
            if q:
                out.append(float(q))
    return out


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Ensemble multiple ProteinGym score directories by averaging (optionally weighted) a score column across dirs. "
            "Writes <DMS_id>.csv files into --out_dir."
        )
    )
    p.add_argument("--in_dirs", type=str, nargs="+", required=True, help="Input score directories (each contains *.csv).")
    p.add_argument("--col", type=str, required=True, help="Score column to ensemble (must exist in each CSV).")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory to write ensembled CSVs.")
    p.add_argument(
        "--out_col",
        type=str,
        default="",
        help="Output score column name (default: same as --col).",
    )
    p.add_argument(
        "--weights",
        type=str,
        default="",
        help=(
            "Optional weights (one per --in_dirs). Accepts ':'/','/whitespace-separated list. "
            "If provided, output = sum_i w_i * score_i (weights are normalized to sum to 1)."
        ),
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files (default: skip).")
    return p


def main() -> None:
    args = create_parser().parse_args()
    in_dirs = [Path(d).resolve() for d in args.in_dirs]
    col = str(args.col)
    out_col = str(args.out_col).strip() or col
    weights = _parse_float_list(str(args.weights))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for d in in_dirs:
        if not d.is_dir():
            raise SystemExit(f"--in_dirs contains non-dir: {d}")

    if weights:
        if len(weights) != len(in_dirs):
            raise SystemExit(f"--weights length ({len(weights)}) must match --in_dirs ({len(in_dirs)})")
        s = float(sum(weights))
        if abs(s) < 1e-12:
            raise SystemExit("--weights sum to 0")
        weights = [float(w) / s for w in weights]
    else:
        weights = [1.0 / float(len(in_dirs))] * len(in_dirs)

    files = sorted(in_dirs[0].glob("*.csv"))
    if not files:
        raise SystemExit(f"No *.csv found in: {in_dirs[0]}")

    missing_in: List[str] = []
    for p in files:
        for d in in_dirs[1:]:
            if not (d / p.name).exists():
                missing_in.append(f"{d.name}/{p.name}")
                break
    if missing_in:
        raise SystemExit(f"Missing {len(missing_in)} input CSVs (showing first 10): {missing_in[:10]}")

    for i, p0 in enumerate(files, start=1):
        p_out = out_dir / p0.name
        if p_out.exists() and (not bool(args.overwrite)):
            continue

        # Keep all columns from the first dir; only ensemble the requested score col.
        df0 = pd.read_csv(p0)
        if "mutant" not in df0.columns:
            raise SystemExit(f"Missing 'mutant' col in {p0}")
        if col not in df0.columns:
            raise SystemExit(f"Missing col {col!r} in {p0}")

        merged = df0[["mutant", col]].rename(columns={col: f"{col}__0"})
        base = df0.drop(columns=[col]).copy()

        for j, d in enumerate(in_dirs[1:], start=1):
            dfj = pd.read_csv(d / p0.name, usecols=["mutant", col], dtype={"mutant": "string"})
            if col not in dfj.columns:
                raise SystemExit(f"Missing col {col!r} in {d / p0.name}")
            if set(df0["mutant"]) != set(dfj["mutant"]):
                raise SystemExit(f"Mutant coverage differs: {p0} vs {d / p0.name}")
            merged = merged.merge(
                dfj.rename(columns={col: f"{col}__{j}"}),
                on="mutant",
                how="inner",
                validate="one_to_one",
            )

        score_cols = [c for c in merged.columns if c.startswith(f"{col}__")]
        if len(score_cols) != len(weights):
            raise SystemExit(f"Internal error: expected {len(weights)} score cols, got {len(score_cols)}")
        merged[out_col] = 0.0
        for w, sc in zip(weights, score_cols):
            merged[out_col] = merged[out_col] + float(w) * merged[sc].astype(float)
        out_df = base.merge(merged[["mutant", out_col]], on="mutant", how="inner", validate="one_to_one")
        out_df.to_csv(p_out, index=False)

        if i % 25 == 0:
            print(f"[ensemble] wrote {i}/{len(files)}: {p_out}")

    print(f"[ensemble] done: out_dir={out_dir} out_col={out_col} weights={weights} in_dirs={[str(d) for d in in_dirs]}")


if __name__ == "__main__":
    main()
