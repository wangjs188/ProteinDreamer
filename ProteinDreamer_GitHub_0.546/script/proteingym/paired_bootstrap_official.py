#!/usr/bin/env python
"""Paired bootstrap for the ProteinGym average Spearman.

Average assays within UniProt ID and selection type, then proteins within each
selection type, then the five types with equal weight. Resample UniProt IDs
within each selection type. Input: the evaluator's per-assay Spearman table.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "UniProt ID" in df.columns and "UniProt_ID" not in df.columns:
        rename["UniProt ID"] = "UniProt_ID"
    if "Selection Type" in df.columns and "Selection_Type" not in df.columns:
        rename["Selection Type"] = "Selection_Type"
    if "DMS ID" in df.columns and "DMS_id" not in df.columns:
        rename["DMS ID"] = "DMS_id"
    if rename:
        df = df.rename(columns=rename)
    return df


def official_average_from_dms_level(
    df_dms_level: pd.DataFrame, model_col: str
) -> Tuple[float, pd.DataFrame]:
    """
    Returns:
      official_avg: float
      uniprot_by_type: DataFrame with columns [Selection_Type, UniProt_ID, value]
    """
    df = _canon_cols(df_dms_level)
    for c in ["Selection_Type", "UniProt_ID"]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in DMS-level csv.")
    if model_col not in df.columns:
        raise ValueError(f"Missing model_col '{model_col}' in DMS-level csv.")

    # UniProt_ID x Selection_Type mean across assays
    uniprot_by_type = (
        df.groupby(["UniProt_ID", "Selection_Type"], dropna=False)[model_col]
        .mean()
        .reset_index()
        .rename(columns={model_col: "value"})
    )

    # Selection_Type mean across UniProt_ID
    type_means = uniprot_by_type.groupby("Selection_Type")["value"].mean()
    official_avg = float(type_means.mean())
    return official_avg, uniprot_by_type


@dataclass
class BootstrapResult:
    delta_mean: float
    ci95_lo: float
    ci95_hi: float
    p_one_sided: float
    p_two_sided: float
    n_boot: int
    categories: list[str]


def paired_bootstrap_official_delta(
    uniprot_by_type_a: pd.DataFrame,
    uniprot_by_type_b: pd.DataFrame,
    n_boot: int = 20000,
    seed: int = 0,
) -> BootstrapResult:
    """
    Paired bootstrap for delta = A - B under ProteinGym official aggregation.

    We bootstrap UniProt_ID within each Selection_Type independently, then average
    category means equally (like ProteinGym).
    """
    rng = np.random.default_rng(seed)

    # Align by (UniProt_ID, Selection_Type) intersection.
    a = uniprot_by_type_a.copy()
    b = uniprot_by_type_b.copy()
    key_cols = ["UniProt_ID", "Selection_Type"]
    a = a[key_cols + ["value"]].rename(columns={"value": "a"})
    b = b[key_cols + ["value"]].rename(columns={"value": "b"})
    m = a.merge(b, on=key_cols, how="inner")

    cats = sorted([c for c in m["Selection_Type"].dropna().unique().tolist()])
    if not cats:
        raise ValueError("No Selection_Type categories found after merge.")

    cat_frames: Dict[str, pd.DataFrame] = {c: m[m["Selection_Type"] == c].reset_index(drop=True) for c in cats}
    for c, dfc in cat_frames.items():
        if dfc.empty:
            raise ValueError(f"Empty category after merge: {c}")

    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        cat_means = []
        for c in cats:
            dfc = cat_frames[c]
            idx = rng.integers(0, len(dfc), size=len(dfc))
            samp = dfc.iloc[idx]
            cat_means.append(float((samp["a"] - samp["b"]).mean()))
        deltas[i] = float(np.mean(cat_means))

    delta_mean = float(np.mean(deltas))
    lo, hi = np.percentile(deltas, [2.5, 97.5])

    # one-sided p-value for improvement (A > B)
    p_one = float((deltas <= 0).mean())
    p_two = float(2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))

    return BootstrapResult(
        delta_mean=delta_mean,
        ci95_lo=float(lo),
        ci95_hi=float(hi),
        p_one_sided=p_one,
        p_two_sided=min(1.0, p_two),
        n_boot=int(n_boot),
        categories=cats,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Paired bootstrap (official ProteinGym aggregation).")
    p.add_argument("--dms_level_csv", required=True, help="ProteinGym DMS-level Spearman CSV.")
    p.add_argument("--model_a", required=True, help="Column name for model A (numerator).")
    p.add_argument("--model_b", required=True, help="Column name for model B (baseline).")
    p.add_argument("--bootstrap_n", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_csv", default=None, help="Optional output CSV to write a one-row summary.")
    args = p.parse_args()

    df = pd.read_csv(args.dms_level_csv)
    avg_a, uni_a = official_average_from_dms_level(df, args.model_a)
    avg_b, uni_b = official_average_from_dms_level(df, args.model_b)
    res = paired_bootstrap_official_delta(uni_a, uni_b, n_boot=args.bootstrap_n, seed=args.seed)

    row = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "official_avg_a": avg_a,
        "official_avg_b": avg_b,
        "delta_a_minus_b": avg_a - avg_b,
        "bootstrap_delta_mean": res.delta_mean,
        "ci95_lo": res.ci95_lo,
        "ci95_hi": res.ci95_hi,
        "p_one_sided(a>b)": res.p_one_sided,
        "p_two_sided": res.p_two_sided,
        "bootstrap_n": res.n_boot,
        "categories": "|".join(res.categories),
    }

    out = pd.DataFrame([row])
    if args.out_csv:
        out.to_csv(args.out_csv, index=False)
        print(f"[paired_bootstrap_official] wrote: {args.out_csv}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

