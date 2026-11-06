#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
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
    return [float(p) for p in parts]


def _alpha_tag(alpha: float) -> str:
    v = int(round(float(alpha) * 1000))
    return f"a{v:03d}"


def _gamma_tag(gamma: float) -> str:
    v = int(round(abs(float(gamma)) * 100))
    sign = "neg" if float(gamma) < 0 else "pos"
    return f"g{sign}{v:03d}"


@dataclass(frozen=True)
class DepthAwareSpec:
    a1: float
    a2: float
    a3: float  # depth>=3
    gamma: float
    col_name: str


@dataclass(frozen=True)
class NeffGateSpec:
    a_low: float
    a_med: float
    a_high: float
    gamma: float
    col_name: str


@dataclass(frozen=True)
class HybridGateSpec:
    """
    Hybrid gate: depth=1 uses a1; depth>=2 uses alpha chosen by Neff/L category (Low/Medium/High).
    """

    a1: float
    a_low: float
    a_med: float
    a_high: float
    gamma: float
    col_name: str


def _build_depthaware_specs(
    *,
    a1_list: list[float],
    a2_list: list[float],
    a3_list: list[float],
    gamma_list: list[float],
    prefix: str,
) -> list[DepthAwareSpec]:
    specs: list[DepthAwareSpec] = []
    for a1 in a1_list:
        for a2 in a2_list:
            for a3 in a3_list:
                for g in gamma_list:
                    tag = f"d1{_alpha_tag(a1)}_d2{_alpha_tag(a2)}_d3p{_alpha_tag(a3)}"
                    if abs(float(g)) > 1e-12:
                        tag = f"{tag}_{_gamma_tag(g)}"
                    specs.append(DepthAwareSpec(a1=float(a1), a2=float(a2), a3=float(a3), gamma=float(g), col_name=f"{prefix}_{tag}"))
    return specs


def _build_neffgate_specs(
    *,
    a_low_list: list[float],
    a_med_list: list[float],
    a_high_list: list[float],
    gamma_list: list[float],
    prefix: str,
) -> list[NeffGateSpec]:
    specs: list[NeffGateSpec] = []
    for a_low in a_low_list:
        for a_med in a_med_list:
            for a_high in a_high_list:
                for g in gamma_list:
                    tag = f"l{_alpha_tag(a_low)}_m{_alpha_tag(a_med)}_h{_alpha_tag(a_high)}"
                    if abs(float(g)) > 1e-12:
                        tag = f"{tag}_{_gamma_tag(g)}"
                    specs.append(
                        NeffGateSpec(
                            a_low=float(a_low),
                            a_med=float(a_med),
                            a_high=float(a_high),
                            gamma=float(g),
                            col_name=f"{prefix}_{tag}",
                        )
                    )
    return specs


def _build_hybrid_specs(
    *,
    a1_list: list[float],
    a_low_list: list[float],
    a_med_list: list[float],
    a_high_list: list[float],
    gamma_list: list[float],
    prefix: str,
) -> list[HybridGateSpec]:
    specs: list[HybridGateSpec] = []
    for a1 in a1_list:
        for a_low in a_low_list:
            for a_med in a_med_list:
                for a_high in a_high_list:
                    for g in gamma_list:
                        tag = f"d1{_alpha_tag(a1)}_l{_alpha_tag(a_low)}_m{_alpha_tag(a_med)}_h{_alpha_tag(a_high)}"
                        if abs(float(g)) > 1e-12:
                            tag = f"{tag}_{_gamma_tag(g)}"
                        specs.append(
                            HybridGateSpec(
                                a1=float(a1),
                                a_low=float(a_low),
                                a_med=float(a_med),
                                a_high=float(a_high),
                                gamma=float(g),
                                col_name=f"{prefix}_{tag}",
                            )
                        )
    return specs


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build ProteinGym score csvs from an existing WMZ+SiteRM bundle with depth-aware fusion and optional "
            "mutation-burden prior: score = student + alpha(depth) * teacher_raw + gamma * (depth-1). "
            "Optionally also emits MSA-depth (Neff/L category) gated fusion variants per assay."
        )
    )
    p.add_argument(
        "--bundle_dir",
        type=str,
        default="data/proteingym_local/zero_shot_substitutions_scores/ProteinDreamer_WMZ_SiteRM_bundle",
        help="Input directory containing <DMS_id>.csv with columns mutant, student_col, teacher_col.",
    )
    p.add_argument(
        "--ref_csv",
        type=str,
        default="external/ProteinGym-main/reference_files/DMS_substitutions.csv",
        help="ProteinGym reference file (DMS_substitutions.csv).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="data/proteingym_local/zero_shot_substitutions_scores/ProteinDreamer_WMZ_SiteRM_fusion_depthaware_v1",
        help="Output score directory (will contain <DMS_id>.csv).",
    )
    p.add_argument("--student_col", type=str, default="ProteinDreamer_WMZ_student")
    p.add_argument("--teacher_col", type=str, default="ProteinDreamer_SiteRM_teacher")

    # Baseline constant-alpha raw fusion.
    p.add_argument("--raw_alphas", type=str, default="0.30", help="Constant alpha list for raw fusion models (':' separated).")
    p.add_argument("--raw_gammas", type=str, default="0.00:-0.50", help="Gamma list for raw fusion (':' separated).")
    # Keep prefix distinct from the official best raw sweep (fusion_raw_a300) so configs can include both.
    p.add_argument("--raw_prefix", type=str, default="ProteinDreamer_WMZ_SiteRM_fusion_raw_burden")

    # Depth-aware fusion.
    p.add_argument("--depth_a1", type=str, default="0.20", help="Alpha for depth=1 (':' separated list).")
    p.add_argument("--depth_a2", type=str, default="0.30", help="Alpha for depth=2 (':' separated list).")
    p.add_argument("--depth_a3p", type=str, default="0.40", help="Alpha for depth>=3 (':' separated list).")
    p.add_argument("--depth_gammas", type=str, default="0.00:-0.50", help="Gamma list for depth-aware models (':' separated).")
    p.add_argument("--depth_prefix", type=str, default="ProteinDreamer_WMZ_SiteRM_fusion_raw_depthgate")

    # Choose alpha by the reference MSA_Neff_L_category.
    p.add_argument(
        "--neff_a_low",
        type=str,
        default="",
        help="Alpha for Low MSA_Neff_L_category assays (':' separated list). Empty disables neff-gated models.",
    )
    p.add_argument(
        "--neff_a_med",
        type=str,
        default="",
        help="Alpha for Medium MSA_Neff_L_category assays (':' separated list).",
    )
    p.add_argument(
        "--neff_a_high",
        type=str,
        default="",
        help="Alpha for High MSA_Neff_L_category assays (':' separated list).",
    )
    p.add_argument("--neff_gammas", type=str, default="0.00:-0.50", help="Gamma list for neff-gated models (':' separated).")
    p.add_argument("--neff_prefix", type=str, default="ProteinDreamer_WMZ_SiteRM_fusion_raw_neffgate")

    # Hybrid gate: fixed alpha for depth=1, Neff/L-gated alpha for depth>=2.
    p.add_argument(
        "--hybrid_a1",
        type=str,
        default="",
        help="Alpha for depth=1 in hybrid gate (':' separated list). Empty disables hybrid models.",
    )
    p.add_argument(
        "--hybrid_a_low",
        type=str,
        default="",
        help="Alpha for Low MSA_Neff_L_category assays (depth>=2) (':' separated list).",
    )
    p.add_argument(
        "--hybrid_a_med",
        type=str,
        default="",
        help="Alpha for Medium MSA_Neff_L_category assays (depth>=2) (':' separated list).",
    )
    p.add_argument(
        "--hybrid_a_high",
        type=str,
        default="",
        help="Alpha for High MSA_Neff_L_category assays (depth>=2) (':' separated list).",
    )
    p.add_argument("--hybrid_gammas", type=str, default="0.00:-0.50", help="Gamma list for hybrid-gated models (':' separated).")
    p.add_argument("--hybrid_prefix", type=str, default="ProteinDreamer_WMZ_SiteRM_fusion_raw_hybridgate")

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--include_base_cols", action="store_true", help="Include student_col and teacher_col in output files.")
    return p


def main() -> None:
    args = create_parser().parse_args()
    bundle_dir = Path(args.bundle_dir)
    ref_csv = Path(args.ref_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = pd.read_csv(ref_csv)
    if "DMS_id" not in ref.columns:
        raise ValueError(f"Reference missing DMS_id: {ref_csv}")
    if "MSA_Neff_L_category" not in ref.columns:
        raise ValueError(f"Reference missing MSA_Neff_L_category: {ref_csv}")
    dms_ids = [str(x) for x in ref["DMS_id"].tolist()]
    dms_to_neff_cat = {str(row["DMS_id"]): str(row["MSA_Neff_L_category"]) for _, row in ref.iterrows()}

    raw_alphas = _parse_float_list(str(args.raw_alphas))
    raw_gammas = _parse_float_list(str(args.raw_gammas))
    if not raw_alphas:
        raise ValueError("--raw_alphas is empty")
    if not raw_gammas:
        raise ValueError("--raw_gammas is empty")

    a1_list = _parse_float_list(str(args.depth_a1))
    a2_list = _parse_float_list(str(args.depth_a2))
    a3_list = _parse_float_list(str(args.depth_a3p))
    depth_gammas = _parse_float_list(str(args.depth_gammas))
    if not (a1_list and a2_list and a3_list and depth_gammas):
        raise ValueError("Depth-aware alpha/gamma lists are empty")

    depth_specs = _build_depthaware_specs(
        a1_list=a1_list,
        a2_list=a2_list,
        a3_list=a3_list,
        gamma_list=depth_gammas,
        prefix=str(args.depth_prefix),
    )

    neff_a_low_list = _parse_float_list(str(args.neff_a_low))
    neff_a_med_list = _parse_float_list(str(args.neff_a_med))
    neff_a_high_list = _parse_float_list(str(args.neff_a_high))
    neff_gammas = _parse_float_list(str(args.neff_gammas))
    neff_specs: list[NeffGateSpec] = []
    if any([neff_a_low_list, neff_a_med_list, neff_a_high_list]):
        if not (neff_a_low_list and neff_a_med_list and neff_a_high_list):
            raise ValueError("Neff-gate requires all of --neff_a_low/--neff_a_med/--neff_a_high (or leave all empty).")
        if not neff_gammas:
            raise ValueError("--neff_gammas is empty")
        neff_specs = _build_neffgate_specs(
            a_low_list=neff_a_low_list,
            a_med_list=neff_a_med_list,
            a_high_list=neff_a_high_list,
            gamma_list=neff_gammas,
            prefix=str(args.neff_prefix),
        )

    hybrid_a1_list = _parse_float_list(str(args.hybrid_a1))
    hybrid_a_low_list = _parse_float_list(str(args.hybrid_a_low))
    hybrid_a_med_list = _parse_float_list(str(args.hybrid_a_med))
    hybrid_a_high_list = _parse_float_list(str(args.hybrid_a_high))
    hybrid_gammas = _parse_float_list(str(args.hybrid_gammas))
    hybrid_specs: list[HybridGateSpec] = []
    if any([hybrid_a1_list, hybrid_a_low_list, hybrid_a_med_list, hybrid_a_high_list]):
        if not (hybrid_a1_list and hybrid_a_low_list and hybrid_a_med_list and hybrid_a_high_list):
            raise ValueError(
                "Hybrid gate requires all of --hybrid_a1/--hybrid_a_low/--hybrid_a_med/--hybrid_a_high (or leave all empty)."
            )
        if not hybrid_gammas:
            raise ValueError("--hybrid_gammas is empty")
        hybrid_specs = _build_hybrid_specs(
            a1_list=hybrid_a1_list,
            a_low_list=hybrid_a_low_list,
            a_med_list=hybrid_a_med_list,
            a_high_list=hybrid_a_high_list,
            gamma_list=hybrid_gammas,
            prefix=str(args.hybrid_prefix),
        )

    missing = 0
    for dms_id in dms_ids:
        src = bundle_dir / f"{dms_id}.csv"
        if not src.exists():
            missing += 1
            continue

        dst = out_dir / f"{dms_id}.csv"
        if dst.exists() and not args.overwrite:
            continue

        usecols = ["mutant", str(args.student_col), str(args.teacher_col)]
        df = pd.read_csv(src, usecols=usecols)

        mutant = df["mutant"].astype(str)
        depth = mutant.str.count(":").to_numpy(dtype=np.int32) + 1
        student = df[str(args.student_col)].to_numpy(dtype=np.float32)
        teacher = df[str(args.teacher_col)].to_numpy(dtype=np.float32)

        out = pd.DataFrame({"mutant": mutant})
        if args.include_base_cols:
            out[str(args.student_col)] = student.astype(np.float32)
            out[str(args.teacher_col)] = teacher.astype(np.float32)

        # Constant-alpha raw fusion variants.
        for alpha in raw_alphas:
            for gamma in raw_gammas:
                name = f"{str(args.raw_prefix)}_{_alpha_tag(alpha)}"
                if abs(float(gamma)) > 1e-12:
                    name = f"{name}_{_gamma_tag(gamma)}"
                out[name] = (student + float(alpha) * teacher + float(gamma) * (depth.astype(np.float32) - 1.0)).astype(np.float32)

        # Depth-aware variants.
        for spec in depth_specs:
            alpha_vec = np.where(depth == 1, spec.a1, np.where(depth == 2, spec.a2, spec.a3)).astype(np.float32)
            out[spec.col_name] = (student + alpha_vec * teacher + float(spec.gamma) * (depth.astype(np.float32) - 1.0)).astype(np.float32)

        # Neff/L category gated variants (per-assay).
        cat = dms_to_neff_cat.get(str(dms_id), "Medium")
        if cat not in {"Low", "Medium", "High"}:
            cat = "Medium"

        if neff_specs:
            for spec in neff_specs:
                if cat == "Low":
                    alpha = spec.a_low
                elif cat == "High":
                    alpha = spec.a_high
                else:
                    alpha = spec.a_med
                out[spec.col_name] = (
                    student + float(alpha) * teacher + float(spec.gamma) * (depth.astype(np.float32) - 1.0)
                ).astype(np.float32)

        if hybrid_specs:
            for spec in hybrid_specs:
                if cat == "Low":
                    a_cat = spec.a_low
                elif cat == "High":
                    a_cat = spec.a_high
                else:
                    a_cat = spec.a_med
                alpha_vec = np.where(depth == 1, float(spec.a1), float(a_cat)).astype(np.float32)
                out[spec.col_name] = (
                    student + alpha_vec * teacher + float(spec.gamma) * (depth.astype(np.float32) - 1.0)
                ).astype(np.float32)

        out.to_csv(dst, index=False)

    if missing:
        print(f"[build_depthaware] warning: missing {missing}/{len(dms_ids)} source bundle files under: {bundle_dir}")
    print(f"[build_depthaware] wrote score dir: {out_dir} (files={len(list(out_dir.glob('*.csv')))})")


if __name__ == "__main__":
    main()
