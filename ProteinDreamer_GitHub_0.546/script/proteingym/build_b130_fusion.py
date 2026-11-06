#!/usr/bin/env python3
"""b130 fusion: fuse each branch, write to CSV, then average.

The CSV round trip preserves the rounding used in the reported scores.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORE = "ProteinDreamer_WMZ_SiteRM_fusion_raw_neffgate_la350_ma300_ha250_gneg050"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle650", required=True, type=Path)
    p.add_argument("--bundle13", required=True, type=Path)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    out = args.out.resolve()
    branches = out.with_name(out.name + "_branches")
    if out.exists() or branches.exists():
        raise SystemExit("Fusion output/intermediate directory exists; use a new --out.")
    for tag, bundle in (("650m", args.bundle650), ("1p3b", args.bundle13)):
        subprocess.run([
            sys.executable, "-B", str(ROOT / "script/proteingym/build_wmz_siterm_fusion_depthaware_from_bundle.py"),
            "--bundle_dir", str(bundle.resolve()), "--ref_csv", str(args.reference.resolve()),
            "--out_dir", str(branches / tag), "--raw_alphas", "0.30", "--raw_gammas", "0.00",
            "--depth_a1", "0.20", "--depth_a2", "0.30", "--depth_a3p", "0.40", "--depth_gammas", "0.00",
            "--neff_a_low", "0.35", "--neff_a_med", "0.30", "--neff_a_high", "0.25",
            "--neff_gammas=-0.50", "--include_base_cols",
        ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, "-B", str(ROOT / "script/proteingym/ensemble_score_dirs.py"),
        "--in_dirs", str(branches / "650m"), str(branches / "1p3b"),
        "--col", SCORE, "--out_dir", str(out),
    ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
