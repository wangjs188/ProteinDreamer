#!/usr/bin/env python3
"""One-assay inference check against archived b130 scores."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ASSAY = "A0A1I9GEU1_NEIME_Kennouche_2019"
FINALCOL = "ProteinDreamer_WMZ_SiteRM_SaProt_add_neffgate_wmz_saprot_plmens2_AF2only_b130"
FUSIONCOL = "ProteinDreamer_WMZ_SiteRM_fusion_raw_neffgate_la350_ma300_ha250_gneg050"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, default=ROOT / "data/example_0546")
    parser.add_argument("--out", type=Path, default=ROOT / "result/example_0546")
    parser.add_argument("--plm650", type=Path, default=ROOT / "model/SaProt_650M_AF2")
    parser.add_argument("--plm13", type=Path, default=ROOT / "model/SaProt_1.3B_AFDB_OMG_NCBI")
    parser.add_argument("--foldseek-bin", default=os.environ.get("FOLDSEEK_BIN", "foldseek"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pll-batch-size", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-5,
                        help="Absolute tolerance; exact equality is reported separately.")
    args = parser.parse_args()
    args.example, args.out = args.example.resolve(), args.out.resolve()
    args.plm650, args.plm13 = args.plm650.resolve(), args.plm13.resolve()
    if args.out.exists():
        raise SystemExit("Output already exists; choose a new --out.")
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; run on a GPU node or explicitly use --device cpu.")
    provenance = json.loads((args.example / "PROVENANCE.json").read_text())
    for relative, record in provenance["copied_inputs"].items():
        path = args.example / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise SystemExit(f"Example checksum mismatch: {relative}")
    for model in (args.plm650, args.plm13):
        if not (model / "config.json").exists():
            raise SystemExit(f"Download SaProt first: {model}")
    args.out.mkdir(parents=True)
    dataset = args.out / "dataset"
    shutil.copytree(args.example / "dataset", dataset)
    logs = args.out / "logs"
    logs.mkdir()
    command_record = []
    def run(label, script, *arguments):
        command = [sys.executable, "-B", str(ROOT / script), *map(str, arguments)]
        print(f"[run] {label}", flush=True)
        command_record.append({"stage": label, "command": command})
        with (logs / f"{label}.log").open("w") as handle:
            proc = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                  env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        if proc.returncode:
            print((logs / f"{label}.log").read_text()[-4000:])
            raise SystemExit(f"{label} failed; inspect {logs / (label + '.log')}")

    for label, model, tag in (
        ("wmz650", args.plm650, "wmz_saprot650m_retrain_v2_sd3407_e60_rs1024_pp16384_cache512"),
        ("wmz13", args.plm13, "wmz_saprot1p3b_retrain_v2_sd3407_e60_rs1024_pp16384_cache512"),
    ):
        run(label, "script/proteingym/score_wmz_siterm_fusion_cv5.py",
            "--seed", 3407, "--device", args.device, "--dataset_root", dataset,
            "--struct_root", dataset, "--folds_dir", args.example / "folds", "--fold", 3,
            "--ckpt_dir", ROOT / "model/world_model_zeroshot" / tag,
            "--plm", model, "--backbone", "saprot", "--saprot_scoring", "fast",
            "--foldseek_bin", args.foldseek_bin, "--teacher_cache_dir", args.example / "teacher_cache",
            "--alphas", "0.10", "--msa_cache_dir", args.example / "msa_cache",
            "--msa_feat_mode", "full", "--msa_gate_hidden", 64, "--msa_gate_init_bias", -10.0,
            "--msa_gate_dropout", 0.0, "--chunk_size", 32768, "--out_dir", args.out / label)
    run("fusion", "script/proteingym/build_b130_fusion.py",
        "--bundle650", args.out / "wmz650", "--bundle13", args.out / "wmz13",
        "--reference", args.example / "reference.csv", "--out", args.out / "fusion")
    run("baseline", "zeroshot_predict.py",
        "--device", args.device, "--backbone", "saprot", "--saprot_scoring", "pll",
        "--ensemble_dtype", "float32",
        "--pll_batch_size", args.pll_batch_size, "--foldseek_bin", args.foldseek_bin,
        "--saprot_models", args.plm650, args.plm13, "--mutant_dataset_dir", dataset,
        "--mutant_score_col", "DMS_score", "--result_dir", args.out / "baseline")
    run("final", "script/proteingym/build_wmz_siterm_baseline_add_sweep_from_dirs.py",
        "--wmz_dir", args.out / "fusion", "--baseline_dir", args.out / "baseline",
        "--out_dir", args.out / "final", "--ref_csv", args.example / "reference.csv",
        "--wmz_col", FUSIONCOL, "--baseline_col", "SaProt_ensemble", "--betas", "0.13",
        "--out_prefix", FINALCOL.rsplit("_b130", 1)[0])
    expected = pd.read_csv(args.example / "expected_scores.csv").set_index("mutant")
    metrics = {}
    for label, (directory, col) in {
        "wmz650": ("wmz650", "ProteinDreamer_WMZ_student"),
        "wmz13": ("wmz13", "ProteinDreamer_WMZ_student"),
        "teacher": ("wmz650", "ProteinDreamer_SiteRM_teacher"),
        "saprot_pll": ("baseline", "SaProt_ensemble"),
        "final_b130": ("final", FINALCOL),
    }.items():
        actual = pd.read_csv(args.out / directory / f"{ASSAY}.csv").set_index("mutant")
        if not actual.index.is_unique or set(actual.index) != set(expected.index):
            raise ValueError(f"{label}: mutant coverage differs from frozen results")
        a = actual.loc[expected.index, col].to_numpy(dtype=np.float64)
        b = expected[label].to_numpy(dtype=np.float64)
        if not np.isfinite(a).all():
            raise ValueError(f"{label}: nonfinite predictions")
        diff = np.abs(a - b)
        metrics[label] = {
            "n_variants": len(a), "max_abs_difference": float(diff.max()),
            "mean_abs_difference": float(diff.mean()), "exact_equal_count": int((a == b).sum()),
            "within_atol_count": int((diff <= args.atol).sum()),
            "spearman_vs_frozen": float(pd.Series(a).corr(pd.Series(b), method="spearman")),
        }
        print(f"[compare] {label}: {json.dumps(metrics[label])}", flush=True)
    report = {
        "assay": ASSAY, "recipe": "complete b130, held-out fold 3",
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
        "atol": args.atol, "metrics": metrics,
        "pass_within_tolerance": all(v["within_atol_count"] == v["n_variants"] for v in metrics.values()),
        "exact_numeric_equality": all(v["exact_equal_count"] == v["n_variants"] for v in metrics.values()),
    }
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "commands.json").write_text(json.dumps(command_record, indent=2) + "\n")
    print(f"[result] {args.out / 'comparison.json'}")
    if not report["pass_within_tolerance"]:
        raise SystemExit("Comparison exceeds tolerance; investigate before claiming numerical equivalence.")


if __name__ == "__main__":
    main()
