#!/usr/bin/env python3
import csv
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


class ReviewerWorkflowTests(unittest.TestCase):
    def run_script(self, path, *args):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run([sys.executable, "-B", str(ROOT / path), *map(str, args)],
                              cwd=ROOT, env=env, text=True, capture_output=True)

    def test_environment_versions(self):
        a = yaml.safe_load((ROOT / "environment.yaml").read_text())
        self.assertEqual(a["name"], "proteindreamer0546")
        self.assertIn("python=3.10.13", a["dependencies"])

    def test_historical_cache_policy(self):
        from script.setup.verify_external_assets import check_caches
        ref = pd.read_csv(ROOT / "data/proteingym_local/reference/DMS_substitutions.csv")
        msa = ROOT / "data/proteingym_local/msa_cache_dms_v1"
        teacher = ROOT / "data/proteingym_local/teacher_siterm_dms_s4096_e100_q64_reg05_v2t1"
        problems = []
        check_caches(ref.to_dict("records"), msa, teacher, problems)
        self.assertEqual(problems, [])
        with tempfile.TemporaryDirectory(prefix="pd-cache-policy-") as tmp:
            d = Path(tmp)
            (d / "msa").mkdir()
            (d / "teacher").mkdir()
            problems = []
            check_caches([{"DMS_id": "not_an_exception", "target_seq": "A"}],
                         d / "msa", d / "teacher", problems)
            self.assertEqual(len(problems), 2)
            assay = "PSAE_PICP2_Tsuboyama_2023_1PSE"
            (d / "msa" / (assay + ".npz")).write_bytes(b"unexpected replacement")
            problems = []
            check_caches([{"DMS_id": assay, "target_seq": "A"}],
                         d / "msa", d / "teacher", problems)
            self.assertTrue(any("must retain its fallback" in x for x in problems))

    def test_historical_fallback_loader_values(self):
        import numpy as np
        from script.setup.verify_external_assets import HISTORICAL_FALLBACK_IDS
        from script.proteingym.score_wmz_siterm_fusion_cv5 import (
            _load_msa_cache_item, _load_teacher_cache_any,
        )
        ref = pd.read_csv(ROOT / "data/proteingym_local/reference/DMS_substitutions.csv").set_index("DMS_id")
        for assay in HISTORICAL_FALLBACK_IDS:
            length = len(ref.loc[assay, "target_seq"])
            msa = _load_msa_cache_item(str(ROOT / "data/proteingym_local/msa_cache_dms_v1"),
                                       assay, expected_len=length)
            teacher = _load_teacher_cache_any(
                str(ROOT / "data/proteingym_local/teacher_siterm_dms_s4096_e100_q64_reg05_v2t1"),
                assay, expected_len=length)
            for values in (msa, teacher):
                self.assertEqual(values[0].shape, (length, 20))
                self.assertTrue(np.all(values[0] == np.float32(-np.log(20.0))))
                for value in values[1:]:
                    self.assertTrue(np.all(np.asarray(value) == 0))

    def test_training_msa_covers_all_217_assays(self):
        import math
        import numpy as np
        import torch
        from train_wmz_saprot import HISTORICAL_NO_MSA_CACHE_IDS, load_msa
        from script.setup.verify_external_assets import HISTORICAL_FALLBACK_IDS

        self.assertEqual(HISTORICAL_NO_MSA_CACHE_IDS, HISTORICAL_FALLBACK_IDS)
        ref = pd.read_csv(ROOT / "data/proteingym_local/reference/DMS_substitutions.csv")
        self.assertEqual(len(ref), 217)
        directory = ROOT / "data/proteingym_local/msa_cache_dms_v1"
        for row in ref.itertuples():
            with self.subTest(assay=row.DMS_id):
                length = len(row.target_seq)
                path = directory / f"{row.DMS_id}.npz"
                actual = load_msa(path, length)
                if row.DMS_id in HISTORICAL_NO_MSA_CACHE_IDS:
                    self.assertFalse(path.exists())
                    expected = (
                        torch.full((length, 20), -math.log(20.0), dtype=torch.float16),
                        torch.zeros(length, dtype=torch.float16),
                        torch.zeros(length, dtype=torch.float16),
                        torch.zeros(length, dtype=torch.uint8), 0.0,
                    )
                else:
                    with np.load(path, allow_pickle=False) as cache:
                        keys = ("log_probs_msa_full", "msa_cov_full", "msa_entropy_full", "msa_has_full")
                        expected = tuple(torch.tensor(cache[key], dtype=dtype) for key, dtype in zip(
                            keys, (torch.float16, torch.float16, torch.float16, torch.uint8)))
                        expected += (float(cache["log_neff_global"]),)
                for value, target in zip(actual[:-1], expected[:-1]):
                    self.assertEqual(value.dtype, target.dtype)
                    self.assertTrue(torch.equal(value, target))
                self.assertEqual(actual[-1], expected[-1])

    def test_training_msa_still_rejects_missing_or_corrupt_inputs(self):
        from train_wmz_saprot import load_msa
        with tempfile.TemporaryDirectory(prefix="pd-training-msa-") as tmp:
            directory = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                load_msa(directory / "unexpected_missing_assay.npz", 68)
            known = "PSAE_PICP2_Tsuboyama_2023_1PSE.npz"
            with self.assertRaises(FileNotFoundError):
                load_msa(directory / "missing_directory" / known, 68)
            path = directory / known
            path.write_bytes(b"corrupt cache")
            with self.assertRaises(ValueError):
                load_msa(path, 68)

    def test_training_no_msa_features_are_zero(self):
        import torch
        from train_wmz_saprot import HISTORICAL_NO_MSA_CACHE_IDS, WildTypeFeatures, load_msa, msa_features
        from src.world_model.msa_gate import MsaReliabilityGate

        gate = MsaReliabilityGate(hidden_dim=64, init_bias=-10.0, dropout=0.0)
        ref = pd.read_csv(ROOT / "data/proteingym_local/reference/DMS_substitutions.csv").set_index("DMS_id")
        for assay in HISTORICAL_NO_MSA_CACHE_IDS:
            with self.subTest(assay=assay):
                length = len(ref.loc[assay, "target_seq"])
                lp, coverage, entropy, has, neff = load_msa(
                    ROOT / "data/proteingym_local/msa_cache_dms_v1" / f"{assay}.npz", length)
                item = WildTypeFeatures(
                    embeddings=torch.zeros(length, 4), log_probs=torch.zeros(length, 20),
                    msa_log_probs=lp, msa_coverage=coverage, msa_entropy=entropy,
                    msa_has=has, log_neff=neff, plddt=torch.full((length,), 90.0))
                extra = msa_features(item, gate, torch.tensor([0, length - 1]),
                                     torch.tensor([0, 1]), torch.tensor([1, 2]),
                                     device=torch.device("cpu"), dtype=torch.float32)
                self.assertTrue(torch.equal(extra, torch.zeros(2, 5)))

    def test_pinned_evaluator(self):
        from script.setup.verify_external_assets import check_evaluator
        snapshot = json.loads((ROOT / "external/ProteinGym-main/SNAPSHOT.json").read_text())
        self.assertTrue({"LICENSE", "proteingym/merge.py",
                         "proteingym/performance_DMS_benchmarks.py", "proteingym/constants.json"}
                        .issubset(snapshot["sha256"]))
        problems = []
        check_evaluator(ROOT / "external/ProteinGym-main", problems)
        self.assertEqual(problems, [])

    def test_metric_rejects_corrupted_input(self):
        manifest = (ROOT / "data/metric_0546/PROVENANCE.json").read_text()
        first = json.loads(manifest)["assays"][0]["file"]
        with tempfile.TemporaryDirectory(prefix="pd-metric-integrity-") as tmp:
            d = Path(tmp)
            (d / "inputs").mkdir()
            (d / "inputs/PROVENANCE.json").write_text(manifest)
            (d / "inputs" / first).write_bytes(b"corrupted")
            run = self.run_script("script/proteingym/reproduce_metric_0546.py",
                                  "--inputs", d / "inputs", "--out", d / "out")
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("checksum mismatch", run.stderr)
            self.assertFalse((d / "out").exists())

    def test_saprot_only_release_scope(self):
        from src.saprot_dataset import load_structure_sequence, resolve_structure_path
        assay = "A0A1I9GEU1_NEIME_Kennouche_2019"
        example = ROOT / "data/example_0546"
        sequence = load_structure_sequence(resolve_structure_path(example / "dataset", assay))
        reference = pd.read_csv(example / "reference.csv").iloc[0]
        self.assertEqual(sequence, reference.target_seq)
        run = self.run_script(
            "zeroshot_predict.py", "--backbone", "not_saprot",
            "--mutant_dataset_dir", example / "dataset", "--result_dir", example / "invalid")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("invalid choice", run.stderr)
        for relative in ("src/models.py", "src/module", "src/config", "src/dataset",
                         "action_predict.py", "epistasis_predict.py",
                         "train_world_model_rollout.py", "zeroshot_scan_local.py"):
            self.assertFalse((ROOT / relative).exists(), relative)

    def test_training_launcher_is_saprot_cv5_only(self):
        env = dict(os.environ, DRY_RUN="1", RUN_EVAL="0", PY=sys.executable)
        run = subprocess.run(
            ["bash", str(ROOT / "script/train_0546.sh")], cwd=ROOT, env=env,
            text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout.count("train_wmz_saprot.py"), 10)
        self.assertEqual(run.stdout.count("family=650m"), 5)
        self.assertEqual(run.stdout.count("family=1p3b"), 5)
        self.assertNotIn("--backbone", run.stdout)

        commands = [shlex.split(line) for line in run.stdout.splitlines()
                    if "train_wmz_saprot.py" in shlex.split(line)]
        self.assertEqual(len(commands), 10)
        expected = {
            "--struct-chain-id": "A",
            "--plddt-mask": "auto",
            "--plddt-threshold": "70.0",
            "--device": "cuda",
            "--seed": "3407",
            "--epochs": "60",
            "--proteins-per-epoch": "999999",
            "--reward-samples-per-protein": "1024",
            "--pairwise-pairs": "16384",
            "--wt-cache-size": "512",
            "--max-rows-per-protein": "50000",
            "--max-eval-rows": "0",
            "--eval-every": "1",
            "--lr": "0.0002",
            "--weight-decay": "0.01",
            "--dropout": "0.1",
        }
        folds_by_model = {"SaProt_650M_AF2": set(),
                          "SaProt_1.3B_AFDB_OMG_NCBI": set()}
        for command in commands:
            self.assertEqual(command[0:3], [sys.executable, "-B", "train_wmz_saprot.py"])
            values = {command[i]: command[i + 1] for i in range(3, len(command), 2)}
            for option, value in expected.items():
                self.assertEqual(values[option], value, option)
            model_name = Path(values["--plm"]).name
            self.assertIn(model_name, folds_by_model)
            folds_by_model[model_name].add(int(values["--fold"]))
            self.assertEqual(Path(values["--dataset-root"]).name,
                             "ProteinGym_v1_SaProt_AF2_v2")
            self.assertEqual(Path(values["--folds-dir"]).name,
                             "PG_AF2_by_protein_k5_seed3407")
            self.assertEqual(Path(values["--msa-cache-dir"]).name,
                             "msa_cache_dms_v1")
        self.assertEqual(folds_by_model,
                         {"SaProt_650M_AF2": set(range(5)),
                          "SaProt_1.3B_AFDB_OMG_NCBI": set(range(5))})

    def test_frozen_reference_folds(self):
        ref = pd.read_csv(ROOT / "data/proteingym_local/reference/DMS_substitutions.csv")
        folds = ROOT / "data/proteingym_local/folds/PG_AF2_by_protein_k5_seed3407"
        membership = {x: i for i in range(5) for x in (folds / f"fold{i}_valid.txt").read_text().splitlines()}
        self.assertEqual(set(ref.DMS_id), set(membership))
        groups = ref.assign(fold=ref.DMS_id.map(membership)).groupby("UniProt_ID").fold.nunique()
        self.assertEqual(len(groups), 186)
        self.assertEqual(int((groups > 1).sum()), 20)

    def test_raw_converter_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="pd-reviewer-") as tmp:
            d = Path(tmp)
            raw = d / "raw"
            raw.mkdir()
            (raw / "assay.csv").write_text("mutant,mutated_sequence,DMS_score\nA1V,V,0.5\n")
            (raw / "protein.pdb").write_text("HEADER    SYNTHETIC TEST FILE\nEND\n")
            (d / "ref.csv").write_text("DMS_id,DMS_filename,pdb_file\nassay,assay.csv,protein.pdb\n")
            args = ("--raw-dir", raw, "--reference", d / "ref.csv",
                    "--dataset", d / "dataset", "--flat-dms", d / "flat")
            run = self.run_script("script/setup/prepare_proteingym_raw.py", *args)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual((raw / "assay.csv").read_bytes(),
                             (d / "dataset/DATASET/assay/assay.csv").read_bytes())
            self.assertEqual((raw / "protein.pdb").read_bytes(),
                             (d / "dataset/DATASET/assay/assay.pdb").read_bytes())
            rerun = self.run_script("script/setup/prepare_proteingym_raw.py", *args)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertIn("Output already exists", rerun.stderr)

    def test_score_coverage_rejects_missing_and_nonfinite(self):
        with tempfile.TemporaryDirectory(prefix="pd-coverage-") as tmp:
            d = Path(tmp)
            (d / "flat").mkdir()
            (d / "scores/model").mkdir(parents=True)
            (d / "ref.csv").write_text("DMS_id\nassay\n")
            (d / "flat/assay.csv").write_text("mutant\nA1V\nA1L\n")
            (d / "config.json").write_text(json.dumps({"model_list_zero_shot_substitutions_DMS": {
                "test": {"location": "model", "input_score_name": "score"}}}))
            args = ("--reference", d / "ref.csv", "--flat-dms", d / "flat",
                    "--scores-root", d / "scores", "--config", d / "config.json")
            path = d / "scores/model/assay.csv"
            path.write_text("mutant,score\nA1V,1\nA1L,2\n")
            good = self.run_script("script/setup/verify_score_coverage.py", *args)
            self.assertEqual(good.returncode, 0, good.stderr)
            for bad in ("mutant,score\nA1V,1\n", "mutant,score\nA1V,NaN\nA1L,2\n"):
                path.write_text(bad)
                run = self.run_script("script/setup/verify_score_coverage.py", *args)
                self.assertNotEqual(run.returncode, 0)

    def test_b130_float32_ensemble(self):
        import argparse
        from zeroshot_predict import _ensemble_cols
        with tempfile.TemporaryDirectory(prefix="pd-ensemble-") as tmp:
            path = Path(tmp) / "assay.csv"
            path.write_text("mutant,m0,m1\nF1I,-7.048122406005859,-6.667111396789551\n")
            args = argparse.Namespace(result_dir=tmp, mutant_score_col="unused", ensemble_dtype="float32")
            _ensemble_cols(args, cols=["m0", "m1"], out_col="SaProt_ensemble")
            # Float32 CSV rounding in the b130 baseline.
            value = pd.read_csv(path, dtype=str).iloc[0]["SaProt_ensemble"]
            self.assertEqual(value, "-6.857617")

    def test_complete_fusion_matches_frozen_exactly(self):
        # b130 order: fuse -> CSV -> average -> residual.
        example = ROOT / "data/example_0546"
        frozen = pd.read_csv(example / "expected_scores.csv", dtype=str)
        with tempfile.TemporaryDirectory(prefix="pd-fusion-") as tmp:
            d = Path(tmp)
            filename = "A0A1I9GEU1_NEIME_Kennouche_2019.csv"
            for label in ("wmz650", "wmz13"):
                (d / label).mkdir()
                frame = frozen[["mutant", label, "teacher"]].rename(columns={
                    label: "ProteinDreamer_WMZ_student", "teacher": "ProteinDreamer_SiteRM_teacher"})
                frame.to_csv(d / label / filename, index=False)
            run = self.run_script("script/proteingym/build_b130_fusion.py",
                                  "--bundle650", d / "wmz650", "--bundle13", d / "wmz13",
                                  "--reference", example / "reference.csv", "--out", d / "fusion")
            self.assertEqual(run.returncode, 0, run.stderr)
            (d / "baseline").mkdir()
            frozen[["mutant", "saprot_pll"]].rename(columns={"saprot_pll": "SaProt_ensemble"}).to_csv(
                d / "baseline" / filename, index=False)
            prefix = "ProteinDreamer_WMZ_SiteRM_SaProt_add_neffgate_wmz_saprot_plmens2_AF2only"
            run = self.run_script("script/proteingym/build_wmz_siterm_baseline_add_sweep_from_dirs.py",
                                  "--wmz_dir", d / "fusion", "--baseline_dir", d / "baseline",
                                  "--out_dir", d / "final", "--ref_csv", example / "reference.csv",
                                  "--wmz_col", "ProteinDreamer_WMZ_SiteRM_fusion_raw_neffgate_la350_ma300_ha250_gneg050",
                                  "--baseline_col", "SaProt_ensemble", "--betas", "0.13",
                                  "--out_prefix", prefix)
            self.assertEqual(run.returncode, 0, run.stderr)
            actual = pd.read_csv(d / "final" / filename).set_index("mutant")
            expected = pd.read_csv(example / "expected_scores.csv").set_index("mutant")
            self.assertTrue((actual.loc[expected.index, prefix + "_b130"] == expected.final_b130).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
