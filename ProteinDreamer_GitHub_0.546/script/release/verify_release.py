#!/usr/bin/env python3

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.world_model.msa_gate import MsaReliabilityGate  # noqa: E402
from src.world_model.zeroshot_world_model import (  # noqa: E402
    ActionStateTransition,
    RewardUncertaintyHead,
)

FAMILIES = (
    "wmz_saprot650m_retrain_v2_sd3407_e60_rs1024_pp16384_cache512",
    "wmz_saprot1p3b_retrain_v2_sd3407_e60_rs1024_pp16384_cache512",
)
MAX_GITHUB_FILE_BYTES = 100 * 1024 * 1024
PRIVATE_PATTERNS = (
    re.compile("/hpcfs" + "/fhome/"),
    re.compile(r"github_pat_[A-Za-z0-9_]+"),
    re.compile(r"ghp_[A-Za-z0-9]+"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
)


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def walk_strings(value: Any, prefix: str = "root") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, str):
        found.append((prefix, value))
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(walk_strings(item, f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(walk_strings(item, f"{prefix}[{index}]"))
    return found


def check_checkpoints() -> None:
    paths = [
        ROOT / "model" / "world_model_zeroshot" / family / f"wmz_fold{fold}.pt"
        for family in FAMILIES
        for fold in range(5)
    ]
    if any(not path.is_file() for path in paths):
        fail("expected 10 WMZ checkpoints")
    for path in paths:
        if path.stat().st_size >= MAX_GITHUB_FILE_BYTES:
            fail(f"checkpoint exceeds GitHub's 100 MiB threshold: {path}")
        obj = torch.load(path, map_location="cpu", weights_only=False)
        required = {"args", "transition_state_dict", "reward_state_dict", "msa_gate_state_dict"}
        if not isinstance(obj, dict) or not required.issubset(obj):
            fail(f"unexpected checkpoint structure: {path}")
        args = obj["args"]
        stale_args = {"gnn", "gnn_config", "gnn_model_dir", "gnn_model_name",
                      "c_alpha_max_neighbors", "mutant_name"} & set(args)
        if stale_args:
            fail(f"unused graph metadata remains in {path}: {sorted(stale_args)}")
        fold = int(path.stem.removeprefix("wmz_fold"))
        expected_plm = (
            "model/SaProt_1.3B_AFDB_OMG_NCBI"
            if "1p3b" in path.parent.name
            else "model/SaProt_650M_AF2"
        )
        expected_args = {
            "fold": fold,
            "seed": 3407,
            "backbone": "saprot",
            "plm": expected_plm,
            "plm_hidden_size": 1280,
            "epochs": 60,
            "reward_samples_per_protein": 1024,
            "pairwise_pairs": 16384,
            "wt_cache_size": 512,
            "msa_feat_mode": "full",
            "no_msa_features": False,
            "no_delta_logp_feature": False,
        }
        for key, expected in expected_args.items():
            if args.get(key) != expected:
                fail(f"unexpected {key} in {path}: {args.get(key)!r} != {expected!r}")
        use_delta_logp = not bool(args.get("no_delta_logp_feature", False))
        use_msa = bool(args.get("msa_cache_dir")) and not bool(args.get("no_msa_features", False))
        extra_feat_dim = 5 if use_msa and args.get("msa_feat_mode", "full") == "full" else (2 if use_msa else 0)
        transition = ActionStateTransition(
            hidden_dim=int(args["plm_hidden_size"]),
            aa_emb_dim=int(args.get("aa_emb_dim", 32)),
            mlp_hidden_dim=int(args.get("transition_hidden", 1024)),
            dropout=0.0,
            use_delta_logp=use_delta_logp,
            extra_feat_dim=extra_feat_dim,
        )
        reward = RewardUncertaintyHead(
            hidden_dim=int(args["plm_hidden_size"]),
            aa_emb_dim=int(args.get("aa_emb_dim", 32)),
            mlp_hidden_dim=int(args.get("reward_hidden", 1024)),
            dropout=0.0,
            use_delta_logp=use_delta_logp,
            extra_feat_dim=extra_feat_dim,
        )
        transition.load_state_dict(obj["transition_state_dict"], strict=True)
        reward.load_state_dict(obj["reward_state_dict"], strict=True)
        transition.eval()
        reward.eval()
        with torch.no_grad():
            inputs = dict(
                wt_embed=torch.zeros(2, int(args["plm_hidden_size"])),
                wt_aa_idx=torch.tensor([0, 1]),
                mut_aa_idx=torch.tensor([1, 2]),
                delta_logp=torch.zeros(2),
                extra_feat=torch.zeros(2, extra_feat_dim),
            )
            delta = transition(**inputs)
            mu, logvar = reward(mut_embed_pred=inputs["wt_embed"] + delta, **inputs)
            if not torch.isfinite(mu).all() or not torch.isfinite(logvar).all():
                fail(f"nonfinite WMZ smoke output: {path}")
        if use_msa and not bool(args.get("msa_no_gate", False)):
            gate = MsaReliabilityGate(
                hidden_dim=int(args.get("msa_gate_hidden", 64)),
                init_bias=float(args.get("msa_gate_init_bias", -10.0)),
                dropout=float(args.get("msa_gate_dropout", 0.0)),
            )
            gate.load_state_dict(obj["msa_gate_state_dict"], strict=True)
        leaks = [(key, value) for key, value in walk_strings(obj) if "/hpcfs/" in value or "wangjs" in value]
        if leaks:
            fail(f"private path in checkpoint {path}: {leaks[0]}")
    print("[ok] 10 loadable, sanitized WMZ checkpoints; synthetic forward passes finite")


def check_folds() -> None:
    root = ROOT / "data" / "proteingym_local" / "folds" / "PG_AF2_by_protein_k5_seed3407"
    proteins = [line.strip() for line in (root / "proteins.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    universe = set(proteins)
    if len(proteins) != 217 or len(universe) != 217:
        fail("frozen proteins.txt must contain 217 unique assays")
    all_valid: list[str] = []
    expected_sizes = [44, 44, 43, 43, 43]
    for fold, expected_size in enumerate(expected_sizes):
        train = {
            line.strip()
            for line in (root / f"fold{fold}_train.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        valid = [
            line.strip()
            for line in (root / f"fold{fold}_valid.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(valid) != expected_size or len(set(valid)) != expected_size:
            fail(f"unexpected fold{fold} validation size")
        if train & set(valid) or train | set(valid) != universe:
            fail(f"fold{fold} is not a disjoint train/validation partition")
        all_valid.extend(valid)
    if len(all_valid) != 217 or set(all_valid) != universe:
        fail("CV5 validation folds do not cover each assay exactly once")
    print("[ok] frozen CV5 split: 217 unique assays, valid sizes 44/44/43/43/43")


def check_hash_manifest() -> None:
    manifest = ROOT / "model" / "CHECKPOINTS.sha256"
    if not manifest.is_file():
        fail("missing model/CHECKPOINTS.sha256")
    entries = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        path = ROOT / relative
        if not path.is_file():
            fail(f"hash manifest target missing: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            fail(f"SHA-256 mismatch: {relative}")
        entries.append(relative)
    if len(entries) != 10:
        fail(f"expected 10 checkpoint hashes, found {len(entries)}")
    print("[ok] checkpoint SHA-256 manifest")


def check_syntax_and_secrets() -> None:
    checked_python = checked_json = checked_yaml = 0
    # Only packaged files; exclude downloaded assets and run outputs.
    paths = [p for p in ROOT.iterdir() if p.is_file()]
    for relative in ("src", "script", "docs", "artifacts",
                     "model/world_model_zeroshot",
                     "data/example_0546",
                     "data/proteingym_local/folds", "data/proteingym_local/reference"):
        paths.extend((ROOT / relative).rglob("*"))
    for path in paths:
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.stat().st_size >= MAX_GITHUB_FILE_BYTES:
            fail(f"file exceeds GitHub's 100 MiB threshold: {path.relative_to(ROOT)}")
        suffix = path.suffix.lower()
        if suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            checked_python += 1
        elif suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
            checked_json += 1
        elif suffix in {".yaml", ".yml"}:
            yaml.safe_load(path.read_text(encoding="utf-8"))
            checked_yaml += 1
        if suffix not in {".pt", ".pdb"}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for pattern in PRIVATE_PATTERNS:
                if pattern.search(text):
                    fail(f"private path or credential-like text in {path.relative_to(ROOT)}")
    print(f"[ok] syntax: python={checked_python} json={checked_json} yaml={checked_yaml}")


def check_saprot_only_scope() -> None:
    forbidden = (
        "action_predict.py", "epistasis_predict.py", "train_action_head.py",
        "train_epistasis_head.py", "train_world_model_rollout.py",
        "train_world_model_zeroshot.py", "zeroshot_scan_local.py",
        "src/models.py", "src/data.py", "src/config", "src/dataset", "src/module",
        "src/world_model/action_head.py", "src/world_model/epistasis_head.py",
        "src/world_model/rollout_world_model.py", "src/world_model/set_encoder.py",
        "data/mutant_example", "facebook",
    )
    present = [relative for relative in forbidden if (ROOT / relative).exists()]
    if present:
        fail(f"non-0.546 model content remains: {present}")
    root_python = {path.name for path in ROOT.glob("*.py")}
    expected_python = {"zeroshot_predict.py", "train_wmz_saprot.py"}
    if root_python != expected_python:
        fail(f"unexpected top-level Python entry points: {sorted(root_python)}")
    for required in ("src/saprot.py", "src/saprot_dataset.py",
                     "script/train_0546.sh",
                     "script/proteingym/protein_dreamer_proteingym_config_best_0546.json"):
        if not (ROOT / required).is_file():
            fail(f"missing SaProt-only release file: {required}")
    print("[ok] release scope: only SaProt-backed 0.546 training/inference entry points")


def check_evidence() -> None:
    evidence = ROOT / "artifacts" / "proteingym_0.546"
    with (evidence / "model_comparison.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["Model_name"] == "ProteinDreamer_final_lite_b130"]
    if len(selected) != 1 or float(selected[0]["Average_Spearman"]) != 0.546:
        fail("model_comparison.csv does not identify b130 at 0.546")
    with (evidence / "paired_bootstrap_final_vs_baseline.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    if abs(float(row["official_avg_a"]) - 0.5455137705176768) > 1e-12:
        fail("unexpected unrounded official score")
    if int(row["bootstrap_n"]) != 20000:
        fail("unexpected paired bootstrap count")
    with (evidence / "dms_level_spearman.csv").open(newline="", encoding="utf-8") as handle:
        assay_count = sum(1 for _ in csv.DictReader(handle))
    if assay_count != 217:
        fail(f"expected 217 assay-level rows, found {assay_count}")
    print("[ok] result evidence: displayed=0.546 assays=217 bootstrap=20000")


def main() -> None:
    try:
        from script.setup.verify_benchmark_inputs import check_bundle
        if len(check_bundle(ROOT / "data/structures_0546")) != 217:
            fail("expected 217 historical structures")
        print("[ok] historical structure archive and per-assay SHA-256")
        check_checkpoints()
        check_folds()
        check_hash_manifest()
        check_saprot_only_scope()
        check_syntax_and_secrets()
        check_evidence()
    except (SyntaxError, json.JSONDecodeError, yaml.YAMLError, OSError, ValueError) as exc:
        fail(str(exc))
    print("[PASS] ProteinDreamer GitHub 0.546 release checks passed")


if __name__ == "__main__":
    main()
