#!/usr/bin/env python3
"""ProteinDreamer training on frozen SaProt features, one fold per run."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.saprot import SaProtBackbone  # noqa: E402
from src.saprot_dataset import (  # noqa: E402
    ProteinGymSaProtDataset,
    resolve_structure_path,
)
from src.struct.foldseek_3di import get_3di_tokens_cached  # noqa: E402
from src.utils.mutation import AA_ORDER, parse_mutant  # noqa: E402
from src.world_model.msa_gate import MsaReliabilityGate  # noqa: E402
from src.world_model.zeroshot_world_model import (  # noqa: E402
    ActionStateTransition,
    RewardUncertaintyHead,
    SingleMutation,
)


# Three historical no-cache assays use the uniform-MSA fallback.
HISTORICAL_NO_MSA_CACHE_IDS = frozenset({
    "F7YBW8_MESOW_Ding_2023",
    "PSAE_PICP2_Tsuboyama_2023_1PSE",
    "Q6WV12_9MAXI_Somermeyer_2022",
})


@dataclass
class WildTypeFeatures:
    embeddings: torch.Tensor
    log_probs: torch.Tensor
    msa_log_probs: torch.Tensor
    msa_coverage: torch.Tensor
    msa_entropy: torch.Tensor
    msa_has: torch.Tensor
    log_neff: float
    plddt: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_names(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"Expected a non-empty list of unique assay names: {path}")
    return names


def load_mutations(
    dataset_root: Path,
    protein: str,
    sequence: str,
    *,
    mutant_col: str,
    label_col: str,
    max_rows: int,
) -> list[SingleMutation]:
    base = dataset_root / "DATASET" / protein
    csv_path, tsv_path = base / f"{protein}.csv", base / f"{protein}.tsv"
    if csv_path.is_file():
        frame = pd.read_csv(csv_path)
    elif tsv_path.is_file():
        frame = pd.read_table(tsv_path)
    else:
        raise FileNotFoundError(f"Missing mutation table for {protein}")
    missing = {mutant_col, label_col} - set(frame.columns)
    if missing:
        raise ValueError(f"{protein}: missing columns {sorted(missing)}")
    frame = frame.dropna(subset=[label_col]).reset_index(drop=True)
    mutations: list[SingleMutation] = []
    for name, label in zip(frame[mutant_col].astype(str), frame[label_col].astype(float)):
        parsed = parse_mutant(name, offset_idx=1)
        if len(parsed) != 1:
            continue
        item = parsed[0]
        if 0 <= item.pos < len(sequence) and sequence[item.pos] == item.wt:
            mutations.append(SingleMutation(
                pos=int(item.pos), wt_aa_idx=int(item.wt_idx),
                mut_aa_idx=int(item.mt_idx), y=float(label)))
        if max_rows and len(mutations) >= max_rows:
            break
    return mutations


def rank_zscore(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, num=values.size, dtype=np.float32)
    return (ranks - float(ranks.mean())) / (float(ranks.std(ddof=0)) + eps)


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    value = spearmanr(y_true, y_pred).correlation
    return 0.0 if value is None or np.isnan(value) else float(value)


def stable_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def pairwise_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    pairs: int,
    seed: int,
) -> torch.Tensor:
    count = int(scores.numel())
    if count < 2:
        return torch.zeros((), device=scores.device, dtype=scores.dtype)
    generator = torch.Generator(device=scores.device)
    generator.manual_seed(seed)
    left = torch.randint(0, count, (pairs,), generator=generator, device=scores.device)
    right = torch.randint(0, count, (pairs,), generator=generator, device=scores.device)
    keep = left != right
    left, right = left[keep], right[keep]
    signs = torch.sign(labels[left] - labels[right])
    keep = signs != 0
    if not bool(keep.any()):
        return torch.zeros((), device=scores.device, dtype=scores.dtype)
    left, right, signs = left[keep], right[keep], signs[keep]
    return nn.functional.softplus(-signs * (scores[left] - scores[right])).mean()


def extract_plddt(path: Path, chain_id: str, expected_len: int) -> torch.Tensor:
    from Bio.PDB import MMCIFParser, PDBParser

    parser = MMCIFParser(QUIET=True) if path.suffix.lower() == ".cif" else PDBParser(QUIET=True)
    model = parser.get_structure("protein", str(path))[0]
    if chain_id not in model:
        raise ValueError(f"{path}: chain {chain_id!r} not found")
    values = []
    for residue in model[chain_id]:
        atoms = [float(atom.get_bfactor()) for atom in residue.get_atoms()]
        values.append(float(np.mean(atoms)) if atoms else 0.0)
    if len(values) != expected_len:
        raise ValueError(f"{path}: pLDDT length {len(values)} != sequence length {expected_len}")
    return torch.tensor(values, dtype=torch.float16)


def load_msa(path: Path, expected_len: int) -> tuple[torch.Tensor, ...]:
    if not path.is_file():
        if (path.stem not in HISTORICAL_NO_MSA_CACHE_IDS
                or not path.parent.is_dir() or path.exists()):
            raise FileNotFoundError(path)
        # Match the historical training tensors, including their float16 dtype.
        return (
            torch.full((expected_len, 20), -math.log(20.0), dtype=torch.float16),
            torch.zeros(expected_len, dtype=torch.float16),
            torch.zeros(expected_len, dtype=torch.float16),
            torch.zeros(expected_len, dtype=torch.uint8),
            0.0,
        )
    with np.load(path) as values:
        required = {
            "log_probs_msa_full", "msa_cov_full", "msa_entropy_full",
            "msa_has_full", "msa_reliable_full", "log_neff_global",
        }
        missing = required - set(values.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        log_probs = np.asarray(values["log_probs_msa_full"], dtype=np.float32)
        coverage = np.asarray(values["msa_cov_full"], dtype=np.float32)
        entropy = np.asarray(values["msa_entropy_full"], dtype=np.float32)
        has = np.asarray(values["msa_has_full"], dtype=np.uint8)
        log_neff = float(values["log_neff_global"])
    if log_probs.shape != (expected_len, 20):
        raise ValueError(f"{path}: MSA log-probability shape {log_probs.shape} != {(expected_len, 20)}")
    for name, array in (("coverage", coverage), ("entropy", entropy), ("has", has)):
        if array.shape != (expected_len,):
            raise ValueError(f"{path}: MSA {name} shape {array.shape} != {(expected_len,)}")
    arrays = (log_probs, coverage, entropy)
    if not all(np.isfinite(array).all() for array in arrays) or not np.isfinite(log_neff):
        raise ValueError(f"{path}: non-finite MSA features")
    return (
        torch.tensor(log_probs, dtype=torch.float16),
        torch.tensor(coverage, dtype=torch.float16),
        torch.tensor(entropy, dtype=torch.float16),
        torch.tensor(has, dtype=torch.uint8),
        log_neff,
    )


def forward_backbone(
    model: SaProtBackbone,
    graph: Data,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(graph.to(device))
    embeddings = output.esm_rep.detach()
    log_probs = output.saprot_logits[:, :20].log_softmax(dim=-1).detach()
    return embeddings, log_probs


def msa_features(
    item: WildTypeFeatures,
    gate: MsaReliabilityGate,
    positions: torch.Tensor,
    wt_indices: torch.Tensor,
    mutant_indices: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    msa_lp = item.msa_log_probs.to(device=device, dtype=dtype)
    coverage_all = item.msa_coverage.to(device=device, dtype=dtype)
    entropy_all = item.msa_entropy.to(device=device, dtype=dtype)
    has_all = item.msa_has.to(device=device, dtype=dtype)
    plddt_all = item.plddt.to(device=device, dtype=dtype)
    delta = (msa_lp[positions, mutant_indices] - msa_lp[positions, wt_indices]).detach()
    coverage, entropy = coverage_all[positions], entropy_all[positions]
    plddt = plddt_all[positions]
    log_neff = torch.full_like(coverage, item.log_neff)
    reliability = gate(log_neff=log_neff, cov=coverage, entropy=entropy, plddt=plddt)
    reliability = (reliability * has_all[positions]).clamp(0.0, 1.0)
    raw = torch.stack([delta, coverage, entropy, log_neff], dim=-1)
    return torch.cat([reliability.view(-1, 1) * raw, reliability.view(-1, 1)], dim=-1)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--plm", type=Path, required=True)
    parser.add_argument("--msa-cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--foldseek-bin", default="foldseek")
    parser.add_argument("--struct-root", type=Path)
    parser.add_argument("--struct-cache-dir", type=Path)
    parser.add_argument("--struct-chain-id", default="A")
    parser.add_argument("--plddt-mask", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--plddt-threshold", type=float, default=70.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--proteins-per-epoch", type=int, default=999999)
    parser.add_argument("--reward-samples-per-protein", type=int, default=1024)
    parser.add_argument("--pairwise-pairs", type=int, default=16384)
    parser.add_argument("--wt-cache-size", type=int, default=512)
    parser.add_argument("--max-rows-per-protein", type=int, default=50000)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mutant-col", default="mutant")
    parser.add_argument("--label-col", default="DMS_score")
    return parser


def main() -> None:
    args = create_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; training requires a GPU allocation.")
    if args.epochs < 1 or args.reward_samples_per_protein < 2 or args.pairwise_pairs < 1:
        raise ValueError("Invalid training schedule")
    dataset_root = args.dataset_root.resolve()
    struct_root = (args.struct_root or dataset_root).resolve()
    folds_dir = args.folds_dir.resolve()
    msa_cache_dir = args.msa_cache_dir.resolve()
    out_dir = args.out_dir.resolve()
    checkpoint = out_dir / f"wmz_fold{args.fold}.pt"
    metrics = out_dir / f"wmz_fold{args.fold}_metrics.csv"
    metrics_last = out_dir / f"wmz_fold{args.fold}_metrics_last.csv"
    existing = [path for path in (checkpoint, metrics, metrics_last) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite training outputs: {existing}")

    train_names = read_names(folds_dir / f"fold{args.fold}_train.txt")
    valid_names = read_names(folds_dir / f"fold{args.fold}_valid.txt")
    if set(train_names) & set(valid_names):
        raise ValueError(f"fold {args.fold}: train/validation overlap")
    set_seed(args.seed)
    device = torch.device(args.device)

    selected = train_names + valid_names
    dataset = ProteinGymSaProtDataset(dataset_root, selected)
    graphs: dict[str, Data] = {}
    for index in range(len(dataset)):
        graph = dataset[index]
        graphs[str(graph.protein_name)] = graph
    struct_cache = (args.struct_cache_dir or dataset_root / "CACHE" / "3di").resolve()
    for protein, graph in graphs.items():
        structure = resolve_structure_path(struct_root, protein)
        tokens = get_3di_tokens_cached(
            protein_name=protein,
            struct_path=str(structure),
            foldseek_bin=args.foldseek_bin,
            cache_dir=str(struct_cache),
            chain_id=args.struct_chain_id,
            plddt_mask=args.plddt_mask,
            plddt_threshold=args.plddt_threshold,
        )
        if int(tokens.numel()) != int(graph.y.numel()):
            raise ValueError(f"{protein}: 3Di and sequence lengths differ")
        graph.struct_tokens = tokens

    model = SaProtBackbone(str(args.plm.resolve()), device=args.device).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    hidden_size = int(model.model.config.hidden_size)
    transition = ActionStateTransition(
        hidden_dim=hidden_size, aa_emb_dim=32, mlp_hidden_dim=1024,
        dropout=args.dropout, use_delta_logp=True, extra_feat_dim=5).to(device)
    reward = RewardUncertaintyHead(
        hidden_dim=hidden_size, aa_emb_dim=32, mlp_hidden_dim=1024,
        dropout=args.dropout, use_delta_logp=True, extra_feat_dim=5).to(device)
    gate = MsaReliabilityGate(hidden_dim=64, init_bias=-10.0, dropout=0.0).to(device)
    trainable = list(transition.parameters()) + list(reward.parameters()) + list(gate.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    wt_cache: dict[str, WildTypeFeatures] = {}
    wt_order: list[str] = []

    def get_wild_type(protein: str) -> WildTypeFeatures:
        if protein in wt_cache:
            wt_order.remove(protein)
            wt_order.append(protein)
            return wt_cache[protein]
        embeddings, log_probs = forward_backbone(model, graphs[protein], device)
        length = int(embeddings.shape[0])
        msa_lp, coverage, entropy, has, log_neff = load_msa(
            msa_cache_dir / f"{protein}.npz", length)
        structure = resolve_structure_path(dataset_root, protein)
        item = WildTypeFeatures(
            embeddings=embeddings.cpu().half(),
            log_probs=log_probs.cpu().half(),
            msa_log_probs=msa_lp,
            msa_coverage=coverage,
            msa_entropy=entropy,
            msa_has=has,
            log_neff=log_neff,
            plddt=extract_plddt(structure, args.struct_chain_id, length),
        )
        wt_cache[protein] = item
        wt_order.append(protein)
        while len(wt_order) > args.wt_cache_size:
            wt_cache.pop(wt_order.pop(0), None)
        return item

    mutation_cache: dict[str, list[SingleMutation]] = {}

    def get_mutations(protein: str) -> list[SingleMutation]:
        if protein not in mutation_cache:
            sequence = "".join(AA_ORDER[int(index)] for index in graphs[protein].y.tolist())
            mutation_cache[protein] = load_mutations(
                dataset_root, protein, sequence,
                mutant_col=args.mutant_col, label_col=args.label_col,
                max_rows=args.max_rows_per_protein)
        return mutation_cache[protein]

    def predict(protein: str, mutations: list[SingleMutation]) -> tuple[torch.Tensor, torch.Tensor]:
        wt = get_wild_type(protein)
        embeddings = wt.embeddings.to(device=device, dtype=torch.float32)
        log_probs = wt.log_probs.to(device=device, dtype=torch.float32)
        positions = torch.tensor([item.pos for item in mutations], device=device)
        wt_indices = torch.tensor([item.wt_aa_idx for item in mutations], device=device)
        mutant_indices = torch.tensor([item.mut_aa_idx for item in mutations], device=device)
        wt_embeddings = embeddings[positions]
        delta_logp = (log_probs[positions, mutant_indices] -
                      log_probs[positions, wt_indices]).detach()
        extra = msa_features(
            wt, gate, positions, wt_indices, mutant_indices,
            device=device, dtype=wt_embeddings.dtype)
        delta = transition(
            wt_embed=wt_embeddings, wt_aa_idx=wt_indices,
            mut_aa_idx=mutant_indices, delta_logp=delta_logp,
            extra_feat=extra)
        predicted = wt_embeddings + delta
        return reward(
            wt_embed=wt_embeddings, mut_embed_pred=predicted,
            wt_aa_idx=wt_indices, mut_aa_idx=mutant_indices,
            delta_logp=delta_logp, extra_feat=extra)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_valid = -1e9
    for epoch in range(1, args.epochs + 1):
        transition.train()
        reward.train()
        gate.train()
        sampled = train_names.copy()
        random.shuffle(sampled)
        sampled = sampled[:min(len(sampled), args.proteins_per_epoch)]
        total_loss, steps = 0.0, 0
        for protein in sampled:
            available = get_mutations(protein)
            if len(available) < 2:
                continue
            picked = random.sample(available, min(len(available), args.reward_samples_per_protein))
            labels_np = np.asarray([item.y for item in picked], dtype=np.float32)
            labels = torch.tensor(rank_zscore(labels_np), device=device)
            mean, log_variance = predict(protein, picked)
            rank_loss = pairwise_loss(
                mean, labels, pairs=args.pairwise_pairs, seed=args.seed + epoch)
            nll = 0.5 * (torch.exp(-log_variance) * (labels - mean) ** 2 + log_variance)
            loss = rank_loss + 0.1 * nll.mean() + 1e-3 * torch.exp(log_variance).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1

        if epoch % args.eval_every:
            continue
        transition.eval()
        reward.eval()
        gate.eval()
        rows = []
        with torch.no_grad():
            for protein in valid_names:
                available = get_mutations(protein)
                if len(available) < 10:
                    continue
                if args.max_eval_rows and len(available) > args.max_eval_rows:
                    generator = random.Random(
                        args.seed + 1009 * args.fold + stable_int(protein))
                    selected_mutations = generator.sample(available, args.max_eval_rows)
                else:
                    selected_mutations = available
                mean, _ = predict(protein, selected_mutations)
                truth = np.asarray([item.y for item in selected_mutations], dtype=np.float32)
                score = spearman(truth, mean.detach().cpu().numpy().astype(np.float32))
                rows.append({"protein": protein, "n": len(selected_mutations),
                             "spearman_wmz": score, "spearman_student": score})
        mean_valid = float(np.mean([row["spearman_wmz"] for row in rows])) if rows else 0.0
        average_loss = total_loss / max(steps, 1)
        pd.DataFrame(rows).to_csv(metrics_last, index=False)
        print(f"epoch={epoch} loss={average_loss:.4f} valid_mean_spearman={mean_valid:.4f}",
              flush=True)
        if mean_valid > best_valid:
            best_valid = mean_valid
            pd.DataFrame(rows).to_csv(metrics, index=False)
            saved_args = vars(args).copy()
            saved_args.update({
                "backbone": "saprot", "plm": str(args.plm),
                "plm_hidden_size": hidden_size, "msa_feat_mode": "full",
                "msa_gate_hidden": 64, "msa_gate_init_bias": -10.0,
                "msa_gate_dropout": 0.0, "no_msa_features": False,
                "no_delta_logp_feature": False, "w_rank": 1.0,
                "w_nll": 0.1, "w_sigma_reg": 0.001, "w_distill": 0.0,
                "distill_samples_per_protein": 0, "single_mut_only": True,
            })
            torch.save({
                "args": saved_args,
                "transition_state_dict": transition.state_dict(),
                "reward_state_dict": reward.state_dict(),
                "msa_gate_state_dict": gate.state_dict(),
                "teacher_alpha": 0.0,
                "teacher_alpha_learned": False,
                "best_valid_mean_spearman": best_valid,
            }, checkpoint)
            print(f"saved {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
