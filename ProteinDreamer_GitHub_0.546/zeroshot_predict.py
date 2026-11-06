#!/usr/bin/env python3
"""SaProt PLL scoring with AF2 structures."""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from transformers import logging

from src.saprot import SaProtBackbone
from src.saprot_dataset import ProteinGymSaProtDataset, resolve_structure_path
from src.struct.foldseek_3di import get_3di_tokens_cached
from src.utils.mutation import AA_ORDER, AA_TO_IDX, parse_mutant

logging.set_verbosity_error()


def mutation_score(mutant: str, sequence: str, log_probs: np.ndarray) -> float:
    score = 0.0
    for mutation in parse_mutant(str(mutant), offset_idx=1):
        if mutation.pos < 0 or mutation.pos >= len(sequence):
            raise ValueError(f"Mutation position is outside the structure: {mutant}")
        if sequence[mutation.pos] != mutation.wt:
            raise ValueError(f"Wild-type residue mismatch: {mutant}")
        score += float(log_probs[mutation.pos, AA_TO_IDX[mutation.mt]]
                       - log_probs[mutation.pos, AA_TO_IDX[mutation.wt]])
    return score


def attach_3di(data, args) -> None:
    protein = data.protein_name[0]
    structure = resolve_structure_path(args.mutant_dataset_dir, protein)
    cache = args.struct_cache_dir or os.path.join(args.mutant_dataset_dir, "CACHE", "3di")
    try:
        tokens = get_3di_tokens_cached(
            protein_name=protein,
            struct_path=str(structure),
            foldseek_bin=args.foldseek_bin,
            cache_dir=cache,
            chain_id=args.struct_chain_id,
            plddt_mask=args.plddt_mask,
            plddt_threshold=args.plddt_threshold,
        )
    except Exception as exc:
        raise RuntimeError(f"{protein}: Foldseek 3Di extraction failed") from exc
    if tokens.numel() != int(data.y.numel()):
        raise ValueError(f"{protein}: 3Di length {tokens.numel()} != sequence length {data.y.numel()}")
    data.struct_tokens = tokens


def load_mutations(root: str, protein: str) -> pd.DataFrame:
    directory = Path(root) / "DATASET" / protein
    tsv = directory / f"{protein}.tsv"
    csv = directory / f"{protein}.csv"
    if tsv.is_file():
        return pd.read_table(tsv)
    if csv.is_file():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing mutation table for {protein}: {tsv} or {csv}")


@torch.no_grad()
def score_model(args, model: SaProtBackbone, loader: DataLoader, score_name: str) -> None:
    summary: list[float] = []
    for data in loader:
        protein = data.protein_name[0]
        attach_3di(data, args)
        sequence = "".join(AA_ORDER[int(i)] for i in data.y.tolist())
        frame = load_mutations(args.mutant_dataset_dir, protein)
        if args.mutant_pos_col not in frame.columns:
            raise ValueError(f"{protein}: missing mutation column {args.mutant_pos_col}")
        positions = sorted({
            mutation.pos
            for value in frame[args.mutant_pos_col].astype(str)
            for mutation in parse_mutant(value, offset_idx=1)
            if 0 <= mutation.pos < len(sequence)
        })
        log_probs = model.saprot_pll_log_probs(
            data, positions=positions, batch_size=args.pll_batch_size).cpu().numpy()
        frame[score_name] = [
            mutation_score(value, sequence, log_probs)
            for value in frame[args.mutant_pos_col].astype(str)
        ]
        output = Path(args.result_dir) / f"{protein}.csv"
        if output.exists():
            previous = pd.read_csv(output)
            if (args.mutant_pos_col not in previous.columns
                    or not previous[args.mutant_pos_col].equals(frame[args.mutant_pos_col])):
                raise ValueError(f"{protein}: mutation grid changed between SaProt branches")
            previous[score_name] = frame[score_name]
            frame = previous
        frame.to_csv(output, index=False)
        if args.mutant_score_col in frame.columns:
            value = spearmanr(frame[args.mutant_score_col], frame[score_name]).correlation
            summary.append(0.0 if value is None or math.isnan(value) else float(value))
        print(f">>> {protein}: variants={len(frame)}")
    if summary:
        print(f">>> {score_name} average Spearman: {float(np.mean(summary))}")


def _ensemble_cols(args, *, cols: Sequence[str], out_col: str) -> None:
    scores: list[float] = []
    for path in tqdm(sorted(Path(args.result_dir).glob("*.csv")), desc=f"Ensemble -> {out_col}"):
        frame = pd.read_csv(path)
        missing = [column for column in cols if column not in frame.columns]
        if missing:
            raise ValueError(f"{path.name}: missing SaProt columns {missing}")
        # b130 averages the two branches in float32.
        frame[out_col] = frame[list(cols)].astype(args.ensemble_dtype).mean(axis=1)
        frame.to_csv(path, index=False)
        if args.mutant_score_col in frame.columns:
            value = spearmanr(frame[args.mutant_score_col], frame[out_col]).correlation
            scores.append(0.0 if value is None or math.isnan(value) else float(value))
    if scores:
        print(f">>> {out_col} average Spearman: {float(np.mean(scores))}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", choices=["saprot"], default="saprot",
                        help="This release contains only the SaProt backbone.")
    parser.add_argument("--plm", default="model/SaProt_650M_AF2")
    parser.add_argument("--saprot_models", nargs="*", default=[])
    parser.add_argument("--saprot_scoring", choices=["pll"], default="pll")
    parser.add_argument("--pll_batch_size", type=int, default=1)
    parser.add_argument("--ensemble_dtype", choices=["float32"], default="float32")
    parser.add_argument("--mutant_dataset_dir", required=True)
    parser.add_argument("--mutant_pos_col", default="mutant")
    parser.add_argument("--mutant_score_col", default="DMS_score")
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--foldseek_bin", default="foldseek")
    parser.add_argument("--struct_cache_dir", default="")
    parser.add_argument("--struct_chain_id", default="A")
    parser.add_argument("--plddt_mask", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--plddt_threshold", type=float, default=70.0)
    return parser


def main() -> None:
    args = create_parser().parse_args()
    output = Path(args.result_dir)
    if output.exists():
        raise SystemExit(f"Output directory already exists: {output}")
    output.mkdir(parents=True)
    dataset = ProteinGymSaProtDataset(args.mutant_dataset_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    models = list(args.saprot_models) or [args.plm]
    columns: list[str] = []
    for index, model_dir in enumerate(models):
        column = f"SaProt_m{index}_s"
        columns.append(column)
        model = SaProtBackbone(str(model_dir), device=args.device).eval()
        score_model(args, model, loader, column)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _ensemble_cols(args, cols=columns, out_col="SaProt_ensemble")


if __name__ == "__main__":
    main()
