#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from transformers import logging

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.saprot import SaProtBackbone  # noqa: E402
from src.saprot_dataset import ProteinGymSaProtDataset  # noqa: E402
from src.struct.foldseek_3di import get_3di_tokens_cached  # noqa: E402
from src.struct.vocab import FOLDSEEK_3DI_VOCAB  # noqa: E402
from src.utils.mutation import AA_ORDER, AA_TO_IDX, parse_mutant  # noqa: E402
from src.world_model.msa_gate import MsaReliabilityGate  # noqa: E402
from src.world_model.zeroshot_world_model import (  # noqa: E402
    ActionStateTransition,
    RewardUncertaintyHead,
)
from script.proteingym.dms_id_aliases import canonicalize_dms_id  # noqa: E402

logging.set_verbosity_error()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _stable_int(text: str) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _read_list(path: str) -> List[str]:
    items: List[str] = []
    with open(path, "r", encoding="utf-8") as r:
        for line in r:
            s = line.strip()
            if s:
                items.append(s)
    return items


def _parse_alphas(text: str) -> List[float]:
    # Colons avoid the comma separator in sbatch --export.
    import re  # noqa: PLC0415

    parts = [p for p in re.split(r"[,\s:]+", str(text).strip()) if p]
    return [float(p) for p in parts] if parts else [0.1]


def _load_mutant_df(dataset_root: str, protein: str) -> pd.DataFrame:
    tsv = os.path.join(dataset_root, "DATASET", protein, f"{protein}.tsv")
    csv = os.path.join(dataset_root, "DATASET", protein, f"{protein}.csv")
    if os.path.exists(tsv):
        return pd.read_table(tsv)
    if os.path.exists(csv):
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing mutant file: {tsv} or {csv}")


def _struct_candidates(struct_root: str, protein_name: str) -> List[str]:
    r = str(struct_root).rstrip("/")
    p = str(protein_name)
    return [
        os.path.join(r, "DATASET", p, f"{p}.pdb"),
        os.path.join(r, "DATASET", p, f"{p}.cif"),
        os.path.join(r, p, f"{p}.pdb"),
        os.path.join(r, p, f"{p}.cif"),
        os.path.join(r, f"{p}.pdb"),
        os.path.join(r, f"{p}.cif"),
    ]


def _resolve_struct_path(struct_root: str, protein_name: str) -> str:
    for cand in _struct_candidates(struct_root, protein_name):
        if os.path.exists(cand):
            return cand
    return ""


@torch.no_grad()
def _forward_backbone(
    *,
    plm_model: SaProtBackbone,
    graph: Data,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    graph = graph.to(device)
    graph = plm_model(graph)
    logits, embeds = graph.saprot_logits, graph.esm_rep
    log_probs = logits[:, :20].log_softmax(dim=-1)
    seq = "".join([AA_ORDER[i] for i in graph.y.tolist()])
    return embeds.detach(), log_probs.detach(), seq


def _rank_zscore(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if y.size < 2:
        return np.zeros_like(y, dtype=np.float32)
    if float(np.max(y) - np.min(y)) < 1e-12:
        return np.zeros_like(y, dtype=np.float32)
    order = np.argsort(y, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, num=y.size, dtype=np.float32)
    return (ranks - float(ranks.mean())) / (float(ranks.std(ddof=0)) + float(eps))


def _load_teacher_cache(
    teacher_cache_dir: str,
    protein: str,
    *,
    expected_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (log_probs[L,20], has[L], reliable[L]).
    Missing cache -> uniform log-probs and has/reliable=0.
    """
    log_uniform = float(-math.log(20.0))
    lp = np.full((expected_len, 20), log_uniform, dtype=np.float32)
    has = np.zeros((expected_len,), dtype=np.uint8)
    reliable = np.zeros((expected_len,), dtype=np.float32)
    if not teacher_cache_dir:
        return lp, has, reliable

    path = Path(teacher_cache_dir) / f"{protein}.npz"
    if not path.exists():
        return lp, has, reliable

    try:
        z = np.load(str(path))
        lp0 = z["log_probs_teacher_full"].astype(np.float32)
        has0 = z.get("teacher_has_full", None)
        rel0 = z.get("teacher_reliable_full", None)
    except Exception as e:
        print(f"[warn] failed to load teacher cache for {protein}: {path} ({e})")
        return lp, has, reliable

    L = int(expected_len)
    if lp0.shape != (L, 20):
        out = np.full((L, 20), log_uniform, dtype=np.float32)
        if lp0.ndim == 2 and int(lp0.shape[1]) == 20:
            m = min(L, int(lp0.shape[0]))
            if m > 0:
                out[:m] = lp0[:m]
        lp0 = out
    lp = lp0

    def _fix_1d(x: Optional[np.ndarray], *, dtype: np.dtype) -> np.ndarray:
        if x is None:
            return np.zeros((L,), dtype=dtype)
        x = np.asarray(x)
        if x.shape == (L,):
            return x.astype(dtype, copy=False)
        out = np.zeros((L,), dtype=dtype)
        m = min(L, int(x.shape[0])) if x.ndim >= 1 else 0
        if m > 0:
            out[:m] = x[:m].astype(dtype, copy=False)
        return out

    has = _fix_1d(has0, dtype=np.uint8)
    reliable = _fix_1d(rel0, dtype=np.float32)
    return lp, has, reliable


def _load_teacher_cache_any(
    teacher_cache_dir: str,
    protein: str,
    *,
    expected_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Try both the raw dataset id and its canonicalized DMS_id.
    This is needed because some local datasets use ProteinGym typos in folder names.
    """
    cand = [str(protein), str(canonicalize_dms_id(protein))]
    seen = set()
    for name in cand:
        if name in seen:
            continue
        seen.add(name)
        path = Path(teacher_cache_dir) / f"{name}.npz"
        if path.exists():
            return _load_teacher_cache(teacher_cache_dir, name, expected_len=expected_len)
    return _load_teacher_cache(teacher_cache_dir, protein, expected_len=expected_len)


def _extract_plddt(struct_path: str, chain_id: str) -> np.ndarray:
    # pLDDT is stored in the B-factor column for AF2 / ESMFold PDBs.
    from Bio.PDB import MMCIFParser, PDBParser  # noqa: PLC0415

    if struct_path.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    elif struct_path.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError("Invalid structure format (expected .pdb or .cif): " + struct_path)

    structure = parser.get_structure("protein", struct_path)
    model = structure[0]
    chain = model[chain_id]

    plddts = []
    for residue in chain:
        residue_plddts = [float(atom.get_bfactor()) for atom in residue.get_atoms()]
        plddts.append(float(np.mean(residue_plddts)) if residue_plddts else 0.0)
    return np.asarray(plddts, dtype=np.float32)


def _load_plddt_item(dataset_root: str, protein: str, *, chain_id: str, expected_len: int) -> np.ndarray:
    pdb_path = os.path.join(dataset_root, "DATASET", protein, f"{protein}.pdb")
    cif_path = os.path.join(dataset_root, "DATASET", protein, f"{protein}.cif")
    struct_path = pdb_path if os.path.exists(pdb_path) else cif_path
    if not os.path.exists(struct_path):
        return np.zeros((expected_len,), dtype=np.float32)
    try:
        plddt = _extract_plddt(struct_path, chain_id)
        if int(len(plddt)) != int(expected_len):
            return np.zeros((expected_len,), dtype=np.float32)
        return plddt.astype(np.float32, copy=False)
    except Exception:
        return np.zeros((expected_len,), dtype=np.float32)


def _load_msa_cache_item(
    msa_cache_dir: str,
    protein: str,
    *,
    expected_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns:
      log_probs[L,20], cov[L], entropy[L], has[L], reliable[L], log_neff (scalar)
    Missing cache -> uniform log-probs and zeros.
    """
    log_uniform = float(-math.log(20.0))
    lp = np.full((expected_len, 20), log_uniform, dtype=np.float32)
    cov = np.zeros((expected_len,), dtype=np.float32)
    ent = np.zeros((expected_len,), dtype=np.float32)
    has = np.zeros((expected_len,), dtype=np.uint8)
    rel = np.zeros((expected_len,), dtype=np.float32)
    log_neff = 0.0

    if not msa_cache_dir:
        return lp, cov, ent, has, rel, log_neff
    path = Path(msa_cache_dir) / f"{protein}.npz"
    if not path.exists():
        return lp, cov, ent, has, rel, log_neff

    try:
        z = np.load(str(path))
        lp0 = z["log_probs_msa_full"].astype(np.float32)
        cv0 = z["msa_cov_full"].astype(np.float32)
        en0 = z["msa_entropy_full"].astype(np.float32)
        hs0 = z["msa_has_full"].astype(np.uint8)
        rl0 = z["msa_reliable_full"].astype(np.float32)
        log_neff = float(z.get("log_neff_global", 0.0))
    except Exception:
        return lp, cov, ent, has, rel, log_neff

    L = int(expected_len)
    if lp0.shape != (L, 20):
        out = np.full((L, 20), log_uniform, dtype=np.float32)
        if lp0.ndim == 2 and int(lp0.shape[1]) == 20:
            m = min(L, int(lp0.shape[0]))
            if m > 0:
                out[:m] = lp0[:m]
        lp0 = out
    lp = lp0

    def _fix_1d(x: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
        if x.shape == (L,):
            return x.astype(dtype, copy=False)
        out = np.zeros((L,), dtype=dtype)
        m = min(L, int(x.shape[0])) if x.ndim >= 1 else 0
        if m > 0:
            out[:m] = x[:m].astype(dtype, copy=False)
        return out

    cov = _fix_1d(cv0, dtype=np.float32)
    ent = _fix_1d(en0, dtype=np.float32)
    has = _fix_1d(hs0, dtype=np.uint8)
    rel = _fix_1d(rl0, dtype=np.float32)
    return lp, cov, ent, has, rel, log_neff


def _msa_extra(
    *,
    msa_gate: Optional[MsaReliabilityGate],
    msa_feat_mode: str,
    msa_no_gate: bool,
    msa_lp: torch.Tensor,  # [L,20]
    msa_cov: torch.Tensor,  # [L]
    msa_ent: torch.Tensor,  # [L]
    msa_has: torch.Tensor,  # [L]
    log_neff: float,
    plddt: torch.Tensor,  # [L]
    pos: torch.Tensor,  # [N]
    wt_idx: torch.Tensor,  # [N]
    mt_idx: torch.Tensor,  # [N]
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    dlogp_msa = (msa_lp[pos, mt_idx] - msa_lp[pos, wt_idx]).detach()  # [N]
    cov = msa_cov[pos].to(device=device, dtype=dtype)
    ent = msa_ent[pos].to(device=device, dtype=dtype)
    has = msa_has[pos].to(device=device, dtype=dtype)
    plddt_pos = plddt[pos].to(device=device, dtype=dtype)
    log_neff_t = torch.full_like(cov, float(log_neff))

    if msa_no_gate or msa_gate is None:
        g = torch.ones_like(cov)
    else:
        g = msa_gate(log_neff=log_neff_t, cov=cov, entropy=ent, plddt=plddt_pos)
    g = (g * has).clamp(0.0, 1.0)

    mode = str(msa_feat_mode)
    if mode == "full":
        raw = torch.stack([dlogp_msa.to(dtype=dtype), cov, ent, log_neff_t, g], dim=-1)  # [N,5]
        # Match training: gated first 4, plus g as last dim.
        gated = raw[:, :4] * g.view(-1, 1)
        return torch.cat([gated, g.view(-1, 1)], dim=-1)
    if mode == "dlogp_only":
        return torch.stack([g * dlogp_msa.to(dtype=dtype), g], dim=-1)
    raise ValueError(f"Invalid msa_feat_mode: {msa_feat_mode}")


def _build_teacher_tables(
    *,
    teacher_lp: np.ndarray,  # [L,20]
    teacher_has: np.ndarray,  # [L]
    teacher_reliable: np.ndarray,  # [L]
    seq: str,
    use_reliable_only: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (teacher_delta_table[L,20], teacher_delta_rankz[L,20]).
    teacher_delta_table[pos, aa] = logP(aa|WT_pos) - logP(WT|WT_pos).
    teacher_delta_rankz is a per-protein rank-zscore over all (pos,aa!=WT) deltas with teacher available.
    """
    L = int(len(seq))
    delta = np.zeros((L, 20), dtype=np.float32)
    rankz = np.zeros((L, 20), dtype=np.float32)

    if teacher_lp.shape != (L, 20):
        return delta, rankz

    mask_pos = teacher_has.astype(bool)
    if use_reliable_only:
        mask_pos = mask_pos & (teacher_reliable > 0)

    items: List[float] = []
    coords: List[Tuple[int, int]] = []
    for pos in range(L):
        if not mask_pos[pos]:
            continue
        wt = seq[pos]
        if wt not in AA_TO_IDX:
            continue
        wt_idx = int(AA_TO_IDX[wt])
        row = teacher_lp[pos]
        d = row - float(row[wt_idx])
        delta[pos] = d
        for aa_idx in range(20):
            if aa_idx == wt_idx:
                continue
            items.append(float(d[aa_idx]))
            coords.append((pos, aa_idx))

    if len(items) >= 2:
        z = _rank_zscore(np.asarray(items, dtype=np.float32))
        for (pos, aa_idx), zi in zip(coords, z):
            rankz[pos, aa_idx] = float(zi)
    return delta, rankz


@torch.no_grad()
def _build_student_mu_table(
    *,
    transition: ActionStateTransition,
    reward: RewardUncertaintyHead,
    embeds: torch.Tensor,  # [L,H]
    log_probs: torch.Tensor,  # [L,20]
    seq: str,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    L = int(len(seq))
    mu_table = np.zeros((L, 20), dtype=np.float32)

    pos_all: List[int] = []
    wt_all: List[int] = []
    mt_all: List[int] = []
    for pos in range(L):
        wt = seq[pos]
        if wt not in AA_TO_IDX:
            continue
        wt_idx = int(AA_TO_IDX[wt])
        for mt_idx in range(20):
            if mt_idx == wt_idx:
                continue
            pos_all.append(pos)
            wt_all.append(wt_idx)
            mt_all.append(mt_idx)

    if not pos_all:
        return mu_table

    embeds = embeds.to(device=device, dtype=torch.float32)
    log_probs = log_probs.to(device=device, dtype=torch.float32)

    for s in range(0, len(pos_all), int(chunk_size)):
        e = min(len(pos_all), s + int(chunk_size))
        pos_t = torch.tensor(pos_all[s:e], device=device, dtype=torch.long)
        wt_t = torch.tensor(wt_all[s:e], device=device, dtype=torch.long)
        mt_t = torch.tensor(mt_all[s:e], device=device, dtype=torch.long)
        wt_embed = embeds[pos_t]
        dlogp = (log_probs[pos_t, mt_t] - log_probs[pos_t, wt_t]).detach()

        delta_embed = transition(
            wt_embed=wt_embed,
            wt_aa_idx=wt_t,
            mut_aa_idx=mt_t,
            delta_logp=dlogp,
            extra_feat=None,
        )
        mut_embed = wt_embed + delta_embed
        mu, _ = reward(
            wt_embed=wt_embed,
            mut_embed_pred=mut_embed,
            wt_aa_idx=wt_t,
            mut_aa_idx=mt_t,
            delta_logp=dlogp,
            extra_feat=None,
        )
        mu_np = mu.detach().cpu().numpy().astype(np.float32)
        for i, (pos, mt_idx) in enumerate(zip(pos_all[s:e], mt_all[s:e])):
            mu_table[int(pos), int(mt_idx)] = float(mu_np[i])

    return mu_table


@torch.no_grad()
def _score_student_rows(
    *,
    transition: ActionStateTransition,
    reward: RewardUncertaintyHead,
    embeds: torch.Tensor,  # [L,H]
    log_probs: torch.Tensor,  # [L,20]
    seq: str,
    mutants: Sequence[str],
    device: torch.device,
    chunk_size: int,
    use_delta_logp: bool,
    msa_gate: Optional[MsaReliabilityGate],
    msa_feat_mode: str,
    msa_no_gate: bool,
    msa_cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]],
) -> List[float]:
    """Compute per-row student scores without building an Lx20 lookup table."""
    L = int(len(seq))
    embeds = embeds.to(device=device, dtype=torch.float32)
    log_probs = log_probs.to(device=device, dtype=torch.float32)

    out: List[float] = []
    offset = 1
    for m in mutants:
        try:
            muts = parse_mutant(str(m), offset_idx=offset)
        except Exception:
            muts = []

        pos_all: List[int] = []
        wt_all: List[int] = []
        mt_all: List[int] = []
        for sm in muts:
            if sm.pos < 0 or sm.pos >= L:
                continue
            if seq[sm.pos] != sm.wt:
                continue
            wt_idx = int(AA_TO_IDX[sm.wt])
            mt_idx = int(AA_TO_IDX[sm.mt])
            if mt_idx == wt_idx:
                continue
            pos_all.append(int(sm.pos))
            wt_all.append(wt_idx)
            mt_all.append(mt_idx)

        if not pos_all:
            out.append(0.0)
            continue

        total = 0.0
        for s in range(0, len(pos_all), int(chunk_size)):
            e = min(len(pos_all), s + int(chunk_size))
            pos_t = torch.tensor(pos_all[s:e], device=device, dtype=torch.long)
            wt_t = torch.tensor(wt_all[s:e], device=device, dtype=torch.long)
            mt_t = torch.tensor(mt_all[s:e], device=device, dtype=torch.long)
            wt_embed = embeds[pos_t]
            dlogp = (log_probs[pos_t, mt_t] - log_probs[pos_t, wt_t]).detach()

            extra = None
            if msa_cache is not None:
                msa_lp, msa_cov, msa_ent, msa_has, log_neff, plddt = msa_cache
                extra = _msa_extra(
                    msa_gate=msa_gate,
                    msa_feat_mode=msa_feat_mode,
                    msa_no_gate=msa_no_gate,
                    msa_lp=msa_lp,
                    msa_cov=msa_cov,
                    msa_ent=msa_ent,
                    msa_has=msa_has,
                    log_neff=float(log_neff),
                    plddt=plddt,
                    pos=pos_t,
                    wt_idx=wt_t,
                    mt_idx=mt_t,
                    device=device,
                    dtype=wt_embed.dtype,
                )

            delta_embed = transition(
                wt_embed=wt_embed,
                wt_aa_idx=wt_t,
                mut_aa_idx=mt_t,
                delta_logp=dlogp if use_delta_logp else None,
                extra_feat=extra,
            )
            mut_embed = wt_embed + delta_embed
            mu, _ = reward(
                wt_embed=wt_embed,
                mut_embed_pred=mut_embed,
                wt_aa_idx=wt_t,
                mut_aa_idx=mt_t,
                delta_logp=dlogp if use_delta_logp else None,
                extra_feat=extra,
            )
            total += float(mu.detach().sum().cpu().item())

        out.append(float(total))

    return out


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Write ProteinGym score CSVs for WMZ + SiteRM fusion (by-protein CV5).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--struct_root", type=str, default="", help="Optional: separate root to locate structures for 3Di (ProteinGym layout preferred).")
    p.add_argument("--folds_dir", type=str, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing wmz_fold{fold}.pt")

    p.add_argument("--mutant_col", type=str, default="mutant")
    p.add_argument("--label_col", type=str, default="DMS_score")

    p.add_argument("--plm", type=str, required=True)
    p.add_argument("--backbone", choices=["saprot"], default="saprot",
                   help="This release contains only the SaProt backbone.")
    p.add_argument("--saprot_scoring", choices=["fast"], default="fast")
    p.add_argument("--foldseek_bin", type=str, default="foldseek")
    p.add_argument("--struct_cache_dir", type=str, default="", help="Cache dir for 3Di tokens (default: dataset_root/CACHE/3di).")
    p.add_argument("--struct_chain_id", type=str, default="A")
    p.add_argument("--plddt_mask", type=str, default="auto", choices=["auto", "on", "off"])
    p.add_argument("--plddt_threshold", type=float, default=70.0)

    p.add_argument("--teacher_cache_dir", type=str, required=True)
    p.add_argument(
        "--alphas",
        type=str,
        default="0.05:0.1:0.2",
        help="Fusion weights (colon/comma/space separated): fused = student + alpha * teacher_rankz_sum.",
    )
    p.add_argument("--teacher_use_reliable_only", action="store_true")

    p.add_argument("--chunk_size", type=int, default=32768, help="Chunk size for scoring all single substitutions.")
    p.add_argument("--no_delta_logp_feature", action="store_true")

    # Optional MSA-lite features (must match WMZ training).
    p.add_argument("--msa_cache_dir", type=str, default="", help="Directory containing <protein>.npz built by script/msa/build_msa_cache.py")
    p.add_argument("--no_msa_features", action="store_true", help="Disable MSA-lite even if --msa_cache_dir is set.")
    p.add_argument("--msa_no_gate", action="store_true", help="Ablation: use raw MSA features (no learnable gate).")
    p.add_argument("--msa_gate_hidden", type=int, default=64)
    p.add_argument("--msa_gate_init_bias", type=float, default=-10.0)
    p.add_argument("--msa_gate_dropout", type=float, default=0.0)
    p.add_argument("--msa_feat_mode", type=str, default="full", choices=["full", "dlogp_only"])
    p.add_argument("--student_mode", choices=["rows"], default="rows")

    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory (one <DMS_id>.csv per protein).",
    )
    p.add_argument(
        "--location_name",
        type=str,
        default="ProteinDreamer_WMZ_SiteRM_bundle",
        help="Folder name under zero_shot_substitutions_scores (for logging only).",
    )
    return p


def main() -> None:
    args = create_parser().parse_args()
    _set_seed(int(args.seed))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = str(device)

    folds_dir = Path(args.folds_dir)
    valid_proteins = _read_list(str(folds_dir / f"fold{int(args.fold)}_valid.txt"))
    if not valid_proteins:
        raise SystemExit("Empty fold list; check --folds_dir and --fold.")

    plm_model = SaProtBackbone(args.plm, device=args.device).to(device).eval()
    args.plm_hidden_size = plm_model.model.config.hidden_size
    for p in plm_model.parameters():
        p.requires_grad = False

    use_delta_logp = not bool(args.no_delta_logp_feature)
    use_msa = bool(str(args.msa_cache_dir).strip()) and (not bool(args.no_msa_features))
    msa_extra_dim = 0
    if use_msa:
        msa_extra_dim = 5 if str(args.msa_feat_mode) == "full" else 2

    msa_gate: Optional[MsaReliabilityGate] = None
    if use_msa and (not bool(args.msa_no_gate)):
        msa_gate = MsaReliabilityGate(
            hidden_dim=int(args.msa_gate_hidden),
            init_bias=float(args.msa_gate_init_bias),
            dropout=float(args.msa_gate_dropout),
        ).to(device)

    transition = ActionStateTransition(
        hidden_dim=int(args.plm_hidden_size),
        aa_emb_dim=32,
        mlp_hidden_dim=1024,
        dropout=0.0,
        use_delta_logp=use_delta_logp,
        extra_feat_dim=int(msa_extra_dim),
    ).to(device)
    reward = RewardUncertaintyHead(
        hidden_dim=int(args.plm_hidden_size),
        aa_emb_dim=32,
        mlp_hidden_dim=1024,
        dropout=0.0,
        use_delta_logp=use_delta_logp,
        extra_feat_dim=int(msa_extra_dim),
    ).to(device)

    ckpt_file = Path(args.ckpt_dir) / f"wmz_fold{int(args.fold)}.pt"
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_file}")
    ckpt = torch.load(str(ckpt_file), map_location="cpu")
    transition.load_state_dict(ckpt["transition_state_dict"], strict=True)
    reward.load_state_dict(ckpt["reward_state_dict"], strict=True)
    if msa_gate is not None and ckpt.get("msa_gate_state_dict") is not None:
        msa_gate.load_state_dict(ckpt["msa_gate_state_dict"], strict=True)
    transition.eval()
    reward.eval()
    if msa_gate is not None:
        msa_gate.eval()

    mm = ProteinGymSaProtDataset(args.dataset_root, valid_proteins)
    loader = DataLoader(mm, batch_size=1, shuffle=False)
    wanted = set(valid_proteins)
    graphs: Dict[str, Data] = {}
    for data in loader:
        protein = data.protein_name[0]
        if protein in wanted:
            graphs[protein] = data.cpu()

    if not str(args.foldseek_bin).strip():
        raise ValueError("--foldseek_bin is required for SaProt")
    struct_root = str(args.struct_root).strip() or str(args.dataset_root)
    cache_dir = str(args.struct_cache_dir).strip() or os.path.join(str(args.dataset_root), "CACHE", "3di")
    for protein, g in list(graphs.items()):
        length = int(g.y.shape[0])
        struct_path = _resolve_struct_path(struct_root, str(protein))
        if not struct_path:
            raise FileNotFoundError(f"{protein}: no structure under {struct_root}")
        try:
            tokens = get_3di_tokens_cached(
                protein_name=str(protein), struct_path=str(struct_path),
                foldseek_bin=str(args.foldseek_bin), cache_dir=str(cache_dir),
                chain_id=str(args.struct_chain_id), plddt_mask=str(args.plddt_mask),
                plddt_threshold=float(args.plddt_threshold),
            )
        except Exception as exc:
            raise RuntimeError(f"{protein}: Foldseek 3Di extraction failed") from exc
        if int(tokens.numel()) != length:
            raise ValueError(f"{protein}: 3Di length {tokens.numel()} != sequence length {length}")
        g.struct_tokens = tokens

    alphas = _parse_alphas(args.alphas)
    alpha_cols = {a: f"ProteinDreamer_WMZ_SiteRM_fusion_a{int(round(a * 1000)):03d}" for a in alphas}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[score] fold={int(args.fold)} proteins={len(valid_proteins)} location={args.location_name}")
    print(f"[score] ckpt={ckpt_file}")
    print(f"[score] teacher_cache_dir={args.teacher_cache_dir} reliable_only={bool(args.teacher_use_reliable_only)}")
    print(f"[score] alphas={alphas} cols={list(alpha_cols.values())}")

    for protein in valid_proteins:
        if protein not in graphs:
            print(f"[warn] missing graph for {protein} (skipped)")
            continue

        out_id = canonicalize_dms_id(protein)

        df = _load_mutant_df(args.dataset_root, protein)
        if args.mutant_col not in df.columns:
            print(f"[warn] {protein}: missing mutant column {args.mutant_col} (skipped)")
            continue

        # WT forward once (embeddings). For SaProt PLL mode we overwrite log_probs at mutation sites.
        embeds, log_probs, seq = _forward_backbone(
            plm_model=plm_model, graph=graphs[protein], device=device)
        L = int(len(seq))

        mutants = df[args.mutant_col].astype(str).tolist()

        # Student score: table mode (fast) or rows mode (mutation-only).
        student_mode = str(args.student_mode)
        if student_mode == "table":
            mu_table = _build_student_mu_table(
                transition=transition,
                reward=reward,
                embeds=embeds,
                log_probs=log_probs,
                seq=seq,
                device=device,
                chunk_size=int(args.chunk_size),
            )
            # Per-row additive scores (multi mutants supported).
            student_scores: List[float] = []
            offset = 1
            for m in mutants:
                try:
                    muts = parse_mutant(str(m), offset_idx=offset)
                except Exception:
                    muts = []
                s = 0.0
                for sm in muts:
                    if sm.pos < 0 or sm.pos >= L:
                        continue
                    if seq[sm.pos] != sm.wt:
                        continue
                    wt_idx = int(AA_TO_IDX[sm.wt])
                    mt_idx = int(AA_TO_IDX[sm.mt])
                    if mt_idx == wt_idx:
                        continue
                    s += float(mu_table[int(sm.pos), mt_idx])
                student_scores.append(float(s))
        elif student_mode == "rows":
            msa_cache = None
            if use_msa:
                msa_lp_np, msa_cov_np, msa_ent_np, msa_has_np, msa_rel_np, log_neff = _load_msa_cache_item(
                    str(args.msa_cache_dir), protein, expected_len=L
                )
                plddt_np = _load_plddt_item(str(args.dataset_root), protein, chain_id=str(args.struct_chain_id), expected_len=L)
                msa_cache = (
                    torch.tensor(msa_lp_np, dtype=torch.float32, device=device),
                    torch.tensor(msa_cov_np, dtype=torch.float32, device=device),
                    torch.tensor(msa_ent_np, dtype=torch.float32, device=device),
                    torch.tensor(msa_has_np.astype(np.float32), dtype=torch.float32, device=device),
                    float(log_neff),
                    torch.tensor(plddt_np, dtype=torch.float32, device=device),
                )
            student_scores = _score_student_rows(
                transition=transition,
                reward=reward,
                embeds=embeds,
                log_probs=log_probs,
                seq=seq,
                mutants=mutants,
                device=device,
                chunk_size=int(args.chunk_size),
                use_delta_logp=use_delta_logp,
                msa_gate=msa_gate,
                msa_feat_mode=str(args.msa_feat_mode),
                msa_no_gate=bool(args.msa_no_gate),
                msa_cache=msa_cache,
            )
        else:
            raise ValueError(f"Invalid --student_mode: {args.student_mode}")

        # Teacher lookup tables: raw ΔlogP and rank-zscore(ΔlogP) for all single substitutions.
        teacher_lp, teacher_has, teacher_rel = _load_teacher_cache_any(str(args.teacher_cache_dir), protein, expected_len=L)
        teacher_delta_table, teacher_rankz_table = _build_teacher_tables(
            teacher_lp=teacher_lp,
            teacher_has=teacher_has,
            teacher_reliable=teacher_rel,
            seq=seq,
            use_reliable_only=bool(args.teacher_use_reliable_only),
        )

        teacher_scores: List[float] = []
        teacher_rankz_scores: List[float] = []
        fused_scores: Dict[float, List[float]] = {a: [] for a in alphas}

        offset = 1
        for i, m in enumerate(mutants):
            muts = []
            try:
                muts = parse_mutant(m, offset_idx=offset)
            except Exception:
                muts = []

            s = float(student_scores[i]) if i < len(student_scores) else 0.0
            t = 0.0
            tz = 0.0
            for sm in muts:
                if sm.pos < 0 or sm.pos >= L:
                    continue
                if seq[sm.pos] != sm.wt:
                    continue
                wt_idx = int(AA_TO_IDX[sm.wt])
                mt_idx = int(AA_TO_IDX[sm.mt])
                if mt_idx == wt_idx:
                    continue
                t += float(teacher_delta_table[int(sm.pos), mt_idx])
                tz += float(teacher_rankz_table[int(sm.pos), mt_idx])

            teacher_scores.append(float(t))
            teacher_rankz_scores.append(float(tz))
            for a in alphas:
                fused_scores[a].append(float(s + float(a) * tz))

        df["ProteinDreamer_WMZ_student"] = student_scores
        df["ProteinDreamer_SiteRM_teacher"] = teacher_scores
        df["ProteinDreamer_SiteRM_teacher_rankzsum"] = teacher_rankz_scores
        for a in alphas:
            df[alpha_cols[a]] = fused_scores[a]

        out_path = out_dir / f"{out_id}.csv"
        df.to_csv(out_path, index=False)
        if _stable_int(protein) % 17 == 0:
            print(f"[score] wrote {out_path}")

    print(f"[score] done fold={int(args.fold)} out_dir={out_dir}")


if __name__ == "__main__":
    main()
