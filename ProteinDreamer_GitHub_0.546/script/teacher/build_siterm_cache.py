from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# CHERRYML_ROOT overrides the local external/CherryML checkout.
_CHERRYML_ROOT = Path(os.environ.get("CHERRYML_ROOT", str(_REPO_ROOT / "external" / "CherryML")))
if (_CHERRYML_ROOT / "cherryml").exists() and str(_CHERRYML_ROOT) not in sys.path:
    sys.path.append(str(_CHERRYML_ROOT))

from src.utils.msa_cache import (  # noqa: E402
    choose_msa_file,
    choose_weights_file,
    compute_weighted_profile,
    protein_to_base_id,
    read_full_sequence_length_from_csv,
    scan_msa,
)


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build phylogenetic teacher cache (.npz) via CherryML SiteRM/FastCherries "
            "(per-site stationary or transition log-probs, mapped to full-length)."
        )
    )

    p.add_argument("--dataset_root", type=str, required=True, help="ProteinGym-style dataset root with DATASET/<protein>/...")
    p.add_argument("--protein", type=str, default="", help="Single protein id to process (folder name under DATASET).")
    p.add_argument("--protein_list", type=str, default="", help="Optional txt list of protein ids (one per line).")

    p.add_argument(
        "--msa_dir",
        type=str,
        default="data/proteingym_data/DMS_msa_files/DMS_msa_files",
        help="Directory containing *.a2m files (DMS).",
    )
    p.add_argument(
        "--weights_dir",
        type=str,
        default="data/proteingym_data/DMS_msa_weights/DMS_msa_weights",
        help="Directory containing *.npy weights aligned to the MSA sequences.",
    )
    p.add_argument(
        "--msa_cache_dir",
        type=str,
        default="data/proteingym_local/msa_cache_dms_v1",
        help="Optional: reuse per-protein MSA-lite cache for cov/entropy/has/logNeff (falls back to recompute if missing).",
    )

    p.add_argument("--out_dir", type=str, required=True, help="Output directory for per-protein .npz cache files.")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument(
        "--max_sequences",
        type=int,
        default=4096,
        help="Maximum number of MSA sequences fed to SiteRM (0 disables; not recommended).",
    )
    p.add_argument(
        "--sampling",
        type=str,
        default="weighted_reservoir",
        choices=["weighted_reservoir", "prefix"],
        help="Sequence sampling strategy when max_sequences < available sequences.",
    )
    p.add_argument("--seed", type=int, default=0)

    # SiteRM parameters.
    p.add_argument("--regularization_strength", type=float, default=0.5)
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--quantization_grid_num_steps", type=int, default=16)
    p.add_argument("--num_rate_categories", type=int, default=20)
    p.add_argument("--use_vectorized_implementation", action="store_true", help="Use CherryML vectorized implementation (may use more RAM).")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    p.add_argument(
        "--teacher_mode",
        type=str,
        default="stationary",
        choices=["stationary", "transition"],
        help="What to store as log_probs_teacher_full. stationary=log pi_i (v1, weak). transition=log P(.|WT_i, t) via exp(Q_i * t) (recommended).",
    )
    p.add_argument(
        "--evolutionary_time",
        type=float,
        default=1.0,
        help="Evolutionary time t used for transition teacher mode: log P(.|WT_i, t). Only used when --teacher_mode=transition.",
    )

    # Reliability mask (used to gate teacher loss/fusion; conservative defaults).
    p.add_argument("--reliable_cov_thresh", type=float, default=0.5)
    p.add_argument("--reliable_neff_thresh", type=float, default=50.0)

    p.add_argument("--dry_run", action="store_true")
    return p


def _read_list(path: str) -> list[str]:
    items: list[str] = []
    with open(path, "r", encoding="utf-8") as r:
        for line in r:
            s = line.strip()
            if s:
                items.append(s)
    return items


def _stable_int(text: str) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _resolve_csv(dataset_root: str, protein: str) -> str:
    base = os.path.join(dataset_root, "DATASET", protein, protein)
    csv_path = base + ".csv"
    tsv_path = base + ".tsv"
    if os.path.exists(csv_path):
        return csv_path
    if os.path.exists(tsv_path):
        return tsv_path
    raise FileNotFoundError(f"Missing mutant file for {protein}: {csv_path} or {tsv_path}")


def _read_full_len_from_dataset(dataset_root: str, protein: str) -> int:
    path = _resolve_csv(dataset_root, protein)
    if path.endswith(".csv"):
        return read_full_sequence_length_from_csv(path)
    import pandas as pd  # noqa: PLC0415

    df = pd.read_table(path, nrows=1)
    if "mutated_sequence" not in df.columns:
        raise ValueError(f"TSV missing mutated_sequence: {path}")
    seq = str(df["mutated_sequence"].iloc[0])
    return int(len(seq.strip()))


def _iter_fasta_records(path: str) -> Iterator[Tuple[str, str]]:
    header: Optional[str] = None
    seq_parts: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as r:
        for raw in r:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:].strip()
                seq_parts = []
                continue
            seq_parts.append(line)
    if header is not None:
        yield header, "".join(seq_parts)


def _read_query_sequence(msa_path: str) -> str:
    for _, seq in _iter_fasta_records(msa_path):
        return seq
    raise ValueError(f"No FASTA records found: {msa_path}")


def _load_lg_matrix() -> pd.DataFrame:
    """
    Load LG matrix from the local CherryML example checkout if present; fall back to WAG from CherryML package.
    """
    env = os.environ.get("CHERRYML_LG_PATH", "").strip()
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))

    candidates.append(_CHERRYML_ROOT / "data/rate_matrices/lg_with_gaps.txt")
    candidates.append(_CHERRYML_ROOT / "data/rate_matrices/lg.txt")

    for p in candidates:
        if p.exists():
            df = pd.read_csv(str(p), sep=r"\s+", index_col=0)
            return df

    # Fallback: WAG shipped inside CherryML.
    from cherryml.markov_chain import wag_matrix  # noqa: PLC0415

    print("[warn] LG matrix not found; falling back to CherryML WAG matrix. Set CHERRYML_LG_PATH to use LG.")
    return wag_matrix()


def _load_reliability_from_msa_cache(
    msa_cache_dir: str, protein: str, *, expected_len: int, cov_thresh: float, neff_thresh: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns (cov_full, ent_full, has_full, reliable_full, log_neff).
    Falls back to zeros if cache missing or invalid.
    """
    cov = np.zeros((expected_len,), dtype=np.float32)
    ent = np.zeros((expected_len,), dtype=np.float32)
    has = np.zeros((expected_len,), dtype=np.uint8)
    reliable = np.zeros((expected_len,), dtype=np.float32)
    log_neff = 0.0

    if not msa_cache_dir:
        return cov, ent, has, reliable, log_neff

    path = Path(msa_cache_dir) / f"{protein}.npz"
    if not path.exists():
        return cov, ent, has, reliable, log_neff

    try:
        z = np.load(str(path))
        cov0 = z["msa_cov_full"].astype(np.float32)
        ent0 = z["msa_entropy_full"].astype(np.float32)
        has0 = z["msa_has_full"].astype(np.uint8)
        log_neff = float(z.get("log_neff_global", 0.0))
    except Exception as e:
        print(f"[warn] failed to load msa_cache for {protein}: {path} ({e})")
        return cov, ent, has, reliable, log_neff

    L = int(expected_len)

    def _fix_1d(x: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
        if x.shape == (L,):
            return x.astype(dtype, copy=False)
        out = np.zeros((L,), dtype=dtype)
        m = min(L, int(x.shape[0])) if x.ndim >= 1 else 0
        if m > 0:
            out[:m] = x[:m].astype(dtype, copy=False)
        return out

    cov = _fix_1d(cov0, dtype=np.float32)
    ent = _fix_1d(ent0, dtype=np.float32)
    has = _fix_1d(has0, dtype=np.uint8)
    reliable = ((cov >= float(cov_thresh)) & (float(math.exp(log_neff)) >= float(neff_thresh))).astype(np.float32)
    reliable = reliable * has.astype(np.float32)
    return cov, ent, has, reliable, float(log_neff)


def _compute_profile_reliability(
    msa_path: str, weights_path: str, *, full_len: int, cov_thresh: float, neff_thresh: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    prof = compute_weighted_profile(msa_path, weights_path, pseudocount=1e-3, max_sequences=0)
    start0 = int(prof.start1) - 1
    end0 = int(prof.end1) - 1

    cov_full = np.zeros((full_len,), dtype=np.float32)
    ent_full = np.zeros((full_len,), dtype=np.float32)
    has_full = np.zeros((full_len,), dtype=np.uint8)

    s = max(0, start0)
    e = min(full_len - 1, end0)
    if e >= s:
        src_s = s - start0
        src_e = src_s + (e - s + 1)
        cov_full[s : e + 1] = prof.cov[src_s:src_e]
        ent_full[s : e + 1] = prof.entropy[src_s:src_e]
        has_full[s : e + 1] = 1

    reliable_full = ((cov_full >= float(cov_thresh)) & (float(prof.neff) >= float(neff_thresh))).astype(np.float32)
    reliable_full = reliable_full * has_full.astype(np.float32)
    return cov_full, ent_full, has_full, reliable_full, float(prof.log_neff)


def _sample_msa_sequences(
    msa_path: str,
    *,
    weights: np.ndarray,
    aln_len: int,
    max_sequences: int,
    sampling: str,
    seed: int,
) -> Dict[str, str]:
    weights_len = int(weights.shape[0])
    if weights_len <= 0:
        raise ValueError("Empty weights.")
    if max_sequences <= 0:
        raise ValueError("--max_sequences must be >0 for SiteRM cache build.")

    k = int(min(max_sequences, weights_len))
    rng = random.Random(int(seed))

    if sampling == "prefix":
        msa: Dict[str, str] = {}
        for i, (_, seq) in enumerate(_iter_fasta_records(msa_path)):
            if i >= weights_len:
                break
            seq_u = seq.upper().replace(".", "-")
            if len(seq_u) != int(aln_len):
                continue
            msa[f"leaf_{i}"] = seq_u
            if len(msa) >= k:
                break
        if not msa:
            raise ValueError("No usable sequences from prefix sampling.")
        return msa

    # weighted reservoir sampling (Efraimidis-Spirakis, without replacement)
    heap: list[tuple[float, int, str]] = []
    for i, (_, seq) in enumerate(_iter_fasta_records(msa_path)):
        if i >= weights_len:
            break
        seq_u = seq.upper().replace(".", "-")
        if len(seq_u) != int(aln_len):
            continue
        w = float(weights[i])
        if i == 0:
            key = float("inf")  # force include query
        else:
            if w <= 0:
                continue
            u = rng.random()
            u = max(u, 1e-12)
            key = math.log(u) / w  # negative; larger is better
        if len(heap) < k:
            heapq.heappush(heap, (float(key), int(i), seq_u))
        else:
            if float(key) > heap[0][0]:
                heapq.heapreplace(heap, (float(key), int(i), seq_u))

    if not heap:
        raise ValueError("No usable sequences from weighted reservoir sampling.")
    heap.sort(key=lambda t: t[1])
    return {f"leaf_{idx}": seq for _, idx, seq in heap}


def _safe_stationary_log_probs(rate_mats: np.ndarray, *, keep_states: int = 20) -> np.ndarray:
    from cherryml.markov_chain import compute_stationary_distribution  # noqa: PLC0415

    L, s1, s2 = rate_mats.shape
    if s1 != s2 or s1 < int(keep_states):
        raise ValueError(f"Unexpected rate_mats shape={rate_mats.shape} (expected [L,S,S] with S>={keep_states})")
    out = np.zeros((L, int(keep_states)), dtype=np.float32)
    for i in range(L):
        pi = compute_stationary_distribution(rate_mats[i])
        pi = np.asarray(pi, dtype=np.float64)
        if pi.shape != (s1,) or not np.isfinite(pi).all():
            pi = np.full((s1,), 1.0 / float(s1), dtype=np.float64)
        pi = np.maximum(pi, 0.0)
        pi = pi[: int(keep_states)]
        s = float(pi.sum())
        if s <= 0:
            pi = np.full((int(keep_states),), 1.0 / float(keep_states), dtype=np.float64)
        else:
            pi = pi / s
        out[i] = np.log(pi.astype(np.float32) + 1e-12)
    return out


def _safe_transition_log_probs(
    rate_mats: np.ndarray,
    *,
    query_seq_aln: str,
    alphabet: List[str],
    t: float,
    keep_states: int = 20,
) -> np.ndarray:
    """
    Build per-site log-probs for substitution scoring:
      log_probs[i, a] = log P(a | WT_i, t) where P = exp(Q_i * t).
    Returns [L, keep_states] aligned to AA_ORDER (assumed to match CherryML amino acid order).
    """
    from cherryml import markov_chain  # noqa: PLC0415

    L, s1, s2 = rate_mats.shape
    if s1 != s2 or s1 < int(keep_states):
        raise ValueError(f"Unexpected rate_mats shape={rate_mats.shape} (expected [L,S,S] with S>={keep_states})")

    aa_to_state = {aa: i for i, aa in enumerate(alphabet)}
    query_seq = str(query_seq_aln).upper().replace(".", "-")
    if len(query_seq) != int(L):
        # Trust the learnt rate matrices length (alignment length).
        if len(query_seq) < int(L):
            query_seq = query_seq + ("-" * (int(L) - len(query_seq)))
        else:
            query_seq = query_seq[: int(L)]

    out = np.full((L, int(keep_states)), float(-math.log(20.0)), dtype=np.float32)
    for i in range(L):
        wt = query_seq[i]
        if wt not in aa_to_state:
            continue
        wt_idx = int(aa_to_state[wt])
        if wt_idx >= int(keep_states):
            continue
        fact = markov_chain.FactorizedReversibleModel(rate_mats[i])
        P = markov_chain.matrix_exponential_reversible(exponents=[float(t)], fact=fact, device="cpu")[0]
        P = np.asarray(P, dtype=np.float64)
        # Fill only the 20 AA states in-order.
        row = P[wt_idx, : int(keep_states)]
        row = np.clip(row, 1e-12, 1.0)
        out[i] = np.log(row.astype(np.float32))
    return out


def _load_wt_seq_from_structure(dataset_root: str, protein: str) -> Optional[str]:
    try:
        from src.saprot_dataset import load_structure_sequence, resolve_structure_path  # noqa: PLC0415
        return load_structure_sequence(resolve_structure_path(dataset_root, protein))
    except Exception as e:
        print(f"[warn] failed to load structure WT for {protein}: {e}")
        return None


def main() -> None:
    args = create_parser().parse_args()

    proteins: list[str] = []
    if args.protein_list:
        proteins = _read_list(args.protein_list)
    elif args.protein:
        proteins = [args.protein]
    else:
        raise SystemExit("Must set --protein or --protein_list")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lg = _load_lg_matrix()
    from cherryml.utils import get_amino_acids  # noqa: PLC0415

    amino_acids = get_amino_acids()
    alphabet = amino_acids + ["-"]
    if "-" in lg.columns and "-" in lg.index:
        lg = lg.loc[alphabet, alphabet]
    else:
        lg = lg.loc[amino_acids, amino_acids]
        alphabet = amino_acids

    # For ProteinGym, CherryML recommends excluding gaps when estimating site rates.
    alphabet_for_site_rate_estimation = amino_acids
    rate_matrix_for_site_rate_estimation = lg.loc[amino_acids, amino_acids].copy()

    from cherryml import learn_site_specific_rate_matrices  # noqa: PLC0415

    for protein in proteins:
        out_path = out_dir / f"{protein}.npz"
        meta_path = out_dir / f"{protein}.json"
        if out_path.exists() and (not bool(args.overwrite)):
            print(f"[skip] {protein}: exists {out_path}")
            continue

        base_id = protein_to_base_id(protein)
        msa_path = choose_msa_file(args.msa_dir, base_id)
        if msa_path is None:
            print(f"[warn] {protein}: no MSA found for base_id={base_id} in {args.msa_dir}")
            continue

        scan = scan_msa(msa_path)
        w_path = choose_weights_file(args.weights_dir, base_id, msa_n_sequences=scan.n_sequences)
        if w_path is None:
            print(f"[warn] {protein}: no matching weights found for base_id={base_id} n_seq={scan.n_sequences} in {args.weights_dir}")
            continue

        full_len = _read_full_len_from_dataset(args.dataset_root, protein)

        cov_full, ent_full, has_full, reliable_full, log_neff = _load_reliability_from_msa_cache(
            str(args.msa_cache_dir),
            protein,
            expected_len=full_len,
            cov_thresh=float(args.reliable_cov_thresh),
            neff_thresh=float(args.reliable_neff_thresh),
        )
        if not bool(has_full.any()):
            cov_full, ent_full, has_full, reliable_full, log_neff = _compute_profile_reliability(
                msa_path,
                w_path,
                full_len=full_len,
                cov_thresh=float(args.reliable_cov_thresh),
                neff_thresh=float(args.reliable_neff_thresh),
            )

        weights = np.load(w_path)
        if weights.ndim != 1:
            print(f"[warn] {protein}: invalid weights shape={weights.shape} for {w_path}")
            continue

        st = time.time()
        msa = _sample_msa_sequences(
            msa_path,
            weights=weights,
            aln_len=int(scan.aln_len),
            max_sequences=int(args.max_sequences),
            sampling=str(args.sampling),
            seed=int(args.seed) + _stable_int(protein),
        )
        t_sample = time.time() - st

        # Fit SiteRM (tree estimated via FastCherries under the hood).
        st = time.time()
        res = learn_site_specific_rate_matrices(
            tree=None,
            msa=msa,
            alphabet=alphabet,
            regularization_rate_matrix=lg,
            regularization_strength=float(args.regularization_strength),
            device=str(args.device),
            num_rate_categories=int(args.num_rate_categories),
            num_epochs=int(args.num_epochs),
            quantization_grid_num_steps=int(args.quantization_grid_num_steps),
            use_vectorized_implementation=bool(args.use_vectorized_implementation),
            alphabet_for_site_rate_estimation=alphabet_for_site_rate_estimation,
            rate_matrix_for_site_rate_estimation=rate_matrix_for_site_rate_estimation,
        )
        t_fit = time.time() - st

        rate_mats = res.get("learnt_rate_matrices")
        if rate_mats is None:
            print(f"[warn] {protein}: learnt_rate_matrices is None (unexpected).")
            continue

        st = time.time()
        rate_mats_np = np.asarray(rate_mats)
        teacher_mode = str(args.teacher_mode)
        if teacher_mode == "stationary":
            log_probs_aln = _safe_stationary_log_probs(rate_mats_np, keep_states=20)
        elif teacher_mode == "transition":
            query_seq_aln = _read_query_sequence(msa_path)
            log_probs_aln = _safe_transition_log_probs(
                rate_mats_np,
                query_seq_aln=query_seq_aln,
                alphabet=alphabet,
                t=float(args.evolutionary_time),
                keep_states=20,
            )
        else:
            raise ValueError(f"Unknown teacher_mode: {teacher_mode}")
        t_pi = time.time() - st

        # Map to full-length positions.
        log_uniform = float(math.log(1.0 / 20.0))
        log_probs_full = np.full((full_len, 20), log_uniform, dtype=np.float32)
        start0 = int(scan.start1) - 1
        end0 = int(scan.end1) - 1
        s = max(0, start0)
        e = min(full_len - 1, end0)
        if e >= s:
            src_s = s - start0
            src_e = src_s + (e - s + 1)
            log_probs_full[s : e + 1] = log_probs_aln[src_s:src_e]

        wt_seq = _load_wt_seq_from_structure(str(args.dataset_root), protein)
        mismatch_count = 0
        if wt_seq is not None:
            # Exclude query residues that do not match the dataset WT sequence.
            query_seq_aln = _read_query_sequence(msa_path).upper().replace(".", "-")
            # Normalize length to the scan span (we treat it as contiguous).
            span = int(scan.end1 - scan.start1 + 1)
            if len(query_seq_aln) != span:
                if len(query_seq_aln) < span:
                    query_seq_aln = query_seq_aln + ("-" * (span - len(query_seq_aln)))
                else:
                    query_seq_aln = query_seq_aln[:span]
            if e >= s:
                for full_pos in range(s, e + 1):
                    aln_i = int(full_pos - start0)
                    if aln_i < 0 or aln_i >= len(query_seq_aln):
                        continue
                    if full_pos < 0 or full_pos >= len(wt_seq):
                        continue
                    wt_msa = query_seq_aln[aln_i]
                    if wt_msa == "-" or wt_msa == ".":
                        continue
                    if wt_seq[full_pos] != wt_msa:
                        mismatch_count += 1
                        has_full[full_pos] = 0
                        reliable_full[full_pos] = 0.0
                        log_probs_full[full_pos] = log_uniform

        if args.dry_run:
            print(
                f"[dry] {protein} base={base_id} full_len={full_len} aln_len={scan.aln_len} "
                f"n_msa_used={len(msa)} num_epochs={args.num_epochs} t_sample={t_sample:.2f}s t_fit={t_fit:.2f}s t_pi={t_pi:.2f}s"
            )
            continue

        meta = {
            "protein": protein,
            "base_id": base_id,
            "dataset_root": str(args.dataset_root),
            "msa_path": str(msa_path),
            "weights_path": str(w_path),
            "msa_query_header": str(scan.query_header),
            "msa_start1": int(scan.start1),
            "msa_end1": int(scan.end1),
            "msa_aln_len": int(scan.aln_len),
            "msa_n_sequences_scan": int(scan.n_sequences),
            "weights_len": int(weights.shape[0]),
            "max_sequences": int(args.max_sequences),
            "sampling": str(args.sampling),
            "n_msa_used": int(len(msa)),
            "teacher_method": f"siterm_{teacher_mode}_logprobs",
            "teacher_mode": teacher_mode,
            "evolutionary_time": float(args.evolutionary_time),
            "alphabet": alphabet,
            "regularization_strength": float(args.regularization_strength),
            "num_epochs": int(args.num_epochs),
            "quantization_grid_num_steps": int(args.quantization_grid_num_steps),
            "num_rate_categories": int(args.num_rate_categories),
            "use_vectorized_implementation": bool(args.use_vectorized_implementation),
            "device": str(args.device),
            "reliable_cov_thresh": float(args.reliable_cov_thresh),
            "reliable_neff_thresh": float(args.reliable_neff_thresh),
            "wt_sequence_source": "dataset structure",
            "wt_seq_mismatch_count": int(mismatch_count),
            "timing_sec": {"sample_msa": float(t_sample), "fit_siterm": float(t_fit), "teacher_logprobs": float(t_pi)},
        }

        np.savez_compressed(
            out_path,
            log_probs_teacher_full=log_probs_full,
            teacher_has_full=has_full.astype(np.uint8),
            teacher_reliable_full=reliable_full.astype(np.float32),
            teacher_cov_full=cov_full.astype(np.float32),
            teacher_entropy_full=ent_full.astype(np.float32),
            log_neff_global=np.asarray(float(log_neff), dtype=np.float32),
            teacher_start1=np.asarray(int(scan.start1), dtype=np.int32),
            teacher_end1=np.asarray(int(scan.end1), dtype=np.int32),
        )
        with open(meta_path, "w", encoding="utf-8") as w:
            json.dump(meta, w, indent=2, ensure_ascii=False)

        print(
            f"[ok] {protein} base={base_id} full_len={full_len} aln_len={scan.aln_len} "
            f"n_msa_used={len(msa)} log_neff={log_neff:.3f} "
            f"t_sample={t_sample:.2f}s t_fit={t_fit:.2f}s t_pi={t_pi:.2f}s"
        )


if __name__ == "__main__":
    main()
