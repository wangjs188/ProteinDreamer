from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from src.utils.mutation import AA_ORDER, AA_TO_IDX


_RANGE_RE = re.compile(r"/(\d+)-(\d+)")
_THETA_RE = re.compile(r"(?:^|_)theta[_]?([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_A3M_INSERTION_DELETE_TABLE = str.maketrans({c: None for c in "abcdefghijklmnopqrstuvwxyz"})


def _strip_a3m_insertions(seq: str) -> str:
    """
    A3M uses lowercase letters to denote insertions w.r.t. the query alignment.
    For profile computations we want alignment columns only, so drop lowercase.
    """
    return str(seq).translate(_A3M_INSERTION_DELETE_TABLE)


def protein_to_base_id(protein_id: str) -> str:
    parts = str(protein_id).split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return str(protein_id)


def parse_msa_range(header: str) -> Optional[Tuple[int, int]]:
    """
    Returns (start1, end1) if header contains /<start>-<end>.
    """
    m = _RANGE_RE.search(str(header))
    if not m:
        return None
    start1, end1 = int(m.group(1)), int(m.group(2))
    if start1 < 1 or end1 < start1:
        return None
    return start1, end1


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


@dataclass(frozen=True)
class MsaScan:
    n_sequences: int
    aln_len: int
    start1: int
    end1: int
    query_header: str


def scan_msa(path: str) -> MsaScan:
    """
    Fast scan to get:
      - number of sequences
      - query alignment length
      - query range (start/end) if present
    """
    n = 0
    query_header = ""
    query_seq_len = 0
    got_query = False
    seq_parts: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as r:
        for raw in r:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                n += 1
                if not got_query:
                    query_header = line[1:].strip()
                    got_query = True
                    seq_parts = []
                else:
                    if query_seq_len == 0 and seq_parts:
                        query_seq_len = len("".join(seq_parts))
                    # We only need the first sequence length; no need to store further.
                continue
            if got_query and query_seq_len == 0:
                seq_parts.append(_strip_a3m_insertions(line))
    if got_query and query_seq_len == 0:
        query_seq_len = len("".join(seq_parts))

    if not got_query or n == 0:
        raise ValueError(f"Empty/invalid MSA: {path}")
    if query_seq_len <= 0:
        raise ValueError(f"Failed to read query length from MSA: {path}")

    rng = parse_msa_range(query_header)
    if rng is None:
        start1, end1 = 1, int(query_seq_len)
    else:
        start1, end1 = int(rng[0]), int(rng[1])
        # Some MSAs may not strictly match the header range; trust the query seq length.
        if (end1 - start1 + 1) != int(query_seq_len):
            end1 = start1 + int(query_seq_len) - 1

    return MsaScan(n_sequences=int(n), aln_len=int(query_seq_len), start1=int(start1), end1=int(end1), query_header=query_header)


def _parse_theta(path: str) -> Optional[float]:
    m = _THETA_RE.search(os.path.basename(path))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def choose_msa_file(msa_dir: str, base_id: str) -> Optional[str]:
    msa_dir_p = Path(msa_dir)
    cands = sorted(str(p) for p in msa_dir_p.glob(f"{base_id}*.a2m"))
    if not cands and "_" in str(base_id):
        # Some ProteinGym MSAs use a shorter prefix than the dataset protein folder name.
        # Fall back to the first token (e.g. "ANCSZ_Hobbs_2022" -> "ANCSZ").
        short = str(base_id).split("_", 1)[0]
        cands = sorted(str(p) for p in msa_dir_p.glob(f"{short}*.a2m"))
    if not cands:
        return None
    scored: List[Tuple[int, int, int, str]] = []
    for p in cands:
        name = os.path.basename(p).lower()
        is_full = 1 if "full" in name else 0
        try:
            scan = scan_msa(p)
            span = int(scan.end1 - scan.start1 + 1)
        except Exception:
            span = -1
        size = int(os.path.getsize(p)) if os.path.exists(p) else 0
        scored.append((is_full, span, size, p))
    scored.sort(reverse=True)
    return scored[0][3]


def choose_weights_file(
    weights_dir: str,
    base_id: str,
    *,
    msa_n_sequences: int,
    prefer_thetas: Sequence[float] = (0.2, 0.01, 0.1),
) -> Optional[str]:
    weights_dir_p = Path(weights_dir)
    cands = sorted(str(p) for p in weights_dir_p.glob(f"{base_id}*.npy"))
    if not cands and "_" in str(base_id):
        short = str(base_id).split("_", 1)[0]
        cands = sorted(str(p) for p in weights_dir_p.glob(f"{short}*.npy"))
    if not cands:
        return None

    matched: List[Tuple[float, float, int, str]] = []
    for p in cands:
        try:
            arr = np.load(p, mmap_mode="r")
        except Exception:
            continue
        if arr.ndim != 1:
            continue
        n_w = int(arr.shape[0])
        # Many ProteinGym-provided weight files are computed on a filtered prefix of the MSA.
        # Accept n_w <= msa_n_sequences and only use the first n_w sequences.
        if n_w <= 0 or n_w > int(msa_n_sequences):
            continue
        theta = _parse_theta(p)
        # default sort key: distance to best preferred theta; unknown theta last
        if theta is None:
            dist = 1e9
        else:
            dist = min(abs(theta - t) for t in prefer_thetas)
        coverage = float(n_w) / float(max(1, int(msa_n_sequences)))
        matched.append((float(dist), -coverage, -n_w, p))
    if not matched:
        return None
    matched.sort()
    return matched[0][3]


def read_full_sequence_length_from_csv(csv_path: str) -> int:
    """
    Reads the first row of a ProteinGym CSV and returns len(mutated_sequence).
    """
    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as r:
        reader = csv.DictReader(r)
        row = next(reader, None)
        if row is None:
            raise ValueError(f"Empty CSV: {csv_path}")
        seq = row.get("mutated_sequence", "")
        if not seq:
            raise ValueError(f"CSV missing mutated_sequence: {csv_path}")
        return int(len(seq.strip()))


@dataclass(frozen=True)
class MsaProfile:
    log_probs: np.ndarray  # [L,20] float32
    cov: np.ndarray  # [L] float32
    entropy: np.ndarray  # [L] float32
    n_sequences: int
    neff: float
    start1: int
    end1: int

    @property
    def aln_len(self) -> int:
        return int(self.log_probs.shape[0])

    @property
    def log_neff(self) -> float:
        return float(math.log(float(self.neff) + 1e-8))


def compute_weighted_profile(
    msa_path: str,
    weights_path: str,
    *,
    pseudocount: float = 1e-3,
    max_sequences: int = 0,
) -> MsaProfile:
    scan = scan_msa(msa_path)
    weights = np.load(weights_path)
    if weights.ndim != 1:
        raise ValueError(f"Expected 1D weights; got shape={weights.shape} for {weights_path}")
    weights_len = int(weights.shape[0])
    if weights_len <= 0:
        raise ValueError(f"Empty weights: shape={weights.shape} for {weights_path}")

    L = int(scan.aln_len)
    counts = np.zeros((L, 20), dtype=np.float64)
    wsum = np.zeros((L,), dtype=np.float64)

    n_used = 0
    for i, (_, seq) in enumerate(_iter_fasta_records(msa_path)):
        if i >= weights_len:
            break
        if max_sequences and n_used >= int(max_sequences):
            break
        seq = _strip_a3m_insertions(seq)
        w = float(weights[i])
        if w <= 0:
            n_used += 1
            continue
        if len(seq) != L:
            raise ValueError(f"Alignment length mismatch at i={i}: len(seq)={len(seq)} expected={L} ({msa_path})")
        for j, ch in enumerate(seq):
            if ch == "." or ch == "-":
                continue
            aa = ch.upper()
            idx = AA_TO_IDX.get(aa)
            if idx is None:
                continue
            counts[j, idx] += w
            wsum[j] += w
        n_used += 1

    neff = float(weights[:n_used].sum()) if n_used > 0 else 0.0
    if neff <= 0:
        # No usable sequences; return uniform profile.
        uni = np.full((L, 20), 1.0 / 20.0, dtype=np.float32)
        log_probs = np.log(uni)
        cov = np.zeros((L,), dtype=np.float32)
        entropy = np.zeros((L,), dtype=np.float32)
        return MsaProfile(
            log_probs=log_probs,
            cov=cov,
            entropy=entropy,
            n_sequences=int(n_used),
            neff=float(neff),
            start1=int(scan.start1),
            end1=int(scan.end1),
        )

    pc = float(pseudocount)
    denom = wsum + 20.0 * pc
    probs = (counts + pc) / denom.reshape(-1, 1)
    probs = probs.astype(np.float32)

    # Positions with zero coverage: set to uniform (and mark cov=0, entropy=0).
    zero = wsum <= 0
    if bool(np.any(zero)):
        probs[zero] = 1.0 / 20.0

    log_probs = np.log(probs + 1e-12, dtype=np.float32)
    cov = (wsum / max(neff, 1e-8)).astype(np.float32)
    cov[zero] = 0.0

    entropy = (-probs * np.log(probs + 1e-12)).sum(axis=1).astype(np.float32)
    entropy[zero] = 0.0

    return MsaProfile(
        log_probs=log_probs,
        cov=cov,
        entropy=entropy,
        n_sequences=int(n_used),
        neff=float(neff),
        start1=int(scan.start1),
        end1=int(scan.end1),
    )
