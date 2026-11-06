from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.msa_cache import (
    choose_msa_file,
    choose_weights_file,
    compute_weighted_profile,
    protein_to_base_id,
    read_full_sequence_length_from_csv,
)


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build MSA-lite cache (.npz) for ProteinDreamer (weighted profile + reliability stats).")

    p.add_argument("--dataset_root", type=str, required=True, help="ProteinGym dataset root with DATASET/<protein>/...")
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

    p.add_argument("--out_dir", type=str, required=True, help="Output directory for per-protein .npz cache files.")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--pseudocount", type=float, default=1e-3)
    p.add_argument("--max_sequences", type=int, default=0, help="Optional cap for faster pilot runs (0 disables).")
    p.add_argument("--reliable_cov_thresh", type=float, default=0.5)
    p.add_argument("--reliable_neff_thresh", type=float, default=50.0)
    p.add_argument("--dry_run", action="store_true", help="Resolve files and print metadata without writing cache.")

    return p


def _read_list(path: str) -> list[str]:
    items: list[str] = []
    with open(path, "r", encoding="utf-8") as r:
        for line in r:
            s = line.strip()
            if s:
                items.append(s)
    return items


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
    import pandas as pd

    df = pd.read_table(path, nrows=1)
    if "mutated_sequence" not in df.columns:
        raise ValueError(f"TSV missing mutated_sequence: {path}")
    seq = str(df["mutated_sequence"].iloc[0])
    return int(len(seq.strip()))


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

        from src.utils.msa_cache import scan_msa

        scan = scan_msa(msa_path)
        w_path = choose_weights_file(args.weights_dir, base_id, msa_n_sequences=scan.n_sequences)
        if w_path is None:
            print(f"[warn] {protein}: no matching weights found for base_id={base_id} n_seq={scan.n_sequences} in {args.weights_dir}")
            continue

        full_len = _read_full_len_from_dataset(args.dataset_root, protein)
        profile = compute_weighted_profile(msa_path, w_path, pseudocount=float(args.pseudocount), max_sequences=int(args.max_sequences))

        # Build full-length arrays (positions outside the MSA range become 'missing').
        start0 = int(profile.start1) - 1
        end0 = int(profile.end1) - 1

        log_uniform = float(math.log(1.0 / 20.0))
        log_probs_full = np.full((full_len, 20), log_uniform, dtype=np.float32)
        cov_full = np.zeros((full_len,), dtype=np.float32)
        ent_full = np.zeros((full_len,), dtype=np.float32)
        has_full = np.zeros((full_len,), dtype=np.uint8)

        # Clip mapping to avoid crashes if the dataset sequence is shorter than the MSA header implies.
        s = max(0, start0)
        e = min(full_len - 1, end0)
        if e >= s:
            src_s = s - start0
            src_e = src_s + (e - s + 1)
            log_probs_full[s : e + 1] = profile.log_probs[src_s:src_e]
            cov_full[s : e + 1] = profile.cov[src_s:src_e]
            ent_full[s : e + 1] = profile.entropy[src_s:src_e]
            has_full[s : e + 1] = 1

        # Simple reliability mask (used as a conservative fallback; the learnable gate uses continuous inputs).
        reliable_full = ((cov_full >= float(args.reliable_cov_thresh)) & (float(profile.neff) >= float(args.reliable_neff_thresh))).astype(
            np.float32
        )
        reliable_full = reliable_full * has_full.astype(np.float32)

        meta: Dict[str, object] = {
            "protein": protein,
            "base_id": base_id,
            "dataset_root": str(args.dataset_root),
            "msa_path": str(msa_path),
            "weights_path": str(w_path),
            "msa_query_header": str(scan.query_header),
            "msa_start1": int(profile.start1),
            "msa_end1": int(profile.end1),
            "msa_aln_len": int(profile.aln_len),
            "msa_n_sequences": int(profile.n_sequences),
            "neff": float(profile.neff),
            "log_neff": float(profile.log_neff),
            "full_len": int(full_len),
            "pseudocount": float(args.pseudocount),
            "max_sequences": int(args.max_sequences),
            "reliable_cov_thresh": float(args.reliable_cov_thresh),
            "reliable_neff_thresh": float(args.reliable_neff_thresh),
        }
        print(f"[ok] {protein} base={base_id} full_len={full_len} msa_len={profile.aln_len} nseq={profile.n_sequences} neff={profile.neff:.2f}")

        if bool(args.dry_run):
            continue

        np.savez_compressed(
            out_path,
            log_probs_msa_full=log_probs_full,
            msa_cov_full=cov_full,
            msa_entropy_full=ent_full,
            msa_reliable_full=reliable_full,
            msa_has_full=has_full,
            neff_global=np.asarray(float(profile.neff), dtype=np.float32),
            log_neff_global=np.asarray(float(profile.log_neff), dtype=np.float32),
            msa_start1=np.asarray(int(profile.start1), dtype=np.int32),
            msa_end1=np.asarray(int(profile.end1), dtype=np.int32),
            msa_nseq=np.asarray(int(profile.n_sequences), dtype=np.int32),
        )
        with open(meta_path, "w", encoding="utf-8") as w:
            json.dump(meta, w, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
