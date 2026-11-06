import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from Bio.PDB import MMCIFParser, PDBParser

from src.struct.vocab import FOLDSEEK_3DI_TO_IDX, FOLDSEEK_3DI_VOCAB


@dataclass(frozen=True)
class Foldseek3DiMeta:
    struct_path: str
    chain_id: str
    foldseek_bin: str
    plddt_mask: str
    plddt_threshold: float
    struct_mtime: float
    struct_size: int
    vocab: str = FOLDSEEK_3DI_VOCAB


def _resolve_foldseek_binary(foldseek_bin: str) -> str:
    candidate = str(foldseek_bin).strip()
    if not candidate:
        raise ValueError("foldseek_bin must be a non-empty command or absolute path")

    has_path_sep = os.path.sep in candidate or (os.path.altsep and os.path.altsep in candidate)
    if has_path_sep:
        resolved = os.path.abspath(candidate)
        if os.path.isfile(resolved) and os.access(resolved, os.X_OK):
            return resolved
        raise FileNotFoundError(f"Foldseek binary not found or not executable: {candidate}")

    resolved = shutil.which(candidate)
    if resolved:
        return os.path.abspath(resolved)

    raise FileNotFoundError(
        f"Foldseek binary '{candidate}' was not found in PATH. "
        "Install foldseek or set FOLDSEEK_BIN=/abs/path/to/foldseek."
    )


def encode_3di(struc_seq: str) -> torch.LongTensor:
    struc_seq = struc_seq.lower()
    try:
        idx = [FOLDSEEK_3DI_TO_IDX[c] for c in struc_seq]
    except KeyError as e:
        raise ValueError(f"Unexpected 3Di token: {e} (seq={struc_seq[:80]}...)") from e
    return torch.tensor(idx, dtype=torch.long)


def decode_3di(tokens: Sequence[int]) -> str:
    return "".join(FOLDSEEK_3DI_VOCAB[int(i)] for i in tokens)


def _extract_plddt(struct_path: str, chain_id: str) -> np.ndarray:
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


def _infer_plddt_mask_mode(struct_path: str, plddt_mask: str) -> str:
    if plddt_mask in {"on", "off"}:
        return plddt_mask
    if plddt_mask != "auto":
        raise ValueError("plddt_mask must be one of: auto|on|off")
    try:
        with open(struct_path, "r", encoding="utf-8", errors="ignore") as r:
            # AF2 and ESMFold commonly store pLDDT in the B-factor column.
            txt = r.read().lower()
            return "on" if ("alphafold" in txt or "esmfold" in txt) else "off"
    except OSError:
        return "off"


def get_struc_seq_foldseek(
    foldseek_bin: str,
    struct_path: str,
    chains: Optional[Sequence[str]] = None,
    plddt_mask: str = "auto",
    plddt_threshold: float = 70.0,
    foldseek_verbose: bool = False,
) -> Dict[str, Tuple[str, str]]:
    """
    Returns:
        dict(chain_id -> (aa_seq, struc_seq)) where struc_seq is Foldseek 3Di characters (may include '#').
    """
    foldseek_exec = _resolve_foldseek_binary(foldseek_bin)
    if not os.path.exists(struct_path):
        raise FileNotFoundError(f"Structure not found: {struct_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_tsv = os.path.join(tmpdir, "foldseek_3di.tsv")
        cmd = [
            foldseek_exec,
            "structureto3didescriptor",
            "--threads",
            "1",
            "--chain-name-mode",
            "1",
        ]
        if not foldseek_verbose:
            cmd += ["-v", "0"]
        cmd += [struct_path, out_tsv]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Foldseek failed (code={proc.returncode}) for {struct_path}: {proc.stderr[-2000:]}"
            )

        mask_mode = _infer_plddt_mask_mode(struct_path, plddt_mask)
        seq_dict: Dict[str, Tuple[str, str]] = {}
        name = os.path.basename(struct_path)
        with open(out_tsv, "r", encoding="utf-8", errors="ignore") as r:
            for line in r:
                desc, aa_seq, struc_seq = line.split("\t")[:3]
                name_chain = desc.split(" ")[0]
                chain_id = name_chain.replace(name, "").split("_")[-1]

                if chains is not None and chain_id not in set(chains):
                    continue

                struc_seq = struc_seq.strip()
                aa_seq = aa_seq.strip()

                if mask_mode == "on":
                    try:
                        plddts = _extract_plddt(struct_path, chain_id)
                        if len(plddts) == len(struc_seq):
                            low = np.where(plddts < plddt_threshold)[0]
                            np_seq = np.array(list(struc_seq), dtype="<U1")
                            np_seq[low] = "#"
                            struc_seq = "".join(np_seq.tolist())
                    except Exception:
                        pass

                seq_dict[chain_id] = (aa_seq, struc_seq)

        return seq_dict


def get_3di_tokens_cached(
    *,
    protein_name: str,
    struct_path: str,
    foldseek_bin: str,
    cache_dir: str,
    chain_id: str = "A",
    plddt_mask: str = "auto",
    plddt_threshold: float = 70.0,
) -> torch.LongTensor:
    """
    Cache format:
      {cache_dir}/{protein_name}.pt          -> torch.LongTensor [L]
      {cache_dir}/{protein_name}.meta.json  -> Foldseek3DiMeta
    """
    os.makedirs(cache_dir, exist_ok=True)
    tok_path = os.path.join(cache_dir, f"{protein_name}.pt")
    meta_path = os.path.join(cache_dir, f"{protein_name}.meta.json")

    foldseek_exec = _resolve_foldseek_binary(foldseek_bin)
    st = os.stat(struct_path)
    meta = Foldseek3DiMeta(
        struct_path=os.path.abspath(struct_path),
        chain_id=chain_id,
        foldseek_bin=foldseek_exec,
        plddt_mask=plddt_mask,
        plddt_threshold=float(plddt_threshold),
        struct_mtime=float(st.st_mtime),
        struct_size=int(st.st_size),
    )

    if os.path.exists(tok_path) and os.path.exists(meta_path):
        try:
            old = json.load(open(meta_path, "r", encoding="utf-8"))
            if old == asdict(meta):
                return torch.load(tok_path)
        except Exception:
            pass

    seq_dict = get_struc_seq_foldseek(
        foldseek_bin=foldseek_exec,
        struct_path=struct_path,
        chains=[chain_id],
        plddt_mask=plddt_mask,
        plddt_threshold=plddt_threshold,
    )
    if chain_id not in seq_dict:
        raise ValueError(f"Chain '{chain_id}' not found in foldseek output for {struct_path}")
    _, struc_seq = seq_dict[chain_id]
    tokens = encode_3di(struc_seq.lower())

    torch.save(tokens, tok_path)
    with open(meta_path, "w", encoding="utf-8") as w:
        json.dump(asdict(meta), w, ensure_ascii=False, indent=2)
    return tokens


def load_3di_tokens_from_cache(
    *,
    protein_name: str,
    cache_dir: str,
    map_location: Optional[object] = "cpu",
) -> Optional[torch.LongTensor]:
    """Load cached 3Di tokens without checking meta.json."""
    tok_path = os.path.join(str(cache_dir), f"{protein_name}.pt")
    if not os.path.exists(tok_path):
        return None
    try:
        tokens = torch.load(tok_path, map_location=map_location)
    except Exception:
        return None
    if not isinstance(tokens, torch.Tensor):
        return None
    return tokens.to(dtype=torch.long).view(-1)
