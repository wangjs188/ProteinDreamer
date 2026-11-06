from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from Bio.PDB import MMCIFParser, PDBParser
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.utils.mutation import AA_ORDER, AA_TO_IDX


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def structure_candidates(dataset_root: str | Path, protein: str) -> list[Path]:
    root = Path(dataset_root)
    return [
        root / "DATASET" / protein / f"{protein}.pdb",
        root / "DATASET" / protein / f"{protein}.cif",
        root / protein / f"{protein}.pdb",
        root / protein / f"{protein}.cif",
        root / f"{protein}.pdb",
        root / f"{protein}.cif",
    ]


def resolve_structure_path(dataset_root: str | Path, protein: str) -> Path:
    for path in structure_candidates(dataset_root, protein):
        if path.is_file():
            return path
    raise FileNotFoundError(f"No PDB/mmCIF structure found for {protein} under {dataset_root}")


def load_structure_sequence(path: str | Path) -> str:
    """Return standard residues with complete N/CA/C atoms in model 0, chain order."""
    path = Path(path)
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() == ".cif" else PDBParser(QUIET=True)
    model = parser.get_structure("protein", str(path))[0]
    sequence: list[str] = []
    for chain in model:
        for residue in chain:
            if residue.get_resname() == "HOH":
                continue
            if not all(atom in residue for atom in ("N", "CA", "C")):
                continue
            resname = residue.get_resname().upper()
            if resname not in THREE_TO_ONE:
                raise ValueError(f"Unsupported residue {resname} in {path}")
            sequence.append(THREE_TO_ONE[resname])
    if not sequence:
        raise ValueError(f"No complete standard residues found in {path}")
    return "".join(sequence)


def sequence_data(sequence: str, protein: str) -> Data:
    try:
        y = torch.tensor([AA_TO_IDX[aa] for aa in sequence], dtype=torch.long)
    except KeyError as exc:
        raise ValueError(f"Non-standard amino acid in {protein}: {exc.args[0]}") from exc
    if "".join(AA_ORDER[int(i)] for i in y) != sequence:
        raise ValueError(f"Amino-acid encoding failed for {protein}")
    return Data(x=F.one_hot(y, num_classes=20).to(torch.float32), y=y,
                protein_name=protein)


class ProteinGymSaProtDataset(Dataset):
    """Read one wild-type sequence per ProteinGym assay directly from its structure."""

    def __init__(self, root: str | Path, proteins: Iterable[str] | None = None):
        self.root = Path(root)
        data_root = self.root / "DATASET"
        if proteins is None:
            if not data_root.is_dir():
                raise FileNotFoundError(f"Missing ProteinGym DATASET directory: {data_root}")
            self.protein_names = sorted(p.name for p in data_root.iterdir() if p.is_dir())
        else:
            self.protein_names = list(dict.fromkeys(str(p) for p in proteins))
        if not self.protein_names:
            raise ValueError("No ProteinGym assays selected")

    def __len__(self) -> int:
        return len(self.protein_names)

    def __getitem__(self, index: int) -> Data:
        protein = self.protein_names[index]
        path = resolve_structure_path(self.root, protein)
        return sequence_data(load_structure_sequence(path), protein)
