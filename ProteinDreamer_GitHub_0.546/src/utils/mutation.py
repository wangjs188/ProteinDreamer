from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Sequence

AA_ORDER: Sequence[str] = (
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
)
AA_TO_IDX: Dict[str, int] = {aa: i for i, aa in enumerate(AA_ORDER)}


@dataclass(frozen=True)
class SingleMutation:
    wt: str
    pos: int  # 0-based
    mt: str

    @property
    def wt_idx(self) -> int:
        return AA_TO_IDX[self.wt]

    @property
    def mt_idx(self) -> int:
        return AA_TO_IDX[self.mt]


def split_mutant_string(mutant: str) -> List[str]:
    if ":" in mutant:
        return mutant.split(":")
    if ";" in mutant:
        return mutant.split(";")
    # Split hyphens only in substitution strings such as A123V-C124D.
    if "-" in mutant:
        s = mutant.strip()
        if re.fullmatch(r"[A-Z][0-9]+[A-Z](?:-[A-Z][0-9]+[A-Z])+", s):
            return s.split("-")
    return [mutant]


def parse_mutant(mutant: str, *, offset_idx: int = 1) -> List[SingleMutation]:
    """
    Parses substitutions like "A123B" or "A123B:C124D" (1-indexed by default).
    Returns [] for "wt".
    """
    if mutant.lower() == "wt":
        return []
    out: List[SingleMutation] = []
    for row in split_mutant_string(mutant):
        wt, pos1, mt = row[0], int(row[1:-1]), row[-1]
        pos0 = pos1 - offset_idx
        out.append(SingleMutation(wt=wt, pos=pos0, mt=mt))
    return out
