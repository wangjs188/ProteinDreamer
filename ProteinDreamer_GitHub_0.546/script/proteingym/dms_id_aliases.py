from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set


# Legacy dataset IDs -> ProteinGym reference IDs.
DMS_ID_ALIASES: Dict[str, str] = {
    "F7YBW7_MESOW_Ding_2023": "F7YBW8_MESOW_Ding_2023",
    "PSAE_SYNP2_Tsuboyama_2023_1PSE": "PSAE_PICP2_Tsuboyama_2023_1PSE",
    "Q6WV13_9MAXI_Somermeyer_2022": "Q6WV12_9MAXI_Somermeyer_2022",
}


def canonicalize_dms_id(dms_id: str) -> str:
    return DMS_ID_ALIASES.get(str(dms_id), str(dms_id))


def alias_candidates(dms_id: str) -> List[str]:
    """
    Returns candidate IDs to try on disk for a requested canonical ID.
    Order matters: try the input first, then any known alias (reverse mapping).
    """
    dms_id = str(dms_id)
    out: List[str] = [dms_id]
    for k, v in DMS_ID_ALIASES.items():
        if v == dms_id and k not in out:
            out.append(k)
    return out
