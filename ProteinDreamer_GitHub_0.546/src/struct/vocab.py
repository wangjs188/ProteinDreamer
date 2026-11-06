from typing import Dict

FOLDSEEK_3DI_VOCAB = "pynwrqhgdlvtmfsaeikc#"
FOLDSEEK_3DI_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(FOLDSEEK_3DI_VOCAB)}
FOLDSEEK_3DI_MASK_IDX: int = FOLDSEEK_3DI_TO_IDX["#"]

