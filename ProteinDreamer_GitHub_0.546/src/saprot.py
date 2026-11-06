from __future__ import annotations

import gc
import os
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from transformers import AutoModelForMaskedLM

from src.struct.vocab import FOLDSEEK_3DI_VOCAB
from src.utils.mutation import AA_ORDER


def _resolve_device(requested: str | None) -> torch.device:
    if requested and requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested == "cuda" and torch.cuda.is_available():
        rank = os.environ.get("LOCAL_RANK")
        if rank is not None and rank.isdigit():
            return torch.device(f"cuda:{int(rank)}")
    return torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))


class SaProtBackbone(nn.Module):
    """Frozen SaProt masked language model with AA+3Di input construction."""

    def __init__(self, model_dir: str, device: str = "cuda") -> None:
        super().__init__()
        self.device = _resolve_device(device)
        self.model = AutoModelForMaskedLM.from_pretrained(model_dir).to(self.device)
        vocab_path = os.path.join(model_dir, "vocab.txt")
        if not os.path.isfile(vocab_path):
            raise FileNotFoundError(f"SaProt vocab.txt not found: {vocab_path}")
        with open(vocab_path, encoding="utf-8") as handle:
            tokens = [line.strip() for line in handle if line.strip()]
        self.token_to_id = {token: i for i, token in enumerate(tokens)}
        for token in ("<cls>", "<pad>", "<eos>", "<unk>", "<mask>"):
            if token not in self.token_to_id:
                raise ValueError(f"SaProt vocabulary is missing {token}: {vocab_path}")
        self.ids_by_3di = {
            di: torch.tensor(
                [self.token_to_id.get(f"{aa}{di}", self.token_to_id["<unk>"]) for aa in AA_ORDER],
                dtype=torch.long,
                device=self.device,
            )
            for di in FOLDSEEK_3DI_VOCAB
        }

    @staticmethod
    def _sequence(data: Data) -> str:
        if not hasattr(data, "x") or data.x.ndim != 2 or data.x.shape[1] < 20:
            raise ValueError("SaProt input must contain x[L,20] amino-acid one-hot features")
        indices = data.x[:, :20].argmax(dim=1).detach().cpu().tolist()
        return "".join(AA_ORDER[int(i)] for i in indices)

    @staticmethod
    def _structure_chars(data: Data, length: int) -> list[str]:
        if not hasattr(data, "struct_tokens") or data.struct_tokens is None:
            raise ValueError("SaProt input is missing Foldseek 3Di tokens")
        values = data.struct_tokens.detach().cpu().view(-1).tolist()
        if len(values) != length:
            raise ValueError(f"3Di length {len(values)} does not match sequence length {length}")
        try:
            return [FOLDSEEK_3DI_VOCAB[int(i)] for i in values]
        except (IndexError, ValueError) as exc:
            raise ValueError("Invalid Foldseek 3Di token index") from exc

    @torch.no_grad()
    def forward(self, data: Data | list[Data]) -> Batch:
        items = data if isinstance(data, list) else [data]
        sequences = [self._sequence(item) for item in items]
        structure = [self._structure_chars(item, len(seq)) for item, seq in zip(items, sequences)]
        maximum = max(len(seq) for seq in sequences)
        ids = torch.full(
            (len(items), maximum + 2), self.token_to_id["<pad>"],
            dtype=torch.long, device=self.device,
        )
        attention = torch.zeros_like(ids)
        for row, (sequence, di_chars) in enumerate(zip(sequences, structure)):
            encoded = [self.token_to_id["<cls>"]]
            encoded.extend(
                self.token_to_id.get(f"{aa}{di}", self.token_to_id["<unk>"])
                for aa, di in zip(sequence, di_chars)
            )
            encoded.append(self.token_to_id["<eos>"])
            ids[row, :len(encoded)] = torch.tensor(encoded, dtype=torch.long, device=self.device)
            attention[row, :len(encoded)] = 1

        output = self.model(input_ids=ids, attention_mask=attention, output_hidden_states=True)
        if output.hidden_states is None:
            raise RuntimeError("SaProt did not return encoder hidden states")
        hidden = output.hidden_states[-1]
        for row, item in enumerate(items):
            length = len(sequences[row])
            item.esm_rep = hidden[row, 1:1 + length]
            vocab_logits = output.logits[row, 1:1 + length]
            aa_logits = torch.empty((length, 20), dtype=vocab_logits.dtype,
                                    device=vocab_logits.device)
            for position, di in enumerate(structure[row]):
                aa_logits[position] = vocab_logits[position].index_select(0, self.ids_by_3di[di])
            item.saprot_logits = aa_logits
            if hasattr(item, "seq"):
                del item.seq
        items = [item.to(self.device) for item in items]
        batch = Batch.from_data_list(items)
        if os.environ.get("PROTDREAMER_PLM_GC", "1") != "0":
            gc.collect()
        if os.environ.get("PROTDREAMER_PLM_EMPTY_CACHE", "1") != "0" and self.device.type == "cuda":
            torch.cuda.empty_cache()
        return batch

    @torch.no_grad()
    def saprot_pll_log_probs(
        self,
        data: Data,
        *,
        positions: Sequence[int] | None = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Return masked-position SaProt log probabilities over the 20 amino acids."""
        y = data.y.detach().cpu().view(-1).tolist()
        sequence = "".join(AA_ORDER[int(i)] for i in y)
        di_chars = self._structure_chars(data, len(sequence))
        length = len(sequence)
        selected = list(range(length)) if positions is None else [int(p) for p in positions]
        selected = [p for p in selected if 0 <= p < length]
        result = torch.zeros((length, 20), dtype=torch.float32, device=self.device)
        if not selected:
            return result

        base = torch.full((length + 2,), self.token_to_id["<pad>"],
                          dtype=torch.long, device=self.device)
        base[0] = self.token_to_id["<cls>"]
        base[-1] = self.token_to_id["<eos>"]
        for position, (aa, di) in enumerate(zip(sequence, di_chars)):
            base[position + 1] = self.token_to_id.get(
                f"{aa}{di}", self.token_to_id["<unk>"])
        attention = torch.ones((length + 2,), dtype=torch.long, device=self.device)

        for start in range(0, len(selected), int(batch_size)):
            batch_positions = selected[start:start + int(batch_size)]
            input_ids = base.unsqueeze(0).repeat(len(batch_positions), 1).contiguous()
            rows = torch.arange(len(batch_positions), dtype=torch.long, device=self.device)
            columns = torch.tensor([p + 1 for p in batch_positions], dtype=torch.long,
                                   device=self.device)
            input_ids[rows, columns] = self.token_to_id["<mask>"]
            masks = attention.unsqueeze(0).repeat(len(batch_positions), 1).contiguous()
            logits = self.model(input_ids=input_ids, attention_mask=masks).logits[rows, columns]
            for row, position in enumerate(batch_positions):
                aa_logits = logits[row].index_select(0, self.ids_by_3di[di_chars[position]])
                result[position] = aa_logits.log_softmax(dim=-1).to(torch.float32)
        return result
