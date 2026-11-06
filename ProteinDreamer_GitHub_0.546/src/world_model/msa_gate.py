from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class MsaReliabilityGate(nn.Module):
    """MSA gate: sigmoid(MLP(logNeff, coverage, entropy_norm, plddt_norm)).

    Zero output weights and a negative bias keep initial MSA contributions small.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        init_bias: float = -10.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.init_bias = float(init_bias)
        self.dropout = float(dropout)

        self.net = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, self.init_bias)

    def forward(
        self,
        *,
        log_neff: torch.Tensor,  # [N] or []
        cov: torch.Tensor,  # [N]
        entropy: torch.Tensor,  # [N]
        plddt: torch.Tensor,  # [N]
        eps: float = 1e-8,
    ) -> torch.Tensor:
        if cov.ndim != 1:
            raise ValueError(f"cov must be [N]; got {tuple(cov.shape)}")
        n = int(cov.shape[0])

        if log_neff.ndim == 0:
            log_neff = log_neff.expand(n)
        elif log_neff.ndim == 1 and int(log_neff.shape[0]) == 1:
            log_neff = log_neff.view(1).expand(n)
        elif log_neff.ndim != 1 or int(log_neff.shape[0]) != n:
            raise ValueError(f"log_neff must be [] or [N]; got {tuple(log_neff.shape)} (N={n})")

        if entropy.shape != cov.shape:
            raise ValueError(f"entropy must match cov shape [N]; got {tuple(entropy.shape)} vs {tuple(cov.shape)}")
        if plddt.shape != cov.shape:
            raise ValueError(f"plddt must match cov shape [N]; got {tuple(plddt.shape)} vs {tuple(cov.shape)}")

        plddt_norm = (plddt / 100.0).clamp(0.0, 1.0)
        entropy_norm = (entropy / float(math.log(20.0))).clamp(0.0, 2.0)

        x = torch.stack(
            [
                log_neff.to(dtype=cov.dtype),
                cov,
                entropy_norm.to(dtype=cov.dtype),
                plddt_norm.to(dtype=cov.dtype),
            ],
            dim=-1,
        )
        return torch.sigmoid(self.net(x).view(-1).clamp(min=-50.0, max=50.0) + float(eps))

