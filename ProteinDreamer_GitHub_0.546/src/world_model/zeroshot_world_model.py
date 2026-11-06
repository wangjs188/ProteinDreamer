from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SingleMutation:
    pos: int
    wt_aa_idx: int
    mut_aa_idx: int
    y: float


class ActionStateTransition(nn.Module):
    """
    Single-step action-conditioned state transition:
      (WT residue embedding, wt_aa, mut_aa, ΔlogP[opt]) -> Δembedding
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        aa_emb_dim: int = 32,
        mlp_hidden_dim: int = 1024,
        dropout: float = 0.1,
        use_delta_logp: bool = True,
        extra_feat_dim: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_delta_logp = bool(use_delta_logp)
        self.extra_feat_dim = int(extra_feat_dim)

        self.aa_emb = nn.Embedding(20, int(aa_emb_dim))
        in_dim = self.hidden_dim + 2 * int(aa_emb_dim) + (1 if self.use_delta_logp else 0) + self.extra_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, int(mlp_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(mlp_hidden_dim), int(mlp_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(mlp_hidden_dim), self.hidden_dim),
        )

    def forward(
        self,
        *,
        wt_embed: torch.Tensor,  # [N,H]
        wt_aa_idx: torch.Tensor,  # [N]
        mut_aa_idx: torch.Tensor,  # [N]
        delta_logp: Optional[torch.Tensor] = None,  # [N]
        extra_feat: Optional[torch.Tensor] = None,  # [N,F]
    ) -> torch.Tensor:
        wt_emb = self.aa_emb(wt_aa_idx.long())
        mut_emb = self.aa_emb(mut_aa_idx.long())
        parts = [wt_embed, wt_emb, mut_emb]
        if self.use_delta_logp:
            if delta_logp is None:
                raise ValueError("delta_logp is required when use_delta_logp=True")
            parts.append(delta_logp.view(-1, 1).to(wt_embed.dtype))
        if self.extra_feat_dim > 0:
            if extra_feat is None:
                raise ValueError("extra_feat is required when extra_feat_dim>0")
            if extra_feat.ndim == 1:
                extra_feat = extra_feat.view(-1, self.extra_feat_dim)
            if extra_feat.ndim != 2 or int(extra_feat.shape[1]) != self.extra_feat_dim:
                raise ValueError(f"extra_feat must be [N,{self.extra_feat_dim}]; got {tuple(extra_feat.shape)}")
            parts.append(extra_feat.to(dtype=wt_embed.dtype))
        x = torch.cat(parts, dim=-1)
        return self.mlp(x)


class RewardUncertaintyHead(nn.Module):
    """
    Reward head with aleatoric uncertainty:
      (predicted mutant embed, wt embed, action, ΔlogP[opt]) -> (μ, logσ²)
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        aa_emb_dim: int = 32,
        mlp_hidden_dim: int = 1024,
        dropout: float = 0.1,
        use_delta_logp: bool = True,
        extra_feat_dim: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_delta_logp = bool(use_delta_logp)
        self.extra_feat_dim = int(extra_feat_dim)

        self.aa_emb = nn.Embedding(20, int(aa_emb_dim))
        in_dim = 2 * self.hidden_dim + 2 * int(aa_emb_dim) + (1 if self.use_delta_logp else 0) + self.extra_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, int(mlp_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(mlp_hidden_dim), int(mlp_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(mlp_hidden_dim), 2),
        )

    def forward(
        self,
        *,
        wt_embed: torch.Tensor,  # [N,H]
        mut_embed_pred: torch.Tensor,  # [N,H]
        wt_aa_idx: torch.Tensor,  # [N]
        mut_aa_idx: torch.Tensor,  # [N]
        delta_logp: Optional[torch.Tensor] = None,  # [N]
        extra_feat: Optional[torch.Tensor] = None,  # [N,F]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        wt_emb = self.aa_emb(wt_aa_idx.long())
        mut_emb = self.aa_emb(mut_aa_idx.long())

        parts = [wt_embed, mut_embed_pred, wt_emb, mut_emb]
        if self.use_delta_logp:
            if delta_logp is None:
                raise ValueError("delta_logp is required when use_delta_logp=True")
            parts.append(delta_logp.view(-1, 1).to(wt_embed.dtype))
        if self.extra_feat_dim > 0:
            if extra_feat is None:
                raise ValueError("extra_feat is required when extra_feat_dim>0")
            if extra_feat.ndim == 1:
                extra_feat = extra_feat.view(-1, self.extra_feat_dim)
            if extra_feat.ndim != 2 or int(extra_feat.shape[1]) != self.extra_feat_dim:
                raise ValueError(f"extra_feat must be [N,{self.extra_feat_dim}]; got {tuple(extra_feat.shape)}")
            parts.append(extra_feat.to(dtype=wt_embed.dtype))

        x = torch.cat(parts, dim=-1)
        out = self.mlp(x)
        mu = out[:, 0].contiguous()
        log_var = out[:, 1].contiguous().clamp(min=-10.0, max=10.0)
        return mu, log_var
