# Copyright (c) Ruopeng Gao. All Rights Reserved.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryTokenEncoder(nn.Module):
    def __init__(
            self,
            reid_dim: int,
            d_model: int,
            motion_dim: int = 4,
            hidden_dim: Optional[int] = None,
            num_layers: int = 2,
            dropout: float = 0.0,
            use_layer_norm: bool = True,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2.")
        if hidden_dim is None:
            hidden_dim = d_model

        self.reid_dim = reid_dim
        self.d_model = d_model
        self.motion_dim = motion_dim

        self.pad_emb = nn.Parameter(torch.zeros(reid_dim))
        self.miss_emb = nn.Parameter(torch.zeros(reid_dim))

        input_dim = reid_dim + motion_dim
        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, d_model))
        if use_layer_norm:
            layers.append(nn.LayerNorm(d_model))

        self.mlp = nn.Sequential(*layers)

    def forward(
            self,
            reid_emb: torch.Tensor,
            delta_box: torch.Tensor,
            pad_mask: torch.Tensor,
            miss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pad_mask.dtype != torch.bool:
            pad_mask = pad_mask.to(torch.bool)
        if miss_mask.dtype != torch.bool:
            miss_mask = miss_mask.to(torch.bool)

        identity = reid_emb.clone()
        if miss_mask.any():
            identity[miss_mask] = self.miss_emb
        if pad_mask.any():
            identity[pad_mask] = self.pad_emb

        token_in = torch.cat([identity, delta_box], dim=-1)
        tokens = self.mlp(token_in)
        valid_mask = ~pad_mask
        return tokens, valid_mask


def build_trajectory_token_encoder(
        reid_dim: int,
        d_model: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        num_layers: int = 2,
) -> TrajectoryTokenEncoder:
    return TrajectoryTokenEncoder(
        reid_dim=reid_dim,
        d_model=d_model,
        hidden_dim=hidden_dim,
        dropout=dropout,
        use_layer_norm=use_layer_norm,
        num_layers=num_layers,
    )
