# Copyright (c) Ruopeng Gao. All Rights Reserved.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TrajectoryTokenEncoder(nn.Module):
    def __init__(
            self,
            track_dim: int,
            d_model: int,
            box_dim: int = 4,
            hidden_dim: Optional[int] = None,
            num_layers: int = 2,
            dropout: float = 0.0,
            use_layer_norm: bool = True,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2.")
        if hidden_dim is None:
            hidden_dim = track_dim

        self.track_dim = track_dim
        self.d_model = d_model
        self.box_dim = box_dim

        self.pad_token = nn.Parameter(torch.zeros(d_model))
        self.miss_token = nn.Parameter(torch.zeros(d_model))

        self.track_ln = nn.LayerNorm(track_dim) if use_layer_norm else nn.Identity()

        bbox_layers = []
        in_dim = box_dim
        for _ in range(num_layers - 1):
            bbox_layers.append(nn.Linear(in_dim, hidden_dim))
            bbox_layers.append(nn.GELU())
            if dropout > 0:
                bbox_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        bbox_layers.append(nn.Linear(in_dim, track_dim))
        self.bbox_mlp = nn.Sequential(*bbox_layers)

        self.out_proj = nn.Linear(track_dim, d_model) if track_dim != d_model else None
        self.out_ln = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()

    def forward(
            self,
            track_emb: torch.Tensor,
            bbox_cxcywh: torch.Tensor,
            pad_mask: torch.Tensor,
            miss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pad_mask.dtype != torch.bool:
            pad_mask = pad_mask.to(torch.bool)
        if miss_mask.dtype != torch.bool:
            miss_mask = miss_mask.to(torch.bool)

        track_feat = self.track_ln(track_emb)
        bbox_in = bbox_cxcywh * 2.0 - 1.0
        bbox_feat = self.bbox_mlp(bbox_in)
        fused = track_feat + bbox_feat

        if self.out_proj is not None:
            tokens = self.out_proj(fused)
        else:
            tokens = fused
        tokens = self.out_ln(tokens)

        if miss_mask.any():
            tokens[miss_mask] = self.miss_token
        if pad_mask.any():
            tokens[pad_mask] = self.pad_token
        valid_mask = ~pad_mask
        return tokens, valid_mask


def build_trajectory_token_encoder(
        track_dim: int,
        d_model: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        num_layers: int = 2,
) -> TrajectoryTokenEncoder:
    return TrajectoryTokenEncoder(
        track_dim=track_dim,
        d_model=d_model,
        hidden_dim=hidden_dim,
        dropout=dropout,
        use_layer_norm=use_layer_norm,
        num_layers=num_layers,
    )
