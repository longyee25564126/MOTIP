# Copyright (c) Ruopeng Gao. All Rights Reserved.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReIDHead(nn.Module):
    def __init__(
            self,
            input_dim: int,
            embed_dim: Optional[int] = None,
            hidden_dim: Optional[int] = None,
            num_layers: int = 2,
            dropout: float = 0.0,
            use_layer_norm: bool = True,
            l2norm: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        embed_dim = input_dim if embed_dim is None else embed_dim
        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        self.l2norm = l2norm

        layers = []
        in_dim = input_dim
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, embed_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(embed_dim))
        else:
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(in_dim, hidden_dim))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, embed_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(embed_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reid_emb = self.mlp(x)
        if self.l2norm:
            reid_emb = F.normalize(reid_emb, p=2, dim=-1)
        return reid_emb


def build_reid_head(config: dict, input_dim: Optional[int] = None) -> ReIDHead:
    if input_dim is None:
        input_dim = config["DETR_HIDDEN_DIM"]
    embed_dim = config.get("REID_EMBED_DIM", input_dim)
    hidden_dim = config.get("REID_HIDDEN_DIM", input_dim)
    num_layers = config.get("REID_NUM_LAYERS", 2)
    dropout = config.get("REID_DROPOUT", 0.0)
    use_layer_norm = config.get("REID_USE_LAYER_NORM", True)
    l2norm = config.get("REID_L2NORM", True)
    return ReIDHead(
        input_dim=input_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_layer_norm=use_layer_norm,
        l2norm=l2norm,
    )
