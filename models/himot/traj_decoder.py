# Copyright (c) Ruopeng Gao. All Rights Reserved.

from __future__ import annotations

from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dropout: float = 0.0,
            attn_dropout: float = 0.0,
            rope_base: int = 10000,
            max_seq_len: int = 64,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.rope_base = rope_base
        self.max_seq_len = max_seq_len

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.attn_drop = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()

        inv_freq = 1.0 / (rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rope(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # x: (N, H, L, D)
        n, h, l, d = x.shape
        d_rot = (d // 2) * 2
        if d_rot == 0:
            return x
        x_rot = x[..., :d_rot]
        x_pass = x[..., d_rot:] if d_rot < d else None

        pos = position_ids.to(x.device).unsqueeze(-1)  # (N, L, 1)
        inv_freq = self.inv_freq.to(x.device).to(x.dtype)
        freqs = pos * inv_freq  # (N, L, D/2)
        cos = freqs.cos().unsqueeze(1)  # (N, 1, L, D/2)
        sin = freqs.sin().unsqueeze(1)  # (N, 1, L, D/2)

        x_even = x_rot[..., 0::2]
        x_odd = x_rot[..., 1::2]
        x_rotated = torch.empty_like(x_rot)
        x_rotated[..., 0::2] = x_even * cos - x_odd * sin
        x_rotated[..., 1::2] = x_even * sin + x_odd * cos

        if x_pass is not None:
            x = torch.cat([x_rotated, x_pass], dim=-1)
        else:
            x = x_rotated
        return x

    def forward(
            self,
            x: torch.Tensor,
            position_ids: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            cache: Optional[dict] = None,
    ) -> tuple[torch.Tensor, dict]:
        n, l, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(n, l, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(n, l, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(n, l, self.nhead, self.head_dim).transpose(1, 2)

        q = self._apply_rope(q, position_ids)
        k = self._apply_rope(k, position_ids)

        past_k = past_v = past_key_padding_mask = None
        past_len = 0
        if cache is not None and cache.get("k") is not None:
            past_k = cache["k"]
            past_v = cache["v"]
            past_key_padding_mask = cache.get("key_padding_mask", None)
            past_len = past_k.shape[2]

        if past_k is not None:
            k_all = torch.cat([past_k, k], dim=2)
            v_all = torch.cat([past_v, v], dim=2)
        else:
            k_all = k
            v_all = v

        total_len = k_all.shape[2]
        attn_scores = torch.matmul(q, k_all.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask[:, None, None, :],
                float("-inf"),
            )

        if past_len == 0:
            causal_mask = torch.triu(
                torch.ones(l, total_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
        else:
            q_pos = torch.arange(l, device=x.device) + past_len
            k_pos = torch.arange(total_len, device=x.device)
            causal_mask = k_pos[None, :] > q_pos[:, None]
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
        attn_probs = self.attn_drop(attn_probs)
        attn_out = torch.matmul(attn_probs, v_all)
        attn_out = attn_out.transpose(1, 2).contiguous().view(n, l, self.d_model)
        attn_out = self.out_proj(attn_out)
        attn_out = self.resid_dropout(attn_out)

        return attn_out, {"k": k, "v": v}


class TrajectoryDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int,
            dropout: float = 0.0,
            attn_dropout: float = 0.0,
            use_layer_norm: bool = True,
            rope_base: int = 10000,
            max_seq_len: int = 64,
    ):
        super().__init__()
        self.self_attn = CausalSelfAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            attn_dropout=attn_dropout,
            rope_base=rope_base,
            max_seq_len=max_seq_len,
        )
        self.norm1 = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(
            self,
            x: torch.Tensor,
            position_ids: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            cache: Optional[dict] = None,
    ) -> tuple[torch.Tensor, dict]:
        attn_in = self.norm1(x)
        attn_out, kv = self.self_attn(
            attn_in,
            position_ids=position_ids,
            key_padding_mask=key_padding_mask,
            cache=cache,
        )
        x = x + attn_out
        ffn_in = self.norm2(x)
        ffn_out = self.ffn(ffn_in)
        x = x + self.dropout(ffn_out)
        return x, kv


class TrajectoryDecoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            num_layers: int = 6,
            dim_feedforward: Optional[int] = None,
            dropout: float = 0.0,
            attn_dropout: float = 0.0,
            use_layer_norm: bool = True,
            rope_base: int = 10000,
            max_seq_len: int = 64,
            reid_dim: Optional[int] = None,
            motion_dim: int = 4,
            l2norm_reid: bool = True,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        if reid_dim is None:
            reid_dim = d_model

        self.d_model = d_model
        self.reid_dim = reid_dim
        self.motion_dim = motion_dim
        self.l2norm_reid = l2norm_reid

        self.spec_token = nn.Parameter(torch.zeros(d_model))

        self.layers = nn.ModuleList([
            TrajectoryDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                attn_dropout=attn_dropout,
                use_layer_norm=use_layer_norm,
                rope_base=rope_base,
                max_seq_len=max_seq_len,
            )
            for _ in range(num_layers)
        ])

        self.reid_head = nn.Linear(d_model, reid_dim)
        self.motion_head = nn.Linear(d_model, motion_dim)

    def _build_position_ids(
            self,
            valid_mask: torch.Tensor,
            past_valid_count: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if past_valid_count is None:
            past_valid_count = torch.zeros(
                valid_mask.shape[0],
                dtype=torch.long,
                device=valid_mask.device,
            )
        valid_int = valid_mask.to(torch.long)
        pos = torch.cumsum(valid_int, dim=1) - 1
        pos = pos + past_valid_count[:, None]
        pos = torch.where(valid_mask, pos, torch.zeros_like(pos))
        valid_count = past_valid_count + valid_int.sum(dim=1)
        return pos, valid_count

    def forward(
            self,
            tokens: torch.Tensor,
            valid_mask: torch.Tensor,
            cache: Optional[dict] = None,
            return_cache: bool = False,
    ):
        n, l, _ = tokens.shape
        spec_token = self.spec_token.view(1, 1, -1).expand(n, 1, -1)
        tokens = torch.cat([tokens, spec_token], dim=1)
        valid_mask = torch.cat(
            [valid_mask, torch.ones((n, 1), dtype=torch.bool, device=valid_mask.device)],
            dim=1,
        )

        past_valid_count = None
        if cache is not None and cache.get("valid_count") is not None:
            past_valid_count = cache["valid_count"]
        pos_hist, valid_count = self._build_position_ids(valid_mask[:, :-1], past_valid_count)
        pos_s = valid_count[:, None]
        position_ids = torch.cat([pos_hist, pos_s], dim=1)

        key_padding_mask = ~valid_mask

        new_cache_layers = []
        x = tokens
        for idx, layer in enumerate(self.layers):
            layer_cache = None
            if cache is not None and cache.get("layers") is not None and idx < len(cache["layers"]):
                layer_cache = cache["layers"][idx]
            combined_key_padding_mask = key_padding_mask
            if layer_cache is not None and layer_cache.get("key_padding_mask") is not None:
                combined_key_padding_mask = torch.cat(
                    [layer_cache["key_padding_mask"], key_padding_mask],
                    dim=1,
                )

            x, kv = layer(
                x=x,
                position_ids=position_ids,
                key_padding_mask=combined_key_padding_mask,
                cache=layer_cache,
            )

            if return_cache:
                current_k = kv["k"][:, :, :-1, :]
                current_v = kv["v"][:, :, :-1, :]
                current_mask = key_padding_mask[:, :-1]
                if layer_cache is not None and layer_cache.get("k") is not None:
                    new_k = torch.cat([layer_cache["k"], current_k], dim=2)
                    new_v = torch.cat([layer_cache["v"], current_v], dim=2)
                    new_mask = torch.cat([layer_cache["key_padding_mask"], current_mask], dim=1)
                else:
                    new_k = current_k
                    new_v = current_v
                    new_mask = current_mask
                new_cache_layers.append({
                    "k": new_k,
                    "v": new_v,
                    "key_padding_mask": new_mask,
                })

        h_s = x[:, -1, :]
        pred_reid = self.reid_head(h_s)
        if self.l2norm_reid:
            pred_reid = F.normalize(pred_reid, p=2, dim=-1)
        pred_delta_box = self.motion_head(h_s)

        if return_cache:
            new_cache = {
                "layers": new_cache_layers,
                "valid_count": valid_count.detach(),
            }
            return pred_reid, pred_delta_box, new_cache
        return pred_reid, pred_delta_box


def build_trajectory_decoder(
        d_model: int,
        nhead: int,
        num_layers: int = 6,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
        rope_base: int = 10000,
        max_seq_len: int = 64,
        reid_dim: Optional[int] = None,
        l2norm_reid: bool = True,
):
    if dim_feedforward is None:
        dim_feedforward = 4 * d_model
    if reid_dim is None:
        reid_dim = d_model
    return TrajectoryDecoder(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        attn_dropout=dropout,
        rope_base=rope_base,
        max_seq_len=max_seq_len,
        reid_dim=reid_dim,
        l2norm_reid=l2norm_reid,
    )
