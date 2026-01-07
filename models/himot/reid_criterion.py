# Copyright (c) Ruopeng Gao. All Rights Reserved.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReIDCriterion(nn.Module):
    def __init__(
            self,
            weight: float,
            temperature: float = 0.07,
            normalize: bool = True,
            loss_type: str = "supcon",
    ):
        super().__init__()
        self.weight = weight
        self.temperature = temperature
        self.normalize = normalize
        self.loss_type = loss_type

    def _supcon_loss(self, emb: torch.Tensor, ids: torch.Tensor) -> Optional[torch.Tensor]:
        if emb.shape[0] <= 1:
            return None
        device = emb.device
        ids = ids.to(device)
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)

        sim = (emb @ emb.T) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True).values
        m = sim.shape[0]
        self_mask = torch.eye(m, dtype=torch.bool, device=device)

        exp_sim = torch.exp(sim)
        exp_sim = exp_sim.masked_fill(self_mask, 0.0)

        pos_mask = (ids[None, :] == ids[:, None]) & ~self_mask
        numerator = (exp_sim * pos_mask).sum(dim=1)
        denom = exp_sim.sum(dim=1)

        valid = numerator > 0
        if valid.sum().item() == 0:
            return None
        loss = -torch.log(numerator[valid] / denom[valid])
        return loss.mean()

    def forward(
            self,
            detr_outputs: dict,
            annotations: list,
            detr_indices: list,
    ) -> dict:
        if self.loss_type != "supcon":
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
        reid_emb = detr_outputs["reid_emb"]
        device = reid_emb.device

        batch_size = len(annotations)
        seq_len = len(annotations[0]) if batch_size > 0 else 0
        clip_losses = []

        for b in range(batch_size):
            clip_embs = []
            clip_ids = []
            for t in range(seq_len):
                ann = annotations[b][t]
                if "id" in ann:
                    gt_ids = ann["id"]
                elif "ids" in ann:
                    gt_ids = ann["ids"]
                else:
                    continue
                if gt_ids.numel() == 0:
                    continue

                flatten_idx = b * seq_len + t
                src_idx, tgt_idx = detr_indices[flatten_idx]
                if tgt_idx.numel() == 0:
                    continue
                go_back_detr_idxs = torch.argsort(tgt_idx)
                matched_src = src_idx[go_back_detr_idxs]
                frame_reid = reid_emb[flatten_idx][matched_src]

                clip_embs.append(frame_reid)
                clip_ids.append(gt_ids.to(device))

            if not clip_embs:
                continue
            clip_emb = torch.cat(clip_embs, dim=0)
            clip_ids = torch.cat(clip_ids, dim=0)

            clip_loss = self._supcon_loss(clip_emb, clip_ids)
            if clip_loss is not None:
                clip_losses.append(clip_loss)

        if not clip_losses:
            zero = reid_emb.new_tensor(0.0)
            return {"loss_reid": zero}

        batch_loss = torch.stack(clip_losses).mean()
        return {"loss_reid": batch_loss * self.weight}


def build_reid_criterion(config: dict) -> ReIDCriterion:
    weight = config.get("REID_WEIGHT", 1.0)
    temperature = config.get("REID_TEMP", 0.07)
    normalize = config.get("REID_NORMALIZE", True)
    loss_type = config.get("REID_LOSS_TYPE", "supcon")
    return ReIDCriterion(
        weight=weight,
        temperature=temperature,
        normalize=normalize,
        loss_type=loss_type,
    )
