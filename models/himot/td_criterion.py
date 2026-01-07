# Copyright (c) Ruopeng Gao. All Rights Reserved.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TDCriterion(nn.Module):
    def __init__(
            self,
            w_reid: float,
            w_box: float,
            reid_loss_type: str = "cosine",
            box_loss_type: str = "smooth_l1",
            box_beta: float = 1.0,
            reduction: str = "mean",
    ):
        super().__init__()
        self.w_reid = w_reid
        self.w_box = w_box
        self.reid_loss_type = reid_loss_type
        self.box_loss_type = box_loss_type
        self.box_beta = box_beta
        self.reduction = reduction

    def forward(
            self,
            pred_reid: torch.Tensor,
            pred_delta: torch.Tensor,
            tgt_reid: torch.Tensor,
            tgt_delta: torch.Tensor,
            valid_targets_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        if valid_targets_mask is not None:
            valid_targets_mask = valid_targets_mask.to(torch.bool)
            if valid_targets_mask.sum().item() == 0:
                zero = pred_reid.new_tensor(0.0)
                return {
                    "loss_td_reid": zero,
                    "loss_td_box": zero,
                    "loss_td_total": zero,
                }
            pred_reid = pred_reid[valid_targets_mask]
            pred_delta = pred_delta[valid_targets_mask]
            tgt_reid = tgt_reid[valid_targets_mask]
            tgt_delta = tgt_delta[valid_targets_mask]

        if self.reid_loss_type != "cosine":
            raise ValueError(f"Unsupported reid_loss_type: {self.reid_loss_type}")
        cosine_sim = F.cosine_similarity(pred_reid, tgt_reid, dim=-1)
        reid_loss = 1.0 - cosine_sim
        if self.reduction == "mean":
            reid_loss = reid_loss.mean()
        elif self.reduction == "sum":
            reid_loss = reid_loss.sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

        if self.box_loss_type == "smooth_l1":
            box_loss = F.smooth_l1_loss(pred_delta, tgt_delta, beta=self.box_beta, reduction=self.reduction)
        elif self.box_loss_type == "l1":
            box_loss = F.l1_loss(pred_delta, tgt_delta, reduction=self.reduction)
        else:
            raise ValueError(f"Unsupported box_loss_type: {self.box_loss_type}")

        loss_td_reid = self.w_reid * reid_loss
        loss_td_box = self.w_box * box_loss
        loss_td_total = loss_td_reid + loss_td_box

        return {
            "loss_td_reid": loss_td_reid,
            "loss_td_box": loss_td_box,
            "loss_td_total": loss_td_total,
        }


def build_td_criterion(config: dict) -> TDCriterion:
    w_reid = config.get("TD_LOSS_REID_WEIGHT", 1.0)
    w_box = config.get("TD_LOSS_BOX_WEIGHT", 1.0)
    reid_loss_type = config.get("TD_REID_LOSS_TYPE", "cosine")
    box_loss_type = config.get("TD_BOX_LOSS_TYPE", "smooth_l1")
    box_beta = config.get("TD_BOX_LOSS_BETA", 1.0)
    reduction = config.get("TD_LOSS_REDUCTION", "mean")
    return TDCriterion(
        w_reid=w_reid,
        w_box=w_box,
        reid_loss_type=reid_loss_type,
        box_loss_type=box_loss_type,
        box_beta=box_beta,
        reduction=reduction,
    )
