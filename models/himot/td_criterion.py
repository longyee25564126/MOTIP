# Copyright (c) Ruopeng Gao. All Rights Reserved.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class TDCriterion(nn.Module):
    def __init__(
            self,
            emb_cos_weight: float,
            emb_mse_weight: float,
            emb_loss_type: str = "cosine",
            box_l1_weight: float = 1.0,
            box_giou_weight: float = 2.0,
            reduction: str = "mean",
    ):
        super().__init__()
        self.emb_cos_weight = emb_cos_weight
        self.emb_mse_weight = emb_mse_weight
        self.emb_loss_type = emb_loss_type
        self.box_l1_weight = box_l1_weight
        self.box_giou_weight = box_giou_weight
        self.reduction = reduction

    def forward(
            self,
            pred_emb: torch.Tensor,
            pred_box: torch.Tensor,
            tgt_emb: torch.Tensor,
            tgt_box: torch.Tensor,
            valid_targets_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        if valid_targets_mask is not None:
            valid_targets_mask = valid_targets_mask.to(torch.bool)
            if valid_targets_mask.sum().item() == 0:
                zero = pred_emb.new_tensor(0.0)
                return {
                    "loss_td_emb": zero,
                    "loss_td_emb_cos": zero,
                    "loss_td_emb_mse": zero,
                    "loss_td_box": zero,
                    "loss_td_total": zero,
                }
            pred_emb = pred_emb[valid_targets_mask]
            pred_box = pred_box[valid_targets_mask]
            tgt_emb = tgt_emb[valid_targets_mask]
            tgt_box = tgt_box[valid_targets_mask]

        if self.emb_loss_type != "cosine":
            raise ValueError(f"Unsupported emb_loss_type: {self.emb_loss_type}")
        norm_pred = F.normalize(pred_emb, p=2, dim=-1)
        norm_tgt = F.normalize(tgt_emb, p=2, dim=-1)
        cosine_sim = F.cosine_similarity(pred_emb, tgt_emb, dim=-1)
        loss_cos = 1.0 - cosine_sim
        loss_mse = F.mse_loss(norm_pred, norm_tgt, reduction="none").mean(dim=-1)
        if self.reduction == "mean":
            loss_cos = loss_cos.mean()
            loss_mse = loss_mse.mean()
        elif self.reduction == "sum":
            loss_cos = loss_cos.sum()
            loss_mse = loss_mse.sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
        emb_loss = self.emb_cos_weight * loss_cos + self.emb_mse_weight * loss_mse

        box_l1 = F.l1_loss(pred_box, tgt_box, reduction="none").sum(dim=-1)
        pred_xyxy = box_cxcywh_to_xyxy(pred_box)
        tgt_xyxy = box_cxcywh_to_xyxy(tgt_box)
        giou = generalized_box_iou(pred_xyxy, tgt_xyxy)
        box_giou = 1.0 - torch.diag(giou)

        if self.reduction == "mean":
            box_l1 = box_l1.mean()
            box_giou = box_giou.mean()
        elif self.reduction == "sum":
            box_l1 = box_l1.sum()
            box_giou = box_giou.sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

        loss_td_emb = emb_loss
        loss_td_box = self.box_l1_weight * box_l1 + self.box_giou_weight * box_giou
        loss_td_total = loss_td_emb + loss_td_box

        return {
            "loss_td_emb": loss_td_emb,
            "loss_td_emb_cos": loss_cos,
            "loss_td_emb_mse": loss_mse,
            "loss_td_box_l1": box_l1,
            "loss_td_box_giou": box_giou,
            "loss_td_box": loss_td_box,
            "loss_td_total": loss_td_total,
        }


def build_td_criterion(config: dict) -> TDCriterion:
    emb_cos_weight = config.get("TD_EMB_COS_WEIGHT", 1.0)
    emb_mse_weight = config.get("TD_EMB_MSE_WEIGHT", 0.1)
    emb_loss_type = config.get("TD_EMB_LOSS_TYPE", "cosine")
    box_l1_weight = config.get("TD_BOX_L1_WEIGHT", 1.0)
    box_giou_weight = config.get("TD_BOX_GIOU_WEIGHT", 2.0)
    reduction = config.get("TD_LOSS_REDUCTION", "mean")
    return TDCriterion(
        emb_cos_weight=emb_cos_weight,
        emb_mse_weight=emb_mse_weight,
        emb_loss_type=emb_loss_type,
        box_l1_weight=box_l1_weight,
        box_giou_weight=box_giou_weight,
        reduction=reduction,
    )
