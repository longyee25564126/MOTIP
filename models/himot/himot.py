# Copyright (c) Ruopeng Gao. All Rights Reserved.

from __future__ import annotations

from typing import Dict, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from structures.args import Args
from models.deformable_detr.deformable_detr import build as build_deformable_detr
from models.himot.traj_token_encoder import build_trajectory_token_encoder
from models.himot.traj_decoder import build_trajectory_decoder


class HiMOT(nn.Module):
    def __init__(
            self,
            detr: nn.Module,
            detr_criterion: nn.Module,
            token_encoder: nn.Module,
            traj_decoder: nn.Module,
    ):
        super().__init__()
        self.detr = detr
        self.detr_criterion = detr_criterion
        self.token_encoder = token_encoder
        self.traj_decoder = traj_decoder

    def forward(self, **kwargs):
        assert "part" in kwargs, "Parameter `part` is required for HiMOT forward."
        match kwargs["part"]:
            case "detr":
                frames = kwargs["frames"]
                if "use_checkpoint" in kwargs:
                    return checkpoint(self.detr, frames, use_reentrant=False)
                return self.detr(samples=frames)
            case "traj_decoder":
                seq_info = kwargs["seq_info"]
                tokens, valid_mask = self.token_encoder(
                    reid_emb=seq_info["reid_seq"],
                    delta_box=seq_info["delta_seq"],
                    pad_mask=seq_info["pad_mask"],
                    miss_mask=seq_info["miss_mask"],
                )
                cache = kwargs.get("cache", None)
                return_cache = kwargs.get("return_cache", False)
                if return_cache:
                    pred_reid, pred_delta, new_cache = self.traj_decoder(
                        tokens=tokens,
                        valid_mask=valid_mask,
                        cache=cache,
                        return_cache=True,
                    )
                    return {
                        "pred_reid": pred_reid,
                        "pred_delta": pred_delta,
                        "valid_mask": valid_mask,
                        "cache": new_cache,
                    }
                else:
                    pred_reid, pred_delta = self.traj_decoder(
                        tokens=tokens,
                        valid_mask=valid_mask,
                        cache=cache,
                        return_cache=False,
                    )
                    return {
                        "pred_reid": pred_reid,
                        "pred_delta": pred_delta,
                        "valid_mask": valid_mask,
                    }
            case _:
                raise NotImplementedError(f"HiMOT forwarding doesn't support part={kwargs['part']}.")


def build(config: dict) -> Tuple[HiMOT, nn.Module]:
    detr_args = Args()
    detr_args.backbone = config["BACKBONE"]
    detr_args.lr_backbone = config["LR"] * config["LR_BACKBONE_SCALE"]
    detr_args.dilation = config["DILATION"]
    detr_args.num_classes = config["NUM_CLASSES"]
    detr_args.device = config["DEVICE"]
    detr_args.num_queries = config["DETR_NUM_QUERIES"]
    detr_args.num_feature_levels = config["DETR_NUM_FEATURE_LEVELS"]
    detr_args.aux_loss = config["DETR_AUX_LOSS"]
    detr_args.with_box_refine = config["DETR_WITH_BOX_REFINE"]
    detr_args.two_stage = config["DETR_TWO_STAGE"]
    detr_args.hidden_dim = config["DETR_HIDDEN_DIM"]
    detr_args.masks = config["DETR_MASKS"]
    detr_args.position_embedding = config["DETR_POSITION_EMBEDDING"]
    detr_args.nheads = config["DETR_NUM_HEADS"]
    detr_args.enc_layers = config["DETR_ENC_LAYERS"]
    detr_args.dec_layers = config["DETR_DEC_LAYERS"]
    detr_args.dim_feedforward = config["DETR_DIM_FEEDFORWARD"]
    detr_args.dropout = config["DETR_DROPOUT"]
    detr_args.dec_n_points = config["DETR_DEC_N_POINTS"]
    detr_args.enc_n_points = config["DETR_ENC_N_POINTS"]
    detr_args.cls_loss_coef = config["DETR_CLS_LOSS_COEF"]
    detr_args.bbox_loss_coef = config["DETR_BBOX_LOSS_COEF"]
    detr_args.giou_loss_coef = config["DETR_GIOU_LOSS_COEF"]
    detr_args.focal_alpha = config["DETR_FOCAL_ALPHA"]
    detr_args.set_cost_class = config["DETR_SET_COST_CLASS"]
    detr_args.set_cost_bbox = config["DETR_SET_COST_BBOX"]
    detr_args.set_cost_giou = config["DETR_SET_COST_GIOU"]
    detr_args.reid_backprop_mode = config.get("REID_BACKPROP_MODE", "head_only")

    detr, detr_criterion, _ = build_deformable_detr(args=detr_args)

    token_encoder = build_trajectory_token_encoder(
        reid_dim=config.get("REID_DIM", config["DETR_HIDDEN_DIM"]),
        d_model=config.get("TD_DMODEL", config.get("TD_D_MODEL", config["DETR_HIDDEN_DIM"])),
        hidden_dim=config.get("TD_ENCODER_HIDDEN_DIM", None),
        dropout=config.get("TD_ENCODER_DROPOUT", 0.0),
        use_layer_norm=config.get("TD_ENCODER_USE_LN", True),
        num_layers=config.get("TD_ENCODER_NUM_LAYERS", 2),
    )
    traj_decoder = build_trajectory_decoder(
        d_model=config.get("TD_DMODEL", config.get("TD_D_MODEL", config["DETR_HIDDEN_DIM"])),
        nhead=config.get("TD_NHEAD", config["DETR_NUM_HEADS"]),
        num_layers=config.get("TD_NUM_LAYERS", 6),
        dim_feedforward=config.get("TD_DIM_FEEDFORWARD", None),
        dropout=config.get("TD_DROPOUT", 0.0),
        rope_base=config.get("TD_ROPE_BASE", 10000),
        max_seq_len=config.get("TD_MAX_SEQ_LEN", 64),
        reid_dim=config.get("REID_DIM", config["DETR_HIDDEN_DIM"]),
        l2norm_reid=config.get("TD_L2NORM_REID", True),
    )

    model = HiMOT(
        detr=detr,
        detr_criterion=detr_criterion,
        token_encoder=token_encoder,
        traj_decoder=traj_decoder,
    )
    return model, detr_criterion


def build_himot(config: dict) -> Tuple[HiMOT, nn.Module]:
    return build(config=config)
