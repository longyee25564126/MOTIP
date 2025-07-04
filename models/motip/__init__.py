# Copyright (c) Ruopeng Gao. All Rights Reserved.

from .motip import MOTIP
from structures.args import Args
from models.deformable_detr.deformable_detr import build as build_deformable_detr
from models.motip.trajectory_modeling import TrajectoryModeling
from models.motip.id_decoder import IDDecoder


def build(config: dict):
    # 準備 DETR 所需的參數（轉成 Args）通通填入 detr_args 這個設定物件中。這些參數會用於 DETR backbone、transformer 結構、損失函數等:
    detr_args = Args()
    # 1. backbone:
    detr_args.backbone = config["BACKBONE"]
    detr_args.lr_backbone = config["LR"] * config["LR_BACKBONE_SCALE"]
    detr_args.dilation = config["DILATION"]
    # 2. transformer:
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

    detr_framework = config["DETR_FRAMEWORK"].lower()
    #根據 DETR_FRAMEWORK 選擇哪個 DETR 模型，根據你在 config 中指定的 DETR 架構（例如 "deformable_detr"）來呼叫對應的 build_deformable_detr()
    match detr_framework:
        case "deformable_detr":
            detr, detr_criterion, _ = build_deformable_detr(args=detr_args)
        case _:
            raise NotImplementedError(f"DETR framework {config['DETR_FRAMEWORK']} is not supported.")

    # Build each component:
    # 1. 建立軌跡建模模組（trajectory_modeling），如果不是只訓練 DETR，會建立這個模組，常見是用 FFN（前饋網路）來對時序特徵做聚合:
    _trajectory_modeling = TrajectoryModeling(
        detr_dim=config["DETR_HIDDEN_DIM"],
        ffn_dim_ratio=config["FFN_DIM_RATIO"],
        feature_dim=config["FEATURE_DIM"],
    ) if config["ONLY_DETR"] is False else None
    # 2. 建立 ID 解碼器（id_decoder）:
    _id_decoder = IDDecoder(
        feature_dim=config["FEATURE_DIM"],
        id_dim=config["ID_DIM"],
        ffn_dim_ratio=config["FFN_DIM_RATIO"],
        num_layers=config["NUM_ID_DECODER_LAYERS"],
        head_dim=config["HEAD_DIM"],
        num_id_vocabulary=config["NUM_ID_VOCABULARY"],
        rel_pe_length=config["REL_PE_LENGTH"],
        use_aux_loss=config["USE_AUX_LOSS"],
        use_shared_aux_head=config["USE_SHARED_AUX_HEAD"],
    ) if config["ONLY_DETR"] is False else None

    # 組裝 MOTIP 模型:
    motip_model = MOTIP(
        detr=detr,
        detr_framework=detr_framework,
        only_detr=config["ONLY_DETR"],
        trajectory_modeling=_trajectory_modeling,
        id_decoder=_id_decoder,
    )
    # 回傳模型與損失函數
    return motip_model, detr_criterion
