# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import einops
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import is_distributed, distributed_world_size, labels_to_one_hot

#負責 Identity Loss（ID Loss）計算 的模組 IDCriterion
class IDCriterion(nn.Module):
    def __init__(
            self,
            weight: float,
            use_focal_loss: bool,
    ):
       
        super().__init__()
        self.weight = weight
        self.use_focal_loss = use_focal_loss
        #決定是要用普通的 CrossEntropy 還是 Focal Loss
        if not self.use_focal_loss:
            self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        return

    def forward(self, id_logits, id_labels, id_masks):
        # _B, _G, _T, _N = id_logits.shape
        # 忽略第 0 幀，第一幀無法從軌跡預測得到 ID，因此不列入 loss 計算:
        id_logits = id_logits[:, :, 1:, :, :]
        id_labels = id_labels[:, :, 1:, :]
        id_masks = id_masks[:, :, 1:, :]
        pass

        # flatten 成 2D 形狀方便計算 loss:
        id_logits_flatten = einops.rearrange(id_logits, "b g t n c -> (b g t n) c")
        id_labels_flatten = einops.rearrange(id_labels, "b g t n -> (b g t n)")
        id_masks_flatten = einops.rearrange(id_masks, "b g t n -> (b g t n)")
        # 根據 mask 過濾無效目標，id_masks=True 表示是 padding 或空格 → 要剔除:
        id_logits_flatten = id_logits_flatten[~id_masks_flatten]
        id_labels_flatten = id_labels_flatten[~id_masks_flatten]
        # 計算損失:
        # 若使用 focal loss：將 label 轉為 one-hot，再代入自定義的 sigmoid_focal_loss() 計算
        if self.use_focal_loss:
            id_labels_flatten_one_hot = labels_to_one_hot(id_labels_flatten, class_num=id_logits_flatten.shape[-1])
            id_labels_flatten_one_hot = torch.tensor(id_labels_flatten_one_hot, device=id_logits.device)
            loss = sigmoid_focal_loss(inputs=id_logits_flatten, targets=id_labels_flatten_one_hot).sum()
        #若使用 CrossEntropy，直接輸入類別編號
        else:
            loss = self.ce_loss(id_logits_flatten, id_labels_flatten).sum()
        # 平均化 loss（可分散式訓練時也可對齊）
        num_ids = torch.as_tensor([len(id_logits_flatten)], dtype=torch.float, device=id_logits.device)
        # 確保在多 GPU 訓練時，loss 不會被重複計算或 scale 錯誤
        if is_distributed():
            torch.distributed.all_reduce(num_ids)
        num_ids = torch.clamp(num_ids / distributed_world_size(), min=1).item()

        return loss / num_ids


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum()


def build(config: dict):
    return IDCriterion(
        weight=config["ID_LOSS_WEIGHT"],
        use_focal_loss=config["USE_FOCAL_LOSS"],
    )