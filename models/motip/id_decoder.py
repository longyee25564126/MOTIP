# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import einops
import torch.nn as nn
from typing import Tuple
from torch.utils.checkpoint import checkpoint

from models.misc import _get_clones, label_to_one_hot
from models.ffn import FFN


class IDDecoder(nn.Module):
    def __init__(
            self,
            feature_dim: int,           # DETR 輸出的特徵維度
            id_dim: int,                # ID embedding 維度
            ffn_dim_ratio: int,         # FFN 的維度比率，實際維度為 (feature_dim + id_dim) * ffn_dim_ratio
            num_layers: int,            # Transformer 的層數
            head_dim: int,              # Multi-head Attention 的 head 維度
            num_id_vocabulary: int,     # ID label 的詞彙表大小，包含一個額外的「未知」ID，即論文中的K
            rel_pe_length: int,         # 相對位置嵌入的長度，表示最多考慮多少個時間步的相對位置
            use_aux_loss: bool,         # 是否使用輔助損失來訓練 ID 頭
            use_shared_aux_head: bool,  # 是否使用共享的輔助頭來計算 ID 頭的輔助損失
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.id_dim = id_dim
        self.ffn_dim_ratio = ffn_dim_ratio
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.n_heads = (self.feature_dim + self.id_dim) // self.head_dim
        self.num_id_vocabulary = num_id_vocabulary
        self.rel_pe_length = rel_pe_length

        self.use_aux_loss = use_aux_loss
        self.use_shared_aux_head = use_shared_aux_head

        # 將 One-hot ID label 映射到一個 id_dim 維的向量，形成所謂的「ID embedding」就是id_learnable dictionary。
        self.word_to_embed = nn.Linear(self.num_id_vocabulary + 1, self.id_dim, bias=False)
        embed_to_word = nn.Linear(self.id_dim, self.num_id_vocabulary + 1, bias=False)

        if self.use_aux_loss and not self.use_shared_aux_head:
            self.embed_to_word_layers = _get_clones(embed_to_word, self.num_layers)
        else:
            self.embed_to_word_layers = nn.ModuleList([embed_to_word for _ in range(self.num_layers)])
        pass

        # 相對位置嵌入 (Relative Positional Embedding):
        self.rel_pos_embeds = nn.Parameter(
            torch.zeros((self.num_layers, self.rel_pe_length, self.n_heads), dtype=torch.float32)
        )
        # Prepare others for rel pe 建立一個「相對時間差」對應表:
        t_idxs = torch.arange(self.rel_pe_length, dtype=torch.int64) # 建立一個從 0 到 rel_pe_length - 1 的時間索引序列
        curr_t_idxs, traj_t_idxs = torch.meshgrid([t_idxs, t_idxs])  # 建立一個二維網格，表示當前時間索引和軌跡時間索引的所有可能組合
        self.rel_pos_map = (curr_t_idxs - traj_t_idxs)      # [curr_t_idx, traj_t_idx] -> rel_pos, like [1, 0] = 1 這一步產生的是一張「相對位置差值圖」
        # 此時即可透過 self.rel_pos_map[a, b] 查出「query 時間 = a、key 時間 = b」時的相對時間差
        pass

        # 初始化 Self-Attention 層
        self_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim + self.id_dim,   # Attention 的輸入維度(即論文中的2C) 
            num_heads=self.n_heads,                     # 多頭注意力中的 head 數目
            dropout=0.0,                                # 注意力機制中的 dropout 機率 
            batch_first=True,                           # 設置為 True 以便輸入和輸出張量的形狀為 (batch_size, seq_len, embed_dim)
            add_zero_attn=True,                         # 在 attention 中額外加入一個全 0 向量，允許模型選擇「不關注任何一個輸入」
        )
        self_attn_norm = nn.LayerNorm(self.feature_dim + self.id_dim) # 對 Self-Attention 的輸出進行層歸一化
        # 初始化 Cross-Attention 層
        cross_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim + self.id_dim,   
            num_heads=self.n_heads,                     
            dropout=0.0,
            batch_first=True,
            add_zero_attn=True,
        )
        cross_attn_norm = nn.LayerNorm(self.feature_dim + self.id_dim)
        # 初始化 Feed-Forward Network (FFN) 層
        ffn = FFN(
            d_model=self.feature_dim + self.id_dim,
            d_ffn=(self.feature_dim + self.id_dim) * self.ffn_dim_ratio,
            activation=nn.GELU(),
        )
        ffn_norm = nn.LayerNorm(self.feature_dim + self.id_dim)

        self.self_attn_layers = _get_clones(self_attn, self.num_layers - 1)
        self.self_attn_norm_layers = _get_clones(self_attn_norm, self.num_layers - 1)
        self.cross_attn_layers = _get_clones(cross_attn, self.num_layers)
        self.cross_attn_norm_layers = _get_clones(cross_attn_norm, self.num_layers)
        self.ffn_layers = _get_clones(ffn, self.num_layers)
        self.ffn_norm_layers = _get_clones(ffn_norm, self.num_layers)

        # 對 IDDecoder 模組中「除了相對位置嵌入以外的所有參數」，進行 Xavier 初始化:
        for n, p in self.named_parameters():
        '''
        遍歷整個 IDDecoder 裡所有 nn.Parameter
            n: 該參數的名稱字串
            p: 該參數的實際權重 Tensor（如 weight: torch.Size([256, 256])）
        '''
            # 只針對「矩陣類型參數」
            if p.dim() > 1 and "rel_pos_embeds" not in n:
                nn.init.xavier_uniform_(p) # Xavier initialization，又叫 Glorot uniform，讓輸入與輸出變異數一致，有利於訓練初期的穩定性

        pass

    def forward(self, seq_info, use_decoder_checkpoint):
        # 產生兩組嵌入（embedding）
        trajectory_features = seq_info["trajectory_features"]   
        unknown_features = seq_info["unknown_features"]
        trajectory_id_labels = seq_info["trajectory_id_labels"]
        unknown_id_labels = seq_info["unknown_id_labels"] if "unknown_id_labels" in seq_info else None
        trajectory_times = seq_info["trajectory_times"]
        unknown_times = seq_info["unknown_times"]
        trajectory_masks = seq_info["trajectory_masks"]
        unknown_masks = seq_info["unknown_masks"]
        _B, _G, _T, _N, _ = trajectory_features.shape
        _curr_B, _curr_G, _curr_T, _curr_N, _ = unknown_features.shape

        # 把每個過去物件的 ID 編號轉成向量，作為「身分提示」
        trajectory_id_embeds = self.id_label_to_embed(id_labels=trajectory_id_labels)
        # 先放一個 “newborn” ID embedding（即論文中的 i_spec）
        unknown_id_embeds = self.generate_empty_id_embed(unknown_features=unknown_features)

        # trajectory_embeds: 歷史軌跡資料 + 對應的 ID 嵌入（有標籤）
        trajectory_embeds = torch.cat([trajectory_features, trajectory_id_embeds], dim=-1)
        # unknown_embeds: 當前畫面偵測物件 + 空白 ID 嵌入（準備預測）
        unknown_embeds = torch.cat([unknown_features, unknown_id_embeds], dim=-1)

        # Prepare some common variables:
        # 遮住一些「無效軌跡」（padding 的，告訴 Transformer：這些 token 是假的，請不要計算它們的 attention，也不要預測它們的 ID。
	    # 禁止模型偷看「未來幀」（你不能預測 ID 的時候還去看答案本身）

        self_attn_key_padding_mask = einops.rearrange(unknown_masks, "b g t n -> (b g t) n").contiguous()
        cross_attn_key_padding_mask = einops.rearrange(trajectory_masks, "b g t n -> (b g) (t n)").contiguous()

        # 把每個時間幀裡的所有 N 個物件攤平成單一維度 → 總共 T * N 個軌跡時間
        _trajectory_times_flatten = einops.rearrange(trajectory_times, "b g t n -> (b g) (t n)")
        # 一樣道理，只是對象換成 unknown detections
        _unknown_times_flatten = einops.rearrange(unknown_times, "b g t n -> (b g) (t n)")
        # 看誰在未來 → 遮掉
        cross_attn_mask = _trajectory_times_flatten[:, None, :] >= _unknown_times_flatten[:, :, None]
        cross_attn_mask = einops.repeat(cross_attn_mask, "bg tn1 tn2 -> (bg n_heads) tn1 tn2", n_heads=self.n_heads).contiguous()
        # Prepare for rel PE :
        # 先前初始化好的 rel_pos_map（一張查表地圖）搬到 GPU 上
        self.rel_pos_map = self.rel_pos_map.to(trajectory_features.device)
        # 建構所有 query-key 配對時間對
        rel_pe_idx_pairs = torch.stack([
            torch.stack(
                torch.meshgrid([_unknown_times_flatten[_], _trajectory_times_flatten[_]]), dim=-1
            )
            for _ in range(len(_trajectory_times_flatten))
        ], dim=0)       # (B*G, T*N of curr, T*N of traj, 2)
        rel_pe_idx_pairs = rel_pe_idx_pairs.to(trajectory_features.device)
        rel_pe_idxs = self.rel_pos_map[rel_pe_idx_pairs[..., 0], rel_pe_idx_pairs[..., 1]]      # (B*G, T_curr, T_traj)
        pass
        # Change Cross-Attn key_padding_mask and attn_mask to float:
        cross_attn_key_padding_mask = torch.masked_fill(
            cross_attn_key_padding_mask.float(),
            mask=cross_attn_key_padding_mask,
            value=float("-inf"),
        ).to(self.dtype)
        cross_attn_mask = torch.masked_fill(
            cross_attn_mask.float(),
            mask=cross_attn_mask,
            value=float("-inf"),
        ).to(self.dtype)
        pass

        all_unknown_id_logits = None
        all_unknown_id_labels = None
        all_unknown_id_masks = None

        for layer in range(self.num_layers):
            # Predict ID logits:
            if use_decoder_checkpoint:
                unknown_embeds = checkpoint(
                    self._forward_a_layer,
                    layer,
                    unknown_embeds, trajectory_embeds,
                    self_attn_key_padding_mask, cross_attn_key_padding_mask,
                    cross_attn_mask, rel_pe_idxs,
                    use_reentrant=False,
                )
            else:
                unknown_embeds = self._forward_a_layer(
                    layer=layer,
                    unknown_embeds=unknown_embeds,
                    trajectory_embeds=trajectory_embeds,
                    self_attn_key_padding_mask=self_attn_key_padding_mask,
                    cross_attn_key_padding_mask=cross_attn_key_padding_mask,
                    cross_attn_mask=cross_attn_mask,
                    rel_pe_idx=rel_pe_idxs,
                )
            # 把 decoder 第 layer 層輸出的 ID embedding 轉成 每個物件對所有 ID 的預測分數（logits）
            _unknown_id_logits = self.embed_to_word_layers[layer](unknown_embeds[..., -self.id_dim:])
            # 儲存對應的 mask
            _unknown_id_masks = unknown_masks.clone()
            # 訓練時保留 unknown_id_labels 作為真實答案
            _unknown_id_labels = None if not self.training else unknown_id_labels
            # 當還沒蒐集任何 logits 時，初始化：
            if all_unknown_id_logits is None:
                all_unknown_id_logits = _unknown_id_logits
                all_unknown_id_labels = _unknown_id_labels
                all_unknown_id_masks = _unknown_id_masks
            else:
                # 接下來的每一層，把每一層的 logits / labels / masks 都 torch.cat() 起來
                all_unknown_id_logits = torch.cat([all_unknown_id_logits, _unknown_id_logits], dim=0)
                all_unknown_id_labels = torch.cat([all_unknown_id_labels, _unknown_id_labels], dim=0) if _unknown_id_labels is not None else None
                all_unknown_id_masks = torch.cat([all_unknown_id_masks, _unknown_id_masks], dim=0)

        if self.training and self.use_aux_loss:
            return all_unknown_id_logits, all_unknown_id_labels, all_unknown_id_masks
        else:
            return _unknown_id_logits, _unknown_id_labels, _unknown_id_masks


    def _forward_a_layer(
            self,
            layer: int,                                 # 第幾層 decoder（從 0 開始）
            unknown_embeds: torch.Tensor,               # 當前幀中待預測物件的 feature + 空 ID embedding（shape: B, G, T, N, C）
            trajectory_embeds: torch.Tensor,            # 歷史軌跡物件的 feature + 已知 ID embedding
            self_attn_key_padding_mask: torch.Tensor,   # self-attn 遮罩（只針對當前物件）
            cross_attn_key_padding_mask: torch.Tensor,  # cross-attn 遮罩（針對歷史軌跡中的 padding）
            cross_attn_mask: torch.Tensor,              # 禁止注意未來的遮罩 + rel PE bias
            rel_pe_idx: torch.Tensor,                   # 每個 (query, key) 對的時間差索引，用來查表
    ):
        _B, _G, _T, _N, _ = trajectory_embeds.shape
        _curr_B, _curr_G, _curr_T, _curr_N, _ = unknown_embeds.shape
        # 只有在第 2 層（layer > 0）以後才會做 self-attention
        # 原因：第一層的輸入還未經歷任何上下文交互，直接進行 cross-attention
        if layer > 0:   # use self-attention to transfer information between unknown features (same time step)
            self_unknown_embeds = einops.rearrange(unknown_embeds, "b g t n c -> (b g t) n c").contiguous()
            self_out, _ = self.self_attn_layers[layer - 1](
                query=self_unknown_embeds, key=self_unknown_embeds, value=self_unknown_embeds,
                key_padding_mask=self_attn_key_padding_mask,
            )
            self_out = self_unknown_embeds + self_out
            self_out = self.self_attn_norm_layers[layer - 1](self_out)
            unknown_embeds = einops.rearrange(self_out, "(b g t) n c -> b g t n c", b=_B, g=_G, t=_curr_T)
            
        # 讓當前的 detection 向歷史軌跡的物件「查詢」資訊（跨時間與 ID）
        # Cross-attention for in-context decoding:
        cross_unknown_embeds = einops.rearrange(unknown_embeds, "b g t n c -> (b g) (t n) c").contiguous()
        cross_trajectory_embeds = einops.rearrange(trajectory_embeds, "b g t n c -> (b g) (t n) c").contiguous()
        # Prepare attn_mask:
        rel_pe_mask = self.rel_pos_embeds[layer][rel_pe_idx]
        cross_attn_mask_with_rel_pe = cross_attn_mask + einops.rearrange(rel_pe_mask, "bg l1 l2 n -> (bg n) l1 l2")
        # Apply cross-attention:
        cross_out, _ = self.cross_attn_layers[layer](
            query=cross_unknown_embeds, key=cross_trajectory_embeds, value=cross_trajectory_embeds,
            key_padding_mask=cross_attn_key_padding_mask,
            attn_mask=cross_attn_mask_with_rel_pe,
        )
        cross_out = cross_unknown_embeds + cross_out
        cross_out = self.cross_attn_norm_layers[layer](cross_out)
        # Feed-forward network:
        cross_out = cross_out + self.ffn_layers[layer](cross_out)
        cross_out = self.ffn_norm_layers[layer](cross_out)
        # Re-shape back to original shape:
        unknown_embeds = einops.rearrange(cross_out, "(b g) (t n) c -> b g t n c", b=_B, g=_G, t=_curr_T)

        return unknown_embeds

    def id_label_to_embed(self, id_labels):
        id_words = label_to_one_hot(id_labels, self.num_id_vocabulary + 1, dtype=self.dtype)
        id_embeds = self.word_to_embed(id_words)
        return id_embeds

    def generate_empty_id_embed(self, unknown_features):
        _shape = unknown_features.shape[:-1]
        empty_id_labels = self.num_id_vocabulary * torch.ones(_shape, dtype=torch.int64, device=unknown_features.device)
        empty_id_embeds = self.id_label_to_embed(id_labels=empty_id_labels)
        return empty_id_embeds

    def shuffle(self):
        shuffle_index = torch.randperm(self.num_id_vocabulary, device=self.word_to_embed.weight.device)
        shuffle_index = torch.cat([shuffle_index, torch.tensor([self.num_id_vocabulary], device=self.word_to_embed.weight.device)])
        self.word_to_embed.weight.data = self.word_to_embed.weight.data[:, shuffle_index]
        self.embed_to_word.weight.data = self.embed_to_word.weight.data[shuffle_index, :]
        pass

    @property
    def dtype(self):
        return self.word_to_embed.weight.dtype