# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from utils.misc import distributed_device
from utils.box_ops import box_cxcywh_to_xywh


@dataclass
class TrackState:
    track_id: int
    last_box: Optional[torch.Tensor]
    hist_emb: List[Optional[torch.Tensor]]
    hist_box: List[Optional[torch.Tensor]]
    hist_is_obs: List[bool]
    miss_count: int

    def append_obs(self, emb: torch.Tensor, box: torch.Tensor, max_len: int):
        self.hist_emb.append(emb)
        self.hist_box.append(box)
        self.hist_is_obs.append(True)
        self.last_box = box
        self.miss_count = 0
        self._trim(max_len)

    def append_miss(self, max_len: int):
        self.hist_emb.append(None)
        self.hist_box.append(None)
        self.hist_is_obs.append(False)
        self.miss_count += 1
        self._trim(max_len)

    def _trim(self, max_len: int):
        if len(self.hist_is_obs) > max_len:
            extra = len(self.hist_is_obs) - max_len
            self.hist_emb = self.hist_emb[extra:]
            self.hist_box = self.hist_box[extra:]
            self.hist_is_obs = self.hist_is_obs[extra:]


class RuntimeTracker:
    def __init__(
            self,
            model,
            # Sequence infos:
            sequence_hw: tuple,
            # Inference settings:
            use_sigmoid: bool = False,
            assignment_protocol: str = "hungarian",
            miss_tolerance: int = 30,
            det_thresh: float = 0.5,
            newborn_thresh: float = 0.5,
            id_thresh: float = 0.1,
            iou_thresh: float = 0.1,
            emb_cos_weight: float = 1.0,
            emb_mse_weight: float = 0.5,
            iou_weight: float = 1.0,
            area_thresh: int = 0,
            only_detr: bool = False,
            dtype: torch.dtype = torch.float32,
            debug: bool = False,
    ):
        self.model = model
        self.model.eval()

        self.dtype = dtype

        # For FP16:
        if self.dtype != torch.float32:
            if self.dtype == torch.float16:
                self.model.half()
            else:
                raise NotImplementedError(f"Unsupported dtype {self.dtype}.")

        self.use_sigmoid = use_sigmoid
        self.assignment_protocol = assignment_protocol.lower()
        self.miss_tolerance = miss_tolerance
        self.det_thresh = det_thresh
        self.newborn_thresh = newborn_thresh
        self.id_thresh = id_thresh
        self.emb_cos_weight = emb_cos_weight
        self.emb_mse_weight = emb_mse_weight
        self.area_thresh = area_thresh
        self.only_detr = only_detr
        self.history_len = 30
        self.debug = debug or bool(os.environ.get("HIMOT_DEBUG", ""))

        self.bbox_unnorm = torch.tensor(
            [sequence_hw[1], sequence_hw[0], sequence_hw[1], sequence_hw[0]],
            dtype=dtype,
            device=distributed_device(),
        )

        # Online tracking states:
        self.next_id = 0
        self.tracks: List[TrackState] = []

        self.current_track_results = {}
        return

    @torch.no_grad()
    def update(self, image):
        detr_out = self.model(frames=image, part="detr")
        scores, categories, boxes, track_emb = self._get_activate_detections(detr_out=detr_out)

        num_tracks_before = len(self.tracks)
        num_det = len(scores)

        if self.only_detr:
            self.tracks = []
            ids = torch.arange(self.next_id, self.next_id + num_det, device=boxes.device, dtype=torch.int64)
            self.next_id += num_det
            self.current_track_results = {
                "score": scores,
                "category": categories,
                "bbox": box_cxcywh_to_xywh(boxes) * self.bbox_unnorm,
                "id": ids,
            }
            if self.debug:
                self._debug_print(num_tracks_before, num_det, 0, num_det, len(self.tracks), None, None)
            return

        if num_det == 0:
            self._append_miss_to_all()
            self._filter_out_inactive_tracks()
            self.current_track_results = {
                "score": torch.zeros((0,), device=distributed_device()),
                "category": torch.zeros((0,), dtype=torch.int64, device=distributed_device()),
                "bbox": torch.zeros((0, 4), device=distributed_device()),
                "id": torch.zeros((0,), dtype=torch.int64, device=distributed_device()),
            }
            if self.debug:
                self._debug_print(num_tracks_before, num_det, 0, 0, len(self.tracks), None, None)
            return

        if len(self.tracks) == 0:
            newborn_mask = scores > self.newborn_thresh
            newborn_idxs = torch.nonzero(newborn_mask, as_tuple=False).flatten()
            newborn_ids = []
            for det_idx in newborn_idxs.tolist():
                det_id = self._create_new_track(track_emb=track_emb[det_idx], box=boxes[det_idx])
                newborn_ids.append(det_id)

            if len(newborn_idxs) > 0:
                ids = torch.tensor(newborn_ids, device=boxes.device, dtype=torch.int64)
                self.current_track_results = {
                    "score": scores[newborn_idxs],
                    "category": categories[newborn_idxs],
                    "bbox": box_cxcywh_to_xywh(boxes[newborn_idxs]) * self.bbox_unnorm,
                    "id": ids,
                }
            else:
                self.current_track_results = {
                    "score": torch.zeros((0,), device=distributed_device()),
                    "category": torch.zeros((0,), dtype=torch.int64, device=distributed_device()),
                    "bbox": torch.zeros((0, 4), device=distributed_device()),
                    "id": torch.zeros((0,), dtype=torch.int64, device=distributed_device()),
                }
            if self.debug:
                self._debug_print(num_tracks_before, num_det, 0, len(newborn_idxs), len(self.tracks), None, None)
            return

        track_seq, bbox_seq, pad_mask, miss_mask = self._build_td_inputs(
            device=track_emb.device,
            track_dim=track_emb.shape[-1],
        )
        td_out = self.model(
            seq_info={
                "track_seq": track_seq,
                "bbox_seq": bbox_seq,
                "pad_mask": pad_mask,
                "miss_mask": miss_mask,
            },
            part="traj_decoder",
        )
        pred_emb = td_out["pred_emb"]
        pred_box = td_out["pred_box"]

        pred_norm = F.normalize(pred_emb, p=2, dim=-1)
        det_norm = F.normalize(track_emb, p=2, dim=-1)
        emb_cost_cos = 1.0 - (pred_norm @ det_norm.T)
        diff = pred_norm[:, None, :] - det_norm[None, :, :]
        emb_cost_mse = (diff * diff).mean(dim=-1)

        cost = self.emb_cos_weight * emb_cost_cos + self.emb_mse_weight * emb_cost_mse
        if self.id_thresh is not None:
            cost = torch.where(emb_cost_cos > self.id_thresh, cost.new_tensor(1e6), cost)

        cost_cpu = cost.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_cpu)

        matches = []
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            if cost_cpu[r, c] < 1e5:
                matches.append((r, c))

        matched_track_idxs = set([m[0] for m in matches])
        matched_det_idxs = set([m[1] for m in matches])

        # Update matched tracks
        det_to_track_id = {}
        for track_idx, det_idx in matches:
            self.tracks[track_idx].append_obs(
                emb=track_emb[det_idx].detach(),
                box=boxes[det_idx].detach(),
                max_len=self.history_len,
            )
            det_to_track_id[det_idx] = self.tracks[track_idx].track_id

        # Update unmatched tracks
        for i in range(len(self.tracks)):
            if i not in matched_track_idxs:
                self.tracks[i].append_miss(max_len=self.history_len)

        # Remove inactive tracks
        self._filter_out_inactive_tracks()

        # Handle unmatched detections
        newborn_mask = scores > self.newborn_thresh
        newborn_det_idxs = [i for i in range(num_det) if i not in matched_det_idxs and newborn_mask[i].item()]
        for det_idx in newborn_det_idxs:
            det_id = self._create_new_track(track_emb=track_emb[det_idx], box=boxes[det_idx])
            det_to_track_id[det_idx] = det_id

        # Build outputs: matched + newborn
        keep_det_idxs = sorted(det_to_track_id.keys())
        if len(keep_det_idxs) > 0:
            keep_det_idxs_t = torch.tensor(keep_det_idxs, device=boxes.device, dtype=torch.int64)
            ids = torch.tensor(
                [det_to_track_id[i] for i in keep_det_idxs],
                device=boxes.device,
                dtype=torch.int64,
            )
            self.current_track_results = {
                "score": scores[keep_det_idxs_t],
                "category": categories[keep_det_idxs_t],
                "bbox": box_cxcywh_to_xywh(boxes[keep_det_idxs_t]) * self.bbox_unnorm,
                "id": ids,
            }
        else:
            self.current_track_results = {
                "score": torch.zeros((0,), device=distributed_device()),
                "category": torch.zeros((0,), dtype=torch.int64, device=distributed_device()),
                "bbox": torch.zeros((0, 4), device=distributed_device()),
                "id": torch.zeros((0,), dtype=torch.int64, device=distributed_device()),
            }

        if self.debug:
            cost_min = float(cost.min().item()) if cost.numel() > 0 else None
            cost_mean = float(cost.mean().item()) if cost.numel() > 0 else None
            self._debug_print(
                num_tracks_before,
                num_det,
                len(matches),
                len(newborn_det_idxs),
                len(self.tracks),
                cost_min,
                cost_mean,
            )
        return

    def get_track_results(self):
        return self.current_track_results

    def _get_activate_detections(self, detr_out: dict):
        logits = detr_out["pred_logits"][0]
        boxes = detr_out["pred_boxes"][0]
        track_emb = detr_out["track_emb"][0]
        scores = logits.sigmoid()
        scores, categories = torch.max(scores, dim=-1)
        area = boxes[:, 2] * self.bbox_unnorm[2] * boxes[:, 3] * self.bbox_unnorm[3]
        activate_indices = (scores > self.det_thresh) & (area > self.area_thresh)
        boxes = boxes[activate_indices]
        track_emb = track_emb[activate_indices]
        scores = scores[activate_indices]
        categories = categories[activate_indices]
        return scores, categories, boxes, track_emb

    def _build_td_inputs(self, device, track_dim: int):
        n_tracks = len(self.tracks)

        track_seq = torch.zeros((n_tracks, self.history_len, track_dim), dtype=torch.float32, device=device)
        bbox_seq = torch.zeros((n_tracks, self.history_len, 4), dtype=torch.float32, device=device)
        pad_mask = torch.ones((n_tracks, self.history_len), dtype=torch.bool, device=device)
        miss_mask = torch.zeros((n_tracks, self.history_len), dtype=torch.bool, device=device)

        for i, track in enumerate(self.tracks):
            hist_len = len(track.hist_is_obs)
            pad_len = max(0, self.history_len - hist_len)
            hist_emb = track.hist_emb[-self.history_len:]
            hist_box = track.hist_box[-self.history_len:]
            hist_is_obs = track.hist_is_obs[-self.history_len:]

            for j in range(len(hist_is_obs)):
                pos = pad_len + j
                pad_mask[i, pos] = False
                if hist_is_obs[j]:
                    if hist_emb[j] is not None:
                        track_seq[i, pos] = hist_emb[j]
                    if hist_box[j] is not None:
                        bbox_seq[i, pos] = hist_box[j]
                else:
                    miss_mask[i, pos] = True

        return track_seq, bbox_seq, pad_mask, miss_mask

    def _append_miss_to_all(self):
        for track in self.tracks:
            track.append_miss(max_len=self.history_len)

    def _filter_out_inactive_tracks(self):
        self.tracks = [t for t in self.tracks if t.miss_count <= self.miss_tolerance]

    def _create_new_track(self, track_emb: torch.Tensor, box: torch.Tensor):
        track_id = self.next_id
        self.next_id += 1
        self.tracks.append(TrackState(
            track_id=track_id,
            last_box=box.detach(),
            hist_emb=[track_emb.detach()],
            hist_box=[box.detach()],
            hist_is_obs=[True],
            miss_count=0,
        ))
        return track_id

    def _debug_print(
            self,
            num_tracks_before: int,
            num_det: int,
            num_matches: int,
            num_newborn: int,
            num_tracks_after: int,
            cost_min: Optional[float],
            cost_mean: Optional[float],
    ):
        print(
            f"[HiMOT-RT] tracks: {num_tracks_before} -> {num_tracks_after} | "
            f"dets: {num_det} | matches: {num_matches} | newborn: {num_newborn} | "
            f"cost_min: {cost_min} | cost_mean: {cost_mean}"
        )
