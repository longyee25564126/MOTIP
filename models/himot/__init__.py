# Copyright (c) Ruopeng Gao. All Rights Reserved.

from .traj_token_encoder import TrajectoryTokenEncoder, build_trajectory_token_encoder
from .traj_decoder import TrajectoryDecoder, build_trajectory_decoder
from .td_criterion import TDCriterion, build_td_criterion

__all__ = [
    "TrajectoryTokenEncoder",
    "build_trajectory_token_encoder",
    "TrajectoryDecoder",
    "build_trajectory_decoder",
    "TDCriterion",
    "build_td_criterion",
    "HiMOT",
    "build",
    "build_himot",
]


def __getattr__(name: str):
    if name in {"HiMOT", "build", "build_himot"}:
        from .himot import HiMOT, build, build_himot
        return {"HiMOT": HiMOT, "build": build, "build_himot": build_himot}[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
