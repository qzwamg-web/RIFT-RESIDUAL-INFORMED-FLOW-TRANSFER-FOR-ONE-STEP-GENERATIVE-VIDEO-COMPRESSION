"""Build the fixed C512 backbone."""

from raqvc.backbone.dit import RAQFlowDiT
from raqvc.backbone.c512_dit import Task438WindowedDiT
import torch


def build_task438_windowed_dit(wan_checkpoint_dir, device):
    return Task438WindowedDiT(RAQFlowDiT(wan_checkpoint_dir, device)).to(device=device, dtype=torch.float32)
