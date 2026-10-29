"""The codec architecture selected by RIFT. Historical alternatives are rejected."""

import json
import os
from pathlib import Path
from raqvc.codec.base import CodecResult, LatentEntropyCodec, GenerativeImageCodec, IntraResult
from raqvc.codec.latent_codec import LatentCodec

_PROFILE = Path(__file__).with_name("c512_profile.json")


def create_latent_codec(codec_type="rt_two_codec", **kwargs):
    if codec_type.replace("-", "_") not in ("rt_two", "rt_two_codec"):
        raise ValueError("only the C512 codec is supported")
    profile = json.loads(_PROFILE.read_text())
    device = kwargs.pop("device", "cuda:0")
    for key in ("type", "enabled", "image_checkpoint_path", "video_checkpoint_path"):
        kwargs.pop(key, None)
    for key, value in kwargs.items():
        if key not in profile or value != profile[key]:
            raise ValueError(f"codec.{key} differs from the fixed C512 architecture")
    root = (
        Path(os.environ.get("RAQVC_ARTIFACTS", "/qzwang/data/raqvc_artifacts"))
        / "C512_FINAL_v1/model/codec_initializers"
    )
    from raqvc.codec.rt_two_codec import RTTwoCodecPipeline

    return RTTwoCodecPipeline(
        device=device,
        image_checkpoint_path=str(root / "cvpr2025_image.pth.tar"),
        video_checkpoint_path=str(root / "cvpr2025_video.pth.tar"),
    )
