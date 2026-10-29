"""Persistent official FloLPIPS video metric on native RGB float frames.

This wraps ``danier97/flolpips::calc_flolpips``'s *video* formula, bypassing
only its MP4 -> YUV -> RGB I/O. Each pair t,t+1 scores the appearance error
at t, weighted by PWC(ref_t, ref_t+1) - PWC(dis_t, dis_t+1). It is not the
upstream triplet/interpolation API. Callers must retain pairs across GOPs.

Upstream applies no crop or external resize. AlexNet sees native RGB;
PWC bilinearly resizes each dimension to ceil64, then scales flow back.
The original RGB order, LPIPS [-1,1] conversion, Alex v0.1 weights, PWC
weights, backwarp geometry, and flow-weight normalization are preserved.
Upstream's zero flow-difference denominator is undefined. Exact-reference
frame pairs explicitly return zero, with a diagnostic flag; all other
undefined values retain NaNs plus flags for the evaluator to record as null.

Official source files and weights live in the torch hub cache, with their
license and SHA256 provenance. No global imports or libraries are patched.
Only the obsolete CuPy correlation kernel is replaced by its exact
81-displacement channel-mean formula in torch; summation roundoff can differ.
The backwarp grid uses its input device instead of hard-coded cuda:0.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import tempfile
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

OFFICIAL_REPOSITORY = "https://github.com/danier97/flolpips"
OFFICIAL_COMMIT = "66dd39937e961c56cf162651074eacbfb5f9aab1"
_RAW = f"https://raw.githubusercontent.com/danier97/flolpips/{OFFICIAL_COMMIT}/"
_SOURCE_HASHES = {
    "flolpips.py": "2f1c976ca4d63ef9545cd6032c081868dda19245602f2f9bd584ae4685056e34",
    "pwcnet.py": "9a0b01f10fa0625c2a0f5d26cb859cd47174747588d38e8bd1f270ed0cf3309d",
    "pretrained_networks.py": "6a27f714c51796db466e86bebba6a617c1bfc4d566f3a1756497629c1248686e",
    "utils.py": "336c0b5d70009cb88327732bb45556758c7c728676fd53b4a08ec780118b903b",
    "correlation/correlation.py": "5c11b0dfba92130505e9ca335b5da776792980f7fdb4cfe2a10465c6cdfaa159",
    "weights/v0.1/alex.pth": "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",
    "LICENSE": "81c6569a99d1c6093f17dbdb6c610f1eb29e85fc59923c6beea73ba1c6c88144",
    "README.md": "117ff9e31fe0d9cd78c94b776de526e4958a6690bb8d4891df57e96dd39c5603",
}
_WEIGHTS = {
    "network-default.pytorch": (
        "https://content.sniklaus.com/github/pytorch-pwc/network-default.pytorch",
        "ad2dc62b63b9d10ad42a6969179b3f996d4115e2ec6421bf532f99cf278ffcd2",
    ),
    "alexnet-owt-7be5be79.pth": (
        "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
        "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_flolpips_assets(cache_dir: str | Path | None = None, *, download: bool = False) -> dict[str, Any]:
    """Verify cached official assets; optionally download missing files only.

    Existing mismatched files are rejected rather than overwritten. Source
    cache location is overridable; ImageNet/PWC weights use torch hub's usual
    checkpoints directory so torchvision uses the already verified AlexNet.
    """
    hub = Path(torch.hub.get_dir())
    source = Path(cache_dir) if cache_dir is not None else hub / f"flolpips-danier97-{OFFICIAL_COMMIT}"
    requested = [(source / name, _RAW + name, sha) for name, sha in _SOURCE_HASHES.items()]
    requested += [(hub / "checkpoints" / name, url, sha) for name, (url, sha) in _WEIGHTS.items()]
    assets = []
    for path, url, expected_sha in requested:
        if not path.is_file():
            if not download:
                raise FileNotFoundError(f"FloLPIPS asset missing: {path}; run prepare_flolpips_assets(download=True)")
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".download") as temporary:
                torch.hub.download_url_to_file(url, temporary.name, progress=False)
                actual_sha = _sha256(Path(temporary.name))
                if actual_sha != expected_sha:
                    raise ValueError(f"FloLPIPS download SHA256 mismatch: {url}: {actual_sha}")
                # Exclusive creation avoids replacing historical/user cache files.
                with path.open("xb") as output, open(temporary.name, "rb") as downloaded:
                    for chunk in iter(lambda: downloaded.read(1024 * 1024), b""):
                        output.write(chunk)
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            raise ValueError(f"FloLPIPS cached SHA256 mismatch: {path}: {actual_sha}")
        assets.append({"path": str(path.resolve()), "url": url, "sha256": actual_sha, "bytes": path.stat().st_size})
    provenance = {"repository": OFFICIAL_REPOSITORY, "commit": OFFICIAL_COMMIT, "assets": assets}
    manifest = source / "PROVENANCE.json"
    if not manifest.exists():
        with manifest.open("x") as output:
            json.dump(provenance, output, indent=2, sort_keys=True)
            output.write("\n")
    return {"source_dir": str(source), "pwc_weights": str(hub / "checkpoints/network-default.pytorch"), **provenance}


def torch_correlation(tenFirst: torch.Tensor, tenSecond: torch.Tensor) -> torch.Tensor:
    """Official cost volume: mean_c first[c,y,x]*second[c,y+dy,x+dx].

    Displacement ordering is dy=-4..4, then dx=-4..4; out-of-image second
    samples are zero. A single displacement at a time bounds native1080 RAM.
    """
    if tenFirst.ndim != 4 or tenFirst.shape != tenSecond.shape:
        raise ValueError("PWC correlation needs equal NCHW tensors")
    height, width = tenFirst.shape[-2:]
    padded = F.pad(tenSecond, (4, 4, 4, 4))
    return torch.stack(
        [
            (tenFirst * padded[:, :, dy : dy + height, dx : dx + width]).mean(dim=1)
            for dy in range(9)
            for dx in range(9)
        ],
        dim=1,
    )


def _definitions(path: Path, names: set[str], namespace: dict[str, Any], source: str | None = None) -> None:
    """Load selected unchanged upstream definitions without legacy imports."""
    tree = ast.parse(source if source is not None else path.read_text(), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError(f"Unexpected official source definitions: {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def _replace_once(source: str, original: str, replacement: str) -> str:
    if source.count(original) != 1:
        raise RuntimeError(f"Unexpected official PWC compatibility target: {original}")
    return source.replace(original, replacement, 1)


def _official_models(assets: dict[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Instantiate actual upstream PWC/FloLPIPS with isolated compatibility."""
    from collections import namedtuple
    from torchvision import models as tv

    source = Path(assets["source_dir"])
    pn = {"__name__": "_task938_flolpips_pretrained", "torch": torch, "tv": tv, "namedtuple": namedtuple}
    _definitions(source / "pretrained_networks.py", {"alexnet"}, pn)
    utils = {"__name__": "_task938_flolpips_utils", "torch": torch}
    _definitions(source / "utils.py", {"normalize_tensor"}, utils)
    metric = {
        "__name__": "_task938_official_flolpips",
        "torch": torch,
        "nn": torch.nn,
        "F": F,
        "pn": types.SimpleNamespace(alexnet=pn["alexnet"]),
        "utils": types.SimpleNamespace(normalize_tensor=utils["normalize_tensor"]),
    }
    _definitions(
        source / "flolpips.py",
        {
            "LPIPS",
            "FloLPIPS",
            "ScalingLayer",
            "NetLinLayer",
            "mw_spatial_average",
            "spatial_average",
            "upsample",
        },
        metric,
    )
    loss_fn = metric["FloLPIPS"](net="alex", version="0.1", model_path=str(source / "weights/v0.1/alex.pth"))

    pwc_source = _replace_once(
        (source / "pwcnet.py").read_text(),
        "torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-pwc/network-' + 'default' + '.pytorch')",
        "_load_pwc_state_dict()",
    )
    pwc_source = _replace_once(
        pwc_source,
        "torch.cat([ tenHor, tenVer ], 1).cuda()",
        "torch.cat([ tenHor, tenVer ], 1).to(device=tenFlow.device)",
    )
    pwc = {
        "__name__": "_task938_official_pwc",
        "torch": torch,
        "math": math,
        "correlation": types.SimpleNamespace(FunctionCorrelation=torch_correlation),
        "_load_pwc_state_dict": lambda: torch.load(assets["pwc_weights"], map_location="cpu", weights_only=True),
    }
    _definitions(source / "pwcnet.py", {"Network", "backwarp"}, pwc, source=pwc_source)
    return pwc["Network"](), loss_fn


class NativeFloLPIPS:
    """Load both frozen FP32 models once and evaluate adjacent RGB pairs.

    ``per_pair(ref_prev, ref_cur, dis_prev, dis_cur)`` accepts float [N,3,H,W]
    in [0,1] and returns float [N] on the model device. No GOP/reset state is
    maintained here; the sequence evaluator owns adjacency and aggregation.
    """

    def __init__(
        self,
        device: str | torch.device = "cuda",
        *,
        cache_dir: str | Path | None = None,
        download: bool = False,
    ) -> None:
        import torchvision

        self.device = torch.device(device)
        assets = prepare_flolpips_assets(cache_dir, download=download)
        self.flownet, self.loss_fn = _official_models(assets)
        for model in (self.flownet, self.loss_fn):
            model.to(device=self.device, dtype=torch.float32).eval().requires_grad_(False)
        self.last_pair_status: list[str] = []
        self.pair_status_counts: dict[str, int] = {}
        self.metadata = {
            "name": "FloLPIPS",
            "lower_is_better": True,
            "protocol": "official_video_formula_on_native_rgb_float",
            "reference_api": "danier97/flolpips::calc_flolpips",
            "flow_network": "official PWCNet default",
            "perceptual_network": "AlexNet ImageNet + LPIPS Alex v0.1 calibration",
            "pair_direction": "previous_to_current",
            "appearance_frame": "previous",
            "flow_weight": "magnitude(PWC(ref_prev,ref_cur)-PWC(dis_prev,dis_cur))",
            "weight_normalization": "per sample and LPIPS layer; sum of interpolated flow magnitude; no epsilon",
            "zero_flow_difference": "exact-reference identity pair -> explicit zero convention; otherwise undefined NaN plus last_pair_status, no fallback",
            "input": "native RGB float32 [0,1]; no quantization, MP4/YUV roundtrip, crop, or external resize",
            "lpips_preprocessing": "normalize=True converts RGB to [-1,1]; official v0.1 ScalingLayer",
            "pwc_preprocessing": "RGB unchanged; bilinear align_corners=False resize H/W to ceil64; flow resized and component-scaled back",
            "correlation_backend": "torch; official 9x9 displacement, zero padding, channel mean; mathematically equivalent, floating reduction order can differ",
            "compatibility_changes": [
                "CuPy correlation to torch formula",
                "backwarp cuda:0 to input device",
                "PWC weight load from verified local cache via HTTPS origin",
            ],
            "aggregation": "caller averages every adjacent pair; include cross-GOP pairs; final frame has no outgoing pair",
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "dtype": "float32; autocast disabled",
            "provenance": assets,
            "scope": "auxiliary motion-aware quality metric originally designed for frame interpolation",
        }

    @torch.inference_mode()
    def per_pair(
        self,
        ref_prev: torch.Tensor,
        ref_cur: torch.Tensor,
        dis_prev: torch.Tensor,
        dis_cur: torch.Tensor,
    ) -> torch.Tensor:
        frames = (ref_prev, ref_cur, dis_prev, dis_cur)
        shape = ref_prev.shape
        if len(shape) != 4 or shape[1] != 3 or shape[0] < 1 or min(shape[-2:]) < 64:
            raise ValueError("FloLPIPS inputs must be [N,3,H,W], N>=1 and H,W>=64")
        if any(frame.shape != shape or not frame.is_floating_point() for frame in frames):
            raise ValueError("FloLPIPS requires four equal-shaped floating RGB tensors")
        # Never clamp or convert from a different range silently.
        if any(
            not bool(torch.isfinite(frame).all()) or bool((frame.amin() < 0) | (frame.amax() > 1)) for frame in frames
        ):
            raise ValueError("FloLPIPS RGB inputs must be finite and within [0,1]")
        identity = [
            torch.equal(ref_prev[index], dis_prev[index]) and torch.equal(ref_cur[index], dis_cur[index])
            for index in range(shape[0])
        ]
        if all(identity):
            scores = torch.zeros(shape[0], device=self.device, dtype=torch.float32)
            self._record_pair_status(["exact_reference_identity_zero"] * shape[0])
            return scores
        with torch.autocast(device_type=self.device.type, enabled=False):
            previous_ref, current_ref, previous_dis, current_dis = (
                frame.to(device=self.device, dtype=torch.float32) for frame in frames
            )
            flow_ref = self.flownet(previous_ref, current_ref)
            flow_dis = self.flownet(previous_dis, current_dis)
            flow_difference = flow_ref - flow_dis
            scores = self.loss_fn(previous_ref, previous_dis, flow_difference, normalize=True).reshape(shape[0])
        statuses = []
        for index, is_identity in enumerate(identity):
            if is_identity:
                scores[index] = 0
                statuses.append("exact_reference_identity_zero")
            elif bool(torch.isfinite(scores[index])):
                statuses.append("ok")
            elif bool((flow_difference[index] == 0).all()):
                statuses.append("undefined_official_zero_flow_weight")
            else:
                statuses.append("undefined_official_nonfinite")
        self._record_pair_status(statuses)
        return scores

    def _record_pair_status(self, statuses: list[str]) -> None:
        self.last_pair_status = statuses
        for status in statuses:
            self.pair_status_counts[status] = self.pair_status_counts.get(status, 0) + 1

    __call__ = per_pair
