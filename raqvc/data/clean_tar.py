"""Map-style direct clean latent tar reader; no codec-dependent cache."""

import io, json, tarfile, hashlib
from pathlib import Path
import torch
from torch.utils.data import Dataset


def build_index(root, output):
    root = Path(root)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((root / "CACHE_MANIFEST.json").read_text())
    if not manifest.get("complete") or "latent_shape" not in manifest or "target_size" not in manifest:
        raise ValueError("complete clean manifest required")
    expected_latent = list(manifest["latent_shape"])
    expected_target = list(manifest["target_size"])
    rows = []
    for shard in manifest["shards"]:
        path = root / shard
        pending = {}
        with tarfile.open(path, "r:") as archive:
            for member in archive:
                if member.name.endswith(".latent.pth"):
                    key = member.name[:-11]
                    pending.setdefault(key, {}).update(offset=member.offset_data, size=member.size)
                elif member.name.endswith(".meta.json"):
                    key = member.name[:-10]
                    pending.setdefault(key, {})["meta"] = json.load(archive.extractfile(member))
        for key, v in pending.items():
            if set(v) != {"offset", "size", "meta"}:
                raise ValueError("incomplete sample " + key)
            m = v["meta"]
            if (
                m.get("latent_shape") != expected_latent
                or m.get("target_size") != expected_target
                or not m.get("spatial_transform")
            ):
                raise ValueError("shape/crop metadata " + key)
            rows.append(dict(key=key, shard=str(path), **v))
    if len(rows) != manifest["samples"] or len({r["key"] for r in rows}) != len(rows):
        raise ValueError("sample count/duplicate")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema": "task938_clean_tar_index_v1",
                "source_manifest": str(root / "CACHE_MANIFEST.json"),
                "source_manifest_sha256": hashlib.sha256((root / "CACHE_MANIFEST.json").read_bytes()).hexdigest(),
                "rows": rows,
            }
        )
    )
    return len(rows)


class CleanTar(Dataset):
    def __init__(self, index, load_rgb=True, allow_arbitrary_temporal=False):
        self.index = Path(index)
        d = json.loads(self.index.read_text())
        if d["schema"] != "task938_clean_tar_index_v1":
            raise ValueError("index schema")
        p = Path(d["source_manifest"])
        if hashlib.sha256(p.read_bytes()).hexdigest() != d["source_manifest_sha256"]:
            raise ValueError("source manifest changed")
        self.rows = d["rows"]
        self.resolution = int(self.rows[0]["meta"]["target_size"][0])
        self.latent_shape = tuple(self.rows[0]["meta"]["latent_shape"])
        self.load_rgb = bool(load_rgb)
        self.allow_arbitrary_temporal = bool(allow_arbitrary_temporal)
        if any(tuple(r["meta"]["target_size"]) != (self.resolution, self.resolution) for r in self.rows):
            raise ValueError("mixed resolutions in clean index")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with open(r["shard"], "rb") as f:
            f.seek(r["offset"])
            payload = f.read(r["size"])
        clean = torch.load(io.BytesIO(payload), weights_only=True, map_location="cpu")
        expected_hw = self.resolution
        if self.allow_arbitrary_temporal:
            if (
                clean.ndim != 4
                or tuple(clean.shape[0:1]) != (48,)
                or tuple(clean.shape[-2:]) != (expected_hw // 16, expected_hw // 16)
                or int(clean.shape[1]) < 2
            ):
                raise ValueError(
                    f"clean GOP shape must be [48,T,{expected_hw//16},{expected_hw//16}], got {tuple(clean.shape)}"
                )
            declared = int(r["meta"].get("latent_frames", clean.shape[1]) or clean.shape[1])
            if declared != int(clean.shape[1]):
                raise ValueError("latent_frames metadata mismatch")
        else:
            expected_latent = (48, 7, expected_hw // 16, expected_hw // 16)
            if tuple(clean.shape) != expected_latent:
                raise ValueError("clean shape")
        if not torch.isfinite(clean).all():
            raise ValueError("nonfinite latent")
        if not self.load_rgb:
            return {"clean": clean}
        from raqvc.data.latent_dataset import _load_aligned_source_video

        rgb = _load_aligned_source_video(r["meta"], (self.resolution, self.resolution))
        expected_raw = 1 + 4 * (int(clean.shape[1]) - 1)
        if tuple(rgb.shape) != (3, expected_raw, expected_hw, expected_hw) or not torch.isfinite(rgb).all():
            raise ValueError(f"RGB shape: expected {(3,expected_raw,expected_hw,expected_hw)}, got {tuple(rgb.shape)}")
        return {"clean": clean, "source_video": rgb}


class Clean512(CleanTar):
    def __init__(self, index, load_rgb=True):
        super().__init__(index, load_rgb=load_rgb)
        if self.resolution != 512:
            raise ValueError("clean index is not 512")


class CleanGOP512(CleanTar):
    def __init__(self, index, load_rgb=True):
        super().__init__(index, load_rgb=load_rgb, allow_arbitrary_temporal=True)
        if self.resolution != 512:
            raise ValueError("clean index is not 512")


class Clean256(CleanTar):
    def __init__(self, index, load_rgb=True):
        super().__init__(index, load_rgb=load_rgb)
        if self.resolution != 256:
            raise ValueError("clean index is not 256")


class CleanGOP256(CleanTar):
    def __init__(self, index, load_rgb=True):
        super().__init__(index, load_rgb=load_rgb, allow_arbitrary_temporal=True)
        if self.resolution != 256:
            raise ValueError("clean index is not 256")


def project_descending(values):
    import math

    blocks = []
    for value in values:
        blocks.append([math.log(value), 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            b = blocks.pop()
            a = blocks.pop()
            n = a[1] + b[1]
            blocks.append([(a[0] * a[1] + b[0] * b[1]) / n, n])
    return [math.exp(v) for v, n in blocks for _ in range(n)]
