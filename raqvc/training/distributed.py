"""Exact mean gradients across workers, retaining globally-unused None gradients."""

import torch
import torch.distributed as dist


def reduce_gradients(parameters, world):
    if world == 1:
        return
    ps = [p for p in parameters if p.requires_grad]
    used = torch.tensor([p.grad is not None for p in ps], device=ps[0].device, dtype=torch.int32)
    dist.all_reduce(used, op=dist.ReduceOp.MAX)
    active = [p for p, u in zip(ps, used.tolist()) if u]
    if not active:
        return
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in active])
    dist.all_reduce(flat)
    flat.div_(world)
    offset = 0
    for p in active:
        g = flat[offset : offset + p.numel()].view_as(p)
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.copy_(g)
        offset += p.numel()
