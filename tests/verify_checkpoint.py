"""Compare the cleaned RIFT runtime against immutable pre-cleanup GPU outputs."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from raqvc.training.c512 import DEFAULT_CHECKPOINT, load_bundle, endpoint
from raqvc.training.codec_training import online_reconstruction
from raqvc.training.conditions import build_condition


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, default=DEFAULT_CHECKPOINT.parent.parent/'preservation/verification/reference_latents.pt')
    p.add_argument('--output', type=Path, default=DEFAULT_CHECKPOINT.parent.parent/'preservation/verification/INFERENCE_PARITY.json')
    a = p.parse_args()
    torch.set_num_threads(4)
    model, heads, codec, _ = load_bundle(DEFAULT_CHECKPOINT, 'cuda')
    rows = []
    for case in torch.load(a.reference, map_location='cpu', weights_only=True):
        with torch.inference_mode():
            source, _, bpp, sigma, width = online_reconstruction(codec, case['clean'].cuda(), case['qp'], hard_eval=True)
            condition = build_condition(codec, source, sigma, width, bpp, case['qp'], 'cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                value, _ = endpoint(model, heads, condition)
        row = {key:case[key] for key in ('resolution','t','qp')}
        row.update(source_max_abs=float((source.cpu()-case['source']).abs().max()),
                   rate_max_abs=float((bpp.cpu()-case['bpp']).abs().max()),
                   output_max_abs=float((value.cpu()-case['output']).abs().max()))
        rows.append(row)
        print(json.dumps(row), flush=True)
    complete = all(r[k] == 0 for r in rows for k in ('source_max_abs','rate_max_abs','output_max_abs'))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(complete=complete, cases=rows, reference=str(a.reference)), indent=2))
    if not complete: raise AssertionError('C512 inference differs from the pre-cleanup reference')


if __name__ == '__main__':
    main()
