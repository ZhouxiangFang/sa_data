"""How far a fine-tuned checkpoint moved from its base model, tensor by tensor.

    python weight_diff.py BASE_DIR CHECKPOINT_DIR...

For each checkpoint, prints the fraction of bf16 weights that changed at all, the overall
relative change ||W - W0|| / ||W0||, the median per-tensor relative change, and the extremes.
Tensors are read one at a time from the safetensors shards, so memory stays low.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def tensor_files(model_dir):
    index = Path(model_dir) / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return {name: Path(model_dir) / shard for name, shard in weight_map.items()}
    files = {}
    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(shard, "pt") as handle:
            files.update({name: shard for name in handle.keys()})
    return files


def read(path, name, handles):
    if path not in handles:
        handles[path] = safe_open(path, "pt")
    return handles[path].get_tensor(name)


base_dir, *checkpoints = sys.argv[1:]
base_files = tensor_files(base_dir)
for checkpoint in checkpoints:
    if not (Path(checkpoint) / "config.json").exists():
        print(f"\n{checkpoint}\n  no checkpoint (training failed or not run); skipped")
        continue
    files = tensor_files(checkpoint)
    missing = sorted(set(base_files) - set(files))
    extra = sorted(set(files) - set(base_files))
    handles, base_handles = {}, {}
    changed = total = 0
    diff_sq = base_sq = 0.0
    per_tensor = []
    for name in sorted(base_files):
        if name not in files:
            continue
        w0 = read(base_files[name], name, base_handles)
        w = read(files[name], name, handles)
        if w.shape != w0.shape:
            print(f"  shape mismatch {name}: {tuple(w0.shape)} vs {tuple(w.shape)}")
            continue
        changed += int((w != w0).sum())
        total += w.numel()
        d = (w.float() - w0.float())
        dn, bn = float(d.norm()), float(w0.float().norm())
        diff_sq += dn ** 2
        base_sq += bn ** 2
        per_tensor.append((dn / bn if bn else float("nan"), name, w0.dtype, w.dtype))
    per_tensor.sort()
    rel = [r for r, *_ in per_tensor]
    print(f"\n{checkpoint}")
    print(f"  tensors compared {len(per_tensor)}; missing {len(missing)}; extra {len(extra)}; "
          f"dtypes base {per_tensor[0][2]} / checkpoint {per_tensor[0][3]}")
    print(f"  weights changed: {changed / total:.2%} of {total:,}")
    print(f"  overall relative change ||W - W0|| / ||W0||: {(diff_sq / base_sq) ** 0.5:.3e}")
    print(f"  per-tensor relative change: median {rel[len(rel) // 2]:.3e}, "
          f"min {rel[0]:.3e} ({per_tensor[0][1]}), max {rel[-1]:.3e} ({per_tensor[-1][1]})")
    if missing or extra:
        print(f"  missing: {missing[:5]}  extra: {extra[:5]}")
