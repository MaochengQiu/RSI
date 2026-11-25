"""Export the single frozen FP16 matrix. Verify its bytes against the evaluated winner."""

import argparse
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().cpu().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("original_matrix", type=Path)
    args = parser.parse_args()
    m = json.loads((args.results / "manifest.json").read_text())
    selected = json.loads((args.results / "selection.json").read_text())
    target, seed = m["target"], selected["winner"]
    original = load_file(args.original_matrix)[target]
    assert digest(original) == m["weight_hashes"]["baseline"]
    noise = torch.randn(original.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
    weight = (original.float() + m["sigma"] * noise).to(original.dtype)
    expected = selected["winner_sha256"]
    assert digest(weight) == expected == m["weight_hashes"][str(seed)]
    path = args.results / "selected-target.safetensors"
    save_file({target: weight}, str(path), metadata={"seed": str(seed), "sigma": str(m["sigma"]),
              "model_revision": m["model_revision"], "weight_sha256": expected})
    assert digest(load_file(path)[target]) == expected
    print(path, expected)


if __name__ == "__main__":
    main()
