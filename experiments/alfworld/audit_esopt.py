"""Replay a completed ALFWorld ESOpt run without loading the language model."""

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

import run_experiment as alf
from run_esopt import (crossover, navigation_prediction, read, replay_trace, save,
                       summarize_candidates, upstream_directory, weight_digest)


def audit(output, data):
    protocol = read(output / "protocol.json")
    summary = read(output / "summary.json")
    development = read(output / "development.json")
    confirmation = read(output / "confirmation.json")
    update = read(output / "update.json")
    direction_ids = list(protocol["directions"])
    all_policies = ("base", *direction_ids)
    layouts = set()
    episodes = 0
    rollouts = {}
    for phase, cases in protocol["scenes"].items():
        phase_layouts = {scene["layout"] for scene in cases}
        assert len(cases) == 6 and not layouts & phase_layouts
        layouts.update(phase_layouts)
        policies = ("base", "updated") if phase == "aggregate" else all_policies
        records = {}
        for policy in policies:
            digest = update["updated_sha256"] if policy == "updated" else development["weights"][policy]
            for index, scene in enumerate(cases):
                path = output / phase / f"{policy}-{index+1:02}.json"
                record = read(path)
                assert (record["policy"], record["scene"], record["weight_sha256"]) == (policy, scene, digest)
                replay_trace(data, scene, path.with_suffix(".jsonl"), record)
                records[policy, index] = record
                if phase == "confirmation":
                    assert path.stat().st_mtime_ns >= (output / "development.json").stat().st_mtime_ns
                if phase == "aggregate":
                    assert path.stat().st_mtime_ns >= (output / "update.json").stat().st_mtime_ns
        rollouts[phase] = records
        with (output / f"{phase}-scores.csv").open(newline="", encoding="utf-8") as handle:
            actual = {(r["policy"], r["scene"]): (int(r["won"]), int(r["rounds"]), int(r["steps"]))
                      for r in csv.DictReader(handle)}
        assert actual == {(p, r["scene"]["id"]): (int(r["won"]), r["rounds"], r["steps"])
                          for (p, _), r in records.items()}
        assert summarize_candidates(records, cases, policies) == summary[phase]
        episodes += len(records)
    assert episodes == summary["episodes"] == 72
    dev_scenes = protocol["scenes"]["development"]
    dev_records = rollouts["development"]
    assert development["prediction"] == summary["prediction"] == {
        "crossover": crossover(summary["development"], direction_ids),
        "apple_navigation": navigation_prediction(dev_records, dev_scenes, direction_ids),
    }
    assert update["request"]["seeds"] == list(protocol["directions"].values())
    assert update["request"]["rewards"] == [
        sum(dev_records[p, i]["won"] for i in range(len(dev_scenes))) / len(dev_scenes) for p in direction_ids
    ]
    assert update["request"]["alpha"] == protocol["alpha"]
    assert (output / "update.json").stat().st_mtime_ns >= (output / "confirmation.json").stat().st_mtime_ns
    aggregate = rollouts["aggregate"]
    pairs = Counter(f"{int(aggregate['base', i]['won'])}->{int(aggregate['updated', i]['won'])}"
                    for i in range(len(protocol["scenes"]["aggregate"])))
    assert pairs == summary["aggregate_pairs"]
    assert confirmation["crossover"] == summary["crossover"]
    assert confirmation["apple_replication"] == summary["apple_replication"]
    print(f"Replayed {episodes} episodes. Aggregate paired outcomes: {dict(pairs)}.")
    for phase in ("development", "confirmation", "aggregate"):
        print(phase, {p: sum(x["won"] for x in group.values()) for p, group in summary[phase].items()})
    print("Development rewards:", update["request"]["rewards"])
    print("Updated weight digest differs:", update["updated_sha256"] != update["center_sha256"])


def audit_zero_update(output, model_dir):
    import torch

    sys.path.insert(0, str(upstream_directory()))
    from algorithms.es.seeded_model_es import SeedReplayModelES

    update = read(output / "update.json")
    assert update["diagnostics"]["weight_norm"] == 0.0
    torch.set_num_threads(4)
    with torch.inference_mode():
        model, _, _, _ = alf.load_model(model_dir)
        es = SeedReplayModelES()
        es.init(model)
        live = [(name, parameter) for name, _, parameter in es.parameter_infos]
        before = [(name, parameter.detach().cpu()) for name, parameter in live]
        assert weight_digest(before) == update["center_sha256"]
        es.update(**update["request"])
        assert weight_digest(live) == update["updated_sha256"]
        changes = {}
        representations = {torch.float16: torch.int16, torch.bfloat16: torch.int16,
                           torch.float32: torch.int32, torch.float64: torch.int64}
        for (name, original), (_, parameter) in zip(before, live):
            dtype = representations[original.dtype]
            after = parameter.detach().cpu()
            if torch.equal(original.view(dtype), after.view(dtype)):
                continue
            counts = Counter()
            for left, right in zip(original.flatten().split(2**22), after.flatten().split(2**22)):
                different = left.view(dtype) != right.view(dtype)
                counts["bitwise"] += int(different.sum())
                counts["numeric"] += int(((left != right) & different).sum())
                counts["negative_zero_to_positive"] += int(
                    (different & (left == 0) & (right == 0) & torch.signbit(left) & ~torch.signbit(right)).sum()
                )
            changes[name] = dict(counts)
    totals = {kind: sum(row[kind] for row in changes.values()) for kind in
              ("bitwise", "numeric", "negative_zero_to_positive")}
    save(output / "zero-update-diagnostics.json", dict(totals=totals, tensors=changes))
    print("Zero-update weight comparison:", totals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "rsi-data/alfworld")
    parser.add_argument("--model-dir", type=Path, default=Path.home() / "rsi-data/models/Qwen3-4B-1cfa9a7")
    parser.add_argument("--check-zero-update", action="store_true")
    args = parser.parse_args()
    audit(args.output, args.data_dir)
    if args.check_zero_update:
        audit_zero_update(args.output, args.model_dir)
