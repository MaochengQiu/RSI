"""Measure within-round score reliability and action divergence without GPU work."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from statistics import median

import numpy as np

from analyze_mask15 import OUT, author_matrices, write_csv
from run_local import ROOT, upstream_directory
from run_direction_transfer import MASKS, paired, prediction as select_prediction, score_table
from run_observable_es import load_upstream, verification_rows


def rank(values):
    order = np.argsort(values, axis=-1, kind="stable")
    ordered = np.take_along_axis(values, order, axis=-1)
    size = values.shape[-1]
    edge = np.arange(size)
    starts = np.empty_like(values)
    starts[..., 0] = 0
    starts[..., 1:] = np.where(ordered[..., 1:] != ordered[..., :-1], edge[1:], 0)
    starts = np.maximum.accumulate(starts, axis=-1)
    ends = np.empty_like(values)
    ends[..., -1] = size - 1
    ends[..., :-1] = np.where(ordered[..., :-1] != ordered[..., 1:], edge[:-1], size - 1)
    ends = np.minimum.accumulate(ends[..., ::-1], axis=-1)[..., ::-1]
    result = np.empty(values.shape, dtype=float)
    np.put_along_axis(result, order, (starts + ends) / 2, axis=-1)
    result -= (size - 1) / 2
    return result


def reliability(matrices, halves):
    left = rank(halves @ matrices.transpose(0, 2, 1))
    right = rank((1 - halves) @ matrices.transpose(0, 2, 1))
    numerator = (left * right).sum(axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    valid = (denominator != 0).sum(axis=-1)
    correlations = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    mean = np.divide(correlations.sum(axis=-1), valid,
                     out=np.full(len(valid), np.nan), where=valid != 0)
    return mean, valid


def author_reliability(upstream):
    author_matrices()
    data = upstream / "data/sudoku/train.jsonl"
    masks = {row["id"]: row["mask_count"] for line in data.read_bytes().splitlines()
             if (row := json.loads(line))}
    with (OUT / "author-direction-task.csv").open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    task_ids = [key for key in raw[0] if key in masks]
    assert len(task_ids) == 32 and {masks[t] for t in task_ids} == {15}
    assert len(raw) == 3200
    rng = np.random.default_rng(20261129)
    halves = np.zeros((64, 32), dtype=np.int16)
    for half in halves:
        half[rng.choice(32, 16, replace=False)] = 1
    rows = []
    for gen in range(100):
        batch = raw[gen * 32 : (gen + 1) * 32]
        assert [(int(r["generation"]), int(r["direction"])) for r in batch] == [(gen, i) for i in range(32)]
        matrix = np.array([[int(r[t]) for t in task_ids] for r in batch], dtype=np.int16)
        null_matrices = rng.permuted(np.broadcast_to(matrix, (199, 32, 32)), axis=1)
        assert np.all(null_matrices.sum(axis=1) == matrix.sum(axis=0))
        observed, valid = reliability(matrix[None, :, :], halves)
        null, _ = reliability(null_matrices, halves)
        eligible_null = null[np.isfinite(null)]
        totals = matrix.sum(axis=0)
        rows.append(dict(round=gen+1, mask=15, all_success=int((totals == 32).sum()),
                         all_failure=int((totals == 0).sum()), mixed=int(((totals > 0) & (totals < 32)).sum()),
                         estimable_splits=int(valid[0]), split_half_spearman=float(observed[0]),
                         null_median=float(np.median(eligible_null)) if len(eligible_null) else None,
                         null_low=float(np.quantile(eligible_null, .025)) if len(eligible_null) else None,
                         null_high=float(np.quantile(eligible_null, .975)) if len(eligible_null) else None))
    write_csv("author-split-half.csv", rows)
    stages = []
    for label, (first, last) in {"early": (1, 20), "middle": (41, 60), "late": (81, 100),
                                "all": (1, 100)}.items():
        subset = rows[first-1:last]
        eligible = [r for r in subset if r["estimable_splits"]]
        stages.append(dict(stage=label, first=first, last=last, estimable_rounds=len(eligible),
                           median_all_success=median(r["all_success"] for r in subset),
                           median_all_failure=median(r["all_failure"] for r in subset),
                           median_mixed=median(r["mixed"] for r in subset),
                           median_observed=median(r["split_half_spearman"] for r in eligible) if eligible else None,
                           median_null=median(r["null_median"] for r in eligible if r["null_median"] is not None)
                           if eligible else None,
                           rounds_above_null95=sum(r["null_high"] is not None and
                                                   r["split_half_spearman"] > r["null_high"] for r in eligible)))
    write_csv("author-split-half-stages.csv", stages)
    return stages


def first_difference(a, b):
    return next((i for i, pair in enumerate(itertools.zip_longest(a, b), 1) if pair[0] != pair[1]), None)


def action_paths(upstream):
    trainer = load_upstream(upstream)
    tasks = {t.id: t for split in ("train", "eval") for t in
             trainer.SudokuEnv(upstream / f"data/sudoku/{split}.jsonl").tasks}
    root = ROOT / "runs/observable-es-01"
    manifest = json.loads((root / "manifest.json").read_text())
    paths = defaultdict(dict)
    for path in sorted(root.glob("*-2026*.json")):
        record = json.loads(path.read_text())
        verification_rows(record, {t: tasks[t] for t in manifest["tasks"][record["split"]]})
        for score in record["scores"]:
            key = (record["split"], score["task_id"], record["rollout_seed"])
            policy = record["policy"]
            assert policy not in paths[key]
            paths[key][policy] = tuple((turn["valid"], turn["action"]) for turn in score["turns"])
    assert len(paths) == 16 and sum(map(len, paths.values())) == 88
    rows = []
    for (split, task, seed), by_policy in sorted(paths.items()):
        for a, b in itertools.combinations(sorted(by_policy), 2):
            changed = first_difference(by_policy[a], by_policy[b])
            kind = ("updated-vs-center" if a == "center" and b == "updated" else
                    "center-vs-candidate" if a == "center" else "candidate-vs-candidate")
            if "updated" in (a, b) and kind != "updated-vs-center":
                continue
            rows.append(dict(split=split, task=task, mask=tasks[task].mask_count, rollout_seed=seed,
                             comparison=kind, a=a, b=b, first_difference=changed,
                             length_a=len(by_policy[a]), length_b=len(by_policy[b])))
    write_csv("local-action-divergence.csv", rows)
    summary = []
    for kind in sorted({r["comparison"] for r in rows}):
        group = [r for r in rows if r["comparison"] == kind]
        first = [r["first_difference"] for r in group if r["first_difference"] is not None]
        summary.append(dict(comparison=kind, pairs=len(group), identical=len(group)-len(first),
                            diverged=len(first), first_step=Counter(first)[1],
                            median_first_difference=median(first) if first else None))
    write_csv("local-action-divergence-summary.csv", summary)
    return summary


def audit_run(upstream, run):
    trainer = load_upstream(upstream)
    boards = json.loads((run / "boards.json").read_text())
    protocol = json.loads((run / "protocol.json").read_text())
    prediction_path = run / "prediction.json"
    prediction = json.loads(prediction_path.read_text())
    prediction_hash = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    summary = json.loads((run / "summary.json").read_text())
    rows, seconds = {}, Counter()
    for split, cases in boards.items():
        masks = prediction["masks"] if split != "calibration" else (5, 10, 15)
        tasks = [trainer.SudokuTask(**case) for case in cases if case["mask_count"] in masks]
        policies = {"center": None, **protocol["parent_manifest"]["directions"]}
        phase_rows = []
        for policy, direction_seed in policies.items():
            for seed in protocol["sample_seeds"][split]:
                for i in range(0, len(tasks), 2):
                    selected = tasks[i:i+2]
                    path = run / f"{split}-{policy}-{seed}-{selected[0].id}.json"
                    record = json.loads(path.read_text())
                    assert (record["split"], record["policy"], record["rollout_seed"], record["batch_seed"],
                            record["direction_seed"]) == (split, policy, seed, seed+i*100, direction_seed)
                    assert record["weight_sha256"] == protocol["weight_sha256"][policy]
                    if split == "confirmation":
                        assert record["prediction_sha256"] == prediction_hash
                        assert path.stat().st_mtime_ns >= prediction_path.stat().st_mtime_ns
                    phase_rows.extend(verification_rows(record, {t.id: t for t in selected}))
                    seconds[split] += record["seconds"]
        rows[split] = phase_rows
        assert len(phase_rows) == summary["rolls_by_phase"][split]
    assert summary["prediction_sha256"] == prediction_hash and summary["prediction"] == prediction
    assert prediction["discovery_sha256"] == hashlib.sha256(json.dumps(rows["discovery"]).encode()).hexdigest()
    successes = Counter((r["mask_count"], r["task"]) for r in rows["calibration"] if r["score"])
    masks = [mask for mask in MASKS if any(0 < n < len(policies)
                                          for (group, _), n in successes.items() if group == mask)][:2]
    assert masks == prediction["masks"]
    chosen = select_prediction(rows["discovery"], masks, list(protocol["parent_manifest"]["directions"]))
    assert json.loads(json.dumps(chosen)) == {key: prediction[key] for key in chosen}
    table = score_table(rows["confirmation"])
    contrasts = [paired(table, r["a"], r["b"], r["mask"]) for r in prediction["discovery"]]
    assert json.loads(json.dumps(contrasts)) == summary["confirmation"]
    matches = bool(contrasts) and all(r["net"] > 0 for r in contrasts)
    stronger = bool(contrasts) and all(r["more_boards"] - r["fewer_boards"] >= 2
                                       and min(r["by_seed"].values()) > 0 for r in contrasts)
    assert matches == summary["directional_match"] and stronger == summary["stronger_replication"]
    all_rows = list(itertools.chain.from_iterable(rows.values()))
    with (run / "scores.csv").open(newline="") as handle:
        expected = [{key: "" if val is None else str(val) for key, val in row.items()} for row in all_rows]
        assert list(csv.DictReader(handle)) == expected
    print(json.dumps(dict(rollouts=summary["rolls_by_phase"], seconds=dict(seconds), prediction=prediction,
                          confirmation=contrasts, directional_match=summary["directional_match"],
                          stronger_replication=summary["stronger_replication"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-run", help="Recompute a completed GPU run from its raw trajectories")
    args = parser.parse_args()
    upstream = upstream_directory()
    if args.audit_run:
        audit_run(upstream, ROOT / "runs" / args.audit_run)
    else:
        print(json.dumps({"author": author_reliability(upstream), "actions": action_paths(upstream)}, indent=2))
