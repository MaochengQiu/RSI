"""Combine all four strengths, verifying paired results against the step logs."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked_episode(folder, goal_id):
    path = folder / f"{goal_id:04d}.json"
    outcome = read(path)
    turns = [json.loads(line) for line in path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    assert outcome["goal_id"] == goal_id and outcome["rounds"] == len(turns)
    assert outcome["actions"] == [t["action"] for t in turns if t["action"]]
    assert outcome["first_state"] == turns[0]["before"]
    assert outcome["reward"] == turns[-1]["reward"] and outcome["bought"] == turns[-1]["bought"]
    return outcome


def summarize(rows):
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["goal_id"]].append(row["delta"])
    task_deltas = {i: mean(values) for i, values in by_task.items()}
    return dict(n=len(rows), changed=sum(r["actions_changed"] for r in rows),
                up=sum(r["delta"] > 0 for r in rows), down=sum(r["delta"] < 0 for r in rows),
                distinct_goals_changed=len({r["goal_id"] for r in rows if r["actions_changed"]}),
                goals_with_mean_gain=sum(d > 1e-10 for d in task_deltas.values()),
                goals_with_mean_loss=sum(d < -1e-10 for d in task_deltas.values()),
                net_delta_by_goal=task_deltas,
                baseline_mean=mean(r["baseline_score"] for r in rows),
                mean_score=mean(r["score"] for r in rows), mean_delta=mean(r["delta"] for r in rows),
                full_gained=sum(r["score"] == 1 and r["baseline_score"] < 1 for r in rows),
                full_lost=sum(r["score"] < 1 and r["baseline_score"] == 1 for r in rows),
                mean_actions=mean(r["actions"] for r in rows), mean_rounds=mean(r["rounds"] for r in rows),
                seconds=sum(r["seconds"] for r in rows), peak_allocated_mib=max(r["peak_mib"] for r in rows),
                peak_reserved_mib=max(r["reserved_mib"] for r in rows),
                first_change_types=dict(Counter(
                    (r["first_original_kind"] or "end") + " -> " + (r["first_candidate_kind"] or "end")
                    for r in rows if r["actions_changed"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot-dir", "strength-dir", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    manifest = read(args.strength_dir / "manifest.json")
    original = {i: checked_episode(args.pilot_dir / "baseline", i) for i in manifest["goal_ids"]}
    strengths = defaultdict(list)
    candidates = {}
    for seed in manifest["seeds"]:
        for label in ("scale-0.01", "scale-0.05", "sigma-0.002", "sigma-0.003"):
            name = f"seed-{seed}-{label}"
            root = args.pilot_dir if label.startswith("scale") else args.strength_dir
            folder = root / name
            pairs = read(folder / "paired.json")
            assert [p["goal_id"] for p in pairs] == manifest["goal_ids"]
            if root == args.strength_dir:
                restoration = read(folder / "restored-forward.json")
                assert all(v["max_abs"] == 0 for v in restoration.values())
                assert read(folder / "weights.json")["forward"]["layer_output"]["rms"] > 0
            rows = []
            for pair in pairs:
                goal_id = pair["goal_id"]
                base, candidate = original[goal_id], checked_episode(folder, goal_id)
                assert base == pair["original"] and candidate == pair["candidate"]
                assert base["first_state"] == candidate["first_state"]
                diff = pair["first_different_action"]
                changed = base["actions"] != candidate["actions"]
                assert changed == bool(diff)
                delta = candidate["reward"] - base["reward"]
                assert abs(delta - pair["delta"]) < 1e-10
                left = (diff["original_effect"] or {}) if diff else {}
                right = (diff["perturbed_effect"] or {}) if diff else {}
                rows.append(dict(strength=label, seed=seed, goal_id=goal_id, actions_changed=changed,
                                 baseline_score=base["reward"], score=candidate["reward"], delta=delta,
                                 first_action_index=diff["action_index"] if diff else None,
                                 first_original_kind=left.get("kind"), first_original_action=left.get("action"),
                                 first_candidate_kind=right.get("kind"), first_candidate_action=right.get("action"),
                                 bought=candidate["bought"], rounds=candidate["rounds"], actions=len(candidate["actions"]),
                                 seconds=candidate["elapsed_seconds"], peak_mib=candidate["peak_allocated_mib"],
                                 reserved_mib=candidate["peak_reserved_mib"]))
            candidates[name] = summarize(rows)
            strengths[label].extend(rows)
    report = {"baseline_mean": mean(r["reward"] for r in original.values()),
              "baseline_actions": mean(len(r["actions"]) for r in original.values()),
              "by_strength": {label: summarize(rows) for label, rows in strengths.items()},
              "by_candidate": candidates}
    args.output.mkdir(exist_ok=True, parents=True)
    (args.output / "strength-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows = [row for group in strengths.values() for row in group]
    with (args.output / "strength-pairs.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for name, values in report["by_strength"].items():
        print(name, values["changed"], values["up"], values["down"], values["mean_delta"])


if __name__ == "__main__":
    main()
