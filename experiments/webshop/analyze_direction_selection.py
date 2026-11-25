"""Recompute candidate ranking and paired generalization results from raw traces."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import mean

from analyze_holdout import bootstrap, load, read_outcome
from analyze_checkout_holdout import compare


def metrics(rows):
    pairs = len(rows)
    deltas = [r["delta"] for r in rows]
    return {"n": pairs, "left_mean": mean(r["left_reward"] for r in rows),
            "right_mean": mean(r["right_reward"] for r in rows), "mean_delta": mean(deltas),
            "paired_bootstrap_95": bootstrap(deltas),
            "up": sum(d > 1e-9 for d in deltas), "down": sum(d < -1e-9 for d in deltas),
            "unchanged_score": sum(abs(d) <= 1e-9 for d in deltas),
            "changed_actions": sum(r["actions_changed"] for r in rows),
            "new_full": sum(r["left_reward"] < 1 and r["right_reward"] == 1 for r in rows),
            "lost_full": sum(r["left_reward"] == 1 and r["right_reward"] < 1 for r in rows),
            "new_purchase": sum(r["right_bought"] and not r["left_bought"] for r in rows),
            "lost_purchase": sum(r["left_bought"] and not r["right_bought"] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    m, selection = load(root / "manifest.json"), load(root / "selection.json")
    seeds = m["seeds"]
    winner, control = selection["winner"], selection["control"]
    assert len(seeds) == len(set(seeds)) == 8 and control == m["control_seed"] == seeds[0]
    assert m["weight_hashes"][str(winner)] == selection["winner_sha256"]
    all_tasks = sum(m["splits"].values(), [])
    assert len({t["goal_id"] for t in all_tasks}) == len({t["goal"]["asin"] for t in all_tasks}) == len(all_tasks)
    summaries, all_pairs = {}, []
    wall = peak = 0.0
    for split, tasks in m["splits"].items():
        candidates = seeds if split == "development" else list(dict.fromkeys([winner, control]))
        folder = root / split
        assert {p.name for p in folder.iterdir() if p.is_dir()} == {"baseline", *map(str, candidates)}
        conditions = {}
        split_summaries = {}
        for name in ["baseline", *map(str, candidates)]:
            condition_dir = folder / name
            restored = load(condition_dir / "restored.json")
            forward = load(condition_dir / "forward.json")
            assert all(v["max_abs"] == 0 for v in restored.values())
            assert (forward["layer_output"]["rms"] > 0) == (name != "baseline")
            data = [read_outcome(condition_dir, t["goal_id"]) for t in tasks]
            conditions[name] = data
            wall += sum(r["elapsed_seconds"] for r in data)
            peak = max(peak, max(r["peak_allocated_mib"] for r in data))
            split_summaries[name] = {"mean": mean(r["reward"] for r in data), "full": sum(r["reward"] == 1 for r in data),
                                     "bought": sum(r["bought"] for r in data), "calls": mean(r["calls"] for r in data),
                                     "rounds": mean(r["rounds"] for r in data), "input_tokens": mean(r["input_tokens"] for r in data)}
        comparisons = [(str(s), "baseline") for s in candidates]
        if split == "development":
            rank = sorted(seeds, key=lambda s: -round(split_summaries[str(s)]["mean"], 12))
            assert rank[0] == winner and all(split_summaries[str(s)]["mean"] == selection["development_means"][str(s)] for s in seeds)
            assert selection["baseline_mean"] == split_summaries["baseline"]["mean"]
            # Descriptive sensitivity of the original ranking, with test results excluded.
            omission_winners = {}
            for index, t in enumerate(tasks):
                omission_winners[t["goal_id"]] = max(seeds, key=lambda s: round(mean(
                    r["reward"] for i, r in enumerate(conditions[str(s)]) if i != index), 12))
        else:
            comparisons.append((str(winner), str(control)))
        pair_stats = {}
        for right, left in comparisons:
            rows = [compare(a, b, int(right)) for a,b in zip(conditions[left], conditions[right])]
            all_pairs.extend(dict(split=split, reference=left, candidate=right, **r) for r in rows)
            pair_stats[f"{right}-vs-{left}"] = metrics(rows)
        summaries[split] = {"conditions": split_summaries, "pairs": pair_stats}
    report = {"winner": winner, "control": control, "winner_sha256": selection["winner_sha256"], "development_ranking": rank,
              "dev_leave_one_out_winners": omission_winners, "dev_leave_one_out_counts": dict(Counter(omission_winners.values())),
              "episode_count": sum(len(m["splits"][s]) * len(summaries[s]["conditions"]) for s in m["splits"]),
              "gpu_episode_seconds": wall, "peak_allocated_mib": peak, **summaries}
    args.output.mkdir(exist_ok=True, parents=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output / "paired.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in all_pairs for k in r)))
        writer.writeheader()
        writer.writerows(all_pairs)
    print("winner", winner, "control", control, "episodes", report["episode_count"])
    print(json.dumps(summaries["test"], indent=2))


if __name__ == "__main__":
    main()
