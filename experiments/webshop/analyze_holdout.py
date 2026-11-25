"""Audit completed WebShop holdout traces and report paired outcomes."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from statistics import mean


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_outcome(folder, goal_id):
    path = folder / f"{goal_id:04d}.json"
    result = load(path)
    with path.with_suffix(".jsonl").open(encoding="utf-8") as stream:
        trace = list(map(json.loads, stream))
    assert result["goal_id"] == goal_id and result["rounds"] == len(trace)
    assert result["actions"] == [row["action"] for row in trace if row["action"]]
    assert result["reward"] == trace[-1]["reward"] and result["bought"] == trace[-1]["bought"]
    seen = result["offline_purchase_scores"]
    scores_by_round = {state["round"]: state["score"] for state in seen}
    if result["bought"]:
        assert scores_by_round[len(trace)] == result["reward"]
    first_full = next((state["round"] for state in seen if state["score"] == 1), None)
    best = max((state["score"] for state in seen), default=0.0)
    result.update({
        "trace_rows": trace,
        "calls": len(result["actions"]),
        "search_calls": sum(row["kind"] == "search" for row in trace),
        "input_tokens": sum(row["input_tokens"] for row in trace),
        "peak_seen": best,
        "found_full_without_purchase": first_full is not None and not result["bought"],
        "bought_below_seen": result["bought"] and result["reward"] < best,
        "full_checkout_delay": len(trace) - first_full if result["reward"] == 1 else None,
    })
    return result


def summarize(rows):
    timing = [r["full_checkout_delay"] for r in rows if r["full_checkout_delay"] is not None]
    return {
        "mean_score": mean(r["reward"] for r in rows),
        "bought": sum(r["bought"] for r in rows),
        "positive": sum(r["reward"] > 0 for r in rows),
        "full": sum(r["reward"] == 1 for r in rows),
        "found_full_without_purchase": [r["goal_id"] for r in rows if r["found_full_without_purchase"]],
        "bought_below_seen": [r["goal_id"] for r in rows if r["bought_below_seen"]],
        "mean_full_checkout_delay": mean(timing) if timing else None,
        "cost_means": {key: mean(r[key] for r in rows)
                       for key in ("rounds", "calls", "search_calls", "input_tokens", "elapsed_seconds")},
        "peak_allocated_mib": max(r["peak_allocated_mib"] for r in rows),
    }


def bootstrap(differences):
    rng = random.Random(20261123)
    resample = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(20000))
    return [resample[500], resample[19499]]


def pair(original, candidate):
    assert original["first_state"] == candidate["first_state"]
    checkout = None
    for left, right in zip(original["trace_rows"], candidate["trace_rows"]):
        if left["prompt"] != right["prompt"]:
            break
        if left["action"] == right["action"]:
            continue
        if (left["bought"] != right["bought"] and left["before"] == right["before"]
                and original["bought"] and candidate["bought"]):
            buyer, later = ("original", candidate) if left["bought"] else ("candidate", original)
            immediate = left["reward"] if left["bought"] else right["reward"]
            if later["reward"] > immediate:
                checkout = {
                    "earlier_buyer": buyer, "round": left["round"], "immediate_score": immediate,
                    "later_score": later["reward"], "initial_asin": left["before"]["asin"],
                }
        break
    return {
        "goal_id": original["goal_id"],
        "score_original": original["reward"], "score_candidate": candidate["reward"],
        "delta": candidate["reward"] - original["reward"],
        "bought_original": original["bought"], "bought_candidate": candidate["bought"],
        "best_seen_original": original["peak_seen"], "best_seen_candidate": candidate["peak_seen"],
        "rounds_original": original["rounds"], "rounds_candidate": candidate["rounds"],
        "full_delay_original": original["full_checkout_delay"],
        "full_delay_candidate": candidate["full_checkout_delay"],
        "early_checkout_followed_by_better_purchase": checkout,
    }


def transitions(pairs, attribute, predicate):
    counts = Counter((predicate(p[f"{attribute}_original"]), predicate(p[f"{attribute}_candidate"])) for p in pairs)
    return {"gained": counts[False, True], "lost": counts[True, False]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    manifest = load(args.results / "manifest.json")
    ids = [task["goal_id"] for task in manifest["tasks"]]
    assert len(ids) == len(set(ids)) == 40
    assert set(ids).isdisjoint(manifest["excluded_goal_ids"])
    assert set(task["goal"]["asin"] for task in manifest["tasks"]).isdisjoint(manifest["excluded_target_asins"])
    baseline = [read_outcome(args.results / "baseline", i) for i in ids]
    baseline_summary = summarize(baseline)
    report = {"n": len(ids), "baseline": baseline_summary, "candidates": {}}
    for condition in manifest["candidates"]:
        name = condition["name"]
        candidates = [read_outcome(args.results / name, i) for i in ids]
        paired = [pair(a, b) for a, b in zip(baseline, candidates)]
        deltas = [row["delta"] for row in paired]
        summary = summarize(candidates)
        shared_full_delays = [p["full_delay_candidate"] - p["full_delay_original"] for p in paired
                              if p["full_delay_candidate"] is not None and p["full_delay_original"] is not None]
        report["candidates"][name] = {
            **summary, "mean_delta": mean(deltas), "bootstrap_95_interval": bootstrap(deltas),
            "up": sum(delta > 0 for delta in deltas), "down": sum(delta < 0 for delta in deltas),
            "purchase_transitions": transitions(paired, "bought", bool),
            "positive_transitions": transitions(paired, "score", lambda score: score > 0),
            "full_transitions": transitions(paired, "score", lambda score: score == 1),
            "shared_full_n": len(shared_full_delays),
            "shared_full_checkout_delay_delta": mean(shared_full_delays) if shared_full_delays else None,
            "cost_delta": {key: value - baseline_summary["cost_means"][key]
                           for key, value in summary["cost_means"].items()},
            "pairs": paired,
        }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for value in report["candidates"].values():
        value["pairs"] = [p for p in value["pairs"] if p["delta"] != 0]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
