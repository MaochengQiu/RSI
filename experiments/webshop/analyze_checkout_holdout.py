"""Audit held-out actions, checkout timing, and checkout on matched baseline prompts."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean

from analyze_holdout import bootstrap, load, read_outcome


def compare(left, right, seed):
    assert left["goal_id"] == right["goal_id"] and left["first_state"] == right["first_state"]
    row = {"seed": seed, "goal_id": left["goal_id"], "actions_changed": left["actions"] != right["actions"],
           "delta": right["reward"] - left["reward"]}
    for key in ("reward", "bought", "calls", "rounds", "peak_seen", "input_tokens"):
        row["left_" + key], row["right_" + key] = left[key], right[key]
    for a, b in zip(left["trace_rows"], right["trace_rows"]):
        if a["prompt"] != b["prompt"] or a["before"] != b["before"]:
            row["first_event"] = "context_changed"
            break
        if a["action"] != b["action"]:
            row.update(first_round=a["round"], left_action=a["action"], right_action=b["action"],
                       left_kind=a["kind"], right_kind=b["kind"])
            if a["bought"] != b["bought"]:
                row["first_event"] = "right_buys" if b["bought"] else "right_continues"
                row["immediate_score"] = b["reward"] if b["bought"] else a["reward"]
            else:
                row["first_event"] = "other_action"
            break
    else:
        row["first_event"] = "same"
        assert not row["actions_changed"]
    return row


def summarize(rows):
    by_id = defaultdict(list)
    events = defaultdict(Counter)
    for r in rows:
        by_id[r["goal_id"]].append(r["delta"])
        outcome = "up" if r["delta"] > 1e-9 else "down" if r["delta"] < -1e-9 else "equal"
        events[r["first_event"]][outcome] += 1
    task_deltas = [mean(v) for v in by_id.values()]
    return {"pairs": len(rows), "actions_changed": sum(r["actions_changed"] for r in rows),
            "up": sum(r["delta"] > 1e-9 for r in rows), "down": sum(r["delta"] < -1e-9 for r in rows),
            "delta": mean(task_deltas), "task_bootstrap_95": bootstrap(task_deltas),
            "task_gains": sum(v > 1e-9 for v in task_deltas), "task_losses": sum(v < -1e-9 for v in task_deltas),
            "first_event_by_outcome": events,
            "first_events_by_seed": {s: dict(Counter(r["first_event"] for r in rows if r["seed"] == s))
                                     for s in sorted({r["seed"] for r in rows})},
            "first_action_types": dict(Counter(f"{r.get('left_kind')} -> {r.get('right_kind')}" for r in rows if "first_round" in r)),
            "by_seed_delta": {s: mean(r["delta"] for r in rows if r["seed"] == s) for s in sorted({r["seed"] for r in rows})},
            "purchase_gain": sum(r["right_bought"] and not r["left_bought"] for r in rows),
            "purchase_loss": sum(r["left_bought"] and not r["right_bought"] for r in rows),
            "full_gain": sum(r["right_reward"] == 1 and r["left_reward"] < 1 for r in rows),
            "full_loss": sum(r["left_reward"] == 1 and r["right_reward"] < 1 for r in rows),
            "means": {k: mean(r[k] for r in rows) for k in ("left_reward", "right_reward", "left_calls", "right_calls",
                                                           "left_rounds", "right_rounds", "left_input_tokens", "right_input_tokens")},
            "totals": {s: {"bought": sum(r[s + "_bought"] for r in rows),
                           "full": sum(r[s + "_reward"] == 1 for r in rows),
                           "reached_full": sum(r[s + "_peak_seen"] == 1 for r in rows)}
                       for s in ("left", "right")}}


def probe_pairs(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        keys = ("goal_id", "round", "prompt_sha256", "buy_now_score", "asin", "options")
        assert all(a[k] == b[k] for k in keys)
        yield {"goal_id": a["goal_id"], "round": a["round"], "score": a["buy_now_score"],
               "left_buy": a["buy"], "right_buy": b["buy"]}


def summarize_probes(probes):
    result = {}
    for name, group in (("full", [p for p in probes if p["score"] == 1]),
                        ("partial", [p for p in probes if 0 < p["score"] < 1]),
                        ("zero", [p for p in probes if p["score"] == 0])):
        result[name] = {"pairs": len(group), "distinct_prompts": len({(p["goal_id"], p["round"]) for p in group}),
                        "left_buy": sum(p["left_buy"] for p in group), "right_buy": sum(p["right_buy"] for p in group),
                        "to_buy": sum(not p["left_buy"] and p["right_buy"] for p in group),
                        "to_continue": sum(p["left_buy"] and not p["right_buy"] for p in group)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = load(args.results / "manifest.json")
    ids = [t["goal_id"] for t in manifest["tasks"]]
    assert len(ids) == len(set(ids)) == len({t["goal"]["asin"] for t in manifest["tasks"]})
    assert set(ids).isdisjoint(manifest["excluded_goal_ids"])
    episodes, probes, rows, replay_pairs = {}, {}, defaultdict(list), defaultdict(list)
    seeds = sorted({c["seed"] for c in manifest["conditions"] if "seed" in c})
    for condition in manifest["conditions"]:
        name = condition["name"]
        folder = args.results / name
        assert all(v["max_abs"] == 0 for v in load(folder / "restored.json").values())
        episodes[name] = [read_outcome(folder, i) for i in ids]
        probes[name] = load(folder / "checkout-probes.json")
    for seed in seeds:
        low, high = f"seed-{seed}-scale-0.05", f"seed-{seed}-sigma-0.003"
        for contrast, left, right in (("low_vs_base", "baseline", low), ("high_vs_base", "baseline", high),
                                      ("high_vs_low", low, high)):
            rows[contrast].extend(compare(a, b, seed) for a, b in zip(episodes[left], episodes[right]))
            replay_pairs[contrast].extend(probe_pairs(probes[left], probes[right]))
    report = {"n_tasks": len(ids), "episodes": sum(map(len, episodes.values())),
              "seconds_in_episodes": sum(r["elapsed_seconds"] for runs in episodes.values() for r in runs),
              "peak_allocated_mib": max(r["peak_allocated_mib"] for runs in episodes.values() for r in runs),
              "contrasts": {k: summarize(v) for k, v in rows.items()},
              "fixed_prompt_checkout": {k: summarize_probes(v) for k, v in replay_pairs.items()}}
    args.output.mkdir(exist_ok=True, parents=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output / "paired.csv").open("w", encoding="utf-8-sig", newline="") as f:
        flat = [dict(contrast=k, **r) for k, vals in rows.items() for r in vals]
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in flat for k in r)))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
