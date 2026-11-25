"""Read paired WebShop output and summarize the first behavioral differences."""

import argparse
from collections import Counter
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def action_type(change):
    if change is None:
        return "none"
    effects = (change["original_effect"], change["perturbed_effect"])
    if not all(effects):
        return "length"
    kinds = sorted({side["kind"] for side in effects})
    return "+".join(kinds)


def describe(name, pair):
    change = pair["first_different_action"]
    print(f"{name}, task {pair['goal_id']}, step {change['action_index']}:")
    for label, effect in (("baseline", change["original_effect"]), ("candidate", change["perturbed_effect"])):
        print(" ", label, "ended" if effect is None else (
            effect["action"], effect["after"]["page_type"], effect["after"]["result_asins"],
            effect["after"]["asin"], effect["after"]["options"],
        ))
    print("  purchases:", pair["original"]["purchase"], pair["candidate"]["purchase"])
    print("  final scores:", pair["original_score"], pair["candidate_score"])
    for label, episode in (("baseline", pair["original"]), ("candidate", pair["candidate"])):
        opportunities = episode["offline_purchase_scores"]
        print(" ", label, "best item seen:", max(opportunities, key=lambda row: row["score"], default=None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    baseline = read(args.output / "baseline-summary.json")
    print("Baseline:", baseline)
    baseline_actions = Counter()
    for path in (args.output / "baseline").glob("*.jsonl"):
        with path.open(encoding="utf-8") as stream:
            baseline_actions.update(row["kind"] for row in map(json.loads, stream) if row["kind"])
    print("Baseline decision counts by action type:", dict(baseline_actions))
    paired = [(path.parent.name, read(path)) for path in sorted(args.output.glob("seed-*/paired.json"))]
    counts = Counter()
    same_score, different_score = [], []
    for name, rows in paired:
        for row in rows:
            change = row["first_different_action"]
            counts[(action_type(change), "same" if row["delta"] == 0 else "up" if row["delta"] > 0 else "down")] += 1
            if change:
                (same_score if row["delta"] == 0 else different_score).append((name, row))
    print("First differing action type and final score direction across all completed pairs:", dict(counts))
    if same_score:
        describe(*max(same_score, key=lambda item: item[1]["first_different_action"]["changed_page_product_or_options"]))
    if different_score:
        describe(*max(different_score, key=lambda item: abs(item[1]["delta"])))


if __name__ == "__main__":
    main()
