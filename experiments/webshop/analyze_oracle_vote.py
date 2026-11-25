"""Offline WebShop complementarity diagnostic. Run with the WebShop output root as argument."""

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def terminal(episode):
    if not episode["bought"]:
        return ("no purchase",)
    purchase = episode["purchase"]
    return purchase["asin"], tuple(sorted(purchase["options"].items()))


def compare(scope, goal, base, episodes):
    keys = [terminal(e) for e in episodes]
    scores = [e["reward"] for e in episodes]
    assert all(e["goal_id"] == goal for e in episodes)
    counts = Counter(keys)
    top = max(counts.values())
    modes = [key for key in counts if counts[key] == top]
    for key in counts:
        assert len({s for k, s in zip(keys, scores) if k == key}) == 1
    b = base["reward"]
    return dict(scope=scope, goal_id=goal, baseline=b, oracle=max(scores),
                oracle_with_base=max(b, *scores), vote=scores[keys.index(modes[0])],
                modal_ties=len(modes), top_votes=top, better_directions=sum(s>b+1e-12 for s in scores),
                vote_tie_low=min(scores[keys.index(key)] for key in modes),
                vote_tie_high=max(scores[keys.index(key)] for key in modes))


def run(root, output):
    archive = root / "direction-selection-01"
    follow = root / "randopt-followup-01"
    manifest = load(archive / "manifest.json")
    with (follow / "analysis/84-scores.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    matrix = {int(r["goal_id"]): r for r in rows}
    assert len(matrix) == len(rows) == 84
    scopes = {"single": sorted(map(int, manifest["seeds"]))}
    roots = {"multi": follow / "scope", "global": root / "randopt-global-01"}
    for scope, folder in roots.items():
        scopes[scope] = sorted(int(p.parent.name) for p in folder.glob("*/calibration.json")
                               if load(p)["matched"])
    assert [len(v) for v in scopes.values()] == [8, 8, 7]
    results = []
    for split in ("development", "test"):
        for task in manifest["splits"][split]:
            goal = task["goal_id"]
            name = f"{goal:04d}.json"
            base = load(archive / split / "baseline" / name)
            assert base["goal_id"] == goal and base["reward"] == float(matrix[goal]["baseline"])
            for scope, seeds in scopes.items():
                if split == "test" and scope != "single":
                    continue
                episodes = []
                for seed in seeds:
                    if scope == "single":
                        folder = archive / split / str(seed)
                        if split == "test" and not folder.exists():
                            folder = follow / "completed" / str(seed)
                    else:
                        folder = roots[scope] / str(seed)
                    ep = load(folder / name)
                    if scope == "single":
                        assert ep["reward"] == float(matrix[goal][str(seed)])
                    episodes.append(ep)
                results.append(dict(split=split, **compare(scope, goal, base, episodes)))
    for scope, n in (("single", 84), ("multi", 24), ("global", 24)):
        group = [r for r in results if r["scope"] == scope]
        assert len({r["goal_id"] for r in group}) == len(group) == n
        for key in ("baseline", "oracle", "oracle_with_base", "vote", "vote_tie_low", "vote_tie_high"):
            diffs = [r[key] - r["baseline"] for r in group]
            print(scope, key, round(mean(r[key] for r in group), 6), round(mean(diffs), 6),
                  "up/down/full", sum(d > 1e-12 for d in diffs), sum(d < -1e-12 for d in diffs),
                  sum(r[key] == 1 for r in group))
        print("number improving:", sorted(Counter(r["better_directions"] for r in group).items()),
              "tied modes:", sum(r["modal_ties"] > 1 for r in group))
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
