"""Run inside the extracted 84-task supplement: python recheck_research_pool.py"""

import csv
import json
from collections import Counter
from math import comb, sqrt
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent


def read(name):
    with (ROOT / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write(name, rows):
    with (ROOT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    matrix = read("84-scores.csv")
    seeds = list(matrix[0])[2:]
    assert len(seeds) == 8 and len({r["goal_id"] for r in matrix}) == len(matrix) == 84
    hits = []
    for s in seeds:
        delta = [float(r[s]) - float(r["baseline"]) for r in matrix]
        hits.append(dict(seed=s, up=sum(d > 1e-12 for d in delta), down=sum(d < -1e-12 for d in delta),
                         only_improvement=sum(float(r[s]) > float(r["baseline"]) + 1e-12 and
                                              all(float(r[t]) <= float(r["baseline"]) + 1e-12 for t in seeds if t != s)
                                              for r in matrix), mean_delta=mean(delta)))
    write("direction-hits.csv", hits)
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    dev, test = [[r["goal_id"] for r in manifest["splits"][s]] for s in ("development", "test")]
    assert len(set(dev)) == 24 and len(set(test)) == 60 and not set(dev).intersection(test)
    by_goal = {int(r["goal_id"]): r for r in matrix}
    ranked = sorted(seeds, key=lambda s: -round(mean(float(by_goal[g][s]) for g in dev), 12))
    pairs = []
    for g in dev + test:
        split = "development" if g in dev else "test"
        terminals = {}
        rewards = {}
        for s in ["baseline", *seeds]:
            path = ROOT / "episodes" / split / s / f"{g:04d}.json"
            ep = json.loads(path.read_text(encoding="utf-8"))
            assert ep["goal_id"] == g and ep["reward"] == float(by_goal[g][s])
            p = ep["purchase"]
            key = (p["asin"], tuple(sorted(p["options"].items()))) if ep["bought"] else ("no purchase",)
            assert ep["reward"] == rewards.setdefault(key, ep["reward"])
            terminals[s] = key
        for k in (1, 3, 5, 8):
            selected = sorted(ranked[:k], key=int)
            count = Counter(terminals[s] for s in selected)
            choice = max(count, key=count.get)
            b = float(by_goal[g]["baseline"])
            pairs.append(dict(split=split, goal_id=g, k=k, baseline=b, vote=rewards[choice],
                              oracle=max(float(by_goal[g][s]) for s in selected)))
    write("topk-pairs.csv", pairs)
    for k in (1, 3, 5, 8):
        rows = [r for r in pairs if r["split"] == "test" and r["k"] == k]
        print("K, vote gain:", k, mean(r["vote"] - r["baseline"] for r in rows))
    print("84-task oracle gain:", mean(r["oracle"] - r["baseline"] for r in pairs if r["k"] == 8))
    rows = read("sealed-60-pairs.csv")
    assert len(rows) == len({r["goal_id"] for r in rows}) == 60
    ds = [float(r["right_reward"]) - float(r["left_reward"]) for r in rows]
    changed = [d for r, d in zip(rows, ds) if r["actions_changed"] == "True"]
    up = sum(d > 1e-12 for d in ds)
    down = sum(d < -1e-12 for d in ds)
    p = min(1, 2 * sum(comb(up + down, i) for i in range(min(up, down) + 1)) / 2**(up + down))
    se = stdev(ds) / sqrt(len(ds))
    print("sealed mean, SD, SE, approximate 80% MDE:", mean(ds), stdev(ds), se, 2.802 * se)
    print("changed n, mean, SD, SE:", len(changed), mean(changed), stdev(changed), stdev(changed)/sqrt(len(changed)))
    print("two-sided sign p, half mean absolute difference, oracle gain:",
          p, mean(map(abs, ds))/2, mean(max(0.0, d) for d in ds))


if __name__ == "__main__":
    main()
