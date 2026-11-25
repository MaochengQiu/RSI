"""Audit archived/new WebShop episodes and apply the registered statistics."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from analyze_checkout_holdout import compare
from analyze_holdout import load, read_outcome

SEED = 2026112103
DELTA = 0.03
HERE = Path(__file__).resolve().parent


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def table(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def submeans(data, repetitions):
    rng = np.random.default_rng(SEED)
    return np.concatenate([data[rng.integers(len(data), size=(count, len(data)))].mean(axis=1)
                           for count in [1000] * (repetitions // 1000)])


def intervals(data, reserved=0):
    n, k = data.shape
    estimate = data.mean(axis=0)
    fixed = np.ptp(data, axis=0) == 0
    draws = submeans(data, 100000)
    special = int(fixed.sum()) + reserved
    alpha = 0.025 if special else 0.05
    bounds = np.column_stack((estimate, estimate))
    if (~fixed).any():
        peak = np.abs(draws[:, ~fixed] - estimate[~fixed]).max(axis=1)
        q = np.quantile(peak, 1 - alpha, method="higher")
        bounds[~fixed] += [-q, q]
    if fixed.any():
        p = 1 - (0.025 / special) ** (1 / n)
        bounds[fixed, 0] += p * (-1 - estimate[fixed])
        bounds[fixed, 1] += p * (1 - estimate[fixed])
    return estimate, np.clip(bounds, -1, 1), draws


def decision(data, names, reserved=0):
    if not names:
        return {"directions": {}, "nominee": None, "status": "insufficient"}
    estimate, bounds, draws = intervals(data, reserved)
    eligible = [i for i in range(len(names)) if round(float(estimate[i]), 12) >= DELTA and bounds[i, 0] > 0]
    top = max(eligible, key=lambda i: bounds[i, 0]) if eligible else None
    report = {"directions": {str(name): {"mean": float(m), "ci95": band.tolist()}
                             for name, m, band in zip(names, estimate, bounds)},
              "nominee": names[top] if top is not None else None,
              "status": "validate" if top is not None else
              "stop" if not reserved and (bounds[:, 1] < DELTA).all() else "insufficient"}
    if len(names) == 8 and len(data) == 84:
        p = np.eye(8) - np.ones((8, 8)) / 8
        variance = (estimate @ p @ estimate - np.trace(p @ np.cov(data, rowvar=False)) / len(data)) / 8
        lo, hi = bounds.T
        a, b = lo.min(), hi.max()
        for _ in range(80):
            mid = (a + b) / 2
            if (mid - np.clip(mid, lo, hi)).sum() > 0:
                b = mid
            else:
                a = mid
        vertices = np.array(np.meshgrid(*bounds)).reshape(8, -1).T
        report["mu"] = {"mean": float(estimate.mean()), "ci95": np.quantile(draws.mean(axis=1), [.025, .975]).tolist()}
        report["tau"] = {"estimate": float(np.sqrt(max(0.0, variance))),
                         "ci95": [float(np.clip((a + b) / 2, lo, hi).std()), float(vertices.std(axis=1).max())]}
    return report


def winner(means, allow_baseline=False):
    rounded = np.round(means, 12)
    i = int(rounded.argmax())
    return -1 if allow_baseline and rounded[i] <= 0 else i


def ranks(values):
    values = np.round(values, 12)
    return np.array([(values < v).sum() + ((values == v).sum() + 1) / 2 for v in values])


def simulations(data, names):
    rng = np.random.default_rng(SEED)
    correlation, gains, picks = [], [], Counter()
    matches = 0
    for _ in range(1000):
        idx = rng.permutation(len(data))
        a, b = data[idx[:42]].mean(axis=0), data[idx[42:]].mean(axis=0)
        matches += winner(a) == winner(b)
        ra, rb = ranks(a), ranks(b)
        if ra.std() and rb.std():
            correlation.append(float(np.corrcoef(ra, rb)[0, 1]))
    for _ in range(1000):
        idx = rng.permutation(len(data))
        selected = winner(data[idx[:24]].mean(axis=0), allow_baseline=True)
        picks["baseline" if selected < 0 else str(names[selected])] += 1
        gains.append(0.0 if selected < 0 else float(data[idx[24:], selected].mean()))
    return {"top_agreement": matches / 1000, "spearman_valid": len(correlation),
            "spearman_mean": float(np.mean(correlation)) if correlation else None,
            "spearman_quantiles": np.quantile(correlation, [.05, .5, .95]).tolist() if correlation else [],
            "selection": dict(picks), "evaluation_mean": float(np.mean(gains)),
            "evaluation_quantiles": np.quantile(gains, [.05, .5, .95]).tolist(),
            "negative_rate": float(np.mean(np.array(gains) < -1e-12)), "simulation_gains": gains}


def audit(folder, tasks):
    ids = [t["goal_id"] for t in tasks]
    assert sorted(p.stem for p in folder.glob("????.json")) == sorted(f"{i:04d}" for i in ids)
    return [read_outcome(folder, i) for i in ids]


def score(rows):
    return np.array([row["reward"] for row in rows])


def paired(left, right, seed):
    rows = []
    for a, b in zip(left, right):
        row = compare(a, b, seed)
        row["call_delta"] = b["calls"] - a["calls"]
        rows.append(row)
    return rows


def direction_summary(rows):
    return {"actions_changed": sum(r["actions_changed"] for r in rows),
            "up": sum(r["delta"] > 1e-12 for r in rows), "down": sum(r["delta"] < -1e-12 for r in rows),
            "full_gains": sum(r["left_reward"] < 1 and r["right_reward"] == 1 for r in rows),
            "full_losses": sum(r["right_reward"] < 1 and r["left_reward"] == 1 for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("stage0", "complete", "scope", "final"))
    for name in ("archive", "results", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    archive, results, out = args.archive, args.results, args.output
    protocol = HERE / "RANDOPT-PREREG.zh.md"
    seal_path = HERE / "randopt-final-seal.json"
    repo = HERE.parents[1]
    for path in (protocol, seal_path):
        committed = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{path.relative_to(repo).as_posix()}"])
        assert committed.replace(b"\r\n", b"\n") == path.read_bytes().replace(b"\r\n", b"\n")
    m = load(archive / "manifest.json")
    names = m["seeds"]
    dev, test = m["splits"]["development"], m["splits"]["test"]
    base = audit(archive / "development/baseline", dev)
    candidates = [audit(archive / "development" / str(s), dev) for s in names]
    data = np.column_stack([score(group) - score(base) for group in candidates])
    rows = [row for s, group in zip(names, candidates) for row in paired(base, group, s)]
    if args.stage == "stage0":
        draws = submeans(data, 10000)
        top = Counter(str(names[winner(a)]) for a in draws)
        picks = Counter("baseline" if (j := winner(a, True)) < 0 else str(names[j]) for a in draws)
        leave = Counter(str(names[winner(np.delete(data, i, axis=0).mean(axis=0))]) for i in range(len(data)))
        cross = Counter((r["first_event"], str(np.sign(r["call_delta"])),
                         f"{int(r['left_bought'])}->{int(r['right_bought'])}", str(np.sign(round(r["delta"], 12)))) for r in rows)
        save(out / "stage0.json", {"bootstrap_top": dict(top), "bootstrap_allow_baseline": dict(picks),
                                   "leave_one_out_top": dict(leave),
                                   "cross_counts": [{"event": e, "call_sign": c, "buy_transition": b, "score_sign": s, "n": n}
                                                    for (e, c, b, s), n in sorted(cross.items())]})
        table(out / "stage0-pairs.csv", rows)
    elif args.stage == "complete":
        assert load(results / "replay.json")["pass"]
        base_test = audit(archive / "test/baseline", test)
        candidate_test = []
        for s in names:
            folder = archive / "test" / str(s)
            if not folder.exists():
                folder = results / "completed" / str(s)
                assert load(folder / "weight.json")["weight_sha256"] == m["weight_hashes"][str(s)]
                assert all(x["max_abs"] == 0 for x in load(folder / "restored.json").values())
            group = audit(folder, test)
            rows.extend(paired(base_test, group, s))
            candidate_test.append(group)
        matrix = np.column_stack([np.r_[score(a), score(b)] for a, b in zip([base, *candidates], [base_test, *candidate_test])])
        assert matrix.shape == (84, 9) and len(rows) == 672
        delta = matrix[:, 1:] - matrix[:, :1]
        report = decision(delta, names)
        report["ranking_stability"] = simulations(delta, names)
        save(out / "complete.json", report)
        table(out / "84-scores.csv", [dict(goal_id=t["goal_id"], **dict(zip(["baseline", *map(str, names)], row)))
                                      for t, row in zip(dev + test, matrix)])
    elif args.stage == "scope":
        plan = load(results / "scope/candidates.json")
        eligible, scope_data, scope_rows = [], [], []
        calibration = {}
        for s in plan:
            folder = results / "scope" / str(s)
            fit = load(folder / "calibration.json")
            calibration[str(s)] = {key: fit[key] for key in ("matched", "sigma", "calibration_kl", "check_kl")}
            if not fit["matched"]:
                continue
            group = audit(folder, dev)
            eligible.append(s)
            scope_data.append(score(group) - score(base))
            scope_rows.extend(paired(base, group, s))
        d = decision(np.array(scope_data).T, eligible, 8 - len(eligible))
        d["calibration"] = calibration
        d["episode_count"] = len(scope_rows)
        for s in plan:
            if s not in eligible:
                d["directions"][str(s)] = {"mean": None, "ci95": [-1, 1]}
        d["old_single"] = {str(s): {"mean": float(data[:, i].mean()), **direction_summary([r for r in rows if r["seed"] == s])}
                           for i, s in enumerate(names)}
        for s in eligible:
            d["directions"][str(s)].update(direction_summary([r for r in scope_rows if r["seed"] == s]))
        save(out / "scope.json", d)
        if scope_rows:
            table(out / "scope-pairs.csv", scope_rows)
        single = load(out / "complete.json")
        line = "single" if single["nominee"] is not None else "scope"
        nomination = {"line": line, "seed": single["nominee"] if line == "single" else d["nominee"]}
        save(results / "nomination.json", nomination)
    else:
        tasks = load(seal_path)["tasks"]
        seed = load(results / "nomination.json")["seed"]
        a, b = [audit(results / "final" / name, tasks) for name in ("baseline", "winner")]
        rows = paired(a, b, seed)
        d = score(b) - score(a)
        ci = np.quantile(submeans(d, 100000), [.025, .975])
        save(out / "final.json", {"seed": seed, "baseline": float(score(a).mean()), "winner": float(score(b).mean()),
                                   "delta": float(d.mean()), "ci95": ci.tolist(), "pass": bool(round(float(d.mean()), 12) >= DELTA and ci[0] > 0),
                                   **direction_summary(rows)})
        table(out / "final-pairs.csv", rows)


if __name__ == "__main__":
    main()
