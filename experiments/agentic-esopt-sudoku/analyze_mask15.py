"""Summarize the pinned author log and the three local paired checkpoint runs."""

import csv
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from run_local import upstream_directory


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis-mask15"
MODELS = ("base", "esopt")
LOG_HASH = "80dfe2b5aa18792d20163fad16720f18aa0bf8ced3f2689ae6288520d1dc8899"


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def author_matrices():
    data = (ROOT / ".cache/author-mask15.log").read_bytes()
    assert hashlib.sha256(data).hexdigest() == LOG_HASH
    log = data.decode()
    sections = log.split("[sigma] gen=")[1:]
    assert len(sections) == 100
    generations, matrix_rows = [], []
    for gen, section in enumerate(sections):
        generation, sigma = section.split()[:2]
        assert int(generation) == gen
        split = section.rindex(f"[sample] gen={gen} ")
        records = re.findall(r"\[case\] task=(train-\d+) mask=15 endpoint=\S+:(\d+) score=(\S+)", section[:split])
        assert len(records) == 1024
        expected = {int(idx): float(score) for idx, score in
                    re.findall(r"\[sample\] gen=\d+ idx=(\d+) reward=(\S+)", section)}
        by_port = defaultdict(list)
        for task, port, score in records:
            assert float(score) in (0, 1)
            by_port[int(port)].append((task, int(float(score))))
        scores = {}
        for port, cases in by_port.items():
            assert len(cases) == 64
            for offset in (0, 1):
                idx = port - 12100 + offset * 16
                row = dict(cases[offset * 32 : (offset + 1) * 32])
                assert len(row) == 32 and sum(row.values()) / 32 == expected[idx]
                scores[idx] = row
        assert set(scores) == set(range(32))
        tasks = sorted(scores[0])
        assert all(sorted(row) == tasks for row in scores.values())
        totals = [sum(row.values()) for row in scores.values()]
        ties = [(a, b) for a, b in itertools.combinations(scores.values(), 2)
                if sum(a.values()) == sum(b.values())]
        generations.append(dict(generation=gen, sigma=float(sigma.split("=")[1]),
                                zero_directions=totals.count(0), min_solved=min(totals),
                                max_solved=max(totals), distinct_totals=len(set(totals)),
                                tasks_no_direction_solved=sum(not any(row[t] for row in scores.values()) for t in tasks),
                                tied_direction_pairs=len(ties), tied_different_solved_sets=sum(a != b for a, b in ties)))
        matrix_rows.extend(dict(generation=gen, direction=idx, **scores[idx]) for idx in range(32))
    write_csv("author-generations.csv", generations)
    write_csv("author-direction-task.csv", matrix_rows)
    # eval-* cases occur only at scheduled checkpoint evaluations.
    checkpoints, outcomes = [], defaultdict(list)
    for line in log.splitlines():
        case = re.fullmatch(r"\[case\] task=(eval-\d+) mask=15 endpoint=\S+ score=(\S+)", line)
        if case:
            outcomes[case[1]].append(float(case[2]))
        point = re.match(r"\[eval\] generation=(-?\d+) split=eval repeats=3 .* average=([\d.]+)", line)
        if point:
            assert len(outcomes) == 32 and all(len(v) == 3 for v in outcomes.values())
            assert abs(mean(map(mean, outcomes.values())) - float(point[2])) < 1e-6
            checkpoints.extend(dict(generation=int(point[1]), task=task, solved_of_three=int(sum(values)))
                               for task, values in sorted(outcomes.items()))
            outcomes.clear()
    assert len(checkpoints) == 11 * 32 and not outcomes
    write_csv("author-eval-puzzles.csv", checkpoints)
    logged_training = {key for row in matrix_rows for key in row if key.startswith("train-")}
    return logged_training, {row["task"] for row in checkpoints}


def audit_split(upstream, logged_training, logged_eval):
    directory = upstream / "data/sudoku"
    rows = {
        split: [json.loads(line) for line in (directory / f"{split}.jsonl").read_bytes().splitlines()]
        for split in ("train", "eval")
    }
    by_id = {split: {r["id"]: r for r in cases} for split, cases in rows.items()}
    assert all(len(by_id[split]) == len(cases) for split, cases in rows.items())
    assert logged_eval == {r["id"] for r in rows["eval"] if r["mask_count"] == 15}
    assert all(by_id["train"][task]["mask_count"] == 15 for task in logged_training)
    training = [by_id["train"][task] for task in sorted(logged_training)]
    evaluation = [by_id["eval"][task] for task in sorted(logged_eval)]
    summary = []
    for name, source in (("logged-mask15", training), ("official-train-all-masks", rows["train"])):
        result = dict(source=name, train_count=len(source), eval_count=len(evaluation))
        for key in ("id", "puzzle_compact", "solution_compact"):
            train_values = {r[key] for r in source}
            result[f"{key}_matches"] = sum(r[key] in train_values for r in evaluation)
        summary.append(result)
        print(result)
    write_csv("train-eval-overlap.csv", summary)


def conflicts_before(board, row, col, value):
    checks = {
        "row": board[row],
        "column": [line[col] for line in board],
        "box": [board[r][c] for r in range(row // 3 * 3, row // 3 * 3 + 3)
                for c in range(col // 3 * 3, col // 3 * 3 + 3)],
    }
    return [name for name, digits in checks.items() if value in digits]


def trajectory(puzzle, score):
    board = [line.copy() for line in puzzle]
    first, action, units = None, "", ""
    invalid = 0
    for n, turn in enumerate(score["turns"], 1):
        assert "error" not in turn and turn["turn"] == n - 1
        if turn["valid"]:
            move = turn["action"]
            r, c, value = move["row"] - 1, move["col"] - 1, move["value"]
            assert board[r][c] == 0
            conflict = conflicts_before(board, r, c, value)
            if conflict and first is None:
                first, action, units = n, f"r{r+1}c{c+1}={value}", "+".join(conflict)
            board[r][c] = value
        else:
            invalid += 1
        assert board == turn["board"]
    full = all(all(row) for row in board)
    assert board == score["prediction"] and int(full and first is None) == score["score"]
    return dict(solved=int(score["score"]), full=int(full), turns=len(score["turns"]),
                rejected_actions=invalid, first_conflict_turn=first,
                first_conflict_action=action, first_conflict_units=units)


def paired_rollouts():
    data = (ROOT / "upstream-snapshot/data/sudoku/eval.jsonl").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    tasks = {r["id"]: r["puzzle"] for line in data.splitlines()
             if (r := json.loads(line))["mask_count"] == 15}
    assert len(tasks) == 32
    traces, batches = [], []
    counts = defaultdict(Counter)
    for batch, seed in enumerate((20261124, 20261125, 20261126), 1):
        run = ROOT / "runs" / f"paired32-s{batch}"
        pairs = json.loads((run / "paired.json").read_text())
        reported = {r["task_id"]: r for r in pairs}
        assert len(pairs) == 32 and set(reported) == tasks.keys()
        for model in MODELS:
            directory = run / model
            manifest = json.loads((directory / "manifest.json").read_text())
            assert (manifest["seed"], manifest["bits"], manifest["eval_data_sha256"]) == (seed, 8, digest)
            history = json.loads((directory / "history.json").read_text())[-1]["eval"]
            repeat, = history["runs"]
            assert repeat["valid_count"] == len(repeat["scores"]) == 32
            assert {r["task_id"] for r in repeat["scores"]} == tasks.keys()
            for row in repeat["scores"]:
                task = row["task_id"]
                stats = trajectory(tasks[task], row)
                assert stats["solved"] == reported[task][model]
                traces.append(dict(batch=batch, seed=seed, model=model, task=task, **stats))
                counts[task][model] += stats["solved"]
        outcomes = Counter((r["base"], r["esopt"]) for r in pairs)
        batches.append(dict(batch=batch, base=sum(r["base"] for r in pairs),
                            esopt=sum(r["esopt"] for r in pairs), gains=outcomes[0, 1],
                            losses=outcomes[1, 0], both=outcomes[1, 1], neither=outcomes[0, 0]))
    write_csv("local-episodes.csv", traces)
    write_csv("local-by-puzzle.csv", [dict(task=t, base=counts[t]["base"], esopt=counts[t]["esopt"])
                                      for t in sorted(tasks)])
    histogram = [dict(successes_of_three=n, **{m: sum(c[m] == n for c in counts.values()) for m in MODELS})
                 for n in range(4)]
    write_csv("local-by-success-count.csv", histogram)
    print("successes per puzzle:", histogram)
    write_csv("local-batches.csv", batches)
    for model in MODELS:
        subset = [r for r in traces if r["model"] == model]
        print(model, "full", sum(r["full"] for r in subset), "rejected actions",
              sum(r["rejected_actions"] for r in subset), "median conflict turn (conflicts only)",
              median(r["first_conflict_turn"] for r in subset if r["first_conflict_turn"]))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    audit_split(upstream_directory(), *author_matrices())
    paired_rollouts()
