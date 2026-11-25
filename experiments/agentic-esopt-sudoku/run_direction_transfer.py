"""Test whether the four existing ES directions retain advantages on fresh boards."""

import argparse
from collections import Counter
import csv
import hashlib
from itertools import permutations
import json
import logging
from pathlib import Path
import random
from threading import Thread
from time import perf_counter

from run_local import ROOT, upstream_directory
from run_observable_es import load_policy, load_upstream, save, verification_rows, weight_digest


MASKS = (5, 10, 15)
BOARD_SEED = 2026112900
SEEDS = {"calibration": (2026112901,), "discovery": (2026113001, 2026113002),
         "confirmation": (2026120101, 2026120102)}
BOARDS_PER_MASK = {"calibration": 2, "discovery": 4, "confirmation": 4}
PILOT = ROOT / "runs/observable-es-01"
GROUPS = [([9 * i + j for j in range(9)]) for i in range(9)]
GROUPS += [[9 * i + j for i in range(9)] for j in range(9)]
GROUPS += [[9 * (r + i) + c + j for i in range(3) for j in range(3)]
           for r in (0, 3, 6) for c in (0, 3, 6)]
PEERS = [set.union(*(set(group) for group in GROUPS if i in group)) - {i} for i in range(81)]


def flat(grid):
    return tuple(v for row in grid for v in row)


def grid(board):
    return [board[i:i + 9] for i in range(0, 81, 9)]


def solve(board, limit, rng=None):
    """Find up to limit completions, using randomized values only for board generation."""
    solutions = []
    digits = set(range(1, 10))

    def search():
        missing = [i for i, value in enumerate(board) if value == 0]
        if not missing:
            solutions.append(board.copy())
            return
        options = [(digits - {board[j] for j in PEERS[i]}, i) for i in missing]
        values, cell = min(options, key=lambda item: len(item[0]))
        ordered = sorted(values)
        if rng is not None:
            rng.shuffle(ordered)
        for value in ordered:
            board[cell] = value
            search()
            if len(solutions) >= limit:
                break
        board[cell] = 0

    search()
    return solutions


def generate_cases(upstream):
    official = [json.loads(line) for split in ("train", "eval") for line in
                (upstream / f"data/sudoku/{split}.jsonl").read_bytes().splitlines()]
    solutions = {flat(row["solution"]) for row in official}
    puzzles = {flat(row["puzzle"]) for row in official}
    rng = random.Random(BOARD_SEED)
    cases = {}
    for split, size in BOARDS_PER_MASK.items():
        cases[split] = []
        for mask in MASKS:
            for index in range(1, size + 1):
                while True:
                    solution, = solve([0] * 81, 1, rng)
                    if tuple(solution) in solutions:
                        continue
                    puzzle = solution.copy()
                    for cell in rng.sample(range(81), mask):
                        puzzle[cell] = 0
                    if tuple(puzzle) not in puzzles and solve(puzzle, 2) == [solution]:
                        break
                solutions.add(tuple(solution))
                puzzles.add(tuple(puzzle))
                cases[split].append(dict(id=f"{split}-m{mask:02}-{index:02}", mask_count=mask,
                                         puzzle=grid(puzzle), solution=grid(solution), source="generated"))
    return cases


def freeze(path, record):
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(record)):
            raise RuntimeError(f"Frozen artifact is different: {path}")
    else:
        save(path, record)


def score_table(rows):
    return {(r["policy"], r["mask_count"], r["task"], r["rollout_seed"]): r["score"] for r in rows}


def paired(table, a, b, mask):
    diffs = {(task, seed): score - table[b, mask, task, seed]
              for (policy, group, task, seed), score in table.items() if policy == a and group == mask}
    boards = Counter()
    seeds = Counter()
    for (task, seed), value in diffs.items():
        boards[task] += value
        seeds[seed] += value
    return dict(a=a, b=b, mask=mask, net=sum(boards.values()),
                more_boards=sum(v > 0 for v in boards.values()),
                fewer_boards=sum(v < 0 for v in boards.values()),
                tied_boards=sum(v == 0 for v in boards.values()), by_seed=dict(seeds), by_board=dict(boards))


def prediction(rows, masks, policies):
    table = score_table(rows)
    contrasts = {(a, b, mask): paired(table, a, b, mask) for a, b in permutations(policies, 2)
                 for mask in masks}
    crossings = []
    if len(masks) == 2:
        low, high = masks
        for a, b in permutations(policies, 2):
            first, second = contrasts[a, b, low]["net"], contrasts[b, a, high]["net"]
            if min(first, second) > 0:
                crossings.append((-min(first, second), -first - second, a, b))
        if crossings:
            _, _, a, b = min(crossings)
            return dict(kind="crossover", discovery=[contrasts[a, b, low], contrasts[b, a, high]])
    ordered = sorted(contrasts.values(), key=lambda r: (-r["net"], r["mask"], r["a"], r["b"]))
    if ordered and ordered[0]["net"] > 0:
        return dict(kind="single-mask", discovery=ordered[:1])
    return dict(kind="none", discovery=[])


def run(output):
    import torch
    from werkzeug.serving import make_server

    upstream = upstream_directory()
    trainer = load_upstream(upstream)
    previous = json.loads((PILOT / "manifest.json").read_text())
    center = json.loads((PILOT / "center.json").read_text())["sha256"]
    directions = previous["directions"]
    expected = {"center": center, **json.loads((PILOT / "development.json").read_text())["direction_hashes"]}
    cases = generate_cases(upstream)
    policy_seeds = {"center": None, **directions}
    freeze(output / "boards.json", cases)
    freeze(output / "protocol.json", dict(
        board_seed=BOARD_SEED, sample_seeds=SEEDS, counts=BOARDS_PER_MASK,
        parent_manifest=previous, weight_sha256=expected,
        calibration="one or more boards with both a success and a failure among five fixed policies",
        mask_selection="first two qualifying masks in priority order 5,10,15",
        discovery="strict crossover, maximize smaller gap then total gap, ties by direction name;"
                  " otherwise strongest positive single-mask contrast, ties by mask and direction name",
        batch_seed_rule="nominal seed plus 100 times the first task index within the phase",
        confirmation="full-board score, each board averages its two decoding seeds;"
                     " directional match requires a positive net in every predicted group;"
                     " stronger replication requires two net favorable boards and positive net at both seeds per group",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    ))
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    app, es, _, selected, snapshot = load_policy(upstream, previous["model"], output)
    if weight_digest(snapshot) != center:
        raise RuntimeError("Loaded center differs from the observable ES run.")

    def restore(policy):
        with torch.no_grad():
            for (_, parameter), (_, original) in zip(selected, snapshot):
                parameter.copy_(original)
        es.active_perturbation = None
        if weight_digest(selected) != center:
            raise RuntimeError("Center restoration differs.")
        if policy != "center":
            es.apply(seed=policy_seeds[policy], sigma=previous["sigma"])
            if weight_digest(selected) != expected[policy]:
                raise RuntimeError(f"{policy} differs from the observable ES run.")

    def phase(split, masks, prediction_sha=None):
        all_rows = []
        phase_cases = [trainer.SudokuTask(**row) for row in cases[split] if row["mask_count"] in masks]
        env = trainer.SudokuEnv(upstream / "data/sudoku/eval.jsonl")
        # Evaluate each pair of boards independently so progress survives interruptions.
        for policy, direction_seed in policy_seeds.items():
            restore(policy)
            for seed in SEEDS[split]:
                for i in range(0, len(phase_cases), 2):
                    tasks = phase_cases[i:i + 2]
                    name = f"{split}-{policy}-{seed}-{tasks[0].id}"
                    path = output / f"{name}.json"
                    context = dict(split=split, policy=policy, direction_seed=direction_seed,
                                   rollout_seed=seed, batch_seed=seed + i * 100, weight_sha256=expected[policy],
                                   prediction_sha256=prediction_sha)
                    task_map = {task.id: task for task in tasks}
                    if path.exists():
                        record = json.loads(path.read_text())
                        if any(record[k] != v for k, v in context.items()):
                            raise RuntimeError(f"Cached record does not match {name}.")
                        rows = verification_rows(record, task_map)
                    else:
                        start = perf_counter()
                        torch.manual_seed(context["batch_seed"])
                        scores = trainer.eval_endpoint_batched(
                            env=env, endpoint=endpoint, tasks=tasks, model="Qwen3.5-4B",
                            endpoint_batch_size=previous["endpoint_batch_size"], **previous["decoding"],
                        )
                        record = dict(**context, scores=scores, seconds=perf_counter() - start)
                        rows = verification_rows(record, task_map)
                        save(path, record)
                    all_rows.extend(rows)
                    print(name, f"{sum(r['score'] for r in rows)}/{len(rows)}",
                          f"{record['seconds']:.1f}s", flush=True)
        return all_rows

    service = make_server("127.0.0.1", 0, app)
    endpoint = f"http://127.0.0.1:{service.server_port}"
    thread = Thread(target=service.serve_forever, daemon=True)
    thread.start()
    try:
        calibration = phase("calibration", MASKS)
        successes = Counter((r["mask_count"], r["task"]) for r in calibration if r["score"])
        masks = [mask for mask in MASKS if any(
            0 < successes[mask, row["id"]] < len(policy_seeds)
            for row in cases["calibration"] if row["mask_count"] == mask
        )][:2]
        freeze(output / "calibration.json", dict(masks=masks, successes={
            str(mask): {r["id"]: successes[mask, r["id"]] for r in cases["calibration"] if r["mask_count"] == mask}
            for mask in MASKS
        }))
        print(f"Calibrated masks: {masks}", flush=True)
        discovery = phase("discovery", masks) if masks else []
        frozen = prediction(discovery, masks, list(directions))
        frozen["masks"] = masks
        frozen["discovery_sha256"] = hashlib.sha256(json.dumps(discovery).encode()).hexdigest()
        path = output / "prediction.json"
        if not path.exists() and any(output.glob("confirmation-*.json")):
            raise RuntimeError("Confirmation already exists without a frozen prediction.")
        freeze(path, frozen)
        frozen_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"Frozen discovery prediction ({frozen['kind']}): {frozen_sha}", flush=True)
        confirmation = phase("confirmation", masks, frozen_sha) if masks else []
    finally:
        service.shutdown()
        thread.join()
        service.server_close()

    rows = calibration + discovery + confirmation
    temporary = output / "scores.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "scores.csv")
    result = [paired(score_table(confirmation), r["a"], r["b"], r["mask"]) for r in frozen["discovery"]]
    versus_center = {split: [paired(score_table(split_rows), p, "center", mask)
                             for p in directions for mask in masks]
                     for split, split_rows in (("discovery", discovery), ("confirmation", confirmation))}
    save(output / "summary.json", dict(
        rolls_by_phase=dict(calibration=len(calibration), discovery=len(discovery), confirmation=len(confirmation)),
        prediction=frozen, prediction_sha256=frozen_sha, confirmation=result, versus_center=versus_center,
        directional_match=bool(result) and all(r["net"] > 0 for r in result),
        stronger_replication=bool(result) and all(r["more_boards"] - r["fewer_boards"] >= 2
                                                 and min(r["by_seed"].values()) > 0 for r in result),
    ))
    print(f"Verified {len(rows)} trajectories. Summary: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        parser.error("--run-id must be one directory name")
    output = ROOT / "runs" / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    run(output)
