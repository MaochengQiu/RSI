"""Run one observable, quantized Sudoku ES round with an independent candidate replay."""

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import random
import runpy
import sys
from threading import Thread

from analyze_mask15 import trajectory
from run_local import MODEL_REVISIONS, REVISION, ROOT, SERVER_PATH, TRAINER_PATH, upstream_directory


MASKS = (5, 15)
PERTURBATION_SEED = 20260701
ROLLOUT_SEEDS = {"development": (2026112701, 2026112702), "holdout": (2026112801, 2026112802)}
SIGMA = 0.0007
ALPHA = 0.0005
DECODE = dict(max_tokens=64, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
              presence_penalty=1.5, repetition_penalty=1.0, timeout=900, max_turns=45)


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_upstream(upstream):
    os.environ["ROOT"] = str(upstream)
    spec = importlib.util.spec_from_file_location("sudoku_trainer", upstream / TRAINER_PATH)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return trainer


def pick_tasks(trainer, upstream):
    data = upstream / "data/sudoku"
    envs = {"development": trainer.SudokuEnv(data / "train.jsonl"),
            "holdout": trainer.SudokuEnv(data / "eval.jsonl")}
    tasks = {}
    for split, env in envs.items():
        tasks[split] = [task for mask in MASKS for task in
                        [t for t in env.tasks if t.mask_count == mask][:2]]
        assert Counter(t.mask_count for t in tasks[split]) == {5: 2, 15: 2}
    for key in ("id", "puzzle", "solution"):
        encode = lambda t: json.dumps(getattr(t, key))
        assert not {encode(t) for t in envs["development"].tasks} & {
            encode(t) for t in tasks["holdout"]
        }
    return envs, tasks


def weight_digest(named_tensors):
    import torch

    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        digest.update(name.encode() + b"\0")
        data = tensor.detach().cpu().contiguous().view(-1).view(torch.uint8)
        digest.update(memoryview(data.numpy()))
    return digest.hexdigest()


def load_policy(upstream, model, output):
    import contextlib
    import torch

    path = upstream / SERVER_PATH
    sys.path.insert(0, str(path.parent))
    sys.argv = [str(path), "--path", str(ROOT / "models" / model), "--load-in-4bit",
                "--dtype", "bfloat16", "--chat-template-enable-thinking", "false"]
    with (output / "server.log").open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        server = runpy.run_path(str(path), run_name="sudoku_policy")
    if not all(p.is_cuda for p in server["model"].parameters()):
        raise RuntimeError("The pilot requires the entire quantized policy on the GPU.")

    # Keep the official prompt and generation code, while moving its per-turn
    # prompt dumps from the progress stream into the run directory.
    original = server["completions"].__globals__["_generate_from_formatted_prompt"]

    def logged_generation(*args, **kwargs):
        with (output / "server.log").open("a", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log):
                return original(*args, **kwargs)

    server["completions"].__globals__["_generate_from_formatted_prompt"] = logged_generation
    es = server["model_es"]
    info = es.init(server["model"])
    if not all(type(p) is torch.nn.Parameter for _, _, p in es.parameter_infos):
        raise RuntimeError("Unexpected floating-point parameter wrapper in ES selection.")
    selected = [(name, parameter) for name, _, parameter in es.parameter_infos]
    snapshot = [(name, parameter.detach().cpu()) for name, parameter in selected]
    return server["app"], es, info, selected, snapshot


def select_pair(development):
    """Pick a crossed pair on development scores only, using deterministic ties."""
    counts = Counter((row["policy"], row["mask_count"]) for row in development if row["score"] == 1)
    policies = sorted({row["policy"] for row in development if row["policy"] != "center"})
    candidates = []
    for low, high in itertools.permutations(policies, 2):
        gaps = (counts[low, 5] - counts[high, 5], counts[high, 15] - counts[low, 15])
        if min(gaps) > 0:
            candidates.append((-min(gaps), -sum(gaps), low, high))
    if not candidates:
        return None
    _, _, low, high = min(candidates)
    return {"mask5": low, "mask15": high}


def verification_rows(record, tasks):
    if {row["task_id"] for row in record["scores"]} != tasks.keys() or len(record["scores"]) != len(tasks):
        raise RuntimeError("Incomplete task coverage.")
    return [
        dict(split=record["split"], policy=record["policy"], direction_seed=record["direction_seed"],
             rollout_seed=record["rollout_seed"], task=row["task_id"], mask_count=row["mask_count"],
             score=int(row["score"]), **trajectory(tasks[row["task_id"]].puzzle, row))
        for row in record["scores"]
    ]


def run(args, output):
    import logging
    import torch
    from werkzeug.serving import make_server

    upstream = upstream_directory()
    trainer = load_upstream(upstream)
    envs, tasks = pick_tasks(trainer, upstream)
    rng = random.Random(PERTURBATION_SEED)
    directions = {f"direction-{i + 1}": rng.randrange(1, 2**31 - 1) for i in range(4)}
    manifest = dict(
        upstream=REVISION, model=args.model, model_revision=MODEL_REVISIONS[args.model],
        bits=4, scope="floating parameters in quantized policy", directions=directions,
        perturbation_rng_seed=PERTURBATION_SEED, rollout_seeds=ROLLOUT_SEEDS, sigma=SIGMA, alpha=ALPHA,
        normalization="zscore ddof=0", endpoint_batch_size=2, decoding=DECODE,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tasks={split: [t.id for t in cases] for split, cases in tasks.items()},
    )
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
            raise RuntimeError("Existing manifest differs. Use a fresh run ID.")
    else:
        save(manifest_path, manifest)
    print("One ES round, four directions, two task groups and two decode seeds per split.", flush=True)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    app, es, info, selected, snapshot = load_policy(upstream, args.model, output)
    center_sha = weight_digest(snapshot)
    if weight_digest(selected) != center_sha:
        raise RuntimeError("Center snapshot does not match the loaded policy.")
    save(output / "center.json", dict(**info, sha256=center_sha))
    print(f"Mutable tensors: {info['n_tensors']}, parameters: {info['total_params']:,}", flush=True)

    def restore():
        with torch.no_grad():
            for (_, parameter), (_, original) in zip(selected, snapshot):
                parameter.copy_(original)
        es.active_perturbation = None
        es.update_history.clear()
        if weight_digest(selected) != center_sha:
            raise RuntimeError("Center did not restore exactly.")

    def evaluate(split, policy, weight_sha, direction_seed=None):
        all_rows = []
        task_map = {t.id: t for t in tasks[split]}
        for seed in ROLLOUT_SEEDS[split]:
            path = output / f"{split}-{policy}-{seed}.json"
            context = dict(split=split, policy=policy, direction_seed=direction_seed,
                           rollout_seed=seed, weight_sha256=weight_sha)
            if path.exists():
                record = json.loads(path.read_text())
                if any(record[k] != v for k, v in context.items()):
                    raise RuntimeError(f"Replay context differs from {path}.")
                rows = verification_rows(record, task_map)
            else:
                torch.manual_seed(seed)
                scores = trainer.eval_endpoint_batched(
                    env=envs[split], endpoint=endpoint, tasks=tasks[split], model="Qwen3.5-4B",
                    endpoint_batch_size=2, **DECODE,
                )
                record = dict(**context, scores=scores)
                rows = verification_rows(record, task_map)
                save(path, record)
            print(split, policy, seed, {mask: sum(r["score"] for r in rows if r["mask_count"] == mask)
                                       for mask in MASKS}, flush=True)
            all_rows.extend(rows)
        return all_rows

    def candidates(split, expected):
        rows, hashes = [], {}
        for policy, seed in directions.items():
            restore()
            es.apply(seed=seed, sigma=SIGMA)
            digest = weight_digest(selected)
            if digest == center_sha or (expected is not None and digest != expected[policy]):
                raise RuntimeError(f"Failed to reconstruct {policy}.")
            hashes[policy] = digest
            rows.extend(evaluate(split, policy, digest, seed))
        if len(set(hashes.values())) != len(directions):
            raise RuntimeError("Distinct direction seeds reconstructed identical weights.")
        restore()
        return rows, hashes

    service = make_server("127.0.0.1", 0, app)
    endpoint = f"http://127.0.0.1:{service.server_port}"
    thread = Thread(target=service.serve_forever, daemon=True)
    thread.start()
    try:
        development = evaluate("development", "center", center_sha)
        direction_rows, hashes = candidates("development", None)
        development.extend(direction_rows)
        rewards = [sum(r["score"] for r in direction_rows if r["policy"] == p)
                   / (len(tasks["development"]) * len(ROLLOUT_SEEDS["development"])) for p in directions]
        selection = select_pair(development)
        save(output / "development.json", dict(direction_hashes=hashes, rewards=rewards, pair=selection))

        update_args = dict(seeds=list(directions.values()), rewards=rewards, alpha=ALPHA,
                           reward_normalization="zscore")
        update = es.update(**update_args)
        updated_sha = weight_digest(selected)
        restore()
        es.update(**update_args)
        if weight_digest(selected) != updated_sha:
            raise RuntimeError("The aggregate update did not replay exactly.")
        save(output / "update.json", dict(request=update_args, result=update,
                                         center_sha256=center_sha, updated_sha256=updated_sha))
        print(f"Aggregate update replayed exactly. Changed weights: {updated_sha != center_sha}", flush=True)
        holdout = evaluate("holdout", "updated", updated_sha)
        restore()
        holdout.extend(evaluate("holdout", "center", center_sha))
        heldout_candidates, _ = candidates("holdout", hashes)
        holdout.extend(heldout_candidates)
    finally:
        service.shutdown()
        thread.join()
        service.server_close()

    rows = development + holdout
    temporary = output / "scores.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "scores.csv")
    summary = {}
    for split, split_rows in (("development", development), ("holdout", holdout)):
        summary[split] = {
            policy: {str(mask): [sum(r["score"] for r in split_rows if r["policy"] == policy
                                    and r["mask_count"] == mask and r["rollout_seed"] == seed)
                                 for seed in ROLLOUT_SEEDS[split]] for mask in MASKS}
            for policy in sorted({r["policy"] for r in split_rows})
        }
    save(output / "summary.json", dict(rollouts=len(rows), counts=summary, development_pair=selection))
    print(f"Completed and replay-verified {len(rows)} trajectories: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("esopt", "base"), default="esopt")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        parser.error("--run-id must be one directory name")
    output = ROOT / "runs" / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    run(args, output)
