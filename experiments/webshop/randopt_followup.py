"""Run the registered RandOpt follow-up using the archived WebShop harness."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import subprocess
import sys

import torch

from analyze_holdout import read_outcome
from run_direction_selection import change
from run_holdout import instruction_key, pilot, read_json, runner

PROTOCOL = Path(__file__).with_name("RANDOPT-PREREG.zh.md")
SEAL = Path(__file__).with_name("randopt-final-seal.json")
LAYOUT = "direction-selection-01"


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    pilot.persist(path, data)


def catalog_hashes(catalog):
    return pilot.fingerprint([catalog / "products.json", catalog / "attributes.json",
                              *(catalog / "index").glob("*")], catalog)


def price_hash(env):
    return hashlib.sha256(json.dumps(env.server.product_prices, sort_keys=True).encode()).hexdigest()


def environment(args, catalog, prior):
    from web_agent_site.engine import engine
    from web_agent_site.envs import web_agent_text_env as text_env

    human = args.checkout / "data/items_human_ins.json"
    assert runner.file_sha(human) == prior["human_sha256"]
    engine.DEFAULT_ATTR_PATH = str(catalog / "attributes.json")
    engine.HUMAN_ATTR_PATH = str(human)
    index = engine.LuceneSearcher(str(catalog / "index"))
    text_env.init_search_engine = lambda num_products=None: index
    random.seed(runner.SEED)
    env = text_env.WebAgentTextEnv(observation_mode="text", human_goals=True, file_path=str(catalog / "products.json"))
    assert index.num_docs == len(env.server.product_item_dict)
    return env


def seal_holdout(args, prior):
    env = environment(args, args.third_catalog, prior)
    previous_catalogs = [args.checkout / "data/items_shuffle_10000.json", args.catalog / "products.json"]
    asins = {p["asin"] for path in previous_catalogs for p in read_json(path)}
    annotations = read_json(args.checkout / "data/items_human_ins.json")
    keys = {instruction_key({"instruction_text": entry["instruction"]})
            for asin in asins for entry in annotations.get(asin, [])}
    ids = list(range(len(env.server.goals)))
    random.Random(2026112105).shuffle(ids)
    tasks = []
    for i in ids:
        goal = env.server.goals[i]
        key = instruction_key(goal)
        if goal["asin"] in asins or key in keys:
            continue
        asins.add(goal["asin"])
        keys.add(key)
        tasks.append({"goal_id": i, "goal": goal})
        if len(tasks) == 60:
            break
    assert len(tasks) == 60, f"Only {len(tasks)} independent final goals"
    assert all(env.server.search_engine.doc(t["goal"]["asin"]) is not None for t in tasks)
    replay = random.Random(2026112104).sample(sorted(t["goal_id"] for t in prior["splits"]["test"]), 20)
    sealed = {"tasks": tasks, "catalog_hashes": catalog_hashes(args.third_catalog),
              "price_hash": price_hash(env), "human_sha256": prior["human_sha256"],
              "seed": 2026112105, "catalog_source": "ce990ff products[20000:30000]",
              "replay_ids": sorted(replay), "archived_weight_hashes": prior["weight_hashes"],
              "archived_manifest_sha256": runner.file_sha(args.results_root / "webshop" / LAYOUT / "manifest.json")}
    if SEAL.exists():
        assert read_json(SEAL) == sealed
    else:
        save(SEAL, sealed)


def check_commit():
    repo = PROTOCOL.parents[2]
    for path in (PROTOCOL, SEAL):
        committed = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{path.relative_to(repo).as_posix()}"])
        assert committed.replace(b"\r\n", b"\n") == path.read_bytes().replace(b"\r\n", b"\n")
    return subprocess.check_output(["git", "-C", str(repo), "log", "-1", "--format=%H",
                                    "--", str(PROTOCOL)], text=True).strip()


def check_trajectory(old, new):
    fields = ("round", "prompt", "before", "action", "kind", "error", "action_index", "after", "reward", "bought")
    left, right = old["trace_rows"], new["trace_rows"]
    for a, b in zip(left, right):
        differences = {k: [a[k], b[k]] for k in fields if a[k] != b[k]}
        if differences:
            return {"round": a["round"], "differences": differences}
    if len(left) != len(right):
        return {"round": min(len(left), len(right)) + 1, "trace_length": [len(left), len(right)]}
    return {"raw_text_differences": [a["round"] for a, b in zip(left, right) if a["raw"] != b["raw"]]}


def run_episodes(env, model, tokenizer, tasks, folder, score, baseline=None):
    folder.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        i = task["goal_id"]
        path = folder / f"{i:04d}.jsonl"
        if not path.with_suffix(".json").exists():
            pilot.episode(env, model, tokenizer, task, path, score)
        result = read_outcome(folder, i)
        if baseline is not None:
            assert result["first_state"] == read_outcome(baseline, i)["first_state"]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("seal", "replay", "complete", "scope", "final"))
    for name in ("checkout", "catalog", "third-catalog", "model-dir", "results-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout))
    from web_agent_site.engine.goal import get_reward

    archive = args.results_root / "webshop" / LAYOUT
    prior = read_json(archive / "manifest.json")
    assert runner.file_sha(Path(pilot.__file__)) == prior["pilot_sha256"]
    assert runner.file_sha(Path(runner.__file__)) == prior["model_runner_sha256"]
    assert subprocess.check_output(["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True).strip() == prior["checkout"]
    if args.stage == "seal":
        seal_holdout(args, prior)
        return

    out = args.output
    revision = check_commit()
    assert read_json(SEAL)["archived_manifest_sha256"] == runner.file_sha(archive / "manifest.json")
    baseline = archive / "test/baseline"
    if args.stage in ("complete", "scope", "final"):
        gate = read_json(out / "replay.json")
        assert gate["pass"] and gate["protocol_commit"] == revision
    catalog = args.third_catalog if args.stage == "final" else args.catalog
    config = read_json(SEAL) if args.stage == "final" else prior
    assert catalog_hashes(catalog) == config["catalog_hashes"]
    env = environment(args, catalog, prior)
    assert price_hash(env) == config["price_hash"]
    tasks = config["tasks"] if args.stage == "final" else prior["splits"]["test"]
    assert all(env.server.goals[t["goal_id"]] == t["goal"] for t in tasks)
    model, tokenizer, layer, stats = runner.load_model(args.model_dir)
    save(out / f"{args.stage}-model.json", {
        **stats, "protocol_commit": revision, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "runtime": {name: version(name)
        for name in ("transformers", "bitsandbytes", "accelerate", "pyserini")},
    })
    original = layer.weight.detach().cpu().clone()
    assert runner.tensor_sha(original) == prior["weight_hashes"]["baseline"]
    if args.stage == "replay":
        selected = read_json(SEAL)["replay_ids"]
        replay = out / "replay"
        run_episodes(env, model, tokenizer, [t for t in tasks if t["goal_id"] in selected], replay, get_reward)
        checks = {str(i): check_trajectory(read_outcome(baseline, i), read_outcome(replay, i)) for i in selected}
        passed = len(checks) == 20 and all("round" not in c for c in checks.values())
        save(out / "replay.json", {"pass": passed, "protocol_commit": revision, "checks": checks})
        print("Replay", sum("round" not in c for c in checks.values()), "/20", flush=True)
        if not passed:
            raise SystemExit("Archived baseline differs. Consult replay.json before merging.")
        return

    ref = args.results_root / "alfworld/compare-01/diagnostic-input.json"
    ids = torch.tensor(read_json(ref), device="cuda")
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    reference = runner.diagnostic(model, layer, inputs)

    def run_single(seed, folder, comparison):
        weight = change(original, seed)
        digest = runner.tensor_sha(weight)
        assert digest == prior["weight_hashes"][str(seed)]
        try:
            layer.weight.copy_(weight)
            assert runner.tensor_sha(layer.weight) == digest
            forward = runner.compare_diagnostics(reference, runner.diagnostic(model, layer, inputs))
            assert forward["layer_input"]["max_abs"] == 0 and forward["layer_output"]["rms"] > 0
            save(folder / "weight.json", {"seed": seed, "sigma": prior["sigma"], "weight_sha256": digest, "forward": forward})
            run_episodes(env, model, tokenizer, tasks, folder, get_reward, comparison)
        finally:
            layer.weight.copy_(original)
            assert runner.tensor_sha(layer.weight) == prior["weight_hashes"]["baseline"]
            restored = runner.compare_diagnostics(reference, runner.diagnostic(model, layer, inputs))
            assert all(item["max_abs"] == 0 for item in restored.values())
            save(folder / "restored.json", restored)
        print(seed, len(tasks), "complete", flush=True)

    if args.stage == "complete":
        for seed in prior["seeds"]:
            if not (archive / "test" / str(seed)).exists():
                run_single(seed, out / "completed" / str(seed), baseline)
    elif args.stage == "scope":
        from randopt_scope import run_scope
        run_scope(args, prior, archive, out, env, model, tokenizer, get_reward)
    else:
        from randopt_scope import apply_scope, direction
        nomination = read_json(out / "nomination.json")
        if nomination["seed"] is None:
            return
        final = out / "final"
        run_episodes(env, model, tokenizer, tasks, final / "baseline", get_reward)
        if nomination["line"] == "single":
            run_single(nomination["seed"], final / "winner", final / "baseline")
        else:
            calibrated = read_json(out / "scope" / str(nomination["seed"]) / "calibration.json")
            noise = direction(model, nomination["seed"])
            assert calibrated["matched"] and [runner.tensor_sha(matrix) for matrix in noise] == calibrated["direction_hashes"]
            with apply_scope(model, nomination["seed"], calibrated["sigma"], noise):
                run_episodes(env, model, tokenizer, tasks, final / "winner", get_reward, final / "baseline")


if __name__ == "__main__":
    main()
