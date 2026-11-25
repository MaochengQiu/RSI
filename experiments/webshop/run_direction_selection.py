"""Screen eight fresh perturbations on development tasks, freeze one, and evaluate once."""

import argparse
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
import subprocess
import sys

import torch

from analyze_holdout import read_outcome
from run_holdout import instruction_key, pilot, read_json, runner

SIGMA = 0.003
SEEDS = random.Random(2026112101).sample(range(1, 2**31), 8)
SPLIT_SEED = 2026112102


def freeze(path, data):
    if path.exists():
        assert read_json(path) == data, f"Frozen file mismatch: {path}"
    else:
        pilot.persist(path, data)


def change(original, seed):
    noise = torch.randn(original.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
    return (original.float() + SIGMA * noise).to(original.dtype)


def split_tasks(goals, old_products, old_annotations):
    seen_asins = {p["asin"] for p in old_products}
    seen_text = {instruction_key({"instruction_text": r["instruction"]}) for rows in old_annotations.values() for r in rows}
    order = list(range(len(goals)))
    random.Random(SPLIT_SEED).shuffle(order)
    eligible = []
    for i in order:
        goal = goals[i]
        key = instruction_key(goal)
        if goal["asin"] in seen_asins or key in seen_text:
            continue
        eligible.append({"goal_id": i, "goal": goal})
        seen_asins.add(goal["asin"])
        seen_text.add(key)
    if len(eligible) <= 24:
        raise ValueError(f"Only {len(eligible)} independent goals available.")
    return {"development": eligible[:24], "test": eligible[24:84]}, len(eligible)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "catalog", "model-dir", "results-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root, catalog, out = args.checkout, args.catalog, args.output
    sys.path.insert(0, str(root))
    from web_agent_site.engine import engine
    from web_agent_site.engine.goal import get_reward
    from web_agent_site.envs import web_agent_text_env as text_env

    prior = read_json(args.results_root / "webshop/pilot-03/manifest.json")
    assert runner.file_sha(Path(pilot.__file__)) == prior["runner_sha256"]
    assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == prior["checkout"]
    human = root / "data/items_human_ins.json"
    assert runner.file_sha(human) == prior["catalog_and_index"]["data/items_human_ins.json"]
    engine.DEFAULT_ATTR_PATH = str(catalog / "attributes.json")
    engine.HUMAN_ATTR_PATH = str(human)
    index = engine.LuceneSearcher(str(catalog / "index"))
    text_env.init_search_engine = lambda num_products=None: index
    random.seed(prior["goal_and_sample_seed"])
    env = text_env.WebAgentTextEnv(observation_mode="text", human_goals=True, file_path=str(catalog / "products.json"))
    assert index.num_docs == len(env.server.product_item_dict)
    splits, eligible = split_tasks(env.server.goals, read_json(root / "data/items_shuffle_10000.json"),
                                   read_json(root / "data/items_human_ins_10000.json"))
    assert all(index.doc(t["goal"]["asin"]) is not None for tasks in splits.values() for t in tasks)
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer, layer, stats = runner.load_model(args.model_dir)
    original = layer.weight.detach().cpu().clone()
    original_hash = runner.tensor_sha(original)
    ref = args.results_root / "alfworld/compare-01"
    assert original_hash == read_json(ref / "weight-verification.json")["original_sha256"]
    hashes = {"baseline": original_hash, **{str(s): runner.tensor_sha(change(original, s)) for s in SEEDS}}
    assert len(set(hashes.values())) == len(hashes)
    catalog_hashes = pilot.fingerprint([catalog / "products.json", catalog / "attributes.json",
                                        *(catalog / "index").glob("*")], catalog)
    manifest = {"seeds": SEEDS, "control_seed": SEEDS[0], "sigma": SIGMA, "target": runner.TARGET + ".weight",
                "weight_hashes": hashes, "splits": splits, "split_seed": SPLIT_SEED, "eligible_count": eligible,
                "catalog_source": "YWZBrandon/webshop-data@ce990ff, zero-based products[10000:20000]",
                "catalog_hashes": catalog_hashes, "human_sha256": runner.file_sha(human),
                "price_hash": hashlib.sha256(json.dumps(env.server.product_prices, sort_keys=True).encode()).hexdigest(),
                "model_revision": runner.MODEL_REVISION, "checkout": prior["checkout"],
                "pilot_sha256": prior["runner_sha256"], "model_runner_sha256": runner.file_sha(Path(runner.__file__)),
                "script_sha256": runner.file_sha(Path(__file__)), "rounds": pilot.MAX_ROUNDS, "decoding": "greedy",
                "selection": "Highest mean official development reward, rounded to 12 decimal places for ties; earliest drawn seed breaks ties.",
                "test": "Winner vs original primary; winner vs first pre-drawn seed secondary. No replacement, tuning or ensemble."}
    freeze(out / "manifest.json", manifest)
    pilot.persist(out / "model-load.json", stats)
    print("Frozen", {k: len(v) for k,v in splits.items()}, "eligible", eligible, "seeds", SEEDS, flush=True)
    ids = torch.tensor(read_json(ref / "diagnostic-input.json"), device="cuda")
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    reference = runner.diagnostic(model, layer, inputs)

    def diff():
        return runner.compare_diagnostics(reference, runner.diagnostic(model, layer, inputs))

    def run(split, seed=None):
        name = str(seed) if seed is not None else "baseline"
        folder = out / split / name
        folder.mkdir(parents=True, exist_ok=True)
        rewards = []
        try:
            if seed is not None:
                layer.weight.copy_(change(original, seed))
            assert runner.tensor_sha(layer.weight) == hashes[name]
            d = diff()
            assert d["layer_input"]["max_abs"] == 0
            assert (d["layer_output"]["rms"] > 0) == (seed is not None)
            if seed is None:
                assert all(v["max_abs"] == 0 for v in d.values())
            pilot.persist(folder / "forward.json", d)
            for task in splits[split]:
                i = task["goal_id"]
                path = folder / f"{i:04d}.jsonl"
                if not path.with_suffix(".json").exists():
                    pilot.episode(env, model, tokenizer, task, path, get_reward)
                result = read_outcome(folder, i)
                if seed is not None:
                    assert result["first_state"] == read_json(out / split / "baseline" / path.with_suffix(".json").name)["first_state"]
                rewards.append(result["reward"])
        finally:
            layer.weight.copy_(original)
            assert runner.tensor_sha(layer.weight) == original_hash
            restored = diff()
            assert all(v["max_abs"] == 0 for v in restored.values())
            pilot.persist(folder / "restored.json", restored)
        print(split, name, len(rewards), "complete", flush=True)
        return mean(rewards)

    baseline = run("development")
    dev = {s: run("development", s) for s in SEEDS}
    winner = max(SEEDS, key=lambda s: round(dev[s], 12))
    freeze(out / "selection.json", {"winner": winner, "winner_sha256": hashes[str(winner)], "control": SEEDS[0],
                                   "baseline_mean": baseline, "development_means": {str(s): v for s,v in dev.items()}})
    print("Locked winner", winner, flush=True)
    run("test")
    for seed in dict.fromkeys([winner, SEEDS[0]]):
        run("test", seed)


if __name__ == "__main__":
    main()
