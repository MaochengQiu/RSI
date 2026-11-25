"""Evaluate two frozen weight candidates on a disjoint set of WebShop tasks."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import random
import re
import subprocess
import sys

import torch

spec = spec_from_file_location("webshop_pilot", Path(__file__).with_name("run_experiment.py"))
pilot = module_from_spec(spec)
spec.loader.exec_module(pilot)
runner = pilot.runner
read_json = pilot.sweep.read_json
SAMPLE_SEED = 20261123
N_GOALS = 40
CANDIDATE_SEEDS = (20261121, 20261122)
SCALE = 0.05


def instruction_key(goal):
    text = re.sub(r", and price lower than .* dollars$", "", goal["instruction_text"].lower())
    return tuple(re.findall(r"\w+", text))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("--checkout", "--model-dir", "--pilot-dir", "--weight-reference", "--sweep-reference", "--output"):
        parser.add_argument(option, required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout))
    from web_agent_site.engine import engine
    from web_agent_site.engine.goal import get_reward
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

    previous = read_json(args.pilot_dir / "manifest.json")
    verified = read_json(args.weight_reference / "weight-verification.json")
    stored = {c["name"]: c for c in read_json(args.sweep_reference / "results.json")["candidates"]}
    conditions = []
    for seed in CANDIDATE_SEEDS:
        name = f"seed-{seed}-scale-{SCALE:.2f}"
        conditions.append({"name": name, "seed": seed, "scale": SCALE,
                           "weight_sha256": stored[name]["weights"]["weight_sha256"]})
    assert runner.file_sha(Path(pilot.__file__)) == previous["runner_sha256"]
    root = args.checkout
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    assert commit == previous["checkout"]
    assert pilot.fingerprint([root / key for key in previous["catalog_and_index"]], root) == previous["catalog_and_index"]
    engine.DEFAULT_ATTR_PATH = str(root / "data/items_ins_v2_10000.json")
    engine.HUMAN_ATTR_PATH = str(root / "data/items_human_ins_10000.json")
    random.seed(previous["goal_and_sample_seed"])
    env = WebAgentTextEnv(observation_mode="text", human_goals=True,
                          file_path=str(root / "data/items_shuffle_10000.json"))
    assert env.server.product_prices == previous["prices"]
    for task in previous["tasks"]:
        assert env.server.goals[task["goal_id"]] == task["goal"]
    excluded_ids = {0, *(task["goal_id"] for task in previous["tasks"])}
    excluded_asins = {env.server.goals[i]["asin"] for i in excluded_ids}
    excluded_texts = {instruction_key(env.server.goals[i]) for i in excluded_ids}
    eligible = [i for i, goal in enumerate(env.server.goals) if
                i not in excluded_ids and goal["asin"] not in excluded_asins and
                instruction_key(goal) not in excluded_texts]
    ids = sorted(random.Random(SAMPLE_SEED).sample(eligible, N_GOALS))
    tasks = [{"goal_id": i, "goal": env.server.goals[i]} for i in ids]
    manifest = {
        "pilot_manifest_sha256": runner.file_sha(args.pilot_dir / "manifest.json"),
        "holdout_runner_sha256": runner.file_sha(Path(__file__)),
        "pilot_runner_sha256": previous["runner_sha256"],
        "canonical_baseline_weight": verified["original_sha256"],
        "candidates": conditions, "sample_seed": SAMPLE_SEED, "eligible_count": len(eligible),
        "excluded_goal_ids": sorted(excluded_ids), "excluded_target_asins": sorted(excluded_asins),
        "tasks": tasks, "webshop_revision": commit, "max_rounds": pilot.MAX_ROUNDS,
        "model_revision": runner.MODEL_REVISION, "catalog_and_index": previous["catalog_and_index"],
        "metrics": {
            "primary": "paired mean official score, with zero for no purchase",
            "success": "official score equals 1; also report purchase and positive-score transitions",
            "timing": "checkout at full score; no checkout after a full-score state; score below a previously "
                      "seen state; lower checkout score followed by a higher purchase on the paired "
                      "trajectory from the same pre-checkout state",
            "cost": "decision rounds, accepted WebShop calls, searches, input tokens and elapsed seconds",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "manifest.json"
    if path.exists():
        assert read_json(path) == manifest
    else:
        pilot.persist(path, manifest)
    print("Frozen holdout IDs:", ids, "from", len(eligible), "eligible goals", flush=True)
    model, tokenizer, layer, load_stats = runner.load_model(args.model_dir)
    pilot.persist(args.output / "model-load.json", load_stats)
    original = layer.weight.detach().cpu().clone()
    assert runner.tensor_sha(original) == verified["original_sha256"]
    original_float = original.float()
    rms = original_float.square().mean().sqrt().item()
    for condition in [{"name": "baseline"}, *conditions]:
        name = condition["name"]
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        try:
            if "seed" in condition:
                generator = torch.Generator(device="cpu").manual_seed(condition["seed"])
                noise = torch.randn(original.shape, generator=generator, dtype=torch.float32)
                perturbed = (original_float + noise * (rms * SCALE)).to(original.dtype)
                layer.weight.copy_(perturbed)
                assert runner.tensor_sha(layer.weight) == condition["weight_sha256"]
            totals = []
            for task in tasks:
                path = folder / f"{task['goal_id']:04d}.jsonl"
                checkpoint = path.with_suffix(".json")
                if checkpoint.exists():
                    result = read_json(checkpoint)
                else:
                    result = pilot.episode(env, model, tokenizer, task, path, get_reward)
                    print(name, task["goal_id"], result["bought"], result["reward"], flush=True)
                totals.append(result)
            pilot.persist(folder / "summary.json", pilot.summarize_baseline(totals))
        finally:
            layer.weight.copy_(original)
            assert runner.tensor_sha(layer.weight) == verified["original_sha256"]


if __name__ == "__main__":
    main()
