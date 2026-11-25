"""Frozen comparison of baseline, 5% and absolute sigma 0.003 on unused goals."""

import argparse
from pathlib import Path
import random
import subprocess
import sys

import torch

from run_holdout import instruction_key, pilot, read_json, runner


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, results = args.checkout, args.results_root
    sys.path.insert(0, str(root))
    from web_agent_site.engine import engine
    from web_agent_site.engine.goal import get_reward
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

    ws = results / "webshop"
    dev = read_json(ws / "pilot-03/manifest.json")
    previous = read_json(ws / "holdout-01/manifest.json")
    strength = read_json(ws / "strength-01/manifest.json")
    assert runner.file_sha(Path(pilot.__file__)) == dev["runner_sha256"]
    assert runner.file_sha(Path(runner.__file__)) == strength["model_runner_sha256"]
    assert runner.file_sha(ws / "pilot-03/manifest.json") == previous["pilot_manifest_sha256"] == strength["pilot_manifest_sha256"]
    assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == dev["checkout"]
    assert pilot.fingerprint([root / k for k in dev["catalog_and_index"]], root) == dev["catalog_and_index"]
    engine.DEFAULT_ATTR_PATH = str(root / "data/items_ins_v2_10000.json")
    engine.HUMAN_ATTR_PATH = str(root / "data/items_human_ins_10000.json")
    random.seed(dev["goal_and_sample_seed"])
    env = WebAgentTextEnv(observation_mode="text", human_goals=True,
                          file_path=str(root / "data/items_shuffle_10000.json"))
    assert env.server.product_prices == dev["prices"]
    goals = env.server.goals
    used = {0}
    for task in dev["tasks"] + previous["tasks"]:
        assert goals[task["goal_id"]] == task["goal"]
        used.add(task["goal_id"])
    asins = {goals[i]["asin"] for i in used}
    keys = {instruction_key(goals[i]) for i in used}
    tasks = []
    for i, goal in enumerate(goals):
        if i not in used and goal["asin"] not in asins and instruction_key(goal) not in keys:
            tasks.append({"goal_id": i, "goal": goal})
            asins.add(goal["asin"])
            keys.add(instruction_key(goal))
    assert tasks
    archived = {c["name"]: c["weights"] for c in read_json(results / "alfworld/sweep-01/results.json")["candidates"]}
    conditions = [{"name": "baseline"}]
    for seed in strength["seeds"]:
        low = f"seed-{seed}-scale-0.05"
        high = f"seed-{seed}-sigma-0.003"
        for name, weights, sigma in ((low, archived[low], archived[low]["sigma"]),
                                      (high, read_json(ws / "strength-01" / high / "weights.json"), 0.003)):
            conditions.append({"name": name, "seed": seed, "sigma": sigma, "sha256": weights["weight_sha256"]})
    ref = results / "alfworld/compare-01"
    original_hash = read_json(ref / "weight-verification.json")["original_sha256"]
    manifest = {"tasks": tasks, "conditions": conditions, "original_sha256": original_hash,
                "excluded_goal_ids": sorted(used), "selection": "All remaining, first ID per target ASIN and normalized instruction",
                "pilot_manifest_sha256": strength["pilot_manifest_sha256"],
                "prior_holdout_sha256": runner.file_sha(ws / "holdout-01/manifest.json"),
                "script_sha256": runner.file_sha(Path(__file__)),
                "primary": "Mean official terminal reward across three frozen directions, paired by goal",
                "contrasts": ["high minus baseline", "low minus baseline", "high minus low within direction"],
                "secondary": ["purchases and full-score gains/losses", "first action divergence with identical prompt and state: buy vs continue, compare terminal reward",
                              "first non-checkout divergence", "mean environment calls", "full-score checkout after reaching a full-score state"]}
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        assert read_json(out / "manifest.json") == manifest
    else:
        pilot.persist(out / "manifest.json", manifest)
    print("Frozen task IDs:", [t["goal_id"] for t in tasks], flush=True)
    model, tokenizer, layer, stats = runner.load_model(args.model_dir)
    pilot.persist(out / "model-load.json", stats)
    original = layer.weight.detach().cpu().clone()
    assert runner.tensor_sha(original) == original_hash
    ids = torch.tensor(read_json(ref / "diagnostic-input.json"), device="cuda")
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    reference = runner.diagnostic(model, layer, inputs)

    def forward():
        return runner.compare_diagnostics(reference, runner.diagnostic(model, layer, inputs))

    for condition in conditions:
        folder = out / condition["name"]
        folder.mkdir(exist_ok=True)
        try:
            if "seed" in condition:
                noise = torch.randn(original.shape, generator=torch.Generator().manual_seed(condition["seed"]))
                layer.weight.copy_((original.float() + noise * condition["sigma"]).to(original.dtype))
                assert runner.tensor_sha(layer.weight) == condition["sha256"]
            diagnostic = forward()
            assert diagnostic["layer_input"]["max_abs"] == 0
            assert (diagnostic["layer_output"]["rms"] > 0) == ("seed" in condition)
            if "seed" not in condition:
                assert all(v["max_abs"] == 0 for v in diagnostic.values())
            pilot.persist(folder / "forward.json", diagnostic)
            for t in tasks:
                path = folder / f"{t['goal_id']:04d}.jsonl"
                if not path.with_suffix(".json").exists():
                    pilot.episode(env, model, tokenizer, t, path, get_reward)
            print(condition["name"], len(tasks), "complete", flush=True)
        finally:
            layer.weight.copy_(original)
            assert runner.tensor_sha(layer.weight) == original_hash
            restored = forward()
            assert all(v["max_abs"] == 0 for v in restored.values())
            pilot.persist(folder / "restored.json", restored)


if __name__ == "__main__":
    main()
