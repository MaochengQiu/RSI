"""Run two absolute Gaussian noise scales on the original 20 WebShop tasks."""

import argparse
from pathlib import Path
import random
import subprocess
import sys

import torch

from run_holdout import pilot, read_json, runner


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "model-dir", "pilot-dir", "weight-reference", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    root = args.checkout
    sys.path.insert(0, str(root))
    from web_agent_site.engine import engine
    from web_agent_site.engine.goal import get_reward
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

    previous = read_json(args.pilot_dir / "manifest.json")
    tasks = previous["tasks"]
    assert len(tasks) == 20 and len({t["goal_id"] for t in tasks}) == 20
    assert runner.file_sha(Path(pilot.__file__)) == previous["runner_sha256"]
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    assert commit == previous["checkout"]
    assert pilot.fingerprint([root / k for k in previous["catalog_and_index"]], root) == previous["catalog_and_index"]
    assert (runner.MODEL_REVISION, runner.TARGET + ".weight", pilot.MAX_ROUNDS, pilot.PROMPT,
            runner.MAX_INPUT_TOKENS, runner.MAX_NEW_TOKENS) == tuple(previous[k] for k in
            ("model_revision", "target", "max_rounds", "prompt", "input_tokens", "max_new_tokens"))
    engine.DEFAULT_ATTR_PATH = str(root / "data/items_ins_v2_10000.json")
    engine.HUMAN_ATTR_PATH = str(root / "data/items_human_ins_10000.json")
    random.seed(previous["goal_and_sample_seed"])
    env = WebAgentTextEnv(observation_mode="text", human_goals=True,
                          file_path=str(root / "data/items_shuffle_10000.json"))
    assert env.server.product_prices == previous["prices"]
    assert all(env.server.goals[t["goal_id"]] == t["goal"] for t in tasks)
    args.output.mkdir(parents=True, exist_ok=True)
    diagnostic_path = args.weight_reference / "diagnostic-input.json"
    baseline_dir = args.pilot_dir / "baseline"
    manifest = {"pilot_manifest_sha256": runner.file_sha(args.pilot_dir / "manifest.json"),
                "script_sha256": runner.file_sha(Path(__file__)),
                "model_runner_sha256": runner.file_sha(Path(runner.__file__)),
                "diagnostic_input_sha256": runner.file_sha(diagnostic_path),
                "baseline_hashes": pilot.fingerprint(baseline_dir.glob("*.json*"), baseline_dir),
                "goal_ids": [t["goal_id"] for t in tasks], "seeds": previous["weight_seeds"],
                "absolute_sigmas": [0.002, 0.003], "primary": "paired mean terminal reward"}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        assert read_json(manifest_path) == manifest
    else:
        pilot.persist(manifest_path, manifest)
    model, tokenizer, layer, stats = runner.load_model(args.model_dir)
    pilot.persist(args.output / "model-load.json", stats)
    original = layer.weight.detach().cpu().clone()
    original_hash = runner.tensor_sha(original)
    assert original_hash == read_json(args.weight_reference / "weight-verification.json")["original_sha256"]
    original_float = original.float()
    rms = original_float.square().mean().sqrt().item()
    ids = torch.tensor(read_json(diagnostic_path), device="cuda")
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    reference = runner.diagnostic(model, layer, inputs)

    def check_forward():
        return runner.compare_diagnostics(reference, runner.diagnostic(model, layer, inputs))

    repeat = check_forward()
    assert all(metric["max_abs"] == 0 for metric in repeat.values())
    pilot.persist(args.output / "baseline-forward.json", repeat)
    first_id = tasks[0]["goal_id"]
    replay = pilot.episode(env, model, tokenizer, tasks[0], args.output / "baseline-replay.jsonl", get_reward)
    replay_pair = pilot.compare(read_json(baseline_dir / f"{first_id:04d}.json"), replay)
    assert replay_pair["first_different_action"] is None and replay_pair["delta"] == 0
    print("Baseline checks passed. Running six candidates.", flush=True)
    for seed in manifest["seeds"]:
        noise = torch.randn(original.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
        for sigma in manifest["absolute_sigmas"]:
            name = f"seed-{seed}-sigma-{sigma:.3f}"
            folder = args.output / name
            folder.mkdir(exist_ok=True)
            try:
                changed = (original_float + sigma * noise).to(original.dtype)
                layer.weight.copy_(changed)
                actual = layer.weight.detach().cpu()
                digest = runner.tensor_sha(actual)
                assert digest == runner.tensor_sha(changed) and digest != original_hash
                forward = check_forward()
                assert forward["layer_input"]["max_abs"] == 0 and forward["layer_output"]["rms"] > 0
                pilot.persist(folder / "weights.json", {
                    "weight_sha256": digest, "original_sha256": original_hash, "absolute_sigma": sigma,
                    "original_rms": rms, "relative_frobenius_delta":
                    ((actual.float() - original_float).norm() / original_float.norm()).item(),
                    "forward": forward,
                })
                pairs = []
                for task in tasks:
                    goal_id = task["goal_id"]
                    path = folder / f"{goal_id:04d}.jsonl"
                    checkpoint = path.with_suffix(".json")
                    result = read_json(checkpoint) if checkpoint.exists() else pilot.episode(
                        env, model, tokenizer, task, path, get_reward)
                    pairs.append(pilot.compare(read_json(baseline_dir / checkpoint.name), result))
                    print(name, goal_id, result["reward"], flush=True)
                pilot.persist(folder / "paired.json", pairs)
            finally:
                layer.weight.copy_(original)
                assert runner.tensor_sha(layer.weight) == original_hash
                restored = check_forward()
                assert all(metric["max_abs"] == 0 for metric in restored.values())
                pilot.persist(folder / "restored-forward.json", restored)


if __name__ == "__main__":
    main()
