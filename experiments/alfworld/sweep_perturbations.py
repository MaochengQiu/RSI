"""Compare three fixed weight perturbation directions at two amplitudes."""

import argparse
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import time

import torch

import run_experiment as runner


SEEDS = (runner.SEED, runner.SEED + 1, runner.SEED + 2)
SCALES = (0.01, 0.05)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_trace(path):
    with path.open(encoding="utf-8") as stream:
        return list(map(json.loads, stream))


def replay_decision(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    count = inputs.input_ids.shape[1]
    stop = runner.StopAtLine(tokenizer)
    stop.prompt_length = count
    config = deepcopy(model.generation_config)
    config.do_sample = False
    config.num_beams = 1
    config.temperature = config.top_p = config.top_k = None
    config.max_new_tokens = runner.MAX_NEW_TOKENS
    result = model.generate(
        **inputs, generation_config=config, use_model_defaults=False,
        do_sample=False, num_beams=1, stopping_criteria=[stop],
        output_scores=True, return_dict_in_generate=True,
    )
    ids = result.sequences[0, count:].tolist()
    raw = tokenizer.decode(ids, skip_special_tokens=True)
    scores = torch.stack(result.scores)[:, 0].float().cpu()
    return raw, ids, scores


def compare_decision(tokenizer, baseline, candidate):
    raw, ids, scores = candidate
    _, original_ids, original_scores = baseline
    rows = []
    for index, (original_id, current_id) in enumerate(zip(original_ids, ids)):
        before, after = original_scores[index], scores[index]
        before_top = before.topk(2)
        after_top = after.topk(2)
        rival = after_top.indices[0] if current_id != original_id else after_top.indices[1]
        rows.append({
            "token_position": index + 1,
            "original_token_id": original_id,
            "original_token": tokenizer.decode([original_id]),
            "candidate_token_id": current_id,
            "candidate_token": tokenizer.decode([current_id]),
            "original_top2_margin": (before_top.values[0] - before_top.values[1]).item(),
            "candidate_top2_margin": (after_top.values[0] - after_top.values[1]).item(),
            "candidate_lead_of_original_token": (after[original_id] - after[rival]).item(),
        })
        if original_id != current_id:
            break  # Later tokens no longer share the same prefix.
    return {
        "raw_generation": raw,
        "token_ids": ids,
        "first_different_token": runner.first_difference(original_ids, ids),
        "shared_prefix_scores": rows,
    }


def compare_episode(baseline, baseline_trace, candidate, trace):
    if baseline["initial_observation"] != candidate["initial_observation"]:
        raise ValueError("Initial observation differs from the baseline.")
    first_line = runner.first_difference(
        [row["parsed_line"] for row in baseline_trace],
        [row["parsed_line"] for row in trace],
    )
    line_change = None
    if first_line:
        index = first_line["action_index"] - 1
        line_change = {
            "round": index + 1,
            "original": first_line["original"],
            "candidate": first_line["perturbed"],
            "original_observation": baseline_trace[index]["observation"] if index < len(baseline_trace) else None,
            "candidate_observation": trace[index]["observation"] if index < len(trace) else None,
        }
    return {
        "baseline_won": baseline["won"],
        "baseline_steps": baseline["steps"],
        "first_different_action": runner.first_difference(baseline["actions"], candidate["actions"]),
        "first_different_line": line_change,
        **candidate,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "rsi-data" / "alfworld")
    args = parser.parse_args()
    reference = read_json(args.reference / "configuration.json")
    verified = read_json(args.reference / "weight-verification.json")
    reference_results = read_json(args.reference / "results.json")
    prompts = args.data_dir / "react-alfworld_3prompts.json"
    scenes = reference["scenes"]
    current_versions = {package: importlib.metadata.version(package) for package in reference["versions"]}
    if (
        runner.file_sha(Path(runner.__file__)) != reference["runner_sha256"]
        or runner.file_sha(prompts) != reference["react_prompt_sha256"]
        or current_versions != reference["versions"]
        or any(runner.file_sha(args.data_dir / scene["id"]) != scene["sha256"] for scene in scenes)
    ):
        raise ValueError("Runner, prompts, games or package versions differ from the reference.")

    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    runner.write_json(args.output / "sweep-config.json", {
        "reference": str(args.reference),
        "reference_configuration_sha256": runner.file_sha(args.reference / "configuration.json"),
        "sweep_sha256": runner.file_sha(Path(__file__)),
        "seeds": SEEDS, "scales": SCALES,
        "model_dir": str(args.model_dir), "settings": reference,
    })
    model, tokenizer, layer, load_stats = runner.load_model(args.model_dir)
    original = layer.weight.detach().cpu().clone()
    original_hash = runner.tensor_sha(original)
    if original_hash != verified["original_sha256"] or str(model.config.quantization_config) != reference["quantization"]:
        raise ValueError("Loaded target weights or quantization differ from the reference.")
    originals = [row["original"] for row in reference_results["rows"]]
    traces = [read_trace(args.reference / episode["trace"]) for episode in originals]
    apple_trace = traces[2]
    failure_index = next(i for i, row in enumerate(apple_trace)
                         if (row["action"] or "").startswith("take apple")
                         and row["feedback"].strip() == "Nothing happens.")
    decision = apple_trace[failure_index + 1]
    baseline_replay = replay_decision(model, tokenizer, decision["prompt"])
    if baseline_replay[0] != decision["raw_generation"]:
        raise RuntimeError("Baseline replay did not reproduce the next line.")
    diagnostic_input = torch.tensor(read_json(args.reference / "diagnostic-input.json"), device="cuda")
    baseline_forward = runner.diagnostic(model, layer, {
        "input_ids": diagnostic_input, "attention_mask": torch.ones_like(diagnostic_input),
    })
    runner.write_json(args.output / "decision-input.json", {
        "scene": scenes[2], "round": decision["round"],
        "previous_action": apple_trace[failure_index]["action"],
        "feedback": apple_trace[failure_index]["feedback"],
        "prompt": decision["prompt"],
        "baseline": compare_decision(tokenizer, baseline_replay, baseline_replay),
    })
    examples = runner.adapt_examples(read_json(prompts))
    candidates = []
    for seed in SEEDS:
        direction = torch.randn(original.shape, generator=torch.Generator().manual_seed(seed))
        for scale in SCALES:
            name = f"seed-{seed}-scale-{scale:.2f}"
            candidate_dir = args.output / name
            candidate_dir.mkdir()
            print(f"{name}: {scale:.0%} target weight perturbation", flush=True)
            sigma = original.float().square().mean().sqrt().item() * scale
            perturbed = (original.float() + direction * sigma).to(original.dtype)
            reuse = (seed, scale) == (runner.SEED, runner.NOISE_SCALE)
            try:
                layer.weight.copy_(perturbed)
                actual = layer.weight.detach().cpu()
                weight_hash = runner.tensor_sha(actual)
                if reuse and weight_hash != verified["perturbed_sha256"]:
                    raise RuntimeError("The recreated candidate differs from the reference perturbation.")
                weights = {
                    "sigma": sigma,
                    "relative_frobenius_delta": ((actual.float() - original.float()).norm() / original.float().norm()).item(),
                    "weight_sha256": weight_hash,
                    "forward_difference": runner.compare_diagnostics(
                        baseline_forward, runner.diagnostic(model, layer, {
                            "input_ids": diagnostic_input, "attention_mask": torch.ones_like(diagnostic_input),
                        }),
                    ),
                }
                if weights["forward_difference"]["layer_output"]["rms"] <= 0.0:
                    raise RuntimeError("Perturbation did not change the target layer output.")
                replay = compare_decision(
                    tokenizer, baseline_replay, replay_decision(model, tokenizer, decision["prompt"]),
                )
                runner.write_json(candidate_dir / "decision-replay.json", replay)
                episodes = []
                for i, scene in enumerate(scenes):
                    if reuse:
                        outcome = reference_results["rows"][i]["perturbed"]
                        path = args.reference / outcome["trace"]
                    else:
                        print(f"  {scene['task']}", flush=True)
                        path = candidate_dir / f"{i + 1:02d}.jsonl"
                        outcome = runner.episode(model, tokenizer, examples, args.data_dir, scene, path)
                    comparison = compare_episode(originals[i], traces[i], outcome, read_trace(path))
                    comparison["trace"] = str(path)
                    episodes.append({"scene": scene, **comparison})
            finally:
                layer.weight.copy_(original)
            if runner.tensor_sha(layer.weight) != original_hash:
                raise RuntimeError("Candidate did not restore the original weight.")
            record = {
                "name": name, "seed": seed, "scale": scale, "reused": reuse,
                "restored_weight_sha256": original_hash, "weights": weights,
                "decision": replay, "episodes": episodes,
            }
            runner.write_json(candidate_dir / "results.json", record)
            candidates.append(record)
    restored_replay = replay_decision(model, tokenizer, decision["prompt"])
    restored_matches = (
        baseline_replay[1] == restored_replay[1] and torch.equal(baseline_replay[2], restored_replay[2])
    )
    runner.write_json(args.output / "results.json", {
        "candidates": candidates, "restored_decision_matches": restored_matches,
        "wall_seconds": time.perf_counter() - start, "model_load": load_stats,
    })
    if not restored_matches:
        raise RuntimeError("The restored model did not reproduce the baseline replay scores.")


if __name__ == "__main__":
    main()
