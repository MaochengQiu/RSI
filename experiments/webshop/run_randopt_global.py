"""Run one fixed, KL-matched full-model perturbation exploration."""

import argparse
from contextlib import contextmanager
from importlib.metadata import version
import math
from pathlib import Path
import random
from statistics import mean
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn.functional import linear, log_softmax

from analyze_randopt_followup import audit, decision, direction_summary, paired, score, table
from randopt_followup import LAYOUT, catalog_hashes, environment, price_hash, run_episodes, save
from randopt_scope import calibration_inputs, logits
from replay_checkout_holdout import opportunities
from run_holdout import pilot, read_json, runner

SEEDS = random.Random(2026112107).sample(range(1, 2**31), 8)
CACHE_BYTES = 1536 * 1024**2


def layout(model):
    quantized = []
    for name, module in model.named_modules():
        if isinstance(module, runner.bnb.nn.Linear4bit):
            assert module.bias is None
            quantized.append((name, module, (module.out_features, module.in_features)))
    floating = [(name, weight) for name, weight in model.named_parameters()
                if not isinstance(weight, runner.bnb.nn.Params4bit)]
    assert all(weight.dtype == torch.float16 for _, weight in floating)
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert sum(math.prod(shape) for _, _, shape in quantized) + sum(w.numel() for _, w in floating) > 3.9e9
    return quantized, floating


def gaussian(shape, generator, seed, sigma, dtype):
    return torch.randn(shape, device="cuda", generator=generator.manual_seed(seed),
                       dtype=torch.float32).mul_(sigma).to(dtype)


@contextmanager
def perturb(model_layout, originals, seed, sigma):
    quantized, floating = model_layout
    streams = random.Random(seed)
    generator = torch.Generator(device="cuda")
    hooks, calls = [], set()
    cached_bytes = 0

    def hook(name, shape, local_seed, cached):
        def add(_module, args, output):
            calls.add(name)
            if not sigma:
                return output + torch.zeros_like(output)
            delta = cached if cached is not None else gaussian(shape, generator, local_seed, sigma, torch.float16)
            return output + linear(args[0].to(delta.dtype), delta).to(output.dtype)
        return add

    try:
        for name, weight in floating:
            if not sigma:
                continue
            generator.manual_seed(streams.getrandbits(63))
            if weight.ndim == 1:
                delta = torch.randn(weight.shape, device="cuda", generator=generator)
                weight.copy_(weight.float().add_(delta, alpha=sigma))
            else:
                for chunk in weight.split(2048):
                    delta = torch.randn(chunk.shape, device="cuda", generator=generator)
                    chunk.copy_(chunk.float().add_(delta, alpha=sigma))
        for name, module, shape in quantized:
            local_seed = streams.getrandbits(63)
            required = math.prod(shape) * 2
            cached = None
            if sigma and cached_bytes + required <= CACHE_BYTES:
                cached = gaussian(shape, generator, local_seed, sigma, torch.float16)
                cached_bytes += required
            hooks.append(module.register_forward_hook(hook(name, shape, local_seed, cached)))
        yield calls
    finally:
        for entry in hooks:
            entry.remove()
        if sigma:
            for name, weight in floating:
                weight.copy_(originals[name])


def token_kl(model, prompts, references):
    total, count = torch.zeros((), device="cuda", dtype=torch.float64), 0
    for prompt, reference in zip(prompts, references):
        log_p = reference.to("cuda")
        log_q = log_softmax(logits(model, prompt).double(), dim=-1)
        total += (log_p.exp() * (log_p - log_q)).sum()
        count += log_p.shape[0]
    value = total.item() / count
    if not math.isfinite(value):
        raise ValueError("Nonfinite Token KL")
    return max(value, 0.0)


def fit(measure, target):
    minimum, maximum, sigma = 0.00001171875, 0.192, 0.0001875
    low = high = sigma
    current = measure(sigma)
    points = {sigma: current}
    while abs(current / target - 1) > .10:
        if current < target:
            low = sigma
            sigma = min(maximum, sigma * 2)
            move_again = sigma > low
        else:
            high = sigma
            sigma = max(minimum, sigma / 2)
            move_again = sigma < high
        if not move_again or sigma in points:
            break
        current = measure(sigma)
        points[sigma] = current
        if points.get(low, math.inf) <= target <= points.get(high, -math.inf):
            break
        if (current >= target and sigma > low) or (current <= target and sigma < high):
            low, high = sorted((sigma, low if current >= target else high))
            break
    if points.get(low, math.inf) <= target <= points.get(high, -math.inf):
        for _ in range(8):
            if min(abs(value / target - 1) for value in points.values()) <= .10:
                break
            mid = (low + high) / 2
            points[mid] = measure(mid)
            if points[mid] <= target:
                low = mid
            else:
                high = mid
    return min(points, key=lambda s: abs(points[s] / target - 1)), sorted(points.items())


def valid_action(line, state):
    match = pilot.COMMAND.fullmatch(line)
    if not match:
        return line.lower() != "search" and line.lower() in state["clickables"]
    action, value = match[1].lower(), match[2].strip().lower()
    if action == "search":
        return bool(value and state["page_type"] == "search")
    return value != "search" and value in state["clickables"]


def previews(results_root, ids):
    baseline = results_root / "webshop/pilot-03/baseline"
    for i in ids[:2]:
        yield pilot.sweep.read_trace(baseline / f"{i:04d}.jsonl")[0]
        yield next(opportunities(baseline, i))[0]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("checkout", "model-dir", "results-root"):
        parser.add_argument("--" + arg, required=True, type=Path)
    parser.add_argument("--first-probe", action="store_true")
    args = parser.parse_args()
    root = args.results_root
    previous = root / "webshop/randopt-followup-01"
    prior = root / "webshop" / LAYOUT
    catalog = root / "webshop-data/direction-01"
    output = root / "webshop/randopt-global-01"
    archive = read_json(prior / "manifest.json")
    assert set(SEEDS).isdisjoint(archive["seeds"] + read_json(previous / "scope/candidates.json"))
    assert runner.file_sha(Path(pilot.__file__)) == archive["pilot_sha256"]
    assert runner.file_sha(Path(runner.__file__)) == archive["model_runner_sha256"]
    assert archive["checkout"] == subprocess.check_output(["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True).strip()
    assert catalog_hashes(catalog) == archive["catalog_hashes"]
    sys.path.insert(0, str(args.checkout))
    from web_agent_site.engine.goal import get_reward

    env = environment(args, catalog, archive)
    assert price_hash(env) == archive["price_hash"]
    tasks = archive["splits"]["development"]
    assert len(tasks) == 24 and all(env.server.goals[t["goal_id"]] == t["goal"] for t in tasks)
    model, tokenizer, layer, stats = runner.load_model(args.model_dir)
    assert runner.tensor_sha(layer.weight) == archive["weight_hashes"]["baseline"]
    model_layout = layout(model)
    quantized, floating = model_layout
    originals = {name: weight.detach().cpu().clone() for name, weight in floating}
    targets = read_json(previous / "scope/targets.json")
    target, validation_target = targets["means"]
    entries = calibration_inputs(root, tokenizer)
    assert [i for i, _ in entries] == targets["prompt_goal_ids"]
    prompts = [p for _, pair in entries for p in pair]
    protocol = Path(__file__).with_name("RANDOPT-GLOBAL.zh.md")
    manifest = {"seeds": SEEDS, "tasks": [t["goal_id"] for t in tasks], "targets": targets,
                "protocol_sha256": runner.file_sha(protocol), "script_sha256": runner.file_sha(Path(__file__)),
                "archive_sha256": runner.file_sha(prior / "manifest.json"), "cache_bytes": CACHE_BYTES,
                "quantized_shapes": [[name, list(shape)] for name, _, shape in quantized],
                "floating_shapes": [[name, list(w.shape)] for name, w in floating],
                "tied_embedding": True, "model": {k: v for k, v in stats.items()
                if k not in ("elapsed_seconds", "peak_allocated_mib", "peak_reserved_mib")},
                "versions": {"torch": torch.__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                **{name: version(name) for name in ("transformers", "bitsandbytes", "accelerate")}}}
    if (output / "manifest.json").exists():
        assert read_json(output / "manifest.json") == manifest
    else:
        save(output / "manifest.json", manifest)
    baseline_logits = logits(model, prompts[0])
    references = [log_softmax(logits(model, p).double(), dim=-1).cpu() for p in prompts]
    with perturb(model_layout, originals, SEEDS[0], 0) as calls:
        assert torch.equal(logits(model, prompts[0]), baseline_logits)
        assert calls == {name for name, _, _ in quantized}
    print("Zero-delta check passed;", len(quantized), "NF4 matrices and", len(floating), "FP16 tensors", flush=True)
    samples = list(previews(root, targets["prompt_goal_ids"]))
    base_dir = prior / "development/baseline"
    for seed in SEEDS[:1] if args.first_probe else SEEDS:
        folder = output / str(seed)
        calibration = folder / "calibration.json"
        if calibration.exists():
            record = read_json(calibration)
        else:
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()

            def measure(sigma):
                with perturb(model_layout, originals, seed, sigma):
                    return token_kl(model, prompts[:24], references[:24])

            try:
                sigma, points = fit(measure, target)
                achieved = dict(points)[sigma]
                with perturb(model_layout, originals, seed, sigma) as calls:
                    validation = token_kl(model, prompts[24:], references[24:])
                    a, b = logits(model, prompts[0]), logits(model, prompts[0])
                    assert torch.equal(a, b) and calls == {name for name, _, _ in quantized}
                    assert not torch.equal(a, baseline_logits)
                    actions = []
                    for sample in samples:
                        raw, _, _ = pilot.sweep.replay_decision(model, tokenizer, sample["prompt"])
                        line = runner.parse_line(raw)
                        actions.append({"line": line, "accepted": valid_action(line, sample["before"])})
                record = {"seed": seed, "sigma": sigma, "kl": achieved, "validation_kl": validation,
                          "points": points, "actions": actions,
                          "matched": abs(achieved / target - 1) <= .10 and abs(validation / validation_target - 1) <= .20
                          and sum(a["accepted"] for a in actions) >= 2}
            except ValueError as exc:
                record = {"seed": seed, "matched": False, "error": str(exc)}
            assert torch.equal(logits(model, prompts[0]), baseline_logits)
            record["performance"] = runner.performance(started)
            save(calibration, record)
        print("Calibration", seed, record.get("kl"), record.get("validation_kl"), record["matched"], flush=True)
        if record["matched"] and not args.first_probe:
            with perturb(model_layout, originals, seed, record["sigma"]):
                run_episodes(env, model, tokenizer, tasks, folder, get_reward, base_dir)
            assert torch.equal(logits(model, prompts[0]), baseline_logits)
            print("Completed", seed, len(tasks), flush=True)
    if not args.first_probe:
        summarize(output, previous, tasks, base_dir)


def summarize(output, previous, tasks, base_dir):
    baseline = audit(base_dir, tasks)
    records, names, deltas, episodes, rows = {}, [], [], [], []
    for seed in SEEDS:
        folder = output / str(seed)
        record = read_json(folder / "calibration.json")
        records[str(seed)] = record
        if record["matched"]:
            group = audit(folder, tasks)
            episodes.extend(group)
            names.append(seed)
            deltas.append(score(group) - score(baseline))
            rows.extend(paired(baseline, group, seed))
    d = decision(np.array(deltas).T, names, 8 - len(names))
    d.pop("nominee")
    d.pop("status")
    for seed in names:
        d["directions"][str(seed)].update(direction_summary([r for r in rows if r["seed"] == seed]))
    data = {"episode_count": len(episodes), "calibration": records, **d,
            "baseline_mean": mean(r["reward"] for r in baseline),
            "episode_seconds": sum(r["elapsed_seconds"] for r in episodes),
            "episode_peak_allocated_mib": max((r["peak_allocated_mib"] for r in episodes), default=0),
            "historical": {k: read_json(previous / "analysis/scope.json")[k] for k in ("old_single", "directions")}}
    save(output / "summary.json", data)
    if rows:
        table(output / "paired.csv", rows)


if __name__ == "__main__":
    main()
