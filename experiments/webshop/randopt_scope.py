"""Apply floating-point v_proj deltas over the unchanged NF4 model and match token KL."""

from contextlib import contextmanager
import random

import torch
from torch.nn.functional import linear, log_softmax

from replay_checkout_holdout import opportunities
from run_direction_selection import change
from randopt_followup import run_episodes, save
from run_holdout import pilot, read_json, runner


def projections(model):
    return [block.self_attn.v_proj for block in model.model.layers]


def direction(model, seed):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn((layer.out_features, layer.in_features), generator=generator, dtype=torch.float32)
            for layer in projections(model)]


@contextmanager
def apply_scope(model, seed, sigma, noise=None):
    if noise is None:
        noise = direction(model, seed)
    hooks, calls = [], set()

    def install(layer, index, delta):
        def add(_module, args, output):
            calls.add(index)
            return output + linear(args[0].to(delta.dtype), delta).to(output.dtype)
        return layer.register_forward_hook(add)

    try:
        for i, (layer, weights) in enumerate(zip(projections(model), noise)):
            delta = (weights * sigma).to(device="cuda", dtype=torch.float16)
            hooks.append(install(layer, i, delta))
        yield calls
    finally:
        for hook in hooks:
            hook.remove()


def calibration_inputs(results_root, tokenizer):
    folder = results_root / "webshop/pilot-03/baseline"
    examples = []
    for path in sorted(folder.glob("????.jsonl")):
        points = list(opportunities(folder, int(path.stem)))
        if not points:
            continue
        turns = pilot.sweep.read_trace(path)
        prompts = [turns[0]["prompt"], points[0][0]["prompt"]]
        tokens = [tokenizer(p, return_tensors="pt", add_special_tokens=False) for p in prompts]
        examples.append((int(path.stem), tokens))
        if len(examples) == 16:
            break
    assert len(examples) == 16
    return examples


def logits(model, tokens):
    args = {k: v.to("cuda") for k,v in tokens.items()}
    return model(**args, use_cache=False, logits_to_keep=16).logits.squeeze(0).float()


def kl(model, tokens, references):
    combined, n = torch.zeros((), dtype=torch.float64, device="cuda"), 0
    for prompt, log_p in zip(tokens, references):
        log_q = log_softmax(logits(model, prompt).double(), dim=-1)
        combined += (log_p.exp() * (log_p - log_q)).sum()
        n += log_p.shape[0]
    return max(0.0, combined.item() / n)


def calibrate(measure, target):
    low, high, sigma = 0.00001171875, 0.192, 0.003
    points = {}

    def evaluate(value):
        if value not in points:
            points[value] = measure(value)
        return points[value]

    current = evaluate(sigma)
    if current < target:
        while sigma < high and current < target:
            low = sigma
            sigma = min(high, sigma * 2)
            current = evaluate(sigma)
        high = sigma
    else:
        while sigma > low and current > target:
            high = sigma
            sigma = max(low, sigma / 2)
            current = evaluate(sigma)
        low = sigma
    if evaluate(low) <= target <= evaluate(high):
        for _ in range(8):
            best = min(points, key=lambda s: abs(points[s] - target))
            if abs(points[best] - target) <= 0.10 * target:
                break
            mid = (low + high) / 2
            if evaluate(mid) < target:
                low = mid
            else:
                high = mid
    sigma = min(points, key=lambda s: abs(points[s] - target))
    return sigma, points


def run_scope(args, prior, archive, out, env, model, tokenizer, score):
    task_entries = calibration_inputs(args.results_root, tokenizer)
    prompts = [t for _, pair in task_entries for t in pair]
    references = [log_softmax(logits(model, t).double(), dim=-1) for t in prompts]
    validation_at = 24
    seeds = random.Random(2026112106).sample(range(1, 2**31), 8)
    assert set(seeds).isdisjoint(prior["seeds"])
    scope = out / "scope"
    save(scope / "candidates.json", seeds)
    with apply_scope(model, seeds[0], 0.0) as calls:
        check = log_softmax(logits(model, prompts[0]).double(), dim=-1)
        assert torch.equal(check, references[0]) and len(calls) == len(projections(model))
    target_file = scope / "targets.json"
    layer = model.get_submodule(runner.TARGET)
    original = layer.weight.detach().cpu().clone()
    assert runner.tensor_sha(original) == prior["weight_hashes"]["baseline"]
    if target_file.exists():
        targets = read_json(target_file)
    else:
        points = []
        for seed in prior["seeds"]:
            try:
                layer.weight.copy_(change(original, seed))
                assert runner.tensor_sha(layer.weight) == prior["weight_hashes"][str(seed)]
                points.append([kl(model, prompts[:validation_at], references[:validation_at]),
                               kl(model, prompts[validation_at:], references[validation_at:])])
            finally:
                layer.weight.copy_(original)
        targets = {"per_seed": points, "means": [sum(p) / len(points) for p in zip(*points)],
                   "prompt_goal_ids": [i for i, _ in task_entries]}
        save(target_file, targets)
    target, check_target = targets["means"]
    assert target > 0 and check_target > 0
    print("Token KL targets", target, check_target, flush=True)
    for seed in seeds:
        folder = scope / str(seed)
        path = folder / "calibration.json"
        noise = direction(model, seed)
        if path.exists():
            assessment = read_json(path)
            assert [runner.tensor_sha(matrix) for matrix in noise] == assessment["direction_hashes"]
        else:
            def measure(sigma):
                with apply_scope(model, seed, sigma, noise):
                    return kl(model, prompts[:validation_at], references[:validation_at])

            sigma, points = calibrate(measure, target)
            with apply_scope(model, seed, sigma, noise):
                check = kl(model, prompts[validation_at:], references[validation_at:])
            assessment = {"seed": seed, "sigma": sigma, "calibration_kl": points[sigma], "check_kl": check,
                          "matched": abs(points[sigma] / target - 1) <= .10 and abs(check / check_target - 1) <= .20,
                          "evaluations": [[s,k] for s,k in sorted(points.items())],
                          "direction_hashes": [runner.tensor_sha(matrix) for matrix in noise]}
            save(path, assessment)
        print("KL", seed, assessment["calibration_kl"], assessment["check_kl"], assessment["matched"], flush=True)
        if assessment["matched"]:
            with apply_scope(model, seed, assessment["sigma"], noise):
                run_episodes(env, model, tokenizer, prior["splits"]["development"], folder, score,
                             archive / "development/baseline")
            assert torch.equal(log_softmax(logits(model, prompts[0]).double(), dim=-1), references[0])
            print("Scope", seed, "24 complete", flush=True)
