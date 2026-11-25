"""Evaluate the frozen global candidate and baseline on the sealed 60 tasks."""

import argparse
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path
from statistics import mean
import subprocess
import sys

import numpy as np
import torch
from torch.nn.functional import log_softmax

from analyze_randopt_followup import DELTA, audit, direction_summary, paired, score, submeans, table
from randopt_followup import SEAL, catalog_hashes, check_commit, environment, price_hash, run_episodes, save
from randopt_scope import calibration_inputs, logits
from run_holdout import pilot, read_json, runner
from run_randopt_global import layout, perturb

SEED = 1732518584
SIGMA = 0.0002578125
ORIGINAL_SCRIPT = "4974f5c5b87b02753559e61d214e2602b43ca3411658f7629f81409548cf5ffe"
ORIGINAL_PROTOCOL = "083fa00b088ef046e8574bc39bfda45b39f51caec4014aa29ecd3633c66911bf"
SHORTENED = "[Page text shortened to fit context.]"


def context_tokens(tokenizer, text):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return len(tokenizer(prompt, add_special_tokens=False).input_ids)


def compact_page(tokenizer, base, current):
    limit = runner.MAX_INPUT_TOKENS
    if context_tokens(tokenizer, base + current) <= limit:
        return current
    page, separator, menu = current.rpartition("\nAllowed actions: ")
    assert separator and menu.endswith("\nAction:")
    header, delimiter, observation = page.partition(":\n")
    assert delimiter and header.startswith("\nCurrent page (")
    observation, _, correction = observation.partition("\nLast command was rejected. ")
    suffix = ("\nLast command was rejected. " + correction) if correction else ""

    def compose(text, actions):
        return header + delimiter + text + "\n" + SHORTENED + suffix + separator + actions + "\nAction:"

    menu = menu.removesuffix("\nAction:")
    if context_tokens(tokenizer, base + compose("", menu)) > limit:
        rest = menu.removeprefix("click[").split(", click[")
        actions = ["click[" + value for value in rest]
        assert all(action.endswith("]") for action in actions)
        navigation = {"click[buy now]", "click[< prev]", "click[back to search]", "click[next >]"}
        ordered = sorted(actions, key=lambda action: action not in navigation)
        kept = []
        for action in ordered:
            trial = ", ".join([*kept, action])
            if context_tokens(tokenizer, base + compose("", trial)) > limit:
                break
            kept.append(action)
        menu = ", ".join(kept)
        assert menu
    tokens = tokenizer(observation, add_special_tokens=False).input_ids
    low, high = 0, len(tokens)
    result = compose("", menu)
    while low <= high:
        middle = (low + high) // 2
        head = (middle + 1) // 2
        tail = middle - head
        clipped = tokens[:head] + (tokens[-tail:] if tail else [])
        text = tokenizer.decode(clipped, clean_up_tokenization_spaces=False)
        trial = compose(text, menu)
        if context_tokens(tokenizer, base + trial) <= limit:
            result = trial
            low = middle + 1
        else:
            high = middle - 1
    return result


def analyze(output, tasks):
    base, candidate = [audit(output / name, tasks) for name in ("baseline", "winner")]
    rows = paired(base, candidate, SEED)
    changes = score(candidate) - score(base)
    ci = np.quantile(submeans(changes, 100000), [.025, .975]).tolist()
    delta = float(changes.mean())
    extras = direction_summary(rows)
    metrics = {}
    for name, episodes in (("baseline", base), ("winner", candidate)):
        metrics[name] = {
            "score": mean(r["reward"] for r in episodes),
            "bought": sum(r["bought"] for r in episodes),
            "full": sum(r["reward"] == 1 for r in episodes),
            "mean_calls": mean(r["calls"] for r in episodes),
            "mean_rounds": mean(r["rounds"] for r in episodes),
            "mean_input_tokens": mean(r["input_tokens"] for r in episodes),
            "mean_search_calls": mean(r["search_calls"] for r in episodes),
            "seconds": sum(r["elapsed_seconds"] for r in episodes),
            "peak_allocated_mib": max(r["peak_allocated_mib"] for r in episodes),
            "peak_reserved_mib": max(r["peak_reserved_mib"] for r in episodes),
            "shortened_prompts": sum(SHORTENED in turn["prompt"] for r in episodes for turn in r["trace_rows"]),
            "shortened_goal_ids": [r["goal_id"] for r in episodes if any(
                SHORTENED in turn["prompt"] for turn in r["trace_rows"])],
        }
    save(output / "summary.json", {"n": len(tasks), "seed": SEED, "sigma": SIGMA, "mean_delta": delta,
                                  "paired_ci95": ci, "meets_original_effect_and_interval_threshold":
                                  round(delta, 12) >= DELTA and ci[0] > 0,
                                  **metrics, **extras, "unchanged_score": len(rows) - extras["up"] - extras["down"],
                                  "purchase_gains": sum(not r["left_bought"] and r["right_bought"] for r in rows),
                                  "purchase_losses": sum(r["left_bought"] and not r["right_bought"] for r in rows)})
    table(output / "paired.csv", rows)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "model-dir", "results-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    root = args.results_root
    global_run = root / "webshop/randopt-global-01"
    output = root / "webshop/randopt-global-final-01"
    seal = read_json(SEAL)
    tasks = seal["tasks"]
    assert len(tasks) == len({t["goal_id"] for t in tasks}) == len({t["goal"]["asin"] for t in tasks}) == 60
    if args.analyze_only:
        assert read_json(output / "manifest.json")["seal_sha256"] == runner.file_sha(SEAL)
        analyze(output, tasks)
        return
    previous = read_json(global_run / "manifest.json")
    protocol = Path(__file__).with_name("RANDOPT-FINAL.zh.md")
    implementation = Path(__file__).with_name("run_randopt_global.py")
    assert runner.file_sha(implementation) == previous["script_sha256"]
    calibration_file = global_run / str(SEED) / "calibration.json"
    calibration = read_json(calibration_file)
    assert calibration["matched"] and calibration["sigma"] == SIGMA
    development = read_json(global_run / "summary.json")["directions"]
    assert str(SEED) == max(development, key=lambda key: development[key]["mean"])
    archive_file = root / "webshop/direction-selection-01/manifest.json"
    assert runner.file_sha(archive_file) == seal["archived_manifest_sha256"] == previous["archive_sha256"]
    archive = read_json(archive_file)
    assert runner.file_sha(Path(pilot.__file__)) == archive["pilot_sha256"]
    assert runner.file_sha(Path(runner.__file__)) == archive["model_runner_sha256"]
    assert subprocess.check_output(["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True).strip() == archive["checkout"]
    catalog = root / "webshop-data/direction-02"
    assert catalog_hashes(catalog) == seal["catalog_hashes"]
    sys.path.insert(0, str(args.checkout))
    from web_agent_site.engine.goal import get_reward

    env = environment(args, catalog, archive)
    assert price_hash(env) == seal["price_hash"]
    assert all(env.server.goals[t["goal_id"]] == t["goal"] for t in tasks)
    manifest = {"seed": SEED, "sigma": SIGMA, "goal_ids": [t["goal_id"] for t in tasks], "n": 60,
                "preregistration_commit": check_commit(), "seal_sha256": runner.file_sha(SEAL),
                "protocol_sha256": runner.file_sha(protocol), "script_sha256": runner.file_sha(Path(__file__)),
                "global_implementation_sha256": runner.file_sha(implementation),
                "calibration_sha256": runner.file_sha(calibration_file), "model_revision": runner.MODEL_REVISION,
                "catalog_hashes": seal["catalog_hashes"], "price_hash": seal["price_hash"], "checkout": archive["checkout"]}
    if (output / "manifest.json").exists():
        stored = read_json(output / "manifest.json")
        if stored != manifest:
            assert stored == {**manifest, "protocol_sha256": ORIGINAL_PROTOCOL, "script_sha256": ORIGINAL_SCRIPT}
            assert not list((output / "winner").glob("????.json"))
            save(output / "diagnostics" / "manifest-before-overflow-fix.json", stored)
            for partial in (output / "baseline").glob("????.jsonl"):
                if not partial.with_suffix(".json").exists():
                    (output / "diagnostics" / partial.name).write_bytes(partial.read_bytes())
            save(output / "manifest.json", manifest)
    else:
        save(output / "manifest.json", manifest)
    model, tokenizer, layer, stats = runner.load_model(args.model_dir)
    save(output / "model-load.json", stats)
    runtime = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
               **{name: version(name) for name in ("transformers", "bitsandbytes", "accelerate")}}
    assert runtime == previous["versions"]
    assert runner.tensor_sha(layer.weight) == archive["weight_hashes"]["baseline"]
    model_layout = layout(model)
    quantized, floating = model_layout
    assert [[name, list(shape)] for name, _, shape in quantized] == previous["quantized_shapes"]
    assert [[name, list(weight.shape)] for name, weight in floating] == previous["floating_shapes"]
    originals = {name: weight.detach().cpu().clone() for name, weight in floating}
    probe = calibration_inputs(root, tokenizer)[0][1][0]
    reference = logits(model, probe)
    with perturb(model_layout, originals, SEED, 0.0) as calls:
        assert torch.equal(logits(model, probe), reference) and len(calls) == len(quantized)
    with perturb(model_layout, originals, SEED, SIGMA) as calls:
        changed = logits(model, probe)
        assert torch.isfinite(changed).all() and not torch.equal(changed, reference)
        assert torch.equal(logits(model, probe), changed) and len(calls) == len(quantized)
    assert torch.equal(logits(model, probe), reference)
    log_p = log_softmax(reference.double(), dim=-1)
    log_q = log_softmax(changed.double(), dim=-1)
    save(output / "forward.json", {"zero_exact": True, "changed": True, "repeat_exact": True,
                                   "restored_exact": True, "one_prompt_kl": float((log_p.exp() * (log_p - log_q)).sum(-1).mean())})
    respond = runner.respond

    def respond_with_page_limit(model, tokenizer, base, history, stop_at_line):
        page = compact_page(tokenizer, base, history[-1])
        return respond(model, tokenizer, base, [*history[:-1], page], stop_at_line)

    runner.respond = respond_with_page_limit
    try:
        for name in ("baseline", "winner"):
            folder = output / name
            condition = nullcontext() if name == "baseline" else perturb(model_layout, originals, SEED, SIGMA)
            with condition:
                for start in range(0, len(tasks), 10):
                    run_episodes(env, model, tokenizer, tasks[start:start + 10], folder, get_reward,
                                 output / "baseline" if name == "winner" else None)
                    print(name, start + 10, "of", len(tasks), flush=True)
    finally:
        runner.respond = respond
    assert torch.equal(logits(model, probe), reference)
    analyze(output, tasks)


if __name__ == "__main__":
    main()
