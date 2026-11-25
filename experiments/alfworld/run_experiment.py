"""Run a paired Qwen3-4B weight perturbation experiment on ALFWorld."""

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time

from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
import bitsandbytes as bnb
from safetensors.torch import save_file
import textworld.gym
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StopStringCriteria


MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
REACT_REVISION = "6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9"
TARGET = "model.layers.18.self_attn.v_proj"
TASKS = {
    "pick_and_place_simple": "put",
    "pick_clean_then_place_in_recep": "clean",
    "pick_cool_then_place_in_recep": "cool",
}
SEED = 20261120
NOISE_SCALE = 0.01
MAX_ROUNDS = 50
MAX_INPUT_TOKENS = 4096
MAX_NEW_TOKENS = 64


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def difference(reference, changed):
    delta = (changed.float() - reference.float()).abs()
    return {
        "max_abs": delta.max().item(),
        "rms": delta.square().mean().sqrt().item(),
        "changed_fraction": (delta != 0).float().mean().item(),
    }


def make_manifest(data_dir):
    split = data_dir / "json_2.1.1" / "valid_seen"
    scenes = []
    for task in TASKS:
        path = min(split.glob(f"{task}-*/trial_*/game.tw-pddl"))
        scenes.append({"task": task, "id": str(path.relative_to(data_dir)), "sha256": file_sha(path)})
    return scenes


def load_model(model_dir):
    source, revision = MODEL_ID, MODEL_REVISION
    if model_dir is not None:
        provenance = json.loads((model_dir / "source-revision.json").read_text(encoding="utf-8"))
        if (provenance["repo"], provenance["revision"]) != (MODEL_ID, MODEL_REVISION):
            raise ValueError("Local model revision differs from the fixed experiment revision.")
        source, revision = model_dir, None
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=[TARGET, "lm_head"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        source, revision=revision, use_safetensors=True,
        quantization_config=quantization, dtype=torch.float16,
        device_map={"": 0}, low_cpu_mem_usage=True,
    ).eval()
    layer = model.get_submodule(TARGET)
    adjacent = model.get_submodule("model.layers.18.self_attn.q_proj")
    assert type(layer) is torch.nn.Linear and layer.weight.dtype == torch.float16
    assert layer.weight.is_cuda and isinstance(adjacent, bnb.nn.Linear4bit)
    assert adjacent.weight.quant_state.quant_type == "nf4"
    torch.cuda.synchronize()
    stats = performance(start)
    stats.update({
        "target_name": f"{TARGET}.weight",
        "target_dtype": str(layer.weight.dtype),
        "target_device": str(layer.weight.device),
        "target_shape": list(layer.weight.shape),
        "neighboring_q_proj_quant_type": adjacent.weight.quant_state.quant_type,
    })
    return model, tokenizer, layer, stats


def performance(start):
    return {
        "elapsed_seconds": time.perf_counter() - start,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }


def new_game(game_file):
    infos = textworld.EnvInfos(won=True)
    env_id = textworld.gym.register_game(
        str(game_file), infos, max_episode_steps=MAX_ROUNDS,
        wrappers=[AlfredDemangler(shuffle=False)],
    )
    env = textworld.gym.make(env_id)
    env.seed(SEED)
    return env


def parse_line(raw):
    line = next((line.strip() for line in raw.splitlines() if line.strip()), "")
    return line.removeprefix(">").strip()


def adapt_examples(examples):
    # The current ALFWorld text grammar uses "move ... to ..." for receptacles.
    return {
        key: re.sub(r"^> put (.+?) in/on (.+)$", r"> move \1 to \2", value, flags=re.MULTILINE)
        for key, value in examples.items()
    }


class StopAtLine(StopStringCriteria):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, "\n")
        self.tokenizer = tokenizer
        self.prompt_length = 0

    def __call__(self, input_ids, scores, **kwargs):
        matches_newline = super().__call__(input_ids, scores, **kwargs)
        if matches_newline.item():
            tail = self.tokenizer.decode(input_ids[0, self.prompt_length:], skip_special_tokens=True)
            matches_newline.fill_(bool(parse_line(tail)))
        return matches_newline


def respond(model, tokenizer, base, history, stop_at_line):
    messages = [{"role": "user", "content": ""}]
    for drop in range(len(history) + 1):
        continuation = "".join(history[drop:])
        messages[0]["content"] = base + continuation
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        count = inputs.input_ids.shape[1]
        if count <= MAX_INPUT_TOKENS:
            break
    else:
        raise RuntimeError("The task and examples exceed the fixed context limit.")
    inputs = inputs.to("cuda")
    generation = deepcopy(model.generation_config)
    generation.do_sample = False
    generation.num_beams = 1
    generation.temperature = None
    generation.top_p = None
    generation.top_k = None
    generation.max_new_tokens = MAX_NEW_TOKENS
    stop_at_line.prompt_length = count
    ids = model.generate(
        **inputs, generation_config=generation, use_model_defaults=False,
        do_sample=False, num_beams=1, stopping_criteria=[stop_at_line],
    )
    raw = tokenizer.decode(ids[0, count:], skip_special_tokens=True)
    return parse_line(raw), raw, prompt, count, drop


def episode(model, tokenizer, examples, data_dir, scene, path):
    game_file = data_dir / scene["id"]
    assert file_sha(game_file) == scene["sha256"]
    env = new_game(game_file)
    history, actions, thoughts = [], [], []
    won, stop = False, "round_limit"
    stop_at_line = StopAtLine(tokenizer)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        raw_initial, _ = env.reset()
        initial = raw_initial.partition("\n\n")[2] or raw_initial
        kind = TASKS[scene["task"]]
        base = (
            "Interact with a household to solve a task. Here are two examples.\n"
            + examples[f"react_{kind}_1"] + examples[f"react_{kind}_0"]
            + "\nContinue the interaction. Output one command or one think: line only."
            + "\nUse move OBJECT to RECEPTACLE to place an object in or on a receptacle."
            + "\nHere is the task.\n" + initial + "\n>"
        )
        observation = initial
        with path.open("w", encoding="utf-8") as trace:
            for index in range(1, MAX_ROUNDS + 1):
                tick = time.perf_counter()
                line, raw, prompt, tokens, dropped = respond(model, tokenizer, base, history, stop_at_line)
                action = None
                if line.startswith("think:"):
                    thoughts.append({"round": index, "text": line})
                    feedback, done = "OK.", False
                elif line:
                    action = line
                    actions.append(action)
                    feedback, _, done, info = env.step(action)
                    won = bool(info["won"])
                else:
                    feedback, done = "", False
                    stop = "empty_output"
                record = {
                    "round": index, "observation": observation, "raw_generation": raw,
                    "parsed_line": line, "action": action,
                    "action_index": len(actions) if action is not None else None,
                    "feedback": feedback, "done": bool(done), "won": won,
                    "environment_feedback": feedback if action is not None else None,
                    "prompt": prompt, "input_tokens": tokens, "history_rounds_dropped": dropped,
                    "elapsed_seconds": time.perf_counter() - tick,
                }
                trace.write(json.dumps(record, ensure_ascii=False) + "\n")
                trace.flush()
                if not line or done:
                    if done:
                        stop = "environment_done"
                    break
                observation = feedback
                if observation.startswith("You arrive at loc "):
                    observation = observation.partition(". ")[2] or observation
                history.append(f" {line}\n{observation}\n>")
        torch.cuda.synchronize()
        return {
            "initial_observation": raw_initial, "won": won, "stop": stop,
            "rounds": index, "steps": len(actions), "actions": actions,
            "thoughts": thoughts, "trace": path.name, **performance(start),
        }
    finally:
        env.close()


def diagnostic(model, layer, inputs):
    captured = []

    def observe(_module, args, output):
        captured.append((args[0].detach().cpu(), output.detach().cpu()))

    hook = layer.register_forward_hook(observe)
    try:
        output = model(**inputs, use_cache=False, logits_to_keep=1)
    finally:
        hook.remove()
    assert len(captured) == 1
    return (*captured[0], output.logits[0, -1].detach().cpu())


def compare_diagnostics(reference, current):
    return {
        name: difference(a, b)
        for name, a, b in zip(("layer_input", "layer_output", "last_token_logits"), reference, current)
    }


def first_difference(left, right):
    for index in range(max(len(left), len(right))):
        a = left[index] if index < len(left) else None
        b = right[index] if index < len(right) else None
        if a != b:
            return {"action_index": index + 1, "original": a, "perturbed": b}
    return None


def trace_behavior(path):
    keys = ("parsed_line", "action", "observation", "feedback", "done", "won")
    with path.open(encoding="utf-8") as trace:
        return [tuple(row[key] for key in keys) for row in map(json.loads, trace)]


def write_table(path, rows):
    lines = [
        "| Scene | Original | Perturbed | First different action | Steps (original / perturbed) |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        scene_id = "/".join(row["scene"]["id"].split("/")[-3:-1])
        a, b = row["original"], row["perturbed"]
        change = row["first_different_action"]
        action = "None"
        if change:
            description = f"{change['original'] or '(ended)'} → {change['perturbed'] or '(ended)'}"
            action = f"{change['action_index']}. {description}".replace("|", r"\|")
        results = ["Success" if item["won"] else "Failure" for item in (a, b)]
        lines.append(f"| {scene_id} | {results[0]} | {results[1]} | {action} | {a['steps']} / {b['steps']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def experiment(args, model, tokenizer, layer, examples, load_stats):
    manifest = make_manifest(args.data_dir)
    settings = {
        "scenes": manifest, "model": MODEL_ID, "revision": MODEL_REVISION,
        "react_revision": REACT_REVISION, "react_prompt_sha256": file_sha(args.prompts),
        "runner_sha256": file_sha(Path(__file__)),
        "reproducibility_seed": SEED, "quantization": str(model.config.quantization_config),
        "do_sample": False, "num_beams": 1, "enable_thinking": False,
        "max_rounds": MAX_ROUNDS, "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS, "model_load": load_stats,
        "generation_stop": "first nonempty line or EOS or token limit",
        "react_examples_adaptation": "Replace put X in/on Y with move X to Y in example action lines.",
        "local_model_dir": str(args.model_dir) if args.model_dir else None,
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("alfworld", "textworld", "torch", "transformers", "accelerate", "bitsandbytes")
        },
    }
    write_json(args.output / "configuration.json", settings)
    inputs = tokenizer(
        "You are in a kitchen. Your task is to put the mug on the table.\n> look",
        return_tensors="pt",
    ).to("cuda")
    write_json(args.output / "diagnostic-input.json", inputs.input_ids.tolist())
    original = layer.weight.detach().cpu().clone()
    save_file({f"{TARGET}.weight": original}, args.output / "original-target.safetensors")
    original_sha = tensor_sha(original)
    a1 = diagnostic(model, layer, inputs)
    a2 = diagnostic(model, layer, inputs)

    def run(scene, condition, index):
        print(f"{condition} {scene['id']}", flush=True)
        path = args.output / f"{index + 1:02d}-{condition}.jsonl"
        result = episode(model, tokenizer, examples, args.data_dir, scene, path)
        write_json(path.with_suffix(".json"), result)
        return result

    baseline = [run(scene, "baseline", i) for i, scene in enumerate(manifest)]
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    sigma = original.float().square().mean().sqrt().item() * NOISE_SCALE
    noise = torch.randn(original.shape, generator=generator, dtype=torch.float32) * sigma
    perturbed = (original.float() + noise).to(original.dtype)
    try:
        layer.weight.copy_(perturbed)
        actual = layer.weight.detach().cpu()
        assert not torch.equal(original, actual)
        b = diagnostic(model, layer, inputs)
        verification = {
            "parameter": f"{TARGET}.weight", "dtype": str(original.dtype),
            "shape": list(original.shape), "sigma": sigma, "noise_seed": SEED,
            "target_rms_fraction": NOISE_SCALE,
            "relative_frobenius_delta": (
                (actual.float() - original.float()).norm() / original.float().norm()
            ).item(),
            "weight_delta": difference(original, actual),
            "original_sha256": original_sha, "perturbed_sha256": tensor_sha(actual),
            "baseline_repeat": compare_diagnostics(a1, a2),
            "perturbation": compare_diagnostics(a1, b),
        }
        write_json(args.output / "weight-verification.json", verification)
        assert verification["perturbation"]["layer_input"]["max_abs"] == 0
        assert (verification["perturbation"]["layer_output"]["rms"]
                > verification["baseline_repeat"]["layer_output"]["rms"])
        changed = [run(scene, "perturbed", i) for i, scene in enumerate(manifest)]
    finally:
        layer.weight.copy_(original)
    assert tensor_sha(layer.weight) == original_sha
    a3 = diagnostic(model, layer, inputs)
    verification["restored_sha256"] = tensor_sha(layer.weight)
    verification["restoration"] = compare_diagnostics(a1, a3)
    write_json(args.output / "weight-verification.json", verification)
    restored = run(manifest[0], "restored", 0)
    rows = []
    for scene, a, b in zip(manifest, baseline, changed):
        assert a["initial_observation"] == b["initial_observation"]
        rows.append({
            "scene": scene, "original": a, "perturbed": b,
            "first_different_action": first_difference(a["actions"], b["actions"]),
        })
    assert restored["initial_observation"] == baseline[0]["initial_observation"]
    before_trace = trace_behavior(args.output / baseline[0]["trace"])
    after_trace = trace_behavior(args.output / restored["trace"])
    result = {
        "rows": rows, "restored": restored,
        "restored_baseline_matches": before_trace == after_trace,
    }
    write_json(args.output / "results.json", result)
    write_table(args.output / "results.md", rows)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["smoke", "compare"])
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "rsi-data" / "alfworld")
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.prompts = args.prompts or args.data_dir / "react-alfworld_3prompts.json"
    examples = adapt_examples(json.loads(args.prompts.read_text(encoding="utf-8")))
    model, tokenizer, layer, load_stats = load_model(args.model_dir)
    write_json(args.output / "model-load.json", load_stats)
    if args.mode == "compare":
        experiment(args, model, tokenizer, layer, examples, load_stats)
        return
    train = args.data_dir / "json_2.1.1" / "train"
    game = min(train.glob("pick_and_place_simple-*/trial_*/game.tw-pddl"))
    scene = {"task": "pick_and_place_simple", "id": str(game.relative_to(args.data_dir)), "sha256": file_sha(game)}
    result = episode(model, tokenizer, examples, args.data_dir, scene, args.output / "smoke.jsonl")
    write_json(args.output / "smoke.json", result)


if __name__ == "__main__":
    main()
