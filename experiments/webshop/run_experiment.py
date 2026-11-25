"""Paired local WebShop using the same six ALFWorld weight candidates."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "alfworld"))
import sweep_perturbations as sweep

runner = sweep.runner
MAX_ROUNDS = 16
N_GOALS = 20
PROMPT = """You are shopping in a simulated store. Follow the instruction and buy one product.
Output one action and no explanation. On the search page output search[keywords], for example search[blue insulated lunch bag].
On other pages copy one of the allowed actions exactly. Product IDs open items, and option buttons select specifications.
Use concise search keywords. On a product page, select any matching required options before choosing Buy Now.
If a product is unsuitable, use < Prev to see other results or Back to Search to change your query.
Avoid repeating a failed action. Make a purchase of the best available fit within the step budget.
"""
COMMAND = re.compile(r"(search|click)\[(.+)\]", re.I)


def persist(path, value):
    temporary = path.with_suffix(".tmp")
    runner.write_json(temporary, value)
    temporary.replace(path)


def fingerprint(paths, root):
    return {str(p.relative_to(root)): runner.file_sha(p) for p in sorted(paths) if p.is_file()}


def snapshot(env):
    state = env.server.user_sessions[env.session]
    page = env.server.get_page_name(env.state["url"])
    links = env._parse_html().select(".product-link")
    asin = state["asin"]
    return {
        "page_type": page or "search",
        "page_number": state["page"],
        "query": state["keywords"],
        "result_asins": [link.get_text(strip=True).upper() for link in links],
        "asin": asin,
        "options": dict(state["options"]),
        "price": env.server.product_prices.get(asin),
        "clickables": env.get_available_actions()["clickables"],
        "screen_sha256": hashlib.sha256(env.observation.encode()).hexdigest(),
    }


def execute(env, line, state):
    match = COMMAND.fullmatch(line)
    if not match and line.strip().lower() in state["clickables"]:
        match = COMMAND.fullmatch(f"click[{line.strip()}]")
    if not match:
        reason = "syntax"
    else:
        name, value = match[1].lower(), match[2].strip().lower()
        if not value or (name == "click" and (value not in state["clickables"] or value == "search")):
            reason = "unavailable_click_or_empty"
        elif name == "search" and state["page_type"] != "search":
            reason = "search_off_start_page"
        else:
            action = f"{name}[{value}]"
            tag = env.text_to_clickable.get(value)
            kind = "search" if name == "search" else "navigation"
            if name == "click":
                if value == "buy now":
                    kind = "purchase"
                elif "product-link" in tag.get("class", []):
                    kind = "product"
                elif tag.name == "input":
                    kind = "option"
            _, reward, done, _ = env.step(action)
            return action, kind, reward, done, None
    return None, None, 0.0, False, reason


def episode(env, model, tokenizer, task, path, get_reward):
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    goal_id, goal = task["goal_id"], task["goal"]
    env.server.user_sessions.pop(str(goal_id), None)
    env.reset(session=goal_id)
    assert env.get_instruction_text().removeprefix("Instruction:").strip() == goal["instruction_text"].strip()
    stop = runner.StopAtLine(tokenizer)
    history, rows, actions = [], [], []
    reward, done, last_error = 0.0, False, None
    with path.open("w", encoding="utf-8") as trace:
        for step in range(1, MAX_ROUNDS + 1):
            before = snapshot(env)
            observation = env.observation
            base = PROMPT + "\nShopping instruction: " + goal["instruction_text"]
            menu = ("search[keywords]" if before["page_type"] == "search" else
                    ", ".join(f"click[{label}]" for label in before["clickables"]))
            correction = "\nLast command was rejected. Use an action in the menu below." if last_error else ""
            current = f"\nCurrent page ({before['page_type']}):\n{observation}{correction}\nAllowed actions: {menu}\nAction:"
            context = history + [current]
            line, raw, prompt, inputs, dropped = runner.respond(model, tokenizer, base, context, stop)
            if dropped >= len(context):
                raise RuntimeError("Current WebShop page exceeds the token budget.")
            action, kind, reward, done, error = execute(env, line, before)
            after = snapshot(env)
            if action:
                actions.append(action)
            row = {
                "round": step, "raw": raw, "parsed": line, "prompt": prompt,
                "input_tokens": inputs, "history_dropped": dropped,
                "observation": observation, "before": before, "after": after,
                "action": action, "kind": kind, "error": error,
                "action_index": len(actions) if action else None,
                "reward": reward, "bought": bool(done),
            }
            rows.append(row)
            trace.write(json.dumps(row, ensure_ascii=False) + "\n")
            trace.flush()
            if done:
                break
            last_error = error
            if action:
                history.append(f"\nPrevious action: {action}\n"
                               f"Outcome: {kind}; "
                               f"results={after['result_asins']}, item={after['asin']}, options={after['options']}.")

    # Score observations only after the policy is finished. No offline labels enter its prompt.
    scores = []
    for row in rows:
        state = row["after"]
        asin = state["asin"]
        if asin:
            score = get_reward(env.server.product_item_dict[asin], goal, state["price"], state["options"])
            scores.append({"round": row["round"], "asin": asin, "options": state["options"], "score": score})
    if done and (not scores or abs(scores[-1]["score"] - reward) > 1e-9):
        raise RuntimeError("Offline purchase score differs from the environment score.")
    outcome = {
        "goal_id": goal_id, "bought": bool(done), "reward": reward if done else 0.0,
        "purchase": {"asin": rows[-1]["after"]["asin"], "options": rows[-1]["after"]["options"],
                     "step": len(actions), "round": len(rows),
                     "official_components": env.server.user_sessions[env.session].get("verbose_info")} if done else None,
        "rounds": len(rows), "actions": actions,
        "errors": dict(Counter(row["error"] for row in rows if row["error"])),
        "first_state": rows[0]["before"], "offline_purchase_scores": scores,
        "trace": str(path), **runner.performance(start),
    }
    persist(path.with_suffix(".json"), outcome)
    return outcome


def summarize_baseline(rows):
    return {
        "n": len(rows), "bought": sum(r["bought"] for r in rows),
        "partial": sum(r["bought"] and 0 < r["reward"] < 1 for r in rows),
        "full": sum(r["bought"] and r["reward"] == 1 for r in rows),
        "purchase_score_counts": dict(Counter(str(r["reward"]) for r in rows if r["bought"])),
        "action_errors": dict(sum((Counter(r["errors"]) for r in rows), Counter())),
    }


def compare(a, b):
    assert a["goal_id"] == b["goal_id"] and a["first_state"] == b["first_state"]
    diff = runner.first_difference(a["actions"], b["actions"])
    if diff:
        step = diff["action_index"]
        sides = []
        for episode_result in (a, b):
            row = next((r for r in sweep.read_trace(Path(episode_result["trace"]))
                        if r["action_index"] == step), None)
            sides.append({k: row[k] for k in ("round", "kind", "action", "before", "after")} if row else None)
        diff["original_effect"], diff["perturbed_effect"] = sides
        keys = ("page_type", "page_number", "result_asins", "asin", "options", "price")
        diff["changed_page_product_or_options"] = (
            not all(sides)
            or any(sides[0]["after"][k] != sides[1]["after"][k] for k in keys)
            or (sides[0]["after"]["page_type"] != "search_results"
                and sides[0]["after"]["screen_sha256"] != sides[1]["after"]["screen_sha256"])
        )
    return {
        "goal_id": a["goal_id"], "first_different_action": diff,
        "original_score": a["reward"], "candidate_score": b["reward"],
        "delta": round(b["reward"] - a["reward"], 12),
        "original": a, "candidate": b,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--weight-reference", required=True, type=Path)
    parser.add_argument("--catalog-size", choices=(1000, 10000), default=10000, type=int)
    parser.add_argument("--stage", choices=("probe", "baseline", "sweep"), required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout))
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    from web_agent_site.engine.goal import get_reward
    from web_agent_site.engine import engine

    root = args.checkout
    random.seed(runner.SEED)
    if args.catalog_size == 10000:
        engine.DEFAULT_ATTR_PATH = str(root / "data" / "items_ins_v2_10000.json")
        engine.HUMAN_ATTR_PATH = str(root / "data" / "items_human_ins_10000.json")
        env = WebAgentTextEnv(observation_mode="text", human_goals=True,
                              file_path=str(root / "data" / "items_shuffle_10000.json"))
        index_dir = "indexes"
    else:
        env = WebAgentTextEnv(observation_mode="text", num_products=1000, human_goals=True)
        index_dir = "indexes_1k"
    ids = sorted(random.Random(runner.SEED).sample(range(1, len(env.server.goals)), N_GOALS))
    tasks = [{"goal_id": i, "goal": env.server.goals[i]} for i in ids]
    provenance = {
        "checkout": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "catalog_and_index": fingerprint(
            list((root / "data").glob("*.json")) + list((root / "search_engine" / index_dir).glob("*")), root,
        ),
        "prices": env.server.product_prices,
        "split": "Local development sample from the fixed catalog's human-goal pool, excluding smoke goal 0",
        "catalog_size": args.catalog_size, "task_pool_size": len(env.server.goals),
        "goal_and_sample_seed": runner.SEED, "tasks": tasks, "prompt": PROMPT,
        "max_rounds": MAX_ROUNDS, "input_tokens": runner.MAX_INPUT_TOKENS,
        "max_new_tokens": runner.MAX_NEW_TOKENS, "decoding": "greedy",
        "model": runner.MODEL_ID, "model_revision": runner.MODEL_REVISION,
        "target": runner.TARGET + ".weight", "weight_seeds": sweep.SEEDS,
        "weight_scales": sweep.SCALES,
        "runner_sha256": runner.file_sha(Path(__file__)),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    if manifest.exists():
        assert sweep.read_json(manifest) == json.loads(json.dumps(provenance))
    else:
        persist(manifest, provenance)
    if args.stage == "probe":
        env.reset(session=0)
        print("development goals", len(tasks), "/", len(env.server.goals), "smoke instruction:", env.get_instruction_text())
        print("search:", env.step("search[" + env.server.goals[0]["query"] + "]")[0][:350])
        return
    model, tokenizer, layer, _ = runner.load_model(args.model_dir)
    original = layer.weight.detach().cpu().clone()
    verified = sweep.read_json(args.weight_reference / "weight-verification.json")
    assert runner.tensor_sha(original) == verified["original_sha256"]

    def run_condition(name):
        directory = args.output / name
        directory.mkdir(exist_ok=True)
        results = []
        for task in tasks:
            path = directory / f"{task['goal_id']:04d}.jsonl"
            checkpoint = path.with_suffix(".json")
            if checkpoint.exists():
                result = sweep.read_json(checkpoint)
            else:
                result = episode(env, model, tokenizer, task, path, get_reward)
                print(name, task["goal_id"], result["bought"], result["reward"], result["errors"], flush=True)
            results.append(result)
        return results

    baseline = run_condition("baseline")
    summary = summarize_baseline(baseline)
    persist(args.output / "baseline-summary.json", summary)
    print(summary, flush=True)
    if args.stage == "baseline" or summary["bought"] < N_GOALS // 2 or len(summary["purchase_score_counts"]) < 2:
        return
    aggregates = {}
    for seed in sweep.SEEDS:
        noise = torch.randn(original.shape, generator=torch.Generator().manual_seed(seed))
        for scale in sweep.SCALES:
            name = f"seed-{seed}-scale-{scale:.2f}"
            try:
                perturbed = (original.float() + noise * (original.float().square().mean().sqrt().item() * scale)).to(original.dtype)
                layer.weight.copy_(perturbed)
                if (seed, scale) == (runner.SEED, runner.NOISE_SCALE):
                    assert runner.tensor_sha(layer.weight) == verified["perturbed_sha256"]
                candidates = run_condition(name)
            finally:
                layer.weight.copy_(original)
            assert runner.tensor_sha(layer.weight) == verified["original_sha256"]
            pairs = [compare(a, b) for a, b in zip(baseline, candidates)]
            persist(args.output / name / "paired.json", pairs)
            aggregates[name] = {
                "changed_actions": sum(r["first_different_action"] is not None for r in pairs),
                "score_up": sum(r["delta"] > 0 for r in pairs),
                "score_down": sum(r["delta"] < 0 for r in pairs),
                "mean_delta": sum(r["delta"] for r in pairs) / len(pairs),
            }
            persist(args.output / "candidate-summary.json", aggregates)
            print(name, aggregates[name], flush=True)


if __name__ == "__main__":
    main()
