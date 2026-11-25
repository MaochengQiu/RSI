"""Compare checkout on identical opportunities drawn only from baseline traces."""

import argparse
import hashlib
from pathlib import Path

import torch

from run_holdout import pilot, read_json, runner


def opportunities(folder, goal_id):
    trace_path = folder / f"{goal_id:04d}.jsonl"
    trace = pilot.sweep.read_trace(trace_path)
    scored = {s["round"]: s for s in read_json(trace_path.with_suffix(".json"))["offline_purchase_scores"]}
    seen = set()
    for turn in trace:
        state = turn["before"]
        key = (state["asin"], tuple(sorted(state["options"].items())))
        if "buy now" not in state["clickables"] or key in seen:
            continue
        seen.add(key)
        prior = scored[turn["round"] - 1]
        assert (prior["asin"], prior["options"]) == (state["asin"], state["options"])
        yield turn, prior["score"]


def buys(line):
    match = pilot.COMMAND.fullmatch(line)
    if match:
        return match[1].lower() == "click" and match[2].strip().lower() == "buy now"
    return line.strip().lower() == "buy now"


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    root = args.results
    manifest = read_json(root / "manifest.json")
    checks = [(t["goal_id"], list(opportunities(root / "baseline", t["goal_id"]))) for t in manifest["tasks"]]
    model, tokenizer, layer, _ = runner.load_model(args.model_dir)
    original = layer.weight.detach().cpu().clone()
    assert runner.tensor_sha(original) == manifest["original_sha256"]
    for condition in manifest["conditions"]:
        folder = root / condition["name"]
        assert (folder / "restored.json").is_file()
        records = []
        try:
            if "seed" in condition:
                noise = torch.randn(original.shape, generator=torch.Generator().manual_seed(condition["seed"]))
                layer.weight.copy_((original.float() + noise * condition["sigma"]).to(original.dtype))
                assert runner.tensor_sha(layer.weight) == condition["sha256"]
            for goal_id, points in checks:
                turns = pilot.sweep.read_trace(folder / f"{goal_id:04d}.jsonl")
                logged = {turn["prompt"]: turn for turn in turns}
                for source, score in points:
                    prompt = source["prompt"]
                    observed = logged.get(prompt)
                    if condition["name"] == "baseline" or observed is None:
                        raw, _, _ = pilot.sweep.replay_decision(model, tokenizer, prompt)
                        line = runner.parse_line(raw)
                        method = "replay"
                    else:
                        assert observed["before"] == source["before"]
                        line = observed["parsed"]
                        method = "on-policy-identical-prompt"
                    if observed is not None:
                        assert buys(line) == observed["bought"]
                        if condition["name"] == "baseline":
                            assert line == observed["parsed"]
                    records.append({"goal_id": goal_id, "round": source["round"],
                                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                    "asin": source["before"]["asin"], "options": source["before"]["options"],
                                    "buy_now_score": score, "line": line, "buy": buys(line), "source": method})
            pilot.persist(folder / "checkout-probes.json", records)
            print(condition["name"], len(records), "fixed prompts audited", flush=True)
        finally:
            layer.weight.copy_(original)
            assert runner.tensor_sha(layer.weight) == manifest["original_sha256"]


if __name__ == "__main__":
    main()
