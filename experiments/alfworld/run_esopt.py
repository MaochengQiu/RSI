"""One standard ES update and a disjoint candidate transfer test in ALFWorld."""

import argparse
from collections import Counter
import csv
import importlib.metadata
from itertools import permutations
import json
from pathlib import Path
import random
import re
import sys
import time

import torch

import run_experiment as alf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agentic-esopt-sudoku"))
from run_local import REVISION, upstream_directory
from run_observable_es import weight_digest


SCENE_SEED = 20261124
DIRECTION_SEED = 20260701
SIGMA = 0.0007
ALPHA = 0.0005
GROUPS = tuple(alf.TASKS)
PHASES = {"development": "valid_seen", "confirmation": "valid_seen", "aggregate": "valid_unseen"}
REFERENCE = Path(__file__).resolve().parent / "handoff-data"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    temporary = path.with_suffix(".tmp")
    alf.write_json(temporary, value)
    temporary.replace(path)


def freeze(path, value):
    if path.exists():
        if read(path) != json.loads(json.dumps(value)):
            raise RuntimeError(f"Frozen artifact differs: {path}")
    else:
        save(path, value)


def scenes(data_dir):
    historical = {s["id"] for s in read(REFERENCE / "baseline-configuration.json")["scenes"]}
    taken_layouts = {Path(scene).parts[-3].rsplit("-", 1)[1] for scene in historical}
    rng = random.Random(SCENE_SEED)
    records = {}
    for phase, partition in PHASES.items():
        root = data_dir / "json_2.1.1" / partition
        records[phase] = []
        phase_layouts, used_families = set(), set()
        # Reserve an Apple layout before filling the other task groups.
        for kind in (GROUPS[-1], *GROUPS[:-1]):
            families = sorted(path for path in root.glob(f"{kind}-*") if path.is_dir())
            rng.shuffle(families)
            slots = (True, False) if kind == GROUPS[-1] and phase != "aggregate" else (None, None)
            for apple in slots:
                for family in families:
                    target = family.name.split("-")[1].casefold()
                    layout = family.name.rsplit("-", 1)[1]
                    if family in used_families or layout in taken_layouts or (
                        phase != "aggregate" and layout in phase_layouts
                    ) or (apple is not None and (target == "apple") != apple):
                        continue
                    games = sorted(family.glob("trial_*/game.tw-pddl"))
                    if games:
                        game = games[0]
                        break
                else:
                    raise RuntimeError(f"Could not choose a disjoint layout for {phase}/{kind}/{apple}.")
                phase_layouts.add(layout)
                used_families.add(family)
                records[phase].append(dict(
                    task=kind, target=target, layout=layout, id=game.relative_to(data_dir).as_posix(),
                    sha256=alf.file_sha(game),
                ))
        taken_layouts.update(phase_layouts)
    if len({s["sha256"] for cases in records.values() for s in cases}) != 18:
        raise RuntimeError("Duplicate game bytes among the selected scenes.")
    return records


def trace_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def behavior(scene, rows):
    target = re.escape(scene["target"])
    object_name = rf"\b{target} \d+\b"
    take = re.compile(rf"take {target} \d+ from (.+)$", re.IGNORECASE)
    actions = [r for r in rows if r["action"] is not None]
    visits, contents = set(), {}
    first = None
    for i, row in enumerate(actions):
        action = row["action"].casefold()
        feedback = row["feedback"].casefold()
        dest = re.fullmatch(r"(?:go to|open) (.+)", action)
        if dest and feedback.startswith(("you arrive at ", "you open ")):
            location = dest[1]
            visits.add(location)
            if "you see" in feedback or "empty" in feedback:
                contents[location] = bool(re.search(object_name, feedback))
        failed_take = take.fullmatch(action)
        if first is None and failed_take and feedback.strip() == "nothing happens.":
            source = failed_take[1]
            following = actions[i+1] if i+1 < len(actions) else None
            next_action = following["action"] if following else None
            next_feedback = following["feedback"] if following else ""
            navigation = re.fullmatch(r"go to (.+)", next_action.casefold()) if next_action else None
            moved = bool(navigation and navigation[1] != source and
                         next_feedback.casefold().startswith("you arrive at "))
            first = dict(step=row["action_index"], action=row["action"], source=source,
                         source_seen_without_target=contents.get(source) is False,
                         next_action=next_action, next_feedback=next_feedback,
                         moved_elsewhere=moved,
                         moved_to_unvisited=bool(moved and navigation[1] not in visits))
    picked = next((r["action_index"] for r in actions
                   if re.search(rf"you pick up (?:the |a )?{object_name}", r["feedback"], re.IGNORECASE)), None)
    return dict(first_failed_target_take=first, first_target_pickup=picked,
                ineffective_actions=sum(r["feedback"].strip().casefold() == "nothing happens." for r in actions))


def summarize_candidates(records, cases, policies):
    summary = {}
    for policy in policies:
        summary[policy] = {}
        for kind in GROUPS:
            indices = [i for i, scene in enumerate(cases) if scene["task"] == kind]
            rows = [records[policy, i] for i in indices]
            failures = [r["behavior"]["first_failed_target_take"] for r in rows]
            deltas = [int(r["won"]) - int(records["base", i]["won"]) for i, r in zip(indices, rows)]
            summary[policy][kind] = dict(
                won=sum(r["won"] for r in rows), n=len(rows), gains=deltas.count(1), losses=deltas.count(-1),
                picked=sum(r["behavior"]["first_target_pickup"] is not None for r in rows),
                recovery_opportunities=sum(r is not None for r in failures),
                moved_elsewhere=sum(r is not None and r["moved_elsewhere"] for r in failures),
                moved_to_unvisited=sum(r is not None and r["moved_to_unvisited"] for r in failures),
            )
    return summary


def crossover(table, candidates):
    options = []
    for a, b in permutations(candidates, 2):
        for first, second in permutations(GROUPS, 2):
            if first > second:
                continue
            gaps = (table[a][first]["won"] - table[b][first]["won"],
                    table[b][second]["won"] - table[a][second]["won"])
            if min(gaps) > 0:
                options.append((-min(gaps), -sum(gaps), a, b, first, second))
    if not options:
        return None
    _, _, a, b, first, second = min(options)
    return [dict(a=a, b=b, task=first), dict(a=b, b=a, task=second)]


def navigation_prediction(records, cases, candidates):
    apple, = (i for i, scene in enumerate(cases) if scene["task"] == GROUPS[-1] and scene["target"] == "apple")
    eligible = {p: records[p, apple]["behavior"]["first_failed_target_take"] for p in ("base", *candidates)}
    base = eligible["base"]
    if base is not None and not base["moved_to_unvisited"]:
        for policy in candidates:
            event = eligible[policy]
            if event is not None and event["moved_to_unvisited"]:
                return dict(a=policy, b="base")
    return None


def replay_trace(data_dir, scene, path, result):
    game = data_dir / scene["id"]
    assert alf.file_sha(game) == scene["sha256"]
    env = alf.new_game(game)
    try:
        initial, _ = env.reset()
        assert initial == result["initial_observation"]
        rows = trace_rows(path)
        actions, won = [], False
        for i, row in enumerate(rows, 1):
            assert row["round"] == i
            assert alf.parse_line(row["raw_generation"]) == row["parsed_line"]
            if row["action"] is None:
                assert row["parsed_line"].startswith("think:") or not row["parsed_line"]
                assert row["feedback"] == ("OK." if row["parsed_line"] else "")
            else:
                action = row["action"]
                assert action == row["parsed_line"]
                actions.append(action)
                feedback, _, done, info = env.step(action)
                won = bool(info["won"])
                assert (feedback, bool(done), won, len(actions)) == (
                    row["feedback"], row["done"], row["won"], row["action_index"],
                )
            if row["done"]:
                assert i == len(rows)
        assert (len(rows), actions, won) == (result["rounds"], result["actions"], result["won"])
        assert behavior(scene, rows) == result["behavior"]
    finally:
        env.close()


def run(args, scene_sets):
    from algorithms.es.seeded_model_es import SeedReplayModelES

    wall_start = time.perf_counter()
    old = read(REFERENCE / "baseline-configuration.json")
    rng = random.Random(DIRECTION_SEED)
    directions = {f"direction-{i + 1}": rng.randrange(1, 2**31 - 1) for i in range(4)}
    prompts_sha = alf.file_sha(args.prompts)
    if prompts_sha != old["react_prompt_sha256"]:
        raise RuntimeError("Prompt file differs from the earlier ALFWorld experiment.")
    freeze(args.output / "protocol.json", dict(
        model=alf.MODEL_ID, revision=alf.MODEL_REVISION, es_revision=REVISION, directions=directions,
        scene_seed=SCENE_SEED, sigma=SIGMA, alpha=ALPHA, parameter_scope="floating parameters in NF4 policy",
        reward="mean of official episode wins on development", normalization="zscore ddof=0",
        prompt_sha256=prompts_sha, generation={k: old[k] for k in (
            "max_rounds", "max_input_tokens", "max_new_tokens", "do_sample", "enable_thinking", "reproducibility_seed",
        )}, scenes=scene_sets, runner_sha256=alf.file_sha(Path(__file__)), environment={
            p: importlib.metadata.version(p) for p in old["versions"]
        },
    ))
    examples = alf.adapt_examples(read(args.prompts))
    model, tokenizer, layer, stats = alf.load_model(args.model_dir)
    if str(model.config.quantization_config) != old["quantization"] or alf.tensor_sha(layer.weight) != read(
        REFERENCE / "weight-verification.json"
    )["original_sha256"]:
        raise RuntimeError("Base model target matrix or quantization differs from the historical run.")
    es = SeedReplayModelES()
    info = es.init(model)
    if not all(type(p) is torch.nn.Parameter and p.is_cuda for _, _, p in es.parameter_infos):
        raise RuntimeError("Unexpected floating parameter type or placement.")
    live = [(name, p) for name, _, p in es.parameter_infos]
    snapshot = [(name, p.detach().cpu()) for name, p in live]
    center_sha = weight_digest(snapshot)
    freeze(args.output / "center.json", dict(es=info, sha256=center_sha))
    save(args.output / "model-load.json", stats)
    print(f"Center ready with {info['total_params']:,} floating parameters", flush=True)

    def reset():
        for (_, param), (_, original) in zip(live, snapshot):
            param.copy_(original)
        es.active_perturbation = None
        es.update_history.clear()
        if weight_digest(live) != center_sha:
            raise RuntimeError("Failed to restore the same center.")

    def phase(name, policies, update=None):
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        results, hashes = {}, {}
        for policy in policies:
            reset()
            if policy in directions:
                es.apply(seed=directions[policy], sigma=SIGMA)
            elif policy == "updated":
                es.update(**update["request"])
            digest = weight_digest(live)
            hashes[policy] = digest
            if policy in directions and digest == center_sha:
                raise RuntimeError("Perturbation did not affect selected parameters.")
            if name == "confirmation" and digest != dev_hashes[policy]:
                raise RuntimeError(f"Candidate rebuild differs for {policy}.")
            if policy == "updated" and digest != update["updated_sha256"]:
                raise RuntimeError("Aggregate replay differs.")
            for i, scene in enumerate(scene_sets[name]):
                path = folder / f"{policy}-{i+1:02}.json"
                trace = path.with_suffix(".jsonl")
                context = dict(policy=policy, scene=scene, weight_sha256=digest)
                if path.exists():
                    result = read(path)
                    if any(result[k] != v for k, v in context.items()):
                        raise RuntimeError(f"Cached episode differs: {path}")
                    replay_trace(args.data_dir, scene, trace, result)
                else:
                    temporary = trace.with_suffix(".partial")
                    result = alf.episode(model, tokenizer, examples, args.data_dir, scene, temporary)
                    result.update(context, trace=trace.name, behavior=behavior(scene, trace_rows(temporary)))
                    replay_trace(args.data_dir, scene, temporary, result)
                    temporary.replace(trace)
                    save(path, result)
                if policy != "base":
                    base = results["base", i]
                    assert result["initial_observation"] == base["initial_observation"]
                    result["first_action_difference"] = alf.first_difference(base["actions"], result["actions"])
                results[policy, i] = result
                print(name, policy, i+1, scene["target"], "won" if result["won"] else "failed",
                      f"{result['rounds']} rounds, {result['elapsed_seconds']:.1f}s", flush=True)
        return results, hashes

    policies = ("base", *directions)
    dev, dev_hashes = phase("development", policies)
    assert len(set(dev_hashes.values())) == len(policies)
    dev_table = summarize_candidates(dev, scene_sets["development"], policies)
    prediction = dict(crossover=crossover(dev_table, list(directions)),
                      apple_navigation=navigation_prediction(dev, scene_sets["development"], list(directions)))
    freeze(args.output / "development.json", dict(scores=dev_table, weights=dev_hashes, prediction=prediction))
    confirm, _ = phase("confirmation", policies)
    confirm_table = summarize_candidates(confirm, scene_sets["confirmation"], policies)
    crossed = [dict(**item, net=confirm_table[item["a"]][item["task"]]["won"] -
                    confirm_table[item["b"]][item["task"]]["won"]) for item in prediction["crossover"] or ()]
    apple_index, = (i for i, scene in enumerate(scene_sets["confirmation"])
                    if scene["task"] == GROUPS[-1] and scene["target"] == "apple")
    apple_events = {p: confirm[p, apple_index]["behavior"] for p in policies}
    apple_replication = None
    if choice := prediction["apple_navigation"]:
        candidate, base = (apple_events[choice[p]]["first_failed_target_take"] for p in ("a", "b"))
        if candidate is not None and base is not None:
            apple_replication = candidate["moved_to_unvisited"] and not base["moved_to_unvisited"]
    freeze(args.output / "confirmation.json", dict(scores=confirm_table, crossover=crossed,
                                                  replicated=bool(crossed) and all(r["net"] > 0 for r in crossed),
                                                  apple_events=apple_events, apple_replication=apple_replication))
    print("Candidate confirmation complete. Applying standard ESOpt update.", flush=True)
    request = dict(seeds=list(directions.values()), rewards=[
        sum(dev[p, i]["won"] for i in range(len(scene_sets["development"]))) / len(scene_sets["development"])
        for p in directions
    ], alpha=ALPHA, reward_normalization="zscore")
    reset()
    update_result = es.update(**request)
    update = dict(request=request, diagnostics=update_result, center_sha256=center_sha,
                  updated_sha256=weight_digest(live))
    freeze(args.output / "update.json", update)
    aggregate, _ = phase("aggregate", ("base", "updated"), update)
    aggregate_table = summarize_candidates(aggregate, scene_sets["aggregate"], ("base", "updated"))
    for phase_name, rows in (("development", dev), ("confirmation", confirm), ("aggregate", aggregate)):
        with (args.output / f"{phase_name}-scores.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("policy", "scene", "task", "target", "won", "rounds",
                                                       "steps", "first_different_step", "picked_target",
                                                       "failed_target_take", "moved_elsewhere", "moved_to_unvisited"))
            writer.writeheader()
            for (policy, _), row in rows.items():
                event = row["behavior"]["first_failed_target_take"]
                first = row.get("first_action_difference")
                writer.writerow(dict(policy=policy, scene=row["scene"]["id"], task=row["scene"]["task"],
                                     target=row["scene"]["target"], won=int(row["won"]), rounds=row["rounds"],
                                     steps=row["steps"], first_different_step=first["action_index"] if first else None,
                                     picked_target=row["behavior"]["first_target_pickup"] is not None,
                                     failed_target_take=event is not None,
                                     moved_elsewhere=event["moved_elsewhere"] if event else None,
                                     moved_to_unvisited=event["moved_to_unvisited"] if event else None))
    pairs = Counter((int(aggregate["base", i]["won"]), int(aggregate["updated", i]["won"]))
                    for i in range(len(scene_sets["aggregate"])))
    result = dict(episodes=len(dev) + len(confirm) + len(aggregate), prediction=prediction,
                  development=dev_table, confirmation=confirm_table, crossover=crossed,
                  apple_replication=apple_replication,
                  aggregate=aggregate_table, aggregate_pairs={f"{a}->{b}": n for (a, b), n in pairs.items()},
                  seconds=time.perf_counter() - wall_start)
    save(args.output / "summary.json", result)
    print("Verified aggregate pairs", result["aggregate_pairs"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "rsi-data/alfworld")
    parser.add_argument("--model-dir", type=Path, default=Path.home() / "rsi-data/models/Qwen3-4B-1cfa9a7")
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.prompts = args.prompts or args.data_dir / "react-alfworld_3prompts.json"
    upstream = upstream_directory()
    sys.path.insert(0, str(upstream))
    selected = scenes(args.data_dir)
    if args.plan_only:
        print(json.dumps(selected, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        run(args, selected)


if __name__ == "__main__":
    main()
