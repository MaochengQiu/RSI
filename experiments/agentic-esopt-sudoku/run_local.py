"""Run the official Sudoku checkpoint evaluator against one GPU endpoint."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
REVISION = "3c6cd6992df3c353299cb9c6d33f733ede7854c8"
TRAINER_PATH = "sudoku-train-time/scripts/run_sudoku_es_train.py"
DATA_PATH = "data/sudoku/eval.jsonl"
SERVER_PATH = (
    "ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/"
    "llama31_instruct_server.py"
)
MODEL_REVISIONS = {
    "base": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "esopt": "1ed50da020e553ed2e027ca828826eb733a5c117",
}
SEED = 20261123


def upstream_directory():
    checkout = ROOT / "upstream"
    lock_file = ROOT / "upstream-snapshot/source-lock.json"
    if lock_file.is_file():
        lock = json.loads(lock_file.read_text(encoding="utf-8-sig"))
        required = {TRAINER_PATH, DATA_PATH, SERVER_PATH}
        if lock["revision"] != REVISION or not required <= lock["blobs"].keys():
            raise RuntimeError("Unexpected upstream snapshot")
        for name, expected in lock["blobs"].items():
            contents = (lock_file.parent / name).read_bytes()
            blob = f"blob {len(contents)}\0".encode() + contents
            if hashlib.sha1(blob).hexdigest() != expected:
                raise RuntimeError(f"Upstream file differs from locked blob: {name}")
        return lock_file.parent
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != REVISION:
        raise RuntimeError(f"Expected official {REVISION} checkout, got {revision}")
    return checkout


def available_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def wait_for_server(process, endpoint):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Model server exited with status {process.returncode}")
        try:
            # A GET returns 405 when the Flask POST endpoint is ready.
            urlopen(endpoint + "/completions", timeout=1).close()
            return
        except HTTPError as error:
            if error.code == 405:
                return
            raise
        except (URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("Model server did not become ready")


def evaluate(upstream, model, bits, count, repeats, output, seed):
    output.mkdir(parents=True, exist_ok=False)
    port = available_port()
    endpoint = f"http://127.0.0.1:{port}"
    policy = ROOT / "models" / model
    index = json.loads((policy / "model.safetensors.index.json").read_text())
    missing = [name for name in set(index["weight_map"].values()) if not (policy / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing shards for {model}: {missing}")
    quantization = {8: "--quantization", 4: "--load-in-4bit"}[bits]
    server = [
        sys.executable, "-u", str(upstream / SERVER_PATH), "--path", str(policy), "--d", "0",
        "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16",
        "--chat-template-enable-thinking", "false", "--seed", str(seed), quantization,
    ]
    trainer = [
        sys.executable, "-u", str(upstream / TRAINER_PATH), "--eval-only", "--endpoints", endpoint,
        "--model", "Qwen3.5-4B", "--eval-data", str(upstream / DATA_PATH), "--mask-count", "15",
        "--eval-limit", str(count), "--eval-repeats", str(repeats), "--case-workers", "1",
        "--max-turns", "45", "--max-tokens", "64", "--temperature", "0.7",
        "--top-p", "0.8", "--top-k", "20", "--min-p", "0", "--presence-penalty",
        "1.5", "--repetition-penalty", "1", "--result-root", str(output),
    ]
    manifest = {
        "upstream_revision": REVISION, "policy": model, "model_revision": MODEL_REVISIONS[model],
        "bits": bits,
        "seed": seed, "cases": count, "repeats": repeats,
        "eval_data_sha256": hashlib.sha256((upstream / DATA_PATH).read_bytes()).hexdigest(),
        "server_command": server, "trainer_command": trainer,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / ".runtime")
    environment["HF_HUB_OFFLINE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["ROOT"] = str(upstream)
    environment["PYTHONPATH"] += os.pathsep + str(upstream)
    print(f"{model}: {bits}-bit, {count} puzzles x {repeats}", flush=True)
    monitor = [
        "nvidia-smi", "--query-gpu=timestamp,memory.used,memory.free",
        "--format=csv,noheader,nounits", "-lms", "500",
    ]
    with (output / "server.log").open("w") as server_log, (output / "gpu.csv").open("w") as gpu_log:
        sampler = subprocess.Popen(monitor, stdout=gpu_log, stderr=subprocess.STDOUT)
        try:
            process = subprocess.Popen(server, cwd=upstream, env=environment, stdout=server_log,
                                       stderr=subprocess.STDOUT)
            try:
                wait_for_server(process, endpoint)
                start = time.monotonic()
                with (output / "eval.log").open("w") as eval_log:
                    subprocess.run(trainer, cwd=upstream, env=environment, stdout=eval_log,
                                   stderr=subprocess.STDOUT, check=True)
                elapsed = time.monotonic() - start
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        finally:
            sampler.terminate()
            sampler.wait()
    history = json.loads((output / "history.json").read_text())
    result = history[-1]["eval"]
    if len(result["runs"]) != repeats or any(
        row["count"] != count or row["valid_count"] != count for row in result["runs"]
    ):
        raise RuntimeError(f"Incomplete or failed model calls in {output / 'history.json'}")
    transport_errors = [
        turn["error"]
        for repeat in result["runs"]
        for row in repeat["scores"]
        for turn in row["turns"]
        if "error" in turn
    ]
    if transport_errors:
        raise RuntimeError(f"{len(transport_errors)} transport errors, first: {transport_errors[0]}")
    summary = {key: result[key] for key in ("count", "repeat_count", "solved_average", "average")}
    summary["seconds"] = elapsed
    summary["solved_by_repeat"] = [row["solved"] for row in result["runs"]]
    used_memory = []
    for line in (output / "gpu.csv").read_text().splitlines():
        _, separator, memory = line.partition(",")
        if separator and memory.split(",", 1)[0].strip().isdigit():
            used_memory.append(int(memory.split(",", 1)[0].strip()))
    summary["peak_device_used_mib"] = max(used_memory, default=None)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pilot", "paired32", "full"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--model", choices=("base", "esopt", "both"), default="both")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    upstream = upstream_directory()
    count, repeats = {"pilot": (2, 1), "paired32": (32, 1), "full": (32, 3)}[args.phase]
    models = ("base", "esopt") if args.model == "both" else (args.model,)
    run = ROOT / "runs" / args.run_id
    for model in models:
        evaluate(upstream, model, args.bits, count, repeats, run / model, args.seed)
    if args.model == "both":
        results = {}
        for model in models:
            history = json.loads((run / model / "history.json").read_text())
            results[model] = {
                (repeat_index, row["task_id"]): row["score"]
                for repeat_index, repeat in enumerate(history[-1]["eval"]["runs"])
                for row in repeat["scores"]
            }
        if results["base"].keys() != results["esopt"].keys():
            raise RuntimeError("The model evaluations used different cases")
        paired = [
            {"repeat": repeat, "task_id": case, "base": score,
             "esopt": results["esopt"][(repeat, case)]}
            for (repeat, case), score in results["base"].items()
        ]
        (run / "paired.json").write_text(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
