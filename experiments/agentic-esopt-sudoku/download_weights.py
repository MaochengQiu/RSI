"""Resume large Hugging Face shards in verified HTTP ranges."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import time

import requests

from run_local import MODEL_REVISIONS, ROOT

REPOS = {"base": "Qwen/Qwen3.5-4B",
         "esopt": "zz1358m/Qwen3.5-4B-Sudoku-Mask15-Agentic-ESOpt"}
CHUNK = 8 * 1024 * 1024


def fetch_range(url, path, start, end, size):
    length = end - start + 1
    if path.is_file() and path.stat().st_size == length:
        return
    for attempt in range(8):
        try:
            with requests.get(url, params={"range_start": start}, stream=True,
                              headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 45)) as response:
                if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                    raise ValueError(f"Unexpected range response: {response.status_code}")
                temporary = path.with_suffix(".partial")
                with temporary.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=256 * 1024):
                        output.write(chunk)
                if temporary.stat().st_size != length:
                    raise ValueError("Incomplete chunk")
                temporary.replace(path)
            return
        except (requests.RequestException, ValueError):
            if attempt == 7:
                raise
            time.sleep(min(2 ** attempt, 30))


def download(model, pool):
    directory = ROOT / "models" / model
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    for name in sorted(set(index["weight_map"].values())):
        target = directory / name
        url = f"https://huggingface.co/{REPOS[model]}/resolve/{MODEL_REVISIONS[model]}/{name}"
        metadata = requests.head(url, timeout=30)
        metadata.raise_for_status()
        size = int(metadata.headers["X-Linked-Size"])
        expected = metadata.headers["X-Linked-Etag"].strip('"')
        if target.is_file():
            with target.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() == expected:
                    continue
        parts = directory / ".ranges" / expected
        parts.mkdir(parents=True, exist_ok=True)
        offsets = range(0, size, CHUNK)
        futures = [
            pool.submit(fetch_range, url, parts / str(start), start, min(start + CHUNK, size) - 1, size)
            for start in offsets
        ]
        print(f"{model} {name}: {size:,} bytes in {len(futures)} ranges", flush=True)
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if count == 1 or count % 16 == 0 or count == len(futures):
                print(f"{name}: {count}/{len(futures)} ranges", flush=True)
        digest = hashlib.sha256()
        staged = target.with_suffix(".assembling")
        with staged.open("wb") as stream:
            for start in offsets:
                data = (parts / str(start)).read_bytes()
                digest.update(data)
                stream.write(data)
        if digest.hexdigest() != expected:
            raise ValueError(f"Checksum mismatch for {name}")
        staged.replace(target)
        print(f"Verified {target.name}", flush=True)


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        download(sys.argv[1], executor)
