"""Download a bounded prefix of the pinned product file and extract a 10k catalog."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

BASE = "https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/ce990ff"
BLOCK = 8 * 1024**2


def fetch(source, total, limit, directory):
    def chunk(start):
        end = min(start + BLOCK, limit) - 1
        path = directory / f"{source}.{start}"
        curl = shutil.which("curl") or "curl"
        response = subprocess.check_output([
            curl, "-fsSL", "--retry", "3", "--max-time", "180", "--max-filesize", str(end-start+1),
            "-r", f"{start}-{end}", "-o", str(path), "-w", "%{http_code} %{header_json}",
            f"{BASE}/{source}?download=true&rsi_range={start}-{end}"])
        status, header = response.decode().split(" ", 1)
        assert status == "206" and json.loads(header)["content-range"] == [f"bytes {start}-{end}/{total}"]
        assert path.stat().st_size == end-start+1
        return path
    with ThreadPoolExecutor(max_workers=4) as pool:
        return b"".join(p.read_bytes() for p in pool.map(chunk, range(0, limit, BLOCK)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--offset", type=int, choices=(10000, 20000), default=10000)
    args = parser.parse_args()
    out = args.output
    out.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        prefix_mib = 112 if args.offset == 10000 else 176
        raw = fetch("items_shuffle.json", 5479720229, prefix_mib*1024**2, directory)
        source = raw.decode("utf-8", errors="surrogateescape")
        decoder = json.JSONDecoder()
        pos = source.index("[") + 1
        products = []
        for i in range(args.offset + 10000):
            while source[pos] in ", \t\r\n":
                pos += 1
            product, pos = decoder.raw_decode(source, pos)
            if i >= args.offset:
                products.append(product)
        del source, raw
        asins = {p["asin"] for p in products}
        assert len(products) == 10000
        attrs = json.loads(fetch("items_ins_v2.json", 186295270, 186295270, directory))
        for name, obj in (("products", products), ("attributes", {k:v for k,v in attrs.items() if k in asins})):
            path = out / f"{name}.json"
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
            temp.replace(path)
        print("Catalog records", len(products), "distinct ASINs", len(asins), flush=True)


if __name__ == "__main__":
    main()
