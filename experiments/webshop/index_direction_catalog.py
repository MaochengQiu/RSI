"""Build the isolated second catalog with WebShop's original document converter."""

import argparse
import os
from pathlib import Path
import runpy
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("catalog", type=Path)
    args = parser.parse_args()
    root, catalog = args.checkout.resolve(), args.catalog.resolve()
    sys.path.insert(0, str(root))
    from web_agent_site import utils
    from web_agent_site.engine import engine
    utils.DEFAULT_FILE_PATH = str(catalog / "products.json")
    engine.DEFAULT_ATTR_PATH = str(catalog / "attributes.json")
    for name in ("resources", "resources_100", "resources_1k", "resources_100k"):
        (catalog / name).mkdir(exist_ok=True)
    os.chdir(catalog)
    runpy.run_path(str(root / "search_engine/convert_product_file_format.py"))
    if (catalog / "index").exists():
        raise FileExistsError("Index already exists. Use it or choose a fresh catalog directory.")
    subprocess.run([sys.executable, "-B", "-m", "pyserini.index.lucene", "--collection", "JsonCollection",
                    "--input", "resources", "--index", "index-building", "--generator",
                    "DefaultLuceneDocumentGenerator", "--threads", "1", "--storePositions",
                    "--storeDocvectors", "--storeRaw"], check=True)
    (catalog / "index-building").rename(catalog / "index")


if __name__ == "__main__":
    main()
