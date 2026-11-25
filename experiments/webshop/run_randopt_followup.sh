#!/usr/bin/env bash
# Run from the prepared Ubuntu environment. All stages can be resumed in place.
set -euo pipefail

scripts="$(cd -- "$(dirname -- "$0")" && pwd)"
results=/mnt/d/RSI-outputs
archive="$results/webshop/direction-selection-01"
output="$results/webshop/randopt-followup-01"
report="$output/analysis"
checkout="$HOME/rsi-data/webshop"
catalog="$results/webshop-data/direction-01"
third="$results/webshop-data/direction-02"
model="$HOME/rsi-data/models/Qwen3-4B-1cfa9a7"
python="$HOME/.venvs/rsi-alfworld/bin/python"
export JAVA_HOME="$checkout/java/usr/lib/jvm/java-17-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH"
cd "$checkout"

run() {
  "$python" -B -u "$scripts/randopt_followup.py" "$1" --checkout "$checkout" \
    --catalog "$catalog" --third-catalog "$third" --model-dir "$model" --results-root "$results" --output "$output"
}

analysis() {
  "$python" -B -u "$scripts/analyze_randopt_followup.py" "$1" --archive "$archive" --results "$output" --output "$report"
}

case "${1:-all}" in
  seal)
    if [[ ! -d "$third/index" ]]; then
      "$python" -B "$scripts/index_direction_catalog.py" "$checkout" "$third"
    fi
    run seal
    ;;
  all)
    analysis stage0
    run replay
    run complete
    analysis complete
    run scope
    analysis scope
    if "$python" -B -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["seed"] is None)' "$output/nomination.json"; then
      run final
      analysis final
    fi
    ;;
  replay|complete|scope|final)
    run "$1"
    if [[ "$1" != replay ]]; then
      analysis "$1"
    fi
    ;;
  stage0)
    analysis stage0
    ;;
  *)
    exit 2
    ;;
esac
