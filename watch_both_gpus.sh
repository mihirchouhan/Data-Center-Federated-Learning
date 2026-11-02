#!/bin/bash

cd /media/csedept/cse2014/Project/m100_training_template || exit 1

GPU0="GPU-4d239ff8-8f62-c45c-7b52-23d05bc69842"
GPU1="GPU-7ef6875b-ba62-88e7-d0b9-e301a25b687e"

OUTPUT="output/experiments/v5_missing_data/model_comparison/v1/pilots/centralized_ff20_seed42_auto_gpu_v1"
LOG="output/experiments/v5_missing_data/model_comparison/v1/pilots/dual_gpu_watcher.log"

mkdir -p "$(dirname "$LOG")"

if [ -e "$OUTPUT" ]; then
    echo "ERROR: Output directory already exists. Stopping."
    exit 1
fi

READY0=0
READY1=0

echo "Dual-GPU watcher started: $(date)" >> "$LOG"

check_gpu() {
    local UUID="$1"
    local FREE UTIL PROCS

    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$UUID" 2>/dev/null | tr -d ' ')
    UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$UUID" 2>/dev/null | tr -d ' ')

    PROCS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null |
        grep -F "$UUID" || true)

    if [[ "$FREE" =~ ^[0-9]+$ ]] &&
       [[ "$UTIL" =~ ^[0-9]+$ ]] &&
       [ "$FREE" -ge 40000 ] &&
       [ "$UTIL" -le 10 ] &&
       [ -z "$PROCS" ]; then
        echo "$FREE"
        return 0
    fi

    return 1
}

while true; do
    FREE0=""
    FREE1=""

    if FREE0=$(check_gpu "$GPU0"); then
        READY0=$((READY0 + 1))
    else
        READY0=0
    fi

    if FREE1=$(check_gpu "$GPU1"); then
        READY1=$((READY1 + 1))
    else
        READY1=0
    fi

    echo "$(date): GPU0 ready=$READY0 free=${FREE0:-unavailable}; GPU1 ready=$READY1 free=${FREE1:-unavailable}" >> "$LOG"

    if [ "$READY0" -ge 5 ] || [ "$READY1" -ge 5 ]; then
        if [ "$READY0" -ge 5 ] && [ "$READY1" -ge 5 ]; then
            if [ "$FREE0" -ge "$FREE1" ]; then
                SELECTED="$GPU0"
            else
                SELECTED="$GPU1"
            fi
        elif [ "$READY0" -ge 5 ]; then
            SELECTED="$GPU0"
        else
            SELECTED="$GPU1"
        fi

        # Final availability check immediately before launch.
        if ! check_gpu "$SELECTED" >/dev/null; then
            READY0=0
            READY1=0
            echo "$(date): GPU became busy before launch; continuing to wait." >> "$LOG"
            sleep 60
            continue
        fi

        break
    fi

    sleep 60
done

echo "$(date): Selected GPU $SELECTED. Starting centralized-SGD smoke test." >> "$LOG"

CUDA_VISIBLE_DEVICES="$SELECTED" .venv/bin/python -B -m src.v5_missing_data.run_policy_experiments \
  --plan configs/experiments/v5_missing_data/gru_gap_policy_v1.json \
  --data-dir output/experiments/v5_missing_data/model_comparison/v1/cohorts \
  --stage smoke --method centralized_sgd --policy ff20 --seed 42 \
  --output-dir "$OUTPUT" \
  --stop-after-round 1 >> "$LOG" 2>&1

STATUS=$?
echo "$(date): Smoke test exited with status $STATUS. Manual review required." >> "$LOG"
exit "$STATUS"
