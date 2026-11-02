#!/bin/bash

cd /media/csedept/cse2014/Project/m100_training_template || exit 1

OUTPUT="output/experiments/v5_missing_data/model_comparison/v1/pilots/centralized_ff20_seed42_auto_gpu_v2"
LOG="output/experiments/v5_missing_data/model_comparison/v1/pilots/dual_gpu_watcher_v2.log"

mkdir -p "$(dirname "$LOG")"

if [ -e "$OUTPUT" ]; then
    echo "ERROR: Output directory exists. Stopping."
    exit 1
fi

READY=(0 0)
UUIDS=(
"GPU-4d239ff8-8f62-c45c-7b52-23d05bc69842"
"GPU-7ef6875b-ba62-88e7-d0b9-e301a25b687e"
)

echo "Dual GPU watcher v2 started: $(date)" >> "$LOG"

while true; do
    for i in 0 1; do
        UUID="${UUIDS[$i]}"

        INFO=$(nvidia-smi -i "$i" --query-gpu=uuid,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
        read -r FOUND FREE UTIL <<< "$(echo "$INFO" | sed 's/,/ /g')"

        PROCS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null |
            grep -F "$UUID" || true)

        if [ "$FOUND" = "$UUID" ] &&
           [[ "$FREE" =~ ^[0-9]+$ ]] &&
           [[ "$UTIL" =~ ^[0-9]+$ ]] &&
           [ "$FREE" -ge 35000 ] &&
           [ "$UTIL" -le 10 ] &&
           [ -z "$PROCS" ]; then
            READY[$i]=$((READY[$i] + 1))
        else
            READY[$i]=0
        fi

        echo "$(date): GPU$i free=${FREE:-unknown} utilization=${UTIL:-unknown}% ready=${READY[$i]}" >> "$LOG"
    done

    SELECTED=-1
    if [ "${READY[0]}" -ge 5 ] && [ "${READY[1]}" -ge 5 ]; then
        SELECTED=0
        if [ "${READY[1]}" -gt "${READY[0]}" ]; then SELECTED=1; fi
    elif [ "${READY[0]}" -ge 5 ]; then
        SELECTED=0
    elif [ "${READY[1]}" -ge 5 ]; then
        SELECTED=1
    fi

    if [ "$SELECTED" -ge 0 ]; then
        UUID="${UUIDS[$SELECTED]}"
        echo "$(date): Final recheck for GPU$SELECTED" >> "$LOG"

        INFO=$(nvidia-smi -i "$SELECTED" --query-gpu=uuid,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
        read -r FOUND FREE UTIL <<< "$(echo "$INFO" | sed 's/,/ /g')"
        PROCS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null |
            grep -F "$UUID" || true)

        if [ "$FOUND" = "$UUID" ] && [ "$FREE" -ge 35000 ] && [ "$UTIL" -le 10 ] && [ -z "$PROCS" ]; then
            break
        fi

        READY=(0 0)
    fi

    sleep 60
done

echo "$(date): Launching smoke test on GPU$SELECTED ($UUID)" >> "$LOG"

CUDA_VISIBLE_DEVICES="$UUID" .venv/bin/python -B -m src.v5_missing_data.run_policy_experiments \
  --plan configs/experiments/v5_missing_data/gru_gap_policy_v1.json \
  --data-dir output/experiments/v5_missing_data/model_comparison/v1/cohorts \
  --stage smoke --method centralized_sgd --policy ff20 --seed 42 \
  --output-dir "$OUTPUT" \
  --stop-after-round 1 >> "$LOG" 2>&1

STATUS=$?
echo "$(date): Smoke test exited with status $STATUS. Manual review required." >> "$LOG"
exit "$STATUS"
