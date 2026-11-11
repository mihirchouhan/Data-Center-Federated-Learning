#!/usr/bin/env bash
# Sequential v5 fit -> verify launcher. Creating this script does not run fits.
# Usage: bash run_remaining_experiments.sh [--dry-run] [--resume]
# The active repository is on /media; the older Desktop copy has no v5 runner.
set -Eeuo pipefail
IFS=$'\n\t'

DRY_RUN=0
RESUME=0
ACTIVE_PID=''
MASTER_LOG=''
PROGRESS_LOG=''
CURRENT_KEY='startup'
CURRENT_OUTPUT=''
LAST_LOG=''
LAST_STATUS=''

usage() {
    cat <<'USAGE'
Usage: bash run_remaining_experiments.sh [--dry-run] [--resume]

  --dry-run  Inspect metadata and print the queue; no runner calls or log writes.
  --resume   Freshly verify and skip completed runs; verify partial checkpoints
             before passing --resume to the existing single-fit runner.
  --help     Show this message.

Default execution starts new fits only. Existing non-excluded runs require
--resume. The six named historical fits are always freshly verified before skip.
No test evaluation, matrix command, deletion, or automatic replacement is used.
USAGE
}

for option in "$@"; do
    case "$option" in
        --dry-run) DRY_RUN=1 ;;
        --resume) RESUME=1 ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'ERROR: Unknown option: %s\n' "$option" >&2; usage >&2; exit 2 ;;
    esac
done

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() {
    local line
    printf -v line '[%s] %s' "$(stamp)" "$*"
    printf '%s\n' "$line"
    if [[ -n "$MASTER_LOG" ]]; then printf '%s\n' "$line" >> "$MASTER_LOG"; fi
}
die() { log "ERROR: $*" >&2; exit 1; }
progress() {
    printf '%s\t%s\t%s\t%s\t%s\n' "$(stamp)" "$CURRENT_KEY" "$1" "$CURRENT_OUTPUT" "$2" >> "$PROGRESS_LOG"
}
on_error() {
    local code=$?
    trap - ERR
    if [[ -n "$ACTIVE_PID" ]]; then
        kill -TERM "$ACTIVE_PID" 2>/dev/null || true
        wait "$ACTIVE_PID" 2>/dev/null || true
    fi
    printf '[%s] ERROR: automation stopped at %s (exit %s). Preserve all outputs.\n' "$(stamp)" "$CURRENT_KEY" "$code" >&2
    if [[ -n "$MASTER_LOG" ]]; then
        printf '[%s] FAILED %s exit=%s\n' "$(stamp)" "$CURRENT_KEY" "$code" >> "$MASTER_LOG"
    fi
    exit "$code"
}
on_signal() {
    local signal=$1 code=$2
    trap '' INT TERM HUP
    log "INTERRUPTED ($signal): stopping the current child; preserving every checkpoint and log."
    # Only one child is active. TERM also works when a noninteractive shell's
    # asynchronous child inherited SIGINT ignored. Atomic runner commits remain.
    if [[ -n "$ACTIVE_PID" ]]; then
        kill -TERM "$ACTIVE_PID" 2>/dev/null || true
        wait "$ACTIVE_PID" 2>/dev/null || true
    fi
    if [[ -n "$PROGRESS_LOG" ]]; then progress INTERRUPTED "$signal"; fi
    exit "$code"
}
trap on_error ERR
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP

# M100_PROJECT_ROOT supports relocation and isolated mock-CLI testing. It never
# changes the locked plan or permits a different experiment schema.
PROJECT_ROOT=${M100_PROJECT_ROOT:-/media/csedept/cse2014/Project/m100_training_template}
[[ -d "$PROJECT_ROOT" ]] || die "Project does not exist: $PROJECT_ROOT"
PROJECT_ROOT=$(cd -- "$PROJECT_ROOT" && pwd -P)
[[ "$PROJECT_ROOT" != *$'\n'* && "$PROJECT_ROOT" != *$'\t'* ]] || die 'Tabs/newlines in the project path are unsupported.'
cd -- "$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
PLAN='configs/experiments/v5_missing_data/gru_gap_policy_v1.json'
DATA='output/experiments/v5_missing_data/model_comparison/v1/cohorts'
OUTPUT_ROOT="$PROJECT_ROOT/output/experiments/v5_missing_data/model_comparison/v1/full_runs"
[[ -x "$PYTHON" ]] || die "Missing executable .venv/bin/python"
[[ -f src/v5_missing_data/run_policy_experiments.py ]] || die 'The selected project has no v5 runner.'
[[ ! -L "$OUTPUT_ROOT" ]] || die 'The output root must not be a symlink.'
export CUDA_VISIBLE_DEVICES='GPU-4d239ff8-8f62-c45c-7b52-23d05bc69842'
export PYTHONDONTWRITEBYTECODE=1

# Metadata inspection imports only Python's standard library: never torch or a
# model. A completion marker is only a candidate for live --stage verify.
metadata() {
    "$PYTHON" -B - "$PROJECT_ROOT" "$RESUME" "$@" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root, resume, action = Path(sys.argv[1]), sys.argv[2] == '1', sys.argv[3]
plan_path = root / 'configs/experiments/v5_missing_data/gru_gap_policy_v1.json'
expected_plan = 'ff63776db8b4be6ace0dab76572f61d426c32f3635fbb25a2280910b56120301'
def read(path):
    return json.loads(path.read_text())
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(part)
    return h.hexdigest()
def fail(message):
    raise ValueError(message)
def safe(path):
    if '\n' in str(path) or '\t' in str(path): fail('Tabs/newlines in paths are unsupported')
    if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
        fail('Unsafe experiment path: ' + str(path))
    return path
def identity(path, method, label, seed):
    safe(path)
    m = read(path / 'manifest.json')
    policy = 'no_fill' if label == 'none' else label
    if (m.get('method'), m.get('policy'), m.get('seed'), m.get('kind'), m.get('cap')) != (method, policy, seed, 'main', 200):
        fail('Wrong method/policy/seed or non-main run: ' + str(path))
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if (m.get('plan') != plan or m.get('plan_digest') != digest or m.get('settings') != plan['settings']
            or m.get('sources') != plan['sources'] or m.get('device') != plan['device']
            or m.get('cohort_receipt_sha256') != plan['cohort_receipt_sha256']):
        fail('Run configuration/provenance does not match the locked plan: ' + str(path))
    complete = path / 'completed.json'
    if complete.exists():
        c = read(complete)
        if c.get('status') != 'complete' or c.get('completed_round') != 200:
            fail('Invalid completion metadata: ' + str(complete))
        return 'complete'
    return 'partial'

try:
    if sha(plan_path) != expected_plan: fail('Locked plan file changed; do not edit expected hashes')
    plan = read(plan_path)
    output = root / 'output/experiments/v5_missing_data/model_comparison/v1/full_runs'
    if action == 'identity':
        method, label, seed, path = sys.argv[4:]
        print(identity(Path(path), method, label, int(seed)))
    elif action == 'inventory':
        # Validate input bindings before any actual fit can be launched.
        for group in ('sources', 'evidence'):
            for name, expected in plan[group].items():
                p = root / name
                if not p.resolve().is_relative_to(root.resolve()) or sha(p) != expected:
                    fail('Changed protected input: ' + name)
        methods = ('centralized_sgd', 'fedavg')
        policies = ('ff10', 'none', 'ff20', 'ff30')  # FF10 seed44 is first pending.
        seeds = (42, 43, 44)
        excluded = {(m, p, s) for m in methods for p, s in (('ff20', 42), ('ff10', 42), ('ff10', 43))}
        candidates = {}
        if output.exists():
            for child in sorted(output.iterdir()):
                manifest = child / 'manifest.json'
                if not manifest.is_file(): continue
                safe(child)
                m = read(manifest)
                p = 'none' if m.get('policy') == 'no_fill' else m.get('policy')
                key = (m.get('method'), p, m.get('seed'))
                if key[0] in methods and key[1] in policies and key[2] in seeds:
                    candidates.setdefault(key, []).append(child)
        records = []
        for label in policies:
            for seed in seeds:
                for method in methods:
                    key = (method, label, seed)
                    canonical = output / f'{method}_{label}_seed{seed}_v1'
                    found = candidates.get(key, [])
                    if len(found) > 1: fail('Duplicate run identities require review: ' + str(found))
                    if (canonical.exists() or canonical.is_symlink()) and (not found or found[0] != canonical):
                        fail('Occupied output is not the unique matching run: ' + str(canonical))
                    if found:
                        path = found[0]
                        state = identity(path, *key)
                        if key in excluded:
                            if state != 'complete': fail('A requested historical skip is incomplete: ' + str(path))
                            mode = 'EXCLUDED'
                        else:
                            if not resume: fail('Existing run requires --resume; preserve it: ' + str(path))
                            mode = 'EXISTING_' + state.upper()
                    else:
                        if key in excluded: fail('Requested historical skip is missing: ' + str(key))
                        path, state, mode = canonical, 'absent', 'NEW'
                    records.append((mode, method, label, str(seed), str(path), state))
        for row in records: print('\t'.join(row))
    else:
        fail('Unknown metadata action')
except (OSError, ValueError, KeyError, TypeError) as error:
    print('PREFLIGHT ERROR: ' + str(error), file=sys.stderr)
    sys.exit(1)
PY
}

# Require a terminal JSON object, not a grep match in an earlier log message.
result_status() {
    "$PYTHON" -B - "$1" "$2" <<'PY'
import json
from pathlib import Path
import sys
try:
    text = Path(sys.argv[1]).read_text(errors='replace')
    result = None
    decoder = json.JSONDecoder()
    for pos, ch in enumerate(text):
        if ch == '{' and (pos == 0 or text[pos-1] == '\n'):
            try: value, end = decoder.raw_decode(text[pos:])
            except ValueError: continue
            if isinstance(value, dict) and not text[pos+end:].strip(): result = value
    if result is None: raise ValueError('No terminal JSON result')
    status, completed, selected = (result.get(k) for k in ('status', 'completed_round', 'selected_round'))
    if type(completed) is not int or type(selected) is not int or not 0 <= selected <= completed <= 200:
        raise ValueError('Invalid checkpoint cursor')
    if sys.argv[2] == 'fit':
        if status not in ('complete', 'verified_complete') or completed != 200:
            raise ValueError('Fit did not complete all 200 rounds')
    else:
        if status not in ('verified_complete', 'verified_partial') or result.get('model_forward_or_scoring') is not False:
            raise ValueError('Verification status/contract is invalid')
        if status == 'verified_complete' and completed != 200:
            raise ValueError('Completion is not a full main fit')
    print(status)
except (OSError, ValueError, TypeError) as error:
    print('RESULT ERROR: ' + str(error), file=sys.stderr)
    sys.exit(1)
PY
}

inventory=$(metadata inventory) || die 'Inventory/configuration preflight failed; no fit was launched.'
BASELINES=()
QUEUE=()
while IFS=$'\t' read -r mode method policy seed path state; do
    record="$mode"$'\t'"$method"$'\t'"$policy"$'\t'"$seed"$'\t'"$path"$'\t'"$state"
    if [[ "$mode" == EXCLUDED ]]; then BASELINES+=("$record"); else QUEUE+=("$record"); fi
done <<< "$inventory"
[[ ${#BASELINES[@]} -eq 6 && ${#QUEUE[@]} -eq 18 ]] || die 'Unexpected matrix size; expected six exclusions and 18 queue entries.'

if (( DRY_RUN )); then
    log 'DRY RUN: metadata/input hashes only; no runner, model, GPU query, output directory, or log creation.'
    log "Project=$PROJECT_ROOT"
    log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (logical cuda:0 in the unchanged plan)"
    for record in "${BASELINES[@]}"; do
        IFS=$'\t' read -r mode method policy seed path state <<< "$record"
        printf 'SKIP-CANDIDATE %s_%s_seed%s -> %s [must freshly verify_complete]\n' "$method" "$policy" "$seed" "$path"
    done
    index=0
    for record in "${QUEUE[@]}"; do
        IFS=$'\t' read -r mode method policy seed path state <<< "$record"
        index=$((index + 1))
        cli_policy=$policy; [[ "$policy" != none ]] || cli_policy=no_fill
        printf 'QUEUE %02d %-18s %s_%s_seed%s cli_policy=%s -> %s\n' "$index" "$mode" "$method" "$policy" "$seed" "$cli_policy" "$path"
    done
    log 'Existing completion markers are candidates only. Real execution calls --stage verify before every skip.'
    exit 0
fi

command -v flock >/dev/null || die 'flock is required for a single automation instance.'
LOG_ROOT="$OUTPUT_ROOT/_automation_logs"
[[ ! -L "$LOG_ROOT" ]] || die 'Log root must not be a symlink.'
mkdir -p -- "$LOG_ROOT"
# Append-open never truncates an existing lock; held by the parent and its child.
exec 9>> "$LOG_ROOT/queue.lock"
flock -n 9 || die 'Another automation instance owns this output root.'
RUN_LOG_DIR=$(mktemp -d "$LOG_ROOT/run_$(date -u '+%Y%m%dT%H%M%SZ')_XXXXXX")
MASTER_LOG="$RUN_LOG_DIR/master.log"
PROGRESS_LOG="$RUN_LOG_DIR/progress.tsv"
(set -o noclobber; : > "$MASTER_LOG"; printf 'utc\texperiment\tstate\toutput\tdetail\n' > "$PROGRESS_LOG")
log "START sequential automation; project=$PROJECT_ROOT"
log "GPU UUID=$CUDA_VISIBLE_DEVICES; plan=$PLAN; cohort=$DATA; resume=$RESUME"
log "Logs=$RUN_LOG_DIR"

# Backgrounding one child solely makes signal forwarding reliable. wait always
# finishes before the next runner invocation: experiments are never parallel.
run_stage() {
    local phase=$1 method=$2 policy=$3 seed=$4 path=$5 resume_fit=${6:-0}
    local cli_policy=$policy code quoted
    [[ "$policy" != none ]] || cli_policy=no_fill
    local args=("$PYTHON" -B -m src.v5_missing_data.run_policy_experiments
        --plan "$PLAN" --data-dir "$DATA" --stage "$phase" --output-dir "$path")
    if [[ "$phase" == fit ]]; then
        args+=(--method "$method" --policy "$cli_policy" --seed "$seed")
        if (( resume_fit )); then args+=(--resume); fi
    fi
    LAST_LOG=$(mktemp "$RUN_LOG_DIR/${CURRENT_KEY}.${phase}.XXXXXX.log")
    printf -v quoted '%q ' "${args[@]}"
    log "RUN $phase $CURRENT_KEY; log=$LAST_LOG"
    log "COMMAND CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $quoted"
    progress "${phase^^}_START" "$LAST_LOG"
    "${args[@]}" >> "$LAST_LOG" 2>&1 &
    ACTIVE_PID=$!
    if wait "$ACTIVE_PID"; then code=0; else code=$?; fi
    ACTIVE_PID=''
    if (( code != 0 )); then
        progress "${phase^^}_FAILED" "exit=$code; $LAST_LOG"
        tail -n 30 -- "$LAST_LOG" >&2 || true
        die "$phase failed for $CURRENT_KEY (exit $code); no next experiment will run."
    fi
    LAST_STATUS=$(result_status "$LAST_LOG" "$phase") || die "Invalid $phase result; inspect $LAST_LOG"
    log "RESULT $CURRENT_KEY $phase=$LAST_STATUS"
    progress "${phase^^}_RESULT" "$LAST_STATUS; $LAST_LOG"
}

# No fitting is allowed until every requested historical exclusion passes a live
# complete-main-fit verification. Legacy centralized_ folder names are preserved.
for record in "${BASELINES[@]}"; do
    IFS=$'\t' read -r mode method policy seed path state <<< "$record"
    CURRENT_KEY="${method}_${policy}_seed${seed}"; CURRENT_OUTPUT=$path
    metadata identity "$method" "$policy" "$seed" "$path" >/dev/null || die 'Historical run identity mismatch.'
    run_stage verify "$method" "$policy" "$seed" "$path"
    [[ "$LAST_STATUS" == verified_complete ]] || die "Historical exclusion is not verified_complete: $CURRENT_KEY"
    log "SKIP verified historical $CURRENT_KEY ($path)"
    progress SKIPPED_VERIFIED historical
done

index=0
for record in "${QUEUE[@]}"; do
    IFS=$'\t' read -r mode method policy seed path state <<< "$record"
    index=$((index + 1))
    CURRENT_KEY="${method}_${policy}_seed${seed}"; CURRENT_OUTPUT=$path
    log "QUEUE $index/${#QUEUE[@]} $CURRENT_KEY"
    resume_fit=0
    if [[ "$mode" != NEW && ! -e "$path" ]]; then
        die "An inventoried run disappeared; refusing to restart it from scratch: $path"
    fi
    if [[ -e "$path" || -L "$path" ]]; then
        (( RESUME )) || die "Output exists; preserve it and rerun with --resume: $path"
        [[ ! -L "$path" ]] || die "Refusing symlink output: $path"
        metadata identity "$method" "$policy" "$seed" "$path" >/dev/null || die 'Existing run identity/provenance mismatch.'
        run_stage verify "$method" "$policy" "$seed" "$path"
        if [[ "$LAST_STATUS" == verified_complete ]]; then
            log "SKIP freshly verified complete $CURRENT_KEY"
            progress SKIPPED_VERIFIED resume
            continue
        fi
        [[ "$LAST_STATUS" == verified_partial ]] || die 'Existing output cannot be resumed safely.'
        resume_fit=1
    fi
    # Do not mkdir the experiment directory: the runner must create it exclusively.
    run_stage fit "$method" "$policy" "$seed" "$path" "$resume_fit"
    metadata identity "$method" "$policy" "$seed" "$path" >/dev/null || die 'Fit output identity/provenance mismatch.'
    run_stage verify "$method" "$policy" "$seed" "$path"
    [[ "$LAST_STATUS" == verified_complete ]] || die "Post-fit verification is incomplete: $CURRENT_KEY"
    log "DONE verified_complete $CURRENT_KEY"
    progress COMPLETE_VERIFIED "$LAST_LOG"
done
CURRENT_KEY=finished
log 'SUCCESS: all 24 identities are verified_complete; six historical exclusions plus 18 queue entries. No test evaluation was run.'
