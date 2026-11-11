# March 2022 server-side background acquisition, version 1

The wrapper is implemented and tested with synthetic loopback data. **The real download is not authorized by Phase 1.9.1.** Authoritative user/project quota confirmation and separate explicit download-only approval are still required. Free filesystem space and a CLI flag are not quota evidence. Historical strict preflight remains FAILED; historical reproduction remains UNKNOWN. This isolated workflow does not bypass or repair either status.

## Files and scope

- `tools/march_background_v1.py`: `start`, `resume`, `status`, `stop`, and read-only `verify`.
- `tools/acquire_march_2022_v1.py`: unchanged verified transfer engine.
- `configs/acquisition/march_2022_v1.json`: unchanged verified acquisition declaration.
- `tests/test_march_background_v1.py`: synthetic lifecycle tests only.
- `output/phase1_9_1/`: implementation, test and preservation evidence.

Future lifecycle records will be created exclusively under `output/march_2022_background/v1/`. The archive and its evidence remain isolated under `data/march_2022_acquisition/v1/` and `output/march_2022_acquisition/v1/`. These real runtime directories were not created during implementation.

All commands below run **on the college Linux server**, from the project root. Use the VS Code Remote SSH terminal or an SSH terminal. The wrapper uses project `.venv` Python and the fixed verified helper/config; it provides no arbitrary executable, URL, path or configuration override through its CLI.

## Safe commands available now: no transfer

```bash
cd /media/csedept/cse2014/Project/m100_training_template

.venv/bin/python -B tools/march_background_v1.py start
.venv/bin/python -B tools/march_background_v1.py resume
.venv/bin/python -B tools/march_background_v1.py status
```

Without `--authorize-download-only`, both `start` and `resume` are dry runs: no network request, child process or runtime directory. Status is also read-only. These commands do not establish quota or remote readiness.

Optional repeated monitoring, if `watch` is installed:

```bash
watch -n 30 '.venv/bin/python -B tools/march_background_v1.py status'
```

Closing this monitor or pressing Ctrl+C in `watch` does not stop an acquisition. Status includes bounded recent helper/supervisor log lines. The helper is quiet during transfer; partial/checkpoint byte counts provide progress even when logs have no new text. Status never hashes a growing archive, fits a model, or makes an HTTP request.

## Future launch: requires separate approval and quota evidence — do not run now

Before launching, obtain the administrator's quota confirmation for the relevant user/project and filesystem, including the configured **80,000,000,000-byte retained budget plus 10,000,000,000-byte free reserve**. Recheck the Phase 1.9 metadata/preservation/quota gates without editing historical checks or baselines. Confirm the helper/config hashes below, current free space and absence of unexpected runtime files. Review the actual remote behavior when the guarded helper makes its first authorized request: strong ETag, strict lengths/ranges, TLS and no redirects are mandatory, not presumed.

```bash
cd /media/csedept/cse2014/Project/m100_training_template

sha256sum tools/acquire_march_2022_v1.py configs/acquisition/march_2022_v1.json
.venv/bin/python -B tools/march_background_v1.py status

# ONLY AFTER separate download-only approval AND authoritative quota confirmation:
.venv/bin/python -B tools/march_background_v1.py start \
  --authorize-download-only \
  --quota-confirmed
```

Required unchanged hashes:

```text
25a3abf0beb5148726eb04a26b0874880355f70ea00efbedaa25e03817253925  tools/acquire_march_2022_v1.py
4a7911cdf1cbf35b109186000b9ad697de1ba06faf66295bfb5d1338542f5725  configs/acquisition/march_2022_v1.json
```

This command returns quickly with `launched_not_yet_verified`, attempt path, supervisor PID and log path. It launches **one** detached supervisor with no controlling terminal and a separate helper process. There is no need for `&`, `nohup` or tmux. Once the supervisor is running, you may close VS Code or disconnect your laptop; computation and network traffic occur on the server.

The supervisor and worker inherit one exclusive workflow `flock`. The helper independently holds its original acquisition lock. A second wrapper launch cannot create a parallel acquisition. A direct invocation of the helper is still subject to its original exclusive namespaces/lock; do not run the helper concurrently by hand. No automatic relaunch, automatic retry beyond the helper's existing budget, matrix, extraction, archive listing or model operation exists.

## Logs, lifecycle and success criteria

Each launch or resume creates a **new** `attempt-NNNN-<UUID>/` directory. Prior attempt records/logs are never replaced.

| Artifact | Meaning |
| --- | --- |
| `workflow.json` (workflow root) | Immutable paths, exact helper/config/wrapper hashes, command, limits and archive identity |
| `.supervisor.lock` (workflow root) | Persistent new-workflow lock inode; never delete it |
| `launch.json` | Request time, authorization/quota assertions, workflow hash and resume owner |
| `spawn.json` | Spawn PID and Linux process identity |
| `supervisor.log` | Startup and supervisory errors |
| `supervisor.json` | Start time, PID, boot ID, `/proc` start ticks and detached session ID |
| `worker.json` | Actual helper argv, worker identity and start time |
| `process.log` | Persistent combined unbuffered helper stdout/stderr |
| `publication.json` | Full-byte archive verification, bound helper receipt hash and final file metadata |
| `exit.json` | End time, elapsed seconds, worker return code, supervisor status, received signals and failure reason |

Exit code zero from the helper alone does **not** mean success. After a zero exit the supervisor requires the owned matching helper session, `completed.json`, official-metadata hash, request count, expected hardlink publication, exact **31,495,884,800 bytes**, published MD5 **4c74ad95b6d837d409e24216389bb25e**, and a freshly calculated SHA-256 matching the helper receipt. It then records `verified_published`. The extra final read is a full archive-sized disk scan.

Normal lifecycle exit status is 0 for verified publication or 2 for failure; launch's exit status only describes launch success. Worker exits/signals are recorded separately. Crash, power loss, full disk or a forced supervisor kill may prevent a final exit record. Missing evidence means **unknown/incomplete**, never success.

Fast status uses live boot/PID/start-tick checks, checkpoint-chain metadata, partial length and prior publication evidence. It labels `partial_content_rehashed=false` and does not certify checkpoint payload contents. During an active write, a partial may temporarily be ahead of its last checkpoint. Status may report a transient inconsistency; a stopped inconsistency requires review. A publication shown by status is a *previous full-byte verification plus unchanged stat and receipt*, not a new content hash. For a fresh full read after the process has stopped:

```bash
.venv/bin/python -B tools/march_background_v1.py verify
```

This command is read-only and returns fresh exact byte/MD5/SHA-256 results. It refuses while the wrapper guard is held. It creates no new receipt. A verified archive is not extracted by this command.

## Safe interruption and resume

To request an interrupt on a running attempt:

```bash
.venv/bin/python -B tools/march_background_v1.py stop
.venv/bin/python -B tools/march_background_v1.py status
```

`stop` pins the verified Linux process using a pidfd (stdlib when available, glibc otherwise), then requests SIGINT. If the supervisor exists it forwards SIGINT to its worker; if only the recorded worker survives it can be signaled directly. No fallback sends signals to a potentially reused PID. No automatic SIGKILL or partial deletion occurs. Wait until status shows no live process and `guard_held=false`; stopping may take time. If stop cannot establish process identity, it fails closed.

After reviewing the previous log, checkpoint state, quota, fresh metadata and the authorization scope:

```bash
# ONLY WITH separate approval covering this resume and current quota confirmation:
.venv/bin/python -B tools/march_background_v1.py resume \
  --authorize-download-only \
  --quota-confirmed
```

Resume obtains the original owner from the identical helper `session.json` records, checks bindings/limits/paths and existing lock, and passes only `--resume-owner <recorded-owner>` to the unchanged helper. It never generates an acquisition checkpoint, truncates data, resets request counts, guesses ETags or changes expected hashes. The helper re-fetches official metadata, validates its exact recorded acquisition identity/ownership, hashes the whole partial, checks the checkpoint chain, and enforces strong ETag, If-Range and exact Content-Range before appending.

The helper's maximum **five archive GET attempts is cumulative across resumes**, not five per wrapper attempt. Metadata JSON requests are separate. Resume is not guaranteed after every interruption: bytes written after the last durable checkpoint, malformed records or changed metadata bytes can block it. Stop and investigate; never repair or delete evidence to force progress.

## Failure behavior and limits

| Event | Behavior and required action |
| --- | --- |
| SSH/VS Code disconnect or laptop sleep/shutdown | No terminal dependency after launch; server continues while powered, connected and permitted by its administrator. A real pseudo-terminal hangup was tested; physical laptop/VS Code shutdown was not reproduced. |
| Missing tmux/nohup | Neither is required; tested with an empty executable-search path and absolute Python argv. |
| Administrator logout cleanup | Detached processes remain subject to server policy. Inspection found effective `KillUserProcesses=false`, `Linger=no`; no policy was modified. Future administrative termination is outside this guarantee. |
| SIGINT / supervisor SIGTERM | Supervisor requests worker SIGINT and records the outcome when storage permits. Synthetic interruption/resume and SIGTERM preservation passed. |
| Supervisor SIGKILL | Worker may continue and retains the guard; status distinguishes live worker from missing supervisor. No second launch is allowed while that guard is held. Completion is unknown if the supervisor cannot verify it. |
| Server reboot/power loss | Process stops. No reboot service or automatic restart exists. Boot ID invalidates stale PID records; review status and resume manually only if helper validation passes. Actual reboot was not tested. |
| Low free space | Wrapper checks the conservative 90 GB threshold before spawning; helper independently checks its storage limits. No deletion or fallback path. |
| Disk or quota exhaustion mid-write | Process may fail before lifecycle evidence is durable. Preserve all files; obtain administrator guidance and inspect, without destructive cleanup. Real quota exhaustion was not induced. |
| Network interruption | Helper's bounded retries/checkpoints apply; wrapper never restarts automatically. Lifetime request cap remains five. |
| Invalid checkpoint/uncommitted prefix | Resume stops; do not invent a new checkpoint or truncate a prefix. |
| Changed strong ETag, Range ignored/inconsistent, HTTP 416, TLS failure or redirect | Helper rejects; retain response/request evidence. No weakened validators or alternate endpoints. Wrapper tests exercise changed ETag/ignored Range; other transport protections are unchanged Phase 1.8 engine behavior. |
| Existing acquisition lock | Launch/resume stops if another writer holds it; never unlink or bypass the lock. |
| Checksum failure | No verified success; partial/checkpoints retained. A complete corrupt partial cannot be made valid by repeated resume. |
| Final archive already exists | Refuse start/resume; use read-only verify or investigate missing completion evidence. Never overwrite. |
| Helper/config/wrapper changed | Hash binding stops further launches/status in an existing workflow. Preserve and review; do not update hashes to silence the check. |
| Hardlink exists without completion receipt | Crash window; read-only investigation required. No automatic regeneration of a receipt. |

## Synthetic tests

```bash
cd /media/csedept/cse2014/Project/m100_training_template
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B tests/test_march_background_v1.py
```

Only temporary fixtures and literal loopback HTTP are used. Both test controller and worker prohibit non-loopback socket connections. Synthetic workers invoke the real protected `Engine(..., synthetic=True)`; the production transfer CLI is never invoked. Temporary test evidence is retained under a unique `/tmp/m100-p191-tests-*` directory. No real acquisition namespace is needed.

Phase 1.9.1 final result: **30 tests passed, zero failures/errors/skips**. Tests cover terminal hangup, persistence of logs, duplicate guards, orphan-worker guard inheritance, safe interruption and resume, immutable prior attempt evidence, checkpoint/prefix/session failures, changed ETag and ignored Range, checksum failure, false success prevention, status read-only behavior, missing tmux/nohup, low-space rejection and helper/config preservation. Full project preservation is checked separately in `output/phase1_9_1/preservation_report.json`.

**Not tested:** real Zenodo body transfer, real archive-sized performance, actual reboot, actual disk/quota exhaustion, physical laptop shutdown, independent hostile same-UID tampering or externally signed provenance. No extraction, preparation, coverage work, GPU operation, model initialization, training, inference, evaluation or FedAvg occurred. No real request/resume readiness or quota confirmation is implied by passing the synthetic tests.
