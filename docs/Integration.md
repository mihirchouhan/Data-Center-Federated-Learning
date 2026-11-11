# Dataset, library and server integrations

Documentation review: 11 November 2026. See [index](INDEX.md) and [evidence and verification](DOCUMENTATION_AUDIT.md).

## Actual integrations

| Integration | Files / boundary | Risks and failure handling |
| --- | --- | --- |
| M100/ExaData Jan–Feb | configs/m100_all_nodes.json, src/download.py, Parquet partitions | Checksum/extraction receipts; missing metrics/units stop preparation |
| March acquisition | helper v2/background v3/isolated v3 under tools; acquisition configs | Explicit quota/authorization, TLS, validators/Range, bounded retry, checksum/no-clobber publication |
| NumPy/pandas/PyArrow | requirements.txt and data/preparation modules | Timestamp/schema/shape and dependency-drift checks |
| PyTorch/CUDA | model, FedAvg and v5 runtime | CUDA required by v5; deterministic settings; driver and physical-device identity need operator review |
| Native XGBoost/SciPy | requirements-experiments.txt, experiment_models.py | Categorical mapping/residual boosting in v1; Phase 2 two-target adapters absent |
| Linux filesystem | fcntl, fsync, exclusive files, atomic commits | Capacity is not quota proof; exhaustion can prevent evidence writes |
| Local report export | Existing analysis/thesis scripts and plotting/export tools | New unused output versions only; browser/export requirements differ by package |

No project payment/email/cloud-storage/LLM service integration was found. Editor tools are not experimental integrations and no REST frontend invokes these services.

## Acquisition boundary

Intended March identity: Zenodo 7590061, version 1.0.0, 22-03.tar, exact 31,495,884,800 bytes, MD5 `4c74ad95b6d837d409e24216389bb25e`. These are expected fields, not verified current local success. No official metadata/network request occurred here.

Strong ETag is preferred. Helper v2 permits constrained Last-Modified fallback only when ETag is absent and its HTTP-date policy passes. Resume sends If-Range and requires exact 206/Content-Range plus unchanged validator. Missing/changed validator, unexpected 200, redirects and 416 fail closed. Never guess a validator, truncate partials or replace a final archive.

Isolated v3 carries one historical consumed request, leaving at most four new within cumulative five. Preserve old sessions/locks. Authorization flags are not quota evidence or user permission. Filesystem free space does not establish user/project quota.

Background execution is server-side, independent of laptop/SSH lifecycle, with attempts/logs/exit status. Reboot recovery is not automatic. Later wrappers inspect nearest existing ancestor for fresh-path capacity without creating directories. The [original background guide](acquisition/MARCH_BACKGROUND_v1.md) is historical; later source-pinned isolated workflow reports take precedence for newer sessions. Do not run an old guide's transfer command blindly.

An externally managed March payload tree is excluded from content hashing because a transfer may be active. This review did not inspect archive contents, interfere with that process or establish March coverage/success.

## Dependencies and future integrations

Existing venv/dependency metadata is recorded in [Setup](Setup.md). Broad ranges alone don't reproduce full environments; resume binds source/config/software/device. Don't upgrade libraries to repair unexplained mismatches. Logical GPU remapping does not guarantee cross-hardware bit identity.

Any future dataset, remote-client deployment or energy-control integration needs a separately versioned interface, failure/permission specification and tests. No dependency install, network request or integration launch occurred here.
