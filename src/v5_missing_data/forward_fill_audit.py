"""Coverage-only diagnostic for split-reset causal forward-fill.

No model, prediction scores, modified frozen arrays, or dense long histories.
The first L missing bins after an OBSERVED value may be filled for L=1,2,3.
Filled values never restart the age counter. Future targets use a separate,
unaltered observed-power vector. Existing output directories are never reused.
"""
from __future__ import annotations

import argparse
import csv
import json
from numbers import Integral
from pathlib import Path
import resource
import shutil
import tempfile
import time

import numpy as np

from src.history_coverage_v4 import audit_node, preflight, write_json
from src.history_data_v4 import ReferenceData, build_binned_grid, node_features, sha256
from src.v5_missing_data import eligibility
from src.v5_missing_data.eligibility import FEATURES

ROOT = Path(__file__).resolve().parents[2]
HISTORIES = (6, 36, 144)
GAP_LIMITS = (1, 2, 3)
MASKED = (7, 8)
HORIZON = 6
SPLITS = ('train', 'val', 'test')
REASONS = ('history_too_short', 'target_crosses_split_end',
           'invalid_observed_target', 'invalid_history')


def _ranges(ranges, length):
    """Require one complete, ordered partition: no overlaps or skipped bins."""
    result = list(ranges)
    previous = 0
    for pair in result:
        if len(pair) != 2:
            raise ValueError('Reset ranges must contain (lower, upper) pairs')
        lower, upper = pair
        if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in pair)
                or lower != previous or not lower < upper <= length):
            raise ValueError('Reset ranges must partition the timeline in chronological order')
        previous = upper
    if not result or previous != length:
        raise ValueError('Reset ranges must cover the complete timeline')
    return result


def _features(features):
    result = np.asarray(features, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != len(FEATURES) or not len(result):
        raise ValueError('Pass exactly one node as [time_steps, 13], never a node batch')
    return result


def causal_forward_fill(features, max_gap, reset_at):
    """Copy-only fill from same-node, same-feature ORIGINAL finite observations.

This is a causal age limit, not retrospective whole-gap interpolation. A long
gap may have its first max_gap bins filled; every later bin stays invalid until
a new observed value arrives. Histories may still look into the preceding split,
but no fill source crosses a split boundary.
"""
    source = _features(features)
    if (isinstance(max_gap, (bool, np.bool_)) or not isinstance(max_gap, Integral)
            or max_gap not in GAP_LIMITS):
        raise ValueError('max_gap must be the integer 1, 2 or 3')
    ranges = _ranges(reset_at, len(source))
    result = source.copy()
    for lower, upper in ranges:
        positions = np.arange(upper - lower)
        for channel in range(source.shape[1]):
            if channel in MASKED:
                continue
            original = source[lower:upper, channel]
            observed = np.isfinite(original)
            last_observed = np.maximum.accumulate(np.where(observed, positions, -1))
            fill = ~observed & (last_observed >= 0) & (positions - last_observed <= max_gap)
            block = result[lower:upper, channel]
            block[fill] = original[last_observed[fill]]
    return result


def eligible_origins(features, boundaries, history, *, observed_power):
    """Keep input imputation strictly separate from future-label eligibility."""
    inputs = _features(features)
    power = np.asarray(observed_power)
    if power.ndim != 1 or len(power) != len(inputs):
        raise ValueError('observed_power must be the separate original one-node power vector')
    if (isinstance(history, (bool, np.bool_)) or not isinstance(history, Integral)
            or history not in HISTORIES):
        raise ValueError('history must be 6, 36 or 144')
    if tuple(boundaries) != SPLITS:
        raise ValueError('Use the unchanged train/val/test split order')
    _ranges(boundaries.values(), len(inputs))
    bad_history = ~np.isfinite(inputs)
    bad_history[:, MASKED] = False
    history_prefix = np.r_[0, np.cumsum(bad_history.any(axis=1))]
    target_prefix = np.r_[0, np.cumsum(~np.isfinite(power))]
    selected, counts = {}, {}
    for split, (lower, upper) in boundaries.items():
        origins = np.arange(lower, upper, dtype=np.int64)
        enough = origins >= history
        inside = origins + HORIZON <= upper
        history_ok = np.zeros(len(origins), dtype=bool)
        target_ok = np.zeros(len(origins), dtype=bool)
        history_ok[enough] = history_prefix[origins[enough]] == history_prefix[origins[enough] - history]
        target_ok[inside] = target_prefix[origins[inside] + HORIZON] == target_prefix[origins[inside]]
        candidates = enough & inside
        accepted = candidates & target_ok & history_ok
        reasons = {
            'history_too_short': int((~enough).sum()),
            'target_crosses_split_end': int((enough & ~inside).sum()),
            'invalid_observed_target': int((candidates & ~target_ok).sum()),
            'invalid_history': int((candidates & target_ok & ~history_ok).sum()),
        }
        if sum(reasons.values()) + int(accepted.sum()) != len(origins):
            raise ValueError('Exclusion counts do not reconcile')
        selected[split] = origins[accepted]
        counts[split] = {'candidate_origins': len(origins), 'accepted': int(accepted.sum()), **reasons}
    return selected, counts


def count_eligible(features, boundaries, history, *, observed_power):
    return eligible_origins(features, boundaries, history, observed_power=observed_power)[1]


def replay_reference_node(reference, index, original, edges, boundaries):
    """Unchanged v4 selection and float32 target construction; one call/node."""
    selected, targets, _ = audit_node(original, edges, boundaries, histories=(6,))
    return reference.verify_node(index, original, edges, selected[6], targets)


def output_path(root, output):
    path = Path(output).absolute()
    allowed = root / 'output/experiments/v5_missing_data/forward_fill_audit'
    if (not path.resolve().is_relative_to(allowed.resolve()) or path.resolve() == allowed.resolve()
            or path.is_symlink()):
        raise ValueError('Output must be a new subdirectory under v5_missing_data/forward_fill_audit')
    return path


def _copy_new(source, destination):
    with source.open('rb') as incoming, destination.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing)


def _baseline(directory, reference_counts):
    path = directory / 'summary.json'
    baseline = json.loads(path.read_text())
    replay = baseline['original_reference_replay']
    if baseline['status'] != 'complete' or replay['status'] != 'exact_match' or replay['counts'] != reference_counts:
        raise ValueError('Existing diagnostic does not bind the expected exact reference replay')
    return baseline, {name: sha256(directory / name) for name in ('summary.json', 'masked_coverage_by_node.csv')}


def _inputs(root, config_path, reference_dir, raw_root, baseline_dir):
    config = json.loads(config_path.read_text())
    if list(FEATURES) != config['feature_names'][:-1]:
        raise ValueError('V5 feature definitions differ from original feature order')
    checked = preflight(root, config, reference_dir, raw_root)
    baseline, hashes = _baseline(baseline_dir, config['reference_counts'])
    return config, checked, baseline, hashes


def run_audit(config_path, reference_dir, raw_root, output, baseline_dir, *, root=None):
    """Explicit real-run entry point; tests inject tiny fixtures, never raw M100."""
    root = ROOT if root is None else root
    output = output_path(root, output)
    if output.exists() or output.is_symlink():
        raise ValueError('Output exists; completed and partial audits are never overwritten')
    config, checked, baseline, baseline_hashes = _inputs(root, config_path, reference_dir, raw_root, baseline_dir)
    contract, meta, edges, boundaries, preflight_info = checked
    reference = ReferenceData(reference_dir, contract, meta)
    source_paths = [Path(__file__).resolve(), Path(eligibility.__file__).resolve()]
    source_hashes = {p.name: sha256(p) for p in source_paths}
    config_hash = sha256(config_path)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(output / 'manifest.json', {
        'purpose': 'Input-coverage diagnostic only; no training, inference or prediction metrics',
        'preparation_config_sha256': config_hash, 'reference_contract_sha256': config['reference_contract_sha256'],
        'sources': source_hashes, 'baseline_diagnostic_dir': str(baseline_dir), 'baseline_artifacts': baseline_hashes,
        'history_steps': list(HISTORIES), 'gap_limits_bins': list(GAP_LIMITS), 'target_bins': HORIZON,
        'fill_policy': 'First L consecutive missing bins after an original finite value; imputed values never renew age',
        'split_policy': 'Reset fill state at every split; histories may use causal preceding-split context',
        'weather_policy': 'Do not fill channels 7/8; ignore them for input eligibility',
        'target_policy': 'All six future power bins must be finite in the separate ORIGINAL observed-power vector',
        'scalers': 'Original frozen training-window scalers unchanged; no new scaler or model dataset',
        'test_status': 'Previously observed development period, not untouched confirmatory data',
        'preflight': preflight_info})
    _copy_new(config_path, output / 'preparation_config.json')
    for path in source_paths:
        _copy_new(path, output / path.name)
    reference_totals = dict.fromkeys(SPLITS, 0)
    no_fill = {str(h): dict.fromkeys(SPLITS, 0) for h in HISTORIES}
    totals = {str(gap): {str(h): dict.fromkeys(SPLITS, 0) for h in HISTORIES} for gap in GAP_LIMITS}
    filled_bins = {str(gap): {split: dict.fromkeys(FEATURES, 0) for split in SPLITS} for gap in GAP_LIMITS}
    with tempfile.TemporaryDirectory(prefix='.ff-scratch-', dir=output) as temporary:
        grid, weather, quality = build_binned_grid(raw_root, config['raw_files'], reference.names,
                                                  edges, meta, Path(temporary), config['reading_limits'])
        try:
            # Complete the exact all-node strict reference gate BEFORE any fill diagnostics.
            for index, _ in enumerate(reference.names):
                original = node_features(grid[index], weather, edges)
                for split, count in replay_reference_node(reference, index, original, edges, boundaries).items():
                    reference_totals[split] += count
            if reference_totals != config['reference_counts']:
                raise ValueError('Original reference replay totals differ')
            write_json(output / 'reference_replay.json', {'status': 'exact_match', 'counts': reference_totals,
                'verified': ['node mapping', 'identities/order', 'splits', 'scaled target bytes', 'six-step input bytes']})
            print('Exact original reference replay passed; starting input-only filling diagnostics', flush=True)
            with (output / 'coverage_by_node.csv').open('x', newline='', encoding='utf8') as handle:
                fields = ['node', 'node_index', 'max_gap_bins', 'max_gap_minutes', 'history_steps', 'history_hours',
                          'split', 'candidate_origins', 'accepted', *REASONS]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for index, name in enumerate(reference.names):
                    original = node_features(grid[index], weather, edges)
                    observed_power = original[:, 0].copy()
                    observed_power.setflags(write=False)
                    previous = {}
                    for history in HISTORIES:
                        selected, counts = eligible_origins(original, boundaries, history, observed_power=observed_power)
                        previous[history] = selected
                        for split in SPLITS:
                            no_fill[str(history)][split] += counts[split]['accepted']
                    for gap in GAP_LIMITS:
                        filled = causal_forward_fill(original, gap, boundaries.values())
                        current = {}
                        for split, (lo, hi) in boundaries.items():
                            changed = ~np.isfinite(original[lo:hi]) & np.isfinite(filled[lo:hi])
                            for channel, feature in enumerate(FEATURES):
                                filled_bins[str(gap)][split][feature] += int(changed[:, channel].sum())
                        for history in HISTORIES:
                            selected, counts = eligible_origins(filled, boundaries, history, observed_power=observed_power)
                            current[history] = selected
                            for split, info in counts.items():
                                if not np.isin(previous[history][split], selected[split]).all():
                                    raise ValueError('Increasing fill limit reduced eligibility')
                                totals[str(gap)][str(history)][split] += info['accepted']
                                writer.writerow({'node': str(name), 'node_index': index, 'max_gap_bins': gap,
                                    'max_gap_minutes': gap * 10, 'history_steps': history, 'history_hours': history / 6,
                                    'split': split, **info})
                        for split in SPLITS:
                            if (not np.isin(current[144][split], current[36][split]).all()
                                    or not np.isin(current[36][split], current[6][split]).all()):
                                raise ValueError('Longer-history eligibility is not nested')
                        previous = current
        finally:
            del grid
    if no_fill != baseline['masked_weather_eligible_windows']:
        raise ValueError('Unfilled masked-weather coverage disagrees with existing v5 diagnostic')
    # End-of-run verification cannot silently accept edits made during a long scan.
    after = _inputs(root, config_path, reference_dir, raw_root, baseline_dir)
    if (sha256(config_path) != config_hash or after[3] != baseline_hashes
            or any(sha256(path) != source_hashes[path.name] for path in source_paths)):
        raise ValueError('Inputs or execution sources changed during audit')
    write_json(output / 'source_quality.json', quality)
    summary = {'status': 'complete', 'purpose': 'Coverage only; greater coverage is not predictive performance',
        'original_reference_replay': {'status': 'exact_match', 'counts': reference_totals},
        'masked_weather_no_fill_windows': no_fill, 'forward_fill_eligible_windows': totals,
        'filled_input_bins_by_feature_and_split': filled_bins,
        'history_steps': list(HISTORIES), 'gap_limits_bins': list(GAP_LIMITS),
        'target_policy': 'Future target power is never imputed',
        'training_or_inference_or_metric_evaluation': False}
    write_json(output / 'summary.json', summary)
    write_json(output / 'preservation.json', {'before': preflight_info, 'after': after[1][-1],
        'baseline_artifacts': baseline_hashes, 'source_hashes': source_hashes})
    write_json(output / 'resources.json', {'wall_seconds_excluding_initial_preflight': time.monotonic() - started,
        'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'scope': 'Linux process lifetime RSS high-water mark; no GPU, no dense long-history arrays'})
    artifacts = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    write_json(output / 'completed.json', {'status': 'complete', 'artifacts': artifacts,
        'preparation_config_sha256': config_hash, 'sources': source_hashes})
    return {'status': 'complete', 'output': str(output), **summary}


def verify_output(config_path, reference_dir, raw_root, output, baseline_dir, *, root=None):
    root = ROOT if root is None else root
    output = output_path(root, output)
    checked = _inputs(root, config_path, reference_dir, raw_root, baseline_dir)
    receipt = json.loads((output / 'completed.json').read_text())
    actual = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p != output / 'completed.json'}
    if (receipt['status'] != 'complete' or receipt['preparation_config_sha256'] != sha256(config_path)
            or actual != set(receipt['artifacts'])):
        raise ValueError('Incomplete or changed output/configuration inventory')
    manifest = json.loads((output / 'manifest.json').read_text())
    if (checked[3] != manifest['baseline_artifacts']
            or any(sha256(path) != receipt['sources'].get(path.name)
                   for path in (Path(__file__).resolve(), Path(eligibility.__file__).resolve()))):
        raise ValueError('Diagnostic baseline or execution sources changed')
    for name, expected in receipt['artifacts'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or (output / name).is_symlink():
            raise ValueError('Unsafe artifact path')
        if sha256(output / name) != expected:
            raise ValueError('Changed artifact: ' + name)
    return {'status': 'verified_complete', 'artifacts_checked': len(actual), 'raw_scan_or_model_evaluation': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--reference-data-dir', required=True)
    parser.add_argument('--raw-data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--baseline-diagnostic-dir', default=str(ROOT / 'output/experiments/v5_missing_data/diagnostic/v1'))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--preflight-only', action='store_true')
    modes.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        output = output_path(ROOT, args.output_dir)
        paths = [Path(value).resolve() for value in (args.config, args.reference_data_dir, args.raw_data_dir)]
        baseline = Path(args.baseline_diagnostic_dir).resolve()
        if args.verify_only:
            result = verify_output(*paths, output, baseline)
        else:
            if output.exists() or output.is_symlink():
                raise ValueError('Output exists; completed and partial audits are never overwritten')
            if args.preflight_only:
                checked = _inputs(ROOT, *paths, baseline)
                result = {'status': 'preflight_passed', 'input_integrity': checked[1][-1], 'raw_scan': False}
            else:
                result = run_audit(*paths, output, baseline)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print('FORWARD FILL AUDIT ERROR: ' + str(exc))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
