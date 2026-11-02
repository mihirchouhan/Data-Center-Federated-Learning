"""Deterministic sampled-pass plans, independent of method and policy labels."""
import hashlib
import numpy as np


def stream_seed(seed, domain, round_id=0, node_index=0):
    return int(np.random.SeedSequence([seed, domain, round_id, node_index]).generate_state(1)[0])


def row_hash(rows):
    return hashlib.sha256(np.asarray(rows, dtype='<i8').tobytes()).hexdigest()


def verify_budget_totals(context):
    """Independently replay client sampling and counts, without loading examples."""
    protocol = context.budgets['protocol']
    counts = np.asarray([c['m_k'] for c in context.budgets['clients']], dtype=np.int64)
    totals = {}
    for seed in protocol['seeds']:
        examples = steps = 0
        for r in range(1, protocol['rounds'] + 1):
            selected = np.random.default_rng(np.random.SeedSequence([seed, 1, r, 0])).choice(
                len(counts), size=protocol['clients_per_round'], replace=False)
            examples += int(counts[selected].sum())
            steps += int(((counts[selected] + protocol['batch_size'] - 1) // protocol['batch_size']).sum())
        totals[str(seed)] = {'examples_processed_per_method_policy': examples,
                            'optimizer_steps_per_method_policy': steps}
    if totals != context.budgets['budget_totals']:
        raise ValueError('Replayed per-client budgets differ from the prepared protocol')
    return totals


def round_plan(registry, seed, round_id, clients_per_round=98, batch_size=512):
    if type(round_id) is not int or round_id < 1 or not 0 < clients_per_round <= len(registry.names):
        raise ValueError('Invalid round/client population')
    chosen = np.random.default_rng(np.random.SeedSequence([seed, 1, round_id, 0])).choice(
        len(registry.names), size=clients_per_round, replace=False)
    result = []
    for node_index in chosen:
        i = int(node_index)
        m, n = registry.m[i], registry.counts[i]
        sample_seed, shuffle_seed = stream_seed(seed, 3, round_id, i), stream_seed(seed, 2, round_id, i)
        # A uniform unordered subset, canonicalized before an independent shuffle.
        subset = np.sort(np.random.default_rng(sample_seed).choice(n, size=m, replace=False))
        rows = registry.offsets[i] + subset[np.random.default_rng(shuffle_seed).permutation(m)]
        ids = registry.data.ids[rows]
        result.append({'client_id': registry.names[i], 'node_index': i, 'native_count': n, 'm_k': m,
            'sample_seed': sample_seed, 'shuffle_seed': shuffle_seed,
            'row_positions_sha256': row_hash(rows), 'window_ids_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
            'batch_sizes': [min(batch_size, m-first) for first in range(0, m, batch_size)],
            'optimizer_steps': (m+batch_size-1)//batch_size, 'rows': rows.astype(np.int64)})
    return result


def metadata(plan):
    return [{key: value for key, value in item.items() if key != 'rows'} for item in plan]


def due(round_id, cap, every=10):
    return round_id == 0 or round_id == cap or round_id % every == 0


def select(score, best_score, round_id, best_round):
    if not np.isfinite(score):
        raise ValueError('Nonfinite validation selection value')
    return (float(score), round_id) if score < best_score else (best_score, best_round)
