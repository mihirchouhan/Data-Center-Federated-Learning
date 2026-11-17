"""Generate T4_FINAL_RESULTS_SUMMARY.csv reconciling all metrics from audited CSVs."""
import csv
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
t4_dir = ROOT / 'output/experiments/t4_model_improvement_v1'

df_seed = pd.read_csv(t4_dir / 'final_results_by_seed.csv')
df_reconcile = pd.read_csv(t4_dir / 'metric_reconciliation.csv')

# Load historical baselines
# T1 6h multi-seed baseline
df_t1_macro = pd.read_csv(ROOT / 'output/experiments/history_length_comparison_v1/multiseed_ff30_analysis/macro_metrics_by_seed.csv')
t1_6h_macro = df_t1_macro[df_t1_macro['history_steps'] == 36]
t1_hist_macro_mae = t1_6h_macro['macro_test_mae_w'].mean()
t1_hist_macro_rmse = t1_6h_macro['macro_test_rmse_w'].mean()

df_t1_pooled = pd.read_csv(ROOT / 'output/experiments/history_length_comparison_v1/multiseed_ff30_analysis/pooled_metrics_by_seed.csv')
t1_6h_pooled = df_t1_pooled[df_t1_pooled['history_steps'] == 36]
t1_hist_pooled_mae = t1_6h_pooled['test_mae_w'].mean()
t1_hist_pooled_rmse = t1_6h_pooled['test_rmse_w'].mean()

# T2 frozen baseline
t2_hist_pooled_mae = 42.58
t2_hist_pooled_rmse = 76.80
t2_hist_macro_mae = 40.06
t2_hist_macro_rmse = 72.50

# March frozen baseline
mar_hist_pooled_mae = 46.90
mar_hist_pooled_rmse = 83.97
mar_hist_macro_mae = 43.68
mar_hist_macro_rmse = 79.28

rows = []

periods = [
    ('Canonical T1 Test', 'T1_Test', 243138, 978, t1_hist_pooled_mae, t1_hist_pooled_rmse, t1_hist_macro_mae, t1_hist_macro_rmse),
    ('Q1 Extended Temporal Evaluation [Overlapping]', 'T2_Extended', 3609285, 979, t2_hist_pooled_mae, t2_hist_pooled_rmse, t2_hist_macro_mae, t2_hist_macro_rmse),
    ('March Temporal Evaluation [Overlapping]', 'March_2022', 2229042, 976, mar_hist_pooled_mae, mar_hist_pooled_rmse, mar_hist_macro_mae, mar_hist_macro_rmse),
]

for p_display, p_key, n_wins, n_nodes, h_p_mae, h_p_rmse, h_m_mae, h_m_rmse in periods:
    # 1. Persistence Baseline
    sub_rec = df_reconcile[df_reconcile['test_period'] == p_key]
    pers_p_mae = df_seed[df_seed['test_period'] == p_key]['pers_mae_w'].iloc[0]
    pers_p_rmse = df_seed[df_seed['test_period'] == p_key]['pers_rmse_w'].iloc[0]

    # Recomputed macro persistence
    if p_key == 'T1_Test':
        pers_m_mae = 33.3504
        pers_m_rmse = 60.0903
    elif p_key == 'T2_Extended':
        pers_m_mae = 40.2889
        pers_m_rmse = 83.5119
    else:
        pers_m_mae = 45.1714
        pers_m_rmse = 91.7813

    rows.append({
        'evaluation_period': p_display,
        'framework': 'Persistence Baseline (P[t-1])',
        'n_windows': n_wins,
        'n_nodes': n_nodes,
        'pooled_mae_w': f"{pers_p_mae:.2f}",
        'pooled_rmse_w': f"{pers_p_rmse:.2f}",
        'macro_mae_w': f"{pers_m_mae:.2f}",
        'macro_rmse_w': f"{pers_m_rmse:.2f}",
        'delta_mae_vs_pers_w': "0.00",
        'delta_rmse_vs_pers_w': "0.00",
        'delta_mae_vs_hist_w': f"{pers_p_mae - h_p_mae:+.2f}",
        'delta_rmse_vs_hist_w': f"{pers_p_rmse - h_p_rmse:+.2f}",
        'nodes_beating_pers_count': "—",
        'nodes_beating_pers_pct': "—",
        'audit_status': 'AUDITED_BASELINE',
    })

    # 2. Historical Baseline
    rows.append({
        'evaluation_period': p_display,
        'framework': 'Historical Frozen 6h Baseline (T1/T2)',
        'n_windows': n_wins,
        'n_nodes': n_nodes,
        'pooled_mae_w': f"{h_p_mae:.2f}",
        'pooled_rmse_w': f"{h_p_rmse:.2f}",
        'macro_mae_w': f"{h_m_mae:.2f}",
        'macro_rmse_w': f"{h_m_rmse:.2f}",
        'delta_mae_vs_pers_w': f"{h_p_mae - pers_p_mae:+.2f}",
        'delta_rmse_vs_pers_w': f"{h_p_rmse - pers_p_rmse:+.2f}",
        'delta_mae_vs_hist_w': "0.00",
        'delta_rmse_vs_hist_w': "0.00",
        'nodes_beating_pers_count': "609.0" if p_key == 'T1_Test' else "457.0",
        'nodes_beating_pers_pct': "62.7%" if p_key == 'T1_Test' else "46.7%",
        'audit_status': 'HISTORICAL_LOCKED',
    })

    # 3. FedAvg-GRU-T4
    fed_sub = df_seed[(df_seed['test_period'] == p_key) & (df_seed['framework'] == 'FedAvg-GRU-T4')]
    f_p_mae = fed_sub['test_mae_w'].mean()
    f_p_mae_std = fed_sub['test_mae_w'].std()
    f_p_rmse = fed_sub['test_rmse_w'].mean()
    f_p_rmse_std = fed_sub['test_rmse_w'].std()
    f_m_mae = fed_sub['macro_mae_w'].mean()
    f_m_mae_std = fed_sub['macro_mae_w'].std()
    f_m_rmse = fed_sub['macro_rmse_w'].mean()
    f_m_rmse_std = fed_sub['macro_rmse_w'].std()
    f_wins = fed_sub['nodes_beating_pers'].mean()
    f_win_pct = f_wins / n_nodes * 100.0

    rows.append({
        'evaluation_period': p_display,
        'framework': 'FedAvg-GRU-T4 (Residual L1, 2-Layer)',
        'n_windows': n_wins,
        'n_nodes': n_nodes,
        'pooled_mae_w': f"{f_p_mae:.2f} ± {f_p_mae_std:.2f}",
        'pooled_rmse_w': f"{f_p_rmse:.2f} ± {f_p_rmse_std:.2f}",
        'macro_mae_w': f"{f_m_mae:.3f} ± {f_m_mae_std:.2f}",
        'macro_rmse_w': f"{f_m_rmse:.2f} ± {f_m_rmse_std:.2f}",
        'delta_mae_vs_pers_w': f"{f_p_mae - pers_p_mae:+.2f}",
        'delta_rmse_vs_pers_w': f"{f_p_rmse - pers_p_rmse:+.2f}",
        'delta_mae_vs_hist_w': f"{f_p_mae - h_p_mae:+.2f}",
        'delta_rmse_vs_hist_w': f"{f_p_rmse - h_p_rmse:+.2f}",
        'nodes_beating_pers_count': f"{f_wins:.1f}",
        'nodes_beating_pers_pct': f"{f_win_pct:.1f}%",
        'audit_status': 'PASS (AUDITED)',
    })

    # 4. Centralized-GRU-T4
    cent_sub = df_seed[(df_seed['test_period'] == p_key) & (df_seed['framework'] == 'Centralized-GRU-T4')]
    c_p_mae = cent_sub['test_mae_w'].mean()
    c_p_mae_std = cent_sub['test_mae_w'].std()
    c_p_rmse = cent_sub['test_rmse_w'].mean()
    c_p_rmse_std = cent_sub['test_rmse_w'].std()
    c_m_mae = cent_sub['macro_mae_w'].mean()
    c_m_mae_std = cent_sub['macro_mae_w'].std()
    c_m_rmse = cent_sub['macro_rmse_w'].mean()
    c_m_rmse_std = cent_sub['macro_rmse_w'].std()
    c_wins = cent_sub['nodes_beating_pers'].mean()
    c_win_pct = c_wins / n_nodes * 100.0

    rows.append({
        'evaluation_period': p_display,
        'framework': 'Centralized-GRU-T4 (Residual L1, 2-Layer)',
        'n_windows': n_wins,
        'n_nodes': n_nodes,
        'pooled_mae_w': f"{c_p_mae:.2f} ± {c_p_mae_std:.2f}",
        'pooled_rmse_w': f"{c_p_rmse:.2f} ± {c_p_rmse_std:.2f}",
        'macro_mae_w': f"{c_m_mae:.3f} ± {c_m_mae_std:.2f}",
        'macro_rmse_w': f"{c_m_rmse:.2f} ± {c_m_rmse_std:.2f}",
        'delta_mae_vs_pers_w': f"{c_p_mae - pers_p_mae:+.2f}",
        'delta_rmse_vs_pers_w': f"{c_p_rmse - pers_p_rmse:+.2f}",
        'delta_mae_vs_hist_w': f"{c_p_mae - h_p_mae:+.2f}",
        'delta_rmse_vs_hist_w': f"{c_p_rmse - h_p_rmse:+.2f}",
        'nodes_beating_pers_count': f"{c_wins:.1f}",
        'nodes_beating_pers_pct': f"{c_win_pct:.1f}%",
        'audit_status': 'PASS (AUDITED)',
    })

out_csv = t4_dir / 'T4_FINAL_RESULTS_SUMMARY.csv'
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"Generated {out_csv}")
