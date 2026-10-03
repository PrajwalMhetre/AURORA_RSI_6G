#!/usr/bin/env python3
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path('simulation/dataset/output/100_samples')
H5 = BASE / 'aurora_100.h5'
OUT = BASE / 'analysis'
GRAPH = OUT / 'graphs'
OUT.mkdir(parents=True, exist_ok=True)
GRAPH.mkdir(parents=True, exist_ok=True)

if not H5.exists():
    raise FileNotFoundError(f'Not found: {H5}')

def sample_array(f, name, n):
    if name not in f:
        return None
    a = np.asarray(f[name][()])
    if a.ndim >= 1 and a.shape[0] == n:
        return a
    return None

def scalar_series(f, name, n):
    a = sample_array(f, name, n)
    if a is None:
        return np.full(n, np.nan)
    if a.ndim == 1:
        return a.astype(float)
    return a.reshape(n, -1)[:, 0].astype(float)

def vector_series(f, name, n):
    a = sample_array(f, name, n)
    if a is None:
        return None
    return a.reshape(n, -1)

with h5py.File(H5, 'r') as f:
    # Infer number of samples from first dataset that has a sample dimension.
    n = None
    for k in f.keys():
        a = f[k]
        if len(a.shape) > 0:
            n = a.shape[0]
            break
    if n is None:
        raise RuntimeError('Could not determine sample count.')

    # Complete flattened CSV: every dataset, one row per sample.
    rows = []
    for i in range(n):
        row = {'sample_id': i}
        for k in f.keys():
            a = np.asarray(f[k][()])
            v = a[i] if a.ndim > 0 and a.shape[0] == n else a
            v = np.asarray(v)
            flat = v.reshape(-1)
            if np.iscomplexobj(v):
                for j, z in enumerate(flat):
                    row[f'{k}_real_{j}'] = float(np.real(z))
                    row[f'{k}_imag_{j}'] = float(np.imag(z))
            elif flat.size == 1:
                row[k] = flat[0].item()
            else:
                for j, z in enumerate(flat):
                    row[f'{k}_{j}'] = z.item()
    dataset_csv = OUT / 'aurora_100_dataset.csv'
    pd.DataFrame(rows).to_csv(dataset_csv, index=False)

    # Metrics CSV.
    metrics = pd.DataFrame({'sample_id': np.arange(n)})
    scalar_names = {
        'baseline_sum_rate': 'baseline_sum_rate_mbps',
        'optimal_sum_rate': 'optimal_sum_rate_mbps',
        'quantized_sum_rate': 'quantized_sum_rate_mbps',
        'sanity_random_sum_rate': 'random_sum_rate_mbps',
        'noise_power_w': 'noise_power_w',
        'snr_db': 'snr_db',
    }
    for src, dst in scalar_names.items():
        s = scalar_series(f, src, n)
        if src.endswith('sum_rate'):
            s = s / 1e6
        metrics[dst] = s

    for src, prefix in [
        ('sanity_zero_sinr','zero_sinr'),
        ('sanity_random_sinr','random_sinr'),
        ('optimal_sinr','optimal_sinr'),
        ('sanity_zero_rx_power_dbm','zero_rx_power_dbm'),
        ('sanity_random_rx_power_dbm','random_rx_power_dbm'),
        ('sanity_optimizer_rx_power_dbm','optimal_rx_power_dbm'),
    ]:
        a = vector_series(f, src, n)
        if a is not None:
            metrics[f'{prefix}_user1'] = a[:,0]
            if a.shape[1] > 1:
                metrics[f'{prefix}_user2'] = a[:,1]

    metrics['optimization_gain_mbps'] = metrics['optimal_sum_rate_mbps'] - metrics['baseline_sum_rate_mbps']
    metrics['optimization_gain_percent'] = np.where(
        metrics['baseline_sum_rate_mbps'] != 0,
        100 * metrics['optimization_gain_mbps'] / metrics['baseline_sum_rate_mbps'],
        np.nan)
    metrics_csv = OUT / 'aurora_100_metrics.csv'
    metrics.to_csv(metrics_csv, index=False)

    # RIS phases.
    phase_name = 'continuous_optimal_phase' if 'continuous_optimal_phase' in f else ('target_phase' if 'target_phase' in f else None)
    if phase_name:
        p = np.asarray(f[phase_name][()])
        if p.ndim == 1:
            p = p[None, :]
        pd.DataFrame(p, columns=[f'ris_phase_{j:02d}' for j in range(p.shape[1])]).assign(sample_id=np.arange(p.shape[0])).set_index('sample_id').reset_index().to_csv(OUT / 'aurora_100_ris_phases.csv', index=False)

    # CSI magnitude stats.
    if 'true_csi_bs_user' in f:
        csi = np.asarray(f['true_csi_bs_user'][()])
        mag = np.abs(csi).reshape(csi.shape[0], -1)
        metrics['mean_bs_user_csi_abs'] = mag.mean(axis=1)
        metrics['max_bs_user_csi_abs'] = mag.max(axis=1)
        metrics.to_csv(metrics_csv, index=False)

# Graph helper.
def line_plot(x, series, labels, title, ylabel, filename):
    plt.figure(figsize=(12,6))
    for y, label in zip(series, labels):
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel('Sample ID')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH / filename, dpi=180)
    plt.close()

x = metrics.sample_id.to_numpy()
line_plot(x,
    [metrics.baseline_sum_rate_mbps, metrics.random_sum_rate_mbps, metrics.optimal_sum_rate_mbps, metrics.quantized_sum_rate_mbps],
    ['Baseline','Random RIS','Optimized RIS','Quantized RIS'],
    'Sum Rate Comparison Across Samples','Sum rate (Mbps)','01_sum_rate_comparison.png')
line_plot(x,[metrics.optimization_gain_mbps],['Optimization gain'],
    'RIS Optimization Gain','Gain (Mbps)','02_optimization_gain.png')

sinr_series=[]; sinr_labels=[]
for c,l in [('zero_sinr_user1','Zero - User 1'),('random_sinr_user1','Random - User 1'),('optimal_sinr_user1','Optimized - User 1'),('zero_sinr_user2','Zero - User 2'),('random_sinr_user2','Random - User 2'),('optimal_sinr_user2','Optimized - User 2')]:
    if c in metrics: sinr_series.append(metrics[c]); sinr_labels.append(l)
if sinr_series: line_plot(x,sinr_series,sinr_labels,'SINR Comparison Across Samples','SINR','03_sinr_comparison.png')

rx_series=[]; rx_labels=[]
for c,l in [('zero_rx_power_dbm_user1','Zero - User 1'),('random_rx_power_dbm_user1','Random - User 1'),('optimal_rx_power_dbm_user1','Optimized - User 1'),('zero_rx_power_dbm_user2','Zero - User 2'),('random_rx_power_dbm_user2','Random - User 2'),('optimal_rx_power_dbm_user2','Optimized - User 2')]:
    if c in metrics: rx_series.append(metrics[c]); rx_labels.append(l)
if rx_series: line_plot(x,rx_series,rx_labels,'Received Power Comparison','Received power (dBm)','04_received_power_comparison.png')

if 'mean_bs_user_csi_abs' in metrics:
    line_plot(x,[metrics.mean_bs_user_csi_abs,metrics.max_bs_user_csi_abs],['Mean |CSI|','Max |CSI|'],
              'BS-User CSI Magnitude','Magnitude','05_csi_magnitude.png')

phase_csv = OUT / 'aurora_100_ris_phases.csv'
if phase_csv.exists():
    p = pd.read_csv(phase_csv).drop(columns=['sample_id']).to_numpy().reshape(-1)
    plt.figure(figsize=(10,6)); plt.hist(p,bins=32); plt.title('RIS Phase Distribution'); plt.xlabel('Phase (rad)'); plt.ylabel('Count'); plt.grid(True,alpha=.25); plt.tight_layout(); plt.savefig(GRAPH/'06_ris_phase_distribution.png',dpi=180); plt.close()

# 3D scene for sample 0.
with h5py.File(H5, 'r') as f:
    if 'user_position' in f:
        pos=np.asarray(f['user_position'][()])
        if pos.ndim == 4:
            p=pos[0]
            fig=plt.figure(figsize=(10,8)); ax=fig.add_subplot(111,projection='3d')
            for u in range(p.shape[1]): ax.plot(p[:,u,0],p[:,u,1],p[:,u,2],marker='o',label=f'User {u+1}')
            def point(name,label):
                if name in f:
                    q=np.asarray(f[name][()]); q=q[0] if q.ndim>1 else q; ax.scatter(q[0],q[1],q[2],s=100,label=label)
            point('bs_position','BS'); point('ris_position','RIS')
            ax.set_title('Sample 0: BS / RIS / User Trajectories'); ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)'); ax.legend(); plt.tight_layout(); plt.savefig(GRAPH/'07_user_trajectory_3d.png',dpi=180); plt.close()

print('ANALYSIS COMPLETE')
print(f'Dataset CSV : {dataset_csv}')
print(f'Metrics CSV : {metrics_csv}')
print(f'Output dir  : {OUT}')
print('Graphs:')
for p in sorted(GRAPH.glob('*.png')): print(' ',p)
