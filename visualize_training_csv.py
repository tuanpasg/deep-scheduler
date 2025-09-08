
#!/usr/bin/env python3
"""
visualize_training_csv.py
Read Stable-Baselines3 CSV logs (progress.csv, monitor.csv) and plot key curves.
Designed for Kaggle where TensorBoard is limited.

Usage:
  python visualize_training_csv.py --logdir runs/ppo_mac_fair --outdir plots

Outputs:
  - PNG charts in --outdir
  - summary_progress_last.csv with last-seen values
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

KEYS_DEFAULT = [
    ("rollout/ep_len_mean", "Episode Length (mean)"),
    ("train/ep_rew_mean", "Episode Reward (mean)"),
    ("kpi/jain_mean", "Jain Fairness (EMA)"),
    ("kpi/cell_tput_Mb_per_tti", "Cell Throughput (Mb/TTI)"),
    ("kpi/mean_latency_ms", "Mean HOL Latency (ms)"),
    ("train/policy_gradient_loss", "Policy Gradient Loss"),
    ("train/value_loss", "Value Loss"),
    ("train/entropy_loss", "Entropy Loss"),
    ("train/approx_kl", "Approx KL"),
    ("train/clip_fraction", "Clip Fraction"),
    ("train/explained_variance", "Explained Variance")
]

def _safe_plot(df, xcol, ycol, title, out_png):
    if ycol not in df.columns or xcol not in df.columns:
        return False
    x = df[xcol].values
    y = df[ycol].values
    if len(x) == 0:
        return False
    plt.figure()
    plt.plot(x, y, linewidth=1.5)
    plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return True

def _moving_avg(a, k=20):
    if len(a) < 2:
        return a
    s = np.cumsum(np.insert(a, 0, 0.0))
    k = max(1, min(k, len(a)))
    return (s[k:] - s[:-k]) / float(k)

def visualize_progress(progress_csv, outdir):
    df = pd.read_csv(progress_csv)
    xcol = "time/total_timesteps" if "time/total_timesteps" in df.columns else df.index.name or "index"
    if xcol == "index":
        df = df.reset_index()

    for key, title in KEYS_DEFAULT:
        out_png = os.path.join(outdir, f"{key.replace('/', '_')}.png")
        _safe_plot(df, xcol, key, title, out_png)

    summary = {}
    for key, _ in KEYS_DEFAULT:
        if key in df.columns and len(df[key]) > 0:
            summary[key] = df[key].iloc[-1]
    sum_df = pd.DataFrame([summary])
    sum_csv = os.path.join(outdir, "summary_progress_last.csv")
    sum_df.to_csv(sum_csv, index=False)

def visualize_monitor(monitor_csv, outdir):
    df = pd.read_csv(monitor_csv, comment='#')
    if 'r' in df.columns:
        r = df['r'].values
        ma = _moving_avg(r, k=max(5, len(r)//50))
        plt.figure()
        plt.plot(np.arange(len(r)), r, alpha=0.4, linewidth=1.0, label='episode reward')
        if len(ma) > 0:
            plt.plot(np.arange(len(ma)) + (len(r)-len(ma)), ma, linewidth=2.0, label='moving avg')
        plt.title("Episode Reward (Monitor)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png = os.path.join(outdir, "monitor_episode_reward.png")
        plt.savefig(out_png, dpi=150)
        plt.close()

    if 'l' in df.columns:
        l = df['l'].values
        ma = _moving_avg(l, k=max(5, len(l)//50))
        plt.figure()
        plt.plot(np.arange(len(l)), l, alpha=0.4, linewidth=1.0, label='episode length')
        if len(ma) > 0:
            plt.plot(np.arange(len(ma)) + (len(l)-len(ma)), ma, linewidth=2.0, label='moving avg')
        plt.title("Episode Length (Monitor)")
        plt.xlabel("Episode")
        plt.ylabel("Length (steps)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png = os.path.join(outdir, "monitor_episode_length.png")
        plt.savefig(out_png, dpi=150)
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=str, required=True, help="Directory containing progress.csv/monitor.csv")
    ap.add_argument("--outdir", type=str, default="plots", help="Directory to save PNGs and summary CSVs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    progress_csv = os.path.join(args.logdir, "progress.csv")
    monitor_csv = os.path.join(args.logdir, "monitor.csv")

    if os.path.exists(progress_csv):
        visualize_progress(progress_csv, args.outdir)
    else:
        print(f"[warn] {progress_csv} not found")

    if os.path.exists(monitor_csv):
        visualize_monitor(monitor_csv, args.outdir)
    else:
        print(f"[warn] {monitor_csv} not found")

if __name__ == "__main__":
    main()
