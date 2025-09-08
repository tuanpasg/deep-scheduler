#!/usr/bin/env python3
"""
visualize_training_csv.py (robust)
- Works in Kaggle (no TensorBoard needed)
- Reads progress.csv and any monitor*.csv files
- Flexible x-axis detection for SB3 versions
- Maps alt keys (e.g., rollout/ep_rew_mean vs train/ep_rew_mean)
- If KPI keys (kpi/*) are missing in progress, derives proxies from monitor CSVs
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- Utility ---------
def choose_xcol(df):
    # Priority list for x-axis in progress.csv
    candidates = [
        "time/total_timesteps", "total_timesteps", "timesteps", "num_timesteps",
        "time/iterations", "iterations", "nupdates", "_step", "step"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback to index
    df.reset_index(inplace=True)
    return "index"

def moving_avg(a, k=50):
    if len(a) == 0:
        return a
    k = max(1, min(k, len(a)//10 if len(a) >= 200 else min(20, len(a))))
    s = np.cumsum(np.insert(a, 0, 0.0))
    return (s[k:] - s[:-k]) / float(k)

def safe_plot(x, y, title, xlabel, ylabel, out_png, ma=True):
    if len(x) == 0 or len(y) == 0:
        return False
    plt.figure()
    plt.plot(x, y, linewidth=1.2, label="raw")
    if ma and len(y) > 5:
        ma_y = moving_avg(np.asarray(y))
        x_ma = x[-len(ma_y):]
        plt.plot(x_ma, ma_y, linewidth=1.6, label="moving avg")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return True

# --------- Visualization ---------
ALT_KEYS = {
    # target_key: [alternatives in progress.csv]
    "ep_rew_mean": ["rollout/ep_rew_mean", "train/ep_rew_mean", "ep_rew_mean"],
    "ep_len_mean": ["rollout/ep_len_mean", "train/ep_len_mean", "ep_len_mean"],
    "policy_loss": ["train/policy_gradient_loss", "policy_gradient_loss"],
    "value_loss": ["train/value_loss", "value_loss"],
    "entropy_loss": ["train/entropy_loss", "entropy_loss"],
    "approx_kl": ["train/approx_kl", "approx_kl"],
    "clip_fraction": ["train/clip_fraction", "clip_fraction"],
    "explained_variance": ["train/explained_variance", "explained_variance"],
    # custom kpi keys (if logged via callback)
    "kpi/jain_mean": ["kpi/jain_mean"],
    "kpi/cell_tput_Mb_per_tti": ["kpi/cell_tput_Mb_per_tti"],
    "kpi/mean_latency_ms": ["kpi/mean_latency_ms"],
}

def find_first_key(df, keys):
    for k in keys:
        if k in df.columns:
            return k
    return None

def load_monitor_files(logdir):
    # Accept standard and variant filenames
    patterns = [
        os.path.join(logdir, "monitor.csv"),
        os.path.join(logdir, "monitor_*.csv"),
        os.path.join(logdir, "*.monitor.csv"),
        os.path.join(logdir, "monitor*.csv"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for f in files:
        if f not in seen and os.path.isfile(f):
            seen.add(f); uniq.append(f)
    return uniq

def derive_kpi_from_monitor(mon_files, outdir):
    # Aggregate all monitor files
    frames = []
    for f in mon_files:
        try:
            df = pd.read_csv(f, comment="#")
            df["__src"] = os.path.basename(f)
            frames.append(df)
        except Exception as e:
            print(f"[warn] failed to read {f}: {e}")
    if not frames:
        return None
    mon = pd.concat(frames, ignore_index=True, sort=False)

    # Available columns check
    have_jain = "jain" in mon.columns
    have_tput = "cell_tput_Mb" in mon.columns
    have_lat  = "mean_hol_ms" in mon.columns

    # Create a proxy x-axis: episode index across concatenated files
    x = np.arange(len(mon))

    # Plot if available
    if have_jain:
        safe_plot(x, mon["jain"].values, "Jain Fairness (episode)", "episode", "jain",
                  os.path.join(outdir, "kpi_jain_from_monitor.png"))
    if have_tput:
        safe_plot(x, mon["cell_tput_Mb"].values, "Cell Throughput per TTI (episode last)",
                  "episode", "Mb per TTI", os.path.join(outdir, "kpi_cell_tput_from_monitor.png"))
    if have_lat:
        safe_plot(x, mon["mean_hol_ms"].values, "Mean HOL Latency (episode)",
                  "episode", "ms", os.path.join(outdir, "kpi_latency_from_monitor.png"))

    # Also write a summary CSV of last values
    summary = {}
    if have_jain: summary["kpi/jain_mean(last_episode)"] = float(mon["jain"].iloc[-1])
    if have_tput: summary["kpi/cell_tput_Mb_per_tti(last_episode)"] = float(mon["cell_tput_Mb"].iloc[-1])
    if have_lat:  summary["kpi/mean_latency_ms(last_episode)"] = float(mon["mean_hol_ms"].iloc[-1])
    if summary:
        pd.DataFrame([summary]).to_csv(os.path.join(outdir, "summary_monitor_last.csv"), index=False)
    return mon

def visualize_progress(progress_csv, outdir):
    df = pd.read_csv(progress_csv)
    xcol = choose_xcol(df)
    x = df[xcol].values

    # Plot default/training keys
    plots = [
        ("ep_rew_mean", "Episode Reward (mean)"),
        ("ep_len_mean", "Episode Length (mean)"),
        ("policy_loss", "Policy Gradient Loss"),
        ("value_loss", "Value Loss"),
        ("entropy_loss", "Entropy Loss"),
        ("approx_kl", "Approx KL"),
        ("clip_fraction", "Clip Fraction"),
        ("explained_variance", "Explained Variance"),
    ]
    for target, title in plots:
        key = find_first_key(df, ALT_KEYS[target])
        if key is None:
            print(f"[info] {target} not found in progress.csv; skipping plot.")
            continue
        out_png = os.path.join(outdir, f"{target}.png")
        safe_plot(x, df[key].values, title, xcol, key, out_png)

    # Custom KPI plots if present in progress.csv (via callback)
    for kpi_key in ["kpi/jain_mean", "kpi/cell_tput_Mb_per_tti", "kpi/mean_latency_ms"]:
        key = find_first_key(df, ALT_KEYS[kpi_key])
        if key is None:
            print(f"[info] {kpi_key} not present in progress.csv (expected if no KPI callback).")
            continue
        out_png = os.path.join(outdir, f"{kpi_key.replace('/','_')}.png")
        safe_plot(x, df[key].values, kpi_key, xcol, kpi_key, out_png)

    # Write summary of last-known values for anything we plotted
    summary = {}
    for target, _ in plots:
        key = find_first_key(df, ALT_KEYS[target])
        if key is not None and len(df[key]) > 0:
            summary[key] = df[key].iloc[-1]
    for kpi_key in ["kpi/jain_mean", "kpi/cell_tput_Mb_per_tti", "kpi/mean_latency_ms"]:
        key = find_first_key(df, ALT_KEYS[kpi_key])
        if key is not None and len(df[key]) > 0:
            summary[key] = df[key].iloc[-1]
    if summary:
        pd.DataFrame([summary]).to_csv(os.path.join(outdir, "summary_progress_last.csv"), index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=str, required=True, help="Directory containing progress.csv/monitor.csv")
    ap.add_argument("--outdir", type=str, default="plots", help="Directory to save PNGs and summary CSVs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    progress_csv = os.path.join(args.logdir, "progress.csv")
    if os.path.exists(progress_csv):
        visualize_progress(progress_csv, args.outdir)
    else:
        print(f"[warn] {progress_csv} not found")

    mon_files = load_monitor_files(args.logdir)
    if mon_files:
        print(f"[info] found {len(mon_files)} monitor file(s):")
        for f in mon_files: print("   -", os.path.basename(f))
        derive_kpi_from_monitor(mon_files, args.outdir)
    else:
        print(f"[info] no monitor CSV found in {args.logdir}")

if __name__ == "__main__":
    main()