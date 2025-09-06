
#!/usr/bin/env python3
"""
Visualizer for MAC Scheduling Test Suite results CSV.
- Reads the CSV produced by mac_test_suite.py
- Prints a summary
- Generates bar charts comparing schedulers across scenarios:
    * Cell Throughput (Mbps)
    * Jain Fairness
    * PRB Utilization
    * Mean Latency (ms)
Charts are saved as PNG files in the current directory.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_bar(df, y, title, ylabel, fname):
    plt.figure()
    pv = df.pivot_table(index="scenario", columns="scheduler", values=y, aggfunc="mean")
    pv.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Scenario")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize MAC scheduler results.")
    parser.add_argument("--csv", type=str, required=True, help="Path to results CSV from mac_test_suite.py")
    parser.add_argument("--outdir", type=str, default=".", help="Directory to save plots")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print("Loaded results:")
    print(df.head())

    os.makedirs(args.outdir, exist_ok=True)

    plot_bar(df, "cell_throughput_Mbps", "Cell Throughput", "Mbps",
             os.path.join(args.outdir, "throughput_comparison.png"))
    plot_bar(df, "jain_fairness", "Jain's Fairness Index", "JFI",
             os.path.join(args.outdir, "fairness_comparison.png"))
    plot_bar(df, "prb_utilization", "PRB Utilization", "Utilization (0-1)",
             os.path.join(args.outdir, "prb_utilization_comparison.png"))
    plot_bar(df, "mean_latency_ms", "Mean HOL Latency", "ms",
             os.path.join(args.outdir, "mean_latency_comparison.png"))

    print("\\nSaved plots to:", args.outdir)

if __name__ == "__main__":
    main()
