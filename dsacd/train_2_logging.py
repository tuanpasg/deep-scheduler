import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

def init_eval_log():
    return {
        "sample": {
            "tti": [],
            "total_cell_tput": [],
            "total_ue_tput": [],
            "avg_cell_tput": [],
            "avg_ue_tput": [],
            "alloc_counts": [],
            "pf_utility": [],
            "avg_layers_per_rbg": [],
        },
        "greedy": {
            "tti": [],
            "total_cell_tput": [],
            "total_ue_tput": [],
            "avg_cell_tput": [],
            "avg_ue_tput": [],
            "alloc_counts": [],
            "pf_utility": [],
            "avg_layers_per_rbg": [],
        },
        "random": {
            "tti": [],
            "total_cell_tput": [],
            "total_ue_tput": [],
            "avg_cell_tput": [],
            "avg_ue_tput": [],
            "alloc_counts": [],
            "pf_utility": [],
            "avg_layers_per_rbg": [],
        },
    }


def init_train_log():
    return {
        "tti": [],
        "alpha": [],
        "loss_q": [],
        "loss_pi": [],
    }


def append_eval(eval_log: dict, mode: str, tti: int, metrics: dict):
    log = eval_log[mode]
    log["tti"].append(int(tti))
    log["total_cell_tput"].append(float(metrics["total_cell_tput"]))
    log["total_ue_tput"].append([float(x) for x in metrics["total_ue_tput"].tolist()])
    log["avg_cell_tput"].append(float(metrics["avg_cell_tput"]))
    log["avg_ue_tput"].append([float(x) for x in metrics["avg_ue_tput"].tolist()])
    log["alloc_counts"].append([float(x) for x in metrics["alloc_counts"].tolist()])
    log["pf_utility"].append(float(metrics["pf_utility"]))
    log["avg_layers_per_rbg"].append(float(metrics["avg_layers_per_rbg"]))

def plot_tput(mode0,log0,mode1,log1,mode2,log2,tti,out_path: str):
    if not tti:
        return
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    ax.plot(tti,log0,label=mode0)
    ax.plot(tti,log1,label=mode1)
    ax.plot(tti,log2,label=mode2)
    ax.set_title("Average Cell throughput")
    ax.set_xlabel("TTI")
    ax.set_ylabel("Throughput[Mbps]")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_eval(mode: str, log: dict, out_path: str):
    if not log["tti"]:
        return
    t = log["tti"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # total_cell_tput + total_ue_tput (per-UE) in the same subplot
    ax = axs[0, 0]
    # ax.plot(t, log["avg_cell_tput"], label="avg_cell_tput", linewidth=2)
    ue_tput = log["avg_ue_tput"]
    if ue_tput:
        n_ue = len(ue_tput[0])
        for u in range(n_ue):
            series = [row[u] for row in ue_tput]
            ax.plot(t, series, label=f"avg_ue{u}_tput", alpha=0.6)
    ax.set_title("Cell + UE throughput")
    ax.set_xlabel("TTI")
    ax.set_ylabel("Throughput")
    ax.legend(loc="best", fontsize=8, ncol=2)

    # alloc_counts
    ax = axs[0, 1]
    alloc = log["alloc_counts"]
    if alloc:
        n_ue = len(alloc[0])
        for u in range(n_ue):
            series = [row[u] for row in alloc]
            ax.plot(t, series, label=f"ue{u}_alloc", alpha=0.7)
    ax.set_title("Average No of Allocations per TTI")
    ax.set_xlabel("TTI")
    ax.set_ylabel("Count")
    ax.legend(loc="best", fontsize=8, ncol=2)

    # pf_utility
    ax = axs[1, 0]
    ax.plot(t, log["pf_utility"], label="pf_utility", linewidth=2)
    ax.set_title("PF utility")
    ax.set_xlabel("TTI")
    ax.set_ylabel("Utility")

    # avg_layers_per_rbg
    ax = axs[1, 1]
    ax.plot(t, log["avg_layers_per_rbg"], label="avg_layers_per_rbg", linewidth=2)
    ax.set_title("Avg layers per RBG")
    ax.set_xlabel("TTI")
    ax.set_ylabel("Layers/RBG")

    fig.suptitle(f"Performance with {mode}")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training(log: dict, out_path: str):
    if not log["tti"]:
        return
    t = log["tti"]
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    axs[0].plot(t, log["alpha"], label="alpha", linewidth=2)
    axs[0].set_title("Alpha")
    axs[0].set_xlabel("TTI")
    axs[0].set_ylabel("Value")

    axs[1].plot(t, log["loss_q"], label="loss_q", linewidth=2)
    axs[1].set_title("Loss Q")
    axs[1].set_xlabel("TTI")
    axs[1].set_ylabel("Value")

    axs[2].plot(t, log["loss_pi"], label="loss_pi", linewidth=2)
    axs[2].set_title("Loss Pi")
    axs[2].set_xlabel("TTI")
    axs[2].set_ylabel("Value")

    fig.suptitle("Training behavior")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_logs(out_dir: str, eval_log: dict, train_log: dict):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval_log.json"), "w", encoding="utf-8") as f:
        json.dump(eval_log, f, indent=2)
    with open(os.path.join(out_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2)

def plot_allocation(alloc_tensor, n_ue, out_path: str):
    # Convert to numpy for plotting
    # alloc_tensor shape: [Layers, RBGs]
    data = alloc_tensor.cpu().numpy()
    
    fig = plt.figure(figsize=(12, 6))
    
    # We use a discrete colormap. 'tab20' is good for up to 20 UEs.
    # We add +1 to account for a "NOOP" or "Empty" value if necessary.
    cmap = plt.get_cmap('tab20', n_ue + 1) 
    
    # Plotting the heatmap
    im = plt.imshow(data, aspect='auto', cmap=cmap, interpolation='nearest', vmin=0, vmax=n_ue)
    
    # Add Colorbar with UE labels
    cbar = plt.colorbar(im, ticks=range(n_ue + 1))
    cbar.set_label('UE Index')
    
    plt.title(f"MU-MIMO Allocation Map (RBG vs Layers) - TTI: {os.path.basename(out_path)}")
    plt.xlabel("Resource Block Groups (RBG)")
    plt.ylabel("Spatial Layers")
    
    # Set integer ticks for clarity
    if data.shape[1] > 0:
        plt.xticks(np.arange(data.shape[1]))
    if data.shape[0] > 0:
        plt.yticks(np.arange(data.shape[0]))
    
    plt.grid(which='both', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

# Example usage:
# plot_allocation(self._alloc, self.n_ue)