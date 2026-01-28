import argparse
import random
import json
import os
from collections import deque

import torch
import matplotlib.pyplot as plt

from toy5g_env_adapter import DeterministicToy5GEnvAdapter

from DSACD_multibranch import (
    MultiBranchActor,
    MultiBranchQuantileCritic,
    DSACDUpdater,
    DSACDHyperParams,
    ensure_nonempty_mask,
    apply_action_mask_to_logits,
)


def _jain_fairness(x: torch.Tensor, eps: float = 1e-12) -> float:
    """Jain's fairness index for non-negative vector x."""
    x = x.float().clamp_min(0.0)
    num = float(x.sum().item()) ** 2
    den = float(x.numel()) * float((x * x).sum().item()) + eps
    return num / den


@torch.no_grad()
def evaluate_scheduler_metrics(
    eval_env: DeterministicToy5GEnvAdapter,
    actor: MultiBranchActor,
    *,
    eval_ttis: int,
    mode: str,
) -> dict:
    """Reward-agnostic evaluator.

    Metrics over `eval_ttis` TTIs:
      throughput:
        - total_cell_tput (sum over scheduled rate[u,m] across all (m,l), over horizon)
        - total_ue_tput (vector, sum throughput per UE over horizon)
      fairness:
        - alloc_counts (vector, number of times UE got any (m,l) branch)
        - jain_throughput (Jain on total UE throughput)
      PF utility:
        - pf_utility = sum_u log( (total_ue_tput[u]/eval_ttis) + eps )
      sanity:
        - invalid_action_rate (chosen action not in mask)
        - no_schedule_rate (chosen action == NOOP)
        - avg_layers_per_rbg (avg #scheduled layers per RBG)
    """
    assert mode in {"sample", "greedy"}, f"mode must be 'sample' or 'greedy', got {mode!r}"

    dev = next(actor.parameters()).device
    eval_env.reset()

    U = eval_env.n_ue
    M = eval_env.n_rbg
    L = eval_env.n_layers
    noop = eval_env.noop
    eps = eval_env.eps

    total_cell_tput = 0.0
    total_ue_tput = torch.zeros((U,), dtype=torch.float32)
    alloc_counts = torch.zeros((U,), dtype=torch.float32)

    invalid = 0
    nosched = 0
    decisions = 0

    # pairing metric: average number of layers scheduled per RBG
    layers_per_rbg_sum = 0.0
    layers_per_rbg_den = 0

    for _ in range(eval_ttis):
        eval_env.begin_tti()

        for layer_ctx in eval_env.layer_iter():
            obs = layer_ctx.obs.unsqueeze(0).to(dev)            # [1, obs_dim]
            masks = layer_ctx.masks_rbg.to(dev)                 # [M, A] bool

            logits_all = actor.forward_all(obs).squeeze(0)      # [M, A]
            logits_all = apply_action_mask_to_logits(logits_all, masks)

            if mode == "greedy":
                actions = torch.argmax(logits_all, dim=-1)      # [M]
            else:
                actions = torch.distributions.Categorical(logits=logits_all).sample()  # [M]

            # sanity stats
            for m in range(M):
                a = int(actions[m].item())
                decisions += 1
                if not bool(masks[m, a].item()):
                    invalid += 1
                if a == noop:
                    nosched += 1

            eval_env.apply_layer_actions(layer_ctx, actions.cpu())

        # After all layers, env._alloc holds the chosen schedule for this TTI
        alloc = eval_env._alloc.detach().cpu()  # [L, M]
        ue_tti = torch.zeros((U,), dtype=torch.float32)

        # pairing metric per RBG
        for m in range(M):
            scheduled_layers = 0
            for l in range(L):
                u = int(alloc[l, m].item())
                if u == noop:
                    continue
                scheduled_layers += 1
                ue_tti[u] += float(eval_env._rate(u, m).detach().cpu().item())
                alloc_counts[u] += 1.0
            layers_per_rbg_sum += float(scheduled_layers)
            layers_per_rbg_den += 1

        total_ue_tput += ue_tti
        total_cell_tput += float(ue_tti.sum().item())

        # advance env dynamics (buffers/avg_tp/reward generation)
        eval_env.end_tti()

    invalid_action_rate = invalid / max(decisions, 1)
    no_schedule_rate = nosched / max(decisions, 1)
    avg_layers_per_rbg = layers_per_rbg_sum / max(layers_per_rbg_den, 1)

    jain_throughput = _jain_fairness(total_ue_tput)
    pf_utility = float(torch.log((total_ue_tput / max(eval_ttis, 1)) + eps).sum().item())

    return {
        "mode": mode,
        "eval_ttis": int(eval_ttis),
        "total_cell_tput": float(total_cell_tput),
        "total_ue_tput": total_ue_tput,          # tensor [U]
        "alloc_counts": alloc_counts,            # tensor [U]
        "jain_throughput": float(jain_throughput),
        "pf_utility": float(pf_utility),
        "invalid_action_rate": float(invalid_action_rate),
        "no_schedule_rate": float(no_schedule_rate),
        "avg_layers_per_rbg": float(avg_layers_per_rbg),
    }

@torch.no_grad()
def evaluate_random_scheduler_metrics(
    eval_env: DeterministicToy5GEnvAdapter,
    *,
    eval_ttis: int,
    seed: int = 0,
) -> dict:
    """Random uniform-over-valid-actions baseline with the same metrics."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))

    eval_env.reset()

    U = eval_env.n_ue
    M = eval_env.n_rbg
    L = eval_env.n_layers
    noop = eval_env.noop
    eps = eval_env.eps

    total_cell_tput = 0.0
    total_ue_tput = torch.zeros((U,), dtype=torch.float32)
    alloc_counts = torch.zeros((U,), dtype=torch.float32)

    invalid = 0
    nosched = 0
    decisions = 0

    layers_per_rbg_sum = 0.0
    layers_per_rbg_den = 0

    for _ in range(eval_ttis):
        eval_env.begin_tti()

        for layer_ctx in eval_env.layer_iter():
            masks = layer_ctx.masks_rbg.cpu()  # [M, A]
            actions = torch.empty((M,), dtype=torch.long)

            for m in range(M):
                valid_idx = torch.nonzero(masks[m], as_tuple=False).view(-1)
                if valid_idx.numel() == 0:
                    actions[m] = int(noop)
                else:
                    j = int(torch.randint(0, int(valid_idx.numel()), (1,), generator=g).item())
                    actions[m] = int(valid_idx[j].item())

                a = int(actions[m].item())
                decisions += 1
                if not bool(masks[m, a].item()):
                    invalid += 1
                if a == noop:
                    nosched += 1

            eval_env.apply_layer_actions(layer_ctx, actions)

        alloc = eval_env._alloc.detach().cpu()  # [L, M]
        ue_tti = torch.zeros((U,), dtype=torch.float32)

        for m in range(M):
            scheduled_layers = 0
            for l in range(L):
                u = int(alloc[l, m].item())
                if u == noop:
                    continue
                scheduled_layers += 1
                ue_tti[u] += float(eval_env._rate(u, m).detach().cpu().item())
                alloc_counts[u] += 1.0

            layers_per_rbg_sum += float(scheduled_layers)
            layers_per_rbg_den += 1

        total_ue_tput += ue_tti
        total_cell_tput += float(ue_tti.sum().item())

        eval_env.end_tti()

    invalid_action_rate = invalid / max(decisions, 1)
    no_schedule_rate = nosched / max(decisions, 1)
    avg_layers_per_rbg = layers_per_rbg_sum / max(layers_per_rbg_den, 1)

    jain_throughput = _jain_fairness(total_ue_tput)
    pf_utility = float(torch.log((total_ue_tput / max(eval_ttis, 1)) + eps).sum().item())

    return {
        "mode": "random",
        "eval_ttis": int(eval_ttis),
        "total_cell_tput": float(total_cell_tput),
        "total_ue_tput": total_ue_tput,
        "alloc_counts": alloc_counts,
        "jain_throughput": float(jain_throughput),
        "pf_utility": float(pf_utility),
        "invalid_action_rate": float(invalid_action_rate),
        "no_schedule_rate": float(no_schedule_rate),
        "avg_layers_per_rbg": float(avg_layers_per_rbg),
    }

# -------------------------
# Minimal uniform replay
# -------------------------
class SimpleReplay:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def add(self, tr: dict):
        self.buf.append(tr)

    @property
    def size(self) -> int:
        return len(self.buf)

    def sample(self, batch_size: int, device: torch.device) -> dict:
        batch = random.sample(self.buf, batch_size)

        def stack(key, dtype=None):
            xs = [b[key] for b in batch]
            x = torch.stack([t if t.ndim > 0 else t.view(1) for t in xs], dim=0)
            if dtype is not None:
                x = x.to(dtype)
            return x.to(device)

        out = {
            "observation": stack("observation", torch.float32),
            "next_observation": stack("next_observation", torch.float32),
            "rbg_index": stack("rbg_index", torch.long).squeeze(-1),
            "action": stack("action", torch.long).squeeze(-1),
            "reward": stack("reward", torch.float32).squeeze(-1),
            "action_mask": stack("action_mask").to(torch.bool).squeeze(1)
            if batch[0]["action_mask"].ndim == 1
            else stack("action_mask").to(torch.bool),
            "next_action_mask": stack("next_action_mask").to(torch.bool).squeeze(1)
            if batch[0]["next_action_mask"].ndim == 1
            else stack("next_action_mask").to(torch.bool),
        }
        return out


# -------------------------
# Action sampling (1 forward per layer)
# -------------------------
@torch.no_grad()
def sample_actions_for_layer(actor: MultiBranchActor,
                             obs_layer: torch.Tensor,        # [obs_dim]
                             masks_rbg: torch.Tensor,        # [NRBG, A] bool
                             device: torch.device,
                             fallback_action: int) -> torch.Tensor:
    obs_b = obs_layer.unsqueeze(0).to(device)          # [1, obs_dim]
    logits_all = actor.forward_all(obs_b).squeeze(0)   # [NRBG, A]

    masks_rbg = ensure_nonempty_mask(masks_rbg.to(device), fallback_action=fallback_action)
    logits_all = apply_action_mask_to_logits(logits_all, masks_rbg)

    dist = torch.distributions.Categorical(logits=logits_all)  # batched over NRBG
    actions = dist.sample()                                    # [NRBG]
    return actions.cpu()


def main(args):
    device = torch.device(args.device)

    env = DeterministicToy5GEnvAdapter(
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        n_layers=args.n_layers,
        n_rbg=args.n_rbg,
        device=args.device,
        seed=args.seed,
        k=0.2,       # paper factor
        gmax=1.0,    # training-only normalizer (toy: keep 1.0)
    )

    eval_env = DeterministicToy5GEnvAdapter(
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        n_layers=args.n_layers,
        n_rbg=args.n_rbg,
        device=args.device,
        seed=args.seed + 12345,   # different seed OK; use same if you want exact-repeat eval
        k=0.2,       # paper factor
        gmax=1.0,    # training-only normalizer (toy: keep 1.0)
    )

    # Networks
    hp = DSACDHyperParams(
        n_quantiles=args.n_quantiles,
        beta=args.beta,
        gamma=args.gamma,
        tau=args.tau,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_alpha=args.lr_alpha,
        fallback_action=args.fallback_action,
    )

    actor = MultiBranchActor(obs_dim=args.obs_dim, n_rbg=args.n_rbg, act_dim=args.act_dim, hidden=args.hidden)
    q1 = MultiBranchQuantileCritic(obs_dim=args.obs_dim, n_rbg=args.n_rbg, act_dim=args.act_dim,
                                   n_quantiles=hp.n_quantiles, hidden=args.hidden)
    q2 = MultiBranchQuantileCritic(obs_dim=args.obs_dim, n_rbg=args.n_rbg, act_dim=args.act_dim,
                                   n_quantiles=hp.n_quantiles, hidden=args.hidden)
    q1_t = MultiBranchQuantileCritic(obs_dim=args.obs_dim, n_rbg=args.n_rbg, act_dim=args.act_dim,
                                     n_quantiles=hp.n_quantiles, hidden=args.hidden)
    q2_t = MultiBranchQuantileCritic(obs_dim=args.obs_dim, n_rbg=args.n_rbg, act_dim=args.act_dim,
                                     n_quantiles=hp.n_quantiles, hidden=args.hidden)
    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())

    updater = DSACDUpdater(
        actor=actor,
        q1=q1, q2=q2,
        q1_target=q1_t, q2_target=q2_t,
        n_rbg=args.n_rbg,
        act_dim=args.act_dim,
        hp=hp,
        device=str(device),
    )

    actor_ref = {k: v.detach().clone() for k, v in updater.actor.state_dict().items()}

    def actor_param_delta():
      s = 0.0
      for k, v in updater.actor.state_dict().items():
          s += (v.detach() - actor_ref[k]).pow(2).sum().item()
      return s ** 0.5
    replay = SimpleReplay(capacity=args.rb_capacity)

    env.reset()

    eval_log = {
        "sample": {
            "tti": [],
            "total_cell_tput": [],
            "total_ue_tput": [],
            "alloc_counts": [],
            "pf_utility": [],
            "avg_layers_per_rbg": [],
        },
        "greedy": {
            "tti": [],
            "total_cell_tput": [],
            "total_ue_tput": [],
            "alloc_counts": [],
            "pf_utility": [],
            "avg_layers_per_rbg": [],
        },
        "random": {
            "tti": [],
            "total_cell_tput": [],
            "total_ue_tput": [],
            "alloc_counts": [],
            "pf_utility": [],
            "avg_layers_per_rbg": [],
        },
    }
    
    train_log = {
        "tti": [],
        "alpha": [],
        "loss_q": [],
        "loss_pi": [],
    }

    def _append_eval(mode: str, tti: int, m: dict):
        log = eval_log[mode]
        log["tti"].append(int(tti))
        log["total_cell_tput"].append(float(m["total_cell_tput"]))
        log["total_ue_tput"].append([float(x) for x in m["total_ue_tput"].tolist()])
        log["alloc_counts"].append([float(x) for x in m["alloc_counts"].tolist()])
        log["pf_utility"].append(float(m["pf_utility"]))
        log["avg_layers_per_rbg"].append(float(m["avg_layers_per_rbg"]))

    def _plot_eval(mode: str, log: dict, out_path: str):
        if not log["tti"]:
            return
        t = log["tti"]

        fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

        # total_cell_tput + total_ue_tput (per-UE) in the same subplot
        ax = axs[0, 0]
        ax.plot(t, log["total_cell_tput"], label="total_cell_tput", linewidth=2)
        ue_tput = log["total_ue_tput"]
        if ue_tput:
            n_ue = len(ue_tput[0])
            for u in range(n_ue):
                series = [row[u] for row in ue_tput]
                ax.plot(t, series, label=f"ue{u}_tput", alpha=0.6)
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
        ax.set_title("Alloc counts")
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

    def _plot_training(log: dict, out_path: str):
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

    for tti in range(args.ttis):
        env.begin_tti()

        for layer_ctx in env.layer_iter():
            actions_rbg = sample_actions_for_layer(
                actor=updater.actor,
                obs_layer=layer_ctx.obs,
                masks_rbg=layer_ctx.masks_rbg,
                device=device,
                fallback_action=args.fallback_action,
            )
            env.apply_layer_actions(layer_ctx, actions_rbg)

        env.end_tti()
        transitions = env.export_branch_transitions()

        for tr in transitions:
            replay.add(tr)

        if replay.size >= args.learning_starts:
            batch = replay.sample(args.batch_size, device=device)
            metrics = updater.update(batch=batch, isw=None)

        # print("actor_delta_l2:", actor_param_delta())

        if tti % args.log_every == 0:
            msg = f"[TTI {tti}] Buffer={replay.size}"

            if replay.size >= args.learning_starts:
                msg += (
                    f" alpha={metrics['alpha']:.4f}"
                    f" loss_q={metrics['loss_q']:.4f}"
                    f" loss_pi={metrics['loss_pi']:.4f}"
                )
                train_log["tti"].append(int(tti))
                train_log["alpha"].append(float(metrics["alpha"]))
                train_log["loss_q"].append(float(metrics["loss_q"]))
                train_log["loss_pi"].append(float(metrics["loss_pi"]))

            print(msg)

        if (tti % args.eval_every == 0) and (replay.size >= args.learning_starts):
            msg = f"[TTI {tti}]"
            m_sample = evaluate_scheduler_metrics(
                eval_env,
                updater.actor,
                eval_ttis=args.eval_ttis,
                mode="sample",
            )
            m_greedy = evaluate_scheduler_metrics(
                eval_env,
                updater.actor,
                eval_ttis=args.eval_ttis,
                mode="greedy",
            )

            # Compact summary: throughput + key sanity + pairing + fairness
            m_rand = evaluate_random_scheduler_metrics(
                eval_env,
                eval_ttis=args.eval_ttis,
                seed=args.seed + 999,
            )

            _append_eval("sample", tti, m_sample)
            _append_eval("greedy", tti, m_greedy)
            _append_eval("random",tti, m_rand)

            msg += (
                f" | SAMPLE cell_tput={m_sample['total_cell_tput']:.2f}"
                f" jain={m_sample['jain_throughput']:.3f}"
                f" pfU={m_sample['pf_utility']:.2f}"
                f" inv={m_sample['invalid_action_rate']:.3f}"
                f" noop={m_sample['no_schedule_rate']:.3f}"
                f" layers/RBG={m_sample['avg_layers_per_rbg']:.2f}"
                f" || GREEDY cell_tput={m_greedy['total_cell_tput']:.2f}"
                f" jain={m_greedy['jain_throughput']:.3f}"
                f" pfU={m_greedy['pf_utility']:.2f}"
                f" inv={m_greedy['invalid_action_rate']:.3f}"
                f" noop={m_greedy['no_schedule_rate']:.3f}"
                f" layers/RBG={m_greedy['avg_layers_per_rbg']:.2f}"
                f" || RANDOM cell_tput={m_rand['total_cell_tput']:.2f}"
                f" jain={m_rand['jain_throughput']:.3f}"
                f" pfU={m_rand['pf_utility']:.2f}"
                f" inv={m_rand['invalid_action_rate']:.3f}"
                f" noop={m_rand['no_schedule_rate']:.3f}"
                f" layers/RBG={m_rand['avg_layers_per_rbg']:.2f}"
            )
            print(msg)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "eval_log.json"), "w", encoding="utf-8") as f:
        json.dump(eval_log, f, indent=2)
    with open(os.path.join(args.out_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2)

    _plot_eval("sample", eval_log["sample"], os.path.join(args.out_dir, "performance_with_sampling.png"))
    _plot_eval("greedy", eval_log["greedy"], os.path.join(args.out_dir, "performance_with_greedy.png"))
    _plot_eval("random", eval_log["random"], os.path.join(args.out_dir, "performance_with_random.png"))
    _plot_training(train_log, os.path.join(args.out_dir, "training_behavior.png"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--ttis", type=int, default=400)
    p.add_argument("--learning_starts", type=int, default=256)
    p.add_argument("--log_every", type=int, default=20)

    p.add_argument("--obs_dim", type=int, default=410)
    p.add_argument("--act_dim", type=int, default=11)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_rbg", type=int, default=18)
    p.add_argument("--fallback_action", type=int, default=0)

    p.add_argument("--rb_capacity", type=int, default=72000)
    p.add_argument("--batch_size", type=int, default=64)

    # DSACD knobs
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--n_quantiles", type=int, default=16)
    p.add_argument("--beta", type=float, default=0.98)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--tau", type=float, default=0.001)
    p.add_argument("--lr_actor", type=float, default=1e-4)
    p.add_argument("--lr_critic", type=float, default=1e-4)
    p.add_argument("--lr_alpha", type=float, default=1e-4)

    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--eval_ttis", type=int, default=50)
    p.add_argument("--out_dir", type=str, default=os.path.join("outputs", "train_2"))

    args = p.parse_args()
    main(args)
