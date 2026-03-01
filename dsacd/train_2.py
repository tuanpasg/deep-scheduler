import argparse
import random
import os
from collections import deque

import torch
import numpy as np
from pprint import pprint

from toy5g_env_adapter import DeterministicToy5GEnvAdapter, tbs_38214_bytes

from DSACD_multibranch import (
    MultiBranchActor,
    MultiBranchQuantileCritic,
    DSACDUpdater,
    DSACDHyperParams,
    ensure_nonempty_mask,
    apply_action_mask_to_logits,
)

from train_2_logging import (
    init_eval_log,
    init_train_log,
    append_eval,
    plot_eval,
    plot_training,
    save_logs,
    plot_allocation,
    plot_tput)


def _jain_fairness(x: torch.Tensor, eps: float = 1e-12) -> float:
    """Jain's fairness index for non-negative vector x."""
    x = x.float().clamp_min(0.0)
    num = float(x.sum().item()) ** 2
    den = float(x.numel()) * float((x * x).sum().item()) + eps
    return num / den

def ue_rate_under_sinr(eval_env: DeterministicToy5GEnvAdapter):
    noop = eval_env.noop
    alloc = eval_env._alloc.detach().cpu()
    selected_ues = eval_env.selected_ues
    served = torch.zeros((eval_env.n_ue,),dtype=torch.float32)
    alloc_counts = torch.zeros((eval_env.n_ue,),dtype=torch.float32)
    duration_s = eval_env.tti_ms / 1000.0
    layers_per_rbg_sum = 0.0
    layers_per_rbg_den = 0

    for m in range(eval_env.n_rbg):
        # Get all unique UEs scheduled on this RBG, ignoring NOOPs.
        scheduled_ues_on_rbg = sorted(list(set(u for u in alloc[:, m].tolist() if u != noop)))

        # Find the maximum cross-correlation between any pair of UEs on this RBG
        max_cross_corr_rbg = 0.0
        if len(scheduled_ues_on_rbg) > 1:
            corrs = []
            for i in range(len(scheduled_ues_on_rbg)):
                for j in range(i + 1, len(scheduled_ues_on_rbg)):
                    u1 = scheduled_ues_on_rbg[i]
                    u2 = scheduled_ues_on_rbg[j]
                    # CRITICAL FIX: max_cross_corr is [U, U, M], requires RBG index 'm'
                    corrs.append(eval_env.max_cross_corr[selected_ues[u1], selected_ues[u2], m])
            if corrs:
                max_cross_corr_rbg = max(corrs)

        # LOGIC FIX: The penalty should model interference, not divide the resource.
        # The original division by scheduled_layers_rbg incorrectly assumes total
        # throughput is constant, which defeats the purpose of MU-MIMO.
        penalty = 1.0 - max_cross_corr_rbg

        scheduled_layers_rbg = len([u for u in alloc[:, m].tolist() if u != noop])
        layers_per_rbg_sum += float(scheduled_layers_rbg)
        layers_per_rbg_den += 1

        for l in range(eval_env.n_layers):
            u = int(alloc[l, m].item())
            if u == noop:
                continue
            global_ue_id = selected_ues[u]
            tbs = eval_env._served_bytes(global_ue_id,m)
            # In full buffer traffic mode, the served bytes will be not contrained by the buffer size
            served[global_ue_id] += float(tbs)*penalty
            alloc_counts[global_ue_id] += 1.0

    ue_tti = (served * 8.0) / 1e6 / max(duration_s, 1e-9)
    avg_layers_per_rbg = layers_per_rbg_sum / max(layers_per_rbg_den, 1)
    return ue_tti, avg_layers_per_rbg, alloc_counts

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
    total_alloc_counts = torch.zeros((U,), dtype=torch.float32)
    total_layers_per_rbg = 0.0

    invalid = 0
    nosched = 0
    decisions = 0

    # pairing metric: average number of layers scheduled per RBG

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
            eval_env.compute_layer_transitions(layer_ctx)

        # After all layers, env._alloc holds the chosen schedule for this TTI
        ue_tti, avg_layers_per_rbg, alloc_counts = ue_rate_under_sinr(eval_env)

        total_alloc_counts += alloc_counts
        total_layers_per_rbg += avg_layers_per_rbg
        total_ue_tput += ue_tti
        total_cell_tput += float(ue_tti.sum().item())

        eval_env.finish_tti()

    avg_cell_tput = float(total_cell_tput)/float(eval_ttis)
    avg_ue_tput = total_ue_tput/float(eval_ttis)

    invalid_action_rate = invalid / max(decisions, 1)
    no_schedule_rate = nosched / max(decisions, 1)
    avg_layers_per_rbg = total_layers_per_rbg / eval_ttis


    jain_throughput = _jain_fairness(total_ue_tput)
    pf_utility = float(torch.log((avg_ue_tput) + eps).sum().item())

    alloc_counts_per_tti = total_alloc_counts / eval_ttis
    return {
        "mode": mode,
        "eval_ttis": int(eval_ttis),
        "total_cell_tput": float(total_cell_tput),
        "total_ue_tput": total_ue_tput,          # tensor [U]
        "avg_cell_tput": avg_cell_tput,
        "avg_ue_tput": avg_ue_tput,          # tensor [U]
        "alloc_counts": alloc_counts_per_tti,            # tensor [U]
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
    total_alloc_counts = torch.zeros((U,), dtype=torch.float32)

    invalid = 0
    nosched = 0
    decisions = 0

    total_layers_per_rbg = 0.0

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
            eval_env.compute_layer_transitions(layer_ctx)

        ue_tti, avg_layers_per_rbg, alloc_counts = ue_rate_under_sinr(eval_env)

        total_layers_per_rbg += avg_layers_per_rbg
        total_alloc_counts += alloc_counts
        total_ue_tput += ue_tti
        total_cell_tput += float(ue_tti.sum().item())

        eval_env.finish_tti()

    avg_cell_tput = float(total_cell_tput)/float(eval_ttis)
    avg_ue_tput = total_ue_tput/float(eval_ttis)

    invalid_action_rate = invalid / max(decisions, 1)
    no_schedule_rate = nosched / max(decisions, 1)
    avg_layers_per_rbg = total_layers_per_rbg / eval_ttis

    jain_throughput = _jain_fairness(total_ue_tput)
    pf_utility = float(torch.log((avg_ue_tput) + eps).sum().item())
    
    alloc_counts_per_tti = total_alloc_counts / eval_ttis
    return {
        "mode": "random",
        "eval_ttis": int(eval_ttis),
        "total_cell_tput": float(total_cell_tput),
        "total_ue_tput": total_ue_tput,
        "avg_cell_tput": avg_cell_tput,
        "avg_ue_tput": avg_ue_tput,
        "alloc_counts": alloc_counts_per_tti,
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
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)

    env = DeterministicToy5GEnvAdapter(
        n_ue=args.n_cell_ue,
        max_sched_ue=args.max_sched_ue,
        n_layers=args.n_layers,
        n_rbg=args.n_rbg,
        device=args.device,
        seed=args.seed,
    )

    # Update args dimensions to match environment internals
    args.obs_dim = env.obs_dim
    args.act_dim = env.act_dim

    eval_env = DeterministicToy5GEnvAdapter(
        n_ue=args.n_cell_ue,
        max_sched_ue=args.max_sched_ue,
        n_layers=args.n_layers,
        n_rbg=args.n_rbg,
        device=args.device,
        seed=args.seed + 12345,   # different seed OK; use same if you want exact-repeat eval
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

    replay = SimpleReplay(capacity=args.rb_capacity)

    env.reset()

    eval_log = init_eval_log()
    train_log = init_train_log()

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
            # print(f"LAYER ID: {layer_ctx.layer}")
            # print(f"ACTIONS: {actions_rbg}")
            # print(f"mask_rbg: {layer_ctx.masks_rbg}")
            # pprint(env.dump_state())
            transitions = env.compute_layer_transitions(layer_ctx)
            for tr in transitions:
                replay.add(tr)

        # Plot allocation map periodically
        if tti > 0 and tti % args.log_every == 0:
          plot_path = os.path.join(args.out_dir, f"alloc_tti_{tti:05d}.png")
          plot_allocation(env._alloc, env.n_ue, plot_path)

        env.finish_tti()

        # Optimizing both critics and actors with past transitions from replay buffers
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
                seed=args.seed + 999 + tti,
            )

            append_eval(eval_log, "sample", tti, m_sample)
            append_eval(eval_log, "greedy", tti, m_greedy)
            append_eval(eval_log, "random", tti, m_rand)

            msg += (
                f" \nSAMPLE cell_tput[Mbps]={m_sample['avg_cell_tput']:.2f}"
                f" jain={m_sample['jain_throughput']:.3f}"
                f" pfU={m_sample['pf_utility']:.2f}"
                f" inv={m_sample['invalid_action_rate']:.3f}"
                f" noop={m_sample['no_schedule_rate']:.3f}"
                f" layers/RBG={m_sample['avg_layers_per_rbg']:.2f}"
                f" \nGREEDY cell_tput[Mbps]={m_greedy['avg_cell_tput']:.2f}"
                f" jain={m_greedy['jain_throughput']:.3f}"
                f" pfU={m_greedy['pf_utility']:.2f}"
                f" inv={m_greedy['invalid_action_rate']:.3f}"
                f" noop={m_greedy['no_schedule_rate']:.3f}"
                f" layers/RBG={m_greedy['avg_layers_per_rbg']:.2f}"
                f" \nRANDOM cell_tput[Mbps]={m_rand['avg_cell_tput']:.2f}"
                f" jain={m_rand['jain_throughput']:.3f}"
                f" pfU={m_rand['pf_utility']:.2f}"
                f" inv={m_rand['invalid_action_rate']:.3f}"
                f" noop={m_rand['no_schedule_rate']:.3f}"
                f" layers/RBG={m_rand['avg_layers_per_rbg']:.2f}"
            )
            print(msg)

    save_logs(args.out_dir, eval_log, train_log)
    plot_eval("sample", eval_log["sample"], os.path.join(args.out_dir, "performance_with_sampling.png"))
    plot_eval("greedy", eval_log["greedy"], os.path.join(args.out_dir, "performance_with_greedy.png"))
    plot_eval("random", eval_log["random"], os.path.join(args.out_dir, "performance_with_random.png"))
    plot_tput("random",eval_log["random"]["avg_cell_tput"],
            "greedy",eval_log["greedy"]["avg_cell_tput"],
            "sample",eval_log["sample"]["avg_cell_tput"],
            eval_log["random"]["tti"],
            os.path.join(args.out_dir, "throughput_comparison.png"))
    plot_training(train_log, os.path.join(args.out_dir, "training_behavior.png"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--ttis", type=int, default=400)
    p.add_argument("--learning_starts", type=int, default=256)
    p.add_argument("--log_every", type=int, default=20)

    p.add_argument("--obs_dim", type=int, default=164)
    p.add_argument("--act_dim", type=int, default=5)
    p.add_argument("--n_cell_ue", type=int, default=4)
    p.add_argument("--max_sched_ue", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_rbg", type=int, default=18)
    p.add_argument("--fallback_action", type=int, default=-1)

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

    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--eval_ttis", type=int, default=20)
    p.add_argument("--out_dir", type=str, default=os.path.join("outputs", "train_2"))

    args = p.parse_args()
    main(args)
