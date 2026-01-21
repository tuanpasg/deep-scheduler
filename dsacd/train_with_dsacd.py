# train_dsacd_torchrl.py
# Option B: env steps per-TTI; replay buffer stores ONE transition per branch (layer, RBG).
# Uses TorchRL replay buffer (PER) + your DSACDUpdater update core.
#
# Requirements:
#   - Your environment must provide a way to:
#       (1) enumerate branch decisions within a TTI
#       (2) apply actions for each branch
#       (3) finalize TTI to obtain rewards/next states
#       (4) export per-branch transitions for replay
#
# This script is clarity-first: each block corresponds to Algorithm 2′ steps.

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import torch

# TorchRL replay buffer (PER)
from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from tensordict import TensorDict

# Your DSACD core (actor/critic/updater)
from train.DSACD.DSACD import Actor, QuantileCritic, DSACDUpdater, DSACDHyperParams

# -------------------------
# 1) Environment contract
# -------------------------
# You can keep your existing env. Just provide these methods (or adapt below):
#
# env.reset() -> None
#
# env.begin_tti() -> None
#   Optional: clear internal logs for the current TTI
#
# env.branch_iter() -> Iterable[BranchCtx]
#   Returns an iterable of branch contexts for this TTI.
#   Each BranchCtx identifies one (layer, rbg) decision and provides:
#     - obs:  torch.Tensor [obs_dim]
#     - mask: torch.BoolTensor [A]
#
# env.apply_branch_action(branch_ctx, action: int) -> None
#   Applies the action for this branch decision (still within the same TTI).
#
# env.end_tti() -> None
#   Finalizes the TTI, computes realized throughput, etc.
#
# env.export_branch_transitions() -> List[Dict[str, torch.Tensor]]
#   Returns per-branch transitions for the TTI, each dict contains:
#     observation, action, reward, next_observation, action_mask, next_action_mask
#   with per-branch shapes:
#     observation      [obs_dim]
#     action           [] (int)
#     reward           []
#     next_observation [obs_dim]
#     action_mask      [A] bool
#     next_action_mask [A] bool
#
# NOTE: The paper’s “reward observed at t+1” can be implemented internally by the env.
#       The replay transition should still be (s, a, r, s') when you export it.


# -------------------------
# 2) Small helpers
# -------------------------
def to_td(transition: Dict[str, torch.Tensor]) -> TensorDict:
    """Convert one per-branch transition dict into a TensorDict of batch size [] (scalar batch)."""
    # Ensure dtypes are correct for DSACDUpdater expectations.
    td = TensorDict(
        {
            "observation": transition["observation"].float(),
            "action": transition["action"].long(),
            "reward": transition["reward"].float(),
            "next_observation": transition["next_observation"].float(),
            "action_mask": transition["action_mask"].bool(),
            "next_action_mask": transition["next_action_mask"].bool(),
        },
        batch_size=[],
    )
    return td


@torch.no_grad()
def select_action(actor: Actor, obs: torch.Tensor, mask: torch.Tensor, device: torch.device) -> int:
    """Sample action from masked categorical policy."""
    obs_b = obs.to(device).unsqueeze(0)        # [1, obs_dim]
    mask_b = mask.to(device).unsqueeze(0)      # [1, A]
    logits = actor(obs_b, mask_b)              # [1, A]
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample().item()
    return int(a)


# -------------------------
# 3) Training config
# -------------------------
@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "cuda"
    total_ttis: int = 200_000

    # Replay / PER
    rb_capacity: int = 1_000_000
    batch_size: int = 512
    learning_starts: int = 10_000          # number of transitions, not TTIs
    gradient_steps_per_tti: int = 1

    # PER hyperparams
    per_alpha: float = 0.6                 # priority exponent ω (paper notation)
    per_beta0: float = 0.4                 # IS exponent start (optional schedule)
    per_beta1: float = 1.0                 # IS exponent end
    per_beta_anneal_steps: int = 200_000   # anneal over TTIs
    per_eps: float = 1e-6

    # Logging
    log_every: int = 200


def linear_anneal(step: int, start: float, end: float, duration: int) -> float:
    if duration <= 0:
        return end
    mix = min(max(step / duration, 0.0), 1.0)
    return start + mix * (end - start)


# -------------------------
# 4) Main training loop
# -------------------------
def train(env, obs_dim: int, act_dim: int, cfg: TrainConfig, hp: DSACDHyperParams):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    # --- Networks ---
    actor = Actor(obs_dim=obs_dim, act_dim=act_dim, hidden=256)
    q1 = QuantileCritic(obs_dim=obs_dim, act_dim=act_dim, n_quantiles=hp.n_quantiles, hidden=256)
    q2 = QuantileCritic(obs_dim=obs_dim, act_dim=act_dim, n_quantiles=hp.n_quantiles, hidden=256)
    q1_t = QuantileCritic(obs_dim=obs_dim, act_dim=act_dim, n_quantiles=hp.n_quantiles, hidden=256)
    q2_t = QuantileCritic(obs_dim=obs_dim, act_dim=act_dim, n_quantiles=hp.n_quantiles, hidden=256)
    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())

    updater = DSACDUpdater(
        actor=actor, q1=q1, q2=q2, q1_target=q1_t, q2_target=q2_t,
        act_dim=act_dim, hp=hp, device=str(device)
    )

    # --- PER replay buffer (TorchRL) ---
    # Storage holds TensorDicts; sampler handles priorities and IS weights.
    storage = LazyTensorStorage(max_size=cfg.rb_capacity)

    # We anneal beta manually by calling sampler.update_beta(...) (see below).
    sampler = PrioritizedSampler(
        max_capacity=cfg.rb_capacity,
        alpha=cfg.per_alpha,
        beta=cfg.per_beta0,
        eps=cfg.per_eps,
    )

    rb = ReplayBuffer(storage=storage, sampler=sampler, batch_size=cfg.batch_size)

    # --- Reset env ---
    env.reset()

    num_added = 0
    for tti in range(cfg.total_ttis):
        # ========== COLLECT (Option B: one outer step is a TTI) ==========
        env.begin_tti()

        # Decide actions for all branches in this TTI
        for branch_ctx in env.branch_iter():
            # branch_ctx must provide obs/mask
            obs = branch_ctx.obs        # torch.Tensor [obs_dim]
            mask = branch_ctx.mask      # torch.BoolTensor [A]
            a = select_action(updater.actor, obs, mask, device)
            env.apply_branch_action(branch_ctx, a)

        # Finalize TTI so rewards/next states become available
        env.end_tti()

        # Export per-branch transitions for replay
        transitions: List[Dict[str, torch.Tensor]] = env.export_branch_transitions()

        # Add to replay buffer (one entry per branch)
        for tr in transitions:
            td = to_td(tr)
            # Option: initialize priority to current max to guarantee sampling.
            rb.add(td)
            num_added += 1

        # ========== LEARN ==========
        if num_added >= cfg.learning_starts:
            # Anneal PER beta (optional but common)
            beta = linear_anneal(tti, cfg.per_beta0, cfg.per_beta1, cfg.per_beta_anneal_steps)
            rb.sampler.beta = beta  # keep it explicit / readable

            for _ in range(cfg.gradient_steps_per_tti):
                batch_td = rb.sample()  # returns a TensorDict with batch_size [B]
                # TorchRL prioritized sampler typically adds sampling metadata like:
                #   - "_weight" : IS weights [B]
                #   - "_index"  : indices in storage [B]
                # Key names can vary slightly by version, so we handle both patterns.

                # Try common TorchRL names; adjust if your version differs.
                if "_weight" in batch_td.keys():
                    isw = batch_td["_weight"]
                elif "weight" in batch_td.keys():
                    isw = batch_td["weight"]
                else:
                    isw = None

                if "_index" in batch_td.keys():
                    indices = batch_td["_index"]
                elif "index" in batch_td.keys():
                    indices = batch_td["index"]
                else:
                    indices = None  # if missing, you must obtain indices from your sampler version

                # Convert TensorDict -> plain dict of tensors for DSACDUpdater
                batch = {
                    "observation": batch_td["observation"],
                    "action": batch_td["action"],
                    "reward": batch_td["reward"],
                    "next_observation": batch_td["next_observation"],
                    "action_mask": batch_td["action_mask"],
                    "next_action_mask": batch_td["next_action_mask"],
                }

                metrics = updater.update(batch=batch, isw=isw)
                prio = metrics["priority"]  # [B]

                # Update priorities in replay (needs indices)
                if indices is not None:
                    rb.update_priority(indices, prio)

        # ========== LOG ==========
        if (tti + 1) % cfg.log_every == 0 and num_added >= cfg.learning_starts:
            # metrics are last step’s values (tensors)
            print(
                f"tti={tti+1}  "
                f"replay={num_added}  "
                f"alpha={float(metrics['alpha']):.4f}  "
                f"loss_q={float(metrics['loss_q']):.4f}  "
                f"loss_pi={float(metrics['loss_pi']):.4f}  "
                f"loss_alpha={float(metrics['loss_alpha']):.4f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # ----- You provide env + dimensions -----
    # Example:
    #   env = YourSchedulerEnv(...)
    #   obs_dim = env.obs_dim_per_branch
    #   act_dim = env.act_dim  # = |U|+1
    raise NotImplementedError("Instantiate your environment here and call train().")


if __name__ == "__main__":
    main()
