"""Official-style DSACD training loop (TorchRL) for Option B (per-TTI env, per-branch replay).

This script is designed to be *paper-faithful* to the multi-branch actor design:
- One forward pass per decision stage (layer) produces logits for ALL RBGs.
- You sample actions for all RBGs from those logits (with per-RBG masks).
- You then STORE transitions per-branch in replay buffer (one item per (layer, rbg)).

It uses:
- DSACD_multibranch.py (MultiBranchActor, MultiBranchQuantileCritic, DSACDUpdater)
- TorchRL Prioritized replay buffer

You must adapt the environment adapter class at the bottom to your simulator.

Key: Keep the data contract for `export_branch_transitions()` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from tensordict import TensorDict
from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from DSACD_multibranch import (
    MultiBranchActor,
    MultiBranchQuantileCritic,
    DSACDUpdater,
    DSACDHyperParams,
    apply_action_mask_to_logits,
    ensure_nonempty_mask,
)


# -------------------------
# 1) Replay conversion
# -------------------------

def transition_to_tensordict(tr: Dict[str, torch.Tensor]) -> TensorDict:
    """Convert one per-branch transition dict into a TensorDict with batch_size=[].

    Required keys:
      observation [obs_dim]
      next_observation [obs_dim]
      rbg_index [] (int64)
      action [] (int64)
      reward []
      action_mask [A] bool
      next_action_mask [A] bool
    """
    return TensorDict(
        {
            "observation": tr["observation"].float(),
            "next_observation": tr["next_observation"].float(),
            "rbg_index": tr["rbg_index"].long(),
            "action": tr["action"].long(),
            "reward": tr["reward"].float(),
            "action_mask": tr["action_mask"].bool(),
            "next_action_mask": tr["next_action_mask"].bool(),
        },
        batch_size=[],
    )


# -------------------------
# 2) Action selection (one forward per layer)
# -------------------------

@torch.no_grad()
def sample_actions_for_layer(
    actor: MultiBranchActor,
    obs_layer: torch.Tensor,          # [obs_dim]
    masks_rbg: torch.Tensor,          # [NRBG, A] bool
    device: torch.device,
    fallback_action: int = -1,
) -> torch.Tensor:
    """Sample action per RBG using ONE actor forward pass.

    Returns:
        actions_rbg: [NRBG] int64
    """
    obs_b = obs_layer.unsqueeze(0).to(device)  # [1, obs_dim]
    logits_all = actor.forward_all(obs_b).squeeze(0)  # [NRBG, A]

    masks_rbg = ensure_nonempty_mask(masks_rbg.to(device), fallback_action=fallback_action)
    logits_all = apply_action_mask_to_logits(logits_all, masks_rbg)

    dist = torch.distributions.Categorical(logits=logits_all)  # batched over NRBG
    actions = dist.sample()  # [NRBG]
    return actions.cpu()


# -------------------------
# 3) Training config
# -------------------------

@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "cuda"

    total_ttis: int = 200_000
    rb_capacity: int = 1_000_000
    batch_size: int = 512

    learning_starts: int = 10_000          # transitions (per-branch) before learning starts
    gradient_steps_per_tti: int = 1        # updates per outer step (TTI)

    # PER
    per_alpha: float = 0.6                 # priority exponent ω
    per_beta0: float = 0.4                 # IS exponent start
    per_beta1: float = 1.0                 # IS exponent end
    per_beta_anneal_steps: int = 200_000
    per_eps: float = 1e-6

    log_every: int = 200


def linear_anneal(step: int, start: float, end: float, duration: int) -> float:
    if duration <= 0:
        return end
    t = min(max(step / duration, 0.0), 1.0)
    return start + t * (end - start)


# -------------------------
# 4) Main training loop
# -------------------------

def train(env, obs_dim: int, act_dim: int, n_rbg: int, cfg: TrainConfig, hp: DSACDHyperParams):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    # ---- Build networks ----
    actor = MultiBranchActor(obs_dim=obs_dim, n_rbg=n_rbg, act_dim=act_dim, hidden=256)
    q1 = MultiBranchQuantileCritic(obs_dim=obs_dim, n_rbg=n_rbg, act_dim=act_dim, n_quantiles=hp.n_quantiles, hidden=256)
    q2 = MultiBranchQuantileCritic(obs_dim=obs_dim, n_rbg=n_rbg, act_dim=act_dim, n_quantiles=hp.n_quantiles, hidden=256)
    q1_t = MultiBranchQuantileCritic(obs_dim=obs_dim, n_rbg=n_rbg, act_dim=act_dim, n_quantiles=hp.n_quantiles, hidden=256)
    q2_t = MultiBranchQuantileCritic(obs_dim=obs_dim, n_rbg=n_rbg, act_dim=act_dim, n_quantiles=hp.n_quantiles, hidden=256)
    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())

    updater = DSACDUpdater(
        actor=actor,
        q1=q1, q2=q2,
        q1_target=q1_t, q2_target=q2_t,
        n_rbg=n_rbg,
        act_dim=act_dim,
        hp=hp,
        device=str(device),
    )

    # ---- PER replay buffer ----
    storage = LazyTensorStorage(max_size=cfg.rb_capacity)
    sampler = PrioritizedSampler(
        max_capacity=cfg.rb_capacity,
        alpha=cfg.per_alpha,
        beta=cfg.per_beta0,
        eps=cfg.per_eps,
    )
    rb = ReplayBuffer(storage=storage, sampler=sampler, batch_size=cfg.batch_size)

    # ---- Reset env ----
    env.reset()

    num_added = 0
    last_metrics: Optional[Dict[str, torch.Tensor]] = None

    for tti in range(cfg.total_ttis):

        # ===========================
        # COLLECT ONE TTI (Option B)
        # ===========================
        env.begin_tti()

        # For each layer decision stage, do ONE actor forward to get all RBG actions.
        for layer_ctx in env.layer_iter():
            obs_layer = layer_ctx.obs                # [obs_dim]
            masks_rbg = layer_ctx.masks_rbg          # [NRBG, A] bool

            actions_rbg = sample_actions_for_layer(
                actor=updater.actor,
                obs_layer=obs_layer,
                masks_rbg=masks_rbg,
                device=device,
                fallback_action=hp.fallback_action,
            )

            env.apply_layer_actions(layer_ctx, actions_rbg)     # apply actions for all RBGs

        env.end_tti()

        # Export PER-BRANCH transitions (one per (layer, rbg))
        transitions: List[Dict[str, torch.Tensor]] = env.export_branch_transitions()
        for tr in transitions:
            rb.add(transition_to_tensordict(tr))
            num_added += 1

        # ===========================
        # LEARN
        # ===========================
        if num_added >= cfg.learning_starts:
            # anneal PER beta
            sampler.beta = linear_anneal(tti, cfg.per_beta0, cfg.per_beta1, cfg.per_beta_anneal_steps)

            for _ in range(cfg.gradient_steps_per_tti):
                batch_td = rb.sample()  # TensorDict batch_size=[B]

                # TorchRL versions may differ in metadata keys. Handle common cases.
                isw = batch_td.get("_weight", batch_td.get("weight", None))
                indices = batch_td.get("_index", batch_td.get("index", None))

                batch = {
                    "observation": batch_td["observation"],
                    "next_observation": batch_td["next_observation"],
                    "rbg_index": batch_td["rbg_index"],
                    "action": batch_td["action"],
                    "reward": batch_td["reward"],
                    "action_mask": batch_td["action_mask"],
                    "next_action_mask": batch_td["next_action_mask"],
                }

                metrics = updater.update(batch=batch, isw=isw)
                last_metrics = metrics

                # update priorities
                if indices is not None:
                    rb.update_priority(indices, metrics["priority"])

        # ===========================
        # LOG
        # ===========================
        if (tti + 1) % cfg.log_every == 0 and last_metrics is not None:
            print(
                f"tti={tti+1}  replay={num_added}  "
                f"alpha={float(last_metrics['alpha']):.4f}  "
                f"loss_q={float(last_metrics['loss_q']):.4f}  "
                f"loss_pi={float(last_metrics['loss_pi']):.4f}  "
                f"loss_alpha={float(last_metrics['loss_alpha']):.4f}"
            )


# -------------------------
# 5) Environment adapter template
# -------------------------
# Adapt this to your simulator. It's intentionally minimal and clarity-first.

class LayerCtx:
    """A single decision stage (layer) context."""
    def __init__(self, obs: torch.Tensor, masks_rbg: torch.Tensor, layer_id: int):
        self.obs = obs                      # [obs_dim]
        self.masks_rbg = masks_rbg          # [NRBG, A] bool
        self.layer_id = layer_id


class EnvAdapterTemplate:
    """Template for an Option-B scheduler environment.

    You can wrap your existing environment with these methods.
    """

    def reset(self) -> None:
        raise NotImplementedError

    def begin_tti(self) -> None:
        """Optional: clear per-TTI logs/buffers."""
        pass

    def layer_iter(self) -> Iterable[LayerCtx]:
        """Yield LayerCtx for each MU-MIMO layer decision stage within this TTI."""
        raise NotImplementedError

    def apply_layer_actions(self, layer_ctx: LayerCtx, actions_rbg: torch.Tensor) -> None:
        """Apply [NRBG] actions for the given layer."""
        raise NotImplementedError

    def end_tti(self) -> None:
        """Finalize the TTI and compute rewards/next states internally."""
        raise NotImplementedError

    def export_branch_transitions(self) -> List[Dict[str, torch.Tensor]]:
        """Return per-branch transitions for the completed TTI.

        Each returned dict must contain:
          observation, next_observation: [obs_dim]
          rbg_index: [] int64
          action: [] int64
          reward: []
          action_mask, next_action_mask: [A] bool
        """
        raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(
        "This is a template script. Instantiate your environment adapter and call train(...)."
    )
