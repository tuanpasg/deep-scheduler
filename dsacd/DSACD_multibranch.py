"""DSACD (Distributional Soft Actor-Critic Discrete) - paper-faithful multi-branch core.

This file implements the *paper-shaped* computation graph:

- Single forward pass per decision stage (layer): actor outputs z then reshape
  to [NRBG, (|U|+1)] and applies softmax along the action dimension for EACH RBG.

To keep Prioritized Experience Replay (PER) and debugging simple, we still store
ONE transition per branch in replay, but each transition additionally carries
the branch index m (rbg_index). During updates, we compute logits/quantiles for
ALL branches in one forward pass and then select the corresponding branch slice.

Replay entry (per-branch) fields and shapes:
  observation        : [B, obs_dim]        (layer-level state s_l)
  rbg_index          : [B]                (int64 in [0..NRBG-1])
  action             : [B]                (int64 in [0..A-1])
  reward             : [B]
  next_observation   : [B, obs_dim]
  action_mask        : [B, A]             (bool, mask for that specific branch m)
  next_action_mask   : [B, A]             (bool)

Model outputs:
  actor.forward_all(obs)        -> logits_all  : [B, NRBG, A]
  critic.forward_all(obs)       -> q_all       : [B, NRBG, A, N]

Selected branch slices (using rbg_index):
  logits_m : [B, A]
  q_m      : [B, A, N]

This layout matches the paper's "one forward pass generates actions for all RBGs"
*and* keeps the loss definitions per-branch, which is what you want for clear
Algorithm-2′ traceability.

Important implementation notes:
- Masking: invalid actions receive logit=-inf. If a mask row has no valid action,
  we force the last action (typically "no allocation") to be valid.
- PER IS weights: applied PER-SAMPLE to each objective (critic, actor, alpha).
- Entropy target is state-dependent via |A(s,m)| from the branch mask.

This is clarity-first code: each update step is grouped to match Algorithm 2′.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


# -----------------------------
# Masking + quantile loss utils
# -----------------------------

def ensure_nonempty_mask(mask: torch.Tensor, fallback_action: int = -1) -> torch.Tensor:
    """Ensure each mask row has at least one valid action.

    Args:
        mask: Bool tensor [..., A]. True means action is valid.
        fallback_action: index to force valid if a row is all False. -1 means last action.

    Returns:
        mask with no all-False rows.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()

    valid = mask.sum(dim=-1)
    bad = valid == 0
    if bad.any():
        mask = mask.clone()
        mask[..., fallback_action] = True
    return mask


def apply_action_mask_to_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set invalid actions' logits to -inf (so Categorical(logits=...) ignores them)."""
    neg_inf = torch.finfo(logits.dtype).min
    return torch.where(mask, logits, torch.tensor(neg_inf, device=logits.device, dtype=logits.dtype))


def huber(u: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """Huber loss elementwise (paper Eq.10)."""
    abs_u = u.abs()
    return torch.where(abs_u < kappa, 0.5 * u * u, kappa * (abs_u - 0.5 * kappa))


def quantile_huber_loss_per_sample(
    td_error: torch.Tensor,  # [B, N]
    taus: torch.Tensor,      # [N]
    kappa: float = 1.0,
) -> torch.Tensor:
    """Quantile Huber loss per sample (paper Eq.11).

    Args:
        td_error: [B, N] tensor of (q(s,a) - y).
        taus: [N] quantile levels in (0,1], e.g. n/N.
        kappa: Huber threshold.

    Returns:
        loss_per_sample: [B]
    """
    taus_ = taus.view(1, -1)  # [1, N]
    indicator = (td_error < 0).to(td_error.dtype)  # [B, N]
    rho = (taus_ - indicator).abs() * huber(td_error, kappa=kappa)  # [B, N]
    return rho.mean(dim=-1)  # [B]


def gather_branch(x: torch.Tensor, rbg_index: torch.Tensor) -> torch.Tensor:
    """Gather branch dimension (NRBG) using per-sample indices.

    Args:
        x: Tensor with shape [B, NRBG, ...]
        rbg_index: [B] int64 in [0..NRBG-1]

    Returns:
        x_m: [B, ...] gathered along dim=1.
    """
    if rbg_index.dtype != torch.long:
        rbg_index = rbg_index.long()
    # Expand indices to match x dims after dim=1.
    # For example, if x is [B, M, A], we want indices [B, 1, A]
    # If x is [B, M, A, N], indices [B, 1, A, N]
    idx = rbg_index.view(-1, 1)
    while idx.ndim < x.ndim:
        idx = idx.unsqueeze(-1)
    idx = idx.expand(-1, 1, *x.shape[2:])
    return x.gather(1, idx).squeeze(1)


# -----------------------------
# Networks (paper-shaped)
# -----------------------------

class MultiBranchActor(nn.Module):
    """Actor producing logits for ALL RBG branches in one forward pass.

    forward_all(obs) -> logits_all: [B, NRBG, A]
    """

    def __init__(self, obs_dim: int, n_rbg: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.n_rbg = int(n_rbg)
        self.act_dim = int(act_dim)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.n_rbg * self.act_dim),
        )

    def forward_all(self, obs: torch.Tensor) -> torch.Tensor:
        """Return unmasked logits for all branches: [B, NRBG, A]."""
        out = self.net(obs)  # [B, NRBG*A]
        return out.view(obs.shape[0], self.n_rbg, self.act_dim)

    def forward_branch(self, obs: torch.Tensor, rbg_index: torch.Tensor, action_mask: torch.Tensor,
                       fallback_action: int = -1) -> torch.Tensor:
        """Return masked logits for a single branch per sample: [B, A]."""
        logits_all = self.forward_all(obs)                    # [B, NRBG, A]
        logits_m = gather_branch(logits_all, rbg_index)       # [B, A]
        mask = ensure_nonempty_mask(action_mask, fallback_action=fallback_action)
        return apply_action_mask_to_logits(logits_m, mask)


class MultiBranchQuantileCritic(nn.Module):
    """Distributional critic producing N quantiles per action for ALL branches.

    forward_all(obs) -> q_all: [B, NRBG, A, N]
    """

    def __init__(self, obs_dim: int, n_rbg: int, act_dim: int, n_quantiles: int = 16, hidden: int = 256):
        super().__init__()
        self.n_rbg = int(n_rbg)
        self.act_dim = int(act_dim)
        self.nq = int(n_quantiles)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.n_rbg * self.act_dim * self.nq),
        )

    def forward_all(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.net(obs)  # [B, NRBG*A*N]
        return out.view(obs.shape[0], self.n_rbg, self.act_dim, self.nq)

    def forward_branch(self, obs: torch.Tensor, rbg_index: torch.Tensor) -> torch.Tensor:
        q_all = self.forward_all(obs)              # [B, NRBG, A, N]
        return gather_branch(q_all, rbg_index)     # [B, A, N]


# -----------------------------
# Updater
# -----------------------------

@dataclass
class DSACDHyperParams:
    # Model / distributional
    n_quantiles: int = 16

    # Entropy target
    beta: float = 0.98

    # Target smoothing
    tau: float = 0.001

    # Discount (paper uses gamma=0 for DSACD reward design)
    gamma: float = 0.0

    # Optimizers
    lr_actor: float = 1e-4
    lr_critic: float = 1e-4
    lr_alpha: float = 1e-4

    # Quantile huber
    kappa: float = 1.0

    # PER + numeric stability
    priority_eps: float = 1e-6
    min_prob: float = 1e-8

    # Mask safety
    fallback_action: int = -1


class DSACDUpdater:
    """One gradient update step for DSACD (paper-faithful multi-branch forward, per-branch replay).

    This implements Algorithm 2′ in a shape-safe way while preserving the paper's
    'single forward pass for all RBGs' design.
    """

    def __init__(
        self,
        actor: MultiBranchActor,
        q1: MultiBranchQuantileCritic,
        q2: MultiBranchQuantileCritic,
        q1_target: MultiBranchQuantileCritic,
        q2_target: MultiBranchQuantileCritic,
        n_rbg: int,
        act_dim: int,
        hp: DSACDHyperParams = DSACDHyperParams(),
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.hp = hp
        self.n_rbg = int(n_rbg)
        self.act_dim = int(act_dim)

        self.actor = actor.to(self.device)
        self.q1 = q1.to(self.device)
        self.q2 = q2.to(self.device)
        self.q1_t = q1_target.to(self.device)
        self.q2_t = q2_target.to(self.device)

        # log_alpha parameterization ensures alpha > 0.
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=hp.lr_actor)
        self.opt_critic = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()),
                                           lr=hp.lr_critic)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=hp.lr_alpha)

        # Quantile levels tau_n = n/N.
        self.taus = (torch.arange(1, hp.n_quantiles + 1, device=self.device, dtype=torch.float32)
                     / hp.n_quantiles)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def soft_update_targets(self) -> None:
        """Polyak update for target critics."""
        tau = self.hp.tau
        for p, pt in zip(self.q1.parameters(), self.q1_t.parameters()):
            pt.data.mul_(1 - tau).add_(tau * p.data)
        for p, pt in zip(self.q2.parameters(), self.q2_t.parameters()):
            pt.data.mul_(1 - tau).add_(tau * p.data)

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()}

    def update(
        self,
        batch: Dict[str, torch.Tensor],
        isw: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Perform one DSACD update step.

        Required batch keys:
          observation, next_observation: [B, obs_dim]
          rbg_index: [B] (0..NRBG-1)
          action: [B] (0..A-1)
          reward: [B]
          action_mask, next_action_mask: [B, A] bool

        Optional:
          isw: PER importance-sampling weights [B]
        """
        batch = self._to_device(batch)
        s = batch["observation"]
        sp = batch["next_observation"]
        rbg = batch["rbg_index"].long()
        a = batch["action"].long()
        r = batch["reward"].float()
        mask = batch["action_mask"].bool()
        maskp = batch["next_action_mask"].bool()

        B = s.shape[0]
        if isw is None:
            isw = torch.ones(B, device=self.device, dtype=torch.float32)
        else:
            isw = isw.to(self.device).float()

        # -------------------------
        # (1) Critic target (Eq.13)
        # -------------------------
        with torch.no_grad():
            # Policy at s' for the selected branch (per-sample m)
            logits_p = self.actor.forward_branch(sp, rbg, maskp, fallback_action=self.hp.fallback_action)  # [B, A]
            dist_p = torch.distributions.Categorical(logits=logits_p)
            ap = dist_p.sample()                                         # [B]
            logp_ap = dist_p.log_prob(ap)                                # [B]

            # Target critics for branch m, mean over quantiles then min across critics
            q1p_all = self.q1_t.forward_branch(sp, rbg)                  # [B, A, N]
            q2p_all = self.q2_t.forward_branch(sp, rbg)                  # [B, A, N]
            qminp_mean = torch.min(q1p_all.mean(-1), q2p_all.mean(-1))    # [B, A]
            q_ap = qminp_mean.gather(1, ap.view(-1, 1)).squeeze(1)        # [B]

            y = r + self.hp.gamma * (q_ap - self.alpha.detach() * logp_ap)  # [B]

        # --------------------------------------------
        # (2) Critic update (distributional, Eq.10-11)
        # --------------------------------------------
        q1_all = self.q1.forward_branch(s, rbg)     # [B, A, N]
        q2_all = self.q2.forward_branch(s, rbg)     # [B, A, N]

        # Gather chosen action quantiles: [B, N]
        a_idx = a.view(-1, 1, 1).expand(-1, 1, self.hp.n_quantiles)
        q1_a = q1_all.gather(1, a_idx).squeeze(1)
        q2_a = q2_all.gather(1, a_idx).squeeze(1)

        td1 = q1_a - y.view(-1, 1)  # [B, N]
        td2 = q2_a - y.view(-1, 1)

        loss_q1_ps = quantile_huber_loss_per_sample(td1, self.taus, kappa=self.hp.kappa)  # [B]
        loss_q2_ps = quantile_huber_loss_per_sample(td2, self.taus, kappa=self.hp.kappa)  # [B]
        loss_q = (isw * (loss_q1_ps + loss_q2_ps)).mean()

        self.opt_critic.zero_grad(set_to_none=True)
        loss_q.backward()
        self.opt_critic.step()

        # ---------------------------
        # (3) Policy update (Eq.15)
        # ---------------------------
        logits = self.actor.forward_branch(s, rbg, mask, fallback_action=self.hp.fallback_action)  # [B, A]
        dist = torch.distributions.Categorical(logits=logits)
        probs = dist.probs  # [B, A]
        log_probs = torch.log(probs.clamp_min(self.hp.min_prob))  # [B, A]

        with torch.no_grad():
            q1m = self.q1.forward_branch(s, rbg).mean(-1)  # [B, A]
            q2m = self.q2.forward_branch(s, rbg).mean(-1)
            qmin = torch.min(q1m, q2m)                     # [B, A]

        # Per-sample (per-branch) policy loss: sum over actions
        pi_obj_ps = (probs * (self.alpha.detach() * log_probs - qmin)).sum(dim=-1)  # [B]
        loss_pi = (isw * pi_obj_ps).mean()

        self.opt_actor.zero_grad(set_to_none=True)
        loss_pi.backward()
        self.opt_actor.step()

        # --------------------------------------------
        # (4) Alpha update with state-dependent target
        # --------------------------------------------
        # |A(s,m)| from mask: number of valid actions for this branch
        A_size = ensure_nonempty_mask(mask, fallback_action=self.hp.fallback_action).sum(dim=-1).float().clamp(min=1.0)
        H_target = -self.hp.beta * torch.log(1.0 / A_size)  # [B]  (paper Eq.17)

        entropy = -(probs * log_probs).sum(dim=-1)          # [B]
        # We want entropy ≈ H_target. A common discrete form:
        #   loss_alpha = E[ alpha * (H_target - entropy) ]  (detach entropy and H_target)
        loss_alpha = (isw * (self.alpha * (H_target.detach() - entropy.detach()))).mean()

        self.opt_alpha.zero_grad(set_to_none=True)
        loss_alpha.backward()
        self.opt_alpha.step()

        # -------------------------
        # (5) Target soft update
        # -------------------------
        self.soft_update_targets()

        # -------------------------
        # (6) PER priority (Eq.19)
        # -------------------------
        with torch.no_grad():
            prio = (td1.abs().mean(dim=-1) + td2.abs().mean(dim=-1)) * 0.5 + self.hp.priority_eps  # [B]

        return {
            "loss_q": loss_q.detach(),
            "loss_pi": loss_pi.detach(),
            "loss_alpha": loss_alpha.detach(),
            "alpha": self.alpha.detach(),
            "priority": prio.detach(),
        }
