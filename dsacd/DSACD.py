"""DSACD (Distributional Soft Actor-Critic Discrete) - per-branch update core.

This module is written to closely mirror Algorithm 2 (DSACD) from:
"From Simulation to Practice: Generalizable Deep Reinforcement Learning for Cellular Schedulers".

Implementation choices for clarity + debuggability:
- Replay buffer stores ONE transition per branch (one (layer, RBG) decision).
- Therefore, each sampled batch item corresponds to exactly ONE masked categorical action.

Expected batch tensor shapes (per-branch layout):
  observation        : [B, obs_dim]
  action             : [B]              (int64 in [0..A-1])
  reward             : [B]
  next_observation   : [B, obs_dim]
  action_mask        : [B, A]           (bool, True=valid)
  next_action_mask   : [B, A]           (bool, True=valid)

Critic outputs (distributional):
  q_quantiles(s)     : [B, A, N]

Notes:
- Mask handling: invalid actions get logit=-inf. If a mask row has no valid actions,
  we force the last action (typically "no allocation") to be valid.
- PER importance-sampling weights (isw): we apply them PER-SAMPLE, not as a single mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


# -----------------------------
# Masking + quantile loss utils
# -----------------------------

def ensure_nonempty_mask(mask: torch.Tensor, fallback_action: int = -1) -> torch.Tensor:
    """Ensure each row has at least one valid action.

    Args:
        mask: Bool tensor [..., A]. True means action is valid.
        fallback_action: Which action index to force valid if a row is all False.
                         -1 means the last action.

    Returns:
        A (possibly) modified mask with no all-False rows.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()

    # mask shape: [..., A]
    valid_count = mask.sum(dim=-1)
    bad = valid_count == 0
    if bad.any():
        mask = mask.clone()
        mask[..., fallback_action] = torch.where(
            bad, torch.ones_like(bad, dtype=torch.bool), mask[..., fallback_action]
        )
    return mask


def apply_action_mask_to_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set invalid actions' logits to -inf so Categorical(logits=...) ignores them."""
    neg_inf = torch.finfo(logits.dtype).min
    return torch.where(mask, logits, torch.tensor(neg_inf, device=logits.device, dtype=logits.dtype))


def huber(u: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """Huber loss elementwise (paper Eq.10)."""
    abs_u = u.abs()
    return torch.where(abs_u < kappa, 0.5 * u * u, kappa * (abs_u - 0.5 * kappa))


def quantile_huber_loss_per_sample(
    td_error: torch.Tensor,
    taus: torch.Tensor,
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
    # Expand taus to [1, N] for broadcasting.
    taus_ = taus.view(1, -1)

    # Indicator 1{td_error < 0}
    indicator = (td_error < 0).to(td_error.dtype)

    # rho^k_tau(u) = |tau - 1{u<0}| * Huber(u)
    rho = (taus_ - indicator).abs() * huber(td_error, kappa=kappa)

    # Average over quantiles -> [B]
    return rho.mean(dim=-1)


# -----------------------------
# Networks
# -----------------------------

class Actor(nn.Module):
    """Simple discrete policy network producing logits over A actions."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        # obs: [B, obs_dim], mask: [B, A]
        logits = self.net(obs)
        action_mask = ensure_nonempty_mask(action_mask)
        return apply_action_mask_to_logits(logits, action_mask)


class QuantileCritic(nn.Module):
    """Distributional critic producing N quantiles per action: [B, A, N]."""

    def __init__(self, obs_dim: int, act_dim: int, n_quantiles: int = 16, hidden: int = 256):
        super().__init__()
        self.act_dim = act_dim
        self.nq = n_quantiles
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim * n_quantiles),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.net(obs)
        return out.view(obs.shape[0], self.act_dim, self.nq)


# -----------------------------
# Updater
# -----------------------------

@dataclass
class DSACDHyperParams:
    n_quantiles: int = 16
    beta: float = 0.98          # entropy target scale (paper tunes this)
    tau: float = 0.001          # Polyak target update
    gamma: float = 0.0          # paper sets gamma=0 for DSACD reward design
    lr_actor: float = 1e-4
    lr_critic: float = 1e-4
    lr_alpha: float = 1e-4
    kappa: float = 1.0          # Huber threshold
    priority_eps: float = 1e-6
    min_prob: float = 1e-8
    fallback_action: int = -1   # force last action valid if mask row is empty


class DSACDUpdater:
    """One gradient update step for DSACD (per-branch replay buffer layout).

    This implements the rewritten Algorithm 2′ in a shape-safe way.
    """

    def __init__(
        self,
        actor: Actor,
        q1: QuantileCritic,
        q2: QuantileCritic,
        q1_target: QuantileCritic,
        q2_target: QuantileCritic,
        act_dim: int,
        hp: DSACDHyperParams = DSACDHyperParams(),
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.hp = hp
        self.act_dim = act_dim

        self.actor = actor.to(self.device)
        self.q1 = q1.to(self.device)
        self.q2 = q2.to(self.device)
        self.q1_t = q1_target.to(self.device)
        self.q2_t = q2_target.to(self.device)

        # log_alpha parameterization ensures alpha > 0.
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=hp.lr_actor)
        self.opt_critic = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=hp.lr_critic
        )
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=hp.lr_alpha)

        # Quantile levels tau_n = n/N.
        self.taus = torch.arange(1, hp.n_quantiles + 1, device=self.device, dtype=torch.float32) / hp.n_quantiles

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def soft_update_targets(self) -> None:
        """Polyak averaging: target <- tau * online + (1-tau) * target."""
        tau = self.hp.tau
        for p, pt in zip(self.q1.parameters(), self.q1_t.parameters()):
            pt.data.mul_(1.0 - tau).add_(tau * p.data)
        for p, pt in zip(self.q2.parameters(), self.q2_t.parameters()):
            pt.data.mul_(1.0 - tau).add_(tau * p.data)

    def _categorical_from_logits(self, logits: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=logits)

    def update(
        self,
        batch: Dict[str, torch.Tensor],
        isw: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Perform one DSACD update.

        Args:
            batch: dict with keys described at top of file.
            isw: importance-sampling weights [B]. If None, uses ones.

        Returns:
            metrics dict including per-sample priorities for PER.
        """
        # -----------------
        # Move tensors
        # -----------------
        s = batch["observation"].to(self.device)
        a = batch["action"].to(self.device).long()
        r = batch["reward"].to(self.device).float()
        sp = batch["next_observation"].to(self.device)

        mask = ensure_nonempty_mask(batch["action_mask"].to(self.device).bool(), fallback_action=self.hp.fallback_action)
        maskp = ensure_nonempty_mask(batch["next_action_mask"].to(self.device).bool(), fallback_action=self.hp.fallback_action)

        B = r.shape[0]
        if isw is None:
            isw = torch.ones(B, device=self.device, dtype=torch.float32)
        else:
            isw = isw.to(self.device).float().view(B)

        # -----------------
        # 1) Target y_i (paper Eq.13) using sampled a' ~ pi(s')
        # -----------------
        with torch.no_grad():
            logits_p = self.actor(sp, maskp)                  # [B, A]
            dist_p = self._categorical_from_logits(logits_p)  # masked categorical
            ap = dist_p.sample()                              # [B]
            logp_ap = dist_p.log_prob(ap)                     # [B]

            # Target critics give quantiles: [B, A, N]
            q1p_q = self.q1_t(sp)
            q2p_q = self.q2_t(sp)

            # Mean Q per action: [B, A]
            q1p_mean = q1p_q.mean(dim=-1)
            q2p_mean = q2p_q.mean(dim=-1)
            qminp_mean = torch.min(q1p_mean, q2p_mean)

            # Q(s', a') from min critics: [B]
            q_ap = qminp_mean.gather(1, ap.view(-1, 1)).squeeze(1)

            y = r + self.hp.gamma * (q_ap - self.alpha.detach() * logp_ap)  # [B]

        # -----------------
        # 2) Critic update (paper Eq.10-12)
        # -----------------
        q1_q = self.q1(s)  # [B, A, N]
        q2_q = self.q2(s)

        # Gather chosen action quantiles: [B, N]
        a_idx = a.view(-1, 1, 1).expand(-1, 1, self.hp.n_quantiles)
        q1_a = q1_q.gather(1, a_idx).squeeze(1)
        q2_a = q2_q.gather(1, a_idx).squeeze(1)

        td1 = q1_a - y.view(-1, 1)  # [B, N]
        td2 = q2_a - y.view(-1, 1)

        # Per-sample critic losses: [B]
        loss_q1_ps = quantile_huber_loss_per_sample(td1, self.taus, kappa=self.hp.kappa)
        loss_q2_ps = quantile_huber_loss_per_sample(td2, self.taus, kappa=self.hp.kappa)

        # Weighted mean across batch
        loss_q = (isw * (loss_q1_ps + loss_q2_ps)).mean()

        self.opt_critic.zero_grad(set_to_none=True)
        loss_q.backward()
        self.opt_critic.step()

        # -----------------
        # 3) Policy update (paper Eq.15)
        #    J_pi(s) = sum_a pi(a|s) [ alpha log pi(a|s) - Q_min(s,a) ]
        # -----------------
        logits = self.actor(s, mask)                # [B, A]
        dist = self._categorical_from_logits(logits)
        probs = dist.probs                          # [B, A]
        log_probs = torch.log(probs.clamp_min(self.hp.min_prob))

        with torch.no_grad():
            # Online critics mean Q per action: [B, A]
            q1_mean = self.q1(s).mean(dim=-1)
            q2_mean = self.q2(s).mean(dim=-1)
            qmin = torch.min(q1_mean, q2_mean)

        # Per-sample policy objective: [B]
        j_pi_ps = (probs * (self.alpha.detach() * log_probs - qmin)).sum(dim=-1)
        loss_pi = (isw * j_pi_ps).mean()

        self.opt_actor.zero_grad(set_to_none=True)
        loss_pi.backward()
        self.opt_actor.step()

        # -----------------
        # 4) Alpha update with state-dependent target entropy (paper Eq.16-18)
        #    H_target(s) = -beta * log(1/|A(s)|)
        # -----------------
        with torch.no_grad():
            valid_count = mask.sum(dim=-1).clamp(min=1).float()  # [B]
            H_target = -self.hp.beta * torch.log(1.0 / valid_count)  # [B]

            # Expectation over actions: E_pi[log pi(a|s) + H_target(s)]
            # This is per sample. Detach policy terms for alpha-only update.
            alpha_term_ps = (probs.detach() * (log_probs.detach() + H_target.view(-1, 1))).sum(dim=-1)  # [B]

        loss_alpha = (isw * (self.alpha * alpha_term_ps)).mean()

        self.opt_alpha.zero_grad(set_to_none=True)
        loss_alpha.backward()
        self.opt_alpha.step()

        # -----------------
        # 5) Target network update
        # -----------------
        self.soft_update_targets()

        # -----------------
        # 6) PER priority update value (paper Eq.19)
        # -----------------
        with torch.no_grad():
            prio = (td1.abs().mean(dim=-1) + td2.abs().mean(dim=-1)) / 2.0
            prio = prio + self.hp.priority_eps

        return {
            "loss_q": loss_q.detach(),
            "loss_pi": loss_pi.detach(),
            "loss_alpha": loss_alpha.detach(),
            "alpha": self.alpha.detach(),
            "priority": prio.detach(),          # [B] per-sample priorities
        }
