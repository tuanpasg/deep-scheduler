import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class LayerCtx:
    obs: torch.Tensor            # [obs_dim]
    masks_rbg: torch.Tensor      # [n_rbg, act_dim] bool
    layer_id: int


class SurrogateEnvAdapter:
    """
    Surrogate environment implementing EnvAdapterTemplate (Option B)
    compatible with train_dsacd_multibranch_optionB.py
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_layers: int,
        n_rbg: int,
        fallback_action: int = 0,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_layers = n_layers
        self.n_rbg = n_rbg
        self.fallback_action = fallback_action
        self.device = torch.device(device)

        self.rng = np.random.default_rng(seed)

        # Hidden action quality (learnable signal)
        self.action_quality = torch.randn(act_dim, device=self.device)

        # Internal buffers
        self._layer_obs = []
        self._layer_masks = []
        self._layer_actions = []
        self._transitions = []

        # For contention penalty
        self._layer_action_counts = None

    # ------------------------------------------------------------------
    # EnvAdapterTemplate API
    # ------------------------------------------------------------------

    def reset(self):
        self._layer_obs.clear()
        self._layer_masks.clear()
        self._layer_actions.clear()
        self._transitions.clear()

        obs0 = self._sample_obs()
        self._layer_obs.append(obs0)

        return obs0

    def begin_tti(self):
        self._layer_action_counts = torch.zeros(
            self.act_dim, device=self.device
        )

    def layer_iter(self):
        """
        Yield LayerCtx for each layer
        """
        for l in range(self.n_layers):
            obs = self._layer_obs[l]
            masks = self._sample_masks()
            self._layer_masks.append(masks)

            yield LayerCtx(
                obs=obs,
                masks_rbg=masks,
                layer_id=l,
            )

    def apply_layer_actions(self, layer_ctx: LayerCtx, actions_rbg: torch.Tensor):
        """
        actions_rbg: [n_rbg] int64
        """
        assert actions_rbg.shape == (self.n_rbg,)
        self._layer_actions.append(actions_rbg.clone())

        # Count actions for contention penalty
        for a in actions_rbg.tolist():
            if a != self.fallback_action:
                self._layer_action_counts[a] += 1

        # Generate next observation (simple dynamics)
        if layer_ctx.layer_id < self.n_layers - 1:
            next_obs = self._transition_obs(layer_ctx.obs)
            self._layer_obs.append(next_obs)

    def end_tti(self):
        """
        Build per-branch transitions
        """
        for l in range(self.n_layers):
            obs = self._layer_obs[l]
            masks = self._layer_masks[l]
            actions = self._layer_actions[l]

            if l < self.n_layers - 1:
                next_obs = self._layer_obs[l + 1]
                next_masks = self._layer_masks[l + 1]
            else:
                next_obs = obs
                next_masks = masks

            for rbg in range(self.n_rbg):
                a = actions[rbg].item()

                # Reward = quality - contention penalty + noise
                base = self.action_quality[a]
                penalty = (
                    0.1 * (self._layer_action_counts[a] - 1)
                    if a != self.fallback_action
                    else 0.0
                )
                noise = 0.01 * torch.randn((), device=self.device)

                reward = base - penalty + noise

                self._transitions.append(
                    {
                        "observation": obs.detach(),
                        "action": torch.tensor(a, device=self.device),
                        "reward": reward.detach(),
                        "next_observation": next_obs.detach(),
                        "action_mask": masks[rbg].detach(),
                        "next_action_mask": next_masks[rbg].detach(),
                        "rbg_index": torch.tensor(rbg, device=self.device),
                    }
                )

    def export_branch_transitions(self):
        """
        Return list of per-branch transitions
        """
        return self._transitions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_obs(self):
        return torch.randn(self.obs_dim, device=self.device)

    def _transition_obs(self, obs):
        return obs + 0.05 * torch.randn_like(obs)

    def _sample_masks(self):
        """
        Random action masks per RBG.
        Fallback action is always valid.
        """
        masks = torch.zeros(
            self.n_rbg, self.act_dim, dtype=torch.bool, device=self.device
        )
        for rbg in range(self.n_rbg):
            valid = self.rng.choice(
                self.act_dim,
                size=max(2, self.act_dim // 4),
                replace=False,
            )
            masks[rbg, valid] = True
            masks[rbg, self.fallback_action] = True
        return masks
