# toy5g_env_adapter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, List, Dict, Optional
import math
import torch


@dataclass
class LayerContext:
    layer: int
    obs: torch.Tensor          # [obs_dim]
    masks_rbg: torch.Tensor    # [NRBG, A] bool


class DeterministicToy5GEnvAdapter:
    """
    Deterministic toy env with 1LDS structure:
      - Each TTI:
          for l in 1..L:
              agent selects per-RBG UE index (or NOOP) for that layer
              reward r_{m,l} computed after finishing the whole layer l (across all RBGs)

    Action space (discrete): 0..(n_ue-1) are UE indices, last index is NOOP.
      act_dim = n_ue + 1

    Observations: flattened deterministic features, padded/truncated to obs_dim.
    Masks: valid UE if buffer>0, NOOP always valid.

    Reward matches paper:
      r_{m,l} = P*v_m for l==1 else k*v_m
      P = G/Gmax, G = geometric mean of per-UE instantaneous throughput at current TTI
      v_m = +1 if chosen PF is best among valid UEs/NOOP else -1
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_layers: int,
        n_rbg: int,
        *,
        seed: int = 0,
        device: str = "cpu",
        # reward params
        k: float = 0.2,
        gmax: float = 1.0,
        # PF/throughput params
        ema_beta: float = 0.98,
        eps: float = 1e-6,
        # deterministic traffic/channel knobs
        base_rate: float = 1.0,
        buf_init: int = 50,
        buf_arrival: int = 10,
    ):
        assert act_dim >= 2, "Need at least 1 UE + NOOP."
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_ue = act_dim - 1
        self.noop = self.n_ue
        self.n_layers = n_layers
        self.n_rbg = n_rbg
        self.device = torch.device(device)

        self.k = float(k)
        self.gmax = float(gmax)
        self.ema_beta = float(ema_beta)
        self.eps = float(eps)
        self.base_rate = float(base_rate)
        self.buf_init = int(buf_init)
        self.buf_arrival = int(buf_arrival)

        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        self._gen = g

        # Deterministic channel quality matrix (UE x RBG), fixed for whole run
        # Keep it deterministic but not uniform.
        # Convert to "SNR-like" positive values.
        u = torch.arange(self.n_ue).float().unsqueeze(1)
        m = torch.arange(self.n_rbg).float().unsqueeze(0)
        self.snr = (1.0 + 0.2 * torch.sin(0.7 * u + 0.3 * m) + 0.1 * torch.cos(0.2 * u - 0.5 * m)).clamp_min(0.05)
        self.snr = self.snr.to(self.device)  # [U, M]

        # State
        self.t = 0
        self.buf = torch.full((self.n_ue,), self.buf_init, device=self.device, dtype=torch.float32)  # "packets"
        self.avg_tp = torch.full((self.n_ue,), 1.0, device=self.device, dtype=torch.float32)          # EMA throughput

        # Allocation caches (per TTI)
        self._alloc = torch.full((self.n_layers, self.n_rbg), self.noop, device=self.device, dtype=torch.long)
        self._last_transitions: List[Dict] = []
        self._cur_layer: Optional[int] = None

    # ---------- Public API ----------
    def reset(self):
        self.t = 0
        self.buf.fill_(self.buf_init)
        self.avg_tp.fill_(1.0)
        self._alloc.fill_(self.noop)
        self._last_transitions = []
        self._cur_layer = None

    def begin_tti(self):
        self._alloc.fill_(self.noop)
        self._last_transitions = []
        self._cur_layer = None

    def layer_iter(self) -> Iterator[LayerContext]:
        # We yield layer contexts one by one; obs is global state + layer id encoding.
        for l in range(self.n_layers):
            self._cur_layer = l
            masks = self._build_masks()  # [M, A]
            obs = self._build_obs(layer=l)  # [obs_dim]
            yield LayerContext(layer=l, obs=obs, masks_rbg=masks)

    def apply_layer_actions(self, layer_ctx: LayerContext, actions_rbg: torch.Tensor):
        """
        actions_rbg: [M] int64 on CPU or GPU. Values in [0..A-1]
        """
        l = layer_ctx.layer
        a = actions_rbg.to(self.device).long().clamp(0, self.act_dim - 1)
        self._alloc[l, :] = a

    def end_tti(self):
        """
        After all layers have been applied, we compute rewards for each layer sequentially,
        storing transitions (s, a, r, s') per (layer, rbg) branch with masks.
        """
        # deterministically add arrivals at start of TTI reward computation
        self.buf = self.buf + self.buf_arrival

        # Pre-compute masks/obs for each layer (s), and next layer (s') as "after reward"
        # We compute reward layer-by-layer (as paper), updating avg_tp each layer using partial throughput.
        # For deterministic toy, we treat each layer contribution as additive throughput on each RBG.
        avg_tp_before_tti = self.avg_tp.clone()

        # For each layer l: compute partial throughput with layers 0..l included
        for l in range(self.n_layers):
            masks = self._build_masks()  # based on current buffers (after arrivals, before serving)
            obs = self._build_obs(layer=l)

            # reward computed after finishing layer l allocation (across all RBGs)
            # 1) compute instantaneous per-UE throughput for *this layer only* and accumulate
            tp_layer = torch.zeros((self.n_ue,), device=self.device)
            for m in range(self.n_rbg):
                ue = int(self._alloc[l, m].item())
                if ue == self.noop:
                    continue
                if self.buf[ue] <= 0:
                    continue  # should be masked but keep safe
                tp_layer[ue] += self._rate(ue, m)

            # 2) update EMA throughput using this layer contribution (paper says reward after each layer iteration)
            self.avg_tp = self.ema_beta * self.avg_tp + (1.0 - self.ema_beta) * tp_layer

            # 3) Off-policy DSACD reward (paper Appendix D.4, Eq.20-21)
            # Use "past average throughput" snapshot from start-of-TTI as R_u.
            # For this toy env, achievable rate does not depend on co-scheduled users,
            # so T_{u,m,l} = rate(u,m) if u is scheduled at (m,l), else 0.
            rewards_m = torch.empty((self.n_rbg,), device=self.device)

            for m in range(self.n_rbg):
                chosen = int(self._alloc[l, m].item())

                # Raw rewards for all UE actions (exclude NOOP) for normalization.
                # NOTE: This is a toy proxy: in a real MU-MIMO simulator, T_{u,m,l}
                # would depend on the co-scheduled set and would be recomputed per candidate.
                raw_all = torch.empty((self.n_ue,), device=self.device, dtype=torch.float32)
                for u in range(self.n_ue):
                    if self.buf[u] <= 0:
                        raw_all[u] = 0.0
                        continue

                    Ru = float((avg_tp_before_tti[u] + self.eps).item())
                    Tu = float(self._rate(u, m).item())

                    if l == 0:
                        raw_all[u] = Tu / Ru
                    else:
                        prev = int(self._alloc[l - 1, m].item())
                        Tu_prev = Tu if (prev == u) else 0.0
                        raw_all[u] = (Tu / Ru) - (Tu_prev / Ru)

                max_raw = float(raw_all.max().item()) if raw_all.numel() > 0 else 0.0

                if max_raw > 0.0:
                    if chosen == self.noop:
                        rewards_m[m] = 0.0
                    else:
                        u = int(chosen)
                        # chosen might be invalid if buf==0; keep safe
                        if u < 0 or u >= self.n_ue or self.buf[u] <= 0:
                            rewards_m[m] = 0.0
                        else:
                            raw = float(raw_all[u].item())
                            rewards_m[m] = max(raw / max_raw, -1.0)
                elif max_raw < 0.0:
                    # Rare in this toy env, but implement paper's special-case anyway.
                    rewards_m[m] = 1.0 if (chosen == self.noop) else -1.0
                else:
                    # All-zero raw reward (e.g., no buffered UEs).
                    rewards_m[m] = 0.0

            # 5) Apply service (drain buffers) for this layer after reward computation
            # Deterministically drain proportional to served rate (cap at available buf)
            for m in range(self.n_rbg):
                ue = int(self._alloc[l, m].item())
                if ue == self.noop:
                    continue
                served = self._rate(ue, m)  # "packets"
                self.buf[ue] = torch.clamp(self.buf[ue] - served, min=0.0)

            # next state after this layer
            next_masks = self._build_masks()
            next_obs = self._build_obs(layer=min(l + 1, self.n_layers - 1))

            # Export transitions per RBG
            for m in range(self.n_rbg):
                self._last_transitions.append({
                    "observation": obs.detach().cpu(),
                    "next_observation": next_obs.detach().cpu(),
                    "rbg_index": torch.tensor(m, dtype=torch.long),
                    "action": torch.tensor(int(self._alloc[l, m].item()), dtype=torch.long),
                    "reward": torch.tensor(float(rewards_m[m].item()), dtype=torch.float32),
                    "action_mask": masks[m].detach().cpu(),           # [A] bool
                    "next_action_mask": next_masks[m].detach().cpu(), # [A] bool
                })

        self.t += 1

    def export_branch_transitions(self) -> List[Dict]:
        return self._last_transitions

    # ---------- Internals ----------
    def _rate(self, ue: int, m: int) -> torch.Tensor:
        # Deterministic "rate": base_rate * log2(1+snr)
        snr = self.snr[ue, m]
        return self.base_rate * torch.log2(1.0 + snr)

    def _build_masks(self) -> torch.Tensor:
        # valid UE if buffer > 0, NOOP always valid
        valid_ue = (self.buf > 0.0)  # [U]
        masks = valid_ue.unsqueeze(0).repeat(self.n_rbg, 1)  # [M, U]
        noop_col = torch.ones((self.n_rbg, 1), device=self.device, dtype=torch.bool)
        return torch.cat([masks, noop_col], dim=1)  # [M, A]

    def _build_obs(self, layer: int) -> torch.Tensor:
        # Build structured features then pad/truncate to obs_dim.
        # Keep deterministic and "5G-ish": UE features + summary of channels.
        # UE features: [avg_tp, buf] per UE
        ue_feat = torch.stack([self.avg_tp, self.buf], dim=1).reshape(-1)  # [2U]
        # RBG summary: mean SNR per RBG across UEs (frequency selectivity hint)
        rbg_snr_mean = self.snr.mean(dim=0)  # [M]
        # Layer one-hot (L)
        layer_oh = torch.zeros((self.n_layers,), device=self.device)
        layer_oh[layer] = 1.0
        core = torch.cat([ue_feat, rbg_snr_mean, layer_oh], dim=0).float()

        if core.numel() >= self.obs_dim:
            return core[: self.obs_dim].clone()
        out = torch.zeros((self.obs_dim,), device=self.device, dtype=torch.float32)
        out[: core.numel()] = core
        return out

    def _geom_mean(self, x: torch.Tensor) -> float:
        # geometric mean exp(mean(log(x)))
        return float(torch.exp(torch.mean(torch.log(x))).item())

    def _pf(self, ue: int, m: int, avg_tp_ref: torch.Tensor) -> float:
        if ue == self.noop:
            return 0.0
        if self.buf[ue] <= 0:
            return -1e9
        rate = float(self._rate(ue, m).item())
        denom = float((avg_tp_ref[ue] + self.eps).item())
        return rate / denom

    def _greedy_indicator_v(self, m: int, chosen: int, avg_tp_ref: torch.Tensor) -> float:
        chosen_pf = self._pf(chosen, m, avg_tp_ref)
        best_pf = chosen_pf
        # greedy search over all valid UEs + NOOP
        for ue in range(self.act_dim):
            pf = self._pf(ue, m, avg_tp_ref)
            if pf > best_pf:
                best_pf = pf
        return -1.0 if (best_pf > chosen_pf + 1e-12) else 1.0
