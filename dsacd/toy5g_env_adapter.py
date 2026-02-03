# toy5g_env_adapter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, List, Dict, Optional, Tuple
import math
import torch
import numpy as np


# ----------------------
# 38.214-inspired helpers
# ----------------------
MCS_TABLE: List[Tuple[int, int]] = [
    (2, 120), (2, 157), (2, 193), (2, 251), (2, 308), (2, 379),
    (4, 449), (4, 526), (4, 602), (4, 679), (6, 340), (6, 378),
    (6, 434), (6, 490), (6, 553), (6, 616), (6, 658), (8, 438),
    (8, 466), (8, 517), (8, 567), (8, 616), (8, 666), (8, 719),
    (8, 772), (8, 822), (8, 873), (8, 910), (8, 948),
]


def tbs_38214_bytes(mcs_idx, n_prb, n_symb=14, n_layers=1, overhead_re_per_prb=18):
    """Very light TBS proxy (bytes)."""
    if n_prb <= 0 or mcs_idx < 0:
        return 0
    mcs_idx = int(max(0, min(28, mcs_idx)))
    Qm, R1024 = MCS_TABLE[mcs_idx]
    R = R1024 / 1024.0
    N_re_per_prb = max(0, 12 * n_symb - overhead_re_per_prb)
    Ninfo = R * Qm * N_re_per_prb * int(n_prb) * n_layers
    if Ninfo <= 0:
        return 0
    if Ninfo <= 3824:
        TBS_bits = int(6 * math.ceil(Ninfo / 6.0))
    else:
        TBS_bits = int(6 * math.ceil((Ninfo - 24) / 6.0))
    TBS_bits = max(TBS_bits, 24)
    return TBS_bits // 8


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
        # PF/throughput params
        ema_beta: float = 0.98,
        eps: float = 1e-6,
        # deterministic traffic/channel knobs
        base_rate: float = 1.0,
        buf_init: int = 50_000,
        buf_arrival: int = 10_000,
        tti_ms: float = 1.0,
        prbs_per_rbg: int = 18,
        n_symb: int = 14,
        overhead_re_per_prb: int = 18,
    ):
        assert act_dim >= 2, "Need at least 1 UE + NOOP."
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_ue = act_dim - 1
        self.noop = self.n_ue
        self.n_layers = n_layers
        self.n_rbg = n_rbg
        self.device = torch.device(device)

        self.max_mcs = 28  # max MCS index (0..28)
        self.max_ue_rank = 2
        self.ema_beta = float(ema_beta)
        self.eps = float(eps)
        self.base_rate = float(base_rate)
        self.buf_init = int(buf_init)
        self.buf_arrival = int(buf_arrival)
        self.tti_ms = float(tti_ms)
        self.prbs_per_rbg = int(prbs_per_rbg)
        self.n_symb = int(n_symb)
        self.overhead = int(overhead_re_per_prb)
        self.max_rate = 1100 # [Mbps] 4 Layers x 273 RB x QAM64 x 1 TTI [1ms]

        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        self._gen = g

        # State
        self.t = 0
        self.buf = torch.full((self.n_ue,), self.buf_init, device=self.device, dtype=torch.float32)  # bytes
        self.avg_tp = torch.full((self.n_ue,), 1.0, device=self.device, dtype=torch.float32)          # EMA throughput (Mbps)

        # Allocation caches (per TTI)
        self._alloc = torch.full((self.n_layers, self.n_rbg), self.noop, device=self.device, dtype=torch.long)
        self._last_transitions: List[Dict] = []
        self._cur_layer: Optional[int] = None
        self._arrivals_added = False
        self._avg_tp_before_tti: Optional[torch.Tensor] = None

        # Per-UE rank capability (spatial layers supported), max rank = 2
        self.ue_rank = (1 + (torch.arange(self.n_ue) % 2)).to(self.device).float()

        # Fading profile: fixed per-UE MCS mean (constant through layers/RBGs)
        self.rng = np.random.default_rng(seed)
        self.mcs_mean = self.rng.integers(0, self.max_mcs + 1, size=(self.n_ue,))
        self.mcs_spread = 0
        self._sample_mcs()

    # ---------- Public API ----------
    def reset(self):
        self.t = 0
        self.buf.fill_(self.buf_init)
        self.avg_tp.fill_(1.0)
        self._alloc.fill_(self.noop)
        self._last_transitions = []
        self._cur_layer = None
        self._arrivals_added = False
        self._avg_tp_before_tti = None

    def begin_tti(self):
        self._alloc.fill_(self.noop)
        self._last_transitions = []
        self._cur_layer = None
        self._arrivals_added = False
        self._avg_tp_before_tti = None

    def layer_iter(self) -> Iterator[LayerContext]:
        # We yield layer contexts one by one; obs is global state + layer id encoding.
        for l in range(self.n_layers):
            self._cur_layer = l
            masks = self._build_masks(layer=l)  # [M, A]
            obs = self._build_obs(layer=l)  # [obs_dim]
            yield LayerContext(layer=l, obs=obs, masks_rbg=masks)

    def apply_layer_actions(self, layer_ctx: LayerContext, actions_rbg: torch.Tensor):
        """
        actions_rbg: [M] int64 on CPU or GPU. Values in [0..A-1]
        """
        l = layer_ctx.layer
        a = actions_rbg.to(self.device).long().clamp(0, self.act_dim - 1)
        self._alloc[l, :] = a

    def export_branch_transitions(self) -> List[Dict]:
        return self._last_transitions

    def dump_state(self) -> Dict:
        """Return a dict of the current environment state (buffers, rates, allocs)."""
        return {
            "t": self.t,
            "buf": self.buf.detach().cpu().tolist(),
            "avg_tp": self.avg_tp.detach().cpu().tolist(),
            "alloc": self._alloc.detach().cpu().tolist(),
            "curr_mcs": self._curr_mcs.tolist() if hasattr(self, "_curr_mcs") else None,
            "ue_rank": self.ue_rank.detach().cpu().tolist(),
            "mcs_mean": self.mcs_mean.tolist() if hasattr(self, "mcs_mean") else None,
            "cur_layer": self._cur_layer,
        }

    def compute_layer_transitions(self, layer_ctx: LayerContext) -> List[Dict]:
        """Compute rewards and package transitions for a single layer using cached obs/masks."""
        self._ensure_tti_start()
        return self._compute_layer_transitions(
            layer=layer_ctx.layer,
            obs=layer_ctx.obs,
            masks=layer_ctx.masks_rbg,
        )

    def finish_tti(self):
        self.t += 1

    # ---------- Internals ----------
    def _served_bytes(self, ue: int) -> torch.Tensor:
        if not hasattr(self, "_curr_mcs"):
            self._sample_mcs()
        mcs = int(self._curr_mcs[ue])
        tbs = tbs_38214_bytes(mcs, self.prbs_per_rbg, n_symb=self.n_symb, overhead_re_per_prb=self.overhead)
        return torch.tensor(float(tbs), device=self.device)

    def _rate_mbps(self, ue: int) -> torch.Tensor:
        served = self._served_bytes(ue)
        duration_s = self.tti_ms / 1000.0
        return (served * 8.0) / 1e6 / max(duration_s, 1e-9)

    def _build_masks(self, layer: Optional[int] = None) -> torch.Tensor:
        if layer is None:
            layer = self._cur_layer if self._cur_layer is not None else 0

        # valid UE if buffer > 0, NOOP always valid
        valid_ue = (self.buf > 0.0).unsqueeze(0).expand(self.n_rbg, -1)  # [M, U]

        # Rank constraint: UE rank >= total allocated layers for this RBG
        # We check: count(allocs in previous layers) < ue_rank
        # Note: This counts spatial layers per RBG. Allocating multiple RBGs on the same layer
        #       does NOT increase the rank count (it counts as 1 layer for those RBGs).
        if layer > 0:
            # prev_alloc: [layer, M]
            prev_alloc = self._alloc[:layer, :]
            # Count occurrences: [M, U]
            u_indices = torch.arange(self.n_ue, device=self.device).view(1, 1, -1)
            matches = (prev_alloc.unsqueeze(-1) == u_indices)
            counts = matches.sum(dim=0)  # [M, U]
            rank_ok = (counts < self.ue_rank.unsqueeze(0))  # [M, U]
            valid_ue = valid_ue & rank_ok

        masks = valid_ue
        noop_col = torch.ones((self.n_rbg, 1), device=self.device, dtype=torch.bool)
        return torch.cat([masks, noop_col], dim=1)  # [M, A]

    def _build_obs(self, layer: int) -> torch.Tensor:
        # Build structured features then pad/truncate to obs_dim.
        # Per-UE features (7 total; last two reserved):
        # 1. Normalized Past Averaged Throughput [1]: avg_tp_u / max(avg_tp)
        # 2. Normalized Rank of UE [1]: rank_u / max_ue_rank
        # 3. Normalized Number of Already Allocated RBGs [1] (layers < current)
        # 4. Normalized Downlink Buffer Status [1]: buf_u / max(buf)
        # 5. Normalized Wideband (CQI->MCS) [1]: mcs_u / 28
        # 6. Reserved (0)
        # 7. Reserved (0)
        
        norm_past_avg_tp = self.avg_tp / self.max_rate

        norm_ue_rank = torch.clamp(self.ue_rank / float(self.max_ue_rank), 0.0, 1.0)

        if layer > 0:
            prev_alloc = self._alloc[:layer, :]  # [L', M]
            alloc_counts = torch.zeros((self.n_ue,), device=self.device, dtype=torch.float32)
            for u in range(self.n_ue):
                alloc_counts[u] = (prev_alloc == u).sum()
            norm_allocated_rbgs = alloc_counts / float(max(self.n_rbg, 1))
        else:
            norm_allocated_rbgs = torch.zeros((self.n_ue,), device=self.device, dtype=torch.float32)

        max_buf = torch.clamp(self.buf.max(), min=self.eps)
        norm_buffer = self.buf / max_buf

        if not hasattr(self, "_curr_mcs"):
            self._sample_mcs()
        mcs = torch.tensor(self._curr_mcs, device=self.device, dtype=torch.float32)
        norm_wb_cqi = torch.clamp(mcs / float(self.max_mcs), 0.0, 1.0)

        reserved = torch.zeros((self.n_ue, 2), device=self.device, dtype=torch.float32)
        ue_feats = torch.stack(
            [norm_past_avg_tp, norm_ue_rank, norm_allocated_rbgs, norm_buffer, norm_wb_cqi],
            dim=1,
        )
        ue_feats = torch.cat([ue_feats, reserved], dim=1)  # [U, 7]
        core = ue_feats.reshape(-1).float()

        if core.numel() >= self.obs_dim:
            return core[: self.obs_dim].clone()
        out = torch.zeros((self.obs_dim,), device=self.device, dtype=torch.float32)
        out[: core.numel()] = core
        return out

    def _sample_mcs(self):
        if self.mcs_spread == 0:
            mcs = self.mcs_mean.copy()
        else:
            jitter = self.rng.integers(-self.mcs_spread, self.mcs_spread + 1, size=self.n_ue)
            mcs = np.clip(self.mcs_mean + jitter, 0, self.max_mcs)
        self._curr_mcs = np.asarray(mcs, dtype=int)
        return self._curr_mcs

    def _ensure_tti_start(self):
        if not self._arrivals_added:
            self.buf = self.buf + self.buf_arrival
            self._avg_tp_before_tti = self.avg_tp.clone()
            self._arrivals_added = True

    def _compute_layer_transitions(self, layer: int, obs: torch.Tensor, masks: torch.Tensor) -> List[Dict]:
        # reward computed after finishing layer allocation (across all RBGs)
        # 1) compute instantaneous per-UE throughput for *this layer only* and accumulate
        #    Use actually served amount (cap by remaining buffer).
        tp_layer = torch.zeros((self.n_ue,), device=self.device)
        served_layer = torch.zeros((self.n_ue,), device=self.device)
        served_rbg = torch.zeros((self.n_rbg,), device=self.device)
        buf_tmp = self.buf.clone()
        for m in range(self.n_rbg):
            ue = int(self._alloc[layer, m].item())
            if ue == self.noop:
                continue
            if buf_tmp[ue] <= 0:
                continue  # should be masked but keep safe
            served = self._served_bytes(ue)
            served = torch.minimum(buf_tmp[ue], served)
            served_rbg[m] = served
            buf_tmp[ue] = torch.clamp(buf_tmp[ue] - served, min=0.0)
            served_layer[ue] += served

        duration_s = self.tti_ms / 1000.0
        tp_layer = (served_layer * 8.0) / 1e6 / max(duration_s, 1e-9)

        # 2) update EMA throughput using this layer contribution (paper says reward after each layer iteration)
        self.avg_tp = self.ema_beta * self.avg_tp + (1.0 - self.ema_beta) * tp_layer

        # 3) Off-policy DSACD reward (paper Appendix D.4, Eq.20-21)
        avg_tp_before_tti = self._avg_tp_before_tti if self._avg_tp_before_tti is not None else self.avg_tp
        rewards_m = torch.empty((self.n_rbg,), device=self.device)

        for m in range(self.n_rbg):
            chosen = int(self._alloc[layer, m].item())

            raw_all = torch.empty((self.n_ue,), device=self.device, dtype=torch.float32)
            for u in range(self.n_ue):
                if self.buf[u] <= 0:
                    raw_all[u] = 0.0
                    continue

                Ru = float((avg_tp_before_tti[u] + self.eps).item())
                Tu = float(self._rate_mbps(u).item())

                if layer == 0:
                    raw_all[u] = Tu / Ru
                else:
                    prev = int(self._alloc[layer - 1, m].item())
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
                rewards_m[m] = 1.0 if (chosen == self.noop) else -1.0
            else:
                rewards_m[m] = 0.0

        # 5) Apply service (drain buffers) for this layer after reward computation
        for m in range(self.n_rbg):
            ue = int(self._alloc[layer, m].item())
            if ue == self.noop:
                continue
            served = served_rbg[m]
            self.buf[ue] = torch.clamp(self.buf[ue] - served, min=0.0)

        # next state after this layer
        next_layer_idx = min(layer + 1, self.n_layers - 1)
        next_masks = self._build_masks(layer=next_layer_idx)
        next_obs = self._build_obs(layer=next_layer_idx)

        out = []
        for m in range(self.n_rbg):
            out.append({
                "observation": obs.detach().cpu(),
                "next_observation": next_obs.detach().cpu(),
                "rbg_index": torch.tensor(m, dtype=torch.long),
                "action": torch.tensor(int(self._alloc[layer, m].item()), dtype=torch.long),
                "reward": torch.tensor(float(rewards_m[m].item()), dtype=torch.float32),
                "action_mask": masks[m].detach().cpu(),           # [A] bool
                "next_action_mask": next_masks[m].detach().cpu(), # [A] bool
            })
            self._last_transitions.append(out[-1])
        return out