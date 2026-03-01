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
        n_ue: int,
        max_sched_ue: int,
        n_layers: int,
        n_rbg: int,
        *,
        seed: int = 0,
        device: str = "cpu",
        # PF/throughput params
        ema_beta: float = 0.9,
        eps: float = 1e-6,
        # deterministic traffic/channel knobs
        buf_init: int = 50_000,
        buf_arrival: int = 50_000,
        tti_ms: float = 1.0,
        prbs_per_rbg: int = 18,
        n_symb: int = 14,
        overhead_re_per_prb: int = 18,
        internal_log: bool = False
    ):
        assert max_sched_ue >= 1, "Need at least 1 UE + NOOP."
        self.n_ue = n_ue
        self.max_sched_ue = max_sched_ue
        self.noop = self.max_sched_ue
        self.n_layers = n_layers
        self.n_rbg = n_rbg

        self.act_dim = self.max_sched_ue + 1
        self.obs_dim = (5+2*self.n_rbg)*self.max_sched_ue

        print(f"OBSERVATION SIZE: {self.obs_dim}")
        print(f"ACTION SIZE: {self.act_dim}")
        
        self.device = torch.device(device)

        self.max_mcs = 28  # max MCS index (0..28)
        self.max_ue_rank = 2
        self.ema_beta = float(ema_beta)
        self.eps = float(eps)
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
        self.avg_tp = torch.full((self.n_ue,), self.eps, device=self.device, dtype=torch.float32)          # EMA throughput (Mbps)
        self.ue_priority = torch.full((self.n_ue,), 0, device=self.device, dtype=torch.int)

        # Allocation caches (per TTI)
        self._alloc = torch.full((self.n_layers, self.n_rbg), self.noop, device=self.device, dtype=torch.long)
        self._last_transitions: List[Dict] = []
        self._cur_layer: Optional[int] = None

        # Per-UE rank capability (spatial layers supported), max rank = 2
        # self.ue_rank = (1 + (torch.arange(self.n_ue) % 2)).to(self.device).float()
        self.ue_rank = torch.ones((self.n_ue,))*2

        # Fading profile: fixed per-UE MCS mean (constant through layers/RBGs)
        self.rng = np.random.default_rng(seed)
        self.mcs_mean = self.rng.integers(5, self.max_mcs + 1, size=(self.n_ue,))
        # self.mcs_mean = np.full((self.n_ue,), 20, dtype=int)
        # self.mcs_mean = np.array([5,10,25,5])
        self.mcs_spread = 1

        self.internal_log = internal_log
        print(f"Initial mcs: {self.mcs_mean}")
        # Pseudo cross correlation table
        self.max_cross_corr = np.random.rand(self.n_ue, self.n_ue, self.n_rbg)
        self.max_cross_corr = (self.max_cross_corr + self.max_cross_corr.transpose(1, 0, 2)) / 2
        ue_indices = np.arange(self.n_ue)
        self.max_cross_corr[ue_indices, ue_indices, :] = 0.9
        
        self._sample_mcs()

    # ---------- Public API ----------
    def reset(self):
        self.t = 0
        self.buf.fill_(self.buf_init)
        self.avg_tp.fill_(self.eps)
        self._alloc.fill_(self.noop)
        self._last_transitions = []
        self._cur_layer = None

    def begin_tti(self):
        self._alloc.fill_(self.noop)
        self._last_transitions = []
        self._cur_layer = None
        # UE Selection
        # selected_ues = self.ue_priority.argsort(descending=True)[:self.max_sched_ue]
        selected_ues = torch.arange(self.max_sched_ue)   
        self.selected_ues = selected_ues
        self.selected_buf = self.buf[selected_ues]
        self.selected_rank = self.ue_rank[selected_ues]
        self.selected_avg_tp = self.avg_tp[selected_ues]
        self.selected_curr_mcs = self._curr_mcs[selected_ues]

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
        return self._compute_layer_transitions(
            layer=layer_ctx.layer,
            obs=layer_ctx.obs,
            masks=layer_ctx.masks_rbg,
        )

    def finish_tti(self):
        self.t += 1

        # Update ue selection priority, reset to 0 if allocated, increase 1 otherwise
        self.ue_priority +=1
        allocated_ue=[]
        for u in range(self.max_sched_ue):
            ue_id = self.selected_ues[u]
            if (u==self._alloc).any():
                self.ue_priority[ue_id]=0
                allocated_ue.append(int(ue_id))
        unscheduled_mask = torch.full((self.n_ue,),True, device=self.device, dtype=torch.bool)
        unscheduled_mask[allocated_ue] = False
  
        # Update USER_THROUGHPUT and BUFFER STATUS at the END OF TTI
        user_tp, remained_buf = self.user_rate_under_sinr()
        self.selected_buf = remained_buf
        self.selected_avg_tp = self.ema_beta * self.selected_avg_tp + (1.0 - self.ema_beta) * user_tp

        # Update buffers and throughput
        self.buf[self.selected_ues] = self.selected_buf
        self.avg_tp[self.selected_ues] = self.selected_avg_tp
        self.avg_tp[unscheduled_mask] = self.ema_beta * self.avg_tp[unscheduled_mask] + (1.0 - self.ema_beta) * 0.0

        # Add new data to buffers for the next TTI
        self.buf = torch.clamp(self.buf + self.buf_arrival, max=100000)
        
    def user_rate_under_sinr(self):
        alloc = self._alloc.detach().cpu().numpy()
        noop = self.noop
        served_bytes_ue_tti = np.zeros((self.max_sched_ue,), dtype=np.float32)
        duration_s = self.tti_ms / 1000.0
        buf_tmp = self.selected_buf.clone()

        for m in range(self.n_rbg):
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
                        corrs.append(self.max_cross_corr[self.selected_ues[u1], self.selected_ues[u2], m])
                if corrs:
                    max_cross_corr_rbg = max(corrs)

            # Model the SINR effect with a penalty
            penalty = 1.0 - max_cross_corr_rbg

            for l in range(self.n_layers):
                u = int(alloc[l, m].item())
                if u == noop:
                    continue
                tbs = self._served_bytes(self.selected_ues[u], m)
                effective_tbs = float(tbs) * penalty
                served = min(float(buf_tmp[u].item()), float(effective_tbs))
                buf_tmp[u] = max(0.0, float(buf_tmp[u].item()) - served)
                served_bytes_ue_tti[u] += served

        rate_ue = (served_bytes_ue_tti * 8.0) / 1e6 / max(duration_s, 1e-9)
        return torch.from_numpy(rate_ue).to(self.device), buf_tmp

    # ---------- Internals ----------
    def _served_bytes(self, ue: int, m: int) -> torch.Tensor:
        if not hasattr(self, "_curr_mcs"):
            self._sample_mcs()
        mcs = int(self.curr_mcs_subband[ue,m])
        tbs = tbs_38214_bytes(mcs, self.prbs_per_rbg, n_symb=self.n_symb, overhead_re_per_prb=self.overhead)
        return torch.tensor(float(tbs), device=self.device)

    def _rate_mbps(self, ue: int, m: int) -> torch.Tensor:
        served = self._served_bytes(ue, m)
        duration_s = self.tti_ms / 1000.0
        return (served * 8.0) / 1e6 / max(duration_s, 1e-9)

    def _build_masks(self, layer: Optional[int] = None) -> torch.Tensor:
        if layer is None:
            layer = self._cur_layer if self._cur_layer is not None else 0

        # valid UE if buffer > 0, NOOP always valid
        valid_ue = (self.selected_buf > 0.0).unsqueeze(0).expand(self.n_rbg, -1)  # [M, U]

        # Rank constraint: UE rank >= total allocated layers for this RBG
        # We check: count(allocs in previous layers) < ue_rank
        # Note: This counts spatial layers per RBG. Allocating multiple RBGs on the same layer
        #       does NOT increase the rank count (it counts as 1 layer for those RBGs).
  
        if layer > 0:
            # 1. Rank Check (Same as before)
            prev_allocs = self._alloc[:layer, :]
            u_indices = torch.arange(self.max_sched_ue, device=self.device).view(1, 1, -1)
            matches = (prev_allocs.unsqueeze(-1) == u_indices)
            counts = matches.sum(dim=0)  # [M, U]
            rank_ok = (counts < self.selected_rank.unsqueeze(0)) 

            # 2. Per-UE Continuity Check. Has the UE ever been seen in this RBG before?
            ever_seen = counts > 0  # [M, U]
            
            # Was the UE in the layer immediately before this one?
            last_layer_alloc = self._alloc[layer - 1, :]
            in_prev_layer = (last_layer_alloc.unsqueeze(-1) == torch.arange(self.max_sched_ue, device=self.device).view(1, -1)) # [M, U]

            # Logic: If ever_seen is True, then in_prev_layer MUST be True.
            # If ever_seen is False, the UE is a "new starter" and is valid.
            continuity_ok = (~ever_seen) | in_prev_layer
            
            valid_ue = valid_ue & rank_ok & continuity_ok

        masks = valid_ue
        noop_col = torch.ones((self.n_rbg, 1), device=self.device, dtype=torch.bool)
        return torch.cat([masks, noop_col], dim=1)  # [M, A]

    def _build_obs(self, layer: int) -> torch.Tensor:
        # Build structured features then pad/truncate to obs_dim.

        # 1. Normalized Past Averaged Throughput [1]: avg_tp_u / max(avg_tp)
        norm_past_avg_tp = self.selected_avg_tp / self.max_rate

        # 2. Normalized Rank of UE [1]: rank_u / max_ue_rank
        norm_ue_rank = torch.clamp(self.selected_rank / float(self.max_ue_rank), 0.0, 1.0)

        # 3. Normalized Number of Already Allocated RBGs [1] (layers < current)
        if layer > 0:
            prev_alloc = self._alloc[:layer, :]  # [L', M]
            alloc_counts = torch.zeros((self.max_sched_ue,), device=self.device, dtype=torch.float32)
            for u in range(self.max_sched_ue):
                alloc_counts[u] = (prev_alloc == u).sum()
            norm_allocated_rbgs = alloc_counts / float(max(self.n_rbg, 1))
        else:
            norm_allocated_rbgs = torch.zeros((self.max_sched_ue,), device=self.device, dtype=torch.float32)

        # 4. Normalized Downlink Buffer Status [1]: buf_u / max(buf)
        max_buf = torch.clamp(self.selected_buf.max(), min=self.eps)
        norm_buffer = self.selected_buf / max_buf

        # 5. Normalized Wideband (CQI->MCS) [1]: mcs_u / 28
        if not hasattr(self, "_curr_mcs"):
            self._sample_mcs()
        mcs = torch.tensor(self.selected_curr_mcs , device=self.device, dtype=torch.float32)
        norm_wb_cqi = torch.clamp(mcs / float(self.max_mcs), 0.0, 1.0)

        # 6. Normalized Subband CQI    
        norm_subband_cqi = torch.tensor(self.curr_mcs_subband[self.selected_ues,:], device=self.device, dtype=torch.float32) / float(self.max_mcs)

        # 7. Max Cross Correlation with previously scheduled UEs on the same RBG, Shape: [n_ue, n_rbg]
        max_corr_feat = torch.zeros((self.max_sched_ue, self.n_rbg), device=self.device, dtype=torch.float32)
        if layer > 0:
            # Convert table to tensor (shape [U, U, M])
            cross_corr = torch.as_tensor(self.max_cross_corr, device=self.device, dtype=torch.float32)
            prev_alloc = self._alloc[:layer, :]  # [L', M]
            candidate_global_ids = self.selected_ues

            for m in range(self.n_rbg):
                # Identify UEs scheduled in this RBG in previous layers
                scheduled_ues_local = prev_alloc[:, m]
                valid_mask = (scheduled_ues_local != self.noop)
                valid_ues_local = scheduled_ues_local[valid_mask]  # [k] local indices

                if valid_ues_local.numel() > 0:
                    # Map local indices to global indices
                    scheduled_ues_global = self.selected_ues[valid_ues_local]

                    # Get correlations between all candidates and already-scheduled UEs
                    sub_corr_matrix = cross_corr[candidate_global_ids.unsqueeze(1), scheduled_ues_global.unsqueeze(0), m]
                    vals, _ = sub_corr_matrix.max(dim=1)
                    max_corr_feat[:, m] = vals

        ue_feats = torch.stack(
            [norm_past_avg_tp, norm_ue_rank, norm_allocated_rbgs, norm_buffer, norm_wb_cqi],
            dim=1,
        )
        
        if(self.internal_log):
          print("UE Feats:", ue_feats)

        ue_feats = torch.cat([ue_feats, norm_subband_cqi, max_corr_feat], dim=1)  # [U, 5 + 2*n_rbg]
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
            jitter = self.rng.integers(-self.mcs_spread, self.mcs_spread + 1, size=(self.n_ue))
            mcs = np.clip(self.mcs_mean + jitter, 0, self.max_mcs)
        self._curr_mcs = np.asarray(mcs, dtype=int)

        jitter = self.rng.integers(-self.mcs_spread, self.mcs_spread + 1, size=(self.n_ue, self.n_rbg))
        self.curr_mcs_subband = np.clip(self._curr_mcs.reshape(-1,1) + jitter, 0, self.max_mcs)

        return self._curr_mcs

    def _compute_layer_transitions(self, layer: int, obs: torch.Tensor, masks: torch.Tensor) -> List[Dict]:
        
        # Calculating reward for current layer
        rewards_m, _ = self.new_reward_compute(layer, masks)
        
        # Generating next observations
        if(self.internal_log):
          print("Next obesrvation:")
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
    
    def compute_set_tput(self, alloc_set, m):
        if len(alloc_set) == 0:
            return 0.0

        max_corr = 0.0
        for i in range(len(alloc_set) - 1):
            u = alloc_set[i]
            for j in range(i + 1, len(alloc_set)):
                v = alloc_set[j]
                max_corr = max(
                    max_corr,
                    float(self.max_cross_corr[self.selected_ues[u], self.selected_ues[v], m].item())
                )

        penalty = (1.0 - max_corr) / max(len(alloc_set), 1)

        tput = 0.0
        for u in alloc_set:
            if self.selected_buf[u] > 0:
                tput += float(self._rate_mbps(self.selected_ues[u], m).item())

        return tput * penalty

    def new_reward_compute(self, layer: int, masks: torch.Tensor):
        rewards_m = torch.zeros((self.n_rbg,), device=self.device)
        set_tp_per_rbg = torch.zeros((self.n_rbg,), device=self.device)
        noop = self.noop

        Ru_all = self.selected_avg_tp.clone()

        for m in range(self.n_rbg):
            # ---------- previous set ----------
            prev_alloc = []
            if layer > 0:
                for l_prev in range(layer):
                    u_prev = int(self._alloc[l_prev, m].item())
                    if u_prev != noop:
                        prev_alloc.append(u_prev)

            T_prev = self.compute_set_tput(prev_alloc, m)

            # ---------- compute raw rewards for all candidates ----------
            chosen = int(self._alloc[layer, m].item())
            if chosen == noop:
                set_tp_per_rbg[m] = T_prev

            raw_all = torch.zeros((self.max_sched_ue,), device=self.device)

            for u in range(self.max_sched_ue):
                if not masks[m, u] or self.selected_buf[u] <= 0:
                    continue

                curr_alloc = prev_alloc + [u]
                T_cur = self.compute_set_tput(curr_alloc, m)

                if u == chosen:
                    set_tp_per_rbg[m] = T_cur

                raw_all[u] = (T_cur - T_prev) / float(Ru_all[u].item())

            # ---------- normalization (Eq. 21) ----------
            max_raw = torch.max(raw_all)

            if max_raw > 0.0:
                if chosen == noop:
                    rewards_m[m] = 0.0
                else:
                    rewards_m[m] = torch.clamp(raw_all[chosen] / max_raw, -1.0, 1.0)
            elif max_raw < 0.0:
                # all UE choices reduce throughput → noop is optimal
                rewards_m[m] = 1.0 if chosen == noop else -1.0
            else:
                rewards_m[m] = 0.0

            if self.internal_log:
                print(f"Computing reward for rgb {m} ....")
                print(f"prev_alloc={prev_alloc}, T_prev={T_prev}")
                print(f"chosen={chosen}, T_cur={set_tp_per_rbg[m]}")
                print(f"raw_all={raw_all}")
                print(f"rewards_m={rewards_m}")
        return rewards_m, set_tp_per_rbg