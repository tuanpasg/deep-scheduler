#!/usr/bin/env python3
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ----------------------
# 38.214-inspired helpers
# ----------------------
MCS_TABLE = [
    (2,120),(2,157),(2,193),(2,251),(2,308),(2,379),
    (4,449),(4,526),(4,602),(4,679),(6,340),(6,378),
    (6,434),(6,490),(6,553),(6,616),(6,658),(8,438),
    (8,466),(8,517),(8,567),(8,616),(8,666),(8,719),
    (8,772),(8,822),(8,873),(8,910),(8,948)
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

def bytes_per_prb(mcs_idx, n_symb=14, overhead_re_per_prb=18):
    return tbs_38214_bytes(mcs_idx, 1, n_symb=n_symb, overhead_re_per_prb=overhead_re_per_prb)

def jain_fairness(values):
    v = np.asarray(values, dtype=float)
    if np.all(v == 0):
        return 1.0
    num = (v.sum())**2
    den = len(v) * (v**2).sum() + 1e-12
    return float(num / den)

def project_scores_to_prbs(scores, prb_budget, ue_load_bytes, ue_mcs_idx, active_mask,
                           n_symb=14, overhead=18, training_mode=False):
    """
    Returns:
      prbs_out: PRBs that will actually be allocated/served (size 4, int)
      prbs_pre: PRBs computed from scores before masking/redistribution (size 4, int)
      wasted_prbs: int number of PRBs assigned to inactive UEs (prbs_pre * (1-active_mask))
    Behavior:
      - training_mode=False: existing safe behavior (mask early or redistribute as before)
      - training_mode=True: compute prbs_pre normally, then prbs_out = prbs_pre * active_mask (no redistribution)
    """
    scores = np.array(scores, dtype=np.float64)
    prb_budget = int(prb_budget)
    if prb_budget <= 0 or np.all(scores <= 0):
        return np.zeros(4, dtype=int), np.zeros(4, dtype=int), 0

    total = scores.sum()
    fracs = scores / total if total > 0 else np.zeros_like(scores)
    raw = fracs * prb_budget
    prbs_pre = np.floor(raw).astype(int)

    # ensure sum(prbs_pre) == prb_budget by distributing leftover (same as before)
    leftover = prb_budget - prbs_pre.sum()
    if leftover > 0:
        rema = raw - prbs_pre
        for k in np.argsort(-rema):
            if leftover == 0:
                break
            prbs_pre[k] += 1
            leftover -= 1

    # robust caps (backlog-based): as before
    bpp_raw = np.array([bytes_per_prb(int(m), n_symb, overhead) for m in ue_mcs_idx], dtype=float)
    bpp = np.maximum(bpp_raw, 4.0)
    caps = np.minimum(np.ceil(np.divide(ue_load_bytes, bpp)).astype(int), prb_budget)
    caps = np.maximum(0, caps)  # **do not multiply by active_mask yet**

    # clip pre-alloc to caps (a UE cannot use more PRBs than its backlog allows)
    prbs_pre = np.minimum(prbs_pre, caps)

    if training_mode:
        # train-mode: do NOT redistribute PRBs intended for inactive UEs
        prbs_out = prbs_pre * active_mask.astype(int)   # lost prbs vanish
        wasted_prbs = int((prbs_pre * (1 - active_mask)).sum())
    else:
        # deployment: mask scores early / redistribute as before (safe behavior)
        # Here we mimic previous behavior: zero scores, compute allocation with caps+active_mask and redistribution
        # Option A: emulate original: recalc using masked scores (simpler)
        scores_masked = scores * active_mask
        total2 = scores_masked.sum()
        if total2 <= 0:
            prbs_out = np.zeros(4, dtype=int)
        else:
            raw2 = (scores_masked / total2) * prb_budget
            prbs_out = np.floor(raw2).astype(int)
            leftover2 = prb_budget - prbs_out.sum()
            if leftover2 > 0:
                rema2 = raw2 - prbs_out
                for k in np.argsort(-rema2):
                    if leftover2 == 0:
                        break
                    if active_mask[k] > 0:
                        prbs_out[k] += 1
                        leftover2 -= 1
            # enforce caps & active mask
            caps_masked = caps * active_mask.astype(int)
            prbs_out = np.minimum(prbs_out, caps_masked)

        wasted_prbs = int((prbs_pre * (1 - active_mask)).sum())  # for bookkeeping

    return prbs_out.astype(int), prbs_pre.astype(int), wasted_prbs

# ----------------------
# Environment
# ----------------------
class MACSchedulerEnv(gym.Env):
    """4-UE toy MAC scheduler env with stable reward/obs scaling and no HOL term."""
    metadata = {'render_modes': []}

    def __init__(self,
                 use_prev_prbs: bool = False,
                 tti_ms: float = 1.0,
                 duration_tti: int = 2000,
                 prb_budget: int = 273,
                 n_symb: int = 14,
                 overhead_re_per_prb: int = 18,
                 alpha_throughput: float = 1.0,
                 beta_fairness: float = 0.2,
                 gamma_latency: float = 0.0,  # kept for compatibility, ignored
                 fairness_ema_rho: float = 0.9,
                 traffic_profile: str = 'mixed',
                 fading_profile: str = 'fast',
                 seed: int = 42,
                 training_mode=True):
        
        super().__init__()
        self.use_prev_prbs = use_prev_prbs
        self.tti_ms = tti_ms
        self.duration_tti = duration_tti
        self.max_prb = prb_budget
        self.n_symb = n_symb
        self.overhead = overhead_re_per_prb
        self.alpha = alpha_throughput
        self.beta = beta_fairness
        self.rho = fairness_ema_rho
        self.traffic_profile = traffic_profile
        self.fading_profile = fading_profile
        self.rng = np.random.default_rng(seed)
        self.training_mode = training_mode

        # Observation: per-UE [backlog_norm, mcs_norm, (prev_prbs_norm if enabled)] + [prb_budget_norm] + [active_mask(4)]
        # All features are in [0,1].
        self.load_clip_bytes = 2_000_000  # 2 MB clip for normalization to [0,1]
        per_ue_dim = 2 + (1 if use_prev_prbs else 0)
        self.obs_dim = per_ue_dim * 4 + 1 + 4
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        # Action: non-negative allocation scores per UE in [0,1]; relative proportions matter
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        self.t = 0
        self.reset_model_state()

    # ----------------------
    # Lifecycle
    # ----------------------
    def reset_model_state(self):
        # Initial zeros; start inactive; prev_prbs 0
        self.backlog = np.zeros(4, dtype=float)
        self.prev_prbs = np.zeros(4, dtype=int)
        self.active_mask = np.zeros(4, dtype=int)  # start inactive

        # Fading profile
        if self.fading_profile == 'fast':
            self.mcs_mean = np.array([14,18,10,22]); self.mcs_spread = 4
        elif self.fading_profile == 'slow':
            self.mcs_mean = np.array([12,16,9,20]); self.mcs_spread = 2
        else:
            self.mcs_mean = np.array([12,18,10,22]); self.mcs_spread = 0

        # Traffic profile (bytes per ms via Poisson; plus optional periodic bursts)
        if self.traffic_profile == 'full_buffer':
            arrival_bps = np.array([1e12,1e12,1e12,1e12])  # effectively infinite
            self.period_ms = np.array([0,0,0,0]); self.period_bytes = np.array([0,0,0,0])
        elif self.traffic_profile == 'mixed':
            arrival_bps = np.array([50e6,0.0,8e6,15e6])
            self.period_ms = np.array([0,20,0,0]); self.period_bytes = np.array([0,80,0,0])
        else:
            arrival_bps = np.array([40e6,10e6,12e6,20e6])
            self.period_ms = np.zeros(4, dtype=int); self.period_bytes = np.zeros(4, dtype=int)

        self.lam_bytes_per_ms = arrival_bps / 8.0 / 1000.0

        # Throughput EMA init (tiny non-zero to avoid jain=1 cold-start)
        self.thr_ema_mbps = np.ones(4, dtype=float) * 1e-6

        # Precompute max Mb per TTI for reward normalization
        max_bpp = bytes_per_prb(28, n_symb=self.n_symb, overhead_re_per_prb=self.overhead)
        self.max_mb_per_tti = (self.max_prb * max_bpp * 8.0) / 1e6  # Mb/TTI at highest MCS

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.reset_model_state()
        # Sample an initial MCS vector for _get_obs normalization
        initial_arrivals = self._arrivals()
        self.backlog += initial_arrivals
        self.active_mask = (self.backlog > 0).astype(int)
        self._last_mcs = self._sample_mcs()
        return self._get_obs(), {}

    def step(self, action):
        # 1) Channel sample for this TTI
        mcs = self._sample_mcs()

        # 2) Use current backlog/active_mask (same state the agent observed)
        scores = np.clip(np.array(action, dtype=float), 0.0, 1.0)
        prbs_out, prbs_pre, wasted_prbs = project_scores_to_prbs(
            scores, self.max_prb, self.backlog, mcs, self.active_mask,
            n_symb=self.n_symb, overhead=self.overhead, training_mode=self.training_mode
        )

        # 3) Serve according to prbs_out
        served = np.zeros(4, dtype=float)
        for i in range(4):
            tbs = tbs_38214_bytes(int(mcs[i]), int(prbs_out[i]), n_symb=self.n_symb, overhead_re_per_prb=self.overhead)
            s = min(self.backlog[i], tbs)
            served[i] = s
            self.backlog[i] -= s

        # 4) Fairness EMA (Mb/s) based on served this TTI
        duration_s = self.tti_ms / 1000.0
        inst_mbps = (served * 8.0) / 1e6 / max(duration_s, 1e-9)
        self.thr_ema_mbps = self.rho * self.thr_ema_mbps + (1.0 - self.rho) * inst_mbps

        # 5) Reward (no HOL): bounded O(1)
        served_mb_tti = (served.sum() * 8.0) / 1e6
        throughput_norm = served_mb_tti / (self.max_mb_per_tti + 1e-12)  # ~[0,1]
        jain = jain_fairness(self.thr_ema_mbps)
        reward = self.alpha * throughput_norm + self.beta * jain

        # 6) Arrivals now produce next state s_{t+1}
        arrivals = self._arrivals()
        self.backlog += arrivals
        self.active_mask = (self.backlog > 0).astype(int)

        # 7) Finalize and info
        self.prev_prbs = prbs_out
        self.t += 1
        terminated = False
        truncated = self.t >= self.duration_tti

        bpp_raw = np.array([bytes_per_prb(int(m), self.n_symb, self.overhead) for m in mcs], dtype=float)
        bpp = np.maximum(bpp_raw, 4.0)
        wasted_prbs_array = (prbs_pre - prbs_out).clip(min=0)
        wasted_bytes = float((wasted_prbs_array * bpp).sum())

        info = {
            'mcs': mcs,
            'arrivals': arrivals,
            'served_bytes': served,
            'prbs': prbs_out,
            'backlog': self.backlog.copy(),  # post-arrivals (next state)
            'jain': jain,
            'thr_ema_mbps': self.thr_ema_mbps.copy(),
            'cell_tput_Mb': float(served_mb_tti),
            'wasted_prbs': int(wasted_prbs),
            'wasted_bytes': wasted_bytes
        }

        next_obs = self._get_obs()
        return next_obs, float(reward), terminated, truncated, info

    # ----------------------
    # Internals
    # ----------------------
    def _get_obs(self):
        obs = []
        # per-UE features
        for i in range(4):
            load_norm = float(np.clip(self.backlog[i] / self.load_clip_bytes, 0.0, 1.0))
            mcs_norm = float(np.clip(getattr(self, '_last_mcs', np.zeros(4))[i] / 28.0, 0.0, 1.0))
            obs.extend([load_norm, mcs_norm])
            if self.use_prev_prbs:
                obs.append(float(self.prev_prbs[i]) / max(1, self.max_prb))
        # global prb budget (normalized to 273)
        obs.append(float(self.max_prb) / 273.0)
        # active mask
        obs.extend(self.active_mask.astype(float).tolist())
        return np.array(obs, dtype=np.float32)

    def _sample_mcs(self):
        if self.mcs_spread == 0:
            mcs = self.mcs_mean.copy()
        else:
            jitter = self.rng.integers(-self.mcs_spread, self.mcs_spread + 1, size=4)
            mcs = np.clip(self.mcs_mean + jitter, 0, 28)
        self._last_mcs = mcs.astype(int)
        return self._last_mcs

    def _arrivals(self):
        # Poisson arrivals (bytes/ms) + optional periodic bursts
        arr = self.rng.poisson(self.lam_bytes_per_ms, size=4).astype(float)
        for i in range(4):
            if self.period_ms[i] > 0 and (self.t % self.period_ms[i] == 0):
                arr[i] += self.period_bytes[i]
        # Active if backlog will be >0 this TTI (after adding arrivals)
        # future_backlog = self.backlog + arr
        # self.active_mask = ((future_backlog > 0).astype(int))
        return arr
