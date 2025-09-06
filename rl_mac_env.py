#!/usr/bin/env python3
\"\"\"
rl_mac_env.py (fairness-aware)
Fixed-N=4 MAC RL environment with masking, demand-aware PRB projection,
38.214-like TBS, and reward = throughput + fairness (Jain over EMA) - latency.

Reward (per TTI):
  r_t = alpha * sum(served_bits_Mb)
      + beta  * Jain(throughput_ema_per_UE)
      - gamma * mean(HOL_ms_norm)

CLI/trainer can set: alpha, beta, gamma, fairness_ema_rho.
\"\"\"
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# -----------------------------
# MCS table (Qm, R*1024) simplified 38.214-like
# -----------------------------
MCS_TABLE = [
    (2, 120), (2, 157), (2, 193), (2, 251), (2, 308), (2, 379),
    (4, 449), (4, 526), (4, 602), (4, 679), (6, 340), (6, 378),
    (6, 434), (6, 490), (6, 553), (6, 616), (6, 658), (8, 438),
    (8, 466), (8, 517), (8, 567), (8, 616), (8, 666), (8, 719),
    (8, 772), (8, 822), (8, 873), (8, 910), (8, 948)
]  # idx 0..28

def tbs_38214_bytes(mcs_idx, n_prb, n_symb=14, n_layers=1, overhead_re_per_prb=18):
    \"\"\"Pragmatic 38.214-like TBS (single codeword); returns BYTES for this TTI.\"\"\"
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

# -----------------------------
# Projection: scores -> integer PRBs (mask + cap + redistribute)
# -----------------------------
def project_scores_to_prbs(scores, prb_budget, ue_load_bytes, ue_mcs_idx, active_mask, n_symb=14, overhead=18):
    scores = np.array(scores, dtype=np.float64) * active_mask
    prb_budget = int(prb_budget)
    if prb_budget <= 0 or np.all(scores <= 0):
        return np.zeros(4, dtype=int)

    total = scores.sum()
    fracs = scores / total if total > 0 else np.zeros_like(scores)
    raw = fracs * prb_budget
    prbs = np.floor(raw).astype(int)

    leftover = prb_budget - prbs.sum()
    if leftover > 0:
        rema = raw - prbs
        for k in np.argsort(-rema):
            if leftover == 0:
                break
            if active_mask[k] > 0:
                prbs[k] += 1
                leftover -= 1

    # Demand-aware caps
    bpp = np.array([bytes_per_prb(int(m), n_symb, overhead) for m in ue_mcs_idx], dtype=float)
    caps = np.ceil(np.divide(ue_load_bytes, np.maximum(bpp, 1e-9))).astype(int)
    caps = np.maximum(0, caps) * active_mask.astype(int)

    prbs = np.minimum(prbs, caps)

    rem = prb_budget - int(prbs.sum())
    if rem > 0:
        remaining = caps - prbs
        priority = (raw - prbs) + 1e-6 * (ue_load_bytes * bpp)
        while rem > 0:
            elig = np.where(remaining > 0)[0]
            if elig.size == 0:
                break
            j = elig[np.argmax(priority[elig])]
            prbs[j] += 1
            remaining[j] -= 1
            rem -= 1
    return prbs.astype(int)

# -----------------------------
# Environment
# -----------------------------
class MACSchedulerEnv(gym.Env):
    \"\"\"
    Fixed N=4 with active_mask.
    Observation per UE: [load_norm, mcs_norm, (prev_prbs_norm?)] * 4 + [prb_budget_norm, active_mask(4)]
    Action: 4 non-negative scores -> PRBs via projection.
    \"\"\"
    metadata = {\"render_modes\": []}

    def __init__(self,
                 use_prev_prbs: bool = False,
                 tti_ms: float = 1.0,
                 duration_tti: int = 2000,
                 prb_budget: int = 273,
                 load_scale: float = 1e6,
                 n_symb: int = 14,
                 overhead_re_per_prb: int = 18,
                 alpha_throughput: float = 1.0,
                 beta_fairness: float = 0.2,
                 gamma_latency: float = 0.05,
                 fairness_ema_rho: float = 0.9,
                 traffic_profile: str = \"mixed\",
                 fading_profile: str = \"fast\",
                 seed: int = 42):
        super().__init__()
        self.use_prev_prbs = use_prev_prbs
        self.tti_ms = tti_ms
        self.duration_tti = duration_tti
        self.max_prb = prb_budget
        self.load_scale = load_scale
        self.n_symb = n_symb
        self.overhead = overhead_re_per_prb
        self.alpha = alpha_throughput
        self.beta = beta_fairness
        self.gamma = gamma_latency
        self.rho = fairness_ema_rho
        self.traffic_profile = traffic_profile
        self.fading_profile = fading_profile
        self.rng = np.random.default_rng(seed)

        per_ue_dim = 2 + (1 if use_prev_prbs else 0)
        self.obs_dim = per_ue_dim * 4 + 1 + 4
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=10.0, shape=(4,), dtype=np.float32)

        self.t = 0
        self.reset_model_state()

    def reset_model_state(self):
        self.backlog = np.zeros(4, dtype=float)
        self.hol_ms = np.zeros(4, dtype=float)
        self.prev_prbs = np.zeros(4, dtype=int)
        self.active_mask = np.ones(4, dtype=int)

        if self.fading_profile == \"fast\":
            self.mcs_mean = np.array([14, 18, 10, 22]); self.mcs_spread = 4
        elif self.fading_profile == \"slow\":
            self.mcs_mean = np.array([12, 16, 9, 20]); self.mcs_spread = 2
        else:
            self.mcs_mean = np.array([12, 18, 10, 22]); self.mcs_spread = 0

        if self.traffic_profile == \"full_buffer\":
            self.arrival_bps = np.array([1e12, 1e12, 1e12, 1e12])
            self.period_ms = np.array([0, 0, 0, 0]); self.period_bytes = np.array([0, 0, 0, 0])
        elif self.traffic_profile == \"mixed\":
            self.arrival_bps = np.array([50e6, 0.0, 8e6, 15e6])
            self.period_ms = np.array([0, 20, 0, 0]); self.period_bytes = np.array([0, 80, 0, 0])
        else:  # poisson
            self.arrival_bps = np.array([40e6, 10e6, 12e6, 20e6])
            self.period_ms = np.zeros(4, dtype=int); self.period_bytes = np.zeros(4, dtype=int)

        self.lam_bytes_per_ms = self.arrival_bps / 8.0 / 1000.0
        self.thr_ema_mbps = np.zeros(4, dtype=float)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.reset_model_state()
        return self._get_obs(), {}

    def step(self, action):
        mcs = self._sample_mcs()
        arrivals = self._arrivals()
        self.backlog += arrivals

        self.hol_ms[self.backlog > 0] += self.tti_ms
        self.hol_ms[self.backlog <= 0] = 0.0

        scores = np.clip(np.array(action, dtype=float), 0.0, None)
        prbs = project_scores_to_prbs(scores, self.max_prb, self.backlog, mcs, self.active_mask,
                                      n_symb=self.n_symb, overhead=self.overhead)

        served = np.zeros(4, dtype=float)
        for i in range(4):
            tbs = tbs_38214_bytes(int(mcs[i]), int(prbs[i]), n_symb=self.n_symb, overhead_re_per_prb=self.overhead)
            served[i] = min(self.backlog[i], tbs)
            self.backlog[i] -= served[i]
            if self.backlog[i] <= 0:
                self.hol_ms[i] = 0.0

        duration_s = self.tti_ms / 1000.0
        inst_mbps = (served * 8.0) / 1e6 / max(duration_s, 1e-9)
        self.thr_ema_mbps = self.rho * self.thr_ema_mbps + (1.0 - self.rho) * inst_mbps

        served_bits = served.sum() * 8.0
        hol_norm = np.minimum(self.hol_ms / 200.0, 1.0)
        jain = jain_fairness(self.thr_ema_mbps)
        reward = self.alpha * (served_bits / 1e6) + self.beta * jain - self.gamma * float(np.mean(hol_norm))

        self.prev_prbs = prbs
        self.t += 1
        terminated = False
        truncated = self.t >= self.duration_tti
        info = {\"mcs\": mcs, \"arrivals\": arrivals, \"served_bytes\": served, \"prbs\": prbs,
                \"backlog\": self.backlog.copy(), \"hol_ms\": self.hol_ms.copy(), \"jain\": jain,
                \"thr_ema_mbps\": self.thr_ema_mbps.copy()}
        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self):
        obs = []
        for i in range(4):
            load_norm = min(self.backlog[i] / 1e6, 10.0)
            mcs_norm = getattr(self, \"_last_mcs\", np.zeros(4))[i] / 28.0 if hasattr(self, \"_last_mcs\") else 0.0
            obs.extend([load_norm, float(np.clip(mcs_norm, 0.0, 1.0))])
            if self.use_prev_prbs:
                obs.append(float(self.prev_prbs[i]) / self.max_prb)
        obs.append(self.max_prb / 273.0)
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
        arr = self.rng.poisson(self.lam_bytes_per_ms, size=4).astype(float)
        for i in range(4):
            if self.period_ms[i] > 0 and (self.t % self.period_ms[i] == 0):
                arr[i] += self.period_bytes[i]
        self.active_mask[:] = 1
        return arr
