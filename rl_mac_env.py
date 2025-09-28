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
        return np.zeros(4, dtype=int), np.zeros(4, dtype=int), 0, 0

    total = scores.sum()
    fracs = scores / total if total > 0 else np.zeros_like(scores)
    raw = fracs * prb_budget
    prbs_pre = np.floor(raw).astype(int)
    
    # print(
    #   f"scores={scores} prbs_pre={prbs_pre}"
    # )
    # ensure sum(prbs_pre) == prb_budget by distributing leftover (same as before)
    leftover = prb_budget - prbs_pre.sum()
    if leftover > 0:
        rema = raw - prbs_pre
        for k in np.argsort(-rema):
            if leftover == 0:
                break
            prbs_pre[k] += 1
            leftover -= 1

    # print(f"prbs_pre={prbs_pre}")
    # robust caps (backlog-based): as before
    bpp_raw = np.array([bytes_per_prb(int(m), n_symb, overhead) for m in ue_mcs_idx], dtype=float)
    bpp = np.maximum(bpp_raw, 4.0)
    caps = np.minimum(np.ceil(np.divide(ue_load_bytes, bpp)).astype(int), prb_budget)
    caps = np.maximum(0, caps)  # **do not multiply by active_mask yet**

    # clip pre-alloc to caps (a UE cannot use more PRBs than its backlog allows)
    prbs_pre = np.minimum(prbs_pre, caps)

    # Metric: PRBs agent assigned to invalid UEs (before masking)
    invalid_allocated_prbs = int((prbs_pre * (1 - active_mask)).sum())

    # print(f"prbs_pre={prbs_pre} active_mask={active_mask}")
    if training_mode:
        # train-mode: do NOT redistribute PRBs intended for inactive UEs
        # Start from prbs_pre but mask invalid UEs (they get 0 actual PRBs)
        prbs_out = (prbs_pre * active_mask.astype(int)).astype(int)

        # How many PRBs are already assigned to valid UEs
        allocated_valid = int(prbs_out.sum())

        # Remaining PRBs available for redistribution (per your definition)
        remaining_prbs = prb_budget - allocated_valid - invalid_allocated_prbs
        # remaining_prbs may be >0 when prbs_pre sum < prb_budget (due to caps) OR when agent
        # assigned PRBs to invalid UEs (those vanish from prbs_out but we count them as 'removed').

        if remaining_prbs > 0:
            # compute how many PRBs each valid UE still needs (caps minus current assigned)
            remaining_caps = np.maximum(caps - prbs_out, 0).astype(int)

            # build candidate list: valid UEs that still need PRBs (backlog > 0 after current allocation)
            candidates = np.where((active_mask == 1) & (remaining_caps > 0))[0]

            if candidates.size > 0:
                # order candidates by agent score (high -> low)
                # if scores contains ties, np.argsort keeps deterministic order
                order = candidates[np.argsort(-scores[candidates])]

                # allocate greedily in that order until we run out or all needs satisfied
                for ue in order:
                    if remaining_prbs <= 0:
                        break
                    need = int(remaining_caps[ue])
                    if need <= 0:
                        continue
                    give = min(need, remaining_prbs)
                    prbs_out[ue] += give
                    remaining_prbs -= give
                    remaining_caps[ue] -= give
                    # stop early if no more remaining_prbs
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

    # # unassigned PRBs (system-level)
    # unassigned_prbs = int(prb_budget - int(np.sum(prbs_out)))

    # # new definition of wasted_prbs (per your request):
    # # - only count unassigned_prbs as 'wasted_prbs' IF there exist zero-score UEs
    # #   that have non-empty backlog (i.e., agent gave score==0 but backlog>0).
    # zero_score_nonempty = np.logical_and(scores == 0.0, ue_load_bytes > 0.0)
    # if np.any(zero_score_nonempty):
    #     wasted_prbs = unassigned_prbs
    # else:
    #     wasted_prbs = 0
    wasted_prbs = prb_budget - invalid_allocated_prbs - int(prbs_out.sum())

    return prbs_out.astype(int), prbs_pre.astype(int), int(wasted_prbs), int(invalid_allocated_prbs)

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
                 training_mode=True,
                 reward_mode: str = "throughput_jain",   # "throughput_jain" | "pf_gain" | "pf_ratio"
                 pf_clip: float = 1.0,                   # clip for delta-PF reward after normalization
                 pf_ratio_clip: float = 10.0,            # clamp r_i / Rbar_i to avoid spikes
                 pf_eps: float = 1e-6                    # numerical floor for logs/ratios
                 ):
        
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
        self.reward_mode = reward_mode
        self.pf_clip = float(pf_clip)
        self.pf_ratio_clip = float(pf_ratio_clip)
        self.pf_eps = float(pf_eps)


        self.global_step = 0
        self.max_mcs = 28  # max MCS index (0..28)

        # Observation: per-UE [backlog_norm, mcs_norm, (prev_prbs_norm if enabled)] + [prb_budget_norm] + [active_mask(4)]
        # All features are in [0,1].
        self.load_clip_bytes = 2_000_000  # 2 MB clip for normalization to [0,1]
        per_ue_dim = 2 + (1 if use_prev_prbs else 0)
        # self.obs_dim = per_ue_dim * 4 + 1 + 4
        self.obs_dim = 4*4
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
            self.mcs_mean = np.array([2,16,7,27]); self.mcs_spread = 2
        else:
            self.mcs_mean = np.array([12,18,10,22]); self.mcs_spread = 0

        # Traffic profile (bytes per ms via Poisson; plus optional periodic bursts)
        if self.traffic_profile == 'full_buffer':
            arrival_bps = np.array([1e12,1e12,1e12,1e12])  # effectively infinite
            self.period_ms = np.array([0,0,0,0]); self.period_bytes = np.array([0,0,0,0])
        elif self.traffic_profile == 'mixed':
            arrival_bps = np.array([50e6,0.0,8e6,15e6]) # Number of bits per second
            self.period_ms = np.array([0,20,0,0]); self.period_bytes = np.array([0,80,0,0])
        else:
            arrival_bps = np.array([40e6,10e6,12e6,20e6])
            self.period_ms = np.zeros(4, dtype=int); self.period_bytes = np.zeros(4, dtype=int)

        self.lam_bytes_per_ms = arrival_bps / 8.0 / 1000.0
        self.arrival_bps = (self.lam_bytes_per_ms * 8.0 * 1000.0).astype(float)
        # Throughput EMA init (tiny non-zero to avoid jain=1 cold-start)
        self.thr_ema_mbps = np.ones(4, dtype=float) * 1e-6

        # Precompute max Mb per TTI for reward normalization
        max_bpp = bytes_per_prb(28, n_symb=self.n_symb, overhead_re_per_prb=self.overhead)
        self.max_mb_per_tti = (self.max_prb * max_bpp * 8.0) / 1e6  # Mb/TTI at highest MCS

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.backlog = np.zeros(4, dtype=float)
        self.prev_prbs = np.zeros(4, dtype=int)
        self.active_mask = np.zeros(4, dtype=int)  # start inactive
        self.thr_ema_mbps = np.ones(4, dtype=float) * 1e-6
        # self.reset_model_state()

        self.global_step += 1
        if self.global_step == 5:
            # Randomize scheme 0
            # bump mcs_mean (vector) by +5, wrap around 0..max_mcs
            # self.mcs_mean = ((np.array(self.mcs_mean, dtype=int) + 2) % (self.max_mcs + 1)).astype(int)

            # # bump arrival_bps by +10e6 and wrap modulo 50e6
            # updated_arrivals = (np.array(self.arrival_bps, dtype=float) + 5e6) % 50e6
            # updated_arrivals[np.isclose(updated_arrivals, 0.0)] = 5e6
            # self.arrival_bps = updated_arrivals

            # Randomize scheme 1
            # idx = np.random.permutation(len(self.mcs_mean))
            # self.mcs_mean[:] = self.mcs_mean[idx]
            # self.arrival_bps[:] = self.arrival_bps[idx]
            
            # Randomizing scheme 2

            self.mcs_mean = self.rng.integers(low=2, high=28, size=4)
            self.arrival_bps = self.rng.integers(low=5, high=50, size=4)*1e6 # Bits per second

            self.lam_bytes_per_ms = self.arrival_bps / 8.0 / 1000.0 #bytes per milli-second or TTI
            print(f"[SCHEDULE] global_step={self.global_step} mcs_mean={self.mcs_mean.tolist()} arrival_bps={self.arrival_bps.tolist()}")

            self.global_step = 0
        # Reset initial state
        
        initial_arrivals = self._arrivals()
        self.backlog += initial_arrivals
        self.active_mask = (self.backlog > 0).astype(int)
        self._curr_mcs = self._sample_mcs()
        return self._get_obs(), {}

    def step(self, action):
        # 1) Retrieve mcs for this TTI
        mcs = self._curr_mcs.copy()

        # 2) Use current backlog/active_mask (same state the agent observed)
        scores = np.clip(np.array(action, dtype=float), 0.0, 1.0)
        prbs_out, prbs_pre, wasted_prbs, invalid_allocated_prbs = project_scores_to_prbs(
            scores, self.max_prb, self.backlog, mcs, self.active_mask,
            n_symb=self.n_symb, overhead=self.overhead, training_mode=self.training_mode
        )

        # 3) Serve according to prbs_out
        served = np.zeros(4, dtype=float) # [Served bytes in a TTI]
        for i in range(4):
            tbs = tbs_38214_bytes(int(mcs[i]), int(prbs_out[i]), n_symb=self.n_symb, overhead_re_per_prb=self.overhead)
            s = min(self.backlog[i], tbs)
            served[i] = s
            self.backlog[i] -= s
     
        # 4 & 5) Calculate reward
        served_mb_tti = (served.sum() * 8.0) / 1e6 # Total served load in this TTI [Megabits per TTI]
        throughput_norm = served_mb_tti / (self.max_mb_per_tti + 1e-12)  # ~[0,1] # Normalized served load in this TTI [Megabits per TTI]

        duration_s = self.tti_ms / 1000.0
        thr_inst_mbps = (served * 8.0) / 1e6 / max(duration_s, 1e-9) # Instantaneous rate per UE [Megabits per second]
        
        Rbar_old = self.thr_ema_mbps.copy()
        Rbar_new = self.rho * Rbar_old + (1.0 - self.rho) * thr_inst_mbps # Long-term rate [Megabits per second]
        
        jain = jain_fairness(Rbar_new) # Fairness metric calculation
        
        reward = None

        if self.reward_mode == "throughput_jain":
            # Your existing reward (leave as-is)
            reward = self.alpha * throughput_norm + self.beta * jain

        elif self.reward_mode == "pf_gain":
            # ----- Reward = normalized ΔPF utility (safer, goal-aligned) -----
            # EMA update: Rbar_new = rho * Rbar_old + (1 - rho) * r
            rho = self.rho
            U_old = np.sum(np.log(np.maximum(Rbar_old, self.pf_eps)))
            U_new = np.sum(np.log(np.maximum(Rbar_new, self.pf_eps)))
            dU = U_new - U_old

            # Normalize by (1 - rho) and number of UEs → roughly scale-free, ∈ ~[0,1]-ish
            denom = max((1.0 - rho) * len(Rbar_old), self.pf_eps)
            reward = np.clip(dU / denom, -self.pf_clip, self.pf_clip)

        elif self.reward_mode == "pf_ratio":
            # ----- Reward = sum_i clamp(r_i / Rbar_i) (bounded & simple) -----
            Rbar = np.maximum(Rbar_old, self.pf_eps)
            ratio = thr_inst_mbps / Rbar

            # Clamp extreme ratios to keep critic sane (cold-starts, bursts)
            ratio = np.clip(ratio, 0.0, self.pf_ratio_clip)

            # Normalize by UE count and clip to [0,1]
            reward = np.clip(np.sum(ratio) / (len(ratio) * self.pf_ratio_clip), 0.0, 1.0)

        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")        
        
        self.thr_ema_mbps = Rbar_new
        # print(
        #     f"[TTI {self.t:04d}] action=[{', '.join(f'{x:.3f}' for x in np.asarray(action, dtype=float))}] "
        #     f"mcs={mcs} "
        #     f"prbs_out={prbs_out.tolist()} served={served.tolist()} "
        #     f"served_mb_tti={served_mb_tti:.3f} max_mb_per_tti={self.max_mb_per_tti:.3f} "
        #     f"active_mask={self.active_mask.tolist()}"
        # ) 

        # 6) Arrivals now produce next state s_{t+1}
        arrivals = self._arrivals() # Number of bytes comming to UE's buffer per TTI(ms) 
        self.backlog += arrivals # Total number of bytes in each UE's buffer
        self.active_mask = (self.backlog > 0).astype(int)
        self.prev_prbs = prbs_out
        self._curr_mcs = self._sample_mcs()

        next_obs = self._get_obs()

        # print(f"arrivals={arrivals}")
        # 7) Finalize and info
        self.t += 1
        terminated = False
        truncated = self.t >= self.duration_tti

        # bpp_raw = np.array([bytes_per_prb(int(m), self.n_symb, self.overhead) for m in mcs], dtype=float)
        # bpp = np.maximum(bpp_raw, 4.0)
        # wasted_prbs_array = (prbs_pre - prbs_out).clip(min=0)
        # wasted_bytes = float((wasted_prbs_array * bpp).sum())

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
            'invalid_allocated_prbs':int(invalid_allocated_prbs)
            # 'wasted_bytes': wasted_bytes
        }

        return next_obs, float(reward), terminated, truncated, info

    # ----------------------
    # Internals
    # ----------------------
    def prep_obs(self, external_state:dict):
        # Calculate the efficient allocated bytes from the previous TTI to update backlog and average throughput
        self.prev_prbs = np.asarray(external_state["prev_prbs"],dtype=int)
        mcs = self._curr_mcs.copy()

        # Update backlog with allocated bytes
        served = np.zeros(4, dtype=float)
        for i in range(4):
            tbs = tbs_38214_bytes(int(mcs[i]), self.prev_prbs[i], n_symb=self.n_symb, overhead_re_per_prb=self.overhead)
            s = min(self.backlog[i], tbs)
            served[i] = s
     
        # Update average past throughput
        duration_s = self.tti_ms / 1000.0
        thr_inst_mbps = (served * 8.0) / 1e6 / max(duration_s, 1e-9) # Instantaneous rate per UE [Megabits per second]
        self.thr_ema_mbps = self.rho * self.thr_ema_mbps + (1.0 - self.rho) * thr_inst_mbps # Long-term rate [Megabits per second]
        
        self.backlog = np.asarray(external_state["loads"],dtype=float)
        self.active_mask = (self.backlog > 0).astype(int)
        self._curr_mcs = np.asarray(external_state["mcs"], dtype=int)
        self.max_prb = int(external_state.get("prb_budget", self.max_prb))

        return self._get_obs()

    def _sample_mcs(self):
        if self.mcs_spread == 0:
            mcs = self.mcs_mean.copy()
        else:
            jitter = self.rng.integers(-self.mcs_spread, self.mcs_spread + 1, size=4)
            mcs = np.clip(self.mcs_mean + jitter, 0, 28)
        self._curr_mcs = mcs.astype(int)
        return self._curr_mcs

    def compute_backlog_and_cap_features(self):
        # mcs: array-like of per-UE MCS indices (integers)
        # backlog: self.backlog (bytes per UE)
        # prev_prbs: self.prev_prbs (int PRBs assigned previously) or zeros if not used
        # self.max_prb exists
        mcs = self._curr_mcs.copy()
        # bytes per PRB per UE (you already use bytes_per_prb)
        bpp_raw = np.array([bytes_per_prb(int(m), n_symb=self.n_symb, overhead_re_per_prb=self.overhead)
                            for m in mcs], dtype=float)
        # numerical floor
        bpp = np.maximum(bpp_raw, 4.0)   # bytes per PRB

        # capacity (bytes) if whole TTI used at best MCS
        max_bpp = np.max(bpp)  # or a precomputed constant for best MCS
        capacity_per_tti_bytes = float(self.max_prb * max_bpp) + 1e-12

        # PRB caps computed from backlog & per-UE bpp (as your code)
        caps = np.minimum(np.ceil(np.divide(self.backlog, bpp)).astype(int), self.max_prb)
        caps = np.maximum(0, caps)   # ensure non-negative

        # 1) backlog_norm: fraction of whole-cell TTI capacity (0..1)
        backlog_norm = np.clip(self.backlog / capacity_per_tti_bytes, 0.0, 1.0)

        # optionally log-scale:
        backlog_norm_log = np.log1p(self.backlog) / np.log1p(capacity_per_tti_bytes)

        # 2) cap_remaining_norm: fraction of PRBs needed
        cap_remaining_norm = np.clip(caps.astype(float) / float(self.max_prb), 0.0, 1.0)

        # 3) remaining after prev PRBs (if you expose prev_prbs)
        prev_prbs = getattr(self, "prev_prbs", np.zeros_like(caps))
        remaining_after_prev = np.maximum(caps - prev_prbs.astype(int), 0)
        cap_remaining_after_prev_norm = np.clip(remaining_after_prev.astype(float) / float(self.max_prb), 0.0, 1.0)

        # 4) Optional small-backlog binary flag (helps policy avoid starving small flows)
        small_backlog_flag = (self.backlog <= (bpp * 1.0)).astype(float)  # needs <=1 PRB

        # print(
        #   f"aver_arrival_tti[bytes] = {self.lam_bytes_per_ms}\n"
        #   f"backlog[bytes]={self.backlog} upper_bound_tti[bytes]={capacity_per_tti_bytes}\n"
        #   f"needed_prbs={caps} norm={cap_remaining_norm}"
        # )
        return {
            "bpp": bpp,
            "caps": caps,
            "backlog_norm": backlog_norm,
            "backlog_norm_log": backlog_norm_log,
            "cap_remaining_norm": cap_remaining_norm,
            "cap_remaining_after_prev_norm": cap_remaining_after_prev_norm,
            "small_backlog_flag": small_backlog_flag
        }

    def _arrivals(self):
        # Poisson arrivals (bytes/ms) + optional periodic bursts
        arr = self.rng.poisson(self.lam_bytes_per_ms, size=4).astype(float)
        for i in range(4):
            if self.period_ms[i] > 0 and (self.t % self.period_ms[i] == 0):
                arr[i] += self.period_bytes[i]
        return arr
