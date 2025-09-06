# Ready-to-run MAC scheduling test suite (no HARQ, 4 key metrics)
# This cell writes a reusable Python module and executes two demo scenarios.
# It produces a CSV of results and displays a summary table.

from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional
import numpy as np
import pandas as pd
import math
import time
import json
import os

# -----------------------------
# Helper: MCS spectral efficiency (toy table)
# -----------------------------
# Simplified "bits per RE" spectral efficiency for demo (not 3GPP-accurate).
# Range roughly from ~0.15 to ~7.0 b/RE across indices.
TOY_MCS_EFF = np.array([
    0.152, 0.234, 0.377, 0.601, 0.877, 1.175, 1.476, 1.914, 2.406, 2.730,
    3.322, 3.902, 4.523, 5.115, 5.554, 6.070, 6.234, 6.5, 6.7, 6.9,
    7.0, 7.1, 7.2, 7.3, 7.35, 7.4, 7.45, 7.48, 7.5
], dtype=float)

def eff_from_mcs(mcs_idx: int) -> float:
    mcs_idx = int(max(0, min(len(TOY_MCS_EFF)-1, mcs_idx)))
    return float(TOY_MCS_EFF[mcs_idx])

def bytes_per_prb(mcs_idx: int, re_per_prb: int = 12*14) -> float:
    # bytes per PRB per TTI (approx) = eff (bits/RE) * RE/PRB / 8
    return eff_from_mcs(mcs_idx) * re_per_prb / 8.0

# -----------------------------
# Metrics helpers
# -----------------------------
def jain_fairness(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.all(values == 0): 
        return 1.0
    num = (values.sum())**2
    den = len(values) * (values**2).sum() + 1e-12
    return float(num / den)

def compute_metrics(logs: List[Dict], tti_ms: float = 1.0) -> Dict[str, float]:
    # logs: list of per-TTI dicts with keys: served_bytes (N,), prbs (N,), backlog_bytes (N,), hol_ms (N,)
    N = len(logs[0]["served_bytes"])
    T = len(logs)
    served = np.array([l["served_bytes"] for l in logs])  # shape T x N
    prbs    = np.array([l["prbs"] for l in logs])         # shape T x N
    hol     = np.array([l["hol_ms"] for l in logs])       # shape T x N
    
    # Throughput
    per_ue_bits = served.sum(axis=0) * 8.0
    duration_s = (T * tti_ms) / 1000.0
    per_ue_mbps = per_ue_bits / duration_s / 1e6
    cell_tput_mbps = per_ue_mbps.sum()
    
    # Jain fairness
    jain = jain_fairness(per_ue_mbps)
    
    # PRB utilization
    used_prb = prbs.sum(axis=1).astype(float)
    # available_prbs may vary per TTI; we log it per TTI for accuracy
    if "prb_budget" in logs[0]:
        budget = np.array([l["prb_budget"] for l in logs], dtype=float)
    else:
        # assume constant budget from prbs peaks
        budget = np.full(T, prbs.sum(axis=1).max(), dtype=float)
    prb_util = np.mean(used_prb / np.maximum(1.0, budget))
    
    # Latency (HOL delay) — mean and p95 across time & UEs
    hol_flat = hol.flatten().astype(float)
    mean_hol = float(np.mean(hol_flat))
    p95_hol  = float(np.percentile(hol_flat, 95))
    
    return {
        "cell_throughput_Mbps": float(cell_tput_mbps),
        "jain_fairness": float(jain),
        "prb_utilization": float(prb_util),
        "mean_latency_ms": mean_hol,
        "p95_latency_ms": p95_hol,
        "per_ue_throughput_Mbps": per_ue_mbps.tolist(),
    }

# -----------------------------
# Traffic models
# -----------------------------
class TrafficModel:
    def arrivals_bytes(self, tti: int) -> np.ndarray:
        raise NotImplementedError

class FullBufferTraffic(TrafficModel):
    def __init__(self, n_ue: int, rate_floor_bps: float = 1e12):
        self.n_ue = n_ue
        self.rate_floor_bps = rate_floor_bps
    def arrivals_bytes(self, tti: int) -> np.ndarray:
        # Infinite backlog: simulate as very large arrival to keep queues non-empty.
        # Use a large but finite value to avoid overflow.
        return np.full(self.n_ue, self.rate_floor_bps / 8 / 1000, dtype=float)  # bytes per ms

class PoissonTraffic(TrafficModel):
    def __init__(self, n_ue: int, mean_bps: List[float], rng: np.random.Generator):
        self.n_ue = n_ue
        self.mean_bps = np.array(mean_bps, dtype=float)
        self.rng = rng
    def arrivals_bytes(self, tti: int) -> np.ndarray:
        lam_bytes_per_ms = self.mean_bps / 8.0 / 1000.0
        return self.rng.poisson(lam=lam_bytes_per_ms).astype(float)

class PeriodicTraffic(TrafficModel):
    def __init__(self, n_ue: int, period_ms: List[int], pkt_bytes: List[int]):
        self.n_ue = n_ue
        self.period_ms = np.array(period_ms, dtype=int)
        self.pkt_bytes  = np.array(pkt_bytes, dtype=int)
    def arrivals_bytes(self, tti: int) -> np.ndarray:
        arr = np.zeros(self.n_ue, dtype=float)
        for i in range(self.n_ue):
            if self.period_ms[i] > 0 and (tti % self.period_ms[i] == 0):
                arr[i] = self.pkt_bytes[i]
        return arr

# -----------------------------
# Channel models (MCS index per UE)
# -----------------------------
class ChannelModel:
    def sample_mcs(self) -> np.ndarray:
        raise NotImplementedError
    def step(self):
        pass

class StaticMCS(ChannelModel):
    def __init__(self, mcs_list: List[int]):
        self.mcs = np.array(mcs_list, dtype=int)
    def sample_mcs(self) -> np.ndarray:
        return self.mcs.copy()

class RandomWalkMCS(ChannelModel):
    def __init__(self, n_ue: int, start: List[int], step_prob: float, rng: np.random.Generator,
                 min_mcs: int = 0, max_mcs: int = 28):
        self.n_ue = n_ue
        self.mcs = np.array(start, dtype=int)
        self.step_prob = step_prob
        self.rng = rng
        self.min_mcs = min_mcs
        self.max_mcs = max_mcs
    def sample_mcs(self) -> np.ndarray:
        return self.mcs.copy()
    def step(self):
        # each TTI, each UE moves -1,0,+1 with some probability
        delta = self.rng.choice([-1, 0, 1], size=self.n_ue, p=[self.step_prob/2, 1-self.step_prob, self.step_prob/2])
        self.mcs = np.clip(self.mcs + delta, self.min_mcs, self.max_mcs)

class FastFadingMCS(ChannelModel):
    def __init__(self, n_ue: int, mean: List[int], spread: int, rng: np.random.Generator,
                 min_mcs: int = 0, max_mcs: int = 28):
        self.n_ue = n_ue
        self.mean = np.array(mean, dtype=int)
        self.spread = spread
        self.rng = rng
        self.min_mcs = min_mcs
        self.max_mcs = max_mcs
    def sample_mcs(self) -> np.ndarray:
        m = self.mean + self.rng.integers(-self.spread, self.spread+1, size=self.n_ue)
        return np.clip(m, self.min_mcs, self.max_mcs)

# -----------------------------
# Schedulers
# -----------------------------
class Scheduler:
    def name(self) -> str: ...
    def decide(self, state: Dict) -> np.ndarray: ...

def _sanitize(prbs: np.ndarray, budget: int) -> np.ndarray:
    prbs = np.maximum(0, np.round(prbs).astype(int))
    s = prbs.sum()
    if s <= budget:
        return prbs
    if s == 0:
        return np.zeros_like(prbs)
    scaled = np.floor(prbs * (budget / s)).astype(int)
    rem = budget - scaled.sum()
    if rem > 0:
        order = np.argsort(-(prbs - scaled))
        for i in order[:rem]:
            scaled[i] += 1
    return scaled

class RoundRobinScheduler(Scheduler):
    def __init__(self, name_: str = "RoundRobin"):
        self._name = name_
    def name(self) -> str: return self._name
    def decide(self, state: Dict) -> np.ndarray:
        N = len(state["ue"])
        budget = int(state["prb_budget"])
        # equal split for active queues
        active = np.array([1 if u["load"] > 0 else 0 for u in state["ue"]])
        active_count = max(1, active.sum())
        base = budget // active_count
        prbs = active * base
        # distribute leftover one by one among active UEs
        leftover = budget - prbs.sum()
        if leftover > 0:
            idxs = np.where(active==1)[0]
            for i in range(leftover):
                prbs[idxs[i % len(idxs)]] += 1
        return prbs

class ProportionalFairScheduler(Scheduler):
    def __init__(self, n_ue: int, ewma: float = 0.9, name_: str = "ProportionalFair"):
        self._name = name_
        self.n_ue = n_ue
        self.ewma = ewma
        self.avg_rate = np.zeros(n_ue, dtype=float) + 1e-6
    def name(self) -> str: return self._name
    def decide(self, state: Dict) -> np.ndarray:
        budget = int(state["prb_budget"])
        ue = state["ue"]
        N = len(ue)
        # Greedy PRB-by-PRB to UE with highest PF metric (inst_rate / avg_rate)
        prbs = np.zeros(N, dtype=int)
        # instantaneous bytes per PRB
        inst_bpp = np.array([bytes_per_prb(int(ue[i]["mcs"])) for i in range(N)])
        # Avoid assigning to empty buffers
        active = np.array([1 if ue[i]["load"] > 0 else 0 for i in range(N)], dtype=int)
        if active.sum() == 0:
            return prbs
        for _ in range(budget):
            # PF metric
            pf = np.where(active==1, inst_bpp / self.avg_rate, -np.inf)
            j = int(np.argmax(pf))
            if not np.isfinite(pf[j]):
                break
            prbs[j] += 1
        # Update avg_rate estimate (caller will pass served later; we use proxy)
        # Here we update with instantaneous allocated capacity proxy:
        inst_bytes = prbs * inst_bpp
        self.avg_rate = self.ewma*self.avg_rate + (1-self.ewma)*(inst_bytes + 1e-9)
        return prbs

class GreedyThroughputScheduler(Scheduler):
    """Allocates PRBs in proportion to backlog * instantaneous capacity (bytes per PRB)."""
    def __init__(self, name_: str = "GreedyThroughput"):
        self._name = name_
    def name(self) -> str: return self._name
    def decide(self, state: Dict) -> np.ndarray:
        budget = int(state["prb_budget"])
        ue = state["ue"]
        N = len(ue)
        weights = np.array([ue[i]["load"] * bytes_per_prb(int(ue[i]["mcs"])) for i in range(N)], dtype=float)
        if np.all(weights <= 0):
            return np.zeros(N, dtype=int)
        weights = np.maximum(0.0, weights)
        if weights.sum() == 0:
            return np.zeros(N, dtype=int)
        fracs = weights / weights.sum()
        raw = fracs * budget
        prbs = np.floor(raw).astype(int)
        rem = budget - prbs.sum()
        if rem > 0:
            remainders = raw - np.floor(raw)
            order = np.argsort(-remainders)
            for k in order[:rem]:
                prbs[k] += 1
        return prbs

# -----------------------------
# Simulator
# -----------------------------
@dataclass
class SimulatorConfig:
    n_ue: int = 4
    tti_ms: float = 1.0
    duration_tti: int = 1000
    available_prbs: int = 273
    control_reserve_prbs: int = 0

@dataclass
class UEState:
    backlog_bytes: float = 0.0
    hol_ms: float = 0.0

class Simulator:
    def __init__(self, cfg: SimulatorConfig, traffic: TrafficModel, channel: ChannelModel, rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.traffic = traffic
        self.channel = channel
        self.rng = rng or np.random.default_rng(1234)
        self.ues = [UEState() for _ in range(cfg.n_ue)]
    
    def step(self, scheduler: Scheduler) -> Dict:
        N = self.cfg.n_ue
        # Channel sample
        mcs = self.channel.sample_mcs()
        # Arrivals
        arrivals = self.traffic.arrivals_bytes(self.t)
        for i in range(N):
            self.ues[i].backlog_bytes += arrivals[i]
        # Update HOL
        for i in range(N):
            if self.ues[i].backlog_bytes > 0:
                self.ues[i].hol_ms += self.cfg.tti_ms
            else:
                self.ues[i].hol_ms = 0.0
        
        # Budget
        prb_budget = self.cfg.available_prbs - self.cfg.control_reserve_prbs
        
        # Build state for scheduler
        state = {
            "prb_budget": prb_budget,
            "ue": [
                {"load": self.ues[i].backlog_bytes, "mcs": int(mcs[i]), "hol_ms": self.ues[i].hol_ms}
                for i in range(N)
            ]
        }
        
        # Decide allocation
        prbs = scheduler.decide(state)
        prbs = _sanitize(np.asarray(prbs, dtype=int), prb_budget)
        
        # Serve bytes (no HARQ): served = min(backlog, tbs_bytes)
        bpp = np.array([bytes_per_prb(int(m)) for m in mcs])  # bytes per PRB
        tbs_bytes = (prbs * bpp).astype(float)
        served = np.zeros(N, dtype=float)
        for i in range(N):
            served[i] = min(self.ues[i].backlog_bytes, tbs_bytes[i])
            self.ues[i].backlog_bytes -= served[i]
            if self.ues[i].backlog_bytes == 0:
                self.ues[i].hol_ms = 0.0
        
        log = {
            "prb_budget": prb_budget,
            "mcs": mcs.tolist(),
            "arrivals": arrivals.tolist(),
            "prbs": prbs.tolist(),
            "tbs_bytes": tbs_bytes.tolist(),
            "served_bytes": served.tolist(),
            "backlog_bytes": [u.backlog_bytes for u in self.ues],
            "hol_ms": [u.hol_ms for u in self.ues],
        }
        
        # advance channel (if stateful)
        self.channel.step()
        self.t += 1
        return log
    
    def run(self, scheduler: Scheduler) -> List[Dict]:
        self.reset()
        logs = []
        for _ in range(self.cfg.duration_tti):
            logs.append(self.step(scheduler))
        return logs
    
    def reset(self):
        self.t = 0
        self.ues = [UEState() for _ in range(self.cfg.n_ue)]

# -----------------------------
# Demo scenarios & execution
# -----------------------------
def run_demo():
    rng = np.random.default_rng(42)
    
    results = []
    rows = []
    
    # Scenario 1: Full-buffer, static channel
    cfg1 = SimulatorConfig(n_ue=4, duration_tti=1000, available_prbs=273)
    traffic1 = FullBufferTraffic(n_ue=4)
    channel1 = StaticMCS([5, 10, 15, 20])  # low..high
    
    sim1 = Simulator(cfg1, traffic1, channel1, rng)
    schedulers1 = [
        RoundRobinScheduler(),
        ProportionalFairScheduler(n_ue=4),
        GreedyThroughputScheduler()
    ]
    for sch in schedulers1:
        logs = sim1.run(sch)
        metrics = compute_metrics(logs, tti_ms=cfg1.tti_ms)
        rows.append({
            "scenario": "full_buffer_static",
            "scheduler": sch.name(),
            **{k:v for k,v in metrics.items() if k not in ["per_ue_throughput_Mbps"]}
        })
    
    # Scenario 2: Mixed traffic, fast fading
    cfg2 = SimulatorConfig(n_ue=4, duration_tti=1500, available_prbs=273)
    traffic2 = CombinedTraffic([
        PoissonTraffic(n_ue=1, mean_bps=[40e6], rng=rng),   # UE0 eMBB ~ 40 Mbps
        PeriodicTraffic(n_ue=1, period_ms=[20], pkt_bytes=[80]),  # UE1 VoIP-like
        PoissonTraffic(n_ue=1, mean_bps=[8e6], rng=rng),    # UE2 BE ~ 8 Mbps
        PoissonTraffic(n_ue=1, mean_bps=[15e6], rng=rng)    # UE3 BE ~ 15 Mbps
    ])
    channel2 = FastFadingMCS(n_ue=4, mean=[14, 18, 10, 22], spread=4, rng=rng)
    sim2 = Simulator(cfg2, traffic2, channel2, rng)
    schedulers2 = [
        RoundRobinScheduler(),
        ProportionalFairScheduler(n_ue=4),
        GreedyThroughputScheduler()
    ]
    for sch in schedulers2:
        logs = sim2.run(sch)
        metrics = compute_metrics(logs, tti_ms=cfg2.tti_ms)
        rows.append({
            "scenario": "mixed_traffic_fastfade",
            "scheduler": sch.name(),
            **{k:v for k,v in metrics.items() if k not in ["per_ue_throughput_Mbps"]}
        })
    
    df = pd.DataFrame(rows)
    out_csv = "/mnt/data/mac_scheduler_eval_results.csv"
    df.to_csv(out_csv, index=False)
    return df, out_csv

# Helper to combine multiple per-UE traffic models into a single model
class CombinedTraffic(TrafficModel):
    def __init__(self, parts: List[TrafficModel]):
        self.parts = parts
        # Validate that sum of n_ue == total UE count
        self.slices = []
        offset = 0
        for p in parts:
            if isinstance(p, FullBufferTraffic):
                n = p.n_ue
            elif isinstance(p, PoissonTraffic):
                n = p.n_ue
            elif isinstance(p, PeriodicTraffic):
                n = p.n_ue
            else:
                raise ValueError("Unknown traffic model in CombinedTraffic")
            self.slices.append((offset, offset+n, p))
            offset += n
        self.n_ue = offset
    def arrivals_bytes(self, tti: int) -> np.ndarray:
        arr = np.zeros(self.n_ue, dtype=float)
        for (lo, hi, p) in self.slices:
            sub = p.arrivals_bytes(tti)
            arr[lo:hi] = sub
        return arr

# Execute demos
df_results, csv_path = run_demo()

# Display result table to the user
from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("MAC Scheduler Evaluation Results", df_results)

csv_path