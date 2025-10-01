
#!/usr/bin/env python3
"""
MAC Scheduling Test Suite (Demo)
- Focus metrics: cell throughput, Jain fairness, PRB utilization, mean & p95 latency
- Simple traffic and channel models
- Schedulers: RoundRobin, ProportionalFair, GreedyThroughput
- Two built-in scenarios:
    1) full_buffer_static
    2) mixed_traffic_fastfade
Outputs a CSV with aggregated results per (scenario, scheduler).
"""

import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import time
import math
import sys
from rl_mac_env import bytes_per_prb as bytes_per_prb, tbs_38214_bytes

# -----------------------------
# MCS efficiency model (toy)
# -----------------------------
TOY_MCS_EFF = np.array([
    0.152, 0.234, 0.377, 0.601, 0.877, 1.175, 1.476, 1.914, 2.406, 2.730,
    3.322, 3.902, 4.523, 5.115, 5.554, 6.070, 6.234, 6.5, 6.7, 6.9,
    7.0, 7.1, 7.2, 7.3, 7.35, 7.4, 7.45, 7.48, 7.5
], dtype=float)

def eff_from_mcs(mcs_idx: int) -> float:
    mcs_idx = int(max(0, min(len(TOY_MCS_EFF)-1, mcs_idx)))
    return float(TOY_MCS_EFF[mcs_idx])

def bytes_per_prb_toy(mcs_idx: int, re_per_prb: int = 12*14) -> float:
    # Approximate bytes per PRB per TTI = eff (bits/RE) * RE / 8
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
    N = len(logs[0]["served_bytes"])
    T = len(logs)
    served = np.array([l["served_bytes"] for l in logs])  # T x N
    prbs    = np.array([l["prbs"] for l in logs])         # T x N
    hol     = np.array([l["hol_ms"] for l in logs])       # T x N
    
    # Throughput
    per_ue_bits = served.sum(axis=0) * 8.0
    duration_s = (T * tti_ms) / 1000.0
    per_ue_mbps = per_ue_bits / duration_s / 1e6
    cell_tput_mbps = per_ue_mbps.sum()
    
    # Jain fairness
    jain = jain_fairness(per_ue_mbps)
    
    # PRB utilization
    used_prb = prbs.sum(axis=1).astype(float)
    if "prb_budget" in logs[0]:
        budget = np.array([l["prb_budget"] for l in logs], dtype=float)
    else:
        budget = np.full(T, prbs.sum(axis=1).max(), dtype=float)
    prb_util = np.mean(used_prb / np.maximum(1.0, budget))
    
    # Latency
    hol_flat = hol.flatten().astype(float)
    mean_hol = float(np.mean(hol_flat))
    p95_hol  = float(np.percentile(hol_flat, 95))

    # Scheduler decision runtime (seconds -> microseconds)
    decision_runtimes = np.array([float(log.get("decision_runtime_s", np.nan)) for log in logs], dtype=float)
    valid_runtime = np.isfinite(decision_runtimes)
    avg_runtime_us = float(decision_runtimes[valid_runtime].mean() * 1e6) if valid_runtime.any() else 0.0

    
    return {
        "cell_throughput_Mbps": float(cell_tput_mbps),
        "jain_fairness": float(jain),
        "prb_utilization": float(prb_util),
        "mean_latency_ms": mean_hol,
        "p95_latency_ms": p95_hol,
        "avg_decision_runtime_us": avg_runtime_us,
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
        return np.full(self.n_ue, self.rate_floor_bps / 8 / 1000, dtype=float)

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

class CombinedTraffic(TrafficModel):
    def __init__(self, parts: List[TrafficModel]):
        self.parts = parts
        self.slices = []
        offset = 0
        for p in parts:
            if hasattr(p, "n_ue"):
                n = p.n_ue
            else:
                raise ValueError("Traffic model missing n_ue")
            self.slices.append((offset, offset+n, p))
            offset += n
        self.n_ue = offset
    def arrivals_bytes(self, tti: int) -> np.ndarray:
        arr = np.zeros(self.n_ue, dtype=float)
        for (lo, hi, p) in self.slices:
            sub = p.arrivals_bytes(tti)
            arr[lo:hi] = sub
        return arr

# -----------------------------
# Channel models (MCS per UE)
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
    def decide(self, state: dict) -> np.ndarray: ...

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

def _per_tti_caps(ue):
    """
    Compute bytes/PRB and the maximum PRBs each UE can consume this TTI.
    ue[i]: {"load": bytes, "mcs": int}
    """
    N = len(ue)
    bpp = np.array([bytes_per_prb(int(ue[i]["mcs"])) for i in range(N)], dtype=float)  # bytes per PRB
    loads = np.array([float(ue[i]["load"]) for i in range(N)], dtype=float)
    # Cap = ceil(load / bpp); handle bpp==0 safely
    caps = np.ceil(loads / np.maximum(1e-9, bpp)).astype(int)
    caps = np.maximum(0, caps)
    return bpp, caps

class RoundRobinScheduler(Scheduler):
    def __init__(self, name_: str = "RoundRobin"):
        self._name = name_
    def name(self) -> str: 
        return self._name

    def decide(self, state: dict) -> np.ndarray:
        budget = int(state["prb_budget"])
        ue = state["ue"]
        N = len(ue)

        prbs = np.zeros(N, dtype=int)
        bpp, caps = _per_tti_caps(ue)

        # Active list = UEs that still need at least 1 PRB
        active = np.where(caps > 0)[0]
        if active.size == 0 or budget <= 0:
            return prbs

        i = 0
        assigned = 0
        # Round-robin across UEs with remaining demand
        while assigned < budget and active.size > 0:
            idx = active[i % active.size]
            prbs[idx] += 1
            caps[idx] -= 1
            assigned += 1

            # If UE reached its cap, remove it from rotation
            if caps[idx] == 0:
                active = np.delete(active, i % active.size)
                # keep i consistent with new active size
                if active.size == 0:
                    break
                i = i % active.size
            else:
                i += 1

        return prbs
    
class ProportionalFairScheduler(Scheduler):
    def __init__(self, n_ue: int, ewma: float = 0.9, name_: str = "ProportionalFair"):
        self._name = name_
        self.n_ue = n_ue
        self.ewma = ewma
        self.avg_rate = np.zeros(n_ue, dtype=float) + 1e-6  # avoid div-by-zero

    def name(self) -> str: 
        return self._name

    def decide(self, state: dict) -> np.ndarray:
        budget = int(state["prb_budget"])
        ue = state["ue"]
        N = len(ue)

        prbs = np.zeros(N, dtype=int)
        bpp, caps = _per_tti_caps(ue)     # bytes/PRB and demand caps

        remaining = caps.copy()
        if budget <= 0 or remaining.sum() == 0:
            return prbs

        # Greedy per-PRB PF allocation, demand-aware
        for _ in range(budget):
            # PF metric only for UEs that can still consume PRBs
            pf_metric = np.where(remaining > 0, bpp / self.avg_rate, -np.inf)
            j = int(np.argmax(pf_metric))
            if not np.isfinite(pf_metric[j]):
                break  # no eligible UE left
            prbs[j] += 1
            remaining[j] -= 1

        # Update avg_rate with instantaneous allocated capacity proxy (bytes this TTI)
        inst_bytes = prbs * bpp
        self.avg_rate = self.ewma * self.avg_rate + (1.0 - self.ewma) * (inst_bytes + 1e-9)

        return prbs

# class RoundRobinScheduler(Scheduler):
#     def __init__(self, name_: str = "RoundRobin"):
#         self._name = name_
#     def name(self) -> str: return self._name
#     def decide(self, state: dict) -> np.ndarray:
#         N = len(state["ue"])
#         budget = int(state["prb_budget"])
#         active = np.array([1 if u["load"] > 0 else 0 for u in state["ue"]])
#         active_count = max(1, active.sum())
#         base = budget // active_count
#         prbs = active * base
#         leftover = budget - prbs.sum()
#         if leftover > 0:
#             idxs = np.where(active==1)[0]
#             for i in range(leftover):
#                 prbs[idxs[i % len(idxs)]] += 1
#         return prbs

# class ProportionalFairScheduler(Scheduler):
#     def __init__(self, n_ue: int, ewma: float = 0.9, name_: str = "ProportionalFair"):
#         self._name = name_
#         self.n_ue = n_ue
#         self.ewma = ewma
#         self.avg_rate = np.zeros(n_ue, dtype=float) + 1e-6
#     def name(self) -> str: return self._name
#     def decide(self, state: dict) -> np.ndarray:
#         budget = int(state["prb_budget"])
#         ue = state["ue"]
#         N = len(ue)
#         prbs = np.zeros(N, dtype=int)
#         inst_bpp = np.array([bytes_per_prb(int(ue[i]["mcs"])) for i in range(N)])
#         active = np.array([1 if ue[i]["load"] > 0 else 0 for i in range(N)], dtype=int)
#         if active.sum() == 0:
#             return prbs
#         for _ in range(budget):
#             pf = np.where(active==1, inst_bpp / self.avg_rate, -np.inf)
#             j = int(np.argmax(pf))
#             if not np.isfinite(pf[j]):
#                 break
#             prbs[j] += 1
#         inst_bytes = prbs * inst_bpp
#         self.avg_rate = self.ewma*self.avg_rate + (1-self.ewma)*(inst_bytes + 1e-9)
#         return prbs

class GreedyThroughputScheduler(Scheduler):
    def __init__(self, name_: str = "GreedyThroughput"):
        self._name = name_
    def name(self) -> str: return self._name
    def decide(self, state: dict) -> np.ndarray:
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
from dataclasses import dataclass

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
    
    def step(self, scheduler: Scheduler) -> dict:
        N = self.cfg.n_ue
        mcs = self.channel.sample_mcs()
        arrivals = self.traffic.arrivals_bytes(self.t)
        for i in range(N):
            self.ues[i].backlog_bytes += arrivals[i]
        for i in range(N):
            if self.ues[i].backlog_bytes > 0:
                self.ues[i].hol_ms += self.cfg.tti_ms
            else:
                self.ues[i].hol_ms = 0.0
        
        prb_budget = self.cfg.available_prbs - self.cfg.control_reserve_prbs
        
        state = {
            "prb_budget": prb_budget,
            "ue": [
                {"load": self.ues[i].backlog_bytes, "mcs": int(mcs[i]), "hol_ms": self.ues[i].hol_ms}
                for i in range(N)
            ]
        }
        
        start_time = time.perf_counter()
        prbs = scheduler.decide(state)
        decision_runtime_s = time.perf_counter() - start_time
        prbs = _sanitize(np.asarray(prbs, dtype=int), prb_budget)
        
        tbs_bytes = np.zeros(N, dtype=float)
        served = np.zeros(N, dtype=float)
        for i in range(N):
            tbs_bytes[i] = tbs_38214_bytes(int(mcs[i]), int(prbs[i]))
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
            "decision_runtime_s": float(decision_runtime_s),
        }
        
        self.channel.step()
        self.t += 1
        return log
    
    def run(self, scheduler: Scheduler) -> list:
        self.reset()
        logs = []
        for _ in range(self.cfg.duration_tti):
            logs.append(self.step(scheduler))
        return logs
    
    def reset(self):
        self.t = 0
        self.ues = [UEState() for _ in range(self.cfg.n_ue)]

# -----------------------------
# Scenarios
# -----------------------------
def build_scenario(scenario_name: str, rng: np.random.Generator):
    scenario_name = scenario_name.strip().lower()
    if scenario_name == "full_buffer_static":
        cfg = SimulatorConfig(n_ue=4, duration_tti=2000, available_prbs=273)
        traffic = FullBufferTraffic(n_ue=4)
        channel = StaticMCS([5, 10, 15, 20])

    elif scenario_name == "full_buffer_fastfade":
        cfg = SimulatorConfig(n_ue=4, duration_tti=2000, available_prbs=273)
        traffic = FullBufferTraffic(n_ue=4)   # saturated buffers
        # per-TTI fast fading around different means to create multi-user diversity
        channel = FastFadingMCS(
            n_ue=4,
            mean=[5, 10, 15, 25],  # heterogeneous averages (poor → good)
            spread=6,              # +/-6 MCS steps per TTI
            rng=rng
        )

    elif scenario_name == "mixed_traffic_lowfade":
        cfg = SimulatorConfig(n_ue=4, duration_tti=2000, available_prbs=273)
        aver_rate_bps = rng.integers(10, 50, size=4) * 1e6
        traffic = CombinedTraffic([
            PoissonTraffic(n_ue=1, mean_bps=[aver_rate_bps[0]], rng=rng),
            PoissonTraffic(n_ue=1, mean_bps=[aver_rate_bps[1]], rng=rng),
            PoissonTraffic(n_ue=1, mean_bps=[aver_rate_bps[2]], rng=rng),
            PoissonTraffic(n_ue=1, mean_bps=[aver_rate_bps[3]], rng=rng),
        ])
        channel = FastFadingMCS(
            n_ue=4,
            mean=rng.integers(5, 25, size=4),
            spread=2,
            rng=rng,
        )

    elif scenario_name == "mixed_traffic_fastfade":
        cfg = SimulatorConfig(n_ue=4, duration_tti=1500, available_prbs=273)
        traffic = CombinedTraffic([
            PoissonTraffic(n_ue=1, mean_bps=[40e6], rng=rng),
            PeriodicTraffic(n_ue=1, period_ms=[20], pkt_bytes=[80]),
            PoissonTraffic(n_ue=1, mean_bps=[8e6], rng=rng),
            PoissonTraffic(n_ue=1, mean_bps=[15e6], rng=rng)
        ])
        channel = FastFadingMCS(n_ue=4, mean=[14, 18, 10, 22], spread=4, rng=rng)
    else:
        raise ValueError(f"Unknown scenario '{scenario_name}'.")
    return cfg, traffic, channel

def available_scenarios():
    return ["full_buffer_static", "full_buffer_fastfade","mixed_traffic_fastfade"]

def available_schedulers(n_ue: int):
    return [
        RoundRobinScheduler(),
        ProportionalFairScheduler(n_ue=n_ue),
        GreedyThroughputScheduler()
    ]

# -----------------------------
# Main CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="MAC Scheduling Test Suite (Demo)")
    parser.add_argument("--scenario", type=str, default=None, help="Scenario to run. Choices: " + ",".join(available_scenarios()))
    parser.add_argument("--all", action="store_true", help="Run all built-in scenarios.")
    parser.add_argument("--out", type=str, default="mac_scheduler_eval_results.csv", help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    scenarios = available_scenarios() if args.all else [args.scenario or "full_buffer_static"]
    rows = []

    for sc_name in scenarios:
        cfg, traffic, channel = build_scenario(sc_name, rng)
        sim = Simulator(cfg, traffic, channel, rng)
        scheds = available_schedulers(cfg.n_ue)
        for sch in scheds:
            logs = sim.run(sch)
            metrics = compute_metrics(logs, tti_ms=cfg.tti_ms)
            print(f"[{sc_name} | {sch.name()}] cell_tput={metrics['cell_throughput_Mbps']:.2f} Mbps, "
                  f"Jain={metrics['jain_fairness']:.3f}, util={metrics['prb_utilization']:.3f}, "
                  f"mean_lat={metrics['mean_latency_ms']:.2f} ms, p95={metrics['p95_latency_ms']:.2f} ms, "
                  f"runtime={metrics['avg_decision_runtime_us']:.1f} us")
            rows.append({
                "scenario": sc_name,
                "scheduler": sch.name(),
                **{k:v for k,v in metrics.items() if k not in ["per_ue_throughput_Mbps"]}
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nSaved results to: {args.out}")

if __name__ == "__main__":
    main()

