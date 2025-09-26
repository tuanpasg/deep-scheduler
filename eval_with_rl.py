#!/usr/bin/env python3
# eval_with_rl.py
# Evaluate a trained PPO policy (from train_rl_scheduler.py) inside mac_test_suite Simulator.
# Saves a CSV similar to mac_test_suite output for easy comparison.

import argparse
import numpy as np
from stable_baselines3 import PPO
from mac_test_suite import Simulator, SimulatorConfig, build_scenario  # mac_test_suite API
from rl_mac_env import MACSchedulerEnv, project_scores_to_prbs  # reuse env helpers
import pandas as pd
import time
import os

# === RL Scheduler wrapper that matches mac_test_suite.Scheduler.decide API ===
class RLSchedulerWrapper:
    def __init__(self, model_path, training_mode=True, env_kwargs=None):
        # load SB3 model
        self.model = PPO.load(model_path)
        # create a helper MACSchedulerEnv instance (for obs preprocessing and cap computations)
        env_kwargs = env_kwargs or {}
        self._env_helper = MACSchedulerEnv(**env_kwargs)
        # don't call reset() (we'll manually set fields)
        # preserve training_mode decision behavior for project_scores_to_prbs call
        self.training_mode = training_mode

    def name(self):
        return "PPO_RL"

    def decide(self, state: dict):
        """
        state from mac_test_suite.Simulator.step():
          {"prb_budget": int, "ue": [{"load":bytes, "mcs":int, "hol_ms":...}, ...]}
        We will:
          - set helper env's internal fields (backlog/_curr_mcs/prev_prbs/active_mask)
          - call helper._get_obs() to get normalized obs
          - pass obs to policy -> action scores
          - call project_scores_to_prbs(...) for consistent mapping
        """
        # set helper state (so its compute_backlog_and_cap_features works)
        ue = state["ue"]
        N = len(ue)
        # backlog in bytes
        backlog = np.array([float(ue[i]["load"]) for i in range(N)], dtype=float)
        mcs = np.array([int(ue[i]["mcs"]) for i in range(N)], dtype=int)

        # configure helper env internals to reflect this simulator state
        self._env_helper.backlog = backlog.copy()
        self._env_helper.prev_prbs = np.zeros(4, dtype=int)  # could be improved if prev known
        self._env_helper._curr_mcs = mcs.copy()
        self._env_helper.max_prb = int(state.get("prb_budget", self._env_helper.max_prb))
        # ensure active mask aligned with backlog
        self._env_helper.active_mask = (backlog > 0).astype(int)

        # Build observation exactly as MACSchedulerEnv._get_obs does
        obs = self._env_helper._get_obs().astype(np.float32)
        # SB3 expects shape (n_envs, obs_dim) for predict? we can pass 1D and set deterministic=True
        action, _ = self.model.predict(obs, deterministic=True)
        # action is e.g. np.array([s0,...,s3]) in [0,1]; ensure shape
        action = np.clip(np.asarray(action, dtype=float), 0.0, 1.0).ravel()

        # Convert action -> prbs using the same function training used
        prbs_out, prbs_pre, wasted_prbs, invalid_allocated_prbs = project_scores_to_prbs(
            action, int(state["prb_budget"]),
            self._env_helper.backlog,
            self._env_helper._curr_mcs,
            self._env_helper.active_mask,
            n_symb=self._env_helper.n_symb,
            overhead=self._env_helper.overhead,
            training_mode=self.training_mode
        )

        # Return prbs_out (list/ndarray of ints) — mac_test_suite will sanitize again
        return prbs_out.astype(int)

# === runner ===
def run_once_for_scheduler(cfg, traffic, channel, scheduler, duration_tti=1000, rng=None):
    sim = Simulator(cfg, traffic, channel, rng=rng)
    logs = sim.run(scheduler)
    from mac_test_suite import compute_metrics
    metrics = compute_metrics(logs, tti_ms=cfg.tti_ms)
    return metrics

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to PPO .zip saved model")
    p.add_argument("--out", default="mac_results_with_rl.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs", type=int, default=3, help="Number of repeats per scenario")
    p.add_argument("--training_mode", type=int, default=1, help="Use training_mode when mapping scores->prbs")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    scenarios = ["full_buffer_static", "full_buffer_fastfade", "mixed_traffic_fastfade"]

    rows = []
    for sc in scenarios:
        # build scenario
        cfg, traffic, channel = build_scenario(sc, rng)
        # baseline schedulers
        from mac_test_suite import available_schedulers
        baselines = available_schedulers(cfg.n_ue)
        # RL wrapper
        rl = RLSchedulerWrapper(args.model, training_mode=bool(args.training_mode),
                                env_kwargs={"tti_ms": cfg.tti_ms, "duration_tti": cfg.duration_tti, "prb_budget": cfg.available_prbs})
        scheds = baselines + [rl]

        for sch in scheds:
            # run multiple times (difference via different seed offsets)
            agg = []
            for r in range(args.runs):
                # micro-seed for each run
                run_rng = np.random.default_rng(args.seed + r + 123)
                sim = Simulator(cfg, traffic, channel, rng=run_rng)
                logs = sim.run(sch)
                from mac_test_suite import compute_metrics
                metrics = compute_metrics(logs, tti_ms=cfg.tti_ms)
                agg.append(metrics)
                print(f"[{sc} | {sch.name()} | run {r}] tput={metrics['cell_throughput_Mbps']:.2f} Mbps, jain={metrics['jain_fairness']:.3f}")

            # average numeric metrics across repeats
            # pick keys to save (same as mac_test_suite)
            avg = {}
            keys = ["cell_throughput_Mbps", "jain_fairness", "prb_utilization", "mean_latency_ms", "p95_latency_ms", "avg_decision_runtime_us"]
            for k in keys:
                avg[k] = float(np.mean([a[k] for a in agg]))
            row = {"scenario": sc, "scheduler": sch.name(), **avg}
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Saved results to: {args.out}")

if __name__ == "__main__":
    main()
