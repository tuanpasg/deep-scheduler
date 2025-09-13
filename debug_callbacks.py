# debug_callbacks.py
import numpy as np
import os
import csv
from stable_baselines3.common.callbacks import BaseCallback

class ValueDebugCallback(BaseCallback):
    """
    At the end of each rollout (i.e. before policy/value update), examine:
    - rollout_buffer.values (v_pred)
    - rollout_buffer.returns (y)
    - rollout_buffer.advantages (adv)
    - rollout_buffer.observations (obs) (shape depends on env)
    - rewards seen in this rollout (must reconstruct from returns+discount)
    Logs:
    - explained_variance, var_targets, var_errors, mse(v_pred, y)
    - advantage mean/std and fraction of near-zero advantages
    - observation mean/std per feature (first few features only)
    - reward mean/std per-step
    - writes a CSV row for each rollout
    """
    def __init__(self, out_dir="debug_rollouts", verbose=1, adv_zero_eps=1e-6):
        super().__init__(verbose)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.csv_path = os.path.join(self.out_dir, "rollout_debug.csv")
        # CSV header
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "update", "explained_var", "var_targets", "var_error",
                    "mse", "adv_mean", "adv_std", "adv_frac_near_zero",
                    "rew_mean", "rew_std", "obs_mean_sample0", "obs_std_sample0",
                    "num_steps"
                ])
        self.adv_zero_eps = adv_zero_eps

    def _on_rollout_end(self) -> None:
        rb = self.model.rollout_buffer

        # Values (predictions) and targets
        # SB3 stores them as numpy arrays on CPU
        v_pred = np.array(rb.values).astype(float).ravel()
        returns = np.array(rb.returns).astype(float).ravel()
        adv = np.array(rb.advantages).astype(float).ravel()

        # Basic stats
        n = len(returns)
        if n == 0:
            return

        # explained variance: 1 - Var(y - v) / Var(y)
        var_targets = float(np.var(returns))
        var_error = float(np.var(returns - v_pred))
        explained_var = 1.0 - (var_error / (var_targets + 1e-12))

        mse = float(np.mean((returns - v_pred)**2))
        adv_mean = float(np.mean(adv))
        adv_std = float(np.std(adv))
        adv_frac_near_zero = float(np.mean(np.abs(adv) < self.adv_zero_eps))

        # Reconstruct per-step rewards approx by undoing discounted returns:
        # This is approximate unless you saved raw rewards; use differences of returns if gamma known
        # But rollout_buffer doesn't expose raw rewards easily; instead compute per-step reward estimate r_t = returns_t - gamma * returns_{t+1}
        # We assume buffer is organized correctly; if not, skip
        try:
            # assume contiguous episodes inside buffer — this is heuristic but useful for trend
            gamma = getattr(self.model, "gamma", 0.99)
            rew_est = np.empty_like(returns)
            # we approximate r_t = ret_t - gamma * ret_{t+1} (last step uses ret itself)
            rew_est[:-1] = returns[:-1] - gamma * returns[1:]
            rew_est[-1] = returns[-1]
            rew_mean = float(np.mean(rew_est))
            rew_std = float(np.std(rew_est))
        except Exception:
            rew_mean = float(np.nan)
            rew_std = float(np.nan)

        # Observation stats: sample first obs feature's mean/std across buffer for quick check
        try:
            obs = np.array(rb.observations)  # may be shape (n_steps, obs_dim) or nested
            # if obs has extra dims, flatten per-step to 2D
            if obs.ndim > 2:
                obs = obs.reshape(obs.shape[0], -1)
            obs_mean0 = float(np.mean(obs[:, 0]))
            obs_std0 = float(np.std(obs[:, 0]))
        except Exception:
            obs_mean0 = float(np.nan)
            obs_std0 = float(np.nan)

        # Log to SB3 tensorboard logger
        self.logger.record("debug/explained_variance", explained_var)
        self.logger.record("debug/var_targets", var_targets)
        self.logger.record("debug/var_error", var_error)
        self.logger.record("debug/mse", mse)
        self.logger.record("debug/adv_mean", adv_mean)
        self.logger.record("debug/adv_std", adv_std)
        self.logger.record("debug/adv_frac_near_zero", adv_frac_near_zero)
        self.logger.record("debug/rew_mean", rew_mean)
        self.logger.record("debug/rew_std", rew_std)
        self.logger.record("debug/obs0_mean", obs_mean0)
        self.logger.record("debug/obs0_std", obs_std0)
        # flush logger so TB sees immediate values
        self.logger.dump(self.num_timesteps)

        # Print brief summary
        if self.verbose > 0:
            print(f"[Rollout {self.num_timesteps}] EV={explained_var:.4g}, var_y={var_targets:.4g}, var_err={var_error:.4g}, mse={mse:.4g}, adv_std={adv_std:.4g}, adv_frac0={adv_frac_near_zero:.3g}, rew_mean={rew_mean:.4g}")

        # Append to CSV for offline plotting
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                int(self.num_timesteps), float(explained_var), float(var_targets), float(var_error),
                float(mse), float(adv_mean), float(adv_std), float(adv_frac_near_zero),
                float(rew_mean), float(rew_std), float(obs_mean0), float(obs_std0),
                int(n)
            ])
