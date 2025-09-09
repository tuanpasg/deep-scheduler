#!/usr/bin/env python3
import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from rl_mac_env import MACSchedulerEnv

def make_env(use_prev_prbs, profile, fading, seed, alpha, beta, gamma, rho, save_dir):
    def _thunk():
        env = MACSchedulerEnv(
            use_prev_prbs=use_prev_prbs,
            traffic_profile=profile,
            fading_profile=fading,
            duration_tti=2000,
            prb_budget=273,
            alpha_throughput=alpha,
            beta_fairness=beta,
            gamma_latency=gamma,
            fairness_ema_rho=rho,
            seed=seed
        )
        # monitor_path = os.path.join(save_dir, f"monitor_seed{seed}.csv")
        monitor_path = os.path.join(save_dir, f"monitor.csv")
        env = Monitor(env, filename=monitor_path, info_keywords=('jain','cell_tput_Mb','mean_hol_ms'))
        return env
    return _thunk

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--use_prev_prbs", type=int, default=0)
    p.add_argument("--profile", type=str, default="mixed", choices=["mixed","poisson","full_buffer"])
    p.add_argument("--fading", type=str, default="fast", choices=["fast","slow","static"])
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--rho", type=float, default=0.9)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_steps", type=int, default=2048)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--save_dir", type=str, default="runs/ppo_mac_fair")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    logger = configure(args.save_dir, ["stdout","csv","tensorboard"])

    env = DummyVecEnv([make_env(bool(args.use_prev_prbs), args.profile, args.fading,
                                args.seed, args.alpha, args.beta, args.gamma, args.rho, args.save_dir)])
    env = VecMonitor(env)

    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    # policy_kwargs = dict(net_arch=[128,128])
    policy_kwargs = dict(net_arch=[dict(pi=[256,256], vf=[256,256])])
    model = PPO("MlpPolicy", env,
                learning_rate=args.lr,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                ent_coef=args.ent_coef,
                vf_coef=1,
                n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, vf_coef=0.5,
                policy_kwargs=policy_kwargs, verbose=1, seed=args.seed)
    model.set_logger(logger)

    model.learn(total_timesteps=args.timesteps)
    model_path = os.path.join(args.save_dir, "ppo_mac_scheduler.zip")
    model.save(model_path)
    print(f"Saved model to: {model_path}")

if __name__ == "__main__":
    main()
