#!/usr/bin/env python3
"""
export_actor_onnx.py

Export a Stable-Baselines3 PPO policy's actor to ONNX for C/C++ inference (e.g., via ONNX Runtime C API).
Also runs a quick parity check between PyTorch and ONNXRuntime.

Usage:
  python export_actor_onnx.py \
      --model-path runs/ppo_mac_fair/ppo_mac_scheduler.zip \
      --onnx-out ppo_actor.onnx \
      --obs-dim 16 \
      --act-dim 4

Notes:
- Assumes an MLP policy (Box action space). If your policy is discrete, set --discrete and --act-dim accordingly.
- The wrapper reproduces SB3's action squashing and rescaling, so exported ONNX outputs are *deployment-ready actions*.
"""

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Optional: load onnx/onnxruntime only when asked to verify to keep imports light
def _lazy_import_onnx():
    import onnx  # noqa: F401
    import onnxruntime as ort
    return ort

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True, help="Path to SB3 .zip model file (PPO.load)")
    p.add_argument("--onnx-out", type=str, default="ppo_actor.onnx", help="Output ONNX file path")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version (>= 12 recommended)")
    p.add_argument("--obs-dim", type=int, required=True, help="Observation dimension (features per batch item)")
    p.add_argument("--act-dim", type=int, required=True, help="Action dimension")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for export")
    p.add_argument("--discrete", action="store_true", help="If set, export logits for a discrete action policy")
    p.add_argument("--no-verify", action="store_true", help="Skip ONNXRuntime parity check")
    p.add_argument("--batch", type=int, default=1, help="Dummy batch size for tracing/export")
    p.add_argument("--metadata-out", type=str, default="", help="If set, write a JSON with basic metadata beside the ONNX")
    return p.parse_args()

def load_sb3_ppo(model_path: str, device: str):
    try:
        from stable_baselines3 import PPO
    except Exception as e:
        print("ERROR: stable-baselines3 is required to load the model. pip install stable-baselines3", file=sys.stderr)
        raise
    model = PPO.load(model_path, device=device)
    model.policy.to(device)
    model.policy.eval()
    return model

class ActorExportContinuous(nn.Module):
    """
    Export wrapper for continuous-action PPO (SB3).

    Reimplements SB3's forward path to produce *scaled actions* in the environment's action space.
    This matches the values returned by `model.predict(obs, deterministic=True)`.
    """
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        # Cache tensors for rescaling bounds
        self.register_buffer("_low", torch.as_tensor(self.policy.action_space.low, dtype=torch.float32))
        self.register_buffer("_high", torch.as_tensor(self.policy.action_space.high, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # SB3 extract_features handles potential feature extractor; returns features for mlp_extractor
        features = self.policy.extract_features(obs)
        latent_pi, _ = self.policy.mlp_extractor(features)
        mean_actions = self.policy.mu(latent_pi)
        # Optional squashing (SB3 sets this True for Box by default)
        if getattr(self.policy, "squash_output", False):
            mean_actions = torch.tanh(mean_actions)
        # Rescale from [-1, 1] → [low, high]
        action = self._low + (mean_actions + 1.0) * 0.5 * (self._high - self._low)
        return action

class ActorExportDiscrete(nn.Module):
    """
    Export wrapper for discrete-action PPO (SB3).
    Produces logits (pre-softmax). Apply softmax in C if you need probabilities.
    """
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.policy.extract_features(obs)
        latent_pi, _ = self.policy.mlp_extractor(features)
        logits = self.policy.action_net(latent_pi)  # shape [B, n_actions]
        return logits

def export_to_onnx(model, onnx_out: str, obs_dim: int, batch: int, opset: int, device: str, discrete: bool):
    policy = model.policy
    if discrete:
        wrapper = ActorExportDiscrete(policy).to(device).eval()
        out_name = "logits"
    else:
        wrapper = ActorExportContinuous(policy).to(device).eval()
        out_name = "actions"

    dummy = torch.randn(batch, obs_dim, dtype=torch.float32, device=device)

    input_names = ["obs"]
    output_names = [out_name]
    dynamic_axes = {
        "obs": {0: "batch"},
        out_name: {0: "batch"},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            onnx_out,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )

def verify_parity(model, onnx_path: str, obs_dim: int, act_dim: int, batch: int, device: str, discrete: bool):
    ort = _lazy_import_onnx()

    # Generate a fixed random input to compare
    rng = np.random.default_rng(123)
    obs = rng.standard_normal(size=(batch, obs_dim)).astype(np.float32)
    torch_obs = torch.from_numpy(obs).to(device)

    # PyTorch output (deployment form)
    policy = model.policy
    if discrete:
        pt_wrapper = ActorExportDiscrete(policy).to(device).eval()
        pt_out = pt_wrapper(torch_obs).detach().cpu().numpy()
        out_name = "logits"
    else:
        pt_wrapper = ActorExportContinuous(policy).to(device).eval()
        pt_out = pt_wrapper(torch_obs).detach().cpu().numpy()
        out_name = "actions"

    # ONNXRuntime output
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    io_binding = sess.io_binding()
    io_binding.bind_cpu_input("obs", obs)
    io_binding.bind_output(out_name, "cpu")
    sess.run_with_iobinding(io_binding)
    ort_out = io_binding.copy_outputs_to_cpu()[0]

    # Compare
    abs_diff = np.max(np.abs(pt_out - ort_out))
    rel_diff = np.max(np.abs(pt_out - ort_out) / (np.maximum(1e-6, np.abs(pt_out))))
    print(f"[VERIFY] max abs diff: {abs_diff:.6g} | max rel diff: {rel_diff:.6g}")

    # Basic sanity checks
    if pt_out.shape != (batch, act_dim) or ort_out.shape != (batch, act_dim):
        raise RuntimeError(f"Shape mismatch: torch {pt_out.shape} vs onnx {ort_out.shape} (expected {(batch, act_dim)})")

    # Tolerances typical for float32 export
    if abs_diff > 1e-5 or rel_diff > 1e-5:
        print("[WARN] Differences exceed tight tolerance; this can still be fine, but re-check opset and wrapper.")
    else:
        print("[OK] ONNX matches PyTorch within 1e-5 tolerance.")

def maybe_write_metadata(path: str, obs_dim: int, act_dim: int, discrete: bool, model_info: dict):
    meta = {
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "action_type": "discrete" if discrete else "continuous",
        "notes": "Outputs are deployment-ready (scaled) actions for continuous case; logits for discrete case.",
        "model_info": model_info,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Wrote metadata: {path}")

def collect_model_info(model) -> dict:
    pi = model.policy
    info = {
        "policy_class": pi.__class__.__name__,
        "feature_extractor": getattr(pi, "features_extractor", None).__class__.__name__ if hasattr(pi, "features_extractor") else "N/A",
        "squash_output": bool(getattr(pi, "squash_output", False)),
        "action_space_low": getattr(pi.action_space, "low", None).tolist() if hasattr(pi.action_space, "low") else None,
        "action_space_high": getattr(pi.action_space, "high", None).tolist() if hasattr(pi.action_space, "high") else None,
        "observation_space_shape": tuple(getattr(pi.observation_space, "shape", ())),
    }
    return info

def main():
    args = parse_args()
    model = load_sb3_ppo(args.model_path, device=args.device)

    # Export to ONNX
    export_to_onnx(model, args.onnx_out, args.obs_dim, args.batch, args.opset, args.device, args.discrete)
    print(f"[OK] Exported ONNX to: {args.onnx_out}")

    # Parity check
    if not args.no_verify:
        verify_parity(model, args.onnx_out, args.obs_dim, args.act_dim, args.batch, args.device, args.discrete)

    # Metadata JSON (optional)
    if args.metadata_out:
        info = collect_model_info(model)
        maybe_write_metadata(args.metadata_out, args.obs_dim, args.act_dim, args.discrete, info)

if __name__ == "__main__":
    main()
