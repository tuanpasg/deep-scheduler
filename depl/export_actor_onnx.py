#!/usr/bin/env python3
import argparse, torch, torch.nn as nn
from stable_baselines3 import PPO
import numpy as np

def _lazy_import_onnxrt():
    import onnx  # noqa: F401
    import onnxruntime as ort
    return ort
    
def verify_parity(actor: nn.Module, onnx_path: str, obs_dim: int, act_dim: int, batch: int):
    """Compare PyTorch vs ONNXRuntime outputs for the same inputs."""
    ort = _lazy_import_onnxrt()

    # Fixed random input for reproducibility
    rng = np.random.default_rng(123)
    obs = rng.random(size=(batch, obs_dim), dtype=np.float32)  # in [0,1]; fine for a smoke test

    # PyTorch output
    with torch.no_grad():
        pt_out = actor(torch.from_numpy(obs)).cpu().numpy()

    # ONNXRuntime output
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["actions"], {"obs": obs})[0]

    # Shape checks
    if pt_out.shape != (batch, act_dim) or onnx_out.shape != (batch, act_dim):
        raise RuntimeError(f"Shape mismatch: torch {pt_out.shape} vs onnx {onnx_out.shape} (expected {(batch, act_dim)})")

    # Range checks (should be (0,1))
    pt_min, pt_max = float(pt_out.min()), float(pt_out.max())
    onnx_min, onnx_max = float(onnx_out.min()), float(onnx_out.max())

    # Numerical diffs
    abs_diff = float(np.max(np.abs(pt_out - onnx_out)))
    rel_diff = float(np.max(np.abs(pt_out - onnx_out) / np.maximum(1e-6, np.abs(pt_out))))

    print(f"[VERIFY] Torch range: [{pt_min:.6f}, {pt_max:.6f}]  | ONNX range: [{onnx_min:.6f}, {onnx_max:.6f}]")
    print(f"[VERIFY] max abs diff: {abs_diff:.6g} | max rel diff: {rel_diff:.6g}")

    # Tight but practical tolerance for float32
    if abs_diff <= 1e-5 and rel_diff <= 1e-5:
        print("[OK] ONNX matches PyTorch within 1e-5 tolerance.")
    else:
        print("[WARN] Differences exceed 1e-5. Check opset, activations, or post-processing.")

class DeterministicActor(nn.Module):
    """
    Wrap SB3 PPO policy -> deterministic actor mean in [0,1].
    - Runs the policy trunk (mlp_extractor)
    - Uses action head (action_net) to produce mean logits
    - Applies sigmoid so outputs match your env's Box(low=0, high=1, shape=(4,))
    """
    def __init__(self, sb3_policy):
        super().__init__()
        self.policy = sb3_policy
        self.mlp_extractor = sb3_policy.mlp_extractor
        self.action_net = sb3_policy.action_net
        # Optional: keep a small epsilon to avoid exact 0/1
        self.eps = 1e-6

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # SB3 policies accept obs as (N, obs_dim) float32
        latent_pi, _ = self.mlp_extractor(obs)
        mean_logits = self.action_net(latent_pi)          # shape (N, act_dim)
        # Map logits -> [0,1]; you can switch to .tanh() and rescale if you prefer
        actions_01 = torch.sigmoid(mean_logits)
        # Lightly clip to stay strictly inside bounds if desired
        actions_01 = torch.clamp(actions_01, self.eps, 1.0 - self.eps)
        return actions_01

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to SB3 PPO .zip (e.g., runs/.../ppo_mac_scheduler.zip)")
    ap.add_argument("--obs-dim", type=int, default=16, help="Observation dimension (your env uses 16)")
    ap.add_argument("--act-dim", type=int, default=4, help="Action dimension (your env uses 4)")
    ap.add_argument("--out", default="ppo_actor.onnx", help="Output ONNX file")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--batch", type=int, default=1, help="Dummy batch size for export/verify")
    ap.add_argument("--no-verify", action="store_true", help="Skip ONNXRuntime parity check")
    args = ap.parse_args()

    # Load SB3 model
    sb3 = PPO.load(args.model, device="cpu")
    sb3.policy.eval()

    # Wrap as a pure nn.Module
    actor = DeterministicActor(sb3.policy).eval()

    # Dummy input for tracing/export (batch dim dynamic)
    dummy = torch.zeros(1, args.obs_dim, dtype=torch.float32)

    # Export
    torch.onnx.export(
        actor, dummy, args.out,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True
    )
    print(f"Exported ONNX actor to: {args.out}")

    if not args.no_verify:
      verify_parity(actor.to("cpu"), args.out, args.obs_dim, args.act_dim, args.batch)

if __name__ == "__main__":
    main()
