import argparse
import random
from collections import deque

import torch

from toy5g_env_adapter import DeterministicToy5GEnvAdapter

from DSACD_multibranch import (
    MultiBranchActor,
    MultiBranchQuantileCritic,
    DSACDUpdater,
    DSACDHyperParams,
    ensure_nonempty_mask,
    apply_action_mask_to_logits,
)

@torch.no_grad()
def evaluate_pf_match(eval_env, actor, *, n_eval_ttis=50, tol=1e-9):
    """
    Runs an independent evaluation rollout on eval_env and measures how often
    the actor's chosen action equals the PF-greedy action (ties allowed).

    Returns:
      pf_match: matches / total decisions
      avg_valid: average number of valid actions per decision
      rand_baseline: average (1 / valid_count) per decision (better than 1/avg_valid)
    """
    dev = next(actor.parameters()).device

    # Reset eval env to a stable starting point each evaluation
    eval_env.reset()

    matches = 0
    total = 0
    valid_sum = 0
    rand_sum = 0.0

    for _ in range(n_eval_ttis):
        # PF reference must be "before TTI" (same semantics as reward code)
        avg_ref = eval_env.avg_tp.detach().clone()

        eval_env.begin_tti()

        for layer_ctx in eval_env.layer_iter():
            obs = layer_ctx.obs.unsqueeze(0).to(dev)          # [1, obs_dim]
            masks = layer_ctx.masks_rbg.to(dev)               # [M, A] bool

            logits_all = actor.forward_all(obs).squeeze(0)    # [M, A]
            logits_all = apply_action_mask_to_logits(logits_all, masks)

            # sample actions (same as training); if you want deterministic eval, replace with argmax
            actions = torch.distributions.Categorical(logits=logits_all).sample()  # [M]

            # PF-greedy check per RBG
            for m in range(eval_env.n_rbg):
                mask_m = masks[m]  # [A]
                valid_idx = torch.nonzero(mask_m, as_tuple=False).view(-1)

                vcnt = int(valid_idx.numel())
                valid_sum += vcnt
                rand_sum += 1.0 / max(vcnt, 1)

                chosen = int(actions[m].item())

                # Compute PF for all valid actions and take argmax set (ties allowed)
                # PF(u,m)=rate(u,m)/(avg_ref[u]+eps); NOOP is allowed but pf=0 in env._pf
                pf_vals = []
                for a in valid_idx.tolist():
                    pf_vals.append(eval_env._pf(a, m, avg_ref))
                pf_vals_t = torch.tensor(pf_vals, device=dev, dtype=torch.float32)

                max_pf = float(pf_vals_t.max().item())
                # tie set: those within tol of max
                best_mask = (pf_vals_t >= (max_pf - tol))
                best_actions = set([valid_idx[i].item() for i in torch.nonzero(best_mask, as_tuple=False).view(-1).tolist()])

                if chosen in best_actions:
                    matches += 1
                total += 1

            # Advance env using the actor actions (so eval state distribution is realistic)
            eval_env.apply_layer_actions(layer_ctx, actions.cpu())

        eval_env.end_tti()

    pf_match = matches / max(total, 1)
    avg_valid = valid_sum / max(total, 1)
    rand_baseline = rand_sum / max(total, 1)
    return pf_match, avg_valid, rand_baseline

# -------------------------
# Diagnostic: greedy match rate
# -------------------------
@torch.no_grad()
def greedy_match_rate(env, actor, n_samples=3):
    dev = next(actor.parameters()).device

    # ---- snapshot env state (minimal set for this toy env) ----
    t0 = env.t
    buf0 = env.buf.detach().clone()
    avg0 = env.avg_tp.detach().clone()
    alloc0 = env._alloc.detach().clone()

    matches = 0
    total = 0
    valid_cnt_sum = 0

    for _ in range(n_samples):
        # Use the SAME reference as reward uses: "before TTI"
        avg_ref = env.avg_tp.detach().clone()

        env.begin_tti()

        for layer_ctx in env.layer_iter():
            obs = layer_ctx.obs.unsqueeze(0).to(dev)     # [1, obs_dim]
            masks = layer_ctx.masks_rbg.to(dev)          # [M, A]

            logits_all = actor.forward_all(obs).squeeze(0)  # [M, A]
            logits_all = apply_action_mask_to_logits(logits_all, masks)
            actions = torch.distributions.Categorical(logits=logits_all).sample()  # [M]

            for m in range(env.n_rbg):
                chosen = int(actions[m].item())
                v_m = env._greedy_indicator_v(m, chosen, avg_ref)  # <- snapshot ref
                matches += (v_m > 0)
                total += 1
                valid_cnt_sum += int(masks[m].sum().item())

        # IMPORTANT: do NOT call env.end_tti() (it mutates buf/avg_tp a lot)

    # ---- restore env state ----
    env.t = t0
    env.buf = buf0
    env.avg_tp = avg0
    env._alloc = alloc0

    avg_valid = valid_cnt_sum / max(total, 1)
    baseline = 1.0 / max(avg_valid, 1.0)
    return float(matches) / max(total, 1), avg_valid, baseline

# -------------------------
# Minimal uniform replay
# -------------------------
class SimpleReplay:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def add(self, tr: dict):
        self.buf.append(tr)

    @property
    def size(self) -> int:
        return len(self.buf)

    def sample(self, batch_size: int, device: torch.device) -> dict:
        batch = random.sample(self.buf, batch_size)

        def stack(key, dtype=None):
            xs = [b[key] for b in batch]
            x = torch.stack([t if t.ndim > 0 else t.view(1) for t in xs], dim=0)
            if dtype is not None:
                x = x.to(dtype)
            return x.to(device)

        out = {
            "observation": stack("observation", torch.float32),
            "next_observation": stack("next_observation", torch.float32),
            "rbg_index": stack("rbg_index", torch.long).squeeze(-1),
            "action": stack("action", torch.long).squeeze(-1),
            "reward": stack("reward", torch.float32).squeeze(-1),
            "action_mask": stack("action_mask").to(torch.bool).squeeze(1)
            if batch[0]["action_mask"].ndim == 1
            else stack("action_mask").to(torch.bool),
            "next_action_mask": stack("next_action_mask").to(torch.bool).squeeze(1)
            if batch[0]["next_action_mask"].ndim == 1
            else stack("next_action_mask").to(torch.bool),
        }
        return out


# -------------------------
# Action sampling (1 forward per layer)
# -------------------------
@torch.no_grad()
def sample_actions_for_layer(actor: MultiBranchActor,
                             obs_layer: torch.Tensor,        # [obs_dim]
                             masks_rbg: torch.Tensor,        # [NRBG, A] bool
                             device: torch.device,
                             fallback_action: int) -> torch.Tensor:
    obs_b = obs_layer.unsqueeze(0).to(device)          # [1, obs_dim]
    logits_all = actor.forward_all(obs_b).squeeze(0)   # [NRBG, A]

    masks_rbg = ensure_nonempty_mask(masks_rbg.to(device), fallback_action=fallback_action)
    logits_all = apply_action_mask_to_logits(logits_all, masks_rbg)

    dist = torch.distributions.Categorical(logits=logits_all)  # batched over NRBG
    actions = dist.sample()                                    # [NRBG]
    return actions.cpu()


def main(args):
    device = torch.device(args.device)


    env = DeterministicToy5GEnvAdapter(
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        n_layers=args.n_layers,
        n_rbg=args.n_rbg,
        device=args.device,
        seed=args.seed,
        k=0.2,       # paper factor
        gmax=1.0,    # training-only normalizer (toy: keep 1.0)
    )

    eval_env = DeterministicToy5GEnvAdapter(
    obs_dim=args.obs_dim,
    act_dim=args.act_dim,
    n_layers=args.n_layers,
    n_rbg=args.n_rbg,
    fallback_action=args.fallback_action,
    device=args.device,
    seed=args.seed + 12345,   # different seed OK; use same if you want exact-repeat eval
    )

    # Networks
    hp = DSACDHyperParams(
        n_quantiles=args.n_quantiles,
        beta=args.beta,
        gamma=args.gamma,
        tau=args.tau,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_alpha=args.lr_alpha,
        fallback_action=args.fallback_action,
    )

    actor = MultiBranchActor(obs_dim=args.obs_dim, n_rbg=args.n_rbg, act_dim=args.act_dim, hidden=args.hidden)
    q1 = MultiBranchQuantileCritic(obs_dim=args.obs_dim, n_rbg=args.n_rbg, act_dim=args.act_dim,
                                   n_quantiles=hp.n_quantiles, hidden=args.hidden)
    q2 = MultiBranchQuantileCritic(obs_dim=args.obs_dim, n_rbg=args.n_rbg, act_dim=args.act_dim,
                                   n_quantiles=hp.n_quantiles, hidden=args.hidden)
    q1_t = MultiBranchQuantileCritic(obs_dim=args.obs_dim, n_rbg=args.n_rbg, act_dim=args.act_dim,
                                     n_quantiles=hp.n_quantiles, hidden=args.hidden)
    q2_t = MultiBranchQuantileCritic(obs_dim=args.obs_dim, n_rbg=args.n_rbg, act_dim=args.act_dim,
                                     n_quantiles=hp.n_quantiles, hidden=args.hidden)
    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())

    updater = DSACDUpdater(
        actor=actor,
        q1=q1, q2=q2,
        q1_target=q1_t, q2_target=q2_t,
        n_rbg=args.n_rbg,
        act_dim=args.act_dim,
        hp=hp,
        device=str(device),
    )

    actor_ref = {k: v.detach().clone() for k, v in updater.actor.state_dict().items()}

    def actor_param_delta():
      s = 0.0
      for k, v in updater.actor.state_dict().items():
          s += (v.detach() - actor_ref[k]).pow(2).sum().item()
      return s ** 0.5
    replay = SimpleReplay(capacity=args.rb_capacity)

    env.reset()

    for tti in range(args.ttis):
        env.begin_tti()

        for layer_ctx in env.layer_iter():
            actions_rbg = sample_actions_for_layer(
                actor=updater.actor,
                obs_layer=layer_ctx.obs,
                masks_rbg=layer_ctx.masks_rbg,
                device=device,
                fallback_action=args.fallback_action,
            )
            env.apply_layer_actions(layer_ctx, actions_rbg)

        env.end_tti()
        transitions = env.export_branch_transitions()

        for tr in transitions:
            replay.add(tr)

        if replay.size >= args.learning_starts:
            batch = replay.sample(args.batch_size, device=device)
            metrics = updater.update(batch=batch, isw=None)

        # print("actor_delta_l2:", actor_param_delta())

        if tti % args.log_every == 0:
            msg = f"[TTI {tti}] Buffer={replay.size}"

            if replay.size >= args.learning_starts:
                msg += (
                    f" alpha={metrics['alpha']:.4f}"
                    f" loss_q={metrics['loss_q']:.4f}"
                    f" loss_pi={metrics['loss_pi']:.4f}"
                )

                # ---- ADD THIS LINE ----
                # gmr, avg_valid, base = greedy_match_rate(env, updater.actor, n_samples=3)
                # msg += f" greedy_match={gmr:.3f} avg_valid={avg_valid:.1f} rand≈{base:.3f}"

            print(msg)

        if (tti % args.eval_every == 0) and (replay.size >= args.learning_starts):
            msg = f"[TTI {tti}]"
            pfm, avg_valid, rbase = evaluate_pf_match(eval_env, 
                                                        updater.actor, 
                                                        n_eval_ttis=args.eval_ttis)
            msg += f" | EVAL pf_match={pfm:.3f} avg_valid={avg_valid:.1f} rand≈{rbase:.3f}"
            print(msg)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--ttis", type=int, default=400)
    p.add_argument("--learning_starts", type=int, default=256)
    p.add_argument("--log_every", type=int, default=20)

    p.add_argument("--obs_dim", type=int, default=410)
    p.add_argument("--act_dim", type=int, default=11)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_rbg", type=int, default=18)
    p.add_argument("--fallback_action", type=int, default=0)

    p.add_argument("--rb_capacity", type=int, default=72000)
    p.add_argument("--batch_size", type=int, default=64)

    # DSACD knobs
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--n_quantiles", type=int, default=16)
    p.add_argument("--beta", type=float, default=0.98)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--tau", type=float, default=0.001)
    p.add_argument("--lr_actor", type=float, default=1e-4)
    p.add_argument("--lr_critic", type=float, default=1e-4)
    p.add_argument("--lr_alpha", type=float, default=1e-4)

    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--eval_ttis", type=int, default=50)

    args = p.parse_args()
    main(args)
