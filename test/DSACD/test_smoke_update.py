import sys
from pathlib import Path

import torch

# Make /mnt/data importable so `import DSACD` works
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from DSACD import Actor, QuantileCritic, DSACDUpdater, DSACDHyperParams, ensure_nonempty_mask


def sample_valid_actions(mask: torch.Tensor) -> torch.Tensor:
    """Sample one valid action per row of a boolean mask [B, A]."""
    B, A = mask.shape
    actions = torch.empty(B, dtype=torch.long)
    for i in range(B):
        valid_idx = torch.nonzero(mask[i], as_tuple=False).view(-1)
        # choose one uniformly
        j = valid_idx[torch.randint(0, valid_idx.numel(), (1,))]
        actions[i] = j
    return actions


def test_smoke_update_runs_and_changes_parameters():
    """Basic DSACD sanity test: update() runs, returns finite outputs, and updates params."""
    torch.manual_seed(0)

    device = "cpu"  # deterministic, runs anywhere
    B, obs_dim, A, N = 64, 32, 11, 16

    hp = DSACDHyperParams(n_quantiles=N, gamma=0.0)

    actor = Actor(obs_dim=obs_dim, act_dim=A, hidden=32)
    q1 = QuantileCritic(obs_dim=obs_dim, act_dim=A, n_quantiles=N, hidden=32)
    q2 = QuantileCritic(obs_dim=obs_dim, act_dim=A, n_quantiles=N, hidden=32)
    q1t = QuantileCritic(obs_dim=obs_dim, act_dim=A, n_quantiles=N, hidden=32)
    q2t = QuantileCritic(obs_dim=obs_dim, act_dim=A, n_quantiles=N, hidden=32)
    q1t.load_state_dict(q1.state_dict())
    q2t.load_state_dict(q2.state_dict())

    updater = DSACDUpdater(
        actor=actor,
        q1=q1,
        q2=q2,
        q1_target=q1t,
        q2_target=q2t,
        act_dim=A,
        hp=hp,
        device=device,
    )

    # Random batch with valid masks and actions consistent with masks.
    obs = torch.randn(B, obs_dim)
    next_obs = torch.randn(B, obs_dim)

    mask = (torch.rand(B, A) > 0.3)
    next_mask = (torch.rand(B, A) > 0.3)
    mask = ensure_nonempty_mask(mask, fallback_action=-1)
    next_mask = ensure_nonempty_mask(next_mask, fallback_action=-1)

    actions = sample_valid_actions(mask)
    rewards = torch.randn(B)

    batch = {
        "observation": obs,
        "action": actions,
        "reward": rewards,
        "next_observation": next_obs,
        "action_mask": mask,
        "next_action_mask": next_mask,
    }

    # IS weights all ones for this test
    isw = torch.ones(B)

    # Snapshot some params to confirm updates
    actor_before = {k: v.detach().clone() for k, v in updater.actor.state_dict().items()}

    metrics = updater.update(batch=batch, isw=isw)

    # Finite outputs
    for k in ["loss_q", "loss_pi", "loss_alpha", "alpha"]:
        assert torch.isfinite(metrics[k]).all(), f"{k} should be finite"

    assert metrics["priority"].shape == (B,)
    assert torch.isfinite(metrics["priority"]).all()
    assert (metrics["priority"] > 0).all()

    # Parameters changed
    changed = False
    for k, v in updater.actor.state_dict().items():
        if not torch.allclose(v, actor_before[k]):
            changed = True
            break
    assert changed, "Actor parameters did not change after update()"
