import sys
from pathlib import Path

import torch

# Make /mnt/data importable so `import DSACD` works when running pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from DSACD import Actor, ensure_nonempty_mask


def _make_actor(obs_dim: int, act_dim: int) -> Actor:
    """Small actor for unit tests."""
    torch.manual_seed(0)
    return Actor(obs_dim=obs_dim, act_dim=act_dim, hidden=32)


def test_masking_invariant_a_no_invalid_action_can_be_sampled():
    """(a) If mask[a] is False, the masked categorical must never sample a."""
    torch.manual_seed(0)
    obs_dim, act_dim = 8, 11
    actor = _make_actor(obs_dim, act_dim)

    # Build a mask with both valid and invalid actions.
    mask = torch.tensor([[True, True, False, True, False, True, False, True, False, True, False]])
    obs = torch.zeros(1, obs_dim)

    logits = actor(obs, mask)  # [1, A]
    dist = torch.distributions.Categorical(logits=logits)
    probs = dist.probs

    # Invalid actions should have exactly zero prob (softmax underflows to 0).
    invalid = ~mask
    assert torch.all(probs[invalid] == 0), "Invalid actions should have prob 0 after masking"
    assert torch.isclose(probs.sum(), torch.tensor(1.0)), "Probs must sum to 1"

    # Sample a bunch and ensure we never get an invalid action.
    samples = dist.sample((50_000,))  # [K, 1]
    samples = samples.view(-1)
    invalid_indices = torch.nonzero(invalid[0], as_tuple=False).view(-1)
    assert not torch.isin(samples, invalid_indices).any(), "Sampled an invalid action"


def test_masking_invariant_b_all_false_mask_is_handled():
    """(b) All-false masks must be repaired so the distribution is valid.

    DSACD.py uses ensure_nonempty_mask() to force a fallback action valid.
    """
    torch.manual_seed(0)
    obs_dim, act_dim = 8, 11
    actor = _make_actor(obs_dim, act_dim)

    # All invalid initially.
    mask0 = torch.zeros(1, act_dim, dtype=torch.bool)
    mask = ensure_nonempty_mask(mask0, fallback_action=-1)

    assert mask.sum().item() == 1, "Mask repair should make exactly one action valid here"
    assert mask[0, -1].item() is True, "Fallback (last) action should be forced valid"

    obs = torch.zeros(1, obs_dim)
    logits = actor(obs, mask)
    dist = torch.distributions.Categorical(logits=logits)

    # Distribution must be well-defined
    probs = dist.probs
    assert torch.isfinite(probs).all()
    assert torch.isclose(probs.sum(), torch.tensor(1.0))

    # Only the fallback should ever be sampled
    samples = dist.sample((10_000,)).view(-1)
    assert (samples == (act_dim - 1)).all(), "With only one valid action, it must always be selected"


def test_masking_invariant_c_entropy_target_increases_with_valid_action_count():
    """(c) Paper Eq.17 uses H_bar(s)= -beta*log(1/|A(s)|) = beta*log(|A(s)|).

    So H_bar should increase monotonically as the number of valid actions increases.
    """
    beta = 0.999

    def H_target(valid_count: int) -> float:
        vc = max(valid_count, 1)
        return float(-beta * torch.log(torch.tensor(1.0 / vc)))

    h1 = H_target(1)
    h2 = H_target(2)
    h4 = H_target(4)
    h9 = H_target(9)

    assert h1 <= h2 <= h4 <= h9, "Entropy target must increase as |A(s)| increases"
    assert abs(h1) < 1e-6, "If |A(s)|=1 then log(1)=0 so target entropy should be ~0"
