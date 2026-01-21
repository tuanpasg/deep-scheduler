import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from DSACD import quantile_huber_loss_per_sample


def test_quantile_huber_loss_zero_td_error_is_zero():
    torch.manual_seed(0)
    B, N = 8, 16
    td = torch.zeros(B, N)
    taus = torch.arange(1, N + 1, dtype=torch.float32) / N
    loss = quantile_huber_loss_per_sample(td, taus, kappa=1.0)
    assert loss.shape == (B,)
    assert torch.allclose(loss, torch.zeros(B)), "Loss should be exactly zero when td_error is zero"


def test_quantile_huber_loss_is_nonnegative():
    torch.manual_seed(0)
    B, N = 8, 16
    td = torch.randn(B, N)
    taus = torch.arange(1, N + 1, dtype=torch.float32) / N
    loss = quantile_huber_loss_per_sample(td, taus, kappa=1.0)
    assert (loss >= 0).all()
    assert torch.isfinite(loss).all()
