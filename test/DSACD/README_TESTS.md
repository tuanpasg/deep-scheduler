# DSACD unit tests (clarity-first)

These tests are intended to verify the **core DSACD update logic** and the paper-critical
**masking/entropy invariants** before you integrate with your scheduler simulator.

## What is covered

### 1) Masking invariants (paper Appendix D.2)
- (a) invalid actions have zero probability and are never sampled
- (b) all-false masks are repaired via a fallback action
- (c) entropy target increases with the number of valid actions: H̄(s)= -β log(1/|A(s)|)

### 2) Smoke update
- `DSACDUpdater.update()` runs end-to-end on a synthetic batch
- outputs are finite
- actor parameters actually update

### 3) Quantile loss
- quantile huber loss behaves sanely (zero td -> zero loss, non-negative)

## How to run

From the same directory as `DSACD.py`:

```bash
pytest -q
```

Or from anywhere:

```bash
pytest -q /mnt/data/tests
```

If you see import errors, ensure `DSACD.py` is in the parent directory of `tests/`.
