"""TSTR (train-strategy-on-synth, test-on-real) backtest evaluation.

The acid test for any financial synth: if a strategy is fitted on
synthetic returns, does it still work when deployed on real markets?

We define a strategy family of K parameter variants. For each variant
we compute its Sharpe on (a) the real OOS panel and (b) each method's
synth. We then ask: **does the synth rank strategies the same way real
data does?** Quantified as Spearman correlation between per-variant
Sharpe-on-real and Sharpe-on-synth.

  - Spearman ≈ 1.0 → synth is a faithful proxy for strategy selection.
    Fitting on synth ≈ fitting on real.
  - Spearman ≈ 0   → synth misranks strategies. Worse than useless.
  - Spearman < 0   → synth actively misleads.

This is the metric a backtest-overfitting framework actually cares
about. It complements the stylized-facts finval metrics by testing
the *downstream consequence* of those metrics.

Strategy family (cross-sectional + time-series):

  * Cross-sectional momentum:    rank assets by past-N return; long top, short bottom
  * Cross-sectional mean-rev:    same, inverted (long bottom, short top)
  * Time-series momentum:        long when past-N return > 0
  * Time-series mean-reversion:  long when past-N z-score < -t

Lookback Ns ∈ {3, 5, 10, 20}.  Thresholds for MR z-score ∈ {0.5, 1.0, 1.5}.

Run from the repo root:

    python examples/tstr_strategy.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Treat all 7 features as tradeable for the strategy family. This is
# fair to all baselines — each method's synth covers the same 7
# features, so each strategy gets evaluated identically on synth and
# real.
N_FEATURES = 7


# ---------------------------------------------------------------------------
# Strategy implementations
#
# Each returns ``(n_paths, T)`` daily PnL (equity-weighted, dollar-neutral).
# Sharpe is computed across the path-mean equity curve.
# ---------------------------------------------------------------------------

def _cs_mom_pnl(R: np.ndarray, lookback: int) -> np.ndarray:
    """Cross-sectional momentum. ``R`` shape ``(n_paths, T, F)``.

    On each rebalance, rank features by past-``lookback`` cumulative return,
    long the top half (weight +1/k), short the bottom half (-1/k). PnL
    realized as next-bar return weighted by these signs.
    """
    n, T, F = R.shape
    if lookback >= T - 1:
        return np.zeros((n, T - lookback - 1), dtype=np.float32)
    cum = np.cumsum(R, axis=1)
    pnl_t = []
    for t in range(lookback, T - 1):
        past = cum[:, t, :] - cum[:, t - lookback, :]
        ranks = np.argsort(np.argsort(past, axis=1), axis=1)
        weights = np.where(ranks >= F // 2, 1.0, -1.0) / (F // 2)
        pnl = (weights * R[:, t + 1, :]).sum(axis=1)
        pnl_t.append(pnl)
    return np.stack(pnl_t, axis=1).astype(np.float32)


def _cs_mr_pnl(R: np.ndarray, lookback: int) -> np.ndarray:
    """Cross-sectional mean-reversion = sign-flipped momentum."""
    return -_cs_mom_pnl(R, lookback)


def _ts_mom_pnl(R: np.ndarray, lookback: int) -> np.ndarray:
    """Time-series momentum, equal-weight per asset.

    For each asset & each bar: long it next bar if its past-``lookback``
    sum is positive, short if negative. PnL = mean signed return.
    """
    n, T, F = R.shape
    if lookback >= T - 1:
        return np.zeros((n, T - lookback - 1), dtype=np.float32)
    cum = np.cumsum(R, axis=1)
    pnl_t = []
    for t in range(lookback, T - 1):
        past = cum[:, t, :] - cum[:, t - lookback, :]
        sign = np.sign(past)
        pnl = (sign * R[:, t + 1, :]).mean(axis=1)
        pnl_t.append(pnl)
    return np.stack(pnl_t, axis=1).astype(np.float32)


def _ts_mr_pnl(R: np.ndarray, lookback: int, thresh: float) -> np.ndarray:
    """Time-series mean-reversion with z-score threshold.

    Long if z-score < -thresh, short if > +thresh, flat otherwise.
    """
    n, T, F = R.shape
    if lookback >= T - 1:
        return np.zeros((n, T - lookback - 1), dtype=np.float32)
    pnl_t = []
    for t in range(lookback, T - 1):
        window = R[:, t - lookback : t, :]
        mu = window.mean(axis=1)
        sd = window.std(axis=1) + 1e-9
        z = (R[:, t, :] - mu) / sd
        sign = np.where(z < -thresh, 1.0, np.where(z > thresh, -1.0, 0.0))
        pnl = (sign * R[:, t + 1, :]).mean(axis=1)
        pnl_t.append(pnl)
    return np.stack(pnl_t, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Strategy family — 26 variants total
# ---------------------------------------------------------------------------

def _build_family() -> list[tuple[str, callable]]:
    variants: list[tuple[str, callable]] = []
    for lb in (3, 5, 10, 20):
        variants.append((f"cs_mom_lb{lb}", lambda R, l=lb: _cs_mom_pnl(R, l)))
        variants.append((f"cs_mr_lb{lb}",  lambda R, l=lb: _cs_mr_pnl(R, l)))
        variants.append((f"ts_mom_lb{lb}", lambda R, l=lb: _ts_mom_pnl(R, l)))
        for t in (0.5, 1.0, 1.5):
            variants.append((f"ts_mr_lb{lb}_t{t}", lambda R, l=lb, t=t: _ts_mr_pnl(R, l, t)))
    return variants


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _annualized_sharpe(pnl: np.ndarray) -> float:
    """Per-path annualized Sharpe → averaged across paths."""
    mu = pnl.mean(axis=1)
    sd = pnl.std(axis=1) + 1e-12
    sharpe = (mu / sd) * np.sqrt(252)
    return float(np.mean(sharpe))


def evaluate_family(R: np.ndarray) -> dict[str, float]:
    """For each variant, compute its mean Sharpe across the paths in ``R``."""
    fam = _build_family()
    out = {}
    for name, fn in fam:
        pnl = fn(R)
        if pnl.size == 0:
            out[name] = 0.0
            continue
        out[name] = _annualized_sharpe(pnl)
    return out


def load_real_reference_windows() -> np.ndarray:
    """Build the same OOS sliding-window reference finbench uses."""
    import sablier_flow

    df = sablier_flow.demo_data("us_equities_macro_2010_2024")
    df.index = pd.to_datetime(df.index)
    oos = df.loc[df.index >= "2020-01-01"]
    cols = list(df.columns)
    FEATURE_TYPES = {
        "IWM": "price", "QQQ": "price", "SPY": "price", "TLT": "price",
        "VIX": "volatility", "TNX": "rate", "DXY": "ratio",
    }
    returns = pd.DataFrame(index=oos.index, columns=cols, dtype=np.float32)
    for c in cols:
        if FEATURE_TYPES[c] in ("price", "ratio", "level"):
            returns[c] = np.log(oos[c]).diff()
        else:
            returns[c] = oos[c].diff()
    returns = returns.dropna().astype(np.float32).values
    L = 60
    T = returns.shape[0]
    n_w = T - L + 1
    windows = np.stack([returns[i : i + L] for i in range(n_w)], axis=0)
    if windows.shape[0] > 200:
        rng = np.random.default_rng(0)
        idx = rng.choice(windows.shape[0], size=200, replace=False)
        idx.sort()
        windows = windows[idx]
    return windows.astype(np.float32)


def score_method(method_dir: Path) -> dict:
    """Aggregate TSTR results across the method's seeds."""
    seed_scores = []
    for sd in sorted(method_dir.glob("seed_*")):
        syn = np.load(sd / "synth_paths.npy")
        seed_scores.append(evaluate_family(syn))
    # Average Sharpe across seeds per variant.
    variant_names = list(seed_scores[0])
    synth_sharpes = {
        v: float(np.mean([s[v] for s in seed_scores]))
        for v in variant_names
    }
    return synth_sharpes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reference_root", type=Path, default=Path("reference"))
    p.add_argument("--out", type=Path, default=Path("tstr_results.json"))
    args = p.parse_args()

    # Real reference Sharpe on the OOS sliding windows.
    print("Loading real OOS reference …")
    real = load_real_reference_windows()
    real_sharpes = evaluate_family(real)
    print(f"  evaluated {len(real_sharpes)} strategy variants on real OOS")

    # Per-method synth Sharpe → correlation.
    methods = ["sablier_flow", "kovae", "diffusion_ts", "timevae", "timegan"]
    rows = []
    for m in methods:
        method_dir = args.reference_root / m
        if not method_dir.exists():
            print(f"  {m}: skipped (no reference dir)")
            continue
        synth_sharpes = score_method(method_dir)
        vals_real = np.array([real_sharpes[v] for v in synth_sharpes])
        vals_synth = np.array([synth_sharpes[v] for v in synth_sharpes])
        rho, p_val = spearmanr(vals_real, vals_synth)
        sharpe_gap = float(np.mean(np.abs(vals_real - vals_synth)))
        rows.append({
            "method": m,
            "spearman": float(rho),
            "p_value": float(p_val),
            "mean_abs_sharpe_gap": sharpe_gap,
        })
        print(f"  {m:14s}  rho = {rho:+.3f}  p = {p_val:.4f}  |Δ Sharpe| = {sharpe_gap:.3f}")

    args.out.write_text(json.dumps({
        "real_sharpes": real_sharpes,
        "method_rows": rows,
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
