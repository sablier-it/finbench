# FinBench v1 — Protocol Spec

**Status: frozen.** This document defines the v1 benchmark. Any
change to the spec requires a new version (v2) on a new branch — v1
numbers must remain comparable forever.

## 1. Panel

The v1 panel is the bundled `us_equities_macro_2010_2024` dataset
shipped in [sablier-flow](https://pypi.org/project/sablier-flow/).
**7 features, daily, 2010-01-04 to 2023-12-28 (3,522 rows).**

Feature order (canonical, frozen for v1):
``[IWM, QQQ, SPY, TLT, VIX, TNX, DXY]``.

| Symbol | Type        | Notes                          | data_type   |
|--------|-------------|--------------------------------|-------------|
| IWM    | Equity ETF  | Russell 2000 small-cap         | `price`     |
| QQQ    | Equity ETF  | Nasdaq-100                     | `price`     |
| SPY    | Equity ETF  | S&P 500                        | `price`     |
| TLT    | Bond ETF    | 20+ year Treasury              | `price`     |
| VIX    | Vol index   | Implied vol of S&P 500         | `volatility`|
| TNX    | Rate        | 10-year Treasury yield         | `rate`      |
| DXY    | FX index    | US Dollar Index                | `ratio`     |

Load via:

```python
import sablier_flow
df = sablier_flow.demo_data("us_equities_macro_2010_2024")
# Returns a (3522, 7) DataFrame with DatetimeIndex.
```

The panel is **fully public** — no licensing required for any use.

## 2. Train / OOS split

- **Train**: `2010-01-04` through `2019-12-31` (inclusive)
- **OOS**:   `2020-01-02` through `2023-12-28` (inclusive)

The OOS slice deliberately includes the COVID crash (Mar 2020) and
the 2022 bear market — the regimes that matter most for backtest
fidelity. Train must not touch OOS.

## 3. Generation requirements

Submit `(n_paths, horizon, n_features) = (200, 60, 7)` synth windows,
matching the **shape** of the FinBench real reference. Generation
parameters:

- **Horizon**: 60 trading days (~3 months)
- **n_paths**: 200
- **n_seeds**: 5 (report mean ± std across 5 random seeds)
- **Output space**: native return space (log_return for price/ratio
  features; first-difference for rate/volatility features). See the
  `data_type` column above for which transform applies to each
  feature.

The real reference (200 sliding windows from the OOS slice, evenly
sampled if the OOS produces more than 200 windows) is shared so
scoring is deterministic.

## 4. Metrics

All 14 metrics come from [finval](https://github.com/sablier-ai/finval) v0.1.0
via `finval.validate_paths(synth, real)`. Lower is better for all.

| Group       | Metric                  | What it measures                                   |
|-------------|-------------------------|----------------------------------------------------|
| Temporal    | `acf_returns`           | Return autocorrelation (efficient-market check)    |
|             | `volatility_clustering` | ACF(|returns|) (Cont 2001 stylized fact)           |
|             | `leverage_effect`       | corr(r_t, σ_{t+k}) (Black 1976)                    |
|             | `cross_correlation`     | Contemporaneous cross-asset correlation            |
| Distribution| `marginal_ks`           | Per-feature KS test                                |
|             | `energy_distance`       | Multivariate distribution distance                 |
|             | `tail_quantiles`        | 1%/5%/95%/99% quantile mismatch                    |
| Dependence  | `pearson_corr`          | Linear correlation matrix error                    |
|             | `spearman_corr`         | Rank correlation matrix error                      |
|             | `copula_distance`       | Cramér–von Mises copula distance                   |
|             | `correlation_breakdown` | Stress vs calm regime correlation shift            |
| Tail        | `tail_dependence_upper` | Rally co-movement (λ_U)                            |
|             | `tail_dependence_lower` | Crash co-movement (λ_L)                            |
| Drawdown    | `drawdown_distribution` | KS distance on drawdown distribution               |

Aggregates: **overall quality** (excellent/good/acceptable/poor),
**overall score** (0–1, higher is better), **pass rate** (fraction
of 14 metrics that pass their domain-specific thresholds).

## 5. Required secondary metric — TSTR (train-on-synth, test-on-real)

The finval metrics above test *properties* of the synthetic
distribution. **TSTR tests a downstream consequence**: if you fit a
trading strategy on synth and deploy it on real markets, does it
work? This is the question that matters most for backtest-overfitting
applications.

**Submissions MUST report TSTR** alongside the finval scores. The
strategy family is frozen for v1; reference implementation in
[`examples/tstr_strategy.py`](./examples/tstr_strategy.py).

### Strategy family (24 variants, frozen v1)

Each variant operates on the 7-feature panel, dollar-neutral
equal-weight when applicable, daily rebalanced.

| Family                          | Lookbacks | Threshold (TS-MR only) | Count |
|---------------------------------|-----------|------------------------|------:|
| Cross-sectional momentum        | 3, 5, 10, 20 | —                   |     4 |
| Cross-sectional mean-reversion  | 3, 5, 10, 20 | —                   |     4 |
| Time-series momentum            | 3, 5, 10, 20 | —                   |     4 |
| Time-series mean-reversion      | 3, 5, 10, 20 | 0.5, 1.0, 1.5       |    12 |

### TSTR scoring

For each strategy variant ``v`` and each method ``m``:

  1. Compute Sharpe-on-real ``S_v^real`` = mean annualised Sharpe of
     ``v`` evaluated on the OOS sliding-window reference (real data).
  2. Compute Sharpe-on-synth ``S_v^m`` = mean annualised Sharpe of
     ``v`` evaluated on ``m``'s synth, averaged across seeds.

Then for each method, compute:

  - **Spearman ρ**: rank correlation between ``{S_v^real}`` and
    ``{S_v^m}`` across the 24 variants. Higher is better
    (ρ = +1 means synth ranks strategies identically to real;
    ρ = −1 means synth inverts the ranking — fitting on synth
    picks the WORST real-market strategies).
  - **|Δ Sharpe|**: mean absolute difference across variants.
    Lower is better (smaller magnitude error in synth's Sharpe estimates).

### Acceptance bands

| ρ range                | Interpretation                                                    |
|------------------------|-------------------------------------------------------------------|
| ρ ≥ +0.80, p < 0.01    | **Faithful**: fitting on synth ≈ fitting on real |
| +0.50 ≤ ρ < +0.80      | **Partial**: directionally correct but unreliable magnitudes |
| −0.50 < ρ < +0.50      | **Weak**: synth ranks strategies essentially at random |
| ρ ≤ −0.50, p < 0.01    | **Inverted**: synth misleads — fitting picks losing strategies |

### Why NOT folded into the finval aggregate

TSTR and the 14 finval metrics test different things (downstream
utility vs distributional properties). We report both side-by-side
on the leaderboard but deliberately do not combine them into a
single number — there is no defensible way to weight "rank
correlation across 24 strategies" against "Wasserstein on marginals"
into one score. Submitters and readers may emphasise whichever
matches their use case.

### v2 strategy family

The 24-variant family above is a deliberately conservative subset
(momentum + mean-reversion only). FinBench v2 may broaden it (carry,
vol-targeting, factor signals, multi-asset overlay). v1 TSTR
numbers will not transfer to v2.

## 6. Hyperparameters

**Each method uses its own published defaults.** No per-dataset
tuning. This is the standard practice in the TS-gen literature
(TimeGAN, TimeVAE, Diffusion-TS, KoVAE, FAR-TS all do this). The
goal is to test the architectures, not the hyperparameter sweep.

If you submit a method that requires tuning to converge, document
it in your submission's `meta.json` (`hyperparameters_tuned: true`,
plus the tuning protocol). Tuned submissions are accepted but
**marked with a `*` in the leaderboard**.

## 6. Reproducibility

- Random seeds: 0, 1, 2, 3, 4 (the integers we standardize on)
- Software versions: pin in your `meta.json` (Python, framework,
  finval version, etc.)
- Hardware: report the GPU you trained on (`meta.json`). Submissions
  are NOT GPU-comparable; we don't care about speed, only quality.

## 7. Submission format

Your submission directory under `reference/<method_name>/` contains:

```
reference/<method_name>/
  seed_0/
    synth_paths.npy        # (200, 60, 7) float32
    real_paths.npy         # symlink or copy of data/real_paths.npy
    meta.json              # per-seed metadata; schema below
    finval_scores.json     # per-seed finval scoring output
  seed_1/ ...
  seed_2/ ...
  seed_3/ ...
  seed_4/ ...
  finval_scores.json       # aggregate across seeds; output of examples/score.py
  tstr_scores.json         # required secondary metric; output of examples/tstr_strategy.py
  README.md                # 1-paragraph description of your method + citation
```

Per-seed `meta.json` schema:

```json
{
  "method": "your_method_name",
  "version": "1.0.0",
  "code": "https://github.com/your-org/your-repo",
  "paper": "https://arxiv.org/abs/...",
  "seed": 0,
  "hyperparameters_tuned": false,
  "framework": "pytorch 2.7.0",
  "wall_seconds": 884.3,
  "gpu": "NVIDIA A100 SXM4 40GB"
}
```

Aggregate `tstr_scores.json` schema:

```json
{
  "method": "your_method_name",
  "spearman_rho": 0.850,
  "p_value": 0.00003,
  "mean_abs_sharpe_gap": 0.361,
  "n_strategy_variants": 24,
  "strategy_family_version": "v1"
}
```

## 8. What FinBench does NOT require

- Your model code (optional, encouraged but not required)
- Your weights (irrelevant — only outputs matter)
- Any specific framework (PyTorch, TF, JAX all work)
- Any specific architecture (GAN, VAE, diffusion, flow, autoregressive
  — all welcome)

We score outputs, not internals. Same protocol everyone runs against.

## 9. Updates

- Bug fixes to finval that don't change metric semantics are
  automatically picked up (finval version pinned per leaderboard
  generation; we re-run reference results when finval bumps).
- Changes to the panel, split, or metric set require a new FinBench
  version (v2, v3, ...). v1 numbers stay valid forever.
