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

All 14 metrics come from [finval](https://github.com/sablier-it/finval) v0.1.0
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

## 5. Hyperparameters

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
    synth_paths.npy   # (200, 60, 7) float32
    real_paths.npy    # symlink or copy of data/real_paths.npy
    meta.json         # see schema below
  seed_1/ ...
  seed_2/ ...
  seed_3/ ...
  seed_4/ ...
  finval_scores.json  # output of examples/score.py
  README.md           # 1-paragraph description of your method + citation
```

`meta.json` schema:

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
