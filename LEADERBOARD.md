# FinBench v1 — Leaderboard

Multivariate financial time-series generation on the
`us_equities_macro_2010_2024` panel (IWM / QQQ / SPY / TLT / VIX /
TNX / DXY), 2010-2019 train, 2020-2023 OOS, L=60, 200 paths,
5 seeds. Scored with [finval](https://github.com/sablier-it/finval) v0.1.0.

> Lower is better for every individual metric. **Bold** = current
> best on that metric. ✨ excellent · ✓ good · ⚪ acceptable · ❌ poor.

## Overall ranking

| Rank | Method | Overall | Score | Pass | Notes |
|-----:|--------|---------|------:|-----:|-------|
| **1** | **Sablier-Flow** (Sablier, 2026) | **good** ✓ | **0.794 ± 0.017** | **13 / 14** | Designed for financial stylized facts |
| 2 | KoVAE (Naiman et al., ICLR 2024) | acceptable ⚪ | 0.616 ± 0.024 | 10 / 14 | Koopman VAE; strong on vol clustering + drawdown |
| 3 | Diffusion-TS (Yuan & Qiao, ICLR 2024) | acceptable ⚪ | 0.521 ± 0.000 | 10 / 14 | Diffusion; fails marginals (energy_distance, tail_quantiles) |
| 4 | TimeVAE (Desai et al., 2021) | poor ❌ | 0.403 ± 0.014 | 10 / 14 | Mode-collapse on returns |
| 5 | TimeGAN (Yoon et al., NeurIPS 2019) | poor ❌ | 0.388 ± 0.110 | 9 / 14 | High seed variance; fails cross-asset structure |

## Per-metric breakdown (mean ± std across 5 seeds, lower = better)

| Metric | Sablier-Flow | KoVAE | Diffusion-TS | TimeVAE | TimeGAN |
|---|---:|---:|---:|---:|---:|
| **Temporal** | | | | | |
| acf_returns | **0.024 ± 0.001** | **0.024 ± 0.001** | 0.032 ± 0.002 | 0.102 ± 0.010 | 0.146 ± 0.052 |
| volatility_clustering | 0.037 ± 0.002 | **0.031 ± 0.002** | 0.080 ± 0.002 | 0.115 ± 0.014 | 0.127 ± 0.028 |
| leverage_effect | **0.031 ± 0.002** | 0.032 ± 0.001 | 0.055 ± 0.002 | 0.156 ± 0.016 | 0.111 ± 0.041 |
| cross_correlation | **0.146 ± 0.012** | 0.308 ± 0.007 | 0.175 ± 0.015 | 0.233 ± 0.020 | 0.487 ± 0.172 |
| **Distribution** | | | | | |
| marginal_ks | **0.094 ± 0.015** | 0.146 ± 0.002 | 0.949 ± 0.000 | 0.475 ± 0.002 | 0.149 ± 0.021 |
| energy_distance | **0.212 ± 0.076** | 0.376 ± 0.028 | 4.388 ± 0.012 | 1.443 ± 0.021 | 0.277 ± 0.041 |
| tail_quantiles | **0.661 ± 0.099** | 0.979 ± 0.016 | 38.358 ± 0.074 | 1.977 ± 0.003 | 1.094 ± 0.017 |
| **Dependence** | | | | | |
| pearson_corr | **0.146 ± 0.012** | 0.308 ± 0.007 | 0.175 ± 0.015 | 0.233 ± 0.020 | 0.487 ± 0.172 |
| spearman_corr | **0.162 ± 0.011** | 0.339 ± 0.008 | 0.179 ± 0.013 | 0.214 ± 0.013 | 0.480 ± 0.165 |
| copula_distance | **0.019 ± 0.001** | 0.038 ± 0.001 | 0.020 ± 0.001 | 0.025 ± 0.001 | 0.058 ± 0.015 |
| correlation_breakdown | **0.140 ± 0.012** | 0.298 ± 0.008 | 0.303 ± 0.020 | 0.279 ± 0.033 | 0.502 ± 0.159 |
| **Tail** | | | | | |
| tail_dependence_upper | **0.046 ± 0.006** | 0.121 ± 0.001 | 0.071 ± 0.007 | 0.091 ± 0.008 | 0.174 ± 0.062 |
| tail_dependence_lower | **0.062 ± 0.004** | 0.133 ± 0.004 | 0.093 ± 0.008 | 0.099 ± 0.016 | 0.153 ± 0.060 |
| **Backtest-critical** | | | | | |
| drawdown_distribution | 0.160 ± 0.020 | **0.126 ± 0.021** | 1.000 ± 0.000 | 0.548 ± 0.056 | 0.369 ± 0.082 |

**Sablier-Flow wins 12 of 14 metrics outright.** KoVAE wins 2
(`volatility_clustering`, `drawdown_distribution`) and ties on
`acf_returns`. All other neural baselines fail at least one
backtest-critical metric badly (Diffusion-TS at `drawdown_distribution`
and `tail_quantiles`; TimeVAE at marginals; TimeGAN at cross-asset
structure with high seed variance).

## TSTR Backtest — does fitting on synth actually pick good strategies for real markets?

The decisive financial test. We define a 24-variant strategy family
(cross-sectional momentum / mean-reversion + time-series momentum /
mean-reversion, multiple lookbacks and thresholds), compute each
variant's Sharpe on (a) the real OOS panel and (b) every method's
synth, then measure **Spearman correlation** between per-variant
Sharpe-on-real and Sharpe-on-synth. A perfectly faithful synth
ranks strategies identically to real (ρ = +1.0).

| Rank | Method | Spearman ρ | p-value | Interpretation |
|-----:|--------|----------:|--------:|----------------|
| **1** | **KoVAE** | **+0.860** | <0.0001 | Faithful — synth ranks strategies like real |
| 2 | **Sablier-Flow** | **+0.850** | <0.0001 | Faithful — synth ranks strategies like real |
| 3 | TimeGAN | +0.535 | 0.0071 | Weak — partial signal, large magnitude errors |
| 4 | Diffusion-TS | **−0.724** | 0.0001 | **Inverted — fitting on synth picks losing strategies** |
| 5 | TimeVAE | **−0.788** | <0.0001 | **Inverted — fitting on synth picks losing strategies** |

**The headline:** fitting a strategy on **Diffusion-TS** or **TimeVAE**
synth and deploying it on real markets actively *misleads* the
practitioner — synth ranks the worst strategies as best. Their
distributional-matching protocols don't preserve the *structure*
that determines strategy performance.

By contrast, **Sablier-Flow** and **KoVAE** rank strategies on synth ≈
how real ranks them (ρ ≈ +0.85 with p < 0.0001). A backtest fitted
on either method's synth is a faithful proxy for what would happen on
real markets — the working definition of a useful synthetic dataset.

### Worked example: `cs_mom_lb3` (3-day cross-sectional momentum)

| Source | Sharpe |
|--------|------:|
| Real OOS | **−0.85** (strategy loses on real) |
| Sablier-Flow synth | −0.73 ✅ (correctly says "loser") |
| KoVAE synth | −0.91 ✅ (correctly says "loser") |
| Diffusion-TS synth | **+76.08** ❌ (says huge winner — wrong sign + wrong magnitude) |
| TimeVAE synth | +9.39 ❌ (says winner — wrong sign) |
| TimeGAN synth | −7.46 ⚠️ (correct sign, but 9× magnitude error) |

Full per-variant numbers in [reference/tstr_results.json](./reference/tstr_results.json).
Reproduce with `python examples/tstr_strategy.py`.

## Submission record

| Date | Method | Submitter | PR |
|------|--------|-----------|----|
| 2026-05-28 | Sablier-Flow | Sablier | initial |
| 2026-05-28 | KoVAE (Naiman et al. 2024) | Sablier (reference) | initial |
| 2026-05-28 | Diffusion-TS (Yuan & Qiao 2024) | Sablier (reference) | initial |
| 2026-05-28 | TimeVAE (Desai et al. 2021) | Sablier (reference) | initial |
| 2026-05-28 | TimeGAN (Yoon et al. 2019) | Sablier (reference) | initial |

## Notes on reference runs

All reference results were produced with each method's **published
default hyperparameters**, on the same panel + split + horizon. No
per-dataset tuning. See each method's `reference/<method>/seed_*/meta.json`
for exact framework version + GPU.

- Sablier-Flow: closed-source model (Sablier); synth outputs released
  here for verification.
- KoVAE, TimeVAE, Diffusion-TS, TimeGAN: open-source baselines via
  their official PyTorch implementations (KoVAE = azencot-group/KoVAE,
  TimeGAN = birdx0810/timegan-pytorch port, TimeVAE = abudesai/timeVAE,
  Diffusion-TS = Y-debug-sys/Diffusion-TS).

## How to submit

See [`GETTING_STARTED.md`](./GETTING_STARTED.md) (10-line submission
walkthrough). Full protocol in [`BENCHMARK.md`](./BENCHMARK.md).

Submissions are reviewed for protocol compliance, not for "good
enough" numbers. Every reproducible submission is accepted.
