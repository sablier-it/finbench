# FinBench v1 — Leaderboard

Multivariate financial time-series generation on the
`us_equities_macro_2010_2024` panel (IWM / QQQ / SPY / TLT / VIX /
TNX / DXY), 2010-2019 train, 2020-2023 OOS, L=60, 200 paths,
5 seeds. Scored with [finval](https://github.com/sablier-ai/finval) v0.1.0.

> Lower is better for every individual metric. **Bold** = current
> best on that metric. ✨ excellent · ✓ good · ⚪ acceptable · ❌ poor.

## Overall ranking

Two metrics, deliberately not aggregated. **finval** measures
distributional properties of the synth (14 stylized-fact metrics).
**TSTR** measures downstream utility — does fitting a strategy on
synth pick winners on real markets? See [BENCHMARK.md §5](./BENCHMARK.md#5-required-secondary-metric--tstr-train-on-synth-test-on-real)
for the strategy family and acceptance bands.

| Rank | Method | finval quality | finval score | finval pass | TSTR ρ | TSTR \|Δ Sharpe\| |
|-----:|--------|----------------|-------------:|------------:|-------:|----------------:|
| **1** | **Sablier-Flow** (Sablier, 2026) | **good** ✓ | **0.794 ± 0.017** | **13 / 14** | **+0.850** ✓ | **0.361** |
| 2 | KoVAE (Naiman et al., ICLR 2024) | acceptable ⚪ | 0.616 ± 0.024 | 10 / 14 | **+0.860** ✓ | 0.308 |
| 3 | Diffusion-TS (Yuan & Qiao, ICLR 2024) | acceptable ⚪ | 0.521 ± 0.000 | 10 / 14 | **−0.724** ❌ | 93.6 |
| 4 | TimeVAE (Desai et al., 2021) | poor ❌ | 0.403 ± 0.014 | 10 / 14 | **−0.788** ❌ | 6.26 |
| 5 | TimeGAN (Yoon et al., NeurIPS 2019) | poor ❌ | 0.388 ± 0.110 | 9 / 14 | +0.535 ⚠️ | 4.91 |

**Rank reasoning:** Sablier-Flow leads on finval; KoVAE edges
Sablier-Flow on TSTR (statistical tie within seed variance at
ρ = 0.85 vs 0.86); both fail TSTR on Diffusion-TS and TimeVAE
(inverted Spearman — fitting on their synth picks losing
strategies). Top-2 vs the rest is the meaningful gap.

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

## Multi-horizon generalisation (preliminary — partial KoVAE data)

Reproduce with `python examples/score_multi_horizon.py`.

**Sablier-Flow** is trained ONCE at L=60 and then sampled at
L ∈ {24, 60, 120, 252} — the generator is horizon-agnostic, so a
single training produces synth at any requested horizon.

**KoVAE** has a fixed sequence dimension baked into its encoder/decoder,
so multi-horizon requires retraining 4 separate models (one per L).
Full KoVAE numbers pending — A100 is still grinding 12 trainings.

| L (sample) | Sablier-Flow (1 training) | KoVAE (separate trainings per L) |
|-----------:|---------------------------:|---------------------------:|
| 24  | **0.717 ± 0.017** · 12 / 14 pass | 0.567 ± 0.013 · 11 / 14 pass _(3 seeds)_ |
| 60  | **0.794 ± 0.017** · 13 / 14 pass | 0.623 ± 0.031 · 11 / 14 pass _(3 seeds)_ |
| 120 | **0.754 ± 0.024** · 13 / 14 pass | _pending_ |
| 252 | **0.652 ± 0.027** · 12 / 14 pass | _pending_ |

**Sablier-Flow's quality is stable across horizons from a single
training run.** All four horizons stay at ≥ 12 / 14 pass and overall
score ≥ 0.65. Quality is best near training horizon (L=60); modestly
degrades at L=24 (−0.077) and L=252 (−0.142). The practical upshot:
running multi-horizon backtests (1-month + 3-month + 1-year strategies
on the same data) gets all horizons from one training. Methods with
fixed sequence dimensions (KoVAE, TimeVAE) require N trainings for
N horizons — and at the two horizons we already have for KoVAE,
Sablier-Flow's single training beats KoVAE's purpose-trained models
by **+0.15** (L=24) and **+0.17** (L=60).

KoVAE's L=120 / L=252 numbers will land once the in-flight training
completes; we'll then have the full A/B comparison.

## Statistical significance

Reproduce with `python examples/statistical_tests.py`.
Full data: [`reference/statistical_tests.json`](./reference/statistical_tests.json).

We use **paired t-tests** across the 5 seeds rather than Wilcoxon
signed-rank. With n=5 paired samples, Wilcoxon's smallest achievable
two-sided p-value is 0.0625 (1 / 2⁴) — the statistic *cannot* declare
significance at p<0.05 regardless of effect size. Paired t-test (with
continuous, approximately-normal finval scores across seeds) gives us
the power 5 seeds can support.

### Per-metric significance — Sablier-Flow vs each baseline

| Sablier-Flow vs | Metrics where SF lower-is-better wins | Metrics where the win is significant at p<0.05 |
|---|---:|---:|
| KoVAE        | 11 / 14 | **11 / 14** |
| Diffusion-TS | 14 / 14 | **13 / 14** |
| TimeVAE      | 14 / 14 | **14 / 14** |
| TimeGAN      | 14 / 14 | **13 / 14** |

Sablier-Flow's finval advantage is **statistically significant on
the overwhelming majority of metrics**, even at our conservative
n=5 seeds. The single non-significant Sablier-Flow-vs-KoVAE rows are
`acf_returns` (tie) and `volatility_clustering` / `drawdown_distribution`
(KoVAE wins these by small margins on point estimates).

### TSTR — bootstrap 95% CIs on Spearman ρ

10000 resamples of the 24 strategy variants.

| Method | ρ point | 95% CI |
|---|---:|---:|
| Sablier-Flow | +0.850 | [+0.640, +0.939] |
| KoVAE        | +0.860 | [+0.656, +0.946] |
| TimeGAN      | +0.535 | [+0.091, +0.751] |
| Diffusion-TS | **−0.724** | **[−0.871, −0.414]** |
| TimeVAE      | **−0.788** | **[−0.905, −0.553]** |

**Diffusion-TS and TimeVAE's CIs are entirely below zero** — their
inverted-ranking verdict is statistically robust, not a single-run
fluke. Sablier-Flow's CI overlaps KoVAE's (statistical tie on TSTR);
both CIs lie almost entirely above the +0.5 threshold.

## Memorisation audit (diagnostic)

Reproduce with `python examples/memorization_audit.py`.
Full data: [`reference/memorization_audit.json`](./reference/memorization_audit.json).

A model that just *copies* its training data isn't a generator —
fitting strategies on its synth would be hidden overfitting on the
training set. We test this with two complementary measures:

1. **`syn_vs_oos_ratio`**: ratio of synth-to-train mean NN distance
   over OOS-to-train mean NN distance. A faithful generator with
   no memorisation should sit somewhere near the OOS baseline
   (ratio ≈ 1.0 under stationarity; lower if the OOS regime
   differs from train, which is our case — COVID + 2022 bear).
2. **`dup_fraction`**: fraction of synth paths within the 5th
   percentile of OOS-to-train distances. 5% is the chance baseline;
   higher means more crowding near training points.

| Method | syn_vs_oos ratio | Dup % | Verdict |
|---|---:|---:|---|
| Diffusion-TS | 0.877 | **0.0%** | HEALTHY (diverse — but combine with finval: diverse *and wrong*) |
| Sablier-Flow | 0.684 | 13.4% | SUSPICIOUS-MILD (passes finval + TSTR; modest training proximity) |
| TimeGAN | 0.643 |  0.1% | SUSPICIOUS-MILD (clean by dup; combine with finval failure) |
| KoVAE | 0.542 | 44.1% | SUSPICIOUS (passes finval + TSTR but hugs training) |
| **TimeVAE** | **0.298** | **100.0%** | **MEMORISATION** (every synth path inside 5th-pct of train; severe) |

**Caveat.** Our OOS slice (2020-2023) contains regime shifts (COVID,
2022 bear) that weren't in the train slice (2010-2019). Real OOS
data naturally sits farther from train than a faithful-to-train
generator's synth would. The strict ρ < 0.85 thresholds for
"SUSPICIOUS" are therefore conservative — modest distance below
the OOS baseline may simply reflect faithful capture of the
training distribution, not memorisation. The 100% TimeVAE
duplicate-fraction result is unambiguous; the 13-44% range for
Sablier-Flow / KoVAE warrants disclosure but isn't a model-killer.

**The bottom line:** among methods that pass finval + TSTR
(Sablier-Flow and KoVAE), Sablier-Flow has the lower duplicate
fraction (13% vs 44%). Only TimeVAE shows definitive
memorisation under this audit.

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
