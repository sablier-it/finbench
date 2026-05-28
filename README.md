# FinBench

**A public benchmark for multivariate financial time-series generation.**

FinBench evaluates synthetic financial time-series generators on the
metrics that actually matter for financial backtesting — fat tails,
volatility clustering, leverage effect, tail dependence, drawdown
distribution — not on the distributional-matching proxies (DS / PS /
Context-FID) that dominate general-purpose TS-gen benchmarks.

> **Why this exists.** Every popular TS-gen benchmark (TimeGAN protocol,
> TSGBench, GenTS) measures *can a classifier distinguish synth from
> real?* That metric **rewards mode collapse** and ignores everything a
> quant cares about. A model that produces smooth, low-variance synth
> can ace DS/PS while violating vol clustering, leverage effect, and
> heavy-tail behavior — exactly the failure modes that destroy
> backtests.
>
> FinBench replaces that beauty contest with **14 financial stylized
> facts** grounded in the empirical-finance literature (Cont 2001,
> Black 1976, Joe 1997, Bailey & López de Prado 2014). If your model
> passes FinBench, your synth is good enough to fit strategies on.

## Quick links

- [Leaderboard](./LEADERBOARD.md) — current rankings
- [Benchmark protocol](./BENCHMARK.md) — frozen v1 spec
- [Submit your model](./examples/submit.py) — 10-line template
- [finval (scoring library)](https://github.com/sablier-it/finval) — the
  metric implementations FinBench uses

## At a glance

| Method                  | Overall    | Score | Pass    | Vol Cluster | Leverage | Tail Q | Drawdown |
|-------------------------|:----------:|------:|--------:|------------:|---------:|-------:|---------:|
| **Sablier-Flow** (Sablier) | **good**   | **0.794** | **13/14** | 0.037 | **0.031** | **0.661** | 0.160 |
| KoVAE (ICLR'24)         | acceptable | 0.616 | 10/14 | **0.031** | 0.032 | 0.979 | **0.126** |
| Diffusion-TS (ICLR'24)  | acceptable | 0.521 | 10/14 | 0.080 | 0.055 | 38.36 | 1.000 |
| TimeVAE (arXiv'21)      | poor       | 0.403 | 10/14 | 0.115 | 0.156 | 1.977 | 0.548 |
| TimeGAN (NeurIPS'19)    | poor       | 0.388 |  9/14 | 0.127 | 0.111 | 1.094 | 0.369 |

Full numbers (mean ± std across 5 seeds) in [LEADERBOARD.md](./LEADERBOARD.md).

## Submit your model

1. Generate `(200, 60, 7)` synthetic returns for the FinBench v1 panel
   (load via `pip install sablier-flow && sablier_flow.demo_data(...)`).
2. Run `python examples/submit.py --synth your_synth.npy --name your_method`.
3. Open a PR adding the resulting `reference/<your_method>/` directory
   and a row in `LEADERBOARD.md`.

The protocol does not require you to share your model code or weights.
Only the synthetic outputs are needed to score. See
[`BENCHMARK.md`](./BENCHMARK.md) for the full submission spec.

## Versioning

FinBench v1 is **frozen**. Once a number is on the leaderboard, the
protocol it was scored against will not change. Future versions
(FinBench v2, FinBench-Intraday, …) live in separate branches with
their own leaderboards.

## License

Code: MIT. Reference results: CC-BY 4.0.
