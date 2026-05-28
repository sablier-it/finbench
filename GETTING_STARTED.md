# Getting Started — submit your method to FinBench in 10 lines

This is the **shortest path** from "I have a multivariate-TS-gen
model" to "my numbers are on the leaderboard." For the full protocol,
see [BENCHMARK.md](./BENCHMARK.md).

## Setup (one time)

```bash
git clone https://github.com/sablier-ai/finbench
cd finbench
pip install sablier-flow                                 # for the panel
pip install finval                                       # for scoring
```

## 1. Load the panel

```python
import sablier_flow
df = sablier_flow.demo_data("us_equities_macro_2010_2024")
# 3522 rows × 7 features (SPY, QQQ, IWM, TLT, VIX, TNX, DXY), daily 2010–2023.
```

## 2. Train your model on 2010-2019; generate (200, 60, 7) returns

```python
train_df = df.loc[df.index < "2020-01-01"]

# YOUR MODEL HERE — any architecture, any framework.
synth = your_model.train_and_sample(
    train_df,
    horizon=60,      # 3 trading months
    n_paths=200,     # FinBench v1 standard
)
# synth.shape must be exactly (200, 60, 7), float32 returns.
```

Per-feature transform conventions (so `synth` is in the right space):

| Feature       | Transform           |
|---------------|---------------------|
| SPY/QQQ/IWM/TLT/DXY | log-returns    |
| VIX/TNX       | first-differences   |

## 3. Score with `examples/submit.py`

For each seed (you need 5):

```bash
python examples/submit.py \
    --synth your_synth_seed0.npy \
    --name your_method \
    --seed 0 \
    --paper https://arxiv.org/abs/YOUR_PAPER
```

This runs all 14 finval metrics and writes the seed directory:

```
reference/your_method/seed_0/
    synth_paths.npy
    real_paths.npy
    meta.json
    finval_scores.json
```

## 4. Aggregate across 5 seeds

```bash
python examples/score.py --name your_method
```

Outputs `reference/your_method/finval_scores.json` (mean ± std) and
prints a ready-to-paste leaderboard row.

## 5. Required: compute TSTR score

The TSTR score (train-on-synth, test-on-real) is a **required
secondary metric** — see [BENCHMARK.md §5](./BENCHMARK.md#5-required-secondary-metric--tstr-train-on-synth-test-on-real).
It measures whether a strategy fitted on your synth picks winners
on real markets (Spearman ρ between Sharpe-on-real and
Sharpe-on-synth across 24 strategy variants).

```bash
python examples/tstr_strategy.py --method your_method
```

Outputs `reference/your_method/tstr_scores.json` with:
- `spearman_rho`: rank correlation (target: ≥ +0.80, p < 0.01)
- `mean_abs_sharpe_gap`: Sharpe magnitude error

Why required: a synth that scores well on finval can still
misrank strategies. Diffusion-TS and TimeVAE both pass finval
acceptably yet have negative TSTR ρ — fitting on their synth picks
the worst real-market strategies. Without TSTR, you'd never know.

## 6. Open a PR

```bash
git checkout -b submit-your_method
git add reference/your_method/
# Edit LEADERBOARD.md, add your row in the right rank slot.
git commit -m "Submit your_method to FinBench"
git push origin submit-your_method
gh pr create
```

We review for **protocol compliance**, not for whether your numbers
are "good enough." Every reproducible submission is accepted.

## FAQ

**Do I need to share my model code?** No. Only the synthetic outputs.

**What if my model needs a different horizon / panel size?** Submit
to FinBench v1 with the standard `(200, 60, 7)` shape. You're welcome
to submit additional configurations to future FinBench versions
(intraday, equity-only, etc.) once those land.

**What if my model needs per-dataset tuning?** Submissions with tuned
hyperparameters are accepted but marked with a `*` on the leaderboard.
Document the tuning protocol in your `meta.json`.

**Can I rerun reference baselines myself?** Yes. The reference
directory ships the seeds + raw scores; you can reproduce or verify
any number.
