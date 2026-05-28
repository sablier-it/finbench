# FinBench v1 panel — `us_equities_macro_2010_2024`

The v1 benchmark uses a fixed 7-feature daily panel covering 2010-2024,
shipped inside `sablier-flow`'s wheel. No download or fetch required.

```python
import sablier_flow
df = sablier_flow.demo_data("us_equities_macro_2010_2024")
# DataFrame: shape (3522, 7), DatetimeIndex 2010-01-04 → 2023-12-28.
```

The panel is **public** — sourced from yfinance / FRED / standard
public market data. No licensing required for benchmark use.

## Feature order (frozen for v1)

```
SPY  QQQ  IWM  TLT  VIX  TNX  DXY
```

## Train / OOS split (frozen for v1)

```
Train: 2010-01-04 to 2019-12-31
OOS:   2020-01-02 to 2023-12-28
```

## Reproducible reference windows

`examples/submit.py::load_real_reference` builds the 200 sliding
windows from OOS that every submission is scored against. Deterministic
across runs.
