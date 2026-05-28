"""Memorization audit — is the synth just a copy of training data?

The standard naive test ("compare synth's NN-distance-to-train against
train's self-NN-distance") is calibrated for high-dimensional sparse
data like images. For low-dimensional continuous financial returns,
that test misfires: a model that perfectly captures the training
distribution will produce synth that's NEAR training points by
construction, giving a "memorization-looking" ratio.

The right baseline is **real OOS data**. Real out-of-sample windows
are NOT memorised (they're held-out data) but they DO sit near the
training distribution. If synth has the same NN-distance-to-train
distribution as real OOS data, the synth is "as close to training
as fresh real data is" — exactly what a faithful generator should
produce.

We compute three reference distances:

  - d_train_self : NN distance within the training set (anchor)
  - d_oos_to_train : NN distance from real OOS data to training set
                     — this is the **natural baseline**
  - d_syn_to_train : same for each method's synth

Healthy generator:  d_syn_to_train ≈ d_oos_to_train
Memorisation:        d_syn_to_train ≪ d_oos_to_train
Wrong-distribution:  d_syn_to_train ≫ d_oos_to_train  (detected separately by finval)

Final score reported as ``syn_vs_oos_ratio = d_syn / d_oos``:

  - 0.85 ≤ ratio ≤ 1.15  → HEALTHY (within ±15% of OOS baseline)
  - 0.50 ≤ ratio < 0.85  → SUSPICIOUS (synth closer to train than OOS is)
  - ratio < 0.50         → MEMORISATION (synth much closer to train)
  - ratio > 1.15         → DISTRIBUTIONAL DRIFT (off the manifold; finval catches this separately)

We also report the **duplicate fraction**: % of synth paths whose
NN-distance is below the 5th percentile of the OOS-to-train NN
distribution. A perfect generator should have ≈ 5% duplicate fraction.

Usage:

    python examples/memorization_audit.py

Writes ``reference/memorization_audit.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


PANEL_NAME = "us_equities_macro_2010_2024"
HORIZON = 60
N_FEATURES = 7
FEATURE_TYPES = {
    "IWM": "price", "QQQ": "price", "SPY": "price", "TLT": "price",
    "VIX": "volatility", "TNX": "rate", "DXY": "ratio",
}


def _to_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float32)
    for c in df.columns:
        if FEATURE_TYPES[c] in ("price", "ratio", "level"):
            out[c] = np.log(df[c]).diff()
        else:
            out[c] = df[c].diff()
    return out.dropna().astype(np.float32)


def _sliding_windows(arr: np.ndarray, L: int) -> np.ndarray:
    T, F = arr.shape
    return np.stack([arr[i : i + L] for i in range(T - L + 1)], axis=0).astype(np.float32)


def build_train_and_oos_windows() -> tuple[np.ndarray, np.ndarray]:
    import sablier_flow
    df = sablier_flow.demo_data(PANEL_NAME)
    df.index = pd.to_datetime(df.index)
    train = df.loc[df.index < "2020-01-01"]
    oos = df.loc[df.index >= "2020-01-01"]
    train_r = _to_returns(train).values
    oos_r = _to_returns(oos).values
    return _sliding_windows(train_r, HORIZON), _sliding_windows(oos_r, HORIZON)


def _nn_distances(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Vectorised per-row Euclidean nearest-neighbour distance from
    ``query`` rows to ``gallery`` rows."""
    n_q = query.shape[0]
    g_norm = (gallery ** 2).sum(axis=1)
    out = np.empty(n_q, dtype=np.float32)
    chunk = 512
    for i in range(0, n_q, chunk):
        q = query[i : i + chunk]
        q_norm = (q ** 2).sum(axis=1, keepdims=True)
        d2 = q_norm + g_norm[None, :] - 2.0 * q @ gallery.T
        d2 = np.maximum(d2, 0.0)
        out[i : i + chunk] = np.sqrt(d2.min(axis=1))
    return out


def _train_self_nn_distances(train_flat: np.ndarray) -> np.ndarray:
    n = train_flat.shape[0]
    g_norm = (train_flat ** 2).sum(axis=1)
    out = np.empty(n, dtype=np.float32)
    chunk = 512
    for i in range(0, n, chunk):
        q = train_flat[i : i + chunk]
        q_norm = (q ** 2).sum(axis=1, keepdims=True)
        d2 = q_norm + g_norm[None, :] - 2.0 * q @ train_flat.T
        for k in range(d2.shape[0]):
            d2[k, i + k] = np.inf
        d2 = np.maximum(d2, 0.0)
        out[i : i + chunk] = np.sqrt(d2.min(axis=1))
    return out


def _verdict(ratio: float) -> str:
    if ratio < 0.50:
        return "MEMORISATION"
    if ratio < 0.85:
        return "SUSPICIOUS"
    if ratio <= 1.15:
        return "HEALTHY"
    return "DISTRIBUTIONAL_DRIFT"


def audit_method(
    method_dir: Path, train_flat: np.ndarray,
    oos_d_mean: float, oos_d_p5: float,
) -> dict:
    ratios = []
    syn_means = []
    dup_fracs = []
    for sd in sorted(method_dir.glob("seed_*")):
        synth = np.load(sd / "synth_paths.npy")
        synth_flat = synth.reshape(synth.shape[0], -1).astype(np.float32)
        d = _nn_distances(synth_flat, train_flat)
        syn_means.append(float(d.mean()))
        ratios.append(float(d.mean() / oos_d_mean))
        dup_fracs.append(float((d < oos_d_p5).mean()))
    return {
        "method": method_dir.name,
        "n_seeds": len(ratios),
        "d_syn_mean": float(np.mean(syn_means)),
        "d_syn_std": float(np.std(syn_means, ddof=0)),
        "syn_vs_oos_ratio_mean": float(np.mean(ratios)),
        "syn_vs_oos_ratio_std": float(np.std(ratios, ddof=0)),
        "duplicate_fraction": float(np.mean(dup_fracs)),
        "verdict": _verdict(float(np.mean(ratios))),
    }


def main() -> None:
    print("Building train (2010-2019) + OOS (2020-2023) windows …")
    train, oos = build_train_and_oos_windows()
    print(f"  train_windows shape: {train.shape}")
    print(f"  oos_windows   shape: {oos.shape}")
    train_flat = train.reshape(train.shape[0], -1).astype(np.float32)
    oos_flat = oos.reshape(oos.shape[0], -1).astype(np.float32)

    t0 = time.time()
    print("Computing train-set self-NN distance …")
    train_self_d = _train_self_nn_distances(train_flat)
    print(f"  d_train_self = mean {train_self_d.mean():.5f}  median {np.median(train_self_d):.5f}  ({time.time() - t0:.1f}s)")

    t1 = time.time()
    print("Computing OOS-to-train NN distance (natural baseline) …")
    oos_d = _nn_distances(oos_flat, train_flat)
    oos_d_mean = float(oos_d.mean())
    oos_d_p5 = float(np.percentile(oos_d, 5))
    print(f"  d_oos_to_train = mean {oos_d_mean:.5f}  5th-pct {oos_d_p5:.5f}  ({time.time() - t1:.1f}s)")
    print(f"  → OOS baseline ratio: 1.000 (perfect healthy)\n")

    rows = []
    for method in ["sablier_flow", "kovae", "diffusion_ts", "timevae", "timegan"]:
        method_dir = Path("reference") / method
        if not method_dir.exists():
            print(f"  {method}: skipped (no reference dir)")
            continue
        t2 = time.time()
        row = audit_method(method_dir, train_flat, oos_d_mean, oos_d_p5)
        rows.append(row)
        print(f"  {method:14s}  d_syn = {row['d_syn_mean']:.4f}  "
              f"ratio_vs_oos = {row['syn_vs_oos_ratio_mean']:.3f} ± {row['syn_vs_oos_ratio_std']:.3f}  "
              f"dup_frac = {row['duplicate_fraction']:.2%}  "
              f"→ {row['verdict']}  ({time.time() - t2:.1f}s)")

    out = Path("reference/memorization_audit.json")
    out.write_text(json.dumps({
        "panel": PANEL_NAME,
        "horizon": HORIZON,
        "train_period": "2010-01-04 to 2019-12-31",
        "oos_period": "2020-01-02 to 2023-12-28",
        "n_train_windows": int(train.shape[0]),
        "n_oos_windows": int(oos.shape[0]),
        "metric": "Euclidean on flattened (60*7) windows",
        "baseline": "real OOS data — d_oos_to_train is the natural reference",
        "verdict_bands": {
            "HEALTHY": "0.85 ≤ ratio ≤ 1.15 — synth as close to train as fresh real data",
            "SUSPICIOUS": "0.50 ≤ ratio < 0.85 — synth closer to train than OOS",
            "MEMORISATION": "ratio < 0.50 — synth much closer to train",
            "DISTRIBUTIONAL_DRIFT": "ratio > 1.15 — synth off the manifold",
        },
        "d_train_self_mean": float(train_self_d.mean()),
        "d_oos_to_train_mean": oos_d_mean,
        "d_oos_to_train_p5": oos_d_p5,
        "methods": rows,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
