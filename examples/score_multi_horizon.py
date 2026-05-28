"""Score multi-horizon synth runs with finval.

The companion to a multi-horizon submission. Expects this layout under
``reference/multi_horizon/``:

    reference/multi_horizon/<method>/L<H>/seed_<k>/synth_paths.npy
    reference/multi_horizon/<method>/L<H>/seed_<k>/real_paths.npy

Runs ``finval.validate_paths`` on every (method, horizon, seed) cell,
aggregates to mean ± std across seeds, and writes a roll-up JSON.

Multi-horizon is the killer experiment for any horizon-agnostic
generator: a single training at the reference horizon (L=60 for
FinBench v1) samples correctly at any other horizon without
retraining. Fixed-seq_len architectures (VAEs, GANs) instead have
to train separately at each horizon.

Usage:

    python examples/score_multi_horizon.py
    python examples/score_multi_horizon.py --root reference/multi_horizon \\
        --horizons 24 60 120 252 \\
        --out reference/multi_horizon/multi_horizon_scores.json
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

import finval
import numpy as np


def _json_safe(v) -> float | None:
    """Convert finval metric value to a JSON-serialisable scalar."""
    if v is None:
        return None
    fv = float(v)
    if not np.isfinite(fv):
        return None
    return fv


def score_cell(syn_path: Path, real_path: Path) -> dict:
    syn = np.load(syn_path)
    real = np.load(real_path)
    rep = finval.validate_paths(syn, real)
    return {
        "overall_score": float(rep.overall_score),
        "pass_rate": float(rep.pass_rate),
        "metrics": {nm: _json_safe(m.value) for nm, m in rep.metrics.items()},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("reference/multi_horizon"))
    p.add_argument("--horizons", type=int, nargs="+", default=[24, 60, 120, 252])
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    root: Path = args.root
    out_path: Path = args.out or (root / "multi_horizon_scores.json")

    rows = []
    for method_dir in sorted(root.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        print(f"\n=== {method} ===")
        for L in args.horizons:
            L_dir = method_dir / f"L{L}"
            if not L_dir.exists():
                continue
            seed_scores = []
            for sd in sorted(L_dir.glob("seed_*")):
                syn_path = sd / "synth_paths.npy"
                real_path = sd / "real_paths.npy"
                if not (syn_path.exists() and real_path.exists()):
                    continue
                score = score_cell(syn_path, real_path)
                score["seed"] = int(sd.name.split("_", 1)[1])
                seed_scores.append(score)
            if not seed_scores:
                continue
            score_vals = [s["overall_score"] for s in seed_scores]
            pass_vals = [s["pass_rate"] for s in seed_scores]
            row = {
                "method": method,
                "horizon": L,
                "n_seeds": len(seed_scores),
                "overall_score_mean": stats.mean(score_vals),
                "overall_score_std": (
                    stats.stdev(score_vals) if len(score_vals) > 1 else 0.0
                ),
                "pass_rate_mean": stats.mean(pass_vals),
                "pass_rate_std": (
                    stats.stdev(pass_vals) if len(pass_vals) > 1 else 0.0
                ),
                "seeds": seed_scores,
            }
            rows.append(row)
            print(
                f"  L={L:3d}  score={row['overall_score_mean']:.3f} ± "
                f"{row['overall_score_std']:.3f}  "
                f"pass={row['pass_rate_mean']:.2f}  ({len(seed_scores)} seeds)"
            )

    out_path.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
