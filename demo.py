"""Deterministic synthetic-data demonstration. No market data or API access."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from intraday_momentum_compare_clean import (
    Params, backtest_one_ticker, capacity_estimate, load_minute_csv,
)


def synthetic_bars(days: int = 60, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    previous_close = 100.0
    for day in pd.bdate_range("2024-01-02", periods=days):
        stamps = pd.date_range(day + pd.Timedelta(hours=9, minutes=30), periods=390, freq="min")
        session_open = previous_close * np.exp(rng.normal(0, 0.003))
        drift = rng.normal(0, 0.000035)
        returns = rng.normal(drift, 0.00065, len(stamps))
        close = session_open * np.exp(np.cumsum(returns))
        open_ = np.r_[session_open, close[:-1]]
        spread = rng.uniform(0.0001, 0.0007, len(stamps))
        frames.append(pd.DataFrame({
            "datetime": stamps,
            "open": open_,
            "high": np.maximum(open_, close) * (1 + spread),
            "low": np.minimum(open_, close) * (1 - spread),
            "close": close,
            "volume": rng.integers(10_000, 200_000, len(stamps)),
        }))
        previous_close = close[-1]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    if args.days < 17:
        parser.error("--days must be at least 17 to exceed the volatility warm-up")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = Params()
    fig, ax = plt.subplots(figsize=(11, 5.4), layout="constrained")
    fig.set_facecolor("#f7f9fb")
    ax.set_facecolor("#f7f9fb")
    metrics, capacities = [], []
    for ticker, seed, color in (("SIM_A", 7, "#168875"), ("SIM_B", 19, "#c04d70")):
        path = args.output_dir / f"{ticker}_1min.csv"
        synthetic_bars(args.days, seed).to_csv(path, index=False)
        bars = load_minute_csv(path)
        daily, trades, summary = backtest_one_ticker(bars, ticker, params)
        daily.to_csv(args.output_dir / f"{ticker}_daily_results.csv", index=False)
        trades.to_csv(args.output_dir / f"{ticker}_trades.csv", index=False)
        np.testing.assert_allclose(
            params.initial_capital + trades["net_pnl"].sum(), daily["aum"].iloc[-1],
            rtol=1e-10, atol=1e-7,
        )
        metrics.append(summary)
        capacities.append(capacity_estimate(bars, ticker, p=params))
        ax.plot(np.arange(1, len(daily) + 1), daily["aum"] / params.initial_capital,
                label=f"{ticker} (seed {seed})", color=color, linewidth=2.2)

    pd.DataFrame(metrics).to_csv(args.output_dir / "strategy_comparison_summary.csv", index=False)
    pd.concat(capacities, ignore_index=True).to_csv(args.output_dir / "capacity_estimate.csv", index=False)
    ax.axhline(1, color="#7f8990", linewidth=0.9, linestyle="--")
    ax.axvspan(1, params.lookback_days + 1, color="#b8c2ce", alpha=0.18, label="Warm-up")
    ax.set(title="Intraday Momentum Research | Synthetic Data Demo",
           xlabel="Simulated business-day session", ylabel="Equity / initial capital")
    ax.set_xlim(1, args.days)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.suptitle("SIMULATED PRICES - NOT HISTORICAL RETURNS OR A FORECAST", fontsize=10, color="#596774")
    fig.savefig(args.output_dir / "equity.png", dpi=160)
    plt.close(fig)
    print(pd.DataFrame(metrics)[["ticker", "num_trades", "final_aum", "max_drawdown"]].to_string(index=False))
    print(f"Synthetic demo complete: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
