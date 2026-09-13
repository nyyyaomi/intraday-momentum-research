from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass
class Params:
    initial_capital: float = 100_000.0
    lookback_days: int = 14
    target_daily_vol: float = 0.02
    max_leverage: float = 4.0
    commission_per_share: float = 0.0035
    slippage_per_share: float = 0.001
    volatility_multiplier: float = 1.0
    first_check: str = "10:00"
    last_check: str = "15:30"
    close_time: str = "16:00"

def load_minute_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    required = ["datetime", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}. Need {required}")

    df = df.rename(columns={cols[c]: c for c in required})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    if getattr(df["datetime"].dt, "tz", None) is not None:
        df["datetime"] = df["datetime"].dt.tz_convert("America/New_York").dt.tz_localize(None)

    df = df.set_index("datetime")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    df = df.between_time("09:30", "16:00")
    df["date"] = df.index.date
    df["time"] = df.index.strftime("%H:%M")
    return df

def prepare_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby("date").agg(
        open=("open", "first"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    daily["ret"] = daily["close"].pct_change()
    return daily

def compute_noise_bands(df: pd.DataFrame, daily: pd.DataFrame, p: Params) -> pd.DataFrame:
    x = df.copy()

    day_open = daily["open"].to_dict()
    x["day_open"] = x["date"].map(day_open)
    x["move_from_open_abs"] = (x["close"] / x["day_open"] - 1.0).abs()

    move_pivot = x.pivot_table(index="date", columns="time", values="move_from_open_abs", aggfunc="last")
    sigma = move_pivot.shift(1).rolling(p.lookback_days, min_periods=p.lookback_days).mean()

    sigma_long = sigma.stack().rename("sigma_intraday").reset_index()
    x = x.reset_index().merge(sigma_long, on=["date", "time"], how="left").set_index("datetime")

    prev_close = daily["close"].shift(1).to_dict()
    x["prev_close"] = x["date"].map(prev_close)

    base_upper = np.maximum(x["day_open"], x["prev_close"])
    base_lower = np.minimum(x["day_open"], x["prev_close"])
    x["upper_band"] = base_upper * (1.0 + p.volatility_multiplier * x["sigma_intraday"])
    x["lower_band"] = base_lower * (1.0 - p.volatility_multiplier * x["sigma_intraday"])

    x["pv"] = x["close"] * x["volume"]
    x["cum_pv"] = x.groupby("date")["pv"].cumsum()
    x["cum_volume"] = x.groupby("date")["volume"].cumsum()
    x["vwap"] = x["cum_pv"] / x["cum_volume"]

    rolling_vol = daily["ret"].shift(1).rolling(p.lookback_days, min_periods=p.lookback_days).std()
    x["daily_vol"] = x["date"].map(rolling_vol.to_dict())
    x["leverage"] = np.minimum(p.max_leverage, p.target_daily_vol / x["daily_vol"])
    x["leverage"] = x["leverage"].replace([np.inf, -np.inf], np.nan).clip(lower=0)

    return x

def check_times(p: Params) -> set[str]:
    times = pd.date_range("10:00", "15:30", freq="30min").strftime("%H:%M").tolist()
    return {t for t in times if p.first_check <= t <= p.last_check}

def backtest_one_ticker(df: pd.DataFrame, ticker: str, p: Params) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    daily = prepare_daily(df)
    x = compute_noise_bands(df, daily, p)
    checks = check_times(p)

    aum = p.initial_capital
    pos = 0
    trades = []
    day_rows = []

    grouped = x.groupby("date", sort=True)

    for date, day in grouped:
        if len(day) == 0:
            continue

        day_start_aum = aum
        day_start_trades = len(trades)
        valid = day.dropna(subset=["upper_band", "lower_band", "vwap", "daily_vol", "leverage"])
        if valid.empty:
            day_rows.append({"date": date, "aum": aum, "daily_ret": 0.0, "traded": False})
            continue

        open_px = float(day.iloc[0]["open"])
        leverage = float(valid.iloc[0]["leverage"])
        target_shares = int(np.floor((aum * leverage) / open_px)) if open_px > 0 else 0

        pos = 0
        current_entry_px = None
        current_entry_time = None
        entry_costs = 0.0

        last_ts = day.index[-1]

        for ts, row in day.iterrows():
            t = row["time"]

            if t in checks:
                px = float(row["close"])
                ub = float(row["upper_band"])
                lb = float(row["lower_band"])
                vwap = float(row["vwap"])

                exit_now = False
                if pos > 0 and px <= max(ub, vwap):
                    exit_now = True
                elif pos < 0 and px >= min(lb, vwap):
                    exit_now = True

                if exit_now and pos != 0:
                    shares = abs(pos)
                    side = np.sign(pos)
                    gross_pnl = side * shares * (px - current_entry_px)
                    costs = shares * (p.commission_per_share + p.slippage_per_share)
                    aum += gross_pnl - costs
                    trades.append({
                        "ticker": ticker,
                        "entry_time": current_entry_time,
                        "exit_time": ts,
                        "side": "long" if side > 0 else "short",
                        "shares": shares,
                        "entry_price": current_entry_px,
                        "exit_price": px,
                        "gross_pnl": gross_pnl,
                        "costs": entry_costs + costs,
                        "net_pnl": gross_pnl - entry_costs - costs,
                    })
                    pos = 0
                    current_entry_px = None
                    current_entry_time = None

                if pos == 0 and target_shares > 0:
                    if px > ub:
                        shares = target_shares
                        costs = shares * (p.commission_per_share + p.slippage_per_share)
                        aum -= costs
                        entry_costs = costs
                        pos = shares
                        current_entry_px = px
                        current_entry_time = ts
                    elif px < lb:
                        shares = target_shares
                        costs = shares * (p.commission_per_share + p.slippage_per_share)
                        aum -= costs
                        entry_costs = costs
                        pos = -shares
                        current_entry_px = px
                        current_entry_time = ts

            if (t == p.close_time or ts == last_ts) and pos != 0:
                px = float(row["close"])
                shares = abs(pos)
                side = np.sign(pos)
                gross_pnl = side * shares * (px - current_entry_px)
                costs = shares * (p.commission_per_share + p.slippage_per_share)
                aum += gross_pnl - costs
                trades.append({
                    "ticker": ticker,
                    "entry_time": current_entry_time,
                    "exit_time": ts,
                    "side": "long" if side > 0 else "short",
                    "shares": shares,
                    "entry_price": current_entry_px,
                    "exit_price": px,
                    "gross_pnl": gross_pnl,
                    "costs": entry_costs + costs,
                    "net_pnl": gross_pnl - entry_costs - costs,
                })
                pos = 0
                current_entry_px = None
                current_entry_time = None

        daily_ret = (aum / day_start_aum - 1.0) if day_start_aum else 0.0
        day_rows.append({"date": date, "aum": aum, "daily_ret": daily_ret, "traded": len(trades) > day_start_trades})

    daily_results = pd.DataFrame(day_rows)
    trades_df = pd.DataFrame(trades, columns=[
        "ticker", "entry_time", "exit_time", "side", "shares", "entry_price",
        "exit_price", "gross_pnl", "costs", "net_pnl",
    ])
    metrics = summarize(daily_results, trades_df, p)
    metrics["ticker"] = ticker
    return daily_results, trades_df, metrics

def summarize(daily_results: pd.DataFrame, trades: pd.DataFrame, p: Params) -> dict:
    if daily_results.empty:
        return {}

    rets = daily_results["daily_ret"].fillna(0.0)
    total_return = daily_results["aum"].iloc[-1] / p.initial_capital - 1.0

    days = len(rets)
    years = days / 252.0
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 and total_return > -1 else np.nan
    vol = rets.std(ddof=1) * np.sqrt(252) if len(rets) > 1 else np.nan
    sharpe = (rets.mean() * 252) / vol if vol and vol > 0 else np.nan
    hit_rate = (rets > 0).mean()

    equity = daily_results["aum"]
    drawdown = equity / equity.cummax().clip(lower=p.initial_capital) - 1.0
    mdd = drawdown.min()

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "max_drawdown": mdd,
        "num_trades": len(trades),
        "avg_trades_per_day": len(trades) / max(days, 1),
        "final_aum": daily_results["aum"].iloc[-1],
        "days": days,
    }

def capacity_estimate(df: pd.DataFrame, ticker: str, aum_values=None, p: Params = Params()) -> pd.DataFrame:
    if aum_values is None:
        aum_values = [10e6, 25e6, 50e6, 100e6, 250e6, 500e6, 1e9]

    x = df.copy()
    x["dollar_volume"] = x["close"] * x["volume"]
    x["interval_dollar_volume"] = x.groupby("date")["dollar_volume"].rolling(30, min_periods=20).sum().reset_index(level=0, drop=True)
    checks = check_times(p)
    sample = x[x["time"].isin(checks)]["interval_dollar_volume"].dropna()

    rows = []
    median_interval_dv = sample.median()
    p10_interval_dv = sample.quantile(0.10)

    for aum in aum_values:
        full_flip_notional = 2 * aum * p.max_leverage
        rows.append({
            "ticker": ticker,
            "aum": aum,
            "full_flip_notional_at_4x": full_flip_notional,
            "median_interval_dollar_volume": median_interval_dv,
            "p10_interval_dollar_volume": p10_interval_dv,
            "flip_pct_of_median_interval_volume": full_flip_notional / median_interval_dv if median_interval_dv else np.nan,
            "flip_pct_of_p10_interval_volume": full_flip_notional / p10_interval_dv if p10_interval_dv else np.nan,
        })

    rows.append({
        "ticker": ticker,
        "aum": "1% median threshold",
        "full_flip_notional_at_4x": 0.01 * median_interval_dv,
        "median_interval_dollar_volume": median_interval_dv,
        "p10_interval_dollar_volume": p10_interval_dv,
        "flip_pct_of_median_interval_volume": 0.01,
        "flip_pct_of_p10_interval_volume": np.nan,
    })
    rows.append({
        "ticker": ticker,
        "aum": "1% p10 threshold",
        "full_flip_notional_at_4x": 0.01 * p10_interval_dv,
        "median_interval_dollar_volume": median_interval_dv,
        "p10_interval_dollar_volume": p10_interval_dv,
        "flip_pct_of_median_interval_volume": np.nan,
        "flip_pct_of_p10_interval_volume": 0.01,
    })

    out = pd.DataFrame(rows)
    mask = out["aum"].astype(str).str.contains("threshold")
    out.loc[mask, "aum_threshold_at_4x"] = out.loc[mask, "full_flip_notional_at_4x"] / (2 * p.max_leverage)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".", help="Folder containing SPY_1min.csv and QQQ_1min.csv")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ"])
    args = parser.parse_args()

    p = Params()
    data_dir = Path(args.data_dir)

    metrics = []
    capacity_tables = []

    for ticker in args.tickers:
        path = data_dir / f"{ticker}_1min.csv"
        print(f"Running {ticker} from {path}...")
        df = load_minute_csv(path)
        daily_results, trades, m = backtest_one_ticker(df, ticker, p)
        metrics.append(m)

        daily_results.to_csv(data_dir / f"{ticker}_daily_results.csv", index=False)
        trades.to_csv(data_dir / f"{ticker}_trades.csv", index=False)

        cap = capacity_estimate(df, ticker, p=p)
        capacity_tables.append(cap)

    summary = pd.DataFrame(metrics).set_index("ticker")
    ordered_cols = ["total_return", "cagr", "annual_vol", "sharpe", "hit_rate", "max_drawdown", "num_trades", "avg_trades_per_day", "final_aum", "days"]
    summary = summary[ordered_cols]
    summary.to_csv(data_dir / "strategy_comparison_summary.csv")

    capacity = pd.concat(capacity_tables, ignore_index=True)
    capacity.to_csv(data_dir / "capacity_estimate.csv", index=False)

    print("\n=== Strategy comparison ===")
    print(summary.to_string(float_format=lambda x: f"{x:,.4f}"))

    print("\nSaved:")
    print("- strategy_comparison_summary.csv")
    print("- capacity_estimate.csv")
    print("- <TICKER>_daily_results.csv")
    print("- <TICKER>_trades.csv")

if __name__ == "__main__":
    main()
