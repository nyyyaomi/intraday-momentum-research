from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from demo import synthetic_bars
from fetch_minute_bars import normalize_bars, merge_with_cache
from intraday_momentum_compare_clean import (
    Params, backtest_one_ticker, capacity_estimate, compute_noise_bands,
    load_minute_csv, prepare_daily, summarize,
)


def signal_fixture(closes):
    times = ["09:30", "10:00", "10:30", "11:00", "16:00"]
    index = pd.DatetimeIndex([f"2024-02-01 {t}" for t in times], name="datetime")
    frame = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 96.0,
                          "close": closes, "volume": 10000}, index=index)
    frame["date"], frame["time"] = index.date, index.strftime("%H:%M")
    for column, value in {"upper_band": 101.0, "lower_band": 99.0,
                          "vwap": 100.0, "daily_vol": 0.02, "leverage": 1.0}.items():
        frame[column] = value
    return frame


class BacktestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def load(self, frame):
        path = self.folder / "bars.csv"
        frame.to_csv(path, index=False)
        return load_minute_csv(path)

    def test_long_and_short_round_trip_costs_reconcile(self):
        frame = signal_fixture([100, 102, 100, 98, 100])
        with patch("intraday_momentum_compare_clean.compute_noise_bands", return_value=frame):
            daily, trades, summary = backtest_one_ticker(frame, "TEST", Params())
        self.assertEqual(trades["side"].tolist(), ["long", "short"])
        np.testing.assert_allclose(trades["gross_pnl"], [-2000, -2000])
        np.testing.assert_allclose(trades["costs"], [9, 9])
        self.assertAlmostEqual(trades["net_pnl"].sum(), daily["aum"].iloc[-1] - 100000)
        self.assertAlmostEqual(summary["final_aum"], 95982)
        self.assertTrue((trades["entry_time"].dt.date == trades["exit_time"].dt.date).all())

    def test_break_even_day_is_still_marked_traded(self):
        frame = signal_fixture([100, 102, 102, 102, 102])
        params = Params(commission_per_share=0, slippage_per_share=0)
        with patch("intraday_momentum_compare_clean.compute_noise_bands", return_value=frame):
            daily, trades, _ = backtest_one_ticker(frame, "TEST", params)
        self.assertEqual(len(trades), 1)
        self.assertEqual(daily["daily_ret"].iloc[0], 0)
        self.assertTrue(daily["traded"].iloc[0])

    def test_flat_prices_produce_no_trades_and_export_headers(self):
        raw = synthetic_bars(20)
        raw[["open", "high", "low", "close"]] = 100.0
        daily, trades, summary = backtest_one_ticker(self.load(raw), "FLAT", Params())
        self.assertTrue(trades.empty)
        self.assertEqual(summary["final_aum"], 100000)
        self.assertTrue((daily["daily_ret"] == 0).all())
        path = self.folder / "trades.csv"
        trades.to_csv(path, index=False)
        self.assertIn("net_pnl", pd.read_csv(path).columns)

    def test_current_day_prices_do_not_change_lagged_estimates(self):
        bars = self.load(synthetic_bars(22))
        params = Params()
        before = compute_noise_bands(bars, prepare_daily(bars), params)
        changed = bars.copy()
        last_day = changed["date"].max()
        changed.loc[changed["date"] == last_day, "close"] *= 1.2
        after = compute_noise_bands(changed, prepare_daily(changed), params)
        columns = ["sigma_intraday", "daily_vol", "leverage", "upper_band", "lower_band"]
        pd.testing.assert_frame_equal(before[columns], after[columns])
        self.assertTrue((before["leverage"].dropna() <= params.max_leverage).all())
        prior_days = before["date"] < last_day
        pd.testing.assert_frame_equal(before.loc[prior_days], after.loc[prior_days])

    def test_drawdown_includes_initial_capital(self):
        daily = pd.DataFrame({"aum": [90000, 95000], "daily_ret": [-0.1, 95000/90000 - 1]})
        metrics = summarize(daily, pd.DataFrame(), Params())
        self.assertAlmostEqual(metrics["max_drawdown"], -0.1)

    def test_full_synthetic_pipeline_reconciles(self):
        bars = self.load(synthetic_bars(24))
        params = Params()
        daily, trades, summary = backtest_one_ticker(bars, "SIM", params)
        self.assertGreater(len(trades), 0)
        self.assertAlmostEqual(params.initial_capital + trades["net_pnl"].sum(), summary["final_aum"], places=6)
        self.assertEqual(len(daily), 24)
        self.assertTrue((trades["shares"] > 0).all())

    def test_capacity_ratios_scale_with_hypothetical_aum(self):
        bars = self.load(synthetic_bars(2))
        cap = capacity_estimate(bars, "SIM", aum_values=[1e6, 2e6], p=Params(max_leverage=2))
        self.assertEqual(cap.iloc[0]["full_flip_notional_at_4x"], 4e6)
        self.assertAlmostEqual(cap.iloc[1]["flip_pct_of_median_interval_volume"],
                               2 * cap.iloc[0]["flip_pct_of_median_interval_volume"])

    def test_provider_normalization_handles_dst_and_duplicates(self):
        raw = pd.DataFrame({
            "datetime": ["2024-01-02T14:30:00Z", "2024-07-02T13:30:00Z", "2024-07-02T13:30:00Z"],
            "open": [100] * 3, "high": [101] * 3, "low": [99] * 3,
            "close": [100] * 3, "volume": [10000] * 3,
        })
        normalized = normalize_bars(raw)
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized["datetime"].dt.strftime("%H:%M").tolist(), ["09:30", "09:30"])
        path = self.folder / "cache.csv"
        normalized.to_csv(path, index=False)
        merged = merge_with_cache(path, normalized)
        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()
