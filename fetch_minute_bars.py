from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


NY_TZ = "America/New_York"
OUTPUT_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]


def parse_date(value: str | None, fallback: pd.Timestamp) -> pd.Timestamp:
    if not value:
        return fallback
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY_TZ)
    else:
        ts = ts.tz_convert(NY_TZ)
    return ts


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"])
    df["datetime"] = df["datetime"].dt.tz_convert(NY_TZ).dt.tz_localize(None)
    df = df[OUTPUT_COLUMNS]

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=OUTPUT_COLUMNS).drop_duplicates(subset=["datetime"])
    return df.sort_values("datetime")


def merge_with_cache(path: Path, fresh: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    if path.exists():
        cached = pd.read_csv(path)
        if not cached.empty:
            pieces.append(cached)
    if not fresh.empty:
        pieces.append(fresh)

    if not pieces:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    out = pd.concat(pieces, ignore_index=True)
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"])
    out = out.sort_values("datetime")
    return out[OUTPUT_COLUMNS]


def next_start_from_cache(path: Path, requested_start: pd.Timestamp) -> pd.Timestamp:
    if not path.exists():
        return requested_start

    try:
        cached = pd.read_csv(path, usecols=["datetime"])
    except Exception:
        return requested_start

    if cached.empty:
        return requested_start

    last = pd.to_datetime(cached["datetime"], errors="coerce").max()
    if pd.isna(last):
        return requested_start

    if last.tzinfo is None:
        last = last.tz_localize(NY_TZ)
    else:
        last = last.tz_convert(NY_TZ)
    return max(requested_start, last + pd.Timedelta(minutes=1))


def save_symbol(path: Path, fresh: pd.DataFrame) -> None:
    merged = merge_with_cache(path, fresh)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    print(f"Saved {len(merged):,} rows -> {path}", flush=True)


def fetch_ibkr_symbol(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    args: argparse.Namespace,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    try:
        from ib_async import IB, Stock, util
        from ib_async.ib import StartupFetch
    except ImportError as exc:
        raise SystemExit(
            "ib_async is not installed. Install it in your Python environment with: "
            "python -m pip install ib_async"
        ) from exc

    ib = IB()
    print(f"Connecting to IBKR at {args.host}:{args.port} clientId={args.client_id}...", flush=True)
    ib.connect(
        args.host,
        args.port,
        clientId=args.client_id,
        timeout=args.timeout,
        readonly=True,
        fetchFields=StartupFetch(0),
    )
    ib.RequestTimeout = args.timeout

    try:
        contract = Stock(symbol, args.exchange, "USD", primaryExchange=args.primary_exchange or "")
        ib.qualifyContracts(contract)

        frames = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + pd.Timedelta(days=args.ib_chunk_days), end)
            end_str = chunk_end.tz_convert("UTC").strftime("%Y%m%d-%H:%M:%S")
            print(f"{symbol}: requesting {cursor} -> {chunk_end}", flush=True)
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_str,
                durationStr=f"{args.ib_chunk_days} D",
                barSizeSetting="1 min",
                whatToShow=args.what_to_show,
                useRTH=not args.include_extended_hours,
                formatDate=2,
                keepUpToDate=False,
            )
            frame = util.df(bars)
            if frame is not None and not frame.empty:
                frame = frame.rename(
                    columns={
                        "date": "datetime",
                        "open": "open",
                        "high": "high",
                        "low": "low",
                        "close": "close",
                        "volume": "volume",
                    }
                )
                frame = normalize_bars(frame)
                start_naive = cursor.tz_convert(NY_TZ).tz_localize(None)
                end_naive = chunk_end.tz_convert(NY_TZ).tz_localize(None)
                frame = frame[(frame["datetime"] >= start_naive) & (frame["datetime"] < end_naive)]
                frames.append(frame)
                print(f"{symbol}: received {len(frame):,} rows", flush=True)
                if cache_path is not None and not frame.empty:
                    save_symbol(cache_path, frame)
            else:
                print(f"{symbol}: received 0 rows", flush=True)

            cursor = chunk_end
            if cursor < end and args.pause_seconds > 0:
                time.sleep(args.pause_seconds)

        if not frames:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["datetime"])
    finally:
        ib.disconnect()


def alpaca_key(name: str) -> str | None:
    aliases = {
        "key": ["APCA_API_KEY_ID", "ALPACA_API_KEY_ID", "ALPACA_KEY_ID"],
        "secret": ["APCA_API_SECRET_KEY", "ALPACA_API_SECRET_KEY", "ALPACA_SECRET_KEY"],
    }[name]
    for alias in aliases:
        value = os.environ.get(alias)
        if value:
            return value
    return None


def fetch_alpaca_symbol(symbol: str, start: pd.Timestamp, end: pd.Timestamp, args: argparse.Namespace) -> pd.DataFrame:
    key = alpaca_key("key")
    secret = alpaca_key("secret")
    if not key or not secret:
        raise SystemExit(
            "Alpaca credentials are not set. Export APCA_API_KEY_ID and APCA_API_SECRET_KEY "
            "or the ALPACA_* equivalents, then rerun."
        )

    endpoint = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    token = None
    rows = []
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }

    while True:
        params = {
            "timeframe": "1Min",
            "start": start.tz_convert(timezone.utc).isoformat(),
            "end": end.tz_convert(timezone.utc).isoformat(),
            "limit": args.alpaca_limit,
            "adjustment": args.adjustment,
            "feed": args.feed,
        }
        if token:
            params["page_token"] = token

        url = f"{endpoint}?{urlencode(params)}"
        request = Request(url, headers=headers)
        with urlopen(request, timeout=args.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))

        bars = payload.get("bars", [])
        rows.extend(
            {
                "datetime": bar["t"],
                "open": bar["o"],
                "high": bar["h"],
                "low": bar["l"],
                "close": bar["c"],
                "volume": bar["v"],
            }
            for bar in bars
        )
        print(f"{symbol}: received {len(bars):,} rows on this page", flush=True)

        token = payload.get("next_page_token")
        if not token:
            break

    return normalize_bars(pd.DataFrame(rows))


def main() -> int:
    today = pd.Timestamp.now(tz=NY_TZ).normalize()
    default_start = today - pd.DateOffset(years=3)

    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["ibkr", "alpaca"], default="ibkr")
    parser.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"])
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--start", help="Start date, for example 2023-01-01. Default: 3 years ago.")
    parser.add_argument("--end", help="End date. Default: today.")
    parser.add_argument("--force", action="store_true", help="Ignore any cached file and refetch from --start.")

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=27)
    parser.add_argument("--exchange", default="SMART")
    parser.add_argument("--primary-exchange", default="ARCA")
    parser.add_argument("--what-to-show", default="TRADES")
    parser.add_argument("--ib-chunk-days", type=int, default=30)
    parser.add_argument("--pause-seconds", type=float, default=11.0)
    parser.add_argument("--include-extended-hours", action="store_true")

    parser.add_argument("--feed", default="iex", choices=["iex", "sip"])
    parser.add_argument("--adjustment", default="raw", choices=["raw", "split", "dividend", "all"])
    parser.add_argument("--alpaca-limit", type=int, default=10000)
    parser.add_argument("--timeout", type=float, default=30.0)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    start = parse_date(args.start, default_start)
    end = parse_date(args.end, today + pd.Timedelta(days=1))

    for symbol in args.symbols:
        path = data_dir / f"{symbol.upper()}_1min.csv"
        symbol_start = start if args.force else next_start_from_cache(path, start)
        if symbol_start >= end:
            print(f"{symbol}: cache is already current through requested end date.", flush=True)
            continue

        if args.provider == "ibkr":
            fresh = fetch_ibkr_symbol(symbol.upper(), symbol_start, end, args, path)
        else:
            fresh = fetch_alpaca_symbol(symbol.upper(), symbol_start, end, args)

        save_symbol(path, fresh)

    return 0


if __name__ == "__main__":
    sys.exit(main())
