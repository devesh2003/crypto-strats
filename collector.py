"""
Binance Vision klines data collector.

Downloads daily klines CSVs from https://data.binance.vision and caches
them locally so they are never re-fetched once present on disk.
"""

import glob
import os
import shutil
from datetime import date, timedelta

import pandas as pd
import requests

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
]

BASE_URL = "https://data.binance.vision/data/spot/daily/klines"


def _parse_date(s: str) -> date:
    parts = s.split("-")
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


def _format_date(d: date) -> str:
    return f"{d.year}-{d.month:02d}-{d.day:02d}"


def _csv_path(symbol: str, interval: str, day: str, data_dir: str) -> str:
    return os.path.join(data_dir, f"{symbol}-{interval}-{day}.csv")


def collect(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
    data_dir: str = "Data",
) -> None:
    """
    Download klines CSVs from Binance Vision for *symbol* at *interval*
    between *start_date* and *end_date* (YYYY-MM-DD, end exclusive).

    Already-downloaded CSVs are skipped automatically.
    """
    os.makedirs(data_dir, exist_ok=True)

    current = _parse_date(start_date)
    end = _parse_date(end_date)

    while current < end:
        day_str = _format_date(current)
        csv = _csv_path(symbol, interval, day_str, data_dir)

        if not os.path.isfile(csv):
            zip_name = f"{symbol}-{interval}-{day_str}.zip"
            zip_path = os.path.join(data_dir, zip_name)
            url = f"{BASE_URL}/{symbol}/{interval}/{zip_name}"

            try:
                r = requests.get(url, verify=False, allow_redirects=True, timeout=30)
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    f.write(r.content)
                shutil.unpack_archive(zip_path, data_dir)
            except Exception as exc:
                print(f"  [warn] {day_str}: {exc}")
            finally:
                if os.path.isfile(zip_path):
                    os.remove(zip_path)
        current += timedelta(days=1)


def load(
    symbol: str,
    interval: str,
    data_dir: str = "Data",
) -> pd.DataFrame:
    """
    Load all cached klines CSVs for *symbol*/*interval* from *data_dir* into a
    single sorted DataFrame with proper column names.
    """
    pattern = os.path.join(data_dir, f"{symbol}-{interval}-*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No data found for {symbol} {interval} in {data_dir}"
        )
    dfs = [pd.read_csv(f, names=COLUMNS) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values("open_time").reset_index(drop=True)
    return df


def load_bt_dataframe(
    symbol: str,
    interval: str,
    data_dir: str = "Data",
) -> pd.DataFrame:
    """
    Convenience: load klines and return a DataFrame ready for
    ``bt.feeds.PandasData`` (datetime index, OHLCV columns).
    """
    df = load(symbol, interval, data_dir)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="us")
    bt_df = df[["open", "high", "low", "close", "volume"]].copy()
    bt_df.index = pd.to_datetime(df["datetime"])
    bt_df.index.name = "datetime"
    return bt_df
