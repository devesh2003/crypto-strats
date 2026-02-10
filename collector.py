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

NUMERIC_COLS = ["open", "high", "low", "close", "volume",
                "quote_asset_volume", "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume"]

BASE_URL = "https://data.binance.vision/data/spot/daily/klines"


def _parse_date(s: str) -> date:
    parts = s.split("-")
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


def _format_date(d: date) -> str:
    return f"{d.year}-{d.month:02d}-{d.day:02d}"


def _csv_path(symbol: str, interval: str, day: str, data_dir: str) -> str:
    return os.path.join(data_dir, f"{symbol}-{interval}-{day}.csv")


def _normalize_open_time(series: "pd.Series") -> "pd.Series":
    """
    Normalise ``open_time`` values to **microseconds** regardless of
    whether individual CSV files stored them as seconds, milliseconds,
    or microseconds.

    Binance Vision CSV dumps are *not* consistent across symbols and
    time ranges — some files use milliseconds (13 digits) while others
    use microseconds (16 digits).  This per-value normalisation handles
    any mix safely.
    """
    s = pd.to_numeric(series, errors="coerce")
    # >1e15 → already microseconds, >1e12 → milliseconds, else seconds
    us_mask = s > 1e15
    ms_mask = (s > 1e12) & ~us_mask
    s_mask = ~us_mask & ~ms_mask
    s = s.copy()
    s.loc[ms_mask] = s.loc[ms_mask] * 1_000        # ms → us
    s.loc[s_mask] = s.loc[s_mask] * 1_000_000      # s  → us
    return s


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

    Duplicates (by ``open_time``) are dropped and OHLCV columns are
    coerced to numeric so downstream code never gets string values.
    """
    pattern = os.path.join(data_dir, f"{symbol}-{interval}-*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No data found for {symbol} {interval} in {data_dir}"
        )
    dfs = [pd.read_csv(f, names=COLUMNS) for f in files]
    df = (
        pd.concat(dfs, ignore_index=True)
        .drop_duplicates(subset=["open_time"])
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    # Ensure OHLCV columns are numeric (guards against mixed-type CSVs)
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalise open_time to microseconds so all downstream code can
    # use a single unit regardless of the source CSV format.
    df["open_time"] = _normalize_open_time(df["open_time"])

    return df


def load_bt_dataframe(
    symbol: str,
    interval: str,
    data_dir: str = "Data",
) -> pd.DataFrame:
    """
    Convenience: load klines and return a DataFrame ready for
    ``bt.feeds.PandasData`` (datetime index, OHLCV columns).

    ``load()`` already normalises ``open_time`` to microseconds, so a
    single ``unit='us'`` conversion is safe here.
    """
    df = load(symbol, interval, data_dir)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="us")

    bt_df = df[["open", "high", "low", "close", "volume"]].copy()
    bt_df.index = df["datetime"]
    bt_df.index.name = "datetime"
    return bt_df
