#!/usr/bin/env python3
"""
CLI backtesting runner.

Usage examples
--------------
Single-timeframe MA strategy on 15m ETHUSDT data:

    python backtest.py --symbol ETHUSDT --start 2025-10-01 --end 2025-12-31 \
        --signal-tf 15m --strategy strategies.MAStrategy

Multi-timeframe strategy (signals on 15m, TP/SL on 1m):

    python backtest.py --symbol ETHUSDT --start 2025-10-01 --end 2025-12-31 \
        --signal-tf 15m --granular-tf 1m \
        --strategy strategies.OversoldBounceMTFStrategy

Using a strategy from your own file:

    python backtest.py --symbol BTCUSDT --start 2025-10-01 --end 2025-12-31 \
        --signal-tf 15m --strategy my_strats.SuperAlpha
"""

import argparse
import importlib
import sys
import warnings
from datetime import datetime

import backtrader as bt

from collector import collect, load_bt_dataframe

# Suppress Binance Vision SSL warnings (self-signed cert)
warnings.filterwarnings("ignore", message="Unverified HTTPS request")


# ---------------------------------------------------------------------------
# Binance Futures-style commission
# ---------------------------------------------------------------------------
class BinanceFuturesCommInfo(bt.CommInfoBase):
    params = (
        ("stocklike", False),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("percabs", True),
        ("commission_maker", 0.0002),
        ("commission_taker", 0.0005),
        ("commission_type", "taker"),
    )

    def _getcommission(self, size, price, pseudoexec):
        if self.p.commission_type == "maker":
            rate = self.p.commission_maker
        elif self.p.commission_type == "taker":
            rate = self.p.commission_taker
        else:
            rate = (self.p.commission_maker + self.p.commission_taker) / 2
        return abs(size) * price * rate


# ---------------------------------------------------------------------------
# Timeframe mapping helpers
# ---------------------------------------------------------------------------
_TF_MAP = {
    "1m": (bt.TimeFrame.Minutes, 1),
    "3m": (bt.TimeFrame.Minutes, 3),
    "5m": (bt.TimeFrame.Minutes, 5),
    "15m": (bt.TimeFrame.Minutes, 15),
    "30m": (bt.TimeFrame.Minutes, 30),
    "1h": (bt.TimeFrame.Minutes, 60),
    "2h": (bt.TimeFrame.Minutes, 120),
    "4h": (bt.TimeFrame.Minutes, 240),
    "1d": (bt.TimeFrame.Days, 1),
}

# Bars per year for Sharpe annualization (crypto 24/7)
_TF_SHARPE_FACTOR = {
    "1m": 525_600,   # 60 * 24 * 365
    "3m": 175_200,
    "5m": 105_120,
    "15m": 35_040,   # 4 * 24 * 365
    "30m": 17_520,
    "1h": 8_760,     # 24 * 365
    "2h": 4_380,
    "4h": 2_190,
    "1d": 365,
}


def _parse_tf(label: str):
    """Return (bt.TimeFrame, compression) for a Binance-style interval."""
    if label not in _TF_MAP:
        sys.exit(f"Unknown timeframe '{label}'. Supported: {', '.join(_TF_MAP)}")
    return _TF_MAP[label]


# ---------------------------------------------------------------------------
# Dynamic strategy loader
# ---------------------------------------------------------------------------
def _load_strategy(dotted_path: str):
    """
    Import a strategy class from a dotted ``module.ClassName`` path.

    Examples:
        strategies.MAStrategy        -> from strategies import MAStrategy
        my_pkg.strats.Alpha          -> from my_pkg.strats import Alpha
    """
    if "." not in dotted_path:
        sys.exit(
            f"Strategy must be 'module.ClassName' (e.g. strategies.MAStrategy), "
            f"got '{dotted_path}'"
        )
    module_path, class_name = dotted_path.rsplit(".", 1)
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError:
        sys.exit(f"Cannot import module '{module_path}'")
    cls = getattr(mod, class_name, None)
    if cls is None:
        sys.exit(f"Module '{module_path}' has no class '{class_name}'")
    if not (isinstance(cls, type) and issubclass(cls, bt.Strategy)):
        sys.exit(f"'{dotted_path}' is not a bt.Strategy subclass")
    return cls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run a backtrader backtest with Binance Vision data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data / collection
    parser.add_argument("--symbol", required=True, help="Trading pair, e.g. ETHUSDT")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD (exclusive)")
    parser.add_argument("--data-dir", default="Data", help="Local data cache directory (default: Data)")

    # Timeframes
    parser.add_argument(
        "--signal-tf", required=True,
        help="Timeframe for signal generation / single-TF strategies (e.g. 15m, 1h)",
    )
    parser.add_argument(
        "--granular-tf", default=None,
        help="Optional finer timeframe for TP/SL execution in MTF mode (e.g. 1m). "
             "When set, data is loaded at this TF and resampled to --signal-tf.",
    )

    # Strategy
    parser.add_argument(
        "--strategy", required=True,
        help="Dotted path to a bt.Strategy class (e.g. strategies.MAStrategy)",
    )

    # Strategy parameters (TP / SL shortcuts + generic --param)
    parser.add_argument("--tp", type=float, default=None,
                        help="Take-profit %% as decimal (e.g. 0.015 = 1.5%%). "
                             "Passed to strategy as takeprofit_pct.")
    parser.add_argument("--sl", type=float, default=None,
                        help="Stop-loss %% as decimal (e.g. 0.006 = 0.6%%). "
                             "Passed to strategy as stoploss_pct.")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                        help="Arbitrary strategy param, e.g. --param rsi_period=10. "
                             "Numeric values are auto-cast. Can be repeated.")

    # Broker
    parser.add_argument("--cash", type=float, default=10_000, help="Starting cash (default: 10000)")
    parser.add_argument("--leverage", type=float, default=5, help="Leverage multiplier (default: 5)")
    parser.add_argument("--position-pct", type=float, default=0.95,
                        help="Fraction of equity per trade (default: 0.95)")
    parser.add_argument("--commission-maker", type=float, default=0.0002,
                        help="Maker commission rate (default: 0.0002)")
    parser.add_argument("--commission-taker", type=float, default=0.0005,
                        help="Taker commission rate (default: 0.0005)")
    parser.add_argument("--commission-type", default="taker",
                        choices=["maker", "taker", "blended"],
                        help="Commission type (default: taker)")

    # Slippage
    parser.add_argument("--slippage", type=float, default=0.0001,
                        help="Slippage per trade as fraction of price (default: 0.0001 = 0.01%%). "
                             "Set to 0 for ideal fills.")

    # Output
    parser.add_argument("--plot", action="store_true", help="Show backtrader chart after run")
    parser.add_argument("--plot-file", default=None,
                        help="Save chart to this file (e.g. result.png)")

    args = parser.parse_args()

    # Parse date range for feed filtering (must match --start/--end)
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")

    # ------------------------------------------------------------------
    # 1. Ensure data is present (skip already-fetched days)
    # ------------------------------------------------------------------
    is_mtf = args.granular_tf is not None

    if is_mtf:
        # In MTF mode we only need the granular data — the signal TF
        # is resampled from it inside backtrader, so downloading signal-TF
        # CSVs would be wasteful (and they're never loaded).
        print(f"[data] Ensuring {args.symbol} {args.granular_tf} data …")
        collect(args.symbol, args.start, args.end, args.granular_tf, args.data_dir)
    else:
        print(f"[data] Ensuring {args.symbol} {args.signal_tf} data …")
        collect(args.symbol, args.start, args.end, args.signal_tf, args.data_dir)

    # ------------------------------------------------------------------
    # 2. Load data into backtrader
    # ------------------------------------------------------------------
    strategy_cls = _load_strategy(args.strategy)

    cerebro = bt.Cerebro()

    if is_mtf:
        # Granular data (e.g. 1m) is datas[0]
        bt_df_granular = load_bt_dataframe(args.symbol, args.granular_tf, args.data_dir)
        data_granular = bt.feeds.PandasData(
            dataname=bt_df_granular,
            fromdate=start_dt,
            todate=end_dt,
        )
        cerebro.adddata(data_granular)

        # Resample to signal timeframe -> datas[1]
        tf, comp = _parse_tf(args.signal_tf)
        cerebro.resampledata(data_granular, timeframe=tf, compression=comp)
    else:
        bt_df = load_bt_dataframe(args.symbol, args.signal_tf, args.data_dir)
        cerebro.adddata(bt.feeds.PandasData(
            dataname=bt_df,
            fromdate=start_dt,
            todate=end_dt,
        ))

    # ------------------------------------------------------------------
    # 3. Strategy (with optional param overrides)
    # ------------------------------------------------------------------
    strat_kwargs = {}

    # MTF: single-TF strategies must use datas[1] for signals (not datas[0]).
    # OversoldBounceMTFStrategy already uses datas[1]; others need target_data_index.
    if is_mtf and hasattr(strategy_cls.params, "target_data_index"):
        strat_kwargs["target_data_index"] = 1

    # TP / SL shortcuts — only pass if the strategy declares the param
    if args.tp is not None:
        if hasattr(strategy_cls.params, "takeprofit_pct"):
            strat_kwargs["takeprofit_pct"] = args.tp
        else:
            print(f"[warn] --tp ignored: {strategy_cls.__name__} has no 'takeprofit_pct' param")
    if args.sl is not None:
        if hasattr(strategy_cls.params, "stoploss_pct"):
            strat_kwargs["stoploss_pct"] = args.sl
        else:
            print(f"[warn] --sl ignored: {strategy_cls.__name__} has no 'stoploss_pct' param")

    # Generic --param KEY=VALUE overrides
    for item in args.param:
        if "=" not in item:
            sys.exit(f"Bad --param format '{item}', expected KEY=VALUE")
        key, val = item.split("=", 1)
        if not hasattr(strategy_cls.params, key):
            sys.exit(f"Strategy {strategy_cls.__name__} has no param '{key}'. "
                     f"Declared params: {list(strategy_cls.params._getkeys())}")
        # Auto-cast: bool, int, float, else string
        v_lower = val.strip().lower()
        if v_lower in ("true", "yes", "1"):
            val = True
        elif v_lower in ("false", "no", "0"):
            val = False
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass  # keep as string
        strat_kwargs[key] = val

    cerebro.addstrategy(strategy_cls, **strat_kwargs)

    # ------------------------------------------------------------------
    # 4. Broker config
    # ------------------------------------------------------------------
    cerebro.broker.setcash(args.cash)
    if args.slippage > 0:
        cerebro.broker.set_slippage_perc(args.slippage, slip_open=True,
                                          slip_limit=False, slip_match=True,
                                          slip_out=False)
    comminfo = BinanceFuturesCommInfo(
        commission_maker=args.commission_maker,
        commission_taker=args.commission_taker,
        commission_type=args.commission_type,
        margin=1 / args.leverage,
    )
    cerebro.broker.addcommissioninfo(comminfo)
    cerebro.addsizer(
        bt.sizers.PercentSizer,
        percents=args.position_pct * args.leverage * 100,
    )

    # ------------------------------------------------------------------
    # 5. Analyzers
    # ------------------------------------------------------------------
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    tf, comp = _parse_tf(args.signal_tf)
    sharpe_factor = _TF_SHARPE_FACTOR.get(args.signal_tf, 252)
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        riskfreerate=0.0,
        timeframe=tf,
        compression=comp,
        factor=sharpe_factor,
        annualize=True,
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

    # ------------------------------------------------------------------
    # 6. Run
    # ------------------------------------------------------------------
    start_val = cerebro.broker.getvalue()
    print(f"\n{'='*50}")
    print(f" Symbol      : {args.symbol}")
    print(f" Period       : {args.start} → {args.end}")
    print(f" Signal TF    : {args.signal_tf}")
    if is_mtf:
        print(f" Granular TF  : {args.granular_tf}")
    print(f" Strategy     : {args.strategy}")
    if strat_kwargs:
        print(f" Params       : {strat_kwargs}")
    print(f" Leverage     : {args.leverage}x")
    print(f" Slippage     : {args.slippage*100:.3f}%")
    print(f" Starting cash: ${start_val:,.2f}")
    print(f"{'='*50}\n")

    results = cerebro.run()
    strat = results[0]
    end_val = cerebro.broker.getvalue()

    # ------------------------------------------------------------------
    # 7. Report
    # ------------------------------------------------------------------
    ta = strat.analyzers.trades.get_analysis()
    total = ta.get("total", {}).get("total", 0) or 0
    won = ta.get("won", {}).get("total", 0) or 0
    lost = ta.get("lost", {}).get("total", 0) or 0
    win_rate = (won / total * 100) if total > 0 else 0.0
    total_return = (end_val - start_val) / start_val * 100

    dd = strat.analyzers.drawdown.get_analysis()
    max_dd = dd.get("max", {}).get("drawdown", 0) or 0

    sharpe = strat.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe.get("sharperatio", None)
    sharpe_str = f"{sharpe_ratio:.3f}" if sharpe_ratio is not None else "N/A"

    print(f"\n{'='*50}")
    print(f" RESULTS")
    print(f"{'='*50}")
    print(f" Ending cash   : ${end_val:,.2f}")
    print(f" Total return  : {total_return:+.2f}%")
    print(f" Trades        : {total}  (won {won} / lost {lost})")
    print(f" Win rate      : {win_rate:.1f}%")
    print(f" Max drawdown  : {max_dd:.2f}%")
    print(f" Sharpe ratio  : {sharpe_str}")
    print(f"{'='*50}\n")

    # ------------------------------------------------------------------
    # 8. Plot (optional)
    # ------------------------------------------------------------------
    PLOT_BAR_LIMIT = 50_000  # 1m data for ~35 days; avoid OOM/freeze

    if args.plot or args.plot_file:
        n_bars = len(bt_df_granular) if is_mtf else len(bt_df)
        if n_bars > PLOT_BAR_LIMIT:
            print(f"[warn] Skipping plot: {n_bars:,} bars exceeds limit ({PLOT_BAR_LIMIT:,}). "
                  f"Use a shorter date range or omit --plot/--plot-file.")
        else:
            import matplotlib
            if args.plot_file and not args.plot:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.rcParams["figure.figsize"] = [15, 12]
            figs = cerebro.plot(iplot=False)
            if args.plot_file:
                for figlist in figs:
                    for fig in figlist:
                        fig.savefig(args.plot_file, dpi=150, bbox_inches="tight")
                        print(f"Chart saved to {args.plot_file}")
            if args.plot:
                plt.show()


if __name__ == "__main__":
    main()
