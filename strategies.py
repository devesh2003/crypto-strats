"""
Built-in backtrader strategy classes.

Import any strategy from here, or point backtest.py at your own module.
Each class must subclass ``bt.Strategy``.
"""

import backtrader as bt


# ---------------------------------------------------------------------------
# Simple MA cross-over
# ---------------------------------------------------------------------------
class MAStrategy(bt.Strategy):
    """
    Buy on fast-MA crossing above slow-MA, sell on the reverse.

    Optional TP/SL: when takeprofit_pct/stoploss_pct are set, they are
    checked before the crossover signal. Use --tp/--sl from the CLI.

    target_data_index: when 1 (MTF mode), use datas[1] for signals instead
    of datas[0]. The runner sets this automatically when --granular-tf is used.
    """
    params = (
        ("fast", 10),
        ("slow", 30),
        ("takeprofit_pct", None),
        ("stoploss_pct", None),
        ("target_data_index", 0),
    )

    def __init__(self):
        data = self.datas[self.params.target_data_index]
        self.fast_ma = bt.indicators.SMA(data.close, period=self.params.fast)
        self.slow_ma = bt.indicators.SMA(data.close, period=self.params.slow)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        self.order = None
        self.entry_price = None

    @property
    def _data(self):
        return self.datas[self.params.target_data_index]

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.entry_price = order.executed.price
            else:
                self.entry_price = None
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def next(self):
        if self.order:
            return
        data = self._data
        if not self.position:
            if self.crossover > 0:
                self.order = self.buy(data=data)
        else:
            # TP/SL check (when set), before crossover signal
            if self.entry_price is not None:
                tp = self.params.takeprofit_pct
                sl = self.params.stoploss_pct
                if tp is not None or sl is not None:
                    high = data.high[0]
                    low = data.low[0]
                    if sl is not None and low <= self.entry_price * (1 - sl):
                        self.order = self.close(data=data)
                        return
                    if tp is not None and high >= self.entry_price * (1 + tp):
                        self.order = self.close(data=data)
                        return
            if self.crossover < 0:
                self.order = self.close(data=data)


# ---------------------------------------------------------------------------
# Oversold-bounce mean-reversion (single timeframe)
# ---------------------------------------------------------------------------
class OversoldBounceStrategy(bt.Strategy):
    """
    Buy extreme oversold dips (RSI + Bollinger lower-band) in an uptrend,
    then take a quick profit or cut losses fast.

    TP/SL are checked against bar high/low (not just close) so intra-bar
    breaches are caught.  Entry price is recorded from the actual fill
    (via notify_order), not the signal bar's close.

    target_data_index: when 1 (MTF mode), use datas[1] for signals instead
    of datas[0]. The runner sets this automatically when --granular-tf is used.
    """
    params = (
        ("rsi_period", 14),
        ("rsi_oversold", 28),
        ("rsi_exit", 55),
        ("trend_ma", 50),
        ("takeprofit_pct", 0.015),
        ("stoploss_pct", 0.006),
        ("target_data_index", 0),
    )

    def __init__(self):
        data = self.datas[self.params.target_data_index]
        self.rsi = bt.indicators.RSI(data.close, period=self.params.rsi_period)
        self.bb = bt.indicators.BollingerBands(data.close, period=20, devfactor=2)
        self.trend_sma = bt.indicators.SMA(data.close, period=self.params.trend_ma)
        self.order = None
        self.entry_price = None

    @property
    def _data(self):
        return self.datas[self.params.target_data_index]

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.entry_price = order.executed.price
            else:  # sell / close
                self.entry_price = None
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def next(self):
        if self.order:
            return

        data = self._data
        if not self.position:
            price = data.close[0]
            in_uptrend = price > self.trend_sma[0]
            oversold = self.rsi[0] < self.params.rsi_oversold
            below_lower_bb = price <= self.bb.lines.bot[0]
            if in_uptrend and oversold and below_lower_bb:
                self.order = self.buy(data=data)
        else:
            if self.entry_price is None:
                return  # fill notification hasn't arrived yet

            high = data.high[0]
            low = data.low[0]
            tp_price = self.entry_price * (1 + self.params.takeprofit_pct)
            sl_price = self.entry_price * (1 - self.params.stoploss_pct)

            # Check SL before TP (conservative: assume worst outcome
            # when both levels are breached on the same bar)
            if low <= sl_price:
                self.order = self.close(data=data)
            elif high >= tp_price:
                self.order = self.close(data=data)
            elif self.rsi[0] > self.params.rsi_exit:
                self.order = self.close(data=data)


# ---------------------------------------------------------------------------
# Multi-timeframe oversold-bounce
#   datas[0] = granular (e.g. 1m) for TP/SL execution
#   datas[1] = signal   (e.g. 15m, resampled) for indicator signals
# ---------------------------------------------------------------------------
class OversoldBounceMTFStrategy(bt.Strategy):
    """
    Same logic as OversoldBounceStrategy but:
      * Indicators & signal generation run on the *higher* timeframe (datas[1])
      * TP / SL checks run on the *lower* (granular) timeframe (datas[0])
    """
    params = (
        ("rsi_period", 14),
        ("rsi_oversold", 28),
        ("rsi_exit", 55),
        ("trend_ma", 50),
        ("takeprofit_pct", 0.015),
        ("stoploss_pct", 0.006),
    )

    def __init__(self):
        # Indicators on signal timeframe (datas[1])
        self.rsi = bt.indicators.RSI(self.datas[1].close, period=self.params.rsi_period)
        self.bb = bt.indicators.BollingerBands(self.datas[1].close, period=20, devfactor=2)
        self.trend_sma = bt.indicators.SMA(self.datas[1].close, period=self.params.trend_ma)
        self.order = None
        self.entry_price = None

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.entry_price = order.executed.price
            else:
                self.entry_price = None
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if len(self.datas[1]) < self.params.trend_ma:
                return
            price_signal = self.datas[1].close[0]
            in_uptrend = price_signal > self.trend_sma[0]
            oversold = self.rsi[0] < self.params.rsi_oversold
            below_lower_bb = price_signal <= self.bb.lines.bot[0]
            if in_uptrend and oversold and below_lower_bb:
                self.order = self.buy(data=self.datas[0])
        else:
            if self.entry_price is None:
                return

            high = self.datas[0].high[0]
            low = self.datas[0].low[0]
            tp_price = self.entry_price * (1 + self.params.takeprofit_pct)
            sl_price = self.entry_price * (1 - self.params.stoploss_pct)

            if low <= sl_price:
                self.order = self.close(data=self.datas[0])
            elif high >= tp_price:
                self.order = self.close(data=self.datas[0])
            elif self.rsi[0] > self.params.rsi_exit:
                self.order = self.close(data=self.datas[0])
