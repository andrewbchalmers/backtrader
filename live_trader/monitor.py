# monitor.py
"""
Live trading monitor - scans for opportunities and manages positions
"""

import json
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import gc
import threading
from positions import PositionManager, PortfolioStateManager

# Suppress yfinance and urllib verbose logging/errors
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)


def _bold(text):
    """Convert ASCII text to Unicode bold sans-serif glyphs.
    Pushbullet note bodies are plain text — this is the only way to get
    visual bold emphasis inside a message body (not just the title).
    """
    out = []
    for ch in str(text):
        if 'A' <= ch <= 'Z':
            out.append(chr(0x1D5D4 + ord(ch) - ord('A')))
        elif 'a' <= ch <= 'z':
            out.append(chr(0x1D5EE + ord(ch) - ord('a')))
        elif '0' <= ch <= '9':
            out.append(chr(0x1D7EC + ord(ch) - ord('0')))
        else:
            out.append(ch)
    return ''.join(out)


def _run_backtest_for_worker(symbol, df, loader, params, interval):
    """
    Run a 1-year backtest on already-fetched data and return a results dict
    identical in structure to LiveTradingMonitor._run_backtest().

    Called from _entry_signal_worker after the entry-signal cerebro run so the
    buy-alert chart carries full stats (win rate, Sharpe, ML accuracy, earnings,
    vs-SPY, etc.) without any data re-fetch and without touching the parent's
    ml_lock or exit-scan thread.

    Args:
        symbol:   Stock ticker
        df:       DataFrame with lowercase columns, tz-naive index, zero-range bars fixed
        loader:   StrategyLoader instance (for filter_strategy_params + strategy_class)
        params:   Strategy params dict
        interval: Bar interval string (e.g. '1d')

    Returns:
        dict: Same structure as _run_backtest() results, or None on failure
    """
    import backtrader as bt
    import numpy as np
    import yfinance as _yf
    import pandas as _pd
    from datetime import datetime, timedelta

    try:
        total_bars = len(df)

        # --- 1-year test window ---
        test_start = datetime.now() - timedelta(days=365)
        test_start_mask = df.index >= _pd.Timestamp(test_start)
        if not test_start_mask.any():
            return None
        test_start_idx = int(test_start_mask.argmax())
        if test_start_idx >= total_bars - 10:
            test_start_idx = max(0, total_bars - 252)

        # --- SPY benchmark ---
        spy_return = 0
        try:
            test_start_date = df.index[test_start_idx]
            test_end_date   = df.index[-1]
            if interval in ('1m', '5m', '15m', '30m', '1h', '4h'):
                max_periods = {'1m': '7d', '5m': '60d', '15m': '60d',
                               '30m': '60d', '1h': '730d', '4h': '730d'}
                spy_df = _yf.download('SPY', period=max_periods.get(interval, '60d'),
                                      interval=interval, progress=False)
                if not spy_df.empty:
                    spy_df.index = spy_df.index.tz_localize(None)
                    spy_df = spy_df[(spy_df.index >= test_start_date) &
                                    (spy_df.index <= test_end_date)]
            else:
                spy_df = _yf.download(
                    'SPY',
                    start=test_start_date.strftime('%Y-%m-%d'),
                    end=(test_end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                    interval=interval, progress=False)
                if not spy_df.empty:
                    spy_df.index = spy_df.index.tz_localize(None)

            if not spy_df.empty:
                if isinstance(spy_df.columns, _pd.MultiIndex):
                    spy_df.columns = spy_df.columns.get_level_values(0)
                spy_df.columns = [c.lower() for c in spy_df.columns]
                spy_df = spy_df.loc[:, ~spy_df.columns.duplicated()]
                if len(spy_df) > 1 and 'close' in spy_df.columns:
                    spy_start = float(spy_df['close'].iloc[0])
                    spy_end   = float(spy_df['close'].iloc[-1])
                    if spy_start > 0:
                        spy_return = ((spy_end / spy_start) - 1) * 100
        except Exception:
            spy_return = 0

        # --- Cerebro setup ---
        cerebro = bt.Cerebro(stdstats=False)
        data = bt.feeds.PandasData(
            dataname=df, datetime=None,
            open='open', high='high', low='low', close='close', volume='volume'
        )
        cerebro.adddata(data)

        strategy_params = loader.filter_strategy_params(params)
        strategy_params['verbose'] = False
        strategy_params['test_start_idx'] = test_start_idx
        strategy_params['cross_symbol_target_symbol'] = symbol
        strategy_params['fundamental_symbol'] = symbol

        parent_strategy_class = loader.strategy_class

        class BacktestCaptureStrategy(parent_strategy_class):
            def __init__(self):
                super().__init__()
                self.buy_signals  = []
                self.sell_signals = []

            def _execute_buy(self):
                self.buy_signals.append({
                    'date':  self.data.datetime.date(0),
                    'price': self.data.close[0],
                    'bar':   len(self),
                })
                super()._execute_buy()

            def _close_position(self, reason):
                self.sell_signals.append({
                    'date':   self.data.datetime.date(0),
                    'price':  self.data.close[0],
                    'reason': reason,
                    'bar':    len(self),
                })
                super()._close_position(reason)

        cerebro.addstrategy(BacktestCaptureStrategy, **strategy_params)

        initial_cash = 10000
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0.0)
        cerebro.broker.set_coc(True)

        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.DrawDown,      _name='dd')

        class PortfolioValue(bt.Observer):
            lines = ('value',)
            plotinfo = dict(plot=False)
            def next(self):     self.lines.value[0] = self._owner.broker.getvalue()
            def prenext(self):  self.lines.value[0] = self._owner.broker.getvalue()

        cerebro.addobserver(PortfolioValue)

        results = cerebro.run()
        strat   = results[0]

        trades = strat.analyzers.trades.get_analysis()

        # Portfolio values for test period only
        test_values = []
        observer = strat.observers.portfoliovalue
        for i in range(len(observer.lines.value)):
            if i >= test_start_idx:
                try:
                    val = observer.lines.value.array[i]
                    if not np.isnan(val) and val > 0:
                        test_values.append(val)
                except Exception:
                    break
        if len(test_values) < 2:
            test_values = [initial_cash, cerebro.broker.getvalue()]

        final_value      = test_values[-1]
        total_return_pct = ((final_value / initial_cash) - 1) * 100

        # Trade metrics
        total_trades = trades.get('total', {}).get('total', 0)
        if total_trades > 0:
            win_count      = trades.get('won',  {}).get('total', 0)
            win_rate       = (win_count / total_trades) * 100
            total_win_pnl  = trades.get('won',  {}).get('pnl', {}).get('total', 0)
            total_loss_pnl = abs(trades.get('lost', {}).get('pnl', {}).get('total', 0))
            profit_factor  = (total_win_pnl / total_loss_pnl) if total_loss_pnl > 0 else 999
            avg_win        = trades.get('won',  {}).get('pnl', {}).get('average', 0)
            avg_loss       = abs(trades.get('lost', {}).get('pnl', {}).get('average', 0))
        else:
            win_rate = profit_factor = avg_win = avg_loss = 0

        # Max drawdown
        peak = test_values[0]
        max_dd = 0
        for val in test_values:
            if val > peak:
                peak = val
            dd = ((peak - val) / peak) * 100
            if dd > max_dd:
                max_dd = dd

        # Sharpe (annualised for daily bars)
        if len(test_values) > 1:
            bar_returns = [(test_values[i] / test_values[i - 1]) - 1
                           for i in range(1, len(test_values))]
            if len(bar_returns) > 1 and np.std(bar_returns) > 0:
                if interval in ('1m', '5m', '15m', '30m', '1h', '4h'):
                    bpd = {'1m': 390, '5m': 78, '15m': 26, '30m': 13, '1h': 7, '4h': 2}
                    ann = np.sqrt(252 * bpd.get(interval, 7))
                else:
                    ann = np.sqrt(252)
                sharpe = (np.mean(bar_returns) / np.std(bar_returns)) * ann
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Annualised return
        years      = len(test_values) / 252
        annualized = ((final_value / initial_cash) ** (1 / years) - 1) * 100 if years > 0 else total_return_pct

        test_df        = df.iloc[test_start_idx:]
        start_date_str = test_df.index[0].strftime('%Y-%m-%d')
        end_date_str   = test_df.index[-1].strftime('%Y-%m-%d')

        buy_signals_list  = [(s['date'], s['price']) for s in strat.buy_signals]
        sell_signals_list = [(s['date'], s['price']) for s in strat.sell_signals]

        # ML stats
        ml_stats = {}
        ml_diagnostics = {}
        try:
            if hasattr(strat, 'get_prediction_stats'):
                ml_stats = strat.get_prediction_stats()
            if hasattr(strat, 'get_diagnostics'):
                ml_diagnostics = strat.get_diagnostics()
        except Exception:
            pass

        # Time in market
        time_in_market_pct = 0.0
        test_bars_count    = len(test_values)
        if test_bars_count > 0 and strat.buy_signals and strat.sell_signals:
            buy_bars  = sorted(s['bar'] for s in strat.buy_signals)
            sell_bars = sorted(s['bar'] for s in strat.sell_signals)
            bars_in_position = 0
            for buy_bar in buy_bars:
                matching = [s for s in sell_bars if s > buy_bar]
                if matching:
                    sell_bar = matching[0]
                    start    = max(buy_bar, test_start_idx)
                    end      = sell_bar
                    if end > start:
                        bars_in_position += (end - start)
            time_in_market_pct = (bars_in_position / test_bars_count) * 100

        # Earnings / fundamentals
        tradeable_quarters = 0
        total_quarters     = 0
        earnings_data      = []
        test_period_start  = test_df.index[0]
        test_period_end    = test_df.index[-1]

        if (hasattr(strat, 'fundamental_provider') and
                strat.fundamental_provider is not None and
                strategy_params.get('use_fundamental_filter', False)):
            fp          = strat.fundamental_provider
            min_quality = strategy_params.get('min_quality_score', 0)
            min_momentum = strategy_params.get('min_momentum_score', 0)
            if hasattr(fp, '_quarter_report_map') and fp._quarter_report_map:
                for _quarter_end, report_date in fp._quarter_report_map.items():
                    report_ts = _pd.Timestamp(report_date)
                    if test_period_start <= report_ts <= test_period_end:
                        total_quarters += 1
                        try:
                            quality  = fp.get_quality_score(as_of_date=report_date)
                            momentum = fp.get_growth_momentum_score(as_of_date=report_date)
                            composite = 0
                            if quality is not None and momentum is not None:
                                composite = (quality + momentum) / 2
                            elif quality is not None:
                                composite = quality
                            elif momentum is not None:
                                composite = momentum
                            earnings_data.append((report_date, composite))
                            quality_ok  = min_quality  == 0 or (quality  is not None and quality  >= min_quality)
                            momentum_ok = min_momentum == 0 or (momentum is not None and momentum >= min_momentum)
                            if quality_ok and momentum_ok:
                                tradeable_quarters += 1
                        except Exception:
                            earnings_data.append((report_date, 0))

        # Fair value lines on the backtest chart
        fair_value_history    = []
        hist_pe_fair_value    = []
        try:
            if (hasattr(strat, 'fundamental_provider') and
                    strat.fundamental_provider is not None and
                    strategy_params.get('use_fundamental_filter', False)):
                fp = strat.fundamental_provider
                fair_value_history = fp.get_fair_value_history(
                    start_date=test_period_start, end_date=test_period_end)
                hist_pe_fair_value = fp.get_historical_pe_fair_value_history(
                    start_date=test_period_start, end_date=test_period_end,
                    price_df=df)
                print(f"   📊 Fair value (yellow): {len(fair_value_history)} qtrs, "
                      f"hist PE (purple): {len(hist_pe_fair_value)} qtrs")
        except Exception as e:
            import traceback
            print(f"   ⚠️  Fair value history error: {e}")
            traceback.print_exc()

        cerebro.runstop()
        del cerebro

        return {
            'start_date':          start_date_str,
            'end_date':            end_date_str,
            'interval':            interval,
            'total_bars':          total_bars,
            'test_bars':           len(test_values),
            'total_return_pct':    total_return_pct,
            'annualized_return':   annualized,
            'spy_return':          spy_return,
            'total_trades':        total_trades,
            'win_rate':            win_rate,
            'profit_factor':       profit_factor,
            'max_drawdown':        max_dd,
            'sharpe_ratio':        sharpe,
            'avg_win':             avg_win,
            'avg_loss':            avg_loss,
            'final_value':         final_value,
            'chart_df':            df,
            'test_start_idx':      test_start_idx,
            'buy_signals':         buy_signals_list,
            'sell_signals':        sell_signals_list,
            'ml_stats':            ml_stats,
            'ml_diagnostics':      ml_diagnostics,
            'time_in_market_pct':  time_in_market_pct,
            'tradeable_quarters':  tradeable_quarters,
            'total_quarters':      total_quarters,
            'earnings_data':       earnings_data,
            'fair_value_history':  fair_value_history,
            'hist_pe_fair_value':  hist_pe_fair_value,
        }

    except Exception:
        import traceback
        traceback.print_exc()
        return None


def _entry_signal_worker(args):
    """
    Worker function for parallel buy-scan.  Runs in a separate spawned process
    so it has no shared memory with the parent — no ml_lock required.

    Each worker independently fetches data and runs ML signal detection.
    Cross-symbol peer data is shared via the on-disk cache in
    ~/.cache/lorentzian_cross_symbol/ so workers don't redundantly download peers.
    """
    (symbol, yf_ticker, period, interval,
     strategy_module, strategy_class, params, live_trader_dir) = args

    import sys
    import logging as _logging
    import gc as _gc

    _logging.getLogger('yfinance').setLevel(_logging.CRITICAL)
    _logging.getLogger('urllib3').setLevel(_logging.CRITICAL)

    # Ensure live_trader and strategy dirs are importable
    if live_trader_dir not in sys.path:
        sys.path.insert(0, live_trader_dir)

    try:
        import yfinance as _yf
        import pandas as _pd
        from strategy_loader import StrategyLoader

        # --- Fetch OHLCV data ---
        df = _yf.download(yf_ticker, period=period, interval=interval,
                          progress=False, auto_adjust=True, prepost=False, threads=False)
        if df.empty or len(df) < 200:
            return symbol, {'signal': False}, 'no_data', None

        if isinstance(df.columns, _pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        df.index = df.index.tz_localize(None)

        # Patch last close with real-time price (mirrors get_live_data logic)
        if interval in ('1d', None):
            try:
                live_price = getattr(_yf.Ticker(yf_ticker).fast_info, 'last_price', None)
                if live_price and live_price > 0:
                    stale = float(df['Close'].iloc[-1])
                    if abs(live_price - stale) / stale > 0.005:
                        df.iloc[-1, df.columns.get_loc('Close')] = live_price
                        df.iloc[-1, df.columns.get_loc('High')] = max(
                            float(df.iloc[-1, df.columns.get_loc('High')]), live_price)
                        df.iloc[-1, df.columns.get_loc('Low')] = min(
                            float(df.iloc[-1, df.columns.get_loc('Low')]), live_price)
            except Exception:
                pass

        # --- Run ML signal detection ---
        loader = StrategyLoader(strategy_module, strategy_class)
        signal = loader.get_entry_signal(df, params, symbol=symbol)

        # --- Generate backtest chart if BUY signal found ---
        # Runs a 1Y backtest in this worker process (already isolated — no ml_lock needed).
        # Produces the identical chart + stats as the BACKTEST command.
        chart_bytes = None
        if signal.get('signal') and signal.get('signal_type') == 'BUY':
            try:
                import io as _io
                from chart_generator import ChartGenerator

                # generate_backtest_chart / _run_backtest_for_worker expect lowercase columns
                df_bt = df.copy()
                df_bt.columns = [c.lower() for c in df_bt.columns]

                # Fix zero-range bars on the copy (same fix applied inside _get_ml_strategy_signal)
                zero_range = df_bt['high'] == df_bt['low']
                if zero_range.any():
                    epsilon = df_bt['close'][zero_range] * 1e-6
                    df_bt.loc[zero_range, 'high'] += epsilon
                    df_bt.loc[zero_range, 'low']  -= epsilon

                bt_results = _run_backtest_for_worker(symbol, df_bt, loader, params, interval)

                if bt_results:
                    chart_gen = ChartGenerator(loader, params)
                    buf = chart_gen.generate_backtest_chart(
                        symbol=symbol,
                        df=df_bt,
                        test_start_idx=bt_results['test_start_idx'],
                        buy_signals=bt_results['buy_signals'],
                        sell_signals=bt_results['sell_signals'],
                        period_label='1Y',
                        interval=interval,
                        results=bt_results,
                        earnings_data=bt_results.get('earnings_data', []),
                        fair_value_data=bt_results.get('fair_value_history', []),
                        hist_pe_data=bt_results.get('hist_pe_fair_value', []),
                    )
                    if buf:
                        chart_bytes = buf.getvalue()
            except Exception:
                pass

        del df, loader
        _gc.collect()
        return symbol, signal, None, chart_bytes

    except Exception as e:
        return symbol, {'signal': False}, str(e), None


class LiveTradingMonitor:
    """Monitor stocks and send notifications for trading opportunities"""

    # Pre-market discovery scan starts at this hour (ET, 24h).
    PRE_MARKET_START_HOUR = 4

    # How often (hours) to run a full discovery scan.
    # e.g. 3 → runs pre-market, then again mid-morning, then again mid-afternoon.
    DISCOVERY_INTERVAL_HOURS = 3

    # Valid timeframes and their configurations
    TIMEFRAMES = {
        '1M': {'interval': '1m', 'period': '7d', 'description': '1 Minute'},
        '5M': {'interval': '5m', 'period': '60d', 'description': '5 Minutes'},
        '15M': {'interval': '15m', 'period': '60d', 'description': '15 Minutes'},
        '30M': {'interval': '30m', 'period': '60d', 'description': '30 Minutes'},
        '1H': {'interval': '1h', 'period': '730d', 'description': '1 Hour'},
        '4H': {'interval': '4h', 'period': '730d', 'description': '4 Hours'},
        '1D': {'interval': '1d', 'period': 'max', 'description': '1 Day'},
    }

    def __init__(self, watchlist, strategy_loader, strategy_params, notifier, warmup_days=300,
                 portfolio_capital=100_000, max_positions=10):
        self.watchlist = watchlist
        self.strategy = strategy_loader
        self.params = strategy_params
        self.notifier = notifier
        self.position_manager = PositionManager()
        self.warmup_days = warmup_days
        self.max_positions = max_positions
        self.buy_alerts_sent = {}  # Track when buy alerts were sent: {symbol: date}
        self.market_open_notified_date = None  # Track when market open notification was sent
        self.current_timeframe = '1D'  # Default to daily bars
        self.ml_lock = threading.Lock()  # Prevent concurrent ML operations
        self.scan_thread = None  # Buy scanning thread
        self.exit_thread = None  # Exit/sell scanning thread
        self.scanning = False  # Flag to control scan loops
        self.pending_replacements = {}  # {new_sym: {'worst_held': sym, 'signal': {...}, 'worst_pnl': float}}
        self.last_discovery_time = None  # Datetime of last full watchlist scan
        self.hot_list = {}               # {symbol: {'score': float, 'added_date': str, 'bars_ago': int}}
        self._load_hot_list()
        self.portfolio_state = PortfolioStateManager(
            initial_capital=portfolio_capital,
            position_manager=self.position_manager,
            max_positions=max_positions,
        )

        # Clear any pending_exit flags from previous sessions
        self._clear_pending_exit_flags()

        # Pre-fetch sector data for cross-symbol peer selection
        if self.params.get('use_cross_symbol_training') and self.params.get('cross_symbol_auto_peers'):
            try:
                from cross_symbol_preloader import prefetch_sectors
                peer_symbols = self.params.get('cross_symbol_etfs', '').split(',')
                print(f"Pre-fetching sector data for {len(peer_symbols)} peer universe symbols...")
                sector_map = prefetch_sectors(peer_symbols)
                print(f"Sectors found: {len(sector_map)}")
                for sector, members in sorted(sector_map.items(), key=lambda x: -len(x[1])):
                    print(f"  {sector}: {len(members)} stocks")
            except Exception as e:
                print(f"⚠️  Sector prefetch failed: {e}")

        # Import chart generator
        from chart_generator import ChartGenerator
        self.chart_gen = ChartGenerator(strategy_loader, strategy_params)

        # Start continuous reply listener
        self.notifier.start_listening(self._handle_reply)

    def _clear_pending_exit_flags(self):
        """Clear pending_exit flags on startup (they don't persist across restarts)"""
        positions = self.position_manager.list_all()
        cleared = 0

        for symbol, position in positions.items():
            if position.get('pending_exit'):
                del position['pending_exit']
                if 'exit_alerted_date' in position:
                    del position['exit_alerted_date']
                self.position_manager.positions[symbol] = position
                cleared += 1

        if cleared > 0:
            self.position_manager._save()
            print(f"ℹ️  Cleared {cleared} pending exit flag(s) from previous session")

    # -------------------------------------------------------------------------
    # Hot list — fast intraday buy scan
    # -------------------------------------------------------------------------

    def _load_hot_list(self):
        """Load persisted hot list from disk, cleaning expired entries."""
        try:
            if os.path.exists('hot_list.json'):
                with open('hot_list.json') as f:
                    self.hot_list = json.load(f)
                self._cleanup_hot_list(save=False)
                if self.hot_list:
                    print(f"ℹ️  Hot list: {len(self.hot_list)} symbols loaded")
        except Exception as e:
            print(f"⚠️  Could not load hot list: {e}")
            self.hot_list = {}

    def _save_hot_list(self):
        """Persist hot list to disk."""
        try:
            with open('hot_list.json', 'w') as f:
                json.dump(self.hot_list, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save hot list: {e}")

    def _add_to_hot_list(self, symbol, score, bars_ago, signal_price=None,
                         bullish_accuracy=None, bullish_total=0):
        """Add or refresh a symbol in the hot list."""
        today = datetime.now().date().isoformat()
        entry = {
            'score': float(score),
            'bars_ago': int(bars_ago),
            'added_date': today,
        }
        if signal_price is not None:
            entry['signal_price'] = float(signal_price)
        if bullish_accuracy is not None:
            entry['bullish_accuracy'] = float(bullish_accuracy)
            entry['bullish_total'] = int(bullish_total)
        self.hot_list[symbol] = entry
        self._save_hot_list()

    def _cleanup_hot_list(self, max_age_days=5, save=True):
        """Remove entries older than max_age_days."""
        cutoff = (datetime.now().date() - timedelta(days=max_age_days)).isoformat()
        before = len(self.hot_list)
        self.hot_list = {
            sym: data for sym, data in self.hot_list.items()
            if data.get('added_date', '') >= cutoff
        }
        removed = before - len(self.hot_list)
        if removed:
            print(f"ℹ️  Hot list: removed {removed} expired entry(s)")
        if save:
            self._save_hot_list()

    def _resolve_symbol(self, user_symbol):
        """
        Resolve a user-provided symbol to its canonical watchlist form.

        Handles exchange suffixes so that, e.g., a user typing 'CCO' is matched
        back to 'CCO.TO' when that is the watchlist/position entry.

        Priority order:
          1. Exact match in recent buy alerts (most likely the intended stock)
          2. Exact match in held positions
          3. Exact match in watchlist
          4. Base-ticker match (strip '.' suffix) in buy alerts, then positions, then watchlist

        Args:
            user_symbol: Symbol as typed by user (e.g. 'CCO' or 'CCO.TO')

        Returns:
            str: Canonical symbol, or user_symbol uppercased if no match found
        """
        u = user_symbol.upper()

        # 1. Exact matches
        if u in self.buy_alerts_sent:
            return u
        if u in self.position_manager.positions:
            return u
        if u in self.watchlist:
            return u

        # 2. Base-ticker fuzzy match (handles exchange suffix mismatch)
        base = u.split('.')[0]

        for sym in self.buy_alerts_sent:
            if sym.split('.')[0] == base:
                print(f"ℹ️  Resolved '{u}' → '{sym}' (from recent alerts)")
                return sym

        for sym in self.position_manager.positions:
            if sym.split('.')[0] == base:
                print(f"ℹ️  Resolved '{u}' → '{sym}' (from held positions)")
                return sym

        for sym in self.watchlist:
            if sym.split('.')[0] == base:
                print(f"ℹ️  Resolved '{u}' → '{sym}' (from watchlist)")
                return sym

        return u

    def get_live_data(self, symbol, period=None, interval=None, use_current_timeframe=True, ticker=None):
        """Fetch live data for a symbol

        Args:
            symbol: Stock ticker (display name, used as fallback)
            period: Data period (e.g., "1y", "6mo", "60d"). If None, uses current timeframe setting.
            interval: Bar interval (e.g., "1d", "1h", "15m"). If None, uses current timeframe setting.
            use_current_timeframe: If True and period/interval not specified, use current timeframe
            ticker: Override yfinance ticker (e.g. 'CCO.TO'). Falls back to symbol if not provided.
        """
        # Use the exchange-specific ticker if provided, otherwise fall back to symbol
        yf_symbol = ticker or symbol
        try:
            # Use current timeframe settings if not explicitly specified
            if use_current_timeframe and (period is None or interval is None):
                tf_config = self.TIMEFRAMES[self.current_timeframe]
                if period is None:
                    period = tf_config['period']
                if interval is None:
                    interval = tf_config['interval']

            # Use yf.download instead of Ticker.history - handles connections better
            df = yf.download(yf_symbol, period=period, interval=interval, progress=False,
                           auto_adjust=True, prepost=False, threads=False)

            if df.empty:
                return None

            # Handle multi-level columns from yf.download
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten to just the price column names (Open, High, Low, Close, Volume)
                df.columns = [col[0] for col in df.columns]

            # Remove any duplicate columns (keep first)
            df = df.loc[:, ~df.columns.duplicated()]

            df.index = df.index.tz_localize(None)

            # Patch the last bar's close with a real-time quote.
            # yf.download with auto_adjust=True can return yesterday's close (not today's
            # intraday price) or apply an incorrect adjustment factor to the most recent bar.
            # This causes false exit signals when the stock has moved significantly intraday.
            # Only applied for daily interval to avoid patching intraday bars unnecessarily.
            if interval in ('1d', None):
                try:
                    fast_info = yf.Ticker(yf_symbol).fast_info
                    live_price = getattr(fast_info, 'last_price', None)
                    if live_price and live_price > 0:
                        stale_close = float(df['Close'].iloc[-1])
                        pct_diff = (live_price - stale_close) / stale_close
                        if abs(pct_diff) > 0.005:  # >0.5% discrepancy
                            print(f"   ⚠️  {yf_symbol}: Stale/adjusted close ${stale_close:.2f} → "
                                  f"real-time ${live_price:.2f} ({pct_diff:+.2%}). Patching last bar.")
                            close_idx = df.columns.get_loc('Close')
                            high_idx = df.columns.get_loc('High')
                            low_idx = df.columns.get_loc('Low')
                            df.iloc[-1, close_idx] = live_price
                            df.iloc[-1, high_idx] = max(float(df.iloc[-1, high_idx]), live_price)
                            df.iloc[-1, low_idx] = min(float(df.iloc[-1, low_idx]), live_price)
                except Exception as e:
                    pass  # Non-fatal: fall back to downloaded close

            return df

        except Exception as e:
            print(f"\n❌ Error fetching {yf_symbol}: {e}")
            return None

    def get_historical_data(self, symbol, start_date, end_date):
        """Fetch historical data for testing mode with warmup period"""
        try:
            start = pd.to_datetime(start_date)

            # Convert trading days to calendar days
            calendar_days = int(self.warmup_days * 1.6)
            lookback_start = start - timedelta(days=calendar_days)

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=lookback_start.strftime('%Y-%m-%d'),
                                end=end_date,
                                interval="1d")

            if df.empty:
                return None

            df.index = df.index.tz_localize(None)
            return df

        except Exception as e:
            print(f"❌ Error fetching historical data for {symbol}: {e}")
            return None

    def scan_for_opportunities(self, symbols=None):
        """Scan stocks for buy opportunities.

        Args:
            symbols: List of symbols to scan.  If None, scans the full watchlist
                     (discovery mode).  Pass a subset for hot-list intraday scans.
        """
        tf_config = self.TIMEFRAMES[self.current_timeframe]
        discovery = symbols is None
        mode_label = "DISCOVERY SCAN" if discovery else f"HOT SCAN ({len(symbols)} symbols)"
        print(f"\n{'='*60}")
        print(f"{mode_label} AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Timeframe: {self.current_timeframe} ({tf_config['description']})")
        print(f"{'='*60}\n")

        # Clean up old buy alerts (remove alerts from previous days)
        today = datetime.now().date()
        self.buy_alerts_sent = {
            symbol: date for symbol, date in self.buy_alerts_sent.items()
            if date == today
        }

        # Get list of held positions
        held_positions = set(self.position_manager.list_all().keys())
        print(f"Currently holding {len(held_positions)} positions: {list(held_positions)}")

        # Scan for BUY opportunities (only symbols we DON'T hold)
        buy_opportunities = 0
        today = datetime.now().date()

        # Choose source list
        source = self.watchlist if discovery else symbols

        # Filter to only symbols we need to scan
        symbols_to_scan = [s for s in source
                          if s not in held_positions
                          and self.buy_alerts_sent.get(s) != today]
        total_to_scan = len(symbols_to_scan)
        scan_start = time.time()
        print(f"Scanning {total_to_scan} symbols for buy opportunities...")

        # Phase 1: collect all signals in parallel using worker processes.
        # Each worker independently fetches data + runs ML — no shared state,
        # no ml_lock needed.  Cross-symbol peer data is shared via disk cache.
        found_signals = []  # list of (symbol, signal_dict, abs_prediction_score)

        # Absolute strategy module path (workers may have different cwd)
        strategy_module_abs = os.path.abspath(self.strategy.module_name)
        live_trader_dir = os.path.dirname(os.path.abspath(__file__))
        tf_config_interval = tf_config['interval']
        tf_config_period = tf_config['period']

        # Cap workers at 4 to stay within yfinance rate limits while still
        # getting meaningful parallelism.  Increase if your network allows it.
        n_workers = min(multiprocessing.cpu_count() or 4, 4)
        print(f"Running parallel scan with {n_workers} worker processes...")

        scan_args = [
            (symbol, symbol,
             tf_config_period, tf_config_interval,
             strategy_module_abs, self.strategy.class_name,
             self.params, live_trader_dir)
            for symbol in symbols_to_scan
        ]

        # Use spawn context — safe with parent's running threads (exit_scan, listener)
        mp_ctx = multiprocessing.get_context('spawn')
        completed = 0

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as executor:
            futures = {executor.submit(_entry_signal_worker, args): args[0]
                       for args in scan_args}

            for future in as_completed(futures):
                symbol = futures[future]
                completed += 1
                print(f"\r   [{completed}/{total_to_scan}] scanning...", end='', flush=True)
                try:
                    sym, signal, error, chart_bytes = future.result()
                    if error and error != 'no_data':
                        print(f"\n   ❌ {sym}: {error}")
                    elif (signal.get('signal') and
                          signal.get('signal_type', 'BUY') == 'BUY' and
                          not self.position_manager.has_position(sym)):
                        bars_ago = signal.get('bars_ago', 0)
                        if bars_ago <= 3:
                            score = abs(signal.get('prediction', 0))
                            found_signals.append((sym, signal, score, chart_bytes))

                            # Always add to hot list — keeps it fresh for intraday re-scans
                            self._add_to_hot_list(sym, score, bars_ago,
                                                  signal_price=signal.get('price'),
                                                  bullish_accuracy=signal.get('bullish_accuracy'),
                                                  bullish_total=signal.get('bullish_total', 0))

                            if self.buy_alerts_sent.get(sym) == today:
                                print(f"\n   🟢 SIGNAL: {sym} (score={score:.0f}, "
                                      f"bars_ago={bars_ago}) | already alerted today")
                            else:
                                cur_slots = max(0, self.max_positions -
                                               len(self.position_manager.list_active()))
                                if cur_slots > 0:
                                    # Compute suggested position size (same formula as handle_bought_reply)
                                    cash = self.portfolio_state.get_cash()
                                    n_active = len(self.position_manager.list_active())
                                    remaining_slots = max(1, self.max_positions - n_active)
                                    suggested_amount = cash / remaining_slots if cash > 0 else 0.0
                                    suggested_shares = int(suggested_amount / signal['price']) if signal['price'] > 0 else 0

                                    # Send immediately — with chart if generated, text-only as fallback
                                    import io
                                    if chart_bytes:
                                        self.notifier.send_buy_alert_with_chart(
                                            sym, signal, io.BytesIO(chart_bytes),
                                            suggested_amount=suggested_amount,
                                            suggested_shares=suggested_shares)
                                    else:
                                        self.notifier.send_buy_alert(sym, signal,
                                                                     suggested_amount=suggested_amount,
                                                                     suggested_shares=suggested_shares)
                                    self.buy_alerts_sent[sym] = today
                                    buy_opportunities += 1
                                    print(f"\n   🟢 BUY ALERT SENT: {sym} (score={score:.0f}, "
                                          f"bars_ago={bars_ago}, suggested=${suggested_amount:,.0f}, "
                                          f"chart={'yes' if chart_bytes else 'no'})")
                                else:
                                    print(f"\n   🟢 SIGNAL: {sym} (score={score:.0f}, "
                                          f"bars_ago={bars_ago}) | no slots (overflow)")
                        else:
                            print(f"\n   ⏭️  {sym}: signal too old ({bars_ago} bars ago)")
                except Exception as e:
                    print(f"\n   ❌ {symbol}: {e}")

        gc.collect()

        scan_duration = time.time() - scan_start
        print(f"\n   Parallel scan complete: {total_to_scan} symbols in {scan_duration:.1f}s "
              f"({n_workers} workers)")

        # Phase 2: handle overflow signals (portfolio full — replacement logic)
        # Alerts for signals with available slots were already sent immediately in Phase 1.
        found_signals.sort(key=lambda x: x[2], reverse=True)

        held_positions_dict = self.position_manager.list_all()
        active_positions_dict = self.position_manager.list_active()
        available_slots = max(0, self.max_positions - len(active_positions_dict))

        # Collect overflow signals (not yet alerted, no slots available)
        overflow_signals = []
        for symbol, signal, score, _chart in found_signals:
            if self.buy_alerts_sent.get(symbol) == today:
                continue  # already sent in Phase 1
            if symbol in held_positions_dict:
                continue
            overflow_signals.append((symbol, signal, score))

        # Replacement suggestion: best overflow vs worst held position
        if overflow_signals and available_slots == 0:
            best_sym, best_signal, best_score = overflow_signals[0]
            worst_sym, worst_pnl = self._find_worst_held_position(held_positions_dict)
            if worst_sym:
                self._send_replacement_alert(best_sym, best_signal, best_score, worst_sym, worst_pnl)

        print(f"✅ Scan complete in {scan_duration:.1f}s - Found {buy_opportunities} buy opportunity(s) "
              f"(active: {len(active_positions_dict)}/{self.max_positions}, "
              f"pending exit: {len(held_positions_dict) - len(active_positions_dict)}, "
              f"overflow: {len(overflow_signals)})")

    def _handle_reply(self, reply):
        """
        Process a single reply from user (called by background listener)

        Args:
            reply: The reply string from user (already uppercased)
        """
        print(f"\n💬 Processing reply: {reply}")

        # Quick commands - process immediately
        if reply.startswith("BOUGHT "):
            self.handle_bought_reply(reply)
        elif reply.startswith("SOLD "):
            self.handle_sold_reply(reply)
        elif reply.startswith("TIMEFRAME "):
            self.handle_timeframe_command(reply)
        elif reply == "HOLDING" or reply == "HOLDINGS":
            self.handle_holdings_query()
        elif reply == "TIMEFRAME":
            self.handle_timeframe_query()
        elif reply.startswith("ADD "):
            self.handle_add_command(reply)
        elif reply.startswith("REMOVE "):
            self.handle_remove_command(reply)
        elif reply.startswith("UPDATE "):
            self.handle_update_command(reply)
        # Commands that use ML - processed immediately on listener thread (ml_lock handles sync)
        elif reply.startswith("LAST "):
            self.handle_last_signal_query(reply)
        elif reply.startswith("BACKTEST "):
            self.handle_backtest_query(reply)
        elif reply.startswith("ANALYZE "):
            self.handle_analyze_query(reply)
        elif reply.startswith("COMPARE "):
            self.handle_compare_query(reply)
        elif reply.startswith("REPLACE "):
            self.handle_replace_command(reply)
        elif reply.startswith("CAPITAL "):
            self.handle_capital_command(reply)
        elif reply == "PORTFOLIO":
            self.handle_portfolio_query()
        elif reply == "PORTFOLIO WORST":
            self.handle_portfolio_worst_query()
        elif reply == "BEST":
            self.handle_best_query()
        elif reply == "HELP":
            self.handle_help_query()
        else:
            print(f"⚠️  Unknown reply format: {reply}")

    def handle_bought_reply(self, reply):
        """Handle 'BOUGHT SYMBOL' or 'BOUGHT SYMBOL AT PRICE' reply"""
        parts = reply.split()

        if len(parts) < 2:
            print(f"⚠️  Invalid BOUGHT format: {reply}")
            return

        # Resolve to canonical watchlist symbol (handles exchange suffix mismatches)
        symbol = self._resolve_symbol(parts[1])

        if self.position_manager.has_position(symbol):
            print(f"⚠️  Already holding {symbol}")
            return

        # Check if price was provided
        price = None
        if len(parts) >= 4 and parts[2].upper() == "AT":
            try:
                price = float(parts[3].replace('$', ''))
            except ValueError:
                print(f"⚠️  Invalid price in: {reply}")

        # Look up exchange info so we always fetch from the right market
        exchange = ''
        try:
            ticker_info = yf.Ticker(symbol).info
            exchange = ticker_info.get('fullExchangeName', '') or ticker_info.get('exchange', '')
        except Exception:
            pass

        # Get current price if not provided, using the canonical ticker
        if price is None:
            df = self.get_live_data(symbol)
            if df is not None:
                price = df['Close'].iloc[-1]
            else:
                print(f"❌ Could not get price for {symbol}")
                return

        # Calculate stop loss using simple percentage (avoid slow ML call)
        stop_loss_pct = self.params.get('stop_loss_pct', 0.05)
        if hasattr(stop_loss_pct, '__float__'):
            stop_loss_pct = float(stop_loss_pct)
        stop_loss = price * (1 - stop_loss_pct)

        # Compute dynamic position size from remaining cash and open slots
        cash = self.portfolio_state.get_cash()
        n_currently_held = len(self.position_manager.list_active())
        remaining_slots = max(1, self.max_positions - n_currently_held)
        allocated_amount = cash / remaining_slots if cash > 0 else 0.0

        # Add position, storing the canonical ticker and exchange for future data fetches
        position = self.position_manager.add(
            symbol, price, stop_loss,
            yf_ticker=symbol, exchange=exchange,
            allocated_amount=allocated_amount,
        )

        if position:
            if allocated_amount > 0:
                self.portfolio_state.deduct_cash(allocated_amount)
            shares = int(allocated_amount / price) if price > 0 else 0
            self.notifier.send_position_confirmation(symbol, price, stop_loss, "added", exchange=exchange,
                                                     allocated_amount=allocated_amount, shares=shares)
            print(f"✅ Auto-added position from reply: {symbol} [{exchange}] @ ${price:.2f} "
                  f"(allocated ${allocated_amount:,.2f}, {shares} shares)")

            # Clear the buy alert tracking since position was confirmed
            if symbol in self.buy_alerts_sent:
                del self.buy_alerts_sent[symbol]

    def handle_sold_reply(self, reply):
        """Handle 'SOLD SYMBOL' reply"""
        parts = reply.split()

        if len(parts) < 2:
            print(f"⚠️  Invalid SOLD format: {reply}")
            return

        # Resolve to canonical symbol (handles exchange suffix mismatches)
        symbol = self._resolve_symbol(parts[1])
        position = self.position_manager.get(symbol)

        if not position:
            print(f"⚠️  No position found for {symbol}")
            return

        # Get entry price for notification
        entry_price = position['entry_price']

        # Use the stored canonical ticker for data fetching (correct exchange)
        yf_ticker = position.get('yf_ticker', symbol)
        exchange = position.get('exchange', '')

        # Get current price to calculate P&L
        df = self.get_live_data(symbol, ticker=yf_ticker)
        if df is not None:
            current_price = df['Close'].iloc[-1]
            pnl = ((current_price / entry_price) - 1) * 100
        else:
            current_price = entry_price
            pnl = 0.0

        # Return proceeds to cash (proportional gain/loss on allocated amount)
        allocated = position.get('allocated_amount', 0.0)
        if allocated > 0 and entry_price > 0:
            proceeds = allocated * (current_price / entry_price)
            self.portfolio_state.add_cash(proceeds)
            print(f"💰 Returned ${proceeds:,.2f} to cash (was ${allocated:,.2f} allocated)")

        # Remove the position
        self.position_manager.remove(symbol)

        # Send confirmation with P&L
        self.notifier.send_position_confirmation(symbol, entry_price, current_price, "removed", pnl, exchange=exchange)
        print(f"✅ Auto-removed position from reply: {symbol} [{exchange}] (P&L: {pnl:+.2f}%)")

    def handle_add_command(self, reply):
        """Handle 'ADD SYMBOL' command to add stock to watchlist"""
        import os

        parts = reply.split()
        if len(parts) < 2:
            print(f"⚠️  Invalid ADD format: {reply}")
            self.notifier.send_notification("Invalid Command", "Usage: ADD <SYMBOL>")
            return

        symbol = parts[1].upper().strip()

        # Get watchlist file path (in same directory as this module)
        watchlist_file = os.path.join(os.path.dirname(__file__), 'watchlist.csv')

        # Read current watchlist from file
        existing_symbols = set()
        try:
            with open(watchlist_file, 'r') as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith('#'):
                        existing_symbols.add(s.upper())
        except FileNotFoundError:
            existing_symbols = set()

        # Check if already in watchlist
        if symbol in existing_symbols:
            msg = f"{symbol} is already in the watchlist"
            print(f"ℹ️  {msg}")
            self.notifier.send_notification("Already in Watchlist", msg)
            return

        # Add to file
        try:
            with open(watchlist_file, 'a') as f:
                f.write(f"{symbol}\n")

            # Also add to in-memory watchlist
            if symbol not in self.watchlist:
                self.watchlist.append(symbol)

            print(f"✅ Added {symbol} to watchlist")
            self.notifier.send_notification("Stock Added to Watchlist", f"{symbol} has been added to your watchlist")

        except Exception as e:
            print(f"❌ Error adding {symbol} to watchlist: {e}")
            self.notifier.send_notification("Error", f"Failed to add {symbol}: {e}")

    def handle_remove_command(self, reply):
        """Handle 'REMOVE SYMBOL' command to remove stock from watchlist"""
        import os

        parts = reply.split()
        if len(parts) < 2:
            self.notifier.send_notification("Invalid Command", "Usage: REMOVE <SYMBOL>")
            return

        symbol = parts[1].upper().strip()

        watchlist_file = os.path.join(os.path.dirname(__file__), 'watchlist.csv')

        try:
            with open(watchlist_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            self.notifier.send_notification("Error", "Watchlist file not found")
            return

        new_lines = [l for l in lines if l.strip().upper() != symbol]

        if len(new_lines) == len(lines):
            self.notifier.send_notification("Not Found", f"{symbol} is not in the watchlist")
            return

        with open(watchlist_file, 'w') as f:
            f.writelines(new_lines)

        if symbol in self.watchlist:
            self.watchlist.remove(symbol)

        print(f"✅ Removed {symbol} from watchlist")
        self.notifier.send_notification("Removed from Watchlist", f"{symbol} has been removed from your watchlist")

    def handle_update_command(self, reply):
        """Handle 'UPDATE SYMBOL PRICE' command to set entry price for a position"""
        parts = reply.split()
        if len(parts) < 3:
            self.notifier.send_notification("Invalid Command", "Usage: UPDATE <SYMBOL> <PRICE>")
            return

        symbol = self._resolve_symbol(parts[1])
        try:
            new_price = float(parts[2].replace('$', ''))
        except ValueError:
            self.notifier.send_notification("Invalid Price", f"Could not parse price: {parts[2]}")
            return

        if not self.position_manager.has_position(symbol):
            self.notifier.send_notification("Not Found", f"No active position for {symbol}")
            return

        old_price = self.position_manager.get(symbol)['entry_price']
        if self.position_manager.update_entry_price(symbol, new_price):
            self.notifier.send_notification(
                f"Updated {symbol}",
                f"Entry price: ${old_price:.2f} → ${new_price:.2f}"
            )
        else:
            self.notifier.send_notification("Error", f"Failed to update {symbol}")

    def _find_worst_held_position(self, held_positions):
        """Return (symbol, pnl_pct) for the worst-performing held position."""
        worst_sym = None
        worst_pnl = float('inf')
        for sym, pos in held_positions.items():
            if pos.get('pending_exit'):
                continue
            entry = pos['entry_price']
            yf_ticker = pos.get('yf_ticker', sym)
            df = self.get_live_data(sym, ticker=yf_ticker)
            if df is None:
                continue
            current = df['Close'].iloc[-1]
            pnl = (current / entry - 1) * 100
            if pnl < worst_pnl:
                worst_pnl = pnl
                worst_sym = sym
        return worst_sym, worst_pnl

    def _send_replacement_alert(self, new_sym, new_signal, new_score, worst_sym, worst_pnl):
        """Send a replacement suggestion alert and store pending state."""
        held_info = self.position_manager.get(worst_sym)
        held_days = 0
        if held_info and held_info.get('entry_date'):
            from datetime import date as _date
            try:
                entry_dt = datetime.fromisoformat(held_info['entry_date']).date()
                held_days = (_date.today() - entry_dt).days
            except Exception:
                pass

        title = f"🔄 REPLACE? {worst_sym} → {new_sym}"
        message = (
            f"Portfolio full ({self.max_positions}/{self.max_positions})\n\n"
            f"NEW SIGNAL: {new_sym}\n"
            f"  ML Score: {new_score:.0f}/100\n"
            f"  Price: ${new_signal['price']:.2f}\n\n"
            f"WORST HELD: {worst_sym}\n"
            f"  P&L: {worst_pnl:+.1f}%\n"
            f"  Held: {held_days} days\n\n"
            f"Reply 'REPLACE {new_sym}' to swap"
        )
        self.notifier.send_notification(title, message)

        self.pending_replacements[new_sym] = {
            'worst_held': worst_sym,
            'signal': new_signal,
            'worst_pnl': worst_pnl,
        }
        print(f"🔄 Replacement suggestion: {new_sym} (score={new_score:.0f}) vs {worst_sym} (P&L={worst_pnl:+.1f}%)")

    def handle_replace_command(self, reply):
        """Handle 'REPLACE <SYMBOL>' — execute pending replacement swap."""
        parts = reply.split()
        if len(parts) < 2:
            self.notifier.send_notification("⚠️ Invalid", "Usage: REPLACE <SYMBOL>")
            return
        new_sym = self._resolve_symbol(parts[1])
        pending = self.pending_replacements.get(new_sym)
        if not pending:
            self.notifier.send_notification("⚠️ No Pending Replacement",
                f"No replacement queued for {new_sym}. Wait for a REPLACE? alert first.")
            return

        worst_sym = pending['worst_held']
        signal = pending['signal']

        # Pre-flight checks before any state is modified
        worst_pos = self.position_manager.get(worst_sym)
        if not worst_pos:
            self.notifier.send_notification("⚠️ Error", f"{worst_sym} no longer held")
            del self.pending_replacements[new_sym]
            return

        if self.position_manager.has_position(new_sym):
            self.notifier.send_notification(
                "⚠️ Replace Failed",
                f"{new_sym} is already held. No changes made.\n"
                f"Send 'SOLD {new_sym}' first if you want to re-enter."
            )
            del self.pending_replacements[new_sym]
            return

        # 1. Sell worst position at current price

        yf_ticker = worst_pos.get('yf_ticker', worst_sym)
        df = self.get_live_data(worst_sym, ticker=yf_ticker)
        exit_price = df['Close'].iloc[-1] if df is not None else worst_pos['entry_price']
        allocated = worst_pos.get('allocated_amount', 0.0)
        proceeds = allocated * (exit_price / worst_pos['entry_price']) if allocated > 0 else 0.0
        pnl_pct = (exit_price / worst_pos['entry_price'] - 1) * 100

        self.portfolio_state.add_cash(proceeds)
        self.position_manager.remove(worst_sym)

        sell_msg = f"Sold {worst_sym} @ ${exit_price:.2f} (P&L: {pnl_pct:+.1f}%)\nProceeds: ${proceeds:,.2f}"
        self.notifier.send_notification(f"✅ SOLD {worst_sym} (replaced)", sell_msg)

        # 2. Buy new signal — allocate exactly the proceeds from the sale (1-for-1 swap)
        alloc = proceeds
        self.portfolio_state.deduct_cash(alloc)

        exchange = ''
        try:
            exchange = yf.Ticker(new_sym).info.get('fullExchangeName', '')
        except Exception:
            pass

        # Fetch current price — signal['price'] is stale (from scan time, possibly hours ago)
        entry_price = signal['price']  # fallback
        df_new = self.get_live_data(new_sym)
        if df_new is not None and not df_new.empty:
            entry_price = float(df_new['Close'].iloc[-1])

        stop_loss_pct = self.params.get('stop_loss_pct', 0.05)
        if hasattr(stop_loss_pct, '__float__'):
            stop_loss_pct = float(stop_loss_pct)
        stop_loss = entry_price * (1 - stop_loss_pct)

        self.position_manager.add(new_sym, entry_price, stop_loss,
                                  yf_ticker=new_sym, exchange=exchange,
                                  allocated_amount=alloc)
        self.buy_alerts_sent[new_sym] = datetime.now().date()

        buy_msg = f"Bought {new_sym} @ ${entry_price:.2f}\nAllocated: ${alloc:,.2f}"
        self.notifier.send_notification(f"✅ BOUGHT {new_sym} (replacement)", buy_msg)

        del self.pending_replacements[new_sym]
        print(f"✅ Replacement complete: {worst_sym} → {new_sym}")

    def handle_capital_command(self, reply):
        """Handle 'CAPITAL SET x' or 'CAPITAL ADD x'."""
        parts = reply.split()
        if len(parts) != 3 or parts[1] not in ('SET', 'ADD'):
            self.notifier.send_notification("⚠️ Invalid",
                "Usage: CAPITAL SET <amount> or CAPITAL ADD <amount>")
            return
        try:
            amount = float(parts[2].replace('$', '').replace(',', ''))
        except ValueError:
            self.notifier.send_notification("⚠️ Invalid Amount", f"Could not parse: {parts[2]}")
            return

        if parts[1] == 'SET':
            self.portfolio_state.set_cash(amount)
            self.notifier.send_notification("💰 Cash Updated", f"Cash set to ${amount:,.2f}")
        else:
            self.portfolio_state.add_cash(amount)
            new_cash = self.portfolio_state.get_cash()
            self.notifier.send_notification("💰 Cash Updated",
                f"+${amount:,.2f} → Total cash: ${new_cash:,.2f}")

    def handle_portfolio_query(self):
        """Handle 'PORTFOLIO' — show full portfolio state."""
        positions = self.position_manager.list_all()
        cash = self.portfolio_state.get_cash()
        n = len(positions)

        lines = []
        total_invested = 0.0
        for sym, pos in positions.items():
            alloc = pos.get('allocated_amount', 0.0)
            entry = pos['entry_price']
            yf_ticker = pos.get('yf_ticker', sym)
            df = self.get_live_data(sym, ticker=yf_ticker)
            if df is not None:
                cur = df['Close'].iloc[-1]
                pnl = (cur / entry - 1) * 100
                cur_val = alloc * (cur / entry) if alloc > 0 else 0.0
            else:
                pnl = 0.0
                cur_val = alloc
            total_invested += cur_val
            status = "⏳" if pos.get('pending_exit') else "  "
            lines.append(f"{status}{sym}: ${cur_val:,.0f} ({pnl:+.1f}%)")

        total = cash + total_invested
        message = (
            f"Slots: {n}/{self.max_positions}\n"
            f"Cash:     ${cash:,.2f}\n"
            f"Invested: ${total_invested:,.2f}\n"
            f"Total:    ${total:,.2f}\n"
            f"\n" + "\n".join(lines)
        )
        self.notifier.send_notification("📊 Portfolio", message)

    def handle_portfolio_worst_query(self):
        """Handle 'PORTFOLIO WORST' — show the 5 most urgent held positions.

        Ranked by: urgency = days_held / distance_to_stop_pct

        This surfaces positions that are both close to being stopped out AND
        have been tying up capital for a long time. A position 1% above its
        stop after 90 days is far more urgent than one 10% above its stop
        after 5 days. Pending-exit positions (already past stop) always
        float to the top.
        """
        positions = self.position_manager.list_all()

        if not positions:
            self.notifier.send_notification("📭 No Holdings", "No positions currently held.")
            return

        ranked = []
        for sym, pos in positions.items():
            entry = pos['entry_price']
            stop = pos['stop_loss']
            try:
                ticker = yf.Ticker(pos.get('yf_ticker', sym))
                cur = getattr(ticker.fast_info, 'last_price', None)
                if not cur:
                    cur = ticker.info.get('regularMarketPrice')
                if not cur:
                    continue
                cur = float(cur)
                pnl = (cur / entry - 1) * 100
                held_days = (datetime.now().date() - datetime.fromisoformat(pos['entry_date']).date()).days
                distance_pct = (cur - stop) / cur * 100  # negative if already past stop
                urgency = held_days / max(distance_pct, 0.5)  # cap so past-stop positions rank highest
                ranked.append((sym, urgency, pnl, entry, cur, stop, held_days, distance_pct,
                               pos.get('pending_exit', False)))
            except Exception:
                continue

        if not ranked:
            self.notifier.send_notification("⚠️ No Data", "Could not fetch prices for any held positions.")
            return

        ranked.sort(key=lambda x: x[1], reverse=True)  # highest urgency first
        worst = ranked[:5]

        lines = [f"Top {len(worst)} of {len(ranked)} by urgency\n"]
        for i, (sym, urgency, pnl, entry, cur, stop, held_days, distance_pct, pending) in enumerate(worst, 1):
            flag = " ⏳" if pending else ""
            lines.append(
                f"{i}. {sym}{flag}  {pnl:+.1f}%  ({held_days}d)\n"
                f"   Stop: ${stop:.2f}  ({distance_pct:+.1f}% away)"
            )

        self.notifier.send_notification("📉 Most Urgent Positions", "\n".join(lines))
        print(f"✓ Sent PORTFOLIO WORST ({len(worst)} positions)")

    def handle_best_query(self):
        """Handle 'BEST' — show top-ranked buy signals from the hot list.

        Composite score = ML_score × freshness_decay × accuracy_weight × fin_factor

          freshness_decay = 0.85 ^ bars_ago
            Each bar since the signal reduces score by 15% — a fresh signal on the
            same close beats a stronger signal from 3 days ago.

          accuracy_weight = historical ML bullish accuracy for this symbol
            Break-even at 60% (weight 1.0). Below 60% penalises; above rewards.
            Neutral (1.0) when fewer than 20 samples exist (insufficient history).
            Range: ~0.2 (very low accuracy) → 1.4 (near-perfect accuracy).

          fin_factor = 0.7 + 0.3 × (fin_score / 9)
            Scales from 0.70 (worst financials, 0/9) to 1.0 (best, 9/9).
            When financials are unavailable: 0.85 (mild penalty for unknown).

        Ranking is two-pass: first rank all candidates by ML composite to select
        the top 20, then fetch financials for those 20, then re-rank with fin_factor.
        """
        _FRESHNESS_DECAY   = 0.85
        _ACC_BREAK_EVEN    = 60.0   # accuracy % where weight = 1.0
        _MIN_SAMPLES       = 20     # below this, treat accuracy as unknown → neutral
        _FIN_UNKNOWN       = 0.85   # fin_factor when financials unavailable

        def _accuracy_weight(acc, total):
            if acc is None or total < _MIN_SAMPLES:
                return 1.0  # insufficient history — neutral
            if acc >= _ACC_BREAK_EVEN:
                return 1.0 + (acc - _ACC_BREAK_EVEN) / 100.0   # 60%→1.0, 80%→1.2, 100%→1.4
            else:
                return max(0.2, acc / _ACC_BREAK_EVEN)           # 0%→0.2, 30%→0.5, 60%→1.0

        def _ml_composite(data):
            freshness = _FRESHNESS_DECAY ** data['bars_ago']
            acc_w = _accuracy_weight(data.get('bullish_accuracy'), data.get('bullish_total', 0))
            return data['score'] * freshness * acc_w

        def _fin_factor(fin):
            if fin is None:
                return _FIN_UNKNOWN
            return 0.7 + 0.3 * (fin.get('score', 0) / 9.0)

        def composite(data, fin=None):
            return _ml_composite(data) * _fin_factor(fin)

        self._cleanup_hot_list(save=False)

        if not self.hot_list:
            self.notifier.send_notification("📭 No Signals", "Hot list is empty. Run a scan first.")
            return

        held = set(self.position_manager.list_all().keys())
        today = datetime.now().date().isoformat()
        yesterday = (datetime.now().date() - timedelta(days=1)).isoformat()

        # Prefer signals from today/yesterday; fall back to all unexpired entries
        recent = {sym: data for sym, data in self.hot_list.items()
                  if data.get('added_date', '') >= yesterday}
        candidates = recent if recent else self.hot_list

        # Pass 1: ML-only rank to get top candidates; count held/available from full list
        ml_ranked = sorted(
            candidates.items(),
            key=lambda x: _ml_composite(x[1]),
            reverse=True
        )
        n_held_signals = sum(1 for sym, _ in ml_ranked if sym in held)
        n_available = len(ml_ranked) - n_held_signals

        if n_available == 0:
            self.notifier.send_notification(
                "📭 No New Signals",
                f"All {len(ml_ranked)} recent signal(s) are already held positions."
            )
            return

        # Fetch financials for top 20 candidates (I/O-bound, use threads)
        pre_candidates = ml_ranked[:20]
        fin_data = {}
        pre_syms = [sym for sym, _ in pre_candidates]
        try:
            with ThreadPoolExecutor(max_workers=min(len(pre_syms), 5)) as tex:
                fut_map = {tex.submit(self._get_quick_financials, sym): sym for sym in pre_syms}
                for fut in as_completed(fut_map):
                    sym = fut_map[fut]
                    try:
                        fin_data[sym] = fut.result()
                    except Exception:
                        fin_data[sym] = None
        except Exception:
            pass  # financial data is bonus; never block BEST output

        # Pass 2: re-rank top 20 using combined ML + financial score
        ranked_top = sorted(
            pre_candidates,
            key=lambda x: composite(x[1], fin_data.get(x[0])),
            reverse=True
        )
        top = ranked_top[:10]
        today_date = datetime.now().date()
        alerted_today = {sym for sym, date in self.buy_alerts_sent.items() if date == today_date}

        header = f"{len(top)} of {len(ml_ranked)}"
        if n_held_signals:
            header += f"  ({n_held_signals} held)"
        lines = [header]

        for i, (sym, data) in enumerate(top, 1):
            bars_ago     = data['bars_ago']
            acc          = data.get('bullish_accuracy')
            total        = data.get('bullish_total', 0)
            fin          = fin_data.get(sym)
            comp         = composite(data, fin)
            signal_price = data.get('signal_price')
            added        = data.get('added_date', '')

            freshness  = "today" if bars_ago == 0 else f"{bars_ago}d ago"
            acc_str    = f"{acc:.0f}%" if acc is not None and total >= _MIN_SAMPLES else "?"
            price_str  = f"  ${signal_price:.2f}" if signal_price else ""
            alert_mark = "  ✉" if sym in alerted_today else ""
            age_mark   = f"  [{added}]" if added < today else ""
            held_mark  = "✓" if sym in held else ""

            # Line 1: rank · symbol (bold) · price · freshness · markers
            # Leading \n creates a blank line between entries when joined
            lines.append(f"\n{i}. {held_mark}{_bold(sym)}{price_str}  {freshness}{alert_mark}{age_mark}")

            # Line 2: financials (priority) then combined composite
            detail_parts = []
            if fin:
                fin_str = f"F:{fin['score']}/9 {_bold(fin['assessment'])}"
                fv  = fin.get('avg_fair_value')
                fup = fin.get('upside')
                if fv and fup is not None:
                    fin_str += f" · Est ${fv:.0f} ({fup:+.0f}%)"
                elif fin.get('target_mean') and fin.get('current_price', 0) > 0:
                    t    = fin['target_mean']
                    t_up = (t / fin['current_price'] - 1) * 100
                    fin_str += f" · Est ${t:.0f} ({t_up:+.0f}%)"
                detail_parts.append(fin_str)
            detail_parts.append(f"Score:{comp:.0f} Acc:{acc_str}")
            lines.append("   " + "  ·  ".join(detail_parts))

        lines.append("\n─────────────────────")
        footer = "✓ held  ·  F = fundamentals/9  ·  Score = ML×freshness×acc×fin"
        if any(sym in alerted_today for sym, _ in top):
            footer += "  ·  ✉ alerted"
        lines.append(footer)

        self.notifier.send_notification("🏆 Best Buy Signals", "\n".join(lines))
        print(f"✓ Sent BEST signals: {len(top)} shown ({len(ml_ranked)} total, {n_held_signals} held)")

    def handle_help_query(self):
        """Handle 'HELP' command — list all available commands."""
        message = (
            "-- Positions --\n"
            "BOUGHT <SYM>\n"
            "BOUGHT <SYM> AT <PRICE>\n"
            "SOLD <SYM>\n"
            "HOLDINGS\n"
            "PORTFOLIO\n"
            "PORTFOLIO WORST\n"
            "CAPITAL SET <AMT>\n"
            "CAPITAL ADD <AMT>\n"
            "\n"
            "-- Analysis --\n"
            "BEST\n"
            "LAST <SYM>\n"
            "BACKTEST <SYM> [1M|3M|6M|1Y|2Y|3Y|5Y]\n"
            "ANALYZE <SYM>\n"
            "COMPARE <SYM1> <SYM2> ...\n"
            "\n"
            "-- Settings --\n"
            "ADD <SYM>\n"
            "REMOVE <SYM>\n"
            "UPDATE <SYM> <PRICE>\n"
            "REPLACE <OLD> <NEW>\n"
            "TIMEFRAME\n"
            "TIMEFRAME SET [1M|5M|15M|30M|1H|4H|1D]"
        )
        self.notifier.send_notification("Commands", message)

    def _get_reliable_price(self, yf_ticker):
        """
        Best-effort real-time price for exit decisions.

        Layer 1 — fast_info.last_price (single lightweight API call).
        Layer 2 — 1-minute intraday download (separate endpoint; immune to the
                   auto_adjust stale-close bug that affects daily bars).

        Returns None when both layers fail; caller falls back to df close.
        """
        # Layer 1: fast_info
        try:
            live_price = getattr(yf.Ticker(yf_ticker).fast_info, 'last_price', None)
            if live_price and live_price > 0:
                return float(live_price)
        except Exception:
            pass

        # Layer 2: recent 1-minute bars
        try:
            df_1m = yf.download(yf_ticker, period='1d', interval='1m',
                                progress=False, auto_adjust=False,
                                prepost=False, threads=False)
            if not df_1m.empty:
                if isinstance(df_1m.columns, pd.MultiIndex):
                    df_1m.columns = [col[0] for col in df_1m.columns]
                price = float(df_1m['Close'].iloc[-1])
                if price > 0:
                    return price
        except Exception:
            pass

        return None

    def check_exit(self, symbol):
        """Check if held position should be exited"""
        position = self.position_manager.get(symbol)
        if not position:
            return

        entry_price = position['entry_price']
        current_stop = position['stop_loss']
        yf_ticker = position.get('yf_ticker', symbol)
        exchange = position.get('exchange', '')

        # === Step 1: Get real-time price independently of bar data ===
        # This is the price used for all exit comparisons on daily bars.
        # fast_info.last_price → 1-minute bar fallback → None (fall back to df close)
        real_price = None
        on_daily = self.TIMEFRAMES[self.current_timeframe]['interval'] == '1d'
        if on_daily:
            real_price = self._get_reliable_price(yf_ticker)
            if real_price:
                print(f"   💰 {yf_ticker}: Real-time price ${real_price:.2f}")
            else:
                print(f"   ⚠️  {yf_ticker}: Real-time price unavailable, using bar close")

        # === Step 2: Fetch historical bars (needed for ATR, ML features, peak high) ===
        df = self.get_live_data(symbol, ticker=yf_ticker)
        if df is None:
            return

        # current_price for display: prefer real_price, fall back to bar close
        current_price = real_price if real_price else float(df['Close'].iloc[-1])

        # === Step 3: Run cerebro — computes chandelier stop level + ML/earnings exits ===
        # Chandelier math uses historical highs/ATR (bar data, unaffected by stale close).
        # The price comparison is intentionally removed from inside cerebro;
        # we do it below with real_price.
        # ml_lock prevents concurrent cerebro runs with user commands (LAST, BACKTEST, etc.)
        entry_date = position.get('entry_date')
        with self.ml_lock:
            sell_signal = self.strategy.get_exit_signal(
                df, self.params, entry_price, current_stop, symbol=symbol, entry_date=entry_date
            )

        # === Step 4: Chandelier stop check with real-time price ===
        # Cerebro returns the computed stop level.  We compare it against real_price,
        # not the stale daily close that was in the df.
        if not sell_signal['signal'] and on_daily and real_price:
            chandelier_stop = sell_signal.get('chandelier_stop', 0)
            if chandelier_stop > 0 and real_price < chandelier_stop:
                peak = sell_signal.get('chandelier_peak', 0)
                mult = sell_signal.get('chandelier_mult', 0)
                bar_date = datetime.now().strftime('%Y-%m-%d')
                sell_signal = {
                    'signal': True,
                    'price': real_price,
                    'stop_type': (
                        f"CHANDELIER STOP ({real_price:.2f} < {chandelier_stop:.2f}"
                        + (f", peak={peak:.2f}" if peak else "")
                        + (f", mult={mult:.2f}x)" if mult else ")")
                    ),
                    'new_stop': chandelier_stop,
                    'bars_ago': 0,
                    'bar_date': bar_date,
                }

        # For any signal, always use real_price in the notification if available
        if sell_signal['signal'] and real_price:
            sell_signal['price'] = real_price

        already_alerted = position.get('pending_exit', False)

        if sell_signal['signal']:
            bars_ago = sell_signal.get('bars_ago', 0)
            bar_date = sell_signal.get('bar_date', 'today')

            if already_alerted:
                # Alert already sent — don't spam. Position is still monitored;
                # user chose to hold past the signal. Just log it.
                print(f"   ⏳ {symbol}: Still below stop (${sell_signal['price']:.2f}), "
                      f"alert already sent — waiting for 'SOLD {symbol}'")
            else:
                if bars_ago > 0:
                    print(f"⚠️  {symbol} hit stop {bars_ago} bar(s) ago on {bar_date}!")

                # Send SELL alert but DON'T auto-remove position.
                # User must confirm with "SOLD SYMBOL".
                self.notifier.send_sell_alert(symbol, sell_signal, entry_price, exchange=exchange)
                exchange_str = f" [{exchange}]" if exchange else ""
                print(f"🔴 SELL ALERT: {symbol}{exchange_str} - Stop hit at ${sell_signal['price']:.2f} on {bar_date}")
                print(f"   ⏳ Waiting for confirmation: Reply 'SOLD {symbol}' to remove position")

                # Mark as pending to suppress duplicate alerts for the same condition
                position['pending_exit'] = True
                position['exit_alerted_date'] = bar_date
                self.position_manager.positions[symbol] = position
                self.position_manager._save()

        else:
            # No exit signal — update trailing stop
            chandelier_stop = sell_signal.get('chandelier_stop', 0)
            new_stop = chandelier_stop if chandelier_stop > current_stop else sell_signal['new_stop']
            self.position_manager.update_stop(symbol, new_stop)

            if already_alerted:
                # Price recovered above stop — clear the pending flag so a future
                # dip will trigger a fresh alert
                position['pending_exit'] = False
                if 'exit_alerted_date' in position:
                    del position['exit_alerted_date']
                self.position_manager.positions[symbol] = position
                self.position_manager._save()
                print(f"   ✅ {symbol}: Price recovered above stop — exit alert cleared")

            # Show stop update info (only if it changed significantly)
            if abs(new_stop - current_stop) > 0.01:
                pnl = ((current_price / entry_price) - 1) * 100
                distance_to_stop = ((current_price - new_stop) / current_price) * 100
                print(f"📊 {symbol}: Price ${current_price:.2f} | P&L {pnl:+.2f}% | "
                      f"Stop ${new_stop:.2f} ({distance_to_stop:.1f}% away)")

    def _et_now(self):
        """Current datetime in US/Eastern timezone."""
        import pytz
        return datetime.now(pytz.timezone('US/Eastern'))

    def _interruptible_sleep(self, seconds):
        """Sleep for up to `seconds`, waking immediately if scanning is stopped."""
        for _ in range(int(seconds)):
            if not self.scanning:
                break
            time.sleep(1)

    def is_market_hours(self):
        """Check if market is open (US Eastern Time)"""
        from datetime import datetime
        import pytz

        # Get current time in US Eastern timezone
        eastern = pytz.timezone('US/Eastern')
        now_et = datetime.now(eastern)

        # Weekend
        if now_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

        is_open = market_open <= now_et <= market_close

        if not is_open:
            print(f"Market closed (ET time: {now_et.strftime('%I:%M %p')})")
        return is_open

    def send_market_open_notification(self):
        """Send notification when market opens (once per day)"""
        from datetime import datetime
        import pytz

        eastern = pytz.timezone('US/Eastern')
        now_et = datetime.now(eastern)
        today = now_et.date()

        # Only send once per day
        if self.market_open_notified_date == today:
            return

        # Get position summary for the notification
        summary = self.position_manager.get_summary()
        position_count = summary['count']

        title = "🔔 Market Open"
        message = (
            f"The stock market is now open.\n"
            f"Time: {now_et.strftime('%I:%M %p')} ET\n"
            f"Watching: {len(self.watchlist)} symbols\n"
            f"Active positions: {position_count}"
        )

        self.notifier.send_notification(title, message)
        self.market_open_notified_date = today
        print(f"📢 Market open notification sent")

    def _scan_loop(self, scan_interval):
        """Background scanning loop - runs in separate thread.

        Time-aware scan schedule (all times US/Eastern):

          Weekdays only.  Two modes alternate automatically:

          DISCOVERY — full watchlist scan (~486 symbols).
            Triggered when:
              (a) system starts / never run today, AND in the pre-market window, OR
              (b) it has been >= DISCOVERY_INTERVAL_HOURS since the last discovery
                  at any point during the active window (pre-market or market hours).
            This means discovery runs pre-market, then refreshes mid-morning,
            then mid-afternoon — catching new signals that emerge as price action
            develops throughout the day.

          HOT SCAN — only hot-list symbols (seconds, not hours).
            Runs every scan_interval minutes when discovery is not due.
            Keeps alerting on already-identified candidates quickly.

          Pre-market window (PRE_MARKET_START_HOUR → 9:30 AM):
            If discovery is due, it runs here. If discovery is done and market
            isn't open yet, sleep quietly until open.

          After hours / overnight:
            Sleep, polling every 5 minutes.

          Weekends: sleep 30 minutes between checks.
        """
        while self.scanning:
            try:
                now_et = self._et_now()
                weekday = now_et.weekday()

                # ── Weekend ──────────────────────────────────────────────────
                if weekday >= 5:
                    self._interruptible_sleep(1800)
                    continue

                # ── Time boundaries ──────────────────────────────────────────
                market_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
                market_close = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
                pre_start    = now_et.replace(hour=self.PRE_MARKET_START_HOUR,
                                              minute=0, second=0, microsecond=0)

                in_pre_market = pre_start <= now_et < market_open
                in_market     = market_open <= now_et < market_close

                # ── Is a discovery scan due? ─────────────────────────────────
                if self.last_discovery_time is None:
                    hours_since = float('inf')
                    discovery_due = True
                else:
                    hours_since = (now_et - self.last_discovery_time).total_seconds() / 3600
                    discovery_due = hours_since >= self.DISCOVERY_INTERVAL_HOURS

                # ── Pre-market window ────────────────────────────────────────
                if in_pre_market:
                    if discovery_due:
                        mins_to_open = int((market_open - now_et).total_seconds() / 60)
                        print(f"\n📡 PRE-MARKET DISCOVERY — {len(self.watchlist)} symbols  "
                              f"({mins_to_open} min before open)")
                        self.scan_for_opportunities(symbols=None)
                        self.last_discovery_time = self._et_now()
                        print(f"✅ Discovery complete. "
                              f"Hot list: {len(self.hot_list)} symbol(s).")
                        # Don't sleep — re-evaluate immediately (may now be market hours)
                    else:
                        mins_to_open = int((market_open - now_et).total_seconds() / 60)
                        hrs_to_next = self.DISCOVERY_INTERVAL_HOURS - hours_since
                        print(f"⏳ Pre-market: discovery done, market opens in {mins_to_open} min "
                              f"(next discovery in ~{hrs_to_next:.1f}h). "
                              f"Hot list: {len(self.hot_list)} symbol(s).")
                        self._interruptible_sleep(60)

                # ── Market hours ─────────────────────────────────────────────
                elif in_market:
                    self.send_market_open_notification()

                    if discovery_due:
                        hours_since_str = (f"{hours_since:.1f}h ago"
                                           if self.last_discovery_time else "never")
                        print(f"\n📡 DISCOVERY REFRESH — {len(self.watchlist)} symbols "
                              f"(last run: {hours_since_str})")
                        self.scan_for_opportunities(symbols=None)
                        self.last_discovery_time = self._et_now()
                        print(f"✅ Discovery refresh complete. "
                              f"Hot list: {len(self.hot_list)} symbol(s).")
                    else:
                        self._cleanup_hot_list()
                        hot_symbols = list(self.hot_list.keys())
                        hrs_to_next = self.DISCOVERY_INTERVAL_HOURS - hours_since

                        if hot_symbols:
                            print(f"\n⚡ HOT SCAN — {len(hot_symbols)} symbol(s) "
                                  f"(discovery refresh in ~{hrs_to_next:.1f}h)")
                            self.scan_for_opportunities(symbols=hot_symbols)
                        else:
                            print(f"\nℹ️  Hot list empty. "
                                  f"Discovery refresh in ~{hrs_to_next:.1f}h.")

                    print(f"\n💤 Next scan in {scan_interval} minutes...")
                    self._interruptible_sleep(scan_interval * 60)

                # ── After hours / overnight ──────────────────────────────────
                else:
                    if now_et < pre_start:
                        secs_to_pre = int((pre_start - now_et).total_seconds())
                    else:
                        tomorrow_pre = pre_start + timedelta(days=1)
                        secs_to_pre = int((tomorrow_pre - now_et).total_seconds())

                    hrs  = secs_to_pre // 3600
                    mins = (secs_to_pre % 3600) // 60
                    print(f"After hours ({now_et.strftime('%I:%M %p')} ET). "
                          f"Pre-market scan at {self.PRE_MARKET_START_HOUR}:00 AM ET "
                          f"(~{hrs}h {mins}m away). Sleeping...")
                    self._interruptible_sleep(300)

            except Exception as e:
                print(f"❌ Error in scan loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(60)

        print("🛑 Scan loop stopped")

    def _reset_stale_pending_exits(self):
        """Clear pending_exit flags that were set on a previous calendar day.

        Each day is a fresh slate — if the stop is still breached, a new alert
        fires naturally on the next exit check.
        """
        today = datetime.now().date().isoformat()
        positions = self.position_manager.list_all()
        cleared = 0
        for symbol, position in positions.items():
            if position.get('pending_exit'):
                alerted_date = position.get('exit_alerted_date', '')
                if alerted_date != today:
                    position['pending_exit'] = False
                    if 'exit_alerted_date' in position:
                        del position['exit_alerted_date']
                    self.position_manager.positions[symbol] = position
                    cleared += 1
        if cleared > 0:
            self.position_manager._save()
            print(f"🔄 Reset {cleared} stale pending exit flag(s) (from previous day)")

    def _exit_scan_loop(self, exit_interval=60):
        """Background exit scanning loop - checks held positions for sell signals"""
        while self.scanning:
            try:
                if self.is_market_hours():
                    # Reset any pending_exit flags left over from a previous day
                    self._reset_stale_pending_exits()

                    held_positions = list(self.position_manager.list_all().keys())

                    if held_positions:
                        print(f"\n🔍 [EXIT] Checking {len(held_positions)} positions for exits...")
                        for symbol in held_positions:
                            if not self.scanning:
                                break
                            try:
                                position = self.position_manager.get(symbol)
                                if position and position.get('pending_exit'):
                                    print(f"   ⏳ {symbol}: Exit alert pending, waiting for 'SOLD {symbol}'")
                                # Always run exit check — pending_exit only suppresses re-alerting,
                                # not monitoring. If price recovers above stop, flag is cleared.
                                self.check_exit(symbol)
                            except Exception as e:
                                print(f"   ❌ Error checking exit for {symbol}: {e}")

                        print(f"✅ [EXIT] Check complete. Next check in {exit_interval}s...")

                    # Sleep in small increments to allow quick shutdown
                    for _ in range(exit_interval):
                        if not self.scanning:
                            break
                        time.sleep(1)
                else:
                    # Market closed - check less frequently
                    for _ in range(300):
                        if not self.scanning:
                            break
                        time.sleep(1)

            except Exception as e:
                print(f"❌ Error in exit scan loop: {e}")
                time.sleep(60)

        print("🛑 Exit scan loop stopped")

    def run(self, scan_interval=15, exit_interval=60):
        """Main entry point - starts scanning threads and keeps main thread alive

        Args:
            scan_interval: Minutes between buy scans (default 15)
            exit_interval: Seconds between exit/sell checks (default 60)
        """
        tf_config = self.TIMEFRAMES[self.current_timeframe]
        print(f"\n🚀 Live Trading Monitor Started")
        print(f"Watching {len(self.watchlist)} symbols")
        print(f"Active positions: {self.position_manager.get_summary()['count']}")
        print(f"Timeframe: {self.current_timeframe} ({tf_config['description']})")
        print(f"Buy scan interval: {scan_interval} minutes")
        print(f"Exit check interval: {exit_interval} seconds")
        print(f"Threading: Buy scan, exit scan, and input all run independently")
        print(f"Reply listener: ACTIVE (responds instantly to your texts)\n")

        # Start threads
        self.scanning = True

        # Buy scanning thread
        self.scan_thread = threading.Thread(target=self._scan_loop, args=(scan_interval,), daemon=True)
        self.scan_thread.start()
        print("✓ Buy scan thread started")

        # Exit scanning thread
        self.exit_thread = threading.Thread(target=self._exit_scan_loop, args=(exit_interval,), daemon=True)
        self.exit_thread.start()
        print("✓ Exit scan thread started")

        try:
            # Keep main thread alive for keyboard interrupt
            while self.scanning:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Stopping monitor...")
        finally:
            # Stop all threads
            self.scanning = False
            if self.scan_thread:
                self.scan_thread.join(timeout=5)
            if self.exit_thread:
                self.exit_thread.join(timeout=5)
            self.notifier.stop_listening()
            print("👋 Monitor stopped")

    def handle_last_signal_query(self, reply):
        """
        Handle 'LAST SYMBOL' query - shows last signal for a stock with both 30-day and 3-month charts

        Args:
            reply: The reply string like "LAST NVDA"
        """
        parts = reply.split()

        if len(parts) < 2:
            self.notifier.send_notification(
                "⚠️ Invalid Format",
                "Usage: LAST <SYMBOL>\nExample: LAST NVDA"
            )
            return

        symbol = self._resolve_symbol(parts[1])

        # Send immediate acknowledgment
        print(f"⏳ Fetching signal data for {symbol}... this may take a moment")

        # Chart configurations with different intervals
        # 30-day chart: hourly bars for more detail
        # 3-month chart: daily bars for broader view
        chart_configs = [
            {
                'period': '60d',      # Fetch 60 days of hourly data
                'interval': '1h',
                'bars': 30 * 7,       # ~30 days * ~7 trading hours
                'title': '30 Day (Hourly)'
            },
            {
                'period': '2y',       # Fetch 2 years of daily data
                'interval': '1d',
                'bars': 90,           # 90 trading days
                'title': '3 Month (Daily)'
            }
        ]

        print(f"📊 Fetching multi-timeframe data for {symbol}...")

        # Fetch data for each timeframe
        chart_data_list = []
        df_daily = None  # Keep daily data for signal checking

        for config in chart_configs:
            print(f"   Fetching {config['interval']} data (period: {config['period']})...")
            df = self.get_live_data(symbol, period=config['period'], interval=config['interval'])

            if df is not None and len(df) > 0:
                print(f"   Got {len(df)} bars of {config['interval']} data")
                chart_data_list.append({
                    'df': df,
                    'bars': config['bars'],
                    'title': config['title'],
                    'interval': config['interval']
                })

                # Keep daily data for status message calculations
                if config['interval'] == '1d':
                    df_daily = df
            else:
                print(f"   ⚠️ Failed to fetch {config['interval']} data")

        if not chart_data_list:
            self.notifier.send_notification(
                f"❌ {symbol}",
                f"Unable to fetch data for any timeframe"
            )
            return

        # Use daily data for status calculations, fall back to first available
        df = df_daily if df_daily is not None else chart_data_list[0]['df']
        print(f"📊 Using {len(df)} bars for signal detection")

        # Build the status message
        position = self.position_manager.get(symbol)
        if position:
            entry_date = position['entry_date'][:10]
            entry_price = position['entry_price']
            stop_loss = position['stop_loss']

            # Get current price and P&L
            current_price = df['Close'].iloc[-1]
            pnl = ((current_price / entry_price) - 1) * 100

            title = f"📊 {symbol} - Current Position"
            message = (
                f"Signal: BUY\n"
                f"Date: {entry_date}\n"
                f"Entry: ${entry_price:.2f}\n"
                f"Current: ${current_price:.2f}\n"
                f"P&L: {pnl:+.2f}%\n"
                f"Stop: ${stop_loss:.2f}"
            )

            if position.get('pending_exit'):
                message += f"\n\n⚠️ Exit signal active!\nReply: SOLD {symbol}"

        else:
            # Not holding - run ML strategy to detect recent signals
            print(f"📊 Running ML strategy signal detection for {symbol}...")
            with self.ml_lock:
                signal = self.strategy.get_entry_signal(df, self.params, symbol=symbol)

            if signal.get('signal'):
                signal_type = signal.get('signal_type', 'BUY')
                signal_date = signal.get('date', df.index[-1])
                if hasattr(signal_date, 'strftime'):
                    signal_date = signal_date.strftime('%Y-%m-%d')
                else:
                    signal_date = str(signal_date)

                signal_price = signal.get('price', df['Close'].iloc[-1])
                current_price = signal.get('current_price', df['Close'].iloc[-1])
                bars_ago = signal.get('bars_ago', 0)
                prediction = signal.get('prediction', 0)

                pnl = ((current_price / signal_price) - 1) * 100

                title = f"📊 {symbol} - Last Signal"
                message = (
                    f"Signal: {signal_type}\n"
                    f"Date: {signal_date} ({bars_ago} bar(s) ago)\n"
                    f"Price then: ${signal_price:.2f}\n"
                    f"Price now: ${current_price:.2f}\n"
                    f"Change: {pnl:+.2f}%\n"
                    f"ML Prediction: {prediction:+d}\n\n"
                    f"⚠️ Not currently holding"
                )
                print(f"✓ Found {signal_type} signal from {signal_date}")
            else:
                title = f"📊 {symbol} - Last Signal"
                message = "No recent BUY/SELL signals in the last 5 bars"
                print(f"✓ No recent signals found for {symbol}")

        # Generate stacked multi-timeframe chart (with lock to prevent concurrent ML)
        print(f"📊 Generating stacked chart for {symbol} (30 Day Hourly + 3 Month Daily)...")
        with self.ml_lock:
            chart_buffer = self.chart_gen.generate_multi_timeframe_chart(
                symbol, chart_data_list
            )

        if chart_buffer:
            self.notifier.send_notification_with_image(
                title, message, chart_buffer, f"{symbol}_chart.png"
            )
            print(f"✓ Sent stacked chart for {symbol}")
        else:
            # Fallback to text only
            self.notifier.send_notification(title, message)
            print(f"✓ Sent last signal info for {symbol} (text only, chart generation failed)")

    def handle_backtest_query(self, reply):
        """
        Handle 'BACKTEST SYMBOL PERIOD' query - runs a backtest and returns results

        Args:
            reply: The reply string like "BACKTEST NVDA 1Y" or "BACKTEST AAPL 6M"
        """
        parts = reply.split()

        if len(parts) < 2:
            self.notifier.send_notification(
                "⚠️ Invalid Format",
                "Usage: BACKTEST <SYMBOL> [PERIOD]\n"
                "Periods: 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y\n"
                "Example: BACKTEST NVDA 1Y"
            )
            return

        symbol = self._resolve_symbol(parts[1])
        period = parts[2].upper() if len(parts) >= 3 else "1Y"

        # Send immediate acknowledgment
        print(f"⏳ Running backtest for {symbol} ({period})... this may take a minute")

        # Parse period to get start/end dates
        period_map = {
            '1M': 30,
            '3M': 90,
            '6M': 180,
            '1Y': 365,
            '2Y': 730,
            '3Y': 1095,
            '5Y': 1825,
        }

        if period not in period_map:
            self.notifier.send_notification(
                "⚠️ Invalid Period",
                f"Unknown period: {period}\n"
                "Valid periods: 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y"
            )
            return

        days = period_map[period]

        tf_config = self.TIMEFRAMES[self.current_timeframe]
        print(f"📊 Running backtest for {symbol} over {period} ({self.current_timeframe} bars)...")
        self.notifier.send_notification(
            f"⏳ Backtest Started",
            f"Running {period} backtest for {symbol}...\n"
            f"Timeframe: {self.current_timeframe} ({tf_config['description']})\n"
            f"This may take a moment."
        )

        try:
            # Run backtest with lock to prevent concurrent ML operations
            with self.ml_lock:
                results = self._run_backtest(symbol, days)

            if results is None:
                self.notifier.send_notification(
                    f"❌ Backtest Failed",
                    f"Could not run backtest for {symbol}\nInsufficient data or error occurred."
                )
                return

            # Format results message
            title = f"📊 {symbol} Backtest ({period} / {self.current_timeframe})"

            # Build detailed message
            message_lines = [
                f"Period: {results['start_date']} to {results['end_date']}",
                f"Timeframe: {self.current_timeframe} ({tf_config['description']})",
                f"",
                f"💰 Returns:",
                f"  Total: {results['total_return_pct']:+.2f}%",
                f"  Annualized: {results['annualized_return']:+.2f}%",
                f"  vs SPY: {results['spy_return']:+.2f}%",
                f"  Alpha: {results['total_return_pct'] - results['spy_return']:+.2f}%",
                f"",
                f"📈 Trades:",
                f"  Total: {results['total_trades']}",
                f"  Win Rate: {results['win_rate']:.1f}%",
                f"  Profit Factor: {results['profit_factor']:.2f}",
                f"",
                f"📉 Risk:",
                f"  Max Drawdown: {results['max_drawdown']:.2f}%",
                f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}",
            ]

            if results['total_trades'] > 0:
                message_lines.extend([
                    f"",
                    f"💵 Avg Trade:",
                    f"  Win: ${results['avg_win']:.2f}",
                    f"  Loss: ${results['avg_loss']:.2f}",
                ])

            # Add ML stats if available
            ml_stats = results.get('ml_stats', {})
            ml_diag = results.get('ml_diagnostics', {})

            if ml_stats.get('total', 0) > 0:
                message_lines.extend([
                    f"",
                    f"🤖 ML Accuracy:",
                    f"  Overall: {ml_stats.get('accuracy_pct', 0):.1f}%",
                    f"  Bullish: {ml_stats.get('bullish_accuracy_pct', 0):.1f}% ({ml_stats.get('bullish_total', 0)} predictions)",
                    f"  Bearish: {ml_stats.get('bearish_accuracy_pct', 0):.1f}% ({ml_stats.get('bearish_total', 0)} predictions)",
                ])

            if ml_diag.get('total_bars', 0) > 0:
                message_lines.extend([
                    f"",
                    f"🔄 Signal Behavior:",
                    f"  Signal Flips: {ml_diag.get('signal_changes', 0)}",
                    f"  Bullish %: {ml_diag.get('bullish_pct', 0):.1f}%",
                    f"  Bearish %: {ml_diag.get('bearish_pct', 0):.1f}%",
                    f"  Neutral %: {ml_diag.get('neutral_pct', 0):.1f}%",
                ])

                # Add filter blocking stats if relevant
                if ml_diag.get('entry_attempts', 0) > 0:
                    blocked_by_kernel = ml_diag.get('kernel_block_pct', 0)
                    if blocked_by_kernel > 0:
                        message_lines.append(f"  Blocked by Kernel: {blocked_by_kernel:.1f}%")

            # Add exposure stats
            time_in_market = results.get('time_in_market_pct', 0)
            tradeable_q = results.get('tradeable_quarters', 0)
            total_q = results.get('total_quarters', 0)

            message_lines.extend([
                f"",
                f"⏱️ Exposure:",
                f"  Time in Market: {time_in_market:.1f}%",
            ])

            if total_q > 0:
                message_lines.append(
                    f"  Tradeable Quarters: {tradeable_q}/{total_q} ({(tradeable_q/total_q)*100:.0f}%)"
                )

            message = "\n".join(message_lines)

            # Generate backtest chart with buy/sell signals
            chart_buf = None
            try:
                chart_buf = self.chart_gen.generate_backtest_chart(
                    symbol=symbol,
                    df=results['chart_df'],
                    test_start_idx=results['test_start_idx'],
                    buy_signals=results['buy_signals'],
                    sell_signals=results['sell_signals'],
                    period_label=period,
                    interval=results['interval'],
                    results=results,
                    earnings_data=results.get('earnings_data', []),
                    fair_value_data=results.get('fair_value_history', []),
                    hist_pe_data=results.get('hist_pe_fair_value', []),
                )
            except Exception as chart_error:
                print(f"⚠️ Chart generation failed: {chart_error}")

            # Send notification with chart if available
            if chart_buf:
                sent = self.notifier.send_notification_with_image(title, message, chart_buf, f"{symbol}_backtest.png")
                if sent:
                    print(f"✓ Sent backtest results for {symbol}")
                else:
                    print(f"❌ Failed to send backtest results for {symbol}")
            else:
                self.notifier.send_notification(title, message)
                print(f"✓ Sent backtest results for {symbol} (no chart)")

        except Exception as e:
            print(f"❌ Backtest error: {e}")
            import traceback
            traceback.print_exc()
            self.notifier.send_notification(
                f"❌ Backtest Error",
                f"Error running backtest for {symbol}:\n{str(e)[:100]}"
            )

    def _run_backtest(self, symbol, days):
        """
        Run a backtest for the given symbol and period.

        Args:
            symbol: Stock ticker
            days: Number of calendar days to backtest

        Returns:
            dict: Backtest results or None on failure
        """
        import backtrader as bt
        import numpy as np

        # Get current timeframe settings
        tf_config = self.TIMEFRAMES[self.current_timeframe]
        interval = tf_config['interval']
        tf_description = tf_config['description']

        # Calculate dates
        end_date = datetime.now()
        test_start = end_date - timedelta(days=days)

        print(f"   Timeframe: {self.current_timeframe} ({tf_description})")

        # Fetch data
        try:
            import yfinance as yf

            # Intraday intervals: yfinance caps how far back they go — use period parameter.
            # Daily: use period='max' so yfinance returns all available history.
            #   Using an explicit start date derived from warmup_days (7000 bars = ~28 years)
            #   causes yfinance to silently return an empty DataFrame for very old start dates.
            #   period='max' is reliable for any ticker; test_start_idx below selects the window.
            if interval in ['1m', '5m', '15m', '30m', '1h', '4h']:
                max_periods = {
                    '1m': '7d',
                    '5m': '60d',
                    '15m': '60d',
                    '30m': '60d',
                    '1h': '730d',
                    '4h': '730d',
                }
                period = max_periods.get(interval, '60d')
                df = yf.download(symbol, period=period, interval=interval,
                                 progress=False, auto_adjust=True, threads=False)
            else:
                df = yf.download(symbol, period='max', interval=interval,
                                 progress=False, auto_adjust=True, threads=False)

            if df.empty or len(df) < 100:
                print(f"   ❌ Insufficient data for {symbol}")
                return None

            df.index = df.index.tz_localize(None)

            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.columns = [c.lower() for c in df.columns]

            # Remove duplicate columns (keep first)
            df = df.loc[:, ~df.columns.duplicated()]

            # Fix zero-range bars (high == low) to prevent division-by-zero
            # in backtrader's ADX indicator which divides by ATR internally
            zero_range = df['high'] == df['low']
            if zero_range.any():
                epsilon = df['close'][zero_range] * 1e-6
                df.loc[zero_range, 'high'] = df.loc[zero_range, 'high'] + epsilon
                df.loc[zero_range, 'low'] = df.loc[zero_range, 'low'] - epsilon

            # Find test start index based on the requested test period
            # For intraday, calculate based on number of bars we want to test
            total_bars = len(df)

            if interval in ['1m', '5m', '15m', '30m', '1h', '4h']:
                # For intraday: calculate how many bars correspond to the test period
                bars_per_day = {'1m': 390, '5m': 78, '15m': 26, '30m': 13, '1h': 7, '4h': 2}
                bpd = bars_per_day.get(interval, 7)
                test_bars = int(days * bpd * 5 / 7)  # Approximate trading days

                # Ensure we have enough warmup
                min_warmup = min(self.warmup_days, total_bars // 2)
                test_start_idx = max(min_warmup, total_bars - test_bars)
            else:
                # Daily: use date-based calculation
                test_start_mask = df.index >= pd.Timestamp(test_start)
                if not test_start_mask.any():
                    print(f"   ❌ No data after test start date")
                    return None
                test_start_idx = test_start_mask.argmax()

            # Ensure test_start_idx is valid
            if test_start_idx >= total_bars - 10:
                test_start_idx = max(0, total_bars - 100)

            print(f"   Got {len(df)} bars ({interval}), test period starts at bar {test_start_idx}")

            # Calculate SPY return over the test period
            spy_return = 0
            try:
                # Get the actual test date range from the stock data
                test_start_date = df.index[test_start_idx]
                test_end_date = df.index[-1]

                # Fetch SPY for just the test period dates
                if interval in ['1m', '5m', '15m', '30m', '1h', '4h']:
                    spy_df = yf.download('SPY', period=period, interval=interval, progress=False)
                else:
                    spy_df = yf.download('SPY', start=test_start_date.strftime('%Y-%m-%d'),
                                        end=(test_end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                                        interval=interval, progress=False)

                if not spy_df.empty:
                    spy_df.index = spy_df.index.tz_localize(None)

                    # Handle multi-level columns from yfinance
                    if isinstance(spy_df.columns, pd.MultiIndex):
                        spy_df.columns = spy_df.columns.get_level_values(0)

                    spy_df.columns = [c.lower() for c in spy_df.columns]
                    spy_df = spy_df.loc[:, ~spy_df.columns.duplicated()]

                    # For intraday, filter to test date range
                    if interval in ['1m', '5m', '15m', '30m', '1h', '4h']:
                        spy_df = spy_df[(spy_df.index >= test_start_date) & (spy_df.index <= test_end_date)]

                    if len(spy_df) > 1 and 'close' in spy_df.columns:
                        spy_start_price = float(spy_df['close'].iloc[0])
                        spy_end_price = float(spy_df['close'].iloc[-1])
                        if spy_start_price > 0:
                            spy_return = ((spy_end_price / spy_start_price) - 1) * 100
            except Exception as spy_err:
                print(f"   ⚠️ Could not calculate SPY benchmark: {spy_err}")
                spy_return = 0

        except Exception as e:
            print(f"   ❌ Data fetch error: {e}")
            return None

        # Run backtrader
        try:
            cerebro = bt.Cerebro(stdstats=False)

            # Add data
            data = bt.feeds.PandasData(
                dataname=df,
                datetime=None,
                open='open',
                high='high',
                low='low',
                close='close',
                volume='volume'
            )
            cerebro.adddata(data)

            # Prepare params: strip live-trader-only keys the strategy doesn't know about
            strategy_params = self.strategy.filter_strategy_params(self.params)
            strategy_params['verbose'] = False
            strategy_params['test_start_idx'] = test_start_idx

            # Set per-symbol params for cross-symbol training and fundamentals
            strategy_params['cross_symbol_target_symbol'] = symbol
            strategy_params['fundamental_symbol'] = symbol

            # Create a strategy wrapper that captures buy/sell signals
            parent_strategy_class = self.strategy.strategy_class

            class SignalCaptureStrategy(parent_strategy_class):
                def __init__(self):
                    super().__init__()
                    self.buy_signals = []
                    self.sell_signals = []

                def _execute_buy(self):
                    # Capture buy signal
                    self.buy_signals.append({
                        'date': self.data.datetime.date(0),
                        'price': self.data.close[0],
                        'bar': len(self)
                    })
                    super()._execute_buy()

                def _close_position(self, reason):
                    # Capture sell signal
                    self.sell_signals.append({
                        'date': self.data.datetime.date(0),
                        'price': self.data.close[0],
                        'reason': reason,
                        'bar': len(self)
                    })
                    super()._close_position(reason)

            # Add strategy
            cerebro.addstrategy(SignalCaptureStrategy, **strategy_params)

            # Broker settings
            initial_cash = 10000
            cerebro.broker.setcash(initial_cash)
            cerebro.broker.setcommission(commission=0.0)
            cerebro.broker.set_coc(True)  # Fill at signal bar's close, matching backtest.py

            # Add analyzers
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")

            # Add portfolio value observer
            class PortfolioValue(bt.Observer):
                lines = ('value',)
                plotinfo = dict(plot=False)
                def next(self):
                    self.lines.value[0] = self._owner.broker.getvalue()
                def prenext(self):
                    self.lines.value[0] = self._owner.broker.getvalue()

            cerebro.addobserver(PortfolioValue)

            print(f"   Running backtest...")
            results = cerebro.run()
            strat = results[0]

            # Extract results
            trades = strat.analyzers.trades.get_analysis()

            # Get portfolio values for test period
            test_values = []
            observer = strat.observers.portfoliovalue
            for i in range(len(observer.lines.value)):
                if i >= test_start_idx:
                    try:
                        val = observer.lines.value.array[i]
                        if not np.isnan(val) and val > 0:
                            test_values.append(val)
                    except:
                        break

            if len(test_values) < 2:
                test_values = [initial_cash, cerebro.broker.getvalue()]

            final_value = test_values[-1]
            total_return_pct = ((final_value / initial_cash) - 1) * 100

            # Calculate metrics
            total_trades = trades.get('total', {}).get('total', 0)

            if total_trades > 0:
                win_count = trades.get('won', {}).get('total', 0)
                win_rate = (win_count / total_trades) * 100

                total_win_pnl = trades.get('won', {}).get('pnl', {}).get('total', 0)
                total_loss_pnl = abs(trades.get('lost', {}).get('pnl', {}).get('total', 0))
                profit_factor = (total_win_pnl / total_loss_pnl) if total_loss_pnl > 0 else 999

                avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0)
                avg_loss = abs(trades.get('lost', {}).get('pnl', {}).get('average', 0))
            else:
                win_rate = 0
                profit_factor = 0
                avg_win = 0
                avg_loss = 0

            # Calculate drawdown from test period
            peak = test_values[0]
            max_dd = 0
            for val in test_values:
                if val > peak:
                    peak = val
                dd = ((peak - val) / peak) * 100
                if dd > max_dd:
                    max_dd = dd

            # Calculate Sharpe ratio (with proper annualization for timeframe)
            if len(test_values) > 1:
                bar_returns = []
                for i in range(1, len(test_values)):
                    ret = (test_values[i] / test_values[i-1]) - 1
                    bar_returns.append(ret)

                if len(bar_returns) > 1 and np.std(bar_returns) > 0:
                    # Annualization factor depends on bar frequency
                    if interval in ['1m', '5m', '15m', '30m', '1h', '4h']:
                        bars_per_day = {'1m': 390, '5m': 78, '15m': 26, '30m': 13, '1h': 7, '4h': 2}
                        bpd = bars_per_day.get(interval, 7)
                        annualization_factor = np.sqrt(252 * bpd)
                    else:
                        annualization_factor = np.sqrt(252)

                    sharpe = (np.mean(bar_returns) / np.std(bar_returns)) * annualization_factor
                else:
                    sharpe = 0
            else:
                sharpe = 0

            # Annualized return (account for bar frequency)
            test_bars = len(test_values)
            if interval in ['1m', '5m', '15m', '30m', '1h', '4h']:
                # Convert bars to approximate trading days
                bars_per_day = {'1m': 390, '5m': 78, '15m': 26, '30m': 13, '1h': 7, '4h': 2}
                bpd = bars_per_day.get(interval, 7)
                test_days = test_bars / bpd
            else:
                test_days = test_bars

            years = test_days / 252
            if years > 0:
                annualized = ((final_value / initial_cash) ** (1 / years) - 1) * 100
            else:
                annualized = total_return_pct

            # Get actual date range
            test_df = df.iloc[test_start_idx:]
            start_date_str = test_df.index[0].strftime('%Y-%m-%d')
            end_date_str = test_df.index[-1].strftime('%Y-%m-%d')

            # Get captured buy/sell signals from strategy
            buy_signals = [(sig['date'], sig['price']) for sig in strat.buy_signals]
            sell_signals = [(sig['date'], sig['price']) for sig in strat.sell_signals]

            # Get ML prediction stats and diagnostics
            ml_stats = {}
            ml_diagnostics = {}
            try:
                if hasattr(strat, 'get_prediction_stats'):
                    ml_stats = strat.get_prediction_stats()
                if hasattr(strat, 'get_diagnostics'):
                    ml_diagnostics = strat.get_diagnostics()
            except Exception as ml_err:
                print(f"   ⚠️ Could not get ML stats: {ml_err}")

            # Calculate time in market from buy/sell signals (post-backtest, read-only)
            test_bars_count = len(test_values)
            time_in_market_pct = 0.0
            if test_bars_count > 0 and strat.buy_signals and strat.sell_signals:
                bars_in_position = 0
                buy_bars = sorted([s['bar'] for s in strat.buy_signals])
                sell_bars = sorted([s['bar'] for s in strat.sell_signals])

                for buy_bar in buy_bars:
                    matching_sells = [s for s in sell_bars if s > buy_bar]
                    if matching_sells:
                        sell_bar = matching_sells[0]
                        start = max(buy_bar, test_start_idx)
                        end = sell_bar
                        if end > start:
                            bars_in_position += (end - start)

                if test_bars_count > 0:
                    time_in_market_pct = (bars_in_position / test_bars_count) * 100

            # Count tradeable quarters from fundamental data (only within test period)
            tradeable_quarters = 0
            total_quarters = 0
            earnings_data = []  # List of (date, score) tuples
            test_period_start = test_df.index[0]
            test_period_end = test_df.index[-1]

            if (hasattr(strat, 'fundamental_provider') and
                strat.fundamental_provider is not None and
                strategy_params.get('use_fundamental_filter', False)):
                fp = strat.fundamental_provider
                min_quality = strategy_params.get('min_quality_score', 0)
                min_momentum = strategy_params.get('min_momentum_score', 0)

                if hasattr(fp, '_quarter_report_map') and fp._quarter_report_map:
                    for quarter_end, report_date in fp._quarter_report_map.items():
                        # Convert report_date to timestamp for comparison
                        report_ts = pd.Timestamp(report_date)

                        # Only count quarters within the test period
                        if report_ts >= test_period_start and report_ts <= test_period_end:
                            total_quarters += 1
                            try:
                                quality = fp.get_quality_score(as_of_date=report_date)
                                momentum = fp.get_growth_momentum_score(as_of_date=report_date)
                                # Calculate composite score (average of quality and momentum)
                                composite = 0
                                if quality is not None and momentum is not None:
                                    composite = (quality + momentum) / 2
                                elif quality is not None:
                                    composite = quality
                                elif momentum is not None:
                                    composite = momentum

                                earnings_data.append((report_date, composite))

                                quality_ok = min_quality == 0 or (quality is not None and quality >= min_quality)
                                momentum_ok = min_momentum == 0 or (momentum is not None and momentum >= min_momentum)
                                if quality_ok and momentum_ok:
                                    tradeable_quarters += 1
                            except:
                                earnings_data.append((report_date, 0))

            # Fair value lines on the backtest chart
            fair_value_history = []
            hist_pe_fair_value = []
            try:
                if (hasattr(strat, 'fundamental_provider') and
                        strat.fundamental_provider is not None and
                        self.params.get('use_fundamental_filter', False)):
                    fp = strat.fundamental_provider
                    fair_value_history = fp.get_fair_value_history(
                        start_date=test_period_start, end_date=test_period_end)
                    hist_pe_fair_value = fp.get_historical_pe_fair_value_history(
                        start_date=test_period_start, end_date=test_period_end,
                        price_df=df)
                    print(f"   📊 Fair value (yellow): {len(fair_value_history)} qtrs, "
                          f"hist PE (purple): {len(hist_pe_fair_value)} qtrs")
            except Exception as e:
                import traceback
                print(f"   ⚠️  Fair value history error: {e}")
                traceback.print_exc()

            return {
                'start_date': start_date_str,
                'end_date': end_date_str,
                'timeframe': self.current_timeframe,
                'interval': interval,
                'total_bars': total_bars,
                'test_bars': len(test_values),
                'total_return_pct': total_return_pct,
                'annualized_return': annualized,
                'spy_return': spy_return,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': max_dd,
                'sharpe_ratio': sharpe,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'final_value': final_value,
                # Chart data
                'chart_df': df,
                'test_start_idx': test_start_idx,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                # ML stats
                'ml_stats': ml_stats,
                'ml_diagnostics': ml_diagnostics,
                # Exposure stats
                'time_in_market_pct': time_in_market_pct,
                'tradeable_quarters': tradeable_quarters,
                'total_quarters': total_quarters,
                'earnings_data': earnings_data,
                'fair_value_history': fair_value_history,
                'hist_pe_fair_value': hist_pe_fair_value,
            }

        except Exception as e:
            print(f"   ❌ Backtest execution error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def handle_holdings_query(self):
        """Handle 'HOLDING' or 'HOLDINGS' query - shows all current positions with P&L"""
        positions = self.position_manager.list_all()

        if not positions:
            self.notifier.send_notification(
                "📭 No Holdings",
                "You are not currently holding any positions"
            )
            print("✓ Sent holdings query response (empty)")
            return

        # Build holdings message
        title = f"📊 Holdings ({len(positions)})"
        message_lines = []

        total_pnl = 0
        count = 0

        for symbol, pos in positions.items():
            entry_price = pos['entry_price']
            entry_date = pos['entry_date'][:10]

            # Get current market price (not timeframe dependent)
            try:
                ticker = yf.Ticker(symbol)
                current_price = getattr(ticker.fast_info, 'last_price', None)
                if not current_price:
                    current_price = ticker.info.get('regularMarketPrice')
                if current_price is None:
                    raise ValueError("No price available")
                current_price = float(current_price)
                pnl = float(((current_price / entry_price) - 1) * 100)
                total_pnl += pnl
                count += 1

                pending_exit = " ⚠️ SELL" if pos.get('pending_exit') else ""
                message_lines.append(
                    f"{symbol}: {pnl:+.2f}%{pending_exit}\n"
                    f"  Entry: ${entry_price:.2f} ({entry_date})\n"
                    f"  Now: ${current_price:.2f}"
                )
            except Exception:
                message_lines.append(
                    f"{symbol}: N/A\n"
                    f"  Entry: ${entry_price:.2f} ({entry_date})\n"
                    f"  (Unable to fetch price)"
                )

        message = "\n\n".join(message_lines)

        # Add summary
        if count > 0:
            avg_pnl = total_pnl / count
            message += f"\n\n{'─'*20}\nAvg P&L: {avg_pnl:+.2f}%"

        self.notifier.send_notification(title, message)
        print(f"✓ Sent holdings query response ({len(positions)} positions)")

    def handle_timeframe_command(self, reply):
        """
        Handle 'TIMEFRAME SET <TF>' command - changes the operating timeframe

        Args:
            reply: The reply string like "TIMEFRAME SET 15M" or "TIMEFRAME SET 1H"
        """
        parts = reply.split()

        if len(parts) < 3 or parts[1] != "SET":
            self.notifier.send_notification(
                "⚠️ Invalid Format",
                "Usage: TIMEFRAME SET <TF>\n\n"
                "Valid timeframes:\n"
                "  1M  - 1 Minute\n"
                "  5M  - 5 Minutes\n"
                "  15M - 15 Minutes\n"
                "  30M - 30 Minutes\n"
                "  1H  - 1 Hour\n"
                "  4H  - 4 Hours\n"
                "  1D  - 1 Day\n\n"
                "Example: TIMEFRAME SET 15M"
            )
            return

        new_tf = parts[2].upper()

        if new_tf not in self.TIMEFRAMES:
            valid_tfs = ", ".join(self.TIMEFRAMES.keys())
            self.notifier.send_notification(
                "⚠️ Invalid Timeframe",
                f"Unknown timeframe: {new_tf}\n\n"
                f"Valid options: {valid_tfs}"
            )
            return

        old_tf = self.current_timeframe
        self.current_timeframe = new_tf
        tf_config = self.TIMEFRAMES[new_tf]

        # Warn about limitations of shorter timeframes
        warnings = []
        if new_tf in ['1M', '5M']:
            warnings.append("⚠️ Very short timeframes may have insufficient data for ML model warmup")
        if new_tf in ['1M', '5M', '15M', '30M']:
            warnings.append("⚠️ Intraday data limited to ~60 days history")

        warning_text = "\n".join(warnings) if warnings else ""

        message = (
            f"Timeframe changed:\n"
            f"  {old_tf} → {new_tf}\n\n"
            f"Now using: {tf_config['description']} bars\n"
            f"Data period: {tf_config['period']}"
        )

        if warning_text:
            message += f"\n\n{warning_text}"

        self.notifier.send_notification(f"⏱️ Timeframe: {new_tf}", message)
        print(f"✓ Timeframe changed from {old_tf} to {new_tf}")

    def handle_timeframe_query(self):
        """Handle 'TIMEFRAME' query - shows current timeframe and available options"""
        tf_config = self.TIMEFRAMES[self.current_timeframe]

        # Build list of available timeframes
        tf_list = []
        for tf, config in self.TIMEFRAMES.items():
            marker = "→ " if tf == self.current_timeframe else "  "
            tf_list.append(f"{marker}{tf}: {config['description']}")

        message = (
            f"Current: {self.current_timeframe} ({tf_config['description']})\n"
            f"Data period: {tf_config['period']}\n\n"
            f"Available timeframes:\n" +
            "\n".join(tf_list) +
            f"\n\nTo change: TIMEFRAME SET <TF>"
        )

        self.notifier.send_notification("⏱️ Timeframe Settings", message)
        print(f"✓ Sent timeframe query response (current: {self.current_timeframe})")

    def _compute_financial_score(self, info):
        """
        Compute a financial score and fair value estimate from a yfinance info dict.
        Shared between handle_analyze_query (full report) and _get_quick_financials
        (compact summary for the BEST command).

        Returns dict with:
            score (int 0-9), assessment (str, no emojis), positives (list),
            negatives (list), avg_fair_value (float|None), upside (float|None),
            fair_pe (float|None), fair_value_details (list[str]),
            target_mean (float|None), recommendation (str), current_price (float)
        """
        import math

        current_price   = float(info.get('currentPrice') or info.get('regularMarketPrice') or 0)
        sector          = info.get('sector', '')
        pe_trailing     = info.get('trailingPE')
        peg_ratio       = info.get('pegRatio')
        eps             = info.get('trailingEps')
        book_value      = info.get('bookValue')
        market_cap      = info.get('marketCap')
        free_cash_flow  = info.get('freeCashflow')
        profit_margin   = info.get('profitMargins')
        roe             = info.get('returnOnEquity')
        revenue_growth  = info.get('revenueGrowth')
        earnings_growth = info.get('earningsGrowth')
        current_ratio   = info.get('currentRatio')
        debt_to_equity  = info.get('debtToEquity')
        target_mean     = info.get('targetMeanPrice')
        recommendation  = info.get('recommendationKey', '')

        if not peg_ratio and pe_trailing and earnings_growth and earnings_growth > 0:
            peg_ratio = pe_trailing / (earnings_growth * 100)

        # === FAIR VALUE ESTIMATES ===
        fair_values = []
        fair_value_details = []

        if eps and eps > 0 and book_value and book_value > 0:
            graham = math.sqrt(22.5 * eps * book_value)
            fair_values.append(graham)
            fair_value_details.append(f"Graham: ${graham:.2f}")

        if free_cash_flow and market_cap and free_cash_flow > 0 and current_price > 0:
            fcf_yield = free_cash_flow / market_cap
            if fcf_yield > 0.02:
                dcf_value = current_price * (fcf_yield / 0.08)
                fair_values.append(dcf_value)
                fair_value_details.append(f"FCF-based: ${dcf_value:.2f}")

        if pe_trailing and pe_trailing > 0 and eps and eps > 0:
            sector_pe = 18 if sector in ['Technology', 'Healthcare'] else 15
            pe_fair_value = eps * sector_pe
            fair_values.append(pe_fair_value)
            fair_value_details.append(f"PE-based: ${pe_fair_value:.2f}")

        if target_mean:
            fair_values.append(target_mean)
            fair_value_details.append(f"Analyst avg: ${target_mean:.2f}")

        avg_fair_value = sum(fair_values) / len(fair_values) if fair_values else None
        upside = ((avg_fair_value / current_price) - 1) * 100 if avg_fair_value and current_price > 0 else None

        fair_pe = None
        if peg_ratio and peg_ratio > 0 and earnings_growth:
            fair_pe = abs(earnings_growth) * 100

        # === SCORE (0-9) ===
        score = 0
        positives = []
        negatives = []

        if pe_trailing:
            if pe_trailing < 20:
                score += 1
                positives.append("Reasonably priced")
            elif pe_trailing > 40:
                negatives.append("Expensive valuation")

        if peg_ratio:
            if peg_ratio < 1.5:
                score += 1
                positives.append("Good value for growth")
            elif peg_ratio > 2.5:
                negatives.append("Overpriced for growth")

        if revenue_growth and revenue_growth > 0.1:
            score += 1
            positives.append("Growing sales")
        elif revenue_growth and revenue_growth < 0:
            negatives.append("Shrinking revenue")

        if earnings_growth and earnings_growth > 0.1:
            score += 1
            positives.append("Growing profits")
        elif earnings_growth and earnings_growth < 0:
            negatives.append("Declining earnings")

        if roe and roe > 0.15:
            score += 1
            positives.append("Profitable business")
        if profit_margin and profit_margin > 0.1:
            score += 1
            positives.append("Good margins")
        elif profit_margin and profit_margin < 0:
            negatives.append("Losing money")

        if current_ratio and current_ratio > 1.5:
            score += 1
            positives.append("Financially stable")
        elif current_ratio and current_ratio < 1:
            negatives.append("Cash flow concerns")

        if debt_to_equity and debt_to_equity < 100:
            score += 1
            positives.append("Low debt")
        elif debt_to_equity and debt_to_equity > 200:
            negatives.append("High debt load")

        if upside and upside > 15:
            score += 1
            positives.append("Looks undervalued")
        elif upside and upside < -20:
            negatives.append("Looks overvalued")

        if score >= 7:
            assessment = "Strong Buy"
        elif score >= 5:
            assessment = "Buy"
        elif score >= 3:
            assessment = "Hold"
        elif score >= 1:
            assessment = "Caution"
        else:
            assessment = "Avoid"

        return {
            'score':              score,
            'assessment':         assessment,
            'positives':          positives,
            'negatives':          negatives,
            'avg_fair_value':     avg_fair_value,
            'upside':             upside,
            'fair_pe':            fair_pe,
            'fair_value_details': fair_value_details,
            'target_mean':        target_mean,
            'recommendation':     recommendation,
            'current_price':      current_price,
        }

    def _get_quick_financials(self, symbol):
        """
        Fetch and score financials for a symbol.  Lightweight wrapper around
        _compute_financial_score used by the BEST command hot-list display.
        Returns the _compute_financial_score dict, or None on failure.
        """
        try:
            info = yf.Ticker(symbol).info
            if not info or not (info.get('currentPrice') or info.get('regularMarketPrice')):
                return None
            return self._compute_financial_score(info)
        except Exception:
            return None

    def handle_analyze_query(self, reply):
        """
        Handle 'ANALYZE SYMBOL' query - performs deep fundamental analysis of a stock

        Args:
            reply: The reply string like "ANALYZE NVDA"
        """
        parts = reply.split()

        if len(parts) < 2:
            self.notifier.send_notification(
                "⚠️ Invalid Format",
                "Usage: ANALYZE <SYMBOL>\nExample: ANALYZE NVDA"
            )
            return

        symbol = self._resolve_symbol(parts[1])

        print(f"📊 Running deep analysis for {symbol}...")

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or info.get('regularMarketPrice') is None:
                self.notifier.send_notification(
                    f"❌ {symbol}",
                    "Unable to fetch data for this symbol"
                )
                return

            # === BASIC INFO ===
            company_name = info.get('shortName', symbol)
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            current_price = info.get('regularMarketPrice', 0)
            currency = info.get('currency', 'USD')

            # === VALUATION METRICS ===
            pe_trailing = info.get('trailingPE')
            pe_forward = info.get('forwardPE')
            ps_ratio = info.get('priceToSalesTrailing12Months')
            pb_ratio = info.get('priceToBook')
            ev_ebitda = info.get('enterpriseToEbitda')

            # === PROFITABILITY ===
            profit_margin = info.get('profitMargins')
            operating_margin = info.get('operatingMargins')
            roe = info.get('returnOnEquity')
            roa = info.get('returnOnAssets')

            # === GROWTH ===
            revenue_growth = info.get('revenueGrowth')
            earnings_growth = info.get('earningsGrowth')
            earnings_quarterly_growth = info.get('earningsQuarterlyGrowth')

            # === PEG RATIO (calculate if not available) ===
            peg_ratio = info.get('pegRatio')
            if not peg_ratio and pe_trailing and earnings_growth and earnings_growth > 0:
                peg_ratio = pe_trailing / (earnings_growth * 100)

            # === FINANCIAL HEALTH ===
            current_ratio = info.get('currentRatio')
            debt_to_equity = info.get('debtToEquity')
            free_cash_flow = info.get('freeCashflow')
            operating_cash_flow = info.get('operatingCashflow')

            # === DIVIDENDS ===
            dividend_yield = info.get('dividendYield')
            dividend_rate = info.get('dividendRate')  # Annual dividend per share
            payout_ratio = info.get('payoutRatio')

            # === PRICE LEVELS ===
            fifty_two_week_high = info.get('fiftyTwoWeekHigh', 0)
            fifty_two_week_low = info.get('fiftyTwoWeekLow', 0)
            fifty_day_avg = info.get('fiftyDayAverage', 0)
            two_hundred_day_avg = info.get('twoHundredDayAverage', 0)

            # === ANALYST DATA ===
            target_high = info.get('targetHighPrice')
            target_low = info.get('targetLowPrice')
            target_mean = info.get('targetMeanPrice')
            recommendation = info.get('recommendationKey', 'N/A')
            num_analysts = info.get('numberOfAnalystOpinions', 0)

            # === FAIR VALUE & SCORE — shared helper ===
            _fs                = self._compute_financial_score(info)
            avg_fair_value     = _fs['avg_fair_value']
            upside             = _fs['upside']
            fair_pe            = _fs['fair_pe']
            fair_value_details = _fs['fair_value_details']

            # === BUILD MESSAGE ===
            lines = [
                f"{company_name}",
                f"{sector} › {industry}",
                f"",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"💰 PRICE",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"   Current        ${current_price:.2f}",
                f"   Year High      ${fifty_two_week_high:.2f}",
                f"   Year Low       ${fifty_two_week_low:.2f}",
            ]

            # Position in 52-week range
            if fifty_two_week_high > fifty_two_week_low:
                range_position = ((current_price - fifty_two_week_low) /
                                  (fifty_two_week_high - fifty_two_week_low)) * 100
                if range_position > 80:
                    range_desc = "Near high"
                elif range_position > 60:
                    range_desc = "Upper"
                elif range_position > 40:
                    range_desc = "Middle"
                elif range_position > 20:
                    range_desc = "Lower"
                else:
                    range_desc = "Near low"
                lines.append(f"   In Range       {range_desc} ({range_position:.0f}%)")

            # Trend info
            if fifty_day_avg and two_hundred_day_avg:
                if current_price > fifty_day_avg > two_hundred_day_avg:
                    trend = "📈 Uptrend"
                elif current_price < fifty_day_avg < two_hundred_day_avg:
                    trend = "📉 Downtrend"
                else:
                    trend = "➡️ Sideways"
                lines.append(f"   Trend          {trend}")

            lines.extend([
                f"",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"📈 VALUATION",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ])

            if pe_trailing:
                if pe_trailing < 15:
                    pe_desc = "Cheap"
                elif pe_trailing < 25:
                    pe_desc = "Fair"
                elif pe_trailing < 40:
                    pe_desc = "Pricey"
                else:
                    pe_desc = "Expensive"
                lines.append(f"   P/E Ratio      {pe_trailing:.1f}x  ({pe_desc})")
            if pe_forward:
                lines.append(f"   Forward P/E    {pe_forward:.1f}x")
            if fair_pe:
                lines.append(f"   Fair P/E       ~{fair_pe:.0f}x")
            if peg_ratio:
                if peg_ratio < 1:
                    peg_desc = "Undervalued"
                elif peg_ratio < 2:
                    peg_desc = "Fair"
                else:
                    peg_desc = "Overvalued"
                lines.append(f"   PEG Ratio      {peg_ratio:.2f}  ({peg_desc})")
            if pb_ratio:
                if pb_ratio < 1:
                    pb_desc = "Below book"
                elif pb_ratio < 3:
                    pb_desc = "Reasonable"
                else:
                    pb_desc = "Premium"
                lines.append(f"   Price/Book     {pb_ratio:.1f}x  ({pb_desc})")

            lines.extend([
                f"",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"💵 PROFITABILITY",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ])
            if profit_margin:
                if profit_margin > 0.20:
                    margin_desc = "Excellent"
                elif profit_margin > 0.10:
                    margin_desc = "Good"
                elif profit_margin > 0:
                    margin_desc = "Low"
                else:
                    margin_desc = "Negative"
                lines.append(f"   Profit Margin  {profit_margin*100:.1f}%  ({margin_desc})")
            if roe:
                if roe > 0.20:
                    roe_desc = "Excellent"
                elif roe > 0.10:
                    roe_desc = "Good"
                else:
                    roe_desc = "Poor"
                lines.append(f"   Return/Equity  {roe*100:.1f}%  ({roe_desc})")

            lines.extend([
                f"",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"🚀 GROWTH",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ])
            if revenue_growth:
                if revenue_growth > 0.25:
                    rev_desc = "Fast"
                elif revenue_growth > 0.10:
                    rev_desc = "Solid"
                elif revenue_growth > 0:
                    rev_desc = "Slow"
                else:
                    rev_desc = "Shrinking"
                lines.append(f"   Revenue        {revenue_growth*100:+.1f}%  ({rev_desc})")
            if earnings_growth:
                if earnings_growth > 0.25:
                    earn_desc = "Strong"
                elif earnings_growth > 0:
                    earn_desc = "Growing"
                else:
                    earn_desc = "Declining"
                lines.append(f"   Earnings       {earnings_growth*100:+.1f}%  ({earn_desc})")

            lines.extend([
                f"",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"🏦 FINANCIAL HEALTH",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ])
            if current_ratio:
                if current_ratio > 2:
                    cr_desc = "Strong"
                elif current_ratio > 1:
                    cr_desc = "OK"
                else:
                    cr_desc = "Weak"
                lines.append(f"   Liquidity      {current_ratio:.1f}x  ({cr_desc})")
            if debt_to_equity:
                if debt_to_equity < 50:
                    de_desc = "Low"
                elif debt_to_equity < 100:
                    de_desc = "Moderate"
                else:
                    de_desc = "High"
                lines.append(f"   Debt/Equity    {debt_to_equity:.0f}%  ({de_desc})")
            if free_cash_flow:
                fcf_b = free_cash_flow / 1e9
                if free_cash_flow > 0:
                    lines.append(f"   Free Cash      ${fcf_b:.1f}B/yr 👍")
                else:
                    lines.append(f"   Cash Burn      ${abs(fcf_b):.1f}B/yr 👎")

            if dividend_yield:
                lines.extend([
                    f"",
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                    f"💸 DIVIDENDS",
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                    f"   Yield          {dividend_yield*100:.2f}%",
                ])
                if dividend_rate:
                    lines.append(f"   Per Share      ${dividend_rate:.2f}/year")
                    # Calculate quarterly payment
                    quarterly = dividend_rate / 4
                    lines.append(f"                  (${quarterly:.2f}/quarter)")
                if payout_ratio:
                    if payout_ratio < 0.5:
                        payout_desc = "Safe"
                    elif payout_ratio < 0.8:
                        payout_desc = "Moderate"
                    else:
                        payout_desc = "Risky"
                    lines.append(f"   Payout Ratio   {payout_ratio*100:.0f}%  ({payout_desc})")

            lines.extend([
                f"",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"🎯 FAIR VALUE ESTIMATE",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ])
            if fair_value_details:
                for detail in fair_value_details:
                    # Parse and reformat
                    lines.append(f"   {detail}")
            if avg_fair_value:
                lines.append(f"   ─────────────────────")
                lines.append(f"   Average        ${avg_fair_value:.2f}")
                lines.append(f"   Current        ${current_price:.2f}")
                if upside:
                    if upside > 20:
                        verdict = f"UNDERVALUED {upside:+.0f}% 🟢"
                    elif upside > 5:
                        verdict = f"Slightly under {upside:+.0f}% 🟢"
                    elif upside > -5:
                        verdict = f"Fairly priced 🟡"
                    elif upside > -20:
                        verdict = f"Slightly over {upside:+.0f}% 🟡"
                    else:
                        verdict = f"OVERVALUED {upside:+.0f}% 🔴"
                    lines.append(f"   Verdict        {verdict}")

            lines.extend([
                f"",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"👔 ANALYST OPINIONS ({num_analysts})",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ])
            if recommendation:
                rec_friendly = {
                    'strongBuy': 'Strong Buy 🟢🟢',
                    'strong_buy': 'Strong Buy 🟢🟢',
                    'buy': 'Buy 🟢',
                    'hold': 'Hold 🟡',
                    'sell': 'Sell 🔴',
                    'strongSell': 'Strong Sell 🔴🔴',
                    'strong_sell': 'Strong Sell 🔴🔴',
                }.get(recommendation, recommendation.replace('_', ' ').title())
                lines.append(f"   Rating         {rec_friendly}")
            if target_mean:
                target_upside = ((target_mean / current_price) - 1) * 100
                lines.append(f"   Target         ${target_mean:.2f}  ({target_upside:+.0f}%)")
            if target_low and target_high:
                lines.append(f"   Range          ${target_low:.2f} - ${target_high:.2f}")

            # === OVERALL ASSESSMENT ===
            lines.extend([
                f"",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"📋 BOTTOM LINE",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ])

            # Score from shared helper (computed alongside fair values above)
            score     = _fs['score']
            positives = _fs['positives']
            negatives = _fs['negatives']
            _emoji_map = {
                'Strong Buy': 'Strong Buy 🟢🟢',
                'Buy':        'Buy 🟢',
                'Hold':       'Hold 🟡',
                'Caution':    'Caution 🔴',
                'Avoid':      'Avoid 🔴🔴',
            }
            assessment = _emoji_map.get(_fs['assessment'], _fs['assessment'])

            lines.append(f"")
            lines.append(f"   Rating         {assessment}")
            lines.append(f"   Score          {score}/9")

            if positives:
                lines.append(f"")
                lines.append(f"   ✅ Strengths")
                for p in positives[:4]:
                    lines.append(f"      • {p}")

            if negatives:
                lines.append(f"")
                lines.append(f"   ⚠️ Concerns")
                for n in negatives[:3]:
                    lines.append(f"      • {n}")

            message = "\n".join(lines)

            self.notifier.send_notification(f"📊 {symbol} Analysis", message)
            print(f"✓ Sent analysis for {symbol}")

        except Exception as e:
            print(f"❌ Analysis error for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            self.notifier.send_notification(
                f"❌ Analysis Error",
                f"Error analyzing {symbol}:\n{str(e)[:100]}"
            )

    def handle_compare_query(self, reply):
        """
        Handle 'COMPARE SYMBOL1 SYMBOL2 ...' query - compares multiple stocks

        Args:
            reply: The reply string like "COMPARE AAPL MSFT GOOGL"
        """
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        from io import BytesIO

        parts = reply.split()

        if len(parts) < 3:
            self.notifier.send_notification(
                "⚠️ Invalid Format",
                "Usage: COMPARE <SYMBOL1> <SYMBOL2> [SYMBOL3] ...\n"
                "Example: COMPARE AAPL MSFT GOOGL"
            )
            return

        symbols = [self._resolve_symbol(s) for s in parts[1:]]

        if len(symbols) > 8:
            self.notifier.send_notification(
                "⚠️ Too Many Stocks",
                "Please compare 8 or fewer stocks at a time."
            )
            return

        print(f"📊 Comparing {len(symbols)} stocks: {', '.join(symbols)}...")

        # Fetch data for all stocks
        stock_data = {}
        failed_symbols = []

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info

                if not info or info.get('regularMarketPrice') is None:
                    failed_symbols.append(symbol)
                    continue

                # Extract key metrics for comparison
                pe_ratio = info.get('trailingPE')
                earnings_growth = info.get('earningsGrowth')

                # Get PEG ratio - calculate if not available
                peg_ratio = info.get('pegRatio')
                if not peg_ratio and pe_ratio and earnings_growth and earnings_growth > 0:
                    # PEG = P/E / (Earnings Growth * 100)
                    peg_ratio = pe_ratio / (earnings_growth * 100)

                # Fetch 1 year of historical data for sparkline chart
                hist = ticker.history(period='1y')
                hist_prices = hist['Close'].values if len(hist) > 0 else None

                stock_data[symbol] = {
                    'name': info.get('shortName', symbol),
                    'price': info.get('regularMarketPrice', 0),
                    'pe_ratio': pe_ratio,
                    'forward_pe': info.get('forwardPE'),
                    'peg_ratio': peg_ratio,
                    'pb_ratio': info.get('priceToBook'),
                    'profit_margin': info.get('profitMargins'),
                    'roe': info.get('returnOnEquity'),
                    'revenue_growth': info.get('revenueGrowth'),
                    'earnings_growth': earnings_growth,
                    'current_ratio': info.get('currentRatio'),
                    'debt_to_equity': info.get('debtToEquity'),
                    'dividend_yield': info.get('dividendYield'),
                    'target_mean': info.get('targetMeanPrice'),
                    'recommendation': info.get('recommendationKey'),
                    'market_cap': info.get('marketCap'),
                    'history': hist_prices,
                }

                print(f"   ✓ Fetched {symbol}")

            except Exception as e:
                print(f"   ❌ Failed to fetch {symbol}: {e}")
                failed_symbols.append(symbol)

        if len(stock_data) < 2:
            self.notifier.send_notification(
                "❌ Comparison Failed",
                f"Need at least 2 valid stocks to compare.\n"
                f"Failed: {', '.join(failed_symbols)}"
            )
            return

        # Calculate upside for each stock
        upside_values = {}
        for s, d in stock_data.items():
            if d['target_mean'] and d['price']:
                upside_values[s] = ((d['target_mean'] / d['price']) - 1) * 100

        # Calculate fair value for each stock
        fair_values = {}
        value_gaps = {}
        for symbol, data in stock_data.items():
            estimates = []
            price = data['price']

            # Method 1: PE-based (using sector-appropriate PE of 18 for tech, 15 otherwise)
            if data['pe_ratio'] and data['pe_ratio'] > 0:
                eps = price / data['pe_ratio']
                fair_pe = 18  # Could vary by sector but keeping it simple
                pe_fair = eps * fair_pe
                estimates.append(pe_fair)

            # Method 2: Analyst target
            if data['target_mean']:
                estimates.append(data['target_mean'])

            # Method 3: PEG-based (fair value at PEG=1)
            if data['peg_ratio'] and data['peg_ratio'] > 0 and data['earnings_growth'] and data['earnings_growth'] > 0:
                # If PEG > 1, stock is overvalued; calculate what price would give PEG=1
                fair_peg_price = price / data['peg_ratio']
                estimates.append(fair_peg_price)

            if estimates and price:
                avg_fair = sum(estimates) / len(estimates)
                fair_values[symbol] = avg_fair
                value_gaps[symbol] = ((avg_fair / price) - 1) * 100  # Positive = undervalued
            else:
                fair_values[symbol] = None
                value_gaps[symbol] = None

        # Find best values FIRST (used for both scoring and highlighting)
        pe_values = {s: d['pe_ratio'] for s, d in stock_data.items() if d['pe_ratio'] and d['pe_ratio'] > 0}
        peg_values = {s: d['peg_ratio'] for s, d in stock_data.items() if d['peg_ratio'] and d['peg_ratio'] > 0}
        margin_values = {s: d['profit_margin'] for s, d in stock_data.items() if d['profit_margin']}
        roe_values = {s: d['roe'] for s, d in stock_data.items() if d['roe']}
        rev_values = {s: d['revenue_growth'] for s, d in stock_data.items() if d['revenue_growth']}
        earn_values = {s: d['earnings_growth'] for s, d in stock_data.items() if d['earnings_growth']}
        de_values = {s: d['debt_to_equity'] for s, d in stock_data.items() if d['debt_to_equity'] is not None}
        div_values = {s: d['dividend_yield'] for s, d in stock_data.items() if d['dividend_yield']}
        gap_values = {s: v for s, v in value_gaps.items() if v is not None}

        best_pe = min(pe_values.values()) if pe_values else None
        best_peg = min(peg_values.values()) if peg_values else None
        best_margin = max(margin_values.values()) if margin_values else None
        best_roe = max(roe_values.values()) if roe_values else None
        best_rev = max(rev_values.values()) if rev_values else None
        best_earn = max(earn_values.values()) if earn_values else None
        best_de = min(de_values.values()) if de_values else None
        best_div = max(div_values.values()) if div_values else None
        best_upside = max(upside_values.values()) if upside_values else None
        best_gap = max(gap_values.values()) if gap_values else None  # Most undervalued

        # Score each stock - now includes bonus for being BEST in each category
        # Base scoring uses thresholds, but being BEST in category adds bonus points
        scores = {}
        for symbol, data in stock_data.items():
            score = 0

            # P/E Ratio (lower is better)
            if data['pe_ratio'] and data['pe_ratio'] > 0:
                if data['pe_ratio'] < 15:
                    score += 2
                elif data['pe_ratio'] < 25:
                    score += 1
                # BONUS: Best P/E among compared stocks
                if best_pe and data['pe_ratio'] == best_pe:
                    score += 1

            # PEG Ratio (lower is better)
            if data['peg_ratio'] and data['peg_ratio'] > 0:
                if data['peg_ratio'] < 1:
                    score += 2
                elif data['peg_ratio'] < 1.5:
                    score += 1
                # BONUS: Best PEG among compared stocks
                if best_peg and data['peg_ratio'] == best_peg:
                    score += 1

            # Profit Margin (higher is better)
            if data['profit_margin'] and data['profit_margin'] > 0.15:
                score += 2
            elif data['profit_margin'] and data['profit_margin'] > 0.08:
                score += 1
            # BONUS: Best margin among compared stocks
            if data['profit_margin'] and best_margin and data['profit_margin'] == best_margin:
                score += 1

            # ROE (higher is better)
            if data['roe'] and data['roe'] > 0.18:
                score += 2
            elif data['roe'] and data['roe'] > 0.10:
                score += 1
            # BONUS: Best ROE among compared stocks
            if data['roe'] and best_roe and data['roe'] == best_roe:
                score += 1

            # Revenue Growth (higher is better)
            if data['revenue_growth'] and data['revenue_growth'] > 0.15:
                score += 2
            elif data['revenue_growth'] and data['revenue_growth'] > 0.05:
                score += 1
            # BONUS: Best revenue growth among compared stocks
            if data['revenue_growth'] and best_rev and data['revenue_growth'] == best_rev:
                score += 1

            # Earnings Growth (higher is better)
            if data['earnings_growth'] and data['earnings_growth'] > 0.15:
                score += 2
            elif data['earnings_growth'] and data['earnings_growth'] > 0.05:
                score += 1
            # BONUS: Best earnings growth among compared stocks
            if data['earnings_growth'] and best_earn and data['earnings_growth'] == best_earn:
                score += 1

            if data['current_ratio'] and data['current_ratio'] > 1.5:
                score += 1

            # Debt/Equity (lower is better)
            if data['debt_to_equity'] and data['debt_to_equity'] < 80:
                score += 1
            # BONUS: Best (lowest) debt among compared stocks
            if data['debt_to_equity'] is not None and best_de is not None and data['debt_to_equity'] == best_de:
                score += 1

            if data['recommendation'] in ['strongBuy', 'strong_buy', 'buy']:
                score += 1

            # Analyst Upside - BONUS for best upside
            if symbol in upside_values and best_upside and upside_values[symbol] == best_upside:
                score += 1

            # Value gap scoring (undervalued = good)
            if value_gaps.get(symbol) is not None:
                gap = value_gaps[symbol]
                if gap > 15:
                    score += 2  # Significantly undervalued
                elif gap > 5:
                    score += 1  # Slightly undervalued
                # BONUS: Best (most undervalued) value gap among compared stocks
                if best_gap and gap == best_gap:
                    score += 1

            scores[symbol] = score

        # Find winner
        winner = max(scores, key=scores.get)
        winner_score = scores[winner]

        # Calculate max possible score (dynamic based on categories)
        max_score = 16 + 10  # Base 16 + up to 10 bonus points for being best

        # Build table data
        symbols_list = list(stock_data.keys())
        metrics = [
            'Price',
            'Fair Value',
            'Value Gap',
            'P/E Ratio',
            'PEG Ratio',
            'Profit Margin',
            'Return on Equity',
            'Revenue Growth',
            'Earnings Growth',
            'Debt/Equity',
            'Dividend Yield',
            'Analyst Upside',
            'Analyst Rating',
            'SCORE',
        ]

        # Create cell data and colors
        cell_data = []
        cell_colors = []

        # Colors
        header_color = '#2C3E50'
        best_color = '#27AE60'
        winner_color = '#F39C12'
        normal_color = '#FFFFFF'
        alt_color = '#F8F9FA'

        for i, metric in enumerate(metrics):
            row_data = []
            row_colors = []
            bg = alt_color if i % 2 == 0 else normal_color

            for symbol in symbols_list:
                data = stock_data[symbol]
                is_best = False
                val = "N/A"

                if metric == 'Price':
                    val = f"${data['price']:.2f}" if data['price'] else "N/A"
                elif metric == 'Fair Value':
                    if fair_values.get(symbol):
                        val = f"${fair_values[symbol]:.2f}"
                    else:
                        val = "N/A"
                elif metric == 'Value Gap':
                    if value_gaps.get(symbol) is not None:
                        gap = value_gaps[symbol]
                        if gap > 0:
                            val = f"+{gap:.0f}%"  # Undervalued
                        else:
                            val = f"{gap:.0f}%"  # Overvalued
                        is_best = best_gap and value_gaps[symbol] == best_gap
                elif metric == 'P/E Ratio':
                    if data['pe_ratio']:
                        val = f"{data['pe_ratio']:.1f}"
                        is_best = best_pe and data['pe_ratio'] == best_pe
                elif metric == 'PEG Ratio':
                    if data['peg_ratio'] and data['peg_ratio'] > 0:
                        val = f"{data['peg_ratio']:.2f}"
                        is_best = best_peg and data['peg_ratio'] == best_peg
                elif metric == 'Profit Margin':
                    if data['profit_margin']:
                        val = f"{data['profit_margin']*100:.1f}%"
                        is_best = best_margin and data['profit_margin'] == best_margin
                elif metric == 'Return on Equity':
                    if data['roe']:
                        val = f"{data['roe']*100:.1f}%"
                        is_best = best_roe and data['roe'] == best_roe
                elif metric == 'Revenue Growth':
                    if data['revenue_growth']:
                        val = f"{data['revenue_growth']*100:+.1f}%"
                        is_best = best_rev and data['revenue_growth'] == best_rev
                elif metric == 'Earnings Growth':
                    if data['earnings_growth']:
                        val = f"{data['earnings_growth']*100:+.1f}%"
                        is_best = best_earn and data['earnings_growth'] == best_earn
                elif metric == 'Debt/Equity':
                    if data['debt_to_equity'] is not None:
                        val = f"{data['debt_to_equity']:.0f}%"
                        is_best = best_de is not None and data['debt_to_equity'] == best_de
                elif metric == 'Dividend Yield':
                    if data['dividend_yield']:
                        val = f"{data['dividend_yield']*100:.2f}%"
                        is_best = best_div and data['dividend_yield'] == best_div
                    else:
                        val = "—"
                elif metric == 'Analyst Upside':
                    if symbol in upside_values:
                        val = f"{upside_values[symbol]:+.1f}%"
                        is_best = best_upside and upside_values[symbol] == best_upside
                elif metric == 'Analyst Rating':
                    rec = data['recommendation']
                    if rec:
                        val = {
                            'strongBuy': 'Strong Buy',
                            'strong_buy': 'Strong Buy',
                            'buy': 'Buy',
                            'hold': 'Hold',
                            'sell': 'Sell',
                            'strongSell': 'Strong Sell',
                            'strong_sell': 'Strong Sell',
                        }.get(rec, rec.replace('_', ' ').title())
                elif metric == 'SCORE':
                    val = f"{scores[symbol]}"
                    is_best = symbol == winner

                row_data.append(val)
                if metric == 'SCORE' and symbol == winner:
                    row_colors.append(winner_color)
                elif is_best:
                    row_colors.append(best_color)
                else:
                    row_colors.append(bg)

            cell_data.append(row_data)
            cell_colors.append(row_colors)

        # Create figure with GridSpec for table + sparkline charts
        from matplotlib.gridspec import GridSpec
        import numpy as np

        fig_width = 3 + len(symbols_list) * 1.8
        fig_height = 1 + len(metrics) * 0.4 + 3.5  # Extra space for charts and winner

        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = GridSpec(3, 1, height_ratios=[6, 1.5, 1], hspace=0.15)

        # Top section: Table
        ax_table = fig.add_subplot(gs[0])
        ax_table.axis('off')

        # Create table
        table = ax_table.table(
            cellText=cell_data,
            rowLabels=metrics,
            colLabels=symbols_list,
            cellColours=cell_colors,
            rowColours=['#ECF0F1'] * len(metrics),
            colColours=[header_color] * len(symbols_list),
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        # Style header cells
        for j in range(len(symbols_list)):
            cell = table[(0, j)]
            cell.set_text_props(weight='bold', color='white')
            cell.set_fontsize(12)

        # Style row labels
        for i in range(len(metrics)):
            cell = table[(i + 1, -1)]
            cell.set_text_props(weight='bold')
            if metrics[i] == 'SCORE':
                cell.set_facecolor('#34495E')
                cell.set_text_props(weight='bold', color='white')

        # Style SCORE row
        for j in range(len(symbols_list)):
            cell = table[(len(metrics), j)]
            cell.set_text_props(weight='bold')

        # Add title
        plt.suptitle('Stock Comparison', fontsize=16, fontweight='bold', y=0.98)

        # Middle section: Sparkline charts (1 year price history)
        ax_charts = fig.add_subplot(gs[1])
        ax_charts.axis('off')

        # Create mini subplots for each stock's sparkline
        num_stocks = len(symbols_list)
        chart_axes = []

        # Calculate positions for mini charts (evenly spaced across the width)
        chart_width = 0.8 / num_stocks
        chart_left_margin = 0.15  # Account for row labels

        for idx, symbol in enumerate(symbols_list):
            # Position each mini chart
            left = chart_left_margin + idx * (0.85 / num_stocks)
            bottom = 0.35  # Position within the middle section
            width = 0.8 / num_stocks - 0.02
            height = 0.55

            # Create inset axes for each sparkline
            ax_spark = fig.add_axes([left, bottom * 0.28 + 0.22, width, height * 0.12])
            chart_axes.append(ax_spark)

            data = stock_data[symbol]
            hist = data.get('history')

            if hist is not None and len(hist) > 10:
                # Normalize to percentage change from start
                prices = np.array(hist)
                pct_change = ((prices / prices[0]) - 1) * 100

                # Determine color based on overall trend
                if prices[-1] > prices[0]:
                    line_color = '#27AE60'  # Green for up
                    fill_color = '#27AE6030'
                else:
                    line_color = '#E74C3C'  # Red for down
                    fill_color = '#E74C3C30'

                # Plot sparkline
                ax_spark.plot(pct_change, color=line_color, linewidth=1.5)
                ax_spark.fill_between(range(len(pct_change)), pct_change, 0,
                                      color=fill_color, alpha=0.3)
                ax_spark.axhline(y=0, color='#888888', linewidth=0.5, linestyle='-')

                # Add YTD return label
                ytd_return = pct_change[-1]
                ax_spark.text(0.5, -0.25, f"{ytd_return:+.0f}% (1Y)",
                             transform=ax_spark.transAxes,
                             fontsize=8, ha='center', va='top',
                             color=line_color, fontweight='bold')
            else:
                ax_spark.text(0.5, 0.5, 'No data', transform=ax_spark.transAxes,
                             fontsize=8, ha='center', va='center', color='#888888')

            # Clean up axes
            ax_spark.set_xlim(0, len(hist) if hist is not None and len(hist) > 0 else 1)
            ax_spark.set_xticks([])
            ax_spark.set_yticks([])
            for spine in ax_spark.spines.values():
                spine.set_visible(False)

        # Add "1Y Chart" label on the left
        fig.text(0.08, 0.30, '1Y Chart', fontsize=10, fontweight='bold',
                 ha='center', va='center', rotation=0,
                 bbox=dict(boxstyle='round', facecolor='#ECF0F1', edgecolor='none'))

        # Bottom section: Winner announcement
        ax_winner = fig.add_subplot(gs[2])
        ax_winner.axis('off')

        winner_data = stock_data[winner]
        winner_text = f"🏆 WINNER: {winner} ({winner_data['name']})\nScore: {winner_score} points"

        # Why it won
        reasons = []
        if winner_data['pe_ratio'] and winner_data['pe_ratio'] < 20:
            reasons.append("Reasonable valuation")
        if winner_data['profit_margin'] and winner_data['profit_margin'] > 0.15:
            reasons.append("High margins")
        if winner_data['roe'] and winner_data['roe'] > 0.15:
            reasons.append("Strong returns")
        if winner_data['revenue_growth'] and winner_data['revenue_growth'] > 0.10:
            reasons.append("Growing revenue")
        if winner_data['debt_to_equity'] and winner_data['debt_to_equity'] < 80:
            reasons.append("Low debt")
        if winner in upside_values and upside_values[winner] > 15:
            reasons.append("High upside")
        if value_gaps.get(winner) and value_gaps[winner] > 10:
            reasons.append("Undervalued")

        if reasons:
            winner_text += "\nStrengths: " + ", ".join(reasons[:3])

        ax_winner.text(0.5, 0.5, winner_text, transform=ax_winner.transAxes,
                       fontsize=12, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor=winner_color, alpha=0.3))

        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        plt.close(fig)

        # Build text message (for notification body)
        message = f"Comparing: {', '.join(symbols_list)}\n\n"
        message += f"🏆 Winner: {winner} ({winner_data['name']})\n"
        message += f"Score: {winner_score} points"
        if reasons:
            message += f"\nStrengths: {', '.join(reasons[:3])}"

        if failed_symbols:
            message += f"\n\n⚠️ Failed to fetch: {', '.join(failed_symbols)}"

        # Send with image
        self.notifier.send_notification_with_image(
            f"📊 {' vs '.join(symbols_list)}",
            message,
            buf,
            "comparison.png"
        )
        print(f"✓ Sent comparison for {len(stock_data)} stocks. Winner: {winner}")