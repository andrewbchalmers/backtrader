"""
Strategy Generator - Creates Dynamic Trading Strategies
"""

import backtrader as bt
from itertools import product
from .indicator_library import IndicatorFactory, IndicatorSignal, get_indicator_description


class GenericStrategy(bt.Strategy):
    """
    Generic strategy template that can use any combination of entry/exit indicators
    """
    params = (
        ('entry_indicator', ''),
        ('entry_params', ()),
        ('entry_type', 'crossover'),  # crossover, threshold, breakout, dual_crossover
        ('entry_indicator2', ''),  # For dual indicator strategies
        ('entry_params2', ()),  # For dual indicator strategies
        ('entry_filter_indicator', ''),  # Optional filter (e.g., trend filter)
        ('entry_filter_params', ()),
        ('entry_filter_type', ''),  # How to check filter (above, below, rising)
        ('exit_indicator', ''),
        ('exit_params', ()),
        ('exit_type', 'crossover'),  # crossover, stop_loss, take_profit, trailing_stop, dual_crossover
        ('exit_indicator2', ''),  # For dual indicator strategies
        ('exit_params2', ()),  # For dual indicator strategies
        ('exit_filter_indicator', ''),  # Optional filter
        ('exit_filter_params', ()),
        ('exit_filter_type', ''),
        ('use_stop_loss', False),  # Whether to use stop loss in addition to exit indicator
        ('stop_loss_pct', 0.02),  # 2% stop loss
        ('use_take_profit', False),  # Whether to use take profit in addition to exit indicator
        ('take_profit_pct', 0.06),  # 6% take profit
        ('trailing_stop_pct', 0.03),  # 3% trailing stop
        ('printlog', False),
    )

    def __init__(self):
        # Create entry indicator(s)
        self.entry_ind = IndicatorFactory.create_indicator(
            self.data,
            self.params.entry_indicator,
            *self.params.entry_params
        )

        # Create second entry indicator if using dual crossover
        if self.params.entry_type == 'dual_crossover':
            self.entry_ind2 = IndicatorFactory.create_indicator(
                self.data,
                self.params.entry_indicator2,
                *self.params.entry_params2
            )
        else:
            self.entry_ind2 = None

        # Create entry filter indicator if specified
        if self.params.entry_filter_indicator:
            self.entry_filter = IndicatorFactory.create_indicator(
                self.data,
                self.params.entry_filter_indicator,
                *self.params.entry_filter_params
            )
        else:
            self.entry_filter = None

        # Create exit indicator(s)
        self.exit_ind = IndicatorFactory.create_indicator(
            self.data,
            self.params.exit_indicator,
            *self.params.exit_params
        )

        # Create second exit indicator if using dual crossover
        if self.params.exit_type == 'dual_crossover':
            self.exit_ind2 = IndicatorFactory.create_indicator(
                self.data,
                self.params.exit_indicator2,
                *self.params.exit_params2
            )
        else:
            self.exit_ind2 = None

        # Create exit filter indicator if specified
        if self.params.exit_filter_indicator:
            self.exit_filter = IndicatorFactory.create_indicator(
                self.data,
                self.params.exit_filter_indicator,
                *self.params.exit_filter_params
            )
        else:
            self.exit_filter = None

        # Track order and entry price
        self.order = None
        self.entry_price = None
        self.highest_price = None

        # For crossover strategies, we need a reference (typically price)
        self.price = self.data.close

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
                self.highest_price = order.executed.price
                if self.params.printlog:
                    print(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
            elif order.issell():
                if self.params.printlog:
                    print(f'SELL EXECUTED, Price: {order.executed.price:.2f}')
                self.entry_price = None
                self.highest_price = None

        self.order = None

    def next(self):
        # Check if we have an order pending
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # Entry logic
            entry_signal = self._check_entry_signal()
            if entry_signal:
                self.order = self.buy()
        else:
            # Update highest price for trailing stop
            if self.data.close[0] > self.highest_price:
                self.highest_price = self.data.close[0]

            # Exit logic
            exit_signal = self._check_exit_signal()
            if exit_signal:
                self.order = self.sell()

    def _check_entry_signal(self):
        """Check if entry conditions are met"""
        entry_signal = False
        entry_type = self.params.entry_type

        if entry_type == 'dual_crossover':
            # Indicator 1 crosses above Indicator 2
            entry_signal = self.entry_ind[0] > self.entry_ind2[0] and self.entry_ind[-1] <= self.entry_ind2[-1]

        elif entry_type == 'threshold':
            # Indicator crosses above a threshold (for oscillators like RSI)
            # For RSI: buy when RSI crosses above 30
            threshold = 30 if self.params.entry_indicator == 'rsi' else 0
            entry_signal = self.entry_ind[0] > threshold and self.entry_ind[-1] <= threshold

        elif entry_type == 'breakout':
            # Price breaks out of bands or channels
            if self.params.entry_indicator == 'bb':
                # Bollinger Bands strategies:
                # 1. Mean reversion (period < 15): Buy when price IS AT/BELOW lower band (oversold)
                # 2. Momentum (period >= 15): Buy when price IS AT/ABOVE upper band (strong trend)

                bb_period = int(self.params.entry_params[0]) if self.params.entry_params else 20

                if bb_period < 15:
                    # Mean reversion: Buy when price is at or below lower band
                    entry_signal = self.price[0] <= self.entry_ind.bot[0]
                else:
                    # Momentum: Buy when price is at or above upper band
                    entry_signal = self.price[0] >= self.entry_ind.top[0]
            elif self.params.entry_indicator in ['keltner', 'donchian']:
                # Similar logic for other band indicators
                entry_signal = self.price[0] <= self.entry_ind.bot[0]
            else:
                # Generic breakout: price above indicator
                entry_signal = self.price[0] > self.entry_ind[0]

        elif entry_type == 'crossover':
            # Price crosses above indicator
            if self.params.entry_indicator == 'bb':
                # For BB crossover, use the middle band (basis)
                entry_signal = self.price[0] > self.entry_ind.mid[0] and self.price[-1] <= self.entry_ind.mid[-1]
            else:
                # Standard crossover
                entry_signal = self.price[0] > self.entry_ind[0] and self.price[-1] <= self.entry_ind[-1]

        # Check entry filter if specified (e.g., only trade with trend)
        if entry_signal and self.entry_filter:
            filter_pass = self._check_filter(
                self.entry_filter,
                self.params.entry_filter_type,
                self.params.entry_filter_indicator
            )
            entry_signal = entry_signal and filter_pass

        return entry_signal

    def _check_filter(self, filter_ind, filter_type, filter_name):
        """
        Check if filter condition is met

        Filter types:
        - 'above': Price must be above filter indicator
        - 'below': Price must be below filter indicator
        - 'rising': Filter indicator must be rising
        - 'high': Filter value must be above threshold (for oscillators/volatility)
        """
        if filter_type == 'above':
            # Price above filter (e.g., price above EMA 200)
            return self.price[0] > filter_ind[0]

        elif filter_type == 'below':
            # Price below filter
            return self.price[0] < filter_ind[0]

        elif filter_type == 'rising':
            # Filter is rising (e.g., EMA is trending up)
            return filter_ind[0] > filter_ind[-1]

        elif filter_type == 'high':
            # Filter value is high (for ATR, RSI, etc.)
            if filter_name in ['atr', 'natr']:
                # ATR above its own 20-period average = high volatility
                atr_avg = sum([filter_ind[-i] for i in range(20)]) / 20
                return filter_ind[0] > atr_avg
            elif filter_name in ['rsi']:
                # RSI above 50 = bullish
                return filter_ind[0] > 50
            elif filter_name in ['aroon']:
                # Aroon up > 70 = strong uptrend
                return filter_ind[0] > 70
            else:
                return True  # Default to passing

        return True  # Default to passing if unknown filter type

    def _check_exit_signal(self):
        """Check if exit conditions are met"""
        exit_signal = False

        # Check primary exit type
        exit_type = self.params.exit_type

        if exit_type == 'dual_crossover':
            # Indicator 1 crosses below Indicator 2
            exit_signal = self.exit_ind[0] < self.exit_ind2[0] and self.exit_ind[-1] >= self.exit_ind2[-1]

        elif exit_type == 'crossover':
            # Price crosses below indicator
            if self.params.exit_indicator == 'bb':
                # BB exits based on strategy type:
                # - Mean reversion: Exit when price REACHES upper band (take profit on reversion)
                # - Momentum: Exit when price goes BELOW middle band (trend reversal)
                bb_period = int(self.params.exit_params[0]) if self.params.exit_params else 20

                if bb_period < 15:
                    # Mean reversion exit: exit when price is at/above upper band (target reached)
                    exit_signal = self.price[0] >= self.exit_ind.top[0]
                else:
                    # Momentum exit: exit when price falls below middle band (trend broken)
                    exit_signal = self.price[0] < self.exit_ind.mid[0]
            else:
                # Standard crossover exit
                exit_signal = self.price[0] < self.exit_ind[0] and self.price[-1] >= self.exit_ind[-1]
                exit_signal = self.price[0] < self.exit_ind[0] and self.price[-1] >= self.exit_ind[-1]

        elif exit_type == 'crossunder':
            # Price crosses above indicator
            exit_signal = self.price[0] > self.exit_ind[0] and self.price[-1] <= self.exit_ind[-1]

        elif exit_type == 'stop_loss':
            # Stop loss - use ATR if available, otherwise percentage
            if self.entry_price:
                if self.params.exit_indicator in ['atr', 'natr']:
                    # ATR-based stop: stop = entry - (ATR * multiplier)
                    stop_distance = self.exit_ind[0]  # ATR value already multiplied
                    stop_price = self.entry_price - stop_distance
                else:
                    # Percentage-based stop loss
                    stop_price = self.entry_price * (1 - self.params.stop_loss_pct)

                exit_signal = self.data.close[0] < stop_price

        elif exit_type == 'take_profit':
            # Take profit only
            if self.entry_price:
                target_price = self.entry_price * (1 + self.params.take_profit_pct)
                exit_signal = self.data.close[0] > target_price

        elif exit_type == 'trailing_stop':
            # Trailing stop - use ATR if available, otherwise percentage
            if self.highest_price:
                if self.params.exit_indicator in ['atr', 'natr']:
                    # ATR-based trailing stop: stop = highest - (ATR * multiplier)
                    stop_distance = self.exit_ind[0]  # ATR value already multiplied
                    stop_price = self.highest_price - stop_distance
                else:
                    # Percentage-based trailing stop
                    stop_price = self.highest_price * (1 - self.params.trailing_stop_pct)

                exit_signal = self.data.close[0] < stop_price

        # Check additional stop loss (if enabled and not already the primary exit type)
        if self.params.use_stop_loss and exit_type != 'stop_loss':
            if self.entry_price:
                stop_price = self.entry_price * (1 - self.params.stop_loss_pct)
                if self.data.close[0] < stop_price:
                    exit_signal = True

        # Check additional take profit (if enabled and not already the primary exit type)
        if self.params.use_take_profit and exit_type != 'take_profit':
            if self.entry_price:
                target_price = self.entry_price * (1 + self.params.take_profit_pct)
                if self.data.close[0] > target_price:
                    exit_signal = True

        return exit_signal


class StrategyGenerator:
    """Generate all possible strategy combinations"""

    def __init__(self, entry_indicators, exit_indicators, config, entry_dual_indicators=None, exit_dual_indicators=None, entry_filters=None, exit_filters=None):
        """
        Initialize strategy generator

        Args:
            entry_indicators: List of (indicator_name, params) tuples for single indicators
            exit_indicators: List of (indicator_name, params) tuples for single indicators
            config: Configuration dictionary
            entry_dual_indicators: Optional list of dual indicator tuples for crossovers
            exit_dual_indicators: Optional list of dual indicator tuples for crossovers
            entry_filters: Optional list of (indicator_name, params, filter_type) tuples
            exit_filters: Optional list of (indicator_name, params, filter_type) tuples
        """
        self.entry_indicators = entry_indicators
        self.exit_indicators = exit_indicators
        self.entry_dual_indicators = entry_dual_indicators or []
        self.exit_dual_indicators = exit_dual_indicators or []
        self.entry_filters = entry_filters or []
        self.exit_filters = exit_filters or []
        self.config = config
        self.entry_types = config.get('entry_types', ['crossover', 'threshold', 'breakout'])
        self.exit_types = config.get('exit_types', ['crossover', 'stop_loss', 'take_profit'])

    def generate_strategies(self):
        """
        Generate all valid strategy combinations including risk management variations

        Returns:
            List of strategy configuration dictionaries
        """
        strategies = []
        strategy_id = 1

        # Get risk management config
        risk_config = self.config.get('risk_management', {})
        variations = risk_config.get('variations', ['none'])
        stop_loss_pcts = risk_config.get('stop_loss_pcts', [0.02])
        take_profit_pcts = risk_config.get('take_profit_pcts', [0.06])

        # Generate all combinations
        for entry_ind, exit_ind in product(self.entry_indicators, self.exit_indicators):
            entry_name, entry_params = entry_ind
            exit_name, exit_params = exit_ind

            # Debug: Show filter count on first iteration
            if strategy_id == 1 and self.entry_filters:
                print(f"DEBUG: Generating strategies with {len(self.entry_filters)} entry filters")
                for f in self.entry_filters:
                    print(f"  - {f}")

            # Determine valid entry types for this indicator
            valid_entry_types = self._get_valid_entry_types(entry_name)

            # Determine valid exit types for this indicator
            valid_exit_types = self._get_valid_exit_types(exit_name)

            # Create strategy for each valid combination
            for entry_type, exit_type in product(valid_entry_types, valid_exit_types):
                # Create variations with different risk management
                for variation in variations:
                    # Determine which stop loss and take profit values to test
                    sl_values = stop_loss_pcts if variation in ['stop_loss', 'both'] else [0.02]
                    tp_values = take_profit_pcts if variation in ['take_profit', 'both'] else [0.06]

                    # If variation is 'none', only test once with defaults
                    if variation == 'none':
                        sl_values = [0.02]
                        tp_values = [0.06]

                    for sl_pct, tp_pct in product(sl_values, tp_values):
                        # Skip redundant combinations
                        if variation == 'none' and (sl_pct != 0.02 or tp_pct != 0.06):
                            continue
                        # For stop_loss variation, only vary SL, keep TP fixed
                        if variation == 'stop_loss' and len(sl_values) > 1:
                            if tp_pct != tp_values[0]:  # Only use first TP value
                                continue
                        # For take_profit variation, only vary TP, keep SL fixed
                        if variation == 'take_profit' and len(tp_values) > 1:
                            if sl_pct != sl_values[0]:  # Only use first SL value
                                continue

                        use_stop_loss = variation in ['stop_loss', 'both']
                        use_take_profit = variation in ['take_profit', 'both']

                        # Test strategy without filter first
                        strategy_config = {
                            'id': strategy_id,
                            'entry_indicator': entry_name,
                            'entry_params': entry_params,
                            'entry_type': entry_type,
                            'entry_filter_indicator': '',
                            'entry_filter_params': (),
                            'entry_filter_type': '',
                            'exit_indicator': exit_name,
                            'exit_params': exit_params,
                            'exit_type': exit_type,
                            'exit_filter_indicator': '',
                            'exit_filter_params': (),
                            'exit_filter_type': '',
                            'use_stop_loss': use_stop_loss,
                            'stop_loss_pct': sl_pct,
                            'use_take_profit': use_take_profit,
                            'take_profit_pct': tp_pct,
                            'description': self._generate_description(
                                entry_name, entry_params, entry_type,
                                exit_name, exit_params, exit_type,
                                use_stop_loss, sl_pct, use_take_profit, tp_pct
                            )
                        }
                        strategies.append(strategy_config)
                        strategy_id += 1

                        # Now test with each entry filter
                        for filter_ind, filter_params, filter_type in self.entry_filters:
                            strategy_config_filtered = {
                                'id': strategy_id,
                                'entry_indicator': entry_name,
                                'entry_params': entry_params,
                                'entry_type': entry_type,
                                'entry_filter_indicator': filter_ind,
                                'entry_filter_params': tuple(filter_params),
                                'entry_filter_type': filter_type,
                                'exit_indicator': exit_name,
                                'exit_params': exit_params,
                                'exit_type': exit_type,
                                'exit_filter_indicator': '',
                                'exit_filter_params': (),
                                'exit_filter_type': '',
                                'use_stop_loss': use_stop_loss,
                                'stop_loss_pct': sl_pct,
                                'use_take_profit': use_take_profit,
                                'take_profit_pct': tp_pct,
                                'description': self._generate_description_with_filter(
                                    entry_name, entry_params, entry_type,
                                    exit_name, exit_params, exit_type,
                                    use_stop_loss, sl_pct, use_take_profit, tp_pct,
                                    filter_ind, filter_params, filter_type, is_entry=True
                                )
                            }
                            strategies.append(strategy_config_filtered)
                            strategy_id += 1

        # Generate dual indicator crossover strategies
        strategies.extend(self._generate_dual_indicator_strategies(strategy_id))

        return strategies

    def _generate_dual_indicator_strategies(self, starting_id):
        """Generate strategies using indicator-vs-indicator crossovers"""
        strategies = []
        strategy_id = starting_id

        if not self.entry_dual_indicators and not self.exit_dual_indicators:
            return strategies

        # Get risk management config
        risk_config = self.config.get('risk_management', {})
        variations = risk_config.get('variations', ['none'])
        stop_loss_pcts = risk_config.get('stop_loss_pcts', [0.02])
        take_profit_pcts = risk_config.get('take_profit_pcts', [0.06])

        # 1. Generate dual entry + regular exit combinations
        for entry_dual in self.entry_dual_indicators:
            (entry_ind1_name, entry_ind1_params), (entry_ind2_name, entry_ind2_params) = entry_dual

            for exit_ind in self.exit_indicators:
                exit_name, exit_params = exit_ind
                valid_exit_types = self._get_valid_exit_types(exit_name)

                for exit_type in valid_exit_types:
                    for variation in variations:
                        sl_values = stop_loss_pcts if variation in ['stop_loss', 'both'] else [0.02]
                        tp_values = take_profit_pcts if variation in ['take_profit', 'both'] else [0.06]

                        if variation == 'none':
                            sl_values = [0.02]
                            tp_values = [0.06]

                        for sl_pct, tp_pct in product(sl_values, tp_values):
                            if variation == 'none' and (sl_pct != 0.02 or tp_pct != 0.06):
                                continue
                            if variation == 'stop_loss' and len(sl_values) > 1 and tp_pct != tp_values[0]:
                                continue
                            if variation == 'take_profit' and len(tp_values) > 1 and sl_pct != sl_values[0]:
                                continue

                            use_stop_loss = variation in ['stop_loss', 'both']
                            use_take_profit = variation in ['take_profit', 'both']

                            strategy_config = {
                                'id': strategy_id,
                                'entry_indicator': entry_ind1_name,
                                'entry_params': entry_ind1_params,
                                'entry_indicator2': entry_ind2_name,
                                'entry_params2': entry_ind2_params,
                                'entry_type': 'dual_crossover',
                                'exit_indicator': exit_name,
                                'exit_params': exit_params,
                                'exit_indicator2': '',
                                'exit_params2': (),
                                'exit_type': exit_type,
                                'use_stop_loss': use_stop_loss,
                                'stop_loss_pct': sl_pct,
                                'use_take_profit': use_take_profit,
                                'take_profit_pct': tp_pct,
                                'description': self._generate_dual_description(
                                    entry_ind1_name, entry_ind1_params, entry_ind2_name, entry_ind2_params,
                                    exit_name, exit_params, exit_type,
                                    use_stop_loss, sl_pct, use_take_profit, tp_pct
                                )
                            }
                            strategies.append(strategy_config)
                            strategy_id += 1

        # 2. Generate regular entry + dual exit combinations
        for entry_ind in self.entry_indicators:
            entry_name, entry_params = entry_ind
            valid_entry_types = self._get_valid_entry_types(entry_name)

            for exit_dual in self.exit_dual_indicators:
                (exit_ind1_name, exit_ind1_params), (exit_ind2_name, exit_ind2_params) = exit_dual

                for entry_type in valid_entry_types:
                    for variation in variations:
                        sl_values = stop_loss_pcts if variation in ['stop_loss', 'both'] else [0.02]
                        tp_values = take_profit_pcts if variation in ['take_profit', 'both'] else [0.06]

                        if variation == 'none':
                            sl_values = [0.02]
                            tp_values = [0.06]

                        for sl_pct, tp_pct in product(sl_values, tp_values):
                            if variation == 'none' and (sl_pct != 0.02 or tp_pct != 0.06):
                                continue
                            if variation == 'stop_loss' and len(sl_values) > 1 and tp_pct != tp_values[0]:
                                continue
                            if variation == 'take_profit' and len(tp_values) > 1 and sl_pct != sl_values[0]:
                                continue

                            use_stop_loss = variation in ['stop_loss', 'both']
                            use_take_profit = variation in ['take_profit', 'both']

                            strategy_config = {
                                'id': strategy_id,
                                'entry_indicator': entry_name,
                                'entry_params': entry_params,
                                'entry_indicator2': '',
                                'entry_params2': (),
                                'entry_type': entry_type,
                                'exit_indicator': exit_ind1_name,
                                'exit_params': exit_ind1_params,
                                'exit_indicator2': exit_ind2_name,
                                'exit_params2': exit_ind2_params,
                                'exit_type': 'dual_crossover',
                                'use_stop_loss': use_stop_loss,
                                'stop_loss_pct': sl_pct,
                                'use_take_profit': use_take_profit,
                                'take_profit_pct': tp_pct,
                                'description': self._generate_dual_exit_description(
                                    entry_name, entry_params, entry_type,
                                    exit_ind1_name, exit_ind1_params, exit_ind2_name, exit_ind2_params,
                                    use_stop_loss, sl_pct, use_take_profit, tp_pct
                                )
                            }
                            strategies.append(strategy_config)
                            strategy_id += 1

        # 3. Generate dual entry + dual exit combinations
        for entry_dual in self.entry_dual_indicators:
            (entry_ind1_name, entry_ind1_params), (entry_ind2_name, entry_ind2_params) = entry_dual

            for exit_dual in self.exit_dual_indicators:
                (exit_ind1_name, exit_ind1_params), (exit_ind2_name, exit_ind2_params) = exit_dual

                for variation in variations:
                    sl_values = stop_loss_pcts if variation in ['stop_loss', 'both'] else [0.02]
                    tp_values = take_profit_pcts if variation in ['take_profit', 'both'] else [0.06]

                    if variation == 'none':
                        sl_values = [0.02]
                        tp_values = [0.06]

                    for sl_pct, tp_pct in product(sl_values, tp_values):
                        if variation == 'none' and (sl_pct != 0.02 or tp_pct != 0.06):
                            continue
                        if variation == 'stop_loss' and len(sl_values) > 1 and tp_pct != tp_values[0]:
                            continue
                        if variation == 'take_profit' and len(tp_values) > 1 and sl_pct != sl_values[0]:
                            continue

                        use_stop_loss = variation in ['stop_loss', 'both']
                        use_take_profit = variation in ['take_profit', 'both']

                        strategy_config = {
                            'id': strategy_id,
                            'entry_indicator': entry_ind1_name,
                            'entry_params': entry_ind1_params,
                            'entry_indicator2': entry_ind2_name,
                            'entry_params2': entry_ind2_params,
                            'entry_type': 'dual_crossover',
                            'exit_indicator': exit_ind1_name,
                            'exit_params': exit_ind1_params,
                            'exit_indicator2': exit_ind2_name,
                            'exit_params2': exit_ind2_params,
                            'exit_type': 'dual_crossover',
                            'use_stop_loss': use_stop_loss,
                            'stop_loss_pct': sl_pct,
                            'use_take_profit': use_take_profit,
                            'take_profit_pct': tp_pct,
                            'description': self._generate_full_dual_description(
                                entry_ind1_name, entry_ind1_params, entry_ind2_name, entry_ind2_params,
                                exit_ind1_name, exit_ind1_params, exit_ind2_name, exit_ind2_params,
                                use_stop_loss, sl_pct, use_take_profit, tp_pct
                            )
                        }
                        strategies.append(strategy_config)
                        strategy_id += 1

        return strategies

    def _get_valid_entry_types(self, indicator_name):
        """Determine valid entry types for an indicator"""
        # Oscillators use threshold crossovers
        oscillators = ['rsi', 'cci', 'stoch', 'stochrsi', 'williams', 'tsi', 'ultimate', 'mfi']

        # Band indicators can use both breakout and crossover
        bands = ['bb', 'keltner', 'donchian']

        # Trend indicators use crossovers
        trend_mas = ['sma', 'ema', 'wma', 'dema', 'tema', 'kama', 'zlema', 'hma', 'vwap']

        # Special cases
        if indicator_name in oscillators:
            return ['threshold']
        elif indicator_name in bands:
            # Bands support both breakout (break bands) and crossover (cross middle)
            return ['breakout', 'crossover']
        elif indicator_name in trend_mas:
            return ['crossover']
        elif indicator_name in ['macd', 'adx', 'dmi', 'aroon', 'psar', 'supertrend']:
            return ['crossover']
        elif indicator_name in ['atr', 'natr']:
            return ['threshold']  # ATR for volatility breakouts
        elif indicator_name in ['roc', 'momentum']:
            return ['threshold']  # Momentum crossing zero
        elif indicator_name in ['obv', 'adl', 'cmf']:
            return ['crossover']  # Volume indicators
        elif indicator_name in ['high', 'low', 'pivot']:
            return ['breakout']  # Support/resistance
        elif indicator_name == 'avgprice':
            return ['crossover']
        else:
            return ['crossover']  # Default to crossover

    def _get_valid_exit_types(self, indicator_name):
        """Determine valid exit types for an indicator"""
        # ATR and volatility indicators are good for stops
        volatility = ['atr', 'natr', 'bb', 'keltner', 'donchian']

        # Oscillators for exit signals
        oscillators = ['rsi', 'cci', 'stoch', 'stochrsi', 'williams', 'mfi']

        # Trend indicators for exits
        trend = ['sma', 'ema', 'wma', 'dema', 'tema', 'kama', 'zlema', 'hma',
                 'macd', 'adx', 'dmi', 'aroon', 'psar', 'supertrend']

        if indicator_name in volatility:
            return ['trailing_stop', 'stop_loss']
        elif indicator_name in oscillators:
            return ['threshold', 'crossover']
        elif indicator_name in trend:
            return ['crossover', 'stop_loss']
        elif indicator_name in ['roc', 'momentum', 'tsi', 'ultimate']:
            return ['crossover', 'threshold']
        elif indicator_name in ['obv', 'adl', 'cmf', 'vwap']:
            return ['crossover']
        elif indicator_name in ['high', 'low', 'pivot']:
            return ['stop_loss']  # Support/resistance as stops
        else:
            return ['crossover', 'stop_loss']  # Default

    def _generate_description(self, entry_ind, entry_params, entry_type,
                              exit_ind, exit_params, exit_type,
                              use_stop_loss=False, stop_loss_pct=0.02,
                              use_take_profit=False, take_profit_pct=0.06):
        """Generate human-readable strategy description"""
        entry_desc = get_indicator_description(entry_ind, entry_params)
        exit_desc = get_indicator_description(exit_ind, exit_params)

        description = f"Entry: {entry_type.upper()} on {entry_desc} | Exit: {exit_type.upper()} on {exit_desc}"

        # Add risk management details
        risk_parts = []
        if use_stop_loss:
            risk_parts.append(f"SL {stop_loss_pct*100:.0f}%")
        if use_take_profit:
            risk_parts.append(f"TP {take_profit_pct*100:.0f}%")

        if risk_parts:
            description += f" + {' + '.join(risk_parts)}"

        return description

    def _generate_description_with_filter(self, entry_ind, entry_params, entry_type,
                                          exit_ind, exit_params, exit_type,
                                          use_stop_loss=False, stop_loss_pct=0.02,
                                          use_take_profit=False, take_profit_pct=0.06,
                                          filter_ind='', filter_params=[], filter_type='', is_entry=True):
        """Generate description for filtered strategy"""
        # Get base description
        base_desc = self._generate_description(
            entry_ind, entry_params, entry_type,
            exit_ind, exit_params, exit_type,
            use_stop_loss, stop_loss_pct, use_take_profit, take_profit_pct
        )

        # Add filter description
        filter_desc = get_indicator_description(filter_ind, filter_params)
        filter_type_desc = {
            'above': f"when price > {filter_desc}",
            'below': f"when price < {filter_desc}",
            'rising': f"when {filter_desc} rising",
            'high': f"when {filter_desc} high"
        }.get(filter_type, f"with {filter_desc} {filter_type}")

        # Insert filter into description
        if is_entry:
            # Add filter to entry part
            parts = base_desc.split(" | ")
            parts[0] = f"{parts[0]} {filter_type_desc}"
            return " | ".join(parts)
        else:
            # Add filter to exit part
            parts = base_desc.split(" | ")
            parts[1] = f"{parts[1]} {filter_type_desc}"
            return " | ".join(parts)

    def _generate_dual_description(self, entry_ind1, entry_params1, entry_ind2, entry_params2,
                                   exit_ind, exit_params, exit_type,
                                   use_stop_loss=False, stop_loss_pct=0.02,
                                   use_take_profit=False, take_profit_pct=0.06):
        """Generate description for dual indicator entry strategy"""
        entry_desc1 = get_indicator_description(entry_ind1, entry_params1)
        entry_desc2 = get_indicator_description(entry_ind2, entry_params2)
        exit_desc = get_indicator_description(exit_ind, exit_params)

        description = f"Entry: {entry_desc1} crosses above {entry_desc2} | Exit: {exit_type.upper()} on {exit_desc}"

        # Add risk management details
        risk_parts = []
        if use_stop_loss:
            risk_parts.append(f"SL {stop_loss_pct*100:.0f}%")
        if use_take_profit:
            risk_parts.append(f"TP {take_profit_pct*100:.0f}%")

        if risk_parts:
            description += f" + {' + '.join(risk_parts)}"

        return description

    def _generate_dual_exit_description(self, entry_ind, entry_params, entry_type,
                                        exit_ind1, exit_params1, exit_ind2, exit_params2,
                                        use_stop_loss=False, stop_loss_pct=0.02,
                                        use_take_profit=False, take_profit_pct=0.06):
        """Generate description for dual indicator exit strategy"""
        entry_desc = get_indicator_description(entry_ind, entry_params)
        exit_desc1 = get_indicator_description(exit_ind1, exit_params1)
        exit_desc2 = get_indicator_description(exit_ind2, exit_params2)

        description = f"Entry: {entry_type.upper()} on {entry_desc} | Exit: {exit_desc1} crosses below {exit_desc2}"

        # Add risk management details
        risk_parts = []
        if use_stop_loss:
            risk_parts.append(f"SL {stop_loss_pct*100:.0f}%")
        if use_take_profit:
            risk_parts.append(f"TP {take_profit_pct*100:.0f}%")

        if risk_parts:
            description += f" + {' + '.join(risk_parts)}"

        return description

    def _generate_full_dual_description(self, entry_ind1, entry_params1, entry_ind2, entry_params2,
                                        exit_ind1, exit_params1, exit_ind2, exit_params2,
                                        use_stop_loss=False, stop_loss_pct=0.02,
                                        use_take_profit=False, take_profit_pct=0.06):
        """Generate description for fully dual indicator strategy"""
        entry_desc1 = get_indicator_description(entry_ind1, entry_params1)
        entry_desc2 = get_indicator_description(entry_ind2, entry_params2)
        exit_desc1 = get_indicator_description(exit_ind1, exit_params1)
        exit_desc2 = get_indicator_description(exit_ind2, exit_params2)

        description = f"Entry: {entry_desc1} crosses above {entry_desc2} | Exit: {exit_desc1} crosses below {exit_desc2}"

        # Add risk management details
        risk_parts = []
        if use_stop_loss:
            risk_parts.append(f"SL {stop_loss_pct*100:.0f}%")
        if use_take_profit:
            risk_parts.append(f"TP {take_profit_pct*100:.0f}%")

        if risk_parts:
            description += f" + {' + '.join(risk_parts)}"

        return description

    def get_strategy_count(self):
        """Get total number of strategies that will be generated"""
        count = 0
        for entry_ind, exit_ind in product(self.entry_indicators, self.exit_indicators):
            entry_name, _ = entry_ind
            exit_name, _ = exit_ind
            valid_entry_types = self._get_valid_entry_types(entry_name)
            valid_exit_types = self._get_valid_exit_types(exit_name)
            count += len(valid_entry_types) * len(valid_exit_types)
        return count


def create_strategy_class(strategy_config):
    """
    Create a backtrader strategy class from configuration

    Args:
        strategy_config: Strategy configuration dictionary

    Returns:
        Configured strategy class (not a tuple)
    """
    # Convert lists to tuples for backtrader compatibility
    entry_params = tuple(strategy_config['entry_params'])
    exit_params = tuple(strategy_config['exit_params'])
    entry_params2 = tuple(strategy_config.get('entry_params2', []))
    exit_params2 = tuple(strategy_config.get('exit_params2', []))
    entry_filter_params = tuple(strategy_config.get('entry_filter_params', ()))
    exit_filter_params = tuple(strategy_config.get('exit_filter_params', ()))

    # Create a proper subclass with params set as class attributes
    # We need to use type() to dynamically create the class to avoid closure issues
    class_params = (
        ('entry_indicator', strategy_config['entry_indicator']),
        ('entry_params', entry_params),
        ('entry_type', strategy_config['entry_type']),
        ('entry_indicator2', strategy_config.get('entry_indicator2', '')),
        ('entry_params2', entry_params2),
        ('entry_filter_indicator', strategy_config.get('entry_filter_indicator', '')),
        ('entry_filter_params', entry_filter_params),
        ('entry_filter_type', strategy_config.get('entry_filter_type', '')),
        ('exit_indicator', strategy_config['exit_indicator']),
        ('exit_params', exit_params),
        ('exit_type', strategy_config['exit_type']),
        ('exit_indicator2', strategy_config.get('exit_indicator2', '')),
        ('exit_params2', exit_params2),
        ('exit_filter_indicator', strategy_config.get('exit_filter_indicator', '')),
        ('exit_filter_params', exit_filter_params),
        ('exit_filter_type', strategy_config.get('exit_filter_type', '')),
        ('use_stop_loss', strategy_config.get('use_stop_loss', False)),
        ('stop_loss_pct', strategy_config.get('stop_loss_pct', 0.02)),
        ('use_take_profit', strategy_config.get('use_take_profit', False)),
        ('take_profit_pct', strategy_config.get('take_profit_pct', 0.06)),
        ('trailing_stop_pct', 0.03),
        ('printlog', False),
    )

    # Create the class using type() to avoid variable capture issues
    ConfiguredStrategy = type(
        'ConfiguredStrategy',
        (GenericStrategy,),
        {'params': class_params}
    )

    return ConfiguredStrategy