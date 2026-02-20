"""
Fundamental Data Module for Lorentzian Classification Strategy

Provides fundamental analysis and earnings awareness:
- Earnings date blackout windows
- Quality Score (0-100): Profitability, Balance Sheet, Valuation, Efficiency
- Growth Momentum Score (0-100): Earnings Surprise, Revenue Acceleration, Margin Expansion, Estimate Revisions
- Combined Trending Probability for position sizing

All scores are POINT-IN-TIME: only data publicly available at the backtest date
is used. Scores update each time a new earnings report is released.

Uses yfinance for all data.
"""

import math
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Suppress yfinance warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class FundamentalDataProvider:
    """
    Provides fundamental data and scores for a given symbol.
    Designed for both live trading and backtesting with point-in-time awareness.

    Scores use only quarterly financial statements that were publicly available
    at the as_of_date (based on earnings report dates).
    """

    def __init__(self, symbol: str, lookback_years: int = 5, verbose: bool = False):
        """
        Initialize and fetch all fundamental data for the symbol.

        Args:
            symbol: Ticker symbol
            lookback_years: Years of historical data to fetch
            verbose: Print debug information
        """
        self.symbol = symbol.upper()
        self.lookback_years = lookback_years
        self.verbose = verbose

        # Staleness parameters
        self.staleness_grace_days = 45  # No decay for first 45 days after earnings
        self.staleness_decay_rate = 0.005  # Decay per day after grace period

        # Data containers
        self._info: Dict = {}
        self._earnings_dates: Optional[pd.DataFrame] = None
        self._quarterly_income: Optional[pd.DataFrame] = None
        self._quarterly_balance: Optional[pd.DataFrame] = None
        self._earnings_history: Optional[pd.DataFrame] = None
        self._eps_revisions: Optional[Dict] = None
        self._shares_outstanding: Optional[float] = None
        self._price_history: Optional[pd.DataFrame] = None

        # Quarter-to-report-date mapping for point-in-time filtering
        self._quarter_report_map: Dict[pd.Timestamp, pd.Timestamp] = {}

        # Fetch all data
        self._fetch_data()

    def _fetch_data(self):
        """Fetch all fundamental data from yfinance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(self.symbol)

            # Basic info (for shares outstanding fallback and revision score)
            try:
                self._info = ticker.info or {}
                # Store shares outstanding for valuation fallback
                shares = self._info.get('sharesOutstanding')
                if shares is not None:
                    try:
                        self._shares_outstanding = float(shares)
                    except (TypeError, ValueError):
                        self._shares_outstanding = None
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not fetch info for {self.symbol}: {e}")
                self._info = {}

            # Earnings dates
            try:
                self._earnings_dates = ticker.get_earnings_dates(limit=20)
                if self._earnings_dates is not None and not self._earnings_dates.empty:
                    # Ensure index is datetime
                    if not isinstance(self._earnings_dates.index, pd.DatetimeIndex):
                        self._earnings_dates.index = pd.to_datetime(self._earnings_dates.index)
                    # Strip timezone info to avoid tz-naive vs tz-aware comparison errors
                    if self._earnings_dates.index.tz is not None:
                        self._earnings_dates.index = self._earnings_dates.index.tz_localize(None)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not fetch earnings dates for {self.symbol}: {e}")
                self._earnings_dates = None

            # Quarterly income statement
            try:
                self._quarterly_income = ticker.quarterly_income_stmt
                if self._quarterly_income is not None and not self._quarterly_income.empty:
                    # Strip timezone from column dates
                    self._quarterly_income.columns = self._strip_tz_from_index(
                        self._quarterly_income.columns)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not fetch quarterly income for {self.symbol}: {e}")
                self._quarterly_income = None

            # Quarterly balance sheet
            try:
                self._quarterly_balance = ticker.quarterly_balance_sheet
                if self._quarterly_balance is not None and not self._quarterly_balance.empty:
                    # Strip timezone from column dates
                    self._quarterly_balance.columns = self._strip_tz_from_index(
                        self._quarterly_balance.columns)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not fetch quarterly balance for {self.symbol}: {e}")
                self._quarterly_balance = None

            # Earnings history (EPS estimates vs actuals)
            try:
                self._earnings_history = ticker.earnings_history
                if self._earnings_history is not None and not self._earnings_history.empty:
                    if isinstance(self._earnings_history.index, pd.DatetimeIndex):
                        if self._earnings_history.index.tz is not None:
                            self._earnings_history.index = self._earnings_history.index.tz_localize(None)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not fetch earnings history for {self.symbol}: {e}")
                self._earnings_history = None

            # EPS revisions (current-only, no historical data available)
            try:
                self._eps_revisions = ticker.eps_revisions
                if self._eps_revisions is not None:
                    self._eps_revisions = self._eps_revisions.to_dict() if hasattr(self._eps_revisions, 'to_dict') else {}
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not fetch EPS revisions for {self.symbol}: {e}")
                self._eps_revisions = None

            # Historical price data (for post-earnings reaction measurement)
            try:
                start = datetime.now() - timedelta(days=lookback_years * 365 + 30)
                self._price_history = ticker.history(start=start.strftime('%Y-%m-%d'), auto_adjust=True)
                if self._price_history is not None and not self._price_history.empty:
                    if self._price_history.index.tz is not None:
                        self._price_history.index = self._price_history.index.tz_localize(None)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not fetch price history for {self.symbol}: {e}")
                self._price_history = None

            # Build quarter-to-report-date mapping
            self._build_quarter_map()

        except ImportError:
            raise ImportError("yfinance is required. Install with: pip install yfinance")
        except Exception as e:
            if self.verbose:
                print(f"Error fetching data for {self.symbol}: {e}")

    # =========================================================================
    # Point-in-Time Infrastructure
    # =========================================================================

    @staticmethod
    def _strip_tz_from_index(idx):
        """Strip timezone from a DatetimeIndex or Index of Timestamps."""
        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
            return idx.tz_localize(None)
        # Handle case where columns are Timestamps but not a DatetimeIndex
        try:
            new_idx = []
            for ts in idx:
                ts = pd.Timestamp(ts)
                if ts.tz is not None:
                    ts = ts.tz_localize(None)
                new_idx.append(ts)
            return pd.DatetimeIndex(new_idx)
        except Exception:
            return idx

    def _build_quarter_map(self):
        """
        Map each quarter-end date to the earnings report date when that data
        became publicly available. This enables point-in-time filtering.
        """
        self._quarter_report_map = {}

        # Collect all quarter-end dates from financial statements
        quarter_ends = set()
        if self._quarterly_income is not None and not self._quarterly_income.empty:
            for col in self._quarterly_income.columns:
                quarter_ends.add(pd.Timestamp(col))
        if self._quarterly_balance is not None and not self._quarterly_balance.empty:
            for col in self._quarterly_balance.columns:
                quarter_ends.add(pd.Timestamp(col))

        if not quarter_ends:
            return

        # Get sorted earnings dates
        earnings_dates_list = []
        if self._earnings_dates is not None and not self._earnings_dates.empty:
            earnings_dates_list = sorted(self._earnings_dates.index.tolist())

        for qe in sorted(quarter_ends):
            qe_ts = pd.Timestamp(qe)
            if qe_ts.tz is not None:
                qe_ts = qe_ts.tz_localize(None)

            # Find the nearest earnings date AFTER the quarter end
            report_date = None
            for ed in earnings_dates_list:
                ed_ts = pd.Timestamp(ed)
                if ed_ts.tz is not None:
                    ed_ts = ed_ts.tz_localize(None)
                if ed_ts > qe_ts:
                    report_date = ed_ts
                    break

            if report_date is None:
                # Fallback: quarter_end + 45 days
                report_date = qe_ts + pd.Timedelta(days=45)

            self._quarter_report_map[qe_ts] = report_date

        if self.verbose:
            print(f"  Quarter map for {self.symbol}: {len(self._quarter_report_map)} quarters mapped")
            for qe, rd in sorted(self._quarter_report_map.items()):
                print(f"    Q ending {qe.strftime('%Y-%m-%d')} -> reported {rd.strftime('%Y-%m-%d')}")

    def _get_available_quarters(self, as_of_date, df):
        """
        Filter a quarterly DataFrame to only columns whose earnings report
        date has passed by as_of_date.

        Args:
            as_of_date: The backtest date (only data reported before this is available)
            df: Quarterly DataFrame (income or balance sheet)

        Returns:
            Filtered DataFrame with only available columns, or None
        """
        if df is None or df.empty:
            return None

        if as_of_date is None:
            return df  # No filtering, use all data

        as_of_ts = pd.Timestamp(as_of_date)
        if as_of_ts.tz is not None:
            as_of_ts = as_of_ts.tz_localize(None)

        available_cols = []
        for col in df.columns:
            col_ts = pd.Timestamp(col)
            if col_ts.tz is not None:
                col_ts = col_ts.tz_localize(None)

            report_date = self._quarter_report_map.get(col_ts)
            if report_date is not None and report_date <= as_of_ts:
                available_cols.append(col)
            elif report_date is None:
                # No mapping found - use fallback: quarter_end + 45 days
                fallback_report = col_ts + pd.Timedelta(days=45)
                if fallback_report <= as_of_ts:
                    available_cols.append(col)

        if not available_cols:
            return None

        return df[available_cols]

    def _safe_get_quarterly(self, row_names, df, quarter_index=0):
        """
        Get a single value from a quarterly DataFrame, trying multiple row name variants.

        Args:
            row_names: List of possible row names (e.g. ['Net Income', 'NetIncome'])
            df: Pre-filtered quarterly DataFrame
            quarter_index: Which quarter to read (0=most recent available)

        Returns:
            Float value or None
        """
        if df is None or df.empty:
            return None

        for name in row_names:
            if name in df.index:
                values = df.loc[name].dropna()
                if len(values) > quarter_index:
                    try:
                        return float(values.iloc[quarter_index])
                    except (TypeError, ValueError):
                        continue

        return None

    def _calculate_ttm_metric(self, row_names, df):
        """
        Calculate trailing twelve months (TTM) value by summing available quarters.

        Args:
            row_names: List of possible row names
            df: Pre-filtered quarterly DataFrame

        Returns:
            TTM value (sum of up to 4 quarters, annualized if fewer), or None
        """
        if df is None or df.empty:
            return None

        for name in row_names:
            if name in df.index:
                values = df.loc[name].dropna()
                if len(values) == 0:
                    continue
                n_quarters = min(4, len(values))
                vals = values.head(n_quarters)
                total = float(vals.sum())
                # Annualize if fewer than 4 quarters
                if n_quarters < 4:
                    total = total * (4 / n_quarters)
                return total

        return None

    # =========================================================================
    # Earnings Date Methods
    # =========================================================================

    def get_earnings_dates(self) -> Optional[pd.DataFrame]:
        """Return DataFrame of earnings dates (past and future)."""
        return self._earnings_dates

    def is_in_earnings_blackout(self, date: datetime, days_before: int = 5, days_after: int = 2) -> bool:
        """
        Check if date falls within earnings blackout window.

        Args:
            date: Date to check
            days_before: Days before earnings to block
            days_after: Days after earnings to block

        Returns:
            True if in blackout window, False otherwise
        """
        if self._earnings_dates is None or self._earnings_dates.empty:
            return False

        # Convert date to pandas Timestamp and ensure tz-naive
        check_date = pd.Timestamp(date)
        if check_date.tz is not None:
            check_date = check_date.tz_localize(None)

        for earnings_date in self._earnings_dates.index:
            earnings_ts = pd.Timestamp(earnings_date)
            if earnings_ts.tz is not None:
                earnings_ts = earnings_ts.tz_localize(None)

            # Calculate blackout window
            blackout_start = earnings_ts - pd.Timedelta(days=days_before)
            blackout_end = earnings_ts + pd.Timedelta(days=days_after)

            if blackout_start <= check_date <= blackout_end:
                return True

        return False

    def days_until_earnings(self, date: datetime) -> Optional[int]:
        """
        Return days until next earnings (positive if future, None if no data).

        Args:
            date: Reference date

        Returns:
            Days until next earnings, or None if no future earnings found
        """
        if self._earnings_dates is None or self._earnings_dates.empty:
            return None

        # Ensure tz-naive for comparison
        check_date = pd.Timestamp(date)
        if check_date.tz is not None:
            check_date = check_date.tz_localize(None)

        # Find future earnings dates
        future_earnings = self._earnings_dates.index[self._earnings_dates.index > check_date]

        if len(future_earnings) == 0:
            return None

        next_earnings = future_earnings.min()
        days = (next_earnings - check_date).days

        return days

    def days_since_earnings(self, date: datetime) -> Optional[int]:
        """
        Return days since last earnings report.

        Args:
            date: Reference date

        Returns:
            Days since last earnings, or None if no past earnings found
        """
        if self._earnings_dates is None or self._earnings_dates.empty:
            return None

        # Ensure tz-naive for comparison
        check_date = pd.Timestamp(date)
        if check_date.tz is not None:
            check_date = check_date.tz_localize(None)

        # Find past earnings dates
        past_earnings = self._earnings_dates.index[self._earnings_dates.index <= check_date]

        if len(past_earnings) == 0:
            return None

        last_earnings = past_earnings.max()
        days = (check_date - last_earnings).days

        return days

    def get_next_earnings_date(self, date: datetime) -> Optional[datetime]:
        """Get the next earnings date after the given date."""
        if self._earnings_dates is None or self._earnings_dates.empty:
            return None

        # Ensure tz-naive for comparison
        check_date = pd.Timestamp(date)
        if check_date.tz is not None:
            check_date = check_date.tz_localize(None)

        future_earnings = self._earnings_dates.index[self._earnings_dates.index > check_date]

        if len(future_earnings) == 0:
            return None

        return future_earnings.min().to_pydatetime()

    # =========================================================================
    # Quality Score (0-100) - Point-in-Time from Quarterly Statements
    # =========================================================================

    def get_quality_score(self, as_of_date: datetime = None, price: float = None) -> float:
        """
        Calculate fundamental quality score using quarterly data available at as_of_date.

        Four pillars, 25 points each:
        - Profitability: ROE, ROA, profit margins, operating efficiency
        - Balance Sheet: Current ratio, debt/equity, interest coverage
        - Valuation: Trailing P/E, earnings yield, price/book (requires price)
        - Efficiency: Asset turnover, operating leverage

        Args:
            as_of_date: Only use data publicly available by this date
            price: Current stock price (for valuation metrics)

        Returns:
            Quality score from 0-100
        """
        # Filter quarterly data to only what was reported by as_of_date
        income_df = self._get_available_quarters(as_of_date, self._quarterly_income)
        balance_df = self._get_available_quarters(as_of_date, self._quarterly_balance)

        # If no quarterly data available yet, return neutral
        if income_df is None and balance_df is None:
            if self.verbose:
                print(f"  Quality Score for {self.symbol}: No quarterly data available as of {as_of_date} -> neutral 50")
            return 50.0

        profitability = self._calculate_profitability_score(income_df, balance_df)
        balance_sheet = self._calculate_balance_sheet_score(income_df, balance_df)
        valuation = self._calculate_valuation_score(income_df, balance_df, price)
        efficiency = self._calculate_efficiency_score(income_df, balance_df)

        total = profitability + balance_sheet + valuation + efficiency

        if self.verbose:
            n_quarters = 0
            if income_df is not None:
                n_quarters = len(income_df.columns)
            print(f"  Quality Score Breakdown for {self.symbol} (as of {as_of_date}, {n_quarters}Q avail):")
            print(f"    Profitability: {profitability:.1f}/25")
            print(f"    Balance Sheet: {balance_sheet:.1f}/25")
            print(f"    Valuation: {valuation:.1f}/25")
            print(f"    Efficiency: {efficiency:.1f}/25")
            print(f"    Total: {total:.1f}/100")

        return total

    def _calculate_profitability_score(self, income_df, balance_df) -> float:
        """Sub-score for profitability pillar (0-25) using quarterly data."""
        score = 0.0

        net_income = self._safe_get_quarterly(
            ['Net Income', 'NetIncome', 'Net Income Common Stockholders'], income_df)
        equity = self._safe_get_quarterly(
            ['Stockholders Equity', 'StockholdersEquity', 'Common Stock Equity',
             'Total Equity Gross Minority Interest'], balance_df)
        total_assets = self._safe_get_quarterly(
            ['Total Assets', 'TotalAssets'], balance_df)
        revenue = self._safe_get_quarterly(
            ['Total Revenue', 'TotalRevenue', 'Revenue'], income_df)
        op_income = self._safe_get_quarterly(
            ['Operating Income', 'OperatingIncome', 'EBIT'], income_df)

        # ROE (annualized from latest quarter) - 0-8 points
        if net_income is not None and equity is not None and equity != 0:
            roe = (net_income * 4) / equity
            if roe > 0.25:
                score += 8
            elif roe > 0.15:
                score += 6
            elif roe > 0.10:
                score += 4
            elif roe > 0.05:
                score += 2

        # ROA (annualized from latest quarter) - 0-6 points
        if net_income is not None and total_assets is not None and total_assets != 0:
            roa = (net_income * 4) / total_assets
            if roa > 0.15:
                score += 6
            elif roa > 0.10:
                score += 4
            elif roa > 0.05:
                score += 2

        # Profit Margin - 0-6 points
        if net_income is not None and revenue is not None and revenue != 0:
            margin = net_income / revenue
            if margin > 0.20:
                score += 6
            elif margin > 0.10:
                score += 4
            elif margin > 0.05:
                score += 2

        # Operating Efficiency - 0-5 points
        if op_income is not None and revenue is not None and revenue != 0:
            op_margin = op_income / revenue
            if op_margin > 0.25:
                score += 5
            elif op_margin > 0.15:
                score += 3
            elif op_margin > 0.05:
                score += 1

        return min(25.0, score)

    def _calculate_balance_sheet_score(self, income_df, balance_df) -> float:
        """Sub-score for balance sheet pillar (0-25) using quarterly data."""
        score = 0.0

        current_assets = self._safe_get_quarterly(
            ['Current Assets', 'CurrentAssets'], balance_df)
        current_liabilities = self._safe_get_quarterly(
            ['Current Liabilities', 'CurrentLiabilities'], balance_df)
        total_debt = self._safe_get_quarterly(
            ['Total Debt', 'TotalDebt'], balance_df)
        equity = self._safe_get_quarterly(
            ['Stockholders Equity', 'StockholdersEquity', 'Common Stock Equity'], balance_df)
        op_income = self._safe_get_quarterly(
            ['Operating Income', 'OperatingIncome', 'EBIT'], income_df)
        interest_expense = self._safe_get_quarterly(
            ['Interest Expense', 'InterestExpense', 'Interest Expense Non Operating',
             'Net Interest Income'], income_df)

        # Current Ratio - 0-8 points
        if (current_assets is not None and current_liabilities is not None
                and current_liabilities != 0):
            current_ratio = current_assets / current_liabilities
            if 1.5 <= current_ratio <= 3.0:
                score += 8
            elif 1.2 <= current_ratio < 1.5:
                score += 5
            elif current_ratio > 3.0:
                score += 4  # Too high = inefficient capital
            elif current_ratio >= 1.0:
                score += 2

        # Debt to Equity - 0-9 points (lower is better)
        if equity is not None and equity > 0:
            if total_debt is not None:
                de = (total_debt / equity) * 100  # As percentage
                if de < 30:
                    score += 9
                elif de < 50:
                    score += 7
                elif de < 100:
                    score += 4
                elif de < 200:
                    score += 2
            else:
                # No debt data found, assume low debt
                score += 7

        # Interest Coverage - 0-8 points
        if op_income is not None and interest_expense is not None and interest_expense != 0:
            coverage = op_income / abs(interest_expense)
            if coverage > 10:
                score += 8
            elif coverage > 5:
                score += 6
            elif coverage > 3:
                score += 4
            elif coverage > 1.5:
                score += 2
        elif total_debt is not None and total_debt == 0:
            score += 8  # No debt is good
        elif op_income is not None and (interest_expense is None or interest_expense == 0):
            # Has operating income but no interest expense
            score += 8

        return min(25.0, score)

    def _calculate_valuation_score(self, income_df, balance_df, price) -> float:
        """Sub-score for valuation pillar (0-25) using quarterly data + stock price."""
        if price is None or price <= 0:
            return 12.5  # Neutral when no price available

        score = 0.0

        # Trailing P/E from TTM EPS - 0-8 points
        ttm_eps = self._calculate_ttm_metric(
            ['Diluted EPS', 'DilutedEPS', 'Basic EPS', 'BasicEPS'], income_df)

        if ttm_eps is not None and ttm_eps > 0:
            trailing_pe = price / ttm_eps
            if trailing_pe < 12:
                score += 8
            elif trailing_pe < 18:
                score += 6
            elif trailing_pe < 25:
                score += 4
            elif trailing_pe < 40:
                score += 2
        elif ttm_eps is not None and ttm_eps <= 0:
            score += 0  # Negative earnings
        else:
            score += 4  # Neutral - no EPS data

        # Earnings Yield (TTM EPS / Price) - 0-9 points
        if ttm_eps is not None and ttm_eps > 0:
            earnings_yield = ttm_eps / price
            if earnings_yield > 0.10:
                score += 9
            elif earnings_yield > 0.06:
                score += 7
            elif earnings_yield > 0.04:
                score += 5
            elif earnings_yield > 0.02:
                score += 3
            else:
                score += 1
        else:
            score += 4.5  # Neutral

        # Price/Book - 0-8 points
        equity = self._safe_get_quarterly(
            ['Stockholders Equity', 'StockholdersEquity', 'Common Stock Equity'], balance_df)
        shares = self._safe_get_quarterly(
            ['Ordinary Shares Number', 'OrdinarySharesNumber',
             'Share Issued', 'ShareIssued'], balance_df)

        # Fallback to ticker.info for shares if not in balance sheet
        if shares is None and self._shares_outstanding is not None:
            shares = self._shares_outstanding

        if equity is not None and shares is not None and shares > 0:
            book_value_per_share = equity / shares
            if book_value_per_share > 0:
                pb = price / book_value_per_share
                if pb < 1.0:
                    score += 8
                elif pb < 2.0:
                    score += 6
                elif pb < 3.0:
                    score += 4
                elif pb < 5.0:
                    score += 2
            else:
                score += 4  # Negative book value
        else:
            score += 4  # Neutral

        return min(25.0, score)

    def _calculate_efficiency_score(self, income_df, balance_df) -> float:
        """Sub-score for efficiency pillar (0-25) using quarterly data."""
        score = 0.0

        revenue = self._safe_get_quarterly(
            ['Total Revenue', 'TotalRevenue', 'Revenue'], income_df)
        total_assets = self._safe_get_quarterly(
            ['Total Assets', 'TotalAssets'], balance_df)
        op_income = self._safe_get_quarterly(
            ['Operating Income', 'OperatingIncome', 'EBIT'], income_df)
        gross_profit = self._safe_get_quarterly(
            ['Gross Profit', 'GrossProfit'], income_df)

        # Asset Turnover (annualized) - 0-12 points
        if revenue is not None and total_assets is not None and total_assets > 0:
            turnover = (revenue * 4) / total_assets
            if turnover > 1.5:
                score += 12
            elif turnover > 1.0:
                score += 9
            elif turnover > 0.7:
                score += 6
            elif turnover > 0.5:
                score += 3
        else:
            score += 6  # Neutral

        # Operating Leverage (Operating Income / Gross Profit) - 0-13 points
        if gross_profit is not None and gross_profit > 0 and op_income is not None:
            op_leverage = op_income / gross_profit
            if op_leverage > 0.6:
                score += 13
            elif op_leverage > 0.4:
                score += 10
            elif op_leverage > 0.3:
                score += 7
            elif op_leverage > 0.2:
                score += 4
        elif op_income is not None and revenue is not None and revenue > 0:
            # Fallback to operating margin
            op_margin = op_income / revenue
            if op_margin > 0.20:
                score += 10
            elif op_margin > 0.10:
                score += 7
            elif op_margin > 0.05:
                score += 4
        else:
            score += 6.5  # Neutral

        return min(25.0, score)

    # =========================================================================
    # Growth Momentum Score (0-100) - Point-in-Time
    # =========================================================================

    def get_growth_momentum_score(self, as_of_date: datetime = None) -> float:
        """
        Calculate growth momentum score using data available at as_of_date.

        Four components, 25 points each:
        - Earnings Surprise: Last 4Q surprise %, weighted toward recent
        - Revenue Acceleration: YoY growth rate, 2nd derivative
        - Margin Expansion: Operating margin trend over available quarters
        - Estimate Revisions: Direction of analyst estimate changes (current-only)

        Args:
            as_of_date: Only use data publicly available by this date

        Returns:
            Growth momentum score from 0-100
        """
        # Filter quarterly income to available data
        income_df = self._get_available_quarters(as_of_date, self._quarterly_income)

        surprise = self._calculate_surprise_score(as_of_date)
        revenue_accel = self._calculate_revenue_acceleration_score(income_df)
        margin_expansion = self._calculate_margin_expansion_score(income_df)
        revisions = self._calculate_revision_score()

        total = surprise + revenue_accel + margin_expansion + revisions

        if self.verbose:
            n_quarters = 0
            if income_df is not None:
                n_quarters = len(income_df.columns)
            print(f"  Growth Momentum Breakdown for {self.symbol} (as of {as_of_date}, {n_quarters}Q avail):")
            print(f"    Earnings Surprise: {surprise:.1f}/25")
            print(f"    Revenue Acceleration: {revenue_accel:.1f}/25")
            print(f"    Margin Expansion: {margin_expansion:.1f}/25")
            print(f"    Estimate Revisions: {revisions:.1f}/25")
            print(f"    Total: {total:.1f}/100")

        return total

    def _calculate_surprise_score(self, as_of_date=None) -> float:
        """
        Score based on earnings surprises reported before as_of_date.
        More recent surprises weighted higher.
        """
        if self._earnings_history is None or self._earnings_history.empty:
            return 12.5  # Neutral score for missing data

        try:
            eh = self._earnings_history

            # Filter to only surprises reported before as_of_date
            if as_of_date is not None:
                as_of_ts = pd.Timestamp(as_of_date)
                if as_of_ts.tz is not None:
                    as_of_ts = as_of_ts.tz_localize(None)

                if isinstance(eh.index, pd.DatetimeIndex):
                    idx = eh.index
                    if idx.tz is not None:
                        idx = idx.tz_localize(None)
                    mask = idx <= as_of_ts
                    eh = eh[mask]

                if eh.empty:
                    return 12.5

            # Get surprise percentages
            if 'surprisePercent' in eh.columns:
                surprises = eh['surprisePercent'].dropna().head(4).tolist()
            elif 'Surprise(%)' in eh.columns:
                surprises = eh['Surprise(%)'].dropna().head(4).tolist()
            else:
                return 12.5

            if not surprises:
                return 12.5

            # Weights: most recent weighted highest
            weights = [0.4, 0.3, 0.2, 0.1]
            weighted_surprise = 0.0
            total_weight = 0.0

            for i, surprise in enumerate(surprises):
                weight = weights[i] if i < len(weights) else 0.1
                weighted_surprise += float(surprise) * weight
                total_weight += weight

            avg_surprise = weighted_surprise / total_weight if total_weight > 0 else 0

            # Map to 0-25 scale
            if avg_surprise > 10:
                return 25
            elif avg_surprise > 5:
                return 20
            elif avg_surprise > 2:
                return 15
            elif avg_surprise > 0:
                return 12
            elif avg_surprise > -2:
                return 8
            elif avg_surprise > -5:
                return 4
            else:
                return 0

        except Exception as e:
            if self.verbose:
                print(f"Error calculating surprise score: {e}")
            return 12.5

    def _calculate_revenue_acceleration_score(self, income_df=None) -> float:
        """
        Score based on YoY revenue growth rate and whether it's accelerating.
        Uses pre-filtered quarterly income data.
        """
        df = income_df if income_df is not None else self._quarterly_income
        if df is None or df.empty:
            return 12.5

        try:
            # Get revenue row
            revenue_row = None
            for name in ['Total Revenue', 'TotalRevenue', 'Revenue']:
                if name in df.index:
                    revenue_row = df.loc[name]
                    break

            if revenue_row is None:
                return 12.5

            # Get quarterly revenue values (columns are dates, most recent first)
            revenues = revenue_row.dropna().head(6).tolist()

            if len(revenues) < 5:
                # Not enough data for YoY comparison
                return 12.5

            # Calculate YoY growth for recent quarter (Q0 vs Q4)
            recent_yoy = (revenues[0] - revenues[4]) / abs(revenues[4]) if revenues[4] != 0 else 0

            # Calculate YoY growth for prior quarter (Q1 vs Q5)
            if len(revenues) >= 6:
                prior_yoy = (revenues[1] - revenues[5]) / abs(revenues[5]) if revenues[5] != 0 else recent_yoy
            else:
                prior_yoy = recent_yoy

            # Growth rate score (0-12.5)
            growth_score = 0
            if recent_yoy > 0.30:
                growth_score = 12.5
            elif recent_yoy > 0.15:
                growth_score = 10
            elif recent_yoy > 0.05:
                growth_score = 7
            elif recent_yoy > 0:
                growth_score = 4
            else:
                growth_score = 0

            # Acceleration score (0-12.5)
            acceleration = recent_yoy - prior_yoy
            accel_score = 0
            if acceleration > 0.05:
                accel_score = 12.5
            elif acceleration > 0.02:
                accel_score = 10
            elif acceleration > 0:
                accel_score = 7
            elif acceleration > -0.02:
                accel_score = 4
            else:
                accel_score = 0

            return min(25.0, growth_score + accel_score)

        except Exception as e:
            if self.verbose:
                print(f"Error calculating revenue acceleration: {e}")
            return 12.5

    def _calculate_margin_expansion_score(self, income_df=None) -> float:
        """
        Score based on operating margin trends over available quarters.
        Uses pre-filtered quarterly income data.
        """
        df = income_df if income_df is not None else self._quarterly_income
        if df is None or df.empty:
            return 12.5

        try:
            # Get operating income and revenue rows
            op_income_row = None
            revenue_row = None

            for name in ['Operating Income', 'OperatingIncome', 'EBIT']:
                if name in df.index:
                    op_income_row = df.loc[name]
                    break

            for name in ['Total Revenue', 'TotalRevenue', 'Revenue']:
                if name in df.index:
                    revenue_row = df.loc[name]
                    break

            if op_income_row is None or revenue_row is None:
                return 12.5  # No data available

            # Calculate quarterly margins
            margins = []
            for col in op_income_row.index[:4]:  # Last 4 available quarters
                if col in revenue_row.index:
                    op = op_income_row[col]
                    rev = revenue_row[col]
                    if pd.notna(op) and pd.notna(rev) and rev != 0:
                        margins.append(op / rev)

            if len(margins) < 2:
                return 12.5

            # Calculate trend (recent vs older)
            recent = np.mean(margins[:2]) if len(margins) >= 2 else margins[0]
            older = np.mean(margins[2:4]) if len(margins) >= 4 else margins[-1]
            margin_change = recent - older

            if margin_change > 0.03:
                return 25  # >3% margin expansion
            elif margin_change > 0.01:
                return 20
            elif margin_change > 0:
                return 15
            elif margin_change > -0.01:
                return 10
            elif margin_change > -0.03:
                return 5
            else:
                return 0

        except Exception as e:
            if self.verbose:
                print(f"Error calculating margin expansion: {e}")
            return 12.5

    def _calculate_revision_score(self) -> float:
        """
        Score based on analyst estimate revision direction.
        Current-only data (no historical data available from yfinance).
        """
        if self._eps_revisions is None or not self._eps_revisions:
            return 12.5  # Neutral for missing data

        try:
            # Extract revision counts
            if isinstance(self._eps_revisions, dict):
                revisions = self._eps_revisions
            else:
                return 12.5

            up_7d = 0
            down_7d = 0
            up_30d = 0
            down_30d = 0

            # Try to extract from the nested structure
            for period, data in revisions.items():
                if isinstance(data, dict):
                    if 'upLast7days' in data:
                        up_7d += data.get('upLast7days', 0) or 0
                    if 'downLast7days' in data:
                        down_7d += data.get('downLast7days', 0) or 0
                    if 'upLast30days' in data:
                        up_30d += data.get('upLast30days', 0) or 0
                    if 'downLast30days' in data:
                        down_30d += data.get('downLast30days', 0) or 0

            # Net revisions
            net_7d = up_7d - down_7d
            net_30d = up_30d - down_30d

            # Weight recent more heavily
            net_score = net_7d * 0.6 + net_30d * 0.4

            if net_score > 5:
                return 25
            elif net_score > 2:
                return 20
            elif net_score > 0:
                return 15
            elif net_score == 0:
                return 12.5
            elif net_score > -2:
                return 8
            elif net_score > -5:
                return 4
            else:
                return 0

        except Exception as e:
            if self.verbose:
                print(f"Error calculating revision score: {e}")
            return 12.5

    # =========================================================================
    # Earnings Improvement (Mini-Composite: EPS YoY + Revenue YoY + Margin)
    # =========================================================================

    def get_earnings_improvement(self, as_of_date: datetime = None) -> Optional[float]:
        """
        Mini-composite earnings improvement score comparing most recent available
        quarter vs the year-ago quarter (YoY).

        Three equally-weighted components:
        - EPS YoY growth (0-33): Is earnings per share growing year-over-year?
        - Revenue YoY growth (0-33): Is revenue growing year-over-year?
        - Operating Margin direction (0-34): Is op margin expanding vs year-ago quarter?

        Returns:
            Score from 0-100 (>50 = improving, <50 = deteriorating), or None if
            insufficient data (fewer than 5 quarters available).

        Point-in-time safe: only uses quarterly data publicly available by as_of_date.
        """
        income_df = self._get_available_quarters(as_of_date, self._quarterly_income)
        if income_df is None or len(income_df.columns) < 5:
            # Need at least 5 quarters: Q0 and Q4 (year-ago) for YoY comparison
            return None

        # --- EPS YoY (0-33) ---
        eps_score = self._earnings_improvement_eps(income_df)

        # --- Revenue YoY (0-33) ---
        revenue_score = self._earnings_improvement_revenue(income_df)

        # --- Operating Margin Direction (0-34) ---
        margin_score = self._earnings_improvement_margin(income_df)

        total = eps_score + revenue_score + margin_score

        if self.verbose:
            print(f"  Earnings Improvement for {self.symbol} (as of {as_of_date}):")
            print(f"    EPS YoY: {eps_score:.1f}/33")
            print(f"    Revenue YoY: {revenue_score:.1f}/33")
            print(f"    Margin Direction: {margin_score:.1f}/34")
            print(f"    Total: {total:.1f}/100")

        return total

    def get_earnings_reaction(self, as_of_date, reaction_days=5):
        """
        Calculate price reaction (%) in the days following the most recent
        earnings release before as_of_date.

        Returns:
            Percentage price change from earnings date close to close
            reaction_days trading days later, or None if insufficient data.
        """
        if self._earnings_dates is None or self._price_history is None:
            return None

        check_date = pd.Timestamp(as_of_date)
        if check_date.tz is not None:
            check_date = check_date.tz_localize(None)

        # Find most recent earnings date before as_of_date
        past_earnings = self._earnings_dates.index[self._earnings_dates.index <= check_date]
        if len(past_earnings) == 0:
            return None
        last_earnings = past_earnings.max()

        # Get close prices on and after earnings date
        prices = self._price_history['Close']
        post_dates = prices.index[prices.index >= last_earnings]
        if len(post_dates) < 2:
            return None
        earnings_close = float(prices.loc[post_dates[0]])

        # Find close reaction_days trading days later
        if len(post_dates) <= reaction_days:
            reaction_close = float(prices.loc[post_dates[-1]])
        else:
            reaction_close = float(prices.loc[post_dates[reaction_days]])

        if earnings_close <= 0:
            return None

        reaction_pct = ((reaction_close - earnings_close) / earnings_close) * 100.0

        if self.verbose:
            print(f"  Earnings Reaction for {self.symbol} (as of {as_of_date}):")
            print(f"    Last earnings: {last_earnings.date()}")
            print(f"    Earnings close: ${earnings_close:.2f}")
            print(f"    Reaction close ({reaction_days}d): ${reaction_close:.2f}")
            print(f"    Reaction: {reaction_pct:+.1f}%")

        return reaction_pct

    def _earnings_improvement_eps(self, income_df) -> float:
        """EPS YoY component (0-33). Compares Q0 vs Q4 (same quarter last year)."""
        eps_now = self._safe_get_quarterly(
            ['Diluted EPS', 'DilutedEPS', 'Basic EPS', 'BasicEPS'], income_df, quarter_index=0)
        eps_yago = self._safe_get_quarterly(
            ['Diluted EPS', 'DilutedEPS', 'Basic EPS', 'BasicEPS'], income_df, quarter_index=4)

        if eps_now is None or eps_yago is None:
            return 16.5  # Neutral

        # Handle edge cases
        if eps_yago == 0:
            if eps_now > 0:
                return 33.0  # Went from zero to positive
            elif eps_now < 0:
                return 0.0   # Went from zero to negative
            return 16.5      # Both zero

        yoy_growth = (eps_now - eps_yago) / abs(eps_yago)

        # Map to 0-33 scale
        if yoy_growth > 0.30:
            return 33.0
        elif yoy_growth > 0.15:
            return 28.0
        elif yoy_growth > 0.05:
            return 23.0
        elif yoy_growth > 0.0:
            return 18.0
        elif yoy_growth > -0.05:
            return 13.0
        elif yoy_growth > -0.15:
            return 8.0
        elif yoy_growth > -0.30:
            return 4.0
        else:
            return 0.0

    def _earnings_improvement_revenue(self, income_df) -> float:
        """Revenue YoY component (0-33). Compares Q0 vs Q4 (same quarter last year)."""
        rev_now = self._safe_get_quarterly(
            ['Total Revenue', 'TotalRevenue', 'Revenue'], income_df, quarter_index=0)
        rev_yago = self._safe_get_quarterly(
            ['Total Revenue', 'TotalRevenue', 'Revenue'], income_df, quarter_index=4)

        if rev_now is None or rev_yago is None:
            return 16.5  # Neutral

        if rev_yago == 0:
            return 16.5

        yoy_growth = (rev_now - rev_yago) / abs(rev_yago)

        # Map to 0-33 scale
        if yoy_growth > 0.20:
            return 33.0
        elif yoy_growth > 0.10:
            return 28.0
        elif yoy_growth > 0.03:
            return 23.0
        elif yoy_growth > 0.0:
            return 18.0
        elif yoy_growth > -0.03:
            return 13.0
        elif yoy_growth > -0.10:
            return 8.0
        elif yoy_growth > -0.20:
            return 4.0
        else:
            return 0.0

    def _earnings_improvement_margin(self, income_df) -> float:
        """Operating margin direction (0-34). Compares Q0 margin vs Q4 margin."""
        op_now = self._safe_get_quarterly(
            ['Operating Income', 'OperatingIncome', 'EBIT'], income_df, quarter_index=0)
        rev_now = self._safe_get_quarterly(
            ['Total Revenue', 'TotalRevenue', 'Revenue'], income_df, quarter_index=0)
        op_yago = self._safe_get_quarterly(
            ['Operating Income', 'OperatingIncome', 'EBIT'], income_df, quarter_index=4)
        rev_yago = self._safe_get_quarterly(
            ['Total Revenue', 'TotalRevenue', 'Revenue'], income_df, quarter_index=4)

        if (op_now is None or rev_now is None or rev_now == 0
                or op_yago is None or rev_yago is None or rev_yago == 0):
            return 17.0  # Neutral

        margin_now = op_now / rev_now
        margin_yago = op_yago / rev_yago
        margin_change = margin_now - margin_yago  # In percentage points

        # Map to 0-34 scale
        if margin_change > 0.05:
            return 34.0   # >5pp expansion
        elif margin_change > 0.02:
            return 28.0
        elif margin_change > 0.0:
            return 22.0
        elif margin_change > -0.02:
            return 12.0
        elif margin_change > -0.05:
            return 6.0
        else:
            return 0.0    # >5pp contraction

    # =========================================================================
    # Combined Score
    # =========================================================================

    def get_trending_probability(self, as_of_date: datetime = None,
                                  quality_weight: float = 0.4,
                                  momentum_weight: float = 0.6,
                                  price: float = None) -> float:
        """
        Calculate combined trending probability.

        Args:
            as_of_date: Date for point-in-time data
            quality_weight: Weight for quality score (default 0.4)
            momentum_weight: Weight for momentum score (default 0.6)
            price: Current stock price (for valuation metrics)

        Returns:
            Combined score from 0-100
        """
        quality = self.get_quality_score(as_of_date, price=price)
        momentum = self.get_growth_momentum_score(as_of_date)

        # Normalize weights so they sum to 1.0 (output stays 0-100)
        total_weight = quality_weight + momentum_weight
        if total_weight > 0:
            q_norm = quality_weight / total_weight
            m_norm = momentum_weight / total_weight
        else:
            q_norm = 0.5
            m_norm = 0.5

        combined = (quality * q_norm) + (momentum * m_norm)

        if self.verbose:
            print(f"  Combined Trending Probability for {self.symbol}:")
            print(f"    Quality Score: {quality:.1f}")
            print(f"    Momentum Score: {momentum:.1f}")
            print(f"    Combined ({quality_weight:.0%} Q + {momentum_weight:.0%} M): {combined:.1f}")

        return combined

    # =========================================================================
    # Staleness / Confidence Decay
    # =========================================================================

    def get_confidence_multiplier(self, as_of_date: datetime) -> float:
        """
        Return confidence multiplier (0.5-1.0) based on days since last earnings.
        Decays as time since last earnings increases.

        Args:
            as_of_date: Current date

        Returns:
            Multiplier from 0.5 to 1.0
        """
        days_since = self.days_since_earnings(as_of_date)

        if days_since is None:
            return 0.75  # Unknown, use moderate confidence

        # No decay for first 45 days (within quarter)
        if days_since <= self.staleness_grace_days:
            return 1.0

        # Linear decay from 1.0 to 0.5 over next ~55 days (to 100 days total)
        decay_days = days_since - self.staleness_grace_days
        decay = decay_days * self.staleness_decay_rate

        return max(0.5, 1.0 - decay)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _safe_get(self, key: str, default=0) -> float:
        """Get value from info dict, returning default if missing or None."""
        value = self._info.get(key)
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def get_summary(self, as_of_date: datetime = None, price: float = None) -> Dict:
        """Get a summary of all scores and key metrics."""
        return {
            'symbol': self.symbol,
            'quality_score': self.get_quality_score(as_of_date, price=price),
            'momentum_score': self.get_growth_momentum_score(as_of_date),
            'trending_probability': self.get_trending_probability(as_of_date, price=price),
            'has_earnings_dates': self._earnings_dates is not None and not self._earnings_dates.empty,
            'has_quarterly_income': self._quarterly_income is not None and not self._quarterly_income.empty,
            'has_earnings_history': self._earnings_history is not None and not self._earnings_history.empty,
            'quarters_mapped': len(self._quarter_report_map),
        }


class FundamentalCache:
    """
    Module-level cache for fundamental data across multiple strategy instances.
    Prevents redundant API calls during optimization/walk-forward.
    """
    _cache: Dict[str, FundamentalDataProvider] = {}

    @classmethod
    def get_provider(cls, symbol: str, lookback_years: int = 5,
                     verbose: bool = False) -> FundamentalDataProvider:
        """
        Get or create cached provider for symbol.

        Args:
            symbol: Ticker symbol
            lookback_years: Years of historical data
            verbose: Print debug info

        Returns:
            FundamentalDataProvider instance
        """
        cache_key = f"{symbol.upper()}_{lookback_years}"

        if cache_key not in cls._cache:
            cls._cache[cache_key] = FundamentalDataProvider(
                symbol, lookback_years, verbose
            )

        return cls._cache[cache_key]

    @classmethod
    def clear_cache(cls):
        """Clear all cached data."""
        cls._cache.clear()

    @classmethod
    def preload_symbols(cls, symbols: List[str], lookback_years: int = 5,
                        verbose: bool = False) -> Dict[str, FundamentalDataProvider]:
        """
        Pre-fetch data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            lookback_years: Years of historical data
            verbose: Print progress

        Returns:
            Dict of symbol -> provider
        """
        providers = {}
        total = len(symbols)

        for i, symbol in enumerate(symbols):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Loading fundamentals: {i + 1}/{total}")

            try:
                providers[symbol] = cls.get_provider(symbol, lookback_years, verbose=False)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not load {symbol}: {e}")

        return providers

    @classmethod
    def get_cache_stats(cls) -> Dict:
        """Get cache statistics."""
        return {
            'cached_symbols': len(cls._cache),
            'symbols': list(cls._cache.keys()),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def get_fundamental_scores(symbol: str, verbose: bool = False) -> Dict:
    """
    Quick function to get all fundamental scores for a symbol.

    Args:
        symbol: Ticker symbol
        verbose: Print debug info

    Returns:
        Dict with quality_score, momentum_score, trending_probability
    """
    provider = FundamentalCache.get_provider(symbol, verbose=verbose)
    return provider.get_summary()


def check_earnings_blackout(symbol: str, date: datetime,
                            days_before: int = 5, days_after: int = 2) -> bool:
    """
    Quick function to check if a date is in earnings blackout.

    Args:
        symbol: Ticker symbol
        date: Date to check
        days_before: Days before earnings to block
        days_after: Days after earnings to block

    Returns:
        True if in blackout, False otherwise
    """
    provider = FundamentalCache.get_provider(symbol)
    return provider.is_in_earnings_blackout(date, days_before, days_after)


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    # Test with a sample symbol
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    print(f"\n{'='*60}")
    print(f"Fundamental Analysis for {symbol}")
    print('='*60)

    provider = FundamentalDataProvider(symbol, verbose=True)

    print(f"\n--- Earnings Dates ---")
    earnings = provider.get_earnings_dates()
    if earnings is not None and not earnings.empty:
        print(earnings.head())
    else:
        print("No earnings dates available")

    print(f"\n--- Quarter-to-Report Map ---")
    for qe, rd in sorted(provider._quarter_report_map.items()):
        print(f"  Q ending {qe.strftime('%Y-%m-%d')} -> reported {rd.strftime('%Y-%m-%d')}")

    print(f"\n--- Earnings Blackout Check ---")
    from datetime import date
    today = date.today()
    in_blackout = provider.is_in_earnings_blackout(today)
    print(f"Today ({today}) in blackout: {in_blackout}")

    days_until = provider.days_until_earnings(today)
    days_since = provider.days_since_earnings(today)
    print(f"Days until next earnings: {days_until}")
    print(f"Days since last earnings: {days_since}")

    # Demonstrate point-in-time scoring
    print(f"\n--- Point-in-Time Quality Score (today) ---")
    quality = provider.get_quality_score(as_of_date=today, price=150.0)

    print(f"\n--- Point-in-Time Growth Momentum Score (today) ---")
    momentum = provider.get_growth_momentum_score(as_of_date=today)

    print(f"\n--- Combined Trending Probability (today) ---")
    trending = provider.get_trending_probability(as_of_date=today, price=150.0)

    # Show how scores would differ at different dates
    if provider._quarter_report_map:
        report_dates = sorted(provider._quarter_report_map.values())
        if len(report_dates) >= 2:
            print(f"\n--- Score Comparison Across Earnings Dates ---")
            for rd in report_dates[-3:]:
                q = provider.get_quality_score(as_of_date=rd, price=150.0)
                m = provider.get_growth_momentum_score(as_of_date=rd)
                print(f"  As of {rd.strftime('%Y-%m-%d')}: Quality={q:.1f}, Momentum={m:.1f}")

    print(f"\n--- Confidence Multiplier ---")
    confidence = provider.get_confidence_multiplier(today)
    print(f"Confidence (staleness): {confidence:.2f}")

    print(f"\n{'='*60}")
    print(f"SUMMARY: Quality={quality:.1f}, Momentum={momentum:.1f}, Trending={trending:.1f}")
    print('='*60)
