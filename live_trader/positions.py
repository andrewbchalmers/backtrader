# positions.py
"""
Shared position management module
Single source of truth for all position operations
"""

import json
import os
from datetime import datetime


class PositionManager:
    """Manages trading positions with persistence"""

    def __init__(self, filename="active_positions.json"):
        self.filename = filename
        self.positions = self._load()

    def _load(self):
        """Load positions from JSON file"""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    print(f"✓ Loaded {len(data)} position(s) from {self.filename}")
                    return data
            except json.JSONDecodeError:
                print(f"⚠️  Corrupted {self.filename}, starting fresh")
                return {}
            except Exception as e:
                print(f"❌ Error loading {self.filename}: {e}")
                return {}
        else:
            print(f"ℹ️  No existing {self.filename}, starting with empty positions")
            return {}

    def _save(self):
        """Save positions to JSON file"""
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.positions, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ Error saving positions to {self.filename}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def add(self, symbol, entry_price, stop_loss, entry_date=None):
        """
        Add a new position

        Args:
            symbol: Stock ticker
            entry_price: Entry price
            stop_loss: Initial stop loss price
            entry_date: Entry date (defaults to now)

        Returns:
            dict: The position data if successful, None otherwise
        """
        if symbol in self.positions:
            print(f"⚠️  Position already exists for {symbol}")
            return None

        if entry_date is None:
            entry_date = datetime.now().isoformat()

        self.positions[symbol] = {
            'entry_price': float(entry_price),
            'entry_date': entry_date,
            'stop_loss': float(stop_loss)
        }

        if self._save():
            print(f"✓ Position added: {symbol} @ ${entry_price:.2f}, SL @ ${stop_loss:.2f}")
            return self.positions[symbol]
        return None

    def remove(self, symbol):
        """
        Remove a position

        Args:
            symbol: Stock ticker

        Returns:
            dict: The removed position data if successful, None otherwise
        """
        if symbol not in self.positions:
            print(f"⚠️  No position found for {symbol}")
            return None

        position = self.positions[symbol]
        del self.positions[symbol]

        if self._save():
            print(f"✓ Position removed: {symbol}")
            return position
        return None

    def update_stop(self, symbol, new_stop):
        """
        Update stop loss for a position

        Args:
            symbol: Stock ticker
            new_stop: New stop loss price

        Returns:
            bool: True if successful
        """
        if symbol not in self.positions:
            return False

        old_stop = self.positions[symbol]['stop_loss']
        self.positions[symbol]['stop_loss'] = float(new_stop)

        if self._save():
            print(f"✓ Updated {symbol} stop: ${old_stop:.2f} → ${new_stop:.2f}")
            return True
        return False

    def get(self, symbol):
        """Get position data for a symbol"""
        return self.positions.get(symbol)

    def has_position(self, symbol):
        """Check if we have a position in this symbol"""
        return symbol in self.positions

    def list_all(self):
        """Get all positions"""
        return self.positions.copy()

    def clear_all(self):
        """Remove all positions"""
        count = len(self.positions)
        self.positions = {}
        if self._save():
            print(f"✓ Cleared {count} positions")
            return count
        return 0

    def get_summary(self):
        """Get summary statistics"""
        if not self.positions:
            return {
                'count': 0,
                'total_risk_pct': 0,
                'avg_risk_pct': 0
            }

        total_risk = 0
        for symbol, data in self.positions.items():
            entry = data['entry_price']
            stop = data['stop_loss']
            risk_pct = ((entry - stop) / entry) * 100
            total_risk += risk_pct

        return {
            'count': len(self.positions),
            'total_risk_pct': total_risk,
            'avg_risk_pct': total_risk / len(self.positions)
        }
