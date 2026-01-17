#!/usr/bin/env python3
"""
Level 1 Time Analysis Tool

Calculates the percentage of time your orders are at Level 1 (best bid/best ask)
by comparing order chains against public orderbook snapshots.

Usage:
    python calculate_level1_time.py YYYY-MM-DD PRODUCT

Example:
    python calculate_level1_time.py 2026-01-06 ADA-TL
"""

import pandas as pd
import numpy as np
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

from results_path import get_results_dir, get_shared_dir


# =============================================================================
# Data Loading
# =============================================================================

def load_markets_map() -> pd.DataFrame:
    """Load the markets map CSV that maps products to exchange/market IDs."""
    # Try shared directory first (new location)
    shared_dir = get_shared_dir()
    markets_map_file = shared_dir / "markets_map.csv"
    
    # Fall back to old location if not in shared
    if not markets_map_file.exists():
        markets_map_file = Path(__file__).parent / "results" / "markets_map.csv"
    
    if not markets_map_file.exists():
        print(f"Error: markets_map.csv not found at {markets_map_file}")
        sys.exit(1)
    
    return pd.read_csv(markets_map_file)


def get_paribu_ids(markets_map: pd.DataFrame, base_currency: str) -> tuple:
    """Get Paribu exchange_id and market_id for a given base currency."""
    row = markets_map[markets_map['base_currency'] == base_currency]
    if row.empty:
        return None, None
    return int(row['paribu_exchange_id'].iloc[0]), int(row['paribu_market_id'].iloc[0])


def load_orderbook(date: str, product: str) -> pd.DataFrame:
    """Load consolidated orderbook data from CSV file."""
    results_dir = get_results_dir(date, product)
    
    # Try new filename first
    file_path = results_dir / "orderbook_consolidated.csv"
    
    # Fall back to old filename if new doesn't exist
    if not file_path.exists():
        old_results_dir = Path(__file__).parent / "results"
        file_path = old_results_dir / f"Public_{date}_{product}_orderbook_consolidated.csv"
    
    if not file_path.exists():
        print(f"Error: Orderbook file not found: {file_path}")
        sys.exit(1)
    
    df = pd.read_csv(file_path)
    df['datetime_ingest'] = pd.to_datetime(df['datetime_ingest'])
    
    return df


def load_chains(date: str, product: str, side: str) -> pd.DataFrame:
    """Load order chains from CSV file for a specific side (buy/sell)."""
    results_dir = get_results_dir(date, product)
    
    # Try new filename first
    file_path = results_dir / f"chains_{side}.csv"
    
    # Fall back to old filename if new doesn't exist (need exchange/market IDs)
    if not file_path.exists():
        # Can't use old filename without exchange/market IDs - return empty
        print(f"Warning: Chain file not found: {file_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    df['datetime_open'] = pd.to_datetime(df['datetime_open'])
    df['datetime_close'] = pd.to_datetime(df['datetime_close'])
    
    return df


def load_trades(date: str, product: str) -> pd.DataFrame:
    """Load public trades data from CSV file.
    
    Args:
        date: Date string in YYYY-MM-DD format
        product: Product symbol (e.g., ADA-TL)
    
    Returns:
        DataFrame with trades data, including:
        - datetime_exchange: Trade timestamp
        - side: 1=buy (taker bought from ask), 2=sell (taker sold to bid)
        - price: Trade price
        - size: Trade size
    """
    results_dir = get_results_dir(date, product)
    
    # Try new filename first
    file_path = results_dir / "trades.csv"
    
    # Fall back to old filename if new doesn't exist
    if not file_path.exists():
        old_results_dir = Path(__file__).parent / "results"
        file_path = old_results_dir / f"Public_{date}_{product}_trades.csv"
    
    if not file_path.exists():
        print(f"Warning: Trades file not found: {file_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    df['datetime_exchange'] = pd.to_datetime(df['datetime_exchange'])
    
    return df


# =============================================================================
# Level 1 Analysis
# =============================================================================

def analyze_level1_time(orderbook: pd.DataFrame, chains: pd.DataFrame, 
                        price_column: str, qty_column: str, side_name: str) -> dict:
    """
    Analyze what fraction of time orders are at Level 1.
    
    For each orderbook snapshot, check if any chain is active (snapshot time 
    between datetime_open and datetime_close) and if so, whether the chain's
    price AND quantity match the Level 1 price and quantity.
    
    Args:
        orderbook: DataFrame with orderbook snapshots
        chains: DataFrame with order chains
        price_column: Column name for Level 1 price ('best_bid_price_1' or 'best_ask_price_1')
        qty_column: Column name for Level 1 quantity ('best_bid_qty_1' or 'best_ask_qty_1')
        side_name: 'Buy' or 'Sell' for display
    
    Returns:
        Dict with analysis results
    """
    if chains.empty:
        return {
            'side': side_name,
            'total_snapshots': len(orderbook),
            'snapshots_with_active_orders': 0,
            'snapshots_at_level1': 0,
            'level1_percentage': 0.0,
            'active_time_seconds': 0.0,
            'level1_time_seconds': 0.0,
        }
    
    # Convert chains to numpy arrays for efficient lookup
    chain_opens = chains['datetime_open'].values.astype('datetime64[ns]')
    chain_closes = chains['datetime_close'].values.astype('datetime64[ns]')
    chain_prices = chains['price'].values
    chain_initial_qtys = chains['initial_qty'].values
    
    # Convert orderbook timestamps and data
    ob_times = orderbook['datetime_ingest'].values.astype('datetime64[ns]')
    ob_level1_prices = orderbook[price_column].values
    ob_level1_qtys = orderbook[qty_column].values
    
    # Calculate time deltas between snapshots for time-weighted analysis
    # Use the time until the next snapshot as the duration for each snapshot
    time_deltas = np.zeros(len(ob_times))
    for i in range(len(ob_times) - 1):
        delta = (ob_times[i + 1] - ob_times[i]) / np.timedelta64(1, 's')
        # Cap at 60 seconds to avoid counting long gaps
        time_deltas[i] = min(delta, 60.0)
    # Last snapshot gets average delta
    if len(time_deltas) > 1:
        time_deltas[-1] = np.mean(time_deltas[:-1])
    
    # Track statistics
    snapshots_with_active = 0
    snapshots_at_level1 = 0
    active_time = 0.0
    level1_time = 0.0
    
    # For each orderbook snapshot, check if any chain is active
    for i, ob_time in enumerate(ob_times):
        # Find chains that are active at this snapshot time
        # Active means: datetime_open <= ob_time <= datetime_close
        active_mask = (chain_opens <= ob_time) & (ob_time <= chain_closes)
        
        if np.any(active_mask):
            snapshots_with_active += 1
            active_time += time_deltas[i]
            
            # Get the price and qty of active chains
            active_prices = chain_prices[active_mask]
            active_qtys = chain_initial_qtys[active_mask]
            
            # Check if any active chain matches Level 1 (both price AND quantity)
            level1_price = ob_level1_prices[i]
            level1_qty = ob_level1_qtys[i]
            
            if not np.isnan(level1_price) and not np.isnan(level1_qty):
                # Check if any active chain matches BOTH price and quantity
                price_matches = np.isclose(active_prices, level1_price, rtol=1e-6)
                qty_matches = np.isclose(active_qtys, level1_qty, rtol=1e-6)
                
                if np.any(price_matches & qty_matches):
                    snapshots_at_level1 += 1
                    level1_time += time_deltas[i]
    
    # Calculate total time (sum of all time deltas)
    total_time = np.sum(time_deltas)
    
    # Level 1 as percentage of active time
    level1_pct_of_active = (snapshots_at_level1 / snapshots_with_active * 100) if snapshots_with_active > 0 else 0.0
    
    # Level 1 as percentage of total day
    level1_pct_of_total = (snapshots_at_level1 / len(orderbook) * 100) if len(orderbook) > 0 else 0.0
    
    return {
        'side': side_name,
        'total_snapshots': len(orderbook),
        'snapshots_with_active_orders': snapshots_with_active,
        'snapshots_at_level1': snapshots_at_level1,
        'level1_percentage': level1_pct_of_active,
        'level1_pct_of_total': level1_pct_of_total,
        'active_time_seconds': active_time,
        'level1_time_seconds': level1_time,
        'total_time_seconds': total_time,
    }


def analyze_trade_attribution(trades: pd.DataFrame, orderbook: pd.DataFrame,
                               buy_chains: pd.DataFrame, sell_chains: pd.DataFrame) -> dict:
    """
    Analyze trade attribution: for each public trade, determine if we were at Level 1.
    
    Key logic:
    - Public buy (side=1): Taker bought from ask -> check if we're at Level 1 on SELL/ASK side
    - Public sell (side=2): Taker sold to bid -> check if we're at Level 1 on BUY/BID side
    
    Args:
        trades: DataFrame with public trades
        orderbook: DataFrame with orderbook snapshots
        buy_chains: DataFrame with our buy order chains
        sell_chains: DataFrame with our sell order chains
    
    Returns:
        Dict with attribution statistics and per-trade details
    """
    if trades.empty:
        return {
            'total_trades': 0,
            'buy_trades': 0,
            'sell_trades': 0,
            'buy_ours': 0,
            'buy_not_ours': 0,
            'buy_no_position': 0,
            'sell_ours': 0,
            'sell_not_ours': 0,
            'sell_no_position': 0,
            'trade_details': pd.DataFrame(),
        }
    
    # Prepare chain data for efficient lookup
    # For sell chains (matched against public buys)
    if not sell_chains.empty:
        sell_chain_opens = sell_chains['datetime_open'].values.astype('datetime64[ns]')
        sell_chain_closes = sell_chains['datetime_close'].values.astype('datetime64[ns]')
        sell_chain_prices = sell_chains['price'].values
        sell_chain_qtys = sell_chains['initial_qty'].values
    else:
        sell_chain_opens = np.array([], dtype='datetime64[ns]')
        sell_chain_closes = np.array([], dtype='datetime64[ns]')
        sell_chain_prices = np.array([])
        sell_chain_qtys = np.array([])
    
    # For buy chains (matched against public sells)
    if not buy_chains.empty:
        buy_chain_opens = buy_chains['datetime_open'].values.astype('datetime64[ns]')
        buy_chain_closes = buy_chains['datetime_close'].values.astype('datetime64[ns]')
        buy_chain_prices = buy_chains['price'].values
        buy_chain_qtys = buy_chains['initial_qty'].values
    else:
        buy_chain_opens = np.array([], dtype='datetime64[ns]')
        buy_chain_closes = np.array([], dtype='datetime64[ns]')
        buy_chain_prices = np.array([])
        buy_chain_qtys = np.array([])
    
    # Prepare orderbook data
    ob_times = orderbook['datetime_ingest'].values.astype('datetime64[ns]')
    ob_bid_prices = orderbook['best_bid_price_1'].values
    ob_bid_qtys = orderbook['best_bid_qty_1'].values
    ob_ask_prices = orderbook['best_ask_price_1'].values
    ob_ask_qtys = orderbook['best_ask_qty_1'].values
    
    # Results storage
    trade_attributions = []
    
    # Counters
    buy_ours = 0
    buy_not_ours = 0
    buy_no_position = 0
    sell_ours = 0
    sell_not_ours = 0
    sell_no_position = 0
    
    for _, trade in trades.iterrows():
        trade_time = trade['datetime_exchange']
        trade_time_ns = np.datetime64(trade_time, 'ns')
        trade_side = trade['side']  # 1=buy, 2=sell
        trade_price = trade['price']
        trade_size = trade['size']
        
        # Find the most recent orderbook snapshot before or at trade time
        ob_mask = ob_times <= trade_time_ns
        if not np.any(ob_mask):
            # No orderbook snapshot before this trade
            attribution = 'no_data'
            if trade_side == 1:
                buy_no_position += 1
            else:
                sell_no_position += 1
        else:
            ob_idx = np.where(ob_mask)[0][-1]  # Last True index
            
            if trade_side == 1:
                # Public buy: taker bought from ask -> check if OUR SELL is at Level 1
                # Our sell order would be on the ask side
                level1_price = ob_ask_prices[ob_idx]
                level1_qty = ob_ask_qtys[ob_idx]
                chain_opens = sell_chain_opens
                chain_closes = sell_chain_closes
                chain_prices = sell_chain_prices
                chain_qtys = sell_chain_qtys
            else:
                # Public sell: taker sold to bid -> check if OUR BUY is at Level 1
                # Our buy order would be on the bid side
                level1_price = ob_bid_prices[ob_idx]
                level1_qty = ob_bid_qtys[ob_idx]
                chain_opens = buy_chain_opens
                chain_closes = buy_chain_closes
                chain_prices = buy_chain_prices
                chain_qtys = buy_chain_qtys
            
            # Check if any of our chains were active at trade time
            if len(chain_opens) == 0:
                attribution = 'no_position'
                if trade_side == 1:
                    buy_no_position += 1
                else:
                    sell_no_position += 1
            else:
                active_mask = (chain_opens <= trade_time_ns) & (trade_time_ns <= chain_closes)
                
                if not np.any(active_mask):
                    attribution = 'no_position'
                    if trade_side == 1:
                        buy_no_position += 1
                    else:
                        sell_no_position += 1
                else:
                    # We have active orders - check if at Level 1
                    active_prices = chain_prices[active_mask]
                    active_qtys = chain_qtys[active_mask]
                    
                    if np.isnan(level1_price) or np.isnan(level1_qty):
                        attribution = 'no_data'
                        if trade_side == 1:
                            buy_no_position += 1
                        else:
                            sell_no_position += 1
                    else:
                        # Check if any active chain matches Level 1 (both price AND quantity)
                        price_matches = np.isclose(active_prices, level1_price, rtol=1e-6)
                        qty_matches = np.isclose(active_qtys, level1_qty, rtol=1e-6)
                        
                        if np.any(price_matches & qty_matches):
                            attribution = 'ours'
                            if trade_side == 1:
                                buy_ours += 1
                            else:
                                sell_ours += 1
                        else:
                            attribution = 'not_ours'
                            if trade_side == 1:
                                buy_not_ours += 1
                            else:
                                sell_not_ours += 1
        
        trade_attributions.append({
            'datetime': trade_time,
            'trade_id': trade.get('trade_id', ''),
            'public_side': 'buy' if trade_side == 1 else 'sell',
            'our_side': 'sell' if trade_side == 1 else 'buy',  # Inverse
            'price': trade_price,
            'size': trade_size,
            'attribution': attribution,
        })
    
    # Create details DataFrame
    trade_details = pd.DataFrame(trade_attributions)
    
    # Count totals
    buy_trades = len(trades[trades['side'] == 1])
    sell_trades = len(trades[trades['side'] == 2])
    
    return {
        'total_trades': len(trades),
        'buy_trades': buy_trades,
        'sell_trades': sell_trades,
        'buy_ours': buy_ours,
        'buy_not_ours': buy_not_ours,
        'buy_no_position': buy_no_position,
        'sell_ours': sell_ours,
        'sell_not_ours': sell_not_ours,
        'sell_no_position': sell_no_position,
        'trade_details': trade_details,
    }


# Price tolerance for fake trade detection: 2.5 basis points (0.025%)
FAKE_TRADE_PRICE_TOLERANCE_BPS = 2.5


def detect_fake_trades(trades: pd.DataFrame, orderbook: pd.DataFrame,
                       time_tolerance_seconds: float = 1.0) -> pd.DataFrame:
    """
    Detect fake trades by checking if trade prices fall inside the bid-ask spread.
    
    A trade is classified as FAKE if:
    1. The trade price is inside the spread (between best bid and best ask)
    2. The price is more than 5 bps (0.05%) away from BOTH the best bid and best ask
    3. This condition holds for ALL orderbook snapshots within the time tolerance window
    
    NOTE: We check ALL snapshots within the time window (both before and after the trade)
    and only classify as fake if the trade appears fake in ALL of them. This conservative
    approach accounts for timing mismatches between trade and orderbook data feeds.
    
    Args:
        trades: DataFrame with public trades (must have datetime_exchange, price columns)
        orderbook: DataFrame with orderbook snapshots (must have datetime_ingest, 
                   best_bid_price_1, best_ask_price_1 columns)
        time_tolerance_seconds: Time window in seconds to check orderbook snapshots
    
    Returns:
        DataFrame with trades and added 'is_fake' boolean column
    """
    if trades.empty:
        return trades.copy()
    
    # Prepare orderbook data for efficient lookup
    ob_times = orderbook['datetime_ingest'].values.astype('datetime64[ns]')
    ob_bid_prices = orderbook['best_bid_price_1'].values
    ob_ask_prices = orderbook['best_ask_price_1'].values
    
    # Convert time tolerance to nanoseconds for numpy comparison
    time_tolerance_ns = np.timedelta64(int(time_tolerance_seconds * 1e9), 'ns')
    
    # Price tolerance as a ratio (5 bps = 0.0005)
    price_tolerance_ratio = FAKE_TRADE_PRICE_TOLERANCE_BPS / 10000.0
    
    fake_flags = []
    
    for _, trade in trades.iterrows():
        trade_time = trade['datetime_exchange']
        trade_time_ns = np.datetime64(trade_time, 'ns')
        trade_price = trade['price']
        
        # Find all orderbook snapshots within the time tolerance window
        # (both before and after the trade time)
        time_diff = np.abs(ob_times - trade_time_ns)
        window_mask = time_diff <= time_tolerance_ns
        
        if not np.any(window_mask):
            # No orderbook data within window - can't determine, assume real
            fake_flags.append(False)
            continue
        
        # Get bid/ask prices for all snapshots in the window
        window_bids = ob_bid_prices[window_mask]
        window_asks = ob_ask_prices[window_mask]
        
        # Check each snapshot in the window
        # Trade is fake only if it appears fake in ALL snapshots
        is_fake_in_all = True
        
        for bid, ask in zip(window_bids, window_asks):
            if np.isnan(bid) or np.isnan(ask):
                # Missing data - can't determine, assume real for this snapshot
                is_fake_in_all = False
                break
            
            # Check if trade price is inside the spread
            if trade_price <= bid or trade_price >= ask:
                # Price is at or outside the spread - real trade
                is_fake_in_all = False
                break
            
            # Price is inside spread - check if within tolerance of either boundary
            # Calculate tolerance thresholds
            bid_tolerance = bid * price_tolerance_ratio
            ask_tolerance = ask * price_tolerance_ratio
            
            # Check if within tolerance of bid or ask
            within_bid_tolerance = (trade_price - bid) <= bid_tolerance
            within_ask_tolerance = (ask - trade_price) <= ask_tolerance
            
            if within_bid_tolerance or within_ask_tolerance:
                # Within tolerance of a boundary - real trade
                is_fake_in_all = False
                break
        
        fake_flags.append(is_fake_in_all)
    
    # Add is_fake column to a copy of trades
    result = trades.copy()
    result['is_fake'] = fake_flags
    
    return result


def get_fake_trade_stats(trades_with_fake: pd.DataFrame) -> dict:
    """
    Calculate fake trade statistics from trades DataFrame with is_fake column.
    
    Args:
        trades_with_fake: DataFrame with trades including 'is_fake' and 'side' columns
    
    Returns:
        Dict with fake trade statistics
    """
    if trades_with_fake.empty or 'is_fake' not in trades_with_fake.columns:
        return {
            'total_trades': 0,
            'fake_trades': 0,
            'real_trades': 0,
            'fake_pct': 0.0,
            'fake_buys': 0,
            'fake_sells': 0,
            'real_buys': 0,
            'real_sells': 0,
        }
    
    total = len(trades_with_fake)
    fake_mask = trades_with_fake['is_fake']
    fake_count = fake_mask.sum()
    real_count = total - fake_count
    
    # Breakdown by side
    buy_mask = trades_with_fake['side'] == 1
    sell_mask = trades_with_fake['side'] == 2
    
    fake_buys = (fake_mask & buy_mask).sum()
    fake_sells = (fake_mask & sell_mask).sum()
    real_buys = (~fake_mask & buy_mask).sum()
    real_sells = (~fake_mask & sell_mask).sum()
    
    return {
        'total_trades': total,
        'fake_trades': fake_count,
        'real_trades': real_count,
        'fake_pct': (fake_count / total * 100) if total > 0 else 0.0,
        'fake_buys': fake_buys,
        'fake_sells': fake_sells,
        'real_buys': real_buys,
        'real_sells': real_sells,
    }


# =============================================================================
# Output Formatting
# =============================================================================

def format_duration(seconds: float) -> str:
    """Format seconds as hours and minutes."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def generate_report(date: str, product: str, buy_stats: dict, sell_stats: dict,
                    buy_chains: pd.DataFrame, sell_chains: pd.DataFrame,
                    trade_attribution: dict = None, fake_stats: dict = None,
                    trade_attribution_excl_fake: dict = None) -> str:
    """Generate markdown report content with explanatory text."""
    
    # Calculate active percentages
    buy_active_pct = buy_stats['snapshots_with_active_orders'] / buy_stats['total_snapshots'] * 100
    sell_active_pct = sell_stats['snapshots_with_active_orders'] / sell_stats['total_snapshots'] * 100
    
    report = f"""# Level 1 Analysis: {date} {product}

This report analyzes order positioning and trade attribution for the trading day.

## Order Positioning

This section shows how much of the day our orders were active and at Level 1 (the best bid or best ask price in the order book).

### Buy Side:
Active orders {buy_active_pct:.1f}% of the day ({format_duration(buy_stats['active_time_seconds'])})
At Level 1: {buy_stats['level1_percentage']:.1f}% of active time / {buy_stats['level1_pct_of_total']:.1f}% of total day

### Sell Side:
Active orders {sell_active_pct:.1f}% of the day ({format_duration(sell_stats['active_time_seconds'])})
At Level 1: {sell_stats['level1_percentage']:.1f}% of active time / {sell_stats['level1_pct_of_total']:.1f}% of total day
"""
    
    # Add trade attribution section if available
    if trade_attribution and trade_attribution['total_trades'] > 0:
        ta = trade_attribution
        
        # Calculate percentages for our fills
        # Public buys hit our sells, public sells hit our buys
        our_sells_pct = (ta['buy_ours'] / ta['buy_trades'] * 100) if ta['buy_trades'] > 0 else 0
        our_buys_pct = (ta['sell_ours'] / ta['sell_trades'] * 100) if ta['sell_trades'] > 0 else 0
        
        report += f"""
## Trade Attribution

When a public trade occurs, we determine whether it was likely our order that got filled.
A "public buy" executes against the ask side (potentially our sell order), while a "public sell" executes against the bid side (potentially our buy order).

Total public trades: {ta['total_trades']}

### Our Sells (hit by public buys):
- Public buys: {ta['buy_trades']}
- Ours (at L1): {ta['buy_ours']} ({our_sells_pct:.1f}%) — we were at best ask when the buy occurred
- Not ours: {ta['buy_not_ours']} — we had a sell order active but not at best ask
- No position: {ta['buy_no_position']} — we had no sell order active

### Our Buys (hit by public sells):
- Public sells: {ta['sell_trades']}
- Ours (at L1): {ta['sell_ours']} ({our_buys_pct:.1f}%) — we were at best bid when the sell occurred
- Not ours: {ta['sell_not_ours']} — we had a buy order active but not at best bid
- No position: {ta['sell_no_position']} — we had no buy order active
"""
    
    # Add fake trade detection section if available
    if fake_stats and fake_stats['total_trades'] > 0:
        fs = fake_stats
        report += f"""
## Fake Trade Detection

Trades are flagged as potentially fake if the reported price falls inside the bid-ask spread and is more than {FAKE_TRADE_PRICE_TOLERANCE_BPS} basis points away from both the best bid and best ask.
These trades could not have occurred through normal order book matching.

- Total trades: {fs['total_trades']}
- Fake trades: {fs['fake_trades']} ({fs['fake_pct']:.1f}%)
- Real trades: {fs['real_trades']}

### Breakdown by Side:
- Fake buys: {fs['fake_buys']}
- Fake sells: {fs['fake_sells']}
- Real buys: {fs['real_buys']}
- Real sells: {fs['real_sells']}
"""
        
        # Add attribution excluding fake trades
        if trade_attribution_excl_fake and trade_attribution_excl_fake['total_trades'] > 0:
            ta_excl = trade_attribution_excl_fake
            our_sells_pct_excl = (ta_excl['buy_ours'] / ta_excl['buy_trades'] * 100) if ta_excl['buy_trades'] > 0 else 0
            our_buys_pct_excl = (ta_excl['sell_ours'] / ta_excl['sell_trades'] * 100) if ta_excl['sell_trades'] > 0 else 0
            
            report += f"""
### Trade Attribution (Excluding Fake Trades)

Attribution recalculated using only trades that passed the fake trade filter.

Total real trades: {ta_excl['total_trades']}

#### Our Sells (hit by real public buys):
- Real public buys: {ta_excl['buy_trades']}
- Ours (at L1): {ta_excl['buy_ours']} ({our_sells_pct_excl:.1f}%)
- Not ours: {ta_excl['buy_not_ours']}
- No position: {ta_excl['buy_no_position']}

#### Our Buys (hit by real public sells):
- Real public sells: {ta_excl['sell_trades']}
- Ours (at L1): {ta_excl['sell_ours']} ({our_buys_pct_excl:.1f}%)
- Not ours: {ta_excl['sell_not_ours']}
- No position: {ta_excl['sell_no_position']}
"""
    
    return report


def print_report(date: str, product: str, buy_stats: dict, sell_stats: dict,
                 buy_chains: pd.DataFrame, sell_chains: pd.DataFrame,
                 trade_attribution: dict = None, fake_stats: dict = None,
                 trade_attribution_excl_fake: dict = None) -> None:
    """Print the analysis report to console."""
    print()
    print("=" * 60)
    print(f"Level 1 Analysis: {date} {product}")
    print("=" * 60)
    
    for stats, chains in [(buy_stats, buy_chains), (sell_stats, sell_chains)]:
        print(f"\n{stats['side']} Side:")
        print(f"  Total orderbook snapshots: {stats['total_snapshots']:,}")
        print(f"  Snapshots with active orders: {stats['snapshots_with_active_orders']:,} "
              f"({stats['snapshots_with_active_orders']/stats['total_snapshots']*100:.1f}% of day)")
        
        if stats['snapshots_with_active_orders'] > 0:
            # Level 1 as % of active time
            print(f"  Snapshots at Level 1: {stats['snapshots_at_level1']:,}")
            print(f"    - Of active time: {stats['level1_percentage']:.1f}%")
            print(f"    - Of total day:   {stats['level1_pct_of_total']:.1f}%")
            print(f"  Time at Level 1: {format_duration(stats['level1_time_seconds'])} "
                  f"(of {format_duration(stats['active_time_seconds'])} active, "
                  f"{format_duration(stats['total_time_seconds'])} total)")
            
            # Show unique order sizes used
            if not chains.empty:
                unique_qtys = sorted(chains['initial_qty'].unique())
                print(f"  Order sizes used: {', '.join(f'{q:.1f}' for q in unique_qtys)}")
        else:
            print(f"  No active orders found")
    
    # Print trade attribution if available
    if trade_attribution and trade_attribution['total_trades'] > 0:
        ta = trade_attribution
        print()
        print("=" * 60)
        print("Trade Attribution")
        print("=" * 60)
        print(f"\nTotal public trades: {ta['total_trades']:,}")
        
        # Our Sells (hit by public buys)
        print(f"\nOur Sells (hit by public buys):")
        print(f"  Public buys: {ta['buy_trades']:,}")
        our_sells_pct = (ta['buy_ours'] / ta['buy_trades'] * 100) if ta['buy_trades'] > 0 else 0
        print(f"  Ours (at L1): {ta['buy_ours']:,} ({our_sells_pct:.1f}%)")
        print(f"  Not ours: {ta['buy_not_ours']:,}")
        print(f"  No position: {ta['buy_no_position']:,}")
        
        # Our Buys (hit by public sells)
        print(f"\nOur Buys (hit by public sells):")
        print(f"  Public sells: {ta['sell_trades']:,}")
        our_buys_pct = (ta['sell_ours'] / ta['sell_trades'] * 100) if ta['sell_trades'] > 0 else 0
        print(f"  Ours (at L1): {ta['sell_ours']:,} ({our_buys_pct:.1f}%)")
        print(f"  Not ours: {ta['sell_not_ours']:,}")
        print(f"  No position: {ta['sell_no_position']:,}")
    
    # Print fake trade detection if available
    if fake_stats and fake_stats['total_trades'] > 0:
        fs = fake_stats
        print()
        print("=" * 60)
        print("Fake Trade Detection")
        print("=" * 60)
        print(f"\nTotal trades: {fs['total_trades']:,}")
        print(f"Fake trades: {fs['fake_trades']:,} ({fs['fake_pct']:.1f}%)")
        print(f"Real trades: {fs['real_trades']:,}")
        
        print(f"\nBreakdown by side:")
        print(f"  Fake buys: {fs['fake_buys']:,}")
        print(f"  Fake sells: {fs['fake_sells']:,}")
        print(f"  Real buys: {fs['real_buys']:,}")
        print(f"  Real sells: {fs['real_sells']:,}")
        
        # Print attribution excluding fake trades
        if trade_attribution_excl_fake and trade_attribution_excl_fake['total_trades'] > 0:
            ta_excl = trade_attribution_excl_fake
            print()
            print("-" * 60)
            print("Trade Attribution (Excluding Fake Trades)")
            print("-" * 60)
            print(f"\nTotal real trades: {ta_excl['total_trades']:,}")
            
            # Our Sells (hit by real public buys)
            print(f"\nOur Sells (hit by real public buys):")
            print(f"  Real public buys: {ta_excl['buy_trades']:,}")
            our_sells_pct_excl = (ta_excl['buy_ours'] / ta_excl['buy_trades'] * 100) if ta_excl['buy_trades'] > 0 else 0
            print(f"  Ours (at L1): {ta_excl['buy_ours']:,} ({our_sells_pct_excl:.1f}%)")
            print(f"  Not ours: {ta_excl['buy_not_ours']:,}")
            print(f"  No position: {ta_excl['buy_no_position']:,}")
            
            # Our Buys (hit by real public sells)
            print(f"\nOur Buys (hit by real public sells):")
            print(f"  Real public sells: {ta_excl['sell_trades']:,}")
            our_buys_pct_excl = (ta_excl['sell_ours'] / ta_excl['sell_trades'] * 100) if ta_excl['sell_trades'] > 0 else 0
            print(f"  Ours (at L1): {ta_excl['sell_ours']:,} ({our_buys_pct_excl:.1f}%)")
            print(f"  Not ours: {ta_excl['sell_not_ours']:,}")
            print(f"  No position: {ta_excl['sell_no_position']:,}")
    
    print()
    print("=" * 60)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Calculate percentage of time orders are at Level 1 (best bid/ask)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python calculate_level1_time.py 2026-01-06 ADA-TL
    python calculate_level1_time.py 2026-01-06 BTC-TL
        """
    )
    parser.add_argument('date', type=str, help='Date in YYYY-MM-DD format')
    parser.add_argument('product', type=str, help='Product symbol (e.g., ADA-TL)')
    parser.add_argument('--time-tolerance', type=float, default=0.5,
                        help='Time tolerance in seconds for fake trade detection (default: 0.5)')
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        print(f"Error: Invalid date format '{args.date}'. Use YYYY-MM-DD.")
        sys.exit(1)
    
    # Extract base currency from product (e.g., ADA from ADA-TL)
    if '-' in args.product:
        base_currency = args.product.split('-')[0]
    else:
        base_currency = args.product
    
    # Get output directory using new folder structure
    results_dir = get_results_dir(args.date, args.product)
    
    print("=" * 60)
    print("Level 1 Time Analysis")
    print("=" * 60)
    print(f"Date: {args.date}")
    print(f"Product: {args.product}")
    print(f"Base Currency: {base_currency}")
    print(f"Output Directory: {results_dir}")
    
    # Load markets map to get exchange/market IDs
    print("\nLoading markets map...")
    markets_map = load_markets_map()
    
    exchange_id, market_id = get_paribu_ids(markets_map, base_currency)
    if exchange_id is None:
        print(f"Error: Could not find Paribu IDs for {base_currency}")
        print(f"Available currencies: {', '.join(markets_map['base_currency'].tolist())}")
        sys.exit(1)
    
    print(f"Paribu Exchange ID: {exchange_id}, Market ID: {market_id}")
    
    # Load orderbook data
    print("\nLoading orderbook data...")
    orderbook = load_orderbook(args.date, args.product)
    print(f"  Loaded {len(orderbook):,} orderbook snapshots")
    
    # Load order chains
    print("\nLoading order chains...")
    buy_chains = load_chains(args.date, args.product, 'buy')
    sell_chains = load_chains(args.date, args.product, 'sell')
    print(f"  Buy chains: {len(buy_chains):,}")
    print(f"  Sell chains: {len(sell_chains):,}")
    
    # Load public trades
    print("\nLoading public trades...")
    trades = load_trades(args.date, args.product)
    print(f"  Loaded {len(trades):,} trades")
    
    # Detect fake trades
    print(f"\nDetecting fake trades (time tolerance: {args.time_tolerance}s)...")
    trades_with_fake = detect_fake_trades(trades, orderbook, args.time_tolerance)
    fake_stats = get_fake_trade_stats(trades_with_fake)
    print(f"  Fake trades detected: {fake_stats['fake_trades']:,} ({fake_stats['fake_pct']:.1f}%)")
    
    # Analyze Level 1 time
    print("\nAnalyzing Level 1 time...")
    buy_stats = analyze_level1_time(orderbook, buy_chains, 'best_bid_price_1', 'best_bid_qty_1', 'Buy')
    sell_stats = analyze_level1_time(orderbook, sell_chains, 'best_ask_price_1', 'best_ask_qty_1', 'Sell')
    
    # Analyze trade attribution (all trades)
    print("\nAnalyzing trade attribution...")
    trade_attribution = analyze_trade_attribution(trades, orderbook, buy_chains, sell_chains)
    
    # Analyze trade attribution (excluding fake trades)
    real_trades = trades_with_fake[~trades_with_fake['is_fake']]
    trade_attribution_excl_fake = analyze_trade_attribution(real_trades, orderbook, buy_chains, sell_chains)
    
    # Merge is_fake flag into trade_attribution details
    if not trade_attribution['trade_details'].empty and not trades_with_fake.empty:
        # Create a mapping from trade_id to is_fake
        fake_map = trades_with_fake.set_index('trade_id')['is_fake'].to_dict()
        trade_attribution['trade_details']['is_fake'] = trade_attribution['trade_details']['trade_id'].map(fake_map)
    
    # Print report to console
    print_report(args.date, args.product, buy_stats, sell_stats, buy_chains, sell_chains, 
                 trade_attribution, fake_stats, trade_attribution_excl_fake)
    
    # Generate and write markdown report (simplified filename)
    report_content = generate_report(args.date, args.product, buy_stats, sell_stats, buy_chains, sell_chains, 
                                     trade_attribution, fake_stats, trade_attribution_excl_fake)
    report_file = results_dir / "level1_analysis.md"
    with open(report_file, 'w') as f:
        f.write(report_content)
    print(f"Report saved to: {report_file}")
    
    # Save trade attribution details CSV (includes is_fake column)
    if not trade_attribution['trade_details'].empty:
        trade_csv_file = results_dir / "trade_attribution.csv"
        trade_attribution['trade_details'].to_csv(trade_csv_file, index=False)
        print(f"Trade attribution details saved to: {trade_csv_file}")


if __name__ == "__main__":
    main()
