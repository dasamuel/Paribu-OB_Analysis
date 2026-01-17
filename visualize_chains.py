#!/usr/bin/env python3
"""
Order Chain Visualization Tool

Visualizes order chains as horizontal lines on a time vs price plot.
Buy orders shown in green, sell orders in red.

Usage:
    python visualize_chains.py YYYY-MM-DD exchange_id market_id HH:MM:SS HH:MM:SS

Example:
    python visualize_chains.py 2026-01-06 115 134 00:00:00 01:00:00
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

from results_path import get_results_dir, get_charts_dir, get_shared_dir


def load_markets_map() -> pd.DataFrame:
    """Load the markets map CSV from the shared directory."""
    # Try shared directory first (new location)
    shared_dir = get_shared_dir()
    markets_file = shared_dir / "markets_map.csv"
    
    # Fall back to old location if not in shared
    if not markets_file.exists():
        markets_file = Path(__file__).parent / "results" / "markets_map.csv"
    
    if not markets_file.exists():
        return pd.DataFrame()
    
    return pd.read_csv(markets_file)


def get_market_symbol(exchange_id: int, market_id: int) -> str:
    """
    Get the market symbol (e.g., 'ADA-TL') from exchange_id and market_id.
    
    Returns the symbol if found, otherwise returns a fallback string.
    """
    markets_df = load_markets_map()
    
    if markets_df.empty:
        return f"exchange-{exchange_id}_market-{market_id}"
    
    # Check Paribu columns
    match = markets_df[
        (markets_df['paribu_exchange_id'] == exchange_id) & 
        (markets_df['paribu_market_id'] == market_id)
    ]
    
    if not match.empty:
        return f"{match.iloc[0]['base_currency']}-TL"
    
    # Check BTCTurk columns
    match = markets_df[
        (markets_df['btcturk_exchange_id'] == exchange_id) & 
        (markets_df['btcturk_market_id'] == market_id)
    ]
    
    if not match.empty:
        return f"{match.iloc[0]['base_currency']}-TL"
    
    # Fallback
    return f"exchange-{exchange_id}_market-{market_id}"


def load_chains(file_path: Path) -> pd.DataFrame:
    """Load chain data from CSV file."""
    if not file_path.exists():
        print(f"Warning: File not found: {file_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    
    # Convert datetime columns
    df['datetime_open'] = pd.to_datetime(df['datetime_open'])
    df['datetime_close'] = pd.to_datetime(df['datetime_close'])
    
    return df


def filter_chains_by_time(df: pd.DataFrame, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Filter chains to those that overlap with the time window.
    
    A chain overlaps if:
    - It opens before the window ends AND closes after the window starts
    """
    if len(df) == 0:
        return df
    
    mask = (df['datetime_open'] < end_dt) & (df['datetime_close'] > start_dt)
    filtered = df[mask].copy()
    
    # Clip the display times to the window boundaries
    filtered['display_open'] = filtered['datetime_open'].clip(lower=start_dt)
    filtered['display_close'] = filtered['datetime_close'].clip(upper=end_dt)
    
    return filtered


def plot_chains(buy_chains: pd.DataFrame, sell_chains: pd.DataFrame,
                start_dt: datetime, end_dt: datetime,
                date: str, exchange_id: int, market_id: int,
                output_path: Path) -> None:
    """
    Plot order chains as horizontal lines at their price levels.
    
    Buy orders in green, sell orders in red.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot sell orders (red)
    for _, row in sell_chains.iterrows():
        ax.hlines(
            y=row['price'],
            xmin=row['display_open'],
            xmax=row['display_close'],
            colors='#e74c3c',  # Red
            linewidth=1.0,
            alpha=0.7
        )
    
    # Plot buy orders (green)
    for _, row in buy_chains.iterrows():
        ax.hlines(
            y=row['price'],
            xmin=row['display_open'],
            xmax=row['display_close'],
            colors='#27ae60',  # Green
            linewidth=1.0,
            alpha=0.7
        )
    
    # Format x-axis as time
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Set x-axis limits to the window
    ax.set_xlim(start_dt, end_dt)
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    
    # Labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    
    start_str = start_dt.strftime('%H:%M:%S')
    end_str = end_dt.strftime('%H:%M:%S')
    ax.set_title(
        f'Order Chains: {date} | Exchange {exchange_id} | Market {market_id}\n'
        f'Time Window: {start_str} - {end_str}'
    )
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#27ae60', linewidth=2, label=f'Buy ({len(buy_chains):,})'),
        Line2D([0], [0], color='#e74c3c', linewidth=2, label=f'Sell ({len(sell_chains):,})')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add grid for readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize order chains as horizontal lines on a time vs price plot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_chains.py 2026-01-06 115 134 00:00:00 01:00:00
    python visualize_chains.py 2026-01-06 115 188 12:00:00 13:30:00
        """
    )
    parser.add_argument('date', type=str, help='Date in YYYY-MM-DD format')
    parser.add_argument('exchange_id', type=int, help='Exchange ID')
    parser.add_argument('market_id', type=int, help='Market ID')
    parser.add_argument('start', type=str, help='Start time in HH:MM:SS format')
    parser.add_argument('end', type=str, help='End time in HH:MM:SS format')
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        base_date = datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        print(f"Error: Invalid date format '{args.date}'. Use YYYY-MM-DD.")
        sys.exit(1)
    
    # Validate and parse time formats
    try:
        start_time = datetime.strptime(args.start, '%H:%M:%S').time()
        end_time = datetime.strptime(args.end, '%H:%M:%S').time()
    except ValueError as e:
        print(f"Error: Invalid time format. Use HH:MM:SS. ({e})")
        sys.exit(1)
    
    # Combine date and time
    start_dt = datetime.combine(base_date.date(), start_time)
    end_dt = datetime.combine(base_date.date(), end_time)
    
    if end_dt <= start_dt:
        print("Error: End time must be after start time.")
        sys.exit(1)
    
    # Look up market symbol from exchange_id and market_id
    market_symbol = get_market_symbol(args.exchange_id, args.market_id)
    
    # Get directories using new folder structure
    results_dir = get_results_dir(args.date, market_symbol)
    charts_dir = get_charts_dir(args.date, market_symbol)
    
    # Chain files (try new filenames first, fall back to old)
    buy_file = results_dir / "chains_buy.csv"
    sell_file = results_dir / "chains_sell.csv"
    
    # Fall back to old location if new files don't exist
    if not buy_file.exists():
        old_results_dir = Path.cwd() / "results"
        buy_file = old_results_dir / f"{args.date}_exchange-{args.exchange_id}_market-{args.market_id}_chains_buy.csv"
    if not sell_file.exists():
        old_results_dir = Path.cwd() / "results"
        sell_file = old_results_dir / f"{args.date}_exchange-{args.exchange_id}_market-{args.market_id}_chains_sell.csv"
    
    print("=" * 60)
    print("Order Chain Visualization")
    print("=" * 60)
    print(f"Date: {args.date}")
    print(f"Exchange ID: {args.exchange_id}")
    print(f"Market ID: {args.market_id}")
    print(f"Market Symbol: {market_symbol}")
    print(f"Time Window: {args.start} - {args.end}")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading buy chains from: {buy_file}")
    buy_chains = load_chains(buy_file)
    print(f"  Loaded {len(buy_chains):,} buy chains")
    
    print(f"Loading sell chains from: {sell_file}")
    sell_chains = load_chains(sell_file)
    print(f"  Loaded {len(sell_chains):,} sell chains")
    
    if len(buy_chains) == 0 and len(sell_chains) == 0:
        print("\nError: No chain data found. Ensure the chain files exist.")
        sys.exit(1)
    
    # Filter to time window
    print(f"\nFiltering to time window {args.start} - {args.end}...")
    buy_filtered = filter_chains_by_time(buy_chains, start_dt, end_dt)
    sell_filtered = filter_chains_by_time(sell_chains, start_dt, end_dt)
    
    print(f"  Buy chains in window: {len(buy_filtered):,}")
    print(f"  Sell chains in window: {len(sell_filtered):,}")
    
    if len(buy_filtered) == 0 and len(sell_filtered) == 0:
        print("\nWarning: No chains found in the specified time window.")
        sys.exit(0)
    
    # Generate output filename (simplified - directory already contains date/market info)
    start_safe = args.start.replace(':', '-')
    end_safe = args.end.replace(':', '-')
    output_file = charts_dir / f"chains_visual_{start_safe}_{end_safe}.png"
    
    # Plot
    print(f"\nGenerating visualization...")
    plot_chains(buy_filtered, sell_filtered, start_dt, end_dt,
                args.date, args.exchange_id, args.market_id, output_file)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

