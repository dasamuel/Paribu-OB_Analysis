#!/usr/bin/env python3
"""
Extract Order Chains Tool

Extracts order chains from OMS parquet files. A chain is defined as a unique order
(by client_order_id) that starts with status=new and ends with status=canceled.

Usage:
    python extract_chains.py YYYY-MM-DD exchange_id market_id

Example:
    python extract_chains.py 2026-01-06 115 134
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from pathlib import Path
from datetime import datetime

from results_path import get_results_dir, get_charts_dir, get_shared_dir
from config import get_oms_logs_path


# =============================================================================
# Market Symbol Lookup
# =============================================================================

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


# =============================================================================
# Data Loading & Cleaning (reused from analyze_oms.py)
# =============================================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up column names by removing CAST function wrappers."""
    new_columns = {}
    for col in df.columns:
        if "CAST(" in col and ", '" in col:
            start = col.find("CAST(") + 5
            end = col.find(", '")
            if end > start:
                new_name = col[start:end].strip()
                new_columns[col] = new_name
            else:
                new_columns[col] = col
        else:
            new_columns[col] = col
    return df.rename(columns=new_columns)


def decode_bytes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert byte strings to regular strings."""
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna()
            if len(sample) > 0 and isinstance(sample.iloc[0], bytes):
                df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    return df


def convert_timestamp(df: pd.DataFrame, time_col: str = 'time') -> pd.DataFrame:
    """Convert Unix timestamp to datetime."""
    if time_col in df.columns:
        df['datetime'] = pd.to_datetime(df[time_col], unit='s')
        cols = [c for c in df.columns if c != 'datetime']
        time_idx = cols.index(time_col) if time_col in cols else 0
        cols.insert(time_idx + 1, 'datetime')
        df = df[cols]
    return df


def load_parquet(file_path: Path) -> pd.DataFrame:
    """Load data from Parquet file."""
    print(f"Loading data from: {file_path}")
    
    df = pd.read_parquet(file_path)
    
    # Clean and transform
    df = clean_column_names(df)
    df = decode_bytes(df)
    
    # Find time column
    time_col = None
    for col in ['time', "CAST(time, 'Float64')"]:
        if col in df.columns:
            time_col = col
            break
    
    if time_col:
        df = convert_timestamp(df, time_col)
    
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


# =============================================================================
# Chain Extraction
# =============================================================================

def extract_chains(df: pd.DataFrame, exchange_id: int, market_id: int) -> pd.DataFrame:
    """
    Extract order chains filtered by exchange_id and market_id.
    
    A chain is a unique order (client_order_id) that has both status=new and status=canceled.
    
    Returns DataFrame with columns:
        time_open, datetime_open, time_close, datetime_close, side, price, initial_qty
    """
    # Filter by exchange and market
    filtered = df[(df['exchange_id'] == exchange_id) & (df['market_id'] == market_id)].copy()
    print(f"Filtered to {len(filtered):,} rows for exchange_id={exchange_id}, market_id={market_id}")
    
    if len(filtered) == 0:
        return pd.DataFrame()
    
    # Get orders with status=new
    new_orders = filtered[filtered['status'] == 'new'][
        ['client_order_id', 'time', 'datetime', 'side', 'price', 'initial_qty']
    ].copy()
    new_orders = new_orders.rename(columns={
        'time': 'time_open',
        'datetime': 'datetime_open'
    })
    
    # Get orders with status=canceled
    canceled_orders = filtered[filtered['status'] == 'canceled'][
        ['client_order_id', 'time', 'datetime']
    ].copy()
    canceled_orders = canceled_orders.rename(columns={
        'time': 'time_close',
        'datetime': 'datetime_close'
    })
    
    print(f"Found {len(new_orders):,} 'new' status records")
    print(f"Found {len(canceled_orders):,} 'canceled' status records")
    
    # Merge to find orders with both statuses (chains)
    chains = new_orders.merge(canceled_orders, on='client_order_id', how='inner')
    
    # Filter out negative time differences (data anomalies)
    chains = chains[chains['time_close'] >= chains['time_open']]
    
    # Calculate duration in seconds to microsecond accuracy
    chains['duration'] = chains['time_close'] - chains['time_open']
    
    # Select and order columns for output
    chains = chains[[
        'time_open', 'datetime_open', 'time_close', 'datetime_close', 'duration',
        'side', 'price', 'initial_qty'
    ]]
    
    # Sort by open time
    chains = chains.sort_values('time_open').reset_index(drop=True)
    
    print(f"Extracted {len(chains):,} order chains")
    return chains


# =============================================================================
# Side Filtering and Empty Gap Calculation
# =============================================================================

def add_empty_gap(chains: pd.DataFrame) -> pd.DataFrame:
    """
    Add empty_gap column: time between previous chain's close and current chain's open.
    
    For the first row, empty_gap is set to -1 (undefined).
    """
    if len(chains) == 0:
        return chains
    
    # Ensure sorted by time_open
    chains = chains.sort_values('time_open').reset_index(drop=True)
    
    # Calculate empty_gap: time_open[i] - time_close[i-1]
    # Shift time_close down by 1 to align with current row
    prev_time_close = chains['time_close'].shift(1)
    chains['empty_gap'] = chains['time_open'] - prev_time_close
    
    # First row has no previous, set to -1
    chains.loc[0, 'empty_gap'] = -1
    
    # Reorder columns to put empty_gap after duration
    cols = list(chains.columns)
    cols.remove('empty_gap')
    duration_idx = cols.index('duration')
    cols.insert(duration_idx + 1, 'empty_gap')
    chains = chains[cols]
    
    return chains


def split_and_save_by_side(chains: pd.DataFrame, output_dir: Path, charts_dir: Path,
                           date: str, exchange_id: int, market_id: int) -> dict:
    """
    Split chains by side, add empty_gap, save to separate CSV files, and generate histograms.
    
    Returns dict with counts for each side.
    """
    results = {}
    
    for side, label in [(1, 'buy'), (-1, 'sell')]:
        side_chains = chains[chains['side'] == side].copy()
        
        if len(side_chains) == 0:
            print(f"\nNo {label} chains found.")
            results[label] = 0
            continue
        
        # Add empty_gap calculation
        side_chains = add_empty_gap(side_chains)
        
        # Output filename (simplified - directory already contains date/market info)
        output_file = output_dir / f"chains_{label}.csv"
        
        # Save to CSV
        side_chains.to_csv(output_file, index=False)
        print(f"\nSaved {len(side_chains):,} {label} chains to: {output_file}")
        
        # Print summary for this side
        print(f"  Duration (seconds): mean={side_chains['duration'].mean():.3f}, median={side_chains['duration'].median():.3f}")
        
        # Empty gap stats (excluding first row which is -1)
        valid_gaps = side_chains[side_chains['empty_gap'] >= 0]['empty_gap']
        if len(valid_gaps) > 0:
            print(f"  Empty gap (seconds): mean={valid_gaps.mean():.3f}, median={valid_gaps.median():.3f}, min={valid_gaps.min():.6f}, max={valid_gaps.max():.3f}")
        
        # Generate histograms
        generate_histograms_for_side(side_chains, charts_dir, label)
        
        results[label] = len(side_chains)
    
    return results


# =============================================================================
# Histogram Generation
# =============================================================================

def generate_histogram(data: pd.Series, column_name: str, output_path: Path, 
                       title: str, xlabel: str) -> None:
    """
    Generate a histogram with 95th percentile as max x-axis and 20 bins.
    
    Args:
        data: Series of values to plot
        column_name: Name of the column (for labeling)
        output_path: Path to save the PNG file
        title: Plot title
        xlabel: X-axis label
    """
    if len(data) == 0:
        print(f"  No data for {column_name} histogram")
        return
    
    # Calculate 95th percentile for x-axis max
    p95 = np.percentile(data, 95)
    
    # Filter data to 0 to 95th percentile for binning
    filtered_data = data[(data >= 0) & (data <= p95)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram with 20 bins from 0 to 95th percentile
    bins = np.linspace(0, p95, 21)  # 21 edges = 20 bins
    ax.hist(filtered_data, bins=bins, edgecolor='black', alpha=0.7)
    
    # Labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.set_title(title)
    
    # Add stats annotation
    stats_text = (
        f"Total: {len(data):,}\n"
        f"In plot: {len(filtered_data):,} ({len(filtered_data)/len(data)*100:.1f}%)\n"
        f"Mean: {data.mean():.3f}s\n"
        f"Median: {data.median():.3f}s\n"
        f"95th pct: {p95:.3f}s"
    )
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"  Saved histogram: {output_path}")


def generate_histograms_for_side(side_chains: pd.DataFrame, charts_dir: Path,
                                  label: str) -> None:
    """
    Generate duration and empty_gap histograms for a side-filtered DataFrame.
    """
    side_title = "Buy" if label == "buy" else "Sell"
    
    # Duration histogram (simplified filename - directory contains context)
    duration_path = charts_dir / f"chains_{label}_duration_hist.png"
    generate_histogram(
        side_chains['duration'],
        'duration',
        duration_path,
        f"{side_title} Order Chain Duration",
        "Duration (seconds)"
    )
    
    # Empty gap histogram (exclude first row which is -1)
    valid_gaps = side_chains[side_chains['empty_gap'] >= 0]['empty_gap']
    empty_gap_path = charts_dir / f"chains_{label}_empty_gap_hist.png"
    generate_histogram(
        valid_gaps,
        'empty_gap',
        empty_gap_path,
        f"{side_title} Order Chain Empty Gap",
        "Empty Gap (seconds)"
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract order chains from OMS parquet files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python extract_chains.py 2026-01-06 115 134
    python extract_chains.py 2026-01-07 115 188
        """
    )
    parser.add_argument('date', type=str, help='Date in YYYY-MM-DD format')
    parser.add_argument('exchange_id', type=int, help='Exchange ID to filter')
    parser.add_argument('market_id', type=int, help='Market ID to filter')
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        print(f"Error: Invalid date format '{args.date}'. Use YYYY-MM-DD.")
        sys.exit(1)
    
    # Construct parquet file path
    oms_logs_dir = get_oms_logs_path()
    parquet_path = oms_logs_dir / f"{args.date}__oms_order_log.parq"
    
    if not parquet_path.exists():
        print(f"Error: Parquet file not found: {parquet_path}")
        sys.exit(1)
    
    # Look up market symbol from exchange_id and market_id
    market_symbol = get_market_symbol(args.exchange_id, args.market_id)
    
    # Get output directories using new folder structure
    output_dir = get_results_dir(args.date, market_symbol)
    charts_dir = get_charts_dir(args.date, market_symbol)
    
    print("=" * 60)
    print("Extract Order Chains")
    print("=" * 60)
    print(f"Date: {args.date}")
    print(f"Exchange ID: {args.exchange_id}")
    print(f"Market ID: {args.market_id}")
    print(f"Market Symbol: {market_symbol}")
    print(f"Output Directory: {output_dir}")
    print("=" * 60)
    
    # Load parquet file
    df = load_parquet(parquet_path)
    
    # Extract chains
    chains = extract_chains(df, args.exchange_id, args.market_id)
    
    if len(chains) == 0:
        print("\nNo order chains found for the specified filters.")
        sys.exit(0)
    
    # Display overall summary
    print("\n" + "=" * 60)
    print("Overall Summary")
    print("=" * 60)
    print(f"Total chains: {len(chains):,}")
    
    duration = chains['duration']
    print(f"Duration (seconds):")
    print(f"  Mean:   {duration.mean():.3f}")
    print(f"  Median: {duration.median():.3f}")
    print(f"  Min:    {duration.min():.3f}")
    print(f"  Max:    {duration.max():.3f}")
    
    print(f"\nSide distribution:")
    side_counts = chains['side'].value_counts()
    for side, count in side_counts.items():
        side_label = "Buy" if side == 1 else "Sell" if side == -1 else str(side)
        print(f"  {side_label}: {count:,} ({count/len(chains)*100:.1f}%)")
    
    # Split by side, add empty_gap, and save separate files
    print("\n" + "=" * 60)
    print("Generating Side-Filtered Files with Empty Gap")
    print("=" * 60)
    
    results = split_and_save_by_side(chains, output_dir, charts_dir, args.date, args.exchange_id, args.market_id)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

