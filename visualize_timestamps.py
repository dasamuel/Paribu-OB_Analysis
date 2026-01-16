#!/usr/bin/env python3
"""
Timestamp Comparison Visualization Tool

Visualizes the relationship between ingest timestamps and exchange timestamps
for public trade data. Helps identify discrepancies and patterns in data ingestion.

Usage:
    python visualize_timestamps.py --date YYYY-MM-DD --product SYMBOL

Example:
    python visualize_timestamps.py --date 2026-01-06 --product ADA-TL
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
from datetime import datetime


# Results directory
RESULTS_PATH = Path("/Users/dasamuel/Projects/Sandbox/results")
CHARTS_PATH = RESULTS_PATH / "charts"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize timestamp discrepancies between ingest and exchange times.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_timestamps.py --date 2026-01-06 --product ADA-TL
    python visualize_timestamps.py --date 2026-01-07 --product BTC-TL
        """
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--product",
        required=True,
        help="Product symbol (e.g., ADA-TL, BTC-TL)"
    )
    return parser.parse_args()


def load_trade_data(date: str, product: str) -> pd.DataFrame:
    """Load trade data from CSV file."""
    csv_path = RESULTS_PATH / f"Public_{date}_{product}_trades.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Trade data file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Convert ingest timestamp from microseconds to milliseconds
    df['ts_ingest_unix_ms'] = df['ts_ingest_unix_us'] / 1000
    
    # Calculate delay in seconds (ingest - exchange)
    df['delay_seconds'] = (df['ts_ingest_unix_ms'] - df['ts_exchange_unix_ms']) / 1000
    
    # Convert to datetime for plotting
    df['dt_ingest'] = pd.to_datetime(df['ts_ingest_unix_ms'], unit='ms')
    df['dt_exchange'] = pd.to_datetime(df['ts_exchange_unix_ms'], unit='ms')
    
    return df


def plot_scatter(df: pd.DataFrame, date: str, product: str, output_path: Path) -> None:
    """
    Create scatter plot of exchange timestamp vs ingest timestamp.
    A perfect match would be a 45-degree line.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Convert to matplotlib date format
    exchange_dates = mdates.date2num(df['dt_exchange'])
    ingest_dates = mdates.date2num(df['dt_ingest'])
    
    # Color by delay
    delays = df['delay_seconds']
    scatter = ax.scatter(
        exchange_dates, 
        ingest_dates,
        c=delays,
        cmap='RdYlGn_r',  # Red = high delay, Green = low delay
        alpha=0.5,
        s=10,
        vmin=0,
        vmax=min(delays.quantile(0.95), 3600)  # Cap at 95th percentile or 1 hour
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Delay (seconds)')
    
    # Add diagonal reference line (perfect match)
    min_val = min(exchange_dates.min(), ingest_dates.min())
    max_val = max(exchange_dates.max(), ingest_dates.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect match (y=x)')
    
    # Format axes as datetime
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.yaxis.set_major_locator(mdates.HourLocator(interval=2))
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    ax.set_xlabel('Exchange Timestamp')
    ax.set_ylabel('Ingest Timestamp')
    ax.set_title(
        f'Timestamp Comparison: {product} on {date}\n'
        f'Exchange Time vs Ingest Time (n={len(df):,} trades)'
    )
    
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Make it square for easier visual comparison
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"  Saved scatter plot to: {output_path}")


def plot_delay_histogram(df: pd.DataFrame, date: str, product: str, output_path: Path) -> None:
    """
    Create histogram of delay distribution (ingest - exchange time).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    delays = df['delay_seconds']
    
    # Left plot: Full distribution
    ax1 = axes[0]
    ax1.hist(delays, bins=50, color='#3498db', edgecolor='white', alpha=0.7)
    ax1.axvline(delays.mean(), color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {delays.mean():.1f}s')
    ax1.axvline(delays.median(), color='#27ae60', linestyle='--', linewidth=2, label=f'Median: {delays.median():.1f}s')
    
    ax1.set_xlabel('Delay (seconds)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Full Distribution of Ingestion Delay')
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Right plot: Zoomed to reasonable range (< 5 minutes)
    ax2 = axes[1]
    short_delays = delays[delays <= 300]  # <= 5 minutes
    if len(short_delays) > 0:
        ax2.hist(short_delays, bins=50, color='#9b59b6', edgecolor='white', alpha=0.7)
        ax2.axvline(short_delays.mean(), color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {short_delays.mean():.1f}s')
        ax2.axvline(short_delays.median(), color='#27ae60', linestyle='--', linewidth=2, label=f'Median: {short_delays.median():.1f}s')
        ax2.set_title(f'Delays ≤ 5 min ({len(short_delays):,} / {len(delays):,} = {100*len(short_delays)/len(delays):.1f}%)')
    else:
        ax2.text(0.5, 0.5, 'No delays ≤ 5 min', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Delays ≤ 5 min')
    
    ax2.set_xlabel('Delay (seconds)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    fig.suptitle(
        f'Ingestion Delay Distribution: {product} on {date}\n'
        f'(Delay = Ingest Time - Exchange Time)',
        fontsize=12, fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"  Saved histogram to: {output_path}")


def plot_delay_timeseries(df: pd.DataFrame, date: str, product: str, output_path: Path) -> None:
    """
    Create time series plot showing delay over time.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Sort by exchange time for proper time series
    df_sorted = df.sort_values('dt_exchange')
    
    # Top plot: Delay over time (by exchange timestamp)
    ax1 = axes[0]
    ax1.scatter(
        df_sorted['dt_exchange'], 
        df_sorted['delay_seconds'],
        alpha=0.3, 
        s=5, 
        c='#3498db'
    )
    
    # Add rolling average (window of 100 trades)
    window_size = min(100, len(df_sorted) // 10)
    if window_size > 1:
        rolling_mean = df_sorted['delay_seconds'].rolling(window=window_size, center=True).mean()
        ax1.plot(df_sorted['dt_exchange'], rolling_mean, color='#e74c3c', linewidth=2, 
                 label=f'Rolling mean (n={window_size})')
        ax1.legend()
    
    ax1.set_ylabel('Delay (seconds)')
    ax1.set_title('Ingestion Delay Over Time (by Exchange Timestamp)')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add horizontal reference lines
    ax1.axhline(60, color='orange', linestyle=':', alpha=0.5, label='1 min')
    ax1.axhline(300, color='red', linestyle=':', alpha=0.5, label='5 min')
    
    # Bottom plot: Delay over time (by ingest timestamp)
    ax2 = axes[1]
    df_by_ingest = df.sort_values('dt_ingest')
    ax2.scatter(
        df_by_ingest['dt_ingest'], 
        df_by_ingest['delay_seconds'],
        alpha=0.3, 
        s=5, 
        c='#9b59b6'
    )
    
    if window_size > 1:
        rolling_mean = df_by_ingest['delay_seconds'].rolling(window=window_size, center=True).mean()
        ax2.plot(df_by_ingest['dt_ingest'], rolling_mean, color='#e74c3c', linewidth=2,
                 label=f'Rolling mean (n={window_size})')
        ax2.legend()
    
    ax2.set_ylabel('Delay (seconds)')
    ax2.set_xlabel('Time')
    ax2.set_title('Ingestion Delay Over Time (by Ingest Timestamp)')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    ax2.axhline(60, color='orange', linestyle=':', alpha=0.5)
    ax2.axhline(300, color='red', linestyle=':', alpha=0.5)
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.xticks(rotation=45, ha='right')
    
    fig.suptitle(
        f'Ingestion Delay Time Series: {product} on {date}',
        fontsize=12, fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"  Saved time series to: {output_path}")


def print_statistics(df: pd.DataFrame, date: str, product: str) -> None:
    """Print summary statistics about timestamp discrepancies."""
    delays = df['delay_seconds']
    
    print("\n" + "=" * 60)
    print("Timestamp Discrepancy Statistics")
    print("=" * 60)
    print(f"Product: {product}")
    print(f"Date: {date}")
    print(f"Total trades: {len(df):,}")
    print("-" * 60)
    
    print("\nDelay Statistics (Ingest - Exchange):")
    print(f"  Min:    {delays.min():>10.2f} seconds ({delays.min()/60:.1f} min)")
    print(f"  Max:    {delays.max():>10.2f} seconds ({delays.max()/60:.1f} min)")
    print(f"  Mean:   {delays.mean():>10.2f} seconds ({delays.mean()/60:.1f} min)")
    print(f"  Median: {delays.median():>10.2f} seconds ({delays.median()/60:.1f} min)")
    print(f"  Std:    {delays.std():>10.2f} seconds")
    
    print("\nDelay Distribution:")
    thresholds = [1, 10, 60, 300, 600, 1800, 3600]
    labels = ['1 sec', '10 sec', '1 min', '5 min', '10 min', '30 min', '1 hour']
    
    for thresh, label in zip(thresholds, labels):
        count = (delays <= thresh).sum()
        pct = 100 * count / len(delays)
        print(f"  ≤ {label:>8}: {count:>6,} ({pct:>5.1f}%)")
    
    print("\nTimestamp Ranges:")
    print(f"  Exchange: {df['dt_exchange'].min()} to {df['dt_exchange'].max()}")
    print(f"  Ingest:   {df['dt_ingest'].min()} to {df['dt_ingest'].max()}")
    
    # Check for negative delays (exchange after ingest - should not happen)
    negative_count = (delays < 0).sum()
    if negative_count > 0:
        print(f"\n⚠️  WARNING: {negative_count} trades have negative delay (exchange > ingest)")
    
    print("=" * 60)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate date format
    try:
        datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        print(f"Error: Invalid date format '{args.date}'. Use YYYY-MM-DD.")
        return 1
    
    # Ensure charts directory exists
    CHARTS_PATH.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Timestamp Comparison Visualization")
    print("=" * 60)
    print(f"Date: {args.date}")
    print(f"Product: {args.product}")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading trade data...")
    try:
        df = load_trade_data(args.date, args.product)
        print(f"  Loaded {len(df):,} trades")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nHint: Run extract_public_data.py first to generate the trade data CSV.")
        return 1
    
    # Print statistics
    print_statistics(df, args.date, args.product)
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    # Scatter plot
    scatter_path = CHARTS_PATH / f"{args.date}_{args.product}_timestamp_scatter.png"
    plot_scatter(df, args.date, args.product, scatter_path)
    
    # Histogram
    hist_path = CHARTS_PATH / f"{args.date}_{args.product}_timestamp_delay_hist.png"
    plot_delay_histogram(df, args.date, args.product, hist_path)
    
    # Time series
    timeseries_path = CHARTS_PATH / f"{args.date}_{args.product}_timestamp_delay_timeseries.png"
    plot_delay_timeseries(df, args.date, args.product, timeseries_path)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
