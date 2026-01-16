#!/usr/bin/env python3
"""
Orderbook Depth Heatmap Visualization Tool

Visualizes consolidated orderbook data as a heatmap where each time snapshot
is a vertical column of 6 colored boxes (3 bids + 3 asks), with colors
representing quantity levels on a rainbow spectrum from red (min) to blue (max).

Usage:
    python visualize_orderbook.py YYYY-MM-DD PRODUCT HH:MM:SS HH:MM:SS [options]

Example:
    # Basic usage (no highlighting)
    python visualize_orderbook.py 2026-01-06 ADA 00:00:00 00:05:00
    
    # Manual highlight quantities
    python visualize_orderbook.py 2026-01-06 ADA 00:00:00 00:05:00 --highlight-qty 1600.0
    python visualize_orderbook.py 2026-01-06 ADA 00:00:00 00:05:00 --highlight-qty 70.9 1600.0
    
    # Auto-detect quantities from chain files
    python visualize_orderbook.py 2026-01-06 ADA 00:00:00 00:05:00 --exchange-id 129 --market-id 293
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sys
import argparse
from pathlib import Path
from datetime import datetime


def load_orderbook(file_path: Path) -> pd.DataFrame:
    """Load consolidated orderbook data from CSV file."""
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    df = pd.read_csv(file_path)
    
    # Convert datetime column
    df['datetime_ingest'] = pd.to_datetime(df['datetime_ingest'])
    
    return df


def load_trades(file_path: Path, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Load trades data from CSV file and filter by time window.
    
    Args:
        file_path: Path to trades CSV file
        start_dt: Start datetime for filtering
        end_dt: End datetime for filtering
    
    Returns:
        DataFrame with trades in the time window, or empty DataFrame if file doesn't exist
    """
    if not file_path.exists():
        print(f"  Warning: Trades file not found: {file_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    
    # Convert datetime column
    df['datetime_exchange'] = pd.to_datetime(df['datetime_exchange'])
    
    # Filter to time window
    mask = (df['datetime_exchange'] >= start_dt) & (df['datetime_exchange'] <= end_dt)
    return df[mask].copy()


def load_highlight_quantities_from_chains(results_dir: Path, date: str,
                                          exchange_id: int, market_id: int) -> set:
    """
    Load unique initial_qty values from buy and sell chain CSV files.
    
    Args:
        results_dir: Path to results directory containing chain files
        date: Date string in YYYY-MM-DD format
        exchange_id: Exchange ID used in chain filename
        market_id: Market ID used in chain filename
    
    Returns:
        Set of unique initial_qty values found across both chain files
    """
    quantities = set()
    
    for side in ['buy', 'sell']:
        chain_file = results_dir / f"{date}_exchange-{exchange_id}_market-{market_id}_chains_{side}.csv"
        if chain_file.exists():
            df = pd.read_csv(chain_file)
            if 'initial_qty' in df.columns:
                side_qtys = df['initial_qty'].unique()
                quantities.update(side_qtys)
                print(f"  Loaded {len(side_qtys)} unique quantities from {side} chains")
            else:
                print(f"  Warning: 'initial_qty' column not found in {chain_file}")
        else:
            print(f"  Warning: Chain file not found: {chain_file}")
    
    return quantities


def map_trades_to_indices(trades: pd.DataFrame, orderbook_timestamps: pd.Series) -> list:
    """
    Map each trade to the nearest orderbook snapshot index.
    
    For each trade, finds the closest orderbook snapshot timestamp and determines
    which row to highlight based on trade side:
    - Buy trade (side=1): buyer lifts the ask -> highlight Ask 1 (row index 2)
    - Sell trade (side=2): seller hits the bid -> highlight Bid 1 (row index 3)
    
    Args:
        trades: DataFrame with 'datetime_exchange' and 'side' columns
        orderbook_timestamps: Series of orderbook snapshot timestamps
    
    Returns:
        List of (snapshot_index, row_index) tuples for overlay rendering
    """
    if trades.empty:
        return []
    
    # Convert timestamps to numpy arrays for efficient searching
    ob_times = orderbook_timestamps.values.astype('datetime64[ns]')
    trade_times = trades['datetime_exchange'].values.astype('datetime64[ns]')
    
    # Row indices for trade sides
    # Row 2 = Ask 1 (best ask), Row 3 = Bid 1 (best bid)
    SIDE_BUY = 1
    ASK1_ROW = 2
    BID1_ROW = 3
    
    overlay_positions = []
    
    for i, trade_time in enumerate(trade_times):
        # Find insertion point (index where trade_time would be inserted to maintain sort)
        idx = np.searchsorted(ob_times, trade_time)
        
        # Determine nearest snapshot index
        if idx == 0:
            # Trade is before or at first snapshot
            snap_idx = 0
        elif idx >= len(ob_times):
            # Trade is after last snapshot
            snap_idx = len(ob_times) - 1
        else:
            # Trade is between two snapshots - find nearest
            time_before = ob_times[idx - 1]
            time_after = ob_times[idx]
            if (trade_time - time_before) <= (time_after - trade_time):
                snap_idx = idx - 1
            else:
                snap_idx = idx
        
        # Determine row based on trade side
        side = trades.iloc[i]['side']
        row_idx = ASK1_ROW if side == SIDE_BUY else BID1_ROW
        
        overlay_positions.append((snap_idx, row_idx))
    
    return overlay_positions


def filter_by_time(df: pd.DataFrame, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Filter orderbook data to the specified time window."""
    mask = (df['datetime_ingest'] >= start_dt) & (df['datetime_ingest'] <= end_dt)
    return df[mask].copy()


def extract_quantities(df: pd.DataFrame) -> np.ndarray:
    """
    Extract quantity columns into a 2D array.
    
    Returns array with shape (6, n_snapshots) where rows are (top to bottom):
    - Row 0: ask3 (deepest ask) - highest price
    - Row 1: ask2
    - Row 2: ask1 (best ask)
    - Row 3: bid1 (best bid)
    - Row 4: bid2
    - Row 5: bid3 (deepest bid) - lowest price
    """
    qty_columns = [
        'best_ask_qty_3',  # ask3 - deepest (highest price)
        'best_ask_qty_2',  # ask2
        'best_ask_qty_1',  # ask1 - best
        'best_bid_qty_1',  # bid1 - best
        'best_bid_qty_2',  # bid2
        'best_bid_qty_3',  # bid3 - deepest (lowest price)
    ]
    
    return df[qty_columns].values.T


def plot_heatmap(quantities: np.ndarray, timestamps: pd.Series,
                 prices: dict,
                 date: str, product: str, start_time: str, end_time: str,
                 output_path: Path, highlight_qtys: list = None,
                 trades_overlay: list = None) -> None:
    """
    Plot dual-panel orderbook visualization: heatmap on top, best bid/ask prices on bottom.
    
    Top panel: Heatmap where each column is a time snapshot, each row is a price level.
    Colors represent quantity magnitude on a rainbow spectrum.
    If highlight_qtys is provided, cells matching those quantities are colored black.
    If trades_overlay is provided, trade cells are colored white.
    
    Bottom panel: Line chart of best bid and best ask prices over time,
    with inverted Y-axis to match heatmap orientation (Ask below Bid).
    
    Args:
        quantities: 2D array of quantities (6 levels x n_snapshots)
        timestamps: Series of datetime timestamps
        prices: Dict with 'best_bid_price_1' and 'best_ask_price_1' Series
        date: Date string for title
        product: Product symbol for title
        start_time: Start time string for title
        end_time: End time string for title
        output_path: Path to save the output image
        highlight_qtys: Optional list of quantities to highlight in black
        trades_overlay: Optional list of (snapshot_index, row_index) tuples for trade markers
    """
    n_levels, n_snapshots = quantities.shape
    
    # Compute global min/max for color normalization
    qty_min = np.nanmin(quantities)
    qty_max = np.nanmax(quantities)
    
    # Create figure with appropriate size
    # Scale width based on number of snapshots, but cap it
    fig_width = min(20, max(10, n_snapshots / 50))
    fig_height = 8  # Increased height to accommodate dual plots
    
    # Create layout with GridSpec: 2 rows, 2 cols (main plots + colorbar column)
    # This ensures heatmap and price chart have identical widths
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[20, 1],
                          hspace=0.05, wspace=0.02)
    
    ax_heatmap = fig.add_subplot(gs[0, 0])
    ax_prices = fig.add_subplot(gs[1, 0], sharex=ax_heatmap)
    ax_cbar = fig.add_subplot(gs[0, 1])  # Colorbar axis (top right)
    
    ax = ax_heatmap  # Use ax_heatmap for existing heatmap code
    
    # Create heatmap using imshow with rainbow colormap
    # Note: imshow expects (rows, cols) so quantities is already correct
    im = ax.imshow(
        quantities,
        aspect='auto',
        cmap='rainbow',
        vmin=qty_min,
        vmax=qty_max,
        interpolation='nearest'
    )
    
    # Overlay white boxes for highlighted quantities (my orders)
    if highlight_qtys is not None and len(highlight_qtys) > 0:
        # Build combined mask for all highlight quantities
        combined_mask = np.zeros_like(quantities, dtype=bool)
        
        for qty in highlight_qtys:
            mask = np.isclose(quantities, qty, rtol=1e-9, atol=1e-9)
            n_matched = np.sum(mask)
            if n_matched > 0:
                print(f"  Highlighted {n_matched:,} cells with quantity {qty}")
            else:
                print(f"  Warning: Quantity {qty} not found in data")
            combined_mask |= mask
        
        n_total_highlighted = np.sum(combined_mask)
        if n_total_highlighted > 0:
            # Create an overlay array with NaN for non-highlighted cells
            overlay = np.where(combined_mask, 1.0, np.nan)
            
            # Plot overlay with white color
            ax.imshow(
                overlay,
                aspect='auto',
                cmap=mcolors.ListedColormap(['white']),
                vmin=0,
                vmax=1,
                interpolation='nearest'
            )
    
    # Overlay black markers for trades
    if trades_overlay is not None and len(trades_overlay) > 0:
        trade_mask = np.zeros_like(quantities, dtype=bool)
        for snap_idx, row_idx in trades_overlay:
            # Bounds check to avoid index errors
            if 0 <= snap_idx < n_snapshots and 0 <= row_idx < n_levels:
                trade_mask[row_idx, snap_idx] = True
        
        n_trades = np.sum(trade_mask)
        if n_trades > 0:
            # Create overlay with black markers
            trade_overlay = np.where(trade_mask, 1.0, np.nan)
            ax.imshow(
                trade_overlay,
                aspect='auto',
                cmap=mcolors.ListedColormap(['black']),
                vmin=0,
                vmax=1,
                interpolation='nearest'
            )
    
    # Add colorbar in dedicated axis (doesn't steal width from heatmap)
    cbar = plt.colorbar(im, cax=ax_cbar, label='Quantity')
    cbar.ax.tick_params(labelsize=9)
    
    # Y-axis labels (price levels) - Ask at top (higher prices), Bid at bottom (lower prices)
    level_labels = ['Ask 3', 'Ask 2', 'Ask 1', 'Bid 1', 'Bid 2', 'Bid 3']
    ax.set_yticks(range(n_levels))
    ax.set_yticklabels(level_labels)
    
    # X-axis time labels (sparse, auto-scaled) - set on shared axis
    # Show approximately 10 labels regardless of data size
    n_labels = min(10, n_snapshots)
    if n_snapshots > 1:
        label_indices = np.linspace(0, n_snapshots - 1, n_labels, dtype=int)
        ax.set_xticks(label_indices)
        time_labels = [timestamps.iloc[i].strftime('%H:%M:%S') for i in label_indices]
        # Hide x-axis tick labels on heatmap (they'll show on bottom plot only)
        ax.tick_params(axis='x', labelbottom=False)
        # Set labels on bottom subplot
        ax_prices.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=8)
    
    # Labels and title (no X-axis label on heatmap since it's shared with bottom plot)
    ax.set_ylabel('Order Book Level')
    title = f'Orderbook Depth Heatmap: {date} | {product}-TL\n'
    title += f'Time Window: {start_time} - {end_time} | Snapshots: {n_snapshots:,}'
    if trades_overlay is not None and len(trades_overlay) > 0:
        title += f' | Trades: {len(trades_overlay)} (black)'
    if highlight_qtys is not None and len(highlight_qtys) > 0:
        title += f' | My Orders (white)'
    ax.set_title(title)
    
    # Add horizontal line between bids and asks
    ax.axhline(y=2.5, color='white', linewidth=2, linestyle='-')
    
    # === Bottom subplot: Best Bid/Ask Price Step Chart ===
    x_indices = np.arange(n_snapshots)
    
    # Plot best bid and best ask prices as step functions
    # Use 'post' to show price as constant until the next change (horizontal then vertical)
    ax_prices.step(x_indices, prices['best_bid_price_1'].values, 
                   where='post', color='green', linewidth=1, label='Best Bid')
    ax_prices.step(x_indices, prices['best_ask_price_1'].values, 
                   where='post', color='red', linewidth=1, label='Best Ask')
    
    # Standard Y-axis orientation: higher prices at top (Ask above Bid)
    
    # Labels and legend for price subplot
    ax_prices.set_ylabel('Price')
    ax_prices.set_xlabel('Time')
    ax_prices.legend(loc='upper right', fontsize=8)
    ax_prices.grid(True, alpha=0.3)
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize consolidated orderbook as a depth heatmap',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_orderbook.py 2026-01-06 ADA 00:00:00 00:05:00
    python visualize_orderbook.py 2026-01-06 ADA 12:00:00 13:30:00
    python visualize_orderbook.py 2026-01-06 ADA 00:00:00 00:05:00 --highlight-qty 1600.0
    python visualize_orderbook.py 2026-01-06 ADA 00:00:00 00:05:00 --highlight-qty 70.9 1600.0
    python visualize_orderbook.py 2026-01-06 ADA 00:00:00 00:05:00 --exchange-id 129 --market-id 293
    python visualize_orderbook.py 2026-01-06 ADA 12:45:00 12:50:00 --no-trades
    python visualize_orderbook.py 2026-01-06 ADA 00:00:00 00:05:00 --no-highlight
        """
    )
    parser.add_argument('date', type=str, help='Date in YYYY-MM-DD format')
    parser.add_argument('product', type=str, help='Product symbol (e.g., ADA)')
    parser.add_argument('start', type=str, help='Start time in HH:MM:SS format')
    parser.add_argument('end', type=str, help='End time in HH:MM:SS format')
    parser.add_argument('--highlight-qty', type=float, nargs='+', default=None,
                        help='Quantity value(s) to highlight in black (can specify multiple)')
    parser.add_argument('--exchange-id', type=int, default=None,
                        help='Exchange ID for auto-loading highlight quantities from chain files')
    parser.add_argument('--market-id', type=int, default=None,
                        help='Market ID for auto-loading highlight quantities from chain files')
    parser.add_argument('--no-highlight', action='store_true',
                        help='Disable quantity highlighting entirely')
    parser.add_argument('--no-trades', action='store_true',
                        help='Disable trades overlay (trades are shown by default)')
    
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
    
    # Construct file paths
    results_dir = Path.cwd() / "results"
    charts_dir = results_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # Input file: Public_YYYY-MM-DD_PRODUCT-TL_orderbook_consolidated.csv
    input_file = results_dir / f"Public_{args.date}_{args.product}-TL_orderbook_consolidated.csv"
    # Trades file: Public_YYYY-MM-DD_PRODUCT-TL_trades.csv
    trades_file = results_dir / f"Public_{args.date}_{args.product}-TL_trades.csv"
    
    # Determine highlight quantities
    # Priority: 1) --no-highlight disables, 2) --highlight-qty explicit values, 3) auto-detect from chains
    highlight_qtys = None
    highlight_mode = "none"
    
    if args.no_highlight:
        highlight_qtys = None
        highlight_mode = "disabled"
    elif args.highlight_qty is not None:
        highlight_qtys = args.highlight_qty
        highlight_mode = "explicit"
    elif args.exchange_id is not None and args.market_id is not None:
        # Auto-detect from chain files
        print(f"\nAuto-detecting highlight quantities from chain files...")
        highlight_set = load_highlight_quantities_from_chains(
            results_dir, args.date, args.exchange_id, args.market_id
        )
        if highlight_set:
            highlight_qtys = list(highlight_set)
            highlight_mode = "auto"
            print(f"  Total unique quantities: {len(highlight_qtys)}")
        else:
            print(f"  Warning: No quantities found in chain files")
    elif args.exchange_id is not None or args.market_id is not None:
        # Only one of exchange_id/market_id provided - warn user
        print("Warning: Both --exchange-id and --market-id are required for auto-detection.")
        print("         No highlighting will be applied.")
    
    print("=" * 60)
    print("Orderbook Depth Heatmap Visualization")
    print("=" * 60)
    print(f"Date: {args.date}")
    print(f"Product: {args.product}")
    print(f"Time Window: {args.start} - {args.end}")
    print(f"Trades Overlay: {'Disabled' if args.no_trades else 'Enabled'}")
    if highlight_mode == "disabled":
        print(f"Highlighting: Disabled")
    elif highlight_mode == "explicit":
        print(f"Highlight Quantities (explicit): {highlight_qtys}")
    elif highlight_mode == "auto":
        print(f"Highlight Quantities (auto from chains): {len(highlight_qtys)} values")
        print(f"  Exchange ID: {args.exchange_id}, Market ID: {args.market_id}")
    else:
        print(f"Highlighting: None (use --highlight-qty or --exchange-id/--market-id)")
    print(f"Orderbook File: {input_file}")
    if not args.no_trades:
        print(f"Trades File: {trades_file}")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading orderbook data...")
    df = load_orderbook(input_file)
    print(f"  Loaded {len(df):,} total snapshots")
    
    # Filter to time window
    print(f"\nFiltering to time window {args.start} - {args.end}...")
    df_filtered = filter_by_time(df, start_dt, end_dt)
    print(f"  Snapshots in window: {len(df_filtered):,}")
    
    if len(df_filtered) == 0:
        print("\nError: No data found in the specified time window.")
        sys.exit(1)
    
    # Extract quantities
    print("\nExtracting quantity data...")
    quantities = extract_quantities(df_filtered)
    qty_min = np.nanmin(quantities)
    qty_max = np.nanmax(quantities)
    print(f"  Quantity range: {qty_min:,.2f} - {qty_max:,.2f}")
    
    # Extract prices for the bottom subplot
    prices = {
        'best_bid_price_1': df_filtered['best_bid_price_1'],
        'best_ask_price_1': df_filtered['best_ask_price_1']
    }
    print(f"  Best bid price range: {prices['best_bid_price_1'].min():.4f} - {prices['best_bid_price_1'].max():.4f}")
    print(f"  Best ask price range: {prices['best_ask_price_1'].min():.4f} - {prices['best_ask_price_1'].max():.4f}")
    
    # Load and process trades (if enabled)
    trades_overlay = None
    if not args.no_trades:
        print(f"\nLoading trades data...")
        trades_df = load_trades(trades_file, start_dt, end_dt)
        if not trades_df.empty:
            print(f"  Trades in window: {len(trades_df):,}")
            # Count by side
            buy_count = (trades_df['side'] == 1).sum()
            sell_count = (trades_df['side'] == 2).sum()
            print(f"    Buy trades: {buy_count:,} (will mark Ask 1 white)")
            print(f"    Sell trades: {sell_count:,} (will mark Bid 1 white)")
            
            # Map trades to snapshot indices
            trades_overlay = map_trades_to_indices(trades_df, df_filtered['datetime_ingest'])
        else:
            print(f"  No trades found in time window")
    
    # Generate output filename
    start_safe = args.start.replace(':', '-')
    end_safe = args.end.replace(':', '-')
    output_file = charts_dir / f"{args.date}_{args.product}_orderbook_heatmap_{start_safe}_{end_safe}.png"
    
    # Plot
    print(f"\nGenerating dual-panel visualization...")
    plot_heatmap(
        quantities,
        df_filtered['datetime_ingest'],
        prices,
        args.date,
        args.product,
        args.start,
        args.end,
        output_file,
        highlight_qtys=highlight_qtys,
        trades_overlay=trades_overlay
    )
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
