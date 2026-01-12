#!/usr/bin/env python3
"""
Script to extract public market data (orderbook and trades) from Paribu hourly parquet files.
Aggregates all hours for a given date and outputs to CSV files.
"""

import argparse
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime


# Base path for public market data
DATA_BASE_PATH = Path("/Users/dasamuel/Data/MarketData/ParibuData/market-data")
RESULTS_PATH = Path("/Users/dasamuel/Projects/Sandbox/results")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract public market data from Paribu hourly parquet files."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Date to extract in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--product",
        required=True,
        help="Product symbol (e.g., ADA-TL, BTC-TL)"
    )
    parser.add_argument(
        "--type",
        choices=["orderbook", "trades", "both"],
        default="both",
        help="Type of data to extract (default: both)"
    )
    return parser.parse_args()


def validate_date(date_str: str) -> str:
    """Validate date format and convert to folder format YYYYMMDD."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%Y%m%d")
    except ValueError:
        print(f"Error: Invalid date format '{date_str}'. Expected YYYY-MM-DD.")
        sys.exit(1)


def get_product_path(product: str) -> Path:
    """Get the path to the product folder, handling underscore vs hyphen variants."""
    product_path = DATA_BASE_PATH / product
    if product_path.exists():
        return product_path
    
    # Try alternative naming (underscore vs hyphen)
    alt_product = product.replace("-", "_")
    alt_path = DATA_BASE_PATH / alt_product
    if alt_path.exists():
        return alt_path
    
    # Try the reverse
    alt_product = product.replace("_", "-")
    alt_path = DATA_BASE_PATH / alt_product
    if alt_path.exists():
        return alt_path
    
    print(f"Error: Product folder not found for '{product}'")
    print(f"Checked: {product_path}")
    print(f"Available products:")
    for p in sorted(DATA_BASE_PATH.iterdir()):
        if p.is_dir():
            print(f"  - {p.name}")
    sys.exit(1)


def read_hourly_data(data_type_path: Path, date_folder: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Read and concatenate all parquet files for a given date.
    
    Returns:
        tuple: (DataFrame, list of hours with data, list of hours without data)
    """
    date_path = data_type_path / date_folder
    
    if not date_path.exists():
        print(f"Warning: Date folder not found: {date_path}")
        return pd.DataFrame(), [], [f"{h:02d}" for h in range(24)]
    
    all_dfs = []
    hours_with_data = []
    hours_without_data = []
    
    for hour in range(24):
        hour_str = f"{hour:02d}"
        hour_path = date_path / hour_str
        
        if not hour_path.exists():
            hours_without_data.append(hour_str)
            continue
        
        # Find all parquet files in the hour folder
        parquet_files = list(hour_path.glob("*.parquet"))
        
        if not parquet_files:
            hours_without_data.append(hour_str)
            continue
        
        hours_with_data.append(hour_str)
        
        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
                all_dfs.append(df)
            except Exception as e:
                print(f"Warning: Failed to read {pf}: {e}")
    
    if not all_dfs:
        return pd.DataFrame(), hours_with_data, hours_without_data
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    return combined_df, hours_with_data, hours_without_data


def transform_data(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """
    Transform the data:
    - Convert timestamps to human-readable datetime
    - Convert price_e8 and size_e8 to decimal values
    - Sort by ingest timestamp (ts_ingest_unix_us) as primary key for sequence ordering
    
    Note: datetime_exchange and ts_exchange_unix_ms may be incorrect/corrupted and are
    included at the end of each row for later analysis.
    """
    if df.empty:
        return df
    
    # Determine which ingest timestamp column is present
    ingest_ts_col = "ts_ingest_unix_ms" if "ts_ingest_unix_ms" in df.columns else "ts_ingest_unix_us"
    
    # Sort by ingest timestamp (primary key for sequence ordering)
    df = df.sort_values(ingest_ts_col).reset_index(drop=True)
    
    # Handle both ts_ingest_unix_ms (older) and ts_ingest_unix_us (newer) schemas
    if "ts_ingest_unix_ms" in df.columns:
        df["datetime_ingest"] = pd.to_datetime(df["ts_ingest_unix_ms"], unit="ms")
    elif "ts_ingest_unix_us" in df.columns:
        df["datetime_ingest"] = pd.to_datetime(df["ts_ingest_unix_us"], unit="us")
    
    # Convert exchange timestamp to datetime (potentially corrupted, kept for analysis)
    df["datetime_exchange"] = pd.to_datetime(df["ts_exchange_unix_ms"], unit="ms")
    
    # Convert price and size from e8 format to decimal
    df["price"] = df["price_e8"] / 1e8
    df["size"] = df["size_e8"] / 1e8
    
    # Map side values to readable names
    # Based on typical conventions: 1 = buy, 2 = sell
    side_map = {1: "buy", 2: "sell"}
    df["side_name"] = df["side"].map(side_map).fillna("unknown")
    
    # Reorder columns for better readability
    # datetime_ingest and ts_ingest_unix_us are the primary key fields (first)
    # datetime_exchange and ts_exchange_unix_ms are potentially corrupted (at end)
    if data_type == "orderbook":
        cols_order = [
            "datetime_ingest", ingest_ts_col,
            "exchange", "symbol_canonical", "symbol_native",
            "side", "side_name", "price", "size",
            "price_e8", "size_e8",
            "seq_exchange", "snapshot_id", "meta_json",
            "datetime_exchange", "ts_exchange_unix_ms"
        ]
    else:  # trades
        cols_order = [
            "datetime_ingest", ingest_ts_col,
            "exchange", "symbol_canonical", "symbol_native",
            "trade_id", "side", "side_name", "price", "size",
            "price_e8", "size_e8",
            "meta_json",
            "datetime_exchange", "ts_exchange_unix_ms"
        ]
    
    # Only include columns that exist
    cols_order = [c for c in cols_order if c in df.columns]
    
    return df[cols_order]


def consolidate_orderbook(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate orderbook data by grouping rows with the same ts_ingest timestamp
    into single rows showing top 3 bid/ask levels.
    
    Args:
        df: DataFrame with raw orderbook data
        
    Returns:
        DataFrame with consolidated orderbook snapshots
    """
    if df.empty:
        return pd.DataFrame()
    
    # Determine which ingest timestamp column is present
    ingest_ts_col = "ts_ingest_unix_ms" if "ts_ingest_unix_ms" in df.columns else "ts_ingest_unix_us"
    
    consolidated_rows = []
    
    # Group by ingest timestamp
    for ts_value, group in df.groupby(ingest_ts_col):
        # Get the datetime_ingest value (should be same for all rows in group)
        datetime_ingest = group["datetime_ingest"].iloc[0]
        
        # Separate bids (side=1) and asks (side=2)
        bids = group[group["side"] == 1].copy()
        asks = group[group["side"] == 2].copy()
        
        # Sort bids by price descending (highest = best)
        bids = bids.sort_values("price", ascending=False)
        
        # Sort asks by price ascending (lowest = best)
        asks = asks.sort_values("price", ascending=True)
        
        # Extract top 3 levels for bids
        bid_prices = bids["price"].head(3).tolist()
        bid_qtys = bids["size"].head(3).tolist()
        
        # Extract top 3 levels for asks
        ask_prices = asks["price"].head(3).tolist()
        ask_qtys = asks["size"].head(3).tolist()
        
        # Pad with None if less than 3 levels
        while len(bid_prices) < 3:
            bid_prices.append(None)
            bid_qtys.append(None)
        while len(ask_prices) < 3:
            ask_prices.append(None)
            ask_qtys.append(None)
        
        # Create consolidated row
        row = {
            "ts_ingest_unix_us": ts_value,
            "datetime_ingest": datetime_ingest,
            "best_bid_price_1": bid_prices[0],
            "best_bid_price_2": bid_prices[1],
            "best_bid_price_3": bid_prices[2],
            "best_bid_qty_1": bid_qtys[0],
            "best_bid_qty_2": bid_qtys[1],
            "best_bid_qty_3": bid_qtys[2],
            "best_ask_price_1": ask_prices[0],
            "best_ask_price_2": ask_prices[1],
            "best_ask_price_3": ask_prices[2],
            "best_ask_qty_1": ask_qtys[0],
            "best_ask_qty_2": ask_qtys[1],
            "best_ask_qty_3": ask_qtys[2],
        }
        consolidated_rows.append(row)
    
    # Create DataFrame and sort by timestamp
    consolidated_df = pd.DataFrame(consolidated_rows)
    consolidated_df = consolidated_df.sort_values("ts_ingest_unix_us").reset_index(drop=True)
    
    return consolidated_df


def extract_data(product: str, date_str: str, date_folder: str, data_type: str) -> None:
    """Extract data for a specific type (orderbook or trades)."""
    product_path = get_product_path(product)
    data_type_path = product_path / data_type
    
    if not data_type_path.exists():
        print(f"Warning: {data_type} folder not found for product {product}")
        return
    
    print(f"\nExtracting {data_type} data for {product} on {date_str}...")
    
    # Read all hourly data
    df, hours_with_data, hours_without_data = read_hourly_data(data_type_path, date_folder)
    
    # Report data availability
    print(f"  Hours with data: {len(hours_with_data)}/24")
    if hours_without_data:
        print(f"  Missing hours: {', '.join(hours_without_data)}")
    
    if df.empty:
        print(f"  No data found for {data_type}.")
        return
    
    print(f"  Raw records: {len(df):,}")
    
    # Transform the data
    df = transform_data(df, data_type)
    
    # Generate output filename
    output_file = RESULTS_PATH / f"Public_{date_str}_{product}_{data_type}.csv"
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"  Saved {len(df):,} records to: {output_file}")
    
    # Print summary statistics
    if not df.empty:
        print(f"\n  Summary for {data_type}:")
        print(f"    Time range (ingest): {df['datetime_ingest'].min()} to {df['datetime_ingest'].max()}")
        print(f"    Price range: {df['price'].min():.8f} to {df['price'].max():.8f}")
        print(f"    Total size: {df['size'].sum():,.2f}")
        
        if data_type == "trades":
            buy_count = (df["side"] == 1).sum()
            sell_count = (df["side"] == 2).sum()
            print(f"    Buy trades: {buy_count:,}, Sell trades: {sell_count:,}")
        
        # Consolidate orderbook data
        if data_type == "orderbook":
            print(f"\n  Consolidating orderbook snapshots...")
            consolidated_df = consolidate_orderbook(df)
            
            if not consolidated_df.empty:
                # Generate consolidated output filename
                consolidated_file = RESULTS_PATH / f"Public_{date_str}_{product}_{data_type}_consolidated.csv"
                consolidated_df.to_csv(consolidated_file, index=False)
                print(f"  Saved {len(consolidated_df):,} consolidated snapshots to: {consolidated_file}")
                
                # Print consolidated summary
                print(f"\n  Consolidated summary:")
                print(f"    Snapshots: {len(consolidated_df):,}")
                print(f"    Time range: {consolidated_df['datetime_ingest'].min()} to {consolidated_df['datetime_ingest'].max()}")
                
                # Calculate spread statistics
                if consolidated_df["best_bid_price_1"].notna().any() and consolidated_df["best_ask_price_1"].notna().any():
                    spreads = consolidated_df["best_ask_price_1"] - consolidated_df["best_bid_price_1"]
                    valid_spreads = spreads.dropna()
                    if len(valid_spreads) > 0:
                        print(f"    Avg spread: {valid_spreads.mean():.8f}")
                        print(f"    Min spread: {valid_spreads.min():.8f}")
                        print(f"    Max spread: {valid_spreads.max():.8f}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate and convert date
    date_folder = validate_date(args.date)
    
    # Ensure results directory exists
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting public market data:")
    print(f"  Product: {args.product}")
    print(f"  Date: {args.date} (folder: {date_folder})")
    print(f"  Type: {args.type}")
    
    # Extract data based on type
    if args.type in ["orderbook", "both"]:
        extract_data(args.product, args.date, date_folder, "orderbook")
    
    if args.type in ["trades", "both"]:
        extract_data(args.product, args.date, date_folder, "trades")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
