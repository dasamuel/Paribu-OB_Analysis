#!/usr/bin/env python3
"""
Find base currencies traded on BTCTURK and PARIBU for a given date.
Lists all traded markets with an 'overlap' flag indicating if traded on both exchanges.

Usage:
    python find_co_markets.py YYYY-MM-DD
    
Example:
    python find_co_markets.py 2026-01-06
"""

import sys
import csv
import pandas as pd
from pathlib import Path


# Constants
BTCTURK_EXCHANGE_ID = 115
PARIBU_EXCHANGE_ID = 129
PARQUET_DIR = Path("/Users/dasamuel/Data/TradingData/data/raw/oms_logs")
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
MARKETS_MAP_FILE = RESULTS_DIR / "markets_map.csv"


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up column names by removing CAST functions."""
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


def load_markets_map() -> dict:
    """
    Load the markets_map.csv and return a lookup structure.
    Returns: dict mapping (exchange_id, market_id) -> base_currency
    """
    market_to_base = {}
    
    with open(MARKETS_MAP_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            base = row['base_currency']
            btcturk_market_id = int(row['btcturk_market_id'])
            paribu_market_id = int(row['paribu_market_id'])
            
            market_to_base[(BTCTURK_EXCHANGE_ID, btcturk_market_id)] = base
            market_to_base[(PARIBU_EXCHANGE_ID, paribu_market_id)] = base
    
    return market_to_base


def load_parquet(date_str: str) -> pd.DataFrame:
    """Load parquet file for the given date."""
    file_path = PARQUET_DIR / f"{date_str}__oms_order_log.parq"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
    
    print(f"Loading data from: {file_path}")
    df = pd.read_parquet(file_path)
    
    # Clean and transform
    df = clean_column_names(df)
    df = decode_bytes(df)
    
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def find_co_markets(date_str: str) -> None:
    """
    Find base currencies traded on BTCTURK and PARIBU for the given date.
    Lists all traded markets with overlap flag (1 if traded on both, 0 otherwise).
    """
    # Load the markets map
    if not MARKETS_MAP_FILE.exists():
        print(f"Error: Markets map file not found: {MARKETS_MAP_FILE}")
        print("Please run create_markets_map.py first.")
        sys.exit(1)
    
    market_to_base = load_markets_map()
    print(f"Loaded {len(market_to_base)} market-to-base mappings")
    
    # Load base -> market_id maps from markets_map
    base_to_btcturk = {}
    base_to_paribu = {}
    
    with open(MARKETS_MAP_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            base = row['base_currency']
            base_to_btcturk[base] = int(row['btcturk_market_id'])
            base_to_paribu[base] = int(row['paribu_market_id'])
    
    # Load the parquet file
    df = load_parquet(date_str)
    
    # Filter for BTCTURK and PARIBU exchanges
    btcturk_df = df[df['exchange_id'] == BTCTURK_EXCHANGE_ID]
    paribu_df = df[df['exchange_id'] == PARIBU_EXCHANGE_ID]
    
    print(f"\nBTCTURK orders: {len(btcturk_df):,}")
    print(f"PARIBU orders: {len(paribu_df):,}")
    
    # Get unique market_ids traded on each exchange
    btcturk_markets = set(btcturk_df['market_id'].unique())
    paribu_markets = set(paribu_df['market_id'].unique())
    
    print(f"\nBTCTURK unique markets: {len(btcturk_markets)}")
    print(f"PARIBU unique markets: {len(paribu_markets)}")
    
    # Find base currencies traded on each exchange (that are in the map)
    btcturk_bases = set()
    for market_id in btcturk_markets:
        key = (BTCTURK_EXCHANGE_ID, market_id)
        if key in market_to_base:
            btcturk_bases.add(market_to_base[key])
    
    paribu_bases = set()
    for market_id in paribu_markets:
        key = (PARIBU_EXCHANGE_ID, market_id)
        if key in market_to_base:
            paribu_bases.add(market_to_base[key])
    
    # All traded base currencies (union of both exchanges)
    all_traded_bases = btcturk_bases | paribu_bases
    common_bases = btcturk_bases & paribu_bases
    
    print(f"\nBTCTURK base currencies (in map): {len(btcturk_bases)}")
    print(f"PARIBU base currencies (in map): {len(paribu_bases)}")
    print(f"Total unique base currencies traded: {len(all_traded_bases)}")
    print(f"Common base currencies (overlap=1): {len(common_bases)}")
    
    # Write output - all traded markets with overlap flag
    output_file = RESULTS_DIR / f"{date_str}_co-markets.csv"
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['base_currency', 'btcturk_exchange_id', 'btcturk_market_id', 
                         'paribu_exchange_id', 'paribu_market_id', 'overlap'])
        
        for base in sorted(all_traded_bases):
            overlap = 1 if base in common_bases else 0
            writer.writerow([
                base,
                BTCTURK_EXCHANGE_ID,
                base_to_btcturk[base],
                PARIBU_EXCHANGE_ID,
                base_to_paribu[base],
                overlap
            ])
    
    print(f"\nOutput written to: {output_file}")
    
    # Print summary
    print(f"\nMarkets traded on {date_str}:")
    for base in sorted(all_traded_bases):
        overlap = 1 if base in common_bases else 0
        traded_on = []
        if base in btcturk_bases:
            traded_on.append(f"BTCTURK={base_to_btcturk[base]}")
        if base in paribu_bases:
            traded_on.append(f"PARIBU={base_to_paribu[base]}")
        print(f"  {base}: {', '.join(traded_on)} (overlap={overlap})")


def main():
    if len(sys.argv) != 2:
        print("Usage: python find_co_markets.py YYYY-MM-DD")
        print("Example: python find_co_markets.py 2026-01-06")
        sys.exit(1)
    
    date_str = sys.argv[1]
    
    # Validate date format
    try:
        from datetime import datetime
        datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        print(f"Error: Invalid date format '{date_str}'. Expected YYYY-MM-DD")
        sys.exit(1)
    
    find_co_markets(date_str)


if __name__ == "__main__":
    main()
