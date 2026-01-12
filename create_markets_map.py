#!/usr/bin/env python3
"""
Create a mapping between BTCTURK and PARIBU market IDs based on shared base currencies.
"""

import csv
from collections import defaultdict


def extract_base_currency(symbol: str) -> str:
    """Extract the base currency from a symbol like 'SPOT.BTC.TRY' -> 'BTC'"""
    parts = symbol.strip('"').split('.')
    if len(parts) >= 2:
        return parts[1]
    return ""


def main():
    # Read market.csv and filter for BTCTURK and PARIBU
    btcturk_markets = {}  # base_currency -> market_id
    paribu_markets = {}   # base_currency -> market_id
    
    with open('market.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            
            market_id = int(row[0])
            exchange_name = row[2].strip('"')
            symbol = row[3]
            
            base_currency = extract_base_currency(symbol)
            
            if exchange_name == "BTCTURK":
                # Keep first occurrence for each base currency
                if base_currency not in btcturk_markets:
                    btcturk_markets[base_currency] = market_id
            elif exchange_name == "PARIBU":
                if base_currency not in paribu_markets:
                    paribu_markets[base_currency] = market_id
    
    # Find common base currencies and create mapping
    common_bases = set(btcturk_markets.keys()) & set(paribu_markets.keys())
    
    # Write output CSV
    output_path = 'results/markets_map.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['base_currency', 'btcturk_exchange_id', 'btcturk_market_id', 
                         'paribu_exchange_id', 'paribu_market_id'])
        
        for base in sorted(common_bases):
            writer.writerow([
                base,
                115,  # BTCTURK exchange_id
                btcturk_markets[base],
                129,  # PARIBU exchange_id
                paribu_markets[base]
            ])
    
    print(f"Found {len(common_bases)} common base currencies between BTCTURK and PARIBU")
    print(f"Output written to {output_path}")
    
    # Print the mapping for verification
    print("\nMapping:")
    for base in sorted(common_bases):
        print(f"  {base}: BTCTURK={btcturk_markets[base]}, PARIBU={paribu_markets[base]}")


if __name__ == "__main__":
    main()
