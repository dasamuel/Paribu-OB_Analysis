#!/usr/bin/env python3
"""
OMS Order Log Analysis Tool

Analyzes OMS order log data from CSV or Parquet files and generates:
- A comprehensive markdown analysis report
- Supporting CSV files with detailed statistics

Usage:
    python analyze_oms.py /path/to/oms_order_log.csv
    python analyze_oms.py /path/to/oms_order_log.parq
"""

import pandas as pd
import numpy as np
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

# =============================================================================
# Data Loading & Cleaning
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


def load_data(file_path: Path) -> pd.DataFrame:
    """Load data from CSV or Parquet file."""
    print(f"Loading data from: {file_path}")
    
    if file_path.suffix.lower() == '.parq' or file_path.suffix.lower() == '.parquet':
        df = pd.read_parquet(file_path)
    elif file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Clean and transform
    df = clean_column_names(df)
    df = decode_bytes(df)
    
    # Find time column
    time_col = None
    for col in ['time', 'CAST(time, \'Float64\')']:
        if col in df.columns:
            time_col = col
            break
    
    if time_col:
        df = convert_timestamp(df, time_col)
    
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def load_market_reference(script_dir: Path) -> Optional[pd.DataFrame]:
    """Load market.csv reference file from script directory."""
    market_file = script_dir / 'market.csv'
    if market_file.exists():
        print(f"Loading market reference from: {market_file}")
        # market.csv has no header: market_id, exchange_id, exchange_name, product_name
        market_df = pd.read_csv(market_file, header=None, 
                                names=['market_id', 'exchange_id', 'exchange_name', 'product_name'])
        return market_df
    else:
        print(f"Warning: market.csv not found at {market_file}")
        return None


def enrich_with_market_data(df: pd.DataFrame, market_df: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Dict]:
    """Enrich order data with market names and build lookup dictionaries."""
    market_lookup = {}
    exchange_lookup = {}
    
    if market_df is not None:
        for _, row in market_df.iterrows():
            market_lookup[row['market_id']] = row['product_name']
            exchange_lookup[row['exchange_id']] = row['exchange_name']
    
    return df, {'market': market_lookup, 'exchange': exchange_lookup}


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_basic_stats(df: pd.DataFrame, lookups: Dict) -> Dict:
    """Compute basic statistics about the data."""
    stats = {
        'total_rows': len(df),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
    }
    
    # Time range
    if 'time' in df.columns:
        stats['time_start'] = df['time'].min()
        stats['time_end'] = df['time'].max()
    if 'datetime' in df.columns:
        stats['datetime_start'] = df['datetime'].min()
        stats['datetime_end'] = df['datetime'].max()
    
    # Unique counts
    if 'exchange_id' in df.columns:
        stats['unique_exchanges'] = sorted(df['exchange_id'].unique().tolist())
    if 'market_id' in df.columns:
        stats['unique_markets'] = sorted(df['market_id'].unique().tolist())
    if 'oms_id' in df.columns:
        stats['unique_oms_ids'] = sorted(df['oms_id'].unique().tolist())
    if 'client_order_id' in df.columns:
        stats['unique_client_orders'] = df['client_order_id'].nunique()
    if 'status' in df.columns:
        stats['status_counts'] = df['status'].value_counts().to_dict()
    if 'side' in df.columns:
        stats['side_mean'] = df['side'].mean()
        stats['buy_count'] = (df['side'] == 1).sum()
        stats['sell_count'] = (df['side'] == -1).sum()
    
    # Market-exchange mapping
    if 'exchange_id' in df.columns and 'market_id' in df.columns:
        market_exchange_map = df.groupby('market_id')['exchange_id'].first().to_dict()
        stats['market_exchange_map'] = market_exchange_map
    
    stats['lookups'] = lookups
    return stats


def compute_order_timing_stats(df: pd.DataFrame, lookups: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute timing statistics for orders that have both 'new' and 'canceled' status.
    Returns DataFrame with stats and summary dict.
    """
    if 'client_order_id' not in df.columns or 'status' not in df.columns or 'time' not in df.columns:
        return pd.DataFrame(), {}
    
    # Get new and canceled orders
    new_orders = df[df['status'] == 'new'][['client_order_id', 'time', 'exchange_id', 'market_id']].copy()
    new_orders = new_orders.rename(columns={'time': 'new_time'})
    
    canceled_orders = df[df['status'] == 'canceled'][['client_order_id', 'time']].copy()
    canceled_orders = canceled_orders.rename(columns={'time': 'canceled_time'})
    
    # Merge to find orders with both statuses
    merged = new_orders.merge(canceled_orders, on='client_order_id', how='inner')
    merged['time_diff'] = merged['canceled_time'] - merged['new_time']
    
    # Filter out negative time differences (data issues)
    merged = merged[merged['time_diff'] >= 0]
    
    summary = {
        'total_analyzed': len(merged),
        'unique_client_orders': df['client_order_id'].nunique(),
        'orders_with_new': len(new_orders),
        'orders_with_canceled': len(canceled_orders),
    }
    
    if len(merged) == 0:
        return pd.DataFrame(), summary
    
    # Group by exchange and market
    stats_list = []
    for (exchange_id, market_id), group in merged.groupby(['exchange_id', 'market_id']):
        time_diffs = group['time_diff']
        stats_list.append({
            'exchange_id': exchange_id,
            'market_id': market_id,
            'count': len(time_diffs),
            'mean': time_diffs.mean(),
            'median': time_diffs.median(),
            'std': time_diffs.std(),
            'min': time_diffs.min(),
            'max': time_diffs.max(),
            'q25': time_diffs.quantile(0.25),
            'q75': time_diffs.quantile(0.75),
        })
    
    stats_df = pd.DataFrame(stats_list)
    return stats_df, summary


def compute_trade_stats(df: pd.DataFrame, lookups: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute trade statistics for orders where qty_remain differs from initial_qty.
    Returns DataFrame with stats and summary dict.
    """
    required_cols = ['client_order_id', 'initial_qty', 'qty_remain', 'exchange_id', 'market_id']
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame(), {}
    
    # Convert qty columns to numeric
    df_trades = df.copy()
    df_trades['initial_qty_num'] = pd.to_numeric(df_trades['initial_qty'], errors='coerce')
    df_trades['qty_remain_num'] = pd.to_numeric(df_trades['qty_remain'], errors='coerce')
    
    # Calculate traded quantity
    df_trades['traded_qty'] = df_trades['initial_qty_num'] - df_trades['qty_remain_num']
    
    # Filter for actual trades (traded_qty > 0)
    trades = df_trades[df_trades['traded_qty'] > 0].copy()
    
    summary = {
        'unique_orders_with_trades': trades['client_order_id'].nunique(),
        'total_trade_records': len(trades),
    }
    
    if len(trades) == 0:
        return pd.DataFrame(), summary
    
    # Group by exchange and market
    stats_list = []
    for (exchange_id, market_id), group in trades.groupby(['exchange_id', 'market_id']):
        traded_qty = group['traded_qty']
        stats_list.append({
            'exchange_id': exchange_id,
            'market_id': market_id,
            'trade_count': len(traded_qty),
            'total_volume': traded_qty.sum(),
            'mean': traded_qty.mean(),
            'median': traded_qty.median(),
            'std': traded_qty.std(),
            'min': traded_qty.min(),
            'max': traded_qty.max(),
            'q25': traded_qty.quantile(0.25),
            'q75': traded_qty.quantile(0.75),
        })
    
    stats_df = pd.DataFrame(stats_list)
    
    # Add exchange totals to summary
    if 'exchange_id' in stats_df.columns:
        for exchange_id in stats_df['exchange_id'].unique():
            exchange_data = stats_df[stats_df['exchange_id'] == exchange_id]
            summary[f'exchange_{exchange_id}_trades'] = exchange_data['trade_count'].sum()
    
    return stats_df, summary


def compute_notional_stats(df: pd.DataFrame, lookups: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute notional value statistics (traded_qty × price) for trades.
    Returns DataFrame with stats and summary dict.
    """
    required_cols = ['client_order_id', 'initial_qty', 'qty_remain', 'price', 'exchange_id', 'market_id']
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame(), {}
    
    # Convert columns to numeric
    df_trades = df.copy()
    df_trades['initial_qty_num'] = pd.to_numeric(df_trades['initial_qty'], errors='coerce')
    df_trades['qty_remain_num'] = pd.to_numeric(df_trades['qty_remain'], errors='coerce')
    df_trades['price_num'] = pd.to_numeric(df_trades['price'], errors='coerce')
    
    # Calculate traded quantity and notional value
    df_trades['traded_qty'] = df_trades['initial_qty_num'] - df_trades['qty_remain_num']
    df_trades['notional_value'] = df_trades['traded_qty'] * df_trades['price_num']
    
    # Filter for actual trades
    trades = df_trades[(df_trades['traded_qty'] > 0) & (df_trades['notional_value'] > 0)].copy()
    
    summary = {
        'total_notional': trades['notional_value'].sum() if len(trades) > 0 else 0,
    }
    
    if len(trades) == 0:
        return pd.DataFrame(), summary
    
    # Group by exchange and market
    stats_list = []
    for (exchange_id, market_id), group in trades.groupby(['exchange_id', 'market_id']):
        notional = group['notional_value']
        stats_list.append({
            'exchange_id': exchange_id,
            'market_id': market_id,
            'trade_count': len(notional),
            'total_notional': notional.sum(),
            'mean': notional.mean(),
            'median': notional.median(),
            'std': notional.std(),
            'min': notional.min(),
            'max': notional.max(),
            'q25': notional.quantile(0.25),
            'q75': notional.quantile(0.75),
        })
    
    stats_df = pd.DataFrame(stats_list)
    
    # Add exchange totals to summary
    if 'exchange_id' in stats_df.columns:
        for exchange_id in stats_df['exchange_id'].unique():
            exchange_data = stats_df[stats_df['exchange_id'] == exchange_id]
            summary[f'exchange_{exchange_id}_notional'] = exchange_data['total_notional'].sum()
    
    return stats_df, summary


# =============================================================================
# Report Generation
# =============================================================================

def format_number(n: float, decimals: int = 2) -> str:
    """Format number with K/M suffix for readability."""
    if pd.isna(n):
        return "N/A"
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:,.{decimals}f}M"
    elif abs(n) >= 1_000:
        return f"{n/1_000:,.{decimals}f}K"
    else:
        return f"{n:,.{decimals}f}"


def get_product_name(market_id: int, lookups: Dict) -> str:
    """Get product name from market_id, return 'Unknown' if not found."""
    return lookups.get('market', {}).get(market_id, 'Unknown')


def get_exchange_name(exchange_id: int, lookups: Dict) -> str:
    """Get exchange name from exchange_id, return the ID if not found."""
    return lookups.get('exchange', {}).get(exchange_id, f'Exchange {exchange_id}')


def generate_report(
    input_file: Path,
    basic_stats: Dict,
    timing_stats: pd.DataFrame,
    timing_summary: Dict,
    trade_stats: pd.DataFrame,
    trade_summary: Dict,
    notional_stats: pd.DataFrame,
    notional_summary: Dict,
) -> str:
    """Generate the markdown analysis report."""
    
    lookups = basic_stats.get('lookups', {'market': {}, 'exchange': {}})
    lines = []
    
    # ==========================================================================
    # Title
    # ==========================================================================
    lines.append(f"# OMS Order Log Analysis Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # ==========================================================================
    # Executive Summary
    # ==========================================================================
    lines.append("## Executive Summary")
    lines.append("")
    
    total_notional = notional_summary.get('total_notional', 0)
    total_trades = trade_summary.get('unique_orders_with_trades', 0)
    total_orders = basic_stats.get('unique_client_orders', 0)
    trade_rate = (total_trades / total_orders * 100) if total_orders > 0 else 0
    
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Records | {basic_stats.get('total_rows', 0):,} |")
    lines.append(f"| Unique Orders | {total_orders:,} |")
    lines.append(f"| Orders Executed | {total_trades:,} ({trade_rate:.2f}%) |")
    lines.append(f"| Total Notional Value | {format_number(total_notional)} TRY |")
    lines.append(f"| Exchanges | {len(basic_stats.get('unique_exchanges', []))} |")
    lines.append(f"| Markets | {len(basic_stats.get('unique_markets', []))} |")
    lines.append("")
    
    # ==========================================================================
    # Data Overview
    # ==========================================================================
    lines.append("## Data Overview")
    lines.append("")
    
    lines.append("### File Information")
    lines.append("")
    lines.append(f"- **Source File:** `{input_file}`")
    lines.append(f"- **Total Records:** {basic_stats.get('total_rows', 0):,} rows")
    lines.append(f"- **Columns:** {len(basic_stats.get('columns', []))}")
    lines.append("")
    
    # Time range
    if 'datetime_start' in basic_stats and 'datetime_end' in basic_stats:
        lines.append("### Time Range")
        lines.append("")
        lines.append(f"- **Start:** {basic_stats['datetime_start']}")
        lines.append(f"- **End:** {basic_stats['datetime_end']}")
        lines.append("")
    
    # Schema
    lines.append("### Data Schema")
    lines.append("")
    lines.append("| Column | Type |")
    lines.append("|--------|------|")
    for col, dtype in basic_stats.get('dtypes', {}).items():
        lines.append(f"| `{col}` | {dtype} |")
    lines.append("")
    
    # ==========================================================================
    # Market Coverage
    # ==========================================================================
    lines.append("## Market Coverage")
    lines.append("")
    
    # Exchange summary
    exchanges = basic_stats.get('unique_exchanges', [])
    market_exchange_map = basic_stats.get('market_exchange_map', {})
    
    for exchange_id in exchanges:
        exchange_name = get_exchange_name(exchange_id, lookups)
        exchange_markets = [m for m, e in market_exchange_map.items() if e == exchange_id]
        
        lines.append(f"### {exchange_name} (ID: {exchange_id})")
        lines.append("")
        lines.append(f"**Markets:** {len(exchange_markets)}")
        lines.append("")
        
        if exchange_markets:
            lines.append("| Market ID | Product |")
            lines.append("|-----------|---------|")
            for market_id in sorted(exchange_markets):
                product = get_product_name(market_id, lookups)
                lines.append(f"| {market_id} | {product} |")
            lines.append("")
    
    # ==========================================================================
    # Order Flow Analysis
    # ==========================================================================
    lines.append("## Order Flow Analysis")
    lines.append("")
    
    # Status distribution
    status_counts = basic_stats.get('status_counts', {})
    if status_counts:
        lines.append("### Order Status Distribution")
        lines.append("")
        lines.append("| Status | Count | Percentage |")
        lines.append("|--------|-------|------------|")
        total = sum(status_counts.values())
        for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100 if total > 0 else 0
            lines.append(f"| {status} | {count:,} | {pct:.1f}% |")
        lines.append("")
    
    # Side distribution
    if 'buy_count' in basic_stats:
        lines.append("### Order Side Distribution")
        lines.append("")
        buy = basic_stats.get('buy_count', 0)
        sell = basic_stats.get('sell_count', 0)
        total_sides = buy + sell
        lines.append(f"- **Buy Orders:** {buy:,} ({buy/total_sides*100:.1f}%)" if total_sides > 0 else f"- **Buy Orders:** {buy:,}")
        lines.append(f"- **Sell Orders:** {sell:,} ({sell/total_sides*100:.1f}%)" if total_sides > 0 else f"- **Sell Orders:** {sell:,}")
        lines.append("")
    
    # Order timing statistics
    if len(timing_stats) > 0:
        lines.append("### Order Cancellation Timing")
        lines.append("")
        lines.append(f"**Analysis:** Time between order placement ('new') and cancellation ('canceled')")
        lines.append("")
        lines.append(f"- **Orders Analyzed:** {timing_summary.get('total_analyzed', 0):,}")
        lines.append("")
        
        for exchange_id in exchanges:
            exchange_name = get_exchange_name(exchange_id, lookups)
            exchange_data = timing_stats[timing_stats['exchange_id'] == exchange_id]
            
            if len(exchange_data) == 0:
                continue
            
            lines.append(f"#### {exchange_name}")
            lines.append("")
            lines.append("| Market | Product | Count | Mean (s) | Median (s) | Min (s) | Max (s) |")
            lines.append("|--------|---------|-------|----------|------------|---------|---------|")
            
            for _, row in exchange_data.sort_values('market_id').iterrows():
                product = get_product_name(int(row['market_id']), lookups)
                lines.append(
                    f"| {int(row['market_id'])} | {product} | {int(row['count']):,} | "
                    f"{row['mean']:.2f} | {row['median']:.2f} | {row['min']:.2f} | {row['max']:.2f} |"
                )
            
            # Exchange summary
            total_count = exchange_data['count'].sum()
            avg_mean = (exchange_data['mean'] * exchange_data['count']).sum() / total_count if total_count > 0 else 0
            lines.append("")
            lines.append(f"**{exchange_name} Summary:** {int(total_count):,} orders, avg cancellation time: {avg_mean:.2f}s")
            lines.append("")
    
    # ==========================================================================
    # Trade Execution
    # ==========================================================================
    lines.append("## Trade Execution")
    lines.append("")
    
    if len(trade_stats) > 0:
        lines.append("### Trade Volume by Market")
        lines.append("")
        lines.append(f"**Trade Execution Rate:** {total_trades:,} / {total_orders:,} orders ({trade_rate:.2f}%)")
        lines.append("")
        
        for exchange_id in exchanges:
            exchange_name = get_exchange_name(exchange_id, lookups)
            exchange_data = trade_stats[trade_stats['exchange_id'] == exchange_id]
            
            if len(exchange_data) == 0:
                continue
            
            lines.append(f"#### {exchange_name}")
            lines.append("")
            lines.append("| Market | Product | Trades | Total Volume | Mean | Median | Max |")
            lines.append("|--------|---------|--------|--------------|------|--------|-----|")
            
            for _, row in exchange_data.sort_values('market_id').iterrows():
                product = get_product_name(int(row['market_id']), lookups)
                lines.append(
                    f"| {int(row['market_id'])} | {product} | {int(row['trade_count']):,} | "
                    f"{format_number(row['total_volume'])} | {format_number(row['mean'])} | "
                    f"{format_number(row['median'])} | {format_number(row['max'])} |"
                )
            
            total_trades_ex = exchange_data['trade_count'].sum()
            total_volume = exchange_data['total_volume'].sum()
            lines.append("")
            lines.append(f"**{exchange_name} Summary:** {int(total_trades_ex):,} trades, total volume: {format_number(total_volume)} units")
            lines.append("")
    
    # Notional value
    if len(notional_stats) > 0:
        lines.append("### Notional Value by Market")
        lines.append("")
        lines.append(f"**Total Notional Value:** {format_number(total_notional)} TRY")
        lines.append("")
        
        for exchange_id in exchanges:
            exchange_name = get_exchange_name(exchange_id, lookups)
            exchange_data = notional_stats[notional_stats['exchange_id'] == exchange_id]
            
            if len(exchange_data) == 0:
                continue
            
            exchange_total = exchange_data['total_notional'].sum()
            lines.append(f"#### {exchange_name} ({format_number(exchange_total)} TRY)")
            lines.append("")
            lines.append("| Market | Product | Trades | Total (TRY) | Mean (TRY) | Median (TRY) | Max (TRY) |")
            lines.append("|--------|---------|--------|-------------|------------|--------------|-----------|")
            
            for _, row in exchange_data.sort_values('total_notional', ascending=False).iterrows():
                product = get_product_name(int(row['market_id']), lookups)
                lines.append(
                    f"| {int(row['market_id'])} | {product} | {int(row['trade_count']):,} | "
                    f"{format_number(row['total_notional'])} | {format_number(row['mean'])} | "
                    f"{format_number(row['median'])} | {format_number(row['max'])} |"
                )
            lines.append("")
    
    # ==========================================================================
    # Key Insights
    # ==========================================================================
    lines.append("## Key Insights")
    lines.append("")
    
    insights = []
    
    # Trade execution rate insight
    if trade_rate < 1:
        insights.append(f"**High Cancellation Rate:** Only {trade_rate:.2f}% of orders resulted in trades, indicating market-making or algorithmic trading activity.")
    
    # Exchange comparison
    if len(exchanges) >= 2 and len(timing_stats) > 0:
        exchange_timing = {}
        for exchange_id in exchanges:
            exchange_data = timing_stats[timing_stats['exchange_id'] == exchange_id]
            if len(exchange_data) > 0:
                total_count = exchange_data['count'].sum()
                avg_time = (exchange_data['mean'] * exchange_data['count']).sum() / total_count if total_count > 0 else 0
                exchange_timing[exchange_id] = avg_time
        
        if len(exchange_timing) >= 2:
            fastest = min(exchange_timing, key=exchange_timing.get)
            slowest = max(exchange_timing, key=exchange_timing.get)
            insights.append(
                f"**Exchange Timing:** {get_exchange_name(fastest, lookups)} has faster average cancellation times "
                f"({exchange_timing[fastest]:.2f}s) compared to {get_exchange_name(slowest, lookups)} ({exchange_timing[slowest]:.2f}s)."
            )
    
    # Notional distribution
    if len(notional_stats) > 0 and len(exchanges) >= 2:
        exchange_notionals = {}
        for exchange_id in exchanges:
            exchange_data = notional_stats[notional_stats['exchange_id'] == exchange_id]
            exchange_notionals[exchange_id] = exchange_data['total_notional'].sum()
        
        total = sum(exchange_notionals.values())
        if total > 0:
            for exchange_id, notional in sorted(exchange_notionals.items(), key=lambda x: -x[1]):
                pct = notional / total * 100
                insights.append(
                    f"**{get_exchange_name(exchange_id, lookups)} Volume:** {format_number(notional)} TRY ({pct:.1f}% of total)"
                )
    
    # Top market by notional
    if len(notional_stats) > 0:
        top_market = notional_stats.loc[notional_stats['total_notional'].idxmax()]
        product = get_product_name(int(top_market['market_id']), lookups)
        exchange = get_exchange_name(int(top_market['exchange_id']), lookups)
        insights.append(
            f"**Top Market by Value:** {product} on {exchange} with {format_number(top_market['total_notional'])} TRY"
        )
    
    for i, insight in enumerate(insights, 1):
        lines.append(f"{i}. {insight}")
    lines.append("")
    
    # ==========================================================================
    # Output Files
    # ==========================================================================
    lines.append("## Output Files")
    lines.append("")
    lines.append("The following CSV files contain detailed statistics:")
    lines.append("")
    lines.append("- `order_timing_stats.csv` - Cancellation timing statistics by exchange and market")
    lines.append("- `trade_statistics.csv` - Trade volume statistics by exchange and market")
    lines.append("- `notional_value_stats.csv` - Notional value statistics by exchange and market")
    lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze OMS order log data and generate reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_oms.py /path/to/oms_order_log.csv
    python analyze_oms.py /path/to/oms_order_log.parq
        """
    )
    parser.add_argument('input_file', type=str, help='Path to input CSV or Parquet file')
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file).resolve()
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Determine output directory (current working directory / results)
    output_dir = Path.cwd() / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Extract date from input filename if possible (e.g., 2026-01-07__oms_order_log)
    input_stem = input_file.stem
    date_prefix = ""
    if "__" in input_stem:
        date_prefix = input_stem.split("__")[0]  # e.g., "2026-01-07"
    else:
        # Fallback to today's date
        date_prefix = datetime.now().strftime("%Y-%m-%d")
    
    # Output filenames with date prefix
    report_file = output_dir / f"{input_stem}_analysis.md"
    timing_csv = output_dir / f"{date_prefix}_order_timing_stats.csv"
    trade_csv = output_dir / f"{date_prefix}_trade_statistics.csv"
    notional_csv = output_dir / f"{date_prefix}_notional_value_stats.csv"
    data_csv = output_dir / f"{date_prefix}_oms_order_log.csv"
    
    # Get script directory for market.csv
    script_dir = Path(__file__).parent.resolve()
    
    print("=" * 60)
    print("OMS Order Log Analysis")
    print("=" * 60)
    
    # Load data
    df = load_data(input_file)
    
    # Load market reference
    market_df = load_market_reference(script_dir)
    
    # Enrich data
    df, lookups = enrich_with_market_data(df, market_df)
    
    # Compute statistics
    print("\nComputing basic statistics...")
    basic_stats = compute_basic_stats(df, lookups)
    
    print("Computing order timing statistics...")
    timing_stats, timing_summary = compute_order_timing_stats(df, lookups)
    
    print("Computing trade statistics...")
    trade_stats, trade_summary = compute_trade_stats(df, lookups)
    
    print("Computing notional value statistics...")
    notional_stats, notional_summary = compute_notional_stats(df, lookups)
    
    # Save CSV files
    print("\nSaving CSV files...")
    
    # Save raw data as CSV
    df.to_csv(data_csv, index=False)
    print(f"  Saved: {data_csv}")
    
    if len(timing_stats) > 0:
        timing_stats.to_csv(timing_csv, index=False)
        print(f"  Saved: {timing_csv}")
    
    if len(trade_stats) > 0:
        trade_stats.to_csv(trade_csv, index=False)
        print(f"  Saved: {trade_csv}")
    
    if len(notional_stats) > 0:
        notional_stats.to_csv(notional_csv, index=False)
        print(f"  Saved: {notional_csv}")
    
    # Generate report
    print("\nGenerating analysis report...")
    report = generate_report(
        input_file,
        basic_stats,
        timing_stats, timing_summary,
        trade_stats, trade_summary,
        notional_stats, notional_summary,
    )
    
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"  Saved: {report_file}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

