#!/usr/bin/env python3
"""
Script to read a .parq file and display its contents as CSV.
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

def clean_column_names(df):
    """Clean up column names by removing CAST functions."""
    new_columns = {}
    for col in df.columns:
        # Remove CAST function wrappers
        if "CAST(" in col and ", '" in col:
            # Extract the actual column name
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

def decode_bytes(df):
    """Convert byte strings to regular strings."""
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if values are bytes
            sample = df[col].dropna()
            if len(sample) > 0 and isinstance(sample.iloc[0], bytes):
                df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    return df

def convert_timestamp(df, time_col='time'):
    """Convert Unix timestamp to datetime."""
    if time_col in df.columns:
        df['datetime'] = pd.to_datetime(df[time_col], unit='s')
        # Reorder columns to put datetime after time
        cols = [c for c in df.columns if c != 'datetime']
        time_idx = cols.index(time_col) if time_col in cols else 0
        cols.insert(time_idx + 1, 'datetime')
        df = df[cols]
    return df

def view_parq_as_csv(file_path, save_csv=False, output_file=None):
    """
    Read a .parq file and display its contents as CSV.
    
    Args:
        file_path: Path to the .parq file
        save_csv: Whether to save the full CSV to a file
        output_file: Output CSV file path (defaults to same name with .csv extension)
    """
    try:
        # Read the parquet file
        print(f"Reading parquet file: {file_path}")
        df = pd.read_parquet(file_path)
        
        # Clean column names
        df = clean_column_names(df)
        
        # Decode byte strings
        df = decode_bytes(df)
        
        # Convert timestamp if time column exists
        time_col = None
        for col in ['time', 'CAST(time, \'Float64\')']:
            if col in df.columns:
                time_col = col
                break
        
        if time_col:
            df = convert_timestamp(df, time_col)
        
        # Display basic info
        print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\nColumn names:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        
        # Display data types
        print(f"\nData types:")
        print(df.dtypes)
        
        # Display first few rows
        print(f"\n{'='*80}")
        print("First 20 rows:")
        print('='*80)
        print(df.head(20).to_csv(index=False))
        
        # Display summary statistics if numeric columns exist
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if 'datetime' in df.columns:
            numeric_cols = [c for c in numeric_cols if c != 'datetime']
        if len(numeric_cols) > 0:
            print(f"\n{'='*80}")
            print("Summary statistics:")
            print('='*80)
            print(df[numeric_cols].describe().to_csv())
        
        # Save CSV if requested
        if save_csv:
            if output_file is None:
                # Save in current working directory (Sandbox folder) with a descriptive name
                source_name = Path(file_path).stem
                output_file = Path.cwd() / f"{source_name}.csv"
            else:
                output_file = Path(output_file)
            print(f"\n{'='*80}")
            print(f"Saving full CSV to: {output_file}")
            df.to_csv(output_file, index=False)
            print(f"Saved {len(df)} rows to {output_file}")
        else:
            print(f"\n{'='*80}")
            print(f"Total rows: {len(df)}")
            print(f"\nTo save full CSV, run:")
            print(f"  python3 view_parq.py {file_path} --save")
            print(f"  or")
            print(f"  python3 view_parq.py {file_path} --save --output output.csv")
        
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Most recent file path
    file_path = "/Users/dasamuel/Data/TradingData/data/raw/oms_logs/2026-01-07__oms_order_log.parq"
    save_csv = False
    output_file = None
    
    # Parse command line arguments
    args = sys.argv[1:]
    if len(args) > 0:
        if args[0] != '--save' and args[0] != '--output':
            file_path = args[0]
            args = args[1:]
    
    if '--save' in args:
        save_csv = True
        save_idx = args.index('--save')
        if '--output' in args:
            output_idx = args.index('--output')
            if output_idx + 1 < len(args):
                output_file = args[output_idx + 1]
    
    view_parq_as_csv(file_path, save_csv=save_csv, output_file=output_file)

