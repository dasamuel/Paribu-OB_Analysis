# Paribu Trade Deduplication Briefing Note

## Context and Problem Statement

### Data Source: Paribu Exchange Trade Data

The Paribu cryptocurrency exchange emits trade data in periodic packets. Each packet contains bundled trades for a specific trading pair (e.g., ADA-TL). These packets are captured and stored as Parquet files.

### The Duplicate Trade Problem

**Critical Issue**: The Paribu exchange retransmits the same trades multiple times across different packets. This is a known behavior of their system, not a data capture error.

**Key Characteristics of Duplicates**:
- The same `trade_id` (UUID) appears in multiple packets/files
- All trade data fields remain identical across retransmissions:
  - `trade_id` (UUID)
  - `ts_exchange_unix_ms` (exchange timestamp in milliseconds)
  - `price_e8` (price scaled by 10^8)
  - `size_e8` (size scaled by 10^8)
  - `side` (1 = buy, 2 = sell)
  - All other trade metadata
- **Only difference**: `ts_ingest_unix_us` (packet capture timestamp) differs between retransmissions
- This means `ts_ingest_datetime` also differs, but this is irrelevant for trade analysis

### Why This Matters

When aggregating multiple Parquet files:
1. **Duplicate trades inflate volume and trade counts** - same trade counted multiple times
2. **Date filtering becomes unreliable** - using ingest timestamps would filter incorrectly
3. **Analytics become skewed** - aggregations, OHLCV calculations, and statistics are wrong

## Data Schema

### Parquet File Schema

Parquet files contain the following columns:

```
exchange              : string (always "paribu")
symbol_canonical      : string (e.g., "ADA-TL")
symbol_native         : string (e.g., "ada_tl")
trade_id              : string (UUID, unique per trade, NOT per packet)
ts_exchange_unix_ms   : int64 (milliseconds since Unix epoch - AUTHORITATIVE timestamp)
ts_ingest_unix_us     : int64 (microseconds since Unix epoch - packet capture time)
price_e8              : int64 (price * 10^8)
size_e8               : int64 (size * 10^8)
side                  : int64 (1 = buy, 2 = sell)
meta_json             : string (optional JSON metadata, often empty)
```

**Note**: Parquet files do NOT contain `ts_exchange_datetime` or `ts_ingest_datetime` columns - these must be derived from the Unix timestamps.

### CSV Export Schema (if exported)

CSV exports may include additional datetime columns:
- `ts_exchange_datetime` : datetime string (derived from `ts_exchange_unix_ms`)
- `ts_ingest_datetime` : datetime string (derived from `ts_ingest_unix_us`)

## Solution: Deduplication Process

### Core Algorithm

1. **Load all Parquet files** from a directory
2. **Concatenate** into a single DataFrame
3. **Deduplicate** by `trade_id` (keep first occurrence)
4. **Filter by date** using `ts_exchange_unix_ms` (convert to date, filter to target date)
5. **Sort** by `ts_exchange_unix_ms` ascending
6. **Drop ingest columns** (`ts_ingest_unix_us`, `ts_ingest_datetime` if present)
7. **Output** deduplicated CSV

### Implementation Details

#### Deduplication Logic

```python
# Drop duplicates on trade_id, keeping the first occurrence
df_deduped = df.drop_duplicates(subset=['trade_id'], keep='first')
```

**Rationale**: `keep='first'` is arbitrary - all duplicates have identical trade data, so which one we keep doesn't matter.

#### Date Filtering Logic

```python
# Convert milliseconds to datetime
df['ts_exchange_datetime'] = pd.to_datetime(df['ts_exchange_unix_ms'], unit='ms')

# Extract date component
df['exchange_date'] = df['ts_exchange_datetime'].dt.date

# Filter to target date
df_filtered = df[df['exchange_date'] == target_date]
```

**Critical**: Always use `ts_exchange_unix_ms` for date filtering, NEVER use ingest timestamps.

#### Column Handling

**Columns to Keep**:
- All trade data columns (exchange, symbol, trade_id, price, size, side, meta_json)
- `ts_exchange_unix_ms` (authoritative timestamp)
- `ts_exchange_datetime` (derived from exchange timestamp)

**Columns to Drop**:
- `ts_ingest_unix_us` (irrelevant after deduplication)
- `ts_ingest_datetime` (irrelevant after deduplication)

## File Naming Convention

Paribu files follow this pattern:
```
paribu_{SYMBOL}_trades_{YYYYMMDD}_{HH}_{TIMESTAMP}.parquet
```

Example:
```
paribu_ADA-TL_trades_20260113_00_639038592395241790.parquet
```

The timestamp in the filename is the packet capture timestamp, not the trade timestamp.

## Usage in Larger Project

### When to Apply Deduplication

Apply this deduplication process whenever:
1. Loading multiple Parquet files from Paribu exchange
2. Aggregating trades across time periods
3. Performing volume/price analysis
4. Building OHLCV (Open/High/Low/Close/Volume) data
5. Calculating statistics or metrics

### Integration Points

1. **Data Ingestion Pipeline**: Add deduplication step after loading Parquet files, before aggregation
2. **ETL Processes**: Include deduplication in transformation layer
3. **Analytics Queries**: Ensure deduplication happens before any GROUP BY or aggregation operations

### Performance Considerations

- **Memory**: Loading all files into memory may be necessary for deduplication
- **Efficiency**: Deduplication is O(n) with pandas `drop_duplicates()` using hash-based approach
- **Scalability**: For very large datasets, consider chunked processing or distributed processing

## Example Results

### Before Deduplication
- 3 Parquet files
- 420 total rows
- Many duplicate `trade_id` values

### After Deduplication
- 51 unique trades
- 33 trades on target date (2026-01-13)
- Clean, deduplicated dataset ready for analysis

## Key Takeaways

1. **Always deduplicate by `trade_id`** when processing Paribu exchange data
2. **Use exchange timestamps (`ts_exchange_unix_ms`)** for all time-based operations
3. **Ignore ingest timestamps** - they only reflect packet capture time, not trade time
4. **Apply deduplication before any aggregation** to ensure accurate metrics
5. **The same trade can appear in multiple files** - this is expected behavior, not an error

## Code Reference

The reference implementation is in `dedup_trades.py`:
- CLI script with `--input-dir`, `--output`, and `--date` arguments
- Handles Parquet loading, deduplication, date filtering, and CSV output
- Includes error handling and progress reporting

## Testing and Validation

To validate deduplication worked correctly:
1. Check that all `trade_id` values are unique in output
2. Verify date filtering uses exchange timestamps, not ingest timestamps
3. Confirm trade counts match expected values (should be lower than raw row counts)
4. Validate that trade data (price, size, side) is consistent for same `trade_id`
