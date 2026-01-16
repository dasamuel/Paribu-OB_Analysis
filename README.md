# OMS Order Log Analysis Suite

A comprehensive toolkit for analyzing Order Management System (OMS) order log data. This suite provides tools for data inspection, order flow analysis, trade execution metrics, order chain extraction, and market data visualization.

## Table of Contents

- [Quick Start](#quick-start)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Data Quality: Paribu Trade Deduplication](#data-quality-paribu-trade-deduplication)
- [Tools Overview](#tools-overview)
  - [analyze_oms.py](#analyze_omspy---main-analysis-tool)
  - [extract_chains.py](#extract_chainspy---order-chain-extraction)
  - [extract_public_data.py](#extract_public_datapy---public-market-data-extraction)
  - [visualize_orderbook.py](#visualize_orderbookpy---orderbook-heatmap-visualization)
  - [visualize_chains.py](#visualize_chainspy---order-chain-visualization)
  - [visualize_timestamps.py](#visualize_timestampspy---timestamp-analysis-visualization)
  - [find_co_markets.py](#find_co_marketspy---co-traded-market-finder)
  - [create_markets_map.py](#create_markets_mappy---market-mapping-generator)
  - [view_parq.py](#view_parqpy---parquet-file-viewer)
- [Input Data Format](#input-data-format)
- [Output Files](#output-files)
- [Market Reference](#market-reference)
- [Analysis Methodology](#analysis-methodology)
- [Example Workflows](#example-workflows)

---

## Quick Start

```bash
# Analyze an OMS order log
python analyze_oms.py /path/to/oms_order_log.parq

# Extract order chains for a specific exchange/market
python extract_chains.py 2026-01-06 115 134

# Extract public market data (orderbook and trades)
python extract_public_data.py --date 2026-01-06 --product ADA-TL

# Visualize orderbook as dual-panel heatmap (quantities) + price chart
python visualize_orderbook.py 2026-01-06 ADA 12:00:00 12:05:00

# Visualize order chains as time vs price plot
python visualize_chains.py 2026-01-06 115 134 00:00:00 01:00:00

# Visualize timestamp discrepancies (ingest vs exchange time)
python visualize_timestamps.py --date 2026-01-06 --product ADA-TL

# Create cross-exchange market mapping
python create_markets_map.py

# Find co-traded markets across BTCTURK and PARIBU
python find_co_markets.py 2026-01-06

# View a parquet file as CSV
python view_parq.py /path/to/file.parq
```

---

## Requirements

- Python 3.8+
- pandas
- pyarrow (for parquet support)
- numpy
- matplotlib (for chart generation)

Install dependencies:

```bash
pip install pandas pyarrow numpy matplotlib
```

---

## Project Structure

```
.
├── analyze_oms.py          # Main OMS order log analysis tool
├── extract_chains.py       # Order chain extraction with histograms
├── extract_public_data.py  # Public market data extraction (orderbook/trades) with deduplication
├── visualize_orderbook.py  # Orderbook depth heatmap + price visualization
├── visualize_chains.py     # Order chain time vs price visualization
├── visualize_timestamps.py # Timestamp analysis visualization (ingest vs exchange time)
├── view_parq.py            # Parquet file viewer/converter
├── find_co_markets.py      # Find co-traded markets across exchanges
├── create_markets_map.py   # Create market reference mapping
├── market.csv              # Reference file mapping market/exchange IDs to names
├── PARIBU_DEDUPLICATION_BRIEFING.md  # Technical documentation on trade deduplication
├── README.md               # This documentation
└── results/                # Output directory (created automatically)
    ├── markets_map.csv                         # Cross-exchange market ID mapping
    ├── YYYY-MM-DD__oms_order_log_analysis.md   # Analysis reports
    ├── YYYY-MM-DD_order_timing_stats.csv       # Cancellation timing stats
    ├── YYYY-MM-DD_trade_statistics.csv         # Trade volume stats
    ├── YYYY-MM-DD_notional_value_stats.csv     # Notional value stats
    ├── YYYY-MM-DD_oms_order_log.csv            # Converted order log data
    ├── YYYY-MM-DD_co-markets.csv               # Co-traded markets with overlap flags
    ├── YYYY-MM-DD_exchange-X_market-Y_chains_buy.csv   # Buy order chains
    ├── YYYY-MM-DD_exchange-X_market-Y_chains_sell.csv  # Sell order chains
    ├── Public_YYYY-MM-DD_PRODUCT_orderbook.csv         # Public orderbook data
    ├── Public_YYYY-MM-DD_PRODUCT_orderbook_consolidated.csv  # Consolidated orderbook
    ├── Public_YYYY-MM-DD_PRODUCT_trades.csv            # Public trades data
    └── charts/                                 # Generated visualizations
        ├── *_orderbook_heatmap_*.png           # Orderbook depth heatmaps with price charts
        ├── *_chains_visual_*.png               # Order chain time vs price plots
        ├── *_timestamp_scatter.png             # Timestamp comparison scatter plots
        ├── *_timestamp_delay_hist.png          # Ingestion delay histograms
        ├── *_timestamp_delay_timeseries.png    # Delay time series plots
        ├── *_duration_hist.png                 # Chain duration histograms
        └── *_empty_gap_hist.png                # Empty gap histograms
```

---

## Data Quality: Paribu Trade Deduplication

### The Problem

The Paribu cryptocurrency exchange emits trade data in periodic packets. **Critical Issue**: The exchange retransmits the same trades multiple times across different packets. This is a known behavior of their system, not a data capture error.

When aggregating multiple Parquet files without deduplication:
- **Duplicate trades inflate volume and trade counts** - the same trade is counted multiple times
- **Analytics become skewed** - aggregations, OHLCV calculations, and statistics are incorrect
- **Typical duplication rate**: 85-95% of raw records are duplicates

### Key Characteristics of Duplicates

- The same `trade_id` (UUID) appears in multiple packets/files
- All trade data fields remain **identical** across retransmissions:
  - `trade_id`, `ts_exchange_unix_ms`, `price_e8`, `size_e8`, `side`
- **Only difference**: `ts_ingest_unix_us` (packet capture timestamp) differs between retransmissions

### The Solution

The `extract_public_data.py` script automatically handles deduplication:

1. **Deduplication by `trade_id`**: After loading all Parquet files, duplicates are removed keeping the first occurrence
2. **Date filtering by exchange timestamp**: Trades are filtered to only include those with `ts_exchange_unix_ms` on the target date (not ingest date)
3. **Authoritative timestamp**: Output is sorted by `ts_exchange_unix_ms` (the actual trade time), not ingest time
4. **Clean output**: Ingest timestamp columns are dropped from trade output (irrelevant after deduplication)

### Example Results

```
Raw records:     11,100
After dedup:        995  (91% were duplicates!)
After date filter:  976  (19 from adjacent dates removed)
```

### Important Notes

- **Always use `ts_exchange_unix_ms`** for time-based operations on trade data
- **Ignore ingest timestamps** for trades - they only reflect packet capture time
- **Apply deduplication before any aggregation** to ensure accurate metrics
- For detailed technical documentation, see `PARIBU_DEDUPLICATION_BRIEFING.md`

---

## Tools Overview

### `analyze_oms.py` - Main Analysis Tool

Analyzes OMS order log data and generates comprehensive reports with statistics on order timing, trade execution, and notional values.

#### Usage

```bash
python analyze_oms.py /path/to/oms_order_log.parq
python analyze_oms.py /path/to/oms_order_log.csv
```

#### Features

- **Data Loading**: Supports both CSV and Parquet formats with automatic column name cleaning and byte string decoding
- **Timestamp Conversion**: Converts Unix timestamps to human-readable datetime
- **Market Enrichment**: Maps numeric IDs to exchange/product names using `market.csv`
- **Comprehensive Statistics**: Computes order timing, trade volume, and notional value metrics
- **Markdown Report Generation**: Creates structured analysis reports with tables and insights

#### Generated Outputs

1. **Analysis Report** (`*_analysis.md`) - Structured markdown with:
   - Executive Summary (key metrics at a glance)
   - Data Overview (file info, schema, time range)
   - Market Coverage (exchanges and products traded)
   - Order Flow Analysis (status distribution, cancellation timing)
   - Trade Execution (volume and notional value by market)
   - Key Insights (auto-generated findings)

2. **CSV Statistics Files**:
   - `*_order_timing_stats.csv` - Time between order placement and cancellation
   - `*_trade_statistics.csv` - Trade volume statistics
   - `*_notional_value_stats.csv` - Trade values in TRY

---

### `extract_chains.py` - Order Chain Extraction

Extracts order chains from OMS parquet files. A chain is defined as a unique order (by `client_order_id`) that starts with `status=new` and ends with `status=canceled`.

#### Usage

```bash
python extract_chains.py YYYY-MM-DD exchange_id market_id
```

#### Examples

```bash
python extract_chains.py 2026-01-06 115 134    # BTCTURK SPOT.DOT.TRY
python extract_chains.py 2026-01-06 129 263    # PARIBU SPOT.FLOKI.TL
```

#### Features

- **Chain Extraction**: Identifies complete order lifecycles (new → canceled)
- **Side Filtering**: Separates buy and sell orders into distinct files
- **Empty Gap Calculation**: Computes time between consecutive order chains
- **Duration Analysis**: Measures how long each order was active
- **Histogram Generation**: Creates visualizations for duration and empty gap distributions

#### Generated Outputs

1. **Chain CSV Files** (per side):
   - `*_chains_buy.csv` - Buy order chains
   - `*_chains_sell.csv` - Sell order chains

2. **Histogram Charts** (in `results/charts/`):
   - `*_duration_hist.png` - Distribution of chain durations
   - `*_empty_gap_hist.png` - Distribution of gaps between chains

#### Chain Data Columns

| Column | Description |
|--------|-------------|
| `time_open` | Unix timestamp when order was placed (new status) |
| `datetime_open` | Human-readable open time |
| `time_close` | Unix timestamp when order was canceled |
| `datetime_close` | Human-readable close time |
| `duration` | Time in seconds the order was active |
| `empty_gap` | Time in seconds since previous chain closed (-1 for first row) |
| `side` | Order side (1 = buy, -1 = sell) |
| `price` | Order price |
| `initial_qty` | Initial order quantity |

---

### `extract_public_data.py` - Public Market Data Extraction

Extracts public market data (orderbook and trades) from Paribu hourly parquet files. Aggregates all hours for a given date and outputs to CSV files.

**Important**: For trade data, this script automatically handles deduplication and date filtering. See [Data Quality: Paribu Trade Deduplication](#data-quality-paribu-trade-deduplication) for details.

#### Data Source

Public market data is located at:
```
/Users/dasamuel/Data/MarketData/ParibuData/market-data/
```

The data is organized by product, then by data type (orderbook/trades), then by date (YYYYMMDD), then by hour (00-23):
```
{PRODUCT}/
  orderbook/
    {YYYYMMDD}/
      {HH}/
        *.parquet
  trades/
    {YYYYMMDD}/
      {HH}/
        *.parquet
```

#### Usage

```bash
# Extract both orderbook and trades for a product on a date
python extract_public_data.py --date 2026-01-06 --product ADA-TL

# Extract only trades (with automatic deduplication)
python extract_public_data.py --date 2026-01-06 --product ADA-TL --type trades

# Extract only orderbook
python extract_public_data.py --date 2026-01-06 --product ADA-TL --type orderbook
```

#### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--date` | Yes | Date to extract in YYYY-MM-DD format |
| `--product` | Yes | Product symbol (e.g., ADA-TL, BTC-TL, ETH-TL) |
| `--type` | No | Type of data: `orderbook`, `trades`, or `both` (default: both) |

#### Features

- **Hourly Aggregation**: Reads and concatenates all parquet files from hours 00-23
- **Multiple Files Per Hour**: Handles cases where trades have multiple files per hour
- **Schema Compatibility**: Supports both older (`ts_ingest_unix_ms`) and newer (`ts_ingest_unix_us`) schemas
- **Trade Deduplication**: Automatically removes duplicate trades by `trade_id` (see [Data Quality](#data-quality-paribu-trade-deduplication))
- **Date Filtering**: Filters trades to only include those with exchange timestamps on the target date
- **Data Transformation**:
  - Converts timestamps to human-readable datetime
  - Converts `price_e8` and `size_e8` to decimal values (divides by 1e8)
  - Maps side values (1=buy, 2=sell) to readable names
- **Orderbook Consolidation**: Automatically consolidates orderbook snapshots into top-of-book format
- **Error Handling**: Gracefully handles missing dates/hours and corrupted files

#### Generated Outputs

1. **Raw Orderbook** (`Public_YYYY-MM-DD_PRODUCT_orderbook.csv`):
   - All orderbook entries with full detail
   - Sorted by ingest timestamp (sequence ordering)
   - Columns: datetime_ingest, ts_ingest_unix_us, exchange, symbol_canonical, symbol_native, side, side_name, price, size, price_e8, size_e8, seq_exchange, snapshot_id, meta_json, datetime_exchange, ts_exchange_unix_ms

2. **Consolidated Orderbook** (`Public_YYYY-MM-DD_PRODUCT_orderbook_consolidated.csv`):
   - One row per orderbook snapshot showing top 3 bid/ask levels
   - Columns: ts_ingest_unix_us, datetime_ingest, best_bid_price_1, best_bid_price_2, best_bid_price_3, best_bid_qty_1, best_bid_qty_2, best_bid_qty_3, best_ask_price_1, best_ask_price_2, best_ask_price_3, best_ask_qty_1, best_ask_qty_2, best_ask_qty_3
   - Bids sorted by price descending (highest = best_bid_price_1)
   - Asks sorted by price ascending (lowest = best_ask_price_1)

3. **Trades** (`Public_YYYY-MM-DD_PRODUCT_trades.csv`):
   - Deduplicated trade records (unique by `trade_id`)
   - Filtered to target date by exchange timestamp
   - Sorted by exchange timestamp (authoritative trade time)
   - Columns: ts_exchange_unix_ms, datetime_exchange, exchange, symbol_canonical, symbol_native, trade_id, side, side_name, price, size, price_e8, size_e8, meta_json
   - **Note**: Ingest timestamps are not included (irrelevant after deduplication)

#### Available Products

Products include (with hyphen or underscore variants):
- ADA-TL, ARB-TL, AVAX-TL, BTC-TL, DOGE-TL, ETH-TL
- FLOKI-TL, PEPE-TL, SOL-TL, TRUMP-TL, XRP-TL
- And many more...

Run with an invalid product to see the full list of available products.

---

### `visualize_orderbook.py` - Orderbook Heatmap Visualization

Generates dual-panel visualizations of orderbook data: a heatmap showing depth/quantity at each price level on top, and a step-chart of best bid/ask prices on the bottom.

#### Usage

```bash
python visualize_orderbook.py YYYY-MM-DD PRODUCT HH:MM:SS HH:MM:SS [--highlight-qty QTY ...]
```

#### Examples

```bash
# Basic heatmap for a 5-minute window
python visualize_orderbook.py 2026-01-06 ADA 12:00:00 12:05:00

# Highlight specific quantities in black (useful for tracking specific order sizes)
python visualize_orderbook.py 2026-01-06 ADA 12:45:00 12:50:00 --highlight-qty 1600.0

# Highlight multiple quantities
python visualize_orderbook.py 2026-01-06 ADA 00:00:00 00:05:00 --highlight-qty 70.9 1600.0
```

#### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `date` | Yes | Date in YYYY-MM-DD format |
| `product` | Yes | Product symbol (e.g., ADA, BTC, ETH) |
| `start` | Yes | Start time in HH:MM:SS format |
| `end` | Yes | End time in HH:MM:SS format |
| `--highlight-qty` | No | One or more quantity values to highlight in black |

#### Features

- **Dual-Panel Layout**: Top panel shows quantity heatmap, bottom panel shows price chart
- **Synchronized Time Axis**: Both panels share the same x-axis for easy correlation
- **Rainbow Colormap**: Quantities displayed from red (low) to blue (high)
- **Step-Style Price Chart**: Prices shown as horizontal steps with vertical jumps (accurate for orderbook data)
- **Standard Price Orientation**: Ask levels at top (higher prices), Bid levels at bottom
- **Quantity Highlighting**: Optionally mark specific quantities in black to track order patterns
- **Top 3 Levels**: Shows best 3 bid and ask levels (6 total rows in heatmap)

#### Generated Output

- `results/charts/YYYY-MM-DD_PRODUCT_orderbook_heatmap_HH-MM-SS_HH-MM-SS.png`

#### Visualization Layout

```
┌─────────────────────────────────────────┐
│  Orderbook Depth Heatmap                │
│  ┌───────────────────────────────┐      │
│  │ Ask 3  ████████████████████   │ ▲    │
│  │ Ask 2  ████████████████████   │ │    │
│  │ Ask 1  ████████████████████   │ Price│
│  │ ─────────────────────────────  │      │
│  │ Bid 1  ████████████████████   │ │    │
│  │ Bid 2  ████████████████████   │ │    │
│  │ Bid 3  ████████████████████   │ ▼    │
│  └───────────────────────────────┘      │
│  ┌───────────────────────────────┐      │
│  │ Best Ask (red) ───────────    │      │
│  │ Best Bid (green) ─────────    │      │
│  └───────────────────────────────┘      │
│            Time →                       │
└─────────────────────────────────────────┘
```

#### Prerequisites

Requires consolidated orderbook data generated by `extract_public_data.py`:
- `results/Public_YYYY-MM-DD_PRODUCT-TL_orderbook_consolidated.csv`

---

### `visualize_chains.py` - Order Chain Visualization

Visualizes order chains as horizontal lines on a time vs price plot. Buy orders shown in green, sell orders in red.

#### Usage

```bash
python visualize_chains.py YYYY-MM-DD exchange_id market_id HH:MM:SS HH:MM:SS
```

#### Examples

```bash
# Visualize chains for BTCTURK SPOT.DOT.TRY from midnight to 1am
python visualize_chains.py 2026-01-06 115 134 00:00:00 01:00:00

# Visualize chains for a different market and time window
python visualize_chains.py 2026-01-06 115 188 12:00:00 13:30:00
```

#### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `date` | Yes | Date in YYYY-MM-DD format |
| `exchange_id` | Yes | Exchange ID (e.g., 115 for BTCTURK, 129 for PARIBU) |
| `market_id` | Yes | Market ID |
| `start` | Yes | Start time in HH:MM:SS format |
| `end` | Yes | End time in HH:MM:SS format |

#### Features

- **Time vs Price Plot**: Each order chain displayed as a horizontal line at its price level
- **Color Coding**: Buy orders in green, sell orders in red
- **Time Window Filtering**: Chains clipped to display window boundaries
- **Chain Counts**: Legend shows number of buy and sell chains in the window
- **Grid Overlay**: Dotted grid for easy price/time reading

#### Generated Output

- `results/charts/YYYY-MM-DD_exchange-X_market-Y_chains_visual_HH-MM-SS_HH-MM-SS.png`

#### Prerequisites

Requires chain data generated by `extract_chains.py`:
- `results/YYYY-MM-DD_exchange-X_market-Y_chains_buy.csv`
- `results/YYYY-MM-DD_exchange-X_market-Y_chains_sell.csv`

---

### `visualize_timestamps.py` - Timestamp Analysis Visualization

Visualizes the relationship between ingest timestamps and exchange timestamps for public trade data. Helps identify discrepancies and patterns in data ingestion timing.

#### Usage

```bash
python visualize_timestamps.py --date YYYY-MM-DD --product SYMBOL
```

#### Examples

```bash
python visualize_timestamps.py --date 2026-01-06 --product ADA-TL
python visualize_timestamps.py --date 2026-01-07 --product BTC-TL
```

#### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--date` | Yes | Date in YYYY-MM-DD format |
| `--product` | Yes | Product symbol (e.g., ADA-TL, BTC-TL) |

#### Features

- **Scatter Plot**: Exchange timestamp vs ingest timestamp with delay coloring (perfect match = 45° line)
- **Delay Histogram**: Distribution of ingestion delays with full and zoomed (≤5 min) views
- **Time Series Plot**: Delay over time with rolling average, sorted by both exchange and ingest timestamps
- **Statistics Summary**: Comprehensive delay statistics including percentile distribution
- **Warning Detection**: Flags negative delays (exchange timestamp after ingest - should not happen)

#### Generated Outputs

1. **Scatter Plot** (`*_timestamp_scatter.png`):
   - X-axis: Exchange timestamp
   - Y-axis: Ingest timestamp
   - Color: Delay in seconds (red = high, green = low)
   - Reference line: Perfect match (y=x)

2. **Delay Histogram** (`*_timestamp_delay_hist.png`):
   - Left panel: Full distribution
   - Right panel: Zoomed to delays ≤ 5 minutes
   - Mean and median reference lines

3. **Time Series** (`*_timestamp_delay_timeseries.png`):
   - Top panel: Delay over time (by exchange timestamp)
   - Bottom panel: Delay over time (by ingest timestamp)
   - Rolling average overlay
   - Reference lines at 1 min and 5 min

#### Prerequisites

Requires trade data generated by `extract_public_data.py`:
- `results/Public_YYYY-MM-DD_PRODUCT_trades.csv`

**Note**: This tool requires the trade CSV to include `ts_ingest_unix_us` column, which is only present when running `extract_public_data.py` in a mode that preserves ingest timestamps.

---

### `find_co_markets.py` - Co-Traded Market Finder

Finds base currencies traded on both BTCTURK and PARIBU exchanges for a given date. Lists all traded markets with an overlap flag indicating if traded on both exchanges.

#### Usage

```bash
python find_co_markets.py YYYY-MM-DD
```

#### Examples

```bash
python find_co_markets.py 2026-01-06
```

#### Features

- **Cross-Exchange Analysis**: Identifies which base currencies are traded on both BTCTURK and PARIBU
- **Overlap Detection**: Flags markets with `overlap=1` if traded on both exchanges
- **Market ID Mapping**: Uses `markets_map.csv` to translate market IDs to base currencies
- **Summary Output**: Prints console summary and writes detailed CSV

#### Generated Outputs

- `results/YYYY-MM-DD_co-markets.csv`:

| Column | Description |
|--------|-------------|
| `base_currency` | The base currency symbol (e.g., BTC, ETH, ADA) |
| `btcturk_exchange_id` | BTCTURK exchange ID (115) |
| `btcturk_market_id` | BTCTURK market ID for this base currency |
| `paribu_exchange_id` | PARIBU exchange ID (129) |
| `paribu_market_id` | PARIBU market ID for this base currency |
| `overlap` | 1 if traded on both exchanges, 0 otherwise |

#### Prerequisites

Requires:
- `results/markets_map.csv` (generated by `create_markets_map.py`)
- OMS parquet file for the target date at the configured data path

---

### `create_markets_map.py` - Market Mapping Generator

Creates a mapping between BTCTURK and PARIBU market IDs based on shared base currencies. This mapping is used by other tools to correlate markets across exchanges.

#### Usage

```bash
python create_markets_map.py
```

#### Features

- **Symbol Parsing**: Extracts base currency from symbols like `SPOT.BTC.TRY` → `BTC`
- **Cross-Exchange Matching**: Finds base currencies available on both exchanges
- **First-Occurrence Preference**: Uses the first market ID for each base currency per exchange

#### Generated Outputs

- `results/markets_map.csv`:

| Column | Description |
|--------|-------------|
| `base_currency` | The base currency symbol (e.g., BTC, ETH, ADA) |
| `btcturk_exchange_id` | BTCTURK exchange ID (115) |
| `btcturk_market_id` | BTCTURK market ID for this base currency |
| `paribu_exchange_id` | PARIBU exchange ID (129) |
| `paribu_market_id` | PARIBU market ID for this base currency |

#### Prerequisites

Requires:
- `market.csv` in the project root directory

---

### `view_parq.py` - Parquet File Viewer

A utility script to read and display Parquet files as CSV, with optional export functionality.

#### Usage

```bash
# View file contents (first 20 rows)
python view_parq.py /path/to/file.parq

# Save full CSV
python view_parq.py /path/to/file.parq --save

# Save to specific output file
python view_parq.py /path/to/file.parq --save --output output.csv
```

#### Features

- Displays file shape, column names, and data types
- Shows first 20 rows as CSV preview
- Provides summary statistics for numeric columns
- Cleans column names (removes CAST wrappers)
- Decodes byte strings to regular strings
- Converts Unix timestamps to datetime

---

## Input Data Format

The analysis tools accept OMS order log files in CSV or Parquet format with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `time` | float64 | Unix timestamp (seconds since epoch) |
| `oms_id` | int | Order Management System ID |
| `client_order_id` | string | Client order UUID |
| `exchange_id` | int | Exchange identifier |
| `market_id` | int | Market identifier |
| `side` | int | Order side (1 = buy, -1 = sell) |
| `status` | string | Order status (new, canceled, pending_new, rejected, filled, partially_filled) |
| `exchange_order_id` | string | Exchange-assigned order ID |
| `price` | string/float | Order price |
| `initial_qty` | string/float | Initial quantity |
| `qty_remain` | string/float | Remaining quantity |

---

## Output Files

All outputs are saved to the `results/` directory with date prefixes for easy identification.

### Analysis Reports

The markdown analysis report includes:

1. **Executive Summary** - Quick overview of key metrics
2. **Data Overview** - Source file info, record counts, time range, schema
3. **Market Coverage** - List of exchanges and markets with product names
4. **Order Flow Analysis** - Status distribution, side distribution, cancellation timing
5. **Trade Execution** - Volume and notional value breakdown by market
6. **Key Insights** - Auto-generated observations about the data

### CSV Statistics

| File | Description |
|------|-------------|
| `*_order_timing_stats.csv` | Cancellation timing statistics (count, mean, median, std, min, max, Q25, Q75) by exchange and market |
| `*_trade_statistics.csv` | Trade volume statistics for orders where `qty_remain < initial_qty` |
| `*_notional_value_stats.csv` | Trade values in TRY (`traded_qty × price`) |

---

## Market Reference

The `market.csv` file maps market and exchange IDs to human-readable names. Format (no header):

```csv
market_id,exchange_id,exchange_name,product_name
116,115,BTCTURK,SPOT.TRX.TRY
134,115,BTCTURK,SPOT.DOT.TRY
263,129,PARIBU,SPOT.FLOKI.TL
```

Place this file in the same directory as the scripts. If not found, numeric IDs will be used.

### Supported Exchanges

| Exchange ID | Exchange Name |
|-------------|---------------|
| 115 | BTCTURK |
| 129 | PARIBU |

---

## Analysis Methodology

### Order Timing Analysis

For each order that has both 'new' and 'canceled' status:
- Calculates time difference between placement and cancellation
- Groups by exchange and market
- Computes: count, mean, median, std, min, max, Q25, Q75

### Trade Statistics

When `qty_remain` differs from `initial_qty`, a trade occurred:
- `traded_qty = initial_qty - qty_remain`
- Aggregates by exchange and market

### Notional Value

Calculates the TRY value of trades:
- `notional_value = traded_qty × price`
- Useful for understanding actual trading volume in currency terms

### Order Chain Analysis

An order chain represents the complete lifecycle of an order:
1. **Chain Identification**: Match orders with both `status=new` and `status=canceled`
2. **Duration Calculation**: `time_close - time_open`
3. **Empty Gap Calculation**: Time between previous chain's close and current chain's open
4. **Side Separation**: Split into buy and sell for independent analysis

### Histogram Generation

Histograms use the 95th percentile as the maximum x-axis value with 20 bins to focus on the typical distribution while excluding outliers.

---

## Example Workflows

### Full Day Analysis

```bash
# Step 1: Analyze the full order log
python analyze_oms.py /Users/dasamuel/Data/TradingData/data/raw/oms_logs/2026-01-07__oms_order_log.parq

# Step 2: Extract chains for markets of interest
python extract_chains.py 2026-01-07 115 134    # BTCTURK SPOT.DOT.TRY
python extract_chains.py 2026-01-07 129 281    # PARIBU SPOT.XRP.TL
```

### Quick Data Inspection

```bash
# Preview parquet file contents
python view_parq.py /path/to/oms_order_log.parq

# Export to CSV for external analysis
python view_parq.py /path/to/oms_order_log.parq --save --output my_export.csv
```

### Multi-Market Chain Extraction

```bash
# Extract chains for all active BTCTURK markets
for market_id in 116 134 186 188 244; do
    python extract_chains.py 2026-01-06 115 $market_id
done

# Extract chains for PARIBU markets
for market_id in 263 268 281 285 288; do
    python extract_chains.py 2026-01-06 129 $market_id
done
```

### Public Market Data Extraction

```bash
# Extract orderbook and trades for ADA-TL on a specific date
# Trades are automatically deduplicated and filtered to target date
python extract_public_data.py --date 2026-01-06 --product ADA-TL

# Extract only trades for multiple products (with deduplication)
for product in ADA-TL BTC-TL ETH-TL XRP-TL; do
    python extract_public_data.py --date 2026-01-06 --product $product --type trades
done

# Extract orderbook data (includes consolidated top-of-book)
python extract_public_data.py --date 2026-01-06 --product BTC-TL --type orderbook

# Example output showing deduplication:
#   Deduplicated: 11,100 -> 995 trades (10,105 duplicates removed)
#   Date filtered: 995 -> 976 trades (19 from other dates removed)
```

### Orderbook Visualization

```bash
# First, extract and consolidate orderbook data
python extract_public_data.py --date 2026-01-06 --product ADA-TL --type orderbook

# Generate heatmaps for different time windows
python visualize_orderbook.py 2026-01-06 ADA 00:00:00 00:05:00
python visualize_orderbook.py 2026-01-06 ADA 12:00:00 12:05:00
python visualize_orderbook.py 2026-01-06 ADA 12:45:00 12:50:00 --highlight-qty 1600.0

# Generate multiple heatmaps for a trading session
for start in 09:00:00 09:05:00 09:10:00 09:15:00 09:20:00; do
    # Calculate end time (5 minutes later)
    end=$(date -j -v+5M -f "%H:%M:%S" "$start" "+%H:%M:%S" 2>/dev/null || echo "09:05:00")
    python visualize_orderbook.py 2026-01-06 ADA $start $end
done
```

### Order Chain Visualization

```bash
# First, extract chains for the market
python extract_chains.py 2026-01-06 115 134

# Visualize chains for different time windows
python visualize_chains.py 2026-01-06 115 134 00:00:00 01:00:00
python visualize_chains.py 2026-01-06 115 134 12:00:00 13:00:00

# Visualize chains for multiple markets
for market_id in 134 188 244; do
    python extract_chains.py 2026-01-06 115 $market_id
    python visualize_chains.py 2026-01-06 115 $market_id 00:00:00 01:00:00
done
```

### Cross-Exchange Market Analysis

```bash
# Step 1: Generate the cross-exchange market mapping (one-time setup)
python create_markets_map.py

# Step 2: Find co-traded markets for a specific date
python find_co_markets.py 2026-01-06

# Step 3: Extract chains for co-traded markets on both exchanges
# Example: ADA is traded on both BTCTURK (market_id from markets_map) and PARIBU
python extract_chains.py 2026-01-06 115 244    # BTCTURK ADA
python extract_chains.py 2026-01-06 129 293    # PARIBU ADA
```

### Timestamp Analysis

```bash
# First, extract trade data with ingest timestamps preserved
python extract_public_data.py --date 2026-01-06 --product ADA-TL --type trades

# Visualize timestamp discrepancies
python visualize_timestamps.py --date 2026-01-06 --product ADA-TL

# Analyze multiple products
for product in ADA-TL BTC-TL ETH-TL; do
    python extract_public_data.py --date 2026-01-06 --product $product --type trades
    python visualize_timestamps.py --date 2026-01-06 --product $product
done
```

---

## Key Metrics Interpretation

### Trade Execution Rate

A low execution rate (e.g., <1%) typically indicates:
- Market-making activity
- Algorithmic trading strategies
- Quote refreshing behavior

### Cancellation Timing

- **Fast cancellations** (<1s): Likely quote updates or market-making
- **Medium cancellations** (1-10s): Standard order management
- **Slow cancellations** (>10s): May indicate resting orders or manual intervention

### Empty Gap Analysis

- **Negative or zero gaps**: Overlapping orders (multiple orders active simultaneously)
- **Small positive gaps** (<1s): Continuous market presence
- **Large gaps** (>10s): Intermittent trading activity

---

## Notes

- All timestamps are in UTC
- Notional values are calculated in TRY (Turkish Lira)
- The `results/` directory is created automatically if it doesn't exist
- Date prefixes in output filenames are extracted from input filenames or default to the current date
- **Paribu Trade Data**: Raw trade data contains 85-95% duplicates due to exchange retransmission behavior. The `extract_public_data.py` script handles this automatically. See [Data Quality](#data-quality-paribu-trade-deduplication) for details.
- **Trade Timestamps**: For trade analysis, always use `ts_exchange_unix_ms` (authoritative trade time), not ingest timestamps
- For detailed technical documentation on the deduplication process, see `PARIBU_DEDUPLICATION_BRIEFING.md`
