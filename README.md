# Paribu Order Book Analysis Suite

A comprehensive toolkit for analyzing cryptocurrency order book data, order flow, trade execution metrics, and market-making performance on Turkish exchanges (BTCTURK and PARIBU).

## Table of Contents

- [Quick Start](#quick-start)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Directory Organization](#directory-organization)
- [Data Quality: Paribu Trade Deduplication](#data-quality-paribu-trade-deduplication)
- [Tools Overview](#tools-overview)
  - [analyze_oms.py](#analyze_omspy---oms-order-log-analysis)
  - [extract_chains.py](#extract_chainspy---order-chain-extraction)
  - [extract_public_data.py](#extract_public_datapy---public-market-data-extraction)
  - [calculate_level1_time.py](#calculate_level1_timepy---level-1-time-analysis)
  - [visualize_orderbook.py](#visualize_orderbookpy---orderbook-heatmap-visualization)
  - [visualize_chains.py](#visualize_chainspy---order-chain-visualization)
  - [visualize_timestamps.py](#visualize_timestampspy---timestamp-analysis-visualization)
  - [find_co_markets.py](#find_co_marketspy---co-traded-market-finder)
  - [create_markets_map.py](#create_markets_mappy---market-mapping-generator)
  - [view_parq.py](#view_parqpy---parquet-file-viewer)
  - [results_path.py](#results_pathpy---results-directory-utility)
  - [config.py](#configpy---configuration-loader)
- [Input Data Format](#input-data-format)
- [Output Files](#output-files)
- [Market Reference](#market-reference)
- [Analysis Methodology](#analysis-methodology)
- [Example Workflows](#example-workflows)

---

## Quick Start

```bash
# 1. Configure data paths (one-time setup)
cp config.yaml.example config.yaml
# Edit config.yaml with your data directory paths

# 2. Create cross-exchange market mapping (one-time setup)
python create_markets_map.py

# 3. Extract public market data (orderbook and trades)
python extract_public_data.py --date 2026-01-16 --product DOGE-TL

# 4. Extract order chains for a specific exchange/market
python extract_chains.py 2026-01-16 129 270

# 5. Calculate Level 1 time and trade attribution
python calculate_level1_time.py 2026-01-16 DOGE-TL

# 6. Visualize orderbook as heatmap with price chart
python visualize_orderbook.py 2026-01-16 DOGE 00:00:00 00:05:00

# 7. Analyze OMS order logs
python analyze_oms.py /path/to/oms_order_log.parq
```

---

## Requirements

- Python 3.8+
- pandas
- pyarrow (for parquet support)
- numpy
- matplotlib (for chart generation)
- pyyaml (for configuration)

Install dependencies:

```bash
pip install pandas pyarrow numpy matplotlib pyyaml
```

---

## Project Structure

```
.
├── config.yaml             # Local configuration (data paths) - create from example
├── config.yaml.example     # Configuration template
├── config.py               # Configuration loader module
├── analyze_oms.py          # OMS order log analysis tool
├── extract_chains.py       # Order chain extraction with histograms
├── extract_public_data.py  # Public market data extraction with deduplication
├── calculate_level1_time.py # Level 1 time, trade attribution & fake trade detection
├── visualize_orderbook.py  # Orderbook depth heatmap + price visualization
├── visualize_chains.py     # Order chain time vs price visualization
├── visualize_timestamps.py # Timestamp analysis visualization
├── view_parq.py            # Parquet file viewer/converter
├── find_co_markets.py      # Find co-traded markets across exchanges
├── create_markets_map.py   # Create market reference mapping
├── results_path.py         # Results directory path utility module
├── market.csv              # Reference file mapping market/exchange IDs to names
├── PARIBU_DEDUPLICATION_BRIEFING.md  # Technical documentation on trade deduplication
├── README.md               # This documentation
└── results/                # Output directory (auto-created)
```

---

## Configuration

Before running the analysis tools, you need to configure the paths to your data directories.

### Setup

```bash
# Copy the example configuration
cp config.yaml.example config.yaml

# Edit config.yaml with your paths
```

### Configuration File (`config.yaml`)

```yaml
# Data source paths - edit these for your environment
paths:
  # Directory containing OMS log parquet files
  # Files are named: YYYY-MM-DD__oms_order_log.parq
  oms_logs: /path/to/your/oms_logs

  # Directory containing Paribu market data
  # Structure: {product}/orderbook/{date}/{hour}/*.parquet
  paribu_market_data: /path/to/your/paribu/market-data
```

### Notes

- `config.yaml` is gitignored (contains machine-specific paths)
- `config.yaml.example` is tracked and serves as a template
- If `config.yaml` is missing, scripts will exit with a helpful error message

---

## Directory Organization

Results are organized hierarchically by date and market symbol:

```
results/
├── shared/
│   └── markets_map.csv                    # Cross-exchange market ID mapping
├── 2026-01-16/
│   ├── DOGE-TL/
│   │   ├── orderbook.csv                  # Raw orderbook data
│   │   ├── orderbook_consolidated.csv     # Top-of-book snapshots
│   │   ├── trades.csv                     # Deduplicated trade data
│   │   ├── chains_buy.csv                 # Buy order chains
│   │   ├── chains_sell.csv                # Sell order chains
│   │   ├── level1_analysis.md             # Level 1 analysis report
│   │   ├── trade_attribution.csv          # Per-trade attribution details
│   │   └── charts/
│   │       ├── orderbook_heatmap_*.png    # Orderbook heatmaps
│   │       ├── chains_buy_duration_hist.png
│   │       ├── chains_buy_empty_gap_hist.png
│   │       ├── chains_sell_duration_hist.png
│   │       └── chains_sell_empty_gap_hist.png
│   ├── AVAX-TL/
│   │   └── ...
│   └── UMA-TL/
│       └── ...
└── 2026-01-06/
    └── ADA-TL/
        └── ...
```

This organization:
- Groups all outputs for a market/date combination together
- Simplifies file naming (no need for date/market prefixes in filenames)
- Makes it easy to archive or share analysis for a specific day/market
- Keeps shared resources (like `markets_map.csv`) in a dedicated location

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

### `analyze_oms.py` - OMS Order Log Analysis

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

1. **Analysis Report** (`oms_order_log_analysis.md`) - Structured markdown with:
   - Executive Summary (key metrics at a glance)
   - Data Overview (file info, schema, time range)
   - Market Coverage (exchanges and products traded)
   - Order Flow Analysis (status distribution, cancellation timing)
   - Trade Execution (volume and notional value by market)
   - Key Insights (auto-generated findings)

2. **CSV Statistics Files**:
   - `order_timing_stats.csv` - Time between order placement and cancellation
   - `trade_statistics.csv` - Trade volume statistics
   - `notional_value_stats.csv` - Trade values in TRY

---

### `extract_chains.py` - Order Chain Extraction

Extracts order chains from OMS parquet files. A chain is defined as a unique order (by `client_order_id`) that starts with `status=new` and ends with `status=canceled`.

#### Usage

```bash
python extract_chains.py YYYY-MM-DD exchange_id market_id
```

#### Examples

```bash
python extract_chains.py 2026-01-16 129 270    # PARIBU DOGE-TL
python extract_chains.py 2026-01-16 129 293    # PARIBU ADA-TL
```

#### Features

- **Chain Extraction**: Identifies complete order lifecycles (new → canceled)
- **Side Filtering**: Separates buy and sell orders into distinct files
- **Empty Gap Calculation**: Computes time between consecutive order chains
- **Duration Analysis**: Measures how long each order was active
- **Histogram Generation**: Creates visualizations for duration and empty gap distributions
- **Auto Market Symbol Lookup**: Automatically determines market symbol from exchange/market IDs

#### Generated Outputs

1. **Chain CSV Files** (per side):
   - `chains_buy.csv` - Buy order chains
   - `chains_sell.csv` - Sell order chains

2. **Histogram Charts** (in `charts/` subdirectory):
   - `chains_buy_duration_hist.png` - Distribution of buy chain durations
   - `chains_buy_empty_gap_hist.png` - Distribution of gaps between buy chains
   - `chains_sell_duration_hist.png` - Distribution of sell chain durations
   - `chains_sell_empty_gap_hist.png` - Distribution of gaps between sell chains

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

#### Usage

```bash
# Extract both orderbook and trades for a product on a date
python extract_public_data.py --date 2026-01-16 --product DOGE-TL

# Extract only trades (with automatic deduplication)
python extract_public_data.py --date 2026-01-16 --product DOGE-TL --type trades

# Extract only orderbook
python extract_public_data.py --date 2026-01-16 --product DOGE-TL --type orderbook
```

#### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--date` | Yes | Date to extract in YYYY-MM-DD format |
| `--product` | Yes | Product symbol (e.g., ADA-TL, BTC-TL, DOGE-TL) |
| `--type` | No | Type of data: `orderbook`, `trades`, or `both` (default: both) |

#### Features

- **Hourly Aggregation**: Reads and concatenates all parquet files from hours 00-23
- **Multiple Files Per Hour**: Handles cases where trades have multiple files per hour
- **Schema Compatibility**: Supports both older (`ts_ingest_unix_ms`) and newer (`ts_ingest_unix_us`) schemas
- **Trade Deduplication**: Automatically removes duplicate trades by `trade_id`
- **Date Filtering**: Filters trades to only include those with exchange timestamps on the target date
- **Data Transformation**:
  - Converts timestamps to human-readable datetime
  - Converts `price_e8` and `size_e8` to decimal values (divides by 1e8)
  - Maps side values (1=buy, 2=sell) to readable names
- **Orderbook Consolidation**: Automatically consolidates orderbook snapshots into top-of-book format

#### Generated Outputs

1. **Raw Orderbook** (`orderbook.csv`):
   - All orderbook entries with full detail
   - Sorted by ingest timestamp (sequence ordering)

2. **Consolidated Orderbook** (`orderbook_consolidated.csv`):
   - One row per orderbook snapshot showing top 3 bid/ask levels
   - Columns: ts_ingest_unix_us, datetime_ingest, best_bid_price_1/2/3, best_bid_qty_1/2/3, best_ask_price_1/2/3, best_ask_qty_1/2/3
   - Bids sorted by price descending (highest = best_bid_price_1)
   - Asks sorted by price ascending (lowest = best_ask_price_1)

3. **Trades** (`trades.csv`):
   - Deduplicated trade records (unique by `trade_id`)
   - Filtered to target date by exchange timestamp
   - Sorted by exchange timestamp (authoritative trade time)

---

### `calculate_level1_time.py` - Level 1 Time Analysis

Calculates the percentage of time your orders are at Level 1 (best bid/best ask) by comparing order chains against public orderbook snapshots. Also performs trade attribution and fake trade detection.

#### Usage

```bash
python calculate_level1_time.py YYYY-MM-DD PRODUCT
```

#### Examples

```bash
python calculate_level1_time.py 2026-01-16 DOGE-TL
python calculate_level1_time.py 2026-01-16 AVAX-TL
```

#### Features

- **Level 1 Time Analysis**: Reports percentage of time at best bid/ask as fraction of active time and total day
- **Price AND Quantity Matching**: Correctly identifies your orders by matching both price and quantity
- **Trade Attribution**: For each public trade, determines if your order was likely filled
- **Fake Trade Detection**: Identifies trades with prices inside the bid-ask spread (potentially fake)
- **Time-Weighted Analysis**: Accounts for varying snapshot intervals
- **Side-Specific Analysis**: Separate statistics for buy (bid) and sell (ask) sides

#### Generated Outputs

1. **Markdown Report** (`level1_analysis.md`):

```markdown
# Level 1 Analysis: 2026-01-16 DOGE-TL

## Order Positioning
### Buy Side:
Active orders 20.7% of the day (4h 51m)
At Level 1: 42.6% of active time / 8.8% of total day

### Sell Side:
Active orders 30.5% of the day (6h 50m)
At Level 1: 37.4% of active time / 11.4% of total day

## Trade Attribution
Total public trades: 527
- Our Sells (hit by public buys): 18 (7.6%)
- Our Buys (hit by public sells): 20 (6.9%)

## Fake Trade Detection
- Total trades: 527
- Fake trades: 4 (0.8%)
```

2. **Trade Attribution CSV** (`trade_attribution.csv`):
   - Per-trade details including datetime, trade_id, public_side, our_side, price, size, attribution, is_fake

#### Methodology: Avoiding False Positives

The analysis correctly identifies when YOUR orders are at Level 1 by checking **both price AND quantity**, not just quantity alone. For each orderbook snapshot, the tool checks:
- Is your order active? (snapshot time between `datetime_open` and `datetime_close`)
- Does your order's **price** match the Level 1 price?
- Does your order's **quantity** match the Level 1 quantity?

Only when ALL three conditions are true is the snapshot counted as "at Level 1".

#### Fake Trade Detection

Trades are flagged as potentially fake if:
1. The trade price is inside the bid-ask spread
2. The price is more than 2.5 basis points away from BOTH the best bid and best ask
3. This condition holds for ALL orderbook snapshots within a configurable time tolerance window

#### Prerequisites

Requires:
- Consolidated orderbook data: `results/YYYY-MM-DD/PRODUCT/orderbook_consolidated.csv`
- Order chain data: `results/YYYY-MM-DD/PRODUCT/chains_buy.csv` and `chains_sell.csv`
- Market mapping: `results/shared/markets_map.csv`

---

### `visualize_orderbook.py` - Orderbook Heatmap Visualization

Generates dual-panel visualizations of orderbook data: a heatmap showing depth/quantity at each price level on top, and a step-chart of best bid/ask prices on the bottom.

#### Usage

```bash
python visualize_orderbook.py YYYY-MM-DD PRODUCT HH:MM:SS HH:MM:SS [options]
```

#### Examples

```bash
# Basic heatmap for a 5-minute window
python visualize_orderbook.py 2026-01-16 DOGE 00:00:00 00:05:00

# Highlight specific quantities in white (your order sizes)
python visualize_orderbook.py 2026-01-16 DOGE 12:45:00 12:50:00 --highlight-qty 1600.0

# Disable trades overlay
python visualize_orderbook.py 2026-01-16 DOGE 00:00:00 00:05:00 --no-trades

# Disable quantity highlighting
python visualize_orderbook.py 2026-01-16 DOGE 00:00:00 00:05:00 --no-highlight
```

#### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `date` | Yes | Date in YYYY-MM-DD format |
| `product` | Yes | Product symbol (e.g., ADA, BTC, DOGE) |
| `start` | Yes | Start time in HH:MM:SS format |
| `end` | Yes | End time in HH:MM:SS format |
| `--highlight-qty` | No | Quantity value(s) to highlight in white |
| `--no-trades` | No | Disable trades overlay |
| `--no-highlight` | No | Disable quantity highlighting |

#### Features

- **Dual-Panel Layout**: Top panel shows quantity heatmap, bottom panel shows price chart
- **Synchronized Time Axis**: Both panels share the same x-axis for easy correlation
- **Rainbow Colormap**: Quantities displayed from red (low) to blue (high)
- **Step-Style Price Chart**: Prices shown as horizontal steps with vertical jumps
- **Quantity Highlighting**: Mark your order sizes in white
- **Trade Markers**: Public trades shown as black markers at the matched price level
- **Top 3 Levels**: Shows best 3 bid and ask levels (6 total rows in heatmap)

#### Generated Output

- `charts/orderbook_heatmap_HH-MM-SS_HH-MM-SS.png`

---

### `visualize_chains.py` - Order Chain Visualization

Visualizes order chains as horizontal lines on a time vs price plot. Buy orders shown in green, sell orders in red.

#### Usage

```bash
python visualize_chains.py YYYY-MM-DD exchange_id market_id HH:MM:SS HH:MM:SS
```

#### Examples

```bash
python visualize_chains.py 2026-01-16 129 270 00:00:00 01:00:00
python visualize_chains.py 2026-01-16 129 293 12:00:00 13:30:00
```

#### Features

- **Time vs Price Plot**: Each order chain displayed as a horizontal line at its price level
- **Color Coding**: Buy orders in green, sell orders in red
- **Time Window Filtering**: Chains clipped to display window boundaries
- **Chain Counts**: Legend shows number of buy and sell chains in the window
- **Grid Overlay**: Dotted grid for easy price/time reading

#### Generated Output

- `charts/chains_visual_HH-MM-SS_HH-MM-SS.png`

---

### `visualize_timestamps.py` - Timestamp Analysis Visualization

Visualizes the relationship between ingest timestamps and exchange timestamps for public trade data. Helps identify discrepancies and patterns in data ingestion timing.

#### Usage

```bash
python visualize_timestamps.py --date YYYY-MM-DD --product SYMBOL
```

#### Examples

```bash
python visualize_timestamps.py --date 2026-01-16 --product DOGE-TL
```

#### Features

- **Scatter Plot**: Exchange timestamp vs ingest timestamp with delay coloring
- **Delay Histogram**: Distribution of ingestion delays with full and zoomed views
- **Time Series Plot**: Delay over time with rolling average
- **Statistics Summary**: Comprehensive delay statistics including percentile distribution
- **Warning Detection**: Flags negative delays (exchange timestamp after ingest)

#### Generated Outputs

1. `charts/timestamp_scatter.png` - Scatter plot of exchange vs ingest timestamps
2. `charts/timestamp_delay_hist.png` - Delay distribution histogram
3. `charts/timestamp_delay_timeseries.png` - Delay over time

---

### `find_co_markets.py` - Co-Traded Market Finder

Finds base currencies traded on both BTCTURK and PARIBU exchanges for a given date.

#### Usage

```bash
python find_co_markets.py YYYY-MM-DD
```

#### Features

- **Cross-Exchange Analysis**: Identifies which base currencies are traded on both exchanges
- **Overlap Detection**: Flags markets with `overlap=1` if traded on both exchanges
- **Market ID Mapping**: Uses `markets_map.csv` to translate market IDs to base currencies

#### Generated Output

- `results/YYYY-MM-DD_co-markets.csv`

---

### `create_markets_map.py` - Market Mapping Generator

Creates a mapping between BTCTURK and PARIBU market IDs based on shared base currencies.

#### Usage

```bash
python create_markets_map.py
```

#### Generated Output

- `results/shared/markets_map.csv`:

| Column | Description |
|--------|-------------|
| `base_currency` | The base currency symbol (e.g., BTC, ETH, ADA) |
| `btcturk_exchange_id` | BTCTURK exchange ID (115) |
| `btcturk_market_id` | BTCTURK market ID for this base currency |
| `paribu_exchange_id` | PARIBU exchange ID (129) |
| `paribu_market_id` | PARIBU market ID for this base currency |

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

### `results_path.py` - Results Directory Utility

A utility module that provides helper functions for managing the results directory structure. Used internally by other scripts.

#### Functions

```python
from results_path import get_results_dir, get_charts_dir, get_shared_dir

# Get base results directory
get_results_dir()  # -> results/

# Get date-level directory
get_results_dir("2026-01-16")  # -> results/2026-01-16/

# Get market-level directory
get_results_dir("2026-01-16", "DOGE-TL")  # -> results/2026-01-16/DOGE-TL/

# Get charts directory
get_charts_dir("2026-01-16", "DOGE-TL")  # -> results/2026-01-16/DOGE-TL/charts/

# Get shared directory for cross-date resources
get_shared_dir()  # -> results/shared/
```

---

### `config.py` - Configuration Loader

A utility module that loads configuration from `config.yaml` and provides accessor functions for data source paths.

#### Functions

```python
from config import get_oms_logs_path, get_paribu_market_data_path

# Get path to OMS logs directory
oms_path = get_oms_logs_path()  # -> Path("/path/to/oms_logs")

# Get path to Paribu market data directory
paribu_path = get_paribu_market_data_path()  # -> Path("/path/to/market-data")
```

#### Error Handling

If `config.yaml` is missing, the module will exit with a helpful message:

```
Error: Configuration file not found: /path/to/config.yaml

Please create config.yaml with your data paths.
You can copy config.yaml.example as a starting point:
  cp /path/to/config.yaml.example /path/to/config.yaml
```

---

## Input Data Format

### OMS Order Log Format

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

### Public Market Data Source

Public market data location is configured in `config.yaml` under `paths.paribu_market_data`.

The data is organized by product, data type, date, and hour:
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

---

## Output Files

All outputs are saved to the `results/` directory organized by date and market.

### Per-Market Outputs (in `results/YYYY-MM-DD/PRODUCT/`)

| File | Description |
|------|-------------|
| `orderbook.csv` | Raw orderbook data |
| `orderbook_consolidated.csv` | Top-of-book snapshots |
| `trades.csv` | Deduplicated trade data |
| `chains_buy.csv` | Buy order chains |
| `chains_sell.csv` | Sell order chains |
| `level1_analysis.md` | Level 1 time analysis report |
| `trade_attribution.csv` | Per-trade attribution details |

### Charts (in `results/YYYY-MM-DD/PRODUCT/charts/`)

| File | Description |
|------|-------------|
| `orderbook_heatmap_*.png` | Orderbook depth heatmaps with price charts |
| `chains_buy_duration_hist.png` | Buy chain duration distribution |
| `chains_buy_empty_gap_hist.png` | Buy chain empty gap distribution |
| `chains_sell_duration_hist.png` | Sell chain duration distribution |
| `chains_sell_empty_gap_hist.png` | Sell chain empty gap distribution |
| `chains_visual_*.png` | Order chain time vs price plots |
| `timestamp_scatter.png` | Timestamp comparison scatter plots |
| `timestamp_delay_hist.png` | Ingestion delay histograms |
| `timestamp_delay_timeseries.png` | Delay time series plots |

### Shared Outputs (in `results/shared/`)

| File | Description |
|------|-------------|
| `markets_map.csv` | Cross-exchange market ID mapping |

---

## Market Reference

The `market.csv` file maps market and exchange IDs to human-readable names. Format (no header):

```csv
market_id,exchange_id,exchange_name,product_name
116,115,BTCTURK,SPOT.TRX.TRY
134,115,BTCTURK,SPOT.DOT.TRY
270,129,PARIBU,SPOT.DOGE.TL
```

### Supported Exchanges

| Exchange ID | Exchange Name |
|-------------|---------------|
| 115 | BTCTURK |
| 129 | PARIBU |

---

## Analysis Methodology

### Order Chain Analysis

An order chain represents the complete lifecycle of an order:
1. **Chain Identification**: Match orders with both `status=new` and `status=canceled`
2. **Duration Calculation**: `time_close - time_open`
3. **Empty Gap Calculation**: Time between previous chain's close and current chain's open
4. **Side Separation**: Split into buy and sell for independent analysis

### Level 1 Time Analysis

For each orderbook snapshot:
1. Check if any order chain is active (snapshot time between open and close)
2. If active, check if the chain's price AND quantity match Level 1
3. Count time-weighted snapshots at Level 1

### Trade Attribution

For each public trade:
1. Find the most recent orderbook snapshot before the trade
2. Determine if your order was at Level 1 at that moment
3. Classify as "ours", "not ours", or "no position"

### Fake Trade Detection

Trades are classified as potentially fake if:
1. Trade price is strictly inside the bid-ask spread
2. Price is more than 2.5 bps away from both best bid and best ask
3. This holds for ALL orderbook snapshots within the time tolerance window

### Histogram Generation

Histograms use the 95th percentile as the maximum x-axis value with 20 bins to focus on the typical distribution while excluding outliers.

---

## Example Workflows

### Complete Market Analysis

```bash
# Step 1: Create market mapping (one-time)
python create_markets_map.py

# Step 2: Extract public data
python extract_public_data.py --date 2026-01-16 --product DOGE-TL

# Step 3: Extract order chains
python extract_chains.py 2026-01-16 129 270

# Step 4: Run Level 1 analysis
python calculate_level1_time.py 2026-01-16 DOGE-TL

# Step 5: Generate visualizations
python visualize_orderbook.py 2026-01-16 DOGE 00:00:00 00:05:00
python visualize_orderbook.py 2026-01-16 DOGE 12:00:00 12:05:00
```

### Multi-Market Batch Analysis

```bash
# Analyze multiple markets for the same date
for product in DOGE-TL AVAX-TL UMA-TL; do
    echo "Processing $product..."
    python extract_public_data.py --date 2026-01-16 --product $product
    python calculate_level1_time.py 2026-01-16 $product
    python visualize_orderbook.py 2026-01-16 ${product%-TL} 00:00:00 00:05:00
done
```

### Cross-Exchange Analysis

```bash
# Find co-traded markets
python find_co_markets.py 2026-01-16

# Extract chains for same base currency on both exchanges
python extract_chains.py 2026-01-16 115 244    # BTCTURK ADA
python extract_chains.py 2026-01-16 129 293    # PARIBU ADA
```

---

## Key Metrics Interpretation

### Level 1 Time

- **"Of active time"**: What percentage of time when you have an order in the market are you at Level 1? Higher percentages indicate you're frequently at the best price.
- **"Of total day"**: What percentage of the entire trading day are you at Level 1? This factors in periods when you have no orders in the market.

### Trade Attribution

- **Ours (at L1)**: Trades where your order was likely filled (you were at Level 1)
- **Not ours**: You had an order active but not at Level 1 (someone else got the fill)
- **No position**: You had no order active when the trade occurred

### Fake Trade Detection

- Trades with prices inside the spread that couldn't have occurred through normal order book matching
- Useful for filtering out potentially manipulated or erroneous trade data

### Empty Gap Analysis

- **Negative or zero gaps**: Overlapping orders (multiple orders active simultaneously)
- **Small positive gaps** (<1s): Continuous market presence
- **Large gaps** (>10s): Intermittent trading activity

---

## Notes

- All timestamps are in UTC
- Notional values are calculated in TRY (Turkish Lira)
- The `results/` directory is created automatically if it doesn't exist
- **Paribu Trade Data**: Raw trade data contains 85-95% duplicates due to exchange retransmission behavior. The `extract_public_data.py` script handles this automatically.
- **Trade Timestamps**: For trade analysis, always use `ts_exchange_unix_ms` (authoritative trade time), not ingest timestamps
- For detailed technical documentation on the deduplication process, see `PARIBU_DEDUPLICATION_BRIEFING.md`
