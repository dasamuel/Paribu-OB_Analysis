#!/usr/bin/env python3
"""
Results Path Utility Module

Provides helper functions for managing the results directory structure.
Results are organized by date and market:

    results/
    ├── 2026-01-06/
    │   └── ADA-TL/
    │       ├── chains_buy.csv
    │       ├── level1_analysis.md
    │       └── charts/
    │           └── orderbook_heatmap_00-00-00_00-05-00.png
    ├── 2026-01-16/
    │   └── ADA-TL/
    │       └── ...
    └── shared/
        └── markets_map.csv
"""

from pathlib import Path


# Base results directory (relative to workspace root)
RESULTS_BASE = Path(__file__).parent / "results"


def get_results_dir(date: str = None, market: str = None) -> Path:
    """
    Get the results directory for a given date and optional market.
    
    Args:
        date: Date string in YYYY-MM-DD format. If None, returns base results dir.
        market: Market symbol (e.g., "ADA-TL"). If None, returns date-level dir.
    
    Returns:
        Path to the appropriate results directory (created if it doesn't exist).
    
    Examples:
        get_results_dir()                      -> results/
        get_results_dir("2026-01-06")          -> results/2026-01-06/
        get_results_dir("2026-01-06", "ADA-TL") -> results/2026-01-06/ADA-TL/
    """
    path = RESULTS_BASE
    
    if date:
        path = path / date
    
    if market:
        if not date:
            raise ValueError("Market requires a date to be specified")
        path = path / market
    
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_charts_dir(date: str, market: str) -> Path:
    """
    Get the charts directory for a given date and market.
    
    Args:
        date: Date string in YYYY-MM-DD format.
        market: Market symbol (e.g., "ADA-TL").
    
    Returns:
        Path to the charts directory (created if it doesn't exist).
    
    Example:
        get_charts_dir("2026-01-06", "ADA-TL") -> results/2026-01-06/ADA-TL/charts/
    """
    charts_path = get_results_dir(date, market) / "charts"
    charts_path.mkdir(parents=True, exist_ok=True)
    return charts_path


def get_shared_dir() -> Path:
    """
    Get the shared directory for date-independent reference files.
    
    Returns:
        Path to the shared directory (created if it doesn't exist).
    
    Example:
        get_shared_dir() -> results/shared/
    """
    shared_path = RESULTS_BASE / "shared"
    shared_path.mkdir(parents=True, exist_ok=True)
    return shared_path
