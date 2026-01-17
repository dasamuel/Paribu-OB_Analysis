#!/usr/bin/env python3
"""
Configuration Loader Module

Loads configuration from config.yaml and provides accessor functions
for data source paths.

Usage:
    from config import get_oms_logs_path, get_paribu_market_data_path
    
    oms_path = get_oms_logs_path()
    paribu_path = get_paribu_market_data_path()
"""

import sys
from pathlib import Path

import yaml


# Path to config file (same directory as this module)
CONFIG_FILE = Path(__file__).parent / "config.yaml"

# Cached config to avoid re-reading file
_config_cache = None


def _load_config() -> dict:
    """Load and cache configuration from config.yaml."""
    global _config_cache
    
    if _config_cache is not None:
        return _config_cache
    
    if not CONFIG_FILE.exists():
        print(f"Error: Configuration file not found: {CONFIG_FILE}")
        print()
        print("Please create config.yaml with your data paths.")
        print("You can copy config.yaml.example as a starting point:")
        print(f"  cp {CONFIG_FILE.parent}/config.yaml.example {CONFIG_FILE}")
        sys.exit(1)
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            _config_cache = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in config file: {e}")
        sys.exit(1)
    
    return _config_cache


def get_oms_logs_path() -> Path:
    """
    Get the path to OMS logs directory.
    
    Returns:
        Path to the OMS logs directory (e.g., /Users/.../oms_logs)
    """
    config = _load_config()
    path_str = config.get('paths', {}).get('oms_logs')
    
    if not path_str:
        print("Error: 'paths.oms_logs' not found in config.yaml")
        sys.exit(1)
    
    return Path(path_str)


def get_paribu_market_data_path() -> Path:
    """
    Get the path to Paribu market data directory.
    
    Returns:
        Path to the Paribu market data directory (e.g., /Users/.../market-data)
    """
    config = _load_config()
    path_str = config.get('paths', {}).get('paribu_market_data')
    
    if not path_str:
        print("Error: 'paths.paribu_market_data' not found in config.yaml")
        sys.exit(1)
    
    return Path(path_str)
