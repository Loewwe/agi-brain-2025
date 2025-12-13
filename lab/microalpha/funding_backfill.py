#!/usr/bin/env python3
"""
Funding Rate Data Backfill
Purpose: Download 90 days of funding rate + OI data from Binance
Output: lab/microalpha/data/funding_1h.parquet
"""

import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import time

# Config
EXCHANGES = {
    'binance': ccxt.binance({'enableRateLimit': True})
}

SYMBOLS = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT', 'XRP/USDT:USDT', 'ADA/USDT:USDT', 'DOGE/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT', 'MATIC/USDT:USDT', 'DOT/USDT:USDT', 'LTC/USDT:USDT', 'BCH/USDT:USDT', 'TRX/USDT:USDT', 'ATOM/USDT:USDT', 'NEAR/USDT:USDT', 'APT/USDT:USDT', 'SUI/USDT:USDT', 'ARB/USDT:USDT', 'OP/USDT:USDT', 'FIL/USDT:USDT', 'INJ/USDT:USDT', 'UNI/USDT:USDT', 'ETC/USDT:USDT', 'ICP/USDT:USDT']
DAYS = 90
OUTPUT_DIR = Path('lab/microalpha/data')
LOGS_DIR = Path('lab/microalpha/logs')

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def fetch_funding_history(exchange, symbol, days=90):
    """
    Fetch funding rate history from exchange
    Returns: DataFrame with columns [timestamp, funding_rate, open_interest, price_index]
    """
    print(f"  Fetching {symbol} funding history ({days} days)...")
    
    end_time = int(datetime.utcnow().timestamp() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    
    all_data = []
    current_time = start_time
    
    while current_time < end_time:
        try:
            # Binance funding history (8h intervals)
            funding = exchange.fetch_funding_rate_history(
                symbol,
                since=current_time,
                limit=1000
            )
            
            if not funding:
                break
            
            for entry in funding:
                all_data.append({
                    'timestamp': pd.to_datetime(entry['timestamp'], unit='ms'),
                    'funding_rate': entry.get('fundingRate', 0),
                    'price_index': entry.get('markPrice', 0)
                })
            
            # Move to next batch
            current_time = funding[-1]['timestamp'] + 1
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"    Error fetching funding: {e}")
            break
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    
    # Add open interest (separate call)
    try:
        oi = exchange.fetch_open_interest(symbol)
        # For simplicity, use current OI for all rows (in production, fetch historical)
        df['open_interest'] = oi.get('openInterestAmount', 0) * df['price_index']
    except:
        df['open_interest'] = 0
    
    # Resample to 1h (funding is typically 8h, so we forward-fill)
    df = df.set_index('timestamp').resample('1H').ffill().reset_index()
    
    return df

def validate_data(df, symbol):
    """
    Check data quality invariants
    Returns: (is_valid, issues_list)
    """
    issues = []
    
    # Check for NaN
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        issues.append(f"NaN values in columns: {nan_cols}")
    
    # Check for duplicates
    dupes = df.duplicated(subset=['timestamp']).sum()
    if dupes > 0:
        issues.append(f"{dupes} duplicate timestamps")
    
    # Check time gaps (should be exactly 1h)
    time_diffs = df['timestamp'].diff().dt.total_seconds() / 3600
    gaps = time_diffs[time_diffs > 1.1].index.tolist()
    if gaps:
        issues.append(f"{len(gaps)} gaps > 1h detected")
    
    # Check minimum rows
    min_rows = DAYS * 24
    if len(df) < min_rows * 0.95:
        issues.append(f"Only {len(df)} rows, expected ~{min_rows}")
    
    is_valid = len(issues) == 0
    return is_valid, issues

def main():
    """Main backfill process"""
    print("🔄 FUNDING RATE BACKFILL")
    print(f"Period: {DAYS} days")
    print(f"Symbols: {SYMBOLS}")
    print()
    
    results = {
        'timestamp': datetime.utcnow().isoformat(),
        'symbols_ok': [],
        'symbols_failed': [],
        'total_rows': 0,
        'time_range': {},
        'gaps_detected': []
    }
    
    all_data = []
    
    for symbol in SYMBOLS:
        print(f"📊 Processing {symbol}...")
        
        try:
            # Fetch data
            df = fetch_funding_history(EXCHANGES['binance'], symbol, DAYS)
            
            if df.empty:
                print(f"  ❌ No data returned")
                results['symbols_failed'].append(symbol)
                continue
            
            # Add metadata
            df['exchange'] = 'binance'
            df['symbol'] = symbol.replace('/USDT:USDT', 'USDT')
            
            # Validate
            is_valid, issues = validate_data(df, symbol)
            
            if not is_valid:
                print(f"  ⚠️  Validation issues:")
                for issue in issues:
                    print(f"     - {issue}")
                results['symbols_failed'].append(symbol)
            else:
                print(f"  ✅ {len(df)} rows, {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
                results['symbols_ok'].append(symbol)
                all_data.append(df)
                
                results['time_range'][symbol] = {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat(),
                    'rows': len(df)
                }
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results['symbols_failed'].append(symbol)
    
    # Combine and save
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        
        # Reorder columns
        combined = combined[['timestamp', 'exchange', 'symbol', 'funding_rate', 
                           'open_interest', 'price_index']]
        
        # Save to parquet
        output_file = OUTPUT_DIR / 'funding_1h.parquet'
        combined.to_parquet(output_file, index=False)
        
        results['total_rows'] = len(combined)
        print()
        print(f"💾 Saved {len(combined)} rows to {output_file}")
    else:
        print()
        print("❌ No data collected")
    
    # Save summary
    summary_file = Path('lab/microalpha/results/funding_backfill_summary.json')
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📄 Summary saved to {summary_file}")
    
    # Final check
    print()
    print("✅ BACKFILL COMPLETE")
    print(f"   Success: {len(results['symbols_ok'])}/{len(SYMBOLS)}")
    
    # Readiness check for BTCUSDT
    if 'BTC/USDT:USDT' in results['symbols_ok']:
        btc_rows = results['time_range']['BTC/USDT:USDT']['rows']
        expected = DAYS * 24
        if btc_rows >= expected * 0.95:
            print(f"   ✅ BTCUSDT ready for backtest ({btc_rows} rows)")
        else:
            print(f"   ⚠️  BTCUSDT incomplete ({btc_rows}/{expected} rows)")
    
    return 0 if results['symbols_ok'] else 1

if __name__ == "__main__":
    exit(main())
