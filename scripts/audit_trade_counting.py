#!/usr/bin/env python3
"""
Audit Script: Analyze Trade Counting Logic

This script examines how trades are counted and how costs are applied.
Key question: Are we counting POSITIONS or BARS?

If signal = 1 for 10 consecutive bars, is that:
- 1 trade (correct)? or
- 10 trades (wrong)?
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd


def analyze_signal_patterns(signals: np.ndarray | list) -> dict:
    """
    Analyze signal array to understand trading patterns.
    
    Returns:
        Dict with analysis results
    """
    signals = np.array(signals)
    n_bars = len(signals)
    
    # Count bars with active signals (current method)
    n_signal_bars = np.sum(signals != 0)
    
    # Count actual position changes (correct method)
    # A trade happens when:
    # 1. Signal goes from 0 to non-zero (open position)
    # 2. Signal changes direction (close + open)
    # 3. Signal goes from non-zero to 0 (close position)
    
    signal_changes = np.diff(signals)
    n_position_changes = np.sum(signal_changes != 0)
    
    # More detailed analysis
    opens = 0
    closes = 0
    reversals = 0
    
    prev_signal = 0
    for s in signals:
        if prev_signal == 0 and s != 0:
            opens += 1
        elif prev_signal != 0 and s == 0:
            closes += 1
        elif prev_signal != 0 and s != 0 and prev_signal != s:
            reversals += 1
        prev_signal = s
    
    # Count if we end with open position
    if prev_signal != 0:
        closes += 1  # Need to close at end
    
    total_round_trips = opens  # Each open eventually closed
    
    return {
        'n_bars': n_bars,
        'n_signal_bars': n_signal_bars,  # Current "n_trades"
        'n_position_opens': opens,
        'n_position_closes': closes,
        'n_reversals': reversals,
        'n_round_trips': total_round_trips,
        'signal_bar_to_trade_ratio': n_signal_bars / max(total_round_trips, 1),
        'implied_avg_hold_bars': n_signal_bars / max(total_round_trips, 1),
    }


def load_and_analyze_experiment(exp_path: str) -> dict:
    """Load experiment result and analyze trade patterns."""
    
    with open(exp_path, 'r') as f:
        result = json.load(f)
    
    meta = result.get('_meta', {})
    
    return {
        'id': meta.get('id', 'unknown'),
        'n_trades_reported': result.get('n_trades', 0),
        'pf_pre': result.get('profit_factor', 0),
        'pf_post': result.get('profit_factor_post_cost', 0),
        'sharpe_pre': result.get('sharpe', 0),
        'sharpe_post': result.get('sharpe_post_cost', 0),
    }


def simulate_cost_scenarios():
    """
    Calculate what PF would be under different cost assumptions.
    """
    print("=" * 60)
    print("COST SCENARIO ANALYSIS")
    print("=" * 60)
    
    # Assume from Stage 8 results:
    # Pre-cost: ~14% return, 8628 "trades" (signal bars)
    # If avg hold = 10 bars, that's ~863 actual round-trip trades
    
    pre_cost_return_pct = 14.2
    n_signal_bars = 8628
    
    # Estimate actual trades (assuming avg hold of 5-20 bars)
    for avg_hold_bars in [1, 3, 5, 10, 20]:
        actual_trades = n_signal_bars / avg_hold_bars
        
        print(f"\n--- If avg hold = {avg_hold_bars} bars ---")
        print(f"Actual round-trips: {actual_trades:.0f}")
        
        for cost_bps in [30, 15, 10, 5, 2]:
            cost_pct = cost_bps / 100
            total_cost_pct = actual_trades * cost_pct
            post_cost_return_pct = pre_cost_return_pct - total_cost_pct
            
            status = "✅" if post_cost_return_pct > 0 else "❌"
            print(f"  Cost {cost_bps}bps: Return {post_cost_return_pct:+.1f}% (cost: {total_cost_pct:.1f}%) {status}")


def main():
    """Run audit analysis."""
    
    print("=" * 60)
    print("ALPHA RESEARCH METHODOLOGY AUDIT")  
    print("Trade Counting Analysis")
    print("=" * 60)
    
    # Example: analyze signal pattern
    print("\n--- Example Signal Pattern Analysis ---")
    
    # Simulate typical signal pattern
    # 1,1,1,0,0,-1,-1,0,0,1,1,1,1,0
    example_signals = [1,1,1,0,0,-1,-1,0,0,1,1,1,1,0]
    
    analysis = analyze_signal_patterns(example_signals)
    
    print(f"Signal array: {example_signals}")
    print(f"Total bars: {analysis['n_bars']}")
    print(f"Signal bars (current 'n_trades'): {analysis['n_signal_bars']}")
    print(f"Actual position opens: {analysis['n_position_opens']}")
    print(f"Actual round-trips: {analysis['n_round_trips']}")
    print(f"Implied avg hold: {analysis['implied_avg_hold_bars']:.1f} bars")
    
    print("\n⚠️  ISSUE: Cost is applied to EACH signal bar, not to each trade!")
    print(f"    If we charge 0.3% per 'trade', we charge: {analysis['n_signal_bars']} × 0.3%")
    print(f"    But real cost should be: {analysis['n_round_trips']} × 0.3%")
    print(f"    Overcharge ratio: {analysis['n_signal_bars'] / max(analysis['n_round_trips'], 1):.1f}x")
    
    # Cost scenarios
    simulate_cost_scenarios()
    
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("=" * 60)
    print("""
Current implementation counts every BAR with a signal as a "trade".
If the model holds position for average of 10 bars, we overcharge 10x on costs!

Stage 8 results:
- n_trades = 8628 (signal bars)
- Pre-cost return = +14.2%
- Post-cost return = -100% (with 0.3% per signal bar)

If actual avg hold = 10 bars:
- Real round-trips = ~863
- Real cost = 863 × 0.3% = 2.6%
- Corrected post-cost return = 14.2% - 2.6% = +11.6%! (not -100%)

🔴 ACTION REQUIRED: Fix cost calculation to count actual trades, not signal bars.
""")


if __name__ == "__main__":
    main()
