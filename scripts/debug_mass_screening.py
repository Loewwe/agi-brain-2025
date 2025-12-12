#!/usr/bin/env python3
"""
Debug Mass Screening - Show raw metrics for all 126 hypotheses WITHOUT filters

Purpose: Understand what metrics hypotheses actually achieve to calibrate filters
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.mass_screening_runner import load_catalog, load_data, calculate_metrics


def analyze_hypothesis_performance():
    """Analyze all hypotheses without filters to see real performance."""
    
    print("="*70)
    print("MASS SCREENING DEBUG - RAW METRICS ANALYSIS")
    print("="*70)
    print("\nLoading catalog and data...")
    
    # Load
    catalog = load_catalog()
    data_map = load_data()
    
    print(f"Loaded {len(catalog)} hypotheses")
    print(f"Loaded {len(data_map)} symbols")
    
    # Test all hypotheses
    all_results = []
    
    for i, hyp_entry in enumerate(catalog):
        if isinstance(hyp_entry, dict):
            hyp_class = hyp_entry.get('class')
            hyp_id = hyp_entry.get('id', f'H{i:03d}')
        else:
            hyp_class = hyp_entry
            hyp_id = hyp_class.__name__ if hasattr(hyp_class, '__name__') else f'H{i:03d}'
        
        print(f"\n[{i+1}/{len(catalog)}] Testing {hyp_id}...", end=' ')
        
        try:
            # Test on each symbol
            for symbol, data in data_map.items():
                try:
                    hyp = hyp_class()
                    
                    # Find triggers
                    triggers = []
                    for idx, row in data.iterrows():
                        event = hyp.check_trigger(row, data)
                        if event:
                            triggers.append(event)
                    
                    if not triggers:
                        continue
                    
                    # Calculate metrics
                    metrics = calculate_metrics(triggers, symbol)
                    metrics['hyp_id'] = hyp_id
                    metrics['symbol'] = symbol
                    
                    all_results.append(metrics)
                    
                except Exception as e:
                    continue
            
            if any(r['hyp_id'] == hyp_id for r in all_results):
                print("✅")
            else:
                print("❌ (no triggers)")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    if not all_results:
        print("\n❌ NO RESULTS! All hypotheses failed to generate triggers.")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_results)
    
    # Aggregate by hypothesis (average across symbols)
    hyp_stats = df.groupby('hyp_id').agg({
        'profit_factor': 'mean',
        'win_rate': 'mean',
        'max_drawdown': 'mean',
        'total_events': 'sum',
        'avg_return': 'mean'
    }).reset_index()
    
    # Sort by profit factor
    hyp_stats = hyp_stats.sort_values('profit_factor', ascending=False)
    
    # Display results
    print("\n" + "="*70)
    print("TOP 20 HYPOTHESES BY PROFIT FACTOR (NO FILTERS APPLIED)")
    print("="*70)
    print(f"\n{'Rank':<6} {'ID':<12} {'PF':<8} {'WR':<8} {'DD':<8} {'Trades':<8} {'Avg Return':<12}")
    print("-"*70)
    
    for i, row in hyp_stats.head(20).iterrows():
        print(f"{i+1:<6} {row['hyp_id']:<12} "
              f"{row['profit_factor']:<8.2f} "
              f"{row['win_rate']:<8.1%} "
              f"{row['max_drawdown']:<8.1%} "
              f"{int(row['total_events']):<8} "
              f"{row['avg_return']:<12.3%}")
    
    # Statistics
    print("\n" + "="*70)
    print("STATISTICAL DISTRIBUTION")
    print("="*70)
    
    print(f"\nProfit Factor:")
    print(f"  Min: {hyp_stats['profit_factor'].min():.2f}")
    print(f"  25%: {hyp_stats['profit_factor'].quantile(0.25):.2f}")
    print(f"  50%: {hyp_stats['profit_factor'].median():.2f}")
    print(f"  75%: {hyp_stats['profit_factor'].quantile(0.75):.2f}")
    print(f"  Max: {hyp_stats['profit_factor'].max():.2f}")
    
    print(f"\nWin Rate:")
    print(f"  Min: {hyp_stats['win_rate'].min():.1%}")
    print(f"  25%: {hyp_stats['win_rate'].quantile(0.25):.1%}")
    print(f"  50%: {hyp_stats['win_rate'].median():.1%}")
    print(f"  75%: {hyp_stats['win_rate'].quantile(0.75):.1%}")
    print(f"  Max: {hyp_stats['win_rate'].max():.1%}")
    
    print(f"\nMax Drawdown:")
    print(f"  Min: {hyp_stats['max_drawdown'].min():.1%}")
    print(f"  25%: {hyp_stats['max_drawdown'].quantile(0.25):.1%}")
    print(f"  50%: {hyp_stats['max_drawdown'].median():.1%}")
    print(f"  75%: {hyp_stats['max_drawdown'].quantile(0.75):.1%}")
    print(f"  Max: {hyp_stats['max_drawdown'].max():.1%}")
    
    # Filter analysis
    print("\n" + "="*70)
    print("FILTER ANALYSIS - How many pass different thresholds?")
    print("="*70)
    
    filters = {
        'PF ≥ 1.0': (hyp_stats['profit_factor'] >= 1.0).sum(),
        'PF ≥ 1.2': (hyp_stats['profit_factor'] >= 1.2).sum(),
        'PF ≥ 1.4': (hyp_stats['profit_factor'] >= 1.4).sum(),
        'PF ≥ 1.6': (hyp_stats['profit_factor'] >= 1.6).sum(),
        'WR ≥ 50%': (hyp_stats['win_rate'] >= 0.50).sum(),
        'WR ≥ 55%': (hyp_stats['win_rate'] >= 0.55).sum(),
        'WR ≥ 58%': (hyp_stats['win_rate'] >= 0.58).sum(),
        'WR ≥ 60%': (hyp_stats['win_rate'] >= 0.60).sum(),
        'DD ≤ 35%': (hyp_stats['max_drawdown'] <= 0.35).sum(),
        'DD ≤ 25%': (hyp_stats['max_drawdown'] <= 0.25).sum(),
        'Trades ≥ 100': (hyp_stats['total_events'] >= 100).sum(),
        'Trades ≥ 150': (hyp_stats['total_events'] >= 150).sum(),
    }
    
    for filter_name, count in filters.items():
        pct = count / len(hyp_stats) * 100
        print(f"  {filter_name:<20}: {count:>3}/{len(hyp_stats)} ({pct:>5.1f}%)")
    
    # Current A_CORE criteria
    print("\n" + "="*70)
    print("CURRENT A_CORE CRITERIA (from configs)")
    print("="*70)
    
    a_core_pass = hyp_stats[
        (hyp_stats['profit_factor'] >= 1.5) &
        (hyp_stats['win_rate'] >= 0.58) &
        (hyp_stats['max_drawdown'] <= 0.25) &
        (hyp_stats['total_events'] >= 150)
    ]
    
    print(f"\nPF ≥ 1.5, WR ≥ 58%, DD ≤ 25%, Trades ≥ 150")
    print(f"Hypotheses passing: {len(a_core_pass)}/{len(hyp_stats)}")
    
    if len(a_core_pass) > 0:
        print(f"\n✅ WINNERS:")
        for _, row in a_core_pass.iterrows():
            print(f"  {row['hyp_id']}: PF={row['profit_factor']:.2f}, WR={row['win_rate']:.1%}")
    else:
        print(f"\n❌ NO HYPOTHESES PASS CURRENT CRITERIA")
        
        # Suggest relaxed criteria
        print(f"\nSUGGESTED RELAXED CRITERIA:")
        
        # Find 90th percentile
        pf_90 = hyp_stats['profit_factor'].quantile(0.90)
        wr_90 = hyp_stats['win_rate'].quantile(0.90)
        
        print(f"  PF ≥ {pf_90:.2f} (90th percentile)")
        print(f"  WR ≥ {wr_90:.1%} (90th percentile)")
        print(f"  DD ≤ 35% (relaxed)")
        print(f"  Trades ≥ 100 (relaxed)")
        
        relaxed_pass = hyp_stats[
            (hyp_stats['profit_factor'] >= pf_90) &
            (hyp_stats['win_rate'] >= wr_90) &
            (hyp_stats['max_drawdown'] <= 0.35) &
            (hyp_stats['total_events'] >= 100)
        ]
        
        print(f"\nWith relaxed criteria: {len(relaxed_pass)}/{len(hyp_stats)} would pass")
        
        if len(relaxed_pass) > 0:
            print(f"\nTop candidates:")
            for _, row in relaxed_pass.head(5).iterrows():
                print(f"  {row['hyp_id']}: PF={row['profit_factor']:.2f}, "
                      f"WR={row['win_rate']:.1%}, DD={row['max_drawdown']:.1%}, "
                      f"Trades={int(row['total_events'])}")
    
    # Save raw results
    output_file = Path(f"lab/results/debug_mass_screening_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_hypotheses': len(catalog),
            'hypotheses_with_results': len(hyp_stats),
            'top_20': hyp_stats.head(20).to_dict('records'),
            'statistics': {
                'pf': {
                    'min': float(hyp_stats['profit_factor'].min()),
                    'median': float(hyp_stats['profit_factor'].median()),
                    'max': float(hyp_stats['profit_factor'].max())
                },
                'wr': {
                    'min': float(hyp_stats['win_rate'].min()),
                    'median': float(hyp_stats['win_rate'].median()),
                    'max': float(hyp_stats['win_rate'].max())
                }
            },
            'filter_counts': filters
        }, f, indent=2, default=str)
    
    print(f"\n✅ Raw results saved to: {output_file}")
    print()


if __name__ == "__main__":
    analyze_hypothesis_performance()
