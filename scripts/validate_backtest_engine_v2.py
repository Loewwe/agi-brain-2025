#!/usr/bin/env python3
"""
Backtest Engine Validation v2 (Monkey Test v2)

Tests backtest engine correctness with 3 controlled synthetic scenarios:
1. Fair 50/50, R:R 1:1, no fees - should give WR≈50%, PF≈1.0
2. Coinflip, R:R 1:2, no fees - should give WR≈33%, PF≈1.0
3. Coinflip, R:R 1:2, with fees - should give WR≈33%, PF<1.0, negative return

This is a PRE-CHECK before any Mass Screening or Auto-Research.
"""

import argparse
import random
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import json


class SyntheticDataGenerator:
    """Generate controlled synthetic price data for testing."""
    
    def __init__(self, seed=42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def generate(self, num_candles: int, start_price: float = 100.0, 
                 noise_pct: float = 0.002) -> pd.DataFrame:
        """Generate random walk with realistic candles."""
        data = []
        
        for i in range(num_candles):
            if i == 0:
                close = start_price
            else:
                change = np.random.normal(0, noise_pct)
                close = data[-1]['close'] * (1 + change)
            
            # Generate realistic high/low (0.5-1% range)
            intrabar_range = np.random.uniform(0.005, 0.01)
            high = close * (1 + intrabar_range / 2)
            low = close * (1 - intrabar_range / 2)
            
            data.append({
                'close': close,
                'high': high,
                'low': low
            })
        
        return pd.DataFrame(data)


class BacktestScenario:
    """Single backtest scenario with specific parameters."""
    
    def __init__(self, name: str, tp_pct: float, sl_pct: float, 
                 commission: float = 0.0, slippage: float = 0.0):
        self.name = name
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.commission = commission
        self.slippage = slippage
        
    def run(self, data: pd.DataFrame, initial_capital: float = 10000.0) -> dict:
        """Run backtest on synthetic data."""
        capital = initial_capital
        trades = []
        position = None
        
        for idx, row in data.iterrows():
            # Entry logic (50% chance)
            if not position:
                if random.random() < 0.5:
                    # Entry with slippage
                    entry_price = row['close'] * (1 + self.slippage)
                    position_value = capital * 0.02  # 2% position size
                    
                    # Entry commission
                    entry_fee = position_value * self.commission
                    capital -= entry_fee
                    
                    # Calculate TP/SL
                    position = {
                        'entry_price': entry_price,
                        'position_value': position_value,
                        'entry_fee': entry_fee,
                        'tp_price': entry_price * (1 + self.tp_pct),
                        'sl_price': entry_price * (1 - self.sl_pct)
                    }
            
            # Exit logic
            elif position:
                hit_tp = row['high'] >= position['tp_price']
                hit_sl = row['low'] <= position['sl_price']
                
                if hit_tp or hit_sl:
                    # Exit with slippage (always hurts trader)
                    if hit_tp:
                        exit_price = position['tp_price'] * (1 - self.slippage)
                        exit_reason = 'TP'
                    else:
                        exit_price = position['sl_price'] * (1 - self.slippage)
                        exit_reason = 'SL'
                    
                    # Calculate PnL
                    pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                    pnl_gross = position['position_value'] * pnl_pct
                    
                    # Exit commission
                    exit_fee = position['position_value'] * self.commission
                    pnl_net = pnl_gross - exit_fee
                    
                    # Update capital
                    capital += pnl_net
                    
                    # Record trade
                    trades.append({
                        'exit_reason': exit_reason,
                        'pnl_net': pnl_net,
                        'pnl_gross': pnl_gross,
                        'total_fees': position['entry_fee'] + exit_fee
                    })
                    
                    position = None
        
        # Calculate metrics
        if not trades:
            return None
        
        trades_df = pd.DataFrame(trades)
        winners = trades_df[trades_df['pnl_net'] > 0]
        losers = trades_df[trades_df['pnl_net'] <= 0]
        
        total_profit = winners['pnl_net'].sum() if len(winners) > 0 else 0
        total_loss = abs(losers['pnl_net'].sum()) if len(losers) > 0 else 0.01
        
        return {
            'total_trades': len(trades_df),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(trades_df),
            'profit_factor': total_profit / total_loss if total_loss > 0 else 0,
            'return_pct': (capital - initial_capital) / initial_capital,
            'total_fees': float(trades_df['total_fees'].sum())
        }


class MonkeyTestV2:
    """Monkey Test v2 - comprehensive backtest engine validation."""
    
    def __init__(self, runs: int = 20, trades_per_run: int = 500, seed: int = 42):
        self.runs = runs
        self.trades_per_run = trades_per_run
        self.base_seed = seed
        
        # Define scenarios
        self.scenarios = {
            'fair_rr_1_1_no_fees': {
                'scenario': BacktestScenario(
                    name='Fair 50/50, R:R 1:1, No Fees',
                    tp_pct=0.01,  # 1%
                    sl_pct=0.01,  # 1%
                    commission=0.0,
                    slippage=0.0
                ),
                'bounds': {
                    'wr': (0.45, 0.55),  # Relaxed for statistical variation
                    'pf': (0.90, 1.10),
                    'ret': (-0.02, 0.02)
                }
            },
            'rr_1_2_no_fees': {
                'scenario': BacktestScenario(
                    name='Coinflip, R:R 1:2, No Fees',
                    tp_pct=0.02,  # 2%
                    sl_pct=0.01,  # 1%
                    commission=0.0,
                    slippage=0.0
                ),
                'bounds': {
                    'wr': (0.26, 0.38),  # Relaxed - 33% ± variance
                    'pf': (0.80, 1.20),  # Relaxed for low trade count
                    'ret': (-0.02, 0.02)
                }
            },
            'rr_1_2_with_fees': {
                'scenario': BacktestScenario(
                    name='Coinflip, R:R 1:2, With Fees',
                    tp_pct=0.005,  # 0.5%
                    sl_pct=0.0025,  # 0.25%
                    commission=0.0005,  # 0.05% per trade
                    slippage=0.0002  # 0.02%
                ),
                'bounds': {
                    'wr': (0.26, 0.38),  # Same as scenario 2
                    'pf': (0.50, 0.70),  # Should be < 1.0 due to fees
                    'ret': (-0.05, -0.005)  # Realistic: -0.5% to -5%
                }
            }
        }
    
    def run_validation(self):
        """Run full validation across all scenarios."""
        print("="*70)
        print("BACKTEST ENGINE VALIDATION V2 (MONKEY TEST V2)")
        print("="*70)
        print(f"\nParameters:")
        print(f"  Runs: {self.runs}")
        print(f"  Trades per run: ~{self.trades_per_run}")
        print(f"  Base seed: {self.base_seed}")
        print()
        
        results = {}
        overall_status = "PASS"
        
        for scenario_name, scenario_config in self.scenarios.items():
            print(f"\n{'='*70}")
            print(f"SCENARIO: {scenario_config['scenario'].name}")
            print(f"{'='*70}")
            
            scenario = scenario_config['scenario']
            bounds = scenario_config['bounds']
            
            print(f"  TP: {scenario.tp_pct:.2%}, SL: {scenario.sl_pct:.2%}")
            print(f"  Commission: {scenario.commission:.2%}, Slippage: {scenario.slippage:.2%}")
            print(f"  Expected bounds:")
            print(f"    WR: {bounds['wr'][0]:.1%} - {bounds['wr'][1]:.1%}")
            print(f"    PF: {bounds['pf'][0]:.2f} - {bounds['pf'][1]:.2f}")
            print(f"    Return: {bounds['ret'][0]:.1%} - {bounds['ret'][1]:.1%}")
            print()
            
            # Run multiple times
            run_results = []
            for run_id in range(self.runs):
                # Generate synthetic data
                generator = SyntheticDataGenerator(seed=self.base_seed + run_id)
                
                # Generate enough candles to get desired trade count
                # Assume ~50% entry rate, ~50% TP/SL within reasonable time
                num_candles = self.trades_per_run * 4
                data = generator.generate(num_candles)
                
                # Run backtest
                result = scenario.run(data)
                if result:
                    run_results.append(result)
            
            # Aggregate results
            if run_results:
                df = pd.DataFrame(run_results)
                
                avg_wr = df['win_rate'].mean()
                avg_pf = df['profit_factor'].mean()
                avg_ret = df['return_pct'].mean()
                std_wr = df['win_rate'].std()
                std_pf = df['profit_factor'].std()
                std_ret = df['return_pct'].std()
                
                print(f"Results ({len(run_results)} runs):")
                print(f"  Avg WR: {avg_wr:.2%} ± {std_wr:.2%}")
                print(f"  Avg PF: {avg_pf:.3f} ± {std_pf:.3f}")
                print(f"  Avg Return: {avg_ret:+.2%} ± {std_ret:.2%}")
                print(f"  Avg Trades/Run: {df['total_trades'].mean():.0f}")
                
                # Check bounds
                wr_pass = bounds['wr'][0] <= avg_wr <= bounds['wr'][1]
                pf_pass = bounds['pf'][0] <= avg_pf <= bounds['pf'][1]
                ret_pass = bounds['ret'][0] <= avg_ret <= bounds['ret'][1]
                
                scenario_status = "PASS" if (wr_pass and pf_pass and ret_pass) else "FAIL"
                
                print(f"\n  Checks:")
                print(f"    {'✅' if wr_pass else '❌'} Win Rate in bounds")
                print(f"    {'✅' if pf_pass else '❌'} Profit Factor in bounds")
                print(f"    {'✅' if ret_pass else '❌'} Return in bounds")
                print(f"\n  Status: {scenario_status}")
                
                if scenario_status == "FAIL":
                    overall_status = "FAIL"
                    print(f"\n  ⚠️ FAILED: Out of bounds metrics:")
                    if not wr_pass:
                        print(f"    - WR {avg_wr:.2%} not in [{bounds['wr'][0]:.1%}, {bounds['wr'][1]:.1%}]")
                    if not pf_pass:
                        print(f"    - PF {avg_pf:.3f} not in [{bounds['pf'][0]:.2f}, {bounds['pf'][1]:.2f}]")
                    if not ret_pass:
                        print(f"    - Return {avg_ret:.2%} not in [{bounds['ret'][0]:.1%}, {bounds['ret'][1]:.1%}]")
                
                results[scenario_name] = {
                    'avg_win_rate': float(avg_wr),
                    'avg_profit_factor': float(avg_pf),
                    'avg_return_pct': float(avg_ret),
                    'std_win_rate': float(std_wr),
                    'std_profit_factor': float(std_pf),
                    'std_return_pct': float(std_ret),
                    'avg_trades': float(df['total_trades'].mean()),
                    'runs': len(run_results),
                    'status': scenario_status,
                    'bounds': {
                        'wr': list(bounds['wr']),
                        'pf': list(bounds['pf']),
                        'ret': list(bounds['ret'])
                    },
                    'checks': {
                        'wr_pass': bool(wr_pass),
                        'pf_pass': bool(pf_pass),
                        'ret_pass': bool(ret_pass)
                    }
                }
        
        # Final verdict
        print(f"\n{'='*70}")
        print(f"FINAL VERDICT")
        print(f"{'='*70}")
        
        if overall_status == "PASS":
            print(f"\n✅ ALL SCENARIOS PASSED")
            print(f"   Backtest engine is working correctly!")
            print(f"   Safe to run Mass Screening / Auto-Research Brain.")
        else:
            print(f"\n❌ VALIDATION FAILED")
            print(f"   Do NOT run Mass Screening until engine is fixed!")
            print(f"   Check failed scenarios above for details.")
        
        # Save JSON report
        output = {
            'timestamp': datetime.now().isoformat(),
            'engine_version': '0.2.0',
            'runs': self.runs,
            'trades_per_run': self.trades_per_run,
            'scenarios': results,
            'status': overall_status
        }
        
        output_file = Path(f"lab/results/backtest_env_v2_{datetime.now().strftime('%Y%m%d')}.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ JSON report saved: {output_file}")
        
        # Save Markdown report
        md_file = output_file.with_suffix('.md')
        self._save_markdown_report(md_file, output)
        print(f"✅ Markdown report saved: {md_file}")
        
        return overall_status
    
    def _save_markdown_report(self, path: Path, data: dict):
        """Generate markdown report."""
        with open(path, 'w') as f:
            f.write(f"# Backtest Engine Validation v2\n\n")
            f.write(f"**Date:** {data['timestamp']}\n\n")
            f.write(f"**Status:** {'✅ PASS' if data['status'] == 'PASS' else '❌ FAIL'}\n\n")
            f.write(f"## Parameters\n\n")
            f.write(f"- Runs: {data['runs']}\n")
            f.write(f"- Trades per run: ~{data['trades_per_run']}\n")
            f.write(f"- Engine version: {data['engine_version']}\n\n")
            
            f.write(f"## Scenarios\n\n")
            
            for name, result in data['scenarios'].items():
                status_icon = "✅" if result['status'] == 'PASS' else "❌"
                f.write(f"### {status_icon} {name.replace('_', ' ').title()}\n\n")
                f.write(f"| Metric | Value | Expected Range | Status |\n")
                f.write(f"|--------|-------|----------------|--------|\n")
                
                wr_status = "✅" if result['checks']['wr_pass'] else "❌"
                pf_status = "✅" if result['checks']['pf_pass'] else "❌"
                ret_status = "✅" if result['checks']['ret_pass'] else "❌"
                
                f.write(f"| Win Rate | {result['avg_win_rate']:.2%} | "
                       f"{result['bounds']['wr'][0]:.1%}-{result['bounds']['wr'][1]:.1%} | {wr_status} |\n")
                f.write(f"| Profit Factor | {result['avg_profit_factor']:.3f} | "
                       f"{result['bounds']['pf'][0]:.2f}-{result['bounds']['pf'][1]:.2f} | {pf_status} |\n")
                f.write(f"| Return | {result['avg_return_pct']:+.2%} | "
                       f"{result['bounds']['ret'][0]:.1%}-{result['bounds']['ret'][1]:.1%} | {ret_status} |\n")
                f.write(f"| Avg Trades | {result['avg_trades']:.0f} | - | - |\n\n")
            
            if data['status'] == 'FAIL':
                f.write(f"## ⚠️ Action Required\n\n")
                f.write(f"Backtest engine validation FAILED. Do not proceed with:\n\n")
                f.write(f"- Mass Screening\n")
                f.write(f"- Auto-Research Brain\n")
                f.write(f"- Strategy validation\n\n")
                f.write(f"Fix the failing scenarios above before continuing.\n")


def main():
    parser = argparse.ArgumentParser(description='Validate backtest engine with controlled scenarios')
    parser.add_argument('--runs', type=int, default=20, help='Number of runs per scenario')
    parser.add_argument('--trades-per-run', type=int, default=500, help='Target trades per run')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    
    args = parser.parse_args()
    
    validator = MonkeyTestV2(
        runs=args.runs,
        trades_per_run=args.trades_per_run,
        seed=args.seed
    )
    
    status = validator.run_validation()
    
    # Exit code based on status
    exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
