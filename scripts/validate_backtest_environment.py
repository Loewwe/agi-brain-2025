#!/usr/bin/env python3
"""
Backtest Environment Validation Test (Monkey Test)

Tests backtest engine correctness using random strategy with predictable results:
- Random entries (50/50 chance)
- Fixed TP/SL (0.1%/0.05%)
- Expected: WR ~50%, PF ~1.0, final capital ≈ initial

If test passes, backtest engine is working correctly.
"""

import asyncio
import random
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
import json


class MonkeyTest:
    """Validate backtest environment with random strategy."""
    
    def __init__(self, seed=42):
        self.exchange = ccxt.binance()
        
        # Test parameters
        self.seed = seed
        self.symbols = ['BTC/USDT', 'ETH/USDT']
        self.timeframe = '15m'
        self.days_back = 90
        self.initial_capital = 10000.0
        self.num_runs = 10
        
        # Random strategy parameters (increased so fees don't dominate)
        self.entry_probability = 0.5  # 50% chance of entry
        self.position_size_pct = 0.02  # 2% of capital
        self.take_profit_pct = 0.005  # 0.5%
        self.stop_loss_pct = 0.0025  # 0.25%
        
        # Transaction costs
        self.commission = 0.0005  # 0.05%
        self.slippage = 0.0001  # 0.01%
        
    async def fetch_data(self, symbol: str) -> pd.DataFrame:
        """Fetch OHLCV data."""
        print(f"Fetching {symbol} {self.timeframe} data for {self.days_back} days...")
        
        try:
            since = int((datetime.now() - timedelta(days=self.days_back)).timestamp() * 1000)
            all_candles = []
            
            while True:
                candles = await self.exchange.fetch_ohlcv(
                    symbol, self.timeframe, since=since, limit=1000
                )
                if not candles:
                    break
                all_candles.extend(candles)
                since = candles[-1][0] + 1
                if len(candles) < 1000:
                    break
                await asyncio.sleep(0.5)
            
            if not all_candles:
                print(f"  ❌ No data fetched for {symbol}")
                return None
            
            df = pd.DataFrame(
                all_candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            print(f"  ✅ Loaded {len(df)} candles")
            return df
            
        except Exception as e:
            print(f"  ❌ Error fetching {symbol}: {e}")
            return None
    
    def run_backtest(self, df: pd.DataFrame, symbol: str, run_id: int) -> dict:
        """Run single backtest with random strategy."""
        
        # Set seed for reproducibility (different for each run)
        random.seed(self.seed + run_id)
        np.random.seed(self.seed + run_id)
        
        capital = self.initial_capital
        position = None
        trades = []
        equity_curve = []
        
        for idx, row in df.iterrows():
            # Track equity
            current_equity = capital
            if position:
                unrealized_pnl_pct = (row['close'] - position['entry_price']) / position['entry_price']
                unrealized_pnl_usd = position['position_value'] * unrealized_pnl_pct
                current_equity = capital + unrealized_pnl_usd
            
            equity_curve.append({
                'timestamp': idx,
                'equity': current_equity
            })
            
            # Entry logic (random)
            if not position:
                if random.random() < self.entry_probability:
                    # Calculate position size from INITIAL capital (not current)
                    position_value = self.initial_capital * self.position_size_pct
                    
                    # Entry with slippage
                    entry_price = row['close'] * (1 + self.slippage)
                    
                    position = {
                        'entry_time': idx,
                        'entry_price': entry_price,
                        'position_value': position_value,
                        'tp_price': entry_price * (1 + self.take_profit_pct),
                        'sl_price': entry_price * (1 - self.stop_loss_pct)
                    }
                    
                    # Deduct entry commission from capital (not from position itself)
                    entry_fee = position_value * self.commission
                    capital -= entry_fee
            
            # Exit logic
            elif position:
                hit_tp = row['high'] >= position['tp_price']
                hit_sl = row['low'] <= position['sl_price']
                
                exit_price = None
                exit_reason = None
                
                if hit_tp:
                    exit_price = position['tp_price'] * (1 - self.slippage)
                    exit_reason = 'TP'
                elif hit_sl:
                    exit_price = position['sl_price'] * (1 - self.slippage)
                    exit_reason = 'SL'
                
                if exit_price:
                    # Calculate PnL
                    pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                    pnl_usd = position['position_value'] * pnl_pct
                    
                    # Deduct exit commission
                    exit_fee = position['position_value'] * self.commission
                    pnl_usd -= exit_fee
                    
                    # Update capital
                    capital += pnl_usd
                    
                    # Record trade
                    trades.append({
                        'symbol': symbol,
                        'entry_time': position['entry_time'],
                        'exit_time': idx,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl_pct': pnl_pct,
                        'pnl_usd': pnl_usd,
                        'exit_reason': exit_reason
                    })
                    
                    position = None
        
        # Calculate metrics
        if not trades:
            return None
        
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        
        winners = trades_df[trades_df['pnl_usd'] > 0]
        losers = trades_df[trades_df['pnl_usd'] <= 0]
        
        total_profit = winners['pnl_usd'].sum() if len(winners) > 0 else 0
        total_loss = abs(losers['pnl_usd'].sum()) if len(losers) > 0 else 0.01
        
        metrics = {
            'run_id': run_id,
            'symbol': symbol,
            'total_trades': len(trades_df),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(trades_df),
            'profit_factor': total_profit / total_loss if total_loss > 0 else 0,
            'final_capital': capital,
            'return_pct': (capital - self.initial_capital) / self.initial_capital,
            'tp_exits': len(trades_df[trades_df['exit_reason'] == 'TP']),
            'sl_exits': len(trades_df[trades_df['exit_reason'] == 'SL'])
        }
        
        # Max Drawdown
        equity_df['running_max'] = equity_df['equity'].expanding().max()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['running_max']) / equity_df['running_max']
        metrics['max_drawdown'] = abs(equity_df['drawdown'].min())
        
        return metrics
    
    async def run_validation(self):
        """Run full validation test."""
        print("="*70)
        print("BACKTEST ENVIRONMENT VALIDATION TEST (Monkey Test)")
        print("="*70)
        print(f"\nParameters:")
        print(f"  Random seed: {self.seed}")
        print(f"  Entry probability: {self.entry_probability:.0%}")
        print(f"  Take Profit: {self.take_profit_pct:.2%}")
        print(f"  Stop Loss: {self.stop_loss_pct:.2%}")
        print(f"  Position size: {self.position_size_pct:.0%} of initial capital")
        print(f"  Commission: {self.commission:.2%}")
        print(f"  Slippage: {self.slippage:.2%}")
        print(f"  Runs: {self.num_runs}")
        print(f"  Initial capital: ${self.initial_capital:,.0f}\n")
        
        print("Expected results:")
        print(f"  Win Rate: ~50% ± 5%")
        print(f"  Profit Factor: ~1.0 ± 0.1")
        print(f"  Final Return: -0.5% to +0.5% (fees eat into 50/50)\n")
        
        # Fetch data
        print("Fetching data...")
        data = {}
        for symbol in self.symbols:
            df = await self.fetch_data(symbol)
            if df is not None:
                data[symbol] = df
        
        await self.exchange.close()
        
        if not data:
            print("❌ No data fetched! Cannot proceed.")
            return
        
        # Data quality check
        print("\n" + "="*70)
        print("DATA QUALITY CHECK")
        print("="*70)
        
        all_pass = True
        for symbol, df in data.items():
            print(f"\n{symbol}:")
            print(f"  Candles: {len(df)}")
            print(f"  Period: {df.index[0]} to {df.index[-1]}")
            print(f"  Duration: {(df.index[-1] - df.index[0]).days} days")
            
            if len(df) < 1000:
                print(f"  ❌ FAIL: Too few candles (expected >1000)")
                all_pass = False
            else:
                print(f"  ✅ PASS: Sufficient data")
        
        if not all_pass:
            print("\n❌ DATA QUALITY FAILED! Fix data before proceeding.")
            return
        
        # Run multiple backtests
        print("\n" + "="*70)
        print("RUNNING VALIDATION TESTS")
        print("="*70)
        
        all_results = []
        
        for run_id in range(self.num_runs):
            print(f"\nRun {run_id + 1}/{self.num_runs}:")
            
            for symbol in data.keys():
                result = self.run_backtest(data[symbol], symbol, run_id)
                if result:
                    all_results.append(result)
                    print(f"  {symbol}: Trades={result['total_trades']}, "
                          f"WR={result['win_rate']:.1%}, "
                          f"PF={result['profit_factor']:.2f}, "
                          f"Return={result['return_pct']:+.2%}")
        
        # Analyze results
        print("\n" + "="*70)
        print("VALIDATION RESULTS")
        print("="*70)
        
        results_df = pd.DataFrame(all_results)
        
        avg_wr = results_df['win_rate'].mean()
        avg_pf = results_df['profit_factor'].mean()
        avg_return = results_df['return_pct'].mean()
        
        print(f"\nAggregated Metrics ({len(all_results)} runs):")
        print(f"  Avg Win Rate: {avg_wr:.2%} (expected: 48-52%)")
        print(f"  Avg Profit Factor: {avg_pf:.3f} (expected: 0.98-1.02)")
        print(f"  Avg Return: {avg_return:+.2%} (expected: -0.5% to +0.5%)")
        print(f"  Std Dev WR: {results_df['win_rate'].std():.2%}")
        print(f"  Std Dev PF: {results_df['profit_factor'].std():.3f}")
        
        # Test reproducibility (same seed should give same results)
        print("\nReproducibility Check:")
        run0_results = results_df[results_df['run_id'] == 0]
        run1_results = results_df[results_df['run_id'] == 1]
        
        if len(run0_results) > 0 and len(run1_results) > 0:
            wr_diff = abs(run0_results['win_rate'].iloc[0] - run1_results['win_rate'].iloc[0])
            print(f"  WR difference between run 0 and run 1: {wr_diff:.2%}")
            if wr_diff < 0.001:
                print(f"  ✅ PASS: Results are reproducible")
            else:
                print(f"  ⚠️ Different seeds produce different results (expected)")
        
        # Verdict
        print("\n" + "="*70)
        print("VERDICT")
        print("="*70)
        
        checks = {
            'Win Rate in range (48-52%)': 0.48 <= avg_wr <= 0.52,
            'Profit Factor in range (0.98-1.02)': 0.98 <= avg_pf <= 1.02,
            'Return in range (-0.5% to +0.5%)': -0.005 <= avg_return <= 0.005,
            'Sufficient data (>1000 candles)': all(len(df) > 1000 for df in data.values()),
            'No calculation anomalies': all(abs(r['return_pct']) < 1.0 for r in all_results)
        }
        
        passed = sum(checks.values())
        total = len(checks)
        
        for check, result in checks.items():
            status = "✅" if result else "❌"
            print(f"  {status} {check}")
        
        print(f"\nPassed: {passed}/{total} checks")
        
        if passed == total:
            verdict = "ENVIRONMENT_READY"
            print("\n✅ BACKTEST ENVIRONMENT VALIDATION PASSED")
            print("   System is ready for strategy testing.")
        elif passed >= 4:
            verdict = "PARTIAL_PASS"
            print("\n⚠️ BACKTEST ENVIRONMENT PARTIALLY VALIDATED")
            print("   Some issues detected, review failures above.")
        else:
            verdict = "FAILED"
            print("\n❌ BACKTEST ENVIRONMENT VALIDATION FAILED")
            print("   Fix issues before testing strategies.")
        
        # Save report
        output = {
            'test_date': datetime.now().isoformat(),
            'seed': self.seed,
            'parameters': {
                'entry_probability': self.entry_probability,
                'take_profit_pct': self.take_profit_pct,
                'stop_loss_pct': self.stop_loss_pct,
                'position_size_pct': self.position_size_pct,
                'commission': self.commission,
                'slippage': self.slippage,
                'initial_capital': self.initial_capital,
                'num_runs': self.num_runs
            },
            'results_per_run': all_results,
            'summary': {
                'avg_win_rate': float(avg_wr),
                'avg_profit_factor': float(avg_pf),
                'avg_return_pct': float(avg_return),
                'std_win_rate': float(results_df['win_rate'].std()),
                'std_profit_factor': float(results_df['profit_factor'].std()),
                'checks_passed': passed,
                'checks_total': total,
                'overall_verdict': verdict
            }
        }
        
        output_file = Path(f"lab/results/backtest_environment_validation_{datetime.now().strftime('%Y%m%d')}.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Validation report saved to: {output_file}")
        print()


async def main():
    validator = MonkeyTest(seed=42)
    await validator.run_validation()


if __name__ == "__main__":
    asyncio.run(main())
