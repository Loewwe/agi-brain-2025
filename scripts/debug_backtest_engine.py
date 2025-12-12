#!/usr/bin/env python3
"""
Debug Backtest Engine - Isolated Test of 5 Trades

Tests core backtest logic with detailed logging to identify bugs:
- Entry/Exit price calculation
- TP/SL trigger logic  
- Commission application
- Capital management

Expected results for 50/50 strategy with TP=0.1%, SL=0.05%:
- Win Rate: ~33% (NOT 50%! Because TP is 2x SL)
- Profit Factor: ~1.0
- Net PnL: Slightly negative due to fees
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


class BacktestDebugger:
    """Minimal backtest engine for debugging 5 trades."""
    
    def __init__(self, seed=42):
        self.initial_capital = 10000.0
        # Random strategy parameters (increased to make fees less dominant)
        self.entry_probability = 0.5  # 50% chance of entry
        self.position_size_pct = 0.02  # 2% of capital
        self.take_profit_pct = 0.005  # 0.5% (was 0.1%)
        self.stop_loss_pct = 0.0025  # 0.25% (was 0.05%)
        self.commission = 0.0005  # 0.05%
        self.slippage = 0.0001  # 0.01%
        
        self.capital = self.initial_capital
        self.trades = []
        
        random.seed(seed)
        np.random.seed(seed)
        
    def generate_test_data(self, num_candles=100) -> pd.DataFrame:
        """Generate simple random walk price data with realistic candles."""
        print(f"\nGenerating {num_candles} candles of test data...")
        
        start_price = 50000.0
        data = []
        
        for i in range(num_candles):
            # Generate close price (random walk)
            if i == 0:
                close = start_price
            else:
                change_pct = np.random.normal(0, 0.003)  # 0.3% std dev
                close = data[-1]['close'] * (1 + change_pct)
            
            # Generate realistic high/low (0.5-1% range to allow TP/SL hits)
            intrabar_range = np.random.uniform(0.005, 0.01)  # 0.5-1% range
            high = close * (1 + intrabar_range / 2)
            low = close * (1 - intrabar_range / 2)
            
            data.append({
                'close': close,
                'high': high,
                'low': low
            })
        
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=7), 
            periods=num_candles, 
            freq='15min'
        )
        
        df = pd.DataFrame(data)
        df['timestamp'] = timestamps
        df.set_index('timestamp', inplace=True)
        
        print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"  Avg candle range: {((df['high'] - df['low']) / df['close']).mean():.2%}")
        return df
    
    def run_debug_backtest(self, df: pd.DataFrame, max_trades=5):
        """Run backtest with detailed logging."""
        
        print("\n" + "="*70)
        print("BACKTEST DEBUG SESSION")
        print("="*70)
        print(f"\nInitial Capital: ${self.capital:,.2f}")
        print(f"Position Size: {self.position_size_pct:.0%} of INITIAL capital")
        print(f"Take Profit: {self.take_profit_pct:.2%}")
        print(f"Stop Loss: {self.stop_loss_pct:.2%}")
        print(f"Commission: {self.commission:.2%}")
        print(f"Slippage: {self.slippage:.2%}\n")
        
        position = None
        trade_count = 0
        
        for idx, row in df.iterrows():
            # Entry logic
            if not position and trade_count < max_trades:
                if random.random() < 0.5:  # 50% chance
                    trade_count += 1
                    
                    print(f"\n{'='*70}")
                    print(f"TRADE #{trade_count}")
                    print(f"{'='*70}")
                    print(f"Time: {idx}")
                    print(f"Market Price: ${row['close']:,.2f}")
                    
                    # Calculate position size from INITIAL capital (NOT current)
                    position_value = self.initial_capital * self.position_size_pct
                    print(f"\nPosition Sizing:")
                    print(f"  Initial Capital: ${self.initial_capital:,.2f}")
                    print(f"  Current Capital: ${self.capital:,.2f}")
                    print(f"  Position %: {self.position_size_pct:.0%}")
                    print(f"  Position Value: ${position_value:,.2f}")
                    
                    # Entry with slippage
                    entry_price = row['close'] * (1 + self.slippage)
                    print(f"\nEntry Execution:")
                    print(f"  Market Price: ${row['close']:,.2f}")
                    print(f"  Slippage: +{self.slippage:.2%}")
                    print(f"  Entry Price: ${entry_price:,.2f}")
                    
                    # Entry commission
                    entry_fee = position_value * self.commission
                    print(f"  Entry Fee: ${entry_fee:.2f} ({self.commission:.2%} of ${position_value:,.2f})")
                    
                    # Update capital (deduct fee immediately)
                    self.capital -= entry_fee
                    print(f"  Capital After Fee: ${self.capital:,.2f}")
                    
                    # Calculate TP/SL levels
                    tp_price = entry_price * (1 + self.take_profit_pct)
                    sl_price = entry_price * (1 - self.stop_loss_pct)
                    
                    print(f"\nExit Levels:")
                    print(f"  Entry: ${entry_price:,.2f}")
                    print(f"  TP: ${tp_price:,.2f} (+{self.take_profit_pct:.2%})")
                    print(f"  SL: ${sl_price:,.2f} (-{self.stop_loss_pct:.2%})")
                    print(f"  R:R Ratio: 1:{self.take_profit_pct/self.stop_loss_pct:.1f}")
                    
                    position = {
                        'entry_time': idx,
                        'entry_price': entry_price,
                        'position_value': position_value,
                        'tp_price': tp_price,
                        'sl_price': sl_price,
                        'trade_num': trade_count
                    }
            
            # Exit logic
            elif position:
                hit_tp = row['high'] >= position['tp_price']
                hit_sl = row['low'] <= position['sl_price']
                
                if hit_tp or hit_sl:
                    print(f"\n--- Exit Triggered ---")
                    print(f"Time: {idx}")
                    print(f"Candle: High=${row['high']:,.2f}, Low=${row['low']:,.2f}, Close=${row['close']:,.2f}")
                    
                    # Slippage logic for LONG positions (we're always long in this test)
                    # Long exit = selling, so slippage makes us sell LOWER (worse price)
                    # For short positions, it would be (1 + slippage) since we'd buy higher
                    
                    if hit_tp:
                        # Long TP: sell at TP level, slippage makes us sell lower
                        exit_price = position['tp_price'] * (1 - self.slippage)
                        exit_reason = 'TP'
                        print(f"✅ Take Profit Hit!")
                        print(f"  TP Level: ${position['tp_price']:,.2f}")
                        print(f"  Slippage: -{self.slippage:.2%} (sell lower than TP)")
                        print(f"  Exit Price: ${exit_price:,.2f}")
                    else:
                        # Long SL: sell at SL level, slippage makes us sell even lower (larger loss)
                        exit_price = position['sl_price'] * (1 - self.slippage)
                        exit_reason = 'SL'
                        print(f"❌ Stop Loss Hit!")
                        print(f"  SL Level: ${position['sl_price']:,.2f}")
                        print(f"  Slippage: -{self.slippage:.2%} (sell lower than SL, larger loss)")
                        print(f"  Exit Price: ${exit_price:,.2f}")
                    
                    # Calculate PnL
                    pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                    pnl_gross = position['position_value'] * pnl_pct
                    
                    print(f"\nP&L Calculation:")
                    print(f"  Entry Price: ${position['entry_price']:,.2f}")
                    print(f"  Exit Price: ${exit_price:,.2f}")
                    print(f"  Price Change: {pnl_pct:+.3%}")
                    print(f"  Position Value: ${position['position_value']:,.2f}")
                    print(f"  Gross P&L: ${pnl_gross:+,.2f}")
                    
                    # Exit commission
                    exit_fee = position['position_value'] * self.commission
                    pnl_net = pnl_gross - exit_fee
                    
                    print(f"  Exit Fee: ${exit_fee:.2f} ({self.commission:.2%})")
                    print(f"  Net P&L: ${pnl_net:+,.2f}")
                    
                    # Update capital
                    self.capital += pnl_net
                    
                    print(f"\nCapital Update:")
                    print(f"  Before: ${self.capital - pnl_net:,.2f}")
                    print(f"  +P&L: ${pnl_net:+,.2f}")
                    print(f"  After: ${self.capital:,.2f}")
                    print(f"  Total Return: {(self.capital - self.initial_capital) / self.initial_capital:+.2%}")
                    
                    # Record trade
                    self.trades.append({
                        'trade_num': position['trade_num'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl_gross': pnl_gross,
                        'entry_fee': entry_fee,
                        'exit_fee': exit_fee,
                        'pnl_net': pnl_net
                    })
                    
                    position = None
                    
                    if trade_count >= max_trades:
                        break
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        trades_df = pd.DataFrame(self.trades)
        
        winners = trades_df[trades_df['pnl_net'] > 0]
        losers = trades_df[trades_df['pnl_net'] <= 0]
       
        print(f"\nTrades: {len(trades_df)}")
        print(f"Winners: {len(winners)} ({len(winners)/len(trades_df):.1%})")
        print(f"Losers: {len(losers)} ({len(losers)/len(trades_df):.1%})")
        
        total_profit = winners['pnl_net'].sum() if len(winners) > 0 else 0
        total_loss = abs(losers['pnl_net'].sum()) if len(losers) > 0 else 0.01
        
        print(f"\nProfit Factor: {total_profit / total_loss:.3f}")
        print(f"Total Fees Paid: ${trades_df['entry_fee'].sum() + trades_df['exit_fee'].sum():.2f}")
        print(f"\nFinal Capital: ${self.capital:,.2f}")
        print(f"Total Return: {(self.capital - self.initial_capital) / self.initial_capital:+.2%}")
        
        print("\n" + "="*70)
        print("EXPECTED vs ACTUAL")
        print("="*70)
        
        print(f"\n❓ Mathematical Expectation for 50/50 strategy:")
        print(f"   With TP=0.1%, SL=0.05% (R:R = 1:2)")
        print(f"   Break-even Win Rate = SL / (TP + SL) = 0.05% / 0.15% = 33.3%")
        print(f"   Expected PF ≈ 1.0 (before fees)")
        print(f"   Expected PF ≈ 0.97-1.0 (after 0.1% round-trip fees)")
        
        actual_wr = len(winners) / len(trades_df)
        actual_pf = total_profit / total_loss
        
        print(f"\n✅ ACTUAL Results:")
        print(f"   Win Rate: {actual_wr:.1%} (expected ~33%)")
        print(f"   Profit Factor: {actual_pf:.3f} (expected ~0.97-1.0)")
        
        if actual_wr < 0.28 or actual_wr > 0.38:
            print(f"\n⚠️ WARNING: Win Rate {actual_wr:.1%} is outside expected range 28-38%")
            print(f"   This suggests a bug in TP/SL trigger logic!")
        
        if actual_pf < 0.90 or actual_pf > 1.10:
            print(f"\n⚠️ WARNING: Profit Factor {actual_pf:.3f} is outside expected range 0.90-1.10")
            print(f"   This suggests a bug in P&L calculation or fees!")


def main():
    debugger = BacktestDebugger(seed=42)
    data = debugger.generate_test_data(num_candles=100)
    debugger.run_debug_backtest(data, max_trades=5)


if __name__ == "__main__":
    main()
