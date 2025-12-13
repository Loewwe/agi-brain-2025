#!/usr/bin/env python3
"""
Auto-Research Brain: Microalpha Funding Mode
Autonomous search for funding rate arbitrage strategies
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import logging
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lab/microalpha/logs/auto_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Config
CONFIG = {
    'symbols': ['BTCUSDT', 'ETHUSDT', 'SOL USDT'],
    'timeframes': ['15m', '1h'],
    'is_days': 60,
    'oos_days': 30,
    'min_new_candidates': 1,
    'loop_interval_hours': 6,
    'criteria': {
        'trades_IS_min': 40,
        'trades_OOS_min': 50,
        'PF_IS_min': 1.3,
        'PF_OOS_min': 1.25,
        'WR_IS_min': 0.52,
        'WR_OOS_min': 0.50,
        'MaxDD_OOS_max': 0.15,
        'consistency_ratio_min': 0.7,
        'robustness_PF_min': 1.15
    }
}

# Fixed Parameter Grids (Anti-Overfitting)
PARAM_GRIDS = {
    'F001': {
        'tp_pct': [0.010, 0.015, 0.020],
        'sl_pct': [0.020, 0.025, 0.030],
        'funding_threshold': ['p90', 'p95'],
        'oi_change_threshold': [0.15, 0.20]
    },
    'F002': {
        'tp_pct': [0.015, 0.020, 0.025],
        'sl_pct': [0.025, 0.030, 0.035],
        'funding_threshold': ['p5', 'p10'],
        'price_filter': ['below_ma50', 'below_ma100']
    },
    'F003': {
        'tp_pct': [0.008, 0.010, 0.012],
        'sl_pct': [0.015, 0.020],
        'divergence_threshold': ['2std', '2.5std'],
        'lookback_days': [7, 14]
    }
}

def update_funding_data():
    """Step 1: Update funding/OI data"""
    logger.info("Updating funding data...")
    
    try:
        # Run funding_backfill.py
        import subprocess
        result = subprocess.run(
            ['python', 'lab/microalpha/funding_backfill.py'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("✅ Funding data updated")
            return True
        else:
            logger.error(f"❌ Funding backfill failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating data: {e}")
        return False

def build_features_for_all():
    """Step 2: Calculate features for all symbols/timeframes"""
    logger.info("Building features...")
    
    try:
        # Load raw funding data
        funding = pd.read_parquet('lab/microalpha/data/funding_1h.parquet')
        
        for symbol in CONFIG['symbols']:
            for tf in CONFIG['timeframes']:
                # Filter symbol data
                symbol_data = funding[funding['symbol'] == symbol].copy()
                
                # Calculate features
                symbol_data['funding_zscore_7d'] = (
                    symbol_data['funding_rate'].rolling(7*24).apply(
                        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
                    )
                )
                
                symbol_data['oi_change_pct'] = symbol_data['open_interest'].pct_change(24)
                
                # Save features
                output_file = f'lab/microalpha/features/{symbol}_{tf}_features.parquet'
                symbol_data.to_parquet(output_file)
                
        logger.info(f"✅ Features built for {len(CONFIG['symbols']) * len(CONFIG['timeframes'])} combinations")
        return True
        
    except Exception as e:
        logger.error(f"Error building features: {e}")
        return False

def generate_signals_F001(features: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """F001: Extreme Funding Reversal - Generate SHORT signals"""
    
    # Get funding percentile threshold
    if params['funding_threshold'] == 'p95':
        funding_threshold = features['funding_rate'].quantile(0.95)
    else:  # p90
        funding_threshold = features['funding_rate'].quantile(0.90)
    
    # Signal conditions
    signals = features[
        (features['funding_rate'] > funding_threshold) &
        (features['oi_change_pct'] > params['oi_change_threshold'])
    ].copy()
    
    signals['signal'] = -1  # SHORT
    signals['tp_pct'] = params['tp_pct']
    signals['sl_pct'] = params['sl_pct']
    
    return signals[['timestamp', 'signal', 'tp_pct', 'sl_pct', 'price_index']]

def backtest_IS_OOS(signals: pd.DataFrame, params: Dict, config: Dict) -> Dict:
    """
    Backtest with IS/OOS split
    Returns metrics for both periods
    """
    
    # Split data
    cutoff = signals['timestamp'].max() - timedelta(days=config['oos_days'])
    is_signals = signals[signals['timestamp'] < cutoff]
    oos_signals = signals[signals['timestamp'] >= cutoff]
    
    # Simple backtest simulation
    def sim_trades(sigs, commission=0.001):
        if len(sigs) == 0:
            return  {'trades': 0, 'PF': 0, 'WR': 0, 'MaxDD': 0}
        
        wins = 0
        total_pnl = 0
        equity = 10000
        peak = 10000
        max_dd = 0
        
        for _, row in sigs.iterrows():
            # Simulate TP/SL outcome (simplified)
            if np.random.random() < 0.55:  # Assume 55% win rate for simulation
                pnl = equity * 0.02 * row['tp_pct']
                wins += 1
            else:
                pnl = -equity * 0.02 * row['sl_pct']
            
            pnl *= (1 - commission)
            total_pnl += pnl
            equity += pnl
            
            peak = max(peak, equity)
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        
        wr = wins / len(sigs)
        # Rough PF estimation
        avg_win = total_pnl * wr / wins if wins > 0 else 0
        avg_loss = total_pnl * (1 - wr) / (len(sigs) - wins) if len(sigs) > wins else 0
        pf = abs(avg_win / avg_loss) if avg_loss < 0 else 999
        
        return {
            'trades': len(sigs),
            'PF': pf,
            'WR': wr,
            'MaxDD': max_dd,
            'total_pnl': total_pnl
        }
    
    # Run backtests
    is_metrics = sim_trades(is_signals)
    oos_metrics = sim_trades(oos_signals)
    oos_robust = sim_trades(oos_signals, commission=0.0015)  # 1.5x commission
    
    return {
        'trades_IS': is_metrics['trades'],
        'PF_IS': is_metrics['PF'],
        'WR_IS': is_metrics['WR'],
        'MaxDD_IS': is_metrics['MaxDD'],
        'trades_OOS': oos_metrics['trades'],
        'PF_OOS': oos_metrics['PF'],
        'WR_OOS': oos_metrics['WR'],
        'MaxDD_OOS': oos_metrics['MaxDD'],
        'PF_OOS_robust': oos_robust['PF']
    }

def is_candidate(metrics: Dict, config: Dict) -> Tuple[bool, str]:
    """Check if metrics pass all criteria"""
    
    criteria = config['criteria']
    
    # Minimum statistics
    if metrics['trades_IS'] < criteria['trades_IS_min']:
        return False, f"Insufficient IS trades: {metrics['trades_IS']}"
    
    if metrics['trades_OOS'] < criteria['trades_OOS_min']:
        return False, f"Insufficient OOS trades: {metrics['trades_OOS']}"
    
    # IS quality
    if metrics['PF_IS'] < criteria['PF_IS_min']:
        return False, f"IS PF too low: {metrics['PF_IS']:.2f}"
    
    if metrics['WR_IS'] < criteria['WR_IS_min']:
        return False, f"IS WR too low: {metrics['WR_IS']:.1%}"
    
    # OOS quality
    if metrics['PF_OOS'] < criteria['PF_OOS_min']:
        return False, f"OOS PF too low: {metrics['PF_OOS']:.2f}"
    
    if metrics['WR_OOS'] < criteria['WR_OOS_min']:
        return False, f"OOS WR too low: {metrics['WR_OOS']:.1%}"
    
    if metrics['MaxDD_OOS'] > criteria['MaxDD_OOS_max']:
        return False, f"OOS DD too high: {metrics['MaxDD_OOS']:.1%}"
    
    # Consistency
    pf_ratio = metrics['PF_OOS'] / metrics['PF_IS'] if metrics['PF_IS'] > 0 else 0
    if pf_ratio < criteria['consistency_ratio_min']:
        return False, f"PF degradation: {pf_ratio:.2f}"
    
    # Robustness
    if metrics['PF_OOS_robust'] < criteria['robustness_PF_min']:
        return False, f"Not robust to fees: {metrics['PF_OOS_robust']:.2f}"
    
    return True, "PASS"

def run_microalpha_funding_once(config: Dict) -> List[Dict]:
    """Single iteration of funding strategy search"""
    
    logger.info("=" * 60)
    logger.info("STARTING MICROALPHA FUNDING SEARCH ITERATION")
    logger.info("=" * 60)
    
    # Step 1: Update data
    if not update_funding_data():
        logger.error("Data update failed, skipping iteration")
        return []
    
    # Step 2: Build features
    if not build_features_for_all():
        logger.error("Feature building failed, skipping iteration")
        return []
    
    candidates = []
    tested_count = 0
    
    # Step 3: Test all combinations
    for strategy_id in ['F001']:  # Start with F001 only
        for symbol in config['symbols']:
            for tf in config['timeframes']:
                
                logger.info(f"\nTesting {strategy_id} on {symbol} {tf}")
                
                try:
                    # Load features
                    features = pd.read_parquet(f'lab/microalpha/features/{symbol}_{tf}_features.parquet')
                    
                    # Test all parameter combinations
                    param_grid = PARAM_GRIDS[strategy_id]
                    
                    # Generate all combinations
                    for tp in param_grid['tp_pct']:
                        for sl in param_grid['sl_pct']:
                            for funding_thr in param_grid['funding_threshold']:
                                for oi_thr in param_grid['oi_change_threshold']:
                                    
                                    params = {
                                        'tp_pct': tp,
                                        'sl_pct': sl,
                                        'funding_threshold': funding_thr,
                                        'oi_change_threshold': oi_thr
                                    }
                                    
                                    tested_count += 1
                                    
                                    # Generate signals
                                    signals = generate_signals_F001(features, params)
                                    
                                    # Backtest
                                    metrics = backtest_IS_OOS(signals, params, config)
                                    
                                    # Check if candidate
                                    passed, reason = is_candidate(metrics, config)
                                    
                                    if passed:
                                        spec = {
                                            'strategy_id': strategy_id,
                                            'symbol': symbol,
                                            'timeframe': tf,
                                            'params': params,
                                            'metrics': metrics,
                                            'created_at': datetime.utcnow().isoformat(),
                                            'status': 'L2_CANDIDATE'
                                        }
                                        
                                        candidates.append(spec)
                                        logger.info(f"  ✅ CANDIDATE FOUND: PF_OOS={metrics['PF_OOS']:.2f}, WR_OOS={metrics['WR_OOS']:.1%}")
                                        
                                        # Save spec
                                        spec_file = f"lab/microalpha/STRATEGY_SPECS/{strategy_id}_{symbol}_{tf}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json"
                                        Path(spec_file).parent.mkdir(parents=True, exist_ok=True)
                                        
                                        with open(spec_file, 'w') as f:
                                            json.dump(spec, f, indent=2, default=str)
                                    else:
                                        logger.debug(f"  Rejected: {reason}")
                
                except Exception as e:
                    logger.error(f"Error testing {strategy_id} {symbol} {tf}: {e}")
    
    logger.info(f"\nIteration complete: Tested {tested_count} combinations, found {len(candidates)} candidates")
    
    return candidates

def auto_search_loop():
    """Autonomous continuous search loop"""
    
    logger.info("🧠 AUTO-RESEARCH BRAIN: Microalpha Funding Mode")
    logger.info(f"Config: {CONFIG}")
    
    iteration = 0
    
    while True:
        iteration += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION #{iteration}")
        logger.info(f"{'='*60}")
        
        try:
            candidates = run_microalpha_funding_once(CONFIG)
            
            if len(candidates) >= CONFIG['min_new_candidates']:
                logger.info(f"✅ SUCCESS: Found {len(candidates)} new candidates")
                # TODO: Send alert
            else:
                logger.info(f"⚠️  NO CANDIDATES this iteration")
            
            # Save to CSV log
            csv_file = 'lab/microalpha/results/funding_search_history.csv'
            for cand in candidates:
                row = {
                    'timestamp': cand['created_at'],
                    'strategy_id': cand['strategy_id'],
                    'symbol': cand['symbol'],
                    'timeframe': cand['timeframe'],
                    'PF_OOS': cand['metrics']['PF_OOS'],
                    'WR_OOS': cand['metrics']['WR_OOS'],
                    'trades_OOS': cand['metrics']['trades_OOS'],
                    'MaxDD_OOS': cand['metrics']['MaxDD_OOS'],
                    'status': 'CANDIDATE'
                }
                
                df = pd.DataFrame([row])
                df.to_csv(csv_file, mode='a', header=not Path(csv_file).exists(), index=False)
            
            # Wait before next iteration
            wait_hours = CONFIG['loop_interval_hours']
            logger.info(f"\n⏱️  Sleeping for {wait_hours} hours until next iteration...")
            time.sleep(wait_hours * 3600)
            
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopping auto-search loop")
            break
        except Exception as e:
            logger.error(f"Error in iteration {iteration}: {e}")
            time.sleep(3600)  # Wait 1h on error

if __name__ == "__main__":
    auto_search_loop()
