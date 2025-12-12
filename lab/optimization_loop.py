#!/usr/bin/env python3
"""
Stage 2 Optimization v2.0

Reads new_strategies_pool.json from Stage 1,
runs parameter tuning for each strategy,
outputs best_live_strategies.json for Stage6.

Usage:
  python lab/optimization_loop.py
"""

import asyncio
import json
import random
import glob
import os
from datetime import datetime
from pathlib import Path
from statistics import pstdev
from typing import Dict

import yaml
import pandas as pd
import numpy as np

# Import Stage 1 backtester
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from backtest_stage6 import DataFetcher


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "lab" / "results"


def load_config():
    cfg_path = BASE_DIR / "optimization_config.yml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def find_latest_pool_file(pool_dir: Path, pattern: str) -> Path:
    files = sorted(
        pool_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"No strategies pool files matching {pattern} in {pool_dir}")
    return files[0]


def load_strategies_pool(path: Path):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Strategies pool file must contain a JSON list")
    return data


def get_param_space_for_strategy(strategy: dict, cfg: dict) -> dict:
    pattern_type = strategy.get("pattern_type")
    spaces_all = cfg.get("param_spaces", {})
    space = spaces_all.get(pattern_type)
    if not space:
        # If no specific space defined, return empty (no tuning)
        return {}
    return space


def sample_params(param_space: dict) -> dict:
    """
    Sample one set of tunable parameters from given ranges.
    param_space comes from cfg['param_spaces'][pattern_type].
    """
    sampled = {}
    for name, spec in param_space.items():
        if "choices" in spec:
            sampled[name] = random.choice(spec["choices"])
        else:
            # assume continuous range
            v = random.uniform(float(spec["min"]), float(spec["max"]))
            # round to reasonable precision
            sampled[name] = round(v, 4)
    return sampled


def merge_params_into_dsl(base_dsl: dict, sampled_params: dict) -> dict:
    """
    Given base DSL from Stage 1 and a dict of sampled params,
    return a new DSL with these params applied.

    This is a hook: adjust according to how params are stored in DSL.
    """
    dsl = json.loads(json.dumps(base_dsl))  # deep copy via json
    base_params = dsl.setdefault("base_params", {})
    for k, v in sampled_params.items():
        base_params[k] = v
    return dsl


def compute_score(metrics: dict, cfg: dict) -> float:
    """
    Score candidate based on metrics and config weights.
    metrics must contain:
      - avg_daily_return
      - max_dd
      - pf
      - wr
      - monthly_returns (optional list[float] for stability penalty)
    """
    target = cfg["target"]
    weights = cfg["score_weights"]

    adr = metrics.get("avg_daily_return", 0.0)
    max_dd = metrics.get("max_dd", 1.0)
    pf = metrics.get("pf", 0.0)
    wr = metrics.get("wr", 0.0)
    monthly_returns = metrics.get("monthly_returns") or []

    # Hard filters
    if metrics.get("num_trades", 0) < target["min_trades"]:
        return -1e9
    if pf < target["min_pf"]:
        return -1e9
    if wr < target["min_wr"]:
        return -1e9
    if max_dd > target["max_drawdown"]:
        return -1e9

    # Base score on average daily return
    score = weights["w_daily_return"] * adr

    # Penalty for exceeding preferred band or being below target
    if adr < target["min_daily_return"]:
        score -= weights["w_daily_return"] * (target["min_daily_return"] - adr) * 2.0

    # Drawdown penalty
    if max_dd > target["max_drawdown"]:
        score -= weights["w_dd_penalty"] * (max_dd - target["max_drawdown"]) * 10.0

    # Profit factor contribution
    score += weights["w_pf"] * (pf - 1.0)

    # Win rate contribution
    score += weights["w_wr"] * (wr - target["min_wr"])

    # Stability penalty (std of monthly returns)
    if monthly_returns:
        vol = pstdev(monthly_returns)
        score -= weights["w_stability_penalty"] * vol

    return score


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


async def load_backtest_data(days: int = 90) -> Dict[str, pd.DataFrame]:
    """Load historical data for backtesting."""
    cache_dir = PROJECT_ROOT / "data" / "backtest_cache"
    symbols = [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "APT/USDT:USDT",
        "OP/USDT:USDT",
        "NEAR/USDT:USDT",
    ]
    
    fetcher = DataFetcher(cache_dir)
    data = await fetcher.fetch_all(symbols, days=days, timeframe="5m")
    return data


def run_quick_backtest_from_dsl(tuned_dsl: dict, symbols: list, data: Dict[str, pd.DataFrame]) -> dict:
    """
    Adapter: converts DSL to config format and runs Stage 1 style backtest.
    """
    # Extract params from DSL
    base_params = tuned_dsl.get("base_params", {})
    exit_params = tuned_dsl.get("exit", {})
    
    # Build config dict compatible with Stage 1 backtester
    config = {
        "id": 0,  # dummy
        "leverage": base_params.get("leverage", 2.0),
        "sl_pct": exit_params.get("sl_pct", base_params.get("sl_pct", 1.5)),
        "tp_pct": exit_params.get("tp_pct", base_params.get("tp_pct", 2.5)),
        "rsi_period": base_params.get("rsi_period", 14),
        "rsi_oversold": base_params.get("rsi_oversold", 30),
        "rsi_overbought": base_params.get("rsi_overbought", 70),
        "volume_surge_min": base_params.get("volume_surge_min", 1.2),
    }
    
    # Run Stage 1 style backtest (inline simplified version)
    equity = 10000.0
    start_equity = equity
    positions = {}
    trades = []
    
    # Get costs
    commission_per_side = 0.0004  # 0.04% for Stage 2
    slippage_per_side = 0.0003    # 0.03%
    total_cost_per_side = commission_per_side + slippage_per_side
    
    # Filter data to specified symbols
    filtered_data = {s: df for s, df in data.items() if any(sym in s for sym in symbols)}
    if not filtered_data:
        filtered_data = data  # fallback to all
    
    # Get all bars
    all_bars = []
    for symbol, df in filtered_data.items():
        df = df.copy()
        
        # Add indicators
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(config['rsi_period']).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['vol_surge'] = df['volume'] / df['vol_ma']
        
        for timestamp, row in df.iterrows():
            all_bars.append((timestamp, symbol, row))
    
    all_bars.sort(key=lambda x: x[0])
    
    # Process bars
    for timestamp, symbol, row in all_bars:
        close = row['close']
        high = row['high']
        low = row['low']
        rsi = row.get('rsi', 50)
        vol_surge = row.get('vol_surge', 1.0)
        
        # Check exits
        if symbol in positions:
            pos = positions[symbol]
            exit_price = None
            
            if pos['side'] == 'LONG':
                if low <= pos['sl']:
                    exit_price = pos['sl']
                elif high >= pos['tp']:
                    exit_price = pos['tp']
            else:  # SHORT
                if high >= pos['sl']:
                    exit_price = pos['sl']
                elif low <= pos['tp']:
                    exit_price = pos['tp']
            
            if exit_price:
                # Calculate PnL
                if pos['side'] == 'LONG':
                    price_change = (exit_price - pos['entry']) / pos['entry']
                else:
                    price_change = (pos['entry'] - exit_price) / pos['entry']
                
                pnl_gross = pos['size'] * price_change * config['leverage']
                
                # Costs
                notional = pos['size'] * config['leverage']
                entry_cost = notional * total_cost_per_side
                exit_cost = notional * total_cost_per_side
                pnl_net = pnl_gross - entry_cost - exit_cost
                
                equity += pnl_net
                trades.append({'pnl': pnl_net, 'timestamp': timestamp})
                del positions[symbol]
        
        # Check entries
        if symbol not in positions and len(positions) < 3:
            signal = None
            
            if rsi < config['rsi_oversold'] and vol_surge >= config['volume_surge_min']:
                signal = 'LONG'
            elif rsi > config['rsi_overbought'] and vol_surge >= config['volume_surge_min']:
                signal = 'SHORT'
            
            if signal:
                size = equity * 0.33
                
                if signal == 'LONG':
                    sl = close * (1 - config['sl_pct'] / 100)
                    tp = close * (1 + config['tp_pct'] / 100)
                else:
                    sl = close * (1 + config['sl_pct'] / 100)
                    tp = close * (1 - config['tp_pct'] / 100)
                
                positions[symbol] = {
                    'side': signal,
                    'entry': close,
                    'size': size,
                    'sl': sl,
                    'tp': tp,
                }
    
    # Calculate metrics
    total_return = equity - start_equity
    total_return_pct = total_return / start_equity * 100
    
    if not trades:
        return {
            "avg_daily_return": 0,
            "max_dd": 0,
            "pf": 0,
            "wr": 0,
            "num_trades": 0,
            "total_return_pct": total_return_pct,
            "monthly_returns": [],
        }
    
    wins = sum(1 for t in trades if t['pnl'] > 0)
    winrate = wins / len(trades)
    
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Daily return calculation
    days = 90  # Stage 2 uses 90 days
    avg_daily_return = total_return_pct / days / 100  # as decimal
    
    # Max DD
    equity_curve = [start_equity]
    running_equity = start_equity
    for t in trades:
        running_equity += t['pnl']
        equity_curve.append(running_equity)
    
    peak = start_equity
    max_dd = 0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (eq - peak) / peak
        max_dd = min(max_dd, dd)
    
    return {
        "avg_daily_return": avg_daily_return,
        "max_dd": abs(max_dd),
        "pf": pf if pf != float('inf') else 999,
        "wr": winrate,
        "num_trades": len(trades),
        "total_return_pct": total_return_pct,
        "monthly_returns": [],  # TODO: calculate if needed
    }


# Global data cache
_BACKTEST_DATA_CACHE = None


def run_backtest_for_candidate(strategy: dict, tuned_dsl: dict, cfg: dict) -> dict:
    """
    Adapter between Stage 2 Optimization and Stage 1 backtester.
    """
    global _BACKTEST_DATA_CACHE
    
    # Load data once globally
    if _BACKTEST_DATA_CACHE is None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _BACKTEST_DATA_CACHE = loop.run_until_complete(load_backtest_data(days=90))
        loop.close()
    
    symbols = strategy.get("symbols") or []
    
    raw = run_quick_backtest_from_dsl(tuned_dsl, symbols, _BACKTEST_DATA_CACHE)
    
    return raw


def optimize_single_strategy(strategy: dict, cfg: dict, trials_log_dir: Path) -> dict | None:
    """
    Run parameter search for one strategy from Stage 1 pool.
    Returns best candidate dict with fields:
      - strategy_name
      - pattern_type
      - symbols
      - dsl
      - backtest_metrics
      - score
    or None if nothing passes filters.
    """
    param_space = get_param_space_for_strategy(strategy, cfg)
    trials_per_strategy = cfg["optimization"]["trials_per_strategy"]

    if not param_space:
        # No tunable params defined -> could still evaluate base DSL once
        trials_per_strategy = 1

    base_dsl = strategy.get("dsl") or strategy.get("base_dsl") or {}
    if not base_dsl:
        return None

    symbol_universe = strategy.get("symbols") or strategy.get("symbol_universe") or []

    strategy_id = strategy.get("strategy_id") or strategy.get("strategy_name") or "unknown"
    pattern_type = strategy.get("pattern_type", "unknown")

    log_path = trials_log_dir / f"{strategy_id}_trials.jsonl"
    best_candidate = None

    with open(log_path, "w") as log_f:
        for trial in range(1, trials_per_strategy + 1):
            if param_space:
                sampled = sample_params(param_space)
            else:
                sampled = {}

            tuned_dsl = merge_params_into_dsl(base_dsl, sampled)

            try:
                metrics = run_backtest_for_candidate(strategy, tuned_dsl, cfg)
            except Exception as e:  # noqa: BLE001
                rec = {
                    "ts": datetime.utcnow().isoformat(),
                    "strategy_id": strategy_id,
                    "trial": trial,
                    "error": str(e),
                }
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                continue

            score = compute_score(metrics, cfg)
            rec = {
                "ts": datetime.utcnow().isoformat(),
                "strategy_id": strategy_id,
                "trial": trial,
                "params": sampled,
                "metrics": metrics,
                "score": score,
            }
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

            if score <= -1e8:
                continue

            if best_candidate is None or score > best_candidate["score"]:
                best_candidate = {
                    "strategy_name": strategy_id,
                    "pattern_type": pattern_type,
                    "symbols": symbol_universe,
                    "dsl": tuned_dsl,
                    "backtest_metrics": metrics,
                    "score": score,
                }

    return best_candidate


def build_best_live_payload(best_candidates: list[dict], cfg: dict) -> list[dict]:
    """
    Transform list of best candidates into payload that matches best_live.schema.json.
    Capital shares are normalized across selected strategies.
    """
    if not best_candidates:
        return []

    max_live = cfg["optimization"]["max_strategies_live"]
    best_sorted = sorted(best_candidates, key=lambda c: c["score"], reverse=True)[:max_live]

    # Equal shares for v1
    n = len(best_sorted)
    equal_share = 1.0 / n if n > 0 else 0.0

    risk_defaults = cfg.get("risk_profile_defaults", {})

    payload = []
    for cand in best_sorted:
        entry = {
            "strategy_name": cand["strategy_name"],
            "pattern_type": cand.get("pattern_type"),
            "symbols": cand.get("symbols") or [],
            "logic_human": "",  # optional: can be filled later by separate script
            "dsl": cand["dsl"],
            "backtest_metrics": cand["backtest_metrics"],
            "recommended_capital_share": equal_share,
            "risk_profile": {
                "max_position_pct_of_trading_capital": risk_defaults.get(
                    "max_position_pct_of_trading_capital", 0.1
                ),
                "max_daily_loss_pct": risk_defaults.get("max_daily_loss_pct", 0.06),
            },
        }
        payload.append(entry)

    return payload


def main():
    cfg = load_config()
    random.seed(cfg["optimization"].get("random_seed", 42))

    pool_dir = PROJECT_ROOT / cfg["input"]["strategies_pool_dir"]
    pool_pattern = cfg["input"]["strategies_pool_pattern"]

    latest_pool = find_latest_pool_file(pool_dir, pool_pattern)
    strategies = load_strategies_pool(latest_pool)

    ensure_dir(RESULTS_DIR)
    trials_log_dir = PROJECT_ROOT / cfg["output"]["trials_log_dir"]
    ensure_dir(trials_log_dir)

    best_candidates = []

    for s in strategies:
        cand = optimize_single_strategy(s, cfg, trials_log_dir=trials_log_dir)
        if cand is not None:
            best_candidates.append(cand)

    best_live = build_best_live_payload(best_candidates, cfg)

    out_path = PROJECT_ROOT / cfg["output"]["best_live_path"]
    ensure_dir(out_path.parent)
    with open(out_path, "w") as f:
        json.dump(best_live, f, indent=2)

    print(f"[Optimization] Done. Selected {len(best_live)} strategies → {out_path}")


if __name__ == "__main__":
    main()
