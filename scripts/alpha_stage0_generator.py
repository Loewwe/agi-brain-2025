#!/usr/bin/env python3
"""
Alpha Search v2.0 - Stage 0: Config Generator + Static Filter

Generates 20,000 alpha configs and applies NO-GO rules.
Output: 8-10k configs ready for Stage 1 backtesting.
"""

import json
import random
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(42)


@dataclass
class AlphaConfig:
    """Alpha strategy configuration."""
    id: int
    
    # Core params
    timeframe: str  # 5m, 15m, 1h
    leverage: float
    max_positions: int
    
    # Indicators
    rsi_period: int
    rsi_oversold: int
    rsi_overbought: int
    ema_fast: int
    ema_slow: int
    bb_period: int
    bb_std: float
    atr_period: int
    
    # Entry
    volume_surge_min: float
    ema_dead_zone_pct: float
    enhanced_breakout: bool
    
    # Exit
    sl_pct: float
    tp_pct: float
    trailing_enabled: bool
    max_hold_hours: int
    
    # Strategy type
    strategy_mode: str  # momentum, mean_reversion, hybrid


# Parameter ranges
RANGES = {
    "timeframe": ["5m", "15m", "1h"],
    "leverage": (1.0, 3.0),
    "max_positions": (1, 5),
    "rsi_period": (7, 21),
    "rsi_oversold": (20, 40),
    "rsi_overbought": (60, 80),
    "ema_fast": (8, 25),
    "ema_slow": (40, 120),
    "bb_period": (15, 30),
    "bb_std": (1.5, 2.5),
    "atr_period": (10, 20),
    "volume_surge_min": (1.0, 2.0),
    "ema_dead_zone_pct": (0.002, 0.01),
    "sl_pct": (0.5, 4.0),
    "tp_pct": (0.7, 8.0),
    "max_hold_hours": (1, 24),
    "strategy_mode": ["momentum", "mean_reversion", "hybrid"],
}


def rand_float(low: float, high: float, decimals: int = 3) -> float:
    return round(random.uniform(low, high), decimals)


def rand_int(low: int, high: int) -> int:
    return random.randint(low, high)


def rand_choice(options: list):
    return random.choice(options)


def generate_config(idx: int) -> AlphaConfig:
    """Generate single random config."""
    return AlphaConfig(
        id=idx,
        timeframe=rand_choice(RANGES["timeframe"]),
        leverage=rand_float(*RANGES["leverage"]),
        max_positions=rand_int(*RANGES["max_positions"]),
        rsi_period=rand_int(*RANGES["rsi_period"]),
        rsi_oversold=rand_int(*RANGES["rsi_oversold"]),
        rsi_overbought=rand_int(*RANGES["rsi_overbought"]),
        ema_fast=rand_int(*RANGES["ema_fast"]),
        ema_slow=rand_int(*RANGES["ema_slow"]),
        bb_period=rand_int(*RANGES["bb_period"]),
        bb_std=rand_float(*RANGES["bb_std"]),
        atr_period=rand_int(*RANGES["atr_period"]),
        volume_surge_min=rand_float(*RANGES["volume_surge_min"]),
        ema_dead_zone_pct=rand_float(*RANGES["ema_dead_zone_pct"]),
        enhanced_breakout=random.random() > 0.5,
        sl_pct=rand_float(*RANGES["sl_pct"]),
        tp_pct=rand_float(*RANGES["tp_pct"]),
        trailing_enabled=random.random() > 0.5,
        max_hold_hours=rand_int(*RANGES["max_hold_hours"]),
        strategy_mode=rand_choice(RANGES["strategy_mode"]),
    )


def apply_static_filter(config: AlphaConfig) -> tuple[bool, str]:
    """
    Apply NO-GO rules.
    Returns: (passed, reason)
    """
    # Rule 1: Leverage > 3
    if config.leverage > 3.0:
        return False, "leverage_too_high"
    
    # Rule 2: Stop loss > 6% AND no reasonable hold limit
    if config.sl_pct > 6.0 and config.max_hold_hours > 48:
        return False, "unbounded_risk"
    
    # Rule 3: Scalping death (TP < 0.4% AND HFT timeframe)
    if config.tp_pct < 0.4 and config.timeframe == "1m":
        return False, "scalping_death"
    
    # Rule 4: Max positions > 10
    if config.max_positions > 10:
        return False, "too_many_positions"
    
    # Rule 5: No SL at all would be config.sl_pct is None, but we always have it
    
    # Rule 6: Insane hold time without exit
    if config.max_hold_hours > 72 and config.sl_pct > 5.0:
        return False, "no_exit_logic"
    
    # Rule 7: TP < SL (bad RR)
    if config.tp_pct < config.sl_pct * 0.8:
        return False, "bad_risk_reward"
    
    return True, "passed"


def generate_and_filter(n_total: int = 20000) -> tuple[List[AlphaConfig], List[AlphaConfig]]:
    """Generate N configs and filter."""
    logger.info(f"Generating {n_total} configs...")
    
    all_configs = []
    passed_configs = []
    rejection_counts = {}
    
    for i in range(n_total):
        if (i + 1) % 5000 == 0:
            logger.info(f"  Generated {i+1}/{n_total}...")
        
        config = generate_config(i)
        all_configs.append(config)
        
        passed, reason = apply_static_filter(config)
        if passed:
            passed_configs.append(config)
        else:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    
    logger.info(f"✅ Generated {len(all_configs)} configs")
    logger.info(f"✅ Passed static filter: {len(passed_configs)} ({len(passed_configs)/len(all_configs)*100:.1f}%)")
    logger.info(f"Rejection breakdown:")
    for reason, count in sorted(rejection_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {reason}: {count}")
    
    return all_configs, passed_configs


def save_configs(configs: List[AlphaConfig], output_path: Path):
    """Save configs to CSV."""
    df = pd.DataFrame([asdict(c) for c in configs])
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(configs)} configs to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Alpha Search v2.0 - Stage 0")
    parser.add_argument("--n", type=int, default=20000, help="Number of configs to generate")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    # Generate and filter
    all_configs, passed_configs = generate_and_filter(args.n)
    
    # Output directory
    root = Path(__file__).parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else root / "reports" / "alpha_search_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all
    save_configs(all_configs, output_dir / "stage0_all_configs.csv")
    
    # Save passed
    save_configs(passed_configs, output_dir / "stage0_passed.csv")
    
    # Summary
    print("\n" + "="*60)
    print("ALPHA SEARCH v2.0 - STAGE 0 COMPLETE")
    print("="*60)
    print(f"Total generated: {len(all_configs)}")
    print(f"Passed filter:   {len(passed_configs)} ({len(passed_configs)/len(all_configs)*100:.1f}%)")
    print(f"Ready for Stage 1: {len(passed_configs)} configs")
    print(f"\nOutput: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
