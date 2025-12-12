#!/usr/bin/env python3
"""
Alpha Search v2.0 - Candidate Generator

Generates 1000+ alpha candidates from 3 families:
1. Momentum / Breakout (864-like)
2. Mean-Reversion 
3. Regime-Aware Hybrid

Each candidate is a JSON config that can be evaluated by alpha_evaluator.py
"""

import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from enum import Enum
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlphaFamily(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    REGIME_AWARE = "regime_aware"


@dataclass
class AlphaCandidate:
    """Alpha candidate configuration."""
    id: str
    family: str
    
    # Timeframe
    tf: str  # 1m, 5m, 15m
    
    # Trend indicators
    ema_fast: int
    ema_slow: int
    
    # Oscillators
    rsi_period: int
    rsi_oversold: int
    rsi_overbought: int
    
    # Volatility
    bb_period: int
    bb_std: float
    atr_period: int
    
    # Entry conditions
    vol_ratio_min: float
    ema_dead_zone_pct: float
    
    # Exit conditions
    atr_mult_sl: float
    atr_mult_tp: float
    max_hold_bars: int
    trailing_enabled: bool
    
    # Filters
    adx_threshold: float  # For regime filter
    enhanced_breakout: bool
    
    # Family-specific
    mean_rev_bb_touch: bool  # Mean-rev: enter on BB touch
    regime_mode: str  # trend_only, flat_only, adaptive


# =============================================================================
# PARAMETER RANGES
# =============================================================================

PARAM_RANGES = {
    "tf": ["1m", "5m", "15m"],
    "ema_fast": (8, 25),
    "ema_slow": (40, 120),
    "rsi_period": (7, 21),
    "rsi_oversold": (20, 40),
    "rsi_overbought": (60, 80),
    "bb_period": (15, 30),
    "bb_std": (1.5, 2.5),
    "atr_period": (10, 20),
    "vol_ratio_min": (1.0, 2.0),
    "ema_dead_zone_pct": (0.002, 0.01),
    "atr_mult_sl": (0.8, 2.0),
    "atr_mult_tp": (1.5, 4.0),
    "max_hold_bars": (12, 96),  # 1h to 8h for 5m bars
    "adx_threshold": (15, 30),
}


# =============================================================================
# GENERATORS
# =============================================================================

def rand_int(low: int, high: int) -> int:
    return random.randint(low, high)


def rand_float(low: float, high: float, decimals: int = 3) -> float:
    return round(random.uniform(low, high), decimals)


def rand_choice(options: list):
    return random.choice(options)


def rand_bool(prob: float = 0.5) -> bool:
    return random.random() < prob


def generate_base_params() -> dict:
    """Generate base parameters common to all families."""
    return {
        "tf": rand_choice(PARAM_RANGES["tf"]),
        "ema_fast": rand_int(*PARAM_RANGES["ema_fast"]),
        "ema_slow": rand_int(*PARAM_RANGES["ema_slow"]),
        "rsi_period": rand_int(*PARAM_RANGES["rsi_period"]),
        "rsi_oversold": rand_int(*PARAM_RANGES["rsi_oversold"]),
        "rsi_overbought": rand_int(*PARAM_RANGES["rsi_overbought"]),
        "bb_period": rand_int(*PARAM_RANGES["bb_period"]),
        "bb_std": rand_float(*PARAM_RANGES["bb_std"]),
        "atr_period": rand_int(*PARAM_RANGES["atr_period"]),
        "vol_ratio_min": rand_float(*PARAM_RANGES["vol_ratio_min"]),
        "ema_dead_zone_pct": rand_float(*PARAM_RANGES["ema_dead_zone_pct"]),
        "atr_mult_sl": rand_float(*PARAM_RANGES["atr_mult_sl"]),
        "atr_mult_tp": rand_float(*PARAM_RANGES["atr_mult_tp"]),
        "max_hold_bars": rand_int(*PARAM_RANGES["max_hold_bars"]),
        "adx_threshold": rand_float(*PARAM_RANGES["adx_threshold"]),
    }


def generate_momentum_alpha(idx: int) -> AlphaCandidate:
    """Generate momentum/breakout alpha (864-like)."""
    params = generate_base_params()
    
    # Momentum-specific: prefer wider dead zone, higher vol filter
    params["vol_ratio_min"] = rand_float(1.2, 2.0)
    params["enhanced_breakout"] = rand_bool(0.7)
    
    return AlphaCandidate(
        id=f"mom_{idx:04d}",
        family=AlphaFamily.MOMENTUM.value,
        trailing_enabled=rand_bool(0.5),
        mean_rev_bb_touch=False,
        regime_mode="adaptive",
        **params,
    )


def generate_mean_reversion_alpha(idx: int) -> AlphaCandidate:
    """Generate mean-reversion alpha."""
    params = generate_base_params()
    
    # Mean-rev specific: tighter extremes, faster exit
    params["rsi_oversold"] = rand_int(15, 30)
    params["rsi_overbought"] = rand_int(70, 85)
    params["atr_mult_sl"] = rand_float(1.5, 3.0)  # Wider SL
    params["atr_mult_tp"] = rand_float(0.8, 1.5)  # Tighter TP
    params["enhanced_breakout"] = False
    
    return AlphaCandidate(
        id=f"mr_{idx:04d}",
        family=AlphaFamily.MEAN_REVERSION.value,
        trailing_enabled=rand_bool(0.3),
        mean_rev_bb_touch=rand_bool(0.7),
        regime_mode="flat_preferred",
        **params,
    )


def generate_regime_aware_alpha(idx: int) -> AlphaCandidate:
    """Generate regime-aware hybrid alpha."""
    params = generate_base_params()
    
    # Regime-aware: ADX-based filtering
    params["adx_threshold"] = rand_float(18, 28)
    params["enhanced_breakout"] = rand_bool(0.5)
    
    return AlphaCandidate(
        id=f"reg_{idx:04d}",
        family=AlphaFamily.REGIME_AWARE.value,
        trailing_enabled=rand_bool(0.5),
        mean_rev_bb_touch=rand_bool(0.3),
        regime_mode=rand_choice(["trend_only", "adaptive", "counter_trend"]),
        **params,
    )


def generate_candidates(
    n_total: int = 1000,
    momentum_pct: float = 0.4,
    mean_rev_pct: float = 0.35,
    regime_pct: float = 0.25,
) -> List[AlphaCandidate]:
    """Generate N alpha candidates."""
    candidates = []
    
    n_momentum = int(n_total * momentum_pct)
    n_mean_rev = int(n_total * mean_rev_pct)
    n_regime = n_total - n_momentum - n_mean_rev
    
    logger.info(f"Generating {n_total} candidates:")
    logger.info(f"  Momentum: {n_momentum}")
    logger.info(f"  Mean-Rev: {n_mean_rev}")
    logger.info(f"  Regime:   {n_regime}")
    
    for i in range(n_momentum):
        candidates.append(generate_momentum_alpha(i))
    
    for i in range(n_mean_rev):
        candidates.append(generate_mean_reversion_alpha(i))
    
    for i in range(n_regime):
        candidates.append(generate_regime_aware_alpha(i))
    
    random.shuffle(candidates)
    return candidates


def save_candidates(candidates: List[AlphaCandidate], output_path: Path):
    """Save candidates to JSON file."""
    data = [asdict(c) for c in candidates]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(candidates)} candidates to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate alpha candidates")
    parser.add_argument("--n", type=int, default=1000, help="Number of candidates")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    args = parser.parse_args()
    
    # Generate
    candidates = generate_candidates(n_total=args.n)
    
    # Save
    root = Path(__file__).parent.parent
    output_dir = root / "reports" / "alpha_search_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = args.output or output_dir / "candidates_raw.json"
    save_candidates(candidates, Path(output_path))
    
    # Stats
    families = {}
    for c in candidates:
        families[c.family] = families.get(c.family, 0) + 1
    
    print("\n" + "="*50)
    print("ALPHA GENERATOR v2.0")
    print("="*50)
    print(f"Total candidates: {len(candidates)}")
    for fam, count in families.items():
        print(f"  {fam}: {count}")
    print(f"Output: {output_path}")
    print("="*50)


if __name__ == "__main__":
    main()
