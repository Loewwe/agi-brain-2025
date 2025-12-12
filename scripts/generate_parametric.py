#!/usr/bin/env python3
"""
Parametric Hypothesis Generator - Stream 3
Создает 50 параметрических вариаций базовых шаблонов

5 templates × 10 variations = 50 hypotheses
"""

import os
from pathlib import Path
import itertools

PARAM_TEMPLATE = '''#!/usr/bin/env python3
"""
Parametric Hypothesis: {name}
Generated from template: {template_name}
Parameters: {params}
"""

from datetime import datetime
import pandas as pd
import numpy as np

class H{id}_{name}(Hypothesis):
    """
    {description}
    
    Parameters:
    -----------
{param_desc}
    """
    def __init__(self):
        super().__init__(
            hyp_id="H{id}",
            name="{name}",
            description="{description}",
            win_threshold=0.002
        )
        
        # Core parameters
        self.param1 = {param1}
        self.param2 = {param2}
        self.param3 = {param3}
    
    def find_triggers(self, data):
        """Find triggers using {template_name} template"""
        events = []
        
        for symbol, df in data.items():
            if len(df) < 100:
                continue
            
            df = df.copy()
            
            # Template-specific pattern detection
{pattern_logic}
        
        return events
'''

# 5 базовых шаблонов
TEMPLATES = {
    'breakout': {
        'description': 'Price breakout with volume confirmation',
        'params': ['lookback_period', 'volume_threshold', 'min_gap_pct'],
        'variations': [
            {'lookback_period': 20, 'volume_threshold': 1.5, 'min_gap_pct': 0.01},
            {'lookback_period': 50, 'volume_threshold': 2.0, 'min_gap_pct': 0.015},
            {'lookback_period': 10, 'volume_threshold': 1.2, 'min_gap_pct': 0.005},
            {'lookback_period': 30, 'volume_threshold': 1.8, 'min_gap_pct': 0.012},
            {'lookback_period': 40, 'volume_threshold': 2.5, 'min_gap_pct': 0.02},
            {'lookback_period': 15, 'volume_threshold': 1.3, 'min_gap_pct': 0.008},
            {'lookback_period': 25, 'volume_threshold': 2.2, 'min_gap_pct': 0.018},
            {'lookback_period': 35, 'volume_threshold': 1.6, 'min_gap_pct': 0.01},
            {'lookback_period': 45, 'volume_threshold': 2.8, 'min_gap_pct': 0.025},
            {'lookback_period': 12, 'volume_threshold': 1.4, 'min_gap_pct': 0.006},
        ],
        'logic': '''
            # Breakout detection
            for i in range(self.param1, len(df)):
                high_{period} = df.iloc[i-self.param1:i]['high'].max()
                vol_avg = df.iloc[i-self.param1:i]['volume'].mean()
                
                current_high = df.iloc[i]['high']
                current_vol = df.iloc[i]['volume']
                
                if (current_high > high_{period} * (1 + self.param3) and 
                    current_vol > vol_avg * self.param2):
                    
                    events.append(HypothesisEvent(
                        timestamp=df.index[i],
                        symbol=symbol,
                        entry_price=df.iloc[i]['close'],
                        direction='long',
                        rew_pct=0, mfe_pct=0, mae_pct=0,
                        context={{'pattern': 'breakout', 'volume_mult': current_vol/vol_avg}}
                    ))
'''
    },
    
    'reversal': {
        'description': 'Mean reversion with RSI extremes',
        'params': ['rsi_period', 'oversold_level', 'volume_confirm'],
        'variations': [
            {'rsi_period': 14, 'oversold_level': 30, 'volume_confirm': 1.2},
            {'rsi_period': 7, 'oversold_level': 25, 'volume_confirm': 1.5},
            {'rsi_period': 21, 'oversold_level': 35, 'volume_confirm': 1.1},
            {'rsi_period': 10, 'oversold_level': 20, 'volume_confirm': 1.8},
            {'rsi_period': 28, 'oversold_level': 40, 'volume_confirm': 1.3},
            {'rsi_period': 5, 'oversold_level': 15, 'volume_confirm': 2.0},
            {'rsi_period': 14, 'oversold_level': 28, 'volume_confirm': 1.4},
            {'rsi_period': 14, 'oversold_level': 32, 'volume_confirm': 1.6},
            {'rsi_period': 9, 'oversold_level': 22, 'volume_confirm': 1.7},
            {'rsi_period': 18, 'oversold_level': 38, 'volume_confirm': 1.25},
        ],
        'logic': '''
            # RSI Reversal detection
            df['rsi'] = calculate_rsi(df['close'], self.param1)
            vol_sma = df['volume'].rolling(20).mean()
            
            for i in range(max(self.param1, 20), len(df)):
                rsi = df.iloc[i]['rsi']
                vol_ratio = df.iloc[i]['volume'] / vol_sma.iloc[i]
                
                if rsi < self.param2 and vol_ratio > self.param3:
                    events.append(HypothesisEvent(
                        timestamp=df.index[i],
                        symbol=symbol,
                        entry_price=df.iloc[i]['close'],
                        direction='long',
                        rew_pct=0, mfe_pct=0, mae_pct=0,
                        context={{'pattern': 'rsi_reversal', 'rsi': rsi}}
                    ))
'''
    },
    
    'momentum': {
        'description': 'Trend following with momentum confirmation',
        'params': ['fast_ma', 'slow_ma', 'min_trend_strength'],
        'variations': [
            {'fast_ma': 10, 'slow_ma': 30, 'min_trend_strength': 0.02},
            {'fast_ma': 5, 'slow_ma': 20, 'min_trend_strength': 0.01},
            {'fast_ma': 15, 'slow_ma': 50, 'min_trend_strength': 0.03},
            {'fast_ma': 8, 'slow_ma': 25, 'min_trend_strength': 0.015},
            {'fast_ma': 12, 'slow_ma': 40, 'min_trend_strength': 0.025},
            {'fast_ma': 6, 'slow_ma': 18, 'min_trend_strength': 0.008},
            {'fast_ma': 20, 'slow_ma': 60, 'min_trend_strength': 0.035},
            {'fast_ma': 9, 'slow_ma': 27, 'min_trend_strength': 0.012},
            {'fast_ma': 14, 'slow_ma': 45, 'min_trend_strength': 0.028},
            {'fast_ma': 7, 'slow_ma': 22, 'min_trend_strength': 0.009},
        ],
        'logic': '''
            # Trend momentum
            df['fast_ma'] = df['close'].rolling(self.param1).mean()
            df['slow_ma'] = df['close'].rolling(self.param2).mean()
            
            for i in range(max(self.param1, self.param2), len(df)):
                trend_strength = (df.iloc[i]['fast_ma'] - df.iloc[i]['slow_ma']) / df.iloc[i]['slow_ma']
                
                if trend_strength > self.param3:
                    events.append(HypothesisEvent(
                        timestamp=df.index[i],
                        symbol=symbol,
                        entry_price=df.iloc[i]['close'],
                        direction='long',
                        rew_pct=0, mfe_pct=0, mae_pct=0,
                        context={{'pattern': 'momentum', 'trend_strength': trend_strength}}
                    ))
'''
    },
    
    'volatility': {
        'description': 'Volatility contraction then expansion',
        'params': ['atr_period', 'bb_period', 'squeeze_threshold'],
        'variations': [
            {'atr_period': 14, 'bb_period': 20, 'squeeze_threshold': 0.015},
            {'atr_period': 10, 'bb_period': 15, 'squeeze_threshold': 0.01},
            {'atr_period': 20, 'bb_period': 30, 'squeeze_threshold': 0.02},
            {'atr_period': 7, 'bb_period': 12, 'squeeze_threshold': 0.008},
            {'atr_period': 21, 'bb_period': 25, 'squeeze_threshold': 0.018},
            {'atr_period': 14, 'bb_period': 18, 'squeeze_threshold': 0.012},
            {'atr_period': 14, 'bb_period': 22, 'squeeze_threshold': 0.016},
            {'atr_period': 12, 'bb_period': 20, 'squeeze_threshold': 0.014},
            {'atr_period': 18, 'bb_period': 27, 'squeeze_threshold': 0.019},
            {'atr_period': 9, 'bb_period': 14, 'squeeze_threshold': 0.009},
        ],
        'logic': '''
            # Volatility expansion
            df['atr'] = calculate_atr(df, self.param1)
            atr_ma = df['atr'].rolling(self.param2).mean()
            
            for i in range(max(self.param1, self.param2), len(df)):
                atr_ratio = df.iloc[i]['atr'] / atr_ma.iloc[i] if atr_ma.iloc[i] > 0 else 0
                
                if atr_ratio > (1 + self.param3):
                    events.append(HypothesisEvent(
                        timestamp=df.index[i],
                        symbol=symbol,
                        entry_price=df.iloc[i]['close'],
                        direction='long',
                        rew_pct=0, mfe_pct=0, mae_pct=0,
                        context={{'pattern': 'vol_expansion', 'atr_ratio': atr_ratio}}
                    ))
'''
    },
    
    'time_pattern': {
        'description': 'Time-of-day seasonal pattern',
        'params': ['target_hour', 'day_filter', 'min_move_pct'],
        'variations': [
            {'target_hour': 9, 'day_filter': [0,1,2,3,4], 'min_move_pct': 0.005},
            {'target_hour': 14, 'day_filter': [0,1,2,3,4], 'min_move_pct': 0.008},
            {'target_hour': 21, 'day_filter': [0,1,2,3,4], 'min_move_pct': 0.006},
            {'target_hour': 0, 'day_filter': [5,6], 'min_move_pct': 0.01},
            {'target_hour': 16, 'day_filter': [4], 'min_move_pct': 0.012},
            {'target_hour': 10, 'day_filter': [0], 'min_move_pct': 0.015},
            {'target_hour': 13, 'day_filter': list(range(7)), 'min_move_pct': 0.007},
            {'target_hour': 18, 'day_filter': [0,1,2,3,4], 'min_move_pct': 0.009},
            {'target_hour': 6, 'day_filter': list(range(7)), 'min_move_pct': 0.004},
            {'target_hour': 22, 'day_filter': [0,1,2,3,4], 'min_move_pct': 0.011},
        ],
        'logic': '''
            # Time pattern
            for i in range(24, len(df)):
                hour = df.index[i].hour
                day = df.index[i].weekday()
                
                if hour == self.param1 and day in self.param2:
                    price_move = abs(df.iloc[i]['close'] - df.iloc[i-24]['close']) / df.iloc[i-24]['close']
                    
                    if price_move > self.param3:
                        events.append(HypothesisEvent(
                            timestamp=df.index[i],
                            symbol=symbol,
                            entry_price=df.iloc[i]['close'],
                            direction='long' if df.iloc[i]['close'] > df.iloc[i-24]['close'] else 'short',
                            rew_pct=0, mfe_pct=0, mae_pct=0,
                            context={{'pattern': 'time_pattern', 'hour': hour}}
                        ))
'''
    }
}

def generate_parametric_hypothesis(template_name, variation_idx, variation_params, start_id=200):
    """Generate one parametric hypothesis"""
    template = TEMPLATES[template_name]
    hyp_id = start_id + len(TEMPLATES) * variation_idx + list(TEMPLATES.keys()).index(template_name)
    
    name = f"{template_name.title()}_{variation_idx+1}"
    
    param_desc = "\n".join([f"    - {k}: {v}" for k, v in variation_params.items()])
    
    code = PARAM_TEMPLATE.format(
        id=hyp_id,
        name=name,
        template_name=template_name,
        description=template['description'],
        params=str(variation_params),
        param_desc=param_desc,
        param1=variation_params[template['params'][0]],
        param2=variation_params[template['params'][1]],
        param3=variation_params[template['params'][2]],
        pattern_logic=template['logic']
    )
    
    return code, hyp_id, name

def main():
    """Generate 50 parametric hypotheses"""
    output_dir = Path('lab/hypotheses/parametric')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎛️  GENERATING 50 PARAMETRIC HYPOTHESES")
    print(f"   Templates: {len(TEMPLATES)}")
    print(f"   Variations per template: 10")
    print(f"   Output: {output_dir}")
    print()
    
    generated = 0
    all_hypotheses = []
    
    for template_name, template_spec in TEMPLATES.items():
        print(f"  Template: {template_name}")
        
        for idx, variation in enumerate(template_spec['variations']):
            try:
                code, hyp_id, name = generate_parametric_hypothesis(
                    template_name, idx, variation
                )
                
                filename = output_dir / f"H{hyp_id}_{name}.py"
                
                with open(filename, 'w') as f:
                    f.write(code)
                
                generated += 1
                all_hypotheses.append((hyp_id, name, template_name))
                print(f"    ✅ H{hyp_id}_{name}")
                
            except Exception as e:
                print(f"    ❌ Variation {idx}: {e}")
        
        print()
    
    print(f"✅ Generated {generated} parametric hypotheses")
    
    # Create index
    index_file = output_dir / 'INDEX.md'
    with open(index_file, 'w') as f:
        f.write("# Parametric Hypotheses Index\n\n")
        for hyp_id, name, template in all_hypotheses:
            f.write(f"- H{hyp_id}: {name} (template: {template})\n")
    
    # Create __init__.py
    init_file = output_dir / '__init__.py'
    with open(init_file, 'w') as f:
        f.write('"""Parametric generated hypotheses"""\n')
    
    return generated

if __name__ == "__main__":
    count = main()
    print(f"\n🎯 Stream 3: {count} parametric hypotheses ready")
