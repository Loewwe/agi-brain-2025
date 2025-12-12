
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import structlog

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.experiment.labels import LabelConfig, LabelsBuilder

logger = structlog.get_logger()

def test_threshold_target():
    print("\n--- Testing Threshold Target ---")
    
    # Create dummy data: 100 bars
    # Close prices that create specific returns
    # Horizon = 1 bar
    # Return = (Close[t+1] - Close[t]) / Close[t]
    
    prices = [100.0] * 100
    # Bar 10 -> 11: +2% return (100 -> 102)
    prices[11] = 102.0
    # Bar 20 -> 21: -2% return (100 -> 98)
    prices[21] = 98.0
    # Bar 30 -> 31: +0.5% return (100 -> 100.5)
    prices[31] = 100.5
    
    df = pd.DataFrame({"close": prices})
    
    config = LabelConfig(
        horizon_bars=1,
        target_type="threshold",
        up_threshold_pct=1.0,    # +1%
        down_threshold_pct=-1.0, # -1%
    )
    
    builder = LabelsBuilder(config)
    df_labeled = builder.add_labels(df)
    
    # Check specific bars
    # Bar 10: Future return (11) is +2% -> Should be 1
    # Bar 20: Future return (21) is -2% -> Should be -1
    # Bar 30: Future return (31) is +0.5% -> Should be 0 (neutral)
    
    print(f"Bar 10 Return: {df_labeled.loc[10, 'future_return']:.2f}%")
    print(f"Bar 10 Target: {df_labeled.loc[10, 'target']}")
    
    print(f"Bar 20 Return: {df_labeled.loc[20, 'future_return']:.2f}%")
    print(f"Bar 20 Target: {df_labeled.loc[20, 'target']}")
    
    print(f"Bar 30 Return: {df_labeled.loc[30, 'future_return']:.2f}%")
    print(f"Bar 30 Target: {df_labeled.loc[30, 'target']}")
    
    assert df_labeled.loc[10, "target"] == 1
    assert df_labeled.loc[20, "target"] == -1
    assert df_labeled.loc[30, "target"] == 0
    
    print("✅ Threshold Target Logic Verified")

def test_decile_target():
    print("\n--- Testing Decile Target ---")
    
    # Create random data
    np.random.seed(42)
    returns = np.random.normal(0, 1, 1000)
    # Construct prices from returns
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r/100))
        
    df = pd.DataFrame({"close": prices})
    
    config = LabelConfig(
        horizon_bars=1,
        target_type="decile",
        decile_period_bars=100,
        decile_threshold=0.9, # Top 10%
    )
    
    builder = LabelsBuilder(config)
    df_labeled = builder.add_labels(df)
    
    # Check distribution
    stats = builder.get_label_stats(df_labeled)
    print("Label Stats:", stats)
    
    # Top ratio should be roughly 10% (ignoring initial warmup)
    print(f"Top Ratio: {stats.get('top_ratio', 0):.2%}")
    
    # Verify logic manually for a window
    # Pick a random index, say 500
    idx = 500
    window_start = idx - 100 + 1
    window_returns = df_labeled.loc[window_start:idx, "future_return"]
    current_return = df_labeled.loc[idx, "future_return"]
    
    quantile_90 = window_returns.quantile(0.9)
    target = df_labeled.loc[idx, "target"]
    
    print(f"Idx {idx}: Return={current_return:.4f}, 90th%={quantile_90:.4f}, Target={target}")
    
    expected_target = 1 if current_return > quantile_90 else 0
    assert target == expected_target
    
    print("✅ Decile Target Logic Verified")

if __name__ == "__main__":
    test_threshold_target()
    test_decile_target()
