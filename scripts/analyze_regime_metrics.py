
import pandas as pd
import numpy as np
import structlog
from datetime import datetime
from src.experiment.dataset_builder import DatasetBuilder, DatasetConfig
from src.experiment.labels import LabelsBuilder, LabelConfig, create_ml_dataset
from src.experiment.alpha_engine import AlphaEngine, AlphaConfig

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()

def analyze_regime():
    print("--- Regime Analysis: Monthly Metrics ---")
    
    # 1. Load Data
    print("Loading data...")
    config = DatasetConfig(
        symbols=["BTC/USDT", "ETH/USDT"],
        date_from=datetime.strptime("2024-01-01", "%Y-%m-%d").date(),
        date_to=datetime.strptime("2024-06-01", "%Y-%m-%d").date(),
        timeframe="5m",
        features_schema_version="v1",
    )
    from src.experiment.market_log import MarketLog
    
    market_log = MarketLog("data/market")
    builder = DatasetBuilder(market_log)
    dataset, _ = builder.build(config, save=False)
    
    # 2. Label Data
    print("Generating labels...")
    label_config = LabelConfig(
        horizon_bars=3,  # 15m horizon (as per latest tuning)
        target_type="direction",
        use_log_returns=True,
    )
    # We use create_ml_dataset to get X and y, but we also need the original index for grouping
    # so we'll do it manually to keep the dataframe structure
    lb = LabelsBuilder(label_config)
    df_labeled = lb.add_labels(dataset)
    df_labeled = df_labeled.dropna(subset=["future_return"])
    
    # 3. Train Model (Jan-Apr)
    print("Training Alpha Engine (Jan-Apr)...")
    
    # Split Train/Test
    # Train: Jan-Apr
    # Test: May
    train_mask = (df_labeled.index >= "2024-01-01") & (df_labeled.index < "2024-05-01")
    test_mask = (df_labeled.index >= "2024-05-01") & (df_labeled.index < "2024-06-01")
    
    train_df = df_labeled[train_mask]
    test_df = df_labeled[test_mask]
    
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    
    alpha_config = AlphaConfig(
        model_type="lightgbm",
        target_column="target",
        train_ratio=0.8, # Internal split for validation
        val_ratio=0.2,
    )
    
    engine = AlphaEngine(alpha_config)
    
    engine.train(train_df)
    
    # 4. Analyze per Month
    print("\n--- Monthly Performance ---")
    
    # Predict on full dataset
    # AlphaEngine.predict handles feature selection internally
    preds = engine.predict(df_labeled)
    df_labeled["alpha_score"] = preds["alpha_score"]
    df_labeled["pred_direction"] = preds["alpha_direction"]
    
    # Group by Month
    df_labeled["month"] = df_labeled.index.to_period("M")
    
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    results = []
    for month, group in df_labeled.groupby("month"):
        y_true = group["target"]
        y_score = group["alpha_score"]
        y_pred = group["pred_direction"]
        
        auc = roc_auc_score(y_true, y_score)
        acc = accuracy_score(y_true, y_pred)
        count = len(group)
        
        results.append({
            "Month": str(month),
            "Count": count,
            "AUC": auc,
            "Accuracy": acc
        })
    
    # Print Table
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Check for drop
    jan_apr_auc = results_df[results_df["Month"] < "2024-05"]["AUC"].mean()
    may_auc = results_df[results_df["Month"] == "2024-05"]["AUC"].values[0]
    
    print(f"\nJan-Apr Avg AUC: {jan_apr_auc:.4f}")
    print(f"May AUC: {may_auc:.4f}")
    
    if jan_apr_auc - may_auc > 0.05:
        print(">> CONFIRMED: Significant performance drop in May (Regime Mismatch).")
    else:
        print(">> INCONCLUSIVE: Performance is stable.")

    # 5. Analyze Regime Characteristics
    print("\n--- Regime Characteristics Analysis ---")
    df = df_labeled.copy()
    
    # Calculate Monthly Stats per Symbol
    df["month"] = df.index.to_period("M")
    
    monthly_stats = []
    for symbol, sym_group in df.groupby("symbol"):
        for month, group in sym_group.groupby("month"):
            # Price Change
            price_change_pct = (group["close"].iloc[-1] - group["close"].iloc[0]) / group["close"].iloc[0] * 100
            
            # Volatility (ATR)
            if "atr_pct" in group.columns:
                avg_volatility = group["atr_pct"].mean()
            else:
                avg_volatility = 0.0
                
            # Bull/Bear Ratio (Close > Open)
            bull_ratio = (group["close"] > group["open"]).mean()
            
            monthly_stats.append({
                "Symbol": symbol,
                "Month": str(month),
                "Return %": price_change_pct,
                "Avg Volatility %": avg_volatility,
                "Bull Ratio": bull_ratio
            })
        
    stats_df = pd.DataFrame(monthly_stats)
    print(stats_df.to_string(index=False))

if __name__ == "__main__":
    analyze_regime()
