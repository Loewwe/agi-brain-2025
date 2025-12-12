"""
DatasetBuilder - Build datasets from local logs for backtesting.

Features:
- Deterministic: same config → same output
- Offline: no external API calls
- Feature engineering with versioned schema
"""

import hashlib
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from .models import DatasetConfig, DatasetMetadata, calculate_checksum
from .market_log import MarketLog
from .event_log import EventLog

logger = structlog.get_logger()


class DatasetBuilder:
    """
    Build datasets from local logs.
    
    Deterministic: same config → same output.
    
    Features schema versions:
    - v1: RSI 14, ATR 14, EMA 200, Volume SMA 20, Volume surge
    """
    
    def __init__(
        self,
        market_log: MarketLog,
        event_log: EventLog | None = None,
        output_path: Path | str | None = None,
    ):
        """
        Initialize DatasetBuilder.
        
        Args:
            market_log: MarketLog instance for OHLCV data
            event_log: Optional EventLog for trade history
            output_path: Optional path for saving datasets
        """
        self.market_log = market_log
        self.event_log = event_log
        self.output_path = Path(output_path) if output_path else None
        
        if self.output_path:
            self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("dataset_builder.initialized")
    
    def build(
        self,
        config: DatasetConfig,
        save: bool = True,
    ) -> tuple[pd.DataFrame, DatasetMetadata]:
        """
        Build dataset from logs.
        
        Args:
            config: Dataset configuration
            save: Whether to save dataset to disk
            
        Returns:
            Tuple of (DataFrame with OHLCV + features, Metadata)
        """
        logger.info(
            "dataset_builder.building",
            symbols=len(config.symbols),
            date_from=config.date_from.isoformat(),
            date_to=config.date_to.isoformat(),
        )
        
        dfs = []
        symbols_included = []
        
        for symbol in config.symbols:
            # Get market data
            df = self.market_log.get_bars(
                symbol=symbol,
                timeframe=config.timeframe,
                date_from=config.date_from,
                date_to=config.date_to,
            )
            
            if df.empty:
                logger.warning("dataset_builder.no_data", symbol=symbol)
                continue
            
            # Add symbol column
            df["symbol"] = symbol
            
            # Add features based on schema version
            if config.features_schema_version == "v1":
                df = self.add_features_v1(df)
            else:
                raise ValueError(f"Unknown features schema: {config.features_schema_version}")
            
            dfs.append(df)
            symbols_included.append(symbol)
        
        if not dfs:
            raise ValueError("No data available for any symbol in the specified range")
        
        # Combine all DataFrames
        dataset = pd.concat(dfs)
        dataset.sort_index(inplace=True)
        
        # Calculate checksum for reproducibility
        checksum = self._calculate_df_checksum(dataset)
        
        # Create metadata
        metadata = DatasetMetadata(
            config=config,
            created_at=datetime.now(),
            row_count=len(dataset),
            symbols_included=symbols_included,
            checksum=checksum,
        )
        
        # Save if requested
        if save and self.output_path:
            self._save_dataset(dataset, metadata)
        
        logger.info(
            "dataset_builder.built",
            row_count=len(dataset),
            symbols=len(symbols_included),
            checksum=checksum[:8],
        )
        
        return dataset, metadata
    
    def add_features_v1(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features (schema v1):
        - RSI 14
        - ATR 14
        - EMA 200
        - Volume SMA 20
        - Volume surge
        - Breakout levels
        
        Deterministic: uses only the data in df, no randomness.
        """
        df = df.copy()
        
        # === RSI 14 ===
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # RSI previous values for reversal detection
        df["rsi_prev"] = df["rsi"].shift(1)
        df["rsi_prev2"] = df["rsi"].shift(2)
        
        # === ATR 14 ===
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr"] / df["close"] * 100
        
        # === ADX 14 ===
        # +DM, -DM
        up = df["high"] - df["high"].shift(1)
        down = df["low"].shift(1) - df["low"]
        
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        
        # Smooth TR, +DM, -DM (using rolling mean for simplicity, matching ATR)
        tr_smooth = tr.rolling(14).mean()
        plus_dm_smooth = pd.Series(plus_dm, index=df.index).rolling(14).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=df.index).rolling(14).mean()
        
        # +DI, -DI
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        # DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # ADX (smooth DX)
        df["adx"] = dx.rolling(14).mean()
        
        # === EMA 200 ===
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
        df["ema_slope"] = df["ema200"].diff(3) > 0  # Rising over last 3 bars
        
        # Distance from EMA200 (Regime Feature)
        df["dist_to_ema200"] = (df["close"] - df["ema200"]) / df["ema200"]
        df["ema_distance_pct"] = df["dist_to_ema200"] * 100  # Keep legacy name for now
        
        # === Volume SMA 20 ===
        df["volume_sma"] = df["volume"].rolling(20).mean()
        df["volume_surge"] = df["volume"] / df["volume_sma"]
        
        # === Breakout levels (2-bar lookback for faster signals) ===
        df["high_2"] = df["high"].rolling(2).max().shift(1)
        df["low_2"] = df["low"].rolling(2).min().shift(1)
        
        # === Additional useful features ===
        # Candle body and wicks
        df["body"] = (df["close"] - df["open"]).abs()
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
        
        # Bar range
        df["bar_range"] = df["high"] - df["low"]
        df["bar_range_pct"] = df["bar_range"] / df["close"] * 100
        
        # Candle Amplitude (Regime Feature)
        df["candle_amplitude"] = df["bar_range"] / df["close"]
        
        # Volatility Ratio (Regime Feature: Current ATR / Long-term ATR)
        # Measures if current volatility is high or low relative to history
        df["volatility_ratio"] = df["atr"] / df["atr"].rolling(50).mean()
        
        # Close position in bar (0 = low, 1 = high)
        df["close_position"] = (df["close"] - df["low"]) / df["bar_range"]
        df["close_position"] = df["close_position"].fillna(0.5)
        
        return df
    
    def _calculate_df_checksum(self, df: pd.DataFrame) -> str:
        """Calculate MD5 checksum of DataFrame for reproducibility."""
        # Use a deterministic string representation
        data_str = df.to_csv(index=True, date_format="%Y-%m-%d %H:%M:%S")
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _save_dataset(self, df: pd.DataFrame, metadata: DatasetMetadata) -> Path:
        """Save dataset and metadata to disk."""
        if not self.output_path:
            raise ValueError("output_path not configured")
        
        # Generate filename from config
        date_range = f"{metadata.config.date_from}_{metadata.config.date_to}"
        filename = f"dataset_{date_range}_{metadata.checksum[:8]}"
        
        # Save dataset as parquet
        data_path = self.output_path / f"{filename}.parquet"
        df.to_parquet(data_path, engine="pyarrow")
        
        # Save metadata as JSON
        meta_path = self.output_path / f"{filename}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)
        
        logger.info(
            "dataset_builder.saved",
            data_path=str(data_path),
            meta_path=str(meta_path),
        )
        
        return data_path
    
    def load_dataset(self, path: Path | str) -> tuple[pd.DataFrame, DatasetMetadata]:
        """
        Load dataset and metadata from disk.
        
        Args:
            path: Path to dataset parquet file
            
        Returns:
            Tuple of (DataFrame, Metadata)
        """
        path = Path(path)
        
        # Load data
        df = pd.read_parquet(path)
        
        # Load metadata
        meta_path = path.with_suffix("").with_suffix("_meta.json")
        if not meta_path.name.endswith("_meta.json"):
            meta_path = path.parent / f"{path.stem}_meta.json"
        
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
        
        metadata = DatasetMetadata.from_dict(meta_dict)
        
        return df, metadata
    
    def validate(self, df: pd.DataFrame, metadata: DatasetMetadata) -> bool:
        """
        Validate dataset integrity.
        
        Args:
            df: Dataset DataFrame
            metadata: Dataset metadata
            
        Returns:
            True if valid, False otherwise
        """
        # Check row count
        if len(df) != metadata.row_count:
            logger.warning(
                "dataset_builder.validation_failed",
                reason="row_count_mismatch",
                expected=metadata.row_count,
                actual=len(df),
            )
            return False
        
        # Check checksum
        actual_checksum = self._calculate_df_checksum(df)
        if actual_checksum != metadata.checksum:
            logger.warning(
                "dataset_builder.validation_failed",
                reason="checksum_mismatch",
                expected=metadata.checksum,
                actual=actual_checksum,
            )
            return False
        
        # Check required columns for v1 schema
        if metadata.config.features_schema_version == "v1":
            required_columns = [
                "open", "high", "low", "close", "volume",
                "rsi", "atr", "ema200", "volume_sma", "volume_surge",
            ]
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                logger.warning(
                    "dataset_builder.validation_failed",
                    reason="missing_columns",
                    missing=missing,
                )
                return False
        
        logger.info("dataset_builder.validation_passed")
        return True
    
    def list_datasets(self) -> list[dict]:
        """List available datasets."""
        if not self.output_path:
            return []
        
        datasets = []
        for meta_path in self.output_path.glob("*_meta.json"):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_dict = json.load(f)
                
                datasets.append({
                    "filename": meta_path.stem.replace("_meta", ""),
                    "metadata": meta_dict,
                })
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("dataset_builder.read_error", path=str(meta_path), error=str(e))
        
        return datasets
