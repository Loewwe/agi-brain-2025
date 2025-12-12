"""Tests for DatasetBuilder functionality and determinism."""

import pytest
import tempfile
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import numpy as np

from src.experiment.dataset_builder import DatasetBuilder
from src.experiment.market_log import MarketLog
from src.experiment.models import DatasetConfig


class TestDatasetBuilderBasic:
    """Basic DatasetBuilder functionality tests."""
    
    @pytest.fixture
    def setup(self):
        """Set up test environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            market_path = tmpdir / "market"
            output_path = tmpdir / "datasets"
            
            # Create MarketLog with sample data
            market_log = MarketLog(market_path)
            
            # Create sample data for 2 symbols
            np.random.seed(42)
            for symbol in ["BTCUSDT", "ETHUSDT"]:
                dates = pd.date_range(
                    start="2024-12-07 00:00:00",
                    periods=288,  # 1 day at 5m
                    freq="5min",
                )
                df = pd.DataFrame({
                    "open": np.random.uniform(50000, 50100, 288) if symbol == "BTCUSDT" else np.random.uniform(3000, 3050, 288),
                    "high": np.random.uniform(50100, 50200, 288) if symbol == "BTCUSDT" else np.random.uniform(3050, 3100, 288),
                    "low": np.random.uniform(49900, 50000, 288) if symbol == "BTCUSDT" else np.random.uniform(2950, 3000, 288),
                    "close": np.random.uniform(50000, 50100, 288) if symbol == "BTCUSDT" else np.random.uniform(3000, 3050, 288),
                    "volume": np.random.uniform(1000, 5000, 288),
                }, index=dates)
                
                market_log.store_dataframe(symbol, "5m", df)
            
            builder = DatasetBuilder(
                market_log=market_log,
                output_path=output_path,
            )
            
            yield builder, market_log, output_path
    
    def test_build_single_symbol(self, setup):
        """Test building dataset for single symbol."""
        builder, market_log, output_path = setup
        
        config = DatasetConfig(
            symbols=["BTCUSDT"],
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
            timeframe="5m",
        )
        
        df, metadata = builder.build(config, save=False)
        
        assert len(df) > 0
        assert "symbol" in df.columns
        assert "rsi" in df.columns
        assert "ema200" in df.columns
        assert metadata.row_count == len(df)
    
    def test_build_multiple_symbols(self, setup):
        """Test building dataset for multiple symbols."""
        builder, market_log, output_path = setup
        
        config = DatasetConfig(
            symbols=["BTCUSDT", "ETHUSDT"],
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
            timeframe="5m",
        )
        
        df, metadata = builder.build(config, save=False)
        
        assert len(df) > 0
        assert len(df["symbol"].unique()) == 2
        assert "BTCUSDT" in df["symbol"].values
        assert "ETHUSDT" in df["symbol"].values
    
    def test_features_v1_added(self, setup):
        """Test that v1 features are correctly added."""
        builder, market_log, output_path = setup
        
        config = DatasetConfig(
            symbols=["BTCUSDT"],
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
            features_schema_version="v1",
        )
        
        df, metadata = builder.build(config, save=False)
        
        # Check all v1 features exist
        v1_features = [
            "rsi", "rsi_prev", "rsi_prev2",
            "atr", "atr_pct",
            "ema200", "ema_slope", "ema_distance_pct",
            "volume_sma", "volume_surge",
            "high_2", "low_2",
            "body", "upper_wick", "lower_wick",
            "bar_range", "bar_range_pct", "close_position",
        ]
        
        for feature in v1_features:
            assert feature in df.columns, f"Missing feature: {feature}"


class TestDatasetBuilderDeterminism:
    """Test DatasetBuilder determinism - same config produces same output."""
    
    @pytest.fixture
    def setup(self):
        """Set up test environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            market_path = tmpdir / "market"
            output_path = tmpdir / "datasets"
            
            market_log = MarketLog(market_path)
            
            # Create deterministic data
            np.random.seed(42)
            dates = pd.date_range(
                start="2024-12-07 00:00:00",
                periods=288,
                freq="5min",
            )
            df = pd.DataFrame({
                "open": np.random.uniform(50000, 50100, 288),
                "high": np.random.uniform(50100, 50200, 288),
                "low": np.random.uniform(49900, 50000, 288),
                "close": np.random.uniform(50000, 50100, 288),
                "volume": np.random.uniform(1000, 5000, 288),
            }, index=dates)
            
            market_log.store_dataframe("BTCUSDT", "5m", df)
            
            builder = DatasetBuilder(
                market_log=market_log,
                output_path=output_path,
            )
            
            yield builder, output_path
    
    def test_same_config_same_checksum(self, setup):
        """Test that same config produces same checksum."""
        builder, output_path = setup
        
        config = DatasetConfig(
            symbols=["BTCUSDT"],
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
            timeframe="5m",
        )
        
        # Build twice
        df1, meta1 = builder.build(config, save=False)
        df2, meta2 = builder.build(config, save=False)
        
        # Checksums should match
        assert meta1.checksum == meta2.checksum
    
    def test_same_config_same_data(self, setup):
        """Test that same config produces identical data."""
        builder, output_path = setup
        
        config = DatasetConfig(
            symbols=["BTCUSDT"],
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
            timeframe="5m",
        )
        
        # Build twice
        df1, _ = builder.build(config, save=False)
        df2, _ = builder.build(config, save=False)
        
        # Data should be identical
        pd.testing.assert_frame_equal(df1, df2)


class TestDatasetBuilderValidation:
    """Test DatasetBuilder validation functionality."""
    
    @pytest.fixture
    def setup(self):
        """Set up test environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            market_path = tmpdir / "market"
            output_path = tmpdir / "datasets"
            
            market_log = MarketLog(market_path)
            
            np.random.seed(42)
            dates = pd.date_range(
                start="2024-12-07 00:00:00",
                periods=288,
                freq="5min",
            )
            df = pd.DataFrame({
                "open": np.random.uniform(50000, 50100, 288),
                "high": np.random.uniform(50100, 50200, 288),
                "low": np.random.uniform(49900, 50000, 288),
                "close": np.random.uniform(50000, 50100, 288),
                "volume": np.random.uniform(1000, 5000, 288),
            }, index=dates)
            
            market_log.store_dataframe("BTCUSDT", "5m", df)
            
            builder = DatasetBuilder(
                market_log=market_log,
                output_path=output_path,
            )
            
            yield builder, output_path
    
    def test_validate_valid_dataset(self, setup):
        """Test validation of valid dataset."""
        builder, output_path = setup
        
        config = DatasetConfig(
            symbols=["BTCUSDT"],
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
            timeframe="5m",
        )
        
        df, metadata = builder.build(config, save=False)
        
        # Should validate successfully
        assert builder.validate(df, metadata)
    
    def test_validate_wrong_row_count(self, setup):
        """Test validation catches wrong row count."""
        builder, output_path = setup
        
        config = DatasetConfig(
            symbols=["BTCUSDT"],
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
            timeframe="5m",
        )
        
        df, metadata = builder.build(config, save=False)
        
        # Modify row count in metadata
        metadata.row_count = metadata.row_count + 100
        
        # Should fail validation
        assert not builder.validate(df, metadata)
    
    def test_validate_wrong_checksum(self, setup):
        """Test validation catches wrong checksum."""
        builder, output_path = setup
        
        config = DatasetConfig(
            symbols=["BTCUSDT"],
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
            timeframe="5m",
        )
        
        df, metadata = builder.build(config, save=False)
        
        # Modify checksum
        metadata.checksum = "invalid_checksum"
        
        # Should fail validation
        assert not builder.validate(df, metadata)
