"""Tests for MarketLog functionality and determinism."""

import pytest
import tempfile
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import numpy as np

from src.experiment.market_log import MarketLog
from src.experiment.models import MarketBar


class TestMarketLogBasic:
    """Basic MarketLog functionality tests."""
    
    @pytest.fixture
    def temp_path(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def sample_bars(self):
        """Create sample market bars."""
        return [
            MarketBar(
                timestamp=datetime(2024, 12, 7, 10, i * 5, 0),
                symbol="BTC/USDT:USDT",
                timeframe="5m",
                open=Decimal("50000.00"),
                high=Decimal("50100.00"),
                low=Decimal("49900.00"),
                close=Decimal("50050.00"),
                volume=Decimal("1000.00"),
            )
            for i in range(12)  # 1 hour of data
        ]
    
    def test_store_and_get_bars(self, temp_path, sample_bars):
        """Test storing and retrieving bars."""
        log = MarketLog(temp_path)
        
        # Store bars
        count = log.store_bars(sample_bars)
        assert count == 12
        
        # Get bars back
        df = log.get_bars(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
        )
        
        assert len(df) == 12
        assert "open" in df.columns
        assert "close" in df.columns
    
    def test_store_dataframe(self, temp_path):
        """Test storing data from DataFrame."""
        log = MarketLog(temp_path)
        
        # Create DataFrame
        dates = pd.date_range(
            start="2024-12-07 10:00:00",
            periods=24,
            freq="5min",
        )
        df = pd.DataFrame({
            "open": np.random.uniform(50000, 50100, 24),
            "high": np.random.uniform(50100, 50200, 24),
            "low": np.random.uniform(49900, 50000, 24),
            "close": np.random.uniform(50000, 50100, 24),
            "volume": np.random.uniform(1000, 2000, 24),
        }, index=dates)
        
        # Store
        count = log.store_dataframe("BTCUSDT", "5m", df)
        assert count == 24
        
        # Get back
        result = log.get_bars(
            symbol="BTCUSDT",
            timeframe="5m",
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
        )
        
        assert len(result) == 24
    
    def test_empty_range(self, temp_path):
        """Test getting bars from empty range."""
        log = MarketLog(temp_path)
        
        df = log.get_bars(
            symbol="BTCUSDT",
            timeframe="5m",
            date_from=date(2024, 12, 1),
            date_to=date(2024, 12, 5),
        )
        
        assert df.empty
        assert "open" in df.columns  # Still has schema


class TestMarketLogDeterminism:
    """Test MarketLog determinism."""
    
    @pytest.fixture
    def temp_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_same_query_same_result(self, temp_path):
        """Test that same query returns identical results."""
        log = MarketLog(temp_path)
        
        # Create and store data
        np.random.seed(42)
        dates = pd.date_range(
            start="2024-12-07 10:00:00",
            periods=100,
            freq="5min",
        )
        df = pd.DataFrame({
            "open": np.random.uniform(50000, 50100, 100),
            "high": np.random.uniform(50100, 50200, 100),
            "low": np.random.uniform(49900, 50000, 100),
            "close": np.random.uniform(50000, 50100, 100),
            "volume": np.random.uniform(1000, 2000, 100),
        }, index=dates)
        
        log.store_dataframe("BTCUSDT", "5m", df)
        
        # Query twice
        result1 = log.get_bars(
            symbol="BTCUSDT",
            timeframe="5m",
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
        )
        
        result2 = log.get_bars(
            symbol="BTCUSDT",
            timeframe="5m",
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
        )
        
        # Should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_no_duplicates_on_reinsert(self, temp_path):
        """Test that reinserting same data doesn't create duplicates."""
        log = MarketLog(temp_path)
        
        # Create data
        dates = pd.date_range(
            start="2024-12-07 10:00:00",
            periods=10,
            freq="5min",
        )
        df = pd.DataFrame({
            "open": [50000.0] * 10,
            "high": [50100.0] * 10,
            "low": [49900.0] * 10,
            "close": [50050.0] * 10,
            "volume": [1000.0] * 10,
        }, index=dates)
        
        # Insert twice
        log.store_dataframe("BTCUSDT", "5m", df)
        log.store_dataframe("BTCUSDT", "5m", df)
        
        # Should still have only 10 rows
        result = log.get_bars(
            symbol="BTCUSDT",
            timeframe="5m",
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
        )
        
        assert len(result) == 10


class TestMarketLogCache:
    """Test MarketLog caching behavior."""
    
    @pytest.fixture
    def temp_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_cache_enabled(self, temp_path):
        """Test that caching works correctly."""
        log = MarketLog(temp_path, cache_enabled=True)
        
        # Create data
        dates = pd.date_range(
            start="2024-12-07 10:00:00",
            periods=10,
            freq="5min",
        )
        df = pd.DataFrame({
            "open": [50000.0] * 10,
            "high": [50100.0] * 10,
            "low": [49900.0] * 10,
            "close": [50050.0] * 10,
            "volume": [1000.0] * 10,
        }, index=dates)
        
        log.store_dataframe("BTCUSDT", "5m", df)
        
        # First query - loads from disk
        result1 = log.get_bars(
            symbol="BTCUSDT",
            timeframe="5m",
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
        )
        
        # Second query - should come from cache
        result2 = log.get_bars(
            symbol="BTCUSDT",
            timeframe="5m",
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
        )
        
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_cache_clear(self, temp_path):
        """Test cache clearing."""
        log = MarketLog(temp_path, cache_enabled=True)
        
        # Store and query to populate cache
        dates = pd.date_range(
            start="2024-12-07 10:00:00",
            periods=10,
            freq="5min",
        )
        df = pd.DataFrame({
            "open": [50000.0] * 10,
            "high": [50100.0] * 10,
            "low": [49900.0] * 10,
            "close": [50050.0] * 10,
            "volume": [1000.0] * 10,
        }, index=dates)
        
        log.store_dataframe("BTCUSDT", "5m", df)
        log.get_bars(
            symbol="BTCUSDT",
            timeframe="5m",
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
        )
        
        # Clear cache
        log.clear_cache()
        
        # Cache should be empty
        assert len(log._cache) == 0
