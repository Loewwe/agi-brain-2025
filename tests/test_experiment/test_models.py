"""Tests for experiment module models."""

import pytest
from datetime import date, datetime
from decimal import Decimal

from src.experiment.models import (
    TradeEvent,
    MarketBar,
    DatasetConfig,
    DatasetMetadata,
    SimulatorConfig,
    ExperimentResult,
    TradeAction,
    DataSource,
    ExitReason,
    calculate_checksum,
)


class TestTradeEvent:
    """Tests for TradeEvent model."""
    
    def test_create_trade_event(self):
        """Test creating a trade event."""
        event = TradeEvent(
            event_id="test-001",
            timestamp=datetime(2024, 12, 7, 10, 0, 0),
            source=DataSource.BACKTEST,
            symbol="BTCUSDT",
            side="long",
            action=TradeAction.OPEN,
            price=Decimal("50000.00"),
            size=Decimal("0.01"),
            size_quote=Decimal("500.00"),
            leverage=10,
            trade_id="trade-001",
            position_id="pos-001",
            strategy_id="stage6",
        )
        
        assert event.event_id == "test-001"
        assert event.symbol == "BTCUSDT"
        assert event.side == "long"
        assert event.price == Decimal("50000.00")
    
    def test_trade_event_serialization(self):
        """Test serialization/deserialization."""
        event = TradeEvent(
            event_id="test-002",
            timestamp=datetime(2024, 12, 7, 10, 0, 0),
            source=DataSource.STAGE6,
            symbol="ETHUSDT",
            side="short",
            action=TradeAction.CLOSE,
            price=Decimal("3000.00"),
            size=Decimal("1.0"),
            size_quote=Decimal("3000.00"),
            entry_price=Decimal("3100.00"),
            exit_price=Decimal("3000.00"),
            realized_pnl=Decimal("100.00"),
            exit_reason=ExitReason.TP,
        )
        
        # Serialize
        data = event.to_dict()
        assert data["event_id"] == "test-002"
        assert data["source"] == "stage6"
        assert data["exit_reason"] == "TP"
        
        # Deserialize
        restored = TradeEvent.from_dict(data)
        assert restored.event_id == event.event_id
        assert restored.source == event.source
        assert restored.exit_reason == event.exit_reason


class TestDatasetConfig:
    """Tests for DatasetConfig model."""
    
    def test_create_dataset_config(self):
        """Test creating dataset config."""
        config = DatasetConfig(
            symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
            date_from=date(2024, 9, 1),
            date_to=date(2024, 12, 1),
            timeframe="5m",
        )
        
        assert len(config.symbols) == 2
        assert config.features_schema_version == "v1"
    
    def test_dataset_config_serialization(self):
        """Test serialization/deserialization."""
        config = DatasetConfig(
            symbols=["SOL/USDT:USDT"],
            date_from=date(2024, 10, 1),
            date_to=date(2024, 11, 1),
        )
        
        data = config.to_dict()
        restored = DatasetConfig.from_dict(data)
        
        assert restored.symbols == config.symbols
        assert restored.date_from == config.date_from


class TestSimulatorConfig:
    """Tests for SimulatorConfig model."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SimulatorConfig()
        
        assert config.start_balance == 200.0
        assert config.leverage == 20
        assert config.risk_per_trade == 0.009
        assert config.max_positions == 3
    
    def test_config_serialization(self):
        """Test serialization/deserialization."""
        config = SimulatorConfig(
            start_balance=1000.0,
            leverage=10,
            risk_per_trade=0.01,
        )
        
        data = config.to_dict()
        restored = SimulatorConfig.from_dict(data)
        
        assert restored.start_balance == 1000.0
        assert restored.leverage == 10


class TestExperimentResult:
    """Tests for ExperimentResult model."""
    
    def test_create_result(self):
        """Test creating experiment result."""
        result = ExperimentResult(
            total_trades=100,
            win_rate=0.55,
            profit_factor=1.8,
            avg_r_multiple=0.3,
            expectancy=0.25,
            total_return_pct=15.0,
            avg_daily_pnl_pct=0.5,
            max_drawdown_pct=-3.0,
            sharpe_ratio=1.5,
            risk_adjusted_return=5.0,
        )
        
        assert result.total_trades == 100
        assert result.win_rate == 0.55
    
    def test_result_serialization(self):
        """Test serialization/deserialization."""
        result = ExperimentResult(
            total_trades=50,
            win_rate=0.6,
            profit_factor=2.0,
            avg_r_multiple=0.4,
            expectancy=0.3,
            total_return_pct=10.0,
            avg_daily_pnl_pct=0.3,
            max_drawdown_pct=-2.0,
            sharpe_ratio=1.2,
            risk_adjusted_return=5.0,
            exits_by_reason={"TP": 30, "SL": 20},
        )
        
        data = result.to_dict()
        restored = ExperimentResult.from_dict(data)
        
        assert restored.total_trades == 50
        assert restored.exits_by_reason == {"TP": 30, "SL": 20}


class TestChecksum:
    """Tests for checksum calculation."""
    
    def test_checksum_determinism(self):
        """Test that checksum is deterministic."""
        data = {"a": 1, "b": "test", "c": [1, 2, 3]}
        
        checksum1 = calculate_checksum(data)
        checksum2 = calculate_checksum(data)
        
        assert checksum1 == checksum2
    
    def test_checksum_different_data(self):
        """Test that different data produces different checksum."""
        data1 = {"a": 1}
        data2 = {"a": 2}
        
        assert calculate_checksum(data1) != calculate_checksum(data2)
