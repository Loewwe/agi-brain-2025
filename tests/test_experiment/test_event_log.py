"""Tests for EventLog determinism and correctness."""

import pytest
import tempfile
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path

from src.experiment.event_log import EventLog
from src.experiment.models import TradeEvent, TradeAction, DataSource, ExitReason


class TestEventLogBasic:
    """Basic EventLog functionality tests."""
    
    @pytest.fixture
    def temp_log_path(self):
        """Create temporary log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def sample_event(self):
        """Create a sample trade event."""
        return TradeEvent(
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
    
    def test_log_and_read_single_event(self, temp_log_path, sample_event):
        """Test logging and reading a single event."""
        log = EventLog(temp_log_path)
        
        # Log event
        log.log_trade(sample_event)
        
        # Read back
        events = log.get_trades_list(
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
        )
        
        assert len(events) == 1
        assert events[0].event_id == "test-001"
        assert events[0].symbol == "BTCUSDT"
    
    def test_log_multiple_events(self, temp_log_path):
        """Test logging multiple events."""
        log = EventLog(temp_log_path)
        
        events = [
            TradeEvent(
                event_id=f"test-{i:03d}",
                timestamp=datetime(2024, 12, 7, 10, i, 0),
                source=DataSource.BACKTEST,
                symbol="BTCUSDT",
                side="long",
                action=TradeAction.OPEN,
                price=Decimal("50000.00"),
                size=Decimal("0.01"),
                size_quote=Decimal("500.00"),
            )
            for i in range(10)
        ]
        
        count = log.log_trades(events)
        assert count == 10
        
        # Read back
        read_events = log.get_trades_list(
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
        )
        
        assert len(read_events) == 10
    
    def test_filter_by_symbol(self, temp_log_path):
        """Test filtering events by symbol."""
        log = EventLog(temp_log_path)
        
        # Log events for different symbols
        for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            for i in range(3):
                log.log_trade(TradeEvent(
                    event_id=f"{symbol}-{i}",
                    timestamp=datetime(2024, 12, 7, 10, i, 0),
                    source=DataSource.BACKTEST,
                    symbol=symbol,
                    side="long",
                    action=TradeAction.OPEN,
                    price=Decimal("100"),
                    size=Decimal("1"),
                    size_quote=Decimal("100"),
                ))
        
        # Filter by symbol
        btc_events = log.get_trades_list(
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
            symbols=["BTCUSDT"],
        )
        
        assert len(btc_events) == 3
        assert all(e.symbol == "BTCUSDT" for e in btc_events)
    
    def test_filter_by_source(self, temp_log_path):
        """Test filtering events by source."""
        log = EventLog(temp_log_path)
        
        # Log events from different sources
        for source in [DataSource.BACKTEST, DataSource.STAGE6]:
            log.log_trade(TradeEvent(
                event_id=f"{source.value}-001",
                timestamp=datetime(2024, 12, 7, 10, 0, 0),
                source=source,
                symbol="BTCUSDT",
                side="long",
                action=TradeAction.OPEN,
                price=Decimal("100"),
                size=Decimal("1"),
                size_quote=Decimal("100"),
            ))
        
        # Filter by source
        backtest_events = log.get_trades_list(
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
            source=DataSource.BACKTEST,
        )
        
        assert len(backtest_events) == 1
        assert backtest_events[0].source == DataSource.BACKTEST


class TestEventLogDeterminism:
    """Test EventLog determinism - same input produces same output."""
    
    @pytest.fixture
    def temp_log_path(self):
        """Create temporary log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_export_determinism(self, temp_log_path):
        """Test that export() produces deterministic output."""
        log = EventLog(temp_log_path)
        
        # Log events in random order
        events = [
            TradeEvent(
                event_id=f"test-{i:03d}",
                timestamp=datetime(2024, 12, 7, 10, i % 60, i // 60),
                source=DataSource.BACKTEST,
                symbol="BTCUSDT",
                side="long",
                action=TradeAction.OPEN,
                price=Decimal("50000.00"),
                size=Decimal("0.01"),
                size_quote=Decimal("500.00"),
            )
            for i in [5, 2, 8, 1, 9, 3, 7, 4, 6, 0]  # Random order
        ]
        
        for event in events:
            log.log_trade(event)
        
        # Export twice
        export_path1 = temp_log_path / "export1.jsonl"
        export_path2 = temp_log_path / "export2.jsonl"
        
        log.export(
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
            output_path=export_path1,
        )
        
        log.export(
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
            output_path=export_path2,
        )
        
        # Files should be identical
        with open(export_path1) as f1, open(export_path2) as f2:
            content1 = f1.read()
            content2 = f2.read()
        
        assert content1 == content2
    
    def test_export_sorted_by_timestamp(self, temp_log_path):
        """Test that exported events are sorted by timestamp."""
        log = EventLog(temp_log_path)
        
        # Log events in reverse order
        for i in range(10, 0, -1):
            log.log_trade(TradeEvent(
                event_id=f"test-{i:03d}",
                timestamp=datetime(2024, 12, 7, 10, i, 0),
                source=DataSource.BACKTEST,
                symbol="BTCUSDT",
                side="long",
                action=TradeAction.OPEN,
                price=Decimal("100"),
                size=Decimal("1"),
                size_quote=Decimal("100"),
            ))
        
        # Export
        export_path = temp_log_path / "export.jsonl"
        log.export(
            date_from=date(2024, 12, 7),
            date_to=date(2024, 12, 7),
            output_path=export_path,
        )
        
        # Read and verify order
        import json
        with open(export_path) as f:
            lines = [json.loads(line) for line in f]
        
        timestamps = [line["timestamp"] for line in lines]
        assert timestamps == sorted(timestamps)


class TestEventLogDateRange:
    """Test EventLog date range handling."""
    
    @pytest.fixture
    def temp_log_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_multi_day_range(self, temp_log_path):
        """Test reading events across multiple days."""
        log = EventLog(temp_log_path)
        
        # Log events on 3 different days
        for day in [5, 6, 7]:
            log.log_trade(TradeEvent(
                event_id=f"day-{day}",
                timestamp=datetime(2024, 12, day, 10, 0, 0),
                source=DataSource.BACKTEST,
                symbol="BTCUSDT",
                side="long",
                action=TradeAction.OPEN,
                price=Decimal("100"),
                size=Decimal("1"),
                size_quote=Decimal("100"),
            ))
        
        # Read full range
        events = log.get_trades_list(
            date_from=date(2024, 12, 5),
            date_to=date(2024, 12, 7),
        )
        
        assert len(events) == 3
    
    def test_empty_range(self, temp_log_path):
        """Test reading from empty date range."""
        log = EventLog(temp_log_path)
        
        events = log.get_trades_list(
            date_from=date(2024, 12, 1),
            date_to=date(2024, 12, 5),
        )
        
        assert len(events) == 0
