"""
EventLog - Unified trade event logging for Stage6/Stage7/Backtest.

Storage format: JSONL (append-only, deterministic)
Path structure: data/events/{source}/trades_{date}.jsonl
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Iterator

import structlog

from .models import TradeEvent, DataSource

logger = structlog.get_logger()


class EventLog:
    """
    Unified event logging for trades and positions.
    
    Features:
    - Append-only JSONL storage
    - Date-based file organization
    - Deterministic export
    - Source filtering (stage6/stage7/backtest)
    
    Path structure:
        data/events/{source}/trades_{YYYY-MM-DD}.jsonl
    """
    
    def __init__(self, base_path: Path | str):
        """
        Initialize EventLog.
        
        Args:
            base_path: Base directory for event storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create source directories
        for source in DataSource:
            (self.base_path / source.value).mkdir(exist_ok=True)
        
        logger.info("event_log.initialized", base_path=str(self.base_path))
    
    def _get_file_path(self, source: DataSource, event_date: date) -> Path:
        """Get file path for a specific source and date."""
        return self.base_path / source.value / f"trades_{event_date.isoformat()}.jsonl"
    
    def log_trade(self, event: TradeEvent) -> None:
        """
        Append trade event to log.
        
        Args:
            event: Trade event to log
        """
        event_date = event.timestamp.date()
        file_path = self._get_file_path(event.source, event_date)
        
        # Append to JSONL file
        with open(file_path, "a", encoding="utf-8") as f:
            json_line = json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)
            f.write(json_line + "\n")
        
        logger.debug(
            "event_log.logged",
            event_id=event.event_id,
            symbol=event.symbol,
            action=event.action.value,
        )
    
    def log_trades(self, events: list[TradeEvent]) -> int:
        """
        Batch log multiple trade events.
        
        Args:
            events: List of trade events to log
            
        Returns:
            Number of events logged
        """
        # Group by source and date for efficient file writing
        grouped: dict[tuple[DataSource, date], list[TradeEvent]] = {}
        
        for event in events:
            key = (event.source, event.timestamp.date())
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(event)
        
        count = 0
        for (source, event_date), batch in grouped.items():
            file_path = self._get_file_path(source, event_date)
            
            with open(file_path, "a", encoding="utf-8") as f:
                for event in batch:
                    json_line = json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)
                    f.write(json_line + "\n")
                    count += 1
        
        logger.info("event_log.batch_logged", count=count)
        return count
    
    def get_trades(
        self,
        date_from: date,
        date_to: date,
        symbols: list[str] | None = None,
        source: DataSource | str | None = None,
    ) -> Iterator[TradeEvent]:
        """
        Read trades for date range.
        
        Args:
            date_from: Start date (inclusive)
            date_to: End date (inclusive)
            symbols: Optional filter by symbols
            source: Optional filter by source
            
        Yields:
            TradeEvent objects
        """
        # Determine sources to read
        if source:
            sources = [DataSource(source) if isinstance(source, str) else source]
        else:
            sources = list(DataSource)
        
        # Normalize symbols filter
        symbol_set = set(symbols) if symbols else None
        
        # Iterate through dates
        current_date = date_from
        while current_date <= date_to:
            for src in sources:
                file_path = self._get_file_path(src, current_date)
                
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            
                            try:
                                data = json.loads(line)
                                event = TradeEvent.from_dict(data)
                                
                                # Apply symbol filter
                                if symbol_set and event.symbol not in symbol_set:
                                    continue
                                
                                yield event
                            except (json.JSONDecodeError, KeyError, ValueError) as e:
                                logger.warning(
                                    "event_log.parse_error",
                                    file=str(file_path),
                                    error=str(e),
                                )
            
            # Move to next day using timedelta
            from datetime import timedelta
            current_date = current_date + timedelta(days=1)
    
    def get_trades_list(
        self,
        date_from: date,
        date_to: date,
        symbols: list[str] | None = None,
        source: DataSource | str | None = None,
    ) -> list[TradeEvent]:
        """
        Read trades for date range as a list (for testing/small datasets).
        
        Args:
            date_from: Start date (inclusive)
            date_to: End date (inclusive)
            symbols: Optional filter by symbols
            source: Optional filter by source
            
        Returns:
            List of TradeEvent objects
        """
        return list(self.get_trades(date_from, date_to, symbols, source))
    
    def export(
        self,
        date_from: date,
        date_to: date,
        output_path: Path | str,
        symbols: list[str] | None = None,
        source: DataSource | str | None = None,
    ) -> Path:
        """
        Export trades to a single file (deterministic).
        
        Exports are sorted by timestamp for determinism.
        Same input parameters → same output file.
        
        Args:
            date_from: Start date (inclusive)
            date_to: End date (inclusive)
            output_path: Path for output file
            symbols: Optional filter by symbols
            source: Optional filter by source
            
        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Collect and sort trades
        trades = list(self.get_trades(date_from, date_to, symbols, source))
        trades.sort(key=lambda t: (t.timestamp, t.event_id))
        
        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            for trade in trades:
                json_line = json.dumps(trade.to_dict(), ensure_ascii=False, sort_keys=True)
                f.write(json_line + "\n")
        
        logger.info(
            "event_log.exported",
            count=len(trades),
            output=str(output_path),
        )
        
        return output_path
    
    def count_trades(
        self,
        date_from: date,
        date_to: date,
        source: DataSource | str | None = None,
    ) -> int:
        """Count trades in date range."""
        count = 0
        for _ in self.get_trades(date_from, date_to, source=source):
            count += 1
        return count
    
    def get_available_dates(self, source: DataSource | None = None) -> list[date]:
        """Get list of dates with events."""
        dates = set()
        
        sources = [source] if source else list(DataSource)
        
        for src in sources:
            source_dir = self.base_path / src.value
            if source_dir.exists():
                for file_path in source_dir.glob("trades_*.jsonl"):
                    try:
                        date_str = file_path.stem.replace("trades_", "")
                        dates.add(date.fromisoformat(date_str))
                    except ValueError:
                        continue
        
        return sorted(dates)
