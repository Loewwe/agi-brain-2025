"""
MarketLog - Market data storage for backtesting.

Storage format: Parquet (efficient, columnar)
Path structure: data/market/{symbol}/{timeframe}_{year}_{month}.parquet
"""

import asyncio
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from .models import MarketBar

logger = structlog.get_logger()


class MarketLog:
    """
    Market data storage for backtesting.
    
    Features:
    - Parquet storage (efficient columnar format)
    - Symbol-based partitioning
    - Month-based files for manageable sizes
    - Caching for repeated reads
    - Automatic fetch from exchange if missing
    
    Path structure:
        data/market/{SYMBOL}/{timeframe}_{YYYY}_{MM}.parquet
        
    Example:
        data/market/BTCUSDT/5m_2024_12.parquet
    """
    
    def __init__(
        self,
        base_path: Path | str,
        cache_enabled: bool = True,
    ):
        """
        Initialize MarketLog.
        
        Args:
            base_path: Base directory for market data storage
            cache_enabled: Whether to cache data in memory
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.cache_enabled = cache_enabled
        self._cache: dict[str, pd.DataFrame] = {}
        
        logger.info("market_log.initialized", base_path=str(self.base_path))
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol name for file paths."""
        # Remove slashes and colons for filesystem compatibility
        return symbol.replace("/", "").replace(":", "")
    
    def _get_file_path(self, symbol: str, timeframe: str, year: int, month: int) -> Path:
        """Get file path for a specific symbol, timeframe, and month."""
        normalized = self._normalize_symbol(symbol)
        symbol_dir = self.base_path / normalized
        symbol_dir.mkdir(exist_ok=True)
        return symbol_dir / f"{timeframe}_{year}_{month:02d}.parquet"
    
    def _get_cache_key(self, symbol: str, timeframe: str, year: int, month: int) -> str:
        """Get cache key."""
        return f"{symbol}_{timeframe}_{year}_{month:02d}"
    
    def store_bars(self, bars: list[MarketBar]) -> int:
        """
        Store OHLCV bars.
        
        Args:
            bars: List of market bars to store
            
        Returns:
            Number of bars stored
        """
        if not bars:
            return 0
        
        # Group by symbol, timeframe, year, month
        grouped: dict[tuple[str, str, int, int], list[dict]] = {}
        
        for bar in bars:
            key = (
                bar.symbol,
                bar.timeframe,
                bar.timestamp.year,
                bar.timestamp.month,
            )
            if key not in grouped:
                grouped[key] = []
            grouped[key].append({
                "timestamp": bar.timestamp,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            })
        
        total_stored = 0
        
        for (symbol, timeframe, year, month), rows in grouped.items():
            file_path = self._get_file_path(symbol, timeframe, year, month)
            
            # Create DataFrame from new data
            new_df = pd.DataFrame(rows)
            new_df.set_index("timestamp", inplace=True)
            new_df.sort_index(inplace=True)
            
            # Load existing data if file exists
            if file_path.exists():
                existing_df = pd.read_parquet(file_path)
                # Merge, keeping new data for duplicates
                combined = pd.concat([existing_df, new_df])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined.sort_index(inplace=True)
                df_to_save = combined
            else:
                df_to_save = new_df
            
            # Save to parquet
            df_to_save.to_parquet(file_path, engine="pyarrow")
            total_stored += len(rows)
            
            # Invalidate cache
            cache_key = self._get_cache_key(symbol, timeframe, year, month)
            self._cache.pop(cache_key, None)
        
        logger.info("market_log.stored", bars_count=total_stored)
        return total_stored
    
    def store_dataframe(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> int:
        """
        Store DataFrame with OHLCV data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., "5m")
            df: DataFrame with columns: open, high, low, close, volume
                Index should be datetime
                
        Returns:
            Number of rows stored
        """
        if df.empty:
            return 0
        
        # Ensure proper index type
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Group by year/month
        df["_year"] = df.index.year
        df["_month"] = df.index.month
        
        total_stored = 0
        
        for (year, month), group in df.groupby(["_year", "_month"]):
            file_path = self._get_file_path(symbol, timeframe, int(year), int(month))
            
            # Prepare data (drop helper columns)
            data = group.drop(columns=["_year", "_month"]).copy()
            data.sort_index(inplace=True)
            
            # Merge with existing if file exists
            if file_path.exists():
                existing = pd.read_parquet(file_path)
                combined = pd.concat([existing, data])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined.sort_index(inplace=True)
                data = combined
            
            # Save
            data.to_parquet(file_path, engine="pyarrow")
            total_stored += len(group)
            
            # Invalidate cache
            cache_key = self._get_cache_key(symbol, timeframe, int(year), int(month))
            self._cache.pop(cache_key, None)
        
        return total_stored
    
    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        date_from: date,
        date_to: date,
    ) -> pd.DataFrame:
        """
        Get bars for symbol and date range.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., "5m")
            date_from: Start date (inclusive)
            date_to: End date (inclusive)
            
        Returns:
            DataFrame with OHLCV data, indexed by timestamp
        """
        dfs = []
        
        # Iterate through months in range
        current = date(date_from.year, date_from.month, 1)
        end_month = date(date_to.year, date_to.month, 1)
        
        while current <= end_month:
            cache_key = self._get_cache_key(symbol, timeframe, current.year, current.month)
            
            # Check cache first
            if self.cache_enabled and cache_key in self._cache:
                df = self._cache[cache_key]
            else:
                file_path = self._get_file_path(symbol, timeframe, current.year, current.month)
                
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    if self.cache_enabled:
                        self._cache[cache_key] = df
                else:
                    df = pd.DataFrame()
            
            if not df.empty:
                dfs.append(df)
            
            # Move to next month
            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)
        
        if not dfs:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        
        # Combine all DataFrames
        result = pd.concat(dfs)
        result = result[~result.index.duplicated(keep='first')]
        result.sort_index(inplace=True)
        
        # Filter to exact date range
        start_dt = datetime.combine(date_from, datetime.min.time())
        end_dt = datetime.combine(date_to, datetime.max.time())
        result = result[(result.index >= start_dt) & (result.index <= end_dt)]
        
        return result
    
    async def ensure_data(
        self,
        symbols: list[str],
        timeframe: str,
        date_from: date,
        date_to: date,
        fetch_missing: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Ensure data exists for symbols and date range.
        Optionally fetches missing data from exchange.
        
        Args:
            symbols: List of trading symbols
            timeframe: Timeframe (e.g., "5m")
            date_from: Start date
            date_to: End date
            fetch_missing: Whether to fetch missing data from exchange
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        result = {}
        missing_symbols = []
        
        for symbol in symbols:
            df = self.get_bars(symbol, timeframe, date_from, date_to)
            
            # Check if data exists and covers the range
            is_missing = False
            if df.empty:
                is_missing = True
            else:
                # Check coverage (allow 1 day buffer)
                data_start = df.index.min().date()
                data_end = df.index.max().date()
                
                if data_start > date_from or data_end < date_to:
                    is_missing = True
                    logger.info(
                        "market_log.partial_data",
                        symbol=symbol,
                        requested_start=date_from,
                        requested_end=date_to,
                        found_start=data_start,
                        found_end=data_end,
                    )
            
            if is_missing and fetch_missing:
                missing_symbols.append(symbol)
            else:
                result[symbol] = df
        
        # Fetch missing data if requested
        if missing_symbols and fetch_missing:
            logger.info(
                "market_log.fetching_missing",
                symbols=missing_symbols,
                date_from=date_from.isoformat(),
                date_to=date_to.isoformat(),
            )
            
            fetched = await self._fetch_from_exchange(
                missing_symbols,
                timeframe,
                date_from,
                date_to,
            )
            
            for symbol, df in fetched.items():
                if not df.empty:
                    self.store_dataframe(symbol, timeframe, df)
                    result[symbol] = df
        
        return result
    
    async def _fetch_from_exchange(
        self,
        symbols: list[str],
        timeframe: str,
        date_from: date,
        date_to: date,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data from exchange.
        
        Uses ccxt for Binance Futures.
        """
        try:
            import ccxt.async_support as ccxt
        except ImportError:
            logger.warning("market_log.ccxt_not_available")
            return {}
        
        exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        
        result = {}
        days = (date_to - date_from).days + 5  # Extra buffer
        bars_needed = days * 288 if timeframe == "5m" else days * 24
        
        for symbol in symbols:
            try:
                logger.info("market_log.fetching", symbol=symbol)
                
                all_ohlcv = []
                since = int(datetime.combine(date_from, datetime.min.time()).timestamp() * 1000)
                batch_size = 1000
                
                while len(all_ohlcv) < bars_needed:
                    ohlcv = await exchange.fetch_ohlcv(
                        symbol, timeframe, since=since, limit=batch_size
                    )
                    
                    if not ohlcv:
                        break
                    
                    all_ohlcv.extend(ohlcv)
                    since = ohlcv[-1][0] + 1
                    
                    await asyncio.sleep(0.05)  # Rate limit
                    
                    if len(ohlcv) < batch_size:
                        break
                
                if all_ohlcv:
                    df = pd.DataFrame(
                        all_ohlcv,
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.set_index("timestamp", inplace=True)
                    df = df[~df.index.duplicated(keep='first')]
                    result[symbol] = df
                    
                    logger.info("market_log.fetched", symbol=symbol, bars=len(df))
                
            except Exception as e:
                logger.warning("market_log.fetch_error", symbol=symbol, error=str(e))
        
        await exchange.close()
        return result
    
    def get_available_symbols(self) -> list[str]:
        """Get list of symbols with stored data."""
        symbols = []
        for path in self.base_path.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                symbols.append(path.name)
        return sorted(symbols)
    
    def get_data_range(self, symbol: str, timeframe: str) -> tuple[date | None, date | None]:
        """Get date range of available data for a symbol."""
        normalized = self._normalize_symbol(symbol)
        symbol_dir = self.base_path / normalized
        
        if not symbol_dir.exists():
            return None, None
        
        dates = []
        for file_path in symbol_dir.glob(f"{timeframe}_*.parquet"):
            try:
                parts = file_path.stem.split("_")
                if len(parts) >= 3:
                    year = int(parts[1])
                    month = int(parts[2])
                    dates.append(date(year, month, 1))
            except (ValueError, IndexError):
                continue
        
        if not dates:
            return None, None
        
        min_date = min(dates)
        max_date = max(dates)
        # Approximate end of last month
        if max_date.month == 12:
            max_date = date(max_date.year + 1, 1, 1) - timedelta(days=1)
        else:
            max_date = date(max_date.year, max_date.month + 1, 1) - timedelta(days=1)
        
        return min_date, max_date
    
    def clear_cache(self) -> None:
        """Clear in-memory cache."""
        self._cache.clear()
        logger.info("market_log.cache_cleared")
