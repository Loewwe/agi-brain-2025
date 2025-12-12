
import logging
import pickle
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import ccxt
import time
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "ADA/USDT",
    "DOT/USDT",
]
TIMEFRAMES = ["5m", "15m", "1h", "4h"]
START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 12, 12, 23, 59, 59, tzinfo=timezone.utc)

CACHE_DIR = Path(__file__).parent.parent / "data" / "backtest_cache" / "mass_screening_2025"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def fetch_data():
    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            logger.info(f"Fetching {symbol} {tf} for 2025...")
            
            cache_file = CACHE_DIR / f"{symbol.replace('/', '_')}_{tf}_2025.pkl"
            if cache_file.exists():
                logger.info(f"  Cache exists: {cache_file}")
                continue
                
            all_ohlcv = []
            since = int(START_DATE.timestamp() * 1000)
            end_ts = int(END_DATE.timestamp() * 1000)
            
            while True:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, tf, since=since, limit=1000)
                except Exception as e:
                    logger.warning(f"  Fetch error: {e}, retrying...")
                    time.sleep(5)
                    continue
                
                if not ohlcv:
                    break
                
                # Filter out candles beyond end date
                ohlcv = [c for c in ohlcv if c[0] <= end_ts]
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                
                if len(ohlcv) < 1000:
                    break
                
                # Polite delay
                time.sleep(0.2)
                
            if not all_ohlcv:
                logger.warning(f"  ⚠️ No data for {symbol} {tf}")
                continue
                
            df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            
            # Ensure unique index
            df = df[~df.index.duplicated(keep='first')]
            
            # Save cache
            with open(cache_file, "wb") as f:
                pickle.dump(df, f)
            logger.info(f"  ✅ Saved {len(df)} candles to {cache_file}")

if __name__ == "__main__":
    fetch_data()
