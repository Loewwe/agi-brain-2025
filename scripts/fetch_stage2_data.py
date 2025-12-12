
import logging
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import ccxt
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Symbols for Stage 2
SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    # Optional Alts
    "APT/USDT",
    "OP/USDT",
    "DOT/USDT",
]

def fetch_data():
    # 3 Years = ~1095 days. Let's fetch 1100 days to be safe.
    days = 1100
    timeframe = "15m"
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    cache_dir = Path(__file__).parent.parent / "data" / "backtest_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"ohlcv_{days}d_{timeframe}.pkl"
    
    if cache_file.exists():
        logger.info(f"Cache exists: {cache_file}")
        # return # Force refetch if needed, but for now let's trust cache if it exists
    
    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    all_data = {}
    
    try:
        for i, symbol in enumerate(SYMBOLS):
            logger.info(f"[{i+1}/{len(SYMBOLS)}] Fetching {symbol}...")
            all_ohlcv = []
            current_since = since
            
            while True:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
                except Exception as e:
                    logger.warning(f"  Fetch error: {e}, retrying...")
                    time.sleep(5)
                    continue
                
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1
                if len(ohlcv) < 1000:
                    break
                
                # Safety break (1100 days * 96 candles/day = ~105k candles)
                if len(all_ohlcv) > 150000: 
                    break
                
                # Polite delay
                time.sleep(0.5)
            
            if not all_ohlcv:
                logger.warning(f"  ⚠️ No data for {symbol}")
                continue
                
            df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            
            # Ensure unique index
            df = df[~df.index.duplicated(keep='first')]
            
            all_data[symbol] = df
            logger.info(f"  ✅ Success: {len(df)} candles")
            
        # Save cache
        with open(cache_file, "wb") as f:
            pickle.dump(all_data, f)
        logger.info(f"Saved cache to {cache_file}")
            
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
    finally:
        pass

if __name__ == "__main__":
    fetch_data()
