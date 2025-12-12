
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

# Spot Symbols (Binance Spot usually doesn't have :USDT suffix in ccxt, just BTC/USDT)
SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "APT/USDT",
    "OP/USDT",
    "NEAR/USDT",
    "DOT/USDT",
]

def fetch_spot_data():
    # Match Stage 1 Train Period: 90 days
    days = 90
    timeframe = "15m"
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    cache_dir = Path(__file__).parent.parent / "data" / "backtest_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"ohlcv_spot_{days}d_{timeframe}.pkl"
    
    if cache_file.exists():
        logger.info(f"Cache exists: {cache_file}")
        # return 
    
    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
    all_data = {}
    
    try:
        for i, symbol in enumerate(SYMBOLS):
            logger.info(f"[{i+1}/{len(SYMBOLS)}] Fetching Spot {symbol}...")
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
                
                # Safety break
                if len(all_ohlcv) > 50000: 
                    break
                
                # Polite delay
                time.sleep(0.5)
            
            if not all_ohlcv:
                logger.warning(f"  ⚠️ No data for {symbol}")
                continue
                
            df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            
            all_data[symbol] = df
            logger.info(f"  ✅ Success: {len(df)} candles")
            
        # Save cache
        with open(cache_file, "wb") as f:
            pickle.dump(all_data, f)
        logger.info(f"Saved Spot cache to {cache_file}")
            
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
    finally:
        pass

if __name__ == "__main__":
    fetch_spot_data()
