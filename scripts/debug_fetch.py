import asyncio
import ccxt.async_support as ccxt
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_fetch():
    logger.info("Starting debug fetch...")
    exchange = ccxt.binance()
    try:
        symbol = "BTC/USDT:USDT"
        timeframe = "15m"
        since = int((datetime.now() - timedelta(days=10)).timestamp() * 1000)
        
        logger.info(f"Fetching {symbol} since {since}...")
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=100)
        
        logger.info(f"Success! Fetched {len(ohlcv)} bars.")
        if ohlcv:
            logger.info(f"First bar: {ohlcv[0]}")
            logger.info(f"Last bar: {ohlcv[-1]}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await exchange.close()
        logger.info("Exchange closed.")

if __name__ == "__main__":
    asyncio.run(test_fetch())
