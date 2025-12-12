import asyncio
import os
import ccxt.async_support as ccxt
from typing import List, Dict

async def apply_stop_loss(exchange, symbol: str, side: str, quantity: float, stop_price: float):
    """
    Places a Stop-Loss order.
    For LONG positions, this is a SELL STOP order.
    For SHORT positions, this is a BUY STOP order.
    """
    try:
        # Determine order side for SL
        sl_side = 'sell' if side == 'long' else 'buy'
        
        print(f"🛡️ Placing SL for {symbol}: {sl_side.upper()} {quantity} @ {stop_price}")
        
        # Bybit specific params for Position SL
        params = {
            'stopLoss': str(stop_price),
            'positionIdx': 0, # One-way mode usually
        }
        
        # In many exchanges, setting SL is a position update, not a separate order
        # But CCXT unifies this via create_order or set_position_mode
        # For Bybit, we can often use set_trading_stop
        
        if exchange.id == 'bybit':
            await exchange.set_trading_stop(
                symbol=symbol,
                side=sl_side,
                stop_loss=stop_price,
                position_idx=0
            )
            print(f"✅ SL set for {symbol} at {stop_price}")
        else:
            # Generic fallback (might create a separate trigger order)
            order = await exchange.create_order(
                symbol=symbol,
                type='stop_market',
                side=sl_side,
                amount=quantity,
                price=None,
                params={'stopPrice': stop_price}
            )
            print(f"✅ SL Order placed for {symbol}: {order['id']}")
            
    except Exception as e:
        print(f"❌ Failed to set SL for {symbol}: {str(e)}")

async def main():
    api_key = os.environ.get("EXCHANGE_API_KEY")
    secret = os.environ.get("EXCHANGE_SECRET")
    
    if not api_key or not secret:
        print("❌ Error: EXCHANGE_API_KEY and EXCHANGE_SECRET must be set.")
        return

    # Initialize Exchange (Switching to Binance based on key length)
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': secret,
        'options': {
            'defaultType': 'future',  # Binance Futures
        }
    })
    
    # Positions to protect (from Analysis)
    # XRP: Entry 2.1308 -> SL ~2.077 (-2.5%)
    # DOGE: Entry 0.14902 -> SL ~0.1453 (-2.5%)
    # SOL: Entry 141.91 -> SL ~138.36 (-2.5%)
    # AVAX: Entry 14.708 -> SL ~14.34 (-2.5%)
    
    targets = [
        {'symbol': 'XRP/USDT:USDT', 'side': 'long', 'qty': 11.3, 'sl': 2.077},
        {'symbol': 'DOGE/USDT:USDT', 'side': 'long', 'qty': 231, 'sl': 0.1453},
        {'symbol': 'SOL/USDT:USDT', 'side': 'long', 'qty': 0.24, 'sl': 138.36},
        {'symbol': 'AVAX/USDT:USDT', 'side': 'long', 'qty': 4, 'sl': 14.34},
    ]
    
    try:
        print("🔌 Connecting to exchange...")
        await exchange.load_markets()
        print("✅ Connected.")
        
        for target in targets:
            await apply_stop_loss(
                exchange, 
                target['symbol'], 
                target['side'], 
                target['qty'], 
                target['sl']
            )
            
    except Exception as e:
        print(f"❌ Critical Error: {str(e)}")
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())
