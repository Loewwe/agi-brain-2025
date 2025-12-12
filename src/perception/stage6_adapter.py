import os
import ccxt.async_support as ccxt
from datetime import datetime
from decimal import Decimal
from typing import Any

from ..world.model import (
    TradingAdapterProtocol,
    Position,
    PositionSide,
    Trade,
    Strategy,
    RiskConfig,
    PnLMetrics,
    RiskViolation,
)


class Stage6TradingAdapter(TradingAdapterProtocol):
    """
    Адаптер для реального торгового бота (Stage 6) через CCXT.
    """
    
    def __init__(self, api_url: str | None = None, api_key: str | None = None):
        # api_url unused for direct exchange connection
        self.api_key = api_key or os.environ.get("EXCHANGE_API_KEY")
        self.api_secret = os.environ.get("EXCHANGE_SECRET")
        
        if not self.api_key or not self.api_secret:
            # Fallback for initialization, but methods will fail
            pass
            
    async def _get_exchange(self):
        """Initialize exchange connection."""
        # Determine exchange based on key length or config
        # Defaulting to Binance for now as per user context
        return ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'options': {
                'defaultType': 'future',
            }
        })

    async def fetch_history(self, days: int = 7, symbol: str | None = None) -> list[Trade]:
        """Получить историю сделок."""
        # TODO: Implement fetch_my_trades
        return []

    async def fetch_ohlcv(self, symbol: str, timeframe: str = '15m', limit: int = 100) -> list[dict]:
        """Fetch OHLCV data."""
        exchange = await self._get_exchange()
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return [
                {
                    'timestamp': datetime.fromtimestamp(c[0]/1000),
                    'open': c[1],
                    'high': c[2],
                    'low': c[3],
                    'close': c[4],
                    'volume': c[5]
                }
                for c in ohlcv
            ]
        except Exception as e:
            print(f"Error fetching OHLCV for {symbol}: {e}")
            return []
        finally:
            await exchange.close()

    async def fetch_positions(self) -> list[Position]:
        """Получить открытые позиции через CCXT."""
        exchange = await self._get_exchange()
        try:
            # Load markets first
            await exchange.load_markets()
            
            # Fetch positions
            # Note: fetch_positions behavior varies by exchange
            raw_positions = await exchange.fetch_positions()
            
            positions = []
            for p in raw_positions:
                size = float(p['contracts']) if p.get('contracts') else float(p['info'].get('positionAmt', 0))
                if size == 0:
                    continue
                    
                symbol = p['symbol']
                side = PositionSide.LONG if p['side'] == 'long' else PositionSide.SHORT
                entry_price = float(p['entryPrice'])
                mark_price = float(p.get('markPrice', 0))
                leverage = int(p.get('leverage', 1))
                unrealized_pnl = float(p.get('unrealizedPnl', 0))
                
                # Calculate PnL %
                initial_margin = (size * entry_price) / leverage
                pnl_percent = (unrealized_pnl / initial_margin * 100) if initial_margin else 0
                
                positions.append(Position(
                    symbol=symbol,
                    side=side,
                    size=Decimal(str(abs(size))),
                    entry_price=Decimal(str(entry_price)),
                    current_price=Decimal(str(mark_price)),
                    leverage=leverage,
                    unrealized_pnl=Decimal(str(unrealized_pnl)),
                    unrealized_pnl_percent=pnl_percent,
                    liquidation_price=Decimal(str(p.get('liquidationPrice', 0) or 0)),
                    stop_loss=Decimal(str(p.get('stopLossPrice', 0) or 0)) if p.get('stopLossPrice') else None,
                    take_profit=Decimal(str(p.get('takeProfitPrice', 0) or 0)) if p.get('takeProfitPrice') else None,
                    opened_at=datetime.now(), # CCXT often doesn't give open time for consolidated position
                ))
            return positions
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []
        finally:
            await exchange.close()

    async def fetch_active_strategies(self) -> list[Strategy]:
        return []

    async def fetch_risk_config(self) -> RiskConfig:
        return RiskConfig(
            max_position_size_usd=Decimal("1000"),
            max_leverage=10,
            max_daily_loss_usd=Decimal("100"),
            max_daily_loss_percent=5.0,
            unsinkable_balance_usd=Decimal("50"),
            max_concurrent_positions=5,
            stop_loss_percent=2.5,
            take_profit_percent=5.0,
        )

    async def get_pnl_metrics(self, period_days: int = 7) -> PnLMetrics:
        # Stub metrics
        return PnLMetrics(
            total_balance_usd=Decimal("0"),
            available_balance_usd=Decimal("0"),
            realized_pnl_today=Decimal("0"),
            realized_pnl_week=Decimal("0"),
            realized_pnl_month=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            pnl_today_percent=0.0,
            pnl_week_percent=0.0,
            pnl_month_percent=0.0,
            max_drawdown_today=0.0,
            max_drawdown_week=0.0,
            max_drawdown_month=0.0,
            win_rate=0.0,
            profit_factor=0.0,
        )

    async def check_risk_violations(self) -> list[RiskViolation]:
        return []

    async def update_risk_config(self, updates: dict[str, Any]) -> bool:
        return True

    async def emergency_stop(self, reason: str) -> bool:
        return True
        
    async def analyze_risk(self, include_recommendations: bool = True) -> Any:
        # Reuse logic from Stub/Fixture
        from .trading_adapter import RiskAnalysis
        positions = await self.fetch_positions()
        config = await self.fetch_risk_config()
        
        violations = []
        recommendations = []
        risk_score = 0.0
        
        # Check positions
        if len(positions) > config.max_concurrent_positions:
            violations.append({
                "rule": "max_positions",
                "current": len(positions),
                "limit": config.max_concurrent_positions,
            })
            risk_score += 20
        
        for pos in positions:
            if pos.leverage > config.max_leverage:
                violations.append({
                    "rule": "max_leverage",
                    "symbol": pos.symbol,
                    "current": pos.leverage,
                    "limit": config.max_leverage,
                })
                risk_score += 15
            
            dist = pos.distance_to_liquidation_percent
            if dist and dist < 10:
                violations.append({
                    "rule": "liquidation_warning",
                    "symbol": pos.symbol,
                    "distance": dist,
                })
                risk_score += 30
        
        if include_recommendations:
            if risk_score > 50:
                recommendations.append("Consider reducing position sizes")
            if any(p.stop_loss is None for p in positions):
                recommendations.append("Add stop losses to unprotected positions")
        
        return RiskAnalysis(
            risk_score=min(100, risk_score),
            violations=violations,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
        
    async def fetch_agent_state(self):
        # Minimal implementation
        from ..world.model import TradingAgent, AgentStatus
        return TradingAgent(
            id="stage6",
            name="Stage 6 Bot",
            status=AgentStatus.RUNNING,
            exchange="binance",
            strategies=[],
            current_positions=await self.fetch_positions(),
            pnl=await self.get_pnl_metrics(),
            risk_config=await self.fetch_risk_config(),
        )

    async def health_check(self) -> bool:
        return True

    async def set_stop_loss(self, symbol: str, price: float) -> bool:
        """Установить Stop-Loss для позиции."""
        exchange = await self._get_exchange()
        try:
            # Fetch position to know side
            positions = await exchange.fetch_positions([symbol])
            target_pos = next((p for p in positions if float(p['contracts']) > 0), None)
            
            if not target_pos:
                print(f"No position found for {symbol}")
                return False
                
            side = 'sell' if target_pos['side'] == 'long' else 'buy'
            
            # Binance Futures SL
            await exchange.create_order(
                symbol=symbol,
                type='STOP_MARKET',
                side=side,
                amount=float(target_pos['contracts']), # Close full position
                price=None,
                params={
                    'stopPrice': price,
                    'closePosition': True # Important for Binance
                }
            )
            return True
        except Exception as e:
            print(f"Error setting SL for {symbol}: {e}")
            return False
        finally:
            await exchange.close()
