"""
Stage6 Trading Strategy Configuration

Goals: WR ≥55%, R:R ≥1.8, Daily DD ≥ -1%
Mode: Autonomous, without AGI-Brain
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal


@dataclass
class EntryConfig:
    """Entry signal configuration."""
    
    # === TREND ===
    ema_period: int = 200
    ema_long_threshold: float = 1.004   # close > EMA200 * 1.004 for LONG
    ema_short_threshold: float = 0.996  # close < EMA200 * 0.996 for SHORT
    ema_dead_zone_pct: float = 0.4      # ±0.4% dead zone
    require_ema_slope: bool = False     # TODO: Enable when 1h EMA slope data available
    
    # === MOMENTUM ===
    rsi_period_15m: int = 14
    rsi_oversold: float = 32.0          # RSI(15m) ≤ 32 for LONG
    rsi_overbought: float = 68.0        # RSI(15m) ≥ 68 for SHORT
    rsi_reversal_candles: int = 2       # RSI(5m) reversal check
    breakout_candles: int = 5           # Breakout of 5 candles high/low
    
    # === VOLUME ===
    volume_ma_period: int = 20
    volume_surge_multiplier: float = 1.5  # volume_5m ≥ 1.5 × SMA20
    max_spread_pct: float = 0.05        # Spread < 0.05%
    
    # === VOLATILITY ===
    atr_period: int = 14
    atr_min_pct: float = 0.5            # 0.5% ≤ ATR/Price
    atr_max_pct: float = 3.0            # ATR/Price ≤ 3%
    atr_rising_candles: int = 4         # ATR rising for last 4 candles
    
    # === COIN FILTER ===
    min_volume_24h_usd: float = 8_000_000  # 24h vol ≥ 8M
    use_top_100: bool = True            # Top-100 coins only
    ban_days_low_wr: int = 7            # Ban coin for 7 days if WR < 45%
    ban_min_trades: int = 30            # Min trades before banning
    ban_wr_threshold: float = 0.45      # WR threshold for ban


@dataclass 
class TradeManagementConfig:
    """Trade management configuration."""
    
    # === STOP LOSS ===
    sl_atr_multiplier: float = 1.0      # SL = 1 × ATR(15m)
    sl_max_pct: float = 1.2             # Max SL = 1.2% of price
    sl_allow_widening: bool = False     # Never widen SL
    
    # === TAKE PROFIT ===
    tp_rr_ratio: float = 2.1            # TP = 2.1 × SL (R:R = 1:2.1)
    
    # === BREAKEVEN ===
    be_trigger_r: float = 1.0           # At +1R, move SL to BE
    be_offset_r: float = 0.15           # BE + 0.15R (lock small profit)
    
    # === PARTIAL CLOSE ===
    partial_trigger_r: float = 1.5      # At +1.5R, partial close
    partial_close_pct: float = 0.5      # Close 50%
    
    # === TRAILING STOP ===
    trailing_trigger_r: float = 1.5     # Start trailing at +1.5R
    trailing_distance_r: float = 1.0    # Trail by 1R
    
    # === TIMEOUT ===
    timeout_candles_5m: int = 40        # 40 × 5m = ~3.3 hours
    timeout_close_market: bool = True   # Close at market on timeout


@dataclass
class RiskConfig:
    """Risk management configuration."""
    
    # === POSITION SIZING ===
    risk_per_trade_pct_min: float = 0.8   # 0.8% risk minimum
    risk_per_trade_pct_max: float = 1.0   # 1.0% risk maximum
    risk_per_trade_pct: float = 0.9       # Default 0.9%
    
    # === POSITION LIMITS ===
    max_concurrent_positions: int = 3     # ≤ 3 positions
    max_trades_per_day: int = 12          # ≤ 12 trades/day
    
    # === DAILY STOP ===
    daily_loss_limit_pct: float = -1.0    # PnL ≤ -1% → stop trading
    daily_reset_hour_utc: int = 0         # Reset at 00:00 UTC
    
    # === GOALS ===
    target_win_rate: float = 0.55         # WR ≥ 55%
    target_rr_ratio: float = 1.8          # R:R ≥ 1.8
    
    # === NO AGI-BRAIN RULES ===
    use_loss_streak_pause: bool = False   # NO "3 losses → 4h pause"
    use_agi_brain: bool = False           # Autonomous mode


@dataclass
class Stage6Config:
    """Complete Stage6 configuration."""
    
    entry: EntryConfig
    trade_management: TradeManagementConfig
    risk: RiskConfig
    
    # === TIMEFRAMES ===
    primary_timeframe: str = "5m"
    higher_timeframe: str = "15m"
    
    # === EXCHANGE ===
    exchange: str = "binance"
    market_type: str = "futures"
    margin_mode: str = "isolated"
    
    # Whitelist (specific coins to include beyond top-100)
    whitelist: list[str] = None
    
    # Blacklist (specific coins to exclude)
    blacklist: list[str] = None
    
    def __post_init__(self):
        if self.whitelist is None:
            self.whitelist = []
        if self.blacklist is None:
            self.blacklist = []


# === DEFAULT CONFIGURATION ===

DEFAULT_CONFIG = Stage6Config(
    entry=EntryConfig(),
    trade_management=TradeManagementConfig(),
    risk=RiskConfig(),
    whitelist=[
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    ],
    blacklist=[],
)


def get_config() -> Stage6Config:
    """Get the default Stage6 configuration."""
    return DEFAULT_CONFIG


def validate_entry_signal(config: EntryConfig, candle_data: dict, side: Literal["long", "short"]) -> dict:
    """
    Validate entry signal against all conditions.
    
    Returns:
        dict with 'valid' (bool), 'reasons' (list of passed checks), 'failures' (list of failed checks)
    """
    reasons = []
    failures = []
    
    close = candle_data.get("close", 0)
    ema200 = candle_data.get("ema200", 0)
    rsi_15m = candle_data.get("rsi_15m", 50)
    rsi_5m = candle_data.get("rsi_5m", 50)
    rsi_5m_prev = candle_data.get("rsi_5m_prev", 50)
    volume = candle_data.get("volume", 0)
    volume_sma20 = candle_data.get("volume_sma20", 0)
    spread_pct = candle_data.get("spread_pct", 0)
    atr_pct = candle_data.get("atr_pct", 0)
    atr_rising = candle_data.get("atr_rising", False)
    ema_slope_positive = candle_data.get("ema_slope_positive", None)
    breakout_high = candle_data.get("breakout_high", False)
    breakout_low = candle_data.get("breakout_low", False)
    
    # === TREND ===
    if side == "long":
        if close > ema200 * config.ema_long_threshold:
            reasons.append(f"Trend: close > EMA200*{config.ema_long_threshold}")
        else:
            failures.append(f"Trend: close {close:.2f} not > EMA200*{config.ema_long_threshold} ({ema200 * config.ema_long_threshold:.2f})")
        
        if config.require_ema_slope and ema_slope_positive:
            reasons.append("EMA slope positive")
        elif config.require_ema_slope and not ema_slope_positive:
            failures.append("EMA slope not positive")
    else:  # short
        if close < ema200 * config.ema_short_threshold:
            reasons.append(f"Trend: close < EMA200*{config.ema_short_threshold}")
        else:
            failures.append(f"Trend: close {close:.2f} not < EMA200*{config.ema_short_threshold} ({ema200 * config.ema_short_threshold:.2f})")
        
        if config.require_ema_slope and not ema_slope_positive:
            reasons.append("EMA slope negative")
        elif config.require_ema_slope and ema_slope_positive:
            failures.append("EMA slope not negative")
    
    # === MOMENTUM ===
    if side == "long":
        if rsi_15m <= config.rsi_oversold:
            reasons.append(f"RSI(15m) {rsi_15m:.1f} ≤ {config.rsi_oversold}")
        else:
            failures.append(f"RSI(15m) {rsi_15m:.1f} not ≤ {config.rsi_oversold}")
        
        if rsi_5m > rsi_5m_prev:
            reasons.append("RSI(5m) reversal up")
        else:
            failures.append("No RSI(5m) reversal up")
        
        if breakout_high:
            reasons.append("Breakout above 5-candle high")
        else:
            failures.append("No breakout above 5-candle high")
    else:  # short
        if rsi_15m >= config.rsi_overbought:
            reasons.append(f"RSI(15m) {rsi_15m:.1f} ≥ {config.rsi_overbought}")
        else:
            failures.append(f"RSI(15m) {rsi_15m:.1f} not ≥ {config.rsi_overbought}")
        
        if rsi_5m < rsi_5m_prev:
            reasons.append("RSI(5m) reversal down")
        else:
            failures.append("No RSI(5m) reversal down")
        
        if breakout_low:
            reasons.append("Breakout below 5-candle low")
        else:
            failures.append("No breakout below 5-candle low")
    
    # === VOLUME ===
    if volume >= volume_sma20 * config.volume_surge_multiplier:
        reasons.append(f"Volume surge {volume/volume_sma20:.1f}x ≥ {config.volume_surge_multiplier}x")
    else:
        failures.append(f"Volume {volume/volume_sma20:.1f}x < {config.volume_surge_multiplier}x")
    
    if spread_pct < config.max_spread_pct:
        reasons.append(f"Spread {spread_pct:.3f}% < {config.max_spread_pct}%")
    else:
        failures.append(f"Spread {spread_pct:.3f}% ≥ {config.max_spread_pct}%")
    
    # === VOLATILITY ===
    if config.atr_min_pct <= atr_pct <= config.atr_max_pct:
        reasons.append(f"ATR {atr_pct:.2f}% in [{config.atr_min_pct}, {config.atr_max_pct}]")
    else:
        failures.append(f"ATR {atr_pct:.2f}% not in [{config.atr_min_pct}, {config.atr_max_pct}]")
    
    if atr_rising:
        reasons.append("ATR rising")
    else:
        failures.append("ATR not rising")
    
    return {
        "valid": len(failures) == 0,
        "reasons": reasons,
        "failures": failures,
    }


def calculate_position_size(
    capital: Decimal,
    entry_price: Decimal,
    stop_loss_price: Decimal,
    config: RiskConfig,
) -> dict:
    """
    Calculate position size based on risk.
    
    Returns:
        dict with 'size_usd', 'size_contracts', 'risk_usd', 'risk_pct'
    """
    risk_pct = Decimal(str(config.risk_per_trade_pct)) / 100
    risk_usd = capital * risk_pct
    
    sl_distance_pct = abs(entry_price - stop_loss_price) / entry_price
    
    # Position size = Risk USD / SL distance %
    if sl_distance_pct > 0:
        position_size_usd = risk_usd / sl_distance_pct
    else:
        position_size_usd = Decimal("0")
    
    # Convert to contracts (simplified - assume 1 contract = 1 USD notional)
    size_contracts = position_size_usd / entry_price
    
    return {
        "size_usd": float(position_size_usd),
        "size_contracts": float(size_contracts),
        "risk_usd": float(risk_usd),
        "risk_pct": float(risk_pct * 100),
        "sl_distance_pct": float(sl_distance_pct * 100),
    }


def calculate_sl_tp(
    entry_price: float,
    atr: float,
    side: Literal["long", "short"],
    config: TradeManagementConfig,
) -> dict:
    """
    Calculate SL and TP levels.
    
    Returns:
        dict with 'sl', 'tp', 'sl_pct', 'tp_pct', 'rr_ratio'
    """
    # SL = min(1 × ATR, 1.2% of price)
    sl_from_atr = atr * config.sl_atr_multiplier
    sl_from_pct = entry_price * (config.sl_max_pct / 100)
    sl_distance = min(sl_from_atr, sl_from_pct)
    
    # TP = 2.1 × SL
    tp_distance = sl_distance * config.tp_rr_ratio
    
    if side == "long":
        sl = entry_price - sl_distance
        tp = entry_price + tp_distance
    else:
        sl = entry_price + sl_distance
        tp = entry_price - tp_distance
    
    sl_pct = (sl_distance / entry_price) * 100
    tp_pct = (tp_distance / entry_price) * 100
    
    return {
        "sl": sl,
        "tp": tp,
        "sl_pct": sl_pct,
        "tp_pct": tp_pct,
        "rr_ratio": config.tp_rr_ratio,
        "sl_from": "ATR" if sl_from_atr < sl_from_pct else "MAX_PCT",
    }
