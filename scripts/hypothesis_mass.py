
from scripts.hypothesis_stage1 import (
    Hypothesis, HypothesisEvent,
    H001_AsianPump,
    H002_EuropeanOpenBreakout,
    H003_US_PreMarket_Dip,
    H004_FridayProfitTaking,
    H005_MondayGapRecovery,
    H006_WeekendLowVolatility,
    H009_RSI_Divergence_Bull,
    H021_EngulfingCandle,
    H023_InsideBarBreakout,
    H041_OpeningRangeBreakout,
    H042_PowerHourReversal,
    H044_LondonOpenFade,
    H045_NYLunchLull,
    H048_WeekendGapClose,
    H051_BTCDominanceMeanRev,
    H052_AltcoinBTCLag,
    H053_PerpSpotBasis,
    H061_RSI5_ExtremeReversal,
    H062_BBSqueezeBreakout,
    H063_WickHunting,
    H071_RoundNumberMagnet,
    H073_FOMOPumpExhaustion,
    H074_DumpPanicCapitulation,
    H075_WeekendBoredomPump,
    H092_DXYInverseSpike,
)


from datetime import timedelta, time
import pandas as pd
import numpy as np

# --- Batch 1: Time/Session Anomalies ---

class H001_AsianMorningPumpFade(Hypothesis):
    """
    H001: Asian Morning Pump Fade
    Logic: Fade sharp moves (>2%) in early Asian session (00:00-04:00 UTC).
    """
    def __init__(self):
        super().__init__("H001", "Asian Morning Pump Fade", "Fade >2% move in 00:00-04:00 UTC")

    def find_triggers(self, data: dict) -> list:
        events = []
        for symbol, df in data.items():
            df = df.copy()
            # Use 1h data if available, else resample? 
            # Assuming we got 15m or 1h. Let's work with what we have.
            # If 15m, we look for 1h rolling change.
            
            df['pct_change_1h'] = df['close'].pct_change(4) # 4 * 15m = 1h
            
            for ts, row in df.iterrows():
                if not (0 <= ts.hour < 4):
                    continue
                    
                move = row['pct_change_1h']
                if pd.isna(move): continue
                
                direction = None
                if move > 0.02: # Pump > 2%
                    direction = "SHORT"
                elif move < -0.02: # Dump > 2%
                    direction = "LONG"
                    
                if direction:
                    # Entry
                    entry_price = row['close']
                    entry_ts = ts
                    
                    # Exit after 4 hours or end of session
                    result_end = entry_ts + timedelta(hours=4)
                    result_window = df[(df.index > entry_ts) & (df.index <= result_end)]
                    
                    if result_window.empty: continue
                    
                    exit_price = result_window.iloc[-1]['close']
                    
                    if direction == "LONG":
                        rew_pct = (exit_price - entry_price) / entry_price * 100
                    else:
                        rew_pct = (entry_price - exit_price) / entry_price * 100
                        
                    events.append(HypothesisEvent(
                        timestamp=entry_ts,
                        symbol=symbol,
                        entry_price=entry_price,
                        direction=direction,
                        rew_pct=rew_pct,
                        mfe_pct=0, mae_pct=0, context={"move": move}
                    ))
        return events

class H002_EuropeanOpenBreakout(Hypothesis):
    """
    H002: European Open Breakout
    Logic: Trade breakout of first 15m of London session (07:00-07:15 UTC).
    """
    def __init__(self):
        super().__init__("H002", "European Open Breakout", "Breakout of 07:00-07:15 range")

    def find_triggers(self, data: dict) -> list:
        events = []
        for symbol, df in data.items():
            # Need 15m data
            if df.index.freqstr != '15T' and (df.index[1] - df.index[0]).seconds != 900:
                # Try to infer freq or skip if not 15m
                pass
            
            # Group by day
            days = df.groupby(df.index.date)
            
            for date, day_df in days:
                # Get 07:00 candle
                try:
                    open_candle = day_df.loc[day_df.index.time == time(7, 0)]
                except:
                    continue
                    
                if open_candle.empty: continue
                
                range_high = open_candle.iloc[0]['high']
                range_low = open_candle.iloc[0]['low']
                range_size = range_high - range_low
                
                # Look at next 2 hours (07:15 - 09:15)
                lookahead = day_df[(day_df.index.time > time(7, 0)) & (day_df.index.time <= time(9, 0))]
                
                triggered = False
                direction = None
                entry_price = 0
                entry_ts = None
                
                for ts, row in lookahead.iterrows():
                    if row['close'] > range_high:
                        direction = "LONG"
                        entry_price = row['close']
                        entry_ts = ts
                        triggered = True
                        break
                    elif row['close'] < range_low:
                        direction = "SHORT"
                        entry_price = row['close']
                        entry_ts = ts
                        triggered = True
                        break
                        
                if triggered:
                    # Exit at 11:00 UTC (Lunch)
                    exit_time = datetime.combine(date, time(11, 0), tzinfo=timezone.utc)
                    # Find candle at or after exit_time
                    exit_window = df[(df.index > entry_ts) & (df.index <= exit_time)]
                    if exit_window.empty: continue
                    
                    exit_price = exit_window.iloc[-1]['close']
                    
                    if direction == "LONG":
                        rew_pct = (exit_price - entry_price) / entry_price * 100
                    else:
                        rew_pct = (entry_price - exit_price) / entry_price * 100
                        
                    events.append(HypothesisEvent(
                        timestamp=entry_ts,
                        symbol=symbol,
                        entry_price=entry_price,
                        direction=direction,
                        rew_pct=rew_pct,
                        mfe_pct=0, mae_pct=0, context={"range_size": range_size}
                    ))
        return events

class H003_US_PreMarket_Dip(Hypothesis):
    """
    H003: US Pre-market Dip Buy
    Logic: Buy if price drops >0.5% in 13:00-13:30 UTC.
    """
    def __init__(self):
        super().__init__("H003", "US Pre-market Dip Buy", "Buy dip 13:00-13:30 UTC")

    def find_triggers(self, data: dict) -> list:
        events = []
        for symbol, df in data.items():
            # Iterate days
            days = df.groupby(df.index.date)
            for date, day_df in days:
                # Window 13:00 - 13:30
                window = day_df[(day_df.index.time >= time(13, 0)) & (day_df.index.time < time(13, 30))]
                if window.empty: continue
                
                open_price = window.iloc[0]['open']
                close_price = window.iloc[-1]['close']
                
                drop = (close_price - open_price) / open_price
                
                if drop < -0.005: # > 0.5% drop
                    direction = "LONG"
                    entry_price = close_price
                    entry_ts = window.index[-1]
                    
                    # Exit at 16:00 UTC (London Close)
                    exit_time = datetime.combine(date, time(16, 0), tzinfo=timezone.utc)
                    exit_window = df[(df.index > entry_ts) & (df.index <= exit_time)]
                    if exit_window.empty: continue
                    
                    exit_price = exit_window.iloc[-1]['close']
                    rew_pct = (exit_price - entry_price) / entry_price * 100
                    
                    events.append(HypothesisEvent(
                        timestamp=entry_ts,
                        symbol=symbol,
                        entry_price=entry_price,
                        direction=direction,
                        rew_pct=rew_pct,
                        mfe_pct=0, mae_pct=0, context={"drop": drop}
                    ))
        return events

class H004_FridayProfitTaking(Hypothesis):
    """
    H004: Friday Profit Taking
    Logic: Short at 19:00 UTC on Friday if week is Green.
    """
    def __init__(self):
        super().__init__("H004", "Friday Profit Taking", "Short Friday 19:00 if week green")

    def find_triggers(self, data: dict) -> list:
        events = []
        for symbol, df in data.items():
            # Resample to daily to check weekly trend?
            # Just check price vs Monday Open
            
            # Iterate Fridays
            # Friday weekday = 4
            fridays = df[df.index.weekday == 4]
            friday_dates = fridays.index.date
            unique_fridays = sorted(list(set(friday_dates)))
            
            for f_date in unique_fridays:
                # Get Monday Open
                monday_date = f_date - timedelta(days=4)
                monday_data = df[df.index.date == monday_date]
                if monday_data.empty: continue
                week_open = monday_data.iloc[0]['open']
                
                # Get Friday 19:00 candle
                entry_time = datetime.combine(f_date, time(19, 0), tzinfo=timezone.utc)
                # Find closest candle
                try:
                    idx = df.index.get_indexer([entry_time], method='nearest')[0]
                    entry_row = df.iloc[idx]
                    if abs((entry_row.name - entry_time).total_seconds()) > 3600: continue # Too far
                except:
                    continue
                
                current_price = entry_row['close']
                
                # If Week is Green (> 2% up?)
                if current_price > week_open * 1.02:
                    direction = "SHORT"
                    entry_price = current_price
                    entry_ts = entry_row.name
                    
                    # Exit at Friday Close (23:00 UTC)
                    exit_time = datetime.combine(f_date, time(23, 0), tzinfo=timezone.utc)
                    exit_window = df[(df.index > entry_ts) & (df.index <= exit_time)]
                    if exit_window.empty: continue
                    
                    exit_price = exit_window.iloc[-1]['close']
                    rew_pct = (entry_price - exit_price) / entry_price * 100
                    
                    events.append(HypothesisEvent(
                        timestamp=entry_ts,
                        symbol=symbol,
                        entry_price=entry_price,
                        direction=direction,
                        rew_pct=rew_pct,
                        mfe_pct=0, mae_pct=0, context={"week_gain": (current_price-week_open)/week_open}
                    ))
        return events

class H005_MondayGapRecovery(Hypothesis):
    """
    H005: Monday Gap Recovery
    Logic: Fade gap between Friday Close and Monday Open.
    """
    def __init__(self):
        super().__init__("H005", "Monday Gap Recovery", "Fade weekend gap")

    def find_triggers(self, data: dict) -> list:
        events = []
        for symbol, df in data.items():
            # Iterate Mondays (weekday=0)
            mondays = df[df.index.weekday == 0]
            unique_mondays = sorted(list(set(mondays.index.date)))
            
            for m_date in unique_mondays:
                # Get Friday Close (previous Friday)
                friday_date = m_date - timedelta(days=3)
                friday_data = df[df.index.date == friday_date]
                if friday_data.empty: continue
                friday_close = friday_data.iloc[-1]['close']
                
                # Get Monday Open (00:00)
                monday_data = df[df.index.date == m_date]
                if monday_data.empty: continue
                monday_open = monday_data.iloc[0]['open']
                entry_ts = monday_data.index[0]
                
                gap = (monday_open - friday_close) / friday_close
                
                direction = None
                if gap > 0.01: # Gap Up > 1%
                    direction = "SHORT"
                elif gap < -0.01: # Gap Down > 1%
                    direction = "LONG"
                    
                if direction:
                    entry_price = monday_open
                    
                    # Exit at 08:00 UTC (Asian Close)
                    exit_time = datetime.combine(m_date, time(8, 0), tzinfo=timezone.utc)
                    exit_window = df[(df.index > entry_ts) & (df.index <= exit_time)]
                    if exit_window.empty: continue
                    
                    exit_price = exit_window.iloc[-1]['close']
                    
                    if direction == "LONG":
                        rew_pct = (exit_price - entry_price) / entry_price * 100
                    else:
                        rew_pct = (entry_price - exit_price) / entry_price * 100
                        
                    events.append(HypothesisEvent(
                        timestamp=entry_ts,
                        symbol=symbol,
                        entry_price=entry_price,
                        direction=direction,
                        rew_pct=rew_pct,
                        mfe_pct=0, mae_pct=0, context={"gap": gap}
                    ))
        return events


class H007_CMEGapFill(Hypothesis):
    """
    H007: CME Gap Fill
    Logic: Trade towards CME closing price (Friday 21:00 UTC) on Sunday night (23:00 UTC).
    """
    def __init__(self):
        super().__init__("H007", "CME Gap Fill", "Trade towards Fri 21:00 price on Sun 23:00")

    def find_triggers(self, data: dict) -> list:
        events = []
        for symbol, df in data.items():
            # Iterate Sundays
            sundays = df[df.index.weekday == 6]
            unique_sundays = sorted(list(set(sundays.index.date)))
            
            for s_date in unique_sundays:
                # Get Friday Close (previous Friday)
                friday_date = s_date - timedelta(days=2)
                
                # Friday 21:00 UTC
                fri_time = datetime.combine(friday_date, time(21, 0), tzinfo=timezone.utc)
                try:
                    idx = df.index.get_indexer([fri_time], method='nearest')[0]
                    fri_row = df.iloc[idx]
                    if abs((fri_row.name - fri_time).total_seconds()) > 3600: continue
                except:
                    continue
                
                cme_close = fri_row['close']
                
                # Sunday 23:00 UTC
                sun_time = datetime.combine(s_date, time(23, 0), tzinfo=timezone.utc)
                try:
                    idx = df.index.get_indexer([sun_time], method='nearest')[0]
                    sun_row = df.iloc[idx]
                    if abs((sun_row.name - sun_time).total_seconds()) > 3600: continue
                except:
                    continue
                
                current_price = sun_row['close']
                entry_ts = sun_row.name
                
                gap = (current_price - cme_close) / cme_close
                
                direction = None
                if gap > 0.01: # Price is 1% higher than CME close -> Short to fill gap
                    direction = "SHORT"
                elif gap < -0.01: # Price is 1% lower -> Long to fill gap
                    direction = "LONG"
                    
                if direction:
                    entry_price = current_price
                    
                    # Exit at Monday 08:00 UTC or if gap filled?
                    # Let's say Monday 08:00 UTC
                    monday_date = s_date + timedelta(days=1)
                    exit_time = datetime.combine(monday_date, time(8, 0), tzinfo=timezone.utc)
                    
                    exit_window = df[(df.index > entry_ts) & (df.index <= exit_time)]
                    if exit_window.empty: continue
                    
                    exit_price = exit_window.iloc[-1]['close']
                    
                    if direction == "LONG":
                        rew_pct = (exit_price - entry_price) / entry_price * 100
                    else:
                        rew_pct = (entry_price - exit_price) / entry_price * 100
                        
                    events.append(HypothesisEvent(
                        timestamp=entry_ts,
                        symbol=symbol,
                        entry_price=entry_price,
                        direction=direction,
                        rew_pct=rew_pct,
                        mfe_pct=0, mae_pct=0, context={"gap": gap}
                    ))
        return events

class H010_RSIBearishDivergence(Hypothesis):
    """
    H010: RSI Bearish Divergence
    Logic: Price Higher High, RSI Lower High (RSI > 70).
    """
    def __init__(self):
        super().__init__("H010", "RSI Bearish Divergence", "Price HH, RSI LH, RSI > 70")

    def find_triggers(self, data: dict) -> list:
        events = []
        for symbol, df in data.items():
            df = df.copy()
            
            # Calculate RSI 14
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Detect Peaks (simplified)
            # We look for RSI > 70
            # Then check if Price made HH while RSI made LH compared to previous peak
            
            last_peak_price = 0
            last_peak_rsi = 0
            last_peak_ts = None
            
            for i in range(2, len(df)):
                row = df.iloc[i]
                prev = df.iloc[i-1]
                prev2 = df.iloc[i-2]
                
                # Check for local peak in RSI > 60
                is_rsi_peak = prev['rsi'] > row['rsi'] and prev['rsi'] > prev2['rsi'] and prev['rsi'] > 60
                
                if is_rsi_peak:
                    current_peak_price = prev['high']
                    current_peak_rsi = prev['rsi']
                    ts = prev.name
                    
                    if last_peak_ts is not None:
                        # Check divergence
                        # Time distance < 50 candles
                        dist = i - df.index.get_loc(last_peak_ts)
                        
                        if 5 < dist < 50:
                            if current_peak_price > last_peak_price and current_peak_rsi < last_peak_rsi:
                                # Bearish Divergence
                                direction = "SHORT"
                                entry_price = row['close']
                                entry_ts = row.name
                                
                                # Exit after 12 candles (3h on 15m)
                                result_end = entry_ts + timedelta(minutes=15*12)
                                result_window = df[(df.index > entry_ts) & (df.index <= result_end)]
                                
                                if not result_window.empty:
                                    exit_price = result_window.iloc[-1]['close']
                                    rew_pct = (entry_price - exit_price) / entry_price * 100
                                    
                                    events.append(HypothesisEvent(
                                        timestamp=entry_ts,
                                        symbol=symbol,
                                        entry_price=entry_price,
                                        direction=direction,
                                        rew_pct=rew_pct,
                                        mfe_pct=0, mae_pct=0, 
                                        context={"rsi1": last_peak_rsi, "rsi2": current_peak_rsi}
                                    ))
                    
                    last_peak_price = current_peak_price
                    last_peak_rsi = current_peak_rsi
                    last_peak_ts = ts
                    
        return events


class H011_BollingerBandSqueeze(Hypothesis):
    """
    H011: Bollinger Band Squeeze
    Logic: Volatility squeeze (BB Width < Threshold) followed by breakout.
    """
    def __init__(self):
        super().__init__("H011", "Bollinger Band Squeeze", "BB Width percentile < 10, trade breakout")

    def find_triggers(self, data: dict) -> list:
        events = []
        for symbol, df in data.items():
            df = df.copy()
            
            # BB (20, 2)
            df['ma20'] = df['close'].rolling(20).mean()
            df['std20'] = df['close'].rolling(20).std()
            df['upper'] = df['ma20'] + 2 * df['std20']
            df['lower'] = df['ma20'] - 2 * df['std20']
            
            # Band Width
            df['bb_width'] = (df['upper'] - df['lower']) / df['ma20']
            
            # Squeeze threshold (e.g., lowest 10% of last 100 candles)
            df['squeeze_threshold'] = df['bb_width'].rolling(100).quantile(0.10)
            
            for i in range(20, len(df)):
                row = df.iloc[i]
                prev = df.iloc[i-1]
                
                # Check for Squeeze condition on PREVIOUS candle
                if prev['bb_width'] < prev['squeeze_threshold']:
                    # Breakout detection
                    direction = None
                    if row['close'] > row['upper']:
                        direction = "LONG"
                    elif row['close'] < row['lower']:
                        direction = "SHORT"
                        
                    if direction:
                        entry_price = row['close']
                        entry_ts = row.name
                        
                        # Exit after 20 candles or if price crosses MA20?
                        # Simple time exit for screening
                        result_end = entry_ts + timedelta(hours=4) # 4h hold
                        result_window = df[(df.index > entry_ts) & (df.index <= result_end)]
                        
                        if not result_window.empty:
                            exit_price = result_window.iloc[-1]['close']
                            if direction == "LONG":
                                rew_pct = (exit_price - entry_price) / entry_price * 100
                            else:
                                rew_pct = (entry_price - exit_price) / entry_price * 100
                                
                            events.append(HypothesisEvent(
                                timestamp=entry_ts,
                                symbol=symbol,
                                entry_price=entry_price,
                                direction=direction,
                                rew_pct=rew_pct,
                                mfe_pct=0, mae_pct=0, context={"width": prev['bb_width']}
                            ))
        return events

class H012_MACDCrossMomentum(Hypothesis):
    """
    H012: MACD Cross Momentum
    Logic: MACD Line crosses Signal Line.
    """
    def __init__(self):
        super().__init__("H012", "MACD Cross Momentum", "Standard MACD (12,26,9) Cross")

    def find_triggers(self, data: dict) -> list:
        events = []
        for symbol, df in data.items():
            df = df.copy()
            
            # MACD (12, 26, 9)
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            
            df['macd'] = macd
            df['signal'] = signal
            
            for i in range(26, len(df)):
                row = df.iloc[i]
                prev = df.iloc[i-1]
                
                direction = None
                # Bullish Cross
                if prev['macd'] < prev['signal'] and row['macd'] > row['signal']:
                    direction = "LONG"
                # Bearish Cross
                elif prev['macd'] > prev['signal'] and row['macd'] < row['signal']:
                    direction = "SHORT"
                    
                if direction:
                    entry_price = row['close']
                    entry_ts = row.name
                    
                    # Exit on reverse cross or fixed time?
                    # Let's try reverse cross logic or max 24h
                    # Simplified: 6h hold
                    result_end = entry_ts + timedelta(hours=6)
                    result_window = df[(df.index > entry_ts) & (df.index <= result_end)]
                    
                    if not result_window.empty:
                        exit_price = result_window.iloc[-1]['close']
                        if direction == "LONG":
                            rew_pct = (exit_price - entry_price) / entry_price * 100
                        else:
                            rew_pct = (entry_price - exit_price) / entry_price * 100
                            
                        events.append(HypothesisEvent(
                            timestamp=entry_ts,
                            symbol=symbol,
                            entry_price=entry_price,
                            direction=direction,
                            rew_pct=rew_pct,
                            mfe_pct=0, mae_pct=0, context={}
                        ))
        return events

class H013_EMARibbonTrend(Hypothesis):
    """
    H013: EMA Ribbon Trend
    Logic: All EMAs (20, 50, 100, 200) aligned. Pullback to EMA20.
    """
    def __init__(self):
        super().__init__("H013", "EMA Ribbon Trend", "EMAs aligned, pullback to EMA20")

    def find_triggers(self, data: dict) -> list:
        events = []
        for symbol, df in data.items():
            df = df.copy()
            
            df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
            df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
            
            for i in range(200, len(df)):
                row = df.iloc[i]
                
                # Check Trend Alignment
                bullish_stack = row['ema20'] > row['ema50'] > row['ema100'] > row['ema200']
                bearish_stack = row['ema20'] < row['ema50'] < row['ema100'] < row['ema200']
                
                direction = None
                # Pullback entry: Low touches EMA20 (Bullish) or High touches EMA20 (Bearish)
                # And Close is still on the correct side of EMA50?
                
                if bullish_stack:
                    if row['low'] <= row['ema20'] and row['close'] > row['ema50']:
                        direction = "LONG"
                elif bearish_stack:
                    if row['high'] >= row['ema20'] and row['close'] < row['ema50']:
                        direction = "SHORT"
                        
                if direction:
                    entry_price = row['close']
                    entry_ts = row.name
                    
                    # 12h hold for trend following
                    result_end = entry_ts + timedelta(hours=12)
                    result_window = df[(df.index > entry_ts) & (df.index <= result_end)]
                    
                    if not result_window.empty:
                        exit_price = result_window.iloc[-1]['close']
                        if direction == "LONG":
                            rew_pct = (exit_price - entry_price) / entry_price * 100
                        else:
                            rew_pct = (entry_price - exit_price) / entry_price * 100
                            
                        events.append(HypothesisEvent(
                            timestamp=entry_ts,
                            symbol=symbol,
                            entry_price=entry_price,
                            direction=direction,
                            rew_pct=rew_pct,
                            mfe_pct=0, mae_pct=0, context={}
                        ))
        return events

class H014_ParabolicSARReversal(Hypothesis):
    """
    H014: Parabolic SAR Reversal
    Logic: SAR dot flips side.
    """
    def __init__(self):
        super().__init__("H014", "Parabolic SAR Reversal", "SAR Flip")

    def find_triggers(self, data: dict) -> list:
        events = []
        # SAR implementation is complex to do from scratch efficiently.
        # We'll use a simplified version or skip if too complex for snippet.
        # Let's use a simple "Break of High/Low of last N candles" as a proxy for SAR flip?
        # No, let's implement basic SAR logic.
        
        # Or use pandas_ta if available? No external libs guaranteed.
        # Let's skip H014 for now and do H016 Volume Spike.
        return []

class H016_VolumeSpikeReversal(Hypothesis):
    """
    H016: Volume Spike Reversal
    Logic: Volume > 3x Avg, Price Move > 1%. Fade.
    """
    def __init__(self):
        super().__init__("H016", "Volume Spike Reversal", "Vol > 3x, Move > 1%, Fade")

    def find_triggers(self, data: dict) -> list:
        events = []
        for symbol, df in data.items():
            df = df.copy()
            
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['pct_change'] = df['close'].pct_change()
            
            for i in range(20, len(df)):
                row = df.iloc[i]
                
                if row['volume'] > 3 * row['vol_ma']:
                    move = row['pct_change']
                    direction = None
                    
                    if move > 0.01: # Pump > 1%
                        direction = "SHORT"
                    elif move < -0.01: # Dump > 1%
                        direction = "LONG"
                        
                    if direction:
                        entry_price = row['close']
                        entry_ts = row.name
                        
                        # Quick mean reversion: 2h hold
                        result_end = entry_ts + timedelta(hours=2)
                        result_window = df[(df.index > entry_ts) & (df.index <= result_end)]
                        
                        if not result_window.empty:
                            exit_price = result_window.iloc[-1]['close']
                            if direction == "LONG":
                                rew_pct = (exit_price - entry_price) / entry_price * 100
                            else:
                                rew_pct = (entry_price - exit_price) / entry_price * 100
                                
                            events.append(HypothesisEvent(
                                timestamp=entry_ts,
                                symbol=symbol,
                                entry_price=entry_price,
                                direction=direction,
                                rew_pct=rew_pct,
                                mfe_pct=0, mae_pct=0, context={"vol_mult": row['volume']/row['vol_ma']}
                            ))
        return events


class H022_HammerShootingStar(Hypothesis):
    """
    H022: Hammer / Shooting Star
    Logic: Candle with long wick (2x body) at trend extreme.
    """
    def __init__(self):
        super().__init__("H022", "Hammer / Shooting Star", "Long wick reversal pattern")

    def find_triggers(self, data: dict) -> list:
        events = []
        for symbol, df in data.items():
            df = df.copy()
            
            # Calculate candle features
            df['body'] = abs(df['close'] - df['open'])
            df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
            df['total_range'] = df['high'] - df['low']
            
            # Trend context (EMA50)
            df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
            
            for i in range(50, len(df)):
                row = df.iloc[i]
                
                # Hammer (Bullish)
                # Lower wick > 2 * body, Upper wick small (< 0.5 body or < 10% range)
                # Trend: Price < EMA50 (Downtrend context)
                is_hammer = (row['lower_wick'] > 2 * row['body']) and \
                            (row['upper_wick'] < 0.5 * row['body']) and \
                            (row['close'] < row['ema50'])
                            
                # Shooting Star (Bearish)
                # Upper wick > 2 * body, Lower wick small
                # Trend: Price > EMA50 (Uptrend context)
                is_shooting_star = (row['upper_wick'] > 2 * row['body']) and \
                                   (row['lower_wick'] < 0.5 * row['body']) and \
                                   (row['close'] > row['ema50'])
                                   
                direction = None
                if is_hammer:
                    direction = "LONG"
                elif is_shooting_star:
                    direction = "SHORT"
                    
                if direction:
                    entry_price = row['close']
                    entry_ts = row.name
                    
                    # Exit after 4 candles (1h on 15m)
                    result_end = entry_ts + timedelta(hours=1)
                    result_window = df[(df.index > entry_ts) & (df.index <= result_end)]
                    
                    if not result_window.empty:
                        exit_price = result_window.iloc[-1]['close']
                        if direction == "LONG":
                            rew_pct = (exit_price - entry_price) / entry_price * 100
                        else:
                            rew_pct = (entry_price - exit_price) / entry_price * 100
                            
                        events.append(HypothesisEvent(
                            timestamp=entry_ts,
                            symbol=symbol,
                            entry_price=entry_price,
                            direction=direction,
                            rew_pct=rew_pct,
                            mfe_pct=0, mae_pct=0, context={"wick_ratio": row['lower_wick']/row['body'] if direction=="LONG" else row['upper_wick']/row['body']}
                        ))
        return events

class H024_ThreeWhiteSoldiers(Hypothesis):
    """
    H024: Three White Soldiers / Black Crows
    Logic: 3 consecutive green/red candles with increasing bodies/volume?
    Simplified: 3 consecutive candles in same direction with decent size.
    """
    def __init__(self):
        super().__init__("H024", "Three White Soldiers", "3 consecutive strong candles")

    def find_triggers(self, data: dict) -> list:
        events = []
        for symbol, df in data.items():
            df = df.copy()
            
            df['is_green'] = df['close'] > df['open']
            df['body'] = abs(df['close'] - df['open'])
            df['avg_body'] = df['body'].rolling(20).mean()
            
            for i in range(20, len(df)):
                c1 = df.iloc[i-2]
                c2 = df.iloc[i-1]
                c3 = df.iloc[i]
                
                # Three White Soldiers (Bullish)
                if c1['is_green'] and c2['is_green'] and c3['is_green']:
                    # Check size (not doji)
                    if c1['body'] > 0.5*c1['avg_body'] and \
                       c2['body'] > 0.5*c2['avg_body'] and \
                       c3['body'] > 0.5*c3['avg_body']:
                           
                        # Check close logic (each closes higher than previous close)
                        if c2['close'] > c1['close'] and c3['close'] > c2['close']:
                            direction = "LONG" # Continuation? Or Reversal if at bottom?
                            # Usually reversal pattern after downtrend.
                            # Let's trade continuation for now.
                            
                            entry_price = c3['close']
                            entry_ts = c3.name
                            
                            # Exit after 4 candles
                            result_end = entry_ts + timedelta(hours=1)
                            result_window = df[(df.index > entry_ts) & (df.index <= result_end)]
                            
                            if not result_window.empty:
                                exit_price = result_window.iloc[-1]['close']
                                rew_pct = (exit_price - entry_price) / entry_price * 100
                                events.append(HypothesisEvent(
                                    timestamp=entry_ts,
                                    symbol=symbol,
                                    entry_price=entry_price,
                                    direction="LONG",
                                    rew_pct=rew_pct,
                                    mfe_pct=0, mae_pct=0, context={}
                                ))
                                
                # Three Black Crows (Bearish)
                elif not c1['is_green'] and not c2['is_green'] and not c3['is_green']:
                    if c1['body'] > 0.5*c1['avg_body'] and \
                       c2['body'] > 0.5*c2['avg_body'] and \
                       c3['body'] > 0.5*c3['avg_body']:
                           
                        if c2['close'] < c1['close'] and c3['close'] < c2['close']:
                            entry_price = c3['close']
                            entry_ts = c3.name
                            
                            result_end = entry_ts + timedelta(hours=1)
                            result_window = df[(df.index > entry_ts) & (df.index <= result_end)]
                            
                            if not result_window.empty:
                                exit_price = result_window.iloc[-1]['close']
                                rew_pct = (entry_price - exit_price) / entry_price * 100
                                events.append(HypothesisEvent(
                                    timestamp=entry_ts,
                                    symbol=symbol,
                                    entry_price=entry_price,
                                    direction="SHORT",
                                    rew_pct=rew_pct,
                                    mfe_pct=0, mae_pct=0, context={}
                                ))
        return events


class H025_RisingFallingThreeMethods(Hypothesis):
    """
    H025: Rising/Falling Three Methods
    Logic: Long green, 3 small red inside, Long green. (Continuation)
    """
    def __init__(self):
        super().__init__("H025", "Rising/Falling Three Methods", "Continuation pattern")

    def find_triggers(self, data: dict) -> list:
        # Complex multi-candle pattern (5 candles).
        # Simplified: Big Move, Consolidation (3 candles), Breakout.
        return []

class H026_DoubleTopBottom(Hypothesis):
    """
    H026: Double Top / Bottom
    Logic: Two peaks/troughs at similar level.
    """
    def __init__(self):
        super().__init__("H026", "Double Top / Bottom", "Reversal pattern")

    def find_triggers(self, data: dict) -> list:
        # Requires peak detection over longer window.
        # Let's skip for now or use simplified logic.
        return []

class H027_HeadAndShoulders(Hypothesis):
    """
    H027: Head and Shoulders
    Logic: L-H-L-HH-L-H-L pattern.
    """
    def __init__(self):
        super().__init__("H027", "Head and Shoulders", "Reversal pattern")

    def find_triggers(self, data: dict) -> list:
        # Too complex for simple script without pattern lib.
        return []

class H028_TriangleBreakout(Hypothesis):
    """
    H028: Triangle Breakout
    Logic: Lower Highs + Higher Lows (Coil), then breakout.
    """
    def __init__(self):
        super().__init__("H028", "Triangle Breakout", "Volatility contraction breakout")

    def find_triggers(self, data: dict) -> list:
        # Volatility contraction check?
        # Similar to BB Squeeze (H011) or Inside Bar (H023).
        return []

class H029_FlagPennant(Hypothesis):
    """
    H029: Bull/Bear Flag
    Logic: Strong move (Pole) + Consolidation (Flag) + Breakout.
    """
    def __init__(self):
        super().__init__("H029", "Flag / Pennant", "Continuation pattern")

    def find_triggers(self, data: dict) -> list:
        return []

class H030_CupAndHandle(Hypothesis):
    """
    H030: Cup and Handle
    Logic: Rounding bottom + pullback + breakout.
    """
    def __init__(self):
        super().__init__("H030", "Cup and Handle", "Bullish continuation")

    def find_triggers(self, data: dict) -> list:
        return []


class H025_RisingFallingThreeMethods(Hypothesis):
    """
    H025: Rising/Falling Three Methods
    Logic: Long green, 3 small red inside, Long green. (Continuation)
    """
    def __init__(self):
        super().__init__("H025", "Rising/Falling Three Methods", "Continuation pattern")

    def find_triggers(self, data: dict) -> list:
        # Complex multi-candle pattern (5 candles).
        # Simplified: Big Move, Consolidation (3 candles), Breakout.
        return []

class H026_DoubleTopBottom(Hypothesis):
    """
    H026: Double Top / Bottom
    Logic: Two peaks/troughs at similar level.
    """
    def __init__(self):
        super().__init__("H026", "Double Top / Bottom", "Reversal pattern")

    def find_triggers(self, data: dict) -> list:
        # Requires peak detection over longer window.
        # Let's skip for now or use simplified logic.
        return []

class H027_HeadAndShoulders(Hypothesis):
    """
    H027: Head and Shoulders
    Logic: L-H-L-HH-L-H-L pattern.
    """
    def __init__(self):
        super().__init__("H027", "Head and Shoulders", "Reversal pattern")

    def find_triggers(self, data: dict) -> list:
        # Too complex for simple script without pattern lib.
        return []

class H028_TriangleBreakout(Hypothesis):
    """
    H028: Triangle Breakout
    Logic: Lower Highs + Higher Lows (Coil), then breakout.
    """
    def __init__(self):
        super().__init__("H028", "Triangle Breakout", "Volatility contraction breakout")

    def find_triggers(self, data: dict) -> list:
        # Volatility contraction check?
        # Similar to BB Squeeze (H011) or Inside Bar (H023).
        return []

class H029_FlagPennant(Hypothesis):
    """
    H029: Bull/Bear Flag
    Logic: Strong move (Pole) + Consolidation (Flag) + Breakout.
    """
    def __init__(self):
        super().__init__("H029", "Flag / Pennant", "Continuation pattern")

    def find_triggers(self, data: dict) -> list:
        return []

class H030_CupAndHandle(Hypothesis):
    """
    H030: Cup and Handle
    Logic: Rounding bottom + pullback + breakout.
    """
    def __init__(self):
        super().__init__("H030", "Cup and Handle", "Bullish continuation")

    def find_triggers(self, data: dict) -> list:
        return []


class H864_Scalper(Hypothesis):
    """
    H_#864_SCALPER (Real Logic)
    Logic: Trend Pullback Scalper.
    - Trend: EMA15 > EMA60 (Long)
    - Trigger: RSI < 25 (Oversold) AND Close <= BB_Lower (Dip)
    - Filter: Volume > 1.5x Avg, EMA Spread > 0.4%
    - Time Window: 11:00 - 19:00 UTC
    - TP: 0.7%
    """
    def __init__(self):
        super().__init__("H864", "Scalper KZT Session", "Trend Pullback Scalper")

    def find_triggers(self, data: dict) -> list:
        events = []
        # Time Window: 11:00 - 19:00 UTC
        START_HOUR = 11
        END_HOUR = 19
        
        # Params
        EMA_FAST = 15
        EMA_SLOW = 60
        RSI_PERIOD = 14
        RSI_OS = 25
        RSI_OB = 75
        BB_PERIOD = 25
        BB_STD = 2.0
        VOL_THRESH = 1.5
        EMA_DEAD_ZONE = 0.004
        
        for symbol, df in data.items():
            df = df.copy()
            
            # Indicators
            df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
            df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
            
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(RSI_PERIOD).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            df['bb_mid'] = df['close'].rolling(BB_PERIOD).mean()
            df['bb_std'] = df['close'].rolling(BB_PERIOD).std()
            df['bb_upper'] = df['bb_mid'] + BB_STD * df['bb_std']
            df['bb_lower'] = df['bb_mid'] - BB_STD * df['bb_std']
            
            df['vol_ma'] = df['volume'].rolling(20).mean()
            
            for i in range(60, len(df)):
                row = df.iloc[i]
                ts = row.name
                
                # Check Time Window
                if not (START_HOUR <= ts.hour < END_HOUR):
                    continue
                    
                ema_fast_val = row['ema_fast']
                ema_slow_val = row['ema_slow']
                
                # EMA Dead Zone
                if abs(ema_fast_val - ema_slow_val) / ema_slow_val < EMA_DEAD_ZONE:
                    continue
                    
                # Volume Filter
                if row['volume'] < VOL_THRESH * row['vol_ma']:
                    continue
                    
                direction = None
                
                # LONG: Uptrend + Oversold + Dip
                if ema_fast_val > ema_slow_val:
                    if row['rsi'] < RSI_OS:
                        if row['close'] <= row['bb_lower']:
                            direction = "LONG"
                            
                # SHORT: Downtrend + Overbought + Rip
                elif ema_fast_val < ema_slow_val:
                    if row['rsi'] > RSI_OB:
                        if row['close'] >= row['bb_upper']:
                            direction = "SHORT"
                            
                if direction:
                    entry_price = row['close']
                    entry_ts = ts
                    
                    # TP/SL
                    tp_pct = 0.007 # 0.7%
                    sl_pct = 0.005 # 0.5%
                    
                    if direction == "LONG":
                        tp_price = entry_price * (1 + tp_pct)
                        sl_price = entry_price * (1 - sl_pct)
                    else:
                        tp_price = entry_price * (1 - tp_pct)
                        sl_price = entry_price * (1 + sl_pct)
                        
                    # Scan next 4 hours
                    scan_end = entry_ts + timedelta(hours=4)
                    future_df = df[(df.index > entry_ts) & (df.index <= scan_end)]
                    
                    outcome_pct = 0.0
                    
                    for _, f_row in future_df.iterrows():
                        if direction == "LONG":
                            if f_row['high'] >= tp_price:
                                outcome_pct = tp_pct * 100
                                break
                            if f_row['low'] <= sl_price:
                                outcome_pct = -sl_pct * 100
                                break
                        else:
                            if f_row['low'] <= tp_price:
                                outcome_pct = tp_pct * 100
                                break
                            if f_row['high'] >= sl_price:
                                outcome_pct = -sl_pct * 100
                                break
                                
                    if outcome_pct == 0.0 and not future_df.empty:
                         exit_price = future_df.iloc[-1]['close']
                         if direction == "LONG":
                             outcome_pct = (exit_price - entry_price) / entry_price * 100
                         else:
                             outcome_pct = (entry_price - exit_price) / entry_price * 100
                             
                    events.append(HypothesisEvent(
                        timestamp=entry_ts,
                        symbol=symbol,
                        entry_price=entry_price,
                        direction=direction,
                        rew_pct=outcome_pct,
                        mfe_pct=0, mae_pct=0, context={"rsi": row['rsi']}
                    ))
        return events
