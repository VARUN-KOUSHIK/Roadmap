import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from kiteconnect import KiteConnect
from datetime import datetime, timedelta

# ==========================================
# 1. PRICE DATA (OHLCV) - Roadmap Section 1
# ==========================================
class PriceDataEngine:
    def __init__(self, api_key, access_token):
        """Manual Provision for Kite API"""
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        
        self.tf_map = {
            "1 Minute": "minute", "3 Minute": "3minute", "5 Minute": "5minute",
            "15 Minute": "15minute", "30 Minute": "30minute", "1 Hour": "60minute",
            "4 Hour": "60minute", "Daily": "day", "Weekly": "day"
        }

    def fetch_ohlcv(self, symbol_token, timeframe):
        lookback = {"1 Minute":1, "3 Minute":2, "5 Minute":5, "15 Minute":15, 
                    "30 Minute":20, "1 Hour":30, "4 Hour":60, "Daily":365, "Weekly":730}
        days_back = lookback.get(timeframe, 30)
        try:
            records = self.kite.historical_data(
                instrument_token=symbol_token,
                from_date=(datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d %H:%M:%S"),
                to_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                interval=self.tf_map.get(timeframe)
            )
            df = pd.DataFrame(records)
            if df.empty: return None
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Resampling for Roadmap specific timeframes
            if timeframe == "4 Hour":
                df = df.resample('4H').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            elif timeframe == "Weekly":
                df = df.resample('W-MON').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            
            return df
        except Exception as e:
            st.error(f"Kite API Error: {e}")
            return None

# ==========================================
# 2. MARKET STRUCTURE - Roadmap Section 1 (Calculate)
# ==========================================
class MarketStructureEngine:
    def calculate_structure(self, df, window=5):
        df = df.copy()
        # Detect Swing points
        df['swing_high'] = df['high'][(df['high'] == df['high'].rolling(window=window, center=True).max())]
        df['swing_low'] = df['low'][(df['low'] == df['low'].rolling(window=window, center=True).min())]
        
        df['label'] = "" 
        df['break'] = "" 
        last_sh, last_sl = 0, 0
        curr_struct_trend = "Sideways"
        
        for i in range(window, len(df)):
            # Higher High / Lower High
            if not np.isnan(df['swing_high'].iloc[i]):
                this_h = df['swing_high'].iloc[i]
                if this_h > last_sh:
                    df.at[df.index[i], 'label'] = "HH"
                    if curr_struct_trend == "Bearish": df.at[df.index[i], 'break'] = "CHOCH"
                    curr_struct_trend = "Bullish"
                else: df.at[df.index[i], 'label'] = "LH"
                last_sh = this_h
            
            # Lower Low / Higher Low
            if not np.isnan(df['swing_low'].iloc[i]):
                this_l = df['swing_low'].iloc[i]
                if this_l < last_sl or last_sl == 0:
                    df.at[df.index[i], 'label'] = "LL"
                    if curr_struct_trend == "Bullish": df.at[df.index[i], 'break'] = "CHOCH"
                    curr_struct_trend = "Bearish"
                else: df.at[df.index[i], 'label'] = "HL"
                last_sl = this_l
            
            # BOS Detection
            if curr_struct_trend == "Bullish" and df['close'].iloc[i] > last_sh and last_sh != 0:
                if df['break'].iloc[i] == "": df.at[df.index[i], 'break'] = "BOS"
            if curr_struct_trend == "Bearish" and df['close'].iloc[i] < last_sl and last_sl != 0:
                if df['break'].iloc[i] == "": df.at[df.index[i], 'break'] = "BOS"

        df['trend_strength'] = df['close'].diff(window).rolling(window).mean()
        return df, curr_struct_trend

# ==========================================
# 3. TREND DIRECTION - Roadmap Section 1 (Methods)
# ==========================================
class TrendDirectionEngine:
    def get_supertrend(self, df, period=10, mult=3):
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=period).average_true_range()
        hl2 = (df['high'] + df['low']) / 2
        upper, lower = hl2 + (mult * atr), hl2 - (mult * atr)
        st_dir = [True] * len(df)
        for i in range(1, len(df)):
            if df['close'].iloc[i] > upper.iloc[i-1]: st_dir[i] = True
            elif df['close'].iloc[i] < lower.iloc[i-1]: st_dir[i] = False
            else: st_dir[i] = st_dir[i-1]
        return st_dir

    def analyze(self, df):
        # EMA Alignment
        df['ema20'] = ta.trend.EMAIndicator(df['close'], 20).ema_indicator()
        df['ema50'] = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
        df['ema200'] = ta.trend.EMAIndicator(df['close'], 200).ema_indicator()
        # ADX (Trend Strength)
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 14).adx()
        # VWAP
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
        # SuperTrend
        df['st_bull'] = self.get_supertrend(df)
        return df

# ==========================================
# MAIN STREAMLIT UI
# ==========================================
st.set_page_config(layout="wide", page_title="Indian Market Roadmap")

st.title("🛡️ Roadmap Section 1: Data, Structure & Trend")

# Sidebar for API Provision
st.sidebar.header("🔑 Kite API Credentials")
API_KEY = st.sidebar.text_input("Kite API Key", type="password")
ACCESS_TOKEN = st.sidebar.text_input("Access Token", type="password")

# Symbols Requested
symbols = {
    "NIFTY 50": 256265, "BANK NIFTY": 260105, "FIN NIFTY": 257801, 
    "SENSEX": 265, "MIDCAP NIFTY": 288009, 
    "RELIANCE": 738561, "HDFC BANK": 341249, "TCS": 295321
}

sym = st.sidebar.selectbox("Symbol", list(symbols.keys()))
tf = st.sidebar.selectbox("Timeframe", ["1 Minute", "3 Minute", "5 Minute", "15 Minute", "30 Minute", "1 Hour", "4 Hour", "Daily", "Weekly"])

if st.sidebar.button("Execute Unified Analysis"):
    if not API_KEY or not ACCESS_TOKEN:
        st.warning("Please enter your Kite API details.")
    else:
        # 1. Price Data
        data_eng = PriceDataEngine(API_KEY, ACCESS_TOKEN)
        df = data_eng.fetch_ohlcv(symbols[sym], tf)
        df_daily = data_eng.fetch_ohlcv(symbols[sym], "Daily")

        if df is not None:
            # 2. Market Structure
            struct_eng = MarketStructureEngine()
            df, s_trend = struct_eng.calculate_structure(df)
            
            # 3. Trend Direction
            trend_eng = TrendDirectionEngine()
            df = trend_eng.analyze(df)
            
            # Final Result Logic
            row = df.iloc[-1]
            score = 0
            if row['close'] > row['ema200']: score += 2
            if row['st_bull']: score += 1
            if row['adx'] > 25: score *= 1.5
            
            state = "STRONG BULLISH" if score >= 3 else "BULLISH" if score >= 1 else "BEARISH/SIDEWAYS"

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Final Market State", state)
            m2.metric("Trend Strength (ADX)", round(row['adx'], 2))
            m3.metric("Structure", s_trend)

            # Candlestick
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
            fig.update_layout(height=600, template="plotly_dark", title=f"{sym} {tf} Roadmap Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            # Events Log
            st.subheader("Market Structure Events (BOS/CHOCH)")
            st.write(df[df['break'] != ""][['close', 'label', 'break']].tail(10))
