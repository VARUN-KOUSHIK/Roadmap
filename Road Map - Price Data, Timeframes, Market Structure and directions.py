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
        # STRIP removes accidental spaces from copy-pasting
        self.api_key = api_key.strip()
        self.access_token = access_token.strip()
        self.kite = KiteConnect(api_key=self.api_key)
        self.kite.set_access_token(self.access_token)
        
        self.tf_map = {
            "1 Minute": "minute", "3 Minute": "3minute", "5 Minute": "5minute",
            "15 Minute": "15minute", "30 Minute": "30minute", "1 Hour": "60minute",
            "4 Hour": "60minute", "Daily": "day", "Weekly": "day"
        }

    def test_connection(self):
        """Verify if the token is valid by fetching profile"""
        try:
            profile = self.kite.profile()
            return True, f"Connected! Welcome, {profile.get('user_name')}"
        except Exception as e:
            return False, f"Connection Failed: {str(e)}"

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
            if not records: return None
            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            if timeframe == "4 Hour":
                df = df.resample('4H').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            elif timeframe == "Weekly":
                df = df.resample('W-MON').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            return df
        except Exception as e:
            st.error(f"Kite Fetch Error: {e}")
            return None

# ==========================================
# 2. MARKET STRUCTURE - Roadmap Section 1
# ==========================================
class MarketStructureEngine:
    def calculate_structure(self, df, window=5):
        df = df.copy()
        df['swing_high'] = df['high'][(df['high'] == df['high'].rolling(window=window, center=True).max())]
        df['swing_low'] = df['low'][(df['low'] == df['low'].rolling(window=window, center=True).min())]
        df['label'] = "" 
        df['break'] = "" 
        last_sh, last_sl = 0, 0
        curr_struct_trend = "Sideways"
        
        for i in range(window, len(df)):
            if not np.isnan(df['swing_high'].iloc[i]):
                this_h = df['swing_high'].iloc[i]
                if this_h > last_sh:
                    df.at[df.index[i], 'label'] = "HH"
                    if curr_struct_trend == "Bearish": df.at[df.index[i], 'break'] = "CHOCH"
                    curr_struct_trend = "Bullish"
                else: df.at[df.index[i], 'label'] = "LH"
                last_sh = this_h
            if not np.isnan(df['swing_low'].iloc[i]):
                this_l = df['swing_low'].iloc[i]
                if this_l < last_sl or last_sl == 0:
                    df.at[df.index[i], 'label'] = "LL"
                    if curr_struct_trend == "Bullish": df.at[df.index[i], 'break'] = "CHOCH"
                    curr_struct_trend = "Bearish"
                else: df.at[df.index[i], 'label'] = "HL"
                last_sl = this_l
            if curr_struct_trend == "Bullish" and df['close'].iloc[i] > last_sh and last_sh != 0:
                if df['break'].iloc[i] == "": df.at[df.index[i], 'break'] = "BOS"
            if curr_struct_trend == "Bearish" and df['close'].iloc[i] < last_sl and last_sl != 0:
                if df['break'].iloc[i] == "": df.at[df.index[i], 'break'] = "BOS"
        return df, curr_struct_trend

# ==========================================
# 3. TREND DIRECTION - Roadmap Section 1
# ==========================================
class TrendDirectionEngine:
    def analyze(self, df):
        df['ema200'] = ta.trend.EMAIndicator(df['close'], 200).ema_indicator()
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 14).adx()
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
        # Custom SuperTrend logic for 'ta' library
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 10).average_true_range()
        df['st_upper'] = ((df['high']+df['low'])/2) + (3 * atr)
        df['st_bull'] = df['close'] > df['st_upper'].shift(1)
        return df

# ==========================================
# MAIN STREAMLIT UI
# ==========================================
st.set_page_config(layout="wide", page_title="Unified Roadmap Dashboard")
st.title("🛡️ Roadmap Section 1: Data, Structure & Trend")

# Sidebar
st.sidebar.header("🔑 Kite API Provision")
in_api_key = st.sidebar.text_input("Kite API Key", value="", type="password")
in_access_token = st.sidebar.text_input("Access Token", value="", type="password")

# Mapping Instrument Tokens (NSE)
symbols = {
    "NIFTY 50": 256265, "BANK NIFTY": 260105, "FIN NIFTY": 257801, 
    "SENSEX": 265, "MIDCAP NIFTY": 288009, 
    "RELIANCE": 738561, "HDFC BANK": 341249, "TCS": 295321, "INFY": 408065
}

sym = st.sidebar.selectbox("Symbol", list(symbols.keys()))
tf = st.sidebar.selectbox("Timeframe", ["1 Minute", "3 Minute", "5 Minute", "15 Minute", "30 Minute", "1 Hour", "Daily"])

if st.sidebar.button("Execute Unified Analysis"):
    if not in_api_key or not in_access_token:
        st.warning("Please enter credentials.")
    else:
        # Initialize
        data_eng = PriceDataEngine(in_api_key, in_access_token)
        
        # Test Connection first
        success, msg = data_eng.test_connection()
        if not success:
            st.error(msg)
            st.info("💡 Ensure you generated a NEW access token today. Tokens from yesterday will not work.")
        else:
            st.sidebar.success(msg)
            with st.spinner("Analyzing Market Structure..."):
                df = data_eng.fetch_ohlcv(symbols[sym], tf)
                if df is not None:
                    # Logic Steps
                    struct_eng = MarketStructureEngine()
                    df, s_trend = struct_eng.calculate_structure(df)
                    
                    trend_eng = TrendDirectionEngine()
                    df = trend_eng.analyze(df)
                    
                    # Confidence Score
                    row = df.iloc[-1]
                    score = 0
                    if row['close'] > row['ema200']: score += 2
                    if row['st_bull']: score += 1
                    if row['adx'] > 25: score += 1
                    
                    state = "STRONG BULLISH" if score >= 3 else "BULLISH" if score >= 1 else "BEARISH/SIDEWAYS"
                    
                    # Display Results
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Market State", state)
                    c2.metric("Confidence Score", score)
                    c3.metric("ADX (Strength)", round(row['adx'], 2))
                    c4.metric("Structure", s_trend)

                    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
                    fig.update_layout(height=600, template="plotly_dark", title=f"{sym} Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Market Structure Breaks (BOS/CHOCH)")
                    st.write(df[df['break'] != ""][['close', 'label', 'break']].tail(10))
