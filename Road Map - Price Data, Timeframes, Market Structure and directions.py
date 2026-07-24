import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import time

# Attempt Scipy for Volume Profile Nodes
try:
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ==========================================
# 1. PRICE DATA (OHLCV) - Roadmap Section 1
# ==========================================
class PriceDataEngine:
    def __init__(self, api_key, access_token):
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
        try:
            profile = self.kite.profile()
            return True, f"Connected: {profile.get('user_name')}"
        except Exception as e:
            return False, f"Connection Failed: {str(e)}"

    def fetch_ohlcv(self, symbol_token, timeframe):
        limits = {"1 Minute": 30, "3 Minute": 60, "5 Minute": 60, "15 Minute": 90, "1 Hour": 180, "Daily": 365}
        days_back = limits.get(timeframe, 30)
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
            return df
        except Exception as e:
            st.error(f"Kite Error: {e}")
            return None

# ==========================================
# 2. MARKET STRUCTURE - Roadmap Section 1
# ==========================================
class MarketStructureEngine:
    def calculate_structure(self, df, window=5):
        df = df.copy()
        df['swing_high'] = df['high'][(df['high'] == df['high'].rolling(window=window, center=True).max())]
        df['swing_low'] = df['low'][(df['low'] == df['low'].rolling(window=window, center=True).min())]
        df['label'], df['break'] = "", "" 
        last_sh, last_sl = 0, 0
        trend = "Sideways"
        for i in range(window, len(df)):
            if not np.isnan(df['swing_high'].iloc[i]):
                h = df['swing_high'].iloc[i]
                df.at[df.index[i], 'label'] = "HH" if h > last_sh else "LH"
                if trend == "Bearish" and h > last_sh: df.at[df.index[i], 'break'] = "CHOCH"
                last_sh, trend = h, "Bullish"
            if not np.isnan(df['swing_low'].iloc[i]):
                l = df['swing_low'].iloc[i]
                df.at[df.index[i], 'label'] = "LL" if (l < last_sl or last_sl == 0) else "HL"
                if trend == "Bullish" and l < last_sl: df.at[df.index[i], 'break'] = "CHOCH"
                last_sl, trend = l, "Bearish"
            if trend == "Bullish" and df['close'].iloc[i] > last_sh and last_sh != 0: df.at[df.index[i], 'break'] = "BOS"
            if trend == "Bearish" and df['close'].iloc[i] < last_sl and last_sl != 0: df.at[df.index[i], 'break'] = "BOS"
        return df, trend

# ==========================================
# 3. SUPPORT & RESISTANCE - Roadmap Section 2
# ==========================================
class SupportResistanceEngine:
    def calculate_levels(self, df):
        lh, ll, lc = df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1]
        pp = (lh + ll + lc) / 3
        bc, tc = (lh + ll) / 2, (pp - (lh + ll) / 2) + pp
        fibs = {"61.8%": lh-((lh-ll)*0.618), "50%": lh-((lh-ll)*0.5), "38.2%": lh-((lh-ll)*0.382)}
        return {"Pivot": pp, "TC": tc, "BC": bc, "Fibs": fibs}

# ==========================================
# 4. VOLUME & MOMENTUM - Roadmap Section 3 & 4
# ==========================================
class AnalysisEngine:
    def process(self, df):
        # Trend & Momentum (Section 4)
        df['ema200'] = ta.trend.EMAIndicator(df['close'], 200).ema_indicator()
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
        
        # Volume (Section 3)
        df['avg_vol'] = df['volume'].rolling(20).mean()
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        return df

# ==========================================
# 5. SIGNAL SCORING (Result Oriented) - Section 21
# ==========================================
def calculate_signal(row, trend):
    score = 0
    reasons = []
    
    if row['close'] > row['ema200']: score += 25; reasons.append("Above EMA 200")
    if 40 < row['rsi'] < 65: score += 15; reasons.append("RSI Neutral-Bullish")
    if trend == "Bullish": score += 30; reasons.append("Market Structure Bullish")
    if row['adx'] > 25: score += 20; reasons.append("Strong Trend Strength")
    
    if score >= 70: return "STRONG BUY", "green", reasons
    if score >= 40: return "BUY / HOLD", "orange", reasons
    return "BEARISH / AVOID", "red", reasons

# ==========================================
# MAIN DASHBOARD
# ==========================================
st.set_page_config(layout="wide", page_title="Result Dashboard")
st.sidebar.header("🔑 Kite API & Live")
api_key = st.sidebar.text_input("API Key", type="password")
token = st.sidebar.text_input("Access Token", type="password")
refresh_rate = st.sidebar.slider("Refresh Interval (Seconds)", 5, 60, 10)
live_on = st.sidebar.toggle("Live Refresh Mode")

symbols = {
    "NIFTY 50": 256265, "BANK NIFTY": 260105, "FIN NIFTY": 257801, 
    "RELIANCE": 738561, "HDFC BANK": 341249, "TCS": 295321
}
sym_name = st.sidebar.selectbox("Symbol", list(symbols.keys()))
tf = st.sidebar.selectbox("Timeframe", ["1 Minute", "5 Minute", "15 Minute", "1 Hour"])

def main():
    if not api_key or not token:
        st.info("Please enter API Key and Access Token in the sidebar to start live data.")
        return

    engine = PriceDataEngine(api_key, token)
    df = engine.fetch_ohlcv(symbols[sym_name], tf)
    
    if df is not None:
        # Process Modules in Roadmap Order
        df, trend = MarketStructureEngine().calculate_structure(df)
        df = AnalysisEngine().process(df)
        levels = SupportResistanceEngine().calculate_levels(df)
        
        # Latest Data for Dashboard
        row = df.iloc[-1]
        sig, sig_col, reasons = calculate_signal(row, trend)

        # ---------------------------
        # RESULT DASHBOARD (Section 24)
        # ---------------------------
        st.markdown(f"### 🎯 Live Analysis: {sym_name} ({tf})")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("SIGNAL", sig, delta=trend, delta_color="normal")
            st.markdown(f"<div style='height:10px; background-color:{sig_col}; border-radius:5px;'></div>", unsafe_allow_html=True)
        c2.metric("PRICE", round(row['close'], 2), f"{round(row['close'] - df['close'].iloc[-2], 2)}")
        c3.metric("RSI (Momentum)", round(row['rsi'], 1))
        c4.metric("ADX (Strength)", round(row['adx'], 1))

        # Chart
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.add_trace(go.Scatter(x=df.index, y=df['ema200'], name="Trend EMA 200", line=dict(color='yellow')))
        fig.update_layout(height=500, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Tabs for Roadmap Details
        t1, t2, t3, t4 = st.tabs(["Structure & Signal", "S/R Levels", "Volume Analysis", "Raw Data"])
        
        with t1:
            st.write("**Signal Confirmation Factors:**")
            for r in reasons: st.write(f"✅ {r}")
            st.write("**Recent BOS/CHOCH:**")
            st.dataframe(df[df['break'] != ""].tail(5))
        
        with t2:
            st.write("**Pivot & Fibonacci Levels:**")
            st.json(levels)
            
        with t3:
            if row['volume'] == 0:
                st.warning("Volume is 0. Note: Zerodha does not provide volume for Spot Indices (Nifty/BankNifty). Use Futures for volume data.")
            st.write(f"**OBV:** {row['obv']} | **MFI:** {row['mfi']}")
            st.line_chart(df['obv'])
            
        with t4:
            st.dataframe(df.tail(20))

if __name__ == "__main__":
    main()
    if live_on:
        time.sleep(refresh_rate)
        st.rerun()
