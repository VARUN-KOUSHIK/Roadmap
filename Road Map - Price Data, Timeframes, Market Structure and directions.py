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
            "Daily": "day"
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
        # Trend (Section 6)
        df['ema200'] = ta.trend.EMAIndicator(df['close'], 200).ema_indicator()
        # Momentum (Section 4)
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
        # Volume (Section 3)
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        return df

# ==========================================
# 5. VOLATILITY INDICATORS - Roadmap Section 5
# ==========================================
class VolatilityEngine:
    def calculate_volatility(self, df):
        df = df.copy()
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_high'], df['bb_low'], df['bb_mid'] = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg()
        # Keltner Channels
        kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
        df['kc_high'], df['kc_low'] = kc.keltner_channel_hband(), kc.keltner_channel_lband()
        # Donchian Channel
        dc = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'])
        df['dc_high'], df['dc_low'] = dc.donchian_channel_hband(), dc.donchian_channel_lband()
        # Standard Deviation
        df['std_dev'] = df['close'].rolling(window=20).std()
        # Historical Volatility (252 day annualization proxy)
        df['hist_vol'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
        # Choppiness Index (Formula: 100 * LOG10( SUM(ATR,n) / (MaxHigh(n) - MinLow(n)) ) / LOG10(n))
        n = 14
        tr_sum = df['atr'].rolling(n).sum()
        price_range = df['high'].rolling(n).max() - df['low'].rolling(n).min()
        df['choppiness'] = 100 * np.log10(tr_sum / price_range) / np.log10(n)
        return df

# ==========================================
# 21. SIGNAL SCORING - Roadmap Section 21
# ==========================================
def calculate_signal(row, trend):
    score = 0
    if row['close'] > row['ema200']: score += 25
    if 40 < row['rsi'] < 65: score += 15
    if trend == "Bullish": score += 30
    if row['adx'] > 25: score += 20
    # Volatility Check (Deduct if too choppy)
    if 'choppiness' in row and row['choppiness'] > 61.8: score -= 10 
    
    if score >= 70: return "STRONG BUY", "green"
    if score >= 40: return "BUY / HOLD", "orange"
    return "BEARISH / AVOID", "red"

# ==========================================
# MAIN DASHBOARD
# ==========================================
st.set_page_config(layout="wide", page_title="Master Roadmap Dashboard")
st.sidebar.header("🔑 Kite API & Refresh")
api_key = st.sidebar.text_input("API Key", type="password")
token = st.sidebar.text_input("Access Token", type="password")
refresh_rate = st.sidebar.slider("Refresh Interval", 5, 60, 10)
live_on = st.sidebar.toggle("Live Refresh")

symbols = {"NIFTY 50": 256265, "BANK NIFTY": 260105, "RELIANCE": 738561, "TCS": 295321}
sym_name = st.sidebar.selectbox("Symbol", list(symbols.keys()))
tf = st.sidebar.selectbox("Timeframe", ["1 Minute", "5 Minute", "15 Minute", "1 Hour"])

def main():
    if not api_key or not token:
        st.info("Please enter credentials in the sidebar.")
        return

    engine = PriceDataEngine(api_key, token)
    df = engine.fetch_ohlcv(symbols[sym_name], tf)
    
    if df is not None:
        # PROCESS IN ROADMAP ORDER
        # 1 & 2
        df, trend = MarketStructureEngine().calculate_structure(df)
        # 3 & 4
        df = AnalysisEngine().process(df)
        # 5 (NEW)
        df = VolatilityEngine().calculate_volatility(df)
        levels = SupportResistanceEngine().calculate_levels(df)
        
        row = df.iloc[-1]
        sig, sig_col = calculate_signal(row, trend)

        # DASHBOARD (Section 24)
        st.markdown(f"### 🎯 Live Dashboard: {sym_name}")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("SIGNAL", sig)
            st.markdown(f"<div style='height:8px; background-color:{sig_col};'></div>", unsafe_allow_html=True)
        c2.metric("PRICE", round(row['close'], 2))
        c3.metric("ATR (Volatility)", round(row['atr'], 2))
        c4.metric("CHOPPINESS", round(row['choppiness'], 1))

        # Chart with Bollinger Bands
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.add_trace(go.Scatter(x=df.index, y=df['bb_high'], name="BB Upper", line=dict(color='rgba(173, 216, 230, 0.4)')))
        fig.add_trace(go.Scatter(x=df.index, y=df['bb_low'], name="BB Lower", line=dict(color='rgba(173, 216, 230, 0.4)'), fill='tonexty'))
        fig.update_layout(height=500, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # TABS
        t1, t2, t3, t4, t5 = st.tabs(["Structure", "S/R Levels", "Volume & Momentum", "Volatility (Sec 5)", "Raw Data"])
        
        with t1: st.dataframe(df[df['break'] != ""].tail(5))
        with t2: st.json(levels)
        with t3: st.write(f"RSI: {round(row['rsi'],1)} | MFI: {round(row['mfi'],1)} | ADX: {round(row['adx'],1)}")
        with t4:
            st.write("**Volatility Analysis Details**")
            st.write(f"Standard Deviation: {round(row['std_dev'], 2)}")
            st.write(f"Historical Volatility: {round(row['hist_vol'], 2)}%")
            st.write(f"Donchian High: {row['dc_high']} | Donchian Low: {row['dc_low']}")
            st.progress(min(max(int(row['choppiness']), 0), 100), text=f"Choppiness Index: {round(row['choppiness'], 1)}")
        with t5: st.dataframe(df.tail(10))

if __name__ == "__main__":
    main()
    if live_on:
        time.sleep(refresh_rate)
        st.rerun()
