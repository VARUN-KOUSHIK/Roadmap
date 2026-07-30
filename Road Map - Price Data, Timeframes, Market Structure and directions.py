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
        # Using the Kite Basic Plan limits provided previously
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
class MomentumVolumeEngine:
    def process(self, df):
        # Momentum (Section 4)
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        # Volume (Section 3)
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        return df

# ==========================================
# 5. VOLATILITY INDICATORS - Roadmap Section 5
# ==========================================
class VolatilityEngine:
    def calculate(self, df):
        df = df.copy()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_high'], df['bb_low'] = bb.bollinger_hband(), bb.bollinger_lband()
        # Choppiness Index
        n = 14
        tr_sum = df['atr'].rolling(n).sum()
        price_range = df['high'].rolling(n).max() - df['low'].rolling(n).min()
        df['choppiness'] = 100 * np.log10(tr_sum / price_range) / np.log10(n)
        return df

# ==========================================
# 6. TREND INDICATORS - Roadmap Section 6
# ==========================================
class TrendIndicatorEngine:
    def calculate(self, df):
        df = df.copy()
        # EMAs (9, 22, 52, 100, 200)
        for p in [9, 22, 52, 100, 200]:
            df[f'ema{p}'] = ta.trend.EMAIndicator(df['close'], window=p).ema_indicator()
        
        # SMA
        df['sma50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        
        # VWMA (Volume Weighted Moving Average)
        def vwma(price, volume, window):
            pv = price * volume
            return pv.rolling(window).sum() / volume.rolling(window).sum()
        df['vwma20'] = vwma(df['close'], df['volume'], 20)

        # Super Trend (Logic from previous sections)
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 10).average_true_range()
        df['st_upper'] = ((df['high']+df['low'])/2) + (3 * atr)
        df['st_bull'] = df['close'] > df['st_upper'].shift(1)

        # Ichimoku Cloud
        ichi = ta.trend.IchimokuIndicator(df['high'], df['low'])
        df['ichi_a'] = ichi.ichimoku_a()
        df['ichi_b'] = ichi.ichimoku_b()
        df['ichi_base'] = ichi.ichimoku_base_line()

        # Parabolic SAR
        df['psar'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()

        # ADX & DMI
        adx_obj = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx_obj.adx()
        df['dmi_plus'] = adx_obj.adx_pos()
        df['dmi_minus'] = adx_obj.adx_neg()
        
        return df

# ==========================================
# 21. SIGNAL SCORING & DASHBOARD
# ==========================================
def calculate_signal(row, trend):
    score = 0
    if row['close'] > row['ema200']: score += 25
    if 40 < row['rsi'] < 65: score += 15
    if trend == "Bullish": score += 30
    if row['adx'] > 25: score += 20
    if row['st_bull']: score += 10
    
    if score >= 75: return "STRONG BUY", "green"
    if score >= 50: return "BUY / HOLD", "orange"
    return "BEARISH / AVOID", "red"

# ==========================================
# MAIN UI EXECUTION
# ==========================================
st.set_page_config(layout="wide", page_title="Master Roadmap")
st.sidebar.header("🔑 Kite Access")
api_key = st.sidebar.text_input("API Key", type="password")
token = st.sidebar.text_input("Access Token", type="password")
live_on = st.sidebar.toggle("Live Refresh")

symbols = {"NIFTY 50": 256265, "BANK NIFTY": 260105, "RELIANCE": 738561, "TCS": 295321}
sym_name = st.sidebar.selectbox("Symbol", list(symbols.keys()))
tf = st.sidebar.selectbox("Timeframe", ["1 Minute", "5 Minute", "15 Minute", "1 Hour"])

def main():
    if not api_key or not token:
        st.info("Enter credentials to start.")
        return

    data_eng = PriceDataEngine(api_key, token)
    df = data_eng.fetch_ohlcv(symbols[sym_name], tf)
    
    if df is not None:
        # EXECUTE IN ORDER
        df, trend = MarketStructureEngine().calculate_structure(df)
        df = MomentumVolumeEngine().process(df)
        df = VolatilityEngine().calculate(df)
        df = TrendIndicatorEngine().calculate(df)
        levels = SupportResistanceEngine().calculate_levels(df)
        
        row = df.iloc[-1]
        sig, sig_col = calculate_signal(row, trend)

        # UI DASHBOARD
        st.markdown(f"### 🎯 Roadmap Dashboard: {sym_name} | Signal: {sig}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", round(row['close'], 2))
        c2.metric("ADX (Trend Strength)", round(row['adx'], 1))
        c3.metric("RSI", round(row['rsi'], 1))
        c4.metric("Market Structure", trend)

        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.add_trace(go.Scatter(x=df.index, y=df['ema200'], name="EMA 200", line=dict(color='yellow')))
        fig.add_trace(go.Scatter(x=df.index, y=df['psar'], mode='markers', name="PSAR", marker=dict(size=3, color='white')))
        fig.update_layout(height=500, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # TABS
        t1, t2, t3, t4, t5 = st.tabs(["Structure", "S/R Levels", "Indicators (3-5)", "Trend Details (Sec 6)", "Raw Logs"])
        
        with t1: st.write(df[df['break'] != ""].tail(5))
        with t2: st.json(levels)
        with t3: 
            st.write(f"Volatility (ATR): {round(row['atr'], 2)}")
            st.write(f"Choppiness: {round(row['choppiness'], 1)}")
        with t4:
            st.write("**Advanced Trend Indicators (Section 6)**")
            st.write(f"DMI+: {round(row['dmi_plus'], 1)} | DMI-: {round(row['dmi_minus'], 1)}")
            st.write(f"EMA Alignment: 9 > 22: {row['ema9'] > row['ema22']}")
            st.write(f"SuperTrend Bullish: {row['st_bull']}")
            st.write(f"VWMA (20): {round(row['vwma20'], 2)}")
        with t5: st.dataframe(df.tail(10))

if __name__ == "__main__":
    main()
    if live_on:
        time.sleep(10)
        st.rerun()
