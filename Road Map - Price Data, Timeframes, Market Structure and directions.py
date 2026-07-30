import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import time

# ==========================================
# 1. PRICE DATA (OHLCV) & MARKET STRUCTURE
# ==========================================
class Section1_Data_Structure:
    def __init__(self, api_key, access_token):
        self.kite = KiteConnect(api_key=api_key.strip())
        self.kite.set_access_token(access_token.strip())
        self.tf_map = {
            "1 Minute": "minute", "3 Minute": "3minute", "5 Minute": "5minute",
            "15 Minute": "15minute", "30 Minute": "30minute", "1 Hour": "60minute",
            "4 Hour": "60minute", "Daily": "day", "Weekly": "day"
        }
        self.limits = {
            "1 Minute": 30, "3 Minute": 60, "5 Minute": 60, "15 Minute": 90,
            "30 Minute": 180, "1 Hour": 180, "4 Hour": 180, "Daily": 365, "Weekly": 365
        }

    def fetch_data(self, token, tf):
        days = self.limits.get(tf, 30)
        try:
            records = self.kite.historical_data(token, (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S"), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.tf_map.get(tf))
            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            if tf == "4 Hour": df = df.resample('4H').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            if tf == "Weekly": df = df.resample('W-MON').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            return df
        except Exception as e:
            st.error(f"Fetch Error: {e}")
            return None

    def calculate_structure(self, df):
        window = 5
        df['sw_h'] = df['high'][(df['high'] == df['high'].rolling(window=window, center=True).max())]
        df['sw_l'] = df['low'][(df['low'] == df['low'].rolling(window=window, center=True).min())]
        df['label'], df['break'] = "", ""
        l_sh, l_sl = 0, 0
        trend = "Sideways"
        for i in range(window, len(df)):
            if not np.isnan(df['sw_h'].iloc[i]):
                h = df['sw_h'].iloc[i]
                df.at[df.index[i], 'label'] = "HH" if h > l_sh else "LH"
                if trend == "Bearish" and h > l_sh: df.at[df.index[i], 'break'] = "CHOCH"
                l_sh, trend = h, "Bullish"
            if not np.isnan(df['sw_l'].iloc[i]):
                l = df['sw_l'].iloc[i]
                df.at[df.index[i], 'label'] = "LL" if (l < l_sl or l_sl == 0) else "HL"
                if trend == "Bullish" and l < l_sl: df.at[df.index[i], 'break'] = "CHOCH"
                l_sl, trend = l, "Bearish"
            if trend == "Bullish" and df['close'].iloc[i] > l_sh and l_sh != 0: df.at[df.index[i], 'break'] = "BOS"
            if trend == "Bearish" and df['close'].iloc[i] < l_sl and l_sl != 0: df.at[df.index[i], 'break'] = "BOS"
        return df, trend

# ==========================================
# 2. SUPPORT & RESISTANCE
# ==========================================
class Section2_S_R:
    def calculate(self, df):
        lh, ll, lc = df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1]
        pp = (lh + ll + lc) / 3
        bc, tc = (lh + ll) / 2, (pp - (lh + ll) / 2) + pp
        mx, mn = df['high'].max(), df['low'].min()
        fib618 = mx - ((mx - mn) * 0.618)
        return {"Pivot": pp, "CPR_Top": tc, "CPR_Bot": bc, "Fib618": fib618}

# ==========================================
# 3. VOLUME ANALYSIS
# ==========================================
class Section3_Volume:
    def calculate(self, df):
        df['avg_vol'] = df['volume'].rolling(20).mean()
        df['rel_vol'] = df['volume'] / df['avg_vol']
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        return df

# ==========================================
# 4 & 5. MOMENTUM & VOLATILITY
# ==========================================
class Section4_5_Momentum_Volatility:
    def calculate(self, df):
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        n = 14
        tr_sum = df['atr'].rolling(n).sum()
        p_range = df['high'].rolling(n).max() - df['low'].rolling(n).min()
        df['choppiness'] = 100 * np.log10(tr_sum / p_range) / np.log10(n)
        return df

# ==========================================
# 6. TREND INDICATORS
# ==========================================
class Section6_Trend:
    def _wma(self, series, period):
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    def calculate(self, df):
        for p in [9, 22, 52, 100, 200]:
            df[f'ema{p}'] = ta.trend.EMAIndicator(df['close'], window=p).ema_indicator()
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 10).average_true_range()
        df['st_upper'] = ((df['high']+df['low'])/2) + (3 * atr)
        df['st_bull'] = df['close'] > df['st_upper'].shift(1)
        adx_obj = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'], df['dmi_p'], df['dmi_m'] = adx_obj.adx(), adx_obj.adx_pos(), adx_obj.adx_neg()
        df['psar'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()
        period = 20
        half_p, sqrt_p = period // 2, int(np.sqrt(period))
        wma_half, wma_full = self._wma(df['close'], half_p), self._wma(df['close'], period)
        df['hma20'] = self._wma(2 * wma_half - wma_full, sqrt_p)
        return df

# ==========================================
# 7. PRICE ACTION RECOGNITION
# ==========================================
class Section7_PriceAction:
    def recognize(self, df):
        df = df.copy()
        df['body'] = df['close'] - df['open']
        df['abs_body'] = abs(df['body'])
        df['range'] = df['high'] - df['low']
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Section 7 requested recognitions
        df['Inside Bar'] = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
        df['Outside Bar'] = (df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))
        df['Marubozu'] = (df['abs_body'] > (df['range'] * 0.9))
        df['Three White Soldiers'] = (df['body'] > 0) & (df['body'].shift(1) > 0) & (df['body'].shift(2) > 0)
        df['Three Black Crows'] = (df['body'] < 0) & (df['body'].shift(1) < 0) & (df['body'].shift(2) < 0)
        return df

# ==========================================
# 8. CANDLESTICK PATTERN DETECTION
# ==========================================
class Section8_CandlestickPatterns:
    def detect(self, df):
        df = df.copy()
        body = df['close'] - df['open']
        abs_body = abs(body)
        range_tot = df['high'] - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        
        # Bullish
        df['Bullish Engulfing'] = (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1)) & (body.shift(1) < 0)
        df['Hammer'] = (lower_wick > (abs_body * 2)) & (upper_wick < (abs_body * 0.5))
        df['Morning Star'] = (body.shift(2) < 0) & (abs_body.shift(1) < abs_body.shift(2)*0.3) & (body > 0)
        df['Piercing Pattern'] = (body.shift(1) < 0) & (df['open'] < df['low'].shift(1)) & (df['close'] > (df['open'].shift(1) + df['close'].shift(1))/2)
        
        # Bearish
        df['Bearish Engulfing'] = (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1)) & (body.shift(1) > 0)
        df['Shooting Star'] = (upper_wick > (abs_body * 2)) & (lower_wick < (abs_body * 0.5))
        df['Evening Star'] = (body.shift(2) > 0) & (abs_body.shift(1) < abs_body.shift(2)*0.3) & (body < 0)
        df['Dark Cloud Cover'] = (body.shift(1) > 0) & (df['open'] > df['high'].shift(1)) & (df['close'] < (df['open'].shift(1) + df['close'].shift(1))/2)
        
        # Neutral
        df['Doji'] = abs_body <= (range_tot * 0.1)
        df['Spinning Top'] = (abs_body < (range_tot * 0.3)) & (upper_wick > abs_body) & (lower_wick > abs_body)
        
        return df

# ==========================================
# MAIN DASHBOARD EXECUTION
# ==========================================
st.set_page_config(layout="wide", page_title="Master Roadmap Bot")
st.sidebar.header("🔑 Kite Access")
api_key = st.sidebar.text_input("API Key", type="password")
token = st.sidebar.text_input("Access Token", type="password")
live_refresh = st.sidebar.toggle("Live Refresh Mode")

symbols = {"NIFTY 50": 256265, "BANK NIFTY": 260105, "RELIANCE": 738561, "TCS": 295321}
sym_name = st.sidebar.selectbox("Symbol", list(symbols.keys()))
tf = st.sidebar.selectbox("Timeframe", ["1 Minute", "5 Minute", "15 Minute", "1 Hour"])

if st.sidebar.button("Execute Unified Analysis") or live_refresh:
    s1 = Section1_Data_Structure(api_key, token)
    df = s1.fetch_data(symbols[sym_name], tf)
    if df is not None:
        # EXECUTE IN ROADMAP ORDER
        df, m_trend = s1.calculate_structure(df)        # 1
        levels = Section2_S_R().calculate(df)           # 2
        df = Section3_Volume().calculate(df)            # 3
        df = Section4_5_Momentum_Volatility().calculate(df) # 4 & 5
        df = Section6_Trend().calculate(df)             # 6
        df = Section7_PriceAction().recognize(df)       # 7
        df = Section8_CandlestickPatterns().detect(df)  # 8 (NEW)
        
        row = df.iloc[-1]
        st.markdown(f"### 🛡️ Unified Roadmap Dashboard: {sym_name}")
        c_main, c_sig = st.columns([2, 1])
        with c_main:
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
            fig.update_layout(height=450, template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with c_sig:
            st.markdown("##### 🚦 Trend Signals (Section 6)")
            sig_df = pd.DataFrame([
                ["EMA 9 vs 22", "BUY" if row['ema9'] > row['ema22'] else "SELL"],
                ["EMA 200 Bias", "BUY" if row['close'] > row['ema200'] else "SELL"],
                ["Super Trend", "BUY" if row['st_bull'] else "SELL"],
                ["DMI Cross", "BUY" if row['dmi_p'] > row['dmi_m'] else "SELL"],
                ["PSAR Signal", "BUY" if row['close'] > row['psar'] else "SELL"]
            ], columns=["Indicator", "Verdict"])
            st.table(sig_df.style.map(lambda x: 'color: green' if x == 'BUY' else 'color: red', subset=['Verdict']))

        tabs = st.tabs(["Structure", "S/R Levels", "Price Action (7)", "Candlestick Patterns (8)", "Vol & Mom", "Trend Details"])
        with tabs[0]: st.write(f"Trend: {m_trend}"); st.dataframe(df[df['break'] != ""].tail(5))
        with tabs[1]: st.json(levels)
        with tabs[2]:
            st.write("**Detected Price Action (Section 7)**")
            pa_list = ['Inside Bar', 'Outside Bar', 'Marubozu', 'Three White Soldiers', 'Three Black Crows']
            st.dataframe(df[df[pa_list].any(axis=1)][pa_list].tail(10))
        with tabs[3]:
            st.write("**Candlestick Pattern Recognition (Section 8)**")
            bull_p = ['Bullish Engulfing', 'Hammer', 'Morning Star', 'Piercing Pattern']
            bear_p = ['Bearish Engulfing', 'Shooting Star', 'Evening Star', 'Dark Cloud Cover']
            neut_p = ['Doji', 'Spinning Top']
            
            c1, c2, c3 = st.columns(3)
            with c1: st.success("Bullish"); st.write([p for p in bull_p if row[p]])
            with c2: st.error("Bearish"); st.write([p for p in bear_p if row[p]])
            with c3: st.info("Neutral"); st.write([p for p in neut_p if row[p]])
            st.dataframe(df[df[bull_p + bear_p + neut_p].any(axis=1)][bull_p + bear_p + neut_p].tail(10))
        with tabs[4]: st.write(f"RSI: {round(row['rsi'],1)} | ATR: {round(row['atr'],2)}")
        with tabs[5]: st.write(f"ADX: {round(row['adx'],1)} | HMA 20: {round(row['hma20'],2)}")

    if live_refresh:
        time.sleep(10)
        st.rerun()
