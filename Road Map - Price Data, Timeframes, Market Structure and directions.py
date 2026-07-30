import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
from scipy.signal import find_peaks
import time

# ==========================================
# 1. PRICE DATA (OHLCV) & MARKET STRUCTURE
# ==========================================
class Section1_Engine:
    def __init__(self, api_key, access_token):
        self.kite = KiteConnect(api_key=api_key.strip())
        self.kite.set_access_token(access_token.strip())
        self.tf_map = {"1 Minute": "minute", "3 Minute": "3minute", "5 Minute": "5minute", "15 Minute": "15minute", "30 Minute": "30minute", "1 Hour": "60minute", "Daily": "day"}
        self.limits = {"1 Minute": 30, "3 Minute": 60, "5 Minute": 60, "15 Minute": 90, "30 Minute": 180, "1 Hour": 180, "Daily": 365}

    def fetch_data(self, token, tf):
        days = self.limits.get(tf, 30)
        try:
            records = self.kite.historical_data(token, (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S"), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.tf_map.get(tf))
            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
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
        df['trend_strength'] = df['close'].diff(5).rolling(10).mean()
        return df, trend

# ==========================================
# 2. SUPPORT & RESISTANCE
# ==========================================
class Section2_Engine:
    def calculate(self, df, kite, token):
        hist = kite.historical_data(token, (datetime.now()-timedelta(days=60)), datetime.now(), "day")
        df_h = pd.DataFrame(hist)
        df_h['date'] = pd.to_datetime(df_h['date'])
        df_h.set_index('date', inplace=True)
        pdh, pdl = df_h['high'].iloc[-2], df_h['low'].iloc[-2]
        lc = df['close'].iloc[-1]
        pp = (pdh + pdl + lc) / 3
        counts, bins = np.histogram(df['close'], bins=30, weights=df['volume'])
        poc = bins[np.argmax(counts)]
        return {"PDH": pdh, "PDL": pdl, "PP": pp, "POC": poc}

# ==========================================
# 3. VOLUME ANALYSIS
# ==========================================
class Section3_Engine:
    def calculate(self, df):
        df['avg_vol'] = df['volume'].rolling(20).mean()
        df['rel_vol'] = df['volume'] / df['avg_vol']
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        return df

# ==========================================
# 4. MOMENTUM INDICATORS
# ==========================================
class Section4_Engine:
    def calculate(self, df):
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['roc'] = ta.momentum.ROCIndicator(df['close']).roc()
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        return df

# ==========================================
# 5. VOLATILITY INDICATORS
# ==========================================
class Section5_Engine:
    def calculate(self, df):
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_h'], df['bb_l'] = bb.bollinger_hband(), bb.bollinger_lband()
        df['std_dev'] = df['close'].rolling(20).std()
        return df

# ==========================================
# 6. TREND INDICATORS
# ==========================================
class Section6_Engine:
    def calculate(self, df):
        for p in [9, 22, 52, 100, 200]:
            df[f'ema{p}'] = ta.trend.EMAIndicator(df['close'], window=p).ema_indicator()
        adx_obj = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx_obj.adx()
        df['psar'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 10).average_true_range()
        df['st_upper'] = ((df['high']+df['low'])/2) + (3 * atr)
        df['st_bull'] = df['close'] > df['st_upper'].shift(1)
        return df

# ==========================================
# 7 & 8. PRICE ACTION & CANDLESTICK DETECTION
# ==========================================
class Section7_8_Engine:
    def detect(self, df):
        b = df['close'] - df['open']
        ab = abs(b)
        df['Bullish Engulfing'] = (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1)) & (b.shift(1) < 0)
        df['Doji'] = ab <= ((df['high'] - df['low']) * 0.1)
        return df

# ==========================================
# 9. CHART PATTERN DETECTION
# ==========================================
class Section9_Engine:
    def detect(self, df):
        peaks = df[df['sw_h'] > 0]['sw_h'].values
        df['Double Top'] = False
        if len(peaks) >= 2 and abs(peaks[-1] - peaks[-2])/peaks[-1] < 0.002:
            df.iloc[-1, df.columns.get_loc('Double Top')] = True
        return df

# ==========================================
# 10. SMART MONEY CONCEPTS (SMC) - NEW
# ==========================================
class Section10_Engine:
    def calculate_smc(self, df):
        df = df.copy()
        
        # 1. Fair Value Gap (FVG)
        df['FVG_Bull'] = (df['low'] > df['high'].shift(2))
        df['FVG_Bear'] = (df['high'] < df['low'].shift(2))
        
        # 2. Order Blocks (OB)
        # Bullish OB: Last bearish candle before a Bullish BOS
        df['OrderBlock_Bull'] = (df['break'] == "BOS") & (df['close'] > df['open'].shift(1))
        # Bearish OB: Last bullish candle before a Bearish BOS
        df['OrderBlock_Bear'] = (df['break'] == "BOS") & (df['close'] < df['open'].shift(1))
        
        # 3. Liquidity Grab / Sweeps
        df['Liq_Grab_High'] = (df['high'] > df['sw_h'].shift(1)) & (df['close'] < df['sw_h'].shift(1))
        df['Liq_Grab_Low'] = (df['low'] < df['sw_l'].shift(1)) & (df['close'] > df['sw_l'].shift(1))
        
        # 4. Equal Highs (EQH) / Equal Lows (EQL)
        df['EQH'] = abs(df['high'] - df['high'].shift(1)) / df['high'] < 0.0005
        df['EQL'] = abs(df['low'] - df['low'].shift(1)) / df['low'] < 0.0005
        
        # 5. Premium & Discount Zones (Based on current swing range)
        max_r = df['high'].rolling(50).max()
        min_r = df['low'].rolling(50).min()
        mid_point = (max_r + min_r) / 2
        df['Zone'] = np.where(df['close'] > mid_point, "Premium", "Discount")
        
        # 6. Breaker & Mitigation Blocks (Simplified logic)
        df['Breaker_Bull'] = (df['close'] > df['sw_h'].shift(5)) & (df['st_bull'] == True)
        
        # 7. Liquidity Pools (Areas near major swings)
        df['Liq_Pool'] = df['sw_h'].ffill()
        
        return df

# ==========================================
# MAIN UI & EXECUTION
# ==========================================
st.set_page_config(layout="wide", page_title="Master Bot Sections 1-10")
st.sidebar.header("🔑 Kite Configuration")
key = st.sidebar.text_input("API Key", type="password")
token = st.sidebar.text_input("Access Token", type="password")
live = st.sidebar.toggle("Live Refresh Mode")

symbols = {"NIFTY 50": 256265, "BANK NIFTY": 260105, "RELIANCE": 738561, "TCS": 295321}
sym_name = st.sidebar.selectbox("Symbol", list(symbols.keys()))
tf = st.sidebar.selectbox("Timeframe", ["1 Minute", "5 Minute", "15 Minute", "1 Hour"])

if st.sidebar.button("Run Master Analysis") or live:
    eng1 = Section1_Engine(key, token)
    df = eng1.fetch_data(symbols[sym_name], tf)
    if df is not None:
        # EXECUTE IN ORDER 1-10
        df, m_trend = eng1.calculate_structure(df)        # 1
        levels = Section2_Engine().calculate(df, eng1.kite, symbols[sym_name]) # 2
        df = Section3_Engine().calculate(df)               # 3
        df = Section4_Engine().calculate(df)               # 4
        df = Section5_Engine().calculate(df)               # 5
        df = Section6_Engine().calculate(df)               # 6
        df = Section7_8_Engine().detect(df)                # 7 & 8
        df = Section9_Engine().detect(df)                  # 9
        df = Section10_Engine().calculate_smc(df)          # 10 (NEW)
        
        row = df.iloc[-1]
        st.markdown(f"### 🛡️ Master Roadmap Dashboard: {sym_name} ({tf})")
        
        c_main, c_sig = st.columns([2, 1])
        with c_main:
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
            fig.update_layout(height=450, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
            
        with c_sig:
            st.markdown("##### 🚦 SMC & Trend Signals")
            sig_df = pd.DataFrame([
                ["Trend", m_trend],
                ["Market Zone", row['Zone']],
                ["SMC FVG", "BULLISH" if row['FVG_Bull'] else "BEARISH" if row['FVG_Bear'] else "NONE"],
                ["Order Block", "DETECTED" if (row['OrderBlock_Bull'] or row['OrderBlock_Bear']) else "NONE"],
                ["Liquidity Grab", "YES" if (row['Liq_Grab_High'] or row['Liq_Grab_Low']) else "NO"]
            ], columns=["Component", "Verdict"])
            st.table(sig_df)

        tabs = st.tabs(["Structure/SMC", "Levels", "Volume/Mom", "Patterns", "Advanced Log"])
        with tabs[0]:
            st.write(f"**Current State:** {m_trend} | **Zone:** {row['Zone']}")
            st.write("**Recent SMC Events (FVG/OB/Breakers):**")
            st.write(df[(df['FVG_Bull']) | (df['OrderBlock_Bull']) | (df['Liq_Grab_High'])].tail(10))
        with tabs[1]:
            st.json(levels)
        with tabs[2]:
            st.write(f"RSI: {round(row['rsi'],1)} | OBV: {row['obv']} | MFI: {round(row['mfi'],1)}")
        with tabs[3]:
            pats = [p for p in ['Bullish Engulfing','Doji','Double Top','EQH','EQL'] if row[p]]
            st.write("Active Patterns:", pats if pats else "None")
        with tabs[4]:
            st.dataframe(df.tail(20))

    if live:
        time.sleep(10)
        st.rerun()
