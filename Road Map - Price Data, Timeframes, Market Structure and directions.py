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
        # HH, HL, LH, LL, Swing High/Low
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
        # Trend Strength
        df['trend_strength'] = df['close'].diff(5).rolling(10).mean()
        return df, trend

# ==========================================
# 2. SUPPORT & RESISTANCE
# ==========================================
class Section2_Engine:
    def calculate(self, df, kite, token):
        # Static: PDH, PDL, Weekly/Monthly H/L
        hist = kite.historical_data(token, (datetime.now()-timedelta(days=60)), datetime.now(), "day")
        df_h = pd.DataFrame(hist)
        df_h['date'] = pd.to_datetime(df_h['date'])
        df_h.set_index('date', inplace=True)
        
        pdh, pdl = df_h['high'].iloc[-2], df_h['low'].iloc[-2]
        wh, wl = df_h.resample('W-SUN').max()['high'].iloc[-2], df_h.resample('W-SUN').min()['low'].iloc[-2]
        mh, ml = df_h.resample('ME').max()['high'].iloc[-2], df_h.resample('ME').min()['low'].iloc[-2]
        
        # Advanced & Volume Profile
        lc = df['close'].iloc[-1]
        pp = (pdh + pdl + lc) / 3
        # Volume Profile POC
        counts, bins = np.histogram(df['close'], bins=30, weights=df['volume'])
        poc = bins[np.argmax(counts)]
        
        return {"PDH": pdh, "PDL": pdl, "WH": wh, "WL": wl, "MH": mh, "ML": ml, "PP": pp, "POC": poc}

# ==========================================
# 3. VOLUME ANALYSIS
# ==========================================
class Section3_Engine:
    def calculate(self, df):
        df['avg_vol'] = df['volume'].rolling(20).mean()
        df['rel_vol'] = df['volume'] / df['avg_vol']
        df['vol_spike'] = df['volume'] > (df['avg_vol'] * 2.5)
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
        df['acc_dist'] = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
        return df

# ==========================================
# 4. MOMENTUM INDICATORS
# ==========================================
class Section4_Engine:
    def calculate(self, df):
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['stoch_rsi'] = ta.momentum.StochRSIIndicator(df['close']).stochrsi()
        df['roc'] = ta.momentum.ROCIndicator(df['close']).roc()
        df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        df['ult_osc'] = ta.momentum.UltimateOscillator(df['high'], df['low'], df['close']).ultimate_oscillator()
        return df

# ==========================================
# 5. VOLATILITY INDICATORS
# ==========================================
class Section5_Engine:
    def calculate(self, df):
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_h'], df['bb_l'] = bb.bollinger_hband(), bb.bollinger_lband()
        kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
        df['kc_h'], df['kc_l'] = kc.keltner_channel_hband(), kc.keltner_channel_lband()
        dc = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'])
        df['dc_h'], df['dc_l'] = dc.donchian_channel_hband(), dc.donchian_channel_lband()
        df['std_dev'] = df['close'].rolling(20).std()
        n = 14
        tr_sum = df['atr'].rolling(n).sum()
        p_range = df['high'].rolling(n).max() - df['low'].rolling(n).min()
        df['choppiness'] = 100 * np.log10(tr_sum / p_range) / np.log10(n)
        return df

# ==========================================
# 6. TREND INDICATORS
# ==========================================
class Section6_Engine:
    def _wma(self, series, period):
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    def calculate(self, df):
        for p in [9, 22, 52, 100, 200]:
            df[f'ema{p}'] = ta.trend.EMAIndicator(df['close'], window=p).ema_indicator()
        df['sma50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['vwma20'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        # HMA
        p = 20
        w1, w2 = self._wma(df['close'], p//2), self._wma(df['close'], p)
        df['hma20'] = self._wma(2 * w1 - w2, int(np.sqrt(p)))
        # Ichimoku
        ichi = ta.trend.IchimokuIndicator(df['high'], df['low'])
        df['ichi_base'], df['ichi_a'], df['ichi_b'] = ichi.ichimoku_base_line(), ichi.ichimoku_a(), ichi.ichimoku_b()
        # ADX/DMI & PSAR
        adx_obj = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'], df['dmi_p'], df['dmi_m'] = adx_obj.adx(), adx_obj.adx_pos(), adx_obj.adx_neg()
        df['psar'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()
        # SuperTrend
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 10).average_true_range()
        df['st_upper'] = ((df['high']+df['low'])/2) + (3 * atr)
        df['st_bull'] = df['close'] > df['st_upper'].shift(1)
        return df

# ==========================================
# 7 & 8. PRICE ACTION & CANDLESTICK DETECTION
# ==========================================
class Section7_8_Engine:
    def detect(self, df):
        b, r = df['close'] - df['open'], df['high'] - df['low']
        ab = abs(b)
        uw, lw = df['high'] - df[['open','close']].max(axis=1), df[['open','close']].min(axis=1) - df['low']
        
        df['Inside Bar'] = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
        df['Bullish Engulfing'] = (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1)) & (b.shift(1) < 0)
        df['Hammer'] = (lw > ab * 2) & (uw < ab * 0.5)
        df['Shooting Star'] = (uw > ab * 2) & (lw < ab * 0.5)
        df['Doji'] = ab <= (r * 0.1)
        df['Spinning Top'] = (ab < r * 0.3) & (uw > ab) & (lw > ab)
        df['Morning Star'] = (b.shift(2) < 0) & (ab.shift(1) < ab.shift(2)*0.3) & (b > 0)
        df['Evening Star'] = (b.shift(2) > 0) & (ab.shift(1) < ab.shift(2)*0.3) & (b < 0)
        df['Marubozu'] = ab > r * 0.9
        return df

# ==========================================
# 9. CHART PATTERN DETECTION
# ==========================================
class Section9_Engine:
    def detect(self, df):
        peaks = df[df['sw_h'] > 0]['sw_h'].values
        troughs = df[df['sw_l'] > 0]['sw_l'].values
        df['Double Top'] = False
        df['Double Bottom'] = False
        if len(peaks) >= 2 and abs(peaks[-1] - peaks[-2])/peaks[-1] < 0.002: df.iloc[-1, df.columns.get_loc('Double Top')] = True
        if len(troughs) >= 2 and abs(troughs[-1] - troughs[-2])/troughs[-1] < 0.002: df.iloc[-1, df.columns.get_loc('Double Bottom')] = True
        
        # Flags / Pennant Detection (Consolidation after pole)
        df['Flag_Consolidation'] = (df['atr'] < df['atr'].rolling(20).mean() * 0.75)
        return df

# ==========================================
# MAIN UI & EXECUTION
# ==========================================
st.set_page_config(layout="wide", page_title="Master Bot Sections 1-9")
st.sidebar.header("🔑 Kite Configuration")
key = st.sidebar.text_input("API Key", type="password")
token = st.sidebar.text_input("Access Token", type="password")
live = st.sidebar.toggle("Live Refresh Mode")

symbols = {"NIFTY 50": 256265, "BANK NIFTY": 260105, "RELIANCE": 738561, "TCS": 295321}
sym_name = st.sidebar.selectbox("Symbol", list(symbols.keys()))
tf = st.sidebar.selectbox("Timeframe", ["1 Minute", "5 Minute", "15 Minute", "1 Hour"])

if st.sidebar.button("Run Unified Analysis") or live:
    eng1 = Section1_Engine(key, token)
    df = eng1.fetch_data(symbols[sym_name], tf)
    if df is not None:
        # EXECUTE IN ROADMAP ORDER
        df, m_trend = eng1.calculate_structure(df)        # Section 1
        levels = Section2_Engine().calculate(df, eng1.kite, symbols[sym_name]) # Section 2
        df = Section3_Engine().calculate(df)               # Section 3
        df = Section4_Engine().calculate(df)               # Section 4
        df = Section5_Engine().calculate(df)               # Section 5
        df = Section6_Engine().calculate(df)               # Section 6
        df = Section7_8_Engine().detect(df)                # Section 7 & 8
        df = Section9_Engine().detect(df)                  # Section 9
        
        row = df.iloc[-1]
        st.markdown(f"### 🛡️ Master Roadmap Dashboard: {sym_name}")
        c_main, c_sig = st.columns([2, 1])
        with c_main:
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
            fig.update_layout(height=450, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
        with c_sig:
            st.markdown("##### 🚦 Trend Verdicts (Section 6)")
            sig_df = pd.DataFrame([
                ["EMA 9 vs 22", "BUY" if row['ema9'] > row['ema22'] else "SELL"],
                ["EMA 200 Bias", "BUY" if row['close'] > row['ema200'] else "SELL"],
                ["Super Trend", "BUY" if row['st_bull'] else "SELL"],
                ["Ichimoku Base", "BUY" if row['close'] > row['ichi_base'] else "SELL"],
                ["DMI Cross", "BUY" if row['dmi_p'] > row['dmi_m'] else "SELL"]
            ], columns=["Indicator", "Verdict"])
            st.table(sig_df.style.map(lambda x: 'color: green' if x=='BUY' else 'color: red', subset=['Verdict']))

        tabs = st.tabs(["1. Structure", "2. Levels", "3. Volume", "4. Momentum", "5. Volatility", "7-9. Patterns"])
        with tabs[0]: st.write(f"Trend: {m_trend}"); st.write(df[df['break'] != ""].tail(5))
        with tabs[1]: st.json(levels)
        with tabs[2]: st.write(f"Rel Vol: {round(row['rel_vol'],2)} | CMF: {round(row['cmf'],2)}")
        with tabs[3]: st.write(f"ROC: {round(row['roc'],2)} | CCI: {round(row['cci'],2)} | Williams: {round(row['williams_r'],2)}")
        with tabs[4]: st.write(f"Choppiness: {round(row['choppiness'],1)} | ATR: {round(row['atr'],2)}")
        with tabs[5]:
            pats = [p for p in ['Hammer','Shooting Star','Doji','Double Top','Double Bottom','Flag_Consolidation'] if row[p]]
            st.write("Active Patterns:", pats if pats else "None")

    if live:
        time.sleep(10)
        st.rerun()
