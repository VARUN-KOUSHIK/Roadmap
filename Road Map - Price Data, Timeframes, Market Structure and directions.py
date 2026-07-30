import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import time
from scipy.signal import find_peaks

# ==========================================
# 1. PRICE DATA (OHLCV) & MARKET STRUCTURE
# ==========================================
class Section1_Data_Structure:
    def __init__(self, api_key, access_token):
        self.kite = KiteConnect(api_key=api_key.strip())
        self.kite.set_access_token(access_token.strip())
        self.tf_map = {"1 Minute": "minute", "3 Minute": "3minute", "5 Minute": "5minute", "15 Minute": "15minute", "30 Minute": "30minute", "1 Hour": "60minute", "4 Hour": "60minute", "Daily": "day"}
        self.limits = {"1 Minute": 30, "3 Minute": 60, "5 Minute": 60, "15 Minute": 90, "30 Minute": 180, "1 Hour": 180, "4 Hour": 180, "Daily": 365}

    def fetch_data(self, token, tf):
        days = self.limits.get(tf, 30)
        try:
            records = self.kite.historical_data(token, (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S"), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.tf_map.get(tf))
            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            if tf == "4 Hour": df = df.resample('4H').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            return df
        except Exception as e:
            st.error(f"Fetch Error: {e}")
            return None

    def calculate_structure(self, df):
        window = 5
        df['sw_h'] = df['high'][(df['high'] == df['high'].rolling(window=window, center=True).max())]
        df['sw_l'] = df['low'][(df['low'] == df['low'].rolling(window=window, center=True).min())]
        df['label'], df['break'] = "", ""
        l_sh, l_sl, trend = 0, 0, "Sideways"
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
# 2. SUPPORT & RESISTANCE (All Levels)
# ==========================================
class Section2_Support_Resistance:
    def calculate(self, df, kite, token):
        # 1. Static Levels
        hist = kite.historical_data(token, (datetime.now()-timedelta(days=60)), datetime.now(), "day")
        df_h = pd.DataFrame(hist)
        df_h['date'] = pd.to_datetime(df_h['date'])
        df_h.set_index('date', inplace=True)
        pdh, pdl = df_h['high'].iloc[-2], df_h['low'].iloc[-2]
        wh, wl = df_h.resample('W-SUN').max()['high'].iloc[-2], df_h.resample('W-SUN').min()['low'].iloc[-2]
        mh, ml = df_h.resample('ME').max()['high'].iloc[-2], df_h.resample('ME').min()['low'].iloc[-2]
        
        # 2. Dynamic (Pivot/CPR)
        pp = (pdh + pdl + df['close'].iloc[-1]) / 3
        bc, tc = (pdh + pdl) / 2, (pp - (pdh + pdl) / 2) + pp
        
        # 3. Fibonacci
        mx, mn = df['high'].max(), df['low'].min()
        fibs = {"61.8%": mx - (mx-mn)*0.618, "50%": mx - (mx-mn)*0.5, "161.8%": mx + (mx-mn)*0.618}
        
        # 4. Volume Profile
        counts, bins = np.histogram(df['close'], bins=50, weights=df['volume'])
        poc = bins[np.argmax(counts)]
        hvn = bins[np.where(counts > np.mean(counts)*2)]
        
        return {"Static": {"PDH":pdh, "PDL":pdl, "WH":wh, "WL":wl}, "Pivots": {"PP":pp, "TC":tc, "BC":bc}, "Fib": fibs, "POC": poc}

# ==========================================
# 3. VOLUME ANALYSIS
# ==========================================
class Section3_Volume:
    def calculate(self, df):
        df['avg_vol'] = df['volume'].rolling(20).mean()
        df['rel_vol'] = df['volume'] / df['avg_vol']
        df['vol_spike'] = df['volume'] > (df['avg_vol'] * 2.5)
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
        df['acc_dist'] = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
        df['vol_delta'] = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
        df['cvd'] = df['vol_delta'].cumsum()
        return df

# ==========================================
# 4. MOMENTUM INDICATORS
# ==========================================
class Section4_Momentum:
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
class Section5_Volatility:
    def calculate(self, df):
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_h'], df['bb_l'] = bb.bollinger_hband(), bb.bollinger_lband()
        kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
        df['kc_h'], df['kc_l'] = kc.keltner_channel_hband(), kc.keltner_channel_lband()
        dc = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'])
        df['dc_h'], df['dc_l'] = dc.donchian_channel_hband(), dc.donchian_channel_lband()
        df['std_dev'] = df['close'].rolling(20).std()
        # Choppiness Index
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
        df['sma50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['vwma20'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        # HMA
        p = 20
        w1, w2 = self._wma(df['close'], p//2), self._wma(df['close'], p)
        df['hma20'] = self._wma(2 * w1 - w2, int(np.sqrt(p)))
        # Ichimoku & PSAR
        ichi = ta.trend.IchimokuIndicator(df['high'], df['low'])
        df['ichi_a'], df['ichi_b'] = ichi.ichimoku_a(), ichi.ichimoku_b()
        df['psar'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()
        # ADX / DMI
        adx_obj = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'], df['dmi_p'], df['dmi_m'] = adx_obj.adx(), adx_obj.adx_pos(), adx_obj.adx_neg()
        # SuperTrend
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 10).average_true_range()
        df['st_bull'] = df['close'] > (((df['high']+df['low'])/2) + (3 * atr)).shift(1)
        return df

# ==========================================
# 7. PRICE ACTION (12 Types)
# ==========================================
class Section7_PriceAction:
    def detect(self, df):
        b, r = df['close'] - df['open'], df['high'] - df['low']
        ab, body_avg = abs(b), abs(b).rolling(10).mean()
        uw = df['high'] - df[['open','close']].max(axis=1)
        lw = df[['open','close']].min(axis=1) - df['low']
        
        df['Inside Bar'] = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
        df['Outside Bar'] = (df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))
        df['Pin Bar'] = (lw > ab * 2) | (uw > ab * 2)
        df['Engulfing'] = (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1)) & (b.shift(1) < 0)
        df['Doji'] = ab <= (r * 0.1)
        df['Hammer'] = (lw > ab * 2) & (uw < ab * 0.5)
        df['Shooting Star'] = (uw > ab * 2) & (lw < ab * 0.5)
        df['Marubozu'] = ab > (r * 0.9)
        df['Morning Star'] = (b.shift(2) < 0) & (ab.shift(1) < ab.shift(2)*0.3) & (b > 0)
        df['Evening Star'] = (b.shift(2) > 0) & (ab.shift(1) < ab.shift(2)*0.3) & (b < 0)
        df['3 White Soldiers'] = (b > 0) & (b.shift(1) > 0) & (b.shift(2) > 0)
        df['3 Black Crows'] = (b < 0) & (b.shift(1) < 0) & (b.shift(2) < 0)
        return df

# ==========================================
# 8. CANDLESTICK PATTERN DETECTION
# ==========================================
class Section8_CandlestickDetection:
    def detect(self, df):
        b = df['close'] - df['open']
        # Bullish
        df['Piercing Pattern'] = (b.shift(1) < 0) & (df['open'] < df['low'].shift(1)) & (df['close'] > (df['open'].shift(1)+df['close'].shift(1))/2)
        # Bearish
        df['Dark Cloud Cover'] = (b.shift(1) > 0) & (df['open'] > df['high'].shift(1)) & (df['close'] < (df['open'].shift(1)+df['close'].shift(1))/2)
        # Neutral
        df['Spinning Top'] = (abs(b) < (df['high']-df['low'])*0.3) & (df['high']-df[['open','close']].max(axis=1) > abs(b))
        return df

# ==========================================
# 9. CHART PATTERN DETECTION
# ==========================================
class Section9_ChartPatterns:
    def detect(self, df):
        # Logic using swing points (Section 1)
        peaks = df[df['sw_h'] > 0]['sw_h'].values
        troughs = df[df['sw_l'] > 0]['sw_l'].values
        df['Double Top'] = False
        if len(peaks) >= 2 and abs(peaks[-1]-peaks[-2])/peaks[-1] < 0.002: df.iloc[-1, df.columns.get_loc('Double Top')] = True
        
        # Triangles / Wedges (Slope based)
        df['Consolidation'] = df['atr'] < df['atr'].rolling(20).mean() * 0.8
        df['Rounded Bottom'] = (df['close'].rolling(20).min() == df['close'].rolling(40).min()) & (df['close'] > df['close'].shift(1))
        return df

# ==========================================
# 10. SMART MONEY CONCEPTS (SMC)
# ==========================================
class Section10_SMC:
    def calculate(self, df):
        df['FVG_UP'] = (df['low'] > df['high'].shift(2))
        df['OB_Bull'] = (df['break'] == "BOS") & (df['close'] > df['open'].shift(1))
        df['Liq_Sweep'] = (df['high'] > df['sw_h'].shift(1)) & (df['close'] < df['sw_h'].shift(1))
        mx, mn = df['high'].rolling(50).max(), df['low'].rolling(50).min()
        df['Zone'] = np.where(df['close'] > (mx+mn)/2, "Premium", "Discount")
        return df

# ==========================================
# 11. ICT CONCEPTS
# ==========================================
class Section11_ICT:
    def calculate(self, df):
        df['time'] = df.index.strftime('%H:%M')
        df['KillZone'] = (df['time'] >= '09:15') & (df['time'] <= '10:30')
        mx, mn = df['high'].rolling(40).max(), df['low'].rolling(40).min()
        df['OTE'] = (df['close'] >= mn + (mx-mn)*0.62) & (df['close'] <= mn + (mx-mn)*0.79)
        return df

# ==========================================
# MAIN DASHBOARD
# ==========================================
st.set_page_config(layout="wide", page_title="Master Roadmap v2")
st.sidebar.header("🔑 Kite Access")
key = st.sidebar.text_input("API Key", type="password")
token = st.sidebar.text_input("Access Token", type="password")
live = st.sidebar.toggle("Live Refresh")

symbols = {"NIFTY 50": 256265, "BANK NIFTY": 260105, "RELIANCE": 738561, "TCS": 295321}
sym_name = st.sidebar.selectbox("Symbol", list(symbols.keys()))
tf = st.sidebar.selectbox("Timeframe", ["1 Minute", "5 Minute", "15 Minute", "1 Hour"])

if st.sidebar.button("Execute Unified Analysis") or live:
    eng1 = Section1_Data_Structure(key, token)
    df = eng1.fetch_data(symbols[sym_name], tf)
    if df is not None:
        # EXECUTE IN STRICT SEQUENTIAL ORDER (1 TO 11)
        df, trend = eng1.calculate_structure(df)        # 1
        lvls = Section2_Support_Resistance().calculate(df, eng1.kite, symbols[sym_name]) # 2
        df = Section3_Volume().calculate(df)             # 3
        df = Section4_Momentum().calculate(df)           # 4
        df = Section5_Volatility().calculate(df)         # 5
        df = Section6_Trend().calculate(df)              # 6
        df = Section7_PriceAction().detect(df)           # 7
        df = Section8_CandlestickDetection().detect(df) # 8
        df = Section9_ChartPatterns().detect(df)        # 9
        df = Section10_SMC().calculate(df)               # 10
        df = Section11_ICT().calculate(df)               # 11

        row = df.iloc[-1]
        st.markdown(f"### 🛡️ Unified Roadmap Dashboard: {sym_name}")
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
            fig.update_layout(height=450, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("##### 🚦 Result Verdict (Sec 6, 10, 11)")
            sig_df = pd.DataFrame([
                ["Structure", trend], ["SMC Zone", row['Zone']],
                ["ICT KillZone", "ACTIVE" if row['KillZone'] else "OFF"],
                ["EMA 9/22", "BUY" if row['ema9'] > row['ema22'] else "SELL"],
                ["OTE Entry", "YES" if row['OTE'] else "NO"]
            ], columns=["Component", "Verdict"])
            st.table(sig_df.style.map(lambda x: 'color: green' if x in ['BUY','ACTIVE','Bullish','Discount','YES'] else 'color: red' if x in ['SELL','Bearish','Premium','NO'] else ''))

        t = st.tabs(["1-2 Structure/Lvls", "3-5 Indicators", "6 Trend", "7-9 Patterns", "10-11 SMC/ICT"])
        with t[0]: st.json(lvls)
        with t[1]: st.write(f"RSI: {round(row['rsi'],1)} | ATR: {round(row['atr'],2)} | CVD: {round(row['cvd'],0)}")
        with t[2]: st.write(f"ADX: {round(row['adx'],1)} | HMA 20: {round(row['hma20'],2)} | PSAR: {round(row['psar'],2)}")
        with t[3]: 
            st.write("Active Price Action/Patterns:", [p for p in ['Inside Bar','Pin Bar','Bullish Engulfing','Double Top','Hammer','Rounded Bottom'] if row[p]])
        with t[4]: st.write(f"SMC Zone: {row['Zone']} | ICT KillZone: {row['KillZone']} | OTE: {row['OTE']}")

    if live:
        time.sleep(10)
        st.rerun()
