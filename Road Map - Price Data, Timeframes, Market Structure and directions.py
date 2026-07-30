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
class Section1_MarketStructure:
    def __init__(self, api_key, access_token):
        self.kite = KiteConnect(api_key=api_key.strip())
        self.kite.set_access_token(access_token.strip())
        self.tf_map = {"1 Minute": "minute", "3 Minute": "3minute", "5 Minute": "5minute", "15 Minute": "15minute", "30 Minute": "30minute", "1 Hour": "60minute", "4 Hour": "60minute", "Daily": "day"}
        self.limits = {"1 Minute": 30, "3 Minute": 60, "5 Minute": 60, "15 Minute": 90, "30 Minute": 100, "1 Hour": 100, "Daily": 365}

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

    def calculate(self, df):
        win = 5
        df['sw_h'] = df['high'][(df['high'] == df['high'].rolling(window=win, center=True).max())]
        df['sw_l'] = df['low'][(df['low'] == df['low'].rolling(window=win, center=True).min())]
        df['HH'], df['HL'], df['LH'], df['LL'] = False, False, False, False
        df['BOS'], df['CHOCH'] = False, False
        l_sh, l_sl, trend = 0, 0, "Sideways"
        for i in range(win, len(df)):
            if not np.isnan(df['sw_h'].iloc[i]):
                h = df['sw_h'].iloc[i]
                if h > l_sh: 
                    df.at[df.index[i], 'HH'] = True
                    if trend == "Bearish": df.at[df.index[i], 'CHOCH'] = True
                    trend = "Bullish"
                else: df.at[df.index[i], 'LH'] = True
                l_sh = h
            if not np.isnan(df['sw_l'].iloc[i]):
                l = df['sw_l'].iloc[i]
                if l < l_sl or l_sl == 0:
                    df.at[df.index[i], 'LL'] = True
                    if trend == "Bullish": df.at[df.index[i], 'CHOCH'] = True
                    trend = "Bearish"
                else: df.at[df.index[i], 'HL'] = True
                l_sl = l
            if trend == "Bullish" and df['close'].iloc[i] > l_sh and l_sh != 0: df.at[df.index[i], 'BOS'] = True
            if trend == "Bearish" and df['close'].iloc[i] < l_sl and l_sl != 0: df.at[df.index[i], 'BOS'] = True
        df['trend_strength'] = abs(df['close'].diff(5).rolling(10).mean())
        return df, trend

# ==========================================
# 2. SUPPORT & RESISTANCE
# ==========================================
class Section2_SupportResistance:
    def calculate(self, df, kite, token):
        # Static
        h_d = pd.DataFrame(kite.historical_data(token, datetime.now()-timedelta(days=60), datetime.now(), "day"))
        pdh, pdl = h_d['high'].iloc[-2], h_d['low'].iloc[-2]
        wh, wl = h_d.set_index(pd.to_datetime(h_d['date'])).resample('W').max()['high'].iloc[-2], h_d.set_index(pd.to_datetime(h_d['date'])).resample('W').min()['low'].iloc[-2]
        # Advanced (Pivots/CPR)
        pp = (pdh + pdl + df['close'].iloc[-1]) / 3
        bc, tc = (pdh + pdl) / 2, (pp - (pdh + pdl) / 2) + pp
        # Volume Profile POC
        counts, bins = np.histogram(df['close'], bins=30, weights=df['volume'])
        poc = bins[np.argmax(counts)]
        # Fibonacci
        mx, mn = df['high'].max(), df['low'].min()
        fibs = {"61.8%": mx - (mx-mn)*0.618, "38.2%": mx - (mx-mn)*0.382}
        return {"PDH": pdh, "PDL": pdl, "WH": wh, "WL": wl, "PP": pp, "TC": tc, "BC": bc, "POC": poc, "Fib": fibs}

# ==========================================
# 3. VOLUME ANALYSIS
# ==========================================
class Section3_VolumeAnalysis:
    def calculate(self, df):
        df['avg_vol'] = df['volume'].rolling(20).mean()
        df['rel_vol'] = df['volume'] / df['avg_vol']
        df['vol_spike'] = df['volume'] > (df['avg_vol'] * 2.5)
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
        df['acc_dist'] = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
        df['cvd'] = (np.where(df['close'] > df['open'], df['volume'], -df['volume'])).cumsum()
        return df

# ==========================================
# 4. MOMENTUM INDICATORS
# ==========================================
class Section4_MomentumIndicators:
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
class Section5_VolatilityIndicators:
    def calculate(self, df):
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_h'], df['bb_l'] = bb.bollinger_hband(), bb.bollinger_lband()
        kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
        df['kc_h'], df['kc_l'] = kc.keltner_channel_hband(), kc.keltner_channel_lband()
        dc = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'])
        df['dc_h'], df['dc_l'] = dc.donchian_channel_hband(), dc.donchian_channel_lband()
        df['std_dev'] = df['close'].rolling(20).std()
        df['choppiness'] = 100 * np.log10(df['atr'].rolling(14).sum() / (df['high'].rolling(14).max() - df['low'].rolling(14).min())) / np.log10(14)
        return df

# ==========================================
# 6. TREND INDICATORS
# ==========================================
class Section6_TrendIndicators:
    def _wma(self, s, p): return s.rolling(p).apply(lambda x: np.dot(x, np.arange(1,p+1))/np.arange(1,p+1).sum(), raw=True)
    def calculate(self, df):
        for p in [9, 20, 50, 100, 200]: df[f'ema{p}'] = ta.trend.EMAIndicator(df['close'], window=p).ema_indicator()
        df['vwma'] = (df['close']*df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['hma20'] = self._wma(2*self._wma(df['close'], 10) - self._wma(df['close'], 20), 4)
        adx_obj = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'], df['dmi_p'], df['dmi_m'] = adx_obj.adx(), adx_obj.adx_pos(), adx_obj.adx_neg()
        df['psar'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()
        ichi = ta.trend.IchimokuIndicator(df['high'], df['low'])
        df['ichi_base'] = ichi.ichimoku_base_line()
        df['st_bull'] = df['close'] > (((df['high']+df['low'])/2) + (3 * ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 10).average_true_range())).shift(1)
        return df

# ==========================================
# 7. PRICE ACTION
# ==========================================
class Section7_PriceAction:
    def detect(self, df):
        b, ab = df['close']-df['open'], abs(df['close']-df['open'])
        df['Inside Bar'] = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
        df['Outside Bar'] = (df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))
        df['Pin Bar'] = (df['high']-df[['open','close']].max(axis=1) > ab*2) | (df[['open','close']].min(axis=1)-df['low'] > ab*2)
        df['Marubozu'] = ab > (df['high']-df['low'])*0.9
        return df

# ==========================================
# 8. CANDLESTICK PATTERNS
# ==========================================
class Section8_CandlestickPatterns:
    def detect(self, df):
        b = df['close'] - df['open']
        df['Bullish Engulfing'] = (df['close']>df['open'].shift(1)) & (df['open']<df['close'].shift(1)) & (b.shift(1)<0)
        df['Bearish Engulfing'] = (df['close']<df['open'].shift(1)) & (df['open']>df['close'].shift(1)) & (b.shift(1)>0)
        df['Hammer'] = ((df[['open','close']].min(axis=1)-df['low']) > abs(b)*2) & (df['high']-df[['open','close']].max(axis=1) < abs(b)*0.5)
        df['Doji'] = abs(b) <= (df['high']-df['low'])*0.1
        return df

# ==========================================
# 9. CHART PATTERNS
# ==========================================
class Section9_ChartPatterns:
    def detect(self, df):
        df['Double Top'] = (df['HH'].shift(1)) & (abs(df['high'] - df['sw_h'].shift(1))/df['high'] < 0.001)
        df['Flag'] = (df['atr'] < df['atr'].rolling(20).mean()*0.8)
        return df

# ==========================================
# 10. SMART MONEY CONCEPTS (SMC)
# ==========================================
class Section10_SMC:
    def calculate(self, df):
        df['FVG_UP'] = (df['low'] > df['high'].shift(2))
        df['OB_Bull'] = (df['BOS']) & (df['close'] > df['open'].shift(1))
        df['EQH'] = abs(df['high'] - df['high'].shift(1))/df['high'] < 0.0003
        mx, mn = df['high'].rolling(50).max(), df['low'].rolling(50).min()
        df['SMC_Zone'] = np.where(df['close'] > (mx+mn)/2, "Premium", "Discount")
        return df

# ==========================================
# 11. ICT CONCEPTS
# ==========================================
class Section11_ICT:
    def calculate(self, df):
        df['ICT_KillZone'] = (df.index.hour == 9) & (df.index.minute >= 15) & (df.index.minute <= 10.30)
        mx, mn = df['high'].rolling(40).max(), df['low'].rolling(40).min()
        df['OTE'] = (df['close'] >= mn + (mx-mn)*0.62) & (df['close'] <= mn + (mx-mn)*0.79)
        return df

# ==========================================
# MAIN DASHBOARD
# ==========================================
st.set_page_config(layout="wide", page_title="Master Bot Sections 1-11")
st.sidebar.header("🔑 Kite Access")
key = st.sidebar.text_input("API Key", type="password")
tok = st.sidebar.text_input("Access Token", type="password")
live = st.sidebar.toggle("Live Refresh")

symbols = {"NIFTY 50": 256265, "BANK NIFTY": 260105, "RELIANCE": 738561, "TCS": 295321}
sym_name = st.sidebar.selectbox("Symbol", list(symbols.keys()))
tf = st.sidebar.selectbox("Timeframe", ["1 Minute", "5 Minute", "15 Minute", "1 Hour"])

if st.sidebar.button("Execute Unified Analysis") or live:
    s1 = Section1_MarketStructure(key, tok)
    df = s1.fetch_data(symbols[sym_name], tf)
    if df is not None:
        # EXECUTE SECTIONS 1 TO 11 INDIVIDUALLY
        df, trend = s1.calculate(df)
        lvls = Section2_SupportResistance().calculate(df, s1.kite, symbols[sym_name])
        df = Section3_VolumeAnalysis().calculate(df)
        df = Section4_MomentumIndicators().calculate(df)
        df = Section5_VolatilityIndicators().calculate(df)
        df = Section6_TrendIndicators().calculate(df)
        df = Section7_PriceAction().detect(df)
        df = Section8_CandlestickPatterns().detect(df)
        df = Section9_ChartPatterns().detect(df)
        df = Section10_SMC().calculate(df)
        df = Section11_ICT().calculate(df)

        row = df.iloc[-1]
        st.markdown(f"### 🛡️ Dashboard: {sym_name} | Signal: {'BUY' if row['ema9']>row['ema20'] else 'SELL'}")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
            fig.update_layout(height=450, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("##### 🚦 Verdicts")
            ver_df = pd.DataFrame([
                ["Structure", trend], ["Zone", row['SMC_Zone']], ["OTE", "YES" if row['OTE'] else "NO"]
            ], columns=["Component", "Verdict"])
            st.table(ver_df.style.map(lambda x: 'color: green' if x in ['YES','Bullish','Discount'] else 'color: red' if x in ['NO','Bearish','Premium'] else ''))

        tabs = st.tabs(["1-2 Structure/Levels", "3-5 Vol/Mom/Volat", "6 Trend", "7-9 Patterns", "10-11 SMC/ICT"])
        with tabs[0]: st.json(lvls)
        with tabs[1]: st.write(f"RSI: {round(row['rsi'],1)} | ATR: {round(row['atr'],2)} | RelVol: {round(row['rel_vol'],2)}")
        with tabs[2]: st.write(f"ADX: {round(row['adx'],1)} | HMA 20: {round(row['hma20'],2)} | Ichimoku Base: {round(row['ichi_base'],2)}")
        with tabs[3]:
            # Using exact names from class logic to avoid KeyError
            active = [p for p in ['Inside Bar','Pin Bar','Bullish Engulfing','Double Top','Flag'] if p in row and row[p]]
            st.write("Active Patterns:", active if active else "None")
        with tabs[4]: st.write(f"SMC Zone: {row['SMC_Zone']} | FVG: {row['FVG_UP']} | ICT OTE: {row['OTE']}")

    if live:
        time.sleep(10)
        st.rerun()
