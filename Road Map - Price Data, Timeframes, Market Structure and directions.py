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
        # Adhering to your provided Kite plan limits
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
            # Resampling for 4H and Weekly
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
        # Static & Advanced (Pivots, Fib, CPR)
        lh, ll, lc = df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1]
        pp = (lh + ll + lc) / 3
        bc = (lh + ll) / 2
        tc = (pp - bc) + pp
        # Fibonacci
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
        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
        return df

# ==========================================
# 4. MOMENTUM INDICATORS
# ==========================================
class Section4_Momentum:
    def calculate(self, df):
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['stoch_rsi'] = ta.momentum.StochRSIIndicator(df['close']).stochrsi()
        return df

# ==========================================
# 5. VOLATILITY INDICATORS
# ==========================================
class Section5_Volatility:
    def calculate(self, df):
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_h'], df['bb_l'] = bb.bollinger_hband(), bb.bollinger_lband()
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
    def calculate(self, df):
        # EMAs (9, 22, 52, 100, 200)
        for p in [9, 22, 52, 100, 200]:
            df[f'ema{p}'] = ta.trend.EMAIndicator(df['close'], window=p).ema_indicator()
        # SMA & VWMA
        df['sma50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['vwma20'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        # Super Trend
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 10).average_true_range()
        df['st_upper'] = ((df['high']+df['low'])/2) + (3 * atr)
        df['st_bull'] = df['close'] > df['st_upper'].shift(1)
        # ADX & DMI
        adx_obj = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx_obj.adx()
        df['dmi_p'] = adx_obj.adx_pos()
        df['dmi_m'] = adx_obj.adx_neg()
        # Ichimoku & PSAR
        ichi = ta.trend.IchimokuIndicator(df['high'], df['low'])
        df['ichi_base'] = ichi.ichimoku_base_line()
        df['psar'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()
        # HMA (Hull Moving Average)
        def hma(series, period):
            wma1 = ta.trend.WMAIndicator(series, window=period//2).wma_indicator()
            wma2 = ta.trend.WMAIndicator(series, window=period).wma_indicator()
            diff = 2 * wma1 - wma2
            return ta.trend.WMAIndicator(diff, window=int(np.sqrt(period))).wma_indicator()
        df['hma20'] = hma(df['close'], 20)
        return df

# ==========================================
# MAIN DASHBOARD EXECUTION
# ==========================================
st.set_page_config(layout="wide", page_title="Master Roadmap")
st.sidebar.header("🔑 Kite Access")
api_key = st.sidebar.text_input("API Key", type="password")
token = st.sidebar.text_input("Access Token", type="password")
live_refresh = st.sidebar.toggle("Live Refresh Mode")

symbols = {"NIFTY 50": 256265, "BANK NIFTY": 260105, "RELIANCE": 738561, "TCS": 295321}
sym_name = st.sidebar.selectbox("Symbol", list(symbols.keys()))
tf = st.sidebar.selectbox("Timeframe", ["1 Minute", "5 Minute", "15 Minute", "1 Hour", "Daily"])

if st.sidebar.button("Execute Unified Analysis") or live_refresh:
    s1 = Section1_Data_Structure(api_key, token)
    df = s1.fetch_data(symbols[sym_name], tf)
    
    if df is not None:
        # EXECUTE IN ROADMAP ORDER
        df, m_trend = s1.calculate_structure(df) # Sec 1
        levels = Section2_S_R().calculate(df)    # Sec 2
        df = Section3_Volume().calculate(df)     # Sec 3
        df = Section4_Momentum().calculate(df)   # Sec 4
        df = Section5_Volatility().calculate(df) # Sec 5
        df = Section6_Trend().calculate(df)      # Sec 6
        
        row = df.iloc[-1]

        # --- DASHBOARD METRICS ---
        st.markdown(f"### 🛡️ Unified Roadmap Dashboard: {sym_name}")
        
        col_main, col_signal = st.columns([2, 1])
        
        with col_main:
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
            fig.add_trace(go.Scatter(x=df.index, y=df['ema200'], name="EMA 200", line=dict(color='yellow', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['psar'], mode='markers', name="PSAR", marker=dict(size=4, color='white')))
            fig.update_layout(height=450, template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col_signal:
            st.markdown("##### 🚦 Trend Signal Table (Section 6)")
            sig_data = [
                ["EMA 9 vs 22", "BUY" if row['ema9'] > row['ema22'] else "SELL"],
                ["EMA 200 Bias", "BUY" if row['close'] > row['ema200'] else "SELL"],
                ["Super Trend", "BUY" if row['st_bull'] else "SELL"],
                ["DMI Cross", "BUY" if row['dmi_p'] > row['dmi_m'] else "SELL"],
                ["Price vs VWMA", "BUY" if row['close'] > row['vwma20'] else "SELL"],
                ["PSAR", "BUY" if row['close'] > row['psar'] else "SELL"]
            ]
            sig_df = pd.DataFrame(sig_data, columns=["Indicator", "Verdict"])
            def style_v(v): return 'color: green' if v == 'BUY' else 'color: red'
            st.table(sig_df.style.applymap(style_v, subset=['Verdict']))

        # SECTION TABS
        tabs = st.tabs(["Structure", "S/R", "Volume", "Momentum", "Volatility", "Trend Indicators"])
        with tabs[0]: st.write(f"Trend: {m_trend}"); st.dataframe(df[df['break'] != ""].tail(5))
        with tabs[1]: st.json(levels)
        with tabs[2]: st.write(f"MFI: {round(row['mfi'],1)} | OBV: {row['obv']} | CMF: {round(row['cmf'],2)}")
        with tabs[3]: st.write(f"RSI: {round(row['rsi'],1)} | StochRSI: {round(row['stoch_rsi'],2)}")
        with tabs[4]: st.write(f"ATR: {round(row['atr'],2)} | Choppiness: {round(row['choppiness'],1)}")
        with tabs[5]: st.write(f"ADX: {round(row['adx'],1)} | DMI+: {round(row['dmi_p'],1)} | HMA 20: {round(row['hma20'],2)}")

    if live_refresh:
        time.sleep(10)
        st.rerun()
