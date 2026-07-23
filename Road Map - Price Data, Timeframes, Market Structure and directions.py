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
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 10).average_true_range()
        df['st_upper'] = ((df['high']+df['low'])/2) + (3 * atr)
        df['st_bull'] = df['close'] > df['st_upper'].shift(1)
        return df

# ==========================================
# 4. SUPPORT & RESISTANCE - Roadmap Section 2
# ==========================================
class SupportResistanceEngine:
    def __init__(self, kite_instance):
        self.kite = kite_instance

    def get_static_levels(self, token):
        """Calculates PDH/L, Weekly H/L, Monthly H/L"""
        # Fetch 60 days of Daily data to calculate Previous Day, Week, and Month
        d_data = self.kite.historical_data(token, (datetime.now()-timedelta(days=60)), datetime.now(), "day")
        if not d_data: return None
        df_d = pd.DataFrame(d_data)
        
        # PDH/PDL
        pdh, pdl = df_d['high'].iloc[-2], df_d['low'].iloc[-2]
        
        df_d['date'] = pd.to_datetime(df_d['date'])
        df_d.set_index('date', inplace=True)
        
        # FIX: 'W' becomes 'W-SUN' and 'M' becomes 'ME' (Month End) for compatibility
        df_w = df_d.resample('W-SUN').agg({'high':'max', 'low':'min'})
        df_m = df_d.resample('ME').agg({'high':'max', 'low':'min'})
        
        return {
            "PDH": pdh, "PDL": pdl,
            "WH": df_w['high'].iloc[-2], "WL": df_w['low'].iloc[-2],
            "MH": df_m['high'].iloc[-2], "ML": df_m['low'].iloc[-2]
        }

    def calculate_advanced_levels(self, df):
        """Pivot Points, Camarilla, CPR, Fibonacci"""
        last_h = df['high'].iloc[-1]
        last_l = df['low'].iloc[-1]
        last_c = df['close'].iloc[-1]
        rng = last_h - last_l

        # Pivot Points (Standard)
        pp = (last_h + last_l + last_c) / 3
        r1 = (2 * pp) - last_l
        s1 = (2 * pp) - last_h

        # CPR
        bc = (last_h + last_l) / 2
        tc = (pp - bc) + pp
        
        # Camarilla
        h4 = last_c + (rng * 1.1 / 2)
        h3 = last_c + (rng * 1.1 / 4)
        l3 = last_c - (rng * 1.1 / 4)
        l4 = last_c - (rng * 1.1 / 2)

        # Fibonacci (from current view range)
        max_p = df['high'].max()
        min_p = df['low'].min()
        diff = max_p - min_p
        fibs = {
            "23.6%": max_p - (diff * 0.236),
            "38.2%": max_p - (diff * 0.382),
            "50.0%": max_p - (diff * 0.5),
            "61.8%": max_p - (diff * 0.618),
            "78.6%": max_p - (diff * 0.786)
        }

        return {"PP": pp, "TC": tc, "BC": bc, "H4": h4, "H3": h3, "L3": l3, "L4": l4, "Fibs": fibs}

    def get_volume_profile(self, df, bins=30):
        """Simple Volume Profile calculation for POC"""
        # Distribute volume across price bins
        price_range = np.linspace(df['low'].min(), df['high'].max(), bins)
        v_profile = df.groupby(pd.cut(df['close'], bins=price_range))['volume'].sum()
        poc_bin = v_profile.idxmax()
        poc = (poc_bin.left + poc_bin.right) / 2
        return poc

# ==========================================
# MAIN STREAMLIT UI
# ==========================================
st.set_page_config(layout="wide", page_title="Unified Roadmap Dashboard")
st.title("🛡️ Roadmap Section 1 & 2: Structure, Trend & S/R")

st.sidebar.header("🔑 Kite API Provision")
in_api_key = st.sidebar.text_input("Kite API Key", value="", type="password")
in_access_token = st.sidebar.text_input("Access Token", value="", type="password")

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
        data_eng = PriceDataEngine(in_api_key, in_access_token)
        success, msg = data_eng.test_connection()
        
        if not success:
            st.error(msg)
        else:
            st.sidebar.success(msg)
            df = data_eng.fetch_ohlcv(symbols[sym], tf)
            
            if df is not None:
                # SECTION 1 Calculation
                struct_eng = MarketStructureEngine()
                df, s_trend = struct_eng.calculate_structure(df)
                trend_eng = TrendDirectionEngine()
                df = trend_eng.analyze(df)
                
                # SECTION 2 Calculation
                sr_eng = SupportResistanceEngine(data_eng.kite)
                static = sr_eng.get_static_levels(symbols[sym])
                advanced = sr_eng.calculate_advanced_levels(df)
                poc = sr_eng.get_volume_profile(df)
                
                # Metrics Display
                row = df.iloc[-1]
                st.subheader(f"Dashboard Analysis: {sym} ({tf})")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Market Structure", s_trend)
                c2.metric("Point of Control (POC)", round(poc, 2))
                c3.metric("Trend Strength (ADX)", round(row['adx'], 2))
                c4.metric("CPR Bias", "Neutral" if abs(advanced['TC']-advanced['BC']) < (row['close']*0.001) else "Wide")

                # Main Plot
                fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
                
                # Visualizing Support/Resistance
                fig.add_hline(y=static['PDH'], line_dash="dash", line_color="green", annotation_text="Prev Day High")
                fig.add_hline(y=static['PDL'], line_dash="dash", line_color="red", annotation_text="Prev Day Low")
                fig.add_hline(y=advanced['PP'], line_color="yellow", line_width=2, annotation_text="Pivot Point")
                fig.add_hline(y=poc, line_color="cyan", line_dash="dot", annotation_text="Volume POC")
                
                fig.update_layout(height=700, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

                # Data Tabs
                t1, t2, t3 = st.tabs(["Static Levels (Highs/Lows)", "Pivot & CPR Details", "Fibonacci Retracement"])
                with t1:
                    st.json(static)
                with t2:
                    st.write("Advanced Levels (Camarilla & CPR)")
                    st.table(pd.DataFrame([advanced]).drop('Fibs', axis=1))
                with t3:
                    st.table(pd.DataFrame.from_dict(advanced['Fibs'], orient='index', columns=['Price Level']))
