import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from kiteconnect import KiteConnect
from datetime import datetime, timedelta

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
            return True, f"Connected! Welcome, {profile.get('user_name')}"
        except Exception as e:
            return False, f"Connection Failed: {str(e)}"

    def fetch_ohlcv(self, symbol_token, timeframe):
        # FIX: Strict adherence to Kite Historical Data limits for Basic Plan
        # Interval: Max Days
        # Minute: 30-60 days (Safe: 30)
        # 3/5/10/15 Minute: 90-180 days (Safe: 90)
        # 30/60 Minute: 180-365 days (Safe: 180)
        # Day: 2000 days
        
        limits = {
            "1 Minute": 30,
            "3 Minute": 90,
            "5 Minute": 90,
            "15 Minute": 90,
            "30 Minute": 180,
            "1 Hour": 180,
            "4 Hour": 180,    # Resampled from 60min
            "Daily": 1000,    # Safe within 2000 limit
            "Weekly": 1000    # Resampled from Day
        }
        
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
            
            if timeframe == "4 Hour":
                df = df.resample('4H').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            elif timeframe == "Weekly":
                df = df.resample('W-MON').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            return df
        except Exception as e:
            st.error(f"Kite Fetch Error for {timeframe}: {e}")
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
# 3. TREND DIRECTION (Dynamic Levels) - Roadmap Section 1 & 2
# ==========================================
class TrendDirectionEngine:
    def analyze(self, df):
        df['ema50'] = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
        df['ema200'] = ta.trend.EMAIndicator(df['close'], 200).ema_indicator()
        df['sma50'] = ta.trend.SMAIndicator(df['close'], 50).sma_indicator()
        df['sma200'] = ta.trend.SMAIndicator(df['close'], 200).sma_indicator()
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 10).average_true_range()
        hl2 = (df['high'] + df['low']) / 2
        df['st_upper'] = hl2 + (3 * atr)
        df['st_lower'] = hl2 - (3 * atr)
        df['st_bull'] = df['close'] > df['st_upper'].shift(1)
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 14).adx()
        return df

# ==========================================
# 4. SUPPORT & RESISTANCE - Roadmap Section 2
# ==========================================
class SupportResistanceEngine:
    def __init__(self, kite_instance):
        self.kite = kite_instance

    def get_static_levels(self, token):
        # Fetching Day data (limit is 2000, 60 is safe)
        d_data = self.kite.historical_data(token, (datetime.now()-timedelta(days=60)), datetime.now(), "day")
        if not d_data: return None
        df_d = pd.DataFrame(d_data)
        pdh, pdl = df_d['high'].iloc[-2], df_d['low'].iloc[-2]
        df_d['date'] = pd.to_datetime(df_d['date'])
        df_d.set_index('date', inplace=True)
        df_w = df_d.resample('W-SUN').agg({'high':'max', 'low':'min'})
        df_m = df_d.resample('ME').agg({'high':'max', 'low':'min'})
        return {
            "PDH": pdh, "PDL": pdl,
            "Weekly High": df_w['high'].iloc[-2], "Weekly Low": df_w['low'].iloc[-2],
            "Monthly High": df_m['high'].iloc[-2], "Monthly Low": df_m['low'].iloc[-2]
        }

    def calculate_advanced_levels(self, df):
        last_h, last_l, last_c = df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1]
        rng = last_h - last_l
        pp = (last_h + last_l + last_c) / 3
        bc = (last_h + last_l) / 2
        tc = (pp - bc) + pp
        h4, h3 = last_c + (rng * 1.1 / 2), last_c + (rng * 1.1 / 4)
        l3, l4 = last_c - (rng * 1.1 / 4), last_c - (rng * 1.1 / 2)
        mx, mn = df['high'].max(), df['low'].min()
        diff = mx - mn
        fibs = {"23.6%": mx-(diff*0.236), "38.2%": mx-(diff*0.382), "50.0%": mx-(diff*0.5), "61.8%": mx-(diff*0.618), "78.6%": mx-(diff*0.786)}
        return {"Pivot": pp, "TC": tc, "BC": bc, "H4": h4, "H3": h3, "L3": l3, "L4": l4, "Fibs": fibs}

    def get_volume_profile(self, df, bins=50):
        vol_counts, p_bins = np.histogram(df['close'], bins=bins, weights=df['volume'])
        b_centers = (p_bins[:-1] + p_bins[1:]) / 2
        poc = b_centers[np.argmax(vol_counts)]
        hvns, lvns = [], []
        if HAS_SCIPY:
            h_idx, _ = find_peaks(vol_counts, height=np.mean(vol_counts))
            hvns = b_centers[h_idx].tolist()
            inv_v = np.max(vol_counts) - vol_counts
            l_idx, _ = find_peaks(inv_v, height=np.mean(inv_v))
            lvns = b_centers[l_idx].tolist()
        return {"POC": poc, "HVNs": hvns, "LVNs": lvns}

# ==========================================
# MAIN STREAMLIT UI
# ==========================================
st.set_page_config(layout="wide", page_title="Unified Roadmap Dashboard")
st.title("🛡️ Institutional Roadmap: Section 1 & 2 Complete")

st.sidebar.header("🔑 Kite API Provision")
in_api_key = st.sidebar.text_input("Kite API Key", type="password")
in_access_token = st.sidebar.text_input("Access Token", type="password")

symbols = {
    "NIFTY 50": 256265, "BANK NIFTY": 260105, "FIN NIFTY": 257801, 
    "SENSEX": 265, "MIDCAP NIFTY": 288009, 
    "RELIANCE": 738561, "HDFC BANK": 341249, "TCS": 295321, "INFY": 408065
}

sym = st.sidebar.selectbox("Symbol", list(symbols.keys()))
tf = st.sidebar.selectbox("Timeframe", ["1 Minute", "5 Minute", "15 Minute", "1 Hour", "Daily"])

if st.sidebar.button("Execute Unified Analysis"):
    if not in_api_key or not in_access_token:
        st.warning("Please enter credentials.")
    else:
        data_eng = PriceDataEngine(in_api_key, in_access_token)
        success, msg = data_eng.test_connection()
        if success:
            df = data_eng.fetch_ohlcv(symbols[sym], tf)
            if df is not None:
                df, s_trend = MarketStructureEngine().calculate_structure(df)
                df = TrendDirectionEngine().analyze(df)
                sr_eng = SupportResistanceEngine(data_eng.kite)
                static = sr_eng.get_static_levels(symbols[sym])
                advanced = sr_eng.calculate_advanced_levels(df)
                v_profile = sr_eng.get_volume_profile(df)
                
                row = df.iloc[-1]
                st.subheader(f"Analysis: {sym} | State: {s_trend}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("POC", round(v_profile['POC'], 2))
                m2.metric("Super Trend", "Bullish" if row['st_bull'] else "Bearish")
                m3.metric("VWAP", round(row['vwap'], 2))
                m4.metric("ADX", round(row['adx'], 2))

                fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
                fig.add_trace(go.Scatter(x=df.index, y=df['ema200'], name="EMA 200", line=dict(color='yellow')))
                fig.update_layout(height=700, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

                t1, t2, t3, t4, t5, t6 = st.tabs([
                    "Static Levels (H/L)", "Pivots & CPR", "Fibonacci", 
                    "Volume Profile", "Market Structure Log", "Dynamic Values"
                ])
                with t1: st.json(static)
                with t2: st.table(pd.DataFrame([advanced]).drop('Fibs', axis=1))
                with t3: st.table(pd.DataFrame.from_dict(advanced['Fibs'], orient='index', columns=['Price']))
                with t4:
                    c_v1, c_v2 = st.columns(2)
                    c_v1.write("High Volume Nodes"); c_v1.write(v_profile['HVNs'])
                    c_v2.write("Low Volume Nodes"); c_v2.write(v_profile['LVNs'])
                with t5: st.write(df[df['break'] != ""][['close', 'label', 'break']].tail(15))
                with t6: st.write({"EMA 50": row['ema50'], "EMA 200": row['ema200'], "VWAP": row['vwap'], "ADX": row['adx']})
        else: st.error(msg)
