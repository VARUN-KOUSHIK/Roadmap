import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
from scipy.signal import find_peaks

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
# 3. TREND DIRECTION (Dynamic Levels) - Section 1 & 2
# ==========================================
class TrendDirectionEngine:
    def analyze(self, df):
        # Dynamic Levels: EMA & SMA
        df['ema200'] = ta.trend.EMAIndicator(df['close'], 200).ema_indicator()
        df['ema50'] = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
        df['sma50'] = ta.trend.SMAIndicator(df['close'], 50).sma_indicator()
        df['sma200'] = ta.trend.SMAIndicator(df['close'], 200).sma_indicator()
        
        # Dynamic Levels: VWAP
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
        
        # Dynamic Levels: Super Trend
        atr_period = 10
        atr_multiplier = 3
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], atr_period).average_true_range()
        
        hl2 = (df['high'] + df['low']) / 2
        df['st_upper'] = hl2 + (atr_multiplier * atr)
        df['st_lower'] = hl2 - (atr_multiplier * atr)
        
        # Roadmap: Trend Strength Indicator (ADX)
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 14).adx()
        df['st_bull'] = df['close'] > df['st_upper'].shift(1)
        
        return df

# ==========================================
# 4. SUPPORT & RESISTANCE (Static & Advanced) - Section 2
# ==========================================
class SupportResistanceEngine:
    def __init__(self, kite_instance):
        self.kite = kite_instance

    def get_static_levels(self, token):
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
            "WH": df_w['high'].iloc[-2], "WL": df_w['low'].iloc[-2],
            "MH": df_m['high'].iloc[-2], "ML": df_m['low'].iloc[-2]
        }

    def calculate_advanced_levels(self, df):
        last_h, last_l, last_c = df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1]
        rng = last_h - last_l
        pp = (last_h + last_l + last_c) / 3
        bc = (last_h + last_l) / 2
        tc = (pp - bc) + pp
        
        # Camarilla
        h4 = last_c + (rng * 1.1 / 2)
        h3 = last_c + (rng * 1.1 / 4)
        l3 = last_c - (rng * 1.1 / 4)
        l4 = last_c - (rng * 1.1 / 2)

        # Fibonacci
        max_p, min_p = df['high'].max(), df['low'].min()
        diff = max_p - min_p
        fibs = {"23.6%": max_p - (diff * 0.236), "38.2%": max_p - (diff * 0.382), "50.0%": max_p - (diff * 0.5), "61.8%": max_p - (diff * 0.618)}

        return {"PP": pp, "TC": tc, "BC": bc, "H4": h4, "H3": h3, "L3": l3, "L4": l4, "Fibs": fibs}

    def get_volume_profile_advanced(self, df, bins=50):
        """Calculates POC, High Volume Nodes (HVN), and Low Volume Nodes (LVN)"""
        # Create Histogram
        volume_counts, price_bins = np.histogram(df['close'], bins=bins, weights=df['volume'])
        bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
        
        # Point of Control (POC)
        poc_idx = np.argmax(volume_counts)
        poc = bin_centers[poc_idx]
        
        # High Volume Nodes (HVN) - Peaks in volume distribution
        hvn_indices, _ = find_peaks(volume_counts, height=np.mean(volume_counts))
        hvns = bin_centers[hvn_indices]
        
        # Low Volume Nodes (LVN) - Valleys in volume distribution
        # We invert volume to find valleys as peaks
        inverted_volume = np.max(volume_counts) - volume_counts
        lvn_indices, _ = find_peaks(inverted_volume, height=np.mean(inverted_volume))
        lvns = bin_centers[lvn_indices]
        
        return {"POC": poc, "HVNs": hvns.tolist(), "LVNs": lvns.tolist()}

# ==========================================
# MAIN STREAMLIT UI
# ==========================================
st.set_page_config(layout="wide", page_title="Unified Roadmap Dashboard")
st.title("🛡️ Roadmap Section 1 & 2: Complete Logic Engine")

st.sidebar.header("🔑 Kite API Provision")
in_api_key = st.sidebar.text_input("Kite API Key", type="password")
in_access_token = st.sidebar.text_input("Access Token", type="password")

symbols = {"NIFTY 50": 256265, "BANK NIFTY": 260105, "RELIANCE": 738561, "HDFC BANK": 341249, "TCS": 295321}
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
                # 1. Structure
                struct_eng = MarketStructureEngine()
                df, s_trend = struct_eng.calculate_structure(df)
                
                # 2. Trend & Dynamic Levels (EMA, SMA, VWAP, SuperTrend)
                trend_eng = TrendDirectionEngine()
                df = trend_eng.analyze(df)
                
                # 3. Support & Resistance (Static, Pivots, Advanced Volume Profile)
                sr_eng = SupportResistanceEngine(data_eng.kite)
                static = sr_eng.get_static_levels(symbols[sym])
                advanced = sr_eng.calculate_advanced_levels(df)
                vol_profile = sr_eng.get_volume_profile_advanced(df)
                
                # UI Metrics
                row = df.iloc[-1]
                st.subheader(f"Dashboard: {sym} | {s_trend}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("POC", round(vol_profile['POC'], 2))
                m2.metric("VWAP", round(row['vwap'], 2))
                m3.metric("Super Trend", "Bullish" if row['st_bull'] else "Bearish")
                m4.metric("EMA 200", round(row['ema200'], 2))

                # Charting
                fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
                
                # Plot Dynamic Levels
                fig.add_trace(go.Scatter(x=df.index, y=df['ema200'], name="EMA 200", line=dict(color='yellow', width=1.5)))
                fig.add_trace(go.Scatter(x=df.index, y=df['vwap'], name="VWAP", line=dict(color='cyan', width=1)))
                
                # Plot POC & HVNs
                fig.add_hline(y=vol_profile['POC'], line_color="orange", line_width=2, annotation_text="POC")
                for hvn in vol_profile['HVNs'][:3]: # Plot top 3 HVNs
                    fig.add_hline(y=hvn, line_dash="dot", line_color="rgba(0, 255, 0, 0.5)", annotation_text="HVN")

                fig.update_layout(height=700, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

                # Data breakdown
                t1, t2 = st.tabs(["Volume Profile Nodes", "Advanced Pivots"])
                with t1:
                    st.write("**High Volume Nodes (Strong S/R):**")
                    st.write(vol_profile['HVNs'])
                    st.write("**Low Volume Nodes (Price Gaps):**")
                    st.write(vol_profile['LVNs'])
                with t2:
                    st.json(advanced)
        else:
            st.error(msg)
