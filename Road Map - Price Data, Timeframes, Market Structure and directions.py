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
            "4 Hour": "60minute", "Daily": "day", "Weekly": "day"
        }

    def test_connection(self):
        try:
            profile = self.kite.profile()
            return True, f"Connected! Welcome, {profile.get('user_name')}"
        except Exception as e:
            return False, f"Connection Failed: {str(e)}"

    def fetch_ohlcv(self, symbol_token, timeframe):
        limits = {
            "1 Minute": 30, "3 Minute": 90, "5 Minute": 90, "15 Minute": 90,
            "30 Minute": 180, "1 Hour": 180, "4 Hour": 180, "Daily": 1000, "Weekly": 1000
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
        df['sma200'] = ta.trend.SMAIndicator(df['close'], 200).sma_indicator()
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 10).average_true_range()
        hl2 = (df['high'] + df['low']) / 2
        df['st_upper'] = hl2 + (3 * atr)
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
        d_data = self.kite.historical_data(token, (datetime.now()-timedelta(days=60)), datetime.now(), "day")
        if not d_data: return None
        df_d = pd.DataFrame(d_data)
        pdh, pdl = df_d['high'].iloc[-2], df_d['low'].iloc[-2]
        df_d['date'] = pd.to_datetime(df_d['date'])
        df_d.set_index('date', inplace=True)
        df_w = df_d.resample('W-SUN').agg({'high':'max', 'low':'min'})
        df_m = df_d.resample('ME').agg({'high':'max', 'low':'min'})
        return {
            "PDH": pdh, "PDL": pdl, "WH": df_w['high'].iloc[-2], "WL": df_w['low'].iloc[-2], "MH": df_m['high'].iloc[-2], "ML": df_m['low'].iloc[-2]
        }

    def calculate_advanced_levels(self, df):
        lh, ll, lc = df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1]
        rng = lh - ll
        pp = (lh + ll + lc) / 3
        bc, tc = (lh + ll) / 2, (pp - bc if 'bc' in locals() else (lh+ll)/2) + pp
        fibs = {"61.8%": lh-((lh-ll)*0.618), "50%": lh-((lh-ll)*0.5), "38.2%": lh-((lh-ll)*0.382)}
        return {"Pivot": pp, "TC": tc, "BC": bc, "Fibs": fibs}

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
# 5. VOLUME ANALYSIS - Roadmap Section 3
# ==========================================
class VolumeAnalysisEngine:
    def calculate_volume_metrics(self, df):
        df = df.copy()
        # Average & Relative Volume
        df['avg_vol'] = df['volume'].rolling(window=20).mean()
        df['rel_vol'] = df['volume'] / df['avg_vol']
        df['vol_spike'] = df['volume'] > (df['avg_vol'] * 2)
        
        # Volume Delta (Proxy: Selling Vol vs Buying Vol based on candle color)
        df['vol_delta'] = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
        df['cvd'] = df['vol_delta'].cumsum()
        
        # Standard Indicators
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
        df['acc_dist'] = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
        
        return df

# ==========================================
# MAIN STREAMLIT UI & LIVE REFRESH
# ==========================================
st.set_page_config(layout="wide", page_title="Institutional Roadmap Dashboard")
st.title("🏛️ Roadmap: Section 1, 2, & 3 Analysis")

# Sidebar
st.sidebar.header("🔑 API & Refresh")
in_api_key = st.sidebar.text_input("Kite API Key", type="password")
in_access_token = st.sidebar.text_input("Access Token", type="password")
live_refresh = st.sidebar.checkbox("Enable Live Refresh (60s)")

symbols = {
    "NIFTY 50": 256265, "BANK NIFTY": 260105, "FIN NIFTY": 257801, 
    "SENSEX": 265, "MIDCAP NIFTY": 288009, "RELIANCE": 738561, "HDFC BANK": 341249, "TCS": 295321
}
sym = st.sidebar.selectbox("Symbol", list(symbols.keys()))
tf = st.sidebar.selectbox("Timeframe", ["1 Minute", "5 Minute", "15 Minute", "1 Hour", "Daily"])

# Main execution loop for live refresh
def run_analysis():
    if not in_api_key or not in_access_token:
        st.warning("Please enter credentials.")
        return

    data_eng = PriceDataEngine(in_api_key, in_access_token)
    success, msg = data_eng.test_connection()
    if not success:
        st.error(msg)
        return

    df = data_eng.fetch_ohlcv(symbols[sym], tf)
    if df is not None:
        # Step 1 & 2: Structure & Trend
        df, s_trend = MarketStructureEngine().calculate_structure(df)
        df = TrendDirectionEngine().analyze(df)
        
        # Step 3: Volume Analysis (NEW)
        df = VolumeAnalysisEngine().calculate_volume_metrics(df)
        
        # Step 4: S/R Advanced
        sr_eng = SupportResistanceEngine(data_eng.kite)
        static = sr_eng.get_static_levels(symbols[sym])
        advanced = sr_eng.calculate_advanced_levels(df)
        v_profile = sr_eng.get_volume_profile(df)
        
        # Layout Results
        row = df.iloc[-1]
        st.subheader(f"Strategy View: {sym} | Last Update: {datetime.now().strftime('%H:%M:%S')}")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Market Structure", s_trend)
        m2.metric("Relative Volume", f"{round(row['rel_vol'], 2)}x")
        m3.metric("Money Flow Index", round(row['mfi'], 1))
        m4.metric("POC Level", round(v_profile['POC'], 2))

        # Charts
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.add_trace(go.Scatter(x=df.index, y=df['ema200'], name="EMA 200", line=dict(color='yellow')))
        fig.update_layout(height=600, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # Tabs
        t1, t2, t3, t4, t5 = st.tabs(["Section 1: Structure", "Section 2: Levels", "Section 3: Volume Analysis", "Volume Profile Nodes", "CVD & OBV"])
        
        with t1:
            st.write("**Recent BOS/CHOCH Logs**")
            st.dataframe(df[df['break'] != ""].tail(10))
        with t2:
            st.write("**Static & Advanced Levels**")
            st.json(static)
            st.write(advanced)
        with t3:
            st.write("**Volume Indicators**")
            st.write({
                "Chaikin Money Flow": row['cmf'],
                "Acc/Dist Index": row['acc_dist'],
                "Avg 20 Vol": row['avg_vol'],
                "Volume Spike Detected": row['vol_spike']
            })
        with t4:
            st.write(f"POC: {v_profile['POC']}")
            st.write("HVNs:", v_profile['HVNs'])
        with t5:
            st.line_chart(df['cvd'], use_container_width=True, height=200)
            st.caption("Cumulative Volume Delta (CVD)")
            st.line_chart(df['obv'], use_container_width=True, height=200)
            st.caption("On-Balance Volume (OBV)")

run_analysis()

# Handle live refresh
if live_refresh:
    time.sleep(60)
    st.rerun()
