import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

st.set_page_config(page_title="Alpha Vanguard | Live", layout="wide", page_icon="⚡")

# --- Custom CSS for Professional Nasdaq Look ---
st.markdown("""
<style>
    /* Absolute Dark Mode / Raw Data Aesthetic */
    .stApp {
        background-color: #0b0e14;
        color: #e2e8f0;
        font-family: 'Inter', 'SF Pro Display', sans-serif;
    }
    
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    h1 {
        border-bottom: 1px solid #1e293b;
        padding-bottom: 10px;
        margin-bottom: 20px;
        font-size: 24px;
        text-transform: uppercase;
    }
    
    /* Metrics Top Bar styling */
    div[data-testid="metric-container"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        padding: 15px;
        border-radius: 4px;
        border-left: 4px solid #3b82f6; /* Accent */
    }
    
    /* Positive / Negative colors */
    .pos-val { color: #10b981; font-weight: bold; }
    .neg-val { color: #ef4444; font-weight: bold; }
    
    /* Clean tables */
    .dataframe {
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: #d1d5db;
    }
    
    /* Hide Streamlit Branding elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Server & State Initialization ---
load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID", "")
API_SECRET = os.getenv("APCA_API_SECRET_KEY", "")

@st.cache_resource
def get_alpaca_client():
    if not API_KEY or not API_SECRET:
        return None
    return TradingClient(API_KEY, API_SECRET, paper=True)

client = get_alpaca_client()

def load_live_state():
    state_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'live_state.json')
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                return json.load(f)
        except:
            pass
    return None

# --- Top Header ---
col_head, col_refresh = st.columns([8, 1])
with col_head:
    st.markdown("<h1>ALPHA VANGUARD | LIVE SYSTEM MONiTOR</h1>", unsafe_allow_html=True)
with col_refresh:
    if st.button("↻ REFRESH"):
        pass # Streamlit natively re-runs on button press

# --- Fetch Live Data ---
if not client:
    st.error("Alpaca API keys missing from .env. System halted.")
    st.stop()

try:
    account = client.get_account()
    equity = float(account.equity)
    buying_power = float(account.buying_power)
    day_pnl = float(account.equity) - float(account.last_equity)
    day_pnl_pct = (day_pnl / float(account.last_equity)) * 100 if float(account.last_equity) > 0 else 0
    
    positions = client.get_all_positions()
except Exception as e:
    st.error(f"Alpaca Connection Error: {e}")
    st.stop()

state = load_live_state()

# --- TOP TAPE ROW ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Gross Account Equity", f"${equity:,.2f}", f"${day_pnl:,.2f} ({day_pnl_pct:.2f}%)")
col2.metric("Available Buying Power", f"${buying_power:,.2f}")
col3.metric("Open Positions", len(positions))

if state and "last_update" in state:
    try:
        last_dt = datetime.fromisoformat(state['last_update'])
        col4.metric("Last Model Sync", last_dt.strftime("%H:%M:%S EST"))
    except:
        col4.metric("Last Model Sync", "N/A")
else:
    col4.metric("Last Model Sync", "Offline")

st.markdown("<hr style='border:1px solid #1f2937'>", unsafe_allow_html=True)

# --- MAIN CONTENT LAYOUT ---
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("🔴 LIVE ALLOCATIONS (BROKER)")
    if positions:
        pos_data = []
        for p in positions:
            pnl_color = "pos-val" if float(p.unrealized_intraday_pl) >= 0 else "neg-val"
            pos_data.append({
                "Symbol": p.symbol,
                "Shares": float(p.qty),
                "Market Value": f"${float(p.market_value):,.2f}",
                "Avg Entry": f"${float(p.avg_entry_price):,.2f}",
                "Current PX": f"${float(p.current_price):,.2f}",
                "Day PnL": f"${float(p.unrealized_intraday_pl):,.2f}",
                "Day PnL %": f"{float(p.unrealized_intraday_plpc)*100:.2f}%"
            })
            
        df_pos = pd.DataFrame(pos_data)
        st.dataframe(df_pos, use_container_width=True, hide_index=True)
    else:
        st.info("No open positions in Alpaca portfolio.")

with right_col:
    st.subheader("⚙️ ML ALPHA ENGINE TARGETS")
    if state and "signals" in state and "target_weights" in state:
        signals = state["signals"]
        targets = state["target_weights"]
        
        target_data = []
        # Merge signals and weights where they overlap or are explicitly tracked
        all_syms = set(signals.keys()).union(set(targets.keys()))
        for sym in all_syms:
            if sym in targets and targets[sym] > 0:
                target_data.append({
                    "Symbol": sym,
                    "Conviction (Alpha)": f"{signals.get(sym, 0.0):.4f}",
                    "Target Weight": f"{targets.get(sym, 0.0)*100:.2f}%",
                    "Target Notional": f"${(targets.get(sym, 0.0) * equity):,.2f}"
                })
        
        df_tgt = pd.DataFrame(target_data)
        if not df_tgt.empty:
            df_tgt = df_tgt.sort_values("Target Weight", ascending=False)
            st.dataframe(df_tgt, use_container_width=True, hide_index=True)
        else:
            st.info("System currently commands 100% Cash / No active signals.")
    else:
        st.warning("No backend live state generated yet. Waiting for `live_trader.py` cycle.")
