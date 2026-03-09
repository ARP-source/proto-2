import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Quant Platform Dashboard", layout="wide", page_icon="📈")

# --- Custom CSS for Premium Look ---
st.markdown("""
<style>
    /* Dark Theme Optimization */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Elegant Title */
    h1 {
        font-family: 'Inter', sans-serif;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0;
    }
    
    /* Micro-animations on metrics */
    div[data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        transition: transform 0.2s ease;
    }
    div[data-testid="stMetricValue"]:hover {
        transform: scale(1.05);
    }
    
    /* Refined metric containers */
    div[data-testid="stMetric"] {
        background-color: #1E212B;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #2A2F3D;
    }
</style>
""", unsafe_allow_html=True)

st.title("Quantitative Trading Platform Live Dashboard")
st.markdown("Monitor high-frequency strategy execution vs benchmark index performance.")

@st.cache_data(ttl=10)
def load_data():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    strat_file = os.path.join(base_dir, 'strategy_results.csv')
    spy_file = os.path.join(base_dir, 'spy_benchmark.csv')
    
    if not os.path.exists(strat_file) or not os.path.exists(spy_file):
        st.error(f"Files not found. Checked: {strat_file} and {spy_file}")
        return None, None
        
    df_strat = pd.read_csv(strat_file, parse_dates=['date'])
    df_spy = pd.read_csv(spy_file, parse_dates=['Date'])
    
    # Normalize dates and merge
    df_strat.rename(columns={'date': 'Date'}, inplace=True)
    df_strat['Date'] = pd.to_datetime(df_strat['Date']).dt.normalize()
    df_spy['Date'] = pd.to_datetime(df_spy['Date']).dt.normalize()
    
    merged = pd.merge(df_strat, df_spy[['Date', 'Close']], on='Date', how='inner')
    merged.rename(columns={'portfolio_value': 'Strategy', 'Close': 'SPY'}, inplace=True)
    merged.sort_values(by='Date', inplace=True)
    
    # Normalize SPY to start at the same initial capital as the Strategy
    initial_capital = merged['Strategy'].iloc[0]
    initial_spy = merged['SPY'].iloc[0]
    
    merged['SPY_Normalized'] = (merged['SPY'] / initial_spy) * initial_capital
    return merged, initial_capital

merged_df, starting_capital = load_data()

if merged_df is None:
    st.warning("Data files not found. Please run the backtester engine first to generate `strategy_results.csv` and `spy_benchmark.csv`.")
    st.stop()

# --- Calculate Metrics ---
current_strat = merged_df['Strategy'].iloc[-1]
current_spy = merged_df['SPY_Normalized'].iloc[-1]

strat_return = ((current_strat - starting_capital) / starting_capital) * 100
spy_return = ((current_spy - starting_capital) / starting_capital) * 100

def get_max_drawdown(series):
    rolling_max = series.cummax()
    drawdown = series / rolling_max - 1.0
    return drawdown.min() * 100

strat_mdd = get_max_drawdown(merged_df['Strategy'])
spy_mdd = get_max_drawdown(merged_df['SPY_Normalized'])

# --- Top Key Metrics Row ---
st.markdown("### Executive Summary")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Strategy Final Value", f"${current_strat:,.2f}", f"{strat_return:.2f}%")
col2.metric("S&P 500 Final Value", f"${current_spy:,.2f}", f"{spy_return:.2f}%")
col3.metric("Strategy Max Drawdown", f"{strat_mdd:.2f}%", delta_color="inverse")
col4.metric("S&P 500 Max Drawdown", f"{spy_mdd:.2f}%", delta_color="inverse")

# --- Interactive Plotly Chart ---
st.markdown("### Equity Curve Comparison")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=merged_df['Date'], 
    y=merged_df['Strategy'],
    mode='lines',
    name='Cross-Sectional Momentum',
    line=dict(color='#4ECDC4', width=3),
    fill='tozeroy',
    fillcolor='rgba(78, 205, 196, 0.1)'
))

fig.add_trace(go.Scatter(
    x=merged_df['Date'], 
    y=merged_df['SPY_Normalized'],
    mode='lines',
    name='S&P 500 Benchmark',
    line=dict(color='#FF6B6B', width=2, dash='dot')
))

fig.update_layout(
    plot_bgcolor='#0E1117',
    paper_bgcolor='#0E1117',
    font_color='#FAFAFA',
    xaxis=dict(showgrid=True, gridcolor='#1E212B', title="Date"),
    yaxis=dict(showgrid=True, gridcolor='#1E212B', title="Portfolio Value ($)", tickformat="$,.0f"),
    hovermode="x unified",
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(0,0,0,0.5)"
    ),
    margin=dict(l=40, r=40, t=40, b=40)
)

st.plotly_chart(fig, use_container_width=True)

# --- Recent Decisions Data Table ---
st.markdown("### Raw Series Output")
st.dataframe(
    merged_df.sort_values(by="Date", ascending=False).set_index("Date").style.format({
        "Strategy": "${:,.2f}", 
        "SPY": "${:,.2f}",
        "SPY_Normalized": "${:,.2f}"
    }),
    use_container_width=True,
    height=300
)
