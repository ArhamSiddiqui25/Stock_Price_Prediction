import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# ── Global styling ───────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background-color: #1C1008;
    color: #E8D5A3;
}

[data-testid="stSidebar"] {
    background-color: #251608 !important;
    border-right: 1px solid #5C3D1E;
}

[data-testid="stSidebar"] * {
    color: #E8D5A3 !important;
}

/* Title */
.dashboard-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #D4A853;
    letter-spacing: 0.02em;
    margin-bottom: 0.2rem;
}

.dashboard-sub {
    font-size: 0.95rem;
    color: #9E8060;
    margin-bottom: 2rem;
    font-weight: 300;
    letter-spacing: 0.05em;
}

/* Metric cards */
.metric-card {
    background: #2A1A0A;
    border: 1px solid #5C3D1E;
    border-left: 4px solid #D4A853;
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}

.metric-label {
    font-size: 0.75rem;
    color: #9E8060;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: #D4A853;
    font-weight: 600;
}

/* Section headers */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    color: #D4A853;
    border-bottom: 1px solid #5C3D1E;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
}

/* Button */
[data-testid="stButton"] > button {
    background-color: #D4A853 !important;
    color: #1C1008 !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.05em !important;
    width: 100% !important;
    transition: background 0.2s !important;
}

[data-testid="stButton"] > button:hover {
    background-color: #C49540 !important;
}

/* Selectbox and inputs */
[data-testid="stSelectbox"] > div > div {
    background-color: #2A1A0A !important;
    border-color: #5C3D1E !important;
    color: #E8D5A3 !important;
}

/* Divider */
hr {
    border-color: #5C3D1E !important;
}

/* Spinner */
[data-testid="stSpinner"] {
    color: #D4A853 !important;
}

/* Remove default padding */
.block-container {
    padding-top: 2rem !important;
}

.sidebar-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: #D4A853;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown('<div class="dashboard-title">StockSight</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-sub">NSE STOCK PRICE PREDICTION · MACHINE LEARNING DASHBOARD</div>', unsafe_allow_html=True)
st.markdown('<hr>', unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Configuration</div>', unsafe_allow_html=True)
    st.markdown("---")

    ticker = st.selectbox("Stock", ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"])
    model_choice = st.selectbox("Model", ["Linear Regression", "Random Forest"])
    start_date = st.date_input("Start Date", value=pd.to_datetime("2019-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31"))

    st.markdown("---")
    run = st.button("Run Prediction")

    st.markdown("---")
    st.markdown('<div style="font-size:0.75rem; color:#6B4F2E; text-align:center;">Built by Zaara Riyaz Khan<br>Amity University Lucknow<br>NTCC 2024</div>', unsafe_allow_html=True)

# ── Chart helper ─────────────────────────────────────────────
def dark_chart(fig, ax):
    fig.patch.set_facecolor('#1C1008')
    ax.set_facecolor('#251608')
    ax.tick_params(colors='#9E8060', labelsize=9)
    ax.xaxis.label.set_color('#9E8060')
    ax.yaxis.label.set_color('#9E8060')
    ax.title.set_color('#D4A853')
    for spine in ax.spines.values():
        spine.set_edgecolor('#5C3D1E')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
    fig.tight_layout()
    return fig, ax

# ── Main ─────────────────────────────────────────────────────
if run:
    with st.spinner("Fetching data and running model..."):

        df = yf.download(ticker, start=start_date, end=end_date)
        df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)

        df['SMA_7'] = SMAIndicator(close=df['Close'], window=7).sma_indicator()
        df['SMA_21'] = SMAIndicator(close=df['Close'], window=21).sma_indicator()
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        bb = BollingerBands(close=df['Close'])
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['Close_lag1'] = df['Close'].shift(1)
        df['Close_lag2'] = df['Close'].shift(2)
        df['Close_lag5'] = df['Close'].shift(5)
        df['Return'] = df['Close'].pct_change()
        df['Target_Return'] = df['Return'].shift(-1)
        df.dropna(inplace=True)

        # ── Price chart ──
        st.markdown('<div class="section-header">Price History</div>', unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(12, 3.5))
        ax1.plot(df.index, df['Close'], color='#D4A853', linewidth=1.4, label='Close')
        ax1.fill_between(df.index, df['Close'], df['Close'].min(), alpha=0.08, color='#D4A853')
        ax1.set_title(f'{ticker} — Closing Price', fontsize=13, pad=10)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (₹)')
        fig1, ax1 = dark_chart(fig1, ax1)
        st.pyplot(fig1)

        # ── Model ──
        features = ['SMA_7', 'SMA_21', 'RSI', 'MACD', 'MACD_Signal',
                    'BB_High', 'BB_Low', 'Close_lag1', 'Close_lag2',
                    'Close_lag5', 'Return']

        X = df[features].values
        y_price = df['Close'].values

        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)

        split = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train = df['Target_Return'].values[:split]
        actual_prices = y_price[split:]
        prev_prices = y_price[split-1:-1]

        if model_choice == "Linear Regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

        model.fit(X_train, y_train)
        ret_pred = model.predict(X_test)
        predicted_prices = prev_prices * (1 + ret_pred)

        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        r2 = r2_score(actual_prices, predicted_prices)

        # ── Metrics ──
        st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">RMSE</div><div class="metric-value">₹{rmse:.2f}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">R² Score</div><div class="metric-value">{r2:.4f}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Model</div><div class="metric-value" style="font-size:1.1rem;padding-top:0.5rem">{model_choice}</div></div>', unsafe_allow_html=True)

        # ── Prediction chart ──
        st.markdown('<div class="section-header">Predicted vs Actual Price</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(actual_prices, color='#9E8060', linewidth=1.2, label='Actual', alpha=0.9)
        ax2.plot(predicted_prices, color='#D4A853', linewidth=1.4, label='Predicted', alpha=0.85)
        ax2.set_title(f'{model_choice} — Predicted vs Actual', fontsize=13, pad=10)
        ax2.set_xlabel('Trading Days')
        ax2.set_ylabel('Price (₹)')
        ax2.legend(facecolor='#2A1A0A', edgecolor='#5C3D1E', labelcolor='#E8D5A3')
        fig2, ax2 = dark_chart(fig2, ax2)
        st.pyplot(fig2)

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center; color:#6B4F2E; font-size:0.8rem;">NTCC Internship Project · Amity University Lucknow · 2024</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color:#6B4F2E;">
        <div style="font-family:'Playfair Display',serif; font-size:1.8rem; color:#5C3D1E; margin-bottom:1rem;">Select your parameters and run a prediction</div>
        <div style="font-size:0.9rem;">Configure the stock, model and date range in the sidebar, then click Run Prediction.</div>
    </div>
    """, unsafe_allow_html=True)
