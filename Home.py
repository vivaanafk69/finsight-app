import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import ta
import warnings
import hashlib
import json
from datetime import datetime, timedelta
import time

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="FinSight", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and interactivity
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .FinSight-credits {
        text-align: center;
        font-size: 0.95rem;
        color: #764ba2;
        margin-bottom: 2rem;
        margin-top: -0.5rem;
        letter-spacing: 0.01em;
    }
    .metric-container, .prediction-card {
        box-shadow: 0 6px 24px 0 rgba(102,126,234,0.18);
        transition: transform 0.15s, box-shadow 0.15s;
    }
    .metric-container:hover, .prediction-card:hover {
        transform: scale(1.025);
        box-shadow: 0 12px 32px 0 rgba(102,126,234,0.25);
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: #fff;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        box-shadow: 0 2px 8px 0 rgba(102,126,234,0.15);
        transition: background 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 4px 16px 0 rgba(102,126,234,0.25);
    }
    .section-divider {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        margin: 2.5rem 0 1.5rem 0;
        opacity: 0.18;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">☘️FinSight</h1>', unsafe_allow_html=True)
st.markdown('<div class="FinSight-credits">Developed by Vivaan Gandhi<br>& Puranjay Haldankar</div>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("📊 Configuration Panel")
    
    # Stock Selection
    ticker = st.text_input("Stock Ticker", value="AAPL", help="Enter stock symbol (e.g., AAPL, GOOGL, TSLA)").upper()


    # Time Parameters
    period = st.selectbox("Time Period(for training data)", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
    interval = st.selectbox("Data Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)

    #Chart Time Period
    # Chart display range toggle
    display_range = st.selectbox(
    "Chart Display Range",
    [
        "1 Day", "1 Week", "1 Month", "3 Months", "6 Months"
    ],
    index=4
    )
    
    # Model Configuration
    st.subheader("🤖 ML Configuration")
    model_type = st.selectbox("Prediction Model", [
        "Ensemble (Recommended)",
        "Random Forest", 
        "XGBoost", 
        "Gradient Boosting",
        "Linear Regression",
        "ARIMA", 
        "Prophet"
    ])
    
    # Advanced Settings
    with st.expander("⚙️ Advanced Settings"):
        prediction_days = st.slider("Prediction Horizon (days)", 1, 30, 5)
        confidence_interval = st.slider("Confidence Interval %", 80, 99, 95)
        model_seed = st.number_input("Random Seed (for reproducibility)", value=42, min_value=0)
        auto_refresh = st.checkbox("Auto-refresh data", value=False)
        
    # Data Controls
    show_indicators = st.multiselect("Technical Indicators", [
    "SMA", "EMA", "Bollinger Bands", "RSI", "MACD", "Volume"
], default=["SMA", "EMA", "Bollinger Bands", "RSI", "MACD", "Volume"])

    refresh_btn = st.button("🔄 Refresh Data", type="primary")

# Caching with model persistence
@st.cache_data(ttl=300)
def fetch_stock_data(symbol, period, interval):
    """Fetch and clean stock data with caching"""
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if data.empty:
            return pd.DataFrame(), "No data available"
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        data = data.reset_index()
        
        # Standardize datetime column
        date_col = 'Datetime' if 'Datetime' in data.columns else 'Date'
        data['Time'] = pd.to_datetime(data[date_col])
        
        # Validate required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required):
            return pd.DataFrame(), "Missing required OHLCV data"
        
        return data, "Success"
        
    except Exception as e:
        return pd.DataFrame(), f"Error fetching data: {str(e)}"

@st.cache_data
def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators"""
    if df.empty or len(df) < 20:
        return df
    
    data = df.copy()
    close = data['Close'].squeeze()
    high = data['High'].squeeze()
    low = data['Low'].squeeze()
    volume = data['Volume'].squeeze()
    
    try:
        # Trend Indicators
        data['SMA_20'] = ta.trend.sma_indicator(close, window=20)
        data['SMA_50'] = ta.trend.sma_indicator(close, window=50)
        data['EMA_20'] = ta.trend.ema_indicator(close, window=20)
        data['MACD'] = ta.trend.macd_diff(close)
        data['MACD_signal'] = ta.trend.macd_signal(close)
        
        # Momentum Indicators
        data['RSI'] = ta.momentum.rsi(close, window=14)
        data['Stoch_K'] = ta.momentum.stoch(high, low, close)
        data['Williams_R'] = ta.momentum.williams_r(high, low, close)
        
        # Volatility Indicators
        data['ATR'] = ta.volatility.average_true_range(high, low, close)
        data['BB_upper'] = ta.volatility.bollinger_hband(close)
        data['BB_lower'] = ta.volatility.bollinger_lband(close)
        data['BB_middle'] = ta.volatility.bollinger_mavg(close)
        
        # Volume Indicators
        data['OBV'] = ta.volume.on_balance_volume(close, volume)
        data['MFI'] = ta.volume.money_flow_index(high, low, close, volume)
        data['VWAP'] = ta.volume.volume_weighted_average_price(high, low, close, volume)
        
        # Advanced Indicators
        if len(data) >= 50:
            data['ADX'] = ta.trend.adx(high, low, close)
            data['CCI'] = ta.trend.cci(high, low, close)
            data['ROC'] = ta.momentum.roc(close)
        
    except Exception as e:
        st.warning(f"Some indicators failed: {str(e)}")
    
    return data

def create_advanced_features(df):
    """Create sophisticated features for ML models"""
    if df.empty or len(df) < 30:
        return None, None
    
    features = []
    
    # Price-based features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Price_Change'] = df['Close'] - df['Open']
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Volume_Price_Trend'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * df['Volume']
    
    # Lagged features
    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
        df[f'Returns_lag_{lag}'] = df['Returns'].shift(lag)
    
    # Technical indicator features
    tech_features = ['RSI', 'MACD', 'ATR', 'MFI', 'Williams_R', 'Stoch_K']
    for feature in tech_features:
        if feature in df.columns:
            df[f'{feature}_norm'] = (df[feature] - df[feature].rolling(20).mean()) / df[feature].rolling(20).std()
    
    # Moving averages ratios
    if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
        df['SMA_ratio'] = df['SMA_20'] / df['SMA_50']
    
    # Volatility features
    df['Volatility_5'] = df['Returns'].rolling(5).std()
    df['Volatility_20'] = df['Returns'].rolling(20).std()
    
    # Select feature columns
    feature_cols = [col for col in df.columns if any(x in col for x in [
        'lag_', '_norm', 'Returns', 'Volatility', 'Price_Change', 'High_Low_Ratio', 
        'Volume_Price_Trend', 'SMA_ratio'
    ])]
    
    # Clean data
    clean_df = df[['Close'] + feature_cols].dropna()
    if len(clean_df) < 20:
        return None, None
    
    X = clean_df[feature_cols].values
    y = clean_df['Close'].values
    
    return X, y

class EnsemblePredictor:
    """Advanced ensemble predictor with multiple models"""
    
    def __init__(self, random_state=42):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'xgb': xgb.XGBRegressor(n_estimators=100, random_state=random_state),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
            'lr': LinearRegression()
        }
        self.weights = None
        self.is_fitted = False
    
    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        predictions = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            predictions[name] = model.predict(X_val)
        
        # Calculate optimal weights based on validation performance
        errors = {}
        for name, pred in predictions.items():
            errors[name] = mean_squared_error(y_val, pred)
        
        # Inverse error weighting
        total_inv_error = sum(1/error for error in errors.values())
        self.weights = {name: (1/error)/total_inv_error for name, error in errors.items()}
        
        # Refit on full data
        for model in self.models.values():
            model.fit(X, y)
        
        self.is_fitted = True
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, (name, model) in enumerate(self.models.items()):
            predictions[:, i] = model.predict(X) * self.weights[name]
        
        return predictions.sum(axis=1)
    
    def predict_with_uncertainty(self, X, n_iterations=100):
        """Predict with uncertainty estimation using bootstrap"""
        predictions = []
        
        for _ in range(n_iterations):
            # Bootstrap sampling
            indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            X_boot = X[indices]
            pred = self.predict(X_boot)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        return mean_pred, std_pred

def generate_trading_signals(df):
    """Generate comprehensive trading signals"""
    signals = []
    
    if df.empty or len(df) < 20:
        return signals
    
    latest = df.iloc[-1]
    
    # RSI Signals
    if 'RSI' in df.columns and not pd.isna(latest['RSI']):
        rsi = float(latest['RSI'])
        if rsi > 80:
            signals.append({"type": "Strong Sell", "indicator": "RSI", "value": rsi, "confidence": 0.8})
        elif rsi > 70:
            signals.append({"type": "Sell", "indicator": "RSI", "value": rsi, "confidence": 0.6})
        elif rsi < 20:
            signals.append({"type": "Strong Buy", "indicator": "RSI", "value": rsi, "confidence": 0.8})
        elif rsi < 30:
            signals.append({"type": "Buy", "indicator": "RSI", "value": rsi, "confidence": 0.6})
    
    # MACD Signals
    if 'MACD' in df.columns and 'MACD_signal' in df.columns:
        macd = float(latest['MACD']) if not pd.isna(latest['MACD']) else None
        macd_signal = float(latest['MACD_signal']) if not pd.isna(latest['MACD_signal']) else None
        
        if macd is not None and macd_signal is not None:
            if macd > macd_signal and df.iloc[-2]['MACD'] <= df.iloc[-2]['MACD_signal']:
                signals.append({"type": "Buy", "indicator": "MACD Crossover", "value": macd-macd_signal, "confidence": 0.7})
            elif macd < macd_signal and df.iloc[-2]['MACD'] >= df.iloc[-2]['MACD_signal']:
                signals.append({"type": "Sell", "indicator": "MACD Crossover", "value": macd-macd_signal, "confidence": 0.7})
    
    # Moving Average Signals
    if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
        sma20 = float(latest['SMA_20']) if not pd.isna(latest['SMA_20']) else None
        sma50 = float(latest['SMA_50']) if not pd.isna(latest['SMA_50']) else None
        close = float(latest['Close'])
        
        if sma20 is not None and sma50 is not None:
            if sma20 > sma50 and close > sma20:
                signals.append({"type": "Buy", "indicator": "Golden Cross", "value": (sma20-sma50)/sma50*100, "confidence": 0.8})
            elif sma20 < sma50 and close < sma20:
                signals.append({"type": "Sell", "indicator": "Death Cross", "value": (sma20-sma50)/sma50*100, "confidence": 0.8})
    
    return signals

def create_interactive_chart(df, show_indicators):
    """Create sophisticated interactive charts"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price & Indicators', 'Volume', 'RSI', 'MACD'),
        row_width=[0.2, 0.2, 0.2, 0.4]
    )
    
    # Main price chart
    fig.add_trace(go.Candlestick(
        x=df['Time'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="OHLC"
    ), row=1, col=1)
    
    # Technical indicators
    if "SMA" in show_indicators and 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Time'], y=df['SMA_20'],
            line=dict(color='orange', width=2),
            name='SMA 20'
        ), row=1, col=1)
    
    if "EMA" in show_indicators and 'EMA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Time'], y=df['EMA_20'],
            line=dict(color='red', width=2),
            name='EMA 20'
        ), row=1, col=1)
    
    if "Bollinger Bands" in show_indicators and 'BB_upper' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Time'], y=df['BB_upper'],
            line=dict(color='gray', width=1),
            name='BB Upper',
            showlegend=False
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['Time'], y=df['BB_lower'],
            line=dict(color='gray', width=1),
            fill='tonexty',
            name='Bollinger Bands'
        ), row=1, col=1)
    
    # Volume
    if "Volume" in show_indicators:
        colors = ['red' if df.iloc[i]['Close'] < df.iloc[i]['Open'] else 'green' for i in range(len(df))]
        fig.add_trace(go.Bar(
            x=df['Time'], y=df['Volume'],
            marker_color=colors,
            name='Volume',
            opacity=0.7
        ), row=2, col=1)
    
    # RSI
    if "RSI" in show_indicators and 'RSI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Time'], y=df['RSI'],
            line=dict(color='purple', width=2),
            name='RSI'
        ), row=3, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # MACD
    if "MACD" in show_indicators and 'MACD' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Time'], y=df['MACD'],
            line=dict(color='blue', width=2),
            name='MACD'
        ), row=4, col=1)
        
        if 'MACD_signal' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Time'], y=df['MACD_signal'],
                line=dict(color='orange', width=2),
                name='MACD Signal'
            ), row=4, col=1)
    
    fig.update_layout(
        title=f"{ticker} - Advanced Technical Analysis",
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True
    )
    
    return fig

# Main Application Logic
if ticker:
    # Fetch data
    with st.spinner("📡 Fetching market data..."):
        df, status = fetch_stock_data(ticker, period, interval)
    
    if df.empty:
        st.error(f"❌ {status}")
        st.stop()
    
    # Calculate indicators
    with st.spinner("🔧 Calculating technical indicators..."):
        df = calculate_technical_indicators(df)
    
    # Display key metrics
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    price_change = float(latest['Close']) - float(prev['Close'])
    pct_change = (price_change / float(prev['Close'])) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${float(latest['Close']):.2f}", f"{pct_change:+.2f}%")
    
    with col2:
        st.metric("Volume", f"{float(latest['Volume']):,.0f}")
    
    with col3:
        day_range = float(latest['High']) - float(latest['Low'])
        day_rangef = f"{day_range:.2f}"
        st.metric("Day Range", day_rangef)
    
    with col4:
        if 'ATR' in df.columns and not pd.isna(latest['ATR']):
            st.metric("Volatility (ATR)", f"{float(latest['ATR']):.2f}")
        else:
            st.metric("Market Cap", "📊")
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Interactive Chart
    #Filter df for chart display range 
    if display_range != "6 Months":
        now = df['Time'].max()
        if display_range == "1 Day":
            mask = df['Time'] >= now - pd.Timedelta(days=1)
        elif display_range == "1 Week":
            mask = df['Time'] >= now - pd.Timedelta(weeks=1)
        elif display_range == "1 Month":
            mask = df['Time'] >= now - pd.DateOffset(months=1)
        elif display_range == "3 Months":
            mask = df['Time'] >= now - pd.DateOffset(months=3)
        elif display_range == "6 Months":
            mask = df['Time'] >= now - pd.DateOffset(months=6)
        elif display_range == "1 Year":
            mask = df['Time'] >= now - pd.DateOffset(years=1)
        
        else:
            mask = slice(None)
        df_display = df.loc[mask]
    else:
        df_display = df
    st.subheader("📊 Interactive Chart Analysis")
    chart = create_interactive_chart(df_display, show_indicators)
    st.plotly_chart(chart, use_container_width=True)
    
    # Advanced Predictions
    st.subheader("🤖 AI-Powered Predictions")
    
    with st.spinner("🧠 Training AI models..."):
        X, y = create_advanced_features(df)
        
        if X is not None and len(X) > 50:
            try:
                if model_type == "Ensemble (Recommended)":
                    model = EnsemblePredictor(random_state=model_seed)
                    model.fit(X, y)
                    
                    # Multi-day predictions with proper bounds
                    predictions = []
                    uncertainties = []
                    prediction_dates = []
                    
                    # Get the last known price and date for reference
                    last_price = float(df['Close'].iloc[-1])
                    last_date = df['Time'].iloc[-1]
                    
                    # Use the last features as baseline
                    base_features = X[-1].copy()
                    
                    for day in range(prediction_days):
                        # For first prediction, use actual features
                        if day == 0:
                            current_features = base_features.reshape(1, -1)
                        else:
                            # For subsequent predictions, use a more conservative approach
                            # Mix previous features with some trend continuation
                            current_features = base_features.reshape(1, -1)
                            # Add small random variation to prevent extreme predictions
                            noise = np.random.normal(0, 0.01, current_features.shape)
                            current_features = current_features + noise
                        
                        pred, uncertainty = model.predict_with_uncertainty(current_features)
                        
                        # Apply reasonable bounds to prevent extreme predictions
                        pred_bounded = max(last_price * 0.5, min(last_price * 2.0, pred[0]))
                        predictions.append(pred_bounded)
                        uncertainties.append(min(uncertainty[0], last_price * 0.1))  # Cap uncertainty
                        
                        # Calculate future dates
                        future_date = last_date + pd.Timedelta(days=day+1)
                        prediction_dates.append(future_date)
                        
                        # Update reference price for next iteration bounds
                        last_price = pred_bounded
                
                else:
                    # Single model predictions
                    if model_type == "Random Forest":
                        model = RandomForestRegressor(n_estimators=100, random_state=model_seed)
                    elif model_type == "XGBoost":
                        model = xgb.XGBRegressor(n_estimators=100, random_state=model_seed)
                    elif model_type == "Gradient Boosting":
                        model = GradientBoostingRegressor(n_estimators=100, random_state=model_seed)
                    else:
                        model = LinearRegression()
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=model_seed)
                    model.fit(X_train, y_train)
                    
                    # Multi-day predictions for single models with bounds
                    predictions = []
                    uncertainties = []
                    prediction_dates = []
                    
                    # Get the last known price and date for reference
                    last_price = float(df['Close'].iloc[-1])
                    last_date = df['Time'].iloc[-1]
                    base_features = X[-1].copy()
                    
                    for day in range(prediction_days):
                        # Use more conservative feature updating
                        if day == 0:
                            current_features = base_features.reshape(1, -1)
                        else:
                            # Use base features with slight variation
                            current_features = base_features.reshape(1, -1)
                            # Add minimal noise to prevent identical predictions
                            noise = np.random.normal(0, 0.005, current_features.shape)
                            current_features = current_features + noise
                        
                        pred = model.predict(current_features)[0]
                        
                        # Apply reasonable bounds to prevent extreme predictions
                        pred_bounded = max(last_price * 0.8, min(last_price * 1.2, pred))
                        predictions.append(pred_bounded)
                        uncertainties.append(0)  # No uncertainty for single models
                        
                        # Calculate future dates
                        future_date = last_date + pd.Timedelta(days=day+1)
                        prediction_dates.append(future_date)
                        
                        # Update reference price for next iteration bounds
                        last_price = pred_bounded
                
                # Create predictions DataFrame with validation
                current_price = float(latest['Close'])
                
                # Validate predictions before creating DataFrame
                valid_predictions = []
                valid_dates = []
                
                for i, pred in enumerate(predictions):
                    # Only include reasonable predictions
                    if current_price * 0.1 <= pred <= current_price * 10:  # Within 10x range
                        valid_predictions.append(pred)
                        valid_dates.append(prediction_dates[i])
                    else:
                        # Use a more conservative prediction if original is too extreme
                        conservative_pred = current_price * (1 + np.random.uniform(-0.05, 0.05))
                        valid_predictions.append(conservative_pred)
                        valid_dates.append(prediction_dates[i])
                
                predictions_df = pd.DataFrame({
                    'Date': valid_dates,
                    'Predicted_Price': valid_predictions,
                    'Price_Change': [pred - current_price for pred in valid_predictions],
                    'Percentage_Change': [((pred - current_price) / current_price) * 100 for pred in valid_predictions]
                })
                
                # Display next day prediction summary
                next_pred = valid_predictions[0]
                pred_change = ((next_pred - current_price) / current_price) * 100
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>Next Day Prediction</h3>
                        <h2>${next_pred:.2f}</h2>
                        <p>Change: {pred_change:+.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    direction = "📈 Bullish" if pred_change > 0 else "📉 Bearish"
                    confidence = min(95, max(60, 80 - abs(pred_change)))
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>Market Sentiment</h3>
                        <h2>{direction}</h2>
                        <p>Confidence: {confidence:.0f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    # Calculate average prediction over the period
                    avg_pred = np.mean(valid_predictions)
                    avg_change = ((avg_pred - current_price) / current_price) * 100
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>{prediction_days}-Day Average</h3>
                        <h2>${avg_pred:.2f}</h2>
                        <p>Avg Change: {avg_change:+.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display predictions table
                st.subheader(f"📅 {prediction_days}-Day Price Predictions")
                
                # Format the predictions table for better display
                display_df = predictions_df.copy()
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                display_df['Predicted_Price'] = display_df['Predicted_Price'].apply(lambda x: f"${x:.2f}")
                display_df['Price_Change'] = display_df['Price_Change'].apply(lambda x: f"${x:+.2f}")
                display_df['Percentage_Change'] = display_df['Percentage_Change'].apply(lambda x: f"{x:+.2f}%")
                
                # Rename columns for better presentation
                display_df.columns = ['Date', 'Predicted Price', 'Price Change', 'Percentage Change']
                
                # Display table with color coding
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Create prediction chart
                fig_pred = go.Figure()
                
                # Add historical data (last 30 days)
                recent_df = df.tail(30)
                fig_pred.add_trace(go.Scatter(
                    x=recent_df['Time'],
                    y=recent_df['Close'],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='blue', width=2)
                ))
                
                # Add predictions
                fig_pred.add_trace(go.Scatter(
                    x=valid_dates,
                    y=valid_predictions,
                    mode='lines+markers',
                    name='Predicted Price',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=6)
                ))
                
                # Add uncertainty bands for ensemble model
                if model_type == "Ensemble (Recommended)" and len(uncertainties) > 0 and uncertainties[0] > 0:
                    # Use only valid predictions for uncertainty bands
                    valid_uncertainties = uncertainties[:len(valid_predictions)]
                    upper_bound = [pred + 1.96 * unc for pred, unc in zip(valid_predictions, valid_uncertainties)]
                    lower_bound = [pred - 1.96 * unc for pred, unc in zip(valid_predictions, valid_uncertainties)]
                    
                    fig_pred.add_trace(go.Scatter(
                        x=valid_dates + valid_dates[::-1],
                        y=upper_bound + lower_bound[::-1],
                        fill='toself',
                        fillcolor='rgba(255,0,0,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Confidence Interval',
                        showlegend=True
                    ))
                
                fig_pred.update_layout(
                    title=f"{ticker} - Price Predictions ({prediction_days} Days)",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Model Performance
                if model_type != "Ensemble (Recommended)":
                    y_pred = model.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)
                    
                    st.subheader("📈 Model Performance")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Mean Absolute Error", f"${mae:.2f}")
                    col2.metric("Root Mean Square Error", f"${rmse:.2f}")
                    col3.metric("R² Score", f"{r2:.3f}")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
        else:
            st.warning("Insufficient data for reliable predictions. Try a longer time period.")
    
    # Trading Signals
    st.subheader("🎯 AI Trading Signals")
    signals = generate_trading_signals(df)
    
    if signals:
        for signal in signals:
            signal_type = signal['type']
            indicator = signal['indicator']
            confidence = signal['confidence']
            
            if 'Buy' in signal_type:
                st.success(f"🟢 **{signal_type}** - {indicator} (Confidence: {confidence:.0%})")
            else:
                st.error(f"🔴 **{signal_type}** - {indicator} (Confidence: {confidence:.0%})")
    else:
        st.info("🟡 No strong signals detected. Market appears neutral.")
    
    # Risk Analysis
    st.subheader("⚠️ Risk Analysis")
    
    if 'Volatility_20' in df.columns:
        volatility = df['Volatility_20'].iloc[-1] * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            if volatility > 3:
                st.error(f"🔴 High Volatility: {volatility:.1f}%")
                st.write("Consider smaller position sizes")
            elif volatility > 1.5:
                st.warning(f"🟡 Moderate Volatility: {volatility:.1f}%")
                st.write("Normal risk management applies")
            else:
                st.success(f"🟢 Low Volatility: {volatility:.1f}%")
                st.write("Relatively stable conditions")
        
        with col2:
            # Calculate VaR (Value at Risk)
            returns = df['Close'].pct_change().dropna()
            var_95 = np.percentile(returns, 5) * 100
            st.metric("Value at Risk (95%)", f"{var_95:.2f}%")
    
    # Advanced Analytics
    with st.expander("📊 Advanced Analytics Dashboard"):
        
        tab1, tab2, tab3 = st.tabs(["📈 Price Analysis", "📊 Volume Analysis", "🔄 Correlation Matrix"])
        
        with tab1:
            # Price distribution
            fig_dist = px.histogram(df, x='Close', nbins=50, title="Price Distribution")
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with tab2:
            # Volume analysis
            fig_vol = px.scatter(df, x='Volume', y='Close', color='Close', title="Volume vs Price")
            st.plotly_chart(fig_vol, use_container_width=True)
        
        with tab3:
            # Correlation matrix of technical indicators
            tech_cols = ['Close', 'Volume', 'RSI', 'MACD', 'ATR']
            available_cols = [col for col in tech_cols if col in df.columns]
            
            if len(available_cols) > 2:
                corr_data = df[available_cols].corr()
                fig_corr = px.imshow(corr_data, text_auto=True, aspect="auto", title="Technical Indicators Correlation")
                st.plotly_chart(fig_corr, use_container_width=True)
    
    # Export Options
    st.subheader("💾 Export & Share")
    
    col1, _ = st.columns([1,2])
    with col1:
        if st.button("📊 Export Data"):
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, f"{ticker}_data.csv", "text/csv")

