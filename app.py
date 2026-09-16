"""
Stock Price Prediction with LSTM — Streamlit App
--------------------------------------------------
An end-to-end interactive app that:
  1. Downloads historical OHLCV data for a chosen ticker (yfinance)
  2. Visualizes it as a candlestick chart + correlation heatmap
  3. Trains an LSTM neural network to predict the Close price from
     Open, High, Low, Volume
  4. Shows training loss, test-set predictions vs actuals, and metrics
  5. Lets the user enter custom feature values to get a live prediction

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, timedelta

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.callbacks import Callback

tf.random.set_seed(42)
np.random.seed(42)

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="LSTM Stock Price Predictor",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Stock Price Prediction with LSTM")
st.caption(
    "Download historical stock data, train an LSTM neural network, and "
    "predict the closing price from Open / High / Low / Volume."
)

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
defaults = {
    "data": None,
    "ticker": None,
    "model": None,
    "history": None,
    "x_scaler": None,
    "y_scaler": None,
    "test_results": None,
    "feature_cols": ["Open", "High", "Low", "Volume"],
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --------------------------------------------------------------------------- #
# Sidebar — data controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("1. Data")
    ticker = st.text_input("Ticker symbol", value="AAPL").strip().upper()

    days_back = st.slider(
        "History length (days)", min_value=180, max_value=3650, value=1500, step=30
    )
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    fetch_clicked = st.button("Fetch data", type="primary", use_container_width=True)

    st.divider()
    st.header("2. Model settings")
    test_size = st.slider("Test set size (%)", 10, 40, 20, step=5) / 100
    epochs = st.slider("Epochs", 5, 150, 30, step=5)
    batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64, 128], value=32)
    lstm_units_1 = st.select_slider("LSTM layer 1 units", options=[16, 32, 64, 128, 256], value=64)
    lstm_units_2 = st.select_slider("LSTM layer 2 units", options=[8, 16, 32, 64, 128], value=32)

    train_clicked = st.button("Train model", type="primary", use_container_width=True)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_data(ticker_symbol: str, start: date, end: date) -> pd.DataFrame:
    """Download and clean OHLCV data for a single ticker."""
    df = yf.download(
        ticker_symbol,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance can return MultiIndex columns like ('Close', 'AAPL') even for
    # a single ticker — flatten them so downstream code always sees plain
    # column names such as 'Close', 'Open', etc.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    # yfinance names the date column 'Date' already, but guard just in case.
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "Date"})

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df[required].dropna().reset_index(drop=True)
    return df


class StreamlitProgress(Callback):
    """Feed Keras training progress into a Streamlit progress bar."""

    def __init__(self, total_epochs, progress_bar, status_text):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_bar = progress_bar
        self.status_text = status_text

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        frac = (epoch + 1) / self.total_epochs
        self.progress_bar.progress(min(frac, 1.0))
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        msg = f"Epoch {epoch + 1}/{self.total_epochs} — loss: {loss:.4f}"
        if val_loss is not None:
            msg += f" — val_loss: {val_loss:.4f}"
        self.status_text.text(msg)


def build_model(n_features: int, units_1: int, units_2: int) -> Sequential:
    model = Sequential([
        Input(shape=(n_features, 1)),
        LSTM(units_1, return_sequences=True),
        LSTM(units_2, return_sequences=False),
        Dense(25, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# --------------------------------------------------------------------------- #
# Fetch data
# --------------------------------------------------------------------------- #
if fetch_clicked:
    if not ticker:
        st.sidebar.error("Please enter a ticker symbol.")
    else:
        with st.spinner(f"Downloading {ticker} data..."):
            df = load_data(ticker, start_date, end_date)
        if df.empty:
            st.sidebar.error(
                f"No data found for '{ticker}'. Check the symbol and try again."
            )
        else:
            st.session_state.data = df
            st.session_state.ticker = ticker
            # Reset any previously trained model since the dataset changed
            st.session_state.model = None
            st.session_state.history = None
            st.session_state.test_results = None
            st.sidebar.success(f"Loaded {len(df)} rows for {ticker}.")

# --------------------------------------------------------------------------- #
# Main content
# --------------------------------------------------------------------------- #
data = st.session_state.data

if data is None:
    st.info("👈 Enter a ticker symbol in the sidebar and click **Fetch data** to begin.")
    st.stop()

tab_overview, tab_train, tab_predict = st.tabs(
    ["📊 Data overview", "🧠 Train model", "🔮 Predict"]
)

# ---------------- Tab 1: Overview ---------------- #
with tab_overview:
    st.subheader(f"{st.session_state.ticker} — historical data")

    col1, col2, col3, col4 = st.columns(4)
    latest = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else latest
    change = latest["Close"] - prev["Close"]
    pct_change = (change / prev["Close"] * 100) if prev["Close"] else 0
    col1.metric("Latest Close", f"${latest['Close']:.2f}", f"{change:+.2f} ({pct_change:+.2f}%)")
    col2.metric("Latest Volume", f"{int(latest['Volume']):,}")
    col3.metric("Rows loaded", f"{len(data):,}")
    col4.metric("Date range", f"{data['Date'].min().date()} → {data['Date'].max().date()}")

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data["Date"],
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
            )
        ]
    )
    fig.update_layout(
        title=f"{st.session_state.ticker} Stock Price",
        xaxis_rangeslider_visible=False,
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Correlation with Close**")
        corr = data[["Open", "High", "Low", "Close", "Volume"]].corr()
        st.dataframe(
            corr["Close"].sort_values(ascending=False).to_frame("Correlation"),
            use_container_width=True,
        )
    with right:
        st.markdown("**Recent rows**")
        st.dataframe(data.tail(10), use_container_width=True)

# ---------------- Tab 2: Train ---------------- #
with tab_train:
    st.subheader("Train the LSTM model")
    st.markdown(
        "The model learns to predict **Close** from **Open, High, Low, Volume** "
        "using a 2-layer LSTM network. Adjust settings in the sidebar, then click "
        "**Train model**."
    )

    if train_clicked:
        feature_cols = st.session_state.feature_cols
        x = data[feature_cols].to_numpy().astype("float64")
        y = data[["Close"]].to_numpy().astype("float64")

        if len(data) < 30:
            st.error("Not enough data to train a model. Increase the history length.")
        else:
            x_scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()
            x_scaled = x_scaler.fit_transform(x)
            y_scaled = y_scaler.fit_transform(y)

            # Chronological split (no shuffling) since this is time-series data
            xtrain, xtest, ytrain, ytest = train_test_split(
                x_scaled, y_scaled, test_size=test_size, shuffle=False
            )

            xtrain_l = xtrain.reshape((xtrain.shape[0], xtrain.shape[1], 1))
            xtest_l = xtest.reshape((xtest.shape[0], xtest.shape[1], 1))

            model = build_model(len(feature_cols), lstm_units_1, lstm_units_2)

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            callback = StreamlitProgress(epochs, progress_bar, status_text)

            with st.spinner("Training..."):
                history = model.fit(
                    xtrain_l,
                    ytrain,
                    validation_split=0.1,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[callback],
                )

            status_text.text("Training complete ✅")

            preds_scaled = model.predict(xtest_l, verbose=0)
            preds = y_scaler.inverse_transform(preds_scaled)
            ytest_actual = y_scaler.inverse_transform(ytest)

            rmse = mean_squared_error(ytest_actual, preds) ** 0.5
            mae = mean_absolute_error(ytest_actual, preds)
            r2 = r2_score(ytest_actual, preds)

            test_dates = data["Date"].iloc[-len(ytest_actual):].reset_index(drop=True)

            st.session_state.model = model
            st.session_state.history = history.history
            st.session_state.x_scaler = x_scaler
            st.session_state.y_scaler = y_scaler
            st.session_state.test_results = {
                "dates": test_dates,
                "actual": ytest_actual.flatten(),
                "predicted": preds.flatten(),
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }

    if st.session_state.history is not None:
        st.markdown("#### Training loss")
        hist = st.session_state.history
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(y=hist["loss"], name="Training loss"))
        if "val_loss" in hist:
            loss_fig.add_trace(go.Scatter(y=hist["val_loss"], name="Validation loss"))
        loss_fig.update_layout(
            xaxis_title="Epoch", yaxis_title="MSE (scaled)", height=350
        )
        st.plotly_chart(loss_fig, use_container_width=True)

    if st.session_state.test_results is not None:
        res = st.session_state.test_results
        st.markdown("#### Test set performance")
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"${res['rmse']:.2f}")
        c2.metric("MAE", f"${res['mae']:.2f}")
        c3.metric("R² score", f"{res['r2']:.4f}")

        pred_fig = go.Figure()
        pred_fig.add_trace(
            go.Scatter(x=res["dates"], y=res["actual"], name="Actual Close", mode="lines")
        )
        pred_fig.add_trace(
            go.Scatter(x=res["dates"], y=res["predicted"], name="Predicted Close", mode="lines")
        )
        pred_fig.update_layout(
            title="Predicted vs. Actual Close Price (test set)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=450,
        )
        st.plotly_chart(pred_fig, use_container_width=True)
    elif not train_clicked:
        st.info("Click **Train model** in the sidebar to fit the LSTM on the loaded data.")

# ---------------- Tab 3: Predict ---------------- #
with tab_predict:
    st.subheader("Predict a closing price")

    if st.session_state.model is None:
        st.info("Train a model first (see the **Train model** tab), then come back here.")
    else:
        st.markdown(
            "Enter feature values (or use the latest row as a starting point) to get "
            "a live Close-price prediction from the trained model."
        )
        latest_row = data.iloc[-1]

        c1, c2 = st.columns(2)
        with c1:
            open_val = st.number_input("Open", value=float(latest_row["Open"]), format="%.2f")
            high_val = st.number_input("High", value=float(latest_row["High"]), format="%.2f")
        with c2:
            low_val = st.number_input("Low", value=float(latest_row["Low"]), format="%.2f")
            volume_val = st.number_input(
                "Volume", value=float(latest_row["Volume"]), format="%.0f", step=1000.0
            )

        if st.button("Predict Close price", type="primary"):
            if high_val < low_val:
                st.error("High must be greater than or equal to Low.")
            else:
                features = np.array([[open_val, high_val, low_val, volume_val]])
                x_scaler = st.session_state.x_scaler
                y_scaler = st.session_state.y_scaler
                features_scaled = x_scaler.transform(features)
                features_lstm = features_scaled.reshape(
                    (1, features_scaled.shape[1], 1)
                )
                pred_scaled = st.session_state.model.predict(features_lstm, verbose=0)
                pred_price = y_scaler.inverse_transform(pred_scaled)[0][0]
                st.success(f"Predicted Close price: **${pred_price:.2f}**")

st.divider()
st.caption(
    "⚠️ For educational purposes only — not financial advice. Stock price "
    "predictions from historical OHLCV data alone are highly uncertain."
)
