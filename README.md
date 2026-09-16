# 📈 Stock Price Prediction with LSTM — Streamlit App

An end-to-end, interactive rebuild of the original notebook as a single-page
Streamlit app. Enter a ticker, fetch data, train an LSTM, and get live
predictions — all from the browser, no code required.

## What it does

1. **Data** — Downloads OHLCV history for any ticker via `yfinance` and shows
   it as a candlestick chart, plus a correlation table and a preview of the
   raw rows.
2. **Train** — Trains a 2-layer LSTM (Open, High, Low, Volume → Close) with
   adjustable epochs, batch size, layer sizes, and test-set split. Shows a
   live progress bar, the training/validation loss curve, and test-set
   metrics (RMSE, MAE, R²) with a predicted-vs-actual chart.
3. **Predict** — Enter custom Open/High/Low/Volume values (or start from the
   latest row) to get an instant predicted Close price from the trained
   model.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Notes on fixes vs. the original notebook

The original notebook had a few issues that would break or silently misbehave
in a fresh environment — all fixed here:

- **`yfinance` column shape**: newer `yfinance` versions return MultiIndex
  columns (`('Close', 'AAPL')`) even for a single ticker, and no longer
  include `Adj Close` by default. The app now flattens columns defensively
  and only relies on `Open/High/Low/Close/Volume`.
- **Unscaled inputs/target**: the original code fed raw prices and volumes
  straight into the LSTM with no normalization, and used a `batch_size=1`
  over 30 epochs (extremely slow, and prone to unstable training). The app
  now scales both features and target with `MinMaxScaler` and inverse-transforms
  predictions back to dollar terms, which trains faster and much more
  reliably.
- **Incorrect input shape**: `input_shape=(xtrain.shape[1], 1)` was declared
  but the training array was never actually reshaped to 3D, which errors on
  current TensorFlow/Keras. The app explicitly reshapes `x` to
  `(samples, features, 1)` before fitting.
- **Chronological split**: switched `train_test_split` to `shuffle=False` so
  the test set is the most recent period, which is the appropriate way to
  validate a time-series model (the original shuffled randomly, letting the
  model "see the future" during training).
- **Hard-coded ticker/values**: the ticker, date range, model
  hyperparameters, and prediction inputs are all now editable in the UI
  instead of hard-coded.

## Disclaimer

This app is for educational purposes only. Predicting stock prices from
historical OHLCV data alone is inherently unreliable — please don't use this
for real investment decisions.
