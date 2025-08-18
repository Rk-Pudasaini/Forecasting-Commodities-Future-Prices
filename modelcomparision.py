"""
EDA + Forecasting Commodities Future Prices (Kalimati Market)
Python script to perform Exploratory Data Analysis (EDA) and apply ARIMA, SARIMA, Holt-Winters, Random Forest, XGBoost, and LSTM models.

Usage:
- Place Kalimati CSV in same folder and set CSV_PATH.
- CSV must contain columns: Date, Commodity, Average.
- Recommended packages: pandas, numpy, matplotlib, statsmodels, scikit-learn, xgboost (optional), tensorflow (for LSTM), pmdarima (optional)

Outputs:
- Diagnostic plots (if run interactively).
- Forecast CSVs in ./outputs/ and a summary metrics CSV comparing models.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Optional imports
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    HAS_TF = True
except Exception:
    HAS_TF = False

# ---------------- User settings ----------------
CSV_PATH = 'kalimati_prices.csv'
COMMODITY = 'Potato'
DATE_COL = 'Date'
PRICE_COL = 'Average'
FREQ = 'D'          # 'D' daily, 'M' monthly
TEST_PERIODS = 30
LAG_DAYS = 14       # how many lag features to create for ML models
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Helpers ----------------

def load_data(path, commodity):
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    if 'Commodity' in df.columns and commodity is not None:
        df = df[df['Commodity'] == commodity]
    df = df.sort_values(DATE_COL)
    df = df[[DATE_COL, PRICE_COL]].dropna()
    df = df.set_index(DATE_COL).asfreq(FREQ)
    return df


def adf_test(series, title=''):
    result = adfuller(series.dropna())
    print(f"ADF p-value for {title}: {result[1]:.4f}")
    return result[1]


def evaluate(true, pred):
    rmse = np.sqrt(mean_squared_error(true, pred))
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return {'RMSE': rmse, 'MAPE': mape}


def create_lag_features(series, lags):
    df = pd.DataFrame(series).copy()
    for i in range(1, lags+1):
        df[f'lag_{i}'] = df[PRICE_COL].shift(i)
    # rolling features
    df['rolling_mean_7'] = df[PRICE_COL].rolling(window=7).mean()
    df['rolling_std_7'] = df[PRICE_COL].rolling(window=7).std()
    df = df.dropna()
    return df


def make_lstm_sequences(values, seq_len):
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i:i+seq_len])
        y.append(values[i+seq_len])
    return np.array(X), np.array(y)

# ---------------- Load Data ----------------
print('Loading data...')
df = load_data(CSV_PATH, COMMODITY)
print(f'Rows: {len(df)} | From {df.index.min().date()} to {df.index.max().date()}')

# Basic EDA
print('
Missing values:')
print(df.isna().sum())
print('
Summary:')
print(df.describe())

plt.figure(figsize=(12,4))
plt.plot(df.index, df[PRICE_COL])
plt.title(f'{COMMODITY} Prices')
plt.show()

# Decompose (choose period depending on freq)
period = 12 if FREQ == 'M' else 365
try:
    decomposition = seasonal_decompose(df[PRICE_COL].dropna(), model='additive', period=period)
    decomposition.plot()
    plt.show()
except Exception:
    pass

# ADF
pval = adf_test(df[PRICE_COL], COMMODITY)
if pval > 0.05:
    print('Series not stationary -> differencing once')
    df['diff_1'] = df[PRICE_COL].diff()

# Train-Test split
train = df.iloc[:-TEST_PERIODS]
test = df.iloc[-TEST_PERIODS:]

# ---------------- Traditional models ----------------
results = {}

# 1) ARIMA (simple fallback order)
print('
Fitting ARIMA...')
try:
    arima = ARIMA(train[PRICE_COL], order=(1,1,1)).fit()
    fc_arima = arima.forecast(steps=TEST_PERIODS)
    arima_metrics = evaluate(test[PRICE_COL].values, fc_arima.values)
    results['ARIMA'] = {'pred': pd.Series(fc_arima.values, index=test.index), 'metrics': arima_metrics}
    print('ARIMA ->', arima_metrics)
except Exception as e:
    print('ARIMA failed:', e)

# 2) SARIMA (seasonal)
print('
Fitting SARIMA...')
try:
    seasonal_period = 12 if FREQ=='M' else 7
    sarima = SARIMAX(train[PRICE_COL], order=(1,1,1), seasonal_order=(1,1,1,seasonal_period), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc_sarima = sarima.forecast(steps=TEST_PERIODS)
    sarima_metrics = evaluate(test[PRICE_COL].values, fc_sarima.values)
    results['SARIMA'] = {'pred': pd.Series(fc_sarima.values, index=test.index), 'metrics': sarima_metrics}
    print('SARIMA ->', sarima_metrics)
except Exception as e:
    print('SARIMA failed:', e)

# 3) Holt-Winters
print('
Fitting Holt-Winters...')
try:
    hw = ExponentialSmoothing(train[PRICE_COL], trend='add', seasonal='add', seasonal_periods=seasonal_period).fit()
    fc_hw = hw.forecast(TEST_PERIODS)
    hw_metrics = evaluate(test[PRICE_COL].values, fc_hw.values)
    results['HoltWinters'] = {'pred': pd.Series(fc_hw.values, index=test.index), 'metrics': hw_metrics}
    print('Holt-Winters ->', hw_metrics)
except Exception as e:
    print('Holt-Winters failed:', e)

# ---------------- Feature-based ML models ----------------
print('
Preparing lag features for ML models...')
ml_df = create_lag_features(df[PRICE_COL], LAG_DAYS)
ml_train = ml_df.loc[train.index]
ml_test = ml_df.loc[test.index]

feature_cols = [c for c in ml_df.columns if c != PRICE_COL]
X_train = ml_train[feature_cols].values
y_train = ml_train[PRICE_COL].values
X_test = ml_test[feature_cols].values
y_test = ml_test[PRICE_COL].values

# 4) Random Forest
print('
Training Random Forest...')
try:
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_metrics = evaluate(y_test, rf_pred)
    results['RandomForest'] = {'pred': pd.Series(rf_pred, index=test.index), 'metrics': rf_metrics}
    print('Random Forest ->', rf_metrics)
except Exception as e:
    print('Random Forest failed:', e)

# 5) XGBoost
if HAS_XGB:
    print('
Training XGBoost...')
    try:
        xgb = XGBRegressor(n_estimators=200, objective='reg:squarederror', random_state=42)
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_metrics = evaluate(y_test, xgb_pred)
        results['XGBoost'] = {'pred': pd.Series(xgb_pred, index=test.index), 'metrics': xgb_metrics}
        print('XGBoost ->', xgb_metrics)
    except Exception as e:
        print('XGBoost failed:', e)
else:
    print('
XGBoost not installed - skipping')

# ---------------- LSTM (sequence model) ----------------
if HAS_TF:
    print('
Preparing and training LSTM...')
    series_values = df[PRICE_COL].dropna().values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series_values)

    seq_len = 14
    X_seq, y_seq = make_lstm_sequences(scaled.flatten(), seq_len)
    # align with dates
    seq_dates = df.dropna().index[seq_len:]
    X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))

    # split
    split_idx = len(X_seq) - TEST_PERIODS
    X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
    y_train_seq, y_test_seq = y_seq[:split_idx], y_seq[split_idx:]
    test_dates = seq_dates[split_idx:]

    model = Sequential()
    model.add(LSTM(64, input_shape=(seq_len,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_seq, y_train_seq, epochs=30, batch_size=16, verbose=0)

    lstm_pred_scaled = model.predict(X_test_seq).flatten()
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled.reshape(-1,1)).flatten()

    # Align test_dates with main test index length
    # If lengths mismatch, map predictions to the last TEST_PERIODS of df
    if len(lstm_pred) >= TEST_PERIODS:
        lstm_pred_series = pd.Series(lstm_pred[-TEST_PERIODS:], index=test.index)
    else:
        # pad or trim
        lstm_pred_series = pd.Series(np.nan, index=test.index)
        lstm_pred_series.iloc[-len(lstm_pred):] = lstm_pred

    lstm_metrics = evaluate(test[PRICE_COL].values, lstm_pred_series.values)
    results['LSTM'] = {'pred': lstm_pred_series, 'metrics': lstm_metrics}
    print('LSTM ->', lstm_metrics)
else:
    print('
TensorFlow not installed - skipping LSTM')

# ---------------- Compare & Save ----------------
print('
Comparing models...')
metrics_rows = []
for name, info in results.items():
    m = info['metrics']
    metrics_rows.append({'Model': name, 'RMSE': m['RMSE'], 'MAPE': m['MAPE']})

metrics_df = pd.DataFrame(metrics_rows).sort_values('RMSE')
metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'model_metrics.csv'), index=False)
print(metrics_df)

best_model = metrics_df.iloc[0]['Model'] if not metrics_df.empty else None
print('
Best model by RMSE:', best_model)

# Save predictions
for name, info in results.items():
    pred = info['pred']
    pred_df = pd.DataFrame({f'{name}_pred': pred})
    pred_df.to_csv(os.path.join(OUTPUT_DIR, f'pred_{name}.csv'))

# Plot comparison
plt.figure(figsize=(12,5))
plt.plot(df.index, df[PRICE_COL], label='Actual')
for name, info in results.items():
    plt.plot(info['pred'].index, info['pred'].values, label=name)
plt.legend()
plt.title(f'Model Comparison - {COMMODITY}')
plt.show()

print('
All outputs saved to', OUTPUT_DIR)
print('Done.')
