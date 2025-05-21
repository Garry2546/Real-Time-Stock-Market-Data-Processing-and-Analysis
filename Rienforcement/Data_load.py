import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path, split_ratio=0.8, window_size=20):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'].str.strip(), infer_datetime_format=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    numeric_cols = ['Price', 'Open', 'High', 'Low']
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    if 'Vol.' in df.columns:
        df['Volume'] = (df['Vol.']
                        .replace({'M': '*1e6', 'B': '*1e9'}, regex=True)
                        .map(pd.eval).astype(float))
    else:
        df['Volume'] = np.nan

    df['Change'] = df['Change %'].str.replace('%', '').astype(float) / 100
    if 'Sentiment' not in df.columns:
        df['Sentiment'] = 0.0

    df = df.rename(columns={
        'Price': 'Close_Price',
        'Open': 'Open_Price',
        'High': 'High_Price',
        'Low': 'Low_Price'
    }).drop(columns=['Vol.', 'Change %'], errors='ignore')

    df['MA_10'] = df['Close_Price'].rolling(window=10).mean()
    df['MA_50'] = df['Close_Price'].rolling(window=50).mean()

    def compute_rsi(prices, window=14):
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean().replace(0, 1e-9)
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    df['RSI'] = compute_rsi(df['Close_Price'])

    df['EMA_12'] = df['Close_Price'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close_Price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Volatility'] = df['Close_Price'].pct_change().rolling(window=window_size).std()
    df['ROC_10'] = df['Close_Price'].pct_change(periods=10)
    df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()

    for lag in [1, 2, 3, 5]:
        df[f'Return_{lag}D'] = df['Close_Price'].pct_change(periods=lag)

    df = df.dropna().reset_index(drop=True)

    split_index = int(len(df) * split_ratio)
    train_df = df.iloc[:split_index].reset_index(drop=True)
    test_df = df.iloc[split_index:].reset_index(drop=True)

    scale_cols = ['Close_Price', 'Volume', 'MA_10', 'MA_50', 'RSI', 'MACD', 'Volatility',
                  'ROC_10', 'Volume_MA_10', 'Sentiment']
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
    test_df[scale_cols] = scaler.transform(test_df[scale_cols])

    return train_df, test_df, scaler
