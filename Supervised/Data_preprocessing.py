import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('/Users/Garry/Year3/Third-Year-Project/Data/Tesla Stock Price History.csv')

print(df.head())

df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)
df['log_Price'] = np.log(df['Price'])
df['day_of_week'] = df.index.dayofweek  
df['month'] = df.index.month
df['day_of_month'] = df.index.day


df['SMA_7'] = df['Price'].rolling(window=7, min_periods=1).mean()
df['log_SMA_7'] = np.log(df['SMA_7'])

df['Return'] = df['Price'].pct_change().fillna(0)

df['lag1_log_Price'] = df['log_Price'].shift(1)

df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

feature_columns = ['day_of_week', 'month', 'day_of_month', 
                   'lag1_log_Price', 'Return', 'log_SMA_7']
target_column = 'log_Price'

data_for_model = df[feature_columns + [target_column]].copy()


train_split = 0.7
n_train = int(len(data_for_model) * train_split)
train_data = data_for_model.iloc[:n_train]
test_data = data_for_model.iloc[n_train:]

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

feature_scaler.fit(train_data[feature_columns])
target_scaler.fit(train_data[[target_column]])

train_data_scaled = train_data.copy()
test_data_scaled = test_data.copy()

train_data_scaled[feature_columns] = feature_scaler.transform(train_data[feature_columns])
train_data_scaled[target_column] = target_scaler.transform(train_data[[target_column]])

test_data_scaled[feature_columns] = feature_scaler.transform(test_data[feature_columns])
test_data_scaled[target_column] = target_scaler.transform(test_data[[target_column]])

def create_sliding_window(data, sequence_length, feature_cols, target_col):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[feature_cols].iloc[i: i + sequence_length].values)
        y.append(data[target_col].iloc[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 20

X_train, y_train = create_sliding_window(train_data_scaled, sequence_length, feature_columns, target_column)

X_test, y_test = create_sliding_window(test_data_scaled, sequence_length, feature_columns, target_column)

print("Train features shape:", X_train.shape)
print("Train target shape:", y_train.shape)
print("Test features shape:", X_test.shape)
print("Test target shape:", y_test.shape)
