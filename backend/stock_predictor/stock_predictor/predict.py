import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# Function to run the model
def run_model(stock_data):
    stock_data = stock_data.tail(360)
    print("Stock Data:\n", stock_data.head())

    df = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    print("Selected Features:\n", df.head())

    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_data = close_scaler.fit_transform(df[['Close']])
    df['Close_Scaled'] = close_data
    
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = feature_scaler.fit_transform(df[['Open', 'High', 'Low', 'Volume']])
    df[['Open_Scaled', 'High_Scaled', 'Low_Scaled', 'Volume_Scaled']] = scaled_data
    
    scaled_data = df[['Open_Scaled', 'High_Scaled', 'Low_Scaled', 'Close_Scaled', 'Volume_Scaled']].values
    print("Scaled Data:\n", scaled_data[:5])

    def create_dataset(data, time_step=1):
        X, Y = [], []
        for i in range(len(data) - time_step - 1):
            a = data[i:(i + time_step)]
            X.append(a)
            Y.append(data[i + time_step, 3])
        return np.array(X), np.array(Y)

    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("First element of X:", X[0])
    print("First element of y:", y[0])

    # Flatten the X data for Random Forest
    n_samples, n_timesteps, n_features = X.shape
    X = X.reshape((n_samples, n_timesteps * n_features))

    # Initialize and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    predictions = model.predict(X)
    print("Predictions:\n", predictions[:5])

    predictions = close_scaler.inverse_transform(predictions.reshape(-1, 1))
    print("Inverse Transformed Predictions:\n", predictions[:5])

    yesterday_close = float(df['Close'].iloc[-2])
    today_close = float(df['Close'].iloc[-1])
    print("Yesterday's Close:", yesterday_close)
    print("Today's Close:", today_close)

    tomorrow_prediction = float(predictions[-1])
    print("Tomorrow's Prediction:", tomorrow_prediction)

    if tomorrow_prediction > today_close > yesterday_close:
        action = 'Buy'
    elif today_close > tomorrow_prediction and yesterday_close > today_close:
        action = 'Sell'
    else:
        action = 'Hold'

    # Calculate prediction accuracy for yesterday's close
    predicted_yesterday_close = float(predictions[-2])
    accuracy = (1 - abs(predicted_yesterday_close - yesterday_close) / yesterday_close) * 100
    print("Prediction Accuracy:", accuracy)

    result = {
        'yesterday_close': yesterday_close,
        'today_close': today_close,
        'tomorrow_prediction': tomorrow_prediction,
        'action': action,
        'prediction_accuracy': accuracy
    }

    return result

if __name__ == '__main__':
    sp500 = yf.Ticker("^GSPC")
    historical_data = sp500.history(period="max").reset_index()
    data = historical_data.tail(100)
    print(run_model(data))










