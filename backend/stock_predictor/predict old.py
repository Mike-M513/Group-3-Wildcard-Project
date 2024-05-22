import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Define the LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(self.dropout(out[:, -1, :]))
        return out

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 5
hidden_size = 100
num_layers = 3
output_size = 1
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

# Define the criterion and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

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

    X = torch.FloatTensor(X).to(device)
    y = torch.FloatTensor(y).view(-1, 1).to(device)
    print("First PyTorch tensor X:", X[0])
    print("First PyTorch tensor y:", y[0])

    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X)
        optimizer.zero_grad()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        predictions = model(X).cpu().numpy()
    print("Predictions:\n", predictions[:5])

    predictions = close_scaler.inverse_transform(predictions)
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









