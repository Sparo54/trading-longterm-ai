import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Download stock data using yfinance
def get_stock_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    stock_data = stock.history(start=start, end=end)
    return stock_data

# Prepare the data for the model
def prepare_data(data):
    data['Date'] = data.index
    data['Date'] = data['Date'].map(lambda x: x.toordinal())  # Convert dates to numeric values

    # We will predict the 'Close' price for the next day
    X = data[['Date']]  # Use the Date as the feature (independent variable)
    y = data['Close']  # Close price as the target variable (dependent variable)
    
    return X, y

# Train the Linear Regression model
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Make prediction for the next day
def predict_next_day_price(model, latest_date):
    next_day = latest_date + timedelta(days=1)
    next_day_ordinal = next_day.toordinal()
    predicted_price = model.predict([[next_day_ordinal]])
    return predicted_price[0]

# Plot the stock data and prediction
def plot_data(data, predicted_price, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Close'], label='Historical Prices', color='blue')
    plt.scatter([data['Date'].iloc[-1] + 1], [predicted_price], color='red', label='Predicted Next Day Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.legend()
    plt.show()

# Main function to get input, train model and predict stock price
def stock_predictor(ticker):
    # Define the time range for data (last 2 years)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years of data

    # Get stock data
    stock_data = get_stock_data(ticker, start_date, end_date)

    # Prepare the data for training
    X, y = prepare_data(stock_data)

    # Train the model
    model = train_model(X, y)

    # Get prediction for the next day
    latest_date = stock_data.index[-1]
    predicted_price = predict_next_day_price(model, latest_date)

    # Print prediction result
    print(f"📈 Stock: {ticker}")
    print(f"🔹 Predicted Next Day Price: ${predicted_price:.2f}")
    
    # Plot the stock data and predicted price
    plot_data(stock_data, predicted_price, ticker)

# User input for the stock ticker
if __name__ == "__main__":
    ticker = input("Enter the stock ticker (e.g., AAPL for Apple): ").upper()
    stock_predictor(ticker)
