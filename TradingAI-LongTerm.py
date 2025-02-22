import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

def get_stock_data(ticker):
    """Fetch stock data from Yahoo Finance."""
    try:
        stock = yf.download(ticker, period="1y", auto_adjust=True)
        stock.reset_index(inplace=True)  # Reset index to make 'Date' a column
        print(stock.columns)  # For debugging to check available columns
        return stock
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

def make_prediction(stock_data):
    """Make a stock price prediction using linear regression."""
    if 'Close' not in stock_data.columns:
        print("Error: 'Close' data not found in fetched data.")
        return

    stock_data['Date'] = stock_data['Date'].map(datetime.toordinal)  # Convert dates
    X = stock_data['Date'].values.reshape(-1, 1)
    Y = stock_data['Close'].values

    model = LinearRegression()
    model.fit(X, Y)

    # Predicting the price 30 days from now
    future_date = datetime.toordinal(datetime.now() + timedelta(days=30))
    predicted_price = model.predict([[future_date]])[0]

    current_price = stock_data['Close'].iloc[-1]
    potential_gain = ((predicted_price - current_price) / current_price) * 100  # Percentage gain

    print(f"Current Price: {current_price:.2f}")
    print(f"Predicted Price in 30 days: {predicted_price:.2f}")
    print(f"Potential Gain (%): {potential_gain:.2f}")
    print("Recommendation: Buy if price <= {:.2f}".format(predicted_price))

if __name__ == "__main__":
    ticker_symbol = input("Enter the stock symbol (e.g., AAPL, GOOGL): ")
    stock_data = get_stock_data(ticker_symbol)
    
    if stock_data is not None:
        make_prediction(stock_data)
    else:
        print("No data available for the specified stock.")
