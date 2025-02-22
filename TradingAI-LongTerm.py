import yfinance as yf
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np

def get_stock_data(tickers):
    """Fetch historical data for multiple stocks."""
    data = {}
    for ticker in tickers:
        stock = yf.download(ticker, period="5y", auto_adjust=True)
        stock.reset_index(inplace=True)
        data[ticker] = stock
    return data

def evaluate_stocks(data):
    """Evaluate stocks based on price performance and affordability."""
    stock_scores = {}
    for ticker, stock in data.items():
        stock['Returns'] = stock['Close'].pct_change()
        avg_return = stock['Returns'].mean() * 252  # Annualized return
        latest_price = stock['Close'].iloc[-1]
        
        # Strategy: Good annual return with an affordable price
        score = avg_return / latest_price  # Higher score means better value

        stock_scores[ticker] = score

    # Sort stocks by score in descending order
    sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_stocks

if __name__ == "__main__":
    # Define a list of stock tickers to analyze
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "FB", "NFLX", "NVDA", "INTC", "CSCO"]
    
    # Get stock data
    stock_data = get_stock_data(tickers)
    
    # Evaluate stocks and get recommendations
    recommendations = evaluate_stocks(stock_data)
    
    print("Top Stock Recommendations (Ticker: Score):")
    for ticker, score in recommendations:
        print(f"{ticker}: {score:.4f}")
