import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from datetime import datetime, timedelta

# Fetch stock data from Yahoo Finance
def get_stock_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    stock_data = stock.history(start=start, end=end)
    stock_data['Returns'] = stock_data['Close'].pct_change()
    stock_data.dropna(inplace=True)
    return stock_data

# Add technical indicators
def add_technical_indicators(data):
    # Simple Moving Average (SMA)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    
    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    data.dropna(inplace=True)
    return data

# Scrape sentiment from news articles
def scrape_sentiment(ticker):
    sources = [
        f"https://finance.yahoo.com/quote/{ticker}/news",
        f"https://www.marketwatch.com/investing/stock/{ticker}",
        f"https://www.cnbc.com/quotes/{ticker}",
    ]
    
    sentiment_score = 0
    article_count = 0
    
    for url in sources:
        try:
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            text = article.summary.lower()
            positive_words = ["growth", "strong", "profit", "bullish"]
            negative_words = ["loss", "decline", "bearish", "crash"]
            
            pos_count = sum(text.count(word) for word in positive_words)
            neg_count = sum(text.count(word) for word in negative_words)
            
            sentiment_score += (pos_count - neg_count)
            article_count += 1
        except Exception as e:
            print(f"Error fetching sentiment from {url}: {e}")
            continue
    
    avg_sentiment = sentiment_score / max(article_count, 1)
    return avg_sentiment

# Train a machine learning model
def train_model(data):
    # Define features and target variable
    features = ['SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_signal']
    data['Target'] = data['Close'].shift(-1)  # Predict the next day's close price
    data.dropna(inplace=True)
    
    X = data[features]
    y = data['Target']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

# Predict the next stock price
def predict_stock_price(model, latest_data):
    return model.predict([latest_data])

# Analyze the stock and make recommendations
def analyze_stock(ticker, investment_amount):
    start = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    end = datetime.today().strftime('%Y-%m-%d')
    
    # Get stock data and add technical indicators
    data = get_stock_data(ticker, start, end)
    data = add_technical_indicators(data)
    
    # Train the model
    model = train_model(data)
    
    # Get the latest data for prediction
    latest_data = data.iloc[-1][['SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_signal']].values
    predicted_price = predict_stock_price(model, latest_data)[0]
    
    # Current price
    current_price = data['Close'].iloc[-1]
    
    # Calculate potential gain
    potential_gain = (predicted_price - current_price) / current_price
    
    # Market sentiment
    sentiment_score = scrape_sentiment(ticker)
    
    # Make decision based on predictions and sentiment
    buy_signal = None
    sell_signal = None
    recommendation = "HOLD"
    
    if sentiment_score > 0 and potential_gain > 0.05:
        recommendation = "BUY"
        buy_signal = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    elif sentiment_score < 0 and potential_gain < -0.05:
        recommendation = "SELL"
        sell_signal = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    
    # Number of shares to buy
    shares_to_buy = int(investment_amount / current_price)
    
    # Display results
    print(f"📈 Stock: {ticker}")
    print(f"🔹 Current Price: ${current_price:.2f}")
    print(f"🔹 Predicted Price (Next Day): ${predicted_price:.2f}")
    print(f"🔹 Potential Gain: {potential_gain * 100:.2f}%")
    print(f"🔹 Sentiment Score: {sentiment_score:.2f}")
    print(f"🔹 Recommendation: {recommendation}")
    print(f"🔹 Shares to Buy: {shares_to_buy}")
    
    if buy_signal:
        print(f"✅ BUY at {buy_signal}")
    if sell_signal:
        print(f"❌ SELL at {sell_signal}")

# User input to analyze a stock
if __name__ == "__main__":
    ticker = input("Enter the stock ticker (e.g., AAPL): ").upper()
    investment_amount = float(input("Enter your investment amount in USD: "))
    analyze_stock(ticker, investment_amount)
