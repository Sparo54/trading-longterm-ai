import yfinance as yf
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from newspaper import Article
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
 
# Import TradingView stock directory
def get_tradingview_stocks():
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "BA"
    ]  # Example stock list
 
# Fetch stock data
def get_stock_data(ticker, start, end):
    stock = yf.download(ticker, start=start, end=end, auto_adjust=True)  # auto_adjust=True to adjust stock data
    stock['Returns'] = stock['Close'].pct_change()  # Use 'Close' instead of 'Adj Close' for the returns
    return stock.dropna()
 
# Fundamental Analysis (P/E Ratio, Earnings Growth)
def get_fundamentals(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/key-statistics"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "html.parser")
 
    pe_ratio = None
    pe_element = soup.find(text="Trailing P/E")
    if pe_element:
        pe_ratio = pe_element.find_next("td").text
 
    return {"PE Ratio": pe_ratio}
 
# Scrape news from multiple sources for sentiment analysis
def scrape_news(ticker):
    sources = [
        f"https://finance.yahoo.com/quote/{ticker}/news",
        f"https://www.marketwatch.com/investing/stock/{ticker}",
        f"https://www.cnbc.com/quotes/{ticker}",
        f"https://www.nasdaq.com/market-activity/stocks/{ticker}/news",
        f"https://seekingalpha.com/symbol/{ticker}/news"
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
            positive_words = ["growth", "bullish", "strong", "rally", "profit"]
            negative_words = ["loss", "bearish", "decline", "sell-off", "crash"]
 
            pos_count = sum(text.count(word) for word in positive_words)
            neg_count = sum(text.count(word) for word in negative_words)
 
            sentiment_score += (pos_count - neg_count)
            article_count += 1
        except Exception as e:  # Specify the exception to catch
            print(f"Error processing article: {e}")
            continue
 
    avg_sentiment = sentiment_score / max(article_count, 1)
    return avg_sentiment
 
# Feature engineering
def create_features(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['Volatility'] = data['Returns'].rolling(window=30).std()
    data.dropna(inplace=True)
    return data
 
# Train model to predict stock prices
def train_model(data):
    data['Target'] = data['Close'].shift(-30)  # Predicting 30 days ahead
    data.dropna(inplace=True)
 
    features = ['SMA_50', 'SMA_200', 'Volatility']
    X = data[features]
    y = data['Target']
 
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
 
    return model
 
# Example usage
if __name__ == "__main__":
    ticker = "AAPL"
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
 
    stock_data = get_stock_data(ticker, start_date, end_date)
    stock_data = create_features(stock_data)
    model = train_model(stock_data)
    fundamentals = get_fundamentals(ticker)
    sentiment = scrape_news(ticker)
 
    print(f"Fundamentals: {fundamentals}")
    print(f"Sentiment Score: {sentiment}")
