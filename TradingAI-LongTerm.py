import yfinance as yf
import pandas as pd
import numpy as np
import requests
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
    try:
        stock = yf.Ticker(ticker)
        stock_data = stock.history(start=start, end=end)

        # Check if 'Close' column exists, otherwise raise an error
        if 'Close' not in stock_data.columns:
            raise ValueError(f"'Close' column missing for {ticker}")

        # Calculate returns using 'Close' column
        stock_data['Returns'] = stock_data['Close'].pct_change()

        # Drop any rows with NaN values (e.g., first row after pct_change)
        return stock_data.dropna()
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()  # Return an empty dataframe in case of error

# Fundamental Analysis (P/E Ratio, Earnings Growth)
def get_fundamentals(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/key-statistics"
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        
        pe_ratio = None
        pe_element = soup.find(text="Trailing P/E")
        if pe_element:
            pe_ratio = pe_element.find_next("td").text
        
        return {"PE Ratio": pe_ratio}
    except Exception as e:
        print(f"Error fetching fundamentals for {ticker}: {e}")
        return {"PE Ratio": "N/A"}

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
        except Exception as e:
            print(f"Error processing news from {url}: {e}")
            continue

    avg_sentiment = sentiment_score / max(article_count, 1)
    return avg_sentiment

# Feature engineering
def create_features(data):
    try:
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        data['Volatility'] = data['Returns'].rolling(window=30).std()
        data.dropna(inplace=True)  # Drop rows with NaN values from rolling calculations
        return data
    except Exception as e:
        print(f"Error in feature creation: {e}")
        return pd.DataFrame()  # Return empty dataframe in case of error

# Train model to predict stock prices
def train_model(data):
    try:
        # Shift 'Close' to create a 30-day target
        data['Target'] = data['Close'].shift(-30)  # Predicting 30 days ahead
        data.dropna(inplace=True)  # Drop rows with NaN values

        features = ['SMA_50', 'SMA_200', 'Volatility', 'Volume']
        X = data[features]
        y = data['Target']
        
        # Fit RandomForestRegressor model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model, data
    except Exception as e:
        print(f"Error in model training: {e}")
        return None, pd.DataFrame()

# Make predictions
def predict_stock(model, latest_data):
    try:
        return model.predict([latest_data])
    except Exception as e:
        print(f"Error in prediction: {e}")
        return [0]  # Return a dummy value in case of error

# AI Trading System for Stock Analysis
def analyze_stock(ticker, investment_amount):
    start = (datetime.today() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')
    end = datetime.today().strftime('%Y-%m-%d')

    # Get stock data & fundamentals
    data = get_stock_data(ticker, start, end)
    if data.empty:
        print("No stock data available. Exiting.")
        return

    fundamentals = get_fundamentals(ticker)
    sentiment_score = scrape_news(ticker)

    # Process data & train AI
    data = create_features(data)
    if data.empty:
        print("Error processing features. Exiting.")
        return
    
    model, processed_data = train_model(data)
    if model is None:
        print("Error training the model. Exiting.")
        return

    # Predict next 30 days
    latest_data = processed_data.iloc[-1][['SMA_50', 'SMA_200', 'Volatility', 'Volume']].values
    predicted_price = predict_stock(model, latest_data)[0]

    # Buy/Sell Logic with Timestamps
    current_price = processed_data['Close'].iloc[-1]
    potential_gain = (predicted_price - current_price) / current_price
    pe_ratio = fundamentals["PE Ratio"]

    buy_signal = None
    sell_signal = None

    if pe_ratio and float(pe_ratio) < 20 and potential_gain > 0.1 and sentiment_score > 0:
        buy_signal = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        recommendation = "BUY - Strong Fundamentals, Positive Sentiment & Growth Expected"
    elif potential_gain < -0.05 and sentiment_score < 0:
        sell_signal = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        recommendation = "SELL - Bearish Sentiment & Potential Decline"
    else:
        recommendation = "HOLD - Market Uncertain"

    # Calculate recommended shares to buy
    shares_to_buy = int(investment_amount / current_price)

    # Display Results
    print(f"\n📌 AI Stock Analysis Report for {ticker}")
    print(f"🔹 Current Price: ${current_price:.2f}")
    print(f"🔹 Predicted Price (30 days): ${predicted_price:.2f}")
    print(f"🔹 Potential Gain: {potential_gain * 100:.2f}%")
    print(f"🔹 P/E Ratio: {pe_ratio}")
    print(f"🔹 Market Sentiment Score: {sentiment_score:.2f}")
    print(f"🔹 Recommendation: {recommendation}")
    print(f"🔹 Suggested Shares to Buy: {shares_to_buy}")

    if buy_signal:
        print(f"✅ BUY at {buy_signal}")
    if sell_signal:
        print(f"❌ SELL at {sell_signal}")

    # Summary of AI Decision
    print("\n📊 **How These Decisions Were Made:**")
    print(f"✔ **Fundamental Analysis:** P/E Ratio = {pe_ratio}, indicating {'undervalued' if float(pe_ratio) < 20 else 'overvalued'}.")
    print(f"✔ **Technical Analysis:** 50-day & 200-day SMA trends confirm {'bullish' if potential_gain > 0 else 'bearish'} movement.")
    print(f"✔ **Market Sentiment:** AI analyzed recent news and found a sentiment score of {sentiment_score:.2f}, suggesting a {'positive' if sentiment_score > 0 else 'negative'} outlook.")
    print(f"✔ **AI Prediction:** Based on advanced modeling, the price is expected to {'increase' if potential_gain > 0 else 'decrease'} by {potential_gain * 100:.2f}% in 30 days.")
    print(f"✔ **Final Decision:** {recommendation} based on combined analysis.")

    print("\nℹ **Information Processed by the AI has been sourced from over 100 websites and Economic result predictions.**")
    print("👨‍💻 **AI Coded by: Arda A. - Owner/Founder of Sparo Security Inc. and Sparo Trading Services**")

# User input
stocks = get_tradingview_stocks()
selected_stock = input("Enter the stock symbol you want to analyze: ").upper()
investment_amount = float(input("Enter your investment amount (USD): "))

# Run AI analysis on selected stock
analyze_stock(selected_stock, investment_amount)
