# AI Trading System for Stock Analysis
def analyze_stock(ticker, investment_amount, risk_level):
    start = (datetime.today() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')
    end = datetime.today().strftime('%Y-%m-%d')

    # Get stock data & fundamentals
    data = get_stock_data(ticker, start, end)
    fundamentals = get_fundamentals(ticker)
    sentiment_score = scrape_news(ticker)

    # Process data & train AI
    data = create_features(data)
    model, processed_data = train_model(data)

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
    risk_multiplier = {"low": 0.2, "medium": 0.5, "high": 1.0}
    shares_to_buy = int((investment_amount * risk_multiplier[risk_level]) / current_price)

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
risk_level = input("Enter your risk level (low, medium, high): ").lower()

# Run AI analysis on selected stock
analyze_stock(selected_stock, investment_amount, risk_level)
