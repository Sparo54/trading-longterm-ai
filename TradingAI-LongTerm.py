import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import messagebox
from datetime import datetime, timedelta

class StockTraderApp:
    def __init__(self, master):
        self.master = master
        master.title("Automated AI Long-term Stock Trading Tool")

        self.label = tk.Label(master, text="Select a Stock:")
        self.label.pack()

        self.stock_list = ["AAPL", "GOOGL", "AMZN"]  # Add more stocks as needed
        self.selected_stock = tk.StringVar(value=self.stock_list[0])
        self.stock_menu = tk.OptionMenu(master, self.selected_stock, *self.stock_list)
        self.stock_menu.pack()

        self.investment_label = tk.Label(master, text="Investment Amount:")
        self.investment_label.pack()
        self.investment_entry = tk.Entry(master)
        self.investment_entry.pack()

        self.risk_label = tk.Label(master, text="Risk Level (Low, Medium, High):")
        self.risk_label.pack()
        self.risk_entry = tk.Entry(master)
        self.risk_entry.pack()

        self.submit_button = tk.Button(master, text="Analyze", command=self.analyze_stock)
        self.submit_button.pack()

    def analyze_stock(self):
        stock_symbol = self.selected_stock.get()
        investment = float(self.investment_entry.get())
        risk_level = self.risk_entry.get()

        stock_data = yf.Ticker(stock_symbol)
        historical_data = stock_data.history(period="1y")

        # Perform preliminary analysis (This would be replaced with more complex AI analysis)
        # Here we only estimate based on recent price trends for the sake of example

        self.make_prediction(historical_data)

        messagebox.showinfo("Analysis Completed", "The analysis has been completed. Check console for results.")

    def make_prediction(self, historical_data):
        # Simple analysis: Predicting the next month's price based on linear regression
        historical_data['Date'] = historical_data.index.map(datetime.toordinal)  # Convert dates
        X = historical_data['Date'].values.reshape(-1, 1)
        Y = historical_data['Close'].values

        model = LinearRegression()
        model.fit(X, Y)

        # Predicting the price 30 days from now
        future_date = datetime.toordinal(datetime.now() + timedelta(days=30))
        predicted_price = model.predict([[future_date]])[0]

        print(f"Current Price: {historical_data['Close'][-1]}")
        print(f"Predicted Price in 30 days: {predicted_price}")
        print(f"Potential Gain: {(predicted_price - historical_data['Close'][-1]) * (int(self.investment_entry.get()) / historical_data['Close'][-1])}")
        print("P/E Ratio: N/A (Requires earnings data)")
        print("Market Sentiment Score: N/A (Requires sentiment analysis)")
        print(f"Recommendation: Buy if price <= {predicted_price}")
        print("Summary: Based on recent trends and a simple linear regression model.")

root = tk.Tk()
app = StockTraderApp(root)
root.mainloop()
