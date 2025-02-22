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
        investment_amount = float(self.investment_entry.get())
        risk_level = self.risk_entry.get().lower()

        stock_data = self.get_stock_data(stock_symbol)
        if stock_data is None:
            messagebox.showerror("Error", "Could not fetch data. Please check the stock symbol.")
            return

        self.make_prediction(stock_data, investment_amount, risk_level)

    def get_stock_data(self, ticker):
        stock = yf.download(ticker, period="1y", auto_adjust=True)
        print(stock.columns)  # For debugging purposes
        return stock.reset_index()  # Reset index to get 'Date' as a column

    def make_prediction(self, stock_data, investment_amount, risk_level):
        # Prepare data for model
        stock_data['Date'] = stock_data['Date'].map(datetime.toordinal)  # Convert dates
        X = stock_data['Date'].values.reshape(-1, 1)
        Y = stock_data['Close'].values

        model = LinearRegression()
        model.fit(X, Y)

        # Predicting the price 30 days from now
        future_date = datetime.toordinal(datetime.now() + timedelta(days=30))
        predicted_price = model.predict([[future_date]])[0]

        current_price = stock_data['Close'].iloc[-1]
        potential_gain = ((predicted_price - current_price) / current_price) * investment_amount

        print(f"Current Price: {current_price:.2f}")
        print(f"Predicted Price in 30 days: {predicted_price:.2f}")
        print(f"Potential Gain: {potential_gain:.2f}")
        print("Recommendation: Buy if price <= {:.2f}".format(predicted_price))
        print("Summary: Based on recent trends and a simple linear regression model.")

        messagebox.showinfo("Analysis Results", f"Current Price: {current_price:.2f}\n"
                                                  f"Predicted Price in 30 days: {predicted_price:.2f}\n"
                                                  f"Potential Gain: {potential_gain:.2f}\n"
                                                  "Recommendations:\n"
                                                  f"Buy if price <= {predicted_price:.2f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = StockTraderApp(root)
    root.mainloop()
