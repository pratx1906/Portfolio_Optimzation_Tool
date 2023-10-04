# The aim of this project is to create a user-friendly portfolio optimization tools that utilizes
# Efficient Frontier theory which is considered to be cornerstone of Modern Portfolio theory. The project also
# acknowledges the limitations of this approach of optimization.

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import matplotlib.pyplot as plt
from fredapi import Fred
import tkinter as tk
from tkinter import simpledialog


# Function to get user input
def get_user_input():
    root = tk.Tk()
    root.withdraw()

    user_tickers = simpledialog.askstring("Input", "Please enter tickers separated by commas:")
    user_weights = simpledialog.askstring("Input", "Please enter weights separated by commas:")

    target_tickers = user_tickers.split(",")
    target_Weights = np.array([float(weight) for weight in user_weights.split(",")])

    return target_tickers, target_Weights


# Function to filter out zero-weighted securities
def filter_zero_weights(input_weights, input_Tickers):
    non_zero_indices = [i for i, weight in enumerate(input_weights) if weight > 0]
    filtered_weights = [input_weights[i] for i in non_zero_indices]
    filtered_tickers = [input_Tickers[i] for i in non_zero_indices]
    return filtered_weights, filtered_tickers


tickers, weights = get_user_input()
#  Five years of data for portfolio analysis
endDate = datetime.today()
startDate = endDate - timedelta(days=5 * 365)
adj_Close = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker, start=startDate, end=endDate)
    adj_Close[ticker] = data['Adj Close']

# Calculate daily returns/ using log normal returns
log_Returns = np.log(adj_Close / adj_Close.shift(1))
log_Returns = log_Returns.dropna()

# covariance Matrix Creation
cov_Matrix = log_Returns.cov() * 252

# Current portfolio statistics:
# Standard Deviation
pot_standDev = np.sqrt(np.dot(weights.T, np.dot(cov_Matrix, weights)))

# Expected Return
port_ExpReturn = np.sum(log_Returns.mean() * weights) * 252

# Risk Free Rate for last 10 years
fred = Fred(api_key='2cd2c8f11c41aa57740ed799fb5a5635')
ten_years_Rates = fred.get_series_latest_release('GS10') / 100
risk_FreeRate = ten_years_Rates.iloc[-1]
sharpe_ratio = (port_ExpReturn - risk_FreeRate) / pot_standDev

# Portfolio optimization
mu = expected_returns.mean_historical_return(adj_Close)
S = risk_models.sample_cov(adj_Close)
# Optimize sharpe ratio
ef = EfficientFrontier(mu, S)

# No single security should not more than 50% of the portfolio.
ef.add_constraint(lambda x: x <= 0.5)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

print("Current Portfolio Information:")
print("Expected Return: " + str(round(port_ExpReturn, 4) * 100) + '%')
print("Standard Deviation/ Annual Volatility: " + str(round(pot_standDev, 4) * 100) + "%")
print("Sharpe Ratio:" + str(round(sharpe_ratio, 3)))
print("-------------------------------------------------------------------")
print("-------------------------------------------------------------------")
print("Optimized Portfolio")
print(cleaned_weights)

ef.portfolio_performance(verbose=True)
portfolio_weights = [cleaned_weights[ticker] for ticker in tickers]

# Enable interactive mode
plt.ion()

# Create a new figure for the optimized portfolio
filtered_weights_max_sharpe, filtered_tickers_max_sharpe = filter_zero_weights(portfolio_weights, tickers)
plt.figure(figsize=(6, 6))
plt.pie(filtered_weights_max_sharpe, labels=filtered_tickers_max_sharpe, autopct='%1.1f%%', startangle=140)
plt.title('Optimized Portfolio Weights')
plt.draw()  # Draw the figure but do not block execution

# Optimize for minimal volatility
ef_min_vol = EfficientFrontier(mu, S)
# No single security should not more than 50% of the portfolio
ef_min_vol.add_constraint(lambda x: x <= 0.5)
min_vol_weights = ef_min_vol.min_volatility()
cleaned_min_vol_weights = ef_min_vol.clean_weights()

# Get the portfolio performance
min_vol_perf = ef_min_vol.portfolio_performance(verbose=True)

# Get the asset weights for the minimal volatility portfolio
min_vol_portfolio_weights = [cleaned_min_vol_weights[ticker] for ticker in tickers]

# Create a new figure for the minimal volatility portfolio
filtered_weights_min_vol, filtered_tickers_min_vol = filter_zero_weights(min_vol_portfolio_weights, tickers)
plt.figure(figsize=(6, 6))
plt.pie(filtered_weights_min_vol, labels=filtered_tickers_min_vol, autopct='%1.1f%%', startangle=140)
plt.title('Minimal Volatility Portfolio Weights')
plt.draw()  # Draw the figure but do not block execution

# Wait for the user to close the plots
plt.show(block=True)  # Now block execution
