"""
The aim of this project is to create a user-friendly portfolio optimization tools that utilizes
Efficient Frontier theory which is considered to be cornerstone of Modern Portfolio theory. The project also
acknowledges the limitations of this approach of optimization.
@author Pratyush Dixit
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from fredapi import Fred
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog

from Forecasting import Forecasting


def ForecastIndividualStocks(tickers):
    for ticker in tickers:
        forecaster = Forecasting(ticker)
        print(f"Forecasting for {ticker}...")
        _, forecast_series = forecaster.forecast(datetime.today())
        forecaster.plot_prediction(ticker, datetime.today())


# Function to get user input. Will be updated with better UI elements and create a search engine
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


'''
This function calculates the asset weights that maximize the Sharpe Ratio for a given portfolio.
The optimization ensures that no single asset constitutes more than 50% of the portfolio.
Utilizes Efficient Frontier from pypfopt.efficient_frontier

Parameters:
- closePrices (DataFrame): Historical closing prices of assets in the portfolio.
- risk (float): Risk-free rate used in the Sharpe Ratio calculation.

Returns:
- dict: Asset weights that maximize the Sharpe Ratio.
'''


def max_SharpeRatio_Optimizer(closePrices, risk):
    mu = expected_returns.mean_historical_return(closePrices)
    S = risk_models.sample_cov(closePrices)
    ef = EfficientFrontier(mu, S, (0, 0.5))
    results = ef.max_sharpe(risk)
    ef.portfolio_performance(verbose=True)
    return results


'''
Function to find the portfolio weights that minimize the portfolio's volatility given the provided historical closing prices.

Parameters:
    - closePrices (DataFrame): Historical closing prices of the securities in the portfolio.
    - risk (float): Risk-free rate to be used in the portfolio performance evaluation.

Returns:
    - results (dict): Optimized portfolio weights that minimize the volatility.

The function first calculates the expected returns and the sample covariance of the provided historical closing prices.
Using the Efficient Frontier method, it then optimizes the portfolio to minimize its volatility.
The constraint set is that no single security can occupy more than 50% of the portfolio.
'''


def min_Volatility_Optimizer(closePrices, risk):
    mu = expected_returns.mean_historical_return(closePrices)
    S = risk_models.sample_cov(closePrices)
    ef = EfficientFrontier(mu, S, (0, 0.5))
    results = ef.min_volatility()
    ef.portfolio_performance(verbose=True)
    return results


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
try:
    ten_years_Rates = fred.get_series_latest_release('GS10') / 100
    risk_FreeRate = ten_years_Rates.iloc[-1]
except Exception as e:
    risk_FreeRate = 0.045

sharpe_ratio = (port_ExpReturn - risk_FreeRate) / pot_standDev

cleaned_weights = max_SharpeRatio_Optimizer(adj_Close, risk_FreeRate)

print("Current Portfolio Information:")
print("Expected Return: " + str(round(port_ExpReturn, 4) * 100) + '%')
print("Standard Deviation/ Annual Volatility: " + str(round(pot_standDev, 4) * 100) + "%")
print("Sharpe Ratio:" + str(round(sharpe_ratio, 3)))
print("-------------------------------------------------------------------")
print("-------------------------------------------------------------------")
print("Optimized Portfolio")
print(cleaned_weights)

portfolio_weights = [cleaned_weights[ticker] for ticker in tickers]

# Enable interactive mode
plt.ion()

# Create a new figure for the optimized portfolio
filtered_weights_max_sharpe, filtered_tickers_max_sharpe = filter_zero_weights(portfolio_weights, tickers)
plt.figure(figsize=(6, 6))
plt.pie(filtered_weights_max_sharpe, labels=filtered_tickers_max_sharpe, autopct='%1.1f%%', startangle=140)
plt.title('Optimized Portfolio Weights')
plt.draw()  # Draw the figure but do not block execution

# Get the portfolio performance
min_vol_perf = min_Volatility_Optimizer(adj_Close, risk_FreeRate)

# Get the asset weights for the minimal volatility portfolio
min_vol_portfolio_weights = [min_vol_perf[ticker] for ticker in tickers]

# Create a new figure for the minimal volatility portfolio
filtered_weights_min_vol, filtered_tickers_min_vol = filter_zero_weights(min_vol_portfolio_weights, tickers)
plt.figure(figsize=(6, 6))
plt.pie(filtered_weights_min_vol, labels=filtered_tickers_min_vol, autopct='%1.1f%%', startangle=140)
plt.title('Minimal Volatility Portfolio Weights')
plt.draw()  # Draw the figure but do not block execution

print("Forecasting for the optimized portfolio for the next 6 months...")

ForecastIndividualStocks(filtered_tickers_max_sharpe)
# Wait for the user to close the plots
plt.show(block=True)  # Now block execution
