Portfolio Optimization and Forecasting Overview


This project aims to create user-friendly portfolio optimization tools that harness the Efficient Frontier theory, a cornerstone of the Modern Portfolio Theory. By doing so, the project offers an innovative way to make investment decisions by maximizing returns for a given level of risk. In addition to optimization, the tool offers forecasting capabilities for individual stock prices using the ARIMA model.

Features
Portfolio Optimization:
Utilizes Efficient Frontier theory.
Offers methods to both maximize the Sharpe ratio and minimize volatility.
Ensures that no single asset constitutes more than 50% of the portfolio to maintain diversification.
Stock Price Forecasting:
Uses ARIMA (AutoRegressive Integrated Moving Average) model for forecasting.
Capable of visual representation of historical vs. forecasted prices.
Short-Term Stock Prediction:
Uses the RandomForestClassifier for short-term stock price predictions.
Processes historical data to predict whether the stock price will increase the next day.
Operates on the last 40% of the data for predictions.
User-Friendly Input System:
Simple GUI-based input mechanism to obtain stock tickers and their weights from the user.
Interactive Visuals:
Provides pie-chart visualizations of the portfolio weights after optimization.
Plot capabilities for the forecasted stock prices.
Files
Main File: Contains the primary functionality of portfolio optimization and user input collection.
Forecasting.py: Houses the Forecasting class designed for stock price forecasting.
ShortTermPredictor.py: Houses the ShortTermPredictor class for short-term stock price predictions using the RandomForestClassifier.
Getting Started
Prerequisites
Ensure you have the following libraries installed:

pandas
numpy
datetime
yfinance
fredapi
pypfopt
matplotlib
tkinter
pmdarima
sklearn
How to Use
Run the main file.
Use the GUI prompt to enter stock tickers separated by commas.
Input the respective weights for the stocks when prompted.
The program will provide current portfolio statistics, optimized weights, and portfolio performance.
Visual representation of the optimized portfolio weights will be displayed.
The tool will forecast stock prices for the stocks in the optimized portfolio.
Limitations
The project acknowledges the inherent limitations of using the Efficient Frontier theory, as it assumes that future returns are distributed exactly as they were in the past. Real-world data can be erratic, and this method may not always provide optimal allocations.

Author
Pratyush Dixit


