from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from pmdarima import auto_arima
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error


class Forecasting:
    def __init__(self, ticker):
        self.ticker = ticker

    def _get_data(self, start_date, end_date):
        data = yf.download(self.ticker, start=end_date, end=start_date)
        return data['Close']

    def _calculate_returns(self, prices):
        # Calculate daily returns
        returns = prices.pct_change().dropna()
        return returns

    def forecast(self, start_date, forecast_period=90, split_ratio=0.8):
        # Adjust end date for 5 years before the start_date
        end_date = start_date - timedelta(days=5 * 365)

        # Fetch the historical data
        closing_prices = self._get_data(start_date, end_date)

        # Split the data into training and test sets
        train_size = int(len(closing_prices) * split_ratio)
        train, test = closing_prices[:train_size], closing_prices[train_size:]

        # Use the training dataset to determine the best ARIMA order
        stepwise_fit = auto_arima(closing_prices, start_p=1, start_q=1, max_p=5, max_q=5, m=12, start_P=0, seasonal=True,
                                  d=1, D=1, trace=True, suppress_warnings=True)

        # Forecasting the prices
        forecasted_prices = stepwise_fit.predict(n_periods=forecast_period)

        # Creating a date range for the forecasted data
        forecast_index = pd.date_range(start=train.index[-1], periods=forecast_period + 1)[1:]

        forecast_series = pd.Series(forecasted_prices, index=forecast_index)
        return closing_prices, forecast_series

    def plot_prediction(self, ticker, start_date, forecast_period=90):
        historical_data, forecast_series = self.forecast(start_date, forecast_period)

        plt.figure(figsize=(10, 6))
        forecast_series.plot(label='Forecast')
        plt.title(f"Forecast for {ticker} for next {forecast_period} days")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()


