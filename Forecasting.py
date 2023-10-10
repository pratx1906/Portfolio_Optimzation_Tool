from datetime import datetime, timedelta
from pmdarima import auto_arima
import pandas as pd
import yfinance as yf


class Forecasting:
    def __init__(self, ticker):
        self.tickers = ticker

    def _get_data(self, ticker, start_date, end_date):
        data = yf.download(ticker, start=end_date, end=start_date)
        return data['Close']

    def _calculate_returns(self, prices):
        # Calculate daily returns
        returns = prices.pct_change().dropna()
        return returns

    def forecast(self, ticker, start_date, forecast_period=180):
        # Adjust end date for 5 years before the start_date
        end_date = start_date - timedelta(days=5 * 365)

        # Fetch the historical data
        closing_prices = self._get_data(ticker, start_date, end_date)

        # Convert to daily returns
        daily_returns = self._calculate_returns(closing_prices)

        # Using auto_arima to find the best ARIMA order
        stepwise_fit = auto_arima(daily_returns, trace=True, suppress_warnings=True)

        # Forecasting the returns
        forecasted_returns = stepwise_fit.predict(n_periods=forecast_period)

        # Convert forecasted returns to prices
        last_price = closing_prices.iloc[-1]
        forecasted_prices = [last_price]
        for ret in forecasted_returns:
            forecasted_prices.append(forecasted_prices[-1] * (1 + ret))

        # Creating a date range for the forecasted data
        forecast_index = pd.date_range(start_date, periods=forecast_period)

        forecast_series = pd.Series(forecasted_prices[1:], index=forecast_index)
        return closing_prices, forecast_series


