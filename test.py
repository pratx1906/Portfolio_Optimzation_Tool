import unittest
import pandas as pd
from datetime import datetime, timedelta

from Forecasting import Forecasting


class TestForecasting(unittest.TestCase):

    def setUp(self):
        self.ticker = "AAPL"  # Singular since it's just one ticker
        self.predictor = Forecasting(self.ticker)
        self.start_date = datetime.today()

    def test_get_data(self):
        data = self.predictor._get_data(self.ticker, self.start_date, self.start_date - timedelta(days=365))
        self.assertIsInstance(data, pd.Series)
        self.assertGreater(len(data), 0)

    def test_calculate_returns(self):
        data = pd.Series([100, 101, 102, 99, 98])
        returns = self.predictor._calculate_returns(data)
        expected_returns = [None, 0.01, 0.00990099009900991, -0.02941176470588236,
                            -0.010101010101010166]  # Added 'None' for the first value as returns have one less entry
        self.assertTrue(all([a == b for a, b in zip(returns, expected_returns)]),
                        "Returns do not match expected values")

    def test_forecast(self):
        historical_data, forecast_series = self.predictor.forecast(self.ticker, self.start_date)
        self.assertIsInstance(historical_data, pd.Series)
        self.assertIsInstance(forecast_series, pd.Series)
        self.assertGreater(len(forecast_series), 0)


if __name__ == "__main__":
    unittest.main()
