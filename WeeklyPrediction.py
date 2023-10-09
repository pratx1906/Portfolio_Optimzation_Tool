import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


class ShortTermPredictor:

    # Constructor for the class
    def __init__(self, tickers):
        self.tickers = tickers
        self.data = None

    def data_process(self, tickers):
        data = yf.Ticker(tickers)
        data = data.history(period="max")
        data.index = pd.to_datetime(data.index)
        data.drop(columns=["Dividends", "Stock Splits"], inplace=True)
        data["Tomorrow"] = data["Close"].shift(-1)
        data["Target"] = (data["Tomorrow"] > data["Close"].astype(int))

        # Calculate the start point for the last 40% of data
        start_row = int(0.60 * len(data))

        # Slice the data to only include the last 40%
        data = data.iloc[start_row:].copy()
        return data

    def trainPrediction(self, data):
        model = RandomForestClassifier(n_estimators=250, min_samples_split=100, random_state=1)
        train = data.iloc[:-100]
        test = data.iloc[-100:]
        predictors = ["Close", "Volume", "Open", "High", "Low"]
        model.fit(train[predictors], train["Target"])
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat()
