import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


class ShortTermPredictor:

    def __int__(self, tickers):
        self.tickers = tickers
