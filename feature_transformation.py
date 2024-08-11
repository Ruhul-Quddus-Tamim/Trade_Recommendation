import pandas as pd
import numpy as np

class TechnicalIndicators:
    def __init__(self, data):
        self.data = data

    def calculate_rsi(self, timeperiod=14):
        delta = self.data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=timeperiod).mean()
        avg_loss = loss.rolling(window=timeperiod).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        self.data['RSI'] = rsi

    def calculate_macd(self, fastperiod=12, slowperiod=26, signalperiod=9):
        ema_fast = self.data['Close'].ewm(span=fastperiod, min_periods=1).mean()
        ema_slow = self.data['Close'].ewm(span=slowperiod, min_periods=1).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signalperiod, min_periods=1).mean()
        hist = macd - signal
        self.data['MACD'] = macd
        self.data['MACD_signal'] = signal
        self.data['MACD_hist'] = hist

    def calculate_stoch(self, fastk_period=14, slowk_period=3, slowd_period=3):
        low_min = self.data['Low'].rolling(window=fastk_period).min()
        high_max = self.data['High'].rolling(window=fastk_period).max()
        stoch_k = 100 * (self.data['Close'] - low_min) / (high_max - low_min)
        stoch_d = stoch_k.rolling(window=slowd_period).mean()
        self.data['Stoch_k'] = stoch_k
        self.data['Stoch_d'] = stoch_d

    def calculate_obv(self):
        direction = np.sign(self.data['Close'].diff())
        self.data['OBV'] = (direction * self.data['Volume']).cumsum()

    def calculate_bbands(self, timeperiod=20):
        sma = self.data['Close'].rolling(window=timeperiod).mean()
        std = self.data['Close'].rolling(window=timeperiod).std()
        self.data['Upper_BB'] = sma + (2 * std)
        self.data['Middle_BB'] = sma
        self.data['Lower_BB'] = sma - (2 * std)

    def calculate_atr(self, timeperiod=14):
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=timeperiod).mean()
        return atr

    def add_volatility_indicators(self):
        self.calculate_bbands(timeperiod=20)
        self.data['ATR_1'] = self.calculate_atr(timeperiod=1)
        self.data['ATR_2'] = self.calculate_atr(timeperiod=2)
        self.data['ATR_5'] = self.calculate_atr(timeperiod=5)
        self.data['ATR_10'] = self.calculate_atr(timeperiod=10)
        self.data['ATR_20'] = self.calculate_atr(timeperiod=20)

    def calculate_adx(self, timeperiod=14):
        high_diff = self.data['High'].diff()
        low_diff = self.data['Low'].diff()
        plus_dm = pd.Series(np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0))
        minus_dm = pd.Series(np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0))
        tr = self.calculate_atr(timeperiod=timeperiod)
        plus_di = 100 * (plus_dm.rolling(window=timeperiod).sum() / tr)
        minus_di = 100 * (minus_dm.rolling(window=timeperiod).sum() / tr)
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(window=timeperiod).mean()
        self.data['ADX'] = adx
        self.data['+DI'] = plus_di
        self.data['-DI'] = minus_di

    def calculate_cci(self, timeperiod=14):
        tp = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        sma = tp.rolling(window=timeperiod).mean()
        mad = (tp - sma).rolling(window=timeperiod).mean()
        self.data['CCI'] = (tp - sma) / (0.015 * mad)

    def add_other_indicators(self):
        self.data['DLR'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['TWAP'] = self.data['Close'].expanding().mean()
        self.data['VWAP'] = (self.data['Volume'] * (self.data['High'] + self.data['Low']) / 2).cumsum() / self.data['Volume'].cumsum()

    def add_all_indicators(self):
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_stoch()
        self.calculate_obv()
        self.add_volatility_indicators()
        self.calculate_adx()
        self.calculate_cci()
        self.add_other_indicators()
        return self.data

def preprocess_and_transform(file_path):
    data = pd.read_csv(file_path)

    # Preprocessing to create necessary columns
    data['price'] = data['price'] / 1e9
    data['bid_px_00'] = data['bid_px_00'] / 1e9
    data['ask_px_00'] = data['ask_px_00'] / 1e9

    data['Close'] = data['price']
    data['Volume'] = data['size']
    data['High'] = data[['bid_px_00', 'ask_px_00']].max(axis=1)
    data['Low'] = data[['bid_px_00', 'ask_px_00']].min(axis=1)
    data['Open'] = data['Close'].shift(1).fillna(data['Close'])

    # Create and apply technical indicators
    ti = TechnicalIndicators(data)
    df_with_indicators = ti.add_all_indicators()

    # Dropping initial rows to avoid NaN values
    market_features_df = df_with_indicators.dropna().reset_index(drop=True)

    return market_features_df
