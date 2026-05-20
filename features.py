#indicators and window building
from data import *
import pandas as pd
import numpy as np
import sqlite3




#saves parquet of df with log returns, upper wick, lower wick, volatility, rvol, ema deviation, donchian channel percentile and rsi from ohlcv
def create_df(df):
    df = log_returns(df)
    # df = add_daily_candles(df)
    # df = volatility(df)
    df = log_rvol(df)
    # df = ema_deviation(df, window=20)
    # df = donchian_channel_percentile(df, window=20)
    # df = rsi(df)
    df.to_parquet("snp500_features.parquet", index=False)
    
    



# converts ohlcv to log returns of stock prices (log of daily returns)

# get stock is a dataframe with ohlcv data from db
# use data from the current day and following day 'Close' column

#stitch log_return for each row of df
# daily return 
def log_returns(df):
    df['log_return'] = df.groupby('Ticker')['Close'].transform(lambda x: np.log(x / x.shift(1)))
    return df



# converts ohlcv to upper and lower wick (log distance from max of open/close to high)
# body tells intraday return
def add_daily_candles(df):
    df['log_body'] = np.log(df['Close'] / df['Open'])
    
    top_of_body = df[['Open', 'Close']].max(axis=1)
    bottom_of_body = df[['Open', 'Close']].min(axis=1)
    
    df['upper_wick'] = np.log(df['High'] / top_of_body)
    df['lower_wick'] = np.log(bottom_of_body / df['Low'])
    
    return df






# converts ohlcv to atr% (Average True Range Percentile) + percentile rank (volatility)
def volatility(df, window=20):
    pass
    




# converts ohlcv to log (volume / 20-day avg volume). (relative volume rvol)
def log_rvol(df, window=20):
    vol = df['Volume'].replace(0, np.nan)
    sma_vol = vol.groupby(df['Ticker']).transform(lambda x: x.rolling(window, min_periods=window).mean())
    df['log_rvol'] = np.log(vol/sma_vol)
    return df



# converts ohlcv to % deviation from EMA (trend strength)
def ema_deviation(df, window=20):
    pass



# converts ohlcv to donchian channel percentile ()
def donchian_channel_percentile(df, window=20):
    pass



# converts ohlcv to rsi (mean reversion momentum)
def rsi(df, window=20):
    pass


# relative return vs market (s&p)