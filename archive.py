# stock_forecasting.py
# In-progress model for stock price prediction using ARIMA + linear regression
# Code includes test models, debug prints, and data exploration


import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

#test
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


def get_stock_data(ticker_symbol, period):
    
    #create ticker object
    ticker = yf.Ticker(ticker_symbol)
    
    #fetch historical market data
    historical_data = ticker.history(period = period)
    
    stock_data = pd.DataFrame(historical_data)
    # print(stock_data.tail(20))
    return stock_data

#simple moving average, exponential moving average
def moving_avgs(df):
    
    #calculates simple moving average for each 50 day window
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

    return df


#rsi <30 = oversold, >70 = overbought
def rsi(df,period):
    #difference in close price today-yesterday
    delta = df["Close"].diff()
    delta.dropna()

    change_up,change_down = delta.copy(),delta.copy()

    # leaves up pos, down neg, otherwise 0
    change_up[change_up<0] = 0
    change_down[change_down>0] = 0

    #SMA based averages
    avg_gain = change_up.rolling(period).mean()
    avg_loss = change_down.rolling(period).mean().abs()
    df["RSI"] = 100 - (100 /( 1 + (avg_gain/avg_loss)))
    
    return df
    
#moving average convergence divergence (momentum)
def MACD(df):
    #12 day ema - 26 day ema
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean()-df["Close"].ewm(span=26, adjust=False).mean()

    #signal line
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    # df.plot.line(y=["MACD","MACD_Signal"], color=['red','blue'], marker='o')
    # plt.grid(True)
    # plt.show()

    return df

#relative volatility
def bollinger_bands(df,period):
    sma = df["Close"].rolling(window=period).mean()
    std_dev = df["Close"].rolling(window=period).std()
    df["Upper_BB"] = sma + (2 * std_dev)
    df["Lower_BB"] = sma - (2 * std_dev)
    return df

#raw volatility
def ATR(df,period):
    df['High-Low'] = df['High'] - df['Low']
    df['High-Close'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-Close'] = abs(df['Low'] - df['Close'].shift(1))
    
    df['TR'] = df[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    
    df['ATR'] = df['TR'].rolling(window=period).mean()
    df.drop(columns=['High-Low', 'High-Close', 'Low-Close', 'TR'], )
    
    return df



    
#price prediction (Linear Regression) (non time dependent, basically random numbers)
def price_prediction_LR(df, period):
    # print("[][][][][][]")
    # print(df.tail(20))
    
    df = df[:len(df)-5]
    # df.dropna()
    
    
    #independent variables depending on prediction horizon
    if period <= 5:
        X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'ATR']]
    elif period <= 30:
        X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'ATR']]
    else:
        X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_200', 'EMA_200', 'RSI', 'MACD', 'MACD_Signal', 'ATR']]
        
    X.dropna(inplace = True)    
        
        
        
        
    #dependent variables
    y = df['Close'].shift(-period)
    
    
    y.dropna(inplace = True)   
    
    
    
    
    #make X an y the same length
    X = X.iloc[:len(y)]
    y = y.iloc[:len(X)]
    

    # Splitting dataset into test/train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    # Regressor model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # print(X.tail)
    # print(y.tail)
    
    
    # # Prediction result
    y_pred_test = regressor.predict(X_test)     # predicted value of y_test
    y_pred_train = regressor.predict(X_train)   # predicted value of y_train
    print(y_pred_test, "++++", y_pred_train)
    
    # # Evaluate test set performance
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    # # Evaluate training set performance
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    print(f"Test Set - MAE: {mae_test}, MSE: {mse_test}, R2: {r2_test}")
    print(f"Train Set - MAE: {mae_train}, MSE: {mse_train}, R2: {r2_train}")







def PLOT(df):
    df.plot.line(y='Close', color='green')
    plt.grid(True)
    plt.show()



#===== work in progress ======
#time series price prediction (ARIMA)
def price_prediction_ARIMA(df, period):
    # df = df[:len(df)-5]
    close = df['Close'].dropna()
    # print(df.isna().sum(), "=================", df.shape)

    # #draws the graph of the stock
    # df.plot.line(y='Close', color='green')
    # plt.grid(True)
    # plt.show()
    


    #train is all but last 30 days, test is last 30 days
    train = close[:len(df)-30].copy()
    test = close[len(df)-30:].copy()
    
    #transform to stationary: differencing
    train_diff = train.diff().dropna()
    # PLOT(test)
    

    #acf and pacf plot for testing
    # acf_diff = plot_acf(train_diff)
    # pacf_diff = plot_pacf(train_diff)
    # plt.show()

    
    #p value test
    # adf_test = adfuller(train_diff)
    # print(f'p-value: {adf_test[1]}')
    # print(f'p-value: {adf_test[1]:.2000f}')

    #ARIMA d=1 to use differenced


    #fit and determine model parameters
    model = ARIMA(train, order=(1,1,0))
    model_fit = model.fit()
    # print(model_fit.summary())

    # model = ARIMA(train, order=(0,1,1))
    # model_fit = model.fit()
    # print(model_fit.summary())

    # random walk
    # model = ARIMA(train, order=(0,1,0))
    # model_fit = model.fit()
    # print(model_fit.summary())



    # #plot residuals and density
    # residuals = model_fit.resid[1:]
    # #1 row 2 columns
    # figure, axes = plt.subplots(1,2)
    # #should look like white noise
    # residuals.plot(title='Residuals', ax=axes[0])
    # residuals.plot(title='Density', kind='kde',ax=axes[1])
    # plt.show()


    # #make sure it shows few to no spikes (residuals are white noise)
    # acf_res = plot_acf(residuals)
    # pacf_res = plot_pacf(residuals)
    # plt.show()


    history = list(train)
    forecast = []

    # # auto fit arima
    # auto_arima = pm.auto_arima(train, stepwise=False, seasonal=False, trace = True)
    # auto_arima
    # print(auto_arima)
    # plt.show()

    auto_arima = pm.auto_arima(history, stepwise=False, seasonal=False, trace = True)
    auto_arima.fit(history)
    forecast_test = auto_arima.predict(n_periods=30)

    forecast.extend(forecast_test)
    history.extend(forecast_test)





    # # recursive arima forcast
    # for i in range(len(test)):
    #     print("TEST_____________________", i)

    #     # #fit and determine model parameters
    #     # model = ARIMA(history, order=(3,1,3))
    #     # model_fit = model.fit()

    #     # auto fit arima
    #     auto_arima = pm.auto_arima(history, stepwise=False, seasonal=False, trace = True)
    #     auto_arima.fit(history)

    #     #predict 1 step ahead
    #     forecast_test = auto_arima.predict(n_periods=1)[0]

    #     #forcast 1 step ahead
    #     # forecast_test = auto_arima.forecast(steps=1)[0]
    #     forecast.append(forecast_test)

    #     print(forecast_test)

        
    #     history.append(forecast_test)
       

    print(len(close), len(train), len(forecast))




    

    #[None]*len(train) creates a list of None values for all training rows + appends values after training set ends
    result_df = pd.DataFrame({'Close': close,
                             'model': [None] * len(train) + forecast})
    


    print(train)
    print(forecast)
    result_df.plot(title='Actual vs Predicted Price')
    plt.show()





stock = get_stock_data("BMI", "2y")
moving_avgs(stock)
rsi(stock,14)
MACD(stock)
bollinger_bands(stock, 20)
ATR(stock,14)
# print(stock.shape)
# print(stock.isna().sum())
# print(stock.head(20))

price_prediction_ARIMA(stock,5)

# price_prediction_LR(stock,5)














#v2

# normalizes list of ticker symbols (replace dots with dashes, remove extra spaces, nan, caps, etc)

# may need to deal with duplicates later
def normalize_list(df):
    
    #check ticker for pd.isna(x) or '-' or ''
    #check name for excluded words excluded_words = ["WARRANT", "CVR", "ESCROW", "RIGHT", "UNIT"]
    #ticker = str(ticker).replace('.', '-').strip().upper()

    for index, row in df.iterrows():
            
        ticker = row['Ticker']
        name = row.get('Name')
        excluded_words = ["WARRANT", "CVR", "ESCROW", "RIGHT", "UNIT"]
        if pd.isna(ticker) or str(ticker).strip() == '-' or str(ticker).strip() == '':
            normalized_ticker = None

        # check if any word is in name, for all words in excluded_words, then true
        elif any(word in str(name).upper() for word in excluded_words):
            normalized_ticker = None
        else:
            normalized_ticker = str(ticker)
            normalized_ticker = normalized_ticker.replace('.', '-')
            normalized_ticker = normalized_ticker.strip()
            normalized_ticker = normalized_ticker.upper()

        df.at[index, 'Ticker'] = normalized_ticker
    df = df.dropna(subset=['Ticker']).drop_duplicates(subset=['Ticker']).reset_index(drop=True)
    return df
















import time
import pandas as pd
import yfinance as yf

def validate_tickers_yfinance(tickers, batch_size=80, period="5d", pause_s=1.0, max_retries=3):
    """
    Returns (valid, invalid, errors) where:
      - valid: list of tickers that returned any data
      - invalid: list of tickers that returned no columns / all-NaN
      - errors: dict[ticker] = error message for batches that failed repeatedly
    """
    tickers = [t for t in tickers if isinstance(t, str) and t.strip()]
    valid, invalid = set(), set()
    errors = {}

    # chunk tickers
    for i in range(0, len(tickers), batch_size):
        chunk = tickers[i:i+batch_size]

        # retry loop (helps with occasional hiccups / rate limiting)
        for attempt in range(max_retries):
            try:
                df = yf.download(
                    tickers=" ".join(chunk),
                    period=period,
                    group_by="ticker",
                    auto_adjust=False,
                    threads=False,
                    progress=False,
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    for t in chunk:
                        errors[t] = str(e)
                    df = None
                else:
                    time.sleep(pause_s * (2 ** attempt))

        if df is None:
            continue

        # If only one ticker, yfinance returns a different shape.
        if isinstance(df.columns, pd.MultiIndex):
            # MultiIndex columns: (field, ticker) or (ticker, field) depending on group_by
            # With group_by="ticker", it's usually (ticker, field)
            # We'll detect tickers by level 0.
            tick_level = df.columns.get_level_values(0)
            for t in chunk:
                if t not in set(tick_level):
                    invalid.add(t)
                    continue
                sub = df[t]  # fields like Open/High/Low/Close/Volume
                if sub.dropna(how="all").empty:
                    invalid.add(t)
                else:
                    valid.add(t)
        else:
            # Single ticker case: columns are fields directly
            t = chunk[0]
            if df.dropna(how="all").empty:
                invalid.add(t)
            else:
                valid.add(t)

        time.sleep(pause_s)

    # Anything neither valid/invalid but not error: treat as invalid
    remaining = set(tickers) - valid - invalid - set(errors.keys())
    invalid |= remaining

    return sorted(valid), sorted(invalid), errors


# Example usage:
# tickers = pd.read_csv("all_stocks.csv")["Ticker"].dropna().astype(str).tolist()
# valid, invalid, errors = validate_tickers_yfinance(tickers)
# print("valid:", len(valid), "invalid:", len(invalid), "errors:", len(errors))

# 1 Failed download:
# ['BRKB']: YFPricesMissingError('possibly delisted; no price data found  (period=5d) (Yahoo error = "No data found, symbol may be delisted")')

# 1 Failed download:
# ['HEIA']: YFPricesMissingError('possibly delisted; no price data found  (period=5d) (Yahoo error = "No data found, symbol may be delisted")')

# 1 Failed download:
# ['UHALB']: YFPricesMissingError('possibly delisted; no price data found  (period=5d) (Yahoo error = "No data found, symbol may be delisted")')

# 1 Failed download:
# ['BFB']: YFPricesMissingError('possibly delisted; no price data found  (period=5d) (Yahoo error = "No data found, symbol may be delisted")')

# 4 Failed downloads:
# ['MOGA', 'CWENA', 'BFA', 'LENB']: YFPricesMissingError('possibly delisted; no price data found  (period=5d) (Yahoo error = "No data found, symbol may be delisted")')

# 1 Failed download:
# ['GEFB']: YFPricesMissingError('possibly delisted; no price data found  (period=5d) (Yahoo error = "No data found, symbol may be delisted")')

# 1 Failed download:
# ['CRDA']: YFPricesMissingError('possibly delisted; no price data found  (period=5d)')

# 1 Failed download:
# ['THRD']: YFPricesMissingError('possibly delisted; no price data found  (period=5d) (Yahoo error = "No data found, symbol may be delisted")')
# valid: 2906 invalid: 11 errors: 0

# BRKB
# HEIA****
# UHALB
# BFB
# MOGA
# CWENA
# BFA
# LENB
# GEFB
# CRDA
# THRD****







from test_subjects import *
from data import *
from features import *
from IDTW import *
from backtest import *


import sqlite3

def print_table(db_path: str, table: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table}")
    rows = cur.fetchall()
    for r in rows:
        print(r)
    conn.close()

def search_ticker_in_db(symbol: str):
    symbol = symbol.strip().upper()
    cached = set(s.upper() for s in list_cached_tickers())
    return symbol in cached

def count_rows(db_path, table):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        return cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    

def main():
    # # snp500_stocks_api = get_snp500_stocks_api()
    # # print(snp500_stocks)
    # snp500_stocks = get_snp500_stocks()

    # # print(snp500_stocks)
    # russell1000_stocks = get_russell1000_stocks()
    # # # print(russell1000_stocks)
    # russell2000_stocks = get_russell2000_stocks()
    # # # print(russell2000_stocks)
    # russell3000_stocks = get_russell3000_stocks()
    # # # print(russell3000_stocks)
    
    snp100_stocks = get_snp100_stocks()

    # normalized_snp500 = normalize_list(snp500_stocks)
    normalized_snp100 = normalize_list(snp100_stocks)


    # print(normalized_snp500)
    # normalized_r1000 = normalize_list(russell1000_stocks)
    # normalized_r2000 = normalize_list(russell2000_stocks)
    # normalized_r3000 = normalize_list(russellu3000_stocks)
    # merged_list = merge_stock_lists(normalized_snp500, normalized_r1000, normalized_r2000, normalized_r3000)
    # list_to_csv(merged_list, "all_stocks.csv")

    # list_to_csv(normalized_r3000, "russell3000_stocks.csv")
    list_to_csv(normalized_snp100, "snp100_stocks.csv")
    # list_to_csv(normalized_snp500, "snp500_stocks.csv")

    #when going through stocks, if ticker not found try adding dash before last letter of ticker symbol

    # remove_ticker_from_db("brkb")
    # cache_stock_to_db("snp500_stocks.csv")
    cache_stock_to_db("snp100_stocks.csv")





    # print_table("stocks_cache.db", "stocks_table")
    # print(search_ticker_in_db("brkb"))
    # print(search_ticker_in_db("BRK-B"))
    # print(list_cached_tickers())
    # print(print_table("stocks_cache.db", "stocks_table"))
    # print(count_rows("stocks_cache.db", "stocks_table"))


    date_range = ("2025-01-01", "2025-12-31")
    # print(get_stock_info("AAPL"))
    # print(get_stock_info("BRK-B", date_range))
    # print(get_stock_info("brkb", date_range))


    # graph_stock_info("mgm", date_range, "Volume")




    # with sqlite3.connect("stocks_cache.db") as conn:
    #     cur = conn.cursor()
    #     print("Rows for MGM:", cur.execute(6
    #         "SELECT COUNT(*) FROM stocks_table WHERE ticker_symbol='MGM'"
    #     ).fetchone()[0])

    #     print("First 5 MGM rows:", cur.execute(
    #         "SELECT ticker_symbol, date FROM stocks_table WHERE ticker_symbol='MGM' ORDER BY date LIMIT 5"
    #     ).fetchall())

    # print(load_stock_from_db("AAPL"))

    df = db_to_df("stocks_cache.db")
    # print(df)
    # df = log_returns(df)
    # df = add_daily_candles(df)
    # df = log_rvol(df)
    
    print(df)
    create_df(df)

    # a = compare_DTW("GOOG", df, w = 20, features_compared = ["log_return", "log_rvol"])
    a = backtest_compare_DTW("META", df, w = 30, features_compared = ["log_return"])
    print(a)
    a = create_backtest_df(a)
    a = add_future_price_movement(a, df)
    backtest = add_future_price_movement_comparison(a)
    print(backtest)
    

if __name__ == "__main__":
    main()