# get finance data and caching (first versoin checks for chached files, second version adds data starting from date in cached file if available)
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
from datetime import date, timedelta

from test_subjects import *


# get open, high, low, close, volume data for x ticker symbol over period

#--------------------------- try to add dashes before last character if ticker not found ---------------------------
# ====================================================================================================
def get_ticker_yf(ticker_symbol, period):
    original_ticker = ticker_symbol
    data = yf.download(ticker_symbol, period=period, progress=False, auto_adjust=True) 
    if data.empty:
        ticker_symbol = try_add_dash(ticker_symbol)
        data = yf.download(ticker_symbol, period=period, progress=False, auto_adjust=True)
    if data.empty:
        return (original_ticker, pd.DataFrame())  # return empty DataFrame if ticker not found, will be skipped when caching to db   
    data.columns = data.columns.get_level_values(0)  # flatten multi-index columns
    hist_ohlcv = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    print(hist_ohlcv)
    return (ticker_symbol, hist_ohlcv)


# tries adding dash before last letter of ticker (used when ticker not found)
def try_add_dash(ticker):
    if len(ticker) > 1 and ticker[-2] != '-':
        return ticker[:-1] + '-' + ticker[-1]
    return ticker


# checks if ticker is in cached db
def is_ticker_cached(ticker_symbol):
    if ticker_symbol in list_cached_tickers():
        return True
    else:
        return False



# cache stock to db from csv
    #sudo code
    #cached list (list_cached_tickers)
        #for all tickers in csv file
            #if ticker not in cached list
                #get_tikcer_yf (get ticker data)
                #add data to db
                #add ticker to cached list
            #else skip

#=================== probably need to remove tickers no longer in the csv file from the db? ===================
def cache_stock_to_db(csv_filename):
    init_db()
    cached = set(list_cached_tickers())
    
    #open db once
    with sqlite3.connect("stocks_cache.db") as conn:
        cursor = conn.cursor()
        
        
        for ticker in pd.read_csv(csv_filename)['Ticker']:
            print("Caching ticker:", ticker)
            if not ticker in cached:
                ticker_data = get_ticker_yf(ticker, "max")
                print("Ticker data:" , ticker_data)
                print("ticker data column names", ticker_data[1].columns)
                ticker = ticker_data[0]
                print("Using ticker symbol:", ticker)
                ticker_data = ticker_data[1]
                ticker_data = ticker_data.reset_index()
                if ticker_data.empty:
                    print(f"Ticker {ticker} not found in yfinance. Skipping caching.")
                    continue
                
                
                else:
                    
                    rows = list(zip([ticker]*len(ticker_data), ticker_data['Date'].dt.strftime("%Y-%m-%d"), ticker_data['Open'], ticker_data['High'], ticker_data['Low'], ticker_data['Close'], ticker_data['Volume']))

                    # save ticker_data to db
                    cursor.executemany('''
                        INSERT OR IGNORE INTO stocks_table (ticker_symbol, date, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', rows)
            cached.add(ticker)
        
        conn.commit()
        


# initialize db
def init_db():
    with sqlite3.connect("stocks_cache.db") as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS stocks_table (
                ticker_symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker_symbol, date)
            );
        ''')
        conn.commit()







#lists cached ticker symbols in db
def list_cached_tickers():
    with sqlite3.connect("stocks_cache.db") as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT DISTINCT ticker_symbol FROM stocks_table;
        ''')
        tickers = cursor.fetchall()
        tickers = [t[0] for t in tickers]
    return tickers

    
# removes ticker from db
def remove_ticker_from_db(ticker_symbol):
    ticker_symbol = ticker_symbol.strip().upper()
    if not is_ticker_cached(ticker_symbol):
        ticker_symbol = try_add_dash(ticker_symbol)
    with sqlite3.connect("stocks_cache.db") as conn:
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM stocks_table WHERE ticker_symbol = ?;
        ''', (ticker_symbol,))
        conn.commit()




# get info for a stock from db
def get_stock_info(ticker_symbol, date_range=(None, None)):
    ticker_symbol = ticker_symbol.strip().upper()
    if is_ticker_cached(ticker_symbol) == False:
        ticker_symbol = try_add_dash(ticker_symbol)
    with sqlite3.connect("stocks_cache.db") as conn:
        cursor = conn.cursor()
        query = '''
            SELECT date, open, high, low, close, volume
            FROM stocks_table
            WHERE ticker_symbol = ?
        '''
        params = [ticker_symbol]

        if date_range is not None:
            start_date, end_date = date_range
            if start_date is not None:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date is not None:
                query += " AND date <= ?"
                params.append(end_date)

        query += " ORDER BY date ASC"

        cursor.execute(query, params)
        rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    return df


# gets data from db and puts it into a df
def db_to_df(db_file_name):

    with sqlite3.connect(db_file_name) as conn:
        cursor = conn.cursor()
        query = '''
            SELECT *
            FROM stocks_table
            ORDER BY ticker_symbol, date ASC
           
        '''
        cursor.execute(query)
        rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    return df


# graphs stock info from db, data_type = 'Open', 'High', 'Low', 'Close', 'Volume'
def graph_stock_info(ticker_symbol, date_range=(None, None), data_type='Close'):
    ticker_symbol = ticker_symbol.strip().upper()
    if is_ticker_cached(ticker_symbol) == False:
        ticker_symbol = try_add_dash(ticker_symbol)
    stock_data = get_stock_info(ticker_symbol, date_range)
    if stock_data.empty:
        print(f"No data found for ticker {ticker_symbol}.")
        return
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)
    stock_data[data_type].plot(title=f"{ticker_symbol} Stock Prices", ylabel="Price", xlabel="Date")
    plt.show()


# get info for all stocks in db minus chose one
# --- loops through and get_stock_info form db for each ticker except for ticker_symbol
def get_all_stocks_info_minus(ticker_symbol):
    for all_ticker in list_cached_tickers():
        if all_ticker != ticker_symbol:
            data = get_stock_info(all_ticker)
    return 


# window of data from db for ticker symbol
def get_window(ticker_symbol, start_date, end_date):
    pass



# future -----------------------------

# # gets range of dates already cached for ticker symbol
# def get_cached_date_range(ticker_symbol):
#     pass

# # updates new dates data into db for ticker symbol
# def update_new_dates_to_db(ticker_symbol, start_date, end_date):
#     pass




#=========================== check to see if the data in db is recent enough (withing last 7 days) ===========================
# gets latest date for ticker symbol in db
def latest_date_in_db(ticker_symbol, db_path="stocks_cache.db"):
    ticker_symbol = ticker_symbol.strip().upper()
    if not is_ticker_cached(ticker_symbol):
        ticker_symbol = try_add_dash(ticker_symbol)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT MAX(date) FROM stocks_table WHERE ticker_symbol = ?;
        ''', (ticker_symbol,))
        result = cursor.fetchone()
        if result and result[0]:
            return result[0]
        else:
            return None
        

# checks if date string is within x days from today
def recent_enough(date_str, days=7):
    if date_str is None:
        return False
    max_date = date.today() - timedelta(days=days)
    date_obj = date.fromisoformat(date_str)
    return date_obj >= max_date
# ------------------------------------