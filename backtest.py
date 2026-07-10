# backtesting
from IDTW import *
import numpy as np
import pandas as pd
from numba import njit
from features import *


def backtest_compare_DTW(stock, test_stocks, w = 20, features_compared = FEATURES_DEFAULT, top_n = 15):
    
    # TODO: remove hordcoded test tickers (used for testing)
    test_tickers = ['GOOG', 'AAPL', 'MSFT', 'AMZN', 'META']
    small_df = test_stocks[test_stocks['Ticker'].isin(test_tickers)]

    stock_df = test_stocks[test_stocks['Ticker'] == stock]
    target_stock = stock_df.iloc[-w*2:-w]
    
    
    print(len(target_stock))
    print(target_stock['Date'].iloc[0])
    print(target_stock['Date'].iloc[-1])
    print(target_stock)
    
    
    
    actual_outcome = stock_df.iloc[-w:]
    # temp to test shortlist
    # test_stocks = small_df.drop(target_stock.index)
    test_stocks = test_stocks.drop(target_stock.index).drop(actual_outcome.index)

    target_values = target_stock[features_compared].values

    target_z = z_score(target_values)

    results = []
    window_step = w//2
    
    results.append({
    'Ticker': stock,
    'start_date': target_stock['Date'].iloc[0],
    'end_date': target_stock['Date'].iloc[-1],
    'DTW_score': 0.0
    })

    for ticker in test_stocks['Ticker'].unique():
        # df of just current ticker
        ticker_df = test_stocks[test_stocks['Ticker'] == ticker]
        print(ticker, "======================================================================")
        
        # slides windows through ticker at each window_step
        for start in range(0, len(ticker_df) - w, window_step):
            comparitor = ticker_df.iloc[start:start + w]
            comparitor_values = comparitor[features_compared].values
            if np.isnan(comparitor_values).any():
                continue
            comparitor_z = z_score(comparitor_values)            
            DTW_score = DTW(target_z, comparitor_z, w // 4)
            results.append({'Ticker': ticker, 'start_date': ticker_df['Date'].iloc[start], 'end_date': ticker_df['Date'].iloc[start + w - 1], 'DTW_score': DTW_score})
            
            # debugging print statement
            # print(DTW_score)


    results.sort(key=lambda x: x['DTW_score'])
    
    
    
    return results[:top_n+1]

# add top n results into db
def create_backtest_df(results):
    top_n = pd.DataFrame(results)
    return top_n
    

# add price change for day 1 2 3 4 5 6 7 following end
def add_future_price_movement(results_df, full_df, days=[1, 2, 3, 4, 5, 6, 7]):
    
    for d in days:
        results_df[f'day_{d}_pct'] = None

    for i, row in results_df.iterrows():
        ticker_df = full_df[full_df['Ticker'] == row['Ticker']].reset_index(drop=True)
        # find where end_date is
        match = ticker_df[ticker_df['Date'] == row['end_date']]
        if match.empty:
            continue
        end_idx = match.index[0]
        end_close = ticker_df.iloc[end_idx]['Close']

        for d in days:
            if end_idx + d < len(ticker_df):
                future_close = ticker_df.iloc[end_idx + d]['Close']
                pct_change = (future_close - end_close) / end_close * 100
                results_df.at[i, f'day_{d}_pct'] = round(pct_change, 2)

    return results_df
    
    
    
# add price change difference from actual for day 1 2 3 4 5 6 7 following end
def add_future_price_movement_comparison(results_df, days=[1, 2, 3, 4, 5, 6, 7]):
    target_row = results_df[results_df['DTW_score'] == 0].iloc[0]

    for d in days:
        target_move = target_row[f'day_{d}_pct']
        results_df[f'day_{d}_diff'] = results_df[f'day_{d}_pct'] - target_move

    return results_df