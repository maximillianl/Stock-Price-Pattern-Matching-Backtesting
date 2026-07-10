from test_subjects import *
from data import *
from features import *
from IDTW import *
from backtest import *


import sqlite3

def main():
    # load and cache stock data
    snp100_stocks = get_snp100_stocks()
    normalized = normalize_list(snp100_stocks)
    list_to_csv(normalized, "snp100_stocks.csv")
    cache_stock_to_db("snp100_stocks.csv")

    # build feature dataframe
    df = db_to_df("stocks_cache.db")
    create_df(df)

    # run backtest comparison
    results = backtest_compare_DTW("META", df, w=20, features_compared=["log_return", "log_rvol"])
    results_df = create_backtest_df(results)
    results_df = add_future_price_movement(results_df, df)
    results_df = add_future_price_movement_comparison(results_df)
    print(results_df)

if __name__ == "__main__":
    main()
