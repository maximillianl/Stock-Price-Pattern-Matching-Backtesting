
# Stock Price Pattern Matching and Backtesting Pipeline

Using Dynamic Time Warping to compare historical stock movement windows.

## Overview

This project compares a chosen stock against historical stock data from `yfinance` to see whether similar past movement patterns can give any useful information about short-term future movement.

Instead of trying to directly predict an exact future price, the project looks for historical windows that moved similarly to the target stock. It then checks what happened after those similar windows to see if there is any kind of repeatable trend.

This is more of a research/testing pipeline than a finished trading strategy.

## Motivation

I built this because I am interested in automated trading, quant finance, and finding more reliable ways to make decisions around trades.

The main idea I wanted to test was whether stock patterns repeat enough to be useful over shorter time periods, mainly around 3–14 day windows. The goal was to compare a stock’s recent movement to similar historical windows from other stocks, then look at what happened afterward to see if those past outcomes could be used as a possible signal.

For example, if 7 of the top 10 most similar historical windows went up over the next 10 days by an average of 5%, that might make a long position more enticing. The project does not automatically make trades, but the goal was to build the kind of pipeline that could test whether an idea like that has any promise.


## Core Idea

Originally, my thought was to take two different windows, two 14-day price windows for example, and calculate the Euclidean distance between them to find the closest matches.

After looking into research papers using Dynamic Time Warping, I decided to pivot to DTW instead. DTW made more sense because stock patterns might look similar without lining up perfectly day by day. One move might happen a little faster, slower, earlier, or later, and DTW can still compare the overall shape.

At a high level, DTW can move through the two windows by stepping forward in one series, the other series, or both. That lets it stretch or squeeze the time alignment within a limit instead of forcing every day to match exactly.

My hypothesis was that, given enough historical data, there might be similar enough patterns to the ones seen today that the following period of those past windows could be used as a rough prediction for the target stock’s next move. Past that, more context could be added later, like market cap, sector, market sentiment, or whether the overall market is bullish, bearish, or sideways.

Right now, the system returns the closest matching windows and shows their DTW scores along with what happened in the days after those windows.

## Project Pipeline

The project works roughly like this:

1. Choose an index or stock universe to compare against.
2. Pull the list of stock tickers from iShares ETF holdings.
3. Clean and normalize the ticker symbols.
4. Download historical OHLCV data using `yfinance`.
5. Use adjusted prices so the pipeline is not thrown off by sudden jumps from splits, dividends, or similar events.
6. Cache the stock data locally in a SQLite database.
7. Load the cached stock data from SQLite into pandas so it can be used for feature generation and DTW comparisons.
8. Engineer features such as log returns and relative volume.
9. Save the processed feature dataframe as a parquet file.
10. Select a target stock and a window size.
11. Compare the target window against historical rolling windows using Dynamic Time Warping.
12. Rank the closest matches by DTW score.
13. Backtest what happened after those matched windows.
14. Show the closest tickers, start dates, end dates, DTW scores, and future price movement differences.

## Data Sources

The historical price data comes from `yfinance`.

The stock lists come from iShares ETF holdings CSV files. I added options for:

- S&P 100
- S&P 500
- Russell 1000
- Russell 2000
- Russell 3000

Most of the testing so far has been with the S&P 100 and S&P 500 because they are faster to run as there are less stocks.

I cached the data locally to avoid pulling from `yfinance` over and over, which would make the runtime much slower and could run into rate limits. Once the data is cached, I can test the pipeline without constantly redownloading the same stock history.

## Features Used

The main features currently used are:

- Log returns
- Log relative volume

Log returns are used instead of raw prices so the comparison focuses more on price movement instead of the actual dollar value of the stock.

Relative volume compares a stock’s volume to its own rolling average volume. I used this because raw volume would make large stocks or heavily traded stocks harder to compare fairly.

I also implemented candle-based features, but they are not currently part of the main testing setup. The goal is for the testing pipeline to eventually be flexible enough to choose which features are included in the DTW comparison.

Features I planned or want to add include:

- Candle body and wick features
- Volatility
- EMA deviation
- Donchian channel percentile
- RSI
- Market-relative returns
- Market regime labels

A big part of this project was trying not to include a bunch of features that all express basically the same thing. For example, using SMA, EMA, MACD, RSI, and other similar indicators together could overweight one type of information and make the comparison less useful.

## Similarity Method

The main comparison method is Dynamic Time Warping.

I used DTW because stock patterns can look similar without lining up perfectly day by day. A simple Euclidean distance comparison would force day 1 to match day 1, day 2 to match day 2, and so on. DTW gives the comparison more flexibility by allowing the time alignment to stretch within a set limit.

The system takes a target stock window, normalizes the selected features, and compares it against historical rolling windows from the cached stock database. Right now, the main features are log returns and log relative volume, but the comparison is set up so different feature combinations can be tested.

Each comparison gets a DTW score. A lower score means the historical window is more similar to the target window.

The goal is to find the closest past windows and then look at the following period after those windows to see how similar setups moved afterward.

## Backtesting

The backtesting step checks whether the closest historical matches had similar future movement to the target stock.

For a target stock, the system takes a past window and compares it against historical rolling windows from the cached database. After finding the closest matches by DTW score, it calculates the future percent movement after each matched window and compares that to the future percent movement after the target stock’s window.

The current output includes:

- Ticker
- Start date
- End date
- DTW score
- Future percent movement over the following days
- Difference between each matched window’s future movement and the target stock’s future movement

A lower DTW score means the historical window was more similar to the target window. The backtesting step then checks whether the most similar windows moved in a similar way afterward.

To reduce direct leakage from the target stock, the current backtest removes the target window and its following outcome window from the comparison pool before searching for similar historical windows.

This is an early backtesting step rather than a full walk-forward trading simulation.

## Current Results

The early results were mixed.

The pipeline worked, but the backtests did not prove anything close to a reliable trading strategy yet. It can find and rank similar historical windows, but the future movement after those windows is still not consistent enough to call it a strong signal.

That said, the project did prove that the main pipeline works: collecting tickers, caching data, generating features, comparing windows with DTW, and checking future movement are all connected into one workflow.

Right now, I would describe the project as a working research prototype, not a finished stock predictor.

## Challenges

Some of the main challenges were:

- Deciding what features to include
- Making sure the features were normalized correctly
- Avoiding overlapping features that measure similar things
- Finding usable and up-to-date stock lists for each index
- Cleaning ticker symbols with differences like `BRK.B` vs `BRK-B`
- Handling `yfinance` issues and avoiding too many repeated pulls
- Making the caching system work correctly
- Understanding how Dynamic Time Warping actually worked
- Keeping the DTW comparison flexible enough to test different features
- Improving runtime speed

The DTW comparisons were slow at first, so I added numba's `@njit` decorator to compile the DTW function to machine code, significantly reducing computation time for the nested loop comparisons. I also optimized the comparison process by moving the rolling comparison window in larger steps instead of checking every possible window.

## What I Learned

Through this project, I learned more about:

- Building a full data pipeline
- Using SQLite for local data caching
- Working with financial time-series data
- Why direct stock prediction is difficult
- How Dynamic Time Warping compares time-series shapes
- Why feature selection matters in trading systems
- Why normalization matters when comparing stocks
- Why backtesting has to be designed carefully

One of the biggest takeaways is that stocks do not seem to repeat themselves cleanly. Even when two windows look similar, what happens next can still be completely different.


## Future Improvements

Future improvements could include:

- Add visualizations for the target window and closest historical matches
- Graph the DTW matches and future movement differences
- Add more features while keeping feature selection flexible
- Add an interface to choose:
  - target stock
  - index/universe to compare against
  - window size
  - prediction horizon
  - features to use
- Try different window sizes
- Try different prediction horizons
- Add market regime context such as bull, bear, or sideways markets
- Add sector or industry context
- Add broader macro market features
- Add market sentiment features
- Add better evaluation metrics
- Test whether the top matches predict direction better than random
- Add a notifier if a strong setup appears, such as 7 of the top 10 similar windows moving up by at least 2%
- Try XGBoost or LSTM models later if the similarity-based approach shows enough promise

## Tech Stack

- Python
- pandas
- NumPy
- yfinance
- SQLite
- SciPy
- Numba
- Matplotlib
- Parquet

## Current Status

This project is currently a research prototype.

Most experiments are run through `main.py`, which is being used as a sandbox for testing the data pipeline, feature generation, DTW comparison, and backtesting workflow.

The repository is not fully plug-and-play yet. The database creates itself if one does not already exist, and an `.env` file is not currently required because the project no longer depends on the API key approach I originally considered.

## Disclaimer

This project is for research and educational purposes only. It is not financial advice and is not a proven profitable trading strategy.