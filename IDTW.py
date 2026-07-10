# Indexing dynamic time warping based mediods clustering + prediction (most similar past windows)

# Pseudo code from: 
# Stock price prediction using k-medoids clustering with indexingdynamic time warping
# Written by Kei Nakagawa, Mitsuyoshi Imamura, Kenichi Yoshida
from features import *
import pandas as pd
import numpy as np
import sqlite3
from scipy.stats import zscore
from numba import njit

FEATURES_DEFAULT = ['log_return', 'log_rvol']


def get_ticker_df(df, ticker):
    return df[df['Ticker'] == ticker]



#gets z score of array W
def z_score(W):
    return zscore(W, axis=0, ddof=0)

@njit
def mat_dist(a, b):
    return np.sum(np.abs(a-b))


# Procedure DTW(x, y, w = 5)                              -> Initialize Matrix D
#   Var D[N, M]
#   D[1, 1] = 0
#   for i = 2 to N do
#       for j = 2 to M do
#           D[i, j] = ∞
#       end for
#   end for
#
#                                                        -> Calculate DTW distance
#   for i = 2 to N do
#       for j = max(1, i - w) to min(M, i + w) do
#           D[i, j] = d(x[i - 1], y[j - 1])              -> d = mat_dist
#                    + min(D[i, j - 1], D[i - 1, j], D[i - 1, j - 1])
#       end for
#   end for
#
#   return D[N, M]
# end procedure

# w is the max number of days the series can be stretched during the search
# x and y are two windows being compared (chosen stock v other stock window)
@njit
def DTW(x, y, w = 5):
    N = x.shape[0]
    M = y.shape[0]
    
    D = np.full((N + 1, M + 1), 1e18)
    D[1, 1] = 0.0
    
    for i in range(2, N + 1):
        for j in range(max(1, i - w), min(M, i + w) + 1):
            D[i, j] = np.sum(np.abs(x[i - 2] - y[j - 2])) + min(D[i, j - 1], D[i - 1, j], D[i - 1, j - 1])
    
    return D[N, M]
    # x = np.asarray(x, dtype=float)
    # y = np.asarray(y, dtype=float)

    # N = len(x)
    # M = len(y)
    
    # D = np.full((N + 1, M + 1), np.inf, dtype=float)
    # D[1, 1] = 0.0
    
    # for i in range(2, N + 1):
    #     for j in range(2, M + 1):
    #         D[i, j] = np.inf
    
    # for i in range (2, N+1):
    #     for j in range (max(1, i - w), min(M, i + w)+1):
    #         D[i, j] = mat_dist(x[i - 2], y[j - 2]) + min(D[i, j - 1], D[i - 1, j], D[i - 1, j - 1])

    # return D[N, M]


# iDTW distance - not used since already using log prices and volume (normalized)
#
# 1: procedure iDTW(x, y)                                    -> Scaling Data
# 2:   Var Ix, Iy
# 3:   Ix[1] = 1, Iy[1] = 1
# 4:   for i = 2 to N do
# 5:     Ix[i] = Ix[i - 1] * x[i] / x[i - 1]
# 6:   end for
# 7:   for j = 2 to M do
# 8:     Iy[j] = Iy[j - 1] * y[j] / y[j - 1]
# 9:   end for
# 10:  return DTW(Ix, Iy)                                   -> Apply DTW
# 11: end procedure

# starts index at 1 for normalization
def iDTW(x, y, w=5, eps=1e-12):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    N = len(x)
    M = len(y)

    Ix = np.empty(N, dtype=float)
    Iy = np.empty(M, dtype=float)
    
    Ix[0] = 1.0
    Iy[0] = 1.0
    
    for i in range (1, N):
        Ix[i] = Ix[i - 1] * (x[i] / (x[i - 1] + eps))
    for j in range (1, M):
        Iy[j] = Iy[j - 1] * (y[j] / (y[j - 1] + eps))
    return DTW(Ix, Iy, w=w)


# IDTW based k-mediods clustering
# procedure iDTW BASED K-CLUSTERING ({x1, ..., xN}, k)
#   Randomize m1, ..., mk
#   while stopping criterion has not been met do                    -> Cluster Assignment
#       for i = 1 to k do
#           ci = { xj | iDTW(xj, mi) <= iDTW(xj, mj) }
#       end for                                                     -> Update Medoids
#       for j = 1 to k do
#           mj = min_{xi ∈ cj} Σ_{l=1..N} iDTW(xi, xl)
#       end for
#   end while                                                       -> Return the medoids
#   return m1, ..., mk
# end procedure

#k mediod clustering of windows
def iDTW_BASED_K_CLUSTERING(X, k, w=5, max_iter=50, seed=67):
    # X = {x1, ..., xN} (python list)
    N = len(X)
    rng = np.random.default_rng(seed)

    # Randomize m1, ..., mk  (store medoids as indices into X)
    m = rng.choice(N, size=k, replace=False).tolist()   # m[0]=m1 ... m[k-1]=mk

    # optional distance cache to avoid recalculating iDTW constantly
    dist_cache = {}

    # while stopping criterion has not been met do
    for _ in range(max_iter):
        m_old = m.copy()

        # -----------------------------
        # Cluster Assignment
        # ci = { xj | iDTW(xj, mi) <= iDTW(xj, mj) }
        # -----------------------------
        c = [[] for _ in range(k)]   # c[0]=c1 ... c[k-1]=ck

        for xj in range(N):
            # find closest medoid
            best_i = 0
            best_d = np.inf
            for i in range(k):
                di = iDTW(X[xj], X[m[i]])
                if di < best_d:
                    best_d = di
                    best_i = i
            c[best_i].append(xj)

        # -----------------------------
        # Update Medoids
        # mj = min_{xi ∈ cj} Σ iDTW(xi, xl)
        # -----------------------------
        for j in range(k):
            cj = c[j]

            # if a cluster is empty, re-seed that medoid randomly
            if len(cj) == 0:
                m[j] = int(rng.integers(0, N))
                continue

            best_xi = cj[0]
            best_cost = np.inf

            for xi in cj:
                cost = 0.0
                for xl in cj:
                    cost += iDTW(X[xi], X[xl])
                if cost < best_cost:
                    best_cost = cost
                    best_xi = xi

            m[j] = best_xi

        # stopping criterion: medoids didn't change
        if m == m_old:
            break

    # return m1, ..., mk (and clusters too, since you’ll need them)
    return m, c


# stock = ticker of stock for prediction
# test_stocks = df of all stocks
# w = window
# features_compared = features compared from df (log_return, log_rvol, any feature column name)
# top_n gives top n closest matches
def compare_DTW(stock, test_stocks, w = 20, features_compared = FEATURES_DEFAULT, top_n = 10):
    

    test_tickers = ['GOOG', 'AAPL', 'MSFT', 'AMZN', 'META']
    small_df = test_stocks[test_stocks['Ticker'].isin(test_tickers)]

    target_stock = test_stocks[test_stocks['Ticker'] == stock].tail(w)
    
    # temp to test shortlist
    # test_stocks = test_stocks.drop(target_stock.index)
    test_stocks = small_df.drop(target_stock.index)

    target_values = target_stock[features_compared].values

    target_z = z_score(target_values)

    results = []
    window_step = w//2

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
    return results[:top_n]


        