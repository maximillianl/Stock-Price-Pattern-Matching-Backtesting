# get stocks to be given to data.py to fetch and cache data of listed stocks
# adds stock names and ticker symbols into test_subjects.csv

# future look into using yfinance to pull etf holdings directly instead of ishares csvs??

import pandas as pd
import numpy as np
import requests
import json
import os
from dotenv import load_dotenv

#============================= only used for api, api not used currently
# load_dotenv() # load environment variables from .env file

# API_NINJAS_KEY = os.getenv("API_NINJAS_KEY")
# if not API_NINJAS_KEY:
#     raise RuntimeError("Missing API_NINJAS_KEY. Put it in a .env file.")
#=============================


# gets list of s&p 500 stocks (~top 500 biggest)
def get_snp500_stocks_api():

    #accesses api for s&p 500 stock list using api key (get request)
    api_url = "https://api.api-ninjas.com/v1/sp500"
    response = requests.get(api_url, headers={'X-Api-Key': API_NINJAS_KEY})

    # puts data into dataframe
    if response.status_code == requests.codes.ok:
        data = response.json()
        df = pd.DataFrame(data)
        return df
    else:
        print("Error:", response.status_code, response.text)
        return None


# reads ishares etf holdings csv from url, skips header/footer rows
def read_ishares_csv(url):
    df = pd.read_csv(url, skiprows=9, skipfooter=10, engine='python')
    return df


# gets index lists from ishares etf holdings csvs

# OEF - s%p 100
def get_snp100_stocks():
    df = read_ishares_csv("https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=us-ishares&locale=en_US&portfolioId=239723&userType=individual&asOfDate=20260515&component=holdings")
    return df


# IVV - s&p 500
def get_snp500_stocks():
    df = read_ishares_csv("https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=us-ishares&locale=en_US&portfolioId=239728&userType=individual&asOfDate=20260515&component=holdings")
    return df

# IWB - russell 1000 (~top 1000 biggest)
def get_russell1000_stocks():
    df = read_ishares_csv("https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=us-ishares&locale=en_US&portfolioId=239707&userType=individual&asOfDate=20260515&component=holdings")
    return df


# IWM - russell 2000 (~smallest 2000)
def get_russell2000_stocks():
    df = read_ishares_csv("https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=us-ishares&locale=en_US&portfolioId=239710&userType=individual&asOfDate=20260515&component=holdings")
    return df

# IWV - russell 3000 (~top 3000 stocks)
def get_russell3000_stocks():
    df = read_ishares_csv("https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=us-ishares&locale=en_US&portfolioId=239714&userType=individual&asOfDate=20260515&component=holdings")
    return df


# normalizes list of ticker symbols (replace dots with dashes, remove extra spaces, nan, caps, etc)

# may need to deal with duplicates later
def normalize_list(df):
    
    #check ticker for pd.isna(x) or '-' or ''
    #check name for excluded words excluded_words = ["WARRANT", "CVR", "ESCROW", "RIGHT", "UNIT"]
    #ticker = str(ticker).replace('.', '-').strip().upper()

    ticker = df['Ticker']

    ticker_na = (ticker.isna())
    ticker = ticker.str.upper().str.strip().str.replace('.', '-', regex=False)
    ticker_mask = ticker_na | (ticker == '-') | (ticker == '')

    df['Ticker'] = ticker
    df = df[~ticker_mask].reset_index(drop=True)


    # removes stocks with excluded words in name
    excluded_words = "WARRANT|CVR|ESCROW|RIGHT|UNIT"
    name_mask = df['Name'].fillna("").str.contains(excluded_words, case = False)
    df = df[~name_mask].reset_index(drop=True)

    # removes stocks with no market exchange
    exchange_mask = df['Exchange'].fillna("").str.contains("NO MARKET", case = False)
    df = df[~exchange_mask].reset_index(drop=True)
    
    # removes cash and derivatives (from etf holdings)
    sector_mask = df['Sector'].fillna("").str.contains("Cash and/or Derivatives", case = False)
    df = df[~sector_mask].reset_index(drop=True)

    # removes duplicates
    df = df.drop_duplicates(subset=['Ticker']).reset_index(drop=True)
    return df



# merges two stock lists, removes duplicates
def merge_stock_lists(*lists):
    merged_df = pd.concat(lists).drop_duplicates(subset=['Ticker']).reset_index(drop=True)
    return merged_df


# saves dataframe to csv
def list_to_csv(df, filename):
    df.to_csv(filename, index=False)

    