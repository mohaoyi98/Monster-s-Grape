# Monster's Grape
packages: numpy, sklearn, yfinance, pandas
https://pythondata.com/stockstats-python-module-various-stock-market-statistics-indicators/

Output from import data: 
* A dictionary containing public pricing data of 399 stocks chosen from the top 500 companies._
Type: Dictionary_
Key: Abbreviated stock name_
Value: Stockstats dataframe_
Stockstats Dataframe columns: ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']_
Stockstats Dataframe rows: Date, from 2010-01-04 to 2014-12-31_


example: 
dist["MSFT"] = a stockstats dataframe
dist.keys() = ["MSFT", "ABT", ...]
//try it on your computer ;)


### Function specifics:

1. functions for each technicals
input:all info of one company in 30 days; input type: one element of dictionary 
output:index e.g. rsi, mfi; output type: one element of dictionary

2. helper functions (apply technicles into dataset)
input:  type:
output:  type: dataframe


### Technicals assignments:

Mohao Yi: MACD, DMI, KDJ
