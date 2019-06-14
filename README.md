# Monster's Grape
packages: numpy, sklearn, yfinance, pandas
https://pythondata.com/stockstats-python-module-various-stock-market-statistics-indicators/

Output from import data: dictionary, ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

example: 
dist["MSFT"] = [ [1,2,3], [2,3,4], [4,5,6],[5,6,7],[100,200,300]  ]
dist.keys() = ["MSFT", "ABT", ...]

1. functions for each technicals
input:all info of one company in 30 days; input type: one element of dictionary 
output:index e.g. rsi, mfi; output type: one element of dictionary


2. helper functions (apply technicles into dataset)
input:  type:
output:  type: dataframe




Mohao Yi: MACD, DMI, KDJ