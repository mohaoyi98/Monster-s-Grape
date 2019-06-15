# Monster's Grape
packages: numpy, sklearn, yfinance, pandas
https://pythondata.com/stockstats-python-module-various-stock-market-statistics-indicators/

Output from import data: 
* A dictionary containing public pricing data of 399 stocks chosen from the top 500 companies.<br />

Type: Dictionary<br />
Key: Abbreviated stock name<br />
Value: Stockstats dataframe<br />
Stockstats Dataframe columns: ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']<br />
Stockstats Dataframe rows: Date, from 2010-01-04 to 2014-12-31<br />


Example: 
dist["MSFT"] = a stockstats dataframe
dist.keys() = ["MSFT", "ABT", ...]
//try it on your computer ;)


Notes on the function $partition$: 

1. If $cri$ is "weekly", then it returns a list of weekly stock data; otherwise, it returns a dictionary.

2. Example of outputs (suppose variable $data$ stores the stock data imported by $YFI$):
>> partition(data["ABT"], 'monthly')
>> {"2010-1": $stock data for 2010-1$, 
    "2010-2": $stock data for 2010-2$, 
    ...
    "2010-12": $stock data for 2010-12$, ...} # the keys are in the form of year-month


>> partition(data["ABT"], 'quarterly')
>> {"2010-1": $stock data for the 1st quarter in 2010$, 
    ...
    "2010-4": $stock data for the 4th quarter in 2010$} # the keys are in the form of year-quarter


### Function specifics:

1. functions for each technicals
input:all info of one company in 30 days; input type: one element of dictionary 
output:index e.g. rsi, mfi; output type: one element of dictionary

2. helper functions (apply technicles into dataset)
input:  type:
output:  type: dataframe


### Technicals assignments:

Mohao Yi: MACD, DMI, KDJ
