#modified from the program in "https://www.pythonforfinance.net/2017/07/31/bollinger-band-trading-strategy-backtest-in-python/"
import pandas as pd
import numpy as np

#define strategy in bollinger band
def bollinger_strat(df,window,std):
    """
    A Bollinger Band is defined by a set of lines plotted two standard deviations (positively and negatively)
    away from a simple moving average (SMA) of the security's price, but can be adjusted to user preferences 
    (here only using SMA).
    When the actual stock price is higher than the uppper bound, it's a signal to buy; when the actual stock 
    price is lower than the lower bound, it's a signal to sell.
    https://www.investopedia.com/terms/b/bollingerbands.asp
    Args:
        'close': the close price on the day
        'Bollinger High': the higher bound of Bollinger band
        'Bollinger Low': the lower bound of Bollinger band
        'Position': the operation needed on the day (1 if buying; -1 if selling)
    """
    #calculate the mean and standard deviation
    rolling_mean = df['close'].rolling(window).mean()
    rolling_std = df['close'].rolling(window).std()

    #calculate the upper and lower band of Bolling band
    df['Bollinger High'] = rolling_mean + (rolling_std * std)
    df['Bollinger Low'] = rolling_mean - (rolling_std * std)

    #create new columns for specific operation: buy or sell
    df['Position'] = None

    #buy(1) if the close price is higher than the upper band; sell(-1) if the close price is lower than the lower band
    for row in range(len(df)):

        if (df['close'].iloc[row] > df['Bollinger High'].iloc[row]) and (df['close'].iloc[row-1] < df['Bollinger High'].iloc[row-1]):
            df['Position'].iloc[row] = -1

        if (df['close'].iloc[row] < df['Bollinger Low'].iloc[row]) and (df['close'].iloc[row-1] > df['Bollinger Low'].iloc[row-1]):
            df['Position'].iloc[row] = 1

    #Forward fill our position column to replace the None with the correct long/short positions to represent the "holding" of our position
    df['Position'].fillna(method='ffill',inplace=True)

    return df['Position']




#take the date (i-th day, i is the input of this function) as input, return what operation is needed on this day.
#return 1 if buying; return -1 if selling
def main():
    #generate strategy return for each company
    strategy_return={}
    for k in dist:
        df = pd.DataFrame(dist[k])
        #using the bollinger band strategy function (consider 30 days)
        srategy_return[k]=bollinger_strat(df,30,2)

    return strategy_return
