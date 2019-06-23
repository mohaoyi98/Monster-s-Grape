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
        'output': the voltage
    """
    #calculate the mean and standard deviation
    rolling_mean = df['close'].rolling(window).mean()
    rolling_std = df['close'].rolling(window).std()

    #calculate the upper and lower band of Bolling band
    df['Bollinger High'] = rolling_mean + (rolling_std * std)
    df['Bollinger Low'] = rolling_mean - (rolling_std * std)

    #create new columns for output
    df['output'] = None
    
    for row in range(len(df)):
        df['output'].iloc[row] = 100*((df['close'].iloc[row] - df['Bollinger High'].iloc[row])/df['Bollinger High'].iloc[row])

    return df['output']




def main():
    #generate voltage return for each company
    strategy_return={}
    for k in dist:
        df = pd.DataFrame(dist[k])
        #return the valtage value (consider 30 days)
        srategy_return[k]=bollinger_strat(df,30,2)

    return strategy_return
