#modified from the program in "https://www.pythonforfinance.net/2017/07/31/bollinger-band-trading-strategy-backtest-in-python/"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#define strategy in bollinger band
def bollinger_strat(df,window,std,dateindex):
    #calculate the mean and standard deviation
    rolling_mean = df['Close'].rolling(window).mean()
    rolling_std = df['Close'].rolling(window).std()

    #calculate the upper and lower band of Bolling band
    df['Bollinger High'] = rolling_mean + (rolling_std * std)
    df['Bollinger Low'] = rolling_mean - (rolling_std * std)

    #plot Bolling band with the following instruction
    #df[['Close','Bollinger High','Bollinger Low']].plot()

    #create new columns for specific operation: buy or sell
    #df['Short'] = None
    #df['Long'] = None
    df['Position'] = None

    #buy(1) if the close price is higher than the upper band; sell(-1) if the close price is lower than the lower band
    for row in range(len(df)):

        if (df['Close'].iloc[row] > df['Bollinger High'].iloc[row]) and (df['Close'].iloc[row-1] < df['Bollinger High'].iloc[row-1]):
            df['Position'].iloc[row] = -1

        if (df['Close'].iloc[row] < df['Bollinger Low'].iloc[row]) and (df['Close'].iloc[row-1] > df['Bollinger Low'].iloc[row-1]):
            df['Position'].iloc[row] = 1

    #Forward fill our position column to replace the None with the correct long/short positions to represent the "holding" of our position
    #forward through time
    df['Position'].fillna(method='ffill',inplace=True)

    #generate strategy return
    df['Market Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Strategy Return'] = df['Market Return'] * df['Position']

    #plot the strategy retrun by the following instruction
    #df['Strategy Return'].cumsum().plot()

    return df['Strategy Return'][dateindex-1]

def main():
    #generate strategy return for each company
    strategy_return={}
    for k in dist:
        df = pd.DataFrame(dist[k], columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'])
        #using the bollinger band strategy function (consider 30 days)
        srategy_return[k]=bollinger_strat(df,30,2,90)

    return strategy_return
