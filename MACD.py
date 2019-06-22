#modified from the program in "https://www.linkedin.com/pulse/python-tutorial-macd-moving-average-andrew-hamlet"
import pandas as pd

def MACD_strat(df, dateindex):
    #calculate the 26-day and 12-day Exponential Moving Average (EMA)
    df['26 ema'] = pd.ewma(df['close'], span=26)
    df['12 ema'] = pd.ewma(df['close'], span=12)

    #calculate MACD by MACD = 12-day EMA - 26-day EMA
    #signal series is the EMA of the MACD series described above
    #divergence series is the difference between the MACD series and average series
    df['MACD'] = (df['12 ema'] - df['26 ema'])
    df['signal'] = pd.ewma(df['MACD'], span=9)
    df['div'] = (df ['MACD'] - df['signal'])

    #create new columns for specific operation: buy or sell or stop
    df['Position'] = None

    #a three-rule system governing entries and exits from "https://www.daytrading.com/macd"
    #1. Enter trades upon a signal line crossover. MACD series above the signal line is a bullish signal. MACD series below the signal line is a bearish signal.
    #2. Enter trades only in the direction of the trade, as dictated by a 50-period SMA.
    #3. Exit when there is another signal line crossover, or the slope of the 50-period SMA changes.

    #50-period simple moving average (SMA)
    rolling_mean = df['close'].rolling(50).mean()

    #return 1 if buying; return -1 if selling
    for row in range(len(df)):

        if (df['div'].iloc[row] > 0) and (df['div'].iloc[row-1] < 0) and (rolling_mean['close'].iloc[row] > rolling_mean['close'].iloc[row-1]):
            df['Position'].iloc[row] = 1

        if ((df['div'].iloc[row] < 0) and (df['div'].iloc[row-1] > 0)) or (rolling_mean['close'].iloc[row] <= rolling_mean['close'].iloc[row-1]):
            df['Position'].iloc[row] = -1

    #Forward fill our position column to replace the None with the correct long/short positions to represent the "holding" of our position
    df['Position'].fillna(method='ffill',inplace=True)

    return df['Position'][dateindex-1]


#return what operation is needed on this day.
#return 1 if buying; return -1 if selling
def main():
    #generate strategy return for each company
    strategy_return={}
    for k in dist:
        df = pd.DataFrame(dist[k])
        #using the bollinger band strategy function (consider 30 days) and giving the strategy return on the 90-th day
        srategy_return[k]=MACD_strat(df, 90)

    return strategy_return
