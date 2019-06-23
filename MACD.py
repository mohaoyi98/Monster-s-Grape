#modified from the program in "https://www.linkedin.com/pulse/python-tutorial-macd-moving-average-andrew-hamlet"
import pandas as pd

def MACD_strat(df, dateindex):
    #calculate the 30-day and 15-day Exponential Moving Average (EMA)
    df['30 ema'] = pd.ewma(df['close'], span=30)
    df['15 ema'] = pd.ewma(df['close'], span=15)

    #calculate MACD by MACD = 15-day EMA - 30-day EMA
    #orignal verion of MACD = 12-day EMA - 26-day EMA
    #signal series is the EMA of the MACD series described above
    #divergence series is the difference between the MACD series and average series
    df['MACD'] = (df['15 ema'] - df['30 ema'])
    df['signal'] = pd.ewma(df['MACD'], span=9)
    df['div'] = (df ['MACD'] - df['signal'])
    
    return df['div']


def main():
    #generate strategy return for each company
    value_return={}
    for k in dist:
        df = pd.DataFrame(dist[k])
        #return divergence value of MACD(15,30,9)
        value_return[k]=MACD_strat(df)

    return value_return
