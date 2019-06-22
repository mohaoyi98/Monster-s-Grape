def mfi(data, n):
    """Money Flow Index (MFI)
    Uses both price and volume to measure buying and selling pressure. It is
    positive when the typical price rises (buying pressure) and negative when
    the typical price declines (selling pressure). A ratio of positive and
    negative money flow is then plugged into an RSI formula to create an
    oscillator that moves between zero and one hundred.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:money_flow_index_mfi
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        n(int): n period.
        pandas.Series: New feature generated.
    """
    # 0 Prepare dataframe to work
    high = data.iloc[:,1]
    low = data.iloc[:,2]
    close = data.iloc[:,3]
    volume = data.iloc[:,5]


    df = pd.DataFrame([high, low, close, volume]).T
    df.columns = ['High', 'Low', 'Close', 'Volume']
    df['Up_or_Down'] = 0
    df.loc[(df['Close'] > df['Close'].shift(1, fill_value=df['Close'].mean())), 'Up_or_Down'] = 1
    df.loc[(df['Close'] < df['Close'].shift(1, fill_value=df['Close'].mean())), 'Up_or_Down'] = 2

    # 1 typical price
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0

    # 2 money flow
    mf = tp * df['Volume']

    # 3 positive and negative money flow with n periods
    df['1p_Positive_Money_Flow'] = 0.0
    df.loc[df['Up_or_Down'] == 1, '1p_Positive_Money_Flow'] = mf
    n_positive_mf = df['1p_Positive_Money_Flow'].rolling(n).sum()

    df['1p_Negative_Money_Flow'] = 0.0
    df.loc[df['Up_or_Down'] == 2, '1p_Negative_Money_Flow'] = mf
    n_negative_mf = df['1p_Negative_Money_Flow'].rolling(n).sum()

    # 4 money flow index
    mr = n_positive_mf / n_negative_mf
    mr = (100 - (100 / (1 + mr)))
    ret = pd.Series(mr, name='mfi_'+str(n))
    return ret
