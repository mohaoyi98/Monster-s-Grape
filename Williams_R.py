def WR(data, n):
    """Money Flow Index (MFI)
    Williams %R reflects the level of the close relative to the highest high for the look-back period.
    In contrast, the Stochastic Oscillator reflects the level of the close relative to the lowest low.
    %R corrects for the inversion by multiplying the raw value by -100. As a result, the Fast Stochastic
    Oscillator and Williams %R produce the exact same lines, but with different scaling. Williams %R
    oscillates from 0 to -100; readings from 0 to -20 are considered overbought, while readings
    from -80 to -100 are considered oversold.
    https://stockcharts.com/school/doku.php?id=chart_school%3Atechnical_indicators%3Awilliams_r
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        n(int): n period.
        pandas.Series: New feature generated.
    """
    # 0 Prepare dataframe to work
    high = data.iloc[:, 1]
    low = data.iloc[:, 2]
    close = data.iloc[:, 3]
    volume = data.iloc[:, 5]

    df = pd.DataFrame([high, low, close, volume]).T
    df.columns = ['High', 'Low', 'Close', 'Volume']
    df['Up_or_Down'] = 0
    df.loc[(df['Close'] > df['Close'].shift(1, fill_value=df['Close'].mean())), 'Up_or_Down'] = 1
    df.loc[(df['Close'] < df['Close'].shift(1, fill_value=df['Close'].mean())), 'Up_or_Down'] = 2

    # 1 highest high
    highest = sorted(high, reverse = True)[1]

    # 2 lowest low
    lowest = sorted(low)[1]

    # 3 %R
    R = (highest - close)/(highest - lowest)*(-100)
    ret = pd.Series(R, name='R_'+str(n))
    return ret
