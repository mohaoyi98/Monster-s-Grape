def obv(data):
    """ calculates OBV (on balance volumn)
    :param data: a list of stockstats dataframes; each dataframe contains weekly/monthly/quarterly stock data;
    :return: a list of obv values related to each time period in $data$
    """

    assert len(data) > 1

    obvs = [0]
    prev_obv = 0
    prev_close = data[0]["close"].mean()

    for period in data[1:]:
        cur_volume = period["volume"].mean()
        cur_close = period["close"].mean()

        if cur_close < prev_close:
            cur_obv = prev_obv - cur_volume
        elif cur_close > prev_close:
            cur_obv = prev_obv + cur_volume
        else:
            cur_obv = prev_obv

        obvs += [cur_obv]
        prev_obv = cur_obv
        prev_close = cur_close

    return obvs

def obv_long_score(obvs, incr_thres):
    """
    :param obvs: obv values obtained from the $obv$ function
    :param incr_thres: a threshold of the increment of adjacent obv values;
                       increment below the threshold is considered to be good, and bad otherwise
    :return: a list of scores of 0 (hold), 1 (buy), and -1 (sell)
    """
    assert incr_thres > 0

    scores = list(range(len(obvs)))
    for i in range(1, len(obvs)):
        obv_diff = obvs[i] - obvs[i-1]
        if obv_diff > 0 and obv_diff <= incr_thres:
            scores[i] = 1
        elif obv_diff == 0:
            scores[i] = 0
        else:
            scores[i] = -1

    return scores

def obv_short_score(obvs, incr_thres):
    """
    :param obvs: obv values obtained from the $obv$ function
    :param incr_thres: a threshold of the increment of adjacent obv values;
                       increment below the threshold is considered to be good, and bad otherwise
    :return: a list of scores of 0 (hold), 1 (sell), and -1 (buy)
    """
    assert incr_thres > 0

    scores = list(range(len(obvs)))
    for i in range(1, len(obvs)):
        obv_diff = obvs[i] - obvs[i-1]
        if obv_diff > 0 and obv_diff <= incr_thres:
            scores[i] = -1
        elif obv_diff == 0:
            scores[i] = 0
        else:
            scores[i] = 1

    return scores
