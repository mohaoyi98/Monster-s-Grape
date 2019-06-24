import numpy as np
import pandas as pd
import yfinance as yf
import stockstats as SS
import datetime

# Symbols of companies in S&P 500
symbols = ['ABT', 'ABBV', 'ACN', 'ACE', 'ADBE', 'ADT', 'AAP', 'AES', 'AET', 'AFL', 'AMG', 'A', 'GAS', 'APD', 'ARG', 'AKAM',
               'AA', 'AGN', 'ALXN', 'ALLE', 'ADS', 'ALL', 'ALTR', 'MO', 'AMZN', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AMP',
               'ABC', 'AME', 'AMGN', 'APH', 'APC', 'ADI', 'AON', 'APA', 'AIV', 'AMAT', 'ADM', 'AIZ', 'T', 'ADSK', 'ADP', 'AN',
               'AZO', 'AVGO', 'AVB', 'AVY', 'BHI', 'BLL', 'BAC', 'BK', 'BCR', 'BXLT', 'BAX', 'BBT', 'BDX', 'BBBY', 'BRK-B',
               'BBY', 'BLX', 'HRB', 'BA', 'BWA', 'BXP', 'BSK', 'BMY', 'BRCM', 'BF-B', 'CHRW', 'CA', 'CVC', 'COG', 'CAM', 'CPB',
               'COF', 'CAH', 'HSIC', 'KMX', 'CCL', 'CAT', 'CBG', 'CBS', 'CELG', 'CNP', 'CTL', 'CERN', 'CF', 'SCHW', 'CHK', 'CVX',
               'CMG', 'CB', 'CI', 'XEC', 'CINF', 'CTAS', 'CSCO', 'C', 'CTXS', 'CLX', 'CME', 'CMS', 'COH', 'KO', 'CCE', 'CTSH', 'CL',
               'CMCSA', 'CMA', 'CSC', 'CAG', 'COP', 'CNX', 'ED', 'STZ', 'GLW', 'COST', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI',
               'DVA', 'DE', 'DLPH', 'DAL', 'XRAY', 'DVN', 'DO', 'DTV', 'DFS', 'DISCA', 'DISCK', 'DG', 'DLTR', 'D', 'DOV', 'DOW', 'DPS',
               'DTE', 'DD', 'DUK', 'DNB', 'ETFC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMC', 'EMR', 'ENDP', 'ESV', 'ETR',
               'EOG', 'EQT', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ES', 'EXC', 'EXPE', 'EXPD', 'ESRX', 'XOM', 'FFIV', 'FB', 'FAST', 'FDX',
               'FIS', 'FITB', 'FSLR', 'FE', 'FSIV', 'FLIR', 'FLS', 'FLR', 'FMC', 'FTI', 'F', 'FOSL', 'BEN', 'FCX', 'FTR', 'GME', 'GPS',
               'GRMN', 'GD', 'GE', 'GGP', 'GIS', 'GM', 'GPC', 'GNW', 'GILD', 'GS', 'GT', 'GOOGL', 'GOOG', 'GWW', 'HAL', 'HBI', 'HOG',
               'HAR', 'HRS', 'HIG', 'HAS', 'HCA', 'HCP', 'HCN', 'HP', 'HES', 'HPQ', 'HD', 'HON', 'HRL', 'HSP', 'HST', 'HCBK', 'HUM',
               'HBAN', 'ITW', 'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IRM', 'JEC', 'JBHT', 'JNJ', 'JCI',
               'JOY', 'JPM', 'JNPR', 'KSU', 'K', 'KEY', 'GMCR', 'KMB', 'KIM', 'KMI', 'KLAC', 'KSS', 'KRFT', 'KR', 'LB', 'LLL', 'LH',
               'LRCX', 'LM', 'LEG', 'LEN', 'LVLT', 'LUK', 'LLY', 'LNC', 'LLTC', 'LMT', 'L', 'LOW', 'LYB', 'MTB', 'MAC', 'M', 'MNK', 'MRO',
               'MPC', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MAT', 'MKC', 'MCD', 'MHFI', 'MCK', 'MJN', 'MMV', 'MDT', 'MRK', 'MET', 'KORS',
               'MCHP', 'MU', 'MSFT', 'MHK', 'TAP', 'MDLZ', 'MON', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MUR', 'MYL', 'NDAQ', 'NOV', 'NAVI',
               'NTAP', 'NFLX', 'NWL', 'NFX', 'NEM', 'NWSA', 'NEE', 'NLSN', 'NKE', 'NI', 'NE', 'NBL', 'JWN', 'NSC', 'NTRS', 'NOC', 'NRG',
               'NUE', 'NVDA', 'ORLY', 'OXY', 'OMC', 'OKE', 'ORCL', 'OI', 'PCAR', 'PLL', 'PH', 'PDCO', 'PAYX', 'PNR', 'PBCT', 'POM', 'PEP',
               'PKI', 'PRGO', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PBI', 'PCL', 'PNC', 'RL', 'PPG', 'PPL', 'PX', 'PCP', 'PCLN', 'PFG',
               'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RRC', 'RTN', 'O', 'RHT', 'REGN', 'RF',
               'RSG', 'RAI', 'RHI', 'ROK', 'COL', 'ROP', 'ROST', 'RLC', 'R', 'CRM', 'SNDK', 'SCG', 'SLB', 'SNI', 'STX', 'SEE', 'SRE',
               'SHW', 'SIAL', 'SPG', 'SWKS', 'SLG', 'SJM', 'SNA', 'SO', 'LUV', 'SWN', 'SE', 'STJ', 'SWK', 'SPLS', 'SBUX', 'HOT', 'STT',
               'SRCL', 'SYK', 'STI', 'SYMC', 'SYY', 'TROW', 'TGT', 'TEL', 'TE', 'TGNA', 'THC', 'TDC', 'TSO', 'TXN', 'TXT', 'HSY', 'TRV',
               'TMO', 'TIF', 'TWX', 'TWC', 'TJK', 'TMK', 'TSS', 'TSCO', 'RIG', 'TRIP', 'FOXA', 'TSN', 'TYC', 'UA', 'UNP', 'UNH', 'UPS',
               'URI', 'UTX', 'UHS', 'UNM', 'URBN', 'VFC', 'VLO', 'VAR', 'VTR', 'VRSN', 'VZ', 'VRTX', 'VIAB', 'V', 'VNO', 'VMC', 'WMT',
               'WBA', 'DIS', 'WM', 'WAT', 'ANTM', 'WFC', 'WDC', 'WU', 'WY', 'WHR', 'WFM', 'WMB', 'WEC', 'WYN', 'WYNN', 'XEL', 'XRX',
               'XLNX', 'XL', 'XYL', 'YHOO', 'YUM', 'ZBH', 'ZION', 'ZTS']

# Available types from Yahoo Finance
tem = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

def YahooFinanceImport(x,y):
    # Get data from Yahoo Finance
    hist = yf.download(symbols, start = x, end = y, auto_adjust = False)
    # Initialize Values
    dist = {}
    for i in symbols:
        if (not pd.isna(hist['Close',i].values.tolist()[0])):
            # If the data of company is available, store it for next steps
            dist[i] = []
            for j in tem:
                # Add corresponding values to dictionary
                dist[i] = dist[i]+[hist[j,i].values.tolist()]
    return dist

def YFI():
    # Get data from Yahoo Finance
    # Turn into Stockstats dataframe, did some initial cleaning
    dist = {}
    for i in symbols:
        hist = SS.StockDataFrame.retype(yf.download(i, start = "2010-01-01", end = "2015-01-01", auto_adjust = False))
        dist[i] = hist.copy()
    # Data cleaning
    # delete empty dataframes or dataframes with large amount of NAs.
    na = []
    for k in dist:
        if(dist[k].empty):
            na = na+[k]
    for i in na:
        del dist[i]
    # replace sporadic NAs with previous day's pricing data

    naList = {}
    for k in dist:
        i = 0
        while i<=5:
            if (dist[k].isna().any()[i]):
                if k in naList:
                    naList[k].append(i)
                else:
                    naList[k] = [i]
            i=i+1
    SnaList = []
    for k in naList:
        j=0
        while j<=5:
            for i in range(len(dist[k].iloc[:,0])):
                if dist[k].iloc[:,j].isna()[i]:
                    SnaList = SnaList +[i]
            j=j+1
    j=0
    for k in naList:
        i=0
        while i <=5:
            temp = dist[k].iloc[:,i][SnaList[j]-1]
            dist[k].iloc[:,i][SnaList[j]] = temp
            i=i+1
            j=j+1

    Na = []
    for k in dist:
        t= pd.to_datetime(list(dist[k].index.values[[0]])[0])
        tstr= t.strftime("%Y-%m-%d")
        if tstr != '2010-01-04':
            Na = Na+[k]
    for i in Na:
        del dist[i]

    return dist

def TechnicalFunctions(dist):
    # Put Technical functions here
    return Scoring(dist)

def Scoring(dist):
    # Takes scores from technicals and come up with a final score,
    # will be divided into a scoring for the longs and a scoring for the shorts
    longs = LongScoring(dist)
    #shorts = ShortScoring(dist)
    shorts = LongScoring(dist)
    return Portfolio(longs, shorts)

def LongScoring(dist):
    # Temporary scoring function, will be replaced by maching learning in the future
    temp = []
    symb = dist.keys()
    for i in symb:
        # Takes average of technicals
        temp = temp + [sum(dist[i])/len(dist[i]), i]
    return temp

def Portfolio(longs, shorts):
    # Select companies for the final portfolio by select top 5 from each kind
    longs.sort(reverse = True)
    shorts.sort(reverse = True)
    temp = longs[:5] + shorts[:5]
    a = []
    for i in temp:
        a+=[i[1]]
    print(a)
    port = 'Recommendation of the longs: '
    for j in range(5):
        port += str(a[j]) +', '
    port = port[:-2] + '\n' + 'Recommendation of the shorts: '
    for k in range(5):
        port += str(a[k+5]) +', '
    port = port[:-2]
    return port

def partition(ssdf, cri):
    # partition a stockstats dataframe $sstf$ based on
    # criterion $cri$ (weekly, monthly, or quarterly)

    if cri == "weekly":
        partitioned_data = {}

        dates = list(ssdf.index)
        prev_dates = []
        future_dates = list(ssdf.index)

        i = 0
        # maintain the date of Monday of current week so that we can
        # know whether we should move on to the next week by counting
        # the days between cur_date and recent_Monday
        recent_Monday = dates[0] - datetime.timedelta(days = dates[0].weekday())
        while len(future_dates) != 0:

            if (i < len(future_dates)):
                cur_date = future_dates[i]
                weekday_delta = (cur_date - recent_Monday).days
            else:
                weekday_delta = 7

            # if we are in a new week, store the data of the recent week,
            # and update the loop
            if weekday_delta >= 7:
                cur_week_key = "week of " + recent_Monday.strftime("%Y-%m-%d %H:%M:%S")

                cur_dates = future_dates[:i]
                future_dates = future_dates[i:]

                dates_to_be_dropped = prev_dates + future_dates
                prev_dates = prev_dates + cur_dates

                cur_week_data = ssdf.drop(dates_to_be_dropped)
                partitioned_data[cur_week_key] = cur_week_data

                i = 0

            recent_Monday = cur_date - datetime.timedelta(days = cur_date.weekday())
            i = i+1

    elif cri == "monthly":
        partitioned_data = {}

        dates = list(ssdf.index)
        prev_dates = []
        future_dates = list(ssdf.index)

        i = 0
        # maintain the month and year of the previous date so that we can know
        # whether the month has changed or not
        prev_month = dates[0].month # initialized to be the month of the 1st date
        prev_year = dates[0].year
        while len(future_dates) != 0:

            if (i < len(future_dates)):
                cur_month = future_dates[i].month
                cur_year = future_dates[i].year
            else:
                cur_month = prev_month + 1

            # if the month changes, store the data of the previous month,
            # and update the loop
            if cur_month != prev_month or cur_year != prev_year:
                cur_month_key = str(prev_year) + "-" + str(prev_month)

                cur_dates = future_dates[:i]
                future_dates = future_dates[i:]
                dates_to_be_dropped = prev_dates + future_dates
                prev_dates = prev_dates + cur_dates

                cur_month_data = ssdf.drop(dates_to_be_dropped)
                partitioned_data[cur_month_key] = cur_month_data

                i = 0

            prev_month = cur_month
            prev_year = cur_year
            i = i+1

    elif cri == "quarterly":
        partitioned_data = {}

        dates = list(ssdf.index)
        prev_dates = []
        future_dates = list(ssdf.index)

        i = 0
        # maintain the quarter and year of the previous date so that
        # we can know whether the quarter has changed or not
        prev_quarter = (dates[0].month + 2) // 3 # initialized to be the quarter of the 1st date
        prev_year = dates[0].year # initialized to be the year of the 1st date
        while len(future_dates) != 0:

            if (i < len(future_dates)):
                cur_quarter = (future_dates[i].month + 2) // 3
                cur_year = future_dates[i].year
            else:
                cur_quarter = prev_quarter + 1

            # if the quarter changes, store the data of the previous quarter,
            # and update the loop
            if cur_quarter != prev_quarter or cur_year != prev_year:
                cur_quarter_key = str(prev_year) + "-" + str(prev_quarter)

                cur_dates = future_dates[:i]
                future_dates = future_dates[i:]
                dates_to_be_dropped = prev_dates + future_dates
                prev_dates = prev_dates + cur_dates

                cur_quarter_data = ssdf.drop(dates_to_be_dropped)
                partitioned_data[cur_quarter_key] = cur_quarter_data

                i = 0

            prev_quarter = cur_quarter
            prev_year = cur_year
            i = i+1

    return partitioned_data


def main():
    p = [[12,"MSFT"],[13,"ABT"],[4,"ACC"],[22,"VVV"],[44,"YY"],[33,"UIO"],]
    # print(Portfolio(p,p))
    return YahooFinanceImport("2010-01-01","2010-01-07")

data = YFI()
