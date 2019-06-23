# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 14:48:16 2019

@author: sheng
"""

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
import yfinance as yf
import stockstats as SS
from tensorflow.contrib import rnn
from tensorflow.keras import datasets, layers, models
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
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
def YFI():
    # Get data from Yahoo Finance
    # Turn into Stockstats dataframe, did some initial cleaning
    dist = {}
    for i in ["MSFT","MLM","MA"]:
        hist = SS.StockDataFrame.retype(yf.download(i, start = "2010-01-01", end = "2015-05-01", auto_adjust = False))
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

def CreateFeatures(dist):
    sym = dist.keys()
    new = {}
    for i in sym:
        ind = dist[i].copy()
        SSdf = SS.StockDataFrame.retype(ind)
        temp=pd.DataFrame(columns = ["macd","rsi_12","atr","cci","dma", "cr-ma3", "kdjk", "boll", "pdi"])
        temp["cr-ma3"] = SSdf["cr-ma3"]
        temp["kdjk"] = SSdf["kdjk"]
        temp["boll"] = SSdf["boll"]
        temp["pdi"] = SSdf["pdi"]
        temp["macd"] = SSdf["macd"]
        temp["rsi_12"] = SSdf["rsi_12"]
        temp["atr"] = SSdf["atr"]
        temp['cci'] = SSdf["cci"]
        temp['dma'] = SSdf["dma"]
        temp.dropna(inplace=True)
        new[i] = temp.copy()
    return new

def main():
    dist = YFI()
    feas = CreateFeatures(dist)
    symbol = dist.keys()
    modelss = {}
    for i in symbol:
        X = feas[i]
        # print(X.shape[0])
        dfs = dist[i][:].copy()
        Y = TrueYTransform(dfs['adj close'])[9:]
        # print(Y.shape[0])
        length = dfs.shape[0]
        split = int(length*0.80)
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]
        modelss[i] = train(X_train, X_test, Y_train, Y_test)
    return modelss

def train(X_train, X_test, Y_train, Y_test):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape = [9]))
    model.add(layers.Dense(128, activation = tf.nn.relu))
    model.add(layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=10, verbose = 0)
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print('accuracy', test_acc)
    print('Result', model.predict(X_test)[0])
    # print('Correct Prediction (%): ', accuracy_score(Y_test, model.predict(X_test), normalize=True)*100.0)
    return model

def TrueYTransform(prices):
    temp=[]
    for i in range(len(prices)-1):
        percentageR = ((prices[i+1]-prices[i])/prices[i])
        rescale = int(sigmoid(percentageR)*10)
        temp+=[rescale]
    return np.asarray(temp)
    
main()