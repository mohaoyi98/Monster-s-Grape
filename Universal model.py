# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:20:26 2019

@author: sheng
"""

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
import datetime as dt
import random

import alp1

symbols = pd.read_excel('SP500.xlsx')
symbols = list(symbols['Symbol'])
#print(symbols)


# Available types from Yahoo Finance
tem = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
def YFI():
    # Get data from Yahoo Finance
    # Turn into Stockstats dataframe, did some initial cleaning
    dist = {}
    #today = str(dt.datetime.today())[:10]
    #startdate = str(dt.datetime.today()-dt.timedelta(days=360))[:10]
    

    hist = yf.download(symbols, period = '10y', interval = '1mo', group_by = 'ticker', auto_adjust = False)
    for i in symbols:
        dist[i] = SS.StockDataFrame.retype(hist[i].copy())
    # Data cleaning
    # delete empty dataframes or dataframes with large amount of NAs.
    na = []
    for k in dist:
        if(dist[k].empty):
            na = na+[k]
    for i in na:
        del dist[i]
    # replace sporadic NAs with previous day's pricing data
    """
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
     """
    for i in dist.keys():
        dist[i].dropna()
        temp = dist[i].index.values
        tflist = dist[i]['close'].isna()
        temp2 = []
        for j in range(len(dist[i]['close'])):
            if tflist[j]:
                temp2 =temp2 + [temp[j]]
        dist[i] = dist[i].drop(temp2)
        
    
    return dist

def CreateFeatures(dist):
    sym = dist.keys()
    new = {}
    for i in sym:
        ind = dist[i].copy()
        SSdf = SS.StockDataFrame.retype(ind)
        temp=pd.DataFrame(columns = ["macd","rsi_12","atr","cci","dma", "cr-ma3", "kdjk", "boll", "pdi", "P/E"])
        temp["cr-ma3"] = SSdf["cr-ma3"]
        temp["kdjk"] = SSdf["kdjk"]
        temp["boll"] = SSdf["boll"]
        temp["pdi"] = SSdf["pdi"]
        temp["macd"] = SSdf["macd"]
        temp["rsi_12"] = SSdf["rsi_12"]
        temp["atr"] = SSdf["atr"]
        temp['cci'] = SSdf["cci"]
        temp['dma'] = SSdf["dma"]
        # print(ind['adj close'])
        temp['P/E'] = SS.StockDataFrame.retype(PriceToEarningPerShare(ind['adj close']))
        temp.dropna(inplace=True)
        new[i] = temp.copy()
    return new

def main():
    dist = YFI()
    # print(dist)
    #feas = CreateFeatures(dist)
    X = GetAlphasAll(dist)
    symbol = dist.keys()
    Y = np.asarray([])
    for i in symbol:
        dfs = dist[i][:].copy()
        temp = TrueYTransform(dfs['adj close'])
        Y = np.append(Y, temp)
    X, Y = check(X, Y)
    print(len(X), len(Y))
    length = len(Y)
    split = int(length*0.75)
    # choice = random.sample(range(length), split)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    #print(len(X_train), len(Y_train), len(X_test), len(Y_test))
    #X_train, Y_train = check(X_train, Y_train)
    #X_test, Y_test = check(X_test, Y_test)
    modeli, iacc = train(X_train, X_test, Y_train, Y_test)
    print(iacc)
    return modeli

def check(X, Y):
    lx = len(X)
    ly = len(Y)
    if lx != ly:
        temp = min(lx,ly)
        return X[:temp], Y[:temp]
    else:
        return X, Y

def train(X_train, X_test, Y_train, Y_test):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape = [82]))
    model.add(layers.Dense(28000, activation = tf.nn.relu))
    #model.add(layers.Dense(20000, activation = tf.nn.relu))
    model.add(layers.Dense(16000, activation = tf.nn.relu))
    #model.add(layers.Dense(12000, activation = tf.nn.relu))
    model.add(layers.Dense(8000, activation = tf.nn.relu))
    #model.add(layers.Dense(6000, activation = tf.nn.relu))
    model.add(layers.Dense(4000, activation = tf.nn.relu))
    #model.add(layers.Dense(2000, activation = tf.nn.relu))
    model.add(layers.Dense(1000, activation = tf.nn.relu))
    #model.add(layers.Dense(500, activation = tf.nn.relu))
    model.add(layers.Dense(256, activation = tf.nn.relu))
    model.add(layers.Dense(16, activation=tf.nn.softmax))
    # model.summary()
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=5, verbose = 1)
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose = 1)
    
    # print('Correct Prediction (%): ', accuracy_score(Y_test, model.predict(X_test), normalize=True)*100.0)
    return model, test_acc

def TrueYTransform(prices):
    temp=[]
    for i in range(len(prices)-1):
        percentageR = ((prices[i+1]-prices[i])/prices[i])
        if percentageR >=0:
            temp+=[1]
        else:
  
            temp+=[0]
    return np.asarray(temp)

def PriceToEarningPerShare(prices):
    temp=[]
    for i in range(len(prices)-1):
        percentageR = (prices[i+1]-prices[i])
        if percentageR == 0:
            res = prices[i+1]/0.01
        else:
            res = prices[i+1]/percentageR
        temp+=[res]
    temp = [prices[1]-prices[0]/0.01]+temp
    
    return np.asarray(temp)

def GetAlphasAll(dist):
    df = pd.DataFrame()
    for i in dist:
        df = pd.concat([df, GetAlphas(dist[i].copy())], ignore_index=True)
    return alp1.get_alpha(df).dropna().drop(['adj close', 'close', 'high', 'low', 'open', 'volume', 'amount', 'pctr'], axis=1)

def GetAlphas(df):
    new = df.copy()[:-1]
    pctr = []
    amount = []
    for i in range(df.shape[0]-1):
        pctr += [(df['close'][i+1]-df['close'][i])/df['close'][i]]
        amount += [df['close'][i]*df['volume'][i]]
    new['pctr'] = pctr
    new['amount'] = amount
    return new

main()
