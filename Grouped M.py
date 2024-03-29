# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:58:25 2019

@author: sheng
"""

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

import alp1 # import alpha features in alp1.py

symbols = pd.read_excel('SP500.xlsx')
tags = pd.concat([symbols['Symbol'],symbols['GICS Sector']], axis = 1)
symbols = list(symbols['Symbol'])
#print(tags)
#print(symbols)


# Available types from Yahoo Finance
tem = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
def YFI():
    '''import stock data from yahoo finance'''
    # Get data from Yahoo Finance
    # Turn into Stockstats dataframe, did some initial cleaning
    dist = {}
    #today = str(dt.datetime.today())[:10]
    #startdate = str(dt.datetime.today()-dt.timedelta(days=360))[:10]
    
    # get monthly data from yahoo finance 
    hist = yf.download(symbols, period = '5y', interval = '1mo', group_by = 'ticker', auto_adjust = False)
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
    '''return all kinds of index; we can use them as features'''
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
        temp = TrueYTransform(dfs['adj close']) # rescale 'adj close' data 
        Y = np.append(Y, temp) # add the rescaled data into Y
    X, Y = check(X, Y) # adjust the length of X and Y
    print(len(X), len(Y))
    length = len(Y)
    split = int(length*0.75) # 75% of the data for trianing; 25% of data for testing
    # choice = random.sample(range(length), split)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    #print(len(X_train), len(Y_train), len(X_test), len(Y_test))
    #X_train, Y_train = check(X_train, Y_train)
    #X_test, Y_test = check(X_test, Y_test)
    modeli, iacc = train(X_train, X_test, Y_train, Y_test) # fit a NN model and give a testing result
    print(iacc)# print the testing output: "accuracy"
    RMCompare(Y_test, iacc)
    return modeli, iacc #return the trained model

def get_weighted_classes(weights, classes):
    '''
    input: weights -- a list of weights assigned to each class, 
                      each weight should be an int
           classes -- a list of classes
    output: a list of weighted classes, where each class appears n times
            (n is the weight assigned to that class)
    '''
    weighted_classes = []
    
    i = 0
    while i < len(weights) and i < len(classes):
        next_weight, next_class = weights[i], classes[i]
        assert type(weights[i]) == int
        
        weighted_classes += weights[i] * [next_class]
        
        i += 1
    
    return weighted_classes
    
def random_simul(length, weighted_classes):
    '''
    input: length -- int, the # of data points in the dataset
           weighted_classes -- a list of weighted classes
    output: a np array with randomly simulated y values
    '''
    y_simul = np.asarray(range(length))
    
    for i in range(length):
        y_simul[i] = random.choice(weighted_classes)
    
    return y_simul

def check(X, Y):
    ''' to keep X and Y in the same length'''
    lx = len(X)
    ly = len(Y)
    if lx != ly:
        temp = min(lx,ly)
        return X[:temp], Y[:temp]
    else:
        return X, Y

def train(X_train, X_test, Y_train, Y_test):
    '''Train and test a regular neural network'''
    model = models.Sequential()
    model.add(layers.Flatten(input_shape = [82]))
    #model.add(layers.Dense(28000, activation = tf.nn.relu))
    #model.add(layers.Dense(20000, activation = tf.nn.relu))
    model.add(layers.Dense(16000, activation = tf.nn.relu))
    #model.add(layers.Dense(12000, activation = tf.nn.relu))
    #model.add(layers.Dense(8000, activation = tf.nn.relu))
    #model.add(layers.Dense(6000, activation = tf.nn.relu))
    #model.add(layers.Dense(4000, activation = tf.nn.relu))
    #model.add(layers.Dense(2000, activation = tf.nn.relu))
    #model.add(layers.Dense(1000, activation = tf.nn.relu))
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
    '''rescale price (Y) into 0/1'''
    temp=[]
    for i in range(len(prices)-1):
        percentageR = ((prices[i+1]-prices[i])/prices[i])
        if percentageR >=0:
            temp+=[1]
        else:
  
            temp+=[0]
    return np.asarray(temp)

def PriceToEarningPerShare(prices):
    '''Earnings per share: a portion of a company's profit that is allocated to one share of stock.
       The definition comes from the website: "https://www.wikihow.com/Calculate-Earnings-Per-Share"'''
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
    '''Return dataframe of companies with corresponding alpha values'''
    df = pd.DataFrame()
    for i in dist:
        df = pd.concat([df, GetAlphas(dist[i].copy())], ignore_index=True) # unite the percentage return, quantum and original data of all the companies into one big dataframe
    return alp1.get_alpha(df).dropna().drop(['adj close', 'close', 'high', 'low', 'open', 'volume', 'amount', 'pctr'], axis=1) # only return the values of alphas of all the companies in alp1.py 

def GetAlphas(df):
    '''return the percentage return and quantum'''
    new = df.copy()[:-1]
    pctr = []
    amount = []
    for i in range(df.shape[0]-1):
        pctr += [(df['close'][i+1]-df['close'][i])/df['close'][i]] # percentage of return
        amount += [df['close'][i]*df['volume'][i]] # quantum (amount) = price * volume
    new['pctr'] = pctr
    new['amount'] = amount
    return new

def RMCompare(test, acc):
    '''Generate comparison between machine and random machine'''
    ran = random_simul(500,[0,1])
    ran, test = check(ran, test)
    temp = ran-test
    ranAcc = 0
    for i in temp:
        if i == 0:
            ranAcc += 1
    ranAcc = ranAcc / len(ran)
    model, iacc = main()
    print('Model Accuracy:', iacc, 'Random Accuracy:', acc)
    return
#test area


