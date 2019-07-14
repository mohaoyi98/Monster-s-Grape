
import pandas as pd
import tensorflow as tf
import numpy as np
import yfinance as yf
import stockstats as SS
from tensorflow.contrib import rnn
from tensorflow.keras import datasets, layers, models
import random

import alp1 # import alpha features in alp1.py

symbols = pd.read_excel('SP500.xlsx')
symbols = list(symbols['Symbol'])
#print(symbols)


# Available types from Yahoo Finance
#tem = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
def YFI():
    '''import stock data from yahoo finance'''
    # Get data from Yahoo Finance
    # Turn into Stockstats dataframe, did some initial cleaning
    dist = {}
    
    # get monthly data from yahoo finance 
    #hist = yf.download(symbols, period = '5y', interval = '1mo', group_by = 'ticker', auto_adjust = False)
    for i in symbols[:20]:
        dist[i] = SS.StockDataFrame.retype(yf.download(i, period = '5y', interval = '1mo', auto_adjust = False))
    # Data cleaning
    # delete empty dataframes or dataframes with large amount of NAs.
    na = []
    for k in dist:
        if(dist[k].empty):
            na = na+[k]
    for i in na:
        del dist[i]
    # delete invalid data

    for i in dist.keys():
        dist[i].dropna()
        temp = dist[i].index.values
        tflist = dist[i]['close'].isna()
        temp2 = []
        for j in range(len(dist[i]['close'])):
            if tflist[j]:
                temp2 =temp2 + [temp[j]]
        dist[i] = dist[i].drop(temp2)
        
    for i in dist.keys():
        for j in ['close', 'open', 'high', 'low']:
            for k in range(dist[i][j].shape[0]-1):
                a = dist[i][j][k]
                b = dist[i][j][k+1].copy()
                if a==b:
                    dist[i][j][k+1] = b+0.001
    na = []
    for k in dist:
        if(dist[k].empty):
            na = na+[k]
    for i in na:
        del dist[i] 

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
    #print('check')
    X = GetAlphasAll(dist)
    Y = TrueYTransform(dist)


    X_train, X_test = splitterX(X)
    Y_train, Y_test = splitterY(Y)
    X_train, Y_train = check(X_train, Y_train)
    X_test, Y_test = check(X_test, Y_test)
    #print(Y_train)
    alphas = X_train.columns
    alphaAcc = []
    for i in alphas:
        print(i)
        modeli, iacc, t = trainSingleAlpha(X_train[i], X_test[i], Y_train, Y_test)
        
        alphaAcc+= [[iacc, i, t]]

    alphaAcc.sort(reverse=True)
    #print(alphaAcc)
    selectedAlphas = []
    for j in range(len(alphaAcc)):
        if alphaAcc[j][2]==1:
            selectedAlphas += [alphaAcc[j]]
    if len(selectedAlphas) >=40:
        selectedAlphas = selectedAlphas[:40]
    alphaIndex = extractAlpha(selectedAlphas)
    #print('index',alphaIndex)
    model, acc = train(X_train[alphaIndex], X_test[alphaIndex], Y_train, Y_test)
    print('Final Accuracy:', acc)
    print(RMCompare(Y_test))
    PortVSSP500(model, dist, X.copy(), alphaIndex)
    return model #return the trained model

def PortVSSP500(model, dist, X, alphaIndex):
    pctrs = {}
    scores = []
    tempp = []
    for i in dist:
        if dist[i].shape[0]>1:
            a = (dist[i]['close'][-1] - dist[i]['close'][-2])/dist[i]['close'][-2]
            pctrs[i] = a
            tempp += [[a,i]]
            b = model.predict(X[i][alphaIndex])
            scores += [[b[-2][1], i]]
    tempp.sort(reverse=True)
    scores.sort(reverse=True)
    for i in range(len(tempp)):
        print(tempp[i][1], scores[i][1])

    temp = 0
    for i in scores[:10]:
        print(pctrs[i[1]])
        temp+=pctrs[i[1]]/10
    sppctr = SPpctr()
    print('S&P500:', sppctr, 'Portfolio return:', temp)
    return
    

def extractAlpha(lis):
    res = []
    for i in lis:
        res+= [i[1]]
    return res

def splitterX(dist):
    newT = pd.DataFrame()
    newR = pd.DataFrame()
    for i in dist.keys():
        cut = int(dist[i].shape[0]*0.8)
        train = dist[i].iloc[:cut]
        test = dist[i].iloc[cut:]
        newT = pd.concat([newT, train], axis=0, ignore_index=True)
        newR = pd.concat([newR, test], axis=0, ignore_index=True)
    return newT, newR

def splitterY(dist):
    newT = np.asarray([])
    newR = np.asarray([])
    for i in dist.keys():
        cut = int(dist[i].shape[0]*0.8)
        train = dist[i][:cut]
        test = dist[i][cut:]
        newT = np.append(newT, train)
        newR = np.append(newR, test)
    return newT, newR

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
        return (X[:temp]), (Y[:temp])
    else:
        return X, Y

def trainSingleAlpha(X_train, X_test, Y_train, Y_test):
    '''Train and test a regular neural network'''
    model = models.Sequential()
    model.add(layers.Flatten(input_shape = [1]))
    #model.add(layers.Dense(28000, activation = tf.nn.relu))
    #model.add(layers.Dense(20000, activation = tf.nn.relu))
    model.add(layers.Dense(512, activation = tf.nn.relu))
    #model.add(layers.Dense(12000, activation = tf.nn.relu))
    #model.add(layers.Dense(8000, activation = tf.nn.relu))
    #model.add(layers.Dense(6000, activation = tf.nn.relu))
    #model.add(layers.Dense(4000, activation = tf.nn.relu))
    #model.add(layers.Dense(2000, activation = tf.nn.relu))
    #model.add(layers.Dense(1000, activation = tf.nn.relu))
    #model.add(layers.Dense(500, activation = tf.nn.relu))
    model.add(layers.Dense(256, activation = tf.nn.relu))
    model.add(layers.Dense(16, activation=tf.nn.relu))
    model.add(layers.Dense(2, activation=tf.nn.softmax))
    # model.summary()
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=5, verbose = 0)
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose = 0)
    print(test_loss, test_acc)
    temp = 1
    if pd.isna(test_loss):
        temp = 0

    # print('Correct Prediction (%): ', accuracy_score(Y_test, model.predict(X_test), normalize=True)*100.0)
    return model, test_acc, temp

def train(X_train, X_test, Y_train, Y_test):
    '''Train and test a regular neural network'''
    model = models.Sequential()
    model.add(layers.Flatten(input_shape = [len(X_train.columns)]))
    #model.add(layers.Dense(28000, activation = tf.nn.relu))
    #model.add(layers.Dense(20000, activation = tf.nn.relu))
    model.add(layers.Dense(512, activation = tf.nn.relu))
    #model.add(layers.Dense(12000, activation = tf.nn.relu))
    #model.add(layers.Dense(8000, activation = tf.nn.relu))
    #model.add(layers.Dense(6000, activation = tf.nn.relu))
    #model.add(layers.Dense(4000, activation = tf.nn.relu))
    #model.add(layers.Dense(2000, activation = tf.nn.relu))
    #model.add(layers.Dense(1000, activation = tf.nn.relu))
    #model.add(layers.Dense(500, activation = tf.nn.relu))
    model.add(layers.Dense(256, activation = tf.nn.relu))
    model.add(layers.Dense(16, activation=tf.nn.relu))
    model.add(layers.Dense(2, activation=tf.nn.softmax))
    # model.summary()
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=5, verbose = 0)
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose = 0)
    
    # print('Correct Prediction (%): ', accuracy_score(Y_test, model.predict(X_test), normalize=True)*100.0)
    return model, test_acc

def TrueYTransform(dist):
    '''rescale price (Y) into 0/1'''
    new = {}
    for i in dist.keys():
        new[i]=np.asarray([])
        for j in range(dist[i].shape[0]-1):
            temp1 = dist[i]['close'][j]
            temp2 = dist[i]['close'][j+1]
            if temp1 < temp2:
                new[i] = np.append(new[i], [0])
            else:
                new[i] = np.append(new[i], [1])
    return new

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
    df = {}
    for i in dist.keys():
        temp = GetAlphas(dist[i].copy())
        if (temp.empty==False):
            df[i] = alp1.get_alpha(temp).drop(['adj close', 'close', 'high', 'low', 'open', 'volume', 'amount', 'pctr'], axis=1).fillna(value=0)
    return df

def GetAlphas(df):
    '''return the percentage return and quantum'''
    new = df.copy()[:-1]
    pctr = []
    amount = []
    for i in range(df.shape[0]-1):
        pctr += [(df['close'][i+1]-df['close'][i])/df['close'][i]] 
        amount += [df['close'][i]*df['volume'][i]]
    new['pctr'] = pctr
    new['amount'] = amount
    return new

def RMCompare(test):
    '''Generate comparison between machine and random machine'''
    ran = random_simul(len(test),[0,1])
    ran, test = check(ran, test)
    temp = ran-test
    ranAcc = 0
    for i in temp:
        if i == 0:
            ranAcc += 1
    ranAcc = ranAcc / len(ran)
    return ranAcc

def SPpctr():
    prices = yf.download('^GSPC', period = '2mo', interval = '1mo')
    price1 = prices['Close'][0]
    price2 = prices['Close'][1]
    return (price2-price1)/price1
#test area
main()

