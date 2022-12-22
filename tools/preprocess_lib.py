from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np

from datetime import datetime
import pandas as pd
import time
import matplotlib.pyplot as plt
import csv
from scipy import stats
from sklearn import preprocessing
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest

from telegrambotalarm import TelegramBot
import traceback

TOKEN = 'REDACTED_TOKEN'
# MYID = '-836623843' #kenergy group
MYID = '1246189043' #personal

bot = TelegramBot(TOKEN, MYID)

def inspectData(data):
    checkNull = data.isnull().any()
    lengthData = len(data)
    maxVal = data.max()
    minVal = data.min()
    varVal = data.var()
    stdVal = data.std()
    meanVal = data.mean()
    medianVal = data.median()
    print('Data nullities: ', checkNull, ' | Data length: ', lengthData)
    print('Data max value: ', maxVal, ' | Data min value: ', minVal)
    print('Data variance: ', varVal, ' | Data standard deviation: ', stdVal)
    print('Data mean: ', meanVal, ' | Data median: ', medianVal)
    plotAny(data)
    
def plotAny(data):
    plt.figure(num=None, figsize=(24, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(data)
    
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

def replace_outlier(df_in, col_name):
    outliers_fraction = float(.001)
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(df_in[col_name].values.reshape(-1, 1))
    data = pd.DataFrame(np_scaled)
    # train isolation forest
    model =  IsolationForest(contamination=outliers_fraction)
    model.fit(data)
    df_in['anomaly'] = model.predict(data)
    # visualization
    fig, ax = plt.subplots(figsize=(10,6))
    a = df_in.loc[df_in['anomaly'] == -1, [col_name]] #anomaly
    ax.plot(df_in.index, df_in[col_name], color='black', label = 'Normal')
    ax.scatter(a.index,a[col_name], color='red', label = 'Anomaly')
    plt.legend()
    plt.show();
    df_in[col_name][a.index] = np.nan
    df_in[col_name] = df_in[col_name].interpolate(method='nearest')
    plt.rc('figure',figsize=(12,6))
    plt.rc('font',size=15)
    df_in[col_name].plot()
    df_in = df_in.drop(columns='anomaly')
   
    return df_in
    
def distCheck(x):
    mean = np.mean(x)
    std = np.std(x)
    print('mean: ', mean, ' std: ', std)
    
    snd = stats.norm(mean, std)
    # x = df['value']
    plt.figure(figsize=(7.5,7.5))
    plt.plot(x, snd.pdf(x))
    # plt.xlim(-60, 60)
    figTitle = 'Normal Distribution | Mean: ' + str(mean) + ' STD: ' + str(std)
    plt.title(figTitle, fontsize='15')
    plt.xlabel('Values of Variable X', fontsize='15')
    plt.ylabel('Probability', fontsize='15')
    plt.show()
    
def diff_tr(data, interval):
    transformed_tr = []
    for i in range(interval, len(data)):
        result = data[i] - data[i-interval]
        transformed_tr.append(result)
    return transformed_tr
    # return [data[i] - data[i - interval] for i in range(interval, len(data))]
    
def genLaggedFeat(df, n_lags):
    df_n = df.copy()
    df_nn = df.copy()
    for i in range(1, n_lags+1):
        df_a = pd.DataFrame()
        df_a[f"lag{i}"] = df_n.shift(i)
        df_nn = pd.concat([df_nn, df_a], axis=1, join='inner')
        # df_n[f"lag{i}"] = df_n["value"].shift(i)
    # df_n = df_n.iloc[n_lags:]
    
    return df_nn

def genTimeFeat(data):
    data = (data
                    .assign(minute = df_transformed.index.minute)
                    # .assign(day = df.index.day)
                    # .assign(month = df.index.month)
                    # .assign(day_of_week = df.index.day_of_week)
                    # .assign(week_of_year = df.index.week)
                    )
    return data

def genCylicalFeat(df, col_name, period, start_num=0):
    kwargs = {
        f'sin_{col_name}': lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
        f'cos_{col_name}': lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)
    }
    return df.assign(**kwargs).drop(columns=col_name)

def sliding_window(data, seq_length, labels_length):
    x = []
    y = []
    z = []
    
    for i in range(len(data)-(seq_length+labels_length)):
        _x = data.iloc[i:(i+seq_length),:]
        _y = data.iloc[(i+seq_length):(i+seq_length+labels_length),0:1]
        _z = data.iloc[(i+seq_length):(i+seq_length+labels_length),1:]
        x.append(np.array(_x))
        y.append(np.array(_y))
        z.append(np.array(_z))
        
    return x,y,z

def conv_to_array(data):
    output = np.array(data)
    return output

def create_dataset (X, y, look_back = 1):
    Xs, ys = [], []
 
    for i in range(0,len(X)-look_back):
        v = X[i:i+look_back]
        w = y[i+look_back]
        Xs.append(v)
        ys.append(w)
 
    return np.array(Xs), np.array(ys)