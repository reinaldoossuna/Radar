# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

df_radar = pd.read_csv("DATA/data.csv",
                      parse_dates=['date'],
                      index_col=["date"])

df_radar.sort_index().index.to_series().describe()

df_radar.head()


def fromstring(x):
    try:
        return np.fromstring(x, sep=" ")
    except:
        return np.nan
    


df = df_radar.applymap(fromstring)

df.MB_SEG2.head(1).values


# +
def extractor(df,mirror):
    mirror = np.reshape(mirror,(25,))
    mirror = mirror.astype(bool)
    def mean_axis(x):
        try:
            return x[mirror].mean()
        except:
            return np.nan
    new_df = df.applymap(mean_axis)
    return new_df



# +
mirror = np.array([[0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,1,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0]])


df_1x1 = extractor(df,mirror)
df_1x1.to_csv("DATA/radar.csv")
df_1x1.head(1)

# +
mirror = np.array([[0,0,0,0,0],
            [0,0,1,0,0],
            [0,1,1,1,0],
            [0,0,1,0,0],
            [0,0,0,0,0]])


df_cruz = extractor(df,mirror)
df_cruz.head(1)

# +
mirror = np.array([[0,0,0,0,0],
            [0,1,1,1,0],
            [0,1,1,1,0],
            [0,1,1,1,0],
            [0,0,0,0,0]])


df_3x3 = extractor(df,mirror)
df_3x3.head(1)


# +
def compare(station,period,df1,df2, df_return= False):
    
    lr = linear_model.LinearRegression()
    
    df1_freg = df1.groupby(pd.Grouper(freq=period)).sum()
    df2_freg = df2.groupby(pd.Grouper(freq=period)).sum()
    
    df1_station = df1_freg[station]
    df2_station = df2_freg[station]
    
    df1_station = df1_station.rename(station + "_ESTACAO")
    df2_station = df2_station.rename(station + "_RADAR")
    
    
    df = pd.concat([df1_station,df2_station],axis=1)
    
    df.dropna(inplace=True)

    X = df.values[:,-1:]
    y = df.values[:,:-1]
    y = y.reshape(-1,)

    lr.fit(X,y)
   
    y_pred = lr.predict(X)
    
    # The coefficients
    print('Coefficients: \n', lr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(y, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('R2 score: %.2f' % r2_score(y, y_pred))
    
    plt.figure(1, figsize=(15,5))
    plt.suptitle("Valores acumulados por {} na Estação {}".format(period,station), fontsize=16)
    
    plt.subplot(121)
    plt.plot(df[station + "_ESTACAO"],alpha=1, color='b',label="ESTAÇÃO")
    plt.plot(df[station + "_RADAR"],alpha=0.6, color='orange',label="RADAR")
    #plt.yscale("log")
    plt.ylabel("Valores Radar")
    plt.legend()
    

    plt.subplot(122)
    plt.ylim(bottom=-1, top=60)
    plt.scatter(df[station + "_RADAR"],df[station + "_ESTACAO"])
    plt.plot(X,y_pred,color='black', linewidth=1)
    plt.xlabel("RADAR")
    plt.ylabel("ESTAÇÃO")
    

    plt.show()
    if df_return:
        return df
    
    
# -

# # MB_PRO1 Comparação entre Pixel, Cruz e 3x3;

# Podemos ver que a a diferença entre usar a média dos valores em torno é insignificante

compare("MB_PRO1","24H",df_estacao,df_1x1)
compare("MB_PRO1","24H",df_estacao,df_cruz)
compare("MB_PRO1","24H",df_estacao,df_3x3)

