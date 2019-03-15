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
from sklearn.metrics import mean_squared_error,\
                            mean_absolute_error,\
                            r2_score
import seaborn as sns

# +
df_estacao = pd.read_csv("DATA/dados_estacoes_2015-2017.csv",
        parse_dates=["DATE"],
        index_col=["DATE"])

df_radar = pd.read_csv("DATA/radar.csv",
                      parse_dates=['date'],
                      index_col=["date"])
# -

df_estacao.sort_index().index.to_series().describe()

df_radar.sort_index().index.to_series().describe()


# +
def compare(station,period,df_estacao,df_radar, df_return= False):
    
   
    df_estacao_freg = df_estacao.groupby(pd.Grouper(freq=period)).sum()
    df_radar_freg = df_radar.groupby(pd.Grouper(freq=period)).sum()
    
    df_estacao_station = df_estacao_freg[station]
    df_radar_station = df_radar_freg[station]
    
    df_estacao_station = df_estacao_station.rename(station + "_ESTACAO")
    df_radar_station = df_radar_station.rename(station + "_RADAR")
    
    
    df = pd.concat([df_estacao_station,df_radar_station],axis=1)
    
    df.dropna(inplace=True)
    

    
    y = df[station + "_ESTACAO"]
    y_pred = df[station + "_RADAR"]

 
    
    plt.figure(1, figsize=(10,10))
    plt.suptitle(station, fontsize=18)
    

    plt.style.use("seaborn")
    plt.subplot(111)
    plt.ylim(bottom=-1, top=70)
    plt.xlim(-1, 70)
    plt.scatter(df[station + "_RADAR"],df[station + "_ESTACAO"])
    sns.regplot(y_pred,y, ci=None)
    plt.xlabel("Radar (mm)")
    plt.ylabel("Estação (mm)")
    plt.text(55,10, f'MSE = {round(mean_squared_error(y, y_pred),2)}', fontsize=15)
    plt.text(55,7, f'MAE = {round(mean_absolute_error(y, y_pred),2)}', fontsize=15)
    plt.text(55,4, f'R2 = {round(r2_score(y, y_pred),2)}', fontsize=15)
    
    

    plt.show()
    if df_return:
        return df
    
    
# -

compare("MB_PRO1","24H",df_estacao["2016"],df_radar["2016"])

stations = [station for station in df_radar.columns if station in df_estacao.columns]
for station in stations:
    compare(station,"24H",df_estacao,df_radar)
