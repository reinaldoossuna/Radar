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
#     display_name: Python [conda env:ic]
#     language: python
#     name: conda-env-ic-py
# ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Timedelta, Timestamp
from collections import namedtuple

# +
df_estacao = pd.read_csv("DATA/dados_estacoes_2015-2017.csv",
        parse_dates=["DATE"],
        index_col=["DATE"])

df_radar = pd.read_csv("DATA/radar.csv",
                      parse_dates=['date'],
                      index_col=["date"])

df_estacao.sort_index(inplace=True)
df_radar.sort_index(inplace=True)
# -

df_estacao.head()

df_radar.head()

# Compatibilizando o tempo nos dois datasets.
#  - O dados do radar estão acumulados a cada 10min, sendo 00:05:00 o acumulado do 00:00:00 até os 00:10:00.
#  - Separando a janela de tempo onde existem dados dos dois.
#  - O dados de radar estão em uma fuso hórario diferente.
#

df_est = df_estacao.groupby(pd.Grouper(freq="10min")).sum()
df_est.head()

df_radar.index[0]

df_radar.index[0]  - Timedelta("5min")

df_rad = df_radar.copy()
df_rad.index = df_radar.index - Timedelta("5min")
df_rad.head()

df_radar.index.min()

df_radar.index.max()

df_est = df_est[df_radar.index.min():df_radar.index.max()]
df_est.head()

# +
plt.figure()
df_est["2017-02-03":"2017-02-10"]["MB_PRO1"].plot(label="estação")
df_rad["2017-02-03":"2017-02-10"]["MB_PRO1"].plot(label="radar",alpha=0.5)

plt.legend()

# +
plt.figure()
df_est["2016-03-02":"2016-03-09"]["MB_PRO1"].plot(label="estação")
df_rad["2016-03-02":"2016-03-09"]["MB_PRO1"].plot(label="radar")

plt.legend()
# -

Horariodeverao = namedtuple("Horariodeverao",["inicio", "final"])
horariosdeverao = [Horariodeverao(Timestamp("2015-10-16"),Timestamp("2016-02-18")),
                  Horariodeverao(Timestamp("2015-10-15"),Timestamp("2016-02-17")),
                  Horariodeverao(Timestamp("2015-11-4"),Timestamp("2016-02-16"))]


horariosdeverao[0].inicio

indices = df_rad.index
new_indices = []
isHorariodeverao = False
for indice in indices:
    for horariodeverao in horariosdeverao:
        if indice >= horariodeverao.inicio and indice <= horariodeverao.final:
            isHorariodeverao = True
    if isHorariodeverao:
        delta = Timedelta("3Hours")
    else:
        delta = Timedelta("4Hours")
    new = indice - delta
    new_indices.append(new)
    
    isHorariodeverao = False


df_radar_corr = df_rad.copy()
df_radar_corr.index = new_indices

# Comparação no horario de verão

# +
plt.figure()
df_est["2017-02-03":"2017-02-10"]["MB_PRO1"].plot(label="estação")
df_radar_corr["2017-02-03":"2017-02-10"]["MB_PRO1"].plot(label="radar",alpha=0.5)

plt.legend()
# -

# Comparação fora do horario de verao

# +
plt.figure()
df_est["2017-03-02":"2017-03-12"]["MB_PRO1"].plot(label="estação")
df_radar_corr["2017-03-02":"2017-03-12"]["MB_PRO1"].plot(label="radar")

plt.legend()
# -

# Houve um tipo de sobreposição por causa do horario de verao. Na minha opiniao a melhor saida e pegar o valor maximo.

df_radar_corr.index[df_radar_corr.index.duplicated()]

duplicateds = df_radar_corr.index[df_radar_corr.index.duplicated()]
for duplicated in duplicateds:

    max_ = df_radar_corr.loc[duplicated].max(axis=0,level=0)
    
    df_radar_corr = df_radar_corr.drop(index=duplicated,axis=1)
    
    df_radar_corr = df_radar_corr.append(max_)


df_radar_corr.index[df_radar_corr.index.duplicated()]

df_radar_corr.index.name= "DATE"

df_radar_corr.head()

df_est.to_csv("DATA/dados_estacoes_5min.csv")
df_radar_corr.to_csv("DATA/dados_radar_semHorarioVerao.csv")


