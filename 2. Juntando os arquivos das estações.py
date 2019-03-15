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
from pathlib import Path

# ### Use o arquivo excel_extract_data.py ao invés de rodar cada célula deste notebook!
#

PATH = Path("DATA/Dados Compilados/")

df = pd.read_excel("DATA/Dados Compilados/Dados Compilados 2018.xlsx",index_col="DATE")

# A colunas do arquivo excell tem um formato diferente do radar!

df.columns


def change_names(list_columns):
    """
    Função com o objetivo de mudar os nomes presentes no excel ficarem 
    iguais ao do data extraido
    """
    new_columns = []
    for name in list_columns:
       new = name.strip().replace(" ","_")
       pos_ = new.find("_",4)
       new = new[:pos_] + new[pos_+1:]
       new_columns.append(new)
    return new_columns



change_names(df.columns)

list(PATH.glob("*.xlsx"))

# +
files = []
for excel in PATH.glob("*.xlsx"):

    df = pd.read_excel(excel, index_col="DATE")
    files.append(df)

df_concat = pd.concat(files,sort=True)
df_concat.groupby("DATE").sum()
df_concat.columns = change_names(df_concat.columns)

# fazendo o delta entre os valores
df_shift =  df_concat - df_concat.shift(1)
# Iguala qualquer valor negativo a 0
df_shift[df_shift < 0] = 0


csv_file =  "DATA/dados_estacoes_2015-2017.csv"
df_shift.to_csv(csv_file, index=True)
            
# -

# # Info sobre o dataset

df = pd.read_csv( "DATA/dados_estacoes_2015-2017.csv",
        parse_dates=["DATE"],
        index_col=["DATE"])

# Como efeito de confirmação que tudo está como esperado escolhemos um evento em particular e comparamos como arquivo do excell

df.loc["201601090530":"201601090552",'MB_PRO1']

# ![2](IMAGENS/excell.png)

df.sort_index()
df.head()

df.describe()

# +

df.index.to_series().describe()
