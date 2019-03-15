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

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline

plt.style.use("bmh")
from IPython.core.debugger import set_trace

# +
df = pd.read_csv("DATA/eventos.csv",parse_dates=["start","duration"])

df.dropna(how="any", axis=0,inplace=True)
# -

df = df[df.RADAR > 0]

df.duration = df.duration.apply(pd.to_timedelta)

df.head()

df.info()

MARCO = 3
df_marco = df[df.start.dt.month == MARCO]
df_marco.head()

radar_hist, bin_edges = np.histogram(df_marco.RADAR.values,bins="auto",normed=True)
pluvi_hist, bin_edges2 = np.histogram(df_marco.Pluviometro.values,bins=bin_edges,normed=True)
dx = bin_edges[1] - bin_edges[0]
radar_cdf = np.cumsum(radar_hist) * dx
pluvi_cdf = np.cumsum(pluvi_hist) * dx
plt.figure()
plt.plot(bin_edges[1:], radar_cdf)
plt.plot(bin_edges[1:],pluvi_cdf)

plt.plot(bin_edges[1:], radar_cdf)

plt.plot(pluvi_cdf,bin_edges[1:])

# Para o ajuste dos valores do radar, primeiro encontramos o valor do CDF correspondente ao do radar. E depois com esse valor do CDF encontramos um valor na curva do Pluviometro.

itp_radar_cdf = np.poly1d(np.polyfit(bin_edges[1:], radar_cdf, 5))
itp_pluvi_ppf = np.poly1d(np.polyfit(pluvi_cdf, bin_edges[1:], 3))

x = np.linspace(0,70)
radar = 20
cdf_radar = itp_radar_cdf(20)
print(cdf_radar)
plt.plot(x,itp_radar_cdf(x))
plt.scatter(radar,cdf_radar)


# Ajustar o gráfico ppf do pluviometro com algum polinomio é mais complicado, veja abaixo como a curva vai ter um comportamento estranho.

# +
valor_corrigido = itp_pluvi_ppf(cdf_radar)

eixo_cdf = np.linspace(0,1)
print(valor_corrigido)
plt.plot(eixo_cdf,itp_pluvi_ppf(eixo_cdf))
#plt.scatter(valor_corrigido,valor_corrigido)
plt.ylim((0,100))
# -

# Para encontrar uma curva satisfatoria para ambos os casos, vou usar uma Spline
#
# * https://en.wikipedia.org/wiki/Spline_(mathematics)
# * https://en.wikipedia.org/wiki/Spline_interpolation
#
# Eu acredito que com uma sigmoid tambem seria possivel ajustar o ppf.
#
# https://en.wikipedia.org/wiki/Sigmoid_function

# +
x = np.linspace(0,80,50)

radar_dist = stats.rv_histogram(np.histogram(df_marco.RADAR.values,bins="auto",normed=True))
pluvi_dist = stats.rv_histogram(np.histogram(df_marco.Pluviometro.values,bins="auto",normed=True))
# -

itp_radar_cdf = UnivariateSpline(x, radar_dist.cdf(x),k=3,s=0)

# Vamos ajustar o um valor de radar de 20mm

itp_radar = itp_radar_cdf(radar)
print(itp_radar)
plt.plot(x,itp_radar_cdf(x))
plt.scatter(radar,itp_radar)

# O problema dessa interpolação é que os valores de x **precisa** estar aumentando

itp_pluvi_ppf = InterpolatedUnivariateSpline(pluvi_dist.cdf(x),x, k=3)

pluvi_cdf = pluvi_dist.cdf(x)
pluvi_cdf

# Descobrir onde o array não está "crescendo"

idx_increasing = (pluvi_cdf - np.roll(pluvi_cdf,1))[1:] > 0
idx_increasing = np.insert(idx_increasing,0,True)
idx_increasing

pluvi_cdf = pluvi_cdf[idx_increasing]
x = x[idx_increasing]

plt.plot(pluvi_cdf,x)

itp_pluvi_ppf = UnivariateSpline(pluvi_cdf,x, k=5,s=1)

# Vamos agora jogar o valor da interpolação do radar, na curva ppf do pluviometro.

itp_pluvi = itp_pluvi_ppf(itp_radar)
print(f"Valor real de precipitação : {round(itp_pluvi.tolist(),2)}mm, quando o Radar mostrar valor de {radar}mm")
plt.plot(itp_radar_cdf(x),itp_pluvi_ppf(itp_radar_cdf(x)));


meses = ["Janeiro",
"Fevereiro",
"Março",
"Abril",
"Maio",
"Junho",
"Julho",
"Agosto",
"Setembro",
"Outubro",
"Novembro",
"Dezembro"]


def save_graph(radar_cdf, pluvi_cdf,corrected_quantiles, month):
    plt.figure()

    n_month = meses[month-1]

    plt.title(n_month)
    
    plt.plot(corrected_quantiles, radar_cdf,label="Radar")
    plt.plot(corrected_quantiles, pluvi_cdf,label="Pluviometro")

    name = "IMAGENS/CDFs/{}.png".format(n_month)
    
    plt.legend()
    plt.savefig(name)


# +
def isIncresing(array):
    return all((array - np.roll(array,1))[1:] > 0)

def idx_of_Increasing(array):
    idx_increasing = (array - np.roll(array,1))[1:] > 0
    idx_increasing = np.insert(idx_increasing,0,True)
    return idx_increasing



# -

def cdf(radar,pluviometro,k=3):
    

    
    radar_dist = stats.rv_histogram(np.histogram(radar, bins="auto",normed=True))
    pluvi_dist = stats.rv_histogram(np.histogram(pluviometro,bins="auto",normed=True))

    max_ = max(np.max(radar),np.max(pluviometro))
    
    x = np.linspace(0,max_,50)
    radar_cdf = radar_dist.cdf(x)
    itp_radar_cdf = UnivariateSpline(x, radar_cdf,s=1,k=3)

    pluvi_cdf = pluvi_dist.cdf(x)
    if not isIncresing(pluvi_cdf):
        idx_increasing = idx_of_Increasing(pluvi_cdf)
        pluvi_cdf = pluvi_cdf[idx_increasing]
        x = x[idx_increasing]
    itp_pluvi_ppf = UnivariateSpline(pluvi_cdf,x, s=1,k=3)
    
    save_graph(radar_dist.cdf(x),pluvi_dist.cdf(x),x,month)
    
    return lambda x: itp_pluvi_ppf(itp_radar_cdf(x))


def correction(serie,itps):
    month = serie["start"].month
    
    serie["CDF_CORR"] = itps[month](serie["RADAR"])
    return serie


months = df.start.dt.month.unique()
functions = {}
itps = {}
for month in months:
    df_month = df[df.start.dt.month == month]    
    itps[month] = cdf(df_month.RADAR.values,df_month.Pluviometro.values)
    

df = df.apply(correction,args=[itps],axis=1)
df.head()

df["CDF_ERROR"] = df.Pluviometro - df.CDF_CORR
df.head()

df[["CDF_CORR", "CDF_ERROR"]] = df[["CDF_CORR", "CDF_ERROR"]].astype(float)

df.describe()

df
