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

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Timedelta
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

    
plt.style.use("bmh")
from IPython.core.debugger import set_trace

# +
df = pd.read_csv("DATA/eventos6h.csv",parse_dates=["start","duration"])

df.dropna(how="any", axis=0,inplace=True)
# -

df.dropna(axis=1,inplace=True)

df.duration = df.duration.apply(pd.to_timedelta)

df.head()

pares = [(0,10),(10,20),(20,30),(30,40),(40,50),(50,60)]


def plot_function(radar_par,pluvi_par,f,par,save = False):
    diff = pluvi_par - radar_par
    r2 = f"R2: {round(r2_score(diff, f(radar_par)),4)}"
    name = f"IMAGENS/cdf2/function-{par}.png"

    plt.figure()
    plt.title("Range " + str(par))
    plt.scatter(radar_par, diff)
    x = np.linspace(par[0],par[1])
    y_pred = f(x)
    plt.plot(x, y_pred,"lightblue")
    plt.ylabel("Difference Gauge - Radar (mm)")
    plt.xlabel("Radar (mm)")
    x = .8 * (plt.xlim()[1] - plt.xlim()[0]) + plt.xlim()[0]
    y = .15 * (plt.ylim()[1] - plt.ylim()[0]) + plt.ylim()[0]
    plt.grid(False)
    plt.annotate(r2,(x,y))
    if save:
        plt.savefig(name)
    plt.show()


functions = {}
radar = df.RADAR.sort_values().values
pluvi = df.Pluviometro.sort_values().values
pares = [(0,5),(5,10),(10,20),(20,30),(30,40),(40,50),(50,60)]
for par in pares:
        
    cond1 = np.where(radar >= par[0])
    cond2 = np.where(radar < par[1])
    start = cond1[0][0]
    end = cond2[0][-1]
    radar_par = radar[start:end]
    pluvi_par = pluvi[start:end]
    diff = pluvi_par - radar_par
    
    
    f = np.poly1d(np.polyfit(radar_par, diff, 5))
    functions[par[1]] = f
    plot_function(radar_par,pluvi_par,f,par,save=True)

df.head()

bias = functions[20](df.RADAR[df.RADAR < 20])

serie = df.iloc[122]
serie

serie["RADAR"] < 20

functions[40](serie["RADAR"])

serie["RADAR"] - functions[40](serie["RADAR"]) if serie["ERROR"] < 0 else functions[40](serie["RADAR"]) + serie["RADAR"]


def correction(serie):

    for par in pares:
        if serie["RADAR"] < par[1]:
            new_serie = serie.copy()
            y = functions[par[1]](serie["RADAR"])
            corr = serie["RADAR"] - y if serie["ERROR"] < 0 else serie["RADAR"] + y
            new_serie["CORREC_RADAR"] = corr
            return new_serie



correction(serie)

df_n = df.apply(correction,axis=1)

any(df_n.CORREC_RADAR < 0)

df_n.CORREC_RADAR[df_n.CORREC_RADAR < 0] = 0

any(df_n.CORREC_RADAR < 0)

df_n.head()

df_n["ERROR_CORR"] = df.Pluviometro - df_n.CORREC_RADAR

df_n.head()

df_n.describe()

pearsonr(df_n.Pluviometro.values,df_n.RADAR.values)

pearsonr(df_n.Pluviometro.values,df_n.CORREC_RADAR.values)

for par in pares:
    df_20 = df_n.query("RADAR >= @par[0] & RADAR < @par[1]")
    print(par)
    print(pearsonr(df_20.Pluviometro.values,df_20.RADAR.values)[0] ** 2)
    print(pearsonr(df_20.Pluviometro.values,df_20.CORREC_RADAR.values)[0] ** 2)
    print()

functions

df_n.sort_values("RADAR",inplace=True)

df_n.dropna(axis=0,how="all",inplace=True)

df_n.tail()

par = pares[1]
radar = df_n.RADAR.values
pluvi = df_n.Pluviometro.values
corr_radar = df_n.CORREC_RADAR.values
cond1 = np.where(radar >= par[0])
cond2 = np.where(radar < par[1])
start = cond1[0][0]
end = cond2[0][-1]
radar_par=radar[start:end]
pluvi_par = pluvi[start:end]
corr_par = corr_radar[start:end]


# +
def plot_scatter(ax, x, y, ylabel):
    x_ = np.arange(0,80)
    f = np.poly1d(np.polyfit(x,y,1))
    y_ = f(x_)
    
    ax.set_title(f"{ylabel} x GAUGE")
    ax.scatter(x,y)
    ax.set_ylim(0,80)
    ax.set_xlim(0,80)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("GAUGE")
    ax.plot(x_,y_,"lightblue")
    r2 = f"R2: {round(pearsonr(x,y)[0]**2,3)}"
    ax.annotate(r2,(65,10))
    ax.grid(False)
    ;
    
    
# -

def plot_figure(radar_par,pluvi_par,corr_par,save=False,name=None):
    
    
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))

    plot_scatter(ax1,radar_par,pluvi_par,"RADAR")
    plot_scatter(ax2,corr_par,pluvi_par,"CORRECTED RADAR")

    if save and name:
        name = f"IMAGENS/cdf2/Distribution-{name}.png"
        plt.savefig(name)
    plt.show();

plot_figure(radar_par,pluvi_par,corr_par)

# +
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))

ax1.scatter(radar_par,pluvi_par)
ax2.scatter(corr_par, pluvi_par)
plt.show();


# -

def plot_cdf(radar_par, pluvi_par, corr_par,save=False,name=None):
    
    cdf = np.arange(len(radar_par)) / len(radar_par)
    
    radar_sort = radar_par.copy()
    radar_sort.sort()
    pluvi_sort = pluvi_par.copy()
    pluvi_sort.sort()
    corr_sort = corr_par.copy()
    corr_sort.sort()

    plt.figure()
    plt.plot(radar_sort, cdf,color="green",label="RADAR")
    plt.plot(pluvi_sort, cdf,color="darkblue",label="Gauge")
    plt.plot(corr_sort, cdf,color="blue",label="Correcter Radar")
    plt.legend()
    plt.grid(False)
    if save and name:
        name = f"IMAGENS/cdf2/cdf-{name}.png"
        plt.savefig(name)
    plt.show();


par = pares[1]
radar = df_n.RADAR.values
pluvi = df_n.Pluviometro.values
corr_radar = df_n.CORREC_RADAR.values
for par in pares:

    cond1 = np.where(radar >= par[0])
    cond2 = np.where(radar < par[1])
    start = cond1[0][0]
    end = cond2[0][-1]
    radar_par=radar[start:end]
    pluvi_par = pluvi[start:end]
    corr_par = corr_radar[start:end]
    
    
    plot_figure(radar_par,pluvi_par,corr_par,save=True,name=str(par))
    plot_cdf(radar_par,pluvi_par,corr_par,save=True, name=str(par))

radar = df_n.RADAR.values
pluvi = df_n.Pluviometro.values
corr = df_n.CORREC_RADAR.values
plot_figure(radar,pluvi,corr,save=True,name="All")
plot_cdf(radar,pluvi,corr,save=True,name="All")

df_5 = df_n[df_n.RADAR >= 5]

df_5.head()

radar = df_5.RADAR.values
pluvi = df_5.Pluviometro.values
corr = df_5.CORREC_RADAR.values
plot_figure(radar,pluvi,corr,save=True,name="All-above5")
plot_cdf(radar,pluvi,corr,save=True,name="All-above5")

df_10 = df_n[df_n.RADAR >= 10]

df_10.head()

radar = df_10.RADAR.values
pluvi = df_10.Pluviometro.values
corr = df_10.CORREC_RADAR.values
plot_figure(radar,pluvi,corr,save=True,name="All-above10")
plot_cdf(radar,pluvi,corr,save=True,name="All-above10")

df_10.head()

df_10.to_csv("DATA/eventos-above10.csv",index=False)


