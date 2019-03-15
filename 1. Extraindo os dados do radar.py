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

# + {"colab": {"autoexec": {"startup": false, "wait_interval": 0}}, "colab_type": "code", "id": "-JzUu8rmBrV1"}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import tarfile
import os
import csv

import geopandas as gpd
from zipfile import ZipFile

# + {"colab": {"autoexec": {"startup": false, "wait_interval": 0}}, "colab_type": "code", "id": "aSrXiFPzBrWA"}
shape_file = "DATA/Estações_localizacao/Localizacao_Remotas-FINAL.shp"
shape = gpd.read_file(filename=shape_file)

# + {"colab": {"autoexec": {"startup": false, "wait_interval": 0}, "base_uri": "https://localhost:8080/", "height": 34}, "colab_type": "code", "executionInfo": {"elapsed": 23, "status": "ok", "timestamp": 1525733256330, "user": {"displayName": "Reinaldo Ossuna", "photoUrl": "//lh6.googleusercontent.com/--MvHA_gg5AQ/AAAAAAAAAAI/AAAAAAAAckk/y3HdCcNpsHs/s50-c-k-no/photo.jpg", "userId": "100120894872685494029"}, "user_tz": 240}, "id": "0w4ZwmK6BrWG", "outputId": "f951f709-f39d-4356-9e2f-94865e8b73ae"}
shape.crs

# + {"colab_type": "text", "id": "9FyYFuhYBrWF", "cell_type": "markdown"}
# ## WGS84
# - Sistema de coordenada Projetado
# - Sistema de coordenada Geodésico

# + {"colab": {"autoexec": {"startup": false, "wait_interval": 0}}, "colab_type": "code", "id": "zmQFSNKOBrWR"}
# Informações disponivel no inicio dos arquivos do radar.
cellsize = 0.005825080633
ncols = 835
nrows = 779
lat_min = -22.542179540896
lat_max = lat_min + cellsize * nrows
lon_min = -56.903502116470
lon_max = lon_min + cellsize * ncols


# + {"colab": {"autoexec": {"startup": false, "wait_interval": 0}}, "colab_type": "code", "id": "KpNdpWSTBrV6"}
def find_nearest_index(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


# + {"colab": {"autoexec": {"startup": false, "wait_interval": 0}}, "colab_type": "code", "id": "nL816NtZBrWT"}
# Obtendo o index correspondente com cada coordenada.
# 
lat = np.linspace(lat_max, lat_min, nrows, endpoint=True)
lat = np.around(lat, decimals=4)

lon = np.linspace(lon_min, lon_max, ncols, endpoint=True)
lon = np.around(lon, decimals=4)


shape['longitude'] = shape['geometry'].apply(lambda x: x.x)
shape['latitude'] = shape['geometry'].apply(lambda x: x.y)

shape['index_longitude'] = shape['longitude'].apply(lambda x: find_nearest_index(lon, x))
shape['index_latitude'] = shape['latitude'].apply(lambda x: find_nearest_index(lat, x))


# + {"colab": {"autoexec": {"startup": false, "wait_interval": 0}}, "colab_type": "code", "id": "K4UnOpwSBrWW", "outputId": "1b01f3a5-686d-4a3d-c0c6-995f6180aedb"}
shape.head()

# + {"colab": {"autoexec": {"startup": false, "wait_interval": 0}}, "colab_type": "code", "id": "4VhsieWvBrWk", "outputId": "544711f5-263d-472b-d2d1-0334379c4930"}
## Trocando o espaço nos nomes das estações por underline (_)
## MB SEG2 >> MB_SEG2

points = shape.Name.tolist()
points = list(map(lambda x: x.replace(" ", "_"), points))
shape.Name = points
shape = shape.set_index('Name')
shape.head()
# -

shape[shape.index == "MB_PRO1"]

# Pastas contendo os arquivos zips
path = 'DATA/Dados_radar_new/'
folders = [f.path for f in os.scandir(path) if f.is_dir()]
folders.sort()

# + {"colab": {"autoexec": {"startup": false, "wait_interval": 0}}, "colab_type": "code", "id": "cmeg8As1BrWr", "outputId": "89d97cf5-e36e-4109-ba87-5a8f547ee726"}
## 

from tqdm import tqdm, tqdm_notebook
import logging


logging.basicConfig(filename='DATA/extractFiles.log',level=logging.DEBUG)


with open('DATA/data.csv','w') as f_out:
    
    logging.info("Starting!\nCreating file %s",f_out)
    out_colnames = ['date']
    out_colnames += points
    
    writer = csv.DictWriter(f_out, fieldnames=out_colnames)
    writer.writeheader()
    path = 'DATA/Dados_radar_new/'
    folders = [f.path for f in os.scandir(path) if f.is_dir()]

    

    for folder in tqdm_notebook(folders,desc="Folders"):
        logging.info("Folder: %s", folder)

        zipfiles = [f.path for f in os.scandir(folder) if not f.is_dir()]
        
        
        for zipfile in tqdm_notebook(zipfiles, desc="ZipFiles", leave= False):
            logging.info("Zipfile: %s", zipfile)
            
            with ZipFile(zipfile,'r') as myzip:
                for file in tqdm_notebook(myzip.namelist(),desc="Files",leave= False):
                    logging.info("File: %s",file)
                    with myzip.open(file,'r') as data:      
                        new_point = {}
                        year   = file[:4]
                        month  = file[4:6]
                        day    = file[6:8]
                        hour   = file[8:10]
                        minute = file[10:12]
                        
                        #Dateformat %Y/%m/%d %H:%M
                        new_point['date'] = "{}/{}/{} {}:{}".format(year,month,day,hour,minute)
                        try:
                            array = np.loadtxt(data,skiprows=6)


                            if array.shape == (779,835):
                                array = np.power([10],array / 10)

                                for point in points:

                                    lat = shape.loc[point]['index_latitude']
                                    long = shape.loc[point]['index_longitude']
                                    
                                    ## Irei extrai um quadrado de 3x3 pixeis em volta
                                    ## do pixel. Isso será ultil para fazer uma média dos
                                    ## pixeis envolta do pixel desejado.
                                    
                                    lat_min = lat - 2
                                    lat_max = lat + 3 
                                    long_min = long - 2
                                    long_max = long + 3
                                    points_array = array[lat_min:lat_max,long_min:long_max]
                                    points_array = points_array.reshape(25,)
                                    points_string = np.array2string(points_array)
                                    
                                    for symbol in ['[',']']:
                                        points_string = points_string.replace(symbol,'')
                                    

                                    new_point[point] = points_string
                            else:
                                #nome dos arquivos para uma futura investigação
                                logging.warn("File wrong size: %s")

                            writer.writerow(new_point)
                        except ValueError:
                            logging.warn("File with strange data")
                        except:
                            logging.warn("SOME PROBLEM")
                        


# + {"colab": {"autoexec": {"startup": false, "wait_interval": 0}}, "colab_type": "code", "id": "YwI06663BrWv"}
df = pd.read_csv("DATA/5x5_RADAR.csv",
            parse_dates=['DATE'],
            index_col=["DATE"])
# -

df.head()


