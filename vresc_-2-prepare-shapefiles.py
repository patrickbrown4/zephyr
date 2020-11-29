###############
### IMPORTS ###
import numpy as np
import pandas as pd
import os, sys, math
import geopandas as gpd
import shapely
import urllib.request, zipfile, tarfile
from tqdm import tqdm, trange

### Only need zephyr for projpath
import zephyr
projpath = zephyr.settings.projpath
datapath = zephyr.settings.datapath

#################
### PROCEDURE ###

###### Download shapefiles from external sources
files = [
    ### (filepath, filename, url)
    (
        os.path.join(projpath,'in','Maps','Census',''),
        'cb_2016_us_nation_5m',
        'https://www2.census.gov/geo/tiger/GENZ2016/shp/cb_2016_us_nation_5m.zip'
    ),
    (
        os.path.join(projpath,'in','Maps','Census',''),
        'tl_2017_us_aitsn',
        'https://www2.census.gov/geo/tiger/TIGER2017/AITSN/tl_2017_us_aitsn.zip'
    ),
    (
        os.path.join(projpath,'in','Maps','Census',''),
        'tl_2010_us_uac10',
        'https://www2.census.gov/geo/tiger/TIGER2010/UA/2010/tl_2010_us_uac10.zip'
    ),
    (
        os.path.join(projpath,'in','Maps','HIFLD',''),
        'Political_Boundaries_(Area)-shp',
        ('https://opendata.arcgis.com/datasets/bee7adfd918e4393995f64e155a1bbdf_0.zip?'
        'outSR=%7B%22wkid%22%3A102100%2C%22latestWkid%22%3A3857%7D')
    ),
    (
        os.path.join(projpath,'in','Maps','HIFLD',''),
        'Electric_Power_Transmission_Lines-shp',
        ('https://opendata.arcgis.com/datasets/70512b03fe994c6393107cc9946e5c22_0.zip?'
         'outSR=%7B%22latestWkid%22%3A3857%2C%22wkid%22%3A102100%7D')
    ),
    (
        os.path.join(projpath,'in','Maps','USGS',''),
        'wtrbdyp010g.shp_nt00886',
        ('https://prd-tnm.s3.amazonaws.com/StagedProducts/Small-scale/data/Hydrography/'
         'wtrbdyp010g.shp_nt00886.tar.gz')
    ),
]

### Loop over files
for (filepath, filename, url) in files:
    ### Download it
    os.makedirs(filepath, exist_ok=True)
    if '.zip' in url:
        suffix = '.zip'
    elif '.tar.gz' in url:
        suffix = '.tar.gz'
    elif '.gz' in url:
        suffix = '.gz'
    urllib.request.urlretrieve(url, os.path.join(filepath,filename+suffix))
    ### Unzip it
    if '.tar' in suffix:
        tar = tarfile.open(os.path.join(filepath,filename+suffix),'r:gz')
        tar.extractall(os.path.join(filepath, filename))
        tar.close()
    else:
        zip_ref = zipfile.ZipFile(os.path.join(filepath,filename+suffix), 'r')
        zip_ref.extractall(os.path.join(filepath, filename))
        zip_ref.close()
    print('Downloaded {}'.format(os.path.join(filepath,filename)))

###### Make the US states zone map
usafile = os.path.join(projpath,'in','Maps','HIFLD','Political_Boundaries_(Area)-shp')
dfzones = gpd.read_file(usafile)
### Downselect to states
dfzones = dfzones.loc[
    (dfzones.COUNTRY=='USA')
    & ~(dfzones.NAME.isin(
        ["water/agua/d'eau", "Puerto Rico", 
         "Alaska", "Hawaii",
         "United States Virgin Islands", "Navassa Island",
        ])),
    ['STATEABB','geometry']
].dissolve('STATEABB')

dfzones = dfzones.reset_index().rename(columns={'STATEABB':'state'})
dfzones.state = dfzones.state.map(lambda x: x.replace('US-',''))
### Save it
os.makedirs(os.path.join(projpath,'in','Maps','states'), exist_ok=True)
dfzones.to_file(os.path.join(projpath,'in','Maps','states'))

###### Prepare the national parks polygon to save time
