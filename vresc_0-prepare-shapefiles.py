###############
#%% IMPORTS ###
import numpy as np
import pandas as pd
import os, sys, math
import geopandas as gpd
import shapely
import urllib.request, zipfile, tarfile, pickle
from tqdm import tqdm, trange
import h5pyd

### Only need zephyr for projpath
import zephyr
projpath = zephyr.settings.projpath
datapath = zephyr.settings.datapath

#################
### PROCEDURE ###

################################################
#%%### Download shapefiles from external sources
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
    (
        os.path.join('in','Maps','NPS',''),
        'nps_boundary',
        'https://irma.nps.gov/DataStore/DownloadFile/647685'
    ),
]

### Loop over files
for (filepath, filename, url) in files:
    ### Download it
    os.makedirs(filepath, exist_ok=True)
    if '.tar.gz' in url:
        suffix = '.tar.gz'
    elif '.gz' in url:
        suffix = '.gz'
    else:
        suffix = '.zip'
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

##################################
#%%### Make the US states zone map
usafile = os.path.join(projpath,'in','Maps','HIFLD','Political_Boundaries_(Area)-shp')
dfzones = gpd.read_file(usafile).to_crs({'init':'epsg:4326'})
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
os.makedirs(os.path.join(projpath,'in','Maps','state'), exist_ok=True)
dfzones.to_file(os.path.join(projpath,'in','Maps','state'))

######################################################
#%%### Prepare the national parks polygon to save time
outfile = os.path.join(projpath,'in','Maps','NPS','nationalparks-all-poly.p')
if not os.path.exists(outfile):
    parks = gpd.read_file(datapath+'Maps/NPS/nps_boundary/')
    allparks = parks.loc[~parks.STATE.isin(['AK','HI','GU','MP','PR','AS','VI'])].copy()
    allparks['dummy'] = 0
    allparks = allparks.dissolve('dummy')
    polyallparks = allparks.loc[0, 'geometry']
    ### Save it
    with open(outfile, 'wb') as p:
        pickle.dump(polyallparks, p)

##############################################
#%%### Create the list of WTK HSDS coordinates

### Set up data access
hsds = h5pyd.File("/nrel/wtk-us.h5", 'r')

### Get the coordinates
windcoords = hsds['coordinates']
windcoords_data = windcoords[:][:]

### Reshape and reformat
nrows = windcoords_data.shape[0]
ncols = windcoords_data.shape[1]

rowsout_lat, rowsout_lon = {}, {}
for row in trange(nrows):
    rowsout_lat[row] = [x[0] for x in windcoords_data[row]]
    rowsout_lon[row] = [x[1] for x in windcoords_data[row]]

latsout = pd.DataFrame(rowsout_lat).T
lonsout = pd.DataFrame(rowsout_lon).T

latsout['row'] = latsout.index
lonsout['row'] = lonsout.index

dfout = (
    pd.melt(latsout, var_name='col', value_name='latitude', id_vars='row')
    .merge(
        pd.melt(lonsout, var_name='col', value_name='longitude', id_vars='row'),
        on=['row','col'])
)

### Save it
dfout.to_csv(
    os.path.join(projpath,'io','geo','hsdscoords.gz'), 
    compression='gzip', index=False)

#######################################################################
#%%### Create the list of WTK HSDS points within 0.5° of continental US
### Get the USA states map
usafile = os.path.join(projpath,'in','Maps','state')
dfstates = gpd.read_file(usafile)
### Dissolve to a single polygon
dfstates['dummy'] = 0
dfusa = dfstates.dissolve('dummy')
dfusa.crs = {'init':'epsg:4326'}
### Buffer it to 0.5°
usa = dfusa.buffer(0.5).loc[0]

### Load the WTK points
dfcoords = pd.read_csv(
    os.path.join(projpath,'io','geo','hsdscoords.gz'),
    dtype={'row':int,'col':int,
           'latitude':float,'longitude':float})

### Get coordinates of downsampled WTK points
wtkpoints = dfcoords.loc[(dfcoords.row % 2 == 0) & (dfcoords.col % 2 == 0)].copy()
wtkpoints.rename(columns={'row':'row_full','col':'col_full'}, inplace=True)
wtkpoints['row'] = (wtkpoints['row_full'] / 2).astype(int)
wtkpoints['col'] = (wtkpoints['col_full'] / 2).astype(int)

### Determine whether each point is inside buffered US polygon
wtkpoints['usa'] = zephyr.toolbox.pointpolymap(
    wtkpoints, usa, x='longitude',y='latitude',resetindex=False)

### Save it
wtkpoints.loc[wtkpoints.usa==True].drop('usa',axis=1).to_csv(
    os.path.join(projpath,'io','geo','hsdscoords-usahifld-bufferhalfdeg.csv.gz'),
    index=False, compression='gzip',
)
