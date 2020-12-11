###############
### IMPORTS ###

import pandas as pd
import numpy as np
import os, sys, site, math, time, pickle, gzip
from glob import glob
from tqdm import tqdm, trange

import geopandas as gpd
import xarray as xr

import zephyr
projpath = zephyr.settings.projpath
datapath = zephyr.settings.datapath
extdatapath = zephyr.settings.extdatapath

##############
### INPUTS ###

years = list(range(1998,2019))

outcols = ['windspeed2M','windspeed10M','windspeed50M','T10M','PS','QV10M','DISPH']
overwrite = False

inpath = os.path.join(extdatapath,'MERRA2','M2T1NXSLV','')
outpath = os.path.join(extdatapath,'MERRA2','M2T1NXSLV-usa1degbuff','')

#################
### PROCEDURE ###
print(years)

### Load a file to use for map
date = '20100101'
infile = inpath+'MERRA2_{}00.tavg1_2d_slv_Nx.{}.nc4.nc'.format(
    4 if date[:4] >= '2011' else 3 if date[:4] >= '2001' else 2 if date[:4] >= '1991' else 1,
    date
)
data = xr.open_dataset(infile)
df = data.to_dataframe()
dfmap = df.reset_index()[['lat','lon']].drop_duplicates().reset_index(drop=True)
df['windspeed'] = np.sqrt(df.V50M**2 + df.U50M**2)


###### Get sites within buffer of US
dfzones = gpd.read_file(os.path.join(projpath,'in','Maps','state'))

dfzones['dummy'] = 'USA'
dfusa = dfzones.dissolve('dummy').buffer(1.)
polyusa = dfusa.loc['USA']
dfusa = gpd.GeoDataFrame(geometry=[polyusa])
dfusa['usa'] = ['usa']

inusa = zephyr.toolbox.pointpolymap(
    dfmap, dfusa, x='lon', y='lat', zone='usa', progressbar=False)
dfmap['usa'] = [True if x == 'usa' else False for x in inusa ]

dfmap['state'] = zephyr.toolbox.pointpolymap(
    dfmap, dfzones, x='lon', y='lat', zone='state', progressbar=False)
dfmap['state'] = dfmap['state'].replace('',False)

point2state = pd.Series(
    data=dfmap.state.values,
    index=[tuple(x) for x in dfmap[['lat','lon']].values.tolist()],
    name='state'
)

usapoints = dfmap.loc[dfmap.usa][['lat','lon']].values
usapoints = [tuple(x) for x in usapoints.tolist()]

####### Do it all
for year in years:
    savename = '{}-{}.df.p.gz'.format(','.join(outcols),year)
    if os.path.exists(outpath+savename) and (overwrite is False):
        continue
    dates = pd.date_range('{}-01-01'.format(year),'{}-12-31'.format(year), freq='d')
    indates = [s.strftime('%Y%m%d') for s in dates]
    ### Load it
    dictout = {col:{} for col in outcols}
    for i, date in enumerate(tqdm(indates, desc=str(year))):
        infile = inpath+'MERRA2_{}00.tavg1_2d_slv_Nx.{}.nc4.nc'.format(
            4 if year >= 2011 else 3 if year >= 2001 else 2 if year >= 1992 else 1,
            date
        )
        data = xr.open_dataset(infile)
        df = data.to_dataframe()
        df['windspeed2M'] = np.sqrt(df.V2M**2 + df.U2M**2)
        df['windspeed10M'] = np.sqrt(df.V10M**2 + df.U10M**2)
        df['windspeed50M'] = np.sqrt(df.V50M**2 + df.U50M**2)
        for col in outcols:
            dictout[col][dates[i]] = pd.concat(
                [df.loc[point,col].rename(point) for point in usapoints], axis=1)
    ### Store it
    dfout = pd.concat(
        {col: pd.concat([dictout[col][date] for date in dates],axis=0)for col in outcols},
        axis=1)
    ### Save it
    with gzip.open(outpath+savename,'wb') as p:
        pickle.dump(dfout, p)

