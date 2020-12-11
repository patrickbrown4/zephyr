###############
#%% IMPORTS ###

import pandas as pd
import numpy as np
import os, sys, site, math, time, pickle, gzip
from glob import glob
from tqdm import tqdm, trange

import zephyr
projpath = zephyr.settings.projpath
datapath = zephyr.settings.datapath
extdatapath = zephyr.settings.extdatapath

##############
#%% INPUTS ###

height = 100
inpath = os.path.join(
    datapath,'NASA','MERRA2','M2T1NXSLV-usa1degbuff-{}m'.format(height),
    'windspeed_std,T10M',''
)
years = list(range(1997,2020))
model = 'Gamesa:G126/2500_low'
pc_vmax = 60
orders = 4
loss_system = 0.15
hystwindow = 5
temp_cutoff = 243.15
nans = 'raise'

#################
#%% PROCEDURE ###

#%%### Load height-extrapolated MERRA data
dictin = {}
for year in tqdm(years):
    with gzip.open(inpath+'{}.df.p.gz'.format(year),'rb') as p:
        dictin[year] = pickle.load(p)

dfin = pd.concat(dictin, axis=0, copy=False)
dfin.reset_index(level=0,drop=True,inplace=True)

#%%### Get power curve
### Get powercurve source
if model.lower().startswith('wtk'):
    powercurvesource = 'wtk'
elif model in ['Gamesa:G126/2500','Vestas:V110/2000']:
    powercurvesource = 'twp'
else:
    powercurvesource = 'literature'

### Load power curve dictionary
dictcf, cutout = zephyr.wind.get_powercurve_dict(
    source=powercurvesource, model=model, pc_vmax=pc_vmax)

#%%### Loop over all sites
latlons = [(x[0],x[1]) for x in dfin.columns.values]
dictout = {}
for latlon in tqdm(latlons):
    lat,lon = latlon
    df = dfin[lat][lon].copy()
    
    ### Set negative windspeeds to zero
    df.windspeed_std.clip(0, inplace=True)
    
    ### Look up capacity factor from power curve
    df['cf'] = np.around(df.windspeed_std.astype(float), orders).map(dictcf)

    ### Correct for hysteresis (assuming cut-offs apply to corrected wind speed)
    th_hi = cutout
    th_lo = cutout - hystwindow
    df['cf_hyst'] = (
        df.cf * (
            ~hyst(x=df['windspeed_std'].values, 
                  th_lo=th_lo, th_hi=th_hi, initial=False)
        )
    )

    ### Temperature cutoff
    if temp_cutoff is not None:
        df.loc[df['T10M']<=temp_cutoff, 'cf_hyst'] = 0

    ### Check for nan's and raise exception if found
    if nans == 'raise':
        if any(df.cf_hyst.isnull()):
            print('{}: {} nans'.format(latlon, df.cf.isnull().sum()))
            print(df.loc[df.cf.isnull()])
            raise Exception("nan's in df.cf")
    elif nans.startswith('inter') or nans.startswith('time'):
        df = df.fillna('time')
    elif nans.isin(['ignore','pass','silent']):
        pass
    
    ### Store it
    dictout[latlon] = df.cf_hyst

dfout = pd.concat(dictout, axis=1, copy=False)

#%% Write it
modelsave = model.replace(':','|').replace('/','_')
outpath = os.path.join(datapath,'NASA','MERRA2','M2T1NXSLV-usa1degbuff-{}m'.format(height),'')
outfile = os.path.join(outpath,'CF-0loss-{}.df.p.gz'.format(modelsave))
with gzip.open(outfile, 'wb') as p:
    pickle.dump(dfout, p, protocol=4)
print(outfile)
