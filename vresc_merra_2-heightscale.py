"""
* prbrown 20200323 13:00
"""

###############
### IMPORTS ###
import pandas as pd
import numpy as np
import os, sys, site, math, time, pickle, gzip
from glob import glob
from tqdm import tqdm, trange

import scipy, scipy.stats

import zephyr
projpath = zephyr.settings.projpath
datapath = zephyr.settings.datapath
extdatapath = zephyr.settings.extdatapath

#######################
### ARGUMENT INPUTS ###

import argparse
parser = argparse.ArgumentParser(description="MERRA2 height extrapolation")
parser.add_argument('year',help='year to run',type=int)
parser.add_argument('-z','--height',help='output height [m]',type=int,default=100)
parser.add_argument('-e','--runend',help='number of sites',type=str,default='all')
parser.add_argument('-v', '--verbose', action='count', default=0)
parser.add_argument('-o', '--overwrite', action='store_true')
parser.add_argument('-w', '--writeout', default='df', choices=['df','dict'])
args = parser.parse_args()
year = args.year
height = args.height
runend = args.runend
verbose = args.verbose
overwrite = args.overwrite
writeout = args.writeout
if args.runend in ['full','all','default','None']:
    runend = None
else:
    runend = int(runend)

##############
### INPUTS ###

inpath = datapath+'NASA/MERRA2/M2T1NXSLV-usa1degbuff/'
incols = ['windspeed2M','windspeed10M','windspeed50M','T10M','PS','QV10M','DISPH']

# outpath = os.path.expanduser('~/Desktop/test/')
outpath = datapath+'NASA/MERRA2/M2T1NXSLV-usa1degbuff-{}m/all/'.format(height)
savename = 'allfields-{}m-{}.{}.p.gz'.format(height,year,writeout)

#################
### FUNCTIONS ###

def specific_humidity_to_vapor_pressure(q, pressure,
    R_dryair=287.058, R_wv=461.495):
    """
    Inputs
    ------
    q: specific humidity in [kg/kg]
    pressure: total air pressure in Pa
    R_dryair, R_wv: [J kg^-1 K^-1]

    ('https://earthscience.stackexchange.com/questions/2360/'
     'how-do-i-convert-specific-humidity-to-relative-humidity')
    """
    return q * pressure * R_dryair / R_wv

def density_specifichumidity(
    pressure, temperature, specifichumidity=0, 
    R_dryair=287.058, R_wv=461.495):
    """
    """
    p_wv_val = specific_humidity_to_vapor_pressure(
        specifichumidity, pressure)
    density_total = (
        ((pressure - p_wv_val) / R_dryair / temperature)
        + (p_wv_val / R_wv / temperature))
    return density_total

def pressure_height_extrapolate(
    pressure, temperature, specifichumidity, height=100,
    R_dryair=287.058, R_wv=461.495, g=9.81):
    """
    """
    ### Calculate gas constant for mixed air & water vapor
    R_mix = R_wv * specifichumidity + R_dryair / (1 + specifichumidity)
    ### Extrapolate to new height
    pressure_out = pressure * np.exp(-height * g / R_mix / temperature)
    return pressure_out

def windspeed_standardize(windspeed, density, density_std=1.225):
    """
    """
    out = windspeed * ((density / density_std)**(1/3))
    return out

#################
### PROCEDURE ###

### Check savename, create output folder
os.makedirs(outpath, exist_ok=True)
print(outpath+savename)
sys.stdout.flush()
if os.path.exists(outpath+savename) and (overwrite == False):
    quit()

### Load input file
infile = '{}-{}.df.p.gz'.format(','.join(incols),year)
dfin = (pd.read_pickle(inpath+infile)
        .reorder_levels([1,2,0],axis=1)
        .sort_index(axis=1,level=[0,1,2]))

### Loop over all sites
latlons = dfin.xs('PS',1,2).T.reset_index()[['level_0','level_1']].values.tolist()
dictout = {}
for latlon in tqdm(latlons[:runend], desc=str(year)):
    lat,lon = latlon
    dfsite = dfin[(lat,lon)].copy()
    x0s, x1s, windspeedheights, pressureheights, densityheights, windspeed_stds, r2s = (
        [],[],[],[],[],[],[])
    if verbose < 1:
        iterator = dfsite.index
    else:
        iterator = tqdm(dfsite.index, desc='{},{}'.format(lat,lon), leave=False)

    for index in iterator:
        ### Get the row variables
        row = dfsite.loc[index]
        T = row['T10M']
        q = row['QV10M']
        d = row['DISPH']
        ### Calculate the heights for fitting
        zd_2m = 2
        zd_10m = 10
        zd_50m = 50 - d
        ### Get the fit coefficients
        results = scipy.stats.linregress(
            #np.log(row[['zd_2M','zd_10M','zd_50M']].values),
            np.log(np.array([zd_2m,zd_10m,zd_50m])),
            row[['windspeed2M','windspeed10M','windspeed50M']].values,
        )
        x0, x1, r2 = results.intercept, results.slope, results.rvalue**2
        ### Extrapolate windspeed to height based on fit coefficients
        windspeedheight = x0 + x1 * np.log(height - d)
        ### Calculate pressure at height
        pressureheight = pressure_height_extrapolate(
            pressure=row['PS'], temperature=T, specifichumidity=q,
            height=height)
        ### Calculate density at height
        densityheight = density_specifichumidity(
            pressure=pressureheight, temperature=T, specifichumidity=q)
        ### Scale windspeed to standard test conditions
        windspeed_std = windspeed_standardize(
            windspeed=windspeedheight, density=densityheight)
        ### Store the results
        x0s.append(x0)
        x1s.append(x1)
        r2s.append(r2)
        windspeedheights.append(windspeedheight)
        pressureheights.append(pressureheight)
        densityheights.append(densityheight)
        windspeed_stds.append(windspeed_std)

    dfsite['x0'] = x0s
    dfsite['x1'] = x1s
    dfsite['r2'] = r2s
    dfsite['windspeedheight'] = windspeedheights
    dfsite['pressureheight'] = pressureheights
    dfsite['densityheight'] = densityheights
    dfsite['windspeed_std'] = windspeed_stds
    
    dictout[tuple(latlon)] = dfsite

### Save it
if writeout == 'dict':
    with gzip.open(outpath+savename,'wb') as p:
        pickle.dump(dictout, p)
else:
    dfout = pd.concat(dictout, axis=1, copy=False)
    with gzip.open(outpath+savename,'wb') as p:
        pickle.dump(dfout, p)
