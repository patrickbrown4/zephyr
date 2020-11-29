###############
### IMPORTS ###

import sys, os, site, time, gzip, pickle, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from glob import glob

import h5pyd
from pyproj import Proj
import dateutil

import zephyr
projpath = zephyr.settings.projpath
datapath = zephyr.settings.datapath
extdatapath = zephyr.settings.extdatapath

##############
### INPUTS ###

parser = argparse.ArgumentParser(description='Resave WTK-HSDS as single-site')
parser.add_argument(
    '-a', '--start', help='integer start point', type=int)
parser.add_argument(
    '-s', '--step', help='integer step', type=int, default=20000)
parser.add_argument(
    '-c', '--savecsv', action='store_true', 
    help='flag to save as csv instead of pickled dataframe')

args = parser.parse_args()
start = args.start
step = args.step
savecsv = args.savecsv

data = [
    'relativehumidity_2m', 'temperature_100m', 'windspeed_100m', 'pressure_100m',
    # 'pressure_0m', 'pressure_200m', 'temperature_140m', 'windspeed_140m',
]

timeseries = pd.date_range(
    '2007-01-01 00:00','2014-01-01 00:00', freq='H', closed='left', tz='UTC')
winddatapath = os.path.join(extdatapath,'WTK-HSDS','every2-offset0','')
outpath = os.path.join(winddatapath,'timeseries','usa-halfdegbuff','')

#################
### PROCEDURE ###

print(start, step)
os.makedirs(outpath, exist_ok=True)

### Get input points
dfusa = pd.read_csv(
    os.path.join(projpath,'io','geo','hsdscoords-usahifld-bufferhalfdeg.csv.gz'),
    # os.path.join(projpath,'io','geo','hsdscoords-offshore200nm-toprocess.csv.gz'),
    dtype={'row':int,'col':int,
           'latitude':float,'longitude':float})
dfusa['rowf_colf'] = (
    dfusa.row_full.astype(str)+'_'+dfusa.col_full.astype(str))

### Get lookup
dfsites = dfusa.iloc[start:start+step].copy()
lookup = (dfsites.row.values, dfsites.col.values)
rowf_colfs = dfsites.rowf_colf.values
if len(dfsites) == 0:
    quit()

###### Load all weather files and store in dict
### 21.36 GB for 10,000 sites
dictwind = {}
for datum in data:
    inpath = os.path.join(winddatapath,datum,'')
    dictout = {}
    for timestamp in tqdm(timeseries, desc=datum):
        timestring = timestamp.strftime('%Y%m%dT%H%M')

        ### Load the timeslice
        filename = '{}{}.npy.gz'.format(inpath, timestring)
        with gzip.open(filename, 'rb') as p:
            array = np.load(p)
        ### Pull out the values for the previously-identified points
        arrayout = array[lookup]
        ### Store in dictionary
        dictout[timestring] = arrayout

    ### Turn into dataframe
    dfout = pd.DataFrame(dictout, index=rowf_colfs).T
    dfout.index = timeseries
    dictwind[datum] = dfout

for rowf_colf in tqdm(rowf_colfs, desc='CF'):
    dfwind = pd.concat(
        [dictwind[datum][rowf_colf].rename(datum) for datum in data], axis=1)
    if savecsv:
        savename = '{}.csv.gz'.format(rowf_colf)
        dfwind.to_csv(outpath+savename, compression='gzip')
    else:
        savename = '{}.df.p.gz'.format(rowf_colf)
        dfwind.to_pickle(outpath+savename, compression='gzip')
