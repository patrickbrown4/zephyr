"""
Data available:
[
    'windspeed_10m', 'windspeed_40m', 'windspeed_60m',
    'windspeed_80m', 'windspeed_100m', 'windspeed_120m', 
    'windspeed_140m', 'windspeed_160m', 'windspeed_200m',
    'pressure_0m', 'pressure_100m', 'pressure_200m',
    'temperature_10m', 'temperature_40m', 'temperature_60m', 
    'temperature_80m', 'temperature_100m', 'temperature_120m', 
    'temperature_140m', 'temperature_160m', 'temperature_200m',
    'relativehumidity_2m', 'temperature_2m',
]
Data needed for model if only using 100m height:
['windspeed_100m', 'temperature_100m', 'relativehumidity_2m', 'pressure_100m']
"""

###############
### IMPORTS ###

import sys, os, site, time
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import h5pyd
import dateutil
from urllib.error import HTTPError

import pickle, gzip

import zephyr
extdatapath = zephyr.settings.extdatapath

#######################
### ARGUMENT INPUTS ###

import argparse
parser = argparse.ArgumentParser(description='Download WTK data')
parser.add_argument('datum', type=str, default='windspeed_100m')
### Parse argument inputs
args = parser.parse_args()
datum = args.datum

##########
### INPUTS

skip = 2
start = 0

outpath = os.path.join(extdatapath,'WTK-HSDS','every{}-offset{}','{}').format(
    skip, start, datum)

numattempts = 200
sleeptime = 10

#############
### PROCEDURE

### Make the output folder
os.makedirs(outpath, exist_ok=True)

### Set up data access
hsds = h5pyd.File("/nrel/wtk-us.h5", 'r')

### Get the datetime index
dt = hsds["datetime"]
dt = pd.Series(dt[:])
dt = dt.apply(dateutil.parser.parse)

### Get the datum
dset = hsds[datum]

### Loop over timestamps and download spatial slices
for i in tqdm(dt.index, desc=datum):
    ### Save it as the timestamp
    savename = dt.loc[i].strftime('%Y%m%dT%H%M')
    filename = os.path.join(outpath,savename+'.npy.gz')

    if not os.path.exists(filename):
        ### Retry if server errors
        attempts = 0
        while attempts < numattempts:
            try:
                ### Access every point
                arrayout = dset[i,start::skip,start::skip]
                break
            except (OSError, HTTPError) as err:
                print('Rebuffed on attempt # {} by "{}". Retry in {} seconds.'.format(
                    attempts, err, sleeptime))
                attempts += 1
                time.sleep(sleeptime)
        
        ### Save it
        with gzip.open(filename, 'wb') as p:
            np.save(p, arrayout)

    else:
        pass
