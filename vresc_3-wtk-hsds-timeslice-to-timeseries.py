###############
### IMPORTS ###

import sys, os, site, time, gzip, pickle, cmocean
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from glob import glob

import zephyr

### Common paths
projpath = zephyr.settings.projpath
datapath = zephyr.settings.datapath
extdatapath = zephyr.settings.extdatapath

##############
### INPUTS ###

# datum = 'windspeed_100m'
# datum = 'pressure_100m'
# datum = 'temperature_100m'
datum = 'relativehumidity_2m'
# datum = 'windspeed_140m'
# datum = 'pressure_200m'
# datum = 'pressure_0m'
# datum = 'temperature_140m'

hsdspoint = 'hsds_maxwindspeed100m_closest' ### OR 'hsds_closest'

inpath = os.path.join(extdatapath,'in','WTK-HSDS','every2-offset0','{}',).format(datum)
outpath = os.path.join(
    extdatapath,'in','WTK-HSDS','every2-offset0','timeseries','icomesh-maxwindspeed100m-onshore')
    ### OR icomesh-closest-onshore

#################
### PROCEDURE ###

os.makedirs(outpath, exist_ok=True)

savename = os.path.join(outpath,'{}.df.p').format(datum)
print(savename)

### Load icomesh points
dfpoints = pd.read_csv(os.path.join(
    projpath,'io',
    'icomesh-nsrdb-info-key-psmv3-eGRID-avert-ico9-wtkhsds_closest,maxspeed100m.csv'))

### Shared parameters
timeseries = pd.date_range(
    '2007-01-01 00:00','2014-01-01 00:00',
    freq='H', closed='left', tz='UTC')

# lookup = (dfpoints.wtk_row.values, dfpoints.wtk_col.values)
lookup = (
    dfpoints[hsdspoint].map(
        lambda x: int(x.split('_')[0]) // 2),
    dfpoints[hsdspoint].map(
        lambda x: int(x.split('_')[1]) // 2)
)

# sites = dfpoints.apply(
#     lambda row: '{}_{}'.format(row['wtk_rowfull'],row['wtk_colfull']),
#     axis=1
# ).values
sites = dfpoints[hsdspoint].values

### Load all files and store in dict
dictout = {}
for timestamp in tqdm(timeseries):
    timestring = timestamp.strftime('%Y%m%dT%H%M')

    ### Load the timeslice
    filename = os.path.join('{}','{}.npy.gz').format(
        inpath, timestring)
    with gzip.open(filename, 'rb') as p:
        array = np.load(p)

    ### Store in dictionary
    dictout[timestring] = array[lookup]

### Turn into dataframe
dfout = pd.DataFrame(dictout, index=sites).T
dfout.index = timeseries

### Save it without compression
with open(savename, 'wb') as p:
    pickle.dump(dfout, p, protocol=4)
