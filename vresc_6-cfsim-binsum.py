###############
### IMPORTS ###
import pandas as pd
import numpy as np
import os, sys, site, pickle, json, shapely, zipfile, gzip
from glob import glob
from tqdm import tqdm, trange
import geopandas as gpd

import zephyr

### Common paths
projpath = zephyr.settings.projpath
datapath = zephyr.settings.datapath
extdatapath = zephyr.settings.extdatapath

######################
### ARGUMENT INPUTS ###

import argparse
parser = argparse.ArgumentParser(description='voronoi-poly overlap')

parser.add_argument(
    'zone', help='zone name (leave out level such as state or eGRID)', type=str)
parser.add_argument(
    '-x', '--level', help='level (e.g. rto, region)', type=str, default='state')
parser.add_argument(
    '-r', '--resource', help="'wind' or 'pv'", type=str, default='wind')
# parser.add_argument(
#     '-g', '--gencostannual', type=float,
#     help='annuitized generator cost in $/kWac-yr; include FOM')
parser.add_argument(
    '-b', '--maxbreaks', default=10, type=int, help="Maximum number of breaks")
parser.add_argument(
    '-s', '--systemtype', help="systemtype for PV, in ['fixed, track']",
    default='track', type=str, choices=['fixed', 'track'])
parser.add_argument(
    '-t', '--axistilt', default='default',
    help="axistilt, in [float, 'latitude', 'default', 'optrev', 'optcf']",)
parser.add_argument(
    '-z', '--axisazimuth', default=180,
    help="axisazimuth, in [float, 'default', 'optrev', 'optcf']",)
parser.add_argument(
    '-e', '--height', default=100, type=int,
    help="wind turbine height in meters, default 100",)
parser.add_argument(
    '-m', '--model', default='Gamesa:G126/2500', type=str,
    help="wind turbine model, default 'Gamesa:G126/2500'",)
parser.add_argument(
    '-u', '--urban', default='edge', type=str, choices=['edge','centroid'],
    help="Urban endpoint: edge (any urban area) or centroid (all urban areas)")
parser.add_argument(
    '-c', '--tempcutoff', default='243.15', type=str,
    help='low-temperature wind cutoff. Default 243.15 (-30°C); can also be None')
parser.add_argument(
    '-i', '--transcostmultiplier', default=1, type=int,
    help='Multiplier for NREL interconnection costs')


### Parse argument inputs
args = parser.parse_args()
zone = args.zone
level = args.level
resource = args.resource
# gencostannual = args.gencostannual
height = args.height
model = args.model
modelsave = model.replace(':','|').replace('/','_')
maxbreaks = args.maxbreaks
urban = args.urban
transcostmultiplier = args.transcostmultiplier
if resource in ['wind','wtk','wtk-hsds','wtk-hsds-every2']:
    resource = 'wind'
elif resource in ['solar','pv','nsrdb','icomesh','icomesh9','psm3','nsrdb-icomesh9']:
    resource = 'pv'
else:
    exc = "Invalid resource: {}. Try 'wind' or 'pv'.".format(resource)
    raise Exception(exc)
systemtype = args.systemtype
## axistilt
try:
    axis_tilt = float(args.axistilt)
except ValueError:
    axis_tilt = str(args.axistilt)
if (axis_tilt == 'default') and (systemtype == 'fixed'):
    axis_tilt = 'latitude'
elif (axis_tilt == 'default') and (systemtype == 'track'):
    axis_tilt = 0
elif axis_tilt == 'None':
    axis_tilt = None
## axisazimuth
try:
    axis_azimuth = float(args.axisazimuth)
except ValueError:
    axis_azimuth = str(args.axisazimuth)
if axis_azimuth in ['default','None']:
    axis_azimuth = 180
## tempcutoff
try:
    temp_cutoff = float(args.tempcutoff)
except ValueError:
    temp_cutoff = str(args.tempcutoff)
if temp_cutoff in ['default','None','none','0','0.']:
    temp_cutoff = None

###### Parse the zone (for use on supercloud)
### States
if level == 'state':
    zonenumber2zone = dict(zip(
        range(48),
        ['AL', 'AZ', 'AR', 'CA', 'CO',  ### 0-4
         'CT', 'DE', 'FL', 'GA', 'ID',  ### 5-9
         'IL', 'IN', 'IA', 'KS', 'KY',  ### 10-14
         'LA', 'ME', 'MD', 'MA', 'MI',  ### 15-19
         'MN', 'MS', 'MO', 'MT', 'NE',  ### 20-24
         'NV', 'NH', 'NJ', 'NM', 'NY',  ### 25-29
         'NC', 'ND', 'OH', 'OK', 'OR',  ### 30-34
         'PA', 'RI', 'SC', 'SD', 'TN',  ### 35-39
         'TX', 'UT', 'VT', 'VA', 'WA',  ### 40-44
         'WV', 'WI', 'WY',]             ### 45-47
        ))
elif level == 'ba':
    zonenumber2zone = dict(zip(
        range(11),
        ['California','Northwest','Mountain','Texas','Central',
         'MidwestN','MidwestS','Southeast','Florida','MidAtlantic','Northeast']
    ))

try:
    zonenumber = int(zone)
    zone = zonenumber2zone[zonenumber]
except ValueError:
    pass

##############
### INPUTS ###

### Generator cost (annualized capex + FOM)
infile = os.path.join(projpath,'io','generator_fuel_assumptions.xlsx')
defaults = zephyr.cpm.Defaults(infile)
pvcost = 'PV_track_2030_mid'
windcost = 'Wind_2030_mid'
if resource == 'pv':
    gen = zephyr.cpm.Gen(pvcost, defaults=defaults)
elif resource == 'wind':
    gen = zephyr.cpm.Gen(windcost, defaults=defaults)
gencostannual = gen.cost_annual + gen.cost_fom

runtype = 'full'

### Wind system losses
loss_system_wind = 0.19 ### Owen Roberts NREL 20200331

### Set output file location and savenames
outpath = os.path.join(
    projpath,
    'io','cf-2007_2013','{}x','{}','{}','{}','binned',''
).format(
    transcostmultiplier,level,resource,
    modelsave if (resource == 'wind') else (
        '{}-{}t-{:.0f}az'.format(systemtype,axis_tilt,axis_azimuth))
)

###### Select wind/PV-specific parameters
if resource == 'pv':
    ### Set PV parameters
    dcac = 1.3

    index_coords = 'psm3id'
    resourcedatapath = os.path.join(regeodatapath,'in','NSRDB','ico9','')

    filename_dfsites = outpath.replace('binned'+os.sep,'')+(
        'mean-nsrdb,icomesh9-{}-{}t-{}az-{:.2f}dcac-{:.0f}USDperkWacyr-{}_{}.csv'
    ).format(
        systemtype, 
        ('{:.0f}'.format(axis_tilt) if type(axis_tilt) in [float,int] 
            else axis_tilt), 
        ('{:.0f}'.format(axis_azimuth) if type(axis_azimuth) in [float,int] 
            else axis_azimuth), 
        dcac, gencostannual, level, zone)
    
elif resource == 'wind':
    ### Set wind parameters
    number_of_turbines = 1
    pc_vmax = 60
    nans = 'raise'

    index_coords = 'rowf_colf'
    resourcedatapath = os.path.join(
        regeodatapath,'in','WTK-HSDS','every2-offset0','timeseries','usa-halfdegbuff','')

    filename_dfsites = (
        outpath.replace('binned'+os.sep,'')
        + ('mean-wtkhsds,every2,offset0,onshore-{}-{}m-{:.0f}pctloss-{:.0f}USDperkWacyr-{}_{}.csv'
        ).format(modelsave, height, loss_system_wind*100, gencostannual, level, zone)
    )

    if model.lower().startswith('wtk'):
        powercurvesource = 'wtk'
    elif model in ['Gamesa:G126/2500','Vestas:V110/2000']:
        powercurvesource = 'twp'
    else:
        powercurvesource = 'literature'

### Specify timeseries to simulate over
if runtype == 'full':
    years = list(range(2007,2014))
elif runtype == 'test':
    years = [2010]

#################
### PROCEDURE ###
#################

### Set up output folders and filenames
os.makedirs(outpath, exist_ok=True)

savename_bins_lcoe = outpath + (
    os.path.basename(filename_dfsites)
    .replace('mean-','cf-')
    .replace('.csv','-NUMBREAKScfbins.csv'))
if os.path.exists(savename_bins_lcoe):
    quit()
print(savename_bins_lcoe)
sys.stdout.flush()

### Load sites dataframe
dfsites = pd.read_csv(filename_dfsites)
### Create lookup dict for site-specific losses
site2loss = dict(zip(dfsites[index_coords].values, dfsites['transloss'].values))
### Create lookup dict for site developable area
site2area = dict(zip(dfsites[index_coords].values, dfsites.km2.values))

### Load the areas for each bin
dictarea = {}
for numbreaks in range(1,maxbreaks+1):
    areafile = (
        savename_bins_lcoe
        .replace(os.path.join('binned','cf-'),os.path.join('binned','area-'))
        .replace('NUMBREAKScfbins.csv'.format(maxbreaks), '{}cfbins.csv'.format(numbreaks))
    )
    dictarea[numbreaks] = pd.read_csv(areafile, index_col='bin_lcoe', squeeze=True).to_dict()

### Make shared index
index = pd.date_range('2007-01-01','2014-01-01',freq='H',closed='left',tz='UTC')

### Load power curve dictionary
dictcf, cutout = zephyr.wind.get_powercurve_dict(
    source=powercurvesource, model=model, pc_vmax=pc_vmax)

############
### Loop over sites; do it in procedure rather than function
### Make starting output series
dictout = {}
for numbreaks in range(1,maxbreaks+1):
    for jbreak in range(numbreaks):
        dictout[numbreaks,jbreak] = pd.Series(data=np.zeros(len(index)), index=index)

###### Simulate CF from weather files and powercurve dict
for i in tqdm(dfsites.index, desc=str(zone)):
    rowf_colf = dfsites.loc[i,'rowf_colf']
    dfwind = pd.read_pickle(
        resourcedatapath+'{}.df.p.gz'.format(rowf_colf)).copy()
    ### Interpolate to 140m if height is 140
    if height == 140:
        dfwind['pressure_140m'] = (
            dfwind.pressure_100m 
            + (dfwind.pressure_200m-dfwind.pressure_100m) * 0.4)
    elif height == 100:
        pass
    else:
        raise Exception(
            'heights not in [100,140] not yet supported; height={}'.format(
                height))
    ### Continue
    cfwind = zephyr.wind.windsim(
        dfwind=dfwind, dictcf=dictcf, height=height,
        number_of_turbines=number_of_turbines, cutout=cutout, nans=nans,
        temp_cutoff=temp_cutoff
    ### Scale up by site area and reduce by transmission losses
    )['cf_calc_hyst'] * site2area[rowf_colf] * site2loss[rowf_colf]
    ### Store it
    for numbreaks in range(1,maxbreaks+1):
        dictout[numbreaks, dfsites.loc[i,'binlcoe_{}'.format(numbreaks)]] += cfwind

### Finish the area-weighted normalization by dividing by bin area
for numbreaks in range(1,maxbreaks+1):
    for jbreak in range(numbreaks):
        dictout[numbreaks,jbreak] /= dictarea[numbreaks][jbreak]
### Make the output dataframe
dfout = pd.concat(dictout, axis=1) * (1 - loss_system_wind)
### Save it
for numbreaks in range(1,maxbreaks+1):
    dfout[numbreaks].to_csv(savename_bins_lcoe.replace('NUMBREAKS',str(numbreaks)))
