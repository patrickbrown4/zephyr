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
    '-r', '--resource', help="'wind' or 'pv'", type=str)
# parser.add_argument(
#     '-g', '--gencostannual', type=float,
#     help='annuitized generator cost in $/kWac-yr; include FOM')
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
    '-m', '--model', default='Gamesa:G126/2500_low', type=str,
    help="wind turbine model, default 'Gamesa:G126/2500_low'",)
parser.add_argument(
    '-y', '--runtype', default='full', choices=['full','test'], type=str)
parser.add_argument(
    '-f', '--fulloutput', action='store_true',
    help="specify whether to save the hourly output")
parser.add_argument(
    '-a', '--allsites', action='store_true',
    help="specify whether to write hourly cf for all sites")
parser.add_argument(
    '-b', '--maxbreaks', default=10, type=int,
    help="Maximum number of jenks breaks to include")
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
runtype = args.runtype
fulloutput = args.fulloutput
allsites = args.allsites
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
defaults = zephyr.system.Defaults(infile)
pvcost = 'PV_track_2030_mid'
windcost = 'Wind_2030_mid'
if resource == 'pv':
    gen = zephyr.system.Gen(pvcost, defaults=defaults)
elif resource == 'wind':
    gen = zephyr.system.Gen(windcost, defaults=defaults)
gencostannual = gen.cost_annual + gen.cost_fom

### Wind system losses
loss_system_wind = 0.19 ### Owen Roberts NREL 20200331
### Transmission losses (1% per 100 miles)
loss_distance = 0.01/1.60934/100

### Set output file location and savenames
distancepath = os.path.join(
    projpath,'io','cf-2007_2013','{}x','{}',''
).format(transcostmultiplier,'state')
outpath = os.path.join(
    projpath,'io','cf-2007_2013','{}x','{}','{}',''
).format(transcostmultiplier,level,resource)
if resource == 'wind':
    outpath = os.path.join(outpath, model.replace(':','|').replace('/','_'))
elif resource == 'pv':
    outpath = os.path.join(outpath, '{}-{}t-{:.0f}az').format(systemtype, axis_tilt, axis_azimuth)

### Set output resolution (bits) at which to save all-site CFs
resolution = 16

### Get region file
dfregion = pd.read_csv(os.path.join(projpath,'io','regions.csv'), index_col=0)

### Get states to loop over
states = dfregion.loc[dfregion[level]==zone].index.values

###### Select wind/PV-specific parameters
if resource == 'pv':
    ### Set PV parameters
    dcac = 1.3

    index_coords = 'psm3id'
    resourcedatapath = os.path.join(extdatapath,'NSRDB','ico9','')

    # weightsfile = (
    #     projpath+'io/geo/developable-area/{}/nsrdb-icomesh9/'
    #     'sitearea-water,parks,native,mountains,urban-{}_{}.csv'
    # ).format(level, level, zone)
    weightsfile = {
        state: os.path.join(
            projpath,'io','geo','developable-area','{}','nsrdb-icomesh9',
            'sitearea-water,parks,native,mountains,urban-{}_{}.csv'
        ).format('state', 'state', state)
        for state in states
    }
    # distancefile = (
    #     distancepath+'pv/distance-station_urban{}-mincost/'
    #     + 'nsrdb,icomesh9-urbanU-trans230-{}_{}.csv'
    # ).format(urban, level, zone)
    distancefile = {
        state: os.path.join(
            distancepath,'pv','distance-station_urban{}-mincost',
            'nsrdb,icomesh9-urbanU-trans230-{}_{}.csv'
        ).format(urban, 'state', state)
        for state in states
    }

    savename_cfmean = os.path.join(
        outpath,
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

    index_coords = 'rowf_colf'
    resourcedatapath = os.path.join(
        extdatapath,'WTK-HSDS','every2-offset0','timeseries','usa-halfdegbuff','')

    weightsfile = {
        state: os.path.join(
            projpath,'io','geo','developable-area','{}','wtk-hsds-every2',
            'sitearea-water,parks,native,mountains,urban-{}_{}.csv'
        ).format('state', 'state', state)
        for state in states
    }
    distancefile = {
        state: os.path.join(
            distancepath,'wind','distance-station_urban{}-mincost',
            'wtkhsds,every2,offset0,onshore-urbanU-trans230-{}_{}.csv'
        ).format(urban, 'state', state)
        for state in states
    }

    savename_cfmean = os.path.join(
        outpath,
        'mean-wtkhsds,every2,offset0,onshore-{}-{}m-{:.0f}pctloss-{:.0f}USDperkWacyr-{}_{}.csv'
    ).format(model.replace(':','|').replace('/','_'), 
             height, loss_system_wind*100, gencostannual, level, zone)

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

### Parse inputs
if resolution == 16:
    fileresolution = np.float16
elif resolution == 32:
    fileresolution = np.float32
elif resolution == 64:
    fileresolution = np.float64

### Set up output folders and filenames
os.makedirs(os.path.join(outpath,'binned'), exist_ok=True)

savename_cfhourly = (
    savename_cfmean.replace('mean-','full-')
    .replace('.csv','.df{}.p.gz'.format(resolution)))
savename_bins_lcoe = (
    savename_cfmean.replace('mean-',os.path.join('binned','cf-'))
    .replace('.csv','-NUMBREAKSlcoebins.csv'))
savename_bins_area = (
    savename_cfmean.replace('mean-',os.path.join('binned','area-'))
    .replace('.csv','-NUMBREAKSlcoebins.csv'))
print('\n'+savename_cfmean)
sys.stdout.flush()

# ### If savename_cfmean exists, quit
# if os.path.exists(savename_cfmean):
#     quit()

### Load site weights for the modeled zone, merging multiply-listed sites
# polyweights = pd.read_csv(weightsfile).groupby(index_coords).sum().reset_index()
polyweights = pd.concat(
    {state: pd.read_csv(weightsfile[state]).groupby(index_coords).sum() 
     for state in states},
    axis=0, names=['state',index_coords]
).reset_index()

### Load distance dataframe
# dfdistance = pd.read_csv(distancefile)[
#     [index_coords,'cost_trans_annual','km_urban_fromstation','km_site_spur']]
dfdistance = pd.concat(
    {state: pd.read_csv(distancefile[state])[
        [index_coords,'cost_trans_annual','km_urban_fromstation','km_site_spur']]
     for state in states},
    axis=0, names=['state','extra']
)

###### Simulate capacity factors for each site
if resource == 'pv':
    ###### Load points to model
    dfcoords = pd.read_csv(
        os.path.join(projpath,'io','geo','icomesh-nsrdb-info-key-psmv3-eGRID-avert-ico9.csv'))
    ### Merge sites with calculated land area
    dfsites = dfcoords.merge(polyweights, on=index_coords, how='inner')
    ### Create the PV system
    PV = zephyr.pv.PVsystem(
        systemtype=systemtype, dcac=dcac, 
        axis_tilt=axis_tilt, axis_azimuth=axis_azimuth)

    ### Simulate PV CFs for each site
    dictout = {}
    for index in tqdm(dfsites.index, desc='cf'):
        nsrdbid = dfsites.loc[index,'psm3id']
        dictpv = []
        for year in years:
            nsrdbfile = os.path.join('{}{}','v3','{}_{}_{}_{}.csv').format(
                resourcedatapath, year, nsrdbid,
                dfsites.loc[index,'Latitude'],
                dfsites.loc[index,'Longitude'],
                year)
            ### Simulate and downsample to hourly
            dictpv.append(
                zephyr.pv.downsample_trapezoid(
                    PV.sim(nsrdbfile, year)/1000, clip=1E-11))
        dfpv = pd.concat(dictpv, axis=0)
        dictout[nsrdbid] = dfpv
    dfout = pd.concat(dictout, axis=1, copy=False)
    if not fulloutput:
        dfout = dfout.mean(axis=0)

elif resource == 'wind':
    ###### Load points to model
    ### Load HSDS points
    dfcoords = pd.read_csv(
        os.path.join(projpath,'io','geo','hsdscoords.gz')
    ).rename(columns={'row':'row_full','col':'col_full'})
    ### Make the lookup index
    dfcoords['rowf_colf'] = (
        dfcoords.row_full.astype(str)+'_'+dfcoords.col_full.astype(str))

    ### Merge sites with calculated land area
    dfsites = dfcoords.merge(polyweights, on=index_coords, how='inner')

    ### Simulate wind CFs
    rowf_colfs = dfsites.rowf_colf.values

    dfout = zephyr.wind.windsim_hsds_timeseries(
        rowf_colfs=rowf_colfs, height=height, 
        number_of_turbines=number_of_turbines, powercurvesource=powercurvesource, 
        model=model, nans='raise', timeseries=runtype, extdatapath=extdatapath,
        windpath=resourcedatapath, verbose=True, fulloutput=fulloutput,
        pc_vmax=pc_vmax, temp_cutoff=temp_cutoff,
    ) * (1 - loss_system_wind)


### Merge with dfsites
if fulloutput:
    dfcf = dfout.mean().reset_index().rename(columns={'index':index_coords,0:'cf'})
else:
    dfcf = dfout.reset_index().rename(columns={'index':index_coords,0:'cf'})
dfsites = dfsites.merge(dfcf, on=index_coords)

###### Incorporate distance costs, calculate LCOE
### Merge with dfdistance
dfsites = dfsites.merge(dfdistance, on=index_coords)

### Calculate LCOE including transmission interconnection cost
def lcoe(row):
    out = (
        gencostannual + row['cost_trans_annual']
    ) / row['cf'] / 8760 * 1000
    return out
dfsites['lcoe'] = dfsites.apply(lcoe, axis=1)

### Calculate transmission losses
dfsites['transloss'] = (
    1 - loss_distance * (dfsites['km_site_spur'] + dfsites['km_urban_fromstation']))
### Calculate CF with transmission losses
dfsites['cf_transloss'] = dfsites['cf'] * dfsites['transloss']

### Calculate LCOE including transmission losses
def lcoe_transloss(row):
    out = (
        gencostannual + row['cost_trans_annual']
    ) / row['cf_transloss'] / 8760 * 1000
    return out
dfsites['lcoe_transloss'] = dfsites.apply(lcoe_transloss, axis=1)

### Create lookup dict for site-specific losses
site2loss = dict(zip(dfsites[index_coords].values, dfsites['transloss'].values))

### Save the modeled CF averaged over 2007-2013
print('\n'+savename_cfmean)
dfsites.to_csv(savename_cfmean, index=False)

###### Calculate the new CF timeseries with losses
###### (note that it only works if dfout is a dataframe; i.e. if fulloutput)
if fulloutput:
    for col in dfout.columns:
        dfout[col] = dfout[col] * site2loss[col]

### Save the hourly modeled CF at low resolution, in case 
### we want to use it to make bins based on correlation
if fulloutput and allsites:
    print('\n'+savename_cfhourly)
    dfout.astype(fileresolution).to_pickle(savename_cfhourly, compression='gzip')


####################################################################
### Calculate and save CF timeseries and km^2 capacities for CF bins

###### Save the modeled CF averaged over 2007-2013 with breaks labeled
### For consistency, make a column for 1 break
dfsites['binlcoe_{}'.format(1)] = 0
jbreaks = {1: [np.nan]}

### Loop over number of breaks, make new column for each
for numbreaks in range(2, maxbreaks+1):
    ###### Calculate Jenks breaks
    dfsites['binlcoe_{}'.format(numbreaks)] = zephyr.toolbox.partition_jenks(
        dfsites, 'lcoe_transloss', numbreaks)
    jbreaks[numbreaks] = zephyr.toolbox.partition_jenks(
        dfsites, 'lcoe_transloss', numbreaks, returnbreaks=True)[1:-1]

### Resave it
print('\n'+savename_cfmean)
dfsites.to_csv(savename_cfmean, index=False)


###### Loop over number of breaks and save each as its own file
for numbreaks in range(maxbreaks, 1, -1):
    ###### Calculate Jenks breaks
    dfsites['bin_lcoe'] = zephyr.toolbox.partition_jenks(
        dfsites, 'lcoe_transloss', numbreaks)
    jbreaks = zephyr.toolbox.partition_jenks(
        dfsites, 'lcoe_transloss', numbreaks, returnbreaks=True)[1:-1]
    ###### Calculate weighting factors (by developable area) for each bin
    dfsites['weight'] = -1
    for jbreak in range(numbreaks):
        dfsites.loc[dfsites.bin_lcoe == jbreak, 'weight'] = (
            ### Divide area of cell by area of all cells in bin
            dfsites.loc[dfsites.bin_lcoe == jbreak, 'km2']
            / dfsites.loc[dfsites.bin_lcoe == jbreak, 'km2'].sum()
        )

    ### Save the total developable areas in each bin
    savename = savename_bins_area.replace('NUMBREAKS', str(numbreaks))
    print(numbreaks)
    pd.DataFrame(dfsites.groupby('bin_lcoe')['km2'].sum()).to_csv(savename)

    if fulloutput:
        ###### Get hourly area-weighted CF for each zone and partition
        dfoutput = {}
        for jbreak in range(numbreaks):
            ### Collect hourly profiles of all sites in bin
            dfoutput_jbreak = []
            for index in dfsites.loc[dfsites.bin_lcoe == jbreak].index:
                site = dfsites.loc[index, index_coords]
                weight = dfsites.loc[index, 'weight']
                ### Weight each profile by the developable area
                dfoutput_jbreak.append(dfout[site] * weight)
            ### Sum and save the weighted hourly profiles
            dfoutput[jbreak] = pd.concat(dfoutput_jbreak, axis=1).sum(axis=1)

        dfoutput = pd.concat(dfoutput, axis=1)

        ### Save the hourly CFs
        savename = savename_bins_lcoe.replace('NUMBREAKS', str(numbreaks))
        dfoutput.to_csv(savename)

###### Now save it without Jenks breaks (just all sites equally weighted)
numbreaks = 1

### Save the total developable area
savename = savename_bins_area.replace('NUMBREAKS', str(numbreaks))
print(numbreaks)
pd.DataFrame({'bin_lcoe':[0], 'km2': dfsites.km2.sum()}).to_csv(
    savename, index=False)

### Calculate weighting factors (by developable area)
if fulloutput:
    dfsites['weight'] = dfsites.km2 / dfsites.km2.sum()
    dfoutput = []
    for index in dfsites.index:
        site = dfsites.loc[index, index_coords]
        weight = dfsites.loc[index, 'weight']
        dfoutput.append(dfout[site] * weight)
    dfoutput = pd.concat(dfoutput, axis=1).sum(axis=1)
    ### Save the hourly CF
    savename = savename_bins_lcoe.replace('NUMBREAKS', str(numbreaks))
    dfoutput.to_csv(savename, header=True)
