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

#######################
### ARGUMENT INPUTS ###
import argparse
parser = argparse.ArgumentParser(description='voronoi-poly overlap')

parser.add_argument(
    'zone', help='zone name (such as state abbreviation)', type=str)
parser.add_argument(
    '-x', '--zonesource', type=str, default='state', help='region type')
parser.add_argument(
    '-a', '--prefix', help='filename before zone', type=str,
    default='polyavailable-water,parks,native,mountains,urban-')
parser.add_argument(
    '-b', '--polybuffer', default=0.5, type=float,
    help='degree buffer; use 0.5 for wind and 2.0 for PV')
parser.add_argument(
    '-r', '--resource', help="'wtk' or 'icomesh9' or 'merra2'",
    type=str)
parser.add_argument(
    '-f', '--returnfull', action='store_true', help='specify whether to save dfpoly')
parser.add_argument(
    '-c', '--returnarea', action='store_true', help='specify whether to save dfarea')

args = parser.parse_args()

zone = args.zone
zonesource = args.zonesource
prefix = args.prefix
polybuffer = args.polybuffer
resource = args.resource
returnfull = args.returnfull
returnarea = args.returnarea
if resource in ['wind','wtk','wtk-hsds','wtk-hsds-every2']:
    resource = 'wtk-hsds-every2'
elif resource in ['solar','pv','nsrdb','icomesh','icomesh9','psm3','nsrdb-icomesh9']:
    resource = 'nsrdb-icomesh9'
elif resource in ['merra2','merra','nasa','merra-2']:
    resource = 'merra2'
else:
    exc = "Invalid resource: {}. Try 'wtk' or 'icomesh9'.".format(resource)
    raise Exception(exc)

##############
### INPUTS ###

inpath = os.path.join(projpath,'io','geo','developable-area','{}','').format(zonesource)
outpath = os.path.join(inpath,resource)
polyfile = prefix+'{}_{}.poly.p'.format(zonesource, zone)

savepoly = polyfile.replace('polyavailable','sitepoly').replace('.poly.p','.gdf.p')
savearea = polyfile.replace('polyavailable','sitearea').replace('.poly.p','.csv')

#################
### PROCEDURE ###

if returnarea:
    print(os.path.join(outpath,savearea))
if returnfull:
    print(os.path.join(outpath,savepoly))
if not any([returnarea, returnfull]):
    print('need to specify -f and/or -c')
    quit()
os.makedirs(outpath, exist_ok=True)

###### Load the available polygon
with open(os.path.join(inpath,polyfile), 'rb') as p:
    polyavailable_region = pickle.load(p)

if resource == 'wtk-hsds-every2':
    ###### Wind
    index_coords = 'rowf_colf'
    ### Get WTK coordinates
    ### Load all HSDS points
    dfcoords = pd.read_csv(
        os.path.join(projpath,'io','geo','hsdscoords.gz')
    ).rename(columns={'row':'row_full','col':'col_full'})
    ### Index by original row,column
    dfcoords[index_coords] = dfcoords.row_full.astype(str)+'_'+dfcoords.col_full.astype(str)
    ### Filter to even points
    dfcoords = dfcoords.loc[
        (dfcoords.row_full % 2 == 0)
        & (dfcoords.col_full % 2 == 0)
    ].reset_index()
elif resource == 'nsrdb-icomesh9':
    ###### NSRDB icomesh9
    index_coords = 'psm3id'
    ### Have to use extra points beyond edge of US, same as for wind
    dfcoords = pd.read_csv(
        os.path.join(
            projpath,'io','geo',
            'world-points-icomesh-9subdiv-psm3id.csv'),
        dtype={'psm3id':'category'}
    )
    dfcoords = dfcoords.loc[dfcoords.psm3id.notnull()].copy()
elif resource == 'merra2':
    ###### MERRA-2
    index_coords = 'latlon'
    ### Load sites from any CF dataframe
    model = 'Gamesa:G126/2500_low'
    modelsave = model.replace(':','|').replace('/','_')
    height = 100
    loss_system_wind = 0.19
    infile = os.path.join(
        datapath,'NASA','MERRA2','M2T1NXSLV-usa1degbuff-{}m',
        'CF-0loss-{}.df.p.gz').format(height, modelsave)
    with gzip.open(infile, 'rb') as p:
        dfin = pickle.load(p) * (1 - loss_system_wind)
    ### Pull the coordinates
    dfcoords = pd.DataFrame({
        'lat': [x[0] for x in dfin.columns.values],
        'lon': [x[1] for x in dfin.columns.values],
        'latlon': ['{}_{}'.format(*x) for x in dfin.columns.values]
    })


###### Get overlap of Voronoi polygons of dfcoords with polyavailable_region
polyweights = zephyr.toolbox.voronoi_polygon_overlap(
    dfcoords=dfcoords, polyoverlap=polyavailable_region, 
    index_coords=index_coords, polybuffer=polybuffer,
    returnfull=returnfull)

###### Save it
if returnfull:
    with open(os.path.join(outpath,savepoly), 'wb') as p:
        pickle.dump(polyweights.dropna(subset=[index_coords]), p)
if returnarea:
    polyweights.dropna(subset=[index_coords]).drop(['geometry'],axis=1).to_csv(
        os.path.join(outpath,savearea), index=False)
