###############
### IMPORTS ###

import pandas as pd
import numpy as np
import os, sys, site, pickle, json, shapely, zipfile, gzip
from glob import glob
from tqdm import tqdm, trange
import geopandas as gpd
import pyproj

import zephyr

projpath = zephyr.settings.projpath
datapath = zephyr.settings.datapath

#######################
### ARGUMENT INPUTS ###
import argparse
parser = argparse.ArgumentParser(description='polyavailable states')
parser.add_argument(
    '-z', '--zone', type=str, help='zone name (such as state abbreviation)',)
parser.add_argument(
    '-s', '--zonesource', type=str, default='state',
    help='shapefile from which to read zones (located at in/Maps/{zonesource}/)',)
args = parser.parse_args()
zone = args.zone
zonesource = args.zonesource

### Parse the input into a state if necessary
statenumber2state = dict(zip(
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
try:
    statenumber = int(zone)
    zone = statenumber2state[statenumber]
except ValueError:
    pass

##############
### INPUTS ###

zonefile = os.path.join(datapath,'Maps',zonesource)
dfzones = gpd.read_file(zonefile)

neighborstates = {
    ### If using a zonesource other than "state", should add entries to this dict
    ### giving the list of states contained within and adjacent to each zone in zonesource.
    ### Otherwise we use urban areas and water bodies for the full US, which slows down
    ### the execution.
    'AL': ['AL','MS','TN','GA','FL'],
    'AK': ['AK',],
    'AZ': ['AZ','CA','NV','UT','CO','NM'],
    'AR': ['AR','TX','OK','MO','TN','MS','LA'],
    'CA': ['CA','OR','NV','AZ'],
    'CO': ['CO','AZ','UT','WY','NE','KS','OK','NM'],
    'CT': ['CT','NJ','NY','MA','RI'],
    'DE': ['DE','MD','DC','VA','PA','NJ'],
    'FL': ['FL','AL','GA'],
    'GA': ['GA','AL','TN','NC','SC','FL'],
    'HI': ['HI',],
    'ID': ['ID','NV','OR','WA','MT','WY','UT'],
    'IL': ['IL','MO','IA','WI','IN','KY'],
    'IN': ['IN','IL','MI','OH','KY'],
    'IA': ['IA','NE','SD','MN','WI','IL','MO'],
    'KS': ['KS','CO','NE','MO','OK'],
    'KY': ['KY','MO','IL','IN','OH','WV','VA','TN'],
    'LA': ['LA','TX','AR','MS'],
    'ME': ['ME','NH','MA'],
    'MD': ['MD','VA','DC','WV','PA','NJ','DE'],
    'MA': ['MA','RI','CT','NY','NJ','VT','NH'],
    'MI': ['MI','IN','IL','WI','OH','KY'],
    'MN': ['MN','IA','SD','ND','WI'],
    'MS': ['MS','LA','AR','TN','AL'],
    'MO': ['MO','AR','OK','KS','NE','IA','IL','KY','TN'],
    'MT': ['MT','ID','WY','SD','ND'],
    'NE': ['NE','KS','CO','WY','SD','IA','MO'],
    'NV': ['NV','CA','OR','ID','UT','AZ'],
    'NH': ['NH','MA','VT','ME'],
    'NJ': ['NJ','DE','MD','DC','PA','NY','CT'],
    'NM': ['NM','AZ','UT','CO','OK','TX'],
    'NY': ['NY','NJ','PA','VT','MA','CT'],
    'NC': ['NC','SC','GA','TN','VA','DC'],
    'ND': ['ND','SD','MT','MN'],
    'OH': ['OH','KY','IN','MI','PA','WV'],
    'OK': ['OK','TX','NM','CO','KS','MO','AR'],
    'OR': ['OR','NV','CA','WA','ID'],
    'PA': ['PA','WV','OH','NY','NJ','DE','MD','DC','VA'],
    'RI': ['RI','CT','NY','MA'],
    'SC': ['SC','GA','NC'],
    'SD': ['SD','NE','WY','MT','ND','MN','IA'],
    'TN': ['TN','MS','AR','MO','KY','VA','NC','GA','AL',],
    'TX': ['TX','NM','OK','AR','LA'],
    'UT': ['UT','AZ','NV','ID','WY','CO','NM'],
    'VT': ['VT','NY','NH','MA'],
    'VA': ['VA','WV','MD','PA','DC','NC','TN','KY'],
    'WA': ['WA','OR','ID'],
    'WV': ['WV','KY','OH','PA','MD','DC','VA'],
    'WI': ['WI','IL','IA','MN','MI'],
    'WY': ['WY','UT','ID','MT','SD','NE','CO'],
    'DC': ['DC','VA','WV','MD','PA','DC'],
}


#################
### PROCEDURE ###

print(zone)

### savename
savename = 'polyavailable-water,parks,native,mountains,urban-{}_{}.poly.p'.format(
    zonesource, zone)
outpath = os.path.join(projpath,'io','geo','developable-area','{}').format(zonesource) + os.sep
os.makedirs(outpath, exist_ok=True)
print(os.path.join(outpath,savename))

### Get zone polygon
dfzone = dfzones.loc[dfzones[zonesource] == zone].reset_index(drop=True).copy()
polyzone = dfzone.loc[0,'geometry'].buffer(0.)

### Get neighboring states from neighborstates; otherwise use all states
allstates = [s for s in zephyr.toolbox.usps if s not in ['AK','HI']]
zonestates = neighborstates.get(zone, allstates)

### Get bounding box for region, add 0.5° buffer
regionbounds = {
    'longitude':[polyzone.bounds[0]-0.5, polyzone.bounds[2]+0.5],
    'latitude':[polyzone.bounds[1]-0.5, polyzone.bounds[3]+0.5],
}

### Get region bounding box
regionbox = shapely.geometry.Polygon([
    (regionbounds['longitude'][0], regionbounds['latitude'][0]),
    (regionbounds['longitude'][1], regionbounds['latitude'][0]),
    (regionbounds['longitude'][1], regionbounds['latitude'][1]),
    (regionbounds['longitude'][0], regionbounds['latitude'][1]),
])

outside = regionbox.difference(polyzone).buffer(0.)
dfoutside = gpd.GeoDataFrame(geometry=[outside])

###################
### Land exclusions

###### Urban areas
### Load urban shapefile
dfurban_all = gpd.read_file(os.path.join(datapath,'Maps','Census','tl_2010_us_uac10'))

dfurban_states = dfurban_all.loc[
    dfurban_all.NAME10.astype(str).map(lambda x: x[-2:] in zonestates)
].copy()

### Merge all the urban areas, since we don't need to know which is which
### Note that we include all urban areas and urban clusters
dfurban = dfurban_states.dissolve('FUNCSTAT10')

### Pull out the urban polygon
polyurban = dfurban.loc['S','geometry'].buffer(0.)
polyurban = polyurban.intersection(regionbox).buffer(0.)

print('imported urban')

###### Native American territories
dfnative_all = gpd.read_file(os.path.join(
    datapath,'Maps','Census','tl_2017_us_aitsn','tl_2017_us_aitsn.shp'))
## Filter out the Oklahoma Tribal Statistical areas
dfnative = dfnative_all.loc[dfnative_all.FUNCSTAT != 'S'].copy()
dfnative['dummy'] = 0
native = dfnative.dissolve('dummy')
polynative = native.loc[0, 'geometry']
polynative = polynative.intersection(regionbox).buffer(0.)

print('imported native')

###### National parks
with open(os.path.join(projpath,'in','Maps','NPS','nationalparks-all-poly.p'), 'rb') as p:
    polyparks = pickle.load(p)

polyparks = polyparks.intersection(regionbox).buffer(0.)
parks = gpd.GeoDataFrame(geometry=[polyparks])

print('imported parks')

###### Water bodies
water = gpd.read_file(os.path.join(
    datapath,'Maps','USGS','wtrbdyp010g.shp_nt00886','wtrbdyp010g.shp'))

water = water.loc[
    ~water.State.isin(['AK','HI','PR','VI','GU'])
].copy()

### Get subset of water within modeled region
regionwater = water.loc[water.State.map(
    lambda x: any([state in x for state in zonestates]))].copy()

regionwater['dummy'] = 0
regionwater = regionwater.dissolve('dummy')
polywater = regionwater.loc[0, 'geometry'].buffer(0.)
polywater = polywater.intersection(regionbox).buffer(0.)

print('imported water')

###### Mountains
mountaintypes = ['high','highscattered']
mountains = {}
for mountain in mountaintypes:
    gdf = gpd.read_file(os.path.join(
        projpath,'io','geo','mountains','usgsgmek3_{}_-170,-30lon_5,70lat'.format(mountain)
    ))
    poly = gdf.loc[0,'geometry']
        
    ### Merge with bounding box
    mountains[mountain] = poly.buffer(0.).intersection(regionbox)

polymountains = mountains[mountaintypes[0]]
for i in range(1,len(mountaintypes)):
    polymountains = polymountains.union(mountains[mountaintypes[i]]).buffer(0.)

mountains = gpd.GeoDataFrame(geometry=[polymountains])

print('imported mountains')

###### Merge
### Merge polygons, get polygon of developable area
polyavailable_region = polyzone
print('polyzone: {:.3f}'.format(polyavailable_region.area))
polyavailable_region = polyavailable_region.difference(polywater)
print('subtracted water: {:.3f}'.format(polyavailable_region.area))
polyavailable_region = polyavailable_region.difference(polyparks)
print('subtracted parks: {:.3f}'.format(polyavailable_region.area))
polyavailable_region = polyavailable_region.difference(polynative)
print('subtracted native: {:.3f}'.format(polyavailable_region.area))
polyavailable_region = polyavailable_region.difference(polymountains)
print('subtracted mountains: {:.3f}'.format(polyavailable_region.area))
polyavailable_region = polyavailable_region.difference(polyurban)
print('subtracted urban: {:.3f}'.format(polyavailable_region.area))
polyavailable_region = polyavailable_region.buffer(0.)
print('buffered 0.: {:.3f}'.format(polyavailable_region.area))

###### Save it
os.makedirs(outpath, exist_ok=True)
with open(os.path.join(outpath,savename), 'wb') as p:
    pickle.dump(polyavailable_region, p)

