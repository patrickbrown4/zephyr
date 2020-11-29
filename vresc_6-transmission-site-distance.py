###############
### IMPORTS ###

import pandas as pd
import numpy as np
import os, sys, site, math, time, pickle, gzip
from glob import glob
from tqdm import tqdm, trange

import shapely
import geopandas as gpd
import geopy, geopy.distance

import zephyr

### Common paths
projpath = zephyr.settings.projpath
datapath = zephyr.settings.datapath
extdatapath = zephyr.settings.extdatapath

#######################
### ARGUMENT INPUTS ###

import argparse
parser = argparse.ArgumentParser(description='transmission-site distance')

parser.add_argument(
    '-r', '--resource', help="'wind' or 'pv'", type=str, choices=['wind','pv'])
parser.add_argument(
    '-z', '--overwrite', help='indicate whether to overwrite existing outputs',
    action='store_true')

### Parse argument inputs
args = parser.parse_args()
resource = args.resource
overwrite = args.overwrite

##############
### INPUTS ###

###### >>> Modify this section based on input shapefiles
zonesource = 'state'
level = 'state'
usafile = os.path.join(
    projpath,'in','Maps','HIFLD',
    'Political_Boundaries_Area','Political_Boundaries_Area.shp')
dfzones = gpd.read_file(usafile)

dfzones = dfzones.loc[
    (dfzones.COUNTRY=='USA')
    & ~(dfzones.NAME.isin(
        ["water/agua/d'eau", "Puerto Rico", 
         "Alaska", "Hawaii",
         "United States Virgin Islands", "Navassa Island",
        ])),
    ['STATEABB','geometry']
].dissolve('STATEABB')

dfzones = dfzones.reset_index().rename(columns={'STATEABB':'state'})
dfzones.state = dfzones.state.map(lambda x: x.replace('US-',''))
dfzones.index = dfzones.state
zones = dfzones.state.unique().tolist()[2::4]

# zones = ['ERCOT']
# level = 'ISO'
# zone = 'ERCOT'
# zonesfile = datapath+'ERCOT/maps/ercot_boundary/'
# dfzones = gpd.read_file(zonesfile)
# dfzones.index = [zone]
# polyzone = dfzones.loc[zone,'geometry']

### Transmission characteristics
voltcutoff = 230
urbanclasses = ['U'] ### or ['U','C']
transcostmultiplier = 1

### Costs - from ReEDS documentation 2019
inflator = zephyr.calc.inflate(2010,2017)
cost = {
    'spur': 3.667 * inflator / 1.60934, # 2010$/kW-mile --> 2017 $/kW-km
    230: 3.667 * inflator / 1.60934, # 2010$/kW-mile --> 2017 $/kW-km
    345: 2.333 * inflator / 1.60934, # 2010$/kW-mile --> 2017 $/kW-km
    500: 1.347 * inflator / 1.60934, # 2010$/kW-mile --> 2017 $/kW-km
    765: 1.400 * inflator / 1.60934, # 2010/kW-mile --> 2017$/kW-km
}

### Cover the in-between voltages
### Set cost equal to cost of next-lowest voltage
for c in [236, 240, 245, 250, 287]:
    cost[c] = cost[230]
for c in [400,450]:
    cost[c] = cost[345]
for c in [1000]:
    cost[c] = cost[765]

### Scale by transmission cost multiplier
for c in cost:
    cost[c] = cost[c] * transcostmultiplier

### Financial assumptions
def crf(wacc, lifetime):
    out = ((wacc * (1 + wacc) ** lifetime) 
           / ((1 + wacc) ** lifetime - 1))
    return out

wacc_gen = 0.042
wacc_trans = 0.036
lifetime_gen = 25
lifetime_trans = 50
crf_gen = crf(wacc=wacc_gen, lifetime=lifetime_gen)
crf_trans = crf(wacc=wacc_trans, lifetime=lifetime_trans)

def cost_trans_annual(row):
    out = row['cost_trunk_upfront'] * crf_trans + row['cost_spur_upfront'] * crf_gen
    return out

### Filepaths
outpath = {
    'pv': os.path.join(
        projpath,'io','cf-2007_2013','v08_states-LCOE-transloss-urbanedge','{}x','{}',
        'pv','distance-station_urbanedge-mincost').format(transcostmultiplier, level),
    'wind': os.path.join(
        projpath,'io','cf-2007_2013','v08_states-LCOE-transloss-urbanedge','{}x','{}',
        'wind','distance-station_urbanedge-mincost').format(transcostmultiplier, level),
}

#####################################################
### MORE INPUTS (adjust based on zones shapefile) ###

neighborstates = {
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

lat = {'pv':'Latitude', 'wind':'latitude'}
lon = {'pv':'Longitude', 'wind':'longitude'}

#################
### PROCEDURE ###

###################
### Load input data

###### EIA transmission
##### First download from 
##### https://hifld-geoplatform.opendata.arcgis.com/datasets/electric-power-transmission-lines/data
filesin = {
    'Transmission': os.path.join(
        projpath,'in','Maps','HIFLD',
        'Electric_Power_Transmission_Lines','Electric_Power_Transmission_Lines.shp'),
}
dftrans = gpd.read_file(filesin['Transmission'])

dftransvolt = dftrans.loc[dftrans.VOLTAGE>0].reset_index(drop=True).copy()

### Cut PR
dftransvolt = dftransvolt.loc[
    ~((dftransvolt.centroid.x >= -70) 
      & (dftransvolt.centroid.y <= 20))
].reset_index(drop=True).copy()

### Get the transmission dataframe
dftranshv = dftransvolt.loc[dftransvolt.VOLTAGE>=voltcutoff].reset_index(drop=True).copy()

### Load urban shapefile
### First download from https://www2.census.gov/geo/tiger/TIGER2019/UAC/
dfurban_all = gpd.read_file(
    os.path.join(
        projpath,'in','Maps','Census','urbanarea',
        'tl_2010_us_uac10','tl_2010_us_uac10.shp')
)

### Create a list of endpoints of transmission segments
transends = []
volts = []
ids = []
for i in dftranshv.index:
    for foo in range(2):
        volts.append(dftranshv.loc[i,'VOLTAGE'])
        ids.append(dftranshv.loc[i,'OBJECTID'])
    line = dftranshv.loc[i,'geometry']
    if type(line) == shapely.geometry.linestring.LineString:
        ### First point
        transends.append((line.coords[0][0], line.coords[0][1]))
        ### Last point
        transends.append((line.coords[-1][0], line.coords[-1][1]))
    else:
        ### Merge the constituent lines
        outcoords = [list(i.coords) for i in line]
        outline = shapely.geometry.LineString([i for sublist in outcoords for i in sublist])
        ### First point
        transends.append((outline.coords[0][0], outline.coords[0][1]))
        ### Last point
        transends.append((outline.coords[-1][0], outline.coords[-1][1]))
dfends = pd.DataFrame(transends, columns=['lon','lat'])
dfends['voltage'] = volts
dfends['objectid'] = ids
dfends['geometry'] = [shapely.geometry.Point(i) for i in transends]
dfends = gpd.GeoDataFrame(dfends)

###################
### Loop over zones

for zone in zones:
    ###### Filepaths
    outfile = {
        'pv': 'nsrdb,icomesh9-urban{}-trans{}-{}_{}.csv'.format(
            ''.join(urbanclasses), voltcutoff, level, zone),
        'wind': 'wtkhsds,every2,offset0,onshore-urban{}-trans{}-{}_{}.csv'.format(
            ''.join(urbanclasses), voltcutoff, level, zone)
    }

    index_coords = {'pv':'psm3id', 'wind':'rowf_colf'}
    weightsfile = {
        'pv': os.path.join(
            projpath,'io','geo','developable-area','{}','nsrdb-icomesh9',
            'sitearea-water,parks,native,mountains,urban-{}_{}.csv'
        ).format(level, level, zone),
        'wind': os.path.join(
            projpath,'io','geo','developable-area','{}','wtk-hsds-every2',
            'sitearea-water,parks,native,mountains,urban-{}_{}.csv'
        ).format(level, level, zone)
    }

    ###################################################
    ### Calculate interconnection distances + costs ###

    ### Make output folder
    os.makedirs(outpath[resource], exist_ok=True)

    ### Skip if it's done
    if os.path.exists(os.path.join(outpath[resource],outfile[resource])) and (overwrite == False):
        continue

    ###### Load site dataframes
    if resource == 'pv':
        dfcoords = pd.read_csv(os.path.join(
            projpath,'io','icomesh-nsrdb-info-key-psmv3-eGRID-avert-ico9.csv'))

    elif resource == 'wind':
        ### Load HSDS points
        dfcoords = pd.read_csv(
            os.path.join(projpath,'io','wind','hsdscoords.gz')
        ).rename(columns={'row':'row_full','col':'col_full'})
        ### Make the lookup index
        dfcoords['rowf_colf'] = (
            dfcoords.row_full.astype(str)+'_'+dfcoords.col_full.astype(str))
    
    ### Load site weights for the modeled zone, merging multiply-listed sites
    polyweights = pd.read_csv(
        weightsfile[resource]
    ).groupby(index_coords[resource]).sum().reset_index()

    ### Merge sites with calculated land area
    dfsites = dfcoords.merge(polyweights, on=index_coords[resource], how='inner')
    
    ###### Transmission
    ### Get the zone site polygons
    polyzone = dfzones.loc[zone,'geometry']

    ### Get the line endpoints within the modeled zone
    dfendszone = dfends.loc[dfends.within(polyzone)].copy()

    ###### Urban areas
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

    dfurban_states = dfurban_all.loc[
        dfurban_all.NAME10.astype(str).map(lambda x: x[-2:] in neighborstates[zone])
    ].copy()

    ### Merge all the urban areas, since we don't need to know which is which
    ### Filter to urban areas included in urbanclasses
    ### 'U' is urban area of >50k people; 'C' is urban cluster of ≥2.5k and <50k
    ### https://www2.census.gov/geo/pdfs/reference/ua/2010ua_faqs.pdf
    dfurban = dfurban_states.loc[dfurban_states.UATYP10.isin(urbanclasses)]
    dfurban = dfurban.dissolve('FUNCSTAT10')

    ### Pull out the urban polygon
    polyurban = dfurban.loc['S','geometry'].buffer(0.)
    polyurban = polyurban.intersection(regionbox).buffer(0.)

    ### Filter to urban areas within the state
    dfurbanzone = dfurban.copy()
    dfurbanzone['geometry'] = dfurbanzone.intersection(polyzone)
    polyurbanzone = polyurban.intersection(polyzone).buffer(0.)

    ### Put it in a geodataframe
    multipolyurbanzone = []
    for x in dfurbanzone.iloc[0]['geometry']:
        if type(x) not in [shapely.geometry.linestring.LineString]:
            multipolyurbanzone.append(shapely.geometry.Polygon(x))
    dfurbanzone = gpd.GeoSeries(shapely.geometry.MultiPolygon(multipolyurbanzone))

    ### Get urban centroid
    urban_centroid_lon, urban_centroid_lat = list(dfurbanzone.iloc[0].centroid.coords)[0]

    ######### Calculate substation distances and costs
    ###### Substation-urban edge distances and costs
    ### Loop over substations
    distances_trunk = []
    lat_urban_fromstation = []
    lon_urban_fromstation = []

    for i in dfendszone.index:
        ### Get the site-urban distance in km
        stationpoint = shapely.geometry.Point(dfendszone.loc[i,'lon'], dfendszone.loc[i,'lat'])
        polypoint = shapely.ops.nearest_points(stationpoint, polyurbanzone)[1]
        km = geopy.distance.distance((stationpoint.y, stationpoint.x), (polypoint.y,polypoint.x)).km
        distances_trunk.append(km)
        lat_urban_fromstation.append(polypoint.y)
        lon_urban_fromstation.append(polypoint.x)

    ### Store the trunk distance data
    dfendszone['km_urban_fromstation'] = distances_trunk
    dfendszone['lat_urban_fromstation'] = lat_urban_fromstation
    dfendszone['lon_urban_fromstation'] = lon_urban_fromstation
    ### Calculate trunk costs
    dfendszone['cost_trunk_upfront'] = (
        dfendszone['voltage'].map(cost) * dfendszone['km_urban_fromstation'])

    ###### Site-substation costs
    ### Loop over sites
    dictout = {}
    for i in tqdm(dfsites.index, desc='{},{}'.format(level,zone)):
        sitepoint = shapely.geometry.Point(
            dfsites.loc[i,lon[resource]], dfsites.loc[i,lat[resource]])
        ### Get all distances
        dfendszone['km_site_spur'] = dfendszone.apply(
            lambda row: geopy.distance.distance((sitepoint.y, sitepoint.x),(row['lat'],row['lon'])).km,
            axis=1,
        )
        dfendszone['cost_spur_upfront'] = dfendszone['km_site_spur'] * cost[230]
        dfendszone['cost_trans_upfront'] = (
            dfendszone['cost_trunk_upfront'] + dfendszone['cost_spur_upfront'])
        ### Calculate annualized cost
        dfendszone['cost_trans_annual'] = dfendszone.apply(cost_trans_annual, axis=1)

        ### Store all the output information
        dictout[i] = (dfendszone.nsmallest(1,'cost_trans_annual')
            .drop(['geometry'],axis=1)
            .rename(columns={'lon':'lon_station','lat':'lat_station'}))

    ### Save it
    dfout = pd.concat(dictout).reset_index(level=1,drop=True)
    dfsites = dfsites.merge(dfout, left_index=True, right_index=True, how='left')
    dfsites.to_csv(os.path.join(outpath[resource],outfile[resource]), index=False)
