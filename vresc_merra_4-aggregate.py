###############
#%% IMPORTS ###

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


#################
#%% PROCEDURE ###

###### Load pre-simulated CF
model = 'Gamesa:G126/2500_low'
modelsave = model.replace(':','|').replace('/','_')
height = 100
loss_system_wind = 0.15

infile = os.path.join(
    datapath,'NASA','MERRA2','M2T1NXSLV-usa1degbuff-{}m','CF-0loss-{}.df.p.gz'
).format(height, modelsave)

with gzip.open(infile, 'rb') as p:
    dfin = pickle.load(p) * (1 - loss_system_wind)

############################
#%% sitearea-zoneoverlap ###

dfba = pd.read_excel(
    os.path.join(projpath,'cases-test.xlsx'), 
    sheet_name='state', index_col=0, header=[0,1],
    dtype={('bins_pv','ba'):str,('bins_wind','ba'):str},
).xs('ba',axis=1,level=1)
bas = dfba.area.unique().tolist()
states = dfba.index.values.tolist()

##########
### INPUTS

zonesource = 'state'
prefix = 'polyavailable-water,parks,native,mountains,urban-'
resource = 'merra2'
polybuffer = 10
returnfull = True
returnarea = True

#############
### PROCEDURE

###### Load the available polygon
with open(inpath+polyfile, 'rb') as p:
    polyavailable_region = pickle.load(p)

### Get coordinates
index_coords = 'latlon'
dfcoords = pd.DataFrame({
    'lat': [x[0] for x in dfin.columns.values],
    'lon': [x[1] for x in dfin.columns.values],
    'latlon': ['{}_{}'.format(*x) for x in dfin.columns.values]
})

for state in states:
    ### INPUTS ###
    zone = '{}_{}'.format(zonesource, state)
    inpath = os.path.join(projpath,'io','geo','developable-area','{}','').format(zonesource)
    outpath = os.path.join(inpath,'{}',''.format(resource))
    polyfile = prefix+'{}.poly.p'.format(zone)
    savepoly = polyfile.replace('polyavailable','sitepoly').replace('.poly.p','.gdf.p')
    savearea = polyfile.replace('polyavailable','sitearea').replace('.poly.p','.csv')
    
    ###### Load the available polygon
    with open(inpath+polyfile, 'rb') as p:
        polyavailable_region = pickle.load(p)
    
    ###### Get overlap of Voronoi polygons of dfcoords with polyavailable_region
    polyweights = zephyr.toolbox.voronoi_polygon_overlap(
        dfcoords=dfcoords, polyoverlap=polyavailable_region, 
        index_coords=index_coords, polybuffer=polybuffer,
        returnfull=returnfull)
    
    ###### Save it
    if returnfull:
        with open(outpath+savepoly, 'wb') as p:
            pickle.dump(polyweights.dropna(subset=[index_coords]), p)
    if returnarea:
        polyweights.dropna(subset=[index_coords]).drop(['geometry'],axis=1).to_csv(
            outpath+savearea, index=False)


##################################
#%% transmission-site-distance ###

##########
### INPUTS

resources = ['merra2']
overwrite = False
zonesource = 'state'

### Get the zones file
zonefile = os.path.join(datapath,'Maps',zonesource)
dfzones = gpd.read_file(zonefile)#.set_index(zonesource)
dfzones.index = dfzones.state
zones = dfzones.state.unique().tolist()

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

# wacc_gen = 0.042
# wacc_trans = 0.036
# lifetime_gen = 25
wacc_gen = 0.07
wacc_trans = 0.04
lifetime_gen = 30
lifetime_trans = 50
crf_gen = crf(wacc=wacc_gen, lifetime=lifetime_gen)
crf_trans = crf(wacc=wacc_trans, lifetime=lifetime_trans)

def cost_trans_annual(row):
    out = row['cost_trunk_upfront'] * crf_trans + row['cost_spur_upfront'] * crf_gen
    return out

### Filepaths
outpath = {
    'merra2': os.path.join(
        projpath,'io','cf-1998-2018','v06_states-LCOE-transloss-urbanedge',
        '{}','merra2','distance-station_urbanedge-mincost',''
    ).format(zonesource)}

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

lat = {'pv':'Latitude', 'wind':'latitude', 'merra2': 'lat'}
lon = {'pv':'Longitude', 'wind':'longitude', 'merra2': 'lon'}

#################
### PROCEDURE ###

###################
### Load input data

###### EIA transmission
##### First download from 
##### https://hifld-geoplatform.opendata.arcgis.com/datasets/electric-power-transmission-lines/data
filesin = {
    'Transmission': os.path.join(
        projpath,'in','Maps','HIFLD','Electric_Power_Transmission_Lines-shp'),
}
dftrans = gpd.read_file(filesin['Transmission']).to_crs({'init':'epsg:4326'})

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
dfurban_all = gpd.read_file(os.path.join(datapath,'Maps','Census','tl_2010_us_uac10'))

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
            ''.join(urbanclasses), voltcutoff, zonesource, zone),
        'wind': 'wtkhsds,every2,offset0,onshore-urban{}-trans{}-{}_{}.csv'.format(
            ''.join(urbanclasses), voltcutoff, zonesource, zone),
        'merra2': 'merra2-urban{}-trans{}-{}_{}.csv'.format(
            ''.join(urbanclasses), voltcutoff, zonesource, zone),
    }

    index_coords = {'pv':'psm3id', 'wind':'rowf_colf', 'merra2': 'latlon'}
    weightsfile = {
        'pv': os.path.join(
            projpath,'io','geo','developable-area','{}','nsrdb-icomesh9',
            'sitearea-water,parks,native,mountains,urban-{}_{}.csv'
        ).format(zonesource, zonesource, zone),
        'wind': os.path.join(
            projpath,'io','geo','developable-area','{}','wtk-hsds-every2',
            'sitearea-water,parks,native,mountains,urban-{}_{}.csv'
        ).format(zonesource, zonesource, zone),
        'merra2': os.path.join(
            projpath,'io','geo','developable-area','{}','merra2',
            'sitearea-water,parks,native,mountains,urban-{}_{}.csv'
        ).format(zonesource, zonesource, zone)
    }

    ########################################################################
    ### Loop over resources, calculate interconnection distances + costs ###

    for resource in resources:
        ### Make output folder
        os.makedirs(outpath[resource], exist_ok=True)

        ### Skip if it's done
        if os.path.exists(outpath[resource]+outfile[resource]) and (overwrite == False):
            continue

        ###### Load site dataframes
        polyweights = pd.read_csv(
            weightsfile[resource]
        )#.groupby(index_coords[resource]).sum().reset_index()
        
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
        for i in tqdm(dfsites.index, desc='{},{}'.format(zonesource,zone)):
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
        dfsites.to_csv(outpath[resource]+outfile[resource], index=False)



######################
#%% cfsim-lcoebins ###





#######################
### ARGUMENT INPUTS ###

zonesource = 'ba'
# zone = 'Central'
for zone in bas:

    resource = 'merra2'
    fulloutput = True
    maxbreaks = 10
    urban = 'edge'
    transcostmultiplier = 1
    height = 100

    ### new input - set timezone
    tz = 'Etc/GMT+6'

    ##############
    ### INPUTS ###

    ###### Include wind losses
    loss_system_wind = 0.15

    ### NEW: Transmission losses (1% per 100 miles)
    loss_distance = 0.01/1.60934/100

    ### Economic and simulation assumptions
    infile = os.path.join(projpath,'io','generator_fuel_assumptions.xlsx')
    defaults = zephyr.system.Defaults(infile)

    windcost = 'Wind_2030_mid'

    ###### Financial assumptions
    wacc_gen = 0.07
    wacc_trans = 0.04
    lifetime_gen = 30
    lifetime_trans = 50
    crf_gen = zephyr.system.crf(wacc=wacc_gen, lifetime=lifetime_gen)
    crf_trans = zephyr.system.crf(wacc=wacc_trans, lifetime=lifetime_trans)

    ### Set output file location and savenames
    distancepath = os.path.join(projpath,'io','cf-1998_2018','state','')
    outpath = os.path.join(
        projpath,'io','cf-1998_2018','{}','{}','').format(zonesource,resource)
    if resource in ['wind','merra2']:
        outpath = os.path.join(outpath, model.replace(':','|').replace('/','_'))

    ### Get region file
    dfregion = pd.read_csv(os.path.join(projpath,'io','regions.csv'), index_col=0)

    ### Get states to loop over
    states = dfregion.loc[dfregion[zonesource]==zone].index.values

    ###### Select wind/PV-specific parameters
    index_coords = 'latlon'
    weightsfile = {
        state: os.path.join(
            projpath,'io','geo','developable-area','state','merra2',
            'sitearea-water,parks,native,mountains,urban-{}_{}.csv'
        ).format('state', state)
        for state in states
    }
    distancefile = {
        state: os.path.join(
            distancepath,'merra2','distance-station_urban{}-mincost',
            'merra2-urbanU-trans230-{}_{}.csv'
        ).format(urban, 'state', state)
        for state in states
    }

    savename_cfmean = outpath+(
        'mean-merra2-{}-{}m-{:.0f}pctloss-{}-{}_{}.csv'
    ).format(model.replace(':','|').replace('/','_'), 
             height, loss_system_wind*100, windcost[5:], zonesource, zone)

    ### Make the generator
    gen = zephyr.system.Gen(windcost, defaults=defaults)

    ### Specify timeseries to simulate over
    years = list(range(1998,2019))

    #################
    ### PROCEDURE ###

    ### Set up output folders and filenames
    os.makedirs(outpath+'binned'+os.sep, exist_ok=True)

    savename_bins_lcoe = (
        savename_cfmean.replace('mean-',os.path.join('binned','cf-'))
        .replace('.csv','-NUMBREAKSlcoebins.csv'))
    savename_bins_area = (
        savename_cfmean.replace('mean-',os.path.join('binned','area-'))
        .replace('.csv','-NUMBREAKSlcoebins.csv'))
    print('\n'+savename_cfmean)
    sys.stdout.flush()

    ### Load site weights for the modeled zone, merging multiply-listed sites
    polyweights = pd.concat(
        {state: pd.read_csv(weightsfile[state]).groupby(index_coords).sum() 
         for state in states},
        axis=0, names=['state',index_coords]
    ).reset_index()
    ### Load distance dataframe
    dfdistance = pd.concat(
        {state: pd.read_csv(distancefile[state])[
            [index_coords,'cost_trans_annual','km_urban_fromstation','km_site_spur']]
         for state in states},
        axis=0, names=['state','extra']
    ).reset_index(level=0).reset_index(drop=True)

    ###### Don't need to simulate capacity factors
    ###### Load points to model

    ### Merge sites with calculated land area
    dfsites = dfcoords.merge(polyweights, on=index_coords, how='inner')
    
    ### Load the pre-simulated CFs
    dfout = (
        dfin[dfsites[['lat','lon']].drop_duplicates().values]
        .tz_localize('UTC').tz_convert(tz)
        .loc[str(years[0]):str(years[-1])]
        .copy())
    dfout.columns = ['{}_{}'.format(*x) for x in dfout.columns.values]

    ### Merge with dfsites
    if fulloutput:
        dfcf = dfout.mean().reset_index().rename(columns={'index':index_coords,0:'cf'})
    else:
        dfcf = dfout.reset_index().rename(columns={'index':index_coords,0:'cf'})
    dfcf.drop_duplicates(inplace=True)
    dfsites = dfsites.merge(dfcf, on=index_coords)

    ###### Incorporate distance costs, calculate LCOE
    ### Merge with dfdistance
    dfsites = dfsites.merge(dfdistance, on=['state',index_coords])

    ### Calculate LCOE including transmission interconnection cost
    def lcoe(row):
        out = (
            gen.cost_annual + gen.cost_fom + row['cost_trans_annual']
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
            gen.cost_annual + gen.cost_fom + row['cost_trans_annual']
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
    # for numbreaks in range(2, maxbreaks+1):
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
            # print('\n'+savename)
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
        # print('\n'+savename)
        dfoutput.to_csv(savename, header=True)
