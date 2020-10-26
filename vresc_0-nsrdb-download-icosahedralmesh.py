###############
#%% IMPORTS ###

import pandas as pd
import numpy as np
import sys, os, site, math, time, requests
from glob import glob
from tqdm import tqdm, trange

##########
### INPUTS

params = {
    ### Use '+' for spaces
    'api_key': '>>> input here',
    'full_name': 'your+name',
    'email': 'your@email.com', ### where you'll get globus emails with the data
    'affiliation': 'your+affiliation',
    'reason': 'your+reason',
    'mailing_list': 'true',
}

years = [2018]
psmversion = 3
chunksize = 1000
chunklistin = None
pointsfile = 'io/usa-points-icomesh-x[-atan(invPHI)+90-lat-11]-z[90+lon]-9subdiv.csv'

#############
### FUNCTIONS

def lonlats2multipoint(lonlats):
    out = ['{}%20{}'.format(lonlat[0], lonlat[1]) for lonlat in lonlats]
    return 'MULTIPOINT({})'.format('%2C'.join(out))

def lonlats2wkt(lonlats):
    out = ['{}%20{}'.format(lonlat[0], lonlat[1]) for lonlat in lonlats]
    return 'MULTIPOINT({})'.format('%2C'.join(out))

def querify(**kwargs):
    out = ['{}={}'.format(key, kwargs[key]) for key in kwargs]
    return '&'.join(out)

def convertattributes_2to3(attributes):
    attributes_2to3 = {
        'surface_air_temperature_nwp': 'air_temperature',
        'surface_pressure_background': 'surface_pressure',
        'surface_relative_humidity_nwp': 'relative_humidity',
        'total_precipitable_water_nwp': 'total_precipitable_water',
        'wind_direction_10m_nwp': 'wind_direction',
        'wind_speed_10m_nwp': 'wind_speed',
    }
    attributes_in = attributes.split(',')
    attributes_out = [attributes_2to3.get(attribute, attribute)
                      for attribute in attributes_in]
    return ','.join(attributes_out)

def convertattributes_3to2(attributes):
    attributes_3to2 = {
        'air_temperature': 'surface_air_temperature_nwp',
        'surface_pressure': 'surface_pressure_background',
        'relative_humidity': 'surface_relative_humidity_nwp',
        'total_precipitable_water': 'total_precipitable_water_nwp',
        'wind_direction': 'wind_direction_10m_nwp',
        'wind_speed': 'wind_speed_10m_nwp',
    }
    attributes_in = attributes.split(',')
    attributes_out = [attributes_3to2.get(attribute, attribute)
                      for attribute in attributes_in]
    return ','.join(attributes_out)

### POST request to download NSRDB files using NSRDB PSM V3
def postNSRDBfiles(
    years,
    lonlats,
    psmversion=3,
    api_key=params['api_key'],
    attributes=(
        'ghi,dni,dhi,solar_zenith_angle,'
        + 'air_temperature,wind_speed'),
    leap_day='true', 
    interval='30', 
    utc='false', 
    full_name=params['full_name'], 
    reason=params['reason'], 
    affiliation=params['affiliation'], 
    email=params['email'], 
    mailing_list=params['mailing_list'],):
    """
    """

    ### Set url based on version of PSM
    if psmversion in [2, '2', 2.]:
        url = 'http://developer.nrel.gov/api/solar/nsrdb_0512_download.json?api_key={}'.format(
            api_key)
        attributes = convertattributes_3to2(attributes)
    elif psmversion in [3, '3', 3.]:
        url = 'http://developer.nrel.gov/api/solar/nsrdb_psm3_download.json?api_key={}'.format(
            api_key)
        attributes = convertattributes_2to3(attributes)
    else:
        raise Exception("Invalid psmversion; must be 2 or 3")

    names = ','.join([str(year) for year in years])
    wkt = lonlats2multipoint(lonlats)

    payload = querify(
        wkt=wkt, attributes=attributes, 
        names=names, utc=utc, leap_day=leap_day,
        interval=interval, full_name=full_name, email=email,
        affiliation=affiliation, reason=reason,
        mailing_list=mailing_list
    )
            
    headers = {
        'content-type': "application/x-www-form-urlencoded",
        'cache-control': "no-cache"
    }
    
    response = requests.request("POST", url, data=payload, headers=headers)
    
    output = response.text
    print(output[output.find("errors"):output.find("inputs")], 
      '\n', output[output.find("outputs"):])

### Determine size of request
def postNSRDBsize(
    years,
    lonlats,
    attributes=(
        'ghi,dni,dhi,solar_zenith_angle,'
        + 'surface_air_temperature_nwp,wind_speed_10m_nwp'),
    leap_day='true', 
    interval='30',
    norm=False):
    """
    """
    def yearhours(year):
        if year == 'tmy':
            hours = 365*24
        elif year % 4 != 0:
            hours = 365*24
        elif year % 100 != 0:
            hours = 366*24
        elif year % 400 != 0:
            hours = 365*24
        else:
            hours = 366*24
        return hours

    numyears = len(years)
    numattributes = len(attributes.split(','))
    numintervals = sum([yearhours(year) * 60 / int(interval)
                        for year in years])
    numsites = len(lonlats)
    if norm:
        return numsites * numattributes * numyears * numintervals / 175000000
    return numsites * numattributes * numyears * numintervals


#############
### PROCEDURE

def main(years, chunksize, chunklistin, pointsfile):
    ### Loop over years
    for year in years:
        print('##########################')
        print('########## {} ##########'.format(year))
        print('##########################')
        ### Identify list of points to download
        pointlist = pd.read_csv(pointsfile).values

        ### Download points from NSRDB in chunksize-unit chunks
        if chunklistin == None:
            chunklist = range(0,len(pointlist), chunksize)
        else:
            chunklist = chunklistin

        for i in chunklist:
            print(i)
            postNSRDBfiles(years=[year], lonlats=pointlist[i:i+chunksize],
                           psmversion=psmversion)
            time.sleep(3)
    
if __name__ == '__main__':
    main(years, chunksize, chunklistin, pointsfile)
