import pandas as pd
import numpy as np
import sys, os, site, zipfile, math, time, json, io
import urllib, shapely, shutil, requests
from glob import glob
from urllib.error import HTTPError
from urllib.request import URLError
from http.client import IncompleteRead
from zipfile import BadZipFile
from tqdm import tqdm, trange
from warnings import warn

###########################
### IMPORT PROJECT PATH ###
import zephyr.settings
projpath = zephyr.settings.projpath
datapath = zephyr.settings.datapath
apikeys = zephyr.settings.apikeys
nsrdbparams = zephyr.settings.nsrdbparams

#####################
### Imports from zephyr
import zephyr.toolbox
import zephyr.io


#######################
### DICTS AND LISTS ###
#######################

isos = ['CAISO', 'ERCOT', 'MISO', 'PJM', 'NYISO', 'ISONE']

################
### DOWNLOAD ###
################

###############
### General use

def constructpayload(**kwargs):
    out = []
    for kwarg in kwargs:
        out.append('{}={}'.format(kwarg, kwargs[kwarg]))
    stringout = '&'.join(out)
    return stringout

def constructquery(urlstart, **kwargs):
    out = '{}{}'.format(urlstart, constructpayload(**kwargs))
    return out

def stampify(date, interval=pd.Timedelta('1H')):
    datetime = pd.Timestamp(date)
    if interval == pd.Timedelta('1H'):
        dateout = '{}{:02}{:02}T{:02}'.format(
            datetime.year, datetime.month, 
            datetime.day, datetime.hour)
    elif interval == pd.Timedelta('1D'):
        dateout = '{}{:02}{:02}'.format(
            datetime.year, datetime.month, 
            datetime.day)
    return dateout

def download_file_series(urlstart, urlend, fileseries, filepath, 
    overwrite=False, sleeptime=60, numattempts=200, seriesname=True):
    """
    Example
    -------
    You want to download a list of files at urls = [
        'http://www.test.com/foo001.csv', 'http://www.test.com/foo002.csv'].
    Then:
        urlstart = 'http://www.test.com/foo'
        urlend = '.csv'
        fileseries = ['001', '002']
    If you want the files to be named 'foo001.csv', use seriesname=False
    If you want the files to be named '001.csv', use seriesname=True
    """
    filepath = zephyr.toolbox.pathify(filepath, make=True)

    ### Make lists of urls, files to download, and filenames
    urls = [(urlstart + file + urlend) for file in fileseries]
    todownload = [os.path.basename(url) for url in urls]

    if seriesname == True:
        filenames = [os.path.basename(file) + urlend for file in fileseries]
    else:
        filenames = todownload

    ### Get the list of downloaded files
    downloaded = [os.path.basename(file) for file in glob(filepath + '*')]

    ### Remake the list if overwrite == False
    if overwrite == False:
        filestodownload = []
        urlstodownload = []
        fileseriesnames = []
        for i in range(len(filenames)):
            if filenames[i] not in downloaded:
                filestodownload.append(todownload[i])
                urlstodownload.append(urls[i])
                fileseriesnames.append(filenames[i])
    elif overwrite == True:
        filestodownload = todownload
        urlstodownload = urls
        fileseriesnames = filenames

    ### Download the files
    for i in trange(len(urlstodownload)):
        ### Attempt the download
        attempts = 0
        while attempts < numattempts:
            try:
                urllib.request.urlretrieve(
                    urlstodownload[i], filepath + fileseriesnames[i])
                break
            except (HTTPError, IncompleteRead, EOFError) as err:
                print(urlstodownload[i])
                print(filestodownload[i])
                print('Rebuffed on attempt # {} at {} by "{}".'
                      'Will retry in {} seconds.'.format(
                        attempts, zephyr.toolbox.nowtime(), err, sleeptime))
                attempts += 1
                time.sleep(sleeptime)


###########################
### Geographic manipulation

def rowlatlon2x(row):
    latrad = row['latitude'] * math.pi / 180
    lonrad = row['longitude'] * math.pi / 180
    x = math.cos(latrad) * math.cos(lonrad)
    return x

def rowlatlon2y(row):
    latrad = row['latitude'] * math.pi / 180
    lonrad = row['longitude'] * math.pi / 180
    y = math.cos(latrad) * math.sin(lonrad)
    return y

def rowlatlon2z(row):
    latrad = row['latitude'] * math.pi / 180
    z = math.sin(latrad)
    return z


###########################
### DOWNLOAD NSRDB DATA ###

def lonlat2wkt(lon, lat):
    return 'POINT({:+f}{:+f})'.format(lon, lat)

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

def postNSRDBsize(
    years,
    lonlats,
    attributes='ghi,dni,dhi,solar_zenith_angle,air_temperature,wind_speed',
    leap_day='true', 
    interval='30',
    norm=False):
    """
    Determine size of NSRDB POST request
    """
    numyears = len(years)
    numattributes = len(attributes.split(','))
    numintervals = sum([zephyr.toolbox.yearhours(year) * 60 / int(interval)
                        for year in years])
    numsites = len(lonlats)
    if norm:
        return numsites * numattributes * numyears * numintervals / 175000000
    return numsites * numattributes * numyears * numintervals

def lonlats2multipoint(lonlats):
    out = ['{}%20{}'.format(lonlat[0], lonlat[1]) for lonlat in lonlats]
    return 'MULTIPOINT({})'.format('%2C'.join(out))

def postNSRDBfiles(
    years, lonlats, psmversion=3,
    api_key=apikeys['nrel'],
    attributes='ghi,dni,dhi,solar_zenith_angle,air_temperature,wind_speed',
    leap_day='true', interval='30', utc='false'):
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
        names=names, utc=utc, leap_day=leap_day, interval=interval, 
        full_name=nsrdbparams['full_name'], email=nsrdbparams['email'],
        affiliation=nsrdbparams['affiliation'], reason=nsrdbparams['reason'],
        mailing_list=nsrdbparams['mailing_list']
    )
            
    headers = {
        'content-type': "application/x-www-form-urlencoded",
        'cache-control': "no-cache"
    }
    
    response = requests.request("POST", url, data=payload, headers=headers)
    
    output = response.text
    print(output[output.find("errors"):output.find("inputs")], 
      '\n', output[output.find("outputs"):])

def downloadNSRDBfile(
    lat, lon, year, filepath=None,
    nodename='default', filetype='.gz',
    attributes='ghi,dni,dhi,solar_zenith_angle,air_temperature,wind_speed', 
    leap_day='true', interval='30', utc='false', psmversion=3,
    write=True, return_savename=False, urlonly=False):
    '''
    Downloads file from NSRDB.
    NOTE: PSM v2 doesn't include 'surface_albedo' attribute.
    NOTE: For PSM v3, can use either v2 or v3 version of attribute labels.
    
    Full list of attributes for PSM v2:
    attributes=(
        'dhi,dni,ghi,clearsky_dhi,clearsky_dni,clearsky_ghi,cloud_type,' + 
        'dew_point,surface_air_temperature_nwp,surface_pressure_background,' +
        'surface_relative_humidity_nwp,solar_zenith_angle,' +
        'total_precipitable_water_nwp,wind_direction_10m_nwp,' +
        'wind_speed_10m_nwp,fill_flag')

    Full list of attributes for PSM v3:
    attributes=(
        'dhi,dni,ghi,clearsky_dhi,clearsky_dni,clearsky_ghi,cloud_type,' + 
        'dew_point,air_temperature,surface_pressure,' +
        'relative_humidity,solar_zenith_angle,' +
        'total_precipitable_water,wind_direction,' +
        'wind_speed,fill_flag,surface_albedo')

    Parameters
    ----------
    filename: string
    nodename: string
    lat: numeric
    lon: numeric
    year: numeric
    
    Returns
    -------
    if write == True: # default
        '.csv' file if filetype == '.csv', or '.gz' file if filetype == '.gz'
    if return_savename == False: pandas.DataFrame # default
    if return_savename == True: (pandas.DataFrame, savename) # type(savename) = str
    '''
    ### Check inputs
    if filetype not in ['.csv', '.gz']:
        raise Exception("filetype must be '.csv' or '.gz'.")

    if write not in [True, False]:
        raise Exception('write must be True or False.')

    if return_savename not in [True, False]:
        raise Exception('return_savename must be True or False.')

    ### Set psmversion to 3 if year is 2016 (since v2 doesn't have 2016)
    if year in [2016, '2016', 2016.]:
        psmversion = 3

    ### Remove solar_zenith_angle if year == 'tmy'
    if year == 'tmy':
        attributes = attributes.replace('solar_zenith_angle,','')
        attributes = attributes.replace('solar_zenith_angle','')

    year = str(year)

    ### Set url based on version of PSM
    if psmversion in [2, '2', 2.]:
        urlbase = 'http://developer.nrel.gov/api/solar/nsrdb_0512_download.csv?'
        attributes = convertattributes_3to2(attributes)
    elif psmversion in [3, '3', 3.]:
        urlbase = 'https://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?'
        attributes = convertattributes_2to3(attributes)
    else:
        raise Exception("Invalid psmversion; must be 2 or 3")
        
    url = (
        urlbase + querify(
            api_key=apikeys['nrel'], full_name=nsrdbparams['full_name'], 
            email=nsrdbparams['email'], affiliation=nsrdbparams['affiliation'], 
            reason=nsrdbparams['reason'], mailing_list=nsrdbparams['mailing_list'],
            wkt=lonlat2wkt(lon, lat), names=year, attributes=attributes,
            leap_day=leap_day, utc=utc, interval=interval))
    if urlonly:
        return url
    
    try:
        df = pd.read_csv(url)
    except HTTPError as err:
        #################### CHANGED 20200209
        # print(url)
        # print(err)
        # raise HTTPError
        try:
            infile = requests.get(url)
            df = pd.read_csv(io.StringIO(infile.text))
        except HTTPError as err:
            print(url)
            print(err)
            raise HTTPError
        ####################
    df = df.fillna('')
    columns = df.columns

    if write == True:
        if len(filepath) != 0 and filepath[-1] != '/':
            filepath = filepath + '/'

        if nodename in [None, 'default']:
            savename = (filepath + df.loc[0,'Location ID'] + '_' + 
                df.loc[0,'Latitude'] + '_' + 
                df.loc[0,'Longitude'] + '_' + year + filetype)
        else:
            # savename = str(filepath + nodename + '-' + year + filetype)
            savename = os.path.join(
                filepath, '{}-{}{}'.format(nodename, year, filetype))

        ### Determine number of columns to write (used to always write 11)
        numcols = max(len(attributes.split(','))+5, 11)

        ### Write the output
        if filetype == '.gz':
            df.to_csv(savename, columns=columns[0:numcols], index=False, 
                      compression='gzip')
        elif filetype == '.csv':
            df.to_csv(savename, columns=columns[0:numcols], index=False)
    
        if return_savename == True:
            return df, savename
        else:
            return df
    return df


def downloadNSRDBfiles(
    dfin, years, nsrdbpath, namecolumn=None,
    resolution=None, latlonlabels=None,
    filetype='.gz', psmversion=3,
    attributes='ghi,dni,dhi,solar_zenith_angle,air_temperature,wind_speed',
    wait=0.5, maxattempts=200,):
    """
    """
    ###### Set defaults
    ### Convert attributes if necessary
    if psmversion in [2, '2', 2.]:
        attributes = convertattributes_3to2(attributes)
    elif psmversion in [3, '3', 3.]:
        attributes = convertattributes_2to3(attributes)

    ### Get lat, lon labels
    if ('latitude' in dfin.columns) and ('longitude' in dfin.columns):
        latlabel, lonlabel = 'latitude', 'longitude'
    elif ('Latitude' in dfin.columns) and ('Longitude' in dfin.columns):
        latlabel, lonlabel = 'Latitude', 'Longitude'
    elif ('lat' in dfin.columns) and ('lon' in dfin.columns):
        latlabel, lonlabel = 'lat', 'lon'
    elif ('lat' in dfin.columns) and ('long' in dfin.columns):
        latlabel, lonlabel = 'lat', 'long'
    elif ('x' in dfin.columns) and ('y' in dfin.columns):
        latlabel, lonlabel = 'x', 'y'
    else:
        latlabel, lonlabel = latlonlabels[0], latlonlabels[1]

    ### Loop over years
    for year in years:
        ### Set defaults
        if (resolution == None) and (year == 'tmy'):
            resolution = 60
        elif (resolution == None) and (type(year) == int):
            resolution = 30

        ### Set up output folder
        outpath = nsrdbpath+'{}/{}min/'.format(year, resolution)
        os.makedirs(outpath, exist_ok=True)

        ### Make list of files downloaded so far
        downloaded = glob(outpath + '*') ## or os.listdir(outpath)
        downloaded = [os.path.basename(file) for file in downloaded]

        ### Make list of files to download
        if 'latlonindex' in dfin.columns:
            dfin.drop_duplicates('latlonindex', inplace=True)
            dfin['name'] = dfin['latlonindex'].copy()
            dfin['file'] = dfin['latlonindex'].map(
                lambda x: '{}{}-{}{}'.format(outpath, x, year, filetype))
        elif namecolumn is not None:
            dfin['name'] = dfin[namecolumn].copy()
            dfin['file'] = dfin[namecolumn].map(
                lambda x: '{}{}-{}{}'.format(outpath, x, year, filetype))
        elif namecolumn is None:
            dfin['name'] = None
            dfin['file'] = None

        dfin['todownload'] = dfin['file'].map(
            lambda x: os.path.basename(x) not in downloaded)

        dftodownload = dfin[dfin['todownload']].reset_index(drop=True)

        print('{}: {} done, {} to download'.format(
            year, len(downloaded), len(dftodownload)))

        ### Loop over locations
        for i in trange(len(dftodownload)):
            attempts = 0
            while attempts < maxattempts:
                try:
                    downloadNSRDBfile(
                        lat=dftodownload[latlabel][i],
                        lon=dftodownload[lonlabel][i],
                        year=year,
                        filepath=outpath,
                        nodename=dftodownload['name'][i],
                        interval=str(resolution),
                        psmversion=psmversion,
                        attributes=attributes)
                    break
                except HTTPError as err:
                    if str(err) in ['HTTP Error 504: Gateway Time-out', 
                                    'HTTP Error 500: Internal Server Error']:
                        print(('Rebuffed on attempt # {} at {} by "{}". '
                               'Retry in 5 minutes.').format(
                                    attempts, zephyr.toolbox.nowtime(), err))
                        attempts += 1
                        time.sleep(5 * 60)
                    else:
                        print(('Rebuffed on attempt # {} at {} by "{}". '
                               'Retry in {} hours.').format(
                                    attempts, zephyr.toolbox.nowtime(), err, wait))
                        attempts += 1
                        time.sleep(wait * 60 * 60)                  
            if attempts >= maxattempts:
                print("Something must be wrong. No response after {} attempts.".format(
                    attempts))


def downloadNSRDBfiles_iso(year, resolution=None, 
    isos=['CAISO', 'ERCOT', 'MISO', 'NYISO', 'PJM', 'ISONE'],
    filetype='.gz', wait=0.5, psmversion=3, 
    attributes='ghi,dni,dhi,solar_zenith_angle,air_temperature,wind_speed'):
    """
    """
    # nodemap = {
    #     'CAISO': os.path.join(projpath, 'CAISO/io/caiso-node-latlon.csv'),
    #     'ERCOT': os.path.join(projpath, 'ERCOT/io/ercot-node-latlon.csv'),
    #     'MISO':  os.path.join(projpath, 'MISO/in/miso-node-map.csv'),
    #     'PJM':   os.path.join(projpath, 'PJM/io/pjm-pnode-latlon-uniquepoints.csv'),
    #     'NYISO': os.path.join(projpath, 'NYISO/io/nyiso-node-latlon.csv'),
    #     'ISONE': os.path.join(projpath, 'ISONE/io/isone-node-latlon.csv') 
    # }[iso]
    ### Set defaults
    if (resolution == None) and (year == 'tmy'):
        resolution = 60
    elif (resolution == None) and (type(year) == int):
        resolution = 30

    ### Convert attributes if necessary
    if psmversion in [2, '2', 2.]:
        attributes = convertattributes_3to2(attributes)
    elif psmversion in [3, '3', 3.]:
        attributes = convertattributes_2to3(attributes)

    for iso in isos:
        nodemap = projpath + '{}/io/{}-node-latlon.csv'.format(
            iso.upper(), iso.lower())
        ### Load node key
        dfin = pd.read_csv(nodemap)
        dfin.rename(
            columns={'name': 'node', 'pnodename': 'node', 'ptid': 'node'},
            inplace=True)

        ### Set up output folder
        outpath = os.path.join(projpath, '{}/in/NSRDB/{}/{}min/'.format(
            iso, year, resolution))
        if not os.path.isdir(outpath):
            os.makedirs(outpath)

        ### Make list of files downloaded so far
        downloaded = glob(outpath + '*') ## or os.listdir(outpath)
        downloaded = [os.path.basename(file) for file in downloaded]

        ### Make list of files to download
        if 'latlonindex' in dfin.columns:
            dfin.drop_duplicates('latlonindex', inplace=True)
            dfin['name'] = dfin['latlonindex'].copy()
            dfin['file'] = dfin['latlonindex'].map(
                lambda x: '{}{}-{}{}'.format(outpath, x, year, filetype))
        else:
            dfin['name'] = dfin['node'].copy()
            dfin['file'] = dfin['node'].map(
                lambda x: '{}{}-{}{}'.format(outpath, x, year, filetype))

        dfin['todownload'] = dfin['file'].map(
            lambda x: os.path.basename(x) not in downloaded)

        dftodownload = dfin[dfin['todownload']].reset_index(drop=True)

        print('{} {}: {} done, {} to download'.format(
            iso.upper(), year, len(downloaded), len(dftodownload)))

        for i in trange(len(dftodownload)):
            attempts = 0
            while attempts < 200:
                try:
                    downloadNSRDBfile(
                        lat=dftodownload['latitude'][i],
                        lon=dftodownload['longitude'][i],
                        year=year,
                        filepath=outpath,
                        nodename=str(dftodownload['name'][i]),
                        interval=str(resolution),
                        psmversion=psmversion,
                        attributes=attributes)
                    break
                except HTTPError as err:
                    if str(err) in ['HTTP Error 504: Gateway Time-out', 
                                    'HTTP Error 500: Internal Server Error']:
                        print(('Rebuffed on attempt # {} at {} by "{}". '
                               'Retry in 5 minutes.').format(
                                    attempts, zephyr.toolbox.nowtime(), err))
                        attempts += 1
                        time.sleep(5 * 60)
                    else:
                        print(('Rebuffed on attempt # {} at {} by "{}". '
                               'Retry in {} hours.').format(
                                    attempts, zephyr.toolbox.nowtime(), err, wait))
                        attempts += 1
                        time.sleep(wait * 60 * 60)                  
            if attempts >= 200:
                print("Something must be wrong. No response after {} attempts.".format(
                    attempts))

def postNSRDBfiles_iso(year, yearkey, resolution=None, 
    isos=['CAISO', 'ERCOT', 'MISO', 'NYISO', 'PJM', 'ISONE'],
    filetype='.gz', wait=3, psmversion=2, chunksize=1000,
    attributes='ghi,dni,dhi,solar_zenith_angle,air_temperature,wind_speed'):
    """
    Notes
    -----
    * This function can only be run after all NSRDB files for a given year have been
    downloaded using downloadNSRDBfiles(), as the POST request scrambles the 
    node-to-NSRDBid correspondence.
    * The files will be sent to settings.nsrdbparams['email']. Need to download and unzip them.
    Default location for unzipped files is projpath+'USA/in/NSRDB/nodes/{}/'.format(year).
    """
    # nodemap = {
    #     'CAISO': os.path.join(projpath, 'CAISO/io/caiso-node-latlon.csv'),
    #     'ERCOT': os.path.join(projpath, 'ERCOT/io/ercot-node-latlon.csv'),
    #     'MISO':  os.path.join(projpath, 'MISO/in/miso-node-map.csv'),
    #     'PJM':   os.path.join(projpath, 'PJM/io/pjm-pnode-latlon-uniquepoints.csv'),
    #     'NYISO': os.path.join(projpath, 'NYISO/io/nyiso-node-latlon.csv'),
    #     'ISONE': os.path.join(projpath, 'ISONE/io/isone-node-latlon.csv') 
    # }[iso]
    ### Set defaults
    if (resolution == None) and (year == 'tmy'):
        resolution = 60
    elif (resolution == None) and (type(year) == int):
        resolution = 30

    ### Convert attributes if necessary
    if psmversion in [2, '2', 2.]:
        attributes = convertattributes_3to2(attributes)
    elif psmversion in [3, '3', 3.]:
        attributes = convertattributes_2to3(attributes)

    ### Make dataframe of nodes from all ISOs
    dictnodes = {}
    for iso in isos:
        ### Load node key
        nodemap = projpath + '{}/io/{}-node-latlon.csv'.format(
            iso.upper(), iso.lower())
        dfin = pd.read_csv(nodemap)
        dfin.rename(
            columns={'name': 'node', 'pnodename': 'node', 'ptid': 'node'},
            inplace=True)
        
        inpath = os.path.join(projpath, '{}/in/NSRDB/{}/{}min/'.format(
            iso, yearkey, resolution))
        
        ### Make list of files downloaded so far
        downloaded = glob(inpath + '*') ## or os.listdir(inpath)

        ### Make list of files to download
        if 'latlonindex' in dfin.columns:
            dfin.drop_duplicates('latlonindex', inplace=True)
            dfin['name'] = dfin['latlonindex'].copy()
            dfin['file'] = dfin['latlonindex'].map(
                lambda x: '{}{}-{}{}'.format(inpath, x, yearkey, filetype))
        else:
            dfin['name'] = dfin['node'].copy()
            dfin['file'] = dfin['node'].map(
                lambda x: '{}{}-{}{}'.format(inpath, x, yearkey, filetype))

        dfin['todownload'] = dfin['file'].map(
            lambda x: x not in downloaded)

        dictnodes[iso] = dfin.copy()

    dfnodes = pd.concat(dictnodes)

    ### Identify locations to include in query
    nsrdbids, nsrdblats, nsrdblons = [], [], []

    for file in tqdm(dfnodes['file'].values):
        df = pd.read_csv(file, nrows=1)
        nsrdbids.append(df['Location ID'][0])
        nsrdblats.append(df['Latitude'][0])
        nsrdblons.append(df['Longitude'][0])

    dfnodes['NSRDBid'] = nsrdbids
    dfnodes['NSRDBlat'] = nsrdblats
    dfnodes['NSRDBlon'] = nsrdblons

    dfnodes = dfnodes.reset_index(level=0).rename(columns={'level_0': 'iso'})
    dfnodes.reset_index(drop=True, inplace=True)

    ### Save dfnodes for use in unpacking
    if not os.path.exists(projpath+'USA/io/'):
        os.makedirs(projpath+'USA/io/')
    dfnodes.to_csv(projpath+'USA/io/nsrdbnodekey-{}.csv'.format(yearkey), index=False)

    ### Post NSRDB requests in 400-unit chunks, dropping duplicate NSRDBids
    dftodownload = dfnodes.drop_duplicates('NSRDBid').copy()
    lonlatstodownload = list(zip(dftodownload['NSRDBlon'], dftodownload['NSRDBlat']))
    for i in range(0,len(lonlatstodownload), chunksize):
        print(i)
        postNSRDBfiles(years=[year], lonlats=lonlatstodownload[i:i+chunksize],
                       psmversion=psmversion, attributes=attributes)
        time.sleep(wait)

def unpackpostNSRDBfiles_iso(year, yearkey, postpath=None,
    isos=['CAISO', 'ERCOT', 'MISO', 'NYISO', 'PJM', 'ISONE'],
    resolution=None, filetypeout='.gz',
    attributes='ghi,dni,dhi,solar_zenith_angle,air_temperature,wind_speed'):
    """
    Notes
    -----
    * This function can only be run after postNSRDBfiles_iso().
    * Default location for unzipped posted files is 
    projpath+'USA/in/NSRDB/nodes/{}/'.format(year).
    * Defualt location for dfnodes is 
    projpath+'USA/io/nsrdbnodekey-{}.csv'.format(yearkey)
    """
    ### Set defaults, if necessary
    if postpath==None:
        postpath = projpath+'USA/in/NSRDB/nodes/{}/'.format(year)
    if (resolution == None) and (year == 'tmy'):
        resolution = 60
    elif (resolution == None) and (type(year) == int):
        resolution = 30

    compression = 'gzip'
    if filetypeout not in ['gzip', '.gz']:
        compression = None

    ### Load dfnodes from default location
    dfnodes = pd.read_csv(projpath+'USA/io/nsrdbnodekey-{}.csv'.format(yearkey))

    ### Get downloaded file list
    postfiles = glob(postpath + '*')

    ### Extract parameters from filename
    def fileparams(filepath, filetype='.csv'):
        filename = os.path.basename(filepath)
        nsrdbid = filename[:filename.find('_')]
        lat = filename[filename.find('_')+1:filename.find('_-')]
        lon = filename[filename.find('_-')+1:filename.find(filetype)][:-5]
        year = filename[-(len(filetype)+4):-len(filetype)]
        return nsrdbid, lat, lon, year

    dfpostfiles = pd.DataFrame(postfiles, columns=['filepath'])
    dfpostfiles['nsrdbid'] = dfpostfiles.filepath.apply(lambda x: fileparams(x)[0])
    dfpostfiles['lat'] = dfpostfiles.filepath.apply(lambda x: fileparams(x)[1])
    dfpostfiles['lon'] = dfpostfiles.filepath.apply(lambda x: fileparams(x)[2])
    dfpostfiles['year'] = dfpostfiles.filepath.apply(lambda x: fileparams(x)[3])

    latlon2postfilepath = dict(zip(
        [tuple(row) for row in dfpostfiles[['lat', 'lon']].values],
        list(dfpostfiles['filepath'].values)
    ))

    ### Add filepath for POST files to dfnodes
    dfnodes['NSRDBfile'] = dfnodes.apply(
        lambda row: latlon2postfilepath[(str(row['NSRDBlat']), str(row['NSRDBlon']))],
        axis=1)
    dfnodes['NSRDBfilebase'] = dfnodes.NSRDBfile.map(os.path.basename)

    ### Create output file folders
    for iso in isos:
        pathout = projpath+'{}/in/NSRDB/{}/{}min/'.format(iso, year, resolution)
        if not os.path.isdir(pathout):
            os.makedirs(pathout)

    ### Save corresponding files with correct nodenames
    for i in trange(len(dfnodes)):
        ### Set filenames
        fileout = os.path.join(
            projpath,
            '{}/in/NSRDB/{}/{}min/'.format(dfnodes.loc[i,'iso'], year, resolution),
            '{}-{}{}'.format(dfnodes.loc[i, 'name'], year, filetypeout)
        )
        filein = dfnodes.loc[i,'NSRDBfile']
        
        ### Load, manipulate, and resave file
        dfin = pd.read_csv(filein, low_memory=False)
        df = dfin.fillna('')
        
        columns = df.columns
        numcols = max(len(attributes.split(','))+5, 11)
        
        df.to_csv(fileout, 
                  columns=columns[:numcols],
                  index=False, compression=compression)

