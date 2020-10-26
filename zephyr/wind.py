import pandas as pd
import numpy as np
import os, sys, site, math, time, pickle, gzip
from glob import glob
from tqdm import tqdm, trange
from warnings import warn

import xarray as xr

###########################
### IMPORT PROJECT PATH ###
import zephyr.settings
projpath = zephyr.settings.projpath
datapath = zephyr.settings.datapath
extdatapath = zephyr.settings.extdatapath

#####################
### Imports from zephyr
import zephyr.toolbox
import zephyr.data
import zephyr.io
import zephyr.pv


###############
### CLASSES ###

class WindSystem:
    """
    """
    def __init__(self,
                 height=100,
                 powercurve_source='wtk',
                 powercurve_model='wtk',
                 density_st=1.225,
                 loss_system=0.05,
                 number_of_turbines=8,
                 hystwindow=5,
                 humidity=None,
                 orders=4,
                 pc_vmax=None,
                 key=None,
                ):
        self.gentype = 'wind'
        self.height = height
        self.powercurve_source = powercurve_source
        self.powercurve_model = powercurve_model
        self.density_st = density_st
        self.loss_system = loss_system
        self.number_of_turbines = number_of_turbines
        self.hystwindow = hystwindow
        self.humidity = humidity
        self.orders = orders
        self.pc_vmax = pc_vmax
        self.key = key

    def sim(self, site_id, 
            source='tech', drive='RAID', 
            nans='raise'):
        """
        Simulate output using WTK meteorological data
        """
        return windsim(
            site_id=site_id, height=self.height, source=source,
            powercurve_source=self.powercurve_source, 
            powercurve_model=self.powercurve_model,
            key=self.key, density_std=self.density_std, 
            loss_system=self.loss_system,
            number_of_turbines=self.number_of_turbines, 
            drive=drive, humidity=self.humidity, 
            hystwindow=self.hystwindow, orders=self.orders,
            pv_vmax=self.pc_vmax, projpath=projpath, nans=nans)


#################
### FUNCTIONS ###


###########
### Physics

def specific_humidity_to_vapor_pressure(q, pressure,
    R_dryair=287.058, R_wv=461.495):
    """
    Inputs
    ------
    q: specific humidity in [kg/kg]
    pressure: total air pressure in Pa
    R_dryair, R_wv: [J kg^-1 K^-1]

    ('https://earthscience.stackexchange.com/questions/2360/'
     'how-do-i-convert-specific-humidity-to-relative-humidity')
    """
    return q * pressure * R_dryair / R_wv


def density_dry(pressure, temperature, R_dryair=287.058):
    """
    """
    return pressure/R_dryair/temperature

def p_wv_sat_buck(T):
    """
    Determines the saturation vapor pressure of water from the temperature
    * input units: Kelvin
    * output units: Pa
    
    https://en.wikipedia.org/wiki/Vapour_pressure_of_water
    """
    t = T - 273.15
    return 611.21 * np.exp ((18.678 - t / 234.5) * (t / (257.14 + t)))

def p_wv_sat_tetens(T):
    """
    Determines the saturation vapor pressure of water from the temperature
    * input units: Kelvin
    * output units: Pa
    
    https://en.wikipedia.org/wiki/Vapour_pressure_of_water
    """
    t = T - 273.15
    return 610.78 * np.exp ((17.27 * t) / (t + 237.3))

def p_wv(humidity, T):
    """
    Determines the partial pressure of water vapor from the 
    relative humidity and temperature
    
    units
    -----
    humidity: percent [%]
    T:        Kelvin  [K]
    output:   Pascal  [Pa]
    """
    # return humidity * p_wv_sat_buck(T)
    return 0.01 * humidity * p_wv_sat_tetens(T)

def density(pressure, temperature, humidity=0, 
    R_dryair=287.058, R_wv=461.495):
    """
    """
    p_wv_val = p_wv(humidity, temperature)
    density_total = (
        ((pressure - p_wv_val) / R_dryair / temperature)
        + (p_wv_val / R_wv / temperature))
    return density_total


#################
### Turbine model

def hyst(x, th_lo, th_hi, initial=False):
    """
    Hysteresis below cutout speed

    Source
    ------
    ('https://stackoverflow.com/questions/23289976/'
     'how-to-find-zero-crossings-with-hysteresis')
    """
    hi = x >= th_hi
    lo_or_hi = (x <= th_lo) | hi
    ind = np.nonzero(lo_or_hi)[0]
    if not ind.size: # prevent index error if ind is empty
        return np.zeros_like(x, dtype=bool) | initial
    cnt = np.cumsum(lo_or_hi) # from 0 to len(x)
    return np.where(cnt, hi[ind[cnt-1]], initial)

def getpowercurve(source='wtk', model=None, normalize=True):
    """
    """
    sources = {'database':'twp'}
    file = projpath+'Wind/in/powercurves/dfpc_{}_clean.p'.format(
        sources.get(source,source))
    
    # with open(file, 'rb') as p:
    #     df = pickle.load(p)
    try:
        df = pd.read_pickle(file)
    except FileNotFoundError:
        df = pd.read_csv(file.replace('.p','.csv'),index_col=0,squeeze=False)
        
    ### Flexibility for WTK
    if source == 'wtk':
        models = {
            1: 'WTKclass1', 2: 'WTKclass2', 3: 'WTKclass3', 
            '1': 'WTKclass1', '2': 'WTKclass2', '3': 'WTKclass3', 
            'off': 'WTKoffshore', 'offshore': 'WTKoffshore'}
        model = models.get(model, model)
    
    if normalize == True:
        df = df / df.max()
    
    if model is not None:
        if model in df.columns:
            return df[model]
        else:
            warn('Could not find {} in columns; returning all'.format(model))
            return df
    else:
        return df

def get_powercurve_dict(source='wtk', model=None, orders=4, 
    pc_vmax=50, pc_vmax_original=35,
    normalize=True, returncutout=True):
    """
    Get powercurve dictionary for lookup

    Inputs
    ------
    orders: integer degree of interpolation; round speed to nearest 10^-order
    pc_vmax: windspeed in m/s to extend power curve to
    """
    powercurve = getpowercurve(source=source, model=model, normalize=normalize)

    ### Extend the power curve to higher wind speeds
    for i in range(pc_vmax_original+1, pc_vmax+1):
        powercurve.loc[i] = 0

    ### Get cutout speed
    cutout = powercurve.loc[powercurve>0].index.max()

    ### Create linearly-interpolated power curve
    xout = np.linspace(0, pc_vmax, 10**orders*pc_vmax+1)
    yout = np.interp(x=xout, 
                     xp=powercurve.index.values,
                     fp=powercurve.values)
    pcmid = pd.Series(yout, xout)

    ### Round off the index
    pcmid.index = np.around(pcmid.index, orders)

    ### Set power output above cutout speed to zero
    pcmid.loc[pcmid.index.map(lambda x: x > cutout)] = 0

    ### Make lookup dict for converting wind speed to power
    dictcf = dict(zip(pcmid.index.values, pcmid.values))

    if returncutout == True:
        return dictcf, cutout
    else:
        return dictcf

def wakeloss_n(n):
    """
    NREL 2014 - Validation of Power Output for the WIND Toolkit
    pg. 8
    """
    assert n >= 1, "n must be >= 1"
    if n > 8:
        warn('number of turbines > 8; wtk only defines in range [1,8]')
    return 1 - 0.05 * (n - 1) / 7

def wakeloss(site_id, key=None):
    ### Load the WTK metadata df if necessary
    if key is None:
        key = pd.read_csv(projpath+'Wind/io/wtk_site_metadata_onshore.csv')
    ### Get the number of turbines at the site
    try:
        n = int(key.loc[key.site_id==site_id, 'capacity'] / 2)
    except AttributeError:
        n = int(key.loc[site_id, 'capacity'] / 2)    
    ### Return the wind speed derate
    return wakeloss_n(n)


##############
### Simulation

def getWINDnode_fromkey(isonode, key, year, 
    resolution=60, 
    yearout='default',
    timezoneout='default',
    drive='external',
    output='df',
    loss_system=0.):
    """

    To do:
    * Incorporate timezone into key file
    * Rewrite to take either isonode and key, or filepath ('248/124476.nc')
        * key='default'
        * isonode --> site

    Inputs
    ------
    isonode: str, ISO:node (example: 'ERCOT:ROANSPRARE')
    key: pd.DataFrame
    year: 'all' or int in range(2007,2014)
    resolution: int in [5, 10, 15, 20, 30, 60]
    timezoneout: 'default' or int in [-5, -6, -7, -8] for U.S.

    Returns
    -------
    """
    ### Check inputs
    if type(key) != pd.core.frame.DataFrame:
        raise Exception("type(key) must be pd.DataFrame.")
    if resolution not in [5, 10, 15, 20, 30, 60]:
        raise Exception('resolution must be in [5, 10, 15, 20, 30, 60]')
    if year not in [2007, 2008, 2009, 2010, 2011, 2012, 2013, 'all', 'full', None]:
        raise Exception("year must be in range(2007,2014),'all','full', or None.")
    if output.lower() not in ['df', 'dataframe', 
            'wind_speed', 'wind_direction', 'power', 
            'density', 'temperature', 'pressure', 'cf']:
        raise Exception("Invalid output requested. Check help.")
    if year == 'all' and yearout == 'default':
        raise Exception("yearout must be integer year if year == 'all'")

    ### Make filepath
    if drive in ['external', 'ext']:
        filepath = extdatapath
    else:
        filepath = os.path.join(projpath,'in','Wind')+os.sep

    ### Determine file to load
    fileWIND = key.loc[isonode, 'full_timeseries_path']

    # if verbose: print(fileWIND)

    ### Assign timezone
    if timezoneout in ['default', None, 'UTC', 'GMT']:
        timezone = 0
        tz = 'UTC'
        # timezone = key.loc[isonode, 'timezoneWIND']
    else:
        timezone = timezoneout
        tz = 'Etc/GMT{0:+}'.format(-timezone)

    ### Load dataframe from .nc file
    data = xr.open_dataset('{}{}'.format(filepath, fileWIND))
    df = data.to_dataframe()

    ### Apply losses
    df['power'] = df['power'] * (1 - loss_system)

    ### Calculate CF
    df['cf'] = df['power'] / key.loc[isonode, 'capacity']

    ### Select output data
    if output.lower() in ['df', 'dataframe']:
        pass
    elif output.lower() in [
            'wind_speed', 'wind_direction', 'power', 
            'density', 'temperature', 'pressure', 'cf']:
        df = df[output]

    df.index = pd.date_range(
        '2007-01-01', periods=len(df), freq='5T', tz='UTC')

    df = df.tz_convert(tz)

    ### Resample time index if necessary
    if resolution != 5:
        df = df.resample('{}min'.format(resolution)).mean()

    ### Select data year to return, or average if necessary
    if year == 'all':
        dfout = df.groupby(
            [df.index.month, df.index.day, 
            df.index.hour, df.index.minute]).mean()

        if yearhours(yearout) == 8760:
            dfout.drop((2,29), inplace=True)
        
        dfout.reset_index(drop=True, inplace=True)
        
        dfout.index = pd.date_range(
            '{}-01-01'.format(yearout), 
            periods=yearhours(yearout) * 60 / resolution,
            freq='{}min'.format(resolution),
            tz=tz)
    elif year in ['full', None]:
        dfout = df
    else:
        dfout = df.loc[str(year):str(year)].copy()

    return dfout

def getwtkfile(site_id, source='technoecon',
    technopath=None, gridpath=None, eiapath=None,
    drive='RAID', key=None):
    """
    """
    import gzip, pickle
    
    ### Create datetime index
    if not source.lower().startswith('tech'):
        index = pd.date_range(
            '2007-01-01 00:00','2014-01-01 00:00', freq='H', tz='UTC', closed='left')
    
    ### Load the data
    if source.lower().startswith('tech'):
        ### Load key if necessary
        if key is None:
            key = pd.read_csv(
                projpath+'Wind/io/wtk_site_metadata_onshore.csv',
                index_col='site_id')
        df = getWINDnode_fromkey(
            isonode=site_id, key=key, year='full',
            resolution=5, timezoneout='UTC',
            drive=drive)
    
    elif source.lower().startswith('grid'):
        with gzip.open(gridpath+'{}-wtphall.p.gz'.format(site_id)) as p:
            df = pickle.load(p)
        df.index = index
            
    elif source.lower().startswith('eia'):
        ### Load the windspeed/temp/pressure and humidity separately
        with gzip.open(eiapath+'{}-wt80_100-p0_100_200.p.gz'.format(site_id)) as p:
            df1 = pickle.load(p)
        with gzip.open(eiapath+'humidity/{}-h2.p.gz'.format(site_id)) as p:
            df2 = pickle.load(p)
        df = df1.merge(df2, left_index=True, right_index=True)
        df.index = index
        
    return df


def windsim(site_id=None, height=100, dfwind=None,
    dspowercurve=None, dictcf=None, cutout=None,
    source='grid', 
    powercurve_source='wtk', powercurve_model='wtk',
    key=None, density_std =1.225,
    loss_system=0.,
    number_of_turbines=None,
    drive='RAID', humidity=None,
    hystwindow=5, orders=4, pc_vmax=None,
    nans='raise',
    temp_cutoff=None,
    ):
    """
    Inputs
    ------
    source: in ['grid', 'tech', 'eia']
    humidity: None or fraction in [0,1]
    density_std: air density of power curve [kg/m^3]
    powercurve_source: in ['wtk', 'twp', 'ninja']
    powercurve_model: 'wtk' or model as column name
    pc_vmax: 35 for powercurve_source in ['wtk', 'twp'], or None
    orders: decimal points on interpolation
    hystwindow: cut-back-in below cutoff speed [m/s] {default 5}
    temp_cutoff: low-temperature cutoff. Default None; -30°C = 243.15

    TODO
    ----
    * Switch from using floats in dictionary to integers. Would need
      to Scale up windspeeds by order.

    """

    ########
    ### Prep
    if source.lower().startswith('tech'):
        vcol = 'wind_speed'
        pcol = 'pressure'
        tcol = 'temperature'
        dcol = 'density'
        hcol = 'humidity'
    else:
        vcol = 'windspeed_{}m'.format(height)
        pcol = 'pressure_{}m'.format(height)
        tcol = 'temperature_{}m'.format(height)
        dcol = 'density_{}m'.format(height)
        hcol = 'relativehumidity_2m'

    if pc_vmax is None:
        pc_vmax = {'wtk':35,'twp':35,'ninja':40}.get(powercurve_source)

    # if key is None:
    #     key = pd.read_csv(
    #         projpath+'Wind/io/wtk_site_metadata_onshore.csv',
    #         index_col='site_id')

    ###### Power curve

    ### Get power curve
    if (dspowercurve is None) and (dictcf is None):
        ### Get power curve label from wtk power curves
        if (powercurve_source == 'wtk') and (powercurve_model == 'wtk'):
            powercurve_model = key.loc[site_id, 'power_curve']
        dspowercurve = getpowercurve(powercurve_source, powercurve_model)

    if dictcf is None:
        ###### Interpolate the power curve
        ### Create the linearly-interpolated power curve
        xout = np.linspace(0, pc_vmax, 10**orders*pc_vmax+1)
        yout = np.interp(x=xout, 
                         xp=dspowercurve.index.values,
                         fp=dspowercurve.values)
        pcmid = pd.Series(yout, xout)

        ### Round off the index
        pcmid.index = np.around(pcmid.index, orders)

        ### Identify cutout speed
        cutout = dspowercurve.loc[dspowercurve>0].index.max()

        ### Set power output above cutout speed to zero
        pcmid.loc[pcmid.index.map(lambda x: x > cutout)] = 0

        ### Make lookup dict for converting wind speed to power
        dictcf = dict(zip(pcmid.index.values, pcmid.values))


    #############
    ### PROCEDURE

    ###### Load dfwind if necessary
    if dfwind is None:
        ### Load input data
        cfwind = getwtkfile(site_id, source, key=key)

        ### Apply assumed humidity if source == 'tech'
        if source.lower().startswith('tech'):
            if humidity is None:
                humidity = 0.5
            cfwind[hcol] = humidity * 100

        ### Interpolate the pressure if necessary
        if (not source.lower().startswith('tech')) and (height not in [100, 200]):
            x = np.array([0,100,200])
            dfvals = cfwind[['pressure_0m','pressure_100m','pressure_200m']].values
            out = []
            for i in range(len(dfvals)):
                out.append(np.interp(height, x, dfvals[i]))
            cfwind[pcol] = out
    else:
        cfwind = dfwind.copy()

    ### Clean up humidity values
    cfwind.loc[cfwind[hcol]>100,hcol] = 100

    ### Calculate air density
    cfwind[dcol+'_calc'] = density(cfwind[pcol], cfwind[tcol], cfwind[hcol])

    ### Calculate effective wind speed at standard conditions
    cfwind['v_std'] = (
        cfwind[vcol] * (cfwind[dcol+'_calc'] / density_std) ** (1/3))

    ### Include effects of wake loss
    if number_of_turbines is None:
        ### Apply default from wtk
        cfwind['v_std_wake'] = cfwind['v_std'] * wakeloss(site_id=site_id, key=key)
    else:
        ### Apply correction from wtk
        cfwind['v_std_wake'] = cfwind['v_std'] * wakeloss_n(number_of_turbines)


    ### Calculate capacity factor from standardized wind speed
    cfwind['cf_calc'] = np.around(cfwind.v_std_wake.astype(float), orders).map(dictcf)#.fillna(0)

    # ### Calculate naive capacity factor
    # cfwind['v_{}m_wake'.format(height)] = (
    #     cfwind['windspeed_{}m'.format(height)] 
    #     * wakeloss(site_id=site_id, key=key))
    # cfwind['cf_calc_naive'] = np.around(
    #     cfwind['v_{}m_wake'.format(height)], orders
    # ).map(dictcf)#.fillna(0)

    ### Correct for hysteresis (assuming cut-offs apply to corrected wind speed)
    th_hi = cutout
    th_lo = cutout - hystwindow
    cfwind['cf_calc_hyst'] = (
        cfwind.cf_calc 
        * (~hyst(x=cfwind['v_std_wake'].values, 
                 th_lo=th_lo, th_hi=th_hi,
    #              th_lo=cutoutin_wtk['WTKclass{}'.format(power_curve_iec)][1], 
    #              th_hi=cutoutin_wtk['WTKclass{}'.format(power_curve_iec)][0], 
                 initial=False)))

    ### Temperature cutoff (NEW 20191210)
    if temp_cutoff is not None:
        cfwind.loc[cfwind[tcol]<=temp_cutoff, 'cf_calc_hyst'] = 0

    ### Apply losses
    cfwind['cf_calc_hyst_loss'] = cfwind['cf_calc_hyst'] * (1 - loss_system)

    ### Check for nan's and raise exception if found
    if nans == 'raise':
        if any(cfwind.cf_calc_hyst.isnull()):
            print('{}: {} nans'.format(site_id, cfwind.cf_calc.isnull().sum()))
            print(cfwind.loc[cfwind.cf_calc.isnull()])
            raise Exception("nan's in cfwind.cf_calc")
    elif nans.startswith('inter') or nans.startswith('time'):
        cfwind = cfwind.fillna('time')
    elif nans.isin(['ignore','pass','silent']):
        pass


    return cfwind

def windsim_hsds_sites(
    rowf_colfs, height=100, number_of_turbines=1,
    powercurvesource='wtk', model='WTKclass2', pc_vmax=50,
    temp_cutoff=None, desc='CF',
    nans='raise', timeseries=None, regeodatapath=None,
    windpath=None, verbose=True, fulloutput=True):
    """
    Inputs
    ------
    fulloutput: if True, return full CF timeseries; otherwise return mean
    """
    ###### Format inputs
    if windpath is None:
        winddatapath = regeodatapath + 'in/WTK-HSDS/every2-offset0/'
    else:
        winddatapath = windpath

    ### Set timeseries if not provided
    if type(timeseries) == pd.DatetimeIndex:
        _timeseries = timeseries
    if timeseries in [None, 'default', 'full']:
        _timeseries = pd.date_range(
            '2007-01-01 00:00','2014-01-01 00:00', 
            freq='H', closed='left', tz='UTC')
    elif timeseries in ['test','testing']:
        _timeseries = pd.date_range(
            '2010-01-01 00:00','2010-01-08 00:00', 
            freq='H', closed='left', tz='UTC')

    ### Get row,col (every2, offset0) coordinates to query
    rowfs = [int(rowf_colf.split('_')[0]) for rowf_colf in rowf_colfs]
    colfs = [int(rowf_colf.split('_')[1]) for rowf_colf in rowf_colfs]

    ### Divide by 2, since we skip every other site
    rows = [int(rowf / 2) for rowf in rowfs]
    cols = [int(colf / 2) for colf in colfs]
    lookup = (np.array(rows), np.array(cols))

    ### Load power curve dictionary
    dictcf, cutout = zephyr.wind.get_powercurve_dict(
        source=powercurvesource, model=model, pc_vmax=pc_vmax)
    
    ### Set up data and indices
    data = [datum+'_{}m'.format(height) for datum in 
            ['windspeed', 'pressure', 'temperature']]
    data += ['relativehumidity_2m']

    ###### Load all weather files and store in dict
    dictwind = {}
    for datum in data:
        inpath = '{}{}/'.format(winddatapath, datum)
        dictout = {}
        for timestamp in tqdm(_timeseries, desc=datum):
            timestring = timestamp.strftime('%Y%m%dT%H%M')

            ### Load the timeslice
            filename = '{}{}.npy.gz'.format(inpath, timestring)
            with gzip.open(filename, 'rb') as p:
                array = np.load(p)
            ### Pull out the values for the previously-identified points
            arrayout = array[lookup]
            ### Store in dictionary
            dictout[timestring] = arrayout

        ### Turn into dataframe
        dfout = pd.DataFrame(dictout, index=rowf_colfs).T
        dfout.index = _timeseries
        dictwind[datum] = dfout

    ###### Simulate CF from weather files and powercurve dict
    dictout = {}
    for rowf_colf in tqdm(rowf_colfs, desc=desc):
        dfwind = pd.concat(
            [dictwind[datum][rowf_colf].rename(datum) for datum in data], axis=1)
        cfwind = zephyr.wind.windsim(
            dfwind=dfwind, dictcf=dictcf, height=height,
            number_of_turbines=number_of_turbines, cutout=cutout, nans=nans,
            temp_cutoff=temp_cutoff,)
        if fulloutput:
            dictout[rowf_colf] = cfwind['cf_calc_hyst']
        else:
            dictout[rowf_colf] = cfwind['cf_calc_hyst'].mean()
    
    ### Drop the weather data to prevent memory errors
    dictwind = 0

    ### Concat into dataframe and return
    if fulloutput:
        dfout = pd.concat(dictout, axis=1, copy=False)
    else:
        dfout = pd.Series(dictout, name='cf')
    return dfout

def windsim_hsds_timeseries(
    rowf_colfs, height=100, number_of_turbines=1,
    powercurvesource='wtk', model='WTKclass2', pc_vmax=60,
    temp_cutoff=None, desc='CF',
    nans='raise', timeseries=None, regeodatapath=None,
    windpath=None, verbose=True, fulloutput=True,
    progressbar=True):
    """
    Inputs
    ------
    fulloutput: if True, return full CF timeseries; otherwise return mean
    """
    ###### Format inputs
    if windpath is None:
        winddatapath = (
            regeodatapath 
            + 'in/WTK-HSDS/every2-offset0//timeseries/usa-halfdegbuff/')
    else:
        winddatapath = windpath

    ### Load power curve dictionary
    dictcf, cutout = zephyr.wind.get_powercurve_dict(
        source=powercurvesource, model=model, pc_vmax=pc_vmax)
    
    ###### Simulate CF from weather files and powercurve dict
    if progressbar == True:
        iterator = tqdm(rowf_colfs, desc=desc)
    else:
        iterator = rowf_colfs
    dictout = {}
    for rowf_colf in iterator:
        dfwind = pd.read_pickle(
            winddatapath+'{}.df.p.gz'.format(rowf_colf)).copy()
        ### 20200923 - need to explicitly set the freq with pandas 1.1.0
        dfwind.index.freq = 'H'
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
            temp_cutoff=temp_cutoff,)
        if fulloutput:
            dictout[rowf_colf] = cfwind['cf_calc_hyst']
        else:
            dictout[rowf_colf] = cfwind['cf_calc_hyst'].mean()
    
    ### Concat into dataframe and return
    if fulloutput:
        dfout = pd.concat(dictout, axis=1, copy=False)
    else:
        dfout = pd.Series(dictout, name='cf')
    return dfout
