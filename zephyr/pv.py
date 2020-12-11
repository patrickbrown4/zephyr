import pandas as pd
import numpy as np
import sys, os, site, zipfile, math, time, json, pickle
from glob import glob
from tqdm import tqdm, trange
import scipy, scipy.optimize
from warnings import warn

import pvlib

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
import zephyr.data
import zephyr.io

############################
### TIME SERIES MANIPULATION

def dropleap(dfin, year, resolution):
    """
    """
    assert len(dfin) % 8784 == 0, "dfin must be a leap year"
    leapstart = pd.Timestamp('{}-02-29 00:00'.format(year))
    dfout = dfin.drop(dfin.loc[
        leapstart
        :(leapstart
          + pd.Timedelta('1D')
          - pd.Timedelta('{}min'.format(resolution)))].index)
    return dfout

def downsample_trapezoid(dfin, output_freq='H', clip=None):
    """
    """
    ### Get label
    if type(dfin) is pd.DataFrame:
        columns = dfin.columns
    elif type(dfin) is pd.Series:
        columns = [dfin.name]
    
    ### Perform trapezoidal integration
    dfin_hours = dfin.iloc[0::2].copy()
    dfin_halfhours = dfin.iloc[1::2].copy()
    
    dfout = pd.DataFrame(
        (dfin_halfhours.values 
         + dfin_hours.rolling(2).mean().shift(-1).fillna(0).values),
        index=dfin_hours.index, columns=columns) / 2
    
    dfout.index.freq = output_freq
    
    ### Clip it, to get rid of small float errors
    if clip is not None:
        dfout = dfout.where(dfout > clip, 0)

    ### Convert back to series if necessary
    if type(dfin) is pd.Series:
        dfout = dfout[dfin.name]
    
    return dfout


def timeserieslineup(
    series1, series2, 
    resolution1=None, resolution2=None,
    tz1=None, tz2=None,
    resamplemethod='ffill', 
    tzout='left', yearout=None,
    mismatch='raise',
    resampledirection='up', clip=None,
    oneyear=True,
    ):
    """
    Inputs
    ------
    * resolution: int [frequency in minutes] or pandas-readable string
    * cliptimezones: in 'left', 'right', [False, 'none', 'neither', None]
    Notes
    -----
    * Both timeseries need to be one year long, and span Jan1 - Dec31
    * Both timeseries need to have timezone information
    * If oneyear==False, it won't correct leap years.
    """
    ### Make copies
    ds1, ds2 = series1.copy(), series2.copy()
    ### Make sure the time series are well-formatted
    # assert (ds1.index[0].month == 1 and ds1.index[0].day == 1)
    if not (ds1.index[0].month == 1 and ds1.index[0].day == 1):
        print(ds1.head())
        print(ds1.tail())
        print('len(series1) = {}'.format(len(ds1)))
        raise Exception('series1 is not well-formatted')
    # assert (ds2.index[0].month == 1 and ds2.index[0].day == 1)
    if not (ds2.index[0].month == 1 and ds2.index[0].day == 1):
        print(ds2.head())
        print(ds2.tail())
        print('len(series2) = {}'.format(len(ds2)))
        raise Exception('series2 is not well-formatted')
    
    ### Get properties
    tz1, tz2 = ds1.index.tz, ds2.index.tz
    name1, name2 = ds1.name, ds2.name
    if name1 == None: name1 = 0
    if name2 == None: name2 = 0
            
    ### Determine resolutions if not entered
    freq2resolution = {'H': 60, '1H': 60, '60T': 60, 
                       '30T': 30, '15T': 15, 
                       '5T': 5, '1T': 1, 'T': 1,
                       '<Hour>': 60,
                       '<60 * Minutes>': 60, 
                       '<30 * Minutes>': 30, 
                       '<15 * Minutes>': 15, 
                       '<5 * Minutes>': 5, 
                       '<Minute>': 1,
                      }
    
    if resolution1 == None:
        resolution1 = str(ds1.index.freq)
        resolution1 = freq2resolution[resolution1]
    else:
        resolution1 = freq2resolution.get(resolution1, resolution1)
        
    if resolution2 == None:
        resolution2 = str(ds2.index.freq)
        resolution2 = freq2resolution[resolution2]
    else:
        resolution2 = freq2resolution.get(resolution2, resolution2)
        
    ### Get timezones if not entered
    tz1 = ds1.index.tz.__str__()
    tz2 = ds2.index.tz.__str__()
    if ((tz1 == 'None') and (tz2 != 'None')) or ((tz2 == 'None') and (tz1 != 'None')):
        print('tz1 = {}\ntz2 = {}'.format(tz1,tz2))
        raise Exception("Can't align series when one is tz-naive and one is tz-aware")
        
    ### Check if it's tmy, and if so, convert to 2001
    if oneyear:
        ## ds1
        if len(ds1.index.map(lambda x: x.year).unique()) != 1:
            ds1.index = pd.date_range(
                '2001-01-01', '2002-01-01',
                freq='{}T'.format(resolution1), tz=tz1, closed='left')
            year1 = 2001
        else:
            year1 = ds1.index[0].year
        ## ds2
        if len(ds2.index.map(lambda x: x.year).unique()) != 1:
            ds2.index = pd.date_range(
                '2001-01-01', '2002-01-01',
                freq='{}T'.format(resolution2), tz=tz2, closed='left')
            year2 = 2001
        else:
            year2 = ds2.index[0].year
    
    ###### Either upsample or downsample
    if resampledirection.startswith('up'):
        ###### Upsample the lower-resolution dataset, if necessary
        resolution = min(resolution1, resolution2)
        
        ## Upsample ds2
        if resolution1 < resolution2:
            ## Extend past the last value, for interpolation
            # ds2.loc[ds2.index.max() + 1] = ds2.iloc[-1]
            ds2 = ds2.append(
                pd.Series(data=ds2.iloc[-1],
                          index=[ds2.index[-1] + pd.Timedelta(resolution2, 'm')]))
            ## Interpolate
            if resamplemethod in ['ffill', 'forward', 'pad']:
                ds2 = ds2.resample('{}T'.format(resolution1)).ffill()
            elif resamplemethod in ['interpolate', 'time']:
                ### NOTE that this still just ffills the last value
                ds2 = ds2.resample('{}T'.format(resolution1)).interpolate('time')
            else:
                raise Exception("Unsupported resamplemethod: {}".format(resamplemethod))
            ## Drop the extended value
            ds2.drop(ds2.index[-1], inplace=True)
        ## Upsample ds1
        elif resolution1 > resolution2:
            ## Extend past the last value, for interpolation
            # ds1.loc[ds1.index.max() + 1] = ds1.iloc[-1]
            ds1 = ds1.append(
                pd.Series(data=ds1.iloc[-1],
                          index=[ds1.index[-1] + pd.Timedelta(resolution1, 'm')]))
            ## Interpolate
            if resamplemethod in ['ffill', 'forward', 'pad']:
                ds1 = ds1.resample('{}T'.format(resolution2)).ffill()
            elif resamplemethod in ['interpolate', 'time']:
                ### NOTE that this still just ffills the last value
                ds1 = ds1.resample('{}T'.format(resolution2)).interpolate('time')
            else:
                raise Exception("Unsupported resamplemethod: {}".format(resamplemethod))
            ## Drop the extended value
            ds1.drop(ds1.index[-1], inplace=True)
    elif resampledirection.startswith('down'):
        ### Only works for 2x difference in frequency. Check to make sure.
        resmin = min(resolution1, resolution2)
        resmax = max(resolution1, resolution2)
        assert ((resmax % resmin == 0) & (resmax // resmin == 2))
        ### Resample ds1 if it has finer resolution than ds2
        if resolution1 < resolution2:
            ds1 = downsample_trapezoid(
                dfin=ds1, output_freq='{}T'.format(resolution2), clip=clip)
        ### Resample ds2 if it has finer resolution than ds1
        elif resolution2 < resolution1:
            ds2 = downsample_trapezoid(
                dfin=ds2, output_freq='{}T'.format(resolution1), clip=clip)
        
    ### Drop leap days if ds1 and ds2 have different year lengths
    if oneyear:
        year1hours, year2hours = zephyr.toolbox.yearhours(year1), zephyr.toolbox.yearhours(year2)
        if (year1hours == 8784) and (year2hours == 8760):
            # ds1 = dropleap(ds1, year1, resolution1)
            ds1 = dropleap(ds1, year1, resolution)

        elif (year1hours == 8760) and (year2hours == 8784):
            # ds2 = dropleap(ds2, year2, resolution2)
            ds2 = dropleap(ds2, year2, resolution)
    
    ### Check for errors
    if len(ds1) != len(ds2):
        if mismatch == 'raise':
            print('Lengths: {}, {}'.format(len(ds1), len(ds2)))
            print('Resolutions: {}, {}'.format(resolution1, resolution2))
            print(ds1.head(3))
            print(ds1.tail(3))
            print(ds2.head(3))
            print(ds2.tail(3))
            raise Exception('Mismatched lengths')
        elif mismatch == 'verbose':
            print('Lengths: {}, {}'.format(len(ds1), len(ds2)))
            print('Resolutions: {}, {}'.format(resolution1, resolution2))
            print(ds1.head(3))
            print(ds1.tail(3))
            print(ds2.head(3))
            print(ds2.tail(3))
            warn('Mismatched lengths')
        elif mismatch == 'warn':
            warn('Mismatched lengths: {}, {}'.format(len(ds1), len(ds2)))
        
    ###### Align years
    if oneyear:
        yeardiff = year1 - year2
        if yearout in ['left', '1', 1]:
            ### Align to ds1.index
            ds2.index = ds2.index + pd.DateOffset(years=yeardiff)
        elif yearout in ['right', '2', 2]:
            ### Align to ds2.index
            ds1.index = ds1.index - pd.DateOffset(years=yeardiff)
        elif isinstance(yearout, int) and (yearout not in [1,2]):
            ### Align both timeseries to yearout
            year1diff = yearout - year1
            year2diff = yearout - year2
            ds1.index = ds1.index + pd.DateOffset(years=year1diff)
            ds2.index = ds2.index + pd.DateOffset(years=year2diff)
        else:
            pass

    ### Align and clip time zones, if necessary
    if tz1 != tz2:
        if tzout in ['left', 1, '1']:
            ds2 = (pd.DataFrame(ds2.tz_convert(tz1))
                   .merge(pd.DataFrame(index=ds1.index), left_index=True, right_index=True)
                  )[name2]
        elif tzout in ['right', 2, '2']:
            ds1 = (pd.DataFrame(ds1.tz_convert(tz2))
                   .merge(pd.DataFrame(index=ds2.index), left_index=True, right_index=True)
                  )[name1]
            
    return ds1, ds2


############################
### SYSTEM OUTPUT SIMULATION

class PVsystem:
    """
    """
    def __init__(self, 
                 systemtype='track',
                 axis_tilt=None, axis_azimuth=180,
                 max_angle=60, backtrack=True, gcr=1./3.,
                 dcac=1.3, 
                 loss_system=0.14, loss_inverter=0.04,
                 n_ar=1.3, n_glass=1.526,
                 tempcoeff=-0.004, 
                 # temp_model='open_rack_cell_polymerback',
                 temp_model='open_rack_glass_polymer', ### 20200126
                 albedo=0.2, 
                 et_method='nrel', diffuse_model='reindl',
                 model_perez='allsitescomposite1990',
                 clip=True
                ):
        ### Defaults
        if (axis_tilt is None) and (systemtype == 'fixed'):
            axis_tilt = 'latitude'
        elif (axis_tilt is None) and (systemtype == 'track'):
            axis_tilt = 0
        ### Parameters
        self.gentype = 'pv'
        self.systemtype = systemtype
        self.axis_tilt = axis_tilt
        self.axis_azimuth = axis_azimuth
        self.max_angle = max_angle
        self.backtrack = backtrack
        self.gcr = gcr
        self.dcac = dcac
        self.loss_system = loss_system
        self.loss_inverter = loss_inverter
        self.n_ar = n_ar
        self.n_glass = n_glass
        self.tempcoeff = tempcoeff
        self.temp_model = temp_model
        self.albedo = albedo
        self.et_method = et_method
        self.diffuse_model = diffuse_model
        self.model_perez = model_perez
        self.clip = clip

    def sim(self, nsrdbfile, year, 
            nsrdbpathin=None, nsrdbtype='.gz',
            resolution='default', 
            output_ac_only=True, output_ac_tz=False,
            return_all=False, query=False, **kwargs
            ):
        """
        Notes
        -----
        * If querying googlemaps and NSRDB with query==True and a latitude/longitude query,
        need to put latitude first, longitude second, and have both in a string.
        E.g. '40, -100' for latitude=40, longitude=-100.
        * Googlemaps query is not foolproof, so should check the output file if you really
        care about getting the right location.
        * Safest approach is to use an explicit nsrdbfile path and query==False.
        """
        if not os.path.exists(
                os.path.join(zephyr.toolbox.pathify(nsrdbpathin), nsrdbfile)):
            if query == True:
                _,_,_,_,fullpath = zephyr.io.queryNSRDBfile(
                    nsrdbfile, year, returnfilename=True, **kwargs)
            elif query == False:
                print(nsrdbpathin)
                print(nsrdbfile)
                raise FileNotFoundError
        else:
            fullpath = nsrdbfile

        return pv_system_sim(
            nsrdbfile=fullpath, year=year, systemtype=self.systemtype, 
            axis_tilt=self.axis_tilt, axis_azimuth=self.axis_azimuth,
            max_angle=self.max_angle, backtrack=self.backtrack, gcr=self.gcr, 
            dcac=self.dcac, loss_system=self.loss_system, 
            loss_inverter=self.loss_inverter, n_ar=self.n_ar, n_glass=self.n_glass,
            tempcoeff=self.tempcoeff, temp_model=self.temp_model, 
            albedo=self.albedo, diffuse_model=self.diffuse_model,
            et_method=self.et_method, model_perez=self.model_perez,
            nsrdbpathin=nsrdbpathin, nsrdbtype=nsrdbtype, resolution=resolution, 
            output_ac_only=output_ac_only, output_ac_tz=output_ac_tz, 
            return_all=return_all, clip=self.clip)

def loss_reflect_abs(aoi, n_glass=1.526, n_ar=1.3, n_air=1, K=4., L=0.002):
    """
    Adapted from pvlib.pvsystem.physicaliam and PVWatts Version 5 section 8
    """
    if isinstance(aoi, pd.Series):
        aoi.loc[aoi <= 1e-6] = 1e-6
    elif isinstance(aoi, float):
        aoi = max(aoi, 1e-6)
    elif isinstance(aoi, int):
        aoi = max(aoi, 1e-6)
    elif isinstance(aoi, np.ndarray):
        aoi[aoi <= 1e-6] = 1e-6

    theta_ar = pvlib.tools.asind(
        (n_air / n_ar) * pvlib.tools.sind(aoi))

    tau_ar = (1 - 0.5 * (
        ((pvlib.tools.sind(theta_ar - aoi)) ** 2)
        / ((pvlib.tools.sind(theta_ar + aoi)) ** 2)
        + ((pvlib.tools.tand(theta_ar - aoi)) ** 2)
        / ((pvlib.tools.tand(theta_ar + aoi)) ** 2)))

    theta_glass = pvlib.tools.asind(
        (n_ar / n_glass) * pvlib.tools.sind(theta_ar))

    tau_glass = (1 - 0.5 * (
        ((pvlib.tools.sind(theta_glass - theta_ar)) ** 2)
        / ((pvlib.tools.sind(theta_glass + theta_ar)) ** 2)
        + ((pvlib.tools.tand(theta_glass - theta_ar)) ** 2)
        / ((pvlib.tools.tand(theta_glass + theta_ar)) ** 2)))

    tau_total = tau_ar * tau_glass

    tau_total = np.where((np.abs(aoi) >= 90) | (tau_total < 0), np.nan, tau_total)

    if isinstance(aoi, pd.Series):
        tau_total = pd.Series(tau_total, index=aoi.index)

    return tau_total

def loss_reflect(
    aoi, n_glass=1.526, n_ar=1.3, n_air=1, K=4., L=0.002,
    fillna=True):
    """
    """
    out = (
        loss_reflect_abs(aoi, n_glass, n_ar, n_air, K, L)
        / loss_reflect_abs(0, n_glass, n_ar, n_air, K, L))

    ####### UPDATED 20180712 ########
    if fillna==True:
        if isinstance(out, (pd.Series, pd.DataFrame)):
            out = out.fillna(0)
        elif isinstance(out, np.ndarray):
            out = np.nan_to_num(out)
    #################################

    return out

def pv_system_sim(
    nsrdbfile, year, systemtype='track', 
    axis_tilt=0, axis_azimuth=180, 
    max_angle=60, backtrack=True, gcr=1./3., 
    dcac=1.3, 
    loss_system=0.14, loss_inverter=0.04, 
    n_ar=1.3, n_glass=1.526, 
    tempcoeff=-0.004, 
    # temp_model='open_rack_cell_polymerback',
    temp_model='open_rack_glass_polymer', ### 20200126
    albedo=0.2, diffuse_model='reindl', 
    et_method='nrel', model_perez='allsitescomposite1990', 
    nsrdbpathin='in/NSRDB/', nsrdbtype='.gz', 
    resolution='default', 
    output_ac_only=True, output_ac_tz=False,
    return_all=False, clip=True):
    """
    Outputs
    -------
    * output_ac: pd.Series of instantaneous Wac per kWac 
        * Divide by 1000 to get instantaneous AC capacity factor [fraction]
        * Take mean to get yearly capacity factor
        * Take sum * resolution / 60 to get yearly energy generation in Wh

    Notes
    -----
    * To return DC (instead of AC) output, set dcac=None and loss_inverter=None
    """
    ### Set resolution, if necessary
    if resolution in ['default', None]:
        if type(year) == int: resolution = 30
        elif str(year).lower() == 'tmy': resolution = 60
        else: raise Exception("year must be 'tmy' or int")

    ### Set NSRDB filepath
    if nsrdbpathin == 'in/NSRDB/':
        nsrdbpath = zephyr.toolbox.pathify(
            nsrdbpathin, add='{}/{}min/'.format(year, resolution))
    else:
        nsrdbpath = nsrdbpathin

    ### Load NSRDB file
    dfsun, info, tz, elevation = zephyr.io.getNSRDBfile(
        filepath=nsrdbpath, filename=nsrdbfile, 
        year=year, resolution=resolution, forcemidnight=False)

    ### Select latitude for tilt, if necessary
    latitude = float(info['Latitude'])
    longitude = float(info['Longitude'])

    if axis_tilt == 'latitude':
        axis_tilt = latitude
    elif axis_tilt == 'winter':
        axis_tilt = latitude + 23.437

    ### Set timezone from info in NSRDB file
    timezone = int(info['Time Zone'])
    tz = 'Etc/GMT{:+}'.format(-timezone)

    ### Determine solar position
    times = dfsun.index.copy()
    solpos = pvlib.solarposition.get_solarposition(
        time=times, latitude=latitude, longitude=longitude)

    ### Set extra parameters for diffuse sky models
    if diffuse_model in ['haydavies', 'reindl', 'perez']:
        dni_et = pvlib.irradiance.get_extra_radiation(
            datetime_or_doy=times, method=et_method, epoch_year=year)
        airmass = pvlib.atmosphere.get_relative_airmass(
            zenith=solpos['apparent_zenith'])
    else:
        dni_et = None
        airmass = None

    ### Get surface tilt, from tracker data if necessary
    if systemtype == 'track':
        with np.errstate(invalid='ignore'):
            tracker_data = pvlib.tracking.singleaxis(
                apparent_zenith=solpos['apparent_zenith'], 
                apparent_azimuth=solpos['azimuth'],
                axis_tilt=axis_tilt, axis_azimuth=axis_azimuth, 
                max_angle=max_angle, backtrack=backtrack, gcr=gcr)
        surface_tilt = tracker_data['surface_tilt']
        surface_azimuth = tracker_data['surface_azimuth']
        surface_tilt = surface_tilt.fillna(axis_tilt).replace(0., axis_tilt)
        surface_azimuth = surface_azimuth.fillna(axis_azimuth)

    elif systemtype == 'fixed':
        surface_tilt = axis_tilt
        surface_azimuth = axis_azimuth

    ### Determine angle of incidence
    aoi = pvlib.irradiance.aoi(
        surface_tilt=surface_tilt, surface_azimuth=surface_azimuth,
        solar_zenith=solpos['apparent_zenith'], solar_azimuth=solpos['azimuth'])

    ### Determine plane-of-array irradiance
    poa_irrad = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt, surface_azimuth=surface_azimuth, 
        solar_zenith=solpos['apparent_zenith'], solar_azimuth=solpos['azimuth'], 
        dni=dfsun['DNI'], ghi=dfsun['GHI'], dhi=dfsun['DHI'], 
        dni_extra=dni_et, airmass=airmass, albedo=albedo,
        model=diffuse_model, model_perez=model_perez)

    ### Correct for reflectance losses
    poa_irrad['poa_global_reflectlosses'] = (
        (poa_irrad['poa_direct'] * loss_reflect(aoi, n_glass, n_ar)) 
        + poa_irrad['poa_diffuse'])
    poa_irrad.fillna(0, inplace=True)

    ### Correct for temperature losses
    # celltemp = pvlib.pvsystem.sapm_celltemp(
    #     poa_irrad['poa_global_reflectlosses'], dfsun['Wind Speed'], 
    #     dfsun['Temperature'], temp_model)['temp_cell']
    temp_model_params = (
        pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm'][temp_model])
    celltemp = pvlib.temperature.sapm_cell(
        poa_global=poa_irrad['poa_global_reflectlosses'], 
        wind_speed=dfsun['Wind Speed'], 
        temp_air=dfsun['Temperature'], 
        a=temp_model_params['a'], b=temp_model_params['b'], 
        deltaT=temp_model_params['deltaT'],
    )

    output_dc_loss_temp = pvlib.pvsystem.pvwatts_dc(
        g_poa_effective=poa_irrad['poa_global_reflectlosses'], 
        temp_cell=celltemp, pdc0=1000, 
        gamma_pdc=tempcoeff, temp_ref=25.)

    ### Corect for total DC system losses
    # output_dc_loss_all = output_dc_loss_temp * eta_system
    output_dc_loss_all = output_dc_loss_temp * (1 - loss_system)

    ### Correct for inverter
    ##################################
    ### Allow for DC output (20180821)
    if (dcac is None) and (loss_inverter is None):
        output_ac = output_dc_loss_all
    else:
        ##############################
        output_ac = pvlib.pvsystem.pvwatts_ac(
            pdc=output_dc_loss_all*dcac,
            ##### UPDATED 20180708 #####
            # pdc0=1000,
            pdc0=1000/(1-loss_inverter),
            ############################
            # eta_inv_nom=eta_inverter,
            eta_inv_nom=(1-loss_inverter),
            eta_inv_ref=0.9637).fillna(0)
        if clip is True:                     ### Added 20181126
            output_ac = output_ac.clip(0)

    ### Return output
    if output_ac_only:
        return output_ac
    if output_ac_tz:
        return output_ac, tz
    if (return_all == True) and (systemtype == 'track'):
        return (dfsun, solpos, aoi, poa_irrad,
            celltemp, output_dc_loss_temp,
            output_dc_loss_all, output_ac, tracker_data)
    if (return_all == True) and (systemtype == 'fixed'):
        return (dfsun, solpos, aoi, poa_irrad, 
            celltemp, output_dc_loss_temp, 
            output_dc_loss_all, output_ac)

def pv_system_sim_fast(
    axis_tilt_and_azimuth,
    dfsun, info, tznode, elevation,
    solpos, dni_et, airmass,
    year, systemtype, 
    max_angle=60, backtrack=True, gcr=1./3., 
    dcac=1.3, 
    loss_system=0.14, loss_inverter=0.04, 
    n_ar=1.3, n_glass=1.526, 
    tempcoeff=-0.004, 
    # temp_model='open_rack_cell_polymerback',
    temp_model='open_rack_glass_polymer', ### 20200126
    albedo=0.2, diffuse_model='reindl', 
    et_method='nrel', model_perez='allsitescomposite1990',
    clip=True):
    """
    Calculate ac_out after solar resource file has already been loaded
    """
    ### Unpack axis_tilt_and_azimuth
    axis_tilt, axis_azimuth = axis_tilt_and_azimuth

    ### Get surface tilt, from tracker data if necessary
    if systemtype == 'track':
        tracker_data = pvlib.tracking.singleaxis(
            apparent_zenith=solpos['apparent_zenith'], 
            apparent_azimuth=solpos['azimuth'],
            axis_tilt=axis_tilt, axis_azimuth=axis_azimuth, 
            max_angle=max_angle, backtrack=backtrack, gcr=gcr)
        surface_tilt = tracker_data['surface_tilt']
        surface_azimuth = tracker_data['surface_azimuth']
        ###### ADDED 20180712 ######
        surface_tilt = surface_tilt.fillna(axis_tilt).replace(0., axis_tilt)
        surface_azimuth = surface_azimuth.fillna(axis_azimuth)

    elif systemtype == 'fixed':
        surface_tilt = axis_tilt
        surface_azimuth = axis_azimuth

    ### Determine angle of incidence
    aoi = pvlib.irradiance.aoi(
        surface_tilt=surface_tilt, surface_azimuth=surface_azimuth,
        solar_zenith=solpos['apparent_zenith'], solar_azimuth=solpos['azimuth'])

    ### Determine plane-of-array irradiance
    poa_irrad = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt, surface_azimuth=surface_azimuth, 
        solar_zenith=solpos['apparent_zenith'], solar_azimuth=solpos['azimuth'], 
        dni=dfsun['DNI'], ghi=dfsun['GHI'], dhi=dfsun['DHI'], 
        dni_extra=dni_et, airmass=airmass, albedo=albedo,
        model=diffuse_model, model_perez=model_perez)

    ### Correct for reflectance losses
    poa_irrad['poa_global_reflectlosses'] = (
        (poa_irrad['poa_direct'] * loss_reflect(aoi, n_glass, n_ar)) 
        + poa_irrad['poa_diffuse'])
    poa_irrad.fillna(0, inplace=True)

    ### Correct for temperature losses
    # celltemp = pvlib.pvsystem.sapm_celltemp(
    #     poa_irrad['poa_global_reflectlosses'], dfsun['Wind Speed'], 
    #     dfsun['Temperature'], temp_model)['temp_cell']
    ### UPDATED 20200126
    temp_model_params = (
        pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm'][temp_model])
    celltemp = pvlib.temperature.sapm_cell(
        poa_global=poa_irrad['poa_global_reflectlosses'], 
        wind_speed=dfsun['Wind Speed'], 
        temp_air=dfsun['Temperature'], 
        a=temp_model_params['a'], b=temp_model_params['b'], 
        deltaT=temp_model_params['deltaT'],
    )


    output_dc_loss_temp = pvlib.pvsystem.pvwatts_dc(
        g_poa_effective=poa_irrad['poa_global_reflectlosses'], 
        temp_cell=celltemp, pdc0=1000, 
        gamma_pdc=tempcoeff, temp_ref=25.)

    ### Corect for total DC system losses
    output_dc_loss_all = output_dc_loss_temp * (1 - loss_system)

    ### Correct for inverter
    output_ac = pvlib.pvsystem.pvwatts_ac(
        pdc=output_dc_loss_all*dcac,
        ##### UPDATED 20180708 #####
        # pdc0=1000,
        pdc0=1000/(1-loss_inverter),
        ############################
        eta_inv_nom=(1-loss_inverter),
        eta_inv_ref=0.9637).fillna(0)
    if clip is True:
        output_ac = output_ac.clip(0)

    return output_ac

################################
### ORIENTATION OPTIMIZATION ###

def pv_optimize_orientation_objective(
    axis_tilt_and_azimuth,
    objective,
    dfsun, info, tznode, elevation,
    solpos, dni_et, airmass,
    systemtype, 
    yearsun, resolutionsun, 
    dflmp=None, yearlmp=None, resolutionlmp=None, tzlmp=None,
    pricecutoff=None,
    max_angle=60, backtrack=True, gcr=1./3., 
    dcac=1.3, 
    loss_system=0.14, loss_inverter=0.04, 
    n_ar=1.3, n_glass=1.526, 
    tempcoeff=-0.004, 
    # temp_model='open_rack_cell_polymerback',
    temp_model='open_rack_glass_polymer', ### 20200126
    albedo=0.2, diffuse_model='reindl', 
    et_method='nrel', model_perez='allsitescomposite1990',
    axis_tilt_constant=None, axis_azimuth_constant=None,
    clip=True,
    ):
    """
    """
    ###### Repackage input variables as necessary
    ### If axis_tilt_and_azimuth is numeric and length 1, require that
    ### only axis_tilt_constant or axis_azimuth_constant be specifed.
    ### axis_tilt_and_azimuth will then be taken as whichever of 
    ### axis_tilt_constant or axis_azimuth_constant is not specified.
    
    ### Make sure at least one of axis_tilt_constant, axis_azimuth_constant is None
    if (axis_tilt_constant is not None) and (axis_azimuth_constant is not None):
        print('axis_tilt_and_azimuth: {}, {}'.format(
            axis_tilt_and_azimuth, type(axis_tilt_and_azimuth)))
        print('axis_tilt_constant: {}, {}'.format(
            axis_tilt_constant, type(axis_tilt_constant)))
        print('axis_azimuth_constant: {}, {}'.format(
            axis_azimuth_constant, type(axis_azimuth_constant)))
        raise Exception("At least one of (axis_tilt_constant, "
                        "axis_azimuth_constant) must be None.")
    ### Azimuth-only optimization case
    elif axis_tilt_constant is not None:
        axis_tilt_and_azimuth = (axis_tilt_constant, float(axis_tilt_and_azimuth))
    ### Tilt-only optimization case
    elif axis_azimuth_constant is not None:
        axis_tilt_and_azimuth = (float(axis_tilt_and_azimuth), axis_azimuth_constant)
    ### Full optimization orientation
    elif (axis_azimuth_constant is None) and (axis_tilt_constant is None):
        pass
    
    ### Continue as usual
    output_ac = pv_system_sim_fast(
        axis_tilt_and_azimuth=axis_tilt_and_azimuth,
        dfsun=dfsun, info=info, tznode=tznode, elevation=elevation,
        solpos=solpos, dni_et=dni_et, airmass=airmass,
        year=yearsun, systemtype=systemtype, 
        max_angle=max_angle, backtrack=backtrack, gcr=gcr,
        dcac=dcac,
        loss_system=loss_system, loss_inverter=loss_inverter,
        n_ar=n_ar, n_glass=n_glass,
        tempcoeff=tempcoeff, 
        temp_model=temp_model,
        albedo=albedo, diffuse_model=diffuse_model, 
        et_method=et_method, model_perez=model_perez,
        clip=clip)

    if objective.lower() in ['cf', 'capacityfactor']:
        capacityfactor = (
            0.001 * output_ac.sum() / len(output_ac))
        return -capacityfactor

    elif objective.lower() in ['rev', 'revenue']:
        ### Drop leap days if yearlmp andyearsun have different year lengths
        if zephyr.toolbox.yearhours(yearlmp) == 8784:
            if (yearsun == 'tmy') or zephyr.toolbox.yearhours(yearsun) == 8760:
                ## Drop lmp leapyear
                dflmp = dropleap(dflmp, yearlmp, resolutionlmp)
        elif zephyr.toolbox.yearhours(yearlmp) == 8760 and (zephyr.toolbox.yearhours(yearsun) == 8784):
            ## Drop sun leapyear
            output_ac = dropleap(output_ac, yearsun, resolutionsun)

        ### Reset indices if yearsun != yearlmp
        if yearsun != yearlmp:
            output_ac.index = pd.date_range(
                '2001-01-01', 
                periods=(8760 * 60 / resolutionsun), 
                freq='{}min'.format(resolutionsun), 
                tz=tznode
            ).tz_convert(tzlmp)

            dflmp.index = pd.date_range(
                '2001-01-01', 
                periods=(8760 * 60 / resolutionlmp),
                freq='{}min'.format(resolutionlmp), 
                tz=tzlmp
            )

        ### upsample solar data if resolutionlmp == 5
        if resolutionlmp == 5:
            ### Original version - no longer works given recent pandas update
            # output_ac.loc[output_ac.index.max() + 1] = output_ac.iloc[-1]
            ### New version
            output_ac = output_ac.append(
                pd.Series(data=output_ac.iloc[-1],
                          index=[output_ac.index[-1] + pd.Timedelta(resolutionsun, 'm')]))
            ### Continue
            output_ac = output_ac.resample('5T').interpolate(method='time')
            output_ac.drop(output_ac.index[-1], axis=0, inplace=True)

        ### upsample LMP data if resolutionsun == 30
        if (resolutionlmp == 60) and (resolutionsun == 30):
            dflmp = dflmp.resample('30T').ffill()
            dflmp.loc[dflmp.index.max() + 1] = dflmp.iloc[-1]

        if len(output_ac) != len(dflmp):
            print('Something probably went wrong with leap years')
            print('len(output_ac.index) = {}'.format(len(output_ac.index)))
            print('len(dflmp.index) = {}'.format(len(dflmp.index)))
            raise Exception('Mismatched lengths for LMP and output_ac files')

        ### put dflmp into same timezone as output_ac
        ### (note that this will drop point(s) from dflmp)
        if tznode != tzlmp:
            dflmp = (
                pd.DataFrame(dflmp.tz_convert(tznode))
                .merge(pd.DataFrame(index=output_ac.index), 
                       left_index=True, right_index=True)
            )['lmp']

        ### Determine final resolution
        resolution = min(resolutionlmp, resolutionsun)
        

        ### Calculate revenue
        if pricecutoff is None:
            revenue_timestep = 0.001 * output_ac * dflmp * resolution / 60 
        ### DO: Would be faster to do this outside of the function
        else:
            dispatch = dflmp.map(lambda x: x > pricecutoff)
            output_dispatched = (output_ac * dispatch)
            revenue_timestep = (
                0.001 * output_dispatched * dflmp * resolution / 60)

        revenue_yearly = revenue_timestep.sum() / 1000 # $/kWac-yr
        return -revenue_yearly

def pv_optimize_orientation(
    objective,
    dfsun, info, tznode, elevation,
    solpos, dni_et, airmass,
    systemtype, 
    yearsun, resolutionsun, 
    dflmp=None, yearlmp=None, resolutionlmp=None, tzlmp=None,
    pricecutoff=None,
    max_angle=60, backtrack=True, gcr=1./3., 
    dcac=1.3, 
    loss_system=0.14, loss_inverter=0.04, 
    n_ar=1.3, n_glass=1.526, 
    tempcoeff=-0.004, 
    # temp_model='open_rack_cell_polymerback',
    temp_model='open_rack_glass_polymer', ### 20200126
    albedo=0.2, diffuse_model='reindl', 
    et_method='nrel', model_perez='allsitescomposite1990',
    ranges='default', full_output=False,
    optimize='both', 
    axis_tilt_constant=None, axis_azimuth_constant=None,
    clip=True,
    ):
    """
    """
    ###### Check inputs, adapt to different choices of optimization
    if optimize in ['both', 'default', None, 'orient', 'orientation']:
        assert (axis_tilt_constant is None) and (axis_azimuth_constant is None)
        if ranges == 'default':
            ranges = (slice(0, 100, 10), slice(90, 280, 10))
        else:
            assert len(ranges) == 2
            assert (isinstance(ranges[0], slice) 
                    and (isinstance(ranges[1], slice)))
    elif optimize in ['azimuth', 'axis_azimuth_constant']:
        assert axis_azimuth_constant is None
        if ranges == 'default':
            ranges = (slice(90, 271, 1),)
        else:
            assert isinstance(ranges, tuple)
            assert len(ranges) == 1
    elif optimize in ['tilt', 'axis_tilt_constant']:
        assert axis_tilt_constant is None
        if ranges == 'default':
            ranges = (slice(0, 91, 1),)
        else:
            assert isinstance(ranges, tuple)
            assert len(ranges) == 1
    else:
        raise Exception("optimize must be in 'both', 'azimuth', 'tilt'")

    params = (
        objective, dfsun, info, tznode, elevation, solpos, dni_et, 
        airmass, systemtype, yearsun, resolutionsun, 
        dflmp, yearlmp, resolutionlmp, tzlmp,
        pricecutoff,
        max_angle, backtrack, gcr, dcac, loss_system, loss_inverter, 
        n_ar, n_glass, tempcoeff, temp_model, albedo, 
        diffuse_model, et_method, model_perez, 
        axis_tilt_constant, axis_azimuth_constant,
        clip)

    results = scipy.optimize.brute(
        pv_optimize_orientation_objective,
        ranges=ranges,
        args=params,
        full_output=True,
        disp=False,
        finish=scipy.optimize.fmin)

    ### Unpack and return results
    opt_orient = results[0]
    opt_objective = results[1]
    opt_input_grid = results[2]
    opt_output_grid = results[3]
    
    if full_output:
        return (opt_orient, -opt_objective, 
                opt_input_grid, opt_output_grid)
    else:
        return opt_orient, -opt_objective
