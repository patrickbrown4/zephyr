###############
#%% IMPORTS ###

import pandas as pd
import numpy as np
import os, sys, site, math, time, pickle, gzip
from glob import glob
from tqdm import tqdm, trange

import zephyr

### Common paths
projpath = zephyr.settings.projpath

#######################
#%% ARGUMENT INPUTS ###

import argparse
parser = argparse.ArgumentParser(description='run zephyr for series of cases')

parser.add_argument(
    'case', type=str, default='0',
    help='integer name of case; can be single integer or comma-delimited',)
parser.add_argument(
    '-r', '--region', default='all', help='region or "all"')
parser.add_argument(
    '-o', '--batchname', default='test', 
    help='case file name: "cases-{batchname}.xlsx"')
parser.add_argument(
    '-m', '--method', type=int, default=-1, 
    help=('gurobi method: -1 default, 0 primal simplex, 1 dual simplex, 2 barrier, '
          '3 concurrent, 4 deterministic concurrent, 5 deterministic concurrent simplex',))
parser.add_argument(
    '-t', '--truncmid', type=float, default=0.0005, help='midpoint for vre truncation')
parser.add_argument(
    '-y', '--savemod', type=str, default='', help='append to end of savename')
parser.add_argument(
    '-s', '--solver', type=str, default='gurobi', choices=['gurobi','clp','cbc','gurobipy'])
parser.add_argument('-v', '--verbose', action='count', default=0)
parser.add_argument('-z', '--overwrite', action='store_true')
parser.add_argument('-a', '--includedual', action='store_true')
parser.add_argument('-b', '--includereserves', action='store_true')

### Parse it
args = parser.parse_args()
runcases = [int(x) for x in args.case.split(',')]
batchname = args.batchname
outpath = os.path.join(projpath,'out',batchname)+os.sep
runregions = args.region.split(',')
verbose = args.verbose
overwrite = args.overwrite
method = args.method
truncmid = args.truncmid
savemod = args.savemod
if len(savemod) > 0:
    savemod = '-{}'.format(savemod)
solver = args.solver
includedual = args.includedual
includereserves = args.includereserves

####################
#%% DEBUG INPUTS ###

# case = 0
# runcases = [0]
# region = 'all'
# runregions = ['all']
# outpath = os.path.join(projpath,'out','test')+os.sep
# method = -1
# truncmid = 0
# savemod = ''
# solver = 'gurobi'
# verbose = True
# overwrite = False
# includedual = False
# includereserves = False

##############
#%% INPUTS ###

truncmax = 2 * truncmid

### Cases to run
os.makedirs(os.path.join(outpath), exist_ok=True)
batchfile = 'cases-{}.xlsx'.format(batchname)
cases = pd.read_excel(
    batchfile, index_col='case', sheet_name='runs',
    dtype={
        'include_hydro_res':bool, 'include_hydro_ror':bool,
        'include_gas':bool, 'gasprice':str,
        'existing_trans':bool, 'build_ac':bool, 'build_dc': str,
        'include_phs':int, 'build_phs':str, 
        'include_nuclear':int, 'build_nuclear':str,
        'bins_pv':str, 'bins_wind':str, 'interconnection_scaler':int,
        'pvscale':float, 'windscale':float, 'storscale':float, 'transscale':float,
    }
)
dfstate = pd.read_excel(
    batchfile, sheet_name='state', index_col=0, header=[0,1],
)

### Economic and simulation assumptions
infile = os.path.join(projpath,'io','generator_fuel_assumptions.xlsx')
defaults = zephyr.cpm.Defaults(infile)

### Distance parameters
urban = 'edge'

### VRE parameters
systemtype = 'track'
axis_tilt = 0
axis_azimuth = 180
dcac = 1.30

losses_wind_upfront = 0

pvcost_bins = 86 ### USD/kWac-yr
windcost_bins = 121 ### USD/kWac-yr

### Load parameters
loadtype = 'both'

### Extra stuff
timezones = {
    'BPA': -8, 'Northwest': -8, 'CAISO': -8,
    'Southwest': -7, 'ERCOT': -6, 'SPP': -6,
    'MISONorth': -6, 'MISOSouth': -6, 'TVA': -6,
    'Southern': -5, 'Florida': -5, 'Carolinas': -5, 'PJM': -5,
    'NYISO': -5, 'ISONE': -5,
    'Pacific': -8, 'Mountain': -7,
    'Texas': -6, 'Central': -6, 'Midwest': -6,
    'Southeast': -5, 'MidAtlantic': -5, 'Northeast': -5,
    'Western': -7, 'Texas': -6, 'Eastern': -5,
    'California': -8, 'MidwestN': -6, 'MidwestS': -6, 
    'USA': -6, 'usa': -6,
}
tzs = {region:zephyr.toolbox.timezone_to_tz(timezones[region]) for region in timezones}

#################
#%% PROCEDURE ###
print(outpath)
print(','.join([str(i) for i in runcases]))
print(','.join(runregions))
sys.stdout.flush()

### Select case
for case in runcases:
    ### Extract case parameters
    co2cap = cases.loc[case,'co2cap']
    if np.isnan(co2cap):
        co2cap = None
    co2price = cases.loc[case,'co2price']
    if np.isnan(co2price):
        co2price = None
    loadyear = cases.loc[case,'loadyear']
    scenario = cases.loc[case,'scenario']
    vreyears = (
        [int(y) for y in cases.loc[case,'vreyears'].split(',')] 
         if type(cases.loc[case,'vreyears']) == str
        else [cases.loc[case,'vreyears']])
    numyears = len(vreyears)
    voll = cases.loc[case,'voll']
    if np.isnan(voll):
        voll = None
    reserve_margin = cases.loc[case,'reserve_margin']
    if np.isnan(reserve_margin):
        reserve_margin = None
    rps = cases.loc[case,'rps']
    if np.isnan(rps):
        rps = None
    level = cases.loc[case,'level']
    unitlevel = cases.loc[case,'unitlevel']
    wacc_gen = cases.loc[case,'wacc_gen']
    if np.isnan(wacc_gen):
        wacc_gen = None
    wacc_trans = cases.loc[case,'wacc_trans']
    if np.isnan(wacc_trans):
        wacc_trans = None
    life_gen = cases.loc[case,'life_gen']
    if np.isnan(life_gen):
        life_gen = None
    life_stor = cases.loc[case,'life_stor']
    if np.isnan(life_stor):
        life_stor = None
    interconnection_scaler = cases.loc[case,'interconnection_scaler']
    region_scaler_file = cases.loc[case,'region_scaler']
    load_scaler = 1

    ###### Distance parameters
    distancepath = os.path.join(
        projpath,'io','cf-2007_2013','{}x','').format(interconnection_scaler)

    ### PV assumptions
    pvpath = os.path.join(
        projpath,'io','cf-2007_2013','{}x','{}','pv','{}-{}t-{:.0f}az','binned',''
    ).format(interconnection_scaler, unitlevel, systemtype, axis_tilt, axis_azimuth)
    def pvfile(zone,bins): 
        return '-nsrdb,icomesh9-{}-{}t-{}az-{:.2f}dcac-{:.0f}USDperkWacyr-{}_{}-{}lcoebins.csv'.format(
            systemtype, axis_tilt, axis_azimuth, dcac, pvcost_bins, unitlevel, zone, bins)
    scaler_pv = cases.loc[case,'scaler_pv'] ### GW per km^2
    ### Scale the PV cost by pvscale
    pvscale = cases.loc[case,'pvscale']

    ### Wind assumptions
    model = cases.loc[case,'wind_model']
    modelsave = model.replace(':','|').replace('/','_')
    loss_system_wind = cases.loc[case,'wind_loss']
    height = cases.loc[case,'wind_height']
    def windfile(zone,bins): 
        return ('-wtkhsds,every2,offset0,onshore-{}-{}m-'
                '{:.0f}pctloss-{:.0f}USDperkWacyr-{}_{}-{}lcoebins.csv').format(
            modelsave, height, loss_system_wind*100, windcost_bins, unitlevel, zone, bins)
    windpath = os.path.join(
        projpath,'io','cf-2007_2013','{}x','{}','wind','{}','binned',''
    ).format(interconnection_scaler, unitlevel, modelsave)
    scaler_wind = cases.loc[case,'scaler_wind'] ### GW per km^2
    vom_wind = cases.loc[case,'vom_wind']
    ### Scale the wind cost by windscale
    windscale = cases.loc[case,'windscale']
    
    ### Storage assumptions
    vom_stor = cases.loc[case,'vom_stor']
    ### Scale the storage cost by storscale
    storscale = cases.loc[case,'storscale']

    ### Hydro assumptions
    include_hydro_res = cases.loc[case,'include_hydro_res']
    include_hydro_ror = cases.loc[case,'include_hydro_ror']
    include_gas = cases.loc[case,'include_gas']
    vom_res = cases.loc[case,'vom_res']

    ### Transmission assumptions
    existing_trans = cases.loc[case,'existing_trans']
    build_ac = cases.loc[case,'build_ac']
    try:
        build_ac_voltage = int(cases.loc[case,'build_ac_voltage'])
        cost_scaler_ac = 1
    except ValueError:
        build_ac_voltage = int(cases.loc[case,'build_ac_voltage'].split('*')[0])
        cost_scaler_ac = float(cases.loc[case,'build_ac_voltage'].split('*')[1])
    
    build_dc = cases.loc[case,'build_dc']
    try:
        build_dc = bool(int(build_dc))
        cost_scaler_dc = 1
    except ValueError:
        build_dc = bool(int(build_dc.split('*')[0]))
        cost_scaler_dc = float(cases.loc[case,'build_dc'].split('*')[1])
    ### Scale the transmission cost by transscale
    transscale = cases.loc[case,'transscale']
    
    try:
        transcost_adder = pd.read_csv(
            os.path.join(projpath,'io','transmission','transmission-adder-{}.csv').format(
                cases.loc[case,'intralevel_transcost']),
            index_col=0)['transcost_adder_MUSD_per_GW_yr'].to_dict()
    except FileNotFoundError:
        print('No transcost_adder: {}'.format(
            cases.loc[case,'intralevel_transcost']))
        sys.stdout.flush()
        transcost_adder = 0

    try:
        region_scaler = pd.read_csv(
            os.path.join(projpath,'io','cost','{}.csv'.format(region_scaler_file)),
            index_col=0)
    except FileNotFoundError:
        print('No regional cost scaler: {}'.format(region_scaler_file))
        sys.stdout.flush()
        region_scaler = 1

    ### Existing infrastructure assumptions
    include_phs = cases.loc[case,'include_phs']
    build_phs = cases.loc[case,'build_phs']

    ### Fossil / nuclear assumptions
    gasprice = 'naturalgas_'+cases.loc[case,'gasprice']
    include_nuclear = cases.loc[case,'include_nuclear']
    build_nuclear = cases.loc[case,'build_nuclear']
    if build_nuclear in ['','none','False','0','0.','false',None]:
        build_nuclear = False
    else:
        build_nuclear = 'Nuclear_' + build_nuclear
    nuclear_gen_min = cases.loc[case,'nuclear_min']
    nuclear_ramp_up = cases.loc[case,'nuclear_ramp_up']
    nuclear_ramp_down = cases.loc[case,'nuclear_ramp_down']

    ###### Inputs for inter-zone transmission
    ### Connectivity matrix
    dftransin = pd.read_csv(
        os.path.join(
            projpath,'io','transmission','reeds-transmission-GW-{}.csv').format(unitlevel),
        index_col=[0,1])
    ### Distance matrix
    dfdistance = pd.read_csv(
        os.path.join(
            projpath,'io','transmission','{}-distance-urbancentroid-km.csv').format(unitlevel),
        index_col=0)

    ### Get level/unitlevel dataframe
    dfregion = pd.read_excel(
        batchfile, sheet_name=unitlevel,
        index_col=0, header=[0,1], 
        dtype={
            ('numbins_pv',unitlevel): int, ('numbins_wind',unitlevel): int,
            ('bins_pv',unitlevel): str, ('bins_wind',unitlevel): str,
            ('numbins_pv',level): int, ('numbins_wind',level): int,
            ('bins_pv',level): str, ('bins_wind',level): str,
        }
    )

    ### Get list of regions to run
    allregions = dfregion['area'][level].unique()
    if 'all' in runregions:
        regions = allregions
    else:
        regions = runregions

    ### Select region
    if len(regions) > 1:
        region_generator = tqdm(regions, desc=str(case))
    else:
        region_generator = regions
    for region in region_generator:
        savename = outpath+'{}-{}{}.p.gz'.format(case, region, savemod)
        if verbose >= 1:
            print(savename)
            sys.stdout.flush()
        if os.path.exists(savename) and (overwrite is False):
            continue
        ### Look up some stuff
        try:
            tz = tzs[region]
        except KeyError:
            tz = zephyr.toolbox.tz_state[region]

        ### Get the list of nodes
        if level == 'usa':
            nodes = dfregion.loc[dfregion['area'][level] == region].index.values
        else:
            nodes = dfstate.loc[dfstate['area'][level] == region].index.values

        ### Bins override - number of bins
        numbins_pv, numbins_wind = {}, {}
        for node in nodes:
            numbins_pv[node] = cases.loc[case,'numbins_pv']
            if pd.isnull(numbins_pv[node]):
                numbins_pv[node] = dfregion.loc[node,('numbins_pv',level)]
            else:
                numbins_pv[node] = int(cases.loc[case,'numbins_pv'])

            numbins_wind[node] = cases.loc[case,'numbins_wind']
            if pd.isnull(numbins_wind[node]):
                numbins_wind[node] = dfregion.loc[node,('numbins_wind',level)]
            else:
                numbins_wind[node] = int(cases.loc[case,'numbins_wind'])

        ### Bins override - specific bins
        bins_pv, bins_wind = {}, {}
        for node in nodes:
            bins_pv[node] = cases.loc[case,'bins_pv']
            if pd.isnull(bins_pv[node]) and pd.isnull(cases.loc[case,'numbins_pv']):
                bins_pv[node] = dfregion.loc[node,('bins_pv',level)]
            elif pd.isnull(bins_pv[node]) and not pd.isnull(cases.loc[case,'numbins_pv']):
                bins_pv[node] = ','.join([str(i) for i in range(numbins_pv[node])])

            bins_wind[node] = cases.loc[case,'bins_wind']
            if pd.isnull(bins_wind[node]) and pd.isnull(cases.loc[case,'numbins_wind']):
                bins_wind[node] = dfregion.loc[node,('bins_wind',level)]
            elif pd.isnull(bins_wind[node]) and not pd.isnull(cases.loc[case,'numbins_wind']):
                bins_wind[node] = ','.join([str(i) for i in range(numbins_wind[node])])

            ### Convert to list of ints
            bins_pv[node] = [int(i) for i in bins_pv[node].split(',')]
            bins_wind[node] = [int(i) for i in bins_wind[node].split(',')]

        ### Capacities
        pv_area = {
            node: pd.read_csv(
                pvpath+'area'+pvfile(node,numbins_pv[node]), 
                index_col='bin_lcoe', squeeze=True).to_dict()
            for node in nodes
        }
        wind_area = {
            node: pd.read_csv(
                windpath+'area'+windfile(node,numbins_wind[node]), 
                index_col='bin_lcoe', squeeze=True).to_dict()
            for node in nodes
        }

        pv_gwcap = {
            node: {jbin: pv_area[node][jbin] * scaler_pv 
            for jbin in range(len(pv_area[node]))} for node in nodes
        }
        wind_gwcap = {
            node: {jbin: wind_area[node][jbin] * scaler_wind 
            for jbin in range(len(wind_area[node]))} for node in nodes
        }

        ### Site CF info
        pv_sites = {
            node: pd.read_csv(
                os.path.join(
                    distancepath,'{}','pv','{}-{}t-{:.0f}az',
                    'mean-nsrdb,icomesh9-{}-{}t-{:.0f}az-{:.2f}dcac-{:.0f}USDperkWacyr-{}_{}.csv'
                ).format(unitlevel, systemtype, axis_tilt, axis_azimuth, 
                         systemtype, axis_tilt, axis_azimuth, dcac, 
                         pvcost_bins, unitlevel, node))
            for node in nodes
        }
        wind_sites = {
            node: pd.read_csv(
                os.path.join(
                    distancepath,'{}','wind','{}',
                    ('mean-wtkhsds,every2,offset0,onshore'
                     + '-{}-{}m-{:.0f}pctloss-{:.0f}USDperkWacyr-{}_{}.csv')
                ).format(unitlevel, modelsave, modelsave, height, 
                         loss_system_wind*100, windcost_bins, unitlevel, node))
            for node in nodes
        }

        ###### Create dicts with area-weighted costs
        ### PV
        pv_trans_cost = {}
        for node in nodes:
            for jbin in bins_pv[node]:
                df = pv_sites[node].loc[
                    pv_sites[node]['binlcoe_{}'.format(numbins_pv[node])]==jbin].copy()
                pv_trans_cost[(node,jbin)] = (df.km2 * df.cost_trans_annual).sum() / df.km2.sum()
        ### Wind
        wind_trans_cost = {}
        for node in nodes:
            for jbin in bins_wind[node]:
                df = wind_sites[node].loc[
                    wind_sites[node]['binlcoe_{}'.format(numbins_wind[node])]==jbin].copy()
                wind_trans_cost[(node,jbin)] = (df.km2 * df.cost_trans_annual).sum() / df.km2.sum()

        ####### Load
        ### Get aggregated state loads
        dfnode = pd.read_csv(
            os.path.join(projpath,'io','load','EFS','central-2007_2013-{}-{}-{}.csv.gz').format(
                scenario,loadtype,loadyear),
            index_col=0, parse_dates=True)

        ### Sum the loads for unitlevels
        dfload = {}
        for node in nodes:
            states = dfstate.loc[dfstate['area'][unitlevel]==node].index.values
            dfload[node] = pd.concat({state: dfnode[state] for state in states}, axis=1).sum(axis=1)
        dfload = pd.concat(dfload, axis=1).tz_convert(tz)
        # ### Correct leap day transfer; shouldn't do anything for USA timezone (-6)
        if tz == 'Etc/GMT+6':
            pass
        elif tz == 'Etc/GMT+5':
            dfload.index = dfload.index.map(
                lambda t: t.replace(day=1).replace(month=3) if ((t.month==2) and (t.day==29)) else t)
            ### Add the 2007-01-01 value for eastern timezone
            dfload.loc[dfload.index[0] - pd.Timedelta('1H')] = dfload.loc['2008-01-01 00:00-05:00']
            dfload.sort_index(inplace=True)
        else:
            dfload.index = dfload.index.map(
                lambda t: t.replace(day=28) if ((t.month==2) and (t.day==29)) else t)
        #######################
        ### Add load scaler (20200305)
        dfload = dfload * load_scaler

        ###### VRE series
        ### Wind
        dictwindall = {
            node: pd.read_csv(
                windpath+'cf'+windfile(node,numbins_wind[node]), 
                index_col=0, parse_dates=True)
            .tz_convert(tz)
            ### * (1 - loss_system_wind) 
            ### For this version, wind losses are already factored
            ### into the CF file (since we need the real CF to calculate the LCOE).
            ### Can't just go back and change the CF, because it would alter the 
            ### balance between generator and transmission costs and change the relative
            ### contribution of generator CF vs distance to transmission assets.
            ### So to change the losses, would need to redo the CF bins for wind.
            for node in nodes
        }

        ### Drop leap days
        for node in nodes:
            dictwindall[node] = (
                dictwindall[node].drop(dictwindall[node].loc['2008-02-29'].index)
                .drop(dictwindall[node].loc['2012-02-29'].index))
            dictwindall[node] = dictwindall[node].tz_convert(tz)
            ### NEW 20200308: truncate between 0-truncmid to 0; truncmid-truncmax to truncmax
            if verbose >= 3:
                print(node)
                print(dictwindall[node].describe())
            for col in dictwindall[node].columns:
                dictwindall[node].loc[dictwindall[node][col] < truncmid, col] = 0
                dictwindall[node].loc[
                ((dictwindall[node][col] >= truncmid)
                 & (dictwindall[node][col] < truncmax)), col] = truncmax
            if verbose >= 3:
                print(dictwindall[node].describe())
                sys.stdout.flush()
            ### continue

        dfwindall = pd.concat(dictwindall, axis=1)

        ### PV
        dictpvall = {
            node: pd.read_csv(
                pvpath+'cf'+pvfile(node,numbins_pv[node]),
                index_col=0, parse_dates=True,
            ).tz_convert(tz)
            for node in nodes
        }

        ### Drop leap days
        for node in nodes:
            dictpvall[node] = (
                dictpvall[node].drop(dictpvall[node].loc['2008-02-29'].index)
                .drop(dictpvall[node].loc['2012-02-29'].index))
            dictpvall[node] = dictpvall[node].tz_convert(tz)
            ### NEW 20200308: truncate between 0-truncmid to 0; truncmid-truncmax to truncmax
            if verbose >= 3:
                print(node)
                print(dictpvall[node].describe())
            for col in dictpvall[node].columns:
                dictpvall[node].loc[dictpvall[node][col] < truncmid, col] = 0
                dictpvall[node].loc[
                ((dictpvall[node][col] >= truncmid)
                 & (dictpvall[node][col] < truncmax)), col] = truncmax
            if verbose >= 3:
                print(dictpvall[node].describe())
                sys.stdout.flush()
            ### continue

        dfpvall = pd.concat(dictpvall, axis=1)

        ###### Hydro (reservoir)
        ### Get existing power capacity 
        hydrorescap = pd.read_csv(
            os.path.join(projpath,'io','hydro','hydro-CHcap-ORNL-GW-nonRORonly.csv'), 
            index_col=0, squeeze=True,
        ).to_dict()
        dfhydroresin = pd.read_csv(
            os.path.join(
                projpath,'io','hydro','hydro-MonthlyGenEIA923-2006_2017-GWh-nonRORonly.csv'), 
            parse_dates=True, index_col=0)
        # print('dfhydroresin\n', dfhydroresin.head()) ### DEBUG

        dicthydrores = {}
        hasreses = {}
        rescap = {}
        for node in nodes:
            states = dfstate.loc[dfstate['area'][unitlevel]==node].index.values
            # print('states', states) ### DEBUG
            dicthydroresnode = {}
            hasresesnode = {}
            for state in states:
                ### Get monthly generation, select state, filter to vreyears
                dfhydroresnode = dfhydroresin.copy()
                ### Check if state has hydro Res
                if state in dfhydroresnode.columns:
                    hasres = True
                    # print('state in dfhydroresnode.columns') ### DEBUG
                    dfhydroresnode = dfhydroresnode[state].loc[
                        slice('{}-01-01'.format(vreyears[0]), '{}-01-01'.format(vreyears[-1]+1))
                    ].copy()

                    ### Convert to average daily generation in GWh
                    dfhydroresnode = dfhydroresnode / dfhydroresnode.index.map(
                        lambda x: zephyr.toolbox.monthhours(x.year,x.month)/24)
                    ### Upsample to daily timeseries
                    dfhydroresnode = dfhydroresnode.resample('1D').ffill().loc[
                        slice(str(vreyears[0]), str(vreyears[-1]))].copy()
                    ### Name it
                    dfhydroresnode = dfhydroresnode.rename('res')

                    ###### IMPORTANT: Clip hydro timeseries to <= power capacity
                    dfhydroresnode = dfhydroresnode.clip(upper=(hydrorescap[state]*24))

                    ### Store it
                    dicthydroresnode[state] = dfhydroresnode
                else:
                    hasres = False
                ### Store it
                hasresesnode[state] = hasres

            ### Aggregate the state hydros
            hasres = any([hasresesnode[state] for state in states])
            if hasres:
                dfhydroresnode = pd.concat(dicthydroresnode,axis=1).sum(axis=1)
                rescap[node] = 0
                for state in states:
                    if hasresesnode[state]:
                        rescap[node] += hydrorescap[state]
                ### Store it if it has hydrores
                dicthydrores[node] = dfhydroresnode
            ### Record whether it has hydrores
            hasreses[node] = hasres
        ### Record if there's any res in the node
        hasres = any([hasreses[node] for node in nodes])
        if hasres:
            dfhydrores = pd.concat(dicthydrores, axis=1)
        else:
            dfhydrores = None

        ###### Hydro (ROR)
        ### Get existing power capacity
        dfhydrororin = pd.read_csv(
            os.path.join(
                projpath,'io','hydro','hydro-MonthlyGenEIA923-2006_2017-GWh-RORonly.csv'), 
            parse_dates=True, index_col=0)
        dicthydroror = {}
        dictrormax = {}
        hasrors = {}
        for node in nodes:
            states = dfstate.loc[dfstate['area'][unitlevel]==node].index.values
            dicthydrorornode = {}
            dictrormaxnode = {}
            hasrorsnode = {}
            for state in states:
                df = dfhydrororin.copy()
                ### Check if state has hydro ROR
                if state in df.columns:
                    hasror = True
                    df = df[state].copy()
                    ### Convert to average hourly generation in GW
                    df = df / df.index.map(
                        lambda x: zephyr.toolbox.monthhours(x.year,x.month))
                    ### Get max gen
                    rormax = float(df.max())
                    ### Subset to simulation year(s)
                    df = df.loc[
                        slice('{}-01-01'.format(vreyears[0]), '{}-01-01'.format(vreyears[-1]+1))
                    ].rename('ror')
                    ### Upsample to hourly timeseries
                    df = df.tz_localize(tz).resample('1H').ffill().loc[
                        slice(str(vreyears[0]), str(vreyears[-1]))].copy()

                    ### Store it
                    dicthydrorornode[state] = df
                    dictrormaxnode[state] = rormax
                else:
                    hasror = False
                ### Store it
                hasrorsnode[state] = hasror

            ### Aggregate the state RORs
            dictrormax[node] = sum(dictrormaxnode.values())
            hasrors[node] = any(hasrorsnode.values())
            if hasrors[node]:
                dicthydroror[node] = pd.concat(dicthydrorornode,axis=1).sum(axis=1)

        hasror = any(hasrors.values())
        if hasror:
            dfhydroror = pd.concat(dicthydroror, axis=1)
            ### Convert to CF
            for col in dfhydroror.columns:
                dfhydroror[col] = dfhydroror[col] / dictrormax[col]
        else:
            dfhydroror = None


        ###### Merge it all
        dfrun = pd.concat(
            {**{'load': pd.concat({'load':dfload},axis=1), 
                'pv': dfpvall, 
                'wind': dfwindall},
             **({'reshydro': pd.concat({'res':dfhydrores},axis=1).tz_localize(tz)} 
                if hasres else {}),
             **({'rorhydro': pd.concat({'ror':dfhydroror},axis=1).fillna(0.)} 
                if hasror else {}),
            },
            axis=1
        ).tz_convert(tz).loc[slice(str(vreyears[0]),str(vreyears[-1]))]#.fillna(0.)
        dfrun['pv'] = dfrun['pv'].fillna(0.)

        ### SPECIAL CASE for WTK: Drop the last day if 2013 is included
        if 2013 in vreyears:
            dfrun.drop(dfrun.loc['2013-12-31'].index, inplace=True)
        ### NEW: Drop the null values (important for EFS since it doesn't include leap days)
        dfrun = dfrun.loc[dfrun['wind'].iloc[:,0].notnull()].copy()

        ### Make sure the timeseries covers an integer number of days
        if len(dfrun) % 24 != 0:
            raise Exception(
                'Timeseries must be daily (divisble by 24) but len(dfrun) == {}'.format(
                    len(dfrun)))
        ### Make sure the timeseries starts at midnight in local timezone
        if dfrun.index[0].hour != 0:
            raise Exception(
                'Timeseries must start at mignight locally but starts at {}'.format(
                    dfrun.index[0].hour))
        ### Debugging: Check for nans
        if verbose >= 3:
            print(dfrun.count().value_counts())
            sys.stdout.flush()

        #####################
        ### Existing capacity
        ###### PHS
        if include_phs:
            phscap_state = pd.read_csv(
                os.path.join(projpath,'io','hydro','PHS-GW-EIA860_2018.csv'), 
                index_col=0, squeeze=True,
            ).to_dict()
            ### Aggregate by node
            phscap = {}
            for node in nodes:
                states = dfstate.loc[dfstate['area'][unitlevel]==node].index.values
                phscap[node] = sum([phscap_state.get(state, 0) for state in states])

        ###### Nuclear
        if include_nuclear:
            nuclearcap_state = pd.read_csv(
                os.path.join(projpath,'io','nuclear','nuclear-GW-EIA860_2018-post{}.csv').format(
                    include_nuclear), 
                index_col=0, squeeze=True,
            ).to_dict()
            ### Aggregate by node
            nuclearcap = {}
            for node in nodes:
                states = dfstate.loc[dfstate['area'][unitlevel]==node].index.values
                nuclearcap[node] = sum([nuclearcap_state.get(state, 0) for state in states])

        #####################
        ### Set up the system
        system = zephyr.cpm.System(region, periods=[0], nodes=list(nodes))
        system.hours = {0:len(dfrun)}
        system.years = numyears

        ###### Transmission
        ### Subset connectivity matrix
        dftrans = dftransin.loc[
            dftransin.index.map(lambda x: (x[0] in nodes) and (x[1] in nodes))
        ].copy()

        ### Make the lines
        for pair in dftrans.index:
            ##### Existing lines
            node1, node2 = pair
            ### AC
            if existing_trans and (dftrans.loc[pair,'AC'] > 0):
                name = '{}|{}_ac_old'.format(*pair)
                system.add_line(
                    name=name,
                    line=zephyr.cpm.Line(
                        node1=node1, node2=node2,
                        distance=dfdistance.loc[node1,node2],
                        capacity=dftrans.loc[pair,'AC'],
                        defaults=defaults, name=name,
                        newbuild=False,
                        ### (voltage doesn't matter since they all have the same losses)
                        voltage=345, 
                    )
                )
            ### DC
            if existing_trans and (dftrans.loc[pair,'DC'] > 0):
                name = '{}|{}_dc_old'.format(*pair)
                system.add_line(
                    name=name,
                    line=zephyr.cpm.Line(
                        node1=node1, node2=node2,
                        distance=dfdistance.loc[node1,node2],
                        capacity=dftrans.loc[pair,'DC'],
                        defaults=defaults, name=name,
                        newbuild=False, 
                        ### (need to specify the voltage because DC has lower losses)
                        voltage='DC',
                    )
                )
            ##### New lines
            #### AC
            ### Only allow them to be built if there's existing AC capacity
            ### (to make sure we don't build AC across an asynchronous interconnect)
            ### and if the switch is on
            if build_ac and (dftrans.loc[pair,'AC'] > 0):
                name = '{}|{}_ac_new'.format(*pair)
                line = zephyr.cpm.Line(
                    node1=node1, node2=node2,
                    distance=dfdistance.loc[node1,node2],
                    voltage=build_ac_voltage,
                    defaults=defaults, name=name,
                    wacc=wacc_trans,
                    newbuild=True,
                )
                ### Adjust cost_annual by the input multiplier
                line.cost_annual = line.cost_annual * cost_scaler_ac * transscale
                ### Make the line
                system.add_line(name=name, line=line)

            #### DC
            ### Can build these anywhere there's an existing AC or DC 
            ### line (including across interconnects), but only if the switch is on
            ### Specific for this version: Can only build these if there's an existing
            ### DC line between nodes and NOT an existing AC line between them.
            ### In a later verion, allow this restriction to be relaxed.
            if build_dc and (dftrans.loc[pair,'DC'] > 0) and (dftrans.loc[pair, 'AC'] == 0):
                name = '{}|{}_dc_new'.format(*pair)
                line = zephyr.cpm.Line(
                    node1=node1, node2=node2,
                    distance=dfdistance.loc[node1,node2],
                    voltage='DC',
                    defaults=defaults, name=name,
                    wacc=wacc_trans,
                    newbuild=True,
                )
                ### Adjust cost_annual by the input multiplier
                line.cost_annual = line.cost_annual * cost_scaler_dc * transscale
                ### Make the line
                system.add_line(name=name, line=line)

        ### Add dispatchable generators based on defaults
        if include_gas:
            for node in nodes:
                for gen in ['OCGT', 'CCGT']:
                    system.add_generator(
                        '{}_{}'.format(gen,node), 
                        zephyr.cpm.Gen(
                            gen, defaults=defaults, fuel=gasprice, wacc=wacc_gen, 
                            lifetime=life_gen,
                        ).localize(node)
                    )

        ###### Add curtailable load
        if voll not in [None, np.nan]:
            for node in nodes:
                system.add_generator(
                    'Lostload_{}'.format(node), 
                    zephyr.cpm.Gen(
                        'Lostload', defaults=defaults, cost_vom=voll).localize(node)
                )

        ### Add storage
        annualcost_E = zephyr.cpm.Storage(
            cases.loc[case,'stor'], defaults=defaults, wacc=wacc_gen,
            lifetime=life_stor,).cost_annual_E * storscale
        annualcost_P = zephyr.cpm.Storage(
            cases.loc[case,'stor'], defaults=defaults, wacc=wacc_gen,
            lifetime=life_stor,).cost_annual_P * storscale
        for node in nodes:
            ### Apply regional cost scaler, if applicable
            if type(region_scaler) == pd.DataFrame:
                annualcost_E_node = annualcost_E * region_scaler.loc[node, 'Stor']
                annualcost_P_node = annualcost_P * region_scaler.loc[node, 'Stor']
            else:
                annualcost_E_node = annualcost_E * 1
                annualcost_P_node = annualcost_P * 1
            system.add_storage(
                'Stor_{}'.format(node), 
                zephyr.cpm.Storage(
                    cases.loc[case,'stor'], defaults=defaults, wacc=wacc_gen,
                    cost_annual_E=annualcost_E_node, cost_annual_P=annualcost_P_node,
                    cost_vom=vom_stor,
                ).localize(node)
            )

        ### Add existing PHS if applicable
        if include_phs:
            for node in nodes:
                if node in phscap:
                    system.add_storage(
                        'PHS_{}'.format(node),
                        (zephyr.cpm.Storage(
                            'PHS', defaults=defaults, cost_capex_E=0, cost_capex_P=0,)
                         .add_duration_min(include_phs).add_duration_max(include_phs)
                         .add_power_bound_hi(phscap[node])
                         # .add_power_bound_lo(phscap[node]) ### 20200429 removed
                         .localize(node))
                    )

        ### Add existing nuclear if applicable
        if include_nuclear:
            for node in nodes:
                if node in nuclearcap:
                    system.add_generator(
                        'Nuclear_{}_old'.format(node),
                        (zephyr.cpm.Gen('Nuclear_existing', defaults=defaults)
                         .add_capacity_bound_hi(nuclearcap[node])
                         ### Following line is commented out to allow retirements
                         #.add_capacity_bound_lo(nuclearcap[node])
                         .localize(node)
                        )
                    )

        ### Add new nuclear if applicable
        if build_nuclear:
            for node in nodes:
                system.add_generator(
                    'Nuclear_{}_new'.format(node),
                    zephyr.cpm.Gen(
                        build_nuclear, defaults=defaults, wacc=wacc_gen,
                        lifetime=life_gen,
                        gen_min=nuclear_gen_min, 
                        ramp_up=nuclear_ramp_up, ramp_down=nuclear_ramp_down,
                    ).localize(node)
                )

        ###### Make the VRE generators
        index = dfrun.index.copy()
        ###### PV
        ### Get the generator annual cost
        annualcost = zephyr.cpm.Gen(
            cases.loc[case,'pv'], defaults=defaults, wacc=wacc_gen,
            lifetime=life_gen,).cost_annual * pvscale
        for node in nodes:
            ### Apply the regional cost multiplier, if applicable
            if type(region_scaler) == pd.DataFrame:
                annualcost_node = annualcost * region_scaler.loc[node, 'PV']
            else:
                annualcost_node = annualcost * 1
            ### Add the generator to the system
            for jbin in bins_pv[node]:
                system.add_generator(
                    'PV_{}_{}'.format(node,jbin),
                    zephyr.cpm.Gen(
                        cases.loc[case,'pv'], defaults=defaults,
                        cost_annual=(
                            annualcost_node
                            ### Add the interconnection cost for the bin
                            + pv_trans_cost[node,jbin]
                            ### Add the intralevel transmission cost
                            + (transcost_adder[node] * cost_scaler_ac * transscale
                                if type(transcost_adder) is dict
                               else transcost_adder * cost_scaler_ac * transscale)
                        )
                    )
                    .add_capacity_bound_hi(pv_gwcap[node][jbin])
                    .add_availability(dfrun.pv[node][str(jbin)], period=0)
                    .localize(node)
                )

        ###### Wind
        ### Get the generator annual cost
        annualcost = zephyr.cpm.Gen(
            cases.loc[case,'wind'], defaults=defaults, wacc=wacc_gen,
            lifetime=life_gen,).cost_annual * windscale
        for node in nodes:
            ### Apply the regional cost multiplier, if applicable
            if type(region_scaler) == pd.DataFrame:
                annualcost_node = annualcost * region_scaler.loc[node, 'Wind']
            else:
                annualcost_node = annualcost * 1
            ### Add the generator to the system
            for jbin in bins_wind[node]:
                system.add_generator(
                    'Wind_{}_{}'.format(node,jbin),
                    zephyr.cpm.Gen(
                        cases.loc[case,'wind'], defaults=defaults, cost_vom=vom_wind,
                        cost_annual=(
                            annualcost_node
                            ### Add the interconnection cost for the bin
                            + wind_trans_cost[node,jbin]
                            ### Add the intralevel transmission cost
                            + (transcost_adder[node] * cost_scaler_ac * transscale
                               if type(transcost_adder) is dict
                               else transcost_adder * cost_scaler_ac * transscale)
                        )
                    )
                    .add_capacity_bound_hi(wind_gwcap[node][jbin])
                    .add_availability(dfrun.wind[node][str(jbin)], period=0)
                    .localize(node)
                )

        ### Add load to model
        for node in nodes:
            system.add_load(node, dfrun.load.load[node], period=0)

        ### Set CO2 cap (unnecessary if there's no fossil)
        if co2cap not in [None,np.nan]:
            system.set_co2cap(co2cap)

        ### Set CO2 price if desired
        if co2price not in [None,np.nan]:
            system.set_co2price(co2price)
            ### DEBUG ###
            if verbose >= 4:
                for gen in system.generators:
                    print(system.generators[gen].__dict__)

        ### Set RPS (unnecessary if there's no fossil or voll)
        # if (rps not in [None,np.nan]) and (voll not in [None,np.nan]):
        if (rps not in [None,np.nan]):
            system.set_rps(rps)

        ### Set reserve margin constraint
        if (reserve_margin not in [None,np.nan]):
            system.set_reserves(reserve_margin)

        ###### Make the Reservoir hydro and add its availability
        if include_hydro_res == True: 
            for node in nodes:
                if hasreses[node]:
                    hydrores = (
                        zephyr.cpm.HydroRes(
                            'Hydro_Res', defaults=defaults,
                            newbuild=False, balancelength='day',
                            cost_vom=vom_res,
                        )
                        .add_capacity_bound_hi(rescap[node])
                        .add_availability(dfrun.reshydro.res[node].iloc[::24], period=0)
                        .localize(node)
                    )
                    system.add_reshydro('Hydro_Res_{}'.format(node), hydrores)

        ###### Make the ROR hydro and add its availability
        if include_hydro_ror == True:
            for node in nodes:
                if hasrors[node]:
                    hydroror = (
                        zephyr.cpm.Gen(
                            'Hydro_ROR', defaults=defaults, cost_capex=0,)
                        ### TODO: Should we include lower bound or let it be retired?
                        .add_capacity_bound_lo(dictrormax[node]) 
                        .add_capacity_bound_hi(dictrormax[node])
                        .add_availability(dfrun.rorhydro.ror[node], period=0)
                        .localize(node)
                    )
                    system.add_generator('Hydro_ROR_{}'.format(node), hydroror)

        ### Solve it
        _ = zephyr.cpm.model(
            system, solver=solver, verbose=verbose, 
            includereserves=includereserves, includedual=includedual,
            savename=savename,
            Method=method,
        )
        if verbose:
            try:
                # (output_capacity, output_operation, output_values, system) = _
                print('{:.2f} $/MWh'.format(_[2]['lcoe']))
                sys.stdout.flush()
            except TypeError as err:
                print(err)
                sys.stdout.flush()
                continue
