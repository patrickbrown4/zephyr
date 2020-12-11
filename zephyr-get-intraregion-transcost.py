"""
Use this script to load a set of planning-area (PA) runs and calculate inter-state intra-PA
transmission cost modifiers for PV and wind to use in full-US runs.
This script expects the model outputs to be saved as .p.gz files.
"""

###############
#%% IMPORTS ###
import pandas as pd
import os, sys, pickle, gzip
from tqdm import tqdm, trange

import zephyr
projpath = zephyr.settings.projpath

#######################
#%% ARGUMENT INPUTS ###

import argparse
parser = argparse.ArgumentParser(description='get intraregion transmission costs')
parser.add_argument('case', type=int, help='integer case number from cases-{batchname}.xlsx')
parser.add_argument('batchname', type=str, help='batch name for cases-{batchname}.xlsx')
parser.add_argument('batchpath', type=str, default=os.path.join(projpath,'out',''),
                    help='parent directory for output files, leaving out {batchname}/')
### Parse it
args = parser.parse_args()
case = args.case
batchname = args.batchname
batchpath = os.path.join(args.batchpath, batchname, '')

#################
#%% FUNCTIONS ###

def get_dfout(dictout_cap, dictout_op, dictout_values, system, periods, nodes, years=7,):
    ###### Get capacities
    dfcap = pd.DataFrame({period: dictout_cap for period in periods}).T

    ### Get values
    dfvalues = pd.DataFrame({period: dictout_values for period in periods}).T

    ###### Get operation
    dfops = pd.concat(
        {k: pd.DataFrame(v).iloc[:-1]
         for k,v in dictout_op.items()}, 
        axis=1)

    for period in periods:
        dfops.loc[:,(period,'PV_all')] = dfops[period][
            [col for col in dfops[period].columns if col.startswith('PV')]].sum(axis=1)
        dfops.loc[:,(period,'Wind_all')] = dfops[period][
            [col for col in dfops[period].columns if col.startswith('Wind')]].sum(axis=1)
        for node in nodes:
            dfops.loc[:,(period,'Stor_{}_power'.format(node))] = (
                dfops[period]['Stor_{}_discharge'.format(node)] 
                + dfops[period]['Stor_{}_charge'.format(node)])
        dfops.sort_index(axis=1, level=0, inplace=True)

    ###### Get cumulative energy generation
    dfenergy = dfops.sum().unstack(level=0).T
        
    ###### Calculate cost
    dfcost = {}
    for gen in system.generators:
        dfcost[gen] = dfcap[gen] * years * (
            system.generators[gen].cost_annual + system.generators[gen].cost_fom
        ) + dfenergy[gen] * (
            system.generators[gen].cost_fuel + system.generators[gen].cost_vom
        )
    ### Include Reservoir hydro
    for reshydro in system.reshydros:
        dfcost[reshydro] = dfcap[reshydro] * years * (
            system.reshydros[reshydro].cost_annual + system.reshydros[reshydro].cost_fom
        ) + dfenergy['{}_power'.format(reshydro)] * system.reshydros[reshydro].cost_vom

    ### Include storage
    for stor in system.storages:
        dfcost[stor] = (
            dfcap['{}_P'.format(stor)] * (
                system.storages[stor].cost_annual_P + system.storages[stor].cost_fom_P) * years
            + dfcap['{}_E'.format(stor)] * (
                system.storages[stor].cost_annual_E + system.storages[stor].cost_fom_E) * years
            + dfenergy['{}_discharge'.format(stor)] * system.storages[stor].cost_vom)
        
    ### Include transmission
    for line in system.lines:
        dfcost[line] = (
            dfcap[line] * (system.lines[line].cost_annual + system.lines[line].cost_fom) * years
        )
    
    ### Concat it
    dfcost = pd.concat(dfcost, axis=1)
    

    ### Group all outputs to save
    dfout = pd.concat(
        {'cap':dfcap,'energy':dfenergy,'values':dfvalues,'cost':dfcost},
        axis=1, sort=False
    )
    
    return dfout, dfops

#################
#%% PROCEDURE ###

#%%% Load the batch files
cases = pd.read_excel(
    os.path.join(batchpath,'cases.xlsx'), index_col='case', sheet_name='runs',
    dtype={'include_hydro_res':bool, 'include_hydro_ror':bool,
           'include_gas':bool, 'build_ac':bool, 'build_dc': str,
           'include_phs':int, 'build_phs':str, 'gasprice': str,
           'include_nuclear':int, 'build_nuclear':str,
           'bins_pv':str, 'bins_wind':str}
)
caseyear = dict(zip(cases.index.values,cases.vreyears.values))
### Constituent zones
dfstate = pd.read_excel(
    os.path.join(batchpath,'cases.xlsx'), 
    sheet_name='state', index_col=0, header=[0,1],
)
states = dfstate.index.values

### PAs
dfregion = pd.read_excel(
    os.path.join(batchpath,'cases.xlsx'), 
    sheet_name='ba', index_col=0, header=[0,1],
    dtype={('bins_pv','ba'):str,('bins_wind','ba'):str},
).xs('ba',axis=1,level=1)

bas = list(dfstate['area']['ba'].unique())

#%% Load the results
dictout = {}
for ba in tqdm(bas):
    infile = os.path.join(batchpath,'results','{}-{}.p.gz'.format(case,ba))
    with gzip.open(infile, 'rb') as p:
        (dictout_cap, dictout_op, dictout_values, system) = pickle.load(p)
    ### Get list of states
    states = dfstate.loc[dfstate.area.ba == ba].index.values
    
    intup = get_dfout(
        dictout_cap=dictout_cap, dictout_op=dictout_op, dictout_values=dictout_values, 
        system=system, periods=[0], nodes=states, years=len(caseyear[case].split(',')))
    dictout[(case,ba)] = intup[0].loc[0]

#%%### Save the inter-state intra-PA transmission cost adders
dictsave = {}
for ba in bas:
    dictsave['transcost_ac_annual'] = [
        dictout[(case,ba)]['cost']
        [[c for c in dictout[(case,ba)]['cost'].index if '|' in c]]
        .sum() / 7
        for ba in bas
        if (case,ba) in dictout.keys()
    ]
    dictsave['cap_pv'] = [
        dictout[(case,ba)]['cap']
        [[c for c in dictout[(case,ba)]['cost'].index if c.startswith('PV')]]
        .sum()
        for ba in bas
        if (case,ba) in dictout.keys()
    ]
    dictsave['cap_wind'] = [
        dictout[(case,ba)]['cap']
        [[c for c in dictout[(case,ba)]['cost'].index if c.startswith('Wind')]]
        .sum()
        for ba in bas
        if (case,ba) in dictout.keys()
    ]
# dfsave = pd.DataFrame(dictsave, index=bas)
dfsave = pd.DataFrame(dictsave, index=[ba for ba in bas if (case,ba) in dictout.keys()])
dfsave['cap_vre'] = dfsave.cap_pv + dfsave.cap_wind
dfsave['transcost_adder_MUSD_per_GW_yr'] = dfsave.transcost_ac_annual / dfsave.cap_vre
### Save it
dfsave.to_csv(
    os.path.join(
        projpath,'io','transmission',
        'transmission-adder-ba-{}_{}.csv'.format(batchname,case))
)
print(dfsave)
