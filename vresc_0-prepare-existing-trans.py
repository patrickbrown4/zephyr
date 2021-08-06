"""
Note that you'll need access to the ReEDS model in order to download
the existing-transmission-capacity data used in this script.
Request access at https://www.nrel.gov/analysis/reeds/request-access.html;
then you can access the ReEDS repository at https://github.com/NREL/ReEDS_OpenAccess.
Note that the text after 'token' in the link to 'transmission_capacity_initial.csv' seems finicky. 
If you get a 404 error, go to 
https://github.com/NREL/ReEDS_OpenAccess/blob/master/inputs/transmission/transmission_capacity_initial.csv,
click on "Raw", and paste the url of that file into the definition of `transfile` below,
or download the file manually and overwrite the `transfile` path below.
"""
###############
#%% IMPORTS ###

import pandas as pd
import os
import geopandas as gpd

import zephyr
projpath = zephyr.settings.projpath
datapath = zephyr.settings.datapath

##############
#%% INPUTS ###

###### ReEDS BA-to-state map
### To start, these can be identified manually from the ReEDS documentation
### (https://www.nrel.gov/docs/fy20osti/74111.pdf, Figure 28)
### or from the BA-to-county map at
### https://github.com/NREL/ReEDS_OpenAccess/blob/master/hourlize/inputs/resource/county_map.csv.
### However, we reassign a few to maintain interconnect boundaries - for example,
### El Paso is assigned to NM so that we don't get AC capacity between the western and texas
### interconnects. This reassignment only applies to transmission capacity, not to
### the land area used to determine developable wind/solar capacity.
pca2state = {
    1: 'WA',
    2: 'WA',
    3: 'WA',
    4: 'WA',
    5: 'OR',
    6: 'OR',
    7: 'OR',
    8: 'CA',
    9: 'CA',
    10: 'CA',
    11: 'CA',
    12: 'NV',
    13: 'NV',
    14: 'ID',
    15: 'ID',
    16: 'ID',
    17: 'MT',
    18: 'MT',
    19: 'MT',
    20: 'MT',
    21: 'WY',
    22: 'WY',
    23: 'WY',
    24: 'WY',
    25: 'UT',
    26: 'UT',
    27: 'AZ',
    28: 'AZ',
    29: 'AZ',
    30: 'AZ',
    31: 'NM',
    32: 'WY', # 32: 'SD',
    33: 'CO',
    34: 'CO',
    35: 'ND', # 35: 'MT',
    36: 'ND',
    37: 'ND',
    38: 'SD',
    39: 'NE',
    40: 'NE',
    41: 'NE',
    42: 'MN',
    43: 'MN',
    44: 'MN',
    45: 'IA',
    46: 'WI',
    47: 'OK', # 47: 'NM',
    48: 'OK', # 48: 'TX',
    49: 'OK',
    50: 'OK',
    51: 'OK',
    52: 'KS',
    53: 'KS',
    54: 'MO',
    55: 'MO',
    56: 'AR',
    57: 'LA',#57: 'TX',
    58: 'LA',
    59: 'NM', #59: 'TX',
    60: 'TX',
    61: 'TX',
    62: 'TX',
    63: 'TX',
    64: 'TX',
    65: 'TX',
    66: 'LA', #66: 'TX',
    67: 'TX',
    68: 'MN',
    69: 'IA',
    70: 'IA',
    71: 'MO',
    72: 'MO',
    73: 'MO',
    74: 'MI',
    75: 'WI',
    76: 'WI',
    77: 'WI',
    78: 'WI',
    79: 'WI',
    80: 'IL',
    81: 'IL',
    82: 'IL',
    83: 'IL',
    84: 'MO',
    85: 'AR',
    86: 'LA',
    87: 'MS',
    88: 'MS',
    89: 'AL',
    90: 'AL',
    91: 'FL',
    92: 'TN',
    93: 'KY',
    94: 'GA',
    95: 'SC',
    96: 'SC',
    97: 'NC',
    98: 'NC',
    99: 'VA',
    100: 'VA',
    101: 'FL',
    102: 'FL',
    103: 'MI',
    104: 'MI',
    105: 'IN',
    106: 'IN',
    107: 'IN',
    108: 'KY',
    109: 'KY',
    110: 'KY',
    111: 'OH',
    112: 'OH',
    113: 'OH',
    114: 'OH',
    115: 'PA',
    116: 'WV',
    117: 'WV',
    118: 'VA',
    119: 'PA',
    120: 'PA',
    121: 'MD',
    122: 'PA',
    123: 'MD',
    124: 'VA',
    125: 'DE',
    126: 'NJ',
    127: 'NY',
    128: 'NY',
    129: 'VT',
    130: 'NH',
    131: 'MA',
    132: 'CT',
    133: 'RI',
    134: 'ME',
}

#################
#%% PROCEDURE ###

#################################
#%% Get BAs from NREL ReEDS model
### From github
transfile = (
    'https://raw.githubusercontent.com/NREL/ReEDS_OpenAccess/master/inputs/transmission/'
    'transmission_capacity_initial.csv?token=AF7WCW2XIUFI6MVHQCPFMH273PTEY'
)
### Locally
# transfile = '/your/path/to/transmission_capacity_initial.csv'
dftrans = pd.read_csv(transfile)

#%% Map ReEDS BAs to states
dftrans['state1'] = dftrans.r.map(lambda x: pca2state.get(int(x[1:]), 'None'))
dftrans['state2'] = dftrans.rr.map(lambda x: pca2state.get(int(x[1:]), 'None'))

#%%### Record AC and DC capacities
transcap_ac = {}
transcap_dc = {}
for i in dftrans.index:
    state1, state2 = dftrans.loc[i,'state1'], dftrans.loc[i,'state2']
    state1, state2 = sorted([state1, state2])
    if (state1 == state2) or (state1 == 'None') or (state2 == 'None'):
        continue
    else:
        ### AC
        if (state1,state2) in transcap_ac:
            transcap_ac[(state1,state2)] += dftrans.loc[i,'AC']
        else:
            transcap_ac[(state1,state2)] = dftrans.loc[i,'AC']
        ### DC
        if (state1,state2) in transcap_dc:
            transcap_dc[(state1,state2)] += dftrans.loc[i,'DC']
        else:
            transcap_dc[(state1,state2)] = dftrans.loc[i,'DC']

dfout = pd.concat(
    [pd.DataFrame(transcap_ac,index=['AC']).T,
     pd.DataFrame(transcap_dc,index=['DC']).T], 
    axis=1,
) / 1000 ### Convert to GW
#%% Save it
dfout.to_csv(os.path.join(projpath,'io','transmission','reeds-transmission-GW-state.csv'))

#%%### Now aggregate to our larger PAs
level = 'ba'
dfregionraw = pd.read_excel(
    os.path.join(projpath,'cases-test.xlsx'), 
    sheet_name='state', index_col=0, header=[0,1],
)['area']
### Add the null line
dfregion = dfregionraw.append(pd.DataFrame({'ba':['None'],}, index=['None'])).copy()
### Aggregate
dftrans['{}1'.format(level)] = dftrans.state1.map(
    lambda x: dfregion.loc[x, level])
dftrans['{}2'.format(level)] = dftrans.state2.map(
    lambda x: dfregion.loc[x, level])

### Add AC capacities
transcap_ac = {}
transcap_dc = {}
for i in dftrans.index:
    region1 = dftrans.loc[i,'{}1'.format(level)]
    region2 = dftrans.loc[i,'{}2'.format(level)]
    region1, region2 = sorted([region1, region2])
    if (region1 == region2) or (region1 == 'None') or (region2 == 'None'):
        continue
    else:
        ### AC
        if (region1,region2) in transcap_ac:
            transcap_ac[(region1,region2)] += dftrans.loc[i,'AC']
        else:
            transcap_ac[(region1,region2)] = dftrans.loc[i,'AC']
        ### DC
        if (region1,region2) in transcap_dc:
            transcap_dc[(region1,region2)] += dftrans.loc[i,'DC']
        else:
            transcap_dc[(region1,region2)] = dftrans.loc[i,'DC']

dfout = pd.concat(
    [pd.DataFrame(transcap_ac,index=['AC']).T,
        pd.DataFrame(transcap_dc,index=['DC']).T], 
    axis=1,
) / 1000 ### Convert to GW
#%% Save it
dfout.to_csv(
    os.path.join(projpath,'io','transmission','reeds-transmission-GW-{}.csv'.format(level))
)
