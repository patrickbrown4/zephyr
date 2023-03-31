#%% General imports
import pandas as pd
import numpy as np
import os, site, copy

### Local imports
site.addsitedir(os.path.dirname(os.path.dirname(__file__)))
import zephyr

#%% Inputs
### Path to a local copy of https://github.com/GridMod/RTS-GMLC
rtspath = os.path.expanduser('~/github/RTS-GMLC')
defaultsfile = os.path.join(
    zephyr.settings.projpath, 'io', 'generator_fuel_assumptions.xlsx')

#%%### Functions
def create_system(rtspath, defaultsfile):
    #%% Inputs
    timeindex = pd.date_range('2020-01-01', '2021-01-01', freq='H', closed='left', tz='EST')
    loadmult = 1.10

    ### Set up zonal inputs - one PV/wind profile per RTS zone
    ### Shared RTS inputs
    dfbus = pd.read_csv(
        os.path.join(rtspath, 'RTS_Data','SourceData','bus.csv')
    )
    area2zone = {1:'z1',2:'z2',3:'z3'}
    bus2zone = dfbus.set_index(['Bus ID'])['Area'].map(area2zone)
    dfgen = pd.read_csv(
        os.path.join(rtspath, 'RTS_Data','SourceData','gen.csv')
    )
    dfgen['zone'] = dfgen['Bus ID'].map(bus2zone)
    dfgen['PMax GW'] = dfgen['PMax MW'] / 1000
    ### Rename and reformat columns
    gencap = dfgen.set_index('GEN UID')['PMax MW']

    #%% PV
    gen_pv = (
        pd.read_csv(
            os.path.join(
                rtspath, 'RTS_Data', 'timeseries_data_files', 'PV', 'DAY_AHEAD_pv.csv'))
        .drop(['Year', 'Month', 'Day', 'Period'], axis=1)
        .set_index(timeindex)
    )
    ## Convert to CF
    cfin_pv = (gen_pv / gencap).dropna(axis=1)
    ### Specify PV profiles for each zone
    zone_pv = {'z1': '101_PV_1', 'z2': '215_PV_1', 'z3': '310_PV_1'}

    #%% Rooftop PV
    gen_rtpv = (
        pd.read_csv(
            os.path.join(
                rtspath,'RTS_Data','timeseries_data_files','RTPV','DAY_AHEAD_rtpv.csv'))
        .drop(['Year','Month','Day','Period'], axis=1)
        .set_index(timeindex)
    )
    ## Convert to CF
    cfin_rtpv = (gen_rtpv / gencap).dropna(axis=1)
    ### Specify profiles for each zone
    zone_rtpv = {'z1':'118_RTPV_7', 'z2':'213_RTPV_1', 'z3':'313_RTPV_1'}

    #%% Wind
    gen_wind = (
        pd.read_csv(
            os.path.join(
                rtspath, 'RTS_Data', 'timeseries_data_files', 'WIND', 'DAY_AHEAD_wind.csv'
            )
        )
        .drop(['Year', 'Month', 'Day', 'Period'], axis=1)
        .set_index(timeindex)
    )
    ## Convert to CF
    cfin_wind = (gen_wind / gencap).dropna(axis=1)
    ### Specify wind profiles for each zone
    zone_wind = {'z1': '122_WIND_1', 'z3': '309_WIND_1'}

    #%% Hydro
    gen_hydro = (
        pd.read_csv(
            os.path.join(
                rtspath,'RTS_Data','timeseries_data_files','Hydro','DAY_AHEAD_hydro.csv'))
        .drop(['Year','Month','Day','Period'], axis=1)
        .set_index(timeindex)
    )
    ## Convert to CF
    cfin_hydro = (gen_hydro / gencap).dropna(axis=1)
    ### Specify profiles for each zone
    zone_hydro = {'z1':'122_HYDRO_1', 'z2': '201_HYDRO_4', 'z3':'322_HYDRO_1'}


    #%% Load
    load_in = (
        pd.read_csv(
            os.path.join(
                rtspath, 'RTS_Data', 'timeseries_data_files',
                'Load', 'DAY_AHEAD_regional_Load.csv'))
        .set_index(timeindex)
        .drop(['Year', 'Month', 'Day', 'Period'], axis=1)
        .rename(columns={'1': 'z1', '2': 'z2', '3': 'z3'})
        ## Convert to GW
        / 1000
    ## Scale up by loadmult
    ) * loadmult

    #%% Transmission distances - just take the centroid of the buses in each area
    try:
        import geopandas as gpd
        dfbus = (
            gpd.read_file(
                os.path.join(rtspath,'RTS_Data','FormattedData','GIS','bus.geojson'))
            .to_crs('ESRI:102008')
        )
        dfbus['zone'] = 'z' + dfbus['Area'].astype(str)
        centroids = dfbus.dissolve('zone').centroid
        dfdistance = {}
        for zone in centroids.index:
            dfdistance[zone] = np.sqrt(
                (centroids.x - float(centroids[[zone]].x))**2
                + (centroids.y - float(centroids[[zone]].y))**2
            ) / 1e3
        dfdistance = pd.concat(dfdistance).unstack().round(3)
    except Exception as err:
        print(f'geopandas error: {err}\nFalling back to filler transmission distances')
        dfdistance = pd.DataFrame(
            {
                'z1': [0, 500, 500],
                'z2': [500, 0, 500],
                'z3': [500, 500, 0],
            },
            index=['z1', 'z2', 'z3'],
        )

    links = [('z1', 'z2'), ('z2', 'z3'), ('z1', 'z3')]

    ### Take interface capacity as sum of interface-crossing-line ratings
    branch = pd.read_csv(os.path.join(rtspath,'RTS_Data','SourceData','branch.csv'))
    branch['zoneFrom'] = branch['From Bus'].map(bus2zone)
    branch['zoneTo'] = branch['To Bus'].map(bus2zone)
    branch['interface'] = branch.apply(
        lambda row: '||'.join(sorted([row.zoneFrom, row.zoneTo])),
        axis=1)
    ## Sum and convert to GW
    cap_trans = (
        branch.loc[branch.zoneFrom != branch.zoneTo]
        .groupby('interface')['LTE Rating'].sum() / 1e3
    )

    ##############################
    #%%### Build the system ######

    ### Load input settings
    defaults = zephyr.system.Defaults(defaultsfile)

    ### Build the system
    zones = ['z1', 'z2', 'z3']
    system = zephyr.system.System('RTS', periods=[0], nodes=zones)
    system.timeindex = timeindex
    system.hours = {0: len(system.timeindex)}

    ### Store the profiles
    system.cfin_pv = cfin_pv
    system.cfin_rtpv = cfin_rtpv
    system.cfin_wind = cfin_wind
    system.cfin_hydro = cfin_hydro
    system.load_in = load_in
    system.defaults = defaults

    #%% Load
    for zone in zones:
        system.add_load(zone, load_in[zone], period=0)

    #%% Existing system
    exist, exist_cap, exist_heatrate = {}, {}, {}
    for i in [
        'coal', 'gas cc', 'gas ct', 'oil ct', 'oil st', 'nuclear',
        'wind', 'storage', 'solar pv', 'csp', 'solar rtpv', 'hydro',
    ]:
        exist[i] = dfgen.loc[dfgen.Category.str.lower() == i].copy()
        exist_cap[i] = exist[i].groupby('zone')['PMax GW'].sum()
        ## Weighted average heat rate
        exist[i]['HR_weighted'] = exist[i]['HR_avg_0'] * exist[i]['PMax GW']
        exist_heatrate[i] = (
            exist[i].groupby('zone')['HR_weighted'].sum() / exist_cap[i]).round(0)

    ### Coal (fixed)
    for zone, cap in exist_cap['coal'].iteritems():
        system.add_generator(
            f'coal_{zone}',
            zephyr.system.Gen('Coal', defaults=defaults,
            heatrate=exist_heatrate['coal'][zone])
            .localize(zone)
            .add_capacity_bound_lo(exist_cap['coal'][zone])
            .add_capacity_bound_hi(exist_cap['coal'][zone])
        )

    ### Nuclear (fixed)
    for zone, cap in exist_cap['nuclear'].iteritems():
        system.add_generator(
            f'nuclear_{zone}',
            zephyr.system.Gen(
                'Nuclear_vogtle', defaults=defaults,
                heatrate=exist_heatrate['nuclear'][zone])
            .localize(zone)
            .add_capacity_bound_lo(exist_cap['nuclear'][zone])
            .add_capacity_bound_hi(exist_cap['nuclear'][zone])
        )

    ### Gas (expandable)
    ## We assume the same heat rate for new additions to reduce model size, but could
    ## instead assume a different characteristic heat rate for new capacity
    for zone, cap in exist_cap['gas cc'].iteritems():
        system.add_generator(
            f'gascc_{zone}',
            zephyr.system.Gen(
                'CCGT_2030_mid', defaults=defaults,
                heatrate=exist_heatrate['gas cc'][zone])
            .localize(zone)
            .add_capacity_bound_lo(exist_cap['gas cc'].get(zone,0))
        )
    for zone, cap in exist_cap['gas ct'].iteritems():
        system.add_generator(
            f'gasct_{zone}',
            zephyr.system.Gen(
                'OCGT_2030_mid', defaults=defaults,
                heatrate=exist_heatrate['gas ct'][zone])
            .localize(zone)
            ## Group oil capacity in with gas-ct to reduce model size
            .add_capacity_bound_lo(
                (exist_cap['gas ct']
                .add(exist_cap['oil st'], fill_value=0)
                .add(exist_cap['oil ct'], fill_value=0)
                ).get(zone,0))
        )

    ### PV
    for zone, pv in zone_pv.items():
        system.add_generator(
            f'pv_{zone}',
            zephyr.system.Gen('PV_track_2030_mid', defaults=defaults)
            .add_availability(cfin_pv[pv], period=0)
            .localize(zone)
            ## Group CSP capacity in with PV to reduce model size
            .add_capacity_bound_lo(
                exist_cap['solar pv'].add(exist_cap['csp'], fill_value=0).get(zone,0))
        )
        system.generators[f'pv_{zone}'].profilename = pv

    ### Wind
    for zone, wind in zone_wind.items():
        system.add_generator(
            f'wind_{zone}',
            zephyr.system.Gen('Wind_2030_mid', defaults=defaults)
            .add_availability(cfin_wind[wind], period=0)
            .localize(zone)
            .add_capacity_bound_lo(exist_cap['wind'].get(zone,0))
        )
        system.generators[f'wind_{zone}'].profilename = wind

    ### Storage
    for zone in zones:
        system.add_storage(
            f'battery_{zone}',
            zephyr.system.Storage('Li_2030_mid', defaults=defaults)
            .localize(zone)
            .add_power_bound_lo(exist_cap['storage'].get(zone,0))
        )

    ### Rooftop PV (fixed)
    for zone, cap in exist_cap['solar rtpv'].items():
        rtpv = zone_rtpv[zone]
        system.add_generator(
            f'rtpv_{zone}',
            zephyr.system.Gen('PV_fixed', defaults=defaults)
            .add_availability(cfin_rtpv[rtpv], period=0)
            .localize(zone)
            .add_capacity_bound_lo(exist_cap['solar rtpv'].get(zone,0))
            .add_capacity_bound_hi(exist_cap['solar rtpv'].get(zone,0))
        )
        system.generators[f'rtpv_{zone}'].profilename = rtpv

    ### Hydro ROR (fixed)
    for zone, cap in exist_cap['hydro'].items():
        hydro = zone_hydro[zone]
        system.add_generator(
            f'hydro_{zone}',
            zephyr.system.Gen('Hydro_ROR', defaults=defaults)
            .add_availability(cfin_hydro[hydro], period=0)
            .localize(zone)
            .add_capacity_bound_lo(exist_cap['hydro'].get(zone,0))
            .add_capacity_bound_hi(exist_cap['hydro'].get(zone,0))
        )
        system.generators[f'hydro_{zone}'].profilename = hydro

    ### Existing transmission
    for pair in links:
        zone1, zone2 = pair
        name = f'{zone1}|{zone2}_ac_old'
        line = zephyr.system.Line(
            node1=zone1, node2=zone2, distance=dfdistance.loc[zone1,zone2],
            voltage=345, defaults=defaults, name=name, newbuild=False,
            capacity=cap_trans['||'.join([zone1, zone2])]
        )
        system.add_line(name=name, line=line)

    ### New transmission
    for pair in links:
        zone1, zone2 = pair
        name = f'{zone1}|{zone2}_ac_new'
        line = zephyr.system.Line(
            node1=zone1, node2=zone2, distance=dfdistance.loc[zone1,zone2],
            voltage=345, defaults=defaults, name=name, newbuild=True,
        )
        system.add_line(name=name, line=line)

    return system


#%%### Procedure
if __name__ == '__main__':
    #%% Build the system
    system = create_system(rtspath=rtspath, defaultsfile=defaultsfile)

    #%% Run the CEM and take a look
    (capacity, operation, values, _system) = zephyr.cem.cem(system)
    print('### CEM results ###')
    print(values)

    #%% Create a PCM system based on the CEM results
    system_pcm = copy.deepcopy(system)

    ### Add resulting optimized capacities
    for gen in system_pcm.generators:
        system_pcm.generators[gen].cap = capacity[gen]
    for stor in system_pcm.storages:
        system_pcm.storages[stor].cappower = capacity[stor+'_P']
        system_pcm.storages[stor].capenergy = capacity[stor+'_E']
    for line in system_pcm.lines:
        system_pcm.lines[line].cap = capacity[line]
    ### Allow for dropped load
    for zone in system_pcm.nodes:
        system_pcm.add_generator(
            f'Lostload_{zone}',
            (zephyr.system.Gen('Lostload', defaults=system_pcm.defaults, cost_vom=20)
             .localize(zone))
        )
        system_pcm.generators[f'Lostload_{zone}'].cap = (
            system_pcm.load_in[zone].max() * 2)

    #%% Run the PCM
    (operation_pcm, values_pcm, prices_pcm) = zephyr.pcm.pcm(system_pcm)
    print('### PCM results ###')
    print(values_pcm)
