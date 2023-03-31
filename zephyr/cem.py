#%%### Imports
import pandas as pd
import numpy as np
import os, sys, time, pickle, gzip

import pulp

#%%### Functions

def cem(system, return_model=False, solver=None, hours=None,
        verbose=True, savename=None, includedual=False, includereserves=False,
        **solverkwargs):
    """
    """
    ###### Input formatting
    if hours is None:
        hours = system.hours
    if type(hours) != dict:
        raise Exception('Need to supply hours as dict with period keys')

    ### Get solver if necessary
    if solver == 'gurobi':
        # solver = pulp.GUROBI_CMD(msg=0)
        # solver = pulp.GUROBI_CMD(msg=(1 if verbose in [2,3] else 0))
        solver = pulp.GUROBI_CMD(msg=(1 if verbose in [2,3] else 0),
                                 options=solverkwargs.items())
    elif solver in ['cbc', 'default', 'coin', 'clp', None]:
        solver = pulp.PULP_CBC_CMD()
    elif solver in ['gurobipy']:
        solver = pulp.GUROBI()

    ###### Get parameters from setup
    gens = system.generators
    stors = system.storages
    reshydros = system.reshydros
    loads = system.loads
    lines = system.lines
    newlines = {l: system.lines[l] for l in system.lines if l.endswith('new')}
    periods = system.periods
    years = system.years
    load_cumulative = sum(
        [loads[period][load].sum() for period in periods for load in loads[period]])

    ###### Instantiate problem class
    tic = time.perf_counter()
    model = pulp.LpProblem("CapacityExpansionModel", pulp.LpMinimize)

    ###### Construct time-resolved variables
    ### Generator powers
    genpower = pulp.LpVariable.dicts(
        'genpower',
        ((gen, period, hour)
         for gen in gens for period in periods for hour in range(hours[period]+1)),
        lowBound=0, cat='Continuous')

    ### Storage charge
    storcharge = pulp.LpVariable.dicts(
        'storcharge',
        ((stor, period, hour)
         for stor in stors for period in periods for hour in range(hours[period]+1)),
        lowBound=0, cat='Continuous')

    ### Storage discharge
    stordischarge = pulp.LpVariable.dicts(
        'stordischarge',
        ((stor, period, hour)
         for stor in stors for period in periods for hour in range(hours[period]+1)),
        lowBound=0, cat='Continuous')

    ### Storage energy
    storenergy = pulp.LpVariable.dicts(
        'storenergy',
        ((stor, period, hour)
         for stor in stors for period in periods for hour in range(hours[period]+1)),
        lowBound=0, cat='Continuous')

    ### Reservoir hydropower, if utilized
    if len(reshydros) > 0:
        ### Power
        respower = pulp.LpVariable.dicts(
            'respower',
            ((res, period, hour)
             for res in reshydros for period in periods for hour in range(hours[period]+1)),
            lowBound=0, cat='Continuous')
        ### Spill
        spillpower = pulp.LpVariable.dicts(
            'spillpower',
            ((res, period, hour)
             for res in reshydros for period in periods for hour in range(hours[period]+1)),
            lowBound=0, cat='Continuous')

    ### Transmission, if utilized
    if len(lines) > 0:
        ### Forward flow
        flowpos = pulp.LpVariable.dicts(
            'flowpos',
            ((line, period, hour)
             for line in lines for period in periods for hour in range(hours[period]+1)),
            lowBound=0, cat='Continuous',
        )
        ### Reverse flow
        flowneg = pulp.LpVariable.dicts(
            'flowneg',
            ((line, period, hour)
             for line in lines for period in periods for hour in range(hours[period]+1)),
            lowBound=0, cat='Continuous',
        )

    ### Reserves, if utilized
    if system.has_reserves:
        ### Generators
        genreserves = pulp.LpVariable.dicts(
            'genreserves',
            ((gen, period, hour)
             for gen in gens for period in periods for hour in range(hours[period]+1)),
            lowBound=0, cat='Continuous')
        ### Storage
        storreserves = pulp.LpVariable.dicts(
            'storreserves',
            ((stor, period, hour)
             for stor in stors for period in periods for hour in range(hours[period]+1)),
            lowBound=0, cat='Continuous')
        ### Reservoir hydro
        if len(reshydros) > 0:
            resreserves = pulp.LpVariable.dicts(
                'resreserves',
                ((res, period, hour)
                 for res in reshydros for period in periods for hour in range(hours[period]+1)),
                lowBound=0, cat='Continuous')
        ### Transmission, if utilized
        if len(lines) > 0:
            ### Forward flow
            flowposreserves = pulp.LpVariable.dicts(
                'flowposreserves',
                ((line, period, hour)
                 for line in lines for period in periods for hour in range(hours[period]+1)),
                lowBound=0, cat='Continuous',
            )
            ### Reverse flow
            flownegreserves = pulp.LpVariable.dicts(
                'flownegreserves',
                ((line, period, hour)
                 for line in lines for period in periods for hour in range(hours[period]+1)),
                lowBound=0, cat='Continuous',
            )


    ###### Construct static variables
    ### Generator capacity
    gencap = pulp.LpVariable.dicts(
        'gencap',
        (gen for gen in gens),
        lowBound=0, cat='Continuous')

    ### Storage power capacity
    storcap_P = pulp.LpVariable.dicts(
        'storcap_P',
        (stor for stor in stors),
        lowBound=0, cat='Continuous')

    ### Storage energy capacity
    storcap_E = pulp.LpVariable.dicts(
        'storcap_E',
        (stor for stor in stors),
        lowBound=0, cat='Continuous')

    ### Hyro power capacity
    ### (If this comes out less than upper bound, it means some has been retired?)
    ### (Confusing. Better just to leave it out and include as a fixed cost, then
    ### compare afterward to see if cost is ever lower if hydro is left out)

    ###### Transmissions capacity
    newlinecap = pulp.LpVariable.dicts(
        'newlinecap',
        (line for line in newlines),
        lowBound=0, cat='Continuous')


    ###### Objective Function
    model += pulp.lpSum(
        ##### Generators
        ### Capacity and FOM cost
        [gencap[gen] * (gens[gen].cost_annual + gens[gen].cost_fom) * years
         for gen in gens]
        ### VOM and fuel cost
        + [genpower[gen,period,hour] * (
            gens[gen].cost_vom + gens[gen].cost_fuel + gens[gen].cost_emissions)
           for gen in gens for period in periods for hour in range(hours[period])]
        ##### Storages
        ### Power capacity and FOM cost
        + [storcap_P[stor] * (stors[stor].cost_annual_P + stors[stor].cost_fom_P) * years
           for stor in stors]
        ### Energy capacity
        + [storcap_E[stor] * (stors[stor].cost_annual_E + stors[stor].cost_fom_E) * years
           for stor in stors]
        ### VOM cost (only applied to discharge)
        + [stordischarge[stor,period,hour] * stors[stor].cost_vom
           for stor in stors for period in periods for hour in range(hours[period])]
        ##### Reservoir hydros (NOTE: Take power capacity as fixed)
        ### Capacity and FOM cost
        + [reshydros[res].capacity_bound_hi * (
            reshydros[res].cost_annual + reshydros[res].cost_fom) * years
           for res in reshydros]
        ### VOM cost
        + [respower[res,period,hour] * reshydros[res].cost_vom
           for res in reshydros for period in periods for hour in range(hours[period])]
        ##### Transmission capacity; new build
        + [newlinecap[line] * (lines[line].cost_annual + lines[line].cost_fom) * years
           for line in newlines]
        #### Transmission VOM
        + [(flowpos[line,period,hour] * lines[line].cost_vom
            + flowneg[line,period,hour] * lines[line].cost_vom)
           for line in lines for period in periods for hour in range(hours[period])]
    )

    ###### Constraints
    for period in periods:
        ### Make the hour2day converter
        hour2day = dict(zip(range(hours[period]),[i//24 for i in range(hours[period])]))
        ##### Generators
        for gen in gens:
            #### Availability
            if gens[gen].always_available:
                for hour in range(hours[period]):
                    model += genpower[gen,period,hour] <= gencap[gen]
            else:
                for hour in range(hours[period]):
                    model += genpower[gen,period,hour] <= (
                        gencap[gen] * gens[gen].availability[period][hour])
            #### Ramp rate
            if gens[gen].perfectly_rampable:
                pass
            else:
                ### Ramp up
                for hour in range(1, hours[period]+1):
                    model += (genpower[gen,period,hour] - genpower[gen,period,hour-1]
                              <= gencap[gen] * gens[gen].ramp_up)
                ### Ramp down
                for hour in range(1, hours[period]+1):
                    model += (genpower[gen,period,hour] - genpower[gen,period,hour-1]
                              >= -gencap[gen] * gens[gen].ramp_down)
            ### Year-end wrap
            model += genpower[gen,period,0] == genpower[gen,period,hours[period]]
            #### Minimum-generation level
            #### Multiply by availability.
            #### That makes it work for ROR (which is must-take but with variable availability)
            #### and could also make it work for collections of generators with scheduled
            #### outages - such as if one nuclear plant is turned off for a month
            for hour in range(hours[period]):
                if gens[gen].always_available:
                    model += genpower[gen,period,hour] >= gencap[gen] * gens[gen].gen_min
                else:
                    model += genpower[gen,period,hour] >= (
                        gencap[gen] * gens[gen].availability[period][hour] * gens[gen].gen_min)
            #### Capacity limit
            if gens[gen].has_capacity_maximum:
                model += gencap[gen] <= gens[gen].capacity_bound_hi
            #### Minimum capacity
            if gens[gen].has_capacity_minimum:
                model += gencap[gen] >= gens[gen].capacity_bound_lo

        ##### Storage
        for stor in stors:
            for hour in range(hours[period]):
                ### Charge limit
                model += storcharge[stor,period,hour] <= storcap_P[stor]
                ### Discharge limit
                model += stordischarge[stor,period,hour] <= storcap_P[stor]
                ### Simultaneous charge/discharge limit
                if system.cycling_reserves == False:
                    model += storcharge[stor,period,hour] + stordischarge[stor,period,hour] <= (
                        storcap_P[stor])
                else:
                    model += (
                        storcharge[stor,period,hour] + stordischarge[stor,period,hour]
                        + storreserves[stor,period,hour] * system.cycling_reserves_power
                    ) <= storcap_P[stor]
            #### Continuity
            ### Default: Storage reserves don't cycle or incur efficiency losses
            if (system.cycling_reserves == False) or (system.has_reserves == False):
                for hour in range(1, hours[period]+1):
                    model += (
                        (storenergy[stor,period,hour-1]
                            - stordischarge[stor,period,hour-1] * (1/stors[stor].efficiency_discharge)
                            + storcharge[stor,period,hour-1] * stors[stor].efficiency_charge)
                        * (1 - stors[stor].leakage_rate)
                    ) == storenergy[stor,period,hour]
            ### Special: Storage reserves cycle and incur efficiency losses
            elif (system.cycling_reserves == True) and (system.has_reserves == True):
                for hour in range(1, hours[period]+1):
                    model += (
                        (storenergy[stor,period,hour-1]
                            - stordischarge[stor,period,hour-1] * (1/stors[stor].efficiency_discharge)
                            + storcharge[stor,period,hour-1] * stors[stor].efficiency_charge
                            - storreserves[stor,period,hour-1] * (
                                1 - stors[stor].efficiency_discharge * stors[stor].efficiency_charge
                            ) * 0.5 * system.cycling_reserves_power
                        ) * (1 - stors[stor].leakage_rate)
                    ) == storenergy[stor,period,hour]

            ### Storage energy capacity
            if stors[stor].constrained_energy:
                for hour in range(hours[period]+1):
                    model += storenergy[stor,period,hour] <= storcap_E[stor]
            ### Year-end wrap
            model += storenergy[stor,period,0] == storenergy[stor,period,hours[period]]
            model += storcharge[stor,period,0] == storcharge[stor,period,hours[period]]
            model += stordischarge[stor,period,0] == stordischarge[stor,period,hours[period]]
            ### Capacity limits (up and down)
            if stors[stor].has_power_maximum:
                model += storcap_P[stor] <= stors[stor].power_bound_hi
            if stors[stor].has_power_minimum:
                model += storcap_P[stor] >= stors[stor].power_bound_lo
            if stors[stor].has_energy_maximum:
                model += storcap_E[stor] <= stors[stor].energy_bound_hi
            if stors[stor].has_energy_minimum:
                model += storcap_E[stor] >= stors[stor].energy_bound_lo
            ### Duration constraints
            if stors[stor].has_duration_maximum:
                model += storcap_E[stor] <= stors[stor].duration_max * storcap_P[stor]
            if stors[stor].has_duration_minimum:
                model += storcap_E[stor] >= stors[stor].duration_min * storcap_P[stor]


        ##### Meet load
        for node in loads[period]: ### Load is labeled by node name
            for hour in range(hours[period]):
                model += (
                    ### Generators
                    pulp.lpSum([genpower[gen,period,hour] for gen in gens
                                if gens[gen].node == node])
                    ### Storage
                    + pulp.lpSum([stordischarge[stor,period,hour] - storcharge[stor,period,hour]
                                  for stor in stors
                                  if stors[stor].node == node])
                    ### Hydro
                    + pulp.lpSum([respower[res,period,hour] for res in reshydros
                                  if reshydros[res].node == node])
                    ##### Transmission
                    ### Forward flow in lines starting at node; no losses (save for end)
                    - pulp.lpSum([flowpos[line,period,hour]
                                  for line in lines
                                  if lines[line].node1 == node])
                    ### Reverse flow in lines starting at node; subject to losses
                    + pulp.lpSum([flowneg[line,period,hour] * (1 - lines[line].loss)
                                  for line in lines
                                  if lines[line].node1 == node])
                    ### Forward flow in lines ending at node; subject to losses
                    + pulp.lpSum([flowpos[line,period,hour] * (1 - lines[line].loss)
                                  for line in lines
                                  if lines[line].node2 == node])
                    ### Reverse flow in lines ending at node; no losses (save for end)
                    - pulp.lpSum([flowneg[line,period,hour]
                                  for line in lines
                                  if lines[line].node2 == node])
                ) == loads[period][node][hour], 'price_p{}_l{}_h{}'.format(period,node,hour)

        ###### Transmission constraints
        for line in lines:
            # if line.split('_')[-1] == 'old':
            if lines[line].newbuild == False:
                for hour in range(hours[period]):
                    ##### Line capacity - measured at dispatched end
                    ### Expand the line capacity by the losses, so that capacity is measured at output
                    ### IMPORTANT: Originally messed this up and had * instead of /
                    model += flowpos[line,period,hour] * (1 - lines[line].loss) <= lines[line].capacity
                    model += flowneg[line,period,hour] * (1 - lines[line].loss) <= lines[line].capacity
                    ##### Simultaneous flow limit
                    model += (
                        flowpos[line,period,hour] + flowneg[line,period,hour]
                    ) * (1 - lines[line].loss) <= lines[line].capacity

            # if line.split('_')[-1] == 'new':
            if lines[line].newbuild == True:
                for hour in range(hours[period]):
                    ##### Line capacity - measured at dispatched end
                    ### Expand the line capacity by the losses, so that capacity is measured at output
                    model += flowpos[line,period,hour] * (1 - lines[line].loss) <= newlinecap[line]
                    model += flowneg[line,period,hour] * (1 - lines[line].loss) <= newlinecap[line]
                    ##### Simultaneous flow limit
                    model += (
                        flowpos[line,period,hour] + flowneg[line,period,hour]
                    ) * (1 - lines[line].loss) <= newlinecap[line]

            ### Year-end wrap
            model += flowpos[line,period,0] == flowpos[line,period,hours[period]]
            model += flowneg[line,period,0] == flowneg[line,period,hours[period]]

        ###### Hydro constraints
        daystarts = list(range(hours[period]))[::24]
        ##### Reservoir
        for res in reshydros:
            #### Availability
            for day, daystart in enumerate(daystarts):
                model += (
                    pulp.lpSum([(respower[res,period,hour] + spillpower[res,period,hour])
                                for hour in range(daystart, daystart+24)]
                    ) == reshydros[res].availability[period][day]
                )
            #### Minimum- and maximum-generation levels
            for hour in range(hours[period]):
                ### Min
                model += (
                    ### ORIGINAL: both power and spill can meet min
                    # (respower[res,period,hour] + spillpower[res,period,hour]
                    ### NEW: only power can meet min (makes more sense, especially since
                    ### we already allow for the low-availabilty times)
                    (respower[res,period,hour]
                    ### The min() is required to deal with cases when the historical
                    ### generation is less than Pmin*cap
                    ) >= min(reshydros[res].capacity_bound_hi * reshydros[res].gen_min,
                             reshydros[res].availability[period][hour2day[hour]]/24)
                )
                ### Max flow restricted to nameplate power
                model += (respower[res,period,hour] <= reshydros[res].capacity_bound_hi)

            #### Ramp rate
            ### For hydro, we only apply the ramprate to power, not spill
            if reshydros[res].perfectly_rampable:
                pass
            else:
                ### Ramp up
                for hour in range(1, hours[period]+1):
                    model += (respower[res,period,hour] - respower[res,period,hour-1]
                              <= reshydros[res].capacity_bound_hi * reshydros[res].ramp_up)
                ### Ramp down
                for hour in range(1, hours[period]+1):
                    model += (respower[res,period,hour] - respower[res,period,hour-1]
                              >= -reshydros[res].capacity_bound_hi * reshydros[res].ramp_down)
            #### Year-end wrap
            model += respower[res,period,0] == respower[res,period,hours[period]]

    ##### CO2 cap
    if system.has_co2cap:
        model += (
            pulp.lpSum([genpower[gen,period,hour] * gens[gen].emissionsrate
                        for gen in gens for period in periods for hour in range(hours[period])])
        ) <= load_cumulative * system.co2cap, 'co2cap' ### GWh * tons/GWh == tons


    ##### RPS
    if system.has_rps:
        ### VRE gen >= RPS formulation: worried that storage could be used to soak up
        ### extra renewables generation without meeting load
        # model += (
        #     pulp.lpSum([genpower[gen,hour] for gen in gens for hour in range(hours)
        #                 if gens[gen].meets_rps is True])
        # ) >= load_cumulative * rps, 'rps' ### GWh * fraction == GWh
        ### non-VRE gen <= RPS formulation: should meet RPS without incentivizing
        ### waste of VRE energy through storage charge/discharge
        model += (
            pulp.lpSum([genpower[gen,period,hour]
                        for gen in gens for period in periods for hour in range(hours[period])
                        if gens[gen].meets_rps is False])
        ) <= load_cumulative * (1 - system.rps), 'rps' ### GWh * fraction == GWh

    ##### Reserves
    if system.has_reserves:
        for period in periods:
            ### Make the timeseries of reserves that must be provided
            # reservemargin = system.reservefraction * sum(
            #     [loads[period][load] for load in loads[period]])
            reservemargin = {
                load: system.reservefraction * loads[period][load] for load in loads[period]
            }
            ##### Enforce availability of reserves in all hours
            for node in loads[period]: ### Load is labeled by node name
                for hour in range(hours[period]):
                    model += (
                        ### Generators
                        pulp.lpSum(
                            [genreserves[gen,period,hour] for gen in gens
                             if gen.lower() not in ['lostload','blackout','voll','demandresponse']
                             and gens[gen].node == node])
                        ### Storage
                        + pulp.lpSum(
                            [storreserves[stor,period,hour] for stor in stors
                             if stors[stor].node == node])
                        ### Hydro
                        + pulp.lpSum(
                            [resreserves[res,period,hour] for res in reshydros
                             if reshydros[res].node == node])
                        ##### Transmission
                        ### Forward flow in lines starting at node; no losses (save for end)
                        - pulp.lpSum(
                            [flowposreserves[line,period,hour]
                             for line in lines if lines[line].node1 == node])
                        ### Reverse flow in lines starting at node; subject to losses
                        + pulp.lpSum(
                            [flownegreserves[line,period,hour] * (1 - lines[line].loss)
                             for line in lines if lines[line].node1 == node])
                        ### Forward flow in lines ending at node; subject to losses
                        + pulp.lpSum(
                            [flowposreserves[line,period,hour] * (1 - lines[line].loss)
                             for line in lines if lines[line].node2 == node])
                        ### Reverse flow in lines ending at node; no losses (save for end)
                        - pulp.lpSum(
                            [flownegreserves[line,period,hour]
                             for line in lines if lines[line].node2 == node])
                    ) >= reservemargin[node][hour], 'reserves_p{}_l{}_h{}'.format(period,node,hour)

            ##### Add the reserve constraints
            for hour in range(hours[period]):
                ##### Constrain reserve variables for generators
                for gen in gens:
                    ### Make sure we don't include lost load
                    if gen.lower() in ['lostload','blackout','voll','demandresponse']:
                        continue
                    ### Non-intermittent resources
                    if gens[gen].always_available:
                        model += genreserves[gen,period,hour] <= (
                            gencap[gen] - genpower[gen,period,hour])
                    ### Intermittent resources
                    else:
                        model += genreserves[gen,period,hour] <= (
                            gencap[gen] * gens[gen].availability[period][hour]
                            - genpower[gen,period,hour])
                    ### Ramp up rate, if necessary
                    if not gens[gen].perfectly_rampable:
                        model += genreserves[gen,period,hour] <= (
                            gencap[gen] * gens[gen].ramp_up)

                ##### Constrain reserve variables for storages
                for stor in stors:
                    ### Storage can provide reserves by increasing hourly discharge
                    ### or by descreasing hourly charge
                    model += storreserves[stor,period,hour] <= (
                        storcap_P[stor]
                        - stordischarge[stor,period,hour] + storcharge[stor,period,hour])
                    ### Energy constraint
                    ### If storage is discharging/charging in a given hour,
                    ### that affects the amount of energy available for providing reserves.
                    ### Essentially should treat reserves like discharge - both get a 1/eta.
                    model += storreserves[stor,period,hour] <= stors[stor].efficiency_discharge * (
                        storenergy[stor,period,hour]
                        - stordischarge[stor,period,hour] * (1/stors[stor].efficiency_discharge)
                        + storcharge[stor,period,hour] * stors[stor].efficiency_charge)

                ##### Constrain reserve variables for hydro
                for res in reshydros:
                    model += resreserves[res,period,hour] <= spillpower[res,period,hour]
                    ### Ramp up rate, if necessary (remember it only applies to power)
                    if not reshydros[res].perfectly_rampable:
                        model += resreserves[res,period,hour] <= (
                            reshydros[res].capacity_bound_hi * reshydros[res].ramp_up)

                ##### Constrain reserve variables for transmission
                for line in lines:
                    if lines[line].newbuild == False:
                        model += (
                            flowposreserves[line,period,hour] + flowpos[line,period,hour]
                        ) * (1 - lines[line].loss) <= lines[line].capacity
                        model += (
                            flownegreserves[line,period,hour] + flowneg[line,period,hour]
                        ) * (1 - lines[line].loss) <= lines[line].capacity
                        ### Simultaneous flow limit
                        model += (
                            flowpos[line,period,hour] + flowneg[line,period,hour]
                            + flowposreserves[line,period,hour] + flownegreserves[line,period,hour]
                        ) * (1 - lines[line].loss) <= lines[line].capacity

                    if lines[line].newbuild == True:
                        model += (
                            flowposreserves[line,period,hour] + flowpos[line,period,hour]
                        ) * (1 - lines[line].loss) <= newlinecap[line]
                        model += (
                            flownegreserves[line,period,hour] + flowneg[line,period,hour]
                        ) * (1 - lines[line].loss) <= newlinecap[line]
                        ### Simultaneous flow limit
                        model += (
                            flowpos[line,period,hour] + flowneg[line,period,hour]
                            + flowposreserves[line,period,hour] + flownegreserves[line,period,hour]
                        ) * (1 - lines[line].loss) <= newlinecap[line]

    ### Finished building model; record time
    toc = time.perf_counter()

    ###### Return model if desired
    if return_model:
        return model

    ###### Report a few things
    time_setup = toc-tic
    if verbose == True:
        print(savename)
        # print([i for i in loads[period].keys() for period in periods])
        print('{} variables'.format(len(model.variables())))
        print('{} constraints'.format(len(model.constraints)))
        print('{:.2f} seconds to set up'.format(time_setup))
        sys.stdout.flush()

    ##################
    ###### Solve model
    tic = time.perf_counter()
    model.solve(solver=solver)
    toc = time.perf_counter()
    time_solve = toc-tic
    if verbose == True:
        print('model status: {}'.format(model.status))
        print('{:.2f} seconds to solve'.format(time_solve))
        sys.stdout.flush()

    ### Return nothing if no solution
    if model.status == 0:
        return None

    ########## Extract output
    ###### Static variables
    output_capacity = {
        **{gen:gencap[gen].varValue for gen in gens},
        **{stor+'_P':storcap_P[stor].varValue for stor in stors},
        **{stor+'_E':storcap_E[stor].varValue for stor in stors},
        ### Reservoir hydros - fixed inputs for now
        **{res:reshydros[res].capacity_bound_hi for res in reshydros},
        ### Transmission lines; existing
        **{line:lines[line].capacity for line in lines if line.endswith('old')},
        ### Transmission lines; new-build
        **{line:newlinecap[line].varValue for line in newlines},
    }

    ###### Hourly variables
    output_operation = {
        period: {
            **{gen: [genpower[gen,period,hour].varValue
                     for hour in range(hours[period]+1)] for gen in gens},
            **{stor+'_charge': [-storcharge[stor,period,hour].varValue
                                for hour in range(hours[period]+1)] for stor in stors},
            **{stor+'_discharge': [stordischarge[stor,period,hour].varValue
                                   for hour in range(hours[period]+1)] for stor in stors},
            **{stor+'_energy': [storenergy[stor,period,hour].varValue
                                for hour in range(hours[period]+1)] for stor in stors},
            **{res+'_power': [respower[res,period,hour].varValue
                              for hour in range(hours[period]+1)] for res in reshydros},
            **{res+'_spill': [spillpower[res,period,hour].varValue
                              for hour in range(hours[period]+1)] for res in reshydros},
            **{'Load_'+load: [loads[period][load][hour]
                      for hour in range(hours[period])]+[np.nan] for load in loads[period]},
            **{line+'_pos': [flowpos[line,period,hour].varValue
                             for hour in range(hours[period]+1)] for line in lines},
            **{line+'_neg': [-flowneg[line,period,hour].varValue
                             for hour in range(hours[period]+1)] for line in lines},
        } for period in periods}
    ### Reserves
    if system.has_reserves:
        output_reserves = {
            period: {
                **{gen: [genreserves[gen,period,hour].varValue
                         for hour in range(hours[period]+1)] for gen in gens},
                **{stor: [storreserves[stor,period,hour].varValue
                          for hour in range(hours[period]+1)] for stor in stors},
                **{res: [resreserves[res,period,hour].varValue
                         for hour in range(hours[period]+1)] for res in reshydros},
            } for period in periods}

    ### Objective function components
    cost_fixed = sum(
        ##### Generators
        ### Capacity and FOM cost
        [gencap[gen].varValue * (gens[gen].cost_annual + gens[gen].cost_fom) * years
         for gen in gens]
        ##### Storages
        ### Power capacity and FOM cost
        + [storcap_P[stor].varValue
           * (stors[stor].cost_annual_P + stors[stor].cost_fom_P) * years
           for stor in stors]
        ### Energy capacity
        + [storcap_E[stor].varValue
           * (stors[stor].cost_annual_E + stors[stor].cost_fom_E) * years
           for stor in stors]
        ##### Reservoir hydros (NOTE: Take power capacity as fixed)
        ### Capacity and FOM cost
        + [reshydros[res].capacity_bound_hi
           * (reshydros[res].cost_annual + reshydros[res].cost_fom) * years
           for res in reshydros]
        ##### Transmission capacity; new build
        + [newlinecap[line].varValue
           * (lines[line].cost_annual + lines[line].cost_fom) * years
           for line in newlines]
    )
    cost_variable = sum(
        ### Generator VOM and fuel cost
        [genpower[gen,period,hour].varValue
         * (gens[gen].cost_vom + gens[gen].cost_fuel + gens[gen].cost_emissions)
         for gen in gens for period in periods for hour in range(hours[period])]
        ### Storage VOM cost (only applied to discharge)
        + [stordischarge[stor,period,hour].varValue * stors[stor].cost_vom
           for stor in stors for period in periods for hour in range(hours[period])]
        ### Hydro VOM cost
        + [respower[res,period,hour].varValue * reshydros[res].cost_vom
           for res in reshydros for period in periods for hour in range(hours[period])]
        #### Transmission VOM
        + [flowpos[line,period,hour].varValue * lines[line].cost_vom
           + flowneg[line,period,hour].varValue * lines[line].cost_vom
           for line in lines for period in periods for hour in range(hours[period])]
    )

    ###### Other values
    output_values = {
        'objective': pulp.value(model.objective),
        'objective_fixed': cost_fixed,
        'objective_variable': cost_variable,
        'co2shadowprice': (-model.constraints['co2cap'].__dict__['pi']*1E6
                           if system.has_co2cap else 0),
        ### LCOE in $/MWh
        'lcoe': pulp.value(model.objective) * 1000 / load_cumulative,
        'co2rate':  sum(
            [(np.array(output_operation[period][gen]) * gens[gen].emissionsrate).sum()
             for period in periods for gen in gens]
        ) / load_cumulative, ### t/GWh == kg/MWh == g/kWh
        'loadsum': load_cumulative,
        'loadpeak': max(
            [sum([loads[period][load] for load in loads[period]]).max() for period in periods]),
        'time_setup': time_setup,
        'time_solve': time_solve,
        'variables': len(model.variables()),
        'constraints': len(model.constraints),
    }

    ### Dual variables
    if includedual:
        dfdual = pd.DataFrame(
            index=model.constraints.keys(),
            data={
                'pi': [model.modifiedConstraints[i].pi
                       for i in range(len(model.modifiedConstraints))],
                'slack': [model.modifiedConstraints[i].slack
                          for i in range(len(model.modifiedConstraints))],
            }
        )

    ###### Format output
    if includedual is False:
        out = (output_capacity, output_operation, output_values, system)
    elif includedual is True:
        out = (output_capacity, output_operation, output_values, system, dfdual)
    ### Include reserves
    if system.has_reserves and (includereserves is True):
        if includedual is False:
            out = (output_capacity, output_operation, output_values, output_reserves, system)
        elif includedual is True:
            out = (output_capacity, output_operation, output_values, output_reserves, system, dfdual)

    ###### Save outputs if desired
    if savename is None:
        pass
    else:
        if savename.endswith('.p.gz'):
            with gzip.open(savename, 'wb') as p:
                pickle.dump(out, p)
        elif savename.endswith('.p'):
            with open(savename, 'wb') as p:
                pickle.dump(out, p)
        elif savename.endswith('.xlsx') or savename.endswith('.xls'):
            ### Output dataframes
            with pd.ExcelWriter(savename, options={'remove_timezone':True}) as writer:
                pd.DataFrame(output_capacity, index=['']).to_excel(writer, sheet_name='capacity')
                pd.DataFrame(output_values, index=['']).to_excel(writer, sheet_name='values')
                if hasattr(system, 'index'):
                    pd.DataFrame(output_operation, index=system.index).tz_convert('UTC').to_excel(
                        writer, sheet_name='operation')
                else:
                    pd.DataFrame(output_operation).to_excel(writer, sheet_name='operation')
                if includedual is True:
                    dfdual.to_excel(writer, sheet_name='dual')
            ### Still save system as object
            with gzip.open(savename
                           .replace('.xlsx','_system.p.gz')
                           .replace('.xls','_system .p.gz'), 'wb') as p:
                pickle.dump(system, p)
            # if includedual is True:
                # with gzip.open(savename
                #                .replace('.xlsx','_model.p.gz')
                #                .replace('.xls','_model.p.gz'), 'wb') as p:
                #     pickle.dump(model, p)
        else:
            with gzip.open(savename+'.p.gz', 'wb') as p:
                pickle.dump(out, p)

    return out
