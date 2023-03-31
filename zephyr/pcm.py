#%%### Imports
import pandas as pd
import numpy as np
import os, sys, time

import pulp

#%%### Functions
#%% Full PCM
def pcm(system, solver=None, hours=None, verbose=2, includedual=True, **solverkwargs):
    """
    First need to add 'Lostload' generator to each zone
    """
    ###### Process inputs
    if hours is None:
        hours = system.hours

    ### Get solver if necessary
    if solver in ["gurobi", "gurobi_cl", "gurobi_cmd"]:
        solver = pulp.GUROBI_CMD(
            msg=(1 if verbose in [2, 3] else 0), options=solverkwargs.items()
        )
    elif solver in ["cbc", "default", "coin", "clp", None]:
        solver = pulp.PULP_CBC_CMD(msg=0)
    elif solver in ["gurobipy"]:
        solver = pulp.GUROBI()

    ###### Get parameters from setup
    gens = system.generators
    stors = system.storages
    reshydros = system.reshydros
    loads = system.loads
    lines = system.lines
    periods = system.periods
    years = system.years
    load_cumulative = sum(
        [loads[period][load].sum() for period in periods for load in loads[period]])

    ###### Instantiate problem class
    tic = time.perf_counter()
    model = pulp.LpProblem("ProductionCostModel", pulp.LpMinimize)

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

    ###### Objective Function: minimize system cost
    model += pulp.lpSum(
        ##### Generators
        ### VOM and fuel cost
        [genpower[gen,period,hour] * (
            gens[gen].cost_vom + gens[gen].cost_fuel + gens[gen].cost_emissions)
           for gen in gens for period in periods for hour in range(hours[period])]
        ##### Storages
        ### VOM cost (only applied to discharge)
        + [stordischarge[stor,period,hour] * stors[stor].cost_vom
           for stor in stors for period in periods for hour in range(hours[period])]
        ##### Reservoir hydros
        ### VOM cost
        + [respower[res,period,hour] * reshydros[res].cost_vom
           for res in reshydros for period in periods for hour in range(hours[period])]
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
                    model += genpower[gen,period,hour] <= gens[gen].cap
            else:
                for hour in range(hours[period]):
                    model += genpower[gen,period,hour] <= (
                        gens[gen].cap * gens[gen].availability[period][hour])
            #### Ramp rate
            if gens[gen].perfectly_rampable:
                pass
            else:
                ### Ramp up
                for hour in range(1, hours[period]+1):
                    model += (genpower[gen,period,hour] - genpower[gen,period,hour-1]
                              <= gens[gen].cap * gens[gen].ramp_up)
                ### Ramp down
                for hour in range(1, hours[period]+1):
                    model += (genpower[gen,period,hour] - genpower[gen,period,hour-1]
                              >= -gens[gen].cap * gens[gen].ramp_down)
            ### Year-end wrap
            model += genpower[gen,period,0] == genpower[gen,period,hours[period]]
            #### Minimum-generation level
            #### Multiply by availability.
            #### That makes it work for ROR (which is must-take but with variable availability)
            #### and could also make it work for collections of generators with scheduled
            #### outages - such as if one nuclear plant is turned off for a month
            for hour in range(hours[period]):
                if gens[gen].always_available:
                    model += genpower[gen,period,hour] >= gens[gen].cap * gens[gen].gen_min
                else:
                    model += genpower[gen,period,hour] >= (
                        gens[gen].cap * gens[gen].availability[period][hour] * gens[gen].gen_min)

        ##### Storage
        for stor in stors:
            for hour in range(hours[period]):
                ### Charge limit
                model += storcharge[stor,period,hour] <= stors[stor].cappower
                ### Discharge limit
                model += stordischarge[stor,period,hour] <= stors[stor].cappower
                ### Simultaneous charge/discharge limit
                if system.cycling_reserves == False:
                    model += storcharge[stor,period,hour] + stordischarge[stor,period,hour] <= (
                        stors[stor].cappower)
                else:
                    model += (
                        storcharge[stor,period,hour] + stordischarge[stor,period,hour]
                        # + storreserves[stor,period,hour] * system.cycling_reserves_power
                    ) <= stors[stor].cappower
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
                            # - storreserves[stor,period,hour-1] * (
                            #     1 - stors[stor].efficiency_discharge * stors[stor].efficiency_charge
                            # ) * 0.5 * system.cycling_reserves_power
                        ) * (1 - stors[stor].leakage_rate)
                    ) == storenergy[stor,period,hour]

            ### Storage energy capacity
            if stors[stor].constrained_energy:
                for hour in range(hours[period]+1):
                    model += storenergy[stor,period,hour] <= stors[stor].capenergy
            ### Year-end wrap
            model += storenergy[stor,period,0] == storenergy[stor,period,hours[period]]
            model += storcharge[stor,period,0] == storcharge[stor,period,hours[period]]
            model += stordischarge[stor,period,0] == stordischarge[stor,period,hours[period]]


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

    toc = time.perf_counter()

    ###### Report a few things
    time_setup = toc - tic
    if verbose >= 2:
        print("{} variables".format(len(model.variables())))
        print("{} constraints".format(len(model.constraints)))
        print("{:.2f} seconds to set up".format(time_setup))
        sys.stdout.flush()


    ##################
    ###### Solve model
    tic = time.perf_counter()
    model.solve(solver=solver)
    toc = time.perf_counter()
    time_solve = toc - tic
    if verbose >= 2:
        print("model status: {}".format(model.status))
        print("{:.2f} seconds to solve".format(time_solve))
        sys.stdout.flush()

    ### Return nothing if no solution
    if model.status == 0:
        return None

    ########## Extract output
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

    ###### Other values
    output_values = {
        'objective': pulp.value(model.objective),
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
                "pi": [model.modifiedConstraints[i].pi
                       for i in range(len(model.modifiedConstraints))],
                "slack": [model.modifiedConstraints[i].slack
                          for i in range(len(model.modifiedConstraints))],
            },
        )
        return output_operation, output_values, dfdual
    else:
        return output_operation, output_values


#%% Simple VRE+storage dispatch model
def dispatch(system, energyinit, day=0, solver=None, verbose=1, **solverkwargs):
    """
    """
    ###### Process inputs
    nhours = 24
    start = day*24
    p = 0 ### Period - deprecated

    ### Get solver if necessary
    if solver in ['gurobi','gurobi_cl','gurobi_cmd']:
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
    loads = system.loads[p]
    lines = system.lines

    ###### Instantiate problem class
    tic = time.perf_counter()
    model = pulp.LpProblem("StorageDispatchModel", pulp.LpMaximize)

    ###### Construct time-resolved variables
    ### Generator powers
    genpower = pulp.LpVariable.dicts(
        'genpower',
        ((gen, hour) for gen in gens for hour in range(start,start+nhours)),
        lowBound=0, cat='Continuous')

    ### Storage charge
    storcharge = pulp.LpVariable.dicts(
        'storcharge',
        ((stor, hour) for stor in stors for hour in range(start,start+nhours)),
        lowBound=0, cat='Continuous')

    ### Storage discharge
    stordischarge = pulp.LpVariable.dicts(
        'stordischarge',
        ((stor, hour) for stor in stors for hour in range(start,start+nhours)),
        lowBound=0, cat='Continuous')

    ### Storage energy
    storenergy = pulp.LpVariable.dicts(
        'storenergy',
        ((stor, hour) for stor in stors for hour in range(start,start+nhours+1)),
        lowBound=0, cat='Continuous')

    ### Reservoir hydropower, if utilized
    if len(reshydros) > 0:
        ### Power
        respower = pulp.LpVariable.dicts(
            'respower',
            ((res, hour) for res in reshydros for hour in range(start,start+nhours)),
            lowBound=0, cat='Continuous')
        ### Spill
        spillpower = pulp.LpVariable.dicts(
            'spillpower',
            ((res, hour) for res in reshydros for hour in range(start,start+nhours)),
            lowBound=0, cat='Continuous')

    ### Transmission, if utilized
    if len(lines) > 0:
        ### Forward flow
        flowpos = pulp.LpVariable.dicts(
            'flowpos',
            ((line, hour) for line in lines for hour in range(start,start+nhours)),
            lowBound=0, cat='Continuous',
        )
        ### Reverse flow
        flowneg = pulp.LpVariable.dicts(
            'flowneg',
            ((line, hour) for line in lines for hour in range(start,start+nhours)),
            lowBound=0, cat='Continuous',
        )

    # ###### Objective Function: maximize storage energy at end of day
    # model += pulp.lpSum([storenergy[stor,start+nhours] for stor in stors])
    ###### Objective Function: maximize storage energy at end of day, minus VOLL
    model += pulp.lpSum(
        [storenergy[stor,start+nhours] for stor in stors]
        + [genpower['Lostload_{}'.format(node),hour]
           * gens['Lostload_{}'.format(node)].params['penalty']
           for node in system.nodes for hour in range(start,start+nhours)]
    )

    ###### Constraints
    ##### Generators
    for gen in gens:
        #### Bypass for Lostload
        if gen.startswith('Lostload'):
            continue
        #### Availability
        if gens[gen].always_available:
            for hour in range(start,start+nhours):
                model += genpower[gen,hour] <= gens[gen].cap
        else:
            for hour in range(start,start+nhours):
                model += genpower[gen,hour] <= (
                    gens[gen].cap * gens[gen].availability[p][hour])
        #### Minimum-generation level
        for hour in range(start,start+nhours):
            if gens[gen].always_available:
                model += genpower[gen,hour] >= gens[gen].cap * gens[gen].gen_min
            else:
                model += genpower[gen,hour] >= (
                    gens[gen].cap * gens[gen].availability[p][hour] * gens[gen].gen_min)

    ##### Storage
    for stor in stors:
        for hour in range(start,start+nhours):
            ### Charge limit
            model += storcharge[stor,hour] <= stors[stor].cappower
            ### Discharge limit
            model += stordischarge[stor,hour] <= stors[stor].cappower
            ### Simultaneous charge/discharge limit
            model += storcharge[stor,hour] + stordischarge[stor,hour] <= (
                stors[stor].cappower)
        ### Continuity
        for hour in range(start+1, start+nhours+1):
            model += (
                (storenergy[stor,hour-1]
                    - stordischarge[stor,hour-1] * (1/stors[stor].efficiency_discharge)
                    + storcharge[stor,hour-1] * stors[stor].efficiency_charge)
                * (1 - stors[stor].leakage_rate)
            ) == storenergy[stor,hour]
        ### Storage energy capacity
        for hour in range(start,start+nhours+1):
            model += storenergy[stor,hour] <= stors[stor].capenergy
        ### Initialize storage energy
        model += storenergy[stor,start] == energyinit[stor]

    ##### Meet load
    for load in loads: ### Load is labeled by node name (usually a state)
        for hour in range(start,start+nhours):
            model += (
                ### Generators
                pulp.lpSum([genpower[gen,hour] for gen in gens
                            if ('_{}'.format(load) in gen)])
                ### Storage
                + pulp.lpSum([stordischarge[stor,hour] - storcharge[stor,hour]
                              for stor in stors
                              if ('_{}'.format(load) in stor)])
                ### Hydro
                + pulp.lpSum([respower[res,hour] for res in reshydros
                              if ('_{}'.format(load) in res)])
                ##### Transmission
                ### Forward flow in lines starting at node; no losses (save for end)
                - pulp.lpSum([flowpos[line,hour]
                              for line in lines
                              if line.split('_')[0].split('|')[0] == load])
                ### Reverse flow in lines starting at node; subject to losses
                + pulp.lpSum([flowneg[line,hour] * (1 - lines[line].loss)
                              for line in lines
                              if line.split('_')[0].split('|')[0] == load])
                ### Forward flow in lines ending at node; subject to losses
                + pulp.lpSum([flowpos[line,hour] * (1 - lines[line].loss)
                              for line in lines
                              if line.split('_')[0].split('|')[1] == load])
                ### Reverse flow in lines ending at node; no losses (save for end)
                - pulp.lpSum([flowneg[line,hour]
                              for line in lines
                              if line.split('_')[0].split('|')[1] == load])
            ) == loads[load][hour], 'price_l{}_h{}'.format(load,hour)

    ###### Transmission constraints
    for line in lines:
        for hour in range(start,start+nhours):
            ##### Line capacity - measured at dispatched end
            ### Expand the line capacity by the losses, so that capacity is measured at output
            model += flowpos[line,hour] * (1 - lines[line].loss) <= lines[line].cap
            model += flowneg[line,hour] * (1 - lines[line].loss) <= lines[line].cap
            ##### Simultaneous flow limit
            model += (
                flowpos[line,hour] + flowneg[line,hour]
            ) * (1 - lines[line].loss) <= lines[line].cap

    ###### Hydro constraints
    ##### Reservoir
    for res in reshydros:
        #### Availability
        model += (
            pulp.lpSum([(respower[res,hour] + spillpower[res,hour])
                        for hour in range(start,start+nhours)]
            ) == reshydros[res].availability[p][day]
        )
        #### Minimum- and maximum-generation levels
        for hour in range(start,start+nhours):
            ### Min
            model += (
                ### The min() is required to deal with cases when the historical
                ### generation is less than Pmin*cap
                (respower[res,hour]
                ) >= min(reshydros[res].capacity_bound_hi * reshydros[res].gen_min,
                         reshydros[res].availability[p][day]/24)
            )
            ### Max flow restricted to nameplate power
            model += (respower[res,hour] <= reshydros[res].capacity_bound_hi)

    toc = time.perf_counter()



    ###### Report a few things
    time_setup = toc-tic
    if verbose >= 2:
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
    if verbose >= 2:
        print('model status: {}'.format(model.status))
        print('{:.2f} seconds to solve'.format(time_solve))
        sys.stdout.flush()

    ### Return nothing if no solution
    if model.status == 0:
        return None

    ########## Extract output
    ###### Hourly variables
    output_operation = {
        **{gen: [genpower[gen,hour].varValue
                 for hour in range(start,start+nhours)] for gen in gens},
        **{stor+'_charge': [-storcharge[stor,hour].varValue
                            for hour in range(start,start+nhours)] for stor in stors},
        **{stor+'_discharge': [stordischarge[stor,hour].varValue
                               for hour in range(start,start+nhours)] for stor in stors},
        **{stor+'_energy': [storenergy[stor,hour].varValue
                            for hour in range(start,start+nhours)] for stor in stors},
        **{res+'_power': [respower[res,hour].varValue
                          for hour in range(start,start+nhours)] for res in reshydros},
        **{res+'_spill': [spillpower[res,hour].varValue
                          for hour in range(start,start+nhours)] for res in reshydros},
        **{'Load_'+load: [loads[load][hour]
                  for hour in range(start,start+nhours)] for load in loads},
        **{line+'_pos': [flowpos[line,hour].varValue
                         for hour in range(start,start+nhours)] for line in lines},
        **{line+'_neg': [-flowneg[line,hour].varValue
                         for hour in range(start,start+nhours)] for line in lines},
    }
    ###### Ending storage energies
    output_storenergy = {stor: storenergy[stor,start+nhours].varValue for stor in stors}

    ###### Other values
    output_values = {
        'objective': pulp.value(model.objective),
        'status': model.status,
        'time_setup': time_setup,
        'time_solve': time_solve,
        'variables': len(model.variables()),
        'constraints': len(model.constraints),
        'lostload': sum(
            [sum(output_operation['Lostload_'+node]) for node in system.nodes]),
        'loadsum': sum([loads[load][start:start+nhours].sum() for load in loads]),
    }

    return output_storenergy, output_operation, output_values

