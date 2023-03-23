#%% Imports
import pandas as pd
import numpy as np
import os, sys, site, math, time, pickle, gzip
from glob import glob
from tqdm import tqdm, trange
from warnings import warn

import pulp

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
    model = pulp.LpProblem("Linear daily dispatch", pulp.LpMaximize)

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

