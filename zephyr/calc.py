import pandas as pd
import numpy as np
import sys, os
import scipy, scipy.optimize

mfile = os.path.abspath(__file__)
mpath = os.path.dirname(mfile)

#############
### STATS ###

def errorstats(calc, meas, returndict=False):
    """
    Inputs
    ------
    calc: calculated/simulated values
    meas: actual measured values

    Returns
    -------
    tuple of:
    * n: Number of observations
    * CC: Pearson correlation coefficient [fraction]
    * MAE: Mean absolute error
    * MBE: Mean bias error
    * rMBE: Relative mean bias error
    * RMSE: Root mean squared error
    * rRMSE: Relative room mean squared error

    Dataframe columns
    -----------------
    columns = ['n', 'CC', 'MAE', 'MBE', 'rMBE', 'RMSE', 'rRMSE']
    columnlabels = ['n [# obs]', 'CC [fraction]', 
                    'MAE [% CF]', 'MBE [% CF]',
                    'rMBE [%]', 'RMSE [% CF]', 'rRMSE [%]']
    """
    error = calc - meas
    rerror = (calc - meas) / np.abs(meas)
    n = len(error)

    data = pd.DataFrame(
        [meas, calc, error, rerror], 
        index=['meas', 'calc', 'error', 'rerror']).T
    data.drop(data[data.rerror == np.inf].index, inplace=True)

    corrcoef = np.corrcoef(meas, calc)[0][1]  # Pearson correlation coefficient
    mae = np.abs(error).sum() / n             # mean absolute error (MAE)
    mbe = error.sum() / n                     # mean bias error (MBE)
    rmse = np.sqrt((error**2).sum() / n)      # root mean square error (RMSE)
    rmbe = mbe / meas.sum() * n               # relative MBE (NREL 2017)
    rrmse = rmse / np.sqrt((meas**2).sum() / n)  # relative RMSE (NREL 2017)

    out = (n, corrcoef, mae, mbe, rmbe*100, rmse, rrmse*100)
    if returndict:
        return dict(zip(
            ('n', 'cc', 'mae', 'mbe', 'rmbe', 'rmse', 'rrmse'),
            out))
    else:
        return out

##################
### FINANCIALS ###

##########
### Common

def inflate(yearin=None, yearout=2017, value=None):
    """
    License (public domain)
    ------
    https://www.bls.gov/bls/linksite.htm

    Usage
    -----
    * Download input data from https://data.bls.gov/timeseries/CUUR0000SA0.
    Select years from 1913--2018 and include annual averages, then 
    download .xlsx file and save at projpath+'Data/BLS/inflation_cpi_level.xlsx'
    """
    ### Set filepaths
    file_inflation_level = os.path.join(mpath,'data','BLS','inflation_cpi_level.xlsx')
    
    ### Load the file
    try:
        dsinflation = pd.read_excel(
            file_inflation_level, skiprows=11, index_col=0)['Annual']
    except FileNotFoundError as err:
        print("Download input data from https://data.bls.gov/timeseries/CUUR0000SA0. "
              "Select years from 1913--2018 and include annual averages, then download "
              ".xlsx file and save at projpath+'Data/BLS/inflation_cpi_level.xlsx'")
        print(err)
        raise FileNotFoundError
    
    ### Return the ds 
    if (yearin is None) and (value is None):
        return dsinflation
    ### Or return the ratio
    elif (value is None):
        return dsinflation[yearout] / dsinflation[yearin]
    ### Or return the inflated value
    else:
        return value * dsinflation[yearout] / dsinflation[yearin]

def depreciation(year, schedule='macrs', period=5):
    """
    Function
    --------
    Returns nominal depreciation rate [%] given choice of 
        depreciation schedule.
        
    Inputs
    ------
    year: int year of operation                [year]
    schedule: str in ['macrs', 'sl', 'none']   (default 'macrs')
    period: int length of depreciation period  [years] (default 5)
    
    Returns
    -------
    float: Nominal depreciation rate [%]. If year is
        outside the schedule, returns 0.
        
    References
    ----------
    * https://www.irs.gov/forms-pubs/about-publication-946
    * https://www.irs.gov/pub/irs-pdf/p946.pdf
    """
    if year == 0:
        # raise Exception('year must be >= 1')
        return 0
    index = int(year) - 1
        
    if schedule == 'macrs':
        if period not in [3, 5, 7, 10, 15, 20]:
            raise Exception("Invalid period")
        rate = {
            3: [33.33, 44.45, 14.81, 7.41],
            5: [20.00, 32.00, 19.20, 11.52, 11.52, 
                 5.76],
            7: [14.29, 24.49, 17.49, 12.49, 8.93, 
                 8.92,  8.93,  4.46],
            10: [10.00, 18.00, 14.40, 11.52, 9.22, 
                  7.37,  6.55,  6.55,  6.56, 6.55, 
                  3.28],
            15: [5.00, 9.50, 8.55, 7.70, 6.93, 
                 6.23, 5.90, 5.90, 5.91, 5.90, 
                 5.91, 5.90, 5.91, 5.90, 5.91,
                 2.95],
            20: [3.750, 7.219, 6.677, 6.177, 5.713,
                 5.285, 4.888, 4.522, 4.462, 4.461,
                 4.462, 4.461, 4.462, 4.461, 4.462,
                 4.461, 4.462, 4.461, 4.462, 4.461,
                 2.231],
        }
        
    elif schedule in ['sl', 'straight', 'straightline']:
        rate = {period: np.ones(period) * 100 / period}
        
    elif schedule in [None, 'none', False, 'no']:
        rate = {period: [0]}
        
    else:
        raise Exception("Invalid schedule")
        
    try:
        return rate[period][index]
    except IndexError:
        return 0

def lcoe(lifetime, discount, capex, cf, fom, itc=0, degradation=0):
    """
    Inputs
    ------
    lifetime:    economic lifetime [years]
    discount:    discount rate [fraction]
    capex:       year-0 capital expenditures [$/kWac]
    cf:          capacity factor [fraction]
    fom:         fixed O&M costs [$/kWac-yr]
    itc:         investment tax credit [fraction]
    degradation: output degradation per year [fraction]

    Outputs
    -------
    LCOE in $/kWh

    Assumptions
    -----------
    * 8760 hours per year
    """
    ### Index
    years = np.arange(0,lifetime+0.1,1)
    ### Discount rate
    discounts = np.array([1/((1+discount)**year) for year in years])
    ### Degradation
    degrades = np.array([(1-degradation)**year for year in years])
    ### FOM costs
    costs = np.ones(len(years)) * fom
    ### Add capex cost to year 0 and remove FOM
    costs[0] = capex * (1 - itc)
    ### Discount costs
    costs_discounted = costs * discounts
    ### Energy generation, discounted and degraded
    energy_discounted = cf * 8760 * discounts * degrades
    ### Set first-year generation to zero
    energy_discounted[0] = 0
    ### Sum and return
    out = costs_discounted.sum() / energy_discounted.sum()
    return out

############
### NREL ATB

def confinfactor(costschedule=None, taxrate=28, interest_nominal=3.7,
    taxrate_federal=None, taxrate_state=None,
    deduct_state_taxrate=False):
    """
    Inputs
    ------
    costschedule: List with percent of construction costs spent in 
        each year. Must sum to 100. 
        Examples: PV is [100], natural gas is [80, 10, 10]
    
    Source
    ------
    NREL ATB 2018: https://atb.nrel.gov/, 
    https://data.nrel.gov/files/89/2018-ATB-data-interim-geo.xlsm
    """
    ### Calculate taxrate if taxrate_federal and taxrate_state
    if (taxrate_federal is not None) and (taxrate_state is not None):
        ### Note: in 2018, the ability to fully deduct state taxes from 
        ### federal income tax was repealed for individuals. 
        ### Not sure about corporations, but probably.
        ### So we simply add the federal and state tax rates.
        if deduct_state_taxrate is False:
            taxrate = taxrate_federal + taxrate_state
        elif deduct_state_taxrate is True:
            taxrate = (taxrate_federal*0.01 * (1 - taxrate_state*0.01) 
                       + taxrate_state*0.01) * 100
    
    ### Assume all construction costs are in first year if no schedule supplied
    if costschedule is None:
        costschedule = [100]
    if sum(costschedule) != 100:
        raise Exception(
            'costschedule elements must sum to 100 but sum to {}'.format(
                sum(costschedule)))
        
    ### Years of construction
    years = range(len(costschedule))
    
    ### Sum costs of debt
    yearlyinterest = [
        (
            costschedule[year]*0.01 
            * (1 + (1 - taxrate*0.01)
                   * ((1 + interest_nominal*0.01)**(year + 0.5) - 1)
              )
        )
        for year in years
    ]
    
    return sum(yearlyinterest)

def depreciation_present_value(
    wacc=7, inflationrate=2.5, schedule='macrs', period=5):
    """
    Notes
    -----
    * Inputs are in percent (so 10% is 10, not 0.1)
    
    Inputs
    ------
    wacc: REAL weighted average cost of capital
    
    Source
    ------
    NREL ATB 2018: https://atb.nrel.gov/, 
    https://data.nrel.gov/files/89/2018-ATB-data-interim-geo.xlsm
    """
    ### Years over which depreciation applies
    ### (note that MACRS has one extra year; we keep it for straight-line
    ### because it's zero after the end of the depreciation period anyway)
    years = range(1,period+2)
    
    ### Sum discounted values of depreciation
    out = sum([
        (
            depreciation(year, schedule, period) 
            / (((1 + wacc*0.01)*(1+inflationrate*0.01))**year)
        )
        for year in years
    ])*0.01 ### Convert to fraction instead of percent

    return out

def projfinfactor(
    wacc=7, taxrate=28, 
    inflationrate=2.5,
    schedule='macrs', period=5,
    taxrate_federal=None, taxrate_state=None,
    deduct_state_taxrate=False,):
    """
    Notes
    -----
    * Inputs are in percent (so 10% is 10, not 0.1)
    
    Inputs
    ------
    wacc: REAL weighted average cost of capital
    
    Source
    ------
    NREL ATB 2018: https://atb.nrel.gov/, 
    https://data.nrel.gov/files/89/2018-ATB-data-interim-geo.xlsm
    """
    ### Calculate taxrate if taxrate_federal and taxrate_state
    if (taxrate_federal is not None) and (taxrate_state is not None):
        ### Note: in 2018, the ability to fully deduct state taxes from 
        ### federal income tax was repealed for individuals. 
        ### Not sure about corporations, but probably.
        ### So we simply add the federal and state tax rates.
        if deduct_state_taxrate is False:
            taxrate = taxrate_federal + taxrate_state
        elif deduct_state_taxrate is True:
            taxrate = (taxrate_federal*0.01 * (1 - taxrate_state*0.01) 
                       + taxrate_state*0.01) * 100
    
    ### Get present value of depreciation
    dpv = depreciation_present_value(
        wacc=wacc, inflationrate=inflationrate, 
        schedule=schedule, period=period)
    
    ### Calculate factor according to NREL ATB
    out = (1 - taxrate*0.01 * dpv) / (1 - taxrate*0.01)
    
    return out

def lcoe_atb(
    overnightcost, cf, wacc, lifetime, fom=0, itc=0,
    taxrate=28, interest_nominal=3.7,
    taxrate_federal=None, taxrate_state=None,
    deduct_state_taxrate=False,
    inflationrate=2.5,
    schedule='macrs', period=5,
    costschedule=None,):
    """
    Notes
    -----
    * All inputs are in percents, not fractions

    Source
    ------
    NREL ATB 2018: https://atb.nrel.gov/, 
    https://data.nrel.gov/files/89/2018-ATB-data-interim-geo.xlsm
    """
    ### Calculate taxrate if taxrate_federal and taxrate_state
    if (taxrate_federal is not None) and (taxrate_state is not None):
        ### Note: in 2018, the ability to fully deduct state taxes from 
        ### federal income tax was repealed for individuals. 
        ### Not sure about corporations, but probably.
        ### So we simply add the federal and state tax rates.
        if deduct_state_taxrate is False:
            taxrate = taxrate_federal + taxrate_state
        elif deduct_state_taxrate is True:
            taxrate = (taxrate_federal*0.01 * (1 - taxrate_state*0.01) 
                       + taxrate_state*0.01) * 100
    
    out = (
        (
            crf(wacc=wacc, lifetime=lifetime)
            * projfinfactor(
                wacc=wacc, taxrate=taxrate, inflationrate=inflationrate, 
                schedule=schedule, period=period,
                taxrate_federal=taxrate_federal, taxrate_state=taxrate_state,
                deduct_state_taxrate=deduct_state_taxrate)
            * confinfactor(
                costschedule=costschedule, taxrate=taxrate, 
                interest_nominal=interest_nominal, 
                taxrate_federal=taxrate_federal, taxrate_state=taxrate_state, 
                deduct_state_taxrate=deduct_state_taxrate)
            * overnightcost
            * (1 - itc*0.01)
            + fom
        ) / (cf * 8760)
    )
    
    return out

#########
### Solar

def npv(
    revenue, carboncost, carbontons, cost_upfront,
    wacc, lifetime, degradationrate,
    cost_om, cost_om_units, 
    inflationrate=2.5,
    taxrate=None, taxrate_federal=None, taxrate_state=None,
    schedule='macrs', period=5, itc=0):
    """
    Function
    --------
    Calculates net present value given yearly revenue and
        lots of other financial assumptions
    
    Inputs
    ------
    revenue: numeric or np.array            [$/kWac-yr]
    wacc: numeric                           [percent]
    lifetime: numeric                       [years]
    cost_om: numeric                        [$/kWac-yr]
                              OR            [percent of cost_upfront]
    cost_upfront: numeric                   [$/kWac]
    carboncost: numeric                     [$/ton]
    carbontons: numeric                     [tons/MWac-yr]
    cost_om_units: str in ['$', '%']      
    inflationrate: numeric                  [percent] (default = 2.5)
    taxrate: numeric                        [percent]
    taxrate_federal: numeric or None        [percent]
    taxrate_state: numeric or None          [percent]
    schedule: str in ['macrs', 'sl', 'none] {input to depreciation()}
                                                (default = 'macrs')
    period: int                             {input to depreciation()}
                                                (default = 5)
    itc: numeric                            [percent] (default = 0)
    
    Outputs
    -------
    npv: numeric or np.array                [$/kWac]
    
    Reference for formulation
    -------------------------
    * http://dx.doi.org/10.3390/en10081100 (Hogan2017)

    Assumptions
    -----------
    References:
    * (Jordan2013) Jordan, D.C.; Kurtz, S.R. "Photovoltaic Degradation
    Rates - an Analytical Review", Prog.Photovolt: Res. Appl. 2013, 21:12-29
    dx.doi.org/10.1002/pip.1182
    * (NREL2017) Fu, R.; Feldman, D.; Margolis, R.; Woodhouse, M.; Ardani, K.
    "U.S. Solar Photovoltaic System Cost Benchmark: Q1 2017"
    NREL/TP-6A20-68925
    * (Hogan2017) Salles, M.B.C.; Huang, J.; Aziz, M.J.; Hogan, W.W.
    "Potential Arbitrage Revenue of Energy Storage Systems in PJM"
    Energies 2017, 10, 1100; dx.doi.org/10.3390/en10081100
    
    Degradation rate:
    * Jordan2013
        * 0.5% median value across ~2000 modules and systems
    * NREL2017
        * 0.75% assumed for utility-scale 2017
    
    O&M cost:
    * NREL2017
        * $15.4/kW-yr for utility-scale fixed-tilt for 2017
        * $18.5/kW-yr for utility-scale 1-ax track for 2017
        * $15/kW-yr for commercial for 2017
                
    Lifetime:
    * NREL2017
        * 30 years for all systems
        
    Tax rate:
    * Hogan 2017
        * 38%
    * NREL2017
        * 35%
        
    Other assumptions for utility-scale 2017:
    * NREL2017
        * Pre-inverter derate (1 - loss_system) = 90.5%
        * Inverter efficiency (1 - loss_inverter) = 2.0%
        * Inflation rate = 2.5%
        * Equity discount rate (real) = 6.3%
        * Debt interest rate = 4.5%
        * Debt fraction = 40%
        * IRR target = 6.46%


    """
    ### Make year index
    years = np.arange(1, lifetime+1, 1)
    ### Calculate cost_om based on cost_om_units, if necessary
    if cost_om_units in ['percent', '%', '%/yr', '%peryr']:
        cost_om_val = cost_om*0.01 * cost_upfront
    elif cost_om_units in ['dollars', '$', '$/kWac-yr', '$perkWac-yr']:
        cost_om_val = cost_om
    else:
        raise Exception("Invalid cost_om_units; try '$' or '%'")
    ### Calculate taxrate if taxrate_federal and taxrate_state
    if (taxrate_federal is not None) and (taxrate_state is not None):
        ### Note: in 2018, the ability to fully deduct state taxes from 
        ### federal income tax was repealed for individuals. 
        ### Not sure about corporations, but probably.
        ### So we simply add the federal and state tax rates.
        taxrate = taxrate_federal + taxrate_state
        ### Old way, with state income tax deduction
        ## taxrate = (taxrate_federal * (1 - taxrate_state / 100) 
        ##            + taxrate_state)
        
    
    npv = (
        sum(
            [(  (  (  (  (revenue + (carboncost * carbontons / 1000)) 
                         * (1 - degradationrate*0.01)**year)
                      - cost_om_val)
                   * (1 - taxrate*0.01)
                   + (  (depreciation(year, schedule, period))*0.01
                        /  (1 + inflationrate*0.01)**year
                        * cost_upfront
                        * taxrate*0.01
                     ) 
                )
                / ((1 + wacc*0.01)**year)
             )
             for year in years]
        )
        - cost_upfront * (1 - itc*0.01)
    )
        
    return npv


def npv_upfrontcost(
    cost_upfront, revenue, 
    carboncost=50, carbontons=0, wacc=7, lifetime=30, 
    degradationrate=0.5, cost_om=15, cost_om_units='$',
    inflationrate=2.5, taxrate=40, 
    taxrate_federal=None, taxrate_state=None,
    schedule='macrs', period=5, itc=0):
    """
    Inputs
    ------
    Same as npv()
    
    Outputs
    -------
    npv: numeric or np.array       [$/kWac]
    """
    out = npv(
        revenue=revenue, carboncost=carboncost, carbontons=carbontons, 
        cost_upfront=cost_upfront, wacc=wacc, lifetime=lifetime, 
        degradationrate=degradationrate, 
        cost_om=cost_om, cost_om_units=cost_om_units, 
        inflationrate=inflationrate, taxrate=taxrate,
        taxrate_federal=taxrate_federal, taxrate_state=taxrate_state,
        schedule=schedule, period=period, itc=itc)
    return out

def npv_carbon(
    carboncost, revenue, 
    carbontons=0, cost_upfront=1400, wacc=7, lifetime=30, 
    degradationrate=0.5, cost_om=15, cost_om_units='$',
    inflationrate=2.5, taxrate=40, 
    taxrate_federal=None, taxrate_state=None,
    schedule='macrs', period=5, itc=0):
    """
    Inputs
    ------
    Same as npv()
    
    Outputs
    -------
    npv: numeric or np.array       [$/kWac]
    """
    out = npv(
        revenue=revenue, carboncost=carboncost, carbontons=carbontons, 
        cost_upfront=cost_upfront, wacc=wacc, lifetime=lifetime, 
        degradationrate=degradationrate, 
        cost_om=cost_om, cost_om_units=cost_om_units, 
        inflationrate=inflationrate, taxrate=taxrate,
        taxrate_federal=taxrate_federal, taxrate_state=taxrate_state,
        schedule=schedule, period=period, itc=itc)
    return out

def npv_wacc(
    wacc, revenue, 
    carboncost=50, carbontons=0, cost_upfront=1400, lifetime=30, 
    degradationrate=0.5, cost_om=15, cost_om_units='$',
    inflationrate=2.5, taxrate=40, 
    taxrate_federal=None, taxrate_state=None,
    schedule='macrs', period=5, itc=0):
    """
    Inputs
    ------
    Same as npv()
    
    Outputs
    -------
    npv: numeric or np.array       [$/kWac]
    """
    out = npv(
        revenue=revenue, carboncost=carboncost, carbontons=carbontons, 
        cost_upfront=cost_upfront, wacc=wacc, lifetime=lifetime, 
        degradationrate=degradationrate, 
        cost_om=cost_om, cost_om_units=cost_om_units, 
        inflationrate=inflationrate, taxrate=taxrate,
        taxrate_federal=taxrate_federal, taxrate_state=taxrate_state,
        schedule=schedule, period=period, itc=itc)
    return out

def npv_revenue(
    revenue, cost_upfront=1443,
    carboncost=50, carbontons=0, wacc=7, lifetime=30, 
    degradationrate=0.5, cost_om=15, cost_om_units='$',
    inflationrate=2.5, taxrate=40, 
    taxrate_federal=None, taxrate_state=None,
    schedule='macrs', period=5, itc=0):
    """
    Inputs
    ------
    Same as npv()
    
    Outputs
    -------
    npv: numeric or np.array       [$/kWac]
    """
    out = npv(
        revenue=revenue, carboncost=carboncost, carbontons=carbontons, 
        cost_upfront=cost_upfront, wacc=wacc, lifetime=lifetime, 
        degradationrate=degradationrate, 
        cost_om=cost_om, cost_om_units=cost_om_units, 
        inflationrate=inflationrate, taxrate=taxrate,
        taxrate_federal=taxrate_federal, taxrate_state=taxrate_state,
        schedule=schedule, period=period, itc=itc)
    return out

def breakeven_upfrontcost(
    revenue, 
    carboncost=50, carbontons=0, wacc=7, lifetime=30, 
    degradationrate=0.5, cost_om=15, cost_om_units='$',
    inflationrate=2.5, taxrate=40,
    taxrate_federal=None, taxrate_state=None,
    schedule='macrs', period=5, itc=0, 
    maxiter=1000, xtol='default', 
    ab=(-1000, 100000), **kwargs):
    """
    Inputs
    ------
    revenue: numeric or np.array      [$/kWac-yr]
    carbontons:                       [tons/MWac-yr]
    cost_upfront: numeric or np.array [$/kWac-yr]
    wacc: numeric                     [percent]
    lifetime: numeric                 [years]
    degradationrate: numeric          [percent/yr]
    cost_om: numeric                  [$/kWac-yr]
                                 OR   [(percent of cost_upfront) / yr]
    cost_om_units: str in ['dollars', 'percent']
    taxrate: numeric                  [percent]
    
    Outputs
    -------
    carboncost: numeric               [$/ton]
    """
    ### Brent method
    result = scipy.optimize.brentq(
        npv_upfrontcost, 
        a=ab[0],
        b=ab[1],
        args=(revenue, carboncost, carbontons, wacc,
              lifetime, degradationrate, 
              cost_om, cost_om_units, 
              inflationrate, taxrate, 
              taxrate_federal, taxrate_state,
              schedule, period, itc),
        maxiter=maxiter,
    )
    return result

def breakeven_carboncost(
    revenue, 
    carbontons=0, cost_upfront=1400, wacc=7, lifetime=30, 
    degradationrate=0.5, cost_om=15, cost_om_units='$',
    inflationrate=2.5, taxrate=40, 
    taxrate_federal=None, taxrate_state=None,
    schedule='macrs', period=5, itc=0, 
    maxiter=1000, xtol='default', 
    ab=(-1000, 100000), **kwargs):
    """
    Inputs
    ------
    revenue: numeric or np.array      [$/kWac-yr]
    carbontons:                       [tons/MWac-yr]
    cost_upfront: numeric or np.array [$/kWac-yr]
    wacc: numeric                     [percent]
    lifetime: numeric                 [years]
    degradationrate: numeric          [percent/yr]
    cost_om: numeric                  [$/kWac-yr]
                                 OR   [(percent of cost_upfront) / yr]
    cost_om_units: str in ['dollars', 'percent']
    taxrate: numeric                  [percent]
    
    Outputs
    -------
    carboncost: numeric               [$/ton]
    """
    ### Brent method
    result = scipy.optimize.brentq(
        npv_carbon, 
        a=ab[0],
        b=ab[1],
        args=(revenue, carbontons, cost_upfront, wacc,
              lifetime, degradationrate, 
              cost_om, cost_om_units, 
              inflationrate, taxrate, 
              taxrate_federal, taxrate_state,
              schedule, period, itc),
        maxiter=maxiter,
    )
    return result

def breakeven_wacc(
    revenue, 
    carboncost=50, carbontons=0, cost_upfront=1400, lifetime=30, 
    degradationrate=0.5, cost_om=15, cost_om_units='$',
    inflationrate=2.5, taxrate=40, 
    taxrate_federal=None, taxrate_state=None,
    schedule='macrs', period=5,
    maxiter=1000, xtol='default', 
    ab=(-80, 10000), **kwargs):
    """
    Inputs
    ------
    revenue: numeric or np.array      [$/kWac-yr]
    carbontons:                       [tons/MWac-yr]
    cost_upfront: numeric or np.array [$/kWac-yr]
    wacc: numeric                     [percent]
    lifetime: numeric                 [years]
    degradationrate: numeric          [percent/yr]
    cost_om: numeric                  [$/kWac-yr]
                                 OR   [(percent of cost_upfront) / yr]
    cost_om_units: str in ['dollars', 'percent']
    taxrate: numeric                  [percent]
    
    Outputs
    -------
    carboncost: numeric               [$/ton]
    """
    ### Brent method
    result = scipy.optimize.brentq(
        npv_wacc, 
        a=ab[0],
        b=ab[1],
        args=(revenue, carboncost, carbontons, cost_upfront,
              lifetime, degradationrate, 
              cost_om, cost_om_units, 
              inflationrate, taxrate, 
              taxrate_federal, taxrate_state,
              schedule, period, itc),
        maxiter=maxiter,
    )
    return result

def breakeven_revenue(
    cost_upfront, 
    carboncost=50, carbontons=0, wacc=7, lifetime=30, 
    degradationrate=0.5, cost_om=15, cost_om_units='$',
    inflationrate=2.5, taxrate=40,
    taxrate_federal=None, taxrate_state=None,
    schedule='macrs', period=5, itc=0, 
    maxiter=1000, xtol='default', 
    ab=(-1000, 100000), **kwargs):
    """
    Inputs
    ------
    revenue: numeric or np.array      [$/kWac-yr]
    carbontons:                       [tons/MWac-yr]
    cost_upfront: numeric or np.array [$/kWac-yr]
    wacc: numeric                     [percent]
    lifetime: numeric                 [years]
    degradationrate: numeric          [percent/yr]
    cost_om: numeric                  [$/kWac-yr]
                                 OR   [(percent of cost_upfront) / yr]
    cost_om_units: str in ['dollars', 'percent']
    taxrate: numeric                  [percent]
    
    Outputs
    -------
    carboncost: numeric               [$/ton]
    """
    ### Brent method
    result = scipy.optimize.brentq(
        npv_revenue, 
        a=ab[0],
        b=ab[1],
        args=(cost_upfront, carboncost, carbontons, wacc,
              lifetime, degradationrate, 
              cost_om, cost_om_units, 
              inflationrate, taxrate, 
              taxrate_federal, taxrate_state,
              schedule, period, itc),
        maxiter=maxiter,
    )
    return result
