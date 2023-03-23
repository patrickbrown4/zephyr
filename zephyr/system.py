#%%### Imports
import pandas as pd


#%%### Functions

def crf(wacc, lifetime):
    out = ((wacc * (1 + wacc) ** lifetime) 
           / ((1 + wacc) ** lifetime - 1))
    return out

#%%### Classes

class Defaults:
    """
    """
    def __init__(self, paramsfile):
        ###### Load the parameters
        ### Generators
        dfin_generators = pd.read_excel(
            paramsfile, sheet_name='Generators', skiprows=[1,2], index_col='Technology',)
        self.params_generators = dfin_generators.T.to_dict()

        ### Fuel cost and carbon content
        dfin_fuels = pd.read_excel(
            paramsfile, sheet_name='Fuels', skiprows=[1,2], index_col='Fuel',)
        self.params_fuels = dfin_fuels.T.to_dict()

        ### Storage_fixed
        dfin_storages_fixed = pd.read_excel(
            paramsfile, sheet_name='Storage', skiprows=[1,2], index_col='Technology',)
        self.params_storage_fixed = dfin_storages_fixed.T.to_dict()
        
        ### Storage
        dfin_storages = pd.read_excel(
            paramsfile, sheet_name='Storage_variable', skiprows=[1,2], index_col='Technology',)
        self.params_storage = dfin_storages.T.to_dict()
        
        ### Hydro
        # try:
        dfin_hydros = pd.read_excel(
            paramsfile, sheet_name='Hydro', skiprows=[1,2], index_col='Technology',)
        self.params_hydro = dfin_hydros.T.to_dict()
        # except:
        #     warn('No Hydro sheet found; try upgrading to version >= 7')
        
        ### Transmission
        # try:
        dfin_transmission = pd.read_excel(
            paramsfile, sheet_name='Transmission', skiprows=[1,2], index_col='voltage',)
        self.params_transmission = dfin_transmission.T.to_dict()
        # except:
        #     warn('No Transmission sheet found; try upgrading to version >= 7')

        # ### Financials
        # self.wacc = float(
        #     pd.read_excel(
        #         paramsfile, sheet_name='Financials', skiprows=[1,2], squeeze=True, 
        #         usecols=['wacc'],
        #     ).dropna())


class Storage:
    """
    """
    def __init__(self, name, defaults=None, infile=None, **kwargs):
        self.name = name
        ### Defaults
        if (defaults is None) and (infile is None):
            self.params = {}
        elif (defaults is None):
            defaults = Defaults(infile)
        elif (infile is None):
            self.params = defaults.params_storage[name].copy()
        ### Overwrite default params if desired
        for key in kwargs:
            if kwargs[key] is None:
                continue
            self.params[key] = kwargs[key]

        ### Default attributes from parameters file
        self.cost_capex_E = self.params['cost_capex_E']
        self.cost_capex_P = self.params['cost_capex_P']
        self.lifetime = self.params['lifetime']
        self.wacc = self.params['wacc']
        # self.cost_fom = self.params['cost_fom']
        self.cost_fom_E = self.params['cost_fom_E']
        self.cost_fom_P = self.params['cost_fom_P']
        self.cost_vom = self.params['cost_vom']
        self.efficiency_charge = self.params['efficiency_charge']
        self.efficiency_discharge = self.params['efficiency_discharge']
        self.leakage_rate = self.params['leakage_rate']
        ### Derived attributes
        if 'cost_annual_E' not in kwargs:
            self.cost_annual_E = self.cost_capex_E * crf(self.wacc, self.lifetime)
        else:
            self.cost_annual_E = kwargs['cost_annual_E']
        if 'cost_annual_P' not in kwargs:
            self.cost_annual_P = self.cost_capex_P * crf(self.wacc, self.lifetime)
        else:
            self.cost_annual_P = kwargs['cost_annual_P']
        ### Default boolean attributes
        self.has_power_maximum = False
        self.has_power_minimum = False
        self.has_energy_maximum = False
        self.has_energy_minimum = False
        self.has_duration_maximum = False
        self.has_duration_minimum = False
        self.provides_reserves = bool(self.params['reserves'])
        self.constrained_energy = True
        
    def add_availability(self, availability=None):
        self.availability = availability
        return self
    
    def add_distance(self, distance=0):
        self.distance = distance
        return self
    
    # def add_capacity_limit_up(self, capacity_limit_up):
    #     self.capacity_limit_up = capacity_limit_up
    #     self.has_capacity_limit = True
    #     return self
    def add_power_bound_hi(self, max_power):
        self.power_bound_hi = max_power
        self.has_power_maximum = True
        return self

    def add_energy_bound_hi(self, max_energy):
        self.energy_bound_hi = max_energy
        self.has_energy_maximum = True
        return self    

    def add_power_bound_lo(self, min_power):
        """ Units: GW """
        self.power_bound_lo = min_power
        self.has_power_minimum = True
        return self

    def add_energy_bound_lo(self, min_energy):
        """ Units: GWh """
        self.energy_bound_lo = min_energy
        self.has_energy_minimum = True
        return self

    def add_duration_min(self, duration_min):
        """ Units: hours [MWh/MWac] """
        self.duration_min = duration_min
        self.has_duration_minimum = True
        return self

    def add_duration_max(self, duration_max):
        """ Units: hours [MWh/MWac] """
        self.duration_max = duration_max
        self.has_duration_maximum = True
        return self
    
    def localize(self, node=0):
        self.node = node
        return self

    def set_constrained_energy(self, constrained_energy=True):
        """Can be used to turn off the energy constraint"""
        self.constrained_energy = constrained_energy
        return self


class Gen:
    """
    """
    def __init__(self, name, defaults=None, infile=None, **kwargs):
        self.name = name
        ### Defaults
        if (defaults is None) and (infile is None):
            self.params = {}
        elif (defaults is None):
            defaults = Defaults(infile)
        elif (infile is None):
            self.params = defaults.params_generators[name].copy()
        ### Overwrite default params if desired
        for key in kwargs:
            if kwargs[key] is None:
                continue
            self.params[key] = kwargs[key]

        ### Default attributes from parameters file
        self.cost_capex = self.params['cost_capex']
        self.lifetime = self.params['lifetime']
        self.wacc = self.params['wacc']
        self.cost_fom = self.params['cost_fom']
        self.cost_vom = self.params['cost_vom']
        self.ramp_up = self.params['ramp_up']
        self.ramp_down = self.params['ramp_down']
        self.gen_min = self.params['gen_min']
        self.fuel = self.params['fuel']
        self.heatrate = self.params['heatrate']
        self.cost_fuel = defaults.params_fuels[self.fuel]['cost'] * self.heatrate
        ### emissionsrate in t/GWh
        self.emissionsrate = defaults.params_fuels[self.fuel]['co2_emissions'] * self.heatrate
        self.cost_emissions = 0
        ### Derived attributes
        if 'cost_annual' not in kwargs:
            self.cost_annual = self.cost_capex * crf(self.wacc, self.lifetime)
        else:
            self.cost_annual = kwargs['cost_annual']
        self.availability = {}
        self.periods = []
        self.has_capacity_maximum = False
        self.has_capacity_minimum = False
        self.always_available = True
        self.provides_reserves = bool(self.params['reserves'])
        self.perfectly_rampable = (
            True if (self.ramp_up == 1) and (self.ramp_down == 1) else False)
        self.meets_rps = (
            True if self.name.lower().startswith(('pv','solar','wind','hydro')) else False)
        
    def add_availability(self, availability, period=None):
        if type(availability) in [pd.DataFrame, pd.Series]:
            # self.availability_timeseries = availability
            # self.availability = self.availability_timeseries.values
            self.availability[period] = availability.values
        else:
            self.availability[period] = availability
        self.always_available = False
        self.periods.append(period)
        return self
    
    def add_distance(self, distance=0):
        """ Units: km """
        self.distance = distance
        return self
    
    def add_capacity_bound_hi(self, capacity_bound_hi):
        """ Units: GW """
        self.capacity_bound_hi = capacity_bound_hi
        self.has_capacity_maximum = True
        return self

    def add_capacity_bound_lo(self, min_capacity):
        """ Units: GW """
        self.capacity_bound_lo = min_capacity
        self.has_capacity_minimum = True
        return self
    
    def localize(self, node=0):
        self.node = node
        return self


class HydroRes:
    """
    """
    def __init__(self, name, defaults=None, infile=None, 
                 newbuild=False, balancelength='day', **kwargs):
        ### Defaults
        if (defaults is None) and (infile is None):
            self.params = {}
        elif (defaults is None):
            defaults = Defaults(infile)
        elif (infile is None):
            self.params = defaults.params_hydro[name].copy()
        ### Balance period
        if balancelength in ['day','Day',None]:
            self.balance = 'day'
        else:
            raise Exception("Only balancelength=='day' is supported")
        ### Overwrite default params if desired
        for key in kwargs:
            if kwargs[key] is None:
                continue
            self.params[key] = kwargs[key]
        
        ### Default attributes from parameters file
        if newbuild == True:
            self.cost_capex = self.params['cost_capex']
        else:
            self.cost_capex = 0
        self.lifetime = self.params['lifetime']
        self.wacc = self.params['wacc']
        self.cost_fom = self.params['cost_fom']
        self.cost_vom = self.params['cost_vom']
        self.ramp_up = self.params['ramp_up']
        self.ramp_down = self.params['ramp_down']
        self.gen_min = self.params['gen_min']
        ### Derived attributes
        if 'cost_annual' not in kwargs:
            self.cost_annual = self.cost_capex * crf(self.wacc, self.lifetime)
        else:
            self.cost_annual = kwargs['cost_annual']
        self.availability = {}
        self.periods = []
        self.has_capacity_maximum = False
        self.has_capacity_minimum = False
        self.always_available = True
        self.provides_reserves = bool(self.params['reserves'])
        self.perfectly_rampable = (
            True if (self.ramp_up == 1) and (self.ramp_down == 1) else False)
        self.meets_rps = True
    
    def add_availability(self, availability, period=None):
        """
        Frequency of availability timeseries must match balancelength.
        Currently only daily balancing is supported, so length of hydro
        availability timeseries should be 1/24 the length of system simulation
        timeseries.
        """
        if type(availability) in [pd.DataFrame, pd.Series]:
            self.availability[period] = availability.values
        else:
            self.availability[period] = availability
        self.always_available = False
        self.periods.append(period)
        return self
    
    def add_distance(self, distance=0):
        """ Units: km """
        self.distance = distance
        return self
    
    def add_capacity_bound_hi(self, capacity_bound_hi):
        """ Units: GW """
        self.capacity_bound_hi = capacity_bound_hi
        self.has_capacity_maximum = True
        return self

    def add_capacity_bound_lo(self, min_capacity):
        """ Units: GW """
        self.capacity_bound_lo = min_capacity
        self.has_capacity_minimum = True
        return self
    
    def localize(self, node=0):
        self.node = node
        return self


class Line:
    """
    """
    ### Defaults
    def __init__(
        self, node1, node2, distance=0, newbuild=False, capacity=0, 
        voltage=345, defaults=None, infile=None, **kwargs):
        """
        TODO:
        * Bring in node1 and node2 as object attributes, then use them
        in the .model() function (rather than splitting the name)
        """
        ### Defaults
        if (defaults is None) and (infile is None):
            self.params = {}
        elif (defaults is None):
            defaults = Defaults(infile)
        elif (infile is None):
            self.params = defaults.params_transmission[voltage].copy()
        ### Overwrite default params if desired
        for key in kwargs:
            if kwargs[key] is None:
                continue
            self.params[key] = kwargs[key]
        
        ### Default attributes from parameters file
        self.cost_distance = self.params['cost_distance']
        self.cost_fixed = self.params['cost_fixed']
        self.lifetime = self.params['lifetime']
        self.wacc = self.params['wacc']
        self.cost_fom = self.params['cost_fom']
        self.cost_vom = self.params['cost_vom']
        self.loss_distance = self.params['loss_distance']
        self.provides_reserves = bool(self.params['reserves'])
        ### Attributes from initialization
        self.node1 = node1
        self.node2 = node2
        self.voltage = voltage
        ### Green/brown-field-dependendent parameters
        self.capacity = capacity
        self.newbuild = newbuild
        self.distance = distance

        ### Derived attributes
        if 'name' not in kwargs:
            self.name = '{}|{}'.format(node1,node2)
        else:
            self.name = kwargs['name']
        
        if 'cost_capex' not in kwargs:
            self.cost_capex = (self.cost_distance * distance + self.cost_fixed)
        else:
            self.cost_capex = kwargs['cost_capex']
        
        if 'cost_annual' not in kwargs:
            self.cost_annual = self.cost_capex * crf(self.wacc, self.lifetime)
        else:
            self.cost_annual = kwargs['cost_annual']
            
        if self.newbuild == False:
            self.cost_capex = 0
            self.cost_annual = 0
        
        self.cost_annual_tot = self.cost_annual + self.cost_fom

        if 'loss' not in kwargs:
            self.loss = self.loss_distance * self.distance
        else:
            self.loss = kwargs['loss']


class System:
    """
    """
    def __init__(self, name=None, periods=[None], years=None, nodes=[None]):
        self.name = name
        self.generators = {}
        self.periods = periods
        self.nodes = nodes
        self.storages = {}
        self.reshydros = {}
        self.loads = {period: {} for period in periods}
        self.lines = {}
        self.co2price = 0
        self.has_co2cap = False
        self.has_rps = False
        self.has_reserves = False
        self.multiperiod = False
        self.cycling_reserves = False
        if years is None:
            self.years = len(periods)
        else:
            self.years = years
    
    def add_generator(self, name, generator):
        self.generators[name] = generator
        self.generators[name].cost_emissions = (
            self.co2price * generator.emissionsrate / 1000000) ### M$/tonCO2
        return self

    def add_storage(self, name, storage):
        self.storages[name] = storage
        return self
    
    def add_reshydro(self, name, hydro):
        self.reshydros[name] = hydro
        return self
    
    def add_load(self, name, load, period=None):
        self.loads[period][name] = load
        return self
    
    def add_line(self, name, line):
        self.lines[name] = line
        return self
    
    def set_co2price(self, co2price=0):
        """
        Input: CO2 price in $/tonCO2
        """
        self.co2price = co2price
        for gen in self.generators:
            self.generators[gen].cost_emissions = (
                self.generators[gen].emissionsrate * co2price / 1000000)
        return self
    
    def set_co2cap(self, co2cap):
        """
        Input: CO2 cap in ton/GWh == kg/MWh == g/kWh
        """
        self.co2cap = co2cap
        self.has_co2cap = True
        return self
    
    def set_rps(self, rps):
        """
        Input: Renewable portfolio standard in fraction
        * Implemented as constraint on primary electricity generation (
        everything except storage) divided by total demand
        """
        if (rps < 0) or (rps > 1):
            raise Exception('RPS should be fraction between [0,1]')
        self.rps = rps
        self.has_rps = True
        return self

    def set_reserves(self, reservefraction):
        """
        Input: Reserves constraint as fraction of hourly demand.
        Notes: * Implemented as a single constraint over all load zones
        """
        if (reservefraction<0) or (reservefraction>1):
            print('Remember that reserves should be a fraction')
        self.reservefraction = reservefraction
        self.has_reserves = True
        return self

    def set_periods(self, periods=[None]):
        """
        Input: List of period names, matching period names for gens
        """
        self.periods = periods
        self.multiperiod = True
        return self

    ### Switches
    def set_cycling_reserves(self, on=True, power=1):
        """
        Source: "NREL 2016 - Capturing the Impact of Storage and Other 
        Flexible Technologies on Electric System Planning"
        Inputs: power: fraction of nameplate capacity at which storage operates
        """
        self.cycling_reserves = on
        self.cycling_reserves_power = power
        return self
