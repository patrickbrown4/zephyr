import numpy as np

###########
### Classes

class StorageSystem:
    """
    """
    def __init__(self,
                 discharge_time=4,
                 roundtrip_eff=0.85,
                 charge_eff=None,
                 discharge_eff=None,
                 # leakage_rate=0.0005,
                 leakage_rate=0.0000416866, ### IRENA 2017: 0.1%/day
                 initial_charge_fraction=0.,
                 cost_vom=0.,
                 foresight='year',
                 wrap_end='wrap',
                ):
        """
        roundtrip_eff takes precedence over charge_eff, discharge_eff
        """
        self.discharge_time = discharge_time
        if roundtrip_eff is not None:
            self.roundtrip_eff = roundtrip_eff
            self.charge_eff = np.sqrt(roundtrip_eff)
            self.discharge_eff = np.sqrt(roundtrip_eff)
        elif (charge_eff is not None) and (discharge_eff is not None):
            self.charge_eff = charge_eff
            self.discharge_eff = discharge_eff
            self.roundtrip_eff = charge_eff * discharge_eff
        else:
            raise Exception('Must specify efficiency')
        self.leakage_rate = leakage_rate
        self.initial_charge_fraction = initial_charge_fraction
        self.cost_vom = cost_vom
        self.foresight = foresight
        self.wrap_end = wrap_end
