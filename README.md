# **ZEPHYR**: **Z**ero-emissions **E**lectricity system **P**lanning with **H**ourl**Y** operational **R**esolution

## Setup instructions
* To set up your python environment, navigate to this directory and run:
    * `conda env create -f environment.yml`
    * `conda activate zephyr`
* To use the renewable-energy supply-curve funcationality (not required for running the base capacity-expansion model), perform the following steps:
    * Request an NREL API key from <https://developer.nrel.gov/signup/> and paste it into `apikeys['nrel']` in `settings.py`.
    * Follow the instructions at <https://github.com/NREL/hsds-examples> to set up h5pyd for the NREL HSDS endpoint. Namely:
        * Run `hsconfigure`
        * At the prompt, enter the text after the `=`:
            * hs_endpoint = `https://developer.nrel.gov/api/hsds`
            * hs_username = `None`
            * hs_password = `None`
            * hs_api_key = the same NREL API key you pasted into `settings.py`
        * These values will be stored in `~/.hscfg`

## Usage notes
### Capacity-planning model
* `zephyr-run-base.py`
    * After setting up the environment, the core model can be run for an example full-US scenario using the command: `python zephyr-run-base.py 23 -o test -s clp`.
        * `23` indicates the case to use from the cases file (in this case, running for the full US at PA resolution over 2007–2013 with new transmission allowed).
        * `-o test` indicates both the folder in which to store the outputs (`out/test`) and the cases file from which to read the input settings (`cases-test.xlsx`).
        * `-s clp` indicates that the COIN-LP solver should be used. `-s gurobi` also works if you have gurobi installed with a working license.
### Renewable-energy supply curves
* **VRE supply curve functionality has not yet been fully tested in this environment.**
* `vresc_-1-icomesh.py`
    * Generates the icosahedral mesh used to define NSRDB (solar) points.
        * Outputs: `io/usa-points-icomesh-x[-atan(invPHI)+90-lat-11]-z[90+lon]-{subdiv}subdiv.csv`
* `vresc_0-nsrdb-download-icosahedralmesh.py`
    * Submits download requests for NSRDB (solar) data. Before use, update `nsrdbparams` in `settings.py` with your email address and other info. Once the downloads are prepared you will receive emails at the address you provided with links to download the data. By default, requests are submitted in 1000-site chunks for 41990 sites over the continental US in 1-year batches from 20007–2013.
* `vresc_0-download-wtkhsds-geoslice.py`
    * Downloads WIND Toolkit (WTK) data. To increase download speed, we download the data in single-timestamp slices over the full array of points, downsampled by a factor of 2x2 (i.e. at 4km resolution instead of 2km resolution).
* `vresc_1-wtk-hsds-timeslice-to-timeseries.py`
    * Reshapes the single-timestamp all-points WTK data into single-point all-timestamps to match the format of the NSRDB data.
* `vresc_2-polyavailable-states.py`
    * Creates and saves a polygon of available area for wind and solar development within each zone accounting for different land exclusions.
        * Outputs: `io/geo/developable-area/{zonesource}/polyavailable-water,parks,native,mountains,urban-{zonesource}_{zone}.poly.p`
* `vresc_3-sitearea-zoneoverlap.py`
    * Calculates the overlap area between the available area for PV and wind calculated in `vresc_2-polyavailale-states.py` and Vornoi polygons around the NSRDB and WTK points.
        * Outputs: `io/geo/developable-area/{zonesource}/{resource}/sitearea-water,parks,native,moutains,urban-{zonesource}_{zone}.csv`
* `vresc_4-transmission-site-distance.py`
    * Determines the interconnection (spur-line + trunk-line reinforcement) cost for each solar and wind site.
        * Outputs: `io/cf-2007_2013/{transcostmultiplier}x/{level}/{resource}/distance-station_urbanedge-mincost/{resource}-urbanU-trans230-{level}_{zone}.csv`
* `vresc_5-cfsim-lcoebins.py`, `vresc_6-cfsim-binsum.py`
    * Simulates the levelized cost of electricity (LCOE) and hourly capacity factor (CF) for each solar/wind site within a given zone. Bins sites by LCOE, then calculates the weighted-average hourly CF (weighted by developable area associated with each site) for each bin. Saves the weighted-average hourly CF and total developable area within each bin.
    * If running for PV on a machine with sufficient memory, use the `-f` flag when runnning `vresc_5-cfsim-lcoebins.py` to perform the entire procedure in one shot.
    * If running for wind, run `vresc_5-cfsim-lcoebins.py` without the `-f` flag to perform the binning procedure, then run `vresc_6-cfsim-binsum.py` to generate the hourly CF profiles.

Sources
-------
Additional documentation on PV modeling is provided in the following papers and repositories:

* [1] Brown, P.R.; O'Sullivan, F. "Shaping photovoltaic array output to align with changing wholesale electricity price profiles." Applied Energy 2019, 256, 113734. <https://doi.org/10.1016/j.apenergy.2019.113734>
    * <https://zenodo.org/record/3368397>
* [2] Brown, P.R.; O'Sullivan, F. "Spatial and temporal variation in the value of solar power across United States Electricity Markets." Renewable and Sustainable Energy Reviews 2020, 121, 109594. <https://doi.org/10.1016/j.rser.2019.109594>
    * <https://zenodo.org/record/3562896>

Some of the code in this repository is copied from the Zenodo repositories linked above.
