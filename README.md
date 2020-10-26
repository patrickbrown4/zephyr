# **ZEPHYR**: **Z**ero-emissions **E**lectricity system **P**lanning with **H**ourl**Y** operational **R**esolution

Setup instructions
------------------
* To set up your python environment, navigate to this directory and run:
    * `conda env create -f environment.yml`
    * `conda activate zephyr`

Usage notes
-----------
* After setting up the environment, the core model can be run for an example full-US scenario using the command: `python zephyr-run-base.py 23 -o test -s clp` (argument descriptions are given within zephyr-run-base.py).
* VRE supply curve functionality has not yet been fully tested in this environment.
* To use the NSRDB download functions, you will need to modify the "settings.py" file to insert a valid NSRDB API key, which can be requested from <https://developer.nrel.gov/signup/>. Locations can be specified by passing latitude, longitude floats to zephyr.data.downloadNSRDBfile(), or by passing a string googlemaps query to zephyr.io.queryNSRDBfile(). To use the googlemaps functionality, you will need to request a googlemaps API key (<https://developers.google.com/maps/documentation/javascript/get-api-key>) and insert it in the "settings.py" file.

Sources
-------
Additional documentation on PV modeling is provided in the following papers and repositories:

* [1] Brown, P.R.; O'Sullivan, F. "Shaping photovoltaic array output to align with changing wholesale electricity price profiles." Applied Energy 2019, 256, 113734. <https://doi.org/10.1016/j.apenergy.2019.113734>
    * <https://zenodo.org/record/3368397>
* [2] Brown, P.R.; O'Sullivan, F. "Spatial and temporal variation in the value of solar power across United States Electricity Markets." Renewable and Sustainable Energy Reviews 2020, 121, 109594. <https://doi.org/10.1016/j.rser.2019.109594>
    * <https://zenodo.org/record/3562896>

Some of the code in this repository is copied from the Zenodo repositories linked above.
