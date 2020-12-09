import os, sys, urllib, requests, time
import pandas as pd
from tqdm import tqdm, trange
from urllib.error import HTTPError

import zephyr
extdatapath = zephyr.settings.extdatapath

##########
### INPUTS

dates = pd.date_range('1998-01-01','2018-12-31',freq='D')
outpath = os.path.join(extdatapath,'MERRA2','M2T1NXSLV','')
os.makedirs(outpath, exist_ok=True)

#############
### PROCEDURE
print('{}-->{}'.format(dates[0],dates[-1]))

basequery = (
    'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2T1NXSLV.5.12.4/'
    '{year}/{month:02}/MERRA2_{v}00.tavg1_2d_slv_Nx.{date}.nc4.nc?'
    'U2M[0:23][140:300][8:248],V250[0:23][140:300][8:248],TROPT[0:23][140:300][8:248],'
    'TROPPB[0:23][140:300][8:248],T2M[0:23][140:300][8:248],TQL[0:23][140:300][8:248],'
    'T500[0:23][140:300][8:248],TOX[0:23][140:300][8:248],U850[0:23][140:300][8:248],'
    'PS[0:23][140:300][8:248],V850[0:23][140:300][8:248],OMEGA500[0:23][140:300][8:248],'
    'H250[0:23][140:300][8:248],Q250[0:23][140:300][8:248],T2MDEW[0:23][140:300][8:248],'
    'PBLTOP[0:23][140:300][8:248],CLDPRS[0:23][140:300][8:248],V50M[0:23][140:300][8:248],'
    'Q500[0:23][140:300][8:248],DISPH[0:23][140:300][8:248],H1000[0:23][140:300][8:248],'
    'TO3[0:23][140:300][8:248],TS[0:23][140:300][8:248],T10M[0:23][140:300][8:248],'
    'TROPPT[0:23][140:300][8:248],TQI[0:23][140:300][8:248],SLP[0:23][140:300][8:248],'
    'U250[0:23][140:300][8:248],Q850[0:23][140:300][8:248],ZLCL[0:23][140:300][8:248],'
    'TQV[0:23][140:300][8:248],V2M[0:23][140:300][8:248],T250[0:23][140:300][8:248],'
    'TROPQ[0:23][140:300][8:248],V10M[0:23][140:300][8:248],H850[0:23][140:300][8:248],'
    'T850[0:23][140:300][8:248],U50M[0:23][140:300][8:248],U10M[0:23][140:300][8:248],'
    'QV2M[0:23][140:300][8:248],CLDTMP[0:23][140:300][8:248],TROPPV[0:23][140:300][8:248],'
    'H500[0:23][140:300][8:248],V500[0:23][140:300][8:248],T2MWET[0:23][140:300][8:248],'
    'U500[0:23][140:300][8:248],QV10M[0:23][140:300][8:248],time,lat[140:300],lon[8:248]'
)

for date in tqdm(dates):
    ### Generate new query
    query = basequery.format(
        date=date.strftime('%Y%m%d'), year=date.year, month=date.month,
        v=(4 if date.year >= 2011 else 3 if date.year >= 2001 else 2 if date.year >= 1992 else 1))
    ### Continue
    savename = outpath + os.path.basename(query.split('?')[0])
    if not os.path.exists(savename):
        attempts = 0
        while attempts < 200:
            try:
                r = requests.get(query)
                with open(savename, 'wb') as f:
                    f.write(r.content)
                break
            except HTTPError as err:
                print(
                    'Rebuffed on attempt # {} at {} by "{}".'
                    'Will retry in 60 seconds.'.format(
                        attempts, zephyr.toolbox.nowtime(), err))
                attempts += 1
                time.sleep(60)
    else:
        pass
