import os
# ### Use these if project folder is different from module folder
# projpath = os.path.expanduser('~/path/to/project/folder/')
# datapath = os.path.expanduser('~/path/to/project/folder/Data/')
### Use these if project folder contains module folder
projpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
datapath = os.path.join(projpath, 'in', '')

### extdatapath is where resource data is stored; can be an external drive or 
### same as datapath
# extdatapath = os.path.expanduser('~/../../Volumes/EXAMPLE/zephyr/in/')
extdatapath = datapath

apikeys = {
    ### Get an NREL API key at https://developer.nrel.gov/signup/
    'nrel': 'yourAPIkey',
    ### Googlemaps API key is optional, for use in zephyr.io.queryNSRDBfile()
    ### https://developers.google.com/maps/documentation/geocoding/get-api-key
    'googlemaps': 'yourAPIkey',
}
nsrdbparams = {
    ### Use '+' for spaces
    'full_name': 'your+name',
    'email': 'your@email.com',
    'affiliation': 'your+affiliation',
    'reason': 'your+reason',
    'mailing_list': 'true',
}
