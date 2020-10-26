import os
# ### Use these if project folder is different from module folder
# projpath = os.path.expanduser('~/path/to/project/folder/')
# datapath = os.path.expanduser('~/path/to/project/folder/Data/')
### Use these if project folder contains module folder
projpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
datapath = os.path.join(projpath, 'in') + os.sep

### Use extdatapath if you store resource data in an external drive
extdatapath = os.path.expanduser('~/../../Volumes/EXAMPLE/Data/')

apikeys = {
    ### Get a googlemaps API key at 
    ### https://developers.google.com/maps/documentation/geocoding/get-api-key
    'googlemaps': 'yourAPIkey',
    ### Get an NSRDB key at https://developer.nrel.gov/signup/
    'nsrdb': 'yourAPIkey',
}
nsrdbparams = {
    ### Use '+' for spaces
    'full_name': 'your+name',
    'email': 'your@email.com',
    'affiliation': 'your+affiliation',
    'reason': 'your+reason',
    'mailing_list': 'true',
}
