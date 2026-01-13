This is how I downloaded the datasets and organized them on kama. The original MAST search query was just for all timeseries observations of transiting planets between 1.6 and 4.0 Earth-radii. Grabbed all public datasets available as of ~Nov. 1, 2025.

Downloading datasets from MAST staging area 
```python
import ftplib
import getpass

passwd = getpass.getpass()
ftps = ftplib.FTP_TLS('archive.stsci.edu')
ftps.login(user='tagordon7@gmail.com', passwd=passwd)
ftps.prot_p()
ftps.cwd('stage')
ftps.cwd('tagordon7/tagordon7_2025113Z_e292be1f') 

filenames = ftps.nlst()

for filename in filenames:
    with open(filename, 'wb') as fp:
        ftps.retrbinary('RETR {}'.format(filename), fp.write)
```

Organizing files into directories by target, program, and visit. Requires [all_obs.csv](https://docs.google.com/spreadsheets/d/19vCIrBQzbx05_PQ-BREKnwgpEgSBExQJHc7yLXvzL0g/edit?gid=970772689#gid=970772689)
```python
import numpy as np
import os
import glob
import pandas as pd

all_obs = pd.read_csv('all_obs.csv')

list(all_obs.loc[(all_obs['program'] == 1033) & (all_obs['observation number'] == 5)]['planet name'])[0]

files = glob.glob('*.fits')

translation_dict = {}
for f in files:
    translation_dict[f] = {}
    translation_dict[f]['program_id'] = np.int64(f[2:7])
    translation_dict[f]['obs_num'] = np.int64(f[7:10])
    translation_dict[f]['visit_num'] = np.int64(f[14:16])

for f in files:
    pid = translation_dict[f]['program_id']
    obs =  translation_dict[f]['obs_num']

    try:
        name = list(all_obs.loc[(all_obs['program'] == pid) & (all_obs['observation number'] == obs)]['planet name'])[0]
        name = name.replace(' ', '_')
    except:
        name = 'program' + str(pid)
        
    dir_name = name + '/' + str(pid) + '_' + str(obs)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    os.rename(f, dir_name + '/' + f)
```
