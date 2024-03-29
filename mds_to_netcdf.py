import os
import sys
import glob
import numpy as np
import xarray as xr
import datetime as dt
import ecco_v4_py as ecco
import matplotlib.pyplot as plt

'''
Convert output produced by diagnostics package to netCDF. This script should be
run in the same directory as data, data.diagnostics, available_diagnostics.log,
and data.cal.
This script can be called with or without arguments. Arguments should be variable
names to be converted, e.g. 
python mds_to_netcdf.py THETA SALT 
If no argument is provided, the script will obtain a list of variables from
data.diagnostics. In either case, the output path will be read from 
data.diagnostics, while information about model timesteps and calendar are read
from data and data.cal.
'''

################################################################################
if not all([os.path.isfile(F) for F in ['data.diagnostics','data.cal',\
                                        'data','available_diagnostics.log']]):
    print('This script should be called from the same location as '\
          +'"data","data.cal","data.diagnostics", and "available_diagnostics.log",'\
          +'but at least one of these files is missing. Exiting')
    sys.exit()
################################################################################
with open('data.diagnostics') as F:
    DD=F.readlines()

if len(sys.argv)>1:
    varlist=sys.argv[1:]
else:
    # Get list of variables in data.diagnostics
    E=[s.split("'")[1] for s in DD if ('fields(' in s) & (s[0]!='#') ]
    varlist=np.unique(E[:])

#1.) go through data.diagnostics and find the number(s) of each output variable
varnums=np.zeros(len(varlist))
varnums={}
for i in range(len(varlist)):
    for j in range(len(DD)):
        if ( ("'"+varlist[i]+"'") in DD[j]) and ('fields(' in DD[j]):
            if DD[j][0]=='#': continue
            if varlist[i] in varnums.keys():
                varnums[varlist[i]].append(\
                                    int(DD[j].split(')')[0].split(',')[-1]) )
            else:
                varnums[varlist[i]]=[]
                varnums[varlist[i]].append(\
                                    int(DD[j].split(')')[0].split(',')[-1]) )

#2.) " " "  and find the output path associated w/ the number of each variable
dirdct={}
for i in range(len(varlist)):
    for j in range(len(varnums[varlist[i]])):
        varnum=varnums[varlist[i]][j]
        for k in range(len(DD)):
            if ('filename('+str('%i' % varnum)+')') in DD[k]:
                if DD[k][0]=='#': continue
                tmpath=(DD[k].split("'")[1])
                if varlist[i] in dirdct.keys():
                    dirdct[varlist[i]].append(tmpath)
                else:
                    dirdct[varlist[i]]=[]
                    dirdct[varlist[i]].append(tmpath)#

#3.) Read available.diagnostics so we can find grid-related info to each var
with open('available_diagnostics.log') as F:
    diag_info=F.readlines()
    
################################################################################
# 4.): go through data and data.cal namelists to get info on output timing
print('getting calendar namelist info')
with open('data.cal') as F:
    cal=F.readlines()
    for L in cal:
        if L[0]=='#': continue
        if 'startDate_1' in L:
            DATE=int(L.split('=')[-1].split(',')[0])
        elif 'startDate_2' in L:
            TIME=int(L.split('=')[-1].split(',')[0])
        elif 'TheCalendar' in L:
            modelcal=L.split('=')[-1].split('\'')[1]

    time0=dt.datetime.strptime(str(DATE)+str(TIME),'%Y%m%d%H%M%S')
print('getting regular namelist info')

with open('data') as F:
    dat=F.readlines()
    for L in dat:
        if L[0]=='#': continue
        if 'deltaTClock' in L:
            deltaT=float(L.split('=')[-1].split(',')[0])
        elif 'nIter0' in L:
            nIter0=int(L.split('=')[-1].split(',')[0])

time0=time0+dt.timedelta(seconds=(deltaT*nIter0))

################################################################################
# 5.): Go through the variable list, read in data & metadata and save to netCDF
for varname in varlist:
    print('getting output info for '+varname+' from available_diagnostics.log')
    var_info=[s for s in diag_info if ('|'+varname+' '*(8-len(varname))+'|') in s]
    if len(var_info)>1: 
        print('sth went wrong')
    else: var_info=var_info[0].split('|')
    diagdct={}
    diagdct['Num']=int(var_info[0])
    diagdct['Name']=(var_info[1]).strip()
    diagdct['Levs']=int(var_info[2])
    diagdct['mate']=int(var_info[3]) if (len(var_info[3].strip())>0) else None
    diagdct['code']=var_info[4]
    diagdct['Units']=var_info[5].strip()
    diagdct['Description']=var_info[6][:-1]

    # Sometimes one variable can have multiple outputs 
    # (e.g. THETA_mon_mean and THETA_mon_inst in the default data.diagnostics)
    # Go through each instance of this variable and process separately

    for outfname in dirdct[varname]:
        ## GET OUTPUT LOCATION INFO
        #"filename" in data.diagnostics includes the path & filename prefix. 
        # First we want just the directory path...
        outdir='/'.join(outfname.split('/')[:-1])
        # ... then we want just the filename prefix
        outpfx=outfname.split('/')[-1]
        F=sorted(glob.glob(outfname+'.*.data')) # Get list of outputs

        ## GET OUTPUT TIMING INFO
        # ITERS=list of timsteps associated with output:
        ITERS=np.array([int(X.split('.')[-2]) for X in F]) 
        # times=datetime array of times using time0 and deltaT from data.cal
        times=[time0+dt.timedelta(seconds=(I-ITERS[0])*deltaT) for I in ITERS]

        ## GET OUTPUT GEOMETRY INFO
        # local gridpoint based on "code" column in available_diagnostics
        GPT={'M':'c','U':'w','V':'s','Z':'z'}[diagdct['code'][1]]
        #find number of model levels this output is written on
        NK=diagdct['Levs']

        ## READ THE OUTPUT
        # Open each separate output file/convert to DataSet, compile into list
        if NK==50:
            varDS=[ecco.llc_tiles_to_xda(ecco.read_llc_to_tiles(\
                         outdir,outpfx+'.'+str(ITER).zfill(10)+'.data',\
                                          nk=NK,llc=90,less_output=True)\
                                         ,var_type=GPT,dim4='depth')\
                              .to_dataset(name=varname) for ITER in ITERS]
        elif NK==1:
            varDS=[ecco.llc_tiles_to_xda(ecco.read_llc_to_tiles(\
                          outdir,outpfx+'.'+str(ITER).zfill(10)+'.data',\
                                           nk=NK,llc=90,less_output=True)\
                                         ,var_type=GPT)\
                               .to_dataset(name=varname) for ITER in ITERS]
        # Concatenate all DataSets into one along time dimension:
        varDS=xr.concat(varDS,dim='time')

        ## Add attributes and time array to xarray DataSet
        varDS[varname]=varDS[varname].assign_attrs(\
             {'units':diagdct['Units'],'description':diagdct['Description']})
        varDS.attrs={'model_deltaT':deltaT,'model_nIter0':nIter0}

        print('set time')
        if modelcal=='noLeapYear':
            varDS['time']=ITERS*deltaT
            varDS['time']=varDS['time'].assign_attrs({\
                       'units':'seconds since '+str(DATE)+','+str(TIME),\
                       'model_calendar':modelcal})
        elif modelcal=='gregorian':
            varDS['time']=times
            varDS['time']=varDS['time'].assign_attrs({'model_calendar':modelcal})

        varDS['iters']=ITERS

        print('save to netcdf')
        try:
            varDS.to_netcdf(outfname+'.nc')
            # Fails with TypeError: Invalid value for attr: ...
        except TypeError as e:
            valid_types = (str, np.ndarray, np.number, list, tuple)
            print(e.__class__.__name__, e)
            for variable in varDS.variables.values():
                for k, v in variable.attrs.items():
                    if not isinstance(v, valid_types) or isinstance(v, bool):
                        variable.attrs[k] = str(v)

            varDS.to_netcdf(outfname+'.nc')
################################################################################
