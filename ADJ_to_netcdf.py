surface_only=1
outdir='./'
rundir='./'

################################################################################
import os
import sys
import glob
import numpy as np
import xarray as xr
import xmitgcm as xm
import datetime as dt
import ecco_v4_py as ecco
################################################################################

# Either grab a list of variables from script call or try to load all values
print(sys.argv)
print(type(sys.argv))
if len(sys.argv)>1:
    varlist=sys.argv[1:]
else:
    varlist=['ADJaqh','ADJclimsst','ADJggl90tke','ADJkapredi','ADJqnet',\
             'ADJsalt','ADJtaux','ADJustress','ADJvvel','ADJatemp','ADJempr',\
             'ADJhflux','ADJlwdown','ADJqsw','ADJsflux','ADJtauy','ADJuvel',\
             'ADJwvel','ADJclimsss','ADJetan','ADJkapgm','ADJprecip','ADJrunoff',\
             'ADJswdown','ADJtheta','ADJvstress','ADJuwind','ADJvwind']

# Lookup dictionary of which gridpoint each variable is at
gptdct={'ADJaqh'     :'c','ADJclimsst' :'c','ADJggl90tke':'c',\
        'ADJkapredi' :'c','ADJqnet'    :'c','ADJsalt'    :'c',\
        'ADJtaux'    :'w','ADJustress' :'w','ADJvvel'    :'s',\
        'ADJatemp'   :'c','ADJempr'    :'c','ADJhflux'   :'c',\
        'ADJlwdown'  :'c','ADJqsw'     :'c','ADJsflux'   :'c',\
        'ADJtauy'    :'s','ADJuvel'    :'w','ADJwvel'    :'c',\
        'ADJclimsss' :'c','ADJetan'    :'c','ADJkapgm'   :'c',\
        'ADJprecip'  :'c','ADJrunoff'  :'c','ADJswdown'  :'c',\
        'ADJtheta'   :'c','ADJvstress' :'s','ADJuwind'   :'w',\
        'ADJvwind'   :'s'}
# Lookup dictionary of whether variables are 2D or 3D
dimdct={'ADJaqh'     :2  ,'ADJclimsst' :2  ,'ADJggl90tke':3,\
        'ADJkapredi' :3  ,'ADJqnet'    :2  ,'ADJsalt'    :3,\
        'ADJtaux'    :2  ,'ADJustress' :2  ,'ADJvvel'    :3,\
        'ADJatemp'   :2  ,'ADJempr'    :2  ,'ADJhflux'   :2,\
        'ADJlwdown'  :2  ,'ADJqsw'     :2  ,'ADJsflux'   :2,\
        'ADJtauy'    :2  ,'ADJuvel'    :3  ,'ADJwvel'    :3,\
        'ADJclimsss' :2  ,'ADJetan'    :2  ,'ADJkapgm'   :3,\
        'ADJprecip'  :2  ,'ADJrunoff'  :2  ,'ADJswdown'  :2,\
        'ADJtheta'   :3  ,'ADJvstress' :2  ,'ADJuwind'   :2,\
        'ADJvwind'   :2}

if surface_only: [dimdct.update({X:2}) for X in dimdct.keys()]

print(varlist)

print('getting calendar namelist info')
with open(rundir+'data.cal') as F:
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
with open(rundir+'data') as F:
    dat=F.readlines()
    for L in dat:
        if L[0]=='#': continue
        if 'deltaTClock' in L:
            deltaT=float(L.split('=')[-1].split(',')[0])
        elif 'nIter0' in L:
            nIter0=int(L.split('=')[-1].split(',')[0]) 

time0=time0+dt.timedelta(seconds=(deltaT*nIter0))


################################################################################

def mds_to_xda(VAR,FNAME,GPT,DIM,*args,**kwargs):
    if DIM==3:
        DA=ecco.llc_tiles_to_xda(ecco.read_llc_to_tiles(outdir,FNAME,*args,nk=50,less_output=True,**kwargs),var_type=GPT,dim4='depth')\
               .to_dataset(name=VAR)
    elif DIM==2:
        DA=ecco.llc_tiles_to_xda(ecco.read_llc_to_tiles(outdir,FNAME,*args,nk=1,less_output=True,**kwargs),var_type=GPT)\
               .to_dataset(name=VAR)
    return DA.rename({'tile':'face'})

################################################################################
#Get attributes from namelist:



for varname in varlist:
    print('getting output info for '+varname)
    F=sorted(glob.glob(outdir+varname+'*.data')) # Get list of outputs
    if len(F)==0: continue
    ITERS=np.array([int(X.split('.')[-2]) for X in F])
    times=[time0+dt.timedelta(seconds=(I-ITERS[0])*deltaT) for I in ITERS]

    ################################################################################

    # Get the actual dataset:
    print('loading output')
    if dimdct[varname]==2:
        varDS=[ecco.llc_tiles_to_xda(ecco.read_llc_to_tiles(\
                          outdir,varname+'.'+str(ITER).zfill(10)+'.data',\
                                           nk=1,llc=90,less_output=True)\
                                         ,var_type=gptdct[varname])\
                               .to_dataset(name=varname) for ITER in ITERS]
    elif dimdct[varname]==3:
        varDS=[ecco.llc_tiles_to_xda(ecco.read_llc_to_tiles(\
                          outdir,varname+'.'+str(ITER).zfill(10)+'.data',\
                                           nk=50,llc=90,less_output=True)\
                                     ,var_type=gptdct[varname],dim4='depth')\
                              .to_dataset(name=varname) for ITER in ITERS]
    # Concatenate all DataSets into one along time dimension:
    varDS=xr.concat(varDS,dim='time')

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
        varDS.to_netcdf(varname+'.nc')
        # Fails with TypeError: Invalid value for attr: ...
    except TypeError as e:
        valid_types = (str, np.ndarray, np.number, list, tuple)
        print(e.__class__.__name__, e)
        for variable in varDS.variables.values():
            for k, v in variable.attrs.items():
                if not isinstance(v, valid_types) or isinstance(v, bool):
                    variable.attrs[k] = str(v)

        varDS.to_netcdf(varname+'.nc')  # Works as expected
################################################################################

