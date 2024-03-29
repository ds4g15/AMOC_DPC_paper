import sys
import time
import numpy as np
import xarray as xr
import ecco_v4_py as ecco

################################################################################
'''
Calculate Atlantic meridional overturning streamfunction as in  the function
ecco.calculate_meridional_stf but with an additional calculation of the Ekman
transport component. Requires output files for 
UVELMASS,VVELMASS,RhoAnoma,oceTAUX,oceTAUY
in netCDF format

Saves a netCDF file containing an xarray Dataset with the following variables
            moc
                meridional overturning strength as maximum of streamfunction
                in depth space, with dimensions 'time' (if in dataset), and 'lat'
            psi_moc
                meridional overturning streamfunction across the section in Sv
                with dimensions 'time' (if in given dataset), 'lat', and 'k'
            trsp_z
                freshwater transport across section at each depth level in Sv
                with dimensions 'time' (if in given dataset), 'lat', and 'k'
            ekman
                Ekman transport across section in Sv with dimensions 'time' (if
                in given dataset), 'lat', and 'k'
'''
################################################################################

T0=time.time()
outdir=sys.argv[-1]

########################### Meridional stream function calculation
u=xr.open_dataset(outdir+'/UVELMASS.nc')
v=xr.open_dataset(outdir+'/VVELMASS.nc')
GDS=xr.open_dataset('~/ECCO-GRID.nc')

uv=xr.merge([u,v,GDS.drF.drop('drF'),GDS.dyG.drop('dyG'),GDS.dxG.drop('dxG')])
print('calculating meridional stream function '+str(time.time()-T0))
msf=ecco.calc_meridional_stf(uv,lat_vals=np.arange(-30,75),basin_name=['atl','mexico'],coords=GDS.assign_coords({'time':uv.time}))
print('done '+str(time.time()-T0))

########################### Ekman transport calculations
# Load wind stress and rearrange into zonal and meridional
XDS=xr.open_dataset(outdir+'/oceTAUX.nc')
YDS=xr.open_dataset(outdir+'/oceTAUY.nc')
taux,_=ecco.vector_calc.UEVNfromUXVY(XDS.oceTAUX,YDS.oceTAUY,GDS)

# Load density anomaly
rhoConst=1029 # set in 'data' namelist
rho     = xr.open_dataset(outdir+'/RHOAnoma.nc').RHOAnoma.isel(k=0)+rhoConst

# Atlantic mask
atlmsk=ecco.get_basin_mask(['atl','mexico'],mask=GDS.hFacC,less_output=True).isel(k=0)

# coriolis frequency
f=(2*7.2921e-5)*np.sin(np.deg2rad(GDS.YC))

# Integrand of ∫-τ/ρf dx
ek_integrand =-( taux *atlmsk*GDS.dxG.values[:,:,:]/(f*rho) ).values

# Resample onto the same latlon grid as that of ecco.calc_meridional_stf and integrate zonally
# (transpose and reshape are used as resample_to_latlon ignores the last dimension (time) but only works on two dimensions)
_,lat1,_,lon1,ek_interp=ecco.resample_to_latlon(GDS.XC.values.ravel(),GDS.YC.values.ravel(),\
                                                ek_integrand.transpose(1,2,3,0).reshape(13*90*90,-1),\
                                                -30,75,1,-100,30,1,mapping_method='nearest_neighbor')
ek_interp= ek_interp.sum(axis=1)*1e-6

# Add the ekman transport calculation to the dataset
ekDA=xr.DataArray(1e-6*ek_interp.T,dims=['time','lat'],coords={'time':msf.time,'lat':msf.lat})
msf=msf.assign(ekman=ekDA)

# Write to netcdf
msf.to_netcdf(outdir+'/atlantic_moc_diagnostics.nc')
